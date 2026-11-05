import asyncio
import hashlib
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import LiteralString

import httpx
from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.concurrency import run_in_threadpool
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry import trace
from neo4j.exceptions import AuthError, DriverError, ServiceUnavailable, SessionExpired
from pydantic import BaseModel

from src.infrastructure.neo4j_client import DbConnect, create_db_connect, run_query
from src.ingestion.parsers import FastTextParser, OcrParser, StructuredParser
from src.ingestion.pipeline import (
    CPCRIngestionResponse,
    classify_page_continuity,
    ingest_pdf,
    revise_chunks_after_cpcr,
)
from src.ingestion.preflight import InvalidPdfError
from src.ingestion.scaledown_client import ScaleDownClient
from src.observability.telemetry import configure_telemetry
from src.security.ingestion_screening.contracts import DocumentContext
from src.security.ingestion_screening.enums import AdmissionDecision, TrustTier
from src.security.ingestion_screening.file_gate import (
    ClamAVScanner,
    FileGateError,
    screen_pdf_file,
)
from src.security.ingestion_screening.policy import ScreeningPolicy
from src.security.ingestion_screening.screening_service import ScreeningService

_TRACER = trace.get_tracer("graph_rag.api")


async def _release_idle_docling_models(
    parsers: tuple[StructuredParser, ...],
    idle_timeout_seconds: int,
) -> None:
    interval = max(1, min(60, idle_timeout_seconds // 4 or 1))
    while True:
        await asyncio.sleep(interval)
        for parser in parsers:
            await run_in_threadpool(parser.close_if_idle, idle_timeout_seconds)


class HealthResponse(BaseModel):
    health: str
class ReadinessResponse(BaseModel):
    status: str
    test_value: str
class ErrorResponse(BaseModel):
    detail: str
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application starting up...")
    load_dotenv()
    idle_timeout_seconds = int(
        os.environ.get("DOCLING_IDLE_TIMEOUT_SECONDS", "1800")
    )
    if idle_timeout_seconds < 0:
        raise ValueError("DOCLING_IDLE_TIMEOUT_SECONDS must be non-negative")
    provider, _ = configure_telemetry()
    scaledown_http_client = httpx.AsyncClient(timeout=30.0)
    app.state.scaledown_http_client = scaledown_http_client
    app.state.scaledown_client = ScaleDownClient(
        http_client=scaledown_http_client
    )
    idle_model_task = None
    docling_parsers: tuple[StructuredParser, ...] = ()
    db_connection = None
    instrumented = False
    try:
        fast_text_parser = FastTextParser()
        structured_parser = StructuredParser()
        layout_ocr_parser = StructuredParser(do_ocr=True)
        ocr_parser = OcrParser()
        screening_policy = ScreeningPolicy()
        malware_scanner = ClamAVScanner()
        await run_in_threadpool(malware_scanner.initialize)
        app.state.screening_policy = screening_policy
        app.state.screening_service = ScreeningService(screening_policy)
        app.state.malware_scanner = malware_scanner
        docling_parsers = (structured_parser, layout_ocr_parser)

        # Lifespan startup does not yield until both PDF pipelines and their
        # model weights are initialized and ready for requests.
        with _TRACER.start_as_current_span("ingestion.docling.initialize") as span:
            span.set_attribute("ingestion.parser.pipeline", "PDF")
            await run_in_threadpool(structured_parser.initialize)
        with _TRACER.start_as_current_span(
            "ingestion.docling.initialize_layout_ocr"
        ) as span:
            span.set_attribute("ingestion.parser.pipeline", "PDF_LAYOUT_OCR")
            await run_in_threadpool(layout_ocr_parser.initialize)
        app.state.ingestion_parsers = {
            "fast_text": fast_text_parser,
            "structured": structured_parser,
            "layout_ocr": layout_ocr_parser,
            "ocr": ocr_parser,
        }
        idle_model_task = asyncio.create_task(
            _release_idle_docling_models(docling_parsers, idle_timeout_seconds),
            name="docling-idle-model-release",
        )

        FastAPIInstrumentor.instrument_app(app)
        instrumented = True

        db_connection = create_db_connect()
        app.state.db = db_connection

        yield
    finally:
        try:
            if idle_model_task is not None:
                idle_model_task.cancel()
                try:
                    await idle_model_task
                except asyncio.CancelledError:
                    pass
            for parser in docling_parsers:
                await run_in_threadpool(parser.close)
            if instrumented:
                FastAPIInstrumentor.uninstrument_app(app)
        finally:
            try:
                if db_connection is not None:
                    await db_connection.close()
            finally:
                try:
                    await scaledown_http_client.aclose()
                finally:
                    try:
                        provider.force_flush(timeout_millis=10000)
                    finally:
                        provider.shutdown()
    print("Application shutting down...")
app=FastAPI(lifespan=lifespan)
MAX_INGESTION_UPLOAD_BYTES = 50 * 1024 * 1024
def get_db(request:Request)->DbConnect:
    return request.app.state.db

@app.get("/health",summary="Check API liveness",description=("Reports whether the API process is running. ""Does not check external dependencies such as Neo4j."),tags=["Health"],response_description="The API process is alive.",response_model=HealthResponse)
def check_health():
    return {"health":"System is healthy"}


@app.post(
    "/ingestion",
    summary="Ingest a PDF",
    description=(
        "Runs PDF preflight, page routing, parsing, chunking and page "
        "continuity review, then applies positive continuity decisions to "
        "the returned chunks. Returns revised chunks and crops for image "
        "blocks with no recognized text; CPCR details and canonical output "
        "are not included."
    ),
    tags=["Ingestion"],
    response_model=CPCRIngestionResponse,
)
async def ingest_pdf_endpoint(
    request: Request, file: UploadFile = File(...)
) -> CPCRIngestionResponse:
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        await file.close()
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Upload must be a PDF file",
        )

    with TemporaryDirectory(prefix="graph-rag-ingestion-") as temp_dir:
        source = Path(temp_dir) / "upload.pdf"
        total_bytes = 0
        try:
            with source.open("wb") as destination:
                while data := await file.read(1024 * 1024):
                    total_bytes += len(data)
                    if total_bytes > MAX_INGESTION_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail="PDF upload exceeds the 50 MiB limit",
                        )
                    destination.write(data)
            if total_bytes == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Uploaded PDF is empty",
                )
            screening_policy = request.app.state.screening_policy
            try:
                with _TRACER.start_as_current_span("ingestion.file_gate") as span:
                    file_screening = await run_in_threadpool(
                        screen_pdf_file,
                        source,
                        policy=screening_policy,
                        scanner=request.app.state.malware_scanner,
                        filename=file.filename,
                    )
                    span.set_attribute(
                        "ingestion.screening.file_decision",
                        file_screening.decision.value,
                    )
                    span.set_attribute(
                        "ingestion.screening.file_bytes",
                        file_screening.file_size_bytes,
                    )
            except FileGateError as exc:
                error_status = (
                    status.HTTP_503_SERVICE_UNAVAILABLE
                    if "SCANNER" in exc.record.justification_codes[0]
                    else status.HTTP_422_UNPROCESSABLE_ENTITY
                )
                raise HTTPException(
                    status_code=error_status,
                    detail={
                        "code": exc.record.justification_codes[0],
                        "file_screening": exc.record.model_dump(mode="json"),
                    },
                ) from exc
            parsers = request.app.state.ingestion_parsers
            ingestion = await run_in_threadpool(
                ingest_pdf,
                source,
                file_screening=file_screening,
                fast_text_parser=parsers["fast_text"],
                structured_parser=parsers["structured"],
                layout_ocr_parser=parsers["layout_ocr"],
                ocr_parser=parsers["ocr"],
            )
            canonical_json = json.dumps(
                ingestion.canonical.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            span_context = trace.get_current_span().get_span_context()
            run_id = (
                f"{span_context.trace_id:032x}" if span_context.is_valid else None
            )
            document_context = DocumentContext(
                source_id=ingestion.document_id,
                tenant_id=None,
                uploader_id=None,
                ingestion_run_id=run_id,
                source_sha256=ingestion.features.content_sha256,
                canonical_document_digest=hashlib.sha256(
                    canonical_json.encode("utf-8")
                ).hexdigest(),
                chunking_policy_version=ingestion.chunking.document.policy_version,
                source_page_count=ingestion.features.page_count,
                trust_tier=TrustTier.T3_UNAUTHENTICATED_EXTERNAL,
            )
            try:
                with _TRACER.start_as_current_span("ingestion.screening") as span:
                    screened = await request.app.state.screening_service.screen_batch(
                        ingestion.chunking.document.chunks,
                        document_context,
                        document_id=ingestion.document_id,
                        file_screening=file_screening,
                    )
                    span.set_attribute(
                        "ingestion.screening.chunks",
                        len(screened.report.chunk_results),
                    )
                    span.set_attribute(
                        "ingestion.screening.quarantined",
                        sum(
                            record.decision == AdmissionDecision.QUARANTINED
                            for record in screened.admission_records
                        ),
                    )
                    span.set_attribute(
                        "ingestion.screening.rejected",
                        sum(
                            record.decision == AdmissionDecision.REJECTED
                            for record in screened.admission_records
                        ),
                    )
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail={
                        "code": "SCREENING_FAILED",
                        "decision": AdmissionDecision.QUARANTINED.value,
                    },
                ) from exc
            cpcr = await classify_page_continuity(
                ingestion,
                request.app.state.scaledown_client,
                screened,
            )
            return CPCRIngestionResponse(
                revised_chunks=revise_chunks_after_cpcr(ingestion, cpcr, screened),
                unrecognized_images=ingestion.unrecognized_images,
            )
        except InvalidPdfError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        finally:
            await file.close()

@app.get("/health/ready",summary="Check API readiness",description=("Checks whether the application can communicate with Neo4j ""and execute a lightweight readiness query."),tags=["Health"],
    responses={
        503: {
            "model": ErrorResponse,
            "description": (
                "The Neo4j dependency is unavailable or cannot be used."
            )
        },
        500: {
            "model": ErrorResponse,
            "description": "An unexpected internal application error occurred."
        },
    },response_description="Neo4j is available and the readiness query succeeded.",response_model=ReadinessResponse)
async def check_neo4j(db:DbConnect=Depends(get_db)):
    try:
        await db.verify_connection()
    except ConnectionError as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Instance connection is not established."
                )from exc
    query:LiteralString="RETURN 'System is Ready' AS connection_test"
    try:
        records=await run_query(query=query,db_connection=db)
        return {"status": "ready", "test_value": records[0][0]}
    except ServiceUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j server is unreachable or unavailable."
        )from exc
    except SessionExpired as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The database session has expired, often due to a connection problem."
            )from exc
    except ConnectionError as exc:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Instance connection is not established."
                    )from exc
    except AuthError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The required database dependency cannot be used because authentication failed. Also log this as a configuration/operational error"
            )from exc
    except DriverError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="driver or connectivity failure"
            )from exc
    except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unanticipated programming errors or failures not classified as dependency outages."
            )from exc
