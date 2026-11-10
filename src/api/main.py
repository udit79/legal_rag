import hashlib
import json
import os
import uuid
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
from src.api.upload_limits import UploadBodyLimitMiddleware
from src.ingestion.pipeline import (
    ChunkDisposition,
    CPCRIngestionResponse,
    INGESTION_ROUTING_POLICY_VERSION,
    UnrecognizedImageCrop,
    classify_page_continuity,
    ingest_pdf,
    revise_chunks_after_cpcr,
)
from src.ingestion.parser_worker.exceptions import (
    ParserWorkerError,
    ParserWorkerTimeout,
)
from src.ingestion.parser_worker.limits import MAX_PDF_PAGES, MAX_PDF_SIZE_BYTES
from src.ingestion.parser_worker.runner import ParserWorkerRunner
from src.ingestion.provenance.supabase_store import SupabaseProvenanceStore
from src.ingestion.screening.image_screening import (
    attach_image_screening_results,
    screen_image_crops,
)
from src.ingestion.preflight import InvalidPdfError
from src.ingestion.router import route_document
from src.ingestion.scaledown_client import ScaleDownClient
from src.ingestion.scaledown_client import ScaleDownError, ScaleDownTimeoutError
from src.ingestion.scaledown_extraction import recover_escalated_pages
from src.observability.telemetry import configure_telemetry
from src.security.ingestion_screening.contracts import DocumentContext
from src.security.ingestion_screening.enums import AdmissionDecision, TrustTier
from src.security.ingestion_screening.file_gate import (
    ClamAVScanner,
    FileGateError,
    apply_pdf_structure_inspection,
    screen_pdf_file,
)
from src.security.ingestion_screening.policy import ScreeningPolicy
from src.security.ingestion_screening.screening_service import ScreeningService

_TRACER = trace.get_tracer("graph_rag.api")


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
    provider, _ = configure_telemetry()
    scaledown_http_client = httpx.AsyncClient(timeout=30.0)
    app.state.scaledown_http_client = scaledown_http_client
    app.state.scaledown_client = ScaleDownClient(
        http_client=scaledown_http_client
    )
    parser_worker = ParserWorkerRunner()
    app.state.parser_worker = parser_worker
    db_connection = None
    instrumented = False
    try:
        screening_policy = ScreeningPolicy()
        malware_scanner = ClamAVScanner()
        await run_in_threadpool(malware_scanner.initialize)
        app.state.screening_policy = screening_policy
        app.state.screening_service = ScreeningService(screening_policy)
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if bool(supabase_url) != bool(supabase_key):
            raise RuntimeError(
                "Set both SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY, or neither"
            )
        app.state.provenance_store = (
            SupabaseProvenanceStore(
                supabase_url, supabase_key, scaledown_http_client
            )
            if supabase_url and supabase_key
            else None
        )
        app.state.malware_scanner = malware_scanner
        await parser_worker.start()
        FastAPIInstrumentor.instrument_app(app)
        instrumented = True

        db_connection = create_db_connect()
        app.state.db = db_connection

        yield
    finally:
        try:
            if instrumented:
                FastAPIInstrumentor.uninstrument_app(app)
        finally:
            try:
                await parser_worker.close()
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
MAX_INGESTION_UPLOAD_BYTES = MAX_PDF_SIZE_BYTES
MULTIPART_REQUEST_OVERHEAD_BYTES = 1024 * 1024
app.add_middleware(
    UploadBodyLimitMiddleware,
    max_request_bytes=(
        MAX_INGESTION_UPLOAD_BYTES + MULTIPART_REQUEST_OVERHEAD_BYTES
    ),
)
def get_db(request:Request)->DbConnect:
    return request.app.state.db

@app.get("/health",summary="Check API liveness",description=("Reports whether the API process is running. ""Does not check external dependencies such as Neo4j."),tags=["Health"],response_description="The API process is alive.",response_model=HealthResponse)
def check_health():
    return {"health":"System is healthy"}


@app.post(
    "/ingestion",
    summary="Ingest a PDF",
    description=(
        "Runs PDF preflight, page routing, parsing, chunking, screening and "
        "page continuity review, then adds bounded source-backed context to "
        "positive paragraph continuations. Returns revised chunks, chunk "
        "disposition codes, and image crops with OCR/admission status. CPCR "
        "details and canonical output are not included."
    ),
    tags=["Ingestion"],
    response_model=CPCRIngestionResponse,
)
async def ingest_pdf_endpoint(
    request: Request, file: UploadFile = File(...)
) -> CPCRIngestionResponse:
    ingestion_run_id = uuid.uuid4().hex
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
                    raw_file_screening = await run_in_threadpool(
                        screen_pdf_file,
                        source,
                        policy=screening_policy,
                        scanner=request.app.state.malware_scanner,
                        filename=file.filename,
                    )
                    structure_inspection = (
                        await request.app.state.parser_worker.inspect_pdf(
                            source,
                            max_pdf_objects=screening_policy.max_pdf_objects,
                        )
                    )
                    file_screening = apply_pdf_structure_inspection(
                        raw_file_screening,
                        structure_inspection,
                        policy=screening_policy,
                        scanner=request.app.state.malware_scanner,
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
                        "request_id": ingestion_run_id,
                    },
                ) from exc
            with _TRACER.start_as_current_span("ingestion.parser_worker") as span:
                parsed = await request.app.state.parser_worker.parse_pdf(
                    source, INGESTION_ROUTING_POLICY_VERSION
                )
                if parsed.features is None or parsed.route_plan is None or parsed.canonical is None:
                    raise ParserWorkerError("Parser worker returned an incomplete document")
                features = parsed.features
                route_plan = parsed.route_plan
                canonical = parsed.canonical
                span.set_attribute("ingestion.parser_worker.page_count", features.page_count)
            expected_route_plan = route_document(
                features, INGESTION_ROUTING_POLICY_VERSION
            )
            if route_plan != expected_route_plan:
                raise ParserWorkerError("Worker returned a route plan that failed validation")
            if features.page_count > MAX_PDF_PAGES:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="PDF exceeds the parser worker's 500-page limit",
                )
            if file_screening.source_sha256 != features.content_sha256:
                raise ParserWorkerError("Worker preflight digest differs from G0")
            if file_screening.page_count != features.page_count:
                raise ParserWorkerError("Worker preflight page count differs from G0")
            if file_screening.file_size_bytes != features.file_size_bytes:
                raise ParserWorkerError("Worker preflight file size differs from G0")
            with _TRACER.start_as_current_span(
                "ingestion.scaledown_extract_fallback"
            ) as span:
                canonical = await recover_escalated_pages(
                    source,
                    canonical,
                    route_plan,
                    page_exporter=request.app.state.parser_worker,
                    text_extractor=request.app.state.scaledown_client,
                )
                recovered_pages = sum(
                    page.parser == "SCALEDOWN_EXTRACT"
                    for page in canonical.pages
                )
                span.set_attribute(
                    "ingestion.scaledown_extract.recovered_pages",
                    recovered_pages,
                )
            worker_crops = await request.app.state.parser_worker.crop_unrecognized_images(
                source, canonical
            )
            prepared_image_crops = tuple(
                UnrecognizedImageCrop.model_validate(crop.model_dump())
                for crop in worker_crops
            )
            ingestion = await run_in_threadpool(
                ingest_pdf,
                source,
                file_screening=file_screening,
                preflight_features=features,
                prepared_route_plan=route_plan,
                parsed_canonical_document=canonical,
                prepared_image_crops=prepared_image_crops,
            )
            if ingestion.chunking.report.outcome in {"PARTIAL", "FAILED"}:
                failed_pages = [
                    page.page_number
                    for page in ingestion.canonical.pages
                    if page.status == "FAILED"
                ]
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "code": "INGESTION_INCOMPLETE",
                        "failed_pages": failed_pages,
                    },
                )
            if (
                ingestion.chunking.report.outcome == "NO_CONTENT"
                and not ingestion.unrecognized_images
            ):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={"code": "NO_ELIGIBLE_CONTENT"},
                )
            canonical_json = json.dumps(
                ingestion.canonical.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            document_context = DocumentContext(
                source_id=ingestion.document_id,
                tenant_id=None,
                uploader_id=None,
                ingestion_run_id=ingestion_run_id,
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
            with _TRACER.start_as_current_span("ingestion.image_ocr_screening") as span:
                image_screening_results = await screen_image_crops(
                    ingestion.unrecognized_images,
                    ocr=request.app.state.parser_worker,
                    screening_service=request.app.state.screening_service,
                )
                span.set_attribute(
                    "ingestion.image_ocr.count", len(image_screening_results)
                )
                span.set_attribute(
                    "ingestion.image_ocr.failed",
                    sum(item.ocr_status == "FAILED" for item in image_screening_results),
                )
                span.set_attribute(
                    "ingestion.image_ocr.flagged",
                    sum(bool(item.flags) for item in image_screening_results),
                )
            try:
                response_images = attach_image_screening_results(
                    ingestion.unrecognized_images, image_screening_results
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail={"code": "IMAGE_SCREENING_ALIGNMENT_FAILED"},
                ) from exc
            provenance_store = request.app.state.provenance_store
            if provenance_store is not None:
                try:
                    await provenance_store.persist_screened_document(
                        screened,
                        document_context,
                        image_results=image_screening_results,
                    )
                except Exception as exc:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail={"code": "PROVENANCE_PERSISTENCE_FAILED"},
                    ) from exc
            document_blocked = screened.report.quarantined_document or any(
                result.decision == AdmissionDecision.REJECTED
                for result in screened.report.chunk_results
            )
            if document_blocked:
                blocked_chunks = [
                    {
                        "chunk_index": result.chunk_index,
                        "decision": result.decision.value,
                        "reason_codes": result.justification_codes,
                    }
                    for result in screened.report.chunk_results
                    if result.decision
                    in {AdmissionDecision.QUARANTINED, AdmissionDecision.REJECTED}
                ]
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "code": (
                            "DOCUMENT_QUARANTINED"
                            if screened.report.quarantined_document
                            else "CONTENT_REJECTED"
                        ),
                        "blocked_chunks": blocked_chunks,
                    },
                )
            cpcr = await classify_page_continuity(
                ingestion,
                request.app.state.scaledown_client,
                screened,
            )
            if cpcr.outcome != "COMPLETED":
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail={
                        "code": "CPCR_CLASSIFICATION_INCOMPLETE",
                        "failed_candidates": [
                            {
                                "candidate_id": result.candidate.candidate_id,
                                "error_type": result.classification_error,
                            }
                            for result in cpcr.candidates
                            if result.classification_status == "FAILED"
                        ],
                    },
                )
            revised = revise_chunks_after_cpcr(ingestion, cpcr, screened)
            if provenance_store is not None:
                try:
                    await provenance_store.persist_cpcr_output(
                        revised,
                        cpcr,
                        ingestion_run_id=ingestion_run_id,
                        source_sha256=document_context.source_sha256,
                    )
                except Exception as exc:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail={"code": "CPCR_PROVENANCE_PERSISTENCE_FAILED"},
                    ) from exc
            return CPCRIngestionResponse(
                revised_chunks=revised.chunks,
                unrecognized_images=response_images,
                chunk_dispositions=tuple(
                    ChunkDisposition(
                        original_chunk_index=result.chunk_index,
                        decision=result.decision,
                        reason_codes=result.justification_codes,
                    )
                    for result in screened.report.chunk_results
                ),
            )
        except InvalidPdfError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        except ParserWorkerTimeout as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="PDF parsing exceeded the configured 60-second limit",
            ) from exc
        except ParserWorkerError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Isolated PDF parser is unavailable or failed",
            ) from exc
        except ScaleDownTimeoutError as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail={"code": "SCALEDOWN_EXTRACT_TIMEOUT"},
            ) from exc
        except ScaleDownError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"code": "SCALEDOWN_EXTRACT_FAILED"},
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
