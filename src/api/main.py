from fastapi import FastAPI,Depends,Request,HTTPException,status,UploadFile,File
from fastapi.concurrency import run_in_threadpool
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import LiteralString
from contextlib import asynccontextmanager
import httpx
from dotenv import load_dotenv
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from src.observability.telemetry import configure_telemetry
from src.infrastructure.neo4j_client import create_db_connect,DbConnect,run_query
from src.ingestion.scaledown_client import ScaleDownClient
from src.ingestion.pipeline import (
    CPCRIngestionResponse,
    classify_page_continuity,
    ingest_pdf,
)
from src.ingestion.preflight import InvalidPdfError
from neo4j.exceptions import ServiceUnavailable,SessionExpired,AuthError,DriverError
from pydantic import BaseModel
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
    db_connection = None
    instrumented = False
    try:
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
        "continuity review. Returns CPCR results and crops for image blocks "
        "with no recognized text; canonical output is not included."
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
            ingestion = await run_in_threadpool(ingest_pdf, source)
            cpcr = await classify_page_continuity(
                ingestion, request.app.state.scaledown_client
            )
            return CPCRIngestionResponse(
                cpcr=cpcr,
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
