from fastapi import FastAPI,Depends,Request,HTTPException,status
from typing import LiteralString
from contextlib import asynccontextmanager
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from src.observability.telemetry import configure_telemetry
from src.infrastructure.neo4j_client import create_db_connect,DbConnect,run_query
from typing import Annotated
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
    provider, _ = configure_telemetry()
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
                    provider.force_flush(timeout_millis=10000)
                finally:
                    provider.shutdown()
    print("Application shutting down...")
app=FastAPI(lifespan=lifespan)
def get_db(request:Request)->DbConnect:
    return request.app.state.db

@app.get("/health",summary="Check API liveness",description=("Reports whether the API process is running. ""Does not check external dependencies such as Neo4j."),tags=["Health"],response_description="The API process is alive.",response_model=HealthResponse)
def check_health():
    return {"health":"System is healthy"}
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
                    detail=f"Instance connection is not established."
                )from exc
    query:LiteralString="RETURN 'System is Ready' AS connection_test"
    try:
        records=await run_query(query=query,db_connection=db)
        return {"status": "ready", "test_value": records[0][0]}
    except ServiceUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Neo4j server is unreachable or unavailable."
        )from exc
    except SessionExpired as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"The database session has expired, often due to a connection problem."
            )from exc
    except ConnectionError as exc:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"Instance connection is not established."
                    )from exc
    except AuthError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"The required database dependency cannot be used because authentication failed. Also log this as a configuration/operational error"
            )from exc
    except DriverError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"driver or connectivity failure"
            )from exc
    except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unanticipated programming errors or failures not classified as dependency outages."
            )from exc
