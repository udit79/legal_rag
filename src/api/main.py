from fastapi import FastAPI
from contextlib import asynccontextmanager
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from src.observability.telemetry import configure_telemetry

@asynccontextmanager
async def lifespan(app:FastAPI):
    print("Application starting up...")
    provider,trace=configure_telemetry()
    FastAPIInstrumentor.instrument_app(app=app)
    try:
        yield
    finally:
        try:
            FastAPIInstrumentor.uninstrument_app(app=app)
        finally:
            try:
                provider.force_flush(timeout_millis=10000)
            finally:
                provider.shutdown()
    print("Application shutting down...")
app=FastAPI(lifespan=lifespan)

@app.get("/health")
def check_health():
    return {"health":"System is healthy"}
