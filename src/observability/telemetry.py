from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from dotenv import load_dotenv

lazy import os
load_dotenv()
def configure_telemetry():
    resource = Resource.create(
        {
            "service.name": "graph-rag",
            "service.version": "0.1.0",
            "deployment.environment": "development",
        }
    )
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    if not endpoint:
        raise ValueError("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT is not configured")
    provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    provider.add_span_processor(processor)

    trace.set_tracer_provider(provider) 

    return provider,trace.get_tracer("graph_rag")
