from src.observability.telemetry import configure_telemetry

provider, tracer = configure_telemetry()

try:
    with tracer.start_as_current_span("telemetry.smoke_test") as span:
        span.set_attribute("operation", "startup_check")
        span.set_attribute("status", "success")

    provider.force_flush(timeout_millis=10000)
    print("Telemetry smoke test completed.")
finally:
    provider.shutdown()