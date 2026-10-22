import asyncio
from opentelemetry.trace import SpanKind
from src.infrastructure.neo4j_client import create_db_connect
from src.observability.telemetry import configure_telemetry
async def main():
    provider,tracer=configure_telemetry()
    db = create_db_connect()
    try:
        with tracer.start_as_current_span("graph_rag.neo4j_smoke_test") as root_span:
            root_span.set_attribute("app.component", "neo4j")
            root_span.set_attribute("app.operation", "smoke_test")

            with tracer.start_as_current_span("test.verify_neo4j_connection") as connection_span:
                connection_span.set_attribute("db.system", "neo4j")
                await db.verify_connection()

            with tracer.start_as_current_span("neo4j.query",kind=SpanKind.CLIENT) as query_span:
                query_span.set_attribute("db.system", "neo4j")
                query_span.set_attribute("db.namespace", db.database)
                query_span.set_attribute("app.operation","connectivity_check")
                async with db.driver.session(database=db.database) as session:
                    result = await session.run("RETURN 1 AS value")
                    record = await result.single()
                    if record is None:
                        raise RuntimeError("Query returned no records")
                    assert record["value"] == 1
                    print("Neo4j query successful")
        provider.force_flush(timeout_millis=10000)
    finally:
        try:
            provider.shutdown()
        finally:
            await db.close()


if __name__ == "__main__":
    asyncio.run(main())