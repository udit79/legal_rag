"""Idempotent structural schema bootstrap for the configured Neo4j database."""

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

_TRACER = trace.get_tracer("graph_rag.graph.schema")

SCHEMA_STATEMENTS = (
    "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:DOCUMENT) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:CHUNK) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_identity_unique IF NOT EXISTS FOR (c:CHUNK) REQUIRE (c.source_sha256, c.chunking_policy_version, c.original_chunk_indices_key) IS UNIQUE",
    "CREATE INDEX document_source_sha256 IF NOT EXISTS FOR (d:DOCUMENT) ON (d.source_sha256)",
    "CREATE INDEX chunk_source_order IF NOT EXISTS FOR (c:CHUNK) ON (c.source_sha256, c.chunk_index)",
    "CREATE FULLTEXT INDEX chunk_retrieval_text_fulltext IF NOT EXISTS FOR (c:CHUNK) ON EACH [c.retrieval_text]",
)


async def initialize_graph_schema(db_connection) -> None:
    with _TRACER.start_as_current_span("graph.neo4j.schema_bootstrap") as span:
        span.set_attribute("db.system", "neo4j")
        database = getattr(db_connection, "database", None)
        if isinstance(database, str) and database:
            span.set_attribute("db.namespace", database)
        span.set_attribute("graph.schema.statement_count", len(SCHEMA_STATEMENTS))
        try:
            async with db_connection.get_session() as session:
                for statement in SCHEMA_STATEMENTS:
                    result = await session.run(statement)
                    await result.consume()
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            raise
