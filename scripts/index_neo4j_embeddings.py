"""Index the current Neo4j corpus with the explicitly configured embedding model."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
import os
import re
from pathlib import Path
from collections.abc import Callable, Mapping
from typing import Any

from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.graph.embeddings import (
    EmbeddingConfig,
    create_embedding_provider,
    create_vector_index,
    embed_texts,
    index_current_chunks,
)
from src.infrastructure.neo4j_client import create_db_connect
from src.observability.telemetry import configure_telemetry

_TRACER = trace.get_tracer("graph_rag.graph.embedding_index_cli")
EXPECTED_DATABASE = "rag"
EXPECTED_PROVIDER = "openai"
EXPECTED_MODEL = "text-embedding-3-small"
EXPECTED_DIMENSIONS = 1024
EXPECTED_CURRENT_CHUNKS = 333
INDEX_POLL_SECONDS = 2.0
INDEX_TIMEOUT_SECONDS = 300.0
SMOKE_TEXT = "Embedding index initialization connectivity check."
REQUIRED_INDEX_PROPERTIES = {
    "embedding",
    "is_current",
    "source_sha256",
    "chunking_policy_version",
    "embedding_space_id",
    "embedding_model",
    "embedding_dimensions",
}

_CURRENT_ELIGIBLE_COUNT_QUERY = """
MATCH (d:DOCUMENT)-[:HAS_CHUNK]->(c:CHUNK)
WHERE c.is_current = true
  AND c.source_sha256 = d.source_sha256
  AND c.chunking_policy_version = d.current_chunking_policy_version
  AND d.current_ingestion_run_id IS NOT NULL
RETURN count(DISTINCT c) AS count
"""

_CURRENT_EMBEDDING_VERIFICATION_QUERY = """
MATCH (d:DOCUMENT)-[:HAS_CHUNK]->(c:CHUNK)
WHERE c.is_current = true
  AND c.source_sha256 = d.source_sha256
  AND c.chunking_policy_version = d.current_chunking_policy_version
  AND d.current_ingestion_run_id IS NOT NULL
RETURN c.id AS chunk_id,
       c.embedding_space_id AS embedding_space_id,
       c.embedding_model AS embedding_model,
       c.embedding_dimensions AS embedding_dimensions,
       size(c.embedding) AS vector_length
ORDER BY c.id
"""

_SHOW_VECTOR_INDEX_QUERY = """
SHOW VECTOR INDEXES YIELD name, state, populationPercent, options,
                          labelsOrTypes, properties, failureMessage
WHERE name = $index_name
RETURN name, state, populationPercent, options, labelsOrTypes,
       properties, failureMessage
"""


def validate_config(config: EmbeddingConfig, database: str) -> None:
    if config.provider != EXPECTED_PROVIDER:
        raise ValueError(f"Embedding provider must be {EXPECTED_PROVIDER!r}")
    if config.model != EXPECTED_MODEL:
        raise ValueError(f"Embedding model must be {EXPECTED_MODEL!r}")
    if config.dimensions != EXPECTED_DIMENSIONS:
        raise ValueError(f"Embedding dimensions must be {EXPECTED_DIMENSIONS}")
    if database != EXPECTED_DATABASE:
        raise ValueError(
            f"Refusing to index database {database!r}; expected {EXPECTED_DATABASE!r}"
        )


def safe_error_message(exc: Exception) -> str:
    """Render useful exception details without exposing configured credentials."""
    message = str(exc)
    for variable in (
        "OPENAI_API_KEY", "VOYAGE_API_KEY", "NEO4J_PASSWORD",
        "SUPABASE_SERVICE_ROLE_KEY",
    ):
        secret = os.environ.get(variable)
        if secret:
            message = message.replace(secret, "[REDACTED]")
    message = re.sub(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{12,}\b", "[REDACTED_API_KEY]", message)
    message = re.sub(
        r"(?i)(api[_ -]?key\s*[:=]\s*)[^\s,;]+", r"\1[REDACTED]", message
    )
    return f"{type(exc).__name__}: {message}"


def validate_vector_index_definition(row: Mapping[str, Any], config: EmbeddingConfig) -> None:
    """Fail closed if the configured name refers to a different index definition."""
    name = config.vector_index_name
    if row.get("name") != name:
        raise ValueError(f"Vector index name mismatch: expected {name!r}")
    labels = row.get("labelsOrTypes")
    properties = row.get("properties")
    options = row.get("options")
    index_config = options.get("indexConfig") if isinstance(options, Mapping) else None
    if not isinstance(labels, (list, tuple)) or set(labels) != {"CHUNK"}:
        raise ValueError(f"Vector index {name!r} has incompatible labels: {labels!r}")
    if (not isinstance(properties, (list, tuple))
            or not REQUIRED_INDEX_PROPERTIES.issubset(set(properties))):
        raise ValueError(
            f"Vector index {name!r} lacks required vector or filter properties; "
            f"found {properties!r}"
        )
    if not isinstance(index_config, Mapping):
        raise ValueError(f"Vector index {name!r} has no inspectable indexConfig")
    dimensions = index_config.get("vector.dimensions")
    similarity = index_config.get("vector.similarity_function")
    if (dimensions != config.dimensions
            or not isinstance(similarity, str)
            or similarity.casefold() != "cosine"):
        raise ValueError(
            f"Vector index {name!r} is incompatible: expected dimensions="
            f"{config.dimensions} and cosine similarity, found dimensions={dimensions!r}, "
            f"similarity={similarity!r}"
        )


async def _get_vector_index(db, index_name: str) -> dict[str, Any] | None:
    async with db.get_session() as session:
        result = await session.run(_SHOW_VECTOR_INDEX_QUERY, index_name=index_name)
        rows = await result.data()
    if len(rows) > 1:
        raise RuntimeError(f"Neo4j returned duplicate vector index identities for {index_name!r}")
    return rows[0] if rows else None


async def _count_current_chunks(db) -> int:
    async with db.get_session() as session:
        result = await session.run(_CURRENT_ELIGIBLE_COUNT_QUERY)
        row = await result.single()
    if row is None or type(row.get("count")) is not int:
        raise RuntimeError("Neo4j did not return a valid current eligible chunk count")
    return row["count"]


def _validate_wait_settings(timeout_seconds: float, poll_interval_seconds: float) -> None:
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0
            or isinstance(poll_interval_seconds, bool)
            or not isinstance(poll_interval_seconds, (int, float))
            or not math.isfinite(poll_interval_seconds) or poll_interval_seconds <= 0):
        raise ValueError("Index wait timeout and poll interval must be finite and positive")


async def wait_for_vector_index(
    db,
    config: EmbeddingConfig,
    *,
    timeout_seconds: float = INDEX_TIMEOUT_SECONDS,
    poll_interval_seconds: float = INDEX_POLL_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Any] = asyncio.sleep,
) -> dict[str, Any]:
    _validate_wait_settings(timeout_seconds, poll_interval_seconds)
    index_name = config.vector_index_name
    deadline = clock() + timeout_seconds
    latest: dict[str, Any] | None = None
    while True:
        latest = await _get_vector_index(db, index_name)
        if latest is None:
            raise RuntimeError(f"Neo4j did not create vector index {index_name!r}")
        validate_vector_index_definition(latest, config)
        state = latest.get("state")
        population = latest.get("populationPercent")
        if state == "FAILED":
            raise RuntimeError(
                f"Vector index {index_name!r} failed: "
                f"{latest.get('failureMessage') or 'Neo4j supplied no failure message'}"
            )
        if state == "ONLINE" and population == 100.0:
            return latest
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out waiting for vector index {index_name!r}: "
                f"state={state!r}, populationPercent={population!r}, "
                f"failureMessage={latest.get('failureMessage')!r}"
            )
        await sleep(min(poll_interval_seconds, remaining))


async def _verify_all_current_embeddings(
    db, config: EmbeddingConfig, expected_chunks: int = EXPECTED_CURRENT_CHUNKS
) -> int:
    async with db.get_session() as session:
        result = await session.run(_CURRENT_EMBEDDING_VERIFICATION_QUERY)
        rows = await result.data()
    if len(rows) != expected_chunks:
        raise RuntimeError(
            f"Expected {expected_chunks} current chunks after indexing, found {len(rows)}"
        )
    invalid = [row for row in rows if (
        row.get("embedding_space_id") != config.space_id
        or row.get("embedding_model") != config.model
        or row.get("embedding_dimensions") != config.dimensions
        or row.get("vector_length") != config.dimensions
    )]
    if invalid:
        example = invalid[0]
        raise RuntimeError(
            f"Current chunk {example.get('chunk_id')!r} has missing or incompatible "
            "embedding metadata or vector length"
        )
    return len(rows)


async def _execute(
    *,
    config_loader: Callable[[], EmbeddingConfig] = EmbeddingConfig.from_env,
    provider_factory: Callable = create_embedding_provider,
    db_factory: Callable = create_db_connect,
    indexer: Callable = index_current_chunks,
    index_creator: Callable = create_vector_index,
    expected_chunks: int = EXPECTED_CURRENT_CHUNKS,
    timeout_seconds: float = INDEX_TIMEOUT_SECONDS,
    poll_interval_seconds: float = INDEX_POLL_SECONDS,
) -> dict[str, Any]:
    load_dotenv()
    _validate_wait_settings(timeout_seconds, poll_interval_seconds)
    with _TRACER.start_as_current_span("graph.neo4j.embedding_index_run") as span:
        span.set_attribute("service.component", "neo4j_embedding_index_cli")
        try:
            config = config_loader()
            validate_config(config, EXPECTED_DATABASE)
            span.set_attribute("graph.embedding.provider", config.provider)
            span.set_attribute("graph.embedding.model", config.model)
            span.set_attribute("graph.embedding.dimensions", config.dimensions)
            span.set_attribute("graph.embedding.space_id", config.space_id)
            span.set_attribute("graph.embedding.vector_index", config.vector_index_name)

            try:
                provider = provider_factory(config)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not initialize the configured {config.provider} embedding provider; "
                    f"check its SDK and API key ({safe_error_message(exc)})"
                ) from exc

            db = db_factory()
            try:
                if getattr(db, "database", None) != EXPECTED_DATABASE:
                    raise ValueError(
                        f"Neo4j client selected {getattr(db, 'database', None)!r}; "
                        f"expected {EXPECTED_DATABASE!r}"
                    )
                try:
                    await db.verify_connection()
                except Exception as exc:
                    raise RuntimeError(
                        "Neo4j connectivity/authentication check failed: "
                        f"{safe_error_message(exc)}"
                    ) from exc

                eligible = await _count_current_chunks(db)
                if eligible != expected_chunks:
                    raise RuntimeError(
                        f"Expected {expected_chunks} current eligible chunks in {EXPECTED_DATABASE!r}, "
                        f"found {eligible}; refusing to index an unexpected corpus"
                    )
                span.set_attribute("graph.chunk.eligible_count", eligible)

                existing_index = await _get_vector_index(db, config.vector_index_name)
                if existing_index is not None:
                    validate_vector_index_definition(existing_index, config)
                    if existing_index.get("state") == "FAILED":
                        raise RuntimeError(
                            f"Existing vector index {config.vector_index_name!r} is FAILED: "
                            f"{existing_index.get('failureMessage') or 'no failure details'}"
                        )

                try:
                    smoke_vectors = await embed_texts(
                        provider, config, [SMOKE_TEXT], input_type="document"
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"{config.provider} embedding API smoke test failed: "
                        f"{safe_error_message(exc)}"
                    ) from exc
                if len(smoke_vectors) != 1:
                    raise RuntimeError("Embedding smoke test returned an unexpected vector count")
                smoke_vector = smoke_vectors[0]
                if (len(smoke_vector) != EXPECTED_DIMENSIONS
                        or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                               or not math.isfinite(value) for value in smoke_vector)):
                    raise RuntimeError(
                        "Embedding smoke test returned a vector that is not 1024 finite numbers"
                    )

                indexed = await indexer(db, provider, config)
                if type(indexed) is not int or not 0 <= indexed <= eligible:
                    raise RuntimeError(f"Embedding indexer returned invalid indexed count: {indexed!r}")
                await index_creator(db, config)
                vector_index = await wait_for_vector_index(
                    db, config, timeout_seconds=timeout_seconds,
                    poll_interval_seconds=poll_interval_seconds,
                )
                verified = await _verify_all_current_embeddings(
                    db, config, expected_chunks=expected_chunks
                )
                summary = {
                    "database": EXPECTED_DATABASE,
                    "provider": config.provider,
                    "model": config.model,
                    "dimensions": config.dimensions,
                    "embedding_space_id": config.space_id,
                    "vector_index": config.vector_index_name,
                    "vector_index_state": vector_index["state"],
                    "vector_index_population_percent": vector_index["populationPercent"],
                    "eligible_current_chunks": eligible,
                    "newly_indexed_chunks": indexed,
                    "already_indexed_chunks": eligible - indexed,
                    "verified_current_embeddings": verified,
                }
                span.set_attribute("graph.embedding.indexed_count", indexed)
                span.set_attribute("graph.embedding.verified_count", verified)
                return summary
            finally:
                await db.close()
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-timeout", type=float, default=INDEX_TIMEOUT_SECONDS,
        help="Maximum seconds to wait for the vector index to become ONLINE (default: 300)",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=INDEX_POLL_SECONDS,
        help="Seconds between SHOW VECTOR INDEXES checks (default: 2)",
    )
    args = parser.parse_args()
    telemetry_provider, tracer = configure_telemetry()
    try:
        with tracer.start_as_current_span("graph.embedding.index_command") as span:
            try:
                summary = asyncio.run(_execute(
                    timeout_seconds=args.index_timeout,
                    poll_interval_seconds=args.poll_interval,
                ))
                print(json.dumps(summary, indent=2, sort_keys=True))
                return 0
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                print(f"ERROR: {safe_error_message(exc)}", file=sys.stderr)
                return 1
    finally:
        telemetry_provider.force_flush(timeout_millis=10000)
        telemetry_provider.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
