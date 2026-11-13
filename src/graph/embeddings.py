"""Provider-independent embedding contracts and explicit Neo4j indexing."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

_TRACER = trace.get_tracer("graph_rag.graph.embeddings")
_EMBEDDING_GUARD_CONSTRAINT = (
    "CREATE CONSTRAINT embedding_space_guard_id_unique IF NOT EXISTS "
    "FOR (g:EMBEDDING_SPACE_GUARD) REQUIRE g.id IS UNIQUE"
)


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: Literal["openai", "voyage"]
    model: str
    dimensions: int

    def __post_init__(self) -> None:
        if self.provider not in {"openai", "voyage"}:
            raise ValueError("Embedding provider must be explicitly set to openai or voyage")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("An explicit embedding model is required")
        object.__setattr__(self, "model", self.model.strip())
        if type(self.dimensions) is not int or not 1 <= self.dimensions <= 4096:
            raise ValueError("Embedding dimensions must be an integer from 1 through 4096")

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        load_dotenv()
        provider = os.getenv("EMBEDDING_PROVIDER")
        model = os.getenv("EMBEDDING_MODEL")
        raw_dimensions = os.getenv("EMBEDDING_DIMENSIONS")
        if not provider or not model or not raw_dimensions:
            raise ValueError(
                "Set EMBEDDING_PROVIDER, EMBEDDING_MODEL and EMBEDDING_DIMENSIONS explicitly"
            )
        try:
            dimensions = int(raw_dimensions)
        except ValueError as exc:
            raise ValueError("EMBEDDING_DIMENSIONS must be an integer") from exc
        return cls(provider.strip().lower(), model.strip(), dimensions)

    @property
    def space_id(self) -> str:
        identity = json.dumps([self.provider, self.model, self.dimensions], separators=(",", ":"))
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    @property
    def vector_index_name(self) -> str:
        return f"chunk_embedding_{self.space_id[:16]}"


class EmbeddingProvider(Protocol):
    """A caller-supplied adapter; this package chooses no provider."""

    provider: str
    model: str
    dimensions: int

    async def embed(self, texts: Sequence[str], *,
                    input_type: Literal["query", "document"]) -> Sequence[Sequence[float]]: ...


def create_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    load_dotenv()
    if config.provider == "openai":
        return OpenAIEmbeddingProvider(config.model, config.dimensions)
    if config.provider == "voyage":
        return VoyageEmbeddingProvider(config.model, config.dimensions)
    raise ValueError("Unsupported embedding provider")


class OpenAIEmbeddingProvider:
    provider = "openai"

    def __init__(self, model: str, dimensions: int):
        from openai import AsyncOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the configured provider")
        self.model = model
        self.dimensions = dimensions
        self._client = AsyncOpenAI(api_key=api_key)

    async def embed(self, texts: Sequence[str], *,
                    input_type: Literal["query", "document"]) -> Sequence[Sequence[float]]:
        del input_type
        response = await self._client.embeddings.create(
            model=self.model, input=list(texts), dimensions=self.dimensions,
            encoding_format="float",
        )
        if response.model != self.model:
            raise ValueError("OpenAI returned an embedding from a different model")
        return [item.embedding for item in response.data]


class VoyageEmbeddingProvider:
    provider = "voyage"

    def __init__(self, model: str, dimensions: int):
        import voyageai

        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError("VOYAGE_API_KEY is required for the configured provider")
        self.model = model
        self.dimensions = dimensions
        self._client = voyageai.AsyncClient(api_key=api_key)

    async def embed(self, texts: Sequence[str], *,
                    input_type: Literal["query", "document"]) -> Sequence[Sequence[float]]:
        response = await self._client.embed(
            list(texts), model=self.model, input_type=input_type,
            output_dimension=self.dimensions,
        )
        return response.embeddings


async def embed_texts(
    provider: EmbeddingProvider,
    config: EmbeddingConfig,
    texts: Sequence[str],
    *,
    input_type: Literal["query", "document"],
) -> list[list[float]]:
    if (provider.provider != config.provider or provider.model != config.model
            or provider.dimensions != config.dimensions):
        raise ValueError("Embedding provider does not match the configured model and dimensions")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Embedding input texts must be non-empty strings")
    vectors = await provider.embed(texts, input_type=input_type)
    if len(vectors) != len(texts):
        raise ValueError("Embedding provider returned an unexpected number of vectors")
    normalized: list[list[float]] = []
    for vector in vectors:
        if len(vector) != config.dimensions:
            raise ValueError("Embedding provider returned a vector with the wrong dimensions")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in vector):
            raise ValueError("Embedding vectors must contain only finite numbers")
        normalized.append([float(value) for value in vector])
    return normalized


def vector_index_statement(config: EmbeddingConfig) -> str:
    index_name = config.vector_index_name
    return (
        f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS FOR (c:CHUNK) ON c.embedding "
        "WITH [c.is_current, c.source_sha256, c.chunking_policy_version, "
        "c.embedding_space_id, c.embedding_model, c.embedding_dimensions] "
        f"OPTIONS {{indexConfig: {{`vector.dimensions`: {config.dimensions}, "
        "`vector.similarity_function`: 'cosine'}}"
    )


async def create_vector_index(db_connection, config: EmbeddingConfig) -> None:
    """Explicitly provision the configured index; never called implicitly."""
    statement = vector_index_statement(config)
    index_name = config.vector_index_name
    with _TRACER.start_as_current_span("graph.neo4j.create_vector_index") as span:
        span.set_attribute("db.system", "neo4j")
        span.set_attribute("graph.embedding.model", config.model)
        span.set_attribute("graph.embedding.dimensions", config.dimensions)
        span.set_attribute("graph.embedding.index_name", index_name)
        try:
            async with db_connection.get_session() as session:
                result = await session.run(_EMBEDDING_GUARD_CONSTRAINT)
                await result.consume()
                result = await session.run(statement)
                await result.consume()
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            raise


_CURRENT_CHUNKS_QUERY = """
MATCH (d:DOCUMENT)-[:HAS_CHUNK]->(c:CHUNK)
WHERE c.is_current = true
  AND c.source_sha256 = d.source_sha256
  AND c.chunking_policy_version = d.current_chunking_policy_version
  AND d.current_ingestion_run_id IS NOT NULL
RETURN DISTINCT c.id AS chunk_id, c.retrieval_text AS retrieval_text,
       c.embedding AS embedding,
       c.embedding_space_id AS embedding_space_id,
       c.embedding_model AS embedding_model,
       c.embedding_dimensions AS embedding_dimensions,
       c.source_sha256 AS source_sha256,
       d.current_chunking_policy_version AS chunking_policy_version,
       d.current_ingestion_run_id AS ingestion_run_id
ORDER BY c.id
"""
_OTHER_EMBEDDING_SPACE_QUERY = """
MATCH (c:CHUNK)
WHERE c.embedding IS NOT NULL
  AND (c.embedding_space_id IS NULL OR c.embedding_space_id <> $space_id
       OR c.embedding_model IS NULL OR c.embedding_model <> $model
       OR c.embedding_dimensions IS NULL OR c.embedding_dimensions <> $dimensions)
RETURN count(c) AS count
"""
_INCOMPLETE_EMBEDDING_QUERY = """
MATCH (c:CHUNK)
WHERE (c.embedding IS NULL AND
       (c.embedding_space_id IS NOT NULL OR c.embedding_model IS NOT NULL
        OR c.embedding_dimensions IS NOT NULL))
   OR (c.embedding IS NOT NULL AND
       (c.embedding_space_id IS NULL OR c.embedding_model IS NULL
        OR c.embedding_dimensions IS NULL))
RETURN count(c) AS count
"""
_STORE_EMBEDDINGS_QUERY = """
UNWIND $rows AS row
MATCH (d:DOCUMENT {id: row.source_sha256})-[:HAS_CHUNK]->(c:CHUNK {id: row.chunk_id})
WHERE c.is_current = true
  AND c.source_sha256 = d.source_sha256
  AND c.chunking_policy_version = d.current_chunking_policy_version
  AND d.current_ingestion_run_id = row.ingestion_run_id
  AND c.embedding IS NULL
  AND c.embedding_space_id IS NULL
  AND c.embedding_model IS NULL
  AND c.embedding_dimensions IS NULL
SET c.embedding = row.embedding,
    c.embedding_model = row.embedding_model,
    c.embedding_dimensions = row.embedding_dimensions,
    c.embedding_space_id = row.embedding_space_id
RETURN count(c) AS stored
"""
_LOCK_EMBEDDING_SPACE_QUERY = """
MERGE (g:EMBEDDING_SPACE_GUARD {id: 'chunk-embedding-space'})
ON CREATE SET g.write_lock = 1
ON MATCH SET g.write_lock = coalesce(g.write_lock, 0) + 1
RETURN g.embedding_space_id AS embedding_space_id
"""
_SET_EMBEDDING_SPACE_QUERY = """
MATCH (g:EMBEDDING_SPACE_GUARD {id: 'chunk-embedding-space'})
SET g.embedding_space_id = $space_id
"""


def _embedding_is_missing(row: dict) -> bool:
    return all(row.get(key) is None for key in (
        "embedding", "embedding_space_id", "embedding_model", "embedding_dimensions"
    ))


def _validate_existing_embedding(row: dict, config: EmbeddingConfig) -> None:
    if _embedding_is_missing(row):
        return
    vector = row.get("embedding")
    if (not isinstance(vector, (list, tuple)) or len(vector) != config.dimensions
            or row.get("embedding_space_id") != config.space_id
            or row.get("embedding_model") != config.model
            or row.get("embedding_dimensions") != config.dimensions
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in vector)):
        raise ValueError("Current chunk has incomplete or conflicting embedding metadata")


def _current_chunk_snapshot(rows: list[dict]) -> tuple[tuple, ...]:
    """Identity and document-run view used to detect stale embedding work."""
    return tuple(sorted((
        row.get("chunk_id"), row.get("source_sha256"), row.get("chunking_policy_version"),
        row.get("ingestion_run_id"), row.get("retrieval_text"),
    ) for row in rows))


def _same_embedding(left: dict, right: dict) -> bool:
    return all(left.get(key) == right.get(key) for key in (
        "embedding", "embedding_space_id", "embedding_model", "embedding_dimensions"
    ))


async def index_current_chunks(db_connection, provider: EmbeddingProvider,
                               config: EmbeddingConfig) -> int:
    """Embed current retrieval text once; fail rather than mix model spaces."""
    if (provider.provider != config.provider or provider.model != config.model
            or provider.dimensions != config.dimensions):
        raise ValueError("Embedding provider does not match the configured model and dimensions")
    with _TRACER.start_as_current_span("graph.neo4j.index_current_chunk_embeddings") as span:
        span.set_attribute("db.system", "neo4j")
        span.set_attribute("graph.embedding.model", config.model)
        span.set_attribute("graph.embedding.dimensions", config.dimensions)
        try:
            async with db_connection.get_session() as session:
                result = await session.run(_EMBEDDING_GUARD_CONSTRAINT)
                await result.consume()
                result = await session.run(_CURRENT_CHUNKS_QUERY)
                chunks = await result.data()
                if not chunks:
                    span.set_attribute("graph.embedding.indexed_chunks", 0)
                    return 0
                for row in chunks:
                    _validate_existing_embedding(row, config)
                initial_snapshot = _current_chunk_snapshot(chunks)
                initial_by_id = {row["chunk_id"]: row for row in chunks}
                missing = [row for row in chunks if _embedding_is_missing(row)]

            # External API calls occur with no open Neo4j session or transaction.
            vectors: list[list[float]] = []
            for offset in range(0, len(missing), 64):
                batch = missing[offset:offset + 64]
                vectors.extend(await embed_texts(
                    provider, config,
                    [row["retrieval_text"] for row in batch],
                    input_type="document",
                ))
            pending_rows = {
                chunk["chunk_id"]: {
                    "chunk_id": chunk["chunk_id"],
                    "source_sha256": chunk["source_sha256"],
                    "ingestion_run_id": chunk["ingestion_run_id"],
                    "embedding": vector,
                    "embedding_model": config.model,
                    "embedding_dimensions": config.dimensions,
                    "embedding_space_id": config.space_id,
                }
                for chunk, vector in zip(missing, vectors, strict=True)
            }

            async def transaction(tx):
                guard_result = await tx.run(_LOCK_EMBEDDING_SPACE_QUERY)
                guard = await guard_result.single()
                existing_space = guard["embedding_space_id"] if guard else None
                if existing_space not in (None, config.space_id):
                    raise ValueError("Neo4j embedding-space lock belongs to another model space")

                current_result = await tx.run(_CURRENT_CHUNKS_QUERY)
                current_chunks = await current_result.data()
                if _current_chunk_snapshot(current_chunks) != initial_snapshot:
                    raise RuntimeError(
                        "Current chunk set or ingestion-run snapshot changed while embeddings were generated"
                    )
                for row in current_chunks:
                    _validate_existing_embedding(row, config)
                current_by_id = {row["chunk_id"]: row for row in current_chunks}
                if current_by_id.keys() != initial_by_id.keys():
                    raise RuntimeError("Current chunk set changed while embeddings were generated")

                for chunk_id, initial in initial_by_id.items():
                    current = current_by_id[chunk_id]
                    if not _embedding_is_missing(initial) and not _same_embedding(initial, current):
                        raise ValueError("Existing embedding changed during indexing; refusing overwrite")

                incomplete_result = await tx.run(_INCOMPLETE_EMBEDDING_QUERY)
                incomplete = await incomplete_result.data()
                if incomplete and incomplete[0]["count"]:
                    raise ValueError("Neo4j contains chunks with incomplete embedding metadata")
                conflict_result = await tx.run(
                    _OTHER_EMBEDDING_SPACE_QUERY, space_id=config.space_id,
                    model=config.model, dimensions=config.dimensions,
                )
                conflicts = await conflict_result.data()
                if conflicts and conflicts[0]["count"]:
                    raise ValueError("Neo4j contains embeddings from a different embedding space")

                rows_to_store = []
                for chunk_id, row in pending_rows.items():
                    current = current_by_id[chunk_id]
                    if _embedding_is_missing(current):
                        rows_to_store.append(row)
                    # Another writer may have completed this exact embedding space
                    # while this request was in flight. Keep its value unchanged.
                count = 0
                if rows_to_store:
                    result = await tx.run(_STORE_EMBEDDINGS_QUERY, rows=rows_to_store)
                    stored = await result.single()
                    count = stored["stored"] if stored else 0
                    if count != len(rows_to_store):
                        raise RuntimeError(
                            "Current chunk set changed while embeddings were being stored"
                        )
                await tx.run(_SET_EMBEDDING_SPACE_QUERY, space_id=config.space_id)
                return count

            async with db_connection.get_session() as write_session:
                count = await write_session.execute_write(transaction)
            span.set_attribute("graph.embedding.indexed_chunks", count)
            return count
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            raise
