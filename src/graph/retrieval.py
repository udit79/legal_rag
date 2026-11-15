"""Current-corpus retrieval, evidence assembly, and rank fusion."""

from __future__ import annotations

import json
import math
import asyncio
from dataclasses import dataclass
from typing import Any, Literal

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.graph.embeddings import EmbeddingConfig, EmbeddingProvider, embed_texts
from src.ingestion.chunk_contracts import ChunkSource, VisualAssetReference

_TRACER = trace.get_tracer("graph_rag.graph.retrieval")
RetrievalChannel = Literal["vector", "fulltext"]

_FULLTEXT_QUERY = """
CYPHER 25
// Cypher 25 SEARCH full-text index lookup.
MATCH (c:CHUNK)
SEARCH c IN (
  FULLTEXT INDEX chunk_retrieval_text_fulltext
  FOR $search_query
  LIMIT $candidate_count
) SCORE AS score
MATCH (d:DOCUMENT)-[:HAS_CHUNK]->(c)
WHERE c.is_current = true
  AND c.source_sha256 = d.source_sha256
  AND c.chunking_policy_version = d.current_chunking_policy_version
  AND d.current_ingestion_run_id IS NOT NULL
  AND d.current_ingestion_created_at IS NOT NULL
RETURN DISTINCT c.id AS chunk_id, d.source_sha256 AS document_sha256,
       c.raw_text AS raw_text, c.sources_json AS sources_json,
       d.current_ingestion_run_id AS ingestion_run_id, score
ORDER BY score DESC, chunk_id ASC
LIMIT $limit
"""

_VECTOR_QUERY_TEMPLATE = """
CYPHER 25
// Cypher 25 SEARCH vector index lookup with in-index current/model-space filters.
MATCH (c:CHUNK)
SEARCH c IN (
  VECTOR INDEX {index_name}
  FOR $query_vector
  WHERE c.is_current = true
    AND c.embedding_space_id = $embedding_space_id
    AND c.embedding_model = $embedding_model
    AND c.embedding_dimensions = $embedding_dimensions
  LIMIT $candidate_count
) SCORE AS score
MATCH (d:DOCUMENT)-[:HAS_CHUNK]->(c)
WHERE c.is_current = true
  AND c.source_sha256 = d.source_sha256
  AND c.chunking_policy_version = d.current_chunking_policy_version
  AND d.current_ingestion_run_id IS NOT NULL
  AND d.current_ingestion_created_at IS NOT NULL
  AND c.embedding_model = $embedding_model
  AND c.embedding_dimensions = $embedding_dimensions
  AND c.embedding_space_id = $embedding_space_id
RETURN DISTINCT c.id AS chunk_id, d.source_sha256 AS document_sha256,
       c.raw_text AS raw_text, c.sources_json AS sources_json,
       d.current_ingestion_run_id AS ingestion_run_id, score
ORDER BY score DESC, chunk_id ASC
LIMIT $limit
"""

_INDEX_STATES_QUERY = """
SHOW INDEXES YIELD name, type, state
WHERE name IN $names
RETURN name, type, state
"""


class SearchIndexUnavailable(RuntimeError):
    pass


async def _require_search_indexes_online(
    db_connection,
    vector_index: str | None = None,
    *,
    require_fulltext: bool = True,
) -> None:
    expected = {}
    if require_fulltext:
        expected["chunk_retrieval_text_fulltext"] = "FULLTEXT"
    if vector_index is not None:
        expected[vector_index] = "VECTOR"
    if not expected:
        raise ValueError("At least one Neo4j search index must be required")
    async with db_connection.get_session() as session:
        result = await session.run(_INDEX_STATES_QUERY, names=list(expected))
        rows = await result.data()
    actual = {row["name"]: (row["type"], row["state"]) for row in rows}
    unavailable = [name for name, index_type in expected.items()
                   if actual.get(name) != (index_type, "ONLINE")]
    if unavailable:
        raise SearchIndexUnavailable(
            "Required Neo4j search indexes are not ONLINE: " + ", ".join(unavailable)
        )


class EvidenceBundle(BaseModel):
    """Normalized source evidence for every retrieval strategy.

    Standalone retrieval stores its native channel score and leaves ``rrf_score``
    unset. Hybrid retrieval stores both native scores when present and its RRF
    score. Abstention uses RRF for hybrid evidence and the sole native score for
    standalone evidence.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(min_length=1)
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_text: str = Field(min_length=1)
    sources: tuple[ChunkSource, ...] = Field(min_length=1)
    ingestion_run_id: str = Field(min_length=1)
    retrieval_channels: tuple[RetrievalChannel, ...] = Field(min_length=1)
    vector_score: float | None = None
    fulltext_score: float | None = None
    rrf_score: float | None = Field(default=None, ge=0)
    visual_assets: tuple[VisualAssetReference, ...] = ()

    @model_validator(mode="after")
    def validate_visual_asset_run(self) -> "EvidenceBundle":
        expected = {
            (source.page_number, source.source_image_block_id)
            for source in self.sources
            if source.extraction_method in {"IMAGE_OCR", "IMAGE_VLM"}
            and source.source_image_block_id is not None
        }
        actual = set()
        for asset in self.visual_assets:
            key = (asset.page_number, asset.source_block_id)
            if asset.ingestion_run_id != self.ingestion_run_id or key not in expected:
                raise ValueError("Visual asset reference does not match evidence provenance")
            actual.add(key)
        if len(actual) != len(self.visual_assets):
            raise ValueError("Evidence contains duplicate visual asset references")
        return self

    @property
    def abstention_score(self) -> float:
        if self.rrf_score is not None:
            return self.rrf_score
        if self.retrieval_channels == ("vector",) and self.vector_score is not None:
            return self.vector_score
        if self.retrieval_channels == ("fulltext",) and self.fulltext_score is not None:
            return self.fulltext_score
        raise ValueError("Evidence does not have an unambiguous abstention score")

    @property
    def abstention_score_type(self) -> str:
        if self.rrf_score is not None:
            return "rrf_score"
        if self.retrieval_channels == ("vector",) and self.vector_score is not None:
            return "vector_relevance_score"
        if self.retrieval_channels == ("fulltext",) and self.fulltext_score is not None:
            return "fulltext_relevance_score"
        raise ValueError("Evidence does not have an unambiguous abstention score")


class VisualAssetResolutionUnavailable(RuntimeError):
    pass


async def resolve_visual_asset_references(
    evidence: list[EvidenceBundle], resolver: Any | None
) -> list[EvidenceBundle]:
    """Resolve run-scoped private assets after Neo4j sessions have closed."""
    keys_by_run: dict[str, set[tuple[int, str]]] = {}
    for item in evidence:
        for source in item.sources:
            if source.extraction_method in {"IMAGE_OCR", "IMAGE_VLM"}:
                if source.source_image_block_id is None:
                    raise ValueError("Derived image evidence lacks its source image block ID")
                keys_by_run.setdefault(item.ingestion_run_id, set()).add(
                    (source.page_number, source.source_image_block_id)
                )
    if not keys_by_run:
        return evidence
    if resolver is None:
        raise VisualAssetResolutionUnavailable(
            "OCR evidence requires a configured Supabase image asset resolver"
        )
    resolved_by_run = {
        run_id: await resolver.resolve(run_id, tuple(sorted(keys)))
        for run_id, keys in keys_by_run.items()
    }
    enriched = []
    for item in evidence:
        references: dict[tuple[int, str], VisualAssetReference] = {}
        for source in item.sources:
            if source.extraction_method not in {"IMAGE_OCR", "IMAGE_VLM"}:
                continue
            assert source.source_image_block_id is not None
            key = (source.page_number, source.source_image_block_id)
            resolved = resolved_by_run[item.ingestion_run_id].get(key)
            expected_digest = (
                source.ocr_text_sha256 if source.extraction_method == "IMAGE_OCR"
                else source.visual_description_sha256
            )
            if resolved is None or (
                source.extraction_method == "IMAGE_OCR"
                and resolved.ocr_text_sha256 != expected_digest
            ) or (
                source.extraction_method == "IMAGE_VLM"
                and (
                    resolved.visual_description_sha256 != expected_digest
                    or resolved.visual_prompt_version != source.visual_prompt_version
                )
            ):
                raise VisualAssetResolutionUnavailable(
                    f"Image asset does not match derived image provenance: {key}"
                )
            references[key] = resolved.reference
        enriched.append(EvidenceBundle.model_validate({
            **item.model_dump(mode="python"),
            "visual_assets": tuple(references[key] for key in sorted(references)),
        }))
    return enriched

def _as_evidence_bundle(hit: RetrievalHit) -> EvidenceBundle:
    return EvidenceBundle(
        chunk_id=hit.chunk_id,
        document_sha256=hit.document_sha256,
        raw_text=hit.raw_text,
        sources=hit.sources,
        ingestion_run_id=hit.ingestion_run_id,
        retrieval_channels=(hit.channel,),
        vector_score=hit.score if hit.channel == "vector" else None,
        fulltext_score=hit.score if hit.channel == "fulltext" else None,
        rrf_score=None,
    )


@dataclass(frozen=True)
class RetrievalHit:
    chunk_id: str
    document_sha256: str
    raw_text: str
    sources: tuple[ChunkSource, ...]
    ingestion_run_id: str
    channel: RetrievalChannel
    score: float


def _hit(record: Any, channel: RetrievalChannel) -> RetrievalHit:
    sources_json = record["sources_json"]
    sources = json.loads(sources_json) if isinstance(sources_json, str) else sources_json
    if not isinstance(sources, list) or not sources:
        raise ValueError("Neo4j returned missing source provenance for a retrieval hit")
    score = float(record["score"])
    if not math.isfinite(score):
        raise ValueError("Neo4j returned a non-finite retrieval score")
    return RetrievalHit(
        chunk_id=record["chunk_id"],
        document_sha256=record["document_sha256"],
        raw_text=record["raw_text"],
        sources=tuple(ChunkSource.model_validate(source) for source in sources),
        ingestion_run_id=record["ingestion_run_id"],
        channel=channel,
        score=score,
    )


class FullTextRetriever:
    def __init__(self, db_connection, candidate_multiplier: int = 5,
                 *, image_asset_resolver: Any | None = None):
        if type(candidate_multiplier) is not int or candidate_multiplier < 1:
            raise ValueError("Candidate multiplier must be a positive integer")
        self._db = db_connection
        self._index_name = "chunk_retrieval_text_fulltext"
        self._candidate_multiplier = candidate_multiplier
        self._image_asset_resolver = image_asset_resolver

    @property
    def configuration(self) -> dict[str, Any]:
        return {
            "strategy": "fulltext",
            "fulltext_index": self._index_name,
            "candidate_multiplier": self._candidate_multiplier,
            "abstention_score_policy": "fulltext_relevance_score",
        }

    async def search(self, query: str, limit: int = 10) -> list[EvidenceBundle]:
        if not query.strip() or type(limit) is not int or limit < 1:
            raise ValueError("Search requires a non-empty query and positive limit")
        await _require_search_indexes_online(self._db)
        evidence = [_as_evidence_bundle(hit) for hit in await self._search(query, limit)]
        return await resolve_visual_asset_references(evidence, self._image_asset_resolver)

    async def _search(self, query: str, limit: int) -> list[RetrievalHit]:
        with _TRACER.start_as_current_span("graph.neo4j.fulltext_retrieval") as span:
            span.set_attribute("db.system", "neo4j")
            span.set_attribute("graph.retrieval.channel", "fulltext")
            span.set_attribute("graph.retrieval.limit", limit)
            try:
                async with self._db.get_session() as session:
                    result = await session.run(
                        _FULLTEXT_QUERY, search_query=query,
                        candidate_count=limit * self._candidate_multiplier,
                        limit=limit,
                    )
                    hits = [_hit(row, "fulltext") for row in await result.data()]
                span.set_attribute("graph.retrieval.result_count", len(hits))
                return hits
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                raise


class VectorRetriever:
    def __init__(self, db_connection, provider: EmbeddingProvider,
                 config: EmbeddingConfig, candidate_multiplier: int = 5,
                 *, image_asset_resolver: Any | None = None):
        index_name = config.vector_index_name
        if type(candidate_multiplier) is not int or candidate_multiplier < 1:
            raise ValueError("Candidate multiplier must be a positive integer")
        if (provider.provider != config.provider or provider.model != config.model
                or provider.dimensions != config.dimensions):
            raise ValueError("Embedding provider does not match the configured model and dimensions")
        self._db = db_connection
        self._provider = provider
        self._config = config
        self._index_name = index_name
        self._candidate_multiplier = candidate_multiplier
        self._image_asset_resolver = image_asset_resolver
        self._query = _VECTOR_QUERY_TEMPLATE.format(index_name=index_name)

    @property
    def configuration(self) -> dict[str, Any]:
        return {
            "strategy": "vector",
            "embedding_provider": self._config.provider,
            "embedding_model": self._config.model,
            "embedding_dimensions": self._config.dimensions,
            "embedding_space_id": self._config.space_id,
            "vector_index": self._index_name,
            "candidate_multiplier": self._candidate_multiplier,
            "abstention_score_policy": "vector_relevance_score",
        }

    async def search(self, query: str, limit: int = 10) -> list[EvidenceBundle]:
        if not query.strip() or type(limit) is not int or limit < 1:
            raise ValueError("Search requires a non-empty query and positive limit")
        await _require_search_indexes_online(
            self._db, self._index_name, require_fulltext=False
        )
        evidence = [_as_evidence_bundle(hit) for hit in await self._search(query, limit)]
        return await resolve_visual_asset_references(evidence, self._image_asset_resolver)

    async def _search(self, query: str, limit: int) -> list[RetrievalHit]:
        with _TRACER.start_as_current_span("graph.neo4j.vector_retrieval") as span:
            span.set_attribute("db.system", "neo4j")
            span.set_attribute("graph.retrieval.channel", "vector")
            span.set_attribute("graph.embedding.provider", self._config.provider)
            span.set_attribute("graph.embedding.model", self._config.model)
            span.set_attribute("graph.embedding.dimensions", self._config.dimensions)
            try:
                vector, = await embed_texts(
                    self._provider, self._config, [query], input_type="query"
                )
                async with self._db.get_session() as session:
                    result = await session.run(
                        self._query,
                        candidate_count=limit * self._candidate_multiplier,
                        query_vector=vector,
                        embedding_model=self._config.model,
                        embedding_dimensions=self._config.dimensions,
                        embedding_space_id=self._config.space_id,
                        limit=limit,
                    )
                    hits = [_hit(row, "vector") for row in await result.data()]
                span.set_attribute("graph.retrieval.result_count", len(hits))
                return hits
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                raise


@dataclass(frozen=True)
class RRFConfig:
    rank_constant: int = 60
    vector_weight: float = 1.0
    fulltext_weight: float = 1.0

    def __post_init__(self) -> None:
        if type(self.rank_constant) is not int or self.rank_constant < 0:
            raise ValueError("RRF rank_constant must be a non-negative integer")
        if (not math.isfinite(self.vector_weight) or self.vector_weight < 0
                or not math.isfinite(self.fulltext_weight) or self.fulltext_weight < 0
                or self.vector_weight + self.fulltext_weight == 0):
            raise ValueError("RRF weights must be finite, non-negative, and not both zero")


def reciprocal_rank_fusion(
    vector_hits: list[RetrievalHit],
    fulltext_hits: list[RetrievalHit],
    config: RRFConfig = RRFConfig(),
    limit: int = 10,
) -> list[EvidenceBundle]:
    if type(limit) is not int or limit < 1:
        raise ValueError("Fusion requires a positive result limit")
    hits_by_id: dict[str, dict[str, Any]] = {}
    for channel, hits, weight in (
        ("vector", vector_hits, config.vector_weight),
        ("fulltext", fulltext_hits, config.fulltext_weight),
    ):
        if weight == 0:
            continue
        seen_channel: set[str] = set()
        for rank, hit in enumerate(hits, start=1):
            if hit.channel != channel:
                raise ValueError(f"Expected {channel} hits in the {channel} result list")
            if hit.chunk_id in seen_channel:
                continue
            seen_channel.add(hit.chunk_id)
            entry = hits_by_id.setdefault(hit.chunk_id, {
                "hit": hit, "channels": [], "rrf_score": 0.0,
                "vector_score": None, "fulltext_score": None,
            })
            original = entry["hit"]
            if (hit.document_sha256 != original.document_sha256
                    or hit.raw_text != original.raw_text
                    or hit.sources != original.sources
                    or hit.ingestion_run_id != original.ingestion_run_id):
                raise ValueError(f"Retrievers returned inconsistent evidence for {hit.chunk_id}")
            entry["channels"].append(channel)
            entry["rrf_score"] += weight / (config.rank_constant + rank)
            entry[f"{channel}_score"] = hit.score

    ordered = sorted(
        hits_by_id.items(),
        key=lambda item: (-item[1]["rrf_score"], item[0]),
    )[:limit]
    return [EvidenceBundle(
        chunk_id=chunk_id,
        document_sha256=entry["hit"].document_sha256,
        raw_text=entry["hit"].raw_text,
        sources=entry["hit"].sources,
        ingestion_run_id=entry["hit"].ingestion_run_id,
        retrieval_channels=tuple(entry["channels"]),
        vector_score=entry["vector_score"],
        fulltext_score=entry["fulltext_score"],
        rrf_score=entry["rrf_score"],
    ) for chunk_id, entry in ordered]


class HybridRetriever:
    def __init__(self, vector: VectorRetriever, fulltext: FullTextRetriever,
                 fusion: RRFConfig = RRFConfig(),
                 *, image_asset_resolver: Any | None = None):
        self._vector = vector
        self._fulltext = fulltext
        self._fusion = fusion
        self._image_asset_resolver = (
            image_asset_resolver
            or vector._image_asset_resolver
            or fulltext._image_asset_resolver
        )

    @property
    def configuration(self) -> dict[str, Any]:
        return {
            "strategy": "hybrid",
            "embedding_provider": self._vector._config.provider,
            "embedding_model": self._vector._config.model,
            "embedding_dimensions": self._vector._config.dimensions,
            "embedding_space_id": self._vector._config.space_id,
            "vector_index": self._vector._index_name,
            "fulltext_index": self._fulltext._index_name,
            "vector_candidate_multiplier": self._vector._candidate_multiplier,
            "fulltext_candidate_multiplier": self._fulltext._candidate_multiplier,
            "rrf": {
                "rank_constant": self._fusion.rank_constant,
                "vector_weight": self._fusion.vector_weight,
                "fulltext_weight": self._fusion.fulltext_weight,
            },
            "abstention_score_policy": "rrf_score",
        }

    async def search(self, query: str, limit: int = 10) -> list[EvidenceBundle]:
        if not query.strip() or type(limit) is not int or limit < 1:
            raise ValueError("Search requires a non-empty query and positive limit")
        with _TRACER.start_as_current_span("graph.retrieval.hybrid") as span:
            span.set_attribute("graph.retrieval.limit", limit)
            try:
                await _require_search_indexes_online(
                    self._vector._db, self._vector._index_name
                )
                vector_hits, fulltext_hits = await asyncio.gather(
                    self._vector._search(query, limit),
                    self._fulltext._search(query, limit),
                )
                evidence = reciprocal_rank_fusion(vector_hits, fulltext_hits,
                                                  self._fusion, limit)
                span.set_attribute("graph.retrieval.result_count", len(evidence))
                return await resolve_visual_asset_references(evidence, self._image_asset_resolver)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                raise
