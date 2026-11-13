"""Read-only smoke test for current-corpus full-text, vector, and hybrid retrieval."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from pathlib import Path
from typing import Literal, Sequence

from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.graph.embeddings import EmbeddingConfig, create_embedding_provider
from src.graph.evaluation import load_evaluation_dataset, run_and_save_evaluation
from src.graph.retrieval import (
    EvidenceBundle,
    FullTextRetriever,
    HybridRetriever,
    RRFConfig,
    SearchIndexUnavailable,
    VectorRetriever,
    _require_search_indexes_online,
)
from src.infrastructure.neo4j_client import create_db_connect
from src.observability.telemetry import configure_telemetry
from src.ingestion.chunk_contracts import ChunkSource

_TRACER = trace.get_tracer("graph_rag.graph.retrieval_smoke")
DEFAULT_QUERIES = (
    "What security safeguards should an organization use to protect personal data?",
    "What duties apply when a data principal withdraws consent?",
    "What controls support business continuity and disaster recovery?",
    "What is the role of the appellate tribunal in personal data protection?",
    "How should an organization manage backup restoration and data retention?",
)
DEFAULT_DATASET_PATH = Path("test_data/phase3_ai_silver_eval_v1.json")
ARTIFACT_DIRECTORY = Path("artifacts/graph_retrieval_smoke")
_VERIFY_EVIDENCE_QUERY = """
MATCH (d:DOCUMENT)-[:HAS_CHUNK]->(c:CHUNK)
WHERE d.source_sha256 IN $document_hashes AND c.id IN $chunk_ids
RETURN c.id AS chunk_id, d.source_sha256 AS document_sha256,
       d.current_ingestion_run_id AS ingestion_run_id,
       d.current_chunking_policy_version AS current_policy,
       c.chunking_policy_version AS chunk_policy,
       c.is_current AS is_current,
       c.raw_text AS raw_text, c.sources_json AS sources_json
"""


class _CachedQueryEmbeddingProvider:
    """Reuse vectors between standalone and hybrid runs in this smoke process."""

    def __init__(self, provider):
        self._provider = provider
        self.provider = provider.provider
        self.model = provider.model
        self.dimensions = provider.dimensions
        self._cache: dict[tuple[str, str], list[float]] = {}

    async def embed(
        self,
        texts: Sequence[str],
        *,
        input_type: Literal["query", "document"],
    ):
        keys = [(input_type, text) for text in texts]
        missing = [text for key, text in zip(keys, texts, strict=True) if key not in self._cache]
        if missing:
            vectors = await self._provider.embed(missing, input_type=input_type)
            self._cache.update({(input_type, text): vector
                                for text, vector in zip(missing, vectors, strict=True)})
        return [self._cache[key] for key in keys]


def _validate_bundle(strategy: str, item: EvidenceBundle) -> None:
    if not item.chunk_id:
        raise ValueError("Retriever returned an empty chunk_id")
    if len(item.document_sha256) != 64:
        raise ValueError(f"Invalid document hash for chunk {item.chunk_id}")
    if not item.raw_text.strip() or not item.sources or not item.ingestion_run_id:
        raise ValueError(f"Missing raw text, sources, or ingestion run for {item.chunk_id}")
    channels = set(item.retrieval_channels)
    if strategy == "fulltext":
        if channels != {"fulltext"} or item.fulltext_score is None or item.vector_score is not None:
            raise ValueError(f"Invalid full-text score/channel metadata for {item.chunk_id}")
        if item.rrf_score is not None:
            raise ValueError("Standalone full-text evidence must not have an RRF score")
    elif strategy == "vector":
        if channels != {"vector"} or item.vector_score is None or item.fulltext_score is not None:
            raise ValueError(f"Invalid vector score/channel metadata for {item.chunk_id}")
        if item.rrf_score is not None:
            raise ValueError("Standalone vector evidence must not have an RRF score")
    elif strategy == "hybrid":
        if not channels or item.rrf_score is None:
            raise ValueError(f"Invalid hybrid channel/RRF metadata for {item.chunk_id}")
        if ("vector" in channels) != (item.vector_score is not None):
            raise ValueError(f"Hybrid vector score does not match channels for {item.chunk_id}")
        if ("fulltext" in channels) != (item.fulltext_score is not None):
            raise ValueError(f"Hybrid full-text score does not match channels for {item.chunk_id}")
    else:
        raise ValueError(f"Unknown retrieval strategy: {strategy}")
    numeric_scores = [item.abstention_score]
    numeric_scores.extend(score for score in (item.vector_score, item.fulltext_score, item.rrf_score)
                          if score is not None)
    if any(not math.isfinite(score) for score in numeric_scores):
        raise ValueError(f"Non-finite retrieval score for {item.chunk_id}")


async def _validate_current_provenance(db, evidence: Sequence[EvidenceBundle]) -> None:
    if not evidence:
        return
    expected = {(item.chunk_id, item.document_sha256): item for item in evidence}
    async with db.get_session() as session:
        result = await session.run(
            _VERIFY_EVIDENCE_QUERY,
            document_hashes=sorted({item.document_sha256 for item in evidence}),
            chunk_ids=sorted({item.chunk_id for item in evidence}),
        )
        rows = await result.data()
    actual = {(row["chunk_id"], row["document_sha256"]): row for row in rows}
    if actual.keys() != expected.keys():
        missing = sorted(expected.keys() - actual.keys())
        raise ValueError(f"Retrieved evidence is not linked to a current document chunk: {missing}")
    for identity, item in expected.items():
        row = actual[identity]
        sources = (json.loads(row["sources_json"])
                   if isinstance(row["sources_json"], str) else row["sources_json"])
        if (row["ingestion_run_id"] != item.ingestion_run_id or row["is_current"] is not True
                or row["chunk_policy"] != row["current_policy"]):
            raise ValueError(f"Retrieved evidence has stale document/chunk provenance: {identity}")
        if (not isinstance(row["raw_text"], str) or not row["raw_text"].strip()
                or row["raw_text"] != item.raw_text or not isinstance(sources, list) or not sources):
            raise ValueError(f"Current chunk has incomplete source provenance: {identity}")
        if tuple(ChunkSource.model_validate(source) for source in sources) != item.sources:
            raise ValueError(f"Retrieved source coordinates differ from stored chunk: {identity}")


def _evidence_summary(item: EvidenceBundle) -> dict:
    return {
        "chunk_id": item.chunk_id,
        "document_sha256": item.document_sha256,
        "ingestion_run_id": item.ingestion_run_id,
        "raw_text_length": len(item.raw_text),
        "sources": {
            "count": len(item.sources),
            "page_numbers": sorted({source.page_number for source in item.sources}),
            "block_ids": [source.block_id for source in item.sources[:4]],
        },
        "retrieval_channels": list(item.retrieval_channels),
        "vector_score": item.vector_score,
        "fulltext_score": item.fulltext_score,
        "rrf_score": item.rrf_score,
        "abstention_score": item.abstention_score,
        "abstention_score_type": item.abstention_score_type,
    }


async def _run(args) -> dict:
    load_dotenv()
    config = EmbeddingConfig.from_env()
    # Validate the selected dataset before opening a database connection or
    # executing any retrieval request.
    dataset = load_evaluation_dataset(args.dataset)
    if dataset.dataset_id == "phase3-ai-silver-development-evaluation":
        if (dataset.human_verification_status != "NOT_PERFORMED"
                or dataset.ground_truth_status != "NOT_ESTABLISHED"):
            raise ValueError("Phase 3 AI-silver benchmark must remain provisional")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    if database != "rag":
        raise ValueError(f"NEO4J_DATABASE must be 'rag' for this smoke test, got {database!r}")
    if not args.queries or any(not query.strip() for query in args.queries):
        raise ValueError("At least one non-empty retrieval query is required")
    if args.limit < 1:
        raise ValueError("Result limit must be positive")

    db = create_db_connect()
    try:
        if db.database != "rag":
            raise ValueError(f"Neo4j client selected {db.database!r}; expected 'rag'")
        await db.verify_connection()
        strategies_requested = args.strategies or ["fulltext", "vector", "hybrid"]
        if any(name in strategies_requested for name in ("fulltext", "hybrid")):
            await _require_search_indexes_online(db)
        if any(name in strategies_requested for name in ("vector", "hybrid")):
            await _require_search_indexes_online(
                db, config.vector_index_name, require_fulltext=False
            )

        fulltext = FullTextRetriever(db)
        strategies: dict[
            str, FullTextRetriever | VectorRetriever | HybridRetriever
        ] = {"fulltext": fulltext}
        if any(name in strategies_requested for name in ("vector", "hybrid")):
            provider = _CachedQueryEmbeddingProvider(create_embedding_provider(config))
            vector = VectorRetriever(db, provider, config)
            strategies["vector"] = vector
            strategies["hybrid"] = HybridRetriever(vector, fulltext, RRFConfig())
        if len(strategies_requested) != len(set(strategies_requested)) or any(
            name not in strategies for name in strategies_requested
        ):
            raise ValueError(f"Invalid or duplicate retrieval strategy: {strategies_requested}")
        strategies = {name: strategies[name] for name in strategies_requested}
        strategy_results = {}
        for name, retriever in strategies.items():
            rows = []
            for query in args.queries:
                evidence = await retriever.search(query, limit=args.limit)
                for item in evidence:
                    _validate_bundle(name, item)
                await _validate_current_provenance(db, evidence)
                rows.append({
                    "query": query,
                    "result_count": len(evidence),
                    "evidence": [_evidence_summary(item) for item in evidence[:1]],
                })
            strategy_results[name] = {
                "configuration": retriever.configuration,
                "queries": rows,
            }

        if not dataset.synthetic:
            raise ValueError("The selected Phase 3 development dataset must be synthetic/provisional")
        evaluation_results = {}
        for name, retriever in strategies.items():
            result, artifact = await run_and_save_evaluation(
                dataset,
                retriever,
                artifact_directory=ARTIFACT_DIRECTORY / name,
                k=args.limit,
                abstention_score_threshold=args.abstention_threshold,
            )
            evaluation_results[name] = {
                "synthetic_smoke_only": True,
                "metrics": result["metrics"],
                "artifact": str(artifact),
            }

        return {
            "database": database,
            "embedding": {
                "provider": config.provider,
                "model": config.model,
                "dimensions": config.dimensions,
                "embedding_space_id": config.space_id,
                "vector_index": config.vector_index_name,
            },
            "queries": list(args.queries),
            "strategies_run": list(strategies),
            "retrieval": strategy_results,
            "synthetic_evaluation": evaluation_results,
        }
    finally:
        await db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=5, help="Results per query")
    parser.add_argument(
        "--abstention-threshold", type=float, default=0.01,
        help="Smoke-only threshold in each strategy's declared score scale",
    )
    parser.add_argument("--query", dest="queries", action="append",
                        help="Override/add a query; may be passed five times")
    parser.add_argument("--strategy", dest="strategies", choices=("fulltext", "vector", "hybrid"),
                        action="append", help="Run a strategy only; may be repeated")
    parser.add_argument(
        "--output", type=Path, default=ARTIFACT_DIRECTORY / "latest.json",
        help="Write the complete smoke report here",
    )
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET_PATH,
        help="Evaluation dataset JSON (validated before retrieval)",
    )
    args = parser.parse_args()
    if args.queries is None:
        args.queries = list(DEFAULT_QUERIES)
    elif len(args.queries) != 5:
        parser.error("Pass exactly five --query values when overriding defaults")

    telemetry_provider, tracer = configure_telemetry()
    try:
        with tracer.start_as_current_span("graph.retrieval.smoke_run") as span:
            try:
                result = asyncio.run(_run(args))
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                concise = {
                    "database": result["database"],
                    "embedding": result["embedding"],
                    "strategies_run": result["strategies_run"],
                    "retrieval": {
                        strategy: {
                            "queries": [{
                                "query": row["query"],
                                "result_count": row["result_count"],
                                "sample_evidence": row["evidence"],
                            } for row in strategy_result["queries"]],
                        }
                        for strategy, strategy_result in result["retrieval"].items()
                    },
                    "synthetic_evaluation": result["synthetic_evaluation"],
                    "complete_report": str(args.output),
                }
                print(json.dumps(concise, indent=2, ensure_ascii=False, sort_keys=True))
                return 0
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
                return 1
    finally:
        telemetry_provider.force_flush(timeout_millis=10000)
        telemetry_provider.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
