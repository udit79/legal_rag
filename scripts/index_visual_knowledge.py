"""Extract and optionally persist unverified image-derived graph candidates.

Safe by default: reads the latest Supabase corpus and private image metadata,
then previews model output. --apply writes only to NEO4J_TEST_DATABASE.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.graph.corpus_reader import SupabaseCorpusReader
from src.graph.schema_bootstrap import initialize_graph_schema
from src.graph.semantic_schema import initialize_semantic_schema
from src.graph.visual_observations import (
    OpenAIVisualGraphExtractor,
    VisualGraphExtractionConfig,
    extract_document_visual_observations,
)
from src.graph.visual_observation_writer import persist_visual_observations
from src.infrastructure.neo4j_client import DbConnect
from src.ingestion.provenance.image_asset_resolver import SupabaseImageAssetResolver


def _hashes(values: list[str]) -> list[str]:
    if not values or any(len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
                         for value in values) or len(values) != len(set(values)):
        raise ValueError("Provide one or more unique lowercase source SHA-256 values")
    return values


def _test_db_config() -> DbConnect:
    database = os.getenv("NEO4J_TEST_DATABASE", "").strip()
    if not database or "test" not in database.casefold() or database.casefold() in {"neo4j", "rag"}:
        raise RuntimeError("Set NEO4J_TEST_DATABASE to an isolated database name containing 'test'")
    required = {
        "NEO4J_URI": os.getenv("NEO4J_URI"),
        "NEO4J_USERNAME": os.getenv("NEO4J_USERNAME"),
        "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD"),
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise RuntimeError("Missing Neo4j configuration: " + ", ".join(missing))
    return DbConnect(
        url=required["NEO4J_URI"], username=required["NEO4J_USERNAME"],
        password=required["NEO4J_PASSWORD"], database=database,
    )


async def run(source_hashes: list[str], apply: bool) -> dict:
    load_dotenv()
    hashes = _hashes(source_hashes)
    if apply:
        target = os.getenv("NEO4J_TEST_DATABASE", "").strip().casefold()
        if not target or "test" not in target or target in {"neo4j", "rag"}:
            raise RuntimeError("Set NEO4J_TEST_DATABASE to an isolated database name containing 'test'")
        if any(not os.getenv(key) for key in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")):
            raise RuntimeError("Set NEO4J_URI, NEO4J_USERNAME and NEO4J_PASSWORD for the isolated Neo4j target")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not supabase_key:
        raise RuntimeError("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY for read-only corpus access")
    config = VisualGraphExtractionConfig.from_env()
    async with httpx.AsyncClient(timeout=60.0) as client:
        documents = await SupabaseCorpusReader(supabase_url, supabase_key, client).read(hashes)
        resolver = SupabaseImageAssetResolver(supabase_url, supabase_key, client)
        extractor = OpenAIVisualGraphExtractor(config)
        try:
            observations = []
            for document in documents:
                observations.extend(await extract_document_visual_observations(
                    extractor, document, resolver
                ))
        finally:
            await extractor.aclose()
    summary = {
        "mode": "apply" if apply else "dry-run",
        "database": os.getenv("NEO4J_TEST_DATABASE") if apply else None,
        "documents": len(documents),
        "visual_observations": len(observations),
        "candidate_entities": sum(len(item.candidates) for item in observations),
        "candidate_relationships": sum(len(item.relationships) for item in observations),
        "observations": [item.model_dump(mode="json") for item in observations],
    }
    if apply:
        db = _test_db_config()
        try:
            await db.verify_connection()
            await initialize_graph_schema(db)
            await initialize_semantic_schema(db)
            summary["write_counts"] = await persist_visual_observations(db, tuple(observations))
        finally:
            await db.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_sha256", nargs="+", help="Source documents to process")
    parser.add_argument("--apply", action="store_true", help="Persist into isolated NEO4J_TEST_DATABASE")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.source_sha256, args.apply)), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
