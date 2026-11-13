from __future__ import annotations
import argparse
import asyncio
import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from src.graph.corpus_reader import SupabaseCorpusReader
from src.graph.schema_bootstrap import initialize_graph_schema
from src.graph.structural_writer import StructuralGraphWriter, validate_structural_document
from src.infrastructure.neo4j_client import create_db_connect
from src.observability.telemetry import configure_telemetry

_TRACER = trace.get_tracer("graph_rag.graph.corpus_cli")


def _manifest(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    hashes = data.get("source_sha256s") if isinstance(data, dict) else None
    if not isinstance(hashes, list) or not hashes:
        raise ValueError("Manifest must contain a non-empty source_sha256s list")
    return hashes


async def _execute(args) -> int:
    load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
    with _TRACER.start_as_current_span("graph.corpus.load_manifest") as span:
        span.set_attribute("graph.manifest.path", str(args.manifest))
        hashes = _manifest(args.manifest)
        span.set_attribute("graph.manifest.document_count", len(hashes))
    async with httpx.AsyncClient(timeout=60.0) as client:
        documents = await SupabaseCorpusReader(url, key, client).read(hashes)
    print("mode:", "apply" if args.apply else "dry-run")
    for document in documents:
        with _TRACER.start_as_current_span("graph.corpus.validate_document") as span:
            span.set_attribute("graph.document.source_sha256", document.source_sha256)
            span.set_attribute("ingestion.run_id", document.ingestion_run_id)
            span.set_attribute("graph.chunk.count", len(document.chunks))
            try:
                validate_structural_document(document)
                span.set_attribute("graph.document.validation", "passed")
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                raise
        print(json.dumps({"source_sha256": document.source_sha256,
                          "ingestion_run_id": document.ingestion_run_id,
                          "status": "validated", "documents": 1,
                          "chunks": len(document.chunks),
                          "has_chunk": len(document.chunks),
                          "next_chunk": sum(
                              left.original_chunk_indices[-1] + 1 == right.original_chunk_indices[0]
                              for left, right in zip(document.chunks, document.chunks[1:]))}))
    if not args.apply:
        print("No Neo4j writes performed; pass --apply to persist.")
        return 0
    db = create_db_connect()
    try:
        await db.verify_connection()
        await initialize_graph_schema(db)
        writer = StructuralGraphWriter()
        counts = await writer.write(db, documents)
        print("write_counts:", json.dumps(counts, sort_keys=True))
        if args.verify_idempotency:
            with _TRACER.start_as_current_span("graph.neo4j.verify_idempotency") as span:
                span.set_attribute("graph.document.count", len(documents))
                try:
                    rerun = await writer.write(db, documents)
                    span.set_attribute("graph.write.created_records", rerun["created_records"])
                    if rerun["created_records"] != 0:
                        raise RuntimeError("Idempotency check created new graph records")
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                    raise
            print("idempotency_counts:", json.dumps(rerun, sort_keys=True))
    finally:
        await db.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("src/graph/manifests/legal_rag_phase2a.json"))
    parser.add_argument("--apply", action="store_true", help="Write validated corpus to Neo4j")
    parser.add_argument("--verify-idempotency", action="store_true",
                        help="After --apply, repeat write and require zero created records")
    args = parser.parse_args()
    if args.verify_idempotency and not args.apply:
        parser.error("--verify-idempotency requires --apply")
    provider, tracer = configure_telemetry()
    try:
        with tracer.start_as_current_span("graph.corpus.import") as span:
            span.set_attribute("service.component", "phase2a_corpus_cli")
            span.set_attribute("graph.corpus.mode", "apply" if args.apply else "dry_run")
            span.set_attribute("graph.manifest.path", str(args.manifest))
            try:
                return asyncio.run(_execute(args))
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                raise
    finally:
        provider.force_flush(timeout_millis=10000)
        provider.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
