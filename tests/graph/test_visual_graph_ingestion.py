from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

from src.graph import corpus_cli
from src.graph.visual_graph_ingestion import (
    process_visual_graph_documents,
    visual_graph_failure_statuses,
)
from tests.graph.test_visual_observation_writer import _Db
from tests.graph.test_visual_observations import _asset, _document


def test_visual_failure_is_run_scoped_and_does_not_block_following_document():
    first, _, _ = _document()
    second = replace(first, source_sha256="f" * 64, ingestion_run_id="run-no-visual", chunks=())

    class Extractor:
        provider = "openai"
        model = "extractor-v1"
        async def extract(self, description):
            raise RuntimeError("simulated model outage")

    class Resolver:
        async def resolve(self, run_id, keys):
            return {(2, "page-2-block-3"): _asset()}

    statuses = asyncio.run(process_visual_graph_documents(
        _Db(), (first, second), Extractor(), Resolver()
    ))
    assert [(item.ingestion_run_id, item.status) for item in statuses] == [
        (first.ingestion_run_id, "FAILED"), (second.ingestion_run_id, "NO_VISUAL_EVIDENCE")
    ]
    assert statuses[0].error_code == "VISUAL_GRAPH_PROCESSING_FAILED"
    assert statuses[0].error_type == "RuntimeError"


def test_setup_failure_is_recorded_for_each_ingestion_run():
    first, _, _ = _document()
    second = replace(first, source_sha256="f" * 64, ingestion_run_id="run-2")
    statuses = visual_graph_failure_statuses(
        (first, second), "VISUAL_GRAPH_CONFIGURATION_FAILED", ValueError("no API key")
    )
    assert len(statuses) == 2
    assert all(item.status == "FAILED" for item in statuses)
    assert [item.ingestion_run_id for item in statuses] == [first.ingestion_run_id, "run-2"]


def test_corpus_cli_triggers_visual_stage_only_after_structural_write(monkeypatch, tmp_path):
    document, _, _ = _document()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"source_sha256s": [document.source_sha256]}), encoding="utf-8")
    monkeypatch.setenv("SUPABASE_URL", "https://supabase.test")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-key")
    monkeypatch.setenv("NEO4J_TEST_DATABASE", "phase6_test")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "test")
    monkeypatch.setenv("NEO4J_PASSWORD", "test")
    events = []

    class Reader:
        def __init__(self, *args): pass
        async def read(self, hashes): return (document,)

    class HttpClient:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass

    class Db:
        database = "phase6_test"
        async def verify_connection(self): events.append("connected")
        async def close(self): events.append("closed")

    class Writer:
        async def write(self, db, documents):
            events.append("structural-write")
            return {"created_records": 0}

    class Extractor:
        def __init__(self, config): events.append("extractor-created")
        async def aclose(self): events.append("extractor-closed")

    class Config:
        @classmethod
        def from_env(cls): return object()

    monkeypatch.setattr(corpus_cli, "SupabaseCorpusReader", Reader)
    monkeypatch.setattr(corpus_cli, "SupabaseImageAssetResolver", lambda *args: object())
    monkeypatch.setattr(corpus_cli, "DbConnect", lambda **kwargs: Db())
    monkeypatch.setattr(corpus_cli, "StructuralGraphWriter", Writer)
    monkeypatch.setattr(corpus_cli, "initialize_graph_schema", _async_event(events, "structural-schema"))
    monkeypatch.setattr(corpus_cli, "initialize_semantic_schema", _async_event(events, "visual-schema"))
    monkeypatch.setattr(corpus_cli, "process_visual_graph_documents", _process_visual(events))
    monkeypatch.setattr(corpus_cli, "OpenAIVisualGraphExtractor", Extractor)
    monkeypatch.setattr(corpus_cli, "VisualGraphExtractionConfig", Config)
    monkeypatch.setattr(corpus_cli.httpx, "AsyncClient", HttpClient)

    code = asyncio.run(corpus_cli._execute(SimpleNamespace(
        manifest=manifest, apply=True, verify_idempotency=False, visual_knowledge=True,
    )))
    assert code == 0
    assert events.index("structural-write") < events.index("visual-schema") < events.index("visual-process")


def _async_event(events, event):
    async def run(*args):
        events.append(event)
    return run


def _process_visual(events):
    async def run(*args):
        events.append("visual-process")
        return ()
    return run
