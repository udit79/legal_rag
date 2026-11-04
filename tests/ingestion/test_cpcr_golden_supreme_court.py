"""Opt-in live CPCR run against the Supreme Court golden canonical fixture.

Run with ``$env:CPCR_LIVE_CDM='1'; uv run pytest -q
tests/ingestion/test_cpcr_golden_supreme_court.py``. This test makes one
ScaleDown request for each candidate deferred by the deterministic gateway.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import Counter
from pathlib import Path

import httpx
import pytest
from dotenv import load_dotenv

from src.ingestion.chunking import ChunkingConfig, chunk_document
from src.ingestion.continuity import generate_continuity_candidates
from src.ingestion.continuity_gateway import (
    CONTINUITY_GATEWAY_POLICY_VERSION,
    GatewayDecision,
    GatewayResult,
    evaluate_candidate,
)
from src.ingestion.continuity_graph import build_cpcr_graph
from src.ingestion.continuity_state import CPCRState
from src.ingestion.parser_contracts import CanonicalDocument
from src.ingestion.scaledown_client import ScaleDownClient

FIXTURE_DIR = (
    Path(__file__).parents[1]
    / "fixtures"
    / "ingestion"
    / "supreme_court_2026_insc_60"
)
DOCUMENT_ID = "supreme_court_2026_insc_60"
OUTPUT_DIR = Path(__file__).parents[2] / "tmp" / DOCUMENT_ID


@pytest.mark.live_cdm
def test_supreme_court_canonical_runs_through_live_cpcr_pipeline() -> None:
    if os.environ.get("CPCR_LIVE_CDM") != "1":
        pytest.skip("Set CPCR_LIVE_CDM=1 to make live ScaleDown API calls")

    load_dotenv()
    if not os.environ.get("SCALEDOWN_API_KEY", "").strip():
        pytest.fail("SCALEDOWN_API_KEY is missing from the environment and .env")

    canonical_path = FIXTURE_DIR / "canonical.json"
    canonical_json = canonical_path.read_text(encoding="utf-8")
    canonical = CanonicalDocument.model_validate_json(canonical_json)
    config = ChunkingConfig()
    chunking_result = chunk_document(canonical, DOCUMENT_ID, config)
    document = chunking_result.document
    before_pipeline = document.model_dump_json()
    candidates = generate_continuity_candidates(document, "cpcr-candidate-1")
    decisions = [evaluate_candidate(candidate, document) for candidate in candidates]
    chunks_by_index = {chunk.index: chunk for chunk in document.chunks}
    deferred_candidate_ids = [
        candidate.candidate_id
        for candidate, decision in zip(candidates, decisions, strict=True)
        if decision.decision == GatewayDecision.DEFER
    ]
    actual_call_ids: list[str] = []

    class CountingLiveClient:
        """Record candidate IDs while forwarding to the real HTTP client."""

        def __init__(self, delegate: ScaleDownClient) -> None:
            self.delegate = delegate

        async def classify(self, request):
            actual_call_ids.append(request.candidate_id)
            return await self.delegate.classify(request)

    async def run_graph() -> list[dict[str, object]]:
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            client = ScaleDownClient(http_client=http_client)
            graph = build_cpcr_graph(CountingLiveClient(client))
            final_states = []
            for candidate, decision in zip(candidates, decisions, strict=True):
                gateway_result = GatewayResult(
                    decision=decision.decision,
                    relation_type=decision.relation_type,
                    evidence_codes=decision.evidence_codes,
                    gateway_policy_version=CONTINUITY_GATEWAY_POLICY_VERSION,
                )
                state: CPCRState = {
                    "candidate": candidate,
                    "source_text": chunks_by_index[candidate.source.chunk_index].text,
                    "target_text": chunks_by_index[candidate.target.chunk_index].text,
                    "gateway_result": gateway_result,
                    "cdm_result": None,
                }
                final_states.append(await graph.ainvoke(state))
            return final_states

    final_states = asyncio.run(run_graph())
    assert canonical.page_count == 20
    assert document.source_page_count == canonical.page_count
    assert len(candidates) == canonical.page_count - 1
    assert [
        (candidate.source.page_number, candidate.target.page_number)
        for candidate in candidates
    ] == [(page, page + 1) for page in range(1, canonical.page_count)]
    assert len({candidate.candidate_id for candidate in candidates}) == len(candidates)
    assert len(final_states) == len(candidates)
    assert Counter(
        state["gateway_result"].decision.value for state in final_states
    ) == Counter(decision.decision.value for decision in decisions)

    assert actual_call_ids == deferred_candidate_ids
    for state in final_states:
        decision = state["gateway_result"].decision
        result = state["cdm_result"]
        if decision == GatewayDecision.DEFER:
            assert result is not None
            assert result.candidate_id == state["candidate"].candidate_id
            assert result.model_id == "classify-1"
        else:
            assert result is None

    assert document.model_dump_json() == before_pipeline
    assert all(len(chunk.text) <= config.max_chunk_chars for chunk in document.chunks)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "canonical.json").write_text(canonical_json, encoding="utf-8")
    (OUTPUT_DIR / "chunking-result.json").write_text(
        chunking_result.model_dump_json(indent=2), encoding="utf-8"
    )
    cpcr_records = [
        {
            "candidate": candidate.model_dump(mode="json"),
            "gateway_result": GatewayResult(
                decision=decision.decision,
                relation_type=decision.relation_type,
                evidence_codes=decision.evidence_codes,
                gateway_policy_version=CONTINUITY_GATEWAY_POLICY_VERSION,
            ).model_dump(mode="json"),
            "cdm_result": state["cdm_result"].model_dump(mode="json")
            if state["cdm_result"] is not None
            else None,
        }
        for candidate, decision, state in zip(
            candidates, decisions, final_states, strict=True
        )
    ]
    (OUTPUT_DIR / "cpcr.json").write_text(
        json.dumps(cpcr_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
