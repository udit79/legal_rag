import pytest
import json

from src.graph.assertion_contracts import (
    AssertionClassification,
    AssertionClassificationProposal,
    AssertionContext,
    applicability_overlaps,
    assertion_identity,
)


def test_assertion_identity_keeps_jurisdiction_scope_and_temporal_validity_distinct():
    base = dict(subject_id="org:1", predicate="OWNS", object_value="app:1")
    general = assertion_identity(**base, context=AssertionContext())
    scoped = assertion_identity(**base, context=AssertionContext(jurisdiction="IN", scope="IFMS"))
    timed = assertion_identity(**base, context=AssertionContext(valid_from="2024-04-10"))
    assert len({general, scoped, timed}) == 3


def test_assertion_context_rejects_invalid_temporal_range():
    with pytest.raises(ValueError, match="ends before"):
        AssertionContext(valid_from="2025-01-02", valid_until="2025-01-01")


def test_assertion_conflict_comparison_respects_jurisdiction_scope_and_time():
    existing = {
        "jurisdiction": "IN", "scope": "security audit",
        "valid_from": "2025-01-01", "valid_until": "2025-01-31",
    }
    assert applicability_overlaps(existing, AssertionContext(
        jurisdiction="IN", scope="security audit", valid_from="2025-01-15"
    ))
    assert not applicability_overlaps(existing, AssertionContext(
        jurisdiction="US", scope="security audit", valid_from="2025-01-15"
    ))
    assert not applicability_overlaps(existing, AssertionContext(
        jurisdiction="IN", scope="privacy audit", valid_from="2025-01-15"
    ))
    assert not applicability_overlaps(existing, AssertionContext(
        jurisdiction="IN", scope="security audit", valid_from="2025-02-01"
    ))


@pytest.mark.parametrize("classification", list(AssertionClassification))
def test_classification_is_a_sourced_non_adjudicative_proposal(classification):
    proposal = AssertionClassificationProposal(
        source_assertion_id="assertion:" + "a" * 64,
        target_assertion_id="assertion:" + "b" * 64,
        classification=classification,
        classifier="automated-evidence-classifier",
        rationale="Store this relation as a proposal only.",
        source_sha256s=("c" * 64,),
        ingestion_run_ids=("run-1",),
        evidence_references=(json.dumps({
            "source_sha256": "c" * 64,
            "ingestion_run_id": "run-1",
            "start_offset": 0,
            "end_offset": 17,
            "quote": "exact source text",
        }),),
    )
    assert proposal.classification is classification
    assert proposal.adjudication_status == "NOT_ADJUDICATED"


def test_assertion_classification_requires_distinct_claims_and_provenance():
    with pytest.raises(ValueError, match="cannot be classified against itself"):
        AssertionClassificationProposal(
            source_assertion_id="assertion:" + "a" * 64,
            target_assertion_id="assertion:" + "a" * 64,
            classification=AssertionClassification.UNRESOLVED,
            classifier="test", rationale="test", source_sha256s=("c" * 64,),
            ingestion_run_ids=("run-1",), evidence_references=(json.dumps({
                "source_sha256": "c" * 64,
                "ingestion_run_id": "run-1",
                "start_offset": 0,
                "end_offset": 17,
                "quote": "exact source text",
            }),),
        )
