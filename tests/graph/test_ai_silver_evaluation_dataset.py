import json
from pathlib import Path

import pytest

from src.graph.evaluation import load_evaluation_dataset


AI_SILVER_PATH = Path("test_data/phase3_ai_silver_eval_v1.json")


def test_ai_silver_dataset_loads_to_existing_evaluation_contract():
    dataset = load_evaluation_dataset(AI_SILVER_PATH)

    assert dataset.dataset_id == "phase3-ai-silver-development-evaluation"
    assert dataset.version == "v1.0-provisional"
    assert dataset.synthetic is True
    assert dataset.annotation_source == "AI-labelled provisional (silver) evaluation set"
    assert dataset.dataset_status == "AI_SILVER_LABELS_READY_FOR_INTERNAL_DEVELOPMENT_BENCHMARK"
    assert dataset.label_provenance == "AI_GENERATED_AND_AI_SOURCE_CHECKED"
    assert dataset.human_verification_status == "NOT_PERFORMED"
    assert dataset.ground_truth_status == "NOT_ESTABLISHED"
    assert len(dataset.queries) == 30
    assert sum(not query.unanswerable for query in dataset.queries) == 24
    assert sum(query.unanswerable for query in dataset.queries) == 6
    assert len(dataset.corpus_snapshot) == 6

    query = dataset.queries[0]
    source = json.loads(AI_SILVER_PATH.read_text(encoding="utf-8"))["queries"][0]
    evidence = query.evidence_references[0]
    source_reference = source["evidence_references"][0]
    assert query.relevant_chunk_ids == tuple(source["expected_relevant_chunk_ids"])
    assert query.expected_document_ids == tuple(source["expected_document_sha256s"])
    assert query.reference_answer == source["reference_answer"]
    assert evidence.document_sha256 == source_reference["document_sha256"]
    assert evidence.cited_page == source_reference["canonical_page"]
    assert evidence.source_page_numbers == tuple(source_reference["source_page_numbers"])
    assert evidence.quote == source_reference["exact_quote_from_raw_text"]
    assert evidence.chunk_ids == (source_reference["chunk_id"],)


def test_ai_silver_dataset_loader_keeps_existing_dataset_loader_compatible():
    dataset = load_evaluation_dataset("src/graph/evaluation/dev_v1.json")
    assert dataset.dataset_id == "legal-rag-phase3-dev"
    assert dataset.dataset_status is None
    assert dataset.human_verification_status is None


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload["queries"][0].update(answerability="MAYBE"),
         "Invalid answerability"),
        (lambda payload: payload["distribution"].update(query_count=29),
         "query count does not match"),
        (lambda payload: payload["queries"][0]["evidence_references"][0].update(
            ingestion_run_id="stale-run"), "does not match corpus snapshot"),
        (lambda payload: payload.update(human_verification_status="VERIFIED"),
         "remain explicitly provisional"),
    ],
)
def test_ai_silver_schema_rejects_inconsistent_or_ground_truth_claims(
    tmp_path: Path, mutate, message: str,
):
    payload = json.loads(AI_SILVER_PATH.read_text(encoding="utf-8"))
    mutate(payload)
    path = tmp_path / "invalid_ai_silver.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_evaluation_dataset(path)


def test_existing_evaluation_jsonl_is_preserved():
    # The repository's pre-existing annotation file is JSONL, not the similarly
    # named evaluation.json path mentioned in the request. The adapter does not
    # rewrite either file.
    assert Path("test_data/evaluation.jsonl").is_file()
    assert not Path("test_data/evaluation.json").exists()
