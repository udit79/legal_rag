import hashlib
from pathlib import Path

import pytest

from src.graph.corpus_reader import CorpusReadError, validate_corpus_response

SOURCE = "a" * 64
RUN = "run-1"
POLICY = "5c-1.5.0+123456789012"
SCREENING_DIGEST = "c" * 64
CANONICAL_DIGEST = "d" * 64


def _row(decisions=("ADMITTED", "ADMITTED")):
    revised, mappings, screenings = [], [], []
    for index, decision in enumerate(decisions):
        raw = f"source passage {index}"
        digest = hashlib.sha256(raw.encode()).hexdigest()
        revised.append({"index": index, "text": raw, "retrieval_text": raw,
                        "retrieval_text_edits": [], "chunk_sha256": digest,
                        "sources": [{"page_number": 1, "block_id": f"page-1-block-{index}",
                                     "source_text_length": len(raw),
                                     "start_offset": 0, "end_offset": len(raw),
                                     "block_reading_order": 0, "block_type": "TEXT"}]})
        record = {"chunk_index": index, "chunk_digest": digest,
                  "decision": decision, "source_sha256": SOURCE,
                  "ingestion_run_id": RUN, "chunking_policy_version": POLICY,
                  "screening_policy_version": "screening-1", "screening_policy_digest": SCREENING_DIGEST}
        mappings.append({"revised_chunk_index": index,
                         "original_chunk_indices": [index],
                         "admission_records": [record]})
        screenings.append({"chunk_index": index, "chunk_digest": digest,
                           "decision": decision})
    return {"source_sha256": SOURCE, "canonical_document_digest": CANONICAL_DIGEST,
            "source_page_count": 3, "ingestion_run_id": RUN,
            "ingestion_created_at": "2026-11-12T10:00:00Z",
            "run_source_sha256": SOURCE, "chunking_policy_version": POLICY,
            "screening_policy_version": "screening-1", "screening_policy_digest": SCREENING_DIGEST,
            "document_quarantined": False, "cpcr_ingestion_run_id": RUN,
            "cpcr_source_sha256": SOURCE,
            "output_payload": {"source_sha256": SOURCE, "ingestion_run_id": RUN,
                               "chunking_policy_version": POLICY,
                               "revised_chunks": revised, "admission_mappings": mappings},
            "screening_results": screenings}


def test_reader_fails_closed_when_cpcr_maps_a_quarantined_chunk():
    with pytest.raises(CorpusReadError, match="quarantined or rejected"):
        validate_corpus_response((SOURCE,), [_row(("ADMITTED", "QUARANTINED"))])


def test_reader_validates_final_chunks_and_preserves_document_metadata():
    document, = validate_corpus_response((SOURCE,), [_row()])
    assert len(document.chunks) == 2
    assert document.chunks[0].raw_text == "source passage 0"
    assert document.chunks[0].retrieval_text == "source passage 0"
    assert document.canonical_document_digest == CANONICAL_DIGEST
    assert document.source_page_count == 3
    assert document.ingestion_created_at.isoformat() == "2026-11-12T10:00:00+00:00"


def test_reader_requires_timezone_aware_authoritative_run_timestamp():
    row = _row()
    row["ingestion_created_at"] = "2026-11-12T10:00:00"
    with pytest.raises(CorpusReadError, match="must include a timezone"):
        validate_corpus_response((SOURCE,), [row])


@pytest.mark.parametrize("timestamp", [None, "not-a-timestamp", "2026-11-12T10:00:00"])
def test_reader_rejects_missing_malformed_or_naive_timestamps(timestamp):
    row = _row()
    row["ingestion_created_at"] = timestamp
    with pytest.raises(CorpusReadError, match="ingestion_created_at"):
        validate_corpus_response((SOURCE,), [row])


def test_reader_allows_valid_contextual_mappings_to_reuse_source_chunks():
    row = _row()
    first_records = row["output_payload"]["admission_mappings"][0]["admission_records"]
    second_records = row["output_payload"]["admission_mappings"][1]["admission_records"]
    row["output_payload"]["admission_mappings"][1]["original_chunk_indices"] = [0, 1]
    row["output_payload"]["admission_mappings"][1]["admission_records"] = [
        first_records[0], second_records[0]
    ]
    documents, = validate_corpus_response((SOURCE,), [row])
    assert len(documents.chunks) == 2


def test_reader_preserves_and_validates_retrieval_edit_offsets():
    row = _row(("ADMITTED",))
    chunk = row["output_payload"]["revised_chunks"][0]
    chunk.update({
        "text": "Header\nBody", "retrieval_text": "Body",
        "chunk_sha256": hashlib.sha256(b"Header\nBody").hexdigest(),
        "retrieval_text_edits": [{"source_start": 0, "source_end": 7,
                                   "retrieval_start": 0, "retrieval_end": 0,
                                   "operation": "REPEATED_HEADER_REMOVED"}],
        "sources": [{"page_number": 1, "block_id": "page-1-block-0",
                     "source_text_length": 11, "start_offset": 0, "end_offset": 11,
                     "block_reading_order": 0, "block_type": "TEXT"}],
    })
    document, = validate_corpus_response((SOURCE,), [row])
    assert document.chunks[0].retrieval_text == "Body"
    assert document.chunks[0].retrieval_text_edits[0]["source_end"] == 7
    chunk["retrieval_text_edits"][0]["source_end"] = 8
    with pytest.raises(CorpusReadError, match="edit offsets"):
        validate_corpus_response((SOURCE,), [row])


@pytest.mark.parametrize("change", [
    {"cpcr_ingestion_run_id": "older-run"},
    {"cpcr_source_sha256": "b" * 64},
    {"run_source_sha256": "b" * 64},
    {"document_quarantined": True},
])
def test_reader_fails_closed_on_identity_or_quarantine(change):
    row = _row()
    row.update(change)
    with pytest.raises(CorpusReadError):
        validate_corpus_response((SOURCE,), [row])


def test_reader_fails_closed_when_latest_run_has_no_cpcr_or_screening_mapping():
    row = _row()
    row["output_payload"] = None
    with pytest.raises(CorpusReadError):
        validate_corpus_response((SOURCE,), [row])


def test_reader_rejects_empty_cpcr_structural_output():
    row = _row()
    row["output_payload"]["revised_chunks"] = []
    row["output_payload"]["admission_mappings"] = []
    with pytest.raises(CorpusReadError, match="no structural chunks"):
        validate_corpus_response((SOURCE,), [row])


@pytest.mark.parametrize("field", ["canonical_document_digest", "source_page_count"])
def test_reader_fails_closed_on_missing_document_metadata(field):
    row = _row()
    row.pop(field)
    with pytest.raises(CorpusReadError):
        validate_corpus_response((SOURCE,), [row])


def test_reader_fails_closed_when_admitted_original_chunk_is_not_in_cpcr_output():
    row = _row(("ADMITTED", "ADMITTED", "ADMITTED"))
    row["output_payload"]["revised_chunks"].pop()
    row["output_payload"]["admission_mappings"].pop()
    with pytest.raises(CorpusReadError, match="Admitted original chunks"):
        validate_corpus_response((SOURCE,), [row])


def test_reader_fails_closed_on_invalid_screen_decision_and_inconsistent_mapping():
    row = _row()
    row["screening_results"][0]["decision"] = "UNKNOWN"
    with pytest.raises(CorpusReadError, match="screening decision"):
        validate_corpus_response((SOURCE,), [row])
    row = _row()
    row["output_payload"]["admission_mappings"][0]["admission_records"][0]["decision"] = "REJECTED"
    with pytest.raises(CorpusReadError, match="does not match"):
        validate_corpus_response((SOURCE,), [row])


def test_reader_detects_payload_latest_run_mismatch():
    row = _row()
    row["output_payload"]["ingestion_run_id"] = "older-run"
    with pytest.raises(CorpusReadError, match="payload identity"):
        validate_corpus_response((SOURCE,), [row])
    row = _row()
    row["output_payload"]["admission_mappings"] = []
    with pytest.raises(CorpusReadError):
        validate_corpus_response((SOURCE,), [row])


def test_corpus_rpc_is_executable_only_by_service_role():
    migration = Path("supabase/migrations/202611120001_read_latest_ingestion_graph_corpus.sql")
    sql = migration.read_text(encoding="utf-8").lower()
    assert "security definer" in sql
    assert "set search_path = pg_catalog, public, pg_temp" in sql
    assert "revoke all on function public.read_latest_ingestion_graph_corpus(text[])" in sql
    assert "from public, anon, authenticated" in sql
    assert "grant execute on function public.read_latest_ingestion_graph_corpus(text[])\n    to service_role" in sql


def test_corpus_rpc_fails_closed_on_ambiguous_latest_timestamps():
    migration = Path("supabase/migrations/202611120001_read_latest_ingestion_graph_corpus.sql")
    sql = migration.read_text(encoding="utf-8").lower()
    assert "order by r.created_at desc nulls last\n        limit 1" in sql
    assert "r.created_at = latest_run.created_at" in sql
    assert "ambiguous latest ingestion timestamp" in sql
    assert "order by r.created_at desc, r.ingestion_run_id desc" not in sql
    assert "alter table public.ingestion_runs\n    alter column created_at set not null" in sql
    assert "do $migration$" not in sql
