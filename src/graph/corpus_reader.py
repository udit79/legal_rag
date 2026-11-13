from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import ValidationError

from src.ingestion.chunk_contracts import Chunk, ChunkSource, RetrievalTextEdit

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ADMITTED = {"ADMITTED", "ADMITTED_FLAGGED"}
_TRACER = trace.get_tracer("graph_rag.graph.corpus_reader")


@dataclass(frozen=True)
class CorpusChunk:
    id: str
    chunk_index: int
    original_chunk_indices: tuple[int, ...]
    raw_text: str
    retrieval_text: str
    chunk_sha256: str
    sources: list[dict[str, Any]]
    retrieval_text_edits: list[dict[str, Any]]
    admission_decisions: list[str]


@dataclass(frozen=True)
class CorpusDocument:
    source_sha256: str
    canonical_document_digest: str
    source_page_count: int
    ingestion_run_id: str
    ingestion_created_at: datetime
    chunking_policy_version: str
    screening_policy_version: str
    screening_policy_digest: str
    chunks: tuple[CorpusChunk, ...]


class CorpusReadError(ValueError):
    pass


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CorpusReadError(f"Missing or invalid {field}")
    return value


def _parse_ingestion_timestamp(value: Any) -> datetime:
    """Parse a Supabase timestamptz value and normalize it to UTC."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise CorpusReadError("Invalid ingestion_created_at timestamp") from exc
    else:
        raise CorpusReadError("Missing or invalid ingestion_created_at timestamp")

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CorpusReadError("ingestion_created_at must include a timezone")
    return parsed.astimezone(timezone.utc)


def validate_corpus_response(requested: tuple[str, ...], rows: Any) -> tuple[CorpusDocument, ...]:
    if not isinstance(rows, list):
        raise CorpusReadError("Supabase corpus RPC response must be a list")
    by_source: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise CorpusReadError("Supabase corpus row must be an object")
        source = _required_string(row.get("source_sha256"), "source_sha256")
        if source not in requested or source in by_source:
            raise CorpusReadError("Supabase returned an unexpected or duplicate source")
        by_source[source] = row
    if set(by_source) != set(requested):
        raise CorpusReadError("Supabase corpus RPC omitted a requested source")

    documents: list[CorpusDocument] = []
    for source in requested:
        row = by_source[source]
        canonical_digest = _required_string(
            row.get("canonical_document_digest"), "canonical_document_digest"
        )
        if not _SHA256.fullmatch(canonical_digest):
            raise CorpusReadError("Invalid canonical document digest")
        page_count = row.get("source_page_count")
        if type(page_count) is not int or page_count <= 0:
            raise CorpusReadError("Missing or invalid source page count")
        run_id = _required_string(row.get("ingestion_run_id"), "ingestion_run_id")
        ingestion_created_at = _parse_ingestion_timestamp(row.get("ingestion_created_at"))
        policy = _required_string(row.get("chunking_policy_version"), "chunking_policy_version")
        screening_policy = _required_string(row.get("screening_policy_version"), "screening_policy_version")
        screening_digest = _required_string(row.get("screening_policy_digest"), "screening_policy_digest")
        if not _SHA256.fullmatch(screening_digest):
            raise CorpusReadError("Invalid screening policy digest")
        if row.get("run_source_sha256") != source or row.get("cpcr_source_sha256") != source:
            raise CorpusReadError("Source hash differs across run and CPCR output")
        if row.get("cpcr_ingestion_run_id") != run_id:
            raise CorpusReadError("CPCR output does not match the latest ingestion run")
        if row.get("document_quarantined") is not False:
            raise CorpusReadError(f"Latest ingestion run is quarantined: {source}")
        payload = row.get("output_payload")
        screenings = row.get("screening_results")
        if not isinstance(payload, dict) or not isinstance(screenings, list):
            raise CorpusReadError("Latest run has no valid CPCR or screening payload")
        if payload.get("source_sha256") != source or payload.get("ingestion_run_id") != run_id:
            raise CorpusReadError("CPCR payload identity does not match its run")
        if payload.get("chunking_policy_version") != policy:
            raise CorpusReadError("CPCR policy version does not match its run")
        screen_by_index: dict[int, dict[str, Any]] = {}
        screening_decisions = {"ADMITTED", "ADMITTED_FLAGGED", "QUARANTINED", "REJECTED"}
        for result in screenings:
            if not isinstance(result, dict) or type(result.get("chunk_index")) is not int:
                raise CorpusReadError("Invalid latest-run screening result")
            if (result["chunk_index"] < 0 or not isinstance(result.get("decision"), str)
                    or result["decision"] not in screening_decisions):
                raise CorpusReadError("Invalid latest-run screening decision or index")
            if not isinstance(result.get("chunk_digest"), str) or not _SHA256.fullmatch(result["chunk_digest"]):
                raise CorpusReadError("Invalid latest-run screening digest")
            if result["chunk_index"] in screen_by_index:
                raise CorpusReadError("Duplicate screening chunk index")
            screen_by_index[result["chunk_index"]] = result

        revised = payload.get("revised_chunks")
        mappings = payload.get("admission_mappings")
        if not isinstance(revised, list) or not isinstance(mappings, list) or len(revised) != len(mappings):
            raise CorpusReadError("CPCR chunks and admission mappings are inconsistent")
        if not revised:
            raise CorpusReadError("Latest CPCR output contains no structural chunks")
        chunks: list[CorpusChunk] = []
        seen_original: set[int] = set()
        for final_index, (chunk, mapping) in enumerate(zip(revised, mappings, strict=True)):
            if not isinstance(chunk, dict) or not isinstance(mapping, dict):
                raise CorpusReadError("Invalid revised chunk mapping")
            indices = mapping.get("original_chunk_indices")
            records = mapping.get("admission_records")
            if (mapping.get("revised_chunk_index") != final_index or chunk.get("index") != final_index
                    or not isinstance(indices, list) or not indices
                    or any(type(index) is not int or index < 0 for index in indices)
                    or indices != sorted(set(indices)) or not isinstance(records, list)
                    or len(records) != len(indices)):
                raise CorpusReadError("Invalid source chunk indices or mapping")
            decisions: list[str] = []
            for original_index, record in zip(indices, records, strict=True):
                screened = screen_by_index.get(original_index)
                if not isinstance(record, dict) or screened is None:
                    raise CorpusReadError("Admission mapping lacks a matching screen result")
                if (record.get("chunk_index") != original_index
                        or record.get("source_sha256") != source
                        or record.get("ingestion_run_id") != run_id
                        or record.get("chunking_policy_version") != policy
                        or record.get("screening_policy_version") != screening_policy
                        or record.get("screening_policy_digest") != screening_digest
                        or record.get("chunk_digest") != screened.get("chunk_digest")
                        or record.get("decision") != screened.get("decision")):
                    raise CorpusReadError("Admission mapping does not match latest-run screening")
                decision = record.get("decision")
                if not isinstance(decision, str) or decision not in screening_decisions:
                    raise CorpusReadError("Invalid admission decision")
                decisions.append(decision)
            # Contextualized CPCR chunks may legitimately reuse an admitted source
            # index; the set tracks coverage, not one-to-one ownership.
            seen_original.update(indices)
            if any(decision not in _ADMITTED for decision in decisions):
                raise CorpusReadError(
                    "Final CPCR chunk maps to a quarantined or rejected source chunk"
                )
            raw = _required_string(chunk.get("text"), "chunk.text")
            retrieval = _required_string(chunk.get("retrieval_text"), "chunk.retrieval_text")
            digest = _required_string(chunk.get("chunk_sha256"), "chunk.chunk_sha256")
            if not _SHA256.fullmatch(digest) or hashlib.sha256(raw.encode("utf-8")).hexdigest() != digest:
                raise CorpusReadError("Final chunk SHA-256 does not match raw text")
            sources = chunk.get("sources")
            edits = chunk.get("retrieval_text_edits")
            if not isinstance(sources, list) or not sources or not isinstance(edits, list):
                raise CorpusReadError("Final chunk provenance or retrieval edits are invalid")
            try:
                typed_sources = tuple(ChunkSource.model_validate(item) for item in sources)
                typed_edits = tuple(RetrievalTextEdit.model_validate(item) for item in edits)
                validated_chunk = Chunk(
                    index=final_index, text=raw, retrieval_text=retrieval,
                    retrieval_text_edits=typed_edits, sources=typed_sources
                )
            except ValidationError as exc:
                raise CorpusReadError("Final chunk provenance or edit offsets are invalid") from exc
            identity = hashlib.sha256(
                (source + "\0" + policy + "\0" + ",".join(map(str, indices))).encode("utf-8")
            ).hexdigest()
            chunks.append(CorpusChunk(identity, final_index, tuple(indices), raw, retrieval, digest,
                                      [source.model_dump(mode="json") for source in validated_chunk.sources],
                                      [edit.model_dump(mode="json") for edit in validated_chunk.retrieval_text_edits],
                                      decisions))
        admitted_original = {
            index for index, result in screen_by_index.items()
            if result["decision"] in _ADMITTED
        }
        if not admitted_original.issubset(seen_original):
            missing = sorted(admitted_original - seen_original)
            raise CorpusReadError(
                f"Admitted original chunks are missing from CPCR output: {missing}"
            )
        documents.append(
            CorpusDocument(
                source_sha256=source,
                canonical_document_digest=canonical_digest,
                source_page_count=page_count,
                ingestion_run_id=run_id,
                ingestion_created_at=ingestion_created_at,
                chunking_policy_version=policy,
                screening_policy_version=screening_policy,
                screening_policy_digest=screening_digest,
                chunks=tuple(chunks),
            )
        )
    return tuple(documents)


class SupabaseCorpusReader:
    def __init__(self, base_url: str, service_role_key: str, client: httpx.AsyncClient):
        if not base_url.startswith(("https://", "http://")) or not service_role_key.strip():
            raise ValueError("Supabase URL and service role key are required")
        self._url = base_url.rstrip("/")
        self._key = service_role_key.strip()
        self._client = client

    async def read(self, source_hashes: list[str]) -> tuple[CorpusDocument, ...]:
        if not source_hashes or len(set(source_hashes)) != len(source_hashes):
            raise ValueError("Provide a non-empty list of unique source hashes")
        if any(not isinstance(value, str) or not _SHA256.fullmatch(value) for value in source_hashes):
            raise ValueError("Every source hash must be a lowercase SHA-256")
        with _TRACER.start_as_current_span("graph.supabase.read_corpus") as span:
            span.set_attribute("db.system", "supabase")
            span.set_attribute("db.operation.name", "read_latest_ingestion_graph_corpus")
            span.set_attribute("graph.corpus.requested_documents", len(source_hashes))
            try:
                response = await self._client.post(
                    f"{self._url}/rest/v1/rpc/read_latest_ingestion_graph_corpus",
                    headers={"apikey": self._key, "Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
                    json={"p_source_sha256s": source_hashes},
                )
                span.set_attribute("http.response.status_code", response.status_code)
                response.raise_for_status()
                documents = validate_corpus_response(tuple(source_hashes), response.json())
                span.set_attribute("graph.corpus.returned_documents", len(documents))
                span.set_attribute(
                    "graph.corpus.admitted_chunks",
                    sum(len(document.chunks) for document in documents),
                )
                span.set_attribute(
                    "graph.corpus.ingestion_run_ids",
                    [document.ingestion_run_id for document in documents],
                )
                return documents
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                raise
