"""Structural candidate generation for adjacent-page continuity."""

from src.ingestion.chunk_contracts import Chunk, ChunkedDocument
from src.ingestion.continuity_contracts import (
    ChunkReference,
    ContinuityCandidate,
    deterministic_candidate_id,
    validate_candidate_against_document,
)


def generate_continuity_candidates(
    document: ChunkedDocument,
    candidate_policy_version: str,
) -> tuple[ContinuityCandidate, ...]:
    """Generate one candidate for each populated adjacent physical-page pair."""
    if (
        not isinstance(candidate_policy_version, str)
        or not candidate_policy_version
        or candidate_policy_version.strip() != candidate_policy_version
    ):
        raise ValueError(
            "candidate_policy_version must be non-empty without surrounding whitespace"
        )

    chunks_by_page: dict[int, list[Chunk]] = {}
    seen_chunk_indices: set[int] = set()
    for chunk in document.chunks:
        if chunk.index in seen_chunk_indices:
            raise ValueError(f"Duplicate chunk index detected: {chunk.index}")
        seen_chunk_indices.add(chunk.index)
        if not chunk.sources:
            raise ValueError(f"Chunk {chunk.index} has no source provenance")
        pages = {source.page_number for source in chunk.sources}
        if len(pages) != 1:
            raise ValueError(f"Chunk {chunk.index} has cross-page provenance")
        page_number = next(iter(pages))
        if not 1 <= page_number <= document.source_page_count:
            raise ValueError(
                f"Chunk {chunk.index} references an invalid page number"
            )
        chunks_by_page.setdefault(page_number, []).append(chunk)

    candidates: list[ContinuityCandidate] = []
    for page_number in range(1, document.source_page_count):
        source_chunks = chunks_by_page.get(page_number, [])
        target_chunks = chunks_by_page.get(page_number + 1, [])
        if not source_chunks or not target_chunks:
            continue

        source_chunk = max(source_chunks, key=lambda chunk: chunk.index)
        target_chunk = min(target_chunks, key=lambda chunk: chunk.index)
        source = ChunkReference(
            document_id=document.document_id,
            chunk_index=source_chunk.index,
            page_number=page_number,
        )
        target = ChunkReference(
            document_id=document.document_id,
            chunk_index=target_chunk.index,
            page_number=page_number + 1,
        )
        candidate = ContinuityCandidate(
            candidate_id=deterministic_candidate_id(
                document.document_id,
                document.policy_version,
                source_chunk.index,
                target_chunk.index,
            ),
            source=source,
            target=target,
            chunking_policy_version=document.policy_version,
            candidate_policy_version=candidate_policy_version,
        )
        validate_candidate_against_document(candidate, document)
        candidates.append(candidate)

    return tuple(candidates)
