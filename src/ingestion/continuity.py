import re
from src.ingestion.chunk_contracts import Chunk, ChunkedDocument
from src.ingestion.continuity_contracts import ChunkReference,ContinuityCandidate,deterministic_candidate_id,validate_candidate_against_document

_FOOTNOTE_PREFIX = re.compile(
    r"(?is)^\s*\d{1,3}\s+(?:"
    r"for short\b|in short\b|hereinafter\b|see\b|"
    r"\(?\d{4}\)?\s+\d+\s+[A-Z]|"
    r"\(\d{4}\)\s+\d+\s+[A-Z])"
)
_NUMBER_ONLY = re.compile(r"^\d{1,4}$")
_COVER_MARKER = re.compile(
    r"(?i)\b(?:reportable|civil appeal no\.|appellant\(s\)|"
    r"respondent\(s\)|versus|in the supreme court)\b"
)

def _is_cover_matter(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    lowercase_ratio = (
        sum(char.islower() for char in letters) / len(letters) if letters else 0.0
    )
    return bool(_COVER_MARKER.search(text)) and lowercase_ratio < 0.4

def _content_role(chunk: Chunk, page_number: int) -> str | None:
    """Identify chunk roles used for continuity, excluding obvious footnotes.

    Role inference is intentionally narrow because the chunk schema carries
    block type and reading order but no semantic content-role annotation.
    """
    block_types = {source.block_type for source in chunk.sources}
    if block_types == {"TABLE"}:
        return "TABLE"
    if not block_types <= {"TEXT", "OTHER"}:
        return None

    compact = chunk.text.strip()
    if _NUMBER_ONLY.fullmatch(compact) and int(compact) == page_number:
        return None
    if _FOOTNOTE_PREFIX.match(chunk.text):
        return None
    if _is_cover_matter(chunk.text):
        return None
    return "TEXT"

def _role_boundary_chunks(
    chunks_by_page: dict[int, list[Chunk]],
    page_number: int,
    role: str,
    excluded_chunk_indices: frozenset[int],
) -> tuple[Chunk, ...]:
    return tuple(
        chunk
        for chunk in chunks_by_page.get(page_number, ())
        if chunk.index not in excluded_chunk_indices
        and _content_role(chunk, page_number) == role
    )

def _has_excluded_role_barrier(
    chunks_by_page: dict[int, list[Chunk]],
    page_number: int,
    role: str,
    excluded_chunk_indices: frozenset[int],
) -> bool:
    for chunk in chunks_by_page.get(page_number, ()):
        if chunk.index not in excluded_chunk_indices:
            continue
        block_types = {source.block_type for source in chunk.sources}
        if role == "TABLE" and block_types == {"TABLE"}:
            return True
        if role == "TEXT" and block_types <= {"TEXT", "OTHER"}:
            return True
    return False

def _candidate(
    document: ChunkedDocument,
    source_chunk: Chunk,
    target_chunk: Chunk,
    candidate_policy_version: str,
) -> ContinuityCandidate:
    source_page = source_chunk.sources[0].page_number
    target_page = target_chunk.sources[0].page_number
    candidate = ContinuityCandidate(
        candidate_id=deterministic_candidate_id(
            document.document_id,
            document.policy_version,
            source_chunk.index,
            target_chunk.index,
        ),
        source=ChunkReference(
            document_id=document.document_id,
            chunk_index=source_chunk.index,
            page_number=source_page,
        ),
        target=ChunkReference(
            document_id=document.document_id,
            chunk_index=target_chunk.index,
            page_number=target_page,
        ),
        chunking_policy_version=document.policy_version,
        candidate_policy_version=candidate_policy_version,
    )
    validate_candidate_against_document(candidate, document)
    return candidate

def generate_continuity_candidates(
    document: ChunkedDocument,
    candidate_policy_version: str,
    *,
    excluded_chunk_indices: frozenset[int] = frozenset(),
) -> tuple[ContinuityCandidate, ...]:
    """Generate independent text-flow and table-flow candidates per boundary.

    Obvious printed page numbers and conventionally numbered footnotes are not
    treated as main text. For each flow, the last source-side and first target-
    side chunk are selected by canonical reading order (chunk index).
    """
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
            raise ValueError(f"Chunk {chunk.index} references an invalid page number")
        chunks_by_page.setdefault(page_number, []).append(chunk)

    candidates: list[ContinuityCandidate] = []
    for page_number in range(1, document.source_page_count):
        boundary_candidates: list[ContinuityCandidate] = []
        for role in ("TEXT", "TABLE"):
            if _has_excluded_role_barrier(
                chunks_by_page, page_number, role, excluded_chunk_indices
            ) or _has_excluded_role_barrier(
                chunks_by_page, page_number + 1, role, excluded_chunk_indices
            ):
                continue
            source_chunks = _role_boundary_chunks(
                chunks_by_page,
                page_number,
                role,
                excluded_chunk_indices,
            )
            target_chunks = _role_boundary_chunks(
                chunks_by_page,
                page_number + 1,
                role,
                excluded_chunk_indices,
            )
            if not source_chunks or not target_chunks:
                continue
            candidate = _candidate(
                document,
                max(source_chunks, key=lambda chunk: chunk.index),
                min(target_chunks, key=lambda chunk: chunk.index),
                candidate_policy_version,
            )
            if (
                candidate.source.chunk_index not in excluded_chunk_indices
                and candidate.target.chunk_index not in excluded_chunk_indices
            ):
                boundary_candidates.append(candidate)
        candidates.extend(
            sorted(
                boundary_candidates,
                key=lambda item: (item.source.chunk_index, item.target.chunk_index),
            )
        )

    return tuple(candidates)
