from __future__ import annotations
import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Literal, TypeAlias
from src.ingestion.chunk_contracts import Chunk, RetrievalTextEdit
from src.ingestion.parser_contracts import CanonicalPage

RETRIEVAL_TEXT_POLICY_VERSION = "retrieval-text-1.0.0"
_LINE_HYPHENATION = re.compile(
    r"(?P<left>[A-Za-z]+)-(?P<newline>\r?\n)(?P<right>[a-z]+)"
)
# Prefixes that form real hyphenated legal terms ("non-obstante", "inter-state").
_COMPOUND_PREFIXES = frozenset(
    {
        "non",
        "inter",
        "intra",
        "anti",
        "quasi",
        "semi",
        "self",
        "ex",
        "pro",
        "co",
        "counter",
        "cross",
        "multi",
        "post",
        "sub",
        "re",
        "pre",
    }
)


def _document_vocabulary(pages: tuple[CanonicalPage, ...]) -> frozenset[str]:
    """Words that appear unbroken in the document (not split across a line)."""
    words: set[str] = set()
    for page in pages:
        for block in page.blocks:
            if block.block_type in {"TEXT", "OTHER"}:
                cleaned = _LINE_HYPHENATION.sub(" ", block.text)
                words.update(re.findall(r"[a-z]+", cleaned.casefold()))
    return frozenset(words)


_NormalizationOperation: TypeAlias = Literal[
    "REPEATED_HEADER_REMOVED",
    "REPEATED_FOOTER_REMOVED",
    "LINE_HYPHENATION_REPAIRED",
]

def _normalized_line(line: str) -> str:
    """Casefold, collapse whitespace and mask digits so "Page 5" == "Page 6"."""
    return re.sub(r"\d+", "#", " ".join(line.casefold().split()))

def _eligible_boundary_line(line: str) -> bool:
    normalized = _normalized_line(line)
    return (
        8 <= len(normalized) <= 120
        and any(char.isalpha() for char in normalized)
        and not normalized.replace(" ", "").replace("#", "").isdecimal()
        and sum(char.isalpha() for char in normalized) >= 4
    )

def _page_boundary_lines(
    pages: tuple[CanonicalPage, ...],
) -> dict[int, tuple[str | None, str | None]]:
    first_pages: dict[str, set[int]] = defaultdict(set)
    last_pages: dict[str, set[int]] = defaultdict(set)
    boundary_by_page: dict[int, tuple[str | None, str | None]] = {}

    for page in pages:
        lines = [
            line.strip()
            for block in page.blocks
            if block.block_type in {"TEXT", "OTHER"}
            for line in block.text.splitlines()
            if line.strip()
        ]
        first = lines[0] if lines else None
        last = lines[-1] if lines else None
        boundary_by_page[page.page_number] = (
            first if first and _eligible_boundary_line(first) else None,
            last if last and _eligible_boundary_line(last) else None,
        )
        if first_boundary := boundary_by_page[page.page_number][0]:
            first_pages[_normalized_line(first_boundary)].add(page.page_number)
        if last_boundary := boundary_by_page[page.page_number][1]:
            last_pages[_normalized_line(last_boundary)].add(page.page_number)

    return {
        page_number: (
            first
            if first and len(first_pages[_normalized_line(first)]) >= 3
            else None,
            last
            if last and len(last_pages[_normalized_line(last)]) >= 3
            else None,
        )
        for page_number, (first, last) in boundary_by_page.items()
    }

def _line_ranges(text: str) -> list[tuple[int, int, str]]:
    ranges: list[tuple[int, int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        end = offset + len(line)
        ranges.append((offset, end, line.rstrip("\r\n")))
        offset = end
    if offset < len(text):
        ranges.append((offset, len(text), text[offset:]))
    return ranges

def _normalization_edits(
    chunk: Chunk,
    page_boundaries: dict[int, tuple[str | None, str | None]],
    vocabulary: frozenset[str] = frozenset(),
) -> tuple[tuple[int, int, _NormalizationOperation], ...]:
    edits: list[tuple[int, int, _NormalizationOperation]] = []
    pages = {source.page_number for source in chunk.sources}

    if len(pages) == 1:
        page_number = next(iter(pages))
        header, footer = page_boundaries.get(page_number, (None, None))
        lines = _line_ranges(chunk.text)
        nonblank = [item for item in lines if item[2].strip()]
        if nonblank:
            first_start, first_end, first_text = nonblank[0]
            if header and _normalized_line(first_text) == _normalized_line(header):
                edits.append((first_start, first_end, "REPEATED_HEADER_REMOVED"))
            last_start, last_end, last_text = nonblank[-1]
            if (
                footer
                and _normalized_line(last_text) == _normalized_line(footer)
                and (last_start, last_end)
                != (first_start, first_end)
            ):
                edits.append((last_start, last_end, "REPEATED_FOOTER_REMOVED"))

    elif len(pages) > 1:
        # Context-prefixed chunks span pages: remove each page's own repeated
        # header/footer wherever it stands alone on its line segment.
        for page_number in sorted(pages):
            header, footer = page_boundaries.get(page_number, (None, None))
            boundary_lines: tuple[
                tuple[str | None, _NormalizationOperation], ...
            ] = (
                (header, "REPEATED_HEADER_REMOVED"),
                (footer, "REPEATED_FOOTER_REMOVED"),
            )
            for raw_line, operation in boundary_lines:
                if not raw_line:
                    continue
                search_from = 0
                while (start := chunk.text.find(raw_line, search_from)) >= 0:
                    end = start + len(raw_line)
                    search_from = max(end, start + 1)
                    before_ok = start == 0 or chunk.text[start - 1] in "\r\n"
                    after_ok = end == len(chunk.text) or chunk.text[end] in "\r\n"
                    if before_ok and after_ok and all(
                        end <= s0 or start >= e0 for s0, e0, _ in edits
                    ):
                        edits.append((start, end, operation))

    block_types = {source.block_type for source in chunk.sources}
    if block_types and block_types <= {"TEXT", "OTHER"}:
        for match in _LINE_HYPHENATION.finditer(chunk.text):
            left, right = match.group("left").casefold(), match.group("right")
            if (left + right) not in vocabulary and left in _COMPOUND_PREFIXES:
                # Real compound ("non-obstante"): keep the hyphen, drop the break.
                start, end = match.start("newline"), match.end("newline")
            else:
                start, end = match.end("left"), match.end("newline")
            if all(end <= s0 or start >= e0 for s0, e0, _ in edits):
                edits.append((start, end, "LINE_HYPHENATION_REPAIRED"))
    return tuple(sorted(edits, key=lambda edit: (edit[0], edit[1])))

def _derive_text(
    source_text: str,
    edits: Iterable[tuple[int, int, _NormalizationOperation]],
) -> tuple[str, tuple[RetrievalTextEdit, ...]]:
    output: list[str] = []
    mapped_edits: list[RetrievalTextEdit] = []
    source_cursor = 0
    retrieval_cursor = 0
    for start, end, operation in edits:
        if start < source_cursor or end > len(source_text):
            raise ValueError("Text normalization edits overlap or exceed chunk bounds")
        unchanged = source_text[source_cursor:start]
        output.append(unchanged)
        retrieval_cursor += len(unchanged)
        mapped_edits.append(
            RetrievalTextEdit(
                source_start=start,
                source_end=end,
                retrieval_start=retrieval_cursor,
                retrieval_end=retrieval_cursor,
                operation=operation,
            )
        )
        source_cursor = end
    output.append(source_text[source_cursor:])
    derived = "".join(output)
    if not derived.strip():
        return source_text, ()
    return derived, tuple(mapped_edits)

def normalize_retrieval_chunks(
    pages: tuple[CanonicalPage, ...],
    chunks: tuple[Chunk, ...],
) -> tuple[Chunk, ...]:
    """Derive search text while preserving source text and reversible offsets.

    Repeated exact boundary lines are removed only when they recur on at least
    three pages. Hyphenated line breaks are joined in text-only chunks.
    """
    page_boundaries = _page_boundary_lines(pages)
    vocabulary = _document_vocabulary(pages)
    normalized = []
    for chunk in chunks:
        retrieval_text, edits = _derive_text(
            chunk.text, _normalization_edits(chunk, page_boundaries, vocabulary)
        )
        normalized.append(
            Chunk(
                index=chunk.index,
                text=chunk.text,
                sources=chunk.sources,
                retrieval_text=retrieval_text,
                retrieval_text_edits=edits,
            )
        )
    return tuple(normalized)


def furniture_block_ids(pages: tuple[CanonicalPage, ...]) -> frozenset[str]:
    """Blocks whose whole text is a detected repeating page header or footer."""
    boundaries = _page_boundary_lines(pages)
    ids: set[str] = set()
    for page in pages:
        header, footer = boundaries.get(page.page_number, (None, None))
        targets = {_normalized_line(line) for line in (header, footer) if line}
        if not targets:
            continue
        for block in page.blocks:
            if block.block_type in {"TEXT", "OTHER"} and (
                _normalized_line(block.text) in targets
            ):
                ids.add(block.block_id)
    return frozenset(ids)
