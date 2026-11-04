from __future__ import annotations

import hashlib
import json
import re
from bisect import bisect_left
from collections.abc import Iterable
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from src.ingestion.chunk_contracts import Chunk, ChunkSource, ChunkedDocument
from src.ingestion.contracts import NonEmptyStr, NonNegativeInt, PositiveInt
from src.ingestion.parser_contracts import (
    BlockType,
    CanonicalBlock,
    CanonicalDocument,
    CanonicalPage,
    PageParseStatus,
    ParserFailure,
)

CHUNKING_ALGORITHM_VERSION = "1.3.0"
PROVISION_MARKER_PATTERNS = (
    r"(?m)^[ \t]*(?!20[0-9]{2}\.[ \t]+[A-Z])[0-9]+(?:\.[0-9]+)*(?:[.)])?(?=[ \t]|$)",
    (
        r"(?im)^[ \t]*(?:article|section|clause|chapter|part|schedule|"
        r"annex(?:ure)?|appendix|exhibit)s?[ \t]+"
        r"(?:[ivxlcdm]+|[A-Z]|[0-9]+(?:\.[0-9]+)*)(?=[ \t]|[.:)]|$)"
    ),
    r"(?m)^[ \t]*\u00a7{1,2}[ \t]*[0-9]+(?:\.[0-9]+)*(?=[ \t]|[.:)]|$)",
    r"(?m)^[ \t]*\((?:[a-z]|[ivxlcdm]+|[0-9]+)\)(?=[ \t]|$)",
)
STRONG_PROVISION_MARKER_PATTERNS = PROVISION_MARKER_PATTERNS[:3]
LEGAL_ABBREVIATIONS = (
    "art.",
    "arts.",
    "ch.",
    "Dr.",
    "e.g.",
    "et al.",
    "i.e.",
    "Inc.",
    "Jr.",
    "Ltd.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "No.",
    "pp.",
    "Prof.",
    "Sr.",
    "St.",
    "U.K.",
    "U.S.",
    "v.",
    "vs.",
)

_NonEmptyLiteralString = Annotated[str, StringConstraints(min_length=1)]


class ChunkingError(Exception):
    """Base class for chunking-specific errors."""


class InvalidChunkingConfigError(ChunkingError, ValueError):
    """Raised when chunking configuration or its version is invalid."""


class InvalidChunkingStateError(ChunkingError, RuntimeError):
    """Raised when the chunking implementation reaches an invalid state."""


class ChunkingConfig(BaseModel):
    """Validated settings that affect deterministic chunk generation."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    max_chunk_chars: Annotated[int, Field(ge=16, strict=True)] = 1500
    overlap_chars: Annotated[int, Field(ge=0, strict=True)] = Field(
        default=100,
        description="Maximum source context repeated between adjacent text chunks; capped at one quarter of the chunk budget.",
    )
    synthetic_separator: _NonEmptyLiteralString = "\n\n"
    marker_breaks: bool = Field(default=True, strict=True)
    marker_break_min_pct: Annotated[int, Field(ge=0, le=100, strict=True)] = 25
    grapheme_backoff_chars: Annotated[int, Field(ge=0, strict=True)] = 32
    provision_marker_patterns: tuple[_NonEmptyLiteralString, ...] = (
        PROVISION_MARKER_PATTERNS
    )
    include_image_text: bool = Field(default=True, strict=True)
    abbreviations: tuple[_NonEmptyLiteralString, ...] = LEGAL_ABBREVIATIONS

    @field_validator("provision_marker_patterns")
    @classmethod
    def validate_provision_marker_patterns(
        cls, patterns: tuple[str, ...]
    ) -> tuple[str, ...]:
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"Invalid provision marker pattern: {pattern!r}") from exc
        return patterns

    def digest(self) -> str:
        """Return the full SHA-256 digest of canonical behavior settings JSON."""
        canonical = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def policy_version(
        self, semantic_version: str = CHUNKING_ALGORITHM_VERSION
    ) -> str:
        """Return ``5c-<semver>+<12-hex-config-digest>``."""
        if not isinstance(semantic_version, str) or not re.fullmatch(
            r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)",
            semantic_version,
        ):
            raise InvalidChunkingConfigError(
                "semantic_version must use MAJOR.MINOR.PATCH format"
            )
        return f"5c-{semantic_version}+{self.digest()[:12]}"


# COMPLETED: supported content was chunked with no failures or unsupported blocks.
# NO_CONTENT: no chunks or unsupported blocks. PARTIAL: unsupported content
# remains, or chunks coexist with failures. FAILED: failures and no chunks.
ChunkingOutcome = Literal["COMPLETED", "NO_CONTENT", "PARTIAL", "FAILED"]
PageChunkingOutcome = Literal[
    "CHUNKED",
    "SKIPPED_NO_ELIGIBLE_TEXT",
    "SKIPPED_UNSUPPORTED_BLOCKS",
    "SKIPPED_BLANK",
    "SKIPPED_VECTOR_ONLY",
    "SKIPPED_FAILED",
    "SKIPPED_IMAGE_NO_TEXT",
]


class PageChunkingReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    page_number: PositiveInt
    parser_status: PageParseStatus
    outcome: PageChunkingOutcome
    chunks_created: NonNegativeInt = 0
    failure: ParserFailure | None = None
    warnings: tuple[ParserFailure, ...] = ()

    @model_validator(mode="after")
    def validate_page_outcome(self) -> "PageChunkingReport":
        allowed_outcomes = {
            "PARSED": {
                "CHUNKED",
                "SKIPPED_NO_ELIGIBLE_TEXT",
                "SKIPPED_UNSUPPORTED_BLOCKS",
                "SKIPPED_IMAGE_NO_TEXT",
            },
            "BLANK": {"SKIPPED_BLANK"},
            "VECTOR_ONLY": {"SKIPPED_VECTOR_ONLY"},
            "FAILED": {"SKIPPED_FAILED"},
        }
        if self.outcome not in allowed_outcomes[self.parser_status]:
            raise ValueError(
                f"Parser status {self.parser_status} cannot have outcome "
                f"{self.outcome}"
            )
        if self.parser_status == "FAILED" and self.failure is None:
            raise ValueError("Failed pages must retain parser failure details")
        if self.parser_status != "FAILED" and self.failure is not None:
            raise ValueError("Only failed pages may include parser failure details")
        if (self.outcome == "CHUNKED") != (self.chunks_created > 0):
            raise ValueError("CHUNKED pages must report at least one created chunk")
        return self


class SkippedBlock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    page_number: PositiveInt
    block_id: NonEmptyStr
    block_type: BlockType
    reason: NonEmptyStr


PageRoutingOutcome = Literal[
    "READY_FOR_SEGMENTATION",
    "SKIPPED_NO_ELIGIBLE_TEXT",
    "SKIPPED_BLANK",
    "SKIPPED_VECTOR_ONLY",
    "SKIPPED_FAILED",
    "SKIPPED_IMAGE_NO_TEXT",
]


class PreparedPage(BaseModel):
    """M2 page classification; it deliberately has no chunk count/outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    page_number: PositiveInt
    parser_status: PageParseStatus
    outcome: PageRoutingOutcome
    eligible_blocks: tuple[CanonicalBlock, ...] = ()
    failure: ParserFailure | None = None
    warnings: tuple[ParserFailure, ...] = ()

    @model_validator(mode="after")
    def validate_route(self) -> "PreparedPage":
        parsed_routes = {
            "READY_FOR_SEGMENTATION",
            "SKIPPED_NO_ELIGIBLE_TEXT",
            "SKIPPED_IMAGE_NO_TEXT",
        }
        expected = {
            "BLANK": "SKIPPED_BLANK",
            "VECTOR_ONLY": "SKIPPED_VECTOR_ONLY",
            "FAILED": "SKIPPED_FAILED",
        }
        if self.parser_status == "PARSED":
            if self.outcome not in parsed_routes:
                raise ValueError("Parsed pages require a parsed-page routing outcome")
            if (self.outcome == "READY_FOR_SEGMENTATION") != bool(
                self.eligible_blocks
            ):
                raise ValueError(
                    "Ready pages must have eligible blocks and skipped pages must not"
                )
        elif self.outcome != expected[self.parser_status]:
            raise ValueError(
                f"Parser status {self.parser_status} requires outcome "
                f"{expected[self.parser_status]}"
            )
        elif self.eligible_blocks:
            raise ValueError("Non-parsed pages cannot contain eligible text blocks")
        if (self.parser_status == "FAILED") != (self.failure is not None):
            raise ValueError("Failed page details must match the parser status")
        if any(
            block.page_number != self.page_number for block in self.eligible_blocks
        ):
            raise ValueError("Eligible blocks must belong to their prepared page")
        return self


class ChunkingPreparation(BaseModel):
    """Intermediate page routing and eligibility result for M2."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pages: tuple[PreparedPage, ...] = ()
    skipped_blocks: tuple[SkippedBlock, ...] = ()

    @model_validator(mode="after")
    def validate_page_records(self) -> "ChunkingPreparation":
        page_numbers = tuple(page.page_number for page in self.pages)
        if page_numbers != tuple(range(1, len(self.pages) + 1)):
            raise ValueError("Prepared pages must be ordered and contiguous")
        if any(block.page_number > len(self.pages) for block in self.skipped_blocks):
            raise ValueError("Skipped blocks must refer to a prepared page")
        return self

    @property
    def eligible_blocks(self) -> tuple[CanonicalBlock, ...]:
        return tuple(block for page in self.pages for block in page.eligible_blocks)


def prepare_document_for_chunking(
    document: CanonicalDocument | Iterable[CanonicalPage],
    config: ChunkingConfig | None = None,
) -> ChunkingPreparation:
    """Classify pages and source blocks without segmenting or packing text."""
    settings = config or ChunkingConfig()
    pages = document.pages if isinstance(document, CanonicalDocument) else tuple(document)
    page_numbers = tuple(page.page_number for page in pages)
    if page_numbers != tuple(range(1, len(pages) + 1)):
        raise InvalidChunkingStateError(
            "Canonical pages must be ordered, contiguous, and one-based"
        )

    eligible_types: frozenset[BlockType] = frozenset({"TEXT", "TABLE", "OTHER"})
    prepared_pages: list[PreparedPage] = []
    skipped_blocks: list[SkippedBlock] = []
    for page in pages:
        eligible: list[CanonicalBlock] = []
        for block in page.blocks:
            has_text = bool(block.text.strip())
            reason: str | None = None
            if block.block_type in eligible_types:
                if has_text:
                    eligible.append(block)
                else:
                    reason = "WHITESPACE_ONLY_TEXT" if block.text else "EMPTY_TEXT"
            elif block.block_type == "IMAGE":
                if not has_text:
                    reason = "IMAGE_NO_USABLE_TEXT"
                elif settings.include_image_text:
                    eligible.append(block)
                else:
                    reason = "IMAGE_TEXT_DISABLED"
            elif block.block_type == "VECTOR":
                reason = "VECTOR_NOT_TEXT_ELIGIBLE"

            if reason is not None:
                skipped_blocks.append(
                    SkippedBlock(
                        page_number=page.page_number,
                        block_id=block.block_id,
                        block_type=block.block_type,
                        reason=reason,
                    )
                )

        if page.status == "BLANK":
            outcome: PageRoutingOutcome = "SKIPPED_BLANK"
        elif page.status == "VECTOR_ONLY":
            outcome = "SKIPPED_VECTOR_ONLY"
        elif page.status == "FAILED":
            outcome = "SKIPPED_FAILED"
        elif eligible:
            outcome = "READY_FOR_SEGMENTATION"
        elif page.blocks and all(
            block.block_type == "IMAGE" and not block.text.strip()
            for block in page.blocks
        ):
            outcome = "SKIPPED_IMAGE_NO_TEXT"
        else:
            outcome = "SKIPPED_NO_ELIGIBLE_TEXT"

        prepared_pages.append(
            PreparedPage(
                page_number=page.page_number,
                parser_status=page.status,
                outcome=outcome,
                eligible_blocks=tuple(eligible),
                failure=page.failure,
                warnings=page.warnings,
            )
        )

    return ChunkingPreparation(
        pages=tuple(prepared_pages), skipped_blocks=tuple(skipped_blocks)
    )


class ChunkingDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pages_processed: NonNegativeInt = 0
    eligible_blocks: NonNegativeInt = 0
    skipped_blocks: NonNegativeInt = 0
    whitespace_only_blocks: NonNegativeInt = 0
    chunks_created: NonNegativeInt = 0
    whitespace_only_chunks: NonNegativeInt = 0
    oversized_table_rows: NonNegativeInt = 0
    unsafe_grapheme_cuts: NonNegativeInt = 0


class ChunkingReport(BaseModel):
    """Outcome semantics for BLP-1 chunking.

    Completed means every eligible block group was processed without failed
    pages. No-content means no eligible chunks and no failed pages, including
    an entirely blank document. Partial means chunks coexist with failed pages
    or an unsupported block group remains. Failed means no chunks were made
    and at least one page failed. Warnings alone do not change the outcome.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: ChunkingOutcome
    policy_version: Annotated[
        str,
        StringConstraints(
            pattern=r"^5c-(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\+[0-9a-f]{12}$"
        ),
    ]
    config_digest: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    pages: tuple[PageChunkingReport, ...] = ()
    skipped_blocks: tuple[SkippedBlock, ...] = ()
    diagnostics: ChunkingDiagnostics = Field(default_factory=ChunkingDiagnostics)

    @model_validator(mode="after")
    def validate_integrity(self) -> "ChunkingReport":
        version_digest = self.policy_version.rsplit("+", maxsplit=1)[1]
        if version_digest != self.config_digest[:12]:
            raise ValueError("Policy version digest must match config_digest prefix")

        has_chunks = any(page.chunks_created for page in self.pages)
        has_failed_pages = any(
            page.outcome == "SKIPPED_FAILED" for page in self.pages
        )
        has_unsupported_blocks = any(
            block.reason == "UNSUPPORTED_M5_BLOCK_TYPE"
            for block in self.skipped_blocks
        )
        if has_unsupported_blocks or (has_chunks and has_failed_pages):
            expected_outcome = "PARTIAL"
        elif has_chunks:
            expected_outcome = "COMPLETED"
        elif has_failed_pages:
            expected_outcome = "FAILED"
        else:
            expected_outcome = "NO_CONTENT"
        if self.outcome != expected_outcome:
            raise ValueError(
                f"Report outcome must be {expected_outcome} for its page outcomes"
            )
        if self.diagnostics.pages_processed != len(self.pages):
            raise ValueError(
                "Processed-page count must match the number of page reports"
            )
        if self.diagnostics.chunks_created != sum(
            page.chunks_created for page in self.pages
        ):
            raise ValueError(
                "Diagnostic chunk count must match the per-page chunk counts"
            )
        if self.diagnostics.skipped_blocks != len(self.skipped_blocks):
            raise ValueError(
                "Diagnostic skipped-block count must match skipped block entries"
            )
        return self


class ChunkingResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document: ChunkedDocument
    report: ChunkingReport

    @model_validator(mode="after")
    def validate_policy_version(self) -> "ChunkingResult":
        if self.document.policy_version != self.report.policy_version:
            raise ValueError("Document and report policy versions must match")
        expected_pages = tuple(range(1, self.document.source_page_count + 1))
        actual_pages = tuple(page.page_number for page in self.report.pages)
        if actual_pages != expected_pages:
            raise ValueError("Report pages must cover each document page in order")
        if self.report.diagnostics.chunks_created != len(self.document.chunks):
            raise ValueError("Reported chunk count must match the chunked document")
        if sum(page.chunks_created for page in self.report.pages) != len(
            self.document.chunks
        ):
            raise ValueError("Per-page chunk counts must match the chunked document")
        return self


def _atom_starts_strong_provision(
    block: CanonicalBlock, atom_start: int, config: ChunkingConfig
) -> bool:
    if not config.marker_breaks:
        return False
    if atom_start != 0 and block.text[atom_start - 1] not in "\r\n":
        return False
    line_end = atom_start
    while line_end < len(block.text) and block.text[line_end] not in "\r\n":
        line_end += 1
    line = block.text[atom_start:line_end]
    return any(
        re.match(pattern, line) is not None
        for pattern in STRONG_PROVISION_MARKER_PATTERNS
    )


def _pack_page_text_blocks(
    blocks: tuple[CanonicalBlock, ...],
    config: ChunkingConfig,
    first_chunk_index: int,
    *,
    allowed_types: frozenset[BlockType] = frozenset({"TEXT", "OTHER"}),
) -> tuple[tuple[Chunk, ...], int]:
    """Greedily pack M4 atoms from one isolated block-type group."""
    from src.ingestion.segmentation import BoundaryTier, segment_oversized_text

    output: list[Chunk] = []
    current_text_parts: list[str] = []
    current_length = 0
    current_sources: list[ChunkSource] = []
    last_block_id: str | None = None
    unsafe_grapheme_cuts = 0
    marker_threshold = (
        config.max_chunk_chars * config.marker_break_min_pct // 100
    )

    def finish_chunk() -> None:
        nonlocal current_text_parts, current_length, current_sources, last_block_id
        if current_sources:
            output.append(
                Chunk(
                    index=first_chunk_index + len(output),
                    text="".join(current_text_parts),
                    sources=tuple(current_sources),
                )
            )
        current_text_parts = []
        current_length = 0
        current_sources = []
        last_block_id = None

    for block in blocks:
        if block.block_type not in allowed_types:
            raise InvalidChunkingStateError(
                "Atom packer received a block outside its isolated block group"
            )
        overlap_limit = (
            min(config.overlap_chars, config.max_chunk_chars // 4)
            if block.block_type in {"TEXT", "OTHER"}
            else 0
        )
        segmentation_config = (
            config.model_copy(
                update={
                    "max_chunk_chars": config.max_chunk_chars - overlap_limit
                }
            )
            if overlap_limit
            else config
        )
        result = segment_oversized_text(block.text, segmentation_config)
        from src.ingestion.segmentation import (
            _grapheme_boundaries,
            paragraph_boundaries,
        )

        paragraph_starts = set(paragraph_boundaries(block.text))
        safe_grapheme_boundaries = (
            _grapheme_boundaries(block.text) if overlap_limit else ()
        )
        unsafe_grapheme_cuts += sum(
            split.tier is BoundaryTier.GRAPHEME_HARD_CUT
            and split.is_grapheme_safe is False
            for split in result.splits
        )
        for atom in result.atoms:
            if atom.start < 0 or atom.end > len(block.text) or atom.start >= atom.end:
                raise InvalidChunkingStateError("M4 returned an invalid source span")
            atom_text = block.text[atom.start : atom.end]
            if len(atom_text) > config.max_chunk_chars:
                raise InvalidChunkingStateError("M4 returned an oversized atom")
            starts_marker = _atom_starts_strong_provision(block, atom.start, config)

            while True:
                if (
                    starts_marker
                    and current_sources
                    and current_length >= marker_threshold
                ):
                    finish_chunk()
                    continue

                separator = (
                    config.synthetic_separator
                    if current_sources and last_block_id != block.block_id
                    else ""
                )
                proposed_length = current_length + len(separator) + len(atom_text)
                if proposed_length > config.max_chunk_chars:
                    if not current_sources:
                        raise InvalidChunkingStateError(
                            "A single M4 atom exceeds the configured chunk budget"
                        )
                    previous_source = current_sources[-1]
                    overlap_start = atom.start - overlap_limit
                    can_overlap = (
                        overlap_limit > 0
                        and not starts_marker
                        and atom.start not in paragraph_starts
                        and previous_source.block_id == block.block_id
                        and previous_source.end_offset == atom.start
                    )
                    if can_overlap:
                        boundary_index = bisect_left(
                            safe_grapheme_boundaries, atom.start - overlap_limit
                        )
                        if boundary_index < len(safe_grapheme_boundaries):
                            boundary = safe_grapheme_boundaries[boundary_index]
                            if boundary < atom.start:
                                overlap_start = boundary
                            else:
                                overlap_start = atom.start
                    finish_chunk()
                    if can_overlap and overlap_start < atom.start:
                        overlap_text = block.text[overlap_start : atom.start]
                        current_text_parts.append(overlap_text)
                        current_length = len(overlap_text)
                        current_sources.append(
                            ChunkSource(
                                page_number=block.page_number,
                                block_id=block.block_id,
                                block_reading_order=block.reading_order,
                                block_type=block.block_type,
                                source_text_length=len(block.text),
                                start_offset=overlap_start,
                                end_offset=atom.start,
                            )
                        )
                        last_block_id = block.block_id
                    continue

                if separator:
                    current_text_parts.append(separator)
                current_text_parts.append(atom_text)
                current_length = proposed_length
                if (
                    current_sources
                    and current_sources[-1].block_id == block.block_id
                    and current_sources[-1].end_offset == atom.start
                ):
                    previous = current_sources[-1]
                    current_sources[-1] = ChunkSource(
                        page_number=previous.page_number,
                        block_id=previous.block_id,
                        block_reading_order=previous.block_reading_order,
                        block_type=previous.block_type,
                        source_text_length=previous.source_text_length,
                        start_offset=previous.start_offset,
                        end_offset=atom.end,
                    )
                else:
                    current_sources.append(
                        ChunkSource(
                            page_number=block.page_number,
                            block_id=block.block_id,
                            block_reading_order=block.reading_order,
                            block_type=block.block_type,
                            source_text_length=len(block.text),
                            start_offset=atom.start,
                            end_offset=atom.end,
                        )
                    )
                last_block_id = block.block_id
                break

    finish_chunk()
    return tuple(output), unsafe_grapheme_cuts


_TABLE_COLUMN_GAP = re.compile(r"[ \t]+\|[ \t]+|\t+| {2,}")


def _table_row_ranges(text: str) -> tuple[tuple[int, int], ...]:
    from src.ingestion.segmentation import line_boundaries

    edges = (0, *line_boundaries(text), len(text))
    return tuple((start, end) for start, end in zip(edges, edges[1:]) if start < end)


def _split_oversized_table_row(
    text: str, start: int, end: int, config: ChunkingConfig
) -> tuple[tuple[tuple[int, int], ...], int]:
    """Split one oversized row by column gaps, words, then grapheme cuts."""
    from src.ingestion.segmentation import (
        _grapheme_boundaries,
        _hard_cut_from_boundaries,
        word_boundaries,
    )

    grapheme_boundaries: tuple[int, ...] | None = None
    grapheme_boundary_set: set[int] | None = None
    result: list[tuple[int, int]] = []
    unsafe_cuts = 0
    # The final tier is a hard cut. Iterative traversal avoids recursion depth
    # limits on very long unbroken table rows.
    pending: list[tuple[int, int, int]] = [(start, end, 0)]
    while pending:
        piece_start, piece_end, tier_start = pending.pop()
        if piece_end - piece_start <= config.max_chunk_chars:
            result.append((piece_start, piece_end))
            continue

        piece = text[piece_start:piece_end]
        candidates: tuple[int, ...] = ()
        next_tier = tier_start
        if tier_start <= 0:
            candidates = tuple(
                piece_start + match.end()
                for match in _TABLE_COLUMN_GAP.finditer(piece)
                if match.end() < len(piece)
            )
            next_tier = 1
        if not candidates and tier_start <= 1:
            candidates = tuple(
                piece_start + position
                for position in word_boundaries(piece)
            )
            next_tier = 2

        if candidates:
            edges = (piece_start, *candidates, piece_end)
            children = tuple(zip(edges, edges[1:]))
            if any(child_start >= child_end for child_start, child_end in children):
                raise InvalidChunkingStateError(
                    "Table row boundary produced an empty range"
                )
            for child_start, child_end in reversed(children):
                pending.append((child_start, child_end, next_tier))
            continue

        if grapheme_boundaries is None:
            grapheme_boundaries = _grapheme_boundaries(text)
            grapheme_boundary_set = set(grapheme_boundaries)
        assert grapheme_boundary_set is not None
        try:
            hard_cut = _hard_cut_from_boundaries(
                text,
                piece_start,
                config.max_chunk_chars,
                config,
                grapheme_boundaries,
                grapheme_boundary_set,
            )
            cut = hard_cut.position
            if not hard_cut.is_grapheme_safe:
                unsafe_cuts += 1
        except InvalidChunkingStateError:
            cut = min(piece_start + config.max_chunk_chars, piece_end)
            unsafe_cuts += 1

        if not piece_start < cut < piece_end:
            raise InvalidChunkingStateError("Table row fallback failed to progress")
        pending.append((cut, piece_end, 2))
        pending.append((piece_start, cut, 2))

    result.sort()
    if result[0][0] != start or result[-1][1] != end or any(
        left_end != right_start
        for (_, left_end), (right_start, _) in zip(result, result[1:])
    ):
        raise InvalidChunkingStateError("Table row fragments are not contiguous")
    if any(end_offset - start_offset > config.max_chunk_chars for start_offset, end_offset in result):
        raise InvalidChunkingStateError("Table row fallback produced an oversized range")
    return tuple(result), unsafe_cuts


def _pack_table_block(
    block: CanonicalBlock, config: ChunkingConfig, first_chunk_index: int
) -> tuple[tuple[Chunk, ...], int, int]:
    """Pack one TABLE block into isolated, contiguous source ranges."""
    if block.block_type != "TABLE":
        raise InvalidChunkingStateError("Table packer requires a TABLE block")
    if not block.text:
        return (), 0, 0

    rows = _table_row_ranges(block.text)
    oversized_rows = 0
    unsafe_cuts = 0
    chunk_ranges: list[tuple[int, int]] = []
    current_start: int | None = None
    current_end: int | None = None

    def finish_range() -> None:
        nonlocal current_start, current_end
        if current_start is not None and current_end is not None:
            chunk_ranges.append((current_start, current_end))
        current_start = current_end = None

    for row_start, row_end in rows:
        row_length = row_end - row_start
        if row_length > config.max_chunk_chars:
            oversized_rows += 1
            finish_range()
            fragments, row_unsafe_cuts = _split_oversized_table_row(
                block.text, row_start, row_end, config
            )
            unsafe_cuts += row_unsafe_cuts
            for fragment_start, fragment_end in fragments:
                if (
                    current_start is not None
                    and current_end is not None
                    and fragment_end - current_start <= config.max_chunk_chars
                ):
                    current_end = fragment_end
                else:
                    finish_range()
                    current_start, current_end = fragment_start, fragment_end
            continue

        if current_start is None:
            current_start, current_end = row_start, row_end
        elif row_end - current_start <= config.max_chunk_chars:
            current_end = row_end
        else:
            finish_range()
            current_start, current_end = row_start, row_end
    finish_range()

    chunks = tuple(
        Chunk(
            index=first_chunk_index + index,
            text=block.text[start:end],
            sources=(
                ChunkSource(
                    page_number=block.page_number,
                    block_id=block.block_id,
                    block_reading_order=block.reading_order,
                    block_type="TABLE",
                    source_text_length=len(block.text),
                    start_offset=start,
                    end_offset=end,
                ),
            ),
        )
        for index, (start, end) in enumerate(chunk_ranges)
    )
    if any(len(chunk.text) > config.max_chunk_chars for chunk in chunks):
        raise InvalidChunkingStateError("Table packer produced an oversized chunk")
    cursor = 0
    for chunk in chunks:
        source = chunk.sources[0]
        if source.start_offset != cursor:
            raise InvalidChunkingStateError("Table chunks are not source-contiguous")
        cursor = source.end_offset
    if cursor != len(block.text):
        raise InvalidChunkingStateError("Table chunks do not cover the complete source")
    if "".join(chunk.text for chunk in chunks) != block.text:
        raise InvalidChunkingStateError("Table chunks do not reconstruct source text")
    return chunks, oversized_rows, unsafe_cuts


def chunk_document(
    document: CanonicalDocument | Iterable[CanonicalPage],
    document_id: str,
    config: ChunkingConfig | None = None,
) -> ChunkingResult:
    """Segment and pack all eligible TEXT, OTHER, TABLE, and IMAGE content.

    Ordinary text blocks retain M5 packing. TABLE and IMAGE blocks are packed
    as isolated groups and can never share a chunk with ordinary text or with
    another table/image block.
    """
    settings = config or ChunkingConfig()
    pages = (
        document.pages
        if isinstance(document, CanonicalDocument)
        else tuple(document)
    )
    preparation = prepare_document_for_chunking(pages, settings)

    skipped_blocks = preparation.skipped_blocks

    chunks: list[Chunk] = []
    page_reports: list[PageChunkingReport] = []
    oversized_table_rows = 0
    unsafe_grapheme_cuts = 0
    for prepared_page in preparation.pages:
        page_chunks: list[Chunk] = []
        ordinary_blocks: list[CanonicalBlock] = []

        def flush_ordinary_blocks() -> None:
            nonlocal unsafe_grapheme_cuts
            if not ordinary_blocks:
                return
            packed, unsafe = _pack_page_text_blocks(
                tuple(ordinary_blocks),
                settings,
                first_chunk_index=len(chunks) + len(page_chunks),
            )
            page_chunks.extend(packed)
            unsafe_grapheme_cuts += unsafe
            ordinary_blocks.clear()

        for block in prepared_page.eligible_blocks:
            if block.block_type in {"TEXT", "OTHER"}:
                ordinary_blocks.append(block)
                continue
            flush_ordinary_blocks()
            first_chunk_index = len(chunks) + len(page_chunks)
            if block.block_type == "TABLE":
                table_chunks, oversized_rows, unsafe = _pack_table_block(
                    block, settings, first_chunk_index
                )
                page_chunks.extend(table_chunks)
                oversized_table_rows += oversized_rows
                unsafe_grapheme_cuts += unsafe
            elif block.block_type == "IMAGE":
                image_chunks, unsafe = _pack_page_text_blocks(
                    (block,),
                    settings,
                    first_chunk_index=first_chunk_index,
                    allowed_types=frozenset({"IMAGE"}),
                )
                page_chunks.extend(image_chunks)
                unsafe_grapheme_cuts += unsafe
            else:
                raise InvalidChunkingStateError(
                    f"No M6 packing strategy for eligible {block.block_type} block"
                )
        flush_ordinary_blocks()
        page_chunks_tuple = tuple(page_chunks)
        chunks.extend(page_chunks_tuple)

        if page_chunks_tuple:
            page_outcome: PageChunkingOutcome = "CHUNKED"
        else:
            page_outcome = prepared_page.outcome  # type: ignore[assignment]
        page_reports.append(
            PageChunkingReport(
                page_number=prepared_page.page_number,
                parser_status=prepared_page.parser_status,
                outcome=page_outcome,
                chunks_created=len(page_chunks_tuple),
                failure=prepared_page.failure,
                warnings=prepared_page.warnings,
            )
        )

    unsupported_remains = any(
        item.reason == "UNSUPPORTED_M5_BLOCK_TYPE" for item in skipped_blocks
    )
    has_failures = any(page.parser_status == "FAILED" for page in preparation.pages)
    if unsupported_remains or (chunks and has_failures):
        overall_outcome: ChunkingOutcome = "PARTIAL"
    elif chunks:
        overall_outcome = "COMPLETED"
    elif has_failures:
        overall_outcome = "FAILED"
    else:
        overall_outcome = "NO_CONTENT"

    config_digest = settings.digest()
    report = ChunkingReport(
        outcome=overall_outcome,
        policy_version=settings.policy_version(),
        config_digest=config_digest,
        pages=tuple(page_reports),
        skipped_blocks=skipped_blocks,
        diagnostics=ChunkingDiagnostics(
            pages_processed=len(preparation.pages),
            eligible_blocks=len(preparation.eligible_blocks),
            skipped_blocks=len(skipped_blocks),
            whitespace_only_blocks=sum(
                item.reason == "WHITESPACE_ONLY_TEXT"
                for item in preparation.skipped_blocks
            ),
            chunks_created=len(chunks),
            whitespace_only_chunks=sum(not chunk.text.strip() for chunk in chunks),
            oversized_table_rows=oversized_table_rows,
            unsafe_grapheme_cuts=unsafe_grapheme_cuts,
        ),
    )
    chunked_document = ChunkedDocument(
        document_id=document_id,
        source_page_count=len(preparation.pages),
        policy_version=settings.policy_version(),
        chunks=tuple(chunks),
    )
    return ChunkingResult(document=chunked_document, report=report)
