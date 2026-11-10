from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum

from src.ingestion.chunking import ChunkingConfig, InvalidChunkingStateError

_PARAGRAPH_BREAK = re.compile(r"(?:\r\n|\r|\n)[ \t]*(?:(?:\r\n|\r|\n)[ \t]*)+")
_CLOSING_PUNCTUATION = frozenset('"\'”’»)]}')
_OPENING_PUNCTUATION = frozenset('"\'“‘«([{')


def _ordered_positions(text: str, positions: set[int] | list[int]) -> tuple[int, ...]:
    return tuple(sorted({position for position in positions if 0 < position < len(text)}))


def paragraph_boundaries(text: str) -> tuple[int, ...]:
    """Return positions after blank-line runs; lone newlines stay inside pieces."""
    return _ordered_positions(
        text, [match.end() for match in _PARAGRAPH_BREAK.finditer(text)]
    )


def provision_marker_boundaries(
    text: str, config: ChunkingConfig | None = None
) -> tuple[int, ...]:
    """Return line-start positions for configured provision marker patterns."""
    settings = config or ChunkingConfig()
    if not settings.marker_breaks:
        return ()
    positions = {
        match.start()
        for pattern in settings.provision_marker_patterns
        for match in re.finditer(pattern, text)
    }
    return _ordered_positions(text, positions)


def _abbreviation_periods(text: str, abbreviations: tuple[str, ...]) -> set[int]:
    periods: set[int] = set()
    for abbreviation in abbreviations:
        pattern = rf"(?<!\w){re.escape(abbreviation)}(?!\w)"
        for match in re.finditer(pattern, text, re.IGNORECASE):
            periods.update(
                match.start() + offset
                for offset, character in enumerate(match.group())
                if character == "."
            )
    return periods


def _terminal_abbreviation_periods(
    text: str, abbreviations: tuple[str, ...]
) -> dict[int, str]:
    """Map each abbreviation's final period to its spelling."""
    terminals: dict[int, str] = {}
    for abbreviation in abbreviations:
        pattern = rf"(?<!\w){re.escape(abbreviation)}(?!\w)"
        for match in re.finditer(pattern, text, re.IGNORECASE):
            if abbreviation.endswith("."):
                terminals[match.end() - 1] = abbreviation.casefold()
    return terminals


def _abbreviation_precedes_citation_value(
    text: str, abbreviation: str, next_position: int
) -> bool:
    if abbreviation in {"v.", "vs."}:
        return True
    if abbreviation in {"no.", "art.", "arts.", "ch.", "pp."}:
        following_text = text[next_position:]
        return re.match(r"(?:[0-9]|[ivxlcdm]+\b)", following_text, re.IGNORECASE) is not None
    return False


def _numbering_period(text: str, period_position: int) -> bool:
    line_start = max(text.rfind("\n", 0, period_position), text.rfind("\r", 0, period_position)) + 1
    line_prefix = text[line_start : period_position + 1]
    return re.fullmatch(r"[ \t]*[0-9]+(?:\.[0-9]+)*(?:[.)])?", line_prefix) is not None


def _initial_before_name(text: str, period_position: int, next_position: int) -> bool:
    initial_position = period_position - 1
    if initial_position < 0 or text[initial_position] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return False
    if initial_position > 0:
        preceding = text[initial_position - 1]
        # Match the previous ``[^\w]`` rule without copying or scanning a
        # prefix. For Python regexes, Unicode alphanumerics and underscore
        # are the relevant word characters here.
        if preceding.isalnum() or preceding == "_":
            return False
    if next_position >= len(text):
        return False
    return text[next_position].isupper()


def _plausible_sentence_start(text: str, position: int) -> bool:
    while position < len(text) and text[position] in _OPENING_PUNCTUATION:
        position += 1
    if position >= len(text):
        return False
    character = text[position]
    codepoint = ord(character)
    return (
        character.isupper()
        or character.isdigit()
        or 0x0900 <= codepoint <= 0x097F
    )


def sentence_boundaries(
    text: str, config: ChunkingConfig | None = None
) -> tuple[int, ...]:
    """Return likely sentence ends, retaining punctuation, quotes, and spacing.

    This is a deterministic heuristic. It deliberately favors avoiding false
    splits after known abbreviations and line-leading clause numbers.
    """
    settings = config or ChunkingConfig()
    abbreviation_periods = _abbreviation_periods(text, settings.abbreviations)
    terminal_abbreviations = _terminal_abbreviation_periods(
        text, settings.abbreviations
    )
    positions: set[int] = set()
    punctuation = ".?!\u0964\u0965"
    for index, character in enumerate(text):
        if character not in punctuation:
            continue
        if character == ".":
            if index in abbreviation_periods:
                abbreviation = terminal_abbreviations.get(index)
                # Titles such as Dr. introduce a following name, while a
                # terminal abbreviation such as U.S. may end a sentence.
                if abbreviation is None or abbreviation in {
                    "dr.", "mr.", "mrs.", "ms.", "prof.", "sr.", "jr."
                }:
                    continue
            if _numbering_period(text, index):
                continue
            if index > 0 and index + 1 < len(text):
                if text[index - 1].isdigit() and text[index + 1].isdigit():
                    continue

        end = index + 1
        while end < len(text) and text[end] in _CLOSING_PUNCTUATION:
            end += 1
        if end == len(text) or not text[end].isspace():
            continue
        while end < len(text) and text[end].isspace():
            end += 1
        if end >= len(text) or not _plausible_sentence_start(text, end):
            continue
        if character == "." and _abbreviation_precedes_citation_value(
            text, terminal_abbreviations.get(index, ""), end
        ):
            continue
        if (
            character == "."
            and index not in terminal_abbreviations
            and _initial_before_name(text, index, end)
        ):
            continue
        positions.add(end)
    return _ordered_positions(text, positions)


def line_boundaries(text: str) -> tuple[int, ...]:
    """Return positions after CRLF, LF, or lone CR terminators only."""
    positions: list[int] = []
    index = 0
    while index < len(text):
        if text[index] == "\r":
            index += 2 if index + 1 < len(text) and text[index + 1] == "\n" else 1
            positions.append(index)
        elif text[index] == "\n":
            index += 1
            positions.append(index)
        else:
            index += 1
    return _ordered_positions(text, positions)


def word_boundaries(text: str) -> tuple[int, ...]:
    """Return positions after each whitespace run, preserving the full run."""
    positions: list[int] = []
    index = 0
    while index < len(text):
        if not text[index].isspace():
            index += 1
            continue
        while index < len(text) and text[index].isspace():
            index += 1
        positions.append(index)
    return _ordered_positions(text, positions)


@dataclass(frozen=True, slots=True)
class HardCutResult:
    position: int
    is_grapheme_safe: bool
    backoff_chars: int


def _is_regional_indicator(character: str) -> bool:
    return 0x1F1E6 <= ord(character) <= 0x1F1FF


def _is_grapheme_extend(character: str) -> bool:
    codepoint = ord(character)
    return (
        unicodedata.category(character).startswith("M")
        or 0xFE00 <= codepoint <= 0xFE0F
        or 0xE0100 <= codepoint <= 0xE01EF
        or 0x1F3FB <= codepoint <= 0x1F3FF
        or 0xE0020 <= codepoint <= 0xE007F
        or codepoint == 0x200C
    )


def _is_virama(character: str) -> bool:
    name = unicodedata.name(character, "")
    return "VIRAMA" in name or "HALANT" in name


def _is_indic_consonant(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x0900 <= codepoint <= 0x0DFF
        and unicodedata.category(character).startswith("L")
    )


def _grapheme_boundaries(text: str) -> tuple[int, ...]:
    """Approximate common grapheme clusters, not the complete UAX #29 rules.

    Groups combining marks, Devanagari virama conjuncts, variation selectors,
    emoji modifiers and tags, ZWJ/ZWNJ sequences, regional-indicator pairs,
    and CRLF. Other Unicode grapheme rules may still require a reported unsafe
    hard cut.
    """
    boundaries = [0]
    index = 0
    while index < len(text):
        if text[index] == "\r" and index + 1 < len(text) and text[index + 1] == "\n":
            index += 2
        elif _is_regional_indicator(text[index]):
            index += 1
            if index < len(text) and _is_regional_indicator(text[index]):
                index += 1
        else:
            index += 1
            linker_seen = False
            while index < len(text):
                if _is_grapheme_extend(text[index]):
                    linker_seen = linker_seen or _is_virama(text[index])
                    index += 1
                elif text[index] == "\u200d":
                    index += 1
                    if index < len(text):
                        index += 1
                    else:
                        break
                    linker_seen = False
                elif linker_seen and _is_indic_consonant(text[index]):
                    index += 1
                    linker_seen = False
                else:
                    break
        boundaries.append(index)
    return tuple(boundaries)


def grapheme_safe_hard_cut(
    text: str,
    start: int,
    max_chars: int,
    *,
    config: ChunkingConfig | None = None,
) -> HardCutResult:
    """Choose an absolute cut position within the character limit.

    Back off at most the configured ``grapheme_backoff_chars`` to a known grapheme
    boundary. If none exists in that window, cut at the hard limit and return
    ``is_grapheme_safe=False`` so the caller can record the degraded cut.
    """
    if type(start) is not int or not 0 <= start <= len(text):
        raise InvalidChunkingStateError("Hard-cut start is outside the source text")
    if type(max_chars) is not int or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")
    settings = config or ChunkingConfig()
    return _hard_cut_from_boundaries(
        text, start, max_chars, settings, _grapheme_boundaries(text)
    )


def _hard_cut_from_boundaries(
    text: str,
    start: int,
    max_chars: int,
    settings: ChunkingConfig,
    safe_positions: tuple[int, ...],
    safe_set: set[int] | None = None,
) -> HardCutResult:
    """Apply the hard-cut policy using a reusable grapheme boundary index."""
    if safe_set is None:
        safe_set = set(safe_positions)
    if start not in safe_set:
        raise InvalidChunkingStateError("Hard-cut start must be a grapheme boundary")
    if start == len(text):
        return HardCutResult(position=start, is_grapheme_safe=True, backoff_chars=0)

    limit = min(start + max_chars, len(text))
    if limit in safe_set:
        return HardCutResult(position=limit, is_grapheme_safe=True, backoff_chars=0)

    minimum = max(start + 1, limit - settings.grapheme_backoff_chars)
    for position in range(limit - 1, minimum - 1, -1):
        if position in safe_set:
            return HardCutResult(
                position=position,
                is_grapheme_safe=True,
                backoff_chars=limit - position,
            )
    return HardCutResult(position=limit, is_grapheme_safe=False, backoff_chars=0)


class BoundaryTier(str, Enum):
    PARAGRAPH = "paragraph"
    PROVISION_MARKER = "provision_marker"
    SENTENCE = "sentence"
    LINE = "line"
    WORD = "word"
    GRAPHEME_HARD_CUT = "grapheme_hard_cut"


@dataclass(frozen=True, slots=True)
class SegmentationAtom:
    """A contiguous half-open span in the original block text."""

    start: int
    end: int
    created_by: BoundaryTier | None


@dataclass(frozen=True, slots=True)
class SegmentationSplit:
    """A split operation and its source-relative boundary positions."""

    start: int
    end: int
    tier: BoundaryTier
    boundaries: tuple[int, ...]
    is_grapheme_safe: bool | None = None
    backoff_chars: int = 0


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    source_text: str
    atoms: tuple[SegmentationAtom, ...]
    splits: tuple[SegmentationSplit, ...]

    @property
    def atom_texts(self) -> tuple[str, ...]:
        return tuple(self.source_text[atom.start : atom.end] for atom in self.atoms)


def segment_oversized_text(
    text: str, config: ChunkingConfig | None = None
) -> SegmentationResult:
    """Split a source block losslessly using the ordered M4 boundary ladder.

    Each oversized span uses the first tier with at least one interior
    boundary. Its child spans are then processed independently from the top
    of the ladder. Hard cuts guarantee progress when no semantic boundary is
    available. All returned offsets refer to ``text``.
    """
    settings = config or ChunkingConfig()
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if text == "":
        return SegmentationResult(source_text=text, atoms=(), splits=())

    atoms: list[SegmentationAtom] = []
    splits: list[SegmentationSplit] = []
    grapheme_boundaries = _grapheme_boundaries(text)
    grapheme_boundary_set = set(grapheme_boundaries)
    # Each child continues below the tier that created it. This prevents
    # repeatedly rescanning a range for boundary types already exhausted.
    pending: list[tuple[int, int, BoundaryTier | None, int]] = [
        (0, len(text), None, 0)
    ]
    ladder = (
        BoundaryTier.PARAGRAPH,
        BoundaryTier.PROVISION_MARKER,
        BoundaryTier.SENTENCE,
        BoundaryTier.LINE,
        BoundaryTier.WORD,
    )

    while pending:
        start, end, created_by, tier_start = pending.pop()
        if end - start <= settings.max_chunk_chars:
            atoms.append(SegmentationAtom(start, end, created_by))
            continue

        piece = text[start:end]
        selected_tier: BoundaryTier | None = None
        selected_tier_index = -1
        selected_boundaries: tuple[int, ...] = ()
        for tier_index in range(tier_start, len(ladder)):
            tier = ladder[tier_index]
            if tier is BoundaryTier.PROVISION_MARKER and not settings.marker_breaks:
                continue
            if tier is BoundaryTier.PARAGRAPH:
                positions = paragraph_boundaries(piece)
            elif tier is BoundaryTier.PROVISION_MARKER:
                positions = provision_marker_boundaries(piece, settings)
            elif tier is BoundaryTier.SENTENCE:
                positions = sentence_boundaries(piece, settings)
            elif tier is BoundaryTier.LINE:
                positions = line_boundaries(piece)
            else:
                positions = word_boundaries(piece)
            if positions:
                selected_tier = tier
                selected_tier_index = tier_index
                selected_boundaries = positions
                break

        if selected_tier is not None:
            absolute_boundaries = tuple(start + position for position in selected_boundaries)
            splits.append(
                SegmentationSplit(start, end, selected_tier, absolute_boundaries)
            )
            child_edges = (start, *absolute_boundaries, end)
            for child_start, child_end in reversed(
                tuple(zip(child_edges, child_edges[1:]))
            ):
                if child_start >= child_end:
                    raise InvalidChunkingStateError(
                        "Boundary ladder produced an empty segmentation span"
                    )
                pending.append(
                    (child_start, child_end, selected_tier, selected_tier_index + 1)
                )
            continue

        try:
            hard_cut = _hard_cut_from_boundaries(
                text,
                start,
                settings.max_chunk_chars,
                settings,
                grapheme_boundaries,
                grapheme_boundary_set,
            )
            cut = hard_cut.position
            is_safe = hard_cut.is_grapheme_safe
            backoff = hard_cut.backoff_chars
        except InvalidChunkingStateError:
            # A prior semantic boundary can land inside a grapheme. Preserve
            # source and size constraints, and expose that degraded cut.
            cut = min(start + settings.max_chunk_chars, end)
            is_safe = False
            backoff = 0

        # A fallback can leave the next span starting inside a grapheme. The
        # reusable index also makes that condition explicit without rescanning.
        if start not in grapheme_boundary_set and selected_tier is None:
            cut = min(start + settings.max_chunk_chars, end)
            is_safe = False
            backoff = 0

        if not start < cut < end:
            raise InvalidChunkingStateError("Hard-cut fallback failed to make progress")
        splits.append(
            SegmentationSplit(
                start,
                end,
                BoundaryTier.GRAPHEME_HARD_CUT,
                (cut,),
                is_grapheme_safe=is_safe,
                backoff_chars=backoff,
            )
        )
        pending.append((cut, end, BoundaryTier.GRAPHEME_HARD_CUT, len(ladder)))
        pending.append((start, cut, BoundaryTier.GRAPHEME_HARD_CUT, len(ladder)))

    atoms.sort(key=lambda atom: atom.start)
    if not atoms and text:
        raise InvalidChunkingStateError("Segmentation produced no atoms for nonempty text")
    if any(atom.end - atom.start > settings.max_chunk_chars for atom in atoms):
        raise InvalidChunkingStateError("Segmentation produced an oversized atom")
    if "".join(text[atom.start : atom.end] for atom in atoms) != text:
        raise InvalidChunkingStateError("Segmentation atoms do not reconstruct source text")
    if any(left.end != right.start for left, right in zip(atoms, atoms[1:])):
        raise InvalidChunkingStateError("Segmentation atoms are not source-contiguous")
    return SegmentationResult(text, tuple(atoms), tuple(splits))
