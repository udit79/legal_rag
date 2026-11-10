from __future__ import annotations
import bisect
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
UNICODE_SECURITY_VERSION = "16.0.0"
_DATA = Path(__file__).with_name("data") / f"unicode-{UNICODE_SECURITY_VERSION}"

@dataclass(frozen=True, slots=True)
class DetectionText:
    text: str
    # One half-open original-text span per detection-view character.
    source_spans: tuple[tuple[int, int], ...]

    def source_range(self, start: int, end: int) -> tuple[int, int]:
        if not 0 <= start < end <= len(self.source_spans):
            raise ValueError("Detection range must be non-empty and in bounds")
        return (
            min(span[0] for span in self.source_spans[start:end]),
            max(span[1] for span in self.source_spans[start:end]),
        )

@dataclass(frozen=True, slots=True)
class _UnicodeData:
    mappings: dict[tuple[int, ...], tuple[int, ...]]
    mapping_lengths: tuple[int, ...]
    ignored_ranges: tuple[tuple[int, int], ...]
    script_ranges: tuple[tuple[int, int, str], ...]
    script_starts: tuple[int, ...]
    script_extension_ranges: tuple[tuple[int, int, tuple[str, ...]], ...]
    script_extension_starts: tuple[int, ...]

def _codepoints(value: str) -> tuple[int, ...]:
    return tuple(int(part, 16) for part in value.split())

def _range(value: str) -> tuple[int, int]:
    left, _, right = value.partition("..")
    return int(left, 16), int(right or left, 16)

@lru_cache(maxsize=1)
def _unicode_data() -> _UnicodeData:
    mappings: dict[tuple[int, ...], tuple[int, ...]] = {}
    for line in (_DATA / "confusables.txt").read_text(encoding="utf-8").splitlines():
        body = line.split("#", 1)[0].strip()
        if not body:
            continue
        fields = [part.strip() for part in body.split(";")]
        if len(fields) >= 2:
            mappings[_codepoints(fields[0])] = _codepoints(fields[1])

    ignored_ranges = []
    for line in (_DATA / "DerivedCoreProperties.txt").read_text(encoding="utf-8").splitlines():
        body = line.split("#", 1)[0].strip()
        if ";" not in body:
            continue
        code_range, property_name = (part.strip() for part in body.split(";", 1))
        if property_name == "Default_Ignorable_Code_Point":
            ignored_ranges.append(_range(code_range))

    script_ranges = []
    for line in (_DATA / "Scripts.txt").read_text(encoding="utf-8").splitlines():
        body = line.split("#", 1)[0].strip()
        if ";" not in body:
            continue
        code_range, script = (part.strip() for part in body.split(";", 1))
        start, end = _range(code_range)
        script_ranges.append((start, end, script.upper()))
    script_ranges.sort()
    script_extension_ranges = _read_script_extensions()
    return _UnicodeData(
        mappings=mappings,
        mapping_lengths=tuple(sorted({len(key) for key in mappings}, reverse=True)),
        ignored_ranges=tuple(sorted(ignored_ranges)),
        script_ranges=tuple(script_ranges),
        script_starts=tuple(item[0] for item in script_ranges),
        script_extension_ranges=script_extension_ranges,
        script_extension_starts=tuple(
            item[0] for item in script_extension_ranges
        ),
    )


def _read_script_extensions() -> tuple[tuple[int, int, tuple[str, ...]], ...]:
    ranges = []
    for line in (_DATA / "ScriptExtensions.txt").read_text(encoding="utf-8").splitlines():
        body = line.split("#", 1)[0].strip()
        if ";" not in body:
            continue
        code_range, scripts = (part.strip() for part in body.split(";", 1))
        start, end = _range(code_range)
        ranges.append((start, end, tuple(script.upper() for script in scripts.split())))
    ranges.sort()
    return tuple(ranges)

def _in_ranges(codepoint: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    index = bisect.bisect_right(ranges, (codepoint, 0x110000)) - 1
    return index >= 0 and ranges[index][0] <= codepoint <= ranges[index][1]

def is_default_ignorable(char: str) -> bool:
    if len(char) != 1:
        raise ValueError("Expected one Unicode character")
    return _in_ranges(ord(char), _unicode_data().ignored_ranges)

def script_of(char: str) -> str | None:
    """Return the Unicode Script value, excluding Common/Inherited/Unknown."""
    if len(char) != 1:
        raise ValueError("Expected one Unicode character")
    data = _unicode_data()
    index = bisect.bisect_right(data.script_starts, ord(char)) - 1
    if index < 0:
        return None
    start, end, script = data.script_ranges[index]
    if not start <= ord(char) <= end or script in {"COMMON", "INHERITED", "UNKNOWN"}:
        return None
    return script

def script_extensions_of(char: str) -> frozenset[str]:
    """Return substantive Script_Extensions, ignoring Common/Inherited."""
    script = script_of(char)
    data = _unicode_data()
    codepoint = ord(char)
    index = bisect.bisect_right(data.script_extension_starts, codepoint) - 1
    if index >= 0:
        start, end, scripts = data.script_extension_ranges[index]
        if start <= codepoint <= end:
            return frozenset(scripts) - {"COMMON", "INHERITED", "UNKNOWN"}
    return frozenset({script}) if script else frozenset()

def _canonical_order(chars: list[tuple[str, tuple[int, int]]]) -> list[tuple[str, tuple[int, int]]]:
    """Apply canonical combining-class ordering while carrying source spans."""
    output: list[tuple[str, tuple[int, int]]] = []
    segment_start = 0
    for item in chars:
        combining = unicodedata.combining(item[0])
        if combining == 0:
            output.append(item)
            segment_start = len(output)
        else:
            insert_at = len(output)
            while insert_at > segment_start and unicodedata.combining(output[insert_at - 1][0]) > combining:
                insert_at -= 1
            output.insert(insert_at, item)
    return output

def _longest_mapping(
    codepoints: tuple[int, ...], position: int, data: _UnicodeData
) -> tuple[tuple[int, ...], int] | None:
    for length in data.mapping_lengths:
        end = position + length
        if end <= len(codepoints):
            mapped = data.mappings.get(codepoints[position:end])
            if mapped is not None:
                return mapped, end
    return None

def skeleton_with_offsets(text: str) -> DetectionText:
    """Compute UTS #39 internal skeleton and preserve a map to original text."""
    data = _unicode_data()
    decomposed: list[tuple[str, tuple[int, int]]] = []
    for index, char in enumerate(text):
        for part in unicodedata.normalize("NFD", char):
            decomposed.append((part, (index, index + 1)))
    decomposed = _canonical_order(decomposed)
    visible = [item for item in decomposed if not is_default_ignorable(item[0])]

    mapped: list[tuple[str, tuple[int, int]]] = []
    codepoints = tuple(ord(char) for char, _ in visible)
    index = 0
    while index < len(visible):
        match = _longest_mapping(codepoints, index, data)
        if match is None:
            value, end = (codepoints[index],), index + 1
        else:
            value, end = match
        source_span = (
            min(span[0] for _, span in visible[index:end]),
            max(span[1] for _, span in visible[index:end]),
        )
        for codepoint in value:
            for part in unicodedata.normalize("NFD", chr(codepoint)):
                mapped.append((part, source_span))
        index = end

    mapped = _canonical_order(mapped)
    return DetectionText(
        text="".join(char for char, _ in mapped),
        source_spans=tuple(span for _, span in mapped),
    )

def skeleton(text: str) -> str:
    """Return a comparison-only UTS #39 skeleton."""
    return skeleton_with_offsets(text).text

def security_detection_view(text: str) -> DetectionText:
    """Make a regex-friendly view without applying UTS skeleton mappings to ASCII.

    UTS skeletons deliberately map some ASCII characters (for example ``m`` to
    ``rn``). That is useful for identifier comparison but breaks literal
    detector expressions, so detector matching applies official confusable
    mappings only to non-ASCII input while preserving exact source offsets.
    """
    output: list[str] = []
    spans: list[tuple[int, int]] = []
    for index, char in enumerate(text):
        if is_default_ignorable(char):
            continue
        folded = (
            char.casefold()
            if char.isascii()
            else skeleton(char).casefold()
        )
        output.extend(folded)
        spans.extend([(index, index + 1)] * len(folded))
    return DetectionText("".join(output), tuple(spans))

def mixed_script_tokens(text: str) -> tuple[tuple[int, int, tuple[str, ...]], ...]:
    """Find alphanumeric tokens containing more than one substantive script."""
    results: list[tuple[int, int, tuple[str, ...]]] = []
    start: int | None = None
    token_script_intersection: set[str] | None = None
    token_scripts: set[str] = set()
    for index, char in enumerate(text + " "):
        category = unicodedata.category(char) if char != " " else "Zs"
        token_char = category[0] in {"L", "N", "M"} or (start is not None and category == "Cf")
        if token_char:
            if start is None:
                start = index
            scripts = set(script_extensions_of(char))
            if scripts:
                token_scripts.update(scripts)
                token_script_intersection = (
                    scripts
                    if token_script_intersection is None
                    else token_script_intersection.intersection(scripts)
                )
        elif start is not None:
            if token_script_intersection is not None and not token_script_intersection:
                results.append((start, index, tuple(sorted(token_scripts))))
            start = None
            token_script_intersection = None
            token_scripts.clear()
    return tuple(results)
