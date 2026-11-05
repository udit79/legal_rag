"""Local, deterministic G1/G2/G4/G5 batch screening."""

import asyncio
import hashlib
import re
import unicodedata
from functools import lru_cache
from urllib.parse import urlsplit

from src.ingestion.chunk_contracts import Chunk, ChunkedDocument
from src.security.ingestion_screening.contracts import (
    AdmissionRecord,
    ChunkScreeningResult,
    DocumentContext,
    FileScreeningRecord,
    ScreeningFlag,
    ScreeningReport,
    ScreenedChunkedDocument,
)
from src.security.ingestion_screening.enums import (
    AdmissionDecision,
    DetectionView,
    FlagSeverity,
    ScreeningStage,
)
from src.security.ingestion_screening.policy import ScreeningPolicy

_URL = re.compile(r"(?i)\bhttps?://[^\s<>\]\[(){}\"']+")
_BIDI_CONTROLS = frozenset(
    "\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e"
    "\u2066\u2067\u2068\u2069"
)
_ZERO_WIDTH_SUSPICIOUS = frozenset("\u200b\ufeff\u2060")
_ALLOWED_CONTROLS = frozenset("\n\r\t\f")
_LEGITIMATE_FORMAT_CONTROLS = _BIDI_CONTROLS | frozenset("\u200c\u200d")
_TAG_FIRST = 0xE0000
_TAG_LAST = 0xE007F
_VALID_FLAG_TAG_EMOJI = frozenset(
    "\U0001F3F4" + "".join(chr(0xE0000 + ord(char)) for char in code) + "\U000E007F"
    for code in ("eng", "sct", "wls")
)
_CONFUSABLES = str.maketrans(
    {
        "а": "a", "А": "a", "в": "b", "В": "b", "с": "c", "С": "c",
        "е": "e", "Е": "e", "һ": "h", "Н": "h", "і": "i", "І": "i",
        "ј": "j", "Ј": "j", "к": "k", "К": "k", "м": "m", "М": "m",
        "о": "o", "О": "o", "р": "p", "Р": "p", "ѕ": "s", "Ѕ": "s",
        "т": "t", "Т": "t", "х": "x", "Х": "x", "у": "y", "У": "y",
        "Α": "a", "α": "a", "Β": "b", "Ε": "e", "ε": "e", "Ι": "i",
        "ι": "i", "Κ": "k", "κ": "k", "Μ": "m", "Ν": "n", "Ο": "o",
        "ο": "o", "Ρ": "p", "ρ": "p", "Τ": "t", "τ": "t", "Χ": "x",
        "χ": "x", "Υ": "y", "υ": "y",
    }
)
_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


@lru_cache(maxsize=256)
def _compiled_pattern(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _tag_sequences(text: str) -> tuple[tuple[int, int, str, bool], ...]:
    sequences: list[tuple[int, int, str, bool]] = []
    index = 0
    while index < len(text):
        if not _TAG_FIRST <= ord(text[index]) <= _TAG_LAST:
            index += 1
            continue
        start = index
        while index < len(text) and _TAG_FIRST <= ord(text[index]) <= _TAG_LAST:
            index += 1
        end = index
        valid_emoji = start > 0 and text[start - 1 : end] in _VALID_FLAG_TAG_EMOJI
        decoded = "".join(
            chr(ord(char) - _TAG_FIRST)
            for char in text[start:end]
            if 0xE0020 <= ord(char) <= 0xE007E
        )
        sequences.append((start, end, decoded, valid_emoji))
    return tuple(sequences)


def _legitimate_joiner_context(text: str, index: int) -> bool:
    char = text[index]
    previous = text[index - 1] if index else ""
    following = text[index + 1] if index + 1 < len(text) else ""
    previous_script = _script(previous) if previous else None
    following_script = _script(following) if following else None
    if previous_script == following_script and previous_script in {"ARABIC", "DEVANAGARI"}:
        return True
    return (
        char == "\u200d"
        and bool(previous and following)
        and unicodedata.category(previous) == "So"
        and unicodedata.category(following) == "So"
    )


def _suspicious_zero_width_at(text: str, index: int) -> bool:
    char = text[index]
    return char in _ZERO_WIDTH_SUSPICIOUS or (
        char in {"\u200c", "\u200d"}
        and not _legitimate_joiner_context(text, index)
    )


def _normalized_detection_view(text: str) -> tuple[str, tuple[int, ...]]:
    normalized: list[str] = []
    offsets: list[int] = []
    tag_replacements: dict[int, str] = {}
    tag_indexes: set[int] = set()
    for start, end, _, valid_emoji in _tag_sequences(text):
        tag_indexes.update(range(start, end))
        if not valid_emoji:
            for index in range(start, end):
                codepoint = ord(text[index])
                if 0xE0020 <= codepoint <= 0xE007E:
                    tag_replacements[index] = chr(codepoint - _TAG_FIRST)
    for original_index, char in enumerate(text):
        if original_index in tag_indexes:
            replacement = tag_replacements.get(original_index)
            if replacement:
                normalized.append(replacement.casefold().translate(_CONFUSABLES))
                offsets.append(original_index)
            continue
        if char in _BIDI_CONTROLS or _suspicious_zero_width_at(text, original_index):
            continue
        folded = unicodedata.normalize("NFKC", char).casefold().translate(_CONFUSABLES)
        normalized.extend(folded)
        offsets.extend([original_index] * len(folded))
    return "".join(normalized), tuple(offsets)


def _flag(
    code: str,
    stage: ScreeningStage,
    severity: FlagSeverity,
    view: DetectionView,
    start: int | None = None,
    end: int | None = None,
    *,
    related_chunk_index: int | None = None,
    text: str | None = None,
    urls: tuple[str, ...] = (),
    domains: tuple[str, ...] = (),
) -> ScreeningFlag:
    return ScreeningFlag(
        code=code,
        stage=stage,
        severity=severity,
        detection_view=view,
        start_offset=start,
        end_offset=end,
        related_chunk_index=related_chunk_index,
        matched_value=text,
        urls=urls,
        domains=domains,
    )


def _script(char: str) -> str | None:
    codepoint = ord(char)
    if 0x0041 <= codepoint <= 0x024F:
        return "LATIN"
    if 0x0370 <= codepoint <= 0x03FF:
        return "GREEK"
    if 0x0400 <= codepoint <= 0x052F:
        return "CYRILLIC"
    if 0x0590 <= codepoint <= 0x05FF:
        return "HEBREW"
    if 0x0600 <= codepoint <= 0x06FF:
        return "ARABIC"
    if 0x0900 <= codepoint <= 0x097F:
        return "DEVANAGARI"
    if char.isascii() and char.isalpha():
        return "LATIN"
    return None


def _integrity_flags(
    text: str, policy: ScreeningPolicy
) -> tuple[list[ScreeningFlag], bool]:
    flags: list[ScreeningFlag] = []
    rejected_tag_payload = False
    for index, char in enumerate(text):
        if _suspicious_zero_width_at(text, index):
            flags.append(
                _flag(
                    "SUSPICIOUS_ZERO_WIDTH_CHARACTER",
                    ScreeningStage.G1_INTEGRITY,
                    FlagSeverity.MEDIUM,
                    DetectionView.RAW,
                    index,
                    index + 1,
                    text=char,
                )
            )
        if char in _BIDI_CONTROLS:
            # A small number of directional marks is legitimate in multilingual
            # text; excessive or unbalanced controls are flagged below.
            continue
        category = unicodedata.category(char)
        if category in {"Cc", "Cs"} and char not in _ALLOWED_CONTROLS:
            flags.append(
                _flag(
                    "UNEXPECTED_CONTROL_CHARACTER",
                    ScreeningStage.G1_INTEGRITY,
                    FlagSeverity.HIGH,
                    DetectionView.RAW,
                    index,
                    index + 1,
                    text=char,
                )
            )

    bidi_positions = [i for i, char in enumerate(text) if char in _BIDI_CONTROLS]
    if len(bidi_positions) > policy.max_bidi_controls:
        flags.append(
            _flag(
                "EXCESSIVE_BIDI_CONTROLS",
                ScreeningStage.G1_INTEGRITY,
                FlagSeverity.HIGH,
                DetectionView.RAW,
                bidi_positions[0],
                bidi_positions[-1] + 1,
                text=text[bidi_positions[0] : bidi_positions[-1] + 1],
            )
        )
    else:
        bidi_stack: list[tuple[str, int]] = []
        unbalanced: list[int] = []
        for index, char in enumerate(text):
            if char in "\u202a\u202b\u202d\u202e\u2066\u2067\u2068":
                bidi_stack.append((char, index))
            elif char == "\u202c":
                if bidi_stack and bidi_stack[-1][0] in "\u202a\u202b\u202d\u202e":
                    bidi_stack.pop()
                else:
                    unbalanced.append(index)
            elif char == "\u2069":
                isolate_index = next(
                    (
                        stack_index
                        for stack_index in range(len(bidi_stack) - 1, -1, -1)
                        if bidi_stack[stack_index][0] in "\u2066\u2067\u2068"
                    ),
                    None,
                )
                if isolate_index is not None:
                    del bidi_stack[isolate_index:]
                else:
                    unbalanced.append(index)
        unbalanced.extend(index for _, index in bidi_stack)
        for index in unbalanced:
            flags.append(
                _flag(
                    "UNBALANCED_BIDI_CONTROL",
                    ScreeningStage.G1_INTEGRITY,
                    FlagSeverity.HIGH,
                    DetectionView.RAW,
                    index,
                    index + 1,
                    text=text[index],
                )
            )

    valid_tag_indexes: set[int] = set()
    for start, end, decoded, valid_emoji in _tag_sequences(text):
        if valid_emoji:
            valid_tag_indexes.update(range(start, end))
            continue
        flags.append(
            _flag(
                "UNICODE_TAG_PAYLOAD",
                ScreeningStage.G1_INTEGRITY,
                FlagSeverity.HIGH,
                DetectionView.RAW,
                start,
                end,
                text=decoded or text[start:end],
            )
        )
        if decoded and any(
            re.search(pattern, decoded)
            for pattern in policy.decoded_tag_instruction_patterns
        ):
            rejected_tag_payload = True

    zero_width_run = 0
    run_start = 0
    for index in range(len(text) + 1):
        suspicious = index < len(text) and _suspicious_zero_width_at(text, index)
        if suspicious:
            if zero_width_run == 0:
                run_start = index
            zero_width_run += 1
        else:
            if zero_width_run >= policy.max_zero_width_run and zero_width_run > 0:
                end = index
                flags.append(
                    _flag(
                        "EXCESSIVE_ZERO_WIDTH_RUN",
                        ScreeningStage.G1_INTEGRITY,
                        FlagSeverity.HIGH,
                        DetectionView.RAW,
                        run_start,
                        end,
                        text=text[run_start:end],
                    )
                )
            zero_width_run = 0

    total = len(text)
    nonprinting = sum(
        unicodedata.category(char).startswith("C")
        and char not in _ALLOWED_CONTROLS
        and char not in _LEGITIMATE_FORMAT_CONTROLS
        and index not in valid_tag_indexes
        for index, char in enumerate(text)
    )
    if total and nonprinting / total > policy.max_nonprinting_ratio:
        flags.append(
            _flag(
                "EXCESSIVE_NONPRINTING_RATIO",
                ScreeningStage.G1_INTEGRITY,
                FlagSeverity.HIGH,
                DetectionView.RAW,
                0,
                len(text),
                text=text,
            )
        )

    for match in _TOKEN_PATTERN.finditer(text):
        scripts = {_script(char) for char in match.group()}
        scripts.discard(None)
        if len(match.group()) >= policy.min_mixed_script_token_chars and len(scripts) > 1:
            flags.append(
                _flag(
                    "MIXED_SCRIPT_TOKEN",
                    ScreeningStage.G1_INTEGRITY,
                    FlagSeverity.MEDIUM,
                    DetectionView.RAW,
                    match.start(),
                    match.end(),
                    text=match.group(),
                )
            )

    base64_pattern = re.compile(
        rf"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{{{policy.min_encoded_blob_chars},}}={{0,2}}(?![A-Za-z0-9+/])"
    )
    hex_pattern = re.compile(
        rf"(?i)(?<![0-9a-f])[0-9a-f]{{{policy.min_encoded_blob_chars},}}(?![0-9a-f])"
    )
    for code, pattern in (
        ("LONG_BASE64_BLOB", base64_pattern),
        ("LONG_HEX_BLOB", hex_pattern),
    ):
        for match in pattern.finditer(text):
            flags.append(
                _flag(
                    code,
                    ScreeningStage.G1_INTEGRITY,
                    FlagSeverity.MEDIUM,
                    DetectionView.RAW,
                    match.start(),
                    match.end(),
                    text=match.group(),
                )
            )
    return flags, rejected_tag_payload


def _pattern_flags(
    text: str, policy: ScreeningPolicy
) -> tuple[list[ScreeningFlag], bool]:
    flags: list[ScreeningFlag] = []
    model_directed = False
    detection_text, offset_map = _normalized_detection_view(text)
    patterns = [
        ("MODEL_DIRECTED_INSTRUCTION", p, FlagSeverity.CRITICAL, True)
        for p in policy.model_directed_patterns
    ] + [
        ("SUSPICIOUS_MARKUP_OR_EXFILTRATION", p, FlagSeverity.HIGH, False)
        for p in policy.high_severity_patterns
    ] + [
        ("SUSPICIOUS_EMBEDDED_CONTENT", p, FlagSeverity.LOW, False)
        for p in policy.low_severity_patterns
    ]
    for code, source_pattern, severity, is_hard in patterns:
        pattern = _compiled_pattern(source_pattern)
        for match in pattern.finditer(detection_text):
            original_start = offset_map[match.start()]
            original_end = offset_map[match.end() - 1] + 1
            matched_raw = text[original_start:original_end]
            urls = tuple(url.group().rstrip(".,;:!?") for url in _URL.finditer(match.group()))
            domains = tuple(
                host
                for url in urls
                if (host := urlsplit(url).hostname) is not None
            )
            flags.append(
                _flag(
                    code,
                    ScreeningStage.G2_INJECTION,
                    severity,
                    DetectionView.NORMALIZED,
                    original_start,
                    original_end,
                    text=matched_raw,
                    urls=urls,
                    domains=domains,
                )
            )
            model_directed = model_directed or is_hard
    for match in _URL.finditer(detection_text):
        original_start = offset_map[match.start()]
        original_end = offset_map[match.end() - 1] + 1
        url = text[original_start:original_end].rstrip(".,;:!?")
        host = urlsplit(url).hostname
        flags.append(
            _flag(
                "URL_REFERENCE",
                ScreeningStage.G2_INJECTION,
                FlagSeverity.INFO,
                DetectionView.NORMALIZED,
                original_start,
                original_end,
                text=text[original_start:original_end],
                urls=(url,),
                domains=(host,) if host else (),
            )
        )
    return flags, model_directed


def _boundary_flags(
    source: Chunk,
    target: Chunk,
    policy: ScreeningPolicy,
) -> list[tuple[int, ScreeningFlag]]:
    tail_size = min(policy.boundary_window_chars, len(source.text))
    head_size = min(policy.boundary_window_chars, len(target.text))
    source_pages = {item.page_number for item in source.sources}
    target_pages = {item.page_number for item in target.sources}
    if len(source_pages) != 1 or len(target_pages) != 1:
        return []
    source_page = next(iter(source_pages))
    target_page = next(iter(target_pages))
    if target_page not in {source_page, source_page + 1}:
        return []
    tail_start = len(source.text) - tail_size
    source_view, source_map = _normalized_detection_view(source.text[tail_start:])
    target_view, target_map = _normalized_detection_view(target.text[:head_size])
    joined = source_view + target_view
    results: list[tuple[int, ScreeningFlag]] = []
    patterns = [
        ("MODEL_DIRECTED_INSTRUCTION", p, FlagSeverity.CRITICAL)
        for p in policy.model_directed_patterns
    ] + [
        ("SUSPICIOUS_MARKUP_OR_EXFILTRATION", p, FlagSeverity.HIGH)
        for p in policy.high_severity_patterns
    ] + [
        ("SUSPICIOUS_EMBEDDED_CONTENT", p, FlagSeverity.LOW)
        for p in policy.low_severity_patterns
    ] + [
        ("URL_REFERENCE", _URL.pattern, FlagSeverity.INFO)
    ]
    for code, source_pattern, severity in patterns:
        for match in _compiled_pattern(source_pattern).finditer(joined):
            start, end = match.span()
            if not (start < len(source_view) < end):
                continue
            matched_offsets = []
            for normalized_offset in range(start, end):
                if normalized_offset < len(source_map):
                    matched_offsets.append((source.index, tail_start + source_map[normalized_offset]))
                else:
                    target_offset = normalized_offset - len(source_map)
                    if target_offset < len(target_map):
                        matched_offsets.append((target.index, target_map[target_offset]))
            source_offsets = [offset for index, offset in matched_offsets if index == source.index]
            target_offsets = [offset for index, offset in matched_offsets if index == target.index]
            urls = (
                (match.group().rstrip(".,;:!?"),)
                if code == "URL_REFERENCE"
                else ()
            )
            domains = tuple(
                host
                for url in urls
                if (host := urlsplit(url).hostname) is not None
            )
            if source_offsets:
                local_start = min(source_offsets)
                local_end = max(source_offsets) + 1
                results.append(
                    (
                        source.index,
                        _flag(
                            code,
                            ScreeningStage.G2_INJECTION,
                            severity,
                            DetectionView.BOUNDARY,
                            local_start,
                            local_end,
                            related_chunk_index=target.index,
                            text=source.text[local_start:local_end],
                            urls=urls,
                            domains=domains,
                        ),
                    )
                )
            if target_offsets:
                target_start = min(target_offsets)
                target_end = max(target_offsets) + 1
                results.append(
                    (
                        target.index,
                        _flag(
                            code,
                            ScreeningStage.G2_INJECTION,
                            severity,
                            DetectionView.BOUNDARY,
                            target_start,
                            target_end,
                            related_chunk_index=source.index,
                            text=target.text[target_start:target_end],
                            urls=urls,
                            domains=domains,
                        ),
                    )
                )
    return results


class ScreeningService:
    """Runs one local screening pass over the complete original chunk batch."""

    def __init__(self, policy: ScreeningPolicy | None = None) -> None:
        self.policy = policy or ScreeningPolicy()

    async def screen_batch(
        self,
        chunks: tuple[Chunk, ...] | list[Chunk],
        document_context: DocumentContext,
        *,
        document_id: str,
        file_screening: FileScreeningRecord | None = None,
    ) -> ScreenedChunkedDocument:
        """Run the local batch screening pass without blocking the event loop."""
        return await asyncio.to_thread(
            self._screen_batch_sync,
            chunks,
            document_context,
            document_id=document_id,
            file_screening=file_screening,
        )

    def _screen_batch_sync(
        self,
        chunks: tuple[Chunk, ...] | list[Chunk],
        document_context: DocumentContext,
        *,
        document_id: str,
        file_screening: FileScreeningRecord | None = None,
    ) -> ScreenedChunkedDocument:
        """Screen all chunks in one invocation; no model or network calls occur."""
        ordered_chunks = tuple(chunks)
        if tuple(chunk.index for chunk in ordered_chunks) != tuple(
            range(len(ordered_chunks))
        ):
            raise ValueError("Screening input chunk indexes must be contiguous")
        if document_context.chunking_policy_version == "":
            raise ValueError("Chunking policy version is required")
        if file_screening is not None:
            if file_screening.decision != AdmissionDecision.ADMITTED:
                raise ValueError("A G0-blocked file cannot enter chunk screening")
            if file_screening.source_sha256 != document_context.source_sha256:
                raise ValueError("G0 and chunk screening source digests do not match")
        policy_digest = self.policy.digest()
        flags_by_index: dict[int, list[ScreeningFlag]] = {
            chunk.index: [] for chunk in ordered_chunks
        }
        model_directed_chunks: set[int] = set()
        rejected_tag_chunks: set[int] = set()
        for chunk in ordered_chunks:
            integrity_flags, has_rejected_tag_payload = _integrity_flags(
                chunk.text, self.policy
            )
            flags_by_index[chunk.index].extend(integrity_flags)
            if has_rejected_tag_payload:
                rejected_tag_chunks.add(chunk.index)
            pattern_flags, has_model_directed = _pattern_flags(
                chunk.text, self.policy
            )
            flags_by_index[chunk.index].extend(pattern_flags)
            if has_model_directed:
                model_directed_chunks.add(chunk.index)

        for source, target in zip(ordered_chunks, ordered_chunks[1:]):
            for chunk_index, flag in _boundary_flags(source, target, self.policy):
                flags_by_index[chunk_index].append(flag)
                if flag.code == "MODEL_DIRECTED_INSTRUCTION":
                    model_directed_chunks.add(chunk_index)

        duplicate_owner: dict[str, int] = {}
        for chunk in ordered_chunks:
            digest = _digest(chunk.text)
            if digest in duplicate_owner:
                flags_by_index[chunk.index].append(
                    _flag(
                        "EXACT_DUPLICATE_CHUNK",
                        ScreeningStage.G4_PROVENANCE,
                        FlagSeverity.LOW,
                        DetectionView.RAW,
                        text=f"duplicate_of_chunk:{duplicate_owner[digest]}",
                    )
                )
            else:
                duplicate_owner[digest] = chunk.index

        results: list[ChunkScreeningResult] = []
        for chunk in ordered_chunks:
            flags = tuple(flags_by_index[chunk.index])
            codes = {flag.code for flag in flags}
            if not document_context.has_required_provenance:
                flags += (
                    _flag(
                        "REQUIRED_PROVENANCE_MISSING",
                        ScreeningStage.G4_PROVENANCE,
                        FlagSeverity.HIGH,
                        DetectionView.RAW,
                        text="Required source, tenant, uploader or run ID is absent",
                    ),
                )
                codes.add("REQUIRED_PROVENANCE_MISSING")
            if chunk.index in rejected_tag_chunks:
                decision = AdmissionDecision.REJECTED
                justification = tuple(sorted(codes | {"DECODED_TAG_INSTRUCTION"}))
            elif any(
                flag.severity in {FlagSeverity.HIGH, FlagSeverity.CRITICAL}
                for flag in flags
            ):
                decision = AdmissionDecision.QUARANTINED
                justification = tuple(sorted(codes)) or ("HIGH_SEVERITY_FLAG",)
            elif not document_context.has_required_provenance:
                decision = AdmissionDecision.QUARANTINED
                justification = ("REQUIRED_PROVENANCE_MISSING",)
            elif any(
                flag.severity in {FlagSeverity.LOW, FlagSeverity.MEDIUM}
                for flag in flags
            ):
                decision = AdmissionDecision.ADMITTED_FLAGGED
                justification = tuple(sorted(codes))
            else:
                decision = AdmissionDecision.ADMITTED
                justification = tuple(sorted(codes)) or ("NO_SCREENING_FLAGS",)

            results.append(
                ChunkScreeningResult(
                    chunk_index=chunk.index,
                    chunk_digest=_digest(chunk.text),
                    chunk_text_length=len(chunk.text),
                    decision=decision,
                    flags=flags,
                    justification_codes=justification,
                    screening_policy_version=self.policy.version,
                    screening_policy_digest=policy_digest,
                    source_id=document_context.source_id,
                    tenant_id=document_context.tenant_id,
                    uploader_id=document_context.uploader_id,
                    ingestion_run_id=document_context.ingestion_run_id,
                    source_sha256=document_context.source_sha256,
                    canonical_document_digest=document_context.canonical_document_digest,
                    chunking_policy_version=document_context.chunking_policy_version,
                    trust_tier=document_context.trust_tier,
                )
            )

        flagged_chunk_count = sum(
            any(flag.severity != FlagSeverity.INFO for flag in flags)
            for flags in flags_by_index.values()
        )
        flagged_chunk_ratio = flagged_chunk_count / max(1, len(ordered_chunks))
        threshold = self.policy.max_flagged_chunk_ratio
        missing_provenance = not document_context.has_required_provenance
        escalation_codes = set()
        if model_directed_chunks:
            escalation_codes.add("MODEL_DIRECTED_INSTRUCTION_PRESENT")
        if rejected_tag_chunks:
            escalation_codes.add("DECODED_TAG_INSTRUCTION_PRESENT")
        if missing_provenance:
            escalation_codes.add("REQUIRED_PROVENANCE_MISSING")
        if (
            threshold is not None
            and flagged_chunk_count > 0
            and flagged_chunk_ratio >= threshold
        ):
            escalation_codes.add("FLAGGED_CHUNK_RATIO_EXCEEDED")
        document_quarantined = (
            bool(escalation_codes)
        )
        if document_quarantined:
            results = [
                result.model_copy(
                    update={
                        "decision": (
                            AdmissionDecision.QUARANTINED
                            if result.decision != AdmissionDecision.REJECTED
                            else result.decision
                        ),
                        "justification_codes": tuple(
                            sorted(set(result.justification_codes) | escalation_codes)
                        ),
                    }
                )
                for result in results
            ]

        report = ScreeningReport(
            document_id=document_id,
            input_chunk_count=len(ordered_chunks),
            chunk_results=tuple(results),
            screening_policy_version=self.policy.version,
            screening_policy_digest=policy_digest,
            quarantined_document=document_quarantined,
            file_screening=file_screening,
        )
        records = tuple(
            AdmissionRecord(
                chunk_index=result.chunk_index,
                chunk_digest=result.chunk_digest,
                decision=result.decision,
                source_id=result.source_id,
                tenant_id=result.tenant_id,
                uploader_id=result.uploader_id,
                ingestion_run_id=result.ingestion_run_id,
                source_sha256=result.source_sha256,
                canonical_document_digest=result.canonical_document_digest,
                chunking_policy_version=result.chunking_policy_version,
                screening_policy_version=result.screening_policy_version,
                screening_policy_digest=result.screening_policy_digest,
                trust_tier=result.trust_tier,
                justification_codes=result.justification_codes,
            )
            for result in results
        )
        screened_document = ChunkedDocument(
            document_id=document_id,
            source_page_count=document_context.source_page_count,
            policy_version=document_context.chunking_policy_version,
            chunks=ordered_chunks,
        )
        return ScreenedChunkedDocument(
            document=screened_document,
            admission_records=records,
            report=report,
        )
