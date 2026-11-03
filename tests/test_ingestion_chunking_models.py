import re

import pytest
from pydantic import ValidationError

from src.ingestion.chunk_contracts import ChunkedDocument
from src.ingestion.chunking import (
    CHUNKING_ALGORITHM_VERSION,
    ChunkingConfig,
    ChunkingDiagnostics,
    ChunkingReport,
    ChunkingResult,
    PageChunkingReport,
    SkippedBlock,
)
from src.ingestion.parser_contracts import CanonicalPage, ParserFailure


def _report(outcome, pages, config=None):
    config = config or ChunkingConfig()
    digest = config.digest()
    return ChunkingReport(
        outcome=outcome,
        policy_version=config.policy_version(),
        config_digest=digest,
        pages=tuple(pages),
        diagnostics=ChunkingDiagnostics(
            pages_processed=len(pages),
            chunks_created=sum(page.chunks_created for page in pages)
        ),
    )


def _failed_page(number=1):
    return PageChunkingReport(
        page_number=number,
        parser_status="FAILED",
        outcome="SKIPPED_FAILED",
        failure=ParserFailure(parser="test", code="parse_error", message="Failed"),
    )


def test_chunking_config_enforces_minimum_and_strict_integer():
    assert ChunkingConfig().max_chunk_chars == 1500
    assert ChunkingConfig().grapheme_backoff_chars == 32
    with pytest.raises(ValidationError):
        ChunkingConfig(grapheme_backoff_chars=1.5)
    with pytest.raises(ValidationError):
        ChunkingConfig(grapheme_backoff_chars=-1)
    for invalid in (15, 1, True, 16.0):
        with pytest.raises(ValidationError):
            ChunkingConfig(max_chunk_chars=invalid)


def test_config_boolean_is_strict_frozen_and_rejects_extra_fields():
    config = ChunkingConfig()
    with pytest.raises(ValidationError):
        ChunkingConfig(include_image_text="false")
    with pytest.raises(ValidationError):
        ChunkingConfig(unexpected_setting=True)
    with pytest.raises(ValidationError):
        config.include_image_text = False


def test_chunking_config_defaults_and_markers_cover_legal_boundaries():
    config = ChunkingConfig()
    assert config.include_image_text is True
    assert config.marker_breaks is True
    assert config.marker_break_min_pct == 25
    numeric = re.compile(config.provision_marker_patterns[0])
    for marker in ("1.", "1)", "1.1", "1.1.1"):
        assert numeric.search(marker)
    assert not numeric.search("2023. The court")
    assert not numeric.search("The rule in 1.1 applies")
    newline_match = numeric.search("Body text\n\n  1.1 Heading")
    assert newline_match is not None
    assert "\n" not in newline_match.group()
    assert newline_match.group().strip() == "1.1"
    keywords = re.compile(config.provision_marker_patterns[1])
    for label in (
        "Article IV",
        "Section 5",
        "Chapter II",
        "Part 2",
        "Schedule 3",
        "Annexure I",
        "Annex A",
        "Annexure A",
        "Appendix A",
        "Exhibit 4",
    ):
        assert keywords.search(label), label
    assert re.compile(config.provision_marker_patterns[2]).search("§ 3.2")
    assert not keywords.search("The court cited Article IV in its ruling")


def test_provision_marker_regex_rejects_invalid_patterns():
    with pytest.raises(ValidationError, match="Invalid provision marker pattern"):
        ChunkingConfig(provision_marker_patterns=("[",))


def test_config_digest_is_repeatable_order_independent_and_policy_is_stable():
    config = ChunkingConfig()
    reordered = ChunkingConfig.model_validate(dict(reversed(list(config.model_dump().items()))))
    changed = ChunkingConfig(max_chunk_chars=1600)

    assert config.digest() == reordered.digest()
    assert config.digest() != changed.digest()
    assert CHUNKING_ALGORITHM_VERSION == "1.2.0"
    assert config.policy_version() == f"5c-1.2.0+{config.digest()[:12]}"
    assert config.digest() == ChunkingConfig().digest()
    assert config.policy_version() == ChunkingConfig().policy_version()


@pytest.mark.parametrize(
    "change",
    [
        {"max_chunk_chars": 1600},
        {"synthetic_separator": "\n"},
        {"marker_breaks": False},
        {"marker_break_min_pct": 30},
        {"grapheme_backoff_chars": 12},
        {"provision_marker_patterns": ("marker",)},
        {"include_image_text": False},
        {"abbreviations": ("etc.",)},
    ],
)
def test_digest_changes_for_each_behavior_setting(change):
    assert ChunkingConfig(**change).digest() != ChunkingConfig().digest()


def test_policy_version_accepts_ascii_semver_and_rejects_non_ascii_digits():
    config = ChunkingConfig()
    assert config.policy_version("1.0.0").startswith("5c-1.0.0+")
    with pytest.raises(ValueError, match="MAJOR.MINOR.PATCH"):
        config.policy_version("١.٠.٠")

    with pytest.raises(ValidationError):
        ChunkingReport(
            outcome="NO_CONTENT",
            policy_version=f"5c-١.0.0+{config.digest()[:12]}",
            config_digest=config.digest(),
        )


def test_chunking_report_rejects_version_digest_mismatch():
    config = ChunkingConfig()
    valid = _report("NO_CONTENT", [])
    assert valid.policy_version.endswith(config.digest()[:12])

    with pytest.raises(ValidationError, match="digest"):
        ChunkingReport(
            outcome="NO_CONTENT",
            policy_version="5c-1.0.0+000000000000",
            config_digest=config.digest(),
        )

    with pytest.raises(ValidationError):
        ChunkingReport(
            outcome="NO_CONTENT",
            policy_version="chunk-v1",
            config_digest=config.digest(),
        )


@pytest.mark.parametrize(
    ("overall", "pages"),
    [
        ("COMPLETED", [PageChunkingReport(page_number=1, parser_status="PARSED", outcome="CHUNKED", chunks_created=1)]),
        ("NO_CONTENT", [PageChunkingReport(page_number=1, parser_status="BLANK", outcome="SKIPPED_BLANK")]),
        ("NO_CONTENT", [PageChunkingReport(page_number=1, parser_status="PARSED", outcome="SKIPPED_NO_ELIGIBLE_TEXT")]),
        ("NO_CONTENT", [PageChunkingReport(page_number=1, parser_status="PARSED", outcome="SKIPPED_IMAGE_NO_TEXT")]),
        ("NO_CONTENT", [PageChunkingReport(page_number=1, parser_status="VECTOR_ONLY", outcome="SKIPPED_VECTOR_ONLY")]),
        ("PARTIAL", [PageChunkingReport(page_number=1, parser_status="PARSED", outcome="CHUNKED", chunks_created=1), _failed_page(2)]),
        ("FAILED", [_failed_page()]),
    ],
)
def test_overall_outcome_semantics(overall, pages):
    assert _report(overall, pages).outcome == overall


def test_overall_outcome_rejects_inconsistent_status():
    page = PageChunkingReport(
        page_number=1,
        parser_status="PARSED",
        outcome="CHUNKED",
        chunks_created=1,
    )
    with pytest.raises(ValidationError, match="must be COMPLETED"):
        _report("NO_CONTENT", [page])


def test_report_rejects_processed_page_count_mismatch():
    with pytest.raises(ValidationError, match="Processed-page count"):
        ChunkingReport(
            outcome="NO_CONTENT",
            policy_version=ChunkingConfig().policy_version(),
            config_digest=ChunkingConfig().digest(),
            pages=(
                PageChunkingReport(
                    page_number=1,
                    parser_status="BLANK",
                    outcome="SKIPPED_BLANK",
                ),
            ),
            diagnostics=ChunkingDiagnostics(pages_processed=0),
        )


def test_report_rejects_diagnostic_chunk_count_mismatch():
    with pytest.raises(ValidationError, match="Diagnostic chunk count"):
        ChunkingReport(
            outcome="COMPLETED",
            policy_version=ChunkingConfig().policy_version(),
            config_digest=ChunkingConfig().digest(),
            pages=(
                PageChunkingReport(
                    page_number=1,
                    parser_status="PARSED",
                    outcome="CHUNKED",
                    chunks_created=1,
                ),
            ),
            diagnostics=ChunkingDiagnostics(pages_processed=1, chunks_created=0),
        )


def test_report_rejects_skipped_block_count_mismatch():
    skipped = SkippedBlock(
        page_number=1,
        block_id="page-1-block-0",
        block_type="TEXT",
        reason="Whitespace-only source block",
    )
    page = PageChunkingReport(
        page_number=1,
        parser_status="PARSED",
        outcome="SKIPPED_NO_ELIGIBLE_TEXT",
    )
    with pytest.raises(ValidationError, match="skipped-block count"):
        ChunkingReport(
            outcome="NO_CONTENT",
            policy_version=ChunkingConfig().policy_version(),
            config_digest=ChunkingConfig().digest(),
            pages=(page,),
            skipped_blocks=(skipped,),
            diagnostics=ChunkingDiagnostics(pages_processed=1, skipped_blocks=0),
        )


def test_report_accepts_matching_skipped_block_count():
    skipped = SkippedBlock(
        page_number=1,
        block_id="page-1-block-0",
        block_type="TEXT",
        reason="Whitespace-only source block",
    )
    page = PageChunkingReport(
        page_number=1,
        parser_status="PARSED",
        outcome="SKIPPED_NO_ELIGIBLE_TEXT",
    )
    report = ChunkingReport(
        outcome="NO_CONTENT",
        policy_version=ChunkingConfig().policy_version(),
        config_digest=ChunkingConfig().digest(),
        pages=(page,),
        skipped_blocks=(skipped,),
        diagnostics=ChunkingDiagnostics(pages_processed=1, skipped_blocks=1),
    )
    assert report.diagnostics.skipped_blocks == len(report.skipped_blocks)


def test_page_report_preserves_parser_failure_and_warnings():
    warning = ParserFailure(parser="test", code="partial", message="Partial output")
    page = PageChunkingReport(
        page_number=1,
        parser_status="FAILED",
        outcome="SKIPPED_FAILED",
        failure=ParserFailure(parser="test", code="failed", message="Parse failed"),
        warnings=(warning,),
    )

    assert page.failure.code == "failed"
    assert page.warnings == (warning,)
    assert CanonicalPage.model_fields["warnings"].annotation == tuple[
        ParserFailure, ...
    ]
    with pytest.raises(ValidationError, match="Only failed pages"):
        PageChunkingReport(
            page_number=2,
            parser_status="PARSED",
            outcome="SKIPPED_NO_ELIGIBLE_TEXT",
            failure=warning,
        )


def test_page_outcomes_match_parser_states():
    for status, outcome in (
        ("BLANK", "SKIPPED_BLANK"),
        ("VECTOR_ONLY", "SKIPPED_VECTOR_ONLY"),
        ("FAILED", "SKIPPED_FAILED"),
    ):
        kwargs = {"failure": _failed_page().failure} if status == "FAILED" else {}
        assert PageChunkingReport(
            page_number=1,
            parser_status=status,
            outcome=outcome,
            **kwargs,
        ).outcome == outcome


def test_chunking_result_requires_report_and_document_policy_versions_to_match():
    report = _report(
        "NO_CONTENT",
        [
            PageChunkingReport(
                page_number=1,
                parser_status="BLANK",
                outcome="SKIPPED_BLANK",
            )
        ],
    )
    document = ChunkedDocument(
        document_id="doc-1",
        source_page_count=1,
        policy_version=report.policy_version,
        chunks=(),
    )
    assert ChunkingResult(document=document, report=report).document == document

    mismatched = document.model_copy(update={"policy_version": "5c-1.0.0+000000000000"})
    with pytest.raises(ValidationError, match="policy versions must match"):
        ChunkingResult(document=mismatched, report=report)


def test_chunking_result_requires_complete_page_coverage():
    report = _report(
        "NO_CONTENT",
        [PageChunkingReport(page_number=1, parser_status="BLANK", outcome="SKIPPED_BLANK")],
    )
    document = ChunkedDocument(
        document_id="doc-1",
        source_page_count=2,
        policy_version=report.policy_version,
        chunks=(),
    )
    with pytest.raises(ValidationError, match="cover each document page"):
        ChunkingResult(document=document, report=report)


def test_chunking_result_requires_diagnostic_and_page_chunk_counts_to_match():
    page = PageChunkingReport(
        page_number=1,
        parser_status="PARSED",
        outcome="CHUNKED",
        chunks_created=1,
    )
    report = ChunkingReport(
        outcome="COMPLETED",
        policy_version=ChunkingConfig().policy_version(),
        config_digest=ChunkingConfig().digest(),
        pages=(page,),
        diagnostics=ChunkingDiagnostics(pages_processed=1, chunks_created=1),
    )
    document = ChunkedDocument(
        document_id="doc-1",
        source_page_count=1,
        policy_version=report.policy_version,
        chunks=(),
    )
    with pytest.raises(ValidationError, match="Reported chunk count"):
        ChunkingResult(document=document, report=report)
