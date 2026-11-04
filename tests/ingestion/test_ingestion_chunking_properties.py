"""Property-based invariants for BLP-1, independent of PDF parser runtimes."""

from __future__ import annotations

from collections import Counter

import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.strategies import SearchStrategy

from src.ingestion.chunking import ChunkingConfig, ChunkingResult, chunk_document
from src.ingestion.parser_contracts import (
    CanonicalBlock,
    CanonicalDocument,
    CanonicalPage,
    ParserFailure,
    SourceCoordinates,
)
from src.ingestion.segmentation import _grapheme_boundaries, line_boundaries


PROPERTY_SETTINGS = settings(
    max_examples=100,
    derandomize=True,
    database=None,
    deadline=None,
)

_COORDINATES = SourceCoordinates(
    x0=0,
    y0=0,
    x1=10,
    y1=10,
    unit="PDF_POINT",
    origin="TOP_LEFT",
)


def legal_text() -> SearchStrategy[str]:
    """Short legal-like prose, markers, punctuation, and line structure."""
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?()§\t\n\r"
    return st.one_of(
        st.text(alphabet=alphabet, min_size=0, max_size=160),
        st.sampled_from(
            [
                "Section 5 Governing law.\nThe agreement applies.",
                "1. First clause.\n1.1 Subclause.",
                "Article IV\nThe provision applies.\n\nNext paragraph.",
                "(a) First item\n(iv) Fourth item\n(1) Numbered item",
                "Dr. Smith signed. The U.S. company filed.",
            ]
        ),
    )


def unicode_text() -> SearchStrategy[str]:
    """Bounded strings containing common combining, emoji, and line cases."""
    fragments = (
        "A",
        "e\u0301",
        "❤️",
        "👍🏽",
        "👩\u200d⚖️",
        "क्\u200dष",
        "क्\u200cष",
        "漢字",
        "\r\n",
        "\n",
        "\r",
        "\u2028",
        " ",
        ". ",
    )
    return st.lists(st.sampled_from(fragments), min_size=0, max_size=36).map("".join)


def table_text() -> SearchStrategy[str]:
    """Rows with blank lines, mixed terminators, columns, and long cells."""
    cell = st.one_of(
        st.text(
            alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,-",
            min_size=0,
            max_size=44,
        ),
        st.sampled_from(["x" * 16, "y" * 24, "plain", "(a)", "§ 4"]),
    )
    gap = st.sampled_from(["\t", "\t\t", "  ", " | ", " |  ", " ", ","])

    @st.composite
    def table_row(draw: st.DrawFn) -> str:
        cell_count = draw(st.integers(min_value=1, max_value=4))
        cells = [draw(cell) for _ in range(cell_count)]
        delimiters = [draw(gap) for _ in range(cell_count - 1)]
        value = cells[0]
        for delimiter, next_cell in zip(delimiters, cells[1:]):
            value += delimiter + next_cell
        return value

    @st.composite
    def generated(draw: st.DrawFn) -> str:
        first_row = draw(st.sampled_from(["", "  "])) + draw(table_row())
        extra_rows = draw(
            st.lists(
                st.one_of(
                    st.just(""),
                    st.just("   "),
                    table_row(),
                ),
                min_size=0,
                max_size=4,
            )
        )
        rows = [first_row, *extra_rows]
        endings = [draw(st.sampled_from(["\n", "\r\n", "\r"])) for _ in rows[1:]]
        text = rows[0]
        for ending, row_value in zip(endings, rows[1:]):
            text += ending + row_value
        return text

    return generated()


def _text_for_type(block_type: str) -> SearchStrategy[str]:
    if block_type == "TABLE":
        return table_text()
    if block_type == "IMAGE":
        return unicode_text()
    if block_type == "VECTOR":
        return st.one_of(legal_text(), st.just(""))
    return st.one_of(legal_text(), unicode_text())


@st.composite
def canonical_block(
    draw: st.DrawFn,
    page_number: int = 1,
    reading_order: int = 0,
    block_type: str | None = None,
) -> CanonicalBlock:
    """Generate one block with valid canonical page/order metadata."""
    selected_type = block_type or draw(
        st.sampled_from(["TEXT", "OTHER", "TABLE", "IMAGE", "VECTOR"])
    )
    text = draw(_text_for_type(selected_type))
    return CanonicalBlock(
        page_number=page_number,
        reading_order=reading_order,
        block_type=selected_type,  # type: ignore[arg-type]
        text=text,
        source_coordinates=_COORDINATES,
    )


@st.composite
def canonical_document(draw: st.DrawFn) -> CanonicalDocument:
    """Generate parser-contract-valid documents with one to three pages."""
    page_count = draw(st.integers(min_value=1, max_value=3))
    pages: list[CanonicalPage] = []
    for page_number in range(1, page_count + 1):
        status = draw(
            st.sampled_from(["PARSED", "BLANK", "VECTOR_ONLY", "FAILED"])
        )
        if status == "BLANK":
            page = CanonicalPage(
                page_number=page_number, parser="hypothesis", status="BLANK"
            )
        elif status == "FAILED":
            failure = ParserFailure(
                parser="hypothesis", code="generated_failure", message="failed"
            )
            warnings = draw(
                st.lists(
                    st.builds(
                        ParserFailure,
                        parser=st.just("hypothesis"),
                        code=st.just("generated_warning"),
                        message=st.just("warning"),
                    ),
                    max_size=1,
                )
            )
            page = CanonicalPage(
                page_number=page_number,
                parser="hypothesis",
                status="FAILED",
                failure=failure,
                warnings=tuple(warnings),
            )
        else:
            block_count = draw(st.integers(min_value=1, max_value=4))
            if status == "VECTOR_ONLY":
                blocks = tuple(
                    draw(
                        canonical_block(
                            page_number, order, block_type="VECTOR"
                        )
                    )
                    for order in range(block_count)
                )
            else:
                kinds = draw(
                    st.lists(
                        st.sampled_from(["TEXT", "OTHER", "TABLE", "IMAGE", "VECTOR"]),
                        min_size=block_count,
                        max_size=block_count,
                    )
                )
                blocks = tuple(
                    draw(canonical_block(page_number, order, kind))
                    for order, kind in enumerate(kinds)
                )
            page = CanonicalPage(
                page_number=page_number,
                parser="hypothesis",
                status=status,
                blocks=blocks,
            )
        pages.append(page)

    can_process = any(page.status in {"PARSED", "VECTOR_ONLY"} for page in pages)
    disposition = "PROCESS" if can_process else draw(
        st.sampled_from(["PROCESS", "REJECT_NO_CONTENT"])
    )
    return CanonicalDocument(
        page_count=page_count,
        document_disposition=disposition,
        pages=tuple(pages),
    )


def chunking_config(*, include_image_text: bool | None = None) -> SearchStrategy[ChunkingConfig]:
    """Generate valid behavior-changing configurations under small budgets."""
    image_setting = (
        st.booleans()
        if include_image_text is None
        else st.just(include_image_text)
    )
    return st.builds(
        ChunkingConfig,
        max_chunk_chars=st.one_of(
            st.sampled_from([16, 17, 24, 32, 48]),
            st.integers(min_value=16, max_value=64),
        ),
        synthetic_separator=st.sampled_from(["\n\n", " ", " :: ", "--SEP--"]),
        marker_breaks=st.booleans(),
        marker_break_min_pct=st.sampled_from([0, 25, 50, 75, 100]),
        grapheme_backoff_chars=st.sampled_from([0, 4, 16, 32]),
        overlap_chars=st.sampled_from([0, 4, 8, 100]),
        include_image_text=image_setting,
    )


def _eligible(block: CanonicalBlock, config: ChunkingConfig, page_status: str) -> bool:
    if page_status != "PARSED" or not block.text.strip():
        return False
    if block.block_type in {"TEXT", "OTHER", "TABLE"}:
        return True
    return block.block_type == "IMAGE" and config.include_image_text


def _assert_block_coverage(
    result: ChunkingResult,
    block: CanonicalBlock,
    config: ChunkingConfig,
) -> None:
    ranges = [
        (
            source.start_offset,
            source.end_offset,
        )
        for chunk in result.document.chunks
        for source in chunk.sources
        if source.block_id == block.block_id
    ]
    cursor = 0
    reconstructed: list[str] = []
    allowed_overlap = (
        min(config.overlap_chars, config.max_chunk_chars // 4)
        if block.block_type in {"TEXT", "OTHER"}
        else 0
    )
    for start, end in ranges:
        assert start is not None and end is not None
        assert 0 <= start < end <= len(block.text)
        assert start <= cursor
        assert cursor - start <= allowed_overlap
        reconstructed.append(block.text[max(start, cursor) : end])
        cursor = max(cursor, end)
    assert cursor == len(block.text)
    assert "".join(reconstructed) == block.text

def _assert_result_invariants(
    document: CanonicalDocument,
    result: ChunkingResult,
    config: ChunkingConfig,
) -> None:
    chunks = result.document.chunks
    assert all(len(chunk.text) <= config.max_chunk_chars for chunk in chunks)
    assert tuple(chunk.index for chunk in chunks) == tuple(range(len(chunks)))
    assert len(result.report.pages) == document.page_count
    assert tuple(page.page_number for page in result.report.pages) == tuple(
        range(1, document.page_count + 1)
    )
    assert result.report.diagnostics.pages_processed == len(result.report.pages)
    assert result.report.diagnostics.chunks_created == len(chunks)
    assert result.report.diagnostics.skipped_blocks == len(result.report.skipped_blocks)
    assert result.report.diagnostics.whitespace_only_blocks == sum(
        item.reason == "WHITESPACE_ONLY_TEXT"
        for item in result.report.skipped_blocks
    )
    assert result.report.diagnostics.whitespace_only_chunks == sum(
        not chunk.text.strip() for chunk in chunks
    )
    assert sum(page.chunks_created for page in result.report.pages) == len(chunks)

    source_blocks = {
        block.block_id: block for page in document.pages for block in page.blocks
    }
    for chunk in chunks:
        page_numbers = {source.page_number for source in chunk.sources}
        assert len(page_numbers) == 1
        source_order = tuple(
            (source.page_number, source.block_reading_order, source.start_offset)
            for source in chunk.sources
        )
        assert source_order == tuple(sorted(source_order))
        rebuilt = []
        previous_block_id: str | None = None
        for source in chunk.sources:
            assert source.start_offset is not None and source.end_offset is not None
            assert source.block_id in source_blocks
            source_block = source_blocks[source.block_id]
            assert source.page_number == source_block.page_number
            assert source.block_reading_order == source_block.reading_order
            assert source.block_type == source_block.block_type
            assert source.source_text_length == len(source_block.text)
            assert 0 <= source.start_offset < source.end_offset <= len(source_block.text)
            if previous_block_id is not None and source.block_id != previous_block_id:
                rebuilt.append(config.synthetic_separator)
            rebuilt.append(
                source_block.text[source.start_offset : source.end_offset]
            )
            previous_block_id = source.block_id
        assert "".join(rebuilt) == chunk.text

    expected_eligible_count = sum(
        _eligible(block, config, page.status)
        for page in document.pages
        for block in page.blocks
    )
    assert result.report.diagnostics.eligible_blocks == expected_eligible_count
    for page in document.pages:
        report = result.report.pages[page.page_number - 1]
        page_chunks = [
            chunk for chunk in chunks
            if chunk.sources[0].page_number == page.page_number
        ]
        assert report.chunks_created == len(page_chunks)
        if page.status == "FAILED":
            assert report.outcome == "SKIPPED_FAILED"
            assert report.failure == page.failure
            assert report.warnings == page.warnings
        elif page.status == "BLANK":
            assert report.outcome == "SKIPPED_BLANK"
        elif page.status == "VECTOR_ONLY":
            assert report.outcome == "SKIPPED_VECTOR_ONLY"
        elif page_chunks:
            assert report.outcome == "CHUNKED"
        elif page.status == "PARSED":
            if page.blocks and all(
                block.block_type == "IMAGE" and not block.text.strip()
                for block in page.blocks
            ):
                assert report.outcome == "SKIPPED_IMAGE_NO_TEXT"
            else:
                assert report.outcome == "SKIPPED_NO_ELIGIBLE_TEXT"

    skipped_by_id = {item.block_id: item for item in result.report.skipped_blocks}
    for page in document.pages:
        for block in page.blocks:
            if _eligible(block, config, page.status):
                _assert_block_coverage(result, block, config)
                assert block.block_id not in skipped_by_id
                continue
            if page.status != "PARSED":
                continue
            expected_reason = (
                "VECTOR_NOT_TEXT_ELIGIBLE"
                if block.block_type == "VECTOR"
                else "IMAGE_TEXT_DISABLED"
                if block.block_type == "IMAGE" and block.text.strip()
                else "IMAGE_NO_USABLE_TEXT"
                if block.block_type == "IMAGE"
                else "EMPTY_TEXT"
                if block.text == ""
                else "WHITESPACE_ONLY_TEXT"
                if not block.text.strip()
                else None
            )
            if expected_reason is not None:
                assert skipped_by_id[block.block_id].reason == expected_reason

    failed = any(page.status == "FAILED" for page in document.pages)
    unsupported = any(
        item.reason == "UNSUPPORTED_M5_BLOCK_TYPE"
        for item in result.report.skipped_blocks
    )
    if unsupported or (chunks and failed):
        assert result.report.outcome == "PARTIAL"
    elif chunks:
        assert result.report.outcome == "COMPLETED"
    elif failed:
        assert result.report.outcome == "FAILED"
    else:
        assert result.report.outcome == "NO_CONTENT"


@pytest.mark.parametrize(
    ("block_type", "text_strategy"),
    [
        ("TEXT", legal_text()),
        ("OTHER", unicode_text()),
        ("TABLE", table_text()),
        ("IMAGE", unicode_text()),
    ],
)
@PROPERTY_SETTINGS
@given(data=st.data(), config=chunking_config(include_image_text=True))
def test_p1_each_eligible_block_type_has_exact_source_coverage(
    block_type: str, text_strategy: SearchStrategy[str], data, config
) -> None:
    text = data.draw(text_strategy)
    if not text.strip():
        text = "eligible source text"
    source_block = CanonicalBlock(
        page_number=1,
        reading_order=0,
        block_type=block_type,  # type: ignore[arg-type]
        text=text,
        source_coordinates=_COORDINATES,
    )
    document = CanonicalDocument(
        page_count=1,
        document_disposition="PROCESS",
        pages=(
            CanonicalPage(
                page_number=1,
                parser="hypothesis",
                status="PARSED",
                blocks=(source_block,),
            ),
        ),
    )
    result = chunk_document(document, "property-doc", config)

    _assert_block_coverage(result, source_block, config)
    _assert_result_invariants(document, result, config)


@PROPERTY_SETTINGS
@given(
    whitespace=st.sampled_from(["", " ", "\t\n"]),
    config=chunking_config(include_image_text=False),
)
def test_p1_disabled_images_whitespace_and_vectors_follow_skip_contract(
    whitespace, config
):
    blocks = (
        CanonicalBlock(
            page_number=1,
            reading_order=0,
            block_type="IMAGE",
            text="OCR is disabled",
            source_coordinates=_COORDINATES,
        ),
        CanonicalBlock(
            page_number=1,
            reading_order=1,
            block_type="OTHER",
            text=whitespace,
            source_coordinates=_COORDINATES,
        ),
        CanonicalBlock(
            page_number=1,
            reading_order=2,
            block_type="VECTOR",
            text="vector label",
            source_coordinates=_COORDINATES,
        ),
    )
    document = CanonicalDocument(
        page_count=1,
        document_disposition="PROCESS",
        pages=(
            CanonicalPage(
                page_number=1,
                parser="hypothesis",
                status="PARSED",
                blocks=blocks,
            ),
        ),
    )
    result = chunk_document(document, "skip-contract-doc", config)

    assert result.document.chunks == ()
    assert result.report.outcome == "NO_CONTENT"
    assert {item.reason for item in result.report.skipped_blocks} == {
        "IMAGE_TEXT_DISABLED",
        "EMPTY_TEXT" if whitespace == "" else "WHITESPACE_ONLY_TEXT",
        "VECTOR_NOT_TEXT_ELIGIBLE",
    }


@PROPERTY_SETTINGS
@given(document=canonical_document(), config=chunking_config())
def test_p2_all_generated_chunks_respect_small_character_budgets(document, config):
    result = chunk_document(document, "property-doc", config)
    assert all(len(chunk.text) <= config.max_chunk_chars for chunk in result.document.chunks)
    _assert_result_invariants(document, result, config)


@PROPERTY_SETTINGS
@given(
    first=st.text(alphabet="abc XYZ", min_size=1, max_size=12),
    second=st.text(alphabet="def 123", min_size=1, max_size=12),
    config=chunking_config(),
)
def test_p1_synthetic_separators_count_toward_limit_without_source_ranges(
    first, second, config
):
    blocks = (
        CanonicalBlock(
            page_number=1,
            reading_order=0,
            block_type="TEXT",
            text=first,
            source_coordinates=_COORDINATES,
        ),
        CanonicalBlock(
            page_number=1,
            reading_order=1,
            block_type="OTHER",
            text=second,
            source_coordinates=_COORDINATES,
        ),
    )
    document = CanonicalDocument(
        page_count=1,
        document_disposition="PROCESS",
        pages=(
            CanonicalPage(
                page_number=1,
                parser="hypothesis",
                status="PARSED",
                blocks=blocks,
            ),
        ),
    )
    result = chunk_document(document, "separator-property-doc", config)

    _assert_result_invariants(document, result, config)
    for chunk in result.document.chunks:
        source_length = sum(
            source.end_offset - source.start_offset
            for source in chunk.sources
        )
        separator_length = sum(
            len(config.synthetic_separator)
            for left, right in zip(chunk.sources, chunk.sources[1:])
            if left.block_id != right.block_id
        )
        assert len(chunk.text) - source_length == separator_length


@PROPERTY_SETTINGS
@given(document=canonical_document(), config=chunking_config())
def test_p3_identical_input_and_config_produce_identical_chunks_and_report(document, config):
    first = chunk_document(document, "stable-property-doc", config)
    second = chunk_document(document, "stable-property-doc", config)
    assert first == second


@PROPERTY_SETTINGS
@given(document=canonical_document(), config=chunking_config())
def test_p4_chunk_metadata_and_report_counts_are_consistent(document, config):
    result = chunk_document(document, "metadata-property-doc", config)
    _assert_result_invariants(document, result, config)


@PROPERTY_SETTINGS
@given(document=canonical_document(), config=chunking_config())
def test_p5_page_and_block_groups_remain_isolated(document, config):
    result = chunk_document(document, "isolation-property-doc", config)
    blocks = {block.block_id: block for page in document.pages for block in page.blocks}

    for chunk in result.document.chunks:
        types = {source.block_type for source in chunk.sources}
        assert len({source.page_number for source in chunk.sources}) == 1
        if "TABLE" in types:
            assert types == {"TABLE"}
            assert len({source.block_id for source in chunk.sources}) == 1
        if "IMAGE" in types:
            assert types == {"IMAGE"}
            assert len({source.block_id for source in chunk.sources}) == 1
        assert types <= {"TEXT", "OTHER", "TABLE", "IMAGE"}
        for source in chunk.sources:
            assert source.block_type == blocks[source.block_id].block_type
    _assert_result_invariants(document, result, config)


@PROPERTY_SETTINGS
@given(text=table_text(), config=chunking_config())
def test_p6_table_ranges_preserve_rows_and_source_text(text, config):
    if not text.strip():
        text = "table row"
    header = "⚖BLP_HEADER⚖"
    text = (
        header
        + "\r\n"
        + "s" * (config.max_chunk_chars - 1)
        + "\n"
        + "e" * config.max_chunk_chars
        + "\r"
        + "l" * (config.max_chunk_chars + 1)
        + "\r\n"
        + text
    )
    table = CanonicalBlock(
        page_number=1,
        reading_order=0,
        block_type="TABLE",
        text=text,
        source_coordinates=_COORDINATES,
    )
    document = CanonicalDocument(
        page_count=1,
        document_disposition="PROCESS",
        pages=(
            CanonicalPage(
                page_number=1,
                parser="hypothesis",
                status="PARSED",
                blocks=(table,),
            ),
        ),
    )
    result = chunk_document(document, "table-property-doc", config)

    _assert_block_coverage(result, table, config)
    assert all(len(chunk.text) <= config.max_chunk_chars for chunk in result.document.chunks)
    assert sum(chunk.text.count(header) for chunk in result.document.chunks) == 1

    row_edges = (0, *line_boundaries(text), len(text))
    rows = tuple(
        (start, end)
        for start, end in zip(row_edges, row_edges[1:])
        if start < end
    )
    table_ranges = tuple(
        (chunk.sources[0].start_offset, chunk.sources[0].end_offset)
        for chunk in result.document.chunks
    )
    original_row_boundaries = set(row_edges)
    for start, end in table_ranges:
        for boundary in (start, end):
            if boundary in original_row_boundaries:
                continue
            containing_rows = [
                (row_start, row_end)
                for row_start, row_end in rows
                if row_start < boundary < row_end
            ]
            assert containing_rows
            assert all(
                row_end - row_start > config.max_chunk_chars
                for row_start, row_end in containing_rows
            )
    for row_start, row_end in rows:
        if row_end - row_start <= config.max_chunk_chars:
            assert sum(
                start <= row_start and row_end <= end
                for start, end in table_ranges
            ) == 1
    _assert_result_invariants(document, result, config)


@PROPERTY_SETTINGS
@given(text=unicode_text(), config=chunking_config())
def test_p7_unicode_and_line_boundaries_preserve_text_under_the_character_limit(text, config):
    if not text.strip():
        text = "e\u0301👩\u200d⚖️"
    source_block = CanonicalBlock(
        page_number=1,
        reading_order=0,
        block_type="TEXT",
        text=text,
        source_coordinates=_COORDINATES,
    )
    document = CanonicalDocument(
        page_count=1,
        document_disposition="PROCESS",
        pages=(
            CanonicalPage(
                page_number=1,
                parser="hypothesis",
                status="PARSED",
                blocks=(source_block,),
            ),
        ),
    )
    result = chunk_document(document, "unicode-property-doc", config)

    _assert_block_coverage(result, source_block, config)
    assert all(len(chunk.text) <= config.max_chunk_chars for chunk in result.document.chunks)
    _assert_result_invariants(document, result, config)


def test_p7_unsafe_bounded_grapheme_fallback_is_reported_without_data_loss():
    text = "a" + "\u0301" * 24 + "z" * 48
    config = ChunkingConfig(max_chunk_chars=16, grapheme_backoff_chars=0)
    source_block = CanonicalBlock(
        page_number=1,
        reading_order=0,
        block_type="TEXT",
        text=text,
        source_coordinates=_COORDINATES,
    )
    document = CanonicalDocument(
        page_count=1,
        document_disposition="PROCESS",
        pages=(
            CanonicalPage(
                page_number=1,
                parser="hypothesis",
                status="PARSED",
                blocks=(source_block,),
            ),
        ),
    )
    result = chunk_document(document, "unsafe-unicode-doc", config)

    _assert_block_coverage(result, source_block, config)
    assert all(len(chunk.text) <= config.max_chunk_chars for chunk in result.document.chunks)
    assert result.report.diagnostics.unsafe_grapheme_cuts > 0


@PROPERTY_SETTINGS
@given(
    cluster=st.sampled_from(
        ["e\u0301", "❤️", "👍🏽", "👩\u200d⚖️", "🇺🇳", "क्ष"]
    )
)
def test_p7_hard_cut_uses_available_safe_grapheme_boundaries(cluster):
    text = "x" * 15 + cluster + "q" * 48
    config = ChunkingConfig(
        max_chunk_chars=16,
        grapheme_backoff_chars=32,
        marker_breaks=False,
    )
    source_block = CanonicalBlock(
        page_number=1,
        reading_order=0,
        block_type="TEXT",
        text=text,
        source_coordinates=_COORDINATES,
    )
    document = CanonicalDocument(
        page_count=1,
        document_disposition="PROCESS",
        pages=(
            CanonicalPage(
                page_number=1,
                parser="hypothesis",
                status="PARSED",
                blocks=(source_block,),
            ),
        ),
    )
    result = chunk_document(document, "safe-unicode-doc", config)
    safe_boundaries = set(_grapheme_boundaries(text))

    _assert_block_coverage(result, source_block, config)
    assert all(
        source.start_offset in safe_boundaries
        and source.end_offset in safe_boundaries
        for chunk in result.document.chunks
        for source in chunk.sources
    )
    assert result.report.diagnostics.unsafe_grapheme_cuts == 0


@PROPERTY_SETTINGS
@given(document=canonical_document(), config=chunking_config())
def test_p8_page_outcomes_skips_and_diagnostics_match_generated_documents(document, config):
    result = chunk_document(document, "report-property-doc", config)
    _assert_result_invariants(document, result, config)


@PROPERTY_SETTINGS
@given(
    config=chunking_config(),
    skipped_text=st.sampled_from(["", " \t\n"]),
)
def test_p8_eligible_skipped_failed_and_blank_content_reports_consistently(
    config, skipped_text
):
    text_block = CanonicalBlock(
        page_number=1,
        reading_order=0,
        block_type="TEXT",
        text="eligible content",
        source_coordinates=_COORDINATES,
    )
    vector_block = CanonicalBlock(
        page_number=1,
        reading_order=1,
        block_type="VECTOR",
        text="vector label",
        source_coordinates=_COORDINATES,
    )
    whitespace_block = CanonicalBlock(
        page_number=1,
        reading_order=2,
        block_type="OTHER",
        text=skipped_text,
        source_coordinates=_COORDINATES,
    )
    failure = ParserFailure(parser="hypothesis", code="failed", message="failure")
    document = CanonicalDocument(
        page_count=3,
        document_disposition="PROCESS",
        pages=(
            CanonicalPage(
                page_number=1,
                parser="hypothesis",
                status="PARSED",
                blocks=(text_block, vector_block, whitespace_block),
            ),
            CanonicalPage(
                page_number=2,
                parser="hypothesis",
                status="FAILED",
                failure=failure,
            ),
            CanonicalPage(page_number=3, parser="hypothesis", status="BLANK"),
        ),
    )
    result = chunk_document(document, "report-mixed-doc", config)

    assert result.report.outcome == "PARTIAL"
    assert [page.outcome for page in result.report.pages] == [
        "CHUNKED",
        "SKIPPED_FAILED",
        "SKIPPED_BLANK",
    ]
    reasons = Counter(item.reason for item in result.report.skipped_blocks)
    assert reasons["VECTOR_NOT_TEXT_ELIGIBLE"] == 1
    assert reasons["EMPTY_TEXT" if skipped_text == "" else "WHITESPACE_ONLY_TEXT"] == 1
    _assert_result_invariants(document, result, config)


def test_grapheme_boundaries_are_code_point_offsets_not_grapheme_lengths():
    text = "e\u0301👍🏽👩\u200d⚖️\r\n"
    boundaries = _grapheme_boundaries(text)
    assert boundaries[0] == 0
    assert boundaries[-1] == len(text)
    assert len(text) > len(boundaries) - 1
