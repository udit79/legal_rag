import pytest

import src.ingestion.chunking as chunking_module
from src.ingestion.chunk_contracts import ChunkedDocument
from src.ingestion.chunking import ChunkingConfig, ChunkingResult, chunk_document
from src.ingestion.parser_contracts import (
    CanonicalBlock,
    CanonicalDocument,
    CanonicalPage,
    ParserFailure,
    SourceCoordinates,
)


def block(page, order, kind, text):
    return CanonicalBlock(
        page_number=page,
        reading_order=order,
        block_type=kind,
        text=text,
        source_coordinates=SourceCoordinates(
            x0=0, y0=0, x1=10, y1=10, unit="PDF_POINT", origin="TOP_LEFT"
        ),
    )


def parsed_page(number, *blocks):
    return CanonicalPage(
        page_number=number,
        parser="test",
        status="PARSED",
        blocks=tuple(blocks),
    )


def document(*pages):
    return CanonicalDocument(
        page_count=len(pages),
        document_disposition="PROCESS",
        pages=tuple(pages),
    )


def run(*pages, config=None):
    return chunk_document(
        document(*pages),
        "legal-doc",
        config or ChunkingConfig(overlap_chars=0),
    )


def test_packs_one_block_and_multiple_blocks_with_synthetic_separator():
    one = run(parsed_page(1, block(1, 0, "TEXT", "one block")))
    several = run(
        parsed_page(
            1,
            block(1, 0, "TEXT", "first"),
            block(1, 1, "OTHER", "second"),
        )
    )

    assert isinstance(one, ChunkingResult)
    assert isinstance(one.document, ChunkedDocument)
    assert [chunk.text for chunk in one.document.chunks] == ["one block"]
    combined = several.document.chunks[0]
    assert combined.text == "first\n\nsecond"
    assert [source.block_id for source in combined.sources] == [
        "page-1-block-0",
        "page-1-block-1",
    ]


def test_closes_chunk_when_next_atom_would_exceed_character_budget():
    text = "abcdefghijklmnopqrstu"
    result = run(
        parsed_page(1, block(1, 0, "TEXT", text)),
        config=ChunkingConfig(max_chunk_chars=16, overlap_chars=0),
    )

    assert [chunk.text for chunk in result.document.chunks] == [text[:16], text[16:]]
    assert all(len(chunk.text) <= 16 for chunk in result.document.chunks)
    assert [
        (source.start_offset, source.end_offset)
        for chunk in result.document.chunks
        for source in chunk.sources
    ] == [(0, 16), (16, len(text))]


def test_contiguous_atoms_from_one_block_merge_into_one_source_range():
    text = "One. Two. Three. Four. Five."
    result = run(
        parsed_page(1, block(1, 0, "TEXT", text)),
        config=ChunkingConfig(max_chunk_chars=16, overlap_chars=0),
    )

    assert len(result.document.chunks) > 1
    for chunk in result.document.chunks:
        assert chunk.text == text[
            chunk.sources[0].start_offset : chunk.sources[0].end_offset
        ]
        assert len(chunk.sources) == 1
    assert "".join(chunk.text for chunk in result.document.chunks) == text


def test_paragraph_boundaries_do_not_receive_overlap_and_long_paragraphs_do():
    paragraphs = "First paragraph.\n\nSecond paragraph."
    paragraph_result = run(
        parsed_page(1, block(1, 0, "TEXT", paragraphs)),
        config=ChunkingConfig(max_chunk_chars=24, overlap_chars=8),
    )
    assert [chunk.text for chunk in paragraph_result.document.chunks] == [
        "First paragraph.\n\n",
        "Second paragraph.",
    ]
    assert paragraph_result.document.chunks[0].sources[0].end_offset == (
        paragraph_result.document.chunks[1].sources[0].start_offset
    )

    long_paragraph = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    overlap_result = run(
        parsed_page(1, block(1, 0, "TEXT", long_paragraph)),
        config=ChunkingConfig(max_chunk_chars=16, overlap_chars=4),
    )
    chunks = overlap_result.document.chunks
    assert all(len(chunk.text) <= 16 for chunk in chunks)
    assert [chunk.text for chunk in chunks] == [
        long_paragraph[:12],
        long_paragraph[8:24],
        long_paragraph[20:36],
        long_paragraph[32:48],
        long_paragraph[44:],
    ]
    assert [
        (chunk.sources[0].start_offset, chunk.sources[0].end_offset)
        for chunk in chunks
    ] == [(0, 12), (8, 24), (20, 36), (32, 48), (44, len(long_paragraph))]

    sentences = "The first sentence stays intact. The second sentence stays intact too."
    sentence_result = run(
        parsed_page(1, block(1, 0, "TEXT", sentences)),
        config=ChunkingConfig(max_chunk_chars=48, overlap_chars=8),
    )
    assert sentence_result.document.chunks[0].text == "The first sentence stays intact. "
    second_chunk = sentence_result.document.chunks[1]
    second_source = second_chunk.sources[0]
    assert second_source.start_offset is not None
    assert second_source.end_offset == len(sentences)
    assert second_chunk.text == sentences[second_source.start_offset :]
    assert second_source.start_offset < len("The first sentence stays intact. ")


def test_chunks_are_isolated_by_page_and_strong_marker_can_close_a_chunk():
    page_one = parsed_page(
        1,
        block(1, 0, "TEXT", "Intro.\n1. X and more text to exceed budget"),
    )
    page_two = parsed_page(2, block(2, 0, "TEXT", "Second page content"))
    result = run(
        page_one,
        page_two,
        config=ChunkingConfig(
            max_chunk_chars=16, overlap_chars=0, marker_break_min_pct=25
        ),
    )

    assert all(
        len({source.page_number for source in chunk.sources}) == 1
        for chunk in result.document.chunks
    )
    first_page_chunks = [
        chunk for chunk in result.document.chunks
        if chunk.sources[0].page_number == 1
    ]
    marker_chunk_index = next(
        i for i, chunk in enumerate(first_page_chunks) if chunk.text.startswith("1.")
    )
    assert marker_chunk_index > 0
    assert first_page_chunks[marker_chunk_index - 1].text == "Intro.\n"


@pytest.mark.parametrize(
    ("marker_breaks", "marker_break_min_pct", "marker_starts_new_chunk"),
    [(True, 25, True), (True, 75, False), (False, 25, False)],
)
def test_marker_break_config_controls_strong_marker_closing(
    marker_breaks, marker_break_min_pct, marker_starts_new_chunk
):
    text = "Intro.\n1. X and more text to exceed budget"
    result = run(
        parsed_page(1, block(1, 0, "TEXT", text)),
        config=ChunkingConfig(
            max_chunk_chars=16,
            overlap_chars=0,
            marker_breaks=marker_breaks,
            marker_break_min_pct=marker_break_min_pct,
        ),
    )

    first_marker_chunk = next(
        chunk for chunk in result.document.chunks if "1." in chunk.text
    )
    assert first_marker_chunk.text.startswith("1.") is marker_starts_new_chunk


def test_weak_list_marker_does_not_trigger_r2_chunk_closing():
    text = "Start.\n(a) xxxxxxxx more content to exceed the maximum size"
    result = run(
        parsed_page(1, block(1, 0, "TEXT", text)),
        config=ChunkingConfig(
            max_chunk_chars=16, overlap_chars=0, marker_break_min_pct=25
        ),
    )

    marker_chunk = next(chunk for chunk in result.document.chunks if "(a)" in chunk.text)
    assert marker_chunk.text.startswith("Start.\n")


def test_marker_detection_matches_only_each_current_line(monkeypatch):
    line = "Section 5 short rule"
    line_count = 2000
    text = "\n".join([line] * line_count)
    source_block = block(1, 0, "TEXT", text)
    original_match = chunking_module.re.match
    inspected_lengths = []

    def record_match(pattern, candidate, *args, **kwargs):
        inspected_lengths.append(len(candidate))
        return original_match(pattern, candidate, *args, **kwargs)

    monkeypatch.setattr(chunking_module.re, "match", record_match)
    config = ChunkingConfig()
    line_starts = [index * (len(line) + 1) for index in range(line_count)]
    matches = [
        chunking_module._atom_starts_strong_provision(
            source_block, start, config
        )
        for start in line_starts
    ]

    assert all(matches)
    assert max(inspected_lengths) == len(line)
    assert sum(inspected_lengths) <= line_count * len(line) * 3


def test_separator_counts_toward_budget_but_has_no_source_range():
    exact = run(
        parsed_page(
            1,
            block(1, 0, "TEXT", "1234567"),
            block(1, 1, "OTHER", "ABCDEFG"),
        ),
        config=ChunkingConfig(max_chunk_chars=16, overlap_chars=0),
    )
    overflow = run(
        parsed_page(
            1,
            block(1, 0, "TEXT", "12345678"),
            block(1, 1, "OTHER", "ABCDEFG"),
        ),
        config=ChunkingConfig(max_chunk_chars=16, overlap_chars=0),
    )

    assert exact.document.chunks[0].text == "1234567\n\nABCDEFG"
    assert len(exact.document.chunks[0].text) == 16
    assert len(exact.document.chunks[0].sources) == 2
    assert [chunk.text for chunk in overflow.document.chunks] == [
        "12345678",
        "ABCDEFG",
    ]
    assert all(len(chunk.text) <= 16 for chunk in overflow.document.chunks)


def test_sources_reconstruct_eligible_blocks_exactly_once_and_output_is_deterministic():
    first = "Intro. More text. " + "A" * 25
    second = "Other block with content."
    pages = (
        parsed_page(
            1,
            block(1, 0, "TEXT", first),
            block(1, 1, "OTHER", second),
        ),
        parsed_page(2, block(2, 0, "TEXT", "Second page.")),
    )
    config = ChunkingConfig(max_chunk_chars=16, overlap_chars=0)
    result = run(*pages, config=config)
    repeated = run(*pages, config=config)
    source_texts = {
        item.block_id: item.text
        for page in pages
        for item in page.blocks
    }

    assert result == repeated
    assert [chunk.index for chunk in result.document.chunks] == list(
        range(len(result.document.chunks))
    )
    for chunk in result.document.chunks:
        assert len(chunk.text) <= config.max_chunk_chars
        assert chunk.sources == tuple(
            sorted(
                chunk.sources,
                key=lambda source: (
                    source.page_number,
                    source.block_reading_order,
                    source.start_offset,
                ),
            )
        )
        expected_parts = []
        previous_block_id = None
        for source in chunk.sources:
            if previous_block_id is not None and source.block_id != previous_block_id:
                expected_parts.append(config.synthetic_separator)
            expected_parts.append(
                source_texts[source.block_id][source.start_offset : source.end_offset]
            )
            previous_block_id = source.block_id
        assert chunk.text == "".join(expected_parts)

    ranges = {}
    for chunk in result.document.chunks:
        for source in chunk.sources:
            ranges.setdefault(source.block_id, []).append(
                (source.start_offset, source.end_offset)
            )
    for block_id, text in source_texts.items():
        ordered_ranges = sorted(ranges[block_id])
        cursor = 0
        reconstructed = []
        for start, end in ordered_ranges:
            assert start == cursor
            reconstructed.append(text[start:end])
            cursor = end
        assert cursor == len(text)
        assert "".join(reconstructed) == text


def test_zero_chunk_blank_and_failed_pages_use_existing_report_contracts():
    blank = CanonicalPage(page_number=1, parser="test", status="BLANK")
    blank_result = run(blank)
    failure = ParserFailure(parser="test", code="error", message="failed")
    failed = CanonicalPage(
        page_number=1, parser="test", status="FAILED", failure=failure
    )
    failed_result = run(failed)

    assert blank_result.document.chunks == ()
    assert blank_result.report.outcome == "NO_CONTENT"
    assert blank_result.report.pages[0].outcome == "SKIPPED_BLANK"
    assert failed_result.document.chunks == ()
    assert failed_result.report.outcome == "FAILED"
    assert failed_result.report.pages[0].failure == failure


def test_report_is_partial_when_chunked_pages_coexist_with_a_failed_page():
    failure = ParserFailure(parser="test", code="error", message="failed")
    warning = ParserFailure(parser="test", code="warning", message="warning")
    pages = (
        parsed_page(1, block(1, 0, "TEXT", "processable text")),
        CanonicalPage(
            page_number=2,
            parser="test",
            status="FAILED",
            failure=failure,
            warnings=(warning,),
        ),
    )
    result = run(*pages)

    assert result.report.outcome == "PARTIAL"
    assert [page.outcome for page in result.report.pages] == [
        "CHUNKED",
        "SKIPPED_FAILED",
    ]
    assert result.report.pages[1].failure == failure
    assert result.report.pages[1].warnings == (warning,)


@pytest.mark.parametrize("kind", ["TABLE", "IMAGE"])
def test_small_table_and_image_blocks_are_processed_as_isolated_chunks(kind):
    text = "layout content"
    result = run(parsed_page(1, block(1, 0, kind, text)))

    assert [chunk.text for chunk in result.document.chunks] == [text]
    assert result.document.chunks[0].sources[0].block_type == kind
    assert result.report.outcome == "COMPLETED"
    assert result.report.pages[0].outcome == "CHUNKED"
    assert result.report.skipped_blocks == ()


@pytest.mark.parametrize(
    ("kind", "text", "reason"),
    [
        ("TABLE", "", "EMPTY_TEXT"),
        ("TABLE", " \t\n", "WHITESPACE_ONLY_TEXT"),
        ("IMAGE", "", "IMAGE_NO_USABLE_TEXT"),
        ("IMAGE", " \t\n", "IMAGE_NO_USABLE_TEXT"),
    ],
)
def test_empty_or_whitespace_table_and_image_keep_existing_skip_reporting(
    kind, text, reason
):
    result = run(parsed_page(1, block(1, 0, kind, text)))

    assert result.document.chunks == ()
    assert result.report.outcome == "NO_CONTENT"
    assert result.report.skipped_blocks[0].reason == reason


def test_supported_text_with_table_is_fully_processed_and_isolated():
    result = run(
        parsed_page(
            1,
            block(1, 0, "TEXT", "processable text"),
            block(1, 1, "TABLE", "table rows"),
        )
    )

    assert [chunk.text for chunk in result.document.chunks] == [
        "processable text",
        "table rows",
    ]
    assert all(len(chunk.sources) == 1 for chunk in result.document.chunks)
    assert result.report.outcome == "COMPLETED"
    assert result.report.pages[0].outcome == "CHUNKED"
    assert result.report.skipped_blocks == ()


def test_chunk_document_processes_text_and_reports_ineligible_vector_block():
    page = parsed_page(
        1,
        block(1, 0, "TEXT", "processable ordinary text"),
        block(1, 1, "VECTOR", "vector label"),
    )

    result = run(page)

    assert [chunk.text for chunk in result.document.chunks] == [
        "processable ordinary text"
    ]
    assert result.report.outcome == "COMPLETED"
    assert result.report.pages[0].outcome == "CHUNKED"
    assert [
        (item.block_type, item.reason) for item in result.report.skipped_blocks
    ] == [("VECTOR", "VECTOR_NOT_TEXT_ELIGIBLE")]


def assert_source_coverage(chunks, source_block):
    ranges = sorted(
        (source.start_offset, source.end_offset)
        for chunk in chunks
        for source in chunk.sources
        if source.block_id == source_block.block_id
    )
    cursor = 0
    reconstructed = []
    for start, end in ranges:
        assert start == cursor
        reconstructed.append(source_block.text[start:end])
        cursor = end
    assert cursor == len(source_block.text)
    assert "".join(reconstructed) == source_block.text


def test_table_blocks_are_isolated_from_text_image_and_other_tables():
    page = parsed_page(
        1,
        block(1, 0, "TEXT", "ordinary"),
        block(1, 1, "TABLE", "table one"),
        block(1, 2, "TABLE", "table two"),
        block(1, 3, "IMAGE", "image text"),
        block(1, 4, "OTHER", "other text"),
    )
    result = run(page)

    assert [chunk.text for chunk in result.document.chunks] == [
        "ordinary",
        "table one",
        "table two",
        "image text",
        "other text",
    ]
    assert [chunk.sources[0].block_type for chunk in result.document.chunks] == [
        "TEXT",
        "TABLE",
        "TABLE",
        "IMAGE",
        "OTHER",
    ]
    assert all(len(chunk.sources) == 1 for chunk in result.document.chunks)


def test_oversized_table_packs_complete_rows_with_terminators_attached():
    text = "row0001\r\n\r\nrow0002\nrow0003\rrow0004"
    table = block(1, 0, "TABLE", text)
    result = run(
        parsed_page(1, table),
        config=ChunkingConfig(max_chunk_chars=16),
    )

    # CRLF, LF, lone CR, and blank rows remain attached to their source rows.
    assert len(result.document.chunks) > 1
    assert all(len(chunk.text) <= 16 for chunk in result.document.chunks)
    assert all(len(chunk.sources) == 1 for chunk in result.document.chunks)
    assert all(
        source.block_type == "TABLE"
        for chunk in result.document.chunks
        for source in chunk.sources
    )
    assert "".join(chunk.text for chunk in result.document.chunks) == text
    assert_source_coverage(result.document.chunks, table)
    for chunk in result.document.chunks[:-1]:
        assert chunk.text.endswith(("\r\n", "\n", "\r"))


@pytest.mark.parametrize(
    ("text", "expected_chunks"),
    [
        (
            "firstcol  secondcol  thirdcol",
            ("firstcol  ", "secondcol  ", "thirdcol"),
        ),
        (
            "firstcol\tsecondcol\tthirdcol",
            ("firstcol\t", "secondcol\t", "thirdcol"),
        ),
        (
            "firstcol | secondcol | thirdcol",
            ("firstcol | ", "secondcol | ", "thirdcol"),
        ),
        (
            "alpha bravo charlie delta",
            ("alpha bravo ", "charlie delta"),
        ),
        ("x" * 40, ("x" * 16, "x" * 16, "x" * 8)),
    ],
)
def test_oversized_table_row_uses_column_word_then_grapheme_fallbacks(
    text, expected_chunks
):
    table = block(1, 0, "TABLE", text)
    result = run(
        parsed_page(1, table),
        config=ChunkingConfig(max_chunk_chars=16),
    )

    assert tuple(chunk.text for chunk in result.document.chunks) == expected_chunks
    assert result.report.diagnostics.oversized_table_rows == 1
    assert all(len(chunk.text) <= 16 for chunk in result.document.chunks)
    assert_source_coverage(result.document.chunks, table)


def test_table_row_fallback_reports_unsafe_grapheme_cuts():
    text = "A" * 15 + "e\u0301" + "B" * 30
    table = block(1, 0, "TABLE", text)
    result = run(
        parsed_page(1, table),
        config=ChunkingConfig(max_chunk_chars=16, grapheme_backoff_chars=0),
    )

    assert result.report.diagnostics.unsafe_grapheme_cuts > 0
    assert all(len(chunk.text) <= 16 for chunk in result.document.chunks)
    assert_source_coverage(result.document.chunks, table)


def test_table_headers_are_not_repeated_and_empty_rows_remain_exact():
    text = "HEADER  COL\n\nrow one\nrow two\nrow three"
    table = block(1, 0, "TABLE", text)
    result = run(
        parsed_page(1, table),
        config=ChunkingConfig(max_chunk_chars=16),
    )

    assert "".join(chunk.text for chunk in result.document.chunks) == text
    assert sum(chunk.text.count("HEADER") for chunk in result.document.chunks) == 1
    assert_source_coverage(result.document.chunks, table)


def test_image_text_respects_size_budget_isolation_and_configuration():
    small_page = parsed_page(
        1,
        block(1, 0, "TEXT", "text"),
        block(1, 1, "IMAGE", "OCR text"),
        block(1, 2, "TABLE", "table"),
    )
    small = run(small_page, config=ChunkingConfig(max_chunk_chars=16))
    oversized_text = "Image OCR sentence. Another sentence. " + "x" * 20
    image = block(1, 0, "IMAGE", oversized_text)
    oversized = run(
        parsed_page(1, image),
        config=ChunkingConfig(max_chunk_chars=16),
    )
    disabled = run(
        parsed_page(1, block(1, 0, "IMAGE", "OCR text")),
        config=ChunkingConfig(include_image_text=False),
    )

    assert [chunk.text for chunk in small.document.chunks] == [
        "text",
        "OCR text",
        "table",
    ]
    assert [chunk.sources[0].block_type for chunk in small.document.chunks] == [
        "TEXT",
        "IMAGE",
        "TABLE",
    ]
    assert len(oversized.document.chunks) > 1
    assert all(len(chunk.text) <= 16 for chunk in oversized.document.chunks)
    assert all(
        len(chunk.sources) == 1 and chunk.sources[0].block_type == "IMAGE"
        for chunk in oversized.document.chunks
    )
    assert_source_coverage(oversized.document.chunks, image)
    assert disabled.document.chunks == ()
    assert disabled.report.skipped_blocks[0].reason == "IMAGE_TEXT_DISABLED"
    assert disabled.report.pages[0].outcome == "SKIPPED_NO_ELIGIBLE_TEXT"


def test_table_output_is_deterministic_and_report_counters_match():
    text = "header  value\nrow one\nrow two\nrow three"
    page = parsed_page(1, block(1, 0, "TABLE", text))
    config = ChunkingConfig(max_chunk_chars=16)
    first = run(page, config=config)
    second = run(page, config=config)

    assert first == second
    assert first.report.outcome == "COMPLETED"
    assert first.report.pages[0].chunks_created == len(first.document.chunks)
    assert first.report.diagnostics.chunks_created == len(first.document.chunks)
    assert first.report.diagnostics.oversized_table_rows == 0
