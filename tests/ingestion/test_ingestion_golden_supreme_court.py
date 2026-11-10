"""Golden-corpus checks for parser and BLP-1 behavior on one real judgment.

The observed JSON is retained for comparison, not treated as perfect output.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pymupdf
import pytest

from src.ingestion.chunk_contracts import ChunkSource
from src.ingestion.chunking import ChunkingConfig, ChunkingResult, chunk_document
from src.ingestion.parsers import StructuredParser
from src.ingestion.parser_contracts import CanonicalBlock, CanonicalDocument


FIXTURE_DIR = (
    Path(__file__).parents[1]
    / "fixtures"
    / "ingestion"
    / "supreme_court_2026_insc_60"
)


def _read_json(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def golden_data():
    manifest = _read_json("manifest.json")
    canonical_data = _read_json("canonical.json")
    canonical = CanonicalDocument.model_validate(canonical_data)
    result = ChunkingResult.model_validate(_read_json("chunking-result.json"))
    return manifest, canonical, result


def _all_blocks(document: CanonicalDocument) -> dict[str, CanonicalBlock]:
    return {
        block.block_id: block
        for page in document.pages
        for block in page.blocks
    }


def _source_slice(source: ChunkSource, blocks: dict[str, CanonicalBlock]) -> str:
    assert source.start_offset is not None
    assert source.end_offset is not None
    return blocks[source.block_id].text[source.start_offset : source.end_offset]


def _assert_chunking_invariants(
    canonical: CanonicalDocument,
    result: ChunkingResult,
    config: ChunkingConfig,
) -> None:
    blocks = _all_blocks(canonical)
    chunks = result.document.chunks
    report_pages = result.report.pages

    assert tuple(chunk.index for chunk in chunks) == tuple(range(len(chunks)))
    assert tuple(page.page_number for page in report_pages) == tuple(
        range(1, canonical.page_count + 1)
    )
    assert result.document.source_page_count == canonical.page_count
    assert result.report.outcome == "COMPLETED"
    assert result.report.diagnostics.chunks_created == len(chunks)
    assert sum(page.chunks_created for page in report_pages) == len(chunks)

    covered: dict[str, list[tuple[int, int]]] = defaultdict(list)
    chunks_per_page: dict[int, int] = defaultdict(int)
    for chunk in chunks:
        assert chunk.text
        assert len(chunk.text) <= config.max_chunk_chars
        pages = {source.page_number for source in chunk.sources}
        assert len(pages) == 1, "BLP-1 chunks must remain within one page"
        page_number = next(iter(pages))
        chunks_per_page[page_number] += 1

        reconstructed: list[str] = []
        previous_block_id: str | None = None
        for source in chunk.sources:
            block = blocks[source.block_id]
            assert source.page_number == block.page_number
            assert source.block_reading_order == block.reading_order
            assert source.block_type == block.block_type
            assert source.source_text_length == len(block.text)
            assert source.start_offset is not None and source.end_offset is not None
            assert 0 <= source.start_offset < source.end_offset <= len(block.text)
            if previous_block_id is not None and source.block_id != previous_block_id:
                reconstructed.append(config.synthetic_separator)
            reconstructed.append(_source_slice(source, blocks))
            covered[source.block_id].append((source.start_offset, source.end_offset))
            previous_block_id = source.block_id
        assert "".join(reconstructed) == chunk.text

        block_types = {blocks[source.block_id].block_type for source in chunk.sources}
        if "TABLE" in block_types:
            assert block_types == {"TABLE"}
            assert len({source.block_id for source in chunk.sources}) == 1
        if "IMAGE" in block_types:
            assert block_types == {"IMAGE"}
            assert len({source.block_id for source in chunk.sources}) == 1

    for page_report in report_pages:
        assert page_report.chunks_created == chunks_per_page[page_report.page_number]
        assert page_report.parser_status == canonical.pages[
            page_report.page_number - 1
        ].status

    for block_id, block in blocks.items():
        if block.block_type not in {"TEXT", "OTHER", "TABLE", "IMAGE"}:
            continue
        if not block.text.strip():
            assert block_id not in covered
            continue
        ranges = covered[block_id]
        assert ranges, f"Eligible source block {block_id} was not chunked"
        cursor = 0
        for start, end in ranges:
            assert start <= cursor, f"Gap in source block {block_id}"
            allowed_overlap = (
                min(config.overlap_chars, config.max_chunk_chars // 4)
                if block.block_type in {"TEXT", "OTHER"}
                else 0
            )
            assert cursor - start <= allowed_overlap
            cursor = max(cursor, end)
        assert cursor == len(block.text), f"Incomplete source block {block_id}"


def test_fixture_manifest_and_observed_outputs_are_pinned(golden_data):
    manifest, canonical, result = golden_data
    for filename, key in (
        ("real_corpus.pdf", "source_sha256"),
        ("canonical.json", "canonical_sha256"),
        ("chunking-result.json", "chunking_result_sha256"),
    ):
        digest = hashlib.sha256((FIXTURE_DIR / filename).read_bytes()).hexdigest()
        expected = (
            manifest[key]
            if key == "source_sha256"
            else manifest["observed_outputs"][key]
        )
        assert digest == expected

    assert canonical.page_count == manifest["physical_page_count"] == 20
    pdf = pymupdf.open(FIXTURE_DIR / "real_corpus.pdf")
    assert pdf.page_count == canonical.page_count
    assert result.document.policy_version == manifest["chunking"]["policy_version"]
    assert result.report.config_digest == manifest["chunking"]["config_digest"]
    assert ChunkingConfig().policy_version().startswith("5c-1.4.0+")


def test_golden_chunks_preserve_source_ranges_and_metadata(golden_data):
    _, canonical, result = golden_data
    _assert_chunking_invariants(canonical, result, ChunkingConfig())


def test_table_chunks_match_rendered_pdf_pages(golden_data):
    _, canonical, result = golden_data
    blocks = _all_blocks(canonical)
    pdf = pymupdf.open(FIXTURE_DIR / "real_corpus.pdf")
    visible_table_text = {
        3: ("INTRODUCTION", "BRIEF", "FACTS", "PAGE", "NO."),
        8: ("23,670,710", "2,282,825", "105,928"),
        9: ("14,754,087", "1,422,897", "66,026"),
        10: ("6.05%", "6.92%", "8.47%"),
    }

    for page_number, expected_words in visible_table_text.items():
        page = pdf[page_number - 1]
        rendered = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
        assert rendered.width > 0 and rendered.height > 0
        words = page.get_text("words")
        page_text = {word[4] for word in words}
        for expected in expected_words:
            assert expected in page_text

        table_blocks = [
            block
            for block in canonical.pages[page_number - 1].blocks
            if block.block_type == "TABLE"
        ]
        assert table_blocks, f"Expected a canonical table on PDF page {page_number}"
        table_text = "\n".join(block.text for block in table_blocks)
        assert all(expected in table_text for expected in expected_words)
        for block in table_blocks:
            box = block.source_coordinates
            page_height = page.rect.height
            left, right = box.x0, box.x1
            top, bottom = page_height - box.y1, page_height - box.y0
            words_in_table_box = {
                word[4]
                for word in words
                if word[0] >= left
                and word[2] <= right
                and word[1] >= top
                and word[3] <= bottom
            }
            assert words_in_table_box, f"Table bbox on page {page_number} misses PDF text"

            sources = [
                source
                for chunk in result.document.chunks
                for source in chunk.sources
                if source.block_id == block.block_id
            ]
            assert sources
            assert all(blocks[source.block_id].block_type == "TABLE" for source in sources)
            assert "".join(_source_slice(source, blocks) for source in sorted(
                sources, key=lambda source: source.start_offset
            )) == block.text
            assert all(
                {blocks[source.block_id].block_type for source in chunk.sources} == {"TABLE"}
                for chunk in result.document.chunks
                if any(source.block_id == block.block_id for source in chunk.sources)
            )


def _page_one_order(canonical: CanonicalDocument) -> list[str]:
    return [block.text for block in canonical.pages[0].blocks]


@pytest.mark.xfail(
    strict=True,
    reason="The preserved pre-fix canonical fixture retains PARSER-READING-ORDER-PAGE-1.",
)
def test_page_one_cover_parties_follow_visual_reading_order(golden_data):
    _, canonical, _ = golden_data
    order = _page_one_order(canonical)
    first_appellant = order.index("THE AUTHORITY FOR ADVANCE RULINGS")
    first_appellant_label = order.index(".... APPELLANT(S)")
    first_versus = order.index("VERSUS")
    first_respondent = order.index("TIGER GLOBAL INTERNATIONAL II HOLDINGS")
    first_respondent_label = order.index(".... RESPONDENT(S)")
    second_appellant = order.index(
        "THE AUTHORITY FOR ADVANCE RULINGS (INCOME TAX) AND OTHERS"
    )
    second_appellant_label = order.index(".... APPELLANT(S)", first_appellant_label + 1)
    second_versus = order.index("VERSUS", first_versus + 1)

    assert first_appellant < first_appellant_label < first_versus
    assert first_versus < first_respondent < first_respondent_label
    assert second_appellant < second_appellant_label < second_versus


def test_preserved_observation_records_original_page_eight_duplicate(golden_data):
    _, canonical, _ = golden_data
    blocks = canonical.pages[7].blocks
    table = next(block for block in blocks if block.block_type == "TABLE")
    text_blocks = [block for block in blocks if block.block_type == "TEXT"]
    duplicated_paragraphs = [
        block.text
        for block in text_blocks
        if block.text and table.text.startswith(block.text)
    ]
    assert duplicated_paragraphs, (
        "The preserved pre-fix observation should retain the defect"
    )


def test_structured_parser_serializes_table_without_caption(golden_data):
    _, observed_canonical, _ = golden_data
    observed_blocks = observed_canonical.pages[7].blocks
    observed_table = next(
        block for block in observed_blocks if block.block_type == "TABLE"
    )
    expected_grid_text = observed_table.text.split("\n\n", maxsplit=1)[1]

    class CapturingConverter:
        def __init__(self, delegate):
            self.delegate = delegate
            self.document = None
            self.table_caption_refs = None
            self.caption_text = None

        def convert(self, *args, **kwargs):
            result = self.delegate.convert(*args, **kwargs)
            self.document = result.document
            table = next(
                item
                for item, _ in self.document.iterate_items()
                if str(getattr(item.label, "value", item.label)).casefold() == "table"
            )
            self.table_caption_refs = tuple(ref.cref for ref in table.captions)
            self.caption_text = table.captions[0].resolve(doc=self.document).text
            return result

    parser = StructuredParser()
    converter = CapturingConverter(parser._get_converter())
    page = StructuredParser(converter=converter).parse_page(
        FIXTURE_DIR / "real_corpus.pdf", 8
    )

    table_block = next(block for block in page.blocks if block.block_type == "TABLE")
    caption_block = next(
        block for block in page.blocks if block.text.startswith("5.4.")
    )

    def table_cells(value: str) -> tuple[tuple[str, ...], ...]:
        rows = tuple(
            tuple(" ".join(cell.split()) for cell in line.strip().strip("|").split("|"))
            for line in value.splitlines()
            if "|" in line
        )
        return tuple(
            row
            for row in rows
            if not all(set(cell) <= {"-", ":"} for cell in row)
        )

    assert table_cells(table_block.text) == table_cells(expected_grid_text)
    assert "5.4." not in table_block.text
    assert caption_block.text == converter.caption_text
    assert caption_block.block_type == "TEXT"
    assert caption_block.page_number == 8

    table = next(
        item
        for item, _ in converter.document.iterate_items()
        if str(getattr(item.label, "value", item.label)).casefold() == "table"
    )
    assert tuple(ref.cref for ref in table.captions) == converter.table_caption_refs
    assert table.captions[0].resolve(doc=converter.document).text == converter.caption_text


def test_structured_parser_recovers_native_text_from_empty_image_region(golden_data):
    _, baseline, _ = golden_data
    parser = StructuredParser()
    assert parser.do_ocr is False
    source = FIXTURE_DIR / "real_corpus.pdf"
    page = parser.parse_page(source, 1)

    recovered_name = [
        block
        for block in page.blocks
        if block.block_type == "TEXT"
        and " ".join(block.text.split()) == "TIGER GLOBAL INTERNATIONAL IV HOLDINGS"
    ]
    assert len(recovered_name) == 1
    name_block = recovered_name[0]
    assert name_block.block_id == f"page-1-block-{len(baseline.pages[0].blocks)}"
    recovery_warning = next(
        warning for warning in page.warnings
        if warning.code == "NATIVE_TEXT_RECOVERED"
    )
    assert "visual row order" in recovery_warning.message
    respondent_label = next(
        block
        for block in reversed(page.blocks)
        if block.text == ".... RESPONDENT(S)"
    )
    assert name_block.reading_order < respondent_label.reading_order

    # PyMuPDF's two original text lines define the recovered block's source box.
    with pymupdf.open(source) as pdf:
        pdf_page = pdf[0]
        pdf_blocks = pdf_page.get_text("dict", sort=False)["blocks"]
        native_block = next(
            raw_block
            for raw_block in pdf_blocks
            if raw_block.get("type") == 0
            and any(
                " ".join(
                    "".join(span.get("text", "") for span in line.get("spans", []))
                    .split()
                )
                == "TIGER GLOBAL INTERNATIONAL IV"
                for line in raw_block.get("lines", [])
            )
        )
        name_lines = [
            line
            for line in native_block["lines"]
            if " ".join(
                "".join(span.get("text", "") for span in line.get("spans", []))
                .split()
            )
            in {"TIGER GLOBAL INTERNATIONAL IV", "HOLDINGS"}
        ]
        assert len(name_lines) == 2
        boxes = [line["bbox"] for line in name_lines]
        expected_box = (
            min(box[0] for box in boxes),
            pdf_page.rect.height - max(box[3] for box in boxes),
            max(box[2] for box in boxes),
            pdf_page.rect.height - min(box[1] for box in boxes),
        )

    coordinates = name_block.source_coordinates
    assert coordinates.origin == "BOTTOM_LEFT"
    assert coordinates.unit == "PDF_POINT"
    assert (
        coordinates.x0,
        coordinates.y0,
        coordinates.x1,
        coordinates.y1,
    ) == pytest.approx(expected_box)

    assert tuple(block.reading_order for block in page.blocks) == tuple(
        range(len(page.blocks))
    )
    assert len({block.block_id for block in page.blocks}) == len(page.blocks)

    ordered_text = [block.text for block in page.blocks]
    assert ordered_text.index("VERSUS") < ordered_text.index("TIGER GLOBAL INTERNATIONAL II HOLDINGS")
    assert ordered_text.index("TIGER GLOBAL INTERNATIONAL II HOLDINGS") < ordered_text.index(".... RESPONDENT(S)")
    second_appellant = ordered_text.index(
        "THE AUTHORITY FOR ADVANCE RULINGS (INCOME TAX) AND OTHERS"
    )
    second_appellant_label = ordered_text.index(".... APPELLANT(S)", second_appellant)
    second_versus = ordered_text.index("VERSUS", second_appellant_label)
    assert second_appellant < second_appellant_label < second_versus

    # Page 2 needs no native-text recovery warning.
    page_two = parser.parse_page(source, 2)
    assert tuple(block.reading_order for block in page_two.blocks) == tuple(
        range(len(page_two.blocks))
    )
    assert page_two.warnings == ()


def test_introduction_continuation_keeps_page_isolation_over_cross_page_overlap(
    golden_data,
):
    _, canonical, _ = golden_data
    result = chunk_document(canonical, "page-boundary-regression", ChunkingConfig())
    preceding = next(
        chunk
        for chunk in result.document.chunks
        if chunk.sources[0].page_number == 3
        and "In a world where nations must necessarily engage" in chunk.text
    )
    continuation = next(
        chunk
        for chunk in result.document.chunks
        if chunk.sources[0].page_number == 4
        and chunk.text.startswith("with each other for mutual economic growth")
    )

    assert preceding.text.endswith("In a world where nations must necessarily engage")
    assert {source.page_number for source in preceding.sources} == {3}
    assert {source.page_number for source in continuation.sources} == {4}
