from src.ingestion.contracts import DocumentFeatures, PageFeatures
from src.ingestion.router import route_document


def make_features(*counts: tuple[int, int, int]) -> DocumentFeatures:
    pages = tuple(
        PageFeatures(
            page_number=page_number,
            text_char_count=text_count,
            image_count=image_count,
            drawing_count=drawing_count,
        )
        for page_number, (text_count, image_count, drawing_count) in enumerate(
            counts, start=1
        )
    )
    return DocumentFeatures(
        content_sha256="a" * 64,
        file_size_bytes=1000,
        page_count=len(pages),
        pages=pages,
    )


def test_all_blank_document_is_rejected_at_document_level():
    features = make_features((0, 0, 0), (0, 0, 0))

    plan = route_document(features, "routing-v1")

    assert plan.document_disposition == "REJECT_NO_CONTENT"
    assert [decision.route for decision in plan.decisions] == [
        "SKIP_BLANK",
        "SKIP_BLANK",
    ]


def test_document_with_blank_and_content_pages_remains_processable():
    features = make_features((0, 0, 0), (100, 0, 0))

    plan = route_document(features, "routing-v1")

    assert plan.document_disposition == "PROCESS"
    assert [decision.route for decision in plan.decisions] == [
        "SKIP_BLANK",
        "FAST_TEXT",
    ]
