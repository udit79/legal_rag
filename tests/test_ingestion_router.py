import pytest

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

@pytest.mark.parametrize(
    ("counts", "expected_route"),
    [
        ((0, 0, 0), "SKIP_BLANK"),
        ((50, 0, 0), "FAST_TEXT"),
        ((50, 1, 0), "STRUCTURED_PARSE"),
        ((49, 1, 0), "OCR"),
        ((49, 0, 1), "LAYOUT_OCR"),
        ((49, 0, 0), "ESCALATE"),
    ],
)
def test_routes_page_based_on_features(counts, expected_route):
    features = make_features(counts)

    plan = route_document(features, "routing-v1")
    decision = plan.decisions[0]

    assert decision.route == expected_route
    assert decision.decision_source == "RULES"
    assert decision.policy_version == "routing-v1"
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
def test_routing_is_deterministic_and_preserves_page_order():
    features = make_features(
        (100, 0, 0),
        (0, 1, 0),
        (0, 0, 0),
        (20, 0, 1),
    )

    first = route_document(features, "routing-v1")
    second = route_document(features, "routing-v1")

    assert first == second
    assert [d.page_number for d in first.decisions] == [1, 2, 3, 4]
    assert [d.route for d in first.decisions] == [
        "FAST_TEXT",
        "OCR",
        "SKIP_BLANK",
        "LAYOUT_OCR",
    ]
    assert first.document_disposition == "PROCESS"
@pytest.mark.parametrize("policy_version", ["", "   "])
def test_router_rejects_blank_policy_version(policy_version):
    features = make_features((100, 0, 0))

    with pytest.raises(ValueError, match="policy_version"):
        route_document(features, policy_version)
