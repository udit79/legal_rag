import pytest
from pydantic import ValidationError

from src.ingestion.contracts import (
    DocumentFeatures,
    DocumentRef,
    DocumentRoutePlan,
    PageFeatures,
    PageRouteDecision,
    RouteDecision,
)


def document_ref_data() -> dict[str, object]:
    return {
        "document_id": "document-123",
        "source_ref": "file:///documents/policy.pdf",
        "filename": "policy.pdf",
        "media_type": "application/pdf",
    }


def page_features_data(page_number: int = 1) -> dict[str, object]:
    return {
        "page_number": page_number,
        "text_char_count": 120,
        "image_count": 1,
        "drawing_count": 0,
    }


def document_features_data() -> dict[str, object]:
    return {
        "content_sha256": "a" * 64,
        "file_size_bytes": 1024,
        "page_count": 2,
        "pages": [page_features_data(1), page_features_data(2)],
    }


def route_decision_data() -> dict[str, object]:
    return {
        "route": "FAST_TEXT",
        "reason": "The document contains extractable text.",
        "decision_source": "RULES",
        "confidence": 0.9,
        "policy_version": "1.0",
    }


def page_route_decision_data(page_number: int, route: str = "FAST_TEXT") -> dict[str, object]:
    return {**route_decision_data(), "page_number": page_number, "route": route}


@pytest.mark.parametrize("field", ["document_id", "source_ref", "filename"])
@pytest.mark.parametrize("blank_value", ["", "   "])
def test_document_ref_rejects_blank_identifiers(field, blank_value):
    data = document_ref_data()
    data[field] = blank_value
    with pytest.raises(ValidationError):
        DocumentRef.model_validate(data)


def test_document_ref_rejects_unknown_fields():
    data = document_ref_data()
    data["unexpected"] = "value"
    with pytest.raises(ValidationError):
        DocumentRef.model_validate(data)


@pytest.mark.parametrize("field", ["text_char_count", "image_count", "drawing_count"])
def test_page_features_rejects_negative_counts(field):
    data = page_features_data()
    data[field] = -1
    with pytest.raises(ValidationError):
        PageFeatures.model_validate(data)


def test_page_features_rejects_zero_or_negative_page_number():
    data = page_features_data(0)
    with pytest.raises(ValidationError):
        PageFeatures.model_validate(data)


def test_page_features_requires_strict_integer_counts():
    data = page_features_data()
    data["text_char_count"] = True
    with pytest.raises(ValidationError):
        PageFeatures.model_validate(data)


def test_page_features_rejects_unknown_fields():
    data = page_features_data()
    data["unexpected"] = "value"
    with pytest.raises(ValidationError):
        PageFeatures.model_validate(data)


def test_document_features_normalizes_uppercase_sha256():
    data = document_features_data()
    data["content_sha256"] = "AB" * 32
    features = DocumentFeatures.model_validate(data)
    assert features.content_sha256 == "ab" * 32


@pytest.mark.parametrize("invalid_hash", ["not-a-hash", "a" * 63, "g" * 64])
def test_document_features_rejects_malformed_hashes(invalid_hash):
    data = document_features_data()
    data["content_sha256"] = invalid_hash
    with pytest.raises(ValidationError):
        DocumentFeatures.model_validate(data)


def test_document_features_rejects_zero_file_size():
    data = document_features_data()
    data["file_size_bytes"] = 0
    with pytest.raises(ValidationError):
        DocumentFeatures.model_validate(data)


def test_document_features_rejects_zero_pages():
    data = document_features_data()
    data["page_count"] = 0
    with pytest.raises(ValidationError):
        DocumentFeatures.model_validate(data)


def test_document_features_requires_one_based_contiguous_pages():
    data = document_features_data()
    data["pages"] = [page_features_data(1), page_features_data(3)]
    with pytest.raises(ValidationError, match="contiguous and one-based"):
        DocumentFeatures.model_validate(data)


def test_document_features_requires_page_records_to_match_page_count():
    data = document_features_data()
    data["pages"] = [page_features_data(1)]
    with pytest.raises(ValidationError, match="equal page_count"):
        DocumentFeatures.model_validate(data)


def test_document_features_rejects_unknown_fields():
    data = document_features_data()
    data["unexpected"] = "value"
    with pytest.raises(ValidationError):
        DocumentFeatures.model_validate(data)


def test_document_route_plan_accepts_mixed_page_outcomes():
    decisions = [
        page_route_decision_data(1, "FAST_TEXT"),
        page_route_decision_data(2, "SKIP_BLANK"),
        page_route_decision_data(3, "OCR"),
    ]
    plan = DocumentRoutePlan.model_validate({"page_count": 3, "decisions": decisions})
    assert [decision.page_number for decision in plan.decisions] == [1, 2, 3]
    assert plan.decisions[1].route == "SKIP_BLANK"


def test_document_route_plan_rejects_duplicate_page_numbers():
    decisions = [page_route_decision_data(1), page_route_decision_data(1)]
    with pytest.raises(ValidationError, match="exactly once"):
        DocumentRoutePlan.model_validate({"page_count": 2, "decisions": decisions})


def test_document_route_plan_rejects_missing_page_number():
    decisions = [page_route_decision_data(1), page_route_decision_data(3)]
    with pytest.raises(ValidationError, match="exactly once"):
        DocumentRoutePlan.model_validate({"page_count": 3, "decisions": decisions})


@pytest.mark.parametrize(
    "decisions",
    [
        [page_route_decision_data(1), page_route_decision_data(3)],
        [page_route_decision_data(2), page_route_decision_data(1)],
    ],
)
def test_document_route_plan_rejects_noncontiguous_or_out_of_order_decisions(decisions):
    with pytest.raises(ValidationError, match="ascending page order"):
        DocumentRoutePlan.model_validate({"page_count": 2, "decisions": decisions})


@pytest.mark.parametrize("page_count", [0, -1, True])
def test_document_route_plan_rejects_invalid_page_count(page_count):
    with pytest.raises(ValidationError):
        DocumentRoutePlan.model_validate({"page_count": page_count, "decisions": []})


def test_document_route_plan_rejects_mixed_policy_versions():
    first = page_route_decision_data(1)
    second = page_route_decision_data(2, "OCR")
    second["policy_version"] = "2.0"
    with pytest.raises(ValidationError, match="same policy version"):
        DocumentRoutePlan.model_validate({"page_count": 2, "decisions": [first, second]})


@pytest.mark.parametrize(
    ("model", "data"),
    [
        (PageRouteDecision, page_route_decision_data(1)),
        (DocumentRoutePlan, {"page_count": 1, "decisions": [page_route_decision_data(1)]}),
    ],
)
def test_route_models_reject_unknown_fields(model, data):
    invalid = {**data, "unexpected": "value"}
    with pytest.raises(ValidationError):
        model.model_validate(invalid)


def test_page_route_decision_rejects_unsupported_route():
    data = page_route_decision_data(1, "UNKNOWN")
    with pytest.raises(ValidationError):
        PageRouteDecision.model_validate(data)


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_route_decisions_reject_confidence_outside_unit_interval(confidence):
    data = route_decision_data()
    data["confidence"] = confidence
    with pytest.raises(ValidationError):
        RouteDecision.model_validate(data)


@pytest.mark.parametrize(
    ("field", "blank_value"),
    [("reason", "   "), ("policy_version", "")],
)
def test_route_decision_rejects_blank_reason_and_policy_version(field, blank_value):
    data = route_decision_data()
    data[field] = blank_value
    with pytest.raises(ValidationError):
        RouteDecision.model_validate(data)


def test_route_decision_rejects_unknown_fields():
    data = route_decision_data()
    data["unexpected"] = "value"
    with pytest.raises(ValidationError):
        RouteDecision.model_validate(data)
