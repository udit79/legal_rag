import pytest
from pydantic import ValidationError

from src.ingestion.contracts import DocumentFeatures, DocumentRef, RouteDecision


def document_ref_data() -> dict[str, object]:
    return {
        "document_id": "document-123",
        "source_ref": "file:///documents/policy.pdf",
        "filename": "policy.pdf",
        "media_type": "application/pdf",
    }


def document_features_data() -> dict[str, object]:
    return {
        "content_sha256": "a" * 64,
        "file_size_bytes": 1024,
        "page_count": 2,
        "page_text_char_counts": [120, 80],
        "page_image_counts": [1, 0],
    }


def route_decision_data() -> dict[str, object]:
    return {
        "route": "FAST_TEXT",
        "reason": "The document contains extractable text.",
        "decision_source": "RULES",
        "confidence": 0.9,
        "policy_version": "1.0",
    }


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


def test_document_features_normalizes_uppercase_sha256():
    data = document_features_data()
    data["content_sha256"] = "AB" * 32

    features = DocumentFeatures.model_validate(data)

    assert features.content_sha256 == "ab" * 32


@pytest.mark.parametrize(
    "invalid_hash",
    ["not-a-hash", "a" * 63, "g" * 64],
)
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


@pytest.mark.parametrize(
    ("field", "counts"),
    [
        ("page_text_char_counts", [-1, 0]),
        ("page_image_counts", [0, -1]),
    ],
)
def test_document_features_rejects_negative_page_counts(field, counts):
    data = document_features_data()
    data[field] = counts

    with pytest.raises(ValidationError):
        DocumentFeatures.model_validate(data)


@pytest.mark.parametrize(
    ("field", "counts"),
    [
        ("page_text_char_counts", [120]),
        ("page_image_counts", [1]),
    ],
)
def test_document_features_rejects_per_page_lists_with_wrong_lengths(
    field, counts
):
    data = document_features_data()
    data[field] = counts

    with pytest.raises(ValidationError):
        DocumentFeatures.model_validate(data)


def test_document_features_accepts_zero_text_and_image_counts():
    data = document_features_data()
    data["page_text_char_counts"] = [0, 0]
    data["page_image_counts"] = [0, 0]

    features = DocumentFeatures.model_validate(data)

    assert features.page_count == 2
    assert features.page_text_char_counts == [0, 0]
    assert features.page_image_counts == [0, 0]


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [("route", "UNKNOWN"), ("decision_source", "UNKNOWN")],
)
def test_route_decision_rejects_unsupported_values(field, invalid_value):
    data = route_decision_data()
    data[field] = invalid_value

    with pytest.raises(ValidationError):
        RouteDecision.model_validate(data)


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_route_decision_rejects_confidence_outside_unit_interval(confidence):
    data = route_decision_data()
    data["confidence"] = confidence

    with pytest.raises(ValidationError):
        RouteDecision.model_validate(data)


@pytest.mark.parametrize(
    ("field", "blank_value"),
    [("reason", "   "), ("policy_version", "")],
)
def test_route_decision_rejects_blank_reason_and_policy_version(
    field, blank_value
):
    data = route_decision_data()
    data[field] = blank_value

    with pytest.raises(ValidationError):
        RouteDecision.model_validate(data)


def test_route_decision_rejects_unknown_fields():
    data = route_decision_data()
    data["unexpected"] = "value"

    with pytest.raises(ValidationError):
        RouteDecision.model_validate(data)
