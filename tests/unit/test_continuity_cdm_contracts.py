import pytest
from pydantic import ValidationError

from src.ingestion.continuity_cdm_contracts import (
    CDMClassificationRequest,
    CDMClassificationResult,
    CDMUsage,
    adapt_scaledown_response,
)
from src.ingestion.continuity_contracts import ChunkReference, ContinuityRelation


def request() -> CDMClassificationRequest:
    return CDMClassificationRequest(
        candidate_id="candidate-1",
        source=ChunkReference(document_id="doc-1", chunk_index=3, page_number=4),
        target=ChunkReference(document_id="doc-1", chunk_index=4, page_number=5),
        source_text="  source text \n",
        target_text="\ttarget text  ",
    )


def probabilities() -> dict[str, float]:
    return {
        "SAME_PARAGRAPH": 0.7,
        "TABLE_CONTINUATION": 0.1,
        "NO_CONTINUATION": 0.15,
        "UNKNOWN": 0.05,
    }


def result_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "candidate_id": "candidate-1",
        "relation_type": "SAME_PARAGRAPH",
        "probabilities": probabilities(),
        "model_id": "classify-1",
    }
    data.update(overrides)
    return data


def test_request_preserves_exact_text_and_has_no_api_key_field() -> None:
    classification_request = request()

    assert classification_request.source_text == "  source text \n"
    assert classification_request.target_text == "\ttarget text  "
    assert classification_request.model_id == "classify-1"
    with pytest.raises(ValidationError):
        CDMClassificationRequest(
            candidate_id="candidate-1",
            source=classification_request.source,
            target=classification_request.target,
            source_text=classification_request.source_text,
            target_text=classification_request.target_text,
            api_key="secret",
        )


@pytest.mark.parametrize("field", ["source_text", "target_text"])
def test_request_rejects_whitespace_only_chunk_text(field: str) -> None:
    values = request().model_dump()
    values[field] = " \t\n "

    with pytest.raises(ValidationError, match="Chunk text must not be blank"):
        CDMClassificationRequest.model_validate(values)


@pytest.mark.parametrize(
    ("target_document", "target_index", "target_page"),
    [("other-doc", 4, 5), ("doc-1", 4, 4), ("doc-1", 3, 5)],
)
def test_request_rejects_invalid_chunk_references(
    target_document: str, target_index: int, target_page: int
) -> None:
    with pytest.raises(ValidationError):
        CDMClassificationRequest(
            candidate_id="candidate-1",
            source=ChunkReference(
                document_id="doc-1", chunk_index=3, page_number=4
            ),
            target=ChunkReference(
                document_id=target_document,
                chunk_index=target_index,
                page_number=target_page,
            ),
            source_text="source",
            target_text="target",
        )


def test_normalized_result_accepts_four_valid_probabilities_and_usage() -> None:
    result = CDMClassificationResult.model_validate(
        result_data(
            usage={"input_tokens": 374, "output_tokens": 1, "cost": 0.0000187}
        )
    )

    assert result.relation_type is ContinuityRelation.SAME_PARAGRAPH
    assert result.usage == CDMUsage(
        input_tokens=374, output_tokens=1, cost=0.0000187
    )


@pytest.mark.parametrize(
    "bad_probabilities",
    [
        {"SAME_PARAGRAPH": 0.7, "TABLE_CONTINUATION": 0.1, "UNKNOWN": 0.2},
        {
            "SAME_PARAGRAPH": 0.6,
            "TABLE_CONTINUATION": 0.1,
            "NO_CONTINUATION": 0.1,
            "UNKNOWN": 0.2,
            "OTHER": 0.0,
        },
    ],
)
def test_result_rejects_missing_or_unexpected_relation_labels(
    bad_probabilities: dict[str, float],
) -> None:
    with pytest.raises(ValidationError):
        CDMClassificationResult.model_validate(
            result_data(probabilities=bad_probabilities)
        )


@pytest.mark.parametrize(
    "bad_probabilities",
    [
        {"SAME_PARAGRAPH": -0.1, "TABLE_CONTINUATION": 0.4, "NO_CONTINUATION": 0.4, "UNKNOWN": 0.3},
        {"SAME_PARAGRAPH": 1.1, "TABLE_CONTINUATION": 0.0, "NO_CONTINUATION": 0.0, "UNKNOWN": 0.0},
        {"SAME_PARAGRAPH": float("nan"), "TABLE_CONTINUATION": 0.0, "NO_CONTINUATION": 0.0, "UNKNOWN": 1.0},
        {"SAME_PARAGRAPH": 0.3, "TABLE_CONTINUATION": 0.3, "NO_CONTINUATION": 0.3, "UNKNOWN": 0.3},
    ],
)
def test_result_rejects_out_of_range_nonfinite_or_unnormalized_probabilities(
    bad_probabilities: dict[str, float],
) -> None:
    with pytest.raises(ValidationError):
        CDMClassificationResult.model_validate(
            result_data(probabilities=bad_probabilities)
        )


def test_result_rejects_selected_relation_below_highest_score() -> None:
    with pytest.raises(ValidationError, match="highest-scoring"):
        CDMClassificationResult.model_validate(
            result_data(
                relation_type="UNKNOWN",
                probabilities={
                    "SAME_PARAGRAPH": 0.7,
                    "TABLE_CONTINUATION": 0.1,
                    "NO_CONTINUATION": 0.15,
                    "UNKNOWN": 0.05,
                },
            )
        )


def test_tied_highest_scores_are_valid_selected_outcomes() -> None:
    result = CDMClassificationResult.model_validate(
        result_data(
            relation_type="UNKNOWN",
            probabilities={
                "SAME_PARAGRAPH": 0.4,
                "TABLE_CONTINUATION": 0.1,
                "NO_CONTINUATION": 0.1,
                "UNKNOWN": 0.4,
            },
        )
    )

    assert result.relation_type is ContinuityRelation.UNKNOWN


def test_rounded_provider_probabilities_allow_four_decimal_accumulation() -> None:
    result = CDMClassificationResult.model_validate(
        result_data(
            relation_type="NO_CONTINUATION",
            probabilities={
                "SAME_PARAGRAPH": 0.0090,
                "TABLE_CONTINUATION": 0.0321,
                "NO_CONTINUATION": 0.9590,
                "UNKNOWN": 0.0,
            },
        )
    )

    assert result.probabilities[ContinuityRelation.NO_CONTINUATION] == 0.959


def test_response_adapter_normalizes_observed_decisions_api_shape() -> None:
    response = {
        "id": "scaledown-request-123",
        "request_id": "legacy-request-id",
        "model": "classify-1",
        "answers": {
            "continuity_relation": {
                "choice": "SAME_PARAGRAPH",
                "confidence": 0.7,
                "probabilities": probabilities(),
                "type": "choice",
            }
        },
        "usage": {"input_tokens": 374, "output_tokens": 1, "cost": 0.0000187},
    }

    normalized = adapt_scaledown_response(response, request())

    assert normalized.candidate_id == "candidate-1"
    assert normalized.relation_type is ContinuityRelation.SAME_PARAGRAPH
    assert normalized.model_id == "classify-1"
    assert normalized.request_id == "scaledown-request-123"
    assert normalized.usage is not None
    assert normalized.usage.input_tokens == 374


def test_response_adapter_rejects_unexpected_model_identifier() -> None:
    response = {
        "model": "different-model",
        "answers": {
            "continuity_relation": {
                "choice": "SAME_PARAGRAPH",
                "probabilities": probabilities(),
                "type": "choice",
            }
        },
    }

    with pytest.raises(ValueError, match="unexpected model"):
        adapt_scaledown_response(response, request())


def test_response_adapter_falls_back_to_request_id() -> None:
    response = {
        "request_id": "fallback-request-id",
        "model": "classify-1",
        "answers": {
            "continuity_relation": {
                "choice": "SAME_PARAGRAPH",
                "probabilities": probabilities(),
            "type": "choice",
            }
        },
    }

    assert adapt_scaledown_response(response, request()).request_id == (
        "fallback-request-id"
    )


def test_response_adapter_rejects_non_choice_answer_type() -> None:
    response = {
        "model": "classify-1",
        "answers": {
            "continuity_relation": {
                "choice": "SAME_PARAGRAPH",
                "probabilities": probabilities(),
                "type": "score",
            }
        },
    }

    with pytest.raises(ValueError, match="Expected a choice answer"):
        adapt_scaledown_response(response, request())


@pytest.mark.parametrize("malformed", [{}, {"answers": {}}, {"answers": {"continuity_relation": {}}}])
def test_response_adapter_rejects_malformed_response(malformed: object) -> None:
    with pytest.raises(ValueError, match="ScaleDown"):
        adapt_scaledown_response(malformed, request())
