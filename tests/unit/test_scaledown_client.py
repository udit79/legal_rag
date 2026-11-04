import asyncio
import json
from collections.abc import Callable

import httpx
import pytest

from src.ingestion.continuity_cdm_contracts import CDMClassificationRequest
from src.ingestion.continuity_contracts import ChunkReference
from src.ingestion.scaledown_client import (
    SCALEDOWN_DECISIONS_PATH,
    ScaleDownAuthenticationError,
    ScaleDownClient,
    ScaleDownQuotaError,
    ScaleDownRateLimitError,
    ScaleDownResponseError,
    ScaleDownServerError,
    ScaleDownTimeoutError,
)


def classification_request() -> CDMClassificationRequest:
    return CDMClassificationRequest(
        candidate_id="candidate-1",
        source=ChunkReference(document_id="doc-1", chunk_index=3, page_number=4),
        target=ChunkReference(document_id="doc-1", chunk_index=4, page_number=5),
        source_text="  source text \n",
        target_text="\ttarget text  ",
    )


def successful_response() -> dict[str, object]:
    return {
        "model": "classify-1",
        "answers": {
            "continuity_relation": {
                "choice": "SAME_PARAGRAPH",
                "confidence": 0.7,
                "probabilities": {
                    "SAME_PARAGRAPH": 0.7,
                    "TABLE_CONTINUATION": 0.1,
                    "NO_CONTINUATION": 0.15,
                    "UNKNOWN": 0.05,
                },
                "type": "choice",
            }
        },
        "usage": {"input_tokens": 374, "output_tokens": 1, "cost": 0.0000187},
    }


def run_classify(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    api_key: str | None = "test-secret",
    api_base: str | None = "https://example.test",
) -> object:
    async def run() -> object:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await ScaleDownClient(
                api_key=api_key,
                api_base=api_base,
                http_client=client,
            ).classify(classification_request())

    return asyncio.run(run())


def test_client_sends_expected_decisions_request_and_parses_response() -> None:
    observed: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["method"] = request.method
        observed["url"] = str(request.url)
        observed["api_key"] = request.headers.get("x-api-key")
        observed["body"] = json.loads(request.content)
        return httpx.Response(200, json=successful_response())

    result = run_classify(handler)

    assert observed["method"] == "POST"
    assert observed["url"] == f"https://example.test{SCALEDOWN_DECISIONS_PATH}"
    assert observed["api_key"] == "test-secret"
    body = observed["body"]
    assert isinstance(body, dict)
    assert body["model"] == "classify-1"
    assert body["state"] == (
        "SOURCE PAGE 4:\n  source text \n\n\nTARGET PAGE 5:\n\ttarget text  "
    )
    assert body["questions"]["continuity_relation"]["type"] == "choice"
    assert "test-secret" not in json.dumps(body)
    assert result.candidate_id == "candidate-1"
    assert result.model_id == "classify-1"


def test_client_trims_explicit_api_key_before_authentication() -> None:
    observed: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request.headers.get("x-api-key"))
        return httpx.Response(200, json=successful_response())

    run_classify(handler, api_key="  test-secret \t")

    assert observed == ["test-secret"]


def test_client_prefers_scaledown_base_url_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SCALEDOWN_BASE_URL", "https://preferred.example")
    monkeypatch.setenv("SCALEDOWN_API_BASE", "https://legacy.example")
    observed: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(str(request.url))
        return httpx.Response(200, json=successful_response())

    run_classify(handler, api_base=None)

    assert observed == [
        "https://preferred.example/v1/scaledown",
    ]


@pytest.mark.parametrize(
    ("status_code", "exception"),
    [
        (401, ScaleDownAuthenticationError),
        (403, ScaleDownAuthenticationError),
        (402, ScaleDownQuotaError),
        (429, ScaleDownRateLimitError),
        (503, ScaleDownServerError),
    ],
)
def test_client_maps_http_failures_without_retry(
    status_code: int, exception: type[Exception]
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status_code, json={"error": "failure"})

    with pytest.raises(exception):
        run_classify(handler)
    assert calls == 1


def test_client_maps_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(ScaleDownTimeoutError):
        run_classify(handler)


def test_client_rejects_response_from_unrequested_model() -> None:
    body = {
        **successful_response(),
        "model": "different-model",
    }

    with pytest.raises(ScaleDownResponseError, match="expected classification shape"):
        run_classify(lambda request: httpx.Response(200, json=body))


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not-json"),
        httpx.Response(200, json={"answers": {}}),
        httpx.Response(200, json={"answers": {"continuity_relation": {"choice": "UNKNOWN"}}}),
    ],
)
def test_client_rejects_malformed_responses(response: httpx.Response) -> None:
    with pytest.raises(ScaleDownResponseError):
        run_classify(lambda request: response)


def test_client_rejects_missing_key_without_making_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SCALEDOWN_API_KEY", raising=False)
    monkeypatch.delenv("CDM_KEY", raising=False)

    with pytest.raises(ScaleDownAuthenticationError, match="not configured"):
        run_classify(lambda request: pytest.fail("request should not be sent"), api_key=None)
