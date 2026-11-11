import asyncio
from io import BytesIO

from fastapi import HTTPException, Request
from starlette.datastructures import Headers, UploadFile

from src.api import main
from src.ingestion.pipeline import CPCRIngestionResponse


def _upload(filename: str) -> UploadFile:
    return UploadFile(
        filename=filename,
        file=BytesIO(b"pdf"),
        headers=Headers({"content-type": "application/pdf"}),
    )


def test_batch_keeps_failures_independent_and_returns_only_compact_results(
    monkeypatch,
):
    observed_run_ids = []
    active = 0
    max_active = 0

    async def process_pdf(request, file, *, state):
        nonlocal active, max_active
        assert request.url.path == "/ingestion/batch"
        observed_run_ids.append(state.ingestion_run_id)
        active += 1
        max_active = max(max_active, active)
        state.document_id = f"document-{file.filename}"
        state.page_count = 2
        state.chunk_count = 3
        await asyncio.sleep(0.01)
        active -= 1
        if file.filename == "bad.pdf":
            raise HTTPException(422, detail={"code": "NO_ELIGIBLE_CONTENT"})
        return main._ProcessedPdf(
            response=CPCRIngestionResponse(
                revised_chunks=(),
                unrecognized_images=(),
                chunk_dispositions=(),
            ),
            document_id=state.document_id,
        )

    monkeypatch.setattr(main, "_process_pdf_upload", process_pdf)
    request = Request(
        {"type": "http", "method": "POST", "path": "/ingestion/batch", "headers": []}
    )

    response = asyncio.run(
        main.ingest_pdf_batch_endpoint(
            request,
            [_upload("one.pdf"), _upload("bad.pdf"), _upload("three.pdf")],
        )
    )

    assert response.batch_id
    assert [item.status for item in response.files] == [
        "SUCCEEDED",
        "FAILED",
        "SUCCEEDED",
    ]
    assert response.files[1].failure_code == "NO_ELIGIBLE_CONTENT"
    assert all(item.page_count == 2 and item.chunk_count == 3 for item in response.files)
    assert len(set(observed_run_ids)) == 3
    assert max_active == main.MAX_BATCH_CONCURRENCY
    assert "revised_chunks" not in response.model_dump_json()
    assert "data_base64" not in response.model_dump_json()
