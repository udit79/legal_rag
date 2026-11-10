import asyncio

from src.api.upload_limits import UploadBodyLimitMiddleware


def test_upload_body_limit_stops_stream_and_returns_413():
    messages = iter(
        (
            {"type": "http.request", "body": b"123", "more_body": True},
            {"type": "http.request", "body": b"456", "more_body": False},
        )
    )
    sent = []

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    async def app(scope, app_receive, app_send):
        while True:
            message = await app_receive()
            if not message.get("more_body", False):
                break

    middleware = UploadBodyLimitMiddleware(app, max_request_bytes=5)
    asyncio.run(
        middleware(
            {"type": "http", "method": "POST", "path": "/ingestion"},
            receive,
            send,
        )
    )

    assert sent[0]["status"] == 413
    assert b"upload limit" in sent[1]["body"]


def test_upload_body_limit_does_not_affect_other_routes():
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"123456", "more_body": False}

    async def send(message):
        sent.append(message)

    async def app(_scope, _receive, app_send):
        await app_send({"type": "http.response.start", "status": 200, "headers": []})
        await app_send({"type": "http.response.body", "body": b"ok"})

    middleware = UploadBodyLimitMiddleware(app, max_request_bytes=1)
    asyncio.run(
        middleware(
            {"type": "http", "method": "POST", "path": "/health"},
            receive,
            send,
        )
    )

    assert sent[0]["status"] == 200
