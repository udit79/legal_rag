"""Early streaming limits for multipart PDF uploads."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

ASGIMessage = dict[str, Any]
ASGIReceive = Callable[[], Awaitable[ASGIMessage]]
ASGISend = Callable[[ASGIMessage], Awaitable[None]]
ASGIApp = Callable[[dict[str, Any], ASGIReceive, ASGISend], Awaitable[None]]


class _UploadBodyTooLarge(Exception):
    pass


class UploadBodyLimitMiddleware:
    """Abort ingestion request bodies while Starlette is parsing multipart data.

    The limit includes multipart framing; the endpoint separately enforces the
    exact PDF file-size limit after the uploaded file is available.
    """

    def __init__(self, app: ASGIApp, *, max_request_bytes: int) -> None:
        self.app = app
        self.max_request_bytes = max_request_bytes

    async def __call__(
        self, scope: dict[str, Any], receive: ASGIReceive, send: ASGISend
    ) -> None:
        if not (
            scope.get("type") == "http"
            and scope.get("method") == "POST"
            and scope.get("path") in {"/ingestion", "/ingestion/batch"}
        ):
            await self.app(scope, receive, send)
            return

        received_bytes = 0

        async def limited_receive() -> ASGIMessage:
            nonlocal received_bytes
            message = await receive()
            if message.get("type") == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_request_bytes:
                    raise _UploadBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _UploadBodyTooLarge:
            body = json.dumps(
                {"detail": "PDF request body exceeds the upload limit"}
            ).encode("utf-8")
            await send(
                {
                    "type": "http.response.start",
                    "status": 413,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": body,
                    "more_body": False,
                }
            )
