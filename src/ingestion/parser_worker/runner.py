"""Monitors and communicates with the network-isolated Docker parser."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Protocol

from src.ingestion.parser_worker.exceptions import (
    ParserWorkerError,
    ParserWorkerParseError,
    ParserWorkerProtocolError,
    ParserWorkerTimeout,
    ParserWorkerUnavailable,
)
from src.ingestion.parser_worker.limits import (
    MAX_EXECUTION_SECONDS,
    MAX_MEMORY_BYTES,
    MAX_PDF_SIZE_BYTES,
    MAX_WORKER_RESPONSE_BYTES,
    MAX_OCR_BATCH_BYTES,
    MAX_OCR_BATCH_IMAGES,
    WORKER_MOUNT_PATH,
)
from src.ingestion.parser_worker.result import (
    ImageOcrInput,
    ImageOcrRequest,
    ImageOcrResponse,
    ParseWorkerRequest,
    ParseWorkerResponse,
)


class ImageCropInput(Protocol):
    page_number: int
    block_id: str
    data_base64: str


class ParserWorkerRunner:
    """Own one warm Docker worker and serialize bounded parse requests to it."""

    def __init__(
        self,
        *,
        image: str | None = None,
        docker_executable: str | None = None,
        startup_timeout_seconds: float = 600.0,
        execution_timeout_seconds: int = MAX_EXECUTION_SECONDS,
    ) -> None:
        self.image = image or os.environ.get(
            "PARSER_WORKER_IMAGE", "garph-rag-parser-worker:latest"
        )
        self.docker_executable = docker_executable or os.environ.get(
            "DOCKER_EXECUTABLE", "docker"
        )
        self.startup_timeout_seconds = startup_timeout_seconds
        self.execution_timeout_seconds = execution_timeout_seconds
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_tail: list[str] = []
        self._container_name: str | None = None
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        self._request_lock = asyncio.Lock()
        self._closed = False

    @property
    def shared_directory(self) -> Path:
        if self._temporary_directory is None:
            self._temporary_directory = tempfile.TemporaryDirectory(
                prefix="graph-rag-parser-worker-"
            )
        return Path(self._temporary_directory.name)

    async def start(self) -> None:
        """Start the restricted worker and wait until parser models are ready."""
        if self._closed:
            raise ParserWorkerUnavailable("Parser worker runner is already closed")
        if self._process is not None and self._process.returncode is None:
            return

        self._stderr_tail.clear()
        self._container_name = f"garph-rag-parser-{uuid.uuid4().hex[:16]}"
        host_work = str(self.shared_directory.resolve())
        command = [
            self.docker_executable,
            "run",
            "--rm",
            "--init",
            "--name",
            self._container_name,
            "--pull=missing",
            "--network=none",
            f"--memory={MAX_MEMORY_BYTES}",
            f"--memory-swap={MAX_MEMORY_BYTES}",
            "--cpus=2",
            "--pids-limit=128",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=256m",
            "--volume",
            f"{host_work}:{WORKER_MOUNT_PATH}:ro",
            "-i",
            self.image,
        ]
        try:
            self._process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            self._process = None
            raise ParserWorkerUnavailable(
                "Unable to start Docker; install Docker Desktop and start its engine"
            ) from exc

        assert self._process.stderr is not None
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        try:
            response = await asyncio.wait_for(
                self._read_response(), timeout=self.startup_timeout_seconds
            )
        except TimeoutError as exc:
            await self._stop_worker(force=True)
            raise ParserWorkerTimeout("Parser worker startup timed out") from exc
        except Exception as exc:
            detail = self._stderr_message()
            await self._stop_worker(force=True)
            raise ParserWorkerUnavailable(
                f"Parser worker did not become ready{': ' + detail if detail else ''}"
            ) from exc
        if response.type != "ready":
            await self._stop_worker(force=True)
            raise ParserWorkerProtocolError("Parser worker sent no readiness message")

    async def parse_pdf(
        self, source: Path, routing_policy_version: str
    ) -> ParseWorkerResponse:
        """Preflight and parse a PDF in the worker, validating its JSON response."""
        if not source.is_file():
            raise ParserWorkerError("Parser input file does not exist")
        size = source.stat().st_size
        if size <= 0 or size > MAX_PDF_SIZE_BYTES:
            raise ParserWorkerError("PDF size is outside configured worker limits")
        if not routing_policy_version.strip():
            raise ParserWorkerError("A routing policy version is required")

        async with self._request_lock:
            await self.start()
            request_id = uuid.uuid4().hex
            filename = f"{request_id}.pdf"
            worker_file = self.shared_directory / filename
            try:
                await asyncio.to_thread(shutil.copyfile, source, worker_file)
                request = ParseWorkerRequest(
                    request_id=request_id,
                    filename=filename,
                    routing_policy_version=routing_policy_version,
                )
                try:
                    response = await asyncio.wait_for(
                        self._exchange(request),
                        timeout=self.execution_timeout_seconds,
                    )
                except TimeoutError as exc:
                    await self._stop_worker(force=True)
                    raise ParserWorkerTimeout(
                        f"PDF parsing exceeded {self.execution_timeout_seconds} seconds"
                    ) from exc
                except (BrokenPipeError, ConnectionError, asyncio.IncompleteReadError) as exc:
                    detail = self._stderr_message()
                    await self._stop_worker(force=True)
                    raise ParserWorkerUnavailable(
                        f"Parser worker exited during a request{': ' + detail if detail else ''}"
                    ) from exc

                if response.request_id != request_id:
                    await self._stop_worker(force=True)
                    raise ParserWorkerProtocolError(
                        "Parser worker response ID did not match the request"
                    )
                if response.type == "error":
                    raise ParserWorkerParseError(
                        f"Parser worker failed ({response.error_code}: {response.error_type})"
                    )
                if response.type != "result" or any(
                    value is None
                    for value in (response.features, response.route_plan, response.canonical)
                ):
                    await self._stop_worker(force=True)
                    raise ParserWorkerProtocolError("Parser worker returned incomplete results")
                features = response.features
                route_plan = response.route_plan
                canonical = response.canonical
                assert features is not None and route_plan is not None and canonical is not None
                if (
                    features.page_count != route_plan.page_count
                    or canonical.page_count != route_plan.page_count
                ):
                    await self._stop_worker(force=True)
                    raise ParserWorkerProtocolError(
                        "Worker preflight, route plan and canonical page counts disagree"
                    )
                if features.file_size_bytes != size:
                    await self._stop_worker(force=True)
                    raise ParserWorkerProtocolError(
                        "Worker preflight file size differs from the uploaded PDF"
                    )
                return ParseWorkerResponse.model_validate(
                    response.model_dump(mode="python")
                )
            finally:
                worker_file.unlink(missing_ok=True)

    async def ocr_image_crops(
        self, crops: tuple[ImageCropInput, ...]
    ) -> dict[str, str]:
        """OCR a crop batch in the warm, network-isolated parser worker."""
        if not crops:
            return {}
        if len(crops) > MAX_OCR_BATCH_IMAGES:
            raise ParserWorkerError("OCR image count exceeds the worker limit")
        images = tuple(
            ImageOcrInput(
                image_id=f"{crop.page_number}:{crop.block_id}",
                data_base64=crop.data_base64,
            )
            for crop in crops
        )
        if sum(len(image.data_base64) for image in images) > MAX_OCR_BATCH_BYTES:
            raise ParserWorkerError("OCR image batch exceeds the worker size limit")
        request = ImageOcrRequest(request_id=uuid.uuid4().hex, images=images)
        payload = request.model_dump_json().encode("utf-8")
        if len(payload) > MAX_WORKER_RESPONSE_BYTES:
            raise ParserWorkerError("OCR image request exceeds the worker frame limit")
        async with self._request_lock:
            await self.start()
            try:
                raw = await asyncio.wait_for(
                    self._exchange_payload(payload),
                    timeout=self.execution_timeout_seconds,
                )
            except TimeoutError as exc:
                await self._stop_worker(force=True)
                raise ParserWorkerTimeout("Image OCR exceeded the worker time limit") from exc
            try:
                envelope = json.loads(raw)
                if envelope.get("type") == "error":
                    failure = ParseWorkerResponse.model_validate(envelope)
                    raise ParserWorkerParseError(
                        f"Parser worker image OCR failed ({failure.error_code}: {failure.error_type})"
                    )
                response = ImageOcrResponse.model_validate(envelope)
            except ParserWorkerParseError:
                raise
            except Exception as exc:
                await self._stop_worker(force=True)
                raise ParserWorkerProtocolError(
                    "Parser worker returned invalid image OCR results"
                ) from exc
            expected_ids = tuple(image.image_id for image in images)
            actual_ids = tuple(item.image_id for item in response.results)
            if response.request_id != request.request_id or actual_ids != expected_ids:
                await self._stop_worker(force=True)
                raise ParserWorkerProtocolError(
                    "Image OCR results do not match the request order and IDs"
                )
            return {item.image_id: item.text for item in response.results}

    async def _exchange(self, request: ParseWorkerRequest) -> ParseWorkerResponse:
        payload = await self._exchange_payload(
            request.model_dump_json().encode("utf-8")
        )
        try:
            return ParseWorkerResponse.model_validate_json(payload)
        except Exception as exc:
            raise ParserWorkerProtocolError(
                "Parser worker returned invalid structured JSON"
            ) from exc

    async def _exchange_payload(self, payload: bytes) -> bytes:
        process = self._process
        if process is None or process.returncode is not None or process.stdin is None:
            raise ConnectionError("Parser worker is not running")
        if not 0 < len(payload) <= MAX_WORKER_RESPONSE_BYTES:
            raise ParserWorkerProtocolError("Parser worker request size is invalid")
        process.stdin.write(len(payload).to_bytes(4, "big") + payload)
        await process.stdin.drain()
        return await self._read_response_payload()

    async def _read_response(self) -> ParseWorkerResponse:
        payload = await self._read_response_payload()
        try:
            return ParseWorkerResponse.model_validate_json(payload)
        except Exception as exc:
            raise ParserWorkerProtocolError(
                "Parser worker returned invalid structured JSON"
            ) from exc

    async def _read_response_payload(self) -> bytes:
        process = self._process
        if process is None or process.stdout is None:
            raise ConnectionError("Parser worker has no response stream")
        header = await process.stdout.readexactly(4)
        size = int.from_bytes(header, "big")
        if size <= 0 or size > MAX_WORKER_RESPONSE_BYTES:
            raise ParserWorkerProtocolError("Parser worker response size is invalid")
        return await process.stdout.readexactly(size)

    async def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        while line := await process.stderr.readline():
            self._stderr_tail.append(line.decode("utf-8", errors="replace").rstrip())
            del self._stderr_tail[:-30]

    def _stderr_message(self) -> str:
        return " ".join(self._stderr_tail)[-1200:]

    async def _stop_worker(self, *, force: bool) -> None:
        process = self._process
        if process is None:
            return
        if not force and process.stdin is not None:
            process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except TimeoutError:
                force = True
        if force and process.returncode is None:
            if self._container_name is not None:
                try:
                    killer = await asyncio.create_subprocess_exec(
                        self.docker_executable,
                        "kill",
                        self._container_name,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    await asyncio.wait_for(killer.wait(), timeout=5)
                except (OSError, TimeoutError):
                    pass
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except TimeoutError:
                    process.kill()
                    await process.wait()
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass
        self._process = None
        self._stderr_task = None
        self._container_name = None

    async def close(self) -> None:
        """Stop the worker and remove the host-side temporary input directory."""
        if self._closed:
            return
        self._closed = True
        await self._stop_worker(force=False)
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None
