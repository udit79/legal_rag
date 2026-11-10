# Isolated PDF parser worker

The API uses the published image by default through `.env`:

```dotenv
PARSER_WORKER_IMAGE=ghcr.io/udit79/garph-rag-parser-worker:latest
```

Docker pulls it on first use if it is not cached. For a private GHCR package,
authenticate the Docker CLI with `docker login ghcr.io` first. To build locally
instead, use the repository root command:

```powershell
docker build -f docker/parser-worker.Dockerfile -t ghcr.io/udit79/garph-rag-parser-worker:latest .
```

The build installs dependencies from the hash-pinned
`requirements-parser-worker.lock`, then downloads and initializes Docling model
assets into the image. It therefore needs network access once and may produce
a large image. The running worker is started by FastAPI with networking
disabled, a 2 GiB memory limit, a two-CPU limit, a read-only root filesystem,
and a read-only mount containing the current PDF. The worker stays warm for the
application lifetime and returns validated canonical JSON to the API process.

Docker Desktop must be running before the API starts. The API fails startup if
the configured image or Docker engine is unavailable. Each parse is limited to
50 MiB and 500 pages. Its timeout scales at 60 seconds per page, capped at 30
minutes for the full document. A timed-out container is killed before the next
request starts a replacement worker.

Tesseract uses automatic page segmentation by default. Set `TESSERACT_LANG` to
one or more installed language codes (for example, `eng+hin`) and
`TESSERACT_PSM` to a mode from 0 to 13 in the API environment to override the
defaults. Required Tesseract language data must be present in the image.

The 2 GiB memory cap is required for Docling model initialization. Ensure
Docker Desktop has enough memory available; a worker that is OOM-killed cannot
process PDF requests.
