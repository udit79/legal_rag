# Isolated PDF parser worker

The API uses the published image by default through `.env`:

```dotenv
PARSER_WORKER_IMAGE=ghcr.io/udit79/garph-rag-parser-worker:latest
```

Docker pulls it on first use if it is not cached. For a private GHCR package,
authenticate the Docker CLI with `docker login ghcr.io` first. To build locally
instead, use the repository root command:

```powershell
docker build -f docker/parser-worker.Dockerfile -t garph-rag-parser-worker:latest .
```

The build downloads and initializes Docling model assets into the image. It
therefore needs network access once and may produce a large image. The running
worker is started by FastAPI with networking disabled, a 1 GiB memory limit, a
two-CPU limit, a read-only root filesystem, and a read-only mount containing
the current PDF. The worker stays warm for the application lifetime and returns
validated canonical JSON to the API process.

Docker Desktop must be running before the API starts. The API fails startup if
the configured image or Docker engine is unavailable. Each parse is limited to
50 MiB, 500 pages, and 60 seconds. A timed-out container is killed before the
next request starts a replacement worker.

The 1 GiB memory cap is intentionally enforced. Docling model initialization
may exceed it on some machines; if the container is OOM-killed, increase the
limit only after revising the deployment policy and corresponding worker
limits.
