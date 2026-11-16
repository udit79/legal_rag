---
name: project-setup
description: Set up the legal_rag development environment on Windows, including Docker parser worker, ClamAV, Supabase, Neo4j, tracing, environment variables and local startup.
---

# legal_rag project setup

This is the repository setup guide for developers and coding agents. Follow it
from top to bottom on a fresh Windows checkout. Keep credentials in the local
`.env` file; never commit or paste them into source files, issues or chat.

## 1. Prerequisites

- Windows 10/11 with PowerShell.
- Git.
- Python 3.14 (the project requires `>=3.14,<3.15`) and `uv`.
- Docker Desktop, running with Linux containers enabled.
- A Neo4j database on Neo4j 2026.09 or later. Use Neo4j Aura/sandbox or a
  local Neo4j installation. This version supports the repository's Cypher 25
  full-text `SEARCH` queries.
- A Supabase project for provenance, private image storage and (optionally)
  chat persistence.
- Provider API keys only for features you enable: OpenAI or Voyage embeddings,
  OpenAI chat/VLM, and ScaleDown page-recovery extraction.

Open PowerShell in the repository root for the commands below.

Install Docker Desktop from <https://docs.docker.com/desktop/setup/install/windows-install/>.
Install Python 3.14 and `uv` using your standard Windows package manager or the
official installers before continuing.

## 2. Start Docker and download the containers

Start Docker Desktop, then verify the engine:

```powershell
docker version
docker info
```

Download the parser worker used by this checkout:

```powershell
docker pull ghcr.io/udit79/garph-rag-parser-worker:latest
```

If GHCR reports that the package is private, authenticate once with
`docker login ghcr.io`, then repeat the pull. The API starts and manages the parser
worker itself; do not start a second parser container manually. The worker runs
with network disabled, a read-only root filesystem, the current PDF mounted
read-only, a 2 GiB memory limit, and a two-CPU limit. Docker Desktop needs at
least 2 GiB available for Docling initialization.

If the published image cannot be pulled, build it from the repository root:

```powershell
docker build -f docker/parser-worker.Dockerfile `
  -t ghcr.io/udit79/garph-rag-parser-worker:latest .
```

The build downloads Docling model assets and may take time and disk space.
Parser requests are limited to 50 MiB and 500 pages, with a maximum 30-minute
parse timeout. Optional Tesseract settings are `TESSERACT_LANG` and
`TESSERACT_PSM`.

### Jaeger tracing container

Jaeger is optional for application startup, but recommended for viewing traces.
The v2 all-in-one container stores traces in memory, so traces disappear when
the container is removed or restarted. Pull and start it:

```powershell
docker pull jaegertracing/jaeger:2.23.0
docker run -d --name graph-rag-jaeger --restart unless-stopped `
  -p 16686:16686 -p 4317:4317 -p 4318:4318 `
  jaegertracing/jaeger:2.23.0
```

Open <http://localhost:16686> to view traces. The API's `.env.example` points
OTLP HTTP traces at `http://localhost:4318/v1/traces`. If port 16686 or 4318 is
already in use, change the host-side port mapping and update the endpoint.

To stop Jaeger without deleting it, run `docker stop graph-rag-jaeger`; start it
again with `docker start graph-rag-jaeger`.

## 3. Prepare ClamAV signatures and logs

The Windows ClamAV scanner is included at
`tools/clamav/1.5.4/windows-x64`. Its signature database and logs are local
runtime files and are intentionally excluded from Git. From the repository
root, create the directories and FreshClam configuration, then download the
current signatures:

```powershell
$ClamHome = (Resolve-Path "tools/clamav/1.5.4/windows-x64").Path
$Database = Join-Path $ClamHome "database"
$Logs = Join-Path $ClamHome "logs"
$Config = Join-Path $ClamHome "freshclam.local.conf"
New-Item -ItemType Directory -Force $Database, $Logs | Out-Null
@"
DatabaseDirectory "$Database"
UpdateLogFile "$(Join-Path $Logs 'freshclam.log')"
LogTime yes
CVDCertsDirectory "$(Join-Path $ClamHome 'certs')"
DatabaseMirror database.clamav.net
"@ | Set-Content -LiteralPath $Config -Encoding ascii
& (Join-Path $ClamHome "freshclam.exe") "--config-file=$Config"
```

Confirm the database files exist and inspect updater output:

```powershell
Get-ChildItem $Database
Get-Content (Join-Path $Logs "freshclam.log") -Tail 30
```

To refresh signatures later, rerun the `freshclam.exe` command. A clean result
usually means the signatures are current; investigate updater errors before
starting ingestion. G0 fails closed when the scanner or database is missing.

Set the corresponding paths in `.env` (use forward slashes or quoted Windows
paths):

```ini
CLAMAV_BINARY=D:/garph_rag/tools/clamav/1.5.4/windows-x64/clamscan.exe
CLAMAV_DATABASE_DIRECTORY=D:/garph_rag/tools/clamav/1.5.4/windows-x64/database
```

Optionally test a PDF and capture the scan log:

```powershell
& "$ClamHome\clamscan.exe" `
  "--database=$Database" `
  "--log=$(Join-Path $Logs 'clamscan.log')" `
  "D:\path\to\sample.pdf"
Get-Content (Join-Path $Logs "clamscan.log") -Tail 30
```

ClamAV exit codes are 0 (clean), 1 (malware found), and 2 (scanner error). The
application writes its own service logs to the console; `clamscan.log` above is
for manual checks, while `freshclam.log` records signature updates.

## 4. Create and configure Supabase

1. Create a project at <https://supabase.com/dashboard> and save its database
   password.
2. In the project dashboard, open **SQL Editor** and run
   [`supabase/setup.sql`](supabase/setup.sql) once for a fresh project. It is
   the consolidated bootstrap for this repository, including provenance,
   private image storage, persistence RPCs and chat checkpoint schema.
3. In **Project Settings → API**, copy the project URL and service-role key.
   The key belongs only in the backend `.env`; never put it in the frontend.
4. In **Connect**, copy the direct PostgreSQL connection string. If the host
   network cannot use the direct IPv6 endpoint, use Supabase's **session
   pooler** connection on port 5432. Do not use the transaction pooler on port
   6543: the chat checkpointer needs a persistent/session connection with
   prepared-statement support.

Set these values in `.env`:

```ini
SUPABASE_URL=https://YOUR_PROJECT_REF.supabase.co
SUPABASE_SERVICE_ROLE_KEY=YOUR_SERVICE_ROLE_KEY
SUPABASE_DATABASE_URL=postgresql://...direct-or-session-connection...
```

`SUPABASE_DATABASE_URL` is needed for persistent chat history. The REST URL and
service-role key are used for ingestion provenance, storage and corpus status.
The image bucket is private; the backend resolves image references without
publishing a public URL.

**Existing project warning:** if this Supabase project already has this
application schema, do not rerun the full bootstrap. Check the deployed schema
and apply only migrations that are genuinely missing. `setup.sql` is intended
for initial setup of a fresh project.

## 5. Create or select Neo4j

Use [Neo4j Aura](https://console.neo4j.io/) or a local Neo4j DBMS. For this codebase, choose Neo4j
2026.09+ and enable/use Cypher 25. Copy the Bolt URI and credentials from the
provider dashboard or local DBMS. Common URI forms are `neo4j+s://...` for a
hosted TLS connection and `bolt://localhost:7687` for local development.

Set the database name explicitly. Use `neo4j` for a newly created default
database, or the name of the database you created (for example, `rag`):

```ini
NEO4J_URI=neo4j+s://YOUR_INSTANCE.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=YOUR_NEO4J_PASSWORD
NEO4J_DATABASE=rag
```

The application does not create your Neo4j DBMS or database. It connects to the
database named here. Structural graph schema/import and embedding indexing are
separate explicit operations; starting the API does not create corpus nodes or
generate embeddings.

After source ingestion is persisted, inspect the corpus import options and
manifest before writing:

```powershell
uv run python -m src.graph.corpus_cli --help
uv run python -m src.graph.corpus_cli --manifest path\to\manifest.json
```

The second command is dry-run by default. `--apply` performs Neo4j writes;
`--verify-idempotency` is available with `--apply`. `--visual-knowledge` also
requires an explicitly isolated Neo4j database configured through
`NEO4J_TEST_DATABASE`. Avoid using live/production databases for initial
validation.

## 6. Configure the application environment

Create the local environment file and edit it:

```powershell
Copy-Item .env.example .env
notepad .env
```

Fill in the ClamAV, Supabase and Neo4j values from the earlier sections. Then
set only the model keys and options needed for your features:

| Feature | Environment settings |
| --- | --- |
| Page recovery | `SCALEDOWN_API_KEY` |
| Text embeddings | `EMBEDDING_PROVIDER` (`openai` or `voyage`), `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`, and the matching API key |
| Document chat | `SUPABASE_DATABASE_URL`, `OPENAI_API_KEY`, embeddings, Neo4j, Supabase REST settings |
| Image VLM descriptions | `IMAGE_VLM_PERSISTENCE_ENABLED=true`, `IMAGE_VLM_PROVIDER=openai`, `IMAGE_VLM_MODEL`, and `OPENAI_API_KEY` |
| Tracing | `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` (optional; set empty to disable export) |

Do not change the embedding model or dimension when you already have a vector
index populated with another embedding space. The provider, model, dimensions
and index must match. `CHAT_MODEL` defaults to `gpt-4o-mini` in `.env.example`.

## 7. Install dependencies and start the API

From the repository root:

```powershell
uv sync
uv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

The API startup checks the ClamAV scanner/signatures, Docker parser worker and
Neo4j connection. Keep Docker Desktop running. Readiness should return a
successful Neo4j check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/ready
```

Open <http://127.0.0.1:8000/docs> for the API routes. Upload a small test PDF
through the ingestion endpoint or the UI. If `SUPABASE_URL` and
`SUPABASE_SERVICE_ROLE_KEY` are configured, ingestion provenance and image
records use that Supabase project. Use a disposable/test corpus for initial
checks.

To use the bundled development UI, open a second PowerShell window in the
repository root and run:

```powershell
uv run uvicorn server:app --app-dir frontend --host 127.0.0.1 --port 3000
```

Then open <http://127.0.0.1:3000> and choose **Live API** in the UI settings.
It connects to the backend at `http://127.0.0.1:8000` by default. The
`frontend/server.py` process also contains offline/demo endpoints backed by
sample in-memory data; use Live API mode for real ingestion and graph/chat
results. Keep this development host local.

For traces, open Jaeger at <http://localhost:16686>. Select the service emitted
by the API (currently `graph_rag.api`). With the in-memory Jaeger setup, traces
are temporary.

## 8. Common setup checks

- **Parser startup error:** check `docker info`, then pull the configured image
  again. Verify `PARSER_WORKER_IMAGE` exactly matches the available tag.
- **ClamAV initialization fails:** verify the `database` directory contains
  current signature files and that both `.env` paths point to this Windows
  checkout.
- **Neo4j connection fails:** verify Bolt URI, database name, credentials,
  network access and that the database is online.
- **Supabase chat connection fails:** use the direct connection or session
  pooler (port 5432), verify the password is URL-encoded when it has reserved
  characters, and ensure `supabase/setup.sql` was applied to a fresh project.
- **No traces appear:** verify Jaeger is running, port 4318 is published and
  `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` ends in `/v1/traces`.
- **Embedding or retrieval reports index mismatch:** confirm configured
  provider/model/dimensions match the stored vectors and configured Neo4j
  vector index. Do not clear or recreate graph data as a setup workaround.
