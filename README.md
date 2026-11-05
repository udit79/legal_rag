# Graph RAG

Graph RAG is a Python service foundation for connecting source documents to a
validated risk-domain graph. The repository contains API health checks, Neo4j
connectivity, telemetry, canonical PDF parser adapters, BLP-1 deterministic
chunking, and CPCR page-continuity classification. The `POST /ingestion`
endpoint runs the full pipeline for an uploaded PDF and returns the CPCR result.
It also returns PNG crops for image blocks with no recognized text. Canonical
output and chunking details are not included.

Upload a PDF with multipart form data:

```powershell
curl.exe -X POST http://localhost:8000/ingestion `
  -F "file=@D:\path\to\document.pdf;type=application/pdf" `
  -o ingestion-result.json
```

The JSON response contains only `revised_chunks` and `unrecognized_images`.
Revised chunks merge positive `SAME_PARAGRAPH` and `TABLE_CONTINUATION`
relationships; failed or non-positive classifications leave chunks separate.
Each revised chunk retains source provenance, and image crops include
page/block metadata and base64 PNG data. Uploads are limited to 50 MiB. Set
`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` to export HTTP, per-page parser, and
ingestion-stage traces.

Docling runs in a persistent, restricted Docker parser worker. Build the image
and configure the local ClamAV installation before starting the API; see
[`docker/PARSER_WORKER.md`](docker/PARSER_WORKER.md) and
[`tools/clamav/README.md`](tools/clamav/README.md).

The Windows x64 ClamAV scanner runtime is included under
`tools/clamav/1.5.4/windows-x64` and runs on the Windows API host. The
signature database, local configuration and logs are not committed. Configure
`CLAMAV_BINARY` and `CLAMAV_DATABASE_DIRECTORY` in the local `.env`, then use
`freshclam.exe` to populate/update the database; setup steps are in
[`tools/clamav/README.md`](tools/clamav/README.md). G0 fails closed when the
scanner or its signature database is unavailable. The API has no user login;
anonymous uploads remain marked T3 and are still screened before parsing and
CPCR.

See [docs/architecture.md](docs/architecture.md) for the current component
structure and data contracts. See [docs/code_map.md](docs/code_map.md) for
the file tree and a file-by-file index of implemented classes and functions.
The implemented BLP-1 chunking strategy is described in
[docs/blp1_chunking.md](docs/blp1_chunking.md).
