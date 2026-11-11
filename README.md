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

`POST /ingestion/batch` accepts repeated multipart `files` fields for up to 10
PDFs. Each file runs independently; the endpoint returns a compact per-file
status, document ID, ingestion run ID, page/chunk counts, and failure code,
without chunks or image bytes. The aggregate PDF payload is limited to 50 MiB,
plus multipart framing overhead.

The JSON response contains `revised_chunks`, `unrecognized_images`, and compact
`chunk_dispositions`. A positive `SAME_PARAGRAPH` continuity result keeps the
original chunks in order and prefixes the continuation chunk with a bounded,
source-backed tail from the preceding page. The output stays within the
configured character limit; this is context overlap, not a calibrated semantic
merge. Image crops include OCR status, admission decision, flags, page/block
metadata, and base64 PNG data. Uploads are limited to 50 MiB.
`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` is optional; when set, an OTLP HTTP
collector must be reachable at that URL or trace export warnings will appear.
For local Jaeger, publish port 4318 to the host; Jaeger UI is available on port
16686.

Docling runs in a persistent, restricted Docker parser worker. Build the image
and configure the local ClamAV installation before starting the API; see
[`docker/PARSER_WORKER.md`](docker/PARSER_WORKER.md) and
[`tools/clamav/README.md`](tools/clamav/README.md).

If a page reaches the `ESCALATE` route, or local OCR returns no usable page
text, the parser worker exports that page as a bounded single-page PDF and the
API sends only that page to ScaleDown Extract. Its raw `ocr_text` is added as
a canonical page text block with a fallback warning. This requires
`SCALEDOWN_API_KEY`; the recovered block has page-level coordinates because
Extract does not return word-level geometry.

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
