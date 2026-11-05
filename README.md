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

Docling's normal and layout-OCR pipelines are initialized during server startup
and reused between uploads. They are released after 30 minutes without a
Docling parse by default. Set `DOCLING_IDLE_TIMEOUT_SECONDS` to change the idle
period; set it to `0` to keep the pipelines resident until server shutdown.

See [docs/architecture.md](docs/architecture.md) for the current component
structure and data contracts. See [docs/code_map.md](docs/code_map.md) for
the file tree and a file-by-file index of implemented classes and functions.
The implemented BLP-1 chunking strategy is described in
[docs/blp1_chunking.md](docs/blp1_chunking.md).
