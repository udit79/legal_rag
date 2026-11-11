# Current Architecture

## Purpose and current scope

This repository is a foundation for a document-backed graph RAG system focused
on enterprise risk concepts. It provides a health/readiness API, Neo4j
connectivity, OpenTelemetry tracing, graph-domain validation, and an
end-to-end PDF ingestion endpoint covering preflight, page routing, parser
dispatch, BLP-1 chunking, and CPCR continuity classification.

The ingestion contracts and state describe how stages can exchange small,
typed values. PDF preflight and deterministic per-page route selection are
implemented, as are replaceable PyMuPDF, Docling, and OCR page parsers with a
shared canonical output contract. Persistent artifact storage and graph writes
are not yet implemented.
See [code_map.md](code_map.md) for the file tree and a function/class-to-file
index.

## Component map

| Area | Location | Responsibility |
| --- | --- | --- |
| API | `src/api/main.py` | FastAPI application, lifecycle setup, health endpoints, and multipart PDF ingestion endpoint. |
| Ingestion contracts | `src/ingestion/contracts.py` | Pydantic models for document references, per-page/document features, and page/document route plans. |
| PDF preflight | `src/ingestion/preflight.py` | SHA-256, file size, page count, and per-page text, image, and drawing counts. |
| Ingestion routing | `src/ingestion/router.py` | Deterministic per-page route selection from validated document features. |
| Parser output contract | `src/ingestion/parser_contracts.py` | Canonical document/page/block models, source coordinates, statuses, and explicit failures. |
| Parser adapters | `src/ingestion/parsers.py` | PyMuPDF native text, Docling structured/table parsing, and Tesseract or RapidOCR. |
| Parser dispatch | `src/ingestion/dispatcher.py` | Selects an adapter per page and preserves page-level failures without dropping remaining pages. |
| Ingestion pipeline | `src/ingestion/pipeline.py` | Traces preflight, routing, per-page parsing, chunking, internal image crops, and CPCR classification. |
| Domain ontology | `src/domain/ontology.py` | Node/relation types and allowed directed relationship triples. |
| Graph schema | `src/domain/graph_schema.py` | Required properties and validation for graph nodes. |
| Neo4j infrastructure | `src/infrastructure/neo4j_client.py` | Async driver setup, connectivity verification, session access, and query execution. |
| Observability | `src/observability/telemetry.py` | OpenTelemetry resource, tracer provider, and OTLP/HTTP exporter setup. |
| Domain model documentation | `docs/domain_model.md` | Definitions and rationale for risk-domain nodes and relationships. |

`main.py` at the repository root is a simple greeting entry point; it is not
the FastAPI service entry point.

## Document ingestion contracts

The current contracts are Pydantic models. They validate data shape; they do
not read or retain PDF contents.

### `DocumentRef`

Identifies a document owned by another system:

- `document_id`: non-empty stable identifier supplied by the owner.
- `source_ref`: non-empty opaque source locator.
- `filename`: non-empty original filename.
- `media_type`: currently restricted to `application/pdf`.

Extra fields are rejected. The contract does not generate identifiers.

### `DocumentFeatures`

Stores preflight metadata rather than the document itself:

- `content_sha256`: 64-character hexadecimal hash, normalized to lowercase.
- `file_size_bytes`: positive integer.
- `page_count`: positive integer.
- `pages`: one `PageFeatures` record for each page, with a one-based page
  number and non-negative text-character, image, and drawing counts.

The number of page records must equal `page_count`, and page numbers must be
contiguous and one-based.

`inspect_pdf(path)` in `src/ingestion/preflight.py` computes the SHA-256 over
the file's original bytes, gets the byte size, and uses PyMuPDF to inspect each
page. Text counts are the character lengths of each page's extracted text;
image and drawing counts come from PyMuPDF. Zero counts are valid values.

Missing paths raise `PdfNotFoundError`. Empty, corrupt, encrypted, non-PDF, or
zero-page inputs raise `InvalidPdfError`. Other filesystem read errors remain
`OSError`s rather than being converted into fabricated metadata.

### `RouteDecision`

Records the selected processing route and its rationale:

- `route`: `FAST_TEXT`, `STRUCTURED_PARSE`, `OCR`, `LAYOUT_OCR`,
  `SKIP_BLANK`, or `ESCALATE`.
- `reason`: non-empty explanatory text.
- `decision_source`: `RULES`, `CDM`, or `FALLBACK`.
- `confidence`: optional value between 0.0 and 1.0.
- `policy_version`: non-empty identifier for the policy version.

The contract models reject unspecified extra fields.

`PageRouteDecision` adds a page number, and `DocumentRoutePlan` validates that
every page has exactly one ordered decision and that all decisions use the
same policy version. The plan also has a document-level disposition:
`PROCESS` or `REJECT_NO_CONTENT`. This is separate from individual page
routes: a blank page in a document with detected content may be `SKIP_BLANK`
while the document remains `PROCESS`. When every page is blank, the document
is `REJECT_NO_CONTENT`. The plan contract rejects inconsistent combinations
of page routes and disposition.

`route_document(features, policy_version)` in `src/ingestion/router.py` applies
the initial deterministic heuristics using
`MIN_NATIVE_TEXT_CHARACTERS` (50). Pages with sufficient text route to
`FAST_TEXT` or, when images/drawings are present, `STRUCTURED_PARSE`. Pages
with less text route to `OCR` when images are present, `LAYOUT_OCR` when
drawings are present, or `ESCALATE` when no visual content is present; pages
with no text or visual content route to `SKIP_BLANK`. If all page decisions
are `SKIP_BLANK`, the document disposition is `REJECT_NO_CONTENT`; otherwise
it is `PROCESS`. Every page decision uses `decision_source="RULES"` and
carries the supplied policy version. These provisional heuristics do not
detect tables or multi-column layouts.

## Parser adapters and canonical output

`dispatch_document(source, route_plan)` in `src/ingestion/dispatcher.py`
dispatches each page independently:

| Route | Adapter |
| --- | --- |
| `FAST_TEXT` | `FastTextParser`, using PyMuPDF native text and visual primitives. |
| `STRUCTURED_PARSE` | `StructuredParser`, using Docling layout and table parsing. |
| `LAYOUT_OCR` | `StructuredParser` with Docling RapidOCR enabled. |
| `OCR` | `OcrParser`, using the Tesseract executable by default or RapidOCR when configured. |
| `SKIP_BLANK` | Validates the page with PyMuPDF and emits a blank result only if it is actually blank. |

The shared output types in `src/ingestion/parser_contracts.py` are
`CanonicalDocument`, `CanonicalPage`, and `CanonicalBlock`. Each block carries
its original one-based page number, text, block type, reading-order index, and
source coordinates. Coordinates identify their units and origin: PyMuPDF and
OCR use top-left PDF points, while Docling provenance retains the origin
reported by Docling (`TOPLEFT` or `BOTTOMLEFT`). OCR confidence is retained
when available.

`POST /ingestion` accepts a multipart `file` upload (`application/pdf`, up to
50 MiB). It runs preflight, routing, parsing, BLP-1 chunking and CPCR, and
returns revised chunks, compact original-chunk dispositions, and base64 PNG
crops for `IMAGE` blocks with blank canonical text. Positive
`SAME_PARAGRAPH` outcomes retain original chunks and prefix the continuation
chunk with a bounded tail from the preceding page. The repeated tail has its
own exact source range, and output chunks stay within the character budget.
Table chunks remain isolated. Each crop includes OCR/admission status, flags,
page, block, reading-order, and coordinate provenance. Canonical output and
full CPCR results are not included in the response.

`POST /ingestion/batch` accepts up to 10 multipart `files` and returns only a
per-file summary; aggregate PDF bytes are limited to 50 MiB, plus multipart
framing overhead. It uses bounded concurrency while preserving the parser
worker's request lock and reports each file's run ID, using an isolated
temporary directory per file.
When Supabase provenance is configured, verified PNG crops are upserted into
the pre-created private `ingestion-image-crops` bucket before their hash,
object path, and size are written through the screening RPC. Base64 image data
and public image URLs are not persisted or returned by the batch endpoint.
If an image upload or screening metadata persistence fails, cleanup first
checks the provenance table and only removes object paths with no recorded
reference; if that check fails, or the RPC outcome is ambiguous due to a
transport error, the private object is retained.

The pipeline emits nested OpenTelemetry spans named `ingestion.run`,
`ingestion.preflight`, `ingestion.routing`, `ingestion.parsing`,
`ingestion.parser.page`, `ingestion.chunking`, `ingestion.image_crops`, and
`ingestion.cpcr.candidate`. Per-page spans include route, parser name, status,
block count, and warning count, making slow parser pages visible. Text and
image bytes are not attached to span attributes.

The FastAPI lifespan initializes and retains shared Docling PDF and layout-OCR
parser instances before yielding readiness. Parser instances are reused for
requests and protected by a lock during conversion. By default, their cached
converters are released after 1,800 seconds without a Docling parse; configure
`DOCLING_IDLE_TIMEOUT_SECONDS` to change this period, or set it to `0` to
disable idle release. A later request recreates and initializes an evicted
pipeline.

## Ingestion screening

The sidecar screening contracts and local G1/G2/G4/G5 batch service live under
`src/security/ingestion_screening/`. G0 validates file identity and limits,
inspects active PDF content, and calls the ClamAV adapter before parser
dispatch. The screening service is invoked once for the complete original
chunk batch after BLP-1. Quarantined or rejected chunk indexes are excluded
from CPCR candidate selection without renumbering or compacting the original
chunk positions. The endpoint places the screening report and admission
records inside `revised_chunks`; original canonical blocks and chunk text are
not mutated.

The current API does not yet provide authenticated tenant or uploader IDs.
Those missing required provenance fields therefore fail closed and quarantine
chunks. G3 learned detection, parser process isolation, decompression-expansion
limits, and G6 graph-write screening remain later work. The Windows x64 ClamAV
runtime is bundled under `tools/clamav/`; its mutable signature database is
kept local and ignored by Git.

`FastTextParser` preserves line boundaries inside each native text block by
joining PyMuPDF lines with newline characters. On OCR routes, native text and
visual blocks are retained and OCR output is merged as additional text. To
avoid duplicates, an OCR line is omitted when its case-folded, whitespace-
normalized text exactly matches a native line or an earlier OCR line and their
source-coordinate rectangles overlap by at least 0.8 relative to the smaller
rectangle. Repeated text at a different location or with less overlap is kept.
New lines in a multi-line OCR result are kept even when another line in that
result duplicates native text. This is exact text matching constrained by
spatial overlap, not fuzzy matching; similar but differently recognized text
may appear in both forms.

Page results distinguish `PARSED`, `BLANK`, `VECTOR_ONLY`, and `FAILED`.
Encrypted or unreadable documents receive a document-level failure; individual
page parser errors are recorded on that page while other pages continue to
parse in order. A page incorrectly routed as blank is checked rather than
silently skipped: detected content is returned with a
`ROUTE_FEATURE_MISMATCH` warning, and corrupt content is recorded as a page
failure. These adapters stop at canonical page output; they do not chunk,
extract domain entities, or write to Neo4j.

The chunk data contracts are defined separately in
`src/ingestion/chunk_contracts.py`. `ChunkSource` references a serialized
stable `CanonicalBlock.block_id` together with document identity, page number,
block reading order, and type. For `TEXT`, `TABLE`, and `OTHER` blocks, it
requires half-open `[start_offset, end_offset)` character ranges and total
source length, all measured in Python Unicode string characters. `IMAGE`
sources may omit ranges when they have no text or provide ranges if the parser
supplies text (Docling's image path retains an item's text when present).
`VECTOR` blocks are currently emitted with empty text and do not accept text
ranges. OCR recognition is emitted as `TEXT` blocks. A chunk can reference
multiple ordered blocks, and one block can be referenced by multiple chunks. A
chunk cannot contain overlapping ranges for the same canonical block, while
adjacent ranges and ranges in separate chunks are valid. The
`ChunkedDocument` records document identity, source page count, and chunking
policy version, and validates zero-based contiguous indices. Contract
serialization is deterministic for the same ordered inputs. BLP-1 chunk
generation is implemented in `src/ingestion/chunking.py` and
`src/ingestion/segmentation.py`: it routes pages, builds lossless source-span
atoms, and packs TEXT, OTHER, TABLE, and IMAGE groups under a character limit.
Block IDs derive deterministically from the one-based page number and
zero-based reading order within a canonical document representation; reparsing
that changes ordering may change IDs. The `chunk_document()` entry point builds
provenance directly from the canonical blocks it processes. See
[`blp1_chunking.md`](blp1_chunking.md) for the algorithm, report semantics, and
current scope.

Chunks retain raw canonical text for screening, hashing, and citations while
also exposing `retrieval_text`, a derived indexing view with reversible
`retrieval_text_edits`. Repeated exact page-boundary lines are removed only
when shared by at least three pages; text-only hyphenated line breaks are
joined. Use `raw_text_ranges_for_retrieval_range()` to map a retrieval match
back to raw chunk text.


## Risk graph domain

The ontology defines these node types: `ORGANIZATION`, `APPLICATION`, `POLICY`,
`OBLIGATION`, `CONTROL`, `FINDING`, `RISK`, `EVIDENCE`, `DOCUMENT`, and
`CHUNK`.
Allowed relationship triples are directed:

| Source | Relationship | Target |
| --- | --- | --- |
| `ORGANIZATION` | `OWNS` | `APPLICATION` |
| `APPLICATION` | `IN_SCOPE_OF` | `POLICY` |
| `POLICY` | `DEFINES` | `OBLIGATION` |
| `CONTROL` | `SATISFIES` | `OBLIGATION` |
| `FINDING` | `EVALUATES` | `CONTROL` |
| `FINDING` | `AFFECTS` | `APPLICATION` |
| `EVIDENCE` | `SUPPORTS` | `FINDING` |
| `EVIDENCE` | `EXTRACTED_FROM` | `CHUNK` |
| `RISK` | `INDICATED_BY` | `FINDING` |
| `DOCUMENT` | `HAS_CHUNK` | `CHUNK` |
| `CHUNK` | `NEXT_CHUNK` | `CHUNK` |

`validate_relation` accepts enum values case-insensitively and returns `False`
for unknown names or disallowed triples.

`validate_node` checks that a node has the required properties defined in
`NODE_PROPERTY_TYPES`. It validates property types, rejects blank strings,
negative chunk indexes, missing properties, and unknown properties. These
validation helpers do not write nodes or relationships to Neo4j.

The domain distinction between a finding and its evidence is intentional:
evidence can support an observation, but neither evidence nor a finding alone
proves that a control failed. Likewise, `SATISFIES` describes the intended
relationship between a control and obligation, not proof of control
effectiveness. See [domain_model.md](domain_model.md) for the full rationale.

## API, database, and telemetry

The FastAPI application in `src/api/main.py` exposes:

- `GET /health`: process liveness; does not check dependencies.
- `GET /health/ready`: verifies Neo4j connectivity and runs a lightweight
  readiness query; dependency failures are returned as service-unavailable
  responses.

During application startup, the API configures telemetry, instruments FastAPI,
and creates the Neo4j connection. On shutdown it removes instrumentation,
closes the database driver, flushes telemetry, and shuts down the tracer
provider.

The Neo4j connector reads `NEO4J_URI`, `NEO4J_USERNAME`, and
`NEO4J_PASSWORD`; `NEO4J_DATABASE` defaults to `neo4j`. The telemetry exporter
requires `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`. Missing required configuration
causes setup to fail. The connector exposes async session access and a
`run_query` helper; there is not yet a repository layer for the risk graph or
ingestion artifacts.

## Tests and running locally

The `tests/` directory includes tests for ontology triples, graph-node
validation, Neo4j connectivity, and telemetry setup. The Neo4j connectivity
test requires a reachable, configured Neo4j instance; telemetry checks require
the OTLP traces endpoint to be configured.

Install the project dependencies, configure the environment variables listed
above, then run the API with:

```console
uvicorn src.api.main:app
```

Run the unit tests with:

```console
python -m pytest
```
