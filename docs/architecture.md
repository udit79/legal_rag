# Current Architecture

## Purpose and current scope

This repository is a foundation for a document-backed graph RAG system focused
on enterprise risk concepts. It currently defines a health/readiness API,
Neo4j connectivity, OpenTelemetry tracing, graph-domain validation, and
contracts and PDF preflight for the beginning of a document-ingestion
workflow.

The ingestion contracts and state describe how stages can exchange small,
typed values. PDF preflight and deterministic per-page route selection are
implemented, as are replaceable PyMuPDF, Docling, and OCR page parsers with a
shared canonical output contract. Artifact storage, graph writes, and an
end-to-end workflow runner are not yet implemented.
See [code_map.md](code_map.md) for the file tree and a function/class-to-file
index.

## Component map

| Area | Location | Responsibility |
| --- | --- | --- |
| API | `src/api/main.py` | FastAPI application, lifecycle setup, liveness and readiness endpoints. |
| Ingestion contracts | `src/ingestion/contracts.py` | Pydantic models for document references, per-page/document features, and page/document route plans. |
| PDF preflight | `src/ingestion/preflight.py` | SHA-256, file size, page count, and per-page text, image, and drawing counts. |
| Ingestion routing | `src/ingestion/router.py` | Deterministic per-page route selection from validated document features. |
| Parser output contract | `src/ingestion/parser_contracts.py` | Canonical document/page/block models, source coordinates, statuses, and explicit failures. |
| Parser adapters | `src/ingestion/parsers.py` | PyMuPDF native text, Docling structured/table parsing, and Tesseract or RapidOCR. |
| Parser dispatch | `src/ingestion/dispatcher.py` | Selects an adapter per page and preserves page-level failures without dropping remaining pages. |
| Ingestion state | `src/ingestion/state.py` | Typed workflow state and structured stage-error shape. |
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
  `SKIP_BLANK`, `ESCALATE`, or `REJECT`.
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

## Ingestion workflow state

`IngestionState` in `src/ingestion/state.py` is a `TypedDict`, intended to
carry only the small values needed between workflow stages:

| Key | Required? | Value |
| --- | --- | --- |
| `document_ref` | Yes | `DocumentRef` for the source document. |
| `current_stage` | Yes | `PREFLIGHT`, `ROUTING`, `PROCESSING`, `COMPLETE`, or `FAILED`. |
| `features` | No | `DocumentFeatures`, once preflight has produced them. |
| `route_decision` | No | `RouteDecision`, once routing has run. |
| `parsed_artifact_ref` | No | Locator for parsed output stored separately. |
| `error` | No | `WorkflowError` when a stage fails. |

`WorkflowError` requires a structured `code` and the failing `stage`
(`PREFLIGHT`, `ROUTING`, or `PROCESSING`); a human-readable `message` is
optional. The state intentionally has no fields for PDF bytes, open file
handles, database sessions, or exception instances. Large parse results should
be kept in artifact storage and referenced through `parsed_artifact_ref`.

These types define a possible data flow—document reference, preflight,
routing, parsing, and completion/failure. Preflight, route selection, and
per-page parser dispatch are available as standalone functions, but a workflow
runner and stage transition rules are not implemented yet.

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
