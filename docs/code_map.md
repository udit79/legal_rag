# Code and file map

This page maps the current source tree to its implemented responsibilities
and symbols. It describes the code as it exists; a declared contract or
placeholder does not imply that a workflow uses it yet.

## Source tree

```text
.
├── main.py                         # Simple greeting script (not the API)
├── src/
│   ├── api/
│   │   ├── __init__.py             # Empty package marker
│   │   └── main.py                 # FastAPI app, lifecycle, health and PDF ingestion endpoints
│   ├── domain/
│   │   ├── graph_schema.py         # Graph-node properties and validation
│   │   ├── ontology.py             # Domain enums and relationship validation
│   │   └── __init___.py            # Empty package marker (currently misspelled)
│   ├── infrastructure/
│   │   ├── __init__.py             # Empty package marker
│   │   └── neo4j_client.py         # Neo4j driver and query helper
│   ├── ingestion/
│   │   ├── __init__.py             # Empty package marker
│   │   ├── contracts.py            # Ingestion data models and validation
│   │   ├── dispatcher.py           # Per-page parser routing and failure records
│   │   ├── parser_contracts.py     # Canonical document/page/block models
│   │   ├── parsers.py              # PyMuPDF, Docling, and OCR adapters
│   │   ├── pipeline.py             # End-to-end orchestration, spans, and unrecognized-image crops
│   │   ├── preflight.py            # PDF inspection
│   │   ├── router.py               # Deterministic per-page route selection
│   │   └── state.py                # Workflow type declarations
│   └── observability/
│       ├── __init__.py             # Empty package marker
│       └── telemetry.py            # OpenTelemetry setup
├── docs/
│   ├── architecture.md             # Components, behavior, and current scope
│   ├── code_map.md                 # This file: source and symbol index
│   └── domain_model.md             # Risk graph domain definitions
└── tests/
    ├── fixtures/test_ingestion_parser_integration.py
    ├── ingestion/                  # Golden parser/chunker and cover-order regressions
    ├── test_ingestion_*.py         # Ingestion contracts, adapters, routing, chunking
    └── test_telemetry.py           # Telemetry smoke check
```

## Implemented symbols by file

### Application

| File | Implemented symbols |
| --- | --- |
| `main.py` | `main()` — prints a greeting; not the service entry point. |
| `src/__init__.py`, `src/api/__init__.py` | Empty package markers. |
| `src/api/main.py` | `app`; `POST /ingestion` accepts a multipart PDF and returns revised chunks and crops for unrecognized images, without separate CPCR or canonical output; `lifespan(app)` loads reusable Docling pipelines, telemetry, ScaleDown HTTP, and Neo4j clients and closes/releases them on shutdown; health endpoints. |

### Domain and graph

| File | Implemented symbols |
| --- | --- |
| `src/domain/ontology.py` | `NodeType`, `RelationType`, `EpistemicMode`, `EntityResolutionMethod`; `NODE_TYPE_DESCRIPTIONS`, `RELATION_DESCRIPTIONS`, `EPISTEMIC_MODE_DESCRIPTIONS`, `ALLOWED_TRIPLES`; `validate_relation(source_type, relation_type, target_type)`. |
| `src/domain/graph_schema.py` | `NODE_PROPERTY_TYPES`, `NODE_PROPERTIES`; `validate_node(node_type, properties)`. |
| `src/domain/__init___.py` | No functions; currently an empty package marker whose filename has three trailing underscores. |

### Ingestion

| File | Implemented symbols |
| --- | --- |
| `src/ingestion/contracts.py` | Type aliases `DocumentMediaType`, `Route`, `DecisionSource`, `DocumentDisposition`, `NonNegativeInt`, `PositiveInt`, `NonEmptyStr`; models and validation described below. |
| `src/ingestion/preflight.py` | `PdfPreflightError`, `PdfNotFoundError`, `InvalidPdfError`; `inspect_pdf(path)` computes document hash/size and per-page PDF features. |
| `src/ingestion/parser_contracts.py` | `SourceCoordinates`, `CanonicalBlock`, `ParserFailure`, `CanonicalPage`, `CanonicalDocument`; validates block coordinates, page statuses, and complete original page order. `canonical_block_id(page_number, reading_order)` and serialized `CanonicalBlock.block_id` provide deterministic IDs within one canonical document representation. |
| `src/ingestion/chunk_contracts.py` | `ChunkSource`, `Chunk`, `ChunkedDocument`, `TEXT_BEARING_BLOCK_TYPES`; immutable contracts for typed source provenance, Unicode text offsets, contiguous chunk indices, source page bounds, and policy version. No splitting algorithm is implemented. |
| `src/ingestion/parsers.py` | `PageParser`; `FastTextParser.parse_page()` uses PyMuPDF; `StructuredParser.parse_page()` uses Docling, with optional RapidOCR for layout OCR; `OcrParser.parse_page()` uses Tesseract by default or RapidOCR. |
| `src/ingestion/dispatcher.py` | `dispatch_document(source, route_plan)` routes parsing per page and emits `ingestion.parser.page` spans with page, route, parser, status, block and warning attributes. It retains page failures and continues in source order. |
| `src/ingestion/pipeline.py` | `ingest_pdf(path)` runs preflight, routing, parsing, chunking, and image crop extraction; `classify_page_continuity(ingestion, client)` evaluates candidates; `revise_chunks_after_cpcr(ingestion, cpcr)` merges positive continuity pairs while retaining provenance. `CPCRIngestionResponse` contains revised chunks and image crops only. |
| `src/ingestion/state.py` | Type declarations `WorkflowStage`, `ErrorStage`, `WorkflowError`, `IngestionState`; no workflow runner or transition functions. |
| `src/ingestion/router.py` | `MIN_NATIVE_TEXT_CHARACTERS` (50); `route_document(features, policy_version)` returns a validated plan with a deterministic decision for each page; `_route_page(page)` applies the per-page heuristics. |
| `src/ingestion/__init__.py` | Empty package marker. |

### Infrastructure and observability

| File | Implemented symbols |
| --- | --- |
| `src/infrastructure/neo4j_client.py` | `DbConnect`; methods `__init__()`, `verify_connection()`, `get_session()`, `close()`; `create_db_connect()` reads connection settings from environment variables; `run_query(query, db_connection)` executes a query in a session. |
| `src/infrastructure/__init__.py`, `src/observability/__init__.py` | Empty package markers. |
| `src/observability/telemetry.py` | `configure_telemetry()` creates and registers the tracer provider and OTLP exporter. |

## Ingestion contract details

The models below are declared in `src/ingestion/contracts.py`. They validate
and carry data. `route_document()` consumes `DocumentFeatures` and returns a
validated `DocumentRoutePlan`; document processing is not implemented yet.

| Model | Fields and behavior |
| --- | --- |
| `DocumentRef` | `document_id`, `source_ref`, and `filename` are non-empty strings; `media_type` is currently only `application/pdf`. |
| `PageFeatures` | `page_number` is a positive, one-based integer; `text_char_count`, `image_count`, and `drawing_count` are non-negative integers. |
| `DocumentFeatures` | `content_sha256` is a 64-digit hexadecimal SHA-256 normalized to lowercase; `file_size_bytes` and `page_count` are positive integers; `pages` is a tuple of `PageFeatures`. `validate_pages()` requires exactly `page_count` records numbered contiguously from 1. |
| `RouteDecision` | `route` is one of `FAST_TEXT`, `STRUCTURED_PARSE`, `OCR`, `LAYOUT_OCR`, `SKIP_BLANK`, `ESCALATE`, or `REJECT`; `reason` and `policy_version` are non-empty strings; `decision_source` is `RULES`, `CDM`, or `FALLBACK`; `confidence` is optional and, when provided, is a strict number in the range 0–1. |
| `PageRouteDecision` | Extends `RouteDecision` with a positive `page_number`. |
| `DocumentRoutePlan` | Contains positive `page_count`, a tuple of page `decisions`, and a required `document_disposition` (`PROCESS` or `REJECT_NO_CONTENT`). `validate_page_decisions()` requires one decision for every page, in ascending order, with a consistent `policy_version`. It rejects `PROCESS` if every page is `SKIP_BLANK`, and rejects `REJECT_NO_CONTENT` if any page is not `SKIP_BLANK`. |

All six models reject extra fields. `NonNegativeInt`, `PositiveInt`, and
`NonEmptyStr` are reusable constrained types; `DocumentMediaType`, `Route`,
`DecisionSource`, and `DocumentDisposition` restrict values to the literals
listed above.

`src/ingestion/state.py` is a separate workflow-state contract: `IngestionState`
requires `document_ref` and `current_stage`, and optionally carries
`features`, `route_decision`, `parsed_artifact_ref`, and `error`.
`WorkflowError` requires an error `code` and `stage` and may include a
`message`. These are type declarations, not state-transition functions.

The parser output contract is separate from route decisions. `CanonicalBlock`
requires an original one-based page number, text, block type, reading order,
and source coordinates tagged with coordinate units and origin.
`CanonicalPage` records parsed, blank, vector-only, or failed outcomes.
`CanonicalDocument` requires exactly one page result per input page in source
order. `dispatch_document()` returns failed page records rather than silently
discarding parser errors or later pages.

The chunking data contracts are in `src/ingestion/chunk_contracts.py`.
`ChunkSource` identifies a canonical block using the `document_id` on
`ChunkedDocument`, its one-based page number, and the canonical block's stable
`block_id` (also retaining reading order and block type). `BlockType` is a
`Literal` alias in `parser_contracts.py`, not an enum, so
`TEXT_BEARING_BLOCK_TYPES` uses those literal strings without changing parser
contracts. `TEXT`, `TABLE`, and
`OTHER` are explicitly treated as text-bearing and require half-open
`[start_offset, end_offset)` ranges and source text length, measured in Python
Unicode string characters (not UTF-8 bytes). `IMAGE` may omit ranges or
provide them when its canonical parser output contains text; Docling retains
an image item's text when present. `VECTOR` is emitted with empty text and
cannot carry offsets; OCR recognition is emitted as `TEXT`. A chunk may
reference multiple ordered sources, and one source block may contribute
distinct ranges to multiple chunks.
Overlapping or duplicate ranges for the same block within a single chunk are
rejected; adjacent ranges and ranges in separate chunks are allowed.
`ChunkedDocument` records the source page count and non-empty policy version,
and validates contiguous zero-based indices and in-range source pages.
`chunking.py` implements validated configuration and policy identity, page
routing, M5 TEXT/OTHER packing, M6 isolated TABLE/IMAGE handling, and final
reports. `segmentation.py` implements the six M3 boundary primitives and the
M4 recursive boundary ladder. `chunk_document()` builds each `ChunkSource`
from the canonical block and atom offsets, preserving exact source slices and
the configured separator and overlap policy. Algorithm version is `1.3.0`. See
[`blp1_chunking.md`](blp1_chunking.md) for the BLP-1 algorithm and its current
limits. `block_id` is deterministic from one-based page number and zero-based
reading order within a canonical document representation, but may change if
reparsing changes block ordering.

`FastTextParser` joins PyMuPDF line spans with newline characters to retain
native line boundaries. OCR adapters preserve native blocks and add
recognized lines, omitting OCR lines only when case-folded,
whitespace-normalized text matches a native line or earlier OCR line and their
source-coordinate rectangles overlap by at least 0.8 relative to the smaller
rectangle. Repeated text at separate locations and novel lines in a mixed OCR
result are retained; there is no fuzzy matching. Docling coordinate origin is
taken from each provenance bbox. Adapter tests check source page numbers and
PDF-point coordinates against the corresponding source page bounds.

## Tests by area

| Test file | Source area covered |
| --- | --- |
| `tests/test_ingestion_adapters.py` | Canonical parser output, all three adapters, route dispatch, encrypted/corrupt pages, and failure continuation. |
| `tests/test_graph_schema.py` | `src/domain/graph_schema.py` |
| `tests/test_ontology.py` | `src/domain/ontology.py` |
| `tests/test_ingestion_contracts.py` | `src/ingestion/contracts.py` |
| `tests/test_ingestion_preflight.py` | `src/ingestion/preflight.py` |
| `tests/test_ingestion_chunk_contracts.py` | Chunk/source provenance model validation. |
| `tests/test_ingestion_chunking_models.py` | Chunking configuration, digest, policy, and report models. |
| `tests/test_ingestion_chunking_routing.py` | Page status routing and text eligibility. |
| `tests/test_ingestion_segmentation.py` | M3 boundary primitives, M4 atom spans, and Unicode hard cuts. |
| `tests/test_ingestion_chunking_packing.py` | M5/M6 greedy packing, row splitting, isolation, provenance, and reporting. |
| `tests/test_neo4j_connection.py` | Neo4j connectivity through `src/infrastructure/neo4j_client.py` |
| `tests/test_telemetry.py` | Telemetry setup through `src/observability/telemetry.py` |
| `tests/__init__.py` | Empty test package marker. |

The test named `test_neo4j_connection.py` contains a standalone async
connectivity check and requires a configured Neo4j instance. The telemetry
smoke test requires `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`.
