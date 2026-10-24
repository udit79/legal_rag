# Current Architecture

## Purpose and current scope

This repository is a foundation for a document-backed graph RAG system focused
on enterprise risk concepts. It currently defines a health/readiness API,
Neo4j connectivity, OpenTelemetry tracing, graph-domain validation, and
contracts for the beginning of a document-ingestion workflow.

The ingestion contracts and state describe how future stages can exchange
small, typed values. They do not yet implement PDF preflight, routing,
parsing/OCR, artifact storage, graph writes, or a workflow runner.

## Component map

| Area | Location | Responsibility |
| --- | --- | --- |
| API | `src/api/main.py` | FastAPI application, lifecycle setup, liveness and readiness endpoints. |
| Ingestion contracts | `src/ingestion/contracts.py` | Pydantic models for document references, preflight features, and route decisions. |
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
- `page_text_char_counts`: non-negative integer count for each page, in page order.
- `page_image_counts`: non-negative integer count for each page, in page order.

The lengths of both per-page lists must equal `page_count`. These lengths
represent the number of pages being described; the values inside the lists
represent that page's character or image count.

### `RouteDecision`

Records the selected processing route and its rationale:

- `route`: `FAST_TEXT`, `STRUCTURED_PARSE`, `OCR`, `LAYOUT_OCR`, `ESCALATE`, or `REJECT`.
- `reason`: non-empty explanatory text.
- `decision_source`: `RULES`, `CDM`, or `FALLBACK`.
- `confidence`: optional value between 0.0 and 1.0.
- `policy_version`: non-empty identifier for the policy version.

All three contract models reject unspecified extra fields.

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
routing, processing, and completion/failure—but no stage execution or
transition rules are implemented yet.

## Risk graph domain

The ontology defines these node types: `ORGANIZATION`, `APPLICATION`, `POLICY`,
`OBLIGATION`, `CONTROL`, `FINDING`, `RISK`, `EVIDENCE`, and `DOCUMENT`.
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
| `EVIDENCE` | `EXTRACTED_FROM` | `DOCUMENT` |
| `RISK` | `INDICATED_BY` | `FINDING` |

`validate_relation` accepts enum names case-insensitively, returns `True` for
an allowed triple and `False` for a known but disallowed triple, and raises
`KeyError` for unknown names.

`validate_node` checks that a node has the required properties defined in
`NODE_PROPERTIES`, and rejects null, blank, or non-string property values.
Additional properties are currently allowed. These validation helpers do not
write nodes or relationships to Neo4j.

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
