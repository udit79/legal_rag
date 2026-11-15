# Phase 4 closure and Phase 5 implementation status

Repository HEAD inspected: `4cad0201be726cc650cca5edf0dda0f2679ac084`.
The Phase 5 implementation changes in the working tree are not committed.

## Phase 4 — automated technical acceptance

**Verdict: PARTIAL.** No human annotation or review was made an acceptance gate.
The Phase 4 OCR/VLM, canonical-block, provenance, chunking, storage, retrieval,
and graph-import code was inspected. The existing Milestone 2 report records a
disposable-project VLM pilot and retrieval of its private image reference; that
pilot was not repeated during this verification. This session did not execute
against Supabase or Neo4j.

The latest complete local regression run after the Phase 5 changes was
**590 passed, 8 skipped in 37.51s**. Local unit tests use mocked external
services and do not establish remote database state.

Skip details from pytest `-rs`:

- 1 OCR host test skipped because Tesseract is unavailable on this host. It
  calls the parser directly; the parser worker Docker image separately installs
  `tesseract-ocr`.
- 1 Neo4j schema bootstrap test and 1 structural-writer integration test
  skipped because Neo4j integration was not enabled.
- 2 retrieval integration tests skipped because isolated Neo4j test settings
  and credentials were not configured.
- 2 live retrieval tests skipped because `GRAPH_RAG_LIVE_RETRIEVAL=1` was not
  enabled.
- 1 semantic knowledge-writer integration test skipped because
  `NEO4J_TEST_DATABASE` and isolated Neo4j credentials were not configured.

### Supabase verification boundary

The psql-only integration harnesses have been removed. The transaction-wrapped
SQL scenarios remain in `tests/sql/` for execution in the disposable Supabase
SQL Editor. They cover wrapper/helper permissions, service-role execution,
run mismatch, invalid image digest, atomic rollback, replay, and write-once
behavior. There is no connected Supabase SQL Editor or database tool available
in this session; these scenarios and remote migration state are **UNVERIFIED**.
No migration was applied and no fixture writes were performed.

The prior Milestone 2 report describes an application-level PostgREST pilot.
That evidence does not replace database catalog, role, or transactional checks.

## Phase 5 — implemented local slice

### Reused architecture

- `src/domain/ontology.py` and `src/domain/graph_schema.py` for node types,
  relation types, directed endpoint rules, and property validation.
- `src/ingestion/chunk_contracts.py` and screening contracts for immutable
  chunk source and admission state.
- `src/infrastructure/neo4j_client.py` for asynchronous Neo4j transactions.
- `GraphWriteGate` as the required validation boundary before persistence.

### Extraction and support verification

`extract_validated_batch()` filters its input to ADMITTED and ADMITTED_FLAGGED
chunks before the extractor call. Rejected/quarantined chunks never enter the
LLM payload. `IMAGE_VLM` chunks are excluded from both extraction and source
fact verification. Their descriptions remain clearly AI-generated and are
currently denied as source facts by the explicit
`DENY_IMAGE_VLM_SOURCE_FACTS` policy. OCR evidence must retain the image SHA-256.

The extraction proposal schema does not accept model-supplied confidence or
epistemic labels. Exact source offsets and quote matches are checked first;
then a separately configured OpenAI verifier model evaluates each entity and
relationship claim against the quote and full chunk context. The verifier model
must differ from the extraction model. A claim is rejected unless the verifier
returns direct support with confidence at least 0.90; the gate binds the
attestation to claim, source context, and quote digests. This is an independent
automated check, not a formal proof of semantic truth. No live model call was
made during this run.

Trusted extraction instructions are sent as a system message. Source chunks
are passed separately as untrusted user payload. Extraction input, verifier
context, output tokens, retries, and timeouts are bounded.

### Graph persistence

Semantic IDs are deterministic from normalized ontology identity fields and do
not include document/run provenance. The Neo4j writer MERGEs entities by those
IDs, validates identity fields, and accumulates unique document hashes, run
IDs, evidence references, and assertion records so cross-document evidence is
retained without changing identity. Node and relationship confidence and
epistemic mode are persisted. Relationship IDs and endpoints are checked in
the write transaction. Writes are one `execute_write` transaction and must
pass `validate_graph_write_batch()` first.

Nested evidence and assertion structures are serialized as JSON strings
because Neo4j properties do not store nested maps. No graph schema or ontology
was changed. Concurrency safety for semantic-node creation is limited by the
absence of new uniqueness constraints; no production write was attempted.

### Changed implementation and tests

- `src/graph/graph_write_gate.py`
- `src/graph/knowledge_extraction.py`
- `src/graph/knowledge_writer.py`
- `src/ingestion/chunk_contracts.py`
- `src/ingestion/chunking.py`
- `src/ingestion/parser_contracts.py`
- `src/ingestion/screening/image_screening.py`
- `src/ingestion/screening/image_vlm.py`
- `tests/graph/test_graph_write_gate.py`
- `tests/graph/test_knowledge_extraction.py`
- `tests/graph/test_knowledge_writer_integration.py` (opt-in isolated DB)
- Phase 4 image asset/screening/VLM regression tests
- `tests/sql/test_phase4_image_vlm_persistence.sql` (Supabase SQL Editor)

No migration, ingestion pipeline behavior, ontology, GraphWriteGate policy
for structural edges, retrieval ranking, or frozen Phase 3 benchmark artifact
was changed. The OCR/VLM source contract additions carry the existing source
image hash into `ChunkSource`.

### Test status and limits

Focused graph and Phase 4 image tests: **192 passed, 7 skipped in 4.29s**.
Six skips are read-only Neo4j integration tests that require explicit isolated
database configuration; one is the semantic-writer integration test. Complete local
suite: **590 passed, 8 skipped in 37.51s**. Python compilation and
`git diff --check` passed.

No live Neo4j transaction test, end-to-end extraction pilot, live OpenAI call,
or Supabase SQL Editor scenario ran. Do not write production graph data yet.
The proposed independent verifier is model-based and may still err; conflicting
or differently worded assertions are retained as provenance, but semantic
contradiction resolution is not implemented. Graph writes are gated in code,
but real model and database behavior still needs disposable-environment
verification.

## Remaining validation steps

1. Verify deployed migration state, then execute
   `tests/sql/test_phase4_image_vlm_persistence.sql` in the disposable Supabase
   SQL Editor and capture the role, rollback, replay, and write-once results.
2. Configure `NEO4J_TEST_DATABASE` to an isolated database whose name contains
   `test`, then run `tests/graph/test_knowledge_writer_integration.py`.
3. Configure separate extraction and verifier models and run one disposable
   end-to-end extraction, checking gate rejection, persisted provenance,
   cross-document enrichment, rollback, and replay before any production use.
