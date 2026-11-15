# Phase 4 closure and Phase 5 implementation status

Inspected repository commit: `3887bb22d6ae115b8b62d1387a2d1fea3f5d11e1`.
No commit was created.

## Phase 4 — automated technical acceptance

**Verdict: PARTIAL.** No human annotation or human review was used as an acceptance
gate. Existing OCR, VLM, chunking, retrieval, Supabase persistence and Neo4j
structural components were inspected. The Phase 4 Milestone 2 source and tests
show separate native/OCR/VLM blocks, strict structured descriptions, screening
before canonicalization, run/page/block/image hash/model/prompt provenance,
bounded calls, idempotent replay handling and private asset references. The
existing milestone report records a successful disposable-project VLM pilot,
Neo4j idempotency rerun and read-only retrieval of the matching private asset.
Those reported live results were not repeated during this task.

The complete local suite was rerun after implementation: **582 passed, 7
skipped in 46.45s**. It exercises Phase 4 contracts
and regressions with mocked integrations. It does not prove the current remote
database state.

Skip breakdown from the pytest `-rs` rerun:

- 1 host OCR ordering test in `tests/ingestion/test_review_fixes.py` skipped
  because host `tesseract` is not installed. That test calls the local parser
  directly; it does not use Docker. The parser worker image installs
  `tesseract-ocr` in `docker/parser-worker.Dockerfile`.
- 1 Neo4j schema integration test and 1 Neo4j structural-writer integration
  test skipped because `NEO4J_INTEGRATION=1` was not enabled.
- 2 Neo4j retrieval integration tests skipped because isolated test database
  settings/credentials were not configured.
- 2 live retrieval tests skipped because `GRAPH_RAG_LIVE_RETRIEVAL=1` was not
  enabled.

The OCR skip is a host-test environment limitation, not evidence that the
Docker worker lacks OCR support.

### Supabase verification boundary

The previous pytest integration launchers required local `psql` and a
`PHASE4_*_SQL_TEST_DSN`. Those launchers have been removed. The paired SQL
scripts remain transaction-wrapped so they can be run in the disposable
Supabase SQL Editor, where fixture writes roll back.

This session had no connected Supabase SQL Editor/API tool, and browser
automation could not reach the editor. No migration was applied. Therefore
these database assertions remain **UNVERIFIED in this run**:

- Remote migration-history registration and deployed schema/permissions.
- SQL role checks for the `service_role` wrapper owner and denied helper calls.
- SQL Editor transaction rollback, mismatch/hash rejection, write-once and replay
  fixture assertions.

Earlier application-level PostgREST pilot evidence is described in the
Milestone 2 implementation report, but is not a substitute for these missing
catalog/transaction checks. No disposable fixtures were created in this run.

## Phase 5 — implemented slice

### Existing architecture reused

- `src/domain/ontology.py`: node/relation enums and directed endpoint allowlist.
- `src/domain/graph_schema.py`: strict ontology property validation.
- `src/graph/graph_write_gate.py`: exact admitted chunk quote/offset validation,
  page-local provenance, allowed endpoints, confidence floor and batch validation.
- `src/ingestion/chunk_contracts.py`: immutable chunks and source provenance.
- `src/infrastructure/neo4j_client.py`: configured asynchronous Neo4j sessions.

### Data path

An injected structured extractor receives bounded raw chunk text and a strict
JSON Schema response contract. The proposal uses only ontology node and
relationship enums, local references, confidence, epistemic mode and exact
chunk evidence spans. Model-provided graph IDs are rejected; IDs are derived
deterministically from normalized entity identity. The proposal is converted to
a `GraphWriteBatch` and passed through GraphWriteGate. Only then may
`persist_approved_batch()` execute a single Neo4j write transaction.

The writer uses fixed labels/types from enums, parameterized row values, MERGE,
and compares stored immutable properties/endpoints before accepting an
identity. Persisted evidence references include document identity, ingestion
run, chunk, exact quote offsets, page and source block IDs. It does not provide
the model with Neo4j credentials or arbitrary Cypher.

### Changed files

- `src/graph/graph_write_gate.py`: epistemic mode, confidence and stable
  relationship identity; batch run provenance.
- `src/graph/knowledge_extraction.py`: bounded proposal schema, explicit
  OpenAI adapter, strict JSON Schema output, retry/timeout/token controls,
  deterministic IDs, fail-closed proposal-to-batch validation.
- `src/graph/knowledge_writer.py`: GraphWriteGate-only transactional writer,
  deterministic MERGE identities, source evidence persistence and conflict
  checks.
- `tests/graph/test_knowledge_extraction.py`: mocked extraction/writer tests.
- Removed the two local-psql pytest launchers under `tests/sql/`; their SQL
  scripts are retained for Supabase SQL Editor use.

No migration, ontology enum, structural schema, retrieval implementation,
ingestion pipeline or Phase 3 benchmark artifact was changed.

### Phase 5 test status and limits

Focused tests: **15 passed**. Full suite: **582 passed, 7 skipped in 46.45s**.
The skips are opt-in live integrations; no live Neo4j write tests were enabled.

No live Neo4j writes or real OpenAI extraction calls were made. Phase 5 is an
initial implementation, not production acceptance: no end-to-end real model
proposal has been exercised, concurrent semantic-node creation is not protected
by new uniqueness constraints (the graph schema was intentionally preserved),
contradiction detection across differently worded relations remains limited,
and no Neo4j transaction integration test was run. Unsupported proposals with
invalid evidence, ontology types, endpoints, missing entities or low confidence
are rejected locally before the writer.

## Next actions

1. Run the retained OCR and VLM SQL integration scripts in the disposable
   Supabase SQL Editor; verify migration history first and record each assertion.
2. Add isolated Neo4j transaction integration tests for semantic-node and
   relationship conflict/idempotency behavior.
3. Configure the explicit OpenAI extraction model in a test environment and
   run one disposable end-to-end extraction through GraphWriteGate; inspect
   persisted provenance and replay behavior before any production use.
