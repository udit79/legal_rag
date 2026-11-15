# Phase 4 closure and Phase 5 completion evidence

Inspected baseline: `b15f97b19559c14823257598ecff82019a3ff518`.
Phase 5 implementation is committed separately; no production graph or
Supabase writes were made.

## Phase 4 status

**PARTIAL.** Phase 4 OCR/VLM code and its prior pilot are present. Remote
Supabase migration catalog, role permissions, and Supabase-native SQL scenarios
were not executed in this Phase 5 run, so those database checks remain
unverified. No migration was applied. No human annotation or review was used
as an acceptance gate. The frozen Phase 3 benchmark was not modified.

## Phase 5 status

**COMPLETE for implementation and isolated technical acceptance.** The
extractor, independent verifier, GraphWriteGate, transactional Neo4j writer,
cross-document identity/provenance, concurrent creation, assertion
classifications, replay, conflict handling, and rollback passed local,
isolated Neo4j, and real-model pilot checks. Semantic writes remain fail-closed
until the semantic uniqueness constraints exist in their target database.
They were installed only in the disposable `test-rag` database; production
schema deployment was not approved or attempted.

### Identity and database uniqueness

The isolated target reported Neo4j **2026.10.2 Enterprise**, Cypher 5/25. Its
pre-test graph had zero nodes, relationships, or constraints. The explicit
`src/graph/semantic_schema.py` bootstrap:

- Runs a duplicate/missing-ID preflight across semantic node labels and
  semantic relationship types before DDL.
- Applies the common `SEMANTIC_ENTITY.id` uniqueness constraint, covering all
  ontology semantic node types.
- Applies `KNOWLEDGE_ASSERTION.id` uniqueness.
- Applies relationship-property uniqueness on each semantic relation type and
  `ASSERTION_CLASSIFICATION.id` (11 constraints total).
- Verifies the resulting constraint shapes before returning success.

Enterprise relationship-property uniqueness was chosen because the deployed
edition supports it. The writer checks the required constraint set inside its
write transaction and refuses semantic writes if it is missing. It does not
silently bootstrap or mutate a production schema. The disposable test database
now contains the 11 constraints and, after fixture cleanup, zero nodes and zero
relationships.

### First-class assertions and conflict handling

Each admitted entity or relationship claim creates or merges a deterministic
`KNOWLEDGE_ASSERTION` node. Its identity includes subject, predicate, object,
jurisdiction, scope, and temporal validity. It retains original evidence
references, document SHA-256 values, ingestion run IDs, extractor provenance,
and independent verifier attestations. Replays deduplicate identical evidence
and attestations; competing values remain separate assertions.

`AssertionClassification` supports `SUPPORTS`, `CONTRADICTS`, `QUALIFIES`,
`SUPERSEDES`, and `UNRESOLVED`. A deterministic scan records only potential
conflicts as `UNRESOLVED`; candidates require matching jurisdiction and scope
and potentially overlapping validity periods. Missing time bounds are treated
conservatively as possibly overlapping. Classification proposals are stored
with source references and `NOT_ADJUDICATED` status. No automated component
overwrites a claim or decides legal truth. Explicit classifications are
accepted only when both assertion endpoints exist and their document/run/
evidence references are already present on those assertions.

### Execution evidence

- Real Neo4j integration command, with `NEO4J_TEST_DATABASE=test-rag`:
  `python -m pytest tests/graph/test_knowledge_writer_integration.py -v`
  — **1 passed**. The test applied the semantic schema after duplicate
  preflight, verified duplicate preflight rejection, issued six concurrent
  writes across two documents sharing entity/relationship identities, checked
  stable node/relationship/assertion counts,
  exact replay, cross-document enrichment, all five classification values,
  unresolved conflict provenance, evidence/attestation integrity, a mid-write
  identity conflict with transaction rollback, and fixture cleanup.
- Real-model pilot against `test-rag`: extractor `gpt-6-luna`, verifier
  `gpt-6.1-sol`; 2 byte-distinct synthetic PDFs and 2 admitted chunks. The
  extractor proposal was replayed against identical source text, and the
  independent verifier ran against each source's own document hash/run. First
  write, exact replay, and second-document write each returned 2 semantic
  nodes / 1 relationship. The pilot verified shared identity, accumulated
  evidence and verifier attestations on semantic records and first-class
  assertions, 11 uniqueness constraints, random-ID preflight, and cleanup.
- Focused assertion/schema/gate/extraction unit suite: **41 passed**; the
  opt-in Neo4j integration case is run separately above.
- Complete local regression suite: **608 passed, 8 skipped in 38.10s**.
- Python compilation and `git diff --check`: passed.

Full-suite skips (`pytest -rs`):

- 1 graph schema bootstrap integration (`NEO4J_INTEGRATION=1` not set).
- 2 hybrid retrieval integrations (isolated read-test database not configured
  in the full-suite run).
- 1 knowledge writer integration (database intentionally unset in the full
  suite; it passed separately against `test-rag`).
- 2 live retrieval evaluations (`GRAPH_RAG_LIVE_RETRIEVAL=1` not set).
- 1 structural writer integration (`NEO4J_INTEGRATION=1` not set).
- 1 OCR host test (Tesseract is not installed on the host).

### Production rollout boundary

No production uniqueness constraints were installed. Before production
semantic writes, run the reviewed duplicate preflight against that target,
approve the schema deployment, and execute `initialize_semantic_schema()` for
the production database. Until then, `persist_approved_batch()` fails closed.
The verification does not claim formal semantic proof from the independent
model; it remains an automated verifier with possible model error.

## Changed files

- `src/graph/assertion_contracts.py` (new assertion/context/classification
  contracts)
- `src/graph/semantic_schema.py` (new preflight and explicit constraints)
- `src/graph/graph_write_gate.py`
- `src/graph/knowledge_writer.py`
- `tests/graph/test_assertion_contracts.py` (new)
- `tests/graph/test_semantic_schema.py` (new)
- `tests/graph/test_graph_write_gate.py`
- `tests/graph/test_knowledge_extraction.py`
- `tests/graph/test_knowledge_writer_integration.py`
- `scripts/phase5_e2e_pilot.py`
- `docs/phase4_closure_phase5_implementation.md`

No ontology, ingestion, CPCR, retrieval ranking, frozen Phase 3 artifact,
Supabase data, or production Neo4j data was changed. No production migration
was applied.
