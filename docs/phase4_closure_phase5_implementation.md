# Phase 4 closure and Phase 5 validation status

Repository inspected at baseline commit `7d6543f5e6cd6f49173504a42b5ee5e6f0edcf01`.
Changes described below are in the uncommitted working tree. No production
Neo4j or Supabase writes were made.

## Phase 4 — final verification

**Verdict: PARTIAL.** No human review or annotation was used as an acceptance
gate. The Phase 4 OCR/VLM implementation is present, but remote Supabase
catalog/role/migration checks and the Supabase-native SQL scenarios were not
executed in this run. Those required checks remain unverified; no migration
was applied. The previous Milestone 2 pilot report is implementation evidence,
not proof of the current remote database state. The frozen Phase 3 benchmark
was not changed.

## Phase 5 — audit and verified slice

### Reused code

- `src/domain/ontology.py` and `src/domain/graph_schema.py` define node
  properties, relation types, and directed endpoint constraints.
- `src/ingestion/chunk_contracts.py` and screening contracts provide immutable
  source chunks and admission decisions.
- `src/infrastructure/neo4j_client.py` provides async sessions and
  transactions.
- `src/graph/graph_write_gate.py` remains the required persistence boundary.

### Gate and extraction behavior

`extract_validated_batch()` sends only ADMITTED or ADMITTED_FLAGGED chunks to
the extractor, and excludes IMAGE_VLM chunks from source-fact extraction.
Trusted extraction instructions use a system message; source text is passed
as explicitly untrusted user data. Extractor output cannot set confidence or
epistemic mode. Exact source quotes and offsets are validated before a
separately configured verifier checks claims. The gate requires verifier
attestations bound to claim, context, and quote digests, rejects identical
extractor/verifier models, and enforces confidence of at least 0.90. Ontology
properties, relation types, endpoints, and evidence references are validated
before a batch is accepted. IMAGE_VLM descriptions remain excluded as
original source facts.

When model-supplied offsets are wrong, extraction recomputes the range only
if the exact quote has exactly one occurrence in the source chunk. Missing or
ambiguous matches are rejected. Persisted offsets therefore come from a
unique literal source match, not the model's offset estimate.

The verifier is a second model-based check, not a formal proof of entailment.
Automated verification can still make semantic errors; this limitation is
retained explicitly rather than claiming verified legal truth.

### Persistence, identity, and concurrency

`src/graph/knowledge_writer.py` uses one Neo4j write transaction, deterministic
semantic node IDs, and deterministic relationship IDs. It checks identity
properties and endpoints, accumulates unique document/run provenance and
assertion evidence, and stores confidence and epistemic metadata. Replays do
not add duplicate evidence. A forced identity conflict in the isolated
integration test rolls back attempted provenance updates.

The current schema has uniqueness constraints for structural DOCUMENT and
CHUNK IDs, but no semantic-node/relationship uniqueness constraints. MERGE
alone does not guarantee uniqueness during concurrent creation. The smallest
safe production hardening is to preflight duplicate IDs, then add constraints
for semantic node labels and relationship identity using the repository's
Neo4j version and supported key strategy. No schema change was made because a
production schema migration was outside the granted scope. Differently worded
contradictory assertions are retained with provenance, but no contradiction
adjudicator is implemented.

### Executed verification

- Focused graph suite including isolated writer integration:
  **28 passed in 2.74s**.
- Required standalone integration command, with
  `NEO4J_TEST_DATABASE=test-rag`:
  `python -m pytest tests/graph/test_knowledge_writer_integration.py -v`
  — **1 passed in 2.39s**. It verified transaction lifecycle, initial node
  and relationship creation, exact replay, cross-document evidence/run
  enrichment, identity conflict rollback, and cleanup of fixture nodes and
  relationships in `test-rag`.
- Real end-to-end pilot on `test-rag`: OpenAI extractor `gpt-6-luna`, separate
  verifier `gpt-6.1-sol`; 2 byte-distinct source PDFs with identical synthetic
  text and 2 admitted chunks. One real extractor response was replayed for
  identical source text; independent verification ran against each PDF's own
  chunk and hash. Each document had 2 entities and 1 relationship. Initial
  write, exact replay, and second-document write each returned 2 nodes / 1
  relationship. Both entities were shared across documents; the stable
  relationship retained evidence and verifier attestations from both runs.
  Source quotes, hashes, run IDs, endpoints, and attestations were checked in
  Neo4j. Random identity preflight and cleanup succeeded; no production data
  was touched.
- Full local suite:
  **594 passed, 8 skipped in 36.55s**.
- Python compilation and `git diff --check`: passed.

Full-suite skips (`pytest -rs`):

- 1 schema-bootstrap Neo4j integration: requires `NEO4J_INTEGRATION=1`.
- 2 hybrid retrieval live tests: isolated Neo4j settings unavailable in the
  full-suite environment.
- 1 semantic writer test: isolated database was intentionally unset for the
  full suite; it was run separately and passed against `test-rag`.
- 2 live retrieval tests: require `GRAPH_RAG_LIVE_RETRIEVAL=1`.
- 1 structural writer integration: requires `NEO4J_INTEGRATION=1`.
- 1 OCR parser test: Tesseract is not installed on the host.

### Phase 5 status

**PARTIAL — the isolated extraction/write path is verified, production-grade
concurrency and contradiction handling are not complete.** Before production
semantic writes, approve and apply an identity-constraint migration after
duplicate preflight, then add concurrent-creation integration coverage.
Consider an explicit contradiction classifier that preserves competing
assertions rather than merging their meaning. No production writes should be
enabled until the identity constraint gap is addressed.

## Changed files in this working tree

- `src/graph/graph_write_gate.py`
- `src/graph/knowledge_extraction.py`
- `tests/graph/test_graph_write_gate.py`
- `tests/graph/test_knowledge_extraction.py`
- `tests/graph/test_knowledge_writer_integration.py`
- `scripts/phase5_e2e_pilot.py` (new, disposable-database-only pilot)
- `docs/phase4_closure_phase5_implementation.md`

No migration, ontology, ingestion pipeline, retrieval ranking, frozen Phase 3
artifact, or production graph was changed.
