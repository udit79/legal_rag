# Phase 6 — Image-aware knowledge graph (implementation status)

## Data flow

The existing Phase 4 path still extracts private image crops, screens OCR and
VLM descriptions, and stores admitted descriptions against the existing
`(ingestion_run_id, page_number, block_id)` image identity. The existing chunker
keeps `IMAGE_VLM` text searchable through the normal retrieval text embedding
space. No image embeddings or CLIP model are used.

The normal structural graph importer now accepts `--visual-knowledge`. After
CPCR output has been read and validated and structural chunks have been
successfully persisted (and optionally replay-verified), it runs the visual
candidate stage. Visual processing has a separate Neo4j transaction per
ingestion run. A failure leaves structural ingestion committed, reports
`FAILED` with the source hash/run ID in CLI output, structured logs and tracing,
and returns a non-zero status so the visual stage can be retried. Retrying the
same manifest is safe because visual IDs and writes are deterministic and
idempotent. The option requires an explicit `NEO4J_TEST_DATABASE` whose name
contains `test` and is not `rag` or `neo4j`; it never targets the default or
production database.

The retained `scripts/index_visual_knowledge.py` is the dry-run/backfill and
recovery tool. It reads the latest CPCR corpus snapshot and private image
metadata, selects admitted `IMAGE_VLM` chunks, asks the separately configured
graph extraction model for schema-constrained candidates, and can persist into
an explicitly selected isolated Neo4j test database.

Multi-source chunks are processed per `IMAGE_VLM` source. For each source, the
run/page/block key resolves its stored description and the canonical source
offsets are checked against its digest and the chunk text. The exact visual span
is extracted independently, while native text and other image sources remain
untouched. Missing, mismatched, repeated or ambiguous spans fail closed before
the graph extraction model sees them. Rejected/quarantined chunks are excluded.

Visual candidates are stored separately as `VISUAL_OBSERVATION` and
`VISUAL_ENTITY_CANDIDATE` nodes. Their `UNVERIFIED_AI_VISUAL_OBSERVATION`
trust status, source document/run/page/chunk, original image block and SHA-256,
coordinates, VLM and extraction model provenance, description/prompt digests,
and exact quoted offsets are retained. Candidate relationships are descriptive
and explicitly unverified. Exact name/type matches may link to existing
`SEMANTIC_ENTITY` nodes, but do not modify or assert facts about those nodes.
`GraphWriteGate` is unchanged and remains the path for verified textual claims.

Schema bootstrap adds uniqueness constraints for visual node and relationship
IDs, a `visual_knowledge_fulltext` node index, a
`visual_relationship_fulltext` relationship index, and a
`visual_semantic_entity_fulltext` index for resolving linked semantic entities.
Retrieval checks that all three visual indexes are ONLINE and uses bounded
candidate counts for each SEARCH branch. It adds no vector index, asset
table, storage bucket, Supabase migration or public image URL. Initialize this
schema only on a disposable isolated Neo4j test database in this task.

`HybridRetriever.search_multimodal()` rank-fuses existing vector and full-text
results with visual-graph results. Image-derived hits return an image result
whose URL targets the API's private asset endpoint. That endpoint resolves the
existing Supabase composite key, checks the requested hash against metadata,
downloads from the private bucket, and verifies byte length and SHA-256 before
serving. The description remains internal supporting metadata. `/graph/search`
exposes multimodal results; `/visual-assets/{run}/{page}/{block}` serves the
verified crop. Existing `GET /graph` remains paginated graph export.

## Configuration and use

Set `GRAPH_VISUAL_EXTRACTION_PROVIDER=openai` and an explicit
`GRAPH_VISUAL_EXTRACTION_MODEL`, plus the existing `OPENAI_API_KEY`, Supabase
read credentials and Neo4j settings. Search also requires the existing
embedding configuration and online full-text/vector indexes.

Preview extraction without graph writes:

```powershell
uv run python scripts/index_visual_knowledge.py <source-sha256>
```

For a disposable Neo4j database only, set `NEO4J_TEST_DATABASE` to its explicit
isolated name. To run the complete graph-ingestion stage after the existing
Supabase ingestion/CPCR step, use:

```powershell
uv run python -m src.graph.corpus_cli --manifest <manifest.json> --apply --verify-idempotency --visual-knowledge
```

This command first writes the structural graph and verifies replay, then
processes visual candidates. A visual failure is reported against each
`ingestion_run_id` and does not undo the structural write. Re-run the same
command to retry. The separate backfill script's `--apply` also requires the
isolated database. Do not target production. Neither command re-ingests PDFs or
changes Supabase records.

## Verification status

Unit and route-level tests exercise admitted-source filtering, per-source
offsets for native+visual and multi-image chunks, ambiguous-span rejection,
asset-key provenance, deterministic identities, retry status, replay/conflict
behavior, relationship/entity graph search, current-run filtering, API result
types, private hash-verified delivery, and the mocked chunk-to-observation
The final full regression run completed with **642 passed and 9 skipped**. The
focused Phase 6/graph-gate/retrieval run completed with **127 passed and 2
skipped**. Skips were the configured live Neo4j schema/retrieval/writer checks
and the OCR test requiring Tesseract. The actual disposable Neo4j/Supabase and OpenAI end-to-end
pilot was not executable in the current environment because the required
Supabase URL/key, isolated Neo4j database and credentials, and OpenAI extraction
configuration/key were not present. The dry-run command stopped before network
access with the explicit missing-Supabase-configuration error. No live database
write, real model request, or production operation was performed.

Therefore this implementation is **PARTIAL**, not accepted end-to-end. The
remaining acceptance step is to run the dry-run extraction against a known
image-bearing document, then `--apply` to the configured disposable Neo4j
database, verify candidate/relationship counts and current-run filtering, and
call `/graph/search` plus the private asset endpoint to confirm the returned
crop hash. Existing Phase 3 benchmark files are untouched.
