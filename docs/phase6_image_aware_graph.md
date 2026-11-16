# Phase 6 — Image-aware knowledge graph (implementation status)

## Data flow

The existing Phase 4 path still extracts private image crops, screens OCR and
VLM descriptions, and stores admitted descriptions against the existing
`(ingestion_run_id, page_number, block_id)` image identity. The existing chunker
keeps `IMAGE_VLM` text searchable through the normal retrieval text embedding
space. No image embeddings or CLIP model are used.

`scripts/index_visual_knowledge.py` reads the latest CPCR corpus snapshot and
private image metadata, selects admitted `IMAGE_VLM` chunks, asks the separately
configured graph extraction model for schema-constrained candidates, validates
each quoted span against the stored description, and can persist the resulting
unverified observations in Neo4j. It is dry-run by default. Persistence requires
`--apply` and an explicitly configured `NEO4J_TEST_DATABASE` whose name contains
`test` and is not `rag` or `neo4j`.

Visual candidates are stored separately as `VISUAL_OBSERVATION` and
`VISUAL_ENTITY_CANDIDATE` nodes. Their `UNVERIFIED_AI_VISUAL_OBSERVATION`
trust status, source document/run/page/chunk, original image block and SHA-256,
coordinates, VLM and extraction model provenance, description/prompt digests,
and exact quoted offsets are retained. Candidate relationships are descriptive
and explicitly unverified. Exact name/type matches may link to existing
`SEMANTIC_ENTITY` nodes, but do not modify or assert facts about those nodes.
`GraphWriteGate` is unchanged and remains the path for verified textual claims.

Schema bootstrap adds uniqueness constraints for visual node and relationship
IDs and a `visual_knowledge_fulltext` index. It adds no vector index, asset
table, storage bucket, Supabase migration or public image URL. Initialize this
schema only on the intended Neo4j database after reviewing its existing schema.

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

Preview without writes:

```powershell
uv run python scripts/index_visual_knowledge.py <source-sha256>
```

For a disposable Neo4j database only, set `NEO4J_TEST_DATABASE` to its explicit
isolated name, then add `--apply`. Do not point this option at a production
database. `--apply` runs existing schema bootstrap and stores visual candidate
records; it does not re-ingest the PDF or alter Supabase data.

## Verification status

Unit and mocked API tests exercise source admission, exact description offsets,
asset-key provenance, deterministic identities, replay/conflict behavior,
visual retrieval, rank fusion, private hash-verified delivery, and the mocked
chunk-to-observation persistence path. The actual disposable Neo4j/Supabase and
OpenAI end-to-end pilot was not executable in the current environment because
the required connection/model environment variables were not present. No live
database write, real API request, or production operation was performed.

Therefore this implementation is **PARTIAL**, not accepted end-to-end. The
remaining acceptance step is to run the dry-run extraction against a known
image-bearing document, then `--apply` to the configured disposable Neo4j
database, verify candidate/relationship counts and current-run filtering, and
call `/graph/search` plus the private asset endpoint to confirm the returned
crop hash. Existing Phase 3 benchmark files are untouched.
