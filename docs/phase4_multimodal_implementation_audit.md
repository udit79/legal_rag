# Phase 4 Multimodal Implementation Audit

**Audit date:** 2026-11-13  
**Scope:** Existing ingestion, canonicalization, Supabase provenance/storage, chunking, graph loading, retrieval/evidence assembly, migrations, tests, and read-only live Supabase state.  
**Change scope:** This audit only. No database, bucket, corpus, graph, retrieval, embedding, ingestion, or Phase 3 benchmark data was modified.

## Executive summary

The project already extracts some visual regions, renders blank-text `IMAGE` blocks into PNG crops, performs local OCR for screening, and stores crops plus screening metadata in the existing private Supabase bucket and `image_screening_results` table. The current project has **32** image-screening rows across six latest ingestion runs. All 32 rows have storage references, and five sampled object paths returned HTTP 200.

The main Phase 4 gap is the boundary between that visual sidecar and searchable evidence. Crop OCR text is screened but only its status, digest, length, decision, and flags are persisted. It is not put back into canonical/chunk text. In the current corpus snapshot, none of the 32 `(run, page, block)` image records has a corresponding source block in an original chunk. There is no explicit asset ID, asset-to-chunk relation, canonical image URL/path, or image evidence in Neo4j retrieval bundles.

## Status summary

| Capability | Status | Finding |
|---|---|---|
| Image extraction from source documents | **PARTIALLY_IMPLEMENTED** | Docling emits image regions; blank-text regions are rendered as page crops. Original embedded image objects and all nonblank image regions are not extracted as asset records. |
| Image persistence in Supabase | **IMPLEMENTED** | PNG crops are uploaded to the existing private `ingestion-image-crops` bucket; metadata is written to `image_screening_results`. Live project verification found 32 rows and five sampled stored objects. |
| Asset metadata and storage references | **IMPLEMENTED** | Metadata includes crop SHA-256, bucket, object path, size, page/block, coordinates, OCR state and screening outcome. Content identity is the image digest; there is no separate asset ID. |
| Provenance linking all requested identifiers | **PARTIALLY_IMPLEMENTED** | Page, block and image digest are explicit. Document SHA is derivable through the ingestion-run FK. No independent asset ID or direct asset-to-document-hash FK exists. |
| OCR/text extraction from images | **PARTIALLY_IMPLEMENTED** | Local OCR is run on blank-text image crops and page OCR can emit canonical text. Crop OCR text is screened but not retained as source text or linked to chunks. |
| Image captions/descriptions/visual summaries | **NOT_IMPLEMENTED** | No image-captioning or visual-summary component or persisted descriptive field was found. |
| Image/table/figure references in canonical documents | **PARTIALLY_IMPLEMENTED** | Canonical blocks support `IMAGE` and `TABLE`, stable page-scoped block IDs, text, and coordinates. They do not carry storage/asset references or typed figure/caption relationships. |
| Relationships between visual assets and document chunks | **NOT_IMPLEMENTED** | No persistent asset-to-chunk relation exists. Current live metadata/source comparison found zero image block keys represented in chunk source references. |
| Visual content in retrieval/evidence assembly | **PARTIALLY_IMPLEMENTED** | Extracted text can be chunked and retrieved; image crops and crop-OCR text are not. Evidence contains raw chunk text and text source coordinates, not visual assets. |
| Failure handling, idempotency and versioning | **PARTIALLY_IMPLEMENTED** | Upload retry, deterministic object names, exact-replay upsert, and orphan reconciliation exist. Versioning is run-scoped and digest-based; there is no standalone asset-version registry or asset ID. |

## Findings by capability

### 1. Image extraction from source documents — PARTIALLY_IMPLEMENTED

`StructuredParser` uses Docling page items to identify `picture`/`image` labels and represents them as canonical `IMAGE` blocks with page, reading order, block ID, text (possibly empty), and source coordinates. `crop_unrecognized_images()` renders the corresponding PDF page rectangle to PNG **only when the canonical image block has blank text**. This is a rendered page crop, not extraction of the PDF's original embedded image bytes. Image blocks with parser-provided text are not cropped by this path.

Source files: `src/ingestion/parsers.py`, `src/ingestion/parser_contracts.py`, `src/ingestion/pipeline.py`, `src/ingestion/parser_worker/worker.py`, `src/ingestion/parser_worker/result.py`. Relevant tests: `tests/test_ingestion_adapters.py::test_structured_parser_preserves_text_on_image_blocks` and `tests/ingestion/test_parser_worker_image_ocr.py`.

### 2. Image persistence in Supabase — IMPLEMENTED

`SupabaseProvenanceStore._store_image_crops()` verifies Base64, PNG signature and SHA-256 before upload. The object path is `sha256(ingestion_run_id)/sha256(crop_bytes).png`. Uploads use `x-upsert: true`. The same payload's metadata is written through `persist_ingestion_screening`.

The live project returned the `ingestion-image-crops` bucket with `public=false`. `image_screening_results` returned **32 rows**; all 32 had a bucket, object path, and size. Five sampled object `HEAD` requests returned HTTP 200. These are read-only verification results, not a full object integrity scan.

Database/storage: `public.image_screening_results`, `public.ingestion_runs`, bucket `ingestion-image-crops`. Schema source: `supabase/migrations/202611050001_ingestion_provenance.sql` and `supabase/migrations/202611110001_ingestion_image_storage.sql`; the latter adds storage columns but assumes the private bucket was pre-created. Code: `src/ingestion/provenance/supabase_store.py`. Tests: `tests/ingestion/test_supabase_provenance.py::test_image_upload_retries_and_persists_private_object_metadata`.

### 3. Asset metadata and storage references — IMPLEMENTED

The existing `image_screening_results` row stores `image_sha256`, `image_bucket`, `image_object_path`, `image_file_size_bytes`, `page_number`, `block_id`, `reading_order`, `source_coordinates`, `ocr_status`, optional `ocr_text_sha256`, `ocr_text_length`, optional screening `decision`, and `flags`. A check constraint requires the three storage metadata fields to be all null or complete and restricted to `ingestion-image-crops`.

There is no separate `asset_id` field or asset registry. The image-byte SHA-256 is the content identity; the run-scoped object path is the storage reference. This is sufficient to reference stored crops, but callers must preserve the existing identifiers rather than inventing another ID.

Database/storage: `public.image_screening_results`, `public.ingestion_runs`, `ingestion-image-crops`. Code/tests: `src/ingestion/provenance/supabase_store.py`; `tests/ingestion/test_supabase_provenance.py` image upload, hash-mismatch, and RPC cleanup cases.

### 4. Provenance linking — PARTIALLY_IMPLEMENTED

The metadata row explicitly links `ingestion_run_id`, one-based `page_number`, page-scoped `block_id`, and `image_sha256`. Its FK to `ingestion_runs` makes `document_sha256` resolvable through `ingestion_runs.source_sha256`; the image row does not itself store `source_sha256`. The storage path includes a hash of the run ID and the crop digest. There is no independent asset ID.

Live inspection confirmed all 32 image rows map to six run IDs in the current corpus snapshot. It does not make the association a direct database FK from image metadata to canonical document blocks or chunks.

Database: `public.image_screening_results` -> `public.ingestion_runs` -> `public.ingestion_documents`. Migration: `supabase/migrations/202611050001_ingestion_provenance.sql`. Code: `src/ingestion/provenance/supabase_store.py`, `src/ingestion/screening/image_screening.py`.

### 5. OCR/text extraction from images — PARTIALLY_IMPLEMENTED

For blank-text image crops, the parser worker runs local Tesseract OCR (`_handle_image_ocr()` calls the worker's Tesseract adapter). `screen_image_crops()` distinguishes `TEXT_FOUND`, `NO_TEXT`, and `FAILED`; nonblank text goes through ingestion security screening. OCR failure is represented as `FAILED`, quarantined, and flagged. Empty OCR output is recorded as `NO_TEXT`.

The raw crop OCR string is transient in this screening path. Supabase receives only its hash and character length, not the OCR text. The result is not appended to `CanonicalDocument`, chunk text, or `retrieval_text`. Separately, the general page OCR parser emits OCR-recognized text as canonical `TEXT` blocks; that text can be chunked and retrieved. These are two distinct paths.

Source files: `src/ingestion/parser_worker/runner.py`, `src/ingestion/parser_worker/worker.py`, `src/ingestion/parsers.py`, `src/ingestion/screening/image_screening.py`, `src/api/main.py`. Tests: `tests/ingestion/test_parser_worker_image_ocr.py`, `tests/ingestion/test_image_screening.py`, and OCR adapter cases in `tests/test_ingestion_adapters.py`.

### 6. Captions, descriptions, and visual summaries — NOT_IMPLEMENTED

No visual-language model, caption-generation service, descriptive asset field, or image-summary persistence path was found. Docling can return image-item text, and ordinary PDF caption text can appear as its own canonical text block, but the code does not create an explicit caption-to-image relation or generated description.

Relevant inspection: `src/ingestion/parsers.py`, `src/ingestion/parser_contracts.py`, `supabase/migrations/202611050001_ingestion_provenance.sql`, `supabase/migrations/202611110001_ingestion_image_storage.sql`. No caption/summary-specific tests were found.

### 7. Canonical image/table/figure references — PARTIALLY_IMPLEMENTED

`CanonicalBlock.block_type` supports `IMAGE` and `TABLE`; blocks have page number, stable `page-{page}-block-{reading_order}` ID, reading order, text, and source coordinates. Docling tables are serialized to text/Markdown and become `TABLE` blocks. Image item text is preserved when supplied. Canonical output is used to derive a digest for `ingestion_documents`; the full canonical object is not persisted as a table record. Final chunk sources retain page/block references where they contribute text.

There are no explicit figure numbers, `asset_id`, bucket/path fields, caption relationship, or visual asset reference in a canonical block. `VECTOR` blocks are also represented, but are not raster image assets.

Source files: `src/ingestion/parser_contracts.py`, `src/ingestion/parsers.py`, `src/ingestion/chunk_contracts.py`, `src/ingestion/chunking.py`, `src/api/main.py`. Tests: `tests/test_ingestion_adapters.py::test_structured_parser_preserves_text_on_image_blocks`, table serialization/parser cases in the same file, and `tests/test_ingestion_chunk_contracts.py` IMAGE/TABLE provenance validation cases.

### 8. Visual asset to chunk relationships — NOT_IMPLEMENTED

The image metadata primary key is `(ingestion_run_id, page_number, block_id)`. Chunk sources store `(page_number, block_id)` plus optional text offsets; `ingestion_chunks` stores these sources as JSON. No FK, relationship table, CPCR field, Neo4j node/relationship, or retrieval field links a crop's `image_sha256`/object path to a chunk identity.

Read-only live comparison against the six current corpus runs found **0 of 32** image `(page_number, block_id)` pairs in the current original chunk source metadata. This is consistent with the crop policy: the persisted crops come from `IMAGE` blocks whose canonical text is blank, so they do not ordinarily contribute chunk text.

Database tables: `public.image_screening_results`, `public.ingestion_chunks`, `public.ingestion_cpcr_outputs`. Code: `src/ingestion/pipeline.py`, `src/ingestion/provenance/supabase_store.py`, `src/graph/corpus_reader.py`, `src/graph/structural_writer.py`. Tests verify chunks preserve block provenance and image metadata persistence independently; no asset-to-chunk relationship test exists.

### 9. Visual content in retrieval and evidence — PARTIALLY_IMPLEMENTED

Retrieval indexes current `CHUNK.retrieval_text` and evidence bundles return `chunk_id`, document hash, raw chunk text, source page/block provenance, run ID, channels, and scores. There is no asset identifier, image bytes/path, signed URL, OCR text from crop screening, or visual evidence type in `EvidenceBundle` or the Neo4j structural graph.

Text already emitted by the page OCR parser or by a nonblank Docling image block can reach chunks and text retrieval. Blank-text image crop OCR cannot. Thus some image-derived text can be retrievable when captured by the page parser, but visual crops and their sidecar OCR are not part of retrieval or evidence assembly.

Source files: `src/graph/retrieval.py`, `src/graph/structural_writer.py`, `src/graph/corpus_reader.py`, `src/ingestion/chunk_contracts.py`, `src/ingestion/parsers.py`. Tests: `tests/graph/test_hybrid_retrieval.py` and `tests/graph/test_retrieval_smoke.py` validate text evidence/source provenance; no visual-evidence retrieval test exists.

### 10. Failure handling, idempotency, and asset versioning — PARTIALLY_IMPLEMENTED

- **Failure handling — IMPLEMENTED:** image/crop/result alignment is checked; PNG and digest are verified before upload; transient upload failures retry three times; failed upload prevents the screening RPC; failed/ambiguous RPC outcomes trigger safe reconciliation or retain objects if references cannot be verified. The endpoint returns a persistence failure instead of silently returning success. OCR failure fails closed with quarantine metadata.
- **Idempotency — IMPLEMENTED for exact replay:** key is deterministic for run+crop bytes; upload uses upsert; duplicate digests within a run are uploaded once; metadata key is `(run, page, block)`. Exact unchanged replay is allowed by the immutability trigger; changed metadata for an existing key is rejected rather than replacing it.
- **Versioning — PARTIALLY_IMPLEMENTED:** a different ingestion run gets a separate hashed run prefix and therefore preserves old objects. The digest identifies bytes, but no explicit asset version sequence, asset registry, or stable cross-run asset ID exists. Identical crops across different runs are stored under different run prefixes.

Source files: `src/ingestion/provenance/supabase_store.py`, `src/ingestion/screening/image_screening.py`, `src/api/main.py`; migrations: `202611050001_ingestion_provenance.sql`, `202611080001_ingestion_provenance_immutability.sql`, `202611110001_ingestion_image_storage.sql`. Tests: `tests/ingestion/test_supabase_provenance.py` upload retry/idempotency, hash mismatch, partial upload cleanup, and RPC reconciliation cases; `tests/ingestion/test_image_screening.py` OCR failure and screening behavior.

## Supabase and storage verification

Read-only checks against the configured Supabase project established:

- Bucket `ingestion-image-crops` exists and is private (`public=false`).
- `image_screening_results` has **32** rows. Its storage metadata columns are present; **32/32** have bucket, object path, and size.
- Rows map to **six ingestion runs**, matching the current six-source corpus snapshot. Run rows resolve to source SHA-256 values.
- **Five** sampled object paths returned HTTP 200 via `HEAD`. This was a sample check; the remaining 27 object bodies/existence were not individually fetched.
- All **32** image page/block pairs were absent from the current source chunk metadata when matched by latest run, page, and block.

The bucket list and selected metadata were inspected only. No image bytes, document text, or storage objects were downloaded. No writes, deletes, or signed/public URLs were created.

## Smallest remaining Phase 4 work

1. **Define the existing-ID join contract.** Use `ingestion_run_id + page_number + block_id` to locate the existing `image_screening_results` row and `image_sha256`/bucket/path; do not mint a second asset ID or duplicate stored images. Add a tested reader for this join.
2. **Make crop OCR source-backed text.** Preserve the OCR output as a canonical source block (or an equivalent immutable provenance record) with page/block identity, text offsets, hash, and screening decision. Only admitted text should be added to chunk/retrieval input. Keep `raw_text` as its source of truth.
3. **Link visual assets to chunks/evidence.** When an admitted OCR/caption block contributes to a chunk, carry its existing page/block identity through CPCR and evidence assembly, then resolve the existing asset metadata on demand. Add retrieval/evidence tests proving page/block/run alignment and that evidence text is exactly source-backed.
4. **Add visual descriptions only if OCR is insufficient.** For charts, diagrams, or non-text images, define a caption/description contract and retention policy, then persist it with the same existing run/page/block identity. No captioning implementation currently exists.
5. **Close lifecycle coverage.** Add tests for identical replay, changed content at the same run/page/block key, new-run versioning, missing object reconciliation, OCR text admission/quarantine, and visual evidence provenance. Keep the current private bucket and current metadata table unless a concrete limitation requires schema change.

These are proposals only. This audit did not implement Phase 4 functionality.
