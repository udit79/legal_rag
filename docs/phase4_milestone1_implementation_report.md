# Phase 4 Milestone 1 implementation report

## Reused implementation

- `screen_image_crops()` remains the source of OCR and its existing screening
  decision; it now retains plaintext only for admitted/flagged OCR.
- `crop_unrecognized_images()` remains the source of actual PDF crop bytes,
  page/block identity and coordinates.
- `ingest_pdf()` and `chunk_document()` remain the canonical chunk pipeline.
  The API now runs image OCR before `ingest_pdf()` so admitted OCR can be added
  to canonical blocks before chunking and normal chunk screening.
- `SupabaseProvenanceStore` continues to store crops and image screening rows;
  the new resolver reads the existing composite-key row. CPCR and retrieval
  continue to use their current contracts and query/index behavior.

## Changes

- Added source-backed OCR canonical blocks with deterministic IDs, text digests,
  OCR method labels and exact chunk offsets.
- Added a narrowly scoped migration to retain admitted OCR text write-once in
  the existing `image_screening_results` row. It adds no asset table or bucket.
- Added composite-key image resolver and typed private asset reference. Stable
  OCR source identity is persisted in existing chunk `sources` JSON; the
  run-specific private path is resolved after retrieval into
  `EvidenceBundle.visual_assets`, avoiding chunk-property changes between runs.
- Added regression coverage for admitted and excluded OCR, empty/failing OCR,
  offsets, resolver validation, evidence references, replay and run versioning.

## Review corrections and migration status

- The shared mutation trigger uses a nested table-name branch before accessing
  OCR fields. Its one-time NULL-to-text allowance is limited to
  `image_screening_results`; the four provenance tables remain append-only.
- Parser-assigned IDs and reading orders remain unchanged. Generated OCR blocks
  retain the image's reading order and use a separate `derived_order` to sort
  after their source image. Chunk provenance retains both values.
- `persist_screened_document()` calls the service-role-only
  `persist_ingestion_screening_with_admitted_ocr` RPC. It runs the existing
  provenance writer and OCR writer in one database transaction. OCR failure
  rolls the entire write back; exact retries use the same ingestion run ID and
  payload, while conflicting OCR is rejected. The API proceeds only after this
  RPC succeeds.
- An explicitly opt-in SQL integration test and Python retry/replay/failure
  tests cover these corrections. The SQL test uses a transaction and rolls back
  fixtures; remote execution additionally requires `PHASE4_SQL_TEST_ALLOW_REMOTE=1`.

Migration `supabase/migrations/202611130001_image_ocr_canonical_text.sql` is
present on the disposable hosted Supabase project used for the successful SQL
test. Apply it to any other target environment before enabling this persistence
path there.

## Verification

- Focused Python tests: `64 passed` across image screening, asset resolution,
  Supabase provenance, and chunk contract tests.
- SQL integration: `1 passed` against the explicitly authorized disposable
  hosted Supabase project. The test transaction rolled back its fixtures.
- Broader ingestion regression: `222 passed, 1 skipped` across `tests/ingestion`,
  the ingestion adapters and chunk contract/packing/routing tests.
- Python compilation passed for all changed modules and tests; `git diff --check`
  passed.
- The other broader-suite skip is the scanned two-column PDF test, which needs
  Tesseract.
- The Supabase test fixture transaction rolled back; no persistent Supabase,
  storage, Neo4j, corpus, or benchmark changes were made.

## Final pilot validation (2026-11-15)

The representative pilot used run
`b28173a879364e44ada2e6bbd787d81c` for the metadata-only copy of
`cyber_security.pdf` (32 pages, 91 chunks). The PDF page renderings were
verified pixel-identical to the source; the copy was used to avoid changing or
re-ingesting the existing corpus document. The run completed successfully.

- **Crop integrity — PASS:** Read-only metadata and Storage checks found 12
  crop objects. All 12 were downloaded in memory; each object SHA-256 and byte
  length matched its `image_screening_results` metadata. Eleven OCR results
  were admitted and one was `NO_TEXT`.
- **OCR retrieval — PASS:** Page-8 OCR evidence retained its original page and
  image block provenance and resolved to the matching private
  `ingestion-image-crops` object by the existing run/page/block key. No public
  URL or copied asset was created.
- **Graph/idempotency — PASS:** The pilot had 91 chunks, 91 `HAS_CHUNK` and 90
  `NEXT_CHUNK` relationships. The structural replay created zero records.
- **CPCR continuation warnings — benign false positives:** Two proposed
  `TABLE_CONTINUATION` links (pages 23→24 and 25→26) were rejected because
  their source chunks were not table chunks. Inspection of those extracted
  chunks and rendered PDF pages showed structured bullet/outline lists, not
  tables. The fail-closed CPCR guard prevented incorrect links; no extraction
  defect was found and no code change was warranted.
- **OCR exclusion — PASS:** Focused regression coverage confirms explicitly
  rejected, quarantined, failed and empty OCR results do not create canonical
  OCR blocks or chunks. The focused run reported `25 passed`.
- **Internal SQL function permissions — PASS:** A read-only privilege query
  confirmed the wrapper owner is `postgres` and it can execute both
  `persist_ingestion_screening(jsonb)` and
  `persist_admitted_image_ocr(text,integer,text,text,text,integer)`.
- **Remote migration history — BLOCKED / NOT REGISTERED:** The remote catalog
  has no `supabase_migrations.schema_migrations` relation; only the platform
  `auth`, `realtime` and `storage` migration tables were present. Therefore
  migration `202611130001` is not verifiably registered in Supabase CLI
  migration history, even though the deployed RPC and pilot behavior confirm
  its SQL is present. It was not reapplied. Reconcile the deployment ledger
  through the approved Supabase migration workflow before staging.

The validation added no persistent remote writes, did not rerun ingestion,
and did not modify Neo4j or Phase 3 evaluation artifacts. The migration-history
discrepancy is the remaining staging gate; recommendation: **HOLD**.
