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
- A loopback-only opt-in SQL integration test and Python retry/replay/failure
  tests cover these corrections. The SQL test uses a transaction and rolls back
  fixtures.

Migration `supabase/migrations/202611130001_image_ocr_canonical_text.sql` is
pending deployment. It was not applied to live Supabase. Apply it to the target
environment before enabling this persistence path.

## Verification

- Focused: `63 passed, 1 skipped` across image screening, asset resolution,
  Supabase provenance, chunk contract, and opt-in SQL integration tests.
- Broader ingestion regression: `220 passed, 2 skipped` across `tests/ingestion`,
  the ingestion adapters and chunk contract/packing/routing tests.
- Python compilation passed for all changed modules and tests; `git diff --check`
  passed.
- SQL integration was skipped because `psql` and
  `PHASE4_SQL_TEST_DSN` are not configured. The SQL test requires a disposable
  loopback database with all migrations applied and rolls back its fixtures.
- The other broader-suite skip is the scanned two-column PDF test, which needs
  Tesseract.
- No live Supabase, storage, Neo4j, corpus, or benchmark writes were performed.
  Migration `202611130001_image_ocr_canonical_text.sql` remains pending
  deployment.
