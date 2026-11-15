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

## Migration requirement

The existing row contains only OCR digest and length, so the actual admitted
text cannot be reconstructed from the current database or included in
canonical chunks. A nullable text column plus a service-role-only write-once
RPC is therefore required. Existing rows remain untouched; fresh ingestion
runs populate the new field. Apply
`supabase/migrations/202611130001_image_ocr_canonical_text.sql` before enabling
the application change.

## Verification

Final focused run: `75 passed` across image screening, image asset resolver,
Supabase provenance, retrieval smoke validation, hybrid retrieval, and
evaluation-pipeline tests. Broader related regression run: `264 passed, 1 skipped`.
Compilation passed for every changed Python module and test module.
The skipped test is the two-column scanned-PDF ordering case, which requires
the Tesseract executable and skips when it is unavailable.
`git diff --check` passed. No live Supabase, storage, Neo4j, corpus, or benchmark
writes were performed. The migration was not applied to Supabase during this
code change; apply it before enabling OCR persistence in a deployed environment.
