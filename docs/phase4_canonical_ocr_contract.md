# Phase 4 canonical image OCR contract

Admitted OCR from a rendered image crop is represented as a second canonical
`IMAGE` text block. The original blank `IMAGE` block remains intact, including
its parser-assigned `block_id`, coordinates and reading order identity.

## Identity and provenance

- The generated block ID is `page-{page_number}-block-ocr-{sha256}`. The digest
  is SHA-256 of `page_number`, the original image block ID, and the OCR text
  digest separated by NUL characters. It is deterministic for a given crop
  identity and OCR text and cannot collide with the parser's numeric block IDs.
- `CanonicalBlock.extraction_method` is `IMAGE_OCR`; `source_image_block_id`
  names the original IMAGE block; `ocr_text_sha256` covers the exact UTF-8 OCR
  string. OCR block offsets are Python Unicode character offsets into its
  canonical text.
- Only `TEXT_FOUND` results with `ADMITTED` or `ADMITTED_FLAGGED` decisions
  enter canonical text. Quarantined, rejected, failed, missing and empty OCR
  results do not become chunks. The OCR text is screened again as part of the
  normal chunk screening pass before CPCR admission.
- `ChunkSource` copies extraction method, OCR digest and original image block
  identity while retaining the generated text block's page and half-open source
  offsets.
- Native parser blocks keep their original `reading_order` and IDs. OCR blocks
  reuse the source image's reading order and carry `derived_order=1`; canonical
  ordering is `(reading_order, derived_order-or-zero)`. The generated OCR block
  sorts immediately after its image without renumbering parser blocks. Chunk
  provenance carries both order fields, keeping crop metadata aligned to its
  original block.

## Stored OCR and visual asset reference

The existing `image_screening_results` row remains the asset record, keyed by
`(ingestion_run_id, page_number, block_id)`. Migration
`202611130001_image_ocr_canonical_text.sql` adds nullable `ocr_text` and a
service-role-only atomic RPC. It persists screening/document/chunk/image
metadata and admitted OCR in one transaction. OCR validation failure rolls
back the entire write; retrying the same ingestion run is idempotent. It
validates the existing screening decision, stored digest and length. Existing
historical rows remain null and are not backfilled. Exact replay is accepted;
different text for an existing key is rejected. The shared append-only trigger
keeps OCR's one-time NULL-to-text exception inside a nested image-table branch;
all other image metadata and provenance tables remain immutable.

The persisted `ChunkSource` carries stable OCR provenance only: page, generated
text block, original image block ID, OCR method/hash/offsets. It does not carry
the run-specific object path, which would mutate a stable Neo4j chunk identity
on each ingestion run. After Neo4j retrieval has closed its session, the
retriever resolves `(EvidenceBundle.ingestion_run_id, page_number,
source_image_block_id)` through `SupabaseImageAssetResolver`. The resulting
`VisualAssetReference` is attached to `EvidenceBundle.visual_assets` with the
run, page, original image block ID, crop SHA-256, bucket, private object path
and file size. No public or signed URL is created, and image bytes are not
loaded during retrieval.

OCR source metadata lives in existing `sources_json`; run-specific storage
references are resolved during evidence assembly. No Neo4j label,
relationship, index or retrieval ranking change is needed.
