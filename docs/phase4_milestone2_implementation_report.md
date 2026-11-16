# Phase 4 Milestone 2 implementation report

## Architecture

- `src/ingestion/screening/image_vlm.py` adds deterministic eligibility,
  validated structured descriptions, an adapter protocol and the explicit
  OpenAI adapter. VLM is opt-in using `IMAGE_VLM_PERSISTENCE_ENABLED=true`,
  `IMAGE_VLM_PROVIDER=openai`, and `IMAGE_VLM_MODEL`. The existing
  `OPENAI_API_KEY` is used without logging it.
- Only `NO_TEXT` images or images with admitted short OCR are candidates.
  Failed, rejected and quarantined OCR is excluded before the VLM adapter is
  called. Every eligible crop under the configured byte limit is attempted;
  calls have per-image timeout/retry bounds and a 600-token output cap. Each
  response is schema-validated and run through the existing text screening
  policy before canonicalization. There is no silent document-wide image,
  timeout or token cutoff.
- `CanonicalBlock` and `ChunkSource` support the distinct `IMAGE_VLM` method.
  Generated text has a deterministic page-scoped identity and retains source
  page, image block identity, exact text offsets, text digest, provider, model
  and the actual `prompt_version` returned with the result. Prompt version is
  part of the derived block identity, is copied into chunk provenance, and is
  checked against persisted asset metadata during evidence resolution. Native
  text and OCR blocks remain separate. The rendered description labels itself
  AI-generated and unverified.
- The existing chunker carries the VLM provenance into sources. Existing full
  text/vector retrieval continues to search the chunk retrieval text; VLM
  evidence resolves the existing private asset through the same
  `(ingestion_run_id, page_number, block_id)` key after the Neo4j session closes.
- Chunking policy semantic version is now `1.6.1`: expanding VLM eligibility can
  add derived blocks and change chunk contents. Structural import therefore
  stores the expanded output as a new immutable policy snapshot instead of
  conflicting with existing `1.6.0` chunk identities.
- New migration
  `supabase/migrations/202611150001_image_vlm_descriptions.sql` adds a
  write-once JSON description to `image_screening_results` and a service-role
  atomic RPC. It preserves the existing table, storage bucket and image SHA.
  The configured disposable Supabase project returned HTTP 200 for the
  `visual_description` column read check, and the live ingestion pilot
  successfully persisted through the wrapper.

## Validation results

### VLM eligibility expansion follow-up (2026-11-16)

- A fresh run of all six PDFs under chunk policy `1.6.1` succeeded. Supabase
  contains 32 image records with 29 distinct image SHA-256 values. Based on the
  deterministic OCR/size policy, 26 unique images were eligible; 25 have an
  admitted description persisted. Three unique images were ineligible.
- The one eligible image without a persisted description is
  `9118a034b77848c9999d83ae2e02b0cd`, page 14, `page-14-block-0`. Failure or
  screening-rejection status is not persisted, so its outcome cannot be
  distinguished from the stored row alone.
- The 28 persisted description records identify OpenAI `gpt-4o-mini`; recorded
  output usage ranged from 34 to 165 tokens, within the 600-token request cap.
- Structural import into the existing `test-rag` graph was attempted but
  rejected by the writer because the same DOCUMENT identity already has a
  different canonical document digest. No Neo4j records were changed by that
  failed transaction; graph refresh requires an isolated fresh test database.
- Focused tests: 40 passed. Python compilation and `git diff --check` passed.

Historical focused command from the earlier psql-launcher run (the launcher has
since been removed and this command is not a current runnable command):

```powershell
.venv\Scripts\python.exe -m pytest tests\ingestion\test_image_vlm.py tests\ingestion\test_image_screening.py tests\ingestion\test_image_asset_resolver.py tests\ingestion\test_supabase_provenance.py tests\sql\test_phase4_image_vlm_sql_integration.py -q --basetemp .tmp_phase4_m2_focused -p no:cacheprovider
```

Focused result: **46 passed, 1 skipped in 3.64s**. At that time the local
psql-based SQL launcher skipped because its DSN and `psql` were unavailable. Its
core online assertions were separately exercised over the configured
disposable Supabase PostgREST endpoint as `service_role`: the VLM wrapper
rejected a run mismatch (HTTP 400), rejected a wrong image hash after the base
write (HTTP 400), and left no document, run or image rows from the failed
transaction. Direct helper execution was denied (HTTP 403). The successful
pilot below also verified service-role wrapper execution and owner helper
execution. The SQL scenario file is retained for Supabase SQL Editor execution;
the psql-only pytest launcher has since been removed.

The latest focused rerun at the time, after the OpenAI SDK response-format
typing fix:
**38 passed, 1 skipped in 3.67s** (image VLM, asset resolver, provenance and
opt-in SQL integration tests). The SQL integration still skips because no
`PHASE4_VLM_SQL_TEST_DSN`/psql setup is configured. `py_compile` passed, and
`mypy src/ingestion/screening/image_vlm.py --follow-imports=silent` exited 0.

The broader regression command was

```powershell
.venv\Scripts\python.exe -m pytest tests\ingestion tests\graph tests\test_ingestion_chunking_packing.py tests\test_ingestion_chunking_routing.py tests\test_ingestion_chunking_m0_probes.py tests\test_ingestion_contracts.py tests\test_ingestion_adapters.py -q --basetemp .tmp_phase4_m2_broad -p no:cacheprovider
```

It reported **392 passed, 7 skipped in 35.93s**. Python compilation and
`git diff --check` passed. Unit tests use mock providers, Supabase HTTP and Neo4j
sessions; they cover eligibility, malformed/retried responses, source
provenance, OCR separation, chunk offsets, exact replay/conflict handling,
asset resolution, and the existing FullTextRetriever path.

### Live one-page pilot

The API ran with `IMAGE_VLM_PERSISTENCE_ENABLED=true`, OpenAI
`gpt-4o-mini`, and the 20,000-token per-document default. The input was a
one-page test copy of page 8 from `cyber_security.pdf`; the original PDF was
not changed. The API returned `SUCCEEDED`, one page and four chunks.

- Source SHA-256: `5072f60b0622470ffd80a376ef3890a3e728933ce3f02fbc6100dd9ca43940b3`
- Ingestion run: `ec6a1c9de53647d896fc08f916d83c0d`
- Image key: page 1, `page-1-block-2`; image SHA-256:
  `d5e1689b18703d91f01130dc6d5c0dbd02ac8b6570719a331704c6a1778bb0fd`
- Storage: private `ingestion-image-crops` bucket; persisted crop size 127,387 bytes.
- OCR remained `TEXT_FOUND` and `ADMITTED`; VLM output was admitted with
  provider `openai`, model `gpt-4o-mini`, prompt version
  `visual-description-v1`, and description SHA-256
  `83f506cdd31dbc246ddd53e21b5c428ae63c9cc82bd25ae030257cec1e069a85`.
- The model described the Cybersecurity Framework Core diagram, its six
  functions, category/subcategory hierarchy and uncertainty about missing
  labels. Reported usage was 14,399 input and 127 output tokens.
- Supabase corpus CLI dry-run validated all four CPCR chunks. Neo4j import
  created one DOCUMENT, four CHUNKs, four HAS_CHUNK and three NEXT_CHUNK
  relationships (12 records); `--verify-idempotency` created zero records.
- Read-only FullTextRetriever returned VLM chunk
  `54ce4d0dbd5297b2ff1b30c2fa83a7efed8d79149e8cb2be6c54e6cae3febd0a` with
  the matching run, image digest, source block and private bucket/object
  reference. No public URL was returned.
- Comparing the VLM-off pilot copy with this run showed the three non-VLM
  chunks and their source mappings were identical; the VLM run added one
  `IMAGE_VLM` source.

After the migration update, a fresh pilot was run again. It returned
`SUCCEEDED` with four chunks; the newest ingestion run was
`bc39895054a74b2ea5bced44065186aa` for source hash
`68865b3f89d19ddb52deb3a68f34c1eff3b68e25fc885de595a63d97d41d7efb`.
The wrapper persisted an admitted description on image
`page-1-block-2` (`visual-description-v1`, description SHA-256
`3a33cc28fe2ab5a2b6b9631e4e5e5dd16db978929f9910fff7b9e23532f8e286`). Corpus
dry-run validated four chunks; Neo4j import created 12 records and its
idempotency rerun created zero. Read-only full-text retrieval returned VLM
chunk `71b12b4c0b258149db57b5fad1f1ac77db642237c358c338796b0ef2ccefffd5`
with the matching image hash and private object reference.

Two earlier isolated uploads did not produce descriptions: one server started
without VLM enabled, and the next exposed JSON-mode response shape failures
and the former 6,000-token ceiling. The adapter now requests strict JSON Schema
output and the default token ceiling is 20,000. Those first attempts remain
as separate test records in the disposable project; the successful run uses
a third unique test-document hash.

`test_data/phase4_m2_visual_eval_candidates_v1.json` contains three source-linked
visual candidates identified from rendered source pages and one explicit
missing-screenshot slot. All are `PENDING_HUMAN_REVIEW`; no labels are treated
as verified. The inspected local corpus has examples of diagrams, a chart, and
a text-heavy cover image, but a suitable screenshot example was not identified.

## Acceptance status and limitations

The one-image pilot and live retrieval path are verified in the disposable
development environment. The candidate evaluation set is still not
human-reviewed and it lacks a sourced screenshot example. That limits
evaluation coverage; human review is not a technical acceptance gate.

Provider-reported input usage is known only after an image call completes, so
the total-token budget can be exceeded by one call. Image count, crop-byte size
and output-token limits bound that exposure. Strict pre-call input-token
enforcement needs provider-specific image-token estimation and remains a
calibration task.

The current overall Phase 4 verdict is **PARTIAL** because the Supabase SQL
Editor permission and transaction scenarios have not been executed in the
current verification run. See
`docs/phase4_closure_phase5_implementation.md` for the current test totals,
skip explanations and database verification boundary.

## Remaining steps

1. Verify remote migration history, then run
   `tests/sql/test_phase4_image_vlm_persistence.sql` in the disposable Supabase
   SQL Editor to complete the database-native role and rollback checks.
2. The host-only OCR reading-order test skips without host Tesseract. The
   isolated parser worker image installs Tesseract in
   `docker/parser-worker.Dockerfile`; rebuild that image only when its code,
   dependencies or Dockerfile change.
3. Deploy to staging with VLM disabled, apply the migration only if it is not
   already deployed, enable VLM for a
   small canary, and review logs, token usage, screening outcomes and retrieval.
4. Production rollout follows staging sign-off; preserve rollback by disabling
   `IMAGE_VLM_PERSISTENCE_ENABLED` without changing existing OCR behavior.
