# Phase 4 Milestone 2 implementation report

## Architecture

- `src/ingestion/screening/image_vlm.py` adds deterministic eligibility,
  validated structured descriptions, an adapter protocol and the explicit
  OpenAI adapter. VLM is opt-in using `IMAGE_VLM_PERSISTENCE_ENABLED=true`,
  `IMAGE_VLM_PROVIDER=openai`, and `IMAGE_VLM_MODEL`. The existing
  `OPENAI_API_KEY` is used without logging it.
- Only `NO_TEXT` images or images with admitted short OCR are candidates.
  Failed, rejected and quarantined OCR is excluded before the VLM adapter is
  called. Responses are schema-validated and run through the existing text
  screening policy before canonicalization. Per-document image, image-byte,
  timeout, retry, output-token and total-token limits are configurable.
- `CanonicalBlock` and `ChunkSource` support the distinct `IMAGE_VLM` method.
  Generated text has a deterministic page-scoped identity and retains source
  page, image block identity, exact text offsets, text digest, provider and
  model. Native text and OCR blocks remain separate. The rendered description
  labels itself AI-generated and unverified.
- The existing chunker carries the VLM provenance into sources. Existing full
  text/vector retrieval continues to search the chunk retrieval text; VLM
  evidence resolves the existing private asset through the same
  `(ingestion_run_id, page_number, block_id)` key after the Neo4j session closes.
- Chunking policy semantic version advances to `1.6.0`, so structural import
  treats chunks containing derived visual descriptions as a new immutable
  policy snapshot rather than conflicting with old same-index chunk identities.
- New migration
  `supabase/migrations/202611150001_image_vlm_descriptions.sql` adds a
  write-once JSON description to `image_screening_results` and a service-role
  atomic RPC. It preserves the existing table, storage bucket and image SHA.
  The migration has **not** been applied. Keep VLM disabled until deployment.

## Validation results

Focused command:

```powershell
.venv\Scripts\python.exe -m pytest tests\ingestion\test_image_vlm.py tests\ingestion\test_image_screening.py tests\ingestion\test_image_asset_resolver.py tests\ingestion\test_supabase_provenance.py tests\sql\test_phase4_image_vlm_sql_integration.py -q --basetemp .tmp_phase4_m2_pytest -p no:cacheprovider
```

Focused command (including the opt-in SQL integration test): **43 passed,
1 skipped in 3.07s**. The SQL integration skipped because the new migration has
not been deployed and no disposable M2 database was configured. The wider
regression command was

```powershell
.venv\Scripts\python.exe -m pytest tests\ingestion tests\graph tests\test_ingestion_chunking_packing.py tests\test_ingestion_chunking_routing.py tests\test_ingestion_chunking_m0_probes.py tests\test_ingestion_contracts.py tests\test_ingestion_adapters.py -q --basetemp .tmp_phase4_m2_broad -p no:cacheprovider
```

It reported **389 passed, 7 skipped in 38.13s**. Compilation passed for
the changed Python modules. Tests use mock providers, Supabase HTTP and Neo4j
sessions; they cover eligibility, malformed/retried responses, source
provenance, OCR separation, chunk offsets, exact replay/conflict handling,
asset resolution, and the existing FullTextRetriever path.

`test_data/phase4_m2_visual_eval_candidates_v1.json` contains three source-linked
visual candidates identified from rendered source pages and one explicit
missing-screenshot slot. All are `PENDING_HUMAN_REVIEW`; no labels are treated
as verified. The inspected local corpus has examples of diagrams, a chart, and
a text-heavy cover image, but a suitable screenshot example was not identified.

## Acceptance status and limitations

The code-level path is implemented, but the end-to-end acceptance pilot is not
complete. I did not send an image to OpenAI, apply the Supabase migration, write
Neo4j data, or re-ingest a document. The current working environment did not
provide an isolated Neo4j test database; Docker access was denied. The
FullTextRetriever integration test is mocked and does not prove a live index
search. The candidate evaluation set is not human-reviewed, and there is no
verified screenshot example.

Provider-reported input usage is known only after an image call completes, so
the total-token budget can be exceeded by one call. Image count, crop-byte size
and output-token limits bound that exposure. Strict pre-call input-token
enforcement needs provider-specific image-token estimation and remains a
calibration task.

Therefore the recommendation is **HOLD FOR MILESTONE 2 ACCEPTANCE** until the
migration is deployed to a disposable Supabase project, a disposable Neo4j
corpus is available, a one-image OpenAI pilot succeeds end to end, and the
evaluation examples are reviewed and completed.

## Remaining steps

1. Apply the migration through the approved Supabase workflow in a disposable
   project, then run the SQL immutability and RPC permission tests.
2. Configure an isolated Neo4j database and OpenAI model, ingest one test PDF
   containing a diagram/chart, import its chunks and run FullTextRetriever;
   verify the returned description digest and private asset composite key.
3. Have a human reviewer finalize expected descriptions and add a real
   screenshot source example. Measure field accuracy and unsafe-description
   rejection before enabling the feature beyond staging.
4. Deploy to staging with VLM disabled, apply the migration, enable VLM for a
   small canary, and review logs, token usage, screening outcomes and retrieval.
5. Production rollout follows staging sign-off; preserve rollback by disabling
   `IMAGE_VLM_PERSISTENCE_ENABLED` without changing existing OCR behavior.
