# Supabase ingestion provenance

Apply all migrations in order in the Supabase SQL editor:

1. [`202611050001_ingestion_provenance.sql`](../supabase/migrations/202611050001_ingestion_provenance.sql)
   creates document, run, source-chunk, screening and image-screening tables.
2. [`202611070001_ingestion_cpcr_output.sql`](../supabase/migrations/202611070001_ingestion_cpcr_output.sql)
   adds an append-only final CPCR output artifact.
3. [`202611080001_ingestion_provenance_immutability.sql`](../supabase/migrations/202611080001_ingestion_provenance_immutability.sql)
   prevents updates or deletes from replacing existing run, chunk, and screening
   records. An attempt to reuse a chunk identity with different content fails
   rather than overwriting the earlier record.

Both migrations enable row-level security without client-facing policies and
expose their write RPCs only to the backend service role.

Set these server-side environment variables (never in browser code):

```dotenv
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<service-role-key>
```

When both variables are configured, the endpoint persists source and canonical
SHA-256 digests, G0 file verdict, original chunk text/ranges, screening results,
and image OCR metadata before CPCR. After CPCR, it stores raw and retrieval
chunk text, reversible normalization edits, source ranges, admission mappings,
CPCR decisions and a canonical payload digest in the append-only output table.
A failure at either persistence step returns `503`; the endpoint does not return
an output it could not durably record. If neither variable is configured, local
ingestion continues without durable PostgreSQL persistence.

Chunk text is stored for auditability and is therefore sensitive source data.
The schema provides no `anon` or `authenticated` access policy. Keep the
service-role key private and establish retention/backups in your Supabase
project before using sensitive production documents.
