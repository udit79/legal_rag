# Supabase ingestion provenance

Apply [`202611050001_ingestion_provenance.sql`](../supabase/migrations/202611050001_ingestion_provenance.sql)
in the Supabase SQL editor. It creates document, ingestion-run, chunk,
chunk-screening and image-screening tables, enables row-level security without
client-facing policies, and exposes one transactional RPC to the backend's
service role.

Set these server-side environment variables (never in browser code):

```dotenv
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<service-role-key>
```

When both variables are configured, the ingestion endpoint persists source and
canonical SHA-256 digests, G0 file verdict, chunk text and source ranges,
screening decisions, flags, and image OCR screening metadata before CPCR. A database failure returns
`503` and stops the request before CPCR. If neither variable is configured,
local ingestion continues without durable PostgreSQL persistence.

Chunk text is stored for auditability and is therefore sensitive source data.
The schema provides no `anon` or `authenticated` access policy. Keep the
service-role key private and establish retention/backups in your Supabase
project before using sensitive production documents.
