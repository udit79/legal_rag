-- Narrow, read-only graph export. The CPCR table remains inaccessible directly.
-- DROP is required because PostgreSQL cannot replace a function with a changed
-- RETURNS TABLE signature in place. The grants are restored below.
-- Ingestion ordering requires every run to have an authoritative timestamp.
-- PostgreSQL aborts this statement if historical rows still contain NULL;
-- repair those rows explicitly rather than inventing ordering timestamps.
alter table public.ingestion_runs
    alter column created_at set not null;

drop function if exists public.read_latest_ingestion_graph_corpus(text[]);

create function public.read_latest_ingestion_graph_corpus(
    p_source_sha256s text[]
)
returns table (
    source_sha256 text,
    canonical_document_digest text,
    source_page_count integer,
    ingestion_run_id text,
    ingestion_created_at timestamptz,
    run_source_sha256 text,
    chunking_policy_version text,
    screening_policy_version text,
    screening_policy_digest text,
    document_quarantined boolean,
    cpcr_ingestion_run_id text,
    cpcr_source_sha256 text,
    output_payload jsonb,
    screening_results jsonb
)
language plpgsql
security definer
set search_path = pg_catalog, public, pg_temp
as $$
declare
    requested_hash text;
    latest_run public.ingestion_runs%rowtype;
    source_document public.ingestion_documents%rowtype;
    cpcr public.ingestion_cpcr_outputs%rowtype;
begin
    if p_source_sha256s is null or cardinality(p_source_sha256s) = 0 then
        raise exception 'At least one source hash is required';
    end if;
    if exists (
        select 1 from unnest(p_source_sha256s) as requested(value)
        where requested.value !~ '^[0-9a-f]{64}$'
    ) or cardinality(p_source_sha256s) <> (
        select count(distinct requested.value)
        from unnest(p_source_sha256s) as requested(value)
    ) then
        raise exception 'Source hashes must be unique lowercase SHA-256 values';
    end if;

    foreach requested_hash in array p_source_sha256s loop
        select r.* into latest_run
        from public.ingestion_runs r
        where r.source_sha256 = requested_hash
        order by r.created_at desc nulls last
        limit 1;
        if not found then
            raise exception 'No ingestion run exists for requested source %', requested_hash;
        end if;
        if latest_run.created_at is null then
            raise exception 'Latest ingestion run has a NULL created_at for source %', requested_hash;
        end if;
        if (
            select count(*)
            from public.ingestion_runs r
            where r.source_sha256 = requested_hash
              and r.created_at = latest_run.created_at
        ) > 1 then
            raise exception 'Ambiguous latest ingestion timestamp for source %', requested_hash;
        end if;

        select d.* into source_document
        from public.ingestion_documents d
        where d.source_sha256 = requested_hash;
        if not found then
            raise exception 'Source document metadata is missing for %', requested_hash;
        end if;

        select c.* into cpcr
        from public.ingestion_cpcr_outputs c
        where c.ingestion_run_id = latest_run.ingestion_run_id
          and c.source_sha256 = requested_hash;
        if not found then
            raise exception 'Latest ingestion run has no matching CPCR output for source %', requested_hash;
        end if;

        return query
        select requested_hash,
               source_document.canonical_document_digest,
               source_document.source_page_count,
               latest_run.ingestion_run_id,
               latest_run.created_at,
               latest_run.source_sha256,
               latest_run.chunking_policy_version,
               latest_run.screening_policy_version,
               latest_run.screening_policy_digest,
               latest_run.document_quarantined,
               cpcr.ingestion_run_id,
               cpcr.source_sha256,
               cpcr.output_payload,
               coalesce(
                   (select jsonb_agg(jsonb_build_object(
                       'chunk_index', s.chunk_index,
                       'chunk_digest', s.chunk_digest,
                       'decision', s.decision
                   ) order by s.chunk_index)
                    from public.chunk_screening_results s
                    where s.ingestion_run_id = latest_run.ingestion_run_id
                      and s.source_sha256 = requested_hash
                      and s.chunking_policy_version = latest_run.chunking_policy_version),
                   '[]'::jsonb
               );
    end loop;
end;
$$;

revoke all on function public.read_latest_ingestion_graph_corpus(text[])
    from public, anon, authenticated;
grant execute on function public.read_latest_ingestion_graph_corpus(text[])
    to service_role;
