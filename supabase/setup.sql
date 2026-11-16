-- Consolidated Supabase bootstrap for a fresh project.
-- Run this file once in the Supabase SQL Editor. Existing migration files are
-- retained as deployment history; do not run this after the schema is already
-- provisioned unless you intend to reapply the idempotent setup.
-- The image bucket remains private; backend access uses service_role.

insert into storage.buckets (id, name, public)
values ('ingestion-image-crops', 'ingestion-image-crops', false)
on conflict (id) do update set public = false;

-- ============================================================================
-- Source: 202611050001_ingestion_provenance.sql
-- ============================================================================

-- Apply in the Supabase SQL editor. The app calls only the service-role RPC;
-- raw chunk text and security results are not exposed to anon/authenticated.
create table if not exists public.ingestion_documents (
    source_sha256 text primary key check (source_sha256 ~ '^[0-9a-f]{64}$'),
    source_id text not null,
    canonical_document_digest text not null check (canonical_document_digest ~ '^[0-9a-f]{64}$'),
    source_page_count integer not null check (source_page_count > 0),
    created_at timestamptz not null default now()
);

create table if not exists public.ingestion_runs (
    ingestion_run_id text primary key,
    source_sha256 text not null references public.ingestion_documents(source_sha256),
    chunking_policy_version text not null,
    screening_policy_version text not null,
    screening_policy_digest text not null check (screening_policy_digest ~ '^[0-9a-f]{64}$'),
    trust_tier text not null check (trust_tier in ('T0', 'T1', 'T2', 'T3')),
    document_quarantined boolean not null default false,
    file_screening jsonb,
    created_at timestamptz not null default now(),
    unique (ingestion_run_id, source_sha256, chunking_policy_version)
);

create table if not exists public.ingestion_chunks (
    source_sha256 text not null references public.ingestion_documents(source_sha256),
    chunking_policy_version text not null,
    chunk_index integer not null check (chunk_index >= 0),
    chunk_digest text not null check (chunk_digest ~ '^[0-9a-f]{64}$'),
    chunk_text text not null,
    sources jsonb not null,
    created_at timestamptz not null default now(),
    primary key (source_sha256, chunking_policy_version, chunk_index)
);

create table if not exists public.chunk_screening_results (
    ingestion_run_id text not null,
    source_sha256 text not null,
    chunking_policy_version text not null,
    chunk_index integer not null check (chunk_index >= 0),
    chunk_digest text not null check (chunk_digest ~ '^[0-9a-f]{64}$'),
    decision text not null check (decision in ('ADMITTED', 'ADMITTED_FLAGGED', 'QUARANTINED', 'REJECTED')),
    flags jsonb not null,
    justification_codes text[] not null,
    result jsonb not null,
    created_at timestamptz not null default now(),
    primary key (ingestion_run_id, chunk_index),
    foreign key (ingestion_run_id, source_sha256, chunking_policy_version)
        references public.ingestion_runs(ingestion_run_id, source_sha256, chunking_policy_version),
    foreign key (source_sha256, chunking_policy_version, chunk_index)
        references public.ingestion_chunks(source_sha256, chunking_policy_version, chunk_index)
);

create table if not exists public.image_screening_results (
    ingestion_run_id text not null references public.ingestion_runs(ingestion_run_id),
    page_number integer not null check (page_number > 0),
    block_id text not null,
    reading_order integer not null check (reading_order >= 0),
    image_sha256 text not null check (image_sha256 ~ '^[0-9a-f]{64}$'),
    ocr_status text not null check (ocr_status in ('TEXT_FOUND', 'NO_TEXT', 'FAILED')),
    ocr_text_sha256 text check (ocr_text_sha256 is null or ocr_text_sha256 ~ '^[0-9a-f]{64}$'),
    ocr_text_length integer not null check (ocr_text_length >= 0),
    decision text check (decision is null or decision in ('ADMITTED', 'ADMITTED_FLAGGED', 'QUARANTINED', 'REJECTED')),
    source_coordinates jsonb not null,
    flags jsonb not null,
    created_at timestamptz not null default now(),
    primary key (ingestion_run_id, page_number, block_id)
);

alter table public.ingestion_documents enable row level security;
alter table public.ingestion_runs enable row level security;
alter table public.ingestion_chunks enable row level security;
alter table public.chunk_screening_results enable row level security;
alter table public.image_screening_results enable row level security;

create or replace function public.persist_ingestion_screening(p_payload jsonb)
returns void
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
    v_document jsonb := p_payload -> 'document';
    v_run jsonb := p_payload -> 'run';
    v_source_sha text := v_document ->> 'source_sha256';
    v_run_id text := v_run ->> 'ingestion_run_id';
    v_chunking_version text := v_run ->> 'chunking_policy_version';
    item jsonb;
begin
    if v_document is null or v_run is null or v_source_sha is null or v_run_id is null then
        raise exception 'Required document and run provenance is missing';
    end if;

    insert into public.ingestion_documents
        (source_sha256, source_id, canonical_document_digest, source_page_count)
    values
        (v_source_sha, v_document ->> 'source_id',
         v_document ->> 'canonical_document_digest',
         (v_document ->> 'source_page_count')::integer)
    on conflict (source_sha256) do update set
        source_id = excluded.source_id,
        canonical_document_digest = excluded.canonical_document_digest,
        source_page_count = excluded.source_page_count;

    insert into public.ingestion_runs
        (ingestion_run_id, source_sha256, chunking_policy_version,
         screening_policy_version, screening_policy_digest, trust_tier,
         document_quarantined, file_screening)
    values
        (v_run_id, v_source_sha, v_chunking_version,
         v_run ->> 'screening_policy_version', v_run ->> 'screening_policy_digest',
         v_run ->> 'trust_tier',
         coalesce((v_run ->> 'document_quarantined')::boolean, false),
         v_run -> 'file_screening')
    on conflict (ingestion_run_id) do update set
        source_sha256 = excluded.source_sha256,
        chunking_policy_version = excluded.chunking_policy_version,
        screening_policy_version = excluded.screening_policy_version,
        screening_policy_digest = excluded.screening_policy_digest,
        trust_tier = excluded.trust_tier,
        document_quarantined = excluded.document_quarantined,
        file_screening = excluded.file_screening;

    for item in select value from jsonb_array_elements(coalesce(p_payload -> 'chunks', '[]'::jsonb)) loop
        insert into public.ingestion_chunks
            (source_sha256, chunking_policy_version, chunk_index, chunk_digest, chunk_text, sources)
        values
            (v_source_sha, v_chunking_version, (item ->> 'chunk_index')::integer,
             item ->> 'chunk_digest', item ->> 'text', item -> 'sources')
        on conflict (source_sha256, chunking_policy_version, chunk_index) do update set
            chunk_digest = excluded.chunk_digest,
            chunk_text = excluded.chunk_text,
            sources = excluded.sources;
    end loop;

    for item in select value from jsonb_array_elements(coalesce(p_payload -> 'screening_results', '[]'::jsonb)) loop
        insert into public.chunk_screening_results
            (ingestion_run_id, source_sha256, chunking_policy_version, chunk_index,
             chunk_digest, decision, flags, justification_codes, result)
        values
            (v_run_id, v_source_sha, v_chunking_version,
             (item ->> 'chunk_index')::integer, item ->> 'chunk_digest',
             item ->> 'decision', coalesce(item -> 'flags', '[]'::jsonb),
             array(select jsonb_array_elements_text(item -> 'justification_codes')), item)
        on conflict (ingestion_run_id, chunk_index) do update set
            chunk_digest = excluded.chunk_digest,
            decision = excluded.decision,
            flags = excluded.flags,
            justification_codes = excluded.justification_codes,
            result = excluded.result;
    end loop;

    for item in select value from jsonb_array_elements(coalesce(p_payload -> 'image_results', '[]'::jsonb)) loop
        insert into public.image_screening_results
            (ingestion_run_id, page_number, block_id, reading_order, image_sha256,
             ocr_status, ocr_text_sha256, ocr_text_length, decision,
             source_coordinates, flags)
        values
            (v_run_id, (item ->> 'page_number')::integer, item ->> 'block_id',
             (item ->> 'reading_order')::integer, item ->> 'image_sha256',
             item ->> 'ocr_status', item ->> 'ocr_text_sha256',
             coalesce((item ->> 'ocr_text_length')::integer, 0), item ->> 'decision',
             item -> 'source_coordinates', coalesce(item -> 'flags', '[]'::jsonb))
        on conflict (ingestion_run_id, page_number, block_id) do update set
            image_sha256 = excluded.image_sha256,
            ocr_status = excluded.ocr_status,
            ocr_text_sha256 = excluded.ocr_text_sha256,
            ocr_text_length = excluded.ocr_text_length,
            decision = excluded.decision,
            source_coordinates = excluded.source_coordinates,
            flags = excluded.flags;
    end loop;
end;
$$;

revoke all on function public.persist_ingestion_screening(jsonb) from public, anon, authenticated;
grant execute on function public.persist_ingestion_screening(jsonb) to service_role;

comment on table public.ingestion_chunks is
    'Contains source text. Access only through the trusted backend service role.';

-- ============================================================================
-- Source: 202611070001_ingestion_cpcr_output.sql
-- ============================================================================

-- Stores the exact CPCR response artifact for each server-generated run.
-- Apply in the Supabase SQL editor before enabling final-output persistence.
create table if not exists public.ingestion_cpcr_outputs (
    ingestion_run_id text primary key
        references public.ingestion_runs(ingestion_run_id),
    source_sha256 text not null
        references public.ingestion_documents(source_sha256),
    payload_sha256 text not null check (payload_sha256 ~ '^[0-9a-f]{64}$'),
    output_payload jsonb not null,
    created_at timestamptz not null default now()
);

alter table public.ingestion_cpcr_outputs enable row level security;
revoke all on table public.ingestion_cpcr_outputs
    from public, anon, authenticated, service_role;

create or replace function public.reject_ingestion_cpcr_output_mutation()
returns trigger
language plpgsql
as $$
begin
    raise exception 'CPCR output records are append-only';
end;
$$;

drop trigger if exists ingestion_cpcr_outputs_immutable
    on public.ingestion_cpcr_outputs;
create trigger ingestion_cpcr_outputs_immutable
    before update or delete on public.ingestion_cpcr_outputs
    for each row execute function public.reject_ingestion_cpcr_output_mutation();

create or replace function public.persist_ingestion_cpcr_output(
    p_ingestion_run_id text,
    p_source_sha256 text,
    p_payload_digest text,
    p_payload jsonb
)
returns void
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
    v_existing_digest text;
begin
    if p_ingestion_run_id is null or p_source_sha256 is null
       or p_payload_digest !~ '^[0-9a-f]{64}$' or p_payload is null then
        raise exception 'Required CPCR output provenance is missing';
    end if;
    if p_payload ->> 'ingestion_run_id' <> p_ingestion_run_id
       or p_payload ->> 'source_sha256' <> p_source_sha256 then
        raise exception 'CPCR output identity does not match its payload';
    end if;
    if not exists (
        select 1 from public.ingestion_runs r
        where r.ingestion_run_id = p_ingestion_run_id
          and r.source_sha256 = p_source_sha256
    ) then
        raise exception 'CPCR output has no matching ingestion run';
    end if;

    insert into public.ingestion_cpcr_outputs
        (ingestion_run_id, source_sha256, payload_sha256, output_payload)
    values (p_ingestion_run_id, p_source_sha256, p_payload_digest, p_payload)
    on conflict (ingestion_run_id) do nothing;

    select payload_sha256 into v_existing_digest
    from public.ingestion_cpcr_outputs
    where ingestion_run_id = p_ingestion_run_id;
    if v_existing_digest is distinct from p_payload_digest then
        raise exception 'Ingestion run already has a different CPCR output';
    end if;
end;
$$;

revoke all on function public.persist_ingestion_cpcr_output(text, text, text, jsonb)
    from public, anon, authenticated;
grant execute on function public.persist_ingestion_cpcr_output(text, text, text, jsonb)
    to service_role;

comment on table public.ingestion_cpcr_outputs is
    'Append-only final chunk and CPCR decision artifact for each ingestion run.';

-- ============================================================================
-- Source: 202611080001_ingestion_provenance_immutability.sql
-- ============================================================================

create or replace function public.reject_ingestion_provenance_mutation()
returns trigger
language plpgsql
set search_path = public, pg_temp
as $$
begin
    if tg_op = 'DELETE' then
        raise exception 'Ingestion provenance records are append-only'
            using errcode = '55000';
    end if;

    if to_jsonb(new) is distinct from to_jsonb(old) then
        raise exception 'Ingestion provenance records are immutable'
            using errcode = '55000';
    end if;

    return new;
end;
$$;

drop trigger if exists ingestion_runs_immutable
    on public.ingestion_runs;
create trigger ingestion_runs_immutable
before update or delete on public.ingestion_runs
for each row execute function public.reject_ingestion_provenance_mutation();

drop trigger if exists ingestion_chunks_immutable
    on public.ingestion_chunks;
create trigger ingestion_chunks_immutable
before update or delete on public.ingestion_chunks
for each row execute function public.reject_ingestion_provenance_mutation();

drop trigger if exists chunk_screening_results_immutable
    on public.chunk_screening_results;
create trigger chunk_screening_results_immutable
before update or delete on public.chunk_screening_results
for each row execute function public.reject_ingestion_provenance_mutation();

drop trigger if exists image_screening_results_immutable
    on public.image_screening_results;
create trigger image_screening_results_immutable
before update or delete on public.image_screening_results
for each row execute function public.reject_ingestion_provenance_mutation();

-- ============================================================================
-- Source: 202611110001_ingestion_image_storage.sql
-- ============================================================================

alter table public.image_screening_results
    add column if not exists image_bucket text,
    add column if not exists image_object_path text,
    add column if not exists image_file_size_bytes bigint;

do $
begin
    if not exists (
        select 1 from pg_catalog.pg_constraint
        where conname = 'image_screening_storage_metadata_complete'
          and conrelid = 'public.image_screening_results'::regclass
    ) then
        alter table public.image_screening_results
            add constraint image_screening_storage_metadata_complete
            check (
                (image_bucket is null and image_object_path is null
                    and image_file_size_bytes is null)
                or (image_bucket = 'ingestion-image-crops'
                    and image_object_path is not null
                    and length(image_object_path) > 0
                    and image_file_size_bytes is not null
                    and image_file_size_bytes >= 0)
            );
    end if;
end;
$;

create or replace function public.persist_ingestion_screening(p_payload jsonb)
returns void
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
    v_document jsonb := p_payload -> 'document';
    v_run jsonb := p_payload -> 'run';
    v_source_sha text := v_document ->> 'source_sha256';
    v_run_id text := v_run ->> 'ingestion_run_id';
    v_chunking_version text := v_run ->> 'chunking_policy_version';
    item jsonb;
begin
    if v_document is null or v_run is null or v_source_sha is null or v_run_id is null then
        raise exception 'Required document and run provenance is missing';
    end if;

    insert into public.ingestion_documents
        (source_sha256, source_id, canonical_document_digest, source_page_count)
    values
        (v_source_sha, v_document ->> 'source_id',
         v_document ->> 'canonical_document_digest',
         (v_document ->> 'source_page_count')::integer)
    on conflict (source_sha256) do update set
        source_id = excluded.source_id,
        canonical_document_digest = excluded.canonical_document_digest,
        source_page_count = excluded.source_page_count;

    insert into public.ingestion_runs
        (ingestion_run_id, source_sha256, chunking_policy_version,
         screening_policy_version, screening_policy_digest, trust_tier,
         document_quarantined, file_screening)
    values
        (v_run_id, v_source_sha, v_chunking_version,
         v_run ->> 'screening_policy_version', v_run ->> 'screening_policy_digest',
         v_run ->> 'trust_tier',
         coalesce((v_run ->> 'document_quarantined')::boolean, false),
         v_run -> 'file_screening')
    on conflict (ingestion_run_id) do update set
        source_sha256 = excluded.source_sha256,
        chunking_policy_version = excluded.chunking_policy_version,
        screening_policy_version = excluded.screening_policy_version,
        screening_policy_digest = excluded.screening_policy_digest,
        trust_tier = excluded.trust_tier,
        document_quarantined = excluded.document_quarantined,
        file_screening = excluded.file_screening;

    for item in select value from jsonb_array_elements(coalesce(p_payload -> 'chunks', '[]'::jsonb)) loop
        insert into public.ingestion_chunks
            (source_sha256, chunking_policy_version, chunk_index, chunk_digest, chunk_text, sources)
        values
            (v_source_sha, v_chunking_version, (item ->> 'chunk_index')::integer,
             item ->> 'chunk_digest', item ->> 'text', item -> 'sources')
        on conflict (source_sha256, chunking_policy_version, chunk_index) do update set
            chunk_digest = excluded.chunk_digest,
            chunk_text = excluded.chunk_text,
            sources = excluded.sources;
    end loop;

    for item in select value from jsonb_array_elements(coalesce(p_payload -> 'screening_results', '[]'::jsonb)) loop
        insert into public.chunk_screening_results
            (ingestion_run_id, source_sha256, chunking_policy_version, chunk_index,
             chunk_digest, decision, flags, justification_codes, result)
        values
            (v_run_id, v_source_sha, v_chunking_version,
             (item ->> 'chunk_index')::integer, item ->> 'chunk_digest',
             item ->> 'decision', coalesce(item -> 'flags', '[]'::jsonb),
             array(select jsonb_array_elements_text(item -> 'justification_codes')), item)
        on conflict (ingestion_run_id, chunk_index) do update set
            chunk_digest = excluded.chunk_digest,
            decision = excluded.decision,
            flags = excluded.flags,
            justification_codes = excluded.justification_codes,
            result = excluded.result;
    end loop;

    for item in select value from jsonb_array_elements(coalesce(p_payload -> 'image_results', '[]'::jsonb)) loop
        insert into public.image_screening_results
            (ingestion_run_id, page_number, block_id, reading_order, image_sha256,
             ocr_status, ocr_text_sha256, ocr_text_length, decision,
             source_coordinates, flags, image_bucket, image_object_path,
             image_file_size_bytes)
        values
            (v_run_id, (item ->> 'page_number')::integer, item ->> 'block_id',
             (item ->> 'reading_order')::integer, item ->> 'image_sha256',
             item ->> 'ocr_status', item ->> 'ocr_text_sha256',
             coalesce((item ->> 'ocr_text_length')::integer, 0), item ->> 'decision',
             item -> 'source_coordinates', coalesce(item -> 'flags', '[]'::jsonb),
             item ->> 'image_bucket', item ->> 'image_object_path',
             (item ->> 'image_file_size_bytes')::bigint)
        on conflict (ingestion_run_id, page_number, block_id) do update set
            image_sha256 = excluded.image_sha256,
            ocr_status = excluded.ocr_status,
            ocr_text_sha256 = excluded.ocr_text_sha256,
            ocr_text_length = excluded.ocr_text_length,
            decision = excluded.decision,
            source_coordinates = excluded.source_coordinates,
            flags = excluded.flags,
            image_bucket = excluded.image_bucket,
            image_object_path = excluded.image_object_path,
            image_file_size_bytes = excluded.image_file_size_bytes;
    end loop;
end;
$$;

revoke all on function public.persist_ingestion_screening(jsonb) from public, anon, authenticated;
grant execute on function public.persist_ingestion_screening(jsonb) to service_role;

-- ============================================================================
-- Source: 202611120001_read_latest_ingestion_graph_corpus.sql
-- ============================================================================

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

-- ============================================================================
-- Source: 202611130001_image_ocr_canonical_text.sql
-- ============================================================================

-- Retain admitted crop OCR under the existing composite asset identity.
alter table public.image_screening_results
    add column if not exists ocr_text text;

create or replace function public.reject_ingestion_provenance_mutation()
returns trigger
language plpgsql
set search_path = pg_catalog, public, pg_temp
as $$
begin
    if tg_op = 'DELETE' then
        raise exception 'Ingestion provenance records are append-only'
            using errcode = '55000';
    end if;

    if tg_table_name = 'image_screening_results' then
        if (to_jsonb(new) - 'ocr_text') = (to_jsonb(old) - 'ocr_text')
           and old.ocr_text is null
           and new.ocr_text is not null
           and new.ocr_status = 'TEXT_FOUND'
           and new.decision in ('ADMITTED', 'ADMITTED_FLAGGED')
           and char_length(new.ocr_text) = new.ocr_text_length
           and encode(sha256(convert_to(new.ocr_text, 'UTF8')), 'hex') = new.ocr_text_sha256 then
            return new;
        end if;
    end if;

    if to_jsonb(new) is distinct from to_jsonb(old) then
        raise exception 'Ingestion provenance records are immutable'
            using errcode = '55000';
    end if;

    return new;
end;
$$;

create or replace function public.persist_admitted_image_ocr(
    p_ingestion_run_id text,
    p_page_number integer,
    p_block_id text,
    p_ocr_text text,
    p_ocr_text_sha256 text,
    p_ocr_text_length integer
)
returns void
language plpgsql
security definer
set search_path = pg_catalog, public, pg_temp
as $$
declare
    current_row public.image_screening_results%rowtype;
begin
    if p_ocr_text is null or p_ocr_text_length is null or p_ocr_text_length <= 0
       or char_length(p_ocr_text) <> p_ocr_text_length
       or p_ocr_text_sha256 is null
       or p_ocr_text_sha256 !~ '^[0-9a-f]{64}$'
       or encode(sha256(convert_to(p_ocr_text, 'UTF8')), 'hex') <> p_ocr_text_sha256 then
        raise exception 'Image OCR text failed digest or length validation';
    end if;

    select * into current_row
    from public.image_screening_results
    where ingestion_run_id = p_ingestion_run_id
      and page_number = p_page_number
      and block_id = p_block_id
    for update;

    if not found or current_row.ocr_status <> 'TEXT_FOUND'
       or current_row.decision not in ('ADMITTED', 'ADMITTED_FLAGGED')
       or current_row.ocr_text_sha256 is distinct from p_ocr_text_sha256
       or current_row.ocr_text_length <> p_ocr_text_length then
        raise exception 'Image OCR does not match an admitted screening record';
    end if;

    if current_row.ocr_text is null then
        update public.image_screening_results
        set ocr_text = p_ocr_text
        where ingestion_run_id = p_ingestion_run_id
          and page_number = p_page_number
          and block_id = p_block_id;
    elsif current_row.ocr_text <> p_ocr_text then
        raise exception 'Image OCR conflicts with immutable stored content'
            using errcode = '55000';
    end if;
end;
$$;

-- Keep component writers private. The service-role entry point below wraps
-- both calls in one PostgreSQL transaction, so a failed OCR persist rolls
-- back the screening/document/chunk/image writes as well.
revoke all on function public.persist_ingestion_screening(jsonb)
    from public, anon, authenticated, service_role;
revoke all on function public.persist_admitted_image_ocr(text, integer, text, text, text, integer)
    from public, anon, authenticated, service_role;

create or replace function public.persist_ingestion_screening_with_admitted_ocr(
    p_payload jsonb,
    p_admitted_ocr jsonb
)
returns void
language plpgsql
security definer
set search_path = pg_catalog, public, pg_temp
as $$
declare
    item jsonb;
begin
    if p_payload is null or p_admitted_ocr is null
       or jsonb_typeof(p_admitted_ocr) <> 'array' then
        raise exception 'Screening payload and admitted OCR array are required';
    end if;

    perform public.persist_ingestion_screening(p_payload);

    for item in select value from jsonb_array_elements(p_admitted_ocr) loop
        perform public.persist_admitted_image_ocr(
            item ->> 'ingestion_run_id',
            (item ->> 'page_number')::integer,
            item ->> 'block_id',
            item ->> 'ocr_text',
            item ->> 'ocr_text_sha256',
            (item ->> 'ocr_text_length')::integer
        );
    end loop;
end;
$$;

revoke all on function public.persist_ingestion_screening_with_admitted_ocr(jsonb, jsonb)
    from public, anon, authenticated;
grant execute on function public.persist_ingestion_screening_with_admitted_ocr(jsonb, jsonb)
    to service_role;

comment on function public.persist_ingestion_screening_with_admitted_ocr(jsonb, jsonb) is
    'Atomically persists ingestion screening, image metadata and admitted image OCR; service_role only.';

-- ============================================================================
-- Source: 202611150001_image_vlm_descriptions.sql
-- ============================================================================

-- Persist admitted VLM descriptions on the existing image asset row.
alter table public.image_screening_results
    add column if not exists visual_description jsonb;

create or replace function public.reject_ingestion_provenance_mutation()
returns trigger
language plpgsql
set search_path = pg_catalog, public, pg_temp
as $$
begin
    if tg_op = 'DELETE' then
        raise exception 'Ingestion provenance records are append-only'
            using errcode = '55000';
    end if;

    if tg_table_name = 'image_screening_results' then
        if (to_jsonb(new) - 'ocr_text' - 'visual_description')
              = (to_jsonb(old) - 'ocr_text' - 'visual_description') then
            if old.ocr_text is null and new.ocr_text is not null
               and old.visual_description is not distinct from new.visual_description
               and new.ocr_status = 'TEXT_FOUND'
               and new.decision in ('ADMITTED', 'ADMITTED_FLAGGED')
               and char_length(new.ocr_text) = new.ocr_text_length
               and encode(sha256(convert_to(new.ocr_text, 'UTF8')), 'hex') = new.ocr_text_sha256 then
                return new;
            end if;

            if old.visual_description is null and new.visual_description is not null
               and old.ocr_text is not distinct from new.ocr_text
               and (new.visual_description ->> 'description_sha256') ~ '^[0-9a-f]{64}$'
               and encode(sha256(convert_to(new.visual_description ->> 'description_text', 'UTF8')), 'hex')
                    = new.visual_description ->> 'description_sha256'
               and new.visual_description ->> 'provider' <> ''
               and new.visual_description ->> 'model' <> ''
               and new.visual_description ->> 'prompt_version' <> ''
               and new.visual_description ->> 'description_text' is not null
               and new.visual_description ->> 'decision' in ('ADMITTED', 'ADMITTED_FLAGGED')
               and (
                   (new.ocr_status = 'NO_TEXT' and new.decision is null)
                   or (new.ocr_status = 'TEXT_FOUND'
                       and new.decision in ('ADMITTED', 'ADMITTED_FLAGGED'))
               )
               and coalesce((new.visual_description ->> 'input_tokens')::integer, -1) >= 0
               and coalesce((new.visual_description ->> 'output_tokens')::integer, -1) >= 0 then
                return new;
            end if;
        end if;
    end if;

    if to_jsonb(new) is distinct from to_jsonb(old) then
        raise exception 'Ingestion provenance records are immutable'
            using errcode = '55000';
    end if;
    return new;
end;
$$;

create or replace function public.persist_admitted_image_vlm_description(
    p_ingestion_run_id text,
    p_page_number integer,
    p_block_id text,
    p_image_sha256 text,
    p_description jsonb
)
returns void
language plpgsql
security definer
set search_path = pg_catalog, public, pg_temp
as $$
declare
    current_row public.image_screening_results%rowtype;
begin
    if p_description is null
       or jsonb_typeof(p_description) <> 'object'
       or p_description ->> 'description_text' is null
       or p_description ->> 'provider' is null
       or p_description ->> 'provider' = ''
       or p_description ->> 'model' is null
       or p_description ->> 'model' = ''
       or p_description ->> 'prompt_version' is null
       or p_description ->> 'prompt_version' = ''
       or p_description ->> 'description_sha256' is null
       or (p_description ->> 'description_sha256') !~ '^[0-9a-f]{64}$'
       or p_image_sha256 is null
       or p_image_sha256 !~ '^[0-9a-f]{64}$'
       or encode(sha256(convert_to(p_description ->> 'description_text', 'UTF8')), 'hex')
            <> p_description ->> 'description_sha256'
       or coalesce((p_description ->> 'input_tokens')::integer, -1) < 0
       or coalesce((p_description ->> 'output_tokens')::integer, -1) < 0
       or p_description ->> 'decision' is null
       or p_description ->> 'decision' not in ('ADMITTED', 'ADMITTED_FLAGGED') then
        raise exception 'Visual description failed provenance validation';
    end if;

    select * into current_row
    from public.image_screening_results
    where ingestion_run_id = p_ingestion_run_id
      and page_number = p_page_number
      and block_id = p_block_id
    for update;

    if not found or current_row.image_sha256 <> p_image_sha256
       or not (
           (current_row.ocr_status = 'NO_TEXT' and current_row.decision is null)
           or (current_row.ocr_status = 'TEXT_FOUND'
               and current_row.decision in ('ADMITTED', 'ADMITTED_FLAGGED'))
       ) then
        raise exception 'Visual description does not match an eligible screened image';
    end if;

    if current_row.visual_description is null then
        update public.image_screening_results
        set visual_description = p_description
        where ingestion_run_id = p_ingestion_run_id
          and page_number = p_page_number
          and block_id = p_block_id;
    elsif current_row.visual_description <> p_description then
        raise exception 'Visual description conflicts with immutable stored content'
            using errcode = '55000';
    end if;
end;
$$;

revoke all on function public.persist_admitted_image_vlm_description(text, integer, text, text, jsonb)
    from public, anon, authenticated, service_role;

create or replace function public.persist_ingestion_screening_with_admitted_ocr_and_vlm(
    p_payload jsonb,
    p_admitted_ocr jsonb,
    p_admitted_vlm jsonb
)
returns void
language plpgsql
security definer
set search_path = pg_catalog, public, pg_temp
as $$
declare
    item jsonb;
    payload_run_id text;
begin
    if p_admitted_vlm is null or jsonb_typeof(p_admitted_vlm) <> 'array' then
        raise exception 'Admitted VLM array is required';
    end if;
    if p_payload is null
       or jsonb_typeof(p_payload -> 'run') <> 'object'
       or p_payload -> 'run' ->> 'ingestion_run_id' is null then
        raise exception 'Base payload is missing its ingestion run ID';
    end if;
    payload_run_id := p_payload -> 'run' ->> 'ingestion_run_id';
    for item in select value from jsonb_array_elements(p_admitted_vlm) loop
        if jsonb_typeof(item) <> 'object'
           or item ->> 'ingestion_run_id' is distinct from payload_run_id then
            raise exception 'VLM result ingestion run does not match base payload';
        end if;
    end loop;

    perform public.persist_ingestion_screening_with_admitted_ocr(
        p_payload, p_admitted_ocr
    );
    for item in select value from jsonb_array_elements(p_admitted_vlm) loop
        perform public.persist_admitted_image_vlm_description(
            item ->> 'ingestion_run_id',
            (item ->> 'page_number')::integer,
            item ->> 'block_id',
            item ->> 'image_sha256',
            item
        );
    end loop;
end;
$$;

revoke all on function public.persist_ingestion_screening_with_admitted_ocr_and_vlm(jsonb, jsonb, jsonb)
    from public, anon, authenticated;
grant execute on function public.persist_ingestion_screening_with_admitted_ocr_and_vlm(jsonb, jsonb, jsonb)
    to service_role;

comment on column public.image_screening_results.visual_description is
    'Write-once, screened AI visual description and model provenance for the existing image composite key.';

-- ============================================================================
-- Source: 202611160001_document_chat.sql
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS legal_chat;
REVOKE ALL ON SCHEMA legal_chat FROM PUBLIC, anon, authenticated, service_role;

CREATE TABLE IF NOT EXISTS legal_chat.conversations (
    id uuid PRIMARY KEY,
    title text NOT NULL CHECK (length(btrim(title)) BETWEEN 1 AND 160),
    document_scope jsonb NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(document_scope) = 'array'),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE legal_chat.conversations ENABLE ROW LEVEL SECURITY;
CREATE INDEX IF NOT EXISTS conversations_updated_at_idx
    ON legal_chat.conversations (updated_at DESC, id DESC);
-- LangGraph checkpoint layout for langgraph-checkpoint-postgres 3.2.1.
CREATE TABLE IF NOT EXISTS legal_chat.checkpoint_migrations (v integer PRIMARY KEY);
CREATE TABLE IF NOT EXISTS legal_chat.checkpoints (
    thread_id text NOT NULL, checkpoint_ns text NOT NULL DEFAULT '',
    checkpoint_id text NOT NULL, parent_checkpoint_id text, type text,
    checkpoint jsonb NOT NULL, metadata jsonb NOT NULL DEFAULT '{}',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);
CREATE TABLE IF NOT EXISTS legal_chat.checkpoint_blobs (
    thread_id text NOT NULL, checkpoint_ns text NOT NULL DEFAULT '',
    channel text NOT NULL, version text NOT NULL, type text NOT NULL, blob bytea,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);
CREATE TABLE IF NOT EXISTS legal_chat.checkpoint_writes (
    thread_id text NOT NULL, checkpoint_ns text NOT NULL DEFAULT '',
    checkpoint_id text NOT NULL, task_id text NOT NULL, idx integer NOT NULL,
    channel text NOT NULL, type text, blob bytea NOT NULL, task_path text NOT NULL DEFAULT '',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);
ALTER TABLE legal_chat.checkpoint_blobs ALTER COLUMN blob DROP NOT NULL;
ALTER TABLE legal_chat.checkpoint_writes ADD COLUMN IF NOT EXISTS task_path text NOT NULL DEFAULT '';
CREATE INDEX IF NOT EXISTS checkpoints_thread_id_idx ON legal_chat.checkpoints(thread_id);
CREATE INDEX IF NOT EXISTS checkpoint_blobs_thread_id_idx ON legal_chat.checkpoint_blobs(thread_id);
CREATE INDEX IF NOT EXISTS checkpoint_writes_thread_id_idx ON legal_chat.checkpoint_writes(thread_id);
INSERT INTO legal_chat.checkpoint_migrations(v)
VALUES (0), (1), (2), (3), (4), (5), (6), (7), (8), (9)
ON CONFLICT (v) DO NOTHING;
ALTER TABLE legal_chat.checkpoint_migrations ENABLE ROW LEVEL SECURITY;
ALTER TABLE legal_chat.checkpoints ENABLE ROW LEVEL SECURITY;
ALTER TABLE legal_chat.checkpoint_blobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE legal_chat.checkpoint_writes ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON ALL TABLES IN SCHEMA legal_chat FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA legal_chat FROM PUBLIC, anon, authenticated, service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA legal_chat
    REVOKE ALL ON TABLES FROM PUBLIC, anon, authenticated, service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA legal_chat
    REVOKE ALL ON SEQUENCES FROM PUBLIC, anon, authenticated, service_role;
