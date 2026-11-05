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
