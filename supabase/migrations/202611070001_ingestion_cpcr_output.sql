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
