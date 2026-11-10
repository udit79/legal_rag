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
