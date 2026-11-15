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
begin
    if p_admitted_vlm is null or jsonb_typeof(p_admitted_vlm) <> 'array' then
        raise exception 'Admitted VLM array is required';
    end if;
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
