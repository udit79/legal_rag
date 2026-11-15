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

    if tg_table_name = 'image_screening_results'
       and (to_jsonb(new) - 'ocr_text') = (to_jsonb(old) - 'ocr_text')
       and old.ocr_text is null
       and new.ocr_text is not null
       and new.ocr_status = 'TEXT_FOUND'
       and new.decision in ('ADMITTED', 'ADMITTED_FLAGGED')
       and char_length(new.ocr_text) = new.ocr_text_length
       and encode(sha256(convert_to(new.ocr_text, 'UTF8')), 'hex') = new.ocr_text_sha256 then
        return new;
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

revoke all on function public.persist_admitted_image_ocr(text, integer, text, text, text, integer)
    from public, anon, authenticated;
grant execute on function public.persist_admitted_image_ocr(text, integer, text, text, text, integer)
    to service_role;
