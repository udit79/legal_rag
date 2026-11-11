alter table public.image_screening_results
    add column if not exists image_bucket text,
    add column if not exists image_object_path text,
    add column if not exists image_file_size_bytes bigint;

alter table public.image_screening_results
    add constraint image_screening_storage_metadata_complete
    check (
        (
            image_bucket is null
            and image_object_path is null
            and image_file_size_bytes is null
        )
        or (
            image_bucket = 'ingestion-image-crops'
            and image_object_path is not null
            and length(image_object_path) > 0
            and image_file_size_bytes is not null
            and image_file_size_bytes >= 0
        )
    );

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
