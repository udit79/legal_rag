-- Run only against a disposable local Supabase/PostgreSQL database after all
-- migrations are applied. The outer transaction rolls back every fixture row.
begin;

create function pg_temp.expect_immutable_error(p_statement text)
returns void
language plpgsql
as $$
begin
    execute p_statement;
    raise exception 'Expected immutability trigger rejection for: %', p_statement;
exception when sqlstate '55000' then
    return;
end;
$$;

do $$
declare
    v_source_sha text := repeat('a', 64);
    v_run_id text := 'phase4-ocr-trigger-integration-test';
    v_policy text := 'phase4-trigger-test-v1';
    v_text text := 'OCR text admitted from a scanned figure';
    v_chunk_text text := 'Source-backed chunk for OCR integration test';
    v_ocr_sha text := encode(sha256(convert_to('OCR text admitted from a scanned figure', 'UTF8')), 'hex');
    v_chunk_sha text := encode(sha256(convert_to('Source-backed chunk for OCR integration test', 'UTF8')), 'hex');
    v_payload jsonb;
    v_ocr jsonb;
    v_failed boolean;
    v_count integer;
begin
    if has_function_privilege('service_role',
        'public.persist_ingestion_screening_with_admitted_ocr(jsonb,jsonb)', 'EXECUTE') is not true then
        raise exception 'service_role cannot execute the atomic persistence RPC';
    end if;
    if has_function_privilege('anon',
        'public.persist_ingestion_screening_with_admitted_ocr(jsonb,jsonb)', 'EXECUTE')
       or has_function_privilege('authenticated',
        'public.persist_ingestion_screening_with_admitted_ocr(jsonb,jsonb)', 'EXECUTE') then
        raise exception 'Atomic persistence RPC is exposed to an untrusted role';
    end if;

    v_payload := jsonb_build_object(
        'document', jsonb_build_object(
            'source_id', 'phase4-test-source', 'source_sha256', v_source_sha,
            'canonical_document_digest', repeat('b', 64), 'source_page_count', 1
        ),
        'run', jsonb_build_object(
            'ingestion_run_id', v_run_id, 'chunking_policy_version', v_policy,
            'screening_policy_version', 'test-screening-v1',
            'screening_policy_digest', repeat('c', 64), 'trust_tier', 'T3',
            'document_quarantined', false, 'file_screening', null
        ),
        'chunks', jsonb_build_array(jsonb_build_object(
            'chunk_index', 0, 'chunk_digest', v_chunk_sha,
            'text', v_chunk_text, 'sources', jsonb_build_array()
        )),
        'screening_results', jsonb_build_array(jsonb_build_object(
            'chunk_index', 0, 'chunk_digest', v_chunk_sha, 'decision', 'ADMITTED',
            'flags', jsonb_build_array(), 'justification_codes', jsonb_build_array(),
            'result', jsonb_build_object('decision', 'ADMITTED')
        )),
        'image_results', jsonb_build_array(jsonb_build_object(
            'page_number', 1, 'block_id', 'page-1-block-7', 'reading_order', 7,
            'image_sha256', repeat('d', 64), 'ocr_status', 'TEXT_FOUND',
            'ocr_text_sha256', v_ocr_sha, 'ocr_text_length', char_length(v_text),
            'decision', 'ADMITTED', 'source_coordinates', jsonb_build_object(),
            'flags', jsonb_build_array(), 'image_bucket', 'ingestion-image-crops',
            'image_object_path', 'phase4-test/image.png', 'image_file_size_bytes', 8
        ))
    );
    v_ocr := jsonb_build_array(jsonb_build_object(
        'ingestion_run_id', v_run_id, 'page_number', 1,
        'block_id', 'page-1-block-7', 'ocr_text', v_text,
        'ocr_text_sha256', v_ocr_sha, 'ocr_text_length', char_length(v_text)
    ));

    -- Simulate failure after the component screening write has started. The
    -- wrapper's statement transaction must roll back all of that write.
    v_failed := false;
    begin
        perform public.persist_ingestion_screening_with_admitted_ocr(
            v_payload,
            jsonb_set(v_ocr, '{0,ocr_text_sha256}', to_jsonb(repeat('0', 64)))
        );
    exception when others then
        v_failed := true;
    end;
    if not v_failed then raise exception 'Invalid OCR unexpectedly persisted'; end if;
    select count(*) into v_count from public.ingestion_runs where ingestion_run_id = v_run_id;
    if v_count <> 0 then raise exception 'Failed atomic OCR write left ingestion run rows'; end if;
    select count(*) into v_count from public.ingestion_chunks where source_sha256 = v_source_sha;
    if v_count <> 0 then raise exception 'Failed atomic OCR write left chunk rows'; end if;

    -- Initial valid write performs the one permitted NULL -> text update.
    perform public.persist_ingestion_screening_with_admitted_ocr(v_payload, v_ocr);
    -- Exact retry/replay is idempotent and does not duplicate any records.
    perform public.persist_ingestion_screening_with_admitted_ocr(v_payload, v_ocr);
    select count(*) into v_count from public.ingestion_runs where ingestion_run_id = v_run_id;
    if v_count <> 1 then raise exception 'Replay duplicated ingestion run'; end if;
    select count(*) into v_count from public.ingestion_chunks where source_sha256 = v_source_sha;
    if v_count <> 1 then raise exception 'Replay duplicated ingestion chunk'; end if;
    select count(*) into v_count from public.chunk_screening_results where ingestion_run_id = v_run_id;
    if v_count <> 1 then raise exception 'Replay duplicated chunk screening row'; end if;
    select count(*) into v_count from public.image_screening_results
      where ingestion_run_id = v_run_id and page_number = 1 and block_id = 'page-1-block-7'
        and ocr_text = v_text and reading_order = 7;
    if v_count <> 1 then raise exception 'OCR text or original crop metadata was not persisted'; end if;

    -- A conflicting text/digest at the same composite key must fail closed.
    v_failed := false;
    begin
        perform public.persist_admitted_image_ocr(
            v_run_id, 1, 'page-1-block-7', 'Different OCR text',
            encode(sha256(convert_to('Different OCR text', 'UTF8')), 'hex'),
            char_length('Different OCR text')
        );
    exception when others then v_failed := true;
    end;
    if not v_failed then raise exception 'Conflicting OCR text unexpectedly persisted'; end if;

    -- Every table attached to the shared trigger remains immutable.
    perform pg_temp.expect_immutable_error(format(
        'update public.image_screening_results set image_sha256 = %L where ingestion_run_id = %L and page_number = 1 and block_id = %L',
        repeat('e', 64), v_run_id, 'page-1-block-7'));
    perform pg_temp.expect_immutable_error(format(
        'update public.ingestion_runs set document_quarantined = true where ingestion_run_id = %L', v_run_id));
    perform pg_temp.expect_immutable_error(format(
        'update public.ingestion_chunks set chunk_text = %L where source_sha256 = %L', 'changed', v_source_sha));
    perform pg_temp.expect_immutable_error(format(
        'update public.chunk_screening_results set decision = %L where ingestion_run_id = %L', 'QUARANTINED', v_run_id));
    perform pg_temp.expect_immutable_error(format(
        'delete from public.image_screening_results where ingestion_run_id = %L', v_run_id));
    perform pg_temp.expect_immutable_error(format(
        'delete from public.ingestion_runs where ingestion_run_id = %L', v_run_id));
    perform pg_temp.expect_immutable_error(format(
        'delete from public.ingestion_chunks where source_sha256 = %L', v_source_sha));
    perform pg_temp.expect_immutable_error(format(
        'delete from public.chunk_screening_results where ingestion_run_id = %L', v_run_id));
end;
$$;

rollback;
