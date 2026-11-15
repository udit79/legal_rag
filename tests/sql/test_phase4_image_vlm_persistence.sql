-- Requires migration 202611150001 on a disposable Supabase/PostgreSQL project.
begin;

do $$
declare
    v_source_sha text := repeat('e', 64);
    v_run_id text := 'phase4-vlm-trigger-integration-test';
    v_description text := 'AI-generated visual description (not verified source text). Shows an arrow from Start to Finish.';
    v_description_sha text := encode(sha256(convert_to(
        'AI-generated visual description (not verified source text). Shows an arrow from Start to Finish.', 'UTF8'
    )), 'hex');
    v_payload jsonb;
    v_vlm jsonb;
    v_failed boolean;
    v_count integer;
begin
    if has_function_privilege('service_role',
       'public.persist_ingestion_screening_with_admitted_ocr_and_vlm(jsonb,jsonb,jsonb)', 'EXECUTE') is not true then
        raise exception 'service_role cannot execute VLM persistence RPC';
    end if;
    if has_function_privilege('anon',
       'public.persist_ingestion_screening_with_admitted_ocr_and_vlm(jsonb,jsonb,jsonb)', 'EXECUTE')
       or has_function_privilege('authenticated',
       'public.persist_ingestion_screening_with_admitted_ocr_and_vlm(jsonb,jsonb,jsonb)', 'EXECUTE') then
        raise exception 'VLM persistence RPC is exposed to an untrusted role';
    end if;

    v_payload := jsonb_build_object(
        'document', jsonb_build_object(
            'source_id', 'phase4-vlm-test-source', 'source_sha256', v_source_sha,
            'canonical_document_digest', repeat('f', 64), 'source_page_count', 1
        ),
        'run', jsonb_build_object(
            'ingestion_run_id', v_run_id, 'chunking_policy_version', '5c-1.6.0+0123456789ab',
            'screening_policy_version', 'test-screening-v1',
            'screening_policy_digest', repeat('c', 64), 'trust_tier', 'T3',
            'document_quarantined', false, 'file_screening', null
        ),
        'chunks', '[]'::jsonb,
        'screening_results', '[]'::jsonb,
        'image_results', jsonb_build_array(jsonb_build_object(
            'page_number', 1, 'block_id', 'page-1-block-7', 'reading_order', 7,
            'image_sha256', repeat('d', 64), 'ocr_status', 'NO_TEXT',
            'ocr_text_sha256', null, 'ocr_text_length', 0,
            'decision', null, 'source_coordinates', jsonb_build_object(),
            'flags', '[]'::jsonb, 'image_bucket', 'ingestion-image-crops',
            'image_object_path', 'phase4-test/vlm.png', 'image_file_size_bytes', 8
        ))
    );
    v_vlm := jsonb_build_array(jsonb_build_object(
        'ingestion_run_id', v_run_id, 'page_number', 1,
        'block_id', 'page-1-block-7', 'image_sha256', repeat('d', 64),
        'description', jsonb_build_object('visual_type', 'diagram'),
        'description_text', v_description, 'description_sha256', v_description_sha,
        'decision', 'ADMITTED', 'provider', 'openai', 'model', 'test-vlm',
        'prompt_version', 'visual-description-v1', 'input_tokens', 20, 'output_tokens', 30
    ));

    perform public.persist_ingestion_screening_with_admitted_ocr_and_vlm(
        v_payload, '[]'::jsonb, v_vlm
    );
    perform public.persist_ingestion_screening_with_admitted_ocr_and_vlm(
        v_payload, '[]'::jsonb, v_vlm
    );
    select count(*) into v_count from public.image_screening_results
    where ingestion_run_id = v_run_id and page_number = 1
      and block_id = 'page-1-block-7'
      and visual_description ->> 'description_sha256' = v_description_sha;
    if v_count <> 1 then raise exception 'VLM replay duplicated or lost the description'; end if;

    v_failed := false;
    begin
        perform public.persist_ingestion_screening_with_admitted_ocr_and_vlm(
            v_payload, '[]'::jsonb,
            jsonb_set(v_vlm, '{0,description_text}', to_jsonb('conflicting description'::text))
        );
    exception when others then v_failed := true;
    end;
    if not v_failed then raise exception 'Conflicting VLM output unexpectedly persisted'; end if;
end;
$$;

rollback;
