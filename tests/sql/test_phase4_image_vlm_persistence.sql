begin;

-- Verify the public RPC is the only VLM write entry point exposed to service_role.
do $$
declare
    wrapper_owner name;
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
    if has_function_privilege('service_role',
       'public.persist_admitted_image_vlm_description(text,integer,text,text,jsonb)', 'EXECUTE') then
        raise exception 'service_role can bypass the VLM wrapper';
    end if;
    select pg_get_userbyid(p.proowner)::name into wrapper_owner
    from pg_proc p where p.oid =
        'public.persist_ingestion_screening_with_admitted_ocr_and_vlm(jsonb,jsonb,jsonb)'::regprocedure;
    if has_function_privilege(wrapper_owner,
       'public.persist_ingestion_screening_with_admitted_ocr(jsonb,jsonb)', 'EXECUTE') is not true
       or has_function_privilege(wrapper_owner,
       'public.persist_admitted_image_vlm_description(text,integer,text,text,jsonb)', 'EXECUTE') is not true then
        raise exception 'VLM wrapper owner lacks an internal helper permission';
    end if;
end;
$$;

-- Invoker helper lets the test call the public API with current_user=service_role,
-- while catching expected failures in a subtransaction for rollback assertions.
create function pg_temp.exercise_vlm_rpc(
    p_run_id text,
    p_source_sha text,
    p_vlm_run_id text,
    p_vlm_image_sha text,
    p_expected_error text default null
) returns void language plpgsql as $$
declare
    payload jsonb;
    vlm jsonb;
    description_text text := 'AI-generated visual description (not verified source text). Shows an arrow from Start to Finish.';
    description_sha text := encode(sha256(convert_to(
        'AI-generated visual description (not verified source text). Shows an arrow from Start to Finish.', 'UTF8'
    )), 'hex');
    actual_error text;
begin
    if current_user <> 'service_role' then
        raise exception 'RPC test was not executed as service_role';
    end if;
    payload := jsonb_build_object(
        'document', jsonb_build_object(
            'source_id', 'phase4-vlm-test-' || p_run_id,
            'source_sha256', p_source_sha,
            'canonical_document_digest', repeat('f', 64),
            'source_page_count', 1
        ),
        'run', jsonb_build_object(
            'ingestion_run_id', p_run_id,
            'chunking_policy_version', '5c-1.6.0+0123456789ab',
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
            'image_object_path', p_run_id || '/vlm.png', 'image_file_size_bytes', 8
        ))
    );
    vlm := jsonb_build_array(jsonb_build_object(
        'ingestion_run_id', p_vlm_run_id, 'page_number', 1,
        'block_id', 'page-1-block-7', 'image_sha256', p_vlm_image_sha,
        'description', jsonb_build_object('visual_type', 'diagram'),
        'description_text', description_text, 'description_sha256', description_sha,
        'decision', 'ADMITTED', 'provider', 'openai', 'model', 'test-vlm',
        'prompt_version', 'visual-description-v1', 'input_tokens', 20, 'output_tokens', 30
    ));
    begin
        perform public.persist_ingestion_screening_with_admitted_ocr_and_vlm(
            payload, '[]'::jsonb, vlm
        );
    exception when others then
        get stacked diagnostics actual_error = message_text;
    end;
    if p_expected_error is null and actual_error is not null then
        raise exception 'Unexpected VLM RPC failure: %', actual_error;
    elsif p_expected_error is not null and
          (actual_error is null or position(p_expected_error in actual_error) = 0) then
        raise exception 'Expected VLM error containing %, got %', p_expected_error, actual_error;
    end if;
end;
$$;

set local role service_role;
select pg_temp.exercise_vlm_rpc(
    'phase4-vlm-success-test', repeat('e', 64),
    'phase4-vlm-success-test', repeat('d', 64), null
);
-- Exact replay must be idempotent.
select pg_temp.exercise_vlm_rpc(
    'phase4-vlm-success-test', repeat('e', 64),
    'phase4-vlm-success-test', repeat('d', 64), null
);
-- A result from a different run is rejected before the base write.
select pg_temp.exercise_vlm_rpc(
    'phase4-vlm-run-mismatch-test', repeat('a', 64),
    'phase4-vlm-other-run', repeat('d', 64),
    'VLM result ingestion run does not match base payload'
);
-- A VLM image mismatch happens after the base RPC; verify the full call rolls back.
select pg_temp.exercise_vlm_rpc(
    'phase4-vlm-rollback-test', repeat('b', 64),
    'phase4-vlm-rollback-test', repeat('9', 64),
    'Visual description does not match an eligible screened image'
);
reset role;

do $$
declare
    run_id text := 'phase4-vlm-success-test';
    count_rows integer;
begin
    select count(*) into count_rows from public.image_screening_results
    where ingestion_run_id = run_id and page_number = 1
      and block_id = 'page-1-block-7'
      and visual_description ->> 'prompt_version' = 'visual-description-v1';
    if count_rows <> 1 then raise exception 'VLM success/replay did not persist exactly once'; end if;

    select count(*) into count_rows from public.ingestion_runs
    where ingestion_run_id = 'phase4-vlm-run-mismatch-test';
    if count_rows <> 0 then raise exception 'Run-mismatched VLM call partially persisted base run'; end if;

    select count(*) into count_rows from public.ingestion_runs
    where ingestion_run_id = 'phase4-vlm-rollback-test';
    if count_rows <> 0 then raise exception 'Failed VLM write did not roll back ingestion run'; end if;
    select count(*) into count_rows from public.ingestion_documents
    where source_sha256 = repeat('b', 64);
    if count_rows <> 0 then raise exception 'Failed VLM write did not roll back document'; end if;
    select count(*) into count_rows from public.ingestion_chunks
    where ingestion_run_id = 'phase4-vlm-rollback-test';
    if count_rows <> 0 then raise exception 'Failed VLM write did not roll back chunks'; end if;
    select count(*) into count_rows from public.chunk_screening_results
    where ingestion_run_id = 'phase4-vlm-rollback-test';
    if count_rows <> 0 then raise exception 'Failed VLM write did not roll back chunk screening'; end if;
    select count(*) into count_rows from public.image_screening_results
    where ingestion_run_id = 'phase4-vlm-rollback-test';
    if count_rows <> 0 then raise exception 'Failed VLM write did not roll back image metadata'; end if;
end;
$$;

rollback;
