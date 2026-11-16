import asyncio
import hashlib
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from src.graph.corpus_reader import CorpusChunk, CorpusDocument
from src.graph.visual_observations import (
    VisualEntityProposal,
    VisualGraphProposal,
    VisualRelationshipProposal,
    _unique_quote_range,
    build_visual_observation,
    extract_document_visual_observations,
)
from src.ingestion.chunk_contracts import ChunkSource
from src.ingestion.parser_contracts import SourceCoordinates, canonical_visual_block_id
from src.ingestion.provenance.image_asset_resolver import ResolvedImageAsset


RUN = "run-visual"
SOURCE_SHA = "a" * 64
IMAGE_SHA = "b" * 64
DESCRIPTION = "Diagram: Bank connects to Core System."
DESCRIPTION_SHA = hashlib.sha256(DESCRIPTION.encode()).hexdigest()
IMAGE_BLOCK = "page-2-block-3"
PROMPT = "visual-description-v1"


def _source(method="IMAGE_VLM"):
    block_id = canonical_visual_block_id(
        2, IMAGE_BLOCK, "openai", "gpt-4o-mini", PROMPT, DESCRIPTION_SHA
    )
    values = dict(
        page_number=2, block_id=block_id, block_reading_order=3,
        block_derived_order=2, block_type="IMAGE", source_text_length=len(DESCRIPTION),
        start_offset=0, end_offset=len(DESCRIPTION), extraction_method=method,
        source_image_block_id=IMAGE_BLOCK, source_image_sha256=IMAGE_SHA,
        visual_description_sha256=DESCRIPTION_SHA,
        visual_model_provider="openai", visual_model="gpt-4o-mini",
        visual_prompt_version=PROMPT,
    )
    return ChunkSource.model_validate(values)


def _document(source=None):
    source = source or _source()
    chunk = CorpusChunk(
        id="chunk-visual-1", chunk_index=0, original_chunk_indices=(0,),
        raw_text=DESCRIPTION, retrieval_text=DESCRIPTION,
        chunk_sha256=hashlib.sha256(DESCRIPTION.encode()).hexdigest(),
        sources=[source.model_dump(mode="json", exclude_defaults=True)],
        retrieval_text_edits=[], admission_decisions=["ADMITTED"],
    )
    document = CorpusDocument(
        source_sha256=SOURCE_SHA, canonical_document_digest="c" * 64,
        source_page_count=2, ingestion_run_id=RUN,
        ingestion_created_at=datetime.now(timezone.utc),
        chunking_policy_version="cpcr-v1", screening_policy_version="screen-v1",
        screening_policy_digest="d" * 64, chunks=(chunk,),
    )
    return document, chunk, source


def _asset():
    return ResolvedImageAsset(
        ingestion_run_id=RUN, page_number=2, block_id=IMAGE_BLOCK,
        image_sha256=IMAGE_SHA, bucket="ingestion-image-crops",
        object_path=f"{RUN}/2/crop.png", file_size_bytes=64,
        ocr_status="NO_TEXT", ocr_text_sha256=None, ocr_text_length=0,
        visual_description_sha256=DESCRIPTION_SHA,
        visual_description_text=DESCRIPTION,
        visual_description_provider="openai", visual_description_model="gpt-4o-mini",
        visual_prompt_version=PROMPT, decision=None,
        source_coordinates=SourceCoordinates(
            x0=1, y0=2, x1=30, y1=40, unit="PDF_POINT", origin="TOP_LEFT"
        ), reading_order=3,
    )


def _proposal():
    return VisualGraphProposal(
        entities=(
            VisualEntityProposal(ref="bank", category="ORGANIZATION", name="Bank",
                                 evidence_quote="Bank"),
            VisualEntityProposal(ref="core", category="APPLICATION", name="Core System",
                                 evidence_quote="Core System"),
        ),
        relationships=(VisualRelationshipProposal(
            source_ref="bank", target_ref="core", observed_relation="connects to",
            evidence_quote="Bank connects to Core System",
        ),),
    )


def test_visual_observation_preserves_image_vlm_provenance_and_is_unverified():
    document, chunk, source = _document()
    observation = build_visual_observation(
        document=document, chunk=chunk, source=source, asset=_asset(),
        proposal=_proposal(), extraction_provider="openai", extraction_model="extractor-v1",
    )
    assert observation.trust_status == "UNVERIFIED_AI_VISUAL_OBSERVATION"
    assert observation.source_sha256 == SOURCE_SHA
    assert observation.ingestion_run_id == RUN
    assert observation.page_number == 2 and observation.source_image_block_id == IMAGE_BLOCK
    assert observation.image_sha256 == IMAGE_SHA
    assert observation.vlm_provider == "openai" and observation.vlm_model == "gpt-4o-mini"
    assert observation.prompt_version == PROMPT
    assert observation.description_sha256 == DESCRIPTION_SHA
    assert observation.candidates[0].quote_start == DESCRIPTION.index("Bank")
    assert observation.relationships[0].quote_end == len(DESCRIPTION) - 1
    assert observation.candidates[0].id.startswith("visual-entity:")


def test_native_or_mismatched_image_provenance_cannot_enter_visual_observation_path():
    native_source = ChunkSource(
        page_number=2, block_id="page-2-block-1", block_reading_order=1,
        block_type="TEXT", source_text_length=len(DESCRIPTION), start_offset=0,
        end_offset=len(DESCRIPTION),
    )
    document, chunk, _ = _document(native_source)
    with pytest.raises(ValueError, match="only IMAGE_VLM"):
        build_visual_observation(
            document=document, chunk=chunk, source=native_source, asset=_asset(),
            proposal=_proposal(), extraction_provider="openai", extraction_model="x",
        )
    document, chunk, source = _document()
    wrong_asset = _asset().model_copy(update={"image_sha256": "e" * 64})
    with pytest.raises(ValueError, match="does not match private asset"):
        build_visual_observation(
            document=document, chunk=chunk, source=source, asset=wrong_asset,
            proposal=_proposal(), extraction_provider="openai", extraction_model="x",
        )


def test_candidate_quotes_must_be_exactly_present_in_vlm_chunk():
    document, chunk, source = _document()
    invalid = VisualGraphProposal(entities=(VisualEntityProposal(
        ref="x", category="COMPONENT", name="Unseen", evidence_quote="not visible"
    ),), relationships=())
    with pytest.raises(ValueError, match="not present"):
        build_visual_observation(
            document=document, chunk=chunk, source=source, asset=_asset(), proposal=invalid,
            extraction_provider="openai", extraction_model="x",
        )


def test_repeated_candidate_quote_is_rejected_as_ambiguous_for_offsets():
    with pytest.raises(ValueError, match="unambiguous source span"):
        _unique_quote_range("Bank connects to Bank", "Bank")


def test_document_extraction_skips_native_chunks_and_resolves_original_asset():
    document, chunk, source = _document()

    class Extractor:
        provider = "openai"
        model = "extractor-v1"

        async def extract(self, text):
            assert text == DESCRIPTION
            return _proposal()

    class Resolver:
        async def resolve(self, run_id, keys):
            assert run_id == RUN
            assert keys == ((2, IMAGE_BLOCK),)
            return {(2, IMAGE_BLOCK): _asset()}

    result = asyncio.run(extract_document_visual_observations(Extractor(), document, Resolver()))
    assert len(result) == 1 and result[0].source_chunk_id == chunk.id




def test_rejected_or_quarantined_vlm_chunks_never_reach_extractor():
    document, chunk, _ = _document()
    rejected_chunk = replace(chunk, admission_decisions=["QUARANTINED"])
    rejected_document = replace(document, chunks=(rejected_chunk,))

    class Extractor:
        provider = "openai"
        model = "extractor-v1"

        async def extract(self, text):
            raise AssertionError("Rejected visual content reached the model")

    class Resolver:
        async def resolve(self, run_id, keys):
            raise AssertionError("Rejected visual content triggered asset lookup")

    assert asyncio.run(extract_document_visual_observations(
        Extractor(), rejected_document, Resolver()
    )) == ()
