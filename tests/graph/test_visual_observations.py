import asyncio
import hashlib
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.graph.corpus_reader import CorpusChunk, CorpusDocument
from src.graph.visual_observations import (
    VisualEntityProposal,
    VisualGraphProposal,
    VisualRelationshipProposal,
    OpenAIVisualGraphExtractor,
    VisualGraphExtractionConfig,
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


def test_visual_extraction_drops_candidates_without_exact_description_evidence():
    document, _, _ = _document()
    invalid = VisualGraphProposal(
        entities=(
            VisualEntityProposal(ref="bank", category="ORGANIZATION", name="Bank",
                                 evidence_quote="Bank"),
            VisualEntityProposal(ref="invented", category="APPLICATION", name="Invented",
                                 evidence_quote="not in the description"),
        ),
        relationships=(VisualRelationshipProposal(
            source_ref="bank", target_ref="invented", observed_relation="owns",
            evidence_quote="Bank owns Invented",
        ),),
    )

    class Extractor:
        provider = "openai"
        model = "extractor-v1"
        async def extract(self, _description):
            return invalid

    class Resolver:
        async def resolve(self, _run_id, _keys):
            return {(2, IMAGE_BLOCK): _asset()}

    observations = asyncio.run(extract_document_visual_observations(
        Extractor(), document, Resolver()
    ))
    assert len(observations) == 1
    assert [item.name for item in observations[0].candidates] == ["Bank"]
    assert observations[0].relationships == ()


def test_openai_visual_extractor_drops_relationship_with_unknown_candidate_reference():
    class Completions:
        async def create(self, **_kwargs):
            content = '{"entities":[{"ref":"bank","category":"ORGANIZATION",' \
                      '"name":"Bank","evidence_quote":"Bank"}],"relationships":[' \
                      '{"source_ref":"bank","target_ref":"missing","observed_relation":"owns",' \
                      '"evidence_quote":"Bank owns missing"}]}'
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=content)
            )])

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    extractor = OpenAIVisualGraphExtractor(
        VisualGraphExtractionConfig(provider="openai", model="test-model"), client=client
    )
    proposal = asyncio.run(extractor.extract(DESCRIPTION))
    assert [item.ref for item in proposal.entities] == ["bank"]
    assert proposal.relationships == ()


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


def _visual_material(page: int, block_number: int, reading_order: int, text: str, fill: str):
    block = f"page-{page}-block-{block_number}"
    digest = hashlib.sha256(text.encode()).hexdigest()
    prompt = PROMPT
    source = ChunkSource(
        page_number=page,
        block_id=canonical_visual_block_id(page, block, "openai", "gpt-4o-mini", prompt, digest),
        block_reading_order=reading_order, block_derived_order=2, block_type="IMAGE",
        source_text_length=len(text), start_offset=0, end_offset=len(text),
        extraction_method="IMAGE_VLM", source_image_block_id=block,
        source_image_sha256=fill * 64, visual_description_sha256=digest,
        visual_model_provider="openai", visual_model="gpt-4o-mini",
        visual_prompt_version=prompt,
    )
    asset = ResolvedImageAsset(
        ingestion_run_id=RUN, page_number=page, block_id=block,
        image_sha256=fill * 64, bucket="ingestion-image-crops",
        object_path=f"{RUN}/{page}/{block}.png", file_size_bytes=64,
        ocr_status="NO_TEXT", ocr_text_sha256=None, ocr_text_length=0,
        visual_description_sha256=digest, visual_description_text=text,
        visual_description_provider="openai", visual_description_model="gpt-4o-mini",
        visual_prompt_version=prompt, decision=None,
        source_coordinates=SourceCoordinates(
            x0=1, y0=2, x1=30, y1=40, unit="PDF_POINT", origin="TOP_LEFT"
        ), reading_order=reading_order,
    )
    return source, asset


def _chunk_with_sources(chunk_id, text, sources, decisions=("ADMITTED",)):
    return CorpusChunk(
        id=chunk_id, chunk_index=0, original_chunk_indices=(0,), raw_text=text,
        retrieval_text=text, chunk_sha256=hashlib.sha256(text.encode()).hexdigest(),
        sources=[source.model_dump(mode="json", exclude_defaults=True) for source in sources],
        retrieval_text_edits=[], admission_decisions=list(decisions),
    )


def test_native_text_and_visual_source_in_one_chunk_extract_only_the_image_span():
    prefix = "Text before the figure. "
    visual, asset = _visual_material(2, 3, 3, "Bank connects to Core", "d")
    native = ChunkSource(
        page_number=2, block_id="page-2-block-1", block_reading_order=1,
        block_type="TEXT", source_text_length=len(prefix), start_offset=0, end_offset=len(prefix),
    )
    combined = prefix + "Bank connects to Core"
    chunk = _chunk_with_sources("mixed-1", combined, (native, visual))
    document, _, _ = _document()
    document = replace(document, chunks=(chunk,))

    class Extractor:
        provider = "openai"
        model = "extractor-v1"
        async def extract(self, text):
            assert text == "Bank connects to Core"
            return VisualGraphProposal(entities=(VisualEntityProposal(
                ref="bank", category="ORGANIZATION", name="Bank", evidence_quote="Bank"
            ),), relationships=())

    class Resolver:
        async def resolve(self, run_id, keys):
            assert keys == ((2, "page-2-block-3"),)
            return {(2, "page-2-block-3"): asset}

    result = asyncio.run(extract_document_visual_observations(Extractor(), document, Resolver()))
    assert len(result) == 1 and result[0].source_image_block_id == "page-2-block-3"
    assert document.chunks[0].raw_text == combined


def test_multiple_visual_sources_resolve_each_page_and_block_independently():
    first_text, second_text = "Bank feeds Portal", "Portal sends Report"
    first, first_asset = _visual_material(1, 2, 2, first_text, "d")
    second, second_asset = _visual_material(3, 4, 4, second_text, "e")
    chunk = _chunk_with_sources("two-images", first_text + "\n" + second_text, (first, second))
    document, _, _ = _document()
    document = replace(document, chunks=(chunk,))

    class Extractor:
        provider = "openai"
        model = "extractor-v1"
        async def extract(self, text):
            return VisualGraphProposal(entities=(VisualEntityProposal(
                ref="entity", category="COMPONENT", name="source entity", evidence_quote=text
            ),), relationships=())

    class Resolver:
        async def resolve(self, run_id, keys):
            assert keys == ((1, "page-1-block-2"), (3, "page-3-block-4"))
            return {(1, "page-1-block-2"): first_asset,
                    (3, "page-3-block-4"): second_asset}

    observations = asyncio.run(extract_document_visual_observations(
        Extractor(), document, Resolver()
    ))
    assert [(item.page_number, item.source_image_block_id) for item in observations] == [
        (1, "page-1-block-2"), (3, "page-3-block-4")
    ]
    assert [item.description_text for item in observations] == [first_text, second_text]


def test_repeated_description_in_combined_chunk_is_rejected_before_model_call():
    visual, asset = _visual_material(2, 3, 3, "Same visual caption", "d")
    raw = "Same visual caption\nSame visual caption"
    chunk = _chunk_with_sources("ambiguous", raw, (visual,))
    document, _, _ = _document()
    document = replace(document, chunks=(chunk,))

    class Extractor:
        provider = "openai"
        model = "extractor-v1"
        async def extract(self, text):
            raise AssertionError("Ambiguous provenance must fail before extraction")

    class Resolver:
        async def resolve(self, run_id, keys):
            return {(2, "page-2-block-3"): asset}

    with pytest.raises(ValueError, match="unambiguous"):
        asyncio.run(extract_document_visual_observations(Extractor(), document, Resolver()))


@pytest.mark.parametrize("decisions", [("REJECTED",), ("QUARANTINED",), ("ADMITTED", "REJECTED")])
def test_any_non_admitted_visual_chunk_decision_excludes_all_sources(decisions):
    visual, _ = _visual_material(2, 3, 3, "Bank connects to Core", "d")
    chunk = _chunk_with_sources("blocked", "Bank connects to Core", (visual,), decisions)
    document, _, _ = _document()
    document = replace(document, chunks=(chunk,))

    class NeverCalled:
        provider = "openai"
        model = "extractor-v1"
        async def extract(self, text):
            raise AssertionError("Non-admitted content reached extraction")

    class Resolver:
        async def resolve(self, run_id, keys):
            raise AssertionError("Non-admitted content resolved an asset")

    assert asyncio.run(extract_document_visual_observations(
        NeverCalled(), document, Resolver()
    )) == ()



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
