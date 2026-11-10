import asyncio
from typing import Literal

from src.ingestion.chunking import ChunkingConfig, chunk_document
from src.ingestion.continuity_contracts import (
    ChunkReference,
    ContinuityCandidate,
    ContinuityRelation,
    deterministic_candidate_id,
)
from src.ingestion.continuity_cdm_contracts import CDMClassificationResult
from src.ingestion.continuity_gateway import (
    CONTINUITY_GATEWAY_POLICY_VERSION,
    GatewayDecision,
    GatewayResult,
)
from src.ingestion.contracts import DocumentFeatures, PageFeatures
from src.ingestion.parser_contracts import (
    CanonicalBlock,
    CanonicalDocument,
    CanonicalPage,
    SourceCoordinates,
)
from src.ingestion.pipeline import (
    CPCRCandidateResult,
    CPCRResult,
    PdfIngestionResult,
    classify_page_continuity,
    revise_chunks_after_cpcr,
)
from src.ingestion.router import route_document
from src.security.ingestion_screening.contracts import (
    DocumentContext,
    FileScreeningRecord,
)
from src.security.ingestion_screening.enums import AdmissionDecision, TrustTier
from src.security.ingestion_screening.policy import ScreeningPolicy
from src.security.ingestion_screening.screening_service import ScreeningService


def _coordinates() -> SourceCoordinates:
    return SourceCoordinates(
        x0=0,
        y0=0,
        x1=300,
        y1=100,
        unit="PDF_POINT",
        origin="TOP_LEFT",
    )


def _ingestion_and_screening(
    source_text: str,
    target_text: str,
    limit: int = 64,
    *,
    block_type: Literal["TEXT", "TABLE"] = "TEXT",
):
    pages = tuple(
        CanonicalPage(
            page_number=page_number,
            parser="TEST",
            status="PARSED",
            blocks=(
                CanonicalBlock(
                    page_number=page_number,
                    text=text,
                    block_type=block_type,
                    reading_order=0,
                    source_coordinates=_coordinates(),
                ),
            ),
        )
        for page_number, text in ((1, source_text), (2, target_text))
    )
    canonical = CanonicalDocument(
        page_count=2,
        document_disposition="PROCESS",
        pages=pages,
    )
    digest = "a" * 64
    config = ChunkingConfig(max_chunk_chars=limit, overlap_chars=16)
    chunking = chunk_document(canonical, digest, config)
    features = DocumentFeatures(
        content_sha256=digest,
        file_size_bytes=100,
        page_count=2,
        pages=tuple(
            PageFeatures(
                page_number=page_number,
                text_char_count=len(text),
                image_count=0,
                drawing_count=0,
            )
            for page_number, text in ((1, source_text), (2, target_text))
        ),
    )
    route_plan = route_document(features, "test-router-1")
    ingestion = PdfIngestionResult(
        document_id=digest,
        features=features,
        route_plan=route_plan,
        canonical=canonical,
        chunking=chunking,
        unrecognized_images=(),
    )
    context = DocumentContext(
        source_id=digest,
        ingestion_run_id="run-test",
        source_sha256=digest,
        canonical_document_digest="b" * 64,
        chunking_policy_version=chunking.document.policy_version,
        source_page_count=2,
        trust_tier=TrustTier.T3_UNAUTHENTICATED_EXTERNAL,
    )
    policy = ScreeningPolicy()
    file_screening = FileScreeningRecord(
        source_sha256=digest,
        file_size_bytes=100,
        decision=AdmissionDecision.ADMITTED,
        scanner="test-scanner",
        scanner_version="1",
        scanner_database_version="test-db-1",
        screening_policy_version=policy.version,
        screening_policy_digest=policy.digest(),
        justification_codes=("G0_CHECKS_PASSED",),
    )
    screened = asyncio.run(
        ScreeningService(policy).screen_batch(
            chunking.document.chunks,
            context,
            document_id=digest,
            file_screening=file_screening,
        )
    )
    return ingestion, screened, config


def _positive_candidate(ingestion: PdfIngestionResult) -> CPCRResult:
    document = ingestion.chunking.document
    source = next(
        chunk
        for chunk in reversed(document.chunks)
        if chunk.sources[0].page_number == 1
    )
    target = next(
        chunk for chunk in document.chunks if chunk.sources[0].page_number == 2
    )
    candidate = ContinuityCandidate(
        candidate_id=deterministic_candidate_id(
            document.document_id,
            document.policy_version,
            source.index,
            target.index,
        ),
        source=ChunkReference(
            document_id=document.document_id,
            chunk_index=source.index,
            page_number=1,
        ),
        target=ChunkReference(
            document_id=document.document_id,
            chunk_index=target.index,
            page_number=2,
        ),
        chunking_policy_version=document.policy_version,
        candidate_policy_version="cpcr-candidate-test",
    )
    gateway = GatewayResult(
        decision=GatewayDecision.ACCEPT,
        relation_type=ContinuityRelation.SAME_PARAGRAPH,
        evidence_codes=("TEST_CONTINUATION",),
        gateway_policy_version=CONTINUITY_GATEWAY_POLICY_VERSION,
    )
    return CPCRResult(
        outcome="COMPLETED",
        candidates=(
            CPCRCandidateResult(
                candidate=candidate,
                gateway_result=gateway,
                classification_status="NOT_REQUIRED",
            ),
        ),
    )


def test_cpcr_adds_source_backed_tail_overlap_without_merging_pages():
    ingestion, screened, config = _ingestion_and_screening(
        "The last line gives the needed context",
        "which continues into the next page.",
    )

    revised = revise_chunks_after_cpcr(
        ingestion, _positive_candidate(ingestion), screened, config
    )

    assert len(revised.chunks) == 2
    assert revised.chunks[0].text == ingestion.chunking.document.chunks[0].text
    continuation = revised.chunks[1]
    assert continuation.text.startswith("needed context which continues")
    assert continuation.sources[0].page_number == 1
    assert continuation.sources[1].page_number == 2
    assert continuation.sources[0].end_offset == len(
        ingestion.canonical.pages[0].blocks[0].text
    )
    assert len(continuation.text) <= config.max_chunk_chars
    assert len(revised.contextualized_candidate_ids) == 1
    assert revised.contextualized_candidate_ids[0] == _positive_candidate(
        ingestion
    ).candidates[0].candidate.candidate_id


def test_cpcr_context_overlap_respects_size_limit_and_splits_target_source_range():
    ingestion, screened, config = _ingestion_and_screening(
        "source ending", "ABCDEFGHIJKLMNOP", limit=16
    )

    revised = revise_chunks_after_cpcr(
        ingestion, _positive_candidate(ingestion), screened, config
    )

    assert all(len(chunk.text) <= config.max_chunk_chars for chunk in revised.chunks)
    contextualized = next(
        chunk for chunk in revised.chunks if len(chunk.sources) > 1
    )
    assert contextualized.text.startswith("ding ")
    assert contextualized.sources[0].page_number == 1
    assert contextualized.sources[1].page_number == 2


def test_cpcr_classification_failure_is_logged_and_returns_partial(caplog):
    ingestion, screened, _ = _ingestion_and_screening(
        "The last line gives the needed context",
        "which continues into the next page.",
    )

    class FailedClassificationClient:
        async def classify(self, request):
            del request
            raise RuntimeError("provider unavailable")

    result = asyncio.run(
        classify_page_continuity(ingestion, FailedClassificationClient(), screened)
    )

    assert result.outcome == "PARTIAL"
    assert result.candidates[0].classification_status == "FAILED"
    assert result.candidates[0].classification_error == "RuntimeError"
    assert "CPCR classification failed" in caplog.text


def test_cpcr_rejects_table_relation_for_text_chunks_without_failing_ingestion():
    ingestion, screened, config = _ingestion_and_screening(
        "The source page ends mid-thought",
        "and the target page continues the paragraph.",
    )

    class InvalidTableClassificationClient:
        async def classify(self, request):
            return CDMClassificationResult(
                candidate_id=request.candidate_id,
                relation_type=ContinuityRelation.TABLE_CONTINUATION,
                probabilities={
                    ContinuityRelation.SAME_PARAGRAPH: 0.1,
                    ContinuityRelation.TABLE_CONTINUATION: 0.7,
                    ContinuityRelation.NO_CONTINUATION: 0.1,
                    ContinuityRelation.UNKNOWN: 0.1,
                },
                model_id=request.model_id,
            )

    cpcr = asyncio.run(
        classify_page_continuity(
            ingestion, InvalidTableClassificationClient(), screened
        )
    )

    assert cpcr.outcome == "COMPLETED"
    assert cpcr.candidates[0].classification_status == "REJECTED"
    assert (
        cpcr.candidates[0].classification_error
        == "TABLE_CONTINUATION_REQUIRES_TABLE_CHUNKS"
    )
    assert cpcr.candidates[0].cdm_result is not None
    assert cpcr.candidates[0].cdm_result.relation_type == (
        ContinuityRelation.TABLE_CONTINUATION
    )
    revised = revise_chunks_after_cpcr(ingestion, cpcr, screened, config)
    assert revised.contextualized_candidate_ids == ()
    assert revised.chunks == ingestion.chunking.document.chunks


def test_cpcr_does_not_apply_relation_below_minimum_confidence():
    ingestion, screened, config = _ingestion_and_screening(
        "The source page ends mid-thought",
        "and the target page continues the paragraph.",
    )

    class LowConfidenceClassificationClient:
        async def classify(self, request):
            return CDMClassificationResult(
                candidate_id=request.candidate_id,
                relation_type=ContinuityRelation.SAME_PARAGRAPH,
                probabilities={
                    ContinuityRelation.SAME_PARAGRAPH: 0.74,
                    ContinuityRelation.TABLE_CONTINUATION: 0.1,
                    ContinuityRelation.NO_CONTINUATION: 0.1,
                    ContinuityRelation.UNKNOWN: 0.06,
                },
                model_id=request.model_id,
            )

    cpcr = asyncio.run(
        classify_page_continuity(
            ingestion, LowConfidenceClassificationClient(), screened
        )
    )

    assert cpcr.outcome == "COMPLETED"
    assert cpcr.candidates[0].classification_status == "REJECTED"
    assert cpcr.candidates[0].classification_error == "LOW_CONFIDENCE"
    assert cpcr.candidates[0].cdm_result is not None
    revised = revise_chunks_after_cpcr(ingestion, cpcr, screened, config)
    assert revised.contextualized_candidate_ids == ()


def test_cpcr_skips_table_only_boundaries_without_calling_classification():
    ingestion, screened, _ = _ingestion_and_screening(
        "| A | B |",
        "| C | D |",
        block_type="TABLE",
    )

    class UnexpectedClassificationClient:
        async def classify(self, request):
            raise AssertionError(f"Table flow was classified: {request.candidate_id}")

    result = asyncio.run(
        classify_page_continuity(
            ingestion,
            UnexpectedClassificationClient(),
            screened,
        )
    )

    assert result.outcome == "COMPLETED"
    assert result.candidates == ()
