"""Frozen-set retrieval evaluation and JSON run artifacts."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Self, Sequence

from pydantic import (
    BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator,
)

from src.graph.retrieval import EvidenceBundle


class ImmutableMapping(Mapping[str, str]):
    """Read-only string mapping with a private, copied backing store."""

    __slots__ = ("__data",)

    def __init__(self, values: Mapping[str, str] | None = None) -> None:
        if values is None:
            values = {}
        if not isinstance(values, Mapping):
            raise TypeError("corpus_snapshot must be a mapping of strings to strings")

        copied = dict(values)
        if any(not isinstance(key, str) or not isinstance(value, str)
               for key, value in copied.items()):
            raise TypeError("corpus_snapshot keys and values must be strings")

        object.__setattr__(self, "_ImmutableMapping__data", MappingProxyType(copied))

    def __getitem__(self, key: str) -> str:
        return self.__data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.__data)

    def __len__(self) -> int:
        return len(self.__data)

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("corpus_snapshot is immutable")

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        # The mapping and its string keys/values are immutable; sharing is safe.
        memo[id(self)] = self
        return self

    def __reduce__(self):
        # Keep the mapping picklable without exposing its MappingProxyType.
        return (type(self), (dict(self.__data),))

    def __repr__(self) -> str:
        return f"ImmutableMapping({dict(self.__data)!r})"


class EvaluationEvidenceReference(BaseModel):
    """Gold quote and provenance for a single expected answer citation.

    ``cited_page`` is the exact canonical page expected in an answer citation,
    using the page numbering in ``ChunkSource.page_number``. ``annotated_page``
    preserves the original page label from the human annotation, which can be a
    printed page number that differs from the PDF source page.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cited_page: int = Field(ge=1)
    annotated_page: int | None = Field(default=None, ge=1)
    source_page_numbers: tuple[int, ...] = Field(min_length=1)
    quote: str = Field(min_length=1)
    chunk_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_provenance(self):
        if (not self.source_page_numbers
                or any(type(page) is not int or page < 1 for page in self.source_page_numbers)
                or len(set(self.source_page_numbers)) != len(self.source_page_numbers)
                or self.cited_page not in self.source_page_numbers
                or len(set(self.chunk_ids)) != len(self.chunk_ids)):
            raise ValueError(
                "Evidence reference cited page must belong to its unique source pages"
            )
        return self


class AnswerCitation(BaseModel):
    """A citation emitted with an externally generated answer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page: int = Field(ge=1)
    quote: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)


class EvaluationQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    relevant_chunk_ids: tuple[str, ...]
    expected_document_ids: tuple[str, ...]
    unanswerable: bool
    reference_answer: str | None = None
    evidence_references: tuple[EvaluationEvidenceReference, ...] = ()

    @model_validator(mode="after")
    def validate_labels(self):
        if len(set(self.relevant_chunk_ids)) != len(self.relevant_chunk_ids):
            raise ValueError("Relevant chunk IDs must be unique")
        if len(set(self.expected_document_ids)) != len(self.expected_document_ids):
            raise ValueError("Expected document IDs must be unique")
        if self.unanswerable and (self.relevant_chunk_ids or self.expected_document_ids):
            raise ValueError("Unanswerable queries cannot have relevant evidence labels")
        if not self.unanswerable and (not self.relevant_chunk_ids or not self.expected_document_ids):
            raise ValueError("Answerable queries require relevant chunk and document labels")
        if self.unanswerable and self.evidence_references:
            raise ValueError("Unanswerable queries cannot have evidence references")
        for reference in self.evidence_references:
            if (reference.document_sha256 not in self.expected_document_ids
                    or not set(reference.chunk_ids).issubset(self.relevant_chunk_ids)):
                raise ValueError("Evidence references must point to labeled source chunks")
        return self


class FrozenEvaluationDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    synthetic: bool
    annotation_source: str | None = None
    corpus_snapshot: Mapping[str, str] = Field(
        default_factory=ImmutableMapping, validate_default=True
    )
    queries: tuple[EvaluationQuery, ...] = Field(min_length=1)

    @field_validator("corpus_snapshot")
    @classmethod
    def freeze_corpus_snapshot(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        return ImmutableMapping(value)

    @field_serializer("corpus_snapshot")
    def serialize_corpus_snapshot(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        """Copy the model without allowing mutable corpus snapshots via update.

        Pydantic's ``model_copy(update=...)`` intentionally skips validation.
        Normalize this field explicitly so its runtime immutability guarantee
        survives that API as well.
        """
        safe_update = dict(update) if update is not None else None
        if safe_update is not None and "corpus_snapshot" in safe_update:
            safe_update["corpus_snapshot"] = ImmutableMapping(
                safe_update["corpus_snapshot"]
            )
        return super().model_copy(update=safe_update, deep=deep)

    @model_validator(mode="after")
    def validate_query_ids(self):
        identifiers = [query.query_id for query in self.queries]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Evaluation query IDs must be unique")
        if not self.synthetic:
            if not self.annotation_source or not self.corpus_snapshot:
                raise ValueError("Non-synthetic datasets require annotation and corpus provenance")
            if any(not re.fullmatch(r"[0-9a-f]{64}", source)
                   or not isinstance(run_id, str) or not run_id.strip()
                   for source, run_id in self.corpus_snapshot.items()):
                raise ValueError("Corpus snapshot must map source hashes to ingestion runs")
            expected = {doc for query in self.queries for doc in query.expected_document_ids}
            if not expected.issubset(self.corpus_snapshot):
                raise ValueError("Dataset labels refer to documents outside its corpus snapshot")
        return self


class AnswerCorrectnessEvaluator(Protocol):
    """Score an externally supplied answer; this harness generates no answers."""

    async def evaluate(self, example: EvaluationQuery, answer: str,
                       evidence: Sequence[EvidenceBundle]) -> bool: ...


def _citation_is_supported(
    citation: AnswerCitation,
    evidence_by_identity: Mapping[tuple[str, str], EvidenceBundle],
    references: Sequence[EvaluationEvidenceReference],
) -> bool:
    evidence = evidence_by_identity.get((citation.chunk_id, citation.document_sha256))
    if evidence is None:
        return False
    actual_source_pages = {source.page_number for source in evidence.sources}
    if not citation.quote or citation.quote not in evidence.raw_text:
        return False
    return any(
        reference.document_sha256 == citation.document_sha256
        and citation.page == reference.cited_page
        and citation.page in actual_source_pages
        and citation.page in reference.source_page_numbers
        and citation.chunk_id in reference.chunk_ids
        and citation.quote in reference.quote
        for reference in references
    )


async def run_evaluation(
    dataset: FrozenEvaluationDataset,
    retriever,
    *,
    k: int = 5,
    abstention_rrf_threshold: float,
    answers: Mapping[str, str] | None = None,
    answer_evaluator: AnswerCorrectnessEvaluator | None = None,
    citations: Mapping[str, Sequence[AnswerCitation | Mapping[str, object]]] | None = None,
) -> dict:
    if type(k) is not int or k < 1:
        raise ValueError("Evaluation k must be a positive integer")
    if (isinstance(abstention_rrf_threshold, bool)
            or not isinstance(abstention_rrf_threshold, (int, float))
            or not math.isfinite(abstention_rrf_threshold)
            or abstention_rrf_threshold < 0):
        raise ValueError("Abstention threshold must be finite and non-negative")
    if answer_evaluator is not None and answers is None:
        raise ValueError("Answers are required when an answer correctness evaluator is supplied")
    if answers is not None:
        query_ids = {example.query_id for example in dataset.queries}
        unexpected = set(answers) - query_ids
        if unexpected:
            raise ValueError(f"Answers contain unknown query IDs: {sorted(unexpected)}")
    if citations is not None:
        query_ids = {example.query_id for example in dataset.queries}
        unexpected = set(citations) - query_ids
        if unexpected:
            raise ValueError(f"Citations contain unknown query IDs: {sorted(unexpected)}")

    rows: list[dict] = []
    answerable_recalls: list[float] = []
    answerable_document_recalls: list[float] = []
    answerable_document_precisions: list[float] = []
    answerable_rrs: list[float] = []
    citation_scores: list[float] = []
    citation_grounding_scores: list[float] = []
    answerable_citation_coverage_count = 0
    answerable_query_count = sum(not example.unanswerable for example in dataset.queries)
    answerable_answer_evaluation_count = 0
    unanswerable_abstentions: list[float] = []
    correctness_scores: list[float] = []

    for example in dataset.queries:
        evidence = await retriever.search(example.query, limit=k)
        top = evidence[:k]
        found_chunks = {item.chunk_id for item in top}
        found_documents = {item.document_sha256 for item in top}
        relevant = set(example.relevant_chunk_ids)
        recall = None
        reciprocal_rank = None
        if not example.unanswerable:
            recall = len(found_chunks & relevant) / len(relevant) if relevant else 0.0
            reciprocal_rank = next(
                (1.0 / rank for rank, item in enumerate(top, start=1)
                 if item.chunk_id in relevant), 0.0
            )
            answerable_recalls.append(recall)
            answerable_rrs.append(reciprocal_rank)
            retrieved_document_precision = (
                len(found_documents & set(example.expected_document_ids)) / len(found_documents)
                if found_documents else 0.0
            )
            document_recall = (
                len(found_documents & set(example.expected_document_ids))
                / len(example.expected_document_ids)
            )
            answerable_document_precisions.append(retrieved_document_precision)
            answerable_document_recalls.append(document_recall)
        else:
            abstained = not top or top[0].rrf_score <= abstention_rrf_threshold
            unanswerable_abstentions.append(float(abstained))

        answer_citation_accuracy = None
        citation_grounding_rate = None
        citation_evaluation_status = None
        if citations is not None and not example.unanswerable:
            has_citations = example.query_id in citations
            answerable_citation_coverage_count += int(has_citations)
            citation_evaluation_status = "EVALUATED" if has_citations else "MISSING_CITATIONS"
            cited = [citation if isinstance(citation, AnswerCitation)
                     else AnswerCitation.model_validate(citation)
                     for citation in citations.get(example.query_id, ())]
            evidence_by_identity = {
                (item.chunk_id, item.document_sha256): item for item in top
            }
            if cited:
                retrieved_identities = set(evidence_by_identity)
                grounded = [
                    (citation.chunk_id, citation.document_sha256) in retrieved_identities
                    for citation in cited
                ]
                citation_grounding_rate = sum(grounded) / len(cited)
                citation_grounding_scores.append(citation_grounding_rate)
                valid_citations = sum(
                    _citation_is_supported(
                        citation, evidence_by_identity, example.evidence_references
                    )
                    for citation in cited
                )
                answer_citation_accuracy = valid_citations / len(cited)
                citation_scores.append(answer_citation_accuracy)
            else:
                citation_grounding_rate = 0.0
                answer_citation_accuracy = 0.0
                citation_grounding_scores.append(0.0)
                citation_scores.append(0.0)

        answer_correctness = None
        answer_evaluation_status = None
        if answer_evaluator is not None and answers is not None and example.query_id in answers:
            answer_correctness = float(await answer_evaluator.evaluate(
                example, answers[example.query_id], top
            ))
            correctness_scores.append(answer_correctness)
            answer_evaluation_status = "EVALUATED"
            if not example.unanswerable:
                answerable_answer_evaluation_count += 1
        elif answer_evaluator is not None and not example.unanswerable:
            answer_evaluation_status = "MISSING_ANSWER"

        rows.append({
            "query_id": example.query_id,
            "query": example.query,
            "unanswerable": example.unanswerable,
            "relevant_chunk_ids": list(example.relevant_chunk_ids),
            "expected_document_ids": list(example.expected_document_ids),
            "evidence_references": [ref.model_dump(mode="json")
                                    for ref in example.evidence_references],
            "retrieved_chunk_ids": [item.chunk_id for item in top],
            "retrieved_document_ids": [item.document_sha256 for item in top],
            "recall_at_k": recall,
            "reciprocal_rank": reciprocal_rank,
            "retrieved_document_precision": retrieved_document_precision
            if not example.unanswerable else None,
            "document_recall_at_k": document_recall if not example.unanswerable else None,
            "citation_grounding_rate": citation_grounding_rate,
            "citation_accuracy": answer_citation_accuracy,
            "citation_evaluation_status": citation_evaluation_status,
            "answer_evaluation_status": answer_evaluation_status,
            "abstained": (not top or top[0].rrf_score <= abstention_rrf_threshold),
            "answer_correctness": answer_correctness,
            "evidence": [item.model_dump(mode="json") for item in top],
        })

    metrics = {
        f"recall_at_{k}": sum(answerable_recalls) / len(answerable_recalls)
        if answerable_recalls else None,
        "mrr": sum(answerable_rrs) / len(answerable_rrs) if answerable_rrs else None,
        "retrieved_document_precision": sum(answerable_document_precisions)
        / len(answerable_document_precisions) if answerable_document_precisions else None,
        f"document_recall_at_{k}": sum(answerable_document_recalls)
        / len(answerable_document_recalls) if answerable_document_recalls else None,
        "citation_accuracy": sum(citation_scores) / len(citation_scores)
        if citation_scores else None,
        "citation_grounding_rate": sum(citation_grounding_scores)
        / len(citation_grounding_scores) if citation_grounding_scores else None,
        "citation_evaluation_coverage": (
            answerable_citation_coverage_count / answerable_query_count
            if citations is not None and answerable_query_count else None
        ),
        "unanswerable_query_abstention": sum(unanswerable_abstentions)
        / len(unanswerable_abstentions) if unanswerable_abstentions else None,
        "answer_correctness": sum(correctness_scores) / len(correctness_scores)
        if correctness_scores else None,
        "answerable_queries_with_answer_evaluation": (
            answerable_answer_evaluation_count if answer_evaluator is not None else None
        ),
        "answerable_answer_evaluation_coverage": (
            answerable_answer_evaluation_count / answerable_query_count
            if answer_evaluator is not None and answerable_query_count else None
        ),
    }
    retrieval_configuration = getattr(retriever, "configuration", {})
    return {
        "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.version,
        "synthetic_dataset": dataset.synthetic,
        "annotation_source": dataset.annotation_source,
        "corpus_snapshot": dict(dataset.corpus_snapshot),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "k": k,
            "rrf": retrieval_configuration.get("rrf", {}),
            "abstention_rrf_threshold": abstention_rrf_threshold,
            "answer_correctness_enabled": answer_evaluator is not None,
            "retrieval": retrieval_configuration,
        },
        "metrics": metrics,
        "per_query": rows,
    }


def load_evaluation_dataset(path: str | Path) -> FrozenEvaluationDataset:
    return FrozenEvaluationDataset.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_evaluation_run(run: dict, directory: str | Path) -> Path:
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_id = run["dataset_id"]
    if not isinstance(dataset_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", dataset_id):
        raise ValueError("Evaluation dataset_id is not safe for an artifact filename")
    stamp = datetime.fromisoformat(run["timestamp"].replace("Z", "+00:00"))
    artifact = output_dir / f"{dataset_id}-{stamp.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    artifact.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return artifact


async def run_and_save_evaluation(
    dataset: FrozenEvaluationDataset,
    retriever,
    *,
    artifact_directory: str | Path,
    abstention_rrf_threshold: float,
    **evaluation_options,
) -> tuple[dict, Path]:
    run = await run_evaluation(
        dataset, retriever, abstention_rrf_threshold=abstention_rrf_threshold,
        **evaluation_options,
    )
    return run, save_evaluation_run(run, artifact_directory)
