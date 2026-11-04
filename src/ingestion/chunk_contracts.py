from typing import Final, Self
from pydantic import BaseModel, ConfigDict, Field, model_validator
from src.ingestion.contracts import NonEmptyStr, NonNegativeInt, PositiveInt
from src.ingestion.parser_contracts import BlockType

TEXT_BEARING_BLOCK_TYPES: Final[frozenset[BlockType]] = frozenset(
    {"TEXT", "TABLE", "OTHER"}
)
OPTIONAL_TEXT_BLOCK_TYPES: Final[frozenset[BlockType]] = frozenset({"IMAGE"})


class ChunkSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    page_number: PositiveInt
    block_id: NonEmptyStr
    block_reading_order: NonNegativeInt
    block_type: BlockType
    source_text_length: NonNegativeInt | None = Field(
        default=None,
        description="Canonical source text length in Python Unicode characters.",
    )
    start_offset: NonNegativeInt | None = Field(
        default=None,
        description="Inclusive Python Unicode character offset into source text.",
    )
    end_offset: NonNegativeInt | None = Field(
        default=None,
        description="Exclusive Python Unicode character offset into source text.",
    )

    @model_validator(mode="after")
    def validate_source_range(self) -> Self:
        prefix = f"page-{self.page_number}-block-"
        suffix = self.block_id.removeprefix(prefix)
        if not self.block_id.startswith(prefix) or not suffix.isascii() or not suffix.isdecimal():
            raise ValueError("block_id must retain its page-scoped canonical format")
        offsets = (
            self.source_text_length,
            self.start_offset,
            self.end_offset,
        )
        if all(offset is None for offset in offsets):
            if self.block_type in TEXT_BEARING_BLOCK_TYPES:
                raise ValueError(
                    "Text-bearing source blocks require a source text range"
                )
            return self
        if self.block_type not in (
            TEXT_BEARING_BLOCK_TYPES | OPTIONAL_TEXT_BLOCK_TYPES
        ):
            raise ValueError("Non-text-bearing source blocks cannot have text offsets")
        if any(offset is None for offset in offsets):
            raise ValueError(
                "Source text length and both offsets must be provided together"
            )
        assert self.source_text_length is not None
        assert self.start_offset is not None
        assert self.end_offset is not None
        if self.end_offset <= self.start_offset:
            raise ValueError("Source offsets must identify a non-empty range")
        if self.end_offset > self.source_text_length:
            raise ValueError("Source end offset exceeds the source text length")
        return self


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    index: NonNegativeInt
    text: str
    sources: tuple[ChunkSource, ...]

    @model_validator(mode="after")
    def validate_chunk(self) -> Self:
        if not self.text:
            raise ValueError("Chunk text cannot be empty")
        if not self.sources:
            raise ValueError("A chunk must identify at least one source")
        source_order = tuple(
            (
                source.page_number,
                source.block_reading_order,
                source.start_offset if source.start_offset is not None else -1,
            )
            for source in self.sources
        )
        if source_order != tuple(sorted(source_order)):
            raise ValueError("Chunk sources must be in document reading order")

        source_metadata: dict[str, tuple[BlockType, int | None]] = {}
        source_ranges: dict[str, list[tuple[int, int]]] = {}
        source_blocks_without_ranges: set[str] = set()
        for source in self.sources:
            metadata = (source.block_type, source.source_text_length)
            existing_metadata = source_metadata.setdefault(source.block_id, metadata)
            if existing_metadata != metadata:
                raise ValueError(
                    "Sources for the same block must have consistent block type "
                    "and source text length"
                )
            if source.start_offset is None or source.end_offset is None:
                if source.block_id in source_blocks_without_ranges:
                    raise ValueError(
                        "A chunk cannot contain a duplicate source reference"
                    )
                source_blocks_without_ranges.add(source.block_id)
                continue
            ranges = source_ranges.setdefault(source.block_id, [])
            if ranges and source.start_offset < ranges[-1][1]:
                raise ValueError(
                    "Source ranges for the same block cannot overlap within a chunk"
                )
            ranges.append((source.start_offset, source.end_offset))
        return self


class ChunkedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    document_id: NonEmptyStr
    source_page_count: PositiveInt
    policy_version: NonEmptyStr
    chunks: tuple[Chunk, ...]

    @model_validator(mode="after")
    def validate_chunks(self) -> Self:
        indices = tuple(chunk.index for chunk in self.chunks)
        if indices != tuple(range(len(self.chunks))):
            raise ValueError("Chunk indices must be zero-based and contiguous")
        if any(
            source.page_number > self.source_page_count
            for chunk in self.chunks
            for source in chunk.sources
        ):
            raise ValueError("Chunk provenance references a page outside the source")
        return self
