"""Read-only resolution of existing image rows and chunk references."""
from __future__ import annotations

import httpx
from pydantic import BaseModel, ConfigDict, Field

from src.ingestion.chunk_contracts import VisualAssetReference
from src.ingestion.parser_contracts import SourceCoordinates

IMAGE_CROP_BUCKET = "ingestion-image-crops"


class ImageAssetResolutionError(ValueError):
    pass


class ResolvedImageAsset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ingestion_run_id: str = Field(min_length=1)
    page_number: int = Field(gt=0)
    block_id: str = Field(min_length=1)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bucket: str = Field(min_length=1)
    object_path: str = Field(min_length=1)
    file_size_bytes: int = Field(ge=0)
    ocr_status: str
    ocr_text_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    ocr_text_length: int = Field(ge=0)
    decision: str | None
    source_coordinates: SourceCoordinates
    reading_order: int = Field(ge=0)

    @property
    def reference(self) -> VisualAssetReference:
        return VisualAssetReference(
            ingestion_run_id=self.ingestion_run_id,
            page_number=self.page_number,
            source_block_id=self.block_id,
            image_sha256=self.image_sha256,
            bucket="ingestion-image-crops",
            object_path=self.object_path,
            file_size_bytes=self.file_size_bytes,
        )


class SupabaseImageAssetResolver:
    def __init__(self, base_url: str, service_role_key: str, client: httpx.AsyncClient):
        if not base_url.startswith(("https://", "http://")) or not service_role_key.strip():
            raise ValueError("Supabase URL and service role key are required")
        self._url = base_url.rstrip("/")
        self._key = service_role_key.strip()
        self._client = client

    async def resolve(
        self, ingestion_run_id: str, keys: tuple[tuple[int, str], ...]
    ) -> dict[tuple[int, str], ResolvedImageAsset]:
        """Resolve by `(ingestion_run_id, page_number, block_id)`; never accesses storage bytes."""
        if not ingestion_run_id.strip() or len(set(keys)) != len(keys):
            raise ValueError("Image asset resolution requires a run and unique keys")
        resolved: dict[tuple[int, str], ResolvedImageAsset] = {}
        for page_number, block_id in keys:
            try:
                response = await self._client.get(
                    f"{self._url}/rest/v1/image_screening_results",
                    headers={"apikey": self._key, "Authorization": f"Bearer {self._key}"},
                    params={
                        "select": "ingestion_run_id,page_number,block_id,image_sha256,image_bucket,image_object_path,image_file_size_bytes,ocr_status,ocr_text_sha256,ocr_text_length,decision,source_coordinates,reading_order",
                        "ingestion_run_id": f"eq.{ingestion_run_id}",
                        "page_number": f"eq.{page_number}",
                        "block_id": f"eq.{block_id}",
                    },
                )
                response.raise_for_status()
                rows = response.json()
            except httpx.HTTPError as exc:
                raise ImageAssetResolutionError("Supabase image asset lookup failed") from exc
            if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
                raise ImageAssetResolutionError(
                    "Image asset composite key did not resolve to exactly one record"
                )
            row = rows[0]
            if (row.get("ingestion_run_id") != ingestion_run_id
                    or row.get("page_number") != page_number
                    or row.get("block_id") != block_id
                    or row.get("image_bucket") != IMAGE_CROP_BUCKET
                    or not isinstance(row.get("image_object_path"), str)
                    or not row["image_object_path"]
                    or type(row.get("image_file_size_bytes")) is not int
                    or type(row.get("reading_order")) is not int
                    or not isinstance(row.get("source_coordinates"), dict)
                    or row.get("ocr_status") != "TEXT_FOUND"
                    or row.get("decision") not in {"ADMITTED", "ADMITTED_FLAGGED"}):
                raise ImageAssetResolutionError("Supabase image asset metadata is inconsistent")
            resolved[(page_number, block_id)] = ResolvedImageAsset(
                ingestion_run_id=ingestion_run_id,
                page_number=page_number,
                block_id=block_id,
                image_sha256=row["image_sha256"],
                bucket=IMAGE_CROP_BUCKET,
                object_path=row["image_object_path"],
                file_size_bytes=row["image_file_size_bytes"],
                ocr_status=row["ocr_status"],
                ocr_text_sha256=row["ocr_text_sha256"],
                ocr_text_length=row["ocr_text_length"],
                decision=row["decision"],
                source_coordinates=SourceCoordinates.model_validate(row["source_coordinates"]),
                reading_order=row["reading_order"],
            )
        return resolved


