"""Read-only resolution of existing image rows and chunk references."""
from __future__ import annotations

import hashlib
from urllib.parse import quote
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
    visual_description_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    visual_description_text: str | None = None
    visual_description_provider: str | None = None
    visual_description_model: str | None = None
    visual_prompt_version: str | None = None
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
            params = {
                "select": "ingestion_run_id,page_number,block_id,image_sha256,image_bucket,image_object_path,image_file_size_bytes,ocr_status,ocr_text_sha256,ocr_text_length,visual_description,decision,source_coordinates,reading_order",
                "ingestion_run_id": f"eq.{ingestion_run_id}",
                "page_number": f"eq.{page_number}",
                "block_id": f"eq.{block_id}",
            }
            try:
                response = await self._client.get(
                    f"{self._url}/rest/v1/image_screening_results",
                    headers={"apikey": self._key, "Authorization": f"Bearer {self._key}"},
                    params=params,
                )
                response.raise_for_status()
                rows = response.json()
            except httpx.HTTPStatusError as exc:
                try:
                    body = exc.response.json()
                except ValueError:
                    body = {}
                missing_vlm_column = (
                    exc.response.status_code == 400
                    and isinstance(body, dict)
                    and body.get("code") == "PGRST204"
                    and "visual_description" in str(body.get("message", ""))
                )
                if not missing_vlm_column:
                    raise ImageAssetResolutionError("Supabase image asset lookup failed") from exc
                params["select"] = params["select"].replace(",visual_description", "")
                try:
                    response = await self._client.get(
                        f"{self._url}/rest/v1/image_screening_results",
                        headers={"apikey": self._key, "Authorization": f"Bearer {self._key}"},
                        params=params,
                    )
                    response.raise_for_status()
                    rows = response.json()
                except httpx.HTTPError as fallback_exc:
                    raise ImageAssetResolutionError("Supabase image asset lookup failed") from fallback_exc
            except httpx.HTTPError as exc:
                raise ImageAssetResolutionError("Supabase image asset lookup failed") from exc
            if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
                raise ImageAssetResolutionError(
                    "Image asset composite key did not resolve to exactly one record"
                )
            row = rows[0]
            description = row.get("visual_description")
            visual_description_sha256 = None
            visual_description_provider = None
            visual_description_model = None
            has_valid_description = False
            if isinstance(description, dict):
                description_text = description.get("description_text")
                description_digest = description.get("description_sha256")
                provider = description.get("provider")
                model = description.get("model")
                prompt_version = description.get("prompt_version")
                if (
                    isinstance(description_text, str)
                    and isinstance(description_digest, str)
                    and hashlib.sha256(description_text.encode("utf-8")).hexdigest()
                    == description_digest
                    and isinstance(provider, str)
                    and bool(provider)
                    and isinstance(model, str)
                    and bool(model)
                    and isinstance(prompt_version, str)
                    and bool(prompt_version.strip())
                ):
                    visual_description_sha256 = description_digest
                    visual_description_text = description_text
                    visual_description_provider = provider
                    visual_description_model = model
                    visual_prompt_version = prompt_version
                    has_valid_description = True
                else:
                    visual_prompt_version = None
                    visual_description_text = None
            else:
                visual_prompt_version = None
                visual_description_text = None
            if (row.get("ingestion_run_id") != ingestion_run_id
                    or row.get("page_number") != page_number
                    or row.get("block_id") != block_id
                    or row.get("image_bucket") != IMAGE_CROP_BUCKET
                    or not isinstance(row.get("image_object_path"), str)
                    or not row["image_object_path"]
                    or type(row.get("image_file_size_bytes")) is not int
                    or type(row.get("reading_order")) is not int
                    or not isinstance(row.get("source_coordinates"), dict)
                    or not (
                        (row.get("ocr_status") == "TEXT_FOUND"
                         and row.get("decision") in {"ADMITTED", "ADMITTED_FLAGGED"})
                        or (row.get("ocr_status") == "NO_TEXT"
                            and row.get("decision") is None
                            and has_valid_description)
                    )):
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
                visual_description_sha256=visual_description_sha256,
                visual_description_text=visual_description_text,
                visual_description_provider=visual_description_provider,
                visual_description_model=visual_description_model,
                visual_prompt_version=visual_prompt_version,
                decision=row["decision"],
                source_coordinates=SourceCoordinates.model_validate(row["source_coordinates"]),
                reading_order=row["reading_order"],
            )
        return resolved

    async def download_verified(self, reference: VisualAssetReference) -> tuple[bytes, str]:
        """Download a private crop only after composite-key and SHA-256 verification."""
        record = (await self.resolve(
            reference.ingestion_run_id,
            ((reference.page_number, reference.source_block_id),),
        )).get((reference.page_number, reference.source_block_id))
        if (record is None or record.image_sha256 != reference.image_sha256
                or record.bucket != reference.bucket or record.object_path != reference.object_path
                or record.file_size_bytes != reference.file_size_bytes):
            raise ImageAssetResolutionError("Visual asset reference does not match stored metadata")
        path = "/".join(quote(part, safe="") for part in record.object_path.split("/"))
        try:
            response = await self._client.get(
                f"{self._url}/storage/v1/object/authenticated/{quote(record.bucket, safe='')}/{path}",
                headers={"apikey": self._key, "Authorization": f"Bearer {self._key}"},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ImageAssetResolutionError("Private image download failed") from exc
        image_bytes = response.content
        if (len(image_bytes) != record.file_size_bytes
                or hashlib.sha256(image_bytes).hexdigest() != record.image_sha256):
            raise ImageAssetResolutionError("Downloaded image bytes do not match persisted SHA-256")
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            media_type = "image/png"
        elif image_bytes.startswith(b"\xff\xd8\xff"):
            media_type = "image/jpeg"
        else:
            raise ImageAssetResolutionError("Stored crop has an unsupported image format")
        return image_bytes, media_type


