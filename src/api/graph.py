"""Read-only, frontend-oriented view of the persisted knowledge graph."""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from src.ingestion.chunk_contracts import VisualAssetReference
from src.ingestion.provenance.image_asset_resolver import (
    ImageAssetResolutionError,
    SupabaseImageAssetResolver,
)

router = APIRouter(tags=["Graph"])

_GRAPH_NODES_QUERY = """
MATCH (n)
WHERE n.id IS NOT NULL
OPTIONAL MATCH (d:DOCUMENT)-[:HAS_CHUNK]->(n)
WITH n, collect(DISTINCT d.current_ingestion_run_id) AS current_runs
RETURN properties(n) AS properties, labels(n) AS labels
       , head(current_runs) AS current_run_id
ORDER BY n.id
SKIP $offset
LIMIT $limit
"""

_GRAPH_EDGES_QUERY = """
MATCH (source)-[relationship]->(target)
WHERE source.id IN $node_ids OR target.id IN $node_ids
OPTIONAL MATCH (source_document:DOCUMENT)-[:HAS_CHUNK]->(source)
OPTIONAL MATCH (target_document:DOCUMENT)-[:HAS_CHUNK]->(target)
RETURN source.id AS source, properties(source) AS source_properties,
       labels(source) AS source_labels,
       coalesce(source_document.current_ingestion_run_id, source.ingestion_run_id) AS source_run_id,
       target.id AS target, properties(target) AS target_properties,
       labels(target) AS target_labels,
       coalesce(target_document.current_ingestion_run_id, target.ingestion_run_id) AS target_run_id,
       type(relationship) AS type,
       properties(relationship) AS properties
ORDER BY source, type, target
"""


class GraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    type: str
    label: str
    properties: dict[str, Any]
    page_anchor: bool = False


class GraphEdge(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    type: str
    source: str
    target: str
    properties: dict[str, Any]


class GraphVisualAsset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    attached_to: str
    reference: VisualAssetReference


class GraphResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    visual_assets: tuple[GraphVisualAsset, ...]
    offset: int
    next_offset: int | None


class GraphSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=10, ge=1, le=50)


class GraphSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    results: tuple[dict[str, Any], ...]


def _json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _evidence_objects(properties: dict[str, Any]) -> list[dict[str, Any]]:
    values = properties.get("evidence_references", ())
    evidence = (
        [item for value in values if (item := _json_object(value)) is not None]
        if isinstance(values, (list, tuple)) else []
    )
    sources = properties.get("sources")
    run_id = properties.get("ingestion_run_id")
    if isinstance(sources, (list, tuple)) and isinstance(run_id, str):
        for source in sources:
            if isinstance(source, dict) and source.get("extraction_method") in {"IMAGE_OCR", "IMAGE_VLM"}:
                evidence.append({
                    "ingestion_run_id": run_id,
                    "page_number": source.get("page_number"),
                    "source_provenance": source,
                })
    return evidence


def _frontend_properties(properties: dict[str, Any]) -> dict[str, Any]:
    """Decode persisted JSON audit arrays and remove URL-shaped fields recursively."""
    decoded = dict(properties)
    for key in ("evidence_references", "support_attestations", "extraction_assertions"):
        values = decoded.get(key)
        if isinstance(values, (list, tuple)):
            decoded[key] = [parsed if (parsed := _json_object(value)) is not None else value
                            for value in values]
    sources_json = decoded.pop("sources_json", None)
    if isinstance(sources_json, str):
        try:
            sources = json.loads(sources_json)
        except (TypeError, ValueError):
            sources = None
        if isinstance(sources, list):
            decoded["sources"] = sources

    def safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: safe(item) for key, item in value.items()
                if key.casefold() not in {"url", "public_url", "signed_url", "image_url"}
            }
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        return value

    return safe(decoded)


def _visual_key(evidence: dict[str, Any]) -> tuple[str, int, str, str | None] | None:
    provenance = evidence.get("source_provenance")
    if not isinstance(provenance, dict) or provenance.get("extraction_method") not in {
        "IMAGE_OCR", "IMAGE_VLM"
    }:
        return None
    run_id = evidence.get("ingestion_run_id")
    page = evidence.get("page_number")
    block_id = provenance.get("source_image_block_id")
    image_sha256 = provenance.get("source_image_sha256")
    if (not isinstance(run_id, str) or not run_id or type(page) is not int or page < 1
            or not isinstance(block_id, str) or not block_id):
        return None
    return run_id, page, block_id, image_sha256 if isinstance(image_sha256, str) else None


async def _resolve_visual_assets(
    attachments: list[tuple[str, dict[str, Any]]],
    resolver: SupabaseImageAssetResolver | None,
) -> tuple[GraphVisualAsset, ...]:
    keys_by_run: dict[str, set[tuple[int, str]]] = {}
    image_hash_by_key: dict[tuple[str, int, str], str | None] = {}
    for attached_to, properties in attachments:
        for evidence in _evidence_objects(properties):
            key = _visual_key(evidence)
            if key is None:
                continue
            run_id, page, block_id, image_sha256 = key
            keys_by_run.setdefault(run_id, set()).add((page, block_id))
            image_hash_by_key[(run_id, page, block_id)] = image_sha256
    if not keys_by_run:
        return ()
    if resolver is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Visual asset metadata is unavailable because Supabase is not configured",
        )
    result: list[GraphVisualAsset] = []
    seen: set[tuple[str, str]] = set()
    try:
        for run_id, keys in sorted(keys_by_run.items()):
            resolved = await resolver.resolve(run_id, tuple(sorted(keys)))
            for attached_to, properties in attachments:
                for evidence in _evidence_objects(properties):
                    key = _visual_key(evidence)
                    if key is None or key[0] != run_id:
                        continue
                    _, page, block_id, _ = key
                    asset = resolved.get((page, block_id))
                    expected_hash = image_hash_by_key.get((run_id, page, block_id))
                    if asset is None or (expected_hash and asset.image_sha256 != expected_hash):
                        raise ImageAssetResolutionError(
                            "Visual evidence does not match the stored image metadata"
                        )
                    unique = (attached_to, asset.image_sha256)
                    if unique not in seen:
                        seen.add(unique)
                        result.append(GraphVisualAsset(
                            attached_to=attached_to, reference=asset.reference
                        ))
    except ImageAssetResolutionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="A referenced private visual asset could not be verified",
        ) from exc
    return tuple(result)


@router.get(
    "/graph",
    response_model=GraphResponse,
    summary="Read the persisted Neo4j graph for frontend rendering",
    description=(
        "Returns all identified nodes and relationships in the selected Neo4j database, "
        "including documents, chunks, semantic entities and assertions, plus verified "
        "private visual asset references. This endpoint performs read-only Neo4j and "
        "Supabase lookups; it never returns image bytes or public/signed storage URLs."
    ),
)
async def get_graph(
    request: Request,
    limit: int = Query(default=2000, ge=1, le=10000),
    offset: int = Query(default=0, ge=0),
) -> GraphResponse:
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Neo4j is not available")
    try:
        async with db.get_session() as session:
            result = await session.run(_GRAPH_NODES_QUERY, limit=limit + 1, offset=offset)
            records = [record async for record in result]
            has_more = len(records) > limit
            records = records[:limit]
            properties_by_id: dict[str, dict[str, Any]] = {}
            labels_by_id: dict[str, list[str]] = {}
            page_ids: list[str] = []
            for record in records:
                properties = dict(record["properties"])
                node_id = properties.get("id")
                labels = list(record.get("labels", ()))
                if not isinstance(node_id, str) or not node_id:
                    continue
                current_run_id = record.get("current_run_id")
                if current_run_id:
                    properties["ingestion_run_id"] = current_run_id
                properties_by_id[node_id] = properties
                labels_by_id[node_id] = labels
                page_ids.append(node_id)
            ids = list(dict.fromkeys(page_ids))
            edges: list[GraphEdge] = []
            if ids:
                edge_result = await session.run(_GRAPH_EDGES_QUERY, node_ids=ids)
                edge_records = [record async for record in edge_result]
                for record in edge_records:
                    for prefix in ("source", "target"):
                        endpoint_id = str(record[prefix])
                        if endpoint_id not in properties_by_id:
                            properties_by_id[endpoint_id] = dict(record[f"{prefix}_properties"])
                            if record.get(f"{prefix}_run_id"):
                                properties_by_id[endpoint_id]["ingestion_run_id"] = record[f"{prefix}_run_id"]
                            labels_by_id[endpoint_id] = list(record[f"{prefix}_labels"])
                    props = dict(record["properties"])
                    edge_type = str(record["type"])
                    source, target = str(record["source"]), str(record["target"])
                    edge_id = str(props.get("id") or f"{source}:{edge_type}:{target}")
                    edges.append(GraphEdge(
                        id=edge_id, type=edge_type, source=source, target=target,
                        properties=_frontend_properties(props),
                    ))
            nodes: list[GraphNode] = []
            page_id_set = set(page_ids)
            for node_id, properties in properties_by_id.items():
                labels = labels_by_id[node_id]
                node_type = (
                    "KNOWLEDGE_ASSERTION" if "KNOWLEDGE_ASSERTION" in labels
                    else next((label for label in labels if label != "SEMANTIC_ENTITY"), "ENTITY")
                )
                nodes.append(GraphNode(
                    id=node_id,
                    type=node_type,
                    label=str(properties.get("name") or properties.get("title")
                              or properties.get("object_value") or properties.get("description")
                              or properties.get("predicate") or node_id),
                    properties=_frontend_properties(properties),
                    page_anchor=node_id in page_id_set,
                ))
        attachments = [
            (node.id, _frontend_properties(properties_by_id[node.id])) for node in nodes
        ] + [
            (f"edge:{edge.id}", edge.properties) for edge in edges
        ]
        assets = await _resolve_visual_assets(
            attachments, getattr(request.app.state, "image_asset_resolver", None)
        )
        return GraphResponse(
            nodes=tuple(nodes), edges=tuple(edges), visual_assets=assets,
            offset=offset, next_offset=offset + len(page_ids) if has_more else None,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j graph data could not be read",
        ) from exc


@router.post(
    "/graph/search",
    response_model=GraphSearchResponse,
    summary="Search text and image-derived graph evidence",
)
async def search_graph(request: Request, body: GraphSearchRequest) -> GraphSearchResponse:
    retriever = getattr(request.app.state, "multimodal_retriever", None)
    visual_retriever = getattr(request.app.state, "visual_graph_retriever", None)
    if retriever is None or visual_retriever is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval is unavailable; configure embedding settings and indexes",
        )
    try:
        results = await retriever.search_multimodal(
            body.query, visual_retriever, body.limit
        )
        return GraphSearchResponse(results=tuple(item.model_dump(mode="json") for item in results))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph retrieval could not complete against the configured indexes",
        ) from exc


@router.get(
    "/visual-assets/{ingestion_run_id}/{page_number}/{source_block_id}",
    response_class=Response,
    summary="Serve a SHA-256 verified private image crop",
)
async def get_private_visual_asset(
    request: Request,
    ingestion_run_id: str,
    page_number: int,
    source_block_id: str,
    image_sha256: str = Query(pattern=r"^[0-9a-f]{64}$"),
) -> Response:
    resolver = getattr(request.app.state, "image_asset_resolver", None)
    if resolver is None:
        raise HTTPException(status_code=503, detail="Private image storage is unavailable")
    try:
        assets = await resolver.resolve(ingestion_run_id, ((page_number, source_block_id),))
        asset = assets.get((page_number, source_block_id))
        if asset is None or asset.image_sha256 != image_sha256:
            raise HTTPException(status_code=404, detail="Visual asset was not found")
        content, media_type = await resolver.download_verified(asset.reference)
        return Response(
            content=content,
            media_type=media_type,
            headers={
                "Cache-Control": "private, max-age=60, no-transform",
                "Content-Disposition": "inline",
                "X-Content-Type-Options": "nosniff",
                "ETag": f'"{asset.image_sha256}"',
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Private image could not be verified or served") from exc
