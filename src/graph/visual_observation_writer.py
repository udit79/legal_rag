"""Transactional persistence for explicitly unverified visual observations."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from src.graph.semantic_schema import semantic_constraints_ready
from src.graph.visual_observations import VisualObservation


class VisualObservationConflict(RuntimeError):
    pass


def _observation_properties(observation: VisualObservation) -> dict[str, Any]:
    properties = observation.model_dump(mode="json", exclude={"candidates", "relationships"})
    coordinates = properties.pop("source_coordinates")
    properties["source_coordinates_json"] = json.dumps(
        coordinates, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return properties


_EXISTING_QUERY = """
UNWIND $ids AS id
OPTIONAL MATCH (n {id: id})
WHERE n:VISUAL_OBSERVATION OR n:VISUAL_ENTITY_CANDIDATE
RETURN id, collect(properties(n)) AS matches
"""
_SEMANTIC_QUERY = """
MATCH (n:SEMANTIC_ENTITY)
RETURN properties(n) AS properties, labels(n) AS labels
"""
_OBSERVATION_WRITE = """
MERGE (o:VISUAL_OBSERVATION {id: $id})
ON CREATE SET o += $properties
RETURN properties(o) AS properties
"""
_CANDIDATE_WRITE = """
MATCH (o:VISUAL_OBSERVATION {id: $observation_id})
MERGE (e:VISUAL_ENTITY_CANDIDATE {id: $id})
ON CREATE SET e += $properties
MERGE (e)-[r:OBSERVED_IN {id: $relationship_id}]->(o)
ON CREATE SET r.trust_status = 'UNVERIFIED_AI_VISUAL_OBSERVATION'
RETURN properties(e) AS properties, properties(r) AS relationship,
       startNode(r).id AS source_id, endNode(r).id AS target_id
"""
_VISUAL_RELATION_WRITE = """
MATCH (a:VISUAL_ENTITY_CANDIDATE {id: $source_id}),
      (b:VISUAL_ENTITY_CANDIDATE {id: $target_id})
MERGE (a)-[r:VISUALLY_RELATED_TO {id: $id}]->(b)
ON CREATE SET r += $properties
RETURN properties(r) AS properties, startNode(r).id AS source_id, endNode(r).id AS target_id
"""
_VISUAL_REFERENCE_WRITE = """
MATCH (e:VISUAL_ENTITY_CANDIDATE {id: $candidate_id}),
      (n:SEMANTIC_ENTITY {id: $semantic_id})
MERGE (e)-[r:VISUALLY_REFERENCES {id: $id}]->(n)
ON CREATE SET r.trust_status = 'UNVERIFIED_VISUAL_ENTITY_MATCH',
              r.match_method = 'EXACT_NORMALIZED_NAME_AND_TYPE'
RETURN properties(r) AS properties, startNode(r).id AS source_id, endNode(r).id AS target_id
"""
_SHOW_CONSTRAINTS = """
SHOW CONSTRAINTS YIELD type, entityType, labelsOrTypes, properties
RETURN collect({type: type, entity_type: entityType,
                labels: labelsOrTypes, properties: properties}) AS constraints
"""
_VISUAL_CONSTRAINT_SHAPES = {
    ("NODE", "NODE PROPERTY UNIQUENESS", ("VISUAL_OBSERVATION",), ("id",)),
    ("NODE", "NODE PROPERTY UNIQUENESS", ("VISUAL_ENTITY_CANDIDATE",), ("id",)),
    *{
        ("RELATIONSHIP", "RELATIONSHIP PROPERTY UNIQUENESS", (label,), ("id",))
        for label in ("OBSERVED_IN", "VISUALLY_RELATED_TO", "VISUALLY_REFERENCES")
    },
}


async def _visual_constraints_ready(tx: Any) -> bool:
    result = await tx.run(_SHOW_CONSTRAINTS)
    record = await result.single()
    actual = set()
    for item in (record["constraints"] if record else ()):
        kind = str(item.get("type", "")).upper()
        entity = str(item.get("entity_type", "")).upper()
        kind = (
            "NODE PROPERTY UNIQUENESS" if entity == "NODE" and "UNIQUENESS" in kind
            else "RELATIONSHIP PROPERTY UNIQUENESS" if entity == "RELATIONSHIP" and "UNIQUENESS" in kind
            else kind
        )
        actual.add((entity, kind, tuple(item.get("labels") or ()),
                    tuple(item.get("properties") or ())))
    return _VISUAL_CONSTRAINT_SHAPES.issubset(actual)

def _identity(prefix: str, *parts: str) -> str:
    value = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return prefix + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalized(value: object) -> str:
    return " ".join(str(value).casefold().split())


_MATCH_FIELDS = {
    "ORGANIZATION": ("ORGANIZATION", ("name",)),
    "APPLICATION": ("APPLICATION", ("name",)),
    "POLICY": ("POLICY", ("title",)),
    "OBLIGATION": ("OBLIGATION", ("requirement_text",)),
    "CONTROL": ("CONTROL", ("name",)),
    "RISK": ("RISK", ("name",)),
    "FINDING": ("FINDING", ("description",)),
}


def _resolve_exact(candidate: dict, semantic_nodes: list[dict]) -> str | None:
    category = candidate["category"]
    spec = _MATCH_FIELDS.get(category)
    if spec is None:
        return None
    label, keys = spec
    name = _normalized(candidate["name"])
    matches = []
    for row in semantic_nodes:
        props = row["properties"]
        if label not in row["labels"]:
            continue
        if any(_normalized(props.get(key, "")) == name for key in keys):
            matches.append(str(props["id"]))
    return matches[0] if len(matches) == 1 else None


async def _write_visual_observation_transaction(tx, observation: VisualObservation) -> dict[str, int]:
    if not await semantic_constraints_ready(tx):
        raise RuntimeError("Graph identity constraints must be initialized before visual writes")
    if not await _visual_constraints_ready(tx):
        raise RuntimeError("Visual observation uniqueness constraints are missing; initialize graph schema first")
    semantic_result = await tx.run(_SEMANTIC_QUERY)
    semantic_nodes = [dict(row) async for row in semantic_result]
    observation_properties = _observation_properties(observation)
    ids = [observation.id, *(candidate.id for candidate in observation.candidates)]
    identity_result = await tx.run(_EXISTING_QUERY, ids=ids)
    identity_rows = [dict(row) async for row in identity_result]
    if len(identity_rows) != len(ids):
        raise VisualObservationConflict("Visual identity preflight returned an incomplete result")
    for row in identity_rows:
        if len(row["matches"]) > 1:
            raise VisualObservationConflict("Duplicate visual graph identity exists")
        if row["matches"]:
            expected = observation_properties if row["id"] == observation.id else next(
                candidate.model_dump(mode="json") for candidate in observation.candidates
                if candidate.id == row["id"]
            )
            existing = row["matches"][0]
            if any(existing.get(key) != value for key, value in expected.items()):
                raise VisualObservationConflict("Existing visual identity has conflicting immutable content")
    result = await tx.run(_OBSERVATION_WRITE, id=observation.id, properties=observation_properties)
    records = [dict(row) async for row in result]
    if len(records) != 1 or any(records[0]["properties"].get(k) != v for k, v in observation_properties.items()):
        raise VisualObservationConflict("Visual observation identity conflicted during write")
    counts = {"observations": 1, "candidates": 0, "candidate_relationships": 0, "entity_links": 0}
    for candidate in observation.candidates:
        properties = candidate.model_dump(mode="json")
        rel_id = _identity("observed-in:", candidate.id, observation.id)
        result = await tx.run(_CANDIDATE_WRITE, observation_id=observation.id,
                              id=candidate.id, properties=properties, relationship_id=rel_id)
        records = [dict(row) async for row in result]
        if (len(records) != 1
                or any(records[0]["properties"].get(k) != v for k, v in properties.items())
                or records[0]["source_id"] != candidate.id
                or records[0]["target_id"] != observation.id
                or records[0]["relationship"].get("id") != rel_id
                or records[0]["relationship"].get("trust_status") != observation.trust_status):
            raise VisualObservationConflict("Visual candidate identity conflicted during write")
        counts["candidates"] += 1
        semantic_id = _resolve_exact(properties, semantic_nodes)
        if semantic_id:
            link_id = _identity("visually-references:", candidate.id, semantic_id)
            link_result = await tx.run(_VISUAL_REFERENCE_WRITE,
                                       candidate_id=candidate.id, semantic_id=semantic_id, id=link_id)
            links = [dict(row) async for row in link_result]
            if len(links) != 1 or links[0]["source_id"] != candidate.id or links[0]["target_id"] != semantic_id:
                raise VisualObservationConflict("Visual entity resolution link was ambiguous")
            counts["entity_links"] += 1
    for relation in observation.relationships:
        properties = {
            "id": relation.id,
            "observed_relation": relation.observed_relation,
            "evidence_quote": relation.evidence_quote,
            "quote_start": relation.quote_start,
            "quote_end": relation.quote_end,
            "trust_status": observation.trust_status,
            "observation_id": observation.id,
        }
        result = await tx.run(_VISUAL_RELATION_WRITE,
                              source_id=relation.source_entity_id,
                              target_id=relation.target_entity_id,
                              id=relation.id, properties=properties)
        records = [dict(row) async for row in result]
        if (len(records) != 1 or records[0]["source_id"] != relation.source_entity_id
                or records[0]["target_id"] != relation.target_entity_id
                or any(records[0]["properties"].get(k) != v for k, v in properties.items())):
            raise VisualObservationConflict("Visual relationship identity conflicted")
        counts["candidate_relationships"] += 1
    return counts


async def persist_visual_observations(db_connection, observations: tuple[VisualObservation, ...]) -> dict[str, int]:
    if not observations:
        return {"observations": 0, "candidates": 0, "candidate_relationships": 0, "entity_links": 0}
    if len({item.id for item in observations}) != len(observations):
        raise ValueError("Visual observation batch contains duplicate identities")
    totals = {"observations": 0, "candidates": 0, "candidate_relationships": 0, "entity_links": 0}
    async with db_connection.get_session() as session:
        async def transaction(tx):
            for observation in observations:
                counts = await _write_visual_observation_transaction(tx, observation)
                for key, value in counts.items():
                    totals[key] += value
            return totals
        return await session.execute_write(transaction)
