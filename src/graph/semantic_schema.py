"""Explicit, preflighted uniqueness schema for semantic graph writes.

This is not called by the structural corpus importer. Run it only after
reviewing its duplicate report for the target database.
"""

from __future__ import annotations

from src.domain.ontology import NodeType, RelationType

SEMANTIC_NODE_LABELS = tuple(
    kind.value for kind in NodeType if kind not in {NodeType.DOCUMENT, NodeType.CHUNK}
)
SEMANTIC_RELATION_TYPES = tuple(
    relation.value
    for relation in RelationType
    if relation not in {RelationType.HAS_CHUNK, RelationType.NEXT_CHUNK, RelationType.EXTRACTED_FROM}
)
ASSERTION_RELATION_TYPES = (*SEMANTIC_RELATION_TYPES, "ASSERTION_CLASSIFICATION")

_NODE_CONSTRAINTS = (
    "CREATE CONSTRAINT semantic_entity_id_unique IF NOT EXISTS "
    "FOR (n:SEMANTIC_ENTITY) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT knowledge_assertion_id_unique IF NOT EXISTS "
    "FOR (n:KNOWLEDGE_ASSERTION) REQUIRE n.id IS UNIQUE",
)
_RELATIONSHIP_CONSTRAINTS = tuple(
    f"CREATE CONSTRAINT semantic_{relation.casefold()}_id_unique IF NOT EXISTS "
    f"FOR ()-[r:{relation}]-() REQUIRE r.id IS UNIQUE"
    for relation in ASSERTION_RELATION_TYPES
)
SEMANTIC_CONSTRAINT_STATEMENTS = (*_NODE_CONSTRAINTS, *_RELATIONSHIP_CONSTRAINTS)

DUPLICATE_NODE_IDS_QUERY = """
MATCH (n)
WHERE any(label IN labels(n) WHERE label IN $labels)
WITH n.id AS id, count(n) AS count
WHERE id IS NULL OR count > 1
RETURN id, count
"""
DUPLICATE_RELATIONSHIP_IDS_QUERY = """
MATCH ()-[r]->()
WHERE type(r) IN $types
WITH type(r) AS type, r.id AS id, count(r) AS count
WHERE id IS NULL OR count > 1
RETURN type, id, count
"""
PREFLIGHT_MATCHED_NODES_QUERY = """
MATCH (n)
WHERE any(label IN labels(n) WHERE label IN $labels)
RETURN count(n) AS count
"""
BACKFILL_SEMANTIC_LABEL_QUERY = """
MATCH (n)
WHERE any(label IN labels(n) WHERE label IN $labels)
SET n:SEMANTIC_ENTITY
RETURN count(n) AS count
"""
SHOW_SEMANTIC_CONSTRAINTS_QUERY = """
SHOW CONSTRAINTS YIELD name, type, entityType, labelsOrTypes, properties
RETURN collect({name: name, type: type, entity_type: entityType,
                labels: labelsOrTypes, properties: properties}) AS constraints
"""


class SemanticSchemaPreflightError(RuntimeError):
    """Existing identities are unsafe to constrain without resolving duplicates."""


def _constraint_shape(record: dict) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
    return (
        str(record.get("entity_type", "")).upper(),
        str(record.get("type", "")).upper(),
        tuple(record.get("labels") or ()),
        tuple(record.get("properties") or ()),
    )


def required_semantic_constraint_shapes() -> set[tuple[str, str, tuple[str, ...], tuple[str, ...]]]:
    return {
        ("NODE", "NODE PROPERTY UNIQUENESS", ("SEMANTIC_ENTITY",), ("id",)),
        ("NODE", "NODE PROPERTY UNIQUENESS", ("KNOWLEDGE_ASSERTION",), ("id",)),
        *{
            ("RELATIONSHIP", "RELATIONSHIP PROPERTY UNIQUENESS", (name,), ("id",))
            for name in ASSERTION_RELATION_TYPES
        },
    }


async def inspect_semantic_schema(tx) -> list[dict]:
    result = await tx.run(SHOW_SEMANTIC_CONSTRAINTS_QUERY)
    record = await result.single()
    constraints = list(record["constraints"]) if record else []
    actual = {_constraint_shape(item) for item in constraints}
    # Neo4j may report KEY constraint variants; both enforce uniqueness.
    normalized = {
        (entity, "NODE PROPERTY UNIQUENESS" if entity == "NODE" and "UNIQUENESS" in kind
         else "RELATIONSHIP PROPERTY UNIQUENESS" if entity == "RELATIONSHIP" and "UNIQUENESS" in kind
         else kind, labels, properties)
        for entity, kind, labels, properties in actual
    }
    required = required_semantic_constraint_shapes()
    return constraints if required.issubset(normalized) else []


async def semantic_constraints_ready(tx) -> bool:
    return bool(await inspect_semantic_schema(tx))


async def preflight_semantic_identities(db_connection) -> dict[str, int]:
    async with db_connection.get_session() as session:
        nodes = await session.run(
            DUPLICATE_NODE_IDS_QUERY, labels=list(SEMANTIC_NODE_LABELS)
        )
        node_issues = [dict(record) async for record in nodes]
        relationships = await session.run(
            DUPLICATE_RELATIONSHIP_IDS_QUERY,
            types=list(ASSERTION_RELATION_TYPES),
        )
        relationship_issues = [dict(record) async for record in relationships]
        matched = await session.run(
            PREFLIGHT_MATCHED_NODES_QUERY, labels=list(SEMANTIC_NODE_LABELS)
        )
        matched_record = await matched.single()
    if node_issues or relationship_issues:
        raise SemanticSchemaPreflightError(
            "Duplicate or missing semantic identities prevent uniqueness constraints: "
            f"nodes={node_issues[:10]}, relationships={relationship_issues[:10]}"
        )
    return {
        "semantic_nodes_to_label": int(matched_record["count"] if matched_record else 0),
        "node_identity_issues": 0,
        "relationship_identity_issues": 0,
    }


async def initialize_semantic_schema(db_connection) -> dict[str, int]:
    """Preflight and apply Neo4j Enterprise property-uniqueness constraints."""
    async with db_connection.get_session() as session:
        edition_result = await session.run(
            "CALL dbms.components() YIELD edition RETURN collect(edition) AS editions"
        )
        edition_record = await edition_result.single()
        editions = set(edition_record["editions"] if edition_record else ())
        if "enterprise" not in {str(item).casefold() for item in editions}:
            raise RuntimeError(
                "Semantic relationship uniqueness requires the deployed Neo4j Enterprise edition"
            )

    report = await preflight_semantic_identities(db_connection)
    async with db_connection.get_session() as session:
        result = await session.run(
            BACKFILL_SEMANTIC_LABEL_QUERY, labels=list(SEMANTIC_NODE_LABELS)
        )
        record = await result.single()
        report["semantic_nodes_labeled"] = int(record["count"] if record else 0)
        for statement in SEMANTIC_CONSTRAINT_STATEMENTS:
            result = await session.run(statement)
            await result.consume()
        result = await session.run(SHOW_SEMANTIC_CONSTRAINTS_QUERY)
        record = await result.single()
        constraints = list(record["constraints"] if record else ())
    actual = {_constraint_shape(item) for item in constraints}
    normalized = {
        (entity, "NODE PROPERTY UNIQUENESS" if entity == "NODE" and "UNIQUENESS" in kind
         else "RELATIONSHIP PROPERTY UNIQUENESS" if entity == "RELATIONSHIP" and "UNIQUENESS" in kind
         else kind, labels, properties)
        for entity, kind, labels, properties in actual
    }
    missing = required_semantic_constraint_shapes() - normalized
    if missing:
        raise RuntimeError(f"Neo4j did not install all semantic uniqueness constraints: {sorted(missing)}")
    report["constraints_verified"] = len(required_semantic_constraint_shapes())
    return report
