import asyncio

import pytest

from src.graph.semantic_schema import (
    SEMANTIC_CONSTRAINT_STATEMENTS,
    SemanticSchemaPreflightError,
    preflight_semantic_identities,
    required_semantic_constraint_shapes,
)


class Result:
    def __init__(self, records):
        self.records = records

    def __aiter__(self):
        async def iterate():
            for record in self.records:
                yield record
        return iterate()

    async def single(self):
        return self.records[0] if self.records else None


class Session:
    def __init__(self, *, node_issues=(), relationship_issues=(), node_count=0):
        self.node_issues = list(node_issues)
        self.relationship_issues = list(relationship_issues)
        self.node_count = node_count

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def run(self, query, **params):
        if "WITH n.id AS id" in query:
            return Result(self.node_issues)
        if "WITH type(r) AS type" in query:
            return Result(self.relationship_issues)
        return Result([{"count": self.node_count}])


class Db:
    def __init__(self, **kwargs):
        self.session = Session(**kwargs)

    def get_session(self):
        return self.session


def test_schema_uses_enterprise_unique_keys_for_semantic_nodes_and_relations():
    shapes = required_semantic_constraint_shapes()
    assert ("NODE", "NODE PROPERTY UNIQUENESS", ("SEMANTIC_ENTITY",), ("id",)) in shapes
    assert ("NODE", "NODE PROPERTY UNIQUENESS", ("KNOWLEDGE_ASSERTION",), ("id",)) in shapes
    assert any("FOR ()-[r:OWNS]-() REQUIRE r.id IS UNIQUE" in item
               for item in SEMANTIC_CONSTRAINT_STATEMENTS)
    assert any("ASSERTION_CLASSIFICATION" in item for item in SEMANTIC_CONSTRAINT_STATEMENTS)


@pytest.mark.parametrize("kwargs", [
    {"node_issues": [{"id": "duplicate", "count": 2}]},
    {"relationship_issues": [{"type": "OWNS", "id": None, "count": 1}]},
])
def test_duplicate_preflight_fails_closed(kwargs):
    with pytest.raises(SemanticSchemaPreflightError):
        asyncio.run(preflight_semantic_identities(Db(**kwargs)))


def test_clean_duplicate_preflight_reports_counts():
    assert asyncio.run(preflight_semantic_identities(Db(node_count=3))) == {
        "semantic_nodes_to_label": 3,
        "node_identity_issues": 0,
        "relationship_identity_issues": 0,
    }
