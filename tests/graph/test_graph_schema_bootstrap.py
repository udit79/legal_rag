import asyncio
import os

import pytest

from src.domain.graph_schema import validate_node
from src.graph.schema_bootstrap import SCHEMA_STATEMENTS, initialize_graph_schema


class _Result:
    async def consume(self):
        return None


class _Session:
    def __init__(self):
        self.statements = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def run(self, statement):
        self.statements.append(statement)
        return _Result()


class _Db:
    def __init__(self):
        self.session = _Session()

    def get_session(self):
        return self.session


def test_document_contract_does_not_require_filename():
    assert validate_node("DOCUMENT", {"id": "a" * 64, "source_sha256": "a" * 64})


def test_schema_bootstrap_is_idempotent_and_contains_required_indexes():
    db = _Db()
    asyncio.run(initialize_graph_schema(db))
    assert db.session.statements == list(SCHEMA_STATEMENTS)
    assert all("IF NOT EXISTS" in statement for statement in SCHEMA_STATEMENTS)
    assert any("FULLTEXT INDEX" in statement and "retrieval_text" in statement
               for statement in SCHEMA_STATEMENTS)
    assert not any("VECTOR" in statement for statement in SCHEMA_STATEMENTS)


@pytest.mark.skipif(os.environ.get("NEO4J_INTEGRATION") != "1",
                    reason="set NEO4J_INTEGRATION=1 for configured Neo4j schema integration")
def test_configured_neo4j_schema_integration():
    async def run():
        from src.infrastructure.neo4j_client import create_db_connect

        db = create_db_connect()
        try:
            await db.verify_connection()
            await initialize_graph_schema(db)
        finally:
            await db.close()
    asyncio.run(run())
