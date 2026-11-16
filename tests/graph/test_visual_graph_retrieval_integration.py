"""Opt-in, read-only verification of visual SEARCH Cypher on isolated Neo4j."""
from __future__ import annotations

import asyncio
import os

import pytest

from src.graph.retrieval import _VISUAL_GRAPH_QUERY
from src.infrastructure.neo4j_client import DbConnect


def _test_database():
    database = os.getenv("NEO4J_TEST_DATABASE", "").strip()
    if (not database or "test" not in database.casefold()
            or database.casefold() in {"rag", "neo4j"}):
        pytest.skip("set NEO4J_TEST_DATABASE to an isolated database name containing 'test'")
    values = {
        "url": os.getenv("NEO4J_URI"),
        "username": os.getenv("NEO4J_USERNAME"),
        "password": os.getenv("NEO4J_PASSWORD"),
    }
    if any(not value for value in values.values()):
        pytest.skip("configure Neo4j credentials for the isolated visual retrieval database")
    return DbConnect(database=database, **values)


@pytest.mark.neo4j_integration
def test_visual_relationship_and_semantic_search_query_runs_read_only():
    async def run():
        db = _test_database()
        try:
            await db.verify_connection()
            async with db.get_session() as session:
                indexes = await session.run(
                    "SHOW INDEXES YIELD name, type, state RETURN name, type, state"
                )
                states = {row["name"]: (row["type"], row["state"])
                          async for row in indexes}
                required = {
                    "visual_knowledge_fulltext": "FULLTEXT",
                    "visual_relationship_fulltext": "FULLTEXT",
                    "visual_semantic_entity_fulltext": "FULLTEXT",
                }
                missing = [name for name, kind in required.items()
                           if states.get(name) != (kind, "ONLINE")]
                if missing:
                    pytest.skip("visual search indexes are not ONLINE: " + ", ".join(missing))
                result = await session.run(
                    _VISUAL_GRAPH_QUERY,
                    search_query="phase6-query-syntax-probe-7f24d9",
                    candidate_count=1,
                    limit=1,
                )
                await result.consume()
        finally:
            await db.close()

    asyncio.run(run())
