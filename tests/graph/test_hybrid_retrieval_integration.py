"""Read-only Cypher 25 checks against Neo4j 2026.10.2."""

import asyncio
import os
import re

import pytest
from dotenv import load_dotenv
from neo4j.exceptions import Neo4jError

from src.graph.retrieval import (
    _FULLTEXT_QUERY,
    _VECTOR_QUERY_TEMPLATE,
)
from src.graph.structural_writer import CURRENT_CHUNK_RETRIEVAL_QUERY

pytestmark = pytest.mark.neo4j_integration
EXPECTED_NEO4J_VERSION = "2026.10.2"


def _connection_settings():
    load_dotenv(".env")
    required = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_TEST_DATABASE"]
    if any(not os.getenv(name) for name in required):
        pytest.skip("configure NEO4J_TEST_DATABASE and Neo4j credentials for read-only integration tests")
    if os.getenv("NEO4J_TEST_DATABASE") == os.getenv("NEO4J_DATABASE", "neo4j"):
        pytest.fail("NEO4J_TEST_DATABASE must be distinct from NEO4J_DATABASE")
    return {
        "url": os.environ["NEO4J_URI"],
        "username": os.environ["NEO4J_USERNAME"],
        "password": os.environ["NEO4J_PASSWORD"],
        "database": os.environ["NEO4J_TEST_DATABASE"],
    }


async def _db_version(db) -> str:
    async with db.get_session() as session:
        result = await session.run(
            "CALL dbms.components() YIELD versions "
            "RETURN versions[0] AS version LIMIT 1"
        )
        record = await result.single()
        return record["version"]


def test_neo4j_cypher25_target_is_2026_10_2():
    async def run():
        from src.infrastructure.neo4j_client import DbConnect

        db = DbConnect(**_connection_settings())
        try:
            assert await _db_version(db) == EXPECTED_NEO4J_VERSION
        finally:
            await db.close()

    asyncio.run(run())


def test_search_queries_parse_or_execute_read_only_on_neo4j_2026_10_2():
    async def run():
        from src.infrastructure.neo4j_client import DbConnect

        db = DbConnect(**_connection_settings())
        try:
            version = await _db_version(db)
            assert version == EXPECTED_NEO4J_VERSION
            async with db.get_session() as session:
                result = await session.run(
                    "SHOW INDEXES YIELD name, type, state, options "
                    "RETURN name, type, state, options"
                )
                indexes = await result.data()

            fulltext = next((row for row in indexes
                             if row["name"] == "chunk_retrieval_text_fulltext"
                             and row["type"] == "FULLTEXT"), None)
            vectors = [row for row in indexes
                       if row["type"] == "VECTOR"
                       and re.fullmatch(r"chunk_embedding_[0-9a-f]{16}", row["name"])]
            fulltext_online = fulltext is not None and fulltext["state"] == "ONLINE"
            vector_online = next((item for item in vectors if item["state"] == "ONLINE"), None)
            vector_index = vector_online or {
                "name": "chunk_embedding_0000000000000000", "options": {}, "state": "ABSENT"
            }
            options = vector_index.get("options") or {}
            index_config = options.get("indexConfig") or {}
            dimensions = index_config.get("vector.dimensions")
            metadata = None
            if vector_online is not None and dimensions is None:
                async with db.get_session() as session:
                    result = await session.run(
                        "MATCH (c:CHUNK) WHERE c.embedding_space_id IS NOT NULL "
                        "AND c.embedding_model IS NOT NULL AND c.embedding_dimensions IS NOT NULL "
                        "RETURN c.embedding_space_id AS space_id, c.embedding_model AS model, "
                        "c.embedding_dimensions AS dimensions LIMIT 1"
                    )
                    metadata = await result.single()
            if dimensions is None and metadata is not None:
                dimensions = metadata["dimensions"]
            if type(dimensions) is not int or dimensions < 1:
                dimensions = metadata["dimensions"] if metadata else 2

            if fulltext is not None and not fulltext_online:
                pytest.skip("full-text index exists but is not ONLINE")
            if vectors and vector_online is None:
                pytest.skip("vector index exists but none are ONLINE")

            async with db.get_session() as session:
                for query, params in (
                    (_FULLTEXT_QUERY, {
                        "search_query": "read only syntax validation",
                        "candidate_count": 10,
                        "limit": 2,
                    }),
                    (CURRENT_CHUNK_RETRIEVAL_QUERY, {
                        "query": "read only syntax validation",
                        "candidate_count": 10,
                        "limit": 2,
                    }),
                ):
                    try:
                        result = await session.run(query, parameters=params)
                        await result.consume()
                        assert fulltext_online
                    except Neo4jError as exc:
                        # A missing index is a semantic prerequisite failure
                        # after Cypher 25 parsed SEARCH.
                        assert fulltext is None
                        assert exc.code == "Neo.ClientError.Schema.IndexNotFound"

                vector_query = _VECTOR_QUERY_TEMPLATE.format(index_name=vector_index["name"])
                try:
                    result = await session.run(
                        vector_query,
                        parameters={
                            "query_vector": [1.0] + [0.0] * (dimensions - 1),
                            "embedding_space_id": metadata["space_id"] if metadata else "no-match-space",
                            "embedding_model": metadata["model"] if metadata else "no-match-model",
                            "embedding_dimensions": dimensions,
                            "candidate_count": 10,
                            "limit": 2,
                        },
                    )
                    await result.consume()
                    assert vector_online is not None
                except Neo4jError as exc:
                    assert vector_online is None
                    assert exc.code == "Neo.ClientError.Schema.IndexNotFound"
        finally:
            await db.close()

    asyncio.run(run())
