"""Supabase Postgres persistence for chat metadata and LangGraph checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit


def validate_chat_database_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
        raise ValueError("SUPABASE_DATABASE_URL must be a PostgreSQL connection URL for the existing Supabase project")
    if parsed.port == 6543:
        raise ValueError(
            "Use the Supabase direct or session-mode database connection, not the transaction pooler on port 6543"
        )
    return value


@dataclass(frozen=True)
class Conversation:
    id: str
    title: str
    doc_ids: tuple[str, ...]
    created_at: str
    updated_at: str


class ChatStore:
    """Metadata access through a private PostgreSQL pool; messages stay in checkpoints."""

    def __init__(self, pool: Any):
        self._pool = pool

    async def create(self, conversation_id: str, title: str, doc_ids: tuple[str, ...]) -> Conversation:
        async with self._pool.connection() as conn:
            row = await (await conn.execute(
                """INSERT INTO legal_chat.conversations (id, title, document_scope)
                   VALUES (%s::uuid, %s, %s::jsonb)
                   RETURNING id::text AS id, title, document_scope,
                             created_at::text AS created_at, updated_at::text AS updated_at""",
                (conversation_id, title, json.dumps(doc_ids)),
            )).fetchone()
        return _conversation(row)

    async def get(self, conversation_id: str) -> Conversation | None:
        async with self._pool.connection() as conn:
            row = await (await conn.execute(
                """SELECT id::text AS id, title, document_scope,
                          created_at::text AS created_at, updated_at::text AS updated_at
                   FROM legal_chat.conversations WHERE id = %s::uuid""",
                (conversation_id,),
            )).fetchone()
        return _conversation(row) if row else None

    async def list(self, limit: int = 100) -> tuple[Conversation, ...]:
        async with self._pool.connection() as conn:
            rows = await (await conn.execute(
                """SELECT id::text AS id, title, document_scope,
                          created_at::text AS created_at, updated_at::text AS updated_at
                   FROM legal_chat.conversations
                   ORDER BY updated_at DESC, id DESC LIMIT %s""",
                (limit,),
            )).fetchall()
        return tuple(_conversation(row) for row in rows)

    async def rename(self, conversation_id: str, title: str) -> Conversation | None:
        async with self._pool.connection() as conn:
            row = await (await conn.execute(
                """UPDATE legal_chat.conversations SET title = %s, updated_at = now()
                   WHERE id = %s::uuid
                   RETURNING id::text AS id, title, document_scope,
                             created_at::text AS created_at, updated_at::text AS updated_at""",
                (title, conversation_id),
            )).fetchone()
        return _conversation(row) if row else None

    async def update_scope(self, conversation_id: str, doc_ids: tuple[str, ...]) -> Conversation | None:
        async with self._pool.connection() as conn:
            row = await (await conn.execute(
                """UPDATE legal_chat.conversations
                   SET document_scope = %s::jsonb, updated_at = now()
                   WHERE id = %s::uuid
                   RETURNING id::text AS id, title, document_scope,
                             created_at::text AS created_at, updated_at::text AS updated_at""",
                (json.dumps(doc_ids), conversation_id),
            )).fetchone()
        return _conversation(row) if row else None

    async def touch(self, conversation_id: str) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "UPDATE legal_chat.conversations SET updated_at = now() WHERE id = %s::uuid",
                (conversation_id,),
            )

    async def delete(self, conversation_id: str) -> bool:
        async with self._pool.connection() as conn:
            cursor = await conn.execute(
                "DELETE FROM legal_chat.conversations WHERE id = %s::uuid", (conversation_id,)
            )
            return cursor.rowcount > 0


def _conversation(row: Any) -> Conversation:
    scope = row["document_scope"]
    return Conversation(
        id=row["id"], title=row["title"], doc_ids=tuple(scope),
        created_at=row["created_at"], updated_at=row["updated_at"],
    )


@dataclass
class ChatPersistence:
    store: ChatStore
    checkpointer: Any
    _checkpoint_connection: Any
    _pool: Any

    async def close(self) -> None:
        await self._pool.close()
        await self._checkpoint_connection.close()


async def open_chat_persistence(database_url: str) -> ChatPersistence:
    """Open the dedicated-schema checkpointer and metadata pool once at startup."""
    validate_chat_database_url(database_url)
    try:
        from psycopg import AsyncConnection
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    except ImportError as exc:
        raise RuntimeError(
            "Chat persistence needs langgraph-checkpoint-postgres and psycopg[binary,pool]"
        ) from exc

    connection_options = "-csearch_path=legal_chat,pg_catalog"
    pool = AsyncConnectionPool(
        conninfo=database_url,
        min_size=1,
        max_size=5,
        kwargs={"row_factory": dict_row, "options": connection_options},
        open=False,
    )
    await pool.open(wait=True)
    connection = None
    try:
        connection = await AsyncConnection.connect(
            database_url,
            autocommit=True,
            row_factory=dict_row,
            options=connection_options,
        )
        checkpointer = AsyncPostgresSaver(
            connection,
            serde=JsonPlusSerializer(
                pickle_fallback=False,
                allowed_msgpack_modules=None,
            ),
        )
        # The project migration provisions all checkpoint tables. Startup only
        # verifies them; it never performs schema changes.
        async with pool.connection() as conn:
            row = await (await conn.execute(
                "SELECT to_regclass('legal_chat.checkpoints') IS NOT NULL AS ready"
            )).fetchone()
        if row is None or not row["ready"]:
            raise RuntimeError(
                "Chat checkpoint schema is missing; apply Supabase migration 202611160001_document_chat.sql"
            )
        return ChatPersistence(ChatStore(pool), checkpointer, connection, pool)
    except Exception:
        await pool.close()
        if connection is not None:
            await connection.close()
        raise
