"""Conversation API for the shared Document Q&A agent and its on-demand skills."""

from __future__ import annotations

import logging
import uuid
from typing import Literal

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from src.graph.document_chat import (
    ChatConfigurationError,
    DocumentScopeError,
    get_ready_documents,
    validate_document_scope,
)

_LOGGER = logging.getLogger(__name__)
router = APIRouter(tags=["Document Chat"])


class ConversationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str | None = Field(default=None, max_length=160)
    doc_ids: list[str] | None = None


class ConversationRename(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=160)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conversation_id: str | None = None
    message: str = Field(min_length=1, max_length=10000)
    doc_ids: list[str] | None = None
    skill_hint: Literal["risk_analysis", "summarization"] | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    citations: list[dict]
    doc_ids: list[str]


def _runtime(request: Request):
    runtime = getattr(request.app.state, "chat_runtime", None)
    if runtime is None:
        raise HTTPException(
            status_code=503,
            detail="Document chat is unavailable; configure chat model and Supabase Postgres settings",
        )
    return runtime


async def _ready(request: Request, source_hashes: list[str] | tuple[str, ...] | None = None):
    reader = getattr(request.app.state, "chat_corpus_reader", None)
    db = getattr(request.app.state, "db", None)
    if reader is None or db is None:
        raise HTTPException(status_code=503, detail="Document readiness could not be verified")
    try:
        return await get_ready_documents(db, reader, source_hashes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ChatConfigurationError as exc:
        _LOGGER.exception("Chat corpus readiness validation failed")
        raise HTTPException(status_code=503, detail="Document ingestion status could not be verified") from exc


def _conversation_json(item):
    return {
        "conversation_id": item.id,
        "title": item.title,
        "doc_ids": list(item.doc_ids),
        "created_at": item.created_at,
        "updated_at": item.updated_at,
    }


@router.post("/chat/conversations", status_code=status.HTTP_201_CREATED)
async def create_conversation(request: Request, body: ConversationCreate):
    runtime = _runtime(request)
    ready = await _ready(request, body.doc_ids)
    try:
        scope = validate_document_scope(body.doc_ids, ready)
        if not scope:
            raise HTTPException(status_code=409, detail={"code": "NO_FULLY_INGESTED_DOCUMENTS"})
    except DocumentScopeError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "DOCUMENT_NOT_FULLY_INGESTED", "document_ids": list(exc.unavailable)},
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    title = (body.title or "New conversation").strip() or "New conversation"
    item = await runtime.persistence.store.create(str(uuid.uuid4()), title, scope)
    return _conversation_json(item)


@router.get("/chat/documents")
async def list_ready_documents(request: Request):
    _runtime(request)
    return {"documents": list(await _ready(request))}


@router.get("/chat/conversations")
async def list_conversations(request: Request):
    runtime = _runtime(request)
    return {"conversations": [_conversation_json(item) for item in await runtime.persistence.store.list()]}


@router.get("/chat/conversations/{conversation_id}")
async def get_conversation(request: Request, conversation_id: str):
    runtime = _runtime(request)
    try:
        item = await runtime.persistence.store.get(str(uuid.UUID(conversation_id)))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Conversation was not found") from exc
    if item is None:
        raise HTTPException(status_code=404, detail="Conversation was not found")
    try:
        messages = await runtime.history(item.id)
    except Exception as exc:
        _LOGGER.exception("Could not load chat checkpoint")
        raise HTTPException(status_code=503, detail="Conversation history could not be loaded") from exc
    return {**_conversation_json(item), "messages": list(messages)}


@router.patch("/chat/conversations/{conversation_id}")
async def rename_conversation(request: Request, conversation_id: str, body: ConversationRename):
    runtime = _runtime(request)
    if not body.title.strip():
        raise HTTPException(status_code=422, detail="Conversation title cannot be blank")
    try:
        identifier = str(uuid.UUID(conversation_id))
        item = await runtime.persistence.store.rename(identifier, body.title.strip())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Conversation was not found") from exc
    if item is None:
        raise HTTPException(status_code=404, detail="Conversation was not found")
    return _conversation_json(item)


@router.delete("/chat/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(request: Request, conversation_id: str):
    runtime = _runtime(request)
    try:
        identifier = str(uuid.UUID(conversation_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Conversation was not found") from exc
    try:
        existing = await runtime.persistence.store.get(identifier)
        if existing is None:
            raise HTTPException(status_code=404, detail="Conversation was not found")
        await runtime.persistence.checkpointer.adelete_thread(identifier)
        await runtime.persistence.store.delete(identifier)
    except HTTPException:
        raise
    except Exception as exc:
        _LOGGER.exception("Could not delete chat thread")
        raise HTTPException(status_code=503, detail="Conversation could not be deleted") from exc


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    runtime = _runtime(request)
    existing_item = None
    if body.conversation_id is not None:
        try:
            identifier = str(uuid.UUID(body.conversation_id))
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Conversation was not found") from exc
        existing_item = await runtime.persistence.store.get(identifier)
        if existing_item is None:
            raise HTTPException(status_code=404, detail="Conversation was not found")
    source_hashes = body.doc_ids if body.doc_ids is not None else (existing_item.doc_ids if existing_item else None)
    ready = await _ready(request, source_hashes)
    try:
        if existing_item is None:
            scope = validate_document_scope(body.doc_ids, ready)
            if not scope:
                raise HTTPException(status_code=409, detail={"code": "NO_FULLY_INGESTED_DOCUMENTS"})
            item = await runtime.persistence.store.create(
                str(uuid.uuid4()),
                body.message.strip()[:80] or "New conversation",
                scope,
            )
        else:
            item = existing_item
            identifier = item.id
            scope = validate_document_scope(body.doc_ids if body.doc_ids is not None else item.doc_ids, ready)
            if scope != item.doc_ids:
                item = await runtime.persistence.store.update_scope(identifier, scope)
                assert item is not None
    except DocumentScopeError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "DOCUMENT_NOT_FULLY_INGESTED", "document_ids": list(exc.unavailable)},
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        answer, citations = await runtime.run(
            conversation_id=item.id,
            message=body.message.strip(),
            doc_ids=scope,
            ready_documents=ready,
            skill_hint=body.skill_hint,
        )
        await runtime.persistence.store.touch(item.id)
        return ChatResponse(
            conversation_id=item.id,
            answer=answer,
            citations=list(citations),
            doc_ids=list(scope),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        _LOGGER.exception("Document chat request failed")
        raise HTTPException(
            status_code=503,
            detail="The answer could not be completed; retry the message in this conversation",
        ) from exc
