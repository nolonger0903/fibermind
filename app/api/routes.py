from __future__ import annotations

from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, Request, WebSocket, WebSocketDisconnect

from app.container import Container
from app.schemas.api import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IntentRequest,
    IntentView,
    MessageView,
    SearchRequest,
)

router = APIRouter(prefix="/api/v1")


def get_container(request: Request) -> Container:
    return request.app.state.container


ContainerDep = Annotated[Container, Depends(get_container)]


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health(container: ContainerDep) -> dict:
    try:
        cache_ok = await container.cache.ping()
    except Exception:
        cache_ok = False
    return {
        "status": "ok" if cache_ok else "degraded",
        "mode": container.settings.app_env,
        "services": {
            "cache": f"{container.settings.cache_backend}:{'up' if cache_ok else 'down'}",
            "sessions": container.settings.session_backend,
            "retrieval": container.settings.retrieval_backend,
            "rerank": container.settings.rerank_backend,
            "embedding": container.settings.embedding_backend,
            "intent": container.settings.intent_backend,
            "llm": container.settings.llm_backend,
        },
    }


@router.post("/chat", response_model=ChatResponse, tags=["diagnosis"])
async def chat(payload: ChatRequest, container: ContainerDep) -> dict:
    return await container.pipeline.run(payload.query, payload.session_id, payload.filters)


@router.post("/diagnose", response_model=ChatResponse, tags=["diagnosis"])
async def diagnose(payload: ChatRequest, container: ContainerDep) -> dict:
    return await container.pipeline.run(payload.query, payload.session_id, payload.filters)


@router.post("/intent/predict", response_model=IntentView, tags=["intent"])
async def predict_intent(payload: IntentRequest, container: ContainerDep) -> dict:
    result = container.intent_classifier.predict(payload.query)
    return {
        "name": result.intent.value,
        "confidence": result.confidence,
        "classifier": result.classifier,
        "entities": result.entities,
    }


@router.post("/knowledge/search", tags=["knowledge"])
async def search(payload: SearchRequest, container: ContainerDep) -> dict:
    candidates = await container.retriever.search(payload.query, payload.top_k, payload.filters)
    reranked = await container.reranker.rerank(
        payload.query, candidates, min(container.settings.rerank_top_n, payload.top_k)
    )
    return {
        "query": payload.query,
        "total": len(reranked),
        "results": [
            {
                "chunk": asdict(item.chunk),
                "scores": {
                    "dense": round(item.dense_score, 4),
                    "sparse": round(item.sparse_score, 4),
                    "fusion": round(item.fusion_score, 4),
                    "rerank": round(item.rerank_score, 4),
                },
            }
            for item in reranked
        ],
    }


@router.get("/sessions/{session_id}/messages", response_model=list[MessageView], tags=["session"])
async def session_messages(session_id: str, container: ContainerDep) -> list:
    return [asdict(item) for item in await container.sessions.list_messages(session_id, limit=100)]


@router.get("/metrics", tags=["system"])
async def metrics(container: ContainerDep) -> dict:
    cache_total = container.cache.hits + container.cache.misses
    return {
        "requests_total": container.pipeline.request_count,
        "routes": container.pipeline.route_counts,
        "cache": {
            "hits": container.cache.hits,
            "misses": container.cache.misses,
            "hit_rate": round(container.cache.hits / cache_total, 4) if cache_total else 0,
        },
        "knowledge": {
            "documents": len(container.knowledge.list_documents()),
            "chunks": container.chunks_count,
        },
        "sessions": await container.sessions.count_sessions(),
    }


async def websocket_chat(websocket: WebSocket) -> None:
    await websocket.accept()
    container: Container = websocket.app.state.container
    session_id = websocket.path_params["session_id"]
    try:
        while True:
            payload = await websocket.receive_json()
            query = str(payload.get("query", "")).strip()
            if not query:
                await websocket.send_json(
                    {"event": "error", "data": {"message": "query is required"}}
                )
                continue

            async def emit(event: str, data: dict) -> None:
                await websocket.send_json({"event": event, "data": data})

            try:
                response = await container.pipeline.run(
                    query=query,
                    session_id=session_id,
                    filters=payload.get("filters"),
                    emit=emit,
                )
                await websocket.send_json({"event": "answer_completed", "data": response})
            except Exception as exc:
                await websocket.send_json(
                    {"event": "error", "data": {"message": str(exc), "type": type(exc).__name__}}
                )
    except WebSocketDisconnect:
        return
