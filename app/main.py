from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router, websocket_chat
from app.config import get_settings
from app.container import build_container


def create_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.container = build_container(settings)
        yield

    app = FastAPI(
        title="FiberMind API",
        version="0.1.0",
        description="面向光网络运维的可解释 RAG 故障诊断服务",
        lifespan=lifespan,
    )
    app.include_router(router)
    app.add_api_websocket_route("/ws/chat/{session_id}", websocket_chat)
    app.mount("/static", StaticFiles(directory=settings.static_path), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(settings.static_path / "index.html")

    return app


app = create_app()
