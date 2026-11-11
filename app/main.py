"""API FastAPI : santé, questions (réponse complète ou en direct) et interface web.

    GET  /health      -> 200, état de l'index
    POST /ask         -> 200, réponse complète (AgentResult)
    POST /ask/stream  -> 200, événements Server-Sent Events, un par étape de l'agent
    GET  /            -> interface web

Lancement local : uv run uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, StringConstraints

from app.agent.graph import Agent
from app.agent.schemas import AgentResult
from app.bootstrap import build_agent
from app.llm import LLMError

STATIC_DIR = Path(__file__).parent / "static"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
for noisy in ("httpx", "httpx2"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=1000)]


class HealthResponse(BaseModel):
    status: str
    model: str
    passages: int
    drugs: list[str]
    semantic_search: bool


def sse(event: dict[str, Any]) -> str:
    """Formate un événement Server-Sent Events."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def create_app(agent_factory: Callable[[], Agent] = build_agent) -> FastAPI:
    """Crée l'application ; les tests passent une fabrique d'agent avec un faux LLM."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # L'index est construit avant que le serveur accepte des requêtes :
        # quand /health répond, /ask est prêt.
        app.state.agent = agent_factory()
        yield

    app = FastAPI(
        title="Drug label research agent",
        description="Agentic question answering over FDA drug labels and MedlinePlus leaflets.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

    def get_agent(request: Request) -> Agent:
        return request.app.state.agent

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        agent = get_agent(request)
        index = agent.toolbox.index
        return HealthResponse(
            status="ok",
            model=agent.llm.model,
            passages=len(index.chunks),
            drugs=index.drugs,
            semantic_search=index.semantic_enabled,
        )

    @app.post("/ask", response_model=AgentResult)
    def ask(body: AskRequest, request: Request) -> AgentResult:
        try:
            return get_agent(request).run(body.question)
        except LLMError as exc:
            logger.error("Échec du LLM : %s", exc)
            raise HTTPException(status_code=502, detail=f"The language model could not be reached: {exc}") from exc

    @app.post("/ask/stream")
    def ask_stream(body: AskRequest, request: Request) -> StreamingResponse:
        agent = get_agent(request)

        def events() -> Iterator[str]:
            try:
                for event in agent.stream(body.question):
                    yield sse(event)
            except LLMError as exc:
                logger.error("Échec du LLM : %s", exc)
                yield sse({"type": "error", "message": f"The language model could not be reached: {exc}"})
            except Exception:  # l'interface doit toujours recevoir un message d'erreur lisible
                logger.exception("Erreur pendant le traitement de la question")
                yield sse({"type": "error", "message": "Internal error while answering. See the server logs."})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()