"""FastAPI service: POST /ask returns an answer with validated page-level citations."""

from __future__ import annotations

import time
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from paper_assistant import config
from paper_assistant.pipeline import Assistant

app = FastAPI(title="Research Paper Assistant", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    k: int = Field(default=config.TOP_K, ge=1, le=10)


class CitationOut(BaseModel):
    paper_id: str
    page: int
    title: str = ""
    url: str = ""


class SourceOut(BaseModel):
    paper_id: str
    page: int
    title: str = ""
    text: str


class AskResponse(BaseModel):
    answer: str
    answerable: bool
    citations: list[CitationOut]
    sources: list[SourceOut]
    latency_ms: int


@lru_cache(maxsize=1)
def get_assistant() -> Assistant:
    return Assistant()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, assistant: Assistant = Depends(get_assistant)) -> AskResponse:
    started = time.perf_counter()
    try:
        result = assistant.ask(request.question, k=request.k)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=f"Index not built: {error}") from error
    except Exception as error:  # noqa: BLE001 - surface API/auth problems as a clean 502
        raise HTTPException(status_code=502, detail=f"Generation failed: {error}") from error

    titles = assistant.titles
    return AskResponse(
        answer=result.answer,
        answerable=result.answerable,
        citations=[
            CitationOut(
                paper_id=c.paper_id,
                page=c.page,
                title=titles.get(c.paper_id, ""),
                url=f"https://arxiv.org/pdf/{c.paper_id}#page={c.page}",
            )
            for c in result.citations
        ],
        sources=[
            SourceOut(
                paper_id=h.chunk.paper_id,
                page=h.chunk.page,
                title=titles.get(h.chunk.paper_id, ""),
                text=h.chunk.text,
            )
            for h in result.hits
        ],
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
