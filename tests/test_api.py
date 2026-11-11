"""Tests de l'API avec un agent branché sur un faux LLM (aucun réseau)."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from app.agent.graph import Agent
from app.agent.schemas import ClaimCheck, Plan, Review
from app.agent.tools import Toolbox
from app.llm import LLMError
from app.main import create_app
from app.retrieval import HybridIndex
from tests.fakes import FakeLLM

CONTRA_ID = "dailymed-warfarin:11"
WARFARIN_SECTION = ("get_section", {"drug": "warfarin", "category": "contraindications", "source": "dailymed"})


def scripted_llm() -> FakeLLM:
    return FakeLLM(
        plans=[
            Plan(
                question_type="lookup",
                drugs=["warfarin"],
                sub_questions=["What are the contraindications of warfarin?"],
                answer_language="English",
            )
        ],
        agent_turns=[{"tool_calls": [WARFARIN_SECTION]}],
        drafts=[f"Warfarin is contraindicated in pregnancy [{CONTRA_ID}]."],
        reviews=[
            Review(
                claims=[
                    ClaimCheck(claim="Pregnancy", verdict="supported", evidence_ids=[CONTRA_ID], comment="")
                ],
                verdict="pass",
                revision_instructions="",
                needs_more_evidence=False,
                clinical_flags=[],
            )
        ],
    )


class BrokenLLM(FakeLLM):
    """Simule un proxy LiteLLM injoignable."""

    def structured(self, messages: list[dict[str, Any]], output_type: type) -> Any:
        raise LLMError("connection refused")


ClientFactory = Callable[[FakeLLM], TestClient]


@pytest.fixture
def make_client(index: HybridIndex) -> Iterator[ClientFactory]:
    """Fabrique de clients HTTP ; `with TestClient(...)` déclenche le lifespan (construction de l'agent)."""
    with ExitStack() as stack:

        def factory(llm: FakeLLM) -> TestClient:
            return stack.enter_context(TestClient(create_app(lambda: Agent(llm, Toolbox(index)))))

        yield factory


def test_health(make_client: ClientFactory) -> None:
    response = make_client(scripted_llm()).get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["passages"] == 500
    assert "warfarin" in body["drugs"]


def test_ask_returns_a_cited_answer(make_client: ClientFactory) -> None:
    response = make_client(scripted_llm()).post("/ask", json={"question": "Contraindications of warfarin?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Warfarin is contraindicated in pregnancy [1]."
    assert body["citations"][0]["file"] == "dailymed-warfarin.md"
    assert "not medical advice" in body["disclaimer"]


def test_ask_rejects_an_empty_question(make_client: ClientFactory) -> None:
    response = make_client(scripted_llm()).post("/ask", json={"question": "  "})
    assert response.status_code == 422


def test_ask_reports_llm_failures_as_502(make_client: ClientFactory) -> None:
    response = make_client(BrokenLLM([], [], [])).post("/ask", json={"question": "Contraindications of warfarin?"})
    assert response.status_code == 502
    assert "could not be reached" in response.json()["detail"]


def read_events(response: httpx.Response) -> list[dict[str, Any]]:
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def test_stream_publishes_each_step_then_the_result(make_client: ClientFactory) -> None:
    response = make_client(scripted_llm()).post("/ask/stream", json={"question": "Contraindications of warfarin?"})
    assert response.status_code == 200
    events = read_events(response)
    nodes = [event["node"] for event in events if event["type"] == "step"]
    assert nodes == ["plan", "agent", "tools", "agent", "write", "critique"]
    assert events[-1]["type"] == "result"
    assert events[-1]["result"]["citations"][0]["chunk_id"] == CONTRA_ID


def test_stream_reports_llm_failures_as_an_event(make_client: ClientFactory) -> None:
    response = make_client(BrokenLLM([], [], [])).post("/ask/stream", json={"question": "Warfarin?"})
    assert read_events(response)[-1]["type"] == "error"


def test_home_page_is_served(make_client: ClientFactory) -> None:
    response = make_client(scripted_llm()).get("/")
    assert response.status_code == 200
    assert "Drug label research agent" in response.text