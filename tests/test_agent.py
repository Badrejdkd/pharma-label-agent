"""Tests du graphe de l'agent avec un LLM scripté : on vérifie la mécanique, pas le modèle."""

from __future__ import annotations

import pytest

from app.agent.citations import renumber
from app.agent.graph import Agent
from app.agent.schemas import ClaimCheck, Plan, Review
from app.agent.tools import Toolbox
from app.retrieval import HybridIndex
from tests.fakes import FakeLLM

WARFARIN_SECTION = ("get_section", {"drug": "warfarin", "category": "contraindications", "source": "dailymed"})
CONTRA_ID = "dailymed-warfarin:11"  # section 4 de la notice de la warfarine


@pytest.fixture(scope="module")
def toolbox(index: HybridIndex) -> Toolbox:
    return Toolbox(index)


def make_plan(language: str = "English") -> Plan:
    return Plan(
        question_type="lookup",
        drugs=["warfarin"],
        sub_questions=["What are the contraindications of warfarin?"],
        answer_language=language,
    )


def make_review(verdict: str = "pass", needs_more_evidence: bool = False) -> Review:
    supported = verdict == "pass"
    return Review(
        claims=[
            ClaimCheck(
                claim="Warfarin is contraindicated in pregnancy.",
                verdict="supported" if supported else "unsupported",
                evidence_ids=[CONTRA_ID] if supported else [],
                comment="" if supported else "Not stated in the cited passage.",
            )
        ],
        verdict=verdict,  # type: ignore[arg-type]
        revision_instructions="" if supported else "Remove the unsupported claim.",
        needs_more_evidence=needs_more_evidence,
        clinical_flags=[],
    )


def test_simple_question_goes_through_every_step(toolbox: Toolbox) -> None:
    llm = FakeLLM(
        plans=[make_plan()],
        agent_turns=[{"tool_calls": [WARFARIN_SECTION]}],
        drafts=[f"Warfarin is contraindicated in pregnancy [{CONTRA_ID}]."],
        reviews=[make_review("pass")],
    )
    result = Agent(llm, toolbox).run("What are the main contraindications for warfarin?")

    assert llm.calls == ["Plan", "agent", "agent", "write", "Review"]
    assert llm.tool_choices == ["required", "auto"]  # premier tour : outil obligatoire
    assert result.answer == "Warfarin is contraindicated in pregnancy [1]."
    assert result.citations[0].file == "dailymed-warfarin.md"
    assert result.tool_calls[0].tool == "get_section"
    assert result.review.verdict == "pass"
    assert "not medical advice" in result.disclaimer
    assert result.meta["llm_calls"] == 5


def test_invented_citation_is_caught_without_calling_the_critic(toolbox: Toolbox) -> None:
    llm = FakeLLM(
        plans=[make_plan()],
        agent_turns=[{"tool_calls": [WARFARIN_SECTION]}],
        drafts=[
            "Warfarin is contraindicated in pregnancy [dailymed-warfarin:999].",
            f"Warfarin is contraindicated in pregnancy [{CONTRA_ID}].",
        ],
        reviews=[make_review("pass")],
    )
    result = Agent(llm, toolbox).run("Contraindications of warfarin?")

    assert llm.calls.count("write") == 2
    assert llm.calls.count("Review") == 1  # le premier brouillon est rejeté par le code
    assert result.review.verdict == "revised"
    assert result.answer.endswith("[1].")


def test_unsupported_claims_are_flagged_after_one_revision(toolbox: Toolbox) -> None:
    llm = FakeLLM(
        plans=[make_plan()],
        agent_turns=[{"tool_calls": [WARFARIN_SECTION]}],
        drafts=[f"Draft one [{CONTRA_ID}].", f"Draft two [{CONTRA_ID}]."],
        reviews=[make_review("revise"), make_review("revise")],
    )
    result = Agent(llm, toolbox).run("Contraindications of warfarin?")

    assert result.meta["revisions"] == 1
    assert result.review.verdict == "flagged"
    assert "could not verify" in result.review.flags[0]


def test_critic_can_send_the_agent_back_to_research(toolbox: Toolbox) -> None:
    llm = FakeLLM(
        plans=[make_plan()],
        agent_turns=[
            {"tool_calls": [WARFARIN_SECTION]},
            {"content": "Found the contraindications."},
            {"tool_calls": [("find_interactions", {"drug_a": "warfarin", "drug_b": "sertraline"})]},
            {"content": "Found the interaction."},
        ],
        drafts=[f"First draft [{CONTRA_ID}].", f"Second draft [{CONTRA_ID}]."],
        reviews=[make_review("revise", needs_more_evidence=True), make_review("pass")],
    )
    result = Agent(llm, toolbox).run("Warfarin contraindications and sertraline interaction?")

    assert [trace.tool for trace in result.tool_calls] == ["get_section", "find_interactions"]
    assert result.review.verdict == "revised"


def test_research_loop_is_bounded(toolbox: Toolbox) -> None:
    llm = FakeLLM(
        plans=[make_plan()],
        agent_turns=[{"tool_calls": [WARFARIN_SECTION]}] * 10,  # le modèle ne s'arrêterait jamais
        drafts=[f"Answer [{CONTRA_ID}]."],
        reviews=[make_review("pass")],
    )
    result = Agent(llm, toolbox, max_steps=3).run("Warfarin?")

    assert result.meta["agent_steps"] == 3
    assert len(result.tool_calls) == 3


def test_tool_errors_are_traced_and_answer_still_produced(toolbox: Toolbox) -> None:
    llm = FakeLLM(
        plans=[make_plan()],
        agent_turns=[
            {
                "tool_calls": [
                    ("get_section", {"drug": "rivaroxaban", "category": "contraindications", "source": "dailymed"})
                ]
            }
        ],
        drafts=["The corpus does not cover rivaroxaban."],
        reviews=[make_review("pass")],
    )
    result = Agent(llm, toolbox).run("Contraindications of rivaroxaban?")

    assert result.tool_calls[0].error is not None
    assert result.citations == []
    assert result.answer == "The corpus does not cover rivaroxaban."


def test_french_question_gets_a_french_disclaimer(toolbox: Toolbox) -> None:
    llm = FakeLLM(
        plans=[make_plan("French")],
        agent_turns=[{"tool_calls": [WARFARIN_SECTION]}],
        drafts=[f"La warfarine est contre-indiquée pendant la grossesse [{CONTRA_ID}]."],
        reviews=[make_review("pass")],
    )
    result = Agent(llm, toolbox).run("Quelles sont les contre-indications de la warfarine ?")
    assert "pas un avis médical" in result.disclaimer


def test_renumber_reuses_numbers_and_drops_unknown_ids(toolbox: Toolbox) -> None:
    evidence = {c.chunk_id: c for c in toolbox.index.chunks if c.chunk_id in {CONTRA_ID, "dailymed-warfarin:24"}}
    text, citations = renumber(
        f"A [{CONTRA_ID}]. B [dailymed-warfarin:24][{CONTRA_ID}]. C [made-up:1].", evidence
    )
    assert text == "A [1]. B [2][1]. C ."
    assert [c.chunk_id for c in citations] == [CONTRA_ID, "dailymed-warfarin:24"]