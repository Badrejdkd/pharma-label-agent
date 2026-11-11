"""Graphe LangGraph de l'agent : planifier -> chercher (outils) -> rédiger -> critiquer.

    START -> plan -> agent <-> tools
                       |
                       v
                     write -> critique -> finalize -> END
                       ^         |
                       +---------+  révision (1 fois) ; retour à "agent" si des preuves manquent

- plan     : le LLM décompose la question (sortie JSON validée par Pydantic) ;
- agent    : le LLM choisit les outils à appeler (function calling) ;
- tools    : le code exécute les appels et garde les passages consultés ;
- write    : le LLM rédige une réponse qui ne cite que ces passages ;
- critique : contrôles déterministes, puis vérification affirmation par affirmation par le LLM ;
- finalize : renumérotation des citations et ajout de l'avertissement, par le code.
"""

from __future__ import annotations
from collections.abc import Iterator
import logging
import operator
import time
from typing import Annotated, Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from app.agent import prompts
from app.agent.citations import cited_ids, disclaimer_for, renumber, unknown_citations
from app.agent.schemas import AgentResult, Plan, Review, ReviewSummary, ToolCallTrace
from app.agent.tools import Toolbox
from app.ingest import Chunk

logger = logging.getLogger(__name__)

MAX_REVISIONS = 1  # nombre de réécritures autorisées après la première critique
MAX_RESEARCH_RETRIES = 1  # retours du critique vers la recherche
RUN_CONFIG = {"recursion_limit": 60}  # filet de sécurité de LangGraph contre les boucles

class LLM(Protocol):
    """Ce dont l'agent a besoin : `LLMClient` le respecte, le faux LLM des tests aussi."""

    model: str

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, tool_choice: str = "auto"
    ) -> Any: ...

    def structured(self, messages: list[dict[str, Any]], output_type: type[BaseModel]) -> Any: ...


def merge_evidence(left: dict[str, Chunk], right: dict[str, Chunk]) -> dict[str, Chunk]:
    """Réducteur LangGraph : ajoute les nouveaux passages sans perdre les anciens."""
    return {**left, **right}


class AgentState(TypedDict, total=False):
    """État partagé entre les nœuds. `Annotated[..., réducteur]` = valeurs cumulées."""

    question: str
    plan: Plan
    messages: Annotated[list[dict[str, Any]], operator.add]
    evidence: Annotated[dict[str, Chunk], merge_evidence]
    tool_calls: Annotated[list[ToolCallTrace], operator.add]
    agent_steps: Annotated[int, operator.add]
    llm_calls: Annotated[int, operator.add]
    draft: str
    review: Review
    revisions: int
    research_retries: int
    retry_research: bool
    result: AgentResult


def format_evidence(chunks: list[Chunk]) -> str:
    """Passages présentés au rédacteur et au critique."""
    return "\n\n".join(f"[{c.chunk_id}] ({c.file} | {c.section})\n{c.text}" for c in chunks)

def describe_step(node: str, update: dict[str, Any]) -> dict[str, Any]:
    """Résumé d'une étape du graphe, envoyé à l'interface pendant le calcul."""
    if node == "plan":
        plan: Plan = update["plan"]
        return {"question_type": plan.question_type, "sub_questions": plan.sub_questions}
    if node == "agent":
        calls = update["messages"][-1].get("tool_calls", [])
        return {"requested_tools": [call["function"]["name"] for call in calls]}
    if node == "tools":
        return {"calls": [trace.model_dump(mode="json") for trace in update.get("tool_calls", [])]}
    if node == "write":
        return {"revision": update.get("revisions", 0)}
    if node == "critique":
        review: Review = update["review"]
        return {"verdict": review.verdict, "checked_by_llm": "llm_calls" in update}
    return {}

class Agent:
    """Assemble le graphe et l'exécute pour une question."""

    def __init__(self, llm: LLM, toolbox: Toolbox, max_steps: int = 6) -> None:
        self.llm = llm
        self.toolbox = toolbox
        self.max_steps = max_steps
        self.graph = self._build()

    # ------------------------------------------------------------------ #
    # Construction du graphe
    # ------------------------------------------------------------------ #
    def _build(self) -> Any:
        graph = StateGraph(AgentState)
        graph.add_node("plan", self.plan)
        graph.add_node("agent", self.agent)
        graph.add_node("tools", self.tools)
        graph.add_node("write", self.write)
        graph.add_node("critique", self.critique)
        graph.add_node("finalize", self.finalize)

        graph.add_edge(START, "plan")
        graph.add_edge("plan", "agent")
        graph.add_conditional_edges("agent", self.after_agent, {"tools": "tools", "write": "write"})
        graph.add_conditional_edges("tools", self.after_tools, {"agent": "agent", "write": "write"})
        graph.add_edge("write", "critique")
        graph.add_conditional_edges(
            "critique", self.after_critique, {"finalize": "finalize", "write": "write", "agent": "agent"}
        )
        graph.add_edge("finalize", END)
        return graph.compile()

    def run(self, question: str) -> AgentResult:
        """Répond à une question : exécute le graphe du début à la fin."""
        start = time.perf_counter()
        state = self.graph.invoke({"question": question}, config=RUN_CONFIG)
        result: AgentResult = state["result"]
        result.meta["latency_ms"] = round((time.perf_counter() - start) * 1000)
        return result

    def stream(self, question: str) -> Iterator[dict[str, Any]]:
        """Comme `run`, mais publie un événement après chaque nœud (pour l'interface en direct)."""
        start = time.perf_counter()
        for update in self.graph.stream({"question": question}, config=RUN_CONFIG, stream_mode="updates"):
            for node, data in update.items():
                if node == "finalize":
                    result: AgentResult = data["result"]
                    result.meta["latency_ms"] = round((time.perf_counter() - start) * 1000)
                    yield {"type": "result", "result": result.model_dump(mode="json")}
                else:
                    yield {"type": "step", "node": node, "detail": describe_step(node, data)}

    # ------------------------------------------------------------------ #
    # Nœuds
    # ------------------------------------------------------------------ #
    def plan(self, state: AgentState) -> dict[str, Any]:
        overview = self.toolbox.corpus_overview()
        plan = self.llm.structured(
            [
                {"role": "system", "content": prompts.PLANNER.format(corpus_overview=overview)},
                {"role": "user", "content": state["question"]},
            ],
            Plan,
        )
        request = (
            f"Question: {state['question']}\n\nResearch plan:\n"
            + "\n".join(f"- {q}" for q in plan.sub_questions)
            + f"\nDrugs mentioned: {', '.join(plan.drugs) or 'none'}"
        )
        messages = [
            {"role": "system", "content": prompts.RESEARCHER.format(corpus_overview=overview)},
            {"role": "user", "content": request},
        ]
        return {"plan": plan, "messages": messages, "llm_calls": 1, "revisions": 0, "research_retries": 0}

    def agent(self, state: AgentState) -> dict[str, Any]:
        # Premier tour : au moins un outil, pour que la réponse s'appuie toujours sur le corpus.
        tool_choice = "required" if state.get("agent_steps", 0) == 0 else "auto"
        message = self.llm.chat(state["messages"], tools=self.toolbox.specs(), tool_choice=tool_choice)
        entry: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }
                for call in message.tool_calls
            ]
        return {"messages": [entry], "agent_steps": 1, "llm_calls": 1}

    def tools(self, state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        messages: list[dict[str, Any]] = []
        evidence: dict[str, Chunk] = {}
        traces: list[ToolCallTrace] = []
        for call in last.get("tool_calls", []):
            result = self.toolbox.execute(call["function"]["name"], call["function"]["arguments"])
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result.content})
            evidence.update({chunk.chunk_id: chunk for chunk in result.chunks})
            traces.append(
                ToolCallTrace(
                    tool=result.name,
                    arguments=result.arguments,
                    passages=[chunk.chunk_id for chunk in result.chunks],
                    error=result.error,
                    duration_ms=result.duration_ms,
                )
            )
            logger.info("Outil %s(%s) -> %d passages", result.name, result.arguments, len(result.chunks))
        return {"messages": messages, "evidence": evidence, "tool_calls": traces}

    def write(self, state: AgentState) -> dict[str, Any]:
        plan = state["plan"]
        evidence = list(state.get("evidence", {}).values())
        messages = [
            {"role": "system", "content": prompts.WRITER.format(language=plan.answer_language)},
            {
                "role": "user",
                "content": f"Question: {state['question']}\n\nEvidence passages:\n\n"
                + (format_evidence(evidence) or "(no passage was found)"),
            },
        ]
        review = state.get("review")
        if review is not None and state.get("draft"):
            messages.append(
                {
                    "role": "user",
                    "content": prompts.WRITER_REVISION.format(
                        instructions=review.revision_instructions, draft=state["draft"]
                    ),
                }
            )
        message = self.llm.chat(messages)
        revisions = state.get("revisions", 0) + (1 if review is not None else 0)
        return {"draft": message.content or "", "revisions": revisions, "llm_calls": 1}

    def critique(self, state: AgentState) -> dict[str, Any]:
        draft = state["draft"]
        evidence = state.get("evidence", {})

        # 1. Contrôles déterministes : pas besoin du LLM pour voir une citation inventée.
        unknown = unknown_citations(draft, evidence)
        if unknown or (evidence and not cited_ids(draft)):
            problem = (
                f"These citations do not match any retrieved passage: {', '.join(unknown)}."
                if unknown
                else "The draft cites no passage. Cite the passage ids for every factual sentence."
            )
            review = Review(
                claims=[],
                verdict="revise",
                revision_instructions=problem,
                needs_more_evidence=False,
                clinical_flags=[],
            )
            return {"review": review, "retry_research": False}

        # 2. Vérification affirmation par affirmation par le LLM, sur les passages cités.
        cited = [evidence[chunk_id] for chunk_id in cited_ids(draft)]
        review = self.llm.structured(
            [
                {"role": "system", "content": prompts.CRITIC},
                {
                    "role": "user",
                    "content": f"Question: {state['question']}\n\nDraft:\n{draft}\n\n"
                    f"Cited passages:\n\n{format_evidence(cited) or '(none)'}",
                },
            ],
            Review,
        )
        update: dict[str, Any] = {"review": review, "llm_calls": 1, "retry_research": False}
        if review.verdict == "revise" and review.needs_more_evidence and self._can_retry_research(state):
            update["messages"] = [
                {
                    "role": "user",
                    "content": prompts.RESEARCH_FEEDBACK.format(instructions=review.revision_instructions),
                }
            ]
            update["research_retries"] = state.get("research_retries", 0) + 1
            update["retry_research"] = True
        return update

    def finalize(self, state: AgentState) -> dict[str, Any]:
        plan = state.get("plan")
        review = state["review"]
        answer, citations = renumber(state["draft"], state.get("evidence", {}))

        flags = list(review.clinical_flags)
        if review.verdict == "pass":
            verdict = "revised" if state.get("revisions", 0) else "pass"
        else:
            verdict = "flagged"
            unsupported = [c.claim for c in review.claims if c.verdict == "unsupported"]
            flags.insert(
                0,
                "The reviewer could not verify every statement against the sources"
                + (": " + "; ".join(unsupported) if unsupported else "."),
            )

        result = AgentResult(
            question=state["question"],
            answer=answer.strip(),
            citations=citations,
            disclaimer=disclaimer_for(plan.answer_language if plan else None),
            plan=plan,
            tool_calls=state.get("tool_calls", []),
            review=ReviewSummary(verdict=verdict, flags=flags, claims=review.claims),
            meta={
                "model": self.llm.model,
                "llm_calls": state.get("llm_calls", 0),
                "agent_steps": state.get("agent_steps", 0),
                "revisions": state.get("revisions", 0),
                "evidence_passages": len(state.get("evidence", {})),
            },
        )
        return {"result": result}

    # ------------------------------------------------------------------ #
    # Aiguillages (conditional edges)
    # ------------------------------------------------------------------ #
    def after_agent(self, state: AgentState) -> str:
        """Le LLM a demandé des outils -> on les exécute ; sinon on rédige."""
        return "tools" if state["messages"][-1].get("tool_calls") else "write"

    def after_tools(self, state: AgentState) -> str:
        """Retour au LLM tant que le budget d'étapes n'est pas épuisé."""
        return "agent" if state.get("agent_steps", 0) < self.max_steps else "write"

    def after_critique(self, state: AgentState) -> str:
        review = state["review"]
        if review.verdict == "pass" or state.get("revisions", 0) >= MAX_REVISIONS:
            return "finalize"
        return "agent" if state.get("retry_research") else "write"

    def _can_retry_research(self, state: AgentState) -> bool:
        return (
            state.get("research_retries", 0) < MAX_RESEARCH_RETRIES
            and state.get("revisions", 0) < MAX_REVISIONS
            and state.get("agent_steps", 0) < self.max_steps
        )