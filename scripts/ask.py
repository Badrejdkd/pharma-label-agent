"""Pose des questions à l'agent depuis le terminal, avec le vrai LLM.

Usage :
    uv run python -m scripts.ask                       # les 4 questions publiques
    uv run python -m scripts.ask "Ta question ici"     # une question libre
    uv run python -m scripts.ask --graph               # affiche le graphe (Mermaid)
"""

from __future__ import annotations

import logging
import sys

from app.agent.schemas import AgentResult
from app.bootstrap import build_agent

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

PUBLIC_QUESTIONS = [
    "What are the main contraindications for warfarin?",
    "Compare the contraindications of warfarin and apixaban. "
    "What conditions are shared and which are unique to each drug?",
    "A patient currently on warfarin is prescribed sertraline. "
    "What is the risk of bleeding and what monitoring is recommended?",
    "A patient with an eGFR of 25 mL/min is being considered for metformin. Is metformin "
    "contraindicated at this level of renal function, and what is the threshold?",
]


def show(result: AgentResult) -> None:
    line = "=" * 100
    print(f"\n{line}\nQUESTION : {result.question}\n{line}")
    if result.plan:
        print(f"Plan ({result.plan.question_type}) : " + " | ".join(result.plan.sub_questions))
    for trace in result.tool_calls:
        status = f"ERREUR {trace.error}" if trace.error else f"{len(trace.passages)} passages"
        print(f"  outil {trace.tool}({trace.arguments}) -> {status} ({trace.duration_ms} ms)")
    print(f"\n{result.answer}\n")
    for citation in result.citations:
        print(f"  [{citation.id}] {citation.file} | {citation.section}")
    print(f"\nCritique : {result.review.verdict}")
    for flag in result.review.flags:
        print(f"  ! {flag}")
    print(f"\n{result.disclaimer}")
    print(f"\nMeta : {result.meta}")


def main() -> None:
    agent = build_agent()
    if "--graph" in sys.argv:
        print(agent.graph.get_graph().draw_mermaid())
        return
    questions = [arg for arg in sys.argv[1:] if not arg.startswith("--")] or PUBLIC_QUESTIONS
    for question in questions:
        show(agent.run(question))


if __name__ == "__main__":
    main()