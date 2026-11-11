"""Évaluation de l'agent : questions publiques de l'énoncé + jeu maison.

Usage :
    uv run python -m eval.run_eval                               # agent en mémoire (même code que l'API)
    uv run python -m eval.run_eval --url http://localhost:8000   # contre l'API, ex. le conteneur Docker
    uv run python -m eval.run_eval --only public                 # seulement les 4 questions de l'énoncé

Mesures, toutes calculées par du code (pas de LLM juge) :
- sources : part des fichiers attendus qui sont cités dans la réponse ;
- faits clés : part des expressions attendues (regex) présentes dans la réponse ;
- verdict du critique, présence de l'avertissement, outils choisis, latence.

Écrit eval/results/latest.md et eval/results/latest.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import sys
import time
import urllib.request
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

EVAL_DIR = Path(__file__).parent
DISCLAIMER_RE = re.compile(r"not medical advice|pas un avis médical", re.IGNORECASE)

Answerer = Callable[[str], dict[str, Any]]


def local_answerer() -> Answerer:
    """Construit l'agent en mémoire, exactement comme le fait l'API."""
    from app.bootstrap import build_agent

    agent = build_agent()
    return lambda question: agent.run(question).model_dump(mode="json")


def api_answerer(base_url: str) -> Answerer:
    """Interroge une API déjà lancée (POST /ask)."""

    def ask(question: str) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/ask",
            data=json.dumps({"question": question}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))

    return ask


def score(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Note une réponse par rapport aux attentes du cas de test."""
    cited_files = {citation["file"] for citation in result["citations"]}
    expected = set(case["expected_files"])
    missing_files = sorted(expected - cited_files)
    facts = {fact: bool(re.search(fact, result["answer"], re.IGNORECASE)) for fact in case["facts"]}
    return {
        "source_recall": 1.0 if not expected else (len(expected) - len(missing_files)) / len(expected),
        "missing_files": missing_files,
        "fact_coverage": sum(facts.values()) / len(facts) if facts else 1.0,
        "missing_facts": [fact for fact, found in facts.items() if not found],
        "verdict": result["review"]["verdict"],
        "disclaimer": bool(DISCLAIMER_RE.search(result["disclaimer"])),
        "tools": [call["tool"] for call in result["tool_calls"]],
        "latency_s": round(result["meta"].get("latency_ms", 0) / 1000, 1),
        "llm_calls": result["meta"].get("llm_calls"),
    }


def run(cases: list[dict[str, Any]], answer: Answerer) -> list[dict[str, Any]]:
    """Pose chaque question ; une erreur sur une question n'arrête pas l'évaluation."""
    rows = []
    for i, case in enumerate(cases, start=1):
        print(f"[{i}/{len(cases)}] {case['question']}", flush=True)
        start = time.perf_counter()
        try:
            result = answer(case["question"])
            row = {"id": case["id"], "set": case["set"], "question": case["question"], **score(case, result)}
            row["answer"] = result["answer"]
        except Exception as exc:  # noqa: BLE001 - un échec ne doit pas arrêter toute l'évaluation
            row = {"id": case["id"], "set": case["set"], "question": case["question"], "error": str(exc)}
            row["latency_s"] = round(time.perf_counter() - start, 1)
        rows.append(row)
        status = row.get("error") or (
            f"sources {row['source_recall']:.0%}, faits {row['fact_coverage']:.0%}, "
            f"{row['verdict']}, {row['latency_s']} s"
        )
        print(f"    -> {status}", flush=True)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if "error" not in row]
    latencies = sorted(row["latency_s"] for row in ok) or [0.0]
    return {
        "questions": len(rows),
        "errors": len(rows) - len(ok),
        "source_recall": statistics.mean(row["source_recall"] for row in ok) if ok else 0.0,
        "fact_coverage": statistics.mean(row["fact_coverage"] for row in ok) if ok else 0.0,
        "disclaimer_rate": statistics.mean(row["disclaimer"] for row in ok) if ok else 0.0,
        "verdicts": dict(Counter(row["verdict"] for row in ok)),
        "latency_median_s": statistics.median(latencies),
        "latency_max_s": latencies[-1],
    }


def to_markdown(rows: list[dict[str, Any]], summary: dict[str, Any], target: str) -> str:
    lines = [
        "# Evaluation results",
        "",
        f"Target: {target}. Date: {time.strftime('%Y-%m-%d %H:%M')}.",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Questions | {summary['questions']} ({summary['errors']} errors) |",
        f"| Expected sources cited | {summary['source_recall']:.0%} |",
        f"| Key facts present | {summary['fact_coverage']:.0%} |",
        f"| Disclaimer present | {summary['disclaimer_rate']:.0%} |",
        f"| Critic verdicts | {', '.join(f'{k}: {v}' for k, v in summary['verdicts'].items())} |",
        f"| Latency (median / max) | {summary['latency_median_s']} s / {summary['latency_max_s']} s |",
        "",
        "| Id | Sources | Facts | Verdict | Tools chosen by the agent | Latency |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if "error" in row:
            lines.append(f"| {row['id']} | error: {row['error'][:80]} | | | | {row['latency_s']} s |")
            continue
        missing_facts = ", ".join(f"`{fact}`" for fact in row["missing_facts"]).replace("|", "\\|")
        missing = f" (missing: {missing_facts})" if missing_facts else ""
        lines.append(
            f"| {row['id']} | {row['source_recall']:.0%} | {row['fact_coverage']:.0%}{missing} | "
            f"{row['verdict']} | {', '.join(row['tools'])} | {row['latency_s']} s |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Évalue l'agent sur un jeu de questions.")
    parser.add_argument("--url", help="URL d'une API lancée (sinon, agent en mémoire)")
    parser.add_argument("--only", choices=["public", "extra"], help="limiter à un sous-ensemble")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    cases = json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))
    if args.only:
        cases = [case for case in cases if case["set"] == args.only]

    answer = api_answerer(args.url) if args.url else local_answerer()
    rows = run(cases, answer)
    summary = summarize(rows)

    results_dir = EVAL_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    report = to_markdown(rows, summary, args.url or "in-process agent")
    (results_dir / "latest.md").write_text(report, encoding="utf-8")
    (results_dir / "latest.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n" + report)


if __name__ == "__main__":
    main()
