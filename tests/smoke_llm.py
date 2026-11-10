"""Essai rapide du client LLM de l'application contre le vrai proxy.

Usage : uv run python -m scripts.smoke_llm
"""

from __future__ import annotations

import logging
import sys

from pydantic import BaseModel

from app.config import get_settings
from app.llm import LLMClient

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


class DemoPlan(BaseModel):
    question_type: str
    drugs: list[str]
    sub_questions: list[str]


def main() -> None:
    settings = get_settings()
    llm = LLMClient(settings)
    print(f"Modèle : {settings.openai_model} via {settings.openai_base_url}\n")

    answer = llm.chat([{"role": "user", "content": "In one sentence, what is warfarin?"}])
    print("chat       ->", answer.content)

    plan = llm.structured(
        [
            {"role": "system", "content": "Decompose the user question into a research plan."},
            {
                "role": "user",
                "content": "A patient on warfarin is prescribed sertraline. What is the bleeding risk?",
            },
        ],
        DemoPlan,
    )
    print("structured ->", plan.model_dump())
    print("JSON strict accepté :", llm.strict_json_supported)

    vectors = llm.embed(["warfarin", "apixaban", "metformin"])
    print(f"embed      -> {len(vectors)} vecteurs de dimension {len(vectors[0])}")


if __name__ == "__main__":
    main()