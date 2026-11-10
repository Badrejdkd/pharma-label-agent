"""Vérifie l'accès au proxy LiteLLM et les capacités du modèle de chat.

Usage : uv run python scripts/check_llm.py

Chaque test affiche [OK] ou [KO]. Le résultat guide les choix de l'agent :
function calling natif ou non, sortie JSON stricte ou mode JSON simple,
température réglable ou non.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable

from dotenv import load_dotenv
from openai import OpenAI

# Évite les erreurs d'encodage dans la console Windows.
sys.stdout.reconfigure(encoding="utf-8")


def env(name: str, *fallbacks: str, default: str | None = None) -> str | None:
    """Lit une variable d'environnement en acceptant des noms de secours
    et en retirant les guillemets que `docker run --env-file` laisse en place."""
    for key in (name, *fallbacks):
        value = os.getenv(key)
        if value:
            return value.strip().strip("'\"")
    return default


def check(label: str, fn: Callable[[], str]) -> bool:
    """Exécute un test, affiche [OK] ou [KO] avec sa durée."""
    start = time.perf_counter()
    try:
        detail = fn()
        print(f"[OK] {label} ({time.perf_counter() - start:.1f}s) {detail}")
        return True
    except Exception as exc:  # on veut voir toutes les erreurs dans ce script
        print(f"[KO] {label} -> {type(exc).__name__}: {str(exc)[:300]}")
        return False


def main() -> int:
    load_dotenv()
    api_key = env("OPENAI_API_KEY")
    base_url = env("OPENAI_BASE_URL", "OPENAI_BASE_UR")
    model = env("OPENAI_MODEL")
    emb_model = env("OPENAI_EMBEDDING_MODEL", default="text-embedding-3-small")

    print(f"base_url   = {base_url}")
    print(f"model      = {model}")
    print(f"embeddings = {emb_model}\n")
    if not (api_key and base_url and model):
        print("Il manque OPENAI_API_KEY, OPENAI_BASE_URL ou OPENAI_MODEL dans .env")
        return 1

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60)

    def list_models() -> str:
        ids = sorted(m.id for m in client.models.list().data)
        return "-> " + ", ".join(ids[:30])

    def simple_chat() -> str:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "In one sentence, what is a contraindication?"}],
        )
        return "-> " + (r.choices[0].message.content or "").strip()[:150]

    def temperature_zero() -> str:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say OK."}],
            temperature=0,
        )
        return "-> temperature=0 acceptée"

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_section",
                "description": "Return a full section of a drug label from the local corpus.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "drug": {"type": "string", "description": "Drug name, e.g. warfarin"},
                        "section": {
                            "type": "string",
                            "enum": ["contraindications", "interactions", "dosage"],
                        },
                    },
                    "required": ["drug", "section"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    def tool_calling() -> str:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Compare the contraindications of warfarin and apixaban. "
                    "Use the available tool.",
                }
            ],
            tools=tools,
            tool_choice="auto",
        )
        calls = r.choices[0].message.tool_calls or []
        if not calls:
            raise RuntimeError("le modèle n'a appelé aucun outil")
        return "-> " + " | ".join(f"{c.function.name}({c.function.arguments})" for c in calls)

    plan_schema = {
        "type": "object",
        "properties": {
            "question_type": {
                "type": "string",
                "enum": ["lookup", "comparison", "interaction", "threshold"],
            },
            "drugs": {"type": "array", "items": {"type": "string"}},
            "sub_questions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["question_type", "drugs", "sub_questions"],
        "additionalProperties": False,
    }

    def structured_output() -> str:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Decompose the user question into a research plan."},
                {"role": "user", "content": "Compare the contraindications of warfarin and apixaban."},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "plan", "strict": True, "schema": plan_schema},
            },
        )
        return f"-> {json.loads(r.choices[0].message.content or '{}')}"

    def json_mode() -> str:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": 'Return the JSON object {"ok": true} and nothing else.'}],
            response_format={"type": "json_object"},
        )
        return f"-> {json.loads(r.choices[0].message.content or '{}')}"

    def embeddings() -> str:
        r = client.embeddings.create(model=emb_model, input=["warfarin", "apixaban"])
        return f"-> {len(r.data)} vecteurs de dimension {len(r.data[0].embedding)}"

    results = {
        "models": check("Liste des modèles", list_models),
        "chat": check("Chat simple", simple_chat),
        "temperature": check("temperature=0", temperature_zero),
        "tools": check("Appel d'outils (function calling)", tool_calling),
        "json_schema": check("Sortie JSON stricte (json_schema)", structured_output),
        "json_mode": check("Mode JSON simple (json_object)", json_mode),
        "embeddings": check("Embeddings", embeddings),
    }

    print("\nRésumé :", {k: "OK" if v else "KO" for k, v in results.items()})
    essentials_ok = all(results[k] for k in ("chat", "tools", "embeddings"))
    return 0 if essentials_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())