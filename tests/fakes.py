"""Faux clients pour les tests : aucun appel réseau, résultats déterministes."""

from __future__ import annotations

import json
import zlib
from types import SimpleNamespace
from typing import Any

from app.retrieval import tokenize


class FakeEmbedder:
    """Embeddings factices mais déterministes : sac de mots haché sur 64 dimensions.

    Deux textes qui partagent des mots ont des vecteurs proches, ce qui suffit
    à tester la mécanique de l'index sans appeler l'API.
    """

    embedding_model = "fake-embedding"
    dimensions = 64

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in tokenize(text):
                vector[zlib.crc32(token.encode()) % self.dimensions] += 1.0
            vectors.append(vector)
        return vectors


def fake_message(content: str = "", tool_calls: list[tuple[str, dict]] | None = None) -> SimpleNamespace:
    """Imite un message de l'API OpenAI (texte et/ou appels d'outils)."""
    calls = [
        SimpleNamespace(
            id=f"call_{i}",
            type="function",
            function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
        )
        for i, (name, arguments) in enumerate(tool_calls or [])
    ]
    return SimpleNamespace(content=content, tool_calls=calls or None)


class FakeLLM:
    """LLM scripté : chaque méthode renvoie, dans l'ordre, les réponses préparées par le test.

    - `agent_turns` : tours du nœud agent (appel avec outils) ; une fois la liste
      vide, le faux LLM répond « Done. » sans outil, ce qui termine la recherche ;
    - `drafts` : brouillons du rédacteur (appel sans outils) ;
    - `plans` / `reviews` : sorties structurées du planificateur et du critique.
    """

    model = "fake-llm"

    def __init__(
        self,
        plans: list[Any],
        drafts: list[str],
        reviews: list[Any],
        agent_turns: list[dict[str, Any]] | None = None,
    ) -> None:
        self.plans = list(plans)
        self.drafts = list(drafts)
        self.reviews = list(reviews)
        self.agent_turns = list(agent_turns or [])
        self.calls: list[str] = []
        self.tool_choices: list[str] = []

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, tool_choice: str = "auto"
    ) -> SimpleNamespace:
        if tools:
            self.calls.append("agent")
            self.tool_choices.append(tool_choice)
            turn = self.agent_turns.pop(0) if self.agent_turns else {"content": "Done."}
            return fake_message(**turn)
        self.calls.append("write")
        return fake_message(content=self.drafts.pop(0))

    def structured(self, messages: list[dict[str, Any]], output_type: type) -> Any:
        self.calls.append(output_type.__name__)
        queue = self.plans if output_type.__name__ == "Plan" else self.reviews
        return queue.pop(0)