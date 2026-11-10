"""Faux clients pour les tests : aucun appel réseau, résultats déterministes."""

from __future__ import annotations

import zlib

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