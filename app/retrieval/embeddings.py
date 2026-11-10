"""Embeddings du corpus : calcul par l'API, normalisation, cache sur disque.

Le cache évite de recalculer les ~500 embeddings à chaque démarrage. Sa clé
dépend du modèle d'embeddings ET du texte exact des passages : si l'un des
deux change, le cache est ignoré et les embeddings sont recalculés.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Ce dont l'index a besoin : `LLMClient` le respecte, un faux client de test aussi."""

    embedding_model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def normalize(matrix: np.ndarray) -> np.ndarray:
    """Ramène chaque ligne à une norme de 1 : le produit scalaire devient le cosinus."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def cache_key(model: str, texts: list[str]) -> str:
    """Empreinte courte du modèle et des textes."""
    digest = hashlib.sha256(model.encode("utf-8"))
    for text in texts:
        digest.update(b"\x00")
        digest.update(text.encode("utf-8"))
    return digest.hexdigest()[:16]


def load_or_compute(texts: list[str], embedder: Embedder, cache_dir: Path) -> np.ndarray:
    """Renvoie la matrice normalisée des embeddings (une ligne par texte)."""
    path = cache_dir / f"embeddings-{cache_key(embedder.embedding_model, texts)}.npy"
    if path.exists():
        logger.info("Embeddings lus depuis le cache %s", path)
        return np.load(path)

    logger.info("Calcul de %d embeddings avec %s", len(texts), embedder.embedding_model)
    matrix = normalize(np.asarray(embedder.embed(texts), dtype=np.float32))
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, matrix)
    except OSError as exc:  # disque en lecture seule : on continue sans cache
        logger.warning("Impossible d'écrire le cache d'embeddings : %s", exc)
    return matrix