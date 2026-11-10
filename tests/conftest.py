"""Fixtures partagées : corpus réel et index construit avec un faux client d'embeddings."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingest import Chunk, load_corpus
from app.retrieval import HybridIndex
from tests.fakes import FakeEmbedder

CORPUS_DIR = Path(__file__).resolve().parents[1] / "corpus"


@pytest.fixture(scope="session")
def corpus_chunks() -> list[Chunk]:
    return load_corpus(CORPUS_DIR)


@pytest.fixture(scope="session")
def index(corpus_chunks: list[Chunk], tmp_path_factory: pytest.TempPathFactory) -> HybridIndex:
    """Index hybride complet, construit une seule fois pour toute la session de tests."""
    cache_dir = tmp_path_factory.mktemp("embeddings-cache")
    return HybridIndex.build(corpus_chunks, FakeEmbedder(), cache_dir)