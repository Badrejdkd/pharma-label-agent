"""Tests de la recherche : BM25, fusion RRF, filtres, cache et repli (aucun réseau)."""

from __future__ import annotations

from pathlib import Path

from app.ingest import Chunk
from app.llm import LLMError
from app.retrieval import BM25, HybridIndex, reciprocal_rank_fusion, tokenize
from tests.fakes import FakeEmbedder


def test_tokenize_stems_and_drops_stopwords() -> None:
    assert tokenize("The contraindications of Warfarin are contraindicated") == [
        "contraindic",
        "warfarin",
        "contraindic",
    ]
    assert tokenize("Bleeding was monitored") == ["bleed", "monitor"]


def test_bm25_ranks_the_matching_document_first() -> None:
    bm25 = BM25(["warfarin bleeding risk", "metformin lactic acidosis", "aspirin and bleeding"])
    scores = bm25.scores("warfarin bleeding")
    assert scores.index(max(scores)) == 0
    assert scores[1] == 0.0  # aucun terme commun


def test_rrf_rewards_documents_ranked_well_by_both_lists() -> None:
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 2]])
    assert sorted(fused, key=fused.get, reverse=True) == [1, 3, 2]


def test_warfarin_contraindications_come_first(index: HybridIndex) -> None:
    top = index.search("What are the main contraindications for warfarin?", k=3)
    assert top[0].chunk.file == "dailymed-warfarin.md"
    assert top[0].chunk.category == "contraindications"


def test_metformin_egfr_threshold_is_found(index: HybridIndex) -> None:
    hits = index.search("metformin eGFR 25 mL/min contraindicated renal threshold", k=5)
    assert any(
        hit.chunk.file == "dailymed-metformin.md" and "below 30" in hit.chunk.text for hit in hits
    )


def test_filters_restrict_results(index: HybridIndex) -> None:
    hits = index.search("bleeding", drugs=["Apixaban"], categories=["contraindications"])
    assert hits
    assert all(h.chunk.drug == "apixaban" and h.chunk.category == "contraindications" for h in hits)
    assert index.search("bleeding", drugs=["unknown-drug"]) == []


def test_hits_report_both_ranks(index: HybridIndex) -> None:
    assert index.semantic_enabled
    hit = index.search("warfarin contraindications", k=1)[0]
    assert hit.lexical_rank is not None and hit.semantic_rank is not None


def test_section_chunks_are_in_file_order(index: HybridIndex) -> None:
    chunks = index.section_chunks("Warfarin", "contraindications")
    assert chunks and all(c.file == "dailymed-warfarin.md" for c in chunks)
    assert [c.order for c in chunks] == sorted(c.order for c in chunks)


def test_embeddings_are_cached_on_disk(corpus_chunks: list[Chunk], tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    HybridIndex.build(corpus_chunks, embedder, tmp_path)
    HybridIndex.build(corpus_chunks, embedder, tmp_path)
    assert embedder.calls == 1  # le second démarrage lit le cache

    embedder.embedding_model = "another-model"
    HybridIndex.build(corpus_chunks, embedder, tmp_path)
    assert embedder.calls == 2  # autre modèle : cache ignoré


class FailingEmbedder:
    embedding_model = "broken"

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise LLMError("proxy injoignable")


def test_falls_back_to_lexical_search(corpus_chunks: list[Chunk], tmp_path: Path) -> None:
    index = HybridIndex.build(corpus_chunks, FailingEmbedder(), tmp_path)
    assert not index.semantic_enabled
    hits = index.search("warfarin contraindications", k=3)
    assert hits and hits[0].semantic_rank is None