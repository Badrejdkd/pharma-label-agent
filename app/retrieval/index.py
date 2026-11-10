"""Index hybride en mémoire : BM25 (lexical) + embeddings (sémantique), fusion RRF.

Pourquoi deux recherches ?
- les embeddings comprennent le sens (« saignement » ~ « hemorrhage ») ;
- BM25 retrouve les termes exacts (eGFR, INR, noms de médicaments).

Pourquoi la fusion par rang (Reciprocal Rank Fusion) ?
Les scores BM25 et cosinus ne sont pas sur la même échelle. RRF ignore les
scores et ne garde que les rangs : un passage bien classé par les deux
recherches remonte en tête, sans réglage de poids.

    score_rrf(d) = somme sur les classements de 1 / (60 + rang(d))
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.ingest import Chunk
from app.llm import LLMError
from app.retrieval.bm25 import BM25
from app.retrieval.embeddings import Embedder, load_or_compute, normalize

logger = logging.getLogger(__name__)

RRF_K = 60  # constante standard de la littérature RRF


@dataclass(frozen=True)
class SearchHit:
    """Un résultat de recherche et la façon dont il a été trouvé."""

    chunk: Chunk
    score: float  # score RRF : plus il est haut, plus le passage est pertinent
    lexical_rank: int | None  # rang BM25 (1 = meilleur), None si aucun terme commun
    semantic_rank: int | None  # rang par embeddings, None si recherche sémantique inactive


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Fusionne plusieurs classements (listes d'indices, du meilleur au moins bon)."""
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] += 1.0 / (k + rank)
    return dict(fused)


class HybridIndex:
    """Recherche hybride filtrable par médicament, catégorie et source."""

    def __init__(
        self,
        chunks: list[Chunk],
        bm25: BM25,
        embeddings: np.ndarray | None,
        embedder: Embedder | None,
    ) -> None:
        self.chunks = chunks
        self._bm25 = bm25
        self._embeddings = embeddings
        self._embedder = embedder
        self._query_cache: dict[str, np.ndarray] = {}

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder | None, cache_dir: Path) -> HybridIndex:
        """Construit l'index ; sans embeddings disponibles, la recherche reste lexicale."""
        texts = [chunk.indexed_text for chunk in chunks]
        bm25 = BM25(texts)
        embeddings = None
        if embedder is not None:
            try:
                embeddings = load_or_compute(texts, embedder, cache_dir)
            except LLMError as exc:
                logger.warning("Embeddings indisponibles, recherche lexicale seule : %s", exc)
        return cls(chunks, bm25, embeddings, embedder)

    @property
    def semantic_enabled(self) -> bool:
        """Vrai si la recherche par embeddings est active."""
        return self._embeddings is not None

    @property
    def drugs(self) -> list[str]:
        """Médicaments présents dans le corpus."""
        return sorted({chunk.drug for chunk in self.chunks if chunk.drug})

    def _candidates(
        self,
        drugs: list[str] | None,
        categories: list[str] | None,
        sources: list[str] | None,
    ) -> list[int]:
        """Indices des passages qui respectent les filtres (None = pas de filtre)."""
        drug_set = {d.lower() for d in drugs} if drugs else None
        category_set = set(categories) if categories else None
        source_set = set(sources) if sources else None
        return [
            i
            for i, chunk in enumerate(self.chunks)
            if (drug_set is None or chunk.drug in drug_set)
            and (category_set is None or chunk.category in category_set)
            and (source_set is None or chunk.source in source_set)
        ]

    def _embed_query(self, query: str) -> np.ndarray | None:
        """Embedding normalisé de la requête, gardé en cache ; None en cas d'échec."""
        if self._embedder is None:
            return None
        if query not in self._query_cache:
            try:
                vector = np.asarray(self._embedder.embed([query])[0], dtype=np.float32)
            except LLMError as exc:
                logger.warning("Embedding de requête impossible, recherche lexicale seule : %s", exc)
                return None
            self._query_cache[query] = normalize(vector)
        return self._query_cache[query]

    def search(
        self,
        query: str,
        k: int = 6,
        drugs: list[str] | None = None,
        categories: list[str] | None = None,
        sources: list[str] | None = None,
    ) -> list[SearchHit]:
        """Les k passages les plus pertinents parmi ceux qui respectent les filtres."""
        candidates = self._candidates(drugs, categories, sources)
        if not candidates:
            return []

        lexical_scores = self._bm25.scores(query)
        lexical_ranking = sorted(
            (i for i in candidates if lexical_scores[i] > 0),
            key=lambda i: (-lexical_scores[i], i),
        )
        rankings = [lexical_ranking]

        semantic_ranking: list[int] = []
        query_vector = self._embed_query(query) if self.semantic_enabled else None
        if query_vector is not None and self._embeddings is not None:
            similarities = self._embeddings[candidates] @ query_vector
            semantic_ranking = [candidates[j] for j in np.argsort(-similarities, kind="stable")]
            rankings.append(semantic_ranking)

        fused = reciprocal_rank_fusion(rankings)
        lexical_pos = {doc: rank for rank, doc in enumerate(lexical_ranking, start=1)}
        semantic_pos = {doc: rank for rank, doc in enumerate(semantic_ranking, start=1)}
        best = sorted(fused, key=lambda i: (-fused[i], i))[:k]
        return [
            SearchHit(
                chunk=self.chunks[i],
                score=fused[i],
                lexical_rank=lexical_pos.get(i),
                semantic_rank=semantic_pos.get(i),
            )
            for i in best
        ]

    def section_chunks(self, drug: str, category: str, source: str = "dailymed") -> list[Chunk]:
        """Tous les passages d'une catégorie pour un médicament, dans l'ordre du fichier."""
        return sorted(
            (
                chunk
                for chunk in self.chunks
                if chunk.drug == drug.lower() and chunk.category == category and chunk.source == source
            ),
            key=lambda chunk: chunk.order,
        )