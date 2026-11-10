"""Essai de la recherche hybride avec les vrais embeddings, sur les 4 questions publiques.

Usage : uv run python -m scripts.try_search

Pour chaque question, affiche les 6 meilleurs passages avec leur rang BM25 (lex)
et leur rang par embeddings (sem). Les fichiers attendus par l'énoncé doivent
apparaître en tête.
"""

from __future__ import annotations

import logging
import sys
import time

from app.config import get_settings
from app.ingest import load_corpus
from app.llm import LLMClient
from app.retrieval import HybridIndex

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

QUESTIONS = {
    "What are the main contraindications for warfarin?": ["dailymed-warfarin.md"],
    "Compare the contraindications of warfarin and apixaban. "
    "What conditions are shared and which are unique to each drug?": [
        "dailymed-warfarin.md",
        "dailymed-apixaban.md",
    ],
    "A patient currently on warfarin is prescribed sertraline. "
    "What is the risk of bleeding and what monitoring is recommended?": [
        "dailymed-warfarin.md",
        "dailymed-sertraline.md",
    ],
    "A patient with an eGFR of 25 mL/min is being considered for metformin. Is metformin "
    "contraindicated at this level of renal function, and what is the threshold?": [
        "dailymed-metformin.md",
    ],
}


def main() -> None:
    settings = get_settings()
    start = time.perf_counter()
    index = HybridIndex.build(load_corpus(settings.corpus_dir), LLMClient(settings), settings.cache_dir)
    print(f"Index prêt en {time.perf_counter() - start:.1f}s, sémantique active : {index.semantic_enabled}")

    for question, expected in QUESTIONS.items():
        hits = index.search(question, k=6)
        found = {hit.chunk.file for hit in hits}
        status = "OK" if set(expected) <= found else "MANQUE " + ", ".join(set(expected) - found)
        print(f"\n[{status}] {question}")
        for hit in hits:
            print(
                f"  {hit.chunk.chunk_id:<26} lex={str(hit.lexical_rank):<4} sem={str(hit.semantic_rank):<4}"
                f" {hit.chunk.section[:60]}"
            )


if __name__ == "__main__":
    main()