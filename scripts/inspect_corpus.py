"""Affiche un résumé du découpage du corpus, pour vérifier l'ingestion à l'œil.

Usage : uv run python -m scripts.inspect_corpus
"""

from __future__ import annotations

import sys
from collections import Counter

from app.config import get_settings
from app.ingest import load_corpus

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    chunks = load_corpus(get_settings().corpus_dir)

    print(f"{'fichier':<30} {'passages':>8} {'max car.':>9}")
    for file in sorted({c.file for c in chunks}):
        file_chunks = [c for c in chunks if c.file == file]
        longest = max(len(c.text) for c in file_chunks)
        print(f"{file:<30} {len(file_chunks):>8} {longest:>9}")

    print(f"\nTotal : {len(chunks)} passages")
    print("\nPassages par catégorie :")
    for category, count in Counter(c.category for c in chunks).most_common():
        print(f"  {category:<22} {count}")

    example = next(c for c in chunks if c.file == "dailymed-warfarin.md" and c.category == "contraindications")
    print(f"\nExemple ({example.chunk_id}) :\n{example.indexed_text[:500]}")


if __name__ == "__main__":
    main()