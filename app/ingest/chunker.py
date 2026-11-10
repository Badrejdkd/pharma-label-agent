"""Découpage des sections en passages (chunks) prêts à indexer.

Règles :
- une section devient un passage si elle tient dans `MAX_CHARS` caractères ;
- sinon elle est coupée sur les paragraphes, puis sur les phrases pour un
  paragraphe trop long ; aucun texte n'est perdu ni dupliqué ;
- chaque passage garde ses métadonnées (fichier, médicament, source,
  catégorie, chemin de section) pour les filtres et les citations.

Pas de chevauchement entre morceaux : chaque morceau répète déjà en en-tête
le médicament et le titre de sa section (voir `Chunk.indexed_text`), ce qui
suffit à le resituer.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

from app.ingest.parser import Document

MAX_CHARS = 2400  # environ 600 tokens

SOURCE_LABELS = {
    "dailymed": "DailyMed FDA label",
    "medlineplus": "MedlinePlus patient information",
    "glossary": "Glossary",
}


@dataclass(frozen=True)
class Chunk:
    """Un passage du corpus, unité de recherche et de citation."""

    chunk_id: str  # ex. "dailymed-warfarin:12", unique dans le corpus
    file: str  # ex. "dailymed-warfarin.md"
    source: str  # "dailymed", "medlineplus" ou "glossary"
    drug: str | None  # ex. "warfarin"
    category: str  # ex. "contraindications"
    section: str  # ex. "7 DRUG INTERACTIONS > 7.3 Drugs that Increase Bleeding Risk"
    text: str  # texte du passage
    order: int  # position dans le fichier, pour reconstituer une section entière

    @property
    def header(self) -> str:
        """Contexte lisible : médicament, source et section."""
        subject = self.drug or "glossary"
        return f"{subject} | {SOURCE_LABELS.get(self.source, self.source)} | {self.section}"

    @property
    def indexed_text(self) -> str:
        """Texte réellement indexé : l'en-tête puis le passage.

        L'en-tête place le nom du médicament dans chaque passage, même quand
        le texte ne le cite pas (la notice de l'apixaban dit « ELIQUIS »).
        """
        return f"{self.header}\n{self.text}"


def _pack(pieces: list[str], max_chars: int, sep: str) -> list[str]:
    """Regroupe des morceaux consécutifs tant que le total tient dans max_chars."""
    parts: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}{sep}{piece}" if current else piece
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                parts.append(current)
            current = piece
    if current:
        parts.append(current)
    return parts


def _split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Coupe un paragraphe trop long sur les fins de phrases."""
    pieces: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        if len(sentence) <= max_chars:
            pieces.append(sentence)
        else:  # phrase géante sans ponctuation : coupe sur les espaces
            pieces.extend(textwrap.wrap(sentence, max_chars))
    return _pack(pieces, max_chars, sep=" ")


def split_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Coupe un texte en morceaux de max_chars caractères au plus.

    Un texte court est renvoyé tel quel ; un texte long est coupé sur les
    paragraphes, puis sur les phrases pour les paragraphes trop longs.
    """
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            pieces.append(paragraph)
        else:
            pieces.extend(_split_long_paragraph(paragraph, max_chars))
    return _pack(pieces, max_chars, sep="\n\n")


def chunk_document(document: Document, max_chars: int = MAX_CHARS) -> list[Chunk]:
    """Transforme les sections d'un document en passages numérotés."""
    stem = Path(document.file).stem
    chunks: list[Chunk] = []
    for section in document.sections:
        for part in split_text(section.text, max_chars):
            chunks.append(
                Chunk(
                    chunk_id=f"{stem}:{len(chunks)}",
                    file=document.file,
                    source=document.source,
                    drug=document.drug,
                    category=section.category,
                    section=" > ".join(section.path),
                    text=part,
                    order=len(chunks),
                )
            )
    return chunks