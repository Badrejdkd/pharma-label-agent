"""Ingestion du corpus : fichiers markdown -> sections -> passages indexables."""

from __future__ import annotations

from pathlib import Path

from app.ingest.chunker import MAX_CHARS, Chunk, chunk_document, split_text
from app.ingest.parser import Document, Section, load_documents, parse_file

__all__ = [
    "MAX_CHARS",
    "Chunk",
    "Document",
    "Section",
    "chunk_document",
    "load_corpus",
    "load_documents",
    "parse_file",
    "split_text",
]


def load_corpus(corpus_dir: Path, max_chars: int = MAX_CHARS) -> list[Chunk]:
    """Lit tout le corpus et renvoie la liste de ses passages, dans l'ordre des fichiers."""
    chunks: list[Chunk] = []
    for document in load_documents(corpus_dir):
        chunks.extend(chunk_document(document, max_chars))
    return chunks