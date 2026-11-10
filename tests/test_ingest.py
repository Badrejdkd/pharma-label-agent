"""Tests de l'ingestion, sur le vrai corpus du dépôt (aucun appel réseau)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingest import MAX_CHARS, Chunk, Document, load_corpus, load_documents, split_text

CORPUS = Path(__file__).resolve().parents[1] / "corpus"


@pytest.fixture(scope="module")
def documents() -> dict[str, Document]:
    return {document.file: document for document in load_documents(CORPUS)}


@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    return load_corpus(CORPUS)


def categories_by_heading(document: Document) -> dict[str, str]:
    """Catégorie de chaque titre de niveau 2 d'un document."""
    return {section.path[0]: section.category for section in document.sections}


def test_all_files_are_parsed_with_metadata(documents: dict[str, Document]) -> None:
    assert len(documents) == 21
    assert documents["GLOSSARY.md"].source == "glossary"
    assert documents["GLOSSARY.md"].drug is None
    assert documents["medlineplus-warfarin.md"].source == "medlineplus"
    # Le médicament vient du nom de fichier, même si la notice dit surtout « ELIQUIS ».
    assert documents["dailymed-apixaban.md"].drug == "apixaban"
    assert documents["dailymed-warfarin.md"].url.startswith("https://dailymed.nlm.nih.gov/")


def test_numbered_label_sections(documents: dict[str, Document]) -> None:
    warfarin = documents["dailymed-warfarin.md"]
    contraindications = [s for s in warfarin.sections if s.category == "contraindications"]
    assert contraindications and "Pregnancy" in contraindications[0].text

    bleeding = next(s for s in warfarin.sections if s.number == "7.3")
    assert bleeding.category == "interactions"
    assert bleeding.path == ("7 DRUG INTERACTIONS", "7.3 Drugs that Increase Bleeding Risk")
    assert "sertraline" in bleeding.text


def test_boxed_warnings_are_recognised(documents: dict[str, Document]) -> None:
    for file in ("dailymed-warfarin.md", "dailymed-apixaban.md", "medlineplus-warfarin.md"):
        assert any(s.category == "boxed_warning" for s in documents[file].sections), file


def test_old_format_label(documents: dict[str, Document]) -> None:
    ibuprofen = documents["dailymed-ibuprofen.md"]
    categories = categories_by_heading(ibuprofen)
    assert categories["CONTRAINDICATIONS"] == "contraindications"
    assert categories["OVERDOSAGE"] == "overdosage"  # et non "dosage"
    assert categories["Drug Interactions"] == "interactions"
    # Une sous-section hérite de la catégorie de son titre de niveau 2.
    gi = next(s for s in ibuprofen.sections if s.heading.startswith("Gastrointestinal Effects"))
    assert gi.category == "warnings"


def test_otc_label(documents: dict[str, Document]) -> None:
    categories = categories_by_heading(documents["dailymed-aspirin.md"])
    assert categories == {
        "Uses": "indications",
        "Directions": "dosage",
        "Warnings": "warnings",
        "In case of overdose": "overdosage",
    }


def test_medlineplus_sections(documents: dict[str, Document]) -> None:
    categories = categories_by_heading(documents["medlineplus-metformin.md"])
    assert categories["Why is this medication prescribed?"] == "indications"
    assert categories["How should this medicine be used?"] == "dosage"
    assert categories["What side effects can this medication cause?"] == "adverse_reactions"
    assert categories["What special precautions should I follow?"] == "warnings"
    assert categories["In case of emergency/overdose"] == "overdosage"


def test_metformin_renal_threshold_is_in_contraindications(chunks: list[Chunk]) -> None:
    texts = [c.text for c in chunks if c.file == "dailymed-metformin.md" and c.category == "contraindications"]
    assert any("eGFR" in text and "below 30" in text for text in texts)


def test_chunks_are_bounded_unique_and_non_empty(chunks: list[Chunk]) -> None:
    assert all(0 < len(c.text) <= MAX_CHARS for c in chunks)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_indexed_text_carries_drug_and_section(chunks: list[Chunk]) -> None:
    chunk = next(c for c in chunks if c.file == "dailymed-apixaban.md" and c.category == "contraindications")
    assert chunk.indexed_text.startswith("apixaban | DailyMed FDA label | 4 CONTRAINDICATIONS")


def test_split_text_keeps_short_text_unchanged() -> None:
    assert split_text("Short text.", max_chars=100) == ["Short text."]


def test_split_text_cuts_on_sentences_without_losing_text() -> None:
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    parts = split_text(text, max_chars=300)
    assert len(parts) > 1
    assert all(len(part) <= 300 for part in parts)
    assert " ".join(parts) == text