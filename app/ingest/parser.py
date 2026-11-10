"""Lecture des fichiers markdown du corpus en sections typées.

Un fichier du corpus ressemble à ceci :

    # Warfarin (DailyMed label)
    Source: https://dailymed.nlm.nih.gov/...
    ## 4 CONTRAINDICATIONS
    texte...
    ## 5 WARNINGS AND PRECAUTIONS
    ### 5.3 Increased Risk of Bleeding
    texte...

Le parseur produit un `Document` (métadonnées du fichier) contenant une
`Section` par titre : chemin complet des titres parents, numéro de section
éventuel et catégorie normalisée, commune aux trois formats du corpus
(notices FDA numérotées, anciennes notices FDA, fiches MedlinePlus).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
SECTION_NUMBER_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+")
SOURCE_URL_RE = re.compile(r"^Source:\s*(\S+)", re.MULTILINE)

# Notices DailyMed au format actuel : la catégorie dépend du numéro de section.
NUMBERED_CATEGORIES: dict[str, str] = {
    "1": "indications",
    "2": "dosage",
    "3": "description",
    "4": "contraindications",
    "5": "warnings",
    "6": "adverse_reactions",
    "7": "interactions",
    "8": "specific_populations",
    "10": "overdosage",
    "11": "description",
    "12": "pharmacology",
    "16": "description",
    "17": "patient_info",
}

# Titres sans numéro (anciennes notices, notices sans ordonnance, MedlinePlus).
# Le premier motif contenu dans le titre (en minuscules) l'emporte : l'ordre compte.
# Ex. "contraindication" doit passer avant "indications", "overdos" avant "dosage".
TITLE_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("warning:", "boxed_warning"),  # "WARNING: BLEEDING RISK", "IMPORTANT WARNING:"
    ("contraindication", "contraindications"),
    ("indications", "indications"),
    ("uses", "indications"),  # "Uses", "Other uses for this medicine"
    ("why is this medication prescribed", "indications"),
    ("overdos", "overdosage"),  # "OVERDOSAGE", "In case of overdose"
    ("dosage", "dosage"),
    ("directions", "dosage"),
    ("how should this medicine be used", "dosage"),
    ("forget a dose", "dosage"),
    ("drug interactions", "interactions"),
    ("adverse reactions", "adverse_reactions"),
    ("side effects", "adverse_reactions"),
    ("pregnancy", "specific_populations"),
    ("nursing mothers", "specific_populations"),
    ("labor and delivery", "specific_populations"),
    ("pediatric use", "specific_populations"),
    ("geriatric use", "specific_populations"),
    ("clinical pharmacology", "pharmacology"),
    ("information for patients", "patient_info"),
    ("patient information", "patient_info"),
    ("special precautions", "warnings"),
    ("warnings", "warnings"),
)


@dataclass(frozen=True)
class Section:
    """Une section du fichier : un titre et le texte placé juste sous lui."""

    heading: str  # ex. "5.3 Increased Risk of Bleeding"
    path: tuple[str, ...]  # titres depuis le niveau 2, ex. ("5 WARNINGS...", "5.3 Increased...")
    level: int  # 2 pour "##", 3 pour "###", etc.
    number: str | None  # ex. "5.3", ou None si le titre n'est pas numéroté
    category: str  # catégorie normalisée, héritée du titre de niveau 2
    text: str  # corps de la section, sans ses sous-sections


@dataclass(frozen=True)
class Document:
    """Un fichier du corpus et ses sections, dans l'ordre du fichier."""

    file: str  # ex. "dailymed-warfarin.md"
    source: str  # "dailymed", "medlineplus" ou "glossary"
    drug: str | None  # ex. "warfarin" ; None pour le glossaire
    title: str  # ex. "Warfarin (DailyMed label)"
    url: str | None  # URL d'origine indiquée par la ligne "Source:"
    sections: tuple[Section, ...]


def categorize(heading: str) -> str:
    """Catégorie normalisée d'un titre de niveau 2."""
    number = SECTION_NUMBER_RE.match(heading)
    if number:
        top_level = number.group(1).split(".")[0]
        return NUMBERED_CATEGORIES.get(top_level, "other")
    lowered = heading.lower()
    for pattern, category in TITLE_CATEGORIES:
        if pattern in lowered:
            return category
    return "other"


def identify(path: Path) -> tuple[str, str | None]:
    """Déduit la source et le médicament du nom de fichier.

    "dailymed-warfarin.md" -> ("dailymed", "warfarin") ; "GLOSSARY.md" -> ("glossary", None).
    Le nom de fichier est plus fiable que le texte : la notice de l'apixaban
    parle surtout d'ELIQUIS, son nom de marque.
    """
    stem = path.stem.lower()
    if stem == "glossary":
        return "glossary", None
    source, _, drug = stem.partition("-")
    return source, drug or None


def parse_file(path: Path) -> Document:
    """Lit un fichier markdown du corpus et le découpe en sections."""
    source, drug = identify(path)
    raw = path.read_text(encoding="utf-8")
    url_match = SOURCE_URL_RE.search(raw)

    title = path.stem
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # titres ouverts : (niveau, texte)
    body: list[str] = []
    category = "other"

    def flush() -> None:
        """Enregistre la section en cours si elle a du texte, puis vide le tampon."""
        text = "\n".join(body).strip()
        body.clear()
        if not stack or not text:
            return
        level, heading = stack[-1]
        number = SECTION_NUMBER_RE.match(heading)
        sections.append(
            Section(
                heading=heading,
                path=tuple(h for _, h in stack),
                level=level,
                number=number.group(1) if number else None,
                category=category,
                text=text,
            )
        )

    for line in raw.splitlines():
        match = HEADING_RE.match(line)
        if not match:
            body.append(line)
            continue
        level, heading = len(match.group(1)), match.group(2)
        if level == 1:  # titre du document, pas une section
            title = heading
            body.clear()
            continue
        flush()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))
        if level == 2:
            category = "glossary" if source == "glossary" else categorize(heading)
    flush()

    return Document(
        file=path.name,
        source=source,
        drug=drug,
        title=title,
        url=url_match.group(1) if url_match else None,
        sections=tuple(sections),
    )


def load_documents(corpus_dir: Path) -> list[Document]:
    """Lit tous les fichiers .md du dossier, triés par nom."""
    paths = sorted(corpus_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"Aucun fichier .md trouvé dans {corpus_dir.resolve()}")
    return [parse_file(path) for path in paths]