"""Citations et avertissement : traitements déterministes, sans LLM.

Le rédacteur cite les passages par leur identifiant brut, ex.
[dailymed-warfarin:11]. Ce module :
- extrait les identifiants cités et repère ceux qui n'existent pas ;
- renumérote les citations en [1], [2]... dans l'ordre d'apparition ;
- fournit l'avertissement médical, ajouté par le code et jamais par le LLM.
"""

from __future__ import annotations

import re

from app.agent.schemas import Citation
from app.ingest import Chunk

CITATION_RE = re.compile(r"\[([A-Za-z0-9_.-]+:\d+)\]")
EXCERPT_CHARS = 280

DISCLAIMERS = {
    "english": (
        "This answer is generated from FDA drug labels (DailyMed) and MedlinePlus leaflets for a "
        "technical demonstration. It is not medical advice: always consult a doctor or pharmacist "
        "before starting, stopping or combining medicines."
    ),
    "french": (
        "Cette réponse est générée à partir de notices FDA (DailyMed) et de fiches MedlinePlus, dans "
        "le cadre d'une démonstration technique. Ce n'est pas un avis médical : consultez toujours un "
        "médecin ou un pharmacien avant de commencer, d'arrêter ou d'associer des médicaments."
    ),
}


def disclaimer_for(language: str | None) -> str:
    """Avertissement dans la langue de la réponse (anglais par défaut)."""
    return DISCLAIMERS.get((language or "").strip().lower(), DISCLAIMERS["english"])


def cited_ids(text: str) -> list[str]:
    """Identifiants cités, sans doublons, dans l'ordre d'apparition."""
    return list(dict.fromkeys(CITATION_RE.findall(text)))


def unknown_citations(text: str, evidence: dict[str, Chunk]) -> list[str]:
    """Identifiants cités qui ne correspondent à aucun passage récupéré."""
    return [chunk_id for chunk_id in cited_ids(text) if chunk_id not in evidence]


def renumber(text: str, evidence: dict[str, Chunk]) -> tuple[str, list[Citation]]:
    """Remplace [chunk_id] par [n] et construit la liste des sources.

    Une citation inconnue est retirée du texte : on ne montre jamais une
    source qui n'a pas été réellement consultée.
    """
    numbers: dict[str, int] = {}
    citations: list[Citation] = []

    def replace(match: re.Match[str]) -> str:
        chunk_id = match.group(1)
        chunk = evidence.get(chunk_id)
        if chunk is None:
            return ""
        if chunk_id not in numbers:
            numbers[chunk_id] = len(numbers) + 1
            excerpt = " ".join(chunk.text.split())
            citations.append(
                Citation(
                    id=numbers[chunk_id],
                    chunk_id=chunk_id,
                    file=chunk.file,
                    section=chunk.section,
                    excerpt=excerpt[:EXCERPT_CHARS] + ("…" if len(excerpt) > EXCERPT_CHARS else ""),
                )
            )
        return f"[{numbers[chunk_id]}]"

    return CITATION_RE.sub(replace, text), citations