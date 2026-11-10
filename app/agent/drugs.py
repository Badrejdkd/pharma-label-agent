"""Noms de médicaments : reconnaissance des marques, synonymes et classes.

L'utilisateur peut écrire « Coumadin », « paracétamol » ou « Eliquis » ; le
corpus est indexé sous les noms génériques (warfarin, acetaminophen,
apixaban). Ce module fait le lien dans les deux sens :
- `resolve()` : nom saisi -> nom générique du corpus ;
- `mention_patterns()` : nom générique -> motifs pour repérer, dans une
  autre notice, une mention du médicament ou de sa classe (ex. la notice de
  la warfarine parle des « Serotonin Reuptake Inhibitors », pas de « Zoloft »).
"""

from __future__ import annotations

import re
import unicodedata

from app.ingest import Chunk

# Classes pharmacologiques, d'après GLOSSARY.md et le vocabulaire des notices.
DRUG_CLASSES: dict[str, tuple[str, ...]] = {
    "warfarin": ("anticoagulant", "vitamin K antagonist", "blood thinner", "coumarin"),
    "apixaban": (
        "anticoagulant",
        "factor Xa inhibitor",
        "DOAC",
        "direct oral anticoagulant",
        "blood thinner",
    ),
    "sertraline": (
        "SSRI",
        "selective serotonin reuptake inhibitor",
        "serotonin reuptake inhibitor",
        "antidepressant",
    ),
    "ibuprofen": ("NSAID", "nonsteroidal anti-inflammatory"),
    "aspirin": ("NSAID", "nonsteroidal anti-inflammatory", "antiplatelet", "salicylate"),
    "atorvastatin": ("statin", "HMG-CoA reductase inhibitor"),
    "amoxicillin": ("penicillin", "antibiotic", "antibacterial"),
    "albuterol": ("beta-agonist", "beta-adrenergic", "β-adrenergic", "bronchodilator", "sympathomimetic"),
    "metformin": ("biguanide",),
    "acetaminophen": (),
}

# Noms français courants (l'utilisateur peut poser sa question en français).
FRENCH_NAMES: dict[str, str] = {
    "warfarine": "warfarin",
    "sertraline": "sertraline",
    "metformine": "metformin",
    "atorvastatine": "atorvastatin",
    "amoxicilline": "amoxicillin",
    "ibuprofene": "ibuprofen",
    "aspirine": "aspirin",
    "paracetamol": "acetaminophen",
    "salbutamol": "albuterol",
}

# Marques courantes absentes des listes MedlinePlus du corpus (dont marques françaises).
COMMON_BRANDS: dict[str, str] = {
    "tylenol": "acetaminophen",
    "doliprane": "acetaminophen",
    "efferalgan": "acetaminophen",
    "advil": "ibuprofen",
    "motrin": "ibuprofen",
    "nurofen": "ibuprofen",
    "glucophage": "metformin",
    "proair": "albuterol",
    "ventolin": "albuterol",
    "proventil": "albuterol",
}

SUP_RE = re.compile(r"<sup>.*?</sup>")


def normalize_name(name: str) -> str:
    """Minuscules, sans accents ni symboles de marque, espaces réduits."""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.replace("®", "").replace("™", "")
    return " ".join(text.lower().split())


class UnknownDrugError(ValueError):
    """Le médicament demandé n'est pas couvert par le corpus."""


class DrugResolver:
    """Table de correspondance nom saisi -> nom générique du corpus."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.drugs = sorted({chunk.drug for chunk in chunks if chunk.drug})
        self._aliases: dict[str, str] = {drug: drug for drug in self.drugs}
        for alias, drug in {**FRENCH_NAMES, **COMMON_BRANDS}.items():
            if drug in self.drugs:
                self._aliases[alias] = drug
        # Marques et autres noms listés par MedlinePlus ("Brand names", "Other names").
        for chunk in chunks:
            if chunk.source == "medlineplus" and chunk.drug and chunk.section in ("Brand names", "Other names"):
                for alias in self._names_in_list(chunk.text):
                    self._aliases.setdefault(alias, chunk.drug)

    @staticmethod
    def _names_in_list(text: str) -> list[str]:
        """Extrait les noms d'une liste markdown ("- Zoloft<sup>®</sup>")."""
        names = []
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("- ") or "combination product" in line:
                continue
            name = SUP_RE.split(line[2:])[0]  # texte avant la première balise <sup>
            if name := normalize_name(name):
                names.append(name)
        return names

    def resolve(self, name: str) -> str:
        """Nom générique du corpus ; lève UnknownDrugError si le médicament est absent."""
        key = normalize_name(name)
        if key in self._aliases:
            return self._aliases[key]
        raise UnknownDrugError(
            f"'{name}' is not covered by the corpus. Available drugs: {', '.join(self.drugs)}."
        )

    def aliases_of(self, drug: str) -> list[str]:
        """Tous les noms connus d'un médicament générique (lui compris)."""
        return sorted(alias for alias, target in self._aliases.items() if target == drug)

    def mention_patterns(self, drug: str) -> list[tuple[str, re.Pattern[str]]]:
        """Motifs (libellé, regex) qui signalent une mention du médicament ou de sa classe."""
        terms = [*self.aliases_of(drug), *DRUG_CLASSES.get(drug, ())]
        return [
            (term, re.compile(rf"\b{re.escape(term)}s?\b", re.IGNORECASE))
            for term in dict.fromkeys(terms)  # sans doublons, ordre conservé
        ]