"""Seuils chiffrés des notices, vérifiés par du code et non par le LLM.

Les LLM se trompent parfois sur les comparaisons de nombres. Ce module
extrait des phrases comme « contraindicated in patients with an eGFR below
30 » ou « initiation ... between 30 to 45 is not recommended », puis compare
la valeur du patient à chaque seuil de façon déterministe.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

NUMBER = r"(\d+(?:\.\d+)?)"

# (opérateur, motif) : l'ordre compte, "between" doit passer en premier.
COMPARISON_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("between", re.compile(rf"between\s+{NUMBER}\s*(?:to|and|-|–)\s*{NUMBER}", re.IGNORECASE)),
    ("<", re.compile(rf"(?:below|less than|lower than|under|<)\s*{NUMBER}", re.IGNORECASE)),
    (">=", re.compile(rf"(?:at least|≥|>=)\s*{NUMBER}", re.IGNORECASE)),
    (">", re.compile(rf"(?:above|greater than|more than|higher than|exceeding|over|>)\s*{NUMBER}", re.IGNORECASE)),
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;])\s+|\n+")


@dataclass(frozen=True)
class Comparison:
    """Un seuil lu dans le texte, ex. ("<", 30.0, None) ou ("between", 30.0, 45.0)."""

    operator: str
    low: float
    high: float | None = None

    def holds_for(self, value: float) -> bool:
        """Vrai si la valeur du patient satisfait ce seuil (bornes incluses pour "between")."""
        if self.operator == "between":
            return self.low <= value <= (self.high if self.high is not None else self.low)
        if self.operator == "<":
            return value < self.low
        if self.operator == ">=":
            return value >= self.low
        return value > self.low

    def describe(self) -> str:
        if self.operator == "between":
            return f"between {self.low:g} and {self.high:g}"
        return f"{self.operator} {self.low:g}"


def extract_comparisons(sentence: str) -> list[Comparison]:
    """Tous les seuils d'une phrase ; une zone déjà lue n'est pas relue par un autre motif."""
    comparisons: list[Comparison] = []
    taken: list[tuple[int, int]] = []
    for operator, pattern in COMPARISON_PATTERNS:
        for match in pattern.finditer(sentence):
            if any(start < match.end() and match.start() < end for start, end in taken):
                continue
            taken.append(match.span())
            numbers = [float(group) for group in match.groups()]
            comparisons.append(Comparison(operator, numbers[0], numbers[1] if len(numbers) > 1 else None))
    return comparisons


def sentences_mentioning(text: str, parameter: str) -> list[str]:
    """Phrases du texte qui citent le paramètre (insensible à la casse)."""
    needle = parameter.lower()
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if needle in s.lower() and s.strip()]