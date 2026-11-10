"""BM25 : score de pertinence lexicale, écrit à la main pour rester explicable.

Pour une requête q et un passage d :

    score(d, q) = somme sur les termes t de q de
                  idf(t) * f(t, d) * (k1 + 1) / (f(t, d) + k1 * (1 - b + b * |d| / moyenne))

- f(t, d) : nombre d'occurrences du terme dans le passage ;
- idf(t) : rareté du terme dans le corpus (un terme rare pèse plus) ;
- k1 : saturation (la 10e occurrence d'un terme apporte peu) ;
- b : normalisation par la longueur (un long passage n'est pas favorisé).

BM25 complète les embeddings : il retrouve les termes exacts (eGFR, INR,
noms de médicaments, numéros de section) que la similarité sémantique rate parfois.
"""

from __future__ import annotations

import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    "a an and are as at be been by can could do does for from had has have how i if in into "
    "is it its may might more most must no not of on or other should so such than that the "
    "their them then there these they this to was were what when where which while who will "
    "with would you your".split()
)

# Racinisation minimale : contraindicated / contraindication / contraindications
# deviennent tous "contraindic" ; bleeding -> "bleed" ; monitored -> "monitor".
SUFFIXES = ("ations", "ation", "ated", "ates", "ions", "ion", "ing", "ed", "es", "s")
MIN_STEM_LENGTH = 4


def stem(word: str) -> str:
    """Retire le premier suffixe courant trouvé, si la racine reste assez longue."""
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= MIN_STEM_LENGTH:
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    """Minuscules, mots alphanumériques, sans mots vides, racinisés."""
    return [stem(word) for word in TOKEN_RE.findall(text.lower()) if word not in STOPWORDS]


class BM25:
    """Index BM25 (variante Okapi) sur une liste de textes."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._term_freqs = [Counter(tokenize(document)) for document in documents]
        self._lengths = [sum(freqs.values()) for freqs in self._term_freqs]
        self._avg_length = (sum(self._lengths) / len(self._lengths)) if self._lengths else 1.0

        n_docs = len(documents)
        doc_freqs = Counter(term for freqs in self._term_freqs for term in freqs)
        self._idf = {
            term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freqs.items()
        }

    def scores(self, query: str) -> list[float]:
        """Score BM25 de chaque document pour la requête (0 si aucun terme commun)."""
        terms = [term for term in tokenize(query) if term in self._idf]
        results: list[float] = []
        for freqs, length in zip(self._term_freqs, self._lengths, strict=True):
            norm = self.k1 * (1 - self.b + self.b * length / self._avg_length)
            score = 0.0
            for term in terms:
                freq = freqs.get(term, 0)
                if freq:
                    score += self._idf[term] * freq * (self.k1 + 1) / (freq + norm)
            results.append(score)
        return results