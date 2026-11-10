"""Outils locaux de l'agent, appelés par le LLM via le function calling.

Chaque outil :
- a des arguments décrits par un modèle Pydantic (le schéma JSON envoyé au
  LLM en est généré, et les arguments reçus sont validés avec lui) ;
- est déterministe : il ne fait que lire l'index du corpus ;
- renvoie au LLM du JSON avec l'identifiant de chaque passage, pour que la
  réponse finale puisse les citer et que le critique puisse les vérifier.

C'est le LLM qui choisit quels outils appeler ; ce module ne fait qu'exécuter.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from pydantic import BaseModel, Field, ValidationError

from app.agent.drugs import DrugResolver, UnknownDrugError
from app.agent.thresholds import extract_comparisons, sentences_mentioning
from app.ingest import Chunk
from app.llm import to_strict_schema
from app.retrieval import HybridIndex

Category = Literal[
    "boxed_warning",
    "indications",
    "dosage",
    "contraindications",
    "warnings",
    "adverse_reactions",
    "interactions",
    "specific_populations",
    "overdosage",
    "pharmacology",
    "patient_info",
    "description",
    "other",
]
Source = Literal["dailymed", "medlineplus"]

logger = logging.getLogger(__name__)

MAX_SECTION_CHUNKS = 8  # au-delà, la section est tronquée (ex. "warnings" de la warfarine)
MAX_MATCHES_PER_LABEL = 4
# Sections où chercher les mentions croisées, de la plus à la moins pertinente.
INTERACTION_CATEGORIES = ("interactions", "boxed_warning", "warnings", "contraindications", "patient_info")


# --------------------------------------------------------------------------- #
# Arguments des outils (le schéma JSON envoyé au LLM est généré à partir d'eux)
# --------------------------------------------------------------------------- #
class SearchCorpusArgs(BaseModel):
    query: str = Field(description="What to look for, in English, e.g. 'warfarin bleeding risk monitoring'.")
    drugs: list[str] | None = Field(description="Restrict to these drugs (generic or brand names), or null.")
    categories: list[Category] | None = Field(description="Restrict to these section categories, or null.")


class GetSectionArgs(BaseModel):
    drug: str = Field(description="Drug name, generic or brand, e.g. 'warfarin' or 'Coumadin'.")
    category: Category = Field(description="Section category to return in full.")
    source: Source = Field(
        description="'dailymed' for the FDA label (default choice), 'medlineplus' for the patient leaflet."
    )


class FindInteractionsArgs(BaseModel):
    drug_a: str = Field(description="First drug.")
    drug_b: str = Field(description="Second drug.")


class CompareSectionsArgs(BaseModel):
    drugs: list[str] = Field(description="Two or more drugs to compare.")
    category: Category = Field(description="Section category to compare, e.g. 'contraindications'.")


class LookupGlossaryArgs(BaseModel):
    term: str = Field(description="Term or abbreviation to define, e.g. 'INR', 'DOAC', 'CYP3A4'.")


class CheckLabThresholdArgs(BaseModel):
    drug: str = Field(description="Drug whose label states the thresholds.")
    parameter: str = Field(description="Lab parameter as written in labels, e.g. 'eGFR'.")
    value: float = Field(description="The patient's value, in the label's unit.")


# --------------------------------------------------------------------------- #
# Résultat d'un appel d'outil
# --------------------------------------------------------------------------- #
@dataclass
class ToolResult:
    """Ce qu'un appel d'outil renvoie au LLM, et ce qu'il garde pour la vérification."""

    name: str
    arguments: dict[str, Any]
    content: str  # JSON renvoyé au LLM
    chunks: list[Chunk] = field(default_factory=list)  # passages consultés (preuves citables)
    error: str | None = None
    duration_ms: float = 0.0


class ToolError(Exception):
    """Erreur prévue (médicament inconnu, section absente...) : renvoyée au LLM comme message."""


def passage(chunk: Chunk) -> dict[str, str]:
    """Représentation d'un passage pour le LLM."""
    return {"id": chunk.chunk_id, "file": chunk.file, "section": chunk.section, "text": chunk.text}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[Any], tuple[dict[str, Any], list[Chunk]]]

    def spec(self) -> dict[str, Any]:
        """Déclaration au format function calling d'OpenAI (mode strict)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": to_strict_schema(self.args_model.model_json_schema()),
                "strict": True,
            },
        }


class Toolbox:
    """Les outils de l'agent, branchés sur l'index du corpus."""

    def __init__(self, index: HybridIndex) -> None:
        self.index = index
        self.drugs = DrugResolver(index.chunks)
        self.tools: dict[str, Tool] = {
            tool.name: tool
            for tool in (
                Tool(
                    "search_corpus",
                    "Hybrid keyword + semantic search over the drug-label corpus. Use it when no "
                    "more specific tool fits, or to find details. Filter by drugs and/or "
                    "categories when you know them. Returns passages with ids to cite.",
                    SearchCorpusArgs,
                    self.search_corpus,
                ),
                Tool(
                    "get_section",
                    "Return one whole section of one drug's label, e.g. all contraindications of "
                    "warfarin. Prefer it over search when the question names a drug and a section "
                    "type. Use source 'dailymed' (FDA label) unless patient-level wording is needed.",
                    GetSectionArgs,
                    self.get_section,
                ),
                Tool(
                    "find_interactions",
                    "Find what each drug's label says about the other drug or its drug class "
                    "(interaction tables, warnings, monitoring advice). Use it for any question "
                    "that combines two drugs.",
                    FindInteractionsArgs,
                    self.find_interactions,
                ),
                Tool(
                    "compare_sections",
                    "Return the same section category for several drugs side by side, for "
                    "comparison questions (shared vs unique items).",
                    CompareSectionsArgs,
                    self.compare_sections,
                ),
                Tool(
                    "lookup_glossary",
                    "Define a pharmacology term or abbreviation (INR, DOAC, SSRI, CYP3A4...) "
                    "from the corpus glossary.",
                    LookupGlossaryArgs,
                    self.lookup_glossary,
                ),
                Tool(
                    "check_lab_threshold",
                    "List the numeric thresholds a drug label states for a lab parameter "
                    "(e.g. metformin and eGFR) and check in code which ones apply to the "
                    "patient's value. Use it whenever the question gives a patient measurement.",
                    CheckLabThresholdArgs,
                    self.check_lab_threshold,
                ),
            )
        }

    # ------------------------------------------------------------------ #
    # Interface utilisée par l'agent
    # ------------------------------------------------------------------ #
    def specs(self) -> list[dict[str, Any]]:
        """Déclarations de tous les outils, à passer au LLM."""
        return [tool.spec() for tool in self.tools.values()]

    def execute(self, name: str, raw_arguments: str | dict[str, Any]) -> ToolResult:
        """Exécute un appel d'outil demandé par le LLM ; ne lève jamais d'exception."""
        start = time.perf_counter()
        arguments: dict[str, Any] = {}
        try:
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else dict(raw_arguments)
            tool = self.tools.get(name)
            if tool is None:
                raise ToolError(f"Unknown tool '{name}'. Available tools: {', '.join(self.tools)}.")
            payload, chunks = tool.handler(tool.args_model.model_validate(arguments))
            content, error = json.dumps(payload, ensure_ascii=False), None
        except (ToolError, UnknownDrugError) as exc:
            content, chunks, error = json.dumps({"error": str(exc)}), [], str(exc)
        except (json.JSONDecodeError, ValidationError) as exc:
            message = f"Invalid arguments for '{name}': {exc}"
            content, chunks, error = json.dumps({"error": message}), [], message
        except Exception:  # bug inattendu : on le journalise sans faire échouer toute la requête
            logger.exception("Erreur interne dans l'outil %s", name)
            message = f"Internal error in tool '{name}'."
            content, chunks, error = json.dumps({"error": message}), [], message
        return ToolResult(
            name=name,
            arguments=arguments,
            content=content,
            chunks=chunks,
            error=error,
            duration_ms=round((time.perf_counter() - start) * 1000, 1),
        )

    def corpus_overview(self) -> str:
        """Description courte du corpus, pour le prompt système de l'agent."""
        categories = ", ".join(get_args(Category))
        return f"Drugs in the corpus: {', '.join(self.drugs.drugs)}.\nSection categories: {categories}."

    # ------------------------------------------------------------------ #
    # Outils
    # ------------------------------------------------------------------ #
    def search_corpus(self, args: SearchCorpusArgs) -> tuple[dict[str, Any], list[Chunk]]:
        drugs = [self.drugs.resolve(d) for d in args.drugs] if args.drugs else None
        categories = list(args.categories) if args.categories else None
        hits = self.index.search(args.query, k=6, drugs=drugs, categories=categories)
        chunks = [hit.chunk for hit in hits]
        return {"passages": [passage(c) for c in chunks]}, chunks

    def _section(self, drug: str, category: str, source: str) -> tuple[list[Chunk], bool]:
        """Passages d'une section (tronquée à MAX_SECTION_CHUNKS) et indicateur de troncature."""
        chunks = self.index.section_chunks(drug, category, source)
        return chunks[:MAX_SECTION_CHUNKS], len(chunks) > MAX_SECTION_CHUNKS

    def get_section(self, args: GetSectionArgs) -> tuple[dict[str, Any], list[Chunk]]:
        drug = self.drugs.resolve(args.drug)
        chunks, truncated = self._section(drug, args.category, args.source)
        payload: dict[str, Any] = {"drug": drug, "category": args.category, "passages": [passage(c) for c in chunks]}
        if not chunks:
            payload["note"] = (
                f"The {args.source} document for {drug} has no '{args.category}' section. "
                "Try the other source or search_corpus."
            )
        if truncated:
            payload["note"] = "Section truncated; use search_corpus with this drug and category for details."
        return payload, chunks

    def compare_sections(self, args: CompareSectionsArgs) -> tuple[dict[str, Any], list[Chunk]]:
        if len(args.drugs) < 2:
            raise ToolError("compare_sections needs at least two drugs.")
        payload: dict[str, Any] = {"category": args.category, "drugs": {}}
        all_chunks: list[Chunk] = []
        for name in args.drugs:
            drug = self.drugs.resolve(name)
            chunks, _ = self._section(drug, args.category, "dailymed")
            source = "dailymed"
            if not chunks:  # ex. notices sans ordonnance : pas de section "contraindications"
                chunks, _ = self._section(drug, args.category, "medlineplus")
                source = "medlineplus"
            payload["drugs"][drug] = {
                "source": source if chunks else None,
                "passages": [passage(c) for c in chunks],
            }
            all_chunks.extend(chunks)
        return payload, all_chunks

    def _mentions(self, label_drug: str, other_drug: str) -> tuple[list[dict[str, Any]], list[Chunk]]:
        """Passages de la notice de `label_drug` qui mentionnent `other_drug` ou sa classe."""
        patterns = self.drugs.mention_patterns(other_drug)
        scored: list[tuple[Chunk, list[str]]] = []
        for chunk in self.index.chunks:
            if chunk.drug != label_drug or chunk.category not in INTERACTION_CATEGORIES:
                continue
            matched = [term for term, pattern in patterns if pattern.search(chunk.text)]
            if matched:
                scored.append((chunk, matched))
        # Notice FDA avant MedlinePlus, section d'interactions avant avertissements, etc.
        scored.sort(
            key=lambda item: (
                item[0].source != "dailymed",
                INTERACTION_CATEGORIES.index(item[0].category),
                -len(item[1]),
                item[0].order,
            )
        )
        selected = scored[:MAX_MATCHES_PER_LABEL]
        results = [{**passage(chunk), "matched_terms": matched} for chunk, matched in selected]
        return results, [chunk for chunk, _ in selected]

    def find_interactions(self, args: FindInteractionsArgs) -> tuple[dict[str, Any], list[Chunk]]:
        drug_a, drug_b = self.drugs.resolve(args.drug_a), self.drugs.resolve(args.drug_b)
        if drug_a == drug_b:
            raise ToolError("find_interactions needs two different drugs.")
        a_mentions_b, chunks_ab = self._mentions(drug_a, drug_b)
        b_mentions_a, chunks_ba = self._mentions(drug_b, drug_a)
        payload = {
            f"{drug_a}_label_mentions_{drug_b}": a_mentions_b,
            f"{drug_b}_label_mentions_{drug_a}": b_mentions_a,
        }
        if not a_mentions_b and not b_mentions_a:
            payload["note"] = (
                "Neither label explicitly mentions the other drug or its class. Say so; "
                "do not infer an interaction that the corpus does not state."
            )
        return payload, chunks_ab + chunks_ba

    def lookup_glossary(self, args: LookupGlossaryArgs) -> tuple[dict[str, Any], list[Chunk]]:
        needle = args.term.lower()
        entries: list[dict[str, str]] = []
        chunks: list[Chunk] = []
        for chunk in self.index.chunks:
            if chunk.source != "glossary":
                continue
            lines = [line.strip("- ").strip() for line in chunk.text.splitlines() if needle in line.lower()]
            if lines:
                entries.extend({"id": chunk.chunk_id, "definition": line} for line in lines)
                chunks.append(chunk)
        if not entries:
            raise ToolError(f"'{args.term}' is not in the glossary. Try search_corpus.")
        return {"term": args.term, "entries": entries}, chunks

    def check_lab_threshold(self, args: CheckLabThresholdArgs) -> tuple[dict[str, Any], list[Chunk]]:
        drug = self.drugs.resolve(args.drug)
        rules: list[dict[str, Any]] = []
        chunks: list[Chunk] = []
        seen: set[str] = set()
        for chunk in self.index.chunks:
            if chunk.drug != drug or chunk.source != "dailymed":
                continue
            for sentence in sentences_mentioning(chunk.text, args.parameter):
                comparisons = extract_comparisons(sentence)
                key = " ".join(sentence.lower().split())
                if not comparisons or key in seen:
                    continue
                seen.add(key)
                rules.append(
                    {
                        "id": chunk.chunk_id,
                        "section": chunk.section,
                        "statement": sentence,
                        "thresholds": [c.describe() for c in comparisons],
                        "applies_to_patient_value": any(c.holds_for(args.value) for c in comparisons),
                    }
                )
                if chunk not in chunks:
                    chunks.append(chunk)
        if not rules:
            raise ToolError(f"No numeric threshold for '{args.parameter}' found in the {drug} label.")
        return {
            "drug": drug,
            "parameter": args.parameter,
            "patient_value": args.value,
            "rules": rules,
            "note": "Comparisons are computed in code. Units are not converted: check they match.",
        }, chunks