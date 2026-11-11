"""Modèles de données de l'agent : sorties structurées du LLM et résultat final."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

QuestionType = Literal["lookup", "comparison", "interaction", "threshold", "other"]


# --------------------------------------------------------------------------- #
# Sorties structurées demandées au LLM
# --------------------------------------------------------------------------- #
class Plan(BaseModel):
    """Plan de recherche produit par le planificateur."""

    question_type: QuestionType = Field(
        description="lookup: one drug, one topic; comparison: same topic across drugs; "
        "interaction: drugs taken together; threshold: a patient value against a label limit; other."
    )
    drugs: list[str] = Field(description="Drug names mentioned in the question, as written.")
    sub_questions: list[str] = Field(
        description="At most 4 sub-questions answerable from drug labels or patient leaflets only."
    )
    answer_language: str = Field(description="Language of the user's question, e.g. 'English' or 'French'.")


class ClaimCheck(BaseModel):
    """Vérification d'une affirmation du brouillon par le critique."""

    claim: str = Field(description="The claim, quoted or briefly paraphrased from the draft.")
    verdict: Literal["supported", "unsupported", "ambiguous"]
    evidence_ids: list[str] = Field(description="Ids of the passages that support the claim, if any.")
    comment: str = Field(description="Why the claim is unsupported or ambiguous; empty if supported.")


class Review(BaseModel):
    """Verdict du critique sur un brouillon."""

    claims: list[ClaimCheck]
    verdict: Literal["pass", "revise"]
    revision_instructions: str = Field(description="Precise fixes for the writer; empty if verdict is pass.")
    needs_more_evidence: bool = Field(
        description="True only if part of the question has no supporting passage and more searching could help."
    )
    clinical_flags: list[str] = Field(
        description="Short notes for the reader on clinically ambiguous or patient-specific points."
    )


# --------------------------------------------------------------------------- #
# Résultat renvoyé par l'API
# --------------------------------------------------------------------------- #
class Citation(BaseModel):
    id: int  # numéro affiché dans la réponse : [1], [2]...
    chunk_id: str
    file: str
    section: str
    excerpt: str


class ToolCallTrace(BaseModel):
    tool: str
    arguments: dict[str, Any]
    passages: list[str]  # identifiants des passages renvoyés
    error: str | None = None
    duration_ms: float


class ReviewSummary(BaseModel):
    verdict: Literal["pass", "revised", "flagged"]
    flags: list[str]
    claims: list[ClaimCheck]


class AgentResult(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    disclaimer: str
    plan: Plan | None
    tool_calls: list[ToolCallTrace]
    review: ReviewSummary
    meta: dict[str, Any]