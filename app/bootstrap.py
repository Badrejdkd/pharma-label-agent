"""Assemblage de l'application : configuration -> LLM -> index -> outils -> agent.

Utilisé par l'API et par les scripts, pour que tout le monde construise
l'agent exactement de la même façon.
"""

from __future__ import annotations

import logging

from app.agent.graph import Agent
from app.agent.tools import Toolbox
from app.config import Settings, get_settings
from app.ingest import load_corpus
from app.llm import LLMClient
from app.retrieval import HybridIndex

logger = logging.getLogger(__name__)


def build_agent(settings: Settings | None = None) -> Agent:
    """Construit l'agent complet ; l'index est calculé (ou lu en cache) ici."""
    settings = settings or get_settings()
    llm = LLMClient(settings)
    chunks = load_corpus(settings.corpus_dir)
    index = HybridIndex.build(chunks, llm, settings.cache_dir)
    logger.info(
        "Agent prêt : %d passages, recherche sémantique %s, modèle %s",
        len(chunks),
        "active" if index.semantic_enabled else "inactive",
        settings.openai_model,
    )
    return Agent(llm, Toolbox(index), max_steps=settings.agent_max_steps)