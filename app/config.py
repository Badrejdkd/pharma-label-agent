"""Configuration de l'application, lue depuis les variables d'environnement.

En local, les valeurs viennent du fichier .env ; dans Docker, de
`docker run --env-file .env`. Ce second mode garde les guillemets autour
des valeurs : on les retire ici pour que les deux cas se comportent pareil.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Tous les réglages de l'application, validés au démarrage."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Accès au LLM (proxy LiteLLM compatible OpenAI) ---
    openai_api_key: SecretStr
    openai_base_url: str | None = Field(
        default=None,
        # OPENAI_BASE_UR : faute de frappe présente dans le .env fourni.
        validation_alias=AliasChoices("OPENAI_BASE_URL", "OPENAI_BASE_UR"),
    )
    openai_model: str
    openai_embedding_model: str = "text-embedding-3-small"

    # --- Réglages des appels LLM (None = paramètre non envoyé) ---
    llm_temperature: float | None = None
    llm_reasoning_effort: str | None = None
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2

    # --- Données et agent ---
    corpus_dir: Path = Path("corpus")
    cache_dir: Path = Path(".cache")
    agent_max_steps: int = 6
    log_level: str = "INFO"

    @field_validator("*", mode="before")
    @classmethod
    def strip_quotes(cls, value: Any) -> Any:
        """Retire espaces et guillemets autour des valeurs ; chaîne vide -> None."""
        if isinstance(value, str):
            value = value.strip().strip("'\"").strip()
            return value or None
        return value


@lru_cache
def get_settings() -> Settings:
    """Renvoie la configuration, lue une seule fois puis gardée en cache."""
    return Settings()