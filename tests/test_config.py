"""Tests de la configuration : aucun appel réseau, le vrai .env est ignoré."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings

ENV_NAMES = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_BASE_UR",
    "OPENAI_MODEL",
    "OPENAI_EMBEDDING_MODEL",
    "LLM_TEMPERATURE",
    "LLM_REASONING_EFFORT",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chaque test part d'un environnement minimal et maîtrisé."""
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")


def load() -> Settings:
    """Charge la configuration sans lire le fichier .env du disque."""
    return Settings(_env_file=None)


def test_strips_quotes_left_by_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "'gpt-5.6-terra'")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", '"text-embedding-3-small"')
    settings = load()
    assert settings.openai_model == "gpt-5.6-terra"
    assert settings.openai_embedding_model == "text-embedding-3-small"


def test_accepts_base_url_typo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_UR", "https://proxy.example")
    assert load().openai_base_url == "https://proxy.example"


def test_correct_base_url_name_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://good.example")
    monkeypatch.setenv("OPENAI_BASE_UR", "https://typo.example")
    assert load().openai_base_url == "https://good.example"


def test_optional_llm_params_absent_by_default() -> None:
    settings = load()
    assert settings.llm_temperature is None
    assert settings.llm_reasoning_effort is None


def test_api_key_is_never_printed() -> None:
    assert "sk-test" not in repr(load())


def test_missing_model_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_MODEL")
    with pytest.raises(ValidationError):
        load()