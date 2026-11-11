"""Client LLM : le seul point d'entrée du code vers le proxy LiteLLM.

Responsabilités :
- créer le client OpenAI à partir de la configuration ;
- n'envoyer que les paramètres configurés (température, effort de raisonnement) ;
- obtenir une sortie JSON validée par Pydantic : mode strict si le modèle le
  gère, sinon mode JSON simple avec une nouvelle tentative ;
- calculer des embeddings par lots.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import TYPE_CHECKING, Any, TypeVar

from openai import APIError, BadRequestError, OpenAI
from pydantic import BaseModel, ValidationError

from app.config import Settings

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessage

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
EMBED_BATCH_SIZE = 128


class LLMError(RuntimeError):
    """Le LLM est injoignable ou a renvoyé une réponse inexploitable."""


def to_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Adapte un schéma JSON produit par Pydantic au mode strict d'OpenAI.

    Le mode strict exige que chaque objet interdise les champs supplémentaires
    et déclare tous ses champs comme obligatoires ; il ne gère pas les valeurs
    par défaut, qu'on retire donc.
    """

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    strict = copy.deepcopy(schema)
    walk(strict)
    return strict


class LLMClient:
    """Enveloppe fine autour du SDK OpenAI, configurée par `Settings`."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.openai_model
        self.embedding_model = settings.openai_embedding_model
        self._client = OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            base_url=settings.openai_base_url,
            timeout=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,  # nouvelles tentatives intégrées au SDK
        )
        # None = pas encore testé ; True / False = résultat mémorisé au premier essai.
        self._strict_json_supported: bool | None = None

    @property
    def strict_json_supported(self) -> bool | None:
        """Indique si le modèle a accepté le mode JSON strict."""
        return self._strict_json_supported

    def _options(self) -> dict[str, Any]:
        """Paramètres optionnels, envoyés seulement s'ils sont configurés."""
        options: dict[str, Any] = {}
        if self.settings.llm_temperature is not None:
            options["temperature"] = self.settings.llm_temperature
        if self.settings.llm_reasoning_effort:
            options["reasoning_effort"] = self.settings.llm_reasoning_effort
        return options

    def _create(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatCompletionMessage:
        """Appel brut à l'API ; laisse remonter les erreurs du SDK."""
        response = self._client.chat.completions.create(
            model=self.model, messages=messages, **self._options(), **kwargs
        )
        return response.choices[0].message

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
    ) -> ChatCompletionMessage:
        """Un appel de chat ; renvoie le message de l'assistant (texte et/ou appels d'outils).

        tool_choice : "auto" (le modèle décide) ou "required" (au moins un appel d'outil).
        """
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        try:
            return self._create(messages, **kwargs)
        except APIError as exc:
            raise LLMError(f"Appel au LLM impossible : {exc}") from exc

    def structured(self, messages: list[dict[str, Any]], output_type: type[T]) -> T:
        """Appel dont la réponse doit respecter le schéma Pydantic `output_type`."""
        if self._strict_json_supported is not False:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": output_type.__name__,
                    "strict": True,
                    "schema": to_strict_schema(output_type.model_json_schema()),
                },
            }
            try:
                message = self._create(messages, response_format=response_format)
                self._strict_json_supported = True
                return output_type.model_validate_json(message.content or "")
            except BadRequestError as exc:
                logger.warning("Mode JSON strict refusé, repli sur le mode JSON simple : %s", exc)
                self._strict_json_supported = False
            except ValidationError as exc:
                logger.warning("Réponse JSON stricte invalide, repli sur le mode JSON simple : %s", exc)
            except APIError as exc:
                raise LLMError(f"Appel au LLM impossible : {exc}") from exc
        return self._structured_json_mode(messages, output_type)

    def _structured_json_mode(
        self, messages: list[dict[str, Any]], output_type: type[T], attempts: int = 2
    ) -> T:
        """Repli : mode JSON simple, schéma donné dans le prompt, validation Pydantic."""
        schema = json.dumps(output_type.model_json_schema(), ensure_ascii=False)
        conversation = [
            *messages,
            {
                "role": "system",
                "content": "Reply with a single JSON object that matches this JSON schema, "
                "with no other text:\n" + schema,
            },
        ]
        last_error: Exception | None = None
        for _ in range(attempts):
            try:
                message = self._create(conversation, response_format={"type": "json_object"})
            except APIError as exc:
                raise LLMError(f"Appel au LLM impossible : {exc}") from exc
            try:
                return output_type.model_validate_json(message.content or "")
            except ValidationError as exc:
                last_error = exc
                conversation = [
                    *conversation,
                    {"role": "assistant", "content": message.content or ""},
                    {"role": "user", "content": f"This JSON is invalid: {exc}. Return a corrected JSON object only."},
                ]
        raise LLMError(f"Réponse JSON invalide après {attempts} essais : {last_error}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embeddings par lots ; l'ordre des vecteurs suit l'ordre des textes."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            try:
                response = self._client.embeddings.create(model=self.embedding_model, input=batch)
            except APIError as exc:
                raise LLMError(f"Calcul des embeddings impossible : {exc}") from exc
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return vectors