"""Tests du client LLM qui ne nécessitent pas de réseau."""

from __future__ import annotations

import json

from pydantic import BaseModel

from app.llm import to_strict_schema


class Inner(BaseModel):
    name: str
    note: str | None = None


class Outer(BaseModel):
    items: list[Inner]
    count: int = 0


def test_strict_schema_closes_objects_and_requires_all_fields() -> None:
    schema = to_strict_schema(Outer.model_json_schema())

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"items", "count"}

    inner = schema["$defs"]["Inner"]
    assert inner["additionalProperties"] is False
    assert set(inner["required"]) == {"name", "note"}

    assert '"default"' not in json.dumps(schema)