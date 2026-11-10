"""Tests des outils de l'agent, sur le vrai corpus (aucun appel réseau)."""

from __future__ import annotations

import json

import pytest

from app.agent.drugs import UnknownDrugError
from app.agent.thresholds import extract_comparisons
from app.agent.tools import Toolbox
from app.retrieval import HybridIndex


@pytest.fixture(scope="module")
def toolbox(index: HybridIndex) -> Toolbox:
    return Toolbox(index)


def run(toolbox: Toolbox, name: str, **arguments: object) -> dict:
    """Exécute un outil comme le ferait l'agent et renvoie le JSON destiné au LLM."""
    result = toolbox.execute(name, json.dumps(arguments))
    assert result.error is None, result.error
    return json.loads(result.content)


def test_specs_are_strict_function_declarations(toolbox: Toolbox) -> None:
    specs = toolbox.specs()
    assert {s["function"]["name"] for s in specs} == {
        "search_corpus",
        "get_section",
        "find_interactions",
        "compare_sections",
        "lookup_glossary",
        "check_lab_threshold",
    }
    for spec in specs:
        assert spec["function"]["strict"] is True
        assert spec["function"]["parameters"]["additionalProperties"] is False


@pytest.mark.parametrize(
    ("name", "expected"),
    [("Coumadin", "warfarin"), ("Eliquis", "apixaban"), ("paracétamol", "acetaminophen"), ("ZOLOFT", "sertraline")],
)
def test_brand_and_french_names_are_resolved(toolbox: Toolbox, name: str, expected: str) -> None:
    assert toolbox.drugs.resolve(name) == expected


def test_unknown_drug_is_reported(toolbox: Toolbox) -> None:
    with pytest.raises(UnknownDrugError):
        toolbox.drugs.resolve("rivaroxaban")
    result = toolbox.execute(
        "get_section", {"drug": "rivaroxaban", "category": "contraindications", "source": "dailymed"}
    )
    assert result.error and "not covered by the corpus" in result.error


def test_get_section_returns_whole_contraindications(toolbox: Toolbox) -> None:
    data = run(toolbox, "get_section", drug="warfarin", category="contraindications", source="dailymed")
    assert data["passages"][0]["file"] == "dailymed-warfarin.md"
    assert "Pregnancy" in data["passages"][0]["text"]


def test_compare_sections_covers_each_drug(toolbox: Toolbox) -> None:
    data = run(toolbox, "compare_sections", drugs=["warfarin", "apixaban"], category="contraindications")
    assert set(data["drugs"]) == {"warfarin", "apixaban"}
    apixaban_text = data["drugs"]["apixaban"]["passages"][0]["text"]
    assert "Active pathological bleeding" in apixaban_text


def test_find_interactions_finds_cross_mentions(toolbox: Toolbox) -> None:
    data = run(toolbox, "find_interactions", drug_a="warfarin", drug_b="sertraline")
    warfarin_side = data["warfarin_label_mentions_sertraline"]
    sertraline_side = data["sertraline_label_mentions_warfarin"]
    # La notice de la warfarine cite la sertraline dans son tableau 7.3.
    assert any(p["section"].startswith("7 DRUG INTERACTIONS > 7.3") for p in warfarin_side)
    # La notice de la sertraline recommande de surveiller l'INR (section 5.3).
    assert any("international normalized ratio" in p["text"] for p in sertraline_side)


def test_find_interactions_says_when_nothing_is_stated(toolbox: Toolbox) -> None:
    data = run(toolbox, "find_interactions", drug_a="metformin", drug_b="amoxicillin")
    assert "note" in data


def test_extract_comparisons() -> None:
    comparisons = extract_comparisons("Initiation in patients with an eGFR between 30 to 45 is not recommended")
    assert [(c.operator, c.low, c.high) for c in comparisons] == [("between", 30.0, 45.0)]
    assert extract_comparisons("contraindicated with an eGFR below 30 mL/min")[0].holds_for(25)


def test_check_lab_threshold_for_metformin(toolbox: Toolbox) -> None:
    data = run(toolbox, "check_lab_threshold", drug="metformin", parameter="eGFR", value=25)
    applying = [r for r in data["rules"] if r["applies_to_patient_value"]]
    not_applying = [r for r in data["rules"] if not r["applies_to_patient_value"]]
    assert any("contraindicated" in r["statement"] and "< 30" in r["thresholds"] for r in applying)
    assert any("between 30 and 45" in r["thresholds"] for r in not_applying)


def test_lookup_glossary(toolbox: Toolbox) -> None:
    data = run(toolbox, "lookup_glossary", term="INR")
    assert "warfarin" in data["entries"][0]["definition"]


def test_bad_calls_return_errors_instead_of_raising(toolbox: Toolbox) -> None:
    assert toolbox.execute("search_corpus", "{not json").error
    assert toolbox.execute("unknown_tool", "{}").error
    assert toolbox.execute("get_section", {"drug": "warfarin"}).error  # arguments manquants


def test_results_keep_the_consulted_chunks(toolbox: Toolbox) -> None:
    result = toolbox.execute("search_corpus", {"query": "warfarin bleeding", "drugs": None, "categories": None})
    content_ids = [p["id"] for p in json.loads(result.content)["passages"]]
    assert content_ids == [c.chunk_id for c in result.chunks]