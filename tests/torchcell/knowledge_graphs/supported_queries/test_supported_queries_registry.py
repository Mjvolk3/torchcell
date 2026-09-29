# tests/torchcell/knowledge_graphs/supported_queries/test_supported_queries_registry.py
# [[tests.torchcell.knowledge_graphs.supported_queries.test_supported_queries_registry]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/knowledge_graphs/supported_queries/test_supported_queries_registry.py
"""The supported-query registry models: byte-stable save, the lifecycle validators, and
the committed ``registry.json``.

Named ``test_supported_queries_registry.py`` (paired in ``pyproject.toml``) because
``tests/torchcell/sequence/genome/test_registry.py`` already owns the basename under
pytest's prepend import mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from torchcell.knowledge_graphs.supported_queries.registry import (
    REGISTRY_RELPATH,
    QueryRegistry,
    SupportedQuery,
    registry_path,
)

REPO = Path(__file__).resolve().parents[4]
COMPOSITE = "44c2336fedab8ff6a85c74c2b94165377b0981f526adb9487895ca6314165e86"


def _query(**overrides: Any) -> SupportedQuery:
    fields: dict[str, Any] = {
        "id": "q_b",
        "title": "Query B",
        "cql_path": "queries/q_b.cql",
        "converter": "torchcell.datamodels.conv.Conv",
        "phenotype_classes": ["FitnessPhenotype"],
        "status": "supported",
        "since_kg_version": "1.2",
        "deprecated_in": None,
        "validated_release": "2026.09.21-ab6d8c5d",
        "dataset_composite": COMPOSITE,
        "docs_page": None,
    }
    fields.update(overrides)
    return SupportedQuery(**fields)


EXPECTED_TEXT = """{
  "schema_version": 1,
  "queries": [
    {
      "id": "q_a",
      "title": "Query B",
      "cql_path": "queries/q_b.cql",
      "converter": null,
      "phenotype_classes": [],
      "status": "deprecated",
      "since_kg_version": null,
      "deprecated_in": "1.3",
      "validated_release": null,
      "dataset_composite": null,
      "docs_page": "docs/source/showcase/a.md"
    },
    {
      "id": "q_b",
      "title": "Query B",
      "cql_path": "queries/q_b.cql",
      "converter": "torchcell.datamodels.conv.Conv",
      "phenotype_classes": [
        "FitnessPhenotype"
      ],
      "status": "supported",
      "since_kg_version": "1.2",
      "deprecated_in": null,
      "validated_release": "2026.09.21-ab6d8c5d",
      "dataset_composite": "44c2336fedab8ff6a85c74c2b94165377b0981f526adb9487895ca6314165e86",
      "docs_page": null
    }
  ]
}
"""


def _two_query_registry() -> QueryRegistry:
    deprecated = _query(
        id="q_a",
        converter=None,
        phenotype_classes=[],
        status="deprecated",
        since_kg_version=None,
        deprecated_in="1.3",
        validated_release=None,
        dataset_composite=None,
        docs_page="docs/source/showcase/a.md",
    )
    return QueryRegistry(queries=[_query(), deprecated])


def test_save_sorts_by_id_and_is_byte_stable(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    _two_query_registry().save(path)
    assert path.read_text(encoding="utf-8") == EXPECTED_TEXT
    reloaded = QueryRegistry.load(path)
    assert [q.id for q in reloaded.queries] == ["q_a", "q_b"]
    reloaded.save(path)
    assert path.read_text(encoding="utf-8") == EXPECTED_TEXT


def test_get_and_replace() -> None:
    registry = _two_query_registry()
    assert registry.get("q_b").title == "Query B"
    updated = registry.replace(_query(title="Renamed"))
    assert updated.get("q_b").title == "Renamed"
    assert updated.get("q_a") == registry.get("q_a")
    assert registry.get("q_b").title == "Query B"
    with pytest.raises(
        KeyError, match=r"no supported query 'q_z'; registered: \['q_a', 'q_b'\]"
    ):
        registry.get("q_z")
    with pytest.raises(KeyError, match="no supported query 'q_z'"):
        registry.replace(_query(id="q_z"))


def test_cql_file_is_relative_to_the_kg_package(tmp_path: Path) -> None:
    assert _query().cql_file(tmp_path) == (
        tmp_path / "torchcell/knowledge_graphs/queries/q_b.cql"
    )
    assert registry_path(tmp_path) == tmp_path / REGISTRY_RELPATH


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"deprecated_in": "1.3"},
            "deprecated_in is set exactly when status is deprecated",
        ),
        (
            {"status": "deprecated"},
            "deprecated_in is set exactly when status is deprecated",
        ),
        (
            {"dataset_composite": None},
            "validated_release and dataset_composite are recorded",
        ),
        (
            {"validated_release": None},
            "validated_release and dataset_composite are recorded",
        ),
        ({"dataset_composite": "abc"}, "dataset_composite must be a sha256 hex digest"),
        ({"cql_path": "/abs/q.cql"}, "cql_path must be a relative .cql path"),
        ({"cql_path": "../q.cql"}, "cql_path must be a relative .cql path"),
        ({"cql_path": "queries/q.cypher"}, "cql_path must be a relative .cql path"),
        ({"id": "Q-1"}, "query id must match"),
        ({"converter": "Conv"}, "converter must be a dotted class path"),
        (
            {"phenotype_classes": ["B", "A"]},
            "phenotype_classes must be sorted and unique",
        ),
        ({"docs": "x"}, "Extra inputs are not permitted"),
    ],
)
def test_invalid_entries_are_refused(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _query(**overrides)


def test_duplicate_ids_are_refused() -> None:
    with pytest.raises(ValidationError, match=r"duplicate query ids: \['q_b'\]"):
        QueryRegistry(queries=[_query(), _query(title="again")])


def test_schema_version_is_one() -> None:
    with pytest.raises(ValidationError, match="Input should be 1"):
        QueryRegistry.model_validate({"schema_version": 2, "queries": []})


def test_committed_registry_is_byte_stable_and_seeded() -> None:
    path = REPO / REGISTRY_RELPATH
    registry = QueryRegistry.load(path)
    assert registry.dumps() == path.read_text(encoding="utf-8")
    assert [q.id for q in registry.queries] == [
        "amino_acid_betaxanthin",
        "essentiality_smf",
        "expression_proteome_morphology",
        "solid_growth_025",
    ]
    assert {q.status for q in registry.queries} == {"supported"}
