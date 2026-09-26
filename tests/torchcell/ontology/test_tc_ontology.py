# tests/torchcell/ontology/test_tc_ontology.py
# [[tests.torchcell.ontology.test_tc_ontology]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/ontology/test_tc_ontology.py
"""``print_schema_mappings`` on a two-node, one-edge schema written to ``tmp_path``.

The module header claims it is untested; it is the tooling behind ``make tc-onto``. The
schema-mapping printer reads only the YAML (no Biolink download), so its output is
checked line by line: one explicit Biolink group, one auto-mapped node (``gene`` matches
a Biolink class by name), one unmapped node, and the summary arithmetic
``1/3 explicit + 1 auto-mapped = 2/3 total``. The module is imported inside the tests
because importing biocypher writes a log directory into the working directory.
"""

from pathlib import Path

import pytest
import yaml

SCHEMA = {
    "dataset": {"represented_as": "node", "is_a": "information content entity"},
    "gene": {"represented_as": "node"},
    "widget": {"represented_as": "node"},
    "gene to dataset": {
        "represented_as": "edge",
        "is_a": "association",
        "source": "gene",
        "target": "dataset",
    },
    "free edge": {"represented_as": "edge", "source": "widget", "target": "gene"},
    "not an entity": "a stray string the parser must skip",
}


@pytest.fixture
def schema_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write the schema and run from tmp_path so biocypher's log lands there."""
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "schema.yaml"
    path.write_text(yaml.safe_dump(SCHEMA))
    return path


def test_compact_mapping_groups_nodes_and_edges_by_biolink_parent(
    schema_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Explicit, auto-mapped and unmapped nodes each get a row; the summary counts them."""
    from torchcell.ontology.tc_ontology import print_schema_mappings

    print_schema_mappings(str(schema_path), compact=True)
    out = capsys.readouterr().out
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    assert "information content entity → dataset" in lines
    # the row label is padded to 25 characters; the warning sign is two code points
    assert f"{'✓ auto-mapped':25} → gene" in lines
    assert f"{'⚠️  unmapped':25} → widget" in lines
    assert f"{'association':25} → gene to dataset" in lines
    assert f"{'⚠️  unmapped':25} → free edge" in lines
    assert "Nodes:    1/3 explicit + 1 auto-mapped = 2/3 total" in lines
    assert "Edges:    1/2 mapped to 1 Biolink concepts" in lines
    assert "Total:    2 unique Biolink concepts used" in lines
    assert "⚠️  Warning: 1 unmapped nodes, 1 unmapped edges" in lines
    assert lines[-4:] == [
        "Biolink concepts (2):",
        "• association",
        "• information content entity",
        "═" * 80,
    ]


def test_expanded_mapping_lists_edge_endpoints(
    schema_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The tree format names source -> target for every edge."""
    from torchcell.ontology.tc_ontology import print_schema_mappings

    print_schema_mappings(str(schema_path), compact=False)
    out = capsys.readouterr().out
    assert "└─ gene to dataset: gene → dataset" in out
    assert "└─ free edge: widget → gene" in out
    assert "is_a: information content entity" in out
    assert "Nodes:    1/3 explicit + 1 auto-mapped = 2/3 total" in out


def test_missing_schema_file_prints_an_error_and_returns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wrong path is reported, not raised."""
    monkeypatch.chdir(tmp_path)
    from torchcell.ontology.tc_ontology import print_schema_mappings

    print_schema_mappings(str(tmp_path / "nope.yaml"))
    assert (
        capsys.readouterr().out.strip()
        == f"Error: Schema config not found at {tmp_path / 'nope.yaml'}"
    )
