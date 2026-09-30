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

2026.09.30 (Phase 15): the committed ``biocypher/config/torchcell_schema_config.yaml``
is read as the real table: 26 nodes, of which 22 sit under five Biolink parents
(environmental exposure 4, genotype 2, information content entity 2, nucleic acid
entity 1, phenotypic feature 13) and 4 are auto-mapped by name (dataset, genome,
genotype, publication); 13 edges under five relations (coexists with 1, genetically
associated with 1, mentions 1, part of 6, participates in 4); 10 concepts in all. Two
Findings are pinned: the compact headers are the literals 16 and 11 whatever the schema
holds, and list-valued edge endpoints print as Python list reprs. A fully mapped
three-node schema exercises the no-warning branches of both formats; ``BioCypher`` is
replaced by a recorder for the two delegating printers, so nothing is fetched.
"""

import sys
from pathlib import Path
from typing import Any

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


# --- 2026.09.30 (Phase 15): the real schema, the no-warning branches, the CLI ------- #
REAL_SCHEMA = (
    Path(__file__).resolve().parents[3]
    / "biocypher"
    / "config"
    / "torchcell_schema_config.yaml"
)

FULLY_MAPPED = {
    "experiment": {"represented_as": "node", "is_a": "information content entity"},
    "dataset": {"represented_as": "node", "is_a": "information content entity"},
    "genome": {"represented_as": "node"},
    "experiment member of": {
        "represented_as": "edge",
        "is_a": "part of",
        "source": "experiment",
        "target": "dataset",
    },
    "genome member of": {
        "represented_as": "edge",
        "is_a": "part of",
        "source": "genome",
        "target": "dataset",
    },
    "gene symbol": {"represented_as": "node property", "is_a": "string"},
}


def _lines(out: str) -> list[str]:
    return [line.strip() for line in out.splitlines() if line.strip()]


def test_real_schema_compact_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The committed ``torchcell_schema_config.yaml``: 26 nodes (22 under five Biolink
    parents, 4 auto-mapped by name), 13 edges under five relations, 10 concepts in all.
    Changing the schema changes this table, which is the point: the ``tc-onto`` view is
    what the knowledge graph's classes map to.
    """
    monkeypatch.chdir(tmp_path)
    from torchcell.ontology.tc_ontology import print_schema_mappings

    print_schema_mappings(str(REAL_SCHEMA), compact=True)
    lines = _lines(capsys.readouterr().out)
    rows = [line for line in lines if "→" in line and not line.startswith("TORCH")]
    assert rows == [
        f"{'environmental exposure':25} → environment, environment perturbation, "
        "media, temperature",
        f"{'genotype':25} → perturbation, segregant genotype",
        "information content entity → experiment, experiment reference",
        f"{'nucleic acid entity':25} → crispr construct",
        f"{'phenotypic feature':25} → calmorph phenotype, environment response "
        "phenotype, fitness phenotype, gene essentiality phenotype, gene interaction "
        "phenotype, metabolite phenotype, microarray expression phenotype, protein "
        "abundance phenotype, pseudobulk expression phenotype, rnaseq expression "
        "phenotype, synthetic lethality phenotype, synthetic rescue phenotype, visual "
        "score phenotype",
        f"{'✓ auto-mapped':25} → dataset, genome, genotype, publication",
        f"{'coexists with':25} → experiment reference of",
        "genetically associated with → perturbation member of",
        f"{'mentions':25} → publication mentions experiment",
        f"{'part of':25} → crispr construct member of, environment perturbation "
        "member of, experiment member of, experiment reference member of, media "
        "member of, temperature member of",
        f"{'participates in':25} → environment member of, genome member of, "
        "genotype member of, phenotype member of",
    ]
    assert "Nodes:    22/26 explicit + 4 auto-mapped = 26/26 total" in lines
    assert "Edges:    13/13 mapped to 5 Biolink concepts" in lines
    assert "Total:    10 unique Biolink concepts used" in lines
    assert "✓ 4 nodes auto-mapped by name matching" in lines
    assert not any("Warning" in line for line in lines)


def test_compact_headers_are_hardcoded_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Finding: the compact headers print the literals ``NODES (16 total)`` and ``EDGES
    (11 total)`` (tc_ontology.py:147 and 160) whatever the schema holds; the real schema
    has 26 nodes and 13 edges, and the three-node test schema prints the same 16 and 11.
    Pinned until the headers use ``len(nodes)`` and ``len(edges)``.
    """
    monkeypatch.chdir(tmp_path)
    from torchcell.ontology.tc_ontology import print_schema_mappings

    small = tmp_path / "schema.yaml"
    small.write_text(yaml.safe_dump(SCHEMA))
    for path in (REAL_SCHEMA, small):
        print_schema_mappings(str(path), compact=True)
        lines = _lines(capsys.readouterr().out)
        assert "📦 NODES (16 total)" in lines
        assert "🔗 EDGES (11 total)" in lines


def test_real_schema_expanded_prints_list_endpoints_as_python_reprs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Finding: an edge whose ``source`` or ``target`` is a list in the YAML is printed
    with the list's Python repr (tc_ontology.py:234-236), brackets and quotes included.
    Single endpoints print bare. Pinned until list endpoints are joined.
    """
    monkeypatch.chdir(tmp_path)
    from torchcell.ontology.tc_ontology import print_schema_mappings

    print_schema_mappings(str(REAL_SCHEMA), compact=False)
    lines = _lines(capsys.readouterr().out)
    assert (
        "└─ genotype member of: ['genotype', 'segregant genotype'] → experiment"
        in lines
    )
    assert (
        "└─ environment member of: environment → ['experiment', 'experiment reference']"
        in lines
    )
    assert "└─ publication mentions experiment: publication → experiment" in lines
    assert "✓ 4 nodes auto-mapped by name matching" in lines


@pytest.mark.parametrize("compact", [True, False])
def test_fully_mapped_schema_prints_no_warning_and_skips_properties(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    compact: bool,
) -> None:
    """Two explicit nodes under one parent (sorted ``dataset, experiment``), one
    auto-mapped node (``genome``), two edges under ``part of``, no unmapped entity: the
    summary reads ``2/3 explicit + 1 auto-mapped = 3/3 total`` and ``2/2 mapped to 1``,
    ends with the auto-mapped line instead of a warning, and the ``node property`` entry
    is in neither count.
    """
    monkeypatch.chdir(tmp_path)
    from torchcell.ontology.tc_ontology import print_schema_mappings

    path = tmp_path / "full.yaml"
    path.write_text(yaml.safe_dump(FULLY_MAPPED))
    print_schema_mappings(str(path), compact=compact)
    lines = _lines(capsys.readouterr().out)
    assert "Nodes:    2/3 explicit + 1 auto-mapped = 3/3 total" in lines
    assert "Edges:    2/2 mapped to 1 Biolink concepts" in lines
    assert "Total:    2 unique Biolink concepts used" in lines
    assert "✓ 1 nodes auto-mapped by name matching" in lines
    assert not any("unmapped" in line or "No Biolink" in line for line in lines)
    assert not any("gene symbol" in line for line in lines)
    if compact:
        assert "information content entity → dataset, experiment" in lines
        assert f"{'part of':25} → experiment member of, genome member of" in lines
    else:
        assert lines[lines.index("🔗 is_a: information content entity") + 1 :][:2] == [
            "└─ dataset",
            "└─ experiment",
        ]
        assert "└─ genome member of: genome → dataset" in lines


@pytest.mark.parametrize("compact", [True, False])
def test_schema_with_neither_auto_mapped_nor_unmapped_prints_neither_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    compact: bool,
) -> None:
    """Every node explicit, in both formats: no auto-mapped row or section, no warning
    and no auto-mapped summary line.
    """
    monkeypatch.chdir(tmp_path)
    from torchcell.ontology.tc_ontology import print_schema_mappings

    path = tmp_path / "explicit.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "represented_as": "node",
                    "is_a": "information content entity",
                }
            }
        )
    )
    print_schema_mappings(str(path), compact=compact)
    lines = _lines(capsys.readouterr().out)
    assert "Nodes:    1/1 explicit + 0 auto-mapped = 1/1 total" in lines
    assert "Edges:    0/0 mapped to 0 Biolink concepts" in lines
    assert not any("auto-mapped by name" in line or "Warning" in line for line in lines)
    assert not any(line.startswith("✓ Auto-mapped") for line in lines)
    assert not any(line.startswith("✓ auto-mapped") for line in lines)


class _FakeBioCypher:
    """Records the constructor arguments and the delegated calls."""

    calls: list[tuple[str, dict[str, Any]]] = []

    def __init__(self, **kwargs: Any) -> None:
        _FakeBioCypher.calls.append(("init", kwargs))

    def show_ontology_structure(self, **kwargs: Any) -> None:
        _FakeBioCypher.calls.append(("show_ontology_structure", kwargs))

    def summary(self) -> None:
        _FakeBioCypher.calls.append(("summary", {}))


def test_structure_and_summary_delegate_to_an_offline_biocypher(  # test-quality: allow both printers return None; asserted through the recorded BioCypher calls
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both printers build ``BioCypher(offline=True, schema_config_path=...)`` (offline so
    no database is contacted) and forward ``full`` and ``to_disk`` unchanged.
    """
    monkeypatch.chdir(tmp_path)
    from torchcell.ontology import tc_ontology

    monkeypatch.setattr(tc_ontology, "BioCypher", _FakeBioCypher)
    _FakeBioCypher.calls = []
    tc_ontology.print_ontology_structure("s.yaml", full=True, to_disk="out")
    tc_ontology.print_ontology_summary("t.yaml")
    assert _FakeBioCypher.calls == [
        ("init", {"offline": True, "schema_config_path": "s.yaml"}),
        ("show_ontology_structure", {"full": True, "to_disk": "out"}),
        ("init", {"offline": True, "schema_config_path": "t.yaml"}),
        ("summary", {}),
    ]


def test_main_passes_the_config_and_expand_flag(  # test-quality: allow main() returns None; asserted through the recorded print_schema_mappings calls
    schema_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``tc-onto -c PATH`` prints compact; ``--expand`` turns ``compact`` off; with no
    ``-c`` the repo-relative default config path is used.
    """
    from torchcell.ontology import tc_ontology

    seen: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        tc_ontology,
        "print_schema_mappings",
        lambda schema_config_path, compact: seen.append((schema_config_path, compact)),
    )
    for argv in (
        ["tc-onto", "-c", str(schema_path)],
        ["tc-onto", "--expand", "--config", str(schema_path)],
        ["tc-onto", "-e"],
    ):
        monkeypatch.setattr(sys, "argv", argv)
        tc_ontology.main()
    assert seen == [
        (str(schema_path), True),
        (str(schema_path), False),
        ("biocypher/config/torchcell_schema_config.yaml", False),
    ]
