"""Tests for turning a BioCypher output directory into an incremental Neo4j import."""

from pathlib import Path

import pytest

from torchcell.knowledge_graphs.incremental_import import (
    CONSTRAINTS_FILENAME,
    INCREMENTAL_CALL_FILENAME,
    REFERENCE_ANALYSIS_FILENAME,
    CsvGroup,
    analyze_references,
    constraints_cypher,
    discover_csv_groups,
    incremental_import_call,
    incremental_node_header,
    prepare_incremental_import,
    write_incremental_headers,
)

Q = "'"
LAB = f"{Q}Experiment|InformationContentEntity|NamedThing|Entity{Q}"


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    """A BioCypher-shaped output directory: two node labels, two edge types."""
    _write(
        tmp_path / "Experiment-header.csv",
        ":ID\tserialized_data\tid\tpreferred_id\t:LABEL\n",
    )
    _write(
        tmp_path / "Experiment-part000.csv",
        f'e1\t{Q}{{"v": 1}}{Q}\t{Q}e1{Q}\t{Q}e1{Q}\t{LAB}\n'
        f'e2\t{Q}{{"v": 2}}{Q}\t{Q}e2{Q}\t{Q}e2{Q}\t{LAB}\n',
    )
    _write(tmp_path / "Dataset-header.csv", ":ID\tid\tpreferred_id\t:LABEL\n")
    _write(
        tmp_path / "Dataset-part000.csv",
        f"D1\t{Q}D1{Q}\t{Q}D1{Q}\t{Q}Dataset|NamedThing|Entity{Q}\n",
    )
    _write(
        tmp_path / "ExperimentMemberOf-header.csv", ":START_ID\tid\t:END_ID\t:TYPE\n"
    )
    _write(
        tmp_path / "ExperimentMemberOf-part000.csv",
        f"e1\t\tD1\t{Q}ExperimentMemberOf{Q}\ne2\t\tD1\t{Q}ExperimentMemberOf{Q}\n",
    )
    _write(tmp_path / "GenomeMemberOf-header.csv", ":START_ID\tid\t:END_ID\t:TYPE\n")
    # G is NOT a node of this increment (it already exists in the served graph);
    # the G -> X edge joins two external nodes and must be flagged.
    _write(
        tmp_path / "GenomeMemberOf-part000.csv",
        f"G\t\te1\t{Q}GenomeMemberOf{Q}\nG\t\tX\t{Q}GenomeMemberOf{Q}\n",
    )
    return tmp_path


def test_discover_groups_classifies_nodes_and_edges(out_dir: Path) -> None:
    groups = discover_csv_groups(out_dir)
    kinds = {g.label: g.kind for g in groups}
    assert kinds == {
        "Dataset": "node",
        "Experiment": "node",
        "ExperimentMemberOf": "edge",
        "GenomeMemberOf": "edge",
    }
    assert groups[1].label == "Experiment"
    assert [Path(p).name for p in groups[1].part_paths] == ["Experiment-part000.csv"]


def test_discover_groups_rejects_orphans(tmp_path: Path) -> None:
    _write(tmp_path / "Foo-part000.csv", "x\n")
    with pytest.raises(ValueError, match="without a header"):
        discover_csv_groups(tmp_path)
    (tmp_path / "Foo-part000.csv").unlink()
    _write(tmp_path / "Foo-header.csv", ":ID\tid\t:LABEL\n")
    with pytest.raises(ValueError, match="without part files"):
        discover_csv_groups(tmp_path)


def test_incremental_node_header_labels_the_id_and_ignores_the_copy() -> None:
    columns = [":ID", "serialized_data", "id", "preferred_id", ":LABEL"]
    assert incremental_node_header(columns, "Experiment") == [
        "id:ID{label:Entity}",
        "serialized_data",
        "id:IGNORE",
        "preferred_id",
        ":LABEL",
    ]
    with pytest.raises(ValueError, match="':ID'"):
        incremental_node_header(["id", ":LABEL"], "Experiment")
    with pytest.raises(ValueError, match="'id'"):
        incremental_node_header([":ID", ":LABEL"], "Experiment")


def test_write_incremental_headers_only_for_nodes(out_dir: Path) -> None:
    groups = discover_csv_groups(out_dir)
    written = write_incremental_headers(groups)
    assert set(written) == {"Dataset", "Experiment"}
    text = written["Dataset"].read_text(encoding="utf-8")
    assert text == "id:ID{label:Entity}\tid:IGNORE\tpreferred_id\t:LABEL\n"
    # rewritten headers are not mistaken for BioCypher headers on rediscovery
    assert {g.label for g in discover_csv_groups(out_dir)} == {g.label for g in groups}


def test_constraints_cypher_is_one_global_entity_constraint() -> None:
    text = constraints_cypher()
    assert text == (
        "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
        "FOR (n:Entity) REQUIRE n.id IS UNIQUE;\n"
    )


def test_prepare_refuses_a_node_without_the_entity_label(out_dir: Path) -> None:
    _write(out_dir / "Dataset-part000.csv", f"D1\t{Q}D1{Q}\t{Q}D1{Q}\t{Q}Dataset{Q}\n")
    with pytest.raises(ValueError, match="lacks the Entity label"):
        prepare_incremental_import(out_dir, "torchcell")


def test_analyze_references_finds_external_endpoints(out_dir: Path) -> None:
    analysis = analyze_references(discover_csv_groups(out_dir))
    assert analysis.node_counts == {"Dataset": 1, "Experiment": 2}
    assert analysis.edge_counts == {"ExperimentMemberOf": 2, "GenomeMemberOf": 2}
    assert analysis.n_node_ids == 3
    assert analysis.n_external_ids == 2  # G and X
    assert analysis.external_ids_sample == ["G", "X"]
    assert analysis.n_edges_between_external == 1
    assert analysis.edges_between_external_sample == [["GenomeMemberOf", "G", "X"]]
    assert analysis.has_duplicate_edge_risk


def test_incremental_import_call_mirrors_full_build_flags(out_dir: Path) -> None:
    groups = discover_csv_groups(out_dir)
    headers = write_incremental_headers(groups)
    call = incremental_import_call(groups, headers, "torchcell")
    # the database is the FIRST argument: a trailing positional would be swallowed by
    # the preceding --relationships=<files>... option
    assert call.startswith(
        "/var/lib/neo4j/bin/neo4j-admin database import incremental torchcell \\\n"
    )
    assert "--schema" not in call  # unsupported for incremental import on 5.26
    for flag in (
        "--force",
        '--delimiter="\\t"',
        '--array-delimiter="|"',
        '--quote="\'"',
        "--skip-duplicate-nodes=true",
        "--skip-bad-relationships=false",
        "--strict=true",
    ):
        assert flag in call
    assert (
        f'--nodes="{out_dir}/Experiment-header.incremental.csv,{out_dir}/Experiment-part.*"'
        in call
    )
    assert (
        f'--relationships="{out_dir}/GenomeMemberOf-header.csv,{out_dir}/GenomeMemberOf-part.*"'
        in call
    )
    assert call.rstrip().endswith('GenomeMemberOf-part.*"')


def test_prepare_writes_everything(out_dir: Path) -> None:
    plan = prepare_incremental_import(out_dir, "torchcell")
    assert plan.node_labels == ["Dataset", "Experiment"]
    assert plan.edge_types == ["ExperimentMemberOf", "GenomeMemberOf"]
    for name in (
        INCREMENTAL_CALL_FILENAME,
        CONSTRAINTS_FILENAME,
        REFERENCE_ANALYSIS_FILENAME,
        "Experiment-header.incremental.csv",
    ):
        assert (out_dir / name).exists(), name
    script = (out_dir / INCREMENTAL_CALL_FILENAME).read_text(encoding="utf-8")
    assert script.startswith("#!/bin/bash\nset -euo pipefail\n")
    assert plan.analysis.n_edges_between_external == 1


def test_filter_existing_edges_rewrites_only_matching_rows(out_dir: Path) -> None:
    from torchcell.knowledge_graphs.incremental_import import (
        EXISTING_EDGES_FILENAME,
        filter_existing_edges,
    )

    seen: dict[str, list[tuple[str, str]]] = {}

    def lookup(group: CsvGroup, pairs: list[tuple[str, str]]) -> set[tuple[str, str]]:
        seen[group.label] = pairs
        # the store already holds e2 -> D1 and G -> e1
        return {p for p in pairs if p in {("e2", "D1"), ("G", "e1")}}

    summary = filter_existing_edges(out_dir, lookup)
    assert seen["ExperimentMemberOf"] == [("e1", "D1"), ("e2", "D1")]
    assert summary.checked == {"ExperimentMemberOf": 2, "GenomeMemberOf": 2}
    assert summary.existing == {"ExperimentMemberOf": 1, "GenomeMemberOf": 1}
    assert summary.n_existing == 2
    kept = (out_dir / "ExperimentMemberOf-part000.csv").read_text(encoding="utf-8")
    assert kept == f"e1\t\tD1\t{Q}ExperimentMemberOf{Q}\n"
    backup = out_dir / "unfiltered" / "ExperimentMemberOf-part000.csv"
    assert backup.read_text(encoding="utf-8").count("\n") == 2
    assert (out_dir / "GenomeMemberOf-part000.csv").read_text(encoding="utf-8") == (
        f"G\t\tX\t{Q}GenomeMemberOf{Q}\n"
    )
    assert (out_dir / EXISTING_EDGES_FILENAME).exists()
    # node files are untouched and rediscovery still sees the same groups
    assert {g.label for g in discover_csv_groups(out_dir)} == {
        "Dataset",
        "Experiment",
        "ExperimentMemberOf",
        "GenomeMemberOf",
    }


def test_filter_existing_edges_is_a_no_op_when_nothing_exists(out_dir: Path) -> None:
    from torchcell.knowledge_graphs.incremental_import import filter_existing_edges

    before = (out_dir / "ExperimentMemberOf-part000.csv").read_text(encoding="utf-8")
    summary = filter_existing_edges(out_dir, lambda group, pairs: set())
    assert summary.n_existing == 0
    assert (out_dir / "ExperimentMemberOf-part000.csv").read_text(
        encoding="utf-8"
    ) == before
    assert not (out_dir / "unfiltered").exists()
