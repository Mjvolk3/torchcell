# tests/torchcell/knowledge_graphs/test_incremental_import.py
# [[tests.torchcell.knowledge_graphs.test_incremental_import]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/knowledge_graphs/test_incremental_import.py
"""Tests for turning a BioCypher output directory into an incremental Neo4j import.

2026.09.30 (Phase 16). This module prepares the CSV side of an increment (headers,
constraint, call script, reference analysis, existing-edge filter); the ADMISSIBLE /
BLOCKED verdict lives in ``kg_manifest`` and is not exercised here. Fixtures are
BioCypher-shaped directories under ``tmp_path`` (the ``out_dir`` fixture: nodes e1, e2,
D1; edges e1->D1, e2->D1, G->e1, G->X with G and X external), a one-node one-edge
directory for the exact call script, and a scripted session standing in for the bolt
reads (no Neo4j). Expected values, derived from the source:

- The call script is ``#!/bin/bash``, ``set -euo pipefail``, then the 12 fixed parts
  joined by a backslash line continuation indented four spaces, the report file, one
  ``--nodes`` / ``--relationships`` per group in label order.
- ``query_existing_edges`` with ``batch_size=2`` on three pairs runs two queries, the
  first with two ``{"s", "e"}`` rows and the second with one.
- ``filter_existing_edges`` re-reads ``unfiltered/`` on a second run, so a row the first
  run dropped comes back when the second lookup no longer lists it (Finding).
- ``main`` prints ``checked <sum of distinct pairs>``, ``dropped <n_existing>`` and the
  nonzero per-type counts, or ``none``.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from torchcell.knowledge_graphs.incremental_import import (
    CONSTRAINTS_FILENAME,
    EXISTING_EDGES_FILENAME,
    INCREMENTAL_CALL_FILENAME,
    REFERENCE_ANALYSIS_FILENAME,
    CsvGroup,
    _unquote,
    analyze_references,
    constraints_cypher,
    discover_csv_groups,
    filter_existing_edges,
    incremental_import_call,
    incremental_node_header,
    main,
    prepare_incremental_import,
    query_existing_edges,
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
        "--bad-tolerance=1000000000",  # duplicates of existing ids are "bad entries"
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


# --------------------------------------------------------------------------- #
# 2026.09.30 (Phase 16)
# --------------------------------------------------------------------------- #


def test_discover_groups_refuses_a_header_that_is_neither_node_nor_edge(
    tmp_path: Path,
) -> None:
    """A header with no ``:ID`` (or ``*:ID``) and no ``:START_ID`` is refused by path."""
    header = tmp_path / "Odd-header.csv"
    _write(header, "name\tvalue\n")
    _write(tmp_path / "Odd-part000.csv", "a\t1\n")
    with pytest.raises(ValueError) as err:
        discover_csv_groups(tmp_path)
    assert str(err.value) == f"header is neither node nor edge: {header}"


def test_discover_groups_exact_orphan_messages_and_suffix_id(tmp_path: Path) -> None:
    """Orphans are named exactly; a named ``uid:ID`` column also marks a node.

    Parts are collected per label in sorted file order, and non-matching files
    (``README``) are ignored.
    """
    _write(tmp_path / "Foo-part001.csv", "x\n")
    _write(tmp_path / "Bar-part000.csv", "x\n")
    with pytest.raises(ValueError) as orphan:
        discover_csv_groups(tmp_path)
    assert str(orphan.value) == "part files without a header: ['Bar', 'Foo']"
    _write(tmp_path / "Foo-header.csv", "uid:ID\t:LABEL\n")
    _write(tmp_path / "Bar-header.csv", ":START_ID\t:END_ID\n")
    _write(tmp_path / "Foo-part000.csv", "y\n")
    _write(tmp_path / "README", "not a csv\n")
    groups = discover_csv_groups(tmp_path)
    assert [(g.label, g.kind) for g in groups] == [("Bar", "edge"), ("Foo", "node")]
    assert [Path(p).name for p in groups[1].part_paths] == [
        "Foo-part000.csv",
        "Foo-part001.csv",
    ]
    assert groups[1].columns == ["uid:ID", ":LABEL"]
    _write(tmp_path / "Baz-header.csv", ":ID\n")
    with pytest.raises(ValueError) as lonely:
        discover_csv_groups(tmp_path)
    assert str(lonely.value) == (
        f"header without part files: {tmp_path / 'Baz-header.csv'}"
    )


def test_unquote_strips_only_a_matched_single_quote_pair() -> None:
    """``'x'`` -> ``x``, ``''`` -> empty; a lone quote or an unmatched one is kept."""
    assert [_unquote(v) for v in ("'x'", "''", "'", "'x", "x'", "x")] == [
        "x",
        "",
        "'",
        "'x",
        "x'",
        "x",
    ]


def test_analyze_references_caps_both_samples(out_dir: Path) -> None:
    """With ``sample=1`` the counts stay exact and each sample keeps its first entry.

    A second external-external row (H -> Y) makes 2 such edges and 4 external ids
    (G, H, X, Y); the id sample is the sorted first one, G.
    """
    _write(out_dir / "GenomeMemberOf-part001.csv", f"H\t\tY\t{Q}GenomeMemberOf{Q}\n")
    analysis = analyze_references(discover_csv_groups(out_dir), sample=1)
    assert analysis.edge_counts == {"ExperimentMemberOf": 2, "GenomeMemberOf": 3}
    assert analysis.n_external_ids == 4
    assert analysis.external_ids_sample == ["G"]
    assert analysis.n_edges_between_external == 2
    assert analysis.edges_between_external_sample == [["GenomeMemberOf", "G", "X"]]


def test_analyze_references_without_external_edges_has_no_risk(tmp_path: Path) -> None:
    """Every endpoint inside the increment: zero external ids, no duplicate-edge risk."""
    _write(tmp_path / "A-header.csv", ":ID\tid\t:LABEL\n")
    _write(tmp_path / "A-part000.csv", f"a1\t\t{Q}A|Entity{Q}\na2\t\t{Q}A|Entity{Q}\n")
    _write(tmp_path / "L-header.csv", ":START_ID\t:END_ID\n")
    _write(tmp_path / "L-part000.csv", f"{Q}a1{Q}\t{Q}a2{Q}\n")
    analysis = analyze_references(discover_csv_groups(tmp_path))
    assert (analysis.n_node_ids, analysis.n_external_ids) == (2, 0)
    assert analysis.n_edges_between_external == 0
    assert analysis.has_duplicate_edge_risk is False


def test_prepare_writes_the_exact_script_constraint_and_analysis(
    tmp_path: Path,
) -> None:
    """One node label and one edge type: the whole call script, byte for byte."""
    _write(tmp_path / "Gene-header.csv", ":ID\tid\t:LABEL\n")
    _write(tmp_path / "Gene-part000.csv", f"g1\t{Q}g1{Q}\t{Q}Gene|Entity{Q}\n")
    _write(tmp_path / "GeneLink-header.csv", ":START_ID\t:END_ID\t:TYPE\n")
    _write(tmp_path / "GeneLink-part000.csv", f"g1\tg0\t{Q}GeneLink{Q}\n")
    plan = prepare_incremental_import(tmp_path, "tcdb", bin_prefix="/opt/bin/")
    sep = " \\\n    "
    expected_call = sep.join(
        [
            "/opt/bin/neo4j-admin database import incremental tcdb",
            "--force",
            "--verbose",
            '--delimiter="\\t"',
            '--array-delimiter="|"',
            '--quote="\'"',
            "--skip-duplicate-nodes=true",
            "--skip-bad-relationships=false",
            "--strict=true",
            "--threads=8",
            "--max-off-heap-memory=16G",
            "--bad-tolerance=1000000000",
            f"--report-file={tmp_path}/incremental-import.report",
            f'--nodes="{tmp_path}/Gene-header.incremental.csv,{tmp_path}/Gene-part.*"',
            f'--relationships="{tmp_path}/GeneLink-header.csv,{tmp_path}/GeneLink-part.*"',
        ]
    )
    script = Path(plan.call_script)
    assert script.read_text(encoding="utf-8") == (
        "#!/bin/bash\nset -euo pipefail\n" + expected_call + "\n"
    )
    assert script.stat().st_mode & 0o777 == 0o755
    assert (tmp_path / "Gene-header.incremental.csv").read_text(encoding="utf-8") == (
        "id:ID{label:Entity}\tid:IGNORE\t:LABEL\n"
    )
    assert Path(plan.constraints_file).read_text(encoding="utf-8") == (
        "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
        "FOR (n:Entity) REQUIRE n.id IS UNIQUE;\n"
    )
    saved = json.loads(Path(plan.reference_analysis_file).read_text(encoding="utf-8"))
    assert saved == {
        "node_counts": {"Gene": 1},
        "edge_counts": {"GeneLink": 1},
        "n_node_ids": 1,
        "n_external_ids": 1,
        "external_ids_sample": ["g0"],
        "n_edges_between_external": 0,
        "edges_between_external_sample": [],
    }
    assert (plan.database, plan.node_labels, plan.edge_types) == (
        "tcdb",
        ["Gene"],
        ["GeneLink"],
    )


def test_constraints_cypher_snake_cases_a_pascal_label() -> None:
    """``GeneOntologyTerm`` -> ``gene_ontology_term_id_unique``."""
    assert constraints_cypher("GeneOntologyTerm") == (
        "CREATE CONSTRAINT gene_ontology_term_id_unique IF NOT EXISTS "
        "FOR (n:GeneOntologyTerm) REQUIRE n.id IS UNIQUE;\n"
    )


class _ScriptedSession:
    """Records every ``run`` and answers with the rows whose pair is in ``present``."""

    def __init__(self, present: set[tuple[str, str]]) -> None:
        self.present = present
        self.calls: list[dict[str, Any]] = []

    def run(self, cypher: str, **params: Any) -> list[dict[str, str]]:
        self.calls.append({"cypher": cypher, **params})
        return [r for r in params["rows"] if (r["s"], r["e"]) in self.present]

    def __enter__(self) -> "_ScriptedSession":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_query_existing_edges_batches_and_returns_the_union(out_dir: Path) -> None:
    """Three pairs, batch 2: two queries (2 rows, then 1), typed by the group label."""
    group = next(
        g for g in discover_csv_groups(out_dir) if g.label == "ExperimentMemberOf"
    )
    session = _ScriptedSession({("a", "b"), ("e", "f")})
    found = query_existing_edges(
        group, [("a", "b"), ("c", "d"), ("e", "f")], session, batch_size=2
    )
    assert found == {("a", "b"), ("e", "f")}
    assert [call["rows"] for call in session.calls] == [
        [{"s": "a", "e": "b"}, {"s": "c", "e": "d"}],
        [{"s": "e", "e": "f"}],
    ]
    assert {call["t"] for call in session.calls} == {"ExperimentMemberOf"}
    assert session.calls[0]["cypher"] == (
        "UNWIND $rows AS r "
        "MATCH (a:Entity {id: r.s})-[x]->(b:Entity {id: r.e}) "
        "WHERE type(x) = $t "
        "RETURN DISTINCT r.s AS s, r.e AS e"
    )
    assert query_existing_edges(group, [], session) == set()
    assert len(session.calls) == 2


def test_filter_existing_edges_rerun_reads_the_backup(out_dir: Path) -> None:
    """Finding: a second run filters from ``unfiltered/``, not from the current file.

    Run 1 drops e2->D1. Run 2's lookup is asked only about the pairs in the CURRENT
    file ([e1->D1]) and reports e1->D1, but the rows are re-read from the backup, so
    e2->D1, which run 1 found in the store, is written back (incremental_import.py lines
    481 and 493 to 508). Pinned until the rerun reads the current file or re-checks
    every backup pair.
    """
    part = out_dir / "ExperimentMemberOf-part000.csv"
    backup = out_dir / "unfiltered" / "ExperimentMemberOf-part000.csv"
    original = part.read_text(encoding="utf-8")
    filter_existing_edges(
        out_dir, lambda g, pairs: {p for p in pairs if p == ("e2", "D1")}
    )
    assert part.read_text(encoding="utf-8") == f"e1\t\tD1\t{Q}ExperimentMemberOf{Q}\n"
    asked: list[list[tuple[str, str]]] = []

    def second(group: CsvGroup, pairs: list[tuple[str, str]]) -> set[tuple[str, str]]:
        if group.label == "ExperimentMemberOf":
            asked.append(pairs)
            return {("e1", "D1")}
        return set()

    summary = filter_existing_edges(out_dir, second)
    assert asked == [[("e1", "D1")]]
    assert part.read_text(encoding="utf-8") == f"e2\t\tD1\t{Q}ExperimentMemberOf{Q}\n"
    assert backup.read_text(encoding="utf-8") == original
    assert summary.existing == {"ExperimentMemberOf": 1, "GenomeMemberOf": 0}


def test_filter_existing_edges_caps_the_sample_at_twenty(tmp_path: Path) -> None:
    """25 existing rows: all 25 dropped and counted, the sample keeps the first 20."""
    _write(tmp_path / "N-header.csv", ":ID\tid\t:LABEL\n")
    _write(tmp_path / "N-part000.csv", f"n\t\t{Q}Entity{Q}\n")
    _write(tmp_path / "L-header.csv", ":START_ID\t:END_ID\n")
    rows = [(f"s{i:02d}", "n") for i in range(25)]
    _write(tmp_path / "L-part000.csv", "".join(f"{s}\t{e}\n" for s, e in rows))
    summary = filter_existing_edges(tmp_path, lambda g, pairs: set(pairs))
    assert summary.checked == {"L": 25}
    assert summary.existing == {"L": 25}
    assert summary.sample == [["L", s, e] for s, e in rows[:20]]
    assert (tmp_path / "L-part000.csv").read_text(encoding="utf-8") == ""
    saved = json.loads((tmp_path / EXISTING_EDGES_FILENAME).read_text(encoding="utf-8"))
    assert saved["existing"] == {"L": 25}
    assert len(saved["sample"]) == 20


class _FakeDriver:
    """Stands in for ``neo4j.GraphDatabase.driver``; one scripted session."""

    instances: list["_FakeDriver"] = []

    def __init__(self, uri: str, auth: tuple[str, str]) -> None:
        self.uri = uri
        self.auth = auth
        self.databases: list[str] = []
        self.closed = False
        self.session_obj = _ScriptedSession({("e2", "D1"), ("G", "e1")})
        _FakeDriver.instances.append(self)

    def session(self, database: str) -> _ScriptedSession:
        self.databases.append(database)
        return self.session_obj

    def close(self) -> None:
        self.closed = True


def test_main_filters_through_the_driver_and_reports(
    out_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI connects with the env credentials, filters, closes, and prints counts.

    ``checked`` is 2 + 2 distinct pairs and ``dropped`` 2 (e2->D1 and G->e1).
    """
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("NEO4J_URI", "bolt://env-host:7687")
    monkeypatch.setenv("NEO4J_USER", "reader")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    _FakeDriver.instances = []
    monkeypatch.setattr("neo4j.GraphDatabase.driver", _FakeDriver)
    assert main(["filter-existing-edges", "--out-dir", str(out_dir)]) == 0
    (driver,) = _FakeDriver.instances
    assert (driver.uri, driver.auth) == ("bolt://env-host:7687", ("reader", "secret"))
    assert driver.databases == ["torchcell"]
    assert driver.closed is True
    assert capsys.readouterr().out == (
        "existing-edge filter: checked 4 distinct relationships, dropped 2 already in "
        "the store (ExperimentMemberOf=1, GenomeMemberOf=1)\n"
    )


def test_main_uri_override_and_nothing_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--uri`` beats NEO4J_URI; with no existing rows the tail reads ``(none)``."""
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("NEO4J_URI", "bolt://env-host:7687")
    _write(tmp_path / "N-header.csv", ":ID\tid\t:LABEL\n")
    _write(tmp_path / "N-part000.csv", f"n\t\t{Q}Entity{Q}\n")
    _write(tmp_path / "L-header.csv", ":START_ID\t:END_ID\n")
    _write(tmp_path / "L-part000.csv", "a\tn\n")
    _FakeDriver.instances = []
    monkeypatch.setattr("neo4j.GraphDatabase.driver", _FakeDriver)
    code = main(
        [
            "filter-existing-edges",
            "--out-dir",
            str(tmp_path),
            "--database",
            "tc2",
            "--uri",
            "bolt://cli:1",
        ]
    )
    assert code == 0
    (driver,) = _FakeDriver.instances
    assert (driver.uri, driver.databases) == ("bolt://cli:1", ["tc2"])
    assert capsys.readouterr().out == (
        "existing-edge filter: checked 1 distinct relationships, dropped 0 already in "
        "the store (none)\n"
    )
