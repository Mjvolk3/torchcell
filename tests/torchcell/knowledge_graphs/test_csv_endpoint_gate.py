"""The CSV endpoint gate on small BioCypher-shaped directories under ``tmp_path``.

The fixture is a four-dataset-free miniature of a build: node files for ``Dataset``,
``Experiment``, ``Genotype`` and ``Perturbation`` (ids are 64-hex content ids except the
dataset's class name, as BioCypher writes them), and the ``member of`` ladder
``PerturbationMemberOf -> GenotypeMemberOf -> ExperimentMemberOf`` plus one
``Mentions`` edge from a publication id that is quoted and not hex. The gaps planted are
the real shape: a ``perturbation member of`` row whose start is a perturbation no lane
emitted (the Carruthers gene addition), resolved to its dataset through the ladder.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from torchcell.knowledge_graphs import csv_endpoint_gate as gate


def _h(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def _write(path: Path, rows: list[list[str]]) -> None:
    path.write_text("".join("\t".join(row) + "\n" for row in rows), encoding="utf-8")


def _nodes(out_dir: Path, label: str, ids: list[str], blob: str = "'{}'") -> None:
    _write(
        out_dir / f"{label}-header.csv", [[":ID", "id", "serialized_data", ":LABEL"]]
    )
    _write(
        out_dir / f"{label}-part000.csv",
        [[i, f"'{i}'", blob, f"'{label}|Entity'"] for i in ids],
    )


def _edges(out_dir: Path, label: str, pairs: list[tuple[str, str]]) -> None:
    _write(out_dir / f"{label}-header.csv", [[":START_ID", "id", ":END_ID", ":TYPE"]])
    _write(
        out_dir / f"{label}-part000.csv", [[s, "", e, f"'{label}'"] for s, e in pairs]
    )


P1, P2, P_MISSING = _h("p1"), _h("p2"), _h("gene addition no lane emitted")
G1, G2 = _h("g1"), _h("g2")
E1, E2 = _h("e1"), _h("e2")
DS_A, DS_B = "FitnessADataset", "FitnessBDataset"


@pytest.fixture
def clean_dir(tmp_path: Path) -> Path:
    """Every endpoint is a node: two experiments in two datasets, one perturbation each."""
    _nodes(tmp_path, "Dataset", [DS_A, DS_B])
    _nodes(tmp_path, "Experiment", [E1, E2], blob='\'{"k": "a multi-field\tblob"}\'')
    _nodes(tmp_path, "Genotype", [G1, G2])
    _nodes(tmp_path, "Perturbation", [P1, P2])
    _nodes(tmp_path, "Publication", ["'12345'"])
    _edges(tmp_path, "PerturbationMemberOf", [(P1, G1), (P2, G2)])
    _edges(tmp_path, "GenotypeMemberOf", [(G1, E1), (G2, E2)])
    _edges(tmp_path, "ExperimentMemberOf", [(E1, DS_A), (E2, DS_B)])
    _edges(tmp_path, "Mentions", [("'12345'", E1), ("'12345'", E2)])
    return tmp_path


def test_pack_id_packs_hex_content_ids_and_keeps_names_and_quoted_ids() -> None:
    assert gate.pack_id(P1.encode()) == bytes.fromhex(P1)
    assert gate.pack_id(f"'{P1}'".encode()) == bytes.fromhex(P1)
    assert gate.pack_id(b"FitnessADataset") == b"FitnessADataset"
    assert gate.pack_id(b"'12345'") == b"12345"
    # 64 characters that are not hex stay as they are
    assert gate.pack_id(b"z" * 64) == b"z" * 64


def test_a_clean_directory_passes_with_every_row_counted(clean_dir: Path) -> None:
    report = gate.check_endpoints(clean_dir)
    assert report.ok
    assert report.gaps == []
    assert report.node_rows == {
        "Dataset": 2,
        "Experiment": 2,
        "Genotype": 2,
        "Perturbation": 2,
        "Publication": 1,
    }
    assert report.edge_rows == {
        "ExperimentMemberOf": 2,
        "GenotypeMemberOf": 2,
        "Mentions": 2,
        "PerturbationMemberOf": 2,
    }
    assert report.n_node_ids == 9
    assert "OK: every edge endpoint is the id of a node" in gate.format_report(report)


def test_a_dangling_start_is_named_with_its_dataset_through_the_ladder(
    clean_dir: Path,
) -> None:
    """The Carruthers shape: an edge from a perturbation no node file holds.

    The known end (genotype G2) is walked ``GenotypeMemberOf -> ExperimentMemberOf``
    to ``FitnessBDataset``, so the report names the dataset whose records produced
    the row, not just the label.
    """
    _edges(
        clean_dir,
        "PerturbationMemberOf",
        [(P1, G1), (P2, G2), (P_MISSING, G2), (P_MISSING, G1)],
    )
    report = gate.check_endpoints(clean_dir)
    assert not report.ok
    assert report.n_dangling == 2
    (gap,) = report.gaps
    assert (gap.edge_label, gap.side, gap.n_rows) == (
        "PerturbationMemberOf",
        ":START_ID",
        2,
    )
    assert gap.sample_ids == [P_MISSING, P_MISSING]
    assert gap.anchor_labels == ["Genotype"]
    assert gap.datasets == {DS_A: 1, DS_B: 1}
    assert gap.n_unresolved == 0
    text = gate.format_report(report)
    assert "DANGLING: 2 edge rows name no node" in text
    assert (
        "PerturbationMemberOf :START_ID: 2 rows; other end ['Genotype']; datasets: "
        f"{DS_A}=1; {DS_B}=1;"
    ) in text


def test_a_dangling_end_with_no_ladder_is_reported_unresolved(clean_dir: Path) -> None:
    """An ExperimentMemberOf whose dataset node is missing: the anchor IS the experiment
    and the only edge toward a Dataset is the dangling one, so no dataset is derivable.
    """
    _edges(clean_dir, "ExperimentMemberOf", [(E1, DS_A), (E2, "FitnessGoneDataset")])
    report = gate.check_endpoints(clean_dir)
    (gap,) = report.gaps
    assert (gap.edge_label, gap.side, gap.n_rows) == (
        "ExperimentMemberOf",
        ":END_ID",
        1,
    )
    assert gap.sample_ids == ["FitnessGoneDataset"]
    assert gap.anchor_labels == ["Experiment"]
    assert (gap.datasets, gap.n_unresolved) == ({}, 1)
    assert "datasets: unresolved" in gate.format_report(report)


def test_without_resolution_the_gap_is_still_found_and_nothing_is_walked(
    clean_dir: Path,
) -> None:
    _edges(clean_dir, "PerturbationMemberOf", [(P1, G1), (P_MISSING, G2)])
    report = gate.check_endpoints(clean_dir, resolve_datasets=False)
    (gap,) = report.gaps
    assert (gap.n_rows, gap.anchor_labels, gap.datasets) == (1, [], {})
    assert not report.datasets_resolved
    assert (
        "PerturbationMemberOf :START_ID: 1 rows; datasets not resolved "
        "(--no-resolve-datasets)"
    ) in gate.format_report(report)


def test_both_sides_dangling_counts_each_side(clean_dir: Path) -> None:
    _edges(clean_dir, "Mentions", [("'12345'", E1), ("'99999'", _h("no experiment"))])
    report = gate.check_endpoints(clean_dir)
    assert [(g.edge_label, g.side, g.n_rows) for g in report.gaps] == [
        ("Mentions", ":START_ID", 1),
        ("Mentions", ":END_ID", 1),
    ]
    start_gap, end_gap = report.gaps
    assert start_gap.sample_ids == ["99999"]
    # the other end of the dangling start is also no node: nothing to anchor on
    assert (start_gap.anchor_labels, start_gap.n_unresolved) == ([], 1)
    assert end_gap.anchor_labels == []


def test_borrowed_headers_describe_exactly_the_part_files(
    clean_dir: Path, tmp_path: Path
) -> None:
    """A generation killed before BioCypher wrote its headers is checked with headers
    from elsewhere, and the two sets must match file for file.
    """
    parts_only = tmp_path / "partial"
    headers = tmp_path / "headers"
    parts_only.mkdir()
    headers.mkdir()
    for path in clean_dir.iterdir():
        if path.name.endswith("-header.csv"):
            (headers / path.name).write_bytes(path.read_bytes())
        elif path.name.endswith(".csv"):
            (parts_only / path.name).write_bytes(path.read_bytes())
    assert gate.check_endpoints(parts_only, headers).ok
    (headers / "Extra-header.csv").write_text(":ID\tid\t:LABEL\n")
    with pytest.raises(ValueError, match="headers without a part file \\['Extra'\\]"):
        gate.discover_groups(parts_only, headers)


def test_a_node_row_whose_id_is_not_the_first_column_is_parsed_as_csv(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "Thing-header.csv", [["name", ":ID", ":LABEL"]])
    _write(
        tmp_path / "Thing-part000.csv",
        [["'a\tquoted\ttab'", _h("t1"), "'Thing'"], ["'plain'", _h("t2"), "'Thing'"]],
    )
    _edges(tmp_path, "ThingMemberOf", [(_h("t1"), _h("t2"))])
    assert gate.check_endpoints(tmp_path).ok


def test_an_edge_row_with_the_wrong_field_count_is_refused(clean_dir: Path) -> None:
    (clean_dir / "GenotypeMemberOf-part000.csv").write_text(f"{G1}\t\t{E1}\n")
    with pytest.raises(ValueError, match="edge row has 3 fields, header 4"):
        gate.check_endpoints(clean_dir)


def test_the_cli_exits_non_zero_on_a_gap_and_writes_the_report(
    clean_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_path = tmp_path / "gate.json"
    assert gate.main([str(clean_dir), "--report", str(report_path)]) == 0
    assert json.loads(report_path.read_text())["gaps"] == []
    _edges(clean_dir, "PerturbationMemberOf", [(P_MISSING, G1)])
    assert gate.main([str(clean_dir), "--report", str(report_path)]) == 1
    out = capsys.readouterr().out
    assert "DANGLING: 1 edge rows name no node" in out
    written = json.loads(report_path.read_text())
    assert written["gaps"][0]["datasets"] == {DS_A: 1}


def test_a_per_adapter_output_is_checked_across_its_directories(tmp_path: Path) -> None:
    """An array run (``generation_array``) writes ``adapters/<dir>/`` per adapter; an edge
    in one directory may end at a node written by another, so the node table spans
    them all, a label's parts are concatenated under the first directory's header, and
    a ``.incomplete.<ts>`` attempt moved aside by a re-run is not read.
    """
    a = tmp_path / "adapters" / "000_AAdapter"
    b = tmp_path / "adapters" / "001_BAdapter"
    aside = tmp_path / "adapters" / "001_BAdapter.incomplete.2026-10-15_00-00-00"
    for d in (a, b, aside):
        d.mkdir(parents=True)
    _nodes(a, "Dataset", [DS_A])
    _nodes(a, "Experiment", [E1])
    _nodes(a, "Genotype", [G1])
    _edges(a, "GenotypeMemberOf", [(G1, E1)])
    _edges(a, "ExperimentMemberOf", [(E1, DS_A)])
    _nodes(b, "Dataset", [DS_B])
    _nodes(b, "Experiment", [E2])
    _edges(b, "GenotypeMemberOf", [(G1, E2)])  # G1 is A's node: fine across directories
    _edges(b, "ExperimentMemberOf", [(E2, DS_B)])
    _nodes(aside, "Experiment", ["deadbeef"])
    _edges(aside, "ExperimentMemberOf", [("deadbeef", "NoDataset")])
    groups = {g.label: g for g in gate.discover_groups(tmp_path)}
    assert [Path(p).parent.name for p in groups["Experiment"].part_paths] == [
        "000_AAdapter",
        "001_BAdapter",
    ]
    assert Path(groups["Experiment"].header_path).parent.name == "000_AAdapter"
    assert "Genotype" in groups and len(groups["Genotype"].part_paths) == 1
    report = gate.check_endpoints(tmp_path)
    assert report.ok
    assert report.node_rows == {"Dataset": 2, "Experiment": 2, "Genotype": 1}
    assert report.edge_rows == {"ExperimentMemberOf": 2, "GenotypeMemberOf": 2}
    # a dangling end in B is found against the global table
    _edges(b, "Mentions", [("'1'", _h("nowhere"))])
    _nodes(b, "Publication", ["'1'"])
    assert not gate.check_endpoints(tmp_path, resolve_datasets=False).ok
