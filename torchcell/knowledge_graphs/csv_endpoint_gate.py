# torchcell/knowledge_graphs/csv_endpoint_gate.py
# [[torchcell.knowledge_graphs.csv_endpoint_gate]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/knowledge_graphs/csv_endpoint_gate
# Test file: tests/torchcell/knowledge_graphs/test_csv_endpoint_gate.py
r"""The CSV endpoint gate: every edge endpoint names a node, checked before the import.

``neo4j-admin database import full`` runs with ``skip_bad_relationships: true`` in every
config under ``biocypher/config/`` (the full build's config, loaded through
``BIOCYPHER_CONFIG_PATH``), so a relationship row whose ``:START_ID`` or ``:END_ID`` is
not the ``:ID`` of any node row is dropped at import and reported, if at all, in an
``import.report`` the validate stage prints twenty lines of and continues past. That is
how the Carruthers ``GeneAdditionPerturbation`` records lost their ``perturbation member
of`` rows: the leaf was built under a conf whose only genotype lane is ``bacterial
perturbation``, the edge method walked it anyway, and the import discarded the row.

This module reads the BioCypher output directory the generator wrote, loads the ``:ID``
column of every node CSV into one id table, streams every edge CSV's ``:START_ID`` and
``:END_ID`` columns against it, and reports every endpoint with no node, grouped by edge
label and side, with the datasets the dangling rows belong to where the graph lets that
be derived (the known endpoint is walked along the ``member of`` ladder to a ``Dataset``
node, the ladder itself read off the CSVs). The live rebuild runs it as a hard stop
right before ``neo4j-admin import`` (``gilahyper_live_rebuild-slurm_docker.slurm``), so a
dangling row stops the job with the dataset and label named instead of an opaque
import report twenty hours in. Flipping ``skip_bad_relationships`` to ``false`` alone
would only move that failure into the import (review 3 finding 8, 2026-10-15); it stays
as it is for now, and this gate is the mandatory check.

Ids are stored packed (a 64-hex content id as its 32 raw bytes, any other id as its
bytes), so the id table of a full build (about 100 M node rows on KG 3.0) is held in
memory once. Rows are read as bytes and split on the tab delimiter: BioCypher writes no
quoted delimiter and no multiline field (the import call passes no
``--multiline-fields``), and a node id is the row's first field.

CLI::

    python -m torchcell.knowledge_graphs.csv_endpoint_gate <biocypher-out dir> \\
        [--header-dir DIR] [--no-resolve-datasets] [--report PATH]

``--header-dir`` names a directory holding the ``<Label>-header.csv`` files when the
output directory has none: BioCypher writes the headers after the last adapter, so a
generation killed mid-way (job 4901, 2026-10-14) leaves part files without them.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import Counter
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import BinaryIO

from pydantic import BaseModel, ConfigDict, Field

from torchcell.knowledge_graphs.incremental_import import CsvGroup, discover_csv_groups

__all__ = [
    "EndpointGap",
    "EndpointReport",
    "DATASET_LABEL",
    "pack_id",
    "discover_groups",
    "check_endpoints",
    "format_report",
    "main",
]

DATASET_LABEL = "Dataset"
"""The node label the dataset-resolution ladder ends at (BioCypher's file name for it)."""
_DELIMITER = b"\t"
_QUOTE = b"'"
_HEADER_RE = re.compile(r"^(?P<label>.+)-header\.csv$")
_PART_RE = re.compile(r"^(?P<label>.+)-part\d+\.csv$")
_LADDER_SAMPLE_ROWS = 1000
"""Rows sampled per edge group to learn which node labels its endpoints carry."""
_MAX_HOPS = 4
"""The longest ``member of`` ladder to a ``Dataset`` node (crispr construct ->
perturbation -> genotype -> experiment -> dataset is four edges)."""


class EndpointGap(BaseModel):
    """Edge rows of one label whose ``side`` endpoint is the id of no node."""

    model_config = ConfigDict(frozen=True)

    edge_label: str
    side: str  # ":START_ID" | ":END_ID"
    n_rows: int
    sample_ids: list[str] = Field(default_factory=list)
    anchor_labels: list[str] = Field(
        default_factory=list
    )  # node labels of the other end
    # dataset node id -> dangling rows whose other end reaches it; empty when the ladder
    # could not be walked (no Dataset node rows, or no path from the anchor label)
    datasets: dict[str, int] = Field(default_factory=dict)
    n_unresolved: int = 0  # dangling rows whose other end reached no Dataset node


class EndpointReport(BaseModel):
    """What the gate found in one BioCypher output directory."""

    model_config = ConfigDict(frozen=True)

    out_dir: str
    node_rows: dict[str, int]  # node label -> rows
    edge_rows: dict[str, int]  # edge label -> rows
    n_node_ids: int  # distinct node ids
    gaps: list[EndpointGap]
    elapsed_s: float
    datasets_resolved: bool = True  # False under --no-resolve-datasets

    @property
    def ok(self) -> bool:
        """True when every edge endpoint is the id of a node."""
        return not self.gaps

    @property
    def n_dangling(self) -> int:
        """Edge rows with at least one endpoint that is no node, summed over the gaps."""
        return sum(gap.n_rows for gap in self.gaps)


def pack_id(raw: bytes) -> bytes:
    """A node id as the bytes the id table keys on: 64 hex chars pack to 32 bytes.

    Content ids are sha256 hex; a ``Dataset`` id is its class name and a publication
    id its PubMed id, and those stay as their bytes. The quotes BioCypher may put
    around a string field are stripped first.
    """
    if len(raw) >= 2 and raw[0:1] == _QUOTE and raw[-1:] == _QUOTE:
        raw = raw[1:-1]
    if len(raw) == 64:
        try:
            return bytes.fromhex(raw.decode("ascii"))
        except ValueError:
            return raw
    return raw


def _unpack_id(packed: bytes) -> str:
    """The id as the CSV spelled it, for a report line."""
    if len(packed) == 32:
        return packed.hex()
    return packed.decode("utf-8")


def _read_header(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").rstrip("\n").split("\t")


def discover_groups(out_dir: Path, header_dir: Path | None = None) -> list[CsvGroup]:
    """The directory's label groups; headers from ``header_dir`` when it has none itself.

    Without ``header_dir`` this is :func:`discover_csv_groups`. With it, every part file
    of ``out_dir`` must have its ``<Label>-header.csv`` in ``header_dir``, and a header
    there with no part file here is an error, so the borrowed headers describe exactly
    the files being checked.
    """
    if header_dir is None:
        return discover_csv_groups(out_dir)
    parts: dict[str, list[Path]] = {}
    for path in sorted(out_dir.iterdir()):
        if (m := _PART_RE.match(path.name)) is not None:
            parts.setdefault(m.group("label"), []).append(path)
    headers: dict[str, Path] = {}
    for path in sorted(header_dir.iterdir()):
        if (m := _HEADER_RE.match(path.name)) is not None:
            headers[m.group("label")] = path
    if set(parts) != set(headers):
        raise ValueError(
            f"part files and borrowed headers differ: parts without a header "
            f"{sorted(set(parts) - set(headers))}, headers without a part file "
            f"{sorted(set(headers) - set(parts))}"
        )
    groups: list[CsvGroup] = []
    for label, header_path in sorted(headers.items()):
        columns = _read_header(header_path)
        if ":ID" in columns or any(c.endswith(":ID") for c in columns):
            kind = "node"
        elif ":START_ID" in columns:
            kind = "edge"
        else:
            raise ValueError(f"header is neither node nor edge: {header_path}")
        groups.append(
            CsvGroup(
                label=label,
                kind=kind,
                header_path=str(header_path),
                part_paths=[str(p) for p in parts[label]],
                columns=columns,
            )
        )
    return groups


def _id_column(columns: list[str]) -> int:
    for index, column in enumerate(columns):
        if column == ":ID" or column.endswith(":ID"):
            return index
    raise ValueError(f"no :ID column in {columns}")


def _node_ids_of_file(handle: BinaryIO, id_index: int, n_columns: int) -> list[bytes]:
    """The packed ids of one node part file.

    The first field is read off each line directly when the id is the first column
    (BioCypher's layout); otherwise the row is parsed with the csv module, which
    handles the quoted multi-MB ``serialized_data`` blobs a node row carries.
    """
    ids: list[bytes] = []
    if id_index == 0:
        for line in handle:
            tab = line.find(_DELIMITER)
            ids.append(pack_id(line[:tab] if tab >= 0 else line.rstrip(b"\n")))
        return ids
    csv.field_size_limit(sys.maxsize)
    import io

    reader = csv.reader(
        io.TextIOWrapper(handle, encoding="utf-8", newline=""),
        delimiter="\t",
        quotechar="'",
        strict=True,
    )
    for row in reader:
        if len(row) != n_columns:
            raise ValueError(f"node row has {len(row)} fields, header {n_columns}")
        ids.append(pack_id(row[id_index].encode("utf-8")))
    return ids


def _load_node_table(groups: list[CsvGroup]) -> tuple[dict[bytes, int], dict[str, int]]:
    """``packed id -> label index`` over every node group, and rows per label."""
    table: dict[bytes, int] = {}
    rows: dict[str, int] = {}
    for label_index, group in enumerate(g for g in groups if g.kind == "node"):
        id_index = _id_column(group.columns)
        n_rows = 0
        for part in group.part_paths:
            with open(part, "rb") as handle:
                ids = _node_ids_of_file(handle, id_index, len(group.columns))
            n_rows += len(ids)
            for packed in ids:
                table[packed] = label_index
        rows[group.label] = n_rows
    return table, rows


def _iter_edge_rows(group: CsvGroup) -> Iterator[tuple[bytes, bytes]]:
    """Stream ``(packed start, packed end)`` over every row of one edge group."""
    start_index = group.columns.index(":START_ID")
    end_index = group.columns.index(":END_ID")
    n_columns = len(group.columns)
    for part in group.part_paths:
        with open(part, "rb") as handle:
            for line in handle:
                fields = line.rstrip(b"\n").split(_DELIMITER)
                if len(fields) != n_columns:
                    raise ValueError(
                        f"{part}: edge row has {len(fields)} fields, header {n_columns}"
                    )
                yield pack_id(fields[start_index]), pack_id(fields[end_index])


class _EdgeScan:
    """One edge group's dangling endpoints and their known other ends."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.n_rows = 0
        self.dangling: dict[str, int] = {":START_ID": 0, ":END_ID": 0}
        self.samples: dict[str, list[bytes]] = {":START_ID": [], ":END_ID": []}
        # side -> the OTHER endpoint of each dangling row (its anchor), for resolution
        self.anchors: dict[str, list[bytes]] = {":START_ID": [], ":END_ID": []}


def _scan_edges(
    groups: list[CsvGroup], table: dict[bytes, int], sample: int, collect_anchors: bool
) -> list[_EdgeScan]:
    """Every edge group's dangling endpoints; anchors kept only when they will be walked."""
    scans: list[_EdgeScan] = []
    for group in groups:
        if group.kind != "edge":
            continue
        scan = _EdgeScan(group.label)
        for start, end in _iter_edge_rows(group):
            scan.n_rows += 1
            if start not in table:
                scan.dangling[":START_ID"] += 1
                if len(scan.samples[":START_ID"]) < sample:
                    scan.samples[":START_ID"].append(start)
                if collect_anchors:
                    scan.anchors[":START_ID"].append(end)
            if end not in table:
                scan.dangling[":END_ID"] += 1
                if len(scan.samples[":END_ID"]) < sample:
                    scan.samples[":END_ID"].append(end)
                if collect_anchors:
                    scan.anchors[":END_ID"].append(start)
        scans.append(scan)
    return scans


def _ladder(
    groups: list[CsvGroup], table: dict[bytes, int], labels: list[str]
) -> dict[str, list[tuple[str, str]]]:
    """``edge label -> [(start node label, end node label), ...]`` seen in a row sample.

    The yaml declares each edge's sources and targets, but its names are not the file
    labels (``publication mentions experiment`` is the ``Mentions`` file), so the ladder
    is read off the rows: the node table names the label of every known endpoint.
    """
    ladder: dict[str, list[tuple[str, str]]] = {}
    for group in groups:
        if group.kind != "edge":
            continue
        pairs: set[tuple[str, str]] = set()
        for start, end in islice(_iter_edge_rows(group), _LADDER_SAMPLE_ROWS):
            if start in table and end in table:
                pairs.add((labels[table[start]], labels[table[end]]))
        ladder[group.label] = sorted(pairs)
    return ladder


def _paths_to_dataset(
    ladder: dict[str, list[tuple[str, str]]], anchor_label: str
) -> list[str] | None:
    """The edge labels, in order, from ``anchor_label`` to ``Dataset``; None if none."""
    frontier: list[tuple[str, list[str]]] = [(anchor_label, [])]
    seen = {anchor_label}
    for _hop in range(_MAX_HOPS):
        next_frontier: list[tuple[str, list[str]]] = []
        for label, path in frontier:
            for edge_label, pairs in ladder.items():
                for start_label, end_label in pairs:
                    if start_label != label or end_label in seen:
                        continue
                    if end_label == DATASET_LABEL:
                        return [*path, edge_label]
                    seen.add(end_label)
                    next_frontier.append((end_label, [*path, edge_label]))
        frontier = next_frontier
    return None


def _walk(
    groups: list[CsvGroup], path: list[str], anchors: list[bytes]
) -> dict[bytes, set[bytes]]:
    """Follow ``path`` (edge labels) from each anchor id; ``anchor -> ids reached``."""
    by_label = {g.label: g for g in groups}
    reached: dict[bytes, set[bytes]] = {a: {a} for a in anchors}
    for edge_label in path:
        frontier: dict[bytes, set[bytes]] = {}
        for anchor, ids in reached.items():
            for node in ids:
                frontier.setdefault(node, set()).add(anchor)
        next_reached: dict[bytes, set[bytes]] = {a: set() for a in anchors}
        for start, end in _iter_edge_rows(by_label[edge_label]):
            if start in frontier:
                for anchor in frontier[start]:
                    next_reached[anchor].add(end)
        reached = next_reached
    return reached


def check_endpoints(
    out_dir: Path,
    header_dir: Path | None = None,
    *,
    resolve_datasets: bool = True,
    sample: int = 5,
) -> EndpointReport:
    """Read every node id, stream every edge endpoint against them, report the gaps.

    With ``resolve_datasets`` the known end of each dangling row is walked along the
    ``member of`` ladder to a ``Dataset`` node (one pass over each edge file on the
    path, only when a gap exists), so the report names the dataset whose records
    produced the row.
    """
    t0 = time.monotonic()
    groups = discover_groups(out_dir, header_dir)
    node_labels = [g.label for g in groups if g.kind == "node"]
    table, node_rows = _load_node_table(groups)
    scans = _scan_edges(groups, table, sample, collect_anchors=resolve_datasets)
    edge_rows = {scan.label: scan.n_rows for scan in scans}
    ladder = (
        _ladder(groups, table, node_labels)
        if resolve_datasets and any(any(scan.dangling.values()) for scan in scans)
        else {}
    )
    gaps: list[EndpointGap] = []
    for scan in scans:
        for side in (":START_ID", ":END_ID"):
            if scan.dangling[side] == 0:
                continue
            anchors = scan.anchors[side]
            anchor_labels = sorted(
                {node_labels[table[a]] for a in anchors if a in table}
            )
            datasets: Counter[str] = Counter()
            n_unresolved = len(anchors)
            if resolve_datasets and DATASET_LABEL in node_labels and anchor_labels:
                n_unresolved = 0
                for anchor_label in anchor_labels:
                    path = _paths_to_dataset(ladder, anchor_label)
                    labeled = [
                        a
                        for a in anchors
                        if a in table and node_labels[table[a]] == anchor_label
                    ]
                    if path is None:
                        n_unresolved += len(labeled)
                        continue
                    for _anchor, reached in _walk(groups, path, labeled).items():
                        names = sorted(_unpack_id(r) for r in reached if r in table)
                        if not names:
                            n_unresolved += 1
                        for name in names:
                            datasets[name] += 1
                n_unresolved += sum(1 for a in anchors if a not in table)
            gaps.append(
                EndpointGap(
                    edge_label=scan.label,
                    side=side,
                    n_rows=scan.dangling[side],
                    sample_ids=[_unpack_id(s) for s in scan.samples[side]],
                    anchor_labels=anchor_labels,
                    datasets=dict(sorted(datasets.items())),
                    n_unresolved=n_unresolved,
                )
            )
    return EndpointReport(
        out_dir=str(out_dir),
        node_rows=node_rows,
        edge_rows=edge_rows,
        n_node_ids=len(table),
        gaps=gaps,
        elapsed_s=time.monotonic() - t0,
        datasets_resolved=resolve_datasets,
    )


def format_report(report: EndpointReport) -> str:
    """The gate's text: one summary line, then one line per gap."""
    lines = [
        f"CSV endpoint gate: {report.out_dir}",
        f"  nodes: {sum(report.node_rows.values())} rows in {len(report.node_rows)} "
        f"labels ({report.n_node_ids} distinct ids); edges: "
        f"{sum(report.edge_rows.values())} rows in {len(report.edge_rows)} labels; "
        f"{report.elapsed_s:.0f} s",
    ]
    if report.ok:
        lines.append("  OK: every edge endpoint is the id of a node")
        return "\n".join(lines)
    lines.append(f"  DANGLING: {report.n_dangling} edge rows name no node")
    for gap in report.gaps:
        if not report.datasets_resolved:
            lines.append(
                f"  {gap.edge_label} {gap.side}: {gap.n_rows} rows; datasets not "
                f"resolved (--no-resolve-datasets); sample {gap.sample_ids}"
            )
            continue
        where = (
            "; ".join(f"{name}={n}" for name, n in gap.datasets.items())
            if gap.datasets
            else "unresolved"
        )
        unresolved = (
            f", {gap.n_unresolved} unresolved"
            if gap.datasets and gap.n_unresolved
            else ""
        )
        lines.append(
            f"  {gap.edge_label} {gap.side}: {gap.n_rows} rows; other end "
            f"{gap.anchor_labels or ['(no node either)']}; datasets: {where}{unresolved}; "
            f"sample {gap.sample_ids}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI: print the report, write it as JSON when asked, exit 1 on any gap."""
    parser = argparse.ArgumentParser(
        prog="python -m torchcell.knowledge_graphs.csv_endpoint_gate",
        description="Refuse a BioCypher output directory with an edge endpoint that is "
        "the id of no node, before neo4j-admin import drops the row.",
    )
    parser.add_argument("out_dir", help="the biocypher-out/<timestamp> directory")
    parser.add_argument(
        "--header-dir",
        default=None,
        help="directory holding <Label>-header.csv when out_dir has none (a generation "
        "killed before BioCypher wrote its headers)",
    )
    parser.add_argument(
        "--no-resolve-datasets",
        action="store_true",
        help="report the gaps without walking the member-of ladder to name datasets",
    )
    parser.add_argument("--report", default=None, help="write the report as JSON here")
    args = parser.parse_args(argv)
    report = check_endpoints(
        Path(args.out_dir),
        Path(args.header_dir) if args.header_dir else None,
        resolve_datasets=not args.no_resolve_datasets,
    )
    print(format_report(report))
    if args.report:
        Path(args.report).write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
