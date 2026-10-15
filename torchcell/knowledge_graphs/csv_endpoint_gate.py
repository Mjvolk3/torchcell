# torchcell/knowledge_graphs/csv_endpoint_gate.py
# [[torchcell.knowledge_graphs.csv_endpoint_gate]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/knowledge_graphs/csv_endpoint_gate
# Test file: tests/torchcell/knowledge_graphs/test_csv_endpoint_gate.py
r"""The CSV endpoint gate: every edge endpoint names a node of a declared label.

``neo4j-admin database import full`` runs with ``skip_bad_relationships: true`` in every
config under ``biocypher/config/`` (the full build's config, loaded through
``BIOCYPHER_CONFIG_PATH``), so a relationship row whose ``:START_ID`` or ``:END_ID`` is
not the ``:ID`` of any node row is dropped at import and reported, if at all, in an
``import.report`` the validate stage prints twenty lines of and continues past. That is
how the Carruthers ``GeneAdditionPerturbation`` records lost their ``perturbation member
of`` rows: the leaf was built under a conf whose only genotype lane is ``bacterial
perturbation``, the edge method walked it anyway, and the import discarded the row.

An endpoint that names SOME node is not enough either. ``neo4j-admin import`` has one
id space, so an edge row whose endpoint id was written under another label is imported
and lands on that other node. That is how 218,258 whole genotypes and 135 Hale 2024
backgrounds of the job 4901 partial, and 14,573 Bloom 2019 and Caudal 2024 genotypes of
the served KG 3.0 store, came to exist only as ``InternedConstant`` nodes: the interned
id was the bare sha256 of the same bytes the genotype node hashes, the sink
deduplicates ids across labels, and ``GenotypeMemberOf`` rows pointed at a constant
(``torchcell.datamodels.interned_constant``, 2026-10-15). An existence check passes
that; a label check does not.

This module reads the BioCypher output directory the generator wrote, loads the ``:ID``
and ``:LABEL`` columns of every node CSV into one id table (id to its ancestry string),
reads the schema yaml's ``source`` and ``target`` of every edge class, streams every
edge CSV's ``:START_ID`` and ``:END_ID`` columns against both, and reports

* every endpoint with no node (a dangling row),
* every endpoint whose node carries NONE of the labels the edge class declares for
  that side (a mislabeled row), where a node's labels are its whole BioCypher ancestry,
  as Neo4j gives it (``PerturbedSegregantGenotype|Genotype|...`` satisfies ``target:
  genotype`` exactly as ``MATCH (g:Genotype)`` matches that node in the store),
* every node id written under two different labels (an id collision), with or without
  an edge on it, since the import keeps one of the two rows and never says which,

grouped by edge label and side, with the datasets the rows belong to where the graph
lets that be derived (the known endpoint is walked along the ``member of`` ladder to a
``Dataset`` node, the ladder itself read off the CSVs). An edge file whose label no
schema class declares (``Mentions``, a Biolink predicate the adapter emits itself) is
listed as undeclared and its labels are not checked. The live rebuild runs the gate as
a hard stop right before ``neo4j-admin import``
(``gilahyper_live_rebuild-slurm_docker.slurm``), so a bad row stops the job with the
dataset and label named instead of an opaque import report twenty hours in. Flipping
``skip_bad_relationships`` to ``false`` alone would only move that failure into the
import (review 3 finding 8, 2026-10-15); it stays as it is for now, and this gate is
the mandatory check.

Ids are stored packed (a 64-hex content id as its 32 raw bytes, any other id as its
bytes) against a small integer, the index of the row's ancestry string among the
distinct ones seen (about a hundred in a build), so the id table of a full build
(about 100 M node rows on KG 3.0) is held in memory once and the label check is one
list index per endpoint. Rows are read as bytes and split on the tab delimiter:
BioCypher writes no quoted delimiter and no multiline field (the import call passes no
``--multiline-fields``), a node id is the row's first field and its ancestry the last.

CLI::

    python -m torchcell.knowledge_graphs.csv_endpoint_gate <biocypher-out dir> \\
        [--header-dir DIR] [--schema YAML] [--no-resolve-datasets] [--report PATH]

``--header-dir`` names a directory holding the ``<Label>-header.csv`` files when the
output directory has none: BioCypher writes the headers after the last adapter, so a
generation killed mid-way (job 4901, 2026-10-14) leaves part files without them.
``--schema`` is the schema config yaml; the default is the repository's
``biocypher/config/torchcell_schema_config.yaml`` next to the imported package.
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

import yaml
from biocypher._misc import sentencecase_to_pascalcase
from pydantic import BaseModel, ConfigDict, Field

import torchcell
from torchcell.knowledge_graphs.generation_array import ADAPTERS_DIRNAME
from torchcell.knowledge_graphs.incremental_import import CsvGroup, discover_csv_groups

__all__ = [
    "EdgeDeclaration",
    "EndpointGap",
    "EndpointReport",
    "IdCollision",
    "LabelGap",
    "DATASET_LABEL",
    "SCHEMA_CONFIG_RELPATH",
    "pack_id",
    "default_schema_path",
    "read_edge_declarations",
    "discover_groups",
    "check_endpoints",
    "format_report",
    "main",
]

DATASET_LABEL = "Dataset"
"""The node label the dataset-resolution ladder ends at (BioCypher's file name for it)."""
SCHEMA_CONFIG_RELPATH = "biocypher/config/torchcell_schema_config.yaml"
"""The schema config yaml, relative to the repository root."""
_DELIMITER = b"\t"
_QUOTE = b"'"
_ANCESTRY_SEPARATOR = "|"
_LABEL_COLUMN = ":LABEL"
_HEADER_RE = re.compile(r"^(?P<label>.+)-header\.csv$")
_PART_RE = re.compile(r"^(?P<label>.+)-part\d+\.csv$")
_LADDER_SAMPLE_ROWS = 1000
"""Rows sampled per edge group to learn which node labels its endpoints carry."""
_MAX_HOPS = 4
"""The longest ``member of`` ladder to a ``Dataset`` node (crispr construct ->
perturbation -> genotype -> experiment -> dataset is four edges)."""
_SIDES = (":START_ID", ":END_ID")


class EdgeDeclaration(BaseModel):
    """The node labels one edge class declares for its start and its end."""

    model_config = ConfigDict(frozen=True)

    edge_label: str  # PascalCase, as BioCypher names the edge file
    start_labels: list[str]  # PascalCase node labels declared as ``source``
    end_labels: list[str]  # PascalCase node labels declared as ``target``


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


class LabelGap(BaseModel):
    """Edge rows of one label whose ``side`` endpoint is a node of an undeclared label."""

    model_config = ConfigDict(frozen=True)

    edge_label: str
    side: str  # ":START_ID" | ":END_ID"
    node_label: str  # the leaf label the endpoint's node row carries
    n_rows: int
    allowed_labels: list[str]  # what the schema declares for this side
    sample_ids: list[str] = Field(default_factory=list)
    anchor_labels: list[str] = Field(default_factory=list)
    datasets: dict[str, int] = Field(default_factory=dict)
    n_unresolved: int = 0


class IdCollision(BaseModel):
    """Node ids written under two different labels (the import keeps one row)."""

    model_config = ConfigDict(frozen=True)

    first_label: str  # the label whose row was read first
    second_label: str
    n_ids: int
    sample_ids: list[str] = Field(default_factory=list)


class EndpointReport(BaseModel):
    """What the gate found in one BioCypher output directory."""

    model_config = ConfigDict(frozen=True)

    out_dir: str
    node_rows: dict[str, int]  # node label -> rows
    edge_rows: dict[str, int]  # edge label -> rows
    n_node_ids: int  # distinct node ids
    gaps: list[EndpointGap]
    label_gaps: list[LabelGap] = Field(default_factory=list)
    id_collisions: list[IdCollision] = Field(default_factory=list)
    undeclared_edges: list[str] = Field(
        default_factory=list
    )  # edge files no schema class declares; their labels are not checked
    schema_path: str | None = None
    elapsed_s: float
    datasets_resolved: bool = True  # False under --no-resolve-datasets

    @property
    def ok(self) -> bool:
        """True when every endpoint is a node of a declared label and no id collides."""
        return not (self.gaps or self.label_gaps or self.id_collisions)

    @property
    def n_dangling(self) -> int:
        """Edge rows with at least one endpoint that is no node, summed over the gaps."""
        return sum(gap.n_rows for gap in self.gaps)

    @property
    def n_mislabeled(self) -> int:
        """Edge rows with an endpoint of an undeclared label, summed over the label gaps."""
        return sum(gap.n_rows for gap in self.label_gaps)


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


def _unquote(raw: bytes) -> str:
    if len(raw) >= 2 and raw[0:1] == _QUOTE and raw[-1:] == _QUOTE:
        raw = raw[1:-1]
    return raw.decode("utf-8")


def _read_header(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").rstrip("\n").split("\t")


def default_schema_path() -> Path:
    """The repository's schema config yaml, located next to the imported package."""
    return Path(torchcell.__file__).resolve().parent.parent / SCHEMA_CONFIG_RELPATH


def read_edge_declarations(schema_path: Path) -> dict[str, EdgeDeclaration]:
    """``edge file label -> declaration`` for every ``represented_as: edge`` class.

    The file label is the PascalCase of the class name, as BioCypher names the file;
    the declared sources and targets are PascalCased the same way, so they compare
    with the ancestry strings the node rows carry.
    """
    raw = yaml.safe_load(schema_path.read_text(encoding="utf-8")) or {}
    declarations: dict[str, EdgeDeclaration] = {}
    for name, spec in raw.items():
        if not isinstance(spec, dict) or spec.get("represented_as") != "edge":
            continue
        source = spec.get("source", [])
        target = spec.get("target", [])
        sources = source if isinstance(source, list) else [source]
        targets = target if isinstance(target, list) else [target]
        label = sentencecase_to_pascalcase(name)
        declarations[label] = EdgeDeclaration(
            edge_label=label,
            start_labels=sorted(sentencecase_to_pascalcase(s) for s in sources),
            end_labels=sorted(sentencecase_to_pascalcase(t) for t in targets),
        )
    return declarations


def discover_groups(out_dir: Path, header_dir: Path | None = None) -> list[CsvGroup]:
    """The directory's label groups; headers from ``header_dir`` when it has none itself.

    Without ``header_dir`` this is :func:`discover_csv_groups`. With it, every part file
    of ``out_dir`` must have its ``<Label>-header.csv`` in ``header_dir``, and a header
    there with no part file here is an error, so the borrowed headers describe exactly
    the files being checked.
    """
    if header_dir is None:
        if (out_dir / ADAPTERS_DIRNAME).is_dir():
            return discover_array_groups(out_dir / ADAPTERS_DIRNAME)
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


def discover_array_groups(adapters_dir: Path) -> list[CsvGroup]:
    """The label groups of a per-adapter generation, every directory's parts together.

    An array run (``generation_array``) writes one directory per adapter under
    ``<out>/adapters/``; an edge in one directory may point at a node written by
    another, so the gate's node table and edge scan span all of them. Each directory is
    discovered as a flat BioCypher output; a label's parts are concatenated in directory
    order under the first directory's header (the merge checks the headers are
    byte-identical). A ``<dir>.incomplete.<ts>`` directory (an attempt moved aside by a
    re-run) is not part of the output and is skipped.
    """
    groups: dict[str, CsvGroup] = {}
    for directory in sorted(p for p in adapters_dir.iterdir() if p.is_dir()):
        if ".incomplete." in directory.name:
            continue
        for group in discover_csv_groups(directory):
            if group.label in groups:
                groups[group.label].part_paths.extend(group.part_paths)
            else:
                groups[group.label] = group
    return [groups[label] for label in sorted(groups)]


def _id_column(columns: list[str]) -> int:
    for index, column in enumerate(columns):
        if column == ":ID" or column.endswith(":ID"):
            return index
    raise ValueError(f"no :ID column in {columns}")


def _node_rows_of_file(
    handle: BinaryIO, id_index: int, label_index: int | None, n_columns: int
) -> Iterator[tuple[bytes, bytes]]:
    """``(packed id, ancestry)`` of every row of one node part file.

    The first and last fields are read off each line directly when the id is the first
    column and the ``:LABEL`` column the last (BioCypher's layout); otherwise the row
    is parsed with the csv module, which handles the quoted multi-MB
    ``serialized_data`` blobs a node row carries. A file with no ``:LABEL`` column
    yields an empty ancestry, and the caller uses the file's label.
    """
    if id_index == 0 and (label_index is None or label_index == n_columns - 1):
        for line in handle:
            line = line.rstrip(b"\n")
            tab = line.find(_DELIMITER)
            if tab < 0:
                yield pack_id(line), b""
                continue
            ancestry = (
                b"" if label_index is None else line[line.rfind(_DELIMITER) + 1 :]
            )
            yield pack_id(line[:tab]), ancestry
        return
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
        labels = "" if label_index is None else row[label_index]
        yield pack_id(row[id_index].encode("utf-8")), labels.encode("utf-8")


class _NodeTable:
    """Every node id, keyed packed, to the index of its ancestry string."""

    def __init__(self) -> None:
        self.table: dict[bytes, int] = {}
        self.ancestries: list[str] = []
        self.leaves: list[str] = []  # the first label of each ancestry
        self._index: dict[str, int] = {}
        self.rows: dict[str, int] = {}
        self.collisions: dict[tuple[str, str], list[bytes]] = {}
        self.n_collisions: Counter[tuple[str, str]] = Counter()

    def ancestry_index(self, ancestry: str) -> int:
        index = self._index.get(ancestry)
        if index is None:
            index = len(self.ancestries)
            self._index[ancestry] = index
            self.ancestries.append(ancestry)
            self.leaves.append(ancestry.split(_ANCESTRY_SEPARATOR, 1)[0])
        return index

    def add(self, packed: bytes, index: int, sample: int) -> None:
        held = self.table.get(packed)
        if held is None:
            self.table[packed] = index
            return
        if held == index:
            return
        key = (self.leaves[held], self.leaves[index])
        self.n_collisions[key] += 1
        samples = self.collisions.setdefault(key, [])
        if len(samples) < sample:
            samples.append(packed)


def _load_node_table(groups: list[CsvGroup], sample: int) -> _NodeTable:
    """The id table over every node group, with the rows per label and the collisions."""
    nodes = _NodeTable()
    for group in groups:
        if group.kind != "node":
            continue
        id_index = _id_column(group.columns)
        label_index = (
            group.columns.index(_LABEL_COLUMN)
            if _LABEL_COLUMN in group.columns
            else None
        )
        file_index = nodes.ancestry_index(group.label)
        n_rows = 0
        for part in group.part_paths:
            with open(part, "rb") as handle:
                for packed, ancestry in _node_rows_of_file(
                    handle, id_index, label_index, len(group.columns)
                ):
                    n_rows += 1
                    index = (
                        file_index
                        if not ancestry
                        else nodes.ancestry_index(_unquote(ancestry))
                    )
                    nodes.add(packed, index, sample)
        nodes.rows[group.label] = nodes.rows.get(group.label, 0) + n_rows
    return nodes


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
    """One edge group's dangling and mislabeled endpoints and their known other ends."""

    def __init__(self, label: str, declaration: EdgeDeclaration | None) -> None:
        self.label = label
        self.declaration = declaration
        self.n_rows = 0
        self.dangling: dict[str, int] = {side: 0 for side in _SIDES}
        self.samples: dict[str, list[bytes]] = {side: [] for side in _SIDES}
        # side -> the OTHER endpoint of each dangling row (its anchor), for resolution
        self.anchors: dict[str, list[bytes]] = {side: [] for side in _SIDES}
        # side -> leaf label of the endpoint's node -> rows, samples and anchors
        self.mislabeled: dict[str, Counter[str]] = {side: Counter() for side in _SIDES}
        self.mis_samples: dict[str, dict[str, list[bytes]]] = {
            side: {} for side in _SIDES
        }
        self.mis_anchors: dict[str, dict[str, list[bytes]]] = {
            side: {} for side in _SIDES
        }

    def allowed_masks(self, nodes: _NodeTable) -> tuple[list[bool], list[bool]] | None:
        """Per ancestry index, whether the node satisfies the declared start / end."""
        if self.declaration is None:
            return None
        masks: list[list[bool]] = []
        for labels in (self.declaration.start_labels, self.declaration.end_labels):
            allowed = set(labels)
            masks.append(
                [
                    any(part in allowed for part in a.split(_ANCESTRY_SEPARATOR))
                    for a in nodes.ancestries
                ]
            )
        return masks[0], masks[1]

    def note_mislabeled(
        self,
        side: str,
        leaf: str,
        endpoint: bytes,
        anchor: bytes,
        sample: int,
        keep: bool,
    ) -> None:
        self.mislabeled[side][leaf] += 1
        samples = self.mis_samples[side].setdefault(leaf, [])
        if len(samples) < sample:
            samples.append(endpoint)
        if keep:
            self.mis_anchors[side].setdefault(leaf, []).append(anchor)


def _scan_edges(
    groups: list[CsvGroup],
    nodes: _NodeTable,
    declarations: dict[str, EdgeDeclaration],
    sample: int,
    collect_anchors: bool,
) -> list[_EdgeScan]:
    """Every edge group's dangling and mislabeled endpoints; anchors kept when walked."""
    table = nodes.table
    leaves = nodes.leaves
    scans: list[_EdgeScan] = []
    for group in groups:
        if group.kind != "edge":
            continue
        scan = _EdgeScan(group.label, declarations.get(group.label))
        masks = scan.allowed_masks(nodes)
        start_ok, end_ok = masks if masks is not None else (None, None)
        for start, end in _iter_edge_rows(group):
            scan.n_rows += 1
            start_index = table.get(start)
            end_index = table.get(end)
            if start_index is None:
                scan.dangling[":START_ID"] += 1
                if len(scan.samples[":START_ID"]) < sample:
                    scan.samples[":START_ID"].append(start)
                if collect_anchors:
                    scan.anchors[":START_ID"].append(end)
            elif start_ok is not None and not start_ok[start_index]:
                scan.note_mislabeled(
                    ":START_ID",
                    leaves[start_index],
                    start,
                    end,
                    sample,
                    collect_anchors,
                )
            if end_index is None:
                scan.dangling[":END_ID"] += 1
                if len(scan.samples[":END_ID"]) < sample:
                    scan.samples[":END_ID"].append(end)
                if collect_anchors:
                    scan.anchors[":END_ID"].append(start)
            elif end_ok is not None and not end_ok[end_index]:
                scan.note_mislabeled(
                    ":END_ID", leaves[end_index], end, start, sample, collect_anchors
                )
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


def _resolve(
    groups: list[CsvGroup],
    nodes: _NodeTable,
    ladder: dict[str, list[tuple[str, str]]],
    anchors: list[bytes],
    resolve_datasets: bool,
) -> tuple[list[str], dict[str, int], int]:
    """``(anchor labels, dataset -> rows, unresolved rows)`` for one set of anchors."""
    table, leaves = nodes.table, nodes.leaves
    anchor_labels = sorted({leaves[table[a]] for a in anchors if a in table})
    datasets: Counter[str] = Counter()
    n_unresolved = len(anchors)
    if resolve_datasets and DATASET_LABEL in leaves and anchor_labels:
        n_unresolved = 0
        for anchor_label in anchor_labels:
            path = _paths_to_dataset(ladder, anchor_label)
            labeled = [
                a for a in anchors if a in table and leaves[table[a]] == anchor_label
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
    return anchor_labels, dict(sorted(datasets.items())), n_unresolved


def check_endpoints(
    out_dir: Path,
    header_dir: Path | None = None,
    *,
    schema_path: Path | None = None,
    resolve_datasets: bool = True,
    sample: int = 5,
) -> EndpointReport:
    """Read every node id and label, stream every edge endpoint against them, report.

    ``schema_path`` is the schema config yaml whose edge classes declare the allowed
    labels (the repository's by default). With ``resolve_datasets`` the known end of
    each bad row is walked along the ``member of`` ladder to a ``Dataset`` node (one
    pass over each edge file on the path, only when a gap exists), so the report names
    the dataset whose records produced the row.
    """
    t0 = time.monotonic()
    schema = default_schema_path() if schema_path is None else schema_path
    declarations = read_edge_declarations(schema)
    groups = discover_groups(out_dir, header_dir)
    nodes = _load_node_table(groups, sample)
    scans = _scan_edges(
        groups, nodes, declarations, sample, collect_anchors=resolve_datasets
    )
    edge_rows = {scan.label: scan.n_rows for scan in scans}
    any_bad = any(
        any(scan.dangling.values()) or any(scan.mislabeled.values()) for scan in scans
    )
    ladder = (
        _ladder(groups, nodes.table, nodes.leaves)
        if resolve_datasets and any_bad
        else {}
    )
    gaps: list[EndpointGap] = []
    label_gaps: list[LabelGap] = []
    for scan in scans:
        for side in _SIDES:
            if scan.dangling[side]:
                anchor_labels, datasets, n_unresolved = _resolve(
                    groups, nodes, ladder, scan.anchors[side], resolve_datasets
                )
                gaps.append(
                    EndpointGap(
                        edge_label=scan.label,
                        side=side,
                        n_rows=scan.dangling[side],
                        sample_ids=[_unpack_id(s) for s in scan.samples[side]],
                        anchor_labels=anchor_labels,
                        datasets=datasets,
                        n_unresolved=n_unresolved,
                    )
                )
            declaration = scan.declaration
            for leaf, n_rows in sorted(scan.mislabeled[side].items()):
                assert declaration is not None
                anchor_labels, datasets, n_unresolved = _resolve(
                    groups,
                    nodes,
                    ladder,
                    scan.mis_anchors[side].get(leaf, []),
                    resolve_datasets,
                )
                label_gaps.append(
                    LabelGap(
                        edge_label=scan.label,
                        side=side,
                        node_label=leaf,
                        n_rows=n_rows,
                        allowed_labels=(
                            declaration.start_labels
                            if side == ":START_ID"
                            else declaration.end_labels
                        ),
                        sample_ids=[
                            _unpack_id(s) for s in scan.mis_samples[side].get(leaf, [])
                        ],
                        anchor_labels=anchor_labels,
                        datasets=datasets,
                        n_unresolved=n_unresolved,
                    )
                )
    id_collisions = [
        IdCollision(
            first_label=first,
            second_label=second,
            n_ids=nodes.n_collisions[(first, second)],
            sample_ids=[_unpack_id(s) for s in samples],
        )
        for (first, second), samples in sorted(nodes.collisions.items())
    ]
    return EndpointReport(
        out_dir=str(out_dir),
        node_rows=nodes.rows,
        edge_rows=edge_rows,
        n_node_ids=len(nodes.table),
        gaps=gaps,
        label_gaps=label_gaps,
        id_collisions=id_collisions,
        undeclared_edges=[s.label for s in scans if s.declaration is None],
        schema_path=str(schema),
        elapsed_s=time.monotonic() - t0,
        datasets_resolved=resolve_datasets,
    )


def _where(datasets: dict[str, int], n_unresolved: int, resolved: bool) -> str:
    if not resolved:
        return "datasets not resolved (--no-resolve-datasets)"
    where = (
        "; ".join(f"{name}={n}" for name, n in datasets.items())
        if datasets
        else "unresolved"
    )
    unresolved = f", {n_unresolved} unresolved" if datasets and n_unresolved else ""
    return f"datasets: {where}{unresolved}"


def format_report(report: EndpointReport) -> str:
    """The gate's text: one summary line, then one line per gap."""
    lines = [
        f"CSV endpoint gate: {report.out_dir}",
        f"  nodes: {sum(report.node_rows.values())} rows in {len(report.node_rows)} "
        f"labels ({report.n_node_ids} distinct ids); edges: "
        f"{sum(report.edge_rows.values())} rows in {len(report.edge_rows)} labels; "
        f"schema {report.schema_path}; {report.elapsed_s:.0f} s",
    ]
    if report.undeclared_edges:
        lines.append(
            f"  undeclared (labels not checked): {', '.join(report.undeclared_edges)}"
        )
    if report.ok:
        lines.append(
            "  OK: every edge endpoint is the id of a node of a declared label, and no "
            "id is written under two labels"
        )
        return "\n".join(lines)
    if report.gaps:
        lines.append(f"  DANGLING: {report.n_dangling} edge rows name no node")
    for gap in report.gaps:
        if not report.datasets_resolved:
            lines.append(
                f"  {gap.edge_label} {gap.side}: {gap.n_rows} rows; "
                f"{_where({}, 0, False)}; sample {gap.sample_ids}"
            )
            continue
        lines.append(
            f"  {gap.edge_label} {gap.side}: {gap.n_rows} rows; other end "
            f"{gap.anchor_labels or ['(no node either)']}; "
            f"{_where(gap.datasets, gap.n_unresolved, True)}; sample {gap.sample_ids}"
        )
    if report.label_gaps:
        lines.append(
            f"  MISLABELED: {report.n_mislabeled} edge rows name a node of an "
            "undeclared label"
        )
    for label_gap in report.label_gaps:
        lines.append(
            f"  {label_gap.edge_label} {label_gap.side}: {label_gap.n_rows} rows on "
            f"{label_gap.node_label} nodes, declared {label_gap.allowed_labels}; other "
            f"end {label_gap.anchor_labels or ['(unknown)']}; "
            f"{_where(label_gap.datasets, label_gap.n_unresolved, report.datasets_resolved)}; "
            f"sample {label_gap.sample_ids}"
        )
    if report.id_collisions:
        lines.append(
            f"  ID COLLISIONS: {sum(c.n_ids for c in report.id_collisions)} node ids "
            "written under two labels"
        )
    for collision in report.id_collisions:
        lines.append(
            f"  {collision.first_label} and {collision.second_label}: "
            f"{collision.n_ids} ids; sample {collision.sample_ids}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI: print the report, write it as JSON when asked, exit 1 on any finding."""
    parser = argparse.ArgumentParser(
        prog="python -m torchcell.knowledge_graphs.csv_endpoint_gate",
        description="Refuse a BioCypher output directory with an edge endpoint that is "
        "the id of no node, or of a node of a label the schema does not declare for "
        "that edge, before neo4j-admin import drops or misplaces the row.",
    )
    parser.add_argument("out_dir", help="the biocypher-out/<timestamp> directory")
    parser.add_argument(
        "--header-dir",
        default=None,
        help="directory holding <Label>-header.csv when out_dir has none (a generation "
        "killed before BioCypher wrote its headers)",
    )
    parser.add_argument(
        "--schema",
        default=None,
        help=f"the schema config yaml (default: the repository's {SCHEMA_CONFIG_RELPATH})",
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
        schema_path=Path(args.schema) if args.schema else None,
        resolve_datasets=not args.no_resolve_datasets,
    )
    print(format_report(report))
    if args.report:
        Path(args.report).write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
