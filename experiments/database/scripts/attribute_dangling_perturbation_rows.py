# experiments/database/scripts/attribute_dangling_perturbation_rows.py
# [[experiments.database.scripts.attribute_dangling_perturbation_rows]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/attribute_dangling_perturbation_rows
r"""Attribute every dangling ``PerturbationMemberOf`` row of a BioCypher output directory.

The CSV endpoint gate (``torchcell.knowledge_graphs.csv_endpoint_gate``) counted 4,233
``PerturbationMemberOf`` rows naming no node in the job 4901 partial (2,302 whose
``:START_ID`` is no perturbation node, 1,931 whose ``:END_ID`` is no genotype node), where
the audit had estimated 12 rows for the Carruthers gene-addition gap. This script reads the
same directory and says, per dangling row, which dataset wrote it and why its endpoint is
missing, so the number is attributed before the KG 4.0 build (lane H of the ontology
unification, 2026-10-15).

Method, all on the written CSVs and nothing else:

1. The ids of every perturbation node file (``Perturbation``, ``BacterialPerturbation``,
   ``BacterialSequenceVariantPerturbation``) and every genotype node file (``Genotype``,
   ``SegregantGenotype``, ``PerturbedSegregantGenotype``), first field of each row.
2. One pass over ``PerturbationMemberOf``: each row's index, start and end; a row is
   ``start_dangling`` when its start is no perturbation id and ``end_dangling`` when its
   end is no genotype id. The index of the last row whose start or end IS known before
   each dangling run is kept, so a dangling row can be attributed to its neighbor's
   dataset when its own genotype is missing.
3. One pass over ``GenotypeMemberOf`` (genotype -> experiment) for the anchor genotype
   ids (the known end of a start-dangling row, the missing end of an end-dangling row and
   the neighbor anchors), then one pass over ``ExperimentMemberOf`` (experiment ->
   dataset; a ``Dataset`` id is the dataset class name, so the empty ``Dataset`` node
   file of a partial run does not matter).
4. The missing ids against the ``:ID`` column of EVERY other node file but
   ``Experiment`` (261 G on the 4901 partial; an experiment id is the sha256 of a whole
   record and cannot equal a sub-object's), because ``torchcell.fast_csv.FastCsvSink``
   deduplicates node ids GLOBALLY across labels (``seen_nodes``, one set), and an
   ``InternedConstant`` id is the sha256 of the same dump a genotype node hashes
   (``torchcell.datamodels.interned_constant``): an id written first under one label is
   dropped under the next. The ``SegregantBackgroundOf`` edge file's start ids are checked
   the same way, since the ``SegregantGenotype`` node file of the partial is empty.
5. The report: per dataset, the dangling rows by side, the label each missing id was
   found under (or none), whether the missing genotype has a ``GenotypeMemberOf`` edge,
   and the contiguous index runs of the dangling rows, which locate the writing adapter.

Usage (from the repo root, the torchcell environment)::

    python experiments/database/scripts/attribute_dangling_perturbation_rows.py \\
        /db/database/biocypher-out/2026-10-14_03-53-33 \\
        --out experiments/database/results/dangling_perturbation_rows_4901.json

Reads the directory only. The two big edge files (14.4 G and 10.9 G on the 4901 partial)
are streamed once each.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

PERTURBATION_LABELS = (
    "Perturbation",
    "BacterialPerturbation",
    "BacterialSequenceVariantPerturbation",
)
GENOTYPE_LABELS = ("Genotype", "SegregantGenotype", "PerturbedSegregantGenotype")
EDGE_LABEL = "PerturbationMemberOf"
GENOTYPE_EDGE = "GenotypeMemberOf"
EXPERIMENT_EDGE = "ExperimentMemberOf"
BACKGROUND_EDGE = "SegregantBackgroundOf"
#: Node files not searched for a missing id: an experiment id hashes a whole record.
UNSEARCHED_LABELS = ("Experiment",)
_TAB = b"\t"


class DanglingRow(BaseModel):
    """One ``PerturbationMemberOf`` row with an endpoint that is no node."""

    model_config = ConfigDict(frozen=True)

    index: int
    side: str  # "start" | "end" | "both"
    start: str
    end: str
    start_label: str | None  # the node label the start id has, None when dangling
    neighbor_index: int  # the last row before this one with a known end
    neighbor_end: str
    #: node label (not a perturbation or genotype label) the missing id was found under
    start_found_under: str | None = None
    end_found_under: str | None = None


class DatasetAttribution(BaseModel):
    """The dangling rows one dataset wrote, by side and by cause."""

    model_config = ConfigDict(frozen=True)

    dataset: str
    n_rows: int
    by_side: dict[str, int]
    start_label_of_end_dangling: dict[str, int]
    #: end-dangling rows whose missing genotype id has a GenotypeMemberOf edge
    n_missing_genotype_with_edge: int
    #: end-dangling rows whose missing genotype id has no GenotypeMemberOf edge
    n_missing_genotype_without_edge: int
    #: how the dataset was reached: the row's own genotype or its neighbor's
    resolved_via: dict[str, int]
    #: the missing start ids by the label they were found under ("(none)" when nowhere)
    start_found_under: dict[str, int]
    end_found_under: dict[str, int]
    #: distinct missing ids (a node row is written once; an edge row once per record)
    n_distinct_missing_start: int
    n_distinct_missing_end: int
    index_runs: list[tuple[int, int]]


class Attribution(BaseModel):
    """What the script measured over one output directory."""

    model_config = ConfigDict(frozen=True)

    out_dir: str
    n_rows: int
    n_perturbation_ids: dict[str, int]
    n_genotype_ids: dict[str, int]
    n_start_dangling: int
    n_end_dangling: int
    n_both_dangling: int
    datasets: list[DatasetAttribution]
    n_unresolved: int
    unresolved_sample: list[DanglingRow] = Field(default_factory=list)
    #: SegregantBackgroundOf rows, and how many of their start ids are a SegregantGenotype
    #: node, found under another label, or nowhere
    n_background_rows: int
    background_start_under: dict[str, int]
    elapsed_s: float


def _pack(raw: bytes) -> bytes:
    if len(raw) >= 2 and raw[:1] == b"'" and raw[-1:] == b"'":
        raw = raw[1:-1]
    if len(raw) == 64:
        return bytes.fromhex(raw.decode("ascii"))
    return raw


def _unpack(packed: bytes) -> str:
    return packed.hex() if len(packed) == 32 else packed.decode("utf-8")


def _part_files(out_dir: Path, label: str) -> list[Path]:
    return sorted(out_dir.glob(f"{label}-part*.csv"))


def _node_ids(out_dir: Path, label: str) -> set[bytes]:
    ids: set[bytes] = set()
    for part in _part_files(out_dir, label):
        with open(part, "rb") as handle:
            for line in handle:
                tab = line.find(_TAB)
                ids.add(_pack(line[:tab] if tab >= 0 else line.rstrip(b"\n")))
    return ids


def _node_labels(out_dir: Path) -> list[str]:
    """Every node label of the directory, read off the part files without headers.

    A node file's rows begin with the id and end with the quoted label list; an edge
    file's rows begin with ``:START_ID`` and end with the quoted type. The two are told
    apart by the first row: an edge row's third field is a 64-hex id or a dataset name
    and its last field is the type, while a node row's last field holds the label list
    with the ``|`` separator or the label itself. Here the distinction is made by name:
    an edge label ends in ``Of`` or is ``Mentions``, which is every edge file BioCypher
    writes for this schema.
    """
    labels: list[str] = []
    for part in sorted(out_dir.glob("*-part*.csv")):
        label = part.name.split("-part")[0]
        if label.endswith("Of") or label == "Mentions" or label in labels:
            continue
        labels.append(label)
    return labels


def _edge_rows(out_dir: Path, label: str) -> Iterator[tuple[bytes, bytes]]:
    """Yield ``(start, end)`` packed over every row of one edge label."""
    for part in _part_files(out_dir, label):
        with open(part, "rb") as handle:
            for line in handle:
                fields = line.rstrip(b"\n").split(_TAB)
                yield _pack(fields[0]), _pack(fields[2])


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    for index in sorted(indices):
        if runs and index == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], index)
        else:
            runs.append((index, index))
    return runs


class _DatasetRows:
    """The dangling rows of one dataset while they are being counted."""

    def __init__(self) -> None:
        self.by_side: Counter[str] = Counter()
        self.start_label: Counter[str] = Counter()
        self.end_under: Counter[str] = Counter()
        self.start_under: Counter[str] = Counter()
        self.via: Counter[str] = Counter()
        self.with_edge = 0
        self.without_edge = 0
        self.missing_start: set[str] = set()
        self.missing_end: set[str] = set()
        self.indices: list[int] = []


def attribute(out_dir: Path) -> Attribution:
    """Measure the dangling rows and attribute each to a dataset."""
    t0 = time.monotonic()
    perturbation_label: dict[bytes, str] = {}
    n_perturbation_ids: dict[str, int] = {}
    for label in PERTURBATION_LABELS:
        ids = _node_ids(out_dir, label)
        n_perturbation_ids[label] = len(ids)
        for packed in ids:
            perturbation_label[packed] = label
    genotype_ids: set[bytes] = set()
    n_genotype_ids: dict[str, int] = {}
    for label in GENOTYPE_LABELS:
        ids = _node_ids(out_dir, label)
        n_genotype_ids[label] = len(ids)
        genotype_ids |= ids
    print(
        f"node ids: {sum(n_perturbation_ids.values())} perturbation, "
        f"{len(genotype_ids)} genotype; {time.monotonic() - t0:.0f} s",
        flush=True,
    )

    rows: list[DanglingRow] = []
    n_rows = 0
    last_known_index = -1
    last_known_end = b""
    for index, (start, end) in enumerate(_edge_rows(out_dir, EDGE_LABEL)):
        n_rows += 1
        start_ok = start in perturbation_label
        end_ok = end in genotype_ids
        if start_ok and end_ok:
            last_known_index = index
            last_known_end = end
            continue
        side = (
            "both"
            if not start_ok and not end_ok
            else ("start" if not start_ok else "end")
        )
        rows.append(
            DanglingRow(
                index=index,
                side=side,
                start=_unpack(start),
                end=_unpack(end),
                start_label=perturbation_label.get(start),
                neighbor_index=last_known_index,
                neighbor_end=_unpack(last_known_end) if last_known_end else "",
            )
        )
        if end_ok:
            last_known_index = index
            last_known_end = end
    print(
        f"{EDGE_LABEL}: {n_rows} rows, {len(rows)} dangling; "
        f"{time.monotonic() - t0:.0f} s",
        flush=True,
    )

    anchors: set[bytes] = set()
    for row in rows:
        anchors.add(bytes.fromhex(row.end))
        if row.neighbor_end:
            anchors.add(bytes.fromhex(row.neighbor_end))
    genotype_to_experiments: dict[bytes, set[bytes]] = defaultdict(set)
    for start, end in _edge_rows(out_dir, GENOTYPE_EDGE):
        if start in anchors:
            genotype_to_experiments[start].add(end)
    experiments: set[bytes] = set()
    for members in genotype_to_experiments.values():
        experiments |= members
    experiment_to_dataset: dict[bytes, set[str]] = defaultdict(set)
    for start, end in _edge_rows(out_dir, EXPERIMENT_EDGE):
        if start in experiments:
            experiment_to_dataset[start].add(_unpack(end))
    print(
        f"ladder: {len(anchors)} anchors, {len(genotype_to_experiments)} with a "
        f"{GENOTYPE_EDGE} edge, {len(experiments)} experiments; "
        f"{time.monotonic() - t0:.0f} s",
        flush=True,
    )

    missing: set[bytes] = set()
    for row in rows:
        if row.side != "end":
            missing.add(bytes.fromhex(row.start))
        if row.side != "start":
            missing.add(bytes.fromhex(row.end))
    background_starts: set[bytes] = set()
    n_background_rows = 0
    for start, _end in _edge_rows(out_dir, BACKGROUND_EDGE):
        n_background_rows += 1
        background_starts.add(start)
    found_under: dict[bytes, str] = {}
    searched = [
        label
        for label in _node_labels(out_dir)
        if label not in UNSEARCHED_LABELS
        and label not in PERTURBATION_LABELS
        and label not in GENOTYPE_LABELS
    ]
    wanted = missing | background_starts
    for label in searched:
        for packed in _node_ids(out_dir, label):
            if packed in wanted and packed not in found_under:
                found_under[packed] = label
    genotype_hits = _node_ids(out_dir, "SegregantGenotype")
    background_start_under: Counter[str] = Counter()
    for packed in background_starts:
        if packed in genotype_hits:
            background_start_under["SegregantGenotype"] += 1
        else:
            background_start_under[found_under.get(packed, "(none)")] += 1
    rows = [
        row.model_copy(
            update={
                "start_found_under": (
                    found_under.get(bytes.fromhex(row.start))
                    if row.side != "end"
                    else None
                ),
                "end_found_under": (
                    found_under.get(bytes.fromhex(row.end))
                    if row.side != "start"
                    else None
                ),
            }
        )
        for row in rows
    ]
    print(
        f"other labels: searched {len(searched)} node files, {len(found_under)} of "
        f"{len(wanted)} missing ids found under another label; "
        f"{time.monotonic() - t0:.0f} s",
        flush=True,
    )

    def datasets_of(genotype_hex: str) -> set[str]:
        names: set[str] = set()
        for experiment in genotype_to_experiments.get(bytes.fromhex(genotype_hex), ()):
            names |= experiment_to_dataset.get(experiment, set())
        return names

    per_dataset: dict[str, _DatasetRows] = {}
    unresolved: list[DanglingRow] = []
    for row in rows:
        names = datasets_of(row.end)
        via = "own genotype"
        if not names and row.neighbor_end:
            names = datasets_of(row.neighbor_end)
            via = "neighbor genotype"
        if not names:
            unresolved.append(row)
            continue
        missing_has_edge = (
            row.side != "start" and bytes.fromhex(row.end) in genotype_to_experiments
        )
        for name in names:
            entry = per_dataset.setdefault(name, _DatasetRows())
            entry.by_side[row.side] += 1
            if row.side != "start":
                entry.start_label[row.start_label or "(none)"] += 1
                entry.end_under[row.end_found_under or "(none)"] += 1
                if missing_has_edge:
                    entry.with_edge += 1
                else:
                    entry.without_edge += 1
                entry.missing_end.add(row.end)
            if row.side != "end":
                entry.start_under[row.start_found_under or "(none)"] += 1
                entry.missing_start.add(row.start)
            entry.via[via] += 1
            entry.indices.append(row.index)

    datasets = [
        DatasetAttribution(
            dataset=name,
            n_rows=len(entry.indices),
            by_side=dict(entry.by_side),
            start_label_of_end_dangling=dict(entry.start_label),
            n_missing_genotype_with_edge=entry.with_edge,
            n_missing_genotype_without_edge=entry.without_edge,
            resolved_via=dict(entry.via),
            start_found_under=dict(entry.start_under),
            end_found_under=dict(entry.end_under),
            n_distinct_missing_start=len(entry.missing_start),
            n_distinct_missing_end=len(entry.missing_end),
            index_runs=_runs(entry.indices),
        )
        for name, entry in sorted(per_dataset.items())
    ]
    return Attribution(
        out_dir=str(out_dir),
        n_rows=n_rows,
        n_perturbation_ids=n_perturbation_ids,
        n_genotype_ids=n_genotype_ids,
        n_start_dangling=sum(1 for r in rows if r.side == "start"),
        n_end_dangling=sum(1 for r in rows if r.side == "end"),
        n_both_dangling=sum(1 for r in rows if r.side == "both"),
        datasets=datasets,
        n_unresolved=len(unresolved),
        unresolved_sample=unresolved[:20],
        n_background_rows=n_background_rows,
        background_start_under=dict(background_start_under),
        elapsed_s=time.monotonic() - t0,
    )


def format_attribution(result: Attribution) -> str:
    """The measurement as text, one line per dataset."""
    lines = [
        f"{EDGE_LABEL} of {result.out_dir}: {result.n_rows} rows; dangling start "
        f"{result.n_start_dangling}, end {result.n_end_dangling}, both "
        f"{result.n_both_dangling}; {result.elapsed_s:.0f} s",
        f"  perturbation node ids {result.n_perturbation_ids}; genotype node ids "
        f"{result.n_genotype_ids}",
    ]
    for d in result.datasets:
        lines.append(
            f"  {d.dataset}: {d.n_rows} rows, sides {d.by_side}, start labels of "
            f"end-dangling {d.start_label_of_end_dangling}, missing start found under "
            f"{d.start_found_under}, missing end found under {d.end_found_under}, "
            f"distinct missing start {d.n_distinct_missing_start}, end "
            f"{d.n_distinct_missing_end}, "
            f"missing genotype with GenotypeMemberOf edge "
            f"{d.n_missing_genotype_with_edge}, without "
            f"{d.n_missing_genotype_without_edge}, resolved via {d.resolved_via}, "
            f"index runs {d.index_runs[:6]}{' ...' if len(d.index_runs) > 6 else ''}"
        )
    lines.append(f"  unresolved: {result.n_unresolved}")
    lines.append(
        f"  {BACKGROUND_EDGE}: {result.n_background_rows} rows; start ids under "
        f"{result.background_start_under}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir", help="the biocypher-out/<timestamp> directory")
    parser.add_argument("--out", required=True, help="write the attribution JSON here")
    args = parser.parse_args(argv)
    result = attribute(Path(args.out_dir))
    print(format_attribution(result))
    Path(args.out).write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
