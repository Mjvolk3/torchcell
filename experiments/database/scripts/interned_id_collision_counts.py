# experiments/database/scripts/interned_id_collision_counts.py
# [[experiments.database.scripts.interned_id_collision_counts]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/interned_id_collision_counts
r"""Count the genotype nodes a BioCypher output directory lost to the interned-id collision.

``torchcell.datamodels.interned_constant.constant_id`` hashes ``json.dumps(value)`` of
the interned field, and the genotype node methods of ``CellAdapter`` hash
``json.dumps(genotype.model_dump())``: the same bytes. A whole genotype interned as a
constant (``kind == "genotype"``) therefore carries the id of its own ``Genotype`` /
``PerturbedSegregantGenotype`` node, and a ``genotype.background`` constant carries the
id of the background's ``SegregantGenotype`` node. ``torchcell.fast_csv.FastCsvSink``
deduplicates node ids in ONE set across every label, so whichever label writes an id
first owns it and the other label's node is silently dropped. This script measures both
directions on a written output directory, independently of
``attribute_dangling_perturbation_rows.py`` (lane H), from the CSVs alone:

* genotype nodes lost to ``InternedConstant``: the ``:END_ID`` of ``PerturbationMemberOf``,
  the ``:START_ID`` of ``GenotypeMemberOf`` and the ``:START_ID`` of
  ``SegregantBackgroundOf`` that name no genotype node and DO name an interned constant,
  by the constant's kind, per dataset (through ``GenotypeMemberOf`` and
  ``ExperimentMemberOf``);
* constants lost to a genotype label: every ``{"$ref": <id>, "kind": "genotype"}`` or
  ``"genotype.background"`` pointer in the ``Experiment`` blobs whose id names no
  interned constant (the query loader raises on such a pointer). The pointers come
  from a prior ``grep`` over the Experiment file (``--pointers``), because that file is
  261 GB on the job 4901 partial.

Usage (from the repo root, the torchcell environment)::

    LC_ALL=C grep -oE '"\$ref": "[0-9a-f]{64}", "kind": "genotype(\.background)?"' \\
        /db/database/biocypher-out/2026-10-14_03-53-33/Experiment-part000.csv \\
        > /tmp/experiment_genotype_pointers.txt
    python experiments/database/scripts/interned_id_collision_counts.py \\
        /db/database/biocypher-out/2026-10-14_03-53-33 \\
        --pointers /tmp/experiment_genotype_pointers.txt \\
        --out experiments/database/results/interned_id_collision_4901.json

Reads the directory only.
"""

from __future__ import annotations

import argparse
import re
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

GENOTYPE_LABELS = ("Genotype", "SegregantGenotype", "PerturbedSegregantGenotype")
PERTURBATION_LABELS = (
    "Perturbation",
    "BacterialPerturbation",
    "BacterialSequenceVariantPerturbation",
    "PhagePerturbation",
)
INTERNED_LABEL = "InternedConstant"
GENOTYPE_KINDS = ("genotype", "genotype.background")
_TAB = b"\t"
_POINTER = re.compile(
    rb'"\$ref": "([0-9a-f]{64})", "kind": "(genotype(?:\.background)?)"'
)


class EdgeSideCounts(BaseModel):
    """One edge file's endpoint resolution on the genotype side."""

    model_config = ConfigDict(frozen=True)

    edge: str
    side: str
    n_rows: int
    n_resolving_to_genotype_node: int
    n_resolving_to_interned_only: dict[str, int] = Field(
        description="rows whose endpoint is ONLY an interned constant, by constant kind"
    )
    n_resolving_to_nothing: int
    n_distinct_interned_only: int


class DatasetLoss(BaseModel):
    """The genotype nodes one dataset lost, and the edge rows that point at them."""

    model_config = ConfigDict(frozen=True)

    dataset: str
    n_lost_genotype_nodes: int
    lost_by_kind: dict[str, int]
    n_perturbation_member_of_rows: int
    n_genotype_member_of_rows: int


class CollisionCounts(BaseModel):
    """The measurement of one output directory."""

    model_config = ConfigDict(frozen=True)

    out_dir: str
    n_node_ids: dict[str, int]
    n_interned_by_kind: dict[str, int]
    n_interned_genotype_kind_also_a_genotype_node: int = Field(
        description="should be 0: the sink never writes one id under two labels"
    )
    edges: list[EdgeSideCounts]
    datasets: list[DatasetLoss]
    n_lost_genotype_ids_unattributed: int
    n_pointers: dict[str, int] | None
    n_pointers_lost_to_genotype_label: dict[str, int] | None = Field(
        description="pointers in the Experiment blobs whose id is no interned constant "
        "but IS a genotype node (the query loader raises on these)"
    )
    n_pointers_resolving_to_nothing: dict[str, int] | None
    elapsed_s: float


def _rows(path: Path) -> Iterator[bytes]:
    with path.open("rb", buffering=1 << 24) as fh:
        yield from fh


def _node_ids(out_dir: Path, label: str) -> set[bytes]:
    path = out_dir / f"{label}-part000.csv"
    if not path.exists():
        return set()
    return {line.split(_TAB, 1)[0] for line in _rows(path)}


def _interned(out_dir: Path) -> dict[bytes, str]:
    """``id -> kind`` of every interned constant."""
    out: dict[bytes, str] = {}
    for line in _rows(out_dir / f"{INTERNED_LABEL}-part000.csv"):
        node_id, kind, _ = line.split(_TAB, 2)
        out[node_id] = kind.strip(b"'").decode()
    return out


def _edge_endpoints(path: Path, side: str) -> Iterator[tuple[bytes, bytes]]:
    """``(endpoint, other endpoint)`` for ``side`` in ``{"start", "end"}``."""
    for line in _rows(path):
        start, _, end, _ = line.split(_TAB, 3)
        yield (start, end) if side == "start" else (end, start)


def measure(out_dir: Path, pointers: Path | None) -> CollisionCounts:
    t0 = time.monotonic()
    genotype_ids: dict[str, set[bytes]] = {
        label: _node_ids(out_dir, label) for label in GENOTYPE_LABELS
    }
    all_genotype = set().union(*genotype_ids.values())
    interned = _interned(out_dir)
    n_node_ids = {label: len(ids) for label, ids in genotype_ids.items()}
    n_node_ids[INTERNED_LABEL] = len(interned)
    for label in PERTURBATION_LABELS:
        n_node_ids[label] = len(_node_ids(out_dir, label))
    n_interned_by_kind = dict(Counter(interned.values()))
    n_both = sum(
        1
        for node_id, kind in interned.items()
        if kind in GENOTYPE_KINDS and node_id in all_genotype
    )
    print(
        f"loaded {len(all_genotype)} genotype ids, {len(interned)} interned ids; "
        f"{time.monotonic() - t0:.0f} s",
        flush=True,
    )

    edges: list[EdgeSideCounts] = []
    lost: dict[bytes, str] = {}  # lost genotype id -> interned kind
    pmo_rows_per_lost: Counter[bytes] = Counter()
    gmo_rows_per_lost: Counter[bytes] = Counter()
    lost_to_experiments: dict[bytes, set[bytes]] = defaultdict(set)
    specs = (
        ("PerturbationMemberOf", "end", all_genotype, pmo_rows_per_lost),
        ("GenotypeMemberOf", "start", all_genotype, gmo_rows_per_lost),
        ("SegregantBackgroundOf", "start", genotype_ids["SegregantGenotype"], None),
    )
    for edge, side, known, per_lost in specs:
        path = out_dir / f"{edge}-part000.csv"
        n_rows = n_known = n_nothing = 0
        interned_only: Counter[str] = Counter()
        distinct: set[bytes] = set()
        for endpoint, other in _edge_endpoints(path, side):
            n_rows += 1
            if endpoint in known:
                n_known += 1
                continue
            kind = interned.get(endpoint)
            if kind is None:
                n_nothing += 1
                continue
            interned_only[kind] += 1
            distinct.add(endpoint)
            lost[endpoint] = kind
            if per_lost is not None:
                per_lost[endpoint] += 1
            if edge == "GenotypeMemberOf":
                lost_to_experiments[endpoint].add(other)
        edges.append(
            EdgeSideCounts(
                edge=edge,
                side=side,
                n_rows=n_rows,
                n_resolving_to_genotype_node=n_known,
                n_resolving_to_interned_only=dict(interned_only),
                n_resolving_to_nothing=n_nothing,
                n_distinct_interned_only=len(distinct),
            )
        )
        print(f"{edge}: {edges[-1]}; {time.monotonic() - t0:.0f} s", flush=True)

    wanted_experiments = set().union(*lost_to_experiments.values()) if lost else set()
    experiment_to_dataset: dict[bytes, str] = {}
    for experiment, dataset in _edge_endpoints(
        out_dir / "ExperimentMemberOf-part000.csv", "start"
    ):
        if experiment in wanted_experiments:
            experiment_to_dataset[experiment] = dataset.decode()
    per_dataset: dict[str, dict[str, Counter[str] | int]] = {}
    unattributed = 0
    for genotype_id, kind in lost.items():
        names = {
            experiment_to_dataset[e]
            for e in lost_to_experiments.get(genotype_id, ())
            if e in experiment_to_dataset
        }
        if not names:
            unattributed += 1
            continue
        for name in names:
            entry = per_dataset.setdefault(
                name, {"kinds": Counter(), "pmo": 0, "gmo": 0}
            )
            entry["kinds"][kind] += 1  # type: ignore[index]
            entry["pmo"] += pmo_rows_per_lost[genotype_id]  # type: ignore[operator]
            entry["gmo"] += gmo_rows_per_lost[genotype_id]  # type: ignore[operator]
    datasets = [
        DatasetLoss(
            dataset=name,
            n_lost_genotype_nodes=sum(entry["kinds"].values()),  # type: ignore[union-attr]
            lost_by_kind=dict(entry["kinds"]),  # type: ignore[arg-type]
            n_perturbation_member_of_rows=int(entry["pmo"]),  # type: ignore[arg-type]
            n_genotype_member_of_rows=int(entry["gmo"]),  # type: ignore[arg-type]
        )
        for name, entry in sorted(per_dataset.items())
    ]

    n_pointers = n_lost_pointers = n_nothing_pointers = None
    if pointers is not None:
        n_pointers_c: Counter[str] = Counter()
        lost_c: Counter[str] = Counter()
        nothing_c: Counter[str] = Counter()
        for line in _rows(pointers):
            match = _POINTER.search(line)
            if match is None:
                continue
            ref, kind = match.group(1), match.group(2).decode()
            n_pointers_c[kind] += 1
            if ref in interned:
                continue
            if ref in all_genotype:
                lost_c[kind] += 1
            else:
                nothing_c[kind] += 1
        n_pointers, n_lost_pointers, n_nothing_pointers = (
            dict(n_pointers_c),
            dict(lost_c),
            dict(nothing_c),
        )
    return CollisionCounts(
        out_dir=str(out_dir),
        n_node_ids=n_node_ids,
        n_interned_by_kind=n_interned_by_kind,
        n_interned_genotype_kind_also_a_genotype_node=n_both,
        edges=edges,
        datasets=datasets,
        n_lost_genotype_ids_unattributed=unattributed,
        n_pointers=n_pointers,
        n_pointers_lost_to_genotype_label=n_lost_pointers,
        n_pointers_resolving_to_nothing=n_nothing_pointers,
        elapsed_s=time.monotonic() - t0,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir", help="the biocypher-out/<timestamp> directory")
    parser.add_argument(
        "--pointers",
        default=None,
        help="grep output of the genotype pointers in the Experiment blobs",
    )
    parser.add_argument("--out", required=True, help="write the counts JSON here")
    args = parser.parse_args(argv)
    result = measure(
        Path(args.out_dir), None if args.pointers is None else Path(args.pointers)
    )
    print(result.model_dump_json(indent=2))
    Path(args.out).write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
