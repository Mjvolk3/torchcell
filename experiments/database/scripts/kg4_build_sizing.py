# experiments/database/scripts/kg4_build_sizing.py
# [[experiments.database.scripts.kg4_build_sizing]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/kg4_build_sizing
r"""Size the KG 4.0 full rebuild from the KG 3.0 build log and the dev stores.

Two measured inputs, no estimates:

* a live-rebuild generation log (``<job>_generate.log`` under
  ``$BUILD_ROOT/database/slurm/output``), which carries one
  ``Writing nodes for adapter: <A>`` line per adapter, the
  ``single-pass chunk N -> M records (B resolved bytes per record)`` line each
  adapter opens with, and the closing ``Generation wall <s> s, peak memory <g> GiB``;
* the DEV tree the build reads (``$DATA_ROOT/data/torchcell/<slug>``), for each
  mapped dataset's LMDB entry count, LMDB bytes, and build-manifest closure size.

The extrapolation rule, stated so it can be argued with: an adapter's generation
time is modeled as its dev-LMDB bytes divided by the throughput of the family its
record count puts it in. The families are the two code paths, not a taxonomy ---
``inprocess_max_records`` in ``kg_uncapped.yaml`` is the boundary, so a dataset at
or below it is rendered in-process and one above it goes through the chunk pool.
The throughput of each family is the MEDIAN of the per-adapter throughputs the
calibration log measured for that family. Applied back to the calibration build
the rule overstates, because the median understates the largest pooled adapters,
which run fastest; the overstatement is reported rather than tuned away, and the
projection for the new datasets inherits it as headroom.

Run from the repo root::

    python experiments/database/scripts/kg4_build_sizing.py \
        --calibration-log /scratch/projects/torchcell/database/slurm/output/3297_generate.log

Writes ``experiments/database/results/kg4_build_sizing.json`` and the markdown
table ``experiments/database/results/kg4_build_sizing_table.md`` that
``notes/database.kg-4-build-sizing.md`` embeds.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import re
import statistics
from datetime import UTC, datetime
from pathlib import Path

import lmdb
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from torchcell.database.build_dataset_lmdb import dataset_default_root
from torchcell.knowledge_graphs.dataset_adapter_map import build_adapter_map

WRITING_NODES = re.compile(r"Writing nodes for adapter: (\w+)")
CHUNK_LINE = re.compile(
    r"single-pass chunk (\d+) -> (\d+) records \((\d+) resolved bytes per record\)"
)
GEN_WALL = re.compile(r"Generation wall (\d+) s, peak memory ([\d.]+) GiB")
FINISHED = re.compile(
    r"Finished iterating nodes and edges: (\d+) nodes, (\d+) edges across (\d+) adapters"
)
LOG_TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d\d\d)\]")


class CalibrationAdapter(BaseModel):
    """One adapter's measured generation cost in the calibration build."""

    adapter: str
    seconds: float
    resolved_bytes_per_record: int | None = None


class Calibration(BaseModel):
    """The measured generation phase of one live-rebuild job."""

    log_path: str
    generation_wall_s: int
    peak_memory_gib: float
    nodes: int
    edges: int
    adapters: int
    per_adapter: list[CalibrationAdapter]


class DevStore(BaseModel):
    """One mapped dataset's dev-tree LMDB, as it stands on disk."""

    dataset: str
    adapter: str
    slug: str
    organism: str
    private: bool
    records: int
    lmdb_bytes: int
    closure_symbols: int


class SizedDataset(DevStore):
    """A dev store with its family, its measured time if any, and its projection."""

    family: str
    measured_s: float | None = None
    projected_s: float


class Sizing(BaseModel):
    """The whole projection, with the rule's own calibration residual."""

    calibration: Calibration
    pooled_threshold_records: int
    pooled_bytes_per_s: float
    inprocess_bytes_per_s: float
    calibration_projected_s: float
    calibration_overstatement: float
    datasets: list[SizedDataset]
    served_total_s: float
    new_total_s: float
    projected_generation_s: float
    projected_generation_upper_s: float
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


def _log_seconds(line: str) -> float | None:
    match = LOG_TS.match(line)
    if match is None:
        return None
    stamp = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
    return stamp.timestamp() + int(match.group(2)) / 1000


def read_calibration(path: Path) -> Calibration:
    """Per-adapter wall time, the closing totals, and the peak memory, from a log.

    An adapter's wall time is the gap between its ``Writing nodes`` line and the
    next adapter's, because generation is strictly sequential across adapters: the
    pool is rebuilt per adapter and only one adapter writes at a time. The last
    adapter is closed by the ``Finished iterating`` line.
    """
    starts: list[tuple[str, float]] = []
    chunk_bytes: dict[str, int] = {}
    wall = peak = totals = None
    current: str | None = None
    end_s: float | None = None
    with open(path, "rb") as handle:
        for raw in handle:
            line = raw.decode("utf-8", "replace")
            if (match := WRITING_NODES.search(line)) is not None:
                seconds = _log_seconds(line)
                if seconds is not None:
                    current = match.group(1)
                    starts.append((current, seconds))
                continue
            if (match := CHUNK_LINE.search(line)) is not None:
                if current is not None and current not in chunk_bytes:
                    chunk_bytes[current] = int(match.group(3))
                continue
            if (match := FINISHED.search(line)) is not None:
                totals = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
                end_s = _log_seconds(line)
                continue
            if (match := GEN_WALL.search(line)) is not None:
                wall, peak = int(match.group(1)), float(match.group(2))
    if wall is None or peak is None or totals is None or end_s is None:
        raise ValueError(f"{path} carries no completed generation phase")
    per_adapter: list[CalibrationAdapter] = []
    for index, (name, seconds) in enumerate(starts):
        stop = starts[index + 1][1] if index + 1 < len(starts) else end_s
        per_adapter.append(
            CalibrationAdapter(
                adapter=name,
                seconds=round(stop - seconds, 1),
                resolved_bytes_per_record=chunk_bytes.get(name),
            )
        )
    return Calibration(
        log_path=str(path),
        generation_wall_s=wall,
        peak_memory_gib=peak,
        nodes=totals[0],
        edges=totals[1],
        adapters=totals[2],
        per_adapter=per_adapter,
    )


def organism_of(module: str) -> str:
    """The host a loader module sits under, as the dataset tree names it."""
    parts = module.split(".")
    return parts[2] if len(parts) > 2 else parts[-1]


def read_dev_stores(data_root: str) -> list[DevStore]:
    """Every mapped dataset's dev LMDB: entries, bytes, and closure size."""
    public = set(build_adapter_map(include_private=False))
    stores: list[DevStore] = []
    for cls, adapter in build_adapter_map(include_private=True).items():
        relative = dataset_default_root(cls)
        root = osp.join(data_root, relative)
        lmdb_dir = osp.join(root, "processed", "lmdb")
        env = lmdb.open(
            lmdb_dir, readonly=True, lock=False, subdir=True, max_readers=512
        )
        with env.begin() as txn:
            records = txn.stat()["entries"]
        env.close()
        lmdb_bytes = sum(
            os.stat(osp.join(lmdb_dir, name)).st_size for name in os.listdir(lmdb_dir)
        )
        manifest = osp.join(root, "preprocess", "build_manifest.json")
        with open(manifest, encoding="utf-8") as handle:
            closure = len(json.load(handle)["closure"])
        stores.append(
            DevStore(
                dataset=cls.__name__,
                adapter=adapter.__name__,
                slug=relative.split("/")[-1],
                organism=organism_of(cls.__module__),
                private=cls not in public,
                records=records,
                lmdb_bytes=lmdb_bytes,
                closure_symbols=closure,
            )
        )
    return sorted(stores, key=lambda store: store.dataset)


def size_build(
    calibration: Calibration, stores: list[DevStore], pooled_threshold: int
) -> Sizing:
    """Fit the two family throughputs, then project every mapped dataset."""
    measured = {entry.adapter: entry.seconds for entry in calibration.per_adapter}
    by_adapter = {store.adapter: store for store in stores}
    throughput: dict[str, list[float]] = {"pool": [], "inprocess": []}
    for adapter, seconds in measured.items():
        store = by_adapter.get(adapter)
        if store is None or seconds <= 0:
            continue
        family = "pool" if store.records > pooled_threshold else "inprocess"
        throughput[family].append(store.lmdb_bytes / seconds)
    pooled_rate = statistics.median(throughput["pool"])
    inprocess_rate = statistics.median(throughput["inprocess"])

    sized: list[SizedDataset] = []
    for store in stores:
        family = "pool" if store.records > pooled_threshold else "inprocess"
        rate = pooled_rate if family == "pool" else inprocess_rate
        sized.append(
            SizedDataset(
                **store.model_dump(),
                family=family,
                measured_s=measured.get(store.adapter),
                projected_s=round(store.lmdb_bytes / rate, 1),
            )
        )
    served = [entry for entry in sized if entry.measured_s is not None]
    new = [entry for entry in sized if entry.measured_s is None]
    calibration_projected = sum(entry.projected_s for entry in served)
    served_measured = float(calibration.generation_wall_s)
    new_projected = sum(entry.projected_s for entry in new)
    return Sizing(
        calibration=calibration,
        pooled_threshold_records=pooled_threshold,
        pooled_bytes_per_s=round(pooled_rate, 1),
        inprocess_bytes_per_s=round(inprocess_rate, 1),
        calibration_projected_s=round(calibration_projected, 1),
        calibration_overstatement=round(calibration_projected / served_measured, 2),
        datasets=sized,
        served_total_s=served_measured,
        new_total_s=round(new_projected, 1),
        projected_generation_s=round(served_measured + new_projected, 1),
        projected_generation_upper_s=round(calibration_projected + new_projected, 1),
    )


def markdown_table(sizing: Sizing) -> str:
    """The per-dataset table the note embeds, heaviest projection first."""
    lines = [
        "| dataset | host | records | LMDB GB | closure | family | KG 3.0 s | projected s |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for entry in sorted(
        sizing.datasets, key=lambda e: -(e.measured_s or e.projected_s)
    ):
        measured = f"{entry.measured_s:.1f}" if entry.measured_s is not None else "new"
        name = entry.dataset + (" (private)" if entry.private else "")
        lines.append(
            f"| {name} | {entry.organism} | {entry.records:,} "
            f"| {entry.lmdb_bytes / 1e9:.2f} | {entry.closure_symbols} "
            f"| {entry.family} | {measured} | {entry.projected_s:.0f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-log",
        required=True,
        help="generation log of the build whose per-adapter times calibrate the rule",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ["DATA_ROOT"],
        help="DEV tree the build reads (default: DATA_ROOT)",
    )
    parser.add_argument(
        "--pooled-threshold",
        type=int,
        default=25000,
        help="inprocess_max_records from the build config (kg_uncapped.yaml)",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/database/results",
        help="where the JSON and the markdown table are written",
    )
    args = parser.parse_args()

    calibration = read_calibration(Path(args.calibration_log))
    stores = read_dev_stores(args.data_root)
    sizing = size_build(calibration, stores, args.pooled_threshold)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kg4_build_sizing.json").write_text(
        sizing.model_dump_json(indent=1), encoding="utf-8"
    )
    (out_dir / "kg4_build_sizing_table.md").write_text(
        markdown_table(sizing), encoding="utf-8"
    )

    served = [entry for entry in sizing.datasets if entry.measured_s is not None]
    new = [entry for entry in sizing.datasets if entry.measured_s is None]
    print(f"calibration {calibration.log_path}")
    print(
        f"  {calibration.adapters} adapters, {calibration.nodes:,} nodes, "
        f"{calibration.edges:,} edges"
    )
    print(
        f"  generation {calibration.generation_wall_s} s "
        f"({calibration.generation_wall_s / 3600:.2f} h), "
        f"peak {calibration.peak_memory_gib} GiB"
    )
    print(
        f"rule: pool {sizing.pooled_bytes_per_s / 1e6:.1f} MB/s, "
        f"in-process {sizing.inprocess_bytes_per_s / 1e6:.1f} MB/s, "
        f"boundary {sizing.pooled_threshold_records:,} records"
    )
    print(
        f"  back-applied to the calibration build: {sizing.calibration_projected_s:.0f} s "
        f"vs {sizing.served_total_s:.0f} s measured "
        f"(x{sizing.calibration_overstatement})"
    )
    print(
        f"served {len(served)} datasets: "
        f"{sum(e.records for e in served):,} records, "
        f"{sum(e.lmdb_bytes for e in served) / 1e9:.1f} GB LMDB"
    )
    print(
        f"new {len(new)} datasets: "
        f"{sum(e.records for e in new):,} records, "
        f"{sum(e.lmdb_bytes for e in new) / 1e9:.1f} GB LMDB, "
        f"projected {sizing.new_total_s / 3600:.2f} h"
    )
    print(
        f"KG 4.0 generation: {sizing.projected_generation_s / 3600:.2f} h projected, "
        f"{sizing.projected_generation_upper_s / 3600:.2f} h upper bound"
    )
    print(f"wrote {out_dir / 'kg4_build_sizing.json'}")
    print(f"wrote {out_dir / 'kg4_build_sizing_table.md'}")


if __name__ == "__main__":
    main()
