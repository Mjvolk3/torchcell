# experiments/database/scripts/kg_build_adapter_resources.py
# [[experiments.database.scripts.kg_build_adapter_resources]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/kg_build_adapter_resources
"""Per-adapter CPU, memory and pool-recycle table of one KG generation run.

The generator samples its container cgroup every 5 s into
``biocypher-out/<ts>/telemetry/resource_samples.csv`` (``torchcell.build_telemetry``), each
sample tagged with the adapter and phase (node or edge) current at that instant. This
script folds those samples into one row per adapter and phase, counts the pool recycles
the generate log reports inside each phase's window, and writes the table as JSON and
Markdown under ``experiments/database/results/``. It reads only what the build already
wrote; nothing here runs a build.

Run from the repo root (job 4901, the KG 4.0 generation, is the calibration run):

    PYTHONPATH=. ~/miniconda3/envs/torchcell/bin/python \
        experiments/database/scripts/kg_build_adapter_resources.py \
        --telemetry /db/database/biocypher-out/2026-10-14_03-53-33/telemetry/resource_samples.csv \
        --generate-log /scratch/projects/torchcell/database/slurm/output/4901_generate.log \
        --label 4901

Columns: hours in the phase, mean and max cores in use (every process in the cgroup),
mean cores of the parent process alone (the CSV writer), mean and max resident memory in
GiB, the largest process count seen, and the pool recycles (``pool recycled at N chunks:
cgroup memory at F of its limit``) whose timestamps fall inside the phase. A phase whose
mean cores sit far below its max while recycles pile up is memory-throttled, not
compute-bound: the pool stops taking chunks once cgroup memory passes
``pool_memory_fraction`` of the container limit and is rebuilt.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

RECYCLE_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+\]\[torchcell\.adapters\.cell_adapter\]"
    r"\[INFO\] - pool recycled at (\d+) chunks: cgroup memory at ([0-9.]+) of its limit"
)


class PhaseResources(BaseModel):
    """One adapter phase folded from the 5 s cgroup samples."""

    adapter: str
    phase_kind: str = Field(description="node | edge")
    start_t: float = Field(description="Unix time of the first sample in the phase")
    end_t: float = Field(description="Unix time of the last sample in the phase")
    hours: float
    samples: int
    cores_mean: float = Field(description="Mean cores in use over the phase")
    cores_max: float
    parent_cores_mean: float = Field(description="Mean cores of the main process alone")
    mem_mean_gb: float
    mem_max_gb: float
    n_procs_max: int
    recycles: int = Field(
        default=0, description="Pool recycles logged inside [start_t, end_t]"
    )

    def markdown_row(self) -> str:
        """One Markdown table row in the column order of MARKDOWN_HEADER."""
        return (
            f"| {self.adapter} | {self.phase_kind} | {self.hours:.2f} | {self.cores_mean:.1f} "
            f"| {self.cores_max:.1f} | {self.parent_cores_mean:.2f} | {self.mem_mean_gb:.0f} "
            f"| {self.mem_max_gb:.0f} | {self.n_procs_max} | {self.recycles} |"
        )


MARKDOWN_HEADER = (
    "| adapter | phase | hours | cores mean | cores max | parent cores | mem mean (GiB) "
    "| mem max (GiB) | procs | recycles |\n"
    "|---|---|---|---|---|---|---|---|---|---|"
)


def read_recycles(generate_log: Path | None) -> list[float]:
    """Unix times of every pool recycle the generate log reports (container clock is UTC)."""
    if generate_log is None:
        return []
    times: list[float] = []
    with open(generate_log, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            for match in RECYCLE_RE.finditer(line):
                stamp = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
                times.append(stamp.replace(tzinfo=UTC).timestamp())
    return sorted(times)


def fold_samples(telemetry: Path, recycles: list[float]) -> list[PhaseResources]:
    """One row per (adapter, phase_kind), in the order the phases ran."""
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    order: list[tuple[str, str]] = []
    with open(telemetry, newline="") as handle:
        for row in csv.DictReader(handle):
            if not row["adapter"]:
                continue
            key = (row["adapter"], row["phase_kind"])
            if key not in groups:
                order.append(key)
            groups[key].append(row)
    rows: list[PhaseResources] = []
    for adapter, kind in order:
        samples = groups[(adapter, kind)]
        ts = [float(s["t"]) for s in samples]
        cores = [float(s["cpu_cores"]) for s in samples]
        parent = [float(s["parent_cores"]) for s in samples]
        mem = [float(s["mem_gb"]) for s in samples]
        start_t, end_t = min(ts), max(ts)
        rows.append(
            PhaseResources(
                adapter=adapter,
                phase_kind=kind,
                start_t=start_t,
                end_t=end_t,
                hours=(end_t - start_t) / 3600,
                samples=len(samples),
                cores_mean=sum(cores) / len(cores),
                cores_max=max(cores),
                parent_cores_mean=sum(parent) / len(parent),
                mem_mean_gb=sum(mem) / len(mem),
                mem_max_gb=max(mem),
                n_procs_max=max(int(s["n_procs"]) for s in samples),
                recycles=sum(1 for t in recycles if start_t <= t <= end_t),
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--generate-log", type=Path, default=None)
    parser.add_argument(
        "--label", required=True, help="job id or run label for the output names"
    )
    parser.add_argument(
        "--top", type=int, default=20, help="rows printed, longest phases first"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("experiments/database/results")
    )
    args = parser.parse_args()

    recycles = read_recycles(args.generate_log)
    rows = fold_samples(args.telemetry, recycles)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.results_dir / f"kg_build_adapter_resources_{args.label}.json"
    md_path = args.results_dir / f"kg_build_adapter_resources_{args.label}.md"
    json_path.write_text(json.dumps([r.model_dump() for r in rows], indent=1))
    longest = sorted(rows, key=lambda r: -r.hours)
    total_h = sum(r.hours for r in rows)
    md_path.write_text(
        f"Per-adapter resources of run {args.label}: {len(rows)} phases, "
        f"{total_h:.2f} adapter-hours, {len(recycles)} pool recycles in the log. "
        f"Generated by `experiments/database/scripts/kg_build_adapter_resources.py` from "
        f"`{args.telemetry}`.\n\n"
        + MARKDOWN_HEADER
        + "\n"
        + "\n".join(r.markdown_row() for r in longest[: args.top])
        + "\n"
    )
    print(md_path.read_text())
    print(f"wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
