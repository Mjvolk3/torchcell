# experiments/tcdb-002-build-speed/scripts/live_rebuild_report.py
# [[experiments.tcdb-002-build-speed.scripts.live_rebuild_report]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/tcdb-002-build-speed/scripts/live_rebuild_report
r"""Per-adapter wall times and whole-build figures of a live KG rebuild, from its logs.

A live rebuild (database/slurm/scripts/gilahyper_live_rebuild-slurm_docker.slurm)
leaves two logs under ``$BUILD_ROOT/database/slurm/output``: ``<job>_generate.log``
(the generation container's stdout: the build script's timestamped INFO lines, one
``Writing nodes for adapter: X`` and one ``Writing edges for adapter: X`` per adapter,
then ``Generation wall N s, peak memory M GiB``) and ``<job>_live_rebuild_kg.out``
(the slurm job: import time, node counts, validation, swap, release). The CSV directory
itself is owned by the import user, so the telemetry samples inside it are read through
the W&B run the generation logged them to (``res/*`` per 5 s sample).

Writes ``results/live_rebuild_<job>.csv`` (one row per adapter: nodes_s, edges_s,
total_s, from the log timestamps, so each is accurate to a second) and
``results/live_rebuild_<job>_summary.csv`` (generation wall, peak cgroup memory, peak
anonymous memory, mean cores, import time, node and edge counts, release).

    ~/miniconda3/envs/torchcell/bin/python \\
        experiments/tcdb-002-build-speed/scripts/live_rebuild_report.py --job 3198
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os.path as osp
import re
from typing import Any

from dotenv import load_dotenv

RESULTS_DIR = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "results")
OUTPUT_DIR = "/scratch/projects/torchcell/database/slurm/output"

EVENT = re.compile(
    r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+\]\[__main__\]\[INFO\] - "
    r"(?:Writing (nodes|edges) for adapter: (\w+)|(Finished iterating)|(Generation wall))"
)
WALL = re.compile(r"Generation wall (\d+) s, peak memory ([0-9.]+) GiB")
IMPORT = re.compile(r"IMPORT DONE in (\d+)m (\d+)s")
COUNTS = re.compile(r"(\d+) nodes, (\d+) edges")
RELEASE = re.compile(r"release (\S+), torchcell (\S+)")
RUN_URL = re.compile(r"https://wandb\.ai/zhao-group/tcdb/runs/[a-z0-9]+")


def adapter_times(generate_log: str) -> list[dict[str, Any]]:
    """One row per adapter from consecutive log timestamps."""
    events: list[tuple[dt.datetime, str, str | None]] = []
    with open(generate_log, errors="replace") as fh:
        for line in fh:
            m = EVENT.search(line)
            if m:
                kind = m.group(2) or m.group(4) or m.group(5)
                events.append((dt.datetime.fromisoformat(m.group(1)), kind, m.group(3)))
    per: dict[str, dict[str, float]] = {}
    for (t, kind, adapter), (t_next, _, _) in zip(events, events[1:], strict=False):
        if adapter is not None:
            per.setdefault(adapter, {})[kind] = (t_next - t).total_seconds()
    rows: list[dict[str, Any]] = [
        {
            "adapter": adapter,
            "nodes_s": times.get("nodes", 0.0),
            "edges_s": times.get("edges", 0.0),
            "total_s": times.get("nodes", 0.0) + times.get("edges", 0.0),
        }
        for adapter, times in per.items()
    ]
    rows.sort(key=lambda r: -float(r["total_s"]))
    return rows


def wandb_telemetry(run_id: str) -> dict[str, float]:
    """Peak anonymous memory and mean cores from the run's 5 s samples."""
    import wandb

    load_dotenv("/home/michaelvolk/Documents/projects/torchcell/.env")
    run = wandb.Api().run(f"zhao-group/tcdb/{run_id}")
    history: Any = run.history(samples=100_000, pandas=True)
    return {
        "anon_peak_gb": float(history["res/anon_gb"].max()),
        "mem_peak_gb": float(history["res/mem_gb"].max()),
        "mem_mean_gb": float(history["res/mem_gb"].mean()),
        "cores_mean": float(history["res/cpu_cores"].mean()),
        "cores_max": float(history["res/cpu_cores"].max()),
        "cpu_core_s": float(run.summary["generation_cpu_core_s"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="slurm job id of the live rebuild")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()
    generate_log = osp.join(args.output_dir, f"{args.job}_generate.log")
    job_log = osp.join(args.output_dir, f"{args.job}_live_rebuild_kg.out")

    rows = adapter_times(generate_log)
    with open(generate_log, errors="replace") as fh:
        generate = fh.read()
    with open(job_log, errors="replace") as fh:
        job = fh.read()
    wall = WALL.search(generate)
    imp = IMPORT.search(job)
    counts = COUNTS.findall(generate)
    release = RELEASE.search(job)
    run_url = RUN_URL.findall(job)
    assert wall and imp and counts and release and run_url, "log lacks a figure"
    telemetry = wandb_telemetry(run_url[-1].rsplit("/", 1)[1])
    summary = {
        "job": args.job,
        "adapters": len(rows),
        "generation_wall_s": int(wall.group(1)),
        "cgroup_peak_gb": float(wall.group(2)),
        "import_s": int(imp.group(1)) * 60 + int(imp.group(2)),
        "nodes": int(counts[-1][0]),
        "edges": int(counts[-1][1]),
        "release": release.group(1),
        "torchcell": release.group(2),
        "wandb_run": run_url[-1],
        **{k: round(v, 1) for k, v in telemetry.items()},
    }
    adapters_out = osp.join(RESULTS_DIR, f"live_rebuild_{args.job}.csv")
    with open(adapters_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary_out = osp.join(RESULTS_DIR, f"live_rebuild_{args.job}_summary.csv")
    with open(summary_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    for row in rows[:8]:
        print(
            f"{row['adapter']:36s} {row['total_s'] / 60:6.1f} min "
            f"(nodes {row['nodes_s'] / 60:.1f}, edges {row['edges_s'] / 60:.1f})"
        )
    print(summary)
    print(adapters_out)
    print(summary_out)


if __name__ == "__main__":
    main()
