# experiments/tcdb-002-build-speed/scripts/bench_report.py
# [[experiments.tcdb-002-build-speed.scripts.bench_report]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/tcdb-002-build-speed/scripts/bench_report
"""Score every benchmark arm from its run directory and write the results tables.

Reads ``$BENCH_ROOT/runs/<job>_<round>_<arm>/`` (arm.tsv, telemetry/phase_timings.json,
telemetry/resource_samples.csv, csv_inventory.tsv, csv_total_bytes, sacct.txt) and writes
to ``experiments/tcdb-002-build-speed/results/``:

* ``arms.csv``            one row per arm: wall, core-seconds, peak memory, rows, bytes
* ``arm_adapters.csv``    one row per (arm, adapter): wall, core-seconds, peak memory
* ``arm_methods.csv``     one row per (arm, adapter, method)
* ``speedup_vs_baseline.csv``  per adapter, each arm's wall over the round-0 baseline's

The W&B run for each arm is found by its job id tag through the API and its URL is
recorded, so the table links back to the run.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from pathlib import Path

import pandas as pd
import wandb
from dotenv import load_dotenv
from pydantic import BaseModel

BENCH_ROOT = Path(
    os.environ.get("BENCH_ROOT", "/scratch/projects/torchcell-scratch/tcdb-002-bench")
)
RESULTS = Path(osp.dirname(osp.dirname(osp.abspath(__file__)))) / "results"
WANDB_PROJECT = "zhao-group/tcdb"


class ArmRecord(BaseModel):
    """Score of one benchmark arm."""

    job: int
    round: str
    arm: str
    commit: str
    cpus: int
    mem_mb: int
    wall_s: float
    setup_s: float
    core_s: float
    mem_peak_gb: float
    n_adapters: int
    csv_rows: int
    csv_bytes: int
    wandb_url: str = ""
    state: str = ""


def read_tsv_kv(path: Path) -> dict[str, str]:
    """Read a two-column key/value TSV into a dict."""
    return dict(
        line.rstrip("\n").split("\t", 1)
        for line in path.read_text().splitlines()
        if line
    )


def wandb_urls_by_job() -> dict[str, str]:
    """Map arm key -> W&B run URL for every tcdb-002 tagged run.

    Keys are the slurm job id when the run logged one (arms after job 2859 pass
    SLURM_JOB_ID into the container) and ``round/arm/commit`` from the tags always.
    """
    api = wandb.Api()
    urls: dict[str, str] = {}
    for run in api.runs(WANDB_PROJECT, filters={"tags": {"$in": ["tcdb-002"]}}):
        tags = set(run.tags)
        job = str(run.summary.get("slurm_job_id", ""))
        if job.isdigit():
            urls[job] = run.url
        commit = next((t[7:] for t in tags if t.startswith("commit-")), "")
        rnd = next((t for t in tags if t in {f"r{i}" for i in range(10)}), "")
        arm = next(
            (
                t
                for t in tags
                if t not in {"tcdb-002", "ladder", rnd}
                and not t.startswith(("commit-", "cpus-", "mem-"))
            ),
            "",
        )
        urls[f"{rnd}/{arm}/{commit}"] = run.url
    return urls


def score_run_dir(
    run_dir: Path, urls: dict[str, str]
) -> tuple[ArmRecord, pd.DataFrame]:
    """Score one arm directory; returns the arm record and its phase table."""
    arm = read_tsv_kv(run_dir / "arm.tsv")
    phases = pd.DataFrame(
        json.loads((run_dir / "telemetry" / "phase_timings.json").read_text())
    )
    inventory = pd.read_csv(
        run_dir / "csv_inventory.tsv", sep="\t", names=["file", "bytes", "rows"]
    )
    # sacct is read while the job's own script is still running, so the job line says
    # RUNNING; a finished arm is one whose phase table exists, which got us here.
    state = "COMPLETED"
    build = phases[phases.phase_kind.isin(["node", "edge"])]
    setup = phases[phases.phase_kind == "setup"]
    record = ArmRecord(
        job=int(arm["job"]),
        round=arm["round"],
        arm=arm["arm"],
        commit=arm["commit"],
        cpus=int(arm["cpus"]),
        mem_mb=int(arm["mem_mb"]),
        wall_s=float(build.seconds.sum() + setup.seconds.sum()),
        setup_s=float(setup.seconds.sum()),
        core_s=float(phases.cpu_core_seconds.sum()),
        mem_peak_gb=float(phases.mem_peak_gb.max()),
        n_adapters=int(build.adapter.nunique()),
        csv_rows=int(inventory.rows.sum() - len(inventory)),  # one header line per file
        csv_bytes=int((run_dir / "csv_total_bytes").read_text().strip()),
        wandb_url=urls.get(arm["job"])
        or urls.get(f"{arm['round']}/{arm['arm']}/{arm['commit']}", ""),
        state=state,
    )
    phases.insert(0, "arm", record.arm)
    phases.insert(0, "round", record.round)
    phases.insert(0, "job", record.job)
    return record, phases


def main() -> None:
    """Score every finished arm and write the results tables."""
    load_dotenv()
    RESULTS.mkdir(parents=True, exist_ok=True)
    urls = wandb_urls_by_job()
    records: list[ArmRecord] = []
    phase_tables: list[pd.DataFrame] = []
    for run_dir in sorted(BENCH_ROOT.glob("runs/*")):
        if (
            not (run_dir / "telemetry" / "phase_timings.json").exists()
            or not (run_dir / "arm.tsv").exists()
        ):
            print(f"skip (unfinished or not a scoring arm): {run_dir.name}")
            continue
        record, phases = score_run_dir(run_dir, urls)
        records.append(record)
        phase_tables.append(phases)
    arms = pd.DataFrame([r.model_dump() for r in records]).sort_values(["round", "job"])
    arms.to_csv(RESULTS / "arms.csv", index=False)
    methods = pd.concat(phase_tables, ignore_index=True)
    methods = methods[methods.phase_kind.isin(["node", "edge"])]
    methods.to_csv(RESULTS / "arm_methods.csv", index=False)
    adapters = (
        methods.groupby(["job", "round", "arm", "adapter"], sort=False)
        .agg(
            wall_s=("seconds", "sum"),
            core_s=("cpu_core_seconds", "sum"),
            mem_peak_gb=("mem_peak_gb", "max"),
            n_methods=("method", "count"),
        )
        .reset_index()
    )
    adapters.to_csv(RESULTS / "arm_adapters.csv", index=False)
    baseline = adapters[(adapters["round"] == "r0") & (adapters.arm == "baseline")]
    if not baseline.empty:
        base = (
            baseline.drop_duplicates("adapter", keep="last").set_index("adapter").wall_s
        )
        speed = adapters.assign(baseline_wall_s=adapters.adapter.map(base))
        speed["speedup"] = speed.baseline_wall_s / speed.wall_s
        speed.to_csv(RESULTS / "speedup_vs_baseline.csv", index=False)
    columns = [
        "job",
        "round",
        "arm",
        "commit",
        "cpus",
        "mem_mb",
        "wall_s",
        "core_s",
        "mem_peak_gb",
        "csv_rows",
        "csv_bytes",
        "state",
    ]
    with open(RESULTS / "arms.md", "w") as handle:
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write(
            "|"
            + "|".join("--:" if arms[c].dtype.kind in "if" else "---" for c in columns)
            + "|\n"
        )
        for _, row in arms[columns].iterrows():
            cells = [
                f"{row[c]:,.0f}" if isinstance(row[c], float) else str(row[c])
                for c in columns
            ]
            handle.write("| " + " | ".join(cells) + " |\n")
    print(arms.to_string(index=False))
    print(
        adapters.pivot_table(
            index="adapter", columns=["round", "arm"], values="wall_s", sort=False
        )
        .round(0)
        .to_string()
    )


if __name__ == "__main__":
    main()
