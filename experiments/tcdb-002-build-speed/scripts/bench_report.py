# experiments/tcdb-002-build-speed/scripts/bench_report.py
# [[experiments.tcdb-002-build-speed.scripts.bench_report]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/tcdb-002-build-speed/scripts/bench_report
"""Score every benchmark arm from its run directory and write the results tables.

Reads ``$BENCH_ROOT/runs/<job>_<round>_<arm>/`` (arm.tsv, telemetry/phase_timings.json,
telemetry/resource_samples.csv, csv_inventory.tsv, csv_total_bytes) and the slurm
outputs ``$BENCH_ROOT/slurm/<job>_tcdb002-<round>-<arm>.out``, and writes to
``experiments/tcdb-002-build-speed/results/``:

* ``arms.csv`` / ``arms.md``  one row per finished arm: wall, core-seconds, peak memory,
  rows, bytes, and CPU utilization (mean cores, share of samples per core band)
* ``arm_adapters.csv``    one row per (arm, adapter): wall, core-seconds, peak memory
* ``arm_adapter_cores.csv``  mean sampled cores per adapter (rows) and arm (columns)
* ``arm_methods.csv``     one row per (arm, adapter, method)
* ``speedup_vs_baseline.csv``  per adapter, each arm's wall over the round-0 baseline's
* ``specs.csv`` / ``specs.md``  one row per submitted job, failed ones included: box
  (cpus, memory), the parsed batching knobs, wall, peak memory, mean cores, state

Scoring rules. ``wall_s`` is the sum of the setup + node + edge phase durations from
phase_timings.json (generation only, no container or pip setup); ``slurm_elapsed_s`` is
slurm's ElapsedRaw for the whole job. Utilization statistics are over the cgroup
samples in resource_samples.csv (one sample every ~5 s), so a fraction is a share of
samples, which approximates a share of wall time. The job state comes from ``sacct``
at report time: sacct.txt in the run dir is captured while the job is still running
and always says RUNNING.

The W&B run for each arm is found by its job id tag through the API and its URL is
recorded, so the table links back to the run.
"""

from __future__ import annotations

import io
import json
import os
import os.path as osp
import re
import shlex
import subprocess
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
# Hydra overrides under ``adapters.`` that set the batching scheme; each gets a column.
KNOBS = (
    "inprocess_max_records",
    "single_pass",
    "fast_writer",
    "io_to_total_worker_ratio",
    "single_pass_chunk_budget_mb",
    "chunks_per_worker",
    "completion_order",
    "inprocess_max_mb",
)
OUT_NAME = re.compile(r"^(\d+)_tcdb002-.+\.out$")
CONFIG_NAME = re.compile(r"--config-name (\S+)")


class Utilization(BaseModel):
    """CPU and memory use of one arm over its cgroup samples."""

    mean_cores: float
    frac_lt4: float
    frac_4_16: float
    frac_16_32: float
    frac_ge32: float
    # None for arms before job 2874, whose telemetry did not sample the parent process.
    mean_parent_cores: float | None
    peak_mem_gb: float
    n_samples: int


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
    mean_cores: float
    frac_lt4: float
    frac_4_16: float
    frac_16_32: float
    frac_ge32: float
    mean_parent_cores: float | None


class ArmSpec(BaseModel):
    """One submitted job: its box, batching knobs and outcome; blank when unmeasured."""

    job: int
    round: str
    arm: str
    commit: str
    kg_config: str
    cpus: int
    mem_gb: float
    inprocess_max_records: str = ""
    single_pass: str = ""
    fast_writer: str = ""
    io_to_total_worker_ratio: str = ""
    single_pass_chunk_budget_mb: str = ""
    chunks_per_worker: str = ""
    completion_order: str = ""
    inprocess_max_mb: str = ""
    other_overrides: str = ""
    wall_s: float | None = None
    slurm_elapsed_s: float | None = None
    peak_mem_gb: float | None = None
    mean_cores: float | None = None
    rows: int | None = None
    csv_bytes: int | None = None
    state: str


def read_tsv_kv(path: Path) -> dict[str, str]:
    """Read a two-column key/value TSV into a dict."""
    return dict(
        line.rstrip("\n").split("\t", 1)
        for line in path.read_text().splitlines()
        if line
    )


def read_out_header(path: Path) -> dict[str, str]:
    """Read the arm description from a slurm output's first line and its config name.

    The first line is ``arm=.. round=.. commit=.. job=.. cpus=.. mem_mb=.. wheel=..
    overrides='..'``. The hydra config name is not echoed; it appears only when the
    shell reports the generation command (a killed run), otherwise it stays blank.
    """
    with open(path) as handle:
        fields = dict(tok.split("=", 1) for tok in shlex.split(handle.readline()))
        fields["kg_config"] = ""
        for line in handle:
            match = CONFIG_NAME.search(line)
            if match:
                fields["kg_config"] = match.group(1)
                break
    # Job 2851 (the first baseline, no overrides) predates the overrides field.
    fields.setdefault("overrides", "")
    return fields


def read_samples(run_dir: Path) -> pd.DataFrame:
    """Read the cgroup resource samples (CRLF line endings) of one arm."""
    text = (run_dir / "telemetry" / "resource_samples.csv").read_text()
    samples = pd.read_csv(io.StringIO(text.replace("\r", "")))
    samples["adapter"] = samples.adapter.fillna("")
    return samples


def utilization(samples: pd.DataFrame) -> Utilization:
    """Summarize CPU use as mean cores and the share of samples in each core band."""
    cores = samples.cpu_cores
    return Utilization(
        mean_cores=float(cores.mean()),
        frac_lt4=float((cores < 4).mean()),
        frac_4_16=float(((cores >= 4) & (cores < 16)).mean()),
        frac_16_32=float(((cores >= 16) & (cores < 32)).mean()),
        frac_ge32=float((cores >= 32).mean()),
        mean_parent_cores=float(samples.parent_cores.mean())
        if "parent_cores" in samples.columns
        else None,
        peak_mem_gb=float(samples.mem_peak_gb.max()),
        n_samples=len(samples),
    )


def phase_wall(phases: pd.DataFrame) -> float:
    """Generation wall time: the summed setup + node + edge phase durations."""
    return float(
        phases[phases.phase_kind.isin(["setup", "node", "edge"])].seconds.sum()
    )


def slurm_states(jobs: list[int]) -> dict[int, tuple[str, float]]:
    """Map job -> (final state, elapsed seconds) from slurm accounting."""
    stdout = subprocess.run(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            ",".join(str(j) for j in jobs),
            "-o",
            "JobID,State,ElapsedRaw",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    states: dict[int, tuple[str, float]] = {}
    for line in stdout.splitlines():
        job, state, elapsed = line.split("|")
        states[int(job)] = (state.split()[0], float(elapsed))
    return states


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
    run_dir: Path, urls: dict[str, str], state: str
) -> tuple[ArmRecord, pd.DataFrame, pd.DataFrame]:
    """Score one arm directory; returns the arm record, its phases and its samples."""
    arm = read_tsv_kv(run_dir / "arm.tsv")
    phases = pd.DataFrame(
        json.loads((run_dir / "telemetry" / "phase_timings.json").read_text())
    )
    inventory = pd.read_csv(
        run_dir / "csv_inventory.tsv", sep="\t", names=["file", "bytes", "rows"]
    )
    samples = read_samples(run_dir)
    use = utilization(samples)
    build = phases[phases.phase_kind.isin(["node", "edge"])]
    setup = phases[phases.phase_kind == "setup"]
    record = ArmRecord(
        job=int(arm["job"]),
        round=arm["round"],
        arm=arm["arm"],
        commit=arm["commit"],
        cpus=int(arm["cpus"]),
        mem_mb=int(arm["mem_mb"]),
        wall_s=phase_wall(phases),
        setup_s=float(setup.seconds.sum()),
        core_s=float(phases.cpu_core_seconds.sum()),
        mem_peak_gb=float(phases.mem_peak_gb.max()),
        n_adapters=int(build.adapter.nunique()),
        # Part files carry no header line (headers are separate *-header.csv files,
        # which wc -l counts as 0 lines), so the data rows are the part files' lines.
        csv_rows=int(inventory[inventory.file.str.contains("-part")].rows.sum()),
        csv_bytes=int((run_dir / "csv_total_bytes").read_text().strip()),
        wandb_url=urls.get(arm["job"])
        or urls.get(f"{arm['round']}/{arm['arm']}/{arm['commit']}", ""),
        state=state,
        mean_cores=use.mean_cores,
        frac_lt4=use.frac_lt4,
        frac_4_16=use.frac_4_16,
        frac_16_32=use.frac_16_32,
        frac_ge32=use.frac_ge32,
        mean_parent_cores=use.mean_parent_cores,
    )
    for frame in (phases, samples):
        frame.insert(0, "arm", record.arm)
        frame.insert(0, "round", record.round)
        frame.insert(0, "job", record.job)
    return record, phases, samples


def parse_overrides(overrides: str) -> dict[str, str]:
    """Split hydra overrides into the knob columns and the remaining overrides."""
    parsed: dict[str, str] = {}
    other: list[str] = []
    for token in overrides.split():
        key, value = token.lstrip("+").split("=", 1)
        name = key.removeprefix("adapters.")
        if key.startswith("adapters.") and name in KNOBS:
            parsed[name] = value
        else:
            other.append(token)
    parsed["other_overrides"] = " ".join(other)
    return parsed


def spec_for_job(
    job: int, run_dir: Path | None, out_file: Path | None, state: tuple[str, float]
) -> ArmSpec:
    """Build one job's spec row from its run dir when it has one, else its slurm out."""
    if run_dir is not None and (run_dir / "arm.tsv").exists():
        arm = read_tsv_kv(run_dir / "arm.tsv")
        # Job 2851 (the first baseline, no overrides) predates the overrides field.
        arm.setdefault("overrides", "")
    else:
        assert out_file is not None, f"job {job} has neither arm.tsv nor a slurm out"
        arm = read_out_header(out_file)
    telemetry = None if run_dir is None else run_dir / "telemetry"
    phases_path = None if telemetry is None else telemetry / "phase_timings.json"
    samples_path = None if telemetry is None else telemetry / "resource_samples.csv"
    wall = None
    if phases_path is not None and phases_path.exists():
        wall = phase_wall(pd.DataFrame(json.loads(phases_path.read_text())))
    use = None
    if run_dir is not None and samples_path is not None and samples_path.exists():
        use = utilization(read_samples(run_dir))
    rows = None
    csv_bytes = None
    if run_dir is not None and (run_dir / "csv_inventory.tsv").exists():
        inventory = pd.read_csv(
            run_dir / "csv_inventory.tsv", sep="\t", names=["file", "bytes", "rows"]
        )
        rows = int(inventory[inventory.file.str.contains("-part")].rows.sum())
        csv_bytes = int((run_dir / "csv_total_bytes").read_text().strip())
    return ArmSpec(
        job=job,
        round=arm["round"],
        arm=arm["arm"],
        commit=arm["commit"],
        kg_config=arm["kg_config"],
        cpus=int(arm["cpus"]),
        mem_gb=int(arm["mem_mb"]) / 1024,
        **parse_overrides(arm["overrides"]),
        wall_s=wall,
        slurm_elapsed_s=state[1],
        peak_mem_gb=None if use is None else use.peak_mem_gb,
        mean_cores=None if use is None else use.mean_cores,
        rows=rows,
        csv_bytes=csv_bytes,
        state=state[0],
    )


def write_markdown(
    frame: pd.DataFrame, columns: list[str], formats: dict[str, str], path: Path
) -> None:
    """Write a markdown table; floats use ``formats`` or ``,.0f``, missing is blank."""
    with open(path, "w") as handle:
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write(
            "|"
            + "|".join("--:" if frame[c].dtype.kind in "if" else "---" for c in columns)
            + "|\n"
        )
        for _, row in frame[columns].iterrows():
            cells = [
                ""
                if pd.isna(row[c])
                else format(row[c], formats.get(c, ",.0f"))
                if isinstance(row[c], float)
                else str(row[c])
                for c in columns
            ]
            handle.write("| " + " | ".join(cells) + " |\n")


def main() -> None:
    """Score every finished arm, tabulate every submitted job, write the tables."""
    load_dotenv()
    RESULTS.mkdir(parents=True, exist_ok=True)
    run_dirs = {int(d.name.split("_")[0]): d for d in BENCH_ROOT.glob("runs/*")}
    out_files = {
        int(match.group(1)): f
        for f in BENCH_ROOT.glob("slurm/*.out")
        if (match := OUT_NAME.match(f.name))
    }
    jobs = sorted(set(run_dirs) | set(out_files))
    states = slurm_states(jobs)
    urls = wandb_urls_by_job()
    records: list[ArmRecord] = []
    phase_tables: list[pd.DataFrame] = []
    sample_tables: list[pd.DataFrame] = []
    for job, run_dir in sorted(run_dirs.items()):
        if (
            not (run_dir / "telemetry" / "phase_timings.json").exists()
            or not (run_dir / "arm.tsv").exists()
        ):
            print(f"skip (unfinished or not a scoring arm): {run_dir.name}")
            continue
        record, phases, samples = score_run_dir(run_dir, urls, states[job][0])
        records.append(record)
        phase_tables.append(phases)
        sample_tables.append(samples)
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
    samples_all = pd.concat(sample_tables, ignore_index=True)
    samples_all = samples_all[samples_all.adapter != ""]
    samples_all["arm_key"] = (
        samples_all.job.astype(str) + "_" + samples_all["round"] + "_" + samples_all.arm
    )
    adapter_cores = samples_all.pivot_table(
        index="adapter",
        columns="arm_key",
        values="cpu_cores",
        aggfunc="mean",
        sort=False,
    )
    adapter_cores.to_csv(RESULTS / "arm_adapter_cores.csv")
    baseline = adapters[(adapters["round"] == "r0") & (adapters.arm == "baseline")]
    if not baseline.empty:
        base = (
            baseline.drop_duplicates("adapter", keep="last").set_index("adapter").wall_s
        )
        speed = adapters.assign(baseline_wall_s=adapters.adapter.map(base))
        speed["speedup"] = speed.baseline_wall_s / speed.wall_s
        speed.to_csv(RESULTS / "speedup_vs_baseline.csv", index=False)
    write_markdown(
        arms,
        [
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
            "mean_cores",
            "frac_lt4",
            "frac_4_16",
            "frac_16_32",
            "frac_ge32",
            "mean_parent_cores",
        ],
        {
            "mean_cores": ".1f",
            "mean_parent_cores": ".2f",
            "frac_lt4": ".2f",
            "frac_4_16": ".2f",
            "frac_16_32": ".2f",
            "frac_ge32": ".2f",
        },
        RESULTS / "arms.md",
    )
    specs = pd.DataFrame(
        [
            spec_for_job(
                job, run_dirs.get(job), out_files.get(job), states[job]
            ).model_dump()
            for job in jobs
        ]
    )
    specs[["rows", "csv_bytes"]] = specs[["rows", "csv_bytes"]].astype("Int64")
    specs.to_csv(RESULTS / "specs.csv", index=False)
    write_markdown(
        specs.sort_values(["kg_config", "wall_s"], na_position="last"),
        [c for c in specs.columns if c != "other_overrides"] + ["other_overrides"],
        {
            "mem_gb": ".0f",
            "wall_s": ",.1f",
            "slurm_elapsed_s": ",.0f",
            "peak_mem_gb": ".1f",
            "mean_cores": ".1f",
        },
        RESULTS / "specs.md",
    )
    print(arms.to_string(index=False))
    print(
        adapters.pivot_table(
            index="adapter", columns=["round", "arm"], values="wall_s", sort=False
        )
        .round(0)
        .to_string()
    )
    print(specs.to_string(index=False))


if __name__ == "__main__":
    main()
