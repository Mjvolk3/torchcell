# experiments/database/scripts/gen_image_startup_measurement.py
# [[experiments.database.scripts.gen_image_startup_measurement]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/gen_image_startup_measurement
"""Read a gen_image_startup_measurement.slurm job's ``.out`` and report what the
prebuilt generation image saves per array task.

Every container line in that log is prefixed with the host's UTC epoch; the markers
(``T0_RUN_START``, ``CONTAINER_SHELL_UP``, ``INSTALL_DONE`` or ``COMMIT_VERIFIED``,
``Generating adapter``, ``DONE=``, ``T1_RUN_END``) bound the intervals. Startup is
``Generating adapter`` minus ``T0_RUN_START``: from the host issuing ``docker run`` to
the generator's first own log line. The projection charges the array's 170 tasks with
the per-task saving and divides by the concurrency for the wall clock.

    python experiments/database/scripts/gen_image_startup_measurement.py \
        --out /scratch/projects/torchcell/database/slurm/output/<job>_gen-image-measure.out \
        --job <job> --baseline-seconds 278

writes ``experiments/database/results/gen_image_startup_<job>.json`` and ``.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

STAMPED = re.compile(r"^(\d+\.\d+) (.*)$")
RUN_START = re.compile(r"T0_RUN_START run=(\S+) mode=(\S+) image=(\S+)")
BUILD_SECONDS = re.compile(r"^(MAIN|GEN)_IMAGE_BUILD_SECONDS=(\d+)$")
ARRAY_TASKS = 170


class RunTiming(BaseModel):
    """One container's intervals, seconds from ``T0_RUN_START``."""

    run: str
    mode: str
    image: str
    shell_up_s: float
    ready_s: float | None
    startup_s: float
    done_s: float | None
    total_s: float | None

    @property
    def python_to_generator_s(self) -> float | None:
        """From the install or the commit check to the generator's first line."""
        return None if self.ready_s is None else self.startup_s - self.ready_s


class Projection(BaseModel):
    """What the image saves across the array against one baseline."""

    baseline_label: str
    baseline_startup_s: float
    image_startup_mean_s: float
    saving_per_task_s: float
    tasks: int
    concurrency: int
    cpu_hours_saved: float
    wall_hours_saved: float


class MeasurementReport(BaseModel):
    """The job's image build times, run timings and projections."""

    job: str
    image_build_seconds: dict[str, int]
    runs: list[RunTiming]
    projections: list[Projection]


def parse_runs(lines: list[str]) -> list[RunTiming]:
    """The run timings in the order the job ran them."""
    runs: list[RunTiming] = []
    current: dict[str, object] | None = None
    for raw in lines:
        match = STAMPED.match(raw)
        if match is None:
            continue
        t, text = float(match.group(1)), match.group(2)
        # the host's own marks are stamped twice (mark, then the stamper): drop the inner one
        inner = STAMPED.match(text)
        if inner is not None:
            text = inner.group(2)
        start = RUN_START.search(text)
        if start is not None:
            current = {
                "run": start.group(1),
                "mode": start.group(2),
                "image": start.group(3),
                "t0": t,
            }
            continue
        if current is None:
            continue
        if text.strip() == "CONTAINER_SHELL_UP":
            current["shell_up"] = t
        elif text.strip() in ("INSTALL_DONE", "COMMIT_VERIFIED"):
            current["ready"] = t
        elif text.startswith("Generating adapter") and "generating" not in current:
            current["generating"] = t
        elif text.startswith("DONE="):
            current["done"] = t
        elif text.startswith("T1_RUN_END"):
            t0 = float(str(current["t0"]))
            runs.append(
                RunTiming(
                    run=str(current["run"]),
                    mode=str(current["mode"]),
                    image=str(current["image"]),
                    shell_up_s=float(str(current["shell_up"])) - t0,
                    ready_s=(
                        float(str(current["ready"])) - t0
                        if "ready" in current
                        else None
                    ),
                    startup_s=float(str(current["generating"])) - t0,
                    done_s=(
                        float(str(current["done"])) - t0 if "done" in current else None
                    ),
                    total_s=t - t0,
                )
            )
            current = None
    return runs


def parse_build_seconds(lines: list[str]) -> dict[str, int]:
    """``{"main": s, "gen": s}`` from the job's build lines."""
    found: dict[str, int] = {}
    for raw in lines:
        match = BUILD_SECONDS.match(raw.strip())
        if match is not None:
            found[match.group(1).lower()] = int(match.group(2))
    return found


def project(
    baseline_label: str,
    baseline_startup_s: float,
    image_runs: list[RunTiming],
    *,
    tasks: int = ARRAY_TASKS,
    concurrency: int = 4,
) -> Projection:
    """The saving per task times the tasks, and over the concurrency for wall clock."""
    mean = statistics.fmean(r.startup_s for r in image_runs)
    saving = baseline_startup_s - mean
    cpu_hours = tasks * saving / 3600
    return Projection(
        baseline_label=baseline_label,
        baseline_startup_s=baseline_startup_s,
        image_startup_mean_s=mean,
        saving_per_task_s=saving,
        tasks=tasks,
        concurrency=concurrency,
        cpu_hours_saved=cpu_hours,
        wall_hours_saved=cpu_hours / concurrency,
    )


def build_report(
    job: str, lines: list[str], *, baseline_seconds: float, concurrency: int
) -> MeasurementReport:
    """Parse the log and project against the in-job install run and the reference."""
    runs = parse_runs(lines)
    image_runs = [r for r in runs if r.mode == "image"]
    install_runs = [r for r in runs if r.mode == "install"]
    if not image_runs:
        raise ValueError("no completed image-mode run in the log")
    projections: list[Projection] = []
    if install_runs:
        projections.append(
            project(
                f"job {job} install run (same adapter, same job)",
                statistics.fmean(r.startup_s for r in install_runs),
                image_runs,
                concurrency=concurrency,
            )
        )
    projections.append(
        project(
            "job 4919_64 install path (lane C, 2026-10-15)",
            baseline_seconds,
            image_runs,
            concurrency=concurrency,
        )
    )
    return MeasurementReport(
        job=job,
        image_build_seconds=parse_build_seconds(lines),
        runs=runs,
        projections=projections,
    )


def render_markdown(report: MeasurementReport) -> str:
    """The results as two tables, for the dendron note."""
    out = [
        f"Job {report.job}; image build seconds: "
        + ", ".join(f"{k} {v}" for k, v in sorted(report.image_build_seconds.items())),
        "",
        "| run | mode | image | shell up (s) | installed / verified (s) | startup to first generator line (s) | DONE (s) | total (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def num(value: float | None) -> str:
        return "" if value is None else f"{value:.1f}"

    for r in report.runs:
        out.append(
            f"| {r.run} | {r.mode} | {r.image} | {num(r.shell_up_s)} | {num(r.ready_s)} "
            f"| {num(r.startup_s)} | {num(r.done_s)} | {num(r.total_s)} |"
        )
    out += [
        "",
        "| baseline | baseline startup (s) | image startup mean (s) | saving per task (s) | tasks | concurrency | CPU hours saved | wall hours saved |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in report.projections:
        out.append(
            f"| {p.baseline_label} | {p.baseline_startup_s:.1f} | {p.image_startup_mean_s:.1f} "
            f"| {p.saving_per_task_s:.1f} | {p.tasks} | {p.concurrency} "
            f"| {p.cpu_hours_saved:.2f} | {p.wall_hours_saved:.2f} |"
        )
    return "\n".join(out) + "\n"


def main() -> None:
    """Parse the job log; write the JSON and markdown results."""
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, type=Path, help="the job's .out")
    parser.add_argument("--job", required=True)
    parser.add_argument("--baseline-seconds", type=float, default=278.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(
            os.environ.get("EXPERIMENT_ROOT", str(Path(__file__).resolve().parents[2]))
        )
        / "database"
        / "results",
    )
    args = parser.parse_args()
    lines = args.out.read_text(encoding="utf-8", errors="replace").splitlines()
    report = build_report(
        args.job,
        lines,
        baseline_seconds=args.baseline_seconds,
        concurrency=args.concurrency,
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    stem = args.results_dir / f"gen_image_startup_{args.job}"
    stem.with_suffix(".json").write_text(
        json.dumps(report.model_dump(), indent=2) + "\n", encoding="utf-8"
    )
    markdown = render_markdown(report)
    stem.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"wrote {stem.with_suffix('.json')} and {stem.with_suffix('.md')}")


if __name__ == "__main__":
    main()
