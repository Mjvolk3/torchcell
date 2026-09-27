# experiments/tcdb-002-build-speed/scripts/full_build_compare.py
# [[experiments.tcdb-002-build-speed.scripts.full_build_compare]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/tcdb-002-build-speed/scripts/full_build_compare
"""Per-adapter wall of a full-build arm against job 2032, from the arm's telemetry.

    python experiments/tcdb-002-build-speed/scripts/full_build_compare.py <run dir> [--job ID]

Works on a finished arm (phase_timings.json) or a killed or running one
(resource_samples.csv only, where a phase's wall is the span of its samples). The
node and edge passes are split out, with the mean container cores over each, so two
arms can be compared pass by pass. Writes ``results/<job>_vs_2032.csv`` and prints
the table plus the totals over the adapters both builds completed. ``<run dir>`` is
``runs/<job>_<round>_<arm>`` or, for a running arm, its BioCypher output directory;
``--job`` names the output when the directory does not start with the job id.
"""

import argparse
import json
import os.path as osp
from pathlib import Path

import pandas as pd

RESULTS = Path(osp.dirname(osp.dirname(osp.abspath(__file__)))) / "results"
KINDS = ["node", "edge"]


def adapter_phases(run_dir: Path) -> pd.DataFrame:
    """Per adapter: wall_s, node_s, edge_s, node_cores, edge_cores."""
    phases = run_dir / "telemetry" / "phase_timings.json"
    samples = pd.read_csv(run_dir / "telemetry" / "resource_samples.csv")
    samples["phase_kind"] = samples.phase_kind.str.strip()
    samples = samples[samples.phase_kind.isin(KINDS)]
    cores = samples.groupby(["adapter", "phase_kind"], sort=False).cpu_cores.mean()
    if phases.exists():
        p = pd.DataFrame(json.loads(phases.read_text()))
        p = p[p.phase_kind.isin(KINDS)]
        walls = p.groupby(["adapter", "phase_kind"], sort=False).seconds.sum()
    else:
        walls = samples.groupby(["adapter", "phase_kind"], sort=False).t.agg(
            lambda t: t.max() - t.min()
        )
    table = pd.DataFrame(
        {
            "wall_s": walls.groupby(level=0, sort=False).sum(),
            **{f"{k}_s": walls.xs(k, level=1) for k in KINDS},
            **{f"{k}_cores": cores.xs(k, level=1) for k in KINDS},
        }
    )
    order = [a for a in samples.adapter.unique() if a in table.index]
    return table.reindex(order)


def main(run_dir: str, job: str | None) -> None:
    """Compare and write the table."""
    table = adapter_phases(Path(run_dir))
    base = pd.read_csv(RESULTS / "build2032_per_adapter.csv").set_index("adapter")
    table["job2032_s"] = base.total_sec.reindex(table.index)
    table["speedup"] = table.job2032_s / table.wall_s
    job = job or osp.basename(run_dir.rstrip("/")).split("_")[0]
    table.round(1).to_csv(RESULTS / f"{job}_vs_2032.csv")
    done = table.dropna(subset=["job2032_s"])
    print(table.round(1).to_string())
    print(
        f"\n{len(done)} adapters: this arm {done.wall_s.sum() / 3600:.2f} h, "
        f"job 2032 {done.job2032_s.sum() / 3600:.2f} h, "
        f"{done.job2032_s.sum() / done.wall_s.sum():.1f}x"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--job", default=None)
    args = ap.parse_args()
    main(args.run_dir, args.job)
