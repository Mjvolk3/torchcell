# experiments/tcdb-002-build-speed/scripts/full_build_compare.py
# [[experiments.tcdb-002-build-speed.scripts.full_build_compare]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/tcdb-002-build-speed/scripts/full_build_compare
"""Per-adapter wall of a full-build arm against job 2032, from the arm's telemetry.

    python experiments/tcdb-002-build-speed/scripts/full_build_compare.py <run dir>

Works on a finished arm (phase_timings.json) or a killed one (resource_samples.csv
only, where an adapter's wall is the span of its samples). Writes
``results/<job>_vs_2032.csv`` and prints the table plus the totals over the adapters
both builds completed.
"""

import json
import os.path as osp
import sys
from pathlib import Path

import pandas as pd

RESULTS = Path(osp.dirname(osp.dirname(osp.abspath(__file__)))) / "results"


def adapter_walls(run_dir: Path) -> pd.Series:
    """Wall seconds per adapter, from the phase table when present, else the samples."""
    phases = run_dir / "telemetry" / "phase_timings.json"
    if phases.exists():
        p = pd.DataFrame(json.loads(phases.read_text()))
        p = p[p.phase_kind.isin(["node", "edge"])]
        return p.groupby("adapter", sort=False).seconds.sum()
    s = pd.read_csv(run_dir / "telemetry" / "resource_samples.csv")
    s = s[s.phase_kind.isin(["node", "edge"])]
    return s.groupby("adapter", sort=False).t.agg(lambda t: t.max() - t.min())


def main(run_dir: str) -> None:
    """Compare and write the table."""
    walls = adapter_walls(Path(run_dir)).rename("wall_s")
    base = pd.read_csv(RESULTS / "build2032_per_adapter.csv").set_index("adapter")
    table = pd.DataFrame({"wall_s": walls, "job2032_s": base.total_sec.reindex(walls.index)})
    table["speedup"] = table.job2032_s / table.wall_s
    job = osp.basename(run_dir.rstrip("/")).split("_")[0]
    table.round(1).to_csv(RESULTS / f"{job}_vs_2032.csv")
    done = table.dropna()
    print(table.round(1).to_string())
    print(
        f"\n{len(done)} adapters: this arm {done.wall_s.sum() / 3600:.2f} h, "
        f"job 2032 {done.job2032_s.sum() / 3600:.2f} h, "
        f"{done.job2032_s.sum() / done.wall_s.sum():.1f}x"
    )


if __name__ == "__main__":
    main(sys.argv[1])
