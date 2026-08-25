# experiments/019-simb-multimodal/scripts/analyze_energy_rank_ablation.py
# [[experiments.019-simb-multimodal.scripts.analyze_energy_rank_ablation]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/analyze_energy_rank_ablation
"""Judge the _007 JOINT ablation (energy_rank 0 vs 32) on a PROPER SCORING RULE.

THE POINT. The sweep ranks trials on `val/expression/pearson_per_feature`, computed from
`DistHead.point()` = mu. The energy head's global factor V enters only through

    Sigma = diag(sigma^2) + V V^T

which never touches mu. So `energy_rank` is STRUCTURALLY INVISIBLE to the ranked metric --
a null there is what the mathematics predicts, not a verdict on the joint head. The quantity
that can see V is the energy score itself, logged as `val/loss` for `dist=energy` runs.

Comparison is restricted to `dist == energy`: `val/loss` is a different scoring rule in each
distributional arm, so it is comparable WITHIN an arm and meaningless across arms.

Reads W&B (the sweep runs on IGB write offline; sync first), and reports:
  * energy score at rank 0 vs 32 -- the honest joint test, LOWER IS BETTER
  * pearson at rank 0 vs 32 -- expected to be a null, shown to make that explicit
  * calibration at each rank
  * a Mann-Whitney U test, because n per rank is small and the distributions are skewed

PROVENANCE CAVEAT -- READ BEFORE QUOTING ANY NUMBER FROM THIS.
The `_007` round these runs come from was declared VOID: its 295 runs have a median final
epoch of 67, because early stopping was watching a monitor measured to move OPPOSITE the
ranked metric on this task (val/loss bottoms ~epoch 103-136; val Pearson climbs to a
project-best 0.1980 at epoch 1367). Every run compared here was therefore cut off in the dip,
roughly 1,200 epochs before the model this architecture actually reaches. The rank-0 vs
rank-32 contrast below is a comparison of TRUNCATED models: it is directionally suggestive
and structurally sound as a method, but it is NOT a verdict on the joint head. The `epoch`
column is carried into the output so that truncation is visible in the data rather than
asserted in prose. Re-run this against a round trained to saturation before citing it.

Run from repo root (no GPU):
    python experiments/019-simb-multimodal/scripts/analyze_energy_rank_ablation.py
"""

from __future__ import annotations

import json
import os.path as osp
import statistics as st
from typing import Any

from dotenv import load_dotenv

_WT_ENV = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", "..", ".env"))
load_dotenv(
    _WT_ENV
    if osp.exists(_WT_ENV)
    else osp.expanduser("~/Documents/projects/torchcell/.env")
)

import wandb  # noqa: E402
from scipy import stats  # noqa: E402

from torchcell.utils.paths import experiment_results_dir  # noqa: E402

PROJECT = "zhao-group/torchcell_019_expr_v7"
PEARSON = "val/expression/pearson_per_feature"
LOSS = "val/loss"
CALIB = ["val/expression/calib/coverage_50", "val/expression/calib/coverage_80",
         "val/expression/calib/pit_ks"]


def _fetch() -> list[dict[str, Any]]:
    api = wandb.Api(timeout=60)
    out = []
    for r in api.runs(PROJECT):
        if r.state == "running":
            continue
        cfg, s = r.config, dict(r.summary)
        if cfg.get("dist") != "energy":
            continue
        # The peak pearson is what the sweep ranks on; the loss is read at the LAST logged
        # value, since a run that early-stops has already passed its best epoch.
        row = {
            "name": r.name,
            "rank": cfg.get("energy_rank"),
            "graph_reg_on": cfg.get("graph_reg_lambda", 0) > 0,
            "emb": cfg.get("node_embeddings"),
            "lr": cfg.get("lr"),
            "pearson": s.get(PEARSON),
            "loss": s.get(LOSS),
            # The epoch the run ENDED on. Carried so the round's truncation is visible in the
            # artifact: see the provenance caveat above -- these runs stopped at a median of
            # ~67 epochs against a metric that peaks near epoch 1367.
            "epoch": s.get("epoch"),
        }
        for k in CALIB:
            row[k.rsplit("/", 1)[-1]] = s.get(k)
        if row["rank"] is not None and row["loss"] is not None:
            out.append(row)
    return out


def _summarize(rows: list[dict[str, Any]], key: str, lower_is_better: bool) -> None:
    groups: dict[Any, list[float]] = {}
    for r in rows:
        v = r.get(key)
        if v is not None:
            groups.setdefault(r["rank"], []).append(float(v))
    if len(groups) < 2:
        print(f"  {key}: need both ranks; have {sorted(groups)}")
        return
    print(f"\n  {key}  ({'LOWER is better' if lower_is_better else 'HIGHER is better'})")
    print(f"    {'rank':>5} {'n':>3} {'mean':>10} {'median':>10} {'sd':>9} {'best':>10}")
    for k in sorted(groups):
        g = groups[k]
        sd = st.stdev(g) if len(g) > 1 else float("nan")
        best = min(g) if lower_is_better else max(g)
        label = "0 (diag)" if k == 0 else f"{k} (joint)"
        print(f"    {label:>5} {len(g):>3} {st.mean(g):>10.4f} {st.median(g):>10.4f} "
              f"{sd:>9.4f} {best:>10.4f}")
    a, b = groups.get(0, []), groups.get(32, [])
    if len(a) >= 3 and len(b) >= 3:
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        d = st.mean(a) - st.mean(b)
        who = "rank 32 better" if (d > 0) == lower_is_better else "rank 0 better"
        print(f"    diff (rank0 - rank32) = {d:+.4f}  -> {who}")
        print(f"    Mann-Whitney U p = {u.pvalue:.3f}  (n={len(a)},{len(b)})")
    else:
        print(f"    too few per group for a test (n={len(a)},{len(b)})")


def main() -> None:
    rows = _fetch()
    print(f"energy-arm runs with a recorded val/loss: {len(rows)}")
    if not rows:
        print("none yet -- the sweep is still producing them, or the sync has not caught up")
        return
    by_rank: dict[Any, int] = {}
    for r in rows:
        by_rank[r["rank"]] = by_rank.get(r["rank"], 0) + 1
    print(f"  by rank: {by_rank}")

    print("\n=== THE JOINT TEST: energy score ===")
    _summarize(rows, "loss", lower_is_better=True)

    print("\n=== THE BLIND METRIC (expected null): per-feature Pearson ===")
    _summarize(rows, "pearson", lower_is_better=False)

    # Coverage has a TARGET, not a direction: 0.50 and 0.80 are the calibrated values, and
    # BOTH over- and under-coverage are miscalibration. Scoring it "higher is better" would
    # reward the more over-dispersed arm. Compare |coverage - nominal| instead, which is a
    # distance and so is genuinely lower-is-better. (`pit_ks` is already a distance.)
    print("\n=== CALIBRATION ===")
    for row in rows:
        for key, target in (("coverage_50", 0.5), ("coverage_80", 0.8)):
            if row.get(key) is not None:
                row[f"|{key}-nominal|"] = abs(float(row[key]) - target)
    for k in ("coverage_50", "coverage_80"):
        _summarize(rows, k, lower_is_better=False)  # raw value, for direction of the error
        _summarize(rows, f"|{k}-nominal|", lower_is_better=True)  # the actual verdict
    _summarize(rows, "pit_ks", lower_is_better=True)

    out = osp.join(
        experiment_results_dir("019-simb-multimodal", __file__),
        "energy_rank_ablation.json",
    )
    epochs = sorted(float(r["epoch"]) for r in rows if r.get("epoch") is not None)
    med_epoch = epochs[len(epochs) // 2] if epochs else None
    print(f"\nmedian final epoch of the compared runs: {med_epoch}")

    with open(out, "w") as f:
        json.dump(
            {
                "n_runs": len(rows),
                "median_final_epoch": med_epoch,
                # Travels WITH the numbers, so the caveat cannot be separated from the data
                # by someone reading the JSON without the script.
                "caveat": (
                    "Runs come from the _007 round, which was declared VOID: median final "
                    "epoch ~67 because early stopping watched val/loss, which is measured to "
                    "move opposite the ranked metric on this task (loss bottoms ~epoch "
                    "103-136; val Pearson peaks ~epoch 1367). These are TRUNCATED models. "
                    "Directionally suggestive, not a verdict on the joint head. Re-run "
                    "against a round trained to saturation before citing."
                ),
                "rows": rows,
            },
            f,
            indent=2,
        )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
