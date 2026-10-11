# experiments/030-solid-growth-multi/scripts/scaling_readout_030.py
# [[experiments.030-solid-growth-multi.scripts.scaling_readout_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/scaling_readout_030
r"""Read the Delta scaling grid: per-screen validation MSE against D and N, fitted.

Runs are found in W&B by ``config.scaling.sweep`` (the 030 grid ``030_sclw1`` in the 030
project, the paired 025 curve ``025_sclw1`` in the 025 project). A run's chain of
continuations is stitched by ``config.resumed_from`` (each link names the checkpoint of the
one before), so a point of the grid is one (size, subset, seed) with every epoch of every
link. Per epoch the script keeps the validation MSE in label units (``val/gene_interaction/
MSE``, after the inverse transform), the per-screen MSE (``val/token/<Tmi...>/
gene_interaction/MSE``, the within-screen read the pooled number hides behind the 0.35
screen-offset floor), the pooled and per-screen Pearson, and the compute keys
(``compute/records_seen``, ``compute/flops_cumulative``, ``compute/gpu_hours_cumulative``,
``compute/params_non_embedding``).

Two reads per point, both from the same epochs:

- **selected**: the epoch of lowest pooled validation MSE (the validation-selected
  checkpoint), the converged read for subsets that converge inside their budget;
- **matched**: the loss at a fixed records-seen budget (``--budget``, default the smallest
  full-data chain's total), read by interpolation on the epoch curve, the compute-matched
  read that does not favor subsets that ran more epochs.

Fits, per screen and pooled, on the selected read: ``L = A D^-beta`` and ``L = E + A D^-beta``
at fixed N (the 180x8 column), ``L = A N^-alpha`` and ``L = E + A N^-alpha`` at fixed D (the
full column), by least squares in log space from a grid of starts, with a bootstrap over
runs for the 5 to 95 percent interval of every exponent, and with the largest point held
out and predicted. The replicate floor is drawn beside every fitted E: in label units,
E_screen = (1 - reliability) x var(tau) with the reliabilities of
``triple_noise_ceiling_030.json`` (2020 same-strain 0.44; 2018 Dango 0.59 and the cross-year
0.89 as the two bounds).

    PYTHONPATH=$PWD python experiments/030-solid-growth-multi/scripts/scaling_readout_030.py \\
        --sweep 030_sclw1 --paired-sweep 025_sclw1

Writes ``results/scaling/<sweep>_epochs.csv`` (every epoch of every run), ``<sweep>_points.csv``
(one row per point and read), ``<sweep>_fits.json`` and the figure
``$ASSET_IMAGES_DIR/030-solid-growth-multi/scaling_<sweep>_<timestamp>.{svg,png}``.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv
from pydantic import BaseModel
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from arm_030 import EXPERIMENT, load_json_artifact, results_dir  # noqa: E402

ENTITY = "zhao-group"
PROJECTS = {
    "030": "torchcell_030-solid-growth-multi_equivariant_cell_graph_transformer",
    "025": "torchcell_025-solid-growth_equivariant_cell_graph_transformer",
}
FArray = np.ndarray[Any, np.dtype[np.float64]]
SCREENS = ("TmiKuzmin2018Dataset", "TmiKuzmin2020Dataset")
MSE_KEYS = {
    "pooled": "val/gene_interaction/MSE",
    "k2018": f"val/token/{SCREENS[0]}/gene_interaction/MSE",
    "k2020": f"val/token/{SCREENS[1]}/gene_interaction/MSE",
}
PEARSON_KEYS = {
    "pooled": "val/gene_interaction/Pearson",
    "k2018": f"val/token/{SCREENS[0]}/gene_interaction/Pearson",
    "k2020": f"val/token/{SCREENS[1]}/gene_interaction/Pearson",
}
COMPUTE_KEYS = [
    "compute/records_seen",
    "compute/flops_cumulative",
    "compute/gpu_hours_cumulative",
    "compute/records_per_epoch",
]
SUBSET_DENOM = {"subset_S3Q_indices": 1, "frac4": 4, "frac16": 16, "frac64": 64}


class FitResult(BaseModel):
    """One fitted form on one axis for one target."""

    axis: str
    target: str
    form: str
    params: dict[str, float]
    exponent_ci90: tuple[float, float] | None
    heldout_x: float | None
    heldout_observed: float | None
    heldout_predicted: float | None
    n_points: int


class ScalingReadout(BaseModel):
    """Every fit of one sweep with the matched budget and the replicate floors."""

    sweep: str
    paired_sweep: str | None
    budget_records: float
    replicate_floor: dict[str, dict[str, float]]
    fits: list[FitResult]


def _denominator(indices_name: str) -> int:
    for key, d in SUBSET_DENOM.items():
        if key in indices_name:
            return d
    raise ValueError(f"unknown subset {indices_name}")


def _history(run: Any, keys: list[str]) -> pd.DataFrame:
    frames = []
    for k in keys:
        h = run.history(keys=["epoch", k], samples=5000, pandas=True)
        if h is None or len(h) == 0:
            continue
        h = h.drop(columns=[c for c in h.columns if c.startswith("_")])
        h[k] = pd.to_numeric(h[k], errors="coerce")
        frames.append(h.dropna().groupby("epoch").last())
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, how="outer")
    return pd.DataFrame(out.sort_index())


def collect(sweep: str, project_key: str) -> pd.DataFrame:
    """Every epoch of every run of ``sweep`` in one project, chains stitched."""
    api = wandb.Api(timeout=180)
    rows = []
    for run in api.runs(f"{ENTITY}/{PROJECTS[project_key]}"):
        cfg = run.config
        if (cfg.get("scaling") or {}).get("sweep") != sweep:
            continue
        model = cfg.get("model", {})
        subset = cfg.get("subset", {})
        h = _history(
            run, list(MSE_KEYS.values()) + list(PEARSON_KEYS.values()) + COMPUTE_KEYS
        )
        if h.empty:
            continue
        for epoch, r in h.iterrows():
            rows.append(
                {
                    "project": project_key,
                    "run_id": run.id,
                    "run_name": run.name,
                    "job_name": run.name,
                    "resumed_from": cfg.get("resumed_from") or "",
                    "width": int(model.get("hidden_channels", 0)),
                    "depth": int(model.get("num_transformer_layers", 0)),
                    "denominator": _denominator(str(subset.get("indices", ""))),
                    "seed": int(cfg.get("seed", 42)),
                    "lr": float(
                        cfg.get("regression_task", {})
                        .get("optimizer", {})
                        .get("lr", float("nan"))
                    ),
                    "epoch": int(str(epoch)),
                    **{f"mse_{k}": r.get(v) for k, v in MSE_KEYS.items()},
                    **{f"pearson_{k}": r.get(v) for k, v in PEARSON_KEYS.items()},
                    "records_seen": r.get("compute/records_seen"),
                    "flops_cumulative": r.get("compute/flops_cumulative"),
                    "gpu_hours": r.get("compute/gpu_hours_cumulative"),
                    "records_per_epoch": r.get("compute/records_per_epoch"),
                    "params_non_embedding": run.summary.get(
                        "compute/params_non_embedding"
                    ),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # A point is (width, depth, denominator, seed, lr); links of a chain share it and
    # contribute disjoint epochs.
    df["point"] = df.apply(
        lambda r: f"N{r.width}L{r.depth}-D{r.denominator}-s{r.seed}-lr{r.lr:g}", axis=1
    )
    return df.sort_values(["point", "epoch"]).drop_duplicates(
        ["point", "epoch"], keep="last"
    )


def points(df: pd.DataFrame, budget: float) -> pd.DataFrame:
    """One row per point and read (selected, matched)."""
    out = []
    for point, g in df.groupby("point"):
        g = g.sort_values("epoch")
        best = g.loc[g["mse_pooled"].idxmin()]
        base = {
            "point": point,
            "width": int(g["width"].iloc[0]),
            "depth": int(g["depth"].iloc[0]),
            "denominator": int(g["denominator"].iloc[0]),
            "seed": int(g["seed"].iloc[0]),
            "lr": float(g["lr"].iloc[0]),
            "N": float(g["params_non_embedding"].dropna().iloc[0])
            if g["params_non_embedding"].notna().any()
            else float("nan"),
            "D_records": float(g["records_per_epoch"].dropna().iloc[0])
            if g["records_per_epoch"].notna().any()
            else float("nan"),
            "epochs": int(g["epoch"].max()) + 1,
            "records_seen_total": float(g["records_seen"].max()),
            "gpu_hours_total": float(g["gpu_hours"].max()),
        }
        out.append(
            {
                **base,
                "read": "selected",
                "epoch": int(best["epoch"]),
                **{
                    k: float(best[k])
                    for k in (
                        "mse_pooled",
                        "mse_k2018",
                        "mse_k2020",
                        "pearson_pooled",
                        "pearson_k2018",
                        "pearson_k2020",
                    )
                },
            }
        )
        rs = g["records_seen"].to_numpy(dtype=float)
        if np.isfinite(rs).all() and rs.max() >= budget:
            row = {
                **base,
                "read": "matched",
                "epoch": int(np.interp(budget, rs, g["epoch"].to_numpy(dtype=float))),
            }
            for k in (
                "mse_pooled",
                "mse_k2018",
                "mse_k2020",
                "pearson_pooled",
                "pearson_k2018",
                "pearson_k2020",
            ):
                row[k] = float(np.interp(budget, rs, g[k].to_numpy(dtype=float)))
            out.append(row)
    return pd.DataFrame(out)


# --- fitting --------------------------------------------------------------------------
def _fit_log(x: FArray, y: FArray, floored: bool) -> dict[str, float]:
    lx, ly = np.log(x), np.log(y)
    best: tuple[float, FArray] | None = None
    exps = np.linspace(0.05, 1.0, 8)
    for e0 in exps:
        for e_frac in [0.0] if not floored else [0.0, 0.3, 0.6, 0.9]:
            E0 = e_frac * y.min()
            A0 = (
                np.exp(np.mean(ly + e0 * lx))
                if E0 == 0
                else np.exp(np.mean(np.log(np.clip(y - E0, 1e-12, None)) + e0 * lx))
            )
            p0 = np.array([np.log(A0), e0] + ([E0] if floored else []))

            def resid(p: FArray) -> FArray:
                pred = np.exp(p[0]) * x ** (-p[1]) + (p[2] if floored else 0.0)
                return np.asarray(np.log(np.clip(pred, 1e-12, None)) - ly)

            lb = [-np.inf, 0.0] + ([0.0] if floored else [])
            ub = [np.inf, 5.0] + ([float(y.min())] if floored else [])
            try:
                r = least_squares(
                    resid, p0, bounds=(lb, ub), loss="huber", f_scale=1e-3
                )
            except ValueError:
                continue
            if best is None or r.cost < best[0]:
                best = (r.cost, r.x)
    assert best is not None
    p = best[1]
    out = {"A": float(np.exp(p[0])), "exponent": float(p[1])}
    if floored:
        out["E"] = float(p[2])
    return out


def _predict(params: dict[str, float], x: float) -> float:
    return float(params["A"] * x ** (-params["exponent"]) + params.get("E", 0.0))


def fit_axis(
    pts: pd.DataFrame,
    xcol: str,
    target: str,
    axis: str,
    n_boot: int = 300,
    seed: int = 0,
) -> list[FitResult]:
    """Both forms on one axis: fit all points, hold out the largest x and predict it,
    bootstrap over runs for the exponent interval.
    """
    d = pts[[xcol, target]].dropna()
    d = d[d[target] > 0]
    if len(d) < 3:
        return []
    x, y = d[xcol].to_numpy(dtype=float), d[target].to_numpy(dtype=float)
    xmax = x.max()
    held = x == xmax
    rng = np.random.default_rng(seed)
    fits = []
    for floored in (False, True):
        if floored and len(np.unique(x[~held])) < 3:
            continue
        params_all = _fit_log(x, y, floored)
        params_in = (
            _fit_log(x[~held], y[~held], floored)
            if (~held).sum() >= (3 if floored else 2)
            else params_all
        )
        boots = []
        idx = np.arange(len(x))
        for _ in range(n_boot):
            s = rng.choice(idx, size=len(idx), replace=True)
            if len(np.unique(x[s])) < (3 if floored else 2):
                continue
            boots.append(_fit_log(x[s], y[s], floored)["exponent"])
        ci = (
            (float(np.percentile(boots, 5)), float(np.percentile(boots, 95)))
            if len(boots) >= 20
            else None
        )
        fits.append(
            FitResult(
                axis=axis,
                target=target,
                form="floored" if floored else "power",
                params=params_all,
                exponent_ci90=ci,
                heldout_x=float(xmax),
                heldout_observed=float(y[held].mean()),
                heldout_predicted=float(_predict(params_in, xmax)),
                n_points=int(len(x)),
            )
        )
    return fits


def replicate_floor() -> dict[str, dict[str, float]]:
    """E per screen in label units from the recorded reliabilities and score sds."""
    nc = load_json_artifact("triple_noise_ceiling_030.json")
    pc = nc["pair_classes"]
    nm = nc["noise_model"]
    sd18 = float(nm[SCREENS[0]]["sd_of_scores"])
    sd20 = float(nm[SCREENS[1]]["sd_of_scores"])
    return {
        "k2018": {
            "E_dango_0.59": (1 - 0.59) * sd18**2,
            "E_crossyear": (1 - float(pc["cross_dataset_2018_vs_2020"]["pearson"]))
            * sd18**2,
        },
        "k2020": {
            "E_same_strain": (1 - float(pc["same_dataset_same_strains"]["pearson"]))
            * sd20**2
        },
    }


def figure(
    pts: pd.DataFrame,
    paired: pd.DataFrame | None,
    floor: dict[str, dict[str, float]],
    fits: list[FitResult],
    sweep: str,
) -> str:
    load_dotenv()
    out_dir = osp.join(os.environ["ASSET_IMAGES_DIR"], EXPERIMENT)
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "svg.fonttype": "none",
            "axes.linewidth": 0.5,
        }
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(55))
    )
    sel = pts[pts["read"] == "selected"]
    targets = [
        ("mse_pooled", "pooled"),
        ("mse_k2018", "Kuzmin 2018"),
        ("mse_k2020", "Kuzmin 2020"),
    ]
    # a: L(D) at 180x8
    ax = axes[0]
    d180 = sel[(sel["width"] == 180) & (sel["depth"] == 8)]
    for (col, name), color in zip(targets, PLOT_PALETTE):
        ax.scatter(d180["D_records"], d180[col], s=8, color=color, label=f"030 {name}")
    if paired is not None and not paired.empty:
        p180 = paired[(paired["read"] == "selected")]
        for (col, name), color in zip(targets, PLOT_PALETTE):
            ax.scatter(
                p180["D_records"],
                p180[col],
                s=8,
                marker="s",
                facecolors="none",
                edgecolors=color,
                label=f"025 {name}",
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("D, training records")
    ax.set_ylabel("validation MSE, label units")
    ax.set_title("data axis at 180x8", fontsize=6)
    # b: L(N) at full D
    ax = axes[1]
    nf = sel[sel["denominator"] == 1]
    for (col, name), color in zip(targets, PLOT_PALETTE):
        ax.scatter(nf["N"], nf[col], s=8, color=color, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N, non-embedding parameters")
    ax.set_title("size axis at full data", fontsize=6)
    # c: the grid, pooled MSE vs records seen per point, with the floors
    ax = axes[2]
    for (col, name), color in zip(targets, PLOT_PALETTE):
        ax.scatter(
            sel["records_seen_total"], sel[col], s=6, color=color, label=name, alpha=0.8
        )
    for color, (screen, vals) in zip(PLOT_PALETTE[1:3], floor.items()):
        for k, v in vals.items():
            ax.axhline(v, color=color, lw=0.5, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("records seen")
    ax.set_title("every point; dashed = replicate floors", fontsize=6)
    for ax, letter in zip(axes, "abc"):
        for s in ax.spines.values():
            s.set_linewidth(0.5)
        ax.text(
            -0.15, 1.05, letter, transform=ax.transAxes, fontsize=8, fontweight="bold"
        )
        ax.legend(fontsize=5, frameon=False)
    fig.tight_layout()
    base = osp.join(out_dir, f"scaling_{sweep}_{timestamp()}")
    savefig_true_size_svg(fig, base + ".svg")
    fig.savefig(base + ".png", dpi=300)
    plt.close(fig)
    return base + ".svg"


def main() -> None:
    parser = argparse.ArgumentParser(description="scaling readout of the Delta grid")
    parser.add_argument("--sweep", default="030_sclw1")
    parser.add_argument("--paired-sweep", default="025_sclw1")
    parser.add_argument(
        "--budget", type=float, default=None, help="matched records-seen budget"
    )
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    out_dir = osp.join(results_dir(), "scaling")
    os.makedirs(out_dir, exist_ok=True)

    df = collect(args.sweep, "030")
    if df.empty:
        raise SystemExit(f"no runs with scaling.sweep == {args.sweep}")
    df.to_csv(osp.join(out_dir, f"{args.sweep}_epochs.csv"), index=False)
    full = df[df["denominator"] == 1]
    budget = (
        args.budget or float(full.groupby("point")["records_seen"].max().min())
        if not full.empty
        else float(df["records_seen"].max())
    )
    pts = points(df, budget)
    pts.to_csv(osp.join(out_dir, f"{args.sweep}_points.csv"), index=False)
    paired = None
    if args.paired_sweep:
        pdf = collect(args.paired_sweep, "025")
        if not pdf.empty:
            pdf.to_csv(
                osp.join(out_dir, f"{args.paired_sweep}_epochs.csv"), index=False
            )
            paired = points(pdf, budget)
            paired.to_csv(
                osp.join(out_dir, f"{args.paired_sweep}_points.csv"), index=False
            )

    sel = pts[pts["read"] == "selected"]
    fits: list[FitResult] = []
    d180 = sel[(sel["width"] == 180) & (sel["depth"] == 8)]
    nfull = sel[sel["denominator"] == 1]
    for target in ("mse_pooled", "mse_k2018", "mse_k2020"):
        fits += fit_axis(d180, "D_records", target, "D")
        fits += fit_axis(nfull, "N", target, "N")
    floor = replicate_floor()
    readout = ScalingReadout(
        sweep=args.sweep,
        paired_sweep=args.paired_sweep or None,
        budget_records=budget,
        replicate_floor=floor,
        fits=fits,
    )
    with open(osp.join(out_dir, f"{args.sweep}_fits.json"), "w") as f:
        f.write(readout.model_dump_json(indent=2))

    print(f"points ({len(sel)} selected reads; matched budget {budget:.3g} records):")
    print(
        sel[
            [
                "point",
                "N",
                "D_records",
                "epochs",
                "records_seen_total",
                "gpu_hours_total",
                "mse_pooled",
                "mse_k2018",
                "mse_k2020",
                "pearson_k2018",
                "pearson_k2020",
            ]
        ].to_string(index=False)
    )
    print("\nfits (selected read):")
    for ft in fits:
        ci = (
            f"[{ft.exponent_ci90[0]:.3f}, {ft.exponent_ci90[1]:.3f}]"
            if ft.exponent_ci90
            else "n/a"
        )
        e = f" E={ft.params['E']:.3g}" if "E" in ft.params else ""
        print(
            f"  {ft.axis} {ft.target:<10} {ft.form:<7} exponent {ft.params['exponent']:.3f} ci90 {ci}{e}  held-out {ft.heldout_x:.3g}: observed {ft.heldout_observed:.3g} predicted {ft.heldout_predicted:.3g}"
        )
    print("replicate floors (label units):", json.dumps(floor))
    if not args.no_figure:
        print("figure:", figure(pts, paired, floor, fits, args.sweep))
    print(
        f"wrote {out_dir}/{args.sweep}_{{epochs,points}}.csv and {args.sweep}_fits.json"
    )


if __name__ == "__main__":
    main()
