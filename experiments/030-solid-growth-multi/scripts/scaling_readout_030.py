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
  checkpoint), the converged read for subsets that converge inside their budget. The
  full-data points of the grid stop at 10 epochs and do not converge, so along D this
  read mixes converged small subsets with an unconverged top point, which flattens the
  curve; it is reported, not trusted alone;
- **matched**: the loss at a fixed records-seen budget (``--budget``, default the
  SMALLEST total any point of the grid reached, so every point has the read), by
  interpolation on the epoch curve. This is loss against unique data at a fixed training
  budget, the read every point of the grid supports.

Fits, per screen and pooled, on BOTH reads: ``L = A D^-beta`` and ``L = E + A D^-beta``
at fixed N (the 180x8 column; D in training records, and on 030 also in gene_interaction
entry rows, the unit the per-entry loss averages over, read from
``subset_s3q_fractions_030_summary.json``), ``L = A N^-alpha`` and ``L = E + A N^-alpha``
at fixed D (the full column), by least squares in log space from a grid of starts, with
a bootstrap over runs for the 5 to 95 percent interval of every exponent, and with the
largest point held out and predicted (the floored form needs four distinct x in sample
for that test; on three it is an exact fit and predicts nothing). The replicate floor
is drawn beside every fitted E: in label units, E_screen = (1 - reliability) x var(tau)
with the reliabilities of ``triple_noise_ceiling_030.json`` (2020 same-strain 0.44; 2018
Dango 0.59 and the cross-year 0.89 as the two bounds).

The paired gap: per subset and read, the 030 pooled MSE minus the 025 pooled MSE at
180x8, each a mean over seeds with the seed sd, in ``<sweep>_gap.csv``. 025 logs no
per-screen key, so the gap is pooled only.

W&B history is queried per key together with ``epoch``; the compute keys of a run whose
callback predates the ``epoch`` field fall back to ``compute/epochs_completed``, which
equals the epoch index plus one when every link of the chain carried the callback.

    PYTHONPATH=$PWD python experiments/030-solid-growth-multi/scripts/scaling_readout_030.py \\
        --sweep 030_sclw1 --paired-sweep 025_sclw1

Writes ``results/scaling/<sweep>_epochs.csv`` (every epoch of every run), ``<sweep>_points.csv``
(one row per point and read), ``<sweep>_gap.csv``, ``<sweep>_fits.json`` and the figure
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
READ_KEYS = (
    "mse_pooled",
    "mse_k2018",
    "mse_k2020",
    "pearson_pooled",
    "pearson_k2018",
    "pearson_k2020",
)


class FitResult(BaseModel):
    """One fitted form on one axis for one target and one read."""

    axis: str
    read: str
    target: str
    form: str
    params: dict[str, float]
    exponent_ci90: tuple[float, float] | None
    heldout_x: float | None
    heldout_observed: float | None
    heldout_predicted: float | None
    n_points: int


class GapRow(BaseModel):
    """The 030 minus 025 pooled validation MSE at one subset and read, over seeds."""

    read: str
    denominator: int
    D_records: float
    mse_030: float
    sd_030: float
    n_030: int
    mse_025: float
    sd_025: float
    n_025: int
    gap: float


class ScalingReadout(BaseModel):
    """Every fit of one sweep with the matched budget, the floors and the gap."""

    sweep: str
    paired_sweep: str | None
    budget_records: float
    replicate_floor: dict[str, dict[str, float]]
    fits: list[FitResult]
    gap: list[GapRow]


def _denominator(indices_name: str) -> int:
    for key, d in SUBSET_DENOM.items():
        if key in indices_name:
            return d
    raise ValueError(f"unknown subset {indices_name}")


def _history(run: Any, keys: list[str]) -> pd.DataFrame:
    frames = []
    for k in keys:
        h = run.history(keys=["epoch", k], samples=5000, pandas=True)
        if (h is None or len(h) == 0) and k.startswith("compute/"):
            # the callback logged no `epoch` before 2026-10-12: derive it
            h = run.history(
                keys=["compute/epochs_completed", k], samples=5000, pandas=True
            )
            if h is not None and len(h):
                h["epoch"] = (
                    pd.to_numeric(h["compute/epochs_completed"], errors="coerce") - 1
                )
                h = h.drop(columns=["compute/epochs_completed"])
        if h is None or len(h) == 0:
            continue
        h = h.drop(columns=[c for c in h.columns if c.startswith("_")])
        h[k] = pd.to_numeric(h[k], errors="coerce")
        h["epoch"] = pd.to_numeric(h["epoch"], errors="coerce")
        frames.append(h.dropna().groupby("epoch").last())
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, how="outer")
    return pd.DataFrame(out.sort_index())


def _num(row: pd.Series, key: str) -> float:
    """A row's value as float, NaN when the key was never logged."""
    v = row.get(key)
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


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
                    "epoch": int(float(str(epoch))),
                    **{f"mse_{k}": _num(r, v) for k, v in MSE_KEYS.items()},
                    **{f"pearson_{k}": _num(r, v) for k, v in PEARSON_KEYS.items()},
                    "records_seen": _num(r, "compute/records_seen"),
                    "flops_cumulative": _num(r, "compute/flops_cumulative"),
                    "gpu_hours": _num(r, "compute/gpu_hours_cumulative"),
                    "records_per_epoch": _num(r, "compute/records_per_epoch"),
                    "params_non_embedding": _num(
                        pd.Series(dict(run.summary)), "compute/params_non_embedding"
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


def _rows_per_subset() -> dict[int, float]:
    """D in gene_interaction entry rows per subset denominator on the 030 build."""
    summary = load_json_artifact("subset_s3q_fractions_030_summary.json")
    out = {1: float(summary["n_s3q_train_gi_entry_rows"])}
    for fs in summary["fractions"]:
        out[int(fs["denominator"])] = float(fs["n_train_gi_entry_rows"])
    return out


def points(
    df: pd.DataFrame, budget: float, rows: dict[int, float] | None
) -> pd.DataFrame:
    """One row per point and read (selected, matched)."""
    out = []
    for point, g in df.groupby("point"):
        g = g.sort_values("epoch")
        if g["mse_pooled"].notna().sum() == 0:
            print(f"  {point}: no validation MSE logged, skipped")
            continue
        best = g.loc[g["mse_pooled"].idxmin()]
        denominator = int(g["denominator"].iloc[0])
        base = {
            "point": point,
            "width": int(g["width"].iloc[0]),
            "depth": int(g["depth"].iloc[0]),
            "denominator": denominator,
            "seed": int(g["seed"].iloc[0]),
            "lr": float(g["lr"].iloc[0]),
            "N": float(g["params_non_embedding"].dropna().iloc[0])
            if g["params_non_embedding"].notna().any()
            else float("nan"),
            "D_records": float(g["records_per_epoch"].dropna().iloc[0])
            if g["records_per_epoch"].notna().any()
            else float("nan"),
            "D_rows": rows.get(denominator, float("nan")) if rows else float("nan"),
            "epochs": int(g["epoch"].max()) + 1,
            "records_seen_total": float(g["records_seen"].max()),
            "gpu_hours_total": float(g["gpu_hours"].max()),
        }
        out.append(
            {
                **base,
                "read": "selected",
                "epoch": int(best["epoch"]),
                **{k: float(best[k]) for k in READ_KEYS},
            }
        )
        # the matched read interpolates on the epochs that carry BOTH a compute row
        # and a validation row (validation may run every k-th epoch)
        m = g[g["records_seen"].notna() & g["mse_pooled"].notna()]
        rs = m["records_seen"].to_numpy(dtype=float)
        if len(rs) >= 1 and rs.max() >= budget:
            row = {
                **base,
                "read": "matched",
                "epoch": int(np.interp(budget, rs, m["epoch"].to_numpy(dtype=float))),
            }
            for k in READ_KEYS:
                row[k] = float(np.interp(budget, rs, m[k].to_numpy(dtype=float)))
            out.append(row)
        else:
            print(
                f"  {point}: no matched read, reached {rs.max() if len(rs) else float('nan'):.3g} "
                f"records of the {budget:.3g} budget"
            )
    return pd.DataFrame(out)


def paired_gap(pts: pd.DataFrame, paired: pd.DataFrame) -> list[GapRow]:
    """030 minus 025 pooled MSE at 180x8 per subset and read, seed means and sds."""
    out: list[GapRow] = []
    a = pts[(pts["width"] == 180) & (pts["depth"] == 8)]
    b = paired[(paired["width"] == 180) & (paired["depth"] == 8)]
    for read in ("selected", "matched"):
        for d in sorted(set(a["denominator"]) & set(b["denominator"])):
            ga = a[(a["read"] == read) & (a["denominator"] == d)]["mse_pooled"].dropna()
            gb = b[(b["read"] == read) & (b["denominator"] == d)]["mse_pooled"].dropna()
            if ga.empty or gb.empty:
                continue
            out.append(
                GapRow(
                    read=read,
                    denominator=int(d),
                    D_records=float(a[a["denominator"] == d]["D_records"].iloc[0]),
                    mse_030=float(ga.mean()),
                    sd_030=float(ga.std(ddof=1)) if len(ga) > 1 else float("nan"),
                    n_030=int(len(ga)),
                    mse_025=float(gb.mean()),
                    sd_025=float(gb.std(ddof=1)) if len(gb) > 1 else float("nan"),
                    n_025=int(len(gb)),
                    gap=float(ga.mean() - gb.mean()),
                )
            )
    return out


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
    read: str,
    n_boot: int = 300,
    seed: int = 0,
) -> list[FitResult]:
    """Both forms on one axis: fit all points, hold out the largest x and predict it,
    bootstrap over runs for the exponent interval. The floored form is fitted on three
    or more distinct x; its held-out prediction needs four in sample (three points fit
    three parameters exactly) and is left None below that.
    """
    d = pts[[xcol, target]].dropna()
    d = d[d[target] > 0]
    if len(d) < 3 or len(np.unique(d[xcol])) < 2:
        return []
    x, y = d[xcol].to_numpy(dtype=float), d[target].to_numpy(dtype=float)
    xmax = x.max()
    held = x == xmax
    rng = np.random.default_rng(seed)
    fits = []
    for floored in (False, True):
        if floored and len(np.unique(x)) < 3:
            continue
        params_all = _fit_log(x, y, floored)
        n_in = len(np.unique(x[~held]))
        params_in = (
            _fit_log(x[~held], y[~held], floored)
            if n_in >= (4 if floored else 2)
            else None
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
                read=read,
                target=target,
                form="floored" if floored else "power",
                params=params_all,
                exponent_ci90=ci,
                heldout_x=float(xmax),
                heldout_observed=float(y[held].mean()),
                heldout_predicted=(
                    float(_predict(params_in, xmax)) if params_in is not None else None
                ),
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
    # the matched budget: the smallest records-seen total ANY point reached, so every
    # point of the grid has the read (the full-data total would leave the subsets out:
    # 240 epochs of 1/64 see half the records of 10 full epochs)
    reached = df.groupby("point")["records_seen"].max().dropna()
    if reached.empty:
        raise SystemExit("no run logged compute/records_seen; nothing to match on")
    budget = float(args.budget) if args.budget else float(reached.min())
    print(
        f"matched budget {budget:.4g} records (smallest total reached: {reached.idxmin()})"
    )
    rows = _rows_per_subset()
    pts = points(df, budget, rows)
    pts.to_csv(osp.join(out_dir, f"{args.sweep}_points.csv"), index=False)
    paired = None
    gap: list[GapRow] = []
    if args.paired_sweep:
        pdf = collect(args.paired_sweep, "025")
        if not pdf.empty:
            pdf.to_csv(
                osp.join(out_dir, f"{args.paired_sweep}_epochs.csv"), index=False
            )
            paired = points(pdf, budget, None)
            paired.to_csv(
                osp.join(out_dir, f"{args.paired_sweep}_points.csv"), index=False
            )
            gap = paired_gap(pts, paired)
            pd.DataFrame([g.model_dump() for g in gap]).to_csv(
                osp.join(out_dir, f"{args.sweep}_gap.csv"), index=False
            )

    fits: list[FitResult] = []
    for read in ("selected", "matched"):
        sub = pts[pts["read"] == read]
        d180 = sub[(sub["width"] == 180) & (sub["depth"] == 8)]
        nfull = sub[sub["denominator"] == 1]
        for target in ("mse_pooled", "mse_k2018", "mse_k2020"):
            fits += fit_axis(d180, "D_records", target, "D_records", read)
            fits += fit_axis(d180, "D_rows", target, "D_rows", read)
            fits += fit_axis(nfull, "N", target, "N", read)
    floor = replicate_floor()
    readout = ScalingReadout(
        sweep=args.sweep,
        paired_sweep=args.paired_sweep or None,
        budget_records=budget,
        replicate_floor=floor,
        fits=fits,
        gap=gap,
    )
    with open(osp.join(out_dir, f"{args.sweep}_fits.json"), "w") as f:
        f.write(readout.model_dump_json(indent=2))

    sel = pts[pts["read"] == "selected"]
    print(
        f"points ({len(sel)} selected, {int((pts['read'] == 'matched').sum())} matched):"
    )
    print(
        pts[
            [
                "point",
                "read",
                "N",
                "D_records",
                "D_rows",
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
    print("\nfits:")
    for ft in fits:
        ci = (
            f"[{ft.exponent_ci90[0]:.3f}, {ft.exponent_ci90[1]:.3f}]"
            if ft.exponent_ci90
            else "n/a"
        )
        e = f" E={ft.params['E']:.3g}" if "E" in ft.params else ""
        pred = (
            f"{ft.heldout_predicted:.3g}" if ft.heldout_predicted is not None else "n/a"
        )
        print(
            f"  {ft.read:<8} {ft.axis:<9} {ft.target:<10} {ft.form:<7} exponent {ft.params['exponent']:.3f} ci90 {ci}{e}  held-out {ft.heldout_x:.3g}: observed {ft.heldout_observed:.3g} predicted {pred}"
        )
    if gap:
        print("\npaired gap (030 minus 025 pooled MSE at 180x8):")
        print(pd.DataFrame([g.model_dump() for g in gap]).to_string(index=False))
    print("replicate floors (label units):", json.dumps(floor))
    if not args.no_figure:
        print("figure:", figure(pts, paired, floor, fits, args.sweep))
    print(
        f"wrote {out_dir}/{args.sweep}_{{epochs,points,gap}}.csv and {args.sweep}_fits.json"
    )


if __name__ == "__main__":
    main()
