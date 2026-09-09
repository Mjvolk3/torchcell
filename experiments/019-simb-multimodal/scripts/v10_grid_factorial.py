# experiments/019-simb-multimodal/scripts/v10_grid_factorial.py
# [[experiments.019-simb-multimodal.scripts.v10_grid_factorial]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/v10_grid_factorial
"""Read out the v10 generalization-gap grid: a 2^4 factorial, 2 seeds, on Delta.

THE DESIGN (delta_expr_v10_grid.slurm, config cgt_expr_v10_grid.yaml). Sixteen cells over
four two-level factors, cell index c = e + 2t + 4r + 8w:

    bit 0  embedding     0 random_1024    1 prot_T5_all
    bit 1  trunk         0 L=6 h=90       1 L=2 h=45
    bit 2  readout       0 two-layer MLP  1 linear
    bit 3  weight decay  0 1e-8           1 1e-4

Everything else is the v9 incumbent (quantile head, pinball loss, masked-label objective,
lr 3e-4, batch 32, dropout 0.1). NOTE that the incumbent's own embedding is `calm`, which is
NOT a level of the embedding factor: the factor contrasts random content against ProtT5
content at a matched width of 1024, so it measures whether the embedding CONTENT matters,
not whether the incumbent's choice was right. Two seeds per cell, 32 runs, `max_epochs`
1,400, one run per GPU, four per node, eight array tasks of job 21830323. Three of the eight
tasks hit the two-day wall before epoch 1,400, so the runs end between 990 and 1,399 epochs.

THE SCORE. `roll_max` (max of a centered ROLL_WINDOW-epoch rolling mean of
`val/expression/pearson_per_feature`, imported from pull_round_leaderboards.py) over the
epochs every run reached, so the contrast is at a MATCHED budget. The matched budget is
computed from the data as the smallest final epoch across the grid, never assumed. The
unmatched full-run `roll_max` is written alongside and is not used for any contrast.

THE STATISTICS. Main effect of a factor = mean of its level-1 cells minus mean of its
level-0 cells, 16 runs a side. Two-way interactions the same way on the product of the
centered factor codes. The error is the pooled within-cell replicate standard deviation
(16 cells, 2 seeds each, 16 degrees of freedom), so every effect has the same standard
error sd_pooled * sqrt(1/16 + 1/16). The reference band drawn beside it comes from
short_budget_spread.json, the eight long-budget v9 runs at the nearest budget. THOSE EIGHT
ARE NOT REPLICATES (found 2026-09-08): they are the arms of the v9 mask-schedule round
(M_sched, M_lo, M_hi, M_fine, M_coarse, M_nomix, M_off, M_gate_rezero), differing in mask
schedule, mixing and gate, so their spread is an arm spread plus nondeterminism and only
bounds the replicate spread from above. The pooled within-cell sd computed HERE is a true
replicate spread.

WHAT THIS CANNOT SAY. The scoring rule is a rolling max, so every score is an upward-biased
order statistic; it is the SAME rule for every cell and the bias cancels in a contrast but
not in an absolute number. A run whose curve never left the chance band is kept in the
factorial (the design stays balanced) and listed by name, and the main effects are also
reported with such runs dropped so the reader can see what they carry.

Outputs: results/v10_grid_factorial.csv (one row per run), results/v10_grid_factorial.json
(effects and summaries), and a full-width three-panel figure in ASSET_IMAGES_DIR.
"""

from __future__ import annotations

import importlib.util
import json
import os
import os.path as osp
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator

load_dotenv()

from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    experiment_results_dir,
    mm_to_in,
    savefig_true_size_svg,
)

_SPEC = importlib.util.spec_from_file_location(
    "_plb", osp.join(osp.dirname(osp.abspath(__file__)), "pull_round_leaderboards.py")
)
_PLB = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PLB)

ENTITY = _PLB.ENTITY
ROLL_WINDOW = _PLB.ROLL_WINDOW
PROJECT = "torchcell_019_expr_v10"
METRIC = "val/expression/pearson_per_feature"
LOSS = "val/loss"
NMSE = "val/expression/nmse"

RESULTS = experiment_results_dir("019-simb-multimodal", __file__)
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "019-simb-multimodal")
SPREAD_JSON = osp.join(RESULTS, "short_budget_spread.json")

FULL_HISTORY_SAMPLES = 50_000
# The grid ran 1,400 epochs; every earlier launch of this project (canaries, the three
# failed array launches) died inside 40 epochs. This is what separates them.
MIN_EPOCHS = 900
# A run whose matched-budget roll_max never clears this is listed as never having left the
# chance band. The per-gene mean baseline scores 0.0000 and the bilinear ProtT5 baseline
# 0.1040 (expression_baselines.py); 0.03 is well under the worst healthy cell.
CHANCE_BAND = 0.03

FACTORS = ("embedding", "trunk", "readout", "weight_decay")
LEVELS = {
    "embedding": ("random_1024", "prot_T5_all"),
    "trunk": ("L6_h90", "L2_h45"),
    "readout": ("mlp", "linear"),
    "weight_decay": ("1e-8", "1e-4"),
}
SHORT = {"embedding": "emb", "trunk": "trunk", "readout": "readout", "weight_decay": "wd"}


def decode_cell(cell: int) -> dict[str, int]:
    return {
        "embedding": cell % 2,
        "trunk": (cell // 2) % 2,
        "readout": (cell // 4) % 2,
        "weight_decay": (cell // 8) % 2,
    }


def roll_mean(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy()


def tag_int(tags: list[str], prefix: str) -> int:
    hits = [t for t in tags if re.fullmatch(prefix + r"\d+", t)]
    assert len(hits) == 1, (prefix, tags)
    return int(hits[0][len(prefix):])


def fetch(api: wandb.Api) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    runs = list(api.runs(f"{ENTITY}/{PROJECT}", per_page=100))
    print(f"{len(runs)} runs in {PROJECT}")
    rows = []
    curves: dict[str, pd.DataFrame] = {}
    for run in runs:
        if METRIC not in run.summary:
            continue
        final_epoch = run.summary.get("epoch")
        if final_epoch is None or final_epoch < MIN_EPOCHS:
            continue
        tags = list(run.tags)
        h = run.history(keys=["epoch", METRIC, LOSS, NMSE], samples=FULL_HISTORY_SAMPLES)
        h = h.dropna(subset=["epoch", METRIC]).sort_values("epoch")
        h = h.drop_duplicates("epoch", keep="last").reset_index(drop=True)
        cell = tag_int(tags, "cell")
        row = {
            "run_id": run.id,
            "state": run.state,
            "cell": cell,
            "seed": tag_int(tags, "seed"),
            "cell_name": [t for t in tags if t.startswith(f"c{cell}_")][0],
            "n_epochs": int(h.epoch.max()),
            "rows_short_by": int(h.epoch.max() + 1 - len(h)),
            **decode_cell(cell),
        }
        rows.append(row)
        curves[run.id] = h
    t = pd.DataFrame(rows)
    assert len(t) == 32, len(t)
    assert (t.groupby("cell").size() == 2).all(), t.groupby("cell").size()
    return t, curves


def score(t: pd.DataFrame, curves: dict[str, pd.DataFrame], budget: int) -> pd.DataFrame:
    t = t.copy()
    for col in (
        "roll_max_matched", "epoch_at_matched", "roll_max_full", "epoch_at_full",
        "pearson_last", "nmse_at_matched_peak", "loss_min_epoch",
    ):
        t[col] = np.nan
    for i, rid in enumerate(t.run_id):
        h = curves[rid]
        pear = h[METRIC].to_numpy()
        roll = roll_mean(pear, ROLL_WINDOW)
        ep = h.epoch.to_numpy()
        within = ep <= budget
        j = int(np.argmax(np.where(within, roll, -np.inf)))
        k = int(np.argmax(roll))
        t.loc[i, "roll_max_matched"] = float(roll[j])
        t.loc[i, "epoch_at_matched"] = int(ep[j])
        t.loc[i, "roll_max_full"] = float(roll[k])
        t.loc[i, "epoch_at_full"] = int(ep[k])
        t.loc[i, "pearson_last"] = float(pear[-1])
        if NMSE in h:
            t.loc[i, "nmse_at_matched_peak"] = float(h[NMSE].iloc[j])
        if LOSS in h:
            loss = h[LOSS].to_numpy()
            t.loc[i, "loss_min_epoch"] = int(ep[int(np.nanargmin(loss))])
    t["chance_band"] = t.roll_max_matched < CHANCE_BAND
    return t


def effects(t: pd.DataFrame, col: str) -> dict[str, object]:
    """Main effects and two-way interactions with the pooled within-cell error."""
    cells = t.groupby("cell")[col]
    within_var = cells.var(ddof=1)  # 16 cells, each with 2 seeds -> 1 df each
    n_cells = int(within_var.notna().sum())
    sd_pooled = float(np.sqrt(within_var.mean()))
    n_side = len(t) / 2
    se = sd_pooled * np.sqrt(1 / n_side + 1 / n_side)
    out: dict[str, object] = {
        "sd_pooled_within_cell": sd_pooled,
        "df": n_cells,
        "se_effect": float(se),
        "n_runs": int(len(t)),
        "grand_mean": float(t[col].mean()),
        "main": {},
        "interaction": {},
        "cell_means": {
            int(c): {"mean": float(g.mean()), "n": int(len(g)),
                     "seeds": [float(x) for x in g]}
            for c, g in cells
        },
    }
    coded = {f: 2 * t[f].to_numpy() - 1 for f in FACTORS}
    for f in FACTORS:
        hi = t[t[f] == 1][col]
        lo = t[t[f] == 0][col]
        eff = float(hi.mean() - lo.mean())
        out["main"][f] = {
            "level0": LEVELS[f][0], "level1": LEVELS[f][1],
            "mean_level0": float(lo.mean()), "mean_level1": float(hi.mean()),
            "effect": eff, "t": eff / se,
        }
    for i, a in enumerate(FACTORS):
        for b in FACTORS[i + 1:]:
            prod = coded[a] * coded[b]
            eff = float(t[col][prod == 1].mean() - t[col][prod == -1].mean())
            out["interaction"][f"{a}x{b}"] = {"effect": eff, "t": eff / se}
    return out


def print_effects(title: str, e: dict[str, object]) -> None:
    print(f"\n=== {title}: n={e['n_runs']}, grand mean {e['grand_mean']:.4f}, "
          f"pooled within-cell sd {e['sd_pooled_within_cell']:.4f} (df {e['df']}), "
          f"se per effect {e['se_effect']:.4f} ===")
    for f, m in e["main"].items():
        print(f"  {f:13s} {m['level0']:>12s} {m['mean_level0']:.4f}  "
              f"{m['level1']:>12s} {m['mean_level1']:.4f}   effect {m['effect']:+.4f}  "
              f"t {m['t']:+.2f}")
    for k, m in e["interaction"].items():
        print(f"  {k:24s} effect {m['effect']:+.4f}  t {m['t']:+.2f}")


def incumbent_at(budget: int) -> dict[str, float]:
    with open(SPREAD_JSON) as fh:
        d = json.load(fh)
    rows = sorted(d["by_budget"], key=lambda r: abs(r["budget_epochs"] - budget))
    r = rows[0]
    return {"budget_epochs": r["budget_epochs"], "mean": r["mean"], "sd": r["sd"],
            "n": r["n"], "history_samples": d["history_samples"]}


def figure(t: pd.DataFrame, curves: dict[str, pd.DataFrame], budget: int,
           e_all: dict[str, object], inc: dict[str, float]) -> str:
    plt.rcParams.update({"font.family": "Arial", "font.size": 6, "svg.fonttype": "none",
                         "axes.linewidth": 0.5})
    fig, axes = plt.subplots(1, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(66)),
                             gridspec_kw={"width_ratios": [1.35, 1.0, 1.15], "wspace": 0.5})
    fig.subplots_adjust(left=0.065, right=0.99, top=0.9, bottom=0.27)
    col_emb = {0: PLOT_PALETTE[5], 1: PLOT_PALETTE[0]}
    marker_seed = {0: "o", 1: "s"}

    # (a) every cell, sorted by cell mean, both seeds
    ax = axes[0]
    order = t.groupby("cell")["roll_max_matched"].mean().sort_values().index.to_list()
    pos = {c: i for i, c in enumerate(order)}
    for c in order:
        sub = t[t.cell == c]
        ax.plot([pos[c]] * 2, sub.roll_max_matched, color="black", lw=0.5, zorder=1)
        for _, r in sub.iterrows():
            ax.scatter(pos[c], r.roll_max_matched, s=12, marker=marker_seed[int(r.seed)],
                       facecolor=col_emb[int(r.embedding)], edgecolor="black", lw=0.4,
                       zorder=3)
    ax.axhspan(inc["mean"] - inc["sd"], inc["mean"] + inc["sd"], color=PLOT_PALETTE[3],
               alpha=0.25, lw=0, zorder=0)
    ax.axhline(inc["mean"], color=PLOT_PALETTE[3], lw=0.8, zorder=0)
    ax.text(len(order) - 0.6, inc["mean"] + inc["sd"] + 0.004,
            f"v9 long-budget arms (calm emb.), n={inc['n']} at {inc['budget_epochs']:,} ep",
            fontsize=5, color=PLOT_PALETTE[9], va="bottom", ha="right")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        [t[t.cell == c].cell_name.iloc[0].split("_", 1)[1].replace("_", " ") for c in order],
        rotation=90, fontsize=5,
    )
    ax.set_ylabel(f"val Pearson roll_max, epochs <= {budget:,}")
    ax.set_ylim(0, 0.2)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.grid(axis="y", lw=0.3, alpha=0.35)
    for lvl, lab in ((0, "random_1024"), (1, "prot_T5_all")):
        ax.scatter([], [], s=12, marker="o", facecolor=col_emb[lvl], edgecolor="black",
                   lw=0.4, label=lab)
    ax.scatter([], [], s=12, marker="o", facecolor="white", edgecolor="black", lw=0.4,
               label="seed 0")
    ax.scatter([], [], s=12, marker="s", facecolor="white", edgecolor="black", lw=0.4,
               label="seed 1")
    ax.legend(frameon=False, loc="lower right", fontsize=5, handlelength=1.0, borderpad=0.2,
              ncol=2, columnspacing=0.8)
    ax.set_title("sixteen cells, two seeds each", fontsize=6, pad=3)
    ax.text(-0.14, 1.04, "a", transform=ax.transAxes, fontsize=8, fontweight="bold")

    # (b) main effects and interactions with 2 se bars
    ax = axes[1]
    names, vals = [], []
    contrast = {"embedding": "ptt5 - rand", "trunk": "small - big",
                "readout": "linear - mlp", "weight_decay": "1e-4 - 1e-8"}
    for f, m in e_all["main"].items():
        names.append(f"{SHORT[f]}: {contrast[f]}")
        vals.append(m["effect"])
    for k, m in e_all["interaction"].items():
        a, b = k.split("x")
        names.append(f"{SHORT[a]} x {SHORT[b]}")
        vals.append(m["effect"])
    y = np.arange(len(names))[::-1]
    se = e_all["se_effect"]
    colors = [PLOT_PALETTE[1]] * 4 + [PLOT_PALETTE[2]] * 6
    ax.barh(y, vals, color=colors, edgecolor="black", lw=0.5, height=0.7)
    ax.errorbar(vals, y, xerr=2 * se, fmt="none", ecolor="black", elinewidth=0.6, capsize=1.5)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=5)
    ax.set_xlabel("effect on roll_max (bars: +/- 2 se)")
    ax.grid(axis="x", lw=0.3, alpha=0.35)
    ax.set_title(f"pooled within-cell sd {e_all['sd_pooled_within_cell']:.4f}",
                 fontsize=6, pad=3)
    ax.text(-0.5, 1.04, "b", transform=ax.transAxes, fontsize=8, fontweight="bold")

    # (c) every curve, colored by embedding
    ax = axes[2]
    for _, r in t.iterrows():
        h = curves[r.run_id]
        ax.plot(h.epoch, roll_mean(h[METRIC].to_numpy(), ROLL_WINDOW),
                color=col_emb[int(r.embedding)], lw=0.5, alpha=0.8)
    ax.axvline(budget, color="black", lw=0.5, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(10, 1500)
    ax.set_ylim(-0.02, 0.2)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"val Pearson, {ROLL_WINDOW}-epoch rolling mean")
    ax.grid(lw=0.3, alpha=0.35)
    ax.set_title("all 32 curves; dashed = matched budget", fontsize=6, pad=3)
    ax.text(-0.22, 1.04, "c", transform=ax.transAxes, fontsize=8, fontweight="bold")

    for a in axes:
        for s in a.spines.values():
            s.set_visible(True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    # Stable name, no timestamp: notes-tex/019-simb-multimodal's `make plots` converts
    # figures by name, and a timestamped file cannot be the target of that rule.
    stem = osp.join(IMAGE_DIR, "v10_grid_factorial")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    plt.close(fig)
    return stem


def main() -> None:
    api = wandb.Api(timeout=120)
    t, curves = fetch(api)
    budget = int(t.n_epochs.min())
    print(f"matched budget = min final epoch = {budget}; final epochs: "
          f"{sorted(t.n_epochs.unique().tolist())}")
    t = score(t, curves, budget)
    t = t.sort_values(["cell", "seed"]).reset_index(drop=True)
    csv_path = osp.join(RESULTS, "v10_grid_factorial.csv")
    t.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")

    cols = ["run_id", "cell", "cell_name", "seed", "n_epochs", "roll_max_matched",
            "epoch_at_matched", "roll_max_full", "pearson_last", "nmse_at_matched_peak",
            "loss_min_epoch", "chance_band"]
    with pd.option_context("display.width", 220, "display.max_rows", 100):
        print(t[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    e_all = effects(t, "roll_max_matched")
    print_effects(f"matched budget (epochs <= {budget}), all 32 runs", e_all)
    healthy = t[~t.chance_band].reset_index(drop=True)
    e_healthy = effects(healthy, "roll_max_matched")
    print_effects(f"matched budget, {len(healthy)} runs above the chance band", e_healthy)
    e_full = effects(t, "roll_max_full")
    print_effects("UNMATCHED full-run roll_max (budget differs by run; not for contrast)",
                  e_full)

    inc = incumbent_at(budget)
    print(f"\nincumbent replicate reference from short_budget_spread.json: "
          f"{inc['n']} runs at {inc['budget_epochs']} epochs, mean {inc['mean']:.4f} "
          f"sd {inc['sd']:.4f} (its curve was read at {inc['history_samples']} samples)")
    best_cell = max(e_all["cell_means"].items(), key=lambda kv: kv[1]["mean"])
    print(f"best cell by mean: c{best_cell[0]} "
          f"{t[t.cell == best_cell[0]].cell_name.iloc[0]} mean {best_cell[1]['mean']:.4f} "
          f"seeds {best_cell[1]['seeds']}")
    chance = t[t.chance_band]
    print(f"{len(chance)} run(s) never left the chance band (< {CHANCE_BAND}): "
          + ", ".join(f"{r.run_id} {r.cell_name} seed{r.seed}" for _, r in chance.iterrows()))

    stem = figure(t, curves, budget, e_all, inc)
    print(f"figure: {stem}.svg")

    summary = {
        "generated_by": "experiments/019-simb-multimodal/scripts/v10_grid_factorial.py",
        "project": PROJECT,
        "metric": METRIC,
        "roll_window": ROLL_WINDOW,
        "history_samples": FULL_HISTORY_SAMPLES,
        "n_runs": int(len(t)),
        "matched_budget_epochs": budget,
        "final_epochs": sorted(int(x) for x in t.n_epochs.unique()),
        "n_rows_short": int((t.rows_short_by > 0).sum()),
        "chance_band": CHANCE_BAND,
        "chance_band_runs": [
            {"run_id": r.run_id, "cell_name": r.cell_name, "seed": int(r.seed),
             "roll_max_matched": float(r.roll_max_matched)}
            for _, r in chance.iterrows()
        ],
        "incumbent_reference": inc,
        "matched_all": e_all,
        "matched_healthy": e_healthy,
        "unmatched_full": e_full,
        "figure": stem + ".svg",
    }
    json_path = osp.join(RESULTS, "v10_grid_factorial.json")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
