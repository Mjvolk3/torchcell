# experiments/040-inhibitor-synergy-wetlab/scripts/rule_choice.py
# [[experiments.040-inhibitor-synergy-wetlab.scripts.rule_choice]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/rule_choice
"""Which rule draws a pair's alive / dead boundary, and can it be predicted.

THE QUESTION. On the three isobole grids no single rule draws the growth boundary: the
two aldehyde grids follow Bliss and the acid grid follows Loewe (``mixture_rules.py``,
``results/isobole_boundary.csv``). The 15 fixed-dose ex23 pairs show the same split.
Bliss multiplies surviving fractions (independent hits); Loewe adds doses (a shared
capacity). So the boundary of a pair is set by its two single-agent curves plus one
choice, which rule, and the thing a learned model could add is that choice.

INPUTS. ``results/mixture_combinations.csv`` (served call: observed fitness with no
growth as 0, Bliss and Loewe from the ex21 Hill fits), ``results/profile_similarity.csv``
(per-pair chemogenomic profile similarity, ``best_available`` and ``all_predicted``
versions, from inhibitor_profiles.py), ``results/isobole_boundary.csv`` and the ex21
Hill fits (``results/single_agent_fits.csv``).

PAIRS. Each pair is labeled by chemical class: ``acid|acid`` (acetic, formic, lactic,
levulinic), ``aldehyde|aldehyde`` (furfural, 5-HMF) or ``aldehyde|acid``. The closer
rule is the one with the smaller absolute error against the observed pair fitness.

PREDICTING THE CHOICE, leave-one-pair-out over the 15 pairs. A predictor gives a
Loewe weight w in [0, 1] and the pair's fitness is w Loewe + (1 - w) Bliss:

- ``bliss`` and ``loewe``: w fixed at 0 and 1 (the model-free references).
- ``global_w``: one w, least squares on the 14 training pairs.
- ``class_rule``: w = 1 for a same-class pair, 0 across classes, no fitting (the
  textbook assignment: dose addition for a shared mechanism, independence otherwise).
- ``class_w``: w per class group (same class, across classes), least squares on the
  training pairs; the fitted version of the class rule.
- ``similarity_<version>_<measure>``: w = clip(a + b s, 0, 1) with s the pair's
  profile similarity, a and b least squares on the training pairs. This is the test
  claim 2 should have been: does chemogenomic similarity select the rule.

Scored on the 15 held-out predictions: mean absolute error against observed fitness,
growth call accuracy (prediction >= tau against grew), Spearman over the 15, and
Spearman over the 12 pairs that grew. With 15 pairs and 3 grids these are small-sample
readouts and are reported as such.

HIGHER ORDERS. The class rule extended to all 63 combinations: Loewe within each
class group, Bliss across the groups (``class_hierarchy``). Scored like
``mixture_rules.score`` by number of inhibitors, beside Bliss and Loewe. This is the
no-parameter hierarchical rule; it is reported whether or not it beats Loewe.

Outputs: ``results/rule_choice_pairs.csv``, ``results/rule_choice_lopo.csv``,
``results/rule_choice_grids.csv``, ``results/rule_choice_higher_orders.csv``, and one
figure in ``$ASSET_IMAGES_DIR/040-inhibitor-synergy-wetlab/``.

Run from the repository root:

    python experiments/040-inhibitor-synergy-wetlab/scripts/rule_choice.py
"""

from __future__ import annotations

import os
import os.path as osp
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import mixture_rules as mr  # noqa: E402

load_dotenv()
EXPERIMENT = osp.dirname(osp.dirname(osp.abspath(__file__)))
RESULTS = osp.join(EXPERIMENT, "results")
IMAGES = osp.join(os.environ["ASSET_IMAGES_DIR"], "040-inhibitor-synergy-wetlab")

CLASS_OF = {
    "AA": "acid",
    "FA": "acid",
    "LVA": "acid",
    "LA": "acid",
    "FF": "aldehyde",
    "HMF": "aldehyde",
}
SIMILARITY_MEASURES = ("spearman_centered", "spearman_raw", "jaccard_top100_centered")
W_GRID = np.linspace(0.0, 1.0, 101)
ORANGE, RED, PURPLE, BLUE, GRAY = (
    PLOT_PALETTE[0],
    PLOT_PALETTE[1],
    PLOT_PALETTE[2],
    PLOT_PALETTE[4],
    PLOT_PALETTE[5],
)

matplotlib.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 6,
        "axes.labelsize": 6,
        "axes.titlesize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.5,
        "svg.fonttype": "none",
    }
)


class PairRow(BaseModel):
    """One ex23 pair with its observed fitness, the two rules and its class."""

    pair: str
    members: list[str]
    pair_class: str
    same_class: bool
    grew: bool
    observed: float
    bliss: float
    loewe: float
    tau: float


def pair_class(members: list[str]) -> str:
    classes = sorted(CLASS_OF[m] for m in members)
    return "|".join(classes)


def load_pairs() -> list[PairRow]:
    c = pd.read_csv(osp.join(RESULTS, "mixture_combinations.csv"))
    c = c[(c["call"] == "served") & (c["n_compounds"] == 2)]
    rows = []
    for _, r in c.iterrows():
        members = [mr.ABBR_OF[x] for x in r["combination"].split("|")]
        rows.append(
            PairRow(
                pair=r["combination"],
                members=members,
                pair_class=pair_class(members),
                same_class=len({CLASS_OF[m] for m in members}) == 1,
                grew=bool(r["grew"]),
                observed=float(r["observed_zero_mean"]),
                bliss=float(r["bliss_ex21"]),
                loewe=float(r["loewe_ex21"]),
                tau=float(r["tau"]),
            )
        )
    assert len(rows) == 15, f"{len(rows)} pairs, not 15"
    return rows


def similarity_table() -> pd.DataFrame:
    s = pd.read_csv(osp.join(RESULTS, "profile_similarity.csv"))
    return s.set_index(["version", "pair"])


def mixed(rows: list[PairRow], w: np.ndarray) -> np.ndarray:
    loewe = np.array([r.loewe for r in rows])
    bliss = np.array([r.bliss for r in rows])
    return w * loewe + (1.0 - w) * bliss


def fit_w(rows: list[PairRow]) -> float:
    """The w on the grid with the least squared error against observed fitness."""
    y = np.array([r.observed for r in rows])
    errors = [((mixed(rows, np.full(len(rows), w)) - y) ** 2).sum() for w in W_GRID]
    return float(W_GRID[int(np.argmin(errors))])


def fit_similarity(rows: list[PairRow], s: np.ndarray) -> tuple[float, float]:
    """a, b with w = clip(a + b s) by least squares on a coarse grid, then refined."""
    y = np.array([r.observed for r in rows])

    def loss(a: float, b: float) -> float:
        w = np.clip(a + b * s, 0.0, 1.0)
        return float(((mixed(rows, w) - y) ** 2).sum())

    best = (np.inf, 0.0, 0.0)
    for a in np.linspace(-2.0, 2.0, 81):
        for b in np.linspace(-4.0, 4.0, 161):
            value = loss(a, b)
            if value < best[0]:
                best = (value, float(a), float(b))
    return best[1], best[2]


def predictors(
    train: list[PairRow], test: PairRow, sims: pd.DataFrame
) -> dict[str, float]:
    """Every predictor's w for the held-out pair, fitted on the training pairs."""
    out = {"bliss": 0.0, "loewe": 1.0, "global_w": fit_w(train)}
    out["class_rule"] = 1.0 if test.same_class else 0.0
    same = [r for r in train if r.same_class]
    across = [r for r in train if not r.same_class]
    group = same if test.same_class else across
    out["class_w"] = fit_w(group) if group else out["global_w"]
    for version in ("best_available", "all_predicted"):
        for measure in SIMILARITY_MEASURES:
            s_train = np.array([sims.loc[(version, r.pair), measure] for r in train])
            s_test = float(sims.loc[(version, test.pair), measure])
            a, b = fit_similarity(train, s_train)
            out[f"similarity_{version}_{measure}"] = float(
                np.clip(a + b * s_test, 0.0, 1.0)
            )
    return out


def leave_one_pair_out(
    rows: list[PairRow], sims: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Held-out predictions per pair and predictor, and the summary per predictor."""
    held = []
    for k, test in enumerate(rows):
        train = [r for i, r in enumerate(rows) if i != k]
        for name, w in predictors(train, test, sims).items():
            pred = w * test.loewe + (1.0 - w) * test.bliss
            held.append(
                {
                    "pair": test.pair,
                    "pair_class": test.pair_class,
                    "predictor": name,
                    "w_loewe": w,
                    "predicted": pred,
                    "observed": test.observed,
                    "grew": test.grew,
                    "tau": test.tau,
                    "abs_error": abs(pred - test.observed),
                }
            )
    held_df = pd.DataFrame(held)
    summary = []
    for name, h in held_df.groupby("predictor", sort=False):
        grown = h[h["grew"]]
        summary.append(
            {
                "predictor": name,
                "n_pairs": len(h),
                "mae": float(h["abs_error"].mean()),
                "growth_accuracy": float(
                    ((h["predicted"] >= h["tau"]) == h["grew"]).mean()
                ),
                "growth_auroc": float(roc_auc_score(h["grew"], h["predicted"])),
                "spearman_all": float(
                    spearmanr(h["observed"], h["predicted"]).statistic
                ),
                "spearman_grown": float(
                    spearmanr(grown["observed"], grown["predicted"]).statistic
                ),
                "n_grown": int(len(grown)),
                "mean_w": float(h["w_loewe"].mean()),
            }
        )
    return held_df, pd.DataFrame(summary)


def pair_table(rows: list[PairRow], sims: pd.DataFrame) -> pd.DataFrame:
    out = []
    for r in rows:
        eb, el = abs(r.bliss - r.observed), abs(r.loewe - r.observed)
        out.append(
            {
                "pair": r.pair,
                "pair_class": r.pair_class,
                "same_class": r.same_class,
                "grew": r.grew,
                "observed": r.observed,
                "bliss_ex21": r.bliss,
                "loewe_ex21": r.loewe,
                "abs_error_bliss": eb,
                "abs_error_loewe": el,
                "closer_rule": "bliss" if eb < el else "loewe",
                "both_predict_growth_observed_none": bool(
                    (not r.grew) and min(r.bliss, r.loewe) >= r.tau
                ),
            }
            | {
                f"{measure}_{version}": float(sims.loc[(version, r.pair), measure])
                for version in ("best_available", "all_predicted")
                for measure in SIMILARITY_MEASURES
            }
        )
    return pd.DataFrame(out)


def grid_table() -> pd.DataFrame:
    b = pd.read_csv(osp.join(RESULTS, "isobole_boundary.csv"))
    b = b[b["observed"] == "monotone"]
    wide = b.pivot(index=["run", "pair"], columns="rule", values="jaccard_alive")
    wide = wide.reset_index()
    wide["closer_rule"] = np.where(wide["loewe"] > wide["bliss"], "loewe", "bliss")
    wide["pair_class"] = [
        pair_class([mr.ABBR_OF[x] for x in p.split("|")]) for p in wide["pair"]
    ]
    return wide.rename(columns={"bliss": "jaccard_bliss", "loewe": "jaccard_loewe"})


def class_hierarchy(members: list[str], hills: dict[str, mr.Hill]) -> float:
    """Loewe within each class group, Bliss across groups."""
    value = 1.0
    for cls in ("acid", "aldehyde"):
        group = [m for m in members if CLASS_OF[m] == cls]
        if group:
            value *= mr.loewe({m: mr.EX23_DOSE[m] for m in group}, hills)
    return value


def higher_orders(hills: dict[str, mr.Hill]) -> pd.DataFrame:
    c = pd.read_csv(osp.join(RESULTS, "mixture_combinations.csv"))
    c = c[c["call"] == "served"].copy()
    c["class_hierarchy"] = [
        class_hierarchy([mr.ABBR_OF[x] for x in p.split("|")], hills)
        for p in c["combination"]
    ]
    tau = float(c["tau"].iloc[0])
    rows = []
    subsets = {"all": c, "2+": c[c["n_compounds"] >= 2]} | {
        str(k): c[c["n_compounds"] == k] for k in range(2, 5)
    }
    for subset, s in subsets.items():
        grown = s[s["grew"]]
        for rule in ("bliss_ex21", "loewe_ex21", "class_hierarchy"):
            both = s["grew"].nunique() == 2
            rows.append(
                {
                    "subset": subset,
                    "rule": rule,
                    "n_combinations": len(s),
                    "n_grew": int(s["grew"].sum()),
                    "accuracy": float(((s[rule] >= tau) == s["grew"]).mean()),
                    "auroc": float(roc_auc_score(s["grew"], s[rule]))
                    if both
                    else np.nan,
                    "worst_scenario_spearman": float(
                        spearmanr(s["observed_zero_mean"], s[rule]).statistic
                    )
                    if s["observed_zero_mean"].nunique() > 1
                    else np.nan,
                    "spearman_grown": float(
                        spearmanr(grown["observed_grown_mean"], grown[rule]).statistic
                    )
                    if len(grown) >= 3
                    else np.nan,
                    "rmse_grown": float(
                        np.sqrt(
                            ((grown["observed_grown_mean"] - grown[rule]) ** 2).mean()
                        )
                    )
                    if len(grown)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def figure(pairs: pd.DataFrame, lopo: pd.DataFrame) -> list[str]:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(55)),
        gridspec_kw={"width_ratios": [1.6, 1.0]},
    )
    order = {"acid|acid": 0, "acid|aldehyde": 1, "aldehyde|aldehyde": 2}
    p = pairs.assign(k=pairs["pair_class"].map(order)).sort_values(
        ["k", "observed"], ascending=[True, False]
    )
    x = np.arange(len(p))
    ax = axes[0]
    ax.scatter(
        x,
        p["bliss_ex21"],
        marker="s",
        s=14,
        color=ORANGE,
        label="Bliss, ex21 fits",
        zorder=3,
    )
    ax.scatter(
        x,
        p["loewe_ex21"],
        marker="^",
        s=14,
        color=BLUE,
        label="Loewe, ex21 fits",
        zorder=3,
    )
    ax.scatter(
        x,
        p["observed"],
        marker="o",
        s=16,
        facecolor="black",
        edgecolor="black",
        label="observed (no growth = 0)",
        zorder=4,
    )
    for xi, (_, r) in zip(x, p.iterrows(), strict=True):
        ax.plot(
            [xi, xi],
            [r["bliss_ex21"], r["loewe_ex21"]],
            color=GRAY,
            linewidth=0.5,
            zorder=1,
        )
    ax.axhline(float(pairs["tau"].iloc[0]), color="black", linewidth=0.4, linestyle=":")
    labels = [
        f"{mr.ABBR_OF[a]}+{mr.ABBR_OF[b]}" for a, b in (s.split("|") for s in p["pair"])
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylim(-0.03, 1.15)
    ax.set_ylabel("pair fitness at the ex23 doses")
    bounds = p.groupby("k")["pair"].count().cumsum().tolist()
    starts = [0] + bounds[:-1]
    names = ["acid + acid", "aldehyde + acid", "aldehyde + aldehyde"]
    for s0, e0, name in zip(starts, bounds, names, strict=True):
        ax.axvspan(
            s0 - 0.5,
            e0 - 0.5,
            color=GRAY,
            alpha=0.08 if name != "aldehyde + acid" else 0.0,
            linewidth=0,
        )
        ax.text((s0 + e0 - 1) / 2, 1.04, name, ha="center", va="bottom")
    handles, labels = ax.get_legend_handles_labels()
    ax.set_title("which rule a pair follows splits by chemical class")
    mr.box(ax)
    mr.tenth_grid(ax, "y")

    ax = axes[1]
    show = lopo[
        lopo["predictor"].isin(
            [
                "bliss",
                "loewe",
                "global_w",
                "class_rule",
                "class_w",
                "similarity_best_available_spearman_centered",
                "similarity_all_predicted_spearman_centered",
            ]
        )
    ].copy()
    short = {
        "bliss": "Bliss",
        "loewe": "Loewe",
        "global_w": "one fitted w",
        "class_rule": "class rule",
        "class_w": "class w, fitted",
        "similarity_best_available_spearman_centered": "profile similarity,\nbest available",
        "similarity_all_predicted_spearman_centered": "profile similarity,\nall predicted",
    }
    y = np.arange(len(show))
    ax.barh(
        y,
        show["mae"],
        color=[GRAY, GRAY, PURPLE, RED, RED, ORANGE, ORANGE],
        edgecolor="black",
        linewidth=0.5,
        height=0.65,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([short[n] for n in show["predictor"]])
    ax.invert_yaxis()
    ax.set_xlabel("mean absolute error, 15 held-out pairs")
    ax.set_title("leave-one-pair-out: error of each rule-choice predictor")
    for yi, (_, r) in zip(y, show.iterrows(), strict=True):
        ax.text(r["mae"] + 0.005, yi, f"rank {r['spearman_grown']:.2f}", va="center")
    ax.set_xlim(0, float(show["mae"].max()) * 1.35)
    mr.box(ax)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.36, wspace=0.55)
    mr.panel_letters(fig, axes)
    os.makedirs(IMAGES, exist_ok=True)
    stem = osp.join(IMAGES, f"rule_choice_{timestamp()}")
    savefig_true_size_svg(fig, stem + ".svg")
    fig.savefig(stem + ".png", dpi=300)
    plt.close(fig)
    return [stem + ".svg", stem + ".png"]


def main() -> None:
    rows = load_pairs()
    sims = similarity_table()
    fits = pd.read_csv(osp.join(RESULTS, "single_agent_fits.csv"))
    hills, _ = mr.ex21_hills(fits)

    pairs = pair_table(rows, sims)
    pairs["tau"] = rows[0].tau
    pairs.to_csv(osp.join(RESULTS, "rule_choice_pairs.csv"), index=False)
    print(
        pairs[
            [
                "pair",
                "pair_class",
                "grew",
                "observed",
                "bliss_ex21",
                "loewe_ex21",
                "closer_rule",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    print(pairs.groupby("pair_class")["closer_rule"].value_counts().to_string())

    held, lopo = leave_one_pair_out(rows, sims)
    held.to_csv(osp.join(RESULTS, "rule_choice_lopo_pairs.csv"), index=False)
    lopo.to_csv(osp.join(RESULTS, "rule_choice_lopo.csv"), index=False)
    print(lopo.round(3).to_string(index=False))

    grids = grid_table()
    grids.to_csv(osp.join(RESULTS, "rule_choice_grids.csv"), index=False)
    print(grids.round(3).to_string(index=False))

    orders = higher_orders(hills)
    orders.to_csv(osp.join(RESULTS, "rule_choice_higher_orders.csv"), index=False)
    print(orders.round(3).to_string(index=False))

    print("\n".join(figure(pairs, lopo)))


if __name__ == "__main__":
    main()
