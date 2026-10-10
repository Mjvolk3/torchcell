# experiments/040-inhibitor-synergy-wetlab/scripts/mixture_rules.py
# [[experiments.040-inhibitor-synergy-wetlab.scripts.mixture_rules]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/mixture_rules
"""Reference mixture rules against the 2021 inhibitor combinations (claim 1 of 040).

INPUTS. ``results/wetlab_wells.csv`` (wetlab_table.py) and the ex21 / isobole-axis Hill
fits with their bootstrap draws (single_agent_curves.py, primary ``zero`` variant: a well
that did not grow is fitness 0).

GROWTH CALLS. Every ex23 score is computed twice: under the SERVED call (the loader's
raw-curve derivation, primary) and the SOFTWARE call (the Bioscreen software's
generation times, the source 039 used), each on its own fitness scale (ex23 control
generation time 1.345 h raw curve, 1.815 h software). The two disagree on 12 wells, all
in HMF combinations (``results/wetlab_growth_call_check.csv``).

RULES, each predicting the fitness of a combination on the ex23 scale (WT = 1):

- Bliss independence: the product of single-agent fractional fitness. ``bliss_ex21``
  takes each single from its ex21 Hill fit at the combination dose (f / fitted top);
  ``bliss_ex23`` takes the observed ex23 single (mean of 3 wells, no growth = 0).
- Loewe additivity (``loewe_ex21``): the effect E with sum_i d_i / D_i(E) = 1, D_i the
  ex21 Hill inverse. A component whose D_i(E) lies above ex21's highest tested dose is
  flagged (the single never reaches E within its measured range; the value is the
  Hill extrapolation).
- Highest single agent: the lowest single fractional fitness (``hsa_ex21``, ``hsa_ex23``).

SCORING over the 63 ex23 combinations (and the 57 with two or more): grew / no-grew with
a threshold derived from the data, tau = the lowest fitness of any grown inhibited ex23
well under that call (the slowest growth the call registers within 85 h); a rule
predicts growth when its prediction >= tau. Confusion counts and AUROC (prediction as
the score). Fitness over the combinations that grew (observed = mean of the grown
wells, 039's definition): Spearman and RMSE, overall and by number of inhibitors.

PAIRS. For the 15 pairs: observed (mean of 3 wells, no growth = 0) minus Bliss and the
log2 ratio (observed floored at tau for the log, flagged), with 95% intervals from 2000
bootstrap resamples of the replicate wells (pair and singles) and, for the ex21 variant,
the ex21 fit draws. Call = synergy if the interval of observed - Bliss(ex23 singles) is
below 0, antagonism if above, else additive. Keyed by the sorted compound pair, names as
the loader serves them (``results/pair_deviation.csv``).

ISOBOLES (served call only; ex27 and ex28 have no software traits). Empirical Bliss from
the grid's own axes, the excess over Bliss on the 81 interior cells (2000 resamples of
the two plate wells per cell), the Loewe additive front at the run's growth threshold
from the axis Hill fits, and counts of interior cells that contradict Loewe in each
direction. HMF x AA (ex28) is analyzed but flagged: one run, growth islands at HMF
2.0 g/L (039).
"""

from __future__ import annotations

import os
import os.path as osp
from itertools import combinations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import MultipleLocator
from pydantic import BaseModel
from scipy.optimize import brentq
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
EXPERIMENT = osp.dirname(osp.dirname(osp.abspath(__file__)))
RESULTS = osp.join(EXPERIMENT, "results")
IMAGES = osp.join(os.environ["ASSET_IMAGES_DIR"], "040-inhibitor-synergy-wetlab")

ABBRS = ["FF", "AA", "HMF", "FA", "LVA", "LA"]
NAMES = {
    "FF": "furfural",
    "AA": "acetic acid",
    "HMF": "5-(hydroxymethyl)furfural",
    "FA": "formic acid",
    "LVA": "levulinic acid",
    "LA": "lactic acid",
}
ABBR_OF = {v: k for k, v in NAMES.items()}
EX23_DOSE = {"FF": 1.5, "AA": 2.0, "HMF": 2.522, "FA": 1.0, "LVA": 6.0, "LA": 20.0}
CALLS = {
    "served": ("fitness", "grew"),
    "software": ("fitness_software", "grew_software"),
}
RULES = ["bliss_ex21", "bliss_ex23", "loewe_ex21", "hsa_ex21", "hsa_ex23"]
ISOBOLES = {"ex26": ("FF", 0.3333), "ex27": ("FA", 0.2), "ex28": ("HMF", 0.504)}
AA_STEP = 0.4
N_BOOT = 2000
SEED = 0
RED, DARK_RED, BLUE = PLOT_PALETTE[1], PLOT_PALETTE[7], PLOT_PALETTE[4]
FITNESS_CMAP = LinearSegmentedColormap.from_list(
    "fitness", [(0.0, DARK_RED), (0.55, RED), (1.0, "#FFFFFF")]
)
EXCESS_CMAP = LinearSegmentedColormap.from_list(
    "excess", [(0.0, DARK_RED), (0.5, "#FFFFFF"), (1.0, BLUE)]
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


class Hill(BaseModel):
    """f / top = 1 / (1 + (d / ic50)^h) and its inverse."""

    ic50: float
    h: float

    def fraction(self, d: float) -> float:
        """The fitness fraction f / top at dose ``d``."""
        return float(1.0 / (1.0 + (d / self.ic50) ** self.h))

    def log_dose_at(self, y: float) -> float:
        """The log dose log D(y) at which f / top = y, 0 < y < 1."""
        return self.log_dose_at_logit(float(np.log(y / (1.0 - y))))

    def log_dose_at_logit(self, z: float) -> float:
        """The log dose at logit(f / top) = z, exact for any z."""
        return float(np.log(self.ic50) - z / self.h)


def loewe(doses: dict[str, float], hills: dict[str, Hill]) -> float:
    """E in (0, 1) with sum d_i / D_i(E) = 1, solved on logit(E).

    The sum is monotone in E (D_i grows without bound as E -> 0 and vanishes as E -> 1),
    so the root is unique and lies in the bracket for any positive dose.
    """

    def g(z: float) -> float:
        return (
            sum(d * np.exp(-hills[a].log_dose_at_logit(z)) for a, d in doses.items())
            - 1.0
        )

    z = brentq(g, -60.0, 60.0)
    return float(1.0 / (1.0 + np.exp(-z)))


def combination_index(
    doses: dict[str, float], hills: dict[str, Hill], y: float
) -> float:
    """The combination index sum d_i / D_i(y): < 1 synergy, > 1 antagonism relative to Loewe."""
    return float(sum(d / np.exp(hills[a].log_dose_at(y)) for a, d in doses.items()))


def box(ax: plt.Axes) -> None:
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.5)


def tenth_grid(ax: plt.Axes, axis: str) -> None:
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_locator(MultipleLocator(0.2))
    target.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", axis=axis, length=0)
    ax.grid(axis=axis, which="both", linewidth=0.3, color="#DDDDDD")
    ax.set_axisbelow(True)


def save(fig: plt.Figure, name: str) -> list[str]:
    os.makedirs(IMAGES, exist_ok=True)
    stem = osp.join(IMAGES, f"{name}_{timestamp()}")
    savefig_true_size_svg(fig, stem + ".svg")
    fig.savefig(stem + ".png", dpi=300)
    plt.close(fig)
    return [stem + ".svg", stem + ".png"]


# --------------------------------------------------------------------------- #
# ex23
# --------------------------------------------------------------------------- #
def ex21_hills(fits: pd.DataFrame) -> tuple[dict[str, Hill], dict[str, float]]:
    """ex21 point fits (zero variant) and each compound's highest tested dose."""
    ex21 = fits[(fits["source"] == "ex21") & (fits["variant"] == "zero")].set_index(
        "inhibitor"
    )
    hills = {
        a: Hill(ic50=ex21.loc[a, "ic50_g_per_l"], h=ex21.loc[a, "h"]) for a in ABBRS
    }
    return hills, {a: float(ex21.loc[a, "max_dose_g_per_l"]) for a in ABBRS}


def ex23_wells(wells: pd.DataFrame, call: str) -> pd.DataFrame:
    """ex23 inhibited wells with ``y`` (no growth = 0) and ``g`` under one call."""
    fitness, grew = CALLS[call]
    w = wells[(wells["run"] == "ex23") & (wells["n_compounds"] > 0)].copy()
    w["g"] = w[grew].astype(bool)
    w["y_grown"] = w[fitness].where(w["g"])
    w["y"] = w["y_grown"].fillna(0.0)
    return w


def threshold(w: pd.DataFrame) -> float:
    """tau: the lowest fitness among grown wells."""
    return float(w.loc[w["g"], "y_grown"].min())


def predict(
    members: list[str],
    hills: dict[str, Hill],
    singles: dict[str, float],
    max_dose: dict[str, float],
) -> dict[str, float | bool | str]:
    """Every rule's prediction for one ex23 combination."""
    doses = {a: EX23_DOSE[a] for a in members}
    frac21 = [hills[a].fraction(EX23_DOSE[a]) for a in members]
    frac23 = [singles[a] for a in members]
    e = loewe(doses, hills)
    beyond = [a for a in members if np.exp(hills[a].log_dose_at(e)) > max_dose[a]]
    return {
        "bliss_ex21": float(np.prod(frac21)),
        "bliss_ex23": float(np.prod(frac23)),
        "loewe_ex21": e,
        "hsa_ex21": float(min(frac21)),
        "hsa_ex23": float(min(frac23)),
        "loewe_beyond_measured_range": bool(beyond),
        "loewe_components_beyond_range": "|".join(NAMES[a] for a in beyond),
    }


def combination_table(
    wells: pd.DataFrame, hills: dict[str, Hill], max_dose: dict[str, float]
) -> tuple[pd.DataFrame, dict[str, float]]:
    """One row per (combination, call)."""
    rows, taus = [], {}
    for call in CALLS:
        w = ex23_wells(wells, call)
        tau = threshold(w)
        taus[call] = tau
        singles = {
            ABBR_OF[c]: float(g["y"].mean())
            for c, g in w[w["n_compounds"] == 1].groupby("compounds")
        }
        for compounds, g in w.groupby("compounds"):
            members = [ABBR_OF[c] for c in compounds.split("|")]
            row = {
                "combination": compounds,
                "call": call,
                "n_compounds": len(members),
                "n_wells": len(g),
                "n_grew": int(g["g"].sum()),
                "grew": bool(g["g"].any()),
                "observed_grown_mean": float(g["y_grown"].mean()),
                "observed_zero_mean": float(g["y"].mean()),
                "tau": tau,
            } | predict(members, hills, singles, max_dose)
            doses = {a: EX23_DOSE[a] for a in members}
            if row["grew"]:
                y = float(np.clip(row["observed_grown_mean"], 1e-6, 1 - 1e-6))
                row["ci_observed"] = combination_index(doses, hills, y)
                row["ci_is_upper_bound"] = False
            else:
                row["ci_observed"] = combination_index(doses, hills, tau)
                row["ci_is_upper_bound"] = True
            rows.append(row)
    return pd.DataFrame(rows), taus


def score(table: pd.DataFrame) -> pd.DataFrame:
    """Growth classification and fitness agreement per rule, call and subset."""
    rows = []
    for call, t in table.groupby("call"):
        tau = float(t["tau"].iloc[0])
        subsets = {"all": t, "2+": t[t["n_compounds"] >= 2]} | {
            str(k): t[t["n_compounds"] == k] for k in range(1, 7)
        }
        for subset, s in subsets.items():
            grown = s[s["grew"]]
            for rule in RULES:
                pred_grew = s[rule] >= tau
                both = s["grew"].nunique() == 2
                rows.append(
                    {
                        "call": call,
                        "subset": subset,
                        "rule": rule,
                        "tau": tau,
                        "n_combinations": len(s),
                        "n_grew": int(s["grew"].sum()),
                        "tp": int((pred_grew & s["grew"]).sum()),
                        "fp": int((pred_grew & ~s["grew"]).sum()),
                        "tn": int((~pred_grew & ~s["grew"]).sum()),
                        "fn": int((~pred_grew & s["grew"]).sum()),
                        "accuracy": float((pred_grew == s["grew"]).mean()),
                        "auroc": float(roc_auc_score(s["grew"], s[rule]))
                        if both
                        else np.nan,
                        "n_fitness": len(grown),
                        "spearman": float(
                            spearmanr(
                                grown["observed_grown_mean"], grown[rule]
                            ).statistic
                        )
                        if len(grown) >= 3
                        else np.nan,
                        "rmse": float(
                            np.sqrt(
                                np.mean(
                                    (grown["observed_grown_mean"] - grown[rule]) ** 2
                                )
                            )
                        )
                        if len(grown)
                        else np.nan,
                        "mean_signed_error": float(
                            (grown["observed_grown_mean"] - grown[rule]).mean()
                        )
                        if len(grown)
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def pair_deviation(
    wells: pd.DataFrame,
    boot: pd.DataFrame,
    taus: dict[str, float],
    hills: dict[str, Hill],
) -> pd.DataFrame:
    """Observed minus Bliss for the 15 pairs, both calls, with bootstrap intervals."""
    rng = np.random.default_rng(SEED)
    b21 = {
        a: boot[(boot["source"] == "ex21") & (boot["inhibitor"] == a)].sort_values(
            "draw"
        )
        for a in ABBRS
    }
    n_draws = len(b21["FF"])
    rows = []
    for a, b in combinations(sorted(ABBRS, key=lambda x: NAMES[x]), 2):
        key = f"{NAMES[a]}|{NAMES[b]}"
        row: dict[str, object] = {
            "pair": key,
            "compound_a": NAMES[a],
            "compound_b": NAMES[b],
            "dose_a_g_per_l": EX23_DOSE[a],
            "dose_b_g_per_l": EX23_DOSE[b],
        }
        frac21 = {}
        for x in (a, b):
            d = b21[x]
            frac21[x] = 1.0 / (
                1.0 + (EX23_DOSE[x] / d["ic50"].to_numpy()) ** d["h"].to_numpy()
            )
        for call in CALLS:
            w = ex23_wells(wells, call)
            tau = taus[call]
            pair = w.loc[w["compounds"] == key, "y"].to_numpy()
            sa = w.loc[w["compounds"] == NAMES[a], "y"].to_numpy()
            sb = w.loc[w["compounds"] == NAMES[b], "y"].to_numpy()
            obs = pair.mean()
            bliss23 = sa.mean() * sb.mean()
            bliss21 = hills[a].fraction(EX23_DOSE[a]) * hills[b].fraction(EX23_DOSE[b])
            obs_b = pair[rng.integers(0, len(pair), size=(N_BOOT, len(pair)))].mean(1)
            sa_b = sa[rng.integers(0, len(sa), size=(N_BOOT, len(sa)))].mean(1)
            sb_b = sb[rng.integers(0, len(sb), size=(N_BOOT, len(sb)))].mean(1)
            draw = rng.integers(0, n_draws, N_BOOT)
            bliss21_b = frac21[a][draw] * frac21[b][draw]
            dev23 = obs_b - sa_b * sb_b
            dev21 = obs_b - bliss21_b
            floor = max(obs, tau)
            l23 = np.log2(np.maximum(obs_b, tau) / (sa_b * sb_b))
            l21 = np.log2(np.maximum(obs_b, tau) / bliss21_b)
            lo23, hi23 = np.percentile(dev23, [2.5, 97.5])
            call_label = (
                "synergy" if hi23 < 0 else "antagonism" if lo23 > 0 else "additive"
            )
            s = f"_{call}"
            row |= {
                f"n_pair_wells{s}": len(pair),
                f"n_pair_grew{s}": int(w.loc[w["compounds"] == key, "g"].sum()),
                f"observed{s}": obs,
                f"bliss_ex23{s}": bliss23,
                f"obs_minus_bliss_ex23{s}": obs - bliss23,
                f"obs_minus_bliss_ex23_lo{s}": lo23,
                f"obs_minus_bliss_ex23_hi{s}": hi23,
                f"log2_obs_over_bliss_ex23{s}": float(np.log2(floor / bliss23)),
                f"log2_obs_over_bliss_ex23_lo{s}": float(np.percentile(l23, 2.5)),
                f"log2_obs_over_bliss_ex23_hi{s}": float(np.percentile(l23, 97.5)),
                f"bliss_ex21{s}": bliss21,
                f"obs_minus_bliss_ex21{s}": obs - bliss21,
                f"obs_minus_bliss_ex21_lo{s}": float(np.percentile(dev21, 2.5)),
                f"obs_minus_bliss_ex21_hi{s}": float(np.percentile(dev21, 97.5)),
                f"log2_obs_over_bliss_ex21{s}": float(np.log2(floor / bliss21)),
                f"log2_obs_over_bliss_ex21_lo{s}": float(np.percentile(l21, 2.5)),
                f"log2_obs_over_bliss_ex21_hi{s}": float(np.percentile(l21, 97.5)),
                f"observed_floored_at_tau{s}": bool(obs < tau),
                f"tau{s}": tau,
                f"call_bliss_ex23{s}": call_label,
            }
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Isoboles
# --------------------------------------------------------------------------- #
class IsoboleGrid(BaseModel):
    """One isobole as (inhibitor rows) x (acetic acid columns) x (2 plates)."""

    run: str
    inhibitor: str
    x_doses: list[float]
    a_doses: list[float]
    y: list[list[list[float]]]  # [row][col][plate], no growth = 0
    g: list[list[list[bool]]]
    tau: float


def isobole_grid(wells: pd.DataFrame, run: str) -> IsoboleGrid:
    inhibitor, _ = ISOBOLES[run]
    w = wells[wells["run"] == run].copy()
    w["y"] = w["fitness"].fillna(0.0)
    xs = sorted(w[f"dose_g_per_l_{inhibitor}"].unique())
    aa = sorted(w["dose_g_per_l_AA"].unique())
    y = np.zeros((len(xs), len(aa), 2))
    g = np.zeros((len(xs), len(aa), 2), dtype=bool)
    for _, r in w.iterrows():
        i = xs.index(r[f"dose_g_per_l_{inhibitor}"])
        j = aa.index(r["dose_g_per_l_AA"])
        p = int(r["biological_replicate_id"]) - 1
        y[i, j, p], g[i, j, p] = r["y"], r["grew"]
    tau = float(w.loc[w["grew"] & (w["n_compounds"] > 0), "fitness"].min())
    return IsoboleGrid(
        run=run,
        inhibitor=inhibitor,
        x_doses=xs,
        a_doses=aa,
        y=y.tolist(),
        g=g.tolist(),
        tau=tau,
    )


def bliss_surface(y: np.ndarray) -> np.ndarray:
    """Empirical Bliss from the grid's own axes: y(x,0) y(0,a) / y(0,0)."""
    return np.outer(y[:, 0], y[0, :]) / y[0, 0]


def isobole_analysis(
    wells: pd.DataFrame, fits: pd.DataFrame, boot: pd.DataFrame
) -> tuple[list[dict[str, object]], pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """Per isobole: excess over Bliss, Loewe front and contradiction counts."""
    rng = np.random.default_rng(SEED)
    summary, cells, surfaces = [], [], {}
    for run in ISOBOLES:
        grid = isobole_grid(wells, run)
        y = np.asarray(grid.y)
        g = np.asarray(grid.g)
        mean = y.mean(axis=2)
        grew = g.any(axis=2)
        bliss = bliss_surface(mean)
        excess = mean - bliss
        interior = (slice(1, None), slice(1, None))
        boot_mean = []
        for _ in range(N_BOOT):
            pick = rng.integers(0, 2, size=mean.shape)
            m = np.take_along_axis(y, pick[..., None], axis=2)[..., 0]
            pick2 = rng.integers(0, 2, size=mean.shape)
            m = (m + np.take_along_axis(y, pick2[..., None], axis=2)[..., 0]) / 2.0
            boot_mean.append(float((m - bliss_surface(m))[interior].mean()))
        lo, hi = np.percentile(boot_mean, [2.5, 97.5])

        point = fits[(fits["source"] == run) & (fits["variant"] == "zero")].set_index(
            "inhibitor"
        )
        hx = Hill(
            ic50=point.loc[grid.inhibitor, "ic50_g_per_l"],
            h=point.loc[grid.inhibitor, "h"],
        )
        ha = Hill(ic50=point.loc["AA", "ic50_g_per_l"], h=point.loc["AA", "h"])

        def loewe_counts(hx: Hill, ha: Hill) -> tuple[np.ndarray, int, int]:
            pred = np.ones_like(mean)
            for i, xd in enumerate(grid.x_doses):
                for j, ad in enumerate(grid.a_doses):
                    doses = {k: v for k, v in (("X", xd), ("AA", ad)) if v > 0}
                    hills = {"X": hx, "AA": ha}
                    pred[i, j] = loewe(doses, hills) if doses else 1.0
            pg = pred >= grid.tau
            syn = int((pg & ~grew)[interior].sum())
            ant = int((~pg & grew)[interior].sum())
            return pred, syn, ant

        loewe_pred, syn, ant = loewe_counts(hx, ha)
        bx = boot[(boot["source"] == run) & (boot["inhibitor"] == grid.inhibitor)]
        ba = boot[(boot["source"] == run) & (boot["inhibitor"] == "AA")]
        draws = rng.integers(0, len(bx), 200)
        syn_b, ant_b = [], []
        for k in draws:
            _, s_k, a_k = loewe_counts(
                Hill(ic50=bx["ic50"].iloc[k], h=bx["h"].iloc[k]),
                Hill(ic50=ba["ic50"].iloc[k], h=ba["h"].iloc[k]),
            )
            syn_b.append(s_k)
            ant_b.append(a_k)
        dx = float(np.exp(hx.log_dose_at(grid.tau)))
        da = float(np.exp(ha.log_dose_at(grid.tau)))
        n_int = int(np.prod(mean[interior].shape))
        call = "synergy" if hi < 0 else "antagonism" if lo > 0 else "additive"
        summary.append(
            {
                "run": run,
                "pair": f"{NAMES['AA']}|{NAMES[grid.inhibitor]}",
                "inhibitor": NAMES[grid.inhibitor],
                "tau": grid.tau,
                "n_interior_cells": n_int,
                "n_interior_grew": int(grew[interior].sum()),
                "mean_excess_over_bliss": float(excess[interior].mean()),
                "mean_excess_lo": float(lo),
                "mean_excess_hi": float(hi),
                "n_interior_bliss_grow_observed_none": int(
                    ((bliss >= grid.tau) & ~grew)[interior].sum()
                ),
                "n_interior_bliss_none_observed_grew": int(
                    ((bliss < grid.tau) & grew)[interior].sum()
                ),
                "loewe_dose_at_tau_inhibitor_g_per_l": dx,
                "loewe_dose_at_tau_aa_g_per_l": da,
                "loewe_front_beyond_grid": bool(
                    dx > max(grid.x_doses) or da > max(grid.a_doses)
                ),
                "n_interior_loewe_grow_observed_none": syn,
                "loewe_grow_observed_none_lo": float(np.percentile(syn_b, 2.5)),
                "loewe_grow_observed_none_hi": float(np.percentile(syn_b, 97.5)),
                "n_interior_loewe_none_observed_grew": ant,
                "loewe_none_observed_grew_lo": float(np.percentile(ant_b, 2.5)),
                "loewe_none_observed_grew_hi": float(np.percentile(ant_b, 97.5)),
                "call_bliss": call,
                "flag": "single run with growth islands at HMF 2.0 g/L (039); "
                "analyzed, not trusted"
                if run == "ex28"
                else "",
            }
        )
        for i, xd in enumerate(grid.x_doses):
            for j, ad in enumerate(grid.a_doses):
                cells.append(
                    {
                        "run": run,
                        "inhibitor_g_per_l": xd,
                        "acetic_acid_g_per_l": ad,
                        "observed_mean": mean[i, j],
                        "grew_any_plate": bool(grew[i, j]),
                        "bliss": bliss[i, j],
                        "excess_over_bliss": excess[i, j],
                        "loewe": loewe_pred[i, j],
                    }
                )
        surfaces[run] = {
            "mean": mean,
            "grew": grew,
            "excess": excess,
            "x": np.asarray(grid.x_doses),
            "a": np.asarray(grid.a_doses),
            "dx": np.array(dx),
            "da": np.array(da),
        }
    return summary, pd.DataFrame(cells), surfaces


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_rules(table: pd.DataFrame) -> list[str]:
    shown = ["bliss_ex21", "bliss_ex23", "loewe_ex21", "hsa_ex21"]
    titles = {
        "bliss_ex21": "Bliss, ex21 fits",
        "bliss_ex23": "Bliss, ex23 singles",
        "loewe_ex21": "Loewe, ex21 fits",
        "hsa_ex21": "highest single agent, ex21 fits",
    }
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(95)),
        sharex=True,
        sharey=True,
    )
    for r, call in enumerate(CALLS):
        t = table[table["call"] == call]
        tau = float(t["tau"].iloc[0])
        for c, rule in enumerate(shown):
            ax = axes[r, c]
            for k in range(1, 7):
                s = t[t["n_compounds"] == k]
                grown = s[s["grew"]]
                ax.scatter(
                    grown[rule],
                    grown["observed_grown_mean"],
                    s=8,
                    color=PLOT_PALETTE[k - 1],
                    edgecolor="black",
                    linewidth=0.3,
                    label=f"{k} inhibitor{'s' if k > 1 else ''}",
                    zorder=3,
                )
                none = s[~s["grew"]]
                ax.scatter(
                    none[rule],
                    np.zeros(len(none)),
                    s=12,
                    marker="x",
                    color=PLOT_PALETTE[k - 1],
                    linewidth=0.7,
                    zorder=3,
                )
            ax.plot([0, 1.1], [0, 1.1], color="black", linewidth=0.5)
            ax.axhline(tau, color="#888888", linewidth=0.5, linestyle="--")
            ax.axvline(tau, color="#888888", linewidth=0.5, linestyle="--")
            ax.set_xlim(-0.03, 1.1)
            ax.set_ylim(-0.05, 1.1)
            tenth_grid(ax, "x")
            tenth_grid(ax, "y")
            ax.set_title(f"{titles[rule]} ({call} call)")
            if r == 1:
                ax.set_xlabel("predicted fitness")
            if c == 0:
                ax.set_ylabel("observed fitness (mean of grown wells)")
            box(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(
        plt.Line2D([], [], marker="x", linestyle="none", color="black", markersize=3)
    )
    labels.append("no growth within 85 h (drawn at 0)")
    handles.append(plt.Line2D([], [], linestyle="--", color="#888888", linewidth=0.5))
    labels.append("growth threshold tau")
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(
        left=0.06, right=0.99, top=0.95, bottom=0.18, hspace=0.3, wspace=0.1
    )
    return save(fig, "mixture_rules_observed_vs_predicted")


def plot_isoboles(
    surfaces: dict[str, dict[str, np.ndarray]], summary: list[dict[str, object]]
) -> list[str]:
    fig, axes = plt.subplots(
        2, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(120))
    )
    by_run = {s["run"]: s for s in summary}
    for c, run in enumerate(ISOBOLES):
        s = surfaces[run]
        inhibitor, step = ISOBOLES[run]
        for r, (key, cmap, norm) in enumerate(
            (
                ("mean", FITNESS_CMAP, None),
                ("excess", EXCESS_CMAP, TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=0.5)),
            )
        ):
            ax = axes[r, c]
            im = ax.imshow(
                s[key],
                cmap=cmap,
                origin="lower",
                **({"vmin": 0, "vmax": 1} if norm is None else {"norm": norm}),
            )
            grew = s["grew"]
            for i in range(grew.shape[0]):
                for j in range(grew.shape[1]):
                    if not grew[i, j]:
                        continue
                    for di, dj, xs, ys in (
                        (-1, 0, (j - 0.5, j + 0.5), (i - 0.5, i - 0.5)),
                        (1, 0, (j - 0.5, j + 0.5), (i + 0.5, i + 0.5)),
                        (0, -1, (j - 0.5, j - 0.5), (i - 0.5, i + 0.5)),
                        (0, 1, (j + 0.5, j + 0.5), (i - 0.5, i + 0.5)),
                    ):
                        ni, nj = i + di, j + dj
                        inside = 0 <= ni < grew.shape[0] and 0 <= nj < grew.shape[1]
                        if inside and grew[ni, nj]:
                            continue
                        ax.plot(xs, ys, color="black", linewidth=0.9)
            # Loewe additive front at tau: x / Dx + a / Da = 1, in grid-index units
            ax.plot(
                [0, float(s["da"]) / AA_STEP],
                [float(s["dx"]) / step, 0],
                color=BLUE if r == 0 else "black",
                linewidth=1.0,
                linestyle="--",
            )
            ax.set_xlim(-0.5, 9.5)
            ax.set_ylim(-0.5, 9.5)
            ax.set_xticks(range(10))
            ax.set_yticks(range(10))
            ax.set_xticklabels([f"{v:g}" for v in s["a"]], rotation=60, fontsize=5)
            ax.set_yticklabels([f"{v:g}" for v in s["x"]], fontsize=5)
            ax.set_xlabel("acetic acid (g/L)")
            ax.set_ylabel(f"{NAMES[inhibitor]} (g/L)")
            sm = by_run[run]
            if r == 0:
                flag = " (flagged)" if run == "ex28" else ""
                ax.set_title(f"{run}{flag}: observed fitness")
            else:
                ax.set_title(
                    f"excess over Bliss, mean {sm['mean_excess_over_bliss']:.2f} "
                    f"[{sm['mean_excess_lo']:.2f}, {sm['mean_excess_hi']:.2f}]"
                )
            box(ax)
            if c == 2:
                cax = fig.add_axes([0.93, 0.56 if r == 0 else 0.1, 0.01, 0.33])
                cbar = fig.colorbar(im, cax=cax)
                cbar.set_label(
                    "fitness (no growth = 0)" if r == 0 else "observed - Bliss"
                )
                cbar.outline.set_linewidth(0.5)
    fig.subplots_adjust(
        left=0.07, right=0.9, top=0.95, bottom=0.1, hspace=0.45, wspace=0.45
    )
    return save(fig, "mixture_rules_isoboles")


def plot_pairs(pairs: pd.DataFrame) -> list[str]:
    fig, axes = plt.subplots(
        1, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(75)), sharey=True
    )
    labels = [
        p.replace("5-(hydroxymethyl)furfural", "5-HMF").replace("|", " + ")
        for p in pairs["pair"]
    ]
    y = np.arange(len(pairs))
    for ax, call in zip(axes, CALLS, strict=True):
        for offset, variant, color in (
            (-0.15, "ex23", PLOT_PALETTE[0]),
            (0.15, "ex21", PLOT_PALETTE[2]),
        ):
            mid = pairs[f"obs_minus_bliss_{variant}_{call}"]
            lo = pairs[f"obs_minus_bliss_{variant}_lo_{call}"]
            hi = pairs[f"obs_minus_bliss_{variant}_hi_{call}"]
            ax.errorbar(
                mid,
                y + offset,
                xerr=[mid - lo, hi - mid],
                fmt="o",
                markersize=2.5,
                color=color,
                markeredgecolor="black",
                markeredgewidth=0.3,
                elinewidth=0.6,
                capsize=1,
                label=f"Bliss from {'ex23 singles' if variant == 'ex23' else 'ex21 fits'}",
            )
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=5)
        ax.set_xlabel("observed - Bliss (fitness; no growth = 0)")
        ax.set_title(f"ex23 pairs, {call} call (95% bootstrap interval)")
        ax.grid(axis="x", linewidth=0.3, color="#DDDDDD")
        ax.set_axisbelow(True)
        box(ax)
    axes[0].legend(frameon=False, loc="lower left")
    fig.subplots_adjust(left=0.2, right=0.99, top=0.93, bottom=0.12, wspace=0.08)
    return save(fig, "mixture_rules_pair_deviation")


def main() -> None:
    wells = pd.read_csv(osp.join(RESULTS, "wetlab_wells.csv"))
    fits = pd.read_csv(osp.join(RESULTS, "single_agent_fits.csv"))
    boot = pd.read_csv(osp.join(RESULTS, "single_agent_bootstrap.csv"))
    hills, max_dose = ex21_hills(fits)

    table, taus = combination_table(wells, hills, max_dose)
    table.to_csv(osp.join(RESULTS, "mixture_combinations.csv"), index=False)
    print(f"growth thresholds tau (lowest grown inhibited ex23 fitness): {taus}")
    scores = score(table)
    scores.to_csv(osp.join(RESULTS, "mixture_scores.csv"), index=False)
    cols = [
        "call",
        "subset",
        "rule",
        "n_combinations",
        "n_grew",
        "tp",
        "fp",
        "tn",
        "fn",
        "accuracy",
        "auroc",
        "n_fitness",
        "spearman",
        "rmse",
        "mean_signed_error",
    ]
    print(
        scores[scores["subset"].isin(["all", "2+", "2", "3"])][cols]
        .round(3)
        .to_string()
    )
    beyond = table[(table["call"] == "served") & table["loewe_beyond_measured_range"]]
    print(
        f"Loewe E beyond a component's measured ex21 range: {len(beyond)} of 63 "
        f"combinations ({sorted(set('|'.join(beyond['loewe_components_beyond_range']).split('|')) - {''})})"
    )

    pairs = pair_deviation(wells, boot, taus, hills)
    pairs.to_csv(osp.join(RESULTS, "pair_deviation.csv"), index=False)
    show = ["pair"] + [
        f"{c}_{call}"
        for call in CALLS
        for c in (
            "n_pair_grew",
            "observed",
            "bliss_ex23",
            "obs_minus_bliss_ex23",
            "obs_minus_bliss_ex23_lo",
            "obs_minus_bliss_ex23_hi",
            "obs_minus_bliss_ex21",
            "call_bliss_ex23",
        )
    ]
    print(pairs[show].round(3).to_string())

    summary, cells, surfaces = isobole_analysis(wells, fits, boot)
    pd.DataFrame(summary).to_csv(osp.join(RESULTS, "isobole_summary.csv"), index=False)
    cells.to_csv(osp.join(RESULTS, "isobole_cells.csv"), index=False)
    print(pd.DataFrame(summary).drop(columns="flag").round(3).T.to_string())

    written = plot_rules(table) + plot_isoboles(surfaces, summary) + plot_pairs(pairs)
    print("\n".join(written))


if __name__ == "__main__":
    main()
