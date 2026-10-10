# experiments/040-inhibitor-synergy-wetlab/scripts/similarity_vs_deviation.py
# [[experiments.040-inhibitor-synergy-wetlab.scripts.similarity_vs_deviation]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/similarity_vs_deviation
"""Claim 2 of 040: does chemogenomic profile similarity anticipate which inhibitor pairs
deviate from Bliss independence?

INPUTS. ``results/pair_deviation.csv`` (mixture_rules.py: observed minus Bliss from the
ex23 singles, both growth calls, 15 pairs) and ``results/profile_similarity.csv``
(inhibitor_profiles.py: Spearman across genes and Jaccard of the 100 most sensitive genes,
for the ``best_available`` and ``all_predicted`` profile matrices), plus
``results/isobole_summary.csv`` for the three dense grids.

TEST. Spearman between similarity and deviation over the 15 pairs, with a permutation p
(10,000 shuffles of the deviation vector) and a leave-one-pair-out range, because n is 15
and one pair (formic acid + lactic acid, which grew in no well) can carry the statistic.
Reported for each similarity measure, profile version and growth call. The pre-registered
expectation was negative: similar profiles (shared targets) behave additively, dissimilar
ones deviate. A positive sign is a failure of the claim, not a finding.

CAVEAT stated in the output: in ``all_predicted`` the three small acids are fingerprint
near-duplicates through the ridge fit (centered Spearman 0.91 to 0.99), so those pairs'
similarity is chemistry, not measured biology.
"""

from __future__ import annotations

import os
import os.path as osp

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from scipy.stats import spearmanr

from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
EXPERIMENT = osp.join(os.environ["EXPERIMENT_ROOT"], "040-inhibitor-synergy-wetlab")
RESULTS = osp.join(EXPERIMENT, "results")
IMAGES = osp.join(os.environ["ASSET_IMAGES_DIR"], "040-inhibitor-synergy-wetlab")

N_PERMUTATIONS = 10_000
SEED = 0
VERSIONS = ("best_available", "all_predicted")
SIMILARITIES = ("spearman_centered", "spearman_raw", "jaccard_top100_centered")
DEVIATIONS = {
    "served": "obs_minus_bliss_ex23_served",
    "software": "obs_minus_bliss_ex23_software",
}
ABBREVIATION = {
    "5-(hydroxymethyl)furfural": "HMF",
    "acetic acid": "AA",
    "formic acid": "FA",
    "furfural": "FF",
    "lactic acid": "LA",
    "levulinic acid": "LVA",
}

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


class SimilarityTest(BaseModel):
    """One Spearman test of similarity against deviation over the 15 pairs."""

    version: str
    similarity: str
    growth_call: str
    n_pairs: int
    spearman: float
    permutation_p: float
    loo_min: float
    loo_max: float
    n_pairs_both_predicted: int
    spearman_without_fa_la: float


def permutation_p(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> float:
    observed = abs(spearmanr(x, y).statistic)
    draws = np.array(
        [abs(spearmanr(x, rng.permutation(y)).statistic) for _ in range(N_PERMUTATIONS)]
    )
    return float((np.sum(draws >= observed) + 1) / (N_PERMUTATIONS + 1))


def leave_one_out(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    values = [
        spearmanr(np.delete(x, i), np.delete(y, i)).statistic for i in range(len(x))
    ]
    return float(min(values)), float(max(values))


def joined(version: str) -> pd.DataFrame:
    deviation = pd.read_csv(osp.join(RESULTS, "pair_deviation.csv"))
    similarity = pd.read_csv(osp.join(RESULTS, "profile_similarity.csv"))
    sim = similarity[similarity["version"] == version]
    table = deviation.merge(sim, on="pair", validate="one_to_one")
    assert len(table) == 15, f"{version}: {len(table)} pairs joined, expected 15"
    table["both_predicted"] = table["profile_a"].str.contains("predicted") & table[
        "profile_b"
    ].str.contains("predicted")
    table["label"] = [
        f"{ABBREVIATION[a]}+{ABBREVIATION[b]}"
        for a, b in zip(table["compound_a"], table["compound_b"], strict=True)
    ]
    return table


def tests(table: pd.DataFrame, version: str) -> list[SimilarityTest]:
    rng = np.random.default_rng(SEED)
    out = []
    for sim in SIMILARITIES:
        for call, column in DEVIATIONS.items():
            x = table[sim].to_numpy(dtype=float)
            y = table[column].to_numpy(dtype=float)
            keep = table["pair"] != "formic acid|lactic acid"
            lo, hi = leave_one_out(x, y)
            out.append(
                SimilarityTest(
                    version=version,
                    similarity=sim,
                    growth_call=call,
                    n_pairs=len(x),
                    spearman=float(spearmanr(x, y).statistic),
                    permutation_p=permutation_p(x, y, rng),
                    loo_min=lo,
                    loo_max=hi,
                    n_pairs_both_predicted=int(table["both_predicted"].sum()),
                    spearman_without_fa_la=float(
                        spearmanr(x[keep.to_numpy()], y[keep.to_numpy()]).statistic
                    ),
                )
            )
    return out


def isobole_rows(version: str) -> pd.DataFrame:
    iso = pd.read_csv(osp.join(RESULTS, "isobole_summary.csv"))
    sim = pd.read_csv(osp.join(RESULTS, "profile_similarity.csv"))
    sim = sim[sim["version"] == version][
        ["pair", "spearman_centered", "jaccard_top100_centered"]
    ]
    return iso.merge(sim, on="pair", validate="one_to_one")[
        [
            "run",
            "pair",
            "mean_excess_over_bliss",
            "mean_excess_lo",
            "mean_excess_hi",
            "call_bliss",
            "flag",
            "spearman_centered",
            "jaccard_top100_centered",
        ]
    ]


def figure(tables: dict[str, pd.DataFrame], results: pd.DataFrame) -> list[str]:
    fig, axes = plt.subplots(
        1, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(62)), dpi=300
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.17, top=0.9, wspace=0.25)
    for ax, version in zip(axes, VERSIONS, strict=True):
        t = tables[version]
        measured = ~t["both_predicted"]
        ax.axhline(0, color="black", linewidth=0.5)
        ax.scatter(
            t.loc[measured, "spearman_centered"],
            t.loc[measured, "obs_minus_bliss_ex23_served"],
            s=14,
            color=PLOT_PALETTE[0],
            edgecolor="black",
            linewidth=0.4,
            label="at least one measured profile",
            zorder=3,
        )
        ax.scatter(
            t.loc[~measured, "spearman_centered"],
            t.loc[~measured, "obs_minus_bliss_ex23_served"],
            s=14,
            color=PLOT_PALETTE[2],
            edgecolor="black",
            linewidth=0.4,
            marker="s",
            label="both profiles predicted",
            zorder=3,
        )
        ax.errorbar(
            t["spearman_centered"],
            t["obs_minus_bliss_ex23_served"],
            yerr=[
                t["obs_minus_bliss_ex23_served"] - t["obs_minus_bliss_ex23_lo_served"],
                t["obs_minus_bliss_ex23_hi_served"] - t["obs_minus_bliss_ex23_served"],
            ],
            fmt="none",
            ecolor="black",
            elinewidth=0.4,
            capsize=1.2,
            zorder=2,
        )
        for _, row in t.iterrows():
            ax.annotate(
                row["label"],
                (row["spearman_centered"], row["obs_minus_bliss_ex23_served"]),
                xytext=(2, 2),
                textcoords="offset points",
                fontsize=5,
            )
        r = results[
            (results["version"] == version)
            & (results["similarity"] == "spearman_centered")
            & (results["growth_call"] == "served")
        ].iloc[0]
        ax.set_title(
            f"{version}: Spearman {r['spearman']:.2f}, permutation p {r['permutation_p']:.2f}, "
            f"leave-one-out {r['loo_min']:.2f} to {r['loo_max']:.2f} (n = 15 pairs)"
        )
        ax.set_xlabel("profile similarity (centered Spearman across genes)")
        ax.set_ylabel("observed minus Bliss (served call; no growth = 0)")
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(0.5)
        ax.grid(linewidth=0.3, color="#DDDDDD")
        ax.set_axisbelow(True)
    axes[0].legend(loc="lower left", frameon=False)
    os.makedirs(IMAGES, exist_ok=True)
    stem = osp.join(IMAGES, f"similarity_vs_deviation_{timestamp()}")
    savefig_true_size_svg(fig, stem + ".svg")
    fig.savefig(stem + ".png", dpi=300)
    plt.close(fig)
    return [stem + ".svg", stem + ".png"]


def main() -> None:
    tables = {v: joined(v) for v in VERSIONS}
    results = pd.DataFrame(
        [t.model_dump() for v in VERSIONS for t in tests(tables[v], v)]
    )
    results.to_csv(osp.join(RESULTS, "similarity_vs_deviation.csv"), index=False)
    pd.concat([tables[v].assign(version=v) for v in VERSIONS])[
        [
            "version",
            "pair",
            "label",
            "both_predicted",
            "profile_a",
            "profile_b",
            "spearman_centered",
            "spearman_raw",
            "jaccard_top100_centered",
            "obs_minus_bliss_ex23_served",
            "obs_minus_bliss_ex23_software",
            "call_bliss_ex23_served",
            "call_bliss_ex23_software",
        ]
    ].to_csv(osp.join(RESULTS, "similarity_vs_deviation_pairs.csv"), index=False)
    iso = pd.concat([isobole_rows(v).assign(version=v) for v in VERSIONS])
    iso.to_csv(osp.join(RESULTS, "similarity_vs_deviation_isoboles.csv"), index=False)
    print(results.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(iso.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    for path in figure(tables, results):
        print("wrote", path)


if __name__ == "__main__":
    main()
