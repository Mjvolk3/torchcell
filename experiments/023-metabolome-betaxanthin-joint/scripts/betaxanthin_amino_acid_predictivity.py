# experiments/023-metabolome-betaxanthin-joint/scripts/betaxanthin_amino_acid_predictivity.py
# [[experiments.023-metabolome-betaxanthin-joint.scripts.betaxanthin_amino_acid_predictivity]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/023-metabolome-betaxanthin-joint/scripts/betaxanthin_amino_acid_predictivity
"""Is betaxanthin production coupled to the free amino-acid pool of the same deletion?

THE QUESTION, and why it is the premise of the whole 023 round. Betalains derive from
L-tyrosine (tyrosine -> L-DOPA -> betalamic acid) and a betaxanthin IS betalamic acid
condensed with an amino acid. So the mechanistic prediction is that the free amino-acid
pool of a deletion strain should carry information about how much betaxanthin that same
deletion makes when the Btx cassette is installed. `conf/igb_bx_aa.yaml` spends an entire
allocation on whether a `mulleder19` auxiliary head moves the betaxanthin metric; this
script asks the cheaper prior question directly from the two MEASURED screens, with no
model in the loop.

WHAT IS COMPARED. Two genome-wide single-deletion screens keyed by systematic ORF:

  Cachera 2023 (`betaxanthin_cachera2023`)  -- CRI-SPA corrected fluorescence at 24 h,
      a betaxanthin proxy, measured in a background carrying the four-gene Btx cassette
      (CYP76AD1, DOD, ARO4-K229L, ARO7-G141S). n = 4,735 deletions.
  Mulleder 2016 (`amino_acid_mulleder2016`) -- intracellular concentration (mM) of 19
      amino acids, measured in the PLAIN deletion collection with no cassette.
      n = 4,678 deletions, `n_replicates = 1`, no SE.

The two backgrounds differ, and the difference is not incidental: ARO4-K229L and
ARO7-G141S are feedback-resistant alleles that deregulate the shikimate pathway, so the
Cachera background's precursor supply is already released from the control that shapes
the Mulleder pools. Any coupling measured here is therefore a coupling between the plain
strain's pool and the engineered strain's output, which is the form the model would have
to exploit, because the model likewise never sees both phenotypes on one strain.

THREE MEASUREMENTS, in increasing order of what they license.

1. MARGINAL, per amino acid. Pearson and Spearman of betaxanthin against each amino
   acid. This reproduces `experiments/019-simb-multimodal/results/pigment_noise_ceiling.json`
   (`mulleder_external_check`) and is recomputed here so the figure and the multivariate
   result come off one merge; agreement with that file is ASSERTED, not assumed.

2. MULTIVARIATE, cross-validated. Ridge from the 19-dimensional log amino-acid profile to
   betaxanthin, scored out-of-fold. A marginal correlation asks whether ONE pool tracks
   production; this asks whether the PROFILE does. The two answers differ by a factor of
   four here, which is the finding.

3. THE FITNESS CONTROL. Both screens are deletion phenotypes and both respond to growth,
   so a shared fitness component would produce coupling with no metabolic content. Single
   mutant fitness (Costanzo 2016 KanMX deletions at 30 C) is regressed out of BOTH sides
   and the cross-validated fit recomputed on the residuals. Reported alongside the same
   fit on the same gene set WITHOUT the control, because the control also shrinks the gene
   set and the two effects must not be confused.

ATTENUATION. Betaxanthin has a measured reliability of 0.836 from its per-record SE
(`pigment_noise_ceiling.json`), so correlations against it are attenuated by a factor
sqrt(0.836) = 0.914 that is known. Mulleder has one replicate per strain and no SE, so
ITS reliability is unknown and no full disattenuation is possible. What is reported is the
partial correction for the betaxanthin side only, which is a LOWER BOUND on the corrected
value; the true correction is larger by an unmeasured amount.

Sources (sha256-pinned mirrors, see each loader's manifest):
  $DATA_ROOT/data/torchcell/betaxanthin_cachera2023/preprocess/data.csv
  $DATA_ROOT/data/torchcell/amino_acid_mulleder2016/preprocess/data.csv
  $DATA_ROOT/data/torchcell/smf_costanzo2016/preprocess/data.csv

Run from repo root:
  PYTHONPATH=. python experiments/023-metabolome-betaxanthin-joint/scripts/betaxanthin_amino_acid_predictivity.py
"""

from __future__ import annotations

import json
import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)

EXPERIMENT = "023-metabolome-betaxanthin-joint"
IMAGE_SUBDIR = "023-metabolome-betaxanthin-joint"

# Reliability of the betaxanthin target, from its own per-record SE. Measured in
# experiments/019-simb-multimodal/scripts/pigment_noise_ceiling.py and recorded in that
# script's results JSON; asserted here against a local recomputation so the two cannot
# drift apart silently.
BETAXANTHIN_RELIABILITY = 0.8355892220682738

# Ridge penalty grid. Wide enough that the selected alpha is interior for every design
# below (checked at run time), so the fit is not silently pinned to an endpoint.
ALPHAS = np.logspace(-2, 4, 25)

# Out-of-fold scoring. Three shuffles of the same 5 folds, so the reported spread is the
# fold-assignment noise on the score rather than a single draw.
N_FOLDS = 5
CV_SEEDS = (0, 1, 2)


def _z(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return (a - a.mean(axis=0)) / a.std(axis=0)


def _residualize(y: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Residual of ``y`` after least-squares removal of ``c`` plus an intercept."""
    design = np.hstack([np.ones((len(c), 1)), c])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return y - design @ beta


def _cv_score(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Out-of-fold Pearson/Spearman of a ridge fit, with THREE named uncertainties.

    The reported score is the mean over ``CV_SEEDS`` of the full-sample out-of-fold
    correlation. Three different spreads can be attached to it, they are not
    interchangeable, and quoting one without naming it is how a plus-minus becomes
    misleading. All three are emitted:

    ``pearson_sd_across_shuffles``
        SD of the full-sample out-of-fold r over the three fold shuffles. It measures
        ONLY fold-assignment noise on the same n deletions, so it is the SMALLEST of the
        three and UNDERSTATES how much the number would move on a different sample of
        deletions. It is the error bar drawn in panel b, and it is the number to read as
        "would a re-run of this same analysis give the same value".
    ``pearson_sd_across_folds``
        SD of the r computed WITHIN each held-out fold, over all ``N_FOLDS *
        len(CV_SEEDS)`` folds. Each fold holds n/5 deletions, so this is the spread of an
        estimate on a fifth of the data and is correspondingly LARGER than the sampling
        error of the reported full-sample estimate. It is an upper reference, not the SE.
    ``pearson_se_fisher``
        The analytic standard error of a correlation at this n, ``1/sqrt(n-3)`` on the
        Fisher-z scale mapped back at the observed r. This is the one that answers "how
        precisely is this coupling measured on this many deletions", and it does not
        depend on the cross-validation at all.
    """
    scores = []
    fold_r: list[float] = []
    alphas_chosen = []
    for seed in CV_SEEDS:
        folds = KFold(N_FOLDS, shuffle=True, random_state=seed)
        oof = np.zeros_like(y)
        for train_idx, test_idx in folds.split(x):
            model = RidgeCV(alphas=ALPHAS).fit(x[train_idx], y[train_idx])
            alphas_chosen.append(float(model.alpha_))
            oof[test_idx] = model.predict(x[test_idx])
            fold_r.append(float(pearsonr(oof[test_idx], y[test_idx])[0]))
        scores.append((pearsonr(oof, y)[0], spearmanr(oof, y)[0]))
    arr = np.asarray(scores)
    lo, hi = float(ALPHAS[0]), float(ALPHAS[-1])
    if min(alphas_chosen) <= lo or max(alphas_chosen) >= hi:
        raise ValueError(
            f"RidgeCV selected an endpoint alpha ({min(alphas_chosen)}, "
            f"{max(alphas_chosen)}) against grid [{lo}, {hi}]; widen ALPHAS"
        )
    r_mean = float(arr[:, 0].mean())
    n = int(len(y))
    # Fisher-z SE mapped back to the r scale: the half-width of the +-1 SE interval on z,
    # transformed, averaged, so it is a symmetric number that can be printed as +-.
    z = np.arctanh(np.clip(r_mean, -0.999999, 0.999999))
    se_z = 1.0 / np.sqrt(n - 3)
    se_fisher = float((np.tanh(z + se_z) - np.tanh(z - se_z)) / 2.0)
    return {
        "pearson": r_mean,
        "pearson_sd_across_shuffles": float(arr[:, 0].std(ddof=1)),
        "pearson_sd_across_folds": float(np.std(fold_r, ddof=1)),
        "pearson_se_fisher": se_fisher,
        "spearman": float(arr[:, 1].mean()),
        "spearman_sd_across_shuffles": float(arr[:, 1].std(ddof=1)),
        "n": n,
        "n_features": int(x.shape[1]),
        "n_folds_scored": len(fold_r),
    }


def load_frames(data_root: str) -> tuple[pd.DataFrame, list[str]]:
    """Merge the two screens plus the fitness control, keyed by systematic ORF."""
    tc = osp.join(data_root, "data", "torchcell")
    betaxanthin = pd.read_csv(
        osp.join(tc, "betaxanthin_cachera2023", "preprocess", "data.csv")
    )
    amino_acid = pd.read_csv(
        osp.join(tc, "amino_acid_mulleder2016", "preprocess", "data.csv")
    )
    fitness = pd.read_csv(
        osp.join(tc, "smf_costanzo2016", "preprocess", "data.csv")
    )
    # KanMX deletions at 30 C only. The NatMX arm is the query-strain copy of the same
    # deletions and the ts/damp alleles are not deletions at all, so pooling them would
    # average different perturbation classes into one control.
    fitness = fitness[
        (fitness["perturbation_type"] == "KanMX_deletion")
        & (fitness["Temperature"] == 30)
    ]
    fitness = (
        fitness[["Systematic gene name", "Single mutant fitness"]]
        .groupby("Systematic gene name", as_index=False)
        .mean()
        .rename(
            columns={
                "Systematic gene name": "orf",
                "Single mutant fitness": "smf",
            }
        )
    )
    names = [c for c in amino_acid.columns if c != "orf"]
    merged = betaxanthin.merge(amino_acid, on="orf", how="inner").merge(
        fitness, on="orf", how="left"
    )
    return merged, names


def marginal_table(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    rows = []
    for name in names:
        r, p = pearsonr(df["level"], df[name])
        rho, p_rho = spearmanr(df["level"], df[name])
        rows.append(
            {
                "amino_acid": name,
                "n": int(len(df)),
                "pearson": float(r),
                "pearson_p": float(p),
                "spearman": float(rho),
                "spearman_p": float(p_rho),
                # Partial disattenuation: the betaxanthin side only. A LOWER BOUND on
                # the fully corrected value, since the Mulleder side's reliability is
                # unmeasurable at one replicate per strain.
                "pearson_bx_disattenuated": float(
                    r / np.sqrt(BETAXANTHIN_RELIABILITY)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("pearson").reset_index(drop=True)


def check_against_019(marginals: pd.DataFrame, data_root: str, root: str) -> str:
    """Assert the marginals reproduce the committed 019 pigment-ceiling artifact."""
    path = osp.join(
        root, "experiments", "019-simb-multimodal", "results", "pigment_noise_ceiling.json"
    )
    with open(path) as fh:
        prior = json.load(fh)["mulleder_external_check"]["per_amino_acid"]
    for _, row in marginals.iterrows():
        expected = prior[row["amino_acid"]]["pearson"]
        if not np.isclose(row["pearson"], expected, atol=1e-10):
            raise ValueError(
                f"{row['amino_acid']}: marginal Pearson {row['pearson']} does not "
                f"reproduce pigment_noise_ceiling.json's {expected}"
            )
    return path


def predictivity(df: pd.DataFrame, names: list[str]) -> dict[str, object]:
    """Cross-validated ridge fits, with and without the fitness control."""
    has_fitness = df["smf"].notna().to_numpy()
    log_pool = np.log(df[names].to_numpy(dtype=float) + 1e-6)
    level = df["level"].to_numpy(dtype=float)

    out: dict[str, object] = {}
    out["all_genes"] = {
        "nineteen_amino_acids": _cv_score(_z(log_pool), level),
        "tyrosine_only": _cv_score(
            _z(np.log(df[["tyrosine"]].to_numpy(dtype=float) + 1e-6)), level
        ),
    }

    # The fitness-controlled comparison is run on ONE gene set, the genes that carry a
    # Costanzo KanMX record, so the control's effect is not confounded with the change in
    # population that requiring it causes. The same fit on the complementary set is
    # reported too, because that set turns out to carry most of the signal.
    sub_pool, sub_level = _z(log_pool[has_fitness]), level[has_fitness]
    sub_fitness = _z(df.loc[has_fitness, "smf"].to_numpy(dtype=float).reshape(-1, 1))
    resid_pool = _residualize(sub_pool, sub_fitness)
    resid_level = _residualize(sub_level.reshape(-1, 1), sub_fitness).ravel()
    out["with_fitness_record"] = {
        "nineteen_amino_acids": _cv_score(sub_pool, sub_level),
        "fitness_only": _cv_score(sub_fitness, sub_level),
        "nineteen_plus_fitness": _cv_score(
            np.hstack([sub_pool, sub_fitness]), sub_level
        ),
        "nineteen_given_fitness": _cv_score(resid_pool, resid_level),
        "betaxanthin_vs_fitness_pearson": float(pearsonr(sub_level, sub_fitness.ravel())[0]),
        "tyrosine_vs_fitness_pearson": float(
            pearsonr(
                np.log(df.loc[has_fitness, "tyrosine"].to_numpy(dtype=float) + 1e-6),
                sub_fitness.ravel(),
            )[0]
        ),
    }
    out["without_fitness_record"] = {
        "nineteen_amino_acids": _cv_score(
            _z(log_pool[~has_fitness]), level[~has_fitness]
        ),
        "betaxanthin_sd": float(level[~has_fitness].std(ddof=1)),
    }
    out["with_fitness_record"]["betaxanthin_sd"] = float(sub_level.std(ddof=1))
    return out


def high_producer_shift(df: pd.DataFrame, quantile: float = 0.95) -> dict[str, object]:
    """Is tyrosine elevated in the top producers, even with no linear correlation?"""
    cut = df["level"].quantile(quantile)
    top = df["level"] >= cut
    stat, p = mannwhitneyu(df.loc[top, "tyrosine"], df.loc[~top, "tyrosine"])
    deciles = (
        df.assign(decile=pd.qcut(df["level"], 10, labels=False))
        .groupby("decile")["tyrosine"]
        .median()
    )
    return {
        "quantile": quantile,
        "cut": float(cut),
        "n_top": int(top.sum()),
        "tyrosine_median_top": float(df.loc[top, "tyrosine"].median()),
        "tyrosine_median_rest": float(df.loc[~top, "tyrosine"].median()),
        "mannwhitneyu_u": float(stat),
        "mannwhitneyu_p": float(p),
        "tyrosine_median_by_betaxanthin_decile": [float(v) for v in deciles],
    }


#: PREDICTOR -> COLOR, held fixed across every group of panel b, so hue names WHAT WAS
#: REGRESSED and horizontal position names WHICH DELETIONS it was fit on. The two
#: encodings were previously merged into four unlabelled bars, which is what made purple
#: unreadable without the table.
PREDICTOR_STYLE = {
    "tyrosine": (PLOT_PALETTE[1], "tyrosine alone (the precursor)"),
    "profile": (PLOT_PALETTE[0], "19 amino acids jointly"),
    "fitness": (PLOT_PALETTE[5], "single-mutant fitness alone"),
    "profile_plus_fitness": (PLOT_PALETTE[3], "19 amino acids and fitness"),
    "profile_given_fitness": (
        PLOT_PALETTE[2],
        "19 amino acids, fitness regressed out of both",
    ),
}


def make_figure(
    df: pd.DataFrame,
    marginals: pd.DataFrame,
    fits: dict[str, object],
    shift: dict[str, object],
    out_png: str,
    out_svg: str,
) -> None:
    """Three panels: a MARGINAL, a CROSS-VALIDATED RIDGE, and the non-monotone decile.

    Panels a and b answer different questions with different machinery and the titles say
    so, because they were previously distinguishable only by reading the caption: (a) is a
    plain Pearson correlation of one measured column against another, no model and no
    held-out data anywhere; (b) is the five-fold, three-shuffle ridge of section 5.3,
    every point of which is predicted by a fit that never saw that deletion.
    """
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6,
            "axes.linewidth": 0.5,
            "svg.fonttype": "none",
            # torchcell.datasets flips savefig.bbox -> "tight" globally, which recrops at
            # save time and defeats the strict width template. Pin it back.
            "savefig.bbox": None,
        }
    )
    width_mm = PANEL_WIDTHS_MM["full"]
    fig, axes = plt.subplots(
        1, 3, figsize=(mm_to_in(width_mm), mm_to_in(54.0)), constrained_layout=True
    )

    # (a) MARGINAL correlation per amino acid, tyrosine called out. No model, no folds.
    ax = axes[0]
    order = marginals.sort_values("pearson")
    is_tyr = (order["amino_acid"] == "tyrosine").to_numpy()
    colors = [PLOT_PALETTE[1] if t else PLOT_PALETTE[5] for t in is_tyr]
    ax.barh(
        np.arange(len(order)),
        order["pearson"],
        color=colors,
        edgecolor="black",
        linewidth=0.4,
        height=0.75,
    )
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order["amino_acid"], fontsize=5)
    ax.axvline(0.0, color="black", linewidth=0.5)
    ax.set_xlabel("Pearson r vs betaxanthin")
    ax.set_title(
        f"Marginal correlation, one amino acid at a time\n"
        f"no model, no held-out data; n = {len(df):,} shared deletions",
        loc="left",
        fontsize=5.5,
    )
    # Limits are set to the data's own range plus a margin, not to +-0.4: no amino acid
    # exceeds |r| = 0.15, and the wide axis shrank every bar to a stub.
    ax.set_xlim(-0.22, 0.22)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.tick_params(axis="x", which="minor", length=0)
    # NAMED AT THE BAR, not in a legend. Every other bar is already named by its y tick
    # label, so the only thing a legend has to say is what the red one is -- and a legend
    # box wide enough to hold that sentence lands on the top bars in either upper corner.
    # The annotation sits on the positive side of the tyrosine row, which is empty because
    # tyrosine's correlation is negative.
    tyr_row = int(np.flatnonzero(is_tyr)[0])
    ax.annotate(
        "tyrosine, the chromophore precursor",
        xy=(0.012, tyr_row),
        ha="left",
        va="center",
        fontsize=4.5,
        color=PLOT_PALETTE[1],
    )

    # (b) CROSS-VALIDATED RIDGE (section 5.3): every row of the predictivity table, drawn
    # in the groups the table separates, because the gene set changes between them and a
    # single row of bars invited comparing 0.455 on 724 deletions against 0.298 on 4,432.
    ax = axes[1]
    wf = fits["with_fitness_record"]
    groups = [
        (
            f"all shared\nn = {fits['all_genes']['nineteen_amino_acids']['n']:,}",
            [
                ("tyrosine", fits["all_genes"]["tyrosine_only"]),
                ("profile", fits["all_genes"]["nineteen_amino_acids"]),
            ],
        ),
        (
            f"with a Costanzo\nfitness record\nn = {wf['nineteen_amino_acids']['n']:,}",
            [
                ("fitness", wf["fitness_only"]),
                ("profile", wf["nineteen_amino_acids"]),
                ("profile_plus_fitness", wf["nineteen_plus_fitness"]),
                ("profile_given_fitness", wf["nineteen_given_fitness"]),
            ],
        ),
        (
            "no fitness record\n"
            f"n = {fits['without_fitness_record']['nineteen_amino_acids']['n']:,}",
            [("profile", fits["without_fitness_record"]["nineteen_amino_acids"])],
        ),
    ]
    pos, centers, gap = [], [], 0.9
    x = 0.0
    for _, bars in groups:
        centers.append(x + (len(bars) - 1) / 2.0)
        for _ in bars:
            pos.append(x)
            x += 1.0
        x += gap
    flat = [(k, s) for _, bars in groups for k, s in bars]
    ax.bar(
        pos,
        [s["pearson"] for _, s in flat],
        yerr=[s["pearson_sd_across_shuffles"] for _, s in flat],
        color=[PREDICTOR_STYLE[k][0] for k, _ in flat],
        edgecolor="black",
        linewidth=0.4,
        width=0.82,
        error_kw={"elinewidth": 0.5, "capsize": 1.2},
        zorder=3,
    )
    for p, (_, s) in zip(pos, flat):
        ax.text(
            p,
            s["pearson"] + s["pearson_sd_across_shuffles"] + 0.012,
            f"{s['pearson']:.3f}",
            ha="center",
            va="bottom",
            fontsize=4.5,
            zorder=4,
        )
    ax.set_xticks(centers)
    ax.set_xticklabels([g[0] for g in groups], fontsize=4.5)
    ax.set_ylabel("out-of-fold Pearson r")
    ax.set_ylim(0, 0.56)
    ax.set_xlim(-0.7, pos[-1] + 0.7)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(axis="y", which="both", linewidth=0.3, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(
        "Cross-validated ridge (section 5.3), out of fold\n"
        "five folds, three shuffles; error bar = SD across shuffles",
        loc="left",
        fontsize=5.5,
    )
    for key, (color, label) in PREDICTOR_STYLE.items():
        ax.bar(0, 0, color=color, edgecolor="black", linewidth=0.4, label=label)
    ax.legend(
        loc="upper left", frameon=False, fontsize=4.5, handletextpad=0.4, labelspacing=0.3
    )

    # (c) tyrosine median by betaxanthin decile: the shift the linear fit misses. The
    # finding is stated IN the panel, because a reader who takes the panel on its own gets
    # a wiggly line and no reason to care about it.
    ax = axes[2]
    medians = np.asarray(shift["tyrosine_median_by_betaxanthin_decile"], dtype=float)
    deciles = np.arange(1, 11)
    ax.plot(
        deciles,
        medians,
        marker="o",
        markersize=2.5,
        linewidth=0.8,
        color=PLOT_PALETTE[1],
        label="median tyrosine within decile",
        zorder=3,
    )
    lo_i, hi_i = int(np.argmin(medians)), int(np.argmax(medians))
    ax.set_xlabel("betaxanthin decile (1 = lowest production)")
    ax.set_ylabel("median tyrosine (mM)")
    ax.set_xticks(deciles)
    ax.set_title(
        "Tyrosine is non-monotone in betaxanthin decile\n"
        "which is why its linear correlation reads near zero",
        loc="left",
        fontsize=5.5,
    )
    span = float(medians.max() - medians.min())
    ax.set_ylim(medians.min() - 0.35 * span, medians.max() + 0.60 * span)
    r_tyr = float(
        marginals.loc[marginals["amino_acid"] == "tyrosine", "pearson"].iloc[0]
    )
    ax.annotate(
        f"highest in decile {hi_i + 1} ({medians[hi_i]:.2f} mM), lowest in decile "
        f"{lo_i + 1} ({medians[lo_i]:.2f} mM),\n"
        f"and back up at decile 10 ({medians[-1]:.2f} mM)\n"
        f"swing {span:.3f} mM = {100 * span / float(np.median(medians)):.0f}% of the "
        f"median; linear r = {r_tyr:+.3f} (panel a)",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=4.5,
        linespacing=1.3,
    )
    ax.legend(loc="lower right", frameon=False, fontsize=4.5, handletextpad=0.4)
    ax.grid(axis="y", which="both", linewidth=0.3, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
    for ax, letter in zip(axes, "abc"):
        panel_label(ax, letter)

    fig.savefig(out_png, dpi=300, facecolor="white")
    savefig_true_size_svg(fig, out_svg, facecolor="white")
    plt.close(fig)


def main() -> None:
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    images_dir = os.environ["ASSET_IMAGES_DIR"]
    experiment_root = os.environ["EXPERIMENT_ROOT"]
    root = osp.dirname(experiment_root.rstrip("/"))

    df, names = load_frames(data_root)
    marginals = marginal_table(df, names)
    prior_path = check_against_019(marginals, data_root, root)
    fits = predictivity(df, names)
    shift = high_producer_shift(df)

    # STABLE NAME, no timestamp. This panel is referenced by
    # notes-tex/019-simb-multimodal/, whose `make plots-all` converts
    # $ASSET_IMAGES_DIR/<slug>/<name>.svg into figures/<name>.pdf. A timestamped filename
    # cannot be the target of a Makefile rule, so the document would silently keep an old
    # PDF. The write timestamp is recorded in the results JSON instead.
    out_dir = osp.join(images_dir, IMAGE_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    png = osp.join(out_dir, "betaxanthin_amino_acid_predictivity.png")
    svg = osp.join(out_dir, "betaxanthin_amino_acid_predictivity.svg")
    make_figure(df, marginals, fits, shift, png, svg)

    results_dir = osp.join(experiment_root, EXPERIMENT, "results")
    os.makedirs(results_dir, exist_ok=True)
    marginals.to_csv(
        osp.join(results_dir, "betaxanthin_amino_acid_marginals.csv"), index=False
    )
    payload = {
        "n_shared_deletions": int(len(df)),
        "n_with_fitness_record": int(df["smf"].notna().sum()),
        "betaxanthin_reliability": BETAXANTHIN_RELIABILITY,
        "marginals_reproduce": prior_path,
        "marginal_tyrosine": marginals.loc[
            marginals["amino_acid"] == "tyrosine"
        ].to_dict("records")[0],
        "marginal_strongest_abs": marginals.reindex(
            marginals["pearson"].abs().sort_values(ascending=False).index
        )
        .head(3)
        .to_dict("records"),
        "cross_validated": fits,
        # WHICH SPREAD IS WHICH, next to the numbers rather than in a docstring, because
        # the table in the document prints one of them as a plus-minus and the reader of
        # that table has to be able to find out which.
        "cross_validated_uncertainty_key": {
            "pearson_sd_across_shuffles": (
                "SD of the full-sample out-of-fold r over the 3 fold shuffles; "
                "fold-assignment noise only, on the same deletions. Smallest of the "
                "three, and it understates sampling uncertainty. Drawn as panel b's "
                "error bar."
            ),
            "pearson_sd_across_folds": (
                "SD of the r computed within each held-out fold, over 5 folds x 3 "
                "shuffles. Each fold holds n/5 deletions, so this is larger than the "
                "sampling error of the reported full-sample estimate; an upper "
                "reference, not the SE."
            ),
            "pearson_se_fisher": (
                "analytic SE of a correlation at this n, 1/sqrt(n-3) on the Fisher-z "
                "scale mapped back to r. Independent of the cross-validation, and the "
                "one to quote for 'how precisely is this measured at this n'."
            ),
        },
        "high_producer_tyrosine_shift": shift,
        "figure": {"png": png, "svg": svg, "written_at": timestamp()},
    }
    with open(
        osp.join(results_dir, "betaxanthin_amino_acid_predictivity.json"), "w"
    ) as fh:
        json.dump(payload, fh, indent=2)

    print(json.dumps(payload["cross_validated"], indent=2))
    # The predictivity table's rows, in table order, with all three spreads side by side.
    print("\nrow                                        n      r      sd_shuf sd_fold se_n")
    wf = fits["with_fitness_record"]
    rows = [
        ("tyrosine alone (the precursor)", fits["all_genes"]["tyrosine_only"]),
        ("19 amino acids jointly", fits["all_genes"]["nineteen_amino_acids"]),
        ("single-mutant fitness alone", wf["fitness_only"]),
        ("19 amino acids", wf["nineteen_amino_acids"]),
        ("19 amino acids and fitness", wf["nineteen_plus_fitness"]),
        (
            "19 amino acids, fitness regressed out",
            wf["nineteen_given_fitness"],
        ),
        (
            "19 amino acids, no fitness record",
            fits["without_fitness_record"]["nineteen_amino_acids"],
        ),
    ]
    for name, s in rows:
        print(
            f"{name:<42s} {s['n']:>5d}  {s['pearson']:.4f}  "
            f"{s['pearson_sd_across_shuffles']:.4f}  "
            f"{s['pearson_sd_across_folds']:.4f}  {s['pearson_se_fisher']:.4f}"
        )
    print(f"\nmarginal tyrosine r = {payload['marginal_tyrosine']['pearson']:.4f}")
    print(f"figure -> {svg}")


if __name__ == "__main__":
    main()
