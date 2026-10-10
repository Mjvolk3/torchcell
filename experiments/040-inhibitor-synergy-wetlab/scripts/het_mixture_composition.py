# experiments/040-inhibitor-synergy-wetlab/scripts/het_mixture_composition.py
# [[experiments.040-inhibitor-synergy-wetlab.scripts.het_mixture_composition]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/het_mixture_composition
"""How do two compounds' deletion profiles compose under the mixture? (040 claim 3)

Public data only: Hillenmeyer 2008 HET (heterozygous diploid deletion pool, YPD), as
flattened into experiment 033's pooled cell table (build 002). The response is
``log2(mean control intensity / treatment intensity)``, so POSITIVE = fitness defect.
One value per (gene, environment) is the MEAN over the cell's ``responses`` list (the
033 flatten keeps every measurement of a cell as a list; HET cells carry 1 to 4).

1. Gene x environment matrix over every HET environment.
2. For each of the 26 two-compound environments, the single-compound environment of each
   partner: exact dose first, else nearest dose on log10 molar; ties broken by the same
   generation count. A match is REASONABLE when the dose is within one twofold step
   (ratio <= 2.05, the 0.05 absorbing the paper's rounding, e.g. 65.3 vs 32.6 uM) and the
   generation count equals the pair's. Pairs with an unreasonable partner are flagged and
   kept out of the headline statistics (all 26 rows are still written).
3. On genes measured in all three environments, composition rules against the observed
   pair profile y12: sum (y1 + y2), mean, max (the sicker of the two), Bliss product on
   the fitness scale (f = 2^-y, f12 = f1 * f2, back to -log2 f12), and a free linear fit
   y12 = a*y1 + b*y2 + c. R^2 is 1 - SS_res / SS_tot of the rule's prediction as is
   (no refit, so it can be negative); Spearman is scale-free.
   NOTE: on this log-ratio scale the Bliss product is exactly the sum,
   -log2(2^-y1 * 2^-y2) = y1 + y2; the script computes it and checks the identity.
4. Hits: robust z within each environment over all of its genes,
   z = (y - median) / (1.4826 * MAD); hit = |z| > 2. Emergent = hit in the pair, in
   neither single; masked = hit in a single, not in the pair. Null: 200 draws of two
   random single-compound YPD environments containing neither partner compound, set
   beside the same pair profile.
5. The methotrexate x 5-fluorouracil 3 x 3 grid: rule and coefficients per dose pair.

A second null draws the two singles only from environments that share a control set
with the pair (every matched single does, so a shared control intensity, which enters
every log-ratio of that set, is held fixed). The noise floor of hit calling is read off
the near-replicate singles: one compound at one dose (within 1%) and one generation count.

Writes results/het_pair_matching.csv, het_rule_fit.csv, het_emergent_masked.csv,
het_near_replicates.csv,
het_summary.json, and three figures to ASSET_IMAGES_DIR/040-inhibitor-synergy-wetlab/.
"""

from __future__ import annotations

import json
import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    apply_paper_style,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
CELL_TABLE = osp.join(
    DATA_ROOT,
    "experiments",
    "033-env-chemgen-pooled",
    "cell_table_002",
    "cell_table.parquet",
)
DATASET = "HetHillenmeyer2008Dataset"
EXP = "040-inhibitor-synergy-wetlab"
RESULTS = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "results")
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, EXP)

Z_HIT = 2.0
N_NULL = 200
SEED = 0
REASONABLE_FOLD = 2.05
# Tacrolimus (FK506) anhydrous, C44H69NO12, 804.0 g/mol (PubChem CID 445643). Only the
# pairs carry an ug/mL dose; the conversion decides that the 50 and 100 uM singles are
# 400 to 1600-fold away, a verdict no plausible molecular weight changes.
MOLAR_MASS_G_PER_MOL = {"tacrolimus": 804.0}
UNIT_TO_MOLAR = {"nM": 1e-9, "uM": 1e-6, "mM": 1e-3, "M": 1.0}
ABBREV = {
    "5-fluorouracil": "5FU",
    "leucovorin": "LV",
    "amphotericin b": "AmB",
    "flucytosine": "5FC",
    "fluconazole": "FLC",
    "ketoconazole": "KTC",
    "itraconazole": "ITC",
    "lithium chloride": "LiCl",
    "tacrolimus": "FK506",
    "methotrexate": "MTX",
    "sodium chloride": "NaCl",
}
FIXED_RULES = ("sum", "mean", "max", "bliss")


class PairMatch(BaseModel):
    """One partner of one two-compound environment and the single it was matched to."""

    pair_env: str
    pair_label: str
    family: str
    partner_index: int
    compound: str
    pair_dose_log10_molar: float
    pair_generations: float
    single_env: str
    single_dose: str
    single_units: str
    single_dose_log10_molar: float
    single_generations: float
    dose_fold: float
    match_kind: str
    reasonable: bool
    shares_control_set: bool
    n_candidates: int
    candidates_without_dose: int


class RuleFit(BaseModel):
    """Fit of every composition rule to one pair profile."""

    pair_env: str
    pair_label: str
    family: str
    flagged: bool
    n_genes: int
    single_similarity_spearman: float
    r2_sum: float
    r2_mean: float
    r2_max: float
    r2_bliss: float
    spearman_sum: float
    spearman_mean: float
    spearman_max: float
    spearman_bliss: float
    bliss_minus_sum_max_abs: float
    lin_a: float
    lin_b: float
    lin_c: float
    r2_linear: float
    spearman_linear: float
    best_rule_r2: str
    best_rule_spearman: str


class EmergentMasked(BaseModel):
    """Emergent and masked hit counts for one pair, with the random-single null."""

    pair_env: str
    pair_label: str
    family: str
    flagged: bool
    n_genes: int
    single_similarity_spearman: float
    hits_pair: int
    hits_single_1: int
    hits_single_2: int
    hits_single_union: int
    emergent: int
    masked: int
    emergent_fraction: float
    masked_fraction: float
    null_emergent_mean: float
    null_emergent_sd: float
    null_emergent_p_le: float
    null_pool_size: int
    ctlnull_emergent_mean: float
    ctlnull_emergent_sd: float
    ctlnull_emergent_p_le: float
    ctlnull_pool_size: int
    spearman_sum: float
    null_spearman_sum_mean: float
    ctlnull_spearman_sum_mean: float
    ctlnull_spearman_sum_p_ge: float


class NearReplicate(BaseModel):
    """Two single environments of one compound at one dose (within 1%) and generation."""

    compound: str
    env_a: str
    dose_a: str
    env_b: str
    dose_b: str
    generations: float
    shares_control_set: bool
    n_genes: int
    spearman: float
    hits_a: int
    hits_b: int
    b_hits_not_in_a_fraction: float
    a_hits_not_in_b_fraction: float


def pair_doses_log10_molar(names: list[str], values: str, units: str) -> list[float]:
    """Log10 molar dose of each compound of a pipe-joined two-compound environment."""
    out = []
    for name, value, unit in zip(names, values.split("|"), units.split("|")):
        v = float(value)
        if unit == "ug/mL":
            molar = v * 1e-3 / MOLAR_MASS_G_PER_MOL[name]
        else:
            molar = v * UNIT_TO_MOLAR[unit]
        out.append(float(np.log10(molar)))
    return out


def load_het() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[str]]]:
    """Gene x environment matrix (mean over each cell's list), env table, control sets."""
    cols = [
        "environment_id",
        "query_gene",
        "compound_names",
        "conc_values",
        "conc_units",
        "log10_molar",
        "n_compounds",
        "duration_generations",
        "base_medium",
        "responses",
        "screen_ids",
    ]
    t = pq.read_table(CELL_TABLE, columns=cols, filters=[("dataset", "==", DATASET)])
    flat = pc.list_flatten(t["responses"]).to_numpy()
    parent = pc.list_parent_indices(t["responses"]).to_numpy()
    sums = np.bincount(parent, weights=flat, minlength=t.num_rows)
    counts = np.bincount(parent, minlength=t.num_rows)
    assert (counts > 0).all(), "a HET cell with an empty responses list"
    means = sums / counts
    env_ids = t["environment_id"].to_numpy(zero_copy_only=False)
    genes = t["query_gene"].to_numpy(zero_copy_only=False)
    long = pd.DataFrame({"env": env_ids, "gene": genes, "y": means})
    assert not long.duplicated(["env", "gene"]).any(), "one gene twice in one env"
    matrix = long.pivot(index="gene", columns="env", values="y")

    screen_flat = pc.list_flatten(t["screen_ids"]).to_numpy(zero_copy_only=False)
    screen_parent = pc.list_parent_indices(t["screen_ids"]).to_numpy()
    controls = (
        pd.DataFrame({"env": env_ids[screen_parent], "screen": screen_flat})
        .drop_duplicates()
        .groupby("env")["screen"]
        .agg(set)
        .to_dict()
    )
    env = (
        t.drop_columns(["responses", "screen_ids", "query_gene"])
        .to_pandas()
        .drop_duplicates("environment_id")
        .set_index("environment_id")
    )
    return matrix, env, controls


def robust_z(matrix: pd.DataFrame) -> pd.DataFrame:
    """Per-environment robust z over that environment's measured genes."""
    med = matrix.median(axis=0)
    mad = (matrix - med).abs().median(axis=0) * 1.4826
    return (matrix - med) / mad


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    """Coefficient of determination of a fixed prediction (may be negative)."""
    return float(1.0 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation as a float."""
    return float(spearmanr(x, y).statistic)


def pair_label(names: list[str], values: str) -> str:
    """Short label, e.g. ``MTX125+5FU19.2``."""
    vals = values.split("|")
    return "+".join(f"{ABBREV[n]}{float(v):g}" for n, v in zip(names, vals))


def match_partners(env: pd.DataFrame, controls: dict[str, set[str]]) -> list[PairMatch]:
    """Exact-or-nearest single-compound environment for each partner of each pair."""
    singles = env[env["n_compounds"] == 1]
    pairs = env[env["n_compounds"] == 2]
    out: list[PairMatch] = []
    for pid, row in pairs.iterrows():
        names = row["compound_names"].split("|")
        doses = pair_doses_log10_molar(names, row["conc_values"], row["conc_units"])
        family = " x ".join(sorted(ABBREV[n] for n in names))
        label = pair_label(names, row["conc_values"])
        for k, (name, dose) in enumerate(zip(names, doses)):
            cand_all = singles[singles["compound_names"] == name]
            cand = cand_all[cand_all["log10_molar"].notna()].copy()
            assert len(cand) > 0, f"no dosed single for {name}"
            cand["delta"] = (cand["log10_molar"] - dose).abs().round(6)
            cand["gen_mismatch"] = (
                cand["duration_generations"] != row["duration_generations"]
            )
            best = cand.sort_values(["delta", "gen_mismatch"]).iloc[0]
            fold = float(10 ** best["delta"])
            same_gen = not bool(best["gen_mismatch"])
            if best["delta"] < 1e-3:
                kind = "exact"
            elif fold <= REASONABLE_FOLD:
                kind = "nearest_within_2fold"
            else:
                kind = "nearest_beyond_2fold"
            out.append(
                PairMatch(
                    pair_env=str(pid),
                    pair_label=label,
                    family=family,
                    partner_index=k + 1,
                    compound=name,
                    pair_dose_log10_molar=dose,
                    pair_generations=float(row["duration_generations"]),
                    single_env=str(best.name),
                    single_dose=str(best["conc_values"]),
                    single_units=str(best["conc_units"]),
                    single_dose_log10_molar=float(best["log10_molar"]),
                    single_generations=float(best["duration_generations"]),
                    dose_fold=fold,
                    match_kind=kind,
                    reasonable=(fold <= REASONABLE_FOLD) and same_gen,
                    shares_control_set=bool(controls[str(pid)] & controls[best.name]),
                    n_candidates=int(len(cand_all)),
                    candidates_without_dose=int(cand_all["log10_molar"].isna().sum()),
                )
            )
    return out


def fit_rules(y1: np.ndarray, y2: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Every rule's R^2 and Spearman, and the free linear fit."""
    preds = {
        "sum": y1 + y2,
        "mean": 0.5 * (y1 + y2),
        "max": np.maximum(y1, y2),
        "bliss": -np.log2(np.exp2(-y1) * np.exp2(-y2)),
    }
    out: dict[str, float] = {}
    for rule, p in preds.items():
        out[f"r2_{rule}"] = r2(y, p)
        out[f"spearman_{rule}"] = rho(y, p)
    out["bliss_minus_sum_max_abs"] = float(np.abs(preds["bliss"] - preds["sum"]).max())
    X = np.column_stack([y1, y2, np.ones_like(y1)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    lin = X @ coef
    out.update(
        lin_a=float(coef[0]),
        lin_b=float(coef[1]),
        lin_c=float(coef[2]),
        r2_linear=r2(y, lin),
        spearman_linear=rho(y, lin),
    )
    return out


def emergent_masked(
    hp: np.ndarray, h1: np.ndarray, h2: np.ndarray
) -> tuple[int, int, int]:
    """Emergent count, masked count, single-union hit count."""
    union = h1 | h2
    return int((hp & ~union).sum()), int((union & ~hp).sum()), int(union.sum())


def null_emergent(
    matrix: pd.DataFrame,
    hits: pd.DataFrame,
    pid: str,
    pool: list[str],
    rng: np.random.Generator,
) -> np.ndarray:
    """Emergent fraction, and Spearman of the sum rule, against random single pairs.

    Returns an (N_NULL, 2) array: column 0 the emergent fraction, column 1 the Spearman
    of y_r1 + y_r2 against the observed pair profile.
    """
    null = np.empty((N_NULL, 2))
    for i in range(N_NULL):
        r1, r2_ = rng.choice(pool, size=2, replace=False)
        sub = matrix[[r1, r2_, pid]].dropna()
        nhp = hits.loc[sub.index, pid].to_numpy()
        n_em = emergent_masked(
            nhp, hits.loc[sub.index, r1].to_numpy(), hits.loc[sub.index, r2_].to_numpy()
        )[0]
        null[i, 0] = n_em / int(nhp.sum())
        null[i, 1] = rho(sub[r1].to_numpy() + sub[r2_].to_numpy(), sub[pid].to_numpy())
    return null


def near_replicates(
    matrix: pd.DataFrame,
    hits: pd.DataFrame,
    env: pd.DataFrame,
    controls: dict[str, set[str]],
) -> list[NearReplicate]:
    """Same compound, dose within 1%, same generations: the hit-calling noise floor."""
    singles = env[(env["n_compounds"] == 1) & env["log10_molar"].notna()]
    out: list[NearReplicate] = []
    for compound, g in singles.groupby("compound_names"):
        ids = g.index.tolist()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = g.loc[ids[i]], g.loc[ids[j]]
                if abs(a["log10_molar"] - b["log10_molar"]) >= np.log10(1.01):
                    continue
                if a["duration_generations"] != b["duration_generations"]:
                    continue
                sub = matrix[[ids[i], ids[j]]].dropna()
                ha = hits.loc[sub.index, ids[i]].to_numpy()
                hb = hits.loc[sub.index, ids[j]].to_numpy()
                out.append(
                    NearReplicate(
                        compound=str(compound),
                        env_a=ids[i],
                        dose_a=str(a["conc_values"]),
                        env_b=ids[j],
                        dose_b=str(b["conc_values"]),
                        generations=float(a["duration_generations"]),
                        shares_control_set=bool(controls[ids[i]] & controls[ids[j]]),
                        n_genes=len(sub),
                        spearman=rho(sub[ids[i]].to_numpy(), sub[ids[j]].to_numpy()),
                        hits_a=int(ha.sum()),
                        hits_b=int(hb.sum()),
                        b_hits_not_in_a_fraction=float((hb & ~ha).sum() / hb.sum()),
                        a_hits_not_in_b_fraction=float((ha & ~hb).sum() / ha.sum()),
                    )
                )
    return out


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    matrix, env, controls = load_het()
    hits = robust_z(matrix).abs() > Z_HIT
    print(f"HET matrix: {matrix.shape[0]} genes x {matrix.shape[1]} environments")
    print(
        f"hit rate |robust z| > {Z_HIT}: {hits.values.sum() / matrix.notna().values.sum():.4f}"
    )

    reps = pd.DataFrame(
        [r.model_dump() for r in near_replicates(matrix, hits, env, controls)]
    )
    reps.to_csv(osp.join(RESULTS, "het_near_replicates.csv"), index=False)
    print(reps.to_string())

    matches = match_partners(env, controls)
    match_df = pd.DataFrame([m.model_dump() for m in matches])
    match_df.to_csv(osp.join(RESULTS, "het_pair_matching.csv"), index=False)

    null_pool_all = env[(env["n_compounds"] == 1) & (env["base_medium"] == "YPD")]
    rng = np.random.default_rng(SEED)
    fits: list[RuleFit] = []
    ems: list[EmergentMasked] = []
    for pid, grp in match_df.groupby("pair_env", sort=False):
        grp = grp.sort_values("partner_index")
        s1, s2 = grp["single_env"].tolist()
        flagged = not bool(grp["reasonable"].all())
        label = grp["pair_label"].iloc[0]
        family = grp["family"].iloc[0]
        sub = matrix[[s1, s2, pid]].dropna()
        y1, y2, y = (sub[c].to_numpy() for c in (s1, s2, pid))
        sim = rho(y1, y2)
        f = fit_rules(y1, y2, y)
        best_r2 = max(FIXED_RULES[:3], key=lambda r: f[f"r2_{r}"])
        best_sp = max(("sum", "max", "linear"), key=lambda r: f[f"spearman_{r}"])
        fits.append(
            RuleFit(
                pair_env=pid,
                pair_label=label,
                family=family,
                flagged=flagged,
                n_genes=len(sub),
                single_similarity_spearman=sim,
                best_rule_r2=best_r2,
                best_rule_spearman=best_sp,
                **f,
            )
        )

        hp = hits.loc[sub.index, pid].to_numpy()
        h1 = hits.loc[sub.index, s1].to_numpy()
        h2 = hits.loc[sub.index, s2].to_numpy()
        n_em, n_mask, n_union = emergent_masked(hp, h1, h2)
        partners = set(grp["compound"])
        pool = [
            e for e, n in null_pool_all["compound_names"].items() if n not in partners
        ]
        ctl_pool = [e for e in pool if controls[e] & controls[pid]]
        null = null_emergent(matrix, hits, pid, pool, rng)
        ctl_null = null_emergent(matrix, hits, pid, ctl_pool, rng)
        em_frac = n_em / int(hp.sum())
        ems.append(
            EmergentMasked(
                pair_env=pid,
                pair_label=label,
                family=family,
                flagged=flagged,
                n_genes=len(sub),
                single_similarity_spearman=sim,
                hits_pair=int(hp.sum()),
                hits_single_1=int(h1.sum()),
                hits_single_2=int(h2.sum()),
                hits_single_union=n_union,
                emergent=n_em,
                masked=n_mask,
                emergent_fraction=em_frac,
                masked_fraction=n_mask / n_union,
                null_emergent_mean=float(null[:, 0].mean()),
                null_emergent_sd=float(null[:, 0].std(ddof=1)),
                null_emergent_p_le=float(
                    (1 + (null[:, 0] <= em_frac).sum()) / (N_NULL + 1)
                ),
                null_pool_size=len(pool),
                ctlnull_emergent_mean=float(ctl_null[:, 0].mean()),
                ctlnull_emergent_sd=float(ctl_null[:, 0].std(ddof=1)),
                ctlnull_emergent_p_le=float(
                    (1 + (ctl_null[:, 0] <= em_frac).sum()) / (N_NULL + 1)
                ),
                ctlnull_pool_size=len(ctl_pool),
                spearman_sum=f["spearman_sum"],
                null_spearman_sum_mean=float(null[:, 1].mean()),
                ctlnull_spearman_sum_mean=float(ctl_null[:, 1].mean()),
                ctlnull_spearman_sum_p_ge=float(
                    (1 + (ctl_null[:, 1] >= f["spearman_sum"]).sum()) / (N_NULL + 1)
                ),
            )
        )

    fit_df = pd.DataFrame([x.model_dump() for x in fits])
    em_df = pd.DataFrame([x.model_dump() for x in ems])
    order = fit_df.sort_values(["family", "pair_label"]).index
    fit_df = fit_df.loc[order].reset_index(drop=True)
    em_df = em_df.loc[order].reset_index(drop=True)
    fit_df.to_csv(osp.join(RESULTS, "het_rule_fit.csv"), index=False)
    em_df.to_csv(osp.join(RESULTS, "het_emergent_masked.csv"), index=False)

    summary = summarize(fit_df, em_df, match_df)
    with open(osp.join(RESULTS, "het_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))

    apply_paper_style()
    ts = timestamp()
    plot_rules(fit_df, ts)
    plot_emergent(em_df, ts)
    plot_grid(fit_df, match_df, ts)


def summarize(
    fit_df: pd.DataFrame, em_df: pd.DataFrame, match_df: pd.DataFrame
) -> dict:
    """Headline numbers on all 26 pairs and on the unflagged ones."""
    out: dict = {
        "n_pairs": int(len(fit_df)),
        "n_flagged": int(fit_df["flagged"].sum()),
        "flagged_pairs": fit_df.loc[fit_df["flagged"], "pair_label"].tolist(),
        "match_kinds": match_df["match_kind"].value_counts().to_dict(),
        "pairs_sharing_control_set_with_both_singles": int(
            match_df.groupby("pair_env")["shares_control_set"].all().sum()
        ),
        "z_hit": Z_HIT,
        "n_null_draws": N_NULL,
    }
    for name, mask in (
        ("all", np.ones(len(fit_df), bool)),
        ("unflagged", ~fit_df["flagged"].to_numpy()),
    ):
        f = fit_df[mask]
        e = em_df[mask]
        block = {
            "n": int(mask.sum()),
            "best_rule_r2_counts": f["best_rule_r2"].value_counts().to_dict(),
            "best_rule_spearman_counts": f["best_rule_spearman"]
            .value_counts()
            .to_dict(),
            "median": {
                c: float(f[c].median())
                for c in (
                    "r2_sum",
                    "r2_mean",
                    "r2_max",
                    "r2_linear",
                    "spearman_sum",
                    "spearman_max",
                    "spearman_linear",
                    "lin_a",
                    "lin_b",
                    "lin_c",
                    "single_similarity_spearman",
                )
            },
            "emergent_total": int(e["emergent"].sum()),
            "pair_hits_total": int(e["hits_pair"].sum()),
            "emergent_fraction_pooled": float(
                e["emergent"].sum() / e["hits_pair"].sum()
            ),
            "masked_total": int(e["masked"].sum()),
            "single_union_hits_total": int(e["hits_single_union"].sum()),
            "masked_fraction_pooled": float(
                e["masked"].sum() / e["hits_single_union"].sum()
            ),
            "median_emergent_fraction": float(e["emergent_fraction"].median()),
            "median_null_emergent_fraction": float(e["null_emergent_mean"].median()),
            "n_pairs_null_p_le_0.05": int((e["null_emergent_p_le"] <= 0.05).sum()),
            "median_ctlnull_emergent_fraction": float(
                e["ctlnull_emergent_mean"].median()
            ),
            "n_pairs_ctlnull_p_le_0.05": int(
                (e["ctlnull_emergent_p_le"] <= 0.05).sum()
            ),
            "median_null_spearman_sum": float(e["null_spearman_sum_mean"].median()),
            "median_ctlnull_spearman_sum": float(
                e["ctlnull_spearman_sum_mean"].median()
            ),
            "n_pairs_ctlnull_spearman_sum_p_le_0.05": int(
                (e["ctlnull_spearman_sum_p_ge"] <= 0.05).sum()
            ),
        }
        for col in ("emergent_fraction", "masked_fraction"):
            res = spearmanr(e["single_similarity_spearman"], e[col])
            block[f"spearman_similarity_vs_{col}"] = {
                "rho": float(res.statistic),
                "p": float(res.pvalue),
                "n": int(len(e)),
            }
        out[name] = block
    grid = fit_df[fit_df["family"] == "5FU x MTX"]
    out["mtx_5fu_grid"] = {
        "n": int(len(grid)),
        "best_rule_r2_counts": grid["best_rule_r2"].value_counts().to_dict(),
        **{
            f"{c}_range": [float(grid[c].min()), float(grid[c].max())]
            for c in (
                "lin_a",
                "lin_b",
                "lin_c",
                "r2_linear",
                "r2_sum",
                "r2_mean",
                "r2_max",
            )
        },
    }
    return out


def style_axes(ax: plt.Axes) -> None:
    """All four spines, 0.5 pt."""
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
        s.set_color("black")


def save(fig: plt.Figure, stem: str, ts: str) -> None:
    """Write the .svg (true size) and the .png beside it."""
    base = osp.join(IMAGE_DIR, f"{stem}_{ts}")
    savefig_true_size_svg(fig, base + ".svg")
    fig.savefig(base + ".png", dpi=300)
    plt.close(fig)
    print("wrote", base + ".svg")


RULE_STYLE = {
    "sum": (PLOT_PALETTE[0], "o", "sum (= Bliss)"),
    "mean": (PLOT_PALETTE[1], "s", "mean"),
    "max": (PLOT_PALETTE[2], "^", "max"),
    "linear": (PLOT_PALETTE[3], "D", "free linear"),
}


def plot_rules(fit_df: pd.DataFrame, ts: str) -> None:
    """Per-environment R^2 and Spearman of every rule, grouped by pair family."""
    fig, axes = plt.subplots(
        2, 1, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(130)), sharex=True
    )
    x = np.arange(len(fit_df))
    floor = -1.0
    for ax, metric in zip(axes, ("r2", "spearman")):
        for rule, (color, marker, name) in RULE_STYLE.items():
            if metric == "spearman" and rule == "mean":
                continue
            v = fit_df[f"{metric}_{rule}"].to_numpy()
            clipped = v < floor
            ax.scatter(
                x[~clipped],
                v[~clipped],
                s=9,
                color=color,
                marker=marker,
                edgecolor="black",
                linewidth=0.3,
                label=name,
                zorder=3,
            )
            if clipped.any():
                ax.scatter(
                    x[clipped],
                    np.full(clipped.sum(), floor + 0.03),
                    s=9,
                    color=color,
                    marker="v",
                    edgecolor="black",
                    linewidth=0.3,
                    zorder=3,
                )
        if metric == "spearman":
            ax.scatter(
                x,
                fit_df["single_similarity_spearman"],
                s=9,
                color="white",
                marker="o",
                edgecolor="black",
                linewidth=0.4,
                label="single vs single",
                zorder=3,
            )
        fam = fit_df["family"].to_numpy()
        for i in range(1, len(fam)):
            if fam[i] != fam[i - 1]:
                ax.axvline(i - 0.5, color="black", linewidth=0.4)
        for i in np.flatnonzero(fit_df["flagged"].to_numpy()):
            ax.axvspan(i - 0.5, i + 0.5, color="#E6E6E6", zorder=0, linewidth=0)
        ax.set_ylim(floor, 1.0)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(axis="y", which="minor", length=0)
        ax.grid(axis="y", which="both", linewidth=0.3, color="#CCCCCC", zorder=0)
        ax.axhline(0, color="black", linewidth=0.4)
        ax.set_xlim(-0.5, len(fit_df) - 0.5)
        style_axes(ax)
    axes[0].set_ylabel("R$^2$ of rule vs observed pair profile\n(below -1 drawn as ▼)")
    axes[1].set_ylabel("Spearman vs observed pair profile")
    axes[0].legend(ncol=4, loc="lower left", frameon=False, handletextpad=0.2)
    axes[1].legend(ncol=4, loc="lower left", frameon=False, handletextpad=0.2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(fit_df["pair_label"], rotation=90)
    fams = (
        pd.Series(np.arange(len(fit_df)), index=fit_df.index)
        .groupby(fit_df["family"], sort=False)
        .mean()
    )
    for k, (name, pos) in enumerate(fams.items()):
        axes[0].text(
            pos,
            1.02 + 0.05 * (k % 2),
            name,
            ha="center",
            va="bottom",
            fontsize=5,
            transform=axes[0].get_xaxis_transform(),
        )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.2, hspace=0.12)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    save(fig, "het_rule_fit", ts)


def plot_emergent(em_df: pd.DataFrame, ts: str) -> None:
    """Emergent and masked fractions against single-profile similarity."""
    fig, axes = plt.subplots(
        1, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(58))
    )
    fl = em_df["flagged"].to_numpy()
    sim = em_df["single_similarity_spearman"].to_numpy()
    for ax, col, color, title in (
        (axes[0], "emergent_fraction", PLOT_PALETTE[0], "emergent / pair hits"),
        (axes[1], "masked_fraction", PLOT_PALETTE[1], "masked / single hits"),
    ):
        v = em_df[col].to_numpy()
        ax.scatter(
            sim[~fl],
            v[~fl],
            s=10,
            color=color,
            edgecolor="black",
            linewidth=0.3,
            label="matched dose",
            zorder=3,
        )
        ax.scatter(
            sim[fl],
            v[fl],
            s=10,
            facecolor="white",
            edgecolor=color,
            linewidth=0.6,
            label="flagged (FK506)",
            zorder=3,
        )
        if col == "emergent_fraction":
            ax.scatter(
                sim,
                em_df["null_emergent_mean"],
                s=6,
                color=PLOT_PALETTE[5],
                marker="_",
                linewidth=0.6,
                label="null mean",
                zorder=2,
            )
            ax.scatter(
                sim,
                em_df["ctlnull_emergent_mean"],
                s=6,
                color=PLOT_PALETTE[4],
                marker="x",
                linewidth=0.5,
                label="control-set null",
                zorder=2,
            )
        res = spearmanr(sim, v)
        ax.set_title(f"{title}\nSpearman {res.statistic:.2f} (n = {len(v)})")
        ax.set_xlabel("Spearman, single 1 vs single 2")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(axis="y", which="minor", length=0)
        ax.grid(axis="y", which="both", linewidth=0.3, color="#CCCCCC", zorder=0)
        style_axes(ax)
    axes[0].set_ylabel("fraction")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        ncol=4,
        handletextpad=0.2,
        columnspacing=0.8,
    )
    fig.subplots_adjust(left=0.11, right=0.98, top=0.83, bottom=0.27, wspace=0.3)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    save(fig, "het_emergent_masked", ts)


def plot_grid(fit_df: pd.DataFrame, match_df: pd.DataFrame, ts: str) -> None:
    """Methotrexate x 5-fluorouracil: linear coefficients and rule R^2 per dose pair."""
    grid = fit_df[fit_df["family"] == "5FU x MTX"].copy()
    doses = match_df.drop_duplicates("pair_env").set_index("pair_env")["pair_label"]
    grid["mtx"] = grid["pair_label"].str.extract(r"MTX([\d.]+)")[0].astype(float)
    grid["fu"] = grid["pair_label"].str.extract(r"5FU([\d.]+)")[0].astype(float)
    assert set(grid["pair_env"]) <= set(doses.index)
    mtx = sorted(grid["mtx"].unique())
    fu = sorted(grid["fu"].unique())
    cmap = LinearSegmentedColormap.from_list(
        "div", [PLOT_PALETTE[4], "#FFFFFF", PLOT_PALETTE[1]]
    )
    panels = (
        ("lin_a", "a (MTX coefficient)", (-1.5, 1.5)),
        ("lin_b", "b (5FU coefficient)", (-1.5, 1.5)),
        ("r2_linear", "R$^2$ free linear", (-1, 1)),
        ("r2_mean", "R$^2$ mean rule", (-1, 1)),
    )
    fig, axes = plt.subplots(
        1, 4, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(48))
    )
    for ax, (col, title, lim), letter in zip(axes, panels, "abcd"):
        m = grid.pivot(index="mtx", columns="fu", values=col).loc[mtx, fu].to_numpy()
        ax.imshow(m, cmap=cmap, vmin=lim[0], vmax=lim[1], origin="lower", aspect="auto")
        for i in range(len(mtx)):
            for j in range(len(fu)):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=6)
        ax.set_xticks(range(len(fu)))
        ax.set_xticklabels([f"{v:g}" for v in fu])
        ax.set_yticks(range(len(mtx)))
        ax.set_yticklabels([f"{v:g}" for v in mtx])
        ax.set_xlabel("5FU (uM)")
        ax.set_title(title)
        style_axes(ax)
        if letter == "a":
            ax.set_ylabel("MTX (uM)")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.8, bottom=0.2, wspace=0.35)
    for ax, letter in zip(axes, "abcd"):
        panel_label(ax, letter)
    save(fig, "het_mtx_5fu_grid", ts)


if __name__ == "__main__":
    main()
