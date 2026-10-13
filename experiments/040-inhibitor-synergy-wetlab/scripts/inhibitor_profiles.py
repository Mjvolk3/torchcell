# experiments/040-inhibitor-synergy-wetlab/scripts/inhibitor_profiles.py
# [[experiments.040-inhibitor-synergy-wetlab.scripts.inhibitor_profiles]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/inhibitor_profiles
"""Chemogenomic deletion profiles of the six 2021 inhibitors, measured and predicted.

THE SIX INHIBITORS are the ones the private loader
(``torchcell.datasets.private_torchcell.volk2021_inhibitor_bioscreen``) doses, under the
names it serves: furfural, acetic acid, 5-(hydroxymethyl)furfural, formic acid,
levulinic acid, lactic acid.

SIGN. Every profile value is Vanacloig 2022's ``log2(inhibitor / control)`` for one
deletion strain: NEGATIVE = the deletion is SICK under the inhibitor (sensitive),
positive = resistant. The Hoepfner HOP comparison profile is the adjusted MADL
sensitivity score, also negative = hypersensitive.

MEASURED PROFILES.

- furfural, 5-(hydroxymethyl)furfural: build 002 of the 033 cell table (TMM, the 32
  published compounds; ``vanacloig_data.load_cells`` of experiment 038).
- sodium acetate, Hoepfner 2014 HOP (homozygous diploid deletions, YPD, aerobic; 125 and
  150 mM averaged; adjusted MADL sensitivity score): the measured profile acetic acid
  takes in ``best_available``. Cross-screen: a different strain collection, medium,
  oxygen regime and readout scale than Vanacloig, so it is compared by rank only.
- levulinic acid, sodium acetate in Vanacloig: NOT in build 002. Issue #501 dropped the
  nine served compounds the paper never reports (GEO matrix only), and these two are
  among them (``vanacloig_data.UNREPORTED_COMPOUNDS``). Build 001 (the 2026.09.21 store,
  CPM) still serves them; they are read from it for SUPPLEMENTARY comparison rows only,
  centered by build 001's own gene mean over the same 32 published compounds, and they
  enter no matrix.

PREDICTED PROFILES are nested kernel ridge on FCFP4 counts exactly as experiment 038's
``baseline_ladder.py`` builds its ``krr | linear:fcfp4_count`` row (its functions are
imported, not copied): the linear kernel on features standardized over the fitted
compounds, (lambda, rank, clip) chosen by leave-one-compound-out inside the fitted pool
on the centered target, then one prediction.

- Leave-one-compound-out over the 32 build-002 compounds: each compound predicted from a
  fit on the other 31. The median over all 32 checks this reimplementation against the
  038 bar (median centered Spearman 0.359 over 96 compound-evaluations of 5-fold
  compound-cold splits, three fold seeds); furfural and 5-HMF are the inhibitors.
- levulinic acid, sodium acetate, acetic acid, formic acid, lactic acid: fit on all 32
  and predicted from their FCFP4 count fingerprints. None of the five is in the 32, so
  each is compound-cold. Formic and lactic acid are in no screen.

FINGERPRINT RECIPE (031 ``embed_compounds.py`` with ``torchcell.molecule.encoders``
``FCFP4Count`` of the 031 worktree; that package is not on this branch, so the three
lines that define it are restated in :func:`fcfp4_count`): RDKit
``GetMorganGenerator(radius=2, fpSize=2048, atomInvariantsGenerator=
GetMorganFeatureAtomInvGen())``, ``GetCountFingerprintAsNumPy`` on the parsed SMILES, cast
to float32; the structure is the curated identity-table SMILES. The script re-featurizes
every Vanacloig compound and every inhibitor that has a row in the 031 npz and stops if
any row differs.

SCORES are 038's: per compound, Spearman and Pearson over genes, raw and centered. The
centered target subtracts the gene's mean over the fitted compounds from the measurement
and the PREDICTED gene mean over the same compounds from the prediction.

THE TWO MATRICES (genes by the six inhibitors):

- ``all_predicted``: every column a ridge prediction (furfural and 5-HMF held out).
- ``best_available``: a build-002 measured profile where its replicate reliability
  ``1 - mean(SE^2) / Var(response)`` is at least ``RELIABILITY_MIN`` (furfural passes,
  5-HMF does not), the Hoepfner HOP SODIUM ACETATE profile for acetic acid (the plan's
  stated identity assumption, same anion at YPD pH, plus the cross-screen caveat), and
  the prediction everywhere else. Every prediction of acetic acid is from the acid's
  own structure.

SIMILARITY for the 15 pairs, per matrix and target: Spearman over the genes both columns
carry, and the Jaccard index of the ``TOP_N`` most sensitive genes (lowest values).

Writes ``results/profile_*.csv`` and ``.json`` and two figures under
``ASSET_IMAGES_DIR/040-inhibitor-synergy-wetlab``.
"""

from __future__ import annotations

import itertools
import json
import os
import os.path as osp
import sys
import warnings
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from joblib import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap
from numpy.typing import NDArray
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import pearsonr, spearmanr

from torchcell.datamodels.compound_identity import _TABLE_PATH
from torchcell.datasets.private_torchcell import bioscreen as bs
from torchcell.datasets.private_torchcell import volk2021_inhibitor_bioscreen as volk
from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
EXP_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
RESULTS_DIR = osp.join(EXP_DIR, "results")
IMAGE_DIR = osp.join(os.environ["ASSET_IMAGES_DIR"], "040-inhibitor-synergy-wetlab")

WORKTREES = "/home/michaelvolk/Documents/projects/torchcell.worktrees/exp"
EXP_038 = osp.join(
    WORKTREES,
    "038-env-chemgen-vanacloig-cgt-corrected",
    "experiments",
    "038-env-chemgen-vanacloig-cgt-corrected",
)
sys.path.insert(0, osp.join(EXP_038, "scripts"))
from baseline_ladder import (  # noqa: E402
    Candidate,
    fold_kernel,
    grid,
    inner_scores,
    predict_candidates,
)
from vanacloig_data import CELL_TABLE, VANACLOIG, load_cells  # noqa: E402

# a gene measured under no fitted compound has no mean; it is predicted NaN, unscored
warnings.filterwarnings("ignore", message="Mean of empty slice")

#: Build 001 of the 033 cell table (the 2026.09.21 store, CPM, 41 Vanacloig compounds).
CELL_TABLE_001 = (
    "/scratch/projects/torchcell-scratch/experiments/033-env-chemgen-pooled/"
    "cell_table/cell_table.parquet"
)
FCFP4_NPZ = osp.join(
    WORKTREES,
    "031-env-chemgen-vanacloig-hillenmeyer",
    "experiments",
    "031-env-chemgen-inhibitor-tolerance",
    "results",
    "embeddings",
    "fcfp4_count.npz",
)
#: the sha256-pinned curated identity table, as 031 ``load_identity_table`` reads it
IDENTITY_TABLE = str(_TABLE_PATH)
LADDER_SCORES_038 = osp.join(EXP_038, "results", "ladder", "ladder_r2_scores.csv")
HOEPFNER = "EnvChemgenHoepfner2014Dataset"
RIDGE_KERNEL = "linear:fcfp4_count"
#: A measured profile enters ``best_available`` only at this replicate reliability or
#: above. A choice, not a sourced value: furfural (0.77 on build 001) passes, 5-HMF
#: (-0.05) and levulinic acid (0.15) do not.
RELIABILITY_MIN = 0.5
TOP_N = 100
SIGN = "log2(inhibitor/control); NEGATIVE = sick (sensitive deletion), positive = resistant"

Source = Literal[
    "measured_vanacloig_b002",
    "measured_vanacloig_b001",
    "measured_hoepfner_hop",
    "predicted_ridge_loco",
    "predicted_ridge_all32",
]


class Inhibitor(BaseModel):
    """One of the six 2021 inhibitors, under the private loader's name."""

    abbreviation: str  # the archive's abbreviation (bioscreen.Inhibitor)
    name: str  # Compound.name the private loader serves
    smiles: str  # the structure the prediction is made from
    smiles_source: str
    inchikey: str  # derived from ``smiles`` with RDKit
    vanacloig_name: str | None  # the Vanacloig compound the measured profile is
    vanacloig_build: Literal["002", "001"] | None


class Profile(BaseModel):
    """One gene profile, raw and centered, over ``genes`` of build 002."""

    model_config = {"arbitrary_types_allowed": True}

    label: str
    source: Source
    raw: NDArray[np.float64]
    centered: NDArray[np.float64]
    #: measured: replicate reliability; predicted: None
    reliability: float | None
    options: str | None = None
    inner_centered_spearman: float | None = None


class ProfileChoice(BaseModel):
    """Which profile fills one inhibitor's column of one matrix, and why."""

    version: Literal["best_available", "all_predicted"]
    inhibitor: str
    abbreviation: str
    profile: str
    source: Source
    measured_reliability: float | None
    #: compound-cold centered Spearman of the prediction against a measured profile of
    #: the same compound (for acetic acid: against measured sodium acetate); None when no
    #: measurement exists
    prediction_check_spearman_centered: float | None
    reason: str


# --------------------------------------------------------------------------- fingerprints
def fcfp4_count(smiles: str) -> NDArray[np.float32]:
    """FCFP4 count fingerprint, 2048 bits: 031 ``torchcell.molecule.encoders.FCFP4Count``.

    Restated from the 031 worktree (``_MorganEncoder`` with ``feature_invariants=True``,
    ``parse_smiles``), which this branch does not carry.
    """
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"unparsable SMILES: {smiles!r}"
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
    )
    return gen.GetCountFingerprintAsNumPy(mol).astype(np.float32)


def inchikey(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"unparsable SMILES: {smiles!r}"
    return str(Chem.MolToInchiKey(mol))


def identity_smiles() -> dict[str, str]:
    """InChIKey -> curated SMILES from the sha256-pinned compound identity table."""
    with open(IDENTITY_TABLE) as f:
        records = json.load(f)["records"]
    return {
        r["inchikey"]: r["smiles"]
        for r in records
        if r.get("inchikey") and r.get("smiles")
    }


def verify_recipe(keys: list[str], names: list[str]) -> pd.DataFrame:
    """Re-featurize every compound in ``keys`` from its curated SMILES; stop on a mismatch."""
    npz = np.load(FCFP4_NPZ, allow_pickle=True)
    row = {k: i for i, k in enumerate(npz["inchikey"])}
    curated = identity_smiles()
    rows = []
    for key, name in zip(keys, names, strict=True):
        assert key in row, f"{name} ({key}) has no row in the 031 npz"
        assert key in curated, f"{name} ({key}) has no curated SMILES"
        x = fcfp4_count(curated[key])
        stored = npz["X"][row[key]]
        rows.append(
            {
                "compound": name,
                "inchikey": key,
                "smiles": curated[key],
                "rdkit_inchikey": inchikey(curated[key]),
                "n_bits_set": int((x > 0).sum()),
                "count_sum": float(x.sum()),
                "max_abs_difference": float(np.abs(x - stored).max()),
                "exact_match": bool(np.array_equal(x, stored)),
            }
        )
    out = pd.DataFrame(rows)
    bad = out[~out["exact_match"]]
    assert bad.empty, f"FCFP4 recipe does not reproduce the 031 npz:\n{bad}"
    return out


def inhibitors() -> list[Inhibitor]:
    """The six inhibitors with the loader's names and the structures predicted from."""
    vanacloig = {
        bs.Inhibitor.FF: ("furfural", "002"),
        bs.Inhibitor.AA: ("sodium acetate", "001"),
        bs.Inhibitor.HMF: ("5-(hydroxymethyl)furfural", "002"),
        bs.Inhibitor.FA: (None, None),
        bs.Inhibitor.LVA: ("levulinic acid", "001"),
        bs.Inhibitor.LA: (None, None),
    }
    # formic and lactic acid carry an identity gap in the loader (no InChIKey, no
    # SMILES); these are the plain structures, lactic acid without stereo because the
    # stock sheet does not state the enantiomer (LACTIC_ACID_IDENTITY_GAP)
    given = {bs.Inhibitor.FA: "OC=O", bs.Inhibitor.LA: "CC(O)C(=O)O"}
    out = []
    for inh in bs.INHIBITORS:
        compound = volk.compound(inh)
        if inh in given:
            assert compound.smiles is None and compound.inchikey is None
            smiles, smiles_source = (
                given[inh],
                "assigned here (loader has an identity gap)",
            )
        else:
            smiles, smiles_source = compound.smiles, "private loader Compound.smiles"
            assert inchikey(smiles) == compound.inchikey, inh
        v_name, v_build = vanacloig[inh]
        out.append(
            Inhibitor(
                abbreviation=inh.value,
                name=compound.name,
                smiles=smiles,
                smiles_source=smiles_source,
                inchikey=inchikey(smiles),
                vanacloig_name=v_name,
                vanacloig_build=v_build,
            )
        )
    return out


# --------------------------------------------------------------------------- measurements
def reliability(response: NDArray[np.float64], se: NDArray[np.float64]) -> float:
    """``1 - mean(SE^2) / Var(response)`` over the genes with both (038 ``ceiling`` squared)."""
    ok = np.isfinite(response) & np.isfinite(se)
    return 1.0 - float(np.mean(se[ok] ** 2)) / float(np.var(response[ok], ddof=1))


def build001_matrix(
    genes: list[str], compounds: list[str]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build 001 Vanacloig responses and SEs as build-002 genes by ``compounds``."""
    table = pd.read_parquet(
        CELL_TABLE_001,
        columns=[
            "dataset",
            "query_gene",
            "n_measurements",
            "n_compounds",
            "compound_names",
            "responses",
            "response_ses",
        ],
        filters=[("dataset", "==", VANACLOIG)],
    )
    assert (table["n_measurements"] == 1).all() and (table["n_compounds"] == 1).all()
    gene_index = {g: i for i, g in enumerate(genes)}
    compound_index = {c: j for j, c in enumerate(compounds)}
    table = table[table["compound_names"].isin(compound_index)]
    assert set(table["compound_names"]) == set(compounds), (
        "a compound is not in build 001"
    )
    kept = table[table["query_gene"].isin(gene_index)]
    print(
        f"build 001: {table['query_gene'].nunique()} queried genes, "
        f"{kept['query_gene'].nunique()} of them in build 002's {len(genes)}"
    )
    y = np.full((len(genes), len(compounds)), np.nan)
    se = np.full_like(y, np.nan)
    gi = kept["query_gene"].map(gene_index).to_numpy()
    cj = kept["compound_names"].map(compound_index).to_numpy()
    y[gi, cj] = kept["responses"].map(lambda r: r[0]).to_numpy(dtype=np.float64)
    se[gi, cj] = kept["response_ses"].map(lambda s: s[0]).to_numpy(dtype=np.float64)
    return y, se


class HoepfnerAcetate(BaseModel):
    """Hoepfner 2014 HOP sodium acetate over build-002 genes, raw and centered."""

    model_config = {"arbitrary_types_allowed": True}

    raw: NDArray[np.float64]  # mean of the 125 and 150 mM columns
    centered: NDArray[np.float64]  # minus the gene's mean over every HOP column
    n_hop_genes: int
    n_in_build002: int
    n_hop_columns: int  # distinct HOP environments the gene mean is taken over
    dose_spearman: float  # 125 mM against 150 mM over the genes both carry
    dose_n_genes: int


def hoepfner_hop_acetate(genes: list[str]) -> HoepfnerAcetate:
    """Hoepfner 2014 HOP sodium acetate, mean over its two dose columns.

    HOP rows are the ``barcoded_kanmx_deletion`` strains (homozygous diploid deletions;
    HIP rows are ``heterozygous_deletion``). A cell folding two measurements takes their
    mean. The centered profile subtracts the gene's mean over every HOP environment in
    the table, the Hoepfner analogue of the Vanacloig centering over the 32 compounds.
    The agreement of the two doses is the replicate check this profile has: the table
    carries no per-gene SE for it.
    """
    table = pd.read_parquet(
        CELL_TABLE,
        columns=[
            "dataset",
            "query_gene",
            "perturbation_type",
            "environment_id",
            "compound_names",
            "log10_molar",
            "responses",
        ],
        filters=[
            ("dataset", "==", HOEPFNER),
            ("perturbation_type", "==", "barcoded_kanmx_deletion"),
        ],
    )
    table["value"] = table["responses"].map(lambda r: float(np.mean(r)))
    gene_mean = table.groupby("query_gene")["value"].mean()
    acetate = table[table["compound_names"] == "sodium acetate"]
    assert acetate["log10_molar"].nunique() == 2, "expected the two HOP acetate doses"
    per_dose = acetate.groupby(["query_gene", "log10_molar"])["value"].mean().unstack()
    both = per_dose.dropna()
    raw = per_dose.mean(axis=1)
    centered = raw - gene_mean.reindex(raw.index)
    out = HoepfnerAcetate(
        raw=raw.reindex(genes).to_numpy(dtype=np.float64),
        centered=centered.reindex(genes).to_numpy(dtype=np.float64),
        n_hop_genes=len(raw),
        n_in_build002=int(raw.index.isin(genes).sum()),
        n_hop_columns=int(table["environment_id"].nunique()),
        dose_spearman=float(spearmanr(both.iloc[:, 0], both.iloc[:, 1])[0]),
        dose_n_genes=len(both),
    )
    print(
        f"Hoepfner HOP sodium acetate: {out.n_hop_genes} genes, {out.n_in_build002} in "
        f"build 002; {out.n_hop_columns} HOP columns; 125 vs 150 mM Spearman "
        f"{out.dose_spearman:.3f} over {out.dose_n_genes} genes"
    )
    return out


# --------------------------------------------------------------------------- ridge
def ridge_candidates() -> list[Candidate]:
    return [c for c in grid(RIDGE_KERNEL) if c.model == "krr"]


def fit_predict(
    x: NDArray[np.float64], y: NDArray[np.float64], pool: list[int], targets: list[int]
) -> tuple[NDArray[np.float64], NDArray[np.float64], Candidate, float]:
    """038's nested ridge: pick (lambda, rank, clip) by inner LOCO over ``pool``, predict.

    ``x`` holds every compound's features (rows index ``pool`` and ``targets``); ``y`` is
    genes by the measured compounds (columns index ``pool``). Returns the predicted raw
    profiles of ``targets`` and the predicted gene mean over ``pool``.
    """
    kernel = fold_kernel("linear", x, np.array(pool))
    candidates = ridge_candidates()
    inner = inner_scores(candidates, kernel, y, pool)
    best = int(np.nanargmax(inner))
    pred = predict_candidates([candidates[best]], kernel, y, pool, pool + targets)[0]
    return (
        pred[:, len(pool) :],
        pred[:, : len(pool)].mean(axis=1),
        candidates[best],
        float(inner[best]),
    )


def loco(
    j: int, x: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[int, NDArray[np.float64], NDArray[np.float64], Candidate, float]:
    pool = [i for i in range(y.shape[1]) if i != j]
    pred, pred_mean, chosen, inner = fit_predict(x, y, pool, [j])
    return j, pred[:, 0], pred_mean, chosen, inner


def score(
    pred: NDArray[np.float64], obs: NDArray[np.float64]
) -> tuple[int, float, float]:
    ok = np.isfinite(pred) & np.isfinite(obs)
    return (
        int(ok.sum()),
        float(spearmanr(pred[ok], obs[ok])[0]),
        float(pearsonr(pred[ok], obs[ok])[0]),
    )


# --------------------------------------------------------------------------- similarity
def top_sensitive(v: NDArray[np.float64], n: int) -> set[int]:
    ok = np.flatnonzero(np.isfinite(v))
    return set(ok[np.argsort(v[ok], kind="stable")[:n]].tolist())


def similarity(a: NDArray[np.float64], b: NDArray[np.float64]) -> dict[str, float]:
    ok = np.isfinite(a) & np.isfinite(b)
    ta, tb = top_sensitive(a, TOP_N), top_sensitive(b, TOP_N)
    return {
        "n_genes": int(ok.sum()),
        "spearman": float(spearmanr(a[ok], b[ok])[0]),
        "jaccard_top": len(ta & tb) / len(ta | tb),
    }


# --------------------------------------------------------------------------- figures
def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.titlesize": 6,
            "axes.labelsize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
            "svg.fonttype": "none",
        }
    )


def save(fig: plt.Figure, title: str) -> str:
    os.makedirs(IMAGE_DIR, exist_ok=True)
    stem = osp.join(IMAGE_DIR, f"{title}_{timestamp()}")
    savefig_true_size_svg(fig, stem + ".svg")
    fig.savefig(stem + ".png", dpi=300)
    plt.close(fig)
    print(f"wrote {stem}.svg")
    return stem + ".svg"


def scatter_figure(
    panels: list[tuple[str, str, str, NDArray[np.float64], NDArray[np.float64]]],
) -> str:
    """Predicted (x) against measured (y), centered, one panel per comparison."""
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(42)),
        constrained_layout=True,
    )
    for k, (ax, (title, xlab, ylab, pred, obs)) in enumerate(
        zip(axes, panels, strict=True)
    ):
        ok = np.isfinite(pred) & np.isfinite(obs)
        rho = spearmanr(pred[ok], obs[ok])[0]
        ax.scatter(
            pred[ok], obs[ok], s=1.0, lw=0, color=PLOT_PALETTE[k], rasterized=True
        )
        ax.axhline(0, color="black", lw=0.3)
        ax.axvline(0, color="black", lw=0.3)
        ax.set_title(title)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.text(
            0.03,
            0.97,
            f"Spearman {rho:.2f}\nn = {int(ok.sum())} genes",
            transform=ax.transAxes,
            va="top",
            ha="left",
        )
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
    return save(fig, "profile_predicted_vs_measured")


def heatmap_figure(sim: pd.DataFrame, order: list[Inhibitor], target: str) -> str:
    """6 x 6 similarity, both matrices, Spearman and top-100 Jaccard."""
    abbr = [i.abbreviation for i in order]
    names = [i.name for i in order]
    diverging = LinearSegmentedColormap.from_list(
        "bwr_palette", [PLOT_PALETTE[4], "#FFFFFF", PLOT_PALETTE[1]]
    )
    sequential = LinearSegmentedColormap.from_list(
        "w_amber", ["#FFFFFF", PLOT_PALETTE[0]]
    )
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(50)),
        constrained_layout=True,
    )
    specs = [
        ("best_available", f"spearman_{target}", "Spearman"),
        ("all_predicted", f"spearman_{target}", "Spearman"),
        ("best_available", f"jaccard_top{TOP_N}_{target}", f"Jaccard, top {TOP_N}"),
        ("all_predicted", f"jaccard_top{TOP_N}_{target}", f"Jaccard, top {TOP_N}"),
    ]
    spearman_max = float(sim[[f"spearman_{target}"]].abs().max().iloc[0])
    for ax, (version, column, label) in zip(axes, specs, strict=True):
        m = np.full((6, 6), np.nan)
        part = sim[sim["version"] == version]
        for _, r in part.iterrows():
            a, b = names.index(r["inhibitor_a"]), names.index(r["inhibitor_b"])
            m[a, b] = m[b, a] = r[column]
        is_spearman = column.startswith("spearman")
        if is_spearman:
            np.fill_diagonal(m, 1.0)
        im = ax.imshow(
            m,
            cmap=diverging if is_spearman else sequential,
            vmin=-spearman_max if is_spearman else 0.0,
            vmax=spearman_max if is_spearman else float(sim[column].max()),
        )
        for a in range(6):
            for b in range(6):
                if a != b:
                    ax.text(
                        b, a, f"{m[a, b]:.2f}", ha="center", va="center", fontsize=5
                    )
        ax.set_xticks(range(6), abbr)
        ax.set_yticks(range(6), abbr)
        ax.set_title(f"{version.replace('_', '-')}\n{label} ({target})")
        cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(width=0.5, length=2)
    return save(fig, f"profile_similarity_{target}")


# --------------------------------------------------------------------------- main
def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    style()
    six = inhibitors()
    for inh in six:
        print(f"{inh.abbreviation}: {inh.name!r} {inh.smiles} {inh.inchikey}")

    cells = load_cells(CELL_TABLE, FCFP4_NPZ)
    genes = cells.genes
    n32 = len(cells.compounds)
    assert n32 == 32
    y = cells.matrix(cells.response)
    se002 = cells.matrix(cells.response_se)
    x32 = cells.compound_features
    idx = {c: j for j, c in enumerate(cells.compounds)}

    # ---- recipe verification: every Vanacloig compound and every inhibitor in the npz
    npz_keys = set(np.load(FCFP4_NPZ, allow_pickle=True)["inchikey"])
    extra = [
        i for i in six if i.inchikey in npz_keys and i.inchikey not in cells.inchikeys
    ]
    acetate_key = inchikey("CC(=O)[O-].[Na+]")
    check = verify_recipe(
        cells.inchikeys + [i.inchikey for i in extra] + [acetate_key],
        cells.compounds + [i.name for i in extra] + ["sodium acetate"],
    )
    check.to_csv(osp.join(RESULTS_DIR, "profile_fingerprint_check.csv"), index=False)
    print(
        f"FCFP4 recipe: {int(check['exact_match'].sum())} of {len(check)} compounds reproduce the npz row exactly"
    )

    # ---- measured
    mean002 = np.nanmean(y, axis=1)
    b001_compounds = list(cells.compounds) + ["levulinic acid", "sodium acetate"]
    y001, se001 = build001_matrix(genes, b001_compounds)
    mean001 = np.nanmean(y001[:, :n32], axis=1)
    profiles: dict[str, Profile] = {}
    for name in ("furfural", "5-(hydroxymethyl)furfural"):
        j = idx[name]
        profiles[f"{name}|measured_b002"] = Profile(
            label=f"{name}|measured_b002",
            source="measured_vanacloig_b002",
            raw=y[:, j],
            centered=y[:, j] - mean002,
            reliability=reliability(y[:, j], se002[:, j]),
        )
    for name in (
        "levulinic acid",
        "sodium acetate",
        "furfural",
        "5-(hydroxymethyl)furfural",
    ):
        j = b001_compounds.index(name)
        profiles[f"{name}|measured_b001"] = Profile(
            label=f"{name}|measured_b001",
            source="measured_vanacloig_b001",
            raw=y001[:, j],
            centered=y001[:, j] - mean001,
            reliability=reliability(y001[:, j], se001[:, j]),
        )
    hop = hoepfner_hop_acetate(genes)
    profiles["sodium acetate|measured_hoepfner_hop"] = Profile(
        label="sodium acetate|measured_hoepfner_hop",
        source="measured_hoepfner_hop",
        raw=hop.raw,
        centered=hop.centered,
        reliability=None,
    )
    for p in profiles.values():
        print(f"{p.label}: reliability {p.reliability}")

    # ---- leave-one-compound-out over the 32
    results = Parallel(n_jobs=8)(delayed(loco)(j, x32, y) for j in range(n32))
    loco_rows = []
    loco_pred: dict[
        str, tuple[NDArray[np.float64], NDArray[np.float64], Candidate, float]
    ] = {}
    for j, pred, pred_mean, chosen, inner in sorted(results, key=lambda r: r[0]):
        name = cells.compounds[j]
        loco_pred[name] = (pred, pred_mean, chosen, inner)
        pool = [i for i in range(n32) if i != j]
        measured_mean = np.nanmean(y[:, pool], axis=1)
        for target, p, o in (
            ("raw", pred, y[:, j]),
            ("centered", pred - pred_mean, y[:, j] - measured_mean),
        ):
            n, rho, r = score(p, o)
            loco_rows.append(
                {
                    "comparison": "loco_32",
                    "compound": name,
                    "predicted": f"{name}|predicted_loco",
                    "measured": f"{name}|measured_b002",
                    "target": target,
                    "n_genes": n,
                    "spearman": rho,
                    "pearson": r,
                    "measured_reliability": reliability(y[:, j], se002[:, j]),
                    "options": chosen.options(),
                    "inner_centered_spearman": inner,
                }
            )
    loco_df = pd.DataFrame(loco_rows)
    median_centered = float(loco_df.query("target == 'centered'")["spearman"].median())
    print(
        f"LOCO over 32: median centered Spearman {median_centered:.3f} (038 bar 0.359, 5-fold, 96 evaluations)"
    )
    for name in ("furfural", "5-(hydroxymethyl)furfural"):
        pred, pred_mean, chosen, inner = loco_pred[name]
        j = idx[name]
        pool = [i for i in range(n32) if i != j]
        profiles[f"{name}|predicted_loco"] = Profile(
            label=f"{name}|predicted_loco",
            source="predicted_ridge_loco",
            raw=pred,
            centered=pred - pred_mean,
            reliability=None,
            options=chosen.options(),
            inner_centered_spearman=inner,
        )

    # ---- fit on all 32, predict the compounds outside it
    new = {
        "levulinic acid": next(i.smiles for i in six if i.abbreviation == "LVA"),
        "sodium acetate": "CC(=O)[O-].[Na+]",
        "acetic acid": next(i.smiles for i in six if i.abbreviation == "AA"),
        "formic acid": next(i.smiles for i in six if i.abbreviation == "FA"),
        "lactic acid": next(i.smiles for i in six if i.abbreviation == "LA"),
    }
    x_new = np.stack([fcfp4_count(s) for s in new.values()]).astype(np.float64)
    pred_new, pred_mean32, chosen32, inner32 = fit_predict(
        np.vstack([x32, x_new]), y, list(range(n32)), list(range(n32, n32 + len(new)))
    )
    print(f"all-32 fit: {chosen32.options()} inner {inner32:.3f}")
    for k, name in enumerate(new):
        profiles[f"{name}|predicted_all32"] = Profile(
            label=f"{name}|predicted_all32",
            source="predicted_ridge_all32",
            raw=pred_new[:, k],
            centered=pred_new[:, k] - pred_mean32,
            reliability=None,
            options=chosen32.options(),
            inner_centered_spearman=inner32,
        )

    # ---- comparisons beyond the LOCO block
    # acid_vs_salt is the identity test; the *_b001 rows are SUPPLEMENTARY: build 001
    # still serves levulinic acid and sodium acetate (CPM), which build 002 dropped as
    # unreported compounds, and they enter no matrix
    pairs = [
        (
            "acid_vs_salt",
            "acetic acid|predicted_all32",
            "sodium acetate|measured_hoepfner_hop",
        ),
        (
            "acid_vs_salt",
            "sodium acetate|predicted_all32",
            "sodium acetate|measured_hoepfner_hop",
        ),
        (
            "acid_vs_salt",
            "acetic acid|predicted_all32",
            "sodium acetate|predicted_all32",
        ),
        (
            "supplementary_b001",
            "acetic acid|predicted_all32",
            "sodium acetate|measured_b001",
        ),
        (
            "supplementary_b001",
            "sodium acetate|predicted_all32",
            "sodium acetate|measured_b001",
        ),
        (
            "supplementary_b001",
            "levulinic acid|predicted_all32",
            "levulinic acid|measured_b001",
        ),
        (
            "supplementary_b001",
            "sodium acetate|measured_b001",
            "sodium acetate|measured_hoepfner_hop",
        ),
        ("supplementary_b001", "furfural|measured_b001", "furfural|measured_b002"),
        (
            "supplementary_b001",
            "5-(hydroxymethyl)furfural|measured_b001",
            "5-(hydroxymethyl)furfural|measured_b002",
        ),
    ]
    extra_rows = []
    for comparison, a, b in pairs:
        pa, pb = profiles[a], profiles[b]
        for target in ("raw", "centered"):
            va, vb = getattr(pa, target), getattr(pb, target)
            n, rho, r = score(va, vb)
            extra_rows.append(
                {
                    "comparison": comparison,
                    "compound": a.split("|")[0],
                    "predicted": a,
                    "measured": b,
                    "target": target,
                    "n_genes": n,
                    "spearman": rho,
                    "pearson": r,
                    "measured_reliability": pb.reliability,
                    "options": pa.options,
                    "inner_centered_spearman": pa.inner_centered_spearman,
                }
            )
    ladder = pd.read_csv(LADDER_SCORES_038)
    ladder = ladder[(ladder["model"] == "krr") & (ladder["kernel"] == RIDGE_KERNEL)]
    ref = (
        ladder.groupby(["compound", "target"])["spearman"]
        .agg(["mean", "size"])
        .reset_index()
    )
    ref.columns = ["compound", "target", "spearman_038_folds_mean", "n_038_fold_seeds"]
    comparisons = pd.concat(
        [loco_df, pd.DataFrame(extra_rows)], ignore_index=True
    ).merge(ref, on=["compound", "target"], how="left")
    comparisons.insert(0, "sign", SIGN)
    comparisons.to_csv(
        osp.join(RESULTS_DIR, "profile_predicted_vs_measured.csv"), index=False
    )
    pd.set_option("display.width", 250)
    print(
        comparisons[comparisons["comparison"] != "loco_32"]
        .drop(columns=["sign"])
        .to_string(index=False)
    )
    print(
        comparisons[
            comparisons["compound"].isin(["furfural", "5-(hydroxymethyl)furfural"])
            & (comparisons["comparison"] == "loco_32")
        ]
        .drop(columns=["sign"])
        .to_string(index=False)
    )

    def check_score(predicted: str, measured: str) -> float:
        row = comparisons[
            (comparisons["predicted"] == predicted)
            & (comparisons["measured"] == measured)
            & (comparisons["target"] == "centered")
        ]
        assert len(row) == 1
        return float(row["spearman"].iloc[0])

    # ---- the two matrices
    # the measured candidate of each column: Vanacloig build 002 where it holds the
    # compound, Hoepfner HOP sodium acetate for acetic acid (cross-screen, by decision:
    # it has no SE-based reliability, only the dose agreement), none otherwise
    plan = {
        "FF": ("furfural|measured_b002", "furfural|predicted_loco"),
        "AA": ("sodium acetate|measured_hoepfner_hop", "acetic acid|predicted_all32"),
        "HMF": (
            "5-(hydroxymethyl)furfural|measured_b002",
            "5-(hydroxymethyl)furfural|predicted_loco",
        ),
        "FA": (None, "formic acid|predicted_all32"),
        "LVA": (None, "levulinic acid|predicted_all32"),
        "LA": (None, "lactic acid|predicted_all32"),
    }
    choices: list[ProfileChoice] = []
    for inh in six:
        measured, predicted = plan[inh.abbreviation]
        check_value = None if measured is None else check_score(predicted, measured)
        rel = None if measured is None else profiles[measured].reliability
        hoepfner = (
            measured is not None
            and profiles[measured].source == "measured_hoepfner_hop"
        )
        for version in ("best_available", "all_predicted"):
            use_measured = version == "best_available" and (
                hoepfner or (rel is not None and rel >= RELIABILITY_MIN)
            )
            chosen = measured if use_measured else predicted
            if version == "all_predicted":
                reason = "every column predicted"
            elif measured is None:
                reason = "no measured profile in build 002 or Hoepfner"
            elif hoepfner:
                reason = (
                    "Hoepfner HOP sodium acetate as acetic acid (cross-screen, salt for "
                    f"acid; 125 vs 150 mM Spearman {hop.dose_spearman:.2f})"
                )
            elif use_measured:
                reason = f"measured reliability {rel:.2f} >= {RELIABILITY_MIN}"
            else:
                reason = f"measured reliability {rel:.2f} < {RELIABILITY_MIN}"
            choices.append(
                ProfileChoice(
                    version=version,
                    inhibitor=inh.name,
                    abbreviation=inh.abbreviation,
                    profile=chosen,
                    source=profiles[chosen].source,
                    measured_reliability=rel,
                    prediction_check_spearman_centered=check_value,
                    reason=reason,
                )
            )
    gene_names = gene_name_map(genes)
    matrix = pd.DataFrame({"gene": genes, "gene_name": [gene_names[g] for g in genes]})
    for c in choices:
        for target in ("raw", "centered"):
            matrix[f"{c.version}|{target}|{c.inhibitor}"] = getattr(
                profiles[c.profile], target
            )
    matrix.to_csv(osp.join(RESULTS_DIR, "profile_matrix.csv"), index=False)
    all_profiles = pd.DataFrame(
        {"gene": genes, "gene_name": [gene_names[g] for g in genes]}
    )
    for label, p in profiles.items():
        all_profiles[f"{label}|raw"] = p.raw
        all_profiles[f"{label}|centered"] = p.centered
    all_profiles["vanacloig_b002_gene_mean_32|raw"] = mean002
    all_profiles.to_csv(osp.join(RESULTS_DIR, "profile_all.csv"), index=False)
    meta = {
        "sign": SIGN,
        "value_columns": "profile_matrix.csv: '<version>|<target>|<inhibitor>'; profile_all.csv: '<compound>|<source>|<target>'",
        "targets": {
            "raw": "the served response (measured) or the ridge prediction (predicted)",
            "centered": "minus the gene's mean over the 32 build-002 compounds (measured b002), over the same 32 in build 001 (measured b001), or minus the PREDICTED gene mean over the fitted compounds (predicted)",
        },
        "reliability_min": RELIABILITY_MIN,
        "top_n": TOP_N,
        "n_genes": len(genes),
        "inhibitors": [i.model_dump() for i in six],
        "choices": [c.model_dump() for c in choices],
        "profiles": {
            k: {
                "source": p.source,
                "reliability": p.reliability,
                "options": p.options,
                "inner_centered_spearman": p.inner_centered_spearman,
                "n_genes_raw": int(np.isfinite(p.raw).sum()),
            }
            for k, p in profiles.items()
        },
        "hoepfner_hop_acetate": hop.model_dump(exclude={"raw", "centered"}),
        "loco_median_centered_spearman": median_centered,
        "loco_n_compounds": n32,
        "cell_table_002": CELL_TABLE,
        "cell_table_001": CELL_TABLE_001,
        "fcfp4_npz": FCFP4_NPZ,
    }
    with open(osp.join(RESULTS_DIR, "profile_matrix.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ---- similarity
    by_name = {i.name: i for i in six}
    sim_rows = []
    for version in ("best_available", "all_predicted"):
        col = {c.inhibitor: c for c in choices if c.version == version}
        for a, b in itertools.combinations(sorted(by_name), 2):
            row: dict[str, object] = {
                "pair": f"{a}|{b}",
                "inhibitor_a": a,
                "inhibitor_b": b,
                "abbreviation_a": by_name[a].abbreviation,
                "abbreviation_b": by_name[b].abbreviation,
                "vanacloig_a": by_name[a].vanacloig_name,
                "vanacloig_b": by_name[b].vanacloig_name,
                "version": version,
                "profile_a": col[a].profile,
                "profile_b": col[b].profile,
            }
            for target in ("raw", "centered"):
                s = similarity(
                    getattr(profiles[col[a].profile], target),
                    getattr(profiles[col[b].profile], target),
                )
                row[f"n_genes_{target}"] = s["n_genes"]
                row[f"spearman_{target}"] = s["spearman"]
                row[f"jaccard_top{TOP_N}_{target}"] = s["jaccard_top"]
            sim_rows.append(row)
    sim = pd.DataFrame(sim_rows)
    sim.to_csv(osp.join(RESULTS_DIR, "profile_similarity.csv"), index=False)
    print(
        sim.drop(
            columns=[
                "vanacloig_a",
                "vanacloig_b",
                "profile_a",
                "profile_b",
                "inhibitor_a",
                "inhibitor_b",
            ]
        ).to_string(index=False)
    )

    # ---- figures
    scatter_figure(
        [
            (
                "furfural (LOCO)",
                "predicted, centered",
                "measured b002, centered",
                profiles["furfural|predicted_loco"].centered,
                profiles["furfural|measured_b002"].centered,
            ),
            (
                "5-HMF (LOCO)",
                "predicted, centered",
                "measured b002, centered",
                profiles["5-(hydroxymethyl)furfural|predicted_loco"].centered,
                profiles["5-(hydroxymethyl)furfural|measured_b002"].centered,
            ),
            (
                "acetic acid vs Hoepfner NaOAc",
                "predicted (acid), centered",
                "Hoepfner HOP NaOAc, centered",
                profiles["acetic acid|predicted_all32"].centered,
                profiles["sodium acetate|measured_hoepfner_hop"].centered,
            ),
            (
                "sodium acetate vs Hoepfner NaOAc",
                "predicted (salt), centered",
                "Hoepfner HOP NaOAc, centered",
                profiles["sodium acetate|predicted_all32"].centered,
                profiles["sodium acetate|measured_hoepfner_hop"].centered,
            ),
            (
                "levulinic acid (suppl., b001)",
                "predicted, centered",
                "measured b001, centered",
                profiles["levulinic acid|predicted_all32"].centered,
                profiles["levulinic acid|measured_b001"].centered,
            ),
        ]
    )
    order = sorted(
        six, key=lambda i: [x.value for x in bs.INHIBITORS].index(i.abbreviation)
    )
    for target in ("raw", "centered"):
        heatmap_figure(sim, order, target)


def gene_name_map(genes: list[str]) -> dict[str, str]:
    """Systematic name -> SGD standard name (the systematic name when it has none).

    The SGD GFF ``Name`` is the systematic name and the file carries no ``gene``
    attribute; SGD lists the standard name first in ``Alias``, so the first alias of the
    standard-name form (three letters and a number) is taken.
    """
    import re

    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    data_root = os.environ["DATA_ROOT"]
    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    out = {}
    for g in genes:
        gene = genome[g]
        alias = None if gene is None else gene.alias
        first = alias[0] if alias else ""
        out[g] = first if re.fullmatch(r"[A-Z]{3}\d+", first) else g
    return out


if __name__ == "__main__":
    main()
