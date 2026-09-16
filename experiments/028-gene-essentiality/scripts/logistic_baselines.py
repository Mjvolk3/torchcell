# experiments/028-gene-essentiality/scripts/logistic_baselines.py
# [[experiments.028-gene-essentiality.scripts.logistic_baselines]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/logistic_baselines.py
"""The classical ladder for gene essentiality: every gene encoding, four model classes.

The paper's traditional-ML supplementary study trained random forests and support vector
machines on a per-strain feature built from one of seventeen gene encodings, from a
one-hot identity code and random vectors up to the protein and nucleotide language models,
and read two things off it: that fitness is solved by gene identity alone, and that
interactions saturate whatever the encoding. This is that study's classification form on
gene essentiality, where the held-out genes are UNSEEN, so an identity code cannot carry
anything across and the encodings are tested on content alone.

Encodings (a gene's vector; a single deletion is one gene, so the sum and mean over the
perturbed set, which were separate variants in the fitness study, coincide here):

* identity and noise: ``one_hot_gene``; ``random_{1,10,100,1000}`` from the embedding
  registry; ``random_6607`` (the one-hot width, the genome's gene count) and
  ``random_8192`` (one step larger), drawn here at seed 0;
* hand-crafted: ``codon_frequency``, ``normalized_chrom_pathways``;
* nucleotide language models: the six ``nt_window_*`` windows, ``fudt_upstream``,
  ``fudt_downstream``, ``species_lm_5p_3p`` (the two fudt windows concatenated), ``calm``;
* protein language models: ``prot_T5_all``, ``prot_T5_no_dubious``,
  ``esm2_t33_650M_UR50D_all``, ``esm2_t33_650M_UR50D_no_dubious``;
* ``mixed``: ``prot_T5_all`` + ``calm`` + ``species_lm_5p_3p`` + ``nt_window_5979`` +
  ``codon_frequency`` + ``normalized_chrom_pathways`` concatenated.

Models, each with one hyperparameter swept: L2 logistic regression (C), a random forest
(max depth, 300 trees), an RBF support vector classifier (C), cosine k-nearest neighbors
(k). Every encoding is also run with FBA's four growth ratios (one per medium) appended,
which is the one physics feature a solver gives for free, and FBA alone is the unfitted
rung.

THE HYPERPARAMETER IS CHOSEN INSIDE THE TRAINING GENES, by five-fold stratified
cross-validation, never on the split's validation genes: the validation set exists for the
transformer's epoch selection and is reported here as a second held-out read, so the
ladder's numbers and the transformer's are both scored on genes no fit touched. The
held-out AUROC on the 195 released test genes carries a 2,000-draw gene bootstrap.

Run from the worktree root (CPU, tens of minutes)::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/028-gene-essentiality/scripts/logistic_baselines.py \
        --split ess_fcl_val0.2_s0.json.gz

Writes ``results/logistic_baselines.{json,csv}`` and the figure
``$ASSET_IMAGES_DIR/028-gene-essentiality/classical_ladder_<timestamp>.{svg,png}``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import os.path as osp
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder  # noqa: E402
from torchcell.graph import SCerevisiaeGraph  # noqa: E402
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402
from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
SPLITS = osp.join(RESULTS, "splits")
FBA_CSV = osp.join(RESULTS, "fba_essentiality_per_gene.csv")
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "028-gene-essentiality")
MEDIA = ("shipped", "SM", "SC", "YPD")

#: Encoding name -> registry names concatenated; "random_<d>" beyond the registry and
#: "mixed" are built below.
ENCODINGS: dict[str, tuple[str, ...]] = {
    "one_hot_gene": ("one_hot_gene",),
    "random_1": ("random_1",),
    "random_10": ("random_10",),
    "random_100": ("random_100",),
    "random_1000": ("random_1000",),
    "random_6607": (),
    "random_8192": (),
    "codon_frequency": ("codon_frequency",),
    "normalized_chrom_pathways": ("normalized_chrom_pathways",),
    "nt_window_5979": ("nt_window_5979",),
    "nt_window_5979_max": ("nt_window_5979_max",),
    "nt_window_three_prime_5979": ("nt_window_three_prime_5979",),
    "nt_window_five_prime_5979": ("nt_window_five_prime_5979",),
    "nt_window_three_prime_300": ("nt_window_three_prime_300",),
    "nt_window_five_prime_1003": ("nt_window_five_prime_1003",),
    "fudt_upstream": ("fudt_upstream",),
    "fudt_downstream": ("fudt_downstream",),
    "species_lm_5p_3p": ("fudt_upstream", "fudt_downstream"),
    "calm": ("calm",),
    "prot_T5_all": ("prot_T5_all",),
    "prot_T5_no_dubious": ("prot_T5_no_dubious",),
    "esm2_t33_650M_UR50D_all": ("esm2_t33_650M_UR50D_all",),
    "esm2_t33_650M_UR50D_no_dubious": ("esm2_t33_650M_UR50D_no_dubious",),
    "mixed": (
        "prot_T5_all",
        "calm",
        "fudt_upstream",
        "fudt_downstream",
        "nt_window_5979",
        "codon_frequency",
        "normalized_chrom_pathways",
    ),
}
RANDOM_DIMS = {"random_6607": 6607, "random_8192": 8192}
C_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
SVC_C_GRID = [0.1, 1.0, 10.0]
K_GRID = [1, 3, 5, 10, 25, 50]
DEPTH_GRID: list[int | None] = [4, 8, None]
N_TREES = 300
N_FOLDS = 5
N_BOOTSTRAP = 2000
N_JOBS = 16
MODELS = ("logistic", "random forest", "SVC", "cosine kNN")


class LadderRow(BaseModel):
    """One (encoding, model, with or without FBA) cell of the ladder."""

    model_config = ConfigDict(extra="forbid")
    model: str
    encoding: str
    with_fba: bool
    dim: int
    hyperparameter: str
    cv_auroc: float = Field(description="Mean five-fold AUROC inside the train genes.")
    val_auroc: float = Field(
        description="On the split's validation genes, never fit on."
    )
    test_auroc: float
    test_ci_low: float
    test_ci_high: float
    n_train: int
    n_val: int
    n_test: int


class LadderReport(BaseModel):
    """The ladder on one split file."""

    model_config = ConfigDict(extra="forbid")
    split_file: str
    n_test_essential: int
    genes_without_vector: dict[str, int]
    rows: list[LadderRow]
    figure: str
    note: str


def bootstrap(label: np.ndarray, score: np.ndarray) -> tuple[float, float, float]:
    point = float(roc_auc_score(label, score))
    rng = np.random.default_rng(0)
    draws: list[float] = []
    n = label.shape[0]
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        if len(np.unique(label[idx])) < 2:
            continue
        draws.append(float(roc_auc_score(label[idx], score[idx])))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def registry_matrix(
    names: tuple[str, ...], genome: Any, graph: Any
) -> dict[str, np.ndarray]:
    """Gene -> concatenated registry embedding, 019's construction."""
    per_gene: dict[str, list[np.ndarray]] = {}
    counts: dict[str, int] = {}
    for name in names:
        built = NodeEmbeddingBuilder.build(
            embedding_names=[name], data_root=DATA_ROOT, genome=genome, graph=graph
        )
        for item in built[name]:
            vec = (
                torch.cat([t.flatten() for t in item.embeddings.values()]).cpu().numpy()
            )
            per_gene.setdefault(item.id, []).append(vec)
            counts[item.id] = counts.get(item.id, 0) + 1
    return {
        g: np.concatenate(v).astype(np.float32)
        for g, v in per_gene.items()
        if counts[g] == len(names)
    }


def random_matrix(genes: list[str], dim: int) -> dict[str, np.ndarray]:
    """A fixed Gaussian vector per gene at seed 0: content-free by construction."""
    rng = np.random.default_rng(0)
    return {g: rng.standard_normal(dim).astype(np.float32) for g in sorted(genes)}


def knn_scores(
    x_fit: np.ndarray, y_fit: np.ndarray, x_eval: np.ndarray, k: int
) -> np.ndarray:
    fit = x_fit / (np.linalg.norm(x_fit, axis=1, keepdims=True) + 1e-12)
    ev = x_eval / (np.linalg.norm(x_eval, axis=1, keepdims=True) + 1e-12)
    nearest = np.argsort(-(ev @ fit.T), axis=1)[:, :k]
    return np.asarray(y_fit[nearest].mean(axis=1), dtype=float)


class Fitter:
    """One model class: fit on rows, score rows; the hyperparameter is the argument."""

    def __init__(self, model: str) -> None:
        """Bind to one of ``MODELS``."""
        self.model = model

    @property
    def grid(self) -> list[Any]:
        """The one hyperparameter's values, in the order they are tried."""
        grids: dict[str, list[Any]] = {
            "logistic": C_GRID,
            "random forest": DEPTH_GRID,
            "SVC": SVC_C_GRID,
            "cosine kNN": K_GRID,
        }
        return grids[self.model]

    def label(self, hp: Any) -> str:
        """The chosen value as it appears in the table."""
        # Branches, not a dict of f-strings: a dict would format every entry, and the
        # forest's depth can be None, which `:g` refuses.
        if self.model == "random forest":
            return f"max_depth={hp}"
        if self.model == "cosine kNN":
            return f"k={hp}"
        return f"C={hp:g}"

    def fit_score(
        self, hp: Any, x_fit: np.ndarray, y_fit: np.ndarray, x_evals: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Fit on ``x_fit`` at ``hp`` and return one score vector per evaluation matrix."""
        if self.model == "cosine kNN":
            return [knn_scores(x_fit, y_fit, x, hp) for x in x_evals]
        scaler = StandardScaler().fit(x_fit)
        s_fit = scaler.transform(x_fit)
        s_evals = [scaler.transform(x) for x in x_evals]
        clf: Any
        if self.model == "logistic":
            clf = LogisticRegression(C=hp, max_iter=5000, class_weight="balanced")
            clf.fit(s_fit, y_fit)
            return [clf.decision_function(s) for s in s_evals]
        if self.model == "SVC":
            clf = SVC(C=hp, kernel="rbf", gamma="scale", class_weight="balanced")
            clf.fit(s_fit, y_fit)
            return [clf.decision_function(s) for s in s_evals]
        clf = RandomForestClassifier(
            n_estimators=N_TREES,
            max_depth=hp,
            class_weight="balanced",
            random_state=0,
            n_jobs=N_JOBS,
        )
        clf.fit(x_fit, y_fit)
        return [clf.predict_proba(x)[:, 1] for x in x_evals]


def cv_select(fitter: Fitter, x_tr: np.ndarray, y_tr: np.ndarray) -> tuple[Any, float]:
    """The grid value with the best mean AUROC over stratified folds of the train genes."""
    folds = list(
        StratifiedKFold(N_FOLDS, shuffle=True, random_state=0).split(x_tr, y_tr)
    )
    best: tuple[Any, float] | None = None
    for hp in fitter.grid:
        scores = []
        for fit_idx, ev_idx in folds:
            (s,) = fitter.fit_score(hp, x_tr[fit_idx], y_tr[fit_idx], [x_tr[ev_idx]])
            scores.append(roc_auc_score(y_tr[ev_idx], s))
        mean = float(np.mean(scores))
        if best is None or mean > best[1]:
            best = (hp, mean)
    assert best is not None
    return best


def figure(rows: list[LadderRow], fba_rows: list[LadderRow]) -> str:
    """Held-out AUROC per encoding, one marker per model; FBA-augmented cells hollow."""
    os.makedirs(IMAGE_DIR, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "svg.fonttype": "none",
            "axes.linewidth": 0.5,
        }
    )
    encodings = list(ENCODINGS)
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(70)))
    x = np.arange(len(encodings))
    width = 0.19
    for i, model in enumerate(MODELS):
        color = PLOT_PALETTE[i]
        for with_fba, filled in ((False, True), (True, False)):
            ys = []
            for enc in encodings:
                r = next(
                    (
                        r
                        for r in rows
                        if r.model == model
                        and r.encoding == enc
                        and r.with_fba == with_fba
                    ),
                    None,
                )
                ys.append(r.test_auroc if r is not None else np.nan)
            ax.scatter(
                x + (i - 1.5) * width,
                ys,
                s=9,
                marker="o",
                facecolors=color if filled else "white",
                edgecolors=color,
                linewidths=0.6,
                zorder=3,
                label=f"{model}{' + FBA' if with_fba else ''}",
            )
    fba_best = max(r.test_auroc for r in fba_rows)
    ax.axhline(fba_best, color=PLOT_PALETTE[5], lw=0.6, ls="--", zorder=1)
    ax.axhline(0.5, color="black", lw=0.4, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(encodings, rotation=60, ha="right")
    ax.set_ylabel("held-out AUROC (195 genes)")
    ax.set_ylim(0.3, 0.9)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.grid(axis="y", which="both", lw=0.3, color="0.85")
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(True)
    ax.legend(
        loc="lower left", bbox_to_anchor=(0, 1.02), ncol=4, frameon=False, fontsize=6
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.85, bottom=0.36)
    base = osp.join(IMAGE_DIR, f"classical_ladder_{timestamp()}")
    savefig_true_size_svg(fig, base + ".svg")
    fig.savefig(base + ".png", dpi=300)
    plt.close(fig)
    return base + ".svg"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="ess_fcl_val0.2_s0.json.gz")
    args = parser.parse_args()
    with gzip.open(osp.join(SPLITS, args.split), "rt") as fh:
        split = json.load(fh)
    labels: dict[str, int] = {g: int(v) for g, v in split["labels"].items()}
    genes = {k: list(v) for k, v in split["genes"].items()}
    assert len(genes["test"]) == 195, (
        f"expected 195 test genes, read {len(genes['test'])}."
    )
    all_labeled = genes["train"] + genes["val"] + genes["test"]
    y_by = {
        k: np.array([labels[g] for g in genes[k]]) for k in ("train", "val", "test")
    }

    fba = pd.read_csv(FBA_CSV).set_index("gene")
    fba_table = {
        g: np.array([-float(fba.loc[g, m]) for m in MEDIA], dtype=np.float32)
        for g in fba.index
    }
    rows: list[LadderRow] = []
    fba_rows: list[LadderRow] = []
    for i, medium in enumerate(MEDIA):
        xv = np.array([fba_table[g][i] for g in genes["val"]])
        xt = np.array([fba_table[g][i] for g in genes["test"]])
        point, lo, hi = bootstrap(y_by["test"], xt)
        fba_rows.append(
            LadderRow(
                model="FBA growth",
                encoding=f"medium={medium}",
                with_fba=True,
                dim=1,
                hyperparameter="none",
                cv_auroc=float("nan"),
                val_auroc=float(roc_auc_score(y_by["val"], xv)),
                test_auroc=point,
                test_ci_low=lo,
                test_ci_high=hi,
                n_train=0,
                n_val=len(xv),
                n_test=len(xt),
            )
        )

    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )
    missing: dict[str, int] = {}
    for enc_name, parts in ENCODINGS.items():
        table = (
            random_matrix(all_labeled, RANDOM_DIMS[enc_name])
            if enc_name in RANDOM_DIMS
            else registry_matrix(parts, genome, graph)
        )
        missing[enc_name] = sum(1 for g in all_labeled if g not in table)
        kept = {
            k: [g for g in genes[k] if g in table] for k in ("train", "val", "test")
        }
        xs = {k: np.stack([table[g] for g in kept[k]]) for k in kept}
        ys = {k: np.array([labels[g] for g in kept[k]]) for k in kept}
        xs_fba = {
            k: np.concatenate(
                [xs[k], np.stack([fba_table[g] for g in kept[k]])], axis=1
            )
            for k in kept
        }
        for model in MODELS:
            fitter = Fitter(model)
            for with_fba, x in ((False, xs), (True, xs_fba)):
                hp, cv = cv_select(fitter, x["train"], ys["train"])
                s_val, s_test = fitter.fit_score(
                    hp, x["train"], ys["train"], [x["val"], x["test"]]
                )
                point, lo, hi = bootstrap(ys["test"], s_test)
                rows.append(
                    LadderRow(
                        model=model,
                        encoding=enc_name,
                        with_fba=with_fba,
                        dim=int(x["train"].shape[1]),
                        hyperparameter=fitter.label(hp),
                        cv_auroc=cv,
                        val_auroc=float(roc_auc_score(ys["val"], s_val)),
                        test_auroc=point,
                        test_ci_low=lo,
                        test_ci_high=hi,
                        n_train=len(ys["train"]),
                        n_val=len(ys["val"]),
                        n_test=len(ys["test"]),
                    )
                )
                r = rows[-1]
                print(
                    f"  {enc_name:32s} {model:14s} {'+FBA' if with_fba else '    '} "
                    f"{r.hyperparameter:16s} cv {cv:.3f} val {r.val_auroc:.3f} "
                    f"test {point:.4f} [{lo:.3f}, {hi:.3f}]",
                    flush=True,
                )

    fig_path = figure(rows, fba_rows)
    report = LadderReport(
        split_file=args.split,
        n_test_essential=int(y_by["test"].sum()),
        genes_without_vector=missing,
        rows=fba_rows + rows,
        figure=fig_path,
        note=(
            "Hyperparameters chosen by five-fold stratified cross-validation inside the "
            "split's train genes; validation and the 195 held-out genes are scored once, "
            "on genes no fit touched. A single deletion is one gene, so the sum and mean "
            "aggregations of the fitness study coincide. random_6607 and random_8192 are "
            "Gaussian vectors at seed 0; the registry's random_* are torch.rand."
        ),
    )
    out = osp.join(RESULTS, "logistic_baselines.json")
    with open(out, "w") as fh:
        json.dump(report.model_dump(), fh, indent=2)
    pd.DataFrame([r.model_dump() for r in report.rows]).to_csv(
        osp.join(RESULTS, "logistic_baselines.csv"), index=False
    )
    print(f"genes without vector: {missing}\n{fig_path}\n{out}")


if __name__ == "__main__":
    main()
