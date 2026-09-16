# experiments/028-gene-essentiality/scripts/logistic_baselines.py
# [[experiments.028-gene-essentiality.scripts.logistic_baselines]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/logistic_baselines.py
"""The classical ladder for gene essentiality, on the split the transformer trains on.

The fitness and expression lines each carry a ladder of models that need no graph and no
training beyond a solve (010's additive baselines; 019's B0 to B3 on external gene
embeddings, copied into 027's `linear_baselines_bx.py`). Essentiality is binary, so the
ladder is the classification form of 019's, with FBA's growth ratio added as the one
physics feature a solver gives for free:

  B0  prior            every gene scores the same; AUROC 0.5 by construction
  B1  FBA growth       the negative knockout-to-wild-type growth ratio on each medium,
                       no fitting (`fba_essentiality_baseline.py`)
  B2  logistic         L2 logistic regression from one external gene embedding to the
                       label, C swept on validation
  B3  cosine kNN       the essential fraction among the k nearest training genes in
                       embedding space, k swept on validation
  B4  random forest    FCL's own model class, on the same embedding, depth swept on
                       validation
  B5  logistic + FBA   B2's embedding with the four FBA growth ratios appended

over 019's four embedding sets (``prot_T5_all``, ``calm``, ``species_lm_5p_3p`` =
``fudt_upstream`` + ``fudt_downstream``, ``normalized_chrom_pathways``). Every model is fit
on the split file's TRAIN genes, its hyperparameter is chosen on the split's VALIDATION
genes, and it is scored once on the 195 held-out genes, which is exactly what the
transformer cells do, so a row of this table and a row of the round tables are the same
protocol with a different model.

Why this ladder is not the transformer's ceiling: 019 measured that on an unseen-gene
regime an external embedding plus kNN recovers most of what a trained model does, and
027's betaxanthin line found the same (B3 on ``prot_T5_all`` beat every flux arm). If B2
or B3 on ``prot_T5_all`` reaches the transformer's 0.76 here, the transformer is learning a
function of its input features that a nearest-neighbor rule already expresses.

Run from the worktree root (CPU, minutes)::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/028-gene-essentiality/scripts/logistic_baselines.py \
        --split ess_fcl_val0.2_s0.json.gz

Writes ``results/logistic_baselines.json`` and ``.csv``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import os.path as osp
from typing import Any

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder
from torchcell.graph import SCerevisiaeGraph
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
SPLITS = osp.join(RESULTS, "splits")
FBA_CSV = osp.join(RESULTS, "fba_essentiality_per_gene.csv")
MEDIA = ("shipped", "SM", "SC", "YPD")

#: 019's four perturbation representations, unchanged.
EMBEDDINGS: dict[str, tuple[str, ...]] = {
    "prot_T5_all": ("prot_T5_all",),
    "calm": ("calm",),
    "species_lm_5p_3p": ("fudt_upstream", "fudt_downstream"),
    "normalized_chrom_pathways": ("normalized_chrom_pathways",),
}
C_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
K_GRID = [1, 3, 5, 10, 25, 50]
DEPTH_GRID: list[int | None] = [4, 8, None]
N_TREES = 300
N_BOOTSTRAP = 2000


class BaselineRow(BaseModel):
    """One model on the held-out genes, with its validation-chosen hyperparameter."""

    model_config = ConfigDict(extra="forbid")
    rung: str
    embedding: str | None
    hyperparameter: str | None
    val_auroc: float
    test_auroc: float
    test_ci_low: float
    test_ci_high: float
    n_train: int
    n_val: int
    n_test: int


class BaselineReport(BaseModel):
    """The ladder on one split file."""

    model_config = ConfigDict(extra="forbid")
    split_file: str
    n_test_essential: int
    genes_without_embedding: dict[str, int] = Field(
        description="Per embedding, labeled genes dropped for lack of a vector."
    )
    rows: list[BaselineRow]
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


def embedding_matrix(
    names: tuple[str, ...], genome: Any, graph: Any
) -> dict[str, np.ndarray]:
    """Gene -> concatenated embedding vector, 019's construction."""
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


def knn_scores(
    x_fit: np.ndarray, y_fit: np.ndarray, x_eval: np.ndarray, k: int
) -> np.ndarray:
    """Essential fraction among the k cosine-nearest fit genes of each evaluated gene."""
    fit = x_fit / (np.linalg.norm(x_fit, axis=1, keepdims=True) + 1e-12)
    ev = x_eval / (np.linalg.norm(x_eval, axis=1, keepdims=True) + 1e-12)
    sim = ev @ fit.T
    nearest = np.argsort(-sim, axis=1)[:, :k]
    return y_fit[nearest].mean(axis=1)


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

    def xy(
        gene_list: list[str], table: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        kept = [g for g in gene_list if g in table]
        return np.stack([table[g] for g in kept]), np.array([labels[g] for g in kept])

    rows: list[BaselineRow] = []
    y_test_full = np.array([labels[g] for g in genes["test"]])
    n_train, n_val, n_test = (len(genes[k]) for k in ("train", "val", "test"))

    # B0: a constant. Reported so the ladder's floor is a number, not an assumption.
    rows.append(
        BaselineRow(
            rung="B0 prior",
            embedding=None,
            hyperparameter=None,
            val_auroc=0.5,
            test_auroc=0.5,
            test_ci_low=0.5,
            test_ci_high=0.5,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
        )
    )

    # B1: FBA growth on each medium, no fitting. The per-gene file is the baseline's own.
    fba = pd.read_csv(FBA_CSV).set_index("gene")
    fba_table = {
        g: np.array([-float(fba.loc[g, m]) for m in MEDIA], dtype=np.float32)
        for g in fba.index
    }
    for i, medium in enumerate(MEDIA):
        xv, yv = xy(genes["val"], fba_table)
        xt, yt = xy(genes["test"], fba_table)
        point, lo, hi = bootstrap(yt, xt[:, i])
        rows.append(
            BaselineRow(
                rung="B1 FBA growth",
                embedding=None,
                hyperparameter=f"medium={medium}",
                val_auroc=float(roc_auc_score(yv, xv[:, i])),
                test_auroc=point,
                test_ci_low=lo,
                test_ci_high=hi,
                n_train=0,
                n_val=len(yv),
                n_test=len(yt),
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
    for emb_name, parts in EMBEDDINGS.items():
        table = embedding_matrix(parts, genome, graph)
        all_labeled = genes["train"] + genes["val"] + genes["test"]
        missing[emb_name] = sum(1 for g in all_labeled if g not in table)
        x_tr, y_tr = xy(genes["train"], table)
        x_va, y_va = xy(genes["val"], table)
        x_te, y_te = xy(genes["test"], table)
        scaler = StandardScaler().fit(x_tr)
        s_tr, s_va, s_te = (scaler.transform(a) for a in (x_tr, x_va, x_te))

        def choose(rung: str, fits: list[tuple[str, np.ndarray, np.ndarray]]) -> None:
            """Keep the hyperparameter with the best validation AUROC; score it on test."""
            best = max(fits, key=lambda t: roc_auc_score(y_va, t[1]))
            point, lo, hi = bootstrap(y_te, best[2])
            rows.append(
                BaselineRow(
                    rung=rung,
                    embedding=emb_name,
                    hyperparameter=best[0],
                    val_auroc=float(roc_auc_score(y_va, best[1])),
                    test_auroc=point,
                    test_ci_low=lo,
                    test_ci_high=hi,
                    n_train=len(y_tr),
                    n_val=len(y_va),
                    n_test=len(y_te),
                )
            )

        # B2: L2 logistic regression, C swept.
        fits = []
        for c in C_GRID:
            clf = LogisticRegression(C=c, max_iter=5000, class_weight="balanced")
            clf.fit(s_tr, y_tr)
            fits.append(
                (f"C={c:g}", clf.decision_function(s_va), clf.decision_function(s_te))
            )
        choose("B2 logistic", fits)

        # B3: cosine kNN on the raw embedding, k swept.
        fits = [
            (f"k={k}", knn_scores(x_tr, y_tr, x_va, k), knn_scores(x_tr, y_tr, x_te, k))
            for k in K_GRID
        ]
        choose("B3 cosine kNN", fits)

        # B4: random forest, FCL's model class, depth swept.
        fits = []
        for depth in DEPTH_GRID:
            rf = RandomForestClassifier(
                n_estimators=N_TREES,
                max_depth=depth,
                class_weight="balanced",
                random_state=0,
                n_jobs=8,
            )
            rf.fit(x_tr, y_tr)
            fits.append(
                (
                    f"max_depth={depth}",
                    rf.predict_proba(x_va)[:, 1],
                    rf.predict_proba(x_te)[:, 1],
                )
            )
        choose("B4 random forest", fits)

        # B5: logistic on the embedding with the four FBA growth ratios appended.
        def with_fba(gene_list: list[str], s: np.ndarray) -> np.ndarray:
            kept = [g for g in gene_list if g in table]
            f = np.stack([fba_table[g] for g in kept])
            return np.concatenate([s, f], axis=1)

        a_tr, a_va, a_te = (
            with_fba(genes[k], s)
            for k, s in (("train", s_tr), ("val", s_va), ("test", s_te))
        )
        fscale = StandardScaler().fit(a_tr[:, -len(MEDIA) :])
        for a in (a_tr, a_va, a_te):
            a[:, -len(MEDIA) :] = fscale.transform(a[:, -len(MEDIA) :])
        fits = []
        for c in C_GRID:
            clf = LogisticRegression(C=c, max_iter=5000, class_weight="balanced")
            clf.fit(a_tr, y_tr)
            fits.append(
                (f"C={c:g}", clf.decision_function(a_va), clf.decision_function(a_te))
            )
        choose("B5 logistic + FBA", fits)

    report = BaselineReport(
        split_file=args.split,
        n_test_essential=int(y_test_full.sum()),
        genes_without_embedding=missing,
        rows=rows,
        note=(
            "Every fitted rung is fit on the split's train genes, its hyperparameter is "
            "chosen on the split's validation genes, and it is scored once on the 195 "
            "held-out genes with a 2,000-draw gene bootstrap, the protocol of the "
            "transformer cells on the same split file. B1 is not fitted."
        ),
    )
    out = osp.join(RESULTS, "logistic_baselines.json")
    with open(out, "w") as fh:
        json.dump(report.model_dump(), fh, indent=2)
    table_df = pd.DataFrame([r.model_dump() for r in rows])
    table_df.to_csv(osp.join(RESULTS, "logistic_baselines.csv"), index=False)
    for r in rows:
        print(
            f"  {r.rung:18s} {str(r.embedding):28s} {str(r.hyperparameter):16s} "
            f"val {r.val_auroc:.3f}  test {r.test_auroc:.4f} [{r.test_ci_low:.3f}, "
            f"{r.test_ci_high:.3f}]"
        )
    print(f"genes without embedding: {missing}\n{out}")


if __name__ == "__main__":
    main()
