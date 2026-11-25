# examples/benchmark/ecoli_metabolome_fuhrer2017.py
"""Metabolome of *E. coli* single-gene deletions (Fuhrer 2017), 7,534 ion targets.

A complete benchmark run in one file: load the standardized task, turn each deleted
gene into features from its protein sequence, fit a ridge regression, choose its penalty
on validation, and score the test part once.

Needs only ``pip install "torchcell>=1.8.8"``. Run it in an empty directory::

    python ecoli_metabolome_fuhrer2017.py

The dataset store and the genome are downloaded on the first run.
"""

from __future__ import annotations

import os

import numpy as np

from torchcell.endpoint_http import DEFAULT_URL

os.environ.setdefault("TC_DATA_URL", DEFAULT_URL)  # download the built dataset store
os.environ.setdefault("DATA_ROOT", ".")  # store and genome files go under here

from torchcell.tasks import load_task  # noqa: E402

TASK = "ecoli-metabolome-fuhrer2017"
SPLIT = "random_s0"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PENALTIES = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)


def gene_features(task, locus_tag: str) -> np.ndarray:
    """Amino-acid composition and log length of one gene's protein (21 numbers)."""
    protein = str(task.genome[locus_tag].protein.seq)
    composition = [protein.count(letter) / len(protein) for letter in AMINO_ACIDS]
    return np.array([*composition, np.log(len(protein))])


def ridge(x_train, y_train, x_eval, penalty: float) -> np.ndarray:
    """Ridge regression in closed form; a target is fit on the samples that measured it."""
    mean, scale = x_train.mean(0), x_train.std(0)
    x = (x_train - mean) / scale
    out = np.empty((len(x_eval), y_train.shape[1]))
    for j in range(y_train.shape[1]):
        keep = ~np.isnan(y_train[:, j])
        offset = y_train[keep, j].mean()
        gram = x[keep].T @ x[keep] + penalty * np.eye(x.shape[1])
        weights = np.linalg.solve(gram, x[keep].T @ (y_train[keep, j] - offset))
        out[:, j] = ((x_eval - mean) / scale) @ weights + offset
    return out


task = load_task(TASK)  # samples, genes, labels, genome and the pinned splits
split = task.split(SPLIT)
print(f"{TASK}: {len(task):,} samples, {task.y.shape[1]:,} targets, split {SPLIT}")
print("sizes", split.sizes())

# One feature row per sample. task.genes holds the locus tags of the deleted genes:
# one column for a single-gene task, two for a gene-pair task (the rows are added).
table = {g: gene_features(task, g) for g in sorted(set(task.genes.ravel()))}
x = np.array([sum(table[g] for g in genes) for genes in task.genes])

# Model selection uses the validation part only.
scores = {
    penalty: task.score(
        task.y[split.val],
        ridge(x[split.train], task.y[split.train], x[split.val], penalty),
    )
    for penalty in PENALTIES
}
best = min(scores, key=lambda penalty: scores[penalty]["mse"])
print(f"penalty chosen on validation: {best:g}")
print("validation", {k: round(v, 4) for k, v in scores[best].items()})

# One final evaluation. unlock_test warns on purpose: nothing is fit or chosen on it.
test = split.unlock_test(reason="final evaluation at the penalty chosen on validation")
prediction = ridge(x[split.train], task.y[split.train], x[test], best)
print(
    "test      ",
    {k: round(v, 4) for k, v in task.score(task.y[test], prediction).items()},
)
