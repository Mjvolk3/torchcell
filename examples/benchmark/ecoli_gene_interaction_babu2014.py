# examples/benchmark/ecoli_gene_interaction_babu2014.py
"""Genetic interaction score of *E. coli* double mutants (Babu 2014), gene-holdout split.

A complete benchmark run in one file: load the standardized task, describe each deleted
gene by the composition of its protein, fit a ridge regression, choose the ridge
penalty on the validation part, and score the test part once.

Needs only ``pip install "torchcell>=1.8.8"``. Run it in an empty directory::

    python ecoli_gene_interaction_babu2014.py

The first run downloads the dataset store and the genome into the current directory.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from torchcell.endpoint_http import DEFAULT_URL
from torchcell.tasks import load_task

TASK = "ecoli-gene-interaction-babu2014"
# Split names are <family>_s<seed>; task.split_names lists them all. gene_holdout splits
# the genes first: a pair is test when either of its genes is a test gene.
SPLIT = "gene_holdout_s0"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PENALTIES = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)
SELECT_ON = "mse"  # lower is better, and defined even when a prediction is constant


def protein_features(genome: Any, locus_tag: str) -> np.ndarray:
    """Fraction of each of the 20 amino acids plus log length: 21 numbers per gene."""
    protein = str(genome[locus_tag].protein.seq)
    fractions = [protein.count(letter) / len(protein) for letter in AMINO_ACIDS]
    return np.array([*fractions, np.log(len(protein))])


def fit_predict_ridge(
    x_fit: np.ndarray, y_fit: np.ndarray, x_new: np.ndarray, penalty: float
) -> np.ndarray:
    """Fit one closed-form ridge per target on ``x_fit`` and predict ``x_new``.

    Features are standardized with the fitting rows' mean and spread. Each target is
    fit only on the rows where it was measured (``y_fit`` is ``nan`` elsewhere).
    """
    mean = x_fit.mean(axis=0)
    spread = x_fit.std(axis=0)
    spread[spread == 0] = 1.0  # a constant feature stays zero instead of dividing by 0
    z_fit, z_new = (x_fit - mean) / spread, (x_new - mean) / spread
    predictions = np.empty((len(x_new), y_fit.shape[1]))
    for target in range(y_fit.shape[1]):
        measured = ~np.isnan(y_fit[:, target])
        z, y = z_fit[measured], y_fit[measured, target]
        gram = z.T @ z + penalty * np.eye(z.shape[1])
        weights = np.linalg.solve(gram, z.T @ (y - y.mean()))
        predictions[:, target] = z_new @ weights + y.mean()
    return predictions


def rounded(metrics: dict[str, float]) -> dict[str, float]:
    """Metrics rounded to four decimals for printing."""
    return {name: round(value, 4) for name, value in metrics.items()}


os.environ.setdefault("TC_DATA_URL", DEFAULT_URL)  # unset, the store is built from raw
os.environ.setdefault("DATA_ROOT", ".")  # the store and genome files go under here

task = load_task(TASK)  # samples, deleted genes, labels, genome and the pinned splits
split = task.split(SPLIT)
print(f"{task.spec.title}: {len(task):,} samples, {len(task.targets):,} target(s)")
print(f"split {SPLIT}: {split.sizes()}")

# task.genes holds locus tags, one column per deleted gene (two for a gene-pair task).
# A sample's features are the sum over its genes, which ignores the order of a pair.
# A sum is additive in the two genes, so this baseline cannot express an interaction.
by_gene = {tag: protein_features(task.genome, tag) for tag in np.unique(task.genes)}
features = np.array([sum(by_gene[tag] for tag in tags) for tags in task.genes])
labels = task.y  # (n_samples, n_targets); nan where a target was not measured

# Model selection: fit on train, score on validation. The test part stays locked.
x_train, y_train = features[split.train], labels[split.train]
validation = {
    penalty: task.score(
        labels[split.val],
        fit_predict_ridge(x_train, y_train, features[split.val], penalty),
    )
    for penalty in PENALTIES
}
best_penalty = min(PENALTIES, key=lambda penalty: validation[penalty][SELECT_ON])
print(f"penalty chosen on validation {SELECT_ON}: {best_penalty:g}")
print("validation", rounded(validation[best_penalty]))

# One final evaluation of the chosen model; unlock_test warns by design.
test = split.unlock_test(reason="final evaluation at the penalty chosen on validation")
test_prediction = fit_predict_ridge(x_train, y_train, features[test], best_penalty)
print("test      ", rounded(task.score(labels[test], test_prediction)))
print(f"headline metric: {task.spec.headline_metric} (mean over targets)")
