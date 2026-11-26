# experiments/050-ecoli-benchmark/scripts/linear_probes.py
# [[experiments.050-ecoli-benchmark.scripts.linear_probes]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/050-ecoli-benchmark/scripts/linear_probes
"""Linear probes on the E. coli benchmark tasks: ten gene representations, every split.

Representations of a gene (rows of a matrix over the task's genes):

- ``random``: 1,280 standard-normal numbers per gene (seed 0), no biology.
- ``one_hot``: the identity of the gene.
- ``codon_frequency``: relative frequency of the 64 codons of the coding sequence.
- ``calm``: mean-pooled CaLM codon embedding of the coding sequence (768).
- ``genslm_25m``, ``genslm_250m``: mean-pooled GenSLM codon embeddings (512, 1,840).
- ``nt_v1_2500m``, ``nt_v2_500m``: mean-pooled Nucleotide Transformer embeddings of the
  coding sequence, v1 2.5B multi-species (2,560) and v2 500M multi-species (1,024).
- ``esm2``: mean-pooled ESM-2 650M protein embedding (1,280).
- ``prot_t5``: mean-pooled ProtT5-XL-UniRef50 protein embedding (1,024).

A single-gene sample is its gene's row. A gene-pair sample is the two rows concatenated
in stored order (``concat``) or added (``sum``). The probe is ordinary least squares
on features standardized with the training mean and standard deviation, with an
intercept, one fit per task, split and representation, all targets at once. It has no
hyperparameter and no random step. When a representation has at least as many
features as there are training samples, or collinear features, least squares has many
solutions and the minimum-norm one is taken (the pseudo-inverse solution), which is
unique.
The test part is unlocked once per split and scored once.

Writes ``experiments/050-ecoli-benchmark/results/probes/<task>.<feature>.csv``, and for
the seed-0 split of each family the validation and test predictions under
``$DATA_ROOT/data/torchcell/experiments/050-ecoli-benchmark/predictions/`` (the
benchmark board's baseline rows are uploaded from those).

    python experiments/050-ecoli-benchmark/scripts/linear_probes.py --task <task> [--features ...]
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import re
import time
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from dotenv import load_dotenv

from torchcell.tasks import (
    BenchmarkTask,
    TestSplitWarning,
    load_task,
    regression_metrics,
)

RESULTS = "experiments/050-ecoli-benchmark/results"
FEATURES = (
    "random",
    "one_hot",
    "codon_frequency",
    "calm",
    "genslm_25m",
    "genslm_250m",
    "nt_v1_2500m",
    "nt_v2_500m",
    "esm2",
    "prot_t5",
)
PREDICTIONS = "data/torchcell/experiments/050-ecoli-benchmark/predictions"
CONSTANT_TOLERANCE = 1e-6  # of the training label spread; below it, a constant
MAX_TARGETS_KEPT = 100  # predictions are kept for the leaderboard, not for 7,534 ions
RANDOM_DIM = 1280
Array = npt.NDArray[np.float64]


def gene_matrix(task: BenchmarkTask, feature: str) -> tuple[list[str], Array]:
    """The genes of a task (sorted) and one representation row per gene."""
    genes = sorted({str(g) for g in task.genes.ravel()})
    if feature == "random":
        return genes, np.random.default_rng(0).standard_normal((len(genes), RANDOM_DIM))
    if feature == "one_hot":
        return genes, np.eye(len(genes))
    path = osp.join(
        os.environ["DATA_ROOT"],
        "data/torchcell/experiments/050-ecoli-benchmark/embeddings",
        feature,
        f"{task.spec.assembly_set}.npz",
    )
    stored = np.load(path)
    row = {str(g): i for i, g in enumerate(stored["genes"])}
    return genes, stored["x"][[row[g] for g in genes]].astype(np.float64)


def sample_features(task: BenchmarkTask, feature: str, combine: str) -> Array:
    """One feature row per sample: the gene's row, or two rows concatenated or added."""
    genes, matrix = gene_matrix(task, feature)
    row = {g: i for i, g in enumerate(genes)}
    index = np.array([[row[str(g)] for g in pair] for pair in task.genes])
    if task.genes.shape[1] == 1:
        return matrix[index[:, 0]]
    if combine == "concat":
        return np.concatenate([matrix[index[:, 0]], matrix[index[:, 1]]], axis=1)
    return matrix[index[:, 0]] + matrix[index[:, 1]]


def least_squares(
    x_train: Array, y_train: Array, x_eval: list[Array]
) -> tuple[list[Array], int]:
    """Ordinary least squares predictions, minimum norm when the fit is not unique.

    Features are standardized and targets centered with training statistics, so the
    model has an intercept. ``y_train`` must hold no ``nan``. The solution is the
    pseudo-inverse solution, computed from the eigendecomposition of the smaller of
    the two Gram matrices; a direction whose eigenvalue is below ``max(n, p)`` times
    machine epsilon of the largest is treated as absent, which is where the rank comes
    from. With full-rank features this is the unique least squares fit.

    A prediction that is constant in exact arithmetic (a one-hot gene that was not in
    training gets the intercept) comes back with rounding noise. A target whose
    predictions vary by less than ``CONSTANT_TOLERANCE`` of the training labels'
    standard deviation is set to its mean, so that noise is not scored as a correlation.

    Returns:
        The predictions for each array of ``x_eval``, and the rank of the standardized
        training features (below the number of features when the solution is not
        unique).
    """
    mean = x_train.mean(0)
    scale = x_train.std(0)
    scale[scale == 0] = 1.0
    y_mean = y_train.mean(0)
    z = (x_train - mean) / scale
    y = y_train - y_mean
    cutoff = max(z.shape) * np.finfo(np.float64).eps
    if z.shape[0] >= z.shape[1]:
        w, v = np.linalg.eigh(z.T @ z)
        keep = w > w.max() * cutoff
        weights = v[:, keep] @ ((v[:, keep].T @ (z.T @ y)) / w[keep][:, None])
    else:
        w, u = np.linalg.eigh(z @ z.T)
        keep = w > w.max() * cutoff
        weights = z.T @ (u[:, keep] @ ((u[:, keep].T @ y) / w[keep][:, None]))
    rank = int(keep.sum())
    spread = y_train.std(0)
    out = []
    for e in x_eval:
        prediction = ((e - mean) / scale) @ weights + y_mean
        flat = prediction.std(0) < CONSTANT_TOLERANCE * spread
        prediction[:, flat] = prediction[:, flat].mean(0)
        out.append(prediction)
    return out, rank


def fit_predict(
    x: Array, y: Array, train: npt.NDArray[np.int64], parts: list[npt.NDArray[np.int64]]
) -> tuple[list[Array], int]:
    """``least_squares`` over targets grouped by which training samples measured them.

    Returns the predictions for each part and the smallest rank over the groups.
    """
    measured = ~np.isnan(y[train])
    patterns, inverse = np.unique(measured, axis=1, return_inverse=True)
    out = [np.full((len(p), y.shape[1]), np.nan) for p in parts]
    ranks = []
    for k in range(patterns.shape[1]):
        columns = np.flatnonzero(inverse.ravel() == k)
        rows = train[patterns[:, k]]
        predictions, rank = least_squares(
            x[rows], y[np.ix_(rows, columns)], [x[p] for p in parts]
        )
        ranks.append(rank)
        for e in range(len(parts)):
            out[e][:, columns] = predictions[e]
    return out, min(ranks)


def mean_metrics(y_true: Array, y_pred: Array, prefix: str) -> dict[str, float]:
    """Mean over targets of every metric, ``nan`` when no target defines it."""
    out = {}
    for name, values in regression_metrics(y_true, y_pred).items():
        finite = values[~np.isnan(values)]
        out[f"{prefix}_{name}"] = float(finite.mean()) if len(finite) else float("nan")
    return out


def main() -> None:
    """Fit every probe of one task and write its result table."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--features", nargs="+", default=list(FEATURES), choices=FEATURES
    )
    args = parser.parse_args()
    load_dotenv(".env")
    warnings.simplefilter("once", TestSplitWarning)
    task = load_task(args.task)
    combines = ("single",) if task.genes.shape[1] == 1 else ("concat", "sum")
    os.makedirs(osp.join(RESULTS, "probes"), exist_ok=True)
    for feature in args.features:
        rows = []
        for combine in combines:
            x = sample_features(task, feature, combine)
            for split_name in task.split_names:
                started = time.perf_counter()
                split = task.split(split_name)
                train, val = split.train, split.val
                test = split.unlock_test(
                    reason="final evaluation of the least-squares fit"
                )
                (val_prediction, test_prediction), rank = fit_predict(
                    x, task.y, train, [val, test]
                )
                family, seed = re.fullmatch(r"(.+)_s(\d+)", split_name).groups()  # type: ignore[union-attr]
                row: dict[str, object] = {
                    "task": args.task,
                    "split": split_name,
                    "family": family,
                    "seed": int(seed),
                    "feature": feature,
                    "combine": combine,
                    "n_features": x.shape[1],
                    "rank": rank,
                    "n_train": len(train),
                    "n_val": len(val),
                    "n_test": len(test),
                    **mean_metrics(task.y[val], val_prediction, "val"),
                    **mean_metrics(task.y[test], test_prediction, "test"),
                }
                if seed == "0" and task.y.shape[1] <= MAX_TARGETS_KEPT:
                    out_dir = osp.join(
                        os.environ["DATA_ROOT"], PREDICTIONS, args.task, split_name
                    )
                    os.makedirs(out_dir, exist_ok=True)
                    np.savez_compressed(
                        osp.join(out_dir, f"{feature}.{combine}.npz"),
                        val_index=val,
                        test_index=test,
                        val_prediction=val_prediction,
                        test_prediction=test_prediction,
                        rank=rank,
                    )
                if task.genes.shape[1] == 2:
                    train_genes = set(task.genes[train].ravel())
                    novel = np.array(
                        [
                            sum(g not in train_genes for g in pair)
                            for pair in task.genes[test]
                        ]
                    )
                    for count, label in (
                        (0, "no_new"),
                        (1, "one_new"),
                        (2, "both_new"),
                    ):
                        keep = novel == count
                        row[f"n_test_{label}"] = int(keep.sum())
                        row[f"test_pearson_{label}"] = (
                            mean_metrics(
                                task.y[test][keep], test_prediction[keep], "t"
                            )["t_pearson"]
                            if keep.sum() >= 3
                            else float("nan")
                        )
                rows.append(row)
                print(
                    f"{args.task} {feature} {combine} {split_name} rank {rank}/{x.shape[1]} "
                    f"val {row['val_pearson']:.3f} test {row['test_pearson']:.3f} "
                    f"({time.perf_counter() - started:.0f} s)",
                    flush=True,
                )
        pd.DataFrame(rows).to_csv(
            osp.join(RESULTS, "probes", f"{args.task}.{feature}.csv"), index=False
        )


if __name__ == "__main__":
    main()
