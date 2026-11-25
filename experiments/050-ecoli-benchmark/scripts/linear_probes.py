# experiments/050-ecoli-benchmark/scripts/linear_probes.py
# [[experiments.050-ecoli-benchmark.scripts.linear_probes]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/050-ecoli-benchmark/scripts/linear_probes
"""Linear probes on the E. coli benchmark tasks: ten gene representations, every split.

Representations of a gene (rows of a matrix over the task's genes):

- ``random``: 1,280 standard-normal numbers per gene (seed 0), no biology.
- ``one_hot``: the identity of the gene.
- ``codon_frequency``: relative frequency of the 64 codons of the coding sequence.
- ``calm``: mean-pooled CaLM codon embedding of the coding sequence (768).
- ``genslm_25m``, ``genslm_250m``: mean-pooled GenSLM codon embeddings (512, 1,024).
- ``nt_v1_2500m``, ``nt_v2_500m``: mean-pooled Nucleotide Transformer embeddings of the
  coding sequence, v1 2.5B multi-species (2,560) and v2 500M multi-species (1,024).
- ``esm2``: mean-pooled ESM-2 650M protein embedding (1,280).
- ``prot_t5``: mean-pooled ProtT5-XL-UniRef50 protein embedding (1,024).

A single-gene sample is its gene's row. A gene-pair sample is the two rows concatenated
in stored order (``concat``) or added (``sum``). The probe is ridge regression on
features standardized with the training mean and standard deviation, one model per
task, split and representation, all targets at once. The penalty is the one of
``ALPHAS`` with the lowest validation mean squared error over all measured entries. The test part is unlocked
once per split, after that choice, and scored at the chosen penalty only.

Writes ``experiments/050-ecoli-benchmark/results/probes/<task>.<feature>.csv``, and for
the seed-0 split of each family the validation and test predictions at the chosen
penalty under ``$DATA_ROOT/data/torchcell/experiments/050-ecoli-benchmark/predictions/``
(the benchmark board's baseline rows are uploaded from those).

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
MAX_TARGETS_KEPT = 100  # predictions are kept for the leaderboard, not for 7,534 ions
ALPHAS = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)
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


def ridge_path(
    x_train: Array, y_train: Array, x_eval: list[Array], alphas: tuple[float, ...]
) -> list[list[Array]]:
    """Ridge predictions for every penalty, from one eigendecomposition.

    Features are standardized and targets centered with training statistics. The
    primal Gram matrix is used when there are at least as many samples as features and
    the dual one otherwise. ``y_train`` must hold no ``nan``.

    Returns:
        ``predictions[a][e]``: predictions for ``x_eval[e]`` at ``alphas[a]``.
    """
    mean = x_train.mean(0)
    scale = x_train.std(0)
    scale[scale == 0] = 1.0
    x = (x_train - mean) / scale
    evals = [(e - mean) / scale for e in x_eval]
    y_mean = y_train.mean(0)
    y = y_train - y_mean
    out: list[list[Array]] = []
    if x.shape[0] >= x.shape[1]:
        w, v = np.linalg.eigh(x.T @ x)
        projected = v.T @ (x.T @ y)
        for alpha in alphas:
            beta = v @ (projected / (w + alpha)[:, None])
            out.append([e @ beta + y_mean for e in evals])
    else:
        w, u = np.linalg.eigh(x @ x.T)
        projected = u.T @ y
        for alpha in alphas:
            dual = u @ (projected / (w + alpha)[:, None])
            out.append([(e @ x.T) @ dual + y_mean for e in evals])
    return out


def fit_predict(
    x: Array, y: Array, train: npt.NDArray[np.int64], parts: list[npt.NDArray[np.int64]]
) -> list[list[Array]]:
    """``ridge_path`` over targets grouped by which training samples measured them."""
    measured = ~np.isnan(y[train])
    patterns, inverse = np.unique(measured, axis=1, return_inverse=True)
    out = [[np.full((len(p), y.shape[1]), np.nan) for p in parts] for _ in ALPHAS]
    for k in range(patterns.shape[1]):
        columns = np.flatnonzero(inverse.ravel() == k)
        rows = train[patterns[:, k]]
        predictions = ridge_path(
            x[rows], y[np.ix_(rows, columns)], [x[p] for p in parts], ALPHAS
        )
        for a in range(len(ALPHAS)):
            for e in range(len(parts)):
                out[a][e][:, columns] = predictions[a][e]
    return out


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
                    reason="final evaluation at the penalty selected on validation"
                )
                predictions = fit_predict(x, task.y, train, [val, test])
                val_mse = [
                    float(np.nanmean((task.y[val] - predictions[a][0]) ** 2))
                    for a in range(len(ALPHAS))
                ]
                best = int(np.argmin(val_mse))
                family, seed = re.fullmatch(r"(.+)_s(\d+)", split_name).groups()  # type: ignore[union-attr]
                row: dict[str, object] = {
                    "task": args.task,
                    "split": split_name,
                    "family": family,
                    "seed": int(seed),
                    "feature": feature,
                    "combine": combine,
                    "n_features": x.shape[1],
                    "alpha": ALPHAS[best],
                    "n_train": len(train),
                    "n_val": len(val),
                    "n_test": len(test),
                    **mean_metrics(task.y[val], predictions[best][0], "val"),
                    **mean_metrics(task.y[test], predictions[best][1], "test"),
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
                        val_prediction=predictions[best][0],
                        test_prediction=predictions[best][1],
                        alpha=ALPHAS[best],
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
                                task.y[test][keep], predictions[best][1][keep], "t"
                            )["t_pearson"]
                            if keep.sum() >= 3
                            else float("nan")
                        )
                rows.append(row)
                print(
                    f"{args.task} {feature} {combine} {split_name} alpha {ALPHAS[best]:g} "
                    f"val {row['val_pearson']:.3f} test {row['test_pearson']:.3f} "
                    f"({time.perf_counter() - started:.0f} s)",
                    flush=True,
                )
        pd.DataFrame(rows).to_csv(
            osp.join(RESULTS, "probes", f"{args.task}.{feature}.csv"), index=False
        )


if __name__ == "__main__":
    main()
