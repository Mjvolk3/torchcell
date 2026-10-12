# experiments/028-gene-essentiality/scripts/cv_ensemble_readout.py
# [[experiments.028-gene-essentiality.scripts.cv_ensemble_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/cv_ensemble_readout.py
"""Score a K-fold ensemble on the held-out genes: the fold models' logits averaged per gene.

This is Flux Cone Learning's own test protocol (Merzbacher 2025, Methods): "The test set
results were computed by running the held-out test set through all 5-fold models and
averaging the deletion-level scores across all models." Each fold cell here trained on four
fifths of the 787 labeled non-test genes, selected its epoch on the fifth, and dumped its
logit for every held-out gene; the ensemble is the mean logit over folds, scored once.

Every fold's dump is resolved through its result file and checked against the result's cell
on the identity fields, so a fold whose dump was overwritten is refused rather than averaged.
The per-fold AUROCs are reported beside the ensemble's, with a gene bootstrap on each, and
the ensemble is compared with the single-split cells of the same configuration.

Run from the worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/028-gene-essentiality/scripts/cv_ensemble_readout.py --prefix w14_cv

Writes ``results/<prefix>_ensemble.json``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp
import re

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import roc_auc_score

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
PRED_DIR = os.getenv("TC_TEST_PRED_DIR", osp.join(DATA_ROOT, "test-predictions-028"))
IDENTITY_FIELDS = (
    "head",
    "arm",
    "seed",
    "hidden",
    "layers",
    "attention_heads",
    "preprocessor_layers",
    "epochs",
    "split_file",
)
N_BOOTSTRAP = 2000


class FoldScore(BaseModel):
    """One fold model on the held-out genes."""

    model_config = ConfigDict(extra="forbid")
    stem: str
    split_file: str
    select_epoch: int
    n_genes: int
    auroc: float
    ci_low: float
    ci_high: float


class EnsembleReport(BaseModel):
    """The K fold models and their mean-logit ensemble on one gene set."""

    model_config = ConfigDict(extra="forbid")
    prefix: str
    n_folds: int
    n_genes: int
    n_essential: int
    folds: list[FoldScore]
    fold_mean_auroc: float = Field(description="Mean of the per-fold AUROCs.")
    ensemble_auroc: float = Field(description="AUROC of the per-gene mean logit.")
    ensemble_ci_low: float
    ensemble_ci_high: float
    seeds_per_fold: int = 1
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


def identity(cell: dict[str, object]) -> dict[str, object]:
    return {k: cell.get(k) for k in IDENTITY_FIELDS}


def load_fold(result_file: str) -> tuple[FoldScore, dict[str, float], dict[str, int]]:
    """A fold's per-gene logit and label on its pinned test genes, identity-checked."""
    with open(result_file) as fh:
        result = json.load(fh)
    stem = osp.splitext(osp.basename(result_file))[0]
    dump_file = osp.join(PRED_DIR, f"{stem}.json")
    assert osp.isfile(dump_file), f"{stem} has no dump at {dump_file}."
    with open(dump_file) as fh:
        dump = json.load(fh)
    assert identity(dump["cell"]) == identity(result["cell"]), (
        f"{dump_file} holds {identity(dump['cell'])}, not {identity(result['cell'])}."
    )
    logits: dict[str, float] = {}
    labels: dict[str, int] = {}
    for p in dump["predictions"]:
        if not p["pinned"] or len(p["genes"]) != 1 or p["target"] is None:
            continue
        gene = str(p["genes"][0])
        logits[gene] = float(p["pred"])
        labels[gene] = int(round(float(p["target"])))
    y = np.array([labels[g] for g in sorted(labels)])
    s = np.array([logits[g] for g in sorted(labels)])
    point, lo, hi = bootstrap(y, s)
    score = FoldScore(
        stem=stem,
        split_file=str(result["cell"].get("split_file")),
        select_epoch=int(result["select_epoch"]),
        n_genes=len(labels),
        auroc=point,
        ci_low=lo,
        ci_high=hi,
    )
    assert abs(point - float(result["test"]["pinned"]["auroc"])) < 1e-6, (
        f"{stem}: dump scores {point:.6f}, result recorded "
        f"{result['test']['pinned']['auroc']:.6f}."
    )
    return score, logits, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="w14_cv")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[2700],
        help=(
            "Seeds per fold. With more than one, each fold model is the per-gene mean "
            "logit over its seeds (as graph_baselines.py --seeds averages), the fold "
            "AUROC scores that mean, and the ensemble averages the K seed-averaged "
            "folds; the report is written as <prefix>_s<n>_ensemble.json."
        ),
    )
    args = parser.parse_args()
    files = sorted(glob.glob(osp.join(RESULTS, f"{args.prefix}*_s*.json")))
    files = [
        f
        for f in files
        if (m := re.search(r"_s(\d+)\.json$", f)) and int(m.group(1)) in args.seeds
    ]
    assert files, (
        f"no {args.prefix}* result files for seeds {args.seeds} under {RESULTS}."
    )
    folds: list[FoldScore] = []
    per_fold_logits: list[dict[str, float]] = []
    labels: dict[str, int] | None = None
    # Group by fold index (the digits between the prefix and the first underscore).
    by_fold: dict[str, list[str]] = {}
    for f in files:
        fold_id = re.match(rf".*{re.escape(args.prefix)}(\d+)_", osp.basename(f))
        assert fold_id, f
        by_fold.setdefault(fold_id.group(1), []).append(f)
    for fold_id in sorted(by_fold, key=int):
        seed_files = by_fold[fold_id]
        assert len(seed_files) == len(args.seeds), (fold_id, seed_files, args.seeds)
        scores, seed_logits = [], []
        for f in seed_files:
            score, logits, lab = load_fold(f)
            scores.append(score)
            seed_logits.append(logits)
            # Every fold pins the SAME released held-out genes with the same labels.
            assert labels is None or lab == labels, f"{f} scores a different gene set."
            labels = lab
        assert labels is not None
        fold_genes = sorted(labels)
        fold_logit = {
            g: float(np.mean([lg[g] for lg in seed_logits])) for g in fold_genes
        }
        if len(seed_files) == 1:
            folds.append(scores[0])
        else:
            y_f = np.array([labels[g] for g in fold_genes])
            s_f = np.array([fold_logit[g] for g in fold_genes])
            point_f, lo_f, hi_f = bootstrap(y_f, s_f)
            folds.append(
                FoldScore(
                    stem=f"{args.prefix}{fold_id} (seeds {', '.join(map(str, args.seeds))})",
                    split_file=scores[0].split_file,
                    select_epoch=-1,
                    n_genes=len(fold_genes),
                    auroc=point_f,
                    ci_low=lo_f,
                    ci_high=hi_f,
                )
            )
        per_fold_logits.append(fold_logit)
    assert labels is not None
    genes = sorted(labels)
    mean_logit = np.array(
        [float(np.mean([lg[g] for lg in per_fold_logits])) for g in genes]
    )
    y = np.array([labels[g] for g in genes])
    point, lo, hi = bootstrap(y, mean_logit)
    report = EnsembleReport(
        prefix=args.prefix,
        n_folds=len(folds),
        n_genes=len(genes),
        n_essential=int(y.sum()),
        folds=folds,
        fold_mean_auroc=float(np.mean([s.auroc for s in folds])),
        ensemble_auroc=point,
        ensemble_ci_low=lo,
        ensemble_ci_high=hi,
        seeds_per_fold=len(args.seeds),
        note=(
            "The ensemble is the per-gene mean of the fold models' logits on the released "
            "held-out genes, scored once; this is the protocol behind FCL's published test "
            "AUROC. The fold mean is the average of K single-model AUROCs and is the "
            "number to compare with a single-split cell."
            + (
                f" Each fold model is the mean logit over seeds {args.seeds}, and the "
                "fold AUROC scores that mean."
                if len(args.seeds) > 1
                else ""
            )
        ),
    )
    suffix = f"_s{len(args.seeds)}" if len(args.seeds) > 1 else ""
    out = osp.join(RESULTS, f"{args.prefix}{suffix}_ensemble.json")
    with open(out, "w") as fh:
        json.dump(report.model_dump(), fh, indent=2)
    for s in folds:
        print(
            f"  {s.stem:40s} sel {s.select_epoch:>3}  AUROC {s.auroc:.4f} "
            f"[{s.ci_low:.3f}, {s.ci_high:.3f}]"
        )
    print(
        f"fold mean {report.fold_mean_auroc:.4f}; ensemble {report.ensemble_auroc:.4f} "
        f"[{lo:.3f}, {hi:.3f}] on {len(genes)} genes ({int(y.sum())} essential)\n{out}"
    )


if __name__ == "__main__":
    main()
