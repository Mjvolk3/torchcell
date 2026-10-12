# experiments/031-betaxanthin-module/scripts/target_coverage_split.py
# [[experiments.031-betaxanthin-module.scripts.target_coverage_split]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-betaxanthin-module/scripts/target_coverage_split

r"""How much of a store arm's read is the store's COVERAGE rather than its values.

The trainer's solve table reads zeros in every column for a gene the store does not
cover (``solve_feature_table``), so a measured store such as the M\"ulleder pool hands
the slot an indicator beside its values: a pinned gene with no pool row is a strain the
screen did not measure. Rebuilding the pool-through-256 slot from its checkpoint with
zeros for those genes reproduces the dump to a Spearman of 1.0000 (2026-09-27), and the
73 uncovered pinned genes have a mean standardized target of $-0.26$ against $+0.10$ for
the 556 covered ones.

For each arm named, from the per-gene prediction dumps, this script reports per seed the
pinned-test Spearman on all pinned genes, on the covered genes alone and on the uncovered
genes alone, the mean target of each group, and where the uncovered genes' predictions
sit in the covered genes' distribution. Coverage is read from the store's ``genes``
array. A control with no store (round 128) is read the same way, so the subset's own
difficulty is separated from what the indicator adds.

Writes ``results/target_coverage_split_<name>.json``. Run from the repo root::

    ~/miniconda3/envs/torchcell/bin/python \
        experiments/031-betaxanthin-module/scripts/target_coverage_split.py \
        --study 031-betaxanthin-module --name aapool --store YPD_bx_aapool \
        --stems w206_bxdelaapoolmlp256_h12l1p1_e200 w137_bxdel_h12l1p1_e200 ...
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import time

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel
from scipy.stats import spearmanr

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
TABLE_DIR = osp.join(DATA_ROOT, "data/torchcell/reference_flux")


class SeedSplit(BaseModel):
    """One cell's pinned read split by the store's coverage of the gene."""

    seed: int
    n_pinned: int
    #: For a vector head every statistic is the mean over its columns, each read on the
    #: rows with a target in that column.
    n_covered: int
    spearman_all: float
    spearman_covered: float
    spearman_uncovered: float
    mean_target_covered: float
    mean_target_uncovered: float
    #: The mean over uncovered genes of the fraction of covered genes predicted BELOW
    #: them: 0.5 is no separation, 0 puts every uncovered gene under every covered one.
    uncovered_prediction_percentile: float
    coverage_indicator_spearman: float


class ArmSplit(BaseModel):
    """An arm's seeds and their means."""

    stem: str
    seeds: list[SeedSplit]
    mean_spearman_all: float
    mean_spearman_covered: float
    mean_spearman_uncovered: float
    mean_uncovered_prediction_percentile: float


def _dump_path(results_dir: str, stem: str, seed: int) -> str:
    with open(osp.join(results_dir, f"{stem}_s{seed}.json")) as fh:
        return str(json.load(fh)["test"]["dump_path"])


def _column(rows: list[dict], key: str, j: int | None) -> np.ndarray:
    """Column ``j`` of a vector head's ``key`` (None for a scalar head), NaN where absent."""
    if j is None:
        return np.array([np.nan if r[key] is None else float(r[key]) for r in rows])
    return np.array(
        [np.nan if r[key][j] is None else float(r[key][j]) for r in rows], dtype=float
    )


def split_cell(results_dir: str, stem: str, seed: int, covered: set[str]) -> SeedSplit:
    """One cell's pinned read by coverage; a vector head averages over its columns."""
    with open(_dump_path(results_dir, stem, seed)) as fh:
        dump = json.load(fh)
    rows = [
        r
        for r in dump["predictions"]
        if r["pinned"]
        and (
            r["target"] is not None
            if not isinstance(r["target"], list)
            else any(t is not None for t in r["target"])
        )
    ]
    cov = np.array([any(g in covered for g in r["genes"]) for r in rows])
    assert 0 < cov.sum() < len(cov), (cov.sum(), len(cov))
    width = len(rows[0]["pred"]) if isinstance(rows[0]["pred"], list) else None
    columns = [None] if width is None else list(range(width))
    per: dict[str, list[float]] = {
        k: [] for k in ("all", "cov", "unc", "pct", "ind", "tc", "tu")
    }
    n_used: list[int] = []
    for j in columns:
        pred, target = _column(rows, "pred", j), _column(rows, "target", j)
        fin = np.isfinite(target)
        p, t, c = pred[fin], target[fin], cov[fin]
        n_used.append(int(fin.sum()))
        per["all"].append(float(spearmanr(p, t)[0]))
        per["cov"].append(float(spearmanr(p[c], t[c])[0]))
        per["unc"].append(float(spearmanr(p[~c], t[~c])[0]))
        per["pct"].append(float(np.mean([np.mean(p[c] < x) for x in p[~c]])))
        per["ind"].append(float(spearmanr(c.astype(float), t)[0]))
        per["tc"].append(float(t[c].mean()))
        per["tu"].append(float(t[~c].mean()))
    return SeedSplit(
        seed=seed,
        n_pinned=len(rows),
        n_covered=int(cov.sum()),
        spearman_all=float(np.mean(per["all"])),
        spearman_covered=float(np.mean(per["cov"])),
        spearman_uncovered=float(np.mean(per["unc"])),
        mean_target_covered=float(np.mean(per["tc"])),
        mean_target_uncovered=float(np.mean(per["tu"])),
        uncovered_prediction_percentile=float(np.mean(per["pct"])),
        coverage_indicator_spearman=float(np.mean(per["ind"])),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument(
        "--store", required=True, help="the store whose coverage splits the pin"
    )
    ap.add_argument("--stems", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(2700, 2706)))
    args = ap.parse_args()
    started = time.time()
    results_dir = osp.join(EXPERIMENT_ROOT, args.study, "results")
    payload = np.load(
        osp.join(TABLE_DIR, f"ess_solve_encodings_{args.store}.npz"), allow_pickle=False
    )
    covered = {str(g) for g in payload["genes"]}
    arms: list[ArmSplit] = []
    for stem in args.stems:
        seeds = [split_cell(results_dir, stem, s, covered) for s in args.seeds]
        arm = ArmSplit(
            stem=stem,
            seeds=seeds,
            mean_spearman_all=float(np.mean([s.spearman_all for s in seeds])),
            mean_spearman_covered=float(np.mean([s.spearman_covered for s in seeds])),
            mean_spearman_uncovered=float(
                np.mean([s.spearman_uncovered for s in seeds])
            ),
            mean_uncovered_prediction_percentile=float(
                np.mean([s.uncovered_prediction_percentile for s in seeds])
            ),
        )
        arms.append(arm)
        s0 = seeds[0]
        print(
            f"{stem:48s} all {arm.mean_spearman_all:+.4f} covered {arm.mean_spearman_covered:+.4f} "
            f"uncovered {arm.mean_spearman_uncovered:+.4f} pct {arm.mean_uncovered_prediction_percentile:.3f} "
            f"(n {s0.n_pinned}, covered {s0.n_covered}; targets {s0.mean_target_covered:+.3f} / "
            f"{s0.mean_target_uncovered:+.3f}; indicator {s0.coverage_indicator_spearman:+.3f})",
            flush=True,
        )
    out = {
        "generated_by": "experiments/031-betaxanthin-module/scripts/target_coverage_split.py",
        "name": args.name,
        "store": args.store,
        "n_store_genes": len(covered),
        "statistic": (
            "pinned-test Spearman on all pinned genes, on the genes the store covers and "
            "on the rest; the mean standardized target of each group; the mean fraction "
            "of covered genes predicted below an uncovered one; the Spearman of the "
            "coverage indicator itself against the target"
        ),
        "arms": [a.model_dump() for a in arms],
        "seconds": time.time() - started,
    }
    path = osp.join(results_dir, f"target_coverage_split_{args.name}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(path)


if __name__ == "__main__":
    main()
