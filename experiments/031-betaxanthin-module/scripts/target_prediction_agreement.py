# experiments/031-betaxanthin-module/scripts/target_prediction_agreement.py
# [[experiments.031-betaxanthin-module.scripts.target_prediction_agreement]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-betaxanthin-module/scripts/target_prediction_agreement

r"""How far two arms' pinned predictions agree, seed by seed, against seed-to-seed spread.

Two arms trained on the same seeds are compared on the per-gene prediction dumps the
trainer writes (``test.dump_path``): for each seed the Spearman and Pearson between the
two arms' predictions on the pinned test records, beside the Spearman between every pair
of seeds of the first arm, which is what two runs that differ only by initialization
agree to. An arm change that moves the prediction less than a seed does is smaller than
the seed; one that leaves a seed's ranking at 1.0000 did not reach the prediction at all
(round 203, transformer dropout 0.3 on the pool through 256).

Writes ``results/target_prediction_agreement_<name>.json`` in the study. Run from the repo
root::

    ~/miniconda3/envs/torchcell/bin/python \
        experiments/031-betaxanthin-module/scripts/target_prediction_agreement.py \
        --study 031-betaxanthin-module --name w220_vs_w206 \
        --arm w220_bxdelaapoolmlp256drop_h12l1p1_e200 \
        --control w206_bxdelaapoolmlp256_h12l1p1_e200
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import time
from itertools import combinations

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel
from scipy.stats import spearmanr

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]


class SeedAgreement(BaseModel):
    """One seed's agreement between the arm's and the control's pinned predictions."""

    seed: int
    n_pinned: int
    spearman: float
    pearson: float
    arm_select_epoch: int
    control_select_epoch: int
    arm_spearman_vs_target: float
    control_spearman_vs_target: float


def _dump(
    results_dir: str, stem: str, seed: int
) -> tuple[dict[int, float], dict[str, object]]:
    path = osp.join(results_dir, f"{stem}_s{seed}.json")
    with open(path) as fh:
        result = json.load(fh)
    with open(result["test"]["dump_path"]) as fh:
        dump = json.load(fh)
    preds = {
        int(r["record_index"]): float(r["pred"])
        for r in dump["predictions"]
        if r["pinned"]
    }
    return preds, result


def _agree(a: dict[int, float], b: dict[int, float]) -> tuple[int, float, float]:
    keys = sorted(set(a) & set(b))
    assert len(keys) == len(a) == len(b), (len(keys), len(a), len(b))
    x = np.array([a[k] for k in keys])
    y = np.array([b[k] for k in keys])
    return len(keys), float(spearmanr(x, y)[0]), float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(2700, 2706)))
    args = ap.parse_args()
    started = time.time()
    results_dir = osp.join(EXPERIMENT_ROOT, args.study, "results")
    rows: list[SeedAgreement] = []
    control_preds: dict[int, dict[int, float]] = {}
    for seed in args.seeds:
        a, ra = _dump(results_dir, args.arm, seed)
        c, rc = _dump(results_dir, args.control, seed)
        control_preds[seed] = c
        n, rho, r = _agree(a, c)
        rows.append(
            SeedAgreement(
                seed=seed,
                n_pinned=n,
                spearman=rho,
                pearson=r,
                arm_select_epoch=int(ra["select_epoch"]),
                control_select_epoch=int(rc["select_epoch"]),
                arm_spearman_vs_target=float(ra["test"]["pinned"]["spearman"]),
                control_spearman_vs_target=float(rc["test"]["pinned"]["spearman"]),
            )
        )
    seed_pairs = [
        {"seeds": [s, t], "spearman": _agree(control_preds[s], control_preds[t])[1]}
        for s, t in combinations(args.seeds, 2)
    ]
    same = [r.spearman for r in rows]
    pairs = [p["spearman"] for p in seed_pairs]
    summary = {
        "mean_same_seed_spearman": float(np.mean(same)),
        "min_same_seed_spearman": float(np.min(same)),
        "n_seeds_at_or_above_0.9999": int(sum(s >= 0.9999 for s in same)),
        "mean_control_seed_pair_spearman": float(np.mean(pairs)),
        "min_control_seed_pair_spearman": float(np.min(pairs)),
        "max_control_seed_pair_spearman": float(np.max(pairs)),
        "n_seeds": len(rows),
    }
    out = {
        "generated_by": "experiments/031-betaxanthin-module/scripts/target_prediction_agreement.py",
        "name": args.name,
        "arm": args.arm,
        "control": args.control,
        "statistic": (
            "Spearman and Pearson between the arm's and the control's pinned-test predictions "
            "at the same seed, from the per-gene dumps; the control's seed pairs as the "
            "reference for what initialization alone moves"
        ),
        "summary": summary,
        "seeds": [r.model_dump() for r in rows],
        "control_seed_pairs": seed_pairs,
        "seconds": time.time() - started,
    }
    path = osp.join(results_dir, f"target_prediction_agreement_{args.name}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    for r in rows:
        print(
            f"seed {r.seed}: arm ep {r.arm_select_epoch} control ep {r.control_select_epoch} "
            f"agreement {r.spearman:.4f} / {r.pearson:.4f}; targets {r.arm_spearman_vs_target:+.4f} "
            f"vs {r.control_spearman_vs_target:+.4f}"
        )
    print(json.dumps(summary, indent=1))
    print(path)


if __name__ == "__main__":
    main()
