# experiments/038-env-chemgen-vanacloig-cgt-corrected/scripts/stack_predictions.py
# [[experiments.038-env-chemgen-vanacloig-cgt-corrected.scripts.stack_predictions]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/038-env-chemgen-vanacloig-cgt-corrected/scripts/stack_predictions
"""Does averaging ridge with a neural model beat either alone?

Every trainer saves its seed-ensemble prediction for every gene and compound
(``$DATA_ROOT/experiments/038-env-chemgen-vanacloig-cgt-corrected/predictions/<sweep>/
<name>_fold<k>_seed<s>.npy``), and the ladder saves the nested ridge reference
(``ridge_fold<k>_seed<s>.npy``). A stack is the plain mean of two or more of these, in
response units, with no fitted weight, so nothing is chosen on the test compounds. A
member written ``a,b,c`` is first averaged within itself, so seed blocks of one model
(three 3-seed ensembles) enter the stack as one nine-seed member with one vote. It is
scored exactly as its members were and written as a pseudo-sweep
``results/factorized/stack/<stack>_scores.csv`` that ``compare_models.py`` reads.

    python stack_predictions.py --stack ridge+r3_multisource/ms_wildenhain \
                                --stack ridge+r2_table/tb_fcfp_bil_raw
"""

from __future__ import annotations

import argparse
import glob
import os
import os.path as osp
import re
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, osp.dirname(__file__))
from train_factorized import CELL_TABLE, EMBEDDING_DIR, PREDICTIONS  # noqa: E402
from vanacloig_data import load_cells, make_folds, score_compounds  # noqa: E402

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
OUT = osp.join(
    EXPERIMENT_ROOT,
    "038-env-chemgen-vanacloig-cgt-corrected",
    "results",
    "factorized",
    "stack",
)
FILE = re.compile(r"_fold(?P<fold>\d+)_seed(?P<seed>\d+)\.npy$")


def member_paths(member: str) -> dict[tuple[int, int], str]:
    """(fold seed, fold) -> file for one member, ``ridge`` or ``<sweep>/<name>``."""
    pattern = (
        osp.join(PREDICTIONS, "ridge_fold*_seed*.npy")
        if member == "ridge"
        else osp.join(PREDICTIONS, f"{member}_fold*_seed*.npy")
    )
    out = {}
    for path in glob.glob(pattern):
        m = FILE.search(path)
        assert m is not None, path
        out[(int(m["seed"]), int(m["fold"]))] = path
    assert out, f"no predictions for {member}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", action="append", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-val", type=int, default=4)
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cells = load_cells(CELL_TABLE, osp.join(EMBEDDING_DIR, "fcfp4_count.npz"))

    for stack in args.stack:
        groups = [
            [member_paths(m) for m in member.split(",")] for member in stack.split("+")
        ]
        keys = set.intersection(*[set(p) for group in groups for p in group])
        frames = []
        for fold_seed, fold_index in sorted(keys):
            fold = make_folds(
                len(cells.compounds), args.n_folds, args.n_val, fold_seed
            )[fold_index]
            pool = sorted(fold.train + fold.val)
            prediction = np.mean(
                [
                    np.mean(
                        [
                            np.load(p[(fold_seed, fold_index)]).astype(np.float64)
                            for p in group
                        ],
                        axis=0,
                    )
                    for group in groups
                ],
                axis=0,
            )
            frames.append(
                score_compounds(cells, prediction, pool, fold.test).assign(
                    member="ensemble",
                    selected_step=-1,
                    fold=fold_index,
                    fold_seed=fold_seed,
                )
            )
        name = stack.replace("/", "_")
        scores = pd.concat(frames, ignore_index=True).assign(name=name)
        scores.to_csv(osp.join(OUT, f"{name}_scores.csv"), index=False)
        centered = scores[scores["target"] == "centered"]
        print(
            f"{stack}: {len(keys)} folds, centered median "
            f"{centered['spearman'].median():.3f}, mean {centered['spearman'].mean():.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
