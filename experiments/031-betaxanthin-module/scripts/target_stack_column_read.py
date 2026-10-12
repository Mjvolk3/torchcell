# experiments/031-betaxanthin-module/scripts/target_stack_column_read.py
# [[experiments.031-betaxanthin-module.scripts.target_stack_column_read]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-betaxanthin-module/scripts/target_stack_column_read

r"""What the stacking store's one column reads on the pinned genes by itself.

``target_forest_stack_store.py`` writes the forest's read of the GO neighbor mean as a
one-column store (``YPD_bx_gonbrforest``): out-of-fold values on the training genes,
full-train predictions everywhere else. The transformer that reads the column (round
234, w254) selects its epoch on the 270 validation strains, so the column's own read on
the 629 pinned genes is the number the transformer should reach if it passes the column
through: this script computes it, as the Spearman of the stored column against the
pinned targets resolved from the reference arm's dump (``target_store_baselines.
arm_targets``), plus the same on the validation genes' targets when a validation dump is
present. Writes ``results/target_stack_column_read.json``. Run from the repo root::

    PYTHONHASHSEED=0 PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/031-betaxanthin-module/scripts/target_stack_column_read.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import sys
import time

import numpy as np
from dotenv import load_dotenv
from scipy.stats import spearmanr

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "scripts"))
sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "031-betaxanthin-module", "scripts"))
from target_store_baselines import arm_targets  # noqa: E402

TABLE_DIR = osp.join(DATA_ROOT, "data/torchcell/reference_flux")
STORES = ("YPD_bx_gonbrforest",)


def main() -> None:
    assert os.environ.get("PYTHONHASHSEED") == "0"
    started = time.time()
    targets = arm_targets("betaxanthin")
    reads = {}
    for name in STORES:
        z = np.load(osp.join(TABLE_DIR, f"ess_solve_encodings_{name}.npz"), allow_pickle=False)
        assert z["v_deletion"].shape[1] == 1, name
        col = {str(g): float(v[0]) for g, v in zip(z["genes"], z["v_deletion"], strict=True)}
        genes = sorted(g for g in targets if g in col)
        rho = float(spearmanr([col[g] for g in genes], [targets[g] for g in genes])[0])
        reads[name] = {
            "n_pinned_with_target": len(targets),
            "n_pinned_with_row": len(genes),
            "pinned_spearman": rho,
            "provenance": json.loads(str(z["provenance"])),
        }
        print(f"  {name}: {len(genes)} pinned genes, column alone Spearman {rho:+.4f}", flush=True)
    out = osp.join(EXPERIMENT_ROOT, "031-betaxanthin-module", "results", "target_stack_column_read.json")
    with open(out, "w") as fh:
        json.dump(
            {
                "generated_by": "experiments/031-betaxanthin-module/scripts/target_stack_column_read.py",
                "statistic": "Spearman of the store's one column against the pinned targets of the reference arm's dump",
                "stores": reads,
                "seconds": time.time() - started,
            },
            fh,
            indent=2,
        )
    print(out)


if __name__ == "__main__":
    main()
