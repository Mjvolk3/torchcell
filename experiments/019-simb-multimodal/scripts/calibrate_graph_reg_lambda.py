# experiments/019-simb-multimodal/scripts/calibrate_graph_reg_lambda.py
# [[experiments.019-simb-multimodal.scripts.calibrate_graph_reg_lambda]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/calibrate_graph_reg_lambda
"""Measure the graph-regularization strength each candidate lambda actually produces.

WHY. lambda is NOT portable. It is only meaningful relative to a specific normalization
and a specific number of regularized layers, and both changed under this branch:

  * normalization is now main's per-graph EDGE-COUNT form with lambda applied exactly
    once, replacing the doubled-lambda / summed-degree form;
  * `graph_reg_layer` went from all 8 layers to a SINGLE layer (010's setting), which
    shrinks the summed KL by roughly L.

Nothing raises when lambda is wrong by orders of magnitude. The first _006 launch ran with
the graph term at 99.99% of the objective, trained happily, and produced garbage -- the
only reason we caught it was the telemetry. So the grid gets measured, not inherited.

READS `val/graph_reg/ratio_to_data` = graph_term / data_term. 1.0 is parity. A usable grid
brackets roughly 0.01 (barely on) to ~10 (dominant), centred near parity.

Composes the config and calls `run_training` directly -- the SAME path
`optuna_joint_sweep.py` uses -- rather than the Hydra CLI entrypoint, so what is
calibrated is what will actually run.

Run from repo root (or via gh_calibrate_graph_reg_lambda.slurm):
    python experiments/019-simb-multimodal/scripts/calibrate_graph_reg_lambda.py
"""

from __future__ import annotations

import os
import os.path as osp
import sys

from dotenv import load_dotenv

# `load_dotenv()` searches upward from CWD; a git worktree has no .env unless
# setup-worktree.sh made one, so resolve the primary checkout as a fallback before any
# torchcell import (torchcell.graph.sgd reads DATA_ROOT at IMPORT time).
_WT_ENV = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", "..", ".env"))
load_dotenv(
    _WT_ENV
    if osp.exists(_WT_ENV)
    else osp.expanduser("~/Documents/projects/torchcell/.env")
)

sys.path.insert(0, osp.dirname(__file__))

from hydra import compose, initialize_config_dir  # noqa: E402
from train_cgt_multitask import run_training  # noqa: E402

CONF_DIR = osp.abspath(osp.join(osp.dirname(__file__), "../conf"))
LAMBDAS = [1e-8, 1e-6, 1e-4, 1e-2]
METRIC = "val/graph_reg/ratio_to_data"


def main() -> None:
    print(f"DATA_ROOT={os.environ['DATA_ROOT']}", flush=True)
    rows: list[tuple[float, float | None, float | None]] = []
    for lam in LAMBDAS:
        with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
            cfg = compose(
                config_name="cgt_expr_006",
                overrides=[
                    f"model.graph_regularization.graph_reg_lambda={lam}",
                    "model.hidden_channels=90",
                    "model.num_transformer_layers=4",
                    "multitask.dist=point",
                    "cell_dataset.node_embeddings=[random_100]",
                    "trainer.max_epochs=1",
                    "trainer.early_stopping.enabled=false",
                    "data_module.num_workers=4",
                    "wandb.project=torchcell_019_lambda_cal",
                ],
            )
        print(f"\n########## graph_reg_lambda={lam:.0e} ##########", flush=True)
        metrics = run_training(cfg)
        ratio = metrics.get(METRIC) if isinstance(metrics, dict) else None
        pear = (
            metrics.get("val/expression/pearson_per_feature")
            if isinstance(metrics, dict)
            else None
        )
        rows.append((lam, ratio, pear))
        print(f"[calib] lambda={lam:.0e}  {METRIC}={ratio}  pearson={pear}", flush=True)

    print("\n" + "=" * 62)
    print(f"{'lambda':>10} {'ratio_to_data':>16} {'val pearson_pf':>16}")
    for lam, ratio, pear in rows:
        r = f"{ratio:.4g}" if isinstance(ratio, float) else str(ratio)
        p = f"{pear:.4f}" if isinstance(pear, float) else str(pear)
        print(f"{lam:>10.0e} {r:>16} {p:>16}")
    print("target: a grid bracketing ~0.01 to ~10, centred near 1.0 (parity)")
    print("=" * 62)


if __name__ == "__main__":
    main()
