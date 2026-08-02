# experiments/020-cachera-betaxanthin/scripts/train_metabolism.py
# [[experiments.020-cachera-betaxanthin.scripts.train_metabolism]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/020-cachera-betaxanthin/scripts/train_metabolism
"""Hydra entrypoint for the three single-target metabolism arms.

WHY THIS IS A THIN WRAPPER RATHER THAN A COPY
---------------------------------------------
The training loop, the multitask loss, the head construction and the metric plumbing all
live in ``experiments/019-simb-multimodal/scripts/train_cgt_multitask.py`` and are shared
verbatim. Copying ~2,000 lines into a new experiment folder would let the two drift, and a
drifted trainer is exactly the kind of difference that silently invalidates a comparison
between a 019 arm and a 020 arm. So 020 owns its **configs, launcher and results** -- which
is what makes it a separate experiment -- and imports the trainer.

The 019 script is not an installed module, so its directory goes on ``sys.path`` first, the
same pattern ``run_pigment_conditions.py`` already uses.

WHAT THESE ARMS ARE, PLAINLY
----------------------------
Gene-token CGT pointed at metabolic labels. 6,607 gene tokens + CLS -> graph-regularized
transformer -> equivariant perturbation operator -> one readout head. **No stoichiometry,
no reactions, no flux.** The metabolism is in the LABEL, not in the model. The flux layer
is specified in [[plan.cgt-metabolism-flux-layer.2026.07.26]] and deliberately deferred:
published k_cat covers 79 of Yeast9's 1,161 genes (6.8 %), so a capacity constraint built
today would be ~93 % predicted parameters.

Usage (one target per GPU):

    python experiments/020-cachera-betaxanthin/scripts/train_metabolism.py \
        --config-name=betaxanthin
"""

import os.path as osp
import sys

import hydra
from omegaconf import DictConfig

_REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
_TRAINER_DIR = osp.join(_REPO_ROOT, "experiments", "019-simb-multimodal", "scripts")
if _TRAINER_DIR not in sys.path:
    sys.path.insert(0, _TRAINER_DIR)

from train_cgt_multitask import (
    run_training,  # type: ignore[import-not-found]  # noqa: E402
)


@hydra.main(
    version_base=None,
    config_path=osp.join(osp.dirname(osp.abspath(__file__)), "..", "conf"),
    config_name="betaxanthin",
)
def main(cfg: DictConfig) -> None:
    """Train one single-target metabolism arm."""
    heads = list(cfg.multitask.active_heads)
    print("=" * 78)
    print(f"020 metabolism arm: active_heads={heads}")
    print(f"  graphs        : {len(list(cfg.cell_dataset.graphs))}")
    print(f"  embeddings    : {list(cfg.cell_dataset.node_embeddings)}")
    print(
        f"  select on     : {cfg.trainer.checkpoint.monitor} ({cfg.trainer.checkpoint.mode})"
    )
    print(f"  dist          : {cfg.multitask.dist}")
    print("=" * 78)
    run_training(cfg)


if __name__ == "__main__":
    main()
