# experiments/032-amino-acid-module/scripts/train_amino_acid_module.py
# [[experiments.032-amino-acid-module.scripts.train_amino_acid_module]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/032-amino-acid-module/scripts/train_amino_acid_module

r"""The amino-acid module's training entry point.

The three metabolic-module studies (028 essentiality, 031 betaxanthin, 032 amino acids)
share one trainer, ``experiments/028-gene-essentiality/scripts/train_essentiality.py``,
where the module was first built. An amino-acid cell launched through that file shows
up on the node as an essentiality process, which misreads what is running
(2026-09-28). This entry point is the same program under the study's own name: it
imports the shared trainer and hands it the same arguments, so every flag, every
result field and every readout are unchanged, and ``code_provenance`` still records the
shared file. ``ess_cell.sh`` selects it from ``ESS_STUDY``. Run from the repo root
exactly as the shared trainer is run::

    python experiments/032-amino-acid-module/scripts/train_amino_acid_module.py \
        --arm pooled --head mulleder19 --seed 2700 ...
"""

from __future__ import annotations

import os.path as osp
import sys

sys.path.insert(
    0, osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))), "028-gene-essentiality", "scripts")
)
from train_essentiality import main  # noqa: E402

if __name__ == "__main__":
    main()
