#!/usr/bin/env python
# scripts/lit_reocr_si.py
# [[scripts.lit_reocr_si]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/lit_reocr_si.py

r"""Re-OCR every SI PDF of a mirror key into per-PDF figure directories (issue #579).

Command-line wrapper for :mod:`torchcell.literature.reocr_si`, which documents the
phases (Zotero and backfill precheck, OCR, check, retire, check, backfill) and holds
the tested logic. Loads the repo ``.env`` (``DATA_ROOT``: the mirror is
``$DATA_ROOT/torchcell-library`` and MinerU's model cache
``$DATA_ROOT/models/mineru/hf_cache``; the ``ZOTERO_*`` credentials).

MinerU runs on the GPU by default (``--device-mode cuda``). Run under slurm with one
GPU from the repo root. The card is the owner's choice, so replace ``<CARD>`` with
whatever pins it (partition ``main`` advertises ``gpu:rtx6000:4``); ``--mem`` and
``--time`` are not measured::

    sbatch -p main -N 1 --ntasks=1 --gres=gpu:rtx6000:1 <CARD> --cpus-per-task=8 \
      --mem=64g --time=0-06:00:00 -J lit-reocr-579 \
      --output=$HOME/lit-reocr-579_%j.out \
      --wrap "cd $HOME/Documents/projects/torchcell && PYTHONPATH=\$PWD \
    $HOME/miniconda3/envs/torchcell/bin/python scripts/lit_reocr_si.py \
    --key leeMappingCellularResponse2014 \
    --key ohyaHighdimensionalLargescalePhenotyping2005"
"""

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

from torchcell.literature.reocr_si import main  # noqa: E402

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    sys.exit(main())
