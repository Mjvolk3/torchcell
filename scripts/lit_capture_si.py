#!/usr/bin/env python
# scripts/lit_capture_si.py
# [[scripts.lit_capture_si]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/lit_capture_si.py

r"""Capture publisher supplementary files into mirrored literature keys.

Command-line wrapper for :mod:`torchcell.literature.capture_si`, which documents the
routes (PMC Article Datasets bucket first, then the publisher by Crossref member),
the outcome vocabulary and the manual path, and holds the tested logic. Loads the repo
``.env`` (``DATA_ROOT``: the mirror is ``$DATA_ROOT/torchcell-library``; the
``ZOTERO_*`` credentials for ``--collection``, which only reads Zotero).

Usage, from the repo root::

    # resolve and list, download nothing; report to a scratch directory
    python scripts/lit_capture_si.py --dry-run --collection Escherichia-coli \
        --collection Pseudomonas-putida --report-dir /path/to/scratch

    # capture into the mirror (report under torchcell-library/_sync_reports/)
    python scripts/lit_capture_si.py --collection Escherichia-coli

    # a few keys, or DOIs already mirrored; --ocr runs MinerU on captured SI PDFs (GPU)
    python scripts/lit_capture_si.py hawkinsMismatchCRISPRiReveals2020 --ocr
    python scripts/lit_capture_si.py --doi 10.1016/j.cels.2020.09.009

    # a scratch copy of the mirror
    python scripts/lit_capture_si.py someKey --mirror-root /scratch/.../torchcell-library

Exit 1 when any key ``failed``.
"""

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

from torchcell.literature.capture_si import main  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

if __name__ == "__main__":
    sys.exit(main())
