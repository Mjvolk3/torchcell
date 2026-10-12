#!/usr/bin/env python
# scripts/zotero_add_paper.py
# [[scripts.zotero_add_paper]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/zotero_add_paper.py

r"""File a dataset's primary paper, with its PDF, into a group-library database collection.

Command-line wrapper for :mod:`torchcell.literature.zotero_add`, which documents the
policy (what an agent may and may not write to Zotero), the PDF routes (publisher
version of record, then the raw mirror's recorded copy, then the PMC Article Datasets
bucket, then a browser-downloaded ``--pdf``) and the refusals, and holds the tested
logic. Loads the repo ``.env`` (``ZOTERO_LIBRARY_ID`` + ``ZOTERO_API_KEY`` of the
TorchCell group library; ``DATA_ROOT`` for the raw mirror).

Usage, from the repo root::

    # see what would be filed, write nothing
    python scripts/zotero_add_paper.py --doi 10.1093/ismejo/wrae096 \
        --collection database/Escherichia-coli --first-author Teteneva --dry-run

    # file it (a new item, with its PDF)
    python scripts/zotero_add_paper.py --doi 10.1093/ismejo/wrae096 \
        --collection database/Escherichia-coli --first-author Teteneva

    # attach a browser-downloaded PDF (and SI PDFs) to an existing item without one
    python scripts/zotero_add_paper.py --doi 10.1016/j.cell.2010.11.052 \
        --collection database/Escherichia-coli --first-author Nichols \
        --attach-to REEL3RAD --pdf ~/Downloads/nichols2011.pdf \
        --pdf-source-url https://doi.org/10.1016/j.cell.2010.11.052 \
        --si ~/Downloads/mmc1.pdf \
        --si-source-url https://doi.org/10.1016/j.cell.2010.11.052

Exit 0 filed (or would file, with ``--dry-run``), 2 refused, 3 no route yielded the PDF
(nothing written; the summary names the URL to open in a browser).
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

from torchcell.literature.zotero_add import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
