#!/usr/bin/env python
# scripts/lit_bib.py
# [[scripts.lit_bib]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/lit_bib.py

"""Regenerate the Dendron bibliography from Zotero -- do not hand-edit ``bib.bib``.

Pulls BibTeX over the Zotero **Web API** (pyzotero) and merges it into
``notes/assets/bib/bib.bib`` and its publish twin. Runs headless: no Zotero desktop,
no Better BibTeX, no localhost endpoint -- so it works on GilaHyper, in cron, or on
the Mac. Entry keys are Zotero's native ``citationKey``, the same string that names
each paper's mirror directory under ``<DATA_ROOT>/torchcell-library/``, so a note's
``@key`` and the OCR markdown it cites line up with no mapping table.

The merge is **add-only** by default: the bibliography predates the Zotero group and
holds hundreds of entries the group does not, so new Zotero keys are appended and
every existing entry is left byte-identical. Nothing is ever dropped -- a sync that
would shrink a file raises instead of writing. Pass ``--update-existing`` to also
refresh shared keys field-wise from Zotero (measured: that expands 23 legacy
Nature-style journal abbreviations to full names, which is why it is not the
default).

Usage::

    python scripts/lit_bib.py                              # add new Zotero papers
    python scripts/lit_bib.py --dry-run                    # report the delta, write nothing
    python scripts/lit_bib.py --collection microbe-perturb-seq   # one collection only
    python scripts/lit_bib.py --update-existing            # also refresh shared entries
    python scripts/lit_bib.py --verbose                    # list every added/updated key
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Resolve project root from this file so cron's cwd does not matter.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

from torchcell.literature.bib import (  # noqa: E402
    BIB_RELPATHS,
    BibFileSyncReport,
    fetch_bibtex_entries,
    sync_bib_file,
)
from torchcell.literature.zotero import ZoteroLibrary  # noqa: E402

log = logging.getLogger("lit_bib")


def _report(report: BibFileSyncReport, verbose: bool) -> None:
    """Log one file's outcome, optionally listing every changed key."""
    log.info(report.summary())
    if not verbose:
        return
    for change in report.by_mode("added"):
        log.info("  added:   %s", change.citation_key)
    for change in report.by_mode("updated"):
        log.info("  updated: %s", change.citation_key)


def main() -> None:
    """Pull Zotero once, merge into every managed bibliography, report the delta."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection",
        default=None,
        metavar="NAME",
        help="Restrict the pull to one Zotero collection (default: whole library).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change; write nothing.",
    )
    parser.add_argument(
        "--update-existing",
        action="store_true",
        help=(
            "Also refresh entries Zotero already knows (field-wise; local-only "
            "fields like biblatex shortjournal are kept). Default is add-only."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="List every added/updated citation key.",
    )
    args = parser.parse_args()

    lib = ZoteroLibrary.from_env()
    incoming = fetch_bibtex_entries(lib, collection=args.collection)
    if not incoming:
        sys.exit(
            "Zotero returned 0 entries -- refusing to touch the bibliography. "
            "Check ZOTERO_* credentials or the --collection name."
        )

    for relpath in BIB_RELPATHS:
        report = sync_bib_file(
            _PROJECT_ROOT / relpath,
            incoming,
            dry_run=args.dry_run,
            update_existing=args.update_existing,
        )
        _report(report, args.verbose)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    main()
