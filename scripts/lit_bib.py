#!/usr/bin/env python
# scripts/lit_bib.py
# [[scripts.lit_bib]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/lit_bib.py

r"""Regenerate the Dendron bibliography from Zotero -- do not hand-edit ``bib.bib``.

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

Per-document bibliography (one group collection unioned with one personal
collection, written wherever you point it)::

    python scripts/lit_bib.py \
        --collection FE8DQKUH --user-collection AC8MFJXK \
        --out notes-tex/microbe-perturb-seq/references.bib

Cadence: this is an EXPLICIT step, never part of a document build. A build that
pulls from a live API is not reproducible, since the same commit yields a
different PDF once Zotero changes. Regenerate when papers are added, commit the
result, and let builds consume the committed file.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr

# Resolve project root from this file so cron's cwd does not matter.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

from torchcell.literature.bib import (  # noqa: E402
    BIB_RELPATHS,
    BibFileSyncReport,
    fetch_bibtex_entries,
    fetch_paired_collection_entries,
    fetch_union_bibtex_entries,
    sync_bib_file,
)
from torchcell.literature.zotero import ZoteroConfig, ZoteroLibrary  # noqa: E402

DEFAULT_ROOT_COLLECTION = "torchcell"

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
        "--group-only",
        action="store_true",
        help="Pull the group library only (skip the personal torchcell tree).",
    )
    parser.add_argument(
        "--root-collection",
        default=os.environ.get("ZOTERO_USER_ROOT_COLLECTION", DEFAULT_ROOT_COLLECTION),
        help="Personal collection tree to union in (default: $ZOTERO_USER_ROOT_COLLECTION).",
    )
    parser.add_argument(
        "--user-collection",
        default=None,
        metavar="NAME_OR_KEY",
        help=(
            "Pair with --collection to build a per-document bibliography from ONE "
            "group collection unioned with ONE personal collection. Accepts a "
            "collection key (stable across renames) or a name."
        ),
    )
    parser.add_argument(
        "--out",
        action="append",
        metavar="PATH",
        help=(
            "Write to this path instead of the two managed bibliographies; "
            "repeatable. Used for notes-tex/<slug>/references.bib."
        ),
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
        "--verbose", action="store_true", help="List every added/updated citation key."
    )
    args = parser.parse_args()

    group = ZoteroLibrary.from_env()
    user = None
    if not args.group_only:
        user = ZoteroLibrary(
            ZoteroConfig(
                library_id=os.environ["ZOTERO_USER_ID"],
                library_type="user",
                api_key=SecretStr(os.environ["ZOTERO_API_KEY"]),
            )
        )

    if args.user_collection is not None:
        if args.collection is None:
            parser.error("--user-collection requires --collection (the group side)")
        if user is None:
            parser.error("--user-collection cannot be combined with --group-only")
        incoming = fetch_paired_collection_entries(
            group,
            user,
            group_collection=args.collection,
            user_collection=args.user_collection,
        )
    elif args.collection is not None:
        incoming = fetch_bibtex_entries(group, collection=args.collection)
    else:
        incoming = fetch_union_bibtex_entries(
            group, user, user_root_collection=args.root_collection
        )
    if not incoming:
        sys.exit(
            "Zotero returned 0 entries -- refusing to touch the bibliography. "
            "Check ZOTERO_* credentials or the --collection name."
        )

    targets = (
        [Path(p) for p in args.out]
        if args.out
        else [_PROJECT_ROOT / rel for rel in BIB_RELPATHS]
    )
    for relpath in targets:
        report = sync_bib_file(
            relpath,
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
