#!/usr/bin/env python
# scripts/lit_bib_store.py
# [[scripts.lit_bib_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/lit_bib_store.py

r"""Export every repo-declared bibliography into the mirror for tc-lit to serve.

Runs on the host that holds the mirror (GilaHyper), headless over the Zotero Web
API, and writes ``<DATA_ROOT>/torchcell-library/_bib/<name>.bib`` plus a
``manifest.json`` pinning each file's sha256. ``tc-lit`` then serves them at
``GET /bib`` (the manifest) and ``GET /bib/<name>`` (the file, with
``X-Artifact-SHA256``), so any machine with a tc-lit key pulls one verified
bibliography instead of regenerating its own through Better BibTeX.

The names are discovered, not configured (see
:func:`torchcell.literature.bib_store.discover_bib_specs`): ``paper`` for the
manuscript's group collection, one per ``notes-tex/<slug>/`` that declares a
``ZOTERO_COLLECTION``, and ``library`` for the whole torchcell union.

Usage::

    python scripts/lit_bib_store.py              # export every bibliography
    python scripts/lit_bib_store.py --list       # print the discovered specs, pull nothing
    python scripts/lit_bib_store.py --name paper --name eqtl-data-model   # a subset
    python scripts/lit_bib_store.py --dry-run    # pull + report counts, write nothing

Cadence: nightly from cron after ``lit_sync.py`` (see ``scripts/crontab.txt``), and
by hand right after a collection changes. Unlike ``lit_bib.py`` this writes nothing
git-tracked, so cron cannot dirty a checkout. Consumers pull an explicit, pinned
copy (``make bib-pull``) and commit it; a build never reads the store directly.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

from torchcell.literature.backfill import LIBRARY_SUBDIR  # noqa: E402
from torchcell.literature.bib_store import (  # noqa: E402
    DEFAULT_USER_ROOT_COLLECTION,
    BibSpec,
    discover_bib_specs,
    export_bib_store,
    fetch_scope_entries,
)
from torchcell.literature.zotero import ZoteroConfig, ZoteroLibrary  # noqa: E402

log = logging.getLogger("lit_bib_store")


def _describe(spec: BibSpec) -> str:
    """One line per spec for ``--list``."""
    return (
        f"{spec.name:<28} {spec.scope.model_dump(exclude_none=True)}  <- {spec.origin}"
    )


def main() -> None:
    """Discover the specs, pull each from Zotero, write the store + manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true", help="Print the discovered specs and exit."
    )
    parser.add_argument(
        "--name",
        action="append",
        metavar="NAME",
        help="Export only this bibliography (repeatable). Default: all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pull and report entry counts; write nothing.",
    )
    parser.add_argument(
        "--root-collection",
        default=os.environ.get(
            "ZOTERO_USER_ROOT_COLLECTION", DEFAULT_USER_ROOT_COLLECTION
        ),
        help="Personal collection tree for the `library` scope.",
    )
    args = parser.parse_args()

    group_id = os.environ["ZOTERO_LIBRARY_ID"]
    user_id = os.environ["ZOTERO_USER_ID"]
    specs = discover_bib_specs(
        _PROJECT_ROOT,
        group_library_id=group_id,
        user_library_id=user_id,
        user_root_collection=args.root_collection,
    )
    if args.name:
        known = {s.name for s in specs}
        missing = sorted(set(args.name) - known)
        if missing:
            sys.exit(f"unknown bibliography name(s): {missing}; known: {sorted(known)}")
        specs = [s for s in specs if s.name in set(args.name)]

    if args.list:
        for spec in specs:
            print(_describe(spec))
        return

    group = ZoteroLibrary.from_env()
    user = ZoteroLibrary(
        ZoteroConfig(
            library_id=user_id,
            library_type="user",
            api_key=SecretStr(os.environ["ZOTERO_API_KEY"]),
        )
    )

    if args.dry_run:
        for spec in specs:
            entries = fetch_scope_entries(spec.scope, group, user)
            log.info("dry-run: %s -> %d entries (not written)", spec.name, len(entries))
        return

    mirror_root = Path(os.environ["DATA_ROOT"]) / LIBRARY_SUBDIR
    manifest = export_bib_store(mirror_root, specs, group, user)
    for record in manifest.bibs:
        log.info(
            "%-28s %5d entries  sha256=%s", record.name, record.n_entries, record.sha256
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    main()
