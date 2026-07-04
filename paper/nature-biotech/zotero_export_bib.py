"""
paper/nature-biotech/zotero_export_bib.py

Regenerate references.bib from the shared **torchcell group** library's `paper`
collection via the Better BibTeX local export endpoint. BBT preserves the *pinned*
citekeys (so they keep matching the manuscript's \\cite commands) and does its own
clean formatting. Reads Zotero's local data over localhost — no API key needed.

Requires Zotero + Better BibTeX running (localhost:23119). The group's `paper`
collection is the single polished source; this pulls it into references.bib.

Usage:
    python paper/nature-biotech/zotero_export_bib.py             # regenerate references.bib
    python paper/nature-biotech/zotero_export_bib.py --dry-run   # print, do not write
    python paper/nature-biotech/zotero_export_bib.py --collection database
    python paper/nature-biotech/zotero_export_bib.py --group-id 6582362 --collection W46ATS7B

Safety: refuses to overwrite references.bib unless the export has >= MIN_ENTRIES
@-entries, so an empty collection or a stopped Zotero can never blank the bib
(override with --force).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BBT_EXPORT = "http://localhost:23119/better-bibtex/export/collection"
DEFAULT_GROUP = os.getenv("ZOTERO_LIBRARY_ID", "6582362")  # torchcell shared group library
DEFAULT_COLLECTION = "W46ATS7B"    # its `paper` collection key (stable across renames)
BIB_PATH = Path(__file__).with_name("references.bib")
MIN_ENTRIES = 3                    # empty-collection / Zotero-down guard

_ENTRY = re.compile(r"(?m)^@\w+\s*\{")


def fetch(group_id: str, collection: str) -> str:
    """BibTeX for the group collection from the Better BibTeX local endpoint."""
    url = f"{BBT_EXPORT}?/{group_id}/{collection}.bibtex"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        sys.exit(f"Better BibTeX export failed: {exc}. Is Zotero + BBT running?  URL: {url}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate references.bib from the group `paper` collection.")
    ap.add_argument("--group-id", default=DEFAULT_GROUP, help="target group library id")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION, help="collection key or name")
    ap.add_argument("--dry-run", action="store_true", help="print export, do not write")
    ap.add_argument("--force", action="store_true", help="write even if below MIN_ENTRIES")
    args = ap.parse_args()

    bib = fetch(args.group_id, args.collection)
    n = len(_ENTRY.findall(bib))
    print(f"fetched {len(bib)} bytes, {n} entries from group {args.group_id}/{args.collection}")

    if args.dry_run:
        print(bib[:2000] + ("\n... (truncated)" if len(bib) > 2000 else ""))
        return
    if n < MIN_ENTRIES and not args.force:
        sys.exit(
            f"Refusing to overwrite {BIB_PATH.name}: only {n} entries (< {MIN_ENTRIES}). "
            "Collection empty or Zotero down? Populate it (zotero_copy_refs.py) or use --force."
        )
    BIB_PATH.write_text(bib)
    print(f"wrote {n} entries -> {BIB_PATH}")


if __name__ == "__main__":
    main()
