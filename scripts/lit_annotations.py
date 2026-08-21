#!/usr/bin/env python
# scripts/lit_annotations.py
# [[scripts.lit_annotations]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/lit_annotations.py

"""Pull Zotero highlights, comments and notes into the literature mirror.

Writes ``annotations.json`` + ``annotations.md`` next to each paper's ``paper.md``,
so the notes written while reading are available on whatever machine the writing
happens on. GPU-free and fast -- annotations come from the Zotero API, not from OCR,
so this never re-downloads a PDF or re-runs MinerU.

Scope matches the capture pipeline: the personal ``torchcell/*`` tree (recursive,
set by ``ZOTERO_USER_ROOT_COLLECTION``) UNION the torchcell group. Every record
records which library it came from, and identical content found in both is emitted
once listing both -- so a paper annotated in both places is never double-counted and
never loses attribution.

Usage::

    python scripts/lit_annotations.py                 # capture into the mirror
    python scripts/lit_annotations.py --dry-run       # report per-paper tallies only
    python scripts/lit_annotations.py --group-only    # skip the personal library
    python scripts/lit_annotations.py --key <citekey> # one paper (repeatable)
"""

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

from torchcell.literature.annotations import (  # noqa: E402
    capture_annotations,
    collect_library_annotations,
    merge_annotations,
    personal_tree_item_keys,
    unmirrored_with_annotations,
)
from torchcell.literature.backfill import library_root  # noqa: E402
from torchcell.literature.zotero import ZoteroConfig, ZoteroLibrary  # noqa: E402

log = logging.getLogger("lit_annotations")

DEFAULT_ROOT_COLLECTION = "torchcell"


def main() -> None:
    """Collect annotations from both libraries and write them into the mirror."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    p.add_argument(
        "--group-only",
        action="store_true",
        help="Skip the personal library (group annotations only).",
    )
    p.add_argument(
        "--key",
        action="append",
        metavar="CITEKEY",
        help="Restrict to these citation keys; repeatable.",
    )
    p.add_argument(
        "--root-collection",
        default=os.environ.get("ZOTERO_USER_ROOT_COLLECTION", DEFAULT_ROOT_COLLECTION),
        help="Personal collection tree to scope to (default: $ZOTERO_USER_ROOT_COLLECTION).",
    )
    args = p.parse_args()

    group = ZoteroLibrary.from_env()
    sources = [collect_library_annotations(group, "group")]

    if not args.group_only:
        personal = ZoteroLibrary(
            ZoteroConfig(
                library_id=os.environ["ZOTERO_USER_ID"],
                library_type="user",
                api_key=SecretStr(os.environ["ZOTERO_API_KEY"]),
            )
        )
        tree = personal_tree_item_keys(personal, args.root_collection)
        sources.append(
            collect_library_annotations(personal, "personal", parent_keys=tree)
        )

    merged = merge_annotations(*sources)
    if args.key:
        merged = {k: v for k, v in merged.items() if k in set(args.key)}

    root = library_root(os.environ["DATA_ROOT"])
    n_comment = sum(len(pa.comments) for pa in merged.values())
    n_hl = sum(len(pa.highlights) for pa in merged.values())
    n_note = sum(len(pa.notes) for pa in merged.values())
    log.info(
        "annotated papers: %d | %d comments, %d highlights, %d notes",
        len(merged),
        n_comment,
        n_hl,
        n_note,
    )

    missing = unmirrored_with_annotations(root, merged)
    if missing:
        log.warning(
            "%d annotated papers are not mirrored yet (skipped): %s",
            len(missing),
            ", ".join(missing[:5]) + ("..." if len(missing) > 5 else ""),
        )

    if args.dry_run:
        for ck, pa in sorted(merged.items()):
            log.info("  %s", pa.summary())
        return

    written = capture_annotations(root, merged)
    log.info("wrote annotations for %d papers", len(written))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    main()
