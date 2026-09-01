#!/usr/bin/env python
# notes-tex/common/zotero_comments.py
# [[notes-tex.common.zotero_comments]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/common/zotero_comments.py
"""Pull review comments off a published notes-tex PDF in Zotero.

The other half of ``zotero_publish.py``: that script sends a build out for
review, this one brings the review back.

**No mirror sync is involved, and none is needed.** The torchcell-library mirror
(``tc-lit``) serves OTHER people's papers; a notes-tex document is ours and lives
only in the personal Zotero library, whose annotations the Web API already
exposes. Annotations are child items of the ATTACHMENT, not of the parent item,
which is the one non-obvious part of the traversal::

    report item  (parent, one per document)
      └── attachment  (one per published version)
            └── annotation  x N   <- highlights, notes, comments

They are also per-version by construction: a comment is anchored to the exact PDF
it was made on. That is the property the whole versioning scheme exists to give,
and it is why this script reads ONE attachment rather than merging across them --
merging would silently reattribute a comment about page 8 of an old build to a
page 8 that has since changed.

Ordering is by ``annotationSortIndex``, which Zotero writes as
``page|offset|y``; splitting it into ints sorts reading-order rather than
lexically (``10`` before ``9`` otherwise).

Usage::

    python notes-tex/common/zotero_comments.py microbe-perturb-seq
    python notes-tex/common/zotero_comments.py microbe-perturb-seq --version 2
    python notes-tex/common/zotero_comments.py microbe-perturb-seq --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys

from dotenv import load_dotenv
from pydantic import BaseModel
from pyzotero import zotero

from zotero_publish import (
    DEFAULT_PARENT_DIR,
    ROOT_COLLECTION,
    find_collection,
    find_parent_item,
)


class Comment(BaseModel):
    """One reviewer annotation, flattened to what a revision actually needs."""

    key: str  # stable Zotero id -- the handle a revision ledger cites
    index: int  # 1-based reading order, for talking about "comment 37"
    page: str
    kind: str  # highlight | note | underline | image | ink
    color: str
    quote: str  # the highlighted text, empty for a standalone note
    comment: str  # what the reviewer wrote

    def as_markdown(self) -> str:
        head = f"### [{self.index}] p{self.page} `{self.key}`"
        body = []
        if self.quote:
            body.append(f"> {self.quote}")
        if self.comment:
            body.append(self.comment)
        return head + "\n\n" + "\n\n".join(body) + "\n"


def sort_key(ann: dict) -> list[int]:
    raw = ann["data"].get("annotationSortIndex", "")
    try:
        return [int(x) for x in raw.split("|")]
    except ValueError:
        return [0, 0, 0]


def fetch(zot: zotero.Zotero, doc: str, version: int | None) -> tuple[str, list[Comment]]:
    # Same resolution rule as zotero_publish.py since it generalized past
    # notes-tex: the collection path IS the repo-relative directory, and a bare
    # name is shorthand for notes-tex/<name>.
    rel_dir = doc.strip("/")
    if "/" not in rel_dir:
        rel_dir = f"{DEFAULT_PARENT_DIR}/{rel_dir}"
    coll = find_collection(zot, ROOT_COLLECTION, None)
    for name in rel_dir.split("/"):
        coll = find_collection(zot, name, coll)
        if not coll:
            sys.exit(f"no Zotero collection {ROOT_COLLECTION}/{rel_dir}")
    parent = find_parent_item(zot, coll, rel_dir)
    if not parent:
        sys.exit(f"nothing published for {doc} yet")

    # Attachments oldest-first, so --version 1 is the first build published.
    atts = sorted(
        (c for c in zot.children(parent["key"])
         if c["data"].get("itemType") == "attachment"),
        key=lambda c: c["data"].get("filename", ""),
    )
    if not atts:
        sys.exit("parent item has no attachments")
    att = atts[version - 1] if version else atts[-1]

    anns = [a for a in zot.everything(zot.children(att["key"]))
            if a["data"].get("itemType") == "annotation"]
    anns.sort(key=sort_key)
    out = []
    for i, a in enumerate(anns, start=1):
        d = a["data"]
        out.append(
            Comment(
                key=a["key"],
                index=i,
                page=d.get("annotationPageLabel", "?"),
                kind=d.get("annotationType", "?"),
                color=d.get("annotationColor", ""),
                quote=(d.get("annotationText") or "").strip(),
                comment=(d.get("annotationComment") or "").strip(),
            )
        )
    return att["data"].get("filename", att["key"]), out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("doc")
    ap.add_argument("--version", type=int, default=None,
                    help="1-based published version; default is the newest")
    ap.add_argument("--json", metavar="PATH", help="also write raw JSON here")
    args = ap.parse_args()

    notes_tex_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    load_dotenv(osp.join(osp.dirname(notes_tex_dir), ".env"))
    load_dotenv()
    user_id, api_key = os.getenv("ZOTERO_USER_ID"), os.getenv("ZOTERO_API_KEY")
    if not (user_id and api_key):
        sys.exit("Set ZOTERO_USER_ID and ZOTERO_API_KEY in repo-root .env.")

    zot = zotero.Zotero(user_id, "user", api_key)
    filename, comments = fetch(zot, args.doc, args.version)

    n_written = sum(1 for c in comments if c.comment)
    print(f"# Review comments on {filename}\n")
    print(f"{len(comments)} annotations, {n_written} carrying a written comment. "
          f"Colours are the reviewer's own; the key is the stable handle to cite "
          f"when recording what was done about each.\n")
    for c in comments:
        print(c.as_markdown())

    if args.json:
        json.dump([c.model_dump() for c in comments], open(args.json, "w"), indent=1)
        print(f"\nwrote {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
