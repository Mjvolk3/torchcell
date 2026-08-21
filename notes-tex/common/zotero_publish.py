#!/usr/bin/env python
# notes-tex/common/zotero_publish.py
# [[notes-tex.common.zotero_publish]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/common/zotero_publish.py
"""Publish a built notes-tex PDF into the personal Zotero library for review.

Why Zotero and not a shared folder: the point is *annotation*. Zotero's reader
keeps highlights and comments attached to a specific PDF attachment, so a comment
always refers to the exact rendering it was made on. That is the property we want
from a review copy -- a note saying "this number is wrong" is meaningless unless
you can tell which build it was made against.

Layout, which is DERIVED from the repo path rather than invented::

    repo    notes-tex/<doc>/main.pdf
    zotero  torchcell / notes-tex          <- every document, one flat index
            torchcell / notes-tex / <doc>  <- that document and its versions

so there is never a question of which Zotero collection a document belongs in.
The topic collections (``torchcell / torchcell-topics / *``) hold OTHER people's
papers; this tree holds ours. Keeping the two apart is what stops a draft from
being mistaken for literature when the bibliography is rebuilt.

The document is filed in BOTH collections, deliberately. Zotero does not show a
sub-collection's items in its parent unless the reader has turned on View ->
Show Items from Subcollections, so filing only into ``notes-tex/<doc>`` makes
``notes-tex`` look empty -- which is exactly what happened the first time this
ran. An item may belong to any number of collections and its file is stored once,
so the cost is nil and the top level becomes a usable index of every typeset
note.

Versioning. Each publish creates one child attachment named::

    <doc>_<YYYY-MM-DD-HH-MM-SS>_<sha256[:8]>.pdf

Both halves earn their place. The timestamp is what a human sorts by; the hash is
what actually identifies the bytes, and it is the same identifier the repo's
provenance rule uses everywhere else -- the stored artifact plus its sha256 is
canonical. Given both, a comment on "the 2026-08-21 build" can be traced to an
exact file, and re-running with unchanged content is detected as a duplicate
rather than piling up near-identical PDFs.

Dedupe by hash only works because the build is reproducible, and it was not until
this script needed it: hyperref writes a wall-clock ``/CreationDate`` into a
compressed object stream, so two builds of identical sources used to differ. The
fix is ``SOURCE_DATE_EPOCH`` in ``Makefile.common``, pinned to the newest source
mtime. Measured, not assumed: two forced rebuilds now produce the same sha256.

So a re-run of ``make`` with nothing edited is recognised as the same version and
skipped, while any real edit produces a new hash. Do not "simplify" that Makefile
line away -- versioning here rests on it.

Provenance is written into the parent item's ``extra`` field: the git commit and
branch the build came from. A reviewer who finds a problem can go from the Zotero
item to the exact source state without asking anyone.

Credentials via repo-root ``.env`` (``ZOTERO_USER_ID``, ``ZOTERO_API_KEY``). The
key needs user-library write and file access; ``--dry-run`` needs neither beyond
read.

Usage::

    python notes-tex/common/zotero_publish.py microbe-perturb-seq --dry-run
    python notes-tex/common/zotero_publish.py microbe-perturb-seq
    python notes-tex/common/zotero_publish.py microbe-perturb-seq --list
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import os.path as osp
import re
import shutil
import subprocess
import sys
import tempfile

from dotenv import load_dotenv
from pydantic import BaseModel
from pyzotero import zotero
from pyzotero.zotero import Zupload

# The document tree lives beside the topic tree, under the same `torchcell` root.
ROOT_COLLECTION = "torchcell"
DOCS_COLLECTION = "notes-tex"

# Marker written into the parent item's `extra`, and the key this script matches
# on when deciding whether a parent already exists. Matching on the title would
# break the moment someone edits the title in Zotero.
DOC_KEY_PREFIX = "Doc Key:"


class BuiltDoc(BaseModel):
    """A built notes-tex PDF and everything needed to identify it later."""

    doc: str  # directory name under notes-tex/, e.g. "microbe-perturb-seq"
    pdf_path: str
    title: str
    subtitle: str | None
    author: str
    sha256: str
    n_bytes: int
    git_commit: str
    git_branch: str
    built_at: str  # YYYY-MM-DD-HH-MM-SS, local

    @property
    def doc_key(self) -> str:
        return f"notes-tex/{self.doc}"

    @property
    def filename(self) -> str:
        return f"{self.doc}_{self.built_at}_{self.sha256[:8]}.pdf"

    @property
    def full_title(self) -> str:
        return f"{self.title}: {self.subtitle}" if self.subtitle else self.title

    def extra_field(self) -> str:
        return "\n".join(
            [
                f"{DOC_KEY_PREFIX} {self.doc_key}",
                f"Git Commit: {self.git_commit}",
                f"Git Branch: {self.git_branch}",
            ]
        )


def _git(repo: str, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", repo, *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _parse_title(tex_path: str) -> tuple[str, str | None, str]:
    """Pull title / subtitle / author out of main.tex.

    Read from the source rather than passed on the command line: a title typed
    into a flag drifts from the one on the page, and then the Zotero item names a
    document that no longer exists under that name.
    """
    src = open(tex_path, encoding="utf-8").read()

    def _strip(s: str) -> str:
        s = re.sub(r"\\vspace\{[^}]*\}", "", s)
        s = re.sub(r"\{\\large\s*", "", s)
        s = s.replace("\\\\", "\x00").replace("}", "").replace("{", "")
        # Collapse whitespace PER LINE. Doing it across the whole string eats the
        # title/subtitle break and silently yields one run-on title.
        return "\n".join(" ".join(part.split()) for part in s.split("\x00"))

    m = re.search(r"\\title\{(.+?)\n?\}\s*\n\\author", src, re.S)
    if not m:
        m = re.search(r"\\title\{(.+?)\}\s*$", src, re.M | re.S)
    if not m:
        sys.exit(f"no \\title{{}} found in {tex_path}")
    lines = [ln.strip() for ln in _strip(m.group(1)).split("\n") if ln.strip()]
    title = lines[0]
    subtitle = lines[1] if len(lines) > 1 else None

    a = re.search(r"\\author\{([^}]*)\}", src)
    return title, subtitle, (a.group(1).strip() if a else "")


def load_built_doc(notes_tex_dir: str, doc: str) -> BuiltDoc:
    doc_dir = osp.join(notes_tex_dir, doc)
    pdf = osp.join(doc_dir, "main.pdf")
    tex = osp.join(doc_dir, "main.tex")
    if not osp.exists(pdf):
        sys.exit(f"{pdf} does not exist -- run `make` in {doc_dir} first.")

    raw = open(pdf, "rb").read()
    title, subtitle, author = _parse_title(tex)
    repo = _git(doc_dir, "rev-parse", "--show-toplevel")
    dirty = _git(repo, "status", "--porcelain") != ""
    commit = _git(repo, "rev-parse", "HEAD")[:12] + ("-dirty" if dirty else "")

    return BuiltDoc(
        doc=doc,
        pdf_path=pdf,
        title=title,
        subtitle=subtitle,
        author=author,
        sha256=hashlib.sha256(raw).hexdigest(),
        n_bytes=len(raw),
        git_commit=commit,
        git_branch=_git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        built_at=datetime.datetime.fromtimestamp(osp.getmtime(pdf)).strftime(
            "%Y-%m-%d-%H-%M-%S"
        ),
    )


def find_collection(zot: zotero.Zotero, name: str, parent: str | None) -> str | None:
    """Find a collection by name under a given parent (None = top level)."""
    for c in zot.everything(zot.collections()):
        d = c["data"]
        if d["name"] == name and (d.get("parentCollection") or None) == parent:
            return c["key"]
    return None


def ensure_collection(
    zot: zotero.Zotero, name: str, parent: str | None, dry: bool
) -> str | None:
    key = find_collection(zot, name, parent)
    if key:
        return key
    if dry:
        print(f"  [dry-run] would create collection {name!r}")
        return None
    payload: dict = {"name": name}
    if parent:
        payload["parentCollection"] = parent
    resp = zot.create_collections([payload])
    key = resp["successful"]["0"]["key"]
    print(f"  created collection {name!r} ({key})")
    return key


def find_parent_item(zot: zotero.Zotero, coll: str, doc_key: str) -> dict | None:
    for it in zot.everything(zot.collection_items_top(coll)):
        if f"{DOC_KEY_PREFIX} {doc_key}" in it["data"].get("extra", ""):
            return it
    return None


def existing_hashes(zot: zotero.Zotero, parent_key: str) -> dict[str, str]:
    """sha8 -> attachment filename, read back off the child attachments.

    Taken from the FILENAME rather than by downloading each attachment and
    hashing it. The filename is written by this script and is the only thing that
    survives into the Zotero UI, so if the two ever disagree the filename is what
    a human would go by.
    """
    out = {}
    for ch in zot.children(parent_key):
        d = ch["data"]
        fn = d.get("filename") or d.get("title") or ""
        m = re.search(r"_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})_([0-9a-f]{8})\.pdf$", fn)
        if m:
            out[m.group(2)] = fn
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("doc", help="directory name under notes-tex/, e.g. microbe-perturb-seq")
    ap.add_argument("--dry-run", action="store_true", help="preview, upload nothing")
    ap.add_argument("--list", action="store_true", help="list published versions and exit")
    args = ap.parse_args()

    notes_tex_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    load_dotenv(osp.join(osp.dirname(notes_tex_dir), ".env"))
    load_dotenv()

    user_id, api_key = os.getenv("ZOTERO_USER_ID"), os.getenv("ZOTERO_API_KEY")
    if not (user_id and api_key):
        sys.exit("Set ZOTERO_USER_ID and ZOTERO_API_KEY in repo-root .env.")
    zot = zotero.Zotero(user_id, "user", api_key)

    built = load_built_doc(notes_tex_dir, args.doc)
    print(f"{built.full_title}")
    print(f"  {built.pdf_path}")
    print(f"  {built.n_bytes:,} bytes  sha256 {built.sha256}")
    print(f"  built {built.built_at}  from {built.git_branch} @ {built.git_commit}")
    print(f"  -> {built.filename}\n")

    root = find_collection(zot, ROOT_COLLECTION, None)
    if not root:
        sys.exit(f"no top-level {ROOT_COLLECTION!r} collection in the personal library.")
    docs = ensure_collection(zot, DOCS_COLLECTION, root, args.dry_run)
    coll = ensure_collection(zot, built.doc, docs, args.dry_run) if docs else None
    if coll:
        print(f"  collection {ROOT_COLLECTION}/{DOCS_COLLECTION}/{built.doc} = {coll}")

    parent = find_parent_item(zot, coll, built.doc_key) if coll else None

    if args.list:
        if not parent:
            print("\nnothing published yet.")
            return
        print(f"\npublished versions of {built.doc}:")
        for sha8, fn in sorted(existing_hashes(zot, parent["key"]).items(), key=lambda x: x[1]):
            here = "  <- current build" if sha8 == built.sha256[:8] else ""
            print(f"  {fn}{here}")
        return

    # --- parent item ---------------------------------------------------------
    # itemType `report`: it is our own document with an author and a date, not a
    # journal article and not a bare file. A standalone attachment would work but
    # gives the collection a list of filenames instead of a titled document with
    # its versions underneath.
    if parent:
        print(f"  parent item exists: {parent['key']}")
        # Backfill for items created before the top-level filing existed, and a
        # cheap self-heal if someone drags the item out of one collection.
        item = parent if parent.get("data") else zot.item(parent["key"])
        cols = set(item["data"].get("collections") or [])
        if not {coll, docs} <= cols and not args.dry_run:
            item["data"]["collections"] = sorted(cols | {coll, docs})
            zot.update_item(item)
            print(f"  filed into torchcell/{DOCS_COLLECTION} as well")
    elif args.dry_run:
        print("  [dry-run] would create parent report item")
    else:
        tmpl = zot.item_template("report")
        tmpl["title"] = built.full_title
        tmpl["creators"] = [
            {
                "creatorType": "author",
                "firstName": built.author.split()[0],
                "lastName": built.author.split()[-1],
            }
        ]
        tmpl["reportType"] = "Internal document"
        tmpl["institution"] = "University of Illinois Urbana-Champaign"
        tmpl["date"] = built.built_at[:10]
        tmpl["extra"] = built.extra_field()
        # Both: the per-document collection AND the flat notes-tex index.
        tmpl["collections"] = [coll, docs]
        resp = zot.create_items([tmpl])
        parent = {"key": resp["successful"]["0"]["key"]}
        print(f"  created parent item {parent['key']}")

    if args.dry_run:
        print("\n[dry-run] nothing uploaded.")
        return

    # --- version attachment --------------------------------------------------
    have = existing_hashes(zot, parent["key"])
    if built.sha256[:8] in have:
        print(f"\nalready published as {have[built.sha256[:8]]} -- identical bytes, nothing to do.")
        return

    # Upload from a temp copy so the versioned name is what lands in Zotero
    # storage rather than a generic "main.pdf".
    #
    # NOT pyzotero's attachment_simple: it puts the path it is handed into the
    # item's `filename` field, and the API rejects that outright --
    # "Stored-file filename '...' cannot contain a directory path". Zupload's
    # basedir is the supported way to say "this directory, that bare filename".
    with tempfile.TemporaryDirectory() as td:
        shutil.copy2(built.pdf_path, osp.join(td, built.filename))
        tmpl = {
            "itemType": "attachment",
            "linkMode": "imported_file",
            "title": built.filename,
            "filename": built.filename,
            "contentType": "application/pdf",
            "charset": "",
            "note": "",
            "tags": [],
            "relations": {},
        }
        resp = Zupload(zot, [tmpl], parent["key"], basedir=td).upload()

    if not (resp.get("success") or resp.get("unchanged")):
        sys.exit(f"upload failed: {resp}")
    print(f"\nuploaded {built.filename}")

    # Keep the parent's provenance pointing at the newest build, so the item a
    # reviewer opens names the commit the top attachment came from.
    if parent.get("data"):
        item = parent
    else:
        item = zot.item(parent["key"])
    item["data"]["extra"] = built.extra_field()
    item["data"]["date"] = built.built_at[:10]
    zot.update_item(item)
    print(f"  parent provenance updated -> {built.git_branch} @ {built.git_commit}")
    print(f"  {len(have) + 1} version(s) now in the collection")


if __name__ == "__main__":
    main()
