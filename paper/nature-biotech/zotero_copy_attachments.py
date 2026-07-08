"""
paper/nature-biotech/zotero_copy_attachments.py

Companion to `zotero_copy_refs.py`. That script copies bibliographic *metadata*
from the personal ("main") library into the shared **torchcell** group library but
carries no files. This script fills the gap: for each item already sitting in a
target group collection, it finds the citekey-matched item in the personal library
and copies that item's stored **PDF attachments** into the group item.

Why a separate step: in Zotero a PDF is a *child attachment item* with its own file
bytes in storage, not a field on the parent record -- so copying it is a download +
re-upload, not a metadata clone.

Matching (group item -> main item), first hit wins:
  1. exact BetterBibTeX pinned citekey (native `citationKey` or `Citation Key:` in extra)
  2. DOI (case-insensitive)
  3. normalized title (lowercased, alphanumerics only)
Step 2/3 catch BBT's disambiguation drift, where the group copy is `foo2022` and the
main copy is `foo2022a`.

Only main attachments that are (a) itemType attachment, (b) contentType
application/pdf, (c) imported_file / imported_url, and (d) have a stored file (md5)
are copied. An item is skipped if the group already has a PDF with the same filename,
so re-runs are idempotent.

Credentials via env (repo-root .env, load_dotenv):
  ZOTERO_API_KEY     - key with WRITE access to the group
  ZOTERO_USER_ID     - your personal (main) library id
  ZOTERO_LIBRARY_ID  - target group id

CAUTION: group-library files count against the group *owner's* storage quota.

Usage (run it yourself, not CI). Preview first:
    python paper/nature-biotech/zotero_copy_attachments.py --dry-run
    python paper/nature-biotech/zotero_copy_attachments.py                    # collection 'paper'
    python paper/nature-biotech/zotero_copy_attachments.py --collection database
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile

from dotenv import load_dotenv
from pyzotero import zotero


def citekey_of(item: dict) -> str | None:
    d = item.get("data", {})
    if d.get("citationKey"):
        return d["citationKey"]
    for line in d.get("extra", "").splitlines():
        if line.startswith("Citation Key:"):
            return line.split(":", 1)[1].strip()
    return None


def norm_title(item: dict) -> str | None:
    t = item.get("data", {}).get("title")
    if not t:
        return None
    key = re.sub(r"[^a-z0-9]", "", t.lower())
    return key or None


def doi_of(item: dict) -> str | None:
    doi = item.get("data", {}).get("DOI")
    return doi.strip().lower() if doi else None


def stored_pdfs(zot: zotero.Zotero, parent_key: str) -> list[dict]:
    """Attachment children that are actual stored PDF files."""
    out = []
    for a in zot.children(parent_key):
        d = a["data"]
        if (
            d.get("itemType") == "attachment"
            and d.get("contentType") == "application/pdf"
            and d.get("linkMode") in ("imported_file", "imported_url")
            and d.get("md5")
        ):
            out.append(a)
    return out


def group_pdf_filenames(zot: zotero.Zotero, parent_key: str) -> set[str]:
    return {
        a["data"].get("filename")
        for a in zot.children(parent_key)
        if a["data"].get("itemType") == "attachment"
        and a["data"].get("contentType") == "application/pdf"
    }


def resolve_collection(zot_group: zotero.Zotero, name: str) -> str:
    for c in zot_group.everything(zot_group.collections()):
        if c["data"]["name"].lower() == name.lower():
            return c["key"]
    sys.exit(f"No collection {name!r} in the group.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Copy PDF attachments: main -> group collection items.")
    ap.add_argument("--collection", default="paper", help="exact target collection name (default: paper)")
    ap.add_argument("--dry-run", action="store_true", help="preview without uploading")
    args = ap.parse_args()

    load_dotenv(os.path.join(os.getcwd(), ".env"))
    api_key = os.getenv("ZOTERO_API_KEY")
    user_id = os.getenv("ZOTERO_USER_ID")
    group_id = os.getenv("ZOTERO_LIBRARY_ID")
    if not (api_key and user_id and group_id):
        sys.exit("Set ZOTERO_API_KEY, ZOTERO_USER_ID, ZOTERO_LIBRARY_ID in repo-root .env.")

    zot_user = zotero.Zotero(user_id, "user", api_key)
    zot_group = zotero.Zotero(group_id, "group", api_key)
    coll_key = resolve_collection(zot_group, args.collection)
    print(f"target: group {group_id} / collection {args.collection!r} ({coll_key})")

    # Index the personal library by citekey / DOI / normalized title (top-level items).
    print("indexing main library ...")
    by_ck, by_doi, by_title = {}, {}, {}
    for it in zot_user.everything(zot_user.top()):
        if it["data"].get("itemType") in ("attachment", "note"):
            continue
        if (ck := citekey_of(it)):
            by_ck.setdefault(ck, it)
        if (doi := doi_of(it)):
            by_doi.setdefault(doi, it)
        if (nt := norm_title(it)):
            by_title.setdefault(nt, it)
    print(f"  {len(by_ck)} keyed items")

    def match_main(g_item: dict) -> tuple[dict | None, str]:
        ck = citekey_of(g_item)
        if ck and ck in by_ck:
            return by_ck[ck], "citekey"
        doi = doi_of(g_item)
        if doi and doi in by_doi:
            return by_doi[doi], "doi"
        nt = norm_title(g_item)
        if nt and nt in by_title:
            return by_title[nt], "title"
        return None, "none"

    g_items = zot_group.everything(zot_group.collection_items_top(coll_key))
    print(f"{len(g_items)} items in {args.collection!r}\n")

    plan = []          # (group_item, main_attachment, via)
    no_match, no_pdf, already = [], [], []
    for g in g_items:
        gck = citekey_of(g) or g["key"]
        main, via = match_main(g)
        if main is None:
            no_match.append(gck)
            continue
        m_pdfs = stored_pdfs(zot_user, main["key"])
        if not m_pdfs:
            no_pdf.append(gck)
            continue
        have = group_pdf_filenames(zot_group, g["key"])
        wanted = [a for a in m_pdfs if a["data"].get("filename") not in have]
        if not wanted:
            already.append(gck)
            continue
        for a in wanted:
            plan.append((g, a, via))

    print(f"== plan: {len(plan)} PDF(s) to copy ==")
    for g, a, via in plan:
        print(f"  [{via:7}] {citekey_of(g)}  <-  {a['data'].get('filename')}")
    if no_match:
        print(f"\n  no main match ({len(no_match)}): {', '.join(no_match)}")
    if no_pdf:
        print(f"  main has no stored PDF ({len(no_pdf)}): {', '.join(no_pdf)}")
    if already:
        print(f"  already has the PDF(s) ({len(already)}): {', '.join(already)}")

    if args.dry_run:
        print("\n[dry-run] nothing uploaded.")
        return
    if not plan:
        print("\nnothing to do.")
        return

    ok, fail = 0, 0
    with tempfile.TemporaryDirectory() as tmp:
        for g, a, _ in plan:
            fname = (a["data"].get("filename") or "attachment.pdf").replace("/", "_")
            path = os.path.join(tmp, fname)
            with open(path, "wb") as fh:
                fh.write(zot_user.file(a["key"]))          # download bytes from main
            resp = zot_group.attachment_simple([path], g["key"])  # create + upload to group
            good = resp.get("success", []) + resp.get("unchanged", [])
            if good and not resp.get("failure"):
                ok += 1
                print(f"  uploaded: {citekey_of(g)} / {fname}")
            else:
                fail += 1
                print(f"  FAILED:   {citekey_of(g)} / {fname} -> {resp.get('failure')}")
    print(f"\ndone: {ok} uploaded, {fail} failed.")


if __name__ == "__main__":
    main()
