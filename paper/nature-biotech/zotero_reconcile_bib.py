"""
paper/nature-biotech/zotero_reconcile_bib.py

Make the manuscript, references.bib, and the shared group `paper` collection agree on
ONE set of pinned Better BibTeX keys — the real keys in your personal `torchcell` Zotero
collection. Goal: the group `paper` collection is a current, up-to-date match of what is
actually \\cite-d in the paper.

Read-only unless --apply. It:
  1. collects every \\cite key used in sections/*.tex,
  2. looks up each key's title in the current references.bib,
  3. matches that title to an item in the personal `torchcell` collection (ICDCVSL6) and
     reads its real *pinned* BBT citekey,
  4. prints the old -> new key map (+ anything unmatched).

With --apply it also:
  - rewrites the \\cite keys in sections/*.tex  (invented -> real BBT), and
  - copies the matched items into the group `paper` collection (ZOTERO_LIBRARY_ID).
Then run  zotero_export_bib.py  to regenerate references.bib from the group, and
`make paper` to confirm the \\cites resolve.

Env: ZOTERO_API_KEY, ZOTERO_USER_ID (personal), ZOTERO_LIBRARY_ID (group).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from pyzotero import zotero

HERE = Path(__file__).parent
SECTIONS = sorted((HERE / "sections").glob("*.tex"))
BIB = HERE / "references.bib"
SRC_COLLECTION = "ICDCVSL6"          # personal `torchcell` collection (from the web URL)
DST_COLLECTION_NAME = "paper"
_DROP = {"key", "version", "dateAdded", "dateModified", "collections", "relations", "citationKey", "inPublications"}
_CITE = re.compile(r"(\\cite[a-zA-Z]*\{)([^}]*)(\})")


def norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())


def manuscript_keys() -> set[str]:
    keys: set[str] = set()
    for tex in SECTIONS:
        for m in _CITE.finditer(tex.read_text()):
            keys.update(k.strip() for k in m.group(2).split(",") if k.strip())
    return keys


def bib_titles() -> dict[str, str]:
    out: dict[str, str] = {}
    key = None
    for line in BIB.read_text().splitlines():
        m = re.match(r"@\w+\{([^,]+),", line.strip())
        if m:
            key = m.group(1).strip()
            continue
        ls = line.strip()
        if key and ls.startswith("title"):
            val = ls.split("=", 1)[1].strip().rstrip(",").strip().strip("{}")
            out[key] = val.replace("{", "").replace("}", "")
            key = None
    return out


def citekey_of(data: dict) -> str | None:
    if data.get("citationKey"):
        return data["citationKey"]
    for line in data.get("extra", "").splitlines():
        if line.startswith("Citation Key:"):
            return line.split(":", 1)[1].strip()
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Reconcile manuscript \\cites to pinned BBT keys.")
    ap.add_argument("--apply", action="store_true", help="rewrite \\cites + copy items into group `paper`")
    args = ap.parse_args()

    load_dotenv()
    api = os.getenv("ZOTERO_API_KEY")
    z = zotero.Zotero(os.getenv("ZOTERO_USER_ID"), "user", api)

    by_title = {norm(it["data"].get("title", "")): it
                for it in z.everything(z.top())
                if it["data"].get("itemType") not in ("attachment", "note")}
    titles = bib_titles()
    cites = manuscript_keys()

    mapping: dict[str, tuple[str, dict]] = {}
    unmatched: list[str] = []
    for k in sorted(cites):
        title = titles.get(k, "")
        it = by_title.get(norm(title)) if title else None
        rk = citekey_of(it["data"]) if it else None
        if rk:
            mapping[k] = (rk, it)
        else:
            unmatched.append(k)

    print(f"\\cite keys in manuscript: {len(cites)} | matched: {len(mapping)} | unmatched: {len(unmatched)}\n")
    for k, (rk, _) in sorted(mapping.items()):
        print(f"  {k:34} -> {rk}" + ("" if k == rk else "   <-- rewrite"))
    if unmatched:
        print("\nUNMATCHED (add to torchcell collection, or fix the title):")
        for k in unmatched:
            print(f"  {k:34} title={titles.get(k, '(not in references.bib)')[:55]!r}")

    if not args.apply:
        print("\n[read-only] re-run with --apply to rewrite \\cites and copy matched items into group `paper`.")
        return

    remap = {k: rk for k, (rk, _) in mapping.items() if k != rk}
    for tex in SECTIONS:
        txt = tex.read_text()
        new = _CITE.sub(lambda m: m.group(1) + ",".join(remap.get(x.strip(), x.strip())
                        for x in m.group(2).split(",")) + m.group(3), txt)
        if new != txt:
            tex.write_text(new)
            print(f"rewrote \\cites in {tex.name}")

    group = os.getenv("ZOTERO_LIBRARY_ID")
    zg = zotero.Zotero(group, "group", api)
    coll_key = next((c["key"] for c in zg.everything(zg.collections())
                     if c["data"]["name"].lower() == DST_COLLECTION_NAME), None)
    if not coll_key:
        sys.exit(f"no `{DST_COLLECTION_NAME}` collection in group {group}")
    present = {citekey_of(it["data"]) for it in zg.everything(zg.collection_items_top(coll_key))}
    payload = []
    for rk, it in {rk: it for _, (rk, it) in mapping.items()}.items():
        if rk in present:
            continue
        d = {kk: vv for kk, vv in it["data"].items() if kk not in _DROP}
        d["collections"] = [coll_key]
        # carry the pin as a BBT-recognisable "Citation Key:" line so the group re-pins it
        extra = d.get("extra", "")
        if f"Citation Key: {rk}" not in extra:
            d["extra"] = (f"Citation Key: {rk}\n" + extra).strip()
        payload.append(d)
    created = 0
    for i in range(0, len(payload), 50):
        resp = zg.create_items(payload[i:i + 50])
        created += len(resp.get("successful", {}))
        if resp.get("failed"):
            print("FAILED:", resp["failed"])
    print(f"copied {created} items into group `paper`. Next: zotero_export_bib.py, then make paper.")


if __name__ == "__main__":
    main()
