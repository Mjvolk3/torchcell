#!/usr/bin/env python
# notes-tex/common/build_bib.py
# [[notes-tex.common.build_bib]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/common/build_bib.py
"""Build a notes-tex document's references.bib as the UNION of two Zotero libraries.

Two tiers, deliberately different, because they answer different questions:

* **The manuscript** (``paper/nature-biotech/references.bib``) is exported from the
  shared **group** ``paper`` collection ONLY, by ``zotero_export_bib.py``. That is a
  publication guarantee: every work cited in the paper lives in a library the whole
  group holds, so the citations stay recoverable after publication. This script
  does not touch it.

* **notes-tex documents** take the **union of the group and the personal**
  collection of the same name. Technical notes cite far more widely than the paper
  and much of that reading lands in the personal library first; requiring a
  promotion-to-group round trip before anything can be cited would just stop notes
  being written.

The union is where citekey collisions become possible, and the whole point of this
script is to make them loud and *searchable* rather than silently resolving them.
Two distinct failures are detected:

1. **Key conflict** -- one citekey, two different works. Fatal: whichever copy wins,
   some citation in the document now points at the wrong paper.
2. **Duplicate work** -- one work (same DOI or title) under two different citekeys.
   Not fatal, but it splits a reference into two numbered entries in the PDF.

Both are reported with the citekey, which library each side came from, the titles,
and the fields that differ, so the item can be found in Zotero without a hunt.

Precedence when a key is in both and the entries agree: the group copy wins, since
that is the curated one.

Usage
-----
    python build_bib.py --group-collection FE8DQKUH \\
                        --personal-collection AC8MFJXK \\
                        --out references.bib

    python build_bib.py ... --check      # report only, write nothing, exit 1 on conflict

Requires Zotero + Better BibTeX running (localhost:23119). Credentials are not
needed: BBT reads the local data store.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import urllib.error
import urllib.request

from dotenv import load_dotenv

BBT = "http://localhost:23119/better-bibtex/export/collection"
# BBT addresses the personal library as 1 and a group by its Zotero group id.
PERSONAL_LIBRARY_ID = "1"

RED, YEL, GRN, GRY, RST = "\033[31m", "\033[33m", "\033[32m", "\033[90m", "\033[0m"

# Fields compared when deciding whether two entries under one key are the same
# work. Deliberately narrow: BBT emits keywords, file paths and timestamps that
# differ between libraries for reasons that have nothing to do with the work.
IDENTITY_FIELDS = ("title", "doi", "year", "author", "journal", "booktitle")


def fetch(library_id: str, collection: str) -> str:
    url = f"{BBT}?/{library_id}/{collection}.bibtex"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        raise SystemExit(
            f"{RED}ERROR{RST} could not reach Better BibTeX at {BBT}\n"
            f"  {e}\n"
            "  Zotero must be running with the Better BibTeX plugin installed."
        )


def split_entries(bib: str) -> dict[str, str]:
    """Split a .bib into {citekey: raw entry text}, brace-balanced."""
    out: dict[str, str] = {}
    i = 0
    while True:
        m = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", re.S).search(bib, i)
        if not m:
            break
        if m.group(1).lower() in ("comment", "preamble", "string"):
            i = m.end()
            continue
        depth, j = 1, m.end()
        while j < len(bib) and depth:
            if bib[j] == "{":
                depth += 1
            elif bib[j] == "}":
                depth -= 1
            j += 1
        out[m.group(2)] = bib[m.start() : j]
        i = j
    return out


def fields(entry: str) -> dict[str, str]:
    """Very small field reader -- enough for the identity comparison."""
    out: dict[str, str] = {}
    for fm in re.finditer(r"(\w+)\s*=\s*", entry):
        k = fm.group(1).lower()
        j = fm.end()
        if j >= len(entry):
            continue
        if entry[j] == "{":
            depth, s = 1, j + 1
            j += 1
            while j < len(entry) and depth:
                if entry[j] == "{":
                    depth += 1
                elif entry[j] == "}":
                    depth -= 1
                j += 1
            out[k] = entry[s : j - 1]
        elif entry[j] == '"':
            e = entry.find('"', j + 1)
            out[k] = entry[j + 1 : e] if e > 0 else ""
    return out


def norm(s: str) -> str:
    """Normalise for comparison: case, whitespace, braces, punctuation."""
    s = re.sub(r"[{}\\]", "", s or "")
    s = re.sub(r"[^a-z0-9]+", " ", s.lower())
    return s.strip()


def identity(entry: str) -> dict[str, str]:
    f = fields(entry)
    return {k: norm(f.get(k, "")) for k in IDENTITY_FIELDS}


def title_of(entry: str) -> str:
    t = fields(entry).get("title", "")
    t = re.sub(r"[{}]", "", t)
    return (t[:78] + "...") if len(t) > 78 else t


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--group-collection", required=True)
    ap.add_argument("--personal-collection", default="")
    ap.add_argument("--group-id", default=os.getenv("ZOTERO_LIBRARY_ID", "6582362"))
    ap.add_argument("--out", default="references.bib")
    ap.add_argument("--check", action="store_true",
                    help="report only; write nothing; exit 1 on a key conflict")
    ap.add_argument("--min-entries", type=int, default=3,
                    help="refuse to write fewer than this (guards a stopped Zotero)")
    a = ap.parse_args()

    group = split_entries(fetch(a.group_id, a.group_collection))
    print(f"group    {a.group_id}/{a.group_collection}: {len(group)} entries")

    personal: dict[str, str] = {}
    if a.personal_collection:
        personal = split_entries(
            fetch(PERSONAL_LIBRARY_ID, a.personal_collection)
        )
        print(f"personal 1/{a.personal_collection}: {len(personal)} entries")

    findings: list[str] = []

    # --- 1. same citekey, different work ------------------------------------
    conflicts = []
    for k in sorted(set(group) & set(personal)):
        gi, pi = identity(group[k]), identity(personal[k])
        diff = [f for f in IDENTITY_FIELDS if gi[f] and pi[f] and gi[f] != pi[f]]
        if diff:
            conflicts.append((k, diff, title_of(group[k]), title_of(personal[k])))

    # --- 2. same work, different citekeys ------------------------------------
    merged_keys = set(group) | set(personal)
    by_doi: dict[str, list[str]] = {}
    by_title: dict[str, list[str]] = {}
    for k in sorted(merged_keys):
        e = group.get(k) or personal[k]
        f = fields(e)
        d = norm(f.get("doi", ""))
        t = norm(f.get("title", ""))
        if d:
            by_doi.setdefault(d, []).append(k)
        if t:
            by_title.setdefault(t, []).append(k)
    dupes = [(v, "DOI", d) for d, v in by_doi.items() if len(v) > 1]
    dupes += [
        (v, "title", t)
        for t, v in by_title.items()
        if len(v) > 1 and not any(set(v) == set(x[0]) for x in dupes)
    ]

    # --- report --------------------------------------------------------------
    def where(k: str) -> str:
        if k in group and k in personal:
            return "group+personal"
        return "group" if k in group else "personal"

    print()
    if conflicts:
        print(f"{RED}{'='*74}{RST}")
        print(f"{RED}KEY CONFLICT{RST}  one citekey, two different works")
        print(f"{RED}{'='*74}{RST}")
        for k, diff, gt, pt in conflicts:
            print(f"\n  citekey : {RED}{k}{RST}")
            print(f"  differs : {', '.join(diff)}")
            print(f"  {GRY}group   {RST}: {gt}")
            print(f"  {GRY}personal{RST}: {pt}")
            print(f"  {GRY}fix     : search Zotero for  {k}  in BOTH libraries and")
            print(f"            re-pin one citekey (BBT: right-click > Change key){RST}")
        findings.append(f"{len(conflicts)} key conflict(s)")

    if dupes:
        print(f"\n{YEL}{'='*74}{RST}")
        print(f"{YEL}DUPLICATE WORK{RST}  one work under two citekeys "
              f"(splits into two references)")
        print(f"{YEL}{'='*74}{RST}")
        for keys, how, val in dupes:
            print(f"\n  matched on {how}: {val[:70]}")
            for k in keys:
                print(f"    {YEL}{k}{RST}  [{where(k)}]  {title_of(group.get(k) or personal[k])}")
            print(f"  {GRY}fix: merge the Zotero items, or delete one from the collection{RST}")
        findings.append(f"{len(dupes)} duplicate work(s)")

    # --- merge ---------------------------------------------------------------
    # Group wins on a clean overlap: it is the curated, shared copy.
    merged = dict(personal)
    merged.update(group)
    overlap = sorted(set(group) & set(personal))
    only_personal = sorted(set(personal) - set(group))

    print(f"\n{'='*74}")
    print(f"union: {len(merged)} entries "
          f"({len(group)} group + {len(only_personal)} personal-only, "
          f"{len(overlap)} shared)")
    if only_personal:
        print(f"{GRY}personal-only (not yet promoted to the group library):{RST}")
        for k in only_personal:
            print(f"  {k}")

    if a.check:
        print()
        if conflicts:
            print(f"{RED}FAIL{RST}  {'; '.join(findings)}")
            return 1
        if dupes:
            print(f"{YEL}WARN{RST}  {'; '.join(findings)}")
            return 0
        print(f"{GRN}OK{RST}  no citekey conflicts or duplicates")
        return 0

    if conflicts:
        print(f"\n{RED}refusing to write {a.out}{RST} -- resolve the key conflict(s) first")
        return 1
    if len(merged) < a.min_entries:
        print(f"\n{RED}refusing to write{RST} -- only {len(merged)} entries")
        return 1

    header = (
        "% GENERATED by notes-tex/common/build_bib.py -- do not hand-edit.\n"
        f"% union of Zotero group {a.group_id}/{a.group_collection}"
        + (f" and personal 1/{a.personal_collection}\n" if a.personal_collection else "\n")
        + "% Regenerate with `make bib`. Only CITED entries reach the PDF.\n\n"
    )
    with open(a.out, "w") as fh:
        fh.write(header + "\n\n".join(merged[k] for k in sorted(merged)) + "\n")
    print(f"\n{GRN}wrote{RST} {a.out}  ({len(merged)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
