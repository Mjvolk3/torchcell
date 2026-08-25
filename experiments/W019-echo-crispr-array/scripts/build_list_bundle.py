# experiments/W019-echo-crispr-array/scripts/build_list_bundle.py
# [[experiments.W019-echo-crispr-array.scripts.build_list_bundle]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/W019-echo-crispr-array/scripts/build_list_bundle
"""Bundle the W019 bench sheet for handoff: the share-view PDF plus its CSVs.

The document goes to Zotero for review, but a reviewed PDF is not what someone
doing the bench work needs. They need the same tables as files they can sort,
filter and paste into a picklist, and they need to know the two forms agree.
This produces one zip holding both, plus a manifest that says where every file
came from.

What is inside, and why exactly this:

    w019-strain-build-list.pdf   the CLEAN share view, no status chips
    csv/t1-existing-singles.csv  the 12 singles in hand, ours vs published
    csv/t2-existing-doubles.csv  the 13 doubles in hand
    csv/t3-new-doubles.csv       D01-D25, the doubles to construct
    csv/t4-new-triples.csv       T01-T20, the triples to construct
    csv/t5-plate.csv             what goes on the measurement plate
    MANIFEST.txt                 per-file sha256, sizes, and the source commit

The PDF is `main-clean.pdf` rather than `main.pdf` because the draft view carries
section-status chips, which are an internal editing signal and say nothing to a
reader outside the group.

The CSVs are copied, never re-derived. `build_list_tables.py` renders each table
ONCE into both .tex and .csv from a single list of records, which is what makes
the typeset table and the data file unable to disagree; regenerating here would
open exactly the gap that design closes. Run that script first if either form is
stale -- this one refuses on a missing input rather than building a partial zip.

Determinism: every member is stamped with a fixed mtime (the epoch below) and the
archive is written with fixed compression, so rebuilding an unchanged bundle
reproduces the same bytes and therefore the same sha256. That is what lets the
manifest's own hash identify a handoff, the same rule the rest of the repo uses
for provenance: the stored artifact plus its sha256 is canonical, not a filename.

The zip itself is NOT committed. Every byte in it is already tracked, so storing
the archive too would put a second, binary copy of the same content in the history
that goes stale the moment either half changes. It is rebuilt on demand instead,
which is also what keeps the manifest's commit line honest.

`--copy-to` drops a copy somewhere reachable, e.g. the scratch handoff directory,
so the bundle can be picked up without digging through the experiment tree. It is
a copy of the artifact and never the artifact itself: the one under `results/` is
what the manifest describes.

Run from repo root:
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/W019-echo-crispr-array/scripts/build_list_bundle.py \
        --copy-to /tmp/screenshots/W019-echo-crispr-array
    ... --out /path/to/somewhere/w019-strain-build-list.zip
"""

from __future__ import annotations

import argparse
import hashlib
import os
import os.path as osp
import shutil
import subprocess
import zipfile

from pydantic import BaseModel

EXP_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
REPO = osp.dirname(osp.dirname(EXP_DIR))
CSV_DIR = osp.join(EXP_DIR, "results", "build_list_tables")
DOC_DIR = osp.join(REPO, "notes-tex", "w019-strain-build-list")
PDF = osp.join(DOC_DIR, "main-clean.pdf")
DEFAULT_OUT = osp.join(EXP_DIR, "results", "w019-strain-build-list.zip")

# Zip entries carry a fixed timestamp so the archive hashes stably. 1980-01-01 is
# the zero of the zip format's DOS timestamp; anything earlier is unrepresentable.
FIXED_MTIME = (1980, 1, 1, 0, 0, 0)

# name in the zip -> (source path, what it is). Ordered as a reader meets them.
CSVS = [
    ("t1-existing-singles.csv", "the 12 single knockouts in hand, ours against Costanzo 2016"),
    ("t2-existing-doubles.csv", "the 13 double knockouts in hand"),
    ("t3-new-doubles.csv", "D01-D25, the doubles to construct"),
    ("t4-new-triples.csv", "T01-T20, the triples to construct"),
    ("t5-plate.csv", "what goes on the measurement plate, and the well budget"),
]


class Member(BaseModel):
    """One file placed in the bundle, with the identity it is placed under."""

    arcname: str
    source: str  # repo-relative, so the manifest says where to look
    note: str
    sha256: str
    n_bytes: int

    @classmethod
    def of(cls, path: str, arcname: str, note: str) -> Member:
        """Read a file and record it under the name it takes inside the zip."""
        raw = open(path, "rb").read()
        return cls(
            arcname=arcname,
            source=osp.relpath(path, REPO),
            note=note,
            sha256=hashlib.sha256(raw).hexdigest(),
            n_bytes=len(raw),
        )


def git_commit() -> str:
    """The commit the bundle was cut from, or a marker when that is unknowable."""
    head = subprocess.run(
        ["git", "-C", REPO, "rev-parse", "--short=12", "HEAD"],
        capture_output=True, text=True,
    )
    if head.returncode != 0:
        return "not-a-git-repo"
    dirty = subprocess.run(
        ["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True,
    ).stdout.strip()
    # A dirty tree is recorded, not tolerated silently: a bundle built from
    # uncommitted edits cannot be reproduced from the commit it names.
    return head.stdout.strip() + (" (dirty)" if dirty else "")


def manifest(members: list[Member], commit: str) -> str:
    lines = [
        "W019 Strain Build List -- handoff bundle",
        "",
        f"source commit: {commit}",
        "generated by:  experiments/W019-echo-crispr-array/scripts/build_list_bundle.py",
        "tables from:   experiments/W019-echo-crispr-array/scripts/build_list_tables.py",
        "",
        "The PDF is the share view of notes-tex/w019-strain-build-list. Each CSV is the",
        "data form of the table with the matching number, rendered from the same records",
        "as the typeset table, so the two cannot disagree.",
        "",
        "file                              bytes  sha256",
    ]
    for m in members:
        lines.append(f"{m.arcname:<30} {m.n_bytes:>8}  {m.sha256}")
    lines += ["", "what each file holds", ""]
    for m in members:
        lines.append(f"  {m.arcname}")
        lines.append(f"      {m.note}")
        lines.append(f"      from {m.source}")
    return "\n".join(lines) + "\n"


def build(out_path: str) -> str:
    missing = [p for p in [PDF] + [osp.join(CSV_DIR, n) for n, _ in CSVS]
               if not osp.exists(p)]
    if missing:
        raise SystemExit(
            "missing input(s):\n  " + "\n  ".join(osp.relpath(p, REPO) for p in missing)
            + "\nRun build_list_tables.py, then `make clean-view` in "
              "notes-tex/w019-strain-build-list."
        )

    members = [Member.of(PDF, "w019-strain-build-list.pdf",
                         "the bench sheet, share view (no status chips)")]
    members += [Member.of(osp.join(CSV_DIR, name), f"csv/{name}", note)
                for name, note in CSVS]

    text = manifest(members, git_commit())
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for m in members:
            info = zipfile.ZipInfo(m.arcname, date_time=FIXED_MTIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, open(osp.join(REPO, m.source), "rb").read())
        info = zipfile.ZipInfo("MANIFEST.txt", date_time=FIXED_MTIME)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        z.writestr(info, text)

    print(f"wrote {out_path}")
    print(f"  sha256 {hashlib.sha256(open(out_path, 'rb').read()).hexdigest()}")
    print(f"  {len(members) + 1} files, {osp.getsize(out_path):,} bytes")
    for m in members:
        print(f"    {m.arcname}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"zip path to write (default: {osp.relpath(DEFAULT_OUT, REPO)})")
    ap.add_argument("--copy-to", metavar="DIR",
                    help="also drop a copy of the zip in DIR, created if absent")
    args = ap.parse_args()
    out = build(args.out)
    if args.copy_to:
        os.makedirs(args.copy_to, exist_ok=True)
        dest = osp.join(args.copy_to, osp.basename(out))
        shutil.copy2(out, dest)
        print(f"  copied to {dest}")


if __name__ == "__main__":
    main()
