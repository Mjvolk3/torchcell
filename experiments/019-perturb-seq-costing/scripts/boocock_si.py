# experiments/019-perturb-seq-costing/scripts/boocock_si.py
# [[experiments.019-perturb-seq-costing.scripts.boocock_si]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-perturb-seq-costing/scripts/boocock_si
"""Retrieve and reduce Boocock et al. 2025 Supplementary file 1, Table S4.

Why this script exists. The Fig. 2 row for this study was taken from a review's
summary table and carried two defects: a UMI value of 4036 that appears in no
source (it is the arithmetic midpoint of the review's quoted 1514-6559 range),
and a range that was being drawn on a per-cell axis when it is actually a range
*across experiments*. Table S4 of the primary paper resolves both, so it is
fetched, hash-pinned, and reduced here rather than being read off by eye.

Provenance, per the repo rule that the stored artifact + its sha256 is canonical
and the URL is retrieval metadata:

    source_url        https://cdn.elifesciences.org/articles/95566/elife-95566-supp1-v1.xlsx
    retrieval_method  direct_url  (eLife CDN is scriptable; no auth, no JS)
    retrieval_command curl -sL <source_url> -o elife-95566-supp1-v1.xlsx
    sha256            7a0d8104d0534eaa4250fc7ec000abaa52c0ee13ba2492a95b261cf87ee1c555
    retrieved_at      2026-08-20
    discovered_via    https://api.elifesciences.org/articles/95566  (supp1 entry)

What the reduction establishes, none of which was previously sourced:

* The review's "100,220 in total" is the SUM of n_cells over all seven Table S4
  datasets (four eQTL + three ASE). It is not a figure the paper states in prose
  -- the abstract says only "over 100,000 single cells from three crosses" -- so
  the exact digits are now primary-sourced for the first time.
* The review's "1514-6559 UMIs" is the MIN and MAX of ``median_umi_per_cell``
  across those same seven datasets. It is a between-experiment range, confirming
  that plotting it as a cell-to-cell spread was wrong.
* The single value to plot is the CELL-WEIGHTED mean of the per-experiment
  medians. Weighted, not plain, because the x-axis of Fig. 2 is total cells
  profiled: an unweighted mean would let the 2,864-cell ASE run count as much as
  the 44,784-cell eQTL run, and the point would no longer describe the 100,220
  cells the x-coordinate claims. This is a derivation from primary data by a
  stated rule, which is what the repo asks for when a paper reports a range and
  no single representative value.

Run:  python experiments/019-perturb-seq-costing/scripts/boocock_si.py
      python experiments/019-perturb-seq-costing/scripts/boocock_si.py --fetch

Without --fetch it reads the already-downloaded workbook from RAW_DIR and fails
loudly if it is absent or its hash does not match. There is no silent re-download
and no fallback: a hash mismatch means upstream changed and needs a NEW versioned
provenance record, not a quiet overwrite.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import os.path as osp
import subprocess
import sys

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "019-perturb-seq-costing", "results")
# The workbook is a 1 MB binary that belongs in the sha256-pinned raw mirror, not
# in git. DATA_ROOT is the machine-local large-file root; the mirror proper lives
# on the tc-lit host, and depositing there is a separate step (see MIRROR_TODO).
RAW_DIR = osp.join(
    os.environ["DATA_ROOT"], "raw", "boocockSinglecellEQTLMapping2025"
)

CITATION_KEY = "boocockSinglecellEQTLMapping2025"
SOURCE_URL = (
    "https://cdn.elifesciences.org/articles/95566/elife-95566-supp1-v1.xlsx"
)
FILENAME = "elife-95566-supp1-v1.xlsx"
SHA256 = "7a0d8104d0534eaa4250fc7ec000abaa52c0ee13ba2492a95b261cf87ee1c555"
SHEET = "Table S4. Dataset Summary"

MIRROR_TODO = """\
NOT YET DEPOSITED IN THE MIRROR. tc-lit serves the torchcell-library mirror from
the gilahyper host read-only over HTTP, so this file cannot be deposited from a
Mac session. On gilahyper, run this script with --fetch, then copy the verified
workbook to $DATA_ROOT/torchcell-library/boocockSinglecellEQTLMapping2025/si/
and add it to that key's manifest.json with the sha256 above. Until then the
values below are reproducible from the recorded retrieval command but are not
backed by our own mirror."""


class DatasetRow(BaseModel):
    """One row of Table S4: a single sequencing dataset, not a single cell."""

    cross: str
    dataset_type: str  # eQTL | ASE
    n_cells: int
    median_umi_per_cell: int
    n_transcripts_per_cell: int


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch() -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    dest = osp.join(RAW_DIR, FILENAME)
    # The exact command recorded in the provenance block above, run verbatim.
    subprocess.run(["curl", "-sL", SOURCE_URL, "-o", dest], check=True)
    return dest


def load() -> str:
    """Return the path to the verified workbook, or fail with the reason."""
    dest = osp.join(RAW_DIR, FILENAME)
    if not osp.exists(dest):
        raise SystemExit(
            f"missing {dest}\n  run with --fetch to retrieve it from {SOURCE_URL}"
        )
    got = sha256_of(dest)
    if got != SHA256:
        raise SystemExit(
            f"sha256 MISMATCH for {dest}\n"
            f"  expected {SHA256}\n  got      {got}\n"
            "  upstream content changed; create a NEW versioned provenance "
            "record rather than overwriting this one"
        )
    return dest


def table_s4(path: str) -> list[DatasetRow]:
    """Parse Table S4 into typed rows.

    The sheet carries a two-line title in row 0 and the real header in row 1, so
    the header is taken explicitly rather than inferred; blank and footnote rows
    at the bottom are dropped by requiring n_cells to be present.
    """
    raw = pd.read_excel(path, sheet_name=SHEET, header=1)
    raw = raw[raw["n_cells"].notna()]
    return [
        DatasetRow(
            # "A*" is the pilot set of previously genotyped segregants; the
            # asterisk is a footnote marker in the sheet, kept verbatim.
            cross=str(r["cross"]),
            dataset_type=str(r["type"]),
            n_cells=int(r["n_cells"]),
            median_umi_per_cell=int(r["median_umi_per_cell"]),
            n_transcripts_per_cell=int(r["n_transcripts_per_cell"]),
        )
        for _, r in raw.iterrows()
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fetch", action="store_true",
                    help="download the workbook before verifying it")
    a = ap.parse_args()

    if a.fetch:
        print(f"fetching {SOURCE_URL}")
        fetch()

    path = load()
    print(f"verified {path}\n  sha256 {SHA256}")

    rows = table_s4(path)
    df = pd.DataFrame([r.model_dump() for r in rows])

    total_cells = int(df.n_cells.sum())
    lo = int(df.median_umi_per_cell.min())
    hi = int(df.median_umi_per_cell.max())
    weighted = float((df.n_cells * df.median_umi_per_cell).sum() / total_cells)

    os.makedirs(RESULTS, exist_ok=True)
    out = osp.join(RESULTS, "boocock2025_table_s4.csv")
    df.to_csv(out, index=False)

    print(f"\n{df.to_string(index=False)}\n")
    print(f"total cells (sum n_cells)          {total_cells:,}")
    print(f"per-experiment median UMI range    {lo:,}--{hi:,}")
    print(f"cell-weighted mean of medians      {weighted:,.0f}")
    print(f"\nwrote {out}")
    print(f"\n{MIRROR_TODO}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
