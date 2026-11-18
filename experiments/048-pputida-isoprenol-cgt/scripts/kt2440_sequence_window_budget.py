# experiments/048-pputida-isoprenol-cgt/scripts/kt2440_sequence_window_budget.py
# [[experiments.048-pputida-isoprenol-cgt.scripts.kt2440_sequence_window_budget]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/048-pputida-isoprenol-cgt/scripts/kt2440_sequence_window_budget
"""How much clear upstream sequence does a KT2440 gene actually have?

The yeast embedding datasets use fixed flanking windows (`nt_window_five_prime_1003`,
`nt_window_three_prime_300`). Whether that transfers to a compact bacterial genome
is an empirical question, not a convention, so this measures it: the distance from
each gene's 5' end to the nearest edge of its upstream neighbor, on the pinned
`pputida_KT2440_ASM756v2` GFF.

If the median gap is far below the window, a fixed upstream window is not a
promoter window; it is mostly the neighboring gene's coding sequence, and the
sequence unit for this organism should be the ORF instead.

Writes JSON + CSV into experiments/048-pputida-isoprenol-cgt/results/.
"""

import glob
import gzip
import json
import os
import os.path as osp
import re

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

RESULTS = osp.join(EXPERIMENT_ROOT, "048-pputida-isoprenol-cgt", "results")
GENOME_DIR = osp.join(DATA_ROOT, "torchcell-genomes", "pputida_KT2440_ASM756v2")

# The windows the yeast embedding datasets use, plus the round numbers worth
# checking for a bacterium.
WINDOWS = (100, 200, 300, 500, 1000, 1003, 2000)


def pinned_gff_path() -> str:
    """The GenBank-assembly GFF from the pinned genome set (PP_ locus tags)."""
    candidates = sorted(glob.glob(osp.join(GENOME_DIR, "*genomic.gff*")))
    if not candidates:
        raise FileNotFoundError(f"no genomic GFF under {GENOME_DIR}")
    gca = [c for c in candidates if "GCA" in osp.basename(c)]
    return gca[0] if gca else candidates[0]


def read_genes(path: str) -> pd.DataFrame:
    opener = gzip.open if path.endswith(".gz") else open
    rows = []
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            match = re.search(r"locus_tag=([^;]+)", fields[8])
            rows.append(
                {
                    "seqid": fields[0],
                    "start": int(fields[3]),
                    "end": int(fields[4]),
                    "strand": fields[6],
                    "locus_tag": match.group(1) if match else None,
                }
            )
    return pd.DataFrame(rows).sort_values(["seqid", "start"]).reset_index(drop=True)


def upstream_gaps(genes: pd.DataFrame) -> pd.DataFrame:
    """Clear bases between each gene's 5' end and its upstream neighbor.

    A non-positive gap means the gene overlaps or abuts its upstream neighbor,
    which is ordinary in an operonic genome.
    """
    out = []
    for _, sub in genes.groupby("seqid"):
        sub = sub.sort_values("start").reset_index(drop=True)
        for i, row in sub.iterrows():
            if row.strand == "+":
                neighbor = sub.loc[i - 1, "end"] if i > 0 else None
                gap = row.start - neighbor - 1 if neighbor is not None else np.nan
            else:
                neighbor = sub.loc[i + 1, "start"] if i + 1 < len(sub) else None
                gap = neighbor - row.end - 1 if neighbor is not None else np.nan
            out.append(gap)
    genes = genes.copy()
    genes["upstream_gap"] = out
    genes["length"] = genes.end - genes.start + 1
    return genes


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    path = pinned_gff_path()
    genes = upstream_gaps(read_genes(path))
    gaps = genes.upstream_gap.dropna()

    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    report = {
        "gff": path,
        "genes": int(len(genes)),
        "replicons": int(genes.seqid.nunique()),
        "upstream_gap_bp": {
            "n": int(len(gaps)),
            "mean": float(gaps.mean()),
            "std": float(gaps.std()),
            "min": float(gaps.min()),
            "max": float(gaps.max()),
            **{f"q{int(q * 100)}": float(gaps.quantile(q)) for q in quantiles},
        },
        "fraction_with_clear_upstream": {
            str(w): float((gaps >= w).mean()) for w in WINDOWS
        },
        "fraction_overlapping_or_abutting": float((gaps <= 0).mean()),
        "gene_length_bp": {
            "median": float(genes.length.median()),
            "q90": float(genes.length.quantile(0.9)),
            "q99": float(genes.length.quantile(0.99)),
            "max": float(genes.length.max()),
        },
    }

    with open(osp.join(RESULTS, "kt2440_sequence_window_budget.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    genes.to_csv(osp.join(RESULTS, "kt2440_gene_upstream_gaps.csv"), index=False)

    print(f"{report['genes']} genes on {report['replicons']} replicon(s)")
    print(
        f"median upstream gap {report['upstream_gap_bp']['q50']:.0f} bp, "
        f"mean {report['upstream_gap_bp']['mean']:.0f} bp"
    )
    print(
        f"overlapping or abutting upstream neighbor: "
        f"{report['fraction_overlapping_or_abutting']:.3f}"
    )
    for w in WINDOWS:
        print(
            f"  clear upstream >= {w:>4} bp: "
            f"{report['fraction_with_clear_upstream'][str(w)]:.3f}"
        )
    print(
        f"gene length median {report['gene_length_bp']['median']:.0f} bp, "
        f"q99 {report['gene_length_bp']['q99']:.0f} bp"
    )
    print(f"results -> {RESULTS}")


if __name__ == "__main__":
    main()
