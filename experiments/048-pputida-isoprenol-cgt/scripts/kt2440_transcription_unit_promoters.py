# experiments/048-pputida-isoprenol-cgt/scripts/kt2440_transcription_unit_promoters.py
# [[experiments.048-pputida-isoprenol-cgt.scripts.kt2440_transcription_unit_promoters]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/048-pputida-isoprenol-cgt/scripts/kt2440_transcription_unit_promoters
"""Where is a KT2440 gene's promoter, and how much sequence can we actually take?

`kt2440_sequence_window_budget.py` showed a fixed upstream window is the wrong
unit here (median 88 bp clear, 0.7% of genes with 1 kb). This script answers the
constructive half: in a bacterium the promoter belongs to the **transcription
unit**, not the gene, so the capture is

  1. chain adjacent same-strand genes into transcription units,
  2. take the promoter window upstream of the TU leader only,
  3. let every gene in the TU inherit its leader's promoter.

The operon threshold is not assumed. It is set by contrast against **divergent**
gene pairs, which cannot share a transcription unit and must contain promoters
in their gap, so they are the natural negative class for "this gap holds a
promoter". Convergent pairs are reported too, as the terminator-containing
control.

Writes JSON + CSV (the TU table and the per-gene promoter assignment) into
experiments/048-pputida-isoprenol-cgt/results/.
"""

import glob
import gzip
import json
import os
import os.path as osp
import re

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

RESULTS = osp.join(EXPERIMENT_ROOT, "048-pputida-isoprenol-cgt", "results")
GENOME_DIR = osp.join(DATA_ROOT, "torchcell-genomes", "pputida_KT2440_ASM756v2")

# Candidate intra-operon gap thresholds, scanned so the choice is reported with
# its separation rather than asserted.
THRESHOLDS = (0, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300)

# The threshold the results below support, and the promoter window cap. The cap
# is near the divergent-gap median, which is the amount of sequence a gap that
# genuinely holds a promoter tends to offer.
OPERON_GAP_MAX = 40
PROMOTER_WINDOW_MAX = 300


def pinned_gff_path() -> str:
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


def junction_table(genes: pd.DataFrame) -> pd.DataFrame:
    """Every adjacent gene pair, its gap, and its relative orientation."""
    rows = []
    for seqid, sub in genes.groupby("seqid"):
        sub = sub.sort_values("start").reset_index(drop=True)
        for i in range(len(sub) - 1):
            left, right = sub.loc[i], sub.loc[i + 1]
            if left.strand == right.strand:
                orientation = "tandem"
            elif left.strand == "-" and right.strand == "+":
                orientation = "divergent"
            else:
                orientation = "convergent"
            rows.append(
                {
                    "seqid": seqid,
                    "left": left.locus_tag,
                    "right": right.locus_tag,
                    "gap": right.start - left.end - 1,
                    "orientation": orientation,
                }
            )
    return pd.DataFrame(rows)


def threshold_scan(junctions: pd.DataFrame) -> list[dict]:
    """Separation between tandem and divergent gaps at each candidate threshold.

    `enrichment` is how many times more often a tandem junction falls under the
    threshold than a divergent one. A high value means a gap that short is
    unlikely to hold a promoter, which is what licenses chaining the pair.
    """
    out = []
    by = {k: sub.gap for k, sub in junctions.groupby("orientation")}
    for threshold in THRESHOLDS:
        tandem = float((by["tandem"] <= threshold).mean())
        divergent = float((by["divergent"] <= threshold).mean())
        convergent = float((by["convergent"] <= threshold).mean())
        out.append(
            {
                "threshold": threshold,
                "tandem_fraction": tandem,
                "divergent_fraction": divergent,
                "convergent_fraction": convergent,
                "enrichment_over_divergent": (
                    tandem / divergent if divergent > 0 else None
                ),
            }
        )
    return out


def build_transcription_units(
    genes: pd.DataFrame, gap_max: int = OPERON_GAP_MAX
) -> pd.DataFrame:
    """Chain adjacent same-strand genes whose gap is at or below `gap_max`.

    On the minus strand the leader is the right-most gene, since transcription
    runs the other way, so the TU's leader is chosen by strand.
    """
    assignments = []
    for seqid, sub in genes.groupby("seqid"):
        sub = sub.sort_values("start").reset_index(drop=True)
        tu_index = 0
        tu_ids = [0]
        for i in range(1, len(sub)):
            left, right = sub.loc[i - 1], sub.loc[i]
            same_tu = (
                left.strand == right.strand and (right.start - left.end - 1) <= gap_max
            )
            if not same_tu:
                tu_index += 1
            tu_ids.append(tu_index)
        sub["tu"] = [f"{seqid}:TU{j:05d}" for j in tu_ids]
        assignments.append(sub)
    assigned = pd.concat(assignments, ignore_index=True)

    units = []
    for tu, sub in assigned.groupby("tu"):
        strand = sub.strand.iloc[0]
        leader = (
            sub.sort_values("start").iloc[0]
            if strand == "+"
            else sub.sort_values("start").iloc[-1]
        )
        units.append(
            {
                "tu": tu,
                "seqid": sub.seqid.iloc[0],
                "strand": strand,
                "n_genes": int(len(sub)),
                "leader": leader.locus_tag,
                "leader_start": int(leader.start),
                "leader_end": int(leader.end),
                "members": ",".join(sub.sort_values("start").locus_tag.astype(str)),
            }
        )
    return assigned, pd.DataFrame(units).sort_values("tu").reset_index(drop=True)


def promoter_windows(
    genes: pd.DataFrame, units: pd.DataFrame, window_max: int = PROMOTER_WINDOW_MAX
) -> pd.DataFrame:
    """Available promoter interval upstream of each TU leader.

    Clipped at the nearest neighboring gene edge in that direction, so the
    window never reads into another gene. `available` is the clear length, which
    is itself a feature: a TU whose leader abuts its neighbor has no promoter
    window at all, and the model should know that rather than be fed a window
    silently filled with coding sequence.
    """
    rows = []
    for seqid, sub_units in units.groupby("seqid"):
        other = genes[genes.seqid == seqid].sort_values("start").reset_index(drop=True)
        starts = other.start.to_numpy()
        ends = other.end.to_numpy()
        for _, unit in sub_units.iterrows():
            if unit.strand == "+":
                upstream_ends = ends[ends < unit.leader_start]
                boundary = int(upstream_ends.max()) if len(upstream_ends) else 0
                available = unit.leader_start - boundary - 1
                take = max(0, min(window_max, available))
                win_start = unit.leader_start - take
                win_end = unit.leader_start - 1
            else:
                downstream_starts = starts[starts > unit.leader_end]
                boundary = (
                    int(downstream_starts.min())
                    if len(downstream_starts)
                    else int(ends.max()) + 1
                )
                available = boundary - unit.leader_end - 1
                take = max(0, min(window_max, available))
                win_start = unit.leader_end + 1
                win_end = unit.leader_end + take
            rows.append(
                {
                    "tu": unit.tu,
                    "seqid": seqid,
                    "strand": unit.strand,
                    "leader": unit.leader,
                    "n_genes": unit.n_genes,
                    "available_bp": int(available),
                    "window_bp": int(take),
                    "window_start": int(win_start),
                    "window_end": int(win_end),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    path = pinned_gff_path()
    genes = read_genes(path)
    junctions = junction_table(genes)
    scan = threshold_scan(junctions)
    assigned, units = build_transcription_units(genes)
    windows = promoter_windows(genes, units)

    gap_stats = {
        orientation: {
            "n": int(len(sub)),
            "median": float(sub.gap.median()),
            "mean": float(sub.gap.mean()),
            "q25": float(sub.gap.quantile(0.25)),
            "q75": float(sub.gap.quantile(0.75)),
        }
        for orientation, sub in junctions.groupby("orientation")
    }

    report = {
        "gff": path,
        "genes": int(len(genes)),
        "junctions": int(len(junctions)),
        "gap_by_orientation": gap_stats,
        "threshold_scan": scan,
        "operon_gap_max": OPERON_GAP_MAX,
        "promoter_window_max": PROMOTER_WINDOW_MAX,
        "transcription_units": int(len(units)),
        "multigene_units": int((units.n_genes > 1).sum()),
        "genes_in_multigene_units": int(units[units.n_genes > 1].n_genes.sum()),
        "largest_unit_genes": int(units.n_genes.max()),
        "promoter_window_bp": {
            "median": float(windows.window_bp.median()),
            "mean": float(windows.window_bp.mean()),
            "zero_window_units": int((windows.window_bp == 0).sum()),
            "full_window_units": int((windows.window_bp == PROMOTER_WINDOW_MAX).sum()),
        },
    }

    with open(
        osp.join(RESULTS, "kt2440_transcription_unit_promoters.json"), "w"
    ) as handle:
        json.dump(report, handle, indent=2)
    units.to_csv(osp.join(RESULTS, "kt2440_transcription_units.csv"), index=False)
    windows.to_csv(osp.join(RESULTS, "kt2440_tu_promoter_windows.csv"), index=False)
    assigned[["locus_tag", "seqid", "start", "end", "strand", "tu"]].to_csv(
        osp.join(RESULTS, "kt2440_gene_to_transcription_unit.csv"), index=False
    )

    print(f"{report['genes']} genes, {report['junctions']} junctions")
    print(
        "median gap by orientation: "
        + ", ".join(
            f"{k} {v['median']:.0f} bp (n={v['n']})" for k, v in gap_stats.items()
        )
    )
    print("\nthreshold scan (tandem vs divergent, the promoter-bearing null):")
    for row in scan:
        enrich = row["enrichment_over_divergent"]
        enrich_s = f"{enrich:5.1f}x" if enrich else "    inf"
        print(
            f"  gap <= {row['threshold']:>3} bp: tandem "
            f"{row['tandem_fraction']:.3f}  divergent "
            f"{row['divergent_fraction']:.3f}  convergent "
            f"{row['convergent_fraction']:.3f}  enrichment {enrich_s}"
        )
    print(
        f"\nat gap <= {OPERON_GAP_MAX} bp: {report['transcription_units']} "
        f"transcription units, {report['multigene_units']} multigene, "
        f"largest {report['largest_unit_genes']} genes"
    )
    print(
        f"promoter window: median {report['promoter_window_bp']['median']:.0f} bp, "
        f"{report['promoter_window_bp']['zero_window_units']} units with none, "
        f"{report['promoter_window_bp']['full_window_units']} at the "
        f"{PROMOTER_WINDOW_MAX} bp cap"
    )
    print(f"results -> {RESULTS}")


if __name__ == "__main__":
    main()
