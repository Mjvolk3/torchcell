# experiments/018-natural-isolate-genomics/scripts/regulatory_divergence.py
# [[experiments.018-natural-isolate-genomics.scripts.regulatory_divergence]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/018-natural-isolate-genomics/scripts/regulatory_divergence

"""Where in the genome does natural variation actually sit -- coding, regulatory, or neither?

Issue #66 asks specifically about the window the **species-aware transformer** consumes
(``FungalUpDownTransformer``, HuggingFace ``gagneurlab/SpeciesLM``). That window is
pinned in ``torchcell/datasets/fungal_up_down_transformer.py:29-32``:

    MODEL_TO_WINDOW = {
        "species_downstream": ("window_three_prime", 300,  True, True),
        "species_upstream":   ("window_five_prime",  1003, True, True),
    }

with codon-inclusive semantics, so the model actually reads

    1000 bp upstream of the start codon  +  ATG      (1003 total)
    stop codon  +  297 bp downstream of it           (300 total)

i.e. the NON-CODING part of its input is exactly 1000 bp upstream and 297 bp downstream.
This script measures how much of the population's sequence variation falls inside that
window versus inside the CDS versus in intergenic space that no model input ever sees.

Data: Peter 2018 ``1011Matrix.gvcf.gz`` -- all SNPs and indels called at the population
level across the 1,011 isolates (md5 42478e3e9dff4bd46993d82d8eab40d4, verified against
the source's published md5.txt; sha256 037773254154e74c36f9cd19d4fee0e413cd371f699b80b
234cca59690f3a37e).

Statistic: nucleotide diversity per site, from the INFO AC/AN fields:

    pi_site = 2 * AC * (AN - AC) / (AN * (AN - 1))      [unbiased heterozygosity]
    pi_region = sum(pi_site over region) / region_length_bp

pi is the expected fraction of positions at which two randomly drawn chromosomes differ.
Reporting pi (not raw variant counts) is what makes regions of different length and
different call depth comparable. Raw variant density is reported alongside it.

Region assignment is by PRECEDENCE, because the yeast genome is compact and a gene's
upstream window frequently overlaps its neighbour's CDS:

    CDS  >  upstream_1000  >  downstream_297  >  intergenic_other

so a base is only counted as "regulatory" if it is not coding for anything. The overlap
that this precedence resolves is reported, not hidden -- it is large, and it is itself a
finding about how little truly-intergenic space yeast has.
"""

import json
import os
import os.path as osp
import subprocess

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
EXP_DIR = osp.join(EXPERIMENT_ROOT, "018-natural-isolate-genomics")
RESULTS_DIR = osp.join(EXP_DIR, "results")

GVCF = os.environ.get(
    "PETER_GVCF", osp.join(DATA_ROOT, "data/peter2018/1011Matrix.gvcf.gz")
)
GVCF_MD5 = "42478e3e9dff4bd46993d82d8eab40d4"
GVCF_SHA256 = "037773254154e74c36f9cd19d4fee0e413cd371f699b80b234cca59690f3a37e"

# The species-aware transformer's non-coding input windows.
UPSTREAM_BP = 1000
DOWNSTREAM_BP = 297

CDS = 1
UPSTREAM = 2
DOWNSTREAM = 3
INTERGENIC = 0
LABELS = {
    CDS: "cds",
    UPSTREAM: f"upstream_{UPSTREAM_BP}",
    DOWNSTREAM: f"downstream_{DOWNSTREAM_BP}",
    INTERGENIC: "intergenic_other",
}
MIN_AN = int(os.environ.get("MIN_AN", "100"))  # drop sites with almost no calls


def build_region_map(genome) -> tuple[dict[int, np.ndarray], dict]:
    """Per-chromosome uint8 label array over the S288C R64 nuclear genome."""
    from Bio import SeqIO

    lens: dict[int, int] = {}

    fasta = osp.join(
        DATA_ROOT,
        "data/sgd/genome/S288C_reference_genome_R64-4-1_20230830",
        "S288C_reference_sequence_R64-4-1_20230830.fsa",
    )
    roman = {
        "I": 1,
        "II": 2,
        "III": 3,
        "IV": 4,
        "V": 5,
        "VI": 6,
        "VII": 7,
        "VIII": 8,
        "IX": 9,
        "X": 10,
        "XI": 11,
        "XII": 12,
        "XIII": 13,
        "XIV": 14,
        "XV": 15,
        "XVI": 16,
    }
    for rec in SeqIO.parse(fasta, "fasta"):
        desc = rec.description
        # e.g. "... [chromosome=I]" ; mitochondrion has [location=mitochondrion]
        if "[chromosome=" in desc:
            r = desc.split("[chromosome=")[1].split("]")[0]
            if r in roman:
                lens[roman[r]] = len(rec.seq)

    maps = {c: np.zeros(n + 1, dtype=np.uint8) for c, n in lens.items()}

    genes = []
    for g in sorted(genome.gene_set):
        gene = genome[g]
        c = int(gene.chromosome)
        if c not in maps:  # skips chrmt (chromosome 0)
            continue
        genes.append((g, c, int(gene.start), int(gene.end), str(gene.strand)))

    # 1) CDS / gene body wins everywhere
    for _, c, s, e, _ in genes:
        maps[c][max(s, 1) : min(e, len(maps[c]) - 1) + 1] = CDS

    n_up_overlap = 0
    n_down_overlap = 0

    # 2) upstream window, only where still unlabelled
    for _, c, s, e, strand in genes:
        L = len(maps[c]) - 1
        if strand == "+":
            a, b = s - UPSTREAM_BP, s - 1
        else:
            a, b = e + 1, e + UPSTREAM_BP
        a, b = max(a, 1), min(b, L)
        if a > b:
            continue
        seg = maps[c][a : b + 1]
        n_up_overlap += int((seg != INTERGENIC).sum())
        seg[seg == INTERGENIC] = UPSTREAM

    # 3) downstream window, only where still unlabelled
    for _, c, s, e, strand in genes:
        L = len(maps[c]) - 1
        if strand == "+":
            a, b = e + 1, e + DOWNSTREAM_BP
        else:
            a, b = s - DOWNSTREAM_BP, s - 1
        a, b = max(a, 1), min(b, L)
        if a > b:
            continue
        seg = maps[c][a : b + 1]
        n_down_overlap += int((seg != INTERGENIC).sum())
        seg[seg == INTERGENIC] = DOWNSTREAM

    bp = {LABELS[k]: 0 for k in LABELS}
    for c, m in maps.items():
        v = m[1:]
        for k, name in LABELS.items():
            bp[name] += int((v == k).sum())

    meta = {
        "n_genes_mapped": len(genes),
        "chromosome_lengths": lens,
        "genome_bp": int(sum(lens.values())),
        "region_bp": bp,
        "upstream_bp_lost_to_overlap": n_up_overlap,
        "downstream_bp_lost_to_overlap": n_down_overlap,
        "note": (
            "precedence CDS > upstream > downstream > intergenic; overlap counts are "
            "bases claimed by an earlier label"
        ),
    }
    return maps, meta


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    print("[1/3] building region map (S288C R64) ...", flush=True)
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    maps, meta = build_region_map(genome)
    print(f"      nuclear genome: {meta['genome_bp']:,} bp", flush=True)
    for name, n in meta["region_bp"].items():
        print(
            f"        {name:22s} {n:>10,} bp  ({100 * n / meta['genome_bp']:5.1f}%)",
            flush=True,
        )
    print(
        f"      upstream bases lost to CDS overlap: "
        f"{meta['upstream_bp_lost_to_overlap']:,}",
        flush=True,
    )

    print(f"[2/3] streaming {GVCF} ...", flush=True)
    n_regions = 4
    pi_sum = np.zeros(n_regions, dtype=np.float64)
    n_var = np.zeros(n_regions, dtype=np.int64)
    n_snp = np.zeros(n_regions, dtype=np.int64)
    n_indel = np.zeros(n_regions, dtype=np.int64)
    an_vals: list[int] = []

    n_lines = 0
    n_skipped_an = 0
    n_off_map = 0

    proc = subprocess.Popen(["zcat", GVCF], stdout=subprocess.PIPE, bufsize=1 << 22)
    assert proc.stdout is not None
    for raw in proc.stdout:
        if raw[:1] == b"#":
            continue
        # split ONLY the fixed columns; the 1,011 sample fields stay one untouched blob
        f = raw.split(b"\t", 8)
        chrom = f[0]
        if not chrom.startswith(b"chromosome"):
            continue
        try:
            c = int(chrom[10:])
        except ValueError:
            continue
        if c not in maps:
            continue
        pos = int(f[1])
        m = maps[c]
        if pos >= len(m):
            n_off_map += 1
            continue

        info = f[7]
        ac = an = -1
        for kv in info.split(b";"):
            if kv.startswith(b"AC="):
                ac = int(kv[3:].split(b",")[0])
            elif kv.startswith(b"AN="):
                an = int(kv[3:])
                break
        if an < MIN_AN or ac <= 0 or ac >= an:
            n_skipped_an += 1
            continue

        r = m[pos]
        pi_sum[r] += 2.0 * ac * (an - ac) / (an * (an - 1))
        n_var[r] += 1
        if len(f[3]) == 1 and len(f[4]) == 1:
            n_snp[r] += 1
        else:
            n_indel[r] += 1
        an_vals.append(an)

        n_lines += 1
        if n_lines % 500_000 == 0:
            print(f"      {n_lines:,} variants", flush=True)
    proc.wait()

    print(
        f"      kept {n_lines:,} variants | dropped (AN<{MIN_AN} or non-polymorphic): "
        f"{n_skipped_an:,}",
        flush=True,
    )

    print("[3/3] writing ...", flush=True)
    rows = []
    for k, name in LABELS.items():
        bp = meta["region_bp"][name]
        rows.append(
            {
                "region": name,
                "bp": bp,
                "frac_of_genome": bp / meta["genome_bp"],
                "n_variants": int(n_var[k]),
                "n_snps": int(n_snp[k]),
                "n_indels": int(n_indel[k]),
                "variants_per_kb": 1000 * n_var[k] / bp if bp else np.nan,
                "pi": pi_sum[k] / bp if bp else np.nan,
                "pi_percent": 100 * pi_sum[k] / bp if bp else np.nan,
            }
        )
    df = pd.DataFrame(rows)

    # the species-aware transformer's total input footprint
    win = df[df.region.isin(["cds", LABELS[UPSTREAM], LABELS[DOWNSTREAM]])]
    total_bp = int(df.bp.sum())
    summary = {
        "gvcf": {"path": GVCF, "md5": GVCF_MD5, "sha256": GVCF_SHA256},
        "min_an": MIN_AN,
        "n_variants_kept": n_lines,
        "n_variants_dropped": n_skipped_an,
        "an_median": float(np.median(an_vals)) if an_vals else None,
        "region_map": meta,
        "species_aware_transformer_window": {
            "upstream_bp": UPSTREAM_BP,
            "downstream_bp": DOWNSTREAM_BP,
            "source": "torchcell/datasets/fungal_up_down_transformer.py:29-32",
            "model": "gagneurlab/SpeciesLM (FungalUpDownTransformer)",
            "bp_covered": int(win.bp.sum()),
            "frac_of_genome_covered": float(win.bp.sum() / total_bp),
            "frac_of_all_variation_captured": float(
                win.n_variants.sum() / df.n_variants.sum()
            ),
            "frac_of_pi_captured": float(
                (win.pi * win.bp).sum() / (df.pi * df.bp).sum()
            ),
        },
    }
    df.to_parquet(osp.join(RESULTS_DIR, "regulatory_divergence_by_region.parquet"))
    df.to_csv(osp.join(RESULTS_DIR, "regulatory_divergence_by_region.csv"), index=False)
    with open(osp.join(RESULTS_DIR, "regulatory_divergence_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print("\n=== VARIATION BY GENOMIC REGION (S288C R64, 1,011 isolates) ===")
    show = df.copy()
    show["bp"] = show["bp"].map("{:,}".format)
    show["pi_percent"] = show["pi_percent"].round(4)
    show["variants_per_kb"] = show["variants_per_kb"].round(2)
    print(
        show[
            [
                "region",
                "bp",
                "frac_of_genome",
                "n_variants",
                "variants_per_kb",
                "pi_percent",
            ]
        ].to_string(index=False)
    )
    saw = summary["species_aware_transformer_window"]
    print(
        f"\n>>> The species-aware transformer's input (CDS + {UPSTREAM_BP} up + "
        f"{DOWNSTREAM_BP} down) covers "
        f"{100 * saw['frac_of_genome_covered']:.1f}% of the genome and captures "
        f"{100 * saw['frac_of_pi_captured']:.1f}% of all nucleotide diversity."
    )
    print(f"\nresults -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
