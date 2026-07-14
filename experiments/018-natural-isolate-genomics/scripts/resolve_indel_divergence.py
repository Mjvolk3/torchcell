# experiments/018-natural-isolate-genomics/scripts/resolve_indel_divergence.py
# [[experiments.018-natural-isolate-genomics.scripts.resolve_indel_divergence]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/018-natural-isolate-genomics/scripts/resolve_indel_divergence

"""Score the length-changing (indel) alleles that Hamming distance cannot reach.

``build_divergence_matrix.py`` scores an isolate allele against S288C position-wise,
which is only defined when the indel-inferred allele preserved the reference span
length. That covers ~90% of gene x isolate pairs; the remaining ~582k carry an indel
and were flagged ``is_indel`` with ``divergence = NaN``.

Leaving them NaN would silently bias the headline divergence downward (indel-bearing
alleles are not a random subset). Here we close the gap with an exact global
Levenshtein distance (BioPython ``PairwiseAligner``, match 0 / mismatch -1 / gap -1,
so ``-score`` is the edit distance) and emit a COMPLETE divergence column.

Two divergence notions are kept distinct on purpose:

* ``snp_divergence``  -- length-preserved pairs, het-weighted per Peter 2018's own
  published convention. This is the quantity comparable to their released
  ``1011DistanceMatrixBasedOnSNPs`` ("the percentage, based on SNPs, of non-identical
  bases").
* ``total_divergence`` -- ``snp_divergence`` where defined, else
  ``edit_distance / max(len_ref, len_iso)``. Complete over all 6.08M pairs.

Caveat, stated rather than hidden: the edit-distance pass charges an IUPAC
heterozygous site full weight instead of half. Het sites are ~0.1% of positions, so
the effect on the indel subset is negligible, but the two columns are therefore not
byte-identical in convention and we never average them silently.
"""

import multiprocessing as mp
import os
import os.path as osp

import pandas as pd
from Bio import Align
from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
EXP_DIR = osp.join(EXPERIMENT_ROOT, "018-natural-isolate-genomics")
RESULTS_DIR = osp.join(EXP_DIR, "results")

GENES_DIR = os.environ.get(
    "PETER_GENES_DIR",
    osp.join(DATA_ROOT, "data/peter2018/reference_genes_with_snps_indels"),
)
N_WORKERS = int(os.environ.get("N_WORKERS", "64"))


def _make_aligner() -> Align.PairwiseAligner:
    a = Align.PairwiseAligner()
    a.mode = "global"
    a.match_score = 0
    a.mismatch_score = -1
    a.open_gap_score = -1
    a.extend_gap_score = -1
    return a


def _read_fasta(path: str) -> tuple[list[str], list[str]]:
    heads: list[str] = []
    seqs: list[str] = []
    buf: list[str] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if buf:
                    seqs.append("".join(buf))
                    buf = []
                heads.append(line[1:].rstrip("\n"))
            else:
                buf.append(line.strip())
    if buf:
        seqs.append("".join(buf))
    return heads, seqs


def _strain_of(header: str) -> str:
    tok = header.split("\t")[0]
    if tok.startswith("SACE_"):
        tok = tok[5:]
    return tok.split("_")[0]


def _worker(job: tuple[str, str, list[str]]) -> list[dict]:
    """Align every indel-bearing allele of one gene against its S288C span."""
    gene, ref_seq, want = job
    aligner = _make_aligner()
    heads, seqs = _read_fasta(osp.join(GENES_DIR, f"{gene}.fasta"))
    want_set = set(want)
    out: list[dict] = []
    for h, s in zip(heads, seqs):
        st = _strain_of(h)
        if st not in want_set:
            continue
        ed = -aligner.score(ref_seq, s)
        out.append(
            {
                "gene": gene,
                "strain": st,
                "edit_distance": int(ed),
                "edit_divergence": float(ed) / max(len(ref_seq), len(s)),
                "len_delta": len(s) - len(ref_seq),
            }
        )
    return out


def main() -> None:
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    print("[1/4] loading divergence matrix ...", flush=True)
    div = pd.read_parquet(osp.join(RESULTS_DIR, "per_gene_isolate_divergence.parquet"))
    indel = div[div["is_indel"]]
    print(
        f"      {len(div):,} pairs | indel pairs to align: {len(indel):,} "
        f"({100 * len(indel) / len(div):.1f}%)",
        flush=True,
    )

    print("[2/4] loading S288C R64 reference spans ...", flush=True)
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    jobs: list[tuple[str, str, list[str]]] = []
    for gene, grp in indel.groupby("gene", observed=True):
        jobs.append(
            (str(gene), str(genome[str(gene)].seq), grp["strain"].astype(str).tolist())
        )
    print(f"      {len(jobs)} genes carry at least one indel allele", flush=True)

    print(f"[3/4] aligning on {N_WORKERS} workers ...", flush=True)
    rows: list[dict] = []
    with mp.Pool(N_WORKERS) as pool:
        for k, out in enumerate(pool.imap_unordered(_worker, jobs, chunksize=4)):
            rows.extend(out)
            if (k + 1) % 500 == 0:
                print(f"      {k + 1}/{len(jobs)} genes", flush=True)

    ed = pd.DataFrame(rows)
    print(f"      aligned {len(ed):,} indel pairs", flush=True)

    print("[4/4] merging into a complete divergence column ...", flush=True)
    div = div.merge(ed, on=["gene", "strain"], how="left")
    div = div.rename(columns={"divergence": "snp_divergence"})
    div["total_divergence"] = div["snp_divergence"].fillna(div["edit_divergence"])

    n_missing = int(div["total_divergence"].isna().sum())
    print(f"      pairs still unscored: {n_missing}", flush=True)

    div.to_parquet(osp.join(RESULTS_DIR, "per_gene_isolate_divergence.parquet"))

    per_strain = pd.read_parquet(
        osp.join(RESULTS_DIR, "per_strain_divergence_summary.parquet")
    )
    tot = (
        div.groupby("strain", observed=True)
        .agg(
            total_divergence_mean=("total_divergence", "mean"),
            edit_distance_sum=("edit_distance", "sum"),
            n_indel_alleles=("is_indel", "sum"),
        )
        .reset_index()
    )
    per_strain = per_strain.drop(
        columns=[c for c in tot.columns if c != "strain" and c in per_strain.columns]
    ).merge(tot, on="strain", how="left")
    per_strain.to_parquet(
        osp.join(RESULTS_DIR, "per_strain_divergence_summary.parquet")
    )

    per_gene = pd.read_parquet(
        osp.join(RESULTS_DIR, "per_gene_divergence_summary.parquet")
    )
    gtot = (
        div.groupby("gene", observed=True)
        .agg(
            total_divergence_mean=("total_divergence", "mean"),
            total_divergence_median=("total_divergence", "median"),
            n_indel_alleles=("is_indel", "sum"),
        )
        .reset_index()
    )
    per_gene = per_gene.drop(
        columns=[c for c in gtot.columns if c != "gene" and c in per_gene.columns]
    ).merge(gtot, on="gene", how="left")
    per_gene.to_parquet(osp.join(RESULTS_DIR, "per_gene_divergence_summary.parquet"))

    print("\n=== SUMMARY (indel-complete) ===")
    print(f"snp_divergence   mean : {div['snp_divergence'].mean() * 100:.4f}%")
    print(f"total_divergence mean : {div['total_divergence'].mean() * 100:.4f}%")
    print(
        f"indel alleles         : {int(div['is_indel'].sum()):,} "
        f"({100 * div['is_indel'].mean():.1f}% of pairs)"
    )
    print(f"median |len_delta|    : {div['len_delta'].abs().median()}")
    print(
        f"mean edit distance on indel alleles: "
        f"{div.loc[div['is_indel'], 'edit_distance'].mean():.1f} bp"
    )


if __name__ == "__main__":
    main()
