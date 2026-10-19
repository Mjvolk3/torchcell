# experiments/044-organism-similarity/scripts/06_isolate_protein_identity.py
# [[experiments.044-organism-similarity.scripts.06_isolate_protein_identity]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/044-organism-similarity/scripts/06_isolate_protein_identity.py

"""Protein identity of each of the 1,011 isolates against S288C, and the single-copy AAI.

Step ``isolates`` (the within-species rung of the claim, at the protein level):

- Input: ``allReferenceGenesWithSNPsAndIndelsInferred.tar.gz`` from the genomes tier set
  ``peter2018_1011_assemblies`` (sha256 verified by ``registry.resolve``), streamed member
  by member. Each member is one reference ORF; each record is one isolate's allele over
  the S288C GENE SPAN (introns included), in gene orientation, SNPs and indels inferred.
- Reference: the S288C R64-4-1 chromosome FASTA and GFF from the tier. Each ORF's CDS
  exons are placed in the gene span; splicing the S288C span must reproduce the SGD
  ``orf_coding_all`` CDS exactly (asserted per ORF), which pins the coordinate mapping.
- ORFs whose Peter span (record header coordinates) differs from the R64-4-1 gene span
  (the annotation was revised after Peter's release) are excluded and counted.
- Isolate CDS: an allele with the S288C span length is spliced with the reference exon
  offsets; a length-changed allele of an intronless ORF is the CDS itself; a
  length-changed allele of an intron-containing ORF (intron length polymorphism is
  common, e.g. 968 of 1,011 EFB1/YAL003W alleles) is aligned to the S288C span at the
  nucleotide level and the bases aligned to exon positions, plus insertions between two
  exon positions, form its CDS.
- Translation (frame 1, terminal stop dropped, internal stops kept as ``*``): a codon
  with exactly one two-allele IUPAC (heterozygous) base yields both amino acids; any other
  ambiguity (N, gap, more than one ambiguous base) is a no-call.
- Scoring against the S288C protein: a residue pair costs 0 (identical), 1 (different),
  or 1/2 for a heterozygous codon one of whose amino acids is the reference one (the
  protein-level analog of Peter 2018's half-weighting of heterozygous differences, which
  018 uses at the nucleotide level); no-calls leave the denominator. Two alignments are
  scored, the ungapped one (only when the protein lengths are equal) and a Biopython
  ``PairwiseAligner`` global alignment (BLOSUM62, gap open -11, extend -1) in which every
  gap column costs 1; the ORF's identity is the higher of the two (1 - cost / scored
  columns). The global alignment is what rescues equal-length alleles carrying
  compensating indels, which an ungapped comparison scores as a long run of mismatches.
  A premature stop costs its one position in an ungapped comparison; a frameshift costs
  every misaligned or gapped column. Identical alleles are scored once per ORF.
- Per isolate: ``n_orfs`` scored, ``mean_identity`` and ``median_identity`` over ORFs,
  ``pooled_identity`` (identical residues over all scored residues), and the same mean
  over equal-length ORFs only. Output ``results/isolate_protein_identity.parquet``.

Step ``scog`` (Peris et al. 2023 define their AAI over BUSCO eukaryota_odb10 single-copy
orthologs; busco is not usable in the orgsim env, BUSCO 4.0.beta1 was the only
resolvable build, so this is the single-copy-OG PROXY): for every pair of eukaryote
proteomes, the Eukaryota-level eggNOG OGs (``@2759``) that hold exactly one protein in
each member; the identity of that protein pair is the mean of both directions' mmseqs
``pident`` from the all-vs-all search (one direction when only one was reported). Output
``results/busco_aai.csv`` (method column says it is the proxy).

Run from the repo root with the torchcell env (Biopython, pyarrow).
"""

from __future__ import annotations

import io
import multiprocessing as mp
import os
import os.path as osp
import sys
import tarfile
from collections.abc import Iterator

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.Seq import Seq

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from orgsim_manifest import RESULTS_DIR, WORK_DIR, load_manifest  # noqa: E402

from torchcell.sequence.genome.registry import resolve  # noqa: E402

PETER_SET = "peter2018_1011_assemblies"
PETER_TAR = "allReferenceGenesWithSNPsAndIndelsInferred.tar.gz"
SGD_SET = "sgd_S288C_R64-4-1_20230830"
SGD_FSA = "S288C_reference_sequence_R64-4-1_20230830.fsa"
SGD_GFF = "saccharomyces_cerevisiae_R64-4-1_20230830.gff"
SGD_CDS = "orf_coding_all_R64-4-1_20230830.fasta"
N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", "48"))
OUT_ISO = osp.join(RESULTS_DIR, "isolate_protein_identity.parquet")
OUT_PER_ORF = osp.join(WORK_DIR, "isolate_protein_identity_per_orf.parquet")
OUT_SCOG = osp.join(RESULTS_DIR, "busco_aai.csv")

IUPAC = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    "R": "AG",
    "Y": "CT",
    "S": "CG",
    "W": "AT",
    "K": "GT",
    "M": "AC",
    "B": "CGT",
    "D": "AGT",
    "H": "ACT",
    "V": "ACG",
}
CODON_TABLE = "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSS*CWCLFLF"
BASE_IDX = {b: i for i, b in enumerate("ACGT")}
ROMAN = {
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


def aa_of(codon: str) -> str:
    """Standard-code amino acid of an unambiguous ACGT codon."""
    return CODON_TABLE[
        16 * BASE_IDX[codon[0]] + 4 * BASE_IDX[codon[1]] + BASE_IDX[codon[2]]
    ]


def read_fasta_text(text: str) -> list[tuple[str, str]]:
    """``[(header, sequence)]`` of a FASTA string."""
    out: list[tuple[str, str]] = []
    head: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if head is not None:
                out.append((head, "".join(buf)))
            head, buf = line[1:], []
        else:
            buf.append(line.strip().upper())
    if head is not None:
        out.append((head, "".join(buf)))
    return out


def strain_of(header: str) -> str:
    """Isolate code of a Peter gene-FASTA header (018's rule)."""
    tok = header.split("\t")[0]
    tok = tok.removeprefix("SACE_")
    return tok.split("_")[0]


def load_reference() -> dict[str, dict]:
    """Per ORF: chromosome, span, strand, exon offsets in the span, CDS, protein."""
    fsa = read_fasta_text(open(resolve(SGD_SET, SGD_FSA)).read())
    chrom: dict[str, str] = {}
    for head, seq in fsa:
        tag = [t for t in head.split("[") if t.startswith("chromosome=")]
        if tag:
            chrom["chr" + tag[0].split("=")[1].rstrip("] ")] = seq
    genes: dict[str, dict] = {}
    cds: dict[str, list[tuple[int, int]]] = {}
    with open(resolve(SGD_SET, SGD_GFF)) as fh:
        for line in fh:
            if line.startswith("##FASTA"):
                break
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 9:
                continue
            attrs = dict(kv.split("=", 1) for kv in c[8].split(";") if "=" in kv)
            if c[2] == "gene":
                genes[attrs["ID"]] = {
                    "chrom": c[0],
                    "start": int(c[3]),
                    "end": int(c[4]),
                    "strand": c[6],
                }
            elif c[2] == "CDS":
                gene = attrs["Name"].removesuffix("_CDS")
                cds.setdefault(gene, [])
                iv = (int(c[3]), int(c[4]))
                if iv not in cds[gene]:
                    cds[gene].append(iv)
    sgd_cds = {
        h.split()[0]: s
        for h, s in read_fasta_text(open(resolve(SGD_SET, SGD_CDS)).read())
    }
    ref: dict[str, dict] = {}
    for name, g in genes.items():
        if name not in cds or name not in sgd_cds or g["chrom"] not in chrom:
            continue
        span = chrom[g["chrom"]][g["start"] - 1 : g["end"]].upper()
        mask = np.zeros(len(span), dtype=bool)
        for s, e in cds[name]:
            assert g["start"] <= s <= e <= g["end"], name
            mask[s - g["start"] : e - g["start"] + 1] = True
        if g["strand"] == "-":
            span = str(Seq(span).reverse_complement())
            mask = mask[::-1]
        idx = np.flatnonzero(mask)
        spliced = "".join(span[i] for i in idx)
        if spliced != sgd_cds[name]:
            continue
        ref[name] = {
            "span": span,
            "exon_idx": idx,
            "intronless": len(idx) == len(span),
            "cds": spliced,
            "chrom": g["chrom"],
            "start": g["start"],
            "end": g["end"],
        }
    return ref


ALIGNERS: dict[str, PairwiseAligner] = {}


def aligner(kind: str) -> PairwiseAligner:
    """Global aligners, one per worker: ``protein`` (BLOSUM62, -11/-1) or ``dna``."""
    if kind not in ALIGNERS:
        a = PairwiseAligner()
        a.mode = "global"
        if kind == "protein":
            a.substitution_matrix = substitution_matrices.load("BLOSUM62")
            a.open_gap_score = -11
            a.extend_gap_score = -1
        else:
            a.match_score = 2
            a.mismatch_score = -3
            a.open_gap_score = -5
            a.extend_gap_score = -2
        ALIGNERS[kind] = a
    return ALIGNERS[kind]


AASet = str | frozenset[str] | None


def translate_sets(cds: str) -> list[AASet]:
    """Frame-1 translation; a het codon (one two-allele base) gives its two amino
    acids (collapsed when equal), any other ambiguity is a no-call (None).
    """
    out: list[AASet] = []
    for k in range(0, len(cds) - len(cds) % 3, 3):
        c = cds[k : k + 3]
        amb = [i for i, b in enumerate(c) if b not in BASE_IDX]
        if not amb:
            out.append(aa_of(c))
        elif len(amb) == 1 and len(IUPAC.get(c[amb[0]], "")) == 2:
            i = amb[0]
            aas = frozenset(aa_of(c[:i] + b + c[i + 1 :]) for b in IUPAC[c[i]])
            out.append(next(iter(aas)) if len(aas) == 1 else aas)
        else:
            out.append(None)
    if out and out[-1] == "*":
        out = out[:-1]
    return out


def charge(iso: AASet, ref: str) -> float | None:
    """Difference charge of one aligned residue pair; None is a no-call."""
    if iso is None:
        return None
    if isinstance(iso, str):
        return float(iso != ref)
    return 0.5 if ref in iso else 1.0


def score_ungapped(prot: list[AASet], ref_aa: list[str]) -> tuple[float, int]:
    """(differences, scored residues) of the identity alignment (equal length)."""
    diff, valid = 0.0, 0
    for a, r in zip(prot, ref_aa):
        w = charge(a, r)
        if w is not None:
            diff += w
            valid += 1
    return diff, valid


def score_aligned(prot: list[AASet], ref_prot: str) -> tuple[float, int]:
    """(differences, scored columns) of the BLOSUM62 global alignment; every gap
    column is charged 1, no-call columns are dropped.
    """
    seq = "".join(a if isinstance(a, str) else "X" for a in prot)
    aln = aligner("protein").align(ref_prot, seq)[0]
    ri, qi = aln.indices
    diff, valid = 0.0, 0
    for r, q in zip(ri, qi):
        if r < 0 or q < 0:
            diff += 1.0
            valid += 1
            continue
        w = charge(prot[q], ref_prot[r])
        if w is not None:
            diff += w
            valid += 1
    return diff, valid


def project_cds(seq: str, r: dict) -> str:
    """Isolate CDS of an intron-containing ORF whose span length changed: align the
    span to the S288C span and keep isolate bases aligned to exon positions plus the
    insertions that fall between two exon positions.
    """
    aln = aligner("dna").align(r["span"], seq)[0]
    ri, qi = aln.indices
    exon = np.zeros(len(r["span"]), dtype=bool)
    exon[r["exon_idx"]] = True
    out = []
    prev_ref = -1
    n = len(ri)
    next_ref = np.full(n, -1)
    nxt = -1
    for c in range(n - 1, -1, -1):
        if ri[c] >= 0:
            nxt = ri[c]
        next_ref[c] = nxt
    for c in range(n):
        rr, qq = ri[c], qi[c]
        if rr >= 0:
            prev_ref = rr
            if exon[rr] and qq >= 0:
                out.append(seq[qq])
        elif qq >= 0:
            nr = next_ref[c]
            if prev_ref >= 0 and nr >= 0 and exon[prev_ref] and exon[nr]:
                out.append(seq[qq])
    return "".join(out)


def score_gene(args: tuple[str, str, dict]) -> list[dict]:
    """Every isolate allele of one ORF against the S288C protein."""
    gene, text, r = args
    recs = read_fasta_text(text)
    ref_cds = r["cds"]
    ref_aa = [aa_of(ref_cds[3 * k : 3 * k + 3]) for k in range(len(ref_cds) // 3)]
    if ref_aa and ref_aa[-1] == "*":
        ref_aa = ref_aa[:-1]
    ref_prot = "".join(ref_aa)
    L = len(r["span"])
    head0 = recs[0][0].split("\t")
    coords = head0[1] if len(head0) > 1 else ""
    expect = f":{r['start']}-{r['end']}"
    if not coords.endswith(expect):
        # Peter's span comes from an earlier R64 annotation of this ORF; the R64-4-1
        # exon offsets do not apply to it, so the ORF is excluded and counted.
        return [
            {
                "gene": gene,
                "strain": strain_of(head),
                "n_diff": np.nan,
                "n_valid": 0.0,
                "identity": np.nan,
                "n_stop": 0,
                "is_indel": False,
                "route": "excluded_span_differs_from_R64-4-1",
            }
            for head, _ in recs
        ]
    cache: dict[str, tuple[float, float, int, bool, str]] = {}
    rows = []
    for head, seq in recs:
        if seq not in cache:
            span_indel = len(seq) != L
            if not span_indel:
                cds = "".join(seq[i] for i in r["exon_idx"])
                route = "span_equal"
            elif r["intronless"]:
                cds = seq
                route = "intronless_indel"
            else:
                cds = project_cds(seq, r)
                route = "intron_projected"
            prot = translate_sets(cds)
            n_stop = sum(1 for a in prot if a == "*")
            best: tuple[float, int] | None = None
            if len(prot) == len(ref_aa):
                best = score_ungapped(prot, ref_aa)
            if best is None or best[1] == 0 or best[0] > 0:
                al = score_aligned(prot, ref_prot)
                if best is None or (
                    al[1] > 0 and (best[1] == 0 or al[0] / al[1] < best[0] / best[1])
                ):
                    best = al
            cache[seq] = (best[0], float(best[1]), n_stop, span_indel, route)
        w_sum, n_valid, n_stop, is_indel, route = cache[seq]
        rows.append(
            {
                "gene": gene,
                "strain": strain_of(head),
                "n_diff": w_sum,
                "n_valid": n_valid,
                "identity": 1.0 - w_sum / n_valid if n_valid > 0 else np.nan,
                "n_stop": n_stop,
                "is_indel": is_indel,
                "route": route,
            }
        )
    return rows


def iter_members(ref: dict[str, dict]) -> Iterator[tuple[str, str, dict]]:
    """Stream the tier tarball's gene FASTAs that have a validated reference."""
    path = resolve(PETER_SET, PETER_TAR)
    with tarfile.open(path, "r:gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".fasta"):
                continue
            gene = osp.basename(m.name).removesuffix(".fasta")
            if gene not in ref:
                continue
            fh = tf.extractfile(m)
            assert fh is not None
            yield gene, io.TextIOWrapper(fh).read(), ref[gene]


def isolates() -> None:
    """Per-isolate protein identity vs S288C."""
    ref = load_reference()
    print(
        f"reference ORFs validated against orf_coding_all: {len(ref)} "
        f"(intronless {sum(r['intronless'] for r in ref.values())})",
        flush=True,
    )
    frames: list[pd.DataFrame] = []
    n_genes = 0
    with mp.Pool(N_WORKERS) as pool:
        for rows in pool.imap_unordered(score_gene, iter_members(ref), chunksize=4):
            frames.append(pd.DataFrame(rows))
            n_genes += 1
            if n_genes % 500 == 0:
                print(f"  {n_genes} ORFs", flush=True)
    per = pd.concat(frames, ignore_index=True)
    per["gene"] = per["gene"].astype("category")
    per["strain"] = per["strain"].astype("category")
    per["route"] = per["route"].astype("category")
    print(per["route"].value_counts().to_string(), flush=True)
    per.to_parquet(OUT_PER_ORF)
    print(f"Peter ORFs scored: {n_genes}; (ORF, isolate) rows {len(per):,}", flush=True)
    ok = per[per["n_valid"] > 0]
    eq = ok[~ok["is_indel"]]
    iso = (
        ok.groupby("strain", observed=True)
        .agg(
            n_orfs=("gene", "size"),
            mean_identity=("identity", "mean"),
            median_identity=("identity", "median"),
            n_diff=("n_diff", "sum"),
            n_valid=("n_valid", "sum"),
            n_indel_orfs=("is_indel", "sum"),
            n_premature_stop_orfs=("n_stop", lambda x: int((x > 0).sum())),
        )
        .reset_index()
    )
    iso["pooled_identity"] = 1.0 - iso["n_diff"] / iso["n_valid"]
    iso_eq = (
        eq.groupby("strain", observed=True)["identity"]
        .mean()
        .rename("mean_identity_equal_length")
    )
    iso = iso.merge(iso_eq, on="strain")
    iso = iso.rename(columns={"strain": "isolate"})
    iso.to_parquet(OUT_ISO)
    print(iso.describe().to_string(), flush=True)
    print(
        f"median over isolates of mean_identity: {iso.mean_identity.median():.5f}; "
        f"of pooled_identity: {iso.pooled_identity.median():.5f}",
        flush=True,
    )


def scog() -> None:
    """Single-copy Eukaryota-OG AAI (proxy for Peris's BUSCO AAI)."""
    lin = pd.read_csv(osp.join(RESULTS_DIR, "organism_lineage.csv")).set_index("key")
    manifest = load_manifest()
    euk = [o.key for o in manifest.present() if not bool(lin.loc[o.key, "is_bacteria"])]
    ann = pd.read_parquet(
        osp.join(WORK_DIR, "emapper", "annotations.parquet"),
        columns=["query", "org", "eggNOG_OGs"],
    )
    ann = ann[ann["org"].isin(euk)].copy()

    def euk_og(v: str) -> str | None:
        hits = [
            x.split("@")[0] for x in v.split(",") if x.split("|")[0].endswith("@2759")
        ]
        return hits[0] if hits else None

    ann["og"] = ann["eggNOG_OGs"].map(euk_og)
    ann = ann[ann["og"].notna()]
    n_per = ann.groupby(["org", "og"]).size()
    single = n_per[n_per == 1].reset_index()[["org", "og"]]
    sc = single.merge(ann[["org", "og", "query"]], on=["org", "og"])
    sc_map = {k: g.set_index("og")["query"] for k, g in sc.groupby("org")}
    hits = pd.read_parquet(
        osp.join(WORK_DIR, "mmseqs", "all_vs_all.parquet"),
        columns=["query", "target", "pident"],
    )
    sc_prots = set(sc["query"])
    hits = hits[hits["query"].isin(sc_prots) & hits["target"].isin(sc_prots)]
    best = hits.groupby(["query", "target"])["pident"].max()
    rows = []
    for a in range(len(euk)):
        for b in range(a + 1, len(euk)):
            i, j = euk[a], euk[b]
            shared = sc_map[i].index.intersection(sc_map[j].index)
            pi = sc_map[i].loc[shared].to_numpy()
            pj = sc_map[j].loc[shared].to_numpy()
            ident = []
            for p, q in zip(pi, pj):
                v = [best.get((p, q)), best.get((q, p))]
                v = [x for x in v if x is not None]
                if v:
                    ident.append(float(np.mean(v)))
            rows.append(
                {
                    "org_i": i,
                    "org_j": j,
                    "method": "single_copy_euNOG_proxy_for_BUSCO",
                    "n_single_copy_shared": len(shared),
                    "n_with_hit": len(ident),
                    "aai_sc_mean": float(np.mean(ident)) if ident else np.nan,
                    "aai_sc_median": float(np.median(ident)) if ident else np.nan,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT_SCOG, index=False)
    print(f"wrote {OUT_SCOG}: {len(df)} eukaryote pairs")


if __name__ == "__main__":
    steps = sys.argv[1:] or ["isolates"]
    for step in steps:
        {"isolates": isolates, "scog": scog}[step]()
