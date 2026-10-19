# experiments/044-organism-similarity/scripts/02_proteome_prep.py
# [[experiments.044-organism-similarity.scripts.02_proteome_prep]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/044-organism-similarity/scripts/02_proteome_prep.py

"""One protein per gene for every organism with a proteome.

Isoforms inflate ortholog counts (human RefSeq has ~120k proteins for ~20k genes), so
before any comparison each proteome is reduced to its LONGEST protein per gene. The gene
of a protein is found in the GFF: the CDS carrying ``protein_id=<id>`` (NCBI) or the
feature whose ``ID`` is the FASTA id (SD108, whose FASTA ids are mRNA ids), then its
``Parent`` chain is walked up to the first ``gene`` or ``pseudogene`` feature. The gene
is keyed by its NCBI ``GeneID`` (``Dbxref``) when the feature carries one, else by the
feature ID: GRCh38.p14 annotates the same GeneID on the primary assembly and on alt-locus
and patch scaffolds (1,239 human protein-coding GeneIDs appear twice, a few up to 47
times), and keying by feature ID would count each copy as a gene. Ties in length keep the
lexicographically smallest protein id, so the output is deterministic.

The S288C SGD ``orf_trans_all`` FASTA has no GFF here: it is one translation per
systematic ORF name already, so it passes through with the rule stated. A protein whose
gene cannot be found in the GFF is not silently kept or dropped: it is counted
(``n_proteins_without_gene``) and kept as its own gene, and a count above 1% of the
proteome fails the run.

Writes ``$DATA_ROOT/data/organism_similarity/proteomes/<key>.faa`` with headers
``<key>|<protein_id>`` and fills ``prep`` in ``results/organisms.json``.
"""

from __future__ import annotations

import gzip
import os
import os.path as osp
import sys
from urllib.parse import unquote

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from orgsim_manifest import (  # noqa: E402
    PROTEOME_DIR,
    FileRecord,
    OrganismRecord,
    ProteomePrep,
    load_manifest,
    sha256_file,
    write_manifest,
)

GENE_TYPES = {"gene", "pseudogene"}


def read_fasta(path: str) -> dict[str, str]:
    """``{first header token: sequence}``; a duplicate id raises."""
    opener = gzip.open if path.endswith(".gz") else open
    seqs: dict[str, str] = {}
    name: str | None = None
    buf: list[str] = []
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(buf)
                name = line[1:].split()[0]
                assert name not in seqs, f"duplicate FASTA id {name} in {path}"
                buf = []
            else:
                buf.append(line.strip())
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs


def parse_attrs(field: str) -> dict[str, str]:
    """GFF3 column 9 as a dict (values URL-decoded)."""
    out: dict[str, str] = {}
    for kv in field.strip().split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k] = unquote(v)
    return out


def gene_map(gff_path: str, ids: set[str]) -> dict[str, str]:
    """``{protein id: gene id}`` for the ids the GFF can place."""
    opener = gzip.open if gff_path.endswith(".gz") else open
    ftype: dict[str, str] = {}
    parent: dict[str, str] = {}
    gene_key: dict[str, str] = {}
    protein_parent: dict[str, str] = {}
    with opener(gff_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9:
                continue
            attrs = parse_attrs(cols[8])
            fid = attrs.get("ID")
            par = attrs.get("Parent")
            if fid is not None and fid not in ftype:
                ftype[fid] = cols[2]
                if cols[2] in GENE_TYPES:
                    xref = [
                        x
                        for x in attrs.get("Dbxref", "").split(",")
                        if x.startswith("GeneID:")
                    ]
                    gene_key[fid] = xref[0] if xref else fid
                if par is not None:
                    parent[fid] = par.split(",")[0]
            if cols[2] == "CDS" and "protein_id" in attrs and par is not None:
                protein_parent.setdefault(attrs["protein_id"], par.split(",")[0])
    out: dict[str, str] = {}
    for pid in ids:
        node = protein_parent.get(pid, pid if pid in ftype else None)
        seen = 0
        while node is not None and ftype.get(node) not in GENE_TYPES and seen < 10:
            node = parent.get(node)
            seen += 1
        if node is not None and ftype.get(node) in GENE_TYPES:
            out[pid] = gene_key[node]
    return out


def prep(org: OrganismRecord) -> ProteomePrep:
    """Reduce one proteome to the longest protein per gene and write it."""
    src = org.proteome
    assert src is not None
    seqs = read_fasta(src.protein_fasta.path)
    assert len(seqs) == src.n_proteins_raw, (org.key, len(seqs), src.n_proteins_raw)
    if src.gff is None:
        assert src.source == "tier" and org.key == "scer_s288c", org.key
        rule = (
            "SGD orf_trans_all: one translation per systematic ORF name, passed through "
            "unchanged (no isoforms to collapse)."
        )
        keep = sorted(seqs)
        n_no_gene = 0
    else:
        rule = (
            "longest protein per GFF gene (CDS protein_id or feature ID, Parent chain to "
            "the first gene/pseudogene feature, gene keyed by Dbxref GeneID when "
            "present); ties keep the smallest protein id."
        )
        g = gene_map(src.gff.path, set(seqs))
        n_no_gene = len(seqs) - len(g)
        assert n_no_gene <= 0.01 * len(seqs), (
            f"{org.key}: {n_no_gene} of {len(seqs)} proteins have no gene in the GFF"
        )
        best: dict[str, str] = {}
        for pid in sorted(seqs):
            gene = g.get(pid, f"__nogene__{pid}")
            cur = best.get(gene)
            if cur is None or len(seqs[pid]) > len(seqs[cur]):
                best[gene] = pid
        keep = sorted(best.values())
    os.makedirs(PROTEOME_DIR, exist_ok=True)
    out = osp.join(PROTEOME_DIR, f"{org.key}.faa")
    with open(out, "w") as fh:
        for pid in keep:
            seq = seqs[pid].rstrip("*")
            fh.write(f">{org.key}|{pid}\n")
            for i in range(0, len(seq), 80):
                fh.write(seq[i : i + 80] + "\n")
    return ProteomePrep(
        rule=rule,
        n_proteins_raw=len(seqs),
        n_proteins_prepped=len(keep),
        n_proteins_without_gene=n_no_gene,
        output=FileRecord(
            path=out, sha256=sha256_file(out), bytes=os.path.getsize(out)
        ),
    )


def main() -> None:
    manifest = load_manifest()
    for org in manifest.present():
        org.prep = prep(org)
        p = org.prep
        print(
            f"{org.key:<16} raw {p.n_proteins_raw:>7,} -> per-gene "
            f"{p.n_proteins_prepped:>6,} (no gene: {p.n_proteins_without_gene})",
            flush=True,
        )
    # Plausibility: mammals and the chicken collapse to ~20k genes; microbes stay put.
    for org in manifest.present():
        p = org.prep
        assert p is not None
        if "mammal" in org.groups or "outgroup" in org.groups:
            assert 15_000 <= p.n_proteins_prepped <= 25_000, (org.key, p)
            assert p.n_proteins_raw >= 1.5 * p.n_proteins_prepped, (org.key, p)
        else:
            assert 1_500 <= p.n_proteins_prepped <= 10_000, (org.key, p)
    write_manifest(manifest)


if __name__ == "__main__":
    main()
