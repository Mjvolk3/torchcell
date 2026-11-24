# experiments/050-ecoli-benchmark/scripts/cluster_proteins.py
# [[experiments.050-ecoli-benchmark.scripts.cluster_proteins]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/050-ecoli-benchmark/scripts/cluster_proteins
"""Cluster the proteins of each genome by sequence identity with MMseqs2.

For each of the two genomes and each identity threshold (30 % and 50 %), every
protein-coding gene is searched against every other with ``mmseqs easy-search``
(coverage 0.8 of both sequences, sensitivity 7.5, E-value 1e-3, one thread). A cluster is
a connected component of the hit graph (single linkage), so two proteins in different
clusters have no reported hit at or above the threshold. The sequence-similarity splits
keep a cluster whole.

Writes ``experiments/050-ecoli-benchmark/results/clusters/<assembly_set>.seqid<NN>.tsv``
(``locus_tag``, ``cluster``; a cluster is named by its first locus tag in sorted order)
and ``clusters.json`` with the MMseqs2 version, the binary's sha256 and cluster counts.

    python experiments/050-ecoli-benchmark/scripts/cluster_proteins.py --mmseqs <binary>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import subprocess
import tempfile

import pandas as pd
from dotenv import load_dotenv
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from torchcell.sequence.genome.standard import load_genome

ASSEMBLY_SETS = ("ecoli_K12_BW25113_ASM75055v1", "ecoli_K12_MG1655_ASM584v2")
THRESHOLDS = (30, 50)
OUT = "experiments/050-ecoli-benchmark/results/clusters"


def main() -> None:
    """Cluster both genomes at both thresholds and write the tables."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mmseqs", required=True, help="path of the mmseqs binary")
    args = parser.parse_args()
    load_dotenv(".env")
    os.makedirs(OUT, exist_ok=True)
    version = subprocess.check_output([args.mmseqs, "version"], text=True).strip()
    with open(args.mmseqs, "rb") as handle:
        binary_sha256 = hashlib.sha256(handle.read()).hexdigest()
    summary: dict[str, object] = {
        "mmseqs_version": version,
        "mmseqs_binary_sha256": binary_sha256,
        "command": "mmseqs easy-search <fasta> <fasta> <hits> <tmp> --min-seq-id <t> "
        "-c 0.8 --cov-mode 0 -s 7.5 -e 1e-3 --max-seqs 5000 --threads 1, then connected "
        "components of the hit graph",
        "clusters": {},
    }
    counts: dict[str, dict[str, int]] = {}
    for assembly_set in ASSEMBLY_SETS:
        genome = load_genome(assembly_set)
        proteins = {}
        for locus in sorted(genome.gene_set):
            protein = genome[locus].protein
            if protein is not None and len(protein.seq) > 0:
                proteins[locus] = str(protein.seq).rstrip("*")
        for threshold in THRESHOLDS:
            with tempfile.TemporaryDirectory() as work:
                fasta = osp.join(work, "proteins.faa")
                with open(fasta, "w", encoding="utf-8") as handle:
                    for locus, sequence in proteins.items():
                        handle.write(f">{locus}\n{sequence}\n")
                hits = osp.join(work, "hits.m8")
                subprocess.run(
                    [
                        args.mmseqs,
                        "easy-search",
                        fasta,
                        fasta,
                        hits,
                        osp.join(work, "tmp"),
                        "--min-seq-id",
                        str(threshold / 100),
                        "-c",
                        "0.8",
                        "--cov-mode",
                        "0",
                        "-s",
                        "7.5",
                        "-e",
                        "1e-3",
                        "--max-seqs",
                        "5000",
                        "--threads",
                        "1",
                        "-v",
                        "1",
                    ],
                    check=True,
                )
                pairs = pd.read_csv(
                    hits, sep="\t", header=None, usecols=[0, 1], dtype=str
                )
            order = sorted(proteins)
            index = {locus: i for i, locus in enumerate(order)}
            graph = coo_matrix(
                (
                    [1] * len(pairs),
                    ([index[q] for q in pairs[0]], [index[t] for t in pairs[1]]),
                ),
                shape=(len(order), len(order)),
            )
            _, component = connected_components(graph, directed=False)
            first: dict[int, str] = {}
            for locus, c in zip(order, component):
                first.setdefault(int(c), locus)
            table = pd.DataFrame(
                {"locus_tag": order, "cluster": [first[int(c)] for c in component]}
            )
            table = table[["locus_tag", "cluster"]].sort_values("locus_tag")
            if set(table["locus_tag"]) != set(proteins):
                raise SystemExit("the clustering does not cover every protein")
            name = f"{assembly_set}.seqid{threshold}"
            table.to_csv(osp.join(OUT, f"{name}.tsv"), sep="\t", index=False)
            sizes = table["cluster"].value_counts()
            counts[name] = {
                "proteins": len(table),
                "clusters": int(len(sizes)),
                "proteins_in_clusters_of_two_or_more": int(sizes[sizes > 1].sum()),
                "largest_cluster": int(sizes.max()),
            }
            print(name, counts[name])
    summary["clusters"] = counts
    with open(osp.join(OUT, "clusters.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
