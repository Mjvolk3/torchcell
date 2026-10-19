# experiments/044-organism-similarity/scripts/05_trees.py
# [[experiments.044-organism-similarity.scripts.05_trees]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/044-organism-similarity/scripts/05_trees.py

"""The three trees of the 044 figure family.

(a) ``results/taxonomy_cladogram.nwk``: NCBI taxonomy topology over the 34 organisms'
    taxids (ete3 ``NCBITaxa`` over a pinned ``taxdump.tar.gz`` under
    ``$DATA_ROOT/data/organism_similarity/taxonomy/``). Leaves are organism keys; two
    organisms sharing a taxid (SD108 and CBS573, both 4909) become two leaves under that
    node. Also writes ``results/organism_lineage.csv`` (bacteria / Saccharomycotina /
    Saccharomyces / Mammalia flags plus the lineage names) for the pair grouping.
(b) ``results/aai_nj.nwk``: neighbor joining (Biopython) on ``d = -ln(aai_mean / 100)``
    from ``results/pairwise_similarity.csv``, rooted with the bacteria as outgroup.
(c) ``results/isolates_nj.nwk``: rapidnj on Peter et al. 2018's
    ``1011DistanceMatrixBasedOnSNPs.tab.gz`` (sha256 pinned; values are the percentage of
    non-identical bases from SNPs, heterozygous differences half-weighted, per Peter's
    ``filesDescription.txt``). Clade labels come from Supplementary Table 1
    (``41586_2018_30_MOESM3_ESM.xls`` in the ``peter2018_1011_assemblies`` tier set,
    byte-identical to the mirror's ``si/si3.xls``; the OCR'd ``si/si1.md`` does not carry
    the table): ``results/isolate_clades.csv``.

Run from the repo root with the orgsim env (ete3, Biopython, rapidnj).
"""

from __future__ import annotations

import gzip
import json
import math
import os
import os.path as osp
import subprocess
import sys
import urllib.request

import pandas as pd
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from orgsim_manifest import (  # noqa: E402
    DATA_ROOT,
    ORGSIM_BIN,
    RESULTS_DIR,
    WORK_DIR,
    load_manifest,
    sha256_file,
)

TAX_DIR = osp.join(WORK_DIR, "taxonomy")
TAXDUMP = osp.join(TAX_DIR, "taxdump.tar.gz")
TAXDUMP_URL = "https://ftp.ncbi.nih.gov/pub/taxonomy/taxdump.tar.gz"
TAXA_DB = osp.join(TAX_DIR, "taxa.sqlite")
PETER_MATRIX = osp.join(
    DATA_ROOT, "data/peter2018/1011DistanceMatrixBasedOnSNPs.tab.gz"
)
PETER_MATRIX_SHA256 = "140da4e5193584c01e60c554a2ba5075a542d925be540afe7c7a92b7377af928"
PETER_TABLE_S1 = osp.join(
    DATA_ROOT,
    "torchcell-genomes/peter2018_1011_assemblies/41586_2018_30_MOESM3_ESM.xls",
)
PETER_TABLE_S1_SHA256 = (
    "b11de13b50bcf91bb2a40cdbd0f2f35372bdef6fc9a018d7538cf5e5eea7f273"
)
ISO_DIR = osp.join(WORK_DIR, "isolates_tree")

#: NCBI taxids of the lineage flags.
FLAGS = {
    "bacteria": 2,
    "fungi": 4751,
    "saccharomycotina": 147537,
    "saccharomyces": 4930,
    "mammalia": 40674,
}


def cladogram() -> None:
    """(a) NCBI taxonomy topology + lineage table."""
    from ete3 import NCBITaxa

    os.makedirs(TAX_DIR, exist_ok=True)
    if not osp.isfile(TAXDUMP):
        urllib.request.urlretrieve(TAXDUMP_URL, TAXDUMP)
    prov = {
        "source_url": TAXDUMP_URL,
        "sha256": sha256_file(TAXDUMP),
        "bytes": os.path.getsize(TAXDUMP),
        "retrieval_command": f"urllib.request.urlretrieve({TAXDUMP_URL!r}, ...)",
    }
    if not osp.isfile(TAXA_DB):
        ncbi = NCBITaxa(dbfile=TAXA_DB, taxdump_file=TAXDUMP)
    else:
        ncbi = NCBITaxa(dbfile=TAXA_DB)
    manifest = load_manifest()
    orgs = manifest.organisms
    taxids = sorted({o.taxid for o in orgs})
    tree = ncbi.get_topology(taxids, intermediate_nodes=False)
    by_taxid: dict[int, list[str]] = {}
    for o in orgs:
        by_taxid.setdefault(o.taxid, []).append(o.key)
    for node in list(tree.traverse()):
        tid = int(node.name)
        keys = by_taxid.get(tid)
        if not keys:
            continue
        if len(keys) == 1 and node.is_leaf():
            node.name = keys[0]
        else:
            for k in keys:
                node.add_child(name=k)
            node.name = str(tid)
    leaves = sorted(tree.get_leaf_names())
    assert leaves == sorted(o.key for o in orgs), set(leaves) ^ {o.key for o in orgs}
    tree.write(format=9, outfile=osp.join(RESULTS_DIR, "taxonomy_cladogram.nwk"))
    rows = []
    for o in orgs:
        lineage = ncbi.get_lineage(o.taxid)
        names = ncbi.get_taxid_translator(lineage)
        ranks = ncbi.get_rank(lineage)
        row = {
            "key": o.key,
            "taxid": o.taxid,
            "organism_name": o.organism_name,
            "color_group": o.groups[0],
            "groups": ";".join(o.groups),
            "has_proteome": o.proteome is not None,
        }
        for flag, tid in FLAGS.items():
            row[f"is_{flag}"] = tid in lineage
        for rank in ("phylum", "class", "order", "family", "genus"):
            hit = [names[t] for t in lineage if ranks[t] == rank]
            row[rank] = hit[0] if hit else None
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        osp.join(RESULTS_DIR, "organism_lineage.csv"), index=False
    )
    with open(osp.join(RESULTS_DIR, "taxonomy_provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2)
    print(f"cladogram: {len(leaves)} leaves; taxdump sha256 {prov['sha256']}")


def aai_nj() -> None:
    """(b) NJ on -ln(AAI), rooted on the bacteria."""
    pw = pd.read_csv(osp.join(RESULTS_DIR, "pairwise_similarity.csv"))
    lin = pd.read_csv(osp.join(RESULTS_DIR, "organism_lineage.csv")).set_index("key")
    keys = list(dict.fromkeys(list(pw.org_i) + list(pw.org_j)))
    assert pw["aai_mean"].notna().all(), pw[pw.aai_mean.isna()]
    d = {}
    for i, j, a in zip(pw.org_i, pw.org_j, pw.aai_mean):
        d[(i, j)] = d[(j, i)] = -math.log(a / 100.0)
    lower = [
        [d[(keys[r], keys[c])] for c in range(r)] + [0.0] for r in range(len(keys))
    ]
    tree = DistanceTreeConstructor().nj(DistanceMatrix(keys, lower))
    bacteria = [k for k in keys if bool(lin.loc[k, "is_bacteria"])]
    tree.root_with_outgroup(*[{"name": b} for b in bacteria])
    for clade in tree.get_nonterminals():
        clade.name = None
    Phylo.write(tree, osp.join(RESULTS_DIR, "aai_nj.nwk"), "newick")
    print(f"AAI NJ: {len(keys)} leaves, rooted on {len(bacteria)} bacteria")


def isolates() -> None:
    """(c) rapidnj on Peter's SNP distance matrix; clade table."""
    assert sha256_file(PETER_MATRIX) == PETER_MATRIX_SHA256
    assert sha256_file(PETER_TABLE_S1) == PETER_TABLE_S1_SHA256
    os.makedirs(ISO_DIR, exist_ok=True)
    with gzip.open(PETER_MATRIX, "rt") as fh:
        mat = pd.read_csv(fh, sep="\t", index_col=0)
    assert mat.index.name == "STD", mat.index.name
    assert list(mat.index) == list(mat.columns), "row and column order differ"
    assert mat.shape == (1011, 1011), mat.shape
    assert (mat.to_numpy() == mat.to_numpy().T).all()
    phy = osp.join(ISO_DIR, "peter1011.phy")
    with open(phy, "w") as fh:
        fh.write(f"{len(mat)}\n")
        for name, row in mat.iterrows():
            fh.write(name + " " + " ".join(f"{v:.9g}" for v in row.to_numpy()) + "\n")
    out = osp.join(RESULTS_DIR, "isolates_nj.nwk")
    cmd = [osp.join(ORGSIM_BIN, "rapidnj"), phy, "-i", "pd", "-o", "t", "-x", out]
    subprocess.run(cmd, check=True, capture_output=True)
    t = Phylo.read(out, "newick")
    assert len(t.get_terminals()) == 1011
    s1 = pd.read_excel(PETER_TABLE_S1, sheet_name="Table S1", header=3)
    s1 = s1[s1["Standardized name"].notna()].copy()
    assert len(s1) == 1011, len(s1)
    s1["clade_raw"] = s1["Clades"].astype("string").str.strip()
    s1["clade"] = s1["clade_raw"].str.replace(r"\s*\(subclade \d+\)$", "", regex=True)
    iso = pd.DataFrame({"isolate": mat.index})
    iso["code"] = iso["isolate"].str.removeprefix("SACE_")
    s1["code"] = s1["Standardized name"].str.removeprefix("SACE_")
    iso = iso.merge(
        s1[["code", "Isolate name", "Ecological origins", "clade_raw", "clade"]],
        on="code",
        how="left",
        validate="one_to_one",
    )
    iso.to_csv(osp.join(RESULTS_DIR, "isolate_clades.csv"), index=False)
    n_clade = int(iso["clade"].notna().sum())
    summary = {
        "matrix": PETER_MATRIX,
        "matrix_sha256": PETER_MATRIX_SHA256,
        "table_s1": PETER_TABLE_S1,
        "table_s1_sha256": PETER_TABLE_S1_SHA256,
        "rapidnj_command": " ".join(cmd),
        "n_isolates": int(len(iso)),
        "n_matched_to_table_s1": int(iso["Isolate name"].notna().sum()),
        "n_with_clade": n_clade,
        "n_clades": int(iso["clade"].nunique()),
    }
    with open(osp.join(RESULTS_DIR, "isolates_tree_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    steps = sys.argv[1:] or ["cladogram", "aai_nj", "isolates"]
    for step in steps:
        {"cladogram": cladogram, "aai_nj": aai_nj, "isolates": isolates}[step]()
