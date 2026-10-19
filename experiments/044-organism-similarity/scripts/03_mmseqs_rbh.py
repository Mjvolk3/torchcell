# experiments/044-organism-similarity/scripts/03_mmseqs_rbh.py
# [[experiments.044-organism-similarity.scripts.03_mmseqs_rbh]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/044-organism-similarity/scripts/03_mmseqs_rbh.py

"""All-vs-all mmseqs2 search over the per-gene proteomes, then reciprocal best hits.

1. Concatenate every ``proteomes/<key>.faa`` (headers ``<key>|<protein_id>``) into one
   FASTA and run ``mmseqs easy-search`` of it against itself with ``-s 5.7 -e 1e-5
   --cov-mode 0 -c 0.5 --max-seqs 4000`` (both query and target covered >= 50%), output
   ``query,target,pident,alnlen,qcov,tcov,evalue,bits``.
2. Best hit per (query protein, target organism) by bitscore (ties: smaller target id),
   hits within the same organism excluded. A reciprocal best hit (RBH) is a pair (a, b)
   where b is a's best hit in b's organism and a is b's best hit in a's organism.
3. Per unordered organism pair: ``n_rbh``, ``n_i`` and ``n_j`` (per-gene proteome sizes),
   ``ortholog_fraction = n_rbh / min(n_i, n_j)``, ``ortholog_jaccard = n_rbh / (n_i + n_j
   - n_rbh)``, ``aai_mean`` and ``aai_median`` = mmseqs percent identity over the RBH
   pairs (the CompareM AAI definition). Identity of an RBH pair is the mean of the two
   directions' pident.

Writes ``results/pairwise_similarity.csv`` and the RBH pair table
``$DATA_ROOT/data/organism_similarity/rbh_pairs.parquet``; prints the sanity checks.

    python 03_mmseqs_rbh.py search   # under slurm (gh_03_mmseqs_rbh.slurm)
    python 03_mmseqs_rbh.py rbh
"""

from __future__ import annotations

import os
import os.path as osp
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from orgsim_manifest import (  # noqa: E402
    ORGSIM_BIN,
    RESULTS_DIR,
    WORK_DIR,
    load_manifest,
)

MMSEQS = osp.join(ORGSIM_BIN, "mmseqs")
SEARCH_DIR = osp.join(WORK_DIR, "mmseqs")
ALL_FAA = osp.join(SEARCH_DIR, "all_proteomes.faa")
HITS_TSV = osp.join(SEARCH_DIR, "all_vs_all.m8")
HITS_PARQUET = osp.join(SEARCH_DIR, "all_vs_all.parquet")
RBH_PARQUET = osp.join(WORK_DIR, "rbh_pairs.parquet")
PAIRWISE_CSV = osp.join(RESULTS_DIR, "pairwise_similarity.csv")
THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", "48"))
FORMAT = "query,target,pident,alnlen,qcov,tcov,evalue,bits"
SEARCH_ARGS = [
    "-s",
    "5.7",
    "-e",
    "1e-5",
    "--cov-mode",
    "0",
    "-c",
    "0.5",
    "--max-seqs",
    "4000",
]


def search() -> None:
    """Concatenate the proteomes and run the all-vs-all search."""
    manifest = load_manifest()
    os.makedirs(SEARCH_DIR, exist_ok=True)
    n = 0
    with open(ALL_FAA, "w") as out:
        for org in manifest.present():
            assert org.prep is not None
            with open(org.prep.output.path) as fh:
                for line in fh:
                    n += line.startswith(">")
                    out.write(line)
    print(
        f"concatenated {len(manifest.present())} proteomes, {n:,} proteins", flush=True
    )
    cmd = [
        MMSEQS,
        "easy-search",
        ALL_FAA,
        ALL_FAA,
        HITS_TSV,
        osp.join(SEARCH_DIR, "tmp"),
        *SEARCH_ARGS,
        "--format-output",
        FORMAT,
        "--threads",
        str(THREADS),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print("search done", flush=True)


def load_hits() -> pd.DataFrame:
    """The m8 table as a typed frame (cached to parquet)."""
    if osp.isfile(HITS_PARQUET) and osp.getmtime(HITS_PARQUET) > osp.getmtime(HITS_TSV):
        return pd.read_parquet(HITS_PARQUET)
    hits = pd.read_csv(
        HITS_TSV,
        sep="\t",
        header=None,
        names=FORMAT.split(","),
        dtype={
            "query": str,
            "target": str,
            "pident": np.float32,
            "alnlen": np.int32,
            "qcov": np.float32,
            "tcov": np.float32,
            "evalue": np.float64,
            "bits": np.float32,
        },
        engine="pyarrow",
    )
    hits.to_parquet(HITS_PARQUET)
    return hits


def rbh() -> None:
    """Reciprocal best hits and the per-pair similarity table."""
    manifest = load_manifest()
    orgs = manifest.present()
    keys = [o.key for o in orgs]
    size = {o.key: o.prep.n_proteins_prepped for o in orgs if o.prep is not None}
    hits = load_hits()
    print(f"hits: {len(hits):,}", flush=True)
    hits = hits[hits["query"] != hits["target"]]
    hits["qorg"] = hits["query"].str.split("|", n=1).str[0].astype("category")
    hits["torg"] = hits["target"].str.split("|", n=1).str[0].astype("category")
    hits = hits[hits["qorg"].astype(str) != hits["torg"].astype(str)]
    print(f"between-organism hits: {len(hits):,}", flush=True)
    hits = hits.sort_values(
        ["query", "torg", "bits", "target"], ascending=[True, True, False, True]
    )
    best = hits.drop_duplicates(["query", "torg"], keep="first")[
        ["query", "target", "qorg", "torg", "pident", "bits", "qcov", "tcov"]
    ]
    print(f"best hits: {len(best):,}", flush=True)
    fwd = best.set_index(["query", "target"])
    rev = best.rename(columns={"query": "target", "target": "query"}).set_index(
        ["query", "target"]
    )
    common = fwd.index.intersection(rev.index)
    pairs = fwd.loc[common].reset_index()
    pairs["pident_rev"] = rev.loc[common, "pident"].to_numpy()
    pairs["identity"] = (pairs["pident"] + pairs["pident_rev"]) / 2.0
    pairs["qorg"] = pairs["qorg"].astype(str)
    pairs["torg"] = pairs["torg"].astype(str)
    order = {k: i for i, k in enumerate(keys)}
    pairs = pairs[pairs["qorg"].map(order) < pairs["torg"].map(order)].reset_index(
        drop=True
    )
    pairs = pairs.rename(
        columns={
            "qorg": "org_i",
            "torg": "org_j",
            "query": "prot_i",
            "target": "prot_j",
        }
    )
    pairs[
        [
            "org_i",
            "org_j",
            "prot_i",
            "prot_j",
            "pident",
            "pident_rev",
            "identity",
            "bits",
            "qcov",
            "tcov",
        ]
    ].to_parquet(RBH_PARQUET)
    print(f"RBH pairs: {len(pairs):,} -> {RBH_PARQUET}", flush=True)

    rows = []
    grouped = {k: g for k, g in pairs.groupby(["org_i", "org_j"])}
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            i, j = keys[a], keys[b]
            g = grouped.get((i, j))
            n_rbh = 0 if g is None else len(g)
            ni, nj = size[i], size[j]
            rows.append(
                {
                    "org_i": i,
                    "org_j": j,
                    "n_rbh": n_rbh,
                    "n_i": ni,
                    "n_j": nj,
                    "ortholog_fraction": n_rbh / min(ni, nj),
                    "ortholog_jaccard": n_rbh / (ni + nj - n_rbh),
                    "aai_mean": float(g["identity"].mean()) if n_rbh else np.nan,
                    "aai_median": float(g["identity"].median()) if n_rbh else np.nan,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(PAIRWISE_CSV, index=False)
    print(f"wrote {PAIRWISE_CSV}: {len(df)} pairs", flush=True)
    sanity(df)


def sanity(df: pd.DataFrame) -> None:
    """Print the plan's sanity pairs."""

    def get(i: str, j: str) -> pd.Series:
        m = ((df.org_i == i) & (df.org_j == j)) | ((df.org_i == j) & (df.org_j == i))
        assert m.sum() == 1, (i, j)
        return df[m].iloc[0]

    checks = [
        ("hsap_grch38", "ptro", "about 99%"),
        ("hsap_grch38", "mmul", "about 93%"),
        ("hsap_grch38", "mmus_c57bl6j", "about 85%"),
        ("hsap_grch38", "ggal", "about 75%"),
        ("scer_s288c", "spar_cbs432", "about 90%"),
        ("scer_s288c", "seub_fm1318", "below S. paradoxus"),
        ("scer_s288c", "ecoli_mg1655", "well below 50%, small ortholog fraction"),
        ("pkud_sd108", "pkud_cbs573", "near 100%"),
    ]
    print("\nSANITY CHECKS (AAI = mean RBH percent identity)")
    for i, j, expect in checks:
        r = get(i, j)
        print(
            f"  {i:>14} vs {j:<14} AAI {r.aai_mean:6.2f} (median {r.aai_median:6.2f}) "
            f"n_rbh {r.n_rbh:>6} ortholog_fraction {r.ortholog_fraction:.3f}   "
            f"expected {expect}"
        )


if __name__ == "__main__":
    step = sys.argv[1]
    {"search": search, "rbh": rbh}[step]()
