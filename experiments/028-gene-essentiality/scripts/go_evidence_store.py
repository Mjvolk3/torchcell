# experiments/028-gene-essentiality/scripts/go_evidence_store.py
# [[experiments.028-gene-essentiality.scripts.go_evidence_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/go_evidence_store

r"""Does GO leak essentiality through its mutant-phenotype evidence? GO stores cut by evidence code.

The GO multi-hot every essentiality GO model reads is built from the SGD GFF, which
carries no evidence codes, so annotations SGD inferred from a mutant's phenotype (IMP,
and for an essential gene usually a conditional allele) are in it. ``SCerevisiaeGraph.G_go``
holds SGD's per-annotation ``go_details`` with the evidence code, so the same vector can
be rebuilt with and without that evidence. One record per (gene, term) pair is stored
there, so a pair supported by more than one code carries one of them; a filter removes
the pairs whose stored code it names.

Stores, all built the one way (the pair's term and every ancestor through ``is_a`` and
``part_of`` in ``go.obo``, the three namespace roots dropped, a column kept when ten or
more genome genes carry it, the rule of ``go_term_hot.py``):

- ``all``: every G_go annotation, the control for the change of source from the GFF;
- ``no_imp``: the IMP pairs removed;
- ``no_mutant``: every mutant-derived code removed (IMP, IGI, HMP, HGI);
- ``random_matched``: as many pairs removed as ``no_mutant`` removes, drawn uniformly at
  a fixed seed from all pairs, the control that separates losing annotations from
  losing the mutant-derived ones;
- ``no_experimental``: only computational and author or curator statements kept (IEA,
  IBA, ISS, ISA, ISO, ISM, IKR, RCA, TAS, NAS, IC), every experimental code removed.

Beside the stores it reports each code's share of the labeled essential and
non-essential genes' annotations, and a fast read of each store on the released split:
L2 logistic regression on the multi-hot, C chosen on the validation genes, held-out
AUROC on the 195. The models the document reports (graph convolution over the union at
400 epochs) read the stores through ``graph_baselines.py --arms go400 --go-dir``.
Writes ``$DATA_ROOT/data/scerevisiae/go_term_hot_evidence_<name>/processed/go.pt`` (and
``go.terms.json``) per store and ``results/go_evidence_store.json``. Run from the repo
root::

    PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
        ~/miniconda3/envs/torchcell/bin/python \
        experiments/028-gene-essentiality/scripts/go_evidence_store.py
"""

from __future__ import annotations

import collections
import gzip
import json
import os
import os.path as osp
import time
from typing import cast

import numpy as np
import torch
from dotenv import load_dotenv
from goatools.obo_parser import GODag, GOTerm
from pydantic import BaseModel, ConfigDict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data, InMemoryDataset

from torchcell.datasets.go_term_hot import GOTermEmbeddingDataset
from torchcell.graph import SCerevisiaeGraph
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
RELEASED = osp.join(RESULTS, "splits", "ess_fcl_val0.2_s0.json.gz")
SUPER_ROOT = "GO:ROOT"
MIN_GENES = 10
MUTANT = {"IMP", "IGI", "HMP", "HGI"}
NON_EXPERIMENTAL = {"IEA", "IBA", "ISS", "ISA", "ISO", "ISM", "IKR", "RCA", "TAS", "NAS", "IC"}
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]


class StoreRead(BaseModel):
    """One evidence-cut store: its size and the fast read on the released split."""

    model_config = ConfigDict(extra="forbid")

    name: str
    store_dir: str
    n_pairs_kept: int
    n_pairs_removed: int
    n_columns: int
    n_labeled_genes_without_a_column: int
    logistic_c: float
    logistic_val_auroc: float
    logistic_test_auroc: float


def store_dir(name: str) -> str:
    return osp.join(DATA_ROOT, f"data/scerevisiae/go_term_hot_evidence_{name}/processed")


def write_store(
    name: str,
    pairs: list[tuple[str, str]],
    genes: list[str],
    dag: GODag,
    memo: dict[str, set[str]],
) -> tuple[np.ndarray, list[str]]:
    """Propagate the kept pairs, cut the columns, write go.pt in go_term_hot's form."""
    gene_terms: dict[str, set[str]] = {g: set() for g in genes}
    for gene, term in pairs:
        if gene not in gene_terms:
            continue
        tid = cast(GOTerm, dag[term]).item_id
        lit = {tid} | GOTermEmbeddingDataset.ancestors(dag, tid, memo)
        gene_terms[gene] |= {t for t in lit if cast(GOTerm, dag[t]).level > 0}
    counts = collections.Counter(t for ts in gene_terms.values() for t in ts)
    columns = sorted(t for t, n in counts.items() if n >= MIN_GENES)
    index = {t: i for i, t in enumerate(columns)}
    hot = np.zeros((len(genes), len(columns)), dtype=np.float32)
    for r, g in enumerate(genes):
        for t in gene_terms[g]:
            if t in index:
                hot[r, index[t]] = 1.0
    out = store_dir(name)
    os.makedirs(out, exist_ok=True)
    data_list = []
    for r, g in enumerate(genes):
        d = Data(id=g)
        d.embeddings = {"go": torch.from_numpy(hot[r]).unsqueeze(0)}
        data_list.append(d)
    torch.save(InMemoryDataset.collate(data_list), osp.join(out, "go.pt"))
    with open(osp.join(out, "go.terms.json"), "w") as fh:
        json.dump(
            {
                "model_name": "go",
                "evidence_cut": name,
                "propagate": True,
                "min_genes": MIN_GENES,
                "n_genes": len(genes),
                "n_columns": len(columns),
                "terms": [
                    {
                        "term_id": t,
                        "name": cast(GOTerm, dag[t]).name,
                        "namespace": cast(GOTerm, dag[t]).namespace,
                        "n_genes": counts[t],
                    }
                    for t in columns
                ],
            },
            fh,
        )
    return hot, columns


def main() -> None:
    assert os.environ.get("PYTHONHASHSEED") == "0", "PYTHONHASHSEED must be 0"
    started = time.time()
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )
    dag = GODag(
        osp.join(genome.go_root, "go.obo"), optional_attrs={"relationship"}, prt=None
    )
    # Every direct (gene, term, code) triple G_go holds, terms resolved through the OBO.
    triples: list[tuple[str, str, str]] = []
    n_unresolved = 0
    for node, d in graph.G_go.nodes(data=True):
        if node == SUPER_ROOT or not d.get("genes"):
            continue
        if node not in dag:
            n_unresolved += len(d["genes"])
            continue
        for gene, info in d["genes"].items():
            triples.append((gene, node, info["go_details"]["experiment"]["display_name"]))
    with gzip.open(RELEASED, "rt") as fh:
        split = json.load(fh)
    labels = {str(k): int(v) for k, v in split["labels"].items()}
    code_share: dict[str, dict[str, float]] = {}
    for cls, name in ((1, "essential"), (0, "nonessential")):
        c = collections.Counter(code for g, _, code in triples if labels.get(g) == cls)
        total = sum(c.values())
        code_share[name] = {k: v / total for k, v in c.most_common()}
        code_share[name]["_n_pairs"] = float(total)
    genes = sorted(genome.gene_set)
    rng = np.random.default_rng(0)
    n_mutant = sum(1 for *_, code in triples if code in MUTANT)
    drop_random = set(rng.choice(len(triples), size=n_mutant, replace=False).tolist())
    cuts = {
        "all": [True] * len(triples),
        "no_imp": [code != "IMP" for *_, code in triples],
        "no_mutant": [code not in MUTANT for *_, code in triples],
        "random_matched": [i not in drop_random for i in range(len(triples))],
        "no_experimental": [code in NON_EXPERIMENTAL for *_, code in triples],
    }
    tr = [g for g in split["genes"]["train"]]
    va = [g for g in split["genes"]["val"]]
    te = [g for g in split["genes"]["test"]]
    row = {g: i for i, g in enumerate(genes)}
    memo: dict[str, set[str]] = {}
    reads: list[StoreRead] = []
    for name, keep in cuts.items():
        pairs = [(g, t) for (g, t, _), k in zip(triples, keep, strict=True) if k]
        hot, columns = write_store(name, pairs, genes, dag, memo)

        def rows_of(gs: list[str]) -> tuple[np.ndarray, np.ndarray]:
            gs = [g for g in gs if g in row]
            return hot[[row[g] for g in gs]], np.array([labels[g] for g in gs])

        xtr, ytr = rows_of(tr)
        xva, yva = rows_of(va)
        xte, yte = rows_of(te)
        best = max(
            C_GRID,
            key=lambda c: roc_auc_score(
                yva,
                LogisticRegression(C=c, max_iter=5000, class_weight="balanced")
                .fit(xtr, ytr)
                .decision_function(xva),
            ),
        )
        m = LogisticRegression(C=best, max_iter=5000, class_weight="balanced").fit(
            np.vstack([xtr, xva]), np.concatenate([ytr, yva])
        )
        val = roc_auc_score(
            yva,
            LogisticRegression(C=best, max_iter=5000, class_weight="balanced")
            .fit(xtr, ytr)
            .decision_function(xva),
        )
        read = StoreRead(
            name=name,
            store_dir=store_dir(name),
            n_pairs_kept=len(pairs),
            n_pairs_removed=len(triples) - len(pairs),
            n_columns=len(columns),
            n_labeled_genes_without_a_column=int(
                sum(1 for g in labels if g in row and hot[row[g]].sum() == 0)
            ),
            logistic_c=float(best),
            logistic_val_auroc=float(val),
            logistic_test_auroc=float(roc_auc_score(yte, m.decision_function(xte))),
        )
        reads.append(read)
        print(
            f"{name:16s} pairs {read.n_pairs_kept:6d} (-{read.n_pairs_removed:5d}) "
            f"columns {read.n_columns:5d} labeled w/o column "
            f"{read.n_labeled_genes_without_a_column:3d} | logistic C {best} val "
            f"{val:.4f} test {read.logistic_test_auroc:.4f}",
            flush=True,
        )
    out = {
        "generated_by": "experiments/028-gene-essentiality/scripts/go_evidence_store.py",
        "split": RELEASED,
        "n_triples": len(triples),
        "n_pairs_unresolved_in_obo": n_unresolved,
        "code_counts": dict(collections.Counter(code for *_, code in triples).most_common()),
        "code_share_by_class": code_share,
        "mutant_codes": sorted(MUTANT),
        "non_experimental_codes": sorted(NON_EXPERIMENTAL),
        "statistic": (
            "held-out AUROC on the released 195 genes of L2 logistic regression on the "
            "store's multi-hot, C selected on the validation genes, refit on train+val"
        ),
        "stores": [r.model_dump() for r in reads],
        "seconds": time.time() - started,
    }
    with open(osp.join(RESULTS, "go_evidence_store.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(osp.join(RESULTS, "go_evidence_store.json"))


if __name__ == "__main__":
    main()
