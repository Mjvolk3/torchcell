# experiments/028-gene-essentiality/scripts/go006_store.py
# [[experiments.028-gene-essentiality.scripts.go006_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/go006_store

r"""The Gene Ontology of the original DCell run (experiment 006) as a multi-hot store.

028's two GO models (``graph_baselines.py --arms go400`` and ``dcell_style_baseline.py``)
read ``$DATA_ROOT/data/scerevisiae/go_term_hot/processed/go.pt``: the GFF's GO ids per
gene, propagated through ``is_a`` and ``part_of``, all three namespaces, a term kept when
ten or more genes carry it, 2,362 columns, no evidence-code filter.

Experiment 006 (``experiments/006-kuzmin-tmi/scripts/dcell.py``) built its hierarchy from
``SCerevisiaeGraph.G_go`` (the SGD per-gene ``go_details``, one node per directly
annotated term) and then applied, with no date filter configured:

1. ``filter_go_IGI``: drop every (gene, term) annotation whose stored evidence is IGI, and
   a term left with no gene;
2. ``filter_redundant_terms``: drop a term whose direct gene set equals a parent's;
3. ``filter_by_contained_genes(n=4, gene_set=genome.gene_set)``: drop a term with fewer
   than four genes on it and its descendants in the graph.

This script rebuilds that term set and writes it in the SAME FORM as the current store, so
the two models can read it unchanged: a genes x terms float multi-hot ``go.pt`` in the
current store's gene order and a ``go.terms.json`` sidecar. Membership is PROPAGATED, as
in the current store: a gene lights a kept term when its IGI-filtered direct annotations
name that term or any descendant of it through ``is_a`` and ``part_of`` in ``go.obo``
(``G_go`` node ``gene_set`` holds DIRECT annotations only, so the propagation is done
here). ``G_go.pkl`` was pickled from another GO release than the ``go.obo`` on disk, so a
few of its edges are not an ancestry relation in the OBO; they are counted, and the OBO
is the one ontology used, because both 028 models wire and propagate with it. An
annotation on a term that filters 2 or 3 removed still lights the kept ancestors. The
three namespace roots and torchcell's ``GO:ROOT`` super node are not columns, as in the
current store; ``dcell_style_baseline.py`` adds them itself.

What differs between the two stores is therefore the annotation source (SGD
``go_details`` against the GFF), the IGI filter, and the term set (directly annotated
terms, redundancy filter, four-gene floor against a ten-gene floor on all propagated
terms).

Writes ``$DATA_ROOT/data/scerevisiae/go_term_hot_dcell006/processed/go.pt`` and
``go.terms.json`` (skipped when both exist) and ``results/go006_store.json``.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
        python experiments/028-gene-essentiality/scripts/go006_store.py
"""

from __future__ import annotations

import gzip
import json
import os
import os.path as osp
from typing import cast

import networkx as nx
import numpy as np
import torch
from dotenv import load_dotenv
from goatools.obo_parser import GODag, GOTerm
from pydantic import BaseModel, ConfigDict
from torch_geometric.data import Data, InMemoryDataset

from torchcell.datasets.go_term_hot import GOTermEmbeddingDataset
from torchcell.graph import SCerevisiaeGraph
from torchcell.graph.graph import (
    filter_by_contained_genes,
    filter_go_IGI,
    filter_redundant_terms,
)
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
CURRENT_DIR = osp.join(DATA_ROOT, "data/scerevisiae/go_term_hot/processed")
STORE_DIR = osp.join(DATA_ROOT, "data/scerevisiae/go_term_hot_dcell006/processed")
RELEASED = osp.join(RESULTS, "splits", "ess_fcl_val0.2_s0.json.gz")
SUPER_ROOT = "GO:ROOT"
MIN_GENES = 4


class StoreReport(BaseModel):
    """The counts that say how the 006 store was cut and how it differs from 028's."""

    model_config = ConfigDict(extra="forbid")
    store_dir: str
    current_store_dir: str
    annotation_source: str
    filters: list[str]
    n_nodes_G_go: int
    n_nodes_after_igi: int
    n_nodes_after_redundant: int
    n_nodes_after_containment: int
    n_direct_annotations_before_igi: int
    n_direct_annotations_after_igi: int
    n_igi_annotations_removed: int
    n_non_term_nodes_excluded: int
    non_term_nodes_excluded: list[str]
    n_columns: int
    n_columns_current: int
    n_terms_shared_with_current: int
    n_terms_only_006: int
    n_terms_only_current: int
    n_columns_by_namespace: dict[str, int]
    obo_version: str
    n_edges_G_go: int
    n_edges_G_go_not_ancestry_in_obo: int
    n_columns_under_min_genes: int
    min_genes_per_column: int
    n_genes: int
    n_genes_annotated: int
    n_genes_annotated_current: int
    n_labeled_genes: int
    mean_columns_per_labeled_gene_006: float
    mean_columns_per_labeled_gene_current: float
    n_labeled_genes_without_a_column_006: int
    n_labeled_genes_without_a_column_current: int


def n_annotations(g: nx.DiGraph) -> int:
    """Direct (gene, term) annotation pairs on the graph's nodes."""
    return sum(len(d["gene_set"]) for _, d in g.nodes(data=True) if "gene_set" in d)


def build() -> dict:
    """Cut G_go as 006 did, propagate, write the store, and return the sidecar."""
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
    g_go = graph.G_go.copy()
    g_igi = filter_go_IGI(g_go)
    g_red = filter_redundant_terms(g_igi)
    g_kept = filter_by_contained_genes(g_red, n=MIN_GENES, gene_set=genome.gene_set)
    print(
        f"G_go {g_go.number_of_nodes()}, after IGI {g_igi.number_of_nodes()}, after "
        f"redundant {g_red.number_of_nodes()}, after containment "
        f"{g_kept.number_of_nodes()}",
        flush=True,
    )
    dag = GODag(
        osp.join(genome.go_root, "go.obo"), optional_attrs={"relationship"}, prt=None
    )
    # Propagate the IGI-filtered DIRECT annotations (every node of g_igi, kept or not)
    # up the OBO through is_a and part_of, the rule of go_term_hot.py.
    memo: dict[str, set[str]] = {}
    gene_terms: dict[str, set[str]] = {}
    for node, d in g_igi.nodes(data=True):
        if node == SUPER_ROOT:
            continue
        term_id = cast(GOTerm, dag[node]).item_id
        lit = {term_id} | GOTermEmbeddingDataset.ancestors(dag, term_id, memo)
        for gene in d["gene_set"]:
            gene_terms.setdefault(gene, set()).update(lit)
    excluded = sorted(
        n for n in g_kept.nodes() if n == SUPER_ROOT or cast(GOTerm, dag[n]).level == 0
    )
    columns = sorted(
        {cast(GOTerm, dag[n]).item_id for n in g_kept.nodes() if n not in excluded}
    )
    index = {t: i for i, t in enumerate(columns)}
    current, _ = torch.load(osp.join(CURRENT_DIR, "go.pt"), weights_only=False)
    gene_ids: list[str] = list(current.id)
    assert set(gene_terms) <= set(gene_ids), "an annotated gene is not in the store"
    hot = np.zeros((len(gene_ids), len(columns)), dtype=np.float32)
    for i, gene in enumerate(gene_ids):
        for t in gene_terms.get(gene, ()):
            if t in index:
                hot[i, index[t]] = 1.0
    counts = hot.sum(axis=0).astype(int)
    # G_go.pkl was pickled from another GO release than the go.obo on disk: some of its
    # edges are not an ancestry relation there, so a term 006 kept on the strength of a
    # graph descendant can hold fewer than four genes under the OBO propagation. Both
    # are counted, and the OBO is the one ontology used (the two 028 models read it).
    n_foreign_edges = sum(
        1
        for c, p in g_go.edges()
        if SUPER_ROOT not in (c, p)
        and cast(GOTerm, dag[p]).item_id
        not in GOTermEmbeddingDataset.ancestors(dag, cast(GOTerm, dag[c]).item_id, memo)
    )
    data_list = []
    for i, gene in enumerate(gene_ids):
        data = Data(id=gene)
        data.embeddings = {"go": torch.from_numpy(hot[i]).unsqueeze(0)}
        data_list.append(data)
    with open(osp.join(CURRENT_DIR, "go.terms.json")) as fh:
        current_terms = {t["term_id"] for t in json.load(fh)["terms"]}
    n_before = n_annotations(g_go)
    n_after = n_annotations(g_igi)
    sidecar = {
        "model_name": "go",
        "propagate": True,
        "namespaces": sorted({cast(GOTerm, dag[t]).namespace for t in columns}),
        "min_genes": MIN_GENES,
        "n_genes": len(gene_ids),
        "n_genes_annotated": int((hot.sum(axis=1) > 0).sum()),
        "n_columns": len(columns),
        "annotation_source": "SCerevisiaeGraph.G_go (SGD go_details, direct terms)",
        "filters": [
            "filter_go_IGI",
            "filter_redundant_terms",
            f"filter_by_contained_genes(n={MIN_GENES}, gene_set=genome.gene_set)",
        ],
        "n_nodes_G_go": g_go.number_of_nodes(),
        "n_nodes_after_igi": g_igi.number_of_nodes(),
        "n_nodes_after_redundant": g_red.number_of_nodes(),
        "n_nodes_after_containment": g_kept.number_of_nodes(),
        "n_direct_annotations_before_igi": n_before,
        "n_direct_annotations_after_igi": n_after,
        "n_igi_annotations_removed": n_before - n_after,
        "non_term_nodes_excluded": excluded,
        "obo_version": dag.version,
        "n_edges_G_go": g_go.number_of_edges(),
        "n_edges_G_go_not_ancestry_in_obo": n_foreign_edges,
        "n_columns_under_min_genes": int((counts < MIN_GENES).sum()),
        "n_terms_shared_with_current": len(set(columns) & current_terms),
        "n_columns_current": len(current_terms),
        "terms": [
            {
                "term_id": t,
                "name": cast(GOTerm, dag[t]).name,
                "namespace": cast(GOTerm, dag[t]).namespace,
                "n_genes": int(counts[index[t]]),
            }
            for t in columns
        ],
    }
    os.makedirs(STORE_DIR, exist_ok=True)
    torch.save(InMemoryDataset.collate(data_list)[:2], osp.join(STORE_DIR, "go.pt"))
    with open(osp.join(STORE_DIR, "go.terms.json"), "w") as fh:
        json.dump(sidecar, fh, indent=1)
    return sidecar


def main() -> None:
    """Build the store when absent, then read both stores back and write the report."""
    paths = [osp.join(STORE_DIR, f) for f in ("go.pt", "go.terms.json")]
    if not all(osp.isfile(p) for p in paths):
        build()
    with open(paths[1]) as fh:
        meta = json.load(fh)
    with open(osp.join(CURRENT_DIR, "go.terms.json")) as fh:
        meta_current = json.load(fh)
    data, _ = torch.load(paths[0], weights_only=False)
    current, _ = torch.load(osp.join(CURRENT_DIR, "go.pt"), weights_only=False)
    assert list(data.id) == list(current.id)
    hot = data.embeddings["go"].numpy()
    hot_current = current.embeddings["go"].numpy()
    terms = [t["term_id"] for t in meta["terms"]]
    terms_current = {t["term_id"] for t in meta_current["terms"]}
    assert hot.shape == (len(data.id), len(terms))
    with gzip.open(RELEASED, "rt") as fh:
        labels = json.load(fh)["labels"]
    row = {g: i for i, g in enumerate(data.id)}
    labeled = [row[g] for g in labels if g in row]
    by_namespace: dict[str, int] = {}
    for t in meta["terms"]:
        by_namespace[t["namespace"]] = by_namespace.get(t["namespace"], 0) + 1
    report = StoreReport(
        store_dir=STORE_DIR,
        current_store_dir=CURRENT_DIR,
        annotation_source=meta["annotation_source"],
        filters=meta["filters"],
        n_nodes_G_go=meta["n_nodes_G_go"],
        n_nodes_after_igi=meta["n_nodes_after_igi"],
        n_nodes_after_redundant=meta["n_nodes_after_redundant"],
        n_nodes_after_containment=meta["n_nodes_after_containment"],
        n_direct_annotations_before_igi=meta["n_direct_annotations_before_igi"],
        n_direct_annotations_after_igi=meta["n_direct_annotations_after_igi"],
        n_igi_annotations_removed=meta["n_igi_annotations_removed"],
        n_non_term_nodes_excluded=len(meta["non_term_nodes_excluded"]),
        non_term_nodes_excluded=meta["non_term_nodes_excluded"],
        n_columns=len(terms),
        n_columns_current=len(terms_current),
        n_terms_shared_with_current=len(set(terms) & terms_current),
        n_terms_only_006=len(set(terms) - terms_current),
        n_terms_only_current=len(terms_current - set(terms)),
        n_columns_by_namespace=by_namespace,
        obo_version=meta["obo_version"],
        n_edges_G_go=meta["n_edges_G_go"],
        n_edges_G_go_not_ancestry_in_obo=meta["n_edges_G_go_not_ancestry_in_obo"],
        n_columns_under_min_genes=meta["n_columns_under_min_genes"],
        min_genes_per_column=int(hot.sum(axis=0).min()),
        n_genes=len(data.id),
        n_genes_annotated=int((hot.sum(axis=1) > 0).sum()),
        n_genes_annotated_current=int((hot_current.sum(axis=1) > 0).sum()),
        n_labeled_genes=len(labeled),
        mean_columns_per_labeled_gene_006=float(hot[labeled].sum(axis=1).mean()),
        mean_columns_per_labeled_gene_current=float(
            hot_current[labeled].sum(axis=1).mean()
        ),
        n_labeled_genes_without_a_column_006=int((hot[labeled].sum(axis=1) == 0).sum()),
        n_labeled_genes_without_a_column_current=int(
            (hot_current[labeled].sum(axis=1) == 0).sum()
        ),
    )
    out = osp.join(RESULTS, "go006_store.json")
    with open(out, "w") as fh:
        json.dump(report.model_dump(), fh, indent=2)
    print(report.model_dump_json(indent=2))
    print(out)


if __name__ == "__main__":
    main()
