# experiments/028-gene-essentiality/scripts/degree_affine_store.py
# [[experiments.028-gene-essentiality.scripts.degree_affine_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/degree_affine_store

r"""The solve store over every gene of the genome, with the physical graph's degree beside it.

Round 30 and its six-seed extension read the transformer with the layer-free solve at 0.570
on the 4,674 genome-wide held-out genes, where two layers of graph convolution read 0.755
and the raw degree in the physical interaction graph, nothing trained, 0.740. Two things
were confounded in that transformer read. The pFBA store covers the 1,121 labeled genes
only, so every held-out gene read an all-zero row in the affine's solve columns
(``n_genes_with_solve`` 1,121), a row no training gene has; and the transformer takes the
gene graphs through a regularizer, never as an input. Two stores in the solve store's shape
(``genes``, ``v_deletion``, ``v_wild_type``, ``rxn_ids``, ``provenance``), read through
``ESS_SOLVE_TABLE`` and encoded by the trainer as $\log(1 + |v|)$, separate them:

* ``ess_solve_encodings_YPD_pfbawt.npz``: every gene of the genome. A gene the pFBA store
  covers keeps its deletion vector; every other gene reads the wild-type vector, which is
  what the program returns for a deletion that closes no reaction. The control.
* ``ess_solve_encodings_YPD_pfbawtdegree.npz``: the same, with two columns appended, the
  gene's degree and its clustering coefficient in the physical graph, the two structural
  inputs of ``graph_baselines.structural`` that carried the convolution's 0.753.

Writes ``results/degree_affine_store.json`` with the counts.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
        python experiments/028-gene-essentiality/scripts/degree_affine_store.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import sys

import networkx as nx
import numpy as np
from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from graph_baselines import gene_graphs  # noqa: E402

TABLE_DIR = osp.join(DATA_ROOT, "data/torchcell/reference_flux")
PFBA_STORE = osp.join(TABLE_DIR, "ess_solve_encodings_YPD.npz")


def main() -> None:
    pfba = np.load(PFBA_STORE, allow_pickle=False)
    covered = [str(g) for g in pfba["genes"]]
    v = pfba["v_deletion"].astype(np.float32)
    v_wt = pfba["v_wild_type"].astype(np.float32)
    rxn_ids = [str(r) for r in pfba["rxn_ids"]]
    row_of = {g: i for i, g in enumerate(covered)}

    gene_ids, graphs = gene_graphs()
    physical = graphs["physical"]
    assert set(covered) <= set(gene_ids), sorted(set(covered) - set(gene_ids))[:5]
    full = np.stack([v[row_of[g]] if g in row_of else v_wt for g in gene_ids])
    degree = dict(physical.degree())
    clustering = nx.clustering(physical)
    structure = np.array(
        [[float(degree[g]), float(clustering[g])] for g in gene_ids], dtype=np.float32
    )

    report: dict[str, object] = {
        "source_store": PFBA_STORE,
        "n_genes": len(gene_ids),
        "n_genes_with_own_deletion_vector": len(covered),
        "n_genes_reading_the_wild_type_vector": len(gene_ids) - len(covered),
        "n_reaction_columns": len(rxn_ids),
        "physical_graph_edges": physical.number_of_edges(),
        "degree_median": float(np.median(structure[:, 0])),
        "degree_max": float(structure[:, 0].max()),
        "n_genes_degree_zero": int((structure[:, 0] == 0).sum()),
        "stores": {},
    }
    for suffix, block, names, wt in (
        ("pfbawt", full, rxn_ids, v_wt),
        (
            "pfbawtdegree",
            np.concatenate([full, structure], axis=1),
            rxn_ids + ["physical_degree", "physical_clustering"],
            np.concatenate([v_wt, np.zeros(2, dtype=np.float32)]),
        ),
    ):
        path = osp.join(TABLE_DIR, f"ess_solve_encodings_YPD_{suffix}.npz")
        provenance = {
            "method": (
                "the pFBA deletion vector of ess_solve_encodings_YPD.npz for the genes it "
                "covers and its wild-type vector for every other gene of the genome"
                + (
                    "; the gene's degree and clustering coefficient in the physical "
                    "interaction graph appended as two columns"
                    if suffix.endswith("degree")
                    else ""
                )
                + "; written by degree_affine_store.py"
            ),
            "source_store": PFBA_STORE,
            "source_store_provenance": json.loads(str(pfba["provenance"])),
        }
        np.savez_compressed(
            path,
            genes=np.array(gene_ids),
            v_deletion=block.astype(np.float32),
            v_wild_type=wt,
            rxn_ids=np.array(names),
            provenance=json.dumps(provenance),
        )
        report["stores"][suffix] = {  # type: ignore[index]
            "path": path,
            "shape": list(block.shape),
        }
        print(f"{path}: {block.shape}")
    out = osp.join(RESULTS_DIR, "degree_affine_store.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(out)


if __name__ == "__main__":
    main()
