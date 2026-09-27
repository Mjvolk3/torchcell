# experiments/025-solid-growth/scripts/graph_reg_flags_smoke.py
# [[experiments.025-solid-growth.scripts.graph_reg_flags_smoke]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/graph_reg_flags_smoke
"""Construct the real-size model under every new round-2 flag and take one step, on CPU.

The unit tests pin the flags on a four-gene graph; this builds the 6,607-gene cell graph
the trainer builds (genome, the nine gene-gene graphs, no LMDB) and, for each of the
round-2 mechanism variants, constructs the model, runs one forward pass on a two-genotype
batch, backpropagates the point loss plus the graph penalty, and reports the time, the
peak resident memory, and that every loss is finite. It is the plumbing check between
the unit tests and the Delta queue: a wrong key, an all-minus-infinity attention row, or
a target that fails to normalize shows up here in minutes rather than in a 24 h job.

    sbatch experiments/025-solid-growth/scripts/gh_graph_reg_flags_smoke.slurm
"""

from __future__ import annotations

import json
import os
import os.path as osp
import resource
import time
from typing import Any

import torch
from dotenv import load_dotenv
from sortedcontainers import SortedDict

from torchcell.data.cell_data import to_cell_data
from torchcell.data.neo4j_cell import create_graph_from_gene_set
from torchcell.graph import SCerevisiaeGraph, build_gene_multigraph
from torchcell.graph.graph import GeneMultiGraph
from torchcell.models.equivariant_cell_graph_transformer import CellGraphTransformer
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "025-solid-growth", "results")
GRAPHS = [
    "physical",
    "regulatory",
    "tflink",
    "string12_0_neighborhood",
    "string12_0_fusion",
    "string12_0_cooccurence",
    "string12_0_coexpression",
    "string12_0_experimental",
    "string12_0_database",
]
HEAD_GRAPHS = {
    0: "physical_interaction",
    1: "regulatory_interaction",
    2: "tflink",
    3: "string12_0_neighborhood",
    4: "string12_0_fusion",
    5: "string12_0_cooccurence",
    6: "string12_0_coexpression",
    7: "string12_0_experimental",
    8: "string12_0_database",
}


def _kl_config(hops: int = 1, symmetrize: bool = False) -> dict[str, Any]:
    heads = {
        "physical": 0,
        "regulatory": 1,
        "tflink": 2,
        "string12_0_neighborhood": 3,
        "string12_0_fusion": 4,
        "string12_0_cooccurence": 5,
        "string12_0_coexpression": 6,
        "string12_0_experimental": 7,
        "string12_0_database": 8,
    }
    return {
        "graph_reg_lambda": 1.0,
        "graph_reg_layer": 1,
        "row_sampling_rate": 1.0,
        "hops": hops,
        "symmetrize": symmetrize,
        "regularized_heads": {
            g: {"layer": 1, "head": h, "lambda": 1.0} for g, h in heads.items()
        },
    }


VARIANTS: dict[str, dict[str, Any]] = {
    "kl_1hop": {"graph_reg_lambda": 1.0, "graph_regularization_config": _kl_config()},
    "kl_2hop": {
        "graph_reg_lambda": 1.0,
        "graph_regularization_config": _kl_config(hops=2),
    },
    "kl_3hop": {
        "graph_reg_lambda": 1.0,
        "graph_regularization_config": _kl_config(hops=3),
    },
    "kl_symmetric": {
        "graph_reg_lambda": 1.0,
        "graph_regularization_config": _kl_config(symmetrize=True),
    },
    "kl_layers_1_4": {
        "graph_reg_lambda": 1.0,
        "graph_regularization_config": {
            **_kl_config(),
            "regularized_heads": {
                g: {"layer": [1, 2, 3, 4], "head": c["head"], "lambda": 1.0}
                for g, c in _kl_config()["regularized_heads"].items()
            },
        },
    },
    "mask_1hop": {
        "attention_mask_config": {
            "enabled": True,
            "layers": [1],
            "head_graphs": HEAD_GRAPHS,
        }
    },
    "mask_directed": {
        "attention_mask_config": {
            "enabled": True,
            "layers": [1],
            "head_graphs": HEAD_GRAPHS,
            "symmetric": False,
        }
    },
    "mask_2hop_directed": {
        "attention_mask_config": {
            "enabled": True,
            "layers": [1],
            "head_graphs": HEAD_GRAPHS,
            "hops": 2,
            "symmetric": False,
        }
    },
    "mask_3hop_directed": {
        "attention_mask_config": {
            "enabled": True,
            "layers": [1],
            "head_graphs": HEAD_GRAPHS,
            "hops": 3,
            "symmetric": False,
        }
    },
    "mask_2hop_symmetric": {
        "attention_mask_config": {
            "enabled": True,
            "layers": [1],
            "head_graphs": HEAD_GRAPHS,
            "hops": 2,
        }
    },
    "mask_layers_3_4": {
        "attention_mask_config": {
            "enabled": True,
            "layers": [3, 4],
            "head_graphs": HEAD_GRAPHS,
        }
    },
    "table_360": {
        "hidden_channels": 360,
        "graph_reg_lambda": 1.0,
        "graph_regularization_config": _kl_config(),
    },
}


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def main() -> None:
    """Build the cell graph once, then every variant."""
    t0 = time.time()
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )
    multigraph = build_gene_multigraph(graph=graph, graph_names=GRAPHS)
    # As Neo4jCellDataset does: the node set is the `base` graph over the genome's genes,
    # and to_cell_data refuses a multigraph without it.
    graphs = SortedDict(multigraph.graphs.copy())
    graphs["base"] = create_graph_from_gene_set(genome.gene_set)
    cell_graph = to_cell_data(
        GeneMultiGraph(graphs=graphs), None, add_remaining_gene_self_loops=True
    )
    n = int(cell_graph["gene"].num_nodes)
    print(
        f"cell graph: {n} genes, edge types {[e[1] for e in cell_graph.edge_types]}; {time.time() - t0:.0f} s, rss {_rss_gb():.1f} GB"
    )

    batch = type(cell_graph)()
    batch["gene"].perturbation_indices = torch.tensor(
        [5, 17, 300, 4000, 4001, 6000], dtype=torch.long
    )
    batch["gene"].perturbation_indices_batch = torch.tensor(
        [0, 0, 0, 1, 1, 1], dtype=torch.long
    )

    report: dict[str, Any] = {"n_genes": n, "variants": {}}
    for name, kwargs in VARIANTS.items():
        torch.manual_seed(0)
        t1 = time.time()
        model = CellGraphTransformer(
            gene_num=n,
            hidden_channels=int(kwargs.pop("hidden_channels", 180)),
            num_transformer_layers=8,
            num_attention_heads=9,
            cell_graph=cell_graph,
            perturbation_head_config={"num_heads": 9, "dropout": 0.1},
            dropout=0.1,
            learnable_embedding_config={
                "enabled": True,
                "size": 180 if "table_360" not in name else 360,
            },
            heads_config=None,
            **kwargs,
        )
        model.train()
        preds, reps = model(cell_graph, batch)
        loss = preds.pow(2).mean()
        graph_reg = reps.get("graph_reg_loss")
        if graph_reg is not None:
            loss = loss + graph_reg
        loss.backward()
        entry = {
            "params": sum(p.numel() for p in model.parameters()),
            "pred_shape": list(preds.shape),
            "loss": float(loss),
            "graph_reg_loss": None if graph_reg is None else float(graph_reg),
            "finite": bool(torch.isfinite(loss)),
            "seconds": round(time.time() - t1, 1),
            "peak_rss_gb": round(_rss_gb(), 1),
        }
        if model.attention_head_mask is not None:
            m = model.attention_head_mask
            entry["mask_allowed_fraction_per_head"] = [
                round(float(m[h, 1:, 1:].float().mean()), 4) for h in range(m.shape[0])
            ]
        if model.adjacency_matrices is not None:
            entry["target_rows_all_zero"] = {
                k: int((v.sum(1) == 0).sum())
                for k, v in model.adjacency_matrices.items()
                if not k.endswith("_interaction")
            }
        report["variants"][name] = entry
        print(
            f"{name:16s} params {entry['params']:>12,} loss {entry['loss']:.4f} graph_reg {entry['graph_reg_loss']} finite {entry['finite']} {entry['seconds']} s rss {entry['peak_rss_gb']} GB"
        )
        del model, preds, reps, loss
    assert all(v["finite"] for v in report["variants"].values()), (
        "a variant produced a non-finite loss"
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(osp.join(RESULTS_DIR, "graph_reg_flags_smoke.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
