# experiments/019-simb-multimodal/scripts/make_split_indices.py
# [[experiments.019-simb-multimodal.scripts.make_split_indices]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/make_split_indices
"""Materialize the CellDataModule partition for a list of split seeds and record it.

Every training run reads its partition from
``<dataset_root>/data_module_cache/index_seed_<k>.json`` (record indices per split) and
``index_details_seed_<k>.json`` (the same, broken out per phenotype label). The datamodule
writes both on first use, so a run on a fresh seed would build them on the compute node.
This script builds them on the CPU ahead of time so (1) the GPU round and the linear
baselines read one file per split rather than two independent derivations, and (2) the
partition has a provenance record: per seed, the expression record counts per split and the
sha256 of the index file, written to ``results/split_indices_manifest.json``.

The construction is the training script's exactly (same dataset root, same query, same
deduplicator and aggregator, same ``split_indices`` keys), so the cached file is the one
``train_cgt_multitask.py`` would have produced. Node embeddings do not enter the split, so
only ``calm`` is loaded, which keeps this a few minutes on CPU.

Run from the repo root:
    python experiments/019-simb-multimodal/scripts/make_split_indices.py --seeds 0 1 2 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp

from dotenv import load_dotenv

load_dotenv()

from torchcell.data import (  # noqa: E402
    GenotypeAggregator,
    MeanExperimentDeduplicator,
    Neo4jCellDataset,
)
from torchcell.data.graph_processor import Perturbation  # noqa: E402
from torchcell.datamodules import CellDataModule  # noqa: E402
from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder  # noqa: E402
from torchcell.graph import SCerevisiaeGraph  # noqa: E402
from torchcell.graph.graph import build_gene_multigraph  # noqa: E402
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402
from torchcell.utils.paths import experiment_results_dir  # noqa: E402

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
EXPRESSION_LABEL = "expression_log2_ratio"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--dataset-tag", default="fig3_core")
    p.add_argument(
        "--label",
        default=EXPRESSION_LABEL,
        help="phenotype label whose per-split record counts the manifest records "
        "(expression_log2_ratio for fig3_core, protein_abundance for fig3_proteome)",
    )
    return p.parse_args()


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    data_root = os.environ["DATA_ROOT"]
    experiment_root = os.environ["EXPERIMENT_ROOT"]
    dataset_root = osp.join(
        data_root, "data/torchcell/experiments/019-simb-multimodal", args.dataset_tag
    )
    with open(
        osp.join(
            experiment_root, "019-simb-multimodal/queries", f"{args.dataset_tag}.cql"
        )
    ) as handle:
        query = handle.read()
    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(data_root, "data/sgd/genome"),
        string_root=osp.join(data_root, "data/string"),
        tflink_root=osp.join(data_root, "data/tflink"),
        genome=genome,
    )
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=build_gene_multigraph(graph=graph, graph_names=GRAPHS),
        incidence_graphs=None,
        node_embeddings=NodeEmbeddingBuilder.build(
            embedding_names=["calm"], data_root=data_root, genome=genome, graph=graph
        ),
        converter=None,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=Perturbation(),
        transform=None,
    )
    cache_dir = osp.join(dataset_root, "data_module_cache")
    # One manifest per dataset tag; fig3_core keeps its original file name.
    manifest_name = (
        "split_indices_manifest.json"
        if args.dataset_tag == "fig3_core"
        else f"split_indices_manifest_{args.dataset_tag}.json"
    )
    manifest_path = osp.join(
        experiment_results_dir("019-simb-multimodal", __file__), manifest_name
    )
    manifest: dict[str, object] = {}
    if osp.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    manifest["generated_by"] = (
        "experiments/019-simb-multimodal/scripts/make_split_indices.py"
    )
    manifest.setdefault("dataset_tag", args.dataset_tag)
    manifest.setdefault("label", args.label)
    per_seed = manifest.setdefault("seeds", {})
    assert isinstance(per_seed, dict)
    for seed in args.seeds:
        index_path = osp.join(cache_dir, f"index_seed_{seed}.json")
        details_path = osp.join(cache_dir, f"index_details_seed_{seed}.json")
        existed = osp.exists(index_path)
        dm = CellDataModule(
            dataset=dataset,
            cache_dir=cache_dir,
            split_indices=["phenotype_label_index", "perturbation_count_index"],
            batch_size=32,
            random_seed=seed,
            num_workers=0,
            pin_memory=False,
            prefetch=False,
            follow_batch=["perturbation_indices", "phenotype_values"],
        )
        dm.setup()
        with open(details_path) as f:
            details = json.load(f)
        counts = {
            split: int(details[split]["phenotype_label_index"][args.label]["count"])
            for split in ("train", "val", "test")
        }
        record = {
            "index_file": index_path,
            "index_sha256": _sha256(index_path),
            "details_sha256": _sha256(details_path),
            "existed_before_this_run": existed,
            "n_records": {
                "train": len(dm.index.train),
                "val": len(dm.index.val),
                "test": len(dm.index.test),
            },
            "n_label_records": counts,
            "label": args.label,
        }
        per_seed[str(seed)] = record
        print(
            f"seed {seed}: records train/val/test "
            f"{record['n_records']['train']}/{record['n_records']['val']}/"
            f"{record['n_records']['test']}; {args.label} "
            f"{counts['train']}/{counts['val']}/{counts['test']}; "
            f"{'existing' if existed else 'NEW'} index sha256 "
            f"{record['index_sha256'][:12]}"
        )
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
