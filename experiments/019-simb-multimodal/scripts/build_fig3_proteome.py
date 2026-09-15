# experiments/019-simb-multimodal/scripts/build_fig3_proteome.py
# [[experiments.019-simb-multimodal.scripts.build_fig3_proteome]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/build_fig3_proteome
"""Build the ``fig3_proteome`` dataset: the three expression panels plus the Messner proteome.

The query (``queries/fig3_proteome.cql``, from ``generate_fig3_cql.py``) unions Kemmeren
2014, Sameith 2015 (single and double deletions) and Messner 2023. The proteome values are
rewritten at build time to ``log2(strain / HIS3 reference)`` per protein by
``ProteinAbundanceLog2RatioConverter``, so the stored label has the same form as the
expression ``log2(strain / wild type)`` and the same per-gene readout trains on either.
Everything else mirrors the ``fig3_core`` build the expression campaign trains on: the
mean deduplicator, the genotype aggregator (a deletion measured in both panels becomes one
record carrying both phenotypes), the ``Perturbation`` processor, and the nine-graph set.
Node embeddings are attached at read time and do not enter the LMDB, so only ``calm`` is
loaded here.

The store is the served graph on GilaHyper (``TC_NEO4J_URI``, default
``bolt://localhost:7687``); the served endpoint is recorded in the census. After the build
the script counts raw against converted records (the converter's base ``process`` logs and
skips a record it cannot convert, so an unequal count here is the failure signal), the
records per phenotype label, the genotypes carrying both, and the log2 range of the first
proteome record, and writes ``results/fig3_proteome_build_census.json``.

Run from the repo root:
    python experiments/019-simb-multimodal/scripts/build_fig3_proteome.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import socket
import time

import lmdb
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from torchcell.data import (  # noqa: E402
    GenotypeAggregator,
    MeanExperimentDeduplicator,
    Neo4jCellDataset,
)
from torchcell.data.graph_processor import Perturbation  # noqa: E402
from torchcell.datamodels.protein_abundance_log2_ratio_conversion import (  # noqa: E402
    ProteinAbundanceLog2RatioConverter,
)
from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder  # noqa: E402
from torchcell.graph import SCerevisiaeGraph  # noqa: E402
from torchcell.graph.graph import build_gene_multigraph  # noqa: E402
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402
from torchcell.utils.paths import experiment_results_dir  # noqa: E402

DATASET_TAG = "fig3_proteome"
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
PROTEOME_LABEL = "protein_abundance"
EXPRESSION_LABEL = "expression_log2_ratio"


def _lmdb_entries(path: str) -> int:
    env = lmdb.open(path, readonly=True, lock=False, subdir=True)
    with env.begin() as txn:
        n = int(txn.stat()["entries"])
    env.close()
    return n


def main() -> None:
    data_root = os.environ["DATA_ROOT"]
    uri = os.environ.get("TC_NEO4J_URI", "bolt://localhost:7687")
    dataset_root = osp.join(
        data_root, "data/torchcell/experiments/019-simb-multimodal", DATASET_TAG
    )
    # The query next to this script, not under EXPERIMENT_ROOT: the build runs from the
    # worktree before the query file has landed on main.
    here = osp.dirname(osp.abspath(__file__))
    query_path = osp.abspath(osp.join(here, "..", "queries", f"{DATASET_TAG}.cql"))
    with open(query_path) as f:
        query = f.read()

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
    t0 = time.time()
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        uri=uri,
        graphs=build_gene_multigraph(graph=graph, graph_names=GRAPHS),
        incidence_graphs=None,
        node_embeddings=NodeEmbeddingBuilder.build(
            embedding_names=["calm"], data_root=data_root, genome=genome, graph=graph
        ),
        converter=ProteinAbundanceLog2RatioConverter,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=Perturbation(),
        transform=None,
    )
    build_seconds = time.time() - t0
    print(f"len(dataset) = {len(dataset)}  ({build_seconds / 60:.1f} min)")

    raw_n = _lmdb_entries(osp.join(dataset_root, "raw", "lmdb"))
    conv_n = _lmdb_entries(osp.join(dataset_root, "conversion", "lmdb"))
    if raw_n != conv_n:
        raise RuntimeError(
            f"conversion dropped records: raw {raw_n} vs converted {conv_n}; the "
            "converter's base process() skipped a record it could not convert"
        )

    label_index = dataset.phenotype_label_index
    prot_idx = set(label_index.get(PROTEOME_LABEL, []))
    expr_idx = set(label_index.get(EXPRESSION_LABEL, []))
    both = prot_idx & expr_idx

    # First proteome record: what the LMDB now holds for the label.
    env = lmdb.open(
        osp.join(dataset_root, "processed", "lmdb"), readonly=True, lock=False
    )
    first = sorted(prot_idx)[0]
    with env.begin() as txn:
        recs = json.loads(txn.get(f"{first}".encode()).decode())
    env.close()
    if isinstance(recs, dict):
        recs = [recs]
    prot = [r for r in recs if r["experiment"]["phenotype"]["label_name"] == PROTEOME_LABEL]
    ph = prot[0]["experiment"]["phenotype"]
    vals = np.array(list(ph[PROTEOME_LABEL].values()), dtype=float)
    ref_vals = np.array(
        list(
            prot[0]["experiment_reference"]["phenotype_reference"][
                PROTEOME_LABEL
            ].values()
        ),
        dtype=float,
    )
    census = {
        "generated_by": "experiments/019-simb-multimodal/scripts/build_fig3_proteome.py",
        "dataset_tag": DATASET_TAG,
        "dataset_root": dataset_root,
        "query_file": query_path,
        "neo4j_uri": uri,
        "host": socket.gethostname(),
        "build_seconds": build_seconds,
        "n_records": len(dataset),
        "n_raw_records": raw_n,
        "n_converted_records": conv_n,
        "n_records_by_label": {k: len(v) for k, v in label_index.items()},
        "n_records_with_proteome_and_expression": len(both),
        "first_proteome_record": {
            "index": first,
            "n_experiments_in_record": len(recs),
            "measurement_type": ph["measurement_type"],
            "n_proteins": int(len(vals)),
            "log2_ratio_min": float(vals.min()),
            "log2_ratio_median": float(np.median(vals)),
            "log2_ratio_max": float(vals.max()),
            "reference_all_zero": bool(np.all(ref_vals == 0.0)),
        },
    }
    out = osp.join(
        experiment_results_dir("019-simb-multimodal", __file__),
        "fig3_proteome_build_census.json",
    )
    with open(out, "w") as f:
        json.dump(census, f, indent=1)
    print(json.dumps(census, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
