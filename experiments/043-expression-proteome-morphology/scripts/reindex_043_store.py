# experiments/043-expression-proteome-morphology/scripts/reindex_043_store.py
# [[experiments.043-expression-proteome-morphology.scripts.reindex_043_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/reindex_043_store
"""Recompute the 043 store's index files from the existing processed LMDB, and check samples.

No query and no rebuild: ``Neo4jCellDataset`` skips ``process()`` when ``processed/lmdb``
exists, and each index property loads its json when present and recomputes it from the
processed store otherwise. Move a stale index json aside first, then run this. It
computes ``perturbation_count_index`` (in ``__init__``), ``is_any_perturbed_gene_index``
and ``is_any_deletion_gene_index``, reports their coverage, and reads samples through the
``Perturbation`` processor: the first record of every dataset, checking that the number of
phenotype values equals the label length (the shared-label fix).

Writes ``results/reindex_043_store.json``. Run under slurm (``gh_reindex_043_store.slurm``).
"""

from __future__ import annotations

import json
import os
import os.path as osp
import socket
import time

import lmdb
from dotenv import load_dotenv

load_dotenv()

from torchcell.data import Neo4jCellDataset  # noqa: E402
from torchcell.data.graph_processor import Perturbation  # noqa: E402
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402
from torchcell.utils.paths import experiment_results_dir  # noqa: E402

EXPERIMENT = "043-expression-proteome-morphology"


def _covered(index: dict[str, list[int]]) -> int:
    return len({i for indices in index.values() for i in indices})


def main() -> None:
    """Recompute the indices, check the samples, write the report."""
    data_root = os.environ["DATA_ROOT"]
    root = osp.join(data_root, "data/torchcell/experiments", EXPERIMENT, "raw_043")
    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()
    t0 = time.time()
    dataset = Neo4jCellDataset(
        root=root,
        query=None,
        gene_set=genome.gene_set,
        graphs=None,
        incidence_graphs=None,
        node_embeddings=None,
        converter=None,
        deduplicator=None,
        aggregator=None,
        graph_processor=Perturbation(),
        transform=None,
    )
    count_index = dataset.perturbation_count_index
    perturbed = dataset.is_any_perturbed_gene_index
    deleted = dataset.is_any_deletion_gene_index
    names = dataset.dataset_name_index
    labels = dataset.phenotype_label_index
    seconds = time.time() - t0

    env = lmdb.open(osp.join(root, "processed", "lmdb"), readonly=True, lock=False)
    samples = {}
    with env.begin() as txn:
        for name, indices in sorted(names.items()):
            index = min(indices)
            (record,) = json.loads(txn.get(f"{index}".encode()))
            phenotype = record["experiment"]["phenotype"]
            label = phenotype["label_name"]
            value = phenotype[label]
            gene = dataset[index]["gene"]
            samples[name] = {
                "index": index,
                "label_name": label,
                "label_length": len(value) if isinstance(value, (dict, list)) else 1,
                "n_phenotype_values": len(gene.phenotype_values),
                "phenotype_types": list(gene.phenotype_types),
                "perturbation_indices_length": len(gene.perturbation_indices),
            }
    env.close()
    report = {
        "generated_by": f"experiments/{EXPERIMENT}/scripts/reindex_043_store.py",
        "root": root,
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "seconds": seconds,
        "len_dataset": len(dataset),
        "perturbation_count_index": {
            "entries_covered": _covered({str(k): v for k, v in count_index.items()}),
            "count_0": len(count_index.get(0, [])),
            "by_count_head": {
                str(k): len(v) for k, v in sorted(count_index.items())[:5]
            },
        },
        "is_any_perturbed_gene_index": {
            "genes": len(perturbed),
            "entries_covered": _covered(perturbed),
        },
        "is_any_deletion_gene_index": {
            "genes": len(deleted),
            "entries_covered": _covered(deleted),
        },
        "dataset_name_index_entries_covered": _covered(names),
        "phenotype_label_index_entries_covered": _covered(labels),
        "samples": samples,
        "samples_value_count_equals_label_length": {
            name: s["n_phenotype_values"] == s["label_length"]
            for name, s in samples.items()
        },
    }
    out = osp.join(
        experiment_results_dir(EXPERIMENT, __file__), "reindex_043_store.json"
    )
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
