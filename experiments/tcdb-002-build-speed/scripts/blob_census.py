# experiments/tcdb-002-build-speed/scripts/blob_census.py
# [[experiments.tcdb-002-build-speed.scripts.blob_census]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/tcdb-002-build-speed/scripts/blob_census
"""Census of the Experiment node's ``serialized_data`` blob across the 51 datasets.

For every dataset in the build's adapter map, sample records evenly from the dev LMDB
the KG build reads, and measure the JSON the adapter writes per Experiment node:
the whole ``experiment.model_dump()`` and its ``genotype``, ``environment`` and
``phenotype`` parts, plus zlib on the whole. Project the Experiment CSV bytes for the
full record count under three layouts: inline (the served build), the environment as
a pointer to a content-addressed node, and environment plus large genotypes as
pointers. ``interned_entries`` is the size of the dataset's own interned table, an
upper bound on its distinct environment plus reference plus publication objects.

Writes ``results/blob_census.csv`` next to this experiment's other results.

    ~/miniconda3/envs/torchcell/bin/python \
        experiments/tcdb-002-build-speed/scripts/blob_census.py [--samples 64]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import os.path as osp
import zlib
from typing import Any

import lmdb
from dotenv import load_dotenv

POINTER_BYTES = 160  # {"$ref": <64 hex>, "name": <hint>} as the adapter writes it
GENOTYPE_POINTER_MIN_BYTES = 8192

RESULTS_DIR = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "results")


def _interned_entries(processed_dir: str) -> int:
    path = osp.join(processed_dir, "interned")
    if not osp.isdir(path):
        return 0
    env = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        n = int(txn.stat()["entries"])
    env.close()
    return n


def _mean(values: list[int]) -> float:
    return sum(values) / len(values)


def census_dataset(dataset: Any, samples: int) -> dict[str, Any]:
    n = len(dataset)
    step = max(1, n // samples)
    rows: list[dict[str, int]] = []
    genotype_types: set[str] = set()
    env_hashes: set[str] = set()
    geno_hashes: set[str] = set()
    for i in range(0, n, step):
        experiment = dataset.transform_item(dataset[i])["experiment"]
        dump = experiment.model_dump()
        genotype = json.dumps(dump["genotype"])
        environment = json.dumps(dump["environment"])
        phenotype = json.dumps(dump["phenotype"])
        whole = json.dumps(dump)
        genotype_types.add(type(experiment.genotype).__name__)
        env_hashes.add(hashlib.sha256(environment.encode()).hexdigest())
        geno_hashes.add(hashlib.sha256(genotype.encode()).hexdigest())
        rows.append(
            {
                "exp": len(whole),
                "genotype": len(genotype),
                "environment": len(environment),
                "phenotype": len(phenotype),
                "zlib": len(zlib.compress(whole.encode(), 6)),
            }
        )
    dataset.close_lmdb()
    exp = _mean([r["exp"] for r in rows])
    geno = _mean([r["genotype"] for r in rows])
    env = _mean([r["environment"] for r in rows])
    phen = _mean([r["phenotype"] for r in rows])
    comp = _mean([r["zlib"] for r in rows])
    env_pointer = exp - env + POINTER_BYTES
    genotype_pointer = geno >= GENOTYPE_POINTER_MIN_BYTES
    both_pointer = env_pointer - (geno - POINTER_BYTES if genotype_pointer else 0)
    return {
        "n_records": n,
        "interned_entries": _interned_entries(dataset.processed_dir),
        "sample_n": len(rows),
        "genotype_types": "|".join(sorted(genotype_types)),
        "distinct_environments_in_sample": len(env_hashes),
        "distinct_genotypes_in_sample": len(geno_hashes),
        "exp_bytes": round(exp),
        "genotype_bytes": round(geno),
        "environment_bytes": round(env),
        "phenotype_bytes": round(phen),
        "exp_zlib_bytes": round(comp),
        "inline_gb": round(n * exp / 1e9, 3),
        "env_pointer_gb": round(n * env_pointer / 1e9, 3),
        "genotype_pointer": genotype_pointer,
        "env_geno_pointer_gb": round(n * both_pointer / 1e9, 3),
        "inline_zlib_gb": round(n * comp / 1e9, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--out", default=osp.join(RESULTS_DIR, "blob_census.csv"))
    args = parser.parse_args()

    os.environ["WANDB_MODE"] = "disabled"
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]

    from torchcell.graph import SCerevisiaeGraph
    from torchcell.knowledge_graphs.dataset_adapter_map import dataset_adapter_map
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(data_root, "data/sgd/genome"),
        string_root=osp.join(data_root, "data/string"),
        tflink_root=osp.join(data_root, "data/tflink"),
        genome=genome,
    )

    results: list[dict[str, Any]] = []
    for dataset_class in dataset_adapter_map:
        params = inspect.signature(dataset_class.__init__).parameters
        root = osp.join(data_root, params["root"].default)
        if not osp.isdir(osp.join(root, "processed", "lmdb")):
            print(f"SKIP {dataset_class.__name__}: no LMDB at {root}", flush=True)
            continue
        kwargs: dict[str, Any] = {"io_workers": 0}
        if "genome" in params:
            kwargs["genome"] = genome
        if "scerevisiae_graph" in params:
            kwargs["scerevisiae_graph"] = graph
        dataset = dataset_class(root=root, **kwargs)
        row = {"dataset": dataset_class.__name__, **census_dataset(dataset, args.samples)}
        results.append(row)
        print(
            f"{row['dataset']:<40} n={row['n_records']:>10,} exp={row['exp_bytes']:>8} "
            f"env={row['environment_bytes']:>6} geno={row['genotype_bytes']:>7} "
            f"inline={row['inline_gb']:>8.3f} GB env_ptr={row['env_pointer_gb']:>8.3f} GB",
            flush=True,
        )

    results.sort(key=lambda r: -r["inline_gb"])
    os.makedirs(osp.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    total = {
        k: round(sum(r[k] for r in results), 1)
        for k in ("inline_gb", "env_pointer_gb", "env_geno_pointer_gb", "inline_zlib_gb")
    }
    print("TOTAL", total)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
