# experiments/019-simb-multimodal/scripts/verify_deletion_keyed_aggregation.py
# [[experiments.019-simb-multimodal.scripts.verify_deletion_keyed_aggregation]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/verify_deletion_keyed_aggregation
"""Verify that deletion-keyed aggregation co-locates pigment and metabolome genotypes.

This is the GATE for the SIMB pigment/metabolome-transfer experiment
([[plan.cgt-metabolism.2026.07.25]] Track A). ``GenotypeAggregator`` keys on the
FULL perturbation set, and the pigment strains carry their heterologous pathway as
``gene_addition`` / ``allele`` perturbations, so a pigment genotype can never share
a bucket with a single-KO metabolome genotype -- co-location is exactly ZERO and the
"does the metabolome help production prediction" contrast has no data at all.

``DeletionKeyedGenotypeAggregator`` keys on the deletion gene-set only (cassette =
reference-strain background, Design Decision 3). This script applies BOTH aggregators
to the real source LMDBs and reports the resulting co-location, so the fix is
verified against data rather than asserted.

Expected (from the live-DB census in ``results/fig6_overlap_census.json``):

===========================  =========  ==============
pair                         full-key   deletion-key
===========================  =========  ==============
betaxanthin / metabolome             0           ~4439
beta-carotene / metabolome           0           ~4226
===========================  =========  ==============

Reads the dev-tree LMDBs under ``$DATA_ROOT/data/torchcell/`` directly -- no Neo4j
build required. Writes ``results/deletion_keyed_aggregation_census.json``.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import pickle
from collections import defaultdict

import lmdb
from dotenv import load_dotenv
from tqdm import tqdm

from torchcell.data import DeletionKeyedGenotypeAggregator, GenotypeAggregator
from torchcell.datamodels import EXPERIMENT_TYPE_MAP

# dataset directory under $DATA_ROOT/data/torchcell -> short modality label
DATASETS = {
    "betaxanthin_cachera2023": "betaxanthin",
    "carotenoid_ozaydin2013": "beta_carotene",
    "amino_acid_mulleder2016": "metabolome",
}


def _iter_experiments(lmdb_dir: str):
    """Yield reconstructed ``{"experiment": Experiment}`` dicts from an LMDB.

    Source dataset LMDBs under ``$DATA_ROOT/data/torchcell`` store **pickled**
    records (the string-interning build), unlike the JSON-encoded intermediate
    stores written by ``torchcell.data.aggregate``.
    """
    env = lmdb.open(lmdb_dir, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        for _, value in txn.cursor():
            record = pickle.loads(value)
            entries = record if isinstance(record, list) else [record]
            for entry in entries:
                experiment_class = EXPERIMENT_TYPE_MAP[
                    entry["experiment"]["experiment_type"]
                ]
                yield {
                    "experiment": experiment_class.model_validate(entry["experiment"])
                }
    env.close()


def _keys_by_modality(
    data_root: str,
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, int]]:
    """Return full-key sets, deletion-key sets, and record counts per modality."""
    full = GenotypeAggregator(root="/tmp/_agg_full")
    deletion = DeletionKeyedGenotypeAggregator(root="/tmp/_agg_deletion")

    full_keys: dict[str, set[str]] = defaultdict(set)
    deletion_keys: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, int] = {}

    for dirname, modality in DATASETS.items():
        lmdb_dir = osp.join(data_root, "data/torchcell", dirname, "processed/lmdb")
        n = 0
        for data in tqdm(_iter_experiments(lmdb_dir), desc=modality):
            full_keys[modality].add(full.aggregate_check(data))
            deletion_keys[modality].add(deletion.aggregate_check(data))
            n += 1
        counts[modality] = n
    return dict(full_keys), dict(deletion_keys), counts


def main() -> None:
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    here = osp.dirname(osp.abspath(__file__))
    results_dir = osp.abspath(osp.join(here, "..", "results"))
    os.makedirs(results_dir, exist_ok=True)

    full_keys, deletion_keys, counts = _keys_by_modality(data_root)

    pairs = [("betaxanthin", "metabolome"), ("beta_carotene", "metabolome")]
    report: dict[str, object] = {
        "n_records": counts,
        "n_unique_keys": {
            "full_geneset": {m: len(k) for m, k in full_keys.items()},
            "deletion_only": {m: len(k) for m, k in deletion_keys.items()},
        },
        "colocation": {},
    }
    colocation: dict[str, dict[str, int]] = {}
    for a, b in pairs:
        colocation[f"{a}__{b}"] = {
            "full_geneset": len(full_keys[a] & full_keys[b]),
            "deletion_only": len(deletion_keys[a] & deletion_keys[b]),
        }
    report["colocation"] = colocation

    out = osp.join(results_dir, "deletion_keyed_aggregation_census.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nwrote {out}")

    # The gate: deletion-keying must recover thousands of shared genotypes.
    for pair, values in colocation.items():
        status = "PASS" if values["deletion_only"] > 1000 else "FAIL"
        print(
            f"{status} {pair}: full-key {values['full_geneset']} "
            f"-> deletion-key {values['deletion_only']}"
        )


if __name__ == "__main__":
    main()
