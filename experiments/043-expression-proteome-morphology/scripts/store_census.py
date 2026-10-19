# experiments/043-expression-proteome-morphology/scripts/store_census.py
# [[experiments.043-expression-proteome-morphology.scripts.store_census]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/store_census
"""Census of the experiment 043 store: per dataset, what the records hold.

Two sources, one table:

* ``--source raw`` (default): the built raw LMDB
  ``$DATA_ROOT/data/torchcell/experiments/043-expression-proteome-morphology/raw_043/raw/lmdb``
  written by ``build_043_store.py``; keys ``data_<i>``, one JSON record per value with the
  interned constants already spliced in by the query.
* ``--source dev``: the 15 dev-tree loader LMDBs
  ``$DATA_ROOT/data/torchcell/<dataset>/processed/lmdb`` (pickled record dicts whose
  interned sub-objects are ``{"$ref": ...}`` pointers, resolved here from the sibling
  ``processed/interned`` env with ``resolve_interned``). This is the census before the
  served store has been queried; it counts what the loaders built, not what KG 4.0 serves.

Per dataset it records: records; distinct genotypes, keyed by the frozenset of
``systematic_gene_name`` over the perturbations (a ``SegregantGenotype``, which has no
perturbation list, by ``(cross, segregant_id)``); distinct perturbation sets keyed by the
content of each perturbation (separates genotypes that share gene names, e.g. a promoter
library whose every member carries the same reporter); distinct environments keyed by
media name, temperature and the environmental perturbations, and distinct media names;
label names; label vector length min / median / max; perturbations per record min / max;
and perturbation type counts. Writes ``results/store_census.json`` (raw) or
``results/store_census_dev.json`` (dev) and prints a markdown table.

Run from the repo root:
    python experiments/043-expression-proteome-morphology/scripts/store_census.py --source raw
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import pickle
from collections import Counter
from collections.abc import Iterator
from typing import Any

import lmdb
import numpy as np
from dotenv import load_dotenv

from torchcell.data.experiment_dataset import resolve_interned
from torchcell.utils.paths import experiment_results_dir

EXPERIMENT = "043-expression-proteome-morphology"

# Dataset class -> dev-tree directory under $DATA_ROOT/data/torchcell.
DEV_DIRS: dict[str, str] = {
    "MicroarrayKemmeren2014Dataset": "microarray_kemmeren2014",
    "SmMicroarraySameith2015Dataset": "sm_microarray_sameith2015",
    "DmMicroarraySameith2015Dataset": "dm_microarray_sameith2015",
    "MicroarrayScoreHu2007Dataset": "microarray_score_hu2007",
    "TfDeletionPseudobulkJackson2020Dataset": "tf_deletion_pseudobulk_jackson2020",
    "NadalRibellesPerturbSeq2025Dataset": "nadal_ribelles_perturbseq2025",
    "ProteomeMessner2023Dataset": "proteome_messner2023",
    "ProteomeZelezniak2018Dataset": "proteome_zelezniak2018",
    "ScmdOhya2005Dataset": "scmd_ohya2005",
    "CaudalPanTranscriptome2024Dataset": "caudal_pantranscriptome2024",
    "ProteomeMuenzner2024Dataset": "proteome_muenzner2024",
    "ProteinTurnoverMuenzner2024Dataset": "protein_turnover_muenzner2024",
    "Boocock2025Dataset": "boocock2025",
    "Nguessan2025Dataset": "nguessan2025",
    "PooledPromoterExpressionVaishnav2022Dataset": "pooled_promoter_expression_vaishnav2022",
}


def _sha(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()


def _iter_raw(path: str) -> Iterator[dict[str, Any]]:
    env = lmdb.open(path, readonly=True, lock=False, readahead=True)
    with env.begin() as txn:
        for _, value in txn.cursor():
            yield json.loads(value)
    env.close()


def _iter_dev(path: str) -> Iterator[dict[str, Any]]:
    interned: dict[str, Any] = {}
    interned_dir = osp.join(osp.dirname(path), "interned")
    if osp.isdir(interned_dir):
        ienv = lmdb.open(interned_dir, readonly=True, lock=False)
        with ienv.begin() as txn:
            for key, value in txn.cursor():
                interned[key.decode()] = pickle.loads(value)
        ienv.close()
    env = lmdb.open(path, readonly=True, lock=False, readahead=True)
    with env.begin() as txn:
        for _, value in txn.cursor():
            yield resolve_interned(pickle.loads(value), interned)
    env.close()


def _genotype_key(genotype: dict[str, Any]) -> Any:
    if "perturbations" not in genotype:
        return ("segregant", genotype["cross"], genotype["segregant_id"])
    return frozenset(p["systematic_gene_name"] for p in genotype["perturbations"])


def _environment_key(environment: dict[str, Any]) -> str:
    return _sha(
        {
            "media": environment["media"]["name"],
            "temperature": environment["temperature"],
            "perturbations": environment["perturbations"],
        }
    )


class DatasetTally:
    """Running per-dataset counts."""

    def __init__(self) -> None:
        """Empty tallies."""
        self.records = 0
        self.genotypes: set[Any] = set()
        self.perturbation_sets: set[frozenset[str]] = set()
        self.environments: set[str] = set()
        self.media: set[str] = set()
        self.labels: Counter[str] = Counter()
        self.label_lengths: list[int] = []
        self.n_perturbations: list[int] = []
        self.perturbation_types: Counter[str] = Counter()

    def add(self, record: dict[str, Any]) -> None:
        """Fold one record in."""
        experiment = record["experiment"]
        genotype = experiment["genotype"]
        environment = experiment["environment"]
        phenotype = experiment["phenotype"]
        perts = genotype.get("perturbations", [])
        self.records += 1
        self.genotypes.add(_genotype_key(genotype))
        self.perturbation_sets.add(
            frozenset(_sha(p) for p in perts)
            if "perturbations" in genotype
            else frozenset({_sha(genotype)})
        )
        self.environments.add(_environment_key(environment))
        self.media.add(environment["media"]["name"])
        label = phenotype["label_name"]
        self.labels[label] += 1
        value = phenotype[label]
        self.label_lengths.append(len(value) if isinstance(value, (dict, list)) else 1)
        self.n_perturbations.append(len(perts))
        self.perturbation_types.update(p["perturbation_type"] for p in perts)

    def summary(self) -> dict[str, Any]:
        """JSON-ready summary."""
        lengths = np.array(self.label_lengths)
        npert = np.array(self.n_perturbations)
        return {
            "records": self.records,
            "genotypes": len(self.genotypes),
            "perturbation_sets": len(self.perturbation_sets),
            "environments": len(self.environments),
            "media_names": len(self.media),
            "labels": dict(self.labels),
            "label_length_min": int(lengths.min()),
            "label_length_median": float(np.median(lengths)),
            "label_length_max": int(lengths.max()),
            "perturbations_per_record_min": int(npert.min()),
            "perturbations_per_record_max": int(npert.max()),
            "perturbation_types": dict(self.perturbation_types.most_common()),
        }


def census(source: str, data_root: str) -> dict[str, dict[str, Any]]:
    """Per-dataset summaries from the chosen source."""
    tallies: dict[str, DatasetTally] = {}
    if source == "raw":
        path = osp.join(
            data_root,
            "data/torchcell/experiments",
            EXPERIMENT,
            "raw_043",
            "raw",
            "lmdb",
        )
        for record in _iter_raw(path):
            name = record["experiment"]["dataset_name"]
            tallies.setdefault(name, DatasetTally()).add(record)
    else:
        for name, directory in DEV_DIRS.items():
            path = osp.join(data_root, "data/torchcell", directory, "processed", "lmdb")
            tally = tallies.setdefault(name, DatasetTally())
            for record in _iter_dev(path):
                tally.add(record)
    return {name: tallies[name].summary() for name in sorted(tallies)}


def markdown(summaries: dict[str, dict[str, Any]]) -> str:
    """The census as a markdown table."""
    lines = [
        "| dataset | records | genotypes | perturbation sets | environments | media | "
        "label | label length min / median / max | perturbations per record | "
        "perturbation types |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in summaries.items():
        types = ", ".join(f"{k} {v}" for k, v in s["perturbation_types"].items())
        lines.append(
            f"| {name} | {s['records']:,} | {s['genotypes']:,} | "
            f"{s['perturbation_sets']:,} | {s['environments']} | {s['media_names']} | "
            f"{', '.join(s['labels'])} | {s['label_length_min']} / "
            f"{s['label_length_median']:g} / {s['label_length_max']} | "
            f"{s['perturbations_per_record_min']} to {s['perturbations_per_record_max']} | "
            f"{types} |"
        )
    total = sum(s["records"] for s in summaries.values())
    lines.append(f"\nTotal records: {total:,}")
    return "\n".join(lines)


def main() -> None:
    """Run the census and write the JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", choices=["raw", "dev"], default="raw")
    source = parser.parse_args().source
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    summaries = census(source, data_root)
    name = "store_census.json" if source == "raw" else "store_census_dev.json"
    out = osp.join(experiment_results_dir(EXPERIMENT, __file__), name)
    with open(out, "w") as f:
        json.dump({"source": source, "datasets": summaries}, f, indent=1)
    print(markdown(summaries))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
