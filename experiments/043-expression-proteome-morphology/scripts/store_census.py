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
and perturbation type counts. The raw source also writes a ``colocation`` key: distinct
genotypes per label and per dataset, genotypes carrying each label pair, the
genotype x environment grid of Nadal-Ribelles and Jackson, and the overlap of the three
isolate sets by genotype and by strain name. Writes ``results/store_census.json`` (raw) or
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


# The label pairs whose shared genotypes the co-location census counts.
LABEL_PAIRS: list[tuple[str, str]] = [
    ("expression_log2_ratio", "calmorph"),
    ("protein_abundance", "calmorph"),
    ("expression_log2_ratio", "protein_abundance"),
    ("expression_tpm", "protein_abundance"),
    ("umi_count", "calmorph"),
    ("expression_score", "calmorph"),
]
GRID_DATASETS = [
    "NadalRibellesPerturbSeq2025Dataset",
    "TfDeletionPseudobulkJackson2020Dataset",
]
ISOLATE_DATASETS = [
    "CaudalPanTranscriptome2024Dataset",
    "ProteomeMuenzner2024Dataset",
    "ProteinTurnoverMuenzner2024Dataset",
]
SKIP_COLOCATION = {"PooledPromoterExpressionVaishnav2022Dataset"}


class Colocation:
    """Which genotypes carry which labels, across datasets (raw source only).

    A genotype is keyed as in the per-dataset census (frozenset of gene names, or
    ``(cross, segregant_id)`` for a segregant); Vaishnav 2022 is skipped (every member
    carries the same reporter gene, so the gene key is meaningless there). Isolates are
    also keyed by the strain name the record carries (``natural_isolate.isolate``).
    """

    def __init__(self) -> None:
        """Empty maps."""
        self.labels_by_genotype: dict[Any, set[str]] = {}
        self.genotypes_by_label: dict[str, set[Any]] = {}
        self.genotypes_by_dataset: dict[str, set[Any]] = {}
        self.grid: dict[str, dict[str, set[Any]]] = {n: {} for n in GRID_DATASETS}
        self.media_by_environment: dict[str, str] = {}
        self.isolate_keys: dict[str, set[Any]] = {n: set() for n in ISOLATE_DATASETS}
        self.isolate_names: dict[str, set[str | None]] = {
            n: set() for n in ISOLATE_DATASETS
        }

    def add(self, record: dict[str, Any]) -> None:
        """Fold one record in."""
        experiment = record["experiment"]
        name = experiment["dataset_name"]
        if name in SKIP_COLOCATION:
            return
        genotype = experiment["genotype"]
        key = _genotype_key(genotype)
        label = experiment["phenotype"]["label_name"]
        self.labels_by_genotype.setdefault(key, set()).add(label)
        self.genotypes_by_label.setdefault(label, set()).add(key)
        self.genotypes_by_dataset.setdefault(name, set()).add(key)
        if name in self.grid:
            environment = _environment_key(experiment["environment"])
            self.grid[name].setdefault(environment, set()).add(key)
            self.media_by_environment[environment] = experiment["environment"]["media"][
                "name"
            ]
        if name in self.isolate_keys:
            self.isolate_keys[name].add(key)
            isolate = genotype.get("natural_isolate")
            self.isolate_names[name].add(isolate["isolate"] if isolate else None)

    @staticmethod
    def _overlap(sets: dict[str, set[Any]]) -> dict[str, int]:
        c, p, t = (sets[n] for n in ISOLATE_DATASETS)
        return {
            "caudal": len(c),
            "muenzner_proteome": len(p),
            "muenzner_turnover": len(t),
            "caudal_and_proteome": len(c & p),
            "caudal_and_turnover": len(c & t),
            "proteome_and_turnover": len(p & t),
            "all_three": len(c & p & t),
        }

    def summary(self) -> dict[str, Any]:
        """JSON-ready co-location tables."""
        pairs = {
            f"{a} x {b}": sum(
                1 for labels in self.labels_by_genotype.values() if {a, b} <= labels
            )
            for a, b in LABEL_PAIRS
        }
        grids = {}
        for name, by_env in self.grid.items():
            genotypes = set().union(*by_env.values())
            grids[name] = {
                "environments": len(by_env),
                "genotypes": len(genotypes),
                "grid_size": len(by_env) * len(genotypes),
                "observed_cells": sum(len(g) for g in by_env.values()),
                "genotypes_per_environment": sorted(
                    (
                        {
                            "environment": env[:12],
                            "media": self.media_by_environment[env],
                            "genotypes": len(g),
                        }
                        for env, g in by_env.items()
                    ),
                    key=lambda row: (-row["genotypes"], row["media"]),
                ),
            }
        return {
            "genotypes_by_label": {
                k: len(v) for k, v in sorted(self.genotypes_by_label.items())
            },
            "genotypes_by_dataset": {
                k: len(v) for k, v in sorted(self.genotypes_by_dataset.items())
            },
            "genotypes_total": len(self.labels_by_genotype),
            "label_pairs": pairs,
            "genotype_environment_grid": grids,
            "isolate_overlap_by_genotype": self._overlap(self.isolate_keys),
            "isolate_overlap_by_strain_name": self._overlap(
                {k: set(v) for k, v in self.isolate_names.items()}
            ),
        }


def colocation_markdown(c: dict[str, Any]) -> str:
    """The co-location tables as markdown."""
    out = ["| label | genotypes |", "|---|---|"]
    out += [f"| {k} | {v:,} |" for k, v in c["genotypes_by_label"].items()]
    out += ["", "| dataset | genotypes |", "|---|---|"]
    out += [f"| {k} | {v:,} |" for k, v in c["genotypes_by_dataset"].items()]
    out += [
        "",
        f"Distinct genotypes over all datasets but Vaishnav: {c['genotypes_total']:,}",
    ]
    out += ["", "| label pair | genotypes carrying both |", "|---|---|"]
    out += [f"| {k} | {v:,} |" for k, v in c["label_pairs"].items()]
    for name, g in c["genotype_environment_grid"].items():
        out += [
            "",
            f"{name}: {g['genotypes']:,} genotypes x {g['environments']} environments = "
            f"grid {g['grid_size']:,}, observed cells {g['observed_cells']:,}",
            "",
            "| environment | media | genotypes |",
            "|---|---|---|",
        ]
        out += [
            f"| {r['environment']} | {r['media']} | {r['genotypes']:,} |"
            for r in g["genotypes_per_environment"]
        ]
    for title in ("isolate_overlap_by_genotype", "isolate_overlap_by_strain_name"):
        out += ["", f"| {title} | count |", "|---|---|"]
        out += [f"| {k} | {v:,} |" for k, v in c[title].items()]
    return "\n".join(out)


def census(
    source: str, data_root: str
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    """Per-dataset summaries, and the co-location tables for the raw source."""
    tallies: dict[str, DatasetTally] = {}
    colocation: Colocation | None = None
    if source == "raw":
        path = osp.join(
            data_root,
            "data/torchcell/experiments",
            EXPERIMENT,
            "raw_043",
            "raw",
            "lmdb",
        )
        colocation = Colocation()
        for record in _iter_raw(path):
            name = record["experiment"]["dataset_name"]
            tallies.setdefault(name, DatasetTally()).add(record)
            colocation.add(record)
    else:
        for name, directory in DEV_DIRS.items():
            path = osp.join(data_root, "data/torchcell", directory, "processed", "lmdb")
            tally = tallies.setdefault(name, DatasetTally())
            for record in _iter_dev(path):
                tally.add(record)
    summaries = {name: tallies[name].summary() for name in sorted(tallies)}
    return summaries, colocation.summary() if colocation is not None else None


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
    summaries, colocation = census(source, data_root)
    name = "store_census.json" if source == "raw" else "store_census_dev.json"
    out = osp.join(experiment_results_dir(EXPERIMENT, __file__), name)
    with open(out, "w") as f:
        payload: dict[str, Any] = {"source": source, "datasets": summaries}
        if colocation is not None:
            payload["colocation"] = colocation
        json.dump(payload, f, indent=1)
    print(markdown(summaries))
    if colocation is not None:
        print("\n## Co-location\n")
        print(colocation_markdown(colocation))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
