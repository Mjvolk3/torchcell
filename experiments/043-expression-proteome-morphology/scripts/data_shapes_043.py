# experiments/043-expression-proteome-morphology/scripts/data_shapes_043.py
# [[experiments.043-expression-proteome-morphology.scripts.data_shapes_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/data_shapes_043
"""What every dataset of the 043 raw store is, as tensors.

Reads the raw LMDB of the 043 store (``$DATA_ROOT/data/torchcell/experiments/
043-expression-proteome-morphology/raw_043/raw/lmdb``; keys ``data_<i>``, one JSON record
per value) and writes ``results/data_shapes_043.json``: per dataset the genotype class, the
perturbation types with their per-record counts (min / median / max), the label key and its
dimensionality (keys per record, min / median / max), the gene index space (how many label
keys and perturbed gene names fall in the 6,607-gene ``gene_set.json`` of the store, by raw
name and after the two released-id maps: Messner 2023 UniProt accession -> ORF, Nadal-Ribelles
2025 standard name -> ORF), the value scale (``measurement_type`` plus the finite value
range), the environments, the record count; for the two segregant panels the block counts
and the posterior distribution (blocks and markers per posterior level, fraction of blocks at
posterior >= 0.9); for Vaishnav 2022 the promoter length, the scaffold and the variable
region span. Every quantity carries its ``n``.

Two passes. The first is a head-only pass over every key (``buffers=True``, the first 300
bytes of each value hold ``dataset_name``) that writes the raw dataset index
``raw_043/raw/raw_dataset_name_index.json`` (dataset -> sorted raw indices; the processed
store's ``dataset_name_index.json`` is in processed order, which is NOT raw order). The
second decodes records: ``--mode full`` every record (slurm:
``gh_data_shapes_043.slurm``, one read of the 163 GB store), ``--mode sample`` an evenly
spaced ``--per-dataset`` records of each dataset (login node, bounded).

Run from the repository root:
    python experiments/043-expression-proteome-morphology/scripts/data_shapes_043.py --mode sample --per-dataset 300
    sbatch experiments/043-expression-proteome-morphology/scripts/gh_data_shapes_043.slurm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import time
from collections import Counter
from typing import Any

import lmdb
import numpy as np
from dotenv import load_dotenv

from torchcell.datasets.scerevisiae.messner2023 import (
    build_orf_to_gene_name_map,
    build_uniprot_to_orf_map,
)
from torchcell.utils.paths import experiment_results_dir

EXPERIMENT = "043-expression-proteome-morphology"
STORE_TAG = "raw_043"
HEAD_BYTES = 300
RAW_INDEX_NAME = "raw_dataset_name_index.json"

# The dataset whose label keys are UniProt accessions, and the one whose keys are SGD
# standard names; every other gene-level dataset keys its label by systematic ORF.
UNIPROT_KEYED = {"ProteomeMessner2023Dataset"}
STANDARD_NAME_KEYED = {"NadalRibellesPerturbSeq2025Dataset"}
SEGREGANT_DATASETS = {"Boocock2025Dataset", "Nguessan2025Dataset"}
PROMOTER_DATASETS = {"PooledPromoterExpressionVaishnav2022Dataset"}

# experiment_type (the record field) -> the schema phenotype class it serializes.
PHENOTYPE_CLASS: dict[str, str] = {
    "microarray_expression": "MicroarrayExpressionPhenotype",
    "microarray_expression_score": "MicroarrayExpressionScorePhenotype",
    "pseudobulk_umi_count": "SingleCellExpressionPhenotype",
    "segregant_expression": "SingleCellExpressionPhenotype",
    "pseudobulk_expression": "PseudobulkExpressionPhenotype",
    "rnaseq_expression": "RNASeqExpressionPhenotype",
    "protein_abundance": "ProteinAbundancePhenotype",
    "protein_turnover": "ProteinTurnoverPhenotype",
    "pooled_sort_seq_expression": "PooledSortSeqExpressionPhenotype",
    "calmorph": "CalMorphPhenotype",
}


def store_root(data_root: str) -> str:
    """``$DATA_ROOT/data/torchcell/experiments/<EXPERIMENT>/raw_043``."""
    return osp.join(data_root, "data/torchcell/experiments", EXPERIMENT, STORE_TAG)


def dataset_name_from_head(head: bytes) -> str:
    """``dataset_name`` from the first bytes of a raw record (``"dataset_name": "<x>"``)."""
    marker = b'"dataset_name": "'
    start = head.index(marker) + len(marker)
    return head[start : head.index(b'"', start)].decode()


def raw_dataset_index(root: str) -> dict[str, list[int]]:
    """Dataset -> sorted raw indices, from a head-only pass; cached beside the LMDB.

    The cache is a derived view of an immutable store; it is reused only when its
    recorded entry count equals the LMDB's.
    """
    lmdb_path = osp.join(root, "raw", "lmdb")
    cache = osp.join(root, "raw", RAW_INDEX_NAME)
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
    entries = int(env.stat()["entries"])
    if osp.exists(cache):
        with open(cache) as f:
            cached = json.load(f)
        if cached["entries"] != entries:
            raise RuntimeError(
                f"{cache} records {cached['entries']} entries, the LMDB has {entries}"
            )
        env.close()
        return {k: v for k, v in cached["index"].items()}
    t0 = time.time()
    index: dict[str, list[int]] = {}
    with env.begin(buffers=True) as txn:
        for i in range(entries):
            value = txn.get(f"data_{i}".encode())
            if value is None:
                raise KeyError(f"data_{i} missing from {lmdb_path}")
            index.setdefault(
                dataset_name_from_head(bytes(value[:HEAD_BYTES])), []
            ).append(i)
    env.close()
    with open(cache, "w") as f:
        json.dump(
            {
                "entries": entries,
                "seconds": time.time() - t0,
                "counts": {k: len(v) for k, v in sorted(index.items())},
                "index": dict(sorted(index.items())),
            },
            f,
        )
    return dict(sorted(index.items()))


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def _stats(values: list[float] | list[int]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
    }


class Shapes:
    """Running shape tallies of one dataset."""

    def __init__(
        self,
        name: str,
        gene_set: set[str],
        up2orf: dict[str, str],
        name2orf: dict[str, str],
    ) -> None:
        """Empty tallies for ``name``."""
        self.name = name
        self.gene_set = gene_set
        self.up2orf = up2orf
        self.name2orf = name2orf
        self.n = 0
        self.experiment_types: Counter[str] = Counter()
        self.genotype_classes: Counter[str] = Counter()
        self.label_names: Counter[str] = Counter()
        self.graph_levels: Counter[str] = Counter()
        self.measurement_types: Counter[str] = Counter()
        self.n_perts: list[int] = []
        self.perts_by_type: dict[str, list[int]] = {}
        self.pert_type_total: Counter[str] = Counter()
        self.pert_genes_in_set_by_type: dict[str, set[str]] = {}
        self.pert_genes_out_of_set_by_type: dict[str, set[str]] = {}
        self.label_len: list[int] = []
        self.label_keys_union: set[str] = set()
        self.label_keys_in_set_per_record: list[int] = []
        self.label_keys_mapped_in_set_per_record: list[int] = []
        self.label_none_per_record: list[int] = []
        self.value_min = float("inf")
        self.value_max = float("-inf")
        self.value_sample: list[float] = []
        self.ref_label_len: list[int] = []
        self.environments: dict[str, dict[str, Any]] = {}
        self.media: Counter[str] = Counter()
        self.n_cells: list[int] = []
        self.total_umis: list[int] = []
        # segregants
        self.n_blocks: list[int] = []
        self.n_markers: list[int] = []
        self.posterior_blocks: Counter[str] = Counter()
        self.posterior_markers: Counter[str] = Counter()
        self.crosses: Counter[str] = Counter()
        self.parents: Counter[str] = Counter()
        self.chromosomes: set[str] = set()
        # promoters
        self.promoter_len: list[int] = []
        self.variable_len: list[int] = []
        self.scaffolds: Counter[str] = Counter()
        self.reporters: Counter[str] = Counter()
        self.promoter_variant_ids: set[str] = set()
        self.isolates: set[str] = set()

    def add(self, record: dict[str, Any]) -> None:
        """Fold one raw record in."""
        ex = record["experiment"]
        ref = record["experiment_reference"]
        genotype = ex["genotype"]
        phenotype = ex["phenotype"]
        environment = ex["environment"]
        self.n += 1
        self.experiment_types[ex["experiment_type"]] += 1
        self.graph_levels[phenotype["graph_level"]] += 1
        label = phenotype["label_name"]
        self.label_names[label] += 1
        mt = phenotype.get("measurement_type")
        if mt is not None:
            self.measurement_types[mt] += 1

        if "perturbations" in genotype:
            if genotype.get("natural_isolate") is not None:
                self.genotype_classes["Genotype+natural_isolate"] += 1
                self.isolates.add(genotype["natural_isolate"]["isolate"])
            else:
                self.genotype_classes["Genotype"] += 1
            perts = genotype["perturbations"]
            self.n_perts.append(len(perts))
            per_type: Counter[str] = Counter()
            for p in perts:
                t = p["perturbation_type"]
                per_type[t] += 1
                g = p["systematic_gene_name"]
                target = (
                    self.pert_genes_in_set_by_type
                    if g in self.gene_set
                    else self.pert_genes_out_of_set_by_type
                )
                target.setdefault(t, set()).add(g)
                if t == "variable_promoter_reporter":
                    seq = p["promoter_sequence"]
                    self.promoter_len.append(len(seq))
                    self.scaffolds[p["scaffold_name"]] += 1
                    self.reporters[p["perturbed_gene_name"]] += 1
                    self.promoter_variant_ids.add(p["promoter_variant_id"])
                    s, e = p["variable_region_start"], p["variable_region_end"]
                    self.variable_len.append(e - s + 1)
            self.pert_type_total.update(per_type)
            for t, c in per_type.items():
                self.perts_by_type.setdefault(t, []).append(c)
        else:
            self.genotype_classes["SegregantGenotype"] += 1
            self.n_perts.append(0)
            blocks = genotype["blocks"]
            self.n_blocks.append(len(blocks))
            self.n_markers.append(sum(b["n_markers"] for b in blocks))
            for b in blocks:
                key = f"{b['posterior']:.2f}"
                self.posterior_blocks[key] += 1
                self.posterior_markers[key] += b["n_markers"]
                self.chromosomes.add(b["chromosome"])
            self.crosses[genotype["cross"]] += 1
            self.parents[
                f"{genotype['parent_1']['name']} x {genotype['parent_2']['name']}"
            ] += 1

        value = phenotype[label]
        if isinstance(value, dict):
            self.label_len.append(len(value))
            keys = list(value)
            self.label_keys_union.update(keys)
            self.label_keys_in_set_per_record.append(
                sum(1 for k in keys if k in self.gene_set)
            )
            if self.name in UNIPROT_KEYED:
                mapped = [self.up2orf.get(k) for k in keys]
            elif self.name in STANDARD_NAME_KEYED:
                mapped = [self.name2orf.get(k.upper()) for k in keys]
            else:
                mapped = keys
            self.label_keys_mapped_in_set_per_record.append(
                sum(1 for k in mapped if k in self.gene_set)
            )
            vals = [v for v in value.values() if v is not None]
            self.label_none_per_record.append(len(value) - len(vals))
            if vals:
                arr = np.asarray(vals, dtype=float)
                finite = arr[np.isfinite(arr)]
                if finite.size:
                    self.value_min = min(self.value_min, float(finite.min()))
                    self.value_max = max(self.value_max, float(finite.max()))
                    self.value_sample.extend(
                        finite[:: max(1, finite.size // 20)].tolist()
                    )
        else:
            self.label_len.append(1)
            self.label_none_per_record.append(1 if value is None else 0)
            if value is not None:
                self.value_min = min(self.value_min, float(value))
                self.value_max = max(self.value_max, float(value))
                self.value_sample.append(float(value))
        ref_value = ref["phenotype_reference"][label]
        self.ref_label_len.append(
            len(ref_value)
            if isinstance(ref_value, dict)
            else (0 if ref_value is None else 1)
        )
        if "n_cells" in phenotype and phenotype["n_cells"] is not None:
            self.n_cells.append(int(phenotype["n_cells"]))
        if "total_umis" in phenotype and phenotype["total_umis"] is not None:
            self.total_umis.append(int(phenotype["total_umis"]))

        ekey = _sha(environment)
        if ekey not in self.environments:
            temp = environment["temperature"]
            self.environments[ekey] = {
                "media": environment["media"]["name"],
                "temperature": None if temp is None else temp["value"],
                "n_environment_perturbations": len(environment["perturbations"]),
            }
        self.media[environment["media"]["name"]] += 1

    def summary(self) -> dict[str, Any]:
        """JSON-ready summary with ``n`` on every quantity."""
        n = self.n
        out: dict[str, Any] = {
            "records_in_pass": n,
            "experiment_types": dict(self.experiment_types),
            "phenotype_class": sorted(
                {PHENOTYPE_CLASS[t] for t in self.experiment_types}
            ),
            "genotype_class": dict(self.genotype_classes),
            "label_name": dict(self.label_names),
            "graph_level": dict(self.graph_levels),
            "measurement_type": dict(self.measurement_types),
            "perturbations_per_record": _stats(self.n_perts),
            "perturbations_per_record_by_type": {
                t: {
                    **_stats(c),
                    "records_with_type": len(c),
                    "total": self.pert_type_total[t],
                }
                for t, c in sorted(self.perts_by_type.items())
            },
            "perturbed_gene_names_by_type": {
                t: {
                    "distinct_in_gene_set": len(
                        self.pert_genes_in_set_by_type.get(t, ())
                    ),
                    "distinct_outside_gene_set": len(
                        self.pert_genes_out_of_set_by_type.get(t, ())
                    ),
                    "examples_outside": sorted(
                        self.pert_genes_out_of_set_by_type.get(t, ())
                    )[:5],
                }
                for t in sorted(self.pert_type_total)
            },
            "label_keys_per_record": _stats(self.label_len),
            "label_keys_distinct": len(self.label_keys_union),
            "label_keys_distinct_in_gene_set_raw_name": sum(
                1 for k in self.label_keys_union if k in self.gene_set
            ),
            "label_keys_in_gene_set_per_record_raw_name": _stats(
                self.label_keys_in_set_per_record
            )
            if self.label_keys_in_set_per_record
            else None,
            "label_keys_in_gene_set_per_record_after_map": _stats(
                self.label_keys_mapped_in_set_per_record
            )
            if self.label_keys_mapped_in_set_per_record
            else None,
            "label_none_values_per_record": _stats(self.label_none_per_record),
            "value_range_finite": {
                "n_values_sampled": len(self.value_sample),
                "min": self.value_min,
                "median_of_sample": float(np.median(self.value_sample))
                if self.value_sample
                else None,
                "max": self.value_max,
            },
            "reference_label_keys_per_record": _stats(self.ref_label_len),
            "environments": len(self.environments),
            "environment_table": self.environments,
            "media_names": dict(self.media),
        }
        if self.name in UNIPROT_KEYED or self.name in STANDARD_NAME_KEYED:
            mapper = self.up2orf if self.name in UNIPROT_KEYED else None
            mapped_union = {
                (mapper.get(k) if mapper is not None else self.name2orf.get(k.upper()))
                for k in self.label_keys_union
            }
            out["label_keys_distinct_in_gene_set_after_map"] = sum(
                1 for k in mapped_union if k in self.gene_set
            )
        if self.n_cells:
            out["n_cells_per_record"] = _stats(self.n_cells)
        if self.total_umis:
            out["total_umis_per_record"] = _stats(self.total_umis)
        if self.isolates:
            out["distinct_isolates"] = len(self.isolates)
        if self.n_blocks:
            blocks_total = sum(self.posterior_blocks.values())
            markers_total = sum(self.posterior_markers.values())
            out["segregant"] = {
                "blocks_per_record": _stats(self.n_blocks),
                "markers_per_record": _stats(self.n_markers),
                "chromosomes": sorted(self.chromosomes),
                "crosses": dict(self.crosses),
                "parent_pairs": dict(self.parents),
                "posterior_levels_blocks": dict(sorted(self.posterior_blocks.items())),
                "posterior_levels_markers": dict(
                    sorted(self.posterior_markers.items())
                ),
                "fraction_blocks_posterior_ge_0.9": sum(
                    c for p, c in self.posterior_blocks.items() if float(p) >= 0.9
                )
                / blocks_total,
                "fraction_markers_posterior_ge_0.9": sum(
                    c for p, c in self.posterior_markers.items() if float(p) >= 0.9
                )
                / markers_total,
                "n_blocks_total": blocks_total,
                "n_markers_total": markers_total,
            }
        if self.promoter_len:
            out["promoter"] = {
                "promoter_length": _stats(self.promoter_len),
                "variable_region_length": _stats(self.variable_len),
                "scaffolds": dict(self.scaffolds),
                "reporter_genes": dict(self.reporters),
                "distinct_promoter_variant_ids": len(self.promoter_variant_ids),
            }
        return out


def markdown(summaries: dict[str, dict[str, Any]]) -> str:
    """The shape table as markdown (metric in every header)."""
    lines = [
        "| dataset | records in pass (n) | genotype class | phenotype class | label key | "
        "label keys per record min / median / max | label keys in 6,607 set per record "
        "median (after map) | perturbations per record min / median / max | perturbation "
        "types (records with type) | measurement_type | value min / max | environments |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in summaries.items():
        pt = ", ".join(
            f"{t} ({v['records_with_type']})"
            for t, v in s["perturbations_per_record_by_type"].items()
        )
        lk = s["label_keys_per_record"]
        pp = s["perturbations_per_record"]
        after = s["label_keys_in_gene_set_per_record_after_map"]
        in_set = f"{after['median']:g}" if after else "scalar"
        vr = s["value_range_finite"]
        lines.append(
            f"| {name} | {s['records_in_pass']:,} | {', '.join(s['genotype_class'])} | "
            f"{', '.join(s['phenotype_class'])} | {', '.join(s['label_name'])} | "
            f"{lk['min']:g} / {lk['median']:g} / {lk['max']:g} | {in_set} | "
            f"{pp['min']:g} / {pp['median']:g} / {pp['max']:g} | {pt or 'none'} | "
            f"{', '.join(s['measurement_type']) or 'n/a'} | {vr['min']:.4g} / {vr['max']:.4g} | "
            f"{s['environments']} |"
        )
    return "\n".join(lines)


def main() -> None:
    """Index the raw store, tally the shapes, write the JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", choices=["sample", "full"], default="sample")
    parser.add_argument("--per-dataset", type=int, default=300)
    args = parser.parse_args()
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    root = store_root(data_root)
    t0 = time.time()
    index = raw_dataset_index(root)
    print(
        f"raw index: {sum(len(v) for v in index.values()):,} records, {time.time() - t0:.0f} s"
    )
    gene_set_path = osp.join(root, "processed", "gene_set.json")
    with open(gene_set_path) as f:
        gene_list = json.load(f)
    gene_set = set(gene_list)
    with open(gene_set_path, "rb") as f:
        gene_set_sha = hashlib.sha256(f.read()).hexdigest()
    up2orf = build_uniprot_to_orf_map(data_root)
    name2orf: dict[str, str] = {}
    for orf, name in build_orf_to_gene_name_map(data_root).items():
        name2orf.setdefault(name.upper(), orf)
        name2orf.setdefault(orf.upper(), orf)

    env = lmdb.open(
        osp.join(root, "raw", "lmdb"), readonly=True, lock=False, readahead=False
    )
    summaries: dict[str, dict[str, Any]] = {}
    with env.begin() as txn:
        for name, ids in index.items():
            if args.mode == "sample" and len(ids) > args.per_dataset:
                chosen = [
                    ids[i]
                    for i in np.linspace(0, len(ids) - 1, args.per_dataset).astype(int)
                ]
            else:
                chosen = ids
            tally = Shapes(name, gene_set, up2orf, name2orf)
            t1 = time.time()
            for i in chosen:
                tally.add(json.loads(txn.get(f"data_{i}".encode())))
            s = tally.summary()
            s["records_in_store"] = len(ids)
            s["raw_index_first_last"] = [ids[0], ids[-1]]
            summaries[name] = s
            print(
                f"{name}: {len(chosen):,} of {len(ids):,} records, {time.time() - t1:.0f} s",
                flush=True,
            )
    env.close()

    payload = {
        "generated_by": "experiments/043-expression-proteome-morphology/scripts/data_shapes_043.py",
        "mode": args.mode,
        "per_dataset_sample": args.per_dataset if args.mode == "sample" else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "seconds": time.time() - t0,
        "store_entries": sum(len(v) for v in index.values()),
        "gene_set": {
            "path": gene_set_path,
            "n": len(gene_list),
            "sha256": gene_set_sha,
        },
        "datasets": summaries,
    }
    out = osp.join(experiment_results_dir(EXPERIMENT, __file__), "data_shapes_043.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=1)
    print(markdown(summaries))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
