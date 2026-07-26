# experiments/019-simb-multimodal/scripts/query_fig6_pigment_transfer.py
# [[experiments.019-simb-multimodal.scripts.query_fig6_pigment_transfer]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/query_fig6_pigment_transfer
"""Build the pigment metabolome-transfer dataset + its co-location census (Track A).

Builds a training-ready ``Neo4jCellDataset`` over EXACTLY THREE datasets --
``BetaxanthinCachera2023Dataset``, ``CarotenoidOzaydin2013Dataset`` and
``AminoAcidMulleder2016Dataset`` -- aggregated with **``DeletionKeyedGenotypeAggregator``**
and tensorized with the ``Perturbation`` graph processor (NOT ``SubgraphRepresentation``:
the transformer consumes per-genotype ``perturbation_indices`` batches).

WHY THE DELETION KEY IS THE WHOLE POINT. ``GenotypeAggregator`` keys a genotype on its
FULL perturbation set. Every pigment strain carries its biosynthesis cassette as
``gene_addition`` perturbations, so a cassette-bearing production genotype can never equal
a single-KO metabolome genotype and the measured co-location is exactly ZERO -- there is
no data at all for "does the metabolome help production prediction". Keying on the
DELETION set alone treats the (constant) cassette as reference-strain background and
recovers thousands of co-located genotypes. This script reports the number it actually
got, per pair, from the built aggregation LMDB -- not from a model of it.

Outputs ``results/fig6_pigment_transfer_census.json``:

* ``per_dataset``      -- rows per dataset in the built LMDB, and the modality it maps to.
* ``modality_census``  -- aggregated genotype groups, groups carrying >=2 modalities, and
  the explicit betaxanthin/metabolome and beta_carotene/metabolome pair counts.
* ``phenotype_shapes`` -- the decoded COO layout of a co-located sample: which label names
  appear and how many values each experiment contributes. This is what the training
  harness's head alignment has to match (betaxanthin 1 value and Mulleder 19 values BOTH
  under label ``metabolite_level``; beta-carotene 1 value under ``visual_score``).

Env:
* ``PIGMENT_CENSUS_ONLY=1`` -- re-run the census against an already-built LMDB, skipping
  the build.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from collections import Counter, defaultdict
from typing import Any

import lmdb
from dotenv import load_dotenv

NEO4J_URI = "neo4j+s://torchcell-database.ncsa.illinois.edu:7687"
NEO4J_AUTH = ("readonly", "ReadOnly")

DATASET_MODALITY: dict[str, str] = {
    "BetaxanthinCachera2023Dataset": "betaxanthin",
    "CarotenoidOzaydin2013Dataset": "beta_carotene",
    "AminoAcidMulleder2016Dataset": "metabolome",
}
QUERY_FILE = "fig6_pigment_transfer.cql"
DATASET_TAG = "fig6_pigment_transfer"


def build_dataset(query_path: str) -> tuple[Any, str]:
    """Build the pigment-transfer ``Neo4jCellDataset`` (deletion-keyed aggregation)."""
    from torchcell.data import (
        DeletionKeyedGenotypeAggregator,
        MeanExperimentDeduplicator,
    )
    from torchcell.data.graph_processor import Perturbation
    from torchcell.data.neo4j_cell import Neo4jCellDataset
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.graph.graph import build_gene_multigraph
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    data_root = os.environ["DATA_ROOT"]
    with open(query_path) as f:
        query = f.read()

    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
    )
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(data_root, "data/sgd/genome"),
        string_root=osp.join(data_root, "data/string"),
        tflink_root=osp.join(data_root, "data/tflink"),
        genome=genome,
    )
    gene_multigraph = build_gene_multigraph(
        graph=graph, graph_names=["physical", "regulatory"]
    )
    dataset_root = osp.join(
        data_root, "data/torchcell/experiments/019-simb-multimodal", DATASET_TAG
    )
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        uri=NEO4J_URI,
        username=NEO4J_AUTH[0],
        password=NEO4J_AUTH[1],
        graphs=gene_multigraph,
        # No metabolism incidence: Track A's heads read the pooled cell representation,
        # not the Yeast9 metabolite node set, so the bipartite graph is dead weight here.
        incidence_graphs=None,
        node_embeddings={},
        converter=None,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=DeletionKeyedGenotypeAggregator,
        graph_processor=Perturbation(),
    )
    return dataset, dataset_root


def aggregation_census(dataset_root: str) -> dict[str, Any]:
    """Census modalities + per-record value counts from the aggregation LMDB.

    Reads the AGGREGATION stage (post-deduplication, post-deletion-keying), which is the
    exact grouping the training harness will see.
    """
    lmdb_dir = osp.join(dataset_root, "aggregation", "lmdb")
    env = lmdb.open(lmdb_dir, readonly=True, lock=False, readahead=False)
    per_group_mods: list[frozenset[str]] = []
    per_dataset_rows: Counter[str] = Counter()
    value_counts: dict[str, Counter[int]] = defaultdict(Counter)
    merged_names: Counter[str] = Counter()
    with env.begin() as txn:
        for _, value in txn.cursor():
            pairs = json.loads(value.decode())
            mods: set[str] = set()
            for p in pairs:
                name = p["experiment"].get("dataset_name", "")
                parts = name.split("+")
                if len(set(parts)) != len(parts) or len(parts) > 1:
                    merged_names[name] += 1
                for part in parts:
                    if part in DATASET_MODALITY:
                        mods.add(DATASET_MODALITY[part])
                        per_dataset_rows[part] += 1
                pheno = p["experiment"]["phenotype"]
                for label in ("metabolite_level", "visual_score"):
                    val = pheno.get(label)
                    if isinstance(val, dict):
                        value_counts[label][len(val)] += 1
                    elif val is not None:
                        value_counts[label][1] += 1
            per_group_mods.append(frozenset(mods))
    env.close()

    ge2 = [m for m in per_group_mods if len(m) >= 2]

    def _pair(a: str, b: str) -> int:
        return sum(1 for m in per_group_mods if a in m and b in m)

    return {
        "aggregated_genotype_groups": len(per_group_mods),
        "groups_ge2_modalities": len(ge2),
        "betaxanthin_and_metabolome": _pair("betaxanthin", "metabolome"),
        "beta_carotene_and_metabolome": _pair("beta_carotene", "metabolome"),
        "betaxanthin_and_beta_carotene": _pair("betaxanthin", "beta_carotene"),
        "all_three_modalities": sum(1 for m in per_group_mods if len(m) == 3),
        "combos": {
            "+".join(sorted(c)): n for c, n in Counter(per_group_mods).most_common()
        },
        "per_dataset_rows": dict(per_dataset_rows.most_common()),
        # Value counts per COO label: how many dict entries each record contributes.
        # betaxanthin -> 1, Mulleder -> 19, both under `metabolite_level`.
        "phenotype_value_counts": {
            label: dict(sorted(c.items())) for label, c in value_counts.items()
        },
        # A dataset_name containing '+' means the deduplicator MERGED two records that
        # shared (experiment_type, full gene-name set). Surfaced because the betaxanthin
        # cassette carries ARO4/ARO7 as alleles, so the dARO4 and dARO7 strains have
        # identical full gene-name sets and get averaged together.
        "deduplicator_merged_dataset_names": dict(merged_names.most_common()),
    }


def sample_phenotype_shapes(dataset: Any, max_scan: int = 4000) -> dict[str, Any]:
    """Describe the first co-located (>=2 modality) built sample's COO layout."""
    from collections import defaultdict as _dd

    out: dict[str, Any] = {"colocated_index": None}
    for idx in range(min(len(dataset), max_scan)):
        sample = dataset[idx]
        gene = sample["gene"]
        types = list(gene["phenotype_types"])
        tidx = gene["phenotype_type_indices"].tolist()
        sidx = gene["phenotype_sample_indices"].tolist()
        groups: dict[tuple[str, int], int] = _dd(int)
        for t, s in zip(tidx, sidx):
            groups[(types[t], s)] += 1
        if len(groups) < 2:
            continue
        out["colocated_index"] = idx
        out["phenotype_types"] = types
        out["value_groups"] = [
            {"label": lab, "sample_index": s, "n_values": n}
            for (lab, s), n in sorted(groups.items())
        ]
        out["perturbed_genes"] = list(gene["perturbed_genes"])
        out["n_perturbation_indices"] = int(gene["perturbation_indices"].numel())
        break
    return out


def main() -> None:
    """Build (unless census-only) and write the census JSON."""
    load_dotenv()
    here = osp.dirname(osp.abspath(__file__))
    results_dir = osp.abspath(osp.join(here, "..", "results"))
    os.makedirs(results_dir, exist_ok=True)
    query_path = osp.abspath(osp.join(here, "..", "queries", QUERY_FILE))

    dataset, dataset_root = build_dataset(query_path)
    print(f"len(dataset) = {len(dataset)}")

    census = aggregation_census(dataset_root)
    report: dict[str, Any] = {
        "query_file": QUERY_FILE,
        "dataset_tag": DATASET_TAG,
        "dataset_root": dataset_root,
        "aggregator": "DeletionKeyedGenotypeAggregator",
        "graph_processor": "Perturbation",
        "datasets": sorted(DATASET_MODALITY),
        "dataset_len": len(dataset),
        "modality_census": census,
        "phenotype_shapes": sample_phenotype_shapes(dataset),
    }
    print(json.dumps(report, indent=2)[:4000])
    out = osp.join(results_dir, "fig6_pigment_transfer_census.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out}")
    dataset.close_lmdb()


if __name__ == "__main__":
    main()
