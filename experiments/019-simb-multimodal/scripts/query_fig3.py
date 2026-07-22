# experiments/019-simb-multimodal/scripts/query_fig3.py
# [[experiments.019-simb-multimodal.scripts.query_fig3]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/query_fig3
"""Fig-3 "one embedding, many phenotypes" build + multimodal-overlap census (WS2).

Builds a training-ready ``Neo4jCellDataset`` that UNIONs expression (Kemmeren +
Sameith Sm/Dm microarray) + morphology (Ohya CalMorph) + fitness (Smf/Dmf), then
aggregates PER GENOTYPE with ``MeanExperimentDeduplicator`` + ``GenotypeAggregator``
so a single strain carries every modality measured on it.

It also reports THE gating numbers for the Fig-3 thesis ("does adding fitness, a
cheap scalar, improve expression, an expensive vector"):

* Q1 -- Sameith Dm double-KO expression pairs that also have a double-mutant
  fitness record (exact gene-pair match) in the DB.
* Q2 -- single-KO genotypes carrying BOTH expression and single-mutant fitness.
* Q3 -- after ``GenotypeAggregator``, how many aggregated instances carry >=2
  modalities (this gates WS11a). Computed from the real aggregation LMDB.

Env / args:
* ``FIG3_QUERY=core`` (default) uses ``queries/fig3_core.cql`` -- the tractable
  multimodal substrate (~29k records). ``FIG3_QUERY=full`` uses
  ``queries/fig3_build.cql`` (adds ~1.04M DmfKuzmin records, 0 extra multimodal
  genotypes -- see generate_fig3_cql.py).
* ``FIG3_CENSUS_ONLY=1`` runs only the live-DB overlap census (Q1/Q2 + Q3
  prediction) and skips the LMDB build -- fast, no embeddings needed.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from collections import Counter, defaultdict
from typing import Any

import lmdb
from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase

# experiment_type (schema) -> coarse modality bucket for the multimodal census.
_MODALITY_OF = {
    "microarray_expression": "expression",
    "rnaseq_expression": "expression",
    "calmorph": "morphology",
    "fitness": "fitness",
    "gene interaction": "gene_interaction",
}

NEO4J_URI = "neo4j+s://torchcell-database.ncsa.illinois.edu:7687"
NEO4J_AUTH = ("readonly", "ReadOnly")
NEO4J_DB = "torchcell"

EXPRESSION_DS = [
    "MicroarrayKemmeren2014Dataset",
    "SmMicroarraySameith2015Dataset",
    "DmMicroarraySameith2015Dataset",
]
SAMEITH_DM = "DmMicroarraySameith2015Dataset"
MORPHOLOGY_DS = ["ScmdOhya2005Dataset"]
SMF_DS = ["SmfCostanzo2016Dataset", "SmfKuzmin2018Dataset", "SmfKuzmin2020Dataset"]
DMF_DS = ["DmfCostanzo2016Dataset", "DmfKuzmin2018Dataset", "DmfKuzmin2020Dataset"]


def _gene_sets(
    driver: Driver, dataset_id: str, deletion_only: bool
) -> set[frozenset[str]]:
    """Unique frozensets of perturbed systematic gene names for one dataset.

    When ``deletion_only`` is set, only genotypes whose perturbations are ALL
    deletions are returned -- the strains a fitness record may legitimately merge
    onto (mean-deduplication forces every deletion to ``mean_deletion``, so a
    ts-allele / DAmP of the same gene is a DIFFERENT strain and is excluded).
    """
    where = (
        "WHERE ALL(p IN perts WHERE p.perturbation_type ENDS WITH 'deletion')"
        if deletion_only
        else ""
    )
    q = f"""
    MATCH (d:Dataset {{id:$id}})<-[:ExperimentMemberOf]-(e:Experiment)
    MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
    WITH e, g, [(g)<-[:PerturbationMemberOf]-(p) | p] AS perts
    {where}
    RETURN [p IN perts | p.systematic_gene_name] AS genes
    """
    out: set[frozenset[str]] = set()
    with driver.session(database=NEO4J_DB) as s:
        for r in s.run(q, id=dataset_id).data():
            out.add(frozenset(r["genes"]))
    return out


def db_overlap_census() -> dict[str, Any]:
    """Compute Q1/Q2 and the predicted Q3 modality census from the live DB."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    kem = _gene_sets(driver, "MicroarrayKemmeren2014Dataset", deletion_only=False)
    sm_sa = _gene_sets(driver, "SmMicroarraySameith2015Dataset", deletion_only=False)
    dm_sa = _gene_sets(driver, SAMEITH_DM, deletion_only=False)
    ohya = _gene_sets(driver, "ScmdOhya2005Dataset", deletion_only=False)

    smf: set[frozenset[str]] = set()
    for ds in SMF_DS:
        smf |= _gene_sets(driver, ds, deletion_only=True)
    dmf: set[frozenset[str]] = set()
    for ds in DMF_DS:
        dmf |= _gene_sets(driver, ds, deletion_only=True)
    driver.close()

    expr_single = kem | sm_sa

    # Q3 predicted: tag every unique genotype with the modalities that measure it.
    mod: dict[frozenset[str], set[str]] = defaultdict(set)
    for gs in kem | sm_sa | dm_sa:
        mod[gs].add("expression")
    for gs in ohya:
        mod[gs].add("morphology")
    for gs in smf | dmf:
        mod[gs].add("fitness")
    multi = {gs: ms for gs, ms in mod.items() if len(ms) >= 2}
    combos = {
        "+".join(sorted(c)): n
        for c, n in Counter(frozenset(ms) for ms in multi.values()).most_common()
    }

    census = {
        "expression_single_unique": len(expr_single),
        "kemmeren_unique": len(kem),
        "sameith_sm_unique": len(sm_sa),
        "sameith_dm_unique": len(dm_sa),
        "ohya_unique": len(ohya),
        "smf_deletion_unique": len(smf),
        "dmf_deletion_unique": len(dmf),
        "Q1_sameith_dm_with_dmf_fitness": len(dm_sa & dmf),
        "Q2_single_expr_with_smf_fitness": len(expr_single & smf),
        "Q2_kemmeren_with_smf": len(kem & smf),
        "Q2_sameith_sm_with_smf": len(sm_sa & smf),
        "ohya_with_expression": len(ohya & expr_single),
        "ohya_with_smf_fitness": len(ohya & smf),
        "Q3_total_unique_genotypes": len(mod),
        "Q3_ge2_modalities": len(multi),
        "Q3_expression_and_fitness": sum(
            1 for ms in mod.values() if "expression" in ms and "fitness" in ms
        ),
        "Q3_expression_and_morphology": sum(
            1 for ms in mod.values() if "expression" in ms and "morphology" in ms
        ),
        "Q3_combos": combos,
    }
    return census


def aggregation_modality_census(dataset_root: str) -> dict[str, Any]:
    """Census modalities per aggregated genotype from the built aggregation LMDB."""
    lmdb_dir = osp.join(dataset_root, "aggregation", "lmdb")
    env = lmdb.open(lmdb_dir, readonly=True, lock=False, readahead=False)
    per_group_mods: list[frozenset[str]] = []
    with env.begin() as txn:
        for _, value in txn.cursor():
            pairs = json.loads(value.decode())
            mods = {
                _MODALITY_OF.get(
                    p["experiment"]["experiment_type"],
                    p["experiment"]["experiment_type"],
                )
                for p in pairs
            }
            per_group_mods.append(frozenset(mods))
    env.close()

    ge2 = [m for m in per_group_mods if len(m) >= 2]
    combos = {"+".join(sorted(c)): n for c, n in Counter(ge2).most_common()}
    return {
        "aggregated_groups": len(per_group_mods),
        "groups_ge2_modalities": len(ge2),
        "groups_expression_and_fitness": sum(
            1 for m in per_group_mods if "expression" in m and "fitness" in m
        ),
        "groups_expression_and_morphology": sum(
            1 for m in per_group_mods if "expression" in m and "morphology" in m
        ),
        "combos": combos,
    }


def build_dataset(query_path: str) -> tuple[Any, str]:
    """Build the Fig-3 Neo4jCellDataset mirroring the 010 build pattern."""
    from torchcell.data import GenotypeAggregator, MeanExperimentDeduplicator
    from torchcell.data.graph_processor import SubgraphRepresentation
    from torchcell.data.neo4j_cell import Neo4jCellDataset
    from torchcell.datasets import CodonFrequencyDataset
    from torchcell.datasets.fungal_up_down_transformer import (
        FungalUpDownTransformerDataset,
    )
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.graph.graph import build_gene_multigraph
    from torchcell.metabolism.yeast_GEM import YeastGEM
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
    codon_frequency = CodonFrequencyDataset(
        root=osp.join(data_root, "data/scerevisiae/codon_frequency_embedding"),
        genome=genome,
    )
    fudt_3prime = FungalUpDownTransformerDataset(
        root=osp.join(data_root, "data/scerevisiae/fudt_embedding"),
        genome=genome,
        model_name="species_downstream",
    )
    fudt_5prime = FungalUpDownTransformerDataset(
        root=osp.join(data_root, "data/scerevisiae/fudt_embedding"),
        genome=genome,
        model_name="species_downstream",
    )
    gene_multigraph = build_gene_multigraph(
        graph=graph, graph_names=["physical", "regulatory"]
    )

    tag = osp.splitext(osp.basename(query_path))[0]
    dataset_root = osp.join(
        data_root, "data/torchcell/experiments/019-simb-multimodal", tag
    )
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        uri=NEO4J_URI,
        username=NEO4J_AUTH[0],
        password=NEO4J_AUTH[1],
        graphs=gene_multigraph,
        incidence_graphs={"metabolism_bipartite": YeastGEM().bipartite_graph},
        node_embeddings={
            "codon_frequency": codon_frequency,
            "fudt_3prime": fudt_3prime,
            "fudt_5prime": fudt_5prime,
        },
        converter=None,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
    )
    return dataset, dataset_root


def describe_item(data: Any) -> dict[str, Any]:
    """Summarize a built HeteroData sample: node/edge stores and label shapes."""
    info: dict[str, Any] = {
        "type": type(data).__name__,
        "node_types": {},
        "labels": {},
    }
    for nt in data.node_types:
        store = data[nt]
        info["node_types"][nt] = {
            "num_nodes": int(getattr(store, "num_nodes", 0) or 0),
            "keys": sorted(k for k in store.keys()),
        }
    gene = data["gene"] if "gene" in data.node_types else None
    if gene is not None:
        for key in gene.keys():
            val = gene[key]
            if hasattr(val, "shape"):
                info["labels"][f"gene.{key}"] = list(val.shape)
    for key in data.keys():
        val = data[key]
        if hasattr(val, "shape"):
            info["labels"][key] = list(val.shape)
    return info


def main() -> None:
    load_dotenv()
    here = osp.dirname(osp.abspath(__file__))
    results_dir = osp.abspath(osp.join(here, "..", "results"))
    os.makedirs(results_dir, exist_ok=True)

    which = os.environ.get("FIG3_QUERY", "core").lower()
    query_file = "fig3_build.cql" if which == "full" else "fig3_core.cql"
    query_path = osp.abspath(osp.join(here, "..", "queries", query_file))

    print("=" * 72)
    print("Fig-3 multimodal overlap census (live DB, deletion-filtered fitness)")
    print("=" * 72)
    census = db_overlap_census()
    for k, v in census.items():
        if k != "Q3_combos":
            print(f"  {k}: {v}")
    print("  Q3_combos:")
    for c, n in census["Q3_combos"].items():
        print(f"    {c}: {n}")

    report: dict[str, Any] = {"query_file": query_file, "db_overlap_census": census}

    if os.environ.get("FIG3_CENSUS_ONLY") == "1":
        with open(osp.join(results_dir, "fig3_overlap_census.json"), "w") as f:
            json.dump(report, f, indent=2)
        print("\nCENSUS_ONLY set -- skipped build.")
        return

    print("\n" + "=" * 72)
    print(f"Building dataset from {query_file} ...")
    print("=" * 72)
    dataset, dataset_root = build_dataset(query_path)
    print(f"len(dataset) = {len(dataset)}")

    agg_census = aggregation_modality_census(dataset_root)
    report["aggregation_modality_census"] = agg_census
    report["dataset_len"] = len(dataset)
    report["dataset_root"] = dataset_root
    print("\nPost-aggregation modality census (from aggregation LMDB):")
    for k, v in agg_census.items():
        print(f"  {k}: {v}")

    # Confirm the pipeline yields a valid HeteroData sample. The stock
    # SubgraphRepresentation processor tensor-izes phenotype fields directly, so it
    # only supports SCALAR phenotypes (fitness/gene interaction); dict-valued VECTOR
    # phenotypes (expression_log2_ratio, calmorph) raise "must be real number, not
    # dict". We therefore scan for the first index that builds -- proving the graph
    # pipeline is valid for scalar groups -- and record how many early indices are
    # blocked by the vector-phenotype limitation (a WS2 substrate finding).
    sample_info: dict[str, Any] = {
        "valid_index": None,
        "vector_blocked_indices": 0,
        "vector_block_error": None,
    }
    scan_cap = min(len(dataset), 500)
    for idx in range(scan_cap):
        try:
            sample = dataset[idx]
        except TypeError as exc:
            sample_info["vector_blocked_indices"] += 1
            sample_info["vector_block_error"] = str(exc)
            continue
        sample_info["valid_index"] = idx
        sample_info["item"] = describe_item(sample)
        break
    report["dataset_sample"] = sample_info
    print("\ndataset sample summary:")
    print(json.dumps(sample_info, indent=2))

    with open(osp.join(results_dir, "fig3_overlap_census.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {osp.join(results_dir, 'fig3_overlap_census.json')}")
    dataset.close_lmdb()


if __name__ == "__main__":
    main()
