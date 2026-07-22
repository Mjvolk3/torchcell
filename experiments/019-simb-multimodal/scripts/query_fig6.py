# experiments/019-simb-multimodal/scripts/query_fig6.py
# [[experiments.019-simb-multimodal.scripts.query_fig6]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/query_fig6
"""Fig-6 production / metabolite build + WS11b co-location census (WS4).

Builds a training-ready ``Neo4jCellDataset`` that UNIONs the Fig-6 targets --
two pigment production screens (beta-carotene visual score Ozaydin2013,
betaxanthin Cachera2023), three metabolome screens (Mulleder AA, Zelezniak,
da Silveira lipids), and the isobutanol production screen + its validated subset
(Lopez2024) -- then aggregates PER GENOTYPE with ``MeanExperimentDeduplicator`` +
``GenotypeAggregator`` (gene-set-keyed, WS2b) so a strain carries every modality
measured on it. ``graph_processor=Perturbation`` (WS1) tensorizes dict-valued
metabolite vectors and scalar visual scores in one COO store.

THE census (the most valuable output -- it gates WS11b "does a metabolome label
improve production prediction"):

* Q1 -- gene-level deletion overlap: how many single genes are deleted in BOTH a
  metabolome screen and a production screen (isobutanol / betaxanthin / carotene).
* Q2 -- GENOTYPE (gene-set) co-location: after gene-set-keyed aggregation, how
  many genotypes carry >=2 of {metabolome, isobutanol, betaxanthin,
  beta_carotene_score}, by pair. This is the number WS11b can actually train on.
* Q3 -- per-dataset built counts + graph_level + media/temp (the sets span
  SC-URA / SC / SM / YPD @ 30 C; media differ -- no record is dropped on env).

THE finding (measured, not assumed): the two pigment strains carry heterologous
biosynthesis cassettes as ``gene_addition`` perturbations, so their FULL
perturbed gene set (which the aggregator keys on) never equals a single-KO
metabolome genotype -- betaxanthin / beta-carotene co-locate with metabolome on
ZERO genotypes. Only the cassette-free isobutanol screen (pure single-KO)
co-locates with metabolome (thousands of genotypes). So WS11b's "metabolome helps
production" test is well-powered for isobutanol but, at the genotype level, NOT
available for the pigment targets -- relating metabolome to betaxanthin /
beta-carotene must go through the shared DELETED gene (gene-level), not a
co-located label. Q1 vs Q2 makes exactly this gap legible.

Env / args:
* ``FIG6_CENSUS_ONLY=1`` runs only the live-DB census (Q1/Q2/Q3) and skips the
  LMDB build -- fast, no embeddings needed.
* ``FIG6_BUILD_CAP=<n>`` (build path only) caps ``dataset[idx]`` sample scanning;
  the LMDB itself always builds fully (~18.8k records, all tractable).
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

NEO4J_URI = "neo4j+s://torchcell-database.ncsa.illinois.edu:7687"
NEO4J_AUTH = ("readonly", "ReadOnly")
NEO4J_DB = "torchcell"

# dataset id -> semantic modality bucket for the WS11b co-location census.
# NB several distinct datasets share experiment_type "metabolite" in the schema,
# so the census MUST key on the dataset, not on experiment_type.
DATASET_MODALITY: dict[str, str] = {
    "CarotenoidOzaydin2013Dataset": "beta_carotene_score",
    "BetaxanthinCachera2023Dataset": "betaxanthin",
    "AminoAcidMulleder2016Dataset": "metabolome",
    "MetaboliteZelezniak2018Dataset": "metabolome",
    "MetaboliteDaSilveira2014Dataset": "metabolome",
    "IsobutanolScreenLopez2024Dataset": "isobutanol",
    "IsobutanolValidatedLopez2024Dataset": "isobutanol",
}
PRODUCTION_DS = [
    "IsobutanolScreenLopez2024Dataset",
    "IsobutanolValidatedLopez2024Dataset",
    "BetaxanthinCachera2023Dataset",
    "CarotenoidOzaydin2013Dataset",
]
METABOLOME_DS = [
    "AminoAcidMulleder2016Dataset",
    "MetaboliteZelezniak2018Dataset",
    "MetaboliteDaSilveira2014Dataset",
]
# experiment_type (schema) -> semantic bucket for the post-build LMDB census.
_MODALITY_OF = {
    "visual_score": "beta_carotene_score",
    "metabolite": "metabolite",  # refined per-dataset below via dataset_name
}


def _dataset_records(
    driver: Driver, dataset_id: str
) -> tuple[set[frozenset[str]], set[str], dict[int, int], dict[str, int]]:
    """Return (unique gene-sets, unique genes, gene-set-size hist, ptype hist).

    ``genes`` is the flat set of every perturbed systematic gene name across the
    dataset (deletions AND heterologous additions); ``gene_sets`` are the unique
    FULL perturbation frozensets -- the exact key ``GenotypeAggregator`` uses.
    """
    q = """
    MATCH (d:Dataset {id:$id})<-[:ExperimentMemberOf]-(e:Experiment)
    MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
    WITH e, g, [(g)<-[:PerturbationMemberOf]-(p) | p] AS perts
    RETURN [p IN perts | p.systematic_gene_name] AS genes,
           [p IN perts | p.perturbation_type] AS ptypes
    """
    gene_sets: set[frozenset[str]] = set()
    genes: set[str] = set()
    sizes: Counter[int] = Counter()
    ptypes: Counter[str] = Counter()
    with driver.session(database=NEO4J_DB) as s:
        for r in s.run(q, id=dataset_id).data():
            gl = r["genes"]
            gene_sets.add(frozenset(gl))
            genes |= set(gl)
            sizes[len(gl)] += 1
            for pt in r["ptypes"]:
                ptypes[pt] += 1
    return gene_sets, genes, dict(sizes), dict(ptypes)


def _deletion_gene_sets(driver: Driver, dataset_id: str) -> set[frozenset[str]]:
    """Unique frozensets of DELETION-only systematic gene names for a dataset.

    Strips heterologous ``gene_addition`` cassettes so a pigment production strain
    is represented by the S288C locus it deletes -- the basis on which a metabolome
    single-KO and a pigment strain could ever be biologically compared.
    """
    q = """
    MATCH (d:Dataset {id:$id})<-[:ExperimentMemberOf]-(e:Experiment)
    MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
    WITH e, g, [(g)<-[:PerturbationMemberOf]-(p)
        WHERE p.perturbation_type ENDS WITH 'deletion' | p.systematic_gene_name] AS dels
    RETURN dels
    """
    out: set[frozenset[str]] = set()
    with driver.session(database=NEO4J_DB) as s:
        for r in s.run(q, id=dataset_id).data():
            out.add(frozenset(r["dels"]))
    return out


def _env(driver: Driver, dataset_id: str) -> dict[str, Any]:
    """Return graph_level(s) + media + temperature carried by a dataset."""
    q = """
    MATCH (d:Dataset {id:$id})<-[:ExperimentMemberOf]-(e:Experiment)
    MATCH (e)<-[:PhenotypeMemberOf]-(ph:PhenotypicFeature)
    MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)<-[:MediaMemberOf]-(m:Media)
    MATCH (env)<-[:TemperatureMemberOf]-(t:Temperature)
    RETURN count(DISTINCT e) AS n, collect(DISTINCT ph.graph_level) AS gl,
           collect(DISTINCT m.name) AS media, collect(DISTINCT t.value) AS temp
    """
    with driver.session(database=NEO4J_DB) as s:
        r = s.run(q, id=dataset_id).single()
        return {
            "n_experiments": r["n"],
            "graph_level": r["gl"],
            "media": r["media"],
            "temperature": r["temp"],
        }


def db_overlap_census() -> dict[str, Any]:
    """Compute Q1 (gene-level), Q2 (genotype co-location), Q3 (per-dataset)."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    per_dataset: dict[str, dict[str, Any]] = {}
    full_sets: dict[str, set[frozenset[str]]] = {}
    genes: dict[str, set[str]] = {}
    del_sets: dict[str, set[frozenset[str]]] = {}
    for did in DATASET_MODALITY:
        gs, gn, sizes, ptypes = _dataset_records(driver, did)
        full_sets[did] = gs
        genes[did] = gn
        del_sets[did] = _deletion_gene_sets(driver, did)
        env = _env(driver, did)
        per_dataset[did] = {
            "modality": DATASET_MODALITY[did],
            "unique_genotypes_full": len(gs),
            "unique_genes": len(gn),
            "unique_deletion_genesets": len(del_sets[did]),
            "geneset_sizes": sizes,
            "perturbation_types": ptypes,
            **env,
        }
    driver.close()

    metab_genes: set[str] = set()
    for did in METABOLOME_DS:
        metab_genes |= genes[did]

    # Q1 -- gene-level deletion overlap (single genes deleted in BOTH).
    q1: dict[str, int] = {}
    for pk in PRODUCTION_DS:
        q1[f"metabolome_genes_and_{DATASET_MODALITY[pk]}_{pk}"] = len(
            metab_genes & genes[pk]
        )
    # metabolome gene union vs each metabolome member (for context)
    q1["metabolome_gene_union"] = len(metab_genes)

    # Q2 -- genotype (full gene-set) co-location, aggregator-keyed.
    mod: dict[frozenset[str], set[str]] = defaultdict(set)
    for did, gsets in full_sets.items():
        for gsx in gsets:
            mod[gsx].add(DATASET_MODALITY[did])
    multi = {gsx: ms for gsx, ms in mod.items() if len(ms) >= 2}
    combos = {
        "+".join(sorted(c)): n
        for c, n in Counter(frozenset(ms) for ms in multi.values()).most_common()
    }
    # explicit pair breakdown for the WS11b question
    def _pair(a: str, b: str) -> int:
        return sum(1 for ms in mod.values() if a in ms and b in ms)

    q2 = {
        "total_unique_genotypes": len(mod),
        "genotypes_ge2_modalities": len(multi),
        "metabolome_and_isobutanol": _pair("metabolome", "isobutanol"),
        "metabolome_and_betaxanthin": _pair("metabolome", "betaxanthin"),
        "metabolome_and_beta_carotene_score": _pair(
            "metabolome", "beta_carotene_score"
        ),
        "isobutanol_and_betaxanthin": _pair("isobutanol", "betaxanthin"),
        "combos": combos,
    }

    # DELETION-gene-set co-location (what WOULD co-locate if cassettes were
    # stripped) -- quantifies exactly how much the cassette key-isolation costs.
    del_mod: dict[frozenset[str], set[str]] = defaultdict(set)
    for did, dsets in del_sets.items():
        for dsx in dsets:
            if dsx:
                del_mod[dsx].add(DATASET_MODALITY[did])
    del_multi = {d: ms for d, ms in del_mod.items() if len(ms) >= 2}
    del_combos = {
        "+".join(sorted(c)): n
        for c, n in Counter(frozenset(ms) for ms in del_multi.values()).most_common()
    }
    q2_deletion_keyed = {
        "genotypes_ge2_modalities": len(del_multi),
        "metabolome_and_betaxanthin": sum(
            1 for ms in del_mod.values() if "metabolome" in ms and "betaxanthin" in ms
        ),
        "metabolome_and_beta_carotene_score": sum(
            1
            for ms in del_mod.values()
            if "metabolome" in ms and "beta_carotene_score" in ms
        ),
        "combos": del_combos,
    }

    return {
        "Q3_per_dataset": per_dataset,
        "Q1_gene_level_deletion_overlap": q1,
        "Q2_genotype_colocation_full_geneset": q2,
        "Q2b_genotype_colocation_deletion_keyed": q2_deletion_keyed,
    }


def aggregation_modality_census(
    dataset_root: str, dataset_name_modality: dict[str, str]
) -> dict[str, Any]:
    """Census semantic modalities per aggregated genotype from the built LMDB.

    ``dataset_name`` (e.g. ``AminoAcidMulleder2016Dataset``, or a "+"-joined name
    after a cross-dataset mean-merge) resolves the semantic bucket -- experiment
    type alone cannot (metabolite is shared across several datasets).
    """
    lmdb_dir = osp.join(dataset_root, "aggregation", "lmdb")
    env = lmdb.open(lmdb_dir, readonly=True, lock=False, readahead=False)
    per_group_mods: list[frozenset[str]] = []
    with env.begin() as txn:
        for _, value in txn.cursor():
            pairs = json.loads(value.decode())
            mods: set[str] = set()
            for p in pairs:
                name = p["experiment"].get("dataset_name", "")
                for part in name.split("+"):
                    if part in dataset_name_modality:
                        mods.add(dataset_name_modality[part])
            per_group_mods.append(frozenset(mods))
    env.close()

    ge2 = [m for m in per_group_mods if len(m) >= 2]
    combos = {"+".join(sorted(c)): n for c, n in Counter(ge2).most_common()}
    return {
        "aggregated_groups": len(per_group_mods),
        "groups_ge2_modalities": len(ge2),
        "groups_metabolome_and_isobutanol": sum(
            1 for m in per_group_mods if "metabolome" in m and "isobutanol" in m
        ),
        "combos": combos,
    }


def build_dataset(query_path: str) -> tuple[Any, str]:
    """Build the Fig-6 Neo4jCellDataset (mirrors the Fig-3 / 010 build pattern)."""
    from torchcell.data import GenotypeAggregator, MeanExperimentDeduplicator
    from torchcell.data.graph_processor import Perturbation
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
        graph_processor=Perturbation(),
    )
    return dataset, dataset_root


def phenotype_breakdown(data: Any) -> dict[str, dict[int, int]]:
    """Return {label_name: {sample_index: n_values}} from the COO phenotype store."""
    gene = data["gene"]
    types = list(gene["phenotype_types"])
    type_idx = gene["phenotype_type_indices"].tolist()
    sample_idx = gene["phenotype_sample_indices"].tolist()
    counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for t, s in zip(type_idx, sample_idx):
        counts[types[t]][s] += 1
    return {lab: dict(smap) for lab, smap in counts.items()}


def describe_item(data: Any) -> dict[str, Any]:
    """Summarize a built HeteroData sample: node stores and label shapes."""
    info: dict[str, Any] = {"type": type(data).__name__, "labels": {}}
    gene = data["gene"] if "gene" in data.node_types else None
    if gene is not None:
        for key in gene.keys():
            val = gene[key]
            if hasattr(val, "shape"):
                info["labels"][f"gene.{key}"] = list(val.shape)
        info["phenotype_types"] = list(gene["phenotype_types"])
        info["phenotype_breakdown"] = phenotype_breakdown(data)
        info["perturbed_genes"] = list(gene["perturbed_genes"])
    return info


def main() -> None:
    load_dotenv()
    here = osp.dirname(osp.abspath(__file__))
    results_dir = osp.abspath(osp.join(here, "..", "results"))
    os.makedirs(results_dir, exist_ok=True)
    query_path = osp.abspath(osp.join(here, "..", "queries", "fig6_build.cql"))

    print("=" * 72)
    print("Fig-6 production/metabolite co-location census (live DB)")
    print("=" * 72)
    census = db_overlap_census()
    print("\nQ3 per-dataset:")
    for did, info in census["Q3_per_dataset"].items():
        print(
            f"  {did}: n={info['n_experiments']} modality={info['modality']} "
            f"graph_level={info['graph_level']} media={info['media']} "
            f"temp={info['temperature']} ptypes={info['perturbation_types']}"
        )
    print("\nQ1 gene-level deletion overlap:")
    for k, v in census["Q1_gene_level_deletion_overlap"].items():
        print(f"  {k}: {v}")
    print("\nQ2 genotype co-location (full gene-set, aggregator-keyed):")
    for k, v in census["Q2_genotype_colocation_full_geneset"].items():
        if k != "combos":
            print(f"  {k}: {v}")
    print("  combos:")
    for c, n in census["Q2_genotype_colocation_full_geneset"]["combos"].items():
        print(f"    {c}: {n}")
    print("\nQ2b deletion-keyed co-location (cassettes stripped):")
    for k, v in census["Q2b_genotype_colocation_deletion_keyed"].items():
        if k != "combos":
            print(f"  {k}: {v}")

    report: dict[str, Any] = {
        "query_file": "fig6_build.cql",
        "db_overlap_census": census,
    }

    if os.environ.get("FIG6_CENSUS_ONLY") == "1":
        with open(osp.join(results_dir, "fig6_overlap_census.json"), "w") as f:
            json.dump(report, f, indent=2)
        print("\nCENSUS_ONLY set -- skipped build.")
        return

    print("\n" + "=" * 72)
    print("Building dataset from fig6_build.cql ...")
    print("=" * 72)
    dataset, dataset_root = build_dataset(query_path)
    print(f"len(dataset) = {len(dataset)}")

    agg_census = aggregation_modality_census(dataset_root, DATASET_MODALITY)
    report["aggregation_modality_census"] = agg_census
    report["dataset_len"] = len(dataset)
    report["dataset_root"] = dataset_root
    print("\nPost-aggregation modality census (from aggregation LMDB):")
    for k, v in agg_census.items():
        print(f"  {k}: {v}")

    # Confirm a valid HeteroData sample: a per-metabolite VECTOR label (metabolite
    # screen, e.g. Mulleder 19-AA) AND, at another index, a scalar visual_score
    # (Ozaydin beta-carotene). Both must tensorize under the Perturbation processor.
    sample_info: dict[str, Any] = {
        "first_valid_index": None,
        "metabolite_vector_index": None,
        "visual_score_scalar_index": None,
    }
    cap_env = os.environ.get("FIG6_BUILD_CAP")
    scan_cap = min(len(dataset), int(cap_env)) if cap_env else len(dataset)
    for idx in range(scan_cap):
        sample = dataset[idx]
        if sample_info["first_valid_index"] is None:
            sample_info["first_valid_index"] = idx
            sample_info["first_item"] = describe_item(sample)
        bd = phenotype_breakdown(sample)
        if sample_info["metabolite_vector_index"] is None and any(
            lab == "metabolite_level" and max(smap.values()) > 1
            for lab, smap in bd.items()
        ):
            sample_info["metabolite_vector_index"] = idx
            sample_info["metabolite_vector_item"] = describe_item(sample)
        if sample_info["visual_score_scalar_index"] is None and (
            "visual_score" in bd and all(v == 1 for v in bd["visual_score"].values())
        ):
            sample_info["visual_score_scalar_index"] = idx
            sample_info["visual_score_item"] = describe_item(sample)
        if (
            sample_info["metabolite_vector_index"] is not None
            and sample_info["visual_score_scalar_index"] is not None
        ):
            break
    report["dataset_sample"] = sample_info
    print("\ndataset sample summary:")
    print(json.dumps(sample_info, indent=2))

    with open(osp.join(results_dir, "fig6_overlap_census.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {osp.join(results_dir, 'fig6_overlap_census.json')}")
    dataset.close_lmdb()


if __name__ == "__main__":
    main()
