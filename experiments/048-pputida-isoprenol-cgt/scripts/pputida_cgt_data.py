# experiments/048-pputida-isoprenol-cgt/scripts/pputida_cgt_data.py
# [[experiments.048-pputida-isoprenol-cgt.scripts.pputida_cgt_data]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/048-pputida-isoprenol-cgt/scripts/pputida_cgt_data
"""Assemble the P. putida isoprenol cell graph, the dataset, and the splits.

The bacterial counterpart of ``torchcell/scratch/load_batch_005.py``, which is
the canonical yeast assembly. The core objects are organism-blind and live in
``torchcell/``; everything that is specific to this campaign is here, which is
the split the owner asked for.

What goes into the cell graph:

===================  ======================================================
gene nodes           5,729 KT2440 loci from the pinned GCA_000007565.2
gene-gene edges      the six STRING v12.5 evidence channels, taxon 160488
reaction/metabolite  the isoprenol-augmented iJN1463, pinned by sha256
gene features        one-hot, codon frequency, random, or GenSLM
phenotype            isoprenol titer, ug/mL, 502 served records
perturbation         the CRISPRi guides, 1 to 4 per strain, on PP_ tags
===================  ======================================================

The CRISPRi guides are the perturbation and the titer is the label, so the
record is exactly the shape the yeast module predicts from: an edit applied to a
reference cell that already carries its production pathway. The five pIY670
pathway genes are constant across every strain, so they are background rather
than signal, and the model never sees them as a gene token because they are not
KT2440 loci. ``Perturbation`` indexes a perturbed gene only when it is in the
cell graph, so they drop out of ``perturbation_indices`` without any filtering
here.

**The DBTL cycle.** The cycle is the campaign's time axis and the fidelity tier,
and it is not a field on a served record. It is recovered here from the per-cycle
control titer that each record's own ``ExperimentReference`` carries: the
campaign normalized every cycle to its no-gRNA controls, so the reference titer
takes exactly seven distinct values over the 465 campaign records, one per cycle.
:func:`campaign_table` does that join and then asserts the per-cycle counts
against the published campaign, so a join that ever stops holding fails loudly
instead of silently mislabeling a cycle. The counts are independently
corroborated by ``preprocess/pass_filter.csv`` in the loader's own build
directory, which carries a ``cycle`` column in LMDB index order.

A record is a campaign record when its genotype carries no deletion and no
native-copy addition. The 37 Source Data panel strains all carry one or the
other, which is what separates them from the 465 campaign strains.
"""

from __future__ import annotations

import json
import logging
import os
import os.path as osp
import random
from collections import Counter, defaultdict
from typing import Any

import lmdb
import pandas as pd
from dotenv import load_dotenv

from torchcell.data import Neo4jCellDataset, ReferenceKeyedGenotypeAggregator
from torchcell.data.graph_processor import Perturbation
from torchcell.datamodules import CellDataModule
from torchcell.datasets.bacteria_common import bacterial_genome
from torchcell.graph.string_db import (
    STRING_SCORE_CHANNELS,
    StringTaxonMirror,
    build_string_gene_multigraph,
)
from torchcell.metabolism.bigg_gem import BiggGEM
from torchcell.sequence import GeneSet

load_dotenv()

log = logging.getLogger(__name__)

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

EXPERIMENT_DIR = osp.join(EXPERIMENT_ROOT, "048-pputida-isoprenol-cgt")
QUERY_PATH = osp.join(EXPERIMENT_DIR, "queries", "isoprenol_titer_048.cql")
RESULTS = osp.join(EXPERIMENT_DIR, "results")

KT2440_TAXON = 160488
STRING_RELEASE = "12.5"
DEFAULT_STRING_ROOT = "/bulk/torchcell-string"
DEFAULT_GEM_ROOT = osp.join(DATA_ROOT, "torchcell-gems")
ISOPRENOL_MODEL_ID = "pputida_isoprenol"

#: The phenotype label, as the served ``ProductTiterPhenotype`` names it.
TITER_LABEL = "titer"

#: The per-cycle no-gRNA control titer, in ug/mL, read off the campaign records'
#: own ExperimentReference. Seven cycles, seven values, measured 2026-11-17 from
#: the served graph and independently from the loader's build directory.
CYCLE_BY_CONTROL_TITER: dict[float, int] = {
    169.7658: 0,
    154.4024: 1,
    156.7333: 2,
    168.8314: 3,
    152.8553: 4,
    163.0200: 5,
    155.1945: 6,
}

#: The published campaign shape, asserted by :func:`campaign_table`. The 465
#: campaign measurements are 465 samples, one per (construct, cycle). The 37
#: Source Data panel measurements become 32 samples, because five of them repeat
#: a strain already measured in the same panel context: two strains appear three
#: times and one twice, and those are replicates of one quantity rather than
#: separate experiments, so they aggregate. The assertion is therefore on the 37
#: records, not on a sample count.
EXPECTED_CYCLE_COUNTS: dict[int, int] = {
    0: 121,
    1: 57,
    2: 59,
    3: 54,
    4: 57,
    5: 58,
    6: 59,
}
EXPECTED_CAMPAIGN_RECORDS = 465
EXPECTED_PANEL_RECORDS = 37

#: Perturbation types that mark a Source Data panel strain rather than a
#: campaign strain.
PANEL_PERTURBATION_TYPES = frozenset({"deletion_strain", "gene_addition"})
CRISPRI_PERTURBATION_TYPE = "crispr_interference"


def pputida_genome() -> Any:
    """The pinned KT2440 genome, read-only."""
    return bacterial_genome("pputida", "KT2440")


def string_multigraph(
    gene_set: GeneSet,
    channels: tuple[str, ...] = STRING_SCORE_CHANNELS,
    string_root: str = DEFAULT_STRING_ROOT,
    release: str = STRING_RELEASE,
    above_score: int = 0,
) -> Any:
    """The STRING gene multigraph for KT2440.

    STRING protein ids for taxon 160488 are the ``PP_`` locus tags themselves,
    so no alias table is needed: measured over the v12.5 slice, all 1,324,300
    links have both ends in the genome's gene set and no STRING protein falls
    outside it.
    """
    mirror = StringTaxonMirror(root=string_root, taxon=KT2440_TAXON, release=release)
    return build_string_gene_multigraph(
        mirror, gene_set, channels=channels, above_score=above_score
    )


def metabolism_incidence(
    model_id: str = ISOPRENOL_MODEL_ID, gem_root: str = DEFAULT_GEM_ROOT
) -> dict[str, Any]:
    """The ``incidence_graphs`` argument for the metabolic module.

    ``induced_gene_set`` is deliberately not used. Restricting the model to the
    genome's gene set would drop every OR term whose genes are all outside it,
    and that is exactly the heterologous route: ``APP`` (the pIY670 acid
    phosphatase that releases isoprenol) is gated by ``aphA``, which is not a
    KT2440 locus. Dropping it would remove the product's own reaction from the
    graph. The genes a GEM names but the cell graph does not carry are mapped to
    -1 and left out of the gene-to-reaction edges downstream, which keeps the
    reaction node and drops only the edge.
    """
    gem = BiggGEM(model_id, gem_root=gem_root)
    return {"metabolism_bipartite": gem.bipartite_graph}


def build_node_embeddings(
    genome: Any, names: tuple[str, ...], data_root: str = DATA_ROOT
) -> dict[str, Any]:
    """The gene feature datasets, by name.

    One-hot and random are the floor the MVP measures against; codon frequency
    is the cheapest real sequence feature; GenSLM is the bacterial language
    model whose PATRIC weights are already on this machine. None of them is
    bound to yeast despite the annotations in their modules, since each reads
    only ``genome.gene_set`` and a gene's sequence.

    **Only one_hot and random run today.** 165 of the 5,729 KT2440 loci are
    non-coding and carry no CDS and no protein: 75 tRNA (``PP_t##``), 67 misc
    RNA (``PP_mr##``), 21 rRNA (``PP_16S*``/``PP_23S*``/``PP_5S*``) and 2 others.
    ``CodonFrequencyDataset`` reads ``genome[gene].cds.seq`` unconditionally and
    raises ``AttributeError: 'NoneType' object has no attribute 'seq'`` on the
    first of them, and every other CDS- or protein-based encoder will do the
    same, GenSLM included. A bacterial gene set is not a yeast ORF set, and what
    those 165 genes should contribute is a decision, not a default: a zero
    vector, a reserved out-of-vocabulary token, or exclusion from the gene set.
    Until that is decided the encoder ladder stops at one_hot and random.
    """
    embeddings: dict[str, Any] = {}
    for name in names:
        if name == "one_hot":
            from torchcell.datasets.one_hot_gene import OneHotGeneDataset

            embeddings[name] = OneHotGeneDataset(
                root=osp.join(data_root, "data/pputida/kt2440/one_hot_embedding"),
                genome=genome,
            )
        elif name == "codon_frequency":
            from torchcell.datasets.codon_frequency import CodonFrequencyDataset

            embeddings[name] = CodonFrequencyDataset(
                root=osp.join(
                    data_root, "data/pputida/kt2440/codon_frequency_embedding"
                ),
                genome=genome,
            )
        elif name.startswith("random_"):
            from torchcell.datasets.random_embedding import RandomEmbeddingDataset

            size = int(name.split("_")[1])
            embeddings[name] = RandomEmbeddingDataset(
                root=osp.join(
                    data_root, f"data/pputida/kt2440/random_{size}_embedding"
                ),
                genome=genome,
                model_name=f"random_{size}",
            )
        else:
            raise ValueError(
                f"unknown node embedding {name!r}; this experiment serves "
                f"one_hot, codon_frequency and random_<size>"
            )
    return embeddings


def build_dataset(
    root: str | None = None,
    graph_channels: tuple[str, ...] = STRING_SCORE_CHANNELS,
    embedding_names: tuple[str, ...] = ("one_hot",),
    use_metabolism: bool = True,
    string_root: str = DEFAULT_STRING_ROOT,
    gem_root: str = DEFAULT_GEM_ROOT,
) -> Neo4jCellDataset:
    """Build or open the isoprenol CGT dataset.

    The aggregator is ``ReferenceKeyedGenotypeAggregator``, not the
    genotype-keyed one the yeast builds use. 502 served records carry only 432
    distinct genotypes, because the campaign re-measured 50 cycle-2 designs in
    cycle 3 on the repaired dCas9 plasmid. A genotype-keyed aggregator would
    average those pairs into one sample and erase the fidelity-tier contrast the
    campaign was run to produce. Keying on the genotype together with the
    record's own per-cycle control reference keeps them apart, so all 502
    records survive as 502 samples. ``deduplicator`` stays ``None`` for the same
    reason: ``MeanExperimentDeduplicator`` keys on the gene set and the
    experiment type, which would merge the same pairs.

    The result is **497 samples from 502 served records**. All 465 campaign
    records survive as 465 distinct samples, which is what matters. The five
    absorbed records are three Source Data panel groups whose members differ in
    nothing the store carries except the titer and its uncertainty, so they are
    replicates of one quantity.

    Grouping by design happens at split time instead, in :func:`design_splits`,
    which is where it belongs: the records stay distinct and the leak is closed
    by keeping a design's records in one split.
    """
    genome = pputida_genome()
    with open(QUERY_PATH) as handle:
        query = handle.read()

    dataset_root = root or osp.join(
        DATA_ROOT, "data/torchcell/experiments/048-pputida-isoprenol-cgt/isoprenol_cgt"
    )
    incidence = metabolism_incidence(gem_root=gem_root) if use_metabolism else None
    embeddings = (
        build_node_embeddings(genome, embedding_names) if embedding_names else None
    )

    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=string_multigraph(
            genome.gene_set, channels=graph_channels, string_root=string_root
        ),
        incidence_graphs=incidence,
        node_embeddings=embeddings,
        converter=None,
        deduplicator=None,
        aggregator=ReferenceKeyedGenotypeAggregator,
        graph_processor=Perturbation(),
        phenotype_labels=[TITER_LABEL],
    )
    # Neo4jCellDataset consumes the embedding datasets into the cell graph's
    # gene.x and does not keep them, but the model needs the objects themselves
    # to infer its input width, so they are attached here rather than rebuilt.
    dataset.node_embedding_datasets = embeddings or {}
    return dataset


def _processed_groups(dataset_root: str) -> list[list[dict[str, Any]]]:
    """Read the dataset's PROCESSED record store, in dataset index order.

    The processed store is what ``dataset[i]`` indexes, so reading it is what
    makes a row of :func:`campaign_table` line up with sample ``i``. Reading the
    raw store instead would be an index bug waiting to happen: aggregation
    groups the 502 raw records into 497 samples, so raw index 500 is not sample
    500.

    Each value is a list of the records in that group, carrying both the
    experiment and its reference. The cycle join needs the reference, which the
    processed HeteroData does not keep.

    The aggregated store keys records by their integer index as a string, which
    is the format the dataset's own index passes read. The raw store keys them
    ``data_{i}`` instead, so the two are not interchangeable.
    """
    path = osp.join(dataset_root, "processed", "lmdb")
    env = lmdb.open(path, readonly=True, lock=False, subdir=True)
    groups: list[list[dict[str, Any]]] = []
    with env.begin() as txn:
        index = 0
        while True:
            value = txn.get(str(index).encode())
            if value is None:
                break
            payload = json.loads(value.decode())
            groups.append(payload if isinstance(payload, list) else [payload])
            index += 1
    env.close()
    return groups


def campaign_table(dataset: Neo4jCellDataset) -> pd.DataFrame:
    """One row per dataset record: cycle, design, guide count and titer.

    Raises:
        ValueError: If the recovered per-cycle counts do not match the published
            campaign. The cycle is reconstructed rather than stored, so it is
            gated on reproducing a shape that is known independently.
    """
    groups = _processed_groups(dataset.root)
    if len(groups) != len(dataset):
        raise ValueError(
            f"processed store holds {len(groups)} groups but the dataset reports "
            f"{len(dataset)} samples; the record table would not align"
        )
    rows = []
    for index, group in enumerate(groups):
        first = group[0]
        experiment = first["experiment"]
        reference = first["experiment_reference"]
        perturbations = experiment["genotype"]["perturbations"]
        types = {p["perturbation_type"] for p in perturbations}
        guides = sorted(
            p["systematic_gene_name"]
            for p in perturbations
            if p["perturbation_type"] == CRISPRI_PERTURBATION_TYPE
        )
        is_campaign = not (types & PANEL_PERTURBATION_TYPES)
        control = round(float(reference["phenotype_reference"]["titer"]), 4)
        cycle = CYCLE_BY_CONTROL_TITER.get(control) if is_campaign else None
        titers = [float(member["experiment"]["phenotype"]["titer"]) for member in group]
        rows.append(
            {
                "index": index,
                "group_size": len(group),
                "is_campaign": is_campaign,
                "cycle": cycle,
                "control_titer": control,
                "design_key": "|".join(guides) if guides else "no_guide",
                "n_guides": len(guides),
                "titer": sum(titers) / len(titers),
                "titer_unit": experiment["phenotype"]["titer_unit"],
                # Summed over the group, not read off its first member, so this
                # column describes the same thing the averaged titer does.
                "n_samples": sum(
                    int(member["experiment"]["phenotype"]["n_samples"])
                    for member in group
                ),
            }
        )
    table = pd.DataFrame(rows)

    merged_campaign = table[table.is_campaign & (table.group_size > 1)]
    if len(merged_campaign) != 0:
        raise ValueError(
            f"{len(merged_campaign)} campaign samples aggregate more than one "
            f"record, which would average across DBTL cycles: indices "
            f"{sorted(merged_campaign['index'])[:10]}"
        )

    campaign = table[table.is_campaign]
    counts = Counter(int(c) for c in campaign.cycle.dropna())
    if len(campaign) != EXPECTED_CAMPAIGN_RECORDS:
        raise ValueError(
            f"expected {EXPECTED_CAMPAIGN_RECORDS} campaign records, "
            f"found {len(campaign)}"
        )
    if int(campaign.cycle.isna().sum()) != 0:
        raise ValueError(
            f"{int(campaign.cycle.isna().sum())} campaign records did not match "
            f"a per-cycle control titer; the cycle join no longer holds"
        )
    if dict(sorted(counts.items())) != EXPECTED_CYCLE_COUNTS:
        raise ValueError(
            f"recovered per-cycle counts {dict(sorted(counts.items()))} do not "
            f"match the published campaign {EXPECTED_CYCLE_COUNTS}"
        )
    panel_records = int(table.loc[~table.is_campaign, "group_size"].sum())
    if panel_records != EXPECTED_PANEL_RECORDS:
        raise ValueError(
            f"expected {EXPECTED_PANEL_RECORDS} panel records, found {panel_records}"
        )
    return table


def design_splits(
    table: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    campaign_only: bool = True,
) -> dict[str, list[int]]:
    """Random split over DESIGNS, not records.

    The MVP criterion is a split by design. A design measured in two cycles is
    two records, so splitting records would put the same guide array in train and
    in test and report a leak as generalization. Every record of a design lands
    in one split.
    """
    subset = table[table.is_campaign] if campaign_only else table
    by_design: dict[str, list[int]] = defaultdict(list)
    for design_key, record_index in zip(
        subset["design_key"], subset["index"], strict=True
    ):
        by_design[str(design_key)].append(int(record_index))

    designs = sorted(by_design)
    rng = random.Random(seed)
    rng.shuffle(designs)

    n_train = int(len(designs) * ratios[0])
    n_val = int(len(designs) * ratios[1])
    assignment = {
        "train": designs[:n_train],
        "val": designs[n_train : n_train + n_val],
        "test": designs[n_train + n_val :],
    }
    return {
        name: sorted(i for design in group for i in by_design[design])
        for name, group in assignment.items()
    }


def cycle_splits(
    table: pd.DataFrame,
    holdout_cycle: int = 6,
    val_cycle: int = 5,
    design_disjoint: bool = True,
) -> dict[str, list[int]]:
    """Cycle extrapolation: train on the earlier cycles, predict a later one.

    ``val_cycle`` is held out of training as well, so the model selected for the
    held-out cycle was never fit on it. Cycle 6 is the campaign's last and widest
    cycle, so cycle 5 is worth reporting beside it rather than instead of it.

    **``design_disjoint`` closes a leak that the cycle boundary does not.** The
    campaign re-measured designs across cycles, so a design tested in cycle 6 can
    already have been measured in cycle 4 or 5. Measured at the defaults, three
    cycle-6 designs were also in train or val, which is 3 of 59 test records, and
    a test Pearson computed over them is partly a memory score. The earlier
    measurements are dropped from train and val rather than the later ones from
    test, so the held-out cycle stays the cycle as released and the condition
    becomes what it claims to be: this design has never been seen.

    Raises:
        ValueError: If the three splits do not partition the campaign. At a
            ``holdout_cycle``/``val_cycle`` pair that leaves a cycle out of all
            three, the function would otherwise silently return a split over
            part of the campaign.
    """
    campaign = table[table.is_campaign]
    train = campaign[campaign.cycle < val_cycle]
    val = campaign[campaign.cycle == val_cycle]
    test = campaign[campaign.cycle == holdout_cycle]

    covered = len(train) + len(val) + len(test)
    if covered != len(campaign):
        raise ValueError(
            f"cycles {sorted(campaign.cycle.unique())} with holdout "
            f"{holdout_cycle} and val {val_cycle} cover {covered} of "
            f"{len(campaign)} campaign records; the remainder would be silently "
            f"dropped. Name a holdout and val cycle that partition the campaign."
        )

    splits = {
        "train": sorted(int(i) for i in train["index"]),
        "val": sorted(int(i) for i in val["index"]),
        "test": sorted(int(i) for i in test["index"]),
    }
    if not design_disjoint:
        return splits

    by_index = dict(zip(campaign["index"], campaign["design_key"], strict=True))
    test_designs = {by_index[i] for i in splits["test"]}
    for name in ("train", "val"):
        kept = [i for i in splits[name] if by_index[i] not in test_designs]
        dropped = len(splits[name]) - len(kept)
        if dropped:
            log.info(
                "cycle split: dropped %d %s record(s) whose design is in the "
                "held-out cycle %d",
                dropped,
                name,
                holdout_cycle,
            )
        splits[name] = kept
    return splits


def build_datamodule(
    dataset: Neo4jCellDataset,
    splits: dict[str, list[int]],
    batch_size: int = 16,
    num_workers: int = 2,
    seed: int = 42,
    cache_tag: str = "design",
) -> CellDataModule:
    """Wrap the dataset in the shared data module, pinned to explicit splits.

    ``pinned_split_indices`` assigns a record to a split but never excludes one,
    so ``index_subset`` is passed as the union of the three splits to restrict
    the record pool to exactly what was assigned. Without it the records outside
    the splits, the 37 Source Data panel strains under a campaign-only split,
    would be seed-assigned around the pinned ones and quietly join training.
    """
    pool = sorted({index for group in splits.values() for index in group})
    module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset.root, f"data_module_cache_{cache_tag}"),
        batch_size=batch_size,
        random_seed=seed,
        num_workers=num_workers,
        pin_memory=False,
        split_indices=["phenotype_label_index"],
        follow_batch=["perturbation_indices", "phenotype_values"],
        pinned_split_indices=splits,
        index_subset=pool,
    )
    module.setup()
    return module


def load_sample_data_batch(
    batch_size: int = 4,
    num_workers: int = 0,
    embedding_names: tuple[str, ...] = ("one_hot",),
    use_metabolism: bool = True,
    split: str = "design",
    seed: int = 42,
) -> tuple[Neo4jCellDataset, Any, pd.DataFrame, dict[str, list[int]]]:
    """Build everything and return one batch, for inspecting the data objects."""
    dataset = build_dataset(
        embedding_names=embedding_names, use_metabolism=use_metabolism
    )
    table = campaign_table(dataset)
    splits = (
        design_splits(table, seed=seed) if split == "design" else cycle_splits(table)
    )
    module = build_datamodule(
        dataset,
        splits,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        cache_tag=split,
    )
    batch = next(iter(module.train_dataloader()))
    return dataset, batch, table, splits


def main() -> None:
    """Print the cell graph, one record, one batch and the split sizes."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(RESULTS, exist_ok=True)

    dataset, batch, table, splits = load_sample_data_batch()

    print("\n=== cell graph ===")
    print(dataset.cell_graph)
    print("\n=== dataset[0] ===")
    print(dataset[0])
    print("\n=== batch ===")
    print(batch)
    print(f"\nrecords {len(dataset)}")
    print(
        f"campaign {int(table.is_campaign.sum())}, panel {int((~table.is_campaign).sum())}"
    )
    print(
        f"per cycle {dict(sorted(Counter(int(c) for c in table.cycle.dropna()).items()))}"
    )
    print(f"designs {table[table.is_campaign].design_key.nunique()}")
    print("design split sizes " + ", ".join(f"{k} {len(v)}" for k, v in splits.items()))
    cycles = cycle_splits(table)
    print("cycle split sizes " + ", ".join(f"{k} {len(v)}" for k, v in cycles.items()))

    table.to_csv(osp.join(RESULTS, "isoprenol_cgt_record_table.csv"), index=False)
    with open(osp.join(RESULTS, "isoprenol_cgt_splits.json"), "w") as handle:
        json.dump({"design": splits, "cycle": cycles}, handle, indent=1)
    print(f"\nresults -> {RESULTS}")


if __name__ == "__main__":
    main()
