# tests/torchcell/datasets/scerevisiae/test_kuzmin2018_synthetic.py
# [[tests.torchcell.datasets.scerevisiae.test_kuzmin2018_synthetic]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_kuzmin2018_synthetic.py
"""Hermetic end-to-end builds of the five Kuzmin 2018 loaders from a hand-written raw table.

The raw file ``aao1729_data_s1.tsv`` is written under ``<tmp_path>/<root>/raw/`` with
the eleven columns the loaders read, so PyG never calls ``download()``; ``process()``
runs once (no ``processed/lmdb`` yet) and ``@post_process`` writes ``gene_set.json``,
``experiment_reference_index.json`` and ``build_manifest.json`` (git + hostname only).
No genome and no ``$DATA_ROOT`` are involved.

The shared table has three rows and exercises every class:

======  ==========================  ===================  ====  ========  =====  ========  ====  =========  ======
row     query strain / alleles      array strain          type  combined  sd     query     smf   epsilon    p
======  ==========================  ===================  ====  ========  =====  ========  ====  =========  ======
0       YAR002W+YDL227C_tm3180      YAL048C_dma5203       dig   0.8103    0.0463 0.9       0.95  -0.05      0.2
        nup60Δ+hoΔ                  gem1Δ
1       YAR002W+YDL227C_tm3180      YBR001C_tsa100        dig   0.7       0.03   0.9       0.8   -0.02      0.5
        nup60Δ+hoΔ                  nth2-5001
2       YAR002W+YML107C_tm2550      YAL048C_dma5203       tri   0.4       0.02   0.5128    0.95  -0.1       0.01
        nup60Δ+pml39Δ               gem1Δ
======  ==========================  ===================  ====  ========  =====  ========  ====  =========  ======

Preprocessing rewrites ``Δ`` to ``_delta`` (and ``'`` to ``_prime``); the ``hoΔ`` /
``YDL227C`` half of a digenic query is stripped from the "no ho" columns; ``dma`` array
strains are KanMX deletions and ``tsa`` strains temperature-sensitive alleles. Every
record carries ``SGA_TM_SELECTION`` at 26 C, reference genome S288C, and PubMed
29674565 / DOI 10.1126/science.aao1729. Combined-mutant SDs are labeled
``sample_sd`` over ``N_SAMPLES_COMBINED_MUTANT = 4`` colonies, so ``fitness_se`` is
``sd / 2``.
"""

import json
import pickle
from pathlib import Path
from typing import Any

import lmdb
import pandas as pd
import pytest

from torchcell.data.data import ExperimentReferenceIndex
from torchcell.datamodels.media import SGA_TM_SELECTION
from torchcell.datamodels.schema import (
    Environment,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    GeneInteractionExperiment,
    GeneInteractionExperimentReference,
    GeneInteractionPhenotype,
    Genotype,
    Publication,
    ReferenceGenome,
    SampleUnit,
    SgaKanMxDeletionPerturbation,
    SgaTsAllelePerturbation,
    Temperature,
    UncertaintyType,
)
from torchcell.datasets.scerevisiae.kuzmin2018 import (
    N_SAMPLES_COMBINED_MUTANT,
    RECORD_KIND_DIGENIC_ARRAY_CROSS,
    RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN,
    DmfKuzmin2018Dataset,
    DmiKuzmin2018Dataset,
    SmfKuzmin2018Dataset,
    TmfKuzmin2018Dataset,
    TmiKuzmin2018Dataset,
)

RAW_NAME = "aao1729_data_s1.tsv"
COLUMNS = [
    "Query strain ID",
    "Query allele name",
    "Array strain ID",
    "Array allele name",
    "Combined mutant type",
    "Combined mutant fitness",
    "Combined mutant fitness standard deviation",
    "Query single/double mutant fitness",
    "Array single mutant fitness",
    "Adjusted genetic interaction score (epsilon or tau)",
    "P-value",
]
DIGENIC_QUERY = "YAR002W+YDL227C_tm3180"
TRIGENIC_QUERY = "YAR002W+YML107C_tm2550"
ARRAY_DMA = "YAL048C_dma5203"
ARRAY_TSA = "YBR001C_tsa100"
ROWS: list[list[Any]] = [
    [DIGENIC_QUERY, "nup60Δ+hoΔ", ARRAY_DMA, "gem1Δ", "digenic", 0.8103, 0.0463, 0.9, 0.95, -0.05, 0.2],
    [DIGENIC_QUERY, "nup60Δ+hoΔ", ARRAY_TSA, "nth2-5001", "digenic", 0.7, 0.03, 0.9, 0.8, -0.02, 0.5],
    [TRIGENIC_QUERY, "nup60Δ+pml39Δ", ARRAY_DMA, "gem1Δ", "trigenic", 0.4, 0.02, 0.5128, 0.95, -0.1, 0.01],
]  # fmt: skip

ENVIRONMENT = Environment(media=SGA_TM_SELECTION, temperature=Temperature(value=26))
GENOME = ReferenceGenome(species="Saccharomyces cerevisiae", strain="S288C")
PUBLICATION = Publication(
    pubmed_id="29674565",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/29674565/",
    doi="10.1126/science.aao1729",
    doi_url="https://www.science.org/doi/10.1126/science.aao1729",
)
# Perturbations as the loaders name them after the Δ -> _delta rewrite.
NUP60_DIGENIC = SgaKanMxDeletionPerturbation(
    systematic_gene_name="YAR002W", perturbed_gene_name="nup60_delta", strain_id=DIGENIC_QUERY
)  # fmt: skip
NUP60_TRIGENIC = SgaKanMxDeletionPerturbation(
    systematic_gene_name="YAR002W", perturbed_gene_name="nup60_delta", strain_id=TRIGENIC_QUERY
)  # fmt: skip
PML39_TRIGENIC = SgaKanMxDeletionPerturbation(
    systematic_gene_name="YML107C", perturbed_gene_name="pml39_delta", strain_id=TRIGENIC_QUERY
)  # fmt: skip
GEM1 = SgaKanMxDeletionPerturbation(
    systematic_gene_name="YAL048C", perturbed_gene_name="gem1_delta", strain_id=ARRAY_DMA
)  # fmt: skip
NTH2_TS = SgaTsAllelePerturbation(
    systematic_gene_name="YBR001C", perturbed_gene_name="nth2-5001", strain_id=ARRAY_TSA
)  # fmt: skip


def write_raw(root: Path, rows: list[list[Any]]) -> None:
    """Write ``rows`` as the tab-separated raw table under ``root/raw``."""
    (root / "raw").mkdir(parents=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(
        root / "raw" / RAW_NAME, sep="\t", index=False
    )


def build(tmp_path: Path, cls: type[Any], rows: list[list[Any]] = ROWS) -> Any:
    """Build ``cls`` under ``tmp_path/<class name>`` from ``rows``."""
    root = tmp_path / cls.__name__
    write_raw(root, rows)
    return cls(root=str(root))


def sample_sd(sd: float) -> dict[str, Any]:
    """The uncertainty fields a combined-mutant SD carries (n = 4 colonies)."""
    return {
        "fitness_uncertainty": sd,
        "fitness_uncertainty_type": UncertaintyType.sample_sd,
        "n_samples": N_SAMPLES_COMBINED_MUTANT,
        "sample_unit": SampleUnit.colony,
    }


def fitness_reference(reference_sd: float) -> FitnessExperimentReference:
    """The fitness reference every record of a fitness loader shares."""
    return FitnessExperimentReference(
        dataset_name="",  # replaced per class below
        genome_reference=GENOME,
        environment_reference=ENVIRONMENT,
        phenotype_reference=FitnessPhenotype(
            fitness=1.0, fitness_std=reference_sd, **sample_sd(reference_sd)
        ),
    )


def experiments(ds: Any) -> list[Any]:
    """Every stored experiment, validated back into the loader's experiment class."""
    return [
        ds.experiment_class.model_validate(ds[i]["experiment"]) for i in range(len(ds))
    ]


def references(ds: Any) -> list[Any]:
    """Every stored reference, validated back into the loader's reference class."""
    return [
        ds.reference_class.model_validate(ds[i]["reference"]) for i in range(len(ds))
    ]


def dumps(models: list[Any]) -> list[dict[str, Any]]:
    """``model_dump`` of each model, for exact list equality."""
    return [m.model_dump() for m in models]


def assert_side_files(ds: Any, gene_set: list[str], n: int) -> None:
    """Pin the four post-process artifacts of a build with ``n`` records.

    ``gene_set.json`` is the sorted systematic names; the reference index has ONE
    entry whose ``member_indices`` are ``0..n-1`` and whose reference equals the
    reference stored on record 0; the build manifest names the root, class and
    module; ``processed/interned`` exists beside ``processed/lmdb``.
    """
    root = Path(ds.root)
    assert json.loads((root / "preprocess" / "gene_set.json").read_text()) == gene_set
    assert sorted(ds.gene_set) == gene_set

    stored = json.loads(
        (root / "preprocess" / "experiment_reference_index.json").read_text()
    )
    assert len(stored) == 1
    index = ExperimentReferenceIndex.from_stored(stored[0])
    assert index.member_indices == list(range(n))
    assert index.reference.model_dump() == references(ds)[0].model_dump()

    manifest = json.loads((root / "preprocess" / "build_manifest.json").read_text())
    assert manifest["dataset_name"] == type(ds).__name__
    assert manifest["loader_class"] == type(ds).__name__
    assert manifest["loader_module"] == "torchcell.datasets.scerevisiae.kuzmin2018"
    assert (root / "processed" / "lmdb").is_dir()
    assert (root / "processed" / "interned").is_dir()


def raw_record(ds: Any, idx: int) -> dict[str, Any]:
    """The pickled LMDB record at ``idx`` BEFORE interned ``$ref`` resolution."""
    env = lmdb.open(
        str(Path(ds.root) / "processed" / "lmdb"), readonly=True, lock=False
    )
    with env.begin() as txn:
        raw = txn.get(str(idx).encode())
    env.close()
    assert raw is not None, f"no record at index {idx}"
    record: dict[str, Any] = pickle.loads(raw)
    return record


def test_smf_records(tmp_path: Path) -> None:
    """Smf: 3 records, array singles first (rows 0, 1) then the digenic query single.

    Array single-mutant fitness is "Array single mutant fitness" (0.95 for gem1Δ,
    0.8 for nth2-5001); the query single is "Query single/double mutant fitness"
    (0.9) named without the hoΔ half, strain id the full query strain. The trigenic
    query (a double mutant) yields no single-mutant record. No SD is reported for
    single mutants, so every phenotype has ``fitness_std`` None and no uncertainty.
    """
    ds = build(tmp_path, SmfKuzmin2018Dataset)
    assert len(ds) == 3
    expected = [
        FitnessExperiment(
            dataset_name="SmfKuzmin2018Dataset",
            genotype=Genotype(perturbations=[GEM1]),
            environment=ENVIRONMENT,
            phenotype=FitnessPhenotype(fitness=0.95, fitness_std=None),
        ),
        FitnessExperiment(
            dataset_name="SmfKuzmin2018Dataset",
            genotype=Genotype(perturbations=[NTH2_TS]),
            environment=ENVIRONMENT,
            phenotype=FitnessPhenotype(fitness=0.8, fitness_std=None),
        ),
        FitnessExperiment(
            dataset_name="SmfKuzmin2018Dataset",
            genotype=Genotype(perturbations=[NUP60_DIGENIC]),
            environment=ENVIRONMENT,
            phenotype=FitnessPhenotype(fitness=0.9, fitness_std=None),
        ),
    ]
    assert dumps(experiments(ds)) == dumps(expected)
    assert [ds[i]["publication"] for i in range(3)] == [PUBLICATION.model_dump()] * 3


def test_smf_reference_uses_mean_sd_over_all_rows(tmp_path: Path) -> None:
    """Smf reference SD is the mean combined-mutant SD over ALL rows, trigenic included.

    ``preprocess_raw`` takes the mean before the digenic filter, so it is
    ``pd.Series([0.0463, 0.03, 0.02]).mean()`` (0.0321 up to float rounding), labeled
    ``sample_sd`` over 4 colonies with ``fitness_se = sd / 2``. Dmf's reference below
    uses the digenic rows only, so the two loaders disagree on the reference noise.
    """
    ds = build(tmp_path, SmfKuzmin2018Dataset)
    reference_sd = float(pd.Series([0.0463, 0.03, 0.02]).mean())
    expected = fitness_reference(reference_sd).model_copy(
        update={"dataset_name": "SmfKuzmin2018Dataset"}
    )
    assert dumps(references(ds)) == [expected.model_dump()] * 3
    assert references(ds)[0].phenotype_reference.fitness_se == reference_sd / 2
    assert_side_files(ds, ["YAL048C", "YAR002W", "YBR001C"], 3)


def test_smf_drops_array_single_when_the_row_query_fitness_is_missing(
    tmp_path: Path,
) -> None:
    """Finding: an array single-mutant record is dropped by the QUERY-fitness filter.

    ``preprocess_raw`` ends with ``df[~df["Query single/double mutant fitness"].isna()]``
    applied to the concatenated array + query frame, so an array single whose only
    row has an empty query fitness is dropped even though its own "Array single mutant
    fitness" (0.88 here) is present. With a fourth trigenic row
    (``YBL002W+YBL003C_tm7`` x ``YCR002C_dma77`` pol32Δ, query fitness empty) the
    build still has 3 records and YCR002C is absent from the gene set.
    """
    extra = [
        "YBL002W+YBL003C_tm7", "htb2Δ+hta2Δ", "YCR002C_dma77", "pol32Δ", "trigenic",
        0.3, 0.05, None, 0.88, -0.2, 0.001,
    ]  # fmt: skip
    ds = build(tmp_path, SmfKuzmin2018Dataset, [*ROWS, extra])
    assert len(ds) == 3
    assert [e.genotype.systematic_gene_names for e in experiments(ds)] == [
        ["YAL048C"],
        ["YBR001C"],
        ["YAR002W"],
    ]
    assert sorted(ds.gene_set) == ["YAL048C", "YAR002W", "YBR001C"]


def test_dmf_records(tmp_path: Path) -> None:
    """Dmf: 2 digenic array crosses (rows 0, 1) then 1 double-mutant query strain.

    Digenic records pair the hoΔ-stripped query gene with the array gene; combined
    fitness 0.8103 / SD 0.0463 (se 0.02315) and 0.7 / 0.03 (se 0.015). The trigenic
    query strain ``YAR002W+YML107C_tm2550`` becomes one record whose genotype is the
    query PAIR (both tagged with the full strain id, no array gene) with fitness
    0.5128 from "Query single/double mutant fitness" and no SD (Data File S4 is not a
    raw file of this loader).
    """
    ds = build(tmp_path, DmfKuzmin2018Dataset)
    assert len(ds) == 3
    expected = [
        FitnessExperiment(
            dataset_name="DmfKuzmin2018Dataset",
            genotype=Genotype(perturbations=[NUP60_DIGENIC, GEM1]),
            environment=ENVIRONMENT,
            phenotype=FitnessPhenotype(
                fitness=0.8103, fitness_std=0.0463, **sample_sd(0.0463)
            ),
        ),
        FitnessExperiment(
            dataset_name="DmfKuzmin2018Dataset",
            genotype=Genotype(perturbations=[NUP60_DIGENIC, NTH2_TS]),
            environment=ENVIRONMENT,
            phenotype=FitnessPhenotype(
                fitness=0.7, fitness_std=0.03, **sample_sd(0.03)
            ),
        ),
        FitnessExperiment(
            dataset_name="DmfKuzmin2018Dataset",
            genotype=Genotype(perturbations=[NUP60_TRIGENIC, PML39_TRIGENIC]),
            environment=ENVIRONMENT,
            phenotype=FitnessPhenotype(fitness=0.5128, fitness_std=None),
        ),
    ]
    assert dumps(experiments(ds)) == dumps(expected)
    assert [e.phenotype.fitness_se for e in experiments(ds)] == [0.02315, 0.015, None]
    assert ds.df["record_kind"].tolist() == [
        RECORD_KIND_DIGENIC_ARRAY_CROSS,
        RECORD_KIND_DIGENIC_ARRAY_CROSS,
        RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN,
    ]


def test_dmf_reference_and_side_files(tmp_path: Path) -> None:
    """Dmf reference SD is the DIGENIC mean ``(0.0463 + 0.03) / 2 = 0.03815``.

    Labeled ``sample_sd`` over 4 colonies, ``fitness_se`` 0.019075; the one
    reference covers all three records. Gene set is the four systematic names.
    """
    ds = build(tmp_path, DmfKuzmin2018Dataset)
    expected = fitness_reference((0.0463 + 0.03) / 2).model_copy(
        update={"dataset_name": "DmfKuzmin2018Dataset"}
    )
    assert dumps(references(ds)) == [expected.model_dump()] * 3
    assert references(ds)[0].phenotype_reference.fitness_std == 0.03815
    assert references(ds)[0].phenotype_reference.fitness_se == 0.019075
    assert_side_files(ds, ["YAL048C", "YAR002W", "YBR001C", "YML107C"], 3)


def test_dmf_drops_query_strain_without_a_fitness(tmp_path: Path) -> None:
    """A trigenic query strain with an empty query fitness yields no record.

    ``_double_mutant_query_strain_rows`` keeps rows whose query fitness is present,
    so the fourth row (``YBL002W+YBL003C_tm7``, query fitness empty) adds nothing:
    still 3 records and neither YBL002W nor YBL003C in the gene set.
    """
    extra = [
        "YBL002W+YBL003C_tm7", "htb2Δ+hta2Δ", "YCR002C_dma77", "pol32Δ", "trigenic",
        0.3, 0.05, None, 0.88, -0.2, 0.001,
    ]  # fmt: skip
    ds = build(tmp_path, DmfKuzmin2018Dataset, [*ROWS, extra])
    assert len(ds) == 3
    assert sorted(ds.gene_set) == ["YAL048C", "YAR002W", "YBR001C", "YML107C"]


def test_dmf_interns_environment_and_reference(tmp_path: Path) -> None:
    """The stored record holds ``$ref`` pointers for environment and reference only.

    Both exceed ``INTERN_MIN_BYTES``; the pointer ``name`` is the media name for the
    environment and the dataset name for the reference. The publication (four short
    strings) stays inline, so the raw record's publication equals the full dict.
    """
    ds = build(tmp_path, DmfKuzmin2018Dataset)
    record = raw_record(ds, 0)
    assert record["experiment"]["environment"]["name"] == SGA_TM_SELECTION.name
    assert set(record["experiment"]["environment"]) == {"$ref", "name"}
    assert record["reference"]["name"] == "DmfKuzmin2018Dataset"
    assert set(record["reference"]) == {"$ref", "name"}
    assert record["publication"] == PUBLICATION.model_dump()
    assert Environment.model_validate(ds[0]["experiment"]["environment"]) == ENVIRONMENT


def test_tmf_record(tmp_path: Path) -> None:
    """Tmf: the trigenic row alone, three perturbations, fitness 0.4 / SD 0.02.

    Both query genes carry the full query strain id, the array gene its own;
    ``fitness_se`` is 0.01. The reference SD is the trigenic mean, 0.02 (one row).
    """
    ds = build(tmp_path, TmfKuzmin2018Dataset)
    assert len(ds) == 1
    expected = FitnessExperiment(
        dataset_name="TmfKuzmin2018Dataset",
        genotype=Genotype(perturbations=[NUP60_TRIGENIC, PML39_TRIGENIC, GEM1]),
        environment=ENVIRONMENT,
        phenotype=FitnessPhenotype(fitness=0.4, fitness_std=0.02, **sample_sd(0.02)),
    )
    assert dumps(experiments(ds)) == [expected.model_dump()]
    assert experiments(ds)[0].phenotype.fitness_se == 0.01
    reference = fitness_reference(0.02).model_copy(
        update={"dataset_name": "TmfKuzmin2018Dataset"}
    )
    assert dumps(references(ds)) == [reference.model_dump()]
    assert_side_files(ds, ["YAL048C", "YAR002W", "YML107C"], 1)


def test_tmf_no_ho_columns_concatenate_both_query_genes(tmp_path: Path) -> None:
    """Finding: the "no ho" columns of a trigenic row glue the two query genes together.

    ``preprocess_raw`` builds them by deleting ``hoΔ`` / ``YDL227C`` and then the
    ``+`` separator, which for a query with no ho half leaves ``YAR002WYML107C`` and
    ``nup60_deltapml39_delta``. ``create_experiment`` never reads them for Tmf, so
    the record is unaffected; ``data.csv`` shows the values.
    """
    ds = build(tmp_path, TmfKuzmin2018Dataset)
    row = ds.df.iloc[0]
    assert row["Query systematic name no ho"] == "YAR002WYML107C"
    assert row["Query allele name no ho"] == "nup60_deltapml39_delta"
    assert row["Query systematic name_1"] == "YAR002W"
    assert row["Query systematic name_2"] == "YML107C"


def test_dmi_records(tmp_path: Path) -> None:
    """Dmi: the 2 digenic rows as edge-level interactions, epsilon -0.05 / -0.02.

    P-values 0.2 and 0.5; ``graph_level`` is "edge" on both the phenotype and the
    reference (interaction 0.0, p-value None). Genotypes match Dmf's digenic pairs.
    """
    ds = build(tmp_path, DmiKuzmin2018Dataset)
    assert len(ds) == 2
    expected = [
        GeneInteractionExperiment(
            dataset_name="DmiKuzmin2018Dataset",
            genotype=Genotype(perturbations=[NUP60_DIGENIC, GEM1]),
            environment=ENVIRONMENT,
            phenotype=GeneInteractionPhenotype(
                gene_interaction=-0.05, gene_interaction_p_value=0.2, graph_level="edge"
            ),
        ),
        GeneInteractionExperiment(
            dataset_name="DmiKuzmin2018Dataset",
            genotype=Genotype(perturbations=[NUP60_DIGENIC, NTH2_TS]),
            environment=ENVIRONMENT,
            phenotype=GeneInteractionPhenotype(
                gene_interaction=-0.02, gene_interaction_p_value=0.5, graph_level="edge"
            ),
        ),
    ]
    assert dumps(experiments(ds)) == dumps(expected)
    reference = GeneInteractionExperimentReference(
        dataset_name="DmiKuzmin2018Dataset",
        genome_reference=GENOME,
        environment_reference=ENVIRONMENT,
        phenotype_reference=GeneInteractionPhenotype(
            gene_interaction=0.0, gene_interaction_p_value=None, graph_level="edge"
        ),
    )
    assert dumps(references(ds)) == [reference.model_dump()] * 2
    assert [ds[i]["publication"] for i in range(2)] == [PUBLICATION.model_dump()] * 2
    assert_side_files(ds, ["YAL048C", "YAR002W", "YBR001C"], 2)


def test_tmi_record(tmp_path: Path) -> None:
    """Tmi: the trigenic row as a hyperedge interaction, tau -0.1, p 0.01.

    ``graph_level`` stays at the class default "hyperedge" on phenotype and reference.
    """
    ds = build(tmp_path, TmiKuzmin2018Dataset)
    assert len(ds) == 1
    expected = GeneInteractionExperiment(
        dataset_name="TmiKuzmin2018Dataset",
        genotype=Genotype(perturbations=[NUP60_TRIGENIC, PML39_TRIGENIC, GEM1]),
        environment=ENVIRONMENT,
        phenotype=GeneInteractionPhenotype(
            gene_interaction=-0.1, gene_interaction_p_value=0.01
        ),
    )
    assert dumps(experiments(ds)) == [expected.model_dump()]
    assert experiments(ds)[0].phenotype.graph_level == "hyperedge"
    reference = GeneInteractionExperimentReference(
        dataset_name="TmiKuzmin2018Dataset",
        genome_reference=GENOME,
        environment_reference=ENVIRONMENT,
        phenotype_reference=GeneInteractionPhenotype(
            gene_interaction=0.0, gene_interaction_p_value=None
        ),
    )
    assert dumps(references(ds)) == [reference.model_dump()]
    assert_side_files(ds, ["YAL048C", "YAR002W", "YML107C"], 1)


@pytest.mark.parametrize(
    ("cls", "n"),
    [
        (SmfKuzmin2018Dataset, 3),
        (DmfKuzmin2018Dataset, 3),
        (TmfKuzmin2018Dataset, 1),
        (DmiKuzmin2018Dataset, 2),
        (TmiKuzmin2018Dataset, 1),
    ],
)
def test_reopen_reads_the_built_store_without_reprocessing(
    tmp_path: Path, cls: type[Any], n: int
) -> None:
    """A second instance on the same root reads the store: same length, same records.

    ``processed/lmdb`` exists, so PyG skips ``process()``; deleting the raw file
    first proves neither download nor process runs (``_download`` returns early when
    the LMDB exists).
    """
    first = build(tmp_path, cls)
    records = [first[i] for i in range(n)]
    (Path(first.root) / "raw" / RAW_NAME).unlink()
    second = cls(root=first.root)
    assert len(second) == n
    assert [second[i] for i in range(n)] == records
