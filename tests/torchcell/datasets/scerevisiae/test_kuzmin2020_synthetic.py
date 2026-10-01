# tests/torchcell/datasets/scerevisiae/test_kuzmin2020_synthetic.py
# [[tests.torchcell.datasets.scerevisiae.test_kuzmin2020_synthetic]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_kuzmin2020_synthetic.py
"""Hermetic end-to-end builds of the five Kuzmin 2020 loaders from hand-written xlsx tables.

Tables S1, S3 and S5 are written with pandas/openpyxl under ``<tmp_path>/<root>/raw/``
with a title row first (the loaders read with ``skiprows=1``), so PyG never calls
``download()``; ``process()`` runs once and ``@post_process`` writes the gene set, the
reference index and the build manifest. No genome and no ``$DATA_ROOT`` are involved.

S1 and S3 share the eleven S1/S3 columns; each holds one digenic and one trigenic row:

=====  ==========================  ================  ====  ========  ====  ========  =======  ======
table  query strain / alleles      array strain      type  combined  sd    query     epsilon  p
=====  ==========================  ================  ====  ========  ====  ========  =======  ======
S1     YAL015C+YDL227C_tm461       YBL007C_dma91     dig   0.9695    0.0465 0.98     0.01     0.6
       ntg1Δ+hoΔ                   sla1Δ
S1     YAL015C+YOL043C_tm72        YBL007C_dma91     tri   0.85      0.05  1.0133    -0.08    0.02
       ntg1Δ+ntg2Δ                 sla1Δ
S3     YAL015C+YDL227C_tm461       YBR001C_tsa100    dig   0.7       0.03  0.98      -0.02    0.5
       ntg1Δ+hoΔ                   nth2-5001
S3     YAL015C+YBR001C_tm99        YCR002C_tsa1      tri   0.6       0.01  0.77      -0.15    0.001
       ntg1Δ+nth2-5001             cdc10-1
=====  ==========================  ================  ====  ========  ====  ========  =======  ======

S5 (``Mutant type, Allele1, ORF1, Gene1, Query Strain ID, Fitness, St.dev.``) has two
single mutants (NTG1 ``delta`` sn123 0.98/0.01; NTH2 ``nth2-5001`` sn124 0.8/0.02), a
single mutant with no fitness (CDC10 sn125, dropped), and the double mutant ``tm72``
(1.0133/0.008) that the Dmf join finds on the bare tm number. ``tm99`` has no S5 row,
so its query-strain record falls back to the S1/S3 column (0.77, no SD).
``YCR002C_tsa1`` is a ts array (an array strain that is neither ``dma`` nor ``tsa`` is
refused, pinned in ``test_kuzmin2020.py``; it used to be stored as an
``SgaAllelePerturbation``). Perturbed gene names are the allele name
before the first ``_`` (``ntg1`` from ``ntg1_delta``), except Smf, which stores
``Gene1`` verbatim. PubMed 32586993 / DOI 10.1126/science.aaz5667 on every record.
"""

import json
from pathlib import Path
from typing import Any

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
    SgaAllelePerturbation,
    SgaKanMxDeletionPerturbation,
    SgaTsAllelePerturbation,
    Temperature,
    UncertaintyType,
)
from torchcell.datasets.scerevisiae.kuzmin2020 import (
    N_SAMPLES_COMBINED_MUTANT,
    N_SAMPLES_QUERY_STRAIN_FITNESS,
    RECORD_KIND_DIGENIC_ARRAY_CROSS,
    RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN,
    DmfKuzmin2020Dataset,
    DmiKuzmin2020Dataset,
    SmfKuzmin2020Dataset,
    TmfKuzmin2020Dataset,
    TmiKuzmin2020Dataset,
)

S1_NAME = "aaz5667-Table-S1.xlsx"
S3_NAME = "aaz5667-Table-S3.xlsx"
S5_NAME = "aaz5667-Table-S5.xlsx"
S13_COLUMNS = [
    "Query strain ID",
    "Query allele name",
    "Array strain ID",
    "Array allele name",
    "Combined mutant type",
    "Double/triple mutant fitness",
    "Double/triple mutant fitness standard deviation",
    "Query single/double mutant fitness",
    "Array single mutant fitness",
    "Adjusted genetic interaction score (epsilon or tau)",
    "P-value",
]
S5_COLUMNS = ["Mutant type", "Allele1", "ORF1", "Gene1", "Query Strain ID", "Fitness", "St.dev."]  # fmt: skip
DIGENIC_QUERY = "YAL015C+YDL227C_tm461"
TM72_QUERY = "YAL015C+YOL043C_tm72"
TM99_QUERY = "YAL015C+YBR001C_tm99"
ARRAY_DMA = "YBL007C_dma91"
ARRAY_TSA = "YBR001C_tsa100"
ARRAY_TS_CDC10 = "YCR002C_tsa1"
S1_ROWS: list[list[Any]] = [
    [DIGENIC_QUERY, "ntg1Δ+hoΔ", ARRAY_DMA, "sla1Δ", "digenic", 0.9695, 0.0465, 0.98, 0.9, 0.01, 0.6],
    [TM72_QUERY, "ntg1Δ+ntg2Δ", ARRAY_DMA, "sla1Δ", "trigenic", 0.85, 0.05, 1.0133, 0.9, -0.08, 0.02],
]  # fmt: skip
S3_ROWS: list[list[Any]] = [
    [DIGENIC_QUERY, "ntg1Δ+hoΔ", ARRAY_TSA, "nth2-5001", "digenic", 0.7, 0.03, 0.98, 0.8, -0.02, 0.5],
    [TM99_QUERY, "ntg1Δ+nth2-5001", ARRAY_TS_CDC10, "cdc10-1", "trigenic", 0.6, 0.01, 0.77, 0.85, -0.15, 0.001],
]  # fmt: skip
S5_SINGLE_ROWS: list[list[Any]] = [
    ["Single mutant", "delta", "YAL015C", "NTG1", "sn123", 0.98, 0.01],
    ["Single mutant", "nth2-5001", "YBR001C", "NTH2", "sn124", 0.8, 0.02],
    ["Single mutant", "delta", "YCR002C", "CDC10", "sn125", None, None],
]  # fmt: skip
S5_DOUBLE_ROW: list[Any] = ["Double mutant", "delta", "YAL015C", "NTG1", "tm72", 1.0133, 0.008]  # fmt: skip
S5_ROWS = [*S5_SINGLE_ROWS, S5_DOUBLE_ROW]

ENVIRONMENT = Environment(media=SGA_TM_SELECTION, temperature=Temperature(value=26))
GENOME = ReferenceGenome(species="Saccharomyces cerevisiae", strain="S288C")
PUBLICATION = Publication(
    pubmed_id="32586993",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/32586993/",
    doi="10.1126/science.aaz5667",
    doi_url="https://www.science.org/doi/10.1126/science.aaz5667",
)
NTG1_DIGENIC = SgaKanMxDeletionPerturbation(
    systematic_gene_name="YAL015C", perturbed_gene_name="ntg1", strain_id=DIGENIC_QUERY
)  # fmt: skip
NTG1_TM72 = SgaKanMxDeletionPerturbation(
    systematic_gene_name="YAL015C", perturbed_gene_name="ntg1", strain_id=TM72_QUERY
)  # fmt: skip
NTG2_TM72 = SgaKanMxDeletionPerturbation(
    systematic_gene_name="YOL043C", perturbed_gene_name="ntg2", strain_id=TM72_QUERY
)  # fmt: skip
NTG1_TM99 = SgaKanMxDeletionPerturbation(
    systematic_gene_name="YAL015C", perturbed_gene_name="ntg1", strain_id=TM99_QUERY
)  # fmt: skip
NTH2_TM99 = SgaAllelePerturbation(
    systematic_gene_name="YBR001C", perturbed_gene_name="nth2-5001", strain_id=TM99_QUERY
)  # fmt: skip
SLA1 = SgaKanMxDeletionPerturbation(
    systematic_gene_name="YBL007C", perturbed_gene_name="sla1", strain_id=ARRAY_DMA
)  # fmt: skip
NTH2_TS = SgaTsAllelePerturbation(
    systematic_gene_name="YBR001C", perturbed_gene_name="nth2-5001", strain_id=ARRAY_TSA
)  # fmt: skip
CDC10_TS = SgaTsAllelePerturbation(
    systematic_gene_name="YCR002C", perturbed_gene_name="cdc10-1", strain_id=ARRAY_TS_CDC10
)  # fmt: skip
# The Tmf and Tmi loaders tag query perturbations with the SPLIT halves of the query
# strain id ("YAL015C" and "YOL043C_tm72"), not the full id (see test_tmf_records).
NTG1_SPLIT = SgaKanMxDeletionPerturbation(
    systematic_gene_name="YAL015C", perturbed_gene_name="ntg1", strain_id="YAL015C"
)  # fmt: skip
NTG2_SPLIT_TM72 = SgaKanMxDeletionPerturbation(
    systematic_gene_name="YOL043C", perturbed_gene_name="ntg2", strain_id="YOL043C_tm72"
)  # fmt: skip
NTH2_SPLIT_TM99 = SgaAllelePerturbation(
    systematic_gene_name="YBR001C", perturbed_gene_name="nth2-5001", strain_id="YBR001C_tm99"
)  # fmt: skip
TM72_TRIGENIC = Genotype(perturbations=[NTG1_SPLIT, NTG2_SPLIT_TM72, SLA1])
TM99_TRIGENIC = Genotype(perturbations=[NTG1_SPLIT, NTH2_SPLIT_TM99, CDC10_TS])


def write_xlsx(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    """Write ``rows`` as a sheet whose header sits on the SECOND row (title row above)."""
    pd.DataFrame(rows, columns=columns).to_excel(path, index=False, startrow=1)


def build(tmp_path: Path, cls: type[Any], s5_rows: list[list[Any]] = S5_ROWS) -> Any:
    """Build ``cls`` under ``tmp_path/<class name>`` from the three tables."""
    root = tmp_path / cls.__name__
    (root / "raw").mkdir(parents=True)
    write_xlsx(root / "raw" / S1_NAME, S13_COLUMNS, S1_ROWS)
    write_xlsx(root / "raw" / S3_NAME, S13_COLUMNS, S3_ROWS)
    write_xlsx(root / "raw" / S5_NAME, S5_COLUMNS, s5_rows)
    return cls(root=str(root))


def sample_sd(sd: float) -> dict[str, Any]:
    """The uncertainty fields a combined-mutant SD carries (n = 4 colonies)."""
    return {
        "fitness_uncertainty": sd,
        "fitness_uncertainty_type": UncertaintyType.sample_sd,
        "n_samples": N_SAMPLES_COMBINED_MUTANT,
        "sample_unit": SampleUnit.colony,
    }


def bootstrap_se(sd: float) -> dict[str, Any]:
    """The uncertainty fields a Table S5 query-strain SD carries (bootstrap, n = 12)."""
    return {
        "fitness_uncertainty": sd,
        "fitness_uncertainty_type": UncertaintyType.bootstrap_se,
        "n_samples": N_SAMPLES_QUERY_STRAIN_FITNESS,
        "sample_unit": SampleUnit.colony,
    }


def fitness_reference(
    name: str, reference_sd: float | None
) -> FitnessExperimentReference:
    """The unlabeled fitness reference of a 2020 fitness loader (fitness 1.0)."""
    return FitnessExperimentReference(
        dataset_name=name,
        genome_reference=GENOME,
        environment_reference=ENVIRONMENT,
        phenotype_reference=FitnessPhenotype(fitness=1.0, fitness_std=reference_sd),
    )


def interaction_reference(
    name: str, graph_level: str
) -> GeneInteractionExperimentReference:
    """The zero-interaction reference of a 2020 interaction loader."""
    return GeneInteractionExperimentReference(
        dataset_name=name,
        genome_reference=GENOME,
        environment_reference=ENVIRONMENT,
        phenotype_reference=GeneInteractionPhenotype(
            gene_interaction=0.0, gene_interaction_p_value=None, graph_level=graph_level
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
    """Pin the post-process artifacts of a build with ``n`` records.

    ``gene_set.json`` is the sorted systematic names; the reference index has ONE
    entry whose ``member_indices`` are ``0..n-1`` and whose reference equals the
    reference stored on record 0; the build manifest names the root, class and
    module; ``processed/interned`` exists beside ``processed/lmdb``; every record's
    publication is the Kuzmin 2020 paper.
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
    assert manifest["loader_module"] == "torchcell.datasets.scerevisiae.kuzmin2020"
    assert (root / "processed" / "lmdb").is_dir()
    assert (root / "processed" / "interned").is_dir()
    assert [ds[i]["publication"] for i in range(n)] == [PUBLICATION.model_dump()] * n


def test_smf_records(tmp_path: Path) -> None:
    """Smf: the two S5 single mutants with a fitness; the fitness-less one is dropped.

    ``Allele1 == "delta"`` gives a KanMX deletion, anything else an SGA allele; the
    perturbed gene name is ``Gene1`` verbatim (upper case, unlike the other four
    loaders) and the strain id the S5 "Query Strain ID". The reference is fitness
    1.0 with no SD.

    Finding: the S5 "St.dev." is stored as ``sample_sd`` over 4 colonies (se = sd / 2:
    0.005 and 0.01) through ``_combined_mutant_uncertainty``, although the module's
    own ``N_SAMPLES_QUERY_STRAIN_FITNESS`` comment quotes the SI calling the query-
    strain fitness SD a bootstrap quantity over 12-24 colonies, and the Dmf loader
    labels the very same S5 column ``bootstrap_se`` for its double-mutant rows.
    """
    ds = build(tmp_path, SmfKuzmin2020Dataset)
    assert len(ds) == 2
    expected = [
        FitnessExperiment(
            dataset_name="SmfKuzmin2020Dataset",
            genotype=Genotype(
                perturbations=[
                    SgaKanMxDeletionPerturbation(
                        systematic_gene_name="YAL015C",
                        perturbed_gene_name="NTG1",
                        strain_id="sn123",
                    )
                ]
            ),
            environment=ENVIRONMENT,
            phenotype=FitnessPhenotype(
                fitness=0.98, fitness_std=0.01, **sample_sd(0.01)
            ),
        ),
        FitnessExperiment(
            dataset_name="SmfKuzmin2020Dataset",
            genotype=Genotype(
                perturbations=[
                    SgaAllelePerturbation(
                        systematic_gene_name="YBR001C",
                        perturbed_gene_name="NTH2",
                        strain_id="sn124",
                    )
                ]
            ),
            environment=ENVIRONMENT,
            phenotype=FitnessPhenotype(
                fitness=0.8, fitness_std=0.02, **sample_sd(0.02)
            ),
        ),
    ]
    assert dumps(experiments(ds)) == dumps(expected)
    assert [e.phenotype.fitness_se for e in experiments(ds)] == [0.005, 0.01]
    reference = fitness_reference("SmfKuzmin2020Dataset", None)
    assert dumps(references(ds)) == [reference.model_dump()] * 2
    assert_side_files(ds, ["YAL015C", "YBR001C"], 2)


def test_dmf_records(tmp_path: Path) -> None:
    """Dmf: 2 digenic crosses (S1 then S3) then 2 query strains (tm72 then tm99).

    Digenic: ntg1 (hoΔ stripped, full strain id) x sla1 at 0.9695 / 0.0465 (se
    0.02325) and x nth2-5001 ts at 0.7 / 0.03 (se 0.015), both ``sample_sd`` n = 4.
    tm72 is the query PAIR ntg1 + ntg2 with Table S5's 1.0133 / 0.008 labeled
    ``bootstrap_se`` n = 12, so ``fitness_se`` is 0.008 undivided. tm99 has no S5
    row: ntg1 (KanMX) + nth2-5001 (allele) at the S1/S3 column value 0.77 with no SD.

    The fallback record's ``fitness_std`` is None, not the float NaN that
    ``where(Fitness.notna())`` leaves in the frame: ``create_experiment`` stores a blank
    SD as "no value reported", matching its empty uncertainty fields (issue #533).
    """
    ds = build(tmp_path, DmfKuzmin2020Dataset)
    assert len(ds) == 4
    stored = dumps(experiments(ds))
    assert stored[3]["phenotype"]["fitness_std"] is None
    expected = [
        FitnessExperiment(
            dataset_name="DmfKuzmin2020Dataset",
            genotype=Genotype(perturbations=[NTG1_DIGENIC, SLA1]),
            environment=ENVIRONMENT,
            phenotype=FitnessPhenotype(
                fitness=0.9695, fitness_std=0.0465, **sample_sd(0.0465)
            ),
        ),
        FitnessExperiment(
            dataset_name="DmfKuzmin2020Dataset",
            genotype=Genotype(perturbations=[NTG1_DIGENIC, NTH2_TS]),
            environment=ENVIRONMENT,
            phenotype=FitnessPhenotype(
                fitness=0.7, fitness_std=0.03, **sample_sd(0.03)
            ),
        ),
        FitnessExperiment(
            dataset_name="DmfKuzmin2020Dataset",
            genotype=Genotype(perturbations=[NTG1_TM72, NTG2_TM72]),
            environment=ENVIRONMENT,
            phenotype=FitnessPhenotype(
                fitness=1.0133, fitness_std=0.008, **bootstrap_se(0.008)
            ),
        ),
        FitnessExperiment(
            dataset_name="DmfKuzmin2020Dataset",
            genotype=Genotype(perturbations=[NTG1_TM99, NTH2_TM99]),
            environment=ENVIRONMENT,
            phenotype=FitnessPhenotype(fitness=0.77, fitness_std=None),
        ),
    ]
    assert stored == dumps(expected)
    assert [e.phenotype.fitness_se for e in experiments(ds)] == [
        0.02325,
        0.015,
        0.008,
        None,
    ]
    assert ds.df["record_kind"].tolist() == [
        RECORD_KIND_DIGENIC_ARRAY_CROSS,
        RECORD_KIND_DIGENIC_ARRAY_CROSS,
        RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN,
        RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN,
    ]
    assert ds.df["Query strain ID"].tolist()[2:] == [TM72_QUERY, TM99_QUERY]


def test_dmf_reference_carries_no_sd(tmp_path: Path) -> None:
    """Finding: Dmf's reference is fitness 1.0 with NO SD, although a mean is computed.

    ``preprocess_raw`` sets ``self.phenotype_reference_std`` to the digenic mean
    ``(0.0465 + 0.03) / 2 = 0.03825``, but ``create_experiment`` never receives it
    and builds ``FitnessPhenotype(fitness=1.0, fitness_std=None)``. The computed
    attribute is pinned alongside the stored reference.
    """
    ds = build(tmp_path, DmfKuzmin2020Dataset)
    assert ds.phenotype_reference_std == 0.03825
    reference = fitness_reference("DmfKuzmin2020Dataset", None)
    assert dumps(references(ds)) == [reference.model_dump()] * 4
    assert_side_files(ds, ["YAL015C", "YBL007C", "YBR001C", "YOL043C"], 4)


def test_dmf_raises_when_table_s5_matches_no_query_strain(tmp_path: Path) -> None:
    """Without a matching "Double mutant" S5 row the build raises ValueError.

    S5 reduced to its single mutants: the tm-number join matches none of the two
    trigenic query strains and ``_double_mutant_query_strain_rows`` refuses to build.
    """
    with pytest.raises(ValueError, match="matched no trigenic query strain"):
        build(tmp_path, DmfKuzmin2020Dataset, s5_rows=S5_SINGLE_ROWS)


def test_tmf_records(tmp_path: Path) -> None:
    """Tmf: the 2 trigenic rows (S1 tm72, S3 tm99) with three perturbations each.

    tm72: ntg1, ntg2 and sla1 at 0.85 / SD 0.05; tm99: ntg1 (KanMX), nth2-5001
    (allele) and cdc10-1 (the ts array, an SGA ts allele) at 0.6 / 0.01.

    Finding: the query perturbations carry the SPLIT halves of the query strain id
    (``YAL015C`` and ``YOL043C_tm72``), not the full id the Dmf query-strain records
    and every 2018 loader use. Finding: ``fitness_std`` is stored with no
    ``fitness_uncertainty`` labeling, so ``fitness_se`` is None on every record.
    """
    ds = build(tmp_path, TmfKuzmin2020Dataset)
    assert len(ds) == 2
    expected = [
        FitnessExperiment(
            dataset_name="TmfKuzmin2020Dataset",
            genotype=TM72_TRIGENIC,
            environment=ENVIRONMENT,
            phenotype=FitnessPhenotype(fitness=0.85, fitness_std=0.05),
        ),
        FitnessExperiment(
            dataset_name="TmfKuzmin2020Dataset",
            genotype=TM99_TRIGENIC,
            environment=ENVIRONMENT,
            phenotype=FitnessPhenotype(fitness=0.6, fitness_std=0.01),
        ),
    ]
    assert dumps(experiments(ds)) == dumps(expected)
    assert [e.phenotype.fitness_se for e in experiments(ds)] == [None, None]


def test_tmf_reference_is_the_trigenic_mean_sd_unlabeled(tmp_path: Path) -> None:
    """Tmf reference: fitness 1.0, ``fitness_std`` = trigenic mean ``(0.05 + 0.01) / 2``.

    The mean is passed as ``fitness_std`` only, with no uncertainty type, so the
    reference ``fitness_se`` is None too. Gene set has the five systematic names.
    """
    ds = build(tmp_path, TmfKuzmin2020Dataset)
    reference = fitness_reference("TmfKuzmin2020Dataset", (0.05 + 0.01) / 2)
    assert dumps(references(ds)) == [reference.model_dump()] * 2
    assert references(ds)[0].phenotype_reference.fitness_se is None
    assert_side_files(ds, ["YAL015C", "YBL007C", "YBR001C", "YCR002C", "YOL043C"], 2)


def test_dmi_records(tmp_path: Path) -> None:
    """Dmi: the 2 digenic rows as edge-level interactions, epsilon 0.01 / -0.02.

    P-values 0.6 and 0.5; ``graph_level`` "edge" on phenotype and reference.
    """
    ds = build(tmp_path, DmiKuzmin2020Dataset)
    assert len(ds) == 2
    expected = [
        GeneInteractionExperiment(
            dataset_name="DmiKuzmin2020Dataset",
            genotype=Genotype(perturbations=[NTG1_DIGENIC, SLA1]),
            environment=ENVIRONMENT,
            phenotype=GeneInteractionPhenotype(
                gene_interaction=0.01, gene_interaction_p_value=0.6, graph_level="edge"
            ),
        ),
        GeneInteractionExperiment(
            dataset_name="DmiKuzmin2020Dataset",
            genotype=Genotype(perturbations=[NTG1_DIGENIC, NTH2_TS]),
            environment=ENVIRONMENT,
            phenotype=GeneInteractionPhenotype(
                gene_interaction=-0.02, gene_interaction_p_value=0.5, graph_level="edge"
            ),
        ),
    ]
    assert dumps(experiments(ds)) == dumps(expected)
    reference = interaction_reference("DmiKuzmin2020Dataset", "edge")
    assert dumps(references(ds)) == [reference.model_dump()] * 2
    assert_side_files(ds, ["YAL015C", "YBL007C", "YBR001C"], 2)


def test_tmi_records(tmp_path: Path) -> None:
    """Tmi: the 2 trigenic rows as hyperedge interactions, tau -0.08 / -0.15.

    P-values 0.02 and 0.001; the genotypes are the same hand-built ``TM72_TRIGENIC`` /
    ``TM99_TRIGENIC`` Tmf stores (split query strain ids, cdc10-1 as an SGA ts
    allele); ``graph_level`` stays "hyperedge".
    """
    ds = build(tmp_path, TmiKuzmin2020Dataset)
    assert len(ds) == 2
    expected = [
        GeneInteractionExperiment(
            dataset_name="TmiKuzmin2020Dataset",
            genotype=TM72_TRIGENIC,
            environment=ENVIRONMENT,
            phenotype=GeneInteractionPhenotype(
                gene_interaction=-0.08, gene_interaction_p_value=0.02
            ),
        ),
        GeneInteractionExperiment(
            dataset_name="TmiKuzmin2020Dataset",
            genotype=TM99_TRIGENIC,
            environment=ENVIRONMENT,
            phenotype=GeneInteractionPhenotype(
                gene_interaction=-0.15, gene_interaction_p_value=0.001
            ),
        ),
    ]
    assert dumps(experiments(ds)) == dumps(expected)
    assert [e.phenotype.graph_level for e in experiments(ds)] == ["hyperedge"] * 2
    reference = interaction_reference("TmiKuzmin2020Dataset", "hyperedge")
    assert dumps(references(ds)) == [reference.model_dump()] * 2
    assert_side_files(ds, ["YAL015C", "YBL007C", "YBR001C", "YCR002C", "YOL043C"], 2)


@pytest.mark.parametrize(
    ("cls", "n"),
    [
        (SmfKuzmin2020Dataset, 2),
        (DmfKuzmin2020Dataset, 4),
        (TmfKuzmin2020Dataset, 2),
        (DmiKuzmin2020Dataset, 2),
        (TmiKuzmin2020Dataset, 2),
    ],
)
def test_reopen_reads_the_built_store_without_reprocessing(
    tmp_path: Path, cls: type[Any], n: int
) -> None:
    """A second instance on the same root reads the store: same length, same records.

    ``processed/lmdb`` exists, so PyG skips ``process()``; deleting the raw tables
    first proves neither download nor process runs (``_download`` returns early when
    the LMDB exists). Records are compared through sorted JSON.
    """
    first = build(tmp_path, cls)
    records = [json.dumps(first[i], sort_keys=True) for i in range(n)]
    # py-lmdb refuses to open one path twice in a process; release the first handle.
    first.close_lmdb()
    for name in (S1_NAME, S3_NAME, S5_NAME):
        (Path(first.root) / "raw" / name).unlink()
    second = cls(root=first.root)
    assert len(second) == n
    assert [json.dumps(second[i], sort_keys=True) for i in range(n)] == records
    second.close_lmdb()
