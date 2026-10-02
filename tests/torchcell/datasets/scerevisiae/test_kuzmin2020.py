# tests/torchcell/datasets/scerevisiae/test_kuzmin2020
# [[tests.torchcell.datasets.scerevisiae.test_kuzmin2020]]
"""Tests for the five Kuzmin 2020 loaders (Smf, Dmf, Tmf, Dmi, Tmi).

The data-gated tests at the bottom (``--data --slow``) call ``preprocess_raw`` /
``create_experiment`` directly on the raw supplementary tables, so nothing is built into
LMDB. They cover both Dmf record kinds: the digenic query x array crosses, which must
stay byte-identical to what is already served, and the double-mutant QUERY strains of
the trigenic screens, whose fitness and bootstrap SD come from Table S5 through a join on
the tm number. They skip when the ``$DATA_ROOT`` raw mirror is absent (CI).

2026.09.30 (Phase 15): hermetic builds added (the import-time ``load_dotenv()`` and the
module-level skip removed, the data gate moved onto the four mirror tests). The
deletion-only paths are pinned in ``test_kuzmin2020_synthetic.py``; this file pins the
ALLELE-query, unknown-array, blank-SD, S5-disagreement, subset, download and ``main``
paths on xlsx tables under ``tmp_path`` (title row first, the loaders read
``skiprows=1``):

=====  ======================  ===============  ===============  ====  ====  =====  =====  =====  =====
table  query strain            query alleles    array strain     type  comb  sd     query  eps    p
=====  ======================  ===============  ===============  ====  ====  =====  =====  =====  =====
S1     YBR160W+YDL227C_tsq508  cdc28-4+hoΔ      YCR002C_tsa1     dig   0.61  blank  0.83   -0.12  0.03
S1     YBR160W+YML107C_tm801   cdc28-4+pml39Δ   YBR001C_tsa100   tri   0.3   0.02   0.55   -0.2   0.001
S3     YBR160W+YDL227C_tsq508  cdc28-4+hoΔ      YAL048C_dma5203  dig   0.72  0.05   0.83   0.04   0.4
S3     YBR160W+YML107C_tm801   cdc28-4+pml39Δ   YAL048C_dma5203  tri   0.25  0.03   0.55   -0.1   0.01
=====  ======================  ===============  ===============  ====  ====  =====  =====  =====  =====

Table S5: single mutants CDC28 (``cdc28-4``, sn tsq508, 0.83, St.dev. blank) and GEM1
(``delta``, sn200, 0.95 / 0.01), and the double mutant tm801 (0.5 / 0.006). Expected:

- Smf: CDC28 as an ``SgaAllelePerturbation`` (Gene1 verbatim, strain ``tsq508``) and GEM1
  as a KanMX deletion (0.95 / 0.01, labeled ``sample_sd`` n 4, se 0.005); the reference
  is fitness 1.0 with no SD.
- Dmf: the S1 then S3 digenic crosses, the allele query paired with the ts array
  ``YCR002C_tsa1`` at 0.61 with a blank SD (stored as ``fitness_std`` None, no
  uncertainty), and with gem1 at 0.72 / 0.05 (se 0.025); then ONE record for tm801
  although it appears in both S1 and S3 (drop_duplicates on the strain id), with S5's
  0.5 / 0.006 labeled ``bootstrap_se`` over ``N_SAMPLES_QUERY_STRAIN_FITNESS = 12``
  colonies (the SI's 12 to 24 colony measurements, lower end), se 0.006 undivided.
  S5's 0.5 and the S1/S3 column's 0.55 differ by 0.05 > 1e-3, so the disagreement
  warning fires with "(max |diff| 0.0500)".
- Tmf: 0.3 / 0.02 and 0.25 / 0.03, allele in query slot 1, the ts array in the first;
  no uncertainty labels; reference SD (0.02 + 0.03) / 2 = 0.025. Dmi: -0.12 (p 0.03) and
  0.04 (p 0.4) at edge level. Tmi: -0.2 (p 0.001) and -0.1 (p 0.01) at hyperedge level.
- ``subset_n=1`` samples position 1 of two with ``random_state=42``.

2026.10.01 (issue #533): the Phase 15 findings are retired. A blank SD is stored as
``fitness_std`` None (not a float NaN) on Smf, Dmf and Tmf; an array strain that is
neither ``tsa`` nor ``dma`` (``YCR002C_sn1``) is refused by every loader that reads the
array column with a named ``ValueError``, where Dmf, Dmi, Tmf and Tmi used to store it as
an ``SgaAllelePerturbation``; a Table S5 "Double mutant" row listed twice is refused
instead of fanning the left merge out into two records; ``main`` builds all five loaders
under ``$DATA_ROOT/data/torchcell``. On the released tables each refused input occurs 0
times (0 blank SDs among stored rows, 0 other array strains in 934,595 S1 and S3 rows,
0 repeated tm numbers among 240 S5 "Double mutant" rows), so no stored record changes.
"""

import hashlib
import json
import os
import os.path as osp
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

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
from torchcell.datasets.scerevisiae import kuzmin2018 as k2018
from torchcell.datasets.scerevisiae import kuzmin2020 as k

S1_NAME = "aaz5667-Table-S1.xlsx"
S3_NAME = "aaz5667-Table-S3.xlsx"
S5_NAME = "aaz5667-Table-S5.xlsx"
URL = "https://uofi.box.com/shared/static/464ogx5kpafav7i3zv7gesb9lcn0dm94.zip"
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
TSQ = "YBR160W+YDL227C_tsq508"
MAIN = "kuzmin2020_s1_diagnostic_array"
PILOT = "kuzmin2020_s3_pilot_genome_wide_arrays"
S5_SCREEN = "kuzmin2020_s5_query_fitness_array"
TM801 = "YBR160W+YML107C_tm801"
TSA = "YBR001C_tsa100"
DMA = "YAL048C_dma5203"
TSA_CDC10 = "YCR002C_tsa1"
SN = "YCR002C_sn1"
S1_ROWS: list[list[Any]] = [
    [TSQ, "cdc28-4+hoΔ", TSA_CDC10, "cdc10-1", "digenic", 0.61, None, 0.83, 0.7, -0.12, 0.03],
    [TM801, "cdc28-4+pml39Δ", TSA, "nth2-5001", "trigenic", 0.3, 0.02, 0.55, 0.8, -0.2, 0.001],
]  # fmt: skip
S3_ROWS: list[list[Any]] = [
    [TSQ, "cdc28-4+hoΔ", DMA, "gem1Δ", "digenic", 0.72, 0.05, 0.83, 0.95, 0.04, 0.4],
    [TM801, "cdc28-4+pml39Δ", DMA, "gem1Δ", "trigenic", 0.25, 0.03, 0.55, 0.95, -0.1, 0.01],
]  # fmt: skip
S5_DOUBLE: list[Any] = [
    "Double mutant",
    "delta",
    "YBR160W",
    "CDC28",
    "tm801",
    0.5,
    0.006,
]
S5_ROWS: list[list[Any]] = [
    ["Single mutant", "cdc28-4", "YBR160W", "CDC28", "tsq508", 0.83, None],
    ["Single mutant", "delta", "YAL048C", "GEM1", "sn200", 0.95, 0.01],
    S5_DOUBLE,
]

ENVIRONMENT = Environment(media=SGA_TM_SELECTION, temperature=Temperature(value=26))
GENOME = ReferenceGenome(species="Saccharomyces cerevisiae", strain="S288C")
PUBLICATION = Publication(
    pubmed_id="32586993",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/32586993/",
    doi="10.1126/science.aaz5667",
    doi_url="https://www.science.org/doi/10.1126/science.aaz5667",
)
CDC10_TS = SgaTsAllelePerturbation(
    systematic_gene_name="YCR002C", perturbed_gene_name="cdc10-1", strain_id=TSA_CDC10
)
GEM1 = SgaKanMxDeletionPerturbation(
    systematic_gene_name="YAL048C", perturbed_gene_name="gem1", strain_id=DMA
)
NTH2_TS = SgaTsAllelePerturbation(
    systematic_gene_name="YBR001C", perturbed_gene_name="nth2-5001", strain_id=TSA
)
PML39_TM801 = SgaKanMxDeletionPerturbation(
    systematic_gene_name="YML107C", perturbed_gene_name="pml39", strain_id=TM801
)
# Tmf and Tmi tag the query genes with the SPLIT halves of the strain id.
CDC28_SPLIT = SgaAllelePerturbation(
    systematic_gene_name="YBR160W", perturbed_gene_name="cdc28-4", strain_id="YBR160W"
)
PML39_SPLIT = SgaKanMxDeletionPerturbation(
    systematic_gene_name="YML107C",
    perturbed_gene_name="pml39",
    strain_id="YML107C_tm801",
)


def cdc28(strain: str) -> SgaAllelePerturbation:
    """The cdc28-4 allele tagged with ``strain``."""
    return SgaAllelePerturbation(
        systematic_gene_name="YBR160W", perturbed_gene_name="cdc28-4", strain_id=strain
    )


def write_xlsx(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    """Write ``rows`` as a sheet whose header sits on the SECOND row (title row above)."""
    pd.DataFrame(rows, columns=columns).to_excel(path, index=False, startrow=1)


def write_tables(
    folder: Path,
    s5_rows: list[list[Any]] = S5_ROWS,
    s1_rows: list[list[Any]] = S1_ROWS,
    s3_rows: list[list[Any]] = S3_ROWS,
) -> None:
    """Write Tables S1, S3 and S5 into ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    write_xlsx(folder / S1_NAME, S13_COLUMNS, s1_rows)
    write_xlsx(folder / S3_NAME, S13_COLUMNS, s3_rows)
    write_xlsx(folder / S5_NAME, S5_COLUMNS, s5_rows)


def build(
    tmp_path: Path,
    cls: type[Any],
    s5_rows: list[list[Any]] = S5_ROWS,
    s1_rows: list[list[Any]] = S1_ROWS,
    s3_rows: list[list[Any]] = S3_ROWS,
    **kw: Any,
) -> Any:
    """Build ``cls`` under ``tmp_path/<class name>`` from the three tables."""
    root = tmp_path / cls.__name__
    write_tables(root / "raw", s5_rows, s1_rows, s3_rows)
    return cls(root=str(root), **kw)


def fitness(
    name: str, perturbations: list[Any], value: float, sd: float | None, **extra: Any
) -> dict[str, Any]:
    """A Kuzmin 2020 fitness experiment dump."""
    return FitnessExperiment(
        dataset_name=name,
        genotype=Genotype(perturbations=perturbations),
        environment=ENVIRONMENT,
        phenotype=FitnessPhenotype(fitness=value, fitness_std=sd, **extra),
    ).model_dump()


def labeled(sd: float, kind: UncertaintyType, n: int) -> dict[str, Any]:
    """The uncertainty fields for an SD of type ``kind`` over ``n`` colonies."""
    return {
        "fitness_uncertainty": sd,
        "fitness_uncertainty_type": kind,
        "n_samples": n,
        "sample_unit": SampleUnit.colony,
    }


def fitness_reference(name: str, sd: float | None) -> dict[str, Any]:
    """The unlabeled fitness reference of a 2020 fitness loader."""
    return FitnessExperimentReference(
        dataset_name=name,
        genome_reference=GENOME,
        environment_reference=ENVIRONMENT,
        phenotype_reference=FitnessPhenotype(fitness=1.0, fitness_std=sd),
    ).model_dump()


def interaction(
    name: str,
    perturbations: list[Any],
    value: float,
    p: float,
    level: str,
    screen_id: str,
) -> dict[str, Any]:
    """A Kuzmin 2020 interaction experiment dump at graph level ``level``."""
    return GeneInteractionExperiment(
        dataset_name=name,
        genotype=Genotype(perturbations=perturbations),
        environment=ENVIRONMENT,
        phenotype=GeneInteractionPhenotype(
            gene_interaction=value,
            gene_interaction_p_value=p,
            graph_level=level,
            screen_id=screen_id,
        ),
    ).model_dump()


def interaction_reference(name: str, level: str) -> dict[str, Any]:
    """The zero-interaction reference of a 2020 interaction loader."""
    return GeneInteractionExperimentReference(
        dataset_name=name,
        genome_reference=GENOME,
        environment_reference=ENVIRONMENT,
        phenotype_reference=GeneInteractionPhenotype(
            gene_interaction=0.0, gene_interaction_p_value=None, graph_level=level
        ),
    ).model_dump()


def stored(ds: Any, key: str) -> list[dict[str, Any]]:
    """The ``key`` dictionary of every stored record, in LMDB order."""
    return [ds[i][key] for i in range(len(ds))]


def test_smf_allele_single_and_blank_sd(tmp_path: Path) -> None:
    """Smf: CDC28 (``Allele1`` "cdc28-4", no "delta") becomes an SGA allele, GEM1 a KanMX
    deletion with ``sample_sd`` n 4 (se 0.005); the reference is fitness 1.0, no SD.
    CDC28's blank St.dev. is stored as ``fitness_std`` None with every uncertainty field
    None: the SI defines no meaning for a blank, so it is "no value reported".
    """
    name = "SmfKuzmin2020Dataset"
    ds = build(tmp_path, k.SmfKuzmin2020Dataset)
    experiments = stored(ds, "experiment")
    assert experiments[0]["phenotype"]["fitness_std"] is None
    expected = [
        fitness(
            name,
            [
                SgaAllelePerturbation(
                    systematic_gene_name="YBR160W",
                    perturbed_gene_name="CDC28",
                    strain_id="tsq508",
                )
            ],
            0.83,
            None,
            screen_id=S5_SCREEN,
        ),
        fitness(
            name,
            [
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name="YAL048C",
                    perturbed_gene_name="GEM1",
                    strain_id="sn200",
                )
            ],
            0.95,
            0.01,
            screen_id=S5_SCREEN,
            **labeled(0.01, UncertaintyType.sample_sd, 4),
        ),
    ]
    assert experiments == expected
    assert experiments[1]["phenotype"]["fitness_se"] == 0.005
    assert stored(ds, "reference") == [fitness_reference(name, None)] * 2


def test_dmf_allele_query_ts_array_and_s5_disagreement(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Dmf: the allele query x ts-array cross (blank SD, stored as ``fitness_std`` None
    with no uncertainty), the allele query x gem1 cross (0.72 / 0.05, se 0.025), then ONE
    tm801 record with Table S5's 0.5 / 0.006 as ``bootstrap_se`` over 12 colonies, se
    0.006 (the S1/S3 column's 0.55 is not used). The join logs its match count and the
    0.05 disagreement as a warning.
    """
    name = "DmfKuzmin2020Dataset"
    with caplog.at_level("INFO", logger=k.log.name):
        ds = build(tmp_path, k.DmfKuzmin2020Dataset)
    experiments = stored(ds, "experiment")
    assert experiments[0]["phenotype"]["fitness_std"] is None
    expected = [
        fitness(name, [cdc28(TSQ), CDC10_TS], 0.61, None, screen_id=MAIN),
        fitness(
            name,
            [GEM1, cdc28(TSQ)],
            0.72,
            0.05,
            screen_id=PILOT,
            **labeled(0.05, UncertaintyType.sample_sd, 4),
        ),
        fitness(
            name,
            [cdc28(TM801), PML39_TM801],
            0.5,
            0.006,
            screen_id=S5_SCREEN,
            **labeled(0.006, UncertaintyType.bootstrap_se, 12),
        ),
    ]
    assert experiments == expected
    assert [e["phenotype"]["fitness_se"] for e in experiments] == [None, 0.025, 0.006]
    assert stored(ds, "reference") == [fitness_reference(name, None)] * 3
    records = [
        (r.levelname, r.getMessage()) for r in caplog.records if r.name == k.log.name
    ]
    assert records[:2] == [
        (
            "INFO",
            "Table S5 matched 1 of 1 trigenic query strains; 1 carry a fitness value",
        ),
        (
            "WARNING",
            "Query-strain fitness disagrees between Table S5 and the S1/S3 trigenic "
            "column for 1 of 1 strains (max |diff| 0.0500); Table S5 is used",
        ),
    ]


def test_dmf_refuses_a_repeated_s5_double_mutant_row(tmp_path: Path) -> None:
    """With the tm801 "Double mutant" row listed twice in Table S5, the second time with
    a different fitness (0.51 against 0.5), the build refuses on the tm number before
    the left merge could fan the one strain out into two records, and no LMDB store is
    written.
    """
    repeat = [*S5_DOUBLE[:5], 0.51, S5_DOUBLE[6]]
    with pytest.raises(ValueError) as info:
        build(tmp_path, k.DmfKuzmin2020Dataset, s5_rows=[*S5_ROWS, repeat])
    assert str(info.value) == (
        "Table S5 lists 1 'Double mutant' query strain(s) more than once: ['tm801']; "
        "the tm-number join would store each strain once per listing"
    )
    assert not (tmp_path / "DmfKuzmin2020Dataset" / "processed" / "lmdb").exists()


UNKNOWN_DIGENIC = [TSQ, "cdc28-4+hoΔ", SN, "cdc10-1", "digenic", 0.61, 0.01, 0.83, 0.7, -0.12, 0.03]  # fmt: skip
UNKNOWN_TRIGENIC = [TM801, "cdc28-4+pml39Δ", SN, "cdc10-1", "trigenic", 0.3, 0.02, 0.55, 0.7, -0.2, 0.001]  # fmt: skip


@pytest.mark.parametrize(
    "cls",
    [
        k.DmfKuzmin2020Dataset,
        k.DmiKuzmin2020Dataset,
        k.TmfKuzmin2020Dataset,
        k.TmiKuzmin2020Dataset,
    ],
)
def test_unknown_array_strain_type_refuses_the_build(
    tmp_path: Path, cls: type[Any]
) -> None:
    """``YCR002C_sn1`` is neither a ``tsa`` nor a ``dma`` strain, so nothing records its
    perturbation class: every loader that reads the array column refuses it by name
    instead of storing it as an ``SgaAllelePerturbation``.
    """
    with pytest.raises(ValueError) as info:
        build(tmp_path, cls, s1_rows=[UNKNOWN_DIGENIC, UNKNOWN_TRIGENIC])
    assert str(info.value) == (
        "array strain 'YCR002C_sn1' is neither a 'tsa' (temperature-sensitive allele) "
        "nor a 'dma' (KanMX deletion) strain; its perturbation type is unknown"
    )
    assert not (tmp_path / cls.__name__ / "processed" / "lmdb").exists()


@pytest.mark.parametrize("module", [k, k2018])
@pytest.mark.parametrize("strain", ["YBR001C_TSA100", "YAL048C_DMA5203"])
def test_array_strain_type_is_matched_case_sensitively(
    module: Any, strain: str
) -> None:
    """Both years match the lowercase ``tsa`` / ``dma`` tags the released tables use; an
    uppercase tag is not silently accepted as the same array but refused by name.
    """
    with pytest.raises(ValueError) as info:
        module._array_perturbation_type(strain)
    assert str(info.value) == (
        f"array strain '{strain}' is neither a 'tsa' (temperature-sensitive allele) "
        "nor a 'dma' (KanMX deletion) strain; its perturbation type is unknown"
    )


def test_tmf_allele_first_query_and_ts_array(tmp_path: Path) -> None:
    """Tmf: tm801 against the ts array (0.3 / 0.02) and gem1 (0.25 / 0.03); the allele
    sits in query slot 1 with the split strain id "YBR160W"; no uncertainty labels and a
    reference SD of (0.02 + 0.03) / 2 = 0.025.
    """
    name = "TmfKuzmin2020Dataset"
    ds = build(tmp_path, k.TmfKuzmin2020Dataset)
    assert stored(ds, "experiment") == [
        fitness(name, [CDC28_SPLIT, PML39_SPLIT, NTH2_TS], 0.3, 0.02, screen_id=MAIN),
        fitness(name, [CDC28_SPLIT, PML39_SPLIT, GEM1], 0.25, 0.03, screen_id=PILOT),
    ]
    assert (0.02 + 0.03) / 2 == 0.025
    assert stored(ds, "reference") == [fitness_reference(name, 0.025)] * 2


def test_tmf_blank_sd_is_stored_as_none(tmp_path: Path) -> None:
    """A trigenic S1 row with a blank SD (tm801 against the ts array, 0.3) is stored with
    ``fitness_std`` None, not a float NaN; the reference SD is the mean of the one
    reported SD, 0.03.
    """
    name = "TmfKuzmin2020Dataset"
    blank = [*S1_ROWS[1][:6], None, *S1_ROWS[1][7:]]
    ds = build(tmp_path, k.TmfKuzmin2020Dataset, s1_rows=[S1_ROWS[0], blank])
    experiments = stored(ds, "experiment")
    assert experiments[0]["phenotype"]["fitness_std"] is None
    assert experiments == [
        fitness(name, [CDC28_SPLIT, PML39_SPLIT, NTH2_TS], 0.3, None, screen_id=MAIN),
        fitness(name, [CDC28_SPLIT, PML39_SPLIT, GEM1], 0.25, 0.03, screen_id=PILOT),
    ]
    assert stored(ds, "reference") == [fitness_reference(name, 0.03)] * 2


def test_dmi_and_tmi_allele_records(tmp_path: Path) -> None:
    """Dmi stores the digenic crosses at edge level (-0.12 p 0.03 against the ts array
    cdc10-1; 0.04 p 0.4); Tmi the trigenic rows at hyperedge level (-0.2 p 0.001
    against the ts array; -0.1 p 0.01 against gem1).
    """
    dmi = build(tmp_path, k.DmiKuzmin2020Dataset)
    assert stored(dmi, "experiment") == [
        interaction(
            "DmiKuzmin2020Dataset", [cdc28(TSQ), CDC10_TS], -0.12, 0.03, "edge", MAIN
        ),
        interaction(
            "DmiKuzmin2020Dataset", [GEM1, cdc28(TSQ)], 0.04, 0.4, "edge", PILOT
        ),
    ]
    assert (
        stored(dmi, "reference")
        == [interaction_reference("DmiKuzmin2020Dataset", "edge")] * 2
    )
    tmi = build(tmp_path, k.TmiKuzmin2020Dataset)
    assert stored(tmi, "experiment") == [
        interaction(
            "TmiKuzmin2020Dataset",
            [CDC28_SPLIT, PML39_SPLIT, NTH2_TS],
            -0.2,
            0.001,
            "hyperedge",
            MAIN,
        ),
        interaction(
            "TmiKuzmin2020Dataset",
            [CDC28_SPLIT, PML39_SPLIT, GEM1],
            -0.1,
            0.01,
            "hyperedge",
            PILOT,
        ),
    ]
    assert (
        stored(tmi, "reference")
        == [interaction_reference("TmiKuzmin2020Dataset", "hyperedge")] * 2
    )


def test_subset_n_samples_position_one_with_seed_42(tmp_path: Path) -> None:
    """``subset_n=1`` keeps position 1 of two (``sample(n=1, random_state=42)``): Tmf 0.25
    with the full-frame reference SD 0.025, Dmi epsilon 0.04, Tmi tau -0.1.
    """
    assert pd.DataFrame({"x": range(2)}).sample(
        n=1, random_state=42
    ).index.tolist() == [1]
    tmf = build(tmp_path, k.TmfKuzmin2020Dataset, subset_n=1)
    assert [e["phenotype"]["fitness"] for e in stored(tmf, "experiment")] == [0.25]
    assert stored(tmf, "reference") == [
        fitness_reference("TmfKuzmin2020Dataset", 0.025)
    ]
    dmi = build(tmp_path, k.DmiKuzmin2020Dataset, subset_n=1)
    assert [e["phenotype"]["gene_interaction"] for e in stored(dmi, "experiment")] == [
        0.04
    ]
    tmi = build(tmp_path, k.TmiKuzmin2020Dataset, subset_n=1)
    assert [e["phenotype"]["gene_interaction"] for e in stored(tmi, "experiment")] == [
        -0.1
    ]


@pytest.mark.parametrize(
    ("cls", "n"),
    [
        (k.SmfKuzmin2020Dataset, 2),
        (k.DmfKuzmin2020Dataset, 3),
        (k.TmfKuzmin2020Dataset, 2),
        (k.DmiKuzmin2020Dataset, 2),
        (k.TmiKuzmin2020Dataset, 2),
    ],
)
def test_download_extracts_the_hosted_zip_and_removes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cls: type[Any], n: int
) -> None:
    """With no raw tables and no ``TC_DATA_URL``, the build calls ``download_url`` once
    with the Box archive URL and the raw directory; the zip (all three tables) is
    extracted there and deleted, and the build proceeds on the extracted tables.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    tables = tmp_path / "tables"
    write_tables(tables)
    calls: list[tuple[str, str]] = []

    def fake_download_url(url: str, folder: str) -> str:
        calls.append((url, folder))
        path = osp.join(folder, "464ogx5kpafav7i3zv7gesb9lcn0dm94.zip")
        with zipfile.ZipFile(path, "w") as archive:
            for name in (S1_NAME, S3_NAME, S5_NAME):
                archive.write(tables / name, name)
        return path

    monkeypatch.setattr(k, "download_url", fake_download_url)
    root = tmp_path / cls.__name__
    ds = cls(root=str(root))
    assert calls == [(URL, str(root / "raw"))]
    assert cls.url == URL
    assert sorted(os.listdir(root / "raw")) == [S1_NAME, S3_NAME, S5_NAME]
    assert len(ds) == n


@pytest.mark.parametrize(
    ("cls", "experiment_class", "reference_class"),
    [
        (k.SmfKuzmin2020Dataset, FitnessExperiment, FitnessExperimentReference),
        (k.DmfKuzmin2020Dataset, FitnessExperiment, FitnessExperimentReference),
        (k.TmfKuzmin2020Dataset, FitnessExperiment, FitnessExperimentReference),
        (
            k.DmiKuzmin2020Dataset,
            GeneInteractionExperiment,
            GeneInteractionExperimentReference,
        ),
        (
            k.TmiKuzmin2020Dataset,
            GeneInteractionExperiment,
            GeneInteractionExperimentReference,
        ),
    ],
)
def test_items_retype_through_the_declared_classes(
    tmp_path: Path,
    cls: type[Any],
    experiment_class: type[Any],
    reference_class: type[Any],
) -> None:
    """``transform_item`` rebuilds record 1 of each loader through its declared classes,
    and both dump back to exactly the stored dictionaries.
    """
    ds = build(tmp_path, cls)
    item = ds[1]
    typed = ds.transform_item(item)
    assert type(typed["experiment"]) is experiment_class
    assert type(typed["reference"]) is reference_class
    assert typed["experiment"].model_dump() == item["experiment"]
    assert typed["reference"].model_dump() == item["reference"]
    assert typed["publication"] == PUBLICATION


MAIN_BUILDS = [
    (k.SmfKuzmin2020Dataset, "smf_kuzmin2020", 2),
    (k.DmfKuzmin2020Dataset, "dmf_kuzmin2020", 3),
    (k.TmfKuzmin2020Dataset, "tmf_kuzmin2020", 2),
    (k.DmiKuzmin2020Dataset, "dmi_kuzmin2020", 2),
    (k.TmiKuzmin2020Dataset, "tmi_kuzmin2020", 2),
]


def test_main_builds_all_five_under_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main`` builds Smf, Dmf, Tmf, Dmi and Tmi at
    ``$DATA_ROOT/data/torchcell/<loader>_kuzmin2020``, nothing under the working
    directory. Its stdout is the five builds' index lines, then per loader the class, the
    length and the first item (the same item a build at any other root stores).
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: False)
    firsts = []
    for cls, _, _ in MAIN_BUILDS:
        reference = build(tmp_path / "reference", cls)
        firsts.append(repr(reference[0]))
        reference.close_lmdb()
    capsys.readouterr()
    data_root = tmp_path / "data_root"
    for _, folder, _ in MAIN_BUILDS:
        write_tables(data_root / "data" / "torchcell" / folder / "raw")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    k.main()
    assert capsys.readouterr().out == (
        "Computing experiment_reference_index (streaming)...\n" * 5
        + "".join(
            f"Testing {cls.__name__}:\nLength: {n}\nFirst item: {first}\n\n\n"
            for (cls, _, n), first in zip(MAIN_BUILDS, firsts, strict=True)
        )
    )
    for _, folder, _ in MAIN_BUILDS:
        assert (
            data_root / "data" / "torchcell" / folder / "processed" / "lmdb"
        ).is_dir()
    assert list(work.iterdir()) == []


def test_main_refuses_an_unset_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``DATA_ROOT`` unset ``main`` refuses before building anything, instead of
    building at a root relative to the working directory: nothing is downloaded and the
    (temporary) working directory stays empty.
    """
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: False)
    monkeypatch.delenv("DATA_ROOT")
    monkeypatch.chdir(tmp_path)
    downloads: list[tuple[str, str]] = []
    monkeypatch.setattr(
        k, "download_url", lambda url, folder: downloads.append((url, folder))
    )
    with pytest.raises(ValueError) as info:
        k.main()
    assert str(info.value) == (
        "DATA_ROOT environment variable is not set. Please set it in your .env file."
    )
    assert downloads == []
    assert list(tmp_path.iterdir()) == []


# Issue #602: one (query strain, array strain) cross released in BOTH Table S1 (main
# diagnostic-array screen) and Table S3 (pilot genome-wide-array screen), S3 spelling
# the array allele differently ("CDC10-ph" against S1's "cdc10-1").
PILOT_REPEAT_DIGENIC = [TSQ, "cdc28-4+hoΔ", TSA_CDC10, "CDC10-ph", "digenic", 0.55, 0.02, 0.83, 0.7, -0.12, 0.03]  # fmt: skip
PILOT_REPEAT_TRIGENIC = [TM801, "cdc28-4+pml39Δ", TSA, "NTH2-PH", "trigenic", 0.28, 0.01, 0.55, 0.8, -0.15, 0.002]  # fmt: skip


def experiment_id(dump: dict[str, Any]) -> str:
    """The knowledge-graph experiment id: sha256 of the experiment dump's JSON."""
    return hashlib.sha256(json.dumps(dump).encode("utf-8")).hexdigest()


def test_a_cross_in_both_screens_is_two_records_named_by_screen(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Issue #602: the TSQ x cdc10-1 cross is in S1 (0.61, eps -0.12 p 0.03) and in S3
    (0.55 / 0.02, the SAME eps -0.12 p 0.03). Both measurements are kept, each with the
    screen it came from, and the array strain keeps ONE name, Table S1's "cdc10-1",
    although S3 calls it "CDC10-ph". The two Dmi records differ ONLY in ``screen_id``, so
    their experiment ids differ only because of it; the old dump (no screen) gave both
    the same id.
    """
    s3_rows = [PILOT_REPEAT_DIGENIC, *S3_ROWS]
    with caplog.at_level("INFO", logger=k.log.name):
        dmf = build(tmp_path, k.DmfKuzmin2020Dataset, s3_rows=s3_rows)
    assert (
        "INFO",
        "Table S3 names 1 array strain(s) differently from Table S1 (1 rows); the "
        "Table S1 name is used",
    ) in [(r.levelname, r.getMessage()) for r in caplog.records]
    name = "DmfKuzmin2020Dataset"
    assert stored(dmf, "experiment")[:2] == [
        fitness(name, [cdc28(TSQ), CDC10_TS], 0.61, None, screen_id=MAIN),
        fitness(
            name,
            [cdc28(TSQ), CDC10_TS],
            0.55,
            0.02,
            screen_id=PILOT,
            **labeled(0.02, UncertaintyType.sample_sd, 4),
        ),
    ]
    dmi = build(tmp_path, k.DmiKuzmin2020Dataset, s3_rows=s3_rows)
    main, pilot = stored(dmi, "experiment")[:2]
    assert main == interaction("DmiKuzmin2020Dataset", [cdc28(TSQ), CDC10_TS], -0.12, 0.03, "edge", MAIN)  # fmt: skip
    assert pilot == {**main, "phenotype": {**main["phenotype"], "screen_id": PILOT}}
    assert experiment_id(main) != experiment_id(pilot)
    for dump in (main, pilot):
        dump["phenotype"].pop("screen_id")
    assert experiment_id(main) == experiment_id(pilot)


def test_a_triple_in_both_screens_keeps_both_with_the_s1_array_name(
    tmp_path: Path,
) -> None:
    """The tm801 x nth2-5001 triple is in S1 (0.3, tau -0.2) and S3 (0.28, tau -0.15,
    array named "NTH2-PH"): Tmf and Tmi each store both, tagged main then pilot, and
    both name the array allele "nth2-5001".
    """
    s3_rows = [PILOT_REPEAT_TRIGENIC, *S3_ROWS]
    tmf = build(tmp_path, k.TmfKuzmin2020Dataset, s3_rows=s3_rows)
    assert stored(tmf, "experiment")[:2] == [
        fitness(
            "TmfKuzmin2020Dataset",
            [CDC28_SPLIT, PML39_SPLIT, NTH2_TS],
            0.3,
            0.02,
            screen_id=MAIN,
        ),  # fmt: skip
        fitness(
            "TmfKuzmin2020Dataset",
            [CDC28_SPLIT, PML39_SPLIT, NTH2_TS],
            0.28,
            0.01,
            screen_id=PILOT,
        ),  # fmt: skip
    ]
    tmi = build(tmp_path, k.TmiKuzmin2020Dataset, s3_rows=s3_rows)
    assert stored(tmi, "experiment")[:2] == [
        interaction(
            "TmiKuzmin2020Dataset",
            [CDC28_SPLIT, PML39_SPLIT, NTH2_TS],
            -0.2,
            0.001,
            "hyperedge",
            MAIN,
        ),  # fmt: skip
        interaction(
            "TmiKuzmin2020Dataset",
            [CDC28_SPLIT, PML39_SPLIT, NTH2_TS],
            -0.15,
            0.002,
            "hyperedge",
            PILOT,
        ),  # fmt: skip
    ]


def test_two_names_for_one_array_strain_within_one_table_are_refused(
    tmp_path: Path,
) -> None:
    """A strain named two ways inside ONE table has no main-screen spelling to prefer,
    so the build refuses it by name instead of storing two names for one strain.
    """
    renamed = [*S3_ROWS[0][:3], "gem1-x", *S3_ROWS[0][4:]]
    with pytest.raises(ValueError) as info:
        build(tmp_path, k.DmiKuzmin2020Dataset, s3_rows=[renamed, *S3_ROWS])
    assert str(info.value) == (
        "Table S3 gives 1 array strain(s) more than one 'Array allele name': "
        "['YAL048C_dma5203']"
    )


# ---------------------------------------------------------------------------------------
# Data-gated tests on the real supplementary tables.
# ---------------------------------------------------------------------------------------

_RAW_DIR = osp.join(os.environ["DATA_ROOT"], "data/torchcell/dmf_kuzmin2020/raw")
_RAW_FILES = [S1_NAME, S3_NAME, S5_NAME]
_needs_raw = pytest.mark.skipif(
    not all(osp.exists(osp.join(_RAW_DIR, f)) for f in _RAW_FILES),
    reason=f"requires the Kuzmin 2020 raw supplementary tables in {_RAW_DIR}",
)

# The digenic row used by the regression test, addressed by strain ids.
_DIGENIC_QUERY_STRAIN = "YAL015C+YDL227C_tm461"
_DIGENIC_ARRAY_STRAIN = "YBL007C_dma91"
_DIGENIC_FITNESS = 0.9695
_DIGENIC_FITNESS_SD = 0.0465

# A double-mutant query strain of the trigenic screens (Table S5 entry "tm72").
_QUERY_STRAIN = "YAL015C+YOL043C_tm72"
_QUERY_STRAIN_FITNESS = 1.0133
_QUERY_STRAIN_SD = 0.008


@pytest.fixture(scope="module")
def raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tables S1, S3, and S5 as released."""
    df_s1, df_s3, df_s5 = (
        pd.read_excel(osp.join(_RAW_DIR, f), skiprows=1) for f in _RAW_FILES
    )
    return df_s1, df_s3, df_s5


@pytest.fixture(scope="module")
def built(
    raw: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> tuple[Any, pd.DataFrame]:
    """The loader instance and its preprocessed frame."""
    from torchcell.datasets.scerevisiae.kuzmin2020 import DmfKuzmin2020Dataset

    df_s1, df_s3, df_s5 = raw
    dataset = DmfKuzmin2020Dataset.__new__(DmfKuzmin2020Dataset)
    return dataset, dataset.preprocess_raw(df_s1.copy(), df_s3.copy(), df_s5.copy())


@pytest.mark.data
@pytest.mark.slow
@_needs_raw
def test_trigenic_query_strains_are_kept_one_record_each(
    raw: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    built: tuple[Any, pd.DataFrame],
) -> None:
    """The trigenic rows are no longer dropped, and collapse to one record/strain."""
    from torchcell.datasets.scerevisiae.kuzmin2020 import (
        RECORD_KIND_DIGENIC_ARRAY_CROSS,
        RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN,
    )

    df_s1, df_s3, _ = raw
    _, df = built
    combined = pd.concat([df_s1, df_s3])
    trigenic = combined[combined["Combined mutant type"] == "trigenic"]

    counts = df["record_kind"].value_counts().to_dict()
    assert counts[RECORD_KIND_DIGENIC_ARRAY_CROSS] == int(
        (combined["Combined mutant type"] == "digenic").sum()
    )
    assert counts[RECORD_KIND_DIGENIC_ARRAY_CROSS] == 632_797
    assert counts[RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN] == 201

    # One record per strain, not one per trigenic row.
    assert len(trigenic) == 301_798
    query_rows = df[df["record_kind"] == RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN]
    assert query_rows["Query strain ID"].is_unique
    assert query_rows["fitness"].notna().all()


@pytest.mark.data
@pytest.mark.slow
@_needs_raw
def test_table_s5_join_is_on_the_tm_number(
    raw: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    built: tuple[Any, pd.DataFrame],
) -> None:
    """The corrected join matches; the old full-strain-id join matched nothing."""
    from torchcell.datasets.scerevisiae.kuzmin2020 import (
        RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN,
    )

    df_s1, df_s3, df_s5 = raw
    combined = pd.concat([df_s1, df_s3])
    trigenic = combined[combined["Combined mutant type"] == "trigenic"]
    s5_double = df_s5[df_s5["Mutant type"] == "Double mutant"]

    # The defect: S5 keys on the bare tm number, S1/S3 on "<ORF>+<ORF>_tm<N>".
    assert not set(trigenic["Query strain ID"]) & set(s5_double["Query Strain ID"])
    tm = trigenic["Query strain ID"].str.rsplit("_", n=1).str[-1]
    matched = trigenic[tm.isin(set(s5_double["Query Strain ID"]))]
    assert len(matched) == 301_798

    # Every query-strain record's SD is the S5 "St.dev." for its tm number.
    _, df = built
    query_rows = df[df["record_kind"] == RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN]
    s5_sd = s5_double.set_index("Query Strain ID")["St.dev."]
    from_s5 = (
        query_rows["Query strain ID"].str.rsplit("_", n=1).str[-1].map(s5_sd).to_numpy()
    )
    assert query_rows["fitness_std"].to_numpy() == pytest.approx(from_s5)
    assert query_rows["fitness_std"].notna().all()


@pytest.mark.data
@pytest.mark.slow
@_needs_raw
def test_query_strain_record_is_the_query_pair(built: tuple[Any, pd.DataFrame]) -> None:
    """Genotype is the two QUERY genes; fitness and bootstrap SD come from S5."""
    from torchcell.datamodels.schema import Genotype, SampleUnit, UncertaintyType
    from torchcell.datasets.scerevisiae.kuzmin2020 import (
        N_SAMPLES_QUERY_STRAIN_FITNESS,
        RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN,
        DmfKuzmin2020Dataset,
    )

    _, df = built
    rows = df[
        (df["record_kind"] == RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN)
        & (df["Query strain ID"] == _QUERY_STRAIN)
    ]
    assert len(rows) == 1
    row = rows.iloc[0]

    experiment, _, _ = DmfKuzmin2020Dataset.create_experiment("dmf_kuzmin2020", row)
    genotype = experiment.genotype
    assert isinstance(genotype, Genotype)
    perturbations = genotype.perturbations
    assert len(perturbations) == 2
    assert [p.systematic_gene_name for p in perturbations] == ["YAL015C", "YOL043C"]
    # kuzmin2020 names perturbed genes without the "_delta" allele suffix, as its
    # digenic / Tmf / Tmi records do.
    assert [p.perturbed_gene_name for p in perturbations] == ["ntg1", "ntg2"]
    assert {p.model_dump()["strain_id"] for p in perturbations} == {_QUERY_STRAIN}
    assert row["Array systematic name"] not in {
        p.systematic_gene_name for p in perturbations
    }

    phenotype = experiment.phenotype
    assert phenotype.fitness == pytest.approx(_QUERY_STRAIN_FITNESS)
    assert phenotype.fitness_std == pytest.approx(_QUERY_STRAIN_SD)
    # A bootstrap SD of the estimator is already an SE: used as-is, not divided.
    assert phenotype.fitness_uncertainty_type == UncertaintyType.bootstrap_se
    assert phenotype.fitness_se == pytest.approx(_QUERY_STRAIN_SD)
    assert phenotype.n_samples == N_SAMPLES_QUERY_STRAIN_FITNESS == 12
    assert phenotype.sample_unit == SampleUnit.colony


@pytest.mark.data
@pytest.mark.slow
@_needs_raw
def test_digenic_record_unchanged(built: tuple[Any, pd.DataFrame]) -> None:
    """Regression: a digenic record equals the object the old path produced, plus the
    main-screen ``screen_id`` (issue #602; this cross is in Table S1 only).
    """
    from torchcell.datamodels.media import SGA_TM_SELECTION
    from torchcell.datamodels.schema import (
        Environment,
        FitnessExperiment,
        FitnessPhenotype,
        Genotype,
        SampleUnit,
        SgaKanMxDeletionPerturbation,
        Temperature,
        UncertaintyType,
    )
    from torchcell.datasets.scerevisiae.kuzmin2020 import (
        RECORD_KIND_DIGENIC_ARRAY_CROSS,
        DmfKuzmin2020Dataset,
    )

    _, df = built
    rows = df[
        (df["record_kind"] == RECORD_KIND_DIGENIC_ARRAY_CROSS)
        & (df["Query strain ID"] == _DIGENIC_QUERY_STRAIN)
        & (df["Array strain ID"] == _DIGENIC_ARRAY_STRAIN)
    ]
    assert len(rows) == 1

    experiment, _, _ = DmfKuzmin2020Dataset.create_experiment(
        "dmf_kuzmin2020", rows.iloc[0]
    )

    expected = FitnessExperiment(
        dataset_name="dmf_kuzmin2020",
        genotype=Genotype(
            perturbations=[
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name="YAL015C",
                    perturbed_gene_name="ntg1",
                    strain_id=_DIGENIC_QUERY_STRAIN,
                ),
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name="YBL007C",
                    perturbed_gene_name="sla1",
                    strain_id=_DIGENIC_ARRAY_STRAIN,
                ),
            ]
        ),
        environment=Environment(
            media=SGA_TM_SELECTION, temperature=Temperature(value=26)
        ),
        phenotype=FitnessPhenotype(
            fitness=_DIGENIC_FITNESS,
            fitness_std=_DIGENIC_FITNESS_SD,
            fitness_uncertainty=_DIGENIC_FITNESS_SD,
            fitness_uncertainty_type=UncertaintyType.sample_sd,
            n_samples=4,
            sample_unit=SampleUnit.colony,
            screen_id=k.SCREEN_ID_MAIN,
        ),
    )
    assert experiment.model_dump() == expected.model_dump()


# Issue #602, the example pair: the digenic cross gpb2 x tfc3 is released in both
# screens with different values, and S3 spells the array allele "tfc3-g349e".
_BOTH_SCREENS_QUERY = "YAL056W+YDL227C_tm1888"
_BOTH_SCREENS_ARRAY = "YAL001C_tsa508"


@pytest.fixture(scope="module")
def dmi_frame(raw: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    """The Dmi preprocessed frame of the released tables."""
    df_s1, df_s3, _ = raw
    dataset = k.DmiKuzmin2020Dataset.__new__(k.DmiKuzmin2020Dataset)
    return dataset.preprocess_raw(df_s1.copy(), df_s3.copy())


@pytest.mark.data
@pytest.mark.slow
@_needs_raw
def test_issue_602_pair_is_two_records_one_per_screen(
    built: tuple[Any, pd.DataFrame], dmi_frame: pd.DataFrame
) -> None:
    """The issue's example cross: Dmf stores S1's 0.9558 / 0.0188 as the main screen and
    S3's 0.8746 / 0.0171 as the pilot screen, Dmi S1's epsilon -0.013157 and S3's
    0.008843; every record names the array allele "tfc3-G349E" (Table S1's spelling).
    """
    _, dmf = built
    rows = dmf[
        (dmf["Query strain ID"] == _BOTH_SCREENS_QUERY)
        & (dmf["Array strain ID"] == _BOTH_SCREENS_ARRAY)
    ]
    phenotypes = [
        k.DmfKuzmin2020Dataset.create_experiment("dmf", row)[0].phenotype
        for _, row in rows.iterrows()
    ]
    assert [(p.screen_id, p.fitness, p.fitness_std) for p in phenotypes] == [
        (k.SCREEN_ID_MAIN, 0.9558, 0.0188),
        (k.SCREEN_ID_PILOT, 0.8746, 0.0171),
    ]
    assert rows["Array allele name"].tolist() == ["tfc3-G349E"] * 2
    dmi = dmi_frame[
        (dmi_frame["Query strain ID"] == _BOTH_SCREENS_QUERY)
        & (dmi_frame["Array strain ID"] == _BOTH_SCREENS_ARRAY)
    ]
    experiments = [
        k.DmiKuzmin2020Dataset.create_experiment("dmi", row)[0]
        for _, row in dmi.iterrows()
    ]
    assert [
        (e.phenotype.screen_id, e.phenotype.gene_interaction) for e in experiments
    ] == [(k.SCREEN_ID_MAIN, -0.013157), (k.SCREEN_ID_PILOT, 0.008843)]
    genotypes = [e.genotype for e in experiments]
    assert all(isinstance(g, Genotype) for g in genotypes)
    assert {
        p.perturbed_gene_name
        for g in genotypes
        if isinstance(g, Genotype)
        for p in g.perturbations
    } == {"gpb2", "tfc3-G349E"}


@pytest.mark.data
@pytest.mark.slow
@_needs_raw
def test_released_digenic_rows_per_screen_and_one_name_per_array(
    dmi_frame: pd.DataFrame,
) -> None:
    """Dmi keeps all 632,797 digenic rows, 537,911 from the main screen and 94,886 from
    the pilot screen, and after the S1-spelling rule each of the 4,553 array strains has
    exactly one allele name (70 shared strains had two before, issue #602).
    """
    assert dmi_frame["screen_id"].value_counts().to_dict() == {
        k.SCREEN_ID_MAIN: 537_911,
        k.SCREEN_ID_PILOT: 94_886,
    }
    names = dmi_frame.groupby("Array strain ID")["Array allele name"].nunique()
    assert (names == 1).all()
    assert len(names) == 4_553
