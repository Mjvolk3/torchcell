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
S1     YBR160W+YDL227C_tsq508  cdc28-4+hoΔ      YCR002C_sn1      dig   0.61  blank  0.83   -0.12  0.03
S1     YBR160W+YML107C_tm801   cdc28-4+pml39Δ   YBR001C_tsa100   tri   0.3   0.02   0.55   -0.2   0.001
S3     YBR160W+YDL227C_tsq508  cdc28-4+hoΔ      YAL048C_dma5203  dig   0.72  0.05   0.83   0.04   0.4
S3     YBR160W+YML107C_tm801   cdc28-4+pml39Δ   YAL048C_dma5203  tri   0.25  0.03   0.55   -0.1   0.01
=====  ======================  ===============  ===============  ====  ====  =====  =====  =====  =====

Table S5: single mutants CDC28 (``cdc28-4``, sn tsq508, 0.83, St.dev. blank) and GEM1
(``delta``, sn200, 0.95 / 0.01), and the double mutant tm801 (0.5 / 0.006). Expected:

- Smf: CDC28 as an ``SgaAllelePerturbation`` (Gene1 verbatim, strain ``tsq508``) and GEM1
  as a KanMX deletion (0.95 / 0.01, labeled ``sample_sd`` n 4, se 0.005); the reference
  is fitness 1.0 with no SD.
- Dmf: the S1 then S3 digenic crosses, the allele query paired with ``YCR002C_sn1``
  (neither ``dma`` nor ``tsa``: stored as an ``SgaAllelePerturbation``, line 604) at 0.61
  with a blank SD, and with gem1 at 0.72 / 0.05 (se 0.025); then ONE record for tm801
  although it appears in both S1 and S3 (drop_duplicates on the strain id), with S5's
  0.5 / 0.006 labeled ``bootstrap_se`` over ``N_SAMPLES_QUERY_STRAIN_FITNESS = 12``
  colonies (the SI's 12 to 24 colony measurements, lower end), se 0.006 undivided.
  S5's 0.5 and the S1/S3 column's 0.55 differ by 0.05 > 1e-3, so the disagreement
  warning fires with "(max |diff| 0.0500)".
- Tmf: 0.3 / 0.02 and 0.25 / 0.03, allele in query slot 1, the ts array in the first;
  no uncertainty labels; reference SD (0.02 + 0.03) / 2 = 0.025. Dmi: -0.12 (p 0.03) and
  0.04 (p 0.4) at edge level. Tmi: -0.2 (p 0.001) and -0.1 (p 0.01) at hyperedge level.
- ``subset_n=1`` samples position 1 of two with ``random_state=42``.

Findings pinned here: a blank SD is stored as ``fitness_std`` NaN on the Smf and Dmf
digenic records (lines 339 and 627 pass the raw cell; only the uncertainty fields go
through ``pd.isna``); a Table S5 "Double mutant" row listed twice doubles that strain's
record (the left merge on the tm number, line 173, fans out); ``main`` builds only Tmi,
at the class default ``root="data/torchcell/tmi_kuzmin2020"`` relative to the working
directory (line 1384).
"""

import math
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
TM801 = "YBR160W+YML107C_tm801"
TSA = "YBR001C_tsa100"
DMA = "YAL048C_dma5203"
SN = "YCR002C_sn1"
S1_ROWS: list[list[Any]] = [
    [TSQ, "cdc28-4+hoΔ", SN, "cdc10-1", "digenic", 0.61, None, 0.83, 0.7, -0.12, 0.03],
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
CDC10_UNKNOWN = SgaAllelePerturbation(
    systematic_gene_name="YCR002C", perturbed_gene_name="cdc10-1", strain_id=SN
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


def write_tables(folder: Path, s5_rows: list[list[Any]] = S5_ROWS) -> None:
    """Write Tables S1, S3 and S5 into ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    write_xlsx(folder / S1_NAME, S13_COLUMNS, S1_ROWS)
    write_xlsx(folder / S3_NAME, S13_COLUMNS, S3_ROWS)
    write_xlsx(folder / S5_NAME, S5_COLUMNS, s5_rows)


def build(
    tmp_path: Path, cls: type[Any], s5_rows: list[list[Any]] = S5_ROWS, **kw: Any
) -> Any:
    """Build ``cls`` under ``tmp_path/<class name>`` from the three tables."""
    root = tmp_path / cls.__name__
    write_tables(root / "raw", s5_rows)
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
    name: str, perturbations: list[Any], value: float, p: float, level: str
) -> dict[str, Any]:
    """A Kuzmin 2020 interaction experiment dump at graph level ``level``."""
    return GeneInteractionExperiment(
        dataset_name=name,
        genotype=Genotype(perturbations=perturbations),
        environment=ENVIRONMENT,
        phenotype=GeneInteractionPhenotype(
            gene_interaction=value, gene_interaction_p_value=p, graph_level=level
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


def pop_nan_std(experiment: dict[str, Any]) -> None:
    """Remove a NaN ``fitness_std`` after asserting it is a float NaN."""
    value = experiment["phenotype"].pop("fitness_std")
    assert isinstance(value, float) and math.isnan(value)


def test_smf_allele_single_and_blank_sd(tmp_path: Path) -> None:
    """Smf: CDC28 (``Allele1`` "cdc28-4", no "delta") becomes an SGA allele, GEM1 a KanMX
    deletion with ``sample_sd`` n 4 (se 0.005); the reference is fitness 1.0, no SD.

    Finding: CDC28's blank St.dev. is stored as ``fitness_std`` NaN with every
    uncertainty field None. Pinned until a blank SD maps to None.
    """
    name = "SmfKuzmin2020Dataset"
    ds = build(tmp_path, k.SmfKuzmin2020Dataset)
    experiments = stored(ds, "experiment")
    pop_nan_std(experiments[0])
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
            **labeled(0.01, UncertaintyType.sample_sd, 4),
        ),
    ]
    expected[0]["phenotype"].pop("fitness_std")
    assert experiments == expected
    assert experiments[1]["phenotype"]["fitness_se"] == 0.005
    assert stored(ds, "reference") == [fitness_reference(name, None)] * 2


def test_dmf_allele_query_unknown_array_and_s5_disagreement(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Dmf: the allele query x unknown-array cross (blank SD), the allele query x gem1
    cross (0.72 / 0.05, se 0.025), then ONE tm801 record with Table S5's 0.5 / 0.006
    as ``bootstrap_se`` over 12 colonies, se 0.006 (the S1/S3 column's 0.55 is not used).
    The join logs its match count and the 0.05 disagreement as a warning.

    Finding: record 0's blank SD is stored as ``fitness_std`` NaN.
    """
    name = "DmfKuzmin2020Dataset"
    with caplog.at_level("INFO", logger=k.log.name):
        ds = build(tmp_path, k.DmfKuzmin2020Dataset)
    experiments = stored(ds, "experiment")
    pop_nan_std(experiments[0])
    expected = [
        fitness(name, [cdc28(TSQ), CDC10_UNKNOWN], 0.61, None),
        fitness(
            name,
            [GEM1, cdc28(TSQ)],
            0.72,
            0.05,
            **labeled(0.05, UncertaintyType.sample_sd, 4),
        ),
        fitness(
            name,
            [cdc28(TM801), PML39_TM801],
            0.5,
            0.006,
            **labeled(0.006, UncertaintyType.bootstrap_se, 12),
        ),
    ]
    expected[0]["phenotype"].pop("fitness_std")
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


def test_dmf_repeated_s5_double_mutant_row_doubles_the_strain_record(
    tmp_path: Path,
) -> None:
    """Finding: with the tm801 "Double mutant" row listed twice in Table S5, the left
    merge on the tm number returns two rows for the one strain, so Dmf stores two
    identical tm801 records (4 in all). Pinned until the join refuses a duplicated key.
    """
    ds = build(tmp_path, k.DmfKuzmin2020Dataset, s5_rows=[*S5_ROWS, S5_DOUBLE])
    assert len(ds) == 4
    assert ds.df["Query strain ID"].tolist() == [TSQ, TSQ, TM801, TM801]
    record = fitness(
        "DmfKuzmin2020Dataset",
        [cdc28(TM801), PML39_TM801],
        0.5,
        0.006,
        **labeled(0.006, UncertaintyType.bootstrap_se, 12),
    )
    assert [ds[2]["experiment"], ds[3]["experiment"]] == [record, record]


def test_tmf_allele_first_query_and_ts_array(tmp_path: Path) -> None:
    """Tmf: tm801 against the ts array (0.3 / 0.02) and gem1 (0.25 / 0.03); the allele
    sits in query slot 1 with the split strain id "YBR160W"; no uncertainty labels and a
    reference SD of (0.02 + 0.03) / 2 = 0.025.
    """
    name = "TmfKuzmin2020Dataset"
    ds = build(tmp_path, k.TmfKuzmin2020Dataset)
    assert stored(ds, "experiment") == [
        fitness(name, [CDC28_SPLIT, PML39_SPLIT, NTH2_TS], 0.3, 0.02),
        fitness(name, [CDC28_SPLIT, PML39_SPLIT, GEM1], 0.25, 0.03),
    ]
    assert (0.02 + 0.03) / 2 == 0.025
    assert stored(ds, "reference") == [fitness_reference(name, 0.025)] * 2


def test_dmi_and_tmi_allele_records(tmp_path: Path) -> None:
    """Dmi stores the digenic crosses at edge level (-0.12 p 0.03 with the unknown array
    as an SGA allele; 0.04 p 0.4); Tmi the trigenic rows at hyperedge level (-0.2 p 0.001
    against the ts array; -0.1 p 0.01 against gem1).
    """
    dmi = build(tmp_path, k.DmiKuzmin2020Dataset)
    assert stored(dmi, "experiment") == [
        interaction(
            "DmiKuzmin2020Dataset", [cdc28(TSQ), CDC10_UNKNOWN], -0.12, 0.03, "edge"
        ),
        interaction("DmiKuzmin2020Dataset", [GEM1, cdc28(TSQ)], 0.04, 0.4, "edge"),
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
        ),
        interaction(
            "TmiKuzmin2020Dataset",
            [CDC28_SPLIT, PML39_SPLIT, GEM1],
            -0.1,
            0.01,
            "hyperedge",
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


def test_main_builds_tmi_under_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Finding: ``main`` builds only ``TmiKuzmin2020Dataset()`` (the other four are
    commented out) at the class default root, which is relative, so the store lands under
    the working directory, not ``$DATA_ROOT``. Its stdout is the build's index line, then
    the class, the length and the first item. Pinned until ``main`` takes a root under ``$DATA_ROOT``.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    reference = build(tmp_path / "reference", k.TmiKuzmin2020Dataset)
    first = repr(reference[0])
    reference.close_lmdb()
    capsys.readouterr()
    work = tmp_path / "work"
    write_tables(work / "data" / "torchcell" / "tmi_kuzmin2020" / "raw")
    monkeypatch.chdir(work)
    k.main()
    assert capsys.readouterr().out == (
        "Computing experiment_reference_index (streaming)...\n"
        f"Testing TmiKuzmin2020Dataset:\nLength: 2\nFirst item: {first}\n\n\n"
    )
    assert (
        work / "data" / "torchcell" / "tmi_kuzmin2020" / "processed" / "lmdb"
    ).is_dir()


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
    """Regression: a digenic record equals the object the old path produced."""
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
        ),
    )
    assert experiment.model_dump() == expected.model_dump()
