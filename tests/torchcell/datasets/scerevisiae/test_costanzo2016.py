# tests/torchcell/datasets/scerevisiae/test_costanzo2016.py
# [[tests.torchcell.datasets.scerevisiae.test_costanzo2016]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_costanzo2016.py
"""Costanzo 2016 SMF / DMF / DMI loaders built end to end over hand-written raw files.

Every build is hermetic: the raw spreadsheet (SMF) and the four SGA text files (DMF,
DMI) are written into ``<root>/raw/`` before construction, so PyG never calls
``download()``; ``process()`` then runs once and ``@post_process`` writes the gene set,
the reference index, and the build manifest. Expected records are hand-built from the
schema classes and compared by ``model_dump`` equality, so a change to any stored field
(media, uncertainty typing, publication) fails here. Nothing touches ``$DATA_ROOT``.

The SMF fixture has six raw rows (five distinct strains covering the five perturbation
suffixes, one exact duplicate of the ``_dma1`` row, and one strain with no 26 C value)
which become nine records. The SGA fixture has five rows across the four files, three
at 30 C and two at 26 C.

2026.09.30 (Phase 14): three more synthetic builds and the three ``download`` paths.

- ``smf_twins`` (issue #410): four raw rows in the released shape, where a deletion or
  DAmP strain repeats ONE temperature-combined value in both columns. ``_dma1`` (0.95,
  0.01 at both), ``_damp1`` (0.80, 0.03 at both), ``_tsa1`` (0.85, 0.02 at 26 C; 0.60,
  0.04 at 30 C) and ``_dma5`` (0.90 with a blank 26 C stddev; 0.90, 0.01 at 30 C). The
  blank stddev drops that strain's 26 C row (``dropna`` over every column), so 7
  records: 26 C [dma1, damp1, tsa1] = 0..2 and 30 C [dma1, damp1, tsa1, dma5] = 3..6.
  Reference noise: 26 C (0.01 + 0.03 + 0.02) / 3 = 0.02; 30 C (0.01 + 0.03 + 0.04 +
  0.01) / 4 = 0.0225 (both exact in float64, checked with pandas). Records 0 and 3 are
  the phantom twins: identical but for the temperature.
- ``_SGA_EDGE_FILES``: a suppressor query against a DAmP array at DMA30 with a blank
  DMF stddev (in ``SGA_ExE.txt``) and a KanMX query against a NatMX array at DMA30
  (in ``SGA_NxN.txt``; the other two files are header only). DMF: the blank SD is
  stored as a NaN SE typed ``sample_sd``; the 30 C reference SD is the NaN-skipping
  mean 0.05, SE 0.05 / sqrt(4) = 0.025. DMI: epsilon 0.1 / p 0.2 and -0.2 / 0.01 with
  the suppressor, DAmP, KanMX and NatMX classes.
- A ``TSA22`` row: DMF raises ``UnboundLocalError`` (no reference SD at 22 C), DMI
  stores it at 22 C. A blank DMF value: DMF refuses with ``Fitness cannot be NaN``.
- ``download``: ``download_url`` is faked to drop a zip with the released
  subdirectory holding the spreadsheet, the four SGA files and a ``readme.txt``. SMF
  keeps only the spreadsheet (every ``.txt`` removed); DMF and DMI keep the four SGA
  files and ``readme.txt`` and remove the spreadsheet; the zip is removed in all three.

Findings pinned: the 26 C and 30 C twins of issue #410 (lines 223-266); a blank SMF
stddev drops a measured fitness (line 267); a blank DMF stddev passes the ``is not
None`` guard as NaN (line 730); DMF and SMF crash with ``UnboundLocalError`` at a
temperature other than 26 or 30 C (lines 377-381 and 742-746).
"""

from __future__ import annotations

import json
import math
import os
import os.path as osp
import socket
import zipfile
from pathlib import Path
from typing import Any

import lmdb
import pandas as pd
import pytest

from torchcell.data import ExperimentDataset
from torchcell.datamodels.media import SGA_DM_SELECTION
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
    SgaDampPerturbation,
    SgaKanMxDeletionPerturbation,
    SgaNatMxDeletionPerturbation,
    SgaSuppressorAllelePerturbation,
    SgaTsAllelePerturbation,
    Temperature,
    UncertaintyType,
)
from torchcell.datasets.scerevisiae import costanzo2016 as c

# --------------------------------------------------------------------------- #
# SMF fixture
# --------------------------------------------------------------------------- #
_SMF_COLUMNS = [
    "Strain ID",
    "Systematic gene name",
    "Allele/Gene name",
    "Single mutant fitness (26°)",
    "Single mutant fitness (26°) stddev",
    "Single mutant fitness (30°)",
    "Single mutant fitness (30°) stddev",
]
# (strain id, systematic, allele, f26, sd26, f30, sd30). Row 3 duplicates row 1 exactly
# (dropped by drop_duplicates); row 2 has no 26 C value (its 26 C row is dropped by
# dropna). Suffixes: tsa -> ts allele, dma -> KanMX, sn -> NatMX, damp -> DAmP, S ->
# suppressor.
_SMF_ROWS: list[tuple[Any, ...]] = [
    ("YAL001C_tsa1", "YAL001C", "tfc3-1", 0.85, 0.02, 0.60, 0.04),
    ("YAL002W_dma1", "YAL002W", "VPS8", 0.95, 0.01, 0.90, 0.02),
    ("YAL003W_sn1", "YAL003W", "EFB1", None, None, 0.70, 0.03),
    ("YAL002W_dma1", "YAL002W", "VPS8", 0.95, 0.01, 0.90, 0.02),
    ("YAL005C_damp1", "YAL005C", "ssa1-damp", 0.80, 0.03, 0.75, 0.05),
    ("YAL007C_S1", "YAL007C", "erp2-S1", 1.05, 0.04, 1.02, 0.06),
]
# Reference noise = mean stddev per temperature over the surviving rows:
# 26 C: (0.02 + 0.01 + 0.03 + 0.04) / 4 ; 30 C: (0.04 + 0.02 + 0.03 + 0.05 + 0.06) / 5
_SMF_REF_STD_26 = 0.025
_SMF_REF_STD_30 = 0.04
# Record order = concat([26 C rows, 30 C rows]) after dropna + drop_duplicates.
_SMF_ORDER = [
    ("YAL001C_tsa1", 26),
    ("YAL002W_dma1", 26),
    ("YAL005C_damp1", 26),
    ("YAL007C_S1", 26),
    ("YAL001C_tsa1", 30),
    ("YAL002W_dma1", 30),
    ("YAL003W_sn1", 30),
    ("YAL005C_damp1", 30),
    ("YAL007C_S1", 30),
]

# --------------------------------------------------------------------------- #
# SGA (DMF / DMI) fixture
# --------------------------------------------------------------------------- #
_SGA_COLUMNS = [
    "Query Strain ID",
    "Query allele name",
    "Array Strain ID",
    "Array allele name",
    "Arraytype/Temp",
    "Genetic interaction score (ε)",
    "P-value",
    "Query single mutant fitness (SMF)",
    "Array SMF",
    "Double mutant fitness",
    "Double mutant fitness standard deviation",
]
# Files are read in raw_file_names order and concatenated, so record i is row i below.
_SGA_FILES: dict[str, list[tuple[Any, ...]]] = {
    "SGA_DAmP.txt": [
        (
            "YAL001C_damp1",
            "tfc3-damp",
            "YAL002W_dma1",
            "vps8",
            "DMA30",
            -0.12,
            0.001,
            0.9,
            0.95,
            0.70,
            0.05,
        )
    ],
    "SGA_ExE.txt": [
        (
            "YAL003W_tsq1",
            "efb1-1",
            "YAL005C_tsa1",
            "ssa1-1",
            "TSA26",
            0.08,
            0.02,
            0.8,
            0.9,
            0.80,
            0.03,
        )
    ],
    "SGA_ExN_NxE.txt": [
        (
            "YAL007C_tsq2",
            "erp2-1",
            "YAL008W_dma2",
            "fun14",
            "DMA26",
            -0.30,
            0.0001,
            0.7,
            0.95,
            0.50,
            0.06,
        )
    ],
    "SGA_NxN.txt": [
        (
            "YAL009W_sn1",
            "spo7",
            "YAL010C_dma3",
            "mdm10",
            "DMA30",
            0.02,
            0.5,
            1.0,
            0.99,
            0.99,
            0.01,
        ),
        (
            "YAL011W_sn2",
            "swc3",
            "YAL012W_S1",
            "cys3-S1",
            "DMA30",
            0.05,
            0.3,
            0.98,
            1.0,
            1.01,
            0.02,
        ),
    ],
}
# Reference noise = mean DMF stddev per temperature:
# 30 C rows 0, 3, 4: (0.05 + 0.01 + 0.02) / 3 ; 26 C rows 1, 2: (0.03 + 0.06) / 2
_DMF_REF_STD_30 = 0.08 / 3
_DMF_REF_STD_26 = 0.045
_SGA_TEMPERATURES = [30, 26, 26, 30, 30]
_SGA_GENES = [
    "YAL001C",
    "YAL002W",
    "YAL003W",
    "YAL005C",
    "YAL007C",
    "YAL008W",
    "YAL009W",
    "YAL010C",
    "YAL011W",
    "YAL012W",
]

_PUBLICATION = Publication(
    pubmed_id="27708008",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/27708008/",
    doi="10.1126/science.aaf1420",
    doi_url="https://www.science.org/doi/10.1126/science.aaf1420",
)
_GENOME = ReferenceGenome(species="Saccharomyces cerevisiae", strain="S288C")


def _write_smf_raw(root: Path) -> None:
    (root / "raw").mkdir(parents=True)
    pd.DataFrame(_SMF_ROWS, columns=_SMF_COLUMNS).to_excel(
        root / "raw" / "strain_ids_and_single_mutant_fitness.xlsx", index=False
    )


def _write_sga_raw(root: Path, files: dict[str, list[tuple[Any, ...]]]) -> None:
    (root / "raw").mkdir(parents=True)
    for name, rows in files.items():
        pd.DataFrame(rows, columns=_SGA_COLUMNS).to_csv(
            root / "raw" / name, sep="\t", index=False
        )


def _environment(temperature: int) -> Environment:
    return Environment(
        media=SGA_DM_SELECTION, temperature=Temperature(value=temperature)
    )


def _interned_entries(processed_dir: str) -> int:
    env = lmdb.open(osp.join(processed_dir, "interned"), readonly=True, lock=False)
    with env.begin() as txn:
        entries = txn.stat()["entries"]
    env.close()
    return int(entries)


def _reference_index(preprocess_dir: str) -> list[tuple[list[int], float]]:
    """``(member_indices, reference temperature)`` per stored reference, in file order."""
    with open(osp.join(preprocess_dir, "experiment_reference_index.json")) as f:
        stored = json.load(f)
    return [
        (
            item["member_indices"],
            item["reference"]["environment_reference"]["temperature"]["value"],
        )
        for item in stored
    ]


@pytest.fixture(scope="module")
def smf(tmp_path_factory: pytest.TempPathFactory) -> c.SmfCostanzo2016Dataset:
    root = tmp_path_factory.mktemp("costanzo2016") / "smf"
    _write_smf_raw(root)
    return c.SmfCostanzo2016Dataset(root=str(root))


@pytest.fixture(scope="module")
def dmf(tmp_path_factory: pytest.TempPathFactory) -> c.DmfCostanzo2016Dataset:
    root = tmp_path_factory.mktemp("costanzo2016") / "dmf"
    _write_sga_raw(root, _SGA_FILES)
    return c.DmfCostanzo2016Dataset(root=str(root), io_workers=1, batch_size=2)


@pytest.fixture(scope="module")
def dmi(tmp_path_factory: pytest.TempPathFactory) -> c.DmiCostanzo2016Dataset:
    root = tmp_path_factory.mktemp("costanzo2016") / "dmi"
    _write_sga_raw(root, _SGA_FILES)
    return c.DmiCostanzo2016Dataset(root=str(root), io_workers=1, batch_size=2)


# --------------------------------------------------------------------------- #
# SMF
# --------------------------------------------------------------------------- #
def test_smf_record_order_is_26c_rows_then_30c_rows_after_dropna_and_dedup(
    smf: c.SmfCostanzo2016Dataset,
) -> None:
    """Six raw rows -> nine records in ``_SMF_ORDER``.

    The 26 C half loses the ``_sn1`` row (NaN fitness, dropna) and the duplicated
    ``_dma1`` row (drop_duplicates); the 30 C half loses only the duplicate. The
    two halves are concatenated 26 C first, which fixes every record index.
    """
    assert len(smf) == 9
    observed = [
        (
            smf[i]["experiment"]["genotype"]["perturbations"][0]["strain_id"],
            smf[i]["experiment"]["environment"]["temperature"]["value"],
        )
        for i in range(len(smf))
    ]
    assert observed == _SMF_ORDER


def test_smf_ts_allele_record_at_26c_matches_the_source_row(
    smf: c.SmfCostanzo2016Dataset,
) -> None:
    """Record 0 is ``YAL001C_tsa1`` at 26 C: fitness 0.85, bootstrap SE 0.02 used as-is.

    ``fitness_se == fitness_std == 0.02`` (bootstrap_se is never divided by sqrt n),
    ``n_samples`` is the 17 control screens, and the reference is fitness 1.0 with the
    26 C mean stddev 0.025 typed the same way. The whole experiment, reference, and
    publication are compared by ``model_dump`` equality.
    """
    phenotype = FitnessPhenotype(
        fitness=0.85,
        fitness_std=0.02,
        fitness_uncertainty=0.02,
        fitness_uncertainty_type=UncertaintyType.bootstrap_se,
        n_samples=17,
        sample_unit=SampleUnit.screen,
    )
    expected = FitnessExperiment(
        dataset_name="SmfCostanzo2016Dataset",
        genotype=Genotype(
            perturbations=[
                SgaTsAllelePerturbation(
                    systematic_gene_name="YAL001C",
                    perturbed_gene_name="tfc3-1",
                    strain_id="YAL001C_tsa1",
                )
            ]
        ),
        environment=_environment(26),
        phenotype=phenotype,
    )
    expected_reference = FitnessExperimentReference(
        dataset_name="SmfCostanzo2016Dataset",
        genome_reference=_GENOME,
        environment_reference=_environment(26),
        phenotype_reference=FitnessPhenotype(
            fitness=1.0,
            fitness_std=_SMF_REF_STD_26,
            fitness_uncertainty=_SMF_REF_STD_26,
            fitness_uncertainty_type=UncertaintyType.bootstrap_se,
            n_samples=17,
            sample_unit=SampleUnit.screen,
        ),
    )
    record = smf[0]
    assert record["experiment"]["phenotype"]["fitness_se"] == 0.02
    assert record["experiment"] == expected.model_dump()
    assert record["reference"] == expected_reference.model_dump()
    assert record["publication"] == _PUBLICATION.model_dump()
    assert c.N_SAMPLES_QUERY_SMF_SCREENS == 17


@pytest.mark.parametrize(
    ("index", "perturbation_type", "kind_field", "kind_value"),
    [
        (
            4,
            "temperature_sensitive_allele",
            "temperature_sensitive_allele_perturbation_type",
            "SGA",
        ),
        (5, "sga_kanmx_deletion", "kanmx_deletion_type", "SGA"),
        (6, "sga_natmx_deletion", "natmx_deletion_type", "SGA"),
        (7, "damp", "damp_perturbation_type", "SGA"),
        (8, "suppressor_allele", "suppressor_allele_perturbation_type", "SGA"),
    ],
)
def test_smf_strain_suffix_selects_the_sga_perturbation_class(
    smf: c.SmfCostanzo2016Dataset,
    index: int,
    perturbation_type: str,
    kind_field: str,
    kind_value: str,
) -> None:
    """The 30 C records 4..8 carry the five suffix-derived perturbation classes.

    ``tsa`` -> SgaTsAllelePerturbation, ``dma`` -> SgaKanMxDeletionPerturbation,
    ``sn`` -> SgaNatMxDeletionPerturbation, ``damp`` -> SgaDampPerturbation, ``S`` ->
    SgaSuppressorAllelePerturbation; each class's own ``*_type`` field is ``"SGA"``.
    """
    (perturbation,) = smf[index]["experiment"]["genotype"]["perturbations"]
    assert perturbation["perturbation_type"] == perturbation_type
    assert perturbation[kind_field] == kind_value
    assert perturbation["strain_id"] == _SMF_ORDER[index][0]


def test_smf_reference_std_is_the_mean_stddev_of_each_temperature(
    smf: c.SmfCostanzo2016Dataset,
) -> None:
    """26 C reference std 0.025 (mean of 0.02, 0.01, 0.03, 0.04); 30 C 0.04 (five rows).

    The reference SE equals the std (bootstrap_se, as-is) at both temperatures, and
    ``compute_phenotype_reference_std`` on the saved ``preprocess/data.csv`` returns the
    same two numbers.
    """
    ref_26 = smf[0]["reference"]["phenotype_reference"]
    ref_30 = smf[4]["reference"]["phenotype_reference"]
    assert ref_26["fitness_std"] == _SMF_REF_STD_26
    assert ref_26["fitness_se"] == ref_26["fitness_std"]
    assert ref_30["fitness_std"] == _SMF_REF_STD_30
    assert ref_30["fitness_se"] == ref_30["fitness_std"]
    df = smf.df
    assert df is not None
    assert df.shape == (9, 7)
    assert df.columns.tolist() == [
        "Strain ID",
        "Systematic gene name",
        "Allele/Gene name",
        "Single mutant fitness",
        "Single mutant fitness stddev",
        "perturbation_type",
        "Temperature",
    ]
    std_26, std_30 = c.SmfCostanzo2016Dataset.compute_phenotype_reference_std(df)
    assert std_26 == _SMF_REF_STD_26
    assert std_30 == _SMF_REF_STD_30


def test_smf_side_files_gene_set_reference_index_manifest_and_interning(
    smf: c.SmfCostanzo2016Dataset,
) -> None:
    """``preprocess/`` holds the four side files with the fixture's exact content.

    Gene set = the five systematic names sorted; two references (26 C members 0..3,
    30 C members 4..8, first-sighting order); manifest names this loader and host;
    ``processed/interned`` holds exactly 4 objects (2 environments + 2 references,
    the publication is under 512 bytes and stays inline).
    """
    assert sorted(os.listdir(smf.preprocess_dir)) == [
        "build_manifest.json",
        "data.csv",
        "experiment_reference_index.json",
        "gene_set.json",
    ]
    with open(osp.join(smf.preprocess_dir, "gene_set.json")) as f:
        assert json.load(f) == ["YAL001C", "YAL002W", "YAL003W", "YAL005C", "YAL007C"]
    assert sorted(smf.gene_set) == [
        "YAL001C",
        "YAL002W",
        "YAL003W",
        "YAL005C",
        "YAL007C",
    ]
    assert _reference_index(smf.preprocess_dir) == [
        ([0, 1, 2, 3], 26.0),
        ([4, 5, 6, 7, 8], 30.0),
    ]
    with open(osp.join(smf.preprocess_dir, "build_manifest.json")) as f:
        manifest = json.load(f)
    assert manifest["dataset_name"] == "smf"
    assert manifest["loader_class"] == "SmfCostanzo2016Dataset"
    assert manifest["loader_module"] == "torchcell.datasets.scerevisiae.costanzo2016"
    assert manifest["hostname"] == socket.gethostname()
    assert _interned_entries(smf.processed_dir) == 4


def test_smf_typed_round_trip_rebuilds_the_natmx_record(
    smf: c.SmfCostanzo2016Dataset,
) -> None:
    """``transform_item`` on record 6 (``YAL003W_sn1``, 30 C only) gives the typed objects.

    fitness 0.70, SE 0.03, reference std 0.04; the strain's 26 C row was NaN and never
    stored, so this is its only record.
    """
    typed = smf.transform_item(smf[6])
    expected = FitnessExperiment(
        dataset_name="SmfCostanzo2016Dataset",
        genotype=Genotype(
            perturbations=[
                SgaNatMxDeletionPerturbation(
                    systematic_gene_name="YAL003W",
                    perturbed_gene_name="EFB1",
                    strain_id="YAL003W_sn1",
                )
            ]
        ),
        environment=_environment(30),
        phenotype=FitnessPhenotype(
            fitness=0.70,
            fitness_std=0.03,
            fitness_uncertainty=0.03,
            fitness_uncertainty_type=UncertaintyType.bootstrap_se,
            n_samples=17,
            sample_unit=SampleUnit.screen,
        ),
    )
    assert typed["experiment"] == expected
    assert typed["reference"].phenotype_reference.fitness_std == 0.04
    assert typed["publication"] == _PUBLICATION


def test_smf_unknown_strain_suffix_is_typed_unknown_and_then_crashes_unbound() -> None:
    """Finding: an unrecognized suffix yields ``perturbation_type == "unknown"``, and
    ``create_experiment`` then raises ``UnboundLocalError`` because no branch binds
    ``genotype``. Pinned as the code behaves; a typed error would be the fix.
    """
    frame = pd.DataFrame(
        [("YAL001C_xyz1", "YAL001C", "TFC3", 0.9, 0.01, 0.8, 0.02)],
        columns=_SMF_COLUMNS,
    )
    loader = c.SmfCostanzo2016Dataset.__new__(c.SmfCostanzo2016Dataset)
    cleaned = loader.preprocess_raw(frame)
    assert cleaned["perturbation_type"].tolist() == ["unknown", "unknown"]
    assert cleaned["Temperature"].tolist() == [26, 30]
    with pytest.raises(UnboundLocalError):
        c.SmfCostanzo2016Dataset.create_experiment("x", cleaned.iloc[0], 0.1, 0.2)


# --------------------------------------------------------------------------- #
# DMF
# --------------------------------------------------------------------------- #
def test_dmf_records_follow_file_order_and_parse_temperature_from_arraytype(
    dmf: c.DmfCostanzo2016Dataset,
) -> None:
    """Five rows across the four files -> five records; ``DMA30``/``TSA26``/``DMA26`` ->
    temperatures [30, 26, 26, 30, 30]; each record has exactly the two source strains.
    """
    assert len(dmf) == 5
    temperatures = [
        dmf[i]["experiment"]["environment"]["temperature"]["value"] for i in range(5)
    ]
    assert temperatures == _SGA_TEMPERATURES
    pairs = [
        [
            (p["strain_id"], p["perturbation_type"])
            for p in dmf[i]["experiment"]["genotype"]["perturbations"]
        ]
        for i in range(5)
    ]
    assert pairs == [
        [("YAL001C_damp1", "damp"), ("YAL002W_dma1", "sga_kanmx_deletion")],
        [
            ("YAL003W_tsq1", "temperature_sensitive_allele"),
            ("YAL005C_tsa1", "temperature_sensitive_allele"),
        ],
        [
            ("YAL007C_tsq2", "temperature_sensitive_allele"),
            ("YAL008W_dma2", "sga_kanmx_deletion"),
        ],
        [("YAL009W_sn1", "sga_natmx_deletion"), ("YAL010C_dma3", "sga_kanmx_deletion")],
        [("YAL011W_sn2", "sga_natmx_deletion"), ("YAL012W_S1", "suppressor_allele")],
    ]


def test_dmf_damp_x_kanmx_record_matches_the_source_row(
    dmf: c.DmfCostanzo2016Dataset,
) -> None:
    """Record 0 (SGA_DAmP.txt): DMF 0.70 with colony sample SD 0.05 over n = 4 colonies.

    ``fitness_se = 0.05 / sqrt(4) = 0.025`` (sample_sd divides), the reference is 1.0
    with the 30 C mean DMF SD 0.08/3 typed sample_sd (SE 0.04/3); the query allele name
    and array allele name become the two ``perturbed_gene_name`` values.
    """
    expected = FitnessExperiment(
        dataset_name="DmfCostanzo2016Dataset",
        genotype=Genotype(
            perturbations=[
                SgaDampPerturbation(
                    systematic_gene_name="YAL001C",
                    perturbed_gene_name="tfc3-damp",
                    strain_id="YAL001C_damp1",
                ),
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name="YAL002W",
                    perturbed_gene_name="vps8",
                    strain_id="YAL002W_dma1",
                ),
            ]
        ),
        environment=_environment(30),
        phenotype=FitnessPhenotype(
            fitness=0.70,
            fitness_std=0.05,
            fitness_uncertainty=0.05,
            fitness_uncertainty_type=UncertaintyType.sample_sd,
            n_samples=4,
            sample_unit=SampleUnit.colony,
        ),
    )
    record = dmf[0]
    assert record["experiment"]["phenotype"]["fitness_se"] == 0.025
    assert record["experiment"] == expected.model_dump()
    assert record["publication"] == _PUBLICATION.model_dump()
    reference = record["reference"]
    assert reference["dataset_name"] == "DmfCostanzo2016Dataset"
    assert reference["genome_reference"] == _GENOME.model_dump()
    assert reference["environment_reference"] == _environment(30).model_dump()
    ref_phenotype = reference["phenotype_reference"]
    assert ref_phenotype["fitness"] == 1.0
    assert ref_phenotype["fitness_std"] == _DMF_REF_STD_30
    assert ref_phenotype["fitness_se"] == _DMF_REF_STD_30 / 2
    assert ref_phenotype["fitness_uncertainty_type"] == UncertaintyType.sample_sd
    assert ref_phenotype["n_samples"] == 4
    assert ref_phenotype["sample_unit"] == SampleUnit.colony
    assert c.N_SAMPLES_DOUBLE_MUTANT == 4


def test_dmf_26c_reference_std_is_the_mean_of_the_two_26c_rows(
    dmf: c.DmfCostanzo2016Dataset,
) -> None:
    """Records 1 and 2 are the 26 C rows: reference std (0.03 + 0.06) / 2 = 0.045, SE
    0.0225; record 2's own SE is 0.06 / 2 = 0.03 and its fitness 0.50.
    """
    for index in (1, 2):
        ref = dmf[index]["reference"]["phenotype_reference"]
        assert ref["fitness_std"] == _DMF_REF_STD_26
        assert ref["fitness_se"] == _DMF_REF_STD_26 / 2
    phenotype = dmf[2]["experiment"]["phenotype"]
    assert phenotype["fitness"] == 0.50
    assert phenotype["fitness_se"] == 0.03
    assert phenotype["fitness_std"] == 0.06


def test_dmf_side_files_gene_set_reference_index_and_interning(
    dmf: c.DmfCostanzo2016Dataset,
) -> None:
    """Gene set = the ten systematic names; references grouped by temperature in
    first-sighting order (30 C: [0, 3, 4], then 26 C: [1, 2]); 4 interned objects.
    """
    with open(osp.join(dmf.preprocess_dir, "gene_set.json")) as f:
        assert json.load(f) == _SGA_GENES
    assert _reference_index(dmf.preprocess_dir) == [([0, 3, 4], 30.0), ([1, 2], 26.0)]
    with open(osp.join(dmf.preprocess_dir, "build_manifest.json")) as f:
        assert json.load(f)["loader_class"] == "DmfCostanzo2016Dataset"
    assert _interned_entries(dmf.processed_dir) == 4
    df = dmf.df
    assert df is not None and df.shape == (5, 16)


def test_dmf_subset_n_samples_rows_with_seed_42(tmp_path: Path) -> None:
    """``subset_n=2`` keeps ``df.sample(n=2, random_state=42)`` = source rows 1 and 4,
    re-indexed to records 0 and 1 (``efb1-1`` x ``ssa1-1`` at 26 C, ``swc3`` x
    ``cys3-S1`` at 30 C).
    """
    root = tmp_path / "dmf_subset"
    _write_sga_raw(root, _SGA_FILES)
    ds = c.DmfCostanzo2016Dataset(
        root=str(root), subset_n=2, io_workers=1, batch_size=2
    )
    assert len(ds) == 2
    kept = [
        (
            [
                p["perturbed_gene_name"]
                for p in ds[i]["experiment"]["genotype"]["perturbations"]
            ],
            ds[i]["experiment"]["environment"]["temperature"]["value"],
        )
        for i in range(2)
    ]
    assert kept == [(["efb1-1", "ssa1-1"], 26.0), (["swc3", "cys3-S1"], 30.0)]


def test_dmf_unknown_array_suffix_silently_stores_a_single_perturbation() -> None:
    """Finding: an array strain with an unrecognized suffix (``YAL002W_xyz1``) is typed
    ``"unknown"`` and ``create_experiment`` appends nothing for it, so the DMF record
    carries ONE perturbation (the query) with no error. The DMI loader asserts on this;
    the DMF loader does not.
    """
    frame = pd.DataFrame(
        [
            (
                "YAL001C_dma1",
                "tfc3",
                "YAL002W_xyz1",
                "vps8",
                "DMA30",
                -0.1,
                0.01,
                0.9,
                0.9,
                0.8,
                0.05,
            )
        ],
        columns=_SGA_COLUMNS,
    )
    loader = c.DmfCostanzo2016Dataset.__new__(c.DmfCostanzo2016Dataset)
    cleaned = loader.preprocess_raw(frame)
    assert cleaned["array_perturbation_type"].tolist() == ["unknown"]
    assert cleaned["query_perturbation_type"].tolist() == ["KanMX_deletion"]
    assert loader.phenotype_reference_std_30 == 0.05
    assert loader.phenotype_reference_std_26 is None
    experiment, reference, _ = c.DmfCostanzo2016Dataset.create_experiment(
        "DmfCostanzo2016Dataset", cleaned.iloc[0], None, 0.05
    )
    genotype = experiment.genotype
    assert isinstance(genotype, Genotype)
    assert genotype.systematic_gene_names == ["YAL001C"]
    assert len(genotype) == 1
    assert reference.phenotype_reference.fitness_se == 0.025


# --------------------------------------------------------------------------- #
# DMI
# --------------------------------------------------------------------------- #
def test_dmi_damp_x_kanmx_record_matches_the_source_row(
    dmi: c.DmiCostanzo2016Dataset,
) -> None:
    """Record 0: epsilon -0.12, p 0.001 at graph_level ``edge``; the reference is
    interaction 0.0 with p None, same 30 C environment; publication PMID 27708008.
    """
    expected = GeneInteractionExperiment(
        dataset_name="DmiCostanzo2016Dataset",
        genotype=Genotype(
            perturbations=[
                SgaDampPerturbation(
                    systematic_gene_name="YAL001C",
                    perturbed_gene_name="tfc3-damp",
                    strain_id="YAL001C_damp1",
                ),
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name="YAL002W",
                    perturbed_gene_name="vps8",
                    strain_id="YAL002W_dma1",
                ),
            ]
        ),
        environment=_environment(30),
        phenotype=GeneInteractionPhenotype(
            gene_interaction=-0.12, gene_interaction_p_value=0.001, graph_level="edge"
        ),
    )
    expected_reference = GeneInteractionExperimentReference(
        dataset_name="DmiCostanzo2016Dataset",
        genome_reference=_GENOME,
        environment_reference=_environment(30),
        phenotype_reference=GeneInteractionPhenotype(
            gene_interaction=0.0, gene_interaction_p_value=None, graph_level="edge"
        ),
    )
    record = dmi[0]
    assert len(dmi) == 5
    assert record["experiment"] == expected.model_dump()
    assert record["reference"] == expected_reference.model_dump()
    assert record["publication"] == _PUBLICATION.model_dump()


def test_dmi_scores_pvalues_and_temperatures_follow_the_files(
    dmi: c.DmiCostanzo2016Dataset,
) -> None:
    """(epsilon, p, temperature) per record equals the five source rows in file order."""
    observed = [
        (
            dmi[i]["experiment"]["phenotype"]["gene_interaction"],
            dmi[i]["experiment"]["phenotype"]["gene_interaction_p_value"],
            dmi[i]["experiment"]["environment"]["temperature"]["value"],
        )
        for i in range(5)
    ]
    assert observed == [
        (-0.12, 0.001, 30.0),
        (0.08, 0.02, 26.0),
        (-0.30, 0.0001, 26.0),
        (0.02, 0.5, 30.0),
        (0.05, 0.3, 30.0),
    ]
    typed = dmi.transform_item(dmi[4])
    assert typed["experiment"].genotype.perturbed_gene_names == ["swc3", "cys3-S1"]
    assert typed["experiment"].genotype.perturbation_types == [
        "sga_natmx_deletion",
        "suppressor_allele",
    ]


def test_dmi_side_files_gene_set_reference_index_and_interning(
    dmi: c.DmiCostanzo2016Dataset,
) -> None:
    """Same ten genes and the same two temperature references as DMF ([0, 3, 4] at
    30 C, [1, 2] at 26 C); the manifest names the DMI loader; 4 interned objects.
    """
    with open(osp.join(dmi.preprocess_dir, "gene_set.json")) as f:
        assert json.load(f) == _SGA_GENES
    assert _reference_index(dmi.preprocess_dir) == [([0, 3, 4], 30.0), ([1, 2], 26.0)]
    with open(osp.join(dmi.preprocess_dir, "build_manifest.json")) as f:
        manifest = json.load(f)
    assert manifest["loader_class"] == "DmiCostanzo2016Dataset"
    assert manifest["dataset_name"] == "dmi"
    assert _interned_entries(dmi.processed_dir) == 4
    assert sorted(os.listdir(dmi.processed_dir)) == [
        "interned",
        "lmdb",
        "pre_filter.pt",
        "pre_transform.pt",
    ]


def test_dmi_unknown_array_suffix_fails_the_two_perturbation_assertion() -> None:
    """An unrecognized array suffix leaves one perturbation and ``create_experiment``
    raises ``AssertionError("Genotype must have 2 perturbations.")``; the same row
    passes through the DMF loader silently (see the DMF finding).
    """
    frame = pd.DataFrame(
        [
            (
                "YAL001C_dma1",
                "tfc3",
                "YAL002W_xyz1",
                "vps8",
                "DMA30",
                -0.1,
                0.01,
                0.9,
                0.9,
                0.8,
                0.05,
            )
        ],
        columns=_SGA_COLUMNS,
    )
    loader = c.DmiCostanzo2016Dataset.__new__(c.DmiCostanzo2016Dataset)
    cleaned = loader.preprocess_raw(frame)
    assert cleaned["array_perturbation_type"].tolist() == ["unknown"]
    with pytest.raises(AssertionError, match="Genotype must have 2 perturbations"):
        c.DmiCostanzo2016Dataset.create_experiment(
            "DmiCostanzo2016Dataset", cleaned.iloc[0]
        )


def test_dmi_nan_interaction_score_is_rejected_by_the_phenotype() -> None:
    """A NaN epsilon is refused at ``GeneInteractionPhenotype`` construction
    (``Gene interaction cannot be NaN``), so a NaN row cannot be stored.
    """
    frame = pd.DataFrame(
        [
            (
                "YAL001C_dma1",
                "tfc3",
                "YAL002W_dma2",
                "vps8",
                "DMA30",
                math.nan,
                0.01,
                0.9,
                0.9,
                0.8,
                0.05,
            )
        ],
        columns=_SGA_COLUMNS,
    )
    loader = c.DmiCostanzo2016Dataset.__new__(c.DmiCostanzo2016Dataset)
    cleaned = loader.preprocess_raw(frame)
    with pytest.raises(ValueError, match="Gene interaction cannot be NaN"):
        c.DmiCostanzo2016Dataset.create_experiment(
            "DmiCostanzo2016Dataset", cleaned.iloc[0]
        )


# --------------------------------------------------------------------------- #
# Phase 14: the #410 twins, the remaining strain classes, refusals, download, main
# --------------------------------------------------------------------------- #
_SMF_TWIN_ROWS: list[tuple[Any, ...]] = [
    ("YAL002W_dma1", "YAL002W", "VPS8", 0.95, 0.01, 0.95, 0.01),
    ("YAL005C_damp1", "YAL005C", "ssa1-damp", 0.80, 0.03, 0.80, 0.03),
    ("YAL001C_tsa1", "YAL001C", "tfc3-1", 0.85, 0.02, 0.60, 0.04),
    ("YAL017W_dma5", "YAL017W", "YAL017W", 0.90, None, 0.90, 0.01),
]


@pytest.fixture(scope="module")
def smf_twins(tmp_path_factory: pytest.TempPathFactory) -> c.SmfCostanzo2016Dataset:
    root = tmp_path_factory.mktemp("costanzo2016") / "smf_twins"
    (root / "raw").mkdir(parents=True)
    pd.DataFrame(_SMF_TWIN_ROWS, columns=_SMF_COLUMNS).to_excel(
        root / "raw" / "strain_ids_and_single_mutant_fitness.xlsx", index=False
    )
    return c.SmfCostanzo2016Dataset(root=str(root))


def _smf_experiment(
    perturbation: Any, temperature: int, fitness: float, std: float
) -> FitnessExperiment:
    return FitnessExperiment(
        dataset_name="SmfCostanzo2016Dataset",
        genotype=Genotype(perturbations=[perturbation]),
        environment=_environment(temperature),
        phenotype=FitnessPhenotype(
            fitness=fitness,
            fitness_std=std,
            fitness_uncertainty=std,
            fitness_uncertainty_type=UncertaintyType.bootstrap_se,
            n_samples=17,
            sample_unit=SampleUnit.screen,
        ),
    )


def _smf_reference(temperature: int, std: float) -> FitnessExperimentReference:
    return FitnessExperimentReference(
        dataset_name="SmfCostanzo2016Dataset",
        genome_reference=_GENOME,
        environment_reference=_environment(temperature),
        phenotype_reference=FitnessPhenotype(
            fitness=1.0,
            fitness_std=std,
            fitness_uncertainty=std,
            fitness_uncertainty_type=UncertaintyType.bootstrap_se,
            n_samples=17,
            sample_unit=SampleUnit.screen,
        ),
    )


def _dma1() -> SgaKanMxDeletionPerturbation:
    return SgaKanMxDeletionPerturbation(
        systematic_gene_name="YAL002W",
        perturbed_gene_name="VPS8",
        strain_id="YAL002W_dma1",
    )


def test_smf_deletion_strain_is_emitted_as_26c_and_30c_twins_issue_410(
    smf_twins: c.SmfCostanzo2016Dataset,
) -> None:
    """Finding (issue #410): the released file repeats one temperature-combined value
    for a deletion strain in both columns, and ``preprocess_raw`` emits a 26 C and a 30 C
    record from it (lines 223-266). Records 0 and 3 are whole-equal to the hand-built
    26 C and 30 C experiments, which differ only in ``temperature``; each points at its
    own temperature's reference (noise 0.02 and 0.0225). Pinned until #410 stores one
    record with a temperature-combined environment.
    """
    assert (
        smf_twins[0]["experiment"]
        == _smf_experiment(_dma1(), 26, 0.95, 0.01).model_dump()
    )
    assert (
        smf_twins[3]["experiment"]
        == _smf_experiment(_dma1(), 30, 0.95, 0.01).model_dump()
    )
    assert smf_twins[0]["reference"] == _smf_reference(26, 0.02).model_dump()
    assert smf_twins[3]["reference"] == _smf_reference(30, 0.0225).model_dump()
    twin_26 = dict(smf_twins[0]["experiment"], environment=None)
    twin_30 = dict(smf_twins[3]["experiment"], environment=None)
    assert twin_26 == twin_30


def test_smf_damp_twins_and_the_genuine_ts_pair(
    smf_twins: c.SmfCostanzo2016Dataset,
) -> None:
    """The DAmP strain gets an ``SgaDampPerturbation`` twin pair (records 1 and 4, 0.80
    both), while the TS allele carries two different measurements (0.85 at 26 C, 0.60
    at 30 C), the case #410 says the loader handles correctly.
    """
    damp = SgaDampPerturbation(
        systematic_gene_name="YAL005C",
        perturbed_gene_name="ssa1-damp",
        strain_id="YAL005C_damp1",
    )
    assert (
        smf_twins[1]["experiment"] == _smf_experiment(damp, 26, 0.80, 0.03).model_dump()
    )
    assert (
        smf_twins[4]["experiment"] == _smf_experiment(damp, 30, 0.80, 0.03).model_dump()
    )
    fitness = [
        (
            smf_twins[i]["experiment"]["genotype"]["perturbations"][0]["strain_id"],
            smf_twins[i]["experiment"]["environment"]["temperature"]["value"],
            smf_twins[i]["experiment"]["phenotype"]["fitness"],
        )
        for i in range(len(smf_twins))
    ]
    assert fitness == [
        ("YAL002W_dma1", 26.0, 0.95),
        ("YAL005C_damp1", 26.0, 0.80),
        ("YAL001C_tsa1", 26.0, 0.85),
        ("YAL002W_dma1", 30.0, 0.95),
        ("YAL005C_damp1", 30.0, 0.80),
        ("YAL001C_tsa1", 30.0, 0.60),
        ("YAL017W_dma5", 30.0, 0.90),
    ]


def test_smf_blank_stddev_drops_a_measured_fitness(
    smf_twins: c.SmfCostanzo2016Dataset,
) -> None:
    """Finding: ``YAL017W_dma5`` has a 26 C fitness of 0.90 and a blank stddev; the
    ``dropna`` over every column (line 267) drops that measurement, so the strain has a
    30 C record only. Pinned until a blank stddev is stored as an uncertainty gap.
    """
    assert _reference_index(smf_twins.preprocess_dir) == [
        ([0, 1, 2], 26.0),
        ([3, 4, 5, 6], 30.0),
    ]
    with open(osp.join(smf_twins.preprocess_dir, "data.csv")) as f:
        rows = f.read().splitlines()
    assert [r.split(",")[0] for r in rows[1:]] == [
        "YAL002W_dma1",
        "YAL005C_damp1",
        "YAL001C_tsa1",
        "YAL002W_dma1",
        "YAL005C_damp1",
        "YAL001C_tsa1",
        "YAL017W_dma5",
    ]


def test_smf_temperature_other_than_26_or_30_is_unbound() -> None:
    """Finding: a row at 22 C binds no reference SD (lines 377-381) and raises
    ``UnboundLocalError`` naming ``phenotype_reference_std``. Pinned until an unknown
    temperature is refused with a named error.
    """
    row = pd.Series(
        {
            "Strain ID": "YAL002W_dma1",
            "Systematic gene name": "YAL002W",
            "Allele/Gene name": "VPS8",
            "Single mutant fitness": 0.95,
            "Single mutant fitness stddev": 0.01,
            "perturbation_type": "KanMX_deletion",
            "Temperature": 22,
        }
    )
    with pytest.raises(UnboundLocalError) as excinfo:
        c.SmfCostanzo2016Dataset.create_experiment("x", row, 0.02, 0.0225)
    assert str(excinfo.value) == (
        "cannot access local variable 'phenotype_reference_std' where it is not "
        "associated with a value"
    )


_SGA_EMPTY: dict[str, list[tuple[Any, ...]]] = {
    "SGA_DAmP.txt": [],
    "SGA_ExE.txt": [],
    "SGA_ExN_NxE.txt": [],
    "SGA_NxN.txt": [],
}
_SUPPRESSOR_X_DAMP = (
    "YAL013W_S2",
    "erg-S2",
    "YAL014C_damp2",
    "ssa-damp",
    "DMA30",
    0.1,
    0.2,
    0.9,
    0.8,
    0.7,
    None,
)
_KANMX_X_NATMX = (
    "YAL015C_dma4",
    "cdc-del",
    "YAL016W_sn3",
    "tps-del",
    "DMA30",
    -0.2,
    0.01,
    0.9,
    0.8,
    0.6,
    0.05,
)
_SGA_EDGE_FILES = dict(
    _SGA_EMPTY, **{"SGA_ExE.txt": [_SUPPRESSOR_X_DAMP], "SGA_NxN.txt": [_KANMX_X_NATMX]}
)


def _edge_pair() -> tuple[Genotype, Genotype]:
    return (
        Genotype(
            perturbations=[
                SgaSuppressorAllelePerturbation(
                    systematic_gene_name="YAL013W",
                    perturbed_gene_name="erg-S2",
                    strain_id="YAL013W_S2",
                ),
                SgaDampPerturbation(
                    systematic_gene_name="YAL014C",
                    perturbed_gene_name="ssa-damp",
                    strain_id="YAL014C_damp2",
                ),
            ]
        ),
        Genotype(
            perturbations=[
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name="YAL015C",
                    perturbed_gene_name="cdc-del",
                    strain_id="YAL015C_dma4",
                ),
                SgaNatMxDeletionPerturbation(
                    systematic_gene_name="YAL016W",
                    perturbed_gene_name="tps-del",
                    strain_id="YAL016W_sn3",
                ),
            ]
        ),
    )


def _dmf_phenotype(fitness: float, std: float) -> FitnessPhenotype:
    return FitnessPhenotype(
        fitness=fitness,
        fitness_std=std,
        fitness_uncertainty=std,
        fitness_uncertainty_type=UncertaintyType.sample_sd,
        n_samples=4,
        sample_unit=SampleUnit.colony,
    )


def test_dmf_suppressor_damp_and_natmx_classes_and_a_blank_sd(tmp_path: Path) -> None:
    """Finding: the blank DMF SD of record 0 passes the ``is not None`` guard as NaN
    (line 730), so it is stored as a NaN SE typed ``sample_sd`` rather than as a gap.
    Record 1 (KanMX x NatMX, 0.60 +- 0.05) is compared whole; both share the 30 C
    reference whose SD is the NaN-skipping mean 0.05 (SE 0.025). Pinned until a blank SD
    is stored as ``None`` with no uncertainty type.
    """
    root = tmp_path / "dmf_edge"
    _write_sga_raw(root, _SGA_EDGE_FILES)
    ds = c.DmfCostanzo2016Dataset(root=str(root), io_workers=1, batch_size=2)
    assert len(ds) == 2
    first, second = _edge_pair()
    blank = ds[0]["experiment"]
    assert blank["genotype"] == first.model_dump()
    phenotype = blank["phenotype"]
    assert phenotype["fitness"] == 0.7
    assert phenotype["fitness_uncertainty_type"] == UncertaintyType.sample_sd
    assert math.isnan(phenotype["fitness_std"])
    assert math.isnan(phenotype["fitness_se"])
    assert math.isnan(phenotype["fitness_uncertainty"])
    expected = FitnessExperiment(
        dataset_name="DmfCostanzo2016Dataset",
        genotype=second,
        environment=_environment(30),
        phenotype=_dmf_phenotype(0.6, 0.05),
    )
    assert ds[1]["experiment"] == expected.model_dump()
    reference = FitnessExperimentReference(
        dataset_name="DmfCostanzo2016Dataset",
        genome_reference=_GENOME,
        environment_reference=_environment(30),
        phenotype_reference=_dmf_phenotype(1.0, 0.05),
    ).model_dump()
    assert ds[0]["reference"] == reference
    assert ds[1]["reference"] == reference
    assert reference["phenotype_reference"]["fitness_se"] == 0.025


def test_dmi_suppressor_damp_and_natmx_classes(tmp_path: Path) -> None:
    """The same two rows through DMI: epsilon 0.1 / p 0.2 and -0.2 / 0.01 at 30 C, the
    edge-level reference at 0.0; the blank DMF SD is not read by DMI at all.
    """
    root = tmp_path / "dmi_edge"
    _write_sga_raw(root, _SGA_EDGE_FILES)
    ds = c.DmiCostanzo2016Dataset(root=str(root), io_workers=1, batch_size=2)
    first, second = _edge_pair()
    for index, genotype, score, p_value in (
        (0, first, 0.1, 0.2),
        (1, second, -0.2, 0.01),
    ):
        expected = GeneInteractionExperiment(
            dataset_name="DmiCostanzo2016Dataset",
            genotype=genotype,
            environment=_environment(30),
            phenotype=GeneInteractionPhenotype(
                gene_interaction=score,
                gene_interaction_p_value=p_value,
                graph_level="edge",
            ),
        )
        assert ds[index]["experiment"] == expected.model_dump()
    assert (
        ds[1]["reference"]
        == GeneInteractionExperimentReference(
            dataset_name="DmiCostanzo2016Dataset",
            genome_reference=_GENOME,
            environment_reference=_environment(30),
            phenotype_reference=GeneInteractionPhenotype(
                gene_interaction=0.0, gene_interaction_p_value=None, graph_level="edge"
            ),
        ).model_dump()
    )


_TSA22 = (
    "YAL015C_tsa4",
    "cdc-ts",
    "YAL016W_dma3",
    "tps-del",
    "TSA22",
    -0.2,
    0.01,
    0.9,
    0.8,
    0.5,
    0.05,
)


def test_dmf_row_at_22c_crashes_unbound_while_dmi_stores_it(tmp_path: Path) -> None:
    """Finding: ``TSA22`` parses to 22 C; DMF has a reference SD only for 26 and 30 C
    (lines 742-746) and the build raises ``UnboundLocalError`` through the thread pool,
    while DMI, which carries no reference SD, stores the row at 22 C. Pinned until DMF
    refuses an unknown temperature by name or computes its reference SD.
    """
    files = dict(_SGA_EMPTY, **{"SGA_ExE.txt": [_TSA22]})
    _write_sga_raw(tmp_path / "dmf22", files)
    with pytest.raises(UnboundLocalError) as excinfo:
        c.DmfCostanzo2016Dataset(root=str(tmp_path / "dmf22"), io_workers=1)
    assert "'phenotype_reference_std'" in str(excinfo.value)
    _write_sga_raw(tmp_path / "dmi22", files)
    dmi22 = c.DmiCostanzo2016Dataset(root=str(tmp_path / "dmi22"), io_workers=1)
    assert len(dmi22) == 1
    assert dmi22[0]["experiment"]["environment"] == _environment(22).model_dump()


def test_dmf_blank_fitness_refuses_the_whole_build(tmp_path: Path) -> None:
    """A blank ``Double mutant fitness`` is refused at ``FitnessPhenotype`` with
    ``Fitness cannot be NaN``; DMF has no ``dropna``, so one blank cell fails the build.
    """
    row = _KANMX_X_NATMX[:9] + (None, 0.05)
    _write_sga_raw(tmp_path / "dmf_nan", dict(_SGA_EMPTY, **{"SGA_NxN.txt": [row]}))
    with pytest.raises(ValueError, match="Fitness cannot be NaN"):
        c.DmfCostanzo2016Dataset(root=str(tmp_path / "dmf_nan"), io_workers=1)


def test_dmi_subset_n_samples_the_same_rows_as_dmf(tmp_path: Path) -> None:
    """``subset_n=2`` with seed 42 keeps source rows 1 and 4, as for DMF: epsilon 0.08
    (26 C) then 0.05 (30 C).
    """
    root = tmp_path / "dmi_subset"
    _write_sga_raw(root, _SGA_FILES)
    ds = c.DmiCostanzo2016Dataset(root=str(root), subset_n=2, io_workers=1)
    kept = [
        (
            ds[i]["experiment"]["phenotype"]["gene_interaction"],
            ds[i]["experiment"]["environment"]["temperature"]["value"],
        )
        for i in range(len(ds))
    ]
    assert kept == [(0.08, 26.0), (0.05, 30.0)]


_ARCHIVE_DIR = (
    "Data File S1. Raw genetic interaction datasets: Pair-wise interaction format"
)
_ARCHIVE_FILES = [
    "strain_ids_and_single_mutant_fitness.xlsx",
    "SGA_DAmP.txt",
    "SGA_ExE.txt",
    "SGA_ExN_NxE.txt",
    "SGA_NxN.txt",
    "readme.txt",
]


def _fake_download_url(
    monkeypatch: pytest.MonkeyPatch, calls: list[tuple[str, str]]
) -> None:
    """``download_url`` writes the released zip layout into ``folder`` and returns it."""

    def download_url(url: str, folder: str) -> str:
        calls.append((url, folder))
        os.makedirs(folder, exist_ok=True)
        path = osp.join(folder, "archive.zip")
        with zipfile.ZipFile(path, "w") as archive:
            for name in _ARCHIVE_FILES:
                archive.writestr(f"{_ARCHIVE_DIR}/{name}", name)
        return path

    monkeypatch.setattr(c, "download_url", download_url)


@pytest.mark.parametrize(
    ("cls", "kept"),
    [
        (c.SmfCostanzo2016Dataset, ["strain_ids_and_single_mutant_fitness.xlsx"]),
        (
            c.DmfCostanzo2016Dataset,
            [
                "SGA_DAmP.txt",
                "SGA_ExE.txt",
                "SGA_ExN_NxE.txt",
                "SGA_NxN.txt",
                "readme.txt",
            ],
        ),
        (
            c.DmiCostanzo2016Dataset,
            [
                "SGA_DAmP.txt",
                "SGA_ExE.txt",
                "SGA_ExN_NxE.txt",
                "SGA_NxN.txt",
                "readme.txt",
            ],
        ),
    ],
)
def test_download_unpacks_the_archive_and_keeps_the_loader_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cls: type[ExperimentDataset],
    kept: list[str],
) -> None:
    """SMF removes every ``.txt`` (the SGA files included); DMF and DMI remove the
    spreadsheet but keep any other text file (``readme.txt``); the zip and the
    subdirectory are gone. The archive URL is the one CellMap URL all three share.
    """
    calls: list[tuple[str, str]] = []
    _fake_download_url(monkeypatch, calls)
    dataset = cls.__new__(cls)
    dataset.root = str(tmp_path)
    dataset.download()
    raw = str(tmp_path / "raw")
    assert calls == [
        (
            "https://thecellmap.org/costanzo2016/data_files/"
            "Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip",
            raw,
        )
    ]
    assert sorted(os.listdir(raw)) == kept
    assert (tmp_path / "raw" / kept[0]).read_text() == kept[0]


def test_dmf_and_dmi_items_retype_through_their_own_schema_classes(
    dmf: c.DmfCostanzo2016Dataset, dmi: c.DmiCostanzo2016Dataset
) -> None:
    """``transform_item`` rebuilds a stored item through the dataset's declared classes
    (``experiment_dataset.py`` lines 638 to 641): a DMF item comes back as a
    ``FitnessExperiment`` and a DMI item as a ``GeneInteractionExperiment``, each dumping
    to exactly the stored dictionary, so a class wired to the wrong schema would fail
    here on the first record.
    """
    for dataset, experiment_class, reference_class in (
        (dmf, FitnessExperiment, FitnessExperimentReference),
        (dmi, GeneInteractionExperiment, GeneInteractionExperimentReference),
    ):
        item = dataset[0]
        typed = dataset.transform_item(item)
        assert type(typed["experiment"]) is experiment_class
        assert type(typed["reference"]) is reference_class
        assert typed["experiment"].model_dump() == item["experiment"]
        assert typed["reference"].model_dump() == item["reference"]


def test_main_builds_the_seven_datasets_under_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main`` constructs SMF, three DMF and three DMI datasets with these exact roots
    and arguments (the 1e5 and 5e5 subsets), printing each length and one record; the
    classes are replaced by recorders so nothing is built.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    calls: list[tuple[str, dict[str, Any]]] = []

    def recorder(name: str) -> type[Any]:
        class _Recorder:
            def __init__(self, **kwargs: Any) -> None:
                calls.append((name, kwargs))

            def __len__(self) -> int:
                return 7

            def __getitem__(self, index: int) -> str:
                return f"{name}[{index}]"

        return _Recorder

    for name in (
        "SmfCostanzo2016Dataset",
        "DmfCostanzo2016Dataset",
        "DmiCostanzo2016Dataset",
    ):
        monkeypatch.setattr(c, name, recorder(name))
    c.main()
    base = osp.join(str(tmp_path), "data/torchcell")
    assert calls == [
        (
            "SmfCostanzo2016Dataset",
            {"root": f"{base}/smf_costanzo2016", "io_workers": 10},
        ),
        (
            "DmfCostanzo2016Dataset",
            {
                "root": f"{base}/dmf_costanzo2016_1e5",
                "io_workers": 10,
                "batch_size": 10000,
                "subset_n": 100000,
            },
        ),
        (
            "DmfCostanzo2016Dataset",
            {
                "root": f"{base}/dmf_costanzo2016_5e5",
                "io_workers": 10,
                "batch_size": 10000,
                "subset_n": 500000,
            },
        ),
        (
            "DmfCostanzo2016Dataset",
            {"root": f"{base}/dmf_costanzo2016", "io_workers": 10, "batch_size": 10000},
        ),
        (
            "DmiCostanzo2016Dataset",
            {
                "root": f"{base}/dmi_costanzo2016_1e5",
                "io_workers": 10,
                "subset_n": 100000,
            },
        ),
        (
            "DmiCostanzo2016Dataset",
            {
                "root": f"{base}/dmi_costanzo2016_5e5",
                "io_workers": 10,
                "subset_n": 500000,
            },
        ),
        (
            "DmiCostanzo2016Dataset",
            {"root": f"{base}/dmi_costanzo2016", "io_workers": 10},
        ),
    ]
    assert capsys.readouterr().out.splitlines() == [
        "7",
        "SmfCostanzo2016Dataset[100]",
        *["7", "DmfCostanzo2016Dataset[0]"] * 3,
        *["7", "DmiCostanzo2016Dataset[0]"] * 3,
    ]
