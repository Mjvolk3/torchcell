# tests/torchcell/datasets/ecoli/test_schmidt2016_growth_rate.py
# [[tests.torchcell.datasets.ecoli.test_schmidt2016_growth_rate]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_schmidt2016_growth_rate.py
"""Schmidt 2016 Table S24 loader (``torchcell.datasets.ecoli.schmidt2016_growth_rate``).

The synthetic tests write a workbook with Table S24's exact two-block layout (a
``<medium>:`` label row carrying the five statistic headers, then one row per strain) and
a Table S23 sheet carrying the per-condition growth rates the medium-header check reads.
They drive the readers, both released-statistics checks, the phenotype and genotype
builders and a full ``process()`` into ``tmp_path``; the genome and the locus-tag
reconciliation are in-test objects, so nothing reads ``$DATA_ROOT``.

The ``@pytest.mark.data`` tests read the real mirror and pin the numbers the dendron note
states: eight released cells, the two cells that release two replicates rather than
three, that the released ``Stdev`` is a sample and not a population standard deviation,
that every fitness ratio is strictly positive so ``FitnessPhenotype``'s clamp never
fires, and that Table S24's medium headers agree with Table S23's own growth rates.
"""

from __future__ import annotations

import json
import math
import os
import os.path as osp
import statistics
from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd
import pytest

import torchcell.datasets.ecoli.schmidt2016 as sm
import torchcell.datasets.ecoli.schmidt2016_growth_rate as gr
from torchcell.data import file_sha256
from torchcell.datamodels.schema import (
    AssemblyReferenceGenome,
    BacterialAssemblySet,
    BacterialGeneNamespace,
    EnvironmentPhysicalPerturbation,
    SampleUnit,
    UncertaintyType,
)
from torchcell.datasets.bacteria_common import LocusTagReconciliation
from torchcell.sequence.genome.base import GeneNameResolution, GeneNameStatus

ASSEMBLY_SET: BacterialAssemblySet = "ecoli_K12_BW25113_ASM75055v1"
NAMESPACE: BacterialGeneNamespace = "ecoli_k12_bw25113_locus_tag"
#: ``{medium: {strain: replicate growth rates}}``. The wild type of each medium releases
#: three replicates in glucose and two in acetate, so BOTH replicate counts are built.
RATES: dict[str, dict[str, tuple[float, ...]]] = {
    "Glucose": {
        "WT": (0.60, 0.58, 0.59),
        "ΔrimI": (0.54, 0.55, 0.56),
        "ΔrimJ": (0.35, 0.37, 0.36),
        "ΔrimL": (0.53, 0.52),
    },
    "Acetate": {
        "WT": (0.32, 0.30),
        "ΔrimI": (0.36, 0.32, 0.26),
        "ΔrimJ": (0.15, 0.13, 0.14),
        "ΔrimL": (0.22, 0.21, 0.20),
    },
}
#: Table S23's own BW25113 rate per condition, near each block's wild-type mean.
S23_RATES = {"Glucose": 0.58, "Acetate": 0.30}


def write_workbook(path: Path) -> Path:
    """Write a workbook with Table S24's block layout and a Table S23 sheet."""
    book = openpyxl.Workbook()
    _write_s24(book.active)
    _write_s23(book.create_sheet(gr.SHEET_S23))
    book.save(path)
    return path


def _write_s24(sheet: Any) -> None:
    sheet.title = gr.SHEET_S24
    sheet.append([gr._Q_TABLE_S24])
    sheet.append([None] * 6)
    for medium, strains in RATES.items():
        sheet.append([None, "Growth rate (h-1)", None, None, None, None])
        sheet.append(
            [f"{medium}:", *gr.REPLICATE_HEADERS, gr.AVERAGE_HEADER, gr.STDEV_HEADER]
        )
        for strain, replicates in strains.items():
            padded = [*replicates, *([None] * (3 - len(replicates)))]
            sheet.append(
                [
                    strain,
                    *padded,
                    statistics.fmean(replicates),
                    statistics.stdev(replicates),
                ]
            )
        sheet.append([None] * 6)


def _write_s23(sheet: Any) -> None:
    sheet.append([gr._Q_TABLE_S23])
    sheet.append([None] * 4)
    sheet.append([gr.S23_CONDITION, gr.S23_STRAIN, gr.S23_RATE, "Stdev"])
    for condition, rate in S23_RATES.items():
        sheet.append([condition, sm.SCHMIDT_REFERENCE_STRAIN, rate, 0.01])
    # A non-BW25113 row and a non-numeric rate, both of which the reader skips.
    sheet.append(["Glucose", "MG1665", 0.67, 0.07])
    sheet.append(["Stationary phase 1 day", sm.SCHMIDT_REFERENCE_STRAIN, "-", "-"])


@pytest.fixture
def workbook(tmp_path: Path) -> Path:
    """The synthetic workbook, written once per test."""
    return write_workbook(tmp_path / sm.SI2)


# --------------------------------------------------------------------------- #
# The declared module constants
# --------------------------------------------------------------------------- #
def test_declared_shape_is_three_deletions_in_two_media() -> None:
    assert gr.DELETION_LABELS == ("ΔrimI", "ΔrimJ", "ΔrimL")
    assert [medium for medium, _ in gr.MEDIUM_BLOCKS] == ["Glucose", "Acetate"]
    assert gr.EXPECTED_RECORDS == len(gr.DELETION_LABELS) * len(gr.MEDIUM_BLOCKS)
    assert gr.EXPECTED_REFERENCES == len(gr.MEDIUM_BLOCKS)
    # Both media are conditions the proteome loader already sources a medium for.
    for _medium, column in gr.MEDIUM_BLOCKS:
        assert sm.CONDITIONS_BY_COLUMN[column].drop is None


def test_sourced_values_quote_the_pinned_artifacts() -> None:
    assert gr.SOURCED_VALUES["deletion_collection"].value == gr.KEIO_COLLECTION
    assert gr.SOURCED_VALUES["sample_unit"].value == "biological_replicate"
    assert gr.SOURCED_VALUES["reference_strain"].value == "BW25113"
    assert {v.provenance.sha256 for v in gr.SOURCED_VALUES.values()} == {
        sm.PAPER_MD_SHA256,
        sm.SI2_SHA256,
    }
    assert gr.PAPER_QUOTES["keio_strains"].startswith("Mutant strains with either")
    assert "KEIO collection" in gr.PAPER_QUOTES["keio_strains"]
    assert {gap.field for gap in gr.PERTURBATION_GAPS} == {"cassette", "construction"}


# --------------------------------------------------------------------------- #
# Reading Table S24
# --------------------------------------------------------------------------- #
def test_read_table_s24_parses_both_blocks_and_their_replicate_counts(
    workbook: Path,
) -> None:
    blocks = gr.read_table_s24(str(workbook))
    assert [block.medium for block in blocks] == ["Glucose", "Acetate"]
    for block in blocks:
        assert [row.strain for row in block.rows] == [
            gr.WILD_TYPE_LABEL,
            *gr.DELETION_LABELS,
        ]
        for row in block.rows:
            assert row.replicates == RATES[block.medium][row.strain]
            assert row.n_samples == len(RATES[block.medium][row.strain])
            assert row.mean == pytest.approx(row.average_released)
            assert row.sample_sd == pytest.approx(row.stdev_released)
            assert row.standard_error == pytest.approx(
                row.sample_sd / math.sqrt(row.n_samples)
            )
    glucose = blocks[0]
    assert glucose.deletions[2].n_samples == 2
    assert [row.gene_symbol for row in glucose.deletions] == ["rimI", "rimJ", "rimL"]
    with pytest.raises(RuntimeError, match="not a deletion strain label"):
        _ = glucose.wild_type.gene_symbol


def test_read_table_s24_refuses_a_renamed_statistic_header(tmp_path: Path) -> None:
    path = write_workbook(tmp_path / sm.SI2)
    book = openpyxl.load_workbook(path)
    book[gr.SHEET_S24].cell(row=4, column=6, value="SD")
    book.save(path)
    with pytest.raises(RuntimeError, match="block is headed"):
        gr.read_table_s24(str(path))


def test_read_table_s24_refuses_a_missing_strain_row(tmp_path: Path) -> None:
    path = write_workbook(tmp_path / sm.SI2)
    book = openpyxl.load_workbook(path)
    book[gr.SHEET_S24].cell(row=7, column=1, value="ΔrimX")
    book.save(path)
    with pytest.raises(RuntimeError, match="releases no"):
        gr.read_table_s24(str(path))


def test_read_table_s24_refuses_a_single_replicate(tmp_path: Path) -> None:
    path = write_workbook(tmp_path / sm.SI2)
    book = openpyxl.load_workbook(path)
    sheet = book[gr.SHEET_S24]
    # openpyxl's cell(..., value=None) means "no value given", so the value is cleared
    # through the cell object instead.
    sheet.cell(row=5, column=3).value = None
    sheet.cell(row=5, column=4).value = None
    book.save(path)
    with pytest.raises(RuntimeError, match="needs at least two"):
        gr.read_table_s24(str(path))


def test_read_table_s24_refuses_a_missing_medium_block(tmp_path: Path) -> None:
    path = write_workbook(tmp_path / sm.SI2)
    book = openpyxl.load_workbook(path)
    book[gr.SHEET_S24].cell(row=4, column=1, value="Glycerol:")
    book.save(path)
    with pytest.raises(RuntimeError, match="carries the medium blocks"):
        gr.read_table_s24(str(path))


def test_read_table_s23_keeps_only_numeric_bw25113_rates(workbook: Path) -> None:
    rates = gr.read_table_s23_wild_type_rates(str(workbook))
    assert rates == S23_RATES


# --------------------------------------------------------------------------- #
# The released statistics, and the medium headers
# --------------------------------------------------------------------------- #
def test_released_statistics_back_solve_the_sample_standard_deviation(
    workbook: Path,
) -> None:
    blocks = gr.read_table_s24(str(workbook))
    summary = gr.check_released_statistics(blocks)
    assert summary["n_rows"] == 8
    assert summary["uncertainty_type"] == UncertaintyType.sample_sd.value
    assert summary["worst_average_abs_dev"] < gr.IDENTITY_RTOL
    assert summary["worst_sample_sd_abs_dev"] < gr.IDENTITY_RTOL
    # The population SD is a DIFFERENT number, which is what identifies the n-1 form.
    assert summary["nearest_population_sd_abs_dev"] > gr.IDENTITY_RTOL
    assert summary["n_samples_by_row"]["Glucose/ΔrimL"] == 2
    assert summary["n_samples_by_row"]["Acetate/WT"] == 2


def test_released_statistics_catch_a_drifted_average_and_a_population_sd(
    workbook: Path,
) -> None:
    blocks = gr.read_table_s24(str(workbook))
    drifted = blocks[0].wild_type.model_copy(update={"average_released": 1.0})
    broken = (blocks[0].model_copy(update={"wild_type": drifted}), blocks[1])
    with pytest.raises(RuntimeError, match="released Average"):
        gr.check_released_statistics(broken)

    population = tuple(
        block.model_copy(
            update={
                "wild_type": block.wild_type.model_copy(
                    update={
                        "stdev_released": statistics.pstdev(block.wild_type.replicates)
                    }
                )
            }
        )
        for block in blocks
    )
    with pytest.raises(RuntimeError, match="not the sample standard deviation"):
        gr.check_released_statistics(population)


def test_medium_headers_agree_with_table_s23_and_a_swap_is_refused(
    workbook: Path,
) -> None:
    blocks = gr.read_table_s24(str(workbook))
    summary = gr.check_medium_headers(blocks, S23_RATES)
    assert summary["swapped"] is False
    assert summary["table_s23_bw25113_rates"] == S23_RATES
    assert summary["relative_distances"]["Glucose"]["Glucose"] < gr.MEDIUM_HEADER_RTOL
    assert summary["relative_distances"]["Glucose"]["Acetate"] > gr.MEDIUM_HEADER_RTOL

    swapped = {"Glucose": S23_RATES["Acetate"], "Acetate": S23_RATES["Glucose"]}
    with pytest.raises(RuntimeError, match="no longer agree"):
        gr.check_medium_headers(blocks, swapped)


# --------------------------------------------------------------------------- #
# Phenotype and genotype
# --------------------------------------------------------------------------- #
def test_fitness_is_the_ratio_to_the_wild_type_of_the_same_medium(
    workbook: Path,
) -> None:
    blocks = gr.read_table_s24(str(workbook))
    for block in blocks:
        for strain in block.deletions:
            phenotype = gr.fitness_phenotype(strain, block.wild_type)
            assert phenotype.fitness == pytest.approx(
                strain.mean / block.wild_type.mean
            )
            assert phenotype.fitness > 0.0
            assert phenotype.n_samples == strain.n_samples
            assert phenotype.sample_unit == SampleUnit.biological_replicate
            assert phenotype.fitness_uncertainty_type == UncertaintyType.sample_sd
            # the uncertainty is the released SD rescaled into the ratio's units
            assert phenotype.fitness_uncertainty == pytest.approx(
                strain.stdev_released / block.wild_type.mean
            )
            # the stored SE also carries the wild type's own spread, so it is never
            # smaller than what the uncertainty alone would derive
            assert phenotype.fitness_se is not None
            uncertainty = phenotype.fitness_uncertainty
            assert uncertainty is not None
            assert phenotype.fitness_se >= uncertainty / math.sqrt(strain.n_samples)


def test_reference_fitness_is_one_with_the_wild_types_own_relative_spread(
    workbook: Path,
) -> None:
    blocks = gr.read_table_s24(str(workbook))
    for block in blocks:
        reference = gr.reference_phenotype(block.wild_type)
        assert reference.fitness == 1.0
        assert reference.n_samples == block.wild_type.n_samples
        uncertainty = reference.fitness_uncertainty
        assert uncertainty is not None
        assert uncertainty == pytest.approx(
            block.wild_type.stdev_released / block.wild_type.mean
        )
        assert reference.fitness_se == pytest.approx(
            uncertainty / math.sqrt(block.wild_type.n_samples)
        )


def test_fitness_refuses_a_propagated_se_below_the_conditioned_one(
    workbook: Path,
) -> None:
    """The conservatism direction is asserted, not assumed."""
    blocks = gr.read_table_s24(str(workbook))
    strain = blocks[0].deletions[0]
    # A wild type with a huge mean and no spread makes the propagated SE the smaller of
    # the two only if the released SD is inflated past its own replicates.
    inflated = strain.model_copy(update={"stdev_released": strain.sample_sd * 100.0})
    with pytest.raises(RuntimeError, match="propagated SE"):
        gr.fitness_phenotype(inflated, blocks[0].wild_type)


def test_genotype_is_one_keio_deletion_against_the_pinned_namespace(
    workbook: Path,
) -> None:
    blocks = gr.read_table_s24(str(workbook))
    strain = blocks[0].deletions[0]
    genotype = gr.build_genotype(strain, "BW25113_4373")
    (perturbation,) = genotype.perturbations
    assert perturbation.perturbation_type == "bacterial_deletion"
    assert perturbation.systematic_gene_name == "BW25113_4373"
    assert perturbation.perturbed_gene_name == "rimI"
    assert perturbation.gene_namespace == NAMESPACE
    assert perturbation.collection == gr.KEIO_COLLECTION
    assert perturbation.cassette is None
    assert perturbation.construction is None
    assert perturbation.identifier_mapping is not None
    assert perturbation.identifier_mapping.route == "gene_symbol"
    assert perturbation.identifier_mapping.source_identifier == "ΔrimI"


def test_both_media_build_distinct_environments_from_the_proteome_condition_table() -> (
    None
):
    environments = gr.build_environments()
    assert set(environments) == {"Glucose", "Acetate"}
    carbon: dict[str, tuple[str, float]] = {}
    for medium, environment in environments.items():
        assert environment.temperature is not None
        assert environment.temperature.value == 37.0
        assert environment.aerobicity == "aerobic"
        (perturbation,) = environment.perturbations
        assert isinstance(perturbation, EnvironmentPhysicalPerturbation)
        assert perturbation.agent is not None
        assert perturbation.magnitude is not None
        dose = perturbation.magnitude.value
        assert dose is not None
        carbon[medium] = (perturbation.agent.name, dose)
    # The agent is the resolved compound identity, so glucose comes back as its
    # canonical name while the acetate reagent is the weighed salt the Methods names.
    assert carbon == {"Glucose": ("D-glucose", 5.0), "Acetate": ("sodium acetate", 3.5)}


# --------------------------------------------------------------------------- #
# Identifier resolution
# --------------------------------------------------------------------------- #
class _Locus:
    def __init__(self, symbol: str | None) -> None:
        self.symbol = symbol


class _Annotation:
    def __init__(self, loci: dict[str, _Locus]) -> None:
        self.loci = loci


LOCUS_TAGS = {"rimI": "BW25113_4373", "rimJ": "BW25113_1066", "rimL": "BW25113_1427"}


class FakeGenome:
    """Carries the three rim loci and nothing else."""

    ASSEMBLY_SET = ASSEMBLY_SET

    def __init__(self) -> None:
        """One locus per deleted gene."""
        self.genbank = _Annotation(
            {tag: _Locus(symbol) for symbol, tag in LOCUS_TAGS.items()}
        )

    def resolve_gene_name(self, name: str) -> GeneNameResolution:
        """Every stored locus tag resolves to itself; anything else is retired."""
        known = name in self.genbank.loci
        return GeneNameResolution(
            input_name=name,
            status=GeneNameStatus.CURRENT if known else GeneNameStatus.RETIRED,
            systematic_name=name,
        )


def _reconciliation(
    names: pd.Series, label: str, resolved: int
) -> LocusTagReconciliation:
    unresolved = len(set(names)) - resolved
    return LocusTagReconciliation(
        label=label,
        assembly_set=ASSEMBLY_SET,
        gene_namespace=NAMESPACE,
        unique_names=len(set(names)),
        status_histogram={
            GeneNameStatus.RENAMED: resolved,
            GeneNameStatus.CURRENT: 0,
            GeneNameStatus.NON_GENE_FEATURE: 0,
            GeneNameStatus.RETIRED: unresolved,
            GeneNameStatus.AMBIGUOUS: 0,
        },
        layer_histogram={"gene symbol": resolved},
        remapped=resolved,
        kept_on_collision=(),
        retired_kept=(),
        ambiguous_kept={},
        case_insensitive=(),
        outside_namespace=tuple(n for n in names if n not in LOCUS_TAGS),
    )


def _fake_reconcile(
    genome: Any, names: pd.Series, *, label: str
) -> tuple[pd.Series, LocusTagReconciliation]:
    """Map each rim symbol to its locus tag; anything else stays outside the namespace."""
    stored = pd.Series([LOCUS_TAGS.get(name, name) for name in names])
    resolved = sum(1 for name in set(names) if name in LOCUS_TAGS)
    return stored, _reconciliation(names, label, resolved)


def _reference_genome() -> AssemblyReferenceGenome:
    return AssemblyReferenceGenome(
        species="Escherichia coli",
        strain=sm.SCHMIDT_REFERENCE_STRAIN,
        ploidy="haploid",
        assembly_set=ASSEMBLY_SET,
        assembly_accession="GCA_000750555.1",
    )


def test_resolve_deletions_maps_every_label_and_refuses_an_unresolved_one(
    workbook: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gr, "reconcile_locus_tags", _fake_reconcile)
    blocks = gr.read_table_s24(str(workbook))
    locus_tag, report = gr.resolve_deletions(blocks, FakeGenome(), label="test")  # type: ignore[arg-type]
    assert locus_tag == LOCUS_TAGS
    assert report.outside_namespace == ()

    renamed = blocks[0].model_copy(
        update={
            "deletions": (
                blocks[0].deletions[0].model_copy(update={"strain": "ΔrimX"}),
                *blocks[0].deletions[1:],
            )
        }
    )
    from torchcell.datasets.bacteria_common import LocusTagResolutionError

    with pytest.raises(LocusTagResolutionError):
        gr.resolve_deletions((renamed, blocks[1]), FakeGenome(), label="test")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# End-to-end build on the synthetic workbook
# --------------------------------------------------------------------------- #
def test_download_links_the_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_workbook(tmp_path / "source.xlsx")
    sha = file_sha256(source)
    monkeypatch.setattr(sm, "SI2_SHA256", sha)
    data_root = tmp_path / "root"
    sm.deposit_raw_mirror(source=source, data_root=str(data_root))
    monkeypatch.setenv("DATA_ROOT", str(data_root))

    root = tmp_path / "build"
    (root / "raw").mkdir(parents=True)
    dataset = gr.GrowthRateSchmidt2016Dataset.__new__(gr.GrowthRateSchmidt2016Dataset)
    monkeypatch.setattr(
        type(dataset), "raw_dir", property(lambda self: str(root / "raw"))
    )
    dataset.download()
    assert file_sha256(root / "raw" / sm.SI2) == sha

    (data_root / sm.RAW_DIR_REL / sm.SI2_MIRROR_RELPATH).unlink()
    with pytest.raises(RuntimeError, match="missing from mirror"):
        dataset.download()


def test_process_builds_six_records_and_two_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "growth_rate_schmidt2016"
    raw = root / "raw"
    raw.mkdir(parents=True)
    path = write_workbook(raw / sm.SI2)
    monkeypatch.setattr(sm, "DATA_SHA256", {sm.SI2: file_sha256(path)})
    monkeypatch.setattr(gr, "reconcile_locus_tags", _fake_reconcile)
    monkeypatch.setattr(gr, "assembly_reference", lambda strain: _reference_genome())

    dataset = gr.GrowthRateSchmidt2016Dataset(
        root=str(root),
        ecoli_genome=FakeGenome(),  # type: ignore[arg-type]
    )
    assert len(dataset) == gr.EXPECTED_RECORDS
    assert dataset.gene_set == set(LOCUS_TAGS.values())

    items = [dataset.transform_item(dataset[i]) for i in range(len(dataset))]
    assert {len(i["experiment"].genotype.perturbations) for i in items} == {1}
    assert {i["publication"].doi for i in items} == {sm.PAPER_DOI}
    assert {i["reference"].phenotype_reference.fitness for i in items} == {1.0}
    assert all(i["experiment"].phenotype.fitness > 0.0 for i in items)
    # six distinct (strain, medium) pairs over three strains and two environments
    pairs = {
        (
            i["experiment"].genotype.perturbations[0].systematic_gene_name,
            i["experiment"].environment.model_dump_json(),
        )
        for i in items
    }
    assert len(pairs) == gr.EXPECTED_RECORDS
    assert len({pair[1] for pair in pairs}) == gr.EXPECTED_REFERENCES
    dataset.close_lmdb()

    preprocess = Path(dataset.preprocess_dir)
    ledger = json.loads((preprocess / "dropped_records.json").read_text())
    assert ledger["source_rows"] == 8
    assert ledger["kept_records"] == gr.EXPECTED_RECORDS
    assert ledger["dropped_records"] == 0
    assert ledger["rules"] == []
    assert ledger["reference_rows"] == ["Glucose/WT", "Acetate/WT"]
    assert any("Table S23" in note for note in ledger["notes"])
    checks = json.loads((preprocess / "released_statistics_check.json").read_text())
    assert checks["replicate_design_back_solved"]["uncertainty_type"] == "sample_sd"
    assert checks["table_s24_medium_headers"]["swapped"] is False
    gaps = json.loads((preprocess / "provenance_gaps.json").read_text())
    assert {gap["field"] for gap in gaps["gaps"]} == {"cassette", "construction"}
    sourced = json.loads((preprocess / "sourced_values.json").read_text())
    assert sourced["deletion_collection"]["value"] == gr.KEIO_COLLECTION
    strains = pd.read_csv(preprocess / "strains.csv")
    assert len(strains) == gr.EXPECTED_RECORDS
    assert set(strains["gene_symbol"]) == set(LOCUS_TAGS)
    assert (preprocess / "build_manifest.json").exists()

    report = gr.verify_build(str(root), genome=FakeGenome())  # type: ignore[arg-type]
    assert [(r.level, r.name) for r in report.results if not r.passed] == []
    assert (preprocess / "verification_report.json").exists()


# --------------------------------------------------------------------------- #
# Data-gated: the real mirror
# --------------------------------------------------------------------------- #
def _real_workbook() -> str:
    data_root = os.environ.get("DATA_ROOT")
    if data_root is None:
        pytest.skip("DATA_ROOT is not set")
    path = osp.join(data_root, sm.RAW_DIR_REL, sm.SI2_MIRROR_RELPATH)
    if not osp.exists(path):
        pytest.skip(f"raw mirror not deposited: {path}")
    return path


@pytest.mark.data
def test_real_table_s24_releases_eight_cells_with_two_short_ones() -> None:
    blocks = gr.read_table_s24(_real_workbook())
    counts = {
        f"{block.medium}/{row.strain}": row.n_samples
        for block in blocks
        for row in block.rows
    }
    assert len(counts) == 8
    assert sorted(key for key, n in counts.items() if n == 2) == [
        "Acetate/ΔrimJ",
        "Glucose/WT",
    ]
    assert set(counts.values()) == {2, 3}


@pytest.mark.data
def test_real_stdev_is_a_sample_not_a_population_standard_deviation() -> None:
    blocks = gr.read_table_s24(_real_workbook())
    summary = gr.check_released_statistics(blocks)
    assert summary["worst_average_abs_dev"] < 1e-15
    assert summary["worst_sample_sd_abs_dev"] < 1e-15
    assert summary["nearest_population_sd_abs_dev"] > 9e-4


@pytest.mark.data
def test_real_medium_headers_are_not_swapped() -> None:
    path = _real_workbook()
    blocks = gr.read_table_s24(path)
    rates = gr.read_table_s23_wild_type_rates(path)
    assert rates["Glucose"] == 0.58
    assert rates["Acetate"] == 0.3
    summary = gr.check_medium_headers(blocks, rates)
    assert summary["swapped"] is False
    assert summary["relative_distances"]["Glucose"]["Glucose"] == pytest.approx(
        0.0209, abs=5e-4
    )
    assert summary["relative_distances"]["Acetate"]["Acetate"] == pytest.approx(
        0.0344, abs=5e-4
    )


@pytest.mark.data
def test_real_fitness_ratios_are_all_positive_so_the_clamp_never_fires() -> None:
    blocks = gr.read_table_s24(_real_workbook())
    ratios = {
        f"{block.medium}/{strain.strain}": gr.fitness_phenotype(
            strain, block.wild_type
        ).fitness
        for block in blocks
        for strain in block.deletions
    }
    assert len(ratios) == gr.EXPECTED_RECORDS
    assert min(ratios.values()) > 0.0
    assert ratios["Acetate/ΔrimJ"] == pytest.approx(0.454350, abs=1e-6)
    assert ratios["Glucose/ΔrimJ"] == pytest.approx(0.611440, abs=1e-6)
    assert ratios["Acetate/ΔrimI"] == pytest.approx(1.012567, abs=1e-6)
