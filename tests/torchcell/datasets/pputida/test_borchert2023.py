# tests/torchcell/datasets/pputida/test_borchert2023.py
# [[tests.torchcell.datasets.pputida.test_borchert2023]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/pputida/test_borchert2023.py
"""Borchert 2023 loader (``torchcell.datasets.pputida.borchert2023``).

Synthetic tests write BOTH releases this loader joins -- a Supplementary File 1 with the
real 21-sheet layout and all 13 comparison sheets, and a Borchert 2024 compendium with
the 42 sample columns the arms are declared to equal -- plus a small served LMDB, then
run the partition proofs and a full ``process()`` build into ``tmp_path``. Four loci are
in the synthetic compendium and two are not, so the build stores the two and refuses the
four. The genome, the locus-tag reconciliation and the assembly pin are in-test objects,
so nothing reads ``$DATA_ROOT``.

The ``@pytest.mark.data`` tests read the real mirrors and, when it has been built, the
dev-tree LMDB, and pin the measured numbers of the dendron note: 42 arms each matching
one compendium sample to 0.0005 with the runner-up no nearer than 1.78, the 26 derived
``_mean`` columns, the 64,853 significance rows that have no home, the 271 recovered
loci and the 10,824 records.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import pickle
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import pandas as pd
import pytest

import torchcell.datasets.pputida.borchert2023 as b23
import torchcell.datasets.pputida.borchert2024 as b24
from torchcell.data import file_sha256
from torchcell.datamodels.schema import (
    AssemblyReferenceGenome,
    EnvironmentPhysicalPerturbation,
    SmallMoleculePerturbation,
)
from torchcell.datasets.bacteria_common import LocusTagReconciliation
from torchcell.literature.manifest import ROLE_RAW_DATA, ArtifactRecord, Manifest
from torchcell.sequence.genome.base import GeneNameResolution, GeneNameStatus

#: Four loci the synthetic compendium carries, two it does not.
SERVED_GENES = ("PP_0001", "PP_0002", "PP_0003", "PP_0004")
NEW_GENES = ("PP_0005", "PP_0006")
ALL_GENES = SERVED_GENES + NEW_GENES
#: ``Glu_v_Glu_NaCl`` omits the last locus, so an arm's new-locus set is a real union.
OMITS_LAST = "Glu_v_Glu_NaCl"
#: Half the released half-ULP: inside the tolerance, so a match still holds.
DRIFT = 0.0004


def _compendium_column(sample_index: int, gene_index: int) -> float:
    """A value separated from every other sample column by at least 1.0 on every gene."""
    return sample_index * 1.0 + gene_index * 0.01


def _sample_index(experiment: b23.ExperimentColumns, replicate: str) -> int:
    """Position of one arm's declared sample in the synthetic compendium."""
    return (experiment.experiment - 1) * 3 + b23.REPLICATES.index(replicate)


def _metadata_row(experiment: b23.ExperimentColumns, replicate: str) -> list[Any]:
    """One compendium metadata row for an arm, in ``METADATA_COLUMNS`` order."""
    cell: dict[str, Any] = {name: None for name in b24.METADATA_COLUMNS}
    cell.update(
        orgId="Putida",
        expName=experiment.compendium[b23.REPLICATES.index(replicate)],
        set="set100",
        expDesc="D-Glucose (C)",
        expGroup="carbon source",
        total_rep=3,
        rep=b23.REPLICATES.index(replicate) + 1,
        mutantLibrary=b23.MUTANT_LIBRARY,
        person="Andrew Borchert",
        media=b23.MEDIA_LABEL,
        temperature=b23.TEMPERATURE_C,
        aerobic="Aerobic",
        liquid="Liquid",
        condition_1="D-Glucose",
        concentration_1=b23.GLUCOSE_MM,
        units_1="mM",
    )
    if experiment.stressor is not None:
        cell.update(
            condition_2=experiment.stressor,
            concentration_2=experiment.dose_mm,
            units_2="mM",
        )
    return [cell[name] for name in b24.METADATA_COLUMNS]


def write_compendium(path: Path) -> Path:
    """A Borchert 2024 release with the 42 declared samples over ``SERVED_GENES``."""
    import openpyxl

    rows = [
        _metadata_row(experiment, replicate)
        for experiment in b23.EXPERIMENTS
        for replicate in b23.REPLICATES
    ]
    workbook = openpyxl.Workbook()
    meta = workbook.active
    meta.title = b24.METADATA_SHEET
    meta.append(list(b24.METADATA_COLUMNS))
    for row in rows:
        meta.append(row)
    headers = [f"{row[1]} {row[3]}" for row in rows]
    fitness = np.array(
        [
            [_compendium_column(s, g) for s in range(len(rows))]
            for g in range(len(SERVED_GENES))
        ]
    )
    for name, values in ((b24.FITNESS_SHEET, fitness), (b24.T_SHEET, fitness + 7.0)):
        sheet = workbook.create_sheet(name)
        sheet.append([*b24.GENE_COLUMNS, *headers])
        for gene, row_values in zip(SERVED_GENES, values, strict=True):
            sheet.append(["Putida", gene, gene, None, "desc", *row_values.tolist()])
    workbook.save(path)
    return path


def _sheet_tags(sheet: str) -> tuple[str, ...]:
    return ALL_GENES[:-1] if sheet == OMITS_LAST else ALL_GENES


def _sheet_value(
    experiment: b23.ExperimentColumns, replicate: str, gene_index: int
) -> float:
    """The released replicate value: the compendium's, off by less than half a ULP."""
    if gene_index < len(SERVED_GENES):
        return (
            _compendium_column(_sample_index(experiment, replicate), gene_index) + DRIFT
        )
    return 0.5 * experiment.experiment + 0.1 * gene_index


def write_release(path: Path) -> Path:
    """A Supplementary File 1 with the real sheet list and the 13 comparison sheets."""
    import openpyxl

    workbook = openpyxl.Workbook()
    first = workbook.active
    first.title = b23.SHEET_ORDER[0]
    first.append(["Tab", "Description"])
    for name in b23.SHEET_ORDER[1:]:
        if name in {c.sheet for c in b23.COMPARISONS}:
            continue
        workbook.create_sheet(name).append(["placeholder"])
    for plan in b23.COMPARISONS:
        sheet = workbook.create_sheet(plan.sheet)
        value_columns = b23._expected_columns(plan)
        sheet.append([*b23.ID_COLUMNS, *value_columns, *b23.STAT_COLUMNS])
        for gene_index, tag in enumerate(_sheet_tags(plan.sheet)):
            values: list[float] = []
            for number in (plan.reference_experiment, plan.condition_experiment):
                experiment = b23.COLUMNS_BY_EXPERIMENT[number]
                replicates = [
                    _sheet_value(experiment, r, gene_index) for r in b23.REPLICATES
                ]
                values.extend(replicates)
                values.append(float(np.mean(replicates)))
            sheet.append(
                [tag, f"{tag}_RS", None, "desc", *values, 3.0, 0.01, 0.02, 0.03]
            )
    workbook.save(path)
    return path


@pytest.fixture
def comparisons(tmp_path: Path) -> tuple[b23.Comparison, ...]:
    return b23.read_comparisons(write_release(tmp_path / b23.DATA_FILE))


@pytest.fixture
def release(tmp_path: Path) -> b24.Release:
    return b24.read_release(write_compendium(tmp_path / b24.DATA_FILE))


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def test_table_s4_covers_every_experiment_with_three_replicates() -> None:
    assert len(b23.TABLE_S4) == b23.N_EXPERIMENTS
    assert [layout.experiment for layout in b23.TABLE_S4] == list(range(1, 15))
    assert all(len(layout.duration) == 3 for layout in b23.TABLE_S4)
    assert b23.N_ARMS == 42


def test_duration_hours_converts_the_released_h_mm() -> None:
    assert b23.duration_hours("10:40") == pytest.approx(10 + 40 / 60)
    assert b23.duration_hours("9:30") == 9.5
    assert b23.duration_hours("31:00") == 31.0


def test_every_experiment_declares_three_distinct_compendium_samples() -> None:
    declared = [name for e in b23.EXPERIMENTS for name in e.compendium]
    assert len(declared) == len(set(declared)) == b23.N_ARMS
    assert len(b23.COMPARISONS) == b23.N_COMPARISONS
    assert b23.EXPERIMENTS[0].arm("A") == "Exp1A"
    assert b23.EXPERIMENTS[13].column("C") == "M9_Glucose_PCA_RepC"


def test_both_glucose_experiments_share_a_column_name() -> None:
    """The reason experiment 13 is its own arm: the sheets reuse the column name."""
    assert b23.EXPERIMENTS[0].prefix == b23.EXPERIMENTS[12].prefix == "M9_Glucose"
    assert b23.EXPERIMENTS[0].sheets != b23.EXPERIMENTS[12].sheets


# --------------------------------------------------------------------------- #
# Reader
# --------------------------------------------------------------------------- #
def test_read_comparisons_parses_every_sheet(
    comparisons: tuple[b23.Comparison, ...],
) -> None:
    assert [c.sheet for c in comparisons] == [c.sheet for c in b23.COMPARISONS]
    first = comparisons[0]
    assert first.tags == ALL_GENES
    assert first.columns == b23._expected_columns(b23.COMPARISONS[0])
    assert first.values.shape == (len(ALL_GENES), 8)
    assert first.stats.shape == (len(ALL_GENES), 4)
    assert next(c for c in comparisons if c.sheet == OMITS_LAST).tags == ALL_GENES[:-1]


def test_read_comparisons_refuses_a_renamed_column(tmp_path: Path) -> None:
    import openpyxl

    path = write_release(tmp_path / b23.DATA_FILE)
    workbook = openpyxl.load_workbook(path)
    workbook["Glu_v_Glu_Van"].cell(row=1, column=5).value = "M9_Glucose_RepZ"
    workbook.save(path)
    with pytest.raises(ValueError, match="Glu_v_Glu_Van: header"):
        b23.read_comparisons(path)


def test_read_comparisons_refuses_an_extra_sheet(tmp_path: Path) -> None:
    import openpyxl

    path = write_release(tmp_path / b23.DATA_FILE)
    workbook = openpyxl.load_workbook(path)
    workbook.create_sheet("Glu_v_Glu_Extra")
    workbook.save(path)
    with pytest.raises(ValueError, match="unexpected sheets"):
        b23.read_comparisons(path)


def test_read_comparisons_refuses_an_empty_cell(tmp_path: Path) -> None:
    import openpyxl

    path = write_release(tmp_path / b23.DATA_FILE)
    workbook = openpyxl.load_workbook(path)
    workbook["Glu_v_Glu_Van"].cell(row=2, column=5).value = None
    workbook.save(path)
    with pytest.raises(ValueError, match="has an empty cell"):
        b23.read_comparisons(path)


# --------------------------------------------------------------------------- #
# The subsumption measurement
# --------------------------------------------------------------------------- #
def test_match_arms_finds_every_declared_compendium_sample(
    comparisons: tuple[b23.Comparison, ...], release: b24.Release
) -> None:
    matches = b23.match_arms(comparisons, release)
    assert len(matches) == b23.N_ARMS
    assert [m.compendium_sample for m in matches] == [
        name for e in b23.EXPERIMENTS for name in e.compendium
    ]
    assert all(m.n_shared_loci == len(SERVED_GENES) for m in matches)
    assert all(m.max_abs_diff == pytest.approx(DRIFT) for m in matches)
    assert all(m.runner_up_max_abs_diff > b23.RUNNER_UP_FLOOR for m in matches)


def _drift_first_column(
    comparisons: tuple[b23.Comparison, ...], amount: float
) -> tuple[b23.Comparison, ...]:
    moved = list(comparisons)
    values = moved[0].values.copy()
    values[:, 0] += amount
    moved[0] = moved[0].model_copy(update={"values": values})
    return tuple(moved)


def test_match_arms_refuses_a_value_past_the_rounding(
    comparisons: tuple[b23.Comparison, ...], release: b24.Release
) -> None:
    """A value that moved past the compendium's rounding stops the build."""
    with pytest.raises(RuntimeError, match="exceeds 0.0005, so the subsumption"):
        b23.match_arms(_drift_first_column(comparisons, 0.2), release)


def test_match_arms_refuses_a_column_that_now_matches_another_sample(
    comparisons: tuple[b23.Comparison, ...], release: b24.Release
) -> None:
    """A column that drifted onto a DIFFERENT compendium sample stops the build."""
    with pytest.raises(RuntimeError, match="the closest compendium column is"):
        b23.match_arms(_drift_first_column(comparisons, 1.0), release)


def test_match_arms_refuses_a_near_runner_up(
    comparisons: tuple[b23.Comparison, ...], tmp_path: Path
) -> None:
    """Two compendium columns that became indistinguishable stop the build."""
    release = b24.read_release(write_compendium(tmp_path / b24.DATA_FILE))
    fitness = release.fitness.copy()
    fitness[:, 1] = fitness[:, 0]
    with pytest.raises(RuntimeError, match="runner-up column"):
        b23.match_arms(comparisons, release.model_copy(update={"fitness": fitness}))


def test_arms_agree_across_sheets_and_the_two_glucose_arms_do_not(
    comparisons: tuple[b23.Comparison, ...],
) -> None:
    proofs = b23.assert_arms_agree_across_sheets(comparisons)
    assert any("experiments 1 and 13" in p for p in proofs)
    assert sum("experiment 1:" in p for p in proofs) == len(b23.DAY1_SHEETS) - 1


def test_arms_agree_across_sheets_refuses_a_disagreement(
    comparisons: tuple[b23.Comparison, ...],
) -> None:
    moved = list(comparisons)
    index = next(i for i, c in enumerate(comparisons) if c.sheet == "Glu_v_Glu_4HBA")
    values = moved[index].values.copy()
    values[0, 0] += 1.0
    moved[index] = moved[index].model_copy(update={"values": values})
    with pytest.raises(RuntimeError, match="disagree by"):
        b23.assert_arms_agree_across_sheets(tuple(moved))


def test_means_are_derived(comparisons: tuple[b23.Comparison, ...]) -> None:
    proofs = b23.assert_means_are_derived(comparisons)
    assert len(proofs) == b23.N_COMPARISONS


def test_means_are_derived_refuses_a_tampered_mean(
    comparisons: tuple[b23.Comparison, ...],
) -> None:
    moved = list(comparisons)
    values = moved[0].values.copy()
    values[0, 3] += 1e-6
    moved[0] = moved[0].model_copy(update={"values": values})
    with pytest.raises(RuntimeError, match="is not the mean of its replicates"):
        b23.assert_means_are_derived(tuple(moved))


def test_new_loci_by_arm_takes_the_union_over_an_arm_s_sheets(
    comparisons: tuple[b23.Comparison, ...], release: b24.Release
) -> None:
    by_arm = b23.new_loci_by_arm(comparisons, release.genes)
    assert len(by_arm) == b23.N_ARMS
    # experiment 8 is the NaCl sheet alone, which omits the last locus
    assert by_arm["Exp8A"] == (NEW_GENES[0],)
    # experiment 1 unions its eleven sheets, so it recovers both
    assert by_arm["Exp1A"] == NEW_GENES


# --------------------------------------------------------------------------- #
# The partition against the served store
# --------------------------------------------------------------------------- #
def _write_served(root: Path, loci: tuple[str, ...], screens: list[str]) -> str:
    """A minimal served store with one record per (locus, screen)."""
    lmdb_dir = root / "processed" / "lmdb"
    lmdb_dir.mkdir(parents=True)
    env = lmdb.open(str(lmdb_dir), map_size=2**26)
    index = 0
    with env.begin(write=True) as txn:
        for screen in screens:
            for locus in loci:
                record = {
                    "experiment": {
                        "genotype": {
                            "perturbations": [{"systematic_gene_name": locus}]
                        },
                        "phenotype": {"screen_id": screen},
                    }
                }
                txn.put(f"{index}".encode(), pickle.dumps(record))
                index += 1
    env.close()
    return str(root)


def _served_screens() -> list[str]:
    return [name for e in b23.EXPERIMENTS for name in e.compendium]


def test_assert_served_partition_proves_both_directions(
    tmp_path: Path, comparisons: tuple[b23.Comparison, ...], release: b24.Release
) -> None:
    root = _write_served(tmp_path / "served", SERVED_GENES, _served_screens())
    matches = b23.match_arms(comparisons, release)
    partition = b23.assert_served_partition(root, NEW_GENES, matches, release)
    assert partition.served_loci == len(SERVED_GENES)
    assert partition.served_screens == b23.N_ARMS
    assert partition.served_records == len(SERVED_GENES) * b23.N_ARMS
    assert partition.shared_loci == 0
    assert partition.matched_samples_present == b23.N_ARMS


def test_assert_served_partition_refuses_a_locus_already_served(
    tmp_path: Path, comparisons: tuple[b23.Comparison, ...], release: b24.Release
) -> None:
    root = _write_served(tmp_path / "served", SERVED_GENES, _served_screens())
    matches = b23.match_arms(comparisons, release)
    with pytest.raises(RuntimeError, match="already served by Borchert 2024"):
        b23.assert_served_partition(root, (*NEW_GENES, "PP_0001"), matches, release)


def test_assert_served_partition_refuses_a_stale_store(
    tmp_path: Path, comparisons: tuple[b23.Comparison, ...], release: b24.Release
) -> None:
    root = _write_served(tmp_path / "served", SERVED_GENES[:2], _served_screens())
    matches = b23.match_arms(comparisons, release)
    with pytest.raises(RuntimeError, match="the served store is stale"):
        b23.assert_served_partition(root, NEW_GENES, matches, release)


def test_assert_served_partition_refuses_a_missing_screen(
    tmp_path: Path, comparisons: tuple[b23.Comparison, ...], release: b24.Release
) -> None:
    root = _write_served(tmp_path / "served", SERVED_GENES, _served_screens()[:-1])
    matches = b23.match_arms(comparisons, release)
    with pytest.raises(RuntimeError, match="the served store has no screen for"):
        b23.assert_served_partition(root, NEW_GENES, matches, release)


def test_read_served_loci_closes_the_environment(tmp_path: Path) -> None:
    """A held handle makes the next open fail; the reader must leave none."""
    root = _write_served(tmp_path / "served", SERVED_GENES, ["set100IT008"])
    loci, screens, records = b23.read_served_loci(root)
    assert loci == frozenset(SERVED_GENES)
    assert screens == frozenset({"set100IT008"})
    assert records == len(SERVED_GENES)
    again = lmdb.open(osp.join(root, "processed", "lmdb"), readonly=True, lock=False)
    again.close()


# --------------------------------------------------------------------------- #
# Environment, phenotype, dose agreement
# --------------------------------------------------------------------------- #
def test_build_environment_carries_the_stressor_and_the_released_duration() -> None:
    vanillin = b23.COLUMNS_BY_EXPERIMENT[6]
    environment = b23.build_environment(vanillin, "B")
    assert environment.media is b24.BORCHERT2023_M9
    assert environment.temperature is not None
    assert environment.temperature.value == b23.TEMPERATURE_C
    assert environment.aerobicity == "aerobic"
    assert environment.duration_hours == pytest.approx(15 + 10 / 60)
    glucose, stressor = environment.perturbations
    assert isinstance(glucose, EnvironmentPhysicalPerturbation)
    assert glucose.magnitude is not None
    assert glucose.magnitude.value == b23.GLUCOSE_MM
    assert isinstance(stressor, SmallMoleculePerturbation)
    assert stressor.compound.name == "vanillin"
    assert stressor.concentration.value == 10.0
    assert [gap.field for gap in environment.provenance_gaps] == [
        "duration_generations"
    ]


def test_build_environment_of_an_unstressed_arm_carries_glucose_only() -> None:
    environment = b23.build_environment(b23.COLUMNS_BY_EXPERIMENT[1], "A")
    assert len(environment.perturbations) == 1
    assert environment.duration_hours == pytest.approx(10 + 40 / 60)


def test_build_phenotype_is_a_signed_log2_ratio_with_typed_gaps() -> None:
    phenotype = b23.build_phenotype(-2.68, "Exp6A")
    assert phenotype.environment_response == -2.68
    assert phenotype.measurement_type == "log2_ratio"
    assert phenotype.assay_type == "pooled_competitive_growth_barcode"
    assert phenotype.screen_id == "Exp6A"
    assert phenotype.environment_response_se is None
    assert {gap.field for gap in phenotype.provenance_gaps} == {
        "environment_response_uncertainty",
        "environment_response_se",
        "n_samples",
        "sample_unit",
    }


def test_assert_dose_agrees_refuses_a_different_compendium_dose(
    release: b24.Release,
) -> None:
    vanillin = b23.COLUMNS_BY_EXPERIMENT[6]
    sample = next(s for s in release.samples if s.exp_name == vanillin.compendium[0])
    b23.assert_dose_agrees(vanillin, sample)
    with pytest.raises(RuntimeError, match="Borchert 2023's Methods give"):
        b23.assert_dose_agrees(
            vanillin, sample.model_copy(update={"concentration_2": 20.0})
        )


def test_assert_dose_agrees_refuses_a_stressor_on_an_unstressed_arm(
    release: b24.Release,
) -> None:
    glucose = b23.COLUMNS_BY_EXPERIMENT[1]
    sample = next(s for s in release.samples if s.exp_name == glucose.compendium[0])
    b23.assert_dose_agrees(glucose, sample)
    with pytest.raises(RuntimeError, match="has no stressor"):
        b23.assert_dose_agrees(
            glucose, sample.model_copy(update={"condition_2": "Vanillin"})
        )


def test_not_stored_names_every_unloaded_quantity(
    comparisons: tuple[b23.Comparison, ...], release: b24.Release
) -> None:
    matches = b23.match_arms(comparisons, release)
    quantities = b23.not_stored(matches, comparisons)
    assert len(quantities) == 6
    significance = next(q for q in quantities if "t-statistic" in q.quantity)
    assert significance.issue == 776
    rows = sum(len(c.tags) for c in comparisons)
    assert significance.values == rows * len(b23.STAT_COLUMNS)
    subsumed = next(q for q in quantities if "4,732 compendium loci" in q.quantity)
    assert subsumed.values == b23.N_ARMS * len(SERVED_GENES)


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def test_deposit_is_idempotent_and_refuses_other_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_release(tmp_path / "source.xlsx")
    data_root = str(tmp_path / "root")
    monkeypatch.setattr(b23, "DATA_SHA256", file_sha256(source))
    root = b23.deposit_raw_mirror(source, data_root=data_root)
    deposited = root / b23.DATA_RELPATH
    assert file_sha256(deposited) == b23.DATA_SHA256
    manifest = b23.load_manifest(data_root)
    assert b23.manifest_sha256(manifest) == b23.DATA_SHA256
    (record,) = manifest.files
    assert record.retrieval is not None
    assert record.retrieval.source_url == b23.DATA_URL
    assert record.original_filename == "mmc1.xlsx"
    b23.deposit_raw_mirror(source, data_root=data_root)
    other = write_compendium(tmp_path / "other.xlsx")
    with pytest.raises(RuntimeError, match="the pin is"):
        b23.deposit_raw_mirror(other, data_root=data_root)


def test_manifest_sha256_refuses_an_unknown_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = Manifest(
        citation_key=b23.CITATION_KEY,
        doi=b23.DOI,
        title=b23.TITLE,
        files=[
            ArtifactRecord(
                path="data/other.xlsx", role=ROLE_RAW_DATA, bytes=1, sha256="0" * 64
            )
        ],
    )
    with pytest.raises(KeyError, match="is not in the Borchert 2023 raw-mirror"):
        b23.manifest_sha256(manifest)


# --------------------------------------------------------------------------- #
# End-to-end build on the two synthetic releases
# --------------------------------------------------------------------------- #
class _Locus:
    def __init__(self, symbol: str | None) -> None:
        self.symbol = symbol


class _Annotation:
    def __init__(self, loci: dict[str, _Locus]) -> None:
        self.loci = loci


class FakeGenome:
    """``genbank.loci[tag].symbol`` and ``resolve_gene_name`` over ``ALL_GENES``."""

    def __init__(self) -> None:
        """PP_0005 carries a unique symbol, PP_0006 none."""
        symbols: dict[str, str | None] = {tag: None for tag in ALL_GENES}
        symbols["PP_0005"] = "ttgB"
        self.genbank = _Annotation(
            {tag: _Locus(symbol) for tag, symbol in symbols.items()}
        )

    def resolve_gene_name(self, name: str) -> GeneNameResolution:
        """``ttgB`` renames PP_0005; a locus tag is itself."""
        if name == "ttgB":
            return GeneNameResolution(
                input_name=name,
                status=GeneNameStatus.RENAMED,
                systematic_name="PP_0005",
                note="gene symbol of PP_0005",
            )
        return GeneNameResolution(
            input_name=name, status=GeneNameStatus.CURRENT, systematic_name=name
        )


def _reference() -> AssemblyReferenceGenome:
    return AssemblyReferenceGenome(
        species="Pseudomonas putida",
        strain="KT2440",
        ploidy="haploid",
        assembly_set="pputida_KT2440_ASM756v2",
        assembly_accession="GCA_000007565.2",
    )


def _identity_reconciliation(
    genome: Any, names: pd.Series, *, label: str
) -> tuple[pd.Series, LocusTagReconciliation]:
    return names, LocusTagReconciliation(
        label=label,
        assembly_set="pputida_KT2440_ASM756v2",
        gene_namespace="pputida_kt2440_locus_tag",
        unique_names=len(names),
        status_histogram={
            s: (len(names) if s is GeneNameStatus.CURRENT else 0)
            for s in GeneNameStatus
        },
        layer_histogram={"locus tag": len(names)},
        remapped=0,
        kept_on_collision=(),
        retired_kept=(),
        ambiguous_kept={},
        case_insensitive=(),
        outside_namespace=(),
    )


def _install_synthetic_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Both releases, the served store and the pins that point the loader at them."""
    root = tmp_path / "rbtnseq_borchert2023"
    raw = root / "raw"
    raw.mkdir(parents=True)
    monkeypatch.setattr(
        b23, "DATA_SHA256", file_sha256(write_release(raw / b23.DATA_FILE))
    )

    compendium_root = tmp_path / "torchcell-raw" / b24.CITATION_KEY
    (compendium_root / "data").mkdir(parents=True)
    compendium = write_compendium(compendium_root / b24.DATA_RELPATH)
    sha = file_sha256(compendium)
    monkeypatch.setattr(b24, "DATA_SHA256", sha)
    monkeypatch.setattr(b24, "raw_mirror_dir", lambda *a, **k: compendium_root)
    monkeypatch.setattr(
        b24,
        "load_manifest",
        lambda *a, **k: Manifest(
            citation_key=b24.CITATION_KEY,
            doi=b24.DOI,
            title=b24.TITLE,
            files=[
                ArtifactRecord(
                    path=b24.DATA_RELPATH,
                    role=ROLE_RAW_DATA,
                    bytes=compendium.stat().st_size,
                    sha256=sha,
                )
            ],
        ),
    )
    _write_served(tmp_path / b23.SERVED_ROOT_REL, SERVED_GENES, _served_screens())
    monkeypatch.setattr(b23, "_data_root", lambda: str(tmp_path))
    monkeypatch.setattr(b23, "reconcile_locus_tags", _identity_reconciliation)
    monkeypatch.setattr(b23, "assembly_reference", lambda strain: _reference())
    monkeypatch.setattr(b23, "EXPECTED_NEW_LOCI", len(NEW_GENES))
    monkeypatch.setattr(
        b23, "EXPECTED_RECORDS", b23.N_ARMS * len(NEW_GENES) - len(b23.REPLICATES)
    )
    return root


def test_process_stores_only_the_loci_the_compendium_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _install_synthetic_sources(tmp_path, monkeypatch)
    dataset = b23.RbTnseqBorchert2023Dataset(
        root=str(root),
        pputida_genome=FakeGenome(),  # type: ignore[arg-type]
    )
    # 42 arms x 2 new loci, less the NaCl arm's missing third locus
    assert len(dataset) == b23.N_ARMS * len(NEW_GENES) - len(b23.REPLICATES)

    stored = {
        dataset[i]["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"]
        for i in range(len(dataset))
    }
    assert stored == set(NEW_GENES)
    assert not stored & set(SERVED_GENES)

    first = dataset.transform_item(dataset[0])
    (perturbation,) = first["experiment"].genotype.perturbations
    assert perturbation.systematic_gene_name == "PP_0005"
    assert perturbation.perturbed_gene_name == "ttgB"
    assert perturbation.library_pool == b23.MUTANT_LIBRARY
    assert first["experiment"].phenotype.screen_id == "Exp1A"
    assert first["experiment"].environment.duration_hours == pytest.approx(10 + 40 / 60)
    assert first["reference"].phenotype_reference.environment_response == 0.0
    assert first["publication"].doi == b23.DOI

    screens = {
        dataset[i]["experiment"]["phenotype"]["screen_id"] for i in range(len(dataset))
    }
    assert screens == {e.arm(r) for e in b23.EXPERIMENTS for r in b23.REPLICATES}
    index = dataset.experiment_reference_index
    assert index is not None
    assert len(index) == b23.N_ARMS

    preprocess = Path(dataset.preprocess_dir)
    subsumption = json.loads((preprocess / "subsumption.json").read_text())
    assert len(subsumption["arms"]) == b23.N_ARMS
    partition = json.loads((preprocess / "served_partition.json").read_text())
    assert partition["shared_loci"] == 0
    assert partition["served_loci"] == len(SERVED_GENES)
    new_loci = json.loads((preprocess / "new_loci.json").read_text())
    assert new_loci["n_loci"] == len(NEW_GENES)
    assert new_loci["n_records"] == len(dataset)
    not_stored = json.loads((preprocess / "not_stored.json").read_text())
    assert len(not_stored["quantities"]) == 6
    assert "20 mM" in not_stored["read_me_conflict"]
    assert (preprocess / "build_manifest.json").exists()
    dataset.close_lmdb()

    report = b23.verify_build(
        str(root),
        genome=FakeGenome(),  # type: ignore[arg-type]
        expected_count=len(dataset),
    )
    failed = [(r.level, r.name, r.message) for r in report.results if not r.passed]
    assert failed == []
    assert (preprocess / "verification_report.json").exists()


def test_process_refuses_when_the_new_locus_count_drifts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _install_synthetic_sources(tmp_path, monkeypatch)
    monkeypatch.setattr(b23, "EXPECTED_NEW_LOCI", 99)
    with pytest.raises(RuntimeError, match="were measured"):
        b23.RbTnseqBorchert2023Dataset(
            root=str(root),
            pputida_genome=FakeGenome(),  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------- #
# Real data
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    from dotenv import load_dotenv

    load_dotenv()
    return os.environ["DATA_ROOT"]


@pytest.mark.data
def test_raw_mirror_matches_the_pin() -> None:
    data_root = _data_root()
    manifest = b23.load_manifest(data_root)
    assert b23.manifest_sha256(manifest) == b23.DATA_SHA256
    assert file_sha256(b23.raw_mirror_dir(data_root) / b23.DATA_RELPATH) == (
        b23.DATA_SHA256
    )
    (record,) = manifest.files
    assert record.bytes == b23.DATA_BYTES
    assert record.retrieval is not None
    assert record.retrieval.last_check is not None
    assert record.retrieval.last_check.matches


@pytest.mark.data
def test_every_quote_is_verbatim_in_its_pinned_mirror() -> None:
    from torchcell.verification.sourced import audit_sourced_value

    library = osp.join(_data_root(), "torchcell-library")
    failures = [
        (key, sv.provenance.source_uri)
        for key, sv in b23.SOURCED_VALUES.items()
        if not audit_sourced_value(sv, library).passed
    ]
    assert failures == []
    assert len(b23.SOURCED_VALUES) == 16 + b23.N_EXPERIMENTS


def _real_comparisons() -> tuple[b23.Comparison, ...]:
    path = b23.raw_mirror_dir(_data_root()) / b23.DATA_RELPATH
    if not path.exists():
        pytest.skip(f"raw mirror not deposited: {path}")
    return b23.read_comparisons(path)


@pytest.mark.data
def test_real_read_me_describes_every_comparison_sheet() -> None:
    """The Read_Me names the poolcount, growth and comparison tabs, and one dose
    disagrees with three other released statements of it.
    """
    described = b23.read_read_me(b23.raw_mirror_dir(_data_root()) / b23.DATA_RELPATH)
    assert set(described) == set(b23.SHEET_ORDER[1:])
    for plan in b23.COMPARISONS:
        assert described[plan.sheet].startswith("Normalized fitness values for")
    assert "20 mM 4-hydroxybenzaldehyde" in described["Glu_v_Glu_4HBald"]
    assert "10 mM 4-hydroxybenzaldehyde" not in described["Glu_v_Glu_4HBald"]
    assert b23.COLUMNS_BY_EXPERIMENT[4].dose_mm == 10.0
    assert "10 mM vanillin" in described["Glu_v_Glu_Van"]


@pytest.mark.data
def test_real_subsumption_and_the_measured_partition() -> None:
    """Every arm IS a served sample, and 271 loci are not served at all."""
    comparisons = _real_comparisons()
    path = b24.raw_mirror_dir(_data_root()) / b24.DATA_RELPATH
    if not path.exists():
        pytest.skip(f"Borchert 2024 raw mirror not deposited: {path}")
    release = b24.read_release(path)
    matches = b23.match_arms(comparisons, release)
    assert len(matches) == b23.N_ARMS
    assert {m.n_shared_loci for m in matches} == {b24.N_GENES}
    assert max(m.max_abs_diff for m in matches) <= b23.COMPENDIUM_HALF_ULP
    assert min(m.runner_up_max_abs_diff for m in matches) > 1.78
    b23.assert_arms_agree_across_sheets(comparisons)
    b23.assert_means_are_derived(comparisons)
    by_arm = b23.new_loci_by_arm(comparisons, release.genes)
    tags = {tag for loci in by_arm.values() for tag in loci}
    assert len(tags) == b23.EXPECTED_NEW_LOCI
    assert sum(len(loci) for loci in by_arm.values()) == b23.EXPECTED_RECORDS
    quantities = b23.not_stored(matches, comparisons)
    significance = next(q for q in quantities if "t-statistic" in q.quantity)
    assert significance.values == 64853 * len(b23.STAT_COLUMNS)
    subsumed = next(q for q in quantities if "4,732 compendium loci" in q.quantity)
    assert subsumed.values == b23.N_ARMS * b24.N_GENES == 198744


@pytest.mark.data
def test_built_lmdb_record_count() -> None:
    root = osp.join(_data_root(), "data/torchcell/rbtnseq_borchert2023")
    lmdb_dir = osp.join(root, "processed", "lmdb")
    if not osp.isdir(lmdb_dir):
        pytest.skip(f"not built: {lmdb_dir}")
    env = lmdb.open(lmdb_dir, readonly=True, lock=False)
    with env.begin() as txn:
        assert txn.stat()["entries"] == b23.EXPECTED_RECORDS
        first = pickle.loads(txn.get(b"0"))
    env.close()
    assert first["experiment"]["experiment_type"] == "bacterial_environment_response"
    partition = json.loads(
        Path(root, "preprocess", "served_partition.json").read_text()
    )
    assert partition["shared_loci"] == 0
    assert partition["served_loci"] == b24.N_GENES
    assert partition["matched_samples_present"] == b23.N_ARMS
