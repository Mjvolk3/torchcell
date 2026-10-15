# tests/torchcell/datasets/pputida/test_borchert2023_contrast.py
# [[tests.torchcell.datasets.pputida.test_borchert2023_contrast]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/pputida/test_borchert2023_contrast.py
"""Borchert 2023's two-arm fitness contrasts and the ``FitnessContrastPhenotype`` leaf.

Hermetic tests build a workbook with the pinned sheet list and comparison headers in
``tmp_path`` (two rows per comparison sheet), build the store and verify it; ``data``
tests read the deposited Supplementary File 1 and the built dev store.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from pathlib import Path
from typing import Any

import pytest
from openpyxl import Workbook

from torchcell.datamodels.schema import (
    FitnessContrastPhenotype,
    ReferenceGenome,
    TransposonInsertionPerturbation,
)
from torchcell.datasets.pputida import borchert2023 as b23
from torchcell.datasets.pputida import borchert2023_contrast as m
from torchcell.verification.report import Level, LevelResult
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

#: (old_locus_tag, gene_name, first-arm replicates, second-arm replicates, t, p, q, aq)
ROWS: list[tuple[Any, ...]] = [
    (
        "PP_0960",
        "mlaD",
        (0.35, 0.34, 0.36),
        (-2.47, -2.25, -2.53),
        32.47,
        5.4e-06,
        0.027,
        0.012,
    ),
    ("PP_1110", None, (0.1, 0.2, 0.3), (0.4, 0.5, 0.6), -3.1, 0.03, 1.0001, 0.9),
]


def _sheet_rows(plan: b23.ComparisonPlan) -> list[list[Any]]:
    header = (
        list(b23.ID_COLUMNS)
        + list(b23._expected_columns(plan))
        + list(b23.STAT_COLUMNS)
    )
    body: list[list[Any]] = []
    for tag, name, first, second, t, p, q, aq in ROWS:
        body.append(
            [tag, f"PP_RS{tag[3:]}", name, "description"]
            + [*first, sum(first) / 3, *second, sum(second) / 3]
            + [t, p, q, aq]
        )
    return [header, *body]


def write_workbook(path: Path) -> Path:
    """A workbook with the pinned sheet order; only the comparison sheets have rows."""
    workbook = Workbook()
    first = workbook.active
    assert first is not None
    workbook.remove(first)
    plans = {plan.sheet: plan for plan in b23.COMPARISONS}
    for name in b23.SHEET_ORDER:
        sheet = workbook.create_sheet(name)
        if name in plans:
            for row in _sheet_rows(plans[name]):
                sheet.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)
    return path


def test_read_contrast_rows_keeps_every_row_and_the_released_name(
    tmp_path: Path,
) -> None:
    rows = m.read_contrast_rows(write_workbook(tmp_path / "si1.xlsx"))
    assert len(rows) == 2 * len(b23.COMPARISONS)
    first = rows[0]
    assert (first.sheet, first.old_locus_tag, first.gene_name) == (
        b23.COMPARISONS[0].sheet,
        "PP_0960",
        "mlaD",
    )
    assert (first.t_statistic, first.p_value, first.q_value) == (32.47, 5.4e-06, 0.027)
    assert rows[1].gene_name is None
    assert rows[1].q_value == 1.0001
    assert all(m.sign_agrees(r) for r in rows)


def test_phenotype_stores_the_four_statistics_and_their_methods() -> None:
    row = m.ContrastRow(
        sheet="Glu_v_Glu_FA",
        old_locus_tag="PP_1110",
        gene_name=None,
        t_statistic=-3.1,
        p_value=0.03,
        q_value=1.0001,
        q_value_adjusted=0.9,
        first_mean=0.2,
        second_mean=0.5,
    )
    phenotype = m.phenotype(row)
    assert phenotype.q_value == 1.0001
    assert phenotype.q_value_method == "positive_false_discovery_rate"
    assert phenotype.q_value_adjustment_method == "monotonicity"
    assert phenotype.contrast_label == "Glu_v_Glu_FA"
    [perturbation] = m.genotype(row).perturbations
    assert isinstance(perturbation, TransposonInsertionPerturbation)
    assert perturbation.perturbed_gene_name == "PP_1110"
    assert perturbation.library_pool == "Putida_ML5_JBEI"


def test_reference_statistics_are_typed_gaps() -> None:
    reference = m.reference_phenotype("Glu_v_Glu_FA")
    assert (reference.t_statistic, reference.p_value, reference.q_value) == (
        None,
        None,
        None,
    )
    assert reference.gapped_fields() == {"t_statistic", "p_value", "q_value"}


def test_arm_environment_gaps_the_pooled_replicate_durations() -> None:
    environment = m.arm_environment(3)
    assert environment.duration_hours is None
    assert "duration_hours" in environment.gapped_fields()
    assert len(environment.perturbations) == 2


def _contrast(**kw: Any) -> FitnessContrastPhenotype:
    fields: dict[str, Any] = dict(
        measurement_type="m",
        test_name="two-sample t test",
        contrast_label="s",
        t_statistic=1.0,
        p_value=0.5,
        q_value=0.5,
        q_value_method="pfdr",
    )
    fields.update(kw)
    return FitnessContrastPhenotype(**fields)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"t_statistic": None}, "carries no ProvenanceGap"),
        ({"t_statistic": float("inf")}, "must be finite"),
        ({"p_value": 1.5}, "not a probability"),
        ({"q_value": -0.1}, "non-negative"),
        ({"q_value_method": None}, "set together"),
        ({"q_value_adjusted": 0.3}, "set together"),
        ({"n_replicates_condition": 0}, "positive integer"),
        ({"contrast_label": " "}, "cannot be blank"),
    ],
)
def test_leaf_refuses_an_incoherent_record(
    change: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _contrast(**change)


def test_leaf_admits_a_q_value_above_one_and_a_gapped_statistic() -> None:
    assert _contrast(q_value=1.2).q_value == 1.2
    gap = ProvenanceGap(
        field="t_statistic", reason=ProvenanceGapReason.not_reported_by_primary
    )
    assert _contrast(t_statistic=None, provenance_gaps=[gap]).t_statistic is None


REFERENCE = ReferenceGenome(
    species="Pseudomonas putida",
    strain="KT2440",
    assembly_set="pputida_KT2440_ASM756v2",
    assembly_accession="GCA_000007565.2",
)


def test_hermetic_build_and_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from torchcell.verification import runners

    data_root = tmp_path / "data_root"
    write_workbook(b23.raw_mirror_dir(str(data_root)) / b23.DATA_RELPATH)
    root = data_root / m.DATASET_ROOT_REL
    write_workbook(root / "raw" / b23.DATA_FILE)
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setattr(m, "verify_raw_files", lambda raw_dir, pins: None)
    monkeypatch.setattr(m, "EXPECTED_RECORDS", 2 * len(b23.COMPARISONS))
    monkeypatch.setattr(m, "EXPECTED_Q_ABOVE_ONE", len(b23.COMPARISONS))
    monkeypatch.setattr(m, "assembly_reference", lambda strain: REFERENCE)
    dataset = m.FitnessContrastBorchert2023Dataset(root=str(root))
    assert len(dataset) == 2 * len(b23.COMPARISONS)
    ledger = json.loads((root / "preprocess" / "contrasts.json").read_text())
    assert ledger["rows_without_gene_name"] == len(b23.COMPARISONS)

    monkeypatch.setattr(
        runners, "_bacterial_gene_set", lambda assembly, base: {"PP_0960"}
    )
    monkeypatch.setattr(
        m,
        "audit_sourced_value",
        lambda value, root: LevelResult(
            level=Level.L3, name="provenance_audit", passed=True, message="stub"
        ),
    )
    report = m.verify_build(str(root), str(data_root))
    assert report.passed, report.summary()
    l4 = next(r for r in report.results if r.level == Level.L4)
    assert l4.details["missing"] == ["PP_1110"]
    assert m.main(["verify"]) == 0


def test_build_refuses_a_count_that_is_not_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / m.DATASET_ROOT_REL
    write_workbook(root / "raw" / b23.DATA_FILE)
    monkeypatch.setattr(m, "verify_raw_files", lambda raw_dir, pins: None)
    with pytest.raises(RuntimeError, match="26 rows, expected 64853"):
        m.FitnessContrastBorchert2023Dataset(root=str(root))


def _data_root() -> str:
    return os.environ.get("DATA_ROOT", "")


@pytest.mark.data
def test_release_counts() -> None:
    rows = m.read_contrast_rows(b23.raw_mirror_dir(_data_root()) / b23.DATA_RELPATH)
    assert len(rows) == 64853
    assert sum(r.q_value > 1.0 for r in rows) == 582
    assert all(m.sign_agrees(r) for r in rows)


@pytest.mark.data
def test_dev_store_verifies() -> None:
    root = osp.join(_data_root(), m.DATASET_ROOT_REL)
    if not osp.isdir(osp.join(root, "processed")):
        pytest.skip("fitness_contrast_borchert2023 dev store not built")
    report = m.verify_build(root, _data_root())
    assert report.passed, report.summary()
