# tests/torchcell/datasets/scerevisiae/test_messner2021.py
# [[tests.torchcell.datasets.scerevisiae.test_messner2021]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_messner2021.py
"""Messner 2021 drug-response proteomes, built end to end on a synthetic DIA-NN report.

The synthetic plate is ONE block of 24 injections in the release's layout (18 agents in
Supplementary Table 4 order with FIN56 and folic acid in their wells, then six DMSO
wells), injection 2 absent as in the release, plus one QC run: 24 runs. Run ``i`` is well
``96 + i``. Quantities (``Genes.MaxLFQ.Unique``) are ``100 * i + k`` for protein ``k``::

    accession  ORF      runs quantified                        rule
    P1         YAL001C  all 24 (QC = 5.0)                      kept
    P2         YBR001W  all but injection 5 (QC = 6.0)         kept
    P5         YDR001C  all 24 (QC = 9.0)                      kept
    P03872     (REP2)   all 24                                 kept by rule 2, keyed P03872
    P3         Q0010    injections 1, 3, 4, 6-14 (12 of 24)    rule 2 drops it (= 50%)
    P4         YCR001W  every run at protein q 0.02            rule 1 drops it
    P1;P2      -        every run, not proteotypic             rule 1 drops it

After rule 2 a run identifies P1, P2, P5 and REP2 (4), but injection 5 (Pemetrexed)
identifies 3, below 0.8 x 4 = 3.2: the outlier. Records are the other 16 drug runs; the
reference is the six DMSO runs (injections 19-24), so P1's reference is the mean of
1901..2401, 2151.0, with SE = SD / sqrt(6).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import os.path as osp
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import lmdb
import numpy as np
import pandas as pd
import pytest

from torchcell.data import RawSha256MismatchError
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.schema import (
    ComponentDefinition,
    ConcentrationUnit,
    DoseBasis,
    Genotype,
    ProteinAbundanceExperiment,
    ProteinAbundanceExperimentReference,
    ReferenceGenome,
    SmallMoleculePerturbation,
)
from torchcell.datasets.scerevisiae import messner2021 as m
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.verification import runners
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)
from torchcell.verification.sourced import SourcedValue, audit_sourced_value

UP2ORF = {
    "P1": "YAL001C",
    "P2": "YBR001W",
    "P3": "Q0010",
    "P4": "YCR001W",
    "P5": "YDR001C",
}
PLATE_ORDER = [*m.AGENTS, *(["DMSO"] * 6)]
INJECTIONS = [i for i in range(1, 25) if i != 2]
QC_RUN = "NA_NA_NA_QC_1_QC_NA"
OUTLIER = "5_plate_2_wt_101_Pemetrexed_NA"
INFO_TEXT = (
    "\ufeffDIA-NN output files\r\n\r\n\r\nYeast drug screen \r\n"
    "20200914_sswath_yeast_drugscreen.tsv\r\n\r\nLibraries:\r\n"
)


def plate_run(injection: int) -> str:
    """The release's run name of one injection; the trailing number is per run."""
    token = "NA" if injection == 5 else f"0.{injection:02d}"
    agent = PLATE_ORDER[injection - 1]
    return f"{injection}_plate_2_wt_{injection + 96}_{agent}_{token}"


def _row(
    run: str, group: str, quantity: float, q: float = 0.001, pt: int = 1
) -> dict[str, Any]:
    return {
        "File.Name": f"{m.FILE_NAME_PREFIX}{run}{m.FILE_NAME_SUFFIX}",
        "Protein.Group": group,
        "Genes": group,
        "Genes.MaxLFQ.Unique": quantity,
        "Protein.Q.Value": q,
        "Proteotypic": pt,
    }


def _value(injection: int | None, k: int, qc: float) -> float:
    """``100 * injection + k`` on a plate run, the fixed ``qc`` value on the QC run."""
    return qc if injection is None else float(100 * injection + k)


def synthetic_report() -> pd.DataFrame:
    """The report table described in the module docstring (two precursors for P1)."""
    rows = []
    runs = [(plate_run(i), i) for i in INJECTIONS] + [(QC_RUN, None)]
    p3 = {1, 3, 4, *range(6, 15)}
    for run, i in runs:
        rows.append(_row(run, "P1", _value(i, 1, 5.0)))
        rows.append(_row(run, "P1", _value(i, 1, 5.0)))
        if i != 5:
            rows.append(_row(run, "P2", _value(i, 2, 6.0)))
        if i in p3:
            rows.append(_row(run, "P3", _value(i, 3, 7.0)))
        rows.append(_row(run, "P4", _value(i, 4, 8.0), q=0.02))
        rows.append(_row(run, "P5", _value(i, 5, 9.0)))
        rows.append(_row(run, "P03872", _value(i, 6, 10.0)))
        rows.append(_row(run, "P1;P2", float("nan"), pt=0))
    rows.append(_row(plate_run(1), "P1", float("nan")))
    return pd.DataFrame(rows, columns=m.REPORT_COLUMNS)


def write_raw(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    synthetic_report().to_csv(raw_dir / m.REPORT_FILENAME, sep="\t", index=False)
    (raw_dir / m.INFO_FILENAME).write_bytes(INFO_TEXT.encode("utf-8"))


@pytest.fixture
def parsed_report(tmp_path: Path) -> pd.DataFrame:
    write_raw(tmp_path / "raw")
    return m.read_report(tmp_path / "raw" / m.REPORT_FILENAME)


@pytest.fixture
def store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> m.ProteomeMessner2021Dataset:
    monkeypatch.setattr(m, "build_uniprot_to_orf_map", lambda: dict(UP2ORF))
    root = tmp_path / "proteome_messner2021"
    write_raw(root / "raw")
    return m.ProteomeMessner2021Dataset(root=str(root))


def _records(root: str) -> list[dict[str, Any]]:
    return m.read_store(root)


# --------------------------------------------------------------------------- #
# Release parsing
# --------------------------------------------------------------------------- #
def test_run_name_strips_the_windows_path_and_refuses_another() -> None:
    name = (
        f"{m.FILE_NAME_PREFIX}12_plate_2_wt_108_Atorvastatin_0.51{m.FILE_NAME_SUFFIX}"
    )
    assert m.run_name(name) == "12_plate_2_wt_108_Atorvastatin_0.51"
    with pytest.raises(ValueError, match="unexpected File.Name"):
        m.run_name("C:\\other\\12_plate_2_wt_108_Atorvastatin_0.51.wiff.dia")


def test_identified_cells_apply_rule_1_and_count_empty_quantities(
    parsed_report: pd.DataFrame,
) -> None:
    cells, empty = m.identified_cells(parsed_report)
    assert empty == 1
    assert sorted(cells["accession"].unique()) == ["P03872", "P1", "P2", "P3", "P5"]
    assert len(cells) == 24 + 23 + 12 + 24 + 24
    one = cells[(cells["run"] == plate_run(3)) & (cells["accession"] == "P2")]
    assert one["quantity"].tolist() == [302.0]


def test_identified_cells_refuse_a_proteotypic_group_of_several_proteins() -> None:
    report = pd.DataFrame([_row("r", "P1;P2", 1.0)]).assign(run="r")
    with pytest.raises(ValueError, match="1 proteotypic rows name several proteins"):
        m.identified_cells(report)


def test_identified_cells_refuse_two_quantities_for_one_cell() -> None:
    report = pd.DataFrame([_row("r", "P1", 1.0), _row("r", "P1", 2.0)]).assign(run="r")
    with pytest.raises(ValueError, match="1 \\(run, protein\\) cells carry several"):
        m.identified_cells(report)


def test_detection_rule_keeps_strictly_more_than_half() -> None:
    three = pd.DataFrame(
        {"r1": [1.0, 1.0], "r2": [1.0, np.nan], "r3": [np.nan, np.nan]},
        index=["two_of_three", "one_of_three"],
    )
    assert m.detection_rule(three).to_dict() == {
        "two_of_three": True,
        "one_of_three": False,
    }
    half = pd.DataFrame({"r1": [1.0], "r2": [np.nan]}, index=["p"])
    assert m.detection_rule(half).to_dict() == {"p": False}


def test_identification_rule_threshold_and_outliers() -> None:
    counts = pd.Series({"a": 10, "b": 8, "c": 7})
    threshold, outliers = m.identification_rule(counts)
    assert threshold == 8.0
    assert outliers == ["c"]


def test_dmso_reference_mean_se_and_single_run_nan() -> None:
    matrix = pd.DataFrame(
        {"d1": [2.0, 5.0, np.nan], "d2": [4.0, np.nan, np.nan]}, index=["A", "B", "C"]
    )
    ref = m.dmso_reference(matrix)
    assert ref.mean == {"A": 3.0, "B": 5.0}
    assert ref.n == {"A": 2, "B": 1}
    assert ref.se["A"] == pytest.approx(math.sqrt(2.0) / math.sqrt(2))
    assert math.isnan(ref.se["B"])


def test_report_is_the_drug_screen_reads_the_authors_index() -> None:
    assert m.report_is_the_drug_screen(INFO_TEXT.lstrip("\ufeff"))
    assert not m.report_is_the_drug_screen("Yeast drug screen\nother.tsv")
    assert not m.report_is_the_drug_screen("K562 benchmark\nx.tsv")
    assert not m.report_is_the_drug_screen("Yeast drug screen")


# --------------------------------------------------------------------------- #
# Plate layout
# --------------------------------------------------------------------------- #
def test_plate_layout_accepts_the_release_order_with_an_absent_injection() -> None:
    layout = {i: PLATE_ORDER[(i - 1) % 24] for i in range(1, 97) if i != 2}
    m.check_plate_layout(layout)
    assert [layout[i] for i in (1, 3, 15, 18, 19, 24, 25, 96)] == [
        "Rovustatin",
        "Pyrimethamine",
        "FIN56",
        "Folic_acid",
        "DMSO",
        "DMSO",
        "Rovustatin",
        "DMSO",
    ]


def test_plate_layout_refuses_a_misplaced_agent() -> None:
    layout = {i: PLATE_ORDER[i - 1] for i in range(1, 25)}
    layout[1], layout[3] = layout[3], layout[1]
    with pytest.raises(m.PlateLayoutError, match="injection 1 is 'Pyrimethamine'"):
        m.check_plate_layout(layout)


def test_plate_layout_refuses_an_unknown_token() -> None:
    with pytest.raises(m.PlateLayoutError, match=r"no compound mapping: \['Aspirin'\]"):
        m.check_plate_layout({1: "Aspirin"})


def test_the_agent_order_minus_the_two_methods_agents_is_table_s4() -> None:
    labels = [s.label for s in m.AGENTS.values() if s.drug_class is not None]
    assert labels == [label for label, _ in m.TABLE_S4]
    assert {s.token for s in m.AGENTS.values() if s.drug_class is None} == {
        "FIN56",
        "Folic_acid",
    }
    assert len(m.AGENTS) == 18


def test_build_rows_refuses_a_well_off_the_injection_offset(
    parsed_report: pd.DataFrame,
) -> None:
    bad = parsed_report.copy()
    bad["run"] = bad["run"].replace(plate_run(3), "3_plate_2_wt_100_Pyrimethamine_0.03")
    with pytest.raises(m.PlateLayoutError, match="well is not injection \\+ 96"):
        m.ProteomeMessner2021Dataset.build_rows(bad, dict(UP2ORF))


def test_build_rows_refuses_a_run_of_neither_kind(parsed_report: pd.DataFrame) -> None:
    bad = parsed_report.copy()
    bad["run"] = bad["run"].replace(QC_RUN, "blank_run")
    with pytest.raises(ValueError, match="'blank_run' is neither a plate nor a QC run"):
        m.ProteomeMessner2021Dataset.build_rows(bad, dict(UP2ORF))


# --------------------------------------------------------------------------- #
# Rows and ledger
# --------------------------------------------------------------------------- #
def test_build_rows_ledger_counts_every_run_and_cell(
    parsed_report: pd.DataFrame,
) -> None:
    ledger, rows = m.ProteomeMessner2021Dataset.build_rows(parsed_report, dict(UP2ORF))
    assert (ledger.n_runs, ledger.n_plate_runs, ledger.n_qc_runs) == (24, 23, 1)
    assert ledger.absent_injections == [2]
    assert ledger.n_proteins_identified == 5
    assert ledger.n_proteins_kept == 4
    assert ledger.n_cells_rule1 == 107
    assert ledger.n_cells_dropped_by_detection_rule == 12
    assert ledger.n_cells_without_quantity == 1
    assert ledger.max_identifications == 4
    assert ledger.identification_threshold == pytest.approx(3.2)
    assert ledger.mean_identifications_after_detection_rule == pytest.approx(95 / 24)
    assert ledger.mean_identifications_rule1_only == pytest.approx(107 / 24)
    assert ledger.outliers == [OUTLIER]
    assert ledger.accessions_stored_as_released == {
        "P03872": m.ACCESSIONS_WITHOUT_ORF["P03872"]
    }
    assert ledger.accession_to_orf == {
        "P03872": "P03872",
        "P1": "YAL001C",
        "P2": "YBR001W",
        "P5": "YDR001C",
    }
    assert (ledger.n_records, ledger.n_reference_runs) == (16, 6)
    assert ledger.n_table_s4_records == 14
    roles = {e.run: e.role for e in ledger.runs}
    assert roles[QC_RUN] == "qc" and roles[OUTLIER] == "outlier"
    assert [e.run for e in ledger.runs if e.role == "reference"] == sorted(
        plate_run(i) for i in range(19, 25)
    )
    first = next(e for e in ledger.runs if e.run == plate_run(1))
    assert first.model_dump() == {
        "run": plate_run(1),
        "kind": "plate",
        "injection": 1,
        "well": 97,
        "block": 0,
        "agent_token": "Rovustatin",
        "compound_label": "Rosuvastatin Calcium",
        "trailing_token": "0.01",
        "n_identified": 5,
        "n_after_detection_rule": 4,
        "role": "experiment",
        "record_index": 0,
    }
    assert [r["run"] for r in rows] == [
        plate_run(i) for i in INJECTIONS if i != 5 and i < 19
    ]
    assert rows[0]["abundance"] == {
        "P03872": 106.0,
        "YAL001C": 101.0,
        "YBR001W": 102.0,
        "YDR001C": 105.0,
    }
    assert rows[0]["reference"].mean == {
        "P03872": 2156.0,
        "YAL001C": 2151.0,
        "YBR001W": 2152.0,
        "YDR001C": 2155.0,
    }
    assert rows[0]["reference"].n == {
        "P03872": 6,
        "YAL001C": 6,
        "YBR001W": 6,
        "YDR001C": 6,
    }
    sd = float(np.std([100 * i + 1 for i in range(19, 25)], ddof=1))
    assert rows[0]["reference"].se["YAL001C"] == pytest.approx(sd / math.sqrt(6))


def test_build_rows_keys_an_unmapped_kept_accession_by_the_accession(
    parsed_report: pd.DataFrame,
) -> None:
    """A kept accession the GFF carries no ORF for is stored under the accession."""
    up2orf = {k: v for k, v in UP2ORF.items() if k != "P5"}
    ledger, rows = m.ProteomeMessner2021Dataset.build_rows(parsed_report, up2orf)
    assert ledger.accession_to_orf["P5"] == "P5"
    assert ledger.accessions_stored_as_released == {
        "P03872": m.ACCESSIONS_WITHOUT_ORF["P03872"],
        "P5": "no feature of the S288C R64 GFF names this accession",
    }
    assert any("P5" in row["abundance"] for row in rows)


def test_build_rows_refuses_two_accessions_on_one_orf(
    parsed_report: pd.DataFrame,
) -> None:
    up2orf = dict(UP2ORF, P5="YAL001C")
    with pytest.raises(RuntimeError, match="several accessions map to one ORF"):
        m.ProteomeMessner2021Dataset.build_rows(parsed_report, up2orf)


def test_build_rows_refuses_a_drug_protein_no_dmso_run_measured(
    parsed_report: pd.DataFrame,
) -> None:
    """P6 is quantified in the 17 drug and QC runs (> 50%) and in no DMSO run."""
    drug_and_qc = [plate_run(i) for i in INJECTIONS if i < 19] + [QC_RUN]
    extra = pd.DataFrame([_row(run, "P6", 50.0) for run in drug_and_qc])
    extra["run"] = drug_and_qc
    report = pd.concat([parsed_report, extra], ignore_index=True)
    with pytest.raises(m.MissingDmsoReferenceError, match="1 protein\\(s\\) no DMSO"):
        m.ProteomeMessner2021Dataset.build_rows(report, dict(UP2ORF, P6="YEL001C"))


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
def test_store_holds_one_wild_type_record_per_kept_drug_run(
    store: m.ProteomeMessner2021Dataset,
) -> None:
    assert len(store) == 16
    records = _records(store.root)
    first = records[0]
    exp = ProteinAbundanceExperiment.model_validate(first["experiment"])
    ref = ProteinAbundanceExperimentReference.model_validate(first["reference"])
    assert isinstance(exp.genotype, Genotype) and exp.genotype.perturbations == []
    assert exp.phenotype.protein_abundance == {
        "P03872": 106.0,
        "YAL001C": 101.0,
        "YBR001W": 102.0,
        "YDR001C": 105.0,
    }
    assert exp.phenotype.n_replicates == {
        "P03872": 1,
        "YAL001C": 1,
        "YBR001W": 1,
        "YDR001C": 1,
    }
    assert exp.phenotype.protein_abundance_se is None
    assert exp.phenotype.measurement_type == m.MEASUREMENT_TYPE
    treatment = exp.environment.perturbations[0]
    assert isinstance(treatment, SmallMoleculePerturbation)
    assert treatment.compound == resolved_compound("Rosuvastatin Calcium")
    assert treatment.compound.inchikey == "LALFOYNTGMUKGG-BGRFNVSISA-L"
    assert (
        treatment.concentration.value,
        treatment.concentration.unit,
        treatment.concentration.basis,
    ) == (10.0, ConcentrationUnit.micromolar, DoseBasis.fixed)
    assert treatment.solvent is not None
    assert (treatment.solvent.name, treatment.solvent.percent) == ("DMSO", None)
    assert exp.environment.temperature is not None
    assert exp.environment.temperature.value == 30.0
    assert exp.environment.media == m.SM_UNSTATED_MESSNER2021
    assert ref.genome_reference == ReferenceGenome(
        species="Saccharomyces cerevisiae",
        strain="BY4741",
        assembly_set=SGD_S288C_R64,
        assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
    )
    vehicle = ref.environment_reference.perturbations[0]
    assert isinstance(vehicle, SmallMoleculePerturbation)
    assert vehicle.compound.name == "dimethyl sulfoxide"
    assert (vehicle.concentration.value, vehicle.concentration.basis) == (
        None,
        DoseBasis.fixed,
    )
    assert vehicle.solvent is None
    assert ref.phenotype_reference.protein_abundance == {
        "P03872": 2156.0,
        "YAL001C": 2151.0,
        "YBR001W": 2152.0,
        "YDR001C": 2155.0,
    }
    assert first["publication"]["doi"] == m.DOI
    assert first["publication"]["pubmed_id"] == m.PUBMED_ID
    folic = records[-1]["experiment"]["environment"]["perturbations"][0]["compound"]
    assert folic["name"] == "folic acid"


def test_store_writes_the_ledger_and_the_run_table(
    store: m.ProteomeMessner2021Dataset,
) -> None:
    ledger = m.BuildLedger.model_validate_json(
        Path(store.preprocess_dir, "build_ledger.json").read_text()
    )
    assert ledger.n_records == 16
    runs = pd.read_csv(osp.join(store.preprocess_dir, "runs.csv"))
    assert len(runs) == 24
    assert runs["role"].value_counts().to_dict() == {
        "experiment": 16,
        "reference": 6,
        "qc": 1,
        "outlier": 1,
    }


def test_process_verifies_both_raw_files_before_reading(
    store: m.ProteomeMessner2021Dataset, raw_pin_calls: list[Any]
) -> None:
    calls = [c for c in raw_pin_calls if c[1] == store.raw_dir]
    assert calls == [
        (
            "messner2021",
            store.raw_dir,
            {m.REPORT_FILENAME: m.REPORT_SHA256, m.INFO_FILENAME: m.INFO_SHA256},
        )
    ]


def test_an_off_pin_report_is_refused_before_any_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from torchcell.data.experiment_dataset import verify_raw_files

    monkeypatch.setattr(m, "verify_raw_files", verify_raw_files)
    root = tmp_path / "offpin"
    write_raw(root / "raw")
    with pytest.raises(RawSha256MismatchError, match=m.REPORT_FILENAME):
        m.ProteomeMessner2021Dataset(root=str(root))
    assert not (root / "processed" / "lmdb").exists()


def test_process_refuses_an_index_that_does_not_name_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m, "build_uniprot_to_orf_map", lambda: dict(UP2ORF))
    root = tmp_path / "wrong_index"
    write_raw(root / "raw")
    (root / "raw" / m.INFO_FILENAME).write_text("Yeast drug screen\nother.tsv\n")
    with pytest.raises(ValueError, match="does not list"):
        m.ProteomeMessner2021Dataset(root=str(root))


def test_medium_names_sm_without_a_recipe() -> None:
    media = m.SM_UNSTATED_MESSNER2021
    assert media.state == "liquid" and media.is_synthetic and media.base_medium is None
    assert [c.definition for c in media.components] == [
        ComponentDefinition.composition_deferred
    ]
    assert media.components[0].concentration is None
    assert media.open_gaps == ["synthetic minimal (SM) medium, composition not stated"]


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _staged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    src = tmp_path / "retrieved"
    src.mkdir()
    report, info = src / m.REPORT_FILENAME, src / m.INFO_FILENAME
    report.write_bytes(b"report bytes")
    info.write_bytes(b"info bytes")
    monkeypatch.setattr(m, "REPORT_SHA256", hashlib.sha256(b"report bytes").hexdigest())
    monkeypatch.setattr(m, "INFO_SHA256", hashlib.sha256(b"info bytes").hexdigest())
    return report, info


def test_deposit_writes_both_files_and_a_direct_url_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, info = _staged(tmp_path, monkeypatch)
    root = m.deposit_raw_mirror(
        report_path=report, info_path=info, data_root=str(tmp_path / "dr")
    )
    assert root == tmp_path / "dr" / m.RAW_DIR_REL
    assert (root / m.REPORT_REL).read_bytes() == b"report bytes"
    manifest = m.load_manifest(str(tmp_path / "dr"))
    assert [f.path for f in manifest.files] == [m.REPORT_REL, m.INFO_REL]
    record = manifest.files[0]
    assert record.retrieval is not None
    assert record.retrieval.model_dump(exclude={"last_check"}) == {
        "method": "direct_url",
        "source_url": m.REPORT_URL,
        "retriever": "torchcell.literature.retrieve.direct_url",
        "params": {"url": m.REPORT_URL},
        "sha256": m.REPORT_SHA256,
        "retrieved_at": "2026-10-11",
    }
    assert m.manifest_sha256(manifest, m.INFO_REL) == m.INFO_SHA256
    with pytest.raises(KeyError, match="not in the raw-mirror manifest"):
        m.manifest_sha256(manifest, "data/other.tsv")
    # idempotent: a second deposit leaves the same bytes
    m.deposit_raw_mirror(
        report_path=report, info_path=info, data_root=str(tmp_path / "dr")
    )
    assert (root / m.INFO_REL).read_bytes() == b"info bytes"


def test_deposit_refuses_off_pin_bytes_and_a_changed_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, info = _staged(tmp_path, monkeypatch)
    data_root = str(tmp_path / "dr")
    m.deposit_raw_mirror(report_path=report, info_path=info, data_root=data_root)
    (m.raw_mirror_dir(data_root) / m.REPORT_REL).write_bytes(b"edited")
    with pytest.raises(RuntimeError, match="exists with a different sha256"):
        m.deposit_raw_mirror(report_path=report, info_path=info, data_root=data_root)
    report.write_bytes(b"not the release")
    with pytest.raises(RuntimeError, match="is not the pin"):
        m.deposit_raw_mirror(report_path=report, info_path=info, data_root=data_root)


def test_download_links_both_mirror_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, info = _staged(tmp_path, monkeypatch)
    data_root = tmp_path / "dr"
    m.deposit_raw_mirror(report_path=report, info_path=info, data_root=str(data_root))
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    raw_dir = tmp_path / "build" / "raw"
    m.ProteomeMessner2021Dataset.download(SimpleNamespace(raw_dir=str(raw_dir)))  # type: ignore[arg-type]
    assert sorted(os.listdir(raw_dir)) == sorted([m.REPORT_FILENAME, m.INFO_FILENAME])
    assert os.path.realpath(raw_dir / m.REPORT_FILENAME) == str(
        m.raw_mirror_dir(str(data_root)) / m.REPORT_REL
    )


def test_download_refuses_a_file_missing_from_the_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, info = _staged(tmp_path, monkeypatch)
    data_root = tmp_path / "dr"
    m.deposit_raw_mirror(report_path=report, info_path=info, data_root=str(data_root))
    (m.raw_mirror_dir(str(data_root)) / m.REPORT_REL).unlink()
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    with pytest.raises(RuntimeError, match="required raw artifact missing"):
        m.ProteomeMessner2021Dataset.download(
            SimpleNamespace(raw_dir=str(tmp_path / "raw"))  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
@pytest.fixture
def gate_env(
    store: m.ProteomeMessner2021Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> str:
    """Point every external read of ``verify_build`` at synthetic stand-ins."""
    library = tmp_path / "dr" / "torchcell-library" / m.CITATION_KEY
    library.mkdir(parents=True)
    text = "The yeast samples were grown at 30 C."
    (library / "paper.md").write_text(text)
    sv = SourcedValue(
        value=30,
        provenance=Provenance(
            source_uri="paper.md",
            citation_key=m.CITATION_KEY,
            sha256=hashlib.sha256(text.encode()).hexdigest(),
        ),
        quote="grown at 30 C",
    )
    monkeypatch.setattr(m, "sourced_values", lambda: [sv])
    monkeypatch.setattr(runners, "_sgd_gene_set", lambda root: set(UP2ORF.values()))
    monkeypatch.setattr(
        m,
        "messner2023_reference_profile",
        lambda root: {"YAL001C": 10.0, "YBR001W": 20.0, "YDR001C": 40.0},
    )
    for name, value in {
        "EXPECTED_RUNS": 24,
        "EXPECTED_PLATE_RUNS": 23,
        "EXPECTED_QC_RUNS": 1,
        "EXPECTED_RECORDS": 16,
        "EXPECTED_REFERENCE_RUNS": 6,
        "EXPECTED_TABLE_S4_RECORDS": 14,
        "PAPER_MEAN_IDENTIFICATIONS": 0,
    }.items():
        monkeypatch.setattr(m, name, value)
    return str(tmp_path / "dr")


def test_verify_build_passes_the_synthetic_store(
    store: m.ProteomeMessner2021Dataset, gate_env: str
) -> None:
    report = m.verify_build(store.root, data_root=gate_env)
    names = [(r.level, r.name, r.passed) for r in report.results]
    # P03872 is stored under its released accession, so the L4 containment rule
    # reports it outside S288C; that rule becomes a finding (never a gate) on the
    # fix/genome-rules-report-not-gate branch, and every other level passes here.
    assert [n for _, n, passed in names if not passed] == ["gene_containment_sgd"], (
        report.summary()
    )
    assert [n for _, n, _ in names] == [
        "structural",
        "count",
        "orf_uniqueness",
        "value_fidelity",
        "se_nonnegative",
        "reference_finite",
        "measurement_type_consistent",
        "wild_type_genotype",
        "runs_per_compound",
        "release_layout",
        "values_match_report",
        "reference_matches_report",
        "sourced_quotes_verbatim",
        "paper_mean_identifications",
        "paper_biological_sample_count",
        "gene_containment_sgd",
        "messner2023_reference_agreement",
    ]
    by_name = {r.name: r for r in report.results}
    assert by_name["values_match_report"].details == {"n_values": 64, "n_bad": 0}
    assert by_name["messner2023_reference_agreement"].details["n_shared"] == 3


def test_verify_build_catches_a_changed_value_and_a_wrong_reference(
    store: m.ProteomeMessner2021Dataset, gate_env: str
) -> None:
    env = lmdb.open(osp.join(store.root, "processed", "lmdb"), map_size=int(1e8))
    with env.begin(write=True) as txn:
        stored = txn.get(b"0")
        assert stored is not None
        record = pickle.loads(stored)
        record["experiment"]["phenotype"]["protein_abundance"]["YAL001C"] = 999.0
        record["reference"]["phenotype_reference"]["protein_abundance"]["YBR001W"] = 1.0
        txn.put(b"0", pickle.dumps(record))
    env.close()
    report = m.verify_build(store.root, data_root=gate_env)
    by_name = {r.name: r for r in report.results}
    assert not report.passed
    assert by_name["values_match_report"].details == {"n_values": 64, "n_bad": 1}
    assert by_name["reference_matches_report"].details["n_bad"] == 1


def test_check_sourced_reports_a_quote_no_longer_in_the_file(
    gate_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = osp.join(gate_env, "torchcell-library")
    sv = m.sourced_values()[0]
    monkeypatch.setattr(
        m, "sourced_values", lambda: [sv, sv.model_copy(update={"quote": "absent"})]
    )
    result = m.check_sourced(library)
    assert not result.passed
    assert result.message == "1/2 quotes verbatim in the mirror"


def test_paper_count_and_containment_checks_fail_off_their_values() -> None:
    ledger = SimpleNamespace(
        mean_identifications_after_detection_rule=1960.0,
        mean_identifications_rule1_only=2279.3,
        n_runs=102,
    )
    records = [
        {
            "experiment": {
                "environment": {"perturbations": [{"compound": {"name": "fin56"}}]},
                "phenotype": {"protein_abundance": {"YAL001C": 1.0, "R0040C": 2.0}},
            }
        }
    ]
    mean_ids, s4 = m.check_paper_counts(records, ledger)  # type: ignore[arg-type]
    assert (mean_ids.passed, s4.passed) == (False, False)
    assert s4.details == {"n": 0}
    containment = m.check_sgd_containment(records, {"YAL001C"})
    assert not containment.passed
    assert containment.details == {"n_keys": 2, "outside": ["R0040C"]}


def test_messner2023_reference_agreement_needs_three_shared_proteins() -> None:
    records = [
        {
            "reference": {
                "phenotype_reference": {"protein_abundance": {"A": 1.0, "B": 2.0}}
            }
        }
    ]
    result = m.check_messner2023_reference(records, {"A": 1.0, "B": 2.0})
    assert not result.passed
    assert math.isnan(result.details["r"])


def test_messner2023_reference_profile_unions_the_store_references(
    tmp_path: Path,
) -> None:
    lmdb_dir = tmp_path / m.MESSNER2023_ROOT / "processed" / "lmdb"
    lmdb_dir.mkdir(parents=True)
    env = lmdb.open(str(lmdb_dir), map_size=int(1e7))
    with env.begin(write=True) as txn:
        for i, profile in enumerate([{"A": 1.0}, {"A": 1.0, "B": 3.0}]):
            txn.put(
                f"{i}".encode(),
                pickle.dumps(
                    {
                        "reference": {
                            "phenotype_reference": {"protein_abundance": profile}
                        }
                    }
                ),
            )
    env.close()
    assert m.messner2023_reference_profile(str(tmp_path), {}) == {"A": 1.0, "B": 3.0}
    # #1045: Messner 2023 keys by released UniProt accession; the join lifts each one to
    # its ORF through the GFF map, and an unmapped accession keeps its released key.
    assert m.messner2023_reference_profile(str(tmp_path), {"A": "YAL001C"}) == {
        "YAL001C": 1.0,
        "B": 3.0,
    }


def test_run_protein_dispatches_a_verify_build_module_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, str]] = []

    def fake(root: str, *, data_root: str) -> VerificationReport:
        seen.append((root, data_root))
        report = VerificationReport(
            dataset_name="proteome_messner2021",
            provenance=runners.PROTEIN_DATASETS["proteome_messner2021"]["provenance"],
        )
        return report.add(
            LevelResult(level=Level.L0, name="stub", passed=True, message="ok")
        )

    monkeypatch.setattr(m, "verify_build", fake)
    monkeypatch.setattr(
        runners,
        "PROTEIN_DATASETS",
        {"proteome_messner2021": runners.PROTEIN_DATASETS["proteome_messner2021"]},
    )
    assert runners.run_protein(str(tmp_path)) is True
    root = str(tmp_path / "data/torchcell/proteome_messner2021")
    assert seen == [(root, str(tmp_path))]
    written = json.loads(
        Path(root, "preprocess", "verification_report.json").read_text()
    )
    assert written["dataset_name"] == "proteome_messner2021"


def test_registry_entry_points_at_this_module_and_the_measured_count() -> None:
    spec = runners.PROTEIN_DATASETS["proteome_messner2021"]
    assert spec["verify_build_module"] == m.__name__
    assert spec["expected_count"] == m.EXPECTED_RECORDS == 70
    assert spec["provenance"].sha256 == m.REPORT_SHA256


# --------------------------------------------------------------------------- #
# Real data (``--data``)
# --------------------------------------------------------------------------- #
_LIBRARY = osp.join(os.environ.get("DATA_ROOT", ""), "torchcell-library")


@pytest.mark.data
@pytest.mark.parametrize("key", sorted(m.SOURCED))
def test_every_sourced_value_audits_against_the_mirror(key: str) -> None:
    result = audit_sourced_value(m.SOURCED[key], _LIBRARY)
    assert result.passed, result.message


@pytest.mark.data
def test_the_built_dev_store_passes_its_gate() -> None:
    data_root = os.environ["DATA_ROOT"]
    root = osp.join(data_root, "data/torchcell/proteome_messner2021")
    report = m.verify_build(root, data_root=data_root)
    assert report.passed, report.summary()
    ledger = m.BuildLedger.model_validate_json(
        Path(root, "preprocess", "build_ledger.json").read_text()
    )
    assert (ledger.n_proteins_identified, ledger.n_proteins_kept) == (4244, 2200)
    assert ledger.outliers == m.EXPECTED_OUTLIERS
    assert round(ledger.mean_identifications_after_detection_rule, 1) == 1976.4
