# tests/torchcell/verification/test_pseudobulk_umi_count.py
# [[tests.torchcell.verification.test_pseudobulk_umi_count]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/verification/test_pseudobulk_umi_count.py
"""The pseudobulk UMI-count verifier on the miniature Jackson 2020 store.

``_jackson2020_release.py`` builds a fifteen-record store from sixteen cells; a synthetic
library mirror holds a ``paper.md`` with every quote and Table 1 row the loader states and
the miniature workbook with every Table 4 quote as a row. Every rule passes on that pair,
and each failure test edits one stored number (or one file) and names the rule that
catches it with its message.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest

from tests.torchcell.datasets.scerevisiae import _jackson2020_release as rel
from torchcell.datamodels.media import CSTARVE_JACKSON2020
from torchcell.datasets.scerevisiae import jackson2020 as j
from torchcell.verification.pseudobulk_umi_count import (
    expected_environments,
    verify_pseudobulk_umi_count_streaming,
)
from torchcell.verification.report import Provenance, VerificationReport
from torchcell.verification.runners import stream_records

PROV = Provenance(source_uri="mini", citation_key=j.CITATION_KEY, sha256="0" * 64)
SGD = {"YAL001C", "YAL002W-A", "YNCG0013W", "YNCQ0002W", "YIL167W"}


def _table4_quotes(settings_path: Path) -> list[str]:
    quotes = {
        sv.quote
        for sv in j.JACKSON_YPD.provenance
        if sv.provenance.source_uri == j.SI_XLSX
    }
    for spec in j.build_conditions(j.read_author_settings(settings_path)).values():
        quotes.update(
            sv.quote
            for sv in [
                *spec.media.provenance,
                *(sv for c in spec.media.components for sv in c.provenance),
            ]
            if sv.provenance.citation_key == j.CITATION_KEY
            and sv.provenance.source_uri == j.SI_XLSX
        )
    return sorted(quotes)


def _library(tmp_path: Path, raw: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    library = tmp_path / "library"
    key = library / j.CITATION_KEY
    (key / "si").mkdir(parents=True)
    conditions = j.build_conditions(j.read_author_settings(raw / j.SETTINGS_NAME))
    paper = "\n".join([*j.PAPER_QUOTES, *(c.table1_row for c in conditions.values())])
    (key / j.PAPER_MD).write_text(paper, encoding="utf-8")
    rel.write_workbook(key / j.SI_XLSX, _table4_quotes(raw / j.SETTINGS_NAME))
    monkeypatch.setattr(
        j,
        "PAPER_MD_SHA256",
        hashlib.sha256((key / j.PAPER_MD).read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        j, "SI_XLSX_SHA256", hashlib.sha256((key / j.SI_XLSX).read_bytes()).hexdigest()
    )
    return library


def _store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root = rel.install(tmp_path, monkeypatch)
    j.TfDeletionPseudobulkJackson2020Dataset(root=str(root), genome=rel.fake_genome())
    return root, _library(tmp_path, root / "raw", monkeypatch)


def _verify(
    root: Path, library: Path, records: Any = None, **overrides: Any
) -> VerificationReport:
    kwargs: dict[str, Any] = {
        "dataset_name": "mini",
        "provenance": PROV,
        "expected_count": rel.N_RECORDS,
        "raw_dir": root / "raw",
        "library_root": library,
        "genome": rel.fake_genome(),
        "sgd_genes": SGD,
        "min_feature_containment": 0.7,
    }
    kwargs.update(overrides)
    return verify_pseudobulk_umi_count_streaming(
        records if records is not None else stream_records(str(root)), **kwargs
    )


def _rows(report: VerificationReport) -> dict[str, Any]:
    return {r.name: r for r in report.results}


def test_every_rule_passes_on_the_miniature_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, library = _store(tmp_path, monkeypatch)
    report = _verify(root, library)
    rows = _rows(report)
    assert [name for name, r in rows.items() if not r.passed] == []
    assert list(rows) == [
        "structural",
        "count",
        "strain_condition_uniqueness",
        "strain_identity",
        "release_fidelity",
        "reference_fidelity",
        "release_totals",
        "measurement_type_consistent",
        "feature_universe",
        "conditions_documented",
        "sourced_values",
        "provenance_gaps",
        "canonical_gene_names",
        "uncertainty_sanity",
        "compound_identity",
        "media_compound_identity",
        "media_membership",
        "feature_containment",
        "feature_ids_corroborated",
    ]
    assert rows["strain_identity"].details["strains_per_gene"] == {
        "YDL227C": 2,
        "YFL021W": 1,
        "YHR006W": 1,
    }
    assert rows["feature_containment"].details["outside"] == ["YIL080W", "YIL168W"]
    assert rows["release_totals"].details == {
        "record_cells": 16,
        "record_umis": rel.N_UMIS,
        "released_cells": 16,
        "released_umis": rel.N_UMIS,
    }
    # 22 paper quotes; the two FY rows of Table S2, the Table 4 rows and the four
    # diploid rows of the workbook; the two condition-label lines of the settings file
    n_table4 = len(_table4_quotes(root / "raw" / j.SETTINGS_NAME))
    assert rows["sourced_values"].details["n_audited"] == 22 + 2 + n_table4 + 4 + 2


def _records(root: Path) -> list[dict[str, Any]]:
    """The stored records as independent copies (interned sub-objects are shared)."""
    return [copy.deepcopy(r) for r in stream_records(str(root))]


def _find(records: list[dict[str, Any]], strain: str, condition: str) -> dict[str, Any]:
    """The one record of ``strain`` in Table 1 condition ``condition``."""
    for rec in records:
        (pert,) = rec["experiment"]["genotype"]["perturbations"]
        if pert["construction"]["strain_accession"] != strain:
            continue
        if (
            rec["experiment"]["environment"]["media"]["name"] == _MEDIA_NAME[condition]
            and (
                rec["experiment"]["environment"]["duration_hours"] == _HOURS[condition]
            )
            and bool(rec["experiment"]["environment"]["perturbations"])
            == (condition == "RAPA")
        ):
            return rec
    raise KeyError((strain, condition))


_MEDIA_NAME = {"YPD": j.JACKSON_YPD.name, "CSTARVE": CSTARVE_JACKSON2020.name}
_HOURS = {"YPD": 4.0, "CSTARVE": 4.0}


def test_an_edited_count_fails_release_fidelity_by_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, library = _store(tmp_path, monkeypatch)
    records = _records(root)
    phenotype = _find(records, "gat1_A5", "YPD")["experiment"]["phenotype"]
    phenotype["umi_count"]["YAL001C"] += 1
    rows = _rows(_verify(root, library, records))
    assert rows["release_fidelity"].passed is False
    assert rows["release_fidelity"].message == (
        "1 records differ from the re-summed release"
    )
    assert rows["release_fidelity"].details["mismatches"] == [
        "gat1_A5/YPD: ['umi_count']"
    ]
    assert rows["structural"].passed is False
    records = _records(root)
    cells = _find(records, "WT(ho)_B1", "YPD")["experiment"]["phenotype"]
    cells["cell_total_umis"] = list(reversed(cells["cell_total_umis"]))
    rows = _rows(_verify(root, library, records))
    assert rows["release_fidelity"].details["mismatches"] == [
        "WT(ho)_B1/YPD: ['cell_total_umis']"
    ]
    assert rows["structural"].passed is True


def test_a_dropped_or_repeated_record_fails_the_pair_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, library = _store(tmp_path, monkeypatch)
    records = _records(root)
    target = _find(records, "stp2_A12", "YPD")
    dropped = _rows(_verify(root, library, [r for r in records if r is not target]))
    assert dropped["count"].message == f"observed 14, expected {rel.N_RECORDS}"
    assert dropped["strain_condition_uniqueness"].details["missing"] == ["stp2_A12/YPD"]
    assert dropped["release_totals"].passed is False
    assert dropped["strain_identity"].passed is False
    repeated = _rows(_verify(root, library, [*records, copy.deepcopy(target)]))
    assert repeated["strain_condition_uniqueness"].details["duplicated"] == [
        "stp2_A12/YPD"
    ]
    assert repeated["release_totals"].passed is False


def test_a_wrong_barcode_or_gene_fails_strain_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, library = _store(tmp_path, monkeypatch)
    records = _records(root)
    for rec in records:
        (pert,) = rec["experiment"]["genotype"]["perturbations"]
        if pert["construction"]["strain_accession"] == "gat1_A5":
            pert["barcode"] = "ACGTACGTACGTACGTACG"
        if pert["construction"]["strain_accession"] == "stp2_A12":
            pert["systematic_gene_name"] = "YFL021W"
    row = _rows(_verify(root, library, records))["strain_identity"]
    assert row.passed is False
    assert row.details["wrong_barcode"] == ["gat1_A5"]
    assert row.details["wrong_gene"] == ["stp2_A12"]
    assert row.details["multi"] == []


def test_an_edited_environment_or_reference_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, library = _store(tmp_path, monkeypatch)
    records = _records(root)
    _find(records, "gat1_A5", "CSTARVE")["experiment"]["environment"][
        "duration_hours"
    ] = 5.0
    rows = _rows(_verify(root, library, records))
    assert rows["conditions_documented"].passed is False
    assert rows["conditions_documented"].details["n_unmatched_records"] == 1
    records = _records(root)
    for rec in records:
        reference = rec["reference"]["phenotype_reference"]
        if reference["n_cells"] == 3:
            reference["cell_barcodes"] = list(reversed(reference["cell_barcodes"]))
    rows = _rows(_verify(root, library, records))
    assert rows["reference_fidelity"].passed is False
    assert rows["reference_fidelity"].details["mismatches"] == [
        "YPD: ['cell_barcodes']"
    ]


def test_the_measurement_type_universe_and_containment_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, library = _store(tmp_path, monkeypatch)
    records = _records(root)
    _find(records, "WT(ho)_B1", "YPD")["experiment"]["phenotype"][
        "measurement_type"
    ] = "other"
    _find(records, "gat1_A5", "YPD")["experiment"]["phenotype"]["umi_count"][
        "YZZ999W"
    ] = 1
    rows = _rows(_verify(root, library, records, min_feature_containment=0.9))
    assert rows["measurement_type_consistent"].passed is False
    assert rows["feature_universe"].details["stray_keys"] == ["YZZ999W"]
    assert rows["feature_containment"].passed is False
    assert rows["feature_containment"].message.startswith(
        "0.71429 of 7 gene features are SGD systematic names (>= 0.9); 2 are not"
    )


def test_a_missing_quote_fails_the_provenance_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, library = _store(tmp_path, monkeypatch)
    paper = library / j.CITATION_KEY / j.PAPER_MD
    text = paper.read_text(encoding="utf-8").replace(j.GEO_QUOTE, "")
    paper.write_text(text, encoding="utf-8")
    monkeypatch.setattr(
        j, "PAPER_MD_SHA256", hashlib.sha256(paper.read_bytes()).hexdigest()
    )
    rows = _rows(_verify(root, library))
    assert rows["sourced_values"].passed is False
    assert rows["sourced_values"].details["failures"] == [
        "'accession number GEO: GSE125162.': quote no longer found in paper.md"
    ]
    monkeypatch.setattr(j, "SI_XLSX_SHA256", "0" * 64)
    rows = _rows(_verify(root, library))
    assert "sha256 drift on si/si6.xlsx" in rows["sourced_values"].details["failures"]
    assert rows["conditions_documented"].passed is True


def test_a_raw_header_that_does_not_undo_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, library = _store(tmp_path, monkeypatch)
    (root / "raw" / j.BULK_NAME).unlink()
    (root / "raw" / j.BULK_NAME).write_text("\t".join(rel.BULK_HEADER[::-1]) + "\n")
    with pytest.raises(j.FeatureMapError, match="differs from TRIZOL_BULK.tsv"):
        _verify(root, library)


def test_expected_environments_key_each_condition_once(tmp_path: Path) -> None:
    path = tmp_path / "s.R"
    path.write_text(rel.SETTINGS_TEXT)
    conditions = j.build_conditions(j.read_author_settings(path))
    keyed = expected_environments(conditions)
    assert sorted(keyed.values()) == sorted(conditions)
