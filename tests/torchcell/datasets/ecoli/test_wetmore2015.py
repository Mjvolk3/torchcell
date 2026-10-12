# tests/torchcell/datasets/ecoli/test_wetmore2015.py
# [[tests.torchcell.datasets.ecoli.test_wetmore2015]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_wetmore2015.py
"""Wetmore 2015 RB-TnSeq: the subsumption record, its evidence and the raw mirror.

The synthetic tests pin the record's logic, the release comparison and the deposit
contract with frames and files built here. The ``data`` tests (``--data``) re-derive the
decision from the sha256-pinned mirrors under ``$DATA_ROOT``: the coverage counts, the
two verdict flips, the withdrawn samples, the value agreement between the releases, the
identifier finding on the deposited K-12 assemblies, and every quote.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import os.path as osp
import shutil
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import lmdb
import numpy as np
import openpyxl
import pandas as pd
import pytest

import torchcell.datasets.ecoli.wetmore2015 as w
from tests.torchcell.sequence.genome._bacterial_fixtures import (
    BW25113_LOCI,
    MG1655_GAF,
    MG1655_LOCI,
    forbid_network,
    serve_tier,
    write_assembly,
)
from torchcell.data import ManifestPinMismatchError
from torchcell.datamodels.media import (
    LB_LENNOX,
    M9_NOCARBON_WETMORE2015,
    M9_NONITROGEN_WETMORE2015,
    MOPS_RICH_DEFINED_NOCARBON_PRICE2018,
)
from torchcell.datamodels.schema import (
    AssemblyReferenceGenome,
    ConcentrationUnit,
    EnvironmentPhysicalPerturbation,
    MeasurementType,
    PhysicalFactor,
    UncertaintyType,
)
from torchcell.datasets.bacteria_common import LocusTagResolutionError
from torchcell.datasets.dataset_registry import dataset_registry
from torchcell.sequence.genome.base import GeneNameResolution, GeneNameStatus
from torchcell.sequence.genome.ecoli.k12 import (
    BW25113_ASSEMBLY,
    MG1655_ASSEMBLY,
    EckCrosswalk,
    EckPair,
    EcoliK12BW25113Genome,
    EcoliK12MG1655Genome,
)
from torchcell.verification.report import Level
from torchcell.verification.sourced import SourcedValue, audit_sourced_value

DATA_ROOT = os.environ.get("DATA_ROOT", "")
MIRRORS_PRESENT = all(
    osp.isfile(osp.join(DATA_ROOT, rel, "manifest.json"))
    for rel in (
        w.RAW_DIR_REL,
        f"{w.LIBRARY_DIR_REL}/{w.CITATION_KEY}",
        f"{w.LIBRARY_DIR_REL}/{w.SUPERSET_CITATION_KEY}",
    )
)


# --------------------------------------------------------------------------- #
# Synthetic frames
# --------------------------------------------------------------------------- #
def _experiment(
    name: str, group: str, short: str, condition: str | None
) -> dict[str, Any]:
    return {
        "name": name,
        "Group": group,
        "short": short,
        "Media": "LB" if group == "Time0" else "M9 minimal media_noCarbon",
        "Condition_1": condition,
        "Concentration_1": None if condition is None else 20.0,
        "Units_1": None if condition is None else "mM",
        "Total Generations": 6.5,
        "EndOD": 1.2,
    }


def _quality(name: str, u: bool, g_med: float, cor12: float) -> dict[str, Any]:
    return {
        "name": name,
        "u": u,
        "gMed": g_med,
        "mad12": 0.2,
        "cor12": cor12,
        "adjcor": 0.01,
        "gccor": 0.02,
    }


EXPERIMENTS = pd.DataFrame(
    [
        _experiment("set1IT001", "Time0", "Time0", None),
        _experiment("set1IT003", "carbon source", "D-Glucose (C)", "D-Glucose"),
        _experiment("set1IT029", "carbon source", "Fumarate (C)", "Sodium Fumarate"),
        _experiment("set2IT045", "LB", "LB", None),
        _experiment("set2IT050", "carbon source", "Glycerol (C)", "Glycerol"),
        _experiment("set1IT007", "carbon source", "Sucrose (C)", "Sucrose"),
    ]
)
QUALITY = pd.DataFrame(
    [
        _quality("set1IT003", True, 200.0, 0.3),
        _quality("set1IT029", True, 50.0, 0.37),
        _quality("set2IT045", False, 127.0, 0.0976),
        _quality("set2IT050", False, 20.0, 0.05),
        _quality("set1IT007", True, 150.0, 0.3),
    ]
)


def _superset_row(name: str) -> dict[str, Any]:
    """A Table S5 ``Keio`` row; a shared name copies Data Set S1's metadata."""
    source = EXPERIMENTS.set_index("name")
    metadata: dict[str, Any] = (
        {k: source.loc[name, k] for k in w.SHARED_METADATA}
        if name in source.index
        else {
            "short": "Glycerol (C)",
            "Media": "M9 minimal media_noCarbon",
            "Condition_1": "Glycerol",
            "Concentration_1": 20.0,
        }
    )
    return {"orgId": "Keio", "name": name} | metadata


SUPERSET_EXPERIMENTS = pd.DataFrame(
    [_superset_row(n) for n in ("set1IT003", "set2IT045", "set1IT007", "set3IT001")]
)
SUPERSET_QUALITY = pd.DataFrame(
    [
        _quality("set1IT029", False, 48.0, 0.368),
        _quality("set2IT045", True, 122.0, 0.109),
    ]
)


def test_record_separates_coverage_flips_and_the_withdrawn_samples() -> None:
    record = w.build_subsumption(
        EXPERIMENTS, QUALITY, SUPERSET_EXPERIMENTS, SUPERSET_QUALITY
    )

    assert [e.name for e in record.experiments] == [
        "set1IT003",
        "set1IT029",
        "set2IT045",
        "set2IT050",
        "set1IT007",
    ]
    assert record.successful == ("set1IT003", "set1IT029", "set1IT007")
    assert record.carried == ("set1IT003", "set2IT045", "set1IT007")
    assert record.covered == ("set1IT003", "set1IT007")
    assert record.not_covered == ("set1IT029",)
    assert record.disregarded == ("set1IT007",)
    assert record.superset_only == ("set3IT001",)
    assert record.superset_experiment_names == (
        "set1IT003",
        "set1IT007",
        "set2IT045",
        "set3IT001",
    )
    assert [
        (f.name, f.source.successful, f.superset.successful)
        for f in record.verdict_flips
    ] == [("set1IT029", True, False), ("set2IT045", False, True)]
    assert record.verdict_flips[0].superset.failed_rules() == ("g_med_min",)
    assert record.verdict_flips[1].source.failed_rules() == ("cor12_min",)
    glucose = record.experiments[0]
    assert (glucose.condition, glucose.concentration, glucose.units) == (
        "D-Glucose",
        20.0,
        "mM",
    )
    lb = record.experiments[2]
    assert (lb.condition, lb.concentration, lb.units) == (None, None, None)
    assert (record.strain, record.mutant_library) == ("BW25113", "KEIO_ML9")
    assert glucose.end_od == 1.2
    assert record.superset_org_id == "Keio"


def test_a_condition_sample_without_a_quality_row_is_refused() -> None:
    with pytest.raises(ValueError, match=r"no quality row: \['set1IT003'\]"):
        w.build_subsumption(
            EXPERIMENTS,
            QUALITY.loc[QUALITY["name"] != "set1IT003"],
            SUPERSET_EXPERIMENTS,
            SUPERSET_QUALITY,
        )


def test_a_repeated_sample_name_is_refused() -> None:
    doubled = pd.concat([EXPERIMENTS, EXPERIMENTS.iloc[[1]]], ignore_index=True)
    with pytest.raises(
        ValueError, match=r"Expts_Keio repeats sample names: \['set1IT003'\]"
    ):
        w.build_subsumption(doubled, QUALITY, SUPERSET_EXPERIMENTS, SUPERSET_QUALITY)


def test_a_shared_name_with_other_metadata_is_not_the_same_sample() -> None:
    renamed = SUPERSET_EXPERIMENTS.copy()
    renamed.loc[renamed["name"] == "set1IT003", "Concentration_1"] = 22.0
    with pytest.raises(ValueError, match=r"differ: \['set1IT003'\]"):
        w.build_subsumption(EXPERIMENTS, QUALITY, renamed, SUPERSET_QUALITY)


def test_a_repeated_compendium_name_is_refused() -> None:
    doubled = pd.concat(
        [SUPERSET_EXPERIMENTS, SUPERSET_EXPERIMENTS.iloc[[0]]], ignore_index=True
    )
    with pytest.raises(
        ValueError, match=r"TableS5_Experiments repeats sample names: \['set1IT003'\]"
    ):
        w.build_subsumption(EXPERIMENTS, QUALITY, doubled, SUPERSET_QUALITY)


def test_a_flip_the_compendium_never_scored_is_refused() -> None:
    with pytest.raises(
        ValueError, match="set1IT029: the compendium release scored no row"
    ):
        w.build_subsumption(
            EXPERIMENTS,
            QUALITY,
            SUPERSET_EXPERIMENTS,
            SUPERSET_QUALITY.loc[SUPERSET_QUALITY["name"] != "set1IT029"],
        )


@pytest.mark.parametrize(
    "metrics,failed",
    [
        ({"g_med": 50.0}, ()),
        ({"g_med": 49.9}, ("g_med_min",)),
        ({"mad12": 0.5}, ()),
        ({"mad12": 0.51}, ("mad12_max",)),
        ({"cor12": 0.1}, ()),
        ({"cor12": 0.0976}, ("cor12_min",)),
        ({"gccor": -0.21}, ("abs_gccor_max",)),
        ({"adjcor": -0.26}, ("abs_adjcor_max",)),
        ({"g_med": 10.0, "adjcor": 0.3}, ("g_med_min", "abs_adjcor_max")),
    ],
)
def test_failed_rules_apply_each_sourced_threshold(
    metrics: dict[str, float], failed: tuple[str, ...]
) -> None:
    base: dict[str, Any] = {
        "successful": True,
        "g_med": 100.0,
        "mad12": 0.2,
        "cor12": 0.3,
        "adjcor": 0.0,
        "gccor": 0.0,
    }
    assert w.QualityMetrics(**(base | metrics)).failed_rules() == failed


def test_superset_experiments_are_the_keio_rows_below_the_preamble(
    tmp_path: Path,
) -> None:
    rows = [
        ["This table lists all of the genome-wide fitness experiments", None, None],
        [None, None, None],
        ["orgId", "organism", "name"],
        ["Keio", "Escherichia coli BW25113", "set1IT003"],
        ["MR1", "Shewanella oneidensis MR-1", "set1H3"],
        ["Keio", "Escherichia coli BW25113", "set2IT045"],
    ]
    path = tmp_path / "s5.xlsx"
    pd.DataFrame(rows).to_excel(
        path, sheet_name=w.SUPERSET_TABLE_S5_SHEET, header=False, index=False
    )
    keio = w.read_superset_experiments(path)
    assert list(keio["name"]) == ["set1IT003", "set2IT045"]
    assert list(keio["organism"]) == ["Escherichia coli BW25113"] * 2


def test_a_table_s5_without_one_header_row_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "s5.xlsx"
    pd.DataFrame([["no header here", None]]).to_excel(
        path, sheet_name=w.SUPERSET_TABLE_S5_SHEET, header=False, index=False
    )
    with pytest.raises(ValueError, match="expected one 'orgId' header row"):
        w.read_superset_experiments(path)


def test_compare_fitness_tables_uses_shared_genes_and_samples_only() -> None:
    source = pd.DataFrame(
        {
            "locusId": [1, 2, 3, 9],
            "sysName": ["b0001", "b0002", "b0003", "b0009"],
            "set1IT003 D-Glucose (C)": [0.0, -1.0, -2.0, 5.0],
            "set1IT029 Fumarate (C)": [0.1, 0.2, 0.3, 0.4],
        }
    )
    superset = pd.DataFrame(
        {
            "locusId": [1, 2, 3, 4],
            "set1IT003 D-Glucose (C)": [0.5, -1.0, -3.0, 7.0],
            "set2IT045 LB": [0.0, 0.0, 0.0, 0.0],
        }
    )
    comparison = w.compare_fitness_tables(source, superset)
    assert (
        comparison.n_source_genes,
        comparison.n_superset_genes,
        comparison.n_source_genes_in_superset,
    ) == (4, 4, 3)
    (only,) = comparison.agreements
    assert (only.name, only.n_genes, only.max_abs_difference) == ("set1IT003", 3, 1.0)
    expected_r = pd.Series([0.0, -1.0, -2.0]).corr(pd.Series([0.5, -1.0, -3.0]))
    assert only.pearson_r == pytest.approx(expected_r, abs=1e-12)
    assert comparison.min_pearson_r == comparison.median_pearson_r == only.pearson_r


def test_a_sample_with_two_fitness_columns_is_refused() -> None:
    table = pd.DataFrame({"locusId": [1], "set1IT003 a": [0.0], "set1IT003 b": [0.0]})
    with pytest.raises(ValueError, match="source: sample set1IT003 has two columns"):
        w.compare_fitness_tables(table, table)


def test_the_module_registers_exactly_its_own_release_dataset() -> None:
    """The 2026-10-10 policy: this paper's own release is loaded, by one class."""
    assert [
        name for name, cls in dataset_registry.items() if cls.__module__ == w.__name__
    ] == ["RbTnseqWetmore2015Dataset"]


def test_sourced_vocabulary_for_the_rb_tnseq_rows() -> None:
    assert w.GENE_FITNESS_IS_LOG2.value is MeasurementType.log2_ratio
    assert w.UNCERTAINTY_TYPE.value is UncertaintyType.standard_error
    assert w.AUTHOR_WITHDRAWAL.conditions == frozenset({"Sucrose", "D-Mannitol"})
    assert w.AUTHOR_WITHDRAWAL.rule == w.DROP_WITHDRAWN
    assert w.AUTHOR_WITHDRAWAL.instruction.quote == (
        "Please disregard any of the data from this publication regarding sucrose or "
        "D-mannitol."
    )
    assert w.AUTHOR_WITHDRAWAL.cause.quote == (
        "After publishing this data, Morgan Price and Adam Deutschbauer discovered "
        "that our stock solutions for sucrose and D-mannitol were problematic."
    )
    assert (
        w.AUTHOR_WITHDRAWAL.instruction.provenance.sha256
        == w.AUTHOR_WITHDRAWAL.cause.provenance.sha256
        == w.RAW_FILES[w.SUPERSET_TOP_PAGE].sha256
    )
    assert w.STRAIN.value == w.STRAIN_TABLE_S1.value == w.REFERENCE_STRAIN == "BW25113"
    assert w.SIGNIFICANT_ABS_T.value == 4.0
    assert w.T_STATISTIC_SIGMA.value == 0.1
    assert w.N_SAMPLES_GAP.field == "n_samples"


# --------------------------------------------------------------------------- #
# The raw-mirror contract (synthetic pins)
# --------------------------------------------------------------------------- #
class _FrozenDatetime:
    @staticmethod
    def now(tz: Any = None) -> datetime:
        return datetime(2026, 10, 7, 12, 0, tzinfo=UTC)


def _synthetic_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, bytes]]:
    """Two fake release files, the module's pins pointed at them."""
    payloads = {"data/a/one.tab": b"first file", "data/b/two.html": b"second file"}
    source = tmp_path / "src"
    raw_files = {}
    for rel, payload in payloads.items():
        (source / rel).parent.mkdir(parents=True, exist_ok=True)
        (source / rel).write_bytes(payload)
        raw_files[rel] = w.RawFile(
            relpath=rel,
            url=f"https://example.org/{rel}",
            sha256=hashlib.sha256(payload).hexdigest(),
            purpose="test",
            retrieved_at="2026-10-07",
        )
    monkeypatch.setattr(w, "RAW_FILES", raw_files)
    monkeypatch.setattr(w, "datetime", _FrozenDatetime)
    return source, payloads


def test_deposit_writes_the_files_and_an_exact_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, payloads = _synthetic_release(tmp_path, monkeypatch)
    data_root = tmp_path / "dr"

    root = w.deposit_raw_mirror(source_dir=source, data_root=str(data_root))

    assert root == data_root / "torchcell-raw" / w.CITATION_KEY
    for rel, payload in payloads.items():
        assert (root / rel).read_bytes() == payload
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["citation_key"] == w.CITATION_KEY
    assert manifest["doi"] == "10.1128/mBio.00306-15"
    assert manifest["created_at"] == "2026-10-07T12:00:00+00:00"
    assert manifest["files"] == [
        {
            "path": rel,
            "role": "raw_data",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "source": f"https://example.org/{rel}",
            "zotero_md5": None,
            "retrieval": {
                "method": "direct_url",
                "source_url": f"https://example.org/{rel}",
                "retriever": "torchcell.literature.retrieve.direct_url",
                "params": {"url": f"https://example.org/{rel}"},
                "sha256": hashlib.sha256(payload).hexdigest(),
                "retrieved_at": "2026-10-07",
                "last_check": None,
            },
            "processing": None,
        }
        for rel, payload in payloads.items()
    ]
    pin = hashlib.sha256(b"first file").hexdigest()
    assert w.manifest_sha256(w.load_manifest(str(data_root)), "data/a/one.tab") == pin
    assert w.raw_path("data/a/one.tab", str(data_root)) == root / "data/a/one.tab"


def test_a_second_deposit_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = _synthetic_release(tmp_path, monkeypatch)
    data_root = str(tmp_path / "dr")
    root = w.deposit_raw_mirror(source_dir=source, data_root=data_root)
    stamps = {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}

    w.deposit_raw_mirror(source_dir=source, data_root=data_root)

    assert {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()} == stamps


def test_a_source_off_its_pin_is_refused_before_the_mirror_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = _synthetic_release(tmp_path, monkeypatch)
    (source / "data/b/two.html").write_bytes(b"drifted upstream")
    data_root = tmp_path / "dr"
    with pytest.raises(RuntimeError, match="two.html sha256 mismatch"):
        w.deposit_raw_mirror(source_dir=source, data_root=str(data_root))
    assert not data_root.exists()


def test_a_differing_mirror_file_is_refused_before_anything_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = _synthetic_release(tmp_path, monkeypatch)
    root = w.raw_mirror_dir(str(tmp_path / "dr"))
    (root / "data/b").mkdir(parents=True)
    (root / "data/b/two.html").write_bytes(b"other bytes")
    with pytest.raises(RuntimeError, match="two.html exists with a different sha256"):
        w.deposit_raw_mirror(source_dir=source, data_root=str(tmp_path / "dr"))
    assert not (root / "data/a/one.tab").exists()
    assert not (root / "manifest.json").exists()


def test_a_manifest_recording_other_retrievals_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = _synthetic_release(tmp_path, monkeypatch)
    data_root = str(tmp_path / "dr")
    root = w.deposit_raw_mirror(source_dir=source, data_root=data_root)
    original = (root / "manifest.json").read_text()
    monkeypatch.setitem(
        w.RAW_FILES,
        "data/a/one.tab",
        w.RAW_FILES["data/a/one.tab"].model_copy(update={"retrieved_at": "2027-01-01"}),
    )
    with pytest.raises(RuntimeError, match="records other files or retrievals"):
        w.deposit_raw_mirror(source_dir=source, data_root=data_root)
    assert (root / "manifest.json").read_text() == original


def test_a_deposit_extends_a_manifest_whose_records_it_repeats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loader's files joined the 2026-10-07 record this way: old entries verbatim,
    the new one appended with its own retrieval date, the first created_at kept.
    """
    source, _ = _synthetic_release(tmp_path, monkeypatch)
    data_root = str(tmp_path / "dr")
    first = dict(w.RAW_FILES)
    monkeypatch.setattr(w, "RAW_FILES", {"data/a/one.tab": first["data/a/one.tab"]})
    root = w.deposit_raw_mirror(source_dir=source, data_root=data_root)
    before = json.loads((root / "manifest.json").read_text())
    assert [f["path"] for f in before["files"]] == ["data/a/one.tab"]
    monkeypatch.setattr(
        w,
        "RAW_FILES",
        first
        | {
            "data/b/two.html": first["data/b/two.html"].model_copy(
                update={"retrieved_at": "2026-10-10"}
            )
        },
    )

    w.deposit_raw_mirror(source_dir=source, data_root=data_root)

    after = json.loads((root / "manifest.json").read_text())
    assert after["created_at"] == before["created_at"]
    assert after["files"][0] == before["files"][0]
    assert [f["path"] for f in after["files"]] == ["data/a/one.tab", "data/b/two.html"]
    assert after["files"][1]["retrieval"]["retrieved_at"] == "2026-10-10"
    assert (root / "data/b/two.html").read_bytes() == b"second file"


def test_an_extension_refuses_a_manifest_without_a_creation_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = _synthetic_release(tmp_path, monkeypatch)
    data_root = str(tmp_path / "dr")
    root = w.deposit_raw_mirror(source_dir=source, data_root=data_root)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["created_at"] = None
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="records no created_at"):
        w.deposit_raw_mirror(source_dir=source, data_root=data_root)


def test_raw_path_refuses_a_manifest_off_the_module_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = _synthetic_release(tmp_path, monkeypatch)
    data_root = str(tmp_path / "dr")
    w.deposit_raw_mirror(source_dir=source, data_root=data_root)
    monkeypatch.setitem(
        w.RAW_FILES,
        "data/a/one.tab",
        w.RAW_FILES["data/a/one.tab"].model_copy(update={"sha256": "0" * 64}),
    )
    with pytest.raises(ManifestPinMismatchError):
        w.raw_path("data/a/one.tab", data_root)


# --------------------------------------------------------------------------- #
# The real mirrors (--data)
# --------------------------------------------------------------------------- #
def _sourced_values() -> list[SourcedValue]:
    return [
        value for name in dir(w) if isinstance(value := getattr(w, name), SourcedValue)
    ]


@pytest.mark.parametrize(
    "name", [n for n in dir(w) if isinstance(getattr(w, n), SourcedValue)]
)
@pytest.mark.data
@pytest.mark.skipif(not MIRRORS_PRESENT, reason="requires the mirrors")
def test_every_sourced_value_is_a_verbatim_quote_of_its_pinned_file(name: str) -> None:
    value = getattr(w, name)
    from_raw = value.provenance.source_uri.startswith("data/")
    root = osp.join(DATA_ROOT, w.RAW_ROOT_REL if from_raw else w.LIBRARY_DIR_REL)
    path = value.source_path(root)
    if path.suffix == ".xlsx":
        assert hashlib.sha256(path.read_bytes()).hexdigest() == value.provenance.sha256
        assert value.quote in w.xlsx_text(str(path)), name
        return
    assert audit_sourced_value(value, root).passed


def test_the_sourced_values_cover_both_papers_and_the_release() -> None:
    values = _sourced_values()
    assert len(values) >= 30
    assert {v.provenance.citation_key for v in values} == {
        w.CITATION_KEY,
        w.SUPERSET_CITATION_KEY,
    }
    assert {v.provenance.source_uri for v in values} == {
        w.PAPER_MD,
        w.TABLE_S1_MD,
        w.DATA_SET_S1,
        w.RELEASE_PAGE,
        w.SUPERSET_PAGE,
        w.SUPERSET_TOP_PAGE,
    }


@pytest.fixture(scope="module")
def record() -> w.Wetmore2015Subsumption:
    return w.subsumption_record()


@pytest.mark.data
@pytest.mark.skipif(not MIRRORS_PRESENT, reason="requires the mirrors")
def test_the_compendium_carries_91_of_92_successes_and_one_rejected_sample(
    record: w.Wetmore2015Subsumption,
) -> None:
    assert len(record.experiments) == 101
    assert len(record.successful) == 92
    assert len(record.covered) == 91
    assert record.not_covered == ("set1IT029",)
    assert len(record.carried) == 92
    assert sorted(set(record.carried) - set(record.covered)) == ["set2IT045"]
    assert len(record.superset_experiment_names) == 162
    assert len(record.superset_only) == 70
    assert record.disregarded == ("set1IT007", "set1IT008", "set1IT043", "set1IT044")
    # The sucrose samples grew, which BW25113 should not do on sucrose (the 2021 note).
    end_od = {e.name: e.end_od for e in record.experiments}
    assert (end_od["set1IT007"], end_od["set1IT008"]) == (1.26, 1.24)


@pytest.mark.data
@pytest.mark.skipif(not MIRRORS_PRESENT, reason="requires the mirrors")
def test_the_two_verdict_flips_and_the_rule_each_trips(
    record: w.Wetmore2015Subsumption,
) -> None:
    flips = {f.name: f for f in record.verdict_flips}
    assert sorted(flips) == ["set1IT029", "set2IT045"]
    fumarate, lb = flips["set1IT029"], flips["set2IT045"]
    assert (fumarate.source.g_med, fumarate.superset.g_med) == (50.0, 48.0)
    assert (fumarate.source.failed_rules(), fumarate.superset.failed_rules()) == (
        (),
        ("g_med_min",),
    )
    assert (lb.source.failed_rules(), lb.superset.failed_rules()) == (
        ("cor12_min",),
        (),
    )
    # The replicate failed both releases, so the compendium has no fumarate sample.
    replicate = next(e for e in record.experiments if e.name == "set1IT030")
    assert replicate.source_quality.failed_rules() == ("g_med_min",)
    assert not replicate.in_superset
    assert not any(
        e.in_superset for e in record.experiments if e.short == "Fumarate (C)"
    )


@pytest.mark.data
@pytest.mark.skipif(not MIRRORS_PRESENT, reason="requires the mirrors")
def test_the_releases_agree_sample_by_sample_over_a_gene_superset() -> None:
    comparison = w.release_comparison()
    assert (
        comparison.n_source_genes,
        comparison.n_superset_genes,
        comparison.n_source_genes_in_superset,
    ) == (3646, 3789, 3646)
    assert len(comparison.agreements) == 91
    assert round(comparison.min_pearson_r, 4) == 0.9890
    assert round(comparison.median_pearson_r, 4) == 0.9971


@pytest.mark.data
@pytest.mark.skipif(not MIRRORS_PRESENT, reason="requires the mirrors")
def test_this_papers_mannitol_samples_show_the_withdrawn_symptom() -> None:
    """MtlA and mtlD do not matter and manXYZ do, the 2021 note's diagnosis."""
    fitness = pd.read_csv(w.raw_path(w.RELEASE_FITNESS), sep="\t")
    genes = pd.read_csv(w.raw_path(w.RELEASE_GENES), sep="\t")
    by_symbol = dict(zip(genes["name"], genes["locusId"], strict=True))
    table = fitness.set_index("locusId")
    for column in ("set1IT043 D-Mannitol (C)", "set1IT044 D-Mannitol (C)"):
        value = table[column]
        for symbol in ("mtlA", "mtlD"):
            assert abs(float(value[by_symbol[symbol]])) < 0.2
        for symbol in ("manX", "manY", "manZ"):
            assert float(value[by_symbol[symbol]]) < -2.0


@pytest.mark.data
@pytest.mark.skipif(not MIRRORS_PRESENT, reason="requires the mirrors")
def test_a_deposit_onto_the_real_mirror_is_a_no_op() -> None:
    root = w.raw_mirror_dir()
    stamps = {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
    assert w.deposit_raw_mirror(source_dir=root) == root
    assert {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()} == stamps


TIER_PRESENT = all(
    osp.isfile(osp.join(DATA_ROOT, "data/ecoli", strain, "genome/data.db"))
    for strain in ("mg1655", "bw25113")
)


@pytest.mark.data
@pytest.mark.skipif(
    not (MIRRORS_PRESENT and TIER_PRESENT),
    reason="requires the mirrors and the built K-12 data.db caches",
)
def test_the_gene_table_is_mg1655_named_for_a_bw25113_library() -> None:
    finding = w.release_identifier_finding()
    assert (finding.n_genes, finding.scaffold_ids, finding.b_number_sysnames) == (
        3789,
        (7023,),
        3789,
    )
    assert finding.mg1655_by_sysname.status_histogram == {
        "current": 3660,
        "renamed": 1,
        "non_gene_feature": 124,
        "retired": 3,
        "ambiguous": 1,
    }
    assert finding.bw25113_by_sysname.resolved_fraction == 0.0
    assert finding.bw25113_by_symbol.status_histogram == {
        "current": 0,
        "renamed": 3514,
        "non_gene_feature": 158,
        "retired": 111,
        "ambiguous": 6,
    }
    assert (finding.eck_one_to_one, finding.eck_numeric_disagreements) == (3768, 1)


# =========================================================================== #
# The loader: this paper's own release (owner policy 2026-10-10)
# =========================================================================== #
def _s1_row(
    name: str,
    group: str,
    media: str,
    condition: str | None,
    concentration: float | None,
    units: str | None,
    *,
    temperature: float = 37.0,
    generations: float | None = 6.5,
    library: str = "Keio_ML9",
    liquid: str = "Liquid",
    aerobic: str = "Aerobic",
    condition_2: str | None = None,
) -> dict[str, Any]:
    """One Data Set S1 ``Expts_Keio`` row, with the columns the loader reads."""
    return {
        "Setname": f"Keio_ML9_{name[:4]}",
        "Mutant Library": library,
        "Media": media,
        "Group": group,
        "Temperature (°C)": temperature,
        "Liquid_v_solid": liquid,
        "Aerobic_v_Aerobic": aerobic,
        "Condition_1": condition,
        "Concentration_1": concentration,
        "Units_1": units,
        "Condition_2": condition_2,
        "EndOD": 1.2,
        "Total Generations": generations,
        "name": name,
        "short": f"{condition} ({group})" if condition else group,
    }


def _s1_quality(name: str, short: str, g_med: float = 200.0) -> dict[str, Any]:
    """A quality row whose ``u`` is what the paper's rules make of its metrics."""
    metrics = {"gMed": g_med, "mad12": 0.2, "cor12": 0.3, "adjcor": 0.01, "gccor": 0.02}
    successful = g_med >= 50.0
    return {"name": name, "short": short, **metrics, "u": successful}


_M9C = "M9 minimal media_noCarbon"
_M9N = "M9 minimal media_noNitrogen"
_MOPS = "MOPS Rich Defined media_noCarbon"
#: Time0, a glucose sample, a withdrawn sucrose sample, a failed fumarate sample, a MOPS
#: Rich Defined sample, a nitrogen source, casamino acids in mg/mL, and an LB sample at
#: 28 C with no Total Generations.
S1_ROWS = [
    _s1_row("set1IT001", "Time0", "LB", None, None, None),
    _s1_row("set1IT003", "carbon source", _M9C, "D-Glucose", 20.0, "mM"),
    _s1_row("set1IT007", "carbon source", _M9C, "Sucrose", 20.0, "mM"),
    _s1_row("set1IT030", "carbon source", _M9C, "Sodium Fumarate dibasic", 20.0, "mM"),
    _s1_row("set1IT067", "carbon source", _MOPS, "D-Glucose", 22.0, "mM"),
    _s1_row("set1IT071", "nitrogen source", _M9N, "L-Arginine", 10.0, "mM"),
    _s1_row("set1IT059", "carbon source", _M9C, "casamino acids", 2.0, "mg/mL"),
    _s1_row(
        "set2IT069", "LB", "LB", None, None, None, temperature=28.0, generations=None
    ),
]
S1_QUALITY = [
    _s1_quality("set1IT001", "Time0"),
    _s1_quality("set1IT003", "D-Glucose (C)"),
    _s1_quality("set1IT007", "Sucrose (C)"),
    _s1_quality("set1IT030", "Fumarate (C)", g_med=49.0),
    _s1_quality("set1IT067", "D-Glucose (C) MOPS"),
    _s1_quality("set1IT071", "L-Arginine (N)"),
    _s1_quality("set1IT059", "casaminos (C)"),
    _s1_quality("set2IT069", "LB"),
]
#: The samples the paper's rules pass, in Data Set S1 order (the good table's columns).
SUCCESSFUL = [
    "set1IT003",
    "set1IT007",
    "set1IT067",
    "set1IT071",
    "set1IT059",
    "set2IT069",
]


def _specs(rows: list[dict[str, Any]] | None = None) -> tuple[w.SampleSpec, ...]:
    return w.classify_samples(
        pd.DataFrame(S1_ROWS if rows is None else rows), pd.DataFrame(S1_QUALITY)
    )


def test_classify_samples_refuses_only_the_withdrawn_sample() -> None:
    specs = {s.name: s for s in _specs()}
    assert list(specs) == [r["name"] for r in S1_ROWS[1:]]
    # #1057: the sample the quality rules fail is kept, its verdict on its spec
    assert {n: s.drop_rule for n, s in specs.items()} == {
        "set1IT003": None,
        "set1IT007": w.DROP_WITHDRAWN,
        "set1IT030": None,
        # stored since #1045 row 16 gave the label a library entry
        "set1IT067": None,
        "set1IT071": None,
        "set1IT059": None,
        "set2IT069": None,
    }
    assert specs["set1IT030"].quality.failed_rules() == ("g_med_min",)
    assert specs["set1IT030"].quality.successful is False
    assert specs["set2IT069"].total_generations is None
    assert specs["set2IT069"].temperature_c == 28.0
    assert specs["set1IT059"].units == "mg/mL"
    assert specs["set1IT003"].screen_id == "Keio:set1IT003"


def test_a_mops_rich_sample_is_stored_on_price_table_s18s_recipe() -> None:
    """#1045 row 16: Data Set S1's own row for the label prints unusable micronutrients.

    The label maps to Price 2018 Table S18's recipe for the same medium (the same lab's
    later table, which re-releases these samples), and the carbon-source group accepts
    it, so the sample's Condition_1 is the carbon factor on the carbon-free medium.
    """
    assert w.MEDIA_BY_LABEL[_MOPS] is MOPS_RICH_DEFINED_NOCARBON_PRICE2018
    assert _MOPS in w.GROUP_MEDIA["carbon_source"]
    assert not hasattr(w, "DROP_MEDIUM_NOT_IN_LIBRARY")
    assert "medium_not_in_media_library" not in w.DROP_RULES


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"Mutant Library": "Keio_ML9a"}, "Mutant Library is 'Keio_ML9a', not"),
        ({"Liquid_v_solid": "Solid"}, "Liquid_v_solid is 'Solid', not 'Liquid'"),
        ({"Aerobic_v_Aerobic": "Anaerobic"}, "Aerobic_v_Aerobic is 'Anaerobic'"),
        ({"Condition_2": "Glycerol"}, "a second condition 'Glycerol'"),
        ({"Media": "marine broth"}, "unknown medium 'marine broth'"),
        # #1057: a stress row is the released medium with no compound, so it must be LB
        ({"Group": "stress"}, "group 'stress' on medium"),
        ({"Media": _M9N}, "group 'carbon source' on medium"),
    ],
)
def test_classify_samples_refuses_what_the_release_does_not_state(
    override: dict[str, Any], message: str
) -> None:
    rows = [dict(r) for r in S1_ROWS]
    rows[1].update(override)
    with pytest.raises(ValueError, match=message):
        _specs(rows)


def test_classify_samples_refuses_a_verdict_its_rules_do_not_reproduce() -> None:
    quality = pd.DataFrame(S1_QUALITY)
    quality.loc[quality["name"] == "set1IT030", "u"] = True
    with pytest.raises(ValueError, match=r"set1IT030: u = True but the paper's rules"):
        w.classify_samples(pd.DataFrame(S1_ROWS), quality)
    with pytest.raises(ValueError, match=r"no quality row: \['set1IT003'\]"):
        w.classify_samples(
            pd.DataFrame(S1_ROWS), quality.loc[quality["name"] != "set1IT003"]
        )


def _good_frames(
    samples: list[str], genes: list[str], *, extra: tuple[str, ...] = ()
) -> dict[str, pd.DataFrame]:
    """A release over ``genes`` x ``samples`` whose t follows the paper's rule exactly.

    ``extra`` samples appear in the t and SE tables only (failed samples, Time0), as
    in the real release.
    """
    every = [*samples, *extra]
    grid = np.arange(len(genes) * len(every), dtype=np.float64).reshape(len(genes), -1)
    fit = grid / 8.0 - 2.0
    obs = 0.1 + grid / 100.0
    naive = 0.3 - grid / 200.0
    t = fit / np.sqrt(0.1**2 + np.maximum(obs, naive) ** 2)
    ids = {"locusId": list(range(100, 100 + len(genes))), "sysName": genes}

    def frame(values: np.ndarray[Any, Any], names: list[str]) -> pd.DataFrame:
        columns = {f"{s} desc": values[:, every.index(s)] for s in names}
        return pd.DataFrame({**ids, "desc": "a gene", **columns})

    return {
        "fit_logratios_good.tab": frame(fit, samples),
        "fit_logratios.tab": frame(fit, every),
        "fit_t.tab": frame(t, every),
        "fit_standard_error_obs.tab": frame(obs, every),
        "fit_standard_error_naive.tab": frame(naive, every),
    }


def _align(frames: dict[str, pd.DataFrame], samples: list[str]) -> w.ReleaseTables:
    return w.align_release(
        fitness=frames["fit_logratios_good.tab"],
        t=frames["fit_t.tab"],
        se_obs=frames["fit_standard_error_obs.tab"],
        se_naive=frames["fit_standard_error_naive.tab"],
        samples=samples,
        sigma=0.1,
    )


def test_align_release_takes_the_larger_se_and_checks_the_released_t() -> None:
    frames = _good_frames(["setA1", "setA2"], ["b0001", "b0002"], extra=("setA3",))
    release = _align(frames, ["setA2", "setA1"])
    assert release.samples == ("setA2", "setA1")
    assert release.b_numbers == ("b0001", "b0002")
    # grid columns: setA1 -> 0, setA2 -> 1; genes are rows of three columns
    assert release.fitness.tolist() == [[-1.875, -2.0], [-1.5, -1.625]]
    assert release.standard_error.tolist() == [
        [pytest.approx(0.295), 0.3],
        [pytest.approx(0.28), pytest.approx(0.285)],
    ]
    assert release.n_values == 4
    assert release.n_naive_larger == 4
    assert release.t_max_abs_deviation < 1e-12
    expected_t = release.fitness / np.sqrt(0.01 + release.standard_error**2)
    np.testing.assert_allclose(release.t, expected_t, rtol=0, atol=1e-12)


def test_align_release_refuses_inconsistent_tables() -> None:
    def refused(frames: dict[str, pd.DataFrame], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            _align(frames, ["setA1", "setA2"])

    frames = _good_frames(["setA1", "setA2"], ["b0001", "b0002"])
    frames["fit_t.tab"].loc[0, "sysName"] = "b0009"
    refused(frames, "t: gene rows differ")
    frames = _good_frames(["setA1"], ["b0001"], extra=("setA2",))
    refused(frames, r"fitness: kept samples absent \['setA2'\]")
    frames = _good_frames(["setA1", "setA2"], ["b0001"])
    frames["fit_standard_error_obs.tab"] = frames["fit_standard_error_obs.tab"].drop(
        columns=["setA2 desc"]
    )
    refused(frames, r"se_obs: samples absent \['setA2'\]")
    frames = _good_frames(["setA1", "setA2"], ["b0001"])
    frames["fit_t.tab"] = frames["fit_t.tab"].rename(
        columns={"setA2 desc": "setA2 other"}
    )
    refused(frames, r"t: header text differs for \['setA2'\]")
    frames = _good_frames(["setA1", "setA2"], ["b0001"])
    frames["fit_logratios_good.tab"].loc[0, "setA1 desc"] = np.nan
    refused(frames, "fitness: 1 missing values")
    frames = _good_frames(["setA1", "setA2"], ["b0001"])
    frames["fit_standard_error_naive.tab"].loc[0, "setA1 desc"] = -1.0
    frames["fit_standard_error_obs.tab"].loc[0, "setA1 desc"] = 0.0
    refused(frames, "a standard-error estimate is not positive")
    frames = _good_frames(["setA1", "setA2"], ["b0001"])
    frames["fit_t.tab"].loc[0, "setA1 desc"] = 99.0
    refused(frames, r"the released t is not fit / sqrt\(0.1\^2 \+ SE\^2\)")
    duplicated = _good_frames(["setA1"], ["b0001"])["fit_t.tab"]
    duplicated["setA1 again"] = duplicated["setA1 desc"]
    with pytest.raises(ValueError, match="t: sample setA1 has two columns"):
        w._fitness_columns(duplicated, "t")


def test_build_environment_reads_each_sample_from_its_own_columns() -> None:
    specs = {s.name: s for s in _specs()}
    glucose = w.build_environment(specs["set1IT003"])
    assert glucose.media == M9_NOCARBON_WETMORE2015
    assert glucose.temperature is not None
    assert glucose.temperature.value == 37.0
    assert glucose.duration_generations == 6.5
    assert [g.field for g in glucose.provenance_gaps] == ["duration_hours"]
    (source,) = glucose.perturbations
    assert isinstance(source, EnvironmentPhysicalPerturbation)
    assert source.magnitude is not None
    assert (source.factor, source.magnitude.value, source.magnitude.unit) == (
        PhysicalFactor.carbon_source,
        20.0,
        ConcentrationUnit.millimolar,
    )
    assert source.agent is not None
    assert source.agent.name == "D-glucose"
    arginine = w.build_environment(specs["set1IT071"])
    assert arginine.media == M9_NONITROGEN_WETMORE2015
    (nitrogen,) = arginine.perturbations
    assert isinstance(nitrogen, EnvironmentPhysicalPerturbation)
    assert nitrogen.factor == PhysicalFactor.nitrogen_source
    (casamino,) = w.build_environment(specs["set1IT059"]).perturbations
    assert isinstance(casamino, EnvironmentPhysicalPerturbation)
    assert casamino.magnitude is not None
    assert (casamino.magnitude.value, casamino.magnitude.unit) == (
        2.0,
        ConcentrationUnit.g_per_l,
    )
    lb = w.build_environment(specs["set2IT069"])
    assert (lb.media, lb.perturbations, lb.duration_generations) == (
        LB_LENNOX,
        [],
        None,
    )
    assert [g.field for g in lb.provenance_gaps] == [
        "duration_hours",
        "duration_generations",
    ]
    assert lb.temperature is not None
    assert lb.temperature.value == 28.0


def test_build_environment_refuses_a_dropped_or_malformed_sample() -> None:
    specs = {s.name: s for s in _specs()}
    with pytest.raises(ValueError, match="set1IT007 is dropped"):
        w.build_environment(specs["set1IT007"])
    with pytest.raises(ValueError, match="a plain sample names 'x'"):
        w.build_environment(specs["set2IT069"].model_copy(update={"condition": "x"}))
    with pytest.raises(ValueError, match="a carbon source sample names no condition"):
        w.build_environment(specs["set1IT003"].model_copy(update={"condition": None}))
    with pytest.raises(ValueError, match="'D-Glucose' has no dose"):
        w.build_environment(specs["set1IT003"].model_copy(update={"units": None}))
    with pytest.raises(ValueError, match="unknown unit 'uM'"):
        w.build_environment(specs["set1IT003"].model_copy(update={"units": "uM"}))


@pytest.fixture
def k12(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[EcoliK12MG1655Genome, EcoliK12BW25113Genome]:
    """Both synthetic K-12 genomes; the network refuses."""
    files = write_assembly(tmp_path / "tier", MG1655_ASSEMBLY, MG1655_LOCI, MG1655_GAF)
    files |= write_assembly(tmp_path / "tier", BW25113_ASSEMBLY, BW25113_LOCI)
    forbid_network(monkeypatch)
    serve_tier(monkeypatch, files)
    return (
        EcoliK12MG1655Genome(genome_root=str(tmp_path / "mg1655"), overwrite=True),
        EcoliK12BW25113Genome(genome_root=str(tmp_path / "bw25113"), overwrite=True),
    )


#: The release's gene table over the synthetic genes, its own ``name`` column; b0004 and
#: b0099 carry none, and b4999 is a gene the fitness table does not value.
GENE_TABLE = pd.DataFrame(
    {
        "locusId": [100, 101, 102, 103, 104, 105, 106, 107],
        "sysName": [
            "b0001",
            "b0002",
            "b0003",
            "b0004",
            "b0005",
            "b0006",
            "b0099",
            "b4999",
        ],
        "name": ["thrL", "thrA", "thrB", None, "proB", "proC", None, "yzzZ"],
    }
)


#: Four map (b0003 crosses numbers, b0004 is a pseudogene); b0005 is not one-to-one,
#: b0006 is absent from BW25113, b0099 is not an MG1655 locus.
SYNTHETIC_GENES = ["b0001", "b0002", "b0003", "b0004", "b0005", "b0006", "b0099"]


#: The shared verifier's genome-reference rules. Under the owner rule (#1057) a gene no
#: ECK pair places is stored as the released MG1655 b-number, which is no BW25113 locus,
#: so these may read it as a failure until the verifier learns the status (a separate
#: branch); the tests accept them failing, and nothing else.
GENOME_REFERENCE_RULES = frozenset(
    {"canonical_gene_names", "gene_containment_sgd", "current_genome_genes"}
)


def test_map_genes_places_one_to_one_eck_pairs_and_types_the_rest(
    k12: tuple[EcoliK12MG1655Genome, EcoliK12BW25113Genome],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mg1655, bw25113 = k12
    with pytest.raises(LocusTagResolutionError, match="places 4 of 7 genes"):
        w.map_genes(SYNTHETIC_GENES, mg1655, bw25113, label="synthetic")
    monkeypatch.setattr(w, "MIN_ECK_ROUTE_FRACTION", 0.5)
    mapping, report = w.map_genes(SYNTHETIC_GENES, mg1655, bw25113, label="synthetic")
    assert {b: (m.locus_tag, m.perturbed_gene_name) for b, m in mapping.items()} == {
        "b0001": ("BW25113_0001", "thrL"),
        "b0002": ("BW25113_0002", "thrA"),
        "b0003": ("BW25113_4412", "hokC"),
        "b0004": ("BW25113_0004", "BW25113_0004"),
        # #1057: stored as released in the MG1655 namespace, named by MG1655's symbol
        "b0005": ("b0005", "proB"),
        "b0006": ("b0006", "proC"),
        "b0099": ("b0099", "b0099"),
    }
    assert {b: mapping[b].gene_namespace for b in report.released_as} == {
        "b0005": "ecoli_k12_mg1655_bnumber",
        "b0006": "ecoli_k12_mg1655_bnumber",
        "b0099": "ecoli_k12_mg1655_bnumber",
    }
    # the crosswalk is a derived artifact (#1057): no record reads it
    assert report.released_as == ("b0005", "b0006", "b0099")
    assert report.n_mapped == 4
    assert report.numeric_disagreements == (("b0003", "BW25113_4412", "ECK0003"),)
    assert [(u.b_number, u.reason, u.eck) for u in report.unmapped] == [
        ("b0005", "eck_not_one_to_one", ("ECK0005",)),
        ("b0006", "eck_absent_from_bw25113", ("ECK0006",)),
        ("b0099", "b_number_not_in_mg1655_annotation", ()),
    ]
    assert report.unmapped_by_reason == {
        "eck_not_one_to_one": 1,
        "eck_absent_from_bw25113": 1,
        "b_number_not_in_mg1655_annotation": 1,
    }
    assert (report.n_symbol_names, report.n_tag_names) == (3, 1)
    assert report.reconcile_layer_histogram["gene synonym"] == 4


def test_unmapped_reason_covers_every_way_a_pair_is_missing() -> None:
    crosswalk = EckCrosswalk(
        shared=("ECK0001", "ECK0005"),
        pairs=(EckPair(eck="ECK0001", mg1655="b0001", bw25113="BW25113_0001"),),
        mg1655_only=("ECK0006",),
        bw25113_only=(),
    )
    ecks = {"b0001": ("ECK0001",), "b0007": (), "b0008": ("ECK0008",)}
    assert w.unmapped_reason("b0007", crosswalk, ecks).reason == (
        "mg1655_locus_has_no_eck_synonym"
    )
    with pytest.raises(ValueError, match="b0001 has a one-to-one pair"):
        w.unmapped_reason("b0001", crosswalk, ecks)
    with pytest.raises(ValueError, match="is paired, absent and not shared"):
        w.unmapped_reason("b0008", crosswalk, ecks)


def test_gene_symbol_keeps_a_symbol_only_when_it_resolves_back() -> None:
    def resolve(name: str) -> GeneNameResolution:
        return GeneNameResolution(
            input_name=name,
            status=GeneNameStatus.RENAMED
            if name == "thrL"
            else GeneNameStatus.AMBIGUOUS,
            systematic_name="BW25113_0001" if name == "thrL" else None,
        )

    assert w.gene_symbol("thrL", "BW25113_0001", resolve) == "thrL"
    assert w.gene_symbol("dup", "BW25113_0002", resolve) == "BW25113_0002"
    assert w.gene_symbol(None, "BW25113_0003", resolve) == "BW25113_0003"


def test_build_phenotype_stores_the_t_and_the_papers_se() -> None:
    kept = w.build_phenotype(-1.5, 0.25, -5.57, screen_id="Keio:set1IT003")
    assert (
        kept.environment_response,
        kept.environment_response_se,
        kept.environment_response_uncertainty_type,
        kept.environment_response_t_statistic,
    ) == (-1.5, 0.25, UncertaintyType.standard_error, -5.57)
    assert [g.field for g in kept.provenance_gaps] == ["n_samples", "sample_unit"]


def test_the_withdrawal_outranks_the_quality_and_medium_rules() -> None:
    """A withdrawn sample is refused for the withdrawal, whatever else it trips.

    The authors' instruction is about the compound, so it holds even for a sample that
    would have been dropped anyway; the ledger must name the reason a reader has to act
    on, and it is the same rule the compendium loader refuses under.
    """
    rows = [dict(r) for r in S1_ROWS]
    quality = [dict(q) for q in S1_QUALITY]
    for row in rows:
        if row["name"] == "set1IT067":
            row["Condition_1"] = "D-Mannitol"
    for entry in quality:
        if entry["name"] == "set1IT030":
            entry["short"] = "Sucrose (C)"
    for row in rows:
        if row["name"] == "set1IT030":
            row["Condition_1"] = "Sucrose"
    specs = {
        spec.name: spec
        for spec in w.classify_samples(pd.DataFrame(rows), pd.DataFrame(quality))
    }
    assert specs["set1IT030"].drop_rule == w.DROP_WITHDRAWN
    assert specs["set1IT067"].drop_rule == w.DROP_WITHDRAWN
    assert specs["set1IT030"].quality.failed_rules() == ("g_med_min",)


def _write_s1(path: Path) -> None:
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(S1_ROWS).to_excel(
            writer, sheet_name=w.DATA_SET_S1_EXPERIMENTS_SHEET, index=False
        )
        pd.DataFrame(S1_QUALITY).to_excel(
            writer, sheet_name=w.DATA_SET_S1_QUALITY_SHEET, index=False
        )
        pd.DataFrame([["Total Generations", "Total number of doublings"]]).to_excel(
            writer, sheet_name="ReadMe", index=False, header=False
        )


@pytest.fixture
def mirrored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    k12: tuple[EcoliK12MG1655Genome, EcoliK12BW25113Genome],
) -> Path:
    """A tmp ``DATA_ROOT`` whose mirrors hold the synthetic release; returns it."""
    source = tmp_path / "source"
    source.mkdir()
    frames = _good_frames(SUCCESSFUL, SYNTHETIC_GENES, extra=("set1IT030", "set1IT001"))
    for name, frame in frames.items():
        frame.to_csv(source / name, sep="\t", index=False)
    GENE_TABLE.to_csv(source / "fit_genes.tab", sep="\t", index=False)
    _write_s1(source / "si1.xlsx")
    monkeypatch.setattr(
        w,
        "LOADER_FILE_NAMES",
        {
            name: (
                "raw",
                f"data/{name}",
                hashlib.sha256((source / name).read_bytes()).hexdigest(),
            )
            for name in sorted(os.listdir(source))
        },
    )
    monkeypatch.setattr(
        w, "loader_source_path", lambda name, data_root=None: source / name
    )
    genomes = dict(zip(("MG1655", "BW25113"), k12, strict=True))
    monkeypatch.setattr(
        w, "bacterial_genome", lambda host, strain, data_root=None: genomes[strain]
    )
    monkeypatch.setattr(w, "assembly_reference", lambda strain: _pin())
    monkeypatch.setattr(w, "MIN_ECK_ROUTE_FRACTION", 0.5)
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    return data_root


def _pin() -> AssemblyReferenceGenome:
    return AssemblyReferenceGenome(
        species="Escherichia coli",
        strain="BW25113",
        assembly_set="ecoli_K12_BW25113_ASM75055v1",
        assembly_accession="GCA_000750555.1",
    )


def _dataset_root(data_root: Path) -> Path:
    return data_root / w.DATASET_ROOT_REL


def test_the_loader_builds_the_synthetic_release_end_to_end(mirrored: Path) -> None:
    root = _dataset_root(mirrored)
    dataset = w.RbTnseqWetmore2015Dataset(root=str(root))
    # six kept samples (set1IT007 withdrawn; set1IT030, which the quality rules fail,
    # kept since #1057) x all seven genes, each under its released b-number
    assert len(dataset) == 42
    assert sorted(dataset.gene_set) == SYNTHETIC_GENES
    references = dataset.experiment_reference_index
    assert references is not None
    assert len(references) == 6
    first = dataset[0]
    phenotype = first["experiment"]["phenotype"]
    assert phenotype["screen_id"] == "Keio:set1IT003"
    assert phenotype["environment_response"] == -2.0
    assert phenotype["environment_response_se"] == pytest.approx(0.3)
    assert phenotype["environment_response_t_statistic"] == pytest.approx(
        -2.0 / math.sqrt(0.01 + 0.09)
    )
    perturbation = first["experiment"]["genotype"]["perturbations"][0]
    assert (
        perturbation["systematic_gene_name"],
        perturbation["perturbed_gene_name"],
        perturbation["gene_namespace"],
        perturbation["identifier_mapping"],
        perturbation["transposon"],
        perturbation["library_pool"],
    ) == (
        "b0001",
        "thrL",
        "ecoli_k12_mg1655_bnumber",
        None,
        "Tn5 transpososome",
        "KEIO_ML9",
    )
    # a gene fit_genes.tab names with no symbol is named by its b-number
    names = {
        dataset[i]["experiment"]["genotype"]["perturbations"][0][
            "systematic_gene_name"
        ]: dataset[i]["experiment"]["genotype"]["perturbations"][0][
            "perturbed_gene_name"
        ]
        for i in range(7)
    }
    assert names["b0099"] == "b0099"
    assert first["publication"]["doi"] == "10.1128/mBio.00306-15"
    # the withdrawn sample is absent from the store, not flagged in it
    assert {
        dataset[i]["experiment"]["phenotype"]["screen_id"] for i in range(len(dataset))
    } == {
        "Keio:set1IT003",
        "Keio:set1IT030",
        "Keio:set1IT067",
        "Keio:set1IT071",
        "Keio:set1IT059",
        "Keio:set2IT069",
    }
    drops = json.loads((root / "preprocess" / "dropped_records.json").read_text())
    assert (drops["source_records"], drops["stored_records"]) == (49, 42)
    assert [(d["rule"], d["items"], d["n_records"]) for d in drops["rules"]] == [
        (w.DROP_WITHDRAWN, ["set1IT007"], 7)
    ]
    # every released value is accounted for: stored + the withdrawal
    assert drops["source_records"] == 42 + 7
    # the stored sample's quality verdict, as released (#1057)
    assert [f["name"] for f in drops["quality_flagged"]] == ["set1IT030"]
    samples = json.loads((root / "preprocess" / "samples.json").read_text())
    flagged = next(s for s in samples if s["name"] == "set1IT030")
    assert (flagged["drop_rule"], flagged["quality"]["successful"]) == (None, False)
    assert drops["withdrawn_refused"] == ["set1IT007"]
    # the refusal carries the authors' own words, bound to the file they were read from
    withdrawal = next(d for d in drops["rules"] if d["rule"] == w.DROP_WITHDRAWN)
    assert withdrawal["source"]["quote"] == (
        "Please disregard any of the data from this publication regarding sucrose or "
        "D-mannitol."
    )
    assert withdrawal["source"]["provenance"]["sha256"] == (
        w.RAW_FILES[w.SUPERSET_TOP_PAGE].sha256
    )
    assert drops["quality_rules_reproduce_u"] is True
    standard_error = json.loads(
        (root / "preprocess" / "standard_error.json").read_text()
    )
    assert (standard_error["n_values"], standard_error["sigma"]) == (42, 0.1)
    assert (root / "preprocess" / "build_manifest.json").is_file()
    assert sorted(os.listdir(root / "raw")) == sorted(w.LOADER_FILE_NAMES)


def test_a_genome_of_the_wrong_strain_is_refused(
    k12: tuple[EcoliK12MG1655Genome, EcoliK12BW25113Genome],
) -> None:
    dataset = w.RbTnseqWetmore2015Dataset.__new__(w.RbTnseqWetmore2015Dataset)
    dataset.ecoli_genome = k12[0]
    with pytest.raises(TypeError, match="expected the BW25113 genome"):
        dataset._bw25113()
    dataset.ecoli_genome = k12[1]
    assert dataset._bw25113() is k12[1]
    with pytest.raises(NotImplementedError, match="builds its records in process"):
        dataset.create_experiment()
    frame = pd.DataFrame({"a": [1]})
    assert dataset.preprocess_raw(frame) is frame


def _served_copy(data_root: Path, drop_rules: dict[str, str | None]) -> Path:
    """A compendium store holding this build's records, with its own samples.json."""
    superset = data_root / w.SUPERSET_DATASET_ROOT_REL
    shutil.copytree(_dataset_root(data_root) / "processed", superset / "processed")
    (superset / "preprocess").mkdir(parents=True)
    rows = [{"name": name, "drop_rule": rule} for name, rule in drop_rules.items()]
    (superset / "preprocess" / "samples.json").write_text(json.dumps(rows))
    return superset


def test_verify_build_runs_every_row_and_the_compendium_cross_check(
    mirrored: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _dataset_root(mirrored)
    w.RbTnseqWetmore2015Dataset(root=str(root))
    with pytest.raises(FileNotFoundError, match="the compendium store the L4"):
        w.verify_build(str(root), data_root=str(mirrored), expected_count=42)
    _served_copy(
        mirrored,
        {
            "set1IT003": None,
            "set1IT007": w.DROP_WITHDRAWN,
            "set1IT067": None,
            "set9": None,
        },
    )
    monkeypatch.setattr(
        w,
        "subsumption_record",
        lambda data_root=None: SimpleNamespace(
            carried=("set1IT003", "set1IT007", "set1IT067", "set1IT071")
        ),
    )
    monkeypatch.setattr(w, "EXPECTED_WITHDRAWN", ("set1IT007",))
    monkeypatch.setattr(w, "EXPECTED_SAMPLES", 6)
    report = w.verify_build(str(root), data_root=str(mirrored), expected_count=42)
    results = {r.name: r for r in report.results}
    # #1057: the released b-numbers against the MG1655 annotation (b0099 is no MG1655
    # locus); only the shared genome-reference rules may read that as a failure
    assert {name for name, r in results.items() if not r.passed} <= (
        GENOME_REFERENCE_RULES
    )
    assert results["count"].passed is True
    assert results["withdrawn_samples_refused_not_stored"].message == (
        "SUPPLEMENTARY: 0 records of the 1 samples the authors withdrew "
        "(['set1IT007']); 0 of them are in the store; 6 samples stored, expected 6"
    )
    cross = results["compendium_cross_check_price2018"]
    assert cross.level == Level.L4
    # the copy holds the same values, so every compared sample agrees exactly; the MOPS
    # rich sample is compared in both stores since #1045 row 16
    assert cross.details["compared"] == ["set1IT003", "set1IT067", "set1IT071"]
    assert cross.details["absent_here"] == {"set1IT007": w.DROP_WITHDRAWN}
    assert cross.details["absent_there"] == {"set1IT007": w.DROP_WITHDRAWN}
    assert [a["pearson_r"] for a in cross.details["agreements"]] == [
        pytest.approx(1.0),
        pytest.approx(1.0),
        pytest.approx(1.0),
    ]
    written = json.loads((root / "preprocess" / "verification_report.json").read_text())
    assert written["dataset_name"] == "RbTnseqWetmore2015Dataset"
    check = json.loads(
        (root / "preprocess" / "compendium_cross_check.json").read_text()
    )
    # three compared samples x all seven genes (three of them as released, #1057)
    assert check["n_values_compared"] == 21


def _value(
    sample: str,
    tag: str,
    value: float,
    *,
    se: float | None = 0.2,
    t: float | None = None,
) -> w.StoredValue:
    return w.StoredValue(
        sample=sample,
        tag=tag,
        gene_namespace="ecoli_k12_bw25113_locus_tag",
        value=value,
        standard_error=se,
        t_statistic=value / math.sqrt(0.01 + 0.04) if t is None else t,
    )


def test_compendium_cross_check_measures_agreement_and_names_every_absence() -> None:
    own = [_value("s1", g, v) for g, v in (("a", 1.0), ("b", 2.0), ("c", 3.0))]
    own += [_value("s2", "a", 0.5)]
    served = [_value("s1", g, v) for g, v in (("a", 1.1), ("b", 1.9), ("c", 3.4))]
    # the compendium's gene set is larger; its extra gene is counted, never compared
    served += [_value("s1", "d", 9.0)]
    served += [_value("s3", "a", 0.0)]
    check = w.compendium_cross_check(
        own,
        served,
        ["s1", "s2", "s3", "s4"],
        {"s3": "failed_paper_quality_rules", "s4": "medium_not_in_media_library"},
        {"s1": None, "s2": "authors_withdrew", "s3": None, "s4": "medium"},
        min_pearson_r=0.9,
    )
    assert check.compared == ("s1",)
    assert check.absent_here == {
        "s3": "failed_paper_quality_rules",
        "s4": "medium_not_in_media_library",
    }
    assert check.absent_there == {"s2": "authors_withdrew", "s4": "medium"}
    (agreement,) = check.agreements
    assert (agreement.n_only_here, agreement.n_only_there) == (0, 1)
    assert agreement.n_genes == 3
    assert agreement.max_abs_difference == pytest.approx(0.4)
    assert agreement.median_abs_difference == pytest.approx(0.1)
    assert agreement.pearson_r == pytest.approx(
        np.corrcoef([1, 2, 3], [1.1, 1.9, 3.4])[0, 1]
    )
    assert check.largest_differences[0].tag == "c"
    assert check.largest_differences[0].difference == pytest.approx(-0.4)
    assert check.n_values_compared == 3
    assert check.passed is True
    result = w.cross_check_result(check, "store")
    assert result.passed is True
    assert "largest single-gene difference -0.400 log2 (s1 c)" in result.message
    strict = w.compendium_cross_check(
        own, served, ["s1"], {}, {"s1": None}, min_pearson_r=0.999
    )
    assert strict.passed is False
    assert w.cross_check_result(strict, "store").passed is False


def test_compendium_cross_check_refuses_an_unexplained_absence_or_gene_set() -> None:
    own = [_value("s1", "a", 1.0), _value("s1", "b", 2.0)]
    served = [_value("s1", "a", 1.0)]
    with pytest.raises(ValueError, match="here \\['s2'\\], in the compendium store"):
        w.compendium_cross_check(own, served, ["s1", "s2"], {}, {"s1": None, "s2": "x"})
    # a gene held here and not in the compendium's sample is counted and fails the row
    nested = w.compendium_cross_check(own, served, ["s1"], {}, {"s1": None})
    (agreement,) = nested.agreements
    assert (agreement.n_genes, agreement.n_only_here, agreement.n_only_there) == (
        1,
        1,
        0,
    )
    assert nested.passed is False
    with pytest.raises(ValueError, match="s1: the two stores share no gene"):
        w.compendium_cross_check(
            own, [_value("s1", "z", 1.0)], ["s1"], {}, {"s1": None}
        )
    empty = w.compendium_cross_check(own, [], ["s1"], {}, {"s1": "dropped there"})
    assert (empty.passed, empty.compared, empty.absent_there) == (
        False,
        (),
        {"s1": "dropped there"},
    )
    assert "per-sample Pearson r none" in w.cross_check_result(empty, "x").message


def test_withdrawal_census_and_t_identity_rows() -> None:
    values = [_value("set1IT003", "a", -1.0), _value("set1IT059", "a", 0.4)]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(w, "EXPECTED_WITHDRAWN", ("set1IT007",))
        patch.setattr(w, "EXPECTED_SAMPLES", 2)
        assert w.withdrawal_census(values).passed is True
        # one record of a withdrawn sample fails the row and names the sample
        stored = [*values, _value("set1IT007", "b", 0.5)]
        result = w.withdrawal_census(stored)
        assert result.passed is False
        assert result.details["withdrawn_present"] == ["set1IT007"]
        assert "1 records of the 1 samples the authors withdrew" in result.message
        assert result.details["quote"].startswith("Please disregard")
        assert result.details["rule"] == w.DROP_WITHDRAWN
        # losing a sample that is not withdrawn fails the row too
        assert w.withdrawal_census(values[:1]).passed is False
    assert w.t_identity(values, 0.1).passed is True
    off = w.t_identity([*values, _value("s", "a", 1.0, t=9.0)], 0.1)
    assert off.passed is False
    assert off.details["max_abs_deviation"] == pytest.approx(9.0 - 1 / math.sqrt(0.05))
    missing = w.t_identity([_value("s", "a", 1.0, se=None)], 0.1)
    assert (missing.passed, missing.details["missing"]) == (False, 1)


def test_stored_values_reads_one_perturbation_per_record() -> None:
    record = {
        "experiment": {
            "phenotype": {
                "screen_id": "Keio:set1IT007",
                "environment_response": -1.0,
                "environment_response_se": 0.2,
                "environment_response_t_statistic": -4.47,
            },
            "genotype": {
                "perturbations": [
                    {
                        "systematic_gene_name": "BW25113_0001",
                        "gene_namespace": "ecoli_k12_bw25113_locus_tag",
                    }
                ]
            },
        }
    }
    (value,) = w.stored_values([record])
    assert value == w.StoredValue(
        sample="set1IT007",
        tag="BW25113_0001",
        gene_namespace="ecoli_k12_bw25113_locus_tag",
        value=-1.0,
        standard_error=0.2,
        t_statistic=-4.47,
    )
    # a compendium record written before the t field existed reads None for it
    old = json.loads(json.dumps(record))
    del old["experiment"]["phenotype"]["environment_response_t_statistic"]
    assert w.stored_values([old])[0].t_statistic is None
    two = json.loads(json.dumps(record))
    two["experiment"]["genotype"]["perturbations"] *= 2
    with pytest.raises(ValueError, match="expected one perturbation, got 2"):
        w.stored_values([two])


def test_stored_tags_are_loci_accepts_genes_and_self_resolving_pseudogenes() -> None:
    def resolve(tag: str) -> GeneNameResolution:
        status = {
            "BW25113_0001": GeneNameStatus.CURRENT,
            "BW25113_0004": GeneNameStatus.NON_GENE_FEATURE,
        }.get(tag, GeneNameStatus.RETIRED)
        return GeneNameResolution(input_name=tag, status=status, systematic_name=tag)

    ok = w.stored_tags_are_loci(["BW25113_0004", "BW25113_0001"], resolve)
    assert (ok.passed, ok.details["statuses"]) == (
        True,
        {"non_gene_feature": 1, "current": 1},
    )
    bad = w.stored_tags_are_loci(["BW25113_9999"], resolve)
    assert (bad.passed, bad.details["not_a_locus"]) == (False, ["BW25113_9999"])


def test_xlsx_text_joins_cells_and_rows(tmp_path: Path) -> None:
    path = tmp_path / "book.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.append(["a", None, "b"])
    sheet.append([None, None])
    sheet.append([1, 2.5])
    workbook.create_sheet("second").append(["c"])
    workbook.save(path)
    assert w.xlsx_text(str(path)) == "a | b / 1 | 2.5\nc"


def test_loader_source_path_checks_the_pin_of_each_mirror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def library(key: str, rel: str, pin: str, root: str | None = None) -> Path:
        calls.append(("library", key, rel, pin, str(root)))
        return Path("/library") / rel

    def raw(rel: str, root: str | None = None) -> Path:
        calls.append(("raw", rel, str(root)))
        return Path("/raw") / rel

    monkeypatch.setattr(w, "library_path", library)
    monkeypatch.setattr(w, "raw_path", raw)
    assert w.loader_source_path("si1.xlsx", "/dr") == Path("/library/si/si1.xlsx")
    assert w.loader_source_path("fit_t.tab", "/dr") == Path(
        "/raw/data/rbarseq/html/Keio/fit_t.tab"
    )
    assert calls == [
        ("library", w.CITATION_KEY, w.DATA_SET_S1, w.DATA_SET_S1_SHA256, "/dr"),
        ("raw", w.RELEASE_T, "/dr"),
    ]


def test_inventory_closes_over_every_released_value() -> None:
    specs = _specs()
    counts = w.inventory(specs, 7)
    assert (
        counts.condition_samples,
        counts.successful,
        counts.stored_samples,
        counts.source_records,
        counts.good_table_records,
        counts.stored_records,
    ) == (7, 6, 6, 49, 42, 42)
    assert counts.stored_samples_by_group == {
        "carbon source": 4,
        "nitrogen source": 1,
        "LB": 1,
    }
    # #1057: the failed sample is stored, its verdict listed as released
    assert [f.name for f in counts.quality_flagged] == ["set1IT030"]
    assert counts.quality_flagged[0].failed_rules == ("g_med_min",)
    assert counts.withdrawn_refused == ("set1IT007",)
    dropped = sum(r.n_records for r in counts.rules)
    assert counts.stored_records + dropped == counts.source_records
    # the refusal of the withdrawn sample is a drop rule carrying the authors' words
    withdrawal = next(r for r in counts.rules if r.rule == w.DROP_WITHDRAWN)
    assert (withdrawal.scope, withdrawal.items, withdrawal.n_records) == (
        "sample",
        ("set1IT007",),
        7,
    )
    assert withdrawal.source is w.AUTHOR_WITHDRAWAL.instruction
    # #1057: a gene no ECK pair places is stored as released, so no gene rule remains
    assert "no_one_to_one_eck_pair" not in {r.rule for r in counts.rules}
    # the kept quality-flagged fumarate sample names a compound the identity table does
    # not resolve, so its records carry the typed inchikey gap (#1057)
    assert counts.unidentified_compounds == {
        "Sodium Fumarate dibasic": 1,
        "casamino acids": 1,
    }


def test_the_command_line_dispatches_each_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(w, "retrieve_raw_files", lambda dest: f"retrieved {dest}")
    monkeypatch.setattr(
        w, "deposit_raw_mirror", lambda source_dir: f"deposited {source_dir}"
    )
    monkeypatch.setattr(w, "report", lambda: {"n": 1})
    verified: list[tuple[str, str]] = []

    def verify(root: str, *, data_root: str) -> SimpleNamespace:
        verified.append((root, data_root))
        return SimpleNamespace(summary=lambda: "verified")

    monkeypatch.setattr(w, "verify_build", verify)
    monkeypatch.setenv("DATA_ROOT", "/dr")
    w.main(["retrieve", "--dest", "/x"])
    w.main(["deposit", "--source-dir", "/y"])
    w.main(["report"])
    w.main(["verify"])
    assert capsys.readouterr().out.splitlines() == [
        "retrieved /x",
        "deposited /y",
        "{",
        '  "n": 1',
        "}",
        "verified",
    ]
    assert verified == [(f"/dr/{w.DATASET_ROOT_REL}", "/dr")]


def test_comparison_summary_names_the_worst_sample_and_the_bar() -> None:
    comparison = w.ReleaseComparison(
        n_source_genes=3,
        n_superset_genes=4,
        n_source_genes_in_superset=3,
        agreements=(
            w.ExperimentAgreement(
                name="s1", n_genes=3, pearson_r=0.99, max_abs_difference=0.4
            ),
            w.ExperimentAgreement(
                name="s2", n_genes=3, pearson_r=0.8, max_abs_difference=1.2
            ),
        ),
    )
    assert w._comparison_summary(comparison) == {
        "n_source_genes": 3,
        "n_superset_genes": 4,
        "n_source_genes_in_superset": 3,
        "n_shared_samples": 2,
        "min_pearson_r": 0.8,
        "median_pearson_r": pytest.approx(0.895),
        "max_abs_difference": 1.2,
        "max_abs_difference_sample": "s2",
        "samples_below_replicate_bar": ["s2"],
    }


# --------------------------------------------------------------------------- #
# The loader on the real mirrors (--data)
# --------------------------------------------------------------------------- #
@pytest.mark.data
@pytest.mark.skipif(
    not (MIRRORS_PRESENT and TIER_PRESENT),
    reason="requires the mirrors and the built K-12 data.db caches",
)
def test_the_loader_inventory_off_the_deposited_bytes() -> None:
    counts, identifiers = w.loader_inventory()
    assert (
        counts.condition_samples,
        counts.successful,
        counts.stored_samples,
        counts.source_genes,
        counts.stored_genes,
        counts.stored_records,
    ) == (101, 92, 97, w.EXPECTED_GENES, w.EXPECTED_GENES, w.EXPECTED_RECORDS)
    # #1057: all 3,646 released genes under their b-numbers; the derived crosswalk would
    # place 3,628 of them on BW25113 and names no record
    assert (w.EXPECTED_GENES, identifiers.n_mapped) == (3646, 3628)
    assert len(identifiers.released_as) == 18
    assert counts.stored_samples == w.EXPECTED_SAMPLES
    assert counts.quality_rules_reproduce_u is True
    # the 9 samples the quality rules fail are stored, their verdict as released
    assert [f.name for f in counts.quality_flagged] == [
        "set1IT030",
        "set1IT033",
        "set1IT034",
        "set2IT024",
        "set2IT034",
        "set2IT045",
        "set2IT055",
        "set2IT079",
        "set2IT090",
    ]
    assert counts.withdrawn_refused == w.EXPECTED_WITHDRAWN
    assert counts.stored_samples_by_group == {
        "carbon source": 64,
        "nitrogen source": 26,
        "stress": 2,
        "LB": 5,
    }
    # 4 withdrawn x 3,646 released genes, refused with the authors' own words
    withdrawal = next(r for r in counts.rules if r.rule == w.DROP_WITHDRAWN)
    assert (withdrawal.items, withdrawal.n_records) == (w.EXPECTED_WITHDRAWN, 14584)
    assert withdrawal.source is w.AUTHOR_WITHDRAWAL.instruction
    dropped = sum(r.n_records for r in counts.rules)
    assert counts.stored_records + dropped == counts.source_records == 101 * 3646
    assert identifiers.unmapped_by_reason == {
        "b_number_not_in_mg1655_annotation": 7,
        "eck_absent_from_bw25113": 6,
        "eck_not_one_to_one": 5,
    }
    # 29 before the #726 curator pass gave the condition labels their rows; measured
    # 2026-10-12 with the quality-flagged samples kept (#1057), the fumarate and lactate
    # salts carry a deferred inchikey gap and casamino acids its terminal one
    assert counts.unidentified_compounds == {
        "Sodium Fumarate dibasic": 2,
        "Sodium L-Lactate": 2,
        "casamino acids": 4,
    }


@pytest.mark.data
@pytest.mark.skipif(not MIRRORS_PRESENT, reason="requires the mirrors")
def test_all_92_shared_samples_agree_with_the_compendium_release() -> None:
    summary = w._comparison_summary(w.all_sample_comparison())
    assert summary["n_shared_samples"] == 92
    assert summary["samples_below_replicate_bar"] == []
    assert summary["min_pearson_r"] == pytest.approx(0.98895, abs=1e-5)
    assert summary["max_abs_difference"] == pytest.approx(2.5725, abs=1e-4)
    assert summary["max_abs_difference_sample"] == "set1IT028"


DEV_STORE = osp.join(DATA_ROOT, w.DATASET_ROOT_REL, "processed", "lmdb")


@pytest.mark.data
@pytest.mark.skipif(not osp.isdir(DEV_STORE), reason="requires the dev LMDB")
def test_the_dev_build_holds_every_expected_record_and_its_report_passes() -> None:
    environment = lmdb.open(DEV_STORE, readonly=True, lock=False)
    try:
        assert environment.stat()["entries"] == w.EXPECTED_RECORDS
    finally:
        environment.close()
    report = json.loads(
        Path(DEV_STORE)
        .parents[1]
        .joinpath("preprocess", "verification_report.json")
        .read_text()
    )
    # #1057: only the shared genome-reference rules may fail on the as-released genes
    assert {r["name"] for r in report["results"] if not r["passed"]} <= (
        GENOME_REFERENCE_RULES
    )
