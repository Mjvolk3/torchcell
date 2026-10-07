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
import os
import os.path as osp
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import torchcell.datasets.ecoli.wetmore2015 as w
from torchcell.data import ManifestPinMismatchError
from torchcell.datamodels.schema import MeasurementType, UncertaintyType
from torchcell.datasets.dataset_registry import dataset_registry
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


def test_the_decision_registers_no_dataset() -> None:
    """Subsumed rows are provenance records, not loaders (checklist item 7)."""
    assert not any(cls.__module__ == w.__name__ for cls in dataset_registry.values())


def test_sourced_vocabulary_for_the_rb_tnseq_rows() -> None:
    assert w.GENE_FITNESS_IS_LOG2.value is MeasurementType.log2_ratio
    assert w.UNCERTAINTY_TYPE.value is UncertaintyType.standard_error
    assert w.DISREGARDED_CONDITIONS == frozenset({"Sucrose", "D-Mannitol"})
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
    with pytest.raises(RuntimeError, match="records other files or retrievals"):
        w.deposit_raw_mirror(
            source_dir=source, retrieved_at="2027-01-01", data_root=data_root
        )
    assert (root / "manifest.json").read_text() == original


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
