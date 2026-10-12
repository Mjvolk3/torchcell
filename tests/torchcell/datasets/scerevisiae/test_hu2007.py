# tests/torchcell/datasets/scerevisiae/test_hu2007.py
# [[tests.torchcell.datasets.scerevisiae.test_hu2007]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_hu2007.py
"""Hu 2007 loader: the SOFT and Supplementary Table 1 readers, the element policy, the
build, the raw-mirror deposit, and ``verify_build``, on a synthetic release.

The synthetic GSE4654 carries 17 array elements and three samples:

    element      fake R64 resolution            key
    YAL001C      current                        YAL001C
    YAL002W      current                        YAL002W
    YAL003W      renamed -> YAL002W             dropped (YAL002W is on the array)
    YBR001C      current                        YBR001C
    YCR001W      current                        YCR001W
    YDR999W      renamed -> YDR001C             YDR001C
    YER109C      non-gene locus                 YER109C
    iYAL001C     retired                        dropped
    AMBIG1       ambiguous                      dropped
    YHR001W..YHR008W  current                   themselves

    GSM1  yal001c deletion   mid-log in Table 1       kept
    GSM2  ybr001c deletion   TetO7 row in Table 1     refused
    GSM3  yer109c deletion   mid-log in Table 1       kept (a non-gene locus deletion)

GSM1 releases YCR001W as ``null / 1`` and YBR001C as ``null / null``, so its record
scores 12 genes. Every p-value is ``2 * (1 - Phi(|X|))`` rounded to six decimals, as in
the release. The real-mirror tests at the end read the deposited bytes and skip when the
mirror is not on disk.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import os.path as osp
import pickle
from pathlib import Path
from typing import Any, cast

import lmdb
import pytest
from scipy.stats import norm

from torchcell.data import ManifestPinMismatchError
from torchcell.datamodels.media import YPD_LIQUID
from torchcell.datasets.scerevisiae import hu2007 as h
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome
from torchcell.verification.report import Level
from torchcell.verification.runners import load_records

_GENES = {
    "YAL001C",
    "YAL002W",
    "YBR001C",
    "YCR001W",
    "YDR001C",
    *(f"YHR00{i}W" for i in range(1, 9)),
}
_PLATFORM = [
    "YAL001C",
    "YAL002W",
    "YAL003W",
    "YBR001C",
    "YCR001W",
    "YDR999W",
    "YER109C",
    "iYAL001C",
    "AMBIG1",
    *(f"YHR00{i}W" for i in range(1, 9)),
]
_HHR = [f"YHR00{i}W" for i in range(1, 9)]


class _FakeGenome:
    """The slice of ``SCerevisiaeGenome`` the Hu 2007 L4 containment row reads."""

    gene_set = _GENES
    feature_index: dict[str, Any] = {
        "standard_to_ids": {"TFC3": ["YAL001C"]},
        "locus_type": {"YER109C": "blocked_reading_frame"},
    }


def _genome() -> SCerevisiaeGenome:
    return cast(SCerevisiaeGenome, _FakeGenome())


def _p(x: float) -> str:
    """The release's p column: the two-sided normal tail, six decimals."""
    return f"{2.0 * float(norm.sf(abs(x))):.6f}".rstrip("0").rstrip(".")


#: GSM1's released scores (the deleted YAL001C lowest).
_GSM1 = {
    "YAL001C": -3.0,
    "YAL002W": 0.5,
    "YAL003W": 0.25,
    "YDR999W": 1.5,
    "YER109C": -0.75,
    "iYAL001C": 2.0,
    "AMBIG1": 0.1,
    **{g: 0.2 * (i + 1) - 0.8 for i, g in enumerate(_HHR)},
}
_GSM3 = {
    **{e: 0.1 for e in _PLATFORM},
    "YER109C": -2.5,
    **{g: 0.8 - 0.2 * i for i, g in enumerate(_HHR)},
}


def _sample(
    gsm: str, title: str, values: dict[str, float], nulls: dict[str, str], **edit: str
) -> list[str]:
    header = {
        "!Sample_title": title,
        "!Sample_source_name_ch1": f"{title} mid-log growth",
        "!Sample_characteristics_ch1": "mid-log growth",
        "!Sample_label_ch1": "Cy5",
        "!Sample_source_name_ch2": "WT mid-log growth",
        "!Sample_characteristics_ch2": "mid-log growth",
        "!Sample_label_ch2": "Cy3",
        "!Sample_platform_id": "GPL3588",
        "!Sample_data_row_count": str(len(_PLATFORM)),
        "#ID_REF": "ID",
        "#VALUE": "X score",
        "#P": "p value",
    }
    header.update(edit)
    lines = [f"^SAMPLE = {gsm}", *(f"{k} = {v}" for k, v in header.items())]
    lines += ["!sample_table_begin", "ID_REF\tVALUE\tP"]
    for element in _PLATFORM:
        if element in nulls:
            lines.append(f"{element}\tnull\t{nulls[element]}")
        else:
            x = values.get(element, 0.0)
            lines.append(f"{element}\t{x!r}\t{_p(x)}")
    return [*lines, "!sample_table_end"]


def _soft_text(**gsm1_edit: str) -> str:
    lines = [
        "^DATABASE = GeoMiame",
        "^SERIES = GSE4654",
        "!Series_summary = Data values are X scores and corresponding p-values",
        "^PLATFORM = GPL3588",
        "!Platform_title = UT Austin Iyer yeast 6.4k ORF-only cDNA platform",
        "!platform_table_begin",
        "ID\tORF\tChromosome\tStart\tEnd\tSize",
        *(f"{e}\t{e}\t1\t1\t2\t1" for e in _PLATFORM),
        "!platform_table_end",
    ]
    lines += _sample(
        "GSM1",
        "yal001c deletion",
        _GSM1,
        {"YCR001W": "1", "YBR001C": "null"},
        **gsm1_edit,
    )
    lines += _sample("GSM2", "ybr001c deletion", {"YBR001C": 5.03}, {})
    lines += _sample("GSM3", "yer109c deletion", _GSM3, {})
    return "\n".join(lines) + "\n"


_TABLE1 = "\n".join(
    [
        "Supplementary Table 1 - Saccharomyces cerevsiae strain information",
        "TF Systematic ",
        "YAL001C TFC3 mig-log phase 0.82  ",
        "YAL001C TFC3 mig-log phase, heat-shock 0.70  ",
        "YBR001C NTH2 mig-log phase, tetracycline treatment 0.46  ",
        "YER109C  mig-log phase 1.00  ",
    ]
)


@pytest.fixture
def synthetic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the module's release-size constants at the synthetic release."""
    monkeypatch.setattr(h, "N_ARRAY_ELEMENTS", len(_PLATFORM))
    monkeypatch.setattr(h, "N_GEO_SAMPLES", 3)


def _write_soft(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# readers
# --------------------------------------------------------------------------- #
def test_read_soft_keeps_scores_and_counts_both_kinds_of_null(
    synthetic: None, tmp_path: Path
) -> None:
    release = h.read_soft(_write_soft(tmp_path / "s.soft.gz", _soft_text()))
    assert release.platform_ids == _PLATFORM
    assert [(s.gsm, s.orf) for s in release.samples] == [
        ("GSM1", "YAL001C"),
        ("GSM2", "YBR001C"),
        ("GSM3", "YER109C"),
    ]
    gsm1 = release.samples[0]
    assert (gsm1.null_both, gsm1.null_score_p_one) == (1, 1)
    assert set(gsm1.values) == set(_PLATFORM) - {"YCR001W", "YBR001C"}
    assert gsm1.values["YAL001C"] == (-3.0, 0.0027)
    assert gsm1.values["YAL002W"] == (0.5, 0.617075)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            {"!Sample_label_ch1": "Cy3"},
            "GSM1: !Sample_label_ch1 is 'Cy3', expected 'Cy5'",
        ),
        ({"#VALUE": "log ratio"}, "GSM1: #VALUE is 'log ratio', expected 'X score'"),
        ({"!Sample_data_row_count": "6429"}, "GSM1: row count is not 17"),
        (
            {"!Sample_source_name_ch1": "WT mid-log growth"},
            "GSM1: ch1 source does not name the sample's own strain",
        ),
    ],
)
def test_read_soft_refuses_a_sample_off_the_series_design(
    synthetic: None, tmp_path: Path, edit: dict[str, str], message: str
) -> None:
    path = _write_soft(tmp_path / "s.soft.gz", _soft_text(**edit))
    with pytest.raises(RuntimeError) as err:
        h.read_soft(path)
    assert str(err.value) == message


def test_read_soft_refuses_a_null_score_beside_a_tested_p(
    synthetic: None, tmp_path: Path
) -> None:
    text = _soft_text().replace("YCR001W\tnull\t1", "YCR001W\tnull\t0.5", 1)
    with pytest.raises(RuntimeError) as err:
        h.read_soft(_write_soft(tmp_path / "s.soft.gz", text))
    assert str(err.value) == "GSM1 YCR001W: null score beside p=0.5"


def test_read_soft_refuses_a_release_of_the_wrong_size(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError) as err:
        h.read_soft(_write_soft(tmp_path / "s.soft.gz", _soft_text()))
    assert str(err.value) == "GPL3588 holds 17 ids, not unique 6429"


def test_read_table1_parses_every_row_and_refuses_an_unparsed_one(
    tmp_path: Path,
) -> None:
    path = tmp_path / "si2.txt"
    path.write_text(_TABLE1, encoding="utf-8")
    rows = h.read_table1(path)
    assert [(r.orf, r.gene_name, r.condition, r.average_od) for r in rows] == [
        ("YAL001C", "TFC3", h.CONDITION_MID_LOG, 0.82),
        ("YAL001C", "TFC3", h.CONDITION_HEAT_SHOCK, 0.70),
        ("YBR001C", "NTH2", h.CONDITION_TETO7, 0.46),
        ("YER109C", None, h.CONDITION_MID_LOG, 1.00),
    ]
    path.write_text(_TABLE1 + "\nYCR001W ABC mig-log phase, cold 0.5", encoding="utf-8")
    with pytest.raises(RuntimeError) as err:
        h.read_table1(path)
    assert (
        str(err.value)
        == "unknown Supplementary Table 1 condition 'mig-log phase, cold'"
    )
    path.write_text("YCR001W ABC stationary 0.5", encoding="utf-8")
    with pytest.raises(RuntimeError) as err:
        h.read_table1(path)
    assert str(err.value) == (
        "unparsed Supplementary Table 1 row: 'YCR001W ABC stationary 0.5'"
    )


# --------------------------------------------------------------------------- #
# element policy, phenotypes, environment
# --------------------------------------------------------------------------- #
def test_phenotype_and_reference_hold_the_release_and_the_neutral_zero() -> None:
    sample = h.SoftSample(
        gsm="GSM1",
        title="yal001c deletion",
        orf="YAL001C",
        values={
            "YAL001C": (-3.0, 0.0027),
            "YDR999W": (1.5, 0.1336),
            "iYAL001C": (2.0, 0.0455),
            "AMBIG1": (1.0, 0.3173),
        },
        null_both=0,
        null_score_p_one=0,
    )
    # owner rule 2026-10-12: every released element ID is a key, verbatim, whatever
    # R64 says of it (a renamed ORF, an intergenic element, an ambiguous name)
    phenotype = h.phenotype_of(sample, 0.82)
    assert dict(phenotype.expression_score) == {
        "AMBIG1": 1.0,
        "YAL001C": -3.0,
        "YDR999W": 1.5,
        "iYAL001C": 2.0,
    }
    assert phenotype.expression_score_p_value is not None
    assert dict(phenotype.expression_score_p_value) == {
        "AMBIG1": 0.3173,
        "YAL001C": 0.0027,
        "YDR999W": 0.1336,
        "iYAL001C": 0.0455,
    }
    assert (phenotype.n_replicates, phenotype.harvest_optical_density) == (2, 0.82)
    assert phenotype.measurement_type == (
        "error_model_x_score_weighted_duplicate_average"
    )
    assert phenotype.array_platform == (
        "GEO GPL3588: UT Austin Iyer yeast 6.4k ORF-only cDNA platform "
        "(spotted cDNA, two-color)"
    )
    reference = h.reference_phenotype(list(phenotype.expression_score))
    assert dict(reference.expression_score) == dict.fromkeys(
        ["AMBIG1", "YAL001C", "YDR999W", "iYAL001C"], 0.0
    )
    assert (reference.expression_score_p_value, reference.harvest_optical_density) == (
        None,
        None,
    )


def test_environment_is_liquid_ypd_with_a_typed_temperature_gap() -> None:
    environment = h.environment()
    assert environment.media == YPD_LIQUID
    assert environment.temperature is None
    assert [(g.field, g.reason.value) for g in environment.provenance_gaps] == [
        ("temperature", "not_reported_by_primary")
    ]


def test_n_replicates_is_the_conservative_end_of_at_least_twice() -> None:
    assert h.N_REPLICATES.value == 2
    assert h.N_REPLICATES.quote == (
        "Each experiment was independently carried out at least twice."
    )


# --------------------------------------------------------------------------- #
# process()
# --------------------------------------------------------------------------- #
def _raw_root(tmp_path: Path, text: str, table: str = _TABLE1) -> Path:
    root = tmp_path / "microarray_score_hu2007"
    _write_soft(root / "raw" / "GSE4654_family.soft.gz", text)
    (root / "raw" / "si2.txt").write_text(table, encoding="utf-8")
    return root


def _build(root: Path) -> h.MicroarrayScoreHu2007Dataset:
    return h.MicroarrayScoreHu2007Dataset(root=str(root))


def test_process_writes_the_mid_log_deletions_and_the_three_ledgers(
    synthetic: None, tmp_path: Path, raw_pin_calls: list[Any]
) -> None:
    root = _raw_root(tmp_path, _soft_text())
    dataset = _build(root)
    assert len(dataset) == 2
    assert raw_pin_calls[-1] == (
        "hu2007",
        str(root / "raw"),
        {"GSE4654_family.soft.gz": h.SOFT.sha256, "si2.txt": h.SI_TABLE1_TEXT.sha256},
    )
    dataset.close_lmdb()
    records = load_records(str(root))
    by_strain = {
        r["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"]: r
        for r in records
    }
    assert sorted(by_strain) == ["YAL001C", "YER109C"]
    gsm1 = by_strain["YAL001C"]
    perturbation = gsm1["experiment"]["genotype"]["perturbations"][0]
    assert (perturbation["perturbed_gene_name"], perturbation["perturbation_type"]) == (
        "TFC3",
        "kanmx_deletion",
    )
    phenotype = gsm1["experiment"]["phenotype"]
    # every element GSM1 released a score for, under its released ID; the two it
    # released null (YBR001C null/null, YCR001W null/p = 1) are not keys
    assert sorted(phenotype["expression_score"]) == sorted(
        e for e in _PLATFORM if e not in {"YBR001C", "YCR001W"}
    )
    assert phenotype["expression_score"]["YDR999W"] == 1.5
    assert phenotype["expression_score"]["iYAL001C"] == 2.0
    assert phenotype["harvest_optical_density"] == 0.82
    reference = gsm1["reference"]
    assert reference["genome_reference"]["strain"] == "BY4741"
    assert set(reference["phenotype_reference"]["expression_score"].values()) == {0.0}
    assert (
        by_strain["YER109C"]["experiment"]["phenotype"]["harvest_optical_density"]
        == 1.0
    )
    # Supplementary Table 1 names no gene for YER109C, so the release's own name for
    # the strain is its ORF
    assert (
        by_strain["YER109C"]["experiment"]["genotype"]["perturbations"][0][
            "perturbed_gene_name"
        ]
        == "YER109C"
    )

    drop_log = json.loads((root / "preprocess" / "dropped_records.json").read_text())
    assert (drop_log["source_records"], drop_log["kept_records"]) == (3, 2)
    assert [(r["rule"], r["n_records"], r["items"]) for r in drop_log["rules"]] == [
        ("teto7_promoter_strain_reference_and_treatment_unresolved", 1, ["YBR001C"])
    ]
    assert drop_log["not_released"] == ["YAL001C (mig-log phase, heat-shock)"]
    elements = json.loads((root / "preprocess" / "array_elements.json").read_text())
    assert (
        elements["n_elements"],
        elements["n_keyed"],
        elements["n_stored_values"],
    ) == (17, 17, 15 + 17)
    assert "renamed" not in elements and "dropped" not in elements
    assert (elements["released_null_both"], elements["released_null_score_p_one"]) == (
        1,
        1,
    )
    with open(root / "preprocess" / "strain_table.csv") as handle:
        table = list(csv.DictReader(handle))
    assert [(r["orf"], r["condition"], r["in_gse4654"]) for r in table] == [
        ("YAL001C", h.CONDITION_MID_LOG, "True"),
        ("YAL001C", h.CONDITION_HEAT_SHOCK, "False"),
        ("YBR001C", h.CONDITION_TETO7, "True"),
        ("YER109C", h.CONDITION_MID_LOG, "True"),
    ]


def test_process_refuses_a_sample_absent_from_table_1(
    synthetic: None, tmp_path: Path
) -> None:
    table = _TABLE1.replace("YER109C  mig-log phase 1.00  ", "")
    root = _raw_root(tmp_path, _soft_text(), table)
    with pytest.raises(RuntimeError) as err:
        _build(root)
    assert str(err.value) == (
        "GSM3 (YER109C) has no mid-log or TetO7 row in Supplementary Table 1"
    )


def test_process_stores_a_deleted_orf_r64_does_not_carry_under_its_released_name(
    synthetic: None, tmp_path: Path
) -> None:
    """The owner rule (2026-10-12): no released row is dropped or refused on genome or
    annotation grounds. A strain whose ORF is not an R64 locus is stored as released.
    """
    text = _soft_text().replace("yer109c deletion", "ypl999c deletion")
    table = _TABLE1.replace("YER109C  mig", "YPL999C  mig")
    root = _raw_root(tmp_path, text, table)
    dataset = _build(root)
    dataset.close_lmdb()
    strains = sorted(
        r["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"]
        for r in load_records(str(root))
    )
    assert strains == ["YAL001C", "YPL999C"]


# --------------------------------------------------------------------------- #
# raw mirror
# --------------------------------------------------------------------------- #
def _staged_raw_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, str | Path]:
    """Synthetic bytes for the three mirror files, with the module pins repointed."""
    sources: dict[str, str | Path] = {}
    staged: list[h.RawFile] = []
    for raw in h.RAW_FILES:
        path = tmp_path / "staged" / raw.relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic {raw.relpath}".encode())
        staged.append(raw.model_copy(update={"sha256": _sha(path)}))
        sources[raw.relpath] = path
    monkeypatch.setattr(h, "RAW_FILES", tuple(staged))
    monkeypatch.setattr(h, "SOFT", staged[0])
    monkeypatch.setattr(h, "SI_TABLE1_PDF", staged[1])
    monkeypatch.setattr(h, "SI_TABLE1_TEXT", staged[2])
    return sources


def test_deposit_writes_the_mirror_and_its_manifest_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _staged_raw_files(tmp_path, monkeypatch)
    data_root = str(tmp_path / "dr")
    root = h.deposit_raw_mirror(sources, data_root=data_root)
    assert root == Path(data_root) / "torchcell-raw" / h.CITATION_KEY
    manifest = h.load_manifest(data_root)
    by_path = {f.path: f for f in manifest.files}
    assert sorted(by_path) == [
        "data/GSE4654_family.soft.gz",
        "si/si2.pdf",
        "si/si2.txt",
    ]
    soft = by_path["data/GSE4654_family.soft.gz"]
    assert soft.retrieval is not None
    assert (soft.retrieval.method.value, soft.retrieval.retriever) == (
        "direct_url",
        "torchcell.literature.retrieve.direct_url",
    )
    assert soft.original_filename == "GSE4654_family.soft.gz"
    text = by_path["si/si2.txt"]
    assert (text.retrieval, text.source) == (None, "derived from si/si2.pdf")
    assert text.processing is not None
    assert text.processing.input_sha256 == [h.SI_TABLE1_PDF.sha256]
    assert h.manifest_sha256(manifest, "si/si2.pdf") == h.SI_TABLE1_PDF.sha256
    with pytest.raises(KeyError):
        h.manifest_sha256(manifest, "absent.txt")
    # a second deposit of the same bytes is a no-op; different bytes are refused
    h.deposit_raw_mirror(sources, data_root=data_root)
    (root / "si/si2.txt").write_bytes(b"edited")
    with pytest.raises(RuntimeError) as err:
        h.deposit_raw_mirror(sources, data_root=data_root)
    assert (
        str(err.value)
        == f"{root / 'si/si2.txt'} exists with a different sha256; refusing"
    )
    del sources["si/si2.pdf"]
    with pytest.raises(RuntimeError) as err:
        h.deposit_raw_mirror(sources, data_root=data_root)
    assert str(err.value) == "no source given for si/si2.pdf"


def test_download_links_the_pinned_files_and_refuses_a_manifest_off_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _staged_raw_files(tmp_path, monkeypatch)
    data_root = tmp_path / "dr"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    h.deposit_raw_mirror(sources, data_root=str(data_root))
    dataset = h.MicroarrayScoreHu2007Dataset.__new__(h.MicroarrayScoreHu2007Dataset)
    dataset.root = str(tmp_path / "store")
    h.MicroarrayScoreHu2007Dataset.download(dataset)
    raw = tmp_path / "store" / "raw"
    assert sorted(os.listdir(raw)) == ["GSE4654_family.soft.gz", "si2.txt"]
    assert os.readlink(raw / "si2.txt") == str(
        data_root / "torchcell-raw" / h.CITATION_KEY / "si/si2.txt"
    )
    monkeypatch.setattr(h, "SOFT", h.SOFT.model_copy(update={"sha256": "0" * 64}))
    with pytest.raises(ManifestPinMismatchError):
        h.MicroarrayScoreHu2007Dataset.download(dataset)


# --------------------------------------------------------------------------- #
# verify_build
# --------------------------------------------------------------------------- #
def _write_store(root: Path, records: list[dict[str, Any]]) -> None:
    path = root / "processed" / "lmdb"
    path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(path), map_size=1 << 24)
    with env.begin(write=True) as txn:
        for i, record in enumerate(records):
            txn.put(str(i).encode(), pickle.dumps(record))
    env.close()


def _kemmeren(gene: str, profile: dict[str, float]) -> dict[str, Any]:
    return {
        "experiment": {
            "genotype": {"perturbations": [{"systematic_gene_name": gene}]},
            "phenotype": {"expression_log2_ratio": profile},
        }
    }


def test_verify_build_passes_l0_to_l4_on_the_synthetic_release(
    synthetic: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "dr"
    soft = _write_soft(
        data_root / "torchcell-raw" / h.CITATION_KEY / h.SOFT.relpath, _soft_text()
    )
    monkeypatch.setattr(h, "SOFT", h.SOFT.model_copy(update={"sha256": _sha(soft)}))
    monkeypatch.setattr(h, "SOURCED_VALUES", ())
    monkeypatch.setattr(h, "EXPECTED_RECORDS", 2)
    # two shared strains cannot reach 1e-3 in a rank test; the threshold is asserted on
    # the real store, here only its arithmetic
    monkeypatch.setattr(h, "KEMMEREN_ALPHA", 1.0)
    store = data_root / h.DATASET_ROOT
    _raw_root(store.parent, _soft_text())
    _build(store).close_lmdb()
    _write_store(
        data_root / h.KEMMEREN_ROOT,
        [
            _kemmeren("YAL001C", {g: _GSM1[g] for g in _HHR}),
            _kemmeren("YER109C", {g: _GSM3[g] for g in _HHR}),
        ],
    )
    report = h.verify_build(str(store), str(data_root), genome=_genome())
    results = {r.name: r for r in report.results}
    assert report.passed, [r.message for r in report.results if not r.passed]
    assert list(results) == [
        "structural",
        "count",
        "score_finite",
        "p_value_in_unit_interval",
        "p_value_keys_match_scores",
        "reference_score_zero",
        "deletion_lowers_score",
        "p_value_is_two_sided_normal",
        "released_value_fidelity",
        "sourced_value_audit",
        "deleted_gene_containment_sgd",
        "matched_profile_agreement_microarray_kemmeren2014",
    ]
    assert results["released_value_fidelity"].details["n_values"] == 15 + 17
    assert results["deleted_gene_containment_sgd"].details["non_gene"] == ["YER109C"]
    agreement = results["matched_profile_agreement_microarray_kemmeren2014"]
    assert agreement.level is Level.L4
    assert agreement.details["n_genes"] == 8
    assert math.isclose(agreement.details["median_matched_r"], 1.0)


def test_released_fidelity_names_a_record_that_drifted(
    synthetic: None, tmp_path: Path
) -> None:
    release = h.read_soft(_write_soft(tmp_path / "s.soft.gz", _soft_text()))
    record = {
        "experiment": {
            "genotype": {"perturbations": [{"systematic_gene_name": "YER109C"}]},
            "phenotype": {
                "expression_score": {"YER109C": -2.4},
                "expression_score_p_value": {"YER109C": 0.0164},
            },
        }
    }
    result = h._l2_released_fidelity([record], release)
    assert not result.passed
    assert result.details["mismatched"] == [
        {"strain": "YER109C", "n_stored": 1, "n_released": 17, "n_differ": 17}
    ]


# --------------------------------------------------------------------------- #
# the real mirrors
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    from dotenv import load_dotenv

    load_dotenv()
    return os.environ.get("DATA_ROOT", "")


_RAW = osp.join(_data_root(), h.RAW_DIR_REL)
_LIBRARY = osp.join(_data_root(), "torchcell-library")


@pytest.mark.data
@pytest.mark.skipif(
    not osp.isfile(osp.join(_RAW, "manifest.json")),
    reason="Hu 2007 raw mirror not on disk",
)
def test_the_release_is_the_pinned_gse4654() -> None:
    manifest = h.load_manifest(_data_root())
    for raw in h.RAW_FILES:
        assert h.manifest_sha256(manifest, raw.relpath) == raw.sha256
    release = h.read_soft(Path(_RAW) / h.SOFT.relpath)
    assert len(release.samples) == 269
    assert sum(len(s.values) for s in release.samples) == 1_707_575
    assert sum(s.null_both for s in release.samples) == 3180
    assert sum(s.null_score_p_one for s in release.samples) == 18646
    rows = h.read_table1(Path(_RAW) / h.SI_TABLE1_TEXT.relpath)
    assert len(rows) == 292
    released = {s.orf for s in release.samples}
    assert released == {
        r.orf for r in rows if r.condition in (h.CONDITION_MID_LOG, h.CONDITION_TETO7)
    }


@pytest.mark.data
@pytest.mark.skipif(
    not osp.isdir(osp.join(_LIBRARY, h.CITATION_KEY)),
    reason="Hu 2007 library mirror not on disk",
)
def test_every_quote_reads_from_the_library_mirror() -> None:
    result = h._l3_provenance_audit(_LIBRARY)
    assert (result.passed, result.details["failed"]) == (True, [])
    assert result.details["n_sourced"] == 16
