# tests/torchcell/datasets/scerevisiae/test_peter2018_synthetic.py
# [[tests.torchcell.datasets.scerevisiae.test_peter2018_synthetic]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_peter2018_synthetic.py
"""Peter 2018 isolate-growth loader and its verifier, end to end on synthetic files.

``DATA_ROOT`` points into ``tmp_path``. The raw mirror is written by the loader's own
``deposit_raw_mirror`` from a two-isolate release and a ``md5.txt`` naming its md5; the
two genome tiers get real ``GenomeManifest``s, so every sha256 check on the build path
runs. Module constants that state the REAL release's size (971 isolates, 33,985 records,
34,956 paper measurements, the published md5) are monkeypatched to the synthetic one's.

SGD FASTA: chromosome I ``AAAACCCCGGGGTTTT``, II ``ACCCTTTGGA``. Reference-gene tarball:
``YAL001C.fasta`` on chromosome1:1-8 + (slice ``AAAACCCC``; AAA equal, SACE_YAU
``AAAACCCT`` differs, ZZZ has a genome but no phenotype and is skipped) and
``YBR001W.fasta`` on chromosome2:2-5 - (slice ``AGGG``; AAA ``AGGC`` differs, SACE_YAU
equal). Presence columns X1.YAL001C, X2.EC1118_1F14_0012g, X3.YBR001W, X4.contig_7:
AAA = 1 1 1 0 (accessory X2 present, copy number 2.5), SACE_YAU = 1 0 0 1 (YBR001W
absent, accessory X4 present, copy number NA -> 1.0), ZZZ = 1 0 1 0.

Release: AAA and SACE_YAU on the 35 conditions; AAA's value on condition k is
``0.<k+1>1`` written to three decimals plus ``0.518089736`` on YPACETATE (the real
file's first cell, so the round-trip parser is exercised), and SACE_YAU's YPDHU is an
exact 0 (the release holds 197 of those).
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from torchcell.datamodels.schema import (
    AssayType,
    ConcentrationUnit,
    MeasurementType,
    SampleUnit,
    SmallMoleculePerturbation,
)
from torchcell.datasets.scerevisiae import peter2018 as p
from torchcell.datasets.scerevisiae.caudal2024 import (
    COPYNUMBER_NAME,
    PRESENCE_NAME,
    REFGENE_TAR_NAME,
    SGD_FSA_NAME,
)
from torchcell.literature.manifest import ArtifactRecord, Manifest
from torchcell.sequence.genome.registry import (
    PETER2018_1011,
    SGD_S288C_R64,
    GenomeManifest,
    load_genome_manifest,
)
from torchcell.verification.natural_isolate_response import (
    expected_condition_signatures,
    verify_natural_isolate_response_streaming,
)
from torchcell.verification.report import Provenance
from torchcell.verification.runners import stream_records
from torchcell.verification.sourced import SourcedValue

_SGD_FSA = (
    ">ref|NC_001133| [org=Saccharomyces cerevisiae] [chromosome=I]\n"
    "AAAACCCC\nGGGGTTTT\n"
    ">ref|NC_001134| [org=Saccharomyces cerevisiae] [chromosome=II]\n"
    "ACCCTTTGGA\n"
)
_GENE_FILES = [
    (
        "YAL001C.fasta",
        ">AAA_YAL001C_TFC3 chromosome1:1-8 +\nAAAACCCC\n"
        ">SACE_YAU_YAL001C_TFC3 chromosome1:1-8 +\nAAAACCCT\n"
        ">ZZZ_YAL001C_TFC3 chromosome1:1-8 +\nGGGGGGGG\n",
    ),
    (
        "YBR001W.fasta",
        ">AAA_YBR001W_NTH2 chromosome2:2-5 -\nAGGC\n"
        ">SACE_YAU_YBR001W_NTH2 chromosome2:2-5 -\nAGGG\n"
        ">ZZZ_YBR001W_NTH2 chromosome2:2-5 -\nAGGG\n",
    ),
]
_COLUMNS = ["X1.YAL001C", "X2.EC1118_1F14_0012g", "X3.YBR001W", "X4.contig_7"]
_PRESENCE = {
    "AAA": ["1", "1", "1", "0"],
    "SACE_YAU": ["1", "0", "0", "1"],
    "ZZZ": ["1", "0", "1", "0"],
}
_COPYNUMBER = {
    "AAA": ["1", "2.5", "1", "0"],
    "SACE_YAU": ["1", "0", "0", "NA"],
    "ZZZ": ["1", "0", "1", "0"],
}
_CODES = list(p.build_conditions())


def _release_text() -> str:
    aaa = [f"0.{k + 1}1" for k in range(len(_CODES))]
    aaa[_CODES.index("YPACETATE")] = "0.518089736"
    yau = [f"1.{k}" for k in range(len(_CODES))]
    yau[_CODES.index("YPDHU")] = "0"
    lines = [
        "\t" + "\t".join(_CODES),
        "AAA\t" + "\t".join(aaa),
        "SACE_YAU\t" + "\t".join(yau),
    ]
    return "\n".join(lines) + "\n"


def _gz(text: str) -> bytes:
    return gzip.compress(text.encode(), mtime=0)


def _matrix(rows: dict[str, list[str]]) -> bytes:
    lines = ["\t".join(["strain", *_COLUMNS])]
    lines += ["\t".join([name, *values]) for name, values in rows.items()]
    return _gz("\n".join(lines) + "\n")


def _tarball() -> bytes:
    buffer = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as gz,
        tarfile.open(fileobj=gz, mode="w") as tar,
    ):
        for name, text in _GENE_FILES:
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _write_tier(data_root: Path, assembly_set: str, files: dict[str, bytes]) -> None:
    tier = data_root / "torchcell-genomes" / assembly_set
    tier.mkdir(parents=True)
    records = []
    for name, data in files.items():
        (tier / name).write_bytes(data)
        records.append(
            ArtifactRecord(
                path=name,
                role="sequence",
                bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
            )
        )
    manifest = GenomeManifest(
        assembly_set=assembly_set,
        organism="Saccharomyces cerevisiae",
        strain_or_population="synthetic",
        source="test",
        release="test",
        files=records,
        provenance_complete=True,
        created_at="2026-10-11T00:00:00+00:00",
    )
    (tier / "manifest.json").write_text(manifest.model_dump_json(indent=2))


_RELEASE = _gz(_release_text())
_RELEASE_MD5 = hashlib.md5(_RELEASE, usedforsecurity=False).hexdigest()


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(root))
    monkeypatch.setattr(p, "N_ISOLATES", 2)
    monkeypatch.setattr(p, "EXPECTED_RECORDS", 70)
    monkeypatch.setattr(p, "PAPER_MEASUREMENTS", 72)
    monkeypatch.setattr(p, "PHENO_PUBLISHED_MD5", _RELEASE_MD5)
    monkeypatch.setattr(p, "PHENO_SHA256", hashlib.sha256(_RELEASE).hexdigest())
    _write_tier(root, SGD_S288C_R64, {SGD_FSA_NAME: _SGD_FSA.encode()})
    _write_tier(
        root,
        PETER2018_1011,
        {
            REFGENE_TAR_NAME: _tarball(),
            PRESENCE_NAME: _matrix(_PRESENCE),
            COPYNUMBER_NAME: _matrix(_COPYNUMBER),
        },
    )
    retrieved = tmp_path / "retrieved"
    retrieved.mkdir()
    (retrieved / p.PHENO_NAME).write_bytes(_RELEASE)
    (retrieved / "md5.txt").write_text(f"MD5 ({p.PHENO_NAME}) = {_RELEASE_MD5}\n")
    p.deposit_raw_mirror(
        pheno_path=retrieved / p.PHENO_NAME,
        md5_path=retrieved / "md5.txt",
        retrieved_at="2026-10-11",
        data_root=str(root),
    )
    return root


@pytest.fixture
def dataset(data_root: Path) -> p.IsolateGrowthPeter2018Dataset:
    return p.IsolateGrowthPeter2018Dataset(root=str(data_root / "store"))


def test_raw_mirror_manifest_records_both_retrievals(data_root: Path) -> None:
    manifest = Manifest.model_validate_json(
        (data_root / p.RAW_DIR_REL / "manifest.json").read_text()
    )
    assert [(f.path, f.role, f.source) for f in manifest.files] == [
        (p.PHENO_REL, "raw_data", p.DATA_HOST + p.PHENO_NAME),
        (p.MD5_REL, "raw_data", p.DATA_HOST + "md5.txt"),
    ]
    assert manifest.files[0].sha256 == hashlib.sha256(_RELEASE).hexdigest()
    retrieval = manifest.files[0].retrieval
    assert retrieval is not None
    assert (retrieval.method.value, retrieval.retriever, retrieval.params) == (
        "direct_url",
        "torchcell.literature.retrieve.direct_url",
        {"url": p.DATA_HOST + p.PHENO_NAME},
    )
    assert (manifest.citation_key, manifest.doi) == (p.CITATION_KEY, p.PAPER_DOI)


def test_deposit_refuses_a_wrong_md5_and_a_changed_file(
    data_root: Path, tmp_path: Path
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    (other / p.PHENO_NAME).write_bytes(_RELEASE)
    (other / "md5.txt").write_text(f"MD5 ({p.PHENO_NAME}) = {'0' * 32}\n")
    with pytest.raises(ValueError, match=f"md5.txt lists {'0' * 32}"):
        p.deposit_raw_mirror(
            pheno_path=other / p.PHENO_NAME,
            md5_path=other / "md5.txt",
            retrieved_at="x",
            data_root=str(data_root),
        )
    (other / "md5.txt").write_text(f"MD5 ({p.PHENO_NAME}) = {_RELEASE_MD5}\n")
    (other / p.PHENO_NAME).write_bytes(_RELEASE + b"\n")
    with pytest.raises(ValueError, match="does not have the published md5"):
        p.deposit_raw_mirror(
            pheno_path=other / p.PHENO_NAME,
            md5_path=other / "md5.txt",
            retrieved_at="x",
            data_root=str(data_root),
        )


def test_read_published_md5_parses_and_refuses(tmp_path: Path) -> None:
    good = tmp_path / "md5.txt"
    good.write_text(
        "MD5 (a.gz) = 479862335552a050e46de163d1ee5f72\n\nMD5 (b c.gz) = "
        "f3c4f2db7a5b2a9bf28039dbdd82e2b1\n"
    )
    assert p.read_published_md5(good) == {
        "a.gz": "479862335552a050e46de163d1ee5f72",
        "b c.gz": "f3c4f2db7a5b2a9bf28039dbdd82e2b1",
    }
    bad = tmp_path / "bad.txt"
    bad.write_text("479862335552a050e46de163d1ee5f72  a.gz\n")
    with pytest.raises(ValueError, match="unparseable md5 line"):
        p.read_published_md5(bad)


def test_register_loss_of_function_matrices_appends_to_the_peter_tier(
    data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "lof"
    src.mkdir()
    payload = {p.LOF_NAME: b"lof\n", p.FRAMESHIFT_NAME: b"frameshift\n"}
    md5s = {
        n: hashlib.md5(b, usedforsecurity=False).hexdigest() for n, b in payload.items()
    }
    for name, data in payload.items():
        (src / name).write_bytes(data)
    (src / "md5.txt").write_text("".join(f"MD5 ({n}) = {m}\n" for n, m in md5s.items()))
    with pytest.raises(ValueError, match="md5.txt lists"):
        p.register_loss_of_function_matrices(
            source_dir=src,
            md5_path=src / "md5.txt",
            retrieved_at="x",
            data_root=str(data_root),
        )
    monkeypatch.setattr(p, "LOF_PUBLISHED_MD5", md5s)
    (src / p.LOF_NAME).write_bytes(b"lof changed\n")
    with pytest.raises(ValueError, match="does not have the published md5"):
        p.register_loss_of_function_matrices(
            source_dir=src,
            md5_path=src / "md5.txt",
            retrieved_at="x",
            data_root=str(data_root),
        )
    (src / p.LOF_NAME).write_bytes(payload[p.LOF_NAME])
    path = p.register_loss_of_function_matrices(
        source_dir=src,
        md5_path=src / "md5.txt",
        retrieved_at="2026-10-11",
        data_root=str(data_root),
    )
    manifest = load_genome_manifest(PETER2018_1011, str(data_root))
    assert path.endswith(f"{PETER2018_1011}/manifest.json")
    assert [r.path for r in manifest.files][-2:] == [p.LOF_NAME, p.FRAMESHIFT_NAME]
    added = manifest.record(p.LOF_NAME)
    assert (added.role, added.bytes, added.source) == (
        "matrix",
        4,
        p.DATA_HOST + p.LOF_NAME,
    )
    assert added.sha256 == hashlib.sha256(b"lof\n").hexdigest()


def test_condition_table_is_table_s2_in_order() -> None:
    conditions = p.build_conditions()
    assert len(conditions) == 35
    assert [s.table_s2_row for s in conditions.values()] == list(range(6, 41))
    temps = {c: s.temperature_c for c, s in conditions.items() if s.temperature_shifted}
    assert temps == {"YPD14": 14.0, "YPD40": 40.0, "YPD42": 42.0}
    carbon = {
        c: s.media.base_medium for c, s in conditions.items() if not s.perturbations
    }
    assert carbon == {
        "YPD14": "YPD",
        "YPD40": "YPD",
        "YPD42": "YPD",
        "YPACETATE": "YP",
        "YPETHANOL": "YP",
        "YPGLYCEROL": "YP",
        "YPSORBITOL": "YP",
        "YPGALACTOSE": "YP",
        "YPRIBOSE": "YP",
        "YPXYLOSE": "YP",
    }
    doses = {
        c: (s.perturbations[0].compound.name, s.perturbations[0].concentration.value,
            s.perturbations[0].concentration.unit)
        for c, s in conditions.items()
        if s.perturbations
    }  # fmt: skip
    assert doses["YPDMV"] == ("paraquat", 20.0, ConcentrationUnit.millimolar)
    assert doses["YPD6AU"] == ("6-azauracil", 600.0, ConcentrationUnit.ug_per_ml)
    assert doses["YPDSODIUMMETAARSENITE"] == (
        "sodium arsenite",
        2.5,
        ConcentrationUnit.millimolar,
    )
    assert doses["YPDHU"] == ("hydroxyurea", 30.0, ConcentrationUnit.g_per_l)
    assert doses["YPDETOH"] == ("ethanol", 15.0, ConcentrationUnit.percent)
    assert doses["YPDKCL2M"] == ("potassium chloride", 2.0, ConcentrationUnit.molar)
    kcl = conditions["YPDKCL2M"].perturbations[0].compound
    assert [(g.field, g.reason.value) for g in kcl.provenance_gaps] == [
        ("inchikey", "deferred_pending_source_review")
    ]
    assert all(
        [g.field for g in s.perturbations[0].provenance_gaps] == ["solvent"]
        for s in conditions.values()
        if s.perturbations
    )
    ribose = conditions["YPRIBOSE"].media
    assert [c.compound.name for c in ribose.components] == [
        "yeast extract",
        "peptone",
        "ribose",
        "agar",
    ]
    assert ribose.name == "YP + 2% ribose, 2% agar (Peter 2018 Table S2)"


def test_environment_phenotype_and_reference_builders() -> None:
    conditions = p.build_conditions()
    hot = p.condition_environment(conditions["YPD42"])
    assert (hot.temperature.value if hot.temperature else None, hot.duration_hours) == (
        42.0,
        None,
    )
    assert [g.field for g in hot.provenance_gaps] == ["duration_hours"]
    salt = p.condition_environment(conditions["YPDNACL1M"])
    assert (salt.duration_hours, salt.media.base_medium, len(salt.perturbations)) == (
        40.0,
        "YPD",
        1,
    )
    assert isinstance(salt.perturbations[0], SmallMoleculePerturbation)
    phenotype = p.growth_phenotype(0.0)
    assert (
        phenotype.measurement_type,
        phenotype.assay_type,
        phenotype.environment_response,
        phenotype.n_samples,
        phenotype.sample_unit,
        phenotype.environment_response_se,
    ) == (
        MeasurementType.condition_growth_ratio,
        AssayType.colony_size_array,
        0.0,
        4,
        SampleUnit.colony,
        None,
    )
    assert [g.field for g in phenotype.provenance_gaps] == ["environment_response_se"]
    reference = p.shared_reference("X")
    assert reference.phenotype_reference.environment_response == 1.0
    assert reference.environment_reference == p.control_environment()
    assert (
        reference.genome_reference.strain,
        p.control_environment().perturbations,
    ) == ("S288C", [])


def test_read_release_refusals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(p, "N_ISOLATES", 2)
    conditions = p.build_conditions()
    path = tmp_path / "r.tab.gz"
    path.write_bytes(_RELEASE)
    frame = p.read_release(path, conditions)
    assert frame.at["AAA", "YPACETATE"] == 0.518089736
    assert frame.at["SACE_YAU", "YPDHU"] == 0.0
    path.write_bytes(_gz(_release_text().replace("1.3\t", "-1.3\t", 1)))
    with pytest.raises(p.ReleaseShapeError, match="negative or non-finite"):
        p.read_release(path, conditions)
    path.write_bytes(_gz(_release_text().replace("1.3\t", "\t", 1)))
    with pytest.raises(p.ReleaseShapeError, match="1 missing cells"):
        p.read_release(path, conditions)
    path.write_bytes(_gz(_release_text().replace("YPDHU", "YPDHUX")))
    with pytest.raises(p.ReleaseShapeError, match="differ from the condition table"):
        p.read_release(path, conditions)
    path.write_bytes(_gz(_release_text().replace("SACE_YAU", "AAA")))
    with pytest.raises(p.ReleaseShapeError, match="appears twice"):
        p.read_release(path, conditions)
    monkeypatch.setattr(p, "N_ISOLATES", 3)
    path.write_bytes(_RELEASE)
    with pytest.raises(p.ReleaseShapeError, match=r"release shape \(2, 35\)"):
        p.read_release(path, conditions)


def _records(dataset: p.IsolateGrowthPeter2018Dataset) -> list[dict[str, Any]]:
    """Records in index order (the LMDB cursor walks keys as strings: 0, 1, 10, ...)."""
    records = [dataset.get_single_item(i) for i in range(len(dataset))]
    dataset.close_lmdb()
    return records


def test_build_writes_seventy_isolate_major_records(
    dataset: p.IsolateGrowthPeter2018Dataset,
) -> None:
    assert len(dataset) == 70
    records = _records(dataset)
    strains = [
        {pert["strain_id"] for pert in r["experiment"]["genotype"]["perturbations"]}
        for r in records
    ]
    assert strains == [{"AAA"}] * 35 + [{"SACE_YAU"}] * 35
    assert records[0]["experiment"]["genotype"] is records[34]["experiment"]["genotype"]
    aaa = {
        (q["perturbation_type"], q["systematic_gene_name"], q.get("copy_number"))
        for q in records[0]["experiment"]["genotype"]["perturbations"]
    }
    assert aaa == {
        ("sequence_variant", "YBR001W", None),
        ("natural_gene_presence", "2-EC1118_1F14_0012g", 2.5),
    }
    yau = {
        (q["perturbation_type"], q["systematic_gene_name"], q.get("copy_number"))
        for q in records[35]["experiment"]["genotype"]["perturbations"]
    }
    assert yau == {
        ("sequence_variant", "YAL001C", None),
        ("natural_gene_absence", "YBR001W", None),
        ("natural_gene_presence", "4-contig_7", 1.0),
    }
    acetate = _CODES.index("YPACETATE")
    hu = 35 + _CODES.index("YPDHU")
    assert (
        records[acetate]["experiment"]["phenotype"]["environment_response"]
        == 0.518089736
    )
    assert records[hu]["experiment"]["phenotype"]["environment_response"] == 0.0
    variant = next(
        q
        for q in records[0]["experiment"]["genotype"]["perturbations"]
        if q["perturbation_type"] == "sequence_variant"
    )
    assert variant["sequence_ref"]["member"] == "YBR001W.fasta#AAA_YBR001W_NTH2"
    assert variant["sequence_ref"]["sha256"] == hashlib.sha256(_tarball()).hexdigest()
    assert {
        r["reference"]["phenotype_reference"]["environment_response"] for r in records
    } == {1.0}


def test_build_writes_the_ledger_and_reloads_the_gene_set(
    dataset: p.IsolateGrowthPeter2018Dataset, data_root: Path
) -> None:
    ledger = json.loads(
        (Path(dataset.preprocess_dir) / "release_ledger.json").read_text()
    )
    assert {k: ledger[k] for k in (
        "n_isolates_released", "n_values_released", "n_values_served", "refused",
        "n_zero_values", "control_values_not_released", "n_isolates_in_caudal2024_store",
    )} == {
        "n_isolates_released": 2, "n_values_released": 70, "n_values_served": 70,
        "refused": {}, "n_zero_values": 1, "control_values_not_released": 2,
        "n_isolates_in_caudal2024_store": None,
    }  # fmt: skip
    expected = {"YAL001C", "YBR001W", "2-EC1118_1F14_0012g", "4-contig_7"}
    assert set(dataset.gene_set) == expected
    reloaded = p.IsolateGrowthPeter2018Dataset(root=dataset.root)
    assert set(reloaded.compute_gene_set()) == expected
    variants = pd.read_parquet(
        Path(dataset.preprocess_dir) / "sequence_variants.parquet"
    )
    assert sorted(zip(variants.strain_id, variants.systematic_gene_name)) == [
        ("AAA", "YBR001W"),
        ("SACE_YAU", "YAL001C"),
    ]
    with pytest.raises(NotImplementedError):
        dataset.create_experiment()


def test_ledger_refuses_an_unaccounted_count() -> None:
    fields: dict[str, Any] = {
        "pheno_sha256": "x",
        "n_isolates_released": 1,
        "n_conditions_released": 2,
        "n_values_released": 2,
        "n_values_served": 1,
        "refused": {},
        "n_zero_values": 0,
        "paper_measurements": 3,
        "control_values_not_released": 1,
        "n_isolates_in_genome_panel": 1,
        "n_isolates_in_caudal2024_store": None,
        "conditions": {},
    }
    with pytest.raises(ValueError, match="served \\+ refused"):
        p.ReleaseLedger(**fields)
    fields.update(n_values_served=2, paper_measurements=4)
    with pytest.raises(ValueError, match="unreleased control"):
        p.ReleaseLedger(**fields)


# --------------------------------------------------------------------------- #
# The verifier on the synthetic store, with a synthetic library mirror.
# --------------------------------------------------------------------------- #
def _write_library(library_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from openpyxl import Workbook

    key = library_root / p.CITATION_KEY
    (key / "si").mkdir(parents=True)
    quotes = [sv.quote for sv in p.PHENOTYPE_SOURCED_VALUES.values()] + [
        p.MEASUREMENT_COUNT_QUOTE,
        p.NORMALIZATION_QUOTE,
        p.PRECULTURE_QUOTE,
    ]
    (key / "paper.md").write_text("\n".join(quotes) + "\n", encoding="utf-8")
    (key / "si" / "si2.md").write_text(p.REPLICATION_QUOTE + "\n", encoding="utf-8")
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = p.TABLE_S2_SHEET
    sheet.cell(row=p.YPD_ROW, column=1, value=p.CONTROL_CONDITION)
    sheet.cell(row=p.YPD_ROW, column=2, value=p.YPD_CELL)
    for spec in p.build_conditions().values():
        sheet.cell(row=spec.table_s2_row, column=1, value=spec.table_s2_condition)
        sheet.cell(row=spec.table_s2_row, column=2, value=spec.table_s2_media)
    sheet.cell(row=42, column=1, value=p.TABLE_S2_TEMPERATURE_QUOTE)
    workbook.save(key / p.SI_TABLES_XLS)

    def sha(rel: str) -> str:
        return hashlib.sha256((key / rel).read_bytes()).hexdigest()

    monkeypatch.setattr(p, "PAPER_MD_SHA256", sha(p.PAPER_MD))
    monkeypatch.setattr(p, "REPORTING_SUMMARY_MD_SHA256", sha(p.REPORTING_SUMMARY_MD))
    monkeypatch.setattr(p, "SI_TABLES_XLS_SHA256", sha(p.SI_TABLES_XLS))
    monkeypatch.setattr(
        p,
        "PHENOTYPE_SOURCED_VALUES",
        {
            name: SourcedValue(
                value=sv.value,
                provenance=sv.provenance.model_copy(update={"sha256": sha(p.PAPER_MD)}),
                quote=sv.quote,
            )
            for name, sv in p.PHENOTYPE_SOURCED_VALUES.items()
        },
    )


def _verify(
    dataset: p.IsolateGrowthPeter2018Dataset, data_root: Path, library_root: Path
) -> dict[str, Any]:
    report = verify_natural_isolate_response_streaming(
        stream_records(dataset.root),
        dataset_name="isolate_growth_peter2018",
        provenance=Provenance(source_uri="test://synthetic"),
        expected_count=70,
        raw_dir=Path(dataset.raw_dir),
        raw_mirror=data_root / p.RAW_DIR_REL,
        library_root=library_root,
        tarball_sha256=hashlib.sha256(_tarball()).hexdigest(),
        sgd_genes={"YAL001C", "YBR001W"},
        min_s288c_containment=1.0,
    )
    return {r.name: r for r in report.results}


def test_verifier_passes_every_level_on_the_synthetic_store(
    dataset: p.IsolateGrowthPeter2018Dataset,
    data_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = tmp_path / "library"
    _write_library(library, monkeypatch)
    results = _verify(dataset, data_root, library)
    failed = {name: r.message for name, r in results.items() if not r.passed}
    assert failed == {}
    assert results["structural"].message == (
        "70 records validated (2 genotypes validated once each and spliced into their records)"
    )
    assert results["isolate_genotype_identity"].message == (
        "2 isolates, one genotype and one strain_id each, 35 records each"
    )
    assert results["release_fidelity"].details["n_cells"] == 70
    assert results["genotype_round_trip"].details["perturbation_types"] == {
        "sequence_variant": 2,
        "natural_gene_presence": 2,
        "natural_gene_absence": 1,
    }
    assert results["conditions_documented"].details["n_conditions"] == 35
    assert results["s288c_gene_containment"].message == (
        "1.0000 of 2 sequence-variant and absence names are S288C genes (>= 1.0); 0 are not"
    )
    assert results["reference_zero"].details["rule"] == "ratio_reference"


def test_verifier_catches_a_changed_release_cell_and_a_missing_gene(
    dataset: p.IsolateGrowthPeter2018Dataset,
    data_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = tmp_path / "library"
    _write_library(library, monkeypatch)
    raw = Path(dataset.raw_dir) / p.PHENO_NAME
    raw.unlink()
    raw.write_bytes(_gz(_release_text().replace("0.518089736", "0.518089737")))
    report = verify_natural_isolate_response_streaming(
        stream_records(dataset.root),
        dataset_name="x",
        provenance=Provenance(source_uri="test://synthetic"),
        expected_count=71,
        raw_dir=Path(dataset.raw_dir),
        raw_mirror=data_root / p.RAW_DIR_REL,
        library_root=library,
        tarball_sha256="0" * 64,
        sgd_genes={"YAL001C"},
        min_s288c_containment=1.0,
    )
    failed = {r.name: r.message for r in report.results if not r.passed}
    assert failed == {
        "count": "observed 70, expected 71",
        "release_fidelity": "1 stored ratios differ from the release; 70 of 70 cells stored",
        "genotype_round_trip": "0 matrix round-trip failures; 2 variant pointers wrong",
        "s288c_gene_containment": (
            "0.5000 of 2 sequence-variant and absence names are S288C genes (>= 1.0); "
            "1 are not"
        ),
        "published_md5": failed["published_md5"],
    }
    assert failed["published_md5"].startswith("release md5 ")


def test_expected_condition_signatures_cover_the_35_conditions() -> None:
    signatures = expected_condition_signatures(p.build_conditions())
    assert sorted(signatures.values()) == sorted(_CODES)


def test_verify_build_delegates_to_the_family_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torchcell.verification import runners

    calls: list[tuple[str, str, str | None]] = []

    def fake(name: str, data_root: str, abs_root: str | None = None) -> str:
        calls.append((name, data_root, abs_root))
        return "report"

    monkeypatch.setattr(runners, "verify_natural_isolate_dataset", fake)
    assert p.verify_build("/store", data_root="/root") == "report"
    monkeypatch.setenv("DATA_ROOT", "/env-root")
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    assert p.verify_build("/store2") == "report"
    assert calls == [
        ("isolate_growth_peter2018", "/root", "/store"),
        ("isolate_growth_peter2018", "/env-root", "/store2"),
    ]
