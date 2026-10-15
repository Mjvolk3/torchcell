# tests/torchcell/datasets/scerevisiae/test_baryshnikova2010_synthetic.py
# [[tests.torchcell.datasets.scerevisiae.test_baryshnikova2010_synthetic]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_baryshnikova2010_synthetic.py
"""Baryshnikova 2010 SMF loader built end to end on a seven-row synthetic workbook.

The loader's ``process()`` is guarded by release self-checksums (row count 6023,
composition 4635/1082/306, the record count) that only the real release satisfies, so the
build tests repoint those module constants at the fixture with ``monkeypatch`` (the
YeastPhenome precedent) and one test pins the unpatched checksums' exact rejections. The
workbook is written with openpyxl under the release's ``.xls`` name; pandas sniffs the zip
signature and reads it with openpyxl, so ``_read_smf`` runs unchanged. No genome is
consulted (#1045): every allele is stored under its released ORF token.

Fixture rows (allele, fitness, SE), in file order, and what each becomes:

    0  YAL001C          0.95      0.010  deletion, 30 C, n = 80 (array side), record 0
    1  YDL227C          1.000155  0.002  deletion, duplicated id -> strain_id YDL227C.1
    2  YBR001C_damp     0.70      0.020  DAmP, 30 C, n_samples gap, record 2
    3  YBR156C_tsq236   0.50      0.050  TS, 26 C, n_samples gap, record 3
    4  YAR037W          1.02      0.004  retired in R64-4-1: stored as released, record 4
    5  YDL227C          1.031582  0.003  second copy -> strain_id YDL227C.2, record 5
    6  yal002w          0.88      0.011  lowercase: ORF YAL002W (upper-cased), record 6

Composition 5 deletion + 1 damp + 1 ts = 7 rows; 7 records. The bootstrap SE column is
stored as ``fitness_uncertainty`` with type ``bootstrap_se`` and is never divided by
sqrt(n). The deletion and DAmP references are identical objects (BY4741, SGA selection
plate, 30 C, fitness 1.0), so the reference index has two entries: 30 C with members
[0, 1, 2, 4, 5, 6] and 26 C with member [3].
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd
import pytest

from torchcell.data import ManifestPinMismatchError, RawSha256MismatchError
from torchcell.datamodels.media import SGA_DM_SELECTION
from torchcell.datamodels.schema import (
    Environment,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    Genotype,
    Publication,
    ReferenceGenome,
    SampleUnit,
    SgaDampPerturbation,
    SgaKanMxDeletionPerturbation,
    SgaTsAllelePerturbation,
    Temperature,
    UncertaintyType,
)
from torchcell.datasets.scerevisiae import baryshnikova2010 as m
from torchcell.literature.manifest import ROLE_RAW_DATA, ArtifactRecord, Manifest
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

_SI_SHA = "f85ea57ce9f5ec60b633b837ff2c5ee18cc65e113db2f034b892739690d8046e"
_CITATION_KEY = "baryshnikovaQuantitativeAnalysisFitness2010"
_DOI = "10.1038/nmeth.1534"


_ROWS: list[tuple[str, float, float]] = [
    ("YAL001C", 0.95, 0.010),
    ("YDL227C", 1.000155, 0.002),
    ("YBR001C_damp", 0.70, 0.020),
    ("YBR156C_tsq236", 0.50, 0.050),
    ("YAR037W", 1.02, 0.004),
    ("YDL227C", 1.031582, 0.003),
    ("yal002w", 0.88, 0.011),
]


def _write_xls(path: Path, rows: list[tuple[str, float, float]]) -> None:
    """The release sheet: three columns, no header, one row per allele."""
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "S1_SMF_standard_100209"
    for row in rows:
        sheet.append(list(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def _patch_release(
    monkeypatch: pytest.MonkeyPatch,
    *,
    n_rows: int = 7,
    composition: dict[str, int] | None = None,
    expected_records: int = 7,
) -> None:
    """Repoint the three release self-checksums at the fixture."""
    monkeypatch.setattr(m, "_EXPECTED_ROWS", n_rows)
    monkeypatch.setattr(
        m,
        "_SV_COMPOSITION",
        m._SV_COMPOSITION.model_copy(
            update={"value": composition or {"deletion": 5, "damp": 1, "ts": 1}}
        ),
    )
    monkeypatch.setattr(m, "EXPECTED_RECORDS", expected_records)


def _root(tmp_path: Path, slug: str = "smf_baryshnikova2010") -> Path:
    root = tmp_path / slug
    _write_xls(root / "raw" / "SupplementaryData1_SMF.xls", _ROWS)
    return root


@pytest.fixture
def dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> m.SmfBaryshnikova2010Dataset:
    _patch_release(monkeypatch)
    return m.SmfBaryshnikova2010Dataset(root=str(_root(tmp_path)))


def _environment(temperature: float) -> Environment:
    return Environment(
        media=SGA_DM_SELECTION, temperature=Temperature(value=temperature)
    )


_N_SAMPLES_GAP = ProvenanceGap(
    field="n_samples",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=Provenance(
        source_uri="si/si1.md",
        citation_key=_CITATION_KEY,
        sha256=_SI_SHA,
        page="Supplementary Table 2, 'Single mutant fitness'",
    ),
    note="SI Supplementary Table 2 states 80 control screens for the ARRAY-side "
    "(nonessential deletion collection) measurement only; this row is a query strain "
    "(_tsq / _damp) and the query-side control-screen count is not released",
)


def _phenotype(fitness: float, se: float, n_samples: int | None) -> FitnessPhenotype:
    return FitnessPhenotype(
        fitness=fitness,
        fitness_uncertainty=se,
        fitness_uncertainty_type=UncertaintyType.bootstrap_se,
        n_samples=n_samples,
        sample_unit=SampleUnit.screen,
        provenance_gaps=[] if n_samples is not None else [_N_SAMPLES_GAP],
    )


def _reference(temperature: float) -> dict[str, object]:
    note = (
        "the reference fitness 1.0 is the mode-normalization convention (the fitness "
        "distribution's mode is set to 1), not an averaged wild-type measurement, so the "
        "primary reports no replicate count or replicate unit for it"
    )
    return FitnessExperimentReference(
        dataset_name="SmfBaryshnikova2010Dataset",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain="BY4741",
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        ),
        environment_reference=_environment(temperature),
        phenotype_reference=FitnessPhenotype(
            fitness=1.0,
            provenance_gaps=[
                ProvenanceGap(
                    field=field,
                    reason=ProvenanceGapReason.not_reported_by_primary,
                    looked_in=Provenance(
                        source_uri="si/si1.md",
                        citation_key=_CITATION_KEY,
                        sha256=_SI_SHA,
                        page="Supplementary Note 1",
                    ),
                    note=note,
                )
                for field in ("n_samples", "sample_unit")
            ],
        ),
    ).model_dump()


def _records(
    dataset: m.SmfBaryshnikova2010Dataset,
) -> list[tuple[str, str, str, str, float, float, int | None, float]]:
    """(leaf type, ORF, perturbed name, strain_id, fitness, SE, n, temperature)."""
    out = []
    for i in range(len(dataset)):
        experiment = dataset[i]["experiment"]
        (perturbation,) = experiment["genotype"]["perturbations"]
        phenotype = experiment["phenotype"]
        out.append(
            (
                perturbation["perturbation_type"],
                perturbation["systematic_gene_name"],
                perturbation["perturbed_gene_name"],
                perturbation["strain_id"],
                phenotype["fitness"],
                phenotype["fitness_uncertainty"],
                phenotype["n_samples"],
                experiment["environment"]["temperature"]["value"],
            )
        )
    return out


def test_seven_rows_build_seven_records_in_file_order(
    dataset: m.SmfBaryshnikova2010Dataset,
) -> None:
    """Every row is a record, under its released ORF token (row 4, YAR037W, is retired
    in R64-4-1 and stored anyway). The deletion rows store n = 80 at 30 C; the DAmP row
    stores n = None at 30 C; the TS row stores n = None at 26 C. Record 0 is checked
    field by field against a hand-built ``FitnessExperiment``: fitness 0.95, bootstrap
    SE 0.010 used as is, 80 screens.
    """
    assert len(dataset) == 7
    assert _records(dataset) == [
        ("sga_kanmx_deletion", "YAL001C", "YAL001C", "YAL001C", 0.95, 0.010, 80, 30.0),
        ("sga_kanmx_deletion", "YDL227C", "YDL227C", "YDL227C.1", 1.000155, 0.002, 80, 30.0),
        ("damp", "YBR001C", "YBR001C", "YBR001C_damp", 0.70, 0.020, None, 30.0),
        ("temperature_sensitive_allele", "YBR156C", "YBR156C", "YBR156C_tsq236", 0.50, 0.050, None, 26.0),
        ("sga_kanmx_deletion", "YAR037W", "YAR037W", "YAR037W", 1.02, 0.004, 80, 30.0),
        ("sga_kanmx_deletion", "YDL227C", "YDL227C", "YDL227C.2", 1.031582, 0.003, 80, 30.0),
        ("sga_kanmx_deletion", "YAL002W", "YAL002W", "yal002w", 0.88, 0.011, 80, 30.0),
    ]  # fmt: skip
    assert (
        dataset[0]["experiment"]
        == FitnessExperiment(
            dataset_name="SmfBaryshnikova2010Dataset",
            genotype=Genotype(
                perturbations=[
                    SgaKanMxDeletionPerturbation(
                        systematic_gene_name="YAL001C",
                        perturbed_gene_name="YAL001C",
                        strain_id="YAL001C",
                    )
                ]
            ),
            environment=_environment(30.0),
            phenotype=_phenotype(0.95, 0.010, 80),
        ).model_dump()
    )
    assert dataset[0]["experiment"]["phenotype"]["fitness_se"] == 0.010
    assert (
        dataset[0]["publication"]
        == Publication(doi=_DOI, doi_url=f"https://doi.org/{_DOI}").model_dump()
    )


def test_query_side_rows_gap_n_samples_and_the_ts_row_runs_at_26c(
    dataset: m.SmfBaryshnikova2010Dataset,
) -> None:
    """Records 2 (DAmP) and 3 (TS) carry the typed ``n_samples`` gap anchored to SI
    Supplementary Table 2 while still storing the SE; the TS environment is 26 C and its
    reference is the 26 C reference, the DAmP reference is the 30 C one.
    """
    assert (
        dataset[2]["experiment"]
        == FitnessExperiment(
            dataset_name="SmfBaryshnikova2010Dataset",
            genotype=Genotype(
                perturbations=[
                    SgaDampPerturbation(
                        systematic_gene_name="YBR001C",
                        perturbed_gene_name="YBR001C",
                        strain_id="YBR001C_damp",
                    )
                ]
            ),
            environment=_environment(30.0),
            phenotype=_phenotype(0.70, 0.020, None),
        ).model_dump()
    )
    assert (
        dataset[3]["experiment"]
        == FitnessExperiment(
            dataset_name="SmfBaryshnikova2010Dataset",
            genotype=Genotype(
                perturbations=[
                    SgaTsAllelePerturbation(
                        systematic_gene_name="YBR156C",
                        perturbed_gene_name="YBR156C",
                        strain_id="YBR156C_tsq236",
                    )
                ]
            ),
            environment=_environment(26.0),
            phenotype=_phenotype(0.50, 0.050, None),
        ).model_dump()
    )
    assert dataset[2]["reference"] == _reference(30.0)
    assert dataset[3]["reference"] == _reference(26.0)


def test_a_lowercase_row_stores_its_own_token_upper_cased_and_the_raw_id_as_strain(
    dataset: m.SmfBaryshnikova2010Dataset,
) -> None:
    """The record's names are the released ORF token (``yal002w`` upper-cased to
    YAL002W, not the R64 synonym YAL003W); ``strain_id`` keeps the raw id verbatim.
    """
    (perturbation,) = dataset[6]["experiment"]["genotype"]["perturbations"]
    assert (
        perturbation
        == SgaKanMxDeletionPerturbation(
            systematic_gene_name="YAL002W",
            perturbed_gene_name="YAL002W",
            strain_id="yal002w",
        ).model_dump()
    )


def test_reference_index_merges_the_deletion_and_damp_references(
    dataset: m.SmfBaryshnikova2010Dataset,
) -> None:
    """``process`` builds one reference per allele kind, but the deletion and DAmP
    references are identical objects (same medium, 30 C, fitness 1.0, same gaps), so the
    content-hashed index has two entries: 30 C with members [0, 1, 2, 4, 5] and 26 C
    with member [3].
    """
    index = json.loads(
        (
            Path(dataset.root) / "preprocess" / "experiment_reference_index.json"
        ).read_text()
    )
    assert [entry["member_indices"] for entry in index] == [[0, 1, 2, 4, 5, 6], [3]]
    assert index[0]["reference"] == _reference(30.0)
    assert index[1]["reference"] == _reference(26.0)


def test_side_files_gene_set_drop_ledger_and_build_manifest(
    dataset: m.SmfBaryshnikova2010Dataset,
) -> None:
    """``gene_set.json`` is the six released ORFs sorted (YDL227C once), the ledger
    records that nothing is dropped, and the build manifest names the root slug, the
    loader and this git HEAD.
    """
    preprocess = Path(dataset.root) / "preprocess"
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "YAL001C",
        "YAL002W",
        "YAR037W",
        "YBR001C",
        "YBR156C",
        "YDL227C",
    ]
    assert json.loads((preprocess / "dropped_records.json").read_text()) == {
        "rule": m.RETENTION_RULE,
        "n_raw_rows": 7,
        "n_dropped": 0,
        "n_kept": 7,
    }
    manifest = json.loads((preprocess / "build_manifest.json").read_text())
    head = subprocess.run(
        ["git", "-C", str(Path(m.__file__).parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert manifest["dataset_name"] == "smf_baryshnikova2010"
    assert manifest["loader_class"] == "SmfBaryshnikova2010Dataset"
    assert (
        manifest["loader_module"] == "torchcell.datasets.scerevisiae.baryshnikova2010"
    )
    assert manifest["hostname"] == socket.gethostname()
    assert manifest["torchcell_commit"] == head
    assert {
        "FitnessExperiment",
        "SgaKanMxDeletionPerturbation",
        "SgaDampPerturbation",
        "SgaTsAllelePerturbation",
    } <= set(manifest["closure"])
    assert dataset.experiment_class is FitnessExperiment
    assert dataset.reference_class is FitnessExperimentReference
    assert dataset.raw_file_names == ["SupplementaryData1_SMF.xls"]
    with pytest.raises(NotImplementedError):
        dataset.create_experiment()
    frame = pd.DataFrame({"untouched": [True]})
    assert dataset.preprocess_raw(frame) is frame


def test_unpatched_self_checksums_reject_any_file_but_the_release(
    tmp_path: Path,
) -> None:
    """Without the patch, the seven-row file fails the row count (7 != 6023) and a
    6023-row all-deletion file fails the composition check with both dicts in the message.
    """
    with pytest.raises(
        RuntimeError, match=r"SMF row-count self-checksum failed: 7 != 6023"
    ):
        m.SmfBaryshnikova2010Dataset(root=str(_root(tmp_path, "rows")))
    root = tmp_path / "composition"
    _write_xls(
        root / "raw" / "SupplementaryData1_SMF.xls",
        [(f"YAL{i:03d}W", 1.0, 0.01) for i in range(6023)],
    )
    with pytest.raises(
        RuntimeError,
        match=(
            r"SMF composition self-checksum failed: \{'deletion': 6023\} != "
            r"\{'deletion': 4635, 'damp': 1082, 'ts': 306\}"
        ),
    ):
        m.SmfBaryshnikova2010Dataset(root=str(root))


def test_the_record_count_oracle_raises_after_the_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``EXPECTED_RECORDS`` 5 against seven written rows raises ``wrote 7 records,
    expected 5``, after the LMDB is written.
    """
    _patch_release(monkeypatch, expected_records=5)
    with pytest.raises(RuntimeError, match=r"wrote 7 records, expected 5"):
        m.SmfBaryshnikova2010Dataset(root=str(_root(tmp_path, "count")))
    assert (tmp_path / "count" / "processed" / "lmdb").is_dir()


def _mirror_manifest(xls: Path, sha256: str) -> Manifest:
    return Manifest(
        citation_key=_CITATION_KEY,
        files=[
            ArtifactRecord(
                path="data/SupplementaryData1_SMF.xls",
                role=ROLE_RAW_DATA,
                bytes=xls.stat().st_size,
                sha256=sha256,
            )
        ],
        provenance_complete=True,
    )


def test_download_links_the_mirror_file_after_verifying_it_against_the_module_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no raw file, ``download()`` reads ``$DATA_ROOT/torchcell-raw/<key>/
    manifest.json``: a missing manifest raises ``FileNotFoundError`` naming it, a listed
    file absent from the mirror raises ``required raw artifact missing``. ``XLS_SHA256``
    is the one pin (issue #561): a manifest recording another digest raises
    ``ManifestPinMismatchError`` naming both, mirror bytes off the pin raise
    ``RawSha256MismatchError`` naming both, and a manifest and bytes matching the pin
    symlink the file into ``raw/`` and the build proceeds to its six records.
    """
    _patch_release(monkeypatch)
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    mirror = data_root / "torchcell-raw" / _CITATION_KEY
    manifest_path = mirror / "manifest.json"
    with pytest.raises(FileNotFoundError, match=str(manifest_path)):
        m.SmfBaryshnikova2010Dataset(root=str(tmp_path / "a"))
    xls = mirror / "data" / "SupplementaryData1_SMF.xls"
    staged = tmp_path / "staged.xls"
    _write_xls(staged, _ROWS)
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(_mirror_manifest(staged, "0" * 64).model_dump_json())
    with pytest.raises(
        RuntimeError, match=f"required raw artifact missing from mirror: {xls}"
    ):
        m.SmfBaryshnikova2010Dataset(root=str(tmp_path / "b"))
    xls.parent.mkdir()
    xls.write_bytes(staged.read_bytes())
    digest = hashlib.sha256(xls.read_bytes()).hexdigest()
    monkeypatch.setattr(m, "XLS_SHA256", digest)
    with pytest.raises(ManifestPinMismatchError) as off_manifest:
        m.SmfBaryshnikova2010Dataset(root=str(tmp_path / "c"))
    assert str(off_manifest.value) == (
        f"raw-mirror manifest records sha256 {'0' * 64} for "
        f"data/SupplementaryData1_SMF.xls, but the loader pins {digest}"
    )
    assert list((tmp_path / "c" / "raw").iterdir()) == []
    monkeypatch.setattr(m, "XLS_SHA256", "f" * 64)
    manifest_path.write_text(_mirror_manifest(xls, "f" * 64).model_dump_json())
    with pytest.raises(RawSha256MismatchError) as off_bytes:
        m.SmfBaryshnikova2010Dataset(root=str(tmp_path / "c2"))
    assert str(off_bytes.value) == (
        f"sha256 mismatch for {xls}: expected {'f' * 64}, observed {digest}"
    )
    assert list((tmp_path / "c2" / "raw").iterdir()) == []
    monkeypatch.setattr(m, "XLS_SHA256", digest)
    manifest_path.write_text(_mirror_manifest(xls, digest).model_dump_json())
    dataset = m.SmfBaryshnikova2010Dataset(root=str(tmp_path / "d"))
    # a raw file already in place is left alone: download() only links what is absent
    raw_e = tmp_path / "e" / "raw"
    raw_e.mkdir(parents=True)
    (raw_e / "SupplementaryData1_SMF.xls").write_bytes(xls.read_bytes())
    m.SmfBaryshnikova2010Dataset(root=str(tmp_path / "e")).download()
    assert not (raw_e / "SupplementaryData1_SMF.xls").is_symlink()
    link = tmp_path / "d" / "raw" / "SupplementaryData1_SMF.xls"
    assert link.is_symlink() and os.readlink(link) == str(xls)
    assert len(dataset) == 7
    assert m.manifest_sha256(m.load_manifest(str(data_root)), m.XLS_REL) == digest
    with pytest.raises(
        KeyError, match="data/other.xls is not in the raw-mirror manifest"
    ):
        m.manifest_sha256(m.load_manifest(str(data_root)), "data/other.xls")


def test_deposit_raw_mirror_is_idempotent_by_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A workbook whose digest is not the pinned release digest is refused with both
    digests. With the pin repointed at the fixture's digest the deposit copies the file,
    writes a manifest whose one record carries the Springer ESM retrieval, leaves an
    identical existing copy alone, and refuses an existing copy with another digest.
    """
    staged = tmp_path / "staged.xls"
    _write_xls(staged, _ROWS)
    digest = hashlib.sha256(staged.read_bytes()).hexdigest()
    data_root = tmp_path / "data_root"
    with pytest.raises(
        RuntimeError, match=f"{staged} sha256 {digest} != expected {m.XLS_SHA256}"
    ):
        m.deposit_raw_mirror(
            xls_path=staged, retrieved_at="2026-09-27", data_root=str(data_root)
        )
    monkeypatch.setattr(m, "XLS_SHA256", digest)
    root = m.deposit_raw_mirror(
        xls_path=staged, retrieved_at="2026-09-27", data_root=str(data_root)
    )
    assert root == data_root / "torchcell-raw" / _CITATION_KEY
    dest = root / "data" / "SupplementaryData1_SMF.xls"
    assert dest.read_bytes() == staged.read_bytes()
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["citation_key"] == _CITATION_KEY
    assert manifest["doi"] == _DOI
    assert manifest["provenance_complete"] is True
    assert manifest["si_expected"] == ["Supplementary Data 1 (single-mutant fitness)"]
    (record,) = manifest["files"]
    assert record["path"] == "data/SupplementaryData1_SMF.xls"
    assert record["role"] == "raw_data"
    assert record["bytes"] == staged.stat().st_size
    assert record["sha256"] == digest
    assert record["retrieval"]["method"] == "springer_esm"
    assert (
        record["retrieval"]["retriever"] == "torchcell.literature.retrieve.springer_esm"
    )
    assert record["retrieval"]["retrieved_at"] == "2026-09-27"
    assert record["retrieval"]["source_url"] == record["source"] == m.XLS_URL
    # a second deposit of the same bytes is a no-op on the file
    m.deposit_raw_mirror(
        xls_path=staged, retrieved_at="2026-09-28", data_root=str(data_root)
    )
    assert dest.read_bytes() == staged.read_bytes()
    other = tmp_path / "other.xls"
    _write_xls(other, _ROWS[:3])
    monkeypatch.setattr(m, "XLS_SHA256", hashlib.sha256(other.read_bytes()).hexdigest())
    with pytest.raises(
        RuntimeError, match=f"{dest} exists with a different sha256; refusing"
    ):
        m.deposit_raw_mirror(
            xls_path=other, retrieved_at="2026-09-28", data_root=str(data_root)
        )


def test_a_raw_file_off_the_pin_is_refused_at_build_time(
    off_pin_raw: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (issue #518's sweep): with the workbook already in ``raw/`` PyG skips
    ``download()``, so ``process()`` verifies it against ``XLS_SHA256`` first and raises
    ``RawSha256MismatchError`` before a row is read; no store is written and the file
    is left as found.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    staged = off_pin_raw(m, ["SupplementaryData1_SMF.xls"])
    raw = staged.root / "raw" / "SupplementaryData1_SMF.xls"
    with pytest.raises(RawSha256MismatchError) as err:
        m.SmfBaryshnikova2010Dataset(root=str(staged.root))
    assert str(err.value) == (
        f"sha256 mismatch for {raw}: expected "
        "086bfadf2684f28940500dd87e3be74c53a957448d2016f7a02370540da8a04e, "
        f"observed {staged.observed}"
    )
    assert list((staged.root / "processed").iterdir()) == []
    assert not (staged.root / "preprocess").exists()
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == staged.observed
