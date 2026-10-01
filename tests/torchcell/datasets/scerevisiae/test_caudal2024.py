# tests/torchcell/datasets/scerevisiae/test_caudal2024.py
# [[tests.torchcell.datasets.scerevisiae.test_caudal2024]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_caudal2024.py
"""Caudal 2024 pan-transcriptome loader: the expression columns it consumes, the
aggregation and rounding, the refusals, a stale raw file, the SGD FASTA tail and ``main``.

2026.09.30 (Phase 17). The end-to-end build is pinned in ``test_caudal2024_synthetic.py``
(two isolates AAA and SACE_YAU off the synthetic Peter tarball, matrices and SGD FASTA);
its fixture helpers are reused here, with no mirror and no ``$DATA_ROOT`` outside
``tmp_path``.

Columns. The released Datafile 1 is long format with the key columns ``systematic_name,
ORF, Strain, count, tpm, Pangenome(Core/Accessory), Group, Precence_in_S288c`` (dendron
note ``torchcell.datasets.scerevisiae.caudal2024``, 2026.07.07). The loader reads exactly
``Strain, systematic_name, count, tpm`` (``usecols``); no replicate or per-sample column is
consumed (the 29-replicate table, Datafile 2, is not a raw file of this loader). The
fixture table in that header order:

    AAA       YAL001C  ORF 1-YAL001C    10    1.5
    AAA       YAL001C  ORF 1-YAL001C_b   5    0.5   (second allele row: summed, 15 / 2.0)
    AAA       YBR001W                   20    4.0
    AAA       (blank)                    7    0.7   (blank gene: dropped by groupby)
    SACE_YAU  YAL001C                   30    3.0
    SACE_YAU  YBR001W                   12.5  1.0   (count round(12.5) = 12, banker's)

Records: AAA tpm {YAL001C 2.0, YBR001W 4.0}, count {15, 20}; SACE_YAU tpm {YAL001C 3.0,
YBR001W 1.0}, count {30, 12}. Reference = mean over the isolates carrying the gene:
YAL001C tpm 2.5, count round(22.5) = 22; YBR001W tpm 2.5, count round(16.25) = 16. The
genotypes are the synthetic file's, unchanged.

Refusals: a table without ``tpm`` is pandas' ``Usecols do not match columns, columns
expected but not found: ['tpm']``; a zip with no ``.tab`` member raises a bare
``StopIteration`` (a Finding). SGD FASTA whose last record is a chromosome keeps it; an
untagged record in the middle is dropped. ``main`` prints the streaming line of the base
class, ``len = 2``, record 0's perturbation-type counts (presence 1, variant 1, absence
1, in that order), its 2 phenotype genes and the S288C genome reference.
"""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
import zipfile
from pathlib import Path
from typing import IO, Any

import pytest

from tests.torchcell.datasets.scerevisiae.test_caudal2024_synthetic import (
    _COLUMNS,
    _EXPECTED,
    _PRESENCE,
    _matrix_gz,
    _write_mirror,
    _write_peter_tier,
    _write_raw,
    _write_sgd_tier,
)
from torchcell.data import RawSha256MismatchError
from torchcell.data.experiment_dataset import verify_raw_files
from torchcell.datamodels.schema import ReferenceGenome
from torchcell.datasets.scerevisiae import caudal2024 as m

_RELEASED_HEADER = (
    "systematic_name,ORF,Strain,count,tpm,Pangenome(Core/Accessory),Group,"
    "Precence_in_S288c\n"
)
_RELEASED_ROWS = (
    "YAL001C,1-YAL001C,AAA,10,1.5,Core,G1,Yes\n"
    "YAL001C,1-YAL001C_b,AAA,5,0.5,Core,G1,Yes\n"
    "YBR001W,4-YBR001W,AAA,20,4.0,Core,G1,Yes\n"
    ",ORFX,AAA,7,0.7,Accessory,G1,No\n"
    "YAL001C,1-YAL001C,SACE_YAU,30,3.0,Core,G2,Yes\n"
    "YBR001W,4-YBR001W,SACE_YAU,12.5,1.0,Core,G2,Yes\n"
)


def _zip(members: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, text in members.items():
            archive.writestr(name, text)
    return buffer.getvalue()


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    _write_sgd_tier(data_root)
    _write_peter_tier(data_root)
    root = tmp_path / "caudal"
    _write_raw(root / "raw")
    return root


def _root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, table: bytes) -> Path:
    """The synthetic raw set with the Caudal zip replaced by ``table``."""
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    _write_sgd_tier(data_root)
    _write_peter_tier(data_root)
    root = tmp_path / "caudal"
    _write_raw(root / "raw")
    (root / "raw" / m.CAUDAL_ZIP_BASENAME).write_bytes(table)
    return root


def test_only_strain_gene_count_and_tpm_are_read_and_allele_rows_are_summed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The released column set in its own order builds; the ORF, pangenome, group and
    S288C-presence columns change nothing; two rows of one (strain, gene) are summed;
    counts round half to even; the reference is the mean over carriers.
    """
    table = _zip({"final.tab": _RELEASED_HEADER + _RELEASED_ROWS})
    dataset = m.CaudalPanTranscriptome2024Dataset(
        root=str(_root(tmp_path, monkeypatch, table))
    )
    assert len(dataset) == 2
    phenotypes = [dataset[i]["experiment"]["phenotype"] for i in range(2)]
    assert [(p["expression_tpm"], p["expression_count"]) for p in phenotypes] == [
        ({"YAL001C": 2.0, "YBR001W": 4.0}, {"YAL001C": 15, "YBR001W": 20}),
        ({"YAL001C": 3.0, "YBR001W": 1.0}, {"YAL001C": 30, "YBR001W": 12}),
    ]
    reference = dataset[0]["reference"]["phenotype_reference"]
    assert (reference["expression_tpm"], reference["expression_count"]) == (
        {"YAL001C": 2.5, "YBR001W": 2.5},
        {"YAL001C": 22, "YBR001W": 16},
    )
    assert [dataset[i]["experiment"]["genotype"] for i in range(2)] == [
        e.model_dump()["genotype"] for e in _EXPECTED
    ]


def test_a_row_with_a_blank_gene_is_dropped_without_a_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: ``_load_caudal`` groups by (Strain, systematic_name) with pandas' default
    ``dropna=True`` (caudal2024.py line 463), so a row whose ``systematic_name`` is blank
    (AAA, 7 counts) vanishes from the record, the reference and the logs, and is not
    stored as a ``"nan"`` gene either. Pinned until a blank gene is refused or counted.
    """
    table = _zip({"final.tab": _RELEASED_HEADER + _RELEASED_ROWS})
    dataset = m.CaudalPanTranscriptome2024Dataset(
        root=str(_root(tmp_path, monkeypatch, table))
    )
    counts = dataset[0]["experiment"]["phenotype"]["expression_count"]
    assert sorted(counts) == ["YAL001C", "YBR001W"]
    assert sum(counts.values()) == 35


def test_a_table_without_tpm_is_refused_by_the_column_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = _zip({"final.tab": "Strain,systematic_name,count\nAAA,YAL001C,10\n"})
    with pytest.raises(
        ValueError,
        match=r"Usecols do not match columns, columns expected but not found: "
        r"\['tpm'\]",
    ):
        m.CaudalPanTranscriptome2024Dataset(
            root=str(_root(tmp_path, monkeypatch, table))
        )


def test_a_zip_without_a_tab_member_raises_a_bare_stop_iteration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: the member lookup is ``next(n for n in zf.namelist() if
    n.endswith(".tab"))`` (line 448) with no default, so an archive holding only
    ``final.csv`` raises ``StopIteration`` with no message naming the archive. Pinned
    until the lookup raises a named error.
    """
    table = _zip({"final.csv": "Strain,systematic_name,count,tpm\n"})
    with pytest.raises(StopIteration) as excinfo:
        m.CaudalPanTranscriptome2024Dataset(
            root=str(_root(tmp_path, monkeypatch, table))
        )
    assert excinfo.value.args == ()


def test_sgd_chromosomes_keeps_a_final_chromosome_and_drops_a_middle_plasmid(
    tmp_path: Path,
) -> None:
    """The last record is flushed after the loop (line 163) only when it is tagged; an
    untagged record between two chromosomes contributes nothing.
    """
    path = tmp_path / "sgd.fsa"
    path.write_text(
        ">ref [chromosome=I]\nacgt\n"
        ">ref [location=plasmid]\nTTTT\n"
        ">ref [chromosome=XVI]\nGG\nCC\n"
    )
    assert m._sgd_chromosomes(str(path)) == {"I": "ACGT", "XVI": "GGCC"}


def test_a_regular_member_that_cannot_be_extracted_loses_its_variants_silently(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: the ``extracted is None`` guard (lines 522-523) is unreachable for a real
    archive, since ``tarfile`` returns a file object for every member that passes
    ``isfile()``. Forced here by returning None for ``YAL001C.fasta``: that gene's
    variant for SACE_YAU disappears with no log, and the all-isolates check still passes
    because both isolates appear in the other gene files. Pinned as the guard's behavior
    until it is removed or made to raise.
    """
    original = tarfile.TarFile.extractfile

    def extractfile(self: tarfile.TarFile, member: Any) -> IO[bytes] | None:
        if getattr(member, "name", member) == "YAL001C.fasta":
            return None
        return original(self, member)

    monkeypatch.setattr(tarfile.TarFile, "extractfile", extractfile)
    dataset = m.CaudalPanTranscriptome2024Dataset(root=str(root))
    variants = [
        p["systematic_gene_name"]
        for p in dataset[1]["experiment"]["genotype"]["perturbations"]
        if p["perturbation_type"] == "sequence_variant"
    ]
    assert variants == ["YBR001W"]


def test_a_stale_raw_matrix_is_refused_at_build_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (issue #518's sweep): ``download`` links only what ``raw/`` lacks, so a
    leftover presence matrix (AAA carrying X3, ``YAL005C``) stays in place, and
    ``process`` verifies it against the genomes tier's pin before reading it: the build
    raises ``RawSha256MismatchError`` naming the raw file, the tier pin and the stale
    digest, and no store is written.
    """
    monkeypatch.setattr(m, "verify_raw_files", verify_raw_files)
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    _write_sgd_tier(data_root)
    raw_bytes = _write_mirror(data_root)
    monkeypatch.setattr(
        m,
        "CAUDAL_ZIP_SHA256",
        hashlib.sha256(raw_bytes[m.CAUDAL_ZIP_BASENAME]).hexdigest(),
    )
    monkeypatch.setattr(
        m,
        "REFGENE_TAR_SHA256",
        hashlib.sha256(raw_bytes[m.REFGENE_TAR_NAME]).hexdigest(),
    )
    raw = tmp_path / "caudal" / "raw"
    raw.mkdir(parents=True)
    stale = {**_PRESENCE, "AAA": ["1", "1", "1", "1", "0", "1"]}
    stale_bytes = _matrix_gz(stale, _COLUMNS)
    (raw / m.PRESENCE_NAME).write_bytes(stale_bytes)
    with pytest.raises(RawSha256MismatchError) as err:
        m.CaudalPanTranscriptome2024Dataset(root=str(tmp_path / "caudal"))
    assert str(err.value) == (
        f"sha256 mismatch for {raw / m.PRESENCE_NAME}: expected "
        f"{hashlib.sha256(raw_bytes[m.PRESENCE_NAME]).hexdigest()}, "
        f"observed {hashlib.sha256(stale_bytes).hexdigest()}"
    )
    assert [os.path.islink(raw / n) for n in m.RAW_FILES] == [True, True, False, True]
    assert list((tmp_path / "caudal" / "processed").iterdir()) == []


def test_main_builds_under_data_root_and_prints_record_zero(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main`` builds ``$DATA_ROOT/data/torchcell/caudal_pantranscriptome2024`` (the
    synthetic raw set is placed there) with ``load_dotenv`` stubbed.
    """
    data_root = Path(os.environ["DATA_ROOT"])
    target = data_root / "data" / "torchcell" / "caudal_pantranscriptome2024"
    target.parent.mkdir(parents=True)
    root.rename(target)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    m.main()
    genome = ReferenceGenome(
        species="Saccharomyces cerevisiae", strain="S288C"
    ).model_dump()
    counts = {
        "natural_gene_presence": 1,
        "sequence_variant": 1,
        "natural_gene_absence": 1,
    }
    assert capsys.readouterr().out == (
        "Computing experiment_reference_index (streaming)...\n"
        "len = 2\n"
        f"record[0] perturbation-type counts: {counts}\n"
        "record[0] phenotype gene count: 2\n"
        f"record[0] genome_reference: {genome}\n"
    )
    assert (target / "processed" / "lmdb" / "data.mdb").is_file()
