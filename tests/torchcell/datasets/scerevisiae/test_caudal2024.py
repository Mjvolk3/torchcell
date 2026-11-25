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
``Strain, systematic_name, count, tpm`` (``usecols``). Datafile 2, the replicate-culture
table, is read since 2026.11.24 (issue #1178; tests at the end of this file). The
fixture table in that header order:

    AAA       YAL001C  ORF 1-YAL001C    10    1.5
    AAA       YAL001C  ORF 1-YAL001C_b   5    0.5   (second allele row: summed, 15 / 2.0)
    AAA       YBR001W                   20    4.0
    AAA       (blank)                    7    0.7   (blank ``absent`` row: dropped, counted)
    SACE_YAU  YAL001C                   30    3.0
    SACE_YAU  YBR001W                   12.5  1.0   (count round(12.5) = 12, banker's)

Records: AAA tpm {YAL001C 2.0, YBR001W 4.0}, count {15, 20}; SACE_YAU tpm {YAL001C 3.0,
YBR001W 1.0}, count {30, 12}. Reference = mean over the isolates carrying the gene:
YAL001C tpm 2.5, count round(22.5) = 22; YBR001W tpm 2.5, count round(16.25) = 16. The
genotypes are the synthetic file's, unchanged.

Refusals: a table without ``tpm`` is pandas' ``Usecols do not match columns, columns
expected but not found: ['tpm']``; a zip without exactly one ``.tab`` member raises
``MissingTabMemberError`` naming the archive and its members, and a tarball member that
cannot be extracted raises ``UnextractableMemberError`` (issue #541, fixed 2026.10.01). SGD FASTA whose last record is a chromosome keeps it; an
untagged record in the middle is dropped. ``main`` prints the streaming line of the base
class, ``len = 2``, record 0's perturbation-type counts (presence 1, variant 1, absence
1, in that order), its 2 phenotype genes and the S288C genome reference.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import os.path as osp
import re
import tarfile
import zipfile
from pathlib import Path
from typing import IO, Any

import pandas as pd
import pytest

from tests.torchcell.datasets.scerevisiae.test_caudal2024_synthetic import (
    _COLUMNS,
    _EXPECTED,
    _PRESENCE,
    _WORKBOOK,
    _matrix_gz,
    _write_mirror,
    _write_peter_tier,
    _write_raw,
    _write_sgd_tier,
)
from torchcell.data import RawSha256MismatchError
from torchcell.data.experiment_dataset import verify_raw_files
from torchcell.datamodels.schema import (
    ArtifactRef,
    ReferenceGenome,
    RNASeqExpressionPhenotype,
)
from torchcell.datasets.scerevisiae import caudal2024 as m
from torchcell.sequence import natural_isolate as ni
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.verification.sourced import audit_sourced_value

_RELEASED_HEADER = (
    "systematic_name,ORF,Strain,count,tpm,Pangenome(Core/Accessory),Group,"
    "Precence_in_S288c,Ortholog_in_SGD_2010,pan_absence\n"
)
_RELEASED_ROWS = (
    "YAL001C,1-YAL001C,AAA,10,1.5,Core,G1,Yes,,present\n"
    "YAL001C,1-YAL001C_b,AAA,5,0.5,Core,G1,Yes,,present\n"
    "YBR001W,4-YBR001W,AAA,20,4.0,Core,G1,Yes,,present\n"
    ",ORFX,AAA,7,0.7,Accessory,G1,No,,absent\n"
    "YAL001C,1-YAL001C,SACE_YAU,30,3.0,Core,G2,Yes,,present\n"
    "YBR001W,4-YBR001W,SACE_YAU,12.5,1.0,Core,G2,Yes,,present\n"
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


# Issue #598: a blank-systematic_name row of each ledger class, in built isolates AAA and
# SACE_YAU (and one in the excluded XTRA_ABC, which never reaches the ledger).
_LEDGER_HEADER = (
    "Strain,systematic_name,ORF,Ortholog_in_SGD_2010,pan_absence,count,tpm\n"
)
_LEDGER_ROWS = (
    "AAA,YAL001C,1-YAL001C,,present,10,1.5\n"
    "AAA,YBR001W,4-YBR001W,,present,20,4.0\n"
    "SACE_YAU,YAL001C,1-YAL001C,,present,30,3.0\n"
    "SACE_YAU,,YBR001W,YBR001W,present,8,2.0\n"
    "SACE_YAU,,X5.contig_7,,present,4,1.0\n"
    "AAA,,ORFX,,absent,7,0.7\n"
    "SACE_YAU,,X7.contig_9,,bad annotation,3,0.3\n"
    "XTRA_ABC,,ORFZ,,unannotated,9,0.9\n"
)


def _frame(rows: str) -> pd.DataFrame:
    frame = pd.read_csv(io.StringIO(_LEDGER_HEADER + rows))
    frame["Strain"] = frame["Strain"].astype(str)
    return frame


def test_blank_name_rows_are_served_or_dropped_by_class_and_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #598: SACE_YAU's blank ``present`` row whose ORF and ortholog are the
    served name YBR001W is served as YBR001W (8 / 2.0); its blank ``present`` pangenome
    row ``X5.contig_7`` is served as ``X5-contig_7``, the form Datafile 1 gives named
    accessory ORFs; its ``bad annotation`` row ``X7.contig_9`` is served the same way
    (as released, with the class-definition gap on the rule); the ``absent`` row is
    dropped as the paper does; the ``XTRA_ABC`` row never reaches the ledger. The
    reference is the mean over the isolates holding each key: YAL001C (1.5 + 3.0) / 2,
    YBR001W (4.0 + 2.0) / 2, X5-contig_7 1.0 and X7-contig_9 0.3 from SACE_YAU alone.
    """
    table = _zip({"final.tab": _LEDGER_HEADER + _LEDGER_ROWS})
    root = _root(tmp_path, monkeypatch, table)
    dataset = m.CaudalPanTranscriptome2024Dataset(root=str(root))
    phenotypes = [dataset[i]["experiment"]["phenotype"] for i in range(2)]
    assert [(p["expression_tpm"], p["expression_count"]) for p in phenotypes] == [
        ({"YAL001C": 1.5, "YBR001W": 4.0}, {"YAL001C": 10, "YBR001W": 20}),
        (
            {"YAL001C": 3.0, "YBR001W": 2.0, "X5-contig_7": 1.0, "X7-contig_9": 0.3},
            {"YAL001C": 30, "YBR001W": 8, "X5-contig_7": 4, "X7-contig_9": 3},
        ),
    ]
    reference = dataset[0]["reference"]["phenotype_reference"]
    assert reference["expression_tpm"] == {
        "YAL001C": 2.25,
        "YBR001W": 3.0,
        "X5-contig_7": 1.0,
        "X7-contig_9": 0.3,
    }
    ledger = json.loads(
        (root / "preprocess" / "blank_systematic_name_ledger.json").read_text()
    )
    assert (ledger["n_rows"], ledger["n_rows_named"], ledger["n_rows_blank"]) == (
        7,
        3,
        4,
    )
    assert ledger["counts"] == {
        "present_s288c_homolog": 1,
        "present_pangenome_orf": 1,
        "absent": 1,
        "bad_annotation": 1,
        "unannotated": 0,
    }
    assert ledger["served_ids"] == {"YBR001W": 1, "X5-contig_7": 1, "X7-contig_9": 1}
    assert ledger["tpm_by_class"]["absent"] == 0.7
    assert ledger["tpm_by_class"]["bad_annotation"] == 0.3
    assert ledger["named_pan_absence_counts"] == {"present": 3}
    assert [r["row_class"] for r in ledger["rules"]] == [
        c.value for c in m.BlankRowClass
    ]


@pytest.mark.parametrize("value", ["duplicated", None])
def test_a_blank_name_row_of_an_unledgered_class_is_refused(value: str | None) -> None:
    """A blank row whose ``pan_absence`` is no ledger class (here a class only named
    rows carry, or no class at all) raises instead of taking any silent path.
    """
    cell = "" if value is None else value
    frame = _frame(f"AAA,,ORFQ,,{cell},1,0.1\n")
    with pytest.raises(m.UnclassifiedBlankRowError, match=r"1 blank-systematic_name"):
        m.resolve_gene_ids(frame)


def test_a_served_blank_row_that_repeats_a_named_gene_is_refused() -> None:
    """AAA already has a named YBR001W row; a blank ``present`` YBR001W homolog row in
    the same isolate would give the record two values for one key.
    """
    frame = _frame(
        "AAA,YBR001W,4-YBR001W,,present,20,4.0\nAAA,,YBR001W,YBR001W,present,8,2.0\n"
    )
    with pytest.raises(m.GeneIdCollisionError, match=r"\('AAA', 'YBR001W'\)"):
        m.resolve_gene_ids(frame)


def test_the_pangenome_rule_refuses_an_orf_that_is_no_pangenome_id() -> None:
    """A blank ``present`` row whose ORF looks like an S288C name but is no served
    systematic_name (nor its own ortholog) falls to the pangenome-id rule, which refuses
    it rather than serving a key no rule vouches for.
    """
    frame = _frame(
        "AAA,YAL001C,1-YAL001C,,present,1,1.0\nAAA,,YCR001W,,present,1,1.0\n"
    )
    with pytest.raises(m.GeneIdCollisionError, match=r"no pangenome id: \['YCR001W'\]"):
        m.resolve_gene_ids(frame)


def test_the_pangenome_rule_refuses_an_id_a_named_row_serves() -> None:
    """A blank ``present`` pangenome row whose served id equals a named row's
    systematic_name (in any isolate) would merge two ORFs under one key.
    """
    frame = _frame(
        "AAA,X5-contig_7,X5.contig_7,,present,1,1.0\n"
        "SACE_YAU,,X5.contig_7,,present,1,1.0\n"
    )
    with pytest.raises(m.GeneIdCollisionError, match=r"\['X5-contig_7'\]"):
        m.resolve_gene_ids(frame)


def test_a_blank_row_rule_needs_exactly_one_basis() -> None:
    """``BlankRowRule`` refuses a class with both a sourced definition and a gap."""
    rule = m.BLANK_ROW_RULES[m.BlankRowClass.absent]
    with pytest.raises(ValueError, match=r"exactly one of definition / gap"):
        m.BlankRowRule(
            **{
                **rule.model_dump(),
                "gap": m.BLANK_ROW_RULES[m.BlankRowClass.bad_annotation].gap,
            }
        )


_LIBRARY = osp.join(os.environ.get("DATA_ROOT", ""), "torchcell-library")
_MIRROR_ZIP = osp.join(os.environ.get("DATA_ROOT", ""), m.CAUDAL_ZIP_REL)
_needs_mirror = pytest.mark.skipif(
    not (osp.exists(_MIRROR_ZIP) and osp.isdir(_LIBRARY)),
    reason="requires the Caudal mirror + Peter genomes tier at $DATA_ROOT",
)


@pytest.mark.data
@_needs_mirror
def test_the_blank_row_definitions_audit_against_the_mirrored_methods() -> None:
    """Every sourced class definition's quote is in ``methods.md`` at its pinned sha256."""
    for rule in m.BLANK_ROW_RULES.values():
        if rule.definition is not None:
            assert audit_sourced_value(rule.definition, _LIBRARY).passed, rule.row_class


@pytest.mark.data
@pytest.mark.slow
@_needs_mirror
def test_the_released_table_ledger_matches_issue_598() -> None:
    """The 943 built isolates of Datafile 1: 459,790 blank rows, 443,200 absent, 16,031
    bad annotation, 0 unannotated and 559 present; the 559 present rows are served, 98
    under 16 S288C names (GAL1 YBR020W in 31 isolates, GAL2 YLR081W in 25) and 461 under
    12 pangenome ids, and the 16,031 bad-annotation rows under their 17 pangenome ids;
    GAL1 is then a key in all 943 records (912 named + 31 served).
    """
    from torchcell.sequence.genome.registry import PETER2018_1011, resolve

    presence = pd.read_csv(
        resolve(PETER2018_1011, ni.PRESENCE_NAME), sep="\t", index_col=0, usecols=[0]
    )
    built = m.restrict_to_built_isolates(
        m.read_caudal_table(_MIRROR_ZIP), set(presence.index.astype(str))
    )
    kept, ledger = m.resolve_gene_ids(built)
    assert (ledger.n_rows, ledger.n_rows_blank) == (6_145_531, 459_790)
    assert ledger.counts == {
        m.BlankRowClass.present_s288c_homolog: 98,
        m.BlankRowClass.present_pangenome_orf: 461,
        m.BlankRowClass.absent: 443_200,
        m.BlankRowClass.bad_annotation: 16_031,
        m.BlankRowClass.unannotated: 0,
    }
    served = {g: n for g, n in ledger.served_ids.items() if ni.S288C_RE.match(g)}
    assert (len(served), sum(served.values())) == (16, 98)
    assert (served["YBR020W"], served["YLR081W"]) == (31, 25)
    pangenome = {g: n for g, n in ledger.served_ids.items() if g not in served}
    assert (len(pangenome), sum(pangenome.values())) == (12 + 17, 461 + 16_031)
    assert pangenome["X39-augustus_masked.2.CGIPLA_MA"] == 190
    assert (
        pangenome["X142-augustus_masked.1731.ATP_4"] == 943
    )  # one row per built isolate
    gal1 = kept[kept["gene_id"] == "YBR020W"]
    assert (gal1["Strain"].nunique(), int(gal1["systematic_name"].isna().sum())) == (
        943,
        31,
    )


def test_a_table_without_tpm_is_refused_by_the_column_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = _zip(
        {
            "final.tab": "Strain,systematic_name,ORF,Ortholog_in_SGD_2010,pan_absence,"
            "count\nAAA,YAL001C,YAL001C,,present,10\n"
        }
    )
    with pytest.raises(
        ValueError,
        match=r"Usecols do not match columns, columns expected but not found: "
        r"\['tpm'\]",
    ):
        m.CaudalPanTranscriptome2024Dataset(
            root=str(_root(tmp_path, monkeypatch, table))
        )


@pytest.mark.parametrize(
    ("members", "tabs"),
    [(["final.csv"], "[]"), (["a.tab", "b.tab"], "['a.tab', 'b.tab']")],
)
def test_a_zip_without_exactly_one_tab_member_is_refused_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, members: list[str], tabs: str
) -> None:
    """No ``.tab`` member, or two, raises ``MissingTabMemberError`` naming the archive,
    the ``.tab`` members found and every member, instead of a bare ``StopIteration``
    or a silent first pick.
    """
    table = _zip({name: "Strain,systematic_name,count,tpm\n" for name in members})
    root = _root(tmp_path, monkeypatch, table)
    with pytest.raises(m.MissingTabMemberError) as excinfo:
        m.CaudalPanTranscriptome2024Dataset(root=str(root))
    assert str(excinfo.value) == (
        f"{root / 'raw' / m.CAUDAL_ZIP_BASENAME} must hold exactly one '.tab' member, "
        f"found {tabs}; members: {members}"
    )
    assert not (root / "processed" / "lmdb").exists()


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
    assert ni.sgd_chromosomes(str(path)) == {"I": "ACGT", "XVI": "GGCC"}


def test_a_regular_member_that_cannot_be_extracted_is_refused(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``extracted is None`` guard is unreachable for a real archive (all 6,015
    members of the pinned tarball extract), since ``tarfile`` returns a file object for
    every member that passes ``isfile()``. Forced here by returning None for
    ``YAL001C.fasta``: the build raises ``UnextractableMemberError`` naming the tarball
    and the member, instead of dropping that gene's variants with no log.
    """
    original = tarfile.TarFile.extractfile

    def extractfile(self: tarfile.TarFile, member: Any) -> IO[bytes] | None:
        if getattr(member, "name", member) == "YAL001C.fasta":
            return None
        return original(self, member)

    monkeypatch.setattr(tarfile.TarFile, "extractfile", extractfile)
    with pytest.raises(ni.UnextractableMemberError) as excinfo:
        m.CaudalPanTranscriptome2024Dataset(root=str(root))
    assert str(excinfo.value) == (
        f"{root / 'raw' / ni.REFGENE_TAR_NAME}: member 'YAL001C.fasta' is a regular "
        "file but tarfile returned no file object for it; refusing to drop its variants"
    )
    assert not (root / "preprocess" / "sequence_variants.parquet").exists()
    assert not (root / "processed" / "lmdb").exists()


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
        "REPLICATE_TAB_SHA256",
        hashlib.sha256(raw_bytes[m.REPLICATE_TAB_BASENAME]).hexdigest(),
    )
    raw = tmp_path / "caudal" / "raw"
    raw.mkdir(parents=True)
    stale = {**_PRESENCE, "AAA": ["1", "1", "1", "1", "0", "1"]}
    stale_bytes = _matrix_gz(stale, _COLUMNS)
    (raw / ni.PRESENCE_NAME).write_bytes(stale_bytes)
    with pytest.raises(RawSha256MismatchError) as err:
        m.CaudalPanTranscriptome2024Dataset(root=str(tmp_path / "caudal"))
    assert str(err.value) == (
        f"sha256 mismatch for {raw / ni.PRESENCE_NAME}: expected "
        f"{hashlib.sha256(raw_bytes[ni.PRESENCE_NAME]).hexdigest()}, "
        f"observed {hashlib.sha256(stale_bytes).hexdigest()}"
    )
    assert [os.path.islink(raw / n) for n in m.RAW_FILES] == [
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
    ]
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
        species="Saccharomyces cerevisiae",
        strain="S288C",
        assembly_set=SGD_S288C_R64,
        assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
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


# ---- 2026.10.06 (Phase 21): the remaining refusals of the ledger models, _sha256 -- #
def test_a_blank_row_rule_gap_must_name_definition_and_served_needs_an_id_rule() -> (
    None
):
    """A gap whose ``field`` is not ``definition`` is refused; a ``served`` class
    without an ``id_rule``, and a ``dropped`` class with one, are refused with the same
    message (the check is an equivalence, lines 427-428).
    """
    bad = m.BLANK_ROW_RULES[m.BlankRowClass.bad_annotation]
    assert bad.gap is not None
    with pytest.raises(
        ValueError,
        match=re.escape("bad_annotation: gap must name the 'definition' field"),
    ):
        m.BlankRowRule(
            **{**bad.model_dump(), "gap": bad.gap.model_copy(update={"field": "tpm"})}
        )
    served = m.BLANK_ROW_RULES[m.BlankRowClass.present_pangenome_orf]
    with pytest.raises(
        ValueError,
        match=re.escape("present_pangenome_orf: a served class needs an id_rule"),
    ):
        m.BlankRowRule(**{**served.model_dump(), "id_rule": None})
    absent = m.BLANK_ROW_RULES[m.BlankRowClass.absent]
    with pytest.raises(
        ValueError, match=re.escape("absent: a served class needs an id_rule")
    ):
        m.BlankRowRule(**{**absent.model_dump(), "id_rule": "serve it"})


def _ledger_fields() -> dict[str, Any]:
    """Ten rows: seven named, three blank (one per served class plus one absent)."""
    classes = m.BlankRowClass
    counts = dict.fromkeys(classes, 0)
    counts[classes.present_s288c_homolog] = 1
    counts[classes.present_pangenome_orf] = 1
    counts[classes.absent] = 1
    return {
        "rules": list(m.BLANK_ROW_RULES.values()),
        "n_rows": 10,
        "n_rows_named": 7,
        "n_rows_blank": 3,
        "counts": counts,
        "tpm_by_class": dict.fromkeys(classes, 0.0),
        "served_ids": {"YBR020W": 1, "X5-contig_7": 1},
        "named_pan_absence_counts": {"present": 7},
    }


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"n_rows": 11}, "named + blank rows must equal all rows"),
        (
            {"n_rows_blank": 4, "n_rows": 11},
            "every blank row must be in exactly one class",
        ),
        (
            {"served_ids": {"YBR020W": 1}},
            "served_ids must account for every served row",
        ),
    ],
    ids=["rows", "classes", "served"],
)
def test_the_blank_name_ledger_refuses_rows_it_cannot_account_for(
    change: dict[str, Any], message: str
) -> None:
    """The consistent ledger validates; 7 + 3 != 11 rows, 3 classified rows against 4
    blank ones, and 1 served id row against the 2 served-class rows (1 s288c homolog +
    1 pangenome ORF) are each refused with their own message.
    """
    assert m.BlankNameLedger(**_ledger_fields()).n_rows == 10
    with pytest.raises(ValueError, match=re.escape(message)):
        m.BlankNameLedger(**{**_ledger_fields(), **change})


# --------------------------------------------------------------------------- #
# 2026.10.07: sequence_ref (ArtifactRef) replaces sequence_uri + sequence_sha256.
# --------------------------------------------------------------------------- #
def test_refgene_tarball_ref_reads_sha256_and_bytes_from_the_tier_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file-level ref names the genomes tier, the Peter set and the tarball, with
    the sha256 and byte count the synthetic tier's manifest pins and no member.
    """
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    raw_bytes = _write_mirror(data_root)
    tar = raw_bytes[ni.REFGENE_TAR_NAME]
    ref = ni.refgene_tarball_ref(str(data_root))
    assert ref == ArtifactRef(
        tier="genomes",
        key="peter2018_1011_assemblies",
        path="allReferenceGenesWithSNPsAndIndelsInferred.tar.gz",
        sha256=hashlib.sha256(tar).hexdigest(),
        bytes=len(tar),
    )
    assert ni.refgene_tarball_ref() == ref  # DATA_ROOT from the environment


def test_a_caudal_sequence_variant_serializes_its_ref_and_round_trips() -> None:
    """``natural_isolate.sequence_perturbations`` narrows the tarball ref to
    ``<gene>.fasta#<token>``;
    the dump carries the ref as a nested object, and its tc:// string parses back to the
    same ref given the sha256.
    """
    tarball = ArtifactRef(
        tier="genomes",
        key="peter2018_1011_assemblies",
        path=ni.REFGENE_TAR_NAME,
        sha256="b" * 64,
        bytes=160_823_369,
    )
    (pert,) = ni.sequence_perturbations(
        "AAA", [("YAL001C", "TFC3", "AAA_YAL001C_TFC3")], tarball
    )
    ref = pert.sequence_ref
    assert ref is not None
    assert str(ref) == (
        "tc://genomes/peter2018_1011_assemblies/"
        "allReferenceGenesWithSNPsAndIndelsInferred.tar.gz#YAL001C.fasta#AAA_YAL001C_TFC3"
    )
    assert pert.model_dump()["sequence_ref"] == {
        "tier": "genomes",
        "key": "peter2018_1011_assemblies",
        "path": "allReferenceGenesWithSNPsAndIndelsInferred.tar.gz",
        "member": "YAL001C.fasta#AAA_YAL001C_TFC3",
        "sha256": "b" * 64,
        "bytes": 160_823_369,
        "media_type": None,
    }
    assert ArtifactRef.parse(str(ref), sha256=ref.sha256, bytes=ref.bytes) == ref
    assert type(pert).model_validate(pert.model_dump()) == pert


_PETER_MANIFEST = osp.join(
    os.environ.get("DATA_ROOT", ""),
    "torchcell-genomes",
    "peter2018_1011_assemblies",
    "manifest.json",
)


@pytest.mark.data
@pytest.mark.skipif(
    not osp.isfile(_PETER_MANIFEST), reason="requires the Peter genomes tier"
)
def test_the_tier_manifest_lists_the_reference_gene_tarball() -> None:
    """The lookup finds the tarball in the real tier at the digest the served Caudal
    records carried as ``sequence_sha256`` before 2026.10.07.
    """
    ref = ni.refgene_tarball_ref()
    assert (ref.path, ref.sha256, ref.bytes) == (
        "allReferenceGenesWithSNPsAndIndelsInferred.tar.gz",
        "b5400b89499fe84b1feada51abd7742c29838ae1f28c0cbd208b6622ca533f25",
        160_823_369,
    )


# ---- 2026.11.24: one record per culture, Datafile 2 (issue #1178) ----------------- #
# Datafile 1 with release-shaped ORFs: AAA's blank ``absent`` X9.contig_1 is dropped, and
# SACE_YAU serves a blank YBR001W homolog row, a present pangenome ORF and a bad-annotation
# ORF under their Datafile 1 ids.
_REP_D1 = (
    "Strain,systematic_name,ORF,Ortholog_in_SGD_2010,pan_absence,count,tpm\n"
    "AAA,YAL001C,YAL001C,,present,10,1.5\n"
    "AAA,YBR001W,YBR001W,,present,20,4.0\n"
    "AAA,,X9.contig_1,,absent,7,0.7\n"
    "SACE_YAU,YAL001C,YAL001C,,present,30,3.0\n"
    "SACE_YAU,,YBR001W,YBR001W,present,8,2.0\n"
    "SACE_YAU,,X5.contig_7,,present,4,1.0\n"
    "SACE_YAU,,X7.contig_9,,bad annotation,3,0.3\n"
)
_REP_HEADER = (
    "ORF,SampleID,tpm,Sample.name.before.reassignement,Standardized.name,Final.Group,"
    "Number.of.clean.mapped.reads\n"
)
# Datafile 2 in a shuffled sample order: AAA Final_set + good_duplicate, SACE_YAU
# Final_set + good_duplicate + good_duplicateBis, and the excluded XTRA_ABC. ORFs are in
# the release's own spelling (``5-contig_7``), which ``r_make_names`` maps to Datafile 1's.
_REP_D2 = _REP_HEADER + (
    'YAL001C,Run1.AAA,1.4,AAA_1,AAA,Final_set,"1,000,000"\n'
    'YBR001W,Run1.AAA,4.1,AAA_1,AAA,Final_set,"1,000,000"\n'
    'YAL001C,Run3.SYB,2.8,SYB_3,SACE_YAU,good_duplicateBis,"3,000,000"\n'
    'YBR001W,Run3.SYB,2.2,SYB_3,SACE_YAU,good_duplicateBis,"3,000,000"\n'
    '5-contig_7,Run3.SYB,1.1,SYB_3,SACE_YAU,good_duplicateBis,"3,000,000"\n'
    '7-contig_9,Run3.SYB,0.2,SYB_3,SACE_YAU,good_duplicateBis,"3,000,000"\n'
    'YAL001C,Run2.AAB,1.6,AAB_2,AAA,good_duplicate,"2,500,000"\n'
    'YBR001W,Run2.AAB,3.9,AAB_2,AAA,good_duplicate,"2,500,000"\n'
    '9-contig_1,Run2.AAB,0.9,AAB_2,AAA,good_duplicate,"2,500,000"\n'
    'YAL001C,Run2.SYA,0.00832811287968815,SYA_2,SACE_YAU,good_duplicate,"1,500,000"\n'
    'YAL001C,Run1.SACE_YAU,2.9,SACE_YAU_1,SACE_YAU,Final_set,"4,000,000"\n'
    'YAL001C,Run1.ABC,9.0,ABC_1,XTRA_ABC,Final_set,"1,000,000"\n'
)


def _replicate_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, d1: str, d2: str
) -> Path:
    """The synthetic raw set with Datafile 1 = ``d1`` and Datafile 2 = ``d2``."""
    root = _root(tmp_path, monkeypatch, _zip({"final.tab": d1}))
    (root / "raw" / m.REPLICATE_TAB_BASENAME).write_text(d2)
    return root


def _phenotype(
    tpm: dict[str, float],
    count: dict[str, int] | None,
    replicate_id: str,
    n_mapped_reads: int | None,
) -> dict[str, Any]:
    return RNASeqExpressionPhenotype(
        expression_tpm=tpm,
        expression_count=count,
        measurement_type="rnaseq_tpm",
        n_mapped_reads=n_mapped_reads,
        replicate_id=replicate_id,
        provenance_gaps=[] if count is not None else [m.REPLICATE_COUNT_GAP],
    ).model_dump()


def test_each_other_culture_of_a_built_isolate_is_its_own_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Five records: per isolate its Datafile 1 culture (counts kept, ``replicate_id`` the
    ``Final_set`` SampleID, no mapped-read count), then its other Datafile 2 cultures in
    ``Final.Group`` order (good_duplicate before good_duplicateBis): TPM verbatim under
    the Datafile 1 gene id of the same ORF, the ``absent`` ORF 9-contig_1 left out, no
    counts (the typed gap), the released mapped reads, ``replicate_id`` the SampleID.
    ``0.00832811287968815`` (a released Run4.7.CCH value) is read to the exact float.
    The cultures share their isolate's genotype; the reference is the Datafile 1 mean,
    untouched by them. XTRA_ABC is not built and is counted in the ledger.
    """
    root = _replicate_root(tmp_path, monkeypatch, _REP_D1, _REP_D2)
    dataset = m.CaudalPanTranscriptome2024Dataset(root=str(root))
    assert len(dataset) == 5
    sace = {"YAL001C": 3.0, "YBR001W": 2.0, "X5-contig_7": 1.0, "X7-contig_9": 0.3}
    assert [dataset[i]["experiment"]["phenotype"] for i in range(5)] == [
        _phenotype(
            {"YAL001C": 1.5, "YBR001W": 4.0},
            {"YAL001C": 10, "YBR001W": 20},
            "Run1.AAA",
            None,
        ),
        _phenotype({"YAL001C": 1.6, "YBR001W": 3.9}, None, "Run2.AAB", 2_500_000),
        _phenotype(
            sace,
            {"YAL001C": 30, "YBR001W": 8, "X5-contig_7": 4, "X7-contig_9": 3},
            "Run1.SACE_YAU",
            None,
        ),
        _phenotype(
            {"YAL001C": float("0.00832811287968815")}, None, "Run2.SYA", 1_500_000
        ),
        _phenotype(
            {"YAL001C": 2.8, "YBR001W": 2.2, "X5-contig_7": 1.1, "X7-contig_9": 0.2},
            None,
            "Run3.SYB",
            3_000_000,
        ),
    ]
    genotypes = [e.model_dump()["genotype"] for e in _EXPECTED]
    assert [dataset[i]["experiment"]["genotype"] for i in range(5)] == [
        genotypes[0],
        genotypes[0],
        genotypes[1],
        genotypes[1],
        genotypes[1],
    ]
    references = {json.dumps(dataset[i]["reference"], sort_keys=True) for i in range(5)}
    assert len(references) == 1
    assert dataset[0]["reference"]["phenotype_reference"]["expression_tpm"] == {
        "YAL001C": 2.25,
        "YBR001W": 3.0,
        "X5-contig_7": 1.0,
        "X7-contig_9": 0.3,
    }
    ledger = m.ReplicateLedger.model_validate_json(
        (root / "preprocess" / "replicate_culture_ledger.json").read_text()
    )
    assert ledger.model_dump(exclude={"samples"}) == {
        "n_rows": 12,
        "n_samples": 6,
        "n_strains": 3,
        "samples_by_group": {
            m.ReplicateGroup.final_set: 3,
            m.ReplicateGroup.good_duplicate: 2,
            m.ReplicateGroup.good_duplicate_bis: 1,
        },
        "strains_not_built": ["XTRA_ABC"],
        "final_set_sample_ids": {"AAA": "Run1.AAA", "SACE_YAU": "Run1.SACE_YAU"},
        "n_culture_records": 3,
        "n_rows_culture_records": 7,
        "n_rows_absent_dropped": 1,
    }
    assert [(s["sample_id"], s["fate"]) for s in ledger.samples] == [
        ("Run1.AAA", "datafile_1_record"),
        ("Run1.ABC", "isolate_not_built"),
        ("Run1.SACE_YAU", "datafile_1_record"),
        ("Run2.AAB", "culture_record"),
        ("Run2.SYA", "culture_record"),
        ("Run3.SYB", "culture_record"),
    ]


def test_verify_build_passes_a_store_with_replicate_cultures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cultures of an isolate share one genotype, so the isolate rules see it once
    (one genotype per isolate, two isolates), and ``replicate_id`` keeps the five
    records apart under the (strain, condition) rule.
    """
    from torchcell.verification import runners

    root = _replicate_root(tmp_path, monkeypatch, _REP_D1, _REP_D2)
    dataset = m.CaudalPanTranscriptome2024Dataset(root=str(root))
    data_root = Path(os.environ["DATA_ROOT"])
    tables = data_root / "torchcell-library" / ni.CITATION_KEY / ni.LIBRARY_TABLES_REL
    tables.parent.mkdir(parents=True)
    tables.write_bytes(_WORKBOOK)
    monkeypatch.setattr(ni, "TABLES_SHA256", hashlib.sha256(_WORKBOOK).hexdigest())
    monkeypatch.setattr(
        runners,
        "_sgd_gene_set",
        lambda root: {"YAL001C", "YBR001W", "Q0010", "X5-contig_7", "X7-contig_9"},
    )
    monkeypatch.setitem(
        runners.RNASEQ_DATASETS["caudal_pantranscriptome2024"], "expected_count", 5
    )
    dataset.close_lmdb()
    report = m.verify_build(str(root), data_root=str(data_root))
    results = {r.name: r for r in report.results}
    assert [name for name, r in results.items() if not r.passed] == []
    assert results["strain_uniqueness"].details["n_records"] == 5
    assert results["isolate_genotype_identity"].details["n_isolates"] == 2
    assert results["isolate_genotype_round_trip"].message == (
        "2 stored genotypes equal their re-derivation from the pinned Peter 2018 "
        "releases"
    )


def _replicate_frames(
    tmp_path: Path, d2: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    built = pd.read_csv(io.StringIO(_REP_D1))
    built["Strain"] = built["Strain"].astype(str)
    kept, _ = m.resolve_gene_ids(built)
    path = tmp_path / "replicates.tab"
    path.write_text(d2)
    return m.read_replicate_table(str(path)), built, kept


def test_r_make_names_gives_the_datafile_1_spelling() -> None:
    """Dashes and parentheses become dots, and a leading digit or a dot before a digit
    takes an ``X``; S288C names are kept.
    """
    assert [
        m.r_make_names(o)
        for o in [
            "1003-augustus_masked-YCM-7680",
            "46-PUT_LAC_TRASP(R)-14850-",
            "YAL068C",
            "Q0010",
            ".5x",
        ]
    ] == [
        "X1003.augustus_masked.YCM.7680",
        "X46.PUT_LAC_TRASP.R..14850.",
        "YAL068C",
        "Q0010",
        "X.5x",
    ]


def test_a_final_group_no_rule_covers_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "replicates.tab"
    path.write_text(_REP_HEADER + 'YAL001C,R1,1.0,A_1,AAA,bad_group,"1,000"\n')
    with pytest.raises(
        m.ReplicateTableError, match=re.escape("no rule covers: ['bad_group']")
    ):
        m.read_replicate_table(str(path))


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        (
            'YAL001C,R2,1.0,A_2,AAA,good_duplicate,"1,000"\n',
            "built isolates without exactly one Final_set sample: ['AAA']",
        ),
        (
            'YAL001C,R1,1.0,A_1,AAA,Final_set,"1,000"\n'
            'YAL001C,R1b,1.0,A_1,AAA,Final_set,"1,000"\n',
            "built isolates without exactly one Final_set sample: ['AAA']",
        ),
        (
            'YAL001C,R1,1.0,A_1,AAA,Final_set,"1,000"\n'
            'YBR001W,R1,1.0,A_1,SACE_YAU,Final_set,"1,000"\n',
            "more than one isolate, group or mapped-read count: ['R1']",
        ),
        (
            'YAL001C,R1,1.0,A_1,AAA,Final_set,"1,000"\n'
            'YCR001W,R2,1.0,A_2,AAA,good_duplicate,"1,000"\n',
            "1 Datafile 2 rows name an ORF the isolate has no Datafile 1 row for: "
            "[('R2', 'YCR001W')]",
        ),
        (
            'YAL001C,R1,1.0,A_1,AAA,Final_set,"1,000"\n'
            'YAL001C,R1,2.0,A_1,AAA,Final_set,"1,000"\n',
            "repeats a (SampleID, ORF) row",
        ),
    ],
)
def test_a_replicate_table_of_another_shape_is_refused(
    tmp_path: Path, rows: str, match: str
) -> None:
    """No Final_set or two for a built isolate, a sample on two isolates, an ORF the
    isolate has no Datafile 1 row for, and a repeated row are each refused.
    """
    replicates, built, kept = _replicate_frames(tmp_path, _REP_HEADER + rows)
    with pytest.raises(m.ReplicateTableError, match=re.escape(match)):
        m.replicate_cultures(replicates, built, kept)


_MIRROR_REPLICATES = osp.join(os.environ.get("DATA_ROOT", ""), m.REPLICATE_TAB_REL)


@pytest.mark.data
@_needs_mirror
def test_the_replicate_sourced_values_audit_against_the_mirror() -> None:
    """Each replicate-design quote is in ``paper.md`` / ``methods.md`` at its sha256."""
    for name, sv in m.REPLICATE_SOURCED_VALUES.items():
        assert audit_sourced_value(sv, _LIBRARY).passed, name


@pytest.mark.data
@pytest.mark.slow
@_needs_mirror
def test_the_released_replicate_table_gives_28_culture_records() -> None:
    """Datafile 2: 387,360 rows, 60 samples of 29 isolates (29 Final_set, 29
    good_duplicate, 2 good_duplicateBis). 26 isolates are built; XTRA_DGR, XTRA_DGT and
    XTRA_DGU have no Peter genome. The 28 other cultures of the 26 become records with
    169,036 keyed rows; 11,732 rows sit on blank-name ``absent`` ORFs and are left out.
    Every built isolate's Final_set sample has the clean mapped reads Caudal's
    Supplementary Table 1 prints for that isolate (the Datafile 1 culture).
    """
    from torchcell.sequence.genome.registry import PETER2018_1011, resolve

    presence = pd.read_csv(
        resolve(PETER2018_1011, ni.PRESENCE_NAME), sep="\t", index_col=0, usecols=[0]
    )
    built = m.restrict_to_built_isolates(
        m.read_caudal_table(_MIRROR_ZIP), set(presence.index.astype(str))
    )
    kept, _ = m.resolve_gene_ids(built)
    replicates = m.read_replicate_table(_MIRROR_REPLICATES)
    cultures, ledger = m.replicate_cultures(replicates, built, kept)
    assert (ledger.n_rows, ledger.n_samples, ledger.n_strains) == (387_360, 60, 29)
    assert ledger.samples_by_group == {
        m.ReplicateGroup.final_set: 29,
        m.ReplicateGroup.good_duplicate: 29,
        m.ReplicateGroup.good_duplicate_bis: 2,
    }
    assert ledger.strains_not_built == ["XTRA_DGR", "XTRA_DGT", "XTRA_DGU"]
    assert len(ledger.final_set_sample_ids) == 26
    assert (
        len(cultures),
        ledger.n_rows_culture_records,
        ledger.n_rows_absent_dropped,
    ) == (28, 169_036, 11_732)
    per_isolate = pd.Series([c.strain for c in cultures]).value_counts()
    assert sorted(per_isolate[per_isolate == 2].index) == ["AFD", "AKL"]
    table_s1 = pd.read_excel(
        osp.join(_LIBRARY, m.CITATION_KEY, m.SI_TABLES_XLSX), sheet_name="Table S1"
    ).set_index("Standardized_name")
    reads = {s["sample_id"]: s["n_mapped_reads"] for s in ledger.samples}
    assert {
        isolate: reads[sample]
        for isolate, sample in ledger.final_set_sample_ids.items()
    } == {
        isolate: int(str(table_s1.loc[isolate, "Mapped_clean_reads"]).replace(",", ""))
        for isolate in ledger.final_set_sample_ids
    }
