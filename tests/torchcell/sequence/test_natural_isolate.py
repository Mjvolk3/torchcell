# tests/torchcell/sequence/test_natural_isolate.py
# [[tests.torchcell.sequence.test_natural_isolate]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/sequence/test_natural_isolate.py
"""The shared natural-isolate genotype (#987, #988) on a synthetic five-isolate panel.

SGD FASTA: chromosome I ``AAAACCCCGGGGTTTT``, II ``ACCCTTTGGA``, IV ``GATCGATC``.
Reference-gene tarball: ``YAL001C.fasta`` on chromosome1:1-8 + (slice ``AAAACCCC``; AAA
``AAAACCCT`` and SACE_YAU ``AAAACCCA`` differ) and ``YDL227C.fasta`` (HO) on
chromosome4:1-4 + (slice ``GATC``; CCC ``GATT`` differs). YBR001W has no gene file.

Presence columns X1.YAL001C, X2.EC1118_1F14_0012g, X3.YDL227C, X4.YBR001W:
AAA 1 1 1 1, BBB 1 0 0 1 (HO absent), CCC 1 0 1 0 (YBR001W absent), SACE_YAU 1 1 1 1,
DDD 1 0 1 1. Copy number of X2: AAA 2.5, SACE_YAU NA (so 1.0).

Table S1: AAA no / 2 / euploid / homozygous; BBB yes / 1 / euploid / homozygous; CCC
yes / 1 / euploid / heterozygous; SACE_YAU no / Xn / aneu;+1*9;-1*3; / homozygous; DDD
no / 3 / (empty) / heterozygous.

Loss-of-function matrix (code, NAME, Gene -> calls): 1 YAL001C TFC3 -> AAA, BBB;
4 YBR001W NTH2 -> AAA, CCC; 3 YDL227C HO -> CCC.

Frameshift matrix: coded 1 TFC3 -> AAA; coded 3 HO -> none; and seven rows with no code,
each stored under its ``Gene`` exactly as released (no name lookup admits, drops or
re-keys one): YAL002W -> DDD; PAU8 -> SACE_YAU; SLM6 -> BBB; AAD16 -> AAA; DDI3 -> CCC;
LTO1 -> DDD; MOVED -> AAA.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import os.path as osp
import tarfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tests.torchcell.sequence._peter_tier import (
    FrameshiftRow,
    LofRow,
    TableS1Row,
    frameshift_matrix,
    genotype_files,
    lof_matrix,
    tables_workbook,
    write_tier,
)
from torchcell.data.experiment_dataset import RawSha256MismatchError
from torchcell.datamodels.schema import (
    ArtifactRef,
    ChromosomeCopyChange,
    DeletionPerturbation,
    GenomeZygosity,
    GenomicSpan,
    Genotype,
    IsolateManipulation,
    NaturalGeneAbsencePerturbation,
    NaturalGenePresencePerturbation,
    NaturalLossOfFunctionCause,
    NaturalLossOfFunctionPerturbation,
    SequenceVariantPerturbation,
)
from torchcell.literature.manifest import Manifest
from torchcell.sequence import natural_isolate as ni
from torchcell.sequence.genome.registry import (
    PETER2018_1011,
    SGD_S288C_R64,
    load_genome_manifest,
)
from torchcell.verification.sourced import ProvenanceGapReason, audit_sourced_value

ISOLATES = ["AAA", "BBB", "CCC", "SACE_YAU", "DDD"]
_SGD_FSA = (
    ">ref|NC_001133| [org=Saccharomyces cerevisiae] [chromosome=I]\nAAAACCCC\n"
    "GGGGTTTT\n>ref|NC_001134| [org=Saccharomyces cerevisiae] [chromosome=II]\n"
    "ACCCTTTGGA\n>ref|NC_001136| [org=Saccharomyces cerevisiae] [chromosome=IV]\n"
    "GATCGATC\n"
)
_GENE_FILES = [
    (
        "YAL001C.fasta",
        ">AAA_YAL001C_TFC3 chromosome1:1-8 +\nAAAACCCT\n"
        ">BBB_YAL001C_TFC3 chromosome1:1-8 +\nAAAACCCC\n"
        ">CCC_YAL001C_TFC3 chromosome1:1-8 +\nAAAACCCC\n"
        ">SACE_YAU_YAL001C_TFC3 chromosome1:1-8 +\nAAAACCCA\n"
        ">DDD_YAL001C_TFC3 chromosome1:1-8 +\naaaacccc\n",
    ),
    (
        "YDL227C.fasta",
        ">AAA_YDL227C_HO chromosome4:1-4 +\nGATC\n"
        ">BBB_YDL227C_HO chromosome4:1-4 +\nGATC\n"
        ">CCC_YDL227C_HO chromosome4:1-4 +\nGATT\n"
        ">SACE_YAU_YDL227C_HO chromosome4:1-4 +\nGATC\n"
        ">DDD_YDL227C_HO chromosome4:1-4 +\nGATC\n",
    ),
]
_COLUMNS = ["X1.YAL001C", "X2.EC1118_1F14_0012g", "X3.YDL227C", "X4.YBR001W"]
_PRESENCE = {
    "AAA": ["1", "1", "1", "1"],
    "BBB": ["1", "0", "0", "1"],
    "CCC": ["1", "0", "1", "0"],
    "SACE_YAU": ["1", "1", "1", "1"],
    "DDD": ["1", "0", "1", "1"],
}
_COPYNUMBER = {
    "AAA": ["1", "2.5", "1", "1"],
    "BBB": ["1", "0", "0", "1"],
    "CCC": ["1", "0", "1", "0"],
    "SACE_YAU": ["1", "NA", "1", "1"],
    "DDD": ["1", "0", "1", "1"],
}
_LOF = [
    LofRow(code=1, name="YAL001C", gene="TFC3", ch=1, start=1, stop=8, calls=frozenset({"AAA", "BBB"})),
    LofRow(code=4, name="YBR001W", gene="NTH2", ch=2, start=2, stop=9, calls=frozenset({"AAA", "CCC"})),
    LofRow(code=3, name="YDL227C", gene="HO", ch=4, start=1, stop=4, calls=frozenset({"CCC"})),
]  # fmt: skip
_FRAMESHIFT = [
    FrameshiftRow(code=1, gene="TFC3", ch=1, start=1, stop=8, calls=frozenset({"AAA"})),
    FrameshiftRow(code=3, gene="HO", ch=4, start=1, stop=4),
    FrameshiftRow(code=None, gene="YAL002W", ch=1, start=10, stop=12, calls=frozenset({"DDD"})),
    FrameshiftRow(code=None, gene="PAU8", ch=1, start=13, stop=16, calls=frozenset({"SACE_YAU"})),
    FrameshiftRow(code=None, gene="SLM6", ch=2, start=739936, stop=740388, calls=frozenset({"BBB"})),
    FrameshiftRow(code=None, gene="AAD16", ch=6, start=14305, stop=14763, calls=frozenset({"AAA"})),
    FrameshiftRow(code=None, gene="DDI3", ch=14, start=11452, stop=12129, calls=frozenset({"CCC"})),
    FrameshiftRow(code=None, gene="LTO1", ch=14, start=156859, stop=157455, calls=frozenset({"DDD"})),
    FrameshiftRow(code=None, gene="MOVED", ch=1, start=20, stop=30, calls=frozenset({"AAA"})),
]  # fmt: skip
_TABLE_S1 = [
    TableS1Row(isolate="AAA"),
    TableS1Row(isolate="BBB", ho_deletion="yes", ploidy=1),
    TableS1Row(isolate="CCC", ho_deletion="yes", ploidy=1, zygosity="heterozygous"),
    TableS1Row(isolate="SACE_YAU", ploidy="Xn", aneuploidies="aneu;+1*9;-1*3;"),
    TableS1Row(isolate="DDD", ploidy=3, aneuploidies=None, zygosity="heterozygous"),
]


def _matrix_gz(rows: dict[str, list[str]]) -> bytes:
    lines = ["\t".join(["strain", *_COLUMNS])]
    lines += ["\t".join([name, *values]) for name, values in rows.items()]
    return gzip.compress(("\n".join(lines) + "\n").encode(), mtime=0)


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


def _peter_files() -> dict[str, bytes]:
    return {
        ni.REFGENE_TAR_NAME: _tarball(),
        ni.PRESENCE_NAME: _matrix_gz(_PRESENCE),
        ni.COPYNUMBER_NAME: _matrix_gz(_COPYNUMBER),
        **genotype_files(
            ISOLATES, lof=_LOF, frameshift=_FRAMESHIFT, table_s1=_TABLE_S1
        ),
    }


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(root))
    write_tier(root, SGD_S288C_R64, {ni.SGD_FSA_NAME: _SGD_FSA.encode()})
    write_tier(root, PETER2018_1011, _peter_files())
    return root


@pytest.fixture
def genotypes(data_root: Path, tmp_path: Path) -> ni.NaturalIsolateGenotypes:
    raw = tmp_path / "raw"
    raw.mkdir()
    ni.link_genotype_files(str(raw), str(data_root))
    cache = tmp_path / "cache"
    cache.mkdir()
    return ni.NaturalIsolateGenotypes(raw, cache, ISOLATES, str(data_root))


def _ref(data_root: Path, name: str, member: str | None = None) -> ArtifactRef:
    record = load_genome_manifest(PETER2018_1011, str(data_root)).record(name)
    return ArtifactRef(
        tier="genomes",
        key=PETER2018_1011,
        path=name,
        member=member,
        sha256=record.sha256,
        bytes=record.bytes,
    )


def _span(chrom: str, start: int, end: int) -> GenomicSpan:
    return GenomicSpan(
        chromosome=chrom, start=start, end=end, assembly="sgd_S288C_R64-4-1_20230830"
    )


# --------------------------------------------------------------------------- #
# One isolate end to end
# --------------------------------------------------------------------------- #
def _call(
    isolate: str,
    key: str,
    ref: ArtifactRef,
    *,
    code: int | None,
    name: str | None,
    gene: str,
    span: GenomicSpan,
) -> NaturalLossOfFunctionPerturbation:
    return NaturalLossOfFunctionPerturbation(
        systematic_gene_name=key,
        gene_namespace="scerevisiae_s288c_sgd_systematic",
        perturbed_gene_name=key,
        strain_id=isolate,
        cause=NaturalLossOfFunctionCause.nonsense_or_sift
        if name is not None
        else NaturalLossOfFunctionCause.homozygous_frameshift,
        call_ref=ref,
        release_code=code,
        release_name=name,
        release_gene_name=gene,
        release_span=span,
    )


def test_ccc_is_its_variants_absence_ho_deletion_calls_and_descriptor(
    genotypes: ni.NaturalIsolateGenotypes, data_root: Path
) -> None:
    """CCC: an HO allele that differs (a variant), YBR001W absent, HO deleted per
    Table S1, and three released calls stored as released: YBR001W and HO from the
    loss-of-function matrix (by NAME), and the code-less DDI3 frameshift row under its
    Gene, although the R64-4-1 annotation now gives that name to another ORF.
    """
    tar = _ref(data_root, ni.REFGENE_TAR_NAME)
    lof = _ref(data_root, ni.LOF_NAME)
    fs = _ref(data_root, ni.FRAMESHIFT_NAME)
    expected = Genotype(
        perturbations=[
            SequenceVariantPerturbation(
                systematic_gene_name="YDL227C",
                perturbed_gene_name="HO",
                strain_id="CCC",
                sequence_source=ni.SEQUENCE_SOURCE,
                sequence_ref=tar.model_copy(
                    update={"member": "YDL227C.fasta#CCC_YDL227C_HO"}
                ),
            ),
            NaturalGeneAbsencePerturbation(
                systematic_gene_name="YBR001W",
                gene_namespace="scerevisiae_s288c_sgd_systematic",
                perturbed_gene_name="YBR001W",
                strain_id="CCC",
                pangenome_orf_id="4-YBR001W",
                sequence_source=ni.SEQUENCE_SOURCE,
            ),
            DeletionPerturbation(
                systematic_gene_name="YDL227C",
                perturbed_gene_name="YDL227C",
                description=ni.HO_DELETION_DESCRIPTION,
            ),
            _call(
                "CCC",
                "YBR001W",
                lof,
                code=4,
                name="YBR001W",
                gene="NTH2",
                span=_span("chrII", 2, 9),
            ),
            _call(
                "CCC",
                "YDL227C",
                lof,
                code=3,
                name="YDL227C",
                gene="HO",
                span=_span("chrIV", 1, 4),
            ),
            _call(
                "CCC",
                "DDI3",
                fs,
                code=None,
                name=None,
                gene="DDI3",
                span=_span("chrXIV", 11452, 12129),
            ),
        ],
        natural_isolate=genotypes.descriptors["CCC"],
    )
    built = genotypes.genotype("CCC")
    assert built.model_dump() == expected.model_dump()
    assert ni.isolate_genotype("CCC", genotypes) == built
    descriptor = built.natural_isolate
    assert descriptor is not None
    assert (
        descriptor.genetic_manipulation,
        descriptor.ploidy,
        descriptor.zygosity,
        descriptor.chromosome_copy_changes,
        descriptor.provenance_gaps,
    ) == (IsolateManipulation.ho_deletion, 1, GenomeZygosity.heterozygous, [], [])
    assert descriptor.annotation_ref == _ref(
        data_root, ni.SUPPLEMENTARY_TABLES_NAME, ni.TABLE_S1_SHEET
    )


def test_ho_deletion_sits_beside_the_natural_absence_the_matrix_states(
    genotypes: ni.NaturalIsolateGenotypes,
) -> None:
    """BBB: the presence matrix marks HO absent and Table S1 marks it deleted; both
    released statements are stored, and the base perturbations (the pre-#988 genotype)
    are untouched. Its two calls stand on their matrices alone: YAL001C, whose inferred
    allele equals S288C, and the code-less SLM6 row under its Gene.
    """
    perts = genotypes.genotype("BBB").perturbations
    assert [(p.perturbation_type, p.systematic_gene_name) for p in perts] == [
        ("natural_loss_of_function", "SLM6"),
        ("natural_loss_of_function", "YAL001C"),
        ("deletion", "YDL227C"),
        ("natural_gene_absence", "YDL227C"),
    ]
    base = genotypes.base_perturbations("BBB")
    assert [(p.perturbation_type, p.systematic_gene_name) for p in base] == [
        ("natural_gene_absence", "YDL227C")
    ]
    assert base[0] in perts
    slm6 = perts[0]
    assert isinstance(slm6, NaturalLossOfFunctionPerturbation)
    assert (slm6.release_code, slm6.release_name, slm6.release_gene_name) == (
        None,
        None,
        "SLM6",
    )
    assert slm6.release_span == _span("chrII", 739936, 740388)


def test_every_isolate_built_with_its_calls_and_descriptor(
    genotypes: ni.NaturalIsolateGenotypes,
) -> None:
    """Each isolate's perturbations by (type, gene key, cause), sorted as ``Genotype``
    sorts. Every released call is stored: the coded frameshift row under its code's ORF
    name, each code-less row under its Gene, and no call is dropped.

    Two calls on one gene tie on ``Genotype``'s sort key (gene, type, name), so they keep
    the builder's order: by cause value, frameshift before nonsense/SIFT.
    """

    def summary(isolate: str) -> list[tuple[str, str, str | None]]:
        return [
            (p.perturbation_type, p.systematic_gene_name, getattr(p, "cause", None))
            for p in genotypes.genotype(isolate).perturbations
        ]

    lof = "nonsense_or_sift"
    fs = "homozygous_frameshift"
    assert summary("AAA") == [
        ("natural_gene_presence", "2-EC1118_1F14_0012g", None),
        ("natural_loss_of_function", "AAD16", fs),
        ("natural_loss_of_function", "MOVED", fs),
        ("natural_loss_of_function", "YAL001C", fs),
        ("natural_loss_of_function", "YAL001C", lof),
        ("sequence_variant", "YAL001C", None),
        ("natural_loss_of_function", "YBR001W", lof),
    ]
    assert summary("SACE_YAU") == [
        ("natural_gene_presence", "2-EC1118_1F14_0012g", None),
        ("natural_loss_of_function", "PAU8", fs),
        ("sequence_variant", "YAL001C", None),
    ]
    assert summary("DDD") == [
        ("natural_loss_of_function", "LTO1", fs),
        ("natural_loss_of_function", "YAL002W", fs),
    ]
    coded = [
        p
        for p in genotypes.genotype("AAA").perturbations
        if isinstance(p, NaturalLossOfFunctionPerturbation)
        and p.cause is NaturalLossOfFunctionCause.homozygous_frameshift
        and p.release_code is not None
    ]
    assert [(p.systematic_gene_name, p.release_gene_name) for p in coded] == [
        ("YAL001C", "TFC3")
    ]
    presence = genotypes.genotype("SACE_YAU").perturbations[0]
    assert isinstance(presence, NaturalGenePresencePerturbation)
    assert presence.copy_number == 1.0


def test_descriptor_gaps_for_xn_empty_aneuploidy_and_unstated_manipulation(
    genotypes: ni.NaturalIsolateGenotypes,
) -> None:
    yau = genotypes.descriptors["SACE_YAU"]
    assert yau.ploidy is None
    assert yau.chromosome_copy_changes == [
        ChromosomeCopyChange(chromosome=9, change=1),
        ChromosomeCopyChange(chromosome=3, change=-1),
    ]
    assert [(g.field, g.reason) for g in yau.provenance_gaps] == [
        ("genetic_manipulation", ProvenanceGapReason.not_reported_by_primary),
        ("ploidy", ProvenanceGapReason.not_reported_by_primary),
    ]
    assert (
        yau.provenance_gaps[1].note
        == "Table S1 writes the ploidy as 'Xn' (not determined)"
    )
    assert yau.provenance_gaps[0].looked_in is not None
    assert yau.provenance_gaps[0].looked_in.page == "sheet 'Table S1' row 8"
    ddd = genotypes.descriptors["DDD"]
    assert (ddd.ploidy, ddd.chromosome_copy_changes) == (3, None)
    assert [g.field for g in ddd.provenance_gaps] == [
        "genetic_manipulation",
        "chromosome_copy_changes",
    ]


def test_ledger_counts_what_the_genotypes_carry(
    genotypes: ni.NaturalIsolateGenotypes, tmp_path: Path
) -> None:
    ledger = genotypes.write_ledger(tmp_path / "ledger.json")
    assert (
        ni.NaturalIsolateGenotypeLedger.model_validate_json(
            (tmp_path / "ledger.json").read_text()
        )
        == ledger
    )
    assert (
        ledger.n_isolates,
        ledger.n_sequence_variants,
        ledger.n_natural_presences,
        ledger.n_natural_absences,
    ) == (5, 3, 2, 2)
    assert ledger.ho_deleted == ["BBB", "CCC"]
    assert ledger.ploidy == {"1": 2, "2": 1, "3": 1, "Xn": 1}
    assert ledger.zygosity == {"heterozygous": 2, "homozygous": 3}
    assert (
        ledger.n_aneuploid,
        ledger.n_euploid,
        ledger.n_aneuploidy_not_annotated,
        ledger.n_manipulation_not_stated,
    ) == (1, 3, 1, 3)
    assert ledger.loss_of_function == {
        "nonsense_or_sift": ni.LossOfFunctionTally(
            n_calls=5, n_genes=3, n_without_pangenome_code=0
        ),
        "homozygous_frameshift": ni.LossOfFunctionTally(
            n_calls=8, n_genes=8, n_without_pangenome_code=7
        ),
    }


def test_genotype_isolate_reads_the_descriptor(
    genotypes: ni.NaturalIsolateGenotypes,
) -> None:
    assert ni.genotype_isolate(genotypes.genotype("DDD").model_dump()) == "DDD"
    assert ni.genotype_isolate({"perturbations": []}) is None


# --------------------------------------------------------------------------- #
# Caching, files and refusals
# --------------------------------------------------------------------------- #
def test_variant_cache_serves_a_subset_and_is_rewritten_for_a_superset(
    data_root: Path, tmp_path: Path
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    ni.link_genotype_files(str(raw), str(data_root))
    cache = tmp_path / "cache"
    cache.mkdir()
    fsa = osp.join(data_root, "torchcell-genomes", SGD_S288C_R64, ni.SGD_FSA_NAME)
    full = ni.load_or_diff_variants(
        raw / ni.REFGENE_TAR_NAME, fsa, set(ISOLATES), cache
    )
    assert sorted(full.variants["strain_id"]) == ["AAA", "CCC", "SACE_YAU"]
    assert full.reference_genes == frozenset({"YAL001C", "YDL227C"})
    index = json.loads((cache / "sequence_variants.index.json").read_text())
    assert index["isolates"] == sorted(ISOLATES)
    # a subset reads the cache: with the FASTA gone, a re-diff would raise
    subset = ni.load_or_diff_variants(
        raw / ni.REFGENE_TAR_NAME, tmp_path / "gone.fsa", {"AAA", "BBB"}, cache
    )
    assert subset.variants["strain_id"].tolist() == ["AAA"]
    with pytest.raises(FileNotFoundError):
        ni.load_or_diff_variants(
            raw / ni.REFGENE_TAR_NAME, tmp_path / "gone.fsa", {*ISOLATES, "EEE"}, cache
        )


def test_link_and_pins_cover_the_six_files(data_root: Path, tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    ni.link_genotype_files(str(raw), str(data_root))
    tier = data_root / "torchcell-genomes" / PETER2018_1011
    assert {p.name: os.readlink(p) for p in raw.iterdir()} == {
        name: str(tier / name) for name in ni.GENOTYPE_FILES
    }
    assert ni.genotype_file_pins(str(data_root)) == {
        name: hashlib.sha256(data).hexdigest() for name, data in _peter_files().items()
    }
    assert ni.panel_isolates(raw) == set(ISOLATES)


def test_table_s1_must_list_exactly_the_panel(data_root: Path, tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    ni.link_genotype_files(str(raw), str(data_root))
    target = raw / ni.SUPPLEMENTARY_TABLES_NAME
    target.unlink()
    target.write_bytes(tables_workbook(_TABLE_S1[:-1]))
    with pytest.raises(ValueError, match="Table S1 does not list exactly the 1,011"):
        ni.NaturalIsolateGenotypes(raw, tmp_path, ["AAA"], str(data_root))


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (TableS1Row(isolate="AAA", ho_deletion="maybe"), "'HO deletion' reads 'maybe'"),
        (TableS1Row(isolate="AAA", ploidy="2n"), "'Ploidy' reads '2n'"),
        (TableS1Row(isolate="AAA", aneuploidies="aneu;+1*9"), "unparseable aneuploidy"),
        (TableS1Row(isolate="AAA", zygosity="mixed"), "is not a valid GenomeZygosity"),
    ],
)
def test_a_cell_outside_the_table_vocabulary_is_refused(
    tmp_path: Path, row: TableS1Row, message: str
) -> None:
    path = tmp_path / "s1.xls"
    path.write_bytes(tables_workbook([row]))
    table = ni.read_table_s1(path)
    workbook = ArtifactRef(
        tier="genomes", key=PETER2018_1011, path="w.xls", sha256="0" * 64
    )
    with pytest.raises(ValueError, match=message):
        ni.isolate_descriptor(table["AAA"], workbook)


def test_table_s1_refuses_a_repeated_isolate(tmp_path: Path) -> None:
    path = tmp_path / "s1.xls"
    path.write_bytes(
        tables_workbook([TableS1Row(isolate="AAA"), TableS1Row(isolate="AAA")])
    )
    with pytest.raises(ValueError, match="Table S1 lists 'AAA' twice"):
        ni.read_table_s1(path)


def _frame(data: bytes) -> pd.DataFrame:
    text = gzip.decompress(data).decode().replace("\r", "\n")
    return pd.read_csv(io.StringIO(text), sep="\t")


def test_a_lof_row_is_keyed_by_its_name_whatever_the_code_names() -> None:
    """The loss-of-function matrix's NAME is the call's key, as released; it is not
    checked against the name another file gives the same code.
    """
    rows, matrix, columns = ni.lof_rows(
        _frame(
            lof_matrix(
                ISOLATES,
                [LofRow(code=1, name="YAL002W", gene="X", ch=1, start=1, stop=8)],
            )
        ),
        set(ISOLATES),
    )
    assert [(r.systematic_gene_name, r.release_code, r.release_name) for r in rows] == [
        ("YAL002W", 1, "YAL002W")
    ]
    assert (matrix.shape, columns) == ((1, 5), ISOLATES)


def test_a_lof_row_without_a_code_or_a_name_is_refused() -> None:
    frame = _frame(lof_matrix(ISOLATES, _LOF[:1]))
    frame.loc[0, "NAME"] = None
    with pytest.raises(ValueError, match="a row without a CODE or a NAME"):
        ni.lof_rows(frame, set(ISOLATES))


def test_a_coded_frameshift_row_must_name_a_reference_orf(
    genotypes: ni.NaturalIsolateGenotypes,
) -> None:
    frame = _frame(
        frameshift_matrix(
            ISOLATES, [FrameshiftRow(code=2, gene="X", ch=1, start=1, stop=8)]
        )
    )
    with pytest.raises(ValueError, match="CODE 2 is not a reference ORF"):
        ni.frameshift_rows(frame, genotypes.pangenome)


def test_a_matrix_must_index_the_panel_with_zeros_and_ones() -> None:
    with pytest.raises(ValueError, match="does not index the 1,011 isolates"):
        ni.lof_rows(_frame(lof_matrix(ISOLATES[:-1], [])), set(ISOLATES))
    frame = _frame(lof_matrix(ISOLATES, _LOF[:1]))
    frame.loc[0, "AAA"] = 2
    with pytest.raises(ValueError, match="a value other than 0 or 1"):
        ni.lof_rows(frame, set(ISOLATES))


def test_parse_aneuploidy_reads_the_legend_and_refuses_other_text() -> None:
    assert ni.parse_aneuploidy("euploid") == []
    assert ni.parse_aneuploidy("aneu;+2*1;-1*16;") == [
        ChromosomeCopyChange(chromosome=1, change=2),
        ChromosomeCopyChange(chromosome=16, change=-1),
    ]
    for text in ("aneuploid", "aneu;+1*9", "aneu;"):
        with pytest.raises(ValueError, match="unparseable aneuploidy annotation"):
            ni.parse_aneuploidy(text)
    with pytest.raises(ValueError, match="less than or equal to 16"):
        ni.parse_aneuploidy("aneu;+1*17;")


def test_register_supplementary_tables_copies_and_records_once(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The library copy is verified, copied into the tier and appended with its PMC
    retrieval; a second call finds it listed and changes nothing.
    """
    tier = data_root / "torchcell-genomes" / PETER2018_1011
    manifest = json.loads((tier / "manifest.json").read_text())
    manifest["files"] = [
        f for f in manifest["files"] if f["path"] != ni.SUPPLEMENTARY_TABLES_NAME
    ]
    (tier / "manifest.json").write_text(json.dumps(manifest))
    (tier / ni.SUPPLEMENTARY_TABLES_NAME).unlink()
    data = tables_workbook(_TABLE_S1)
    library = data_root / "torchcell-library" / ni.CITATION_KEY / ni.LIBRARY_TABLES_REL
    library.parent.mkdir(parents=True)
    library.write_bytes(data)
    monkeypatch.setattr(ni, "TABLES_SHA256", hashlib.sha256(data).hexdigest())
    path = ni.register_supplementary_tables(str(data_root))
    record = load_genome_manifest(PETER2018_1011, str(data_root)).record(
        ni.SUPPLEMENTARY_TABLES_NAME
    )
    assert (record.role, record.sha256, record.bytes, record.source) == (
        "annotation",
        hashlib.sha256(data).hexdigest(),
        len(data),
        ni.TABLES_URL,
    )
    assert record.retrieval is not None
    assert record.retrieval.method.value == "pmc_cloud"
    before = (tier / "manifest.json").read_text()
    assert ni.register_supplementary_tables(str(data_root)) == path
    assert (tier / "manifest.json").read_text() == before


def test_register_release_readme_copies_and_records_once(
    data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The host's README is verified, copied into the raw mirror and appended to its
    manifest with a direct_url retrieval; a second call finds it listed.
    """
    mirror = data_root / "torchcell-raw" / ni.CITATION_KEY
    mirror.mkdir(parents=True)
    (mirror / "data").mkdir()
    (mirror / "manifest.json").write_text(
        Manifest(citation_key=ni.CITATION_KEY, files=[]).model_dump_json()
    )
    readme = tmp_path / "README.txt"
    readme.write_bytes(b"- " + ni.LOF_README_QUOTE.encode() + b" \r\n")
    sha = hashlib.sha256(readme.read_bytes()).hexdigest()
    monkeypatch.setattr(ni, "README_SHA256", sha)
    path = ni.register_release_readme(readme, str(data_root))
    manifest = Manifest.model_validate_json(path.read_text())
    (record,) = manifest.files
    assert (record.path, record.sha256, record.source, record.role) == (
        ni.README_REL,
        sha,
        ni.RELEASE_HOST + ni.README_NAME,
        "raw_data",
    )
    assert record.retrieval is not None
    assert record.retrieval.method.value == "direct_url"
    assert (mirror / ni.README_REL).read_bytes() == readme.read_bytes()
    before = path.read_text()
    assert ni.register_release_readme(readme, str(data_root)) == path
    assert path.read_text() == before
    wrong = tmp_path / "other.txt"
    wrong.write_bytes(b"other")
    (mirror / "manifest.json").write_text(
        Manifest(citation_key=ni.CITATION_KEY, files=[]).model_dump_json()
    )
    with pytest.raises(RawSha256MismatchError, match="sha256 mismatch"):
        ni.register_release_readme(wrong, str(data_root))
    # the refusal leaves the mirror's copy as it was
    assert (mirror / ni.README_REL).read_bytes() == readme.read_bytes()


# --------------------------------------------------------------------------- #
# The real release (skipped where the mirror is not on disk)
# --------------------------------------------------------------------------- #
def _real_data_root() -> str:
    from dotenv import load_dotenv

    load_dotenv()
    return os.environ.get("DATA_ROOT", "")


_REAL_TIER = osp.join(_real_data_root(), "torchcell-genomes", PETER2018_1011)
_REAL_LIBRARY = osp.join(_real_data_root(), "torchcell-library")
_PHENO = osp.join(
    _real_data_root(),
    "torchcell-raw",
    ni.CITATION_KEY,
    "data/phenoMatrix_35ConditionsNormalizedByYPD.tab.gz",
)


@pytest.mark.data
@pytest.mark.skipif(not osp.isfile(_PHENO), reason="Peter 2018 mirrors not on disk")
def test_table_s1_counts_over_the_971_phenotyped_isolates() -> None:
    """The counts #988 states, reproduced from the tier's Table S1 by the descriptor."""
    phenotyped = set(pd.read_csv(_PHENO, sep="\t", index_col=0).index.astype(str))
    path = osp.join(_REAL_TIER, ni.SUPPLEMENTARY_TABLES_NAME)
    table = ni.read_table_s1(path)
    workbook = ni.tier_file_ref(ni.SUPPLEMENTARY_TABLES_NAME)
    descriptors = [
        ni.isolate_descriptor(table[i], workbook) for i in sorted(phenotyped)
    ]
    from collections import Counter

    assert len(descriptors) == 971
    assert Counter(
        "Xn" if d.ploidy is None else str(d.ploidy) for d in descriptors
    ) == {"2": 750, "1": 131, "3": 46, "4": 38, "5": 5, "Xn": 1}
    assert Counter(d.zygosity.value for d in descriptors) == {
        "heterozygous": 501,
        "homozygous": 470,
    }
    assert sum(1 for d in descriptors if d.chromosome_copy_changes) == 211
    assert sum(1 for d in descriptors if d.chromosome_copy_changes is None) == 1
    assert sum(1 for d in descriptors if d.genetic_manipulation is not None) == 122
    assert len(table) == 1011
    assert sum(1 for r in table.values() if r.ho_deletion == "yes") == 124


@pytest.mark.data
@pytest.mark.skipif(not osp.isdir(_REAL_LIBRARY), reason="library mirror not on disk")
def test_every_sourced_value_is_verbatim_in_its_pinned_source() -> None:
    for key, sv in ni.SOURCED_VALUES.items():
        audit = audit_sourced_value(sv, _REAL_LIBRARY)
        assert audit.passed, (key, audit.message)
    raw_root = osp.join(_real_data_root(), "torchcell-raw")
    for key, sv in ni.RELEASE_SOURCED_VALUES.items():
        audit = audit_sourced_value(sv, raw_root)
        assert audit.passed, (key, audit.message)
    sheet = pd.read_excel(
        osp.join(_REAL_LIBRARY, ni.CITATION_KEY, ni.LIBRARY_TABLES_REL),
        sheet_name=ni.TABLE_S1_SHEET,
        header=None,
    )
    assert sheet.iloc[1, 0] == ni.ANEUPLOIDY_LEGEND
    assert list(sheet.iloc[3, :6]) == [
        "Isolate name",
        "Standardized name",
        "Isolation",
        "Ecological origins",
        "Geographical origins",
        "HO deletion",
    ]


@pytest.mark.slow
@pytest.mark.data
@pytest.mark.skipif(not osp.isfile(_PHENO), reason="Peter 2018 mirrors not on disk")
def test_the_971_isolate_ledger_reproduces_the_issue_inventory(tmp_path: Path) -> None:
    """#987's inventory over the 971 phenotyped isolates: every released call."""
    raw = tmp_path / "raw"
    raw.mkdir()
    ni.link_genotype_files(str(raw))
    phenotyped = set(pd.read_csv(_PHENO, sep="\t", index_col=0).index.astype(str))
    ledger: Any = ni.NaturalIsolateGenotypes(raw, tmp_path, phenotyped).ledger()
    lof = ledger.loss_of_function["nonsense_or_sift"]
    assert (lof.n_calls, lof.n_genes, lof.n_without_pangenome_code) == (231339, 4069, 0)
    fs = ledger.loss_of_function["homozygous_frameshift"]
    assert (fs.n_calls, fs.n_without_pangenome_code) == (141453, 8625)
    assert len(ledger.ho_deleted) == 122
