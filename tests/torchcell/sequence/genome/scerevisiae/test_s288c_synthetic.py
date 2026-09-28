# tests/torchcell/sequence/genome/scerevisiae/test_s288c_synthetic.py
# [[tests.torchcell.sequence.genome.scerevisiae.test_s288c_synthetic]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/sequence/genome/scerevisiae/test_s288c_synthetic.py
"""SCerevisiaeGenome on a hand-written release under tmp_path.

The genomes-tier ``resolve`` is stubbed at its import site
(``torchcell.sequence.genome.scerevisiae.s288c.resolve``) to map each release filename
onto a file written by the fixture, and ``data.db`` is built by the fixture with the
same ``gffutils.create_db`` arguments the constructor uses, so the genome is built with
``overwrite=False``. ``go.obo`` is written too, so ``download_url`` is never reached
(one test stubs it to pin the missing-file branch).

Fixture (GFF coordinates are 1-based inclusive; Python slices are ``[start - 1:end]``):

* ``chrI`` (60 nt), ``chrII`` (48 nt), ``chrmt`` (12 nt); ``chrIII`` is in the GFF but
  not in the FASTA.
* ``YAL001C`` gene, ``-``, 11..25, with an mRNA and a CDS child; standard name ``TFC3``,
  aliases ``TFC3, SHARED, FUN24``; GO terms ``GO:0000001`` and ``GO:0000003``.
* ``tA(UGC)A`` ``tRNA_gene``, ``+``, 1..8, standard name ``SUP1``, alias ``TRNAX``.
* ``YAL002W`` gene, ``+``, 31..45, a ``five_prime_UTR_intron`` at 31..33 and one CDS
  at 34..45, so its sequence is the CDS (``chrI[33:45]``); aliases ``OLD1, SHARED``.
* ``YBL001W`` dubious gene, ``+``, 5..16, no GO terms, no aliases.
* ``YBR999W`` ``blocked_reading_frame``, ``-``, 20..30, standard name ``FLO8X``, alias
  ``TRNAX`` (so ``TRNAX`` names two non-gene loci).
* ``ADE1`` ``region``, 32..34 (a non-locus feature whose id is a gene name).
* ``YBL002W`` gene, ``+``, 35..47, intron 35..36 and two Verified CDS 37..40 and
  43..47, so its coordinates collapse to min/max = 37..47.
* ``YCL001W`` gene on ``chrIII`` (absent from the FASTA), alias ``ORPHAN``.
* ``Q0010`` gene on ``chrmt``, 1..9, GO ``GO:0000002``.

``go.obo`` holds ``GO:0000001`` (live) and ``GO:0000002`` (obsolete); ``GO:0000003`` is
absent. ``GODag`` skips obsolete terms by default, so ``GO:0000002`` is also absent
from the loaded DAG. Attribute lists come back in file order.
"""

import os
import os.path as osp
import pickle
import re
from pathlib import Path

import gffutils
import pandas as pd
import pytest
from gffutils.exceptions import FeatureNotFoundError
from sortedcontainers import SortedDict, SortedSet

import torchcell.sequence.genome.scerevisiae.s288c as s288c
from torchcell.sequence import DnaWindowResult
from torchcell.sequence.genome.scerevisiae.s288c import (
    GeneNameResolution,
    GeneNameStatus,
    SCerevisiaeGenome,
)

VERSION = "R64-4-1_20230830"

CHR_I = "GATTACAGCAATGAAACCCGGGTAACCTTAGGACTGCAATGTCTAAAGGCTGATCCGATC"
CHR_II = "CCGGAATTCATGCATGCAAGCTTGGATCCTCTAGAGTCGACCTGCAGG"
CHR_MT = "ATGTTTAAACCC"

FASTA_DNA = (
    ">ref|NC_001133| [org=Saccharomyces cerevisiae] [chromosome=I]\n"
    f"{CHR_I}\n"
    ">ref|NC_001134| [org=Saccharomyces cerevisiae] [chromosome=II]\n"
    f"{CHR_II}\n"
    ">ref|NC_001224| [org=Saccharomyces cerevisiae] [location=mitochondrion]\n"
    f"{CHR_MT}\n"
)

# 15 nt coding sequence of YAL001C: reverse complement of CHR_I[10:25].
CDS_YAL001C = "TTACCCGGGTTTCAT"
FASTA_PROTEIN = ">YAL001C TFC3 SGDID:S000000001\nMKPG*\n"
FASTA_CDS = f">YAL001C TFC3 SGDID:S000000001\n{CDS_YAL001C}\n"

GFF_ROWS = [
    (
        "chrI",
        "gene",
        11,
        25,
        "-",
        "ID=YAL001C;Name=YAL001C;gene=TFC3;Alias=TFC3,SHARED,FUN24;"
        "Ontology_term=GO:0000003,GO:0000001,SO:0000704;Note=Subunit;display=Subunit;"
        "dbxref=SGD:S000000001;orf_classification=Verified",
    ),
    ("chrI", "mRNA", 11, 25, "-", "ID=YAL001C_mRNA;Name=YAL001C_mRNA;Parent=YAL001C"),
    (
        "chrI",
        "CDS",
        11,
        25,
        "-",
        "ID=YAL001C_CDS;Name=YAL001C_CDS;Parent=YAL001C_mRNA;orf_classification=Verified",
    ),
    ("chrI", "tRNA_gene", 1, 8, "+", "ID=tA(UGC)A;Name=tA(UGC)A;gene=SUP1;Alias=TRNAX"),
    (
        "chrI",
        "gene",
        31,
        45,
        "+",
        "ID=YAL002W;Name=YAL002W;gene=VPS8;Alias=OLD1,SHARED;"
        "Ontology_term=GO:0000002;orf_classification=Verified",
    ),
    (
        "chrI",
        "five_prime_UTR_intron",
        31,
        33,
        "+",
        "ID=YAL002W_intron;Name=YAL002W_intron;Parent=YAL002W",
    ),
    (
        "chrI",
        "CDS",
        34,
        45,
        "+",
        "ID=YAL002W_CDS;Name=YAL002W_CDS;Parent=YAL002W;orf_classification=Verified",
    ),
    ("chrII", "gene", 5, 16, "+", "ID=YBL001W;Name=YBL001W;orf_classification=Dubious"),
    (
        "chrII",
        "blocked_reading_frame",
        20,
        30,
        "-",
        "ID=YBR999W;Name=YBR999W;gene=FLO8X;Alias=TRNAX",
    ),
    ("chrII", "region", 32, 34, "+", "ID=ADE1;Name=ADE1"),
    (
        "chrII",
        "gene",
        35,
        47,
        "+",
        "ID=YBL002W;Name=YBL002W;Ontology_term=GO:0000001;orf_classification=Verified",
    ),
    (
        "chrII",
        "five_prime_UTR_intron",
        35,
        36,
        "+",
        "ID=YBL002W_intron;Name=YBL002W_intron;Parent=YBL002W",
    ),
    (
        "chrII",
        "CDS",
        37,
        40,
        "+",
        "ID=YBL002W_CDS1;Name=YBL002W_CDS1;Parent=YBL002W;orf_classification=Verified",
    ),
    (
        "chrII",
        "CDS",
        43,
        47,
        "+",
        "ID=YBL002W_CDS2;Name=YBL002W_CDS2;Parent=YBL002W;orf_classification=Verified",
    ),
    ("chrIII", "gene", 1, 6, "+", "ID=YCL001W;Name=YCL001W;Alias=ORPHAN"),
    (
        "chrmt",
        "gene",
        1,
        9,
        "+",
        "ID=Q0010;Name=Q0010;Ontology_term=GO:0000002;orf_classification=Verified",
    ),
]

GO_OBO = """format-version: 1.2

[Term]
id: GO:0000001
name: mitochondrion inheritance
namespace: biological_process

[Term]
id: GO:0000002
name: retired process
namespace: biological_process
is_obsolete: true
"""


def _gff_text() -> str:
    lines = ["##gff-version 3"]
    for seqid, ftype, start, end, strand, attrs in GFF_ROWS:
        lines.append(
            "\t".join(
                [seqid, "SGD", ftype, str(start), str(end), ".", strand, ".", attrs]
            )
        )
    return "\n".join(lines) + "\n"


def write_release(root: Path) -> dict[str, str]:
    """Write the four release files and go.obo; return filename -> path."""
    release = root / "release"
    release.mkdir(parents=True)
    files = {
        f"S288C_reference_sequence_{VERSION}.fsa": FASTA_DNA,
        f"saccharomyces_cerevisiae_{VERSION}.gff": _gff_text(),
        f"orf_trans_all_{VERSION}.fasta": FASTA_PROTEIN,
        f"orf_coding_all_{VERSION}.fasta": FASTA_CDS,
    }
    paths: dict[str, str] = {}
    for name, text in files.items():
        (release / name).write_text(text)
        paths[name] = str(release / name)
    return paths


def build_db(gff_path: str, genome_root: Path) -> None:
    """Seed data.db exactly as ``SCerevisiaeGenome(overwrite=True)`` would."""
    genome_root.mkdir(parents=True, exist_ok=True)
    gffutils.create_db(
        gff_path,
        dbfn=str(genome_root / "data.db"),
        force=True,
        keep_order=True,
        merge_strategy="merge",
        sort_attribute_values=True,
    )


@pytest.fixture
def release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Release files on disk plus the ``resolve`` stub that serves them."""
    paths = write_release(tmp_path)
    monkeypatch.setattr(
        s288c, "resolve", lambda assembly_set, filename: paths[filename]
    )
    go_root = tmp_path / "go"
    go_root.mkdir()
    (go_root / "go.obo").write_text(GO_OBO)
    paths["__go_root__"] = str(go_root)
    paths["__genome_root__"] = str(tmp_path / "genome")
    return paths


@pytest.fixture
def genome(release: dict[str, str]) -> SCerevisiaeGenome:
    """The genome over the fixture release, built with ``overwrite=False``."""
    build_db(
        release[f"saccharomyces_cerevisiae_{VERSION}.gff"],
        Path(release["__genome_root__"]),
    )
    return SCerevisiaeGenome(
        genome_root=release["__genome_root__"],
        go_root=release["__go_root__"],
        overwrite=False,
    )


def _window(
    gene_id: str,
    chromosome: int,
    strand: str,
    start: int,
    end: int,
    seq: str,
    start_window: int,
    end_window: int,
) -> DnaWindowResult:
    return DnaWindowResult(
        id=gene_id,
        chromosome=chromosome,
        strand=strand,
        start=start,
        end=end,
        seq=seq,
        start_window=start_window,
        end_window=end_window,
    )


def test_constructor_resolves_the_four_release_files(
    release: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The constructor asks the tier for exactly the four R64-4-1 files, in order."""
    calls: list[tuple[str, str]] = []
    paths = {k: v for k, v in release.items() if not k.startswith("__")}

    def recording_resolve(assembly_set: str, filename: str) -> str:
        calls.append((assembly_set, filename))
        return paths[filename]

    build_db(
        release[f"saccharomyces_cerevisiae_{VERSION}.gff"],
        Path(release["__genome_root__"]),
    )
    monkeypatch.setattr(s288c, "resolve", recording_resolve)
    genome = SCerevisiaeGenome(
        genome_root=release["__genome_root__"],
        go_root=release["__go_root__"],
        overwrite=False,
    )
    assert calls == [
        (SCerevisiaeGenome.ASSEMBLY_SET, f"S288C_reference_sequence_{VERSION}.fsa"),
        (SCerevisiaeGenome.ASSEMBLY_SET, f"saccharomyces_cerevisiae_{VERSION}.gff"),
        (SCerevisiaeGenome.ASSEMBLY_SET, f"orf_trans_all_{VERSION}.fasta"),
        (SCerevisiaeGenome.ASSEMBLY_SET, f"orf_coding_all_{VERSION}.fasta"),
    ]
    assert genome.genome_version == VERSION


def test_chromosome_maps(genome: SCerevisiaeGenome) -> None:
    """FASTA descriptions map roman numerals to ints and the mitochondrion to 0."""
    assert genome.chr_to_nc == {
        1: "ref|NC_001133|",
        2: "ref|NC_001134|",
        0: "ref|NC_001224|",
    }
    assert genome.nc_to_chr == {
        "ref|NC_001133|": 1,
        "ref|NC_001134|": 2,
        "ref|NC_001224|": 0,
    }
    assert genome.chr_to_len == {1: 60, 2: 48, 0: 12}


def test_gene_set_holds_only_gene_features(genome: SCerevisiaeGenome) -> None:
    """The six ``gene`` rows; tRNA, blocked_reading_frame, region, CDS are excluded."""
    assert list(genome.gene_set) == [
        "Q0010",
        "YAL001C",
        "YAL002W",
        "YBL001W",
        "YBL002W",
        "YCL001W",
    ]


def test_feature_types(genome: SCerevisiaeGenome) -> None:
    """Every featuretype written in the GFF, sorted."""
    assert genome.feature_types == [
        "CDS",
        "blocked_reading_frame",
        "five_prime_UTR_intron",
        "gene",
        "mRNA",
        "region",
        "tRNA_gene",
    ]


def test_minus_strand_gene_is_reverse_complemented(genome: SCerevisiaeGenome) -> None:
    """YAL001C (-, 11..25): seq = revcomp(CHR_I[10:25]) = revcomp(ATGAAACCCGGGTAA)."""
    gene = genome["YAL001C"]
    assert gene is not None
    assert (gene.id, gene.chromosome, gene.strand, gene.start, gene.end) == (
        "YAL001C",
        1,
        "-",
        11,
        25,
    )
    assert CHR_I[10:25] == "ATGAAACCCGGGTAA"
    assert gene.seq == "TTACCCGGGTTTCAT"
    assert gene.alias == ["TFC3", "SHARED", "FUN24"]
    assert gene.name == ["YAL001C"]
    assert gene.ontology_term == ["GO:0000003", "GO:0000001", "SO:0000704"]
    assert gene.note == ["Subunit"]
    assert gene.display == ["Subunit"]
    assert gene.dbxref == ["SGD:S000000001"]
    assert gene.orf_classification == ["Verified"]
    assert gene.go == SortedSet(["GO:0000001", "GO:0000003"])
    assert gene.protein is not None and str(gene.protein.seq) == "MKPG*"
    assert str(gene.cds.seq) == CDS_YAL001C


def test_five_prime_utr_intron_gene_uses_its_cds(genome: SCerevisiaeGenome) -> None:
    """YAL002W: intron 31..33 at the gene start, so the CDS 34..45 is the sequence."""
    gene = genome["YAL002W"]
    assert gene is not None
    assert (gene.start, gene.end, gene.strand) == (34, 45, "+")
    assert gene.seq == CHR_I[33:45] == "CTGCAATGTCTA"
    assert gene.protein is None and gene.cds is None


def test_two_verified_cds_collapse_to_min_max(genome: SCerevisiaeGenome) -> None:
    """YBL002W: two Verified CDS 37..40 and 43..47 give start 37, end 47."""
    gene = genome["YBL002W"]
    assert gene is not None
    assert (gene.chromosome, gene.start, gene.end) == (2, 37, 47)
    assert gene.seq == CHR_II[36:47] == "TCGACCTGCAG"


def test_mitochondrial_gene_is_chromosome_zero(genome: SCerevisiaeGenome) -> None:
    """Q0010 on chrmt: chromosome 0 and the gene row itself (1..9) is the sequence."""
    gene = genome["Q0010"]
    assert gene is not None
    assert (gene.chromosome, gene.start, gene.end, gene.seq) == (0, 1, 9, "ATGTTTAAA")


def test_dubious_gene_without_go_has_go_none(genome: SCerevisiaeGenome) -> None:
    """YBL001W has no Ontology_term, so ``go`` is None and so are its aliases."""
    gene = genome["YBL001W"]
    assert gene is not None
    assert (gene.seq, gene.go, gene.alias, gene.orf_classification) == (
        CHR_II[4:16],
        None,
        None,
        ["Dubious"],
    )


def test_gene_on_chromosome_missing_from_fasta_returns_none(
    genome: SCerevisiaeGenome, capsys: pytest.CaptureFixture[str]
) -> None:
    """YCL001W is on chrIII, absent from chr_to_nc: the KeyError becomes None + a print."""
    assert genome["YCL001W"] is None
    assert capsys.readouterr().out == (
        "Gene YCL001W not found in genome, only systematic names (ID) are supported.\n"
    )


def test_unknown_id_raises_feature_not_found(genome: SCerevisiaeGenome) -> None:
    """Finding: s288c.py:964 catches KeyError, but gffutils raises FeatureNotFoundError.

    ``FeatureNotFoundError`` derives from ``Exception``, not ``KeyError``, so an id that
    is not in the database escapes ``__getitem__`` instead of returning None.
    """
    assert not issubclass(FeatureNotFoundError, KeyError)
    with pytest.raises(FeatureNotFoundError):
        genome["NOPE"]


def test_gene_repr_names_dna_selection_result(genome: SCerevisiaeGenome) -> None:
    """Finding: s288c.py:398 labels the gene repr ``DnaSelectionResult`` (double space)."""
    gene = genome["YAL001C"]
    assert repr(gene) == (
        "DnaSelectionResult(id=YAL001C, chromosome=1, strand=-, start=11, end=25,  "
        "seq=TTACCCGGGTTTCAT)"
    )


def test_gene_alias_to_systematic_references_genome_api(
    genome: SCerevisiaeGenome,
) -> None:
    """Finding: s288c.py:220 iterates ``self.gene_set`` on a gene, which has none."""
    gene = genome["YAL001C"]
    assert gene is not None
    with pytest.raises(
        AttributeError, match="'SCerevisiaeGene' object has no attribute 'gene_set'"
    ):
        gene.alias_to_systematic  # noqa: B018


def test_codon_frequency_of_cds(genome: SCerevisiaeGenome) -> None:
    """CDS TTA CCC GGG TTT CAT: five codons, 1/5 each, the other 59 at zero."""
    gene = genome["YAL001C"]
    assert gene is not None
    freq = gene.codon_frequency
    assert len(freq) == 64
    assert {k: v for k, v in freq.items() if v} == {
        "TTA": 0.2,
        "CCC": 0.2,
        "GGG": 0.2,
        "TTT": 0.2,
        "CAT": 0.2,
    }


def test_window_max_size_and_symmetric(genome: SCerevisiaeGenome) -> None:
    """YAL001C, 0-based [10, 25), length 15, chrI length 60.

    Max-size 21: flank (21 - 15) // 2 = 3 gives [7, 28). Symmetric 20: flank
    (20 - 15) // 2 = 2 gives [8, 27), 19 nt. Max-size 20: the same flank gives [8, 27),
    one short of 20, and on ``-`` the extra base goes to the end, so [8, 28). All three
    are reverse complemented on ``-``.
    """
    gene = genome["YAL001C"]
    assert gene is not None
    assert gene.window(21) == _window(
        "YAL001C", 1, "-", 11, 25, "AGGTTACCCGGGTTTCATTGC", 7, 28
    )
    assert gene.window(20, is_max_size=False) == _window(
        "YAL001C", 1, "-", 11, 25, "GGTTACCCGGGTTTCATTG", 8, 27
    )
    assert gene.window(20) == _window(
        "YAL001C", 1, "-", 11, 25, "AGGTTACCCGGGTTTCATTG", 8, 28
    )


def test_window_plus_strand_is_forward(genome: SCerevisiaeGenome) -> None:
    """YAL002W, [33, 45), length 12, window 16: flank 2 gives [31, 47) forward.

    Window 15: flank (15 - 12) // 2 = 1 gives [32, 46), one short, and on ``+`` the extra
    base goes upstream, so [31, 46).
    """
    gene = genome["YAL002W"]
    assert gene is not None
    assert gene.window(16) == _window("YAL002W", 1, "+", 34, 45, CHR_I[31:47], 31, 47)
    assert CHR_I[31:47] == "GACTGCAATGTCTAAA"
    assert gene.window(15) == _window(
        "YAL002W", 1, "+", 34, 45, "GACTGCAATGTCTAA", 31, 46
    )


def test_five_prime_minus_strand(genome: SCerevisiaeGenome) -> None:
    """YAL001C (-, end 25): the 5' window is [25, 25 + w), reverse complemented.

    With the start codon it starts at 25 - 3 = 22. Window 40 ends at 65, 5 bp past 60.
    """
    gene = genome["YAL001C"]
    assert gene is not None
    assert gene.window_five_prime(5) == _window(
        "YAL001C", 1, "-", 11, 25, "TAAGG", 25, 30
    )
    assert gene.window_five_prime(5, include_start_codon=True) == _window(
        "YAL001C", 1, "-", 11, 25, "GGTTA", 22, 27
    )
    assert gene.window_five_prime(40, allow_undersize=True) == _window(
        "YAL001C", 1, "-", 11, 25, "GATCGGATCAGCCTTTAGACATTGCAGTCCTAAGG", 25, 60
    )
    with pytest.raises(
        ValueError,
        match=re.escape("five prime size (40) too large ('- strand 5bp outside.)"),
    ):
        gene.window_five_prime(40)


def test_five_prime_plus_strand_includes_first_cds_base(
    genome: SCerevisiaeGenome,
) -> None:
    """Finding: s288c.py:287 sets ``start = self.start`` (1-based) on ``+``.

    The minus strand excludes the gene's own base, but on ``+`` the window
    [34 - w, 34) ends at 0-based 34, so it includes base 33, the first CDS base: the
    5-nt window is CHR_I[29:34] = AGGAC whose last C is CDS position 1. With the start
    codon it is [31, 36). Window 40 starts at -6.
    """
    gene = genome["YAL002W"]
    assert gene is not None
    assert gene.seq[0] == CHR_I[33] == "C"
    assert gene.window_five_prime(5) == _window(
        "YAL002W", 1, "+", 34, 45, "AGGAC", 29, 34
    )
    assert gene.window_five_prime(5, include_start_codon=True) == _window(
        "YAL002W", 1, "+", 34, 45, "GACTG", 31, 36
    )
    assert gene.window_five_prime(40, allow_undersize=True) == _window(
        "YAL002W", 1, "+", 34, 45, CHR_I[0:34], 0, 34
    )
    with pytest.raises(
        ValueError,
        match=re.escape("five prime size (40) too large ('+ strand 6bp outside.)"),
    ):
        gene.window_five_prime(40)


def test_three_prime_minus_strand(genome: SCerevisiaeGenome) -> None:
    """YAL001C (-, start 11): the 3' window is [10 - w, 10), reverse complemented.

    Finding: s288c.py:378 joins the message pieces without a space ("large(").
    With the stop codon the window ends at 10 + 3 = 13. Window 12 starts at -2.
    """
    gene = genome["YAL001C"]
    assert gene is not None
    assert gene.window_three_prime(4) == _window(
        "YAL001C", 1, "-", 11, 25, "TGCT", 6, 10
    )
    assert gene.window_three_prime(4, include_stop_codon=True) == _window(
        "YAL001C", 1, "-", 11, 25, "CATT", 9, 13
    )
    assert gene.window_three_prime(12, allow_undersize=True) == _window(
        "YAL001C", 1, "-", 11, 25, "TGCTGTAATC", 0, 10
    )
    with pytest.raises(
        ValueError, match=re.escape("3utr size (12) too large('- strand 2bp outside.)")
    ):
        gene.window_three_prime(12)


def test_three_prime_plus_strand(genome: SCerevisiaeGenome) -> None:
    """YAL002W (+, end 45): [45, 45 + w); with the stop codon [42, 42 + w).

    Window 20 ends at 65, 5 bp past 60; undersized it is [45, 60).
    """
    gene = genome["YAL002W"]
    assert gene is not None
    assert gene.window_three_prime(5) == _window(
        "YAL002W", 1, "+", 34, 45, "AAGGC", 45, 50
    )
    assert gene.window_three_prime(5, include_stop_codon=True) == _window(
        "YAL002W", 1, "+", 34, 45, "CTAAA", 42, 47
    )
    assert gene.window_three_prime(20, allow_undersize=True) == _window(
        "YAL002W", 1, "+", 34, 45, "AAGGCTGATCCGATC", 45, 60
    )
    with pytest.raises(
        ValueError, match=re.escape("3utr size (20) too large('+ strand 5bp outside.)")
    ):
        gene.window_three_prime(20)


def test_alias_to_systematic(genome: SCerevisiaeGenome) -> None:
    """Aliases of live, materializable genes only (YCL001W's ORPHAN is dropped)."""
    assert genome.alias_to_systematic == {
        "TFC3": ["YAL001C"],
        "SHARED": ["YAL001C", "YAL002W"],
        "FUN24": ["YAL001C"],
        "OLD1": ["YAL002W"],
    }


def test_feature_index(genome: SCerevisiaeGenome) -> None:
    """Locus features only; the ``region`` ADE1 and CDS/mRNA rows are not indexed."""
    assert genome.feature_index == {
        "genes": {"Q0010", "YAL001C", "YAL002W", "YBL001W", "YBL002W", "YCL001W"},
        "locus_type": {"TA(UGC)A": "tRNA_gene", "YBR999W": "blocked_reading_frame"},
        "standard_to_ids": {
            "TFC3": ["YAL001C"],
            "SUP1": ["tA(UGC)A"],
            "VPS8": ["YAL002W"],
            "FLO8X": ["YBR999W"],
        },
        "alias_to_ids": {
            "TFC3": ["YAL001C"],
            "SHARED": ["YAL001C", "YAL002W"],
            "FUN24": ["YAL001C"],
            "TRNAX": ["tA(UGC)A", "YBR999W"],
            "OLD1": ["YAL002W"],
            "ORPHAN": ["YCL001W"],
        },
    }


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            " yal002w ",
            GeneNameResolution(
                input_name=" yal002w ",
                status=GeneNameStatus.CURRENT,
                systematic_name="YAL002W",
            ),
        ),
        (
            "YBR999W",
            GeneNameResolution(
                input_name="YBR999W",
                status=GeneNameStatus.NON_GENE_FEATURE,
                systematic_name="YBR999W",
                feature_type="blocked_reading_frame",
                note="valid R64 blocked_reading_frame, not a gene feature",
            ),
        ),
        (
            "TFC3",
            GeneNameResolution(
                input_name="TFC3",
                status=GeneNameStatus.RENAMED,
                systematic_name="YAL001C",
                note="standard name of current gene YAL001C",
            ),
        ),
        (
            "old1",
            GeneNameResolution(
                input_name="old1",
                status=GeneNameStatus.RENAMED,
                systematic_name="YAL002W",
                note="alias of current gene YAL002W",
            ),
        ),
        (
            "SHARED",
            GeneNameResolution(
                input_name="SHARED",
                status=GeneNameStatus.AMBIGUOUS,
                systematic_name=None,
                candidates=["YAL001C", "YAL002W"],
                note="alias of multiple current genes",
            ),
        ),
        (
            "SUP1",
            GeneNameResolution(
                input_name="SUP1",
                status=GeneNameStatus.NON_GENE_FEATURE,
                systematic_name="tA(UGC)A",
                feature_type="tRNA_gene",
                note="standard name of tRNA_gene tA(UGC)A (not a gene feature)",
            ),
        ),
        (
            "TRNAX",
            GeneNameResolution(
                input_name="TRNAX",
                status=GeneNameStatus.AMBIGUOUS,
                systematic_name=None,
                candidates=["YBR999W", "tA(UGC)A"],
                note="alias of multiple non-gene loci",
            ),
        ),
        (
            "ADE1",
            GeneNameResolution(
                input_name="ADE1",
                status=GeneNameStatus.RETIRED,
                systematic_name="ADE1",
                note="not found in R64-4-1; retained as a legacy systematic name",
            ),
        ),
        (
            "yal999w",
            GeneNameResolution(
                input_name="yal999w",
                status=GeneNameStatus.RETIRED,
                systematic_name="YAL999W",
                note="not found in R64-4-1; retained as a legacy systematic name",
            ),
        ),
    ],
)
def test_resolve_gene_name(
    genome: SCerevisiaeGenome, name: str, expected: GeneNameResolution
) -> None:
    """Each resolver layer on one name; ``ADE1`` (a ``region``) never shadows a gene."""
    assert genome.resolve_gene_name(name).model_dump() == expected.model_dump()


def test_is_current_gene() -> None:
    """True for CURRENT and RENAMED, False for the other three statuses."""
    flags = {
        status: GeneNameResolution(
            input_name="x", status=status, systematic_name="X"
        ).is_current_gene
        for status in GeneNameStatus
    }
    assert flags == {
        GeneNameStatus.CURRENT: True,
        GeneNameStatus.RENAMED: True,
        GeneNameStatus.NON_GENE_FEATURE: False,
        GeneNameStatus.RETIRED: False,
        GeneNameStatus.AMBIGUOUS: False,
    }


def test_go_tables(genome: SCerevisiaeGenome) -> None:
    """GO sets from the four annotated, materializable genes; cached on second call."""
    assert genome.go == SortedSet(["GO:0000001", "GO:0000002", "GO:0000003"])
    assert genome.go is genome.go
    assert genome.go_genes == SortedDict(
        {
            "GO:0000001": SortedSet(["YAL001C", "YBL002W"]),
            "GO:0000002": SortedSet(["Q0010", "YAL002W"]),
            "GO:0000003": SortedSet(["YAL001C"]),
        }
    )
    assert genome.go_genes is genome.go_genes
    subset = SortedSet(["Q0010", "YAL001C", "YBL001W", "YCL001W"])
    assert genome.go_subset(subset) == SortedSet(
        ["GO:0000001", "GO:0000002", "GO:0000003"]
    )
    assert genome.go_subset_genes(subset) == SortedDict(
        {
            "GO:0000001": SortedSet(["YAL001C"]),
            "GO:0000002": SortedSet(["Q0010"]),
            "GO:0000003": SortedSet(["YAL001C"]),
        }
    )


def test_gene_attribute_table(genome: SCerevisiaeGenome) -> None:
    """One row per gene; only single-valued attributes kept, multi-valued dropped.

    YAL001C has three aliases and three ontology terms, so both are absent from its
    row; YAL002W's two aliases drop too; YCL001W's one alias survives.
    """
    table = genome.gene_attribute_table
    nan = float("nan")
    expected = pd.DataFrame(
        {
            "ID": ["YAL001C", "YAL002W", "YBL001W", "YBL002W", "YCL001W", "Q0010"],
            "Name": ["YAL001C", "YAL002W", "YBL001W", "YBL002W", "YCL001W", "Q0010"],
            "gene": ["TFC3", "VPS8", nan, nan, nan, nan],
            "Note": ["Subunit", nan, nan, nan, nan, nan],
            "display": ["Subunit", nan, nan, nan, nan, nan],
            "dbxref": ["SGD:S000000001", nan, nan, nan, nan, nan],
            "orf_classification": [
                "Verified",
                "Verified",
                "Dubious",
                "Verified",
                nan,
                "Verified",
            ],
            "Ontology_term": [nan, "GO:0000002", nan, "GO:0000001", nan, "GO:0000002"],
            "Alias": [nan, nan, nan, nan, "ORPHAN", nan],
        }
    )
    pd.testing.assert_frame_equal(table, expected)


def test_get_seq_references_missing_genome_id(genome: SCerevisiaeGenome) -> None:
    """Finding: s288c.py:867 passes ``self.id``, which a genome never defines."""
    with pytest.raises(
        AttributeError, match="'SCerevisiaeGenome' object has no attribute 'id'"
    ):
        genome.get_seq(1, 0, 5, "+")


def test_go_dag_loads_live_terms_only(genome: SCerevisiaeGenome) -> None:
    """GODag keeps GO:0000001 and skips the obsolete GO:0000002; loaded once."""
    dag = genome.go_dag
    assert sorted(dag.keys()) == ["GO:0000001"]
    assert genome.go_dag is dag


def test_remove_deprecated_go_terms(genome: SCerevisiaeGenome) -> None:
    """GO:0000003 (absent) and GO:0000002 (not loaded) are stripped; SO terms stay.

    YAL001C keeps ``GO:0000001`` + ``SO:0000704``; YAL002W and Q0010 lose their only
    GO term, so the attribute is deleted; YBL002W is untouched.
    """
    genome.remove_deprecated_go_terms()
    terms = {}
    for gid in ["YAL001C", "YAL002W", "Q0010", "YBL002W"]:
        gene = genome[gid]
        assert gene is not None
        terms[gid] = gene.ontology_term
    assert terms == {
        "YAL001C": ["GO:0000001", "SO:0000704"],
        "YAL002W": None,
        "Q0010": None,
        "YBL002W": ["GO:0000001"],
    }


def test_drop_chrmt(genome: SCerevisiaeGenome) -> None:
    """Q0010 leaves both the cached gene set and the database."""
    assert "Q0010" in genome.gene_set
    genome.drop_chrmt()
    assert list(genome.gene_set) == [
        "YAL001C",
        "YAL002W",
        "YBL001W",
        "YBL002W",
        "YCL001W",
    ]
    assert [f.id for f in genome.db.features_of_type("gene")] == [
        "YAL001C",
        "YAL002W",
        "YBL001W",
        "YBL002W",
        "YCL001W",
    ]


def test_drop_empty_go(genome: SCerevisiaeGenome) -> None:
    """YBL001W (no GO) is dropped; YCL001W survives because ``self[...]`` is None."""
    genome.drop_empty_go()
    assert list(genome.gene_set) == [
        "Q0010",
        "YAL001C",
        "YAL002W",
        "YBL002W",
        "YCL001W",
    ]
    assert "YBL001W" not in [f.id for f in genome.db.features_of_type("gene")]


def test_overwrite_true_builds_the_database(release: dict[str, str]) -> None:
    """With ``overwrite=True`` and no data.db, the constructor creates it."""
    genome_root = Path(release["__genome_root__"])
    genome_root.mkdir()
    genome = SCerevisiaeGenome(
        genome_root=str(genome_root), go_root=release["__go_root__"], overwrite=True
    )
    assert os.listdir(genome_root) == ["data.db"]
    assert list(genome.gene_set) == [
        "Q0010",
        "YAL001C",
        "YAL002W",
        "YBL001W",
        "YBL002W",
        "YCL001W",
    ]


def test_missing_obo_calls_download_and_go_dag_raises(
    release: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No go.obo: ``download_url`` is called with the GO URL; the DAG then fails."""
    build_db(
        release[f"saccharomyces_cerevisiae_{VERSION}.gff"],
        Path(release["__genome_root__"]),
    )
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        s288c, "download_url", lambda url, folder: calls.append((url, folder))
    )
    go_root = str(tmp_path / "empty_go")
    genome = SCerevisiaeGenome(
        genome_root=release["__genome_root__"], go_root=go_root, overwrite=False
    )
    assert calls == [("http://current.geneontology.org/ontology/go.obo", go_root)]
    assert osp.isdir(go_root)
    with pytest.raises(
        FileNotFoundError,
        match=re.escape(f"GO OBO file not found at {osp.join(go_root, 'go.obo')}"),
    ):
        genome.go_dag  # noqa: B018


def test_pickle_round_trip_drops_go_dag(genome: SCerevisiaeGenome) -> None:
    """``__reduce_ex__`` rebuilds from (genome_root, go_root, overwrite), DAG cleared."""
    genome.go_dag  # noqa: B018
    restored = pickle.loads(pickle.dumps(genome))
    assert restored._go_dag is None
    assert (restored.genome_root, restored.go_root, restored.overwrite) == (
        genome.genome_root,
        genome.go_root,
        False,
    )
    assert list(restored.gene_set) == list(genome.gene_set)
