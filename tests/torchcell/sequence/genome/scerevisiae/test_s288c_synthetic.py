# tests/torchcell/sequence/genome/scerevisiae/test_s288c_synthetic.py
# [[tests.torchcell.sequence.genome.scerevisiae.test_s288c_synthetic]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/sequence/genome/scerevisiae/test_s288c_synthetic.py
"""SCerevisiaeGenome on a hand-written release under tmp_path.

The genomes-tier ``resolve`` is stubbed at its import site
(``torchcell.sequence.genome.scerevisiae.s288c.resolve``) to map each release filename
onto a file written by the fixture, and ``data.db`` is built by the fixture with the
module's own ``build_genome_database`` (the constructor's build, source record
included), so the genome is built with ``overwrite=False``. ``go.obo`` is written too, so ``download_url`` is never reached
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

2026.09.30 (Phase 16). ``_single_gene_genome`` builds a separate genome (same FASTA,
``overwrite=False``) whose GFF holds only the rows of one gene ``G1``, so no other
feature falls inside its region. Expected values, derived from the source:

* A 1-bp CDS at the 5' end (``+`` at the gene start, ``-`` at the gene end) or a
  ``five_prime_UTR_intron`` strictly inside the gene makes the constructor use the gene
  row: ``+`` 31..45 gives ``CHR_I[30:45]`` = ``GGACTGCAATGTCTA``; ``-`` 5..16 gives
  revcomp(``CHR_II[4:16]`` = ``AATTCATGCATG``) = ``CATGCATGAATT``.
* Two CDS rows, one Verified (40..45): the Verified one, ``CHR_I[39:45]`` = ``TGTCTA``.
* A 5' intron with no CDS, or two CDS with none Verified, raises ValueError naming the
  gene; a CDS without ``orf_classification`` raises ValueError naming the gene, the CDS
  and its coordinates, so it no longer reads as "gene not found" (issue #538).
* A ``.`` strand is refused at construction with the strand and the gene id (issue #538).
* ``get_seq`` with an ``id`` supplied: ``-`` on chrI [0, 5) is revcomp(``GATTA``) =
  ``TAATC``; a FASTA-key chromosome and a ``.`` strand are refused by name before any
  slicing (issue #538).
* ``drop_chrmt`` resets the locus index and the GO-to-genes map, so Q0010 resolves as
  RETIRED and GO:0000002 maps to YAL002W only (issue #538).
* ``drop_empty_go`` resets the same two caches (issue #570): YBL001W, which carries no
  GO term, resolves as RETIRED afterwards, and ``go_genes`` is rebuilt from the
  post-drop gene set with the same three terms.
"""

import copy
import gc
import hashlib
import json
import logging
import os
import os.path as osp
import pickle
import re
import shutil
import socket
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gffutils
import pandas as pd
import pytest
from attrs import fields as attrs_fields
from gffutils.exceptions import FeatureNotFoundError
from sortedcontainers import SortedDict, SortedSet

import torchcell.sequence.genome.scerevisiae.s288c as s288c
from tests.torchcell.conftest import (
    guard_real_genome_root,
    never_migrate_a_real_genome_root,
    require_trusted_genome_database,
)
from torchcell.sequence import DnaSelectionResult, DnaWindowResult
from torchcell.sequence.data import GeneSet
from torchcell.sequence.genome.scerevisiae.s288c import (
    GeneNameResolution,
    GeneNameStatus,
    GenomeDatabaseRecord,
    GenomeDatabaseSource,
    GenomeDatabaseSourceError,
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
    source = s288c.genome_database_source(
        SCerevisiaeGenome.ASSEMBLY_SET,
        f"saccharomyces_cerevisiae_{VERSION}.gff",
        gff_path,
    )
    tmp_path = s288c.write_genome_database(gff_path, str(genome_root), source)
    os.replace(tmp_path, genome_root / "data.db")


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


def test_five_prime_plus_strand_ends_before_the_first_cds_base(
    genome: SCerevisiaeGenome,
) -> None:
    """On ``+`` the 5' window is [33 - w, 33): it stops before the gene (issue #543).

    YAL002W's first CDS base is 0-based 33 (1-based 34), so the 5-nt window is
    CHR_I[28:33] = TAGGA and excludes CHR_I[33] = C, matching the minus strand, which
    also excludes the gene's own bases. With the start codon it is [31, 36), ending
    with the 3 codon bases. Window 40 starts at 33 - 40 = -7.
    """
    gene = genome["YAL002W"]
    assert gene is not None
    assert gene.seq[0] == CHR_I[33] == "C"
    assert CHR_I[28:33] == "TAGGA"
    assert gene.window_five_prime(5) == _window(
        "YAL002W", 1, "+", 34, 45, "TAGGA", 28, 33
    )
    assert gene.window_five_prime(5, include_start_codon=True) == _window(
        "YAL002W", 1, "+", 34, 45, "GACTG", 31, 36
    )
    assert gene.window_five_prime(40, allow_undersize=True) == _window(
        "YAL002W", 1, "+", 34, 45, CHR_I[0:33], 0, 33
    )
    with pytest.raises(
        ValueError,
        match=re.escape("five prime size (40) too large ('+ strand 7bp outside.)"),
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


# --------------------------------------------------------------------------- #
# 2026.09.30 (Phase 16): the CDS-selection branches on one-gene genomes, the strand
# refusals, get_seq, the obsolete-term branch, and the caches.
# --------------------------------------------------------------------------- #

GffRow = tuple[str, str, int, int, str, str]


def _single_gene_genome(
    root: Path, monkeypatch: pytest.MonkeyPatch, rows: list[GffRow]
) -> SCerevisiaeGenome:
    """A genome over the fixture FASTA whose GFF is exactly ``rows``."""
    release = root / "release"
    release.mkdir(parents=True)
    lines = ["##gff-version 3"]
    for seqid, ftype, start, end, strand, attrs in rows:
        lines.append(
            "\t".join([seqid, "SGD", ftype, str(start), str(end), ".", strand, "."])
            + "\t"
            + attrs
        )
    texts = {
        f"S288C_reference_sequence_{VERSION}.fsa": FASTA_DNA,
        f"saccharomyces_cerevisiae_{VERSION}.gff": "\n".join(lines) + "\n",
        f"orf_trans_all_{VERSION}.fasta": FASTA_PROTEIN,
        f"orf_coding_all_{VERSION}.fasta": FASTA_CDS,
    }
    paths: dict[str, str] = {}
    for name, text in texts.items():
        (release / name).write_text(text)
        paths[name] = str(release / name)
    monkeypatch.setattr(
        s288c, "resolve", lambda assembly_set, filename: paths[filename]
    )
    (root / "go").mkdir()
    (root / "go" / "go.obo").write_text(GO_OBO)
    build_db(paths[f"saccharomyces_cerevisiae_{VERSION}.gff"], root / "genome")
    return SCerevisiaeGenome(
        genome_root=str(root / "genome"), go_root=str(root / "go"), overwrite=False
    )


_GENE_PLUS: GffRow = ("chrI", "gene", 31, 45, "+", "ID=G1;Name=G1")
_INTRON_START: GffRow = (
    "chrI",
    "five_prime_UTR_intron",
    31,
    33,
    "+",
    "ID=G1_i;Parent=G1",
)


def _cds(start: int, end: int, cls: str | None, strand: str = "+") -> GffRow:
    tail = "" if cls is None else f";orf_classification={cls}"
    return ("chrI", "CDS", start, end, strand, f"ID=G1_c{start};Parent=G1{tail}")


def test_one_bp_five_prime_cds_on_plus_keeps_the_gene_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``+`` 31..45 with a 5' intron and a 1-bp CDS at 31: the gene row, 31..45."""
    genome = _single_gene_genome(
        tmp_path,
        monkeypatch,
        [_GENE_PLUS, _INTRON_START, _cds(31, 31, "Verified"), _cds(34, 45, "Verified")],
    )
    gene = genome["G1"]
    assert gene is not None
    assert (gene.start, gene.end, gene.strand, gene.seq) == (
        31,
        45,
        "+",
        "GGACTGCAATGTCTA",
    )
    assert CHR_I[30:45] == gene.seq


def test_one_bp_five_prime_cds_on_minus_keeps_the_gene_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``-`` 5..16 with a 1-bp CDS at the gene end (its 5' end): the gene row."""
    rows: list[GffRow] = [
        ("chrII", "gene", 5, 16, "-", "ID=G1;Name=G1"),
        ("chrII", "five_prime_UTR_intron", 14, 16, "-", "ID=G1_i;Parent=G1"),
        ("chrII", "CDS", 16, 16, "-", "ID=G1_c0;Parent=G1;orf_classification=Verified"),
        ("chrII", "CDS", 5, 13, "-", "ID=G1_c1;Parent=G1;orf_classification=Verified"),
    ]
    gene = _single_gene_genome(tmp_path, monkeypatch, rows)["G1"]
    assert gene is not None
    assert (gene.chromosome, gene.start, gene.end, gene.seq) == (
        2,
        5,
        16,
        "CATGCATGAATT",
    )
    assert CHR_II[4:16] == "AATTCATGCATG"


def test_middle_five_prime_intron_keeps_the_gene_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An intron at 35..37, strictly inside 31..45, is not a 5' UTR intron: gene row."""
    genome = _single_gene_genome(
        tmp_path,
        monkeypatch,
        [
            _GENE_PLUS,
            ("chrI", "five_prime_UTR_intron", 35, 37, "+", "ID=G1_i;Parent=G1"),
            _cds(38, 45, "Verified"),
        ],
    )
    gene = genome["G1"]
    assert gene is not None
    assert (gene.start, gene.end, gene.seq) == (31, 45, "GGACTGCAATGTCTA")


def test_two_cds_one_verified_selects_the_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Dubious 34..39 and a Verified 40..45: the gene is 40..45, CHR_I[39:45]."""
    genome = _single_gene_genome(
        tmp_path,
        monkeypatch,
        [_GENE_PLUS, _INTRON_START, _cds(34, 39, "Dubious"), _cds(40, 45, "Verified")],
    )
    gene = genome["G1"]
    assert gene is not None
    assert (gene.start, gene.end, gene.seq) == (40, 45, "TGTCTA")


@pytest.mark.parametrize(
    "extra",
    [[], [_cds(34, 39, "Dubious"), _cds(40, 45, "Dubious")]],
    ids=["no_cds", "no_verified_cds"],
)
def test_five_prime_intron_without_a_usable_cds_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: list[GffRow]
) -> None:
    """No CDS, or several with none Verified, is refused with a ValueError naming G1.

    These used to leave ``feature`` unbound (UnboundLocalError, issue #538). Refusing
    rather than falling back to the gene row keeps the 5' UTR intron out of the
    sequence; no R64-4-1 gene takes either branch (checked on the served data.db).
    """
    genome = _single_gene_genome(
        tmp_path, monkeypatch, [_GENE_PLUS, _INTRON_START, *extra]
    )
    message = (
        "Gene G1 has a five_prime_UTR_intron but no CDS feature"
        if not extra
        else "Gene G1 has a five_prime_UTR_intron and 2 CDS features, none Verified"
    )
    with pytest.raises(ValueError) as refused:
        genome["G1"]
    assert str(refused.value) == message


def test_cds_without_orf_classification_names_the_cds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A CDS missing ``orf_classification`` raises ValueError naming gene, CDS and span.

    It used to be a KeyError that ``__getitem__`` turned into None plus the "only
    systematic names" print, so a malformed CDS read as an unknown gene id (issue #538).
    Nothing is printed now.
    """
    genome = _single_gene_genome(
        tmp_path,
        monkeypatch,
        [_GENE_PLUS, _INTRON_START, _cds(34, 39, None), _cds(40, 45, None)],
    )
    with pytest.raises(ValueError) as refused:
        genome["G1"]
    assert str(refused.value) == (
        "Gene G1: CDS G1_c34 at 34..39 has no orf_classification attribute"
    )
    assert capsys.readouterr().out == ""


def test_unstranded_gene_is_refused_at_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``.`` strand is refused when the gene is built, naming the strand and the id.

    It used to build a gene with ``seq`` None whose every window raised
    UnboundLocalError later (issue #538). No R64-4-1 gene is unstranded.
    """
    genome = _single_gene_genome(
        tmp_path, monkeypatch, [("chrI", "gene", 31, 45, ".", "ID=G1;Name=G1")]
    )
    with pytest.raises(ValueError) as refused:
        genome["G1"]
    assert str(refused.value) == (
        "Gene G1 has strand '.'; a gene needs '+' or '-' to orient its sequence "
        "and windows"
    )


def test_get_seq_with_an_id_reverse_complements_and_refuses_a_named_chromosome(
    genome: SCerevisiaeGenome,
) -> None:
    """With an ``id`` supplied (the pinned AttributeError is otherwise first):

    ``-`` on chrI [0, 5) is revcomp(GATTA) = TAATC; ``+`` on chrII [5, 9) is ATTC. A
    FASTA key such as ``ref|NC_001133|``, or a chromosome number the genome lacks, is
    refused by name before slicing, as is a ``.`` strand (both used to fail later, in
    ``DnaSelectionResult`` validation or on an unbound ``seq``; issue #538).
    """
    vars(genome)["id"] = "S288C"
    assert genome.get_seq(1, 0, 5, "-") == DnaSelectionResult(
        id="S288C", chromosome=1, strand="-", start=0, end=5, seq="TAATC"
    )
    assert genome.get_seq(2, 5, 9, "+").seq == CHR_II[5:9] == "ATTC"
    with pytest.raises(ValueError) as fasta_key:
        genome.get_seq("ref|NC_001133|", 0, 5, "+")
    assert str(fasta_key.value) == (
        "Chromosome must be one of the chromosome numbers [0, 1, 2], "
        "got 'ref|NC_001133|'"
    )
    with pytest.raises(ValueError) as absent:
        genome.get_seq(3, 0, 5, "+")
    assert str(absent.value) == (
        "Chromosome must be one of the chromosome numbers [0, 1, 2], got 3"
    )
    with pytest.raises(ValueError) as dot:
        genome.get_seq(1, 0, 5, ".")
    assert str(dot.value) == "Strand must be '+' or '-', got '.'"


def test_remove_deprecated_go_terms_drops_an_obsolete_term(
    genome: SCerevisiaeGenome,
) -> None:
    """With a DAG that marks GO:0000003 obsolete, YAL001C keeps only GO:0000001 + SO.

    The real GODag never loads obsolete terms, so a two-term stand-in reaches the
    ``is_obsolete`` branch; GO:0000002 (absent from it) is still stripped.
    """
    stand_in: Any = {
        "GO:0000001": SimpleNamespace(is_obsolete=False),
        "GO:0000003": SimpleNamespace(is_obsolete=True),
    }
    genome._go_dag = stand_in
    genome.remove_deprecated_go_terms()
    kept = {}
    for gid in ["YAL001C", "YAL002W", "YBL002W"]:
        gene = genome[gid]
        assert gene is not None
        kept[gid] = gene.ontology_term
    assert kept == {
        "YAL001C": ["GO:0000001", "SO:0000704"],
        "YAL002W": None,
        "YBL002W": ["GO:0000001"],
    }


def test_alias_map_is_computed_once(
    genome: SCerevisiaeGenome, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first access materializes all six genes; the second calls ``[]`` zero times."""
    seen: list[str] = []
    original = SCerevisiaeGenome.__getitem__

    def counting(self: SCerevisiaeGenome, item: str) -> s288c.SCerevisiaeGene | None:
        seen.append(item)
        return original(self, item)

    monkeypatch.setattr(SCerevisiaeGenome, "__getitem__", counting)
    first = genome.alias_to_systematic
    assert len(seen) == 6
    assert genome.alias_to_systematic == first
    assert len(seen) == 6


def test_drop_chrmt_rebuilds_the_locus_index_and_go_genes(
    genome: SCerevisiaeGenome,
) -> None:
    """After ``drop_chrmt`` the warm caches are rebuilt without Q0010.

    Both caches are populated first. They used to survive the drop, so Q0010 resolved
    as CURRENT and GO:0000002 still mapped to it (issue #538). Now Q0010 resolves as
    RETIRED (absent from the annotation) and GO:0000002 maps to YAL002W only.
    """
    assert genome.resolve_gene_name("Q0010").status is GeneNameStatus.CURRENT
    assert list(genome.go_genes["GO:0000002"]) == ["Q0010", "YAL002W"]
    genome.drop_chrmt()
    assert "Q0010" not in genome.gene_set
    resolution = genome.resolve_gene_name("Q0010")
    assert (resolution.status, resolution.systematic_name) == (
        GeneNameStatus.RETIRED,
        "Q0010",
    )
    assert "Q0010" not in genome.feature_index["genes"]
    assert list(genome.go_genes["GO:0000002"]) == ["YAL002W"]


def test_drop_empty_go_rebuilds_the_locus_index_and_go_genes(
    genome: SCerevisiaeGenome,
) -> None:
    """After ``drop_empty_go`` the warm caches are rebuilt without YBL001W.

    Both caches are populated first. YBL001W has no GO term, so it never appears in
    ``go_genes``; the stale cache it left was the locus index, under which YBL001W
    still resolved as CURRENT after the drop (issue #570). Now it resolves as RETIRED,
    the index lists the five surviving genes, and ``go_genes`` is a fresh map, built
    from the post-drop gene set, with the exact three terms and no YBL001W.
    """
    assert genome.resolve_gene_name("YBL001W").status is GeneNameStatus.CURRENT
    warm_go_genes = genome.go_genes
    genome.drop_empty_go()
    resolution = genome.resolve_gene_name("YBL001W")
    assert (resolution.status, resolution.systematic_name) == (
        GeneNameStatus.RETIRED,
        "YBL001W",
    )
    assert genome.feature_index["genes"] == {
        "Q0010",
        "YAL001C",
        "YAL002W",
        "YBL002W",
        "YCL001W",
    }
    assert genome.go_genes is not warm_go_genes
    assert genome.go_genes == SortedDict(
        {
            "GO:0000001": SortedSet(["YAL001C", "YBL002W"]),
            "GO:0000002": SortedSet(["Q0010", "YAL002W"]),
            "GO:0000003": SortedSet(["YAL001C"]),
        }
    )


def test_drop_chrmt_before_the_gene_set_is_cached(genome: SCerevisiaeGenome) -> None:
    """With no cached gene set, the drop only deletes rows; the set computed afterwards
    comes from the database and already lacks Q0010.
    """
    genome.drop_chrmt()
    assert list(genome.gene_set) == [
        "YAL001C",
        "YAL002W",
        "YBL001W",
        "YBL002W",
        "YCL001W",
    ]


def test_go_subset_genes_merges_genes_sharing_a_term(genome: SCerevisiaeGenome) -> None:
    """YAL001C and YBL002W both carry GO:0000001; only YAL001C carries GO:0000003."""
    assert genome.go_subset_genes(SortedSet(["YAL001C", "YBL002W"])) == SortedDict(
        {
            "GO:0000001": SortedSet(["YAL001C", "YBL002W"]),
            "GO:0000003": SortedSet(["YAL001C"]),
        }
    )


class _RecordingGenome:
    """Stands in for the genome class in ``main``: records the constructor kwargs."""

    kwargs: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        _RecordingGenome.kwargs.append(kwargs)
        self.gene_set = GeneSet(["YAL001C", "YAL002W"])


def test_main_builds_under_data_root_with_overwrite_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main`` reuses the existing ``data.db`` (``overwrite=False``) under DATA_ROOT.

    The genome and GO roots are ``$DATA_ROOT/data/sgd/genome`` and ``$DATA_ROOT/data/go``;
    the repo ``.env`` is not read (``load_dotenv`` stubbed). ``overwrite=True`` used to
    rebuild the shared database every run (issue #538; memory:
    genome-overwrite-true-rebuild-race).
    """
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    _RecordingGenome.kwargs = []
    monkeypatch.setattr(s288c, "SCerevisiaeGenome", _RecordingGenome)
    s288c.main()
    assert _RecordingGenome.kwargs == [
        {
            "genome_root": f"{tmp_path}/data/sgd/genome",
            "go_root": f"{tmp_path}/data/go",
            "overwrite": False,
        }
    ]
    assert capsys.readouterr().out == (
        "genome.gene_set: GeneSet(size=2, items=['YAL001C', 'YAL002W'])\n\n"
    )
    assert os.listdir(tmp_path) == []


# --------------------------------------------------------------------------- #
# 2026.10.01: overwrite defaults to False; data.db is recorded, verified at open,
# migrated when untrusted (keeping at most one file), and never written after it is
# built; writes go to a private, swept, replayable copy.
# --------------------------------------------------------------------------- #

GFF_NAME = f"saccharomyces_cerevisiae_{VERSION}.gff"
HOST = socket.gethostname()
#: Rows per featuretype of the fixture GFF (one feature per row, see GFF_ROWS) and the
#: relations gffutils derives from its Parent chains (direct and grandparent links).
FIXTURE_COUNTS = {
    "CDS": 4,
    "blocked_reading_frame": 1,
    "five_prime_UTR_intron": 2,
    "gene": 6,
    "mRNA": 1,
    "region": 1,
    "tRNA_gene": 1,
}
FIXTURE_RELATIONS = 8
ALL_GENES = ["Q0010", "YAL001C", "YAL002W", "YBL001W", "YBL002W", "YCL001W"]
NO_CHRMT = ["YAL001C", "YAL002W", "YBL001W", "YBL002W", "YCL001W"]


def _expected_source(release: dict[str, str]) -> GenomeDatabaseSource:
    """The source a build from the fixture GFF records, sha256 computed here."""
    return GenomeDatabaseSource(
        assembly_set="sgd_S288C_R64-4-1_20230830",
        gff_filename=GFF_NAME,
        gff_sha256=hashlib.sha256(Path(release[GFF_NAME]).read_bytes()).hexdigest(),
        keep_order=True,
        merge_strategy="merge",
        sort_attribute_values=True,
    )


def _header_counter(path: Path) -> int:
    return int.from_bytes(path.read_bytes()[24:28], "big")


def _assert_recorded(release: dict[str, str], db_path: Path) -> None:
    """The record names the fixture source, its exact counts, and the file's own
    change counter (the record transaction was the last write).
    """
    assert s288c.read_genome_database_record(str(db_path)) == GenomeDatabaseRecord(
        source=_expected_source(release),
        featuretype_counts=FIXTURE_COUNTS,
        relations_count=FIXTURE_RELATIONS,
        change_counter=_header_counter(db_path),
        version=1,
    )


def _identity(path: Path) -> tuple[int, int]:
    st = path.stat()
    return (st.st_ino, st.st_mtime_ns)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rebuild_call(release: dict[str, str]) -> str:
    return (
        f"SCerevisiaeGenome(genome_root={release['__genome_root__']!r}, "
        f"go_root={release['__go_root__']!r}, overwrite=True)"
    )


def _root(release: dict[str, str]) -> Path:
    root = Path(release["__genome_root__"])
    root.mkdir(exist_ok=True)
    return root


def _old_code_rebuild(release: dict[str, str]) -> Path:
    """What main's default ``overwrite=True`` did: create_db(force=True) in place."""
    db_path = _root(release) / "data.db"
    gffutils.create_db(
        release[GFF_NAME], dbfn=str(db_path), force=True, **s288c.CREATE_DB_KWARGS
    )
    return db_path


def _old_code_delete(db_path: Path, feature_id: str) -> None:
    """What main's drop_chrmt did to the shared file: a delete in place."""
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM features WHERE id = ?", (feature_id,))
    conn.commit()
    conn.close()


def _construct(release: dict[str, str], **kwargs: Any) -> SCerevisiaeGenome:
    return SCerevisiaeGenome(
        genome_root=release["__genome_root__"], go_root=release["__go_root__"], **kwargs
    )


def _dead_pid() -> int:
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


@pytest.fixture(autouse=True)
def private_tmp(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Every test in this module gets its own temp dir for private copies
    (``tempfile.gettempdir()``), so no test writes to or sweeps the real one.
    """
    d = tmp_path_factory.mktemp("private_tmp")
    monkeypatch.setattr(tempfile, "tempdir", str(d))
    return d


def test_overwrite_defaults_to_false() -> None:
    """Omitting ``overwrite`` must never rebuild the shared database."""
    assert attrs_fields(SCerevisiaeGenome).overwrite.default is False


def test_default_builds_an_absent_database_once_then_reuses_it(
    release: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing root without data.db: the first default construction builds it in
    a temporary file inside the root (one create_db call, the record inside, mode
    0644, no temporary left); the second opens the same file.
    """
    create_calls: list[tuple[str, str]] = []
    real_create_db = gffutils.create_db

    def recording_create_db(data: str, dbfn: str, **kwargs: Any) -> Any:
        create_calls.append((data, osp.dirname(dbfn)))
        return real_create_db(data, dbfn=dbfn, **kwargs)

    monkeypatch.setattr(gffutils, "create_db", recording_create_db)
    genome_root = _root(release)
    db_path = genome_root / "data.db"

    first = _construct(release)
    assert first.overwrite is False
    assert create_calls == [(release[GFF_NAME], str(genome_root))]
    assert os.listdir(genome_root) == ["data.db"]
    assert db_path.stat().st_mode & 0o777 == 0o644
    _assert_recorded(release, db_path)
    built = _identity(db_path)

    second = _construct(release)
    assert len(create_calls) == 1
    assert _identity(db_path) == built
    assert list(second.gene_set) == ALL_GENES


def test_missing_genome_root_is_refused_unless_asked_to_build(
    release: dict[str, str],
) -> None:
    """A genome_root that does not exist (a wrong or relative path) is refused by
    name and nothing is created; ``overwrite=True`` creates it and builds.
    """
    genome_root = Path(release["__genome_root__"])
    with pytest.raises(s288c.GenomeRootNotFoundError) as exc:
        _construct(release)
    assert str(exc.value) == (
        f"genome_root {str(genome_root)!r} does not exist. Pass the existing genome "
        "cache directory, or build a new one deliberately: "
        f"{_rebuild_call(release)}"
    )
    assert not genome_root.exists()
    genome = _construct(release, overwrite=True)
    assert os.listdir(genome_root) == ["data.db"]
    assert list(genome.gene_set) == ALL_GENES


def test_overwrite_true_rebuilds_atomically(release: dict[str, str]) -> None:
    """An explicit ``overwrite=True`` renames a fresh build onto data.db: a new inode,
    while a connection opened on the old file still reads it.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    db_path = Path(release["__genome_root__"]) / "data.db"
    before = db_path.stat().st_ino
    old_reader = sqlite3.connect(db_path)
    _construct(release, overwrite=True)
    assert db_path.stat().st_ino != before
    assert old_reader.execute("SELECT COUNT(*) FROM features").fetchone() == (16,)
    old_reader.close()
    assert os.listdir(release["__genome_root__"]) == ["data.db"]
    _assert_recorded(release, db_path)


def test_legacy_database_with_fresh_rows_is_replaced_and_nothing_kept(
    release: dict[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    """A record-less data.db whose rows equal a fresh build (every pre-change build,
    every old-code rebuild) is replaced and not kept, with one WARNING; an open
    reader keeps the old inode; the next construction opens the migrated file.
    """
    db_path = _old_code_rebuild(release)
    old_reader = sqlite3.connect(db_path)
    with caplog.at_level(logging.WARNING, logger=s288c.__name__):
        genome = _construct(release)
    assert os.listdir(release["__genome_root__"]) == ["data.db"]
    assert old_reader.execute("SELECT COUNT(*) FROM features").fetchone() == (16,)
    old_reader.close()
    _assert_recorded(release, db_path)
    assert [(r.levelname, r.getMessage()) for r in caplog.records] == [
        (
            "WARNING",
            f"genome database {db_path} was not trusted (it carries no "
            "torchcell_genome_db_source record); its rows equal a fresh build, so it "
            "was replaced by the recorded build and nothing was kept",
        )
    ]
    assert list(genome.gene_set) == ALL_GENES
    migrated = _identity(db_path)
    _construct(release)
    assert _identity(db_path) == migrated


def test_rows_deleted_in_place_are_kept_once_and_migrated(
    release: dict[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    """Old code deleted a row in place: the counts differ from the record, so the file
    is copied to data.db.untrusted and replaced, with one WARNING naming the case.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    db_path = Path(release["__genome_root__"]) / "data.db"
    _old_code_delete(db_path, "Q0010")
    damaged = _sha(db_path)
    with caplog.at_level(logging.WARNING, logger=s288c.__name__):
        genome = _construct(release)
    kept = Path(release["__genome_root__"]) / "data.db.untrusted"
    assert sorted(os.listdir(release["__genome_root__"])) == [
        "data.db",
        "data.db.untrusted",
    ]
    assert _sha(kept) == damaged
    assert [r.getMessage() for r in caplog.records] == [
        f"genome database {db_path} was not trusted (its row counts differ from its "
        "record (features 15 vs 16 recorded, relations 8 vs 8 recorded)); its rows "
        f"differ from a fresh build, so it was kept as {kept} (replacing any earlier "
        "one) and replaced by the recorded build"
    ]
    assert list(genome.gene_set) == ALL_GENES


def test_count_preserving_rewrite_in_place_is_detected_by_the_change_counter(
    release: dict[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    """Old code rewrote an attribute in place (every count unchanged): the sqlite
    change counter moved past the recorded one, so the file is migrated.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    db_path = Path(release["__genome_root__"]) / "data.db"
    recorded = _header_counter(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE features SET attributes = '{}' WHERE id = 'YAL002W'")
    conn.commit()
    conn.close()
    with caplog.at_level(logging.WARNING, logger=s288c.__name__):
        genome = _construct(release)
    assert (
        caplog.records[0]
        .getMessage()
        .startswith(
            f"genome database {db_path} was not trusted (it was written in place after "
            f"its build (sqlite change counter {recorded + 1}, {recorded} recorded)); its "
            "rows differ from a fresh build"
        )
    )
    assert genome.db["YAL002W"].attributes["Ontology_term"] == ["GO:0000002"]


@pytest.mark.parametrize(
    ("damage", "kept"), [(False, []), (True, ["data.db.untrusted"])]
)
def test_old_code_ping_pong_never_accumulates_kept_files(
    release: dict[str, str], damage: bool, kept: list[str]
) -> None:
    """Eight cycles of an old-code rebuild in place (optionally followed by an old-code
    delete) and a new-code open: nothing is kept when the rows equal a fresh build, and
    at most one file when they differ.
    """
    for _ in range(8):
        db_path = _old_code_rebuild(release)
        if damage:
            _old_code_delete(db_path, "Q0010")
        genome = _construct(release)
        assert list(genome.gene_set) == ALL_GENES
    assert sorted(os.listdir(release["__genome_root__"])) == ["data.db", *kept]


def test_database_built_from_a_different_gff_is_refused(
    release: dict[str, str],
) -> None:
    """The pinned GFF changed after data.db was built: a real source change, refused
    by name with the deliberate rebuild call, and the file is left alone.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    db_path = Path(release["__genome_root__"]) / "data.db"
    before = _identity(db_path)
    stale = _expected_source(release)
    with open(release[GFF_NAME], "a") as fh:
        fh.write("chrI\tSGD\tregion\t50\t55\t.\t+\t.\tID=EXTRA\n")
    current = _expected_source(release)
    assert stale.gff_sha256 != current.gff_sha256
    message = (
        f"{db_path} was built from {stale.model_dump()} but this genome's source is "
        f"{current.model_dump()}. Rebuild it deliberately, once, while no job reads "
        f"it: {_rebuild_call(release)}"
    )
    with pytest.raises(GenomeDatabaseSourceError) as exc:
        _construct(release)
    assert str(exc.value) == message
    assert _identity(db_path) == before
    assert os.listdir(release["__genome_root__"]) == ["data.db"]


def test_source_table_with_two_rows_is_refused(release: dict[str, str]) -> None:
    """The record table must hold exactly one row; two is a defect, named."""
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    db_path = Path(release["__genome_root__"]) / "data.db"
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO torchcell_genome_db_source (record) VALUES ('{}')")
    conn.commit()
    conn.close()
    with pytest.raises(GenomeDatabaseSourceError) as exc:
        s288c.read_genome_database_record(str(db_path))
    assert str(exc.value) == (
        f"{db_path}: torchcell_genome_db_source holds 2 rows, expected exactly 1"
    )


@pytest.mark.parametrize(
    ("damage", "kept"), [(False, []), (True, ["data.db.untrusted"])]
)
def test_interleaved_migrations_end_on_one_database(
    release: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    damage: bool,
    kept: list[str],
) -> None:
    """Process B migrates completely while process A is between its build and its
    install: A then finds data.db trusted and discards its own build. One installed
    database, at most one kept file, no temporaries, one WARNING.
    """
    db_path = _old_code_rebuild(release)
    if damage:
        _old_code_delete(db_path, "Q0010")
    real_write = s288c.write_genome_database
    calls: list[str] = []

    def interleaving_write(gff: str, db_dir: str, source: GenomeDatabaseSource) -> str:
        tmp = real_write(gff, db_dir, source)
        calls.append(tmp)
        if len(calls) == 1:  # A built; B runs its whole migration now
            reason = s288c.untrusted_reason(str(db_path), source, "call")
            assert reason is not None
            s288c.migrate_genome_database(gff, str(db_path), source, "call", reason)
        return tmp

    monkeypatch.setattr(s288c, "write_genome_database", interleaving_write)
    with caplog.at_level(logging.WARNING, logger=s288c.__name__):
        genome = _construct(release)
    assert len(calls) == 2 and calls[0] != calls[1]
    assert sorted(os.listdir(release["__genome_root__"])) == ["data.db", *kept]
    assert len(caplog.records) == 1
    _assert_recorded(release, db_path)
    assert list(genome.gene_set) == ALL_GENES


def test_read_only_root_refuses_an_untrusted_database_by_name(
    release: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A legacy data.db in a directory this process cannot write: a named refusal
    stating why it must be rebuilt and how; no build is attempted and the file is
    never opened by gffutils.
    """
    db_path = _old_code_rebuild(release)
    root = Path(release["__genome_root__"])
    opened: list[str] = []
    monkeypatch.setattr(s288c, "GffutilsConnectionManager", lambda p: opened.append(p))
    root.chmod(0o555)
    try:
        with pytest.raises(s288c.GenomeRootNotWritableError) as exc:
            _construct(release)
    finally:
        root.chmod(0o755)
    assert str(exc.value) == (
        f"{db_path} must be built (it carries no torchcell_genome_db_source record), "
        f"but {root} is not writable by this process, so the database is not opened "
        f"or built here. Construct SCerevisiaeGenome(genome_root={str(root)!r}, "
        f"go_root={release['__go_root__']!r}) once from a process that can write "
        "that directory."
    )
    assert opened == []
    assert os.listdir(root) == ["data.db"]


def test_read_only_root_refuses_a_missing_database_by_name(
    release: dict[str, str],
) -> None:
    """No data.db in a directory this process cannot write: refused by name."""
    root = _root(release)
    root.chmod(0o555)
    try:
        with pytest.raises(s288c.GenomeRootNotWritableError) as exc:
            _construct(release)
    finally:
        root.chmod(0o755)
    assert str(exc.value).startswith(
        f"{root / 'data.db'} must be built (it does not exist), but {root} is not "
        "writable"
    )


def test_trusted_database_in_a_read_only_root_still_works(
    release: dict[str, str], private_tmp: Path
) -> None:
    """A recorded data.db in a read-only directory opens and drops still work (the
    private copy lives in the temp dir).
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    root = Path(release["__genome_root__"])
    root.chmod(0o555)
    try:
        genome = _construct(release)
        genome.drop_chrmt()
        assert list(genome.gene_set) == NO_CHRMT
    finally:
        root.chmod(0o755)
    assert os.listdir(root) == ["data.db"]


def test_failed_build_leaves_no_temporary(
    release: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """create_db dies after creating the temporary file: it is removed and the error
    propagates.
    """
    root = _root(release)

    def dying_create_db(data: str, dbfn: str, **kwargs: Any) -> None:
        Path(dbfn).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(gffutils, "create_db", dying_create_db)
    with pytest.raises(OSError, match="^disk full$"):
        _construct(release)
    assert os.listdir(root) == []


def test_failure_between_build_and_install_leaves_no_temporary(
    release: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The migration fails after its build (here the content digest): the build is
    removed, and the untrusted data.db is left as it was.
    """
    db_path = _old_code_rebuild(release)
    before = _sha(db_path)

    def failing_digest(path: str) -> str:
        raise OSError("read error")

    monkeypatch.setattr(s288c, "database_content_digest", failing_digest)
    with pytest.raises(OSError, match="^read error$"):
        _construct(release)
    assert os.listdir(release["__genome_root__"]) == ["data.db"]
    assert _sha(db_path) == before


def test_construction_sweeps_dead_build_temporaries(release: dict[str, str]) -> None:
    """Build temporaries of a dead pid on this host are removed at construction; a
    live pid's and another host's are left, and so is any file that is not a build
    temporary (the kept data.db.untrusted, or a dead-pid-like name without the
    .building suffix).
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    root = Path(release["__genome_root__"])
    dead = f"data.db.{HOST}.{_dead_pid()}.abc123.building"
    dead_copy = f"data.db.untrusted.{HOST}.{_dead_pid()}.x_9.building"
    live = f"data.db.{HOST}.{os.getpid()}.def456.building"
    other = f"data.db.otherhost.{_dead_pid()}.ghi789.building"
    kept = "data.db.untrusted"
    not_temp = f"data.db.{HOST}.{_dead_pid()}.jkl012.db"
    no_suffix = f"data.db.{HOST}.{_dead_pid()}.mno345"
    for name in (dead, dead_copy, live, other, kept, not_temp, no_suffix):
        (root / name).write_bytes(b"x")
    _construct(release)
    assert sorted(os.listdir(root)) == sorted(
        ["data.db", live, other, kept, not_temp, no_suffix]
    )


def test_private_copy_creation_sweeps_dead_copies(
    release: dict[str, str], private_tmp: Path
) -> None:
    """A private copy is named torchcell-genome-<host>-<pid>-<random>.db; creating one
    removes this host's copies of dead pids and leaves a live pid's and another
    host's.
    """
    dead = f"torchcell-genome-{HOST}-{_dead_pid()}-abc123.db"
    live = f"torchcell-genome-{HOST}-{os.getpid()}-def456.db"
    other = f"torchcell-genome-otherhost-{_dead_pid()}-ghi789.db"
    for name in (dead, live, other):
        (private_tmp / name).write_bytes(b"x")
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    genome = _construct(release)
    genome.drop_chrmt()
    path = genome._private_db_path
    assert path is not None
    assert re.fullmatch(
        rf"torchcell-genome-{re.escape(HOST)}-{os.getpid()}-[a-z0-9_]+\.db",
        osp.basename(path),
    )
    assert sorted(os.listdir(private_tmp)) == sorted([live, other, osp.basename(path)])


def test_unpickling_an_overwrite_true_genome_does_not_rebuild(
    release: dict[str, str],
) -> None:
    """A genome built with ``overwrite=True`` and sent to a worker by pickle opens the
    parent's database (same inode and mtime); ``overwrite`` itself round-trips.
    """
    genome = _construct(release, overwrite=True)
    db_path = Path(release["__genome_root__"]) / "data.db"
    built = _identity(db_path)
    restored = pickle.loads(pickle.dumps(genome))
    assert restored.overwrite is True
    assert _identity(db_path) == built
    assert list(restored.gene_set) == ALL_GENES


def test_drops_never_write_the_shared_database(
    release: dict[str, str], private_tmp: Path
) -> None:
    """``drop_chrmt``, ``drop_empty_go`` and ``remove_deprecated_go_terms`` act on this
    instance only: the shared data.db keeps its bytes, inode and mtime, no .bak is
    written anywhere, and a second genome on the same root built after the drops sees
    every feature. The private copy is removed with the instance.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    genome = _construct(release)
    db_path = Path(release["__genome_root__"]) / "data.db"
    before = (_identity(db_path), _sha(db_path))
    genome.drop_chrmt()
    genome.drop_empty_go()
    genome.remove_deprecated_go_terms()
    assert list(genome.gene_set) == ["YAL001C", "YAL002W", "YBL002W", "YCL001W"]
    assert [f.id for f in genome.db.features_of_type("gene")] == [
        "YAL001C",
        "YAL002W",
        "YBL002W",
        "YCL001W",
    ]
    assert (_identity(db_path), _sha(db_path)) == before
    assert os.listdir(release["__genome_root__"]) == ["data.db"]
    private = genome._private_db_path
    assert private is not None
    assert os.listdir(private_tmp) == [osp.basename(private)]

    second = _construct(release)
    assert list(second.gene_set) == ALL_GENES
    assert sorted(f.id for f in second.db.features_of_type("gene")) == ALL_GENES
    assert second.db["YAL002W"].attributes["Ontology_term"] == ["GO:0000002"]
    assert second["Q0010"] is not None

    del genome
    gc.collect()
    assert os.listdir(private_tmp) == []


def test_remove_deprecated_go_terms_as_the_first_write_stays_private(
    release: dict[str, str], private_tmp: Path
) -> None:
    """``remove_deprecated_go_terms`` before any drop: the shared file is untouched,
    the private copy drops the obsolete GO:0000002 from YAL002W.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    genome = _construct(release)
    db_path = Path(release["__genome_root__"]) / "data.db"
    before = (_identity(db_path), _sha(db_path))
    genome.remove_deprecated_go_terms()
    assert (_identity(db_path), _sha(db_path)) == before
    assert genome._private_db_path is not None
    assert "Ontology_term" not in genome.db["YAL002W"].attributes
    assert os.listdir(private_tmp) == [osp.basename(genome._private_db_path)]


def test_forked_child_collection_keeps_the_parents_copy(
    release: dict[str, str], private_tmp: Path
) -> None:
    """A forked child that collects the instance it inherited must not delete the
    parent's private copy (the finalizer is pid-guarded).
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    genome = _construct(release)
    genome.drop_chrmt()
    private = genome._private_db_path
    assert private is not None
    pid = os.fork()
    if pid == 0:  # child
        del genome
        gc.collect()
        os._exit(0)
    os.waitpid(pid, 0)
    assert osp.exists(private)


def test_pickled_dropped_genome_keeps_its_drops_and_copies_on_write(
    release: dict[str, str], private_tmp: Path
) -> None:
    """A dropped genome sent to a worker reads the parent's private copy (the drops
    persist); a write in the restored instance goes to a copy of its own.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    genome = _construct(release)
    genome.drop_chrmt()
    parent_copy = genome._private_db_path
    restored = pickle.loads(pickle.dumps(genome))
    assert restored._db_connection_manager.db_path == parent_copy
    assert sorted(f.id for f in restored.db.features_of_type("gene")) == NO_CHRMT
    restored.drop_empty_go()
    assert restored._private_db_path not in (None, parent_copy)
    assert sorted(f.id for f in genome.db.features_of_type("gene")) == NO_CHRMT


def test_unpickled_genome_rebuilds_its_copy_after_the_parent_is_collected(
    release: dict[str, str], private_tmp: Path
) -> None:
    """The parent that wrote is collected (its private copy deleted) before the
    unpickled instance first reads: the read makes a copy of the shared file and
    replays the logged drops, so the instance still sees them.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    genome = _construct(release)
    genome.drop_chrmt()
    genome.remove_deprecated_go_terms()
    parent_copy = genome._private_db_path
    restored = pickle.loads(pickle.dumps(genome))
    del genome
    gc.collect()
    assert parent_copy is not None and not osp.exists(parent_copy)
    assert sorted(f.id for f in restored.db.features_of_type("gene")) == NO_CHRMT
    assert "Ontology_term" not in restored.db["YAL002W"].attributes
    assert restored._private_db_path not in (None, parent_copy)
    assert restored._db_writes == [
        ("delete", ("Q0010",)),
        ("remove_deprecated_go_terms", ()),
    ]


def test_genome_database_untrusted_reason_reads_only(release: dict[str, str]) -> None:
    """The read-only probe the data-gated tests use: missing, untrusted and trusted,
    without building or migrating anything.
    """
    root = _root(release)
    assert s288c.genome_database_untrusted_reason(str(root)) == "it does not exist"
    db_path = _old_code_rebuild(release)
    before = _identity(db_path)
    assert (
        s288c.genome_database_untrusted_reason(str(root))
        == "it carries no torchcell_genome_db_source record"
    )
    assert s288c.read_genome_database_record(str(db_path)) is None
    assert _identity(db_path) == before
    db_path.unlink()
    build_db(release[GFF_NAME], root)
    assert s288c.genome_database_untrusted_reason(str(root)) is None


def test_data_gated_helper_refuses_an_untrusted_real_root_by_name(
    release: dict[str, str],
) -> None:
    """``require_trusted_genome_database`` fails the test, by name, when a
    construction would migrate the root, and passes a trusted root; it never builds.
    """
    root = _root(release)
    db_path = _old_code_rebuild(release)
    before = _identity(db_path)
    with pytest.raises(pytest.fail.Exception) as exc:
        require_trusted_genome_database(str(root))
    assert str(exc.value) == (
        f"refusing to build or migrate the real genome database {db_path} from a "
        "test: it carries no torchcell_genome_db_source record. Construct "
        f"SCerevisiaeGenome(genome_root={str(root)!r}, ...) once outside the tests, "
        "then rerun."
    )
    assert _identity(db_path) == before
    db_path.unlink()
    build_db(release[GFF_NAME], root)
    require_trusted_genome_database(str(root))


def test_pid_alive_treats_another_users_process_as_alive() -> None:
    """Pid 1 belongs to root: signalling it raises PermissionError, which means alive;
    a reaped child's pid is dead.
    """
    assert s288c._pid_alive(1) is True
    assert s288c._pid_alive(_dead_pid()) is False


def test_sweep_leaves_another_users_dead_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dead pid's private copy owned by another uid is not this user's to remove."""
    name = f"torchcell-genome-{HOST}-{_dead_pid()}-abc123.db"
    (tmp_path / name).write_bytes(b"x")
    monkeypatch.setattr(os, "getuid", lambda: os.stat(tmp_path).st_uid + 1)
    assert s288c._sweep_dead(str(tmp_path), s288c._PRIVATE_COPY) == []
    assert os.listdir(tmp_path) == [name]


def test_record_transaction_must_advance_the_change_counter_once(
    release: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the record transaction did not leave the counter the record names, the
    build is refused by name and its temporary file removed.
    """
    root = _root(release)
    monkeypatch.setattr(s288c, "_change_counter", lambda path: 7)
    with pytest.raises(RuntimeError) as exc:
        s288c.write_genome_database(
            release[GFF_NAME], str(root), _expected_source(release)
        )
    assert re.fullmatch(
        rf"{re.escape(str(root))}/data\.db\.{re.escape(HOST)}\.{os.getpid()}\.[a-z0-9_]+"
        r"\.building: the record transaction left the sqlite change counter at 7, "
        "not 8",
        str(exc.value),
    )
    assert os.listdir(root) == []


def _rewrite_record(db_path: Path, edit: Any) -> None:
    conn = sqlite3.connect(db_path)
    (raw,) = conn.execute("SELECT record FROM torchcell_genome_db_source").fetchone()
    record = json.loads(raw)
    edit(record)
    conn.execute(
        "UPDATE torchcell_genome_db_source SET record = ?", (json.dumps(record),)
    )
    conn.commit()
    conn.close()


def test_record_without_a_version_is_migrated(
    release: dict[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    """A record with no ``version`` (an earlier draft wrote one without it and
    without ``change_counter``) is an older record: migrated once, never opened on a
    record this code cannot read.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    db_path = Path(release["__genome_root__"]) / "data.db"

    def strip(record: dict[str, Any]) -> None:
        del record["version"]
        del record["change_counter"]

    _rewrite_record(db_path, strip)
    with caplog.at_level(logging.WARNING, logger=s288c.__name__):
        genome = _construct(release)
    assert (
        caplog.records[0]
        .getMessage()
        .startswith(
            f"genome database {db_path} was not trusted (its record is version 0, "
            "older than 1); "
        )
    )
    _assert_recorded(release, db_path)
    assert list(genome.gene_set) == ALL_GENES


def test_record_from_newer_code_is_refused(release: dict[str, str]) -> None:
    """A record of a higher version was written by newer code: refused by name and
    left alone, so two code versions never rebuild the file back and forth.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    db_path = Path(release["__genome_root__"]) / "data.db"
    _rewrite_record(db_path, lambda record: record.update(version=2))
    before = _sha(db_path)
    with pytest.raises(s288c.GenomeDatabaseVersionError) as exc:
        _construct(release)
    assert str(exc.value) == (
        f"{db_path} carries a record of version 2, written by newer code than this "
        "checkout (which reads version 1); refusing to replace it. Update this "
        "checkout, or resubmit the job from an updated checkout."
    )
    assert _sha(db_path) == before


def test_unpickling_after_the_parent_is_collected_never_claims_its_copy(
    release: dict[str, str], private_tmp: Path
) -> None:
    """The parent that wrote is collected BEFORE the unpickle in the same process
    (CPython may reuse its address): the restored instance owns nothing, rebuilds a
    copy of its own by replay at its first read, and writes to it.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    genome = _construct(release)
    genome.drop_chrmt()
    parent_copy = genome._private_db_path
    blob = pickle.dumps(genome)
    del genome
    gc.collect()
    assert parent_copy is not None and not osp.exists(parent_copy)
    restored = pickle.loads(blob)
    assert restored._private_db_owner is None
    assert sorted(f.id for f in restored.db.features_of_type("gene")) == NO_CHRMT
    own = restored._private_db_path
    assert own not in (None, parent_copy) and osp.exists(own)
    restored.drop_empty_go()
    assert restored._private_db_path == own
    assert sorted(f.id for f in restored.db.features_of_type("gene")) == [
        "YAL001C",
        "YAL002W",
        "YBL002W",
        "YCL001W",
    ]


def test_sweep_skips_a_file_another_sweeper_removed_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A listed dead-pid copy vanishes before ``lstat`` or before ``remove`` (another
    sweeper won the race): skipped, no error, and the sweep goes on.
    """
    d = tmp_path / "sweep"
    d.mkdir()
    pid = _dead_pid()
    gone_at_stat = f"torchcell-genome-{HOST}-{pid}-aaa111.db"
    gone_at_remove = f"torchcell-genome-{HOST}-{pid}-bbb222.db"
    swept = f"torchcell-genome-{HOST}-{pid}-ccc333.db"
    for name in (gone_at_stat, gone_at_remove, swept):
        (d / name).write_bytes(b"x")
    real_lstat, real_remove = os.lstat, os.remove

    def racing_lstat(path: Any, *a: Any, **k: Any) -> os.stat_result:
        if str(path).endswith(gone_at_stat):
            real_remove(path)
        return real_lstat(path, *a, **k)

    def racing_remove(path: Any, *a: Any, **k: Any) -> None:
        if str(path).endswith(gone_at_remove):
            real_remove(path)
        real_remove(path, *a, **k)

    monkeypatch.setattr(os, "lstat", racing_lstat)
    monkeypatch.setattr(os, "remove", racing_remove)
    assert s288c._sweep_dead(str(d), s288c._PRIVATE_COPY) == [swept]
    assert os.listdir(d) == []


def test_sweep_skips_out_of_range_pids_and_directories(tmp_path: Path) -> None:
    """A crafted 20-digit pid and a directory with a matching name are left alone."""
    d = tmp_path / "sweep"
    d.mkdir()
    huge = f"torchcell-genome-{HOST}-{'9' * 20}-aaa111.db"
    zero = f"torchcell-genome-{HOST}-0-bbb222.db"
    a_dir = f"torchcell-genome-{HOST}-{_dead_pid()}-ccc333.db"
    (d / huge).write_bytes(b"x")
    (d / zero).write_bytes(b"x")
    (d / a_dir).mkdir()
    assert s288c._sweep_dead(str(d), s288c._PRIVATE_COPY) == []
    assert sorted(os.listdir(d)) == sorted([huge, zero, a_dir])


def test_sweep_removes_a_dead_builds_journal_with_it(release: dict[str, str]) -> None:
    """A kill inside the record transaction leaves ``...building`` and its
    ``...building-journal``: both are swept at the next construction.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    root = Path(release["__genome_root__"])
    stem = f"data.db.{HOST}.{_dead_pid()}.abc123.building"
    (root / stem).write_bytes(b"x")
    (root / f"{stem}-journal").write_bytes(b"x")
    _construct(release)
    assert os.listdir(root) == ["data.db"]


def test_kept_untrusted_file_is_replaced_atomically(
    release: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second damaged database replaces the earlier ``data.db.untrusted``, by a
    copy written to a temporary file in the root and renamed onto it.
    """
    db_path = _old_code_rebuild(release)
    _old_code_delete(db_path, "Q0010")
    _construct(release)
    root = Path(release["__genome_root__"])
    first_kept = _sha(root / "data.db.untrusted")
    db_path = _old_code_rebuild(release)
    _old_code_delete(db_path, "YAL001C")
    second_damaged = _sha(db_path)
    copies: list[str] = []
    real_copyfile = shutil.copyfile

    def recording_copyfile(src: str, dst: str) -> Any:
        copies.append(dst)
        return real_copyfile(src, dst)

    monkeypatch.setattr(shutil, "copyfile", recording_copyfile)
    _construct(release)
    assert len(copies) == 1
    assert re.fullmatch(
        rf"{re.escape(str(root))}/data\.db\.untrusted\.{re.escape(HOST)}\."
        rf"{os.getpid()}\.[a-z0-9_]+\.building",
        copies[0],
    )
    assert _sha(root / "data.db.untrusted") == second_damaged != first_kept
    assert sorted(os.listdir(root)) == ["data.db", "data.db.untrusted"]


def test_relations_alone_differing_are_kept(release: dict[str, str]) -> None:
    """A legacy database whose features equal a fresh build but whose relations lost
    a row differs from the fresh build: it is kept, not discarded.
    """
    db_path = _old_code_rebuild(release)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "DELETE FROM relations WHERE rowid = (SELECT MIN(rowid) FROM relations)"
    )
    conn.commit()
    conn.close()
    damaged = _sha(db_path)
    _construct(release)
    root = Path(release["__genome_root__"])
    assert sorted(os.listdir(root)) == ["data.db", "data.db.untrusted"]
    assert _sha(root / "data.db.untrusted") == damaged


def test_installed_database_is_mode_0644_after_migration_and_rebuild(
    release: dict[str, str],
) -> None:
    """Both install paths leave the shared file readable by every user (0644), even
    under a umask of 077, where sqlite alone would create it 0600.
    """
    db_path = _old_code_rebuild(release)
    old_umask = os.umask(0o077)
    try:
        _construct(release)
        migrated = db_path.stat().st_mode & 0o777
        _construct(release, overwrite=True)
        rebuilt = db_path.stat().st_mode & 0o777
    finally:
        os.umask(old_umask)
    assert (migrated, rebuilt) == (0o644, 0o644)


#: The field sets of the two record models at each RECORD_VERSION. Adding, removing
#: or renaming a field without bumping RECORD_VERSION fails here.
RECORD_FIELDS: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {
    1: (
        {
            "version": int,
            "source": GenomeDatabaseSource,
            "featuretype_counts": dict[str, int],
            "relations_count": int,
            "change_counter": int,
        },
        {
            "assembly_set": str,
            "gff_filename": str,
            "gff_sha256": str,
            "keep_order": bool,
            "merge_strategy": str,
            "sort_attribute_values": bool,
        },
    )
}


def test_record_field_sets_are_pinned_to_the_record_version() -> None:
    """Any field change (name or type) to GenomeDatabaseRecord or GenomeDatabaseSource
    bumps RECORD_VERSION (and adds its fields here).
    """
    current = (
        {
            name: f.annotation
            for name, f in s288c.GenomeDatabaseRecord.model_fields.items()
        },
        {name: f.annotation for name, f in GenomeDatabaseSource.model_fields.items()},
    )
    assert current == RECORD_FIELDS[s288c.RECORD_VERSION], (
        "GenomeDatabaseRecord/GenomeDatabaseSource fields changed: bump "
        "s288c.RECORD_VERSION and add the new fields and types to RECORD_FIELDS"
    )


@pytest.mark.parametrize(
    ("edit", "errors"),
    [
        (
            lambda r: r.update(extra_field=1),
            "1 errors: extra_field: Extra inputs are not permitted",
        ),
        (
            lambda r: r.pop("relations_count"),
            "1 errors: relations_count: Field required",
        ),
        (
            lambda r: (r.pop("relations_count"), r.update(extra_field=1)),
            "2 errors: relations_count: Field required; extra_field: Extra inputs "
            "are not permitted",
        ),
        (
            lambda r: r["source"].update(extra_field=1),
            "1 errors: source.extra_field: Extra inputs are not permitted",
        ),
    ],
)
def test_current_version_record_that_does_not_validate_is_refused_by_name(
    release: dict[str, str], edit: Any, errors: str
) -> None:
    """A version-1 record with a field this checkout does not know or lacks (a later
    edit changed the models without bumping the version) is refused by name with the
    remedy, never migrated, and the file is left alone.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    db_path = Path(release["__genome_root__"]) / "data.db"
    _rewrite_record(db_path, edit)
    before = _sha(db_path)
    with pytest.raises(s288c.GenomeDatabaseRecordError) as exc:
        _construct(release)
    assert str(exc.value) == (
        f"{db_path} carries a version 1 record that this checkout cannot read "
        f"({errors}); it was written by code with a different record schema at the "
        "same version. Update this checkout, or resubmit the job from an updated "
        "checkout."
    )
    assert _sha(db_path) == before


@pytest.mark.parametrize(
    ("bad", "shown"), [("1", "'1'"), (None, "None"), (True, "True")]
)
def test_non_integer_record_version_is_refused_by_name(
    release: dict[str, str], bad: Any, shown: str
) -> None:
    """A string, null or boolean ``version`` raises the named record error."""
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    db_path = Path(release["__genome_root__"]) / "data.db"
    _rewrite_record(db_path, lambda r: r.update(version=bad))
    before = _sha(db_path)
    with pytest.raises(s288c.GenomeDatabaseRecordError) as exc:
        _construct(release)
    assert str(exc.value) == (
        f"{db_path} carries a record whose version is {shown}, not an integer. "
        "Update this checkout, or resubmit the job from an updated checkout."
    )
    assert _sha(db_path) == before


def test_record_that_is_not_a_json_object_is_refused_by_name(
    release: dict[str, str],
) -> None:
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    db_path = Path(release["__genome_root__"]) / "data.db"
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE torchcell_genome_db_source SET record = '[1]'")
    conn.commit()
    conn.close()
    before = _sha(db_path)
    with pytest.raises(s288c.GenomeDatabaseRecordError) as exc:
        _construct(release)
    assert str(exc.value) == (
        f"{db_path} carries a record that is not a JSON object (list). Update this "
        "checkout, or resubmit the job from an updated checkout."
    )
    assert _sha(db_path) == before


def test_record_that_is_not_json_is_refused_by_name(release: dict[str, str]) -> None:
    """A record that is not JSON at all raises the named record error, not a bare
    JSONDecodeError, and the file is left alone.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    db_path = Path(release["__genome_root__"]) / "data.db"
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE torchcell_genome_db_source SET record = 'not json'")
    conn.commit()
    conn.close()
    before = _sha(db_path)
    with pytest.raises(s288c.GenomeDatabaseRecordError) as exc:
        _construct(release)
    assert str(exc.value) == (
        f"{db_path} carries a record that is not valid JSON (Expecting value: line 1 "
        "column 1 (char 0)). Update this checkout, or resubmit the job from an "
        "updated checkout."
    )
    assert _sha(db_path) == before


def test_overwrite_true_refuses_to_downgrade_a_newer_record(
    release: dict[str, str],
) -> None:
    """An older checkout's explicit rebuild over a database written by newer code is
    refused too; the file keeps its bytes.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    db_path = Path(release["__genome_root__"]) / "data.db"
    _rewrite_record(db_path, lambda r: r.update(version=2))
    before = _sha(db_path)
    with pytest.raises(s288c.GenomeDatabaseVersionError) as exc:
        _construct(release, overwrite=True)
    assert str(exc.value) == (
        f"{db_path} carries a record of version 2, written by newer code than this "
        "checkout (which reads version 1); refusing to replace it. Update this "
        "checkout, or resubmit the job from an updated checkout."
    )
    assert _sha(db_path) == before


def test_overwrite_true_rebuilds_over_an_older_record(release: dict[str, str]) -> None:
    """An explicit rebuild over a version-0 record (older code) proceeds and leaves
    a current record.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    db_path = Path(release["__genome_root__"]) / "data.db"
    _rewrite_record(db_path, lambda r: r.update(version=0))
    genome = _construct(release, overwrite=True)
    _assert_recorded(release, db_path)
    assert list(genome.gene_set) == ALL_GENES


@pytest.mark.parametrize(
    "make_copy",
    [copy.copy, copy.deepcopy, lambda g: pickle.loads(pickle.dumps(g))],
    ids=["copy", "deepcopy", "pickle"],
)
def test_a_write_on_a_copy_never_changes_the_original(
    release: dict[str, str], make_copy: Any
) -> None:
    """A shallow copy, a deep copy and a pickle round trip each get their own write
    log and gene-set cache: drop_chrmt on the copy leaves the original's gene set,
    cache, log and database untouched.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    genome = _construct(release)
    genome.drop_empty_go()
    original_log = list(genome._db_writes)
    original_genes = list(genome.gene_set)
    clone = make_copy(genome)
    clone.drop_chrmt()
    assert list(clone.gene_set) == ["YAL001C", "YAL002W", "YBL002W", "YCL001W"]
    assert (
        list(genome.gene_set)
        == original_genes
        == ["Q0010", "YAL001C", "YAL002W", "YBL002W", "YCL001W"]
    )
    assert genome._db_writes == original_log == [("delete", ("YBL001W",))]
    assert clone._db_writes == [("delete", ("YBL001W",)), ("delete", ("Q0010",))]
    assert sorted(f.id for f in genome.db.features_of_type("gene")) == original_genes


def test_first_write_owner_is_this_process_and_this_instances_token(
    release: dict[str, str],
) -> None:
    """Ownership is (pid, per-instance token), never id()."""
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    genome = _construct(release)
    genome.drop_chrmt()
    assert genome._private_db_owner == (os.getpid(), genome._instance_token)
    assert genome._instance_token != str(id(genome))
    assert re.fullmatch(r"[0-9a-f]{16}", genome._instance_token)


def test_replay_applies_the_logged_writes_in_order(
    release: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The replay after the parent's copy is gone applies the log in its recorded
    order. The three writes happen to commute in end state (each delete carries a
    fixed id list, and remove_deprecated_go_terms is recomputed on the current rows),
    so the order is asserted on the applied calls themselves.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    genome = _construct(release)
    genome.remove_deprecated_go_terms()
    genome.drop_chrmt()
    genome.drop_empty_go()
    log = list(genome._db_writes)
    # drop_empty_go ran after remove_deprecated_go_terms stripped YAL002W's only
    # (obsolete) term, so it dropped YAL002W as well as YBL001W.
    assert log == [
        ("remove_deprecated_go_terms", ()),
        ("delete", ("Q0010",)),
        ("delete", ("YAL002W", "YBL001W")),
    ]
    parent_genes = sorted(f.id for f in genome.db.features_of_type("gene"))
    assert parent_genes == ["YAL001C", "YBL002W", "YCL001W"]
    blob = pickle.dumps(genome)
    del genome
    gc.collect()
    restored = pickle.loads(blob)
    applied: list[tuple[str, tuple[str, ...]]] = []
    real_apply = SCerevisiaeGenome._apply_write

    def recording_apply(self: SCerevisiaeGenome, op: str, ids: tuple[str, ...]) -> None:
        applied.append((op, ids))
        real_apply(self, op, ids)

    monkeypatch.setattr(SCerevisiaeGenome, "_apply_write", recording_apply)
    assert sorted(f.id for f in restored.db.features_of_type("gene")) == parent_genes
    assert applied == log


class _Node:
    """A test item stand-in carrying a set of marker names."""

    def __init__(self, *markers: str) -> None:
        self.markers = set(markers)

    def get_closest_marker(self, name: str) -> Any:
        return name if name in self.markers else None


def _legacy_data_root(release: dict[str, str], tmp_path: Path) -> Path:
    """A DATA_ROOT whose genome root holds a record-less data.db, with the tier dir."""
    data_root = tmp_path / "data_root"
    genome_root = data_root / "data/sgd/genome"
    genome_root.mkdir(parents=True)
    (data_root / "torchcell-genomes" / SCerevisiaeGenome.ASSEMBLY_SET).mkdir(
        parents=True
    )
    gffutils.create_db(
        release[GFF_NAME],
        dbfn=str(genome_root / "data.db"),
        force=True,
        **s288c.CREATE_DB_KWARGS,
    )
    return data_root


@pytest.mark.parametrize("marker", ["data", "slow"])
def test_real_root_guard_refuses_data_and_slow_tests_by_name(
    release: dict[str, str], tmp_path: Path, marker: str
) -> None:
    """A data- or slow-marked test on a root a construction would migrate fails by
    name, and the database is untouched.
    """
    data_root = _legacy_data_root(release, tmp_path)
    db_path = data_root / "data/sgd/genome/data.db"
    before = _identity(db_path)
    with pytest.raises(pytest.fail.Exception) as exc:
        guard_real_genome_root(_Node(marker), lambda: str(data_root))
    assert str(exc.value) == (
        f"refusing to build or migrate the real genome database {db_path} from a "
        "test: it carries no torchcell_genome_db_source record. Construct "
        f"SCerevisiaeGenome(genome_root={str(db_path.parent)!r}, ...) once outside "
        "the tests, then rerun."
    )
    assert _identity(db_path) == before


def test_real_root_guard_never_reads_data_root_for_unmarked_tests() -> None:
    """An unmarked test returns before DATA_ROOT is resolved at all."""

    def untouchable() -> str:
        raise AssertionError("DATA_ROOT was read for an unmarked test")

    guard_real_genome_root(_Node(), untouchable)
    guard_real_genome_root(_Node("gpu", "network"), untouchable)


def test_real_root_guard_is_autouse(request: pytest.FixtureRequest) -> None:
    """The guard fixture runs for every test, marked or not."""
    assert "_never_migrate_a_real_genome_root" in request.fixturenames


def test_unpickled_instance_gets_a_fresh_token(release: dict[str, str]) -> None:
    """Every instance has its own ownership token, an unpickled one included."""
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    genome = _construct(release)
    genome.drop_chrmt()
    restored = pickle.loads(pickle.dumps(genome))
    assert restored._private_db_owner is None
    assert re.fullmatch(r"[0-9a-f]{16}", restored._instance_token)
    assert restored._instance_token != genome._instance_token


def test_sweep_range_check_holds_even_if_every_pid_looked_dead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pid 0 (which os.kill treats as the process group) and pids beyond a 32-bit
    pid_t are never ours, independently of the liveness probe.
    """
    d = tmp_path / "sweep"
    d.mkdir()
    zero = f"torchcell-genome-{HOST}-0-aaa111.db"
    huge = f"torchcell-genome-{HOST}-{2**31}-bbb222.db"
    top = f"torchcell-genome-{HOST}-{2**31 - 1}-ccc333.db"
    for name in (zero, huge, top):
        (d / name).write_bytes(b"x")
    monkeypatch.setattr(s288c, "_pid_alive", lambda pid: False)
    assert s288c._sweep_dead(str(d), s288c._PRIVATE_COPY) == [top]
    assert sorted(os.listdir(d)) == sorted([zero, huge])


def _damage(db_path: Path, kind: str) -> None:
    """Leave the file the way a killed in-place rebuild or a bad disk can."""
    data = db_path.read_bytes()
    if kind == "truncated":
        db_path.write_bytes(data[: len(data) // 2])
    elif kind == "garbage":
        db_path.write_bytes(b"\x07 not a database \x00" * 4096)
    elif kind == "zero_length":
        db_path.write_bytes(b"")
    else:  # zeroed page: the second 4096-byte page is all zeros
        db_path.write_bytes(data[:4096] + b"\0" * 4096 + data[8192:])


@pytest.mark.parametrize("kind", ["truncated", "garbage", "zero_length", "zeroed_page"])
def test_unreadable_database_is_migrated_on_the_default_path(
    release: dict[str, str], caplog: pytest.LogCaptureFixture, kind: str
) -> None:
    """A file sqlite cannot read (an old-code rebuild killed mid-write) is untrusted:
    kept as the single data.db.untrusted and replaced by a recorded build, with one
    WARNING; the genome opens with every gene.
    """
    db_path = _old_code_rebuild(release)
    _damage(db_path, kind)
    damaged = _sha(db_path)
    with caplog.at_level(logging.WARNING, logger=s288c.__name__):
        genome = _construct(release)
    root = Path(release["__genome_root__"])
    assert sorted(os.listdir(root)) == ["data.db", "data.db.untrusted"]
    assert _sha(root / "data.db.untrusted") == damaged
    _assert_recorded(release, db_path)
    assert len(caplog.records) == 1
    assert "its rows differ from a fresh build" in caplog.records[0].getMessage()
    assert list(genome.gene_set) == ALL_GENES


@pytest.mark.parametrize("kind", ["truncated", "garbage", "zero_length", "zeroed_page"])
def test_overwrite_true_rebuilds_over_an_unreadable_database(
    release: dict[str, str], kind: str
) -> None:
    """An explicit rebuild over a file sqlite cannot read proceeds (an unreadable
    file carries no newer record) and leaves a recorded database.
    """
    db_path = _old_code_rebuild(release)
    _damage(db_path, kind)
    genome = _construct(release, overwrite=True)
    assert os.listdir(release["__genome_root__"]) == ["data.db"]
    _assert_recorded(release, db_path)
    assert list(genome.gene_set) == ALL_GENES


def test_unreadable_reason_names_sqlite(release: dict[str, str]) -> None:
    """The untrusted reason for a malformed file carries sqlite's own message."""
    db_path = _old_code_rebuild(release)
    _damage(db_path, "garbage")
    reason = s288c.untrusted_reason(str(db_path), _expected_source(release), "call")
    assert reason == "sqlite cannot read it (file is not a database)"


def test_real_root_guard_skips_when_the_tier_or_the_root_is_absent(
    release: dict[str, str], tmp_path: Path
) -> None:
    """With the genome root present but no genomes tier, or the tier present but no
    genome root, a construction would fail before writing: the guard returns.
    """
    data_root = _legacy_data_root(release, tmp_path)
    db_path = data_root / "data/sgd/genome/data.db"
    before = _identity(db_path)
    (data_root / "torchcell-genomes" / SCerevisiaeGenome.ASSEMBLY_SET).rmdir()
    roots_read: list[str] = []

    def reading(root: Path) -> Any:
        def read() -> str:
            roots_read.append(str(root))
            return str(root)

        return read

    guard_real_genome_root(_Node("data"), reading(data_root))
    other = tmp_path / "other_root"
    (other / "torchcell-genomes" / SCerevisiaeGenome.ASSEMBLY_SET).mkdir(parents=True)
    guard_real_genome_root(_Node("data"), reading(other))
    assert roots_read == [str(data_root), str(other)]
    assert _identity(db_path) == before
    assert not (other / "data").exists()


def test_autouse_fixture_body_checks_the_request_node_under_data_root(
    release: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The function registered as the autouse fixture refuses a data-marked request
    whose DATA_ROOT holds an untrusted genome root: it reads ``request.node`` (not the
    session) and ``DATA_ROOT`` (not any other variable).
    """
    data_root = _legacy_data_root(release, tmp_path)
    db_path = data_root / "data/sgd/genome/data.db"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    request = SimpleNamespace(node=_Node("data"), session=_Node())
    with pytest.raises(pytest.fail.Exception) as exc:
        never_migrate_a_real_genome_root(request)  # type: ignore[arg-type]
    assert str(exc.value) == (
        f"refusing to build or migrate the real genome database {db_path} from a "
        "test: it carries no torchcell_genome_db_source record. Construct "
        f"SCerevisiaeGenome(genome_root={str(db_path.parent)!r}, ...) once outside "
        "the tests, then rerun."
    )


def test_a_copy_keeps_an_assigned_gene_set(release: dict[str, str]) -> None:
    """A gene set assigned through the ``gene_set`` setter (not derivable from the
    database) is carried by a copy as its own cache, not recomputed from rows.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    genome = _construct(release)
    genome.gene_set = GeneSet(["YAL001C", "YBL002W"])
    for clone in (copy.copy(genome), pickle.loads(pickle.dumps(genome))):
        assert list(clone.gene_set) == ["YAL001C", "YBL002W"]
        assert clone._gene_set is not genome._gene_set


def test_rows_sqlite_cannot_read_behind_a_readable_record_are_untrusted(
    release: dict[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    """The record reads fine but the root pages of ``features``, ``relations`` and
    their indexes are zeroed: the count read fails, the file is untrusted with sqlite's message, kept and
    replaced.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    db_path = Path(release["__genome_root__"]) / "data.db"
    conn = sqlite3.connect(db_path)
    rootpages = [
        page
        for (page,) in conn.execute(
            "SELECT rootpage FROM sqlite_master "
            "WHERE tbl_name IN ('features', 'relations') AND rootpage > 0"
        )
    ]
    (page_size,) = conn.execute("PRAGMA page_size").fetchone()
    conn.close()
    data = bytearray(db_path.read_bytes())
    for rootpage in rootpages:
        start = (rootpage - 1) * page_size
        data[start : start + page_size] = bytes(page_size)
    db_path.write_bytes(bytes(data))
    assert s288c._read_record_json(str(db_path)) is not None
    reason = s288c.untrusted_reason(str(db_path), _expected_source(release), "call")
    assert reason == "sqlite cannot read it (database disk image is malformed)"
    with caplog.at_level(logging.WARNING, logger=s288c.__name__):
        genome = _construct(release)
    assert sorted(os.listdir(release["__genome_root__"])) == [
        "data.db",
        "data.db.untrusted",
    ]
    assert list(genome.gene_set) == ALL_GENES
