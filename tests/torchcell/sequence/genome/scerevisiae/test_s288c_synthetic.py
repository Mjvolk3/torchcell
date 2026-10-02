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
import errno
import fcntl
import filecmp
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
import sys
import tempfile
import threading
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gffutils
import pandas as pd
import pytest
from attrs import fields as attrs_fields
from gffutils.exceptions import FeatureNotFoundError
from sortedcontainers import SortedDict, SortedSet

import tests.torchcell.conftest as tests_conftest
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


def _vanished_message(path: Path | str) -> str:
    """The named error's full message for ``path`` vanishing under this process."""
    return (
        f"{path} vanished while this process was reading, migrating or rebuilding it "
        f"([Errno 2] No such file or directory: '{path}'); another process "
        "(pre-2026.10.01 code) is rebuilding it. Every file is left alone. Retry "
        "when it has finished."
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
            "2 errors: extra_field: Extra inputs are not permitted; "
            "relations_count: Field required",
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
    assert os.listdir(release["__genome_root__"]) == ["data.db"]
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
    registered = tests_conftest._never_migrate_a_real_genome_root
    assert registered._get_wrapped_function() is never_migrate_a_real_genome_root


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


#: The untrusted reason each kind of damage produces (sqlite's own message).
UNREADABLE_REASON = {
    "truncated": "sqlite cannot read it (database disk image is malformed)",
    "garbage": "sqlite cannot read it (file is not a database)",
    "zero_length": "it carries no torchcell_genome_db_source record",
    "zeroed_page": "it carries no torchcell_genome_db_source record",
}


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
    assert [r.getMessage() for r in caplog.records] == [
        f"genome database {db_path} was not trusted ({UNREADABLE_REASON[kind]}); its "
        "rows differ from a fresh build, so it was kept as "
        f"{root / 'data.db.untrusted'} (replacing any earlier one) and replaced by the "
        "recorded build"
    ]
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


def test_validation_summary_is_independent_of_pydantic_error_order() -> None:
    """Two errors in either order give one string, sorted by location then message
    (pydantic 2.12 and 2.13 report them in different orders).
    """
    first = {"loc": ("relations_count",), "msg": "Field required"}
    second = {"loc": ("extra_field",), "msg": "Extra inputs are not permitted"}
    nested = {"loc": ("source", "extra_field"), "msg": "Extra inputs are not permitted"}
    expected = (
        "extra_field: Extra inputs are not permitted; relations_count: Field "
        "required; source.extra_field: Extra inputs are not permitted"
    )
    assert s288c.validation_summary([first, second, nested]) == expected
    assert s288c.validation_summary([nested, second, first]) == expected


_HOT_JOURNAL_WRITER = """
import os, sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.isolation_level = None
conn.execute("PRAGMA journal_mode=DELETE")
conn.execute("PRAGMA cache_size=1")
conn.execute("BEGIN")
conn.execute("DELETE FROM features WHERE seqid = 'chrI'")
conn.execute("CREATE TABLE spill (x TEXT)")
conn.executemany("INSERT INTO spill VALUES (?)", [("x" * 200,)] * 5000)
os.kill(os.getpid(), 9)
"""


def _leave_hot_journal(db_path: Path) -> None:
    """Kill an in-place write mid-transaction (what old code's gffutils ``update`` or
    ``delete`` can leave): ``data.db-journal`` stays hot beside a half-written file.
    """
    subprocess.run([sys.executable, "-c", _HOT_JOURNAL_WRITER, str(db_path)])
    assert (db_path.parent / "data.db-journal").stat().st_size > 0
    with pytest.raises(sqlite3.DatabaseError) as exc:
        s288c._read_record_json(str(db_path))
    assert exc.value.sqlite_errorname == "SQLITE_READONLY_ROLLBACK"


def _integrity(db_path: Path) -> list[tuple[str]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = conn.execute("PRAGMA integrity_check").fetchall()
    conn.close()
    return rows


def test_hot_journal_follows_the_kept_copy_on_the_default_path(
    release: dict[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    """A hot rollback journal beside the shared file makes it untrusted
    (SQLITE_READONLY_ROLLBACK); the journal moves beside the kept copy, so it is never
    paired with the fresh build, which passes integrity_check and opens with every
    gene.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    db_path = Path(release["__genome_root__"]) / "data.db"
    _leave_hot_journal(db_path)
    with caplog.at_level(logging.WARNING, logger=s288c.__name__):
        genome = _construct(release)
    root = Path(release["__genome_root__"])
    assert sorted(os.listdir(root)) == [
        "data.db",
        "data.db.untrusted",
        "data.db.untrusted-journal",
    ]
    assert _integrity(db_path) == [("ok",)]
    _assert_recorded(release, db_path)
    assert [r.getMessage() for r in caplog.records] == [
        f"genome database {db_path} was not trusted (sqlite cannot read it (attempt "
        "to write a readonly database)); its rows differ from a fresh build, so it was "
        f"kept as {root / 'data.db.untrusted'} (replacing any earlier one) and "
        "replaced by the recorded build"
    ]
    assert list(genome.gene_set) == ALL_GENES
    assert sorted(f.id for f in genome.db.features_of_type("gene")) == ALL_GENES


def test_explicit_rebuild_over_a_hot_journal_keeps_the_pair(
    release: dict[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    """``overwrite=True`` over a file with a hot journal keeps the old file with its
    journal (no crash point can leave a torn file without its journal), and the new
    build is never rolled back into; one WARNING names the kept pair.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    db_path = Path(release["__genome_root__"]) / "data.db"
    _leave_hot_journal(db_path)
    original = _sha(db_path)
    journal = _sha(db_path.parent / "data.db-journal")
    with caplog.at_level(logging.WARNING, logger=s288c.__name__):
        genome = _construct(release, overwrite=True)
    root = Path(release["__genome_root__"])
    assert sorted(os.listdir(root)) == [
        "data.db",
        "data.db.untrusted",
        "data.db.untrusted-journal",
    ]
    assert _sha(root / "data.db.untrusted") == original
    assert _sha(root / "data.db.untrusted-journal") == journal
    assert [r.getMessage() for r in caplog.records] == [
        f"genome database {db_path} had a hot journal; the explicit rebuild kept the "
        f"old file with its journal as {root / 'data.db.untrusted'} (replacing any "
        "earlier one) and installed the recorded build"
    ]
    assert _integrity(db_path) == [("ok",)]
    _assert_recorded(release, db_path)
    assert sorted(f.id for f in genome.db.features_of_type("gene")) == ALL_GENES


_LOCK_HOLDER = """
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.isolation_level = None
conn.execute("BEGIN EXCLUSIVE")
print("locked", flush=True)
sys.stdin.readline()
conn.execute("ROLLBACK")
"""


@pytest.fixture
def exclusive_lock(release: dict[str, str]) -> Any:
    """A second process holding an EXCLUSIVE lock on the shared data.db."""
    holders: list[subprocess.Popen[str]] = []

    def hold(db_path: Path) -> None:
        proc = subprocess.Popen(
            [sys.executable, "-c", _LOCK_HOLDER, str(db_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        holders.append(proc)
        assert proc.stdout is not None
        assert proc.stdout.readline() == "locked\n"

    yield hold
    for proc in holders:
        proc.communicate("release\n", timeout=30)


def test_locked_healthy_database_is_left_alone_on_the_default_path(
    release: dict[str, str], exclusive_lock: Any
) -> None:
    """Another process holds an exclusive lock on a healthy trusted database: a
    named refusal, nothing migrated or kept, the file's bytes unchanged.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    db_path = Path(release["__genome_root__"]) / "data.db"
    before = _sha(db_path)
    exclusive_lock(db_path)
    with pytest.raises(s288c.GenomeDatabaseUnavailableError) as exc:
        _construct(release)
    assert str(exc.value) == (
        f"{db_path} is locked by another process (SQLITE_BUSY: database is locked); "
        "every file is left alone. Retry when that process has finished."
    )
    assert os.listdir(release["__genome_root__"]) == ["data.db"]
    assert _sha(db_path) == before


def test_locked_newer_record_is_not_downgraded_by_overwrite_true(
    release: dict[str, str], exclusive_lock: Any
) -> None:
    """A version-2 record under an exclusive lock: ``overwrite=True`` refuses by name
    instead of reading the lock as damage and rebuilding over newer code's file.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    db_path = Path(release["__genome_root__"]) / "data.db"
    _rewrite_record(db_path, lambda r: r.update(version=2))
    before = _sha(db_path)
    exclusive_lock(db_path)
    with pytest.raises(s288c.GenomeDatabaseUnavailableError) as exc:
        _construct(release, overwrite=True)
    assert str(exc.value) == (
        f"{db_path} is locked by another process (SQLITE_BUSY: database is locked); "
        "every file is left alone. Retry when that process has finished."
    )
    assert os.listdir(release["__genome_root__"]) == ["data.db"]
    assert _sha(db_path) == before


@pytest.mark.parametrize("mode", [0o000, 0o200])
def test_unopenable_database_is_refused_by_name(
    release: dict[str, str], mode: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A data.db this process may not read (mode 000 or write-only) in a writable
    root: a named refusal naming the mode; no build, nothing kept, file untouched.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    db_path = Path(release["__genome_root__"]) / "data.db"
    real_connect = sqlite3.connect
    tmp_root = Path(release["__genome_root__"]).parent

    def refusing_connect(database: Any, *args: Any, **kwargs: Any) -> Any:
        # The permission check sqlite does would pass for root; reproduce its real
        # SQLITE_CANTOPEN error (an unopenable path) for every uid instead.
        if str(database).startswith(f"file:{db_path}?"):
            return real_connect(
                f"file:{tmp_root / 'absent' / 'x.db'}?mode=ro", *args, **kwargs
            )
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", refusing_connect)
    db_path.chmod(mode)
    try:
        with pytest.raises(s288c.GenomeDatabaseUnavailableError) as exc:
            _construct(release)
    finally:
        db_path.chmod(0o644)
    assert str(exc.value) == (
        f"{db_path} cannot be opened by this process (SQLITE_CANTOPEN: unable to "
        f"open database file; mode {oct(mode)}); every file is left alone. Fix its "
        "permissions or ownership, then retry."
    )
    assert os.listdir(release["__genome_root__"]) == ["data.db"]


def test_other_sqlite_errors_are_named_and_damage_codes_pass(tmp_path: Path) -> None:
    """An I/O error is refused by name; corrupt, not-a-database, a missing table and a
    hot journal are damage (the caller migrates).
    """

    def error(code: int, name: str, msg: str) -> sqlite3.DatabaseError:
        exc = sqlite3.OperationalError(msg)
        exc.sqlite_errorcode = code
        exc.sqlite_errorname = name
        return exc

    db = str(tmp_path / "data.db")
    with pytest.raises(s288c.GenomeDatabaseUnavailableError) as raised:
        s288c.require_damage(
            db, error(sqlite3.SQLITE_IOERR, "SQLITE_IOERR", "disk I/O error")
        )
    assert str(raised.value) == (
        f"{db} cannot be read right now (SQLITE_IOERR: disk I/O error); every file is "
        "left alone. Retry, and check the filesystem if it persists."
    )
    for code, name in [
        (sqlite3.SQLITE_CORRUPT, "SQLITE_CORRUPT"),
        (sqlite3.SQLITE_NOTADB, "SQLITE_NOTADB"),
        (sqlite3.SQLITE_ERROR, "SQLITE_ERROR"),
        (sqlite3.SQLITE_READONLY_ROLLBACK, "SQLITE_READONLY_ROLLBACK"),
    ]:
        s288c.require_damage(db, error(code, name, "x"))
    with pytest.raises(s288c.GenomeDatabaseUnavailableError):
        s288c.require_damage(db, error(sqlite3.SQLITE_READONLY, "SQLITE_READONLY", "x"))


def test_failed_install_removes_the_build_and_names_both_paths(tmp_path: Path) -> None:
    """The rename onto data.db fails (a non-empty directory sits there): the build is
    removed and the named error carries both paths.
    """
    db_path = tmp_path / "data.db"
    db_path.mkdir()
    (db_path / "x").write_bytes(b"x")
    build = tmp_path / "data.db.h.1.abc.building"
    build.write_bytes(b"build")
    with pytest.raises(s288c.GenomeDatabaseInstallError) as exc:
        s288c.install_genome_database(str(build), str(db_path))
    assert str(exc.value) == (
        f"the build {build} could not be renamed onto {db_path} ([Errno 21] "
        f"Is a directory: '{build}' -> '{db_path}'); the build was removed."
    )
    assert sorted(os.listdir(tmp_path)) == ["data.db"]


def test_two_row_record_table_is_refused_through_both_constructor_paths(
    release: dict[str, str],
) -> None:
    """The record-table defect propagates through the record-read catch points of the
    default path and of ``overwrite=True`` (they catch only sqlite errors).
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    db_path = Path(release["__genome_root__"]) / "data.db"
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO torchcell_genome_db_source (record) VALUES ('{}')")
    conn.commit()
    conn.close()
    before = _sha(db_path)
    for overwrite in (False, True):
        with pytest.raises(GenomeDatabaseSourceError) as exc:
            _construct(release, overwrite=overwrite)
        assert str(exc.value) == (
            f"{db_path}: torchcell_genome_db_source holds 2 rows, expected exactly 1"
        )
    assert _sha(db_path) == before


@pytest.mark.parametrize("when", ["before_copy", "after_copy"])
def test_concurrent_migration_keeps_the_original_untrusted_file(
    release: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    when: str,
) -> None:
    """Process B migrates completely while process A is at its copy step, either
    before A's copy is taken (A copies B's fresh build: it must keep nothing) or after
    it (A keeps its copy of the original): data.db.untrusted is always the ORIGINAL
    untrusted file.
    """
    db_path = _old_code_rebuild(release)
    _old_code_delete(db_path, "Q0010")
    original = _sha(db_path)
    source = _expected_source(release)
    real_copyfile = shutil.copyfile
    ran: list[str] = []
    installed: list[int] = []

    def interleaving_copyfile(src: str, dst: str) -> Any:
        def migrate_b() -> None:
            ran.append("B")
            reason = s288c.untrusted_reason(str(db_path), source, "call")
            assert reason is not None
            monkeypatch.setattr(shutil, "copyfile", real_copyfile)
            s288c.migrate_genome_database(
                release[GFF_NAME], str(db_path), source, "call", reason
            )
            installed.append(db_path.stat().st_ino)

        if not ran and when == "before_copy":
            migrate_b()
            return real_copyfile(src, dst)
        result = real_copyfile(src, dst)
        if not ran:
            migrate_b()
        return result

    monkeypatch.setattr(shutil, "copyfile", interleaving_copyfile)
    with caplog.at_level(logging.WARNING, logger=s288c.__name__):
        genome = _construct(release)
    root = Path(release["__genome_root__"])
    assert ran == ["B"]
    assert sorted(os.listdir(root)) == ["data.db", "data.db.untrusted"]
    assert _sha(root / "data.db.untrusted") == original
    _assert_recorded(release, db_path)
    assert list(genome.gene_set) == ALL_GENES
    kept_warnings = [r for r in caplog.records if "it was kept as" in r.getMessage()]
    # The second migrator finds data.db trusted under the root lock: one WARNING.
    assert len(kept_warnings) == 1
    if when == "before_copy":  # A found B's build installed and left it in place
        assert db_path.stat().st_ino == installed[0]


def test_legacy_file_whose_table_pages_are_corrupt_is_kept(
    release: dict[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    """A record-less file whose ``features`` table root page is zeroed (its indexes
    intact): the digest read raises SQLITE_CORRUPT, so it differs from a fresh build
    and is kept.
    """
    db_path = _old_code_rebuild(release)
    conn = sqlite3.connect(db_path)
    (rootpage,) = conn.execute(
        "SELECT rootpage FROM sqlite_master WHERE type = 'table' AND name = 'features'"
    ).fetchone()
    (page_size,) = conn.execute("PRAGMA page_size").fetchone()
    conn.close()
    data = bytearray(db_path.read_bytes())
    data[(rootpage - 1) * page_size : rootpage * page_size] = bytes(page_size)
    db_path.write_bytes(bytes(data))
    damaged = _sha(db_path)
    with caplog.at_level(logging.WARNING, logger=s288c.__name__):
        genome = _construct(release)
    root = Path(release["__genome_root__"])
    assert sorted(os.listdir(root)) == ["data.db", "data.db.untrusted"]
    assert _sha(root / "data.db.untrusted") == damaged
    assert list(genome.gene_set) == ALL_GENES


def _set_byte(db_path: Path, needle: bytes, value: int) -> None:
    """Overwrite the first byte of ``needle`` in the file, out of band."""
    data = bytearray(db_path.read_bytes())
    at = data.index(needle)
    data[at] = value
    db_path.write_bytes(bytes(data))


def test_record_cell_that_is_not_utf8_is_damage_on_both_paths(
    release: dict[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    """A 0xFF byte in the record cell makes Python's sqlite3 layer raise a
    DatabaseError with no sqlite error code: damaged content, so the default path
    migrates it (its feature and relation rows equal a fresh build, so nothing is
    kept) and ``overwrite=True`` rebuilds over it.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    db_path = Path(release["__genome_root__"]) / "data.db"
    _set_byte(db_path, b'"assembly_set"', 0xFF)
    reason = s288c.untrusted_reason(str(db_path), _expected_source(release), "call")
    assert reason is not None
    assert reason.startswith("sqlite cannot read it (Could not decode to UTF-8 column")
    with caplog.at_level(logging.WARNING, logger=s288c.__name__):
        genome = _construct(release)
    assert os.listdir(release["__genome_root__"]) == ["data.db"]
    assert len(caplog.records) == 1
    _assert_recorded(release, db_path)
    assert list(genome.gene_set) == ALL_GENES
    _set_byte(db_path, b'"assembly_set"', 0xFF)
    rebuilt = _construct(release, overwrite=True)
    _assert_recorded(release, db_path)
    assert list(rebuilt.gene_set) == ALL_GENES


_SILENT_HOT_WRITER = """
import os, sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.isolation_level = None
conn.execute("PRAGMA journal_mode=DELETE")
conn.execute("PRAGMA cache_size=1")
conn.execute("BEGIN")
conn.execute("UPDATE features SET attributes = attributes || ' '")
for (_,) in conn.execute("SELECT id FROM features ORDER BY featuretype"):
    pass
os.kill(os.getpid(), 9)
"""


def test_hot_journal_whose_file_pages_still_match_is_kept_as_a_pair(
    release: dict[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    """A killed in-place update spilled table pages but never page 1: the file's own
    pages still match its record and only the hot journal makes it unreadable. The
    pair is kept, the journal never stays beside the new build, the shared file is
    not rolled back in place, and exactly one WARNING names the case.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    db_path = Path(release["__genome_root__"]) / "data.db"
    subprocess.run([sys.executable, "-c", _SILENT_HOT_WRITER, str(db_path)])
    with pytest.raises(sqlite3.DatabaseError) as hot:
        s288c._read_record_json(str(db_path))
    assert hot.value.sqlite_errorname == "SQLITE_READONLY_ROLLBACK"
    assert s288c._committed_record_json(str(db_path)) is not None
    with caplog.at_level(logging.WARNING, logger=s288c.__name__):
        genome = _construct(release)
    root = Path(release["__genome_root__"])
    assert sorted(os.listdir(root)) == [
        "data.db",
        "data.db.untrusted",
        "data.db.untrusted-journal",
    ]
    assert _integrity(db_path) == [("ok",)]
    _assert_recorded(release, db_path)
    assert [r.getMessage() for r in caplog.records] == [
        f"genome database {db_path} was not trusted (sqlite cannot read it (attempt "
        "to write a readonly database)); its own pages match its record and only a "
        "companion journal made it unreadable, so the pair was kept as "
        f"{root / 'data.db.untrusted'} (replacing any earlier one) and replaced by the "
        "recorded build"
    ]
    assert sorted(f.id for f in genome.db.features_of_type("gene")) == ALL_GENES


@pytest.mark.parametrize("overwrite", [False, True])
def test_newer_record_beside_a_hot_journal_is_refused(
    release: dict[str, str], overwrite: bool
) -> None:
    """A version-2 record whose file also carries a hot journal is still refused (the
    record is read from the file's own pages), naming data.db, on both paths; nothing
    is touched.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    db_path = Path(release["__genome_root__"]) / "data.db"
    _rewrite_record(db_path, lambda r: r.update(version=2))
    subprocess.run([sys.executable, "-c", _SILENT_HOT_WRITER, str(db_path)])
    before = sorted(
        (n, _sha(Path(release["__genome_root__"]) / n))
        for n in os.listdir(release["__genome_root__"])
    )
    with pytest.raises(s288c.GenomeDatabaseVersionError) as exc:
        _construct(release, overwrite=overwrite)
    assert str(exc.value) == (
        f"{db_path} carries a record of version 2, written by newer code than this "
        "checkout (which reads version 1); refusing to replace it. Update this "
        "checkout, or resubmit the job from an updated checkout."
    )
    after = sorted(
        (n, _sha(Path(release["__genome_root__"]) / n))
        for n in os.listdir(release["__genome_root__"])
    )
    assert after == before
    assert [n for n, _ in after] == ["data.db", "data.db-journal"]


def test_directory_named_data_db_is_refused_by_name(release: dict[str, str]) -> None:
    """A directory where data.db belongs is named as such on both paths."""
    root = _root(release)
    (root / "data.db").mkdir()
    for overwrite in (False, True):
        with pytest.raises(s288c.GenomeDatabaseUnavailableError) as exc:
            _construct(release, overwrite=overwrite)
        assert str(exc.value) == (
            f"{root / 'data.db'} is a directory, not a database file; every file is "
            "left alone. Remove or rename it, then retry."
        )
    assert os.listdir(root) == ["data.db"]


def _sqlite_error(code: int, name: str, msg: str) -> sqlite3.DatabaseError:
    exc = sqlite3.OperationalError(msg)
    exc.sqlite_errorcode = code
    exc.sqlite_errorname = name
    return exc


@pytest.mark.parametrize(
    ("code", "name", "kind"),
    [
        (sqlite3.SQLITE_BUSY, "SQLITE_BUSY", "locked"),
        (sqlite3.SQLITE_LOCKED, "SQLITE_LOCKED", "locked"),
        (sqlite3.SQLITE_BUSY_SNAPSHOT, "SQLITE_BUSY_SNAPSHOT", "locked"),
        (sqlite3.SQLITE_CANTOPEN, "SQLITE_CANTOPEN", "access"),
        (sqlite3.SQLITE_PERM, "SQLITE_PERM", "access"),
        (sqlite3.SQLITE_IOERR_READ, "SQLITE_IOERR_READ", "other"),
        (sqlite3.SQLITE_CORRUPT_INDEX, "SQLITE_CORRUPT_INDEX", "damage"),
    ],
)
def test_sqlite_errors_are_classified_by_primary_code(
    tmp_path: Path, code: int, name: str, kind: str
) -> None:
    """Extended codes classify by their primary code (code & 0xFF)."""
    db = tmp_path / "data.db"
    db.write_bytes(b"x")
    db.chmod(0o640)
    exc = _sqlite_error(code, name, "msg")
    if kind == "damage":
        s288c.require_damage(str(db), exc)
        return
    with pytest.raises(s288c.GenomeDatabaseUnavailableError) as raised:
        s288c.require_damage(str(db), exc)
    expected = {
        "locked": f"{db} is locked by another process ({name}: msg); every file is "
        "left alone. Retry when that process has finished.",
        "access": f"{db} cannot be opened by this process ({name}: msg; mode 0o640); "
        "every file is left alone. Fix its permissions or ownership, then retry.",
        "other": f"{db} cannot be read right now ({name}: msg); every file is left "
        "alone. Retry, and check the filesystem if it persists.",
    }[kind]
    assert str(raised.value) == expected


def test_peek_of_an_unreadable_file_finds_no_record(tmp_path: Path) -> None:
    """The committed-record peek returns None when the file and its journal are not
    a database (nothing to refuse); both private copies are removed.
    """
    db = tmp_path / "data.db"
    db.write_bytes(b"\x07 not a database \x00" * 4096)
    (tmp_path / "data.db-journal").write_bytes(b"\x00" * 512)
    before = set(os.listdir(tempfile.gettempdir()))
    assert s288c._committed_record_json(str(db)) is None
    assert set(os.listdir(tempfile.gettempdir())) == before


@pytest.mark.parametrize("kept", [False, True])
def test_every_companion_kind_follows_the_kept_copy_or_is_removed(
    tmp_path: Path, kept: bool
) -> None:
    """``-journal``, ``-wal`` and ``-shm`` beside the old file are all moved beside
    the kept copy (or removed when nothing is kept) before the build is renamed in.
    """
    db = tmp_path / "data.db"
    db.write_bytes(b"old")
    for suffix in ("-journal", "-wal", "-shm"):
        (tmp_path / f"data.db{suffix}").write_bytes(suffix.encode())
    build = tmp_path / "data.db.h.1.abc.building"
    build.write_bytes(b"new")
    keep = tmp_path / "data.db.untrusted"
    if kept:
        keep.write_bytes(b"old")
    s288c.install_genome_database(str(build), str(db), str(keep) if kept else None)
    names = sorted(os.listdir(tmp_path))
    if kept:
        assert names == [
            "data.db",
            "data.db.untrusted",
            "data.db.untrusted-journal",
            "data.db.untrusted-shm",
            "data.db.untrusted-wal",
        ]
        assert (tmp_path / "data.db.untrusted-wal").read_bytes() == b"-wal"
    else:
        assert names == ["data.db"]
    assert db.read_bytes() == b"new"


def test_companions_are_handled_before_the_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At the moment the build is renamed onto data.db, no companion of the old file
    is still beside it.
    """
    db = tmp_path / "data.db"
    db.write_bytes(b"old")
    (tmp_path / "data.db-journal").write_bytes(b"j")
    build = tmp_path / "data.db.h.1.abc.building"
    build.write_bytes(b"new")
    seen: list[list[str]] = []
    real_replace = os.replace

    def watching_replace(src: Any, dst: Any) -> None:
        if str(src) == str(build):
            seen.append(sorted(os.listdir(tmp_path)))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", watching_replace)
    s288c.install_genome_database(str(build), str(db))
    assert seen == [["data.db", "data.db.h.1.abc.building"]]
    assert db.read_bytes() == b"new"
    assert os.listdir(tmp_path) == ["data.db"]


def test_companion_moved_away_by_another_migrator_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The companion vanishes just before this process moves it (another migrator
    took it): no error, and the install completes.
    """
    db = tmp_path / "data.db"
    db.write_bytes(b"old")
    journal = tmp_path / "data.db-journal"
    journal.write_bytes(b"j")
    build = tmp_path / "data.db.h.1.abc.building"
    build.write_bytes(b"new")
    keep = tmp_path / "data.db.untrusted"
    keep.write_bytes(b"old")
    real_replace = os.replace

    def racing_replace(src: Any, dst: Any) -> None:
        if str(src) == str(journal):
            real_replace(src, tmp_path / "taken-by-other")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", racing_replace)
    s288c.install_genome_database(str(build), str(db), str(keep))
    assert sorted(os.listdir(tmp_path)) == [
        "data.db",
        "data.db.untrusted",
        "taken-by-other",
    ]
    assert db.read_bytes() == b"new"


def test_stale_companion_of_an_earlier_kept_copy_is_removed(
    release: dict[str, str],
) -> None:
    """A first migration keeps a hot-journal pair; a later one keeps a different
    file with no journal: the earlier pair's journal does not stay beside it.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    db_path = Path(release["__genome_root__"]) / "data.db"
    _leave_hot_journal(db_path)
    _construct(release)
    root = Path(release["__genome_root__"])
    assert (root / "data.db.untrusted-journal").exists()
    _old_code_delete(db_path, "Q0010")
    damaged = _sha(db_path)
    _construct(release)
    assert sorted(os.listdir(root)) == ["data.db", "data.db.untrusted"]
    assert _sha(root / "data.db.untrusted") == damaged


def test_concurrent_migrators_keep_the_journal_with_the_kept_copy(
    release: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Two migrators both copy the same hot-journal file: B keeps the pair first;
    A, finding the same bytes already kept, leaves the kept file and its journal
    together.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    db_path = Path(release["__genome_root__"]) / "data.db"
    _leave_hot_journal(db_path)
    original = _sha(db_path)
    journal = _sha(db_path.parent / "data.db-journal")
    source = _expected_source(release)
    real_copyfile = shutil.copyfile
    ran: list[str] = []

    def interleaving_copyfile(src: str, dst: str) -> Any:
        result = real_copyfile(src, dst)
        if not ran and osp.basename(dst).startswith("data.db.untrusted."):
            ran.append("B")
            reason = s288c.untrusted_reason(str(db_path), source, "call")
            assert reason is not None
            monkeypatch.setattr(shutil, "copyfile", real_copyfile)
            s288c.migrate_genome_database(
                release[GFF_NAME], str(db_path), source, "call", reason
            )
        return result

    monkeypatch.setattr(shutil, "copyfile", interleaving_copyfile)
    with caplog.at_level(logging.WARNING, logger=s288c.__name__):
        _construct(release)
    # A finds data.db trusted under the root lock and leaves B's pair: one WARNING.
    assert len([r for r in caplog.records if "was kept as" in r.getMessage()]) == 1
    root = Path(release["__genome_root__"])
    assert ran == ["B"]
    assert sorted(os.listdir(root)) == [
        "data.db",
        "data.db.untrusted",
        "data.db.untrusted-journal",
    ]
    assert _sha(root / "data.db.untrusted") == original
    assert _sha(root / "data.db.untrusted-journal") == journal
    assert _integrity(db_path) == [("ok",)]


def test_rows_equal_migration_removes_an_old_companion(release: dict[str, str]) -> None:
    """A record-less file whose rows equal a fresh build, with a stray -wal beside it:
    replaced, nothing kept, and the -wal does not stay beside the new build.
    """
    db_path = _old_code_rebuild(release)
    (db_path.parent / "data.db-wal").write_bytes(b"")
    _construct(release)
    assert os.listdir(release["__genome_root__"]) == ["data.db"]
    _assert_recorded(release, db_path)


def test_locked_at_the_record_read_builds_nothing(
    release: dict[str, str], exclusive_lock: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lock seen at the first read is refused before any build."""
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    db_path = Path(release["__genome_root__"]) / "data.db"
    builds: list[str] = []
    real_write = s288c.write_genome_database

    def recording_write(*args: Any) -> str:
        builds.append(args[1])
        return real_write(*args)

    monkeypatch.setattr(s288c, "write_genome_database", recording_write)
    exclusive_lock(db_path)
    with pytest.raises(s288c.GenomeDatabaseUnavailableError):
        _construct(release)
    assert builds == []


def test_lock_at_the_count_read_is_refused_by_name(
    release: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lock first seen by the count read (the record read succeeded) is refused by
    name before any build.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    db_path = Path(release["__genome_root__"]) / "data.db"

    def locked_counts(conn: Any) -> Any:
        raise _sqlite_error(sqlite3.SQLITE_BUSY, "SQLITE_BUSY", "database is locked")

    monkeypatch.setattr(s288c, "_database_counts", locked_counts)
    with pytest.raises(s288c.GenomeDatabaseUnavailableError) as exc:
        _construct(release)
    assert str(exc.value).startswith(f"{db_path} is locked by another process")
    assert os.listdir(release["__genome_root__"]) == ["data.db"]


def test_lock_at_the_digest_read_keeps_nothing(
    release: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lock first seen while digesting the untrusted file is refused by name: no
    kept file, no installed build, no temporary left.
    """
    db_path = _old_code_rebuild(release)
    before = _sha(db_path)
    real_digest = s288c.database_content_digest

    def digest(path: str) -> str:
        if path == str(db_path):
            raise _sqlite_error(
                sqlite3.SQLITE_BUSY, "SQLITE_BUSY", "database is locked"
            )
        return real_digest(path)

    monkeypatch.setattr(s288c, "database_content_digest", digest)
    with pytest.raises(s288c.GenomeDatabaseUnavailableError):
        _construct(release)
    assert os.listdir(release["__genome_root__"]) == ["data.db"]
    assert _sha(db_path) == before


def test_non_sqlite_error_at_the_count_read_propagates(
    release: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The count read catches only sqlite errors: an OSError there is not mistaken
    for damage, and nothing is built or migrated.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))

    def failing_counts(conn: Any) -> Any:
        raise OSError("count read failed")

    builds: list[str] = []
    monkeypatch.setattr(s288c, "_database_counts", failing_counts)

    def recording_write(gff: str, db_dir: str, source: Any) -> str:
        builds.append(db_dir)
        return ""

    monkeypatch.setattr(s288c, "write_genome_database", recording_write)
    with pytest.raises(OSError, match="^count read failed$"):
        _construct(release)
    assert builds == []
    assert os.listdir(release["__genome_root__"]) == ["data.db"]


def test_non_sqlite_error_at_the_digest_read_propagates(
    release: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The untrusted digest catches only sqlite errors: an OSError there is not
    mistaken for damage, nothing is kept, and the file is left as it was.
    """
    db_path = _old_code_rebuild(release)
    before = _sha(db_path)
    real_digest = s288c.database_content_digest

    def digest(path: str) -> str:
        if path == str(db_path):
            raise OSError("digest read failed")
        return real_digest(path)

    monkeypatch.setattr(s288c, "database_content_digest", digest)
    with pytest.raises(OSError, match="^digest read failed$"):
        _construct(release)
    assert os.listdir(release["__genome_root__"]) == ["data.db"]
    assert _sha(db_path) == before


# --------------------------------------------------------------------------- #
# 2026.10.02 (seventh review): root lock, kept-copy edges, crash windows, URIs,
# non-regular files, committed-record peek, in-place rebuild under a reader.
# --------------------------------------------------------------------------- #

_ROOT_LOCK_HOLDER = """
import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDONLY)
fcntl.flock(fd, fcntl.LOCK_EX)
print("locked", flush=True)
sys.stdin.readline()
"""


def test_migration_waits_for_the_root_lock(release: dict[str, str]) -> None:
    """The keep and install steps run under an exclusive flock on the genome root: a
    migration started while another process holds it changes nothing until the lock
    is released, then completes.
    """
    db_path = _old_code_rebuild(release)
    _old_code_delete(db_path, "Q0010")
    root = Path(release["__genome_root__"])
    holder = subprocess.Popen(
        [sys.executable, "-c", _ROOT_LOCK_HOLDER, str(root)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline() == "locked\n"
    done: list[list[str]] = []
    worker = threading.Thread(
        target=lambda: done.append(list(_construct(release).gene_set))
    )
    worker.start()
    worker.join(timeout=3)
    try:
        assert worker.is_alive()
        assert sorted(n for n in os.listdir(root) if not n.endswith(".building")) == [
            "data.db"
        ]
    finally:
        holder.communicate("release\n", timeout=30)
    worker.join(timeout=60)
    assert done == [ALL_GENES]
    assert sorted(os.listdir(root)) == ["data.db", "data.db.untrusted"]


_SHARED_ROOT_LOCK_HOLDER = _ROOT_LOCK_HOLDER.replace("LOCK_EX", "LOCK_SH")


def test_migration_lock_is_exclusive(release: dict[str, str]) -> None:
    """The root lock is EXCLUSIVE: a migration waits even for a SHARED holder, so two
    migrators can never hold it together (a shared lock would let them interleave
    their keep and install steps).
    """
    db_path = _old_code_rebuild(release)
    _old_code_delete(db_path, "Q0010")
    root = Path(release["__genome_root__"])
    holder = subprocess.Popen(
        [sys.executable, "-c", _SHARED_ROOT_LOCK_HOLDER, str(root)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline() == "locked\n"
    done: list[list[str]] = []
    worker = threading.Thread(
        target=lambda: done.append(list(_construct(release).gene_set))
    )
    worker.start()
    worker.join(timeout=3)
    try:
        assert worker.is_alive()
        assert sorted(n for n in os.listdir(root) if not n.endswith(".building")) == [
            "data.db"
        ]
    finally:
        holder.communicate("release\n", timeout=30)
    worker.join(timeout=60)
    assert done == [ALL_GENES]
    assert sorted(os.listdir(root)) == ["data.db", "data.db.untrusted"]


def test_root_that_cannot_be_locked_is_refused_by_name(
    release: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A filesystem without flock support (ENOLCK) is refused by name; the migration
    never proceeds unlocked and every file is left alone.
    """
    db_path = _old_code_rebuild(release)
    before = _sha(db_path)
    root = Path(release["__genome_root__"])

    def no_locks(fd: int, op: int) -> None:
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(fcntl, "flock", no_locks)
    with pytest.raises(s288c.GenomeDatabaseUnavailableError) as exc:
        _construct(release)
    assert str(exc.value) == (
        f"{root} cannot be locked ([Errno 37] No locks available); a migration or "
        "rebuild needs an exclusive lock on the genome root, so every file is left "
        "alone. Put genome_root on a filesystem that supports flock."
    )
    assert os.listdir(root) == ["data.db"]
    assert _sha(db_path) == before


_CONSTRUCT_IN_CHILD = """
import json, sys
import torchcell.sequence.genome.scerevisiae.s288c as s288c
release = json.loads(sys.argv[1])
s288c.resolve = lambda assembly_set, filename: release[filename]
genome = s288c.SCerevisiaeGenome(
    genome_root=release["__genome_root__"], go_root=release["__go_root__"]
)
print(len(genome.gene_set))
"""


def test_concurrent_processes_keep_the_original_pair_over_an_earlier_one(
    release: dict[str, str], private_tmp: Path
) -> None:
    """Four real processes migrate one hot-journal file while an earlier, different,
    same-size kept pair is present: the kept file ends as the original and its
    journal is the original journal.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    root = Path(release["__genome_root__"])
    db_path = root / "data.db"
    earlier = bytearray(db_path.read_bytes())
    earlier[100] ^= 0xFF
    (root / "data.db.untrusted").write_bytes(bytes(earlier))
    (root / "data.db.untrusted-journal").write_bytes(b"earlier journal")
    subprocess.run([sys.executable, "-c", _SILENT_HOT_WRITER, str(db_path)])
    original = _sha(db_path)
    journal = _sha(root / "data.db-journal")
    # Fresh interpreters (plain subprocesses, not a fork of this multi-threaded
    # pytest process), started together.
    env = {**os.environ, "TMPDIR": str(private_tmp)}
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _CONSTRUCT_IN_CHILD, json.dumps(release)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        for _ in range(4)
    ]
    results = [proc.communicate(timeout=120) for proc in procs]
    assert [proc.returncode for proc in procs] == [0, 0, 0, 0], [e for _, e in results]
    assert sorted(out for out, _ in results) == ["6\n", "6\n", "6\n", "6\n"]
    assert sorted(os.listdir(root)) == [
        "data.db",
        "data.db.untrusted",
        "data.db.untrusted-journal",
    ]
    assert _sha(root / "data.db.untrusted") == original
    assert _sha(root / "data.db.untrusted-journal") == journal
    assert _integrity(db_path) == [("ok",)]


def test_kept_copy_takes_the_mode_of_the_file_it_preserves(
    release: dict[str, str],
) -> None:
    """The kept copy has the original's mode (0640 here), not mkstemp's 0600."""
    db_path = _old_code_rebuild(release)
    _old_code_delete(db_path, "Q0010")
    db_path.chmod(0o640)
    _construct(release)
    kept = Path(release["__genome_root__"]) / "data.db.untrusted"
    assert kept.stat().st_mode & 0o777 == 0o640


def test_unreadable_earlier_kept_copy_counts_as_different(
    release: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An earlier kept copy this process cannot read (another user's 0600 copy) is a
    different file: it is replaced by the original, never an unnamed
    PermissionError.
    """
    db_path = _old_code_rebuild(release)
    _old_code_delete(db_path, "Q0010")
    damaged = _sha(db_path)
    root = Path(release["__genome_root__"])
    kept = root / "data.db.untrusted"
    kept.write_bytes(db_path.read_bytes()[:-1] + b"\x01")
    real_cmp = filecmp.cmp

    def unreadable_cmp(a: Any, b: Any, shallow: bool = True) -> bool:
        if str(kept) in (str(a), str(b)):
            raise PermissionError(13, "Permission denied", str(kept))
        return real_cmp(a, b, shallow=shallow)

    monkeypatch.setattr(filecmp, "cmp", unreadable_cmp)
    _construct(release)
    assert _sha(kept) == damaged


@pytest.mark.parametrize("kind", ["directory", "dangling_symlink", "symlink_to_file"])
def test_non_regular_kept_path_is_refused_by_name(
    release: dict[str, str], kind: str
) -> None:
    """``data.db.untrusted`` as a directory or a dangling symlink: the untrusted file
    cannot be kept there, so the migration is refused by name and nothing changes.
    """
    db_path = _old_code_rebuild(release)
    _old_code_delete(db_path, "Q0010")
    before = _sha(db_path)
    root = Path(release["__genome_root__"])
    kept = root / "data.db.untrusted"
    if kind == "directory":
        kept.mkdir()
    elif kind == "dangling_symlink":
        kept.symlink_to(root / "absent")
    else:
        kept.symlink_to(db_path)
    with pytest.raises(s288c.GenomeDatabaseUnavailableError) as exc:
        _construct(release)
    assert str(exc.value) == (
        f"{kept} is not a regular file, so the untrusted {db_path} cannot be kept "
        "there; every file is left alone. Remove or rename it, then retry."
    )
    assert _sha(db_path) == before
    assert sorted(n for n in os.listdir(root)) == ["data.db", "data.db.untrusted"]


def test_every_kill_point_of_a_pair_migration_leaves_a_recoverable_root(
    release: dict[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    """A pair migration (a file whose own pages match its record, with a hot journal)
    has these kill points; each leaves a root the next construction repairs:

    * K0, before anything under the lock changed: the file and its hot journal, the
      plain hot-journal case.
    * K1, after the copy was renamed onto ``data.db.untrusted``, before the journal
      moved: the file keeps its hot journal (untrusted); the next migration finds the
      same bytes kept and leaves them, then moves the journal.
    * K2, after the journal moved beside the kept copy, before the build was renamed
      in: the torn file has no journal and its cheap checks pass, but it is
      byte-identical to the kept copy whose journal exists, which marks it untrusted.
    * K3, after the rename: the recorded build, trusted.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    root = Path(release["__genome_root__"])
    db_path = root / "data.db"
    subprocess.run([sys.executable, "-c", _SILENT_HOT_WRITER, str(db_path)])
    original = _sha(db_path)
    journal = _sha(root / "data.db-journal")
    # K1: the copy is kept, the journal is still beside data.db.
    shutil.copyfile(db_path, root / "data.db.untrusted")
    assert s288c.untrusted_reason(str(db_path), _expected_source(release), "c") == (
        "sqlite cannot read it (attempt to write a readonly database)"
    )
    # K2: the journal moved, the build was never renamed in.
    os.replace(root / "data.db-journal", root / "data.db.untrusted-journal")
    assert s288c.untrusted_reason(str(db_path), _expected_source(release), "c") == (
        "its hot journal was moved beside data.db.untrusted by a migration that did "
        "not finish"
    )
    with caplog.at_level(logging.WARNING, logger=s288c.__name__):
        genome = _construct(release)
    assert sorted(os.listdir(root)) == [
        "data.db",
        "data.db.untrusted",
        "data.db.untrusted-journal",
    ]
    assert _sha(root / "data.db.untrusted") == original
    assert _sha(root / "data.db.untrusted-journal") == journal
    assert _integrity(db_path) == [("ok",)]
    _assert_recorded(release, db_path)
    assert [r.getMessage() for r in caplog.records] == [
        f"genome database {db_path} was not trusted (its hot journal was moved beside "
        "data.db.untrusted by a migration that did not finish); its own pages match "
        "its record and only a companion journal made it unreadable, so the pair was "
        f"kept as {root / 'data.db.untrusted'} (which already held these bytes and "
        "was left in place) and replaced by the recorded build"
    ]
    assert sorted(f.id for f in genome.db.features_of_type("gene")) == ALL_GENES
    # K3: trusted; the persistent kept pair does not make the new file untrusted.
    assert s288c.untrusted_reason(str(db_path), _expected_source(release), "c") is None


def test_k1_state_migrates_and_moves_the_journal(release: dict[str, str]) -> None:
    """K1 (copy kept, journal still beside data.db): the next migration leaves the
    equal kept copy in place and moves the journal beside it.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    root = Path(release["__genome_root__"])
    db_path = root / "data.db"
    subprocess.run([sys.executable, "-c", _SILENT_HOT_WRITER, str(db_path)])
    journal = _sha(root / "data.db-journal")
    shutil.copyfile(db_path, root / "data.db.untrusted")
    kept_inode = (root / "data.db.untrusted").stat().st_ino
    _construct(release)
    assert (root / "data.db.untrusted").stat().st_ino == kept_inode
    assert _sha(root / "data.db.untrusted-journal") == journal
    assert _integrity(db_path) == [("ok",)]


def test_companion_that_cannot_be_moved_stops_the_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A companion that cannot be moved for any reason other than being gone stops the
    install: the old file and its journal stay together and the build is not renamed.
    """
    db = tmp_path / "data.db"
    db.write_bytes(b"old")
    journal = tmp_path / "data.db-journal"
    journal.write_bytes(b"j")
    build = tmp_path / "data.db.h.1.abc.building"
    build.write_bytes(b"new")
    keep = tmp_path / "data.db.untrusted"
    keep.write_bytes(b"old")
    real_replace = os.replace

    def failing_replace(src: Any, dst: Any) -> None:
        if str(src) == str(journal):
            raise PermissionError(13, "Permission denied", str(src))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(PermissionError, match="Permission denied"):
        s288c.install_genome_database(str(build), str(db), str(keep))
    assert db.read_bytes() == b"old"
    assert journal.read_bytes() == b"j"
    assert build.read_bytes() == b"new"


def test_hot_journal_whose_committed_record_is_unreadable_is_damage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Beside a hot journal, a committed state with no readable record re-raises the
    sqlite error (damage), never a TypeError from checking a missing record.
    """
    conn = sqlite3.connect(tmp_path / "data.db")
    conn.execute("CREATE TABLE features (seqid TEXT)")
    conn.executemany("INSERT INTO features VALUES (?)", [("chrI",)] * 2000)
    conn.commit()
    conn.close()
    db_path = tmp_path / "data.db"
    _leave_hot_journal(db_path)
    monkeypatch.setattr(s288c, "_committed_record_json", lambda path: None)
    with pytest.raises(sqlite3.DatabaseError) as exc:
        s288c._read_record_json_checked(str(db_path))
    assert exc.value.sqlite_errorname == "SQLITE_READONLY_ROLLBACK"


def test_peek_copies_go_to_the_temp_dir_named_for_the_sweep(
    release: dict[str, str], monkeypatch: pytest.MonkeyPatch, private_tmp: Path
) -> None:
    """The committed-record peek copies the file and its journal into the temp dir,
    never the genome root, under names the dead-pid sweep matches; both are removed
    afterwards.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    db_path = Path(release["__genome_root__"]) / "data.db"
    _leave_hot_journal(db_path)
    copies: list[str] = []
    real_copyfile = shutil.copyfile

    def recording_copyfile(src: str, dst: str) -> Any:
        copies.append(dst)
        return real_copyfile(src, dst)

    monkeypatch.setattr(shutil, "copyfile", recording_copyfile)
    assert s288c._committed_record_json(str(db_path)) is not None
    assert len(copies) == 2
    assert {osp.dirname(c) for c in copies} == {str(private_tmp)}
    assert copies[1] == copies[0] + "-journal"
    for copied in copies:
        m = s288c._PRIVATE_COPY.match(osp.basename(copied))
        assert m is not None and m["host"] == HOST and int(m["pid"]) == os.getpid()
    assert os.listdir(private_tmp) == []


def test_dead_processes_peek_copies_are_swept(
    release: dict[str, str], private_tmp: Path
) -> None:
    """A peek copy and its journal left by a killed process are removed by the next
    private-copy creation.
    """
    dead = _dead_pid()
    names = [
        f"torchcell-genome-{HOST}-{dead}-peekabc123.db",
        f"torchcell-genome-{HOST}-{dead}-peekabc123.db-journal",
    ]
    for name in names:
        (private_tmp / name).write_bytes(b"x")
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    genome = _construct(release)
    genome.drop_chrmt()
    assert genome._private_db_path is not None
    assert os.listdir(private_tmp) == [osp.basename(genome._private_db_path)]


@pytest.mark.parametrize("piece", ["q?x", "h#x", "p%41x"])
def test_genome_root_with_uri_characters_opens_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, piece: str
) -> None:
    """A genome root whose name holds ``?``, ``#`` or ``%`` is opened through an
    encoded URI: the database is found and nothing is created outside the root.
    """
    paths = write_release(tmp_path)
    monkeypatch.setattr(
        s288c, "resolve", lambda assembly_set, filename: paths[filename]
    )
    (tmp_path / "go").mkdir()
    (tmp_path / "go" / "go.obo").write_text(GO_OBO)
    parent = tmp_path / "parent"
    root = parent / piece
    build_db(paths[GFF_NAME], root)
    genome = SCerevisiaeGenome(genome_root=str(root), go_root=str(tmp_path / "go"))
    assert list(genome.gene_set) == ALL_GENES
    assert os.listdir(parent) == [piece]
    assert os.listdir(root) == ["data.db"]


@pytest.mark.parametrize("kind", ["fifo", "dangling_symlink", "symlink_loop"])
def test_non_regular_data_db_is_refused_by_name(
    release: dict[str, str], kind: str
) -> None:
    """A FIFO (which would block every open), a dangling symlink or a symlink loop
    named data.db is refused by name on both paths and left alone.
    """
    root = _root(release)
    db_path = root / "data.db"
    if kind == "fifo":
        os.mkfifo(db_path)
    elif kind == "dangling_symlink":
        db_path.symlink_to(root / "absent.db")
    else:
        db_path.symlink_to(db_path)
    for overwrite in (False, True):
        # In a daemon thread, so a regression that opens the FIFO (which blocks
        # forever) fails here instead of hanging the suite.
        outcome: list[BaseException] = []

        def attempt(overwrite: bool = overwrite) -> None:
            try:
                _construct(release, overwrite=overwrite)
            except BaseException as caught:  # recorded and asserted below
                outcome.append(caught)

        worker = threading.Thread(target=attempt, daemon=True)
        worker.start()
        worker.join(timeout=30)
        assert not worker.is_alive()
        assert len(outcome) == 1
        assert isinstance(outcome[0], s288c.GenomeDatabaseUnavailableError)
        assert str(outcome[0]) == (
            f"{db_path} is not a regular file (a FIFO, socket or device, or a symlink "
            "that does not lead to a regular file); every file is left alone. Remove "
            "or rename it, then retry."
        )
    assert os.listdir(root) == ["data.db"]
    assert os.path.lexists(db_path)


_RECORD_REWRITER = """
import json, os, sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.isolation_level = None
conn.execute("PRAGMA journal_mode=DELETE")
conn.execute("PRAGMA cache_size=1")
conn.execute("BEGIN")
(raw,) = conn.execute("SELECT record FROM torchcell_genome_db_source").fetchone()
record = json.loads(raw)
record["version"] = 1
conn.execute("UPDATE torchcell_genome_db_source SET record = ?", (json.dumps(record),))
conn.execute("CREATE TABLE spill (x TEXT)")
conn.executemany("INSERT INTO spill VALUES (?)", [("x" * 200,)] * 5000)
os.kill(os.getpid(), 9)
"""


@pytest.mark.parametrize("overwrite", [False, True])
def test_committed_newer_record_under_an_uncommitted_older_one_is_refused(
    release: dict[str, str], overwrite: bool
) -> None:
    """The committed record is version 2; a killed writer left version 1 in the file's
    raw pages with a hot journal. The refusal reads the committed (rolled-back)
    record, so it still refuses, naming data.db, and nothing is touched.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    root = Path(release["__genome_root__"])
    db_path = root / "data.db"
    _rewrite_record(db_path, lambda r: r.update(version=2))
    subprocess.run([sys.executable, "-c", _RECORD_REWRITER, str(db_path)])
    raw_copy = root.parent / "raw.db"
    shutil.copyfile(db_path, raw_copy)
    raw = s288c._read_record_json(str(raw_copy))
    assert raw is not None and json.loads(raw)["version"] == 1
    raw_copy.unlink()
    before = sorted((n, _sha(root / n)) for n in os.listdir(root))
    with pytest.raises(s288c.GenomeDatabaseVersionError) as exc:
        _construct(release, overwrite=overwrite)
    assert str(exc.value).startswith(f"{db_path} carries a record of version 2")
    assert sorted((n, _sha(root / n)) for n in os.listdir(root)) == before


def test_two_committed_records_beside_a_hot_journal_name_data_db(
    release: dict[str, str],
) -> None:
    """A committed record table with two rows, beside a hot journal: the defect names
    data.db, not the private peek copy.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    db_path = Path(release["__genome_root__"]) / "data.db"
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO torchcell_genome_db_source (record) VALUES ('{}')")
    conn.commit()
    conn.close()
    _leave_hot_journal(db_path)
    with pytest.raises(GenomeDatabaseSourceError) as exc:
        _construct(release)
    assert str(exc.value) == (
        f"{db_path}: torchcell_genome_db_source holds 2 rows, expected exactly 1"
    )


def test_in_place_rebuild_under_a_reader_is_named(release: dict[str, str]) -> None:
    """Old code rebuilds data.db in place after this genome was constructed: its first
    read finds an empty gffutils meta table and raises the named error, not
    gffutils' bare TypeError.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    genome = _construct(release)
    db_path = Path(release["__genome_root__"]) / "data.db"
    partial = db_path.parent / "partial.db"
    conn = sqlite3.connect(partial)
    conn.executescript(gffutils.constants.SCHEMA)
    conn.commit()
    conn.close()
    os.replace(partial, db_path)
    with pytest.raises(s288c.GenomeDatabaseUnavailableError) as exc:
        genome.gene_set  # noqa: B018
    assert str(exc.value) == (
        f"{db_path} has no gffutils metadata: another process (pre-2026.10.01 code) "
        "is rebuilding it in place; nothing was changed. Retry when it has finished; "
        "the next construction migrates its result."
    )


def test_file_changed_during_the_migration_is_refused_by_name(
    release: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Old code writes data.db in place after it was inspected and copied but before
    the lock is taken: the migration is refused by name, nothing is kept or
    installed, and the file keeps that write.
    """
    db_path = _old_code_rebuild(release)
    _old_code_delete(db_path, "Q0010")
    root = Path(release["__genome_root__"])
    real_copyfile = shutil.copyfile
    written: list[str] = []

    def copy_then_write(src: str, dst: str) -> Any:
        result = real_copyfile(src, dst)
        if osp.basename(dst).startswith("data.db.untrusted.") and not written:
            _old_code_delete(db_path, "YAL001C")
            written.append(_sha(db_path))
        return result

    monkeypatch.setattr(shutil, "copyfile", copy_then_write)
    with pytest.raises(s288c.GenomeDatabaseUnavailableError) as exc:
        _construct(release)
    assert str(exc.value) == (
        f"{db_path} was replaced or written by another process while this one was "
        "migrating it; every file is left alone. Retry."
    )
    assert os.listdir(root) == ["data.db"]
    assert _sha(db_path) == written[0]


def test_explicit_rebuild_removes_a_stray_non_hot_companion(
    release: dict[str, str],
) -> None:
    """``overwrite=True`` over a file whose only companion is a stray, non-hot -wal:
    nothing is kept and the -wal does not stay beside the new build.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    db_path = Path(release["__genome_root__"]) / "data.db"
    (db_path.parent / "data.db-wal").write_bytes(b"")
    _construct(release, overwrite=True)
    assert os.listdir(release["__genome_root__"]) == ["data.db"]
    _assert_recorded(release, db_path)


def test_explicit_rebuild_waits_for_the_root_lock(release: dict[str, str]) -> None:
    """The explicit rebuild's install runs under the root lock too."""
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    root = Path(release["__genome_root__"])
    before = (root / "data.db").stat().st_ino
    holder = subprocess.Popen(
        [sys.executable, "-c", _ROOT_LOCK_HOLDER, str(root)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline() == "locked\n"
    done: list[int] = []
    worker = threading.Thread(
        target=lambda: done.append(len(_construct(release, overwrite=True).gene_set))
    )
    worker.start()
    worker.join(timeout=3)
    try:
        assert worker.is_alive()
        assert (root / "data.db").stat().st_ino == before
    finally:
        holder.communicate("release\n", timeout=30)
    worker.join(timeout=60)
    assert done == [6]
    assert (root / "data.db").stat().st_ino != before


def test_only_the_rollback_code_triggers_the_committed_peek(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another READONLY extended code (the file was moved) is re-raised as is: the
    committed-record peek runs only for a hot rollback journal.
    """
    db = tmp_path / "data.db"
    db.write_bytes(b"x")
    exc = _sqlite_error(
        sqlite3.SQLITE_READONLY_DBMOVED, "SQLITE_READONLY_DBMOVED", "moved"
    )
    peeks: list[str] = []

    def raising_read(path: str, name: str | None = None) -> str | None:
        raise exc

    monkeypatch.setattr(s288c, "_read_record_json", raising_read)
    monkeypatch.setattr(
        s288c, "_committed_record_json", lambda path: peeks.append(path)
    )
    with pytest.raises(sqlite3.DatabaseError) as raised:
        s288c._read_record_json_checked(str(db))
    assert raised.value is exc
    assert peeks == []


def test_keep_copy_compares_content_not_stat(tmp_path: Path) -> None:
    """An earlier kept copy with the same size and mtime but different bytes is a
    different file: it is replaced.
    """
    copy = tmp_path / "data.db.untrusted.h.1.abc.building"
    copy.write_bytes(b"new bytes")
    kept = tmp_path / "data.db.untrusted"
    kept.write_bytes(b"old bytes")
    os.utime(kept, ns=(copy.stat().st_atime_ns, copy.stat().st_mtime_ns))
    s288c._keep_copy(str(copy), str(kept), str(tmp_path / "data.db"))
    assert kept.read_bytes() == b"new bytes"
    assert not copy.exists()


def test_kept_journal_that_undoes_nothing_stays_trusted_beside_an_own_journal(
    release: dict[str, str],
) -> None:
    """data.db byte-identical to the kept copy, a kept journal present that rolls
    back as a no-op, and a (non-hot) journal of data.db's own: trusted, because the
    rollback leaves the kept copy unchanged. data.db's own journal is not consulted
    (a torn file with one is the held-transaction test below).
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    root = Path(release["__genome_root__"])
    db_path = root / "data.db"
    shutil.copyfile(db_path, root / "data.db.untrusted")
    (root / "data.db.untrusted-journal").write_bytes(b"journal")
    (root / "data.db-journal").write_bytes(b"")
    assert s288c.untrusted_reason(str(db_path), _expected_source(release), "c") is None


def test_unrelated_type_error_at_the_first_read_is_not_renamed(
    release: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only gffutils' empty-meta TypeError becomes the named in-place-rebuild error;
    a TypeError while the meta table has rows propagates unchanged.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    genome = _construct(release)
    manager = genome._db_connection_manager
    assert manager is not None

    def broken_connection() -> Any:
        raise TypeError("unrelated")

    monkeypatch.setattr(manager, "get_connection", broken_connection)
    with pytest.raises(TypeError, match="^unrelated$"):
        genome.db  # noqa: B018


def test_moved_journal_check_needs_the_kept_copy_and_its_bytes(
    release: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A kept journal without a kept file, or a kept file this process cannot read,
    is not the interrupted-migration state: the trusted data.db stays trusted.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    root = Path(release["__genome_root__"])
    db_path = root / "data.db"
    (root / "data.db.untrusted-journal").write_bytes(b"journal")
    source = _expected_source(release)
    assert s288c.untrusted_reason(str(db_path), source, "c") is None
    shutil.copyfile(db_path, root / "data.db.untrusted")

    def unreadable(a: Any, b: Any, shallow: bool = True) -> bool:
        raise PermissionError(13, "Permission denied", str(b))

    monkeypatch.setattr(filecmp, "cmp", unreadable)
    assert s288c.untrusted_reason(str(db_path), source, "c") is None


def test_explicit_rebuild_over_a_cold_journal_keeps_nothing(
    release: dict[str, str],
) -> None:
    """An empty (not hot) journal beside the file: the explicit rebuild removes it
    and keeps nothing.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    db_path = Path(release["__genome_root__"]) / "data.db"
    (db_path.parent / "data.db-journal").write_bytes(b"")
    _construct(release, overwrite=True)
    assert os.listdir(release["__genome_root__"]) == ["data.db"]


def test_explicit_rebuild_refused_at_the_kept_path_leaves_no_temporary(
    release: dict[str, str],
) -> None:
    """The explicit rebuild over a hot journal cannot keep the pair (a directory sits
    at the kept path): refused by name, the file and its journal untouched, no build
    or copy left behind.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    root = Path(release["__genome_root__"])
    db_path = root / "data.db"
    _leave_hot_journal(db_path)
    (root / "data.db.untrusted").mkdir()
    before = (_sha(db_path), _sha(root / "data.db-journal"))
    with pytest.raises(s288c.GenomeDatabaseUnavailableError) as exc:
        _construct(release, overwrite=True)
    assert str(exc.value) == (
        f"{root / 'data.db.untrusted'} is not a regular file, so the untrusted "
        f"{db_path} cannot be kept there; every file is left alone. Remove or rename "
        "it, then retry."
    )
    assert sorted(os.listdir(root)) == [
        "data.db",
        "data.db-journal",
        "data.db.untrusted",
    ]
    assert (_sha(db_path), _sha(root / "data.db-journal")) == before


@pytest.mark.parametrize("overwrite", [False, True])
def test_fresh_build_equal_to_a_kept_pair_that_undoes_nothing_stays_trusted(
    release: dict[str, str],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    overwrite: bool,
) -> None:
    """A hot journal that never reached the file's pages (a writer killed between its
    journal sync and its first page write) is kept with a copy byte-identical to the
    fresh build that replaces it. Rolling the kept pair back changes nothing, so the
    installed build is not a torn file: the next construction trusts it.
    """
    root = _root(release)
    build_db(release[GFF_NAME], root)
    db_path = root / "data.db"
    work = tmp_path / "work.db"
    shutil.copyfile(db_path, work)
    subprocess.run([sys.executable, "-c", _SILENT_HOT_WRITER, str(work)])
    shutil.copyfile(str(work) + "-journal", str(db_path) + "-journal")
    fresh = _sha(db_path)
    _construct(release, overwrite=overwrite)
    assert sorted(os.listdir(root)) == [
        "data.db",
        "data.db.untrusted",
        "data.db.untrusted-journal",
    ]
    assert _sha(db_path) == fresh == _sha(root / "data.db.untrusted")
    assert s288c.untrusted_reason(str(db_path), _expected_source(release), "c") is None
    inode = db_path.stat().st_ino
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=s288c.__name__):
        _construct(release)
    assert caplog.records == []
    assert db_path.stat().st_ino == inode


def test_data_db_vanishing_mid_migration_is_named(
    release: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-2026.10.01 code rebuilding with create_db(force=True) unlinks data.db; a
    migration that finds it gone after its build raises the named error.
    """
    db_path = _old_code_rebuild(release)
    real = s288c.write_genome_database

    def build_then_unlink(*args: Any) -> str:
        built = real(*args)
        os.remove(db_path)
        return built

    monkeypatch.setattr(s288c, "write_genome_database", build_then_unlink)
    with pytest.raises(s288c.GenomeDatabaseUnavailableError) as exc:
        _construct(release)
    assert str(exc.value) == _vanished_message(db_path)
    assert os.listdir(db_path.parent) == []


@pytest.mark.parametrize("piece", ["q?x", "h#x", "p%41x"])
def test_genome_root_with_uri_characters_migrates_and_drops_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, piece: str
) -> None:
    """The migration's row digest and a drop's private copy also open the shared file
    through the encoded URI: an old-code database in such a root is migrated and a
    drop works, and nothing is created outside the root.
    """
    paths = write_release(tmp_path)
    monkeypatch.setattr(
        s288c, "resolve", lambda assembly_set, filename: paths[filename]
    )
    (tmp_path / "go").mkdir()
    (tmp_path / "go" / "go.obo").write_text(GO_OBO)
    parent = tmp_path / "parent"
    root = parent / piece
    root.mkdir(parents=True)
    gffutils.create_db(
        paths[GFF_NAME],
        dbfn=str(root / "data.db"),
        force=True,
        **s288c.CREATE_DB_KWARGS,
    )
    genome = SCerevisiaeGenome(genome_root=str(root), go_root=str(tmp_path / "go"))
    genome.drop_chrmt()
    assert "Q0010" not in genome.gene_set
    assert os.listdir(parent) == [piece]
    assert os.listdir(root) == ["data.db"]


def test_relative_genome_root_opens_migrates_and_drops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``genome_root`` defaults to a RELATIVE path: every read-only open resolves it
    (a relative path cannot be a file: URI).
    """
    paths = write_release(tmp_path)
    monkeypatch.setattr(
        s288c, "resolve", lambda assembly_set, filename: paths[filename]
    )
    (tmp_path / "go").mkdir()
    (tmp_path / "go" / "go.obo").write_text(GO_OBO)
    (tmp_path / "rel").mkdir()
    gffutils.create_db(
        paths[GFF_NAME],
        dbfn=str(tmp_path / "rel" / "data.db"),
        force=True,
        **s288c.CREATE_DB_KWARGS,
    )
    monkeypatch.chdir(tmp_path)
    genome = SCerevisiaeGenome(genome_root="rel", go_root="go")
    genome.drop_chrmt()
    assert "Q0010" not in genome.gene_set
    assert s288c.genome_database_untrusted_reason("rel") is None


def test_symlink_to_a_trusted_database_is_followed(release: dict[str, str]) -> None:
    """A data.db symlink to a trusted database is followed and read; the link stays a
    link and its target is unchanged.
    """
    root = _root(release)
    target_root = root.parent / "target"
    build_db(release[GFF_NAME], target_root)
    target = target_root / "data.db"
    before = _sha(target)
    (root / "data.db").symlink_to(target)
    genome = _construct(release)
    assert list(genome.gene_set) == ALL_GENES
    assert (root / "data.db").is_symlink()
    assert _sha(target) == before


@pytest.mark.parametrize("overwrite", [False, True])
def test_symlink_is_replaced_by_a_regular_file_never_writing_the_target(
    release: dict[str, str], overwrite: bool
) -> None:
    """A data.db symlink whose target is untrusted (default path), or any symlink
    under ``overwrite=True``: the build is renamed onto the LINK, which becomes a
    regular recorded file; the target keeps its bytes.
    """
    root = _root(release)
    target_root = root.parent / "target"
    target_root.mkdir()
    target = target_root / "data.db"
    gffutils.create_db(
        release[GFF_NAME], dbfn=str(target), force=True, **s288c.CREATE_DB_KWARGS
    )
    before = _sha(target)
    (root / "data.db").symlink_to(target)
    _construct(release, overwrite=overwrite)
    assert not (root / "data.db").is_symlink()
    _assert_recorded(release, root / "data.db")
    assert _sha(target) == before
    assert os.listdir(target_root) == ["data.db"]


def test_explicit_rebuild_kept_copy_takes_the_original_mode(
    release: dict[str, str],
) -> None:
    """The pair kept by an explicit rebuild over a hot journal has the original's
    mode (0640), not mkstemp's 0600.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    db_path = Path(release["__genome_root__"]) / "data.db"
    _leave_hot_journal(db_path)
    db_path.chmod(0o640)
    _construct(release, overwrite=True)
    kept = db_path.parent / "data.db.untrusted"
    assert kept.stat().st_mode & 0o777 == 0o640


def test_explicit_rebuild_over_garbage_with_a_cold_journal_keeps_nothing(
    release: dict[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    """Contract: a journal counts as hot only when sqlite reports a rollback is due
    (``SQLITE_READONLY_ROLLBACK``). A journal sqlite itself treats as cold beside a
    file that is not a database is removed by ``overwrite=True`` without a WARNING,
    and nothing is kept. The default path keeps even that pair, with a WARNING.
    A garbage file beside a really hot journal is the next test.
    """
    root = _root(release)
    db_path = root / "data.db"
    garbage = b"\x07 not a database \x00" * 4096
    db_path.write_bytes(garbage)
    (root / "data.db-journal").write_bytes(b"\x00" * 512)
    assert s288c._has_hot_journal(str(db_path)) is False
    with caplog.at_level(logging.WARNING, logger=s288c.__name__):
        _construct(release, overwrite=True)
    assert os.listdir(root) == ["data.db"]
    assert [r.levelname for r in caplog.records] == []
    _assert_recorded(release, db_path)
    db_path.write_bytes(garbage)
    (root / "data.db-journal").write_bytes(b"\x00" * 512)
    with caplog.at_level(logging.WARNING, logger=s288c.__name__):
        _construct(release)
    assert sorted(os.listdir(root)) == [
        "data.db",
        "data.db.untrusted",
        "data.db.untrusted-journal",
    ]
    assert (root / "data.db.untrusted").read_bytes() == garbage
    assert (root / "data.db.untrusted-journal").read_bytes() == b"\x00" * 512
    assert [r.levelname for r in caplog.records] == ["WARNING"]
    _assert_recorded(release, db_path)


@pytest.mark.parametrize("overwrite", [False, True])
def test_garbage_file_beside_a_really_hot_journal_is_kept_with_a_warning(
    release: dict[str, str], caplog: pytest.LogCaptureFixture, overwrite: bool
) -> None:
    """A file whose header was destroyed beside a really hot journal reads
    ``SQLITE_READONLY_ROLLBACK``, not NOTADB: the journal is hot, and on both paths
    the pair is kept as data.db.untrusted with one WARNING.
    """
    root = _root(release)
    build_db(release[GFF_NAME], root)
    db_path = root / "data.db"
    _leave_hot_journal(db_path)
    with open(db_path, "r+b") as fh:
        fh.write(b"\x00" * 100)
    db_bytes = db_path.read_bytes()
    journal_bytes = (root / "data.db-journal").read_bytes()
    assert s288c._has_hot_journal(str(db_path)) is True
    with caplog.at_level(logging.WARNING, logger=s288c.__name__):
        _construct(release, overwrite=overwrite)
    assert sorted(os.listdir(root)) == [
        "data.db",
        "data.db.untrusted",
        "data.db.untrusted-journal",
    ]
    assert (root / "data.db.untrusted").read_bytes() == db_bytes
    assert (root / "data.db.untrusted-journal").read_bytes() == journal_bytes
    assert [r.levelname for r in caplog.records] == ["WARNING"]
    assert "data.db.untrusted" in caplog.records[0].getMessage()
    _assert_recorded(release, db_path)


def test_moved_journal_state_requires_a_kept_journal(release: dict[str, str]) -> None:
    """data.db byte-identical to a kept copy that has NO journal beside it is not the
    interrupted-migration state: trusted.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    root = Path(release["__genome_root__"])
    db_path = root / "data.db"
    shutil.copyfile(db_path, root / "data.db.untrusted")
    assert s288c.untrusted_reason(str(db_path), _expected_source(release), "c") is None


_RAW_V2_WRITER = """
import json, os, sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.isolation_level = None
conn.execute("PRAGMA journal_mode=DELETE")
conn.execute("PRAGMA cache_size=1")
conn.execute("BEGIN")
(raw,) = conn.execute("SELECT record FROM torchcell_genome_db_source").fetchone()
record = json.loads(raw)
record["version"] = 2
conn.execute("UPDATE torchcell_genome_db_source SET record = ?", (json.dumps(record),))
conn.execute("CREATE TABLE spill (x TEXT)")
conn.executemany("INSERT INTO spill VALUES (?)", [("x" * 200,)] * 5000)
os.kill(os.getpid(), 9)
"""


def test_raw_newer_record_over_a_committed_current_one_names_data_db(
    release: dict[str, str],
) -> None:
    """The mirror of the committed-newer case: the committed record is current, but a
    killed writer left version 2 in the raw pages under a hot journal. The copy taken
    for keeping shows version 2; the refusal names data.db, not the private copy.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    root = Path(release["__genome_root__"])
    db_path = root / "data.db"
    subprocess.run([sys.executable, "-c", _RAW_V2_WRITER, str(db_path)])
    with pytest.raises(s288c.GenomeDatabaseVersionError) as exc:
        _construct(release)
    assert str(exc.value) == (
        f"{db_path} carries a record of version 2, written by newer code than this "
        "checkout (which reads version 1); refusing to replace it. Update this "
        "checkout, or resubmit the job from an updated checkout."
    )


def test_moved_journal_check_compares_content_not_stat(release: dict[str, str]) -> None:
    """A kept copy with the same size and mtime as data.db but different bytes, with
    a kept journal: not the interrupted-migration state, so data.db stays trusted.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    root = Path(release["__genome_root__"])
    db_path = root / "data.db"
    other = bytearray(db_path.read_bytes())
    other[100] ^= 0xFF
    kept = root / "data.db.untrusted"
    kept.write_bytes(bytes(other))
    (root / "data.db.untrusted-journal").write_bytes(b"journal")
    st = db_path.stat()
    os.utime(kept, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert s288c.untrusted_reason(str(db_path), _expected_source(release), "c") is None


@pytest.mark.parametrize("change", ["replaced_same_size_and_mtime", "grown_same_mtime"])
def test_identity_check_sees_inode_and_size(
    release: dict[str, str], monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """data.db is replaced by another file of the same size and mtime, or grows in
    place with its mtime restored, between inspection and the lock: either change is
    seen, and the migration is refused by name.
    """
    db_path = _old_code_rebuild(release)
    _old_code_delete(db_path, "Q0010")
    real_copyfile = shutil.copyfile
    done: list[bool] = []

    def copy_then_change(src: str, dst: str) -> Any:
        result = real_copyfile(src, dst)
        if osp.basename(dst).startswith("data.db.untrusted.") and not done:
            st = db_path.stat()
            if change == "replaced_same_size_and_mtime":
                twin = db_path.parent / "twin.db"
                real_copyfile(db_path, twin)
                os.replace(twin, db_path)
            else:
                with open(db_path, "ab") as fh:
                    fh.write(b"\0" * 4096)
            os.utime(db_path, ns=(st.st_atime_ns, st.st_mtime_ns))
            done.append(True)
        return result

    monkeypatch.setattr(shutil, "copyfile", copy_then_change)
    with pytest.raises(s288c.GenomeDatabaseUnavailableError) as exc:
        _construct(release)
    assert str(exc.value) == (
        f"{db_path} was replaced or written by another process while this one was "
        "migrating it; every file is left alone. Retry."
    )


def test_data_db_vanishing_before_the_keep_copy_is_named(
    release: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """data.db is unlinked (old code's create_db(force=True)) after the digest says
    its rows differ and before the copy for keeping: the named error, and nothing
    is kept or installed.
    """
    db_path = _old_code_rebuild(release)
    _old_code_delete(db_path, "Q0010")

    def digest_then_unlink(path: str) -> str | None:
        os.remove(path)
        return None

    monkeypatch.setattr(s288c, "_content_digest_or_none", digest_then_unlink)
    with pytest.raises(s288c.GenomeDatabaseUnavailableError) as exc:
        _construct(release)
    assert str(exc.value) == _vanished_message(db_path)
    assert os.listdir(db_path.parent) == []


def test_rollback_of_an_unreadable_kept_pair_is_not_equal(tmp_path: Path) -> None:
    """When the kept pair cannot be opened at all, the rollback cannot show the
    journal undoes nothing: not equal (the moved-journal state stands).
    """
    kept = tmp_path / "data.db.untrusted"
    kept.write_bytes(b"\x07 not a database \x00" * 4096)
    (tmp_path / "data.db.untrusted-journal").write_bytes(b"\x00" * 512)
    db = tmp_path / "data.db"
    db.write_bytes(kept.read_bytes())
    before = set(os.listdir(tempfile.gettempdir()))
    assert s288c._rollback_equals(str(kept), str(db)) is False
    assert set(os.listdir(tempfile.gettempdir())) == before


_HELD_WRITER = """
import sqlite3, sys
c = sqlite3.connect(sys.argv[1], isolation_level=None)
c.execute("BEGIN IMMEDIATE")
c.execute("UPDATE features SET source = 'zz' WHERE rowid IN (SELECT rowid FROM features LIMIT 2)")
print("held", flush=True)
sys.stdin.readline()
c.execute("ROLLBACK")
"""


def test_torn_file_stays_untrusted_while_an_old_code_writer_holds_a_transaction(
    release: dict[str, str],
) -> None:
    """K2 (journal moved beside the kept copy) while a pre-2026.10.01 writer has an
    open transaction on data.db: its own journal exists but is not hot, and the torn
    file must still be reported (reviewer, eighth round: J4).
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    root = Path(release["__genome_root__"])
    db_path = root / "data.db"
    subprocess.run([sys.executable, "-c", _SILENT_HOT_WRITER, str(db_path)])
    shutil.copyfile(db_path, root / "data.db.untrusted")
    os.replace(root / "data.db-journal", root / "data.db.untrusted-journal")
    moved = (
        "its hot journal was moved beside data.db.untrusted by a migration that did "
        "not finish"
    )
    assert s288c.untrusted_reason(str(db_path), _expected_source(release), "c") == moved
    writer = subprocess.Popen(
        [sys.executable, "-c", _HELD_WRITER, str(db_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert writer.stdout is not None
    try:
        assert writer.stdout.readline() == "held\n"
        assert (root / "data.db-journal").exists()
        assert (
            s288c.untrusted_reason(str(db_path), _expected_source(release), "c")
            == moved
        )
    finally:
        writer.communicate("go\n", timeout=30)


def test_data_db_vanishing_before_the_change_counter_read_is_named(
    release: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trusted reader whose data.db is unlinked by an old-code rebuild between the
    count read and the change-counter read gets the named error, and so does the
    read-only probe ``genome_database_untrusted_reason`` (a vanished file is never
    read as change counter 0, which would report it as written in place).
    """
    root = Path(release["__genome_root__"])
    build_db(release[GFF_NAME], root)
    db_path = root / "data.db"
    real = s288c._change_counter

    def unlink_then_read(path: str) -> int:
        if path == str(db_path):
            os.remove(db_path)
        return real(path)

    monkeypatch.setattr(s288c, "_change_counter", unlink_then_read)
    with pytest.raises(s288c.GenomeDatabaseUnavailableError) as probe:
        s288c.genome_database_untrusted_reason(str(root))
    assert str(probe.value) == _vanished_message(db_path)
    build_db(release[GFF_NAME], root)
    with pytest.raises(s288c.GenomeDatabaseUnavailableError) as exc:
        _construct(release)
    assert str(exc.value) == _vanished_message(db_path)


def test_kept_journal_vanishing_before_the_rollback_is_named(tmp_path: Path) -> None:
    kept = tmp_path / "data.db.untrusted"
    kept.write_bytes(b"x" * 4096)
    db = tmp_path / "data.db"
    db.write_bytes(b"x" * 4096)
    with pytest.raises(s288c.GenomeDatabaseUnavailableError) as exc:
        s288c._rollback_equals(str(kept), str(db))
    assert str(exc.value) == _vanished_message(f"{kept}-journal")


def test_journal_vanishing_before_the_committed_peek_is_named(tmp_path: Path) -> None:
    db = tmp_path / "data.db"
    db.write_bytes(b"x" * 4096)
    with pytest.raises(s288c.GenomeDatabaseUnavailableError) as exc:
        s288c._committed_record_json(str(db))
    assert str(exc.value) == _vanished_message(f"{db}-journal")


def test_kept_pair_rollback_check_runs_in_the_temp_dir_for_a_read_only_root(
    release: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rollback check of the moved-journal state copies the kept pair into the
    temp dir, never into the genome root: a reader that cannot write the root opens
    the trusted file, and nothing is left in the temp dir (reviewer, eighth round:
    V22, V25).
    """
    root = _root(release)
    build_db(release[GFF_NAME], root)
    db_path = root / "data.db"
    work = tmp_path / "work.db"
    shutil.copyfile(db_path, work)
    subprocess.run([sys.executable, "-c", _SILENT_HOT_WRITER, str(work)])
    shutil.copyfile(str(work) + "-journal", str(db_path) + "-journal")
    _construct(release)
    listing = sorted(os.listdir(root))
    assert listing == ["data.db", "data.db.untrusted", "data.db.untrusted-journal"]
    seen: list[str] = []
    real = tempfile.mkstemp

    def spy(*args: Any, **kwargs: Any) -> tuple[int, str]:
        fd, path = real(*args, **kwargs)
        seen.append(path)
        return fd, path

    before = set(os.listdir(tempfile.gettempdir()))
    monkeypatch.setattr(tempfile, "mkstemp", spy)
    os.chmod(root, 0o555)
    try:
        genome = _construct(release)
        assert sorted(f.id for f in genome.db.features_of_type("gene")) == ALL_GENES
    finally:
        os.chmod(root, 0o755)
    assert sorted(os.listdir(root)) == listing
    assert [osp.dirname(p) for p in seen] == [tempfile.gettempdir()]
    name = s288c._PRIVATE_COPY.match(osp.basename(seen[0]))
    assert name is not None
    assert name["pid"] == str(os.getpid())
    assert name["host"] == socket.gethostname()
    assert set(os.listdir(tempfile.gettempdir())) == before


@pytest.mark.parametrize(
    "site",
    [
        "_identity",
        "_copy_preserving_mode",
        "_change_counter",
        "_committed_record_json",
        "_rollback_equals",
        "_rollback_equals_compare",
        "_journal_moved_to_kept",
    ],
)
def test_permission_denied_is_not_relabeled_as_vanished(
    tmp_path: Path, site: str
) -> None:
    """Only ``FileNotFoundError`` is named "vanished": a file this user may not read
    (EACCES) propagates as ``PermissionError`` from every site that names a vanished
    file, never as :class:`GenomeDatabaseUnavailableError`. The one exception is
    ``_journal_moved_to_kept``'s byte comparison, by design: a kept copy this user
    cannot compare (another user's) is not this file's journal pair, so the check
    returns False (not torn) instead of raising.
    """
    locked_dir = tmp_path / "locked"
    locked_dir.mkdir()
    valid = tmp_path / "valid.db"
    with closing(sqlite3.connect(valid)) as conn:
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.commit()
    (tmp_path / "valid.db-journal").write_bytes(b"")
    # data.db has the rolled-back copy's size, so the comparison opens it.
    db = tmp_path / "data.db"
    db.write_bytes(valid.read_bytes())
    (tmp_path / "data.db-journal").write_bytes(b"x" * 512)
    kept = tmp_path / "data.db.untrusted"
    kept.write_bytes(b"x" * 4096)
    (tmp_path / "data.db.untrusted-journal").write_bytes(b"x" * 512)
    calls = {
        "_identity": lambda: s288c._identity(str(locked_dir / "data.db")),
        "_rollback_equals_compare": lambda: s288c._rollback_equals(str(valid), str(db)),
        "_copy_preserving_mode": lambda: s288c._copy_preserving_mode(
            str(db), str(tmp_path / "copy.db")
        ),
        "_change_counter": lambda: s288c._change_counter(str(db)),
        "_committed_record_json": lambda: s288c._committed_record_json(str(db)),
        "_rollback_equals": lambda: s288c._rollback_equals(str(kept), str(db)),
    }
    if site == "_journal_moved_to_kept":
        # Equal sizes, so the comparison opens both files.
        moved = tmp_path / "moved"
        moved.mkdir()
        (moved / "data.db").write_bytes(b"x" * 4096)
        (moved / "data.db.untrusted").write_bytes(b"x" * 4096)
        (moved / "data.db.untrusted-journal").write_bytes(b"x" * 512)
        (moved / "data.db").chmod(0)
        try:
            assert s288c._journal_moved_to_kept(str(moved / "data.db")) is False
        finally:
            (moved / "data.db").chmod(0o700)
        return
    locked = (
        locked_dir
        if site == "_identity"
        else kept
        if site == "_rollback_equals"
        else db
    )
    locked.chmod(0)
    try:
        with pytest.raises(PermissionError) as exc:
            calls[site]()
    finally:
        locked.chmod(0o700)
    assert exc.value.errno == errno.EACCES


def test_io_error_at_the_kept_byte_comparison_is_not_relabeled_as_vanished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An I/O error (EIO) while ``_journal_moved_to_kept`` compares data.db with the
    kept copy propagates as itself; only ``FileNotFoundError`` is named "vanished"
    (reviewer, ninth round: D10).
    """
    (tmp_path / "data.db").write_bytes(b"x" * 4096)
    (tmp_path / "data.db.untrusted").write_bytes(b"x" * 4096)
    (tmp_path / "data.db.untrusted-journal").write_bytes(b"x" * 512)
    failure = OSError(errno.EIO, "Input/output error")

    def failing_cmp(a: str, b: str, shallow: bool = True) -> bool:
        raise failure

    monkeypatch.setattr(s288c, "filecmp", SimpleNamespace(cmp=failing_cmp))
    with pytest.raises(OSError) as exc:
        s288c._journal_moved_to_kept(str(tmp_path / "data.db"))
    assert exc.value is failure


def test_data_db_vanishing_before_the_kept_byte_comparison_is_named(
    tmp_path: Path,
) -> None:
    """A kept pair sits in the root and data.db has been unlinked (old code's
    create_db(force=True)) before the byte comparison: the named error.
    """
    (tmp_path / "data.db.untrusted").write_bytes(b"x" * 4096)
    (tmp_path / "data.db.untrusted-journal").write_bytes(b"x" * 512)
    db = tmp_path / "data.db"
    with pytest.raises(s288c.GenomeDatabaseUnavailableError) as exc:
        s288c._journal_moved_to_kept(str(db))
    assert str(exc.value) == _vanished_message(db)


def test_data_db_vanishing_before_the_rollback_comparison_is_named(
    tmp_path: Path,
) -> None:
    """The kept pair rolls back in its private copy, then data.db is gone before the
    copy is compared with it: the named error, and the private copy is removed.
    """
    kept = tmp_path / "data.db.untrusted"
    with closing(sqlite3.connect(kept)) as conn:
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.commit()
    (tmp_path / "data.db.untrusted-journal").write_bytes(b"")
    db = tmp_path / "data.db"
    before = set(os.listdir(tempfile.gettempdir()))
    with pytest.raises(s288c.GenomeDatabaseUnavailableError) as exc:
        s288c._rollback_equals(str(kept), str(db))
    assert str(exc.value) == _vanished_message(db)
    assert set(os.listdir(tempfile.gettempdir())) == before


@pytest.mark.parametrize("stage", ["unlinked", "zero_bytes", "meta_empty"])
def test_first_write_copies_the_file_this_instance_reads_during_an_old_code_rebuild(
    release: dict[str, str], stage: str, private_tmp: Path
) -> None:
    """A pre-2026.10.01 rebuild that starts in place after this instance read the shared
    file (unlinked it, created it empty, or filled its tables before ``meta``) does not
    change what this instance's first write copies: the copy comes from the connection
    this instance reads, so ``drop_chrmt`` leaves exactly the five non-chrmt genes.
    """
    build_db(release[GFF_NAME], _root(release))
    genome = _construct(release)
    assert "Q0010" in genome.gene_set
    db_path = _root(release) / "data.db"
    os.remove(db_path)
    if stage == "zero_bytes":
        db_path.write_bytes(b"")
    elif stage == "meta_empty":
        _old_code_rebuild(release)
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM meta")
        conn.commit()
        conn.close()
    listing = sorted(os.listdir(db_path.parent))
    genome.drop_chrmt()
    expected = ["YAL001C", "YAL002W", "YBL001W", "YBL002W", "YCL001W"]
    assert list(genome.gene_set) == expected
    assert [f.id for f in genome.db.features_of_type("gene")] == expected
    assert sorted(os.listdir(db_path.parent)) == listing
    private = genome._private_db_path
    assert private is not None
    assert os.listdir(private_tmp) == [osp.basename(private)]


def test_migrating_a_torn_file_under_an_old_code_transaction_keeps_the_moved_journal(
    release: dict[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    """K2 while a pre-2026.10.01 writer holds a transaction on the torn data.db: the
    migration installs a recorded build and leaves the kept pair as it was. The
    writer's own journal (which undoes nothing on those bytes) does not overwrite
    ``data.db.untrusted-journal``, so the kept pair still rolls back to the committed
    file. The writer's journal is removed, and the migration's one WARNING names it.
    """
    build_db(release[GFF_NAME], Path(release["__genome_root__"]))
    root = Path(release["__genome_root__"])
    db_path = root / "data.db"
    committed = db_path.read_bytes()
    subprocess.run([sys.executable, "-c", _SILENT_HOT_WRITER, str(db_path)])
    torn = db_path.read_bytes()
    assert torn != committed
    shutil.copyfile(db_path, root / "data.db.untrusted")
    os.replace(root / "data.db-journal", root / "data.db.untrusted-journal")
    moved_journal = (root / "data.db.untrusted-journal").read_bytes()
    writer = subprocess.Popen(
        [sys.executable, "-c", _HELD_WRITER, str(db_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert writer.stdout is not None
    try:
        assert writer.stdout.readline() == "held\n"
        with caplog.at_level(logging.WARNING, logger=s288c.__name__):
            _construct(release)
        assert [r.getMessage() for r in caplog.records] == [
            f"genome database {db_path} was not trusted (its hot journal was moved "
            "beside data.db.untrusted by a migration that did not finish); its own "
            "pages match its record and only a companion journal made it unreadable, "
            f"so the pair was kept as {root / 'data.db.untrusted'} (which already held "
            f"these bytes and was left in place; {db_path}-journal beside data.db was "
            "removed instead of overwriting the kept companion) and replaced by the "
            "recorded build"
        ]
        assert sorted(p.name for p in root.iterdir()) == [
            "data.db",
            "data.db.untrusted",
            "data.db.untrusted-journal",
        ]
        assert (root / "data.db.untrusted").read_bytes() == torn
        assert (root / "data.db.untrusted-journal").read_bytes() == moved_journal
        committed_file = root.parent / "committed.db"
        committed_file.write_bytes(committed)
        assert s288c._rollback_equals(
            str(root / "data.db.untrusted"), str(committed_file)
        )
        assert (
            s288c.untrusted_reason(str(db_path), _expected_source(release), "c") is None
        )
    finally:
        writer.communicate("go\n", timeout=30)


def test_replay_with_the_private_copy_and_data_db_gone_is_named_and_leaves_no_copy(
    release: dict[str, str], private_tmp: Path
) -> None:
    """An unpickled instance whose parent's private copy was collected copies data.db
    at its first read; when old code has also unlinked data.db, the read raises the
    named error and leaves nothing in the temp dir.
    """
    build_db(release[GFF_NAME], _root(release))
    genome = _construct(release)
    genome.drop_chrmt()
    restored = pickle.loads(pickle.dumps(genome))
    del genome
    gc.collect()
    assert os.listdir(private_tmp) == []
    db_path = _root(release) / "data.db"
    os.remove(db_path)
    with pytest.raises(s288c.GenomeDatabaseUnavailableError) as exc:
        restored.db
    assert str(exc.value) == (
        f"{db_path} cannot be opened by this process (SQLITE_CANTOPEN: unable to open "
        "database file; mode absent); every file is left alone. Fix its permissions "
        "or ownership, then retry."
    )
    assert os.listdir(private_tmp) == []
    assert os.listdir(db_path.parent) == []


@pytest.mark.parametrize("stage", ["zero_bytes", "meta_empty"])
def test_replay_from_a_data_db_without_metadata_is_named_and_leaves_no_copy(
    release: dict[str, str], private_tmp: Path, stage: str
) -> None:
    """The replay copies a data.db that old code is rebuilding in place (created
    empty, so the copy has no ``meta`` table, or its ``meta`` table still empty): the
    copy is removed and the named error raised, and the shared file keeps its bytes.
    """
    build_db(release[GFF_NAME], _root(release))
    genome = _construct(release)
    genome.drop_chrmt()
    restored = pickle.loads(pickle.dumps(genome))
    del genome
    gc.collect()
    db_path = _root(release) / "data.db"
    if stage == "zero_bytes":
        os.remove(db_path)
        db_path.write_bytes(b"")
    else:
        _old_code_rebuild(release)
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute("DELETE FROM meta")
            conn.commit()
    before = (_identity(db_path), _sha(db_path))
    with pytest.raises(s288c.GenomeDatabaseUnavailableError) as exc:
        restored.db
    assert str(exc.value) == (
        f"{db_path} has no gffutils metadata: another process (pre-2026.10.01 code) "
        "is rebuilding it in place; nothing was changed. Retry when it has finished."
    )
    assert os.listdir(private_tmp) == []
    assert (_identity(db_path), _sha(db_path)) == before


def test_a_failed_lazy_first_write_leaves_no_copy(
    release: dict[str, str], private_tmp: Path
) -> None:
    """The first write is also this instance's first read (``remove_deprecated_go_terms``
    on a never-read instance) and old code has unlinked data.db: the connection's
    ``FileNotFoundError`` propagates (the lazy first read, which the named errors do
    not cover) and the copy made for the write is removed.
    """
    build_db(release[GFF_NAME], _root(release))
    genome = _construct(release)
    db_path = _root(release) / "data.db"
    os.remove(db_path)
    with pytest.raises(FileNotFoundError) as exc:
        genome.remove_deprecated_go_terms()
    assert str(exc.value) == f"Database not found at {db_path}"
    assert os.listdir(private_tmp) == []
    assert genome._private_db_path is None
    assert genome._db_writes == []


def test_keep_copy_replaces_a_kept_file_removed_before_its_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The kept file is removed by hand between ``_keep_copy``'s file check and its
    byte comparison: it counts as a different file, the stale kept journal is
    dropped and the copy becomes the kept file (reviewer, ninth round: P8).
    """
    copy = tmp_path / "data.db.untrusted.h.1.abc.building"
    copy.write_bytes(b"torn bytes")
    kept = tmp_path / "data.db.untrusted"
    kept.write_bytes(b"torn bytes")
    (tmp_path / "data.db.untrusted-journal").write_bytes(b"stale journal")
    real = filecmp.cmp

    def remove_then_compare(a: str, b: str, shallow: bool = True) -> bool:
        os.remove(kept)
        return real(a, b, shallow=shallow)

    monkeypatch.setattr(s288c, "filecmp", SimpleNamespace(cmp=remove_then_compare))
    assert s288c._keep_copy(str(copy), str(kept), str(tmp_path / "data.db")) is False
    assert sorted(os.listdir(tmp_path)) == ["data.db.untrusted"]
    assert kept.read_bytes() == b"torn bytes"


def test_keep_copy_reports_whether_it_left_the_kept_pair(tmp_path: Path) -> None:
    """Equal bytes: the kept pair stays (True). Different bytes: the copy replaces the
    kept file and its companions go (False).
    """
    kept = tmp_path / "data.db.untrusted"
    kept.write_bytes(b"torn bytes")
    (tmp_path / "data.db.untrusted-journal").write_bytes(b"kept journal")
    same = tmp_path / "same.building"
    same.write_bytes(b"torn bytes")
    assert s288c._keep_copy(str(same), str(kept), str(tmp_path / "data.db")) is True
    assert same.read_bytes() == b"torn bytes"
    assert (tmp_path / "data.db.untrusted-journal").read_bytes() == b"kept journal"
    other = tmp_path / "other.building"
    other.write_bytes(b"other bytes")
    assert s288c._keep_copy(str(other), str(kept), str(tmp_path / "data.db")) is False
    assert sorted(os.listdir(tmp_path)) == ["data.db.untrusted", "same.building"]
    assert kept.read_bytes() == b"other bytes"


@pytest.mark.parametrize("kept_as_is", [False, True])
@pytest.mark.parametrize("kept_companion", [False, True])
def test_install_beside_a_kept_pair_never_overwrites_a_kept_companion_left_in_place(
    tmp_path: Path, kept_as_is: bool, kept_companion: bool
) -> None:
    """data.db's journal is moved beside the kept copy, except when the kept copy was
    left in place (``kept_as_is``) and already has a journal: then data.db's is
    removed, returned for the WARNING, and the kept journal stays.
    """
    build = tmp_path / "build.db"
    build.write_bytes(b"build")
    db = tmp_path / "data.db"
    db.write_bytes(b"torn")
    (tmp_path / "data.db-journal").write_bytes(b"own journal")
    keep = tmp_path / "data.db.untrusted"
    keep.write_bytes(b"torn")
    if kept_companion:
        (tmp_path / "data.db.untrusted-journal").write_bytes(b"kept journal")
    removed = s288c.install_genome_database(str(build), str(db), str(keep), kept_as_is)
    assert db.read_bytes() == b"build"
    assert sorted(os.listdir(tmp_path)) == [
        "data.db",
        "data.db.untrusted",
        "data.db.untrusted-journal",
    ]
    stays = kept_as_is and kept_companion
    assert removed == ([f"{db}-journal"] if stays else [])
    assert (tmp_path / "data.db.untrusted-journal").read_bytes() == (
        b"kept journal" if stays else b"own journal"
    )


def test_explicit_rebuild_beside_an_equal_kept_pair_names_the_removed_journal(
    release: dict[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    """``overwrite=True`` over a file with a hot journal whose bytes and journal a
    kept pair already holds: the kept pair stays as it was, data.db's journal is
    removed, and the one WARNING names both.
    """
    root = Path(release["__genome_root__"])
    build_db(release[GFF_NAME], root)
    db_path = root / "data.db"
    subprocess.run([sys.executable, "-c", _SILENT_HOT_WRITER, str(db_path)])
    shutil.copyfile(db_path, root / "data.db.untrusted")
    shutil.copyfile(root / "data.db-journal", root / "data.db.untrusted-journal")
    kept = (_sha(root / "data.db.untrusted"), _sha(root / "data.db.untrusted-journal"))
    with caplog.at_level(logging.WARNING, logger=s288c.__name__):
        _construct(release, overwrite=True)
    assert [r.getMessage() for r in caplog.records] == [
        f"genome database {db_path} had a hot journal; the explicit rebuild kept the "
        f"old file with its journal as {root / 'data.db.untrusted'} (which already "
        f"held these bytes and was left in place; {db_path}-journal beside data.db "
        "was removed instead of overwriting the kept companion) and installed the "
        "recorded build"
    ]
    assert sorted(os.listdir(root)) == [
        "data.db",
        "data.db.untrusted",
        "data.db.untrusted-journal",
    ]
    assert (
        _sha(root / "data.db.untrusted"),
        _sha(root / "data.db.untrusted-journal"),
    ) == kept
    assert s288c.untrusted_reason(str(db_path), _expected_source(release), "c") is None


def test_replay_from_a_damaged_data_db_is_named_and_leaves_no_copy(
    release: dict[str, str], private_tmp: Path
) -> None:
    """The replay copies a data.db sqlite reports as not a database: the copy is
    removed and the named error raised, and the shared file keeps its bytes.
    """
    build_db(release[GFF_NAME], _root(release))
    genome = _construct(release)
    genome.drop_chrmt()
    restored = pickle.loads(pickle.dumps(genome))
    del genome
    gc.collect()
    db_path = _root(release) / "data.db"
    os.remove(db_path)
    db_path.write_bytes(b"\x07 not a database \x00" * 4096)
    before = (_identity(db_path), _sha(db_path))
    with pytest.raises(s288c.GenomeDatabaseUnavailableError) as exc:
        restored.db
    assert str(exc.value) == (
        f"{db_path} cannot be copied for this instance's writes (file is not a "
        "database); every file is left alone. Retry when the process rebuilding it "
        "has finished."
    )
    assert os.listdir(private_tmp) == []
    assert (_identity(db_path), _sha(db_path)) == before


def test_unpickled_first_write_copies_the_writers_copy_not_the_rewritten_shared_file(
    release: dict[str, str], private_tmp: Path
) -> None:
    """An unpickled instance reads its writer's private copy. When pre-2026.10.01 code
    rewrites the shared file in place after the unpickle, the unpickled instance's
    first write still copies the writer's copy (no replay from the shared file), so a
    gene old code deleted from the shared file stays in this instance.
    """
    build_db(release[GFF_NAME], _root(release))
    genome = _construct(release)
    genome.drop_chrmt()
    clone = pickle.loads(pickle.dumps(genome))
    _old_code_delete(_root(release) / "data.db", "YAL001C")
    clone.drop_empty_go()
    remaining = [f.id for f in clone.db.features_of_type("gene")]
    assert "YAL001C" in remaining
    assert "Q0010" not in remaining
    assert clone._private_db_path != genome._private_db_path


def test_rows_differ_migration_beside_an_equal_kept_copy_says_it_was_left_in_place(
    release: dict[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    """A migration killed after keeping a file whose rows differ leaves the kept copy
    byte-equal to data.db; the next migration leaves it in place and its WARNING says so.
    """
    build_db(release[GFF_NAME], _root(release))
    db_path = _root(release) / "data.db"
    _old_code_delete(db_path, "Q0010")
    kept = _root(release) / "data.db.untrusted"
    shutil.copyfile(db_path, kept)
    with caplog.at_level(logging.WARNING, logger=s288c.__name__):
        _construct(release)
    assert [r.getMessage() for r in caplog.records] == [
        f"genome database {db_path} was not trusted (its row counts differ from its "
        "record (features 15 vs 16 recorded, relations 8 vs 8 recorded)); its rows "
        f"differ from a fresh build, so it was kept as {kept} (which already held "
        "these bytes and was left in place) and replaced by the recorded build"
    ]


def test_an_interrupted_first_write_leaves_no_copy(
    release: dict[str, str], private_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A KeyboardInterrupt during the first write's copy propagates and removes the copy."""
    build_db(release[GFF_NAME], _root(release))
    genome = _construct(release)
    assert "Q0010" in genome.gene_set

    def interrupt(path: str) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(s288c, "_meta_rows_or_zero", interrupt)
    with pytest.raises(KeyboardInterrupt):
        genome.drop_chrmt()
    assert os.listdir(private_tmp) == []
    assert genome._private_db_path is None
