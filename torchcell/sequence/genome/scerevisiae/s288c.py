# torchcell/sequence/genome/scerevisiae/s288c
# [[torchcell.sequence.genome.scerevisiae.s288c]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/sequence/genome/scerevisiae/s288c
# Test file: tests/torchcell/sequence/genome/scerevisiae/test_s288c.py

"""S. cerevisiae S288C genome access over SGD FASTA/GFF with GO and sequence windows."""

import hashlib
import json
import logging
import os
import os.path as osp
import re
import secrets
import shutil
import socket
import sqlite3
import stat
import tempfile
import weakref
from enum import StrEnum
from itertools import product
from pathlib import Path
from typing import Any, ClassVar, SupportsIndex, cast

import gffutils
import pandas as pd
from attrs import define, field
from Bio import SeqIO
from gffutils.feature import Feature
from goatools.obo_parser import GODag
from pydantic import BaseModel, ConfigDict, Field
from sortedcontainers import SortedDict, SortedSet
from torch_geometric.data import download_url

from torchcell.literature.manifest import sha256_file
from torchcell.sequence import (
    DnaSelectionResult,
    DnaWindowResult,
    Gene,
    GeneSet,
    Genome,
    calculate_window_bounds,
    calculate_window_bounds_symmetric,
    compute_codon_frequency,
    get_chr_from_description,
    roman_to_int,
)
from torchcell.sequence.db_connection import GffutilsConnectionManager
from torchcell.sequence.genome.registry import SGD_S288C_R64, resolve

log = logging.getLogger(__name__)

# We put MT at 0, because it is circular, and this preserves arabic to roman
CHROMOSOMES = [
    "chrmt",
    "chrI",
    "chrII",
    "chrIII",
    "chrIV",
    "chrV",
    "chrVI",
    "chrVII",
    "chrVIII",
    "chrIX",
    "chrX",
    "chrXI",
    "chrXII",
    "chrXIII",
    "chrXIV",
    "chrXV",
    "chrXVI",
]


nucleotides = ["A", "T", "G", "C"]
all_codons = ["".join(codon) for codon in product(nucleotides, repeat=3)]


@define
class SCerevisiaeGene(Gene):
    """A single S. cerevisiae gene resolved from the SGD GFF database and FASTA files."""

    id: str = field(repr=False)
    db: Any = field(repr=False)
    fasta_dna: dict[str, Any] = field(repr=False)
    fasta_protein: dict[str, Any] = field(repr=False)
    fasta_cds: dict[str, Any] = field(repr=False)
    chr_to_nc: dict[int, str] = field(repr=False)
    chromosome_lengths: dict[int, int] = field(repr=False)
    # below are set in __attrs_post_init__
    chromosome: int = field(default=None)
    start: int = field(default=None)
    end: int = field(default=None)
    seq: str = field(default=None, repr=True)
    feature: Feature = field(default=None, repr=False)

    def __attrs_post_init__(self) -> None:
        """Resolve the gene feature, coordinates, sequence, and GO terms from the DB."""
        # process the feature region and produce a feature
        feature_region = self.db.region(
            region=(
                self.db[self.id].chrom,
                self.db[self.id].start,
                self.db[self.id].end,
            ),
            completely_within=True,
        )
        features = [feature for feature in feature_region]
        contains_five_prime_UTR_intron = False
        no_middle_intron = True
        not_chrmt = self.db[self.id].chrom != "chrmt"
        for some_feature in features:
            if some_feature.featuretype == "five_prime_UTR_intron":
                contains_five_prime_UTR_intron = True
                five_prime_UTR_intron_feature = some_feature

        # 4 genes with single bp CDS 5prime
        no_five_prime_one_bp_cds = True
        for some_feature in features:
            if some_feature.featuretype == "CDS":
                cds_difference = some_feature.start - some_feature.end
                if (
                    some_feature.strand == "+"
                    and cds_difference == 0
                    and self.db[self.id].start == some_feature.start
                ):
                    no_five_prime_one_bp_cds = False

                elif (
                    some_feature.strand == "-"
                    and cds_difference == 0
                    and self.db[self.id].end == some_feature.end
                ):
                    no_five_prime_one_bp_cds = False

            self.db[self.id].start
        # No guarantee introns are same as five_prime_UTR_intron
        if contains_five_prime_UTR_intron:
            if (
                five_prime_UTR_intron_feature.start > self.db[self.id].start
                and five_prime_UTR_intron_feature.end < self.db[self.id].end
            ):
                no_middle_intron = False

        # TODO logic is bit complicated, might want to abstract away.
        if (
            contains_five_prime_UTR_intron
            and no_middle_intron
            and not_chrmt
            and no_five_prime_one_bp_cds
        ):
            cds_features = [
                feature for feature in features if feature.featuretype == "CDS"
            ]
            if len(cds_features) == 0:
                raise ValueError(
                    f"Gene {self.id} has a five_prime_UTR_intron but no CDS feature"
                )
            if len(cds_features) == 1:
                feature = cds_features[0]
            # sometimes we have more than one CDS, we need to select the one we have most confidence in with "Verified" ORF
            else:
                for cds in cds_features:
                    # A ValueError, not KeyError: __getitem__ reads KeyError as
                    # "gene not found", which would hide the malformed CDS.
                    if "orf_classification" not in cds.attributes:
                        raise ValueError(
                            f"Gene {self.id}: CDS {cds.id} at {cds.start}..{cds.end} "
                            "has no orf_classification attribute"
                        )
                verified_orfs = [
                    feature
                    for feature in cds_features
                    if feature.attributes["orf_classification"][0] == "Verified"
                ]
                if len(verified_orfs) == 0:
                    raise ValueError(
                        f"Gene {self.id} has a five_prime_UTR_intron and "
                        f"{len(cds_features)} CDS features, none Verified"
                    )
                if len(verified_orfs) == 1:
                    feature = verified_orfs[0]
                if len(verified_orfs) > 1:
                    feature = Feature()
                    feature.chrom = self.db[self.id].chrom
                    feature.strand = self.db[self.id].strand
                    feature.start = min([feature.start for feature in verified_orfs])
                    feature.end = max([feature.end for feature in verified_orfs])
            assert isinstance(feature, Feature), "feature is not a gffutils Feature"
            # log.warning(f"{self.id} - Using CDS Sequence")
        else:
            feature = self.db[self.id]
        gene_feature = self.db[self.id]

        #
        self.id = self.id
        # chromosome
        seqid = gene_feature.seqid
        maybe_roman_numeral = seqid.split("chr")[-1]
        if maybe_roman_numeral == "mt":
            self.chromosome = 0
        else:
            self.chromosome = roman_to_int(maybe_roman_numeral)
        # others
        self.start = feature.start
        self.end = feature.end
        self.strand = feature.strand
        if self.strand not in ("+", "-"):
            raise ValueError(
                f"Gene {self.id} has strand {self.strand!r}; a gene needs '+' or '-' "
                "to orient its sequence and windows"
            )

        # dna sequence
        chr = self.chr_to_nc[self.chromosome]
        if self.strand == "+":
            self.seq = str(self.fasta_dna[chr].seq[self.start - 1 : self.end])
        elif self.strand == "-":
            self.seq = str(
                self.fasta_dna[chr].seq[self.start - 1 : self.end].reverse_complement()
            )

        # protein sequence
        self.protein = self.fasta_protein.get(self.id)
        # cds sequence
        self.cds: Any = self.fasta_cds.get(self.id)

        # TODO consider adding these to ABC...
        # Some might be too specific to S. cerevisiae, but so they could be optional
        # Must use the gene since it has all of the annotations, not the OverflowError
        self.alias = gene_feature.attributes.get("Alias", None)
        self.name = gene_feature.attributes.get("Name", None)
        self.ontology_term = gene_feature.attributes.get("Ontology_term", None)
        self.note = gene_feature.attributes.get("Note", None)
        self.display = gene_feature.attributes.get("display", None)
        self.dbxref = gene_feature.attributes.get("dbxref", None)
        self.orf_classification = gene_feature.attributes.get(
            "orf_classification", None
        )

        # Handle GO terms
        if self.ontology_term is not None:
            self.go = SortedSet(
                [term for term in self.ontology_term if term.startswith("GO:")]
            )
        else:
            self.go = None

    @property
    def alias_to_systematic(self) -> dict[str, str]:
        """Return a mapping from each gene alias to its systematic gene ID."""
        alias_map = {}
        for gene_id in self.gene_set:  # type: ignore[attr-defined]  # references Genome API on Gene (pre-existing)
            gene = self[gene_id]  # type: ignore[index]  # references Genome API on Gene (pre-existing)
            if gene and gene.alias:
                for alias in gene.alias:
                    if alias in alias_map:
                        log.warning(
                            f"Duplicate alias {alias} mapped to multiple genes."
                        )
                    alias_map[alias] = gene_id
        return alias_map

    @property
    def codon_frequency(self) -> SortedDict[str, float]:
        """Return the codon-usage frequency of the gene's CDS sequence."""
        codon_frequency = compute_codon_frequency(self.cds.seq)
        return codon_frequency

    def window(self, window_size: int, is_max_size: bool = True) -> DnaWindowResult:
        """Return a DNA sequence window centered on the gene's coding region."""
        if is_max_size:
            start_window, end_window = calculate_window_bounds(
                start=self.start - 1,
                end=self.end,
                strand=self.strand,
                window_size=window_size,
                chromosome_length=self.chromosome_lengths[self.chromosome],
            )

        else:
            start_window, end_window = calculate_window_bounds_symmetric(
                start=self.start - 1,
                end=self.end,
                window_size=window_size,
                chromosome_length=self.chromosome_lengths[self.chromosome],
            )
        chr_id = self.chr_to_nc[self.chromosome]
        if self.strand == "+":
            seq = str(self.fasta_dna[chr_id].seq[start_window:end_window])
        elif self.strand == "-":
            seq = str(
                self.fasta_dna[chr_id].seq[start_window:end_window].reverse_complement()
            )
        return DnaWindowResult(
            id=self.id,
            chromosome=self.chromosome,
            strand=self.strand,
            start=self.start,
            end=self.end,
            seq=seq,
            start_window=start_window,
            end_window=end_window,
        )

    def window_five_prime(  # type: ignore[override]  # adds include_start_codon kwarg vs base ABC
        self,
        window_size: int,
        include_start_codon: bool = False,
        allow_undersize: bool = False,
    ) -> DnaWindowResult:
        """Return the sequence window upstream of the gene's 5' start.

        The window never includes a base of the gene unless ``include_start_codon``
        is set, in which case it ends with the 3 bases of the start codon. On ``+``
        it is ``[start0 - w, start0)`` (``[start0 + 3 - w, start0 + 3)`` with the
        codon), where ``start0 = self.start - 1`` is the 0-based first base; on ``-``
        it is the reverse complement of ``[end, end + w)`` (``[end - 3, end - 3 + w)``
        with the codon).
        """
        # offset for gff file 1
        start = self.start - 1
        chr_id = self.chr_to_nc[self.chromosome]
        if self.strand == "+":
            if include_start_codon:
                start = start + 3
            start_window = start - window_size
            end_window = start
            if start_window < 0 and allow_undersize:
                start_window = 0
                end_window = start
            elif start_window < 0 and not allow_undersize:
                outside = abs(start_window)
                raise ValueError(
                    f"five prime size ({window_size}) too large ('{self.strand} strand {outside}bp outside.)"
                )
            seq = str(self.fasta_dna[chr_id].seq[start_window:end_window])
        elif self.strand == "-":
            if include_start_codon:
                end = self.end - 3
            else:
                end = self.end
            start_window = end
            end_window = end + window_size
            if (
                end_window > self.chromosome_lengths[self.chromosome]
                and allow_undersize
            ):
                end_window = self.chromosome_lengths[self.chromosome]
            elif (
                end_window > self.chromosome_lengths[self.chromosome]
                and not allow_undersize
            ):
                outside = abs(end_window - self.chromosome_lengths[self.chromosome])
                raise ValueError(
                    f"five prime size ({window_size}) too large ('{self.strand} strand {outside}bp outside.)"
                )

            seq = str(
                self.fasta_dna[chr_id].seq[start_window:end_window].reverse_complement()
            )
        return DnaWindowResult(
            id=self.id,
            seq=seq,
            chromosome=self.chromosome,
            start=self.start,
            end=self.end,
            strand=self.strand,
            start_window=start_window,
            end_window=end_window,
        )

    def window_three_prime(  # type: ignore[override]  # adds include_stop_codon kwarg vs base ABC
        self,
        window_size: int,
        include_stop_codon: bool = False,
        allow_undersize: bool = False,
    ) -> DnaWindowResult:
        """Return the sequence window downstream of the gene's 3' end."""
        # offset for gff file 1
        start = self.start - 1
        chr_id = self.chr_to_nc[self.chromosome]
        if self.strand == "+":
            if include_stop_codon:
                end = self.end - 3
            else:
                end = self.end
            start_window = end
            end_window = end + window_size
            if (
                end_window > self.chromosome_lengths[self.chromosome]
                and allow_undersize
            ):
                end_window = self.chromosome_lengths[self.chromosome]
            elif (
                end_window > self.chromosome_lengths[self.chromosome]
                and not allow_undersize
            ):
                outside = abs(end_window - self.chromosome_lengths[self.chromosome])
                raise ValueError(
                    f"3utr size ({window_size}) too large"
                    f"('{self.strand} strand {outside}bp outside.)"
                )
            seq = str(self.fasta_dna[chr_id].seq[start_window:end_window])
        elif self.strand == "-":
            if include_stop_codon:
                start = start + 3
            else:
                start = start
            start_window = start - window_size
            end_window = start
            if start_window < 0 and allow_undersize:
                start_window = 0
            elif start_window < 0 and not allow_undersize:
                outside = abs(start_window)
                raise ValueError(
                    f"3utr size ({window_size}) too large"
                    f"('{self.strand} strand {outside}bp outside.)"
                )
            seq = str(
                self.fasta_dna[chr_id].seq[start_window:end_window].reverse_complement()
            )

        return DnaWindowResult(
            id=self.id,
            seq=seq,
            chromosome=self.chromosome,
            start=self.start,
            end=self.end,
            strand=self.strand,
            start_window=start_window,
            end_window=end_window,
        )

    def __repr__(self) -> str:
        """Return a string with the gene's ID, location, strand, and sequence."""
        return f"DnaSelectionResult(id={self.id}, chromosome={self.chromosome}, strand={self.strand}, start={self.start}, end={self.end},  seq={self.seq})"


# Gene-like LOCUS feature types in the SGD R64 GFF (a deletion/perturbation can target
# these). "gene" is the protein-coding-ORF universe (== gene_set); the rest are RNA genes,
# transposon genes, pseudogenes, and blocked_reading_frame pseudogenes. Every OTHER GFF
# featuretype (region, CDS, mRNA, ARS, intron, telomere, ...) is deliberately excluded so a
# non-locus feature id can never shadow a real gene name during resolution.
_LOCUS_FEATURE_TYPES = frozenset(
    {
        "gene",
        "tRNA_gene",
        "snoRNA_gene",
        "snRNA_gene",
        "ncRNA_gene",
        "rRNA_gene",
        "telomerase_RNA_gene",
        "transposable_element_gene",
        "pseudogene",
        "blocked_reading_frame",
    }
)


class GeneNameStatus(StrEnum):
    """Outcome of resolving a source gene name against the current R64 annotation.

    The layered resolver (:meth:`SCerevisiaeGenome.resolve_gene_name`) reconciles a
    dataset's source gene name -- systematic (e.g. Ohya 2005) or common (e.g. Cachera) --
    to the current R64-4-1 annotation. Datasets carry historical names: SGD renames/merges
    features and retires dubious ORFs, so a 2005-era systematic name may no longer be a
    live "gene". This status tells the loader what the name resolved to so it can RETAIN
    the record (a real strain / perturbation) with the correct identifier and provenance,
    rather than silently dropping it.
    """

    CURRENT = "current"  # a live R64 "gene" feature (systematic_name == input)
    RENAMED = (
        "renamed"  # alias of exactly one current gene (systematic_name = that gene)
    )
    NON_GENE_FEATURE = "non_gene_feature"  # a valid R64 feature that is not a "gene"
    RETIRED = "retired"  # not present in R64-4-1 at all; retained as a legacy name
    AMBIGUOUS = "ambiguous"  # alias mapping to >1 current feature; needs human review


class GeneNameResolution(BaseModel):
    """Typed result of :meth:`SCerevisiaeGenome.resolve_gene_name` (pydantic-first)."""

    input_name: str
    status: GeneNameStatus
    # The identifier the loader should store. For CURRENT/RENAMED this is a live systematic
    # gene id; for NON_GENE_FEATURE the (valid) feature id; for RETIRED the original name
    # kept as a legacy systematic identifier; None only for AMBIGUOUS.
    systematic_name: str | None
    feature_type: str | None = None  # GFF featuretype for NON_GENE_FEATURE resolutions
    candidates: list[str] = Field(default_factory=list)  # populated only for AMBIGUOUS
    note: str | None = None

    @property
    def is_current_gene(self) -> bool:
        """True when the name resolved to a live R64 gene (CURRENT or RENAMED)."""
        return self.status in (GeneNameStatus.CURRENT, GeneNameStatus.RENAMED)


#: The gffutils database every genome builds under its ``genome_root``.
GENOME_DB_FILENAME = "data.db"
#: Where a migrated database whose rows differ from a fresh build is kept (one file).
UNTRUSTED_DB_FILENAME = GENOME_DB_FILENAME + ".untrusted"
#: The table inside ``data.db`` that records what the database was built from.
SOURCE_TABLE = "torchcell_genome_db_source"
#: The :class:`GenomeDatabaseRecord` schema version this code writes and reads.
RECORD_VERSION = 1
#: The ``gffutils.create_db`` arguments every build uses (also recorded in the source).
CREATE_DB_KWARGS: dict[str, Any] = {
    "keep_order": True,
    "merge_strategy": "merge",
    "sort_attribute_values": True,
}
#: Temporary files this module writes in a genome directory: ``data.db.<host>.<pid>.
#: <random>.building`` (a build) and ``data.db.untrusted.<host>.<pid>.<random>.
#: building`` (the copy of an untrusted database on its way to ``data.db.untrusted``).
_BUILD_TEMP = re.compile(
    r"^data\.db\.(?:untrusted\.)?(?P<host>.+)\.(?P<pid>\d+)\.[a-z0-9_]+"
    r"\.building(?:-journal)?$"
)
#: Largest pid a sweep considers (a 32-bit pid_t); a name beyond it is not ours.
_PID_LIMIT = 2**31 - 1
#: Private database copies in the temp dir: ``torchcell-genome-<host>-<pid>-<random>.db``.
_PRIVATE_COPY = re.compile(
    r"^torchcell-genome-(?P<host>.+)-(?P<pid>\d+)-[a-z0-9_]+\.db$"
)


class GenomeDatabaseSource(BaseModel):
    """What a genome ``data.db`` is built from: the pinned GFF and the build arguments."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assembly_set: str
    gff_filename: str
    gff_sha256: str
    keep_order: bool
    merge_strategy: str
    sort_attribute_values: bool


class GenomeDatabaseRecord(BaseModel):
    """The one row of :data:`SOURCE_TABLE`: the source plus what the build left.

    The record lives inside the sqlite file, so the record and the rows it describes
    are replaced together by one atomic rename. At every open the row counts (one
    ``GROUP BY`` over ``features``, one ``COUNT`` over ``relations``) and sqlite's file
    change counter (header bytes 24-27, advanced by every committed write in the
    rollback-journal modes gffutils uses) are compared with the record, so any write
    made in place after the build, by a process running pre-2026.10.01 code, is
    detected: a deletion changes the counts, any write changes the counter.

    ``version`` is :data:`RECORD_VERSION` of the code that wrote it. A reader treats a
    lower (or absent) version as untrusted and migrates it once; a higher version was
    written by newer code and is refused by name
    (:class:`GenomeDatabaseVersionError`), so two code versions never rebuild the
    file back and forth.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int
    source: GenomeDatabaseSource
    featuretype_counts: dict[str, int]
    relations_count: int
    change_counter: int


class GenomeDatabaseSourceError(RuntimeError):
    """``data.db`` records a different source than this genome's pinned GFF."""


class GenomeDatabaseVersionError(RuntimeError):
    """``data.db`` carries a record written by newer code than this checkout."""


class GenomeRootNotFoundError(FileNotFoundError):
    """``genome_root`` does not exist and the constructor was not asked to build."""


class GenomeRootNotWritableError(PermissionError):
    """``data.db`` must be (re)built but this process cannot write ``genome_root``."""


def genome_database_source(
    assembly_set: str, gff_filename: str, gff_path: str
) -> GenomeDatabaseSource:
    """The source a database built from ``gff_path`` now would record."""
    return GenomeDatabaseSource(
        assembly_set=assembly_set,
        gff_filename=gff_filename,
        gff_sha256=sha256_file(Path(gff_path)),
        **CREATE_DB_KWARGS,
    )


def _change_counter(db_path: str) -> int:
    """Sqlite's file change counter: header bytes 24-27, big-endian."""
    with open(db_path, "rb") as fh:
        header = fh.read(28)
    return int.from_bytes(header[24:28], "big")


def _database_counts(conn: sqlite3.Connection) -> tuple[dict[str, int], int]:
    """Per-featuretype row counts of ``features`` and the row count of ``relations``."""
    featuretype_counts = {
        str(featuretype): int(n)
        for featuretype, n in conn.execute(
            "SELECT featuretype, COUNT(*) FROM features GROUP BY featuretype"
        )
    }
    relations_count = int(conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0])
    return featuretype_counts, relations_count


def database_content_digest(db_path: str) -> str:
    """sha256 over every ``features`` and ``relations`` row, in a fixed order."""
    h = hashlib.sha256()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    for row in conn.execute("SELECT * FROM features ORDER BY id"):
        h.update(repr(row).encode())
    h.update(b"relations")
    for row in conn.execute("SELECT * FROM relations ORDER BY parent, child, level"):
        h.update(repr(row).encode())
    conn.close()
    return h.hexdigest()


def _pid_alive(pid: int) -> bool:
    """Whether ``pid`` names a live process on this host."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # alive, owned by another user
        return True
    return True


def _sweep_dead(directory: str, pattern: re.Pattern[str]) -> list[str]:
    """Remove this host's and this user's regular files in ``directory`` matching
    ``pattern`` whose writer pid is dead; another host's, another user's, a live pid's,
    a directory and a name whose pid is out of range are left. Safe to run in many
    processes at once: a file another sweeper removed first is skipped. Returns the
    names this call removed.
    """
    host = socket.gethostname()
    removed = []
    for name in sorted(os.listdir(directory)):
        m = pattern.match(name)
        if m is None or m["host"] != host or not 0 < int(m["pid"]) <= _PID_LIMIT:
            continue
        path = osp.join(directory, name)
        try:
            st = os.lstat(path)
        except FileNotFoundError:  # another sweeper removed it first
            continue
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid():
            continue
        if _pid_alive(int(m["pid"])):
            continue
        try:
            os.remove(path)
        except FileNotFoundError:  # another sweeper removed it first
            continue
        removed.append(name)
    return removed


def _temp_in(directory: str, prefix: str) -> str:
    """A new empty file ``<prefix>.<host>.<pid>.<random>.building`` in ``directory``."""
    fd, path = tempfile.mkstemp(
        prefix=f"{prefix}.{socket.gethostname()}.{os.getpid()}.",
        suffix=".building",
        dir=directory,
    )
    os.close(fd)
    return path


def write_genome_database(
    gff_path: str, db_dir: str, source: GenomeDatabaseSource
) -> str:
    """Build a complete, recorded database in a new temporary file in ``db_dir``.

    The file is in ``db_dir`` itself, so the rename that installs it is atomic, and its
    name is unique per call. Returns its path; the caller renames it into place. On any
    failure the temporary file is removed before the exception propagates.
    """
    tmp_path = _temp_in(db_dir, GENOME_DB_FILENAME)
    try:
        gffutils.create_db(gff_path, dbfn=tmp_path, force=True, **CREATE_DB_KWARGS)
        conn = sqlite3.connect(tmp_path)
        featuretype_counts, relations_count = _database_counts(conn)
        expected_counter = _change_counter(tmp_path) + 1
        record = GenomeDatabaseRecord(
            version=RECORD_VERSION,
            source=source,
            featuretype_counts=featuretype_counts,
            relations_count=relations_count,
            change_counter=expected_counter,
        )
        # One explicit transaction, so the change counter advances exactly once.
        conn.isolation_level = None
        conn.execute("BEGIN")
        conn.execute(f"CREATE TABLE {SOURCE_TABLE} (record TEXT NOT NULL)")
        conn.execute(
            f"INSERT INTO {SOURCE_TABLE} (record) VALUES (?)",
            (record.model_dump_json(),),
        )
        conn.execute("COMMIT")
        conn.close()
        if _change_counter(tmp_path) != expected_counter:
            raise RuntimeError(
                f"{tmp_path}: the record transaction left the sqlite change counter at "
                f"{_change_counter(tmp_path)}, not {expected_counter}"
            )
        os.chmod(tmp_path, 0o644)
    except BaseException:
        os.remove(tmp_path)
        raise
    return tmp_path


def _read_record_json(db_path: str) -> str | None:
    """The raw record JSON stored inside ``db_path``, or None when it carries none."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    has_table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (SOURCE_TABLE,)
    ).fetchone()
    if has_table is None:
        conn.close()
        return None
    rows = conn.execute(f"SELECT record FROM {SOURCE_TABLE}").fetchall()
    conn.close()
    if len(rows) != 1:
        raise GenomeDatabaseSourceError(
            f"{db_path}: {SOURCE_TABLE} holds {len(rows)} rows, expected exactly 1"
        )
    return str(rows[0][0])


def read_genome_database_record(db_path: str) -> GenomeDatabaseRecord | None:
    """The record stored inside ``db_path``, or None when it carries none."""
    raw = _read_record_json(db_path)
    if raw is None:
        return None
    return GenomeDatabaseRecord.model_validate_json(raw)


def untrusted_reason(
    db_path: str, expected: GenomeDatabaseSource, rebuild_call: str
) -> str | None:
    """Why the existing ``db_path`` cannot be trusted, or None when it can.

    Untrusted (returned as a reason): no record, which is every database built before
    2026.10.01 and every one rebuilt in place by a process still running older code; a
    record of a lower (or absent) :data:`RECORD_VERSION`; row counts that differ from
    the record (rows deleted in place by such a process); or a change counter that
    differs from the record (any other write in place). A record of a higher version
    raises :class:`GenomeDatabaseVersionError`, and a record for a different source
    raises :class:`GenomeDatabaseSourceError`: the pinned GFF changed, which is a real
    source change, not a migration.
    """
    raw = _read_record_json(db_path)
    if raw is None:
        return f"it carries no {SOURCE_TABLE} record"
    version = json.loads(raw).get("version", 0)  # absent: written before versioning
    if version > RECORD_VERSION:
        raise GenomeDatabaseVersionError(
            f"{db_path} carries a record of version {version}, written by newer code "
            f"than this checkout (which reads version {RECORD_VERSION}); refusing to "
            "replace it. Update this checkout."
        )
    if version < RECORD_VERSION:
        return f"its record is version {version}, older than {RECORD_VERSION}"
    record = GenomeDatabaseRecord.model_validate_json(raw)
    if record.source != expected:
        raise GenomeDatabaseSourceError(
            f"{db_path} was built from {record.source.model_dump()} but this genome's "
            f"source is {expected.model_dump()}. Rebuild it deliberately, once, while "
            f"no job reads it: {rebuild_call}"
        )
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    featuretype_counts, relations_count = _database_counts(conn)
    conn.close()
    if (featuretype_counts, relations_count) != (
        record.featuretype_counts,
        record.relations_count,
    ):
        return (
            "its row counts differ from its record (features "
            f"{sum(featuretype_counts.values())} vs "
            f"{sum(record.featuretype_counts.values())} recorded, relations "
            f"{relations_count} vs {record.relations_count} recorded)"
        )
    counter = _change_counter(db_path)
    if counter != record.change_counter:
        return (
            "it was written in place after its build (sqlite change counter "
            f"{counter}, {record.change_counter} recorded)"
        )
    return None


def migrate_genome_database(
    gff_path: str,
    db_path: str,
    expected: GenomeDatabaseSource,
    rebuild_call: str,
    reason: str,
) -> None:
    """Replace the untrusted ``db_path`` with a fresh build, keeping at most one file.

    A fresh build is written beside ``db_path``. If ``db_path`` is trusted again by
    then (another process finished the same migration), the build is discarded.
    Otherwise the two are compared by :func:`database_content_digest`: identical rows
    (an old-code rebuild without a record) replace ``db_path`` and nothing is kept;
    different rows (deleted or rewritten in place) are copied to
    ``data.db.untrusted``, replacing any earlier kept file, before the fresh build is
    renamed onto ``db_path``. The path never goes missing and readers holding the old
    inode keep it. One WARNING names the case. Every temporary file is removed on
    every path.
    """
    db_dir = osp.dirname(db_path)
    tmp_path = write_genome_database(gff_path, db_dir, expected)
    copy_path: str | None = None
    try:
        if untrusted_reason(db_path, expected, rebuild_call) is None:
            return
        if database_content_digest(db_path) == database_content_digest(tmp_path):
            os.replace(tmp_path, db_path)
            log.warning(
                "genome database %s was not trusted (%s); its rows equal a fresh "
                "build, so it was replaced by the recorded build and nothing was kept",
                db_path,
                reason,
            )
            return
        kept = osp.join(db_dir, UNTRUSTED_DB_FILENAME)
        copy_path = _temp_in(db_dir, UNTRUSTED_DB_FILENAME)
        shutil.copyfile(db_path, copy_path)
        os.replace(copy_path, kept)
        os.replace(tmp_path, db_path)
        log.warning(
            "genome database %s was not trusted (%s); its rows differ from a fresh "
            "build, so it was kept as %s (replacing any earlier one) and replaced by "
            "the recorded build",
            db_path,
            reason,
            kept,
        )
    finally:
        for path in (tmp_path, copy_path):
            if path is not None and osp.exists(path):
                os.remove(path)


def _remove_private_copy(path: str, owner_pid: int) -> None:
    """Delete an instance's private database copy, only in the process that made it."""
    if os.getpid() == owner_pid and osp.exists(path):
        os.remove(path)


def _restore_genome(
    cls: type["SCerevisiaeGenome"],
    genome_root: str,
    go_root: str,
    private_db_path: str | None,
) -> "SCerevisiaeGenome":
    """Unpickling target: reopen with ``overwrite=False``, on the same database file."""
    genome = cls(genome_root, go_root, False)
    if private_db_path is not None:
        genome._db_connection_manager = GffutilsConnectionManager(private_db_path)
    return genome


def genome_database_untrusted_reason(genome_root: str) -> str | None:
    """Why ``<genome_root>/data.db`` would be built or migrated by a construction, or
    None when a construction would open it as is. Reads only; data-gated tests call
    it so that a test never performs the first migration of a real root.
    """
    gff_filename = f"saccharomyces_cerevisiae_{SCerevisiaeGenome.GENOME_VERSION}.gff"
    gff_path = resolve(SCerevisiaeGenome.ASSEMBLY_SET, gff_filename)
    db_path = osp.join(genome_root, GENOME_DB_FILENAME)
    if not osp.exists(db_path):
        return "it does not exist"
    source = genome_database_source(
        SCerevisiaeGenome.ASSEMBLY_SET, gff_filename, gff_path
    )
    return untrusted_reason(db_path, source, "SCerevisiaeGenome(..., overwrite=True)")


@define(eq=False)
class SCerevisiaeGenome(Genome):
    """S288C genome wrapper exposing genes, GO annotations, and sequence queries.

    ``<genome_root>/data.db`` is the gffutils database built from the pinned GFF. It
    is shared by every process on the same ``genome_root`` and is never written after
    it is built: every build goes to a unique temporary file in ``genome_root`` that is
    renamed into place (:func:`write_genome_database`). :meth:`drop_chrmt`,
    :meth:`drop_empty_go` and :meth:`remove_deprecated_go_terms` write to a private
    copy owned by this instance, ``torchcell-genome-<host>-<pid>-<random>.db`` in
    ``tempfile.gettempdir()`` (so ``TMPDIR`` controls where it goes), made on its
    first write and deleted when the instance is collected in the process that made
    it. A process killed before that leaves its copy behind; every creation of a
    private copy removes this host's and this user's copies whose pid is dead. The
    writes are logged on the instance, so a pickled copy whose parent's private file
    is gone rebuilds its own from the shared file at its first read.

    ``genome_root`` must exist unless ``overwrite=True``. ``overwrite`` decides how
    the constructor treats the shared file:

    * ``False`` (the default): build it when absent. When present, open it if its
      record (:class:`GenomeDatabaseRecord`) names this genome's source and its row
      counts and change counter still match. A record for a different source (the
      pinned GFF changed) raises :class:`GenomeDatabaseSourceError`. A database with
      no record, or written in place after its build, was built or modified by code
      from before 2026.10.01; it is migrated (:func:`migrate_genome_database`). This
      is a stated one-time migration of a database that cannot be trusted, not a
      fallback: the replacement is built from the same sha256-pinned GFF. When the
      root is not writable, a build or migration raises
      :class:`GenomeRootNotWritableError` and the untrusted file is never opened.
    * ``True``: rebuild it unconditionally (atomically). Pass it only deliberately.

    Construction also removes this host's ``data.db.*.building`` files whose writer
    pid is dead (a build killed mid-way), when the root is writable.
    """

    #: The assembly set in the genomes tier this class reads its release files from.
    ASSEMBLY_SET: ClassVar[str] = SGD_S288C_R64
    #: The release whose files the constructor resolves.
    GENOME_VERSION: ClassVar[str] = "R64-4-1_20230830"

    genome_root: str = field(init=True, repr=False, default="data/sgd/genome")
    go_root: str = field(init=True, repr=False, default="data/go")
    overwrite: bool = field(init=True, repr=True, default=False)
    fasta_dna: dict[str, Any] = field(init=False, default=None, repr=False)
    chr_to_nc: dict[int, str] = field(init=False, default=None, repr=False)
    nc_to_chr: dict[str, int] = field(init=False, default=None, repr=False)
    chr_to_len: dict[int, int] = field(init=False, default=None, repr=False)
    _gene_set: GeneSet = field(init=False, default=None, repr=False)
    _dna_fasta_path: str = field(init=False, default=None, repr=False)
    _protein_fasta_path: str = field(init=False, default=None, repr=False)
    _cds_fasta_path: str = field(init=False, default=None, repr=False)
    _gff_path: str = field(init=False, default=None, repr=False)
    _go: SortedSet[str] = field(init=False, default=None, repr=False)
    _go_genes: SortedDict[str, SortedSet[str]] = field(
        init=False, default=None, repr=False
    )
    _alias_to_systematic: dict[str, list[str]] = field(
        init=False, default=None, repr=False
    )
    # Cached all-feature index (upper-cased) backing resolve_gene_name.
    _feature_index: dict[str, Any] | None = field(init=False, default=None, repr=False)
    # Use factory to ensure GO DAG is not pickled
    _go_dag: GODag | None = field(init=False, factory=lambda: None, repr=False)
    _obo_path: str | None = field(init=False, default=None, repr=False)
    # This instance's private copy of data.db (made on its first write) and the
    # (pid, instance token) that made it; a pickled or forked copy writes to a copy of its own.
    _private_db_path: str | None = field(init=False, default=None, repr=False)
    _private_db_owner: tuple[int, str] | None = field(
        init=False, default=None, repr=False
    )
    # A random token per instance: ownership never depends on id(), which CPython
    # reuses after collection.
    _instance_token: str = field(
        init=False, factory=lambda: secrets.token_hex(8), repr=False
    )
    # Every write this instance made, in order, so a copy can be rebuilt from the
    # shared file: ("delete", ids) or ("remove_deprecated_go_terms", []).
    _db_writes: list[tuple[str, list[str]]] = field(
        init=False, factory=list, repr=False
    )

    def __attrs_post_init__(self) -> None:
        """Resolve the release files from the genomes tier and build the GFF database."""
        # Call parent class init to ensure all base attributes are set
        super().__init__(data_root=self.genome_root)
        self.genome_version = self.GENOME_VERSION

        # The release files come from the genomes tier, sha256-verified on every
        # resolve; genome_root stays the CACHE root (data.db, and through
        # SCerevisiaeGraph.sgd_root the genes/ and graph/ caches). There is no
        # download path: a machine without the tier fails here with the rsync that
        # seeds it, never with unpinned bytes.
        self._dna_fasta_path: str = resolve(
            self.ASSEMBLY_SET,
            "S288C_reference_sequence_" + self.genome_version + ".fsa",
        )
        gff_filename = "saccharomyces_cerevisiae_" + self.genome_version + ".gff"
        self._gff_path: str = resolve(self.ASSEMBLY_SET, gff_filename)
        self._protein_fasta_path = resolve(
            self.ASSEMBLY_SET, "orf_trans_all_" + self.genome_version + ".fasta"
        )
        self._cds_fasta_path = resolve(
            self.ASSEMBLY_SET, "orf_coding_all_" + self.genome_version + ".fasta"
        )

        db_path = osp.join(self.genome_root, GENOME_DB_FILENAME)
        source = genome_database_source(self.ASSEMBLY_SET, gff_filename, self._gff_path)
        rebuild_call = (
            f"SCerevisiaeGenome(genome_root={self.genome_root!r}, "
            f"go_root={self.go_root!r}, overwrite=True)"
        )
        if not osp.isdir(self.genome_root):
            if not self.overwrite:
                raise GenomeRootNotFoundError(
                    f"genome_root {self.genome_root!r} does not exist. Pass the "
                    "existing genome cache directory, or build a new one "
                    f"deliberately: {rebuild_call}"
                )
            os.makedirs(self.genome_root)
        writable = os.access(self.genome_root, os.W_OK)
        if writable:
            _sweep_dead(self.genome_root, _BUILD_TEMP)
        if self.overwrite or not osp.exists(db_path):
            why = "overwrite=True" if self.overwrite else "it does not exist"
            self._require_writable(writable, db_path, why)
            tmp_path = write_genome_database(self._gff_path, self.genome_root, source)
            os.replace(tmp_path, db_path)
        else:
            reason = untrusted_reason(db_path, source, rebuild_call)
            if reason is not None:
                self._require_writable(writable, db_path, reason)
                migrate_genome_database(
                    self._gff_path, db_path, source, rebuild_call, reason
                )

        # Set up connection manager for thread/process-safe database access
        self._db_connection_manager = GffutilsConnectionManager(db_path)

        self.fasta_dna = SeqIO.to_dict(SeqIO.parse(self._dna_fasta_path, "fasta"))  # type: ignore[no-untyped-call]  # Bio.SeqIO is untyped
        self.fasta_protein = SeqIO.to_dict(  # type: ignore[no-untyped-call]  # Bio.SeqIO is untyped
            SeqIO.parse(self._protein_fasta_path, "fasta")  # type: ignore[no-untyped-call]  # Bio.SeqIO is untyped
        )
        self.fasta_cds = SeqIO.to_dict(SeqIO.parse(self._cds_fasta_path, "fasta"))  # type: ignore[no-untyped-call]  # Bio.SeqIO is untyped
        # Create mapping from chromosome number to sequence identifier
        self.chr_to_nc = {
            get_chr_from_description(self.fasta_dna[key].description): key
            for key in self.fasta_dna.keys()
        }
        self.nc_to_chr = {v: k for k, v in self.chr_to_nc.items()}
        self.chr_to_len = {
            self.nc_to_chr[chr]: len(self.fasta_dna[chr].seq)
            for chr in self.fasta_dna.keys()
        }

        # TODO Not sure if this is now to tightly coupled to GO
        # We do want to remove inaccurate info as early as possible
        # Store GO path for lazy loading
        self._obo_path = osp.join(self.go_root, "go.obo")
        if not osp.exists(self._obo_path):
            os.makedirs(self.go_root, exist_ok=True)
            download_url(
                "http://current.geneontology.org/ontology/go.obo", self.go_root
            )
        # GO DAG will be loaded lazily via property
        # Call the method to remove deprecated GO terms
        # BUG this line doesn't work with ddp, I think the issue is merge=replace
        # self.remove_deprecated_go_terms()

    def _require_writable(self, writable: bool, db_path: str, why: str) -> None:
        """Refuse by name a build or migration this process cannot write."""
        if not writable:
            raise GenomeRootNotWritableError(
                f"{db_path} must be built ({why}), but {self.genome_root} is not "
                "writable by this process, so the database is not opened or built "
                "here. Construct SCerevisiaeGenome(genome_root="
                f"{self.genome_root!r}, go_root={self.go_root!r}) once from a process "
                "that can write that directory."
            )

    @property
    def db(self) -> Any:
        """Return the gffutils connection (non-None in this genome).

        An instance unpickled from a genome that had written reads the writer's
        private copy; when that file is gone (the writer was collected), the first
        read rebuilds a private copy of its own from the shared file and replays the
        logged writes.
        """
        if (
            self._private_db_path is not None
            and self._private_db_owner != (os.getpid(), self._instance_token)
            and not osp.exists(self._private_db_path)
        ):
            self._writable_db()
        return super().db

    @property
    def go_dag(self) -> GODag:
        """Lazy-load GO DAG - one per process"""
        if self._go_dag is None:
            if self._obo_path and osp.exists(self._obo_path):
                self._go_dag = GODag(self._obo_path)
            else:
                raise FileNotFoundError(f"GO OBO file not found at {self._obo_path}")
        return self._go_dag

    def __reduce_ex__(self, protocol: SupportsIndex) -> tuple[Any, ...]:
        """Custom pickling that handles non-pickleable objects."""
        # Get the attrs-generated __getstate__ if it exists
        state: dict[str, Any] = cast(
            "dict[str, Any]",
            (
                self.__getstate__()
                if hasattr(self, "__getstate__")
                else self.__dict__.copy()
            ),
        )

        # Clear non-pickleable objects
        if "_go_dag" in state:
            state["_go_dag"] = None
        # An unpickled instance never owns the pickled private path: it reads it,
        # and copies on its own first write (or first read, once the path is gone).
        state["_private_db_owner"] = None
        state["_instance_token"] = secrets.token_hex(8)

        # Reconstruct with overwrite=False on the same database file (the private
        # copy when this instance has written): unpickling in a worker must never
        # rebuild the shared file. The original ``overwrite`` comes back in ``state``.
        return (
            _restore_genome,
            (self.__class__, self.genome_root, self.go_root, self._private_db_path),
            state,
            None,
            iter([]),
        )

    def _writable_db(self) -> Any:
        """This instance's private copy of the database, made on its first write.

        The copy is a sqlite backup of the file this instance currently reads, so
        every later read and write of this instance sees exactly the database it had,
        and the shared ``data.db`` is never written. A pickled or forked copy of the
        instance (another pid, or an unpickled instance) makes a copy of its own; when the file
        it reads is gone, it copies the shared file and replays :attr:`_db_writes`.
        """
        owner = (os.getpid(), self._instance_token)
        if self._private_db_owner != owner:
            assert self._db_connection_manager is not None
            source_path = self._db_connection_manager.db_path
            replay = not osp.exists(source_path)
            if replay:
                source_path = osp.join(self.genome_root, GENOME_DB_FILENAME)
            temp_dir = tempfile.gettempdir()
            _sweep_dead(temp_dir, _PRIVATE_COPY)
            fd, path = tempfile.mkstemp(
                prefix=f"torchcell-genome-{socket.gethostname()}-{os.getpid()}-",
                suffix=".db",
                dir=temp_dir,
            )
            os.close(fd)
            src = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
            dst = sqlite3.connect(path)
            src.backup(dst)
            dst.close()
            src.close()
            self._db_connection_manager = GffutilsConnectionManager(path)
            self._private_db_path = path
            self._private_db_owner = owner
            weakref.finalize(self, _remove_private_copy, path, os.getpid())
            if replay:
                for op, ids in self._db_writes:
                    self._apply_write(op, ids)
        return super().db

    def _apply_write(self, op: str, ids: list[str]) -> None:
        """Apply one logged write to this instance's private copy (no backup file)."""
        db = super().db
        assert db is not None
        if op == "delete":
            for feature_id in ids:
                db.delete(feature_id, make_backup=False)
        else:
            db.update(
                self._deprecated_go_updates(db),
                merge_strategy="replace",
                make_backup=False,
            )
        db.conn.commit()

    def _write(self, op: str, ids: list[str]) -> None:
        """Make or reuse the private copy, apply ``op`` to it, and log it."""
        self._writable_db()
        self._apply_write(op, ids)
        self._db_writes.append((op, ids))

    def remove_deprecated_go_terms(self) -> None:
        """Drop GO terms absent from or obsolete in the GO DAG in this instance's copy."""
        self._write("remove_deprecated_go_terms", [])

    def _deprecated_go_updates(self, db: Any) -> list[Feature]:
        """The gene features of ``db`` with absent or obsolete GO terms removed."""
        # Create a list to hold updated features
        updated_features = []

        # Iterate over each feature in the database
        invalid_go_terms: dict[str, list[str]] = {"not_in_go_dag": [], "obsolete": []}
        for feature in db.features_of_type("gene"):
            # Check if the feature has the "Ontology_term" attribute
            if "Ontology_term" in feature.attributes:
                # Filter out deprecated GO terms
                valid_onto_terms = []
                valid_go_terms = []
                for term in feature.attributes["Ontology_term"]:
                    if term.startswith("GO:"):
                        if term not in self.go_dag:
                            invalid_go_terms["not_in_go_dag"].append(term)
                        elif self.go_dag[term].is_obsolete:
                            invalid_go_terms["obsolete"].append(term)
                        else:
                            valid_go_terms.append(term)
                    else:
                        valid_onto_terms.append(term)
                # Update the "Ontology_term" attribute for the feature
                if valid_go_terms:
                    feature.attributes["Ontology_term"] = (
                        valid_go_terms + valid_onto_terms
                    )
                else:
                    del feature.attributes["Ontology_term"]

                # Add the updated feature to the list
                updated_features.append(feature)

        return updated_features

    @property
    def alias_to_systematic(self) -> dict[str, list[str]]:
        """Return a cached mapping from each alias to its list of systematic IDs."""
        if self._alias_to_systematic is None:
            alias_map: dict[str, list[str]] = {}
            for gene_id in self.gene_set:
                gene = self[gene_id]
                if gene and gene.alias:
                    for alias in gene.alias:
                        if alias not in alias_map:
                            alias_map[alias] = []
                        alias_map[alias].append(gene_id)
            self._alias_to_systematic = alias_map
        return self._alias_to_systematic

    @property
    def feature_index(self) -> dict[str, Any]:
        """Cached locus-feature index (upper-cased) backing :meth:`resolve_gene_name`.

        Indexes only GENE-LIKE LOCUS features (``_LOCUS_FEATURE_TYPES``: ``gene`` plus the
        RNA-gene / pseudogene / transposon-gene / ``blocked_reading_frame`` types) so that
        non-locus features (``region``, ``CDS``, ``mRNA``, ``ARS`` ...) never shadow a real
        gene name -- e.g. a ``region`` feature literally id'd ``ADE1`` must NOT intercept the
        gene ADE1. Unlike :attr:`alias_to_systematic` (``"gene"`` features only), this also
        covers non-``"gene"`` loci (e.g. ``blocked_reading_frame`` pseudogenes like
        ``YER109C``/FLO8). Keys: ``genes`` (upper ids of live ``"gene"`` features),
        ``locus_type`` (upper id -> featuretype for non-``"gene"`` loci), ``standard_to_ids``
        (upper standard/common name from the GFF ``gene`` attribute -> locus ids) and
        ``alias_to_ids`` (upper ``Alias`` value -> locus ids).
        """
        if self._feature_index is None:
            genes = {g.upper() for g in self.gene_set}
            locus_type: dict[str, str] = {}
            standard_to_ids: dict[str, list[str]] = {}
            alias_to_ids: dict[str, list[str]] = {}
            for feat in self.db.all_features():
                if feat.featuretype not in _LOCUS_FEATURE_TYPES:
                    continue
                fid = feat.id
                if feat.featuretype != "gene":
                    locus_type[fid.upper()] = feat.featuretype
                for std in feat.attributes.get("gene", []) or []:
                    standard_to_ids.setdefault(std.strip().upper(), []).append(fid)
                for alias in feat.attributes.get("Alias", []) or []:
                    alias_to_ids.setdefault(alias.strip().upper(), []).append(fid)
            self._feature_index = {
                "genes": genes,
                "locus_type": locus_type,
                "standard_to_ids": standard_to_ids,
                "alias_to_ids": alias_to_ids,
            }
        return self._feature_index

    def resolve_gene_name(self, name: str) -> GeneNameResolution:
        """Reconcile a source gene name to the current R64-4-1 annotation (layered).

        Layers, in order: (1) exact live gene id; (2) a valid non-``"gene"`` LOCUS id
        (e.g. a ``blocked_reading_frame`` pseudogene); (3) the standard/common name of a
        locus (from the GFF ``gene`` attribute); (4) a secondary ``Alias`` of a locus; (5)
        not found -> retired, retained with the original name. The standard-name layer runs
        BEFORE the alias layer so a common name resolves to the gene that OWNS it (e.g.
        ``AAP1`` -> ``YHR047C``, whose standard name is AAP1) rather than to a gene that
        merely lists it as a secondary alias (``Q0080``). Within layers 3-4 a unique gene
        wins; multiple genes are ``AMBIGUOUS``; a unique non-gene locus is
        ``NON_GENE_FEATURE``. The resolver is pure/per-name and applies NO drop policy --
        callers decide retention and any batch-level collision handling.
        """
        raw = name
        n = name.strip().upper()
        idx = self.feature_index
        genes: set[str] = idx["genes"]
        locus_type: dict[str, str] = idx["locus_type"]

        # 1. Exact live gene.
        if n in genes:
            return GeneNameResolution(
                input_name=raw, status=GeneNameStatus.CURRENT, systematic_name=n
            )
        # 2. A valid non-"gene" locus by its own id (e.g. blocked_reading_frame pseudogene).
        if n in locus_type:
            return GeneNameResolution(
                input_name=raw,
                status=GeneNameStatus.NON_GENE_FEATURE,
                systematic_name=n,
                feature_type=locus_type[n],
                note=f"valid R64 {locus_type[n]}, not a gene feature",
            )
        # 3. Standard/common name; then 4. secondary alias -- same resolution semantics.
        for mapping_key, layer in (
            ("standard_to_ids", "standard name"),
            ("alias_to_ids", "alias"),
        ):
            ids = sorted(set(idx[mapping_key].get(n, [])))
            if not ids:
                continue
            gene_ids = [i for i in ids if i.upper() in genes]
            if len(gene_ids) == 1:
                return GeneNameResolution(
                    input_name=raw,
                    status=GeneNameStatus.RENAMED,
                    systematic_name=gene_ids[0],
                    note=f"{layer} of current gene {gene_ids[0]}",
                )
            if len(gene_ids) > 1:
                return GeneNameResolution(
                    input_name=raw,
                    status=GeneNameStatus.AMBIGUOUS,
                    systematic_name=None,
                    candidates=gene_ids,
                    note=f"{layer} of multiple current genes",
                )
            if len(ids) == 1:
                t = ids[0]
                return GeneNameResolution(
                    input_name=raw,
                    status=GeneNameStatus.NON_GENE_FEATURE,
                    systematic_name=t,
                    feature_type=locus_type.get(t.upper()),
                    note=f"{layer} of {locus_type.get(t.upper())} {t} (not a gene feature)",
                )
            return GeneNameResolution(
                input_name=raw,
                status=GeneNameStatus.AMBIGUOUS,
                systematic_name=None,
                candidates=ids,
                note=f"{layer} of multiple non-gene loci",
            )
        # 5. Not found anywhere in R64-4-1: a retired/dubious 2005-era ORF.
        return GeneNameResolution(
            input_name=raw,
            status=GeneNameStatus.RETIRED,
            systematic_name=n,
            note="not found in R64-4-1; retained as a legacy systematic name",
        )

    @property
    def go(self) -> SortedSet[str]:
        """Return the cached set of all GO terms across genes in the gene set."""
        if self._go is None:
            all_go = SortedSet()

            # Iterate through all genes in self.gene_set
            for gene_id in self.gene_set:
                gene = self[gene_id]  # Retrieve the gene object

                # Use the go attribute of the gene object if it exists and is not None
                if gene and hasattr(gene, "go") and gene.go is not None:
                    all_go.update(gene.go)
            self._go = all_go
            return self._go
        else:
            return self._go

    def go_subset(self, gene_set: SortedSet[str]) -> SortedSet[str]:
        """Return the set of GO terms covered by the given subset of genes."""
        go_subset = SortedSet()

        # Iterate through the provided subset of genes
        for gene_id in gene_set:
            gene = self[gene_id]  # Retrieve the gene object

            # Use the go attribute of the gene object if it exists and is not None
            if gene and hasattr(gene, "go") and gene.go is not None:
                go_subset.update(gene.go)

        return go_subset

    @property
    def go_genes(self) -> SortedDict[str, SortedSet[str]]:
        """Return a cached mapping from each GO term to the genes annotated with it."""
        # CHECK could this contain obselete terms? We don't check if the terms are in self.go...
        if self._go_genes is None:
            go_genes_dict = SortedDict()

            # Iterate through all genes in self.gene_set
            for gene_id in self.gene_set:
                gene = self[gene_id]  # Retrieve the gene object

                # Use the go attribute of the gene object if it exists and is not None
                if gene and hasattr(gene, "go") and gene.go is not None:
                    for go_term in gene.go:
                        if go_term not in go_genes_dict:
                            go_genes_dict[go_term] = SortedSet()
                        go_genes_dict[go_term].add(gene_id)
            self._go_genes = go_genes_dict
            return go_genes_dict
        else:
            return self._go_genes

    def go_subset_genes(
        self, gene_set: SortedSet[str]
    ) -> SortedDict[str, SortedSet[str]]:
        """Return a GO-term-to-genes mapping restricted to the given gene subset."""
        go_subset_genes_dict = SortedDict()

        # Iterate through the provided subset of genes
        for gene_id in gene_set:
            gene = self[gene_id]  # Retrieve the gene object

            # Use the go attribute of the gene object if it exists and is not None
            if gene and hasattr(gene, "go") and gene.go is not None:
                for go_term in gene.go:
                    if go_term not in go_subset_genes_dict:
                        go_subset_genes_dict[go_term] = SortedSet()
                    go_subset_genes_dict[go_term].add(gene_id)

        return go_subset_genes_dict

    def get_seq(
        self, chr: int | str, start: int, end: int, strand: str
    ) -> DnaSelectionResult:
        """Return the DNA sequence for the given chromosome region and strand.

        ``chr`` is the chromosome number (0 is the mitochondrion), the number that
        ``DnaSelectionResult.chromosome`` stores; a FASTA key or any other string is
        refused by name, as is a strand other than ``+``/``-``.
        """
        if not isinstance(chr, int) or chr not in self.chr_to_nc:
            raise ValueError(
                f"Chromosome must be one of the chromosome numbers "
                f"{sorted(self.chr_to_nc)}, got {chr!r}"
            )
        if strand not in ("+", "-"):
            raise ValueError(f"Strand must be '+' or '-', got {strand!r}")
        fasta_key = self.chr_to_nc[chr]
        if strand == "+":
            seq = self.fasta_dna[fasta_key].seq[start:end]
        else:
            seq = self.fasta_dna[fasta_key].seq[start:end].reverse_complement()
        return DnaSelectionResult(
            id=self.id,  # type: ignore[attr-defined]  # no 'id' on genome (pre-existing)
            chromosome=chr,
            strand=strand,
            start=start,
            end=end,
            seq=str(seq),
        )

    @property
    def gene_attribute_table(self) -> pd.DataFrame:
        """Return a DataFrame of single-valued GFF attributes for every gene."""
        data = []
        for gene_feature in self.db.features_of_type("gene"):
            gene_data = {}
            for attr_name in gene_feature.attributes.keys():
                # We only add attributes with length 1 or less
                if len(gene_feature.attributes[attr_name]) <= 1:
                    # If the attribute is a list with one value, we unpack it
                    gene_data[attr_name] = (
                        gene_feature.attributes[attr_name][0]
                        if len(gene_feature.attributes[attr_name]) == 1
                        else None
                    )
            data.append(gene_data)
        return pd.DataFrame(data)

    @property
    def feature_types(self) -> list[str]:
        """Return the list of feature types present in the GFF database."""
        return list(self.db.featuretypes())

    def compute_gene_set(self) -> GeneSet:
        """Return the GeneSet of all gene IDs in the database."""
        genes = [feat.id for feat in list(self.db.features_of_type("gene"))]
        assert len(genes) == len(set(genes)), (
            "Duplicate genes found... chekc handled by gff."
        )
        return GeneSet(genes)

    def drop_chrmt(self) -> None:
        """Remove all chrmt features from this instance's database copy and cache."""
        mitochondrial_features = [
            f for f in self.db.all_features() if f.seqid == "chrmt"
        ]

        # Remove these features from the gene set cache if it existsc
        if self._gene_set is not None:
            for feature in mitochondrial_features:
                self._gene_set.discard(feature.id)

        # Remove these features from this instance's private copy of the database
        self._write("delete", [feature.id for feature in mitochondrial_features])

        # The locus index and the GO-to-genes map were built from the pre-drop
        # database; reset them so the next access rebuilds without chrmt.
        self._feature_index = None
        self._go_genes = None

    def drop_empty_go(self) -> None:
        """Remove genes without GO terms from this instance's database copy and gene set."""
        # Initialize a list to hold genes to be removed
        genes_to_remove = []

        # Iterate through all genes in the current gene_set
        for gene_id in self.gene_set:
            gene = self[gene_id]
            if gene is not None:
                # Check if the GO terms are empty
                # None case for never annotated, 0 for no GO terms
                if gene.go is None or len(gene.go) == 0:
                    genes_to_remove.append(gene_id)

        # Remove these genes from the gene set cache
        for gene_id in genes_to_remove:
            self._gene_set.discard(gene_id)

        # Remove these genes from this instance's private copy of the database
        self._write("delete", list(genes_to_remove))

        # Same as drop_chrmt: the locus index and the GO-to-genes map were built from
        # the pre-drop gene set; reset them so the next access rebuilds without the
        # dropped genes.
        self._feature_index = None
        self._go_genes = None

    def __getitem__(self, item: str) -> SCerevisiaeGene | None:
        """Return the SCerevisiaeGene for a systematic ID, or None if absent."""
        # For now we only support the systematic names
        # ising region instead, since it give more options on dealing with gene processing in gene class
        try:
            gene = SCerevisiaeGene(
                id=item,
                db=self.db,
                fasta_dna=self.fasta_dna,
                fasta_protein=self.fasta_protein,
                fasta_cds=self.fasta_cds,
                chr_to_nc=self.chr_to_nc,
                chromosome_lengths=self.chr_to_len,
            )
            return gene
        except KeyError:
            print(
                f"Gene {item} not found in genome, only systematic names (ID) are supported."
            )
            return None


def main() -> None:
    """Build the genome from DATA_ROOT and print its gene set."""
    import os

    from dotenv import load_dotenv

    load_dotenv()
    DATA_ROOT = cast(str, os.getenv("DATA_ROOT"))

    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    print(f"genome.gene_set: {genome.gene_set}")
    # orf_classes = []
    # lengths = []
    # for gene in genome.gene_set:
    #     orf_classes.append(genome[gene].orf_classification[0])
    #     lengths.append(len(genome[gene].protein.seq))
    # print(pd.Series(orf_classes).value_counts())
    # genome.go
    # genome.drop_chrmt()
    # print(len(genome.gene_set))
    # genome.drop_empty_go()

    # # genes_not_divisible_by_3 = [
    # #     gene for gene in genome.gene_set if len(genome[gene]) % 3 != 0
    # # ]
    # # print(len(genes_not_divisible_by_3))
    # # print(genes_not_divisible_by_3)
    # # genes_no_start = [
    # #     gene for gene in genes_not_divisible_by_3 if genome[gene].seq[:3] != "ATG"
    # # ]
    # # print(len(genes_no_start))
    # # print(genes_no_start)
    # # print()

    # not_divisible_by_3 = []
    # for gene in genome.gene_set:
    #     if len(str(genome["YIL111W"].cds.seq)) % 3 != 0:
    #         not_divisible_by_3.append(gene)
    # print(len(not_divisible_by_3))
    # print(compute_codon_frequency(str(genome["YIL111W"].cds.seq)))
    print()


if __name__ == "__main__":
    main()
