# torchcell/sequence/genome/scerevisiae/ncbi.py
# [[torchcell.sequence.genome.scerevisiae.ncbi]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/sequence/genome/scerevisiae/ncbi.py
# Test file: tests/torchcell/sequence/genome/scerevisiae/test_ncbi.py

"""A yeast genome read GenBank-first from an NCBI assembly set (#990).

``SCerevisiaeGenome`` reads the SGD release: SGD's own GFF, its chromosome naming, its
systematic names, and GO from the GFF's ``Ontology_term``. A strain SGD does not cover
has none of those. What it does have is an NCBI assembly, and this module reads one:
the GenBank flat file first (the loci and their CDS), with the GFF for coordinates, the
replicon FASTA for sequence, and the assembly report as the replicon table.

The identifiers are the assembly's OWN. CEN.PK113-7D's locus tags are the ``SCEN_*``
namespace, which is a different identifier space from S288C's systematic names, and
this class never maps between them: a genotype written against this assembly is keyed by
its tags, and a crosswalk to S288C, if one is ever wanted, is a derived artifact
computed where it is used, never a silent substitution at read time.

``go_annotations`` is EMPTY for such a set, and that is a stated absence rather than a
gap this class fills: no GO release is keyed to ``SCEN_*`` tags, and borrowing S288C's
GO through a name match would attribute S288C's evidence to another strain's loci.
``go_source`` records which set and member the absence was decided from.

Why this is not a ``BacterialGenome``: that class is correct for one replicon
(``fasta_chromosome`` and ``_read_sequences`` refuse a flat file with any other) and its
GO route requires a GAF or GOA file. A yeast assembly has 17 replicons and no GAF, so it
is read here, beside it, reusing its file readers rather than widening its invariants.
"""

from __future__ import annotations

import os.path as osp
import re
from types import MappingProxyType
from typing import Any, ClassVar

from attrs import define, field
from pydantic import BaseModel, ConfigDict, Field
from sortedcontainers import SortedSet

from torchcell.sequence.genome.bacterial import (
    GO_BASIC_OBO,
    GenBankAnnotation,
    GenBankLocus,
    GenomeAnnotationMismatchError,
    read_fasta,
    read_genbank,
    read_protein_fasta,
)
from torchcell.sequence.genome.base import (
    AnnotatedGene,
    AnnotatedGenome,
    GenomeReleaseFiles,
)
from torchcell.sequence.genome.registry import (
    GO_RELEASE_20260805,
    SCEREVISIAE_CENPK113_7D,
    resolve,
)

#: The mitochondrion's chromosome key, the SGD convention this repo already uses.
MITOCHONDRION = 0
#: The 2-micron plasmid's key: not a chromosome, and outside the 1-16 + 0 range, so a
#: query over the nuclear chromosomes never picks it up by accident.
PLASMID_2MICRON = 17

_ROMAN = MappingProxyType(
    {
        "I": 1,
        "II": 2,
        "III": 3,
        "IV": 4,
        "V": 5,
        "VI": 6,
        "VII": 7,
        "VIII": 8,
        "IX": 9,
        "X": 10,
        "XI": 11,
        "XII": 12,
        "XIII": 13,
        "XIV": 14,
        "XV": 15,
        "XVI": 16,
    }
)


class NcbiYeastAssembly(BaseModel):
    """One NCBI yeast assembly as a genome class reads it from its assembly set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    organism: str
    strain: str
    assembly_set: str
    genbank_assembly: str = Field(description='e.g. "GCA_002571405.2_ASM257140v2".')
    locus_tag_pattern: str = Field(
        description="Every GenBank locus tag of the assembly fully matches it."
    )
    replicons: dict[str, int] = Field(
        description="GenBank accession to chromosome key, the assembly report's table."
    )
    default_genome_root: str = Field(
        description="The default data.db cache root, relative like data/sgd/genome."
    )
    notes: str

    @property
    def genbank_member(self) -> str:
        """The GenBank flat file, the primary ingest."""
        return f"{self.genbank_assembly}_genomic.gbff.gz"

    @property
    def gff_member(self) -> str:
        """The GenBank GFF3 that ``data.db`` is built from."""
        return f"{self.genbank_assembly}_genomic.gff.gz"

    @property
    def dna_fasta_member(self) -> str:
        """The replicon sequences."""
        return f"{self.genbank_assembly}_genomic.fna.gz"

    @property
    def protein_fasta_member(self) -> str:
        """The proteins, keyed by protein accession."""
        return f"{self.genbank_assembly}_protein.faa.gz"

    @property
    def assembly_report_member(self) -> str:
        """The assembly report, whose table is the source of :attr:`replicons`."""
        return f"{self.genbank_assembly}_assembly_report.txt"


def read_assembly_report(path: str) -> dict[str, int]:
    """The replicon table of an NCBI assembly report: GenBank accession to chromosome key.

    Roman-numbered chromosomes take their own number, ``MT`` the mitochondrion's 0 and
    ``p2uM`` the 2-micron plasmid's 17. A molecule name outside those is refused rather
    than given a key, because a wrong key silently moves a gene to another chromosome.
    """
    out: dict[str, int] = {}
    with open(path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            molecule, accession = fields[2], fields[4]
            if molecule in _ROMAN:
                key = _ROMAN[molecule]
            elif molecule == "MT":
                key = MITOCHONDRION
            elif molecule == "p2uM":
                key = PLASMID_2MICRON
            else:
                raise ValueError(
                    f"{osp.basename(path)}: molecule {molecule!r} has no chromosome key"
                )
            out[accession] = key
    if not out:
        raise ValueError(f"{osp.basename(path)} lists no replicon")
    return out


#: CEN.PK113-7D (GCA_002571405.2, ASM257140v2), deposited by
#: ``scripts/provision_yeast_genomes.py``. The replicon table is the literal of the
#: deposited assembly report, which ``test_ncbi.py`` re-reads and compares.
CENPK113_7D_ASSEMBLY = NcbiYeastAssembly(
    organism="Saccharomyces cerevisiae",
    strain="CEN.PK113-7D",
    assembly_set=SCEREVISIAE_CENPK113_7D,
    genbank_assembly="GCA_002571405.2_ASM257140v2",
    # MEASURED on the deposited flat file: 6,171 distinct tags, 5,886 of the form
    # SCEN_<replicon letter><5 digits> and 285 tRNA tags of the form
    # SCEN_<letter>trna<n>[<amino-acid letter>] (4 of them with no trailing letter).
    locus_tag_pattern=r"SCEN_[A-Z](?:\d{5}|trna\d+[A-Z]?)",
    replicons={
        "CP046081.1": 1,
        "CP046082.1": 2,
        "CP046083.1": 3,
        "CP046084.1": 4,
        "CP046085.1": 5,
        "CP046086.1": 6,
        "CP046087.1": 7,
        "CP046088.1": 8,
        "CP046089.1": 9,
        "CP046090.1": 10,
        "CP046091.1": 11,
        "CP046092.1": 12,
        "CP046093.1": 13,
        "CP046094.1": 14,
        "CP046095.1": 15,
        "CP046096.1": 16,
        "CP046097.1": PLASMID_2MICRON,
        "CP046098.1": MITOCHONDRION,
    },
    default_genome_root="data/ncbi/cenpk113_7d",
    notes=(
        "The CEN.PK reference the CEN.PK-derived engineering strains are written "
        "against (#990). Locus tags are SCEN_*, never S288C systematic names, and the "
        "set carries no GO file."
    ),
)


@define(repr=False)
class NcbiYeastGene(AnnotatedGene):
    """A gene of an :class:`NcbiYeastGenome`: coordinates from ``data.db``, names and
    product from its :class:`GenBankLocus`, and no GO.

    The same shape as the bacterial gene, minus GO: the assembly's loci are reached by
    no GO release, so ``go`` is ``None`` on every gene and the absence is recorded on
    the genome rather than implied per gene.
    """

    #: NCBI writes a gene's GFF ``ID`` as ``gene-<locus_tag>``.
    FEATURE_ID_PREFIX: ClassVar[str] = "gene-"
    #: GFF seqid (the GenBank replicon accession) to the chromosome key.
    REPLICONS: ClassVar[Any]

    locus: GenBankLocus = field(kw_only=True, repr=False)

    @classmethod
    def seqid_to_chromosome(cls, seqid: str) -> int:
        """The chromosome key of a replicon in :attr:`REPLICONS`; any other is refused."""
        if seqid not in cls.REPLICONS:
            raise ValueError(
                f"{cls.__name__}: seqid {seqid!r} is not one of the replicons "
                f"{sorted(cls.REPLICONS)}"
            )
        return int(cls.REPLICONS[seqid])

    def annotate(self, gene_feature: Any) -> None:
        """GFF ``Name`` and ``Note``, then names, product and xrefs from the flat file."""
        self.name = gene_feature.attributes.get("Name", None)
        self.note = gene_feature.attributes.get("Note", None)
        self.alias = list(self.locus.synonyms) or None
        self.go = None
        self.symbol = self.locus.symbol
        self.synonyms = list(self.locus.synonyms)
        self.old_locus_tags = list(self.locus.old_locus_tags)
        self.product = self.locus.product
        self.protein_id = self.locus.protein_id
        self.db_xrefs = list(self.locus.db_xrefs)
        self.pseudo = self.locus.pseudo


@define(eq=False)
class NcbiYeastGenome[GeneT: NcbiYeastGene](AnnotatedGenome[GeneT]):
    """A yeast genome read GenBank-first from one NCBI assembly set.

    A subclass binds :attr:`ASSEMBLY` and the base class variables and declares its
    gene class, whose ``REPLICONS`` is the assembly's replicon table. The ``data.db``
    cache contract is the base class's.

    ``genbank`` is the parsed flat file and ``go_annotations`` is empty by construction
    (:attr:`GO_ABSENCE` says why). ``fasta_cds`` holds each coding locus's CDS from the
    flat file, so the set needs no CDS FASTA member.
    """

    #: The assembly this class reads.
    ASSEMBLY: ClassVar[NcbiYeastAssembly]
    #: The gene-like locus types of an NCBI GenBank GFF3.
    LOCUS_FEATURE_TYPES: ClassVar[frozenset[str]] = frozenset({"gene", "pseudogene"})
    #: Why a set read by this class carries no GO.
    GO_ABSENCE: ClassVar[str] = (
        "no GO release is keyed to this assembly's locus tags; S288C's GO is not "
        "borrowed through a name match, because that would attribute one strain's "
        "evidence to another strain's loci"
    )

    genbank: GenBankAnnotation = field(init=False, default=None, repr=False)
    _genbank_path: str = field(init=False, default=None, repr=False)

    @classmethod
    def release_files(cls) -> GenomeReleaseFiles:
        """The GenBank GFF, the replicon FASTA and the protein FASTA; no CDS FASTA."""
        return GenomeReleaseFiles(
            dna_fasta=cls.ASSEMBLY.dna_fasta_member,
            gff=cls.ASSEMBLY.gff_member,
            protein_fasta=cls.ASSEMBLY.protein_fasta_member,
            cds_fasta=None,
        )

    @classmethod
    def fasta_chromosome(cls, record: Any) -> int:
        """The record's replicon accession, through the assembly's replicon table."""
        return cls.gene_class().seqid_to_chromosome(record.id)

    def _prepare_go_obo(self) -> str:
        """``go-basic.obo`` of the pinned GO release; never a download."""
        return resolve(GO_RELEASE_20260805, GO_BASIC_OBO)

    def _read_sequences(self) -> None:
        """The flat file (loci and CDS), the replicon FASTA and the protein FASTA.

        Refuses a flat file whose replicons are not the assembly's table or whose locus
        tags leave the assembly's pattern, which is what keeps a wrong deposit from
        being read as this strain.
        """
        member = self.ASSEMBLY.genbank_member
        self._genbank_path = resolve(self.ASSEMBLY_SET, member)
        self.genbank, self.fasta_cds = read_genbank(self._genbank_path, member)
        accessions = sorted(r.accession for r in self.genbank.replicons)
        expected = sorted(self.ASSEMBLY.replicons)
        if accessions != expected:
            raise GenomeAnnotationMismatchError(
                f"{member} holds replicons {accessions}, not the assembly's {expected}"
            )
        pattern = re.compile(self.ASSEMBLY.locus_tag_pattern)
        outside = [t for t in self.genbank.loci if not pattern.fullmatch(t)]
        if outside:
            raise GenomeAnnotationMismatchError(
                f"{member}: {len(outside)} locus tags do not match "
                f"{self.ASSEMBLY.locus_tag_pattern}: {outside[:10]}"
            )
        self.fasta_dna = read_fasta(self._dna_fasta_path)
        self.fasta_protein = read_protein_fasta(self._protein_fasta_path, self.genbank)

    @property
    def go_annotations(self) -> dict[str, SortedSet[str]]:
        """Empty: this assembly's loci are reached by no GO release (:attr:`GO_ABSENCE`)."""
        return {}

    def __getitem__(self, item: str) -> GeneT | None:
        """The gene (or pseudogene) with this locus tag, or None when it is absent.

        The gene carries its :class:`GenBankLocus`, which is where its names, product
        and xrefs come from; a joined pseudogene is refused by name, since one
        start/end interval cannot represent it.
        """
        locus = self.genbank.loci.get(item)
        if locus is None or not (locus.pseudo or item in self.gene_set):
            return None
        if locus.segments > 1:
            raise ValueError(
                f"{item} is a joined {'pseudogene' if locus.pseudo else 'locus'} "
                f"({locus.segments} segments in {self.ASSEMBLY.genbank_member}); a "
                "gene with one start/end interval cannot represent it"
            )
        return self.gene_class()(
            id=item,
            db=self.db,
            fasta_dna=self.fasta_dna,
            fasta_protein=self.fasta_protein,
            fasta_cds=self.fasta_cds,
            chr_to_nc=self.chr_to_nc,
            chromosome_lengths=self.chr_to_len,
            locus=locus,
        )


@define(repr=False)
class CenPk1137DGene(NcbiYeastGene):
    """A CEN.PK113-7D gene, keyed by its own ``SCEN_*`` locus tag."""

    REPLICONS: ClassVar[Any] = MappingProxyType(dict(CENPK113_7D_ASSEMBLY.replicons))


@define(eq=False)
class CenPk1137DGenome(NcbiYeastGenome[CenPk1137DGene]):
    """The CEN.PK113-7D genome (GCA_002571405.2), read from its deposited NCBI set."""

    ASSEMBLY: ClassVar[NcbiYeastAssembly] = CENPK113_7D_ASSEMBLY
    ASSEMBLY_SET: ClassVar[str] = SCEREVISIAE_CENPK113_7D
    GENOME_VERSION: ClassVar[str] = "ASM257140v2"
    ANNOTATION_NAME: ClassVar[str] = "GenBank"
    ANNOTATION_RELEASE: ClassVar[str] = "GCA_002571405.2"

    genome_root: str = field(
        init=True, repr=False, default=CENPK113_7D_ASSEMBLY.default_genome_root
    )
    overwrite: bool = field(init=True, repr=True, default=False)

    @classmethod
    def gene_class(cls) -> type[CenPk1137DGene]:
        """CEN.PK113-7D genes are :class:`CenPk1137DGene`."""
        return CenPk1137DGene
