# torchcell/sequence/genome/pkudriavzevii/sd108.py
# [[torchcell.sequence.genome.pkudriavzevii.sd108]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/sequence/genome/pkudriavzevii/sd108.py
# Test file: tests/torchcell/sequence/genome/pkudriavzevii/test_sd108.py

r"""Pichia kudriavzevii (Issatchenkia orientalis) SD108, genome v1, read GFF3 + FASTA.

The set ``pkudriavzevii_SD108_v1`` (:data:`PKUDRIAVZEVII_SD108`) is the group's curated
SD108 assembly: one haploid copy of the diploid strain, five chromosomes
(``SD108_chr1`` .. ``SD108_chr5``), the circular mitochondrion (``SD108_mito``) and an
unplaced rDNA sequence (``SD108_rDNA_unit``). The repository stance is GenBank first,
then GFF3 + FASTA; this release ships no GenBank flat file, so it is read the way the
SGD release is (:class:`~torchcell.sequence.genome.scerevisiae.s288c.SCerevisiaeGenome`):
the GFF3 into ``data.db`` through gffutils, the genome FASTA for sequence, and the
release's own ``proteins.faa`` and ``cds.fna``.

The identifiers are the assembly's OWN (``NOMENCLATURE.md`` of the set):
``SD108_X<seq><type><number>``, where ``<seq>`` is ``1``-``5``, ``M`` (mitochondrion)
or ``U`` (rDNA unit), ``<type>`` is ``G`` (protein-coding), ``T`` (tRNA) or ``R``
(rRNA), and the four-digit number is unique genome-wide within its type. A gene added
in a later version takes its left neighbor's number plus a capital letter
(``SD108_X1G0123A``), so :data:`SD108_GENE_ID_PATTERN` admits that suffix; v1 has no
such id. Haplotype-2 alleles (``_H2``) belong to the second copy, which this class does
not read. Transcripts and proteins add ``.t1``.

*S. cerevisiae* names are a crosswalk, never an identity. ``sc_gene`` and
``sc_systematic`` (``gene_xref.tsv``) are exposed read-only on each gene as an
:class:`ScerevisiaeNameCrosswalk` together with the protein evidence behind them
(``sc_name_check``); a lookup by ``URA3`` is not a lookup of ``SD108_X2G1933``, and
nothing here maps one to the other at read time. The rules are those of
``CenPk1137DGenome`` (``torchcell/sequence/genome/scerevisiae/ncbi.py``): a genotype
written against this assembly is keyed by its ids, and a crosswalk to S288C, if one is
wanted, is a derived artifact computed where it is used.

GO is EMPTY, and that is a stated absence (:attr:`PKudriavzeviiSD108Genome.go_source`):
no GO release is keyed to SD108 ids, and borrowing S288C's GO through ``sc_gene`` would
attribute one species' evidence to another species' loci.

Coordinates. A protein-coding gene row of this GFF3 spans the transcript model, UTRs
included (3,069 of the 5,159 differ from their CDS span). As in the SGD reader, a
gene's ``start``/``end``/``seq`` are its CODING span (the first to the last CDS base,
introns included), so sequence windows sit on the start and stop codons; the gene
row's span is kept as ``locus_start``/``locus_end``. tRNA and rRNA genes use their
gene row.

Proteins vs genes. ``proteins.faa`` and ``cds.fna`` hold 5,159 records each, one per
protein-coding gene (5,144 nuclear + 15 mitochondrial, the mitochondrial ones
translated with code 3), keyed ``<gene>.t1``. The other 217 of the 5,376 gene rows are
199 tRNA and 18 rRNA genes, which have no protein. Construction refuses a protein or CDS
record that maps to no protein-coding gene, and a protein-coding gene with none.
"""

from __future__ import annotations

import csv
import os.path as osp
import re
from collections import Counter
from types import MappingProxyType
from typing import Any, ClassVar, Final, Literal

from attrs import define, field
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from gffutils.feature import Feature
from pydantic import BaseModel, ConfigDict, Field
from sortedcontainers import SortedSet

from torchcell.sequence.genome.bacterial import (
    GO_BASIC_OBO,
    GenomeAnnotationMismatchError,
)
from torchcell.sequence.genome.base import (
    AnnotatedGene,
    AnnotatedGenome,
    GenomeReleaseFiles,
)
from torchcell.sequence.genome.registry import (
    GO_RELEASE_20260805,
    PKUDRIAVZEVII_SD108,
    resolve,
)

#: Every gene id of the primary assembly (v1 measured: 5,376 of 5,376 gene rows, and
#: no mRNA, exon or CDS id). The optional capital letter is the nomenclature's rule for
#: a gene inserted in a later version.
SD108_GENE_ID_PATTERN: Final = r"SD108_X[1-5MU][GTR]\d{4}[A-Z]?"
#: The suffix the release adds to a gene id for its transcript, CDS and protein.
TRANSCRIPT_SUFFIX: Final = ".t1"

#: The mitochondrion's chromosome key, the SGD convention this repo already uses.
MITOCHONDRION: Final = 0
#: The unplaced rDNA unit's key: not a chromosome, one past the five, so a query over
#: chromosomes 1-5 never picks it up.
RDNA_UNIT: Final = 6

#: FASTA / GFF3 seqid to chromosome key; ``test_sd108.py`` re-reads the deposited
#: ``.fai`` and compares.
SD108_REPLICONS: Final = MappingProxyType(
    {
        "SD108_chr1": 1,
        "SD108_chr2": 2,
        "SD108_chr3": 3,
        "SD108_chr4": 4,
        "SD108_chr5": 5,
        "SD108_mito": MITOCHONDRION,
        "SD108_rDNA_unit": RDNA_UNIT,
    }
)

GENOME_FASTA: Final = "SD108_genome_v1.fasta"
GENOME_FAI: Final = "SD108_genome_v1.fasta.fai"
GFF3: Final = "SD108_genome_v1.gff3"
PROTEIN_FASTA: Final = "proteins.faa"
CDS_FASTA: Final = "cds.fna"
GENE_XREF: Final = "gene_xref.tsv"

ScNameCheck = Literal["confirmed", "best_hit_weak", "paralog"]


class ScerevisiaeNameCrosswalk(BaseModel):
    """One gene's *S. cerevisiae* name and the protein evidence for it (read-only).

    A row of ``gene_xref.tsv`` that carries an ``sc_gene``. ``sc_name_check`` is the
    release's verdict on the name (``NOMENCLATURE.md`` section 3): ``confirmed`` is a
    reciprocal best hit (E <= 1e-10, >= 50% of both proteins aligned),
    ``best_hit_weak`` a best hit below those thresholds, ``paralog`` one of two
    near-equal *S. cerevisiae* copies that is not the best hit (``sc_best_hit`` names
    the other). A label, never an identifier.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sc_gene: str
    sc_systematic: str | None
    sc_name_source: str
    sc_name_check: ScNameCheck
    sc_identity: float | None = Field(description="Percent identity.")
    sc_coverage_sd108: float | None
    sc_coverage_scer: float | None
    sc_evalue: float | None
    sc_best_hit: str | None


class GoAbsence(BaseModel):
    """GO stated absent for an assembly set: which set, what was inspected, and why."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assembly_set: str
    members_inspected: tuple[str, ...]
    reason: str


def read_fai(path: str) -> dict[str, int]:
    """A samtools ``.fai`` index: sequence name to length."""
    out: dict[str, int] = {}
    with open(path) as handle:
        for line in handle:
            name, length = line.rstrip("\n").split("\t")[:2]
            out[name] = int(length)
    if not out:
        raise ValueError(f"{osp.basename(path)} lists no sequence")
    return out


def _optional(value: str) -> str | None:
    return value if value else None


def _optional_float(value: str) -> float | None:
    return float(value) if value else None


def read_gene_xref(path: str) -> tuple[list[str], dict[str, ScerevisiaeNameCrosswalk]]:
    """``gene_xref.tsv``: every gene id in file order, and the named genes' crosswalks.

    A row with an ``sc_gene`` and no ``sc_name_check`` is refused: the name would then
    be carried without the evidence that qualifies it.
    """
    ids: list[str] = []
    crosswalk: dict[str, ScerevisiaeNameCrosswalk] = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            gene_id = row["gene_id"]
            ids.append(gene_id)
            if not row["sc_gene"]:
                continue
            if not row["sc_name_check"]:
                raise GenomeAnnotationMismatchError(
                    f"{GENE_XREF}: {gene_id} names {row['sc_gene']} with no sc_name_check"
                )
            crosswalk[gene_id] = ScerevisiaeNameCrosswalk(
                sc_gene=row["sc_gene"],
                sc_systematic=_optional(row["sc_systematic"]),
                sc_name_source=row["sc_name_source"],
                sc_name_check=row["sc_name_check"],  # type: ignore[arg-type]  # pydantic validates the Literal
                sc_identity=_optional_float(row["sc_identity"]),
                sc_coverage_sd108=_optional_float(row["sc_coverage_sd108"]),
                sc_coverage_scer=_optional_float(row["sc_coverage_scer"]),
                sc_evalue=_optional_float(row["sc_evalue"]),
                sc_best_hit=_optional(row["sc_best_hit"]),
            )
    return ids, crosswalk


def rekey_by_gene(
    records: dict[str, SeqRecord], coding_genes: set[str], member: str
) -> dict[str, SeqRecord]:
    """``<gene>.t1`` records re-keyed by gene id; any record or gene left over is refused.

    The id of each returned record is the gene id; the original record id stays at the
    head of its description.
    """
    out: dict[str, SeqRecord] = {}
    unmapped = []
    for record_id, record in records.items():
        gene_id = record_id.removesuffix(TRANSCRIPT_SUFFIX)
        if gene_id == record_id or gene_id not in coding_genes:
            unmapped.append(record_id)
            continue
        record.id = gene_id
        out[gene_id] = record
    if unmapped:
        raise GenomeAnnotationMismatchError(
            f"{member}: {len(unmapped)} record(s) map to no protein-coding gene: "
            f"{sorted(unmapped)[:10]}"
        )
    missing = sorted(coding_genes - set(out))
    if missing:
        raise GenomeAnnotationMismatchError(
            f"{member}: {len(missing)} protein-coding gene(s) have no record: "
            f"{missing[:10]}"
        )
    return out


@define(repr=False)
class PKudriavzeviiSD108Gene(AnnotatedGene):
    """An SD108 gene, keyed by its own ``SD108_X...`` id.

    ``start``/``end``/``seq`` are the coding span for a protein-coding gene and the gene
    row for a tRNA or rRNA gene; ``locus_start``/``locus_end`` are the gene row's span
    (UTRs included). ``go`` is None on every gene (the genome's ``go_source`` says
    why). ``sc_crosswalk`` is the *S. cerevisiae* name with its evidence, or None.
    """

    REPLICONS: ClassVar[Any] = SD108_REPLICONS

    crosswalk: ScerevisiaeNameCrosswalk | None = field(kw_only=True, repr=False)

    @classmethod
    def seqid_to_chromosome(cls, seqid: str) -> int:
        """The chromosome key of a replicon in :data:`SD108_REPLICONS`; any other is
        refused.
        """
        if seqid not in cls.REPLICONS:
            raise ValueError(
                f"{cls.__name__}: seqid {seqid!r} is not one of the replicons "
                f"{sorted(cls.REPLICONS)}"
            )
        return int(cls.REPLICONS[seqid])

    def coding_feature(self) -> Feature:
        """The CDS span of a protein-coding gene; the gene row of an RNA gene."""
        gene = self.db[self.id]
        if gene.attributes["gene_biotype"][0] != "protein_coding":
            return gene
        cds = list(self.db.children(gene, featuretype="CDS"))
        if not cds:
            raise ValueError(f"protein-coding gene {self.id} has no CDS row")
        span = Feature(
            seqid=gene.seqid,
            featuretype="CDS",
            start=min(c.start for c in cds),
            end=max(c.end for c in cds),
            strand=gene.strand,
        )
        return span

    def annotate(self, gene_feature: Any) -> None:
        """The GFF3 attributes, the crosswalk, and no GO.

        The GFF3's ``sc_gene`` must equal ``gene_xref.tsv``'s, so the two members of
        the release cannot disagree about a name unnoticed.
        """
        attributes = gene_feature.attributes
        self.name = attributes["Name"][0]
        self.alias = None
        self.note = None
        self.go = None
        self.biotype = attributes["gene_biotype"][0]
        self.gene_name = attributes.get("gene_name", [None])[0]
        self.confidence = attributes.get("confidence", [None])[0]
        self.cbs573_locus_tag = attributes.get("cbs573_locus_tag", [None])[0]
        self.ncbi_jl09 = attributes.get("ncbi_jl09", [None])[0]
        self.locus_start = gene_feature.start
        self.locus_end = gene_feature.end
        # GFF3 reserves the comma as a value separator and the release writes three
        # names with a literal one (ADE5,7, ARG5,6, DUR1,2), which gffutils splits;
        # rejoined, they are the names gene_xref.tsv carries whole.
        gff_name = ",".join(attributes.get("sc_gene", [])) or None
        xref_name = None if self.crosswalk is None else self.crosswalk.sc_gene
        if gff_name != xref_name:
            raise GenomeAnnotationMismatchError(
                f"{self.id}: sc_gene {gff_name!r} in {GFF3}, {xref_name!r} in "
                f"{GENE_XREF}"
            )

    @property
    def sc_crosswalk(self) -> ScerevisiaeNameCrosswalk | None:
        """The *S. cerevisiae* name and its evidence; a label, never an identifier."""
        return self.crosswalk

    @property
    def sc_gene(self) -> str | None:
        """The *S. cerevisiae* gene name of :attr:`sc_crosswalk`, or None."""
        return None if self.crosswalk is None else self.crosswalk.sc_gene

    @property
    def sc_systematic(self) -> str | None:
        """The *S. cerevisiae* systematic name of :attr:`sc_crosswalk`, or None."""
        return None if self.crosswalk is None else self.crosswalk.sc_systematic


@define(eq=False)
class PKudriavzeviiSD108Genome(AnnotatedGenome[PKudriavzeviiSD108Gene]):
    """The P. kudriavzevii SD108 v1 genome, read from its deposited set.

    ``sc_crosswalk`` maps a gene id to its :class:`ScerevisiaeNameCrosswalk` (3,471
    genes in v1); it is keyed by SD108 id only. ``go_annotations`` is empty
    (:attr:`go_source`). The ``data.db`` cache contract is the base class's.
    """

    ASSEMBLY_SET: ClassVar[str] = PKUDRIAVZEVII_SD108
    GENOME_VERSION: ClassVar[str] = "v1"
    #: Every locus of the GFF3 (protein-coding, tRNA and rRNA) is a ``gene`` row.
    LOCUS_FEATURE_TYPES: ClassVar[frozenset[str]] = frozenset({"gene"})
    ANNOTATION_NAME: ClassVar[str] = "SD108_genome_v1"
    ANNOTATION_RELEASE: ClassVar[str] = "SD108_genome_v1"
    #: Why the set carries no GO.
    GO_ABSENCE: ClassVar[str] = (
        "no GO release is keyed to SD108 gene ids; S. cerevisiae's GO is not borrowed "
        "through sc_gene, because that would attribute one species' evidence to "
        "another species' loci"
    )

    genome_root: str = field(
        init=True, repr=False, default="data/pkudriavzevii/sd108/genome"
    )
    overwrite: bool = field(init=True, repr=True, default=False)
    sc_crosswalk: dict[str, ScerevisiaeNameCrosswalk] = field(
        init=False, default=None, repr=False
    )

    @classmethod
    def gene_class(cls) -> type[PKudriavzeviiSD108Gene]:
        """SD108 genes are :class:`PKudriavzeviiSD108Gene`."""
        return PKudriavzeviiSD108Gene

    @classmethod
    def release_files(cls) -> GenomeReleaseFiles:
        """The genome FASTA, the GFF3, and the release's protein and CDS FASTAs."""
        return GenomeReleaseFiles(
            dna_fasta=GENOME_FASTA,
            gff=GFF3,
            protein_fasta=PROTEIN_FASTA,
            cds_fasta=CDS_FASTA,
        )

    @classmethod
    def fasta_chromosome(cls, record: Any) -> int:
        """The record's seqid, through :data:`SD108_REPLICONS`."""
        return cls.gene_class().seqid_to_chromosome(record.id)

    def _prepare_go_obo(self) -> str:
        """``go-basic.obo`` of the pinned GO release; never a download."""
        return resolve(GO_RELEASE_20260805, GO_BASIC_OBO)

    def _read_sequences(self) -> None:
        """The genome FASTA, the proteins and CDS re-keyed by gene id, the crosswalk.

        Refuses a FASTA whose sequences or lengths are not the ``.fai``'s and the
        replicon table's, a gene id outside :data:`SD108_GENE_ID_PATTERN`, a
        ``gene_xref.tsv`` whose genes are not the GFF3's, and any protein or CDS record
        that maps to no protein-coding gene (or a coding gene with none).
        """
        self.fasta_dna = SeqIO.to_dict(SeqIO.parse(self._dna_fasta_path, "fasta"))  # type: ignore[no-untyped-call]  # Bio.SeqIO is untyped
        fai = read_fai(resolve(self.ASSEMBLY_SET, GENOME_FAI))
        lengths = {name: len(rec.seq) for name, rec in self.fasta_dna.items()}
        if lengths != fai or set(fai) != set(SD108_REPLICONS):
            raise GenomeAnnotationMismatchError(
                f"{GENOME_FASTA} holds {lengths}; {GENOME_FAI} lists {fai}; the "
                f"replicon table is {sorted(SD108_REPLICONS)}"
            )

        pattern = re.compile(SD108_GENE_ID_PATTERN)
        biotypes: dict[str, str] = {}
        for feature in self.db.features_of_type("gene"):
            biotypes[feature.id] = feature.attributes["gene_biotype"][0]
        outside = sorted(g for g in biotypes if not pattern.fullmatch(g))
        if outside:
            raise GenomeAnnotationMismatchError(
                f"{GFF3}: {len(outside)} gene ids do not match "
                f"{SD108_GENE_ID_PATTERN}: {outside[:10]}"
            )

        xref_ids, self.sc_crosswalk = read_gene_xref(
            resolve(self.ASSEMBLY_SET, GENE_XREF)
        )
        duplicated = sorted(g for g, n in Counter(xref_ids).items() if n > 1)
        if duplicated or set(xref_ids) != set(biotypes):
            raise GenomeAnnotationMismatchError(
                f"{GENE_XREF} and {GFF3} disagree on the gene set: duplicated "
                f"{duplicated[:10]}, only in {GENE_XREF} "
                f"{sorted(set(xref_ids) - set(biotypes))[:10]}, only in {GFF3} "
                f"{sorted(set(biotypes) - set(xref_ids))[:10]}"
            )

        coding = {g for g, biotype in biotypes.items() if biotype == "protein_coding"}
        proteins = SeqIO.to_dict(SeqIO.parse(self._protein_fasta_path, "fasta"))  # type: ignore[no-untyped-call]  # Bio.SeqIO is untyped
        self.fasta_protein = rekey_by_gene(proteins, coding, PROTEIN_FASTA)
        assert self._cds_fasta_path is not None
        cds = SeqIO.to_dict(SeqIO.parse(self._cds_fasta_path, "fasta"))  # type: ignore[no-untyped-call]  # Bio.SeqIO is untyped
        self.fasta_cds = rekey_by_gene(cds, coding, CDS_FASTA)

    @property
    def go_annotations(self) -> dict[str, SortedSet[str]]:
        """Empty: SD108's genes are reached by no GO release (:attr:`go_source`)."""
        return {}

    @property
    def go_source(self) -> GoAbsence:
        """The stated absence of GO: the set, the members inspected, the reason."""
        return GoAbsence(
            assembly_set=self.ASSEMBLY_SET,
            members_inspected=(GFF3, GENE_XREF),
            reason=self.GO_ABSENCE,
        )

    def __getitem__(self, item: str) -> PKudriavzeviiSD108Gene | None:
        """The gene with this SD108 id, or None when it is absent.

        Only SD108 ids are keys: an *S. cerevisiae* name such as ``URA3`` is not a gene
        id of this assembly and returns None.
        """
        if item not in self.gene_set:
            return None
        return self.gene_class()(
            id=item,
            db=self.db,
            fasta_dna=self.fasta_dna,
            fasta_protein=self.fasta_protein,
            fasta_cds=self.fasta_cds,
            chr_to_nc=self.chr_to_nc,
            chromosome_lengths=self.chr_to_len,
            crosswalk=self.sc_crosswalk.get(item),
        )
