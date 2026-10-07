# torchcell/sequence/genome/bacterial
# [[torchcell.sequence.genome.bacterial]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/sequence/genome/bacterial
# Test file: tests/torchcell/sequence/genome/test_bacterial.py

"""NCBI-annotated bacterial genomes, GenBank first, over one assembly set of the tier.

:class:`BacterialGenome` is the layer the bacterial hosts share
(``torchcell.sequence.genome.ecoli.k12``, ``torchcell.sequence.genome.pputida.kt2440``).
Every file comes from the genomes tier through
:func:`~torchcell.sequence.genome.registry.resolve` (sha256-verified); nothing is
downloaded at construction, including GO.

* **GenBank first.** The GCA ``_genomic.gbff.gz`` member is the primary source of the
  locus universe: locus tags, ``/gene`` symbols, ``/gene_synonym`` (split on ``;``),
  ``/old_locus_tag``, ``/db_xref``, ``/pseudo``, products and protein ids, parsed into
  :class:`GenBankLocus` records (:func:`read_genbank`). The gene set is the locus tags
  of the non-pseudo ``gene`` features; pseudogenes are valid loci that resolve as
  ``NON_GENE_FEATURE``. CDS sequences are extracted from the flat file's CDS features,
  and the GCA ``_protein.faa.gz`` is re-keyed from protein accession to locus tag.
* **GFF3 second.** The GCA ``_genomic.gff.gz`` feeds the shared gffutils ``data.db``
  cache of :class:`~torchcell.sequence.genome.base.AnnotatedGenome` (coordinates,
  windows, the recorded source that detects a cache built from another set). NCBI writes
  a gene's ``ID`` as ``gene-<locus_tag>`` (``BacterialGene.FEATURE_ID_PREFIX``).
  Construction checks that the database's gene and pseudogene features carry exactly
  the GenBank locus tags.
* **RefSeq crosswalk.** The GCF ``_genomic.gff.gz`` gives each RefSeq locus tag
  (``PP_RS00005``) its ``old_locus_tag`` (the GenBank tag) where RefSeq retagged the
  assembly, and the inline ``Ontology_term`` GO of the RefSeq annotation
  (:func:`read_refseq_gff`).
* **GO.** One route per host, named by :class:`GoSourceSpec`: a GAF whose column 11
  (DB Object Synonym) carries the host's locus tags (:func:`read_gaf_synonym_go`), or the
  RefSeq GFF's ``Ontology_term`` crosswalked through ``old_locus_tag``
  (:func:`refseq_inline_go`). The DAG is ``go-basic.obo`` of the
  ``go_release_2026-08-05`` set. What the route reached is recorded in a
  :class:`GoAnnotationSource`.
* **Name resolution.** :meth:`BacterialGenome.resolve_gene_name` reconciles a source
  name in this order: exact locus tag, ``old_locus_tag``, RefSeq locus tag, gene
  symbol, ``gene_synonym``, retired. Within a layer an exact-case match wins and a
  case-insensitive match is used only when there is none (E. coli carries ``Pro2`` and
  ``pro2`` as synonyms of two different genes).

A host subclass names its :class:`BacterialAssembly` (assembly set, GenBank and RefSeq
assembly names, replicon, locus-tag pattern, GO source, default cache root) and its gene
class (the replicons a GFF seqid may name).
"""

import gzip
import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import IO, Any, ClassVar, Literal
from urllib.parse import unquote

import pandas as pd
from attrs import define, field
from Bio import SeqIO
from Bio.SeqFeature import CompoundLocation, SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord
from gffutils.feature import Feature
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sortedcontainers import SortedSet

from torchcell.literature.manifest import sha256_file
from torchcell.sequence import GeneSet
from torchcell.sequence.genome.base import (
    AnnotatedGene,
    AnnotatedGenome,
    GeneNameResolution,
    GeneNameStatus,
    GenomeReleaseFiles,
)
from torchcell.sequence.genome.registry import GO_RELEASE_20260805, resolve

log = logging.getLogger(__name__)

#: The GO ontology member of the ``go_release_2026-08-05`` set every bacterial genome
#: loads its DAG from.
GO_BASIC_OBO = "go-basic.obo"
#: GenBank feature types that are a locus's product (the feature that names it).
PRODUCT_FEATURE_TYPES = frozenset(
    {"CDS", "tRNA", "rRNA", "ncRNA", "tmRNA", "misc_RNA", "precursor_RNA"}
)
#: A GO identifier as GAF column 5 and ``Ontology_term`` carry it.
GO_ID_PATTERN = re.compile(r"GO:\d{7}")
#: GAF 2.x rows have 17 tab-separated columns.
GAF_COLUMNS = 17


class GenomeAnnotationMismatchError(ValueError):
    """Two members of one assembly set disagree about the loci they annotate."""


def _open_text(path: str) -> IO[str]:
    """Open a tier member as text: gzip when its name ends in ``.gz``."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


class GenBankLocus(BaseModel):
    """One ``gene`` feature of a GenBank flat file, with its product feature.

    The product feature is the one feature of :data:`PRODUCT_FEATURE_TYPES` on the
    locus whose span (start, end, strand) equals the gene's; the others (alternative
    starts such as MG1655 ``mrcB``'s PBP-1Bgamma) contribute only their protein ids, in
    ``isoform_protein_ids``. Coordinates are 1-based inclusive (GFF convention) and
    span every segment of a joined location.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    locus_tag: str
    symbol: str | None = Field(description="The /gene qualifier.")
    synonyms: tuple[str, ...] = Field(
        description="/gene_synonym values, each split on ';' (INSDC separator)."
    )
    old_locus_tags: tuple[str, ...]
    db_xrefs: tuple[str, ...] = Field(
        description="/db_xref of the gene feature, then of its product feature."
    )
    pseudo: bool = Field(description="/pseudo or /pseudogene on the gene feature.")
    replicon: str
    start: int
    end: int
    strand: Literal["+", "-"]
    segments: int = Field(description="Parts of the gene feature's location.")
    product_feature_type: str | None
    product: str | None
    protein_id: str | None = Field(
        description="protein_id of the product CDS when it is not /pseudo."
    )
    isoform_protein_ids: tuple[str, ...]


class GenBankReplicon(BaseModel):
    """One record (replicon) of a GenBank flat file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    accession: str
    length: int
    topology: str


class GenBankAnnotation(BaseModel):
    """The loci of one GenBank flat file, keyed by locus tag in file order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    member: str
    replicons: tuple[GenBankReplicon, ...]
    loci: dict[str, GenBankLocus]

    @property
    def gene_tags(self) -> list[str]:
        """Locus tags of the non-pseudo loci, in file order."""
        return [tag for tag, locus in self.loci.items() if not locus.pseudo]

    @property
    def pseudogene_tags(self) -> list[str]:
        """Locus tags of the pseudo loci, in file order."""
        return [tag for tag, locus in self.loci.items() if locus.pseudo]


def _location(feature: SeqFeature) -> SimpleLocation | CompoundLocation:
    """The feature's location; a GenBank feature without one is refused."""
    if feature.location is None:
        raise ValueError(f"a {feature.type} feature has no location")
    return feature.location


def _strand(feature: SeqFeature, what: str) -> Literal["+", "-"]:
    strand = _location(feature).strand
    if strand == 1:
        return "+"
    if strand == -1:
        return "-"
    raise ValueError(f"{what} has strand {strand!r}; a locus needs +1 or -1")


def _span(feature: SeqFeature) -> tuple[int, int, int | None]:
    location = _location(feature)
    return int(location.start), int(location.end), location.strand


def _qualifier(feature: SeqFeature, key: str) -> list[str]:
    return list(feature.qualifiers.get(key, []))


def _single(feature: SeqFeature, key: str, what: str) -> str | None:
    values = _qualifier(feature, key)
    if len(values) > 1:
        raise ValueError(f"{what} carries {len(values)} /{key} qualifiers: {values}")
    return values[0] if values else None


def _locus(
    gene: SeqFeature, products: list[SeqFeature], replicon: str, member: str
) -> tuple[GenBankLocus, SeqFeature | None]:
    """The :class:`GenBankLocus` of one gene feature, and its coding CDS (or None)."""
    tag = _single(gene, "locus_tag", f"{member} gene feature")
    if tag is None:
        raise ValueError(
            f"{member}: a gene feature at {gene.location} has no locus_tag"
        )
    what = f"{member} locus {tag}"
    location = _location(gene)
    spanning = [p for p in products if _span(p) == _span(gene)]
    if len(spanning) > 1:
        raise ValueError(
            f"{what}: {len(spanning)} product features span the gene "
            f"({[p.type for p in spanning]}); the product is not unique"
        )
    product = spanning[0] if spanning else None
    coding = (
        product
        if product is not None
        and product.type == "CDS"
        and "pseudo" not in product.qualifiers
        else None
    )
    isoforms = tuple(
        pid
        for p in products
        if p is not product and p.type == "CDS" and "pseudo" not in p.qualifiers
        for pid in _qualifier(p, "protein_id")
    )
    synonyms = tuple(
        name.strip()
        for value in _qualifier(gene, "gene_synonym")
        for name in value.split(";")
        if name.strip()
    )
    xrefs = _qualifier(gene, "db_xref") + (
        _qualifier(product, "db_xref") if product is not None else []
    )
    locus = GenBankLocus(
        locus_tag=tag,
        symbol=_single(gene, "gene", what),
        synonyms=synonyms,
        old_locus_tags=tuple(_qualifier(gene, "old_locus_tag")),
        db_xrefs=tuple(dict.fromkeys(xrefs)),
        pseudo="pseudo" in gene.qualifiers or "pseudogene" in gene.qualifiers,
        replicon=replicon,
        start=int(location.start) + 1,
        end=int(location.end),
        strand=_strand(gene, what),
        segments=len(location.parts),
        product_feature_type=product.type if product is not None else None,
        product=_single(product, "product", what) if product is not None else None,
        protein_id=_single(coding, "protein_id", what) if coding is not None else None,
        isoform_protein_ids=isoforms,
    )
    return locus, coding


def read_genbank(
    path: str, member: str
) -> tuple[GenBankAnnotation, dict[str, SeqRecord]]:
    """Parse a GenBank flat file into its loci and the CDS sequence of each coding locus.

    ``member`` is the file's name in its assembly set (recorded on the annotation). The
    CDS of a locus is its product CDS (not ``/pseudo``) extracted from the record, joins
    included, as a ``SeqRecord`` whose id is the locus tag and whose description is the
    protein id. A product feature without a ``gene`` feature, a repeated locus tag, a
    gene without a locus tag and a non-unique product are refused by name.
    """
    replicons: list[GenBankReplicon] = []
    loci: dict[str, GenBankLocus] = {}
    cds: dict[str, SeqRecord] = {}
    with _open_text(path) as handle:
        for record in SeqIO.parse(handle, "genbank"):  # type: ignore[no-untyped-call]  # Bio.SeqIO is untyped
            replicons.append(
                GenBankReplicon(
                    accession=record.id,
                    length=len(record.seq),
                    topology=record.annotations["topology"],
                )
            )
            genes: dict[str, SeqFeature] = {}
            products: dict[str, list[SeqFeature]] = {}
            for feature in record.features:
                if feature.type == "gene":
                    tag = _single(feature, "locus_tag", f"{member} gene feature")
                    if tag is None:
                        raise ValueError(
                            f"{member}: a gene feature at {feature.location} has no "
                            "locus_tag"
                        )
                    if tag in genes or tag in loci:
                        raise ValueError(f"{member}: locus tag {tag} is repeated")
                    genes[tag] = feature
                elif feature.type in PRODUCT_FEATURE_TYPES:
                    tag = _single(feature, "locus_tag", f"{member} {feature.type}")
                    if tag is None:
                        raise ValueError(
                            f"{member}: a {feature.type} at {feature.location} has no "
                            "locus_tag"
                        )
                    products.setdefault(tag, []).append(feature)
            orphans = sorted(set(products) - set(genes))
            if orphans:
                raise ValueError(
                    f"{member}: product features name locus tags with no gene "
                    f"feature: {orphans[:10]}"
                )
            for tag, gene in genes.items():
                locus, coding = _locus(gene, products.get(tag, []), record.id, member)
                loci[tag] = locus
                if coding is not None:
                    cds[tag] = SeqRecord(
                        coding.extract(record.seq),  # type: ignore[no-untyped-call]  # Bio.SeqFeature.extract is untyped
                        id=tag,
                        name=tag,
                        description=locus.protein_id or "",
                    )
    annotation = GenBankAnnotation(member=member, replicons=tuple(replicons), loci=loci)
    return annotation, cds


def read_fasta(path: str) -> dict[str, SeqRecord]:
    """A FASTA member (plain or ``.gz``) keyed by record id."""
    with _open_text(path) as handle:
        records: dict[str, SeqRecord] = SeqIO.to_dict(SeqIO.parse(handle, "fasta"))  # type: ignore[no-untyped-call]  # Bio.SeqIO is untyped
    return records


def read_protein_fasta(
    path: str, annotation: GenBankAnnotation
) -> dict[str, SeqRecord]:
    """The protein FASTA re-keyed from protein accession to locus tag.

    Each coding locus's ``protein_id`` must be in the FASTA; the record keeps the
    sequence, takes the locus tag as its id, and carries the accession and the
    original description in its description. Isoform proteins stay keyed by accession
    in the FASTA and are not included.
    """
    by_accession = read_fasta(path)
    proteins: dict[str, SeqRecord] = {}
    missing = []
    for tag, locus in annotation.loci.items():
        if locus.protein_id is None:
            continue
        record = by_accession.get(locus.protein_id)
        if record is None:
            missing.append(locus.protein_id)
            continue
        proteins[tag] = SeqRecord(
            record.seq, id=tag, name=tag, description=record.description
        )
    if missing:
        raise GenomeAnnotationMismatchError(
            f"{Path(path).name} lacks {len(missing)} protein ids the GenBank file "
            f"{annotation.member} names: {missing[:10]}"
        )
    return proteins


def _gff_attributes(column: str) -> dict[str, list[str]]:
    """GFF3 column 9 as ``{key: [values]}``: values split on ``,`` and percent-decoded."""
    attributes: dict[str, list[str]] = {}
    for pair in column.split(";"):
        if not pair:
            continue
        key, _, value = pair.partition("=")
        attributes[key] = [unquote(v) for v in value.split(",")]
    return attributes


class RefSeqGffAnnotation(BaseModel):
    """What a RefSeq (GCF) GFF3 adds to a GenBank assembly: the locus-tag crosswalk and
    the inline GO.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    member: str
    old_locus_tags: dict[str, tuple[str, ...]] = Field(
        description="RefSeq locus_tag -> the old_locus_tag values of its gene row."
    )
    ontology_term_rows: tuple[tuple[str, tuple[str, ...]], ...] = Field(
        description="(RefSeq locus_tag, GO ids) of every row with an Ontology_term."
    )


def read_refseq_gff(path: str, member: str) -> RefSeqGffAnnotation:
    """The ``old_locus_tag`` of every gene and pseudogene row and the ``GO:`` values of
    every ``Ontology_term`` attribute of a RefSeq GFF3. A locus tag whose rows disagree
    on ``old_locus_tag`` is refused.
    """
    old_locus_tags: dict[str, tuple[str, ...]] = {}
    rows: list[tuple[str, tuple[str, ...]]] = []
    with _open_text(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            columns = line.rstrip("\n").split("\t")
            if len(columns) != 9:
                raise ValueError(
                    f"{member}: a GFF3 row has 9 columns, got {len(columns)}: "
                    f"{line[:80]!r}"
                )
            attributes = _gff_attributes(columns[8])
            if columns[2] in ("gene", "pseudogene"):
                tag = attributes["locus_tag"][0]
                olds = tuple(attributes.get("old_locus_tag", []))
                if old_locus_tags.get(tag, olds) != olds:
                    raise ValueError(
                        f"{member}: rows of {tag} disagree on old_locus_tag: "
                        f"{old_locus_tags[tag]} and {olds}"
                    )
                old_locus_tags[tag] = olds
            if "Ontology_term" in attributes:
                terms = tuple(
                    t for t in attributes["Ontology_term"] if t.startswith("GO:")
                )
                bad = [t for t in terms if not GO_ID_PATTERN.fullmatch(t)]
                if bad:
                    raise ValueError(f"{member}: malformed GO ids {bad}")
                rows.append((attributes["locus_tag"][0], terms))
    return RefSeqGffAnnotation(
        member=member, old_locus_tags=old_locus_tags, ontology_term_rows=tuple(rows)
    )


class GafSynonymGo(BaseModel):
    """GO terms per locus tag read from GAF column 11 (DB Object Synonym)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    member: str
    rows: int
    not_rows: int = Field(description="Rows whose qualifier has NOT; excluded.")
    rows_without_identifier: int
    annotations: dict[str, tuple[str, ...]]


def read_gaf_synonym_go(
    path: str, member: str, identifier_pattern: str
) -> GafSynonymGo:
    """GO ids per identifier from a GAF 2.x whose column 11 carries the identifiers.

    Column 11 is split on ``|`` and each value on ``/`` (UniProt joins the ordered locus
    names of one protein encoded by several genes as ``b1239/b1240``); a token counts
    when it fully matches ``identifier_pattern``, so a versioned Blattner name such as
    ``b0018.1`` (a gene between b0018 and b0019) never reads as ``b0018``. Rows whose
    qualifier (column 4) contains ``NOT`` are counted and excluded. A row's GO id
    (column 5) is assigned to every identifier the row carries.
    """
    pattern = re.compile(identifier_pattern)
    rows = not_rows = without = 0
    annotations: dict[str, set[str]] = {}
    with _open_text(path) as handle:
        for line in handle:
            if line.startswith("!"):
                continue
            columns = line.rstrip("\n").split("\t")
            if len(columns) != GAF_COLUMNS:
                raise ValueError(
                    f"{member}: a GAF 2.x row has {GAF_COLUMNS} columns, got "
                    f"{len(columns)}: {line[:80]!r}"
                )
            rows += 1
            if "NOT" in columns[3].split("|"):
                not_rows += 1
                continue
            go_id = columns[4]
            if not GO_ID_PATTERN.fullmatch(go_id):
                raise ValueError(f"{member}: malformed GO id {go_id!r}")
            identifiers = {
                token
                for value in columns[10].split("|")
                for token in value.split("/")
                if pattern.fullmatch(token)
            }
            if not identifiers:
                without += 1
                continue
            for identifier in identifiers:
                annotations.setdefault(identifier, set()).add(go_id)
    return GafSynonymGo(
        member=member,
        rows=rows,
        not_rows=not_rows,
        rows_without_identifier=without,
        annotations={k: tuple(sorted(v)) for k, v in sorted(annotations.items())},
    )


def refseq_inline_go(
    refseq: RefSeqGffAnnotation,
) -> tuple[dict[str, tuple[str, ...]], int]:
    """GO ids per GenBank locus tag from the RefSeq ``Ontology_term`` rows, through the
    ``old_locus_tag`` of each row's RefSeq gene, and the number of rows whose RefSeq gene
    carries no ``old_locus_tag`` (a gene RefSeq added; it reaches no GenBank locus).
    """
    annotations: dict[str, set[str]] = {}
    without = 0
    for refseq_tag, terms in refseq.ontology_term_rows:
        if refseq_tag not in refseq.old_locus_tags:
            raise ValueError(
                f"{refseq.member}: an Ontology_term row names {refseq_tag}, which has "
                "no gene or pseudogene row"
            )
        olds = refseq.old_locus_tags[refseq_tag]
        if not olds:
            without += 1
            continue
        for old in olds:
            annotations.setdefault(old, set()).update(terms)
    return ({k: tuple(sorted(v)) for k, v in sorted(annotations.items())}, without)


GoRoute = Literal["gaf_synonym_column", "refseq_gff_ontology_term"]


class GoSourceSpec(BaseModel):
    """Where a bacterial genome reads its GO annotation from, and on which column."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    route: GoRoute
    assembly_set: str
    member: str
    identifier_pattern: str | None = Field(
        description="The locus-tag pattern a GAF column-11 token must fully match; "
        "None for the RefSeq GFF route, whose identifier is the row's locus_tag."
    )

    @model_validator(mode="after")
    def _pattern_matches_route(self) -> "GoSourceSpec":
        if (self.route == "gaf_synonym_column") != (
            self.identifier_pattern is not None
        ):
            raise ValueError(
                "identifier_pattern is required for the gaf_synonym_column route and "
                "absent for the refseq_gff_ontology_term route"
            )
        return self

    @property
    def identifier(self) -> str:
        """The identifier column, in words."""
        if self.route == "gaf_synonym_column":
            return (
                "GAF column 11 (DB Object Synonym), values split on '|' and '/', "
                f"tokens fully matching {self.identifier_pattern}"
            )
        return (
            "RefSeq GFF Ontology_term row's locus_tag, through its gene row's "
            "old_locus_tag"
        )


class GoAnnotationSource(BaseModel):
    """What a genome's GO route read and reached, measured at construction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    route: GoRoute
    assembly_set: str
    member: str
    sha256: str
    identifier: str
    rows: int
    not_rows_excluded: int
    rows_without_identifier: int
    identifiers: int = Field(description="Distinct identifiers the rows reached.")
    identifiers_not_in_annotation: tuple[str, ...] = Field(
        description="Identifiers that are no locus of the GenBank annotation."
    )
    pseudogenes_annotated: int
    genes_annotated: int = Field(description="Gene-set members with >= 1 GO id.")
    terms: int = Field(description="Distinct GO ids over the gene-set members.")


class BacterialAssembly(BaseModel):
    """One NCBI bacterial assembly as a genome class reads it from its assembly set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    organism: str
    strain: str
    assembly_set: str
    genbank_assembly: str = Field(description='e.g. "GCA_000005845.2_ASM584v2".')
    refseq_assembly: str = Field(description='e.g. "GCF_000005845.2_ASM584v2".')
    replicon: str = Field(description="The GenBank accession of the one replicon.")
    locus_tag_pattern: str = Field(
        description="Every GenBank locus tag of the assembly fully matches it."
    )
    go_source: GoSourceSpec
    default_genome_root: str = Field(
        description="The default data.db cache root, relative like data/sgd/genome."
    )

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
        """The replicon sequence."""
        return f"{self.genbank_assembly}_genomic.fna.gz"

    @property
    def protein_fasta_member(self) -> str:
        """The proteins, keyed by protein accession."""
        return f"{self.genbank_assembly}_protein.faa.gz"

    @property
    def refseq_gff_member(self) -> str:
        """The RefSeq GFF3: the old_locus_tag crosswalk and the inline GO."""
        return f"{self.refseq_assembly}_genomic.gff.gz"


@define(repr=False)
class BacterialGene(AnnotatedGene):
    """A gene of a :class:`BacterialGenome`: coordinates from ``data.db``, names and
    product from its :class:`GenBankLocus`, GO from the genome's GO route.

    ``go`` is the GO ids the route assigns the locus (``None`` when the route reaches
    it with none). ``alias`` is the ``/gene_synonym`` values (``None`` when there are
    none), ``name`` the GFF ``Name``; ``symbol``, ``synonyms``, ``old_locus_tags``,
    ``product``, ``protein_id``, ``db_xrefs`` and ``pseudo`` come from the locus.
    """

    #: NCBI writes a gene's GFF ``ID`` as ``gene-<locus_tag>``.
    FEATURE_ID_PREFIX: ClassVar[str] = "gene-"
    #: GFF seqid (the GenBank replicon accession) to the integer chromosome key.
    REPLICONS: ClassVar[Mapping[str, int]]

    locus: GenBankLocus = field(kw_only=True, repr=False)
    go_terms: SortedSet[str] | None = field(kw_only=True, repr=False)

    @classmethod
    def seqid_to_chromosome(cls, seqid: str) -> int:
        """The chromosome key of a replicon in :attr:`REPLICONS`; any other is refused."""
        if seqid not in cls.REPLICONS:
            raise ValueError(
                f"{cls.__name__}: seqid {seqid!r} is not one of the replicons "
                f"{sorted(cls.REPLICONS)}"
            )
        return cls.REPLICONS[seqid]

    def annotate(self, gene_feature: Feature) -> None:
        """GFF ``Name`` and ``Note``; names, product and xrefs from the GenBank locus;
        GO from the genome's route (never from a GFF attribute).
        """
        self.name = gene_feature.attributes.get("Name", None)
        self.note = gene_feature.attributes.get("Note", None)
        self.alias = list(self.locus.synonyms) or None
        self.go = self.go_terms
        self.symbol = self.locus.symbol
        self.synonyms = list(self.locus.synonyms)
        self.old_locus_tags = list(self.locus.old_locus_tags)
        self.product = self.locus.product
        self.protein_id = self.locus.protein_id
        self.db_xrefs = list(self.locus.db_xrefs)
        self.pseudo = self.locus.pseudo


def _lookup(
    layer: tuple[dict[str, list[str]], dict[str, list[str]]], name: str
) -> tuple[list[str], bool]:
    """Loci a layer maps ``name`` to, and whether the match was case-insensitive."""
    exact, folded = layer
    if name in exact:
        return sorted(set(exact[name])), False
    return sorted(set(folded.get(name.upper(), []))), True


@define(eq=False)
class BacterialGenome[GeneT: BacterialGene](AnnotatedGenome[GeneT]):
    """A bacterial genome read GenBank first from one NCBI assembly set.

    The ``data.db`` cache contract is the base class's
    (:class:`~torchcell.sequence.genome.base.AnnotatedGenome`). On top of it:
    ``genbank`` (the :class:`GenBankAnnotation`), ``refseq`` (the
    :class:`RefSeqGffAnnotation`), ``go_annotations`` (locus tag to GO ids, every locus
    the route reaches, pseudogenes included) and ``go_source`` (what the route
    reached). A subclass sets :attr:`ASSEMBLY` and the base class variables, and its
    :meth:`gene_class`.
    """

    #: The assembly this class reads.
    ASSEMBLY: ClassVar[BacterialAssembly]
    #: The gene-like locus types of an NCBI GenBank GFF3.
    LOCUS_FEATURE_TYPES: ClassVar[frozenset[str]] = frozenset({"gene", "pseudogene"})
    #: The name layers after the exact locus tag, in resolution order: (index key,
    #: how a resolution ``note`` names the layer).
    NAME_LAYERS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("old_locus_tag", "old locus tag"),
        ("refseq_locus_tag", "RefSeq locus tag"),
        ("symbol", "gene symbol"),
        ("synonym", "gene synonym"),
    )

    genbank: GenBankAnnotation = field(init=False, default=None, repr=False)
    refseq: RefSeqGffAnnotation = field(init=False, default=None, repr=False)
    go_annotations: dict[str, SortedSet[str]] = field(
        init=False, factory=dict, repr=False
    )
    go_source: GoAnnotationSource = field(init=False, default=None, repr=False)
    _genbank_path: str = field(init=False, default=None, repr=False)

    @classmethod
    def release_files(cls) -> GenomeReleaseFiles:
        """The GenBank GFF, replicon FASTA and protein FASTA; no CDS FASTA (the CDS
        sequences come from the flat file's CDS features).
        """
        return GenomeReleaseFiles(
            dna_fasta=cls.ASSEMBLY.dna_fasta_member,
            gff=cls.ASSEMBLY.gff_member,
            protein_fasta=cls.ASSEMBLY.protein_fasta_member,
            cds_fasta=None,
        )

    @classmethod
    def fasta_chromosome(cls, record: Any) -> int:
        """The assembly's one replicon is chromosome 1; any other record is refused."""
        if record.id != cls.ASSEMBLY.replicon:
            raise ValueError(
                f"{cls.__name__}: FASTA record {record.id!r} is not the replicon "
                f"{cls.ASSEMBLY.replicon}"
            )
        return cls.gene_class().seqid_to_chromosome(record.id)

    def _prepare_go_obo(self) -> str:
        """``go-basic.obo`` of the ``go_release_2026-08-05`` set; never a download."""
        return resolve(GO_RELEASE_20260805, GO_BASIC_OBO)

    def _read_sequences(self) -> None:
        """The GenBank flat file (loci and CDS), the replicon FASTA and the protein
        FASTA re-keyed to locus tags. Refuses a flat file whose replicons are not the
        assembly's one replicon or whose locus tags leave the assembly's pattern.
        """
        member = self.ASSEMBLY.genbank_member
        self._genbank_path = resolve(self.ASSEMBLY_SET, member)
        self.genbank, self.fasta_cds = read_genbank(self._genbank_path, member)
        accessions = [r.accession for r in self.genbank.replicons]
        if accessions != [self.ASSEMBLY.replicon]:
            raise GenomeAnnotationMismatchError(
                f"{member} holds replicons {accessions}, not the one replicon "
                f"{self.ASSEMBLY.replicon}"
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

    def __attrs_post_init__(self) -> None:
        """The base construction, then the GenBank/database agreement check, the RefSeq
        crosswalk and the GO route.
        """
        super().__attrs_post_init__()
        self._check_database_matches_genbank()
        refseq_member = self.ASSEMBLY.refseq_gff_member
        self.refseq = read_refseq_gff(
            resolve(self.ASSEMBLY_SET, refseq_member), refseq_member
        )
        self.go_annotations, self.go_source = self._load_go()

    def _check_database_matches_genbank(self) -> None:
        """``data.db``'s gene features are exactly the GenBank non-pseudo loci (each
        ``ID`` is the prefixed locus tag) and its pseudogene features carry exactly the
        GenBank pseudo loci (a joined pseudogene is several rows with one locus tag).
        """
        prefix = self.gene_class().FEATURE_ID_PREFIX
        db_genes: set[str] = set()
        db_pseudogenes: set[str] = set()
        for feature in self.db.features_of_type(("gene", "pseudogene")):
            tag = feature.attributes["locus_tag"][0]
            if feature.featuretype == "pseudogene":
                db_pseudogenes.add(tag)
                continue
            if feature.id != prefix + tag:
                raise GenomeAnnotationMismatchError(
                    f"data.db gene {feature.id!r} does not carry its locus tag {tag!r} "
                    f"as {prefix + tag!r}"
                )
            db_genes.add(tag)
        genbank_genes = set(self.genbank.gene_tags)
        genbank_pseudogenes = set(self.genbank.pseudogene_tags)
        if db_genes != genbank_genes or db_pseudogenes != genbank_pseudogenes:
            raise GenomeAnnotationMismatchError(
                f"{self.ASSEMBLY.gff_member} and {self.ASSEMBLY.genbank_member} "
                "disagree: genes only in the GFF "
                f"{sorted(db_genes - genbank_genes)[:10]}, only in the GenBank file "
                f"{sorted(genbank_genes - db_genes)[:10]}; pseudogenes only in the GFF "
                f"{sorted(db_pseudogenes - genbank_pseudogenes)[:10]}, only in the "
                f"GenBank file {sorted(genbank_pseudogenes - db_pseudogenes)[:10]}"
            )

    def _load_go(self) -> tuple[dict[str, SortedSet[str]], GoAnnotationSource]:
        """Read the assembly's GO route and keep what reaches a GenBank locus."""
        spec = self.ASSEMBLY.go_source
        path = resolve(spec.assembly_set, spec.member)
        match spec.route:
            case "gaf_synonym_column":
                assert spec.identifier_pattern is not None  # the spec validator's rule
                gaf = read_gaf_synonym_go(path, spec.member, spec.identifier_pattern)
                raw = gaf.annotations
                rows, not_rows, without = (
                    gaf.rows,
                    gaf.not_rows,
                    gaf.rows_without_identifier,
                )
            case "refseq_gff_ontology_term":
                if (spec.assembly_set, spec.member) != (
                    self.ASSEMBLY_SET,
                    self.ASSEMBLY.refseq_gff_member,
                ):
                    raise ValueError(
                        f"{type(self).__name__}: the RefSeq GFF GO route reads this "
                        f"assembly's own {self.ASSEMBLY.refseq_gff_member}, not "
                        f"{spec.assembly_set}/{spec.member}"
                    )
                raw, without = refseq_inline_go(self.refseq)
                rows, not_rows = len(self.refseq.ontology_term_rows), 0
        loci = self.genbank.loci
        stored = {tag: SortedSet(terms) for tag, terms in raw.items() if tag in loci}
        genes = [tag for tag in stored if not loci[tag].pseudo]
        source = GoAnnotationSource(
            route=spec.route,
            assembly_set=spec.assembly_set,
            member=spec.member,
            sha256=sha256_file(Path(path)),
            identifier=spec.identifier,
            rows=rows,
            not_rows_excluded=not_rows,
            rows_without_identifier=without,
            identifiers=len(raw),
            identifiers_not_in_annotation=tuple(sorted(set(raw) - set(loci))),
            pseudogenes_annotated=len(stored) - len(genes),
            genes_annotated=len(genes),
            terms=len({t for tag in genes for t in stored[tag]}),
        )
        return stored, source

    def compute_gene_set(self) -> GeneSet:
        """The locus tags of the GenBank file's non-pseudo ``gene`` features."""
        return GeneSet(self.genbank.gene_tags)

    def __getitem__(self, item: str) -> GeneT | None:
        """The gene (or pseudogene) with this locus tag, or None when it is absent: not
        a locus of the annotation, or a gene this instance dropped from its gene set
        (``drop_empty_go``).

        A joined pseudogene (an IS insertion splits it into segments) is refused by
        name: one start/end interval cannot represent it.
        """
        locus = self.genbank.loci.get(item)
        if locus is None or not (locus.pseudo or item in self.gene_set):
            log.warning("%s is not a gene or pseudogene of this genome", item)
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
            go_terms=self.go_annotations.get(item),
        )

    @property
    def locus_table(self) -> pd.DataFrame:
        """One row per GenBank locus (every :class:`GenBankLocus` field)."""
        return pd.DataFrame(
            [locus.model_dump() for locus in self.genbank.loci.values()]
        )

    @property
    def alias_to_systematic(self) -> dict[str, list[str]]:
        """Each gene symbol and ``gene_synonym`` (exact case) to the gene-set locus tags
        that carry it. :meth:`resolve_gene_name` is the ordered resolver; this map does
        not rank a symbol above a synonym.
        """
        if self._alias_to_systematic is None:
            alias_map: dict[str, list[str]] = {}
            for tag in self.gene_set:
                locus = self.genbank.loci[tag]
                names = ([locus.symbol] if locus.symbol else []) + list(locus.synonyms)
                for name in dict.fromkeys(names):
                    alias_map.setdefault(name, []).append(tag)
            self._alias_to_systematic = alias_map
        return self._alias_to_systematic

    @property
    def feature_index(self) -> dict[str, Any]:
        """The resolution index over the current gene set and the pseudogenes.

        Keys: ``genes`` (gene-set locus tags), ``pseudogenes``, and per layer
        (``locus_tag``, ``old_locus_tag``, ``refseq_locus_tag``, ``symbol``,
        ``synonym``) a pair of maps, exact name -> locus tags and upper-cased name ->
        locus tags. A locus dropped from the gene set (``drop_empty_go``) is in no
        layer.
        """
        if self._feature_index is None:
            genes = set(self.gene_set)
            loci = {
                tag: locus
                for tag, locus in self.genbank.loci.items()
                if tag in genes or locus.pseudo
            }
            exact: dict[str, dict[str, list[str]]] = {
                key: {} for key in ("locus_tag", *(k for k, _ in self.NAME_LAYERS))
            }
            for tag, locus in loci.items():
                exact["locus_tag"].setdefault(tag, []).append(tag)
                for old in locus.old_locus_tags:
                    exact["old_locus_tag"].setdefault(old, []).append(tag)
                if locus.symbol:
                    exact["symbol"].setdefault(locus.symbol, []).append(tag)
                for synonym in locus.synonyms:
                    exact["synonym"].setdefault(synonym, []).append(tag)
            for refseq_tag, olds in self.refseq.old_locus_tags.items():
                for old in olds:
                    if old in loci:
                        exact["refseq_locus_tag"].setdefault(refseq_tag, []).append(old)
            index: dict[str, Any] = {
                "genes": genes,
                "pseudogenes": {t for t, locus in loci.items() if locus.pseudo},
            }
            for key, names in exact.items():
                folded: dict[str, list[str]] = {}
                for name, tags in names.items():
                    folded.setdefault(name.upper(), []).extend(tags)
                index[key] = (names, folded)
            self._feature_index = index
        return self._feature_index

    def resolve_gene_name(self, name: str) -> GeneNameResolution:
        """Reconcile a source gene name to this GenBank annotation (layered).

        Layers, in order: (1) a locus tag: ``CURRENT`` for a gene, ``NON_GENE_FEATURE``
        for a pseudogene; (2) an ``old_locus_tag``; (3) a RefSeq locus tag whose
        ``old_locus_tag`` is a locus here; (4) a gene symbol; (5) a ``gene_synonym``
        (ECK and JW numbers in E. coli); (6) not found: ``RETIRED``, retained as given.
        In layers 2 to 5 a unique gene is ``RENAMED``, several genes are ``AMBIGUOUS``,
        a unique pseudogene is ``NON_GENE_FEATURE``. Within every layer an exact-case
        match wins; a case-insensitive match is used only when there is none and is
        stated in the ``note``. Returned locus tags keep the annotation's case. Pure
        and per name; callers decide retention.
        """
        raw = name
        key = name.strip()
        index = self.feature_index
        genes: set[str] = index["genes"]
        tags, folded = _lookup(index["locus_tag"], key)
        via = " (case-insensitive match)" if folded else ""
        if len(tags) == 1:
            tag = tags[0]
            if tag in genes:
                return GeneNameResolution(
                    input_name=raw,
                    status=GeneNameStatus.CURRENT,
                    systematic_name=tag,
                    note=f"locus tag{via}" if folded else None,
                )
            return GeneNameResolution(
                input_name=raw,
                status=GeneNameStatus.NON_GENE_FEATURE,
                systematic_name=tag,
                feature_type="pseudogene",
                note=f"valid {self.ANNOTATION_NAME} pseudogene, not a gene feature{via}",
            )
        if len(tags) > 1:
            return GeneNameResolution(
                input_name=raw,
                status=GeneNameStatus.AMBIGUOUS,
                systematic_name=None,
                candidates=tags,
                note=f"locus tag of multiple loci{via}",
            )
        for layer_key, layer in self.NAME_LAYERS:
            ids, folded = _lookup(index[layer_key], key)
            if not ids:
                continue
            via = " (case-insensitive match)" if folded else ""
            gene_ids = [i for i in ids if i in genes]
            if len(gene_ids) == 1:
                return GeneNameResolution(
                    input_name=raw,
                    status=GeneNameStatus.RENAMED,
                    systematic_name=gene_ids[0],
                    note=f"{layer} of current gene {gene_ids[0]}{via}",
                )
            if len(gene_ids) > 1:
                return GeneNameResolution(
                    input_name=raw,
                    status=GeneNameStatus.AMBIGUOUS,
                    systematic_name=None,
                    candidates=gene_ids,
                    note=f"{layer} of multiple current genes{via}",
                )
            if len(ids) == 1:
                return GeneNameResolution(
                    input_name=raw,
                    status=GeneNameStatus.NON_GENE_FEATURE,
                    systematic_name=ids[0],
                    feature_type="pseudogene",
                    note=f"{layer} of pseudogene {ids[0]} (not a gene feature){via}",
                )
            return GeneNameResolution(
                input_name=raw,
                status=GeneNameStatus.AMBIGUOUS,
                systematic_name=None,
                candidates=ids,
                note=f"{layer} of multiple pseudogenes{via}",
            )
        return GeneNameResolution(
            input_name=raw,
            status=GeneNameStatus.RETIRED,
            systematic_name=key,
            note=f"not found in {self.ANNOTATION_RELEASE}; retained as given",
        )

    def remove_deprecated_go_terms(self) -> None:
        """Drop GO ids absent from or obsolete in the GO DAG from this instance's
        ``go_annotations`` (a locus left with none loses its entry). GO is held in
        memory here, not in ``data.db``, so nothing is written to a database copy.
        """
        dag = self.go_dag
        kept: dict[str, SortedSet[str]] = {}
        for tag, terms in self.go_annotations.items():
            live = SortedSet(t for t in terms if t in dag and not dag[t].is_obsolete)
            if live:
                kept[tag] = live
        self.go_annotations = kept
        self._go = None
        self._go_genes = None
