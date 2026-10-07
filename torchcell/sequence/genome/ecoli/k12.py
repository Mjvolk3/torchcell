# torchcell/sequence/genome/ecoli/k12
# [[torchcell.sequence.genome.ecoli.k12]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/sequence/genome/ecoli/k12
# Test file: tests/torchcell/sequence/genome/ecoli/test_k12.py

r"""E. coli K-12 genomes, MG1655 and BW25113, read GenBank first from the genomes tier.

:class:`EcoliK12Genome` is parameterized by strain: each strain is a
:class:`~torchcell.sequence.genome.bacterial.BacterialAssembly` naming its assembly
set, locus-tag pattern and GO source, and a concrete class per strain binds it
(:class:`EcoliK12MG1655Genome`, :class:`EcoliK12BW25113Genome`;
:meth:`EcoliK12Genome.for_strain` picks one by name). One class per strain keeps the
assembly set a class-level fact, which the base class's cache checks read without an
instance.

* **MG1655** (:data:`MG1655_ASSEMBLY`): GCA_000005845.2, replicon U00096.3, locus tags
  ``b\d{4}``. GO from the GO Consortium's ``ECOLI-uniprot.gaf.gz`` (2026-08-05 release),
  read on column 11, where UniProt lists the b-numbers.
* **BW25113** (:data:`BW25113_ASSEMBLY`, the Keio background): GCA_000750555.1,
  replicon CP009273.1, locus tags ``BW25113_\d{4}``. Decision D9 of
  [[plan.bacteria-ontology-genome]]: the stored GO is BW25113's own RefSeq
  annotation, the RefSeq GFF's inline ``Ontology_term`` (IEA) crosswalked to the GenBank
  tags through ``old_locus_tag``. MG1655's GAF mapped through the ECK synonym is an
  inference (a BW25113 gene inherits its MG1655 partner's annotation), offered only as
  the explicitly named derived view
  :meth:`EcoliK12BW25113Genome.go_annotations_via_mg1655_eck`, which no genome stores.

A b-number is not derivable from a ``BW25113_`` number: 45 BW25113 genes carry a number
other than their MG1655 b-number. The ECK synonym is the published 1:1 key between the
two strains (:func:`eck_crosswalk`).
"""

import re
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar, Literal

from attrs import define, field
from pydantic import BaseModel, ConfigDict, Field

from torchcell.literature.manifest import sha256_file
from torchcell.sequence.genome.bacterial import (
    BacterialAssembly,
    BacterialGene,
    BacterialGenome,
    GenBankAnnotation,
    GoSourceSpec,
    read_gaf_synonym_go,
    read_genbank,
)
from torchcell.sequence.genome.registry import (
    ECOLI_K12_BW25113,
    ECOLI_K12_MG1655,
    resolve,
)

#: MG1655 locus tags: Blattner b-numbers.
MG1655_LOCUS_TAG_PATTERN = r"b\d{4}"
#: BW25113 GenBank locus tags (CP009273.1).
BW25113_LOCUS_TAG_PATTERN = r"BW25113_\d{4}"
#: The EcoCyc ECK accession, a ``gene_synonym`` in both strains' GenBank files.
ECK_PATTERN = re.compile(r"ECK\d{4}")

MG1655_ASSEMBLY = BacterialAssembly(
    organism="Escherichia coli",
    strain="K-12 MG1655",
    assembly_set=ECOLI_K12_MG1655,
    genbank_assembly="GCA_000005845.2_ASM584v2",
    refseq_assembly="GCF_000005845.2_ASM584v2",
    replicon="U00096.3",
    locus_tag_pattern=MG1655_LOCUS_TAG_PATTERN,
    go_source=GoSourceSpec(
        route="gaf_synonym_column",
        assembly_set=ECOLI_K12_MG1655,
        member="ECOLI-uniprot.gaf.gz",
        identifier_pattern=MG1655_LOCUS_TAG_PATTERN,
    ),
    default_genome_root="data/ecoli/mg1655/genome",
)

BW25113_ASSEMBLY = BacterialAssembly(
    organism="Escherichia coli",
    strain="K-12 BW25113",
    assembly_set=ECOLI_K12_BW25113,
    genbank_assembly="GCA_000750555.1_ASM75055v1",
    refseq_assembly="GCF_000750555.1_ASM75055v1",
    replicon="CP009273.1",
    locus_tag_pattern=BW25113_LOCUS_TAG_PATTERN,
    go_source=GoSourceSpec(
        route="refseq_gff_ontology_term",
        assembly_set=ECOLI_K12_BW25113,
        member="GCF_000750555.1_ASM75055v1_genomic.gff.gz",
        identifier_pattern=None,
    ),
    default_genome_root="data/ecoli/bw25113/genome",
)

EcoliK12StrainName = Literal["MG1655", "BW25113"]


@define(repr=False)
class EcoliK12Gene(BacterialGene):
    """A K-12 gene: the one chromosome of either strain is chromosome 1."""

    REPLICONS: ClassVar[MappingProxyType[str, int]] = MappingProxyType(
        {MG1655_ASSEMBLY.replicon: 1, BW25113_ASSEMBLY.replicon: 1}
    )


@define(eq=False)
class EcoliK12Genome(BacterialGenome[EcoliK12Gene]):
    """An E. coli K-12 genome; a strain class binds :attr:`ASSEMBLY`.

    Construct a strain class directly, or by name through :meth:`for_strain`. The
    ``data.db`` cache, GenBank ingest, GO route and name resolution are
    :class:`~torchcell.sequence.genome.bacterial.BacterialGenome`'s.
    """

    @classmethod
    def gene_class(cls) -> type[EcoliK12Gene]:
        """K-12 genes are :class:`EcoliK12Gene`."""
        return EcoliK12Gene

    @property
    def strain(self) -> str:
        """The strain this genome reads, e.g. ``"K-12 BW25113"``."""
        return self.ASSEMBLY.strain

    @staticmethod
    def for_strain(
        strain: EcoliK12StrainName, genome_root: str, overwrite: bool = False
    ) -> "EcoliK12Genome":
        """The genome of ``strain`` (``"MG1655"`` or ``"BW25113"``) on ``genome_root``."""
        if strain not in ECOLI_K12_GENOMES:
            raise ValueError(
                f"unknown E. coli K-12 strain {strain!r}; known: "
                f"{sorted(ECOLI_K12_GENOMES)}"
            )
        return ECOLI_K12_GENOMES[strain](genome_root=genome_root, overwrite=overwrite)


@define(eq=False)
class EcoliK12MG1655Genome(EcoliK12Genome):
    """E. coli K-12 MG1655 (GCA_000005845.2): b-numbers, GO from ECOLI-uniprot.gaf."""

    ASSEMBLY: ClassVar[BacterialAssembly] = MG1655_ASSEMBLY
    ASSEMBLY_SET: ClassVar[str] = ECOLI_K12_MG1655
    GENOME_VERSION: ClassVar[str] = "ASM584v2"
    ANNOTATION_NAME: ClassVar[str] = "GenBank GCA_000005845.2"
    ANNOTATION_RELEASE: ClassVar[str] = "GCA_000005845.2_ASM584v2"

    genome_root: str = field(
        init=True, repr=False, default=MG1655_ASSEMBLY.default_genome_root
    )
    overwrite: bool = field(init=True, repr=True, default=False)


class EckPair(BaseModel):
    """One ECK accession carried by exactly one locus in each K-12 strain."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    eck: str
    mg1655: str
    bw25113: str

    @property
    def numerics_agree(self) -> bool:
        """Whether the b-number and the ``BW25113_`` number have the same digits."""
        return self.mg1655.removeprefix("b") == self.bw25113.removeprefix("BW25113_")


class EckCrosswalk(BaseModel):
    """The ECK synonym join between the MG1655 and BW25113 GenBank annotations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    shared: tuple[str, ...] = Field(description="ECK ids carried in both strains.")
    pairs: tuple[EckPair, ...] = Field(
        description="Shared ECK ids carried by exactly one locus in each strain."
    )
    mg1655_only: tuple[str, ...]
    bw25113_only: tuple[str, ...]

    @property
    def numeric_disagreements(self) -> tuple[EckPair, ...]:
        """One-to-one pairs whose b-number and ``BW25113_`` number differ."""
        return tuple(pair for pair in self.pairs if not pair.numerics_agree)

    @property
    def not_one_to_one(self) -> tuple[str, ...]:
        """Shared ECK ids that some strain carries on more than one locus."""
        paired = {pair.eck for pair in self.pairs}
        return tuple(eck for eck in self.shared if eck not in paired)


def eck_loci(annotation: GenBankAnnotation) -> dict[str, list[str]]:
    """Each ECK accession among the ``gene_synonym`` values to the loci carrying it."""
    loci: dict[str, list[str]] = {}
    for tag, locus in annotation.loci.items():
        for synonym in locus.synonyms:
            if ECK_PATTERN.fullmatch(synonym):
                loci.setdefault(synonym, []).append(tag)
    return loci


def eck_crosswalk(
    mg1655: GenBankAnnotation, bw25113: GenBankAnnotation
) -> EckCrosswalk:
    """Join the MG1655 and BW25113 annotations on their ECK synonyms, every locus
    (pseudogenes included). Refuses annotations whose locus tags are not the named
    strain's.
    """
    for annotation, pattern, strain in (
        (mg1655, MG1655_LOCUS_TAG_PATTERN, "MG1655"),
        (bw25113, BW25113_LOCUS_TAG_PATTERN, "BW25113"),
    ):
        if not all(re.fullmatch(pattern, tag) for tag in annotation.loci):
            raise ValueError(
                f"{annotation.member} is not a {strain} annotation (locus tags outside "
                f"{pattern})"
            )
    mg_eck, bw_eck = eck_loci(mg1655), eck_loci(bw25113)
    shared = sorted(set(mg_eck) & set(bw_eck))
    pairs = tuple(
        EckPair(eck=eck, mg1655=mg_eck[eck][0], bw25113=bw_eck[eck][0])
        for eck in shared
        if len(mg_eck[eck]) == 1 and len(bw_eck[eck]) == 1
    )
    return EckCrosswalk(
        shared=tuple(shared),
        pairs=pairs,
        mg1655_only=tuple(sorted(set(mg_eck) - set(bw_eck))),
        bw25113_only=tuple(sorted(set(bw_eck) - set(mg_eck))),
    )


class DerivedGoAnnotation(BaseModel):
    """A GO view inferred from another strain's annotation; never stored on a genome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    view: Literal["mg1655_gaf_via_eck"]
    statement: str
    basis_assembly_set: str
    basis_member: str
    basis_sha256: str
    crosswalk_pairs: int
    annotations: dict[str, tuple[str, ...]] = Field(
        description="BW25113 locus tag -> GO ids of its one-to-one ECK partner."
    )
    genes_annotated: int = Field(description="BW25113 gene-set members with GO.")
    terms: int = Field(description="Distinct GO ids over those genes.")


@define(eq=False)
class EcoliK12BW25113Genome(EcoliK12Genome):
    """E. coli K-12 BW25113 (GCA_000750555.1), the Keio background.

    ``go_annotations`` is the RefSeq GFF's inline GO through ``old_locus_tag`` (D9);
    :meth:`go_annotations_via_mg1655_eck` is the labeled MG1655-derived alternative.
    """

    ASSEMBLY: ClassVar[BacterialAssembly] = BW25113_ASSEMBLY
    ASSEMBLY_SET: ClassVar[str] = ECOLI_K12_BW25113
    GENOME_VERSION: ClassVar[str] = "ASM75055v1"
    ANNOTATION_NAME: ClassVar[str] = "GenBank GCA_000750555.1"
    ANNOTATION_RELEASE: ClassVar[str] = "GCA_000750555.1_ASM75055v1"

    genome_root: str = field(
        init=True, repr=False, default=BW25113_ASSEMBLY.default_genome_root
    )
    overwrite: bool = field(init=True, repr=True, default=False)

    def mg1655_eck_crosswalk(self) -> EckCrosswalk:
        """The ECK join of this annotation with MG1655's GenBank file (read from the
        MG1655 set; no MG1655 genome or cache is built).
        """
        member = MG1655_ASSEMBLY.genbank_member
        mg1655, _ = read_genbank(resolve(MG1655_ASSEMBLY.assembly_set, member), member)
        return eck_crosswalk(mg1655, self.genbank)

    def go_annotations_via_mg1655_eck(self) -> DerivedGoAnnotation:
        """DERIVED VIEW, never stored: MG1655's GAF GO on each BW25113 locus whose ECK
        synonym is carried by exactly one locus in each strain.

        It asserts that a BW25113 gene has its MG1655 partner's annotation, which is an
        inference across strains, not a fact about BW25113. ``go_annotations`` (the
        RefSeq inline GO, D9) is unchanged by calling it.
        """
        crosswalk = self.mg1655_eck_crosswalk()
        spec = MG1655_ASSEMBLY.go_source
        assert spec.identifier_pattern is not None  # a GAF route always names one
        path = resolve(spec.assembly_set, spec.member)
        gaf = read_gaf_synonym_go(path, spec.member, spec.identifier_pattern)
        annotations = {
            pair.bw25113: gaf.annotations[pair.mg1655]
            for pair in crosswalk.pairs
            if pair.mg1655 in gaf.annotations
        }
        genes = set(self.gene_set)
        return DerivedGoAnnotation(
            view="mg1655_gaf_via_eck",
            statement=(
                "Inferred: each BW25113 locus carries the GO of the MG1655 locus that "
                "shares its ECK synonym one-to-one. Not BW25113's own annotation."
            ),
            basis_assembly_set=spec.assembly_set,
            basis_member=spec.member,
            basis_sha256=sha256_file(Path(path)),
            crosswalk_pairs=len(crosswalk.pairs),
            annotations=dict(sorted(annotations.items())),
            genes_annotated=sum(1 for tag in annotations if tag in genes),
            terms=len(
                {t for tag, terms in annotations.items() if tag in genes for t in terms}
            ),
        )


#: The K-12 genome class of each strain name.
ECOLI_K12_GENOMES: dict[str, type[EcoliK12Genome]] = {
    "MG1655": EcoliK12MG1655Genome,
    "BW25113": EcoliK12BW25113Genome,
}
