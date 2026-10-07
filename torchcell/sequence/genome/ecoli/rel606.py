# torchcell/sequence/genome/ecoli/rel606
# [[torchcell.sequence.genome.ecoli.rel606]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/sequence/genome/ecoli/rel606
# Test file: tests/torchcell/sequence/genome/ecoli/test_rel606.py

r"""E. coli B REL606, read GenBank first from the genomes tier.

REL606 is the ancestor of the Lenski long-term evolution experiment and an E. coli
**B** strain, so it is a host of its own beside the K-12 strains of
:mod:`torchcell.sequence.genome.ecoli.k12`: an ``ECB_`` locus tag is neither a b-number
nor a ``BW25113_`` number, and the ECK crosswalk (an EcoCyc K-12 accession) does not
apply.

GCA_000017985.1 (ASM1798v1), one circular replicon CP000819.1 (RefSeq NC_012967.1,
4,629,812 bp). Locus tags are ``ECB_\d{5}`` for the numbered loci, ``ECB_t\d{5}`` for
tRNAs and ``ECB_r\d{5}`` for rRNAs (:data:`REL606_LOCUS_TAG_PATTERN`). RefSeq retags to
``ECB_RS...`` and keeps the GenBank tag as ``old_locus_tag``, which is the crosswalk the
resolver's RefSeq layer reads.

GO: the GO Consortium release has no E. coli B GAF and EBI GOA publishes no proteome file
for taxon 413997, so the GO source is the RefSeq GFF's inline ``Ontology_term`` rows
(IEA, from NCBI PGAP) mapped to the GenBank tags through ``old_locus_tag``, the route
BW25113 uses. NCBI's ``_gene_ontology.gaf.gz`` is a member of the set but is keyed on
``WP_`` proteins and names no locus tag.
"""

from types import MappingProxyType
from typing import ClassVar, Literal

from attrs import define, field

from torchcell.sequence.genome.bacterial import (
    BacterialAssembly,
    BacterialGene,
    BacterialGenome,
    GoSourceSpec,
)
from torchcell.sequence.genome.registry import ECOLI_B_REL606

#: REL606 GenBank locus tags (CP000819.1): numbered, tRNA (``t``) and rRNA (``r``) loci.
REL606_LOCUS_TAG_PATTERN = r"ECB_[rt]?\d{5}"

REL606_ASSEMBLY = BacterialAssembly(
    organism="Escherichia coli",
    strain="B REL606",
    assembly_set=ECOLI_B_REL606,
    genbank_assembly="GCA_000017985.1_ASM1798v1",
    refseq_assembly="GCF_000017985.1_ASM1798v1",
    replicon="CP000819.1",
    locus_tag_pattern=REL606_LOCUS_TAG_PATTERN,
    go_source=GoSourceSpec(
        route="refseq_gff_ontology_term",
        assembly_set=ECOLI_B_REL606,
        member="GCF_000017985.1_ASM1798v1_genomic.gff.gz",
        identifier_pattern=None,
    ),
    default_genome_root="data/ecoli/rel606/genome",
)

EcoliBStrainName = Literal["REL606"]


@define(repr=False)
class EcoliBREL606Gene(BacterialGene):
    """A REL606 gene: the one chromosome is chromosome 1."""

    REPLICONS: ClassVar[MappingProxyType[str, int]] = MappingProxyType(
        {REL606_ASSEMBLY.replicon: 1}
    )


@define(eq=False)
class EcoliBREL606Genome(BacterialGenome[EcoliBREL606Gene]):
    """E. coli B REL606 (GCA_000017985.1): ``ECB_`` tags, GO from the RefSeq GFF.

    The ``data.db`` cache, GenBank ingest, GO route and name resolution (locus tag,
    ``old_locus_tag``, RefSeq locus tag, gene symbol, ``gene_synonym``, retired) are
    :class:`~torchcell.sequence.genome.bacterial.BacterialGenome`'s.
    """

    ASSEMBLY: ClassVar[BacterialAssembly] = REL606_ASSEMBLY
    ASSEMBLY_SET: ClassVar[str] = ECOLI_B_REL606
    GENOME_VERSION: ClassVar[str] = "ASM1798v1"
    ANNOTATION_NAME: ClassVar[str] = "GenBank GCA_000017985.1"
    ANNOTATION_RELEASE: ClassVar[str] = "GCA_000017985.1_ASM1798v1"

    genome_root: str = field(
        init=True, repr=False, default=REL606_ASSEMBLY.default_genome_root
    )
    overwrite: bool = field(init=True, repr=True, default=False)

    @classmethod
    def gene_class(cls) -> type[EcoliBREL606Gene]:
        """REL606 genes are :class:`EcoliBREL606Gene`."""
        return EcoliBREL606Gene

    @property
    def strain(self) -> str:
        """The strain this genome reads, ``"B REL606"``."""
        return self.ASSEMBLY.strain
