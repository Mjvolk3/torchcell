# torchcell/sequence/genome/ecoli/w3110
# [[torchcell.sequence.genome.ecoli.w3110]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/sequence/genome/ecoli/w3110
# Test file: tests/torchcell/sequence/genome/ecoli/test_w3110.py

r"""E. coli K-12 W3110, read RefSeq first from the genomes tier (#910).

W3110 is a third K-12 strain beside MG1655 and BW25113, with an assembly set of its own
(``ecoli_K12_W3110_ASM1024v1``): GCA_000010245.1 / GCF_000010245.2 (ASM1024v1), one
circular replicon AP009048.1 / NC_007779.1 of 4,646,332 bp.

**Why RefSeq first.** The GenBank annotation (the 2006 DDBJ/NIG submission) carries 4,444
gene features and NOT ONE ``locus_tag``; it keys genes by symbol and carries the
``ECK:JW:b`` crosswalk as a ``/note`` on its CDS features, so the GenBank-first ingest of
:class:`~torchcell.sequence.genome.bacterial.BacterialGenome` refuses it by name. The
RefSeq annotation (NCBI PGAP) parses: 4,531 gene and pseudogene features, every one tagged
``Y75_RS\d{5}``, with the earlier RefSeq tags ``Y75_p\d{4}`` as ``old_locus_tag`` on 4,254
of them. The owner chose this route on 2026-10-12 (#910) for the Teteneva 2024 row, whose
reads the authors aligned to "NCBI reference sequence: NC_007779.1".

So the assembly is a :class:`~torchcell.sequence.genome.bacterial.RefSeqPrimaryAssembly`:
the GCF flat file is the locus universe, the GCF GFF3 is ``data.db``, GO is the GCF GFF's
inline ``Ontology_term`` rows keyed by their own locus tag, and the proteins are the GCF
flat file's ``/translation`` values. The DNA is the set's one FASTA, the GCA record
AP009048.1, which construction requires to be byte-identical to NC_007779.1 before keying
it by the RefSeq accession (the GCF assembly report states "RefSeq assembly and GenBank
assemblies identical: yes"). The name resolver reads the RefSeq ``/gene`` symbols and the
``old_locus_tag`` layer; the RefSeq annotation carries ``/gene_synonym`` on 84 features.
"""

from types import MappingProxyType
from typing import ClassVar, Literal

from attrs import define, field

from torchcell.sequence.genome.bacterial import (
    BacterialGene,
    BacterialGenome,
    GoSourceSpec,
    RefSeqPrimaryAssembly,
)
from torchcell.sequence.genome.registry import ECOLI_K12_W3110

#: W3110 RefSeq locus tags (NC_007779.1), measured on all 4,531 gene and pseudogene
#: features of ``GCF_000010245.2_ASM1024v1_genomic.gbff.gz``.
W3110_LOCUS_TAG_PATTERN = r"Y75_RS\d{5}"

W3110_ASSEMBLY = RefSeqPrimaryAssembly(
    organism="Escherichia coli",
    strain="K-12 W3110",
    assembly_set=ECOLI_K12_W3110,
    genbank_assembly="GCA_000010245.1_ASM1024v1",
    refseq_assembly="GCF_000010245.2_ASM1024v1",
    replicon="NC_007779.1",
    sequence_replicon="AP009048.1",
    locus_tag_pattern=W3110_LOCUS_TAG_PATTERN,
    go_source=GoSourceSpec(
        route="refseq_gff_ontology_term_primary",
        assembly_set=ECOLI_K12_W3110,
        member="GCF_000010245.2_ASM1024v1_genomic.gff.gz",
        identifier_pattern=None,
    ),
    default_genome_root="data/ecoli/w3110/genome",
)

EcoliW3110StrainName = Literal["W3110"]


@define(repr=False)
class EcoliK12W3110Gene(BacterialGene):
    """A W3110 gene: the one chromosome, NC_007779.1, is chromosome 1."""

    REPLICONS: ClassVar[MappingProxyType[str, int]] = MappingProxyType(
        {W3110_ASSEMBLY.replicon: 1}
    )


@define(eq=False)
class EcoliK12W3110Genome(BacterialGenome[EcoliK12W3110Gene]):
    """E. coli K-12 W3110 (GCF_000010245.2): ``Y75_RS`` tags, RefSeq first.

    The ``data.db`` cache, the RefSeq-primary ingest, the GO route and the name
    resolution (locus tag, ``old_locus_tag``, gene symbol, ``gene_synonym``, retired) are
    :class:`~torchcell.sequence.genome.bacterial.BacterialGenome`'s.
    """

    ASSEMBLY: ClassVar[RefSeqPrimaryAssembly] = W3110_ASSEMBLY
    ASSEMBLY_SET: ClassVar[str] = ECOLI_K12_W3110
    GENOME_VERSION: ClassVar[str] = "ASM1024v1"
    ANNOTATION_NAME: ClassVar[str] = "RefSeq GCF_000010245.2"
    ANNOTATION_RELEASE: ClassVar[str] = "GCF_000010245.2_ASM1024v1"

    genome_root: str = field(
        init=True, repr=False, default=W3110_ASSEMBLY.default_genome_root
    )
    overwrite: bool = field(init=True, repr=True, default=False)

    @classmethod
    def gene_class(cls) -> type[EcoliK12W3110Gene]:
        """W3110 genes are :class:`EcoliK12W3110Gene`."""
        return EcoliK12W3110Gene

    @property
    def strain(self) -> str:
        """The strain this genome reads, ``"K-12 W3110"``."""
        return self.ASSEMBLY.strain
