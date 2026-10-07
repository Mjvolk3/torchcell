# torchcell/sequence/genome/pputida/kt2440
# [[torchcell.sequence.genome.pputida.kt2440]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/sequence/genome/pputida/kt2440
# Test file: tests/torchcell/sequence/genome/pputida/test_kt2440.py

r"""P. putida KT2440, read GenBank first from the genomes tier.

GCA_000007565.2 (replicon AE015451.2, 6,181,873 bp). Its locus tags are ``PP_\d{4}``
for the 5,621 numbered loci and named tags for the RNA loci (``PP_16SA``, ``PP_23SB``,
``PP_5SC``, ``PP_t01`` tRNAs, ``PP_tm01`` tmRNA, ``PP_mr01`` misc RNAs, ``PP_r01``):
:data:`KT2440_LOCUS_TAG_PATTERN` covers all 5,786. The GenBank file carries no
``gene_synonym`` for this assembly, so the synonym layer of
:meth:`~torchcell.sequence.genome.bacterial.BacterialGenome.resolve_gene_name` is
empty; RefSeq's ``PP_RS`` tags resolve through the RefSeq GFF's ``old_locus_tag``.

GO comes from the EBI GOA proteome file ``109.P_putida_KT2440.goa`` (taxon 160488,
almost entirely IEA), read on column 11, where every row carries a ``PP_`` tag. EBI
keeps no dated archive of these files, so the deposited copy is the version.
"""

from types import MappingProxyType
from typing import ClassVar

from attrs import define, field

from torchcell.sequence.genome.bacterial import (
    BacterialAssembly,
    BacterialGene,
    BacterialGenome,
    GoSourceSpec,
)
from torchcell.sequence.genome.registry import PPUTIDA_KT2440

#: KT2440 GenBank locus tags: numbered loci and the named RNA loci.
KT2440_LOCUS_TAG_PATTERN = r"PP_(?:\d{4}|tm?\d{2}|mr\d{2}|r\d{2}|(?:5|16|23)S[A-Z])"

KT2440_ASSEMBLY = BacterialAssembly(
    organism="Pseudomonas putida",
    strain="KT2440",
    assembly_set=PPUTIDA_KT2440,
    genbank_assembly="GCA_000007565.2_ASM756v2",
    refseq_assembly="GCF_000007565.2_ASM756v2",
    replicon="AE015451.2",
    locus_tag_pattern=KT2440_LOCUS_TAG_PATTERN,
    go_source=GoSourceSpec(
        route="gaf_synonym_column",
        assembly_set=PPUTIDA_KT2440,
        member="109.P_putida_KT2440.goa",
        identifier_pattern=KT2440_LOCUS_TAG_PATTERN,
    ),
    default_genome_root="data/pputida/kt2440/genome",
)


@define(repr=False)
class PPutidaKT2440Gene(BacterialGene):
    """A KT2440 gene: the one chromosome is chromosome 1."""

    REPLICONS: ClassVar[MappingProxyType[str, int]] = MappingProxyType(
        {KT2440_ASSEMBLY.replicon: 1}
    )


@define(eq=False)
class PPutidaKT2440Genome(BacterialGenome[PPutidaKT2440Gene]):
    """P. putida KT2440 (GCA_000007565.2): ``PP_`` tags, GO from the GOA proteome file.

    The ``data.db`` cache, GenBank ingest, GO route and name resolution are
    :class:`~torchcell.sequence.genome.bacterial.BacterialGenome`'s.
    """

    ASSEMBLY: ClassVar[BacterialAssembly] = KT2440_ASSEMBLY
    ASSEMBLY_SET: ClassVar[str] = PPUTIDA_KT2440
    GENOME_VERSION: ClassVar[str] = "ASM756v2"
    ANNOTATION_NAME: ClassVar[str] = "GenBank GCA_000007565.2"
    ANNOTATION_RELEASE: ClassVar[str] = "GCA_000007565.2_ASM756v2"

    genome_root: str = field(
        init=True, repr=False, default=KT2440_ASSEMBLY.default_genome_root
    )
    overwrite: bool = field(init=True, repr=True, default=False)

    @classmethod
    def gene_class(cls) -> type[PPutidaKT2440Gene]:
        """KT2440 genes are :class:`PPutidaKT2440Gene`."""
        return PPutidaKT2440Gene
