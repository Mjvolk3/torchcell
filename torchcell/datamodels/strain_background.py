# torchcell/datamodels/strain_background.py
# [[torchcell.datamodels.strain_background]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/strain_background
# Test file: tests/torchcell/datamodels/test_strain_background.py
"""Shared constructors for typed strain backgrounds (issue #507).

The record classes (``StrainBackground``, ``BackgroundAllele``, ...) live in
``schema.py`` so their contracts are fingerprinted with the rest of the schema
surface. This module holds the shared VALUES the four chemogenomic loaders
(Vanacloig, Wildenhain, Hillenmeyer, Hoepfner) build them from, so two loaders that
state the same BY strain build the same alleles:

- ``STANDARD_ALLELES``: the BY / SGA allele designations mapped to their R64 locus
  and edit kind. The locus (systematic + current standard name) is read from the
  R64-4-1 GFF (``R64_GFF``). The edit kind and function of each designation are the
  literature-standard reading (Brachmann 1998 for the BY ``delta0`` / ``his3-delta1``
  alleles; Tong 2006 / Piotrowski 2017 for the SGA markers) and are NOT verified
  against a mirrored artifact for the BY alleles; that is why ``standard_allele``
  demands either a quote or a pending-review gap.
- ``STANDARD_BY_GENOTYPES``: BY4741 / BY4742 / BY4743 as typed allele tables.
  Hypothesis-level until Brachmann 1998 is mirrored (only BY4741's string is quoted in
  a mirrored source: Wildenhain 2016 Sci Data); ``standard_background`` therefore
  takes one ``provenance`` list for every element or gaps every element.
- ``KANMX4_CASSETTE``: the YKO cassette, sourced to Giaever 2014 (mirrored).
- ``BRACHMANN_1998`` / ``GIAEVER_2002``: ``resolve_with`` targets for the gaps.
- ``baid_background()`` / ``BAID_STRAIN``: the CRISPR-AID host bAID, shared by every
  dataset screened in it (Lian 2019 and the in-house Bioscreen dataset), so the two
  join on one typed background rather than two spellings of the same strain.

Design + worked examples per dataset: ``[[torchcell.datamodels.strain-background]]``.
"""

from __future__ import annotations

from typing import Literal

from torchcell.datamodels.pydant import ModelStrict
from torchcell.datamodels.schema import (
    AlleleEdit,
    BackgroundAllele,
    IntegratedCassette,
    MatingType,
    StrainBackground,
    Zygosity,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

__all__ = [
    "AlleleSpec",
    "BAID_CONSTRUCTION",
    "BAID_GENOTYPE",
    "BAID_PARENT_GENOTYPE",
    "BAID_STRAIN",
    "BRACHMANN_1998",
    "GIAEVER_2002",
    "KANMX4_CASSETTE",
    "LIAN_2019_METHODS",
    "LIAN_2019_SI1",
    "R64_GFF",
    "STANDARD_ALLELES",
    "STANDARD_BY_GENOTYPES",
    "ByGenotype",
    "baid_background",
    "pending_source_review",
    "standard_allele",
    "standard_background",
]

R64_GFF = Provenance(
    source_uri="data/sgd/genome/S288C_reference_genome_R64-4-1_20230830/"
    "saccharomyces_cerevisiae_R64-4-1_20230830.gff",
    sha256="64f61e3153083a8ef6d853721c9e83e4469cdc120883ec281e51a0df4ba390fa",
    method="gene rows: ID (systematic name) and gene= (standard name)",
)
"""Where each ``STANDARD_ALLELES`` locus was read (path under ``$DATA_ROOT``)."""

BRACHMANN_1998 = Provenance(
    source_uri="https://doi.org/10.1002/(SICI)1097-0061(19980130)14:2<115::AID-YEA204>3.0.CO;2-2",
    method="not mirrored; BY4741/BY4742/BY4743 genotypes and the delta0 designer "
    "deletions (Brachmann et al. 1998, Yeast 14:115-132)",
)
"""``resolve_with`` target for a pending BY genotype element. DOI as recorded from
the citation, not yet checked against a mirrored copy."""

GIAEVER_2002 = Provenance(
    source_uri="https://doi.org/10.1038/nature00935",
    method="not mirrored; YKO collection construction, incl. homozygous diploids by "
    "mating MATa x MATalpha deletants (Giaever et al. 2002, Nature 418:387)",
)
"""``resolve_with`` target for a pending YKO-collection construction element."""

KANMX4_CASSETTE = SourcedValue(
    value="kanMX4",
    provenance=Provenance(
        source_uri="paper.md",
        citation_key="giaeverYeastDeletionCollection2014",
        sha256="4a177a8658be57938eb0e45e736762281793ac7799d358e3a3349af61acf1719",
    ),
    quote="74-bp UPTAG and 74-bp DNTAG primers amplify the KanMX gene from "
    "pFA6-kanMX4 DNA",
    note="the YKO (SGDP) deletion cassette. The same review names Euroscarf as a "
    "source of the YKO collection ('reliable sources of the collection are "
    "Euroscarf'); that the OpenBiosystems YSC1055/YSC1056 pools are YKO strains is "
    "not stated in a mirrored source",
)


class AlleleSpec(ModelStrict):
    """The locus and edit kind a standard allele designation denotes."""

    systematic_gene_name: str
    gene_name: str
    edit: AlleleEdit
    functional: bool
    cassette: str | None = None


STANDARD_ALLELES: dict[str, AlleleSpec] = {
    # BY designer alleles (Brachmann 1998, not mirrored). his3-delta1 is an internal
    # deletion; the delta0 alleles remove the ORF with no marker left.
    "his3Δ1": AlleleSpec(
        systematic_gene_name="YOR202W",
        gene_name="HIS3",
        edit=AlleleEdit.partial_deletion,
        functional=False,
    ),
    "leu2Δ0": AlleleSpec(
        systematic_gene_name="YCL018W",
        gene_name="LEU2",
        edit=AlleleEdit.full_deletion,
        functional=False,
    ),
    "ura3Δ0": AlleleSpec(
        systematic_gene_name="YEL021W",
        gene_name="URA3",
        edit=AlleleEdit.full_deletion,
        functional=False,
    ),
    # met15 is the historical name; R64-4-1 calls YLR303W MET17.
    "met15Δ0": AlleleSpec(
        systematic_gene_name="YLR303W",
        gene_name="MET17",
        edit=AlleleEdit.full_deletion,
        functional=False,
    ),
    "lys2Δ0": AlleleSpec(
        systematic_gene_name="YBR115C",
        gene_name="LYS2",
        edit=AlleleEdit.full_deletion,
        functional=False,
    ),
    # SGA reporters (Piotrowski 2017 / Ohnuki 2022, mirrored).
    "can1Δ::STE2pr-Sp_his5": AlleleSpec(
        systematic_gene_name="YEL063C",
        gene_name="CAN1",
        edit=AlleleEdit.cassette_replacement,
        functional=False,
        cassette="STE2pr-Sp_his5",
    ),
    # Hypothesis (untested): lyp1-delta is a marker-free full ORF deletion; no
    # mirrored source describes how it was made.
    "lyp1Δ": AlleleSpec(
        systematic_gene_name="YNL268W",
        gene_name="LYP1",
        edit=AlleleEdit.full_deletion,
        functional=False,
    ),
    # Vanacloig / Piotrowski 3-delta drug-sensitizing deletions.
    "pdr1Δ::natMX": AlleleSpec(
        systematic_gene_name="YGL013C",
        gene_name="PDR1",
        edit=AlleleEdit.cassette_replacement,
        functional=False,
        cassette="natMX",
    ),
    "pdr3Δ::KlURA3": AlleleSpec(
        systematic_gene_name="YBL005W",
        gene_name="PDR3",
        edit=AlleleEdit.cassette_replacement,
        functional=False,
        cassette="KlURA3",
    ),
    "snq2Δ::KlLEU2": AlleleSpec(
        systematic_gene_name="YDR011W",
        gene_name="SNQ2",
        edit=AlleleEdit.cassette_replacement,
        functional=False,
        cassette="KlLEU2",
    ),
}


class ByGenotype(ModelStrict):
    """A standard BY strain as mating type, ploidy and allele -> zygosity."""

    mating_type: MatingType
    ploidy: Literal["haploid", "diploid"]
    parents: list[str]
    alleles: dict[str, Zygosity]


STANDARD_BY_GENOTYPES: dict[str, ByGenotype] = {
    "BY4741": ByGenotype(
        mating_type=MatingType.a,
        ploidy="haploid",
        parents=["S288C"],
        alleles={
            "his3Δ1": Zygosity.haploid,
            "leu2Δ0": Zygosity.haploid,
            "met15Δ0": Zygosity.haploid,
            "ura3Δ0": Zygosity.haploid,
        },
    ),
    "BY4742": ByGenotype(
        mating_type=MatingType.alpha,
        ploidy="haploid",
        parents=["S288C"],
        alleles={
            "his3Δ1": Zygosity.haploid,
            "leu2Δ0": Zygosity.haploid,
            "lys2Δ0": Zygosity.haploid,
            "ura3Δ0": Zygosity.haploid,
        },
    ),
    # BY4741 x BY4742: his3Δ1/his3Δ1 leu2Δ0/leu2Δ0 LYS2/lys2Δ0 met15Δ0/MET15
    # ura3Δ0/ura3Δ0.
    "BY4743": ByGenotype(
        mating_type=MatingType.a_alpha,
        ploidy="diploid",
        parents=["BY4741", "BY4742"],
        alleles={
            "his3Δ1": Zygosity.homozygous,
            "leu2Δ0": Zygosity.homozygous,
            "lys2Δ0": Zygosity.heterozygous,
            "met15Δ0": Zygosity.heterozygous,
            "ura3Δ0": Zygosity.homozygous,
        },
    ),
}
"""Literature-standard BY genotypes (Brachmann 1998, NOT mirrored). Only BY4741's
string is quoted in a mirrored source (Wildenhain 2016 Sci Data). The parents of
BY4741/BY4742 are written as their S288C lineage, not the immediate progenitor."""


def pending_source_review(
    field: str, resolve_with: Provenance, note: str | None = None
) -> ProvenanceGap:
    """A ``deferred_pending_source_review`` gap on ``field`` naming its resolver."""
    return ProvenanceGap(
        field=field,
        reason=ProvenanceGapReason.deferred_pending_source_review,
        resolve_with=resolve_with,
        note=note,
    )


def _sourcing(
    provenance: list[SourcedValue] | None, resolve_with: Provenance | None
) -> None:
    """Exactly one of a quote list or a resolver: the element is sourced or gapped."""
    if (provenance is None) == (resolve_with is None):
        raise ValueError(
            "pass exactly one of provenance (a quote list) or resolve_with (the "
            "unmirrored source a pending-review gap names)"
        )
    if provenance is not None and not provenance:
        raise ValueError("provenance must be a non-empty quote list")


def standard_allele(
    allele_name: str,
    zygosity: Zygosity,
    *,
    provenance: list[SourcedValue] | None = None,
    resolve_with: Provenance | None = None,
    note: str | None = None,
) -> BackgroundAllele:
    """A ``BackgroundAllele`` for a ``STANDARD_ALLELES`` designation.

    Pass ``provenance`` when a mirrored source states the allele (a quote); pass
    ``resolve_with`` otherwise, which asserts the allele with a
    ``deferred_pending_source_review`` gap on ``provenance`` naming that source.
    """
    _sourcing(provenance, resolve_with)
    spec = STANDARD_ALLELES[allele_name]
    gaps = (
        []
        if resolve_with is None
        else [pending_source_review("provenance", resolve_with, note)]
    )
    return BackgroundAllele(
        systematic_gene_name=spec.systematic_gene_name,
        gene_name=spec.gene_name,
        allele_name=allele_name,
        edit=spec.edit,
        functional=spec.functional,
        zygosity=zygosity,
        cassette=spec.cassette,
        provenance=provenance,
        provenance_gaps=gaps,
    )


def standard_background(
    name: str,
    *,
    provenance: list[SourcedValue] | None = None,
    resolve_with: Provenance | None = None,
    note: str | None = None,
    extra_alleles: list[BackgroundAllele] | None = None,
    construction: str | None = None,
) -> StrainBackground:
    """A ``StrainBackground`` for BY4741 / BY4742 / BY4743.

    ``provenance`` (one quote list that states the whole genotype, e.g. Wildenhain
    2016's "isogenic to BY4741, which has the genotype MATa his3Δ1 leu2Δ0 met15Δ0
    ura3Δ0") sources the background and every allele; ``resolve_with`` instead
    asserts each element with a pending-review gap naming that source (Brachmann
    1998 for BY4743 until it is mirrored). ``extra_alleles`` are appended as given,
    each carrying its own sourcing.
    """
    _sourcing(provenance, resolve_with)
    genotype = STANDARD_BY_GENOTYPES[name]
    alleles = [
        standard_allele(
            allele_name,
            zygosity,
            provenance=provenance,
            resolve_with=resolve_with,
            note=note,
        )
        for allele_name, zygosity in genotype.alleles.items()
    ]
    gaps = (
        []
        if resolve_with is None
        else [pending_source_review("provenance", resolve_with, note)]
    )
    return StrainBackground(
        name=name,
        parents=list(genotype.parents),
        construction=construction,
        mating_type=genotype.mating_type,
        ploidy=genotype.ploidy,
        alleles=[*alleles, *(extra_alleles or [])],
        provenance=provenance,
        provenance_gaps=gaps,
    )


# --------------------------------------------------------------------------- #
# bAID: the CRISPR-AID host, shared by every dataset screened in it
# --------------------------------------------------------------------------- #
#: The CRISPR-AID host strain, as Lian 2019 names it. The in-house Bioscreen dataset was
#: run on the same strain under the thesis name ``BY4742-iAID6``, so both datasets join
#: on ``baid_background()`` rather than on two spellings of one strain.
BAID_STRAIN = "bAID"

LIAN_2019_SI1 = Provenance(
    source_uri="si/si1.md",
    citation_key="lianMultifunctionalGenomewideCRISPR2019",
    sha256="b2bcfe2e672674438216472e3e06903c93d4ee54cd8b6fd9b5f964ad2a3d32db",
    method="MinerU OCR of the publisher Supplementary Information PDF "
    "(torchcell-library mirror)",
    page="Supplementary Table 11, 'Strains constructed in this study' (si1.md line 104)",
)
"""Lian 2019's Supplementary Information 1 OCR, which carries the strain table."""

LIAN_2019_METHODS = Provenance(
    source_uri="paper.md",
    citation_key="lianMultifunctionalGenomewideCRISPR2019",
    sha256="63fe2b7101fc48feb297f9e34b83d108b74f03f28bbc280e08c7219bc975086c",
    method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
    page="Methods, 'Plasmid and strain construction'",
)
"""Lian 2019's paper OCR, which carries bAID's construction sentence."""

BAID_PARENT_GENOTYPE = SourcedValue(
    value="MATα his3∆1 leu2∆0 lys2∆0 ura3∆0",
    quote="<td rowspan=1 colspan=1>BY4742</td><td rowspan=1 colspan=1>MATα his3∆1 "
    "leu2∆0 lys2∆0 ura3∆0</td>",
    provenance=LIAN_2019_SI1,
    note="bAID's parent genotype as the strain table states it; the stored alleles are "
    "the STANDARD_ALLELES spellings his3Δ1, leu2Δ0, lys2Δ0, ura3Δ0 (the OCR writes the "
    "delta as the mathematical operator ∆)",
)
BAID_GENOTYPE = SourcedValue(
    value="BY4742-Delta::KanMX-[dLbCpf1-VP]-[Csy4]-[dSpCas9-RD1152]-[SaCas9]",
    quote="<td rowspan=1 colspan=1>bAID</td><td rowspan=1 colspan=1>BY4742-Delta::KanMX-"
    "[dLbCpf1-VP]-[Csy4]-[dSpCas9-RD1152]-[SaCas9]</td>",
    provenance=LIAN_2019_SI1,
    note="the integration SITE of the CRISPR-AID cassette: the strain table writes it as "
    "the Delta site (the Ty1 delta repeat family), carrying KanMX plus the four "
    "orthogonal effector cassettes",
)
BAID_CONSTRUCTION = SourcedValue(
    value=BAID_STRAIN,
    quote="The CRISPR-AID strain (bAID) was constructed by integrating PmeI-digested "
    "$\\mathrm { \\ p A I D } 6 ^ { 8 }$ into the genome of BY4742 and selection for "
    "G418 resistance.",
    provenance=LIAN_2019_METHODS,
    note="how the host was made: pAID6 integrated into BY4742, selected on G418. The "
    "in-house Bioscreen dataset's thesis strain BY4742-iAID6 is this same construction",
)

_BAID_ALLELE_NOTE = (
    "Supplementary Table 11 states BY4742's genotype string ('MATα his3∆1 leu2∆0 "
    "lys2∆0 ura3∆0', BAID_PARENT_GENOTYPE), so WHICH alleles the strain carries is "
    "sourced; how each was constructed (the delta0 designer deletions vs the "
    "his3-delta1 internal deletion, which is what STANDARD_ALLELES encodes as an "
    "AlleleEdit) is stated only by Brachmann 1998, which is not mirrored"
)


def baid_background() -> StrainBackground:
    """The CRISPR-AID host bAID as a typed ``StrainBackground``.

    BY4742's four auxotrophies as ``BackgroundAllele``s plus the one cassette the host
    carries integrated: ``Delta::KanMX-[dLbCpf1-VP]-[Csy4]-[dSpCas9-RD1152]-[SaCas9]``,
    at the Delta site (the Ty1 delta repeat family). ``locus_systematic_gene_name``
    stays None, because a delta repeat family is not an R64 ORF.

    One helper, so every dataset screened in this strain joins on one background: Lian
    2019's genome-wide MAGIC screen, and the in-house Bioscreen dataset, whose thesis
    strain **BY4742-iAID6** is this strain (pAID6 integrated into BY4742, selected on
    G418 resistance).

    The auxotrophies are asserted with a ``deferred_pending_source_review`` gap naming
    Brachmann 1998 even though the strain table states the genotype string: the SI says
    WHICH alleles BY4742 carries, not how any of them was made, and ``STANDARD_ALLELES``
    reads each designation as a specific edit kind (``his3Δ1`` partial, the three ``Δ0``
    alleles full) that only Brachmann 1998 states.
    """
    genotype = STANDARD_BY_GENOTYPES["BY4742"]
    alleles = [
        standard_allele(
            allele_name, zygosity, resolve_with=BRACHMANN_1998, note=_BAID_ALLELE_NOTE
        )
        for allele_name, zygosity in genotype.alleles.items()
    ]
    cassette = IntegratedCassette(
        name="Delta::KanMX-[dLbCpf1-VP]-[Csy4]-[dSpCas9-RD1152]-[SaCas9]",
        locus="Delta",
        elements=["KanMX", "dLbCpf1-VP", "Csy4", "dSpCas9-RD1152", "SaCas9"],
        marker="KanMX",
        zygosity=Zygosity.haploid,
        provenance=[BAID_GENOTYPE, BAID_CONSTRUCTION],
    )
    return StrainBackground(
        name=BAID_STRAIN,
        parents=["BY4742"],
        construction=BAID_CONSTRUCTION.quote,
        mating_type=genotype.mating_type,
        ploidy=genotype.ploidy,
        alleles=alleles,
        integrations=[cassette],
        provenance=[BAID_PARENT_GENOTYPE, BAID_GENOTYPE, BAID_CONSTRUCTION],
    )
