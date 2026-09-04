# torchcell/metabolism/betaxanthin.py
# [[torchcell.metabolism.betaxanthin]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/metabolism/betaxanthin.py
# Test file: tests/torchcell/metabolism/test_betaxanthin.py

"""The betaxanthin cassette Cachera transferred into the yeast knockout collection.

Cachera 2023 (``cacheraCRISPAHighthroughputMethod2023``) names the four cassette genes and
defers every mechanistic detail to its references, so no reaction here is sourced from it.
The chemistry comes from DeLoache 2015 (``deloacheEnzymecoupledBiosensorEnables2015``),
whose Figure 1 and Supplementary Figure 1 give the pathway and which of its steps are
enzymatic.

Two facts drive the design and are easy to get wrong:

**Betaxanthin is a family, not a molecule.** Supplementary Figure 1: "DOD will generate
betalamic acid, a reactive aldehyde that undergoes spontaneous condensation with amino
acids and other free amines to form betaxanthins, which are yellow and fluorescent." So the
measured yellowness is a SUM over condensation partners, and the partner pool is the cell's
free amino acids. That is what couples this readout to the Mulleder panel: the nineteen
measured amino-acid pools are literally the substrates of the last step.

**The growth medium is itself a partner.** DeLoache, Methods: "We discovered that the
standard medium component para-aminobenzoic acid (PABA) was capable of spontaneous
condensation with betalamic acid to produce PABA-betaxanthin." PABA is 4-aminobenzoate, a
component of every synthetic recipe in :mod:`torchcell.metabolism.media`, so the medium
enters this phenotype as a substrate and not only as a nutrient.

The condensation stoichiometry (aldehyde plus amine, losing water) is verified against
DeLoache's own reported masses rather than asserted: computed neutral masses for the
tyrosine, proline, valine, leucine, phenylalanine and tryptophan betaxanthins and for
betanidin agree with their reported protonated masses to within 1.3 millidaltons.
"""

from __future__ import annotations

import logging

import cobra

from torchcell.metabolism.pathway import (
    EvidenceTier,
    HeterologousPathway,
    MetaboliteSpec,
    ReactionSpec,
)

log = logging.getLogger(__name__)

#: yeast-GEM 9.0.2 cytosolic species the cassette reads or writes.
TYROSINE = "s_1051"
OXYGEN = "s_1275"
NADPH = "s_1212"
NADP = "s_1207"
WATER = "s_0803"
PROTON = "s_0794"

#: Cytosolic condensation partners, by yeast-GEM metabolite id. The twenty proteinogenic
#: amino acids plus 4-aminobenzoate, which DeLoache identified as a medium-derived partner.
DEFAULT_PARTNERS: dict[str, str] = {
    "s_0955": "alanine",
    "s_0965": "arginine",
    "s_0969": "asparagine",
    "s_0973": "aspartate",
    "s_0981": "cysteine",
    "s_0991": "glutamate",
    "s_0999": "glutamine",
    "s_1006": "histidine",
    "s_1016": "isoleucine",
    "s_1021": "leucine",
    "s_1025": "lysine",
    "s_1029": "methionine",
    "s_1032": "phenylalanine",
    "s_1035": "proline",
    "s_1039": "serine",
    "s_1045": "threonine",
    "s_1048": "tryptophan",
    "s_1051": "tyrosine",
    "s_1056": "valine",
    "s_0271": "4_aminobenzoate",
}

_TH_QUOTE = (
    "a cytochrome P450 from the sugar beet Beta vulgaris and represents what is to our "
    "knowledge the first known example of a P450 capable of L-tyrosine hydroxylation"
)
_DO_QUOTE = (
    "the wild-type version of this enzyme catalyzes an additional unwanted oxidation of "
    "L-DOPA into L-dopaquinone"
)
_DOD_QUOTE = (
    "DOPA dioxygenase (DOD) is a plant enzyme found in members of the order "
    "Caryophyllales that converts L-DOPA into a yellow, highly fluorescent family of "
    "pigments called betaxanthins"
)
_CONDENSE_QUOTE = (
    "DOD will generate betalamic acid, a reactive aldehyde that undergoes spontaneous "
    "condensation with amino acids and other free amines to form betaxanthins, which are "
    "yellow and fluorescent"
)
_CYCLO_QUOTE = (
    "After a spontaneous conversion to cyclo-DOPA, L-dopaquinone can undergo condensation "
    "with betalamic acid to form betanidin, a violet pigment"
)
_COFACTOR_NOTE = (
    "Cofactor balance follows from the enzyme class, not from DeLoache, who states the "
    "conversion without a balanced equation. P450 monooxygenase and extradiol dioxygenase "
    "conventions respectively."
)


def build_betaxanthin_pathway(
    model: cobra.Model,
    *,
    partners: dict[str, str] | None = None,
    include_oxidase_branch: bool = True,
) -> HeterologousPathway:
    """Build the cassette as a pathway object against one genome-scale model.

    Args:
        model: The base model, read to check which partners it actually carries.
        partners: Metabolite id to short name for the condensation partners. Defaults to
            :data:`DEFAULT_PARTNERS`, filtered to those present in ``model``.
        include_oxidase_branch: Include the DOPA oxidase side activity and the betanidin
            branch. Cachera's cassette carries wild-type CYP76AD1, verified by translating
            the ``pBTX002`` open reading frame, which is tryptophan at 13 and phenylalanine
            at 309, so neither the W13L nor the F309L mutation that suppresses this
            activity is present. The branch is therefore chemically live in that screen,
            though its magnitude there is unmeasured, since DeLoache quantified it only
            under ascorbic acid, which stabilizes the otherwise self-polymerizing betanidin.

    Returns:
        The pathway, ready for :func:`torchcell.metabolism.pathway.apply_pathway`.
    """
    partners = partners if partners is not None else DEFAULT_PARTNERS
    present = {k: v for k, v in partners.items() if k in model.metabolites}
    missing = sorted(set(partners) - set(present))
    if missing:
        log.warning(
            "betaxanthin: %d partner(s) absent from model: %s", len(missing), missing
        )

    mets: list[MetaboliteSpec] = [
        MetaboliteSpec(
            id="s_ldopa_c",
            name="L-DOPA",
            formula="C9H11NO4",
            charge=0,
            metanetx_id="MNXM279",
            chebi_id="CHEBI:57504",
            evidence=EvidenceTier.SOURCED,
            quote=_TH_QUOTE,
        ),
        MetaboliteSpec(
            id="s_betalamic_c",
            name="betalamic acid",
            formula="C9H9NO5",
            charge=0,
            metanetx_id="MNXM732452",
            chebi_id="CHEBI:27483",
            evidence=EvidenceTier.SOURCED,
            quote=_DOD_QUOTE,
            note="Neutral mass 211.05 reproduces DeLoache's reported m/z 212.055 [M+H]+.",
        ),
    ]
    rxns: list[ReactionSpec] = [
        ReactionSpec(
            id="r_CYP76AD1_th",
            name="tyrosine 3-hydroxylase (CYP76AD1)",
            stoichiometry={
                TYROSINE: -1,
                OXYGEN: -1,
                NADPH: -1,
                PROTON: -1,
                "s_ldopa_c": 1,
                WATER: 1,
                NADP: 1,
            },
            gene_reaction_rule="CYP76AD1",
            evidence=EvidenceTier.CONVENTION,
            quote=_TH_QUOTE,
            note=_COFACTOR_NOTE,
        ),
        ReactionSpec(
            id="r_DOD",
            name="L-DOPA 4,5-dioxygenase (DOD)",
            stoichiometry={"s_ldopa_c": -1, OXYGEN: -1, "s_betalamic_c": 1, WATER: 1},
            gene_reaction_rule="DOD",
            evidence=EvidenceTier.CONVENTION,
            quote=_DOD_QUOTE,
            note=_COFACTOR_NOTE,
        ),
    ]

    if include_oxidase_branch:
        mets += [
            MetaboliteSpec(
                id="s_dopaquinone_c",
                name="L-dopaquinone",
                formula="C9H9NO4",
                charge=0,
                metanetx_id="MNXM729106",
                chebi_id="CHEBI:57924",
                evidence=EvidenceTier.SOURCED,
                quote=_DO_QUOTE,
                note="Neutral mass 195.05 reproduces DeLoache's reported m/z 196.06.",
            ),
            MetaboliteSpec(
                id="s_cyclodopa_c",
                name="cyclo-DOPA (leucodopachrome)",
                formula="C9H9NO4",
                charge=0,
                metanetx_id="MNXM2390",
                chebi_id="CHEBI:231766",
                evidence=EvidenceTier.SOURCED,
                quote=_CYCLO_QUOTE,
                note="Intramolecular cyclization of dopaquinone, so the formula is identical.",
            ),
            MetaboliteSpec(
                id="s_betanidin_c",
                name="betanidin",
                metanetx_id="MNXM736187",
                chebi_id="CHEBI:3079",
                evidence=EvidenceTier.DERIVED,
                quote=_CYCLO_QUOTE,
            ),
        ]
        rxns += [
            ReactionSpec(
                id="r_CYP76AD1_do",
                name="DOPA oxidase side activity (CYP76AD1)",
                stoichiometry={
                    "s_ldopa_c": -1,
                    OXYGEN: -0.5,
                    "s_dopaquinone_c": 1,
                    WATER: 1,
                },
                gene_reaction_rule="CYP76AD1",
                evidence=EvidenceTier.CONVENTION,
                quote=_DO_QUOTE,
                note=_COFACTOR_NOTE,
            ),
            ReactionSpec(
                id="r_cyclodopa_spont",
                name="dopaquinone cyclization (spontaneous)",
                stoichiometry={"s_dopaquinone_c": -1, "s_cyclodopa_c": 1},
                spontaneous=True,
                evidence=EvidenceTier.SOURCED,
                quote=_CYCLO_QUOTE,
            ),
            ReactionSpec(
                id="r_betanidin_spont",
                name="betanidin condensation (spontaneous)",
                stoichiometry={
                    "s_betalamic_c": -1,
                    "s_cyclodopa_c": -1,
                    "s_betanidin_c": 1,
                    WATER: 1,
                },
                spontaneous=True,
                evidence=EvidenceTier.SOURCED,
                quote=_CYCLO_QUOTE,
            ),
            ReactionSpec(
                id="DM_betanidin",
                name="betanidin demand",
                stoichiometry={"s_betanidin_c": -1},
                evidence=EvidenceTier.DERIVED,
                note="Products escape into the medium; DeLoache measures them in supernatant.",
            ),
        ]

    # One condensation per partner. The product formula is DERIVED from the model's own
    # species, because yeast-GEM carries the dominant charge state at pH 7 and a formula
    # copied from a neutral mass would leave the reaction unbalanced in that convention.
    for met_id, short in sorted(present.items(), key=lambda kv: kv[1]):
        product = f"s_btx_{short}_c"
        mets.append(
            MetaboliteSpec(
                id=product,
                name=f"{short}-betaxanthin",
                evidence=EvidenceTier.DERIVED,
                quote=_CONDENSE_QUOTE,
            )
        )
        rxns.append(
            ReactionSpec(
                id=f"r_btx_{short}_spont",
                name=f"{short}-betaxanthin condensation (spontaneous)",
                stoichiometry={"s_betalamic_c": -1, met_id: -1, product: 1, WATER: 1},
                spontaneous=True,
                evidence=EvidenceTier.SOURCED,
                quote=_CONDENSE_QUOTE,
            )
        )
        rxns.append(
            ReactionSpec(
                id=f"DM_btx_{short}",
                name=f"{short}-betaxanthin demand",
                stoichiometry={product: -1},
                evidence=EvidenceTier.DERIVED,
            )
        )

    return HeterologousPathway(
        name="betaxanthin_cassette",
        base_strain="BY4741",
        description=(
            "BvCYP76AD1 + MjDOD betaxanthin cassette integrated at XII-5, as transferred "
            "into the yeast knockout collection by CRI-SPA. The feedback-resistant "
            "ARO4-K229L and ARO7-G141S alleles that accompany it are edits to genes yeast "
            "already has, so they are genotype perturbations rather than pathway additions "
            "and are not represented here."
        ),
        metabolites=mets,
        reactions=rxns,
        constitutive_genes=["CYP76AD1", "DOD"],
        source_keys=[
            "deloacheEnzymecoupledBiosensorEnables2015",
            "cacheraCRISPAHighthroughputMethod2023",
        ],
    )


def betaxanthin_demand_ids(pathway: HeterologousPathway) -> list[str]:
    """Demand reactions whose flux sums to the measured yellow pigment.

    Cachera's colony score is total yellowness and cannot separate the species, so the
    quantity a head should read is this sum, not any single reaction.
    """
    return sorted(r.id for r in pathway.reactions if r.id.startswith("DM_btx_"))
