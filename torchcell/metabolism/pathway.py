# torchcell/metabolism/pathway.py
# [[torchcell.metabolism.pathway]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/metabolism/pathway.py
# Test file: tests/torchcell/metabolism/test_pathway.py

"""Add a heterologous pathway to a genome-scale model, as a typed perturbation.

A strain is not only a set of gene deletions. Cachera's screened strains each carry a
five-gene cassette that yeast does not have, and the phenotype measured is the output of
that cassette. Representing the deletions but not the cassette means the model is asked to
predict a molecule its stoichiometry cannot make, which is what forced the earlier
betaxanthin head to read an aromatic-precursor proxy instead of the product.

This module makes the cassette a first-class object of the same kind as a deletion: an edit
to the metabolic model, declared once, applied deterministically, and carrying provenance
per value. The perturbation is to the MODEL; the genotype perturbation that accompanies it
lives in ``torchcell.datamodels.schema``.

Three properties are load-bearing:

* **Every coefficient is tiered.** A stoichiometry the source states is ``sourced``; a
  cofactor balance that follows from the enzyme class rather than from the paper is
  ``convention``; a formula computed from the model's own species is ``derived``. A
  pathway whose reactions are all ``convention`` is a guess with citations attached, and
  :meth:`HeterologousPathway.evidence_census` makes that visible rather than implicit.
* **Product formulas are derived from the model's own metabolites**, never typed in. The
  genome-scale model carries charged species at physiological pH, so a formula copied from
  a paper's neutral mass would leave the reaction unbalanced in the model's own convention.
  Deriving it guarantees mass and charge balance by construction.
* **Heterologous genes are constitutive.** They are not yeast open reading frames, so they
  have no gene token and cannot be deleted by a knockout library. They must be marked so a
  flux layer does not read their absence from the token universe as availability zero,
  which would switch the pathway off in every strain.
"""

from __future__ import annotations

import logging
from collections import Counter
from enum import StrEnum

import cobra
from pydantic import BaseModel, Field, model_validator

log = logging.getLogger(__name__)

#: Elemental composition of water, removed by every Schiff-base condensation here.
_WATER = {"H": 2, "O": 1}


class EvidenceTier(StrEnum):
    """Where a coefficient came from, so a guess can never read as a measurement."""

    SOURCED = "sourced"
    """Stated in a cited source, with a quote recorded on the spec."""

    CONVENTION = "convention"
    """Follows from the enzyme class, not from the source. Cofactor balance, usually."""

    DERIVED = "derived"
    """Computed from the model's own species, so it balances by construction."""


class MetaboliteSpec(BaseModel):
    """A metabolite the genome-scale model does not have yet."""

    id: str
    name: str
    compartment: str = "c"
    formula: str | None = Field(
        default=None,
        description="Charged formula in the model's convention. None means derive it.",
    )
    charge: int | None = None
    metanetx_id: str | None = None
    chebi_id: str | None = None
    evidence: EvidenceTier = EvidenceTier.SOURCED
    quote: str | None = None
    note: str | None = None


class ReactionSpec(BaseModel):
    """A reaction the genome-scale model does not have yet."""

    id: str
    name: str
    stoichiometry: dict[str, float] = Field(
        description="Metabolite id to coefficient. Negative consumes, positive produces."
    )
    lower_bound: float = 0.0
    upper_bound: float = 1000.0
    gene_reaction_rule: str = ""
    spontaneous: bool = False
    subsystem: str = ""
    evidence: EvidenceTier = EvidenceTier.SOURCED
    quote: str | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _spontaneous_has_no_gene(self) -> ReactionSpec:
        """A spontaneous reaction with a gene rule is a contradiction, not a detail.

        The distinction is the whole reason the betaxanthin family exists: the condensation
        is uncatalyzed, so no deletion in the knockout library can remove it.
        """
        if self.spontaneous and self.gene_reaction_rule:
            raise ValueError(
                f"{self.id}: spontaneous reaction cannot carry a gene_reaction_rule "
                f"({self.gene_reaction_rule!r})"
            )
        return self


class HeterologousPathway(BaseModel):
    """A named set of metabolites, reactions and genes added to a base strain."""

    name: str
    base_strain: str
    description: str = ""
    metabolites: list[MetaboliteSpec] = Field(default_factory=list)
    reactions: list[ReactionSpec] = Field(default_factory=list)
    constitutive_genes: list[str] = Field(
        default_factory=list,
        description=(
            "Heterologous gene ids. Always available: they are not yeast open reading "
            "frames, carry no gene token, and no knockout library can delete them."
        ),
    )
    source_keys: list[str] = Field(
        default_factory=list, description="Citation keys backing this pathway."
    )

    def evidence_census(self) -> dict[str, int]:
        """Count reactions by evidence tier, so the mix is reportable rather than assumed."""
        return dict(Counter(r.evidence.value for r in self.reactions))

    def product_ids(self) -> list[str]:
        """Metabolite ids the pathway produces and consumes only by draining them.

        These are the terminal products. A demand or exchange reaction, the single-species
        reactions, does not count as consumption: draining a product is how it leaves, not
        a step that uses it. For betaxanthin this is one species per condensation partner,
        and the measured pigment is their sum.
        """
        internal = [r for r in self.reactions if len(r.stoichiometry) > 1]
        produced = {m for r in internal for m, c in r.stoichiometry.items() if c > 0}
        consumed = {m for r in internal for m, c in r.stoichiometry.items() if c < 0}
        return sorted(produced - consumed)


def _derive_formula_and_charge(
    model: cobra.Model, stoichiometry: dict[str, float], product_id: str
) -> tuple[str, int]:
    """Balance one product against everything else in its reaction.

    Args:
        model: The model holding every OTHER species in the reaction.
        stoichiometry: The reaction, including the product being derived.
        product_id: The species whose formula is unknown.

    Returns:
        ``(formula, charge)`` that make the reaction mass and charge balanced.
    """
    elements: Counter[str] = Counter()
    charge = 0
    for met_id, coeff in stoichiometry.items():
        if met_id == product_id:
            continue
        met = model.metabolites.get_by_id(met_id)
        for element, count in met.elements.items():
            elements[element] -= int(coeff * count)
        charge -= int(coeff * (met.charge or 0))
    product_coeff = stoichiometry[product_id]
    if product_coeff != 1:
        raise ValueError(
            f"{product_id}: deriving a formula needs coefficient 1, got {product_coeff}"
        )
    negative = {e: n for e, n in elements.items() if n < 0}
    if negative:
        raise ValueError(f"{product_id}: derived a negative element count {negative}")
    formula = "".join(
        f"{e}{elements[e]}" if elements[e] > 1 else e
        for e in sorted(elements)
        if elements[e] > 0
    )
    return formula, charge


def apply_pathway(
    model: cobra.Model, pathway: HeterologousPathway, *, inplace: bool = False
) -> cobra.Model:
    """Add a pathway's metabolites, reactions and genes to a genome-scale model.

    Args:
        model: The base genome-scale model.
        pathway: The pathway to add.
        inplace: Edit ``model`` rather than a copy. Default copies, because the base model
            is shared across arms and a mutated copy is an invisible difference between them.

    Returns:
        The edited model.

    Raises:
        ValueError: A pathway id already exists, or a reaction cannot be balanced.
    """
    out = model if inplace else model.copy()

    clash = [m.id for m in pathway.metabolites if m.id in out.metabolites]
    if clash:
        raise ValueError(f"{pathway.name}: metabolite ids already in model: {clash}")
    clash = [r.id for r in pathway.reactions if r.id in out.reactions]
    if clash:
        raise ValueError(f"{pathway.name}: reaction ids already in model: {clash}")

    for met_spec in pathway.metabolites:
        met = cobra.Metabolite(
            met_spec.id,
            name=met_spec.name,
            compartment=met_spec.compartment,
            formula=met_spec.formula,
            charge=met_spec.charge,
        )
        annotation: dict[str, str] = {}
        if met_spec.metanetx_id:
            annotation["metanetx.chemical"] = met_spec.metanetx_id
        if met_spec.chebi_id:
            annotation["chebi"] = met_spec.chebi_id
        met.annotation = annotation
        out.add_metabolites([met])

    for rxn_spec in pathway.reactions:
        rxn = cobra.Reaction(
            rxn_spec.id,
            name=rxn_spec.name,
            lower_bound=rxn_spec.lower_bound,
            upper_bound=rxn_spec.upper_bound,
            subsystem=rxn_spec.subsystem or pathway.name,
        )
        out.add_reactions([rxn])
        rxn.add_metabolites(
            {out.metabolites.get_by_id(m): c for m, c in rxn_spec.stoichiometry.items()}
        )
        if rxn_spec.gene_reaction_rule:
            rxn.gene_reaction_rule = rxn_spec.gene_reaction_rule
        rxn.annotation = {"evidence_tier": rxn_spec.evidence.value}

    # Derive any formula left as None, then assert the reaction actually balances. A
    # heterologous reaction that silently unbalances the network turns every downstream
    # mass-balance residual into a constant offset nobody can attribute.
    for met_spec in pathway.metabolites:
        if met_spec.formula is not None:
            continue
        producing = [
            r
            for r in pathway.reactions
            if met_spec.id in r.stoichiometry and r.stoichiometry[met_spec.id] > 0
        ]
        if len(producing) != 1:
            raise ValueError(
                f"{met_spec.id}: need exactly one producing reaction to derive a "
                f"formula, found {len(producing)}"
            )
        formula, charge = _derive_formula_and_charge(
            out, producing[0].stoichiometry, met_spec.id
        )
        met = out.metabolites.get_by_id(met_spec.id)
        met.formula, met.charge = formula, charge
        log.info(
            "pathway %s: derived %s -> %s (charge %+d)",
            pathway.name,
            met_spec.id,
            formula,
            charge,
        )

    unbalanced: dict[str, dict[str, float]] = {}
    for rxn_spec in pathway.reactions:
        rxn = out.reactions.get_by_id(rxn_spec.id)
        if len(rxn.metabolites) == 1:
            continue  # a demand or exchange reaction is unbalanced by definition
        mass = rxn.check_mass_balance()
        if mass:
            unbalanced[rxn_spec.id] = mass
    if unbalanced:
        raise ValueError(f"{pathway.name}: unbalanced reactions {unbalanced}")

    return out
