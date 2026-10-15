# torchcell/verification/perturbation_families.py
# [[torchcell.verification.perturbation_families]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/perturbation_families.py
# Test file: tests/torchcell/verification/test_perturbation_families.py
"""Which perturbation leaves a verification rule is about, derived from the schema.

A rule that needs "the gene additions", "the full deletions" or "the pathway members"
used to name them as a literal set of ``perturbation_type`` strings, one copy per rule.
That is the defect shape PR #1104 closed on the adapter side and the ontology
unification removed from the node lanes (``torchcell.knowledge_graphs.perturbation_lanes``
emits every ``GenePerturbationType`` member rather than a pinned class tuple): a new
leaf lands, no hand-written set names it, and the rule silently stops covering it. Three
such sets were measured on the integrated branch and one of them was already wrong --
``ADDITION_PERTURBATION_TYPES`` held ``gene_addition`` and ``heterologous_pathway`` while
the union has four subclasses of ``GeneAdditionPerturbation``, so the construct-placement
census skipped every promoter-driven integration and every variable-promoter reporter.

So a family is DERIVED here, from the union and from the class the leaves inherit:

- :func:`addition_perturbation_types` -- the subclasses of ``GeneAdditionPerturbation``.
- :func:`full_deletion_perturbation_types` -- the subclasses of ``DeletionPerturbation``.
- :func:`pathway_member_perturbation_types` -- the leaves that declare ``pathway_name``.
- :func:`site_keyed_perturbation_types` -- the leaves whose identifier is a derived site
  id rather than a locus, which is what declaring ``flanking_systematic_gene_names``
  says.
- :func:`heterologous_identifier` -- the ONE field test for "is this identifier a gene of
  another genome", so the yeast and bacterial families stop answering that question two
  ways (audit C section 2).

Where no property of the class decides the family, the set stays explicit and is made
EXHAUSTIVE BY TEST instead of by hand: ``ALLELE_BEARING_PERTURBATION_TYPES`` and
``PLAIN_GENE_NAME_PERTURBATION_TYPES`` must together be every genotype leaf, so a new
leaf fails ``tests/torchcell/verification/test_perturbation_families.py`` until someone
says which of the two it is.

Nothing here reads the schema at import time (the imports are inside the functions and
the answers are cached), so a verification module can hold a reference to a family
without paying for ``torchcell.datamodels.schema`` to be imported.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from typing import Any, get_args

__all__ = [
    "ALLELE_BEARING_PERTURBATION_TYPES",
    "PLAIN_GENE_NAME_PERTURBATION_TYPES",
    "addition_perturbation_types",
    "full_deletion_perturbation_types",
    "gene_perturbation_types",
    "heterologous_identifier",
    "pathway_member_perturbation_types",
    "perturbation_types_declaring",
    "perturbation_types_under",
    "site_keyed_perturbation_types",
]


@cache
def _genotype_leaves() -> tuple[type, ...]:
    """Every ``GenePerturbationType`` member, nested unions flattened by ``get_args``."""
    from torchcell.datamodels.schema import GenePerturbationType

    return tuple(get_args(GenePerturbationType))


def _type_tag(leaf: type) -> str:
    """The leaf's ``perturbation_type`` literal, which is unique per leaf."""
    return str(leaf.model_fields["perturbation_type"].default)  # type: ignore[attr-defined]


@cache
def gene_perturbation_types() -> frozenset[str]:
    """The ``perturbation_type`` literal of every genotype-axis leaf."""
    return frozenset(_type_tag(leaf) for leaf in _genotype_leaves())


@cache
def perturbation_types_under(base: type) -> frozenset[str]:
    """The literals of the genotype leaves that are subclasses of ``base``.

    ``base`` itself is included when it is a leaf of the union, which is the usual case
    (``GeneAdditionPerturbation`` and ``DeletionPerturbation`` are both placed leaves and
    parents of placed leaves).
    """
    return frozenset(
        _type_tag(leaf) for leaf in _genotype_leaves() if issubclass(leaf, base)
    )


@cache
def perturbation_types_declaring(field: str) -> frozenset[str]:
    """The literals of the genotype leaves that declare ``field``.

    Declaring a field is the strongest statement a leaf makes about what it holds, so a
    rule whose subject is that field reads this rather than a list of leaf names: a new
    leaf that carries the field is covered the moment it lands.
    """
    return frozenset(
        _type_tag(leaf)
        for leaf in _genotype_leaves()
        if field in leaf.model_fields  # type: ignore[attr-defined]
    )


def addition_perturbation_types() -> frozenset[str]:
    """Every leaf that ADDS DNA: the subclasses of ``GeneAdditionPerturbation``.

    Four on the integrated branch (``gene_addition``, ``heterologous_pathway``,
    ``promoter_driven_gene_integration``, ``variable_promoter_reporter``). The two
    promoter leaves were missing from every hand-written copy of this set.
    """
    from torchcell.datamodels.schema import GeneAdditionPerturbation

    return perturbation_types_under(GeneAdditionPerturbation)


def full_deletion_perturbation_types() -> frozenset[str]:
    """Every leaf that deletes a whole gene: the subclasses of ``DeletionPerturbation``.

    What a viable-deletion cross-check needs (an essential gene must have no viable
    full-deletion single mutant). The hand-written set held the two SGA leaves only, so
    the check could not be reused for a Keio-based one although ``DeletionStrainPerturbation``
    asserts the same absence. ``HeterozygousDeletionPerturbation`` is deliberately NOT a
    member: it is a ``PresenceAbsencePerturbation`` and not a ``DeletionPerturbation``,
    because one deleted copy of two is not the gene's absence.
    """
    from torchcell.datamodels.schema import DeletionPerturbation

    return perturbation_types_under(DeletionPerturbation)


def pathway_member_perturbation_types() -> frozenset[str]:
    """The leaves that state pathway membership, i.e. that declare ``pathway_name``."""
    return perturbation_types_declaring("pathway_name")


def site_keyed_perturbation_types() -> frozenset[str]:
    """The leaves whose identifier is a site id rather than a locus of the assembly.

    A leaf keyed on ``<replicon>:<position>`` carries the loci that FLANK the site, which
    is what ``flanking_systematic_gene_names`` is for and what a containment rule checks
    in place of the site id itself.
    """
    return perturbation_types_declaring("flanking_systematic_gene_names")


def heterologous_identifier(perturbation: Mapping[str, Any], species: str) -> bool:
    """True when this perturbation's identifier is a gene of ANOTHER genome.

    The one test for the question both families used to answer their own way. A leaf that
    adds DNA carries ``source_organism``; when that organism is not the record's own
    species the stored identifier is a gene of that other genome (``MvaSEf``, ``ATF1``),
    so it is not a claim about a locus of the host and a host-containment rule that
    counted it would fail a correct record. An extra copy of a NATIVE gene names its real
    locus and is not heterologous (measured: 162 of Brunk 2016's 603 pathway
    perturbations carry ``source_organism`` equal to the record's own species). A leaf
    that carries no ``source_organism`` at all is never heterologous, which is why the
    default is the record's species rather than ``None``.
    """
    return bool(perturbation.get("source_organism", species) != species)


#: Leaves whose ``perturbed_gene_name`` carries an ALLELE DESIGNATION (``act1-101``,
#: ``cys3_damp``, ``lte1-supp1``) rather than the gene's common name, because the strain
#: the source names is keyed on the allele. The one-spelling and round-trip gene-name
#: rules do not apply to them; their ``systematic_gene_name`` is still checked.
#:
#: Explicit rather than derived: no property of these classes distinguishes them. The
#: leaves that hold a designation in a FIELD of its own (``allele_name`` on
#: ``NamedAllelePerturbation`` and ``ConditionalAllelePerturbation``, ``call`` on the
#: variant-call leaves) keep the plain gene name in ``perturbed_gene_name`` and so are
#: not members, which is the opposite of what a field-based derivation would say.
#: :data:`PLAIN_GENE_NAME_PERTURBATION_TYPES` is its complement and the test holds the
#: two to the union, so a new leaf cannot be silently judged by the wrong rule.
ALLELE_BEARING_PERTURBATION_TYPES: frozenset[str] = frozenset(
    {
        "allele",
        "damp",
        "suppressor_allele",
        "temperature_sensitive_allele",
        "sequence_variant",
    }
)

#: The complement: leaves whose ``perturbed_gene_name`` is the gene's own common name, so
#: the spelling rules apply. Grouped by why, in the order the union declares them:
#:
#: - the deletion leaves (``deletion``, the Kan/Nat and SGA variants, the barcoded and
#:   mean ones, ``deletion_strain``, ``crispr_deletion``, ``marker_deletion``,
#:   ``heterozygous_deletion``, ``span_deletion_call``): a deleted gene is named, not
#:   allele-designated, and the construct rides on ``cassette`` / ``collection``.
#: - the addition leaves (``gene_addition``, ``heterologous_pathway``,
#:   ``promoter_driven_gene_integration``, ``variable_promoter_reporter``): the name is
#:   the added gene's own, in its source organism's spelling.
#: - the expression-modulation leaves (``crispr_activation``, ``crispr_interference``,
#:   ``inducible_promoter_insertion``, ``promoter_replacement``): the target gene is
#:   named and the construct is in ``crispr`` / ``promoter_name``.
#: - the designation-in-a-field leaves (``named_allele``, ``conditional_allele``,
#:   ``marked_allele``, ``degron``, ``released_identifier``, the three
#:   ``*_variant_call`` leaves, ``natural_loss_of_function``): the designation has its own
#:   typed field, so ``perturbed_gene_name`` is the gene.
#: - the natural-variation and copy-number leaves (``natural_gene_absence``,
#:   ``natural_gene_presence``, ``copy_number_variant``, ``engineered_copy_number``) and
#:   ``transposon_insertion``: a presence, a copy count or an insertion is not an allele.
PLAIN_GENE_NAME_PERTURBATION_TYPES: frozenset[str] = frozenset(
    {
        "barcoded_kanmx_deletion",
        "conditional_allele",
        "copy_number_variant",
        "crispr_activation",
        "crispr_deletion",
        "crispr_interference",
        "degron",
        "deletion",
        "deletion_strain",
        "engineered_copy_number",
        "gene_addition",
        "heterologous_pathway",
        "heterozygous_deletion",
        "inducible_promoter_insertion",
        "kanmx_deletion",
        "locus_variant_call",
        "marked_allele",
        "marker_deletion",
        "mean_deletion",
        "named_allele",
        "natmx_deletion",
        "natural_gene_absence",
        "natural_gene_presence",
        "natural_loss_of_function",
        "promoter_driven_gene_integration",
        "promoter_replacement",
        "released_identifier",
        "sga_kanmx_deletion",
        "sga_natmx_deletion",
        "site_variant_call",
        "span_deletion_call",
        "transposon_insertion",
        "variable_promoter_reporter",
    }
)
