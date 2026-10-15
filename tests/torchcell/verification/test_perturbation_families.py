# tests/torchcell/verification/test_perturbation_families.py
# [[tests.torchcell.verification.test_perturbation_families]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/verification/test_perturbation_families.py
"""A perturbation family is derived from the union, or it is exhaustive by test.

``torchcell.verification.perturbation_families`` answers "which leaves is this rule
about" from the ``GenePerturbationType`` union and from the class the leaves inherit, so
a new leaf joins the family it belongs to the moment it lands. These tests hold the
derivations to the union, hold the one set that cannot be derived to a partition of it,
and pin the two memberships a hand-written list had already got wrong.
"""

from __future__ import annotations

from typing import get_args

import pytest

from torchcell.datamodels.schema import (
    DeletionPerturbation,
    GeneAdditionPerturbation,
    GenePerturbationType,
    HeterozygousDeletionPerturbation,
    SequencePerturbation,
)
from torchcell.verification.perturbation_families import (
    ALLELE_BEARING_PERTURBATION_TYPES,
    PLAIN_GENE_NAME_PERTURBATION_TYPES,
    addition_perturbation_types,
    full_deletion_perturbation_types,
    gene_perturbation_types,
    heterologous_identifier,
    pathway_member_perturbation_types,
    perturbation_types_declaring,
    perturbation_types_under,
    site_keyed_perturbation_types,
)


def _tags_of(base: type) -> set[str]:
    """The type tags of the union's members that subclass ``base``, read here directly."""
    return {
        str(leaf.model_fields["perturbation_type"].default)
        for leaf in get_args(GenePerturbationType)
        if issubclass(leaf, base)
    }


def test_the_type_tag_of_every_union_member_is_unique_and_counted() -> None:
    """One tag per leaf: 38 genotype leaves on the integrated branch, no tag shared."""
    leaves = get_args(GenePerturbationType)
    tags = [str(leaf.model_fields["perturbation_type"].default) for leaf in leaves]
    assert len(tags) == len(set(tags)) == len(leaves) == 38
    assert gene_perturbation_types() == set(tags)


def test_the_addition_family_is_every_subclass_of_the_addition_leaf() -> None:
    """Four members, and the two the hand-written set missed are named.

    ``ADDITION_PERTURBATION_TYPES`` held ``{"gene_addition", "heterologous_pathway"}``
    while ``GeneAdditionPerturbation`` has two further placed subclasses, so a rule that
    read the literal set treated a promoter-driven integration and a
    variable-promoter reporter as if they added no DNA.
    """
    assert addition_perturbation_types() == _tags_of(GeneAdditionPerturbation)
    assert addition_perturbation_types() == {
        "gene_addition",
        "heterologous_pathway",
        "promoter_driven_gene_integration",
        "variable_promoter_reporter",
    }


def test_the_full_deletion_family_is_every_subclass_of_the_deletion_leaf() -> None:
    """Eleven members including the Keio leaf; a heterozygous deletion is not one.

    The hand-written set held the two SGA leaves only. ``HeterozygousDeletionPerturbation``
    is excluded by the schema's own placement, not by a list: it is a
    ``PresenceAbsencePerturbation`` and not a ``DeletionPerturbation``, because one
    deleted copy of two is not the gene's absence.
    """
    assert full_deletion_perturbation_types() == _tags_of(DeletionPerturbation)
    assert {"sga_kanmx_deletion", "sga_natmx_deletion", "deletion_strain"} <= (
        full_deletion_perturbation_types()
    )
    assert not issubclass(HeterozygousDeletionPerturbation, DeletionPerturbation)
    assert "heterozygous_deletion" not in full_deletion_perturbation_types()


def test_a_family_can_be_derived_from_a_declared_field() -> None:
    """``pathway_name`` and ``flanking_systematic_gene_names`` name their own families."""
    assert pathway_member_perturbation_types() == perturbation_types_declaring(
        "pathway_name"
    )
    assert pathway_member_perturbation_types() == {"heterologous_pathway"}
    assert site_keyed_perturbation_types() == {"site_variant_call"}
    assert perturbation_types_declaring("a_field_no_leaf_declares") == frozenset()


def test_perturbation_types_under_reads_the_whole_subtree() -> None:
    """An abstract parent (``SequencePerturbation``) answers for every leaf below it."""
    assert perturbation_types_under(SequencePerturbation) == _tags_of(
        SequencePerturbation
    )
    assert ALLELE_BEARING_PERTURBATION_TYPES <= perturbation_types_under(
        SequencePerturbation
    )


def test_the_allele_bearing_split_is_a_partition_of_the_union() -> None:
    """The one set no class property decides is exhaustive by test instead.

    Together the two sets are every genotype leaf and they share no member, so a new
    leaf fails this test until it is named in one of them: its ``perturbed_gene_name``
    either carries an allele designation (the spelling rules do not apply) or is the
    gene's common name (they do).
    """
    assert (
        ALLELE_BEARING_PERTURBATION_TYPES | PLAIN_GENE_NAME_PERTURBATION_TYPES
        == gene_perturbation_types()
    )
    assert not (ALLELE_BEARING_PERTURBATION_TYPES & PLAIN_GENE_NAME_PERTURBATION_TYPES)
    assert ALLELE_BEARING_PERTURBATION_TYPES == {
        "allele",
        "damp",
        "suppressor_allele",
        "temperature_sensitive_allele",
        "sequence_variant",
    }


def test_the_leaves_that_hold_a_designation_in_a_field_are_not_allele_bearing() -> None:
    """``allele_name`` is the reason those leaves keep the plain gene name.

    The membership is the opposite of what a field-based derivation would produce, which
    is why the set is explicit: ``named_allele`` and ``conditional_allele`` declare
    ``allele_name`` and are NOT members, while the four SGA leaves and
    ``sequence_variant`` declare no such field and are.
    """
    with_a_field = perturbation_types_declaring("allele_name")
    assert "named_allele" in with_a_field
    assert not (with_a_field & ALLELE_BEARING_PERTURBATION_TYPES)


@pytest.mark.parametrize(
    ("perturbation", "species", "expected"),
    [
        # a deletion carries no source_organism at all: never heterologous
        (
            {"perturbation_type": "sga_kanmx_deletion"},
            "Saccharomyces cerevisiae",
            False,
        ),
        # an extra copy of a NATIVE gene names its real locus (162 of Brunk 2016's 603)
        (
            {
                "perturbation_type": "heterologous_pathway",
                "source_organism": "Escherichia coli",
            },
            "Escherichia coli",
            False,
        ),
        # another organism's gene: the identifier is not a locus of the host
        (
            {
                "perturbation_type": "heterologous_pathway",
                "source_organism": "Staphylococcus aureus",
            },
            "Escherichia coli",
            True,
        ),
        # a stated-but-unknown source is not the host either, so it stays out
        (
            {"perturbation_type": "heterologous_pathway", "source_organism": None},
            "Escherichia coli",
            True,
        ),
    ],
)
def test_the_heterologous_test_reads_the_field_not_the_leaf(
    perturbation: dict[str, object], species: str, expected: bool
) -> None:
    """One test for one question, keyed on the field every addition leaf carries."""
    assert heterologous_identifier(perturbation, species) is expected
