# tests/torchcell/datamodels/test_media_library.py
"""The shared media library as the join layer.

Three properties, one per way the join breaks:

1. Every single-substance component and every dropout carries a structure identifier,
   so a medium's glucose is the same object as a chemogenomic dataset's glucose.
2. Every ``base_medium`` names a ``MEDIA_LIBRARY`` key, so "aggregate every record on
   an SD/MSG base" resolves to an object with components and provenance.
3. Every medium passes the L3 rules a dataset verifier runs
   (``media_membership``, ``media_compound_identity``), so a loader that imports a
   library constant cannot fail those rules for a reason the library owns.

Pure and offline: the pinned compound table plus the live pydantic models, no LMDB.
"""

from __future__ import annotations

from typing import Any

import pytest

from torchcell.datamodels import ontology_checks as oc
from torchcell.datamodels.identity import media_identity
from torchcell.datamodels.media import (
    HILLENMEYER_DROPOUT_MEDIA,
    MEDIA_LIBRARY,
    SC,
    SC_URA,
    SD_MSG,
    SGA_DM_SELECTION,
    YPD,
    YPD_AGAR,
    YPD_LIQUID,
    dropout,
)
from torchcell.datamodels.schema import (
    ComponentDefinition,
    DoseBasis,
    Environment,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    MediaComponentRole,
    ReferenceGenome,
    Temperature,
)
from torchcell.verification.common import shared_rule_results

L3_MEDIA_RULES = ("media_membership", "media_compound_identity")


def _records(media: Media) -> list[dict[str, Any]]:
    """One synthetic fitness record grown in ``media``, as the verifier reads them."""
    environment = Environment(media=media, temperature=Temperature(value=30.0))
    experiment = FitnessExperiment(
        dataset_name="media-library",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name="YAL001C", perturbed_gene_name="TFC3"
                )
            ]
        ),
        environment=environment,
        phenotype=FitnessPhenotype(fitness=0.9),
    )
    reference = FitnessExperimentReference(
        dataset_name="media-library",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="S288C"
        ),
        environment_reference=environment.model_copy(),
        phenotype_reference=FitnessPhenotype(fitness=1.0),
    )
    return [
        {"experiment": experiment.model_dump(), "reference": reference.model_dump()}
    ]


# --- identity ---------------------------------------------------------------- #
def test_every_single_substance_compound_is_identified() -> None:
    """No ``defined`` component and no dropout is name-only, anywhere in the library."""
    assert [i.model_dump() for i in oc.media_library_compound_issues()] == []


def test_identified_means_identified_not_merely_gapped() -> None:
    """The library's defined components carry real identifiers, not just honest gaps.

    ``media_library_compound_issues`` accepts a typed gap, which is right for a
    verifier (an unresolvable compound must still be encodable as unresolvable). The
    library itself should do better than that, and today it does: only the substances
    with no single-molecule InChIKey fall back to a gap.
    """
    gapped: dict[str, str] = {}
    for key, media in sorted(MEDIA_LIBRARY.items()):
        for compound in [
            *(
                c.compound
                for c in media.components
                if c.definition is ComponentDefinition.defined
            ),
            *media.dropouts,
        ]:
            if not oc.compound_has_identity(compound):
                gapped[compound.name] = key
    # agar has a ChEBI and a CID but no single-molecule InChIKey, so it IS identified;
    # cellobiose is the one library substance the pinned table has not resolved yet.
    assert sorted(gapped) == ["cellobiose"], gapped


def test_undefined_preparations_carry_no_identity_gap() -> None:
    """Yeast extract, peptone, commercial YNB and the drop-out powders are not gaps.

    ``resolved_compound`` would attach a ``ProvenanceGap`` on ``inchikey``, which would
    read as "we have not found it yet". There is nothing to find: the truth is that
    the preparation is undefined, and ``ComponentDefinition`` is where that is said.
    """
    for key, media in sorted(MEDIA_LIBRARY.items()):
        for component in media.components:
            if component.definition is ComponentDefinition.defined:
                continue
            assert component.compound.provenance_gaps == [], f"{key}:{component}"
            assert not oc.compound_has_identity(component.compound), (
                f"{key}:{component}"
            )


def test_selection_agents_share_one_identity_across_families() -> None:
    """G418 in the SGA media and in Lian 2019's SED-URA is ONE compound."""
    from torchcell.datamodels.media import SED_URA_G418

    def _agent(media: Media, stem: str) -> Any:
        return next(
            c.compound
            for c in media.components
            if c.role is MediaComponentRole.selection_agent
            and c.compound.name.startswith(stem)
        )

    sga = _agent(SGA_DM_SELECTION, "G418")
    lian = _agent(SED_URA_G418, "G418")
    assert sga.name == lian.name
    assert sga.inchikey == lian.inchikey is not None


# --- bases ------------------------------------------------------------------- #
def test_every_base_medium_resolves_to_a_library_member() -> None:
    assert [i.model_dump() for i in oc.media_base_issues()] == []


def test_a_base_is_a_component_subset_of_its_derivatives() -> None:
    assert [i.model_dump() for i in oc.media_derivation_issues()] == []


def test_the_sga_family_derives_from_a_real_sd_msg_object() -> None:
    """``SD_MSG`` is an object with components, not a label two loaders agree on."""
    assert SGA_DM_SELECTION.base_medium == "SD_MSG"
    base = {(c.compound.name, c.role) for c in SD_MSG.components}
    derived = {(c.compound.name, c.role) for c in SGA_DM_SELECTION.components}
    assert base and base <= derived


# --- the YPD family ---------------------------------------------------------- #
def test_ypd_and_ypd_liquid_are_one_recipe_differing_only_in_state() -> None:
    assert YPD.components == YPD_LIQUID.components
    assert (YPD.state, YPD_LIQUID.state) == ("solid", "liquid")
    assert YPD.base_medium == YPD_LIQUID.base_medium == "YPD"


def test_ypd_agar_adds_only_the_gelling_agent() -> None:
    extra = [c for c in YPD_AGAR.components if c not in YPD.components]
    assert [c.compound.name for c in extra] == ["agar"]
    assert extra[0].role is MediaComponentRole.gelling_agent


# --- dropouts ---------------------------------------------------------------- #
def test_sc_ura_is_a_typed_edit_of_sc() -> None:
    assert SC_URA.base_medium == "SC"
    assert [c.name for c in SC_URA.dropouts] == ["uracil"]
    assert "uracil" not in {c.compound.name for c in SC_URA.components}
    assert len(SC_URA.components) == len(SC.components) - 1


def test_dropout_rejects_a_nutrient_the_base_does_not_contain() -> None:
    """A dropout that removes nothing would assert an edit that never happened."""
    with pytest.raises(ValueError, match="not a component"):
        dropout(SC, "peptone", name="SC minus peptone")


def test_hillenmeyer_dropout_media_cover_every_hom_condition_label() -> None:
    """15 derived media plus the control, which is plain SC rather than a dropout."""
    assert len(HILLENMEYER_DROPOUT_MEDIA) == 16
    assert HILLENMEYER_DROPOUT_MEDIA["vitamin drop-out control media"] is SC
    tryptophan = HILLENMEYER_DROPOUT_MEDIA["tryptophan dropout"]
    assert [c.name for c in tryptophan.dropouts] == ["L-tryptophan"]
    assert tryptophan.base_medium == "SC"


def test_a_partial_dropout_keeps_the_nutrient_as_a_typed_reduction() -> None:
    """A partial drop-out is a reduced level the source never states, not a removal.

    The reduction is typed as a dose basis so the medium's composition identity
    differs from the full SC recipe (four partial-dropout media collapsed onto SC when
    the reduction lived only in a note).
    """
    partial = HILLENMEYER_DROPOUT_MEDIA["biotin partial drop-out"]
    assert partial.dropouts == []
    biotin = next(c for c in partial.components if c.compound.name == "biotin")
    assert biotin.concentration is not None
    assert biotin.concentration.value is None
    assert biotin.concentration.basis == DoseBasis.reduced_from_standard
    assert "partial drop-out" in (biotin.note or "")
    assert media_identity(partial) != media_identity(SC)


# --- the verifier's own L3 rules --------------------------------------------- #
@pytest.mark.parametrize("key", sorted(MEDIA_LIBRARY))
def test_every_library_medium_passes_the_l3_media_rules(key: str) -> None:
    """A loader importing a library constant cannot fail a rule the library owns."""
    results = {
        r.name: r
        for r in shared_rule_results(_records(MEDIA_LIBRARY[key]), sgd_genes=None)
    }
    for rule in L3_MEDIA_RULES:
        assert results[rule].passed, f"{key}: {results[rule].message}"


def test_media_membership_reports_a_library_match_not_a_derivation() -> None:
    """A shipped constant matches by VALUE, which is the strong form of the rule."""
    results = {
        r.name: r for r in shared_rule_results(_records(YPD_LIQUID), sgd_genes=None)
    }
    membership = results["media_membership"]
    assert membership.details["n_library_records"] == 1
    assert membership.details["matched_media"] == {
        YPD_LIQUID.name: "library:YPD_LIQUID"
    }


def test_provenance_quotes_name_a_mirrored_artifact() -> None:
    """Every sourced value pins a citation key and a sha256, never a bare URL."""
    for key, media in sorted(MEDIA_LIBRARY.items()):
        sourced = [
            *media.provenance,
            *(sv for c in media.components for sv in c.provenance),
        ]
        for value in sourced:
            assert value.provenance.citation_key, key
            assert value.provenance.sha256, f"{key}: {value.quote[:60]}"
            assert value.quote.strip(), key
