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

import os
import os.path as osp
from typing import Any

import pytest

from torchcell.datamodels import ontology_checks as oc
from torchcell.datamodels.identity import media_identity
from torchcell.datamodels.media import (
    CARBON_FREE_MEDIA,
    HILLENMEYER_DROPOUT_MEDIA,
    MEDIA_LIBRARY,
    SC,
    SC_URA,
    SD_MSG,
    SGA_DM_SELECTION,
    SM,
    SM_AGAR,
    SM_DEFERRED,
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
from torchcell.verification.sourced import audit_sourced_value

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


# --- the SM family ----------------------------------------------------------- #
def test_the_sm_media_are_composed_and_sourced() -> None:
    """The three SM objects replace a stub that carried no composition at all.

    Before this, Mulleder, Zelezniak and Messner all emitted
    ``Media(name="SM", state=..., is_synthetic=True)``: no components, no provenance,
    joinable only on the four-character string.
    """
    for media in (SM, SM_AGAR, SM_DEFERRED):
        assert media.components, media.name
        assert media.provenance, media.name
        assert media.is_synthetic
        assert media.base_medium in MEDIA_LIBRARY


def test_sm_agar_adds_only_the_gelling_agent() -> None:
    """The plate is the liquid recipe plus agar, which is why they share a base."""
    extra = [c for c in SM_AGAR.components if c not in SM.components]
    assert [c.compound.name for c in extra] == ["agar"]
    assert extra[0].role is MediaComponentRole.gelling_agent
    assert (SM.state, SM_AGAR.state) == ("liquid", "solid")
    assert SM.base_medium == SM_AGAR.base_medium == "SM"


def test_sm_names_its_carbon_and_nitrogen_sources() -> None:
    """A minimal medium that resolves neither would reach FBA describing nothing."""
    roles = {c.role: c.compound.name for c in SM.components}
    assert roles[MediaComponentRole.carbon_source] == "D-glucose"
    assert roles[MediaComponentRole.nitrogen_source] == "ammonium sulfate"


def test_sm_ammonium_sulfate_is_an_identity_without_an_amount() -> None:
    """Its mass is already inside the 6.7 g/L YNB line, so no number is recorded.

    The identity is sourced (the paper's nitrogen-starvation medium is a DIFFERENT,
    ammonium-sulfate-free product), the amount is not, and ``open_gaps`` says so
    rather than a plausible 5 g/L being written in.
    """
    nitrogen = next(
        c for c in SM.components if c.role is MediaComponentRole.nitrogen_source
    )
    assert nitrogen.concentration is None
    assert nitrogen.definition is ComponentDefinition.defined
    assert nitrogen.provenance
    assert "ammonium sulfate" in SM.open_gaps


def test_zelezniak_sm_is_a_distinct_node_with_a_deferred_composition() -> None:
    """Two papers' different SM recipes must not collapse onto one node.

    Zelezniak 2018 states no recipe, so its medium keeps the grams OUT rather than
    borrowing Mulleder's, and it is a documented carbon-free medium for that reason.
    """
    assert media_identity(SM_DEFERRED) != media_identity(SM)
    assert media_identity(SM_DEFERRED) != media_identity(SM_AGAR)
    only = SM_DEFERRED.components[0]
    assert len(SM_DEFERRED.components) == 1
    assert only.definition is ComponentDefinition.composition_deferred
    assert only.defers_to == ["mullederPrototrophicDeletionMutant2012"]
    assert "SM_DEFERRED" in CARBON_FREE_MEDIA


def _library_root() -> str | None:
    data_root = os.environ.get("DATA_ROOT")
    if data_root is None:
        return None
    root = osp.join(data_root, "torchcell-library")
    return root if osp.isdir(root) else None


def test_sm_quotes_are_verbatim_in_the_mirrored_papers() -> None:
    """Every SM number is a quote that is still a substring of the pinned bytes."""
    root = _library_root()
    if root is None:
        pytest.skip("torchcell-library mirror not mounted")
    for media in (SM, SM_AGAR, SM_DEFERRED):
        sourced = [
            *media.provenance,
            *(sv for c in media.components for sv in c.provenance),
        ]
        assert sourced, media.name
        for value in sourced:
            result = audit_sourced_value(value, root)
            assert result.passed, f"{media.name}: {result.message}"


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
