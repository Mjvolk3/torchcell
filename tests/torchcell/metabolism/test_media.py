# tests/torchcell/metabolism/test_media.py
"""The SGA selection media as exchange bounds: resolution and the organic-nitrogen bound.

Pure tests on a small synthetic cobra model, so they need no yeast-GEM download. The
model carries one exchange per species the recipes name; anything the recipe names that
the model lacks shows up as ``unresolved``, which is the failure these tests guard.
"""

import cobra
import pytest

from torchcell.datamodels.media import (
    CARBON_FREE_MEDIA,
    MEDIA_LIBRARY,
    SGA_TM_SELECTION,
    SM,
    SM_AGAR,
)
from torchcell.datamodels.schema import ComponentDefinition, MediaComponentRole
from torchcell.metabolism.media import (
    SGA_DM_SELECTION_FBA,
    SGA_TM_SELECTION_FBA,
    SM_FBA,
    ExchangeIndex,
    MediaBounds,
    UptakePolicy,
    build_exchange_index,
    media_to_bounds,
)

SPECIES = [
    "H2O",
    "oxygen",
    "H+",
    "phosphate",
    "sulphate",
    "sodium",
    "potassium",
    "chloride",
    "iron(2+)",
    "Cu2(+)",
    "Mn(2+)",
    "Zn(2+)",
    "Mg(2+)",
    "Ca(2+)",
    "D-glucose",
    "D-galactose",
    "D-fructose",
    "D-mannose",
    "D-xylose",
    "maltose",
    "sucrose",
    "raffinose",
    "trehalose",
    "glycerol",
    "oleate",
    "myristate",
    "acetate",
    "(S)-lactate",
    "ethanol",
    "ammonium",
    "L-glutamate",
    "biotin",
    "(R)-pantothenate",
    "folate",
    "myo-inositol",
    "nicotinate",
    "4-aminobenzoate",
    "pyridoxine",
    "riboflavin",
    "thiamine",
    "L-alanine",
    "L-arginine",
    "L-asparagine",
    "L-aspartate",
    "L-cysteine",
    "L-glutamine",
    "L-glycine",
    "L-histidine",
    "L-isoleucine",
    "L-leucine",
    "L-lysine",
    "L-methionine",
    "L-phenylalanine",
    "L-proline",
    "L-serine",
    "L-threonine",
    "L-tryptophan",
    "L-tyrosine",
    "L-valine",
    "adenine",
    "uracil",
]


@pytest.fixture(scope="module")
def model() -> cobra.Model:
    m = cobra.Model("toy")
    for i, name in enumerate(SPECIES):
        met = cobra.Metabolite(f"s_{i}", name=name, compartment="e")
        rxn = cobra.Reaction(
            f"EX_{i}", name=f"{name} exchange", lower_bound=0.0, upper_bound=1000.0
        )
        rxn.add_metabolites({met: -1.0})
        m.add_reactions([rxn])
    return m


def _bound_by_name(bounds: MediaBounds, name: str) -> float:
    hits = [b for b in bounds.bounds.values() if b.metabolite_name == name]
    assert len(hits) == 1, (name, hits)
    return float(hits[0].uptake_bound)


def test_bloom_yp_media_resolve_their_carbon_source(model: cobra.Model) -> None:
    """A YP + sugar medium lets the model see the sugar; YP alone has no carbon bound.

    The Bloom 2019 carbon-source conditions are MEDIA (typed components) rather than
    ``carbon_source`` perturbations precisely so this resolution happens; yeast extract
    and peptone stay ``excluded_by_role`` (intrinsically undefined digests).
    """
    from torchcell.datamodels.media import YP, YP_GALACTOSE, YP_LACTATE, YPD_ETHANOL

    galactose = media_to_bounds(YP_GALACTOSE, model)
    assert _bound_by_name(galactose, "D-galactose") == UptakePolicy().carbon_uptake
    assert sorted(galactose.excluded_names) == ["peptone", "yeast extract"]
    assert galactose.unresolved_names == []
    bare = media_to_bounds(YP, model)
    assert bare.bounds == {}
    lactate = media_to_bounds(YP_LACTATE, model)
    assert _bound_by_name(lactate, "(S)-lactate") == UptakePolicy().carbon_uptake
    both = media_to_bounds(YPD_ETHANOL, model)
    assert {b.metabolite_name for b in both.bounds.values()} == {"D-glucose", "ethanol"}


def test_sga_tm_resolves_every_nutrient(model: cobra.Model) -> None:
    bounds = media_to_bounds(SGA_TM_SELECTION_FBA, model)
    assert bounds.unresolved_names == []
    assert sorted(bounds.excluded_names) == sorted(
        [
            "agar",
            "L-canavanine",
            "thialysine (S-(2-aminoethyl)-L-cysteine)",
            "G418 (geneticin)",
            "nourseothricin (clonNAT)",
        ]
    )


def test_sga_tm_dropouts_are_closed(model: cobra.Model) -> None:
    bounds = media_to_bounds(SGA_TM_SELECTION_FBA, model)
    opened = {b.metabolite_name for b in bounds.bounds.values()}
    for dropped in ("L-histidine", "L-arginine", "L-lysine", "uracil"):
        assert dropped not in opened
    assert "adenine" in opened
    assert "ammonium" not in opened, "SD/MSG replaces ammonium sulfate with MSG"


def test_sga_dm_keeps_uracil(model: cobra.Model) -> None:
    bounds = media_to_bounds(SGA_DM_SELECTION_FBA, model)
    opened = {b.metabolite_name for b in bounds.bounds.values()}
    assert "uracil" in opened
    assert "L-histidine" not in opened


def test_msg_is_bounded_at_the_supplement_rate_not_opened(model: cobra.Model) -> None:
    policy = UptakePolicy()
    bounds = media_to_bounds(SGA_TM_SELECTION_FBA, model, policy=policy)
    assert (
        _bound_by_name(bounds, "L-glutamate") == policy.organic_nitrogen_uptake == 0.165
    )
    assert _bound_by_name(bounds, "sodium") == policy.unlimited_uptake
    assert _bound_by_name(bounds, "D-glucose") == policy.carbon_uptake == 3.3


def test_ammonium_stays_open_in_sm(model: cobra.Model) -> None:
    bounds = media_to_bounds(SM_FBA, model)
    assert _bound_by_name(bounds, "ammonium") == 1000.0


def test_fba_recipe_tracks_the_ontology_dropouts() -> None:
    assert [d.name for d in SGA_TM_SELECTION_FBA.dropouts] == [
        d.name for d in SGA_TM_SELECTION.dropouts
    ]
    assert SGA_TM_SELECTION_FBA.provenance == SGA_TM_SELECTION.provenance


@pytest.fixture(scope="module")
def index(model: cobra.Model) -> ExchangeIndex:
    return build_exchange_index(model)


@pytest.mark.parametrize("key", sorted(MEDIA_LIBRARY))
def test_every_library_medium_states_a_carbon_source(key: str) -> None:
    """A medium names a carbon source, or is a base listed as carbon-free with a reason.

    The failure this locks out is silent: the shipped ``SC`` carried the 9 YNB
    vitamins, all 20 amino acids and two nucleobases and NO sugar, so
    ``media_to_bounds`` opened no carbon exchange and every dataset that grew on SC
    reached FBA describing a medium nothing can grow in.
    """
    media = MEDIA_LIBRARY[key]
    carbon = [c for c in media.components if c.role is MediaComponentRole.carbon_source]
    assert carbon or key in CARBON_FREE_MEDIA, (
        f"{key} names no carbon source and is not a documented carbon-free base"
    )


@pytest.mark.parametrize("key", sorted(MEDIA_LIBRARY))
def test_every_library_medium_resolves_or_says_why_not(
    key: str, model: cobra.Model, index: ExchangeIndex
) -> None:
    """Each component reaches an exchange, is excluded by role, or is a named mixture.

    "Unresolved" is only honest when the component is not one species: a commercial
    YNB, an SC or CSM drop-out powder, a polysorbate, or the SynH3- base whose
    composition is deferred to an unmirrored paper. A ``defined`` component failing to
    resolve is a naming drift between this library and the metabolism resolver, which
    is exactly what this catches.
    """
    media = MEDIA_LIBRARY[key]
    bounds = media_to_bounds(media, model, index=index)
    by_name = {c.compound.name: c for c in media.components}
    undocumented = [
        name
        for name in bounds.unresolved_names
        if by_name[name].definition is ComponentDefinition.defined
    ]
    assert undocumented == [], f"{key}: {undocumented}"
    if key not in CARBON_FREE_MEDIA:
        carbon = {
            c.compound.name
            for c in media.components
            if c.role is MediaComponentRole.carbon_source
        }
        opened = {
            b.source.split("[component: ")[1].rstrip("]")
            for b in bounds.bounds.values()
        }
        assert carbon & opened, f"{key}: no carbon source reached an exchange"


def test_the_ontology_sm_reaches_glucose_and_ammonium(
    model: cobra.Model, index: ExchangeIndex
) -> None:
    """The sourced SM opens the two exchanges a minimal medium exists to supply.

    ``SM_FBA`` in this module already did, but it is a modeling object with the mineral
    base bolted on; this asserts the BENCH object the loaders now emit resolves too, so
    the ontology is not handing FBA a medium nothing can grow in.
    """
    for media in (SM, SM_AGAR):
        bounds = media_to_bounds(media, model, index=index)
        opened = {b.metabolite_name for b in bounds.bounds.values()}
        assert {"D-glucose", "ammonium"} <= opened, media.name
        assert bounds.unresolved_names == ["yeast nitrogen base (w/o amino acids)"], (
            media.name
        )


def test_smith_phosphate_buffer_dissociates(
    model: cobra.Model, index: ExchangeIndex
) -> None:
    """The buffered fatty-acid plates place their buffer on potassium + phosphate."""
    from torchcell.datamodels.media import YPBO

    bounds = media_to_bounds(YPBO, model, index=index)
    opened = {b.metabolite_name for b in bounds.bounds.values()}
    assert {"potassium", "phosphate", "oleate"} <= opened
    assert "Tween 40" in bounds.unresolved_names
