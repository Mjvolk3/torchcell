# tests/torchcell/metabolism/test_media.py
"""The SGA selection media as exchange bounds: resolution and the organic-nitrogen bound.

Pure tests on a small synthetic cobra model, so they need no yeast-GEM download. The
model carries one exchange per species the recipes name; anything the recipe names that
the model lacks shows up as ``unresolved``, which is the failure these tests guard.
"""

import cobra
import pytest

from torchcell.datamodels.media import SGA_TM_SELECTION
from torchcell.metabolism.media import (
    SGA_DM_SELECTION_FBA,
    SGA_TM_SELECTION_FBA,
    SM_FBA,
    MediaBounds,
    UptakePolicy,
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
