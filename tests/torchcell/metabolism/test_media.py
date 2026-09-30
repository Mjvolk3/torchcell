# tests/torchcell/metabolism/test_media.py
"""The SGA selection media as exchange bounds: resolution and the organic-nitrogen bound.

Pure tests on a small synthetic cobra model, so they need no yeast-GEM download. The
model carries one exchange per species the recipes name; anything the recipe names that
the model lacks shows up as ``unresolved``, which is the failure these tests guard.

2026.09.30 (Phase 16). Added fixtures: a four-exchange cobra model (``_tiny_model``) whose
metabolites carry a ChEBI CURIE list, a bare numeric ChEBI id, a name that normalizes to
an existing one, and a plain name, every exchange starting at ``lower_bound = -7``; and a
duck-typed model with a two-metabolite "exchange". Expected values, derived from the
source: a CURIE ``CHEBI:4167`` normalizes to the index key ``chebi:chebi:4167`` and the
bare form ``4167`` to ``chebi:4167`` (``_annotation_keys_for`` tries both); a salt in
``_DISSOCIATION`` records every ion candidate it tried, in ion order; ``apply`` sets every
exchange lower bound to 0 and then ``-uptake_bound`` for each bound (3.3 for the carbon
source, 0.165 for a nucleobase); when two components land on one exchange the larger
magnitude wins in either order. Recipe relations, pinned through ``diff_bounds`` on the
62-species model: SC minus SC-Ura is exactly the uracil exchange at 0.165, SC minus
YPD-approx is exactly adenine and uracil at 0.165 each (the YPD-approx docstring's
"identical to SC minus its two nucleobases"), SGA DM minus SGA TM is exactly uracil. Recipe
sizes: 14 mineral-base species + glucose + ammonium sulfate + 9 vitamins = 25 for SM; + 20
amino acids + 2 nucleobases = 47 for SC; 46 for SC-Ura; 45 for YPD-approx.
"""

from types import SimpleNamespace
from typing import Any

import cobra
import pytest

from torchcell.datamodels.media import (
    CARBON_FREE_MEDIA,
    MEDIA_LIBRARY,
    SGA_TM_SELECTION,
    SM,
    SM_AGAR,
)
from torchcell.datamodels.schema import (
    ComponentDefinition,
    Compound,
    Media,
    MediaComponent,
    MediaComponentRole,
)
from torchcell.metabolism.media import (
    SC_FBA,
    SC_URA_FBA,
    SGA_DM_SELECTION_FBA,
    SGA_TM_SELECTION_FBA,
    SM_FBA,
    YPD_APPROX_FBA,
    ExchangeIndex,
    MediaBounds,
    UptakePolicy,
    _candidate_names,
    _normalize,
    build_exchange_index,
    diff_bounds,
    media_to_bounds,
    resolve_component,
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


# --------------------------------------------------------------------------- #
# 2026.09.30 (Phase 16): resolution channels, apply, diff, and the recipe relations.
# --------------------------------------------------------------------------- #


def _tiny_model() -> cobra.Model:
    """Four exchanges, all opened to -7 so ``apply`` has something to close."""
    specs: list[tuple[str, dict[str, Any]]] = [
        ("glucose-x", {"chebi": ["CHEBI:17634", "CHEBI:4167"]}),
        ("sodium", {"chebi": "29101"}),
        ("Sodium ", {}),
        ("uracil", {}),
    ]
    tiny = cobra.Model("tiny")
    for i, (name, annotation) in enumerate(specs):
        met = cobra.Metabolite(f"m{i}", name=name, compartment="e")
        met.annotation = annotation
        rxn = cobra.Reaction(f"EX_{i}", lower_bound=-7.0, upper_bound=1000.0)
        rxn.add_metabolites({met: -1.0})
        tiny.add_reactions([rxn])
    return tiny


def _component(
    name: str,
    role: MediaComponentRole,
    chebi_id: str | None = None,
    definition: ComponentDefinition = ComponentDefinition.defined,
) -> MediaComponent:
    return MediaComponent(
        compound=Compound(name=name, chebi_id=chebi_id),
        role=role,
        definition=definition,
    )


def _named(diff: dict[str, float], index: ExchangeIndex) -> dict[str, float]:
    return {index.metabolite_of[ex][1]: value for ex, value in diff.items()}


def test_exchange_index_annotation_lists_bare_ids_and_first_name_wins() -> None:
    """A list annotation indexes every entry; a normalized name collision keeps the first.

    ``"Sodium "`` normalizes to ``"sodium"``, already claimed by EX_1, so ``by_name``
    maps it to EX_1 while ``metabolite_of`` still records EX_2 verbatim.
    """
    index = build_exchange_index(_tiny_model())
    assert index.model_id == "tiny"
    assert index.by_annotation == {
        "chebi:chebi:17634": "EX_0",
        "chebi:chebi:4167": "EX_0",
        "chebi:29101": "EX_1",
    }
    assert index.by_name == {"glucose-x": "EX_0", "sodium": "EX_1", "uracil": "EX_3"}
    assert index.metabolite_of["EX_2"] == ("m2", "Sodium ")


def test_exchange_index_skips_a_multi_metabolite_exchange() -> None:
    """An 'exchange' with two metabolites names no single species and is not indexed.

    cobra's own ``model.exchanges`` never yields one (a boundary reaction has one
    metabolite), so a duck-typed model stands in; only ``.exchanges``, ``.id`` and the
    metabolites' ``id``/``name``/``annotation`` are read.
    """
    single = SimpleNamespace(id="m_a", name="A", annotation={})
    pair = [
        SimpleNamespace(id="m_b", name="B", annotation={}),
        SimpleNamespace(id="m_c", name="C", annotation={}),
    ]
    fake: Any = SimpleNamespace(
        id="fake",
        exchanges=[
            SimpleNamespace(id="EX_a", metabolites=[single]),
            SimpleNamespace(id="EX_bc", metabolites=pair),
        ],
    )
    index = build_exchange_index(fake)
    assert index.metabolite_of == {"EX_a": ("m_a", "A")}
    assert index.by_name == {"a": "EX_a"}


def test_resolve_by_chebi_curie_and_by_bare_numeric_id() -> None:
    """The annotation channel wins before any name is tried; both key forms match."""
    index = build_exchange_index(_tiny_model())
    curie = resolve_component(
        _component("bench sugar", MediaComponentRole.carbon_source, "CHEBI:4167"), index
    )
    assert (curie.outcome, curie.exchange_ids, curie.match_channel) == (
        "resolved",
        ["EX_0"],
        "annotation:chebi",
    )
    assert curie.candidates_tried == []
    bare = resolve_component(
        _component("table salt ion", MediaComponentRole.bulk_salt, "CHEBI:29101"), index
    )
    assert (bare.outcome, bare.exchange_ids, bare.match_channel) == (
        "resolved",
        ["EX_1"],
        "annotation:chebi",
    )


def test_dissociation_partial_match_and_total_miss() -> None:
    """Sodium chloride keeps the sodium it found; ammonium sulfate finds neither ion.

    Candidates per ion are ``[ion, l-ion]``, plus the ``sulphate`` synonym for sulfate,
    in the order ``_candidate_names`` emits them.
    """
    index = build_exchange_index(_tiny_model())
    partial = resolve_component(
        _component("sodium chloride", MediaComponentRole.bulk_salt), index
    )
    assert partial.outcome == "resolved"
    assert partial.exchange_ids == ["EX_1"]
    assert partial.match_channel == "name:dissociation(sodium chloride)"
    assert partial.candidates_tried == ["sodium", "l-sodium", "chloride", "l-chloride"]
    miss = resolve_component(
        _component("ammonium sulfate", MediaComponentRole.nitrogen_source), index
    )
    assert miss.outcome == "unresolved"
    assert miss.reason == "salt dissociates but no ion matched a model exchange"
    assert miss.candidates_tried == [
        "ammonium",
        "l-ammonium",
        "sulfate",
        "sulphate",
        "l-sulfate",
        "l-sulphate",
    ]


def test_unresolved_mixture_reason_names_its_definition() -> None:
    """A composition-deferred miss says it names a mixture; a defined miss does not."""
    index = build_exchange_index(_tiny_model())
    mixture = resolve_component(
        _component(
            "yeast nitrogen base",
            MediaComponentRole.other,
            definition=ComponentDefinition.composition_deferred,
        ),
        index,
    )
    assert mixture.reason == (
        "no model exchange matched by annotation or by any chemical name variant; "
        "component identity is 'composition_deferred' so it names a mixture rather "
        "than a single species"
    )
    assert mixture.candidates_tried == ["yeast nitrogen base", "l-yeast nitrogen base"]
    defined = resolve_component(_component("zz", MediaComponentRole.other), index)
    assert defined.reason == (
        "no model exchange matched by annotation or by any chemical name variant"
    )


def test_media_bounds_counts_and_apply_closes_then_opens() -> None:
    """Four components: 2 resolved, 1 excluded (agar), 1 unresolved (zz).

    ``apply`` closes all four exchanges (the preset -7 goes to 0) and opens EX_0 to
    -3.3 (carbon) and EX_3 to -0.165 (nucleobase); leaving ``with`` restores -7.
    """
    tiny = _tiny_model()
    media = Media(
        name="tiny medium",
        state="liquid",
        is_synthetic=True,
        components=[
            _component("bench sugar", MediaComponentRole.carbon_source, "CHEBI:4167"),
            _component("uracil", MediaComponentRole.nucleobase),
            _component("agar", MediaComponentRole.gelling_agent),
            _component("zz", MediaComponentRole.other),
        ],
    )
    bounds = media_to_bounds(media, tiny)
    assert (bounds.n_components, bounds.n_resolved) == (4, 2)
    assert bounds.unresolved_names == ["zz"]
    assert bounds.excluded_names == ["agar"]
    assert bounds.bounds["EX_0"].source == (
        "suthersGenomescaleMetabolicReconstruction2020 sec2.5 default carbon substrate "
        "uptake 3.3 mmol/gDW/h [component: bench sugar]"
    )
    with tiny:
        bounds.apply(tiny)
        applied = {r.id: r.lower_bound for r in tiny.reactions}
    assert applied == {"EX_0": -3.3, "EX_1": 0.0, "EX_2": 0.0, "EX_3": -0.165}
    assert {r.id: r.lower_bound for r in tiny.reactions} == {
        "EX_0": -7.0,
        "EX_1": -7.0,
        "EX_2": -7.0,
        "EX_3": -7.0,
    }


@pytest.mark.parametrize("excess_first", [True, False])
def test_shared_exchange_keeps_the_larger_uptake(excess_first: bool) -> None:
    """Uracil as a nucleobase (0.165) and as an excess species (1000): 1000 wins either way.

    The source string names the component that set the winning magnitude.
    """
    supplement = _component("uracil", MediaComponentRole.nucleobase)
    excess = _component("Uracil", MediaComponentRole.other)
    order = [excess, supplement] if excess_first else [supplement, excess]
    media = Media(name="m", state="liquid", is_synthetic=True, components=order)
    bound = media_to_bounds(media, _tiny_model()).bounds["EX_3"]
    assert bound.uptake_bound == 1000.0
    assert bound.source == (
        "cobra convention: species assumed in excess, opened to 1000.0 mmol/gDW/h "
        "[component: Uracil]"
    )


def test_supplement_rate_does_not_follow_carbon_uptake() -> None:
    """Raising carbon to the OptKnock 10.0 leaves supplements and MSG at 0.165.

    Ammonium sulfate is inorganic nitrogen and opens at 1000; only the names in
    ``_ORGANIC_NITROGEN_SOURCES`` (after normalization) are held at 0.165.
    """
    policy = UptakePolicy(carbon_uptake=10.0)
    role = MediaComponentRole
    assert policy.bound_for(role.carbon_source) == 10.0
    assert policy.bound_for(role.amino_acid) == 0.165
    assert policy.bound_for(role.vitamin) == 0.165
    assert policy.bound_for(role.nitrogen_source, "  Monosodium  Glutamate ") == 0.165
    assert policy.bound_for(role.nitrogen_source, "ammonium sulfate") == 1000.0
    assert policy.source_for(role.carbon_source) == (
        "suthersGenomescaleMetabolicReconstruction2020 sec2.5 default carbon substrate "
        "uptake 10.0 mmol/gDW/h"
    )
    assert policy.source_for(role.nitrogen_source, "L-glutamine") == (
        "torchcell convention: amino-acid nitrogen source bounded at the "
        "suthersGenomescaleMetabolicReconstruction2020 supplement rate 0.165 "
        "mmol/gDW/h, not opened, so it cannot become the carbon source"
    )


def test_candidate_names_rules_in_order() -> None:
    """Literal first, then synonym, then salt-stripped, then acid-to-base and ``l-``.

    ``niacin hydrate``: strip ``hydrate`` to ``niacin``, whose synonym is ``nicotinate``.
    ``L-aspartic acid``: ``ic acid -> ate`` gives ``l-aspartate``; the generic
    `` acid -> ate`` rule also fires and yields the non-chemical ``l-asparticate``, a
    harmless extra candidate. A duplicate candidate is kept once.
    """
    assert _candidate_names("niacin hydrate") == [
        "niacin hydrate",
        "niacin",
        "nicotinate",
        "l-niacin hydrate",
        "l-niacin",
        "l-nicotinate",
    ]
    assert _candidate_names("L-aspartic acid") == [
        "l-aspartic acid",
        "l-aspartate",
        "l-asparticate",
    ]
    assert _candidate_names("Glycine") == ["glycine", "l-glycine"]


def test_candidate_names_duplicate_is_kept_once() -> None:
    """``nicotinic acid``: the synonym and the ``ic acid -> ate`` rule both give
    ``nicotinate``, which appears once, at the synonym's position.
    """
    assert _candidate_names("nicotinic acid") == [
        "nicotinic acid",
        "nicotinate",
        "nicotinicate",
        "l-nicotinic acid",
        "l-nicotinate",
    ]


def test_hydrate_is_stripped_before_monohydrate_and_dihydrate() -> None:
    """Finding: ``hydrate`` precedes ``monohydrate``/``dihydrate`` in ``_SALT_QUALIFIERS``.

    The replace loop (media.py lines 339 to 340) removes the ``hydrate`` substring
    first, leaving ``mono`` / ``di`` behind, so ``L-cysteine hydrochloride monohydrate``
    never yields the candidate ``l-cysteine`` and the two longer qualifiers never
    match. Pinned until the longer qualifiers are stripped first.
    """
    assert _candidate_names("L-cysteine hydrochloride monohydrate") == [
        "l-cysteine hydrochloride monohydrate",
        "l-cysteine mono",
    ]
    assert _candidate_names("calcium chloride dihydrate")[1] == "calcium chloride di"


def test_normalize_keeps_punctuation_despite_its_docstring() -> None:
    """Finding: ``_normalize`` claims to "drop punctuation a model never carries".

    It lowercases, strips, collapses whitespace and turns a curly apostrophe into a
    straight one; no punctuation is dropped (media.py lines 316 to 321). Pinned until
    the docstring or the function changes.
    """
    assert _normalize("  L-Glu\u2019s,   Acid. ") == "l-glu's, acid."


def test_diff_bounds_sc_minus_sc_ura_is_exactly_uracil(
    model: cobra.Model, index: ExchangeIndex
) -> None:
    """The dropout withholds uracil and nothing else; the reverse side is empty."""
    sc = media_to_bounds(SC_FBA, model, index=index)
    sc_ura = media_to_bounds(SC_URA_FBA, model, index=index)
    diff = diff_bounds(sc, sc_ura)
    assert (diff.left, diff.right) == (SC_FBA.name, SC_URA_FBA.name)
    assert _named(diff.only_in_left, index) == {"uracil": 0.165}
    assert (diff.only_in_right, diff.differing, diff.n_differences) == ({}, {}, 1)
    assert [c.name for c in SC_URA_FBA.dropouts] == ["uracil"]
    assert len(SC_FBA.components) - len(SC_URA_FBA.components) == 1


def test_ypd_approx_is_sc_without_its_two_nucleobases(
    model: cobra.Model, index: ExchangeIndex
) -> None:
    """The YPD-approx docstring's claim, checked on the bound vector."""
    diff = diff_bounds(
        media_to_bounds(SC_FBA, model, index=index),
        media_to_bounds(YPD_APPROX_FBA, model, index=index),
    )
    assert _named(diff.only_in_left, index) == {"adenine": 0.165, "uracil": 0.165}
    assert (diff.only_in_right, diff.differing) == ({}, {})
    assert YPD_APPROX_FBA.is_synthetic is True
    assert YPD_APPROX_FBA.base_medium == "YPD_approx"


def test_sga_dm_minus_sga_tm_is_exactly_uracil(
    model: cobra.Model, index: ExchangeIndex
) -> None:
    """The trigenic medium adds one dropout (uracil) to the digenic one."""
    dm = media_to_bounds(SGA_DM_SELECTION_FBA, model, index=index)
    tm = media_to_bounds(SGA_TM_SELECTION_FBA, model, index=index)
    diff = diff_bounds(dm, tm)
    assert _named(diff.only_in_left, index) == {"uracil": 0.165}
    assert (diff.only_in_right, diff.differing) == ({}, {})
    assert [d.name for d in SGA_DM_SELECTION_FBA.dropouts] == [
        "L-histidine",
        "L-arginine",
        "L-lysine",
    ]


def test_diff_bounds_differing_magnitude() -> None:
    """Same medium under two carbon policies differs only on the carbon exchange."""
    tiny = _tiny_model()
    media = Media(
        name="sugar",
        state="liquid",
        is_synthetic=True,
        components=[_component("glucose-x", MediaComponentRole.carbon_source)],
    )
    diff = diff_bounds(
        media_to_bounds(media, tiny),
        media_to_bounds(media, tiny, policy=UptakePolicy(carbon_uptake=10.0)),
    )
    assert diff.differing == {"EX_0": (3.3, 10.0)}
    assert diff.n_differences == 1


def test_the_sga_expansion_replaces_ynb_and_drops_the_selection_amino_acids() -> None:
    """The SGA DM expansion replaces YNB with its vitamins and the supplement powder
    with the amino acids minus His, Arg and Lys (the selection markers), so no component
    is named after yeast nitrogen base and the three markers are absent; the TM
    expansion is the DM expansion minus exactly uracil and adds nothing; the expanded
    recipe carries the " [FBA expansion]" suffix.
    """
    dm_names = [c.compound.name for c in SGA_DM_SELECTION_FBA.components]
    assert not any(n.startswith("yeast nitrogen base") for n in dm_names)
    assert {"L-histidine", "L-arginine", "L-lysine"}.isdisjoint(dm_names)
    tm_names = [c.compound.name for c in SGA_TM_SELECTION_FBA.components]
    assert sorted(set(dm_names) - set(tm_names)) == ["uracil"]
    assert set(tm_names) - set(dm_names) == set()
    assert SGA_DM_SELECTION_FBA.name.endswith(" [FBA expansion]")
