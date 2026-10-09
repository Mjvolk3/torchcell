# tests/torchcell/datamodels/test_bacterial_morphology_features.py
# [[tests.torchcell.datamodels.test_bacterial_morphology_features]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datamodels/test_bacterial_morphology_features.py
"""The bacterial morphology feature vocabularies (issue #774).

What is asserted here is the CONTENT of the Campos 2018 vocabulary against its source,
Appendix Table S1 ("Features considered in this study and their associated symbols",
``si/si1.docx``, sha256 ``72cc3510fa63cbf625bb1cd17acebf2a4ed764be0b11acae312b7afbaec40c75``),
and the structural rules a ``MorphologyAssay`` enforces. The counts are the source's own:
Table S1 names 21 morphological, 2 growth and 5 cell cycle symbols, this assay holds the
26 non-growth ones, and 10 of the morphological ones are means against 11 coefficients of
variation (there is deliberately no mean division ratio).

The per-feature units are pinned symbol by symbol because they come from ONE place, the
release's own ``Legend normalized data`` sheet and ``Normalized data`` header row, and a
unit silently acquiring a value would be a fabricated quantity.
"""

from __future__ import annotations

import pytest

from torchcell.datamodels.bacterial_morphology_features import (
    CAMPOS2018_MORPHOLOGY_ASSAY,
    CAMPOS2018_MORPHOLOGY_FEATURES,
    MORPHOLOGY_ASSAYS,
    MorphologyAssay,
    MorphologyFeature,
    MorphologyFeatureGroup,
    MorphologyStatistic,
    morphology_assay,
)

#: Appendix Table S1's morphological block, in its own order, minus the two growth rows.
TABLE_S1_MORPHOLOGICAL: tuple[tuple[str, str], ...] = (
    ("<L>", "Mean cell length"),
    ("CV_L", "Cell length variability"),
    ("<W>", "Mean cell width"),
    ("CV_W", "Cell width variability"),
    ("<A>", "Mean cell area"),
    ("CV_A", "Cell area variability"),
    ("<V>", "Mean cell volume"),
    ("CV_V", "Cell volume variability"),
    ("<SA>", "Mean cell surface area"),
    ("CV_SA", "Cell surface area variability"),
    ("<P>", "Mean cell perimeter"),
    ("CV_P", "Cell perimeter variability"),
    ("<C>", "Mean circularity"),
    ("CV_C", "Circularity variability"),
    ("<Ar>", "Mean aspect ratio"),
    ("CV_Ar", "Aspect ratio variability"),
    ("<SA/V>", "Mean surface area-to-volume ratio"),
    ("CV_SA/V", "Surface area-to-volume ratio variability"),
    ("CV_DR", "Division ratio variability"),
    ("<NA>", "Mean nucleoid area"),
    ("CV_NA", "Nucleoid area variability"),
)

#: Appendix Table S1's cell cycle block, in its own order.
TABLE_S1_CELL_CYCLE: tuple[tuple[str, str], ...] = (
    ("rho_CD", "Correlation in nucleoid and cell constriction"),
    ("CDN_C0", "Nucleoid constriction degree at the initiation of cell constriction"),
    ("Rel.timing div", "Relative timing of cell constriction"),
    ("Rel.timing nuc", "Relative timing of nucleoid separation"),
    ("%2N", "Fraction of cells with 2 nucleoids"),
)

#: The only eight symbols the release states a unit for, as it writes it.
UNITS: dict[str, str] = {
    "<L>": "µm",
    "<W>": "µm",
    "<A>": "µm2",
    "<V>": "µm3",
    "<SA>": "µm2",
    "<P>": "µm",
    "<SA/V>": "µm-1",
    "<NA>": "µm2",
}


def test_the_campos_assay_is_table_s1_minus_its_growth_rows_in_order() -> None:
    """The vocabulary is the source's table, symbol and name, in the source's order."""
    expected = TABLE_S1_MORPHOLOGICAL + TABLE_S1_CELL_CYCLE
    assert [(f.symbol, f.name) for f in CAMPOS2018_MORPHOLOGY_FEATURES] == list(
        expected
    )
    assert len(CAMPOS2018_MORPHOLOGY_FEATURES) == 26
    assert CAMPOS2018_MORPHOLOGY_ASSAY.features == CAMPOS2018_MORPHOLOGY_FEATURES
    assert MORPHOLOGY_ASSAYS == {"campos2018": CAMPOS2018_MORPHOLOGY_ASSAY}


def test_the_growth_symbols_are_absent_because_they_are_not_morphology() -> None:
    """``alpha_max`` is served as a FitnessPhenotype and ``ODmax`` has no class."""
    symbols = {f.symbol for f in CAMPOS2018_MORPHOLOGY_FEATURES}
    assert "alpha_max" not in symbols
    assert "ODmax" not in symbols
    assert "Odmax" not in symbols


def test_the_two_released_columns_table_s1_never_names_are_absent() -> None:
    """``%non-div`` and ``%1N`` are EV2 columns outside the source's symbol table.

    Nothing is lost by it: the two relative timings in the vocabulary are exact
    deterministic transforms of those two columns (measured to 1.3e-15 on the release),
    so they are the same information under the names the source does give.
    """
    symbols = {f.symbol for f in CAMPOS2018_MORPHOLOGY_FEATURES}
    assert "%non-div" not in symbols
    assert "%1N" not in symbols


def test_the_group_split_is_table_s1_s_own_headings() -> None:
    """21 under "Morphological features" and 5 under "Cell cycle features"."""
    assay = CAMPOS2018_MORPHOLOGY_ASSAY
    morphological = assay.symbols_of_group(MorphologyFeatureGroup.morphological)
    cell_cycle = assay.symbols_of_group(MorphologyFeatureGroup.cell_cycle)
    assert morphological == frozenset(s for s, _ in TABLE_S1_MORPHOLOGICAL)
    assert cell_cycle == frozenset(s for s, _ in TABLE_S1_CELL_CYCLE)
    assert (len(morphological), len(cell_cycle)) == (21, 5)
    assert not morphological & cell_cycle


def test_ten_means_against_eleven_coefficients_of_variation() -> None:
    """The asymmetry is the source's: the division ratio contributes only its CV.

    "measurements of mean division ratio were meaningless and not included in our
    analysis. However, the CV of the division ratio was included", so there is no
    ``<DR>`` to pair with ``CV_DR``.
    """
    by_statistic: dict[MorphologyStatistic, list[str]] = {}
    for feature in CAMPOS2018_MORPHOLOGY_FEATURES:
        by_statistic.setdefault(feature.statistic, []).append(feature.symbol)
    assert {k: len(v) for k, v in by_statistic.items()} == {
        MorphologyStatistic.mean: 10,
        MorphologyStatistic.coefficient_of_variation: 11,
        MorphologyStatistic.pearson_correlation: 1,
        MorphologyStatistic.regression_intercept: 1,
        MorphologyStatistic.inferred_relative_timing: 2,
        MorphologyStatistic.fraction_of_cells: 1,
    }
    assert "<DR>" not in {f.symbol for f in CAMPOS2018_MORPHOLOGY_FEATURES}
    assert by_statistic[MorphologyStatistic.pearson_correlation] == ["rho_CD"]
    assert by_statistic[MorphologyStatistic.regression_intercept] == ["CDN_C0"]
    assert by_statistic[MorphologyStatistic.fraction_of_cells] == ["%2N"]
    assert by_statistic[MorphologyStatistic.inferred_relative_timing] == [
        "Rel.timing div",
        "Rel.timing nuc",
    ]


def test_the_two_dict_partitions_are_the_statistic_declaration() -> None:
    """``value_symbols`` and the CV symbols partition the vocabulary, 15 against 11."""
    assay = CAMPOS2018_MORPHOLOGY_ASSAY
    values = assay.value_symbols
    coefficients = assay.coefficient_of_variation_symbols
    assert len(values) == 15
    assert len(coefficients) == 11
    assert not values & coefficients
    assert values | coefficients == frozenset(assay.by_symbol)
    assert all(symbol.startswith("CV_") for symbol in coefficients)


def test_only_the_eight_dimensional_symbols_carry_a_unit() -> None:
    """Every unit is the release's own string; the other 18 features carry None."""
    units = {
        f.symbol: f.unit for f in CAMPOS2018_MORPHOLOGY_FEATURES if f.unit is not None
    }
    assert units == UNITS
    assert all(
        f.unit is None
        for f in CAMPOS2018_MORPHOLOGY_FEATURES
        if f.statistic is not MorphologyStatistic.mean
    )
    assert CAMPOS2018_MORPHOLOGY_ASSAY.by_symbol["<C>"].unit is None
    assert CAMPOS2018_MORPHOLOGY_ASSAY.by_symbol["<Ar>"].unit is None


def test_the_features_whose_cell_subset_differs_say_so() -> None:
    """A feature over a subset of the strain's cells records the subset in its note."""
    by_symbol = CAMPOS2018_MORPHOLOGY_ASSAY.by_symbol
    assert "CONSTRICTED cells only" in (by_symbol["CV_DR"].note or "")
    assert "constriction degree (above 0.15) only" in (by_symbol["rho_CD"].note or "")
    assert "no unit" in (by_symbol["CDN_C0"].note or "")


def test_morphology_assay_resolves_a_registered_name_and_names_the_rest() -> None:
    assert morphology_assay("campos2018") is CAMPOS2018_MORPHOLOGY_ASSAY
    with pytest.raises(
        KeyError, match=r"unknown bacterial morphology assay 'ohya2005'"
    ):
        morphology_assay("ohya2005")
    with pytest.raises(KeyError, match=r"registered assays are \['campos2018'\]"):
        morphology_assay("ohya2005")


def test_an_assay_refuses_to_be_empty_or_to_repeat_a_symbol() -> None:
    feature = MorphologyFeature(
        symbol="<L>",
        name="Mean cell length",
        group=MorphologyFeatureGroup.morphological,
        statistic=MorphologyStatistic.mean,
    )
    with pytest.raises(ValueError, match="declares no features"):
        MorphologyAssay(name="empty", description="none", features=())
    with pytest.raises(ValueError, match=r"repeats symbols \['<L>'\]"):
        MorphologyAssay(
            name="twice", description="two", features=(feature, feature.model_copy())
        )


def test_a_feature_refuses_an_empty_symbol_or_name() -> None:
    group = MorphologyFeatureGroup.morphological
    statistic = MorphologyStatistic.mean
    with pytest.raises(ValueError, match="needs a symbol and a name"):
        MorphologyFeature(
            symbol="", name="Mean cell length", group=group, statistic=statistic
        )
    with pytest.raises(ValueError, match="needs a symbol and a name"):
        MorphologyFeature(symbol="<L>", name="", group=group, statistic=statistic)


def test_a_feature_and_an_assay_are_frozen_and_reject_unknown_fields() -> None:
    """The vocabulary is a sourced record, so it is not mutated after construction."""
    feature = CAMPOS2018_MORPHOLOGY_FEATURES[0]
    with pytest.raises(ValueError, match="frozen"):
        feature.unit = "nm"
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        MorphologyFeature(
            symbol="<L>",
            name="Mean cell length",
            group=MorphologyFeatureGroup.morphological,
            statistic=MorphologyStatistic.mean,
            source="invented",  # type: ignore[call-arg]
        )
