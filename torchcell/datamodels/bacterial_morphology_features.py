# torchcell/datamodels/bacterial_morphology_features.py
# [[torchcell.datamodels.bacterial-morphology-features]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/bacterial_morphology_features.py
# Test file: tests/torchcell/datamodels/test_bacterial_morphology_features.py
"""Feature vocabularies for bacterial cell-morphology assays.

``CalMorphPhenotype`` carries its vocabulary as two module-level frozensets of the 281
base and 220 CV parameters a yeast imaging program emits
(:mod:`torchcell.datamodels.calmorph_labels`), and its two validators reject anything
outside them. That shape is right and that vocabulary is not transferable: CalMorph is a
*S. cerevisiae* image-analysis program, so a bacterial screen's own symbols are not
CalMorph parameters and storing them under CalMorph keys would assert a measurement
nobody made.

This module keeps the shape and moves the vocabulary one level out: the permitted label
set is a property of the ASSAY, declared here as a typed :class:`MorphologyAssay`, and
``BacterialMorphologyPhenotype.assay`` names which assay a record was measured by. A
later bacterial imaging screen adds an assay here; it does not widen anyone else's label
set, and a record measured by one assay can never be read under another's vocabulary.

Each :class:`MorphologyFeature` carries four things the source states and nothing it does
not: the key a record uses, the source's own name for the feature, WHAT STATISTIC the
number is, and its unit as the source writes it (``None`` when the source writes none).
The statistic is not decoration. Campos 2018's 26 morphology symbols are six different
statistics (a mean over cells, a coefficient of variation over them, a Pearson
correlation across them, a regression intercept fitted to them, a population fraction,
and a cell age inferred from such a fraction), and averaging a correlation with a
relative timing is meaningless.

VOCABULARY SOURCE, and why it is Appendix Table S1 rather than the main text. Campos's
main text counts "19 morphological features", while Appendix Table S1 ("Features
considered in this study and their associated symbols") names 21 morphological symbols,
the difference being the mean and variability of nucleoid area, which the text leaves out
of its headline count and Fig 8A treats as its own category. Table S1 is the release's
naming authority (the main text points at it: "The name and abbreviation for all the
features can be found in Appendix Table S1"), it names every released feature column, and
it is what the loader's keys are checked against. Table S1 also lists two GROWTH symbols,
``alpha_max`` and ``ODmax``, which are plate-reader population measurements rather than
morphology and are therefore not in this vocabulary (``alpha_max`` is served as the
``FitnessPhenotype`` of ``GrowthRateCampos2018Dataset``; ``ODmax`` has no phenotype class,
recorded as a separate gap). So Table S1's 28 symbols are 26 morphology plus 2 growth,
and the 26 here are exactly the morphology ones.

KEY SPELLING. The key is the Dataset EV2 column symbol, not Table S1's typeset symbol.
Table S1 writes the coefficients of variation with a subscript (``CV`` plus a subscripted
``L``) and the rho with a Symbol-font glyph, so reading the ``.docx`` text flattens them
to ``CVL`` and ``r CD``; the released data columns write the same features as ``CV_L`` and
``rho_CD``. The machine-readable form is the one a loader and a stored record can round
trip, so it is the key, and Table S1's own text is kept verbatim in
:attr:`MorphologyFeature.name`.

UNIT SOURCE. Appendix Table S1 has no unit column and ``paper.md`` states no feature
unit anywhere. The units come from the release itself: Dataset EV2's ``Legend normalized
data`` sheet names each feature with its unit (``Mean cell length (µm)``), and the
``Normalized data`` header row carries the same parenthetical (``<L> (µm)``). Eight of the
26 have a unit there and the other 18 have none, which is the honest record: a coefficient
of variation, a correlation, a shape factor, a relative timing and a population fraction
are all dimensionless, and nothing is invented for them.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MorphologyFeatureGroup(StrEnum):
    """The group a feature belongs to, as its source's own feature table groups them.

    Campos 2018's Appendix Table S1 carries three group headings, "Morphological
    features", "Growth features" and "Cell cycle features". The growth features are
    plate-reader population measurements and are not morphology, so only the other two
    are groups a morphology assay can declare.

    - ``morphological``: a cell-shape or nucleoid-size feature of the imaged cells.
    - ``cell_cycle``: a cell-cycle feature derived from the same images (a constriction
      correlation, a nucleoid-constriction intercept, an inferred relative timing, or the
      fraction of cells in a nucleoid state).
    """

    morphological = "morphological"
    cell_cycle = "cell_cycle"


class MorphologyStatistic(StrEnum):
    """What one released morphology number IS, so two of them are never pooled.

    A per-strain morphology profile is not a vector of like quantities. Campos 2018
    releases a mean, a coefficient of variation, a Pearson correlation, a fitted
    intercept, an inferred cell age and a population fraction side by side in one row,
    and only the mean/CV pair shares a scale. Naming the statistic is what lets a
    consumer refuse to average across them.

    - ``mean``: the arithmetic mean of a per-cell quantity over the strain's cells.
    - ``coefficient_of_variation``: the standard deviation of that per-cell quantity
      divided by its mean, over the same cells. Non-negative by construction.
    - ``pearson_correlation``: a Pearson correlation coefficient between two per-cell
      quantities, computed across the strain's cells. Bounded in [-1, 1].
    - ``regression_intercept``: the intercept of a line fitted to the strain's
      single-cell data, the slope being set by a correlation coefficient.
    - ``fraction_of_cells``: the proportion of the strain's scored cells in a state.
      Bounded in [0, 1] up to the source's own bias correction.
    - ``inferred_relative_timing``: a population-level cell age at which a cell-cycle
      event occurs, inferred from such a fraction under a steady-state assumption, in
      relative cell-cycle units rather than any clock unit.
    """

    mean = "mean"
    coefficient_of_variation = "coefficient_of_variation"
    pearson_correlation = "pearson_correlation"
    regression_intercept = "regression_intercept"
    fraction_of_cells = "fraction_of_cells"
    inferred_relative_timing = "inferred_relative_timing"


class MorphologyFeature(BaseModel):
    """One named feature of a bacterial cell-morphology assay.

    ``symbol`` is the dictionary key a stored record uses, so it is the release's own
    machine-readable column symbol. ``name`` and ``unit`` are the source's own words,
    verbatim; ``unit`` is ``None`` when the source writes none, never a guess.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str = Field(
        description="the record's dictionary key: the release's own column symbol"
    )
    name: str = Field(description="the source's own name for the feature, verbatim")
    group: MorphologyFeatureGroup
    statistic: MorphologyStatistic
    unit: str | None = Field(
        default=None,
        description="the unit as the source writes it; None when the source writes none",
    )
    note: str | None = Field(
        default=None,
        description="what the source states about this feature beyond its name: the "
        "subset of cells it is computed over, a definition, or a conflict in the "
        "release's own legends",
    )

    @field_validator("symbol", "name")
    @classmethod
    def non_empty(cls, value: str) -> str:
        """A feature with no symbol or no name cannot be checked against a source."""
        if not value:
            raise ValueError("a morphology feature needs a symbol and a name")
        return value


class MorphologyAssay(BaseModel):
    """The feature vocabulary of one bacterial cell-morphology assay.

    This is the object ``BacterialMorphologyPhenotype`` validates against: the permitted
    label set is a property of the assay, so adding an assay cannot widen another one's
    vocabulary.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="the assay key a phenotype record names")
    description: str = Field(
        description="what was imaged, with what software, and where the vocabulary "
        "comes from"
    )
    features: tuple[MorphologyFeature, ...]

    @model_validator(mode="after")
    def validate_features(self) -> MorphologyAssay:
        """An assay needs features, and its symbols must be distinct."""
        if not self.features:
            raise ValueError(f"morphology assay {self.name!r} declares no features")
        symbols = [feature.symbol for feature in self.features]
        duplicates = sorted({s for s in symbols if symbols.count(s) > 1})
        if duplicates:
            raise ValueError(
                f"morphology assay {self.name!r} repeats symbols {duplicates}"
            )
        return self

    @property
    def by_symbol(self) -> dict[str, MorphologyFeature]:
        """Every feature of this assay, keyed by the symbol a record uses."""
        return {feature.symbol: feature for feature in self.features}

    @property
    def value_symbols(self) -> frozenset[str]:
        """Symbols belonging in ``morphology``: every statistic but the CVs."""
        return frozenset(
            feature.symbol
            for feature in self.features
            if feature.statistic is not MorphologyStatistic.coefficient_of_variation
        )

    @property
    def coefficient_of_variation_symbols(self) -> frozenset[str]:
        """Symbols belonging in ``morphology_coefficient_of_variation``."""
        return frozenset(
            feature.symbol
            for feature in self.features
            if feature.statistic is MorphologyStatistic.coefficient_of_variation
        )

    def symbols_of_group(self, group: MorphologyFeatureGroup) -> frozenset[str]:
        """The symbols this assay assigns to one group."""
        return frozenset(
            feature.symbol for feature in self.features if feature.group is group
        )


# --------------------------------------------------------------------------- #
# Campos 2018: the imaged Keio collection
# --------------------------------------------------------------------------- #
_MORPH = MorphologyFeatureGroup.morphological
_CYCLE = MorphologyFeatureGroup.cell_cycle
_MEAN = MorphologyStatistic.mean
_CV = MorphologyStatistic.coefficient_of_variation

#: Every coefficient of variation of the Campos assay is this statistic, stated once:
#: "We also measured the variability of these features by calculating their coefficient
#: of variation (CV, the standard deviation divided by the mean)."
_CV_NOTE = (
    "the standard deviation of the per-cell quantity divided by its mean, over the "
    "strain's imaged cells"
)

#: The 26 morphology symbols of Appendix Table S1, in Table S1's own order. The two
#: growth symbols Table S1 also lists, ``alpha_max`` ("Max growth rate") and ``ODmax``
#: ("Optical density at growth saturation"), are plate-reader population measurements
#: and are deliberately absent.
CAMPOS2018_MORPHOLOGY_FEATURES: tuple[MorphologyFeature, ...] = (
    MorphologyFeature(
        symbol="<L>", name="Mean cell length", group=_MORPH, statistic=_MEAN, unit="µm"
    ),
    MorphologyFeature(
        symbol="CV_L",
        name="Cell length variability",
        group=_MORPH,
        statistic=_CV,
        note=_CV_NOTE,
    ),
    MorphologyFeature(
        symbol="<W>", name="Mean cell width", group=_MORPH, statistic=_MEAN, unit="µm"
    ),
    MorphologyFeature(
        symbol="CV_W",
        name="Cell width variability",
        group=_MORPH,
        statistic=_CV,
        note=_CV_NOTE,
    ),
    MorphologyFeature(
        symbol="<A>",
        name="Mean cell area",
        group=_MORPH,
        statistic=_MEAN,
        unit="µm2",
        note="the main text calls it cross-sectional area",
    ),
    MorphologyFeature(
        symbol="CV_A",
        name="Cell area variability",
        group=_MORPH,
        statistic=_CV,
        note=_CV_NOTE,
    ),
    MorphologyFeature(
        symbol="<V>",
        name="Mean cell volume",
        group=_MORPH,
        statistic=_MEAN,
        unit="µm3",
        note="a quantity derived from the length and width series; the paper says only "
        "that it extracted the mean and CV of it, and does not state whether the "
        "derivation runs per cell before averaging or on the averaged dimensions",
    ),
    MorphologyFeature(
        symbol="CV_V",
        name="Cell volume variability",
        group=_MORPH,
        statistic=_CV,
        note=_CV_NOTE,
    ),
    MorphologyFeature(
        symbol="<SA>",
        name="Mean cell surface area",
        group=_MORPH,
        statistic=_MEAN,
        unit="µm2",
        note="derived from the length and width series, with the same open question as "
        "the mean cell volume",
    ),
    MorphologyFeature(
        symbol="CV_SA",
        name="Cell surface area variability",
        group=_MORPH,
        statistic=_CV,
        note=_CV_NOTE,
    ),
    MorphologyFeature(
        symbol="<P>",
        name="Mean cell perimeter",
        group=_MORPH,
        statistic=_MEAN,
        unit="µm",
    ),
    MorphologyFeature(
        symbol="CV_P",
        name="Cell perimeter variability",
        group=_MORPH,
        statistic=_CV,
        note=_CV_NOTE,
    ),
    MorphologyFeature(
        symbol="<C>",
        name="Mean circularity",
        group=_MORPH,
        statistic=_MEAN,
        note="Appendix Table S1's caption defines it as 4*pi*A/P^2 at the single-cell "
        "level, where P stands for perimeter and A for area, so it is a dimensionless "
        "shape factor and the release states no unit for it",
    ),
    MorphologyFeature(
        symbol="CV_C",
        name="Circularity variability",
        group=_MORPH,
        statistic=_CV,
        note=_CV_NOTE,
    ),
    MorphologyFeature(
        symbol="<Ar>",
        name="Mean aspect ratio",
        group=_MORPH,
        statistic=_MEAN,
        note="Appendix Table S1's caption defines it as the ratio of cell width over "
        "cell length at the single-cell level, so a rod scores below 1 (the released "
        "range is 0.169 to 0.571) and the release states no unit",
    ),
    MorphologyFeature(
        symbol="CV_Ar",
        name="Aspect ratio variability",
        group=_MORPH,
        statistic=_CV,
        note=_CV_NOTE,
    ),
    MorphologyFeature(
        symbol="<SA/V>",
        name="Mean surface area-to-volume ratio",
        group=_MORPH,
        statistic=_MEAN,
        unit="µm-1",
        note="derived from the length and width series, with the same open question as "
        "the mean cell volume",
    ),
    MorphologyFeature(
        symbol="CV_SA/V",
        name="Surface area-to-volume ratio variability",
        group=_MORPH,
        statistic=_CV,
        note=_CV_NOTE,
    ),
    MorphologyFeature(
        symbol="CV_DR",
        name="Division ratio variability",
        group=_MORPH,
        statistic=_CV,
        note="the CV of the relative position of division along the cell length, over "
        "the strain's CONSTRICTED cells only. There is deliberately no mean division "
        "ratio: pole identity was unknown, so randomization would produce a mean of 0.5 "
        "even for an off-center division and the mean was judged meaningless",
    ),
    MorphologyFeature(
        symbol="<NA>",
        name="Mean nucleoid area",
        group=_MORPH,
        statistic=_MEAN,
        unit="µm2",
        note="from DAPI-stained nucleoids read with the objectDetection module of Oufti, "
        "summing the nucleoids in the cell. Appendix Table S1 lists it under the "
        "morphological features; the main text's headline count of 19 leaves it and its "
        "variability out",
    ),
    MorphologyFeature(
        symbol="CV_NA",
        name="Nucleoid area variability",
        group=_MORPH,
        statistic=_CV,
        note=_CV_NOTE,
    ),
    MorphologyFeature(
        symbol="rho_CD",
        name="Correlation in nucleoid and cell constriction",
        group=_CYCLE,
        statistic=MorphologyStatistic.pearson_correlation,
        note="the Pearson correlation coefficient between the constriction degrees of "
        "the cell and of its nucleoid, over the strain's cells with a significant "
        "constriction degree (above 0.15) only",
    ),
    MorphologyFeature(
        symbol="CDN_C0",
        name="Nucleoid constriction degree at the initiation of cell constriction",
        group=_CYCLE,
        statistic=MorphologyStatistic.regression_intercept,
        note="the intercept of a line whose slope is set by rho_CD, fitted to the same "
        "single-cell data. Unit conflict in the release's own legends: Dataset EV1's "
        "legend writes the name with '(% cell width)' and Dataset EV2's legend drops it, "
        "and the consumed EV2 values lie in [0.145, 0.741], so they are fractions of "
        "cell width rather than percentages; the consumed sheet states no unit, so none "
        "is recorded",
    ),
    MorphologyFeature(
        symbol="Rel.timing div",
        name="Relative timing of cell constriction",
        group=_CYCLE,
        statistic=MorphologyStatistic.inferred_relative_timing,
        note="inferred from the proportion of cells without any significant "
        "constriction under a steady-state cell-age assumption, so it is a relative "
        "cell-cycle age and no absolute duration follows from it. Measured on the "
        "release: it is age(F) = -ln(1 - F/2)/ln(2) applied to the %non-div column, to "
        "a maximum absolute error of 1.2e-15",
    ),
    MorphologyFeature(
        symbol="Rel.timing nuc",
        name="Relative timing of nucleoid separation",
        group=_CYCLE,
        statistic=MorphologyStatistic.inferred_relative_timing,
        note="inferred the same way from the proportion of cells with a single "
        "nucleoid. Measured on the release: it is age(F) = -ln(1 - F/2)/ln(2) applied to "
        "the %1N column, to a maximum absolute error of 1.3e-15",
    ),
    MorphologyFeature(
        symbol="%2N",
        name="Fraction of cells with 2 nucleoids",
        group=_CYCLE,
        statistic=MorphologyStatistic.fraction_of_cells,
        note="a FRACTION despite the percent sign in the symbol: the release's own "
        "legend calls it 'Fraction of cells with 2 nucleoids' and the released values "
        "lie in [0.0, 0.560], so the percent sign is part of the name and not a unit",
    ),
)

CAMPOS2018_MORPHOLOGY_ASSAY = MorphologyAssay(
    name="campos2018",
    description=(
        "Phase-contrast and DAPI imaging of the Keio single-gene deletion collection in "
        "one medium, segmented and quantified with MicrobeTracker and Oufti (Campos et "
        "al. 2018, Mol Syst Biol 14:e7573). The vocabulary is Appendix Table S1, "
        "'Features considered in this study and their associated symbols', minus its two "
        "growth features: 21 morphological and 5 cell cycle symbols, keyed by their "
        "Dataset EV2 column symbol."
    ),
    features=CAMPOS2018_MORPHOLOGY_FEATURES,
)

#: Every bacterial morphology assay a phenotype record may name. An assay is added here
#: with its own vocabulary; nothing widens an existing one.
MORPHOLOGY_ASSAYS: dict[str, MorphologyAssay] = {
    CAMPOS2018_MORPHOLOGY_ASSAY.name: CAMPOS2018_MORPHOLOGY_ASSAY
}


def morphology_assay(name: str) -> MorphologyAssay:
    """The assay ``name`` declares, or a ``KeyError`` naming the registered assays."""
    try:
        return MORPHOLOGY_ASSAYS[name]
    except KeyError:
        raise KeyError(
            f"unknown bacterial morphology assay {name!r}; registered assays are "
            f"{sorted(MORPHOLOGY_ASSAYS)}"
        ) from None


__all__ = [
    "CAMPOS2018_MORPHOLOGY_ASSAY",
    "CAMPOS2018_MORPHOLOGY_FEATURES",
    "MORPHOLOGY_ASSAYS",
    "MorphologyAssay",
    "MorphologyFeature",
    "MorphologyFeatureGroup",
    "MorphologyStatistic",
    "morphology_assay",
]
