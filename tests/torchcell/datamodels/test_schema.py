# tests/torchcell/datamodels/test_schema.py
# [[tests.torchcell.datamodels.test_schema]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datamodels/test_schema.py
"""Tests for the torchcell datamodels schema.

2026.09.30, Phase 16: every validator the rest of the suite never refused, each pinned
with one exact accepted value and one exact refusal message (pydantic prefixes a
``ValueError`` with ``"Value error, "``, so each ``match`` is the escaped message the
validator raises). The inputs are hand-built records: two genes YAL001C and YAL002W for
the per-gene phenotypes, the medium ``YEPD`` (solid, not synthetic), a 1 mM
concentration. Closed forms used: ``fitness_se`` from a standard error of 0.1 is 0.1;
an environment-response sample SD of 0.2 over ``n_samples = 4`` gives SE
``0.2 / sqrt(4) = 0.1``; microarray, RNA-seq and pseudobulk reprs count the genes of
each per-gene dict (2). The discriminated unions are checked by tag
(``perturbation_type``) and by a JSON round trip, and the persisted enum vocabularies
are pinned value by value because the KG stores the values.
"""

import math
import re
import typing
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError
from sortedcontainers import SortedDict

from torchcell.datamodels import schema as s
from torchcell.datamodels.schema import (
    Environment,
    ExperimentType,
    FitnessExperiment,
    FitnessPhenotype,
    GeneInteractionExperiment,
    GeneInteractionPhenotype,
    Genotype,
    Media,
    MicroarrayExpressionPhenotype,
    SgaDampPerturbation,
    SgaKanMxDeletionPerturbation,
    Temperature,
)
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason


@pytest.fixture
def fitness_experiment():
    perturbation1 = SgaKanMxDeletionPerturbation(
        systematic_gene_name="YAL001C", perturbed_gene_name="TFC3", strain_id="DMA1"
    )
    perturbation2 = SgaDampPerturbation(
        systematic_gene_name="YAL003W", perturbed_gene_name="EFB1", strain_id="DMA2"
    )
    genotype = Genotype(perturbations=[perturbation1, perturbation2])
    media = Media(name="YEPD", state="solid", is_synthetic=False)
    temperature = Temperature(value=30.0)
    environment = Environment(media=media, temperature=temperature)
    phenotype = FitnessPhenotype(
        graph_level="global",
        label_name="fitness",
        label_statistic_name="fitness_std",
        fitness=0.85,
        fitness_std=0.05,
    )
    return FitnessExperiment(
        dataset_name="test_fitness_dataset",
        genotype=genotype,
        environment=environment,
        phenotype=phenotype,
    )


@pytest.fixture
def gene_interaction_experiment():
    perturbation1 = SgaKanMxDeletionPerturbation(
        systematic_gene_name="YBR001C", perturbed_gene_name="AAC1", strain_id="DMA3"
    )
    perturbation2 = SgaKanMxDeletionPerturbation(
        systematic_gene_name="YBR002W", perturbed_gene_name="AAC2", strain_id="DMA4"
    )
    genotype = Genotype(perturbations=[perturbation1, perturbation2])
    media = Media(name="YEPD", state="solid", is_synthetic=False)
    temperature = Temperature(value=30.0)
    environment = Environment(media=media, temperature=temperature)
    phenotype = GeneInteractionPhenotype(
        graph_level="edge",
        label_name="gene_interaction",
        label_statistic_name="gene_interaction_p_value",
        gene_interaction=-0.2,
        gene_interaction_p_value=0.01,
    )
    return GeneInteractionExperiment(
        dataset_name="test_gene_interaction_dataset",
        genotype=genotype,
        environment=environment,
        phenotype=phenotype,
    )


@pytest.fixture
def experiment_adapter():
    return TypeAdapter(ExperimentType)


def test_fitness_experiment_serialization(fitness_experiment, experiment_adapter):
    # Serialize experiment
    fitness_json = fitness_experiment.model_dump_json()

    # Deserialize experiment
    loaded_fitness_experiment = experiment_adapter.validate_json(fitness_json)

    # Assert correct type
    assert isinstance(loaded_fitness_experiment, FitnessExperiment)

    # Assert correct experiment_type
    assert loaded_fitness_experiment.experiment_type == "fitness"

    # Assert equality
    assert fitness_experiment == loaded_fitness_experiment


def test_gene_interaction_experiment_serialization(
    gene_interaction_experiment, experiment_adapter
):
    # Serialize experiment
    gene_interaction_json = gene_interaction_experiment.model_dump_json()

    # Deserialize experiment
    loaded_gene_interaction_experiment = experiment_adapter.validate_json(
        gene_interaction_json
    )

    # Assert correct type
    assert isinstance(loaded_gene_interaction_experiment, GeneInteractionExperiment)

    # Assert correct experiment_type
    assert loaded_gene_interaction_experiment.experiment_type == "gene interaction"

    # Assert equality
    assert gene_interaction_experiment == loaded_gene_interaction_experiment


def test_experiment_type_discrimination(
    fitness_experiment, gene_interaction_experiment, experiment_adapter
):
    # Serialize experiments
    fitness_json = fitness_experiment.model_dump_json()
    gene_interaction_json = gene_interaction_experiment.model_dump_json()

    # Deserialize experiments
    loaded_fitness_experiment = experiment_adapter.validate_json(fitness_json)
    loaded_gene_interaction_experiment = experiment_adapter.validate_json(
        gene_interaction_json
    )

    # Assert correct types
    assert isinstance(loaded_fitness_experiment, FitnessExperiment)
    assert isinstance(loaded_gene_interaction_experiment, GeneInteractionExperiment)

    # Assert correct experiment_types
    assert loaded_fitness_experiment.experiment_type == "fitness"
    assert loaded_gene_interaction_experiment.experiment_type == "gene interaction"


def _microarray_kwargs(**overrides):
    """Build valid MicroarrayExpressionPhenotype kwargs (two genes); override to break."""
    genes = ["YAL001C", "YAL002W"]
    kwargs = dict(
        graph_level="node",
        label_name="expression_log2_ratio",
        label_statistic_name="expression_log2_ratio_se",
        expression={g: 100.0 for g in genes},
        expression_log2_ratio={g: 0.5 for g in genes},
        expression_log2_ratio_se={g: 0.1 for g in genes},
        n_replicates={g: 3 for g in genes},
    )
    kwargs.update(overrides)
    return kwargs


def test_microarray_valid_construction():
    phenotype = MicroarrayExpressionPhenotype(**_microarray_kwargs())
    assert set(phenotype.n_replicates) == set(phenotype.expression)
    assert phenotype.n_replicates["YAL001C"] == 3
    assert phenotype.expression_log2_ratio_se is not None
    assert phenotype.expression_log2_ratio_se["YAL001C"] == 0.1


def test_microarray_se_is_optional():
    kwargs = _microarray_kwargs()
    del kwargs["expression_log2_ratio_se"]
    phenotype = MicroarrayExpressionPhenotype(**kwargs)
    assert phenotype.expression_log2_ratio_se is None


def test_microarray_n_replicates_required():
    kwargs = _microarray_kwargs()
    del kwargs["n_replicates"]
    with pytest.raises(ValidationError):
        MicroarrayExpressionPhenotype(**kwargs)


def test_microarray_n_replicates_must_be_positive_int():
    with pytest.raises(ValidationError):
        MicroarrayExpressionPhenotype(
            **_microarray_kwargs(n_replicates={"YAL001C": 0, "YAL002W": 3})
        )


def test_microarray_n_replicates_keys_must_match_expression():
    with pytest.raises(ValidationError):
        MicroarrayExpressionPhenotype(**_microarray_kwargs(n_replicates={"YAL001C": 3}))


@pytest.mark.parametrize(
    "legacy_field",
    [
        "n_samples",
        "expression_se",
        "expression_technical_std",
        "expression_log2_ratio_std",
    ],
)
def test_microarray_rejects_legacy_drift_fields(legacy_field):
    """Guards #14: ModelStrict must reject the pre-refactor field names whose use in
    sameith2015 raised a Pydantic ValidationError before the n_replicates migration.
    """
    with pytest.raises(ValidationError):
        MicroarrayExpressionPhenotype(
            **_microarray_kwargs(**{legacy_field: {"YAL001C": 0.1, "YAL002W": 0.1}})
        )


# --------------------------------------------------------------------------- #
# 2026.09.30, Phase 16: the remaining validators, unions and vocabularies.
# --------------------------------------------------------------------------- #
GENES = ["YAL001C", "YAL002W"]
YEPD = s.Media(name="YEPD", state="solid", is_synthetic=False)
ONE_MM = s.Concentration(value=1.0, unit=s.ConcentrationUnit.millimolar)


def _refuses(message: str) -> Any:
    """``pytest.raises`` for a pydantic refusal carrying exactly ``message``."""
    return pytest.raises(ValidationError, match=re.escape(f"Value error, {message}"))


def _kanmx(name: str = "YAL001C", pert: str = "TFC3", **kw: Any) -> Any:
    return s.SgaKanMxDeletionPerturbation(
        systematic_gene_name=name, perturbed_gene_name=pert, strain_id="DMA1", **kw
    )


def test_gene_perturbation_names_and_provenance_vocabulary() -> None:
    """Coding, dash-suffixed, mitochondrial and ncRNA names pass; a trailing prime
    becomes ``_prime``; provenance accepts ``natural`` and refuses anything else.
    """
    for name in ["YAL001C", "YAL001C-A", "Q0045", "YNCA0001W"]:
        assert _kanmx(name).systematic_gene_name == name
    assert _kanmx(pert="TFC3'").perturbed_gene_name == "TFC3_prime"
    assert _kanmx(provenance="natural").provenance == "natural"
    with _refuses("Invalid systematic gene name format"):
        _kanmx("YZL001C")
    with _refuses("provenance must be 'engineered' or 'natural', got 'wild'"):
        _kanmx(provenance="wild")


_RELAXED: list[tuple[Any, dict[str, Any]]] = [
    (
        s.GeneAdditionPerturbation,
        {
            "source_organism": "Xanthophyllomyces dendrorhous",
            "is_heterologous": True,
            "localization": "plasmid",
        },
    ),
    (s.NaturalGeneAbsencePerturbation, {"strain_id": "AAB"}),
    (s.NaturalGenePresencePerturbation, {"strain_id": "AAB"}),
    (s.CopyNumberVariantPerturbation, {"strain_id": "AAB", "copy_number": 2.0}),
]


@pytest.mark.parametrize(("cls", "extra"), _RELAXED)
def test_relaxed_gene_name_leaves_accept_any_nonempty_identifier(
    cls: Any, extra: dict[str, Any]
) -> None:
    """``crtYB`` (no yeast systematic name) is accepted where the base validator would
    refuse it; the empty string is still refused.
    """
    made = cls(systematic_gene_name="crtYB", perturbed_gene_name="crtYB", **extra)
    assert made.systematic_gene_name == "crtYB"
    with _refuses("systematic_gene_name must be non-empty"):
        cls(systematic_gene_name="", perturbed_gene_name="x", **extra)


def test_genotype_equality_is_set_equality_and_declines_other_types() -> None:
    """Finding: ``Genotype.__eq__`` compares ``set(perturbations)``
    (``schema.py:1008``), so a genotype listing one deletion twice equals the genotype
    listing it once although their lengths are 2 and 1. Pinned until duplicates are
    refused or equality compares the sorted lists. A non-genotype operand gets
    ``NotImplemented``, so ``==`` falls back to False.
    """
    once = s.Genotype(perturbations=[_kanmx()])
    twice = s.Genotype(perturbations=[_kanmx(), _kanmx()])
    assert (len(once), len(twice)) == (1, 2)
    assert once == twice
    assert once.__eq__("YAL001C") is NotImplemented
    assert (once == "YAL001C") is False


def test_temperature_floor_applies_to_celsius_only() -> None:
    """-273 C is accepted and -273.5 C refused.

    Finding: the floor is checked only for Celsius (``schema.py:1035``), so -300 Kelvin,
    which is below absolute zero by 300 K, validates. Pinned until every unit is bounded.
    """
    assert s.Temperature(value=-273.0).value == -273.0
    with _refuses("Temperature cannot be below -273 degrees Celsius"):
        s.Temperature(value=-273.5)
    assert s.Temperature(value=-300.0, unit=s.TemperatureUnit.kelvin).value == -300.0


def _gap(field: str) -> ProvenanceGap:
    return ProvenanceGap(
        field=field, reason=ProvenanceGapReason.not_carried_by_curation
    )


def test_provenance_gap_rules_on_environment() -> None:
    """A gapped None temperature is accepted; gapping ``provenance_gaps`` itself, a
    populated field, or a name that is not a field are each refused.
    """
    env = s.Environment(media=YEPD, provenance_gaps=[_gap("temperature")])
    assert env.temperature is None
    with _refuses("provenance_gaps cannot itself be gapped"):
        s.Environment(media=YEPD, provenance_gaps=[_gap("provenance_gaps")])
    with _refuses(
        "field 'temperature' has a ProvenanceGap but is not None (cannot both store "
        "a value and declare it missing)"
    ):
        s.Environment(
            media=YEPD,
            temperature=s.Temperature(value=30.0),
            provenance_gaps=[_gap("temperature")],
        )
    with _refuses("provenance_gap field 'ph' is not a field of Environment"):
        s.Environment(media=YEPD, provenance_gaps=[_gap("ph")])


def test_concentration_needs_a_value_with_a_unit_or_a_basis() -> None:
    """A basis alone and a value with a unit are accepted; each other form is refused."""
    assert s.Concentration(basis=s.DoseBasis.IC30).value is None
    assert ONE_MM.unit is s.ConcentrationUnit.millimolar
    with _refuses("Concentration needs at least a numeric value or a basis"):
        s.Concentration()
    with _refuses("a numeric concentration value requires a unit"):
        s.Concentration(value=1.0)
    with _refuses("concentration value must be non-negative"):
        s.Concentration(value=-1.0, unit=s.ConcentrationUnit.molar)


def test_media_characterization_and_state_vocabulary() -> None:
    """No components: not characterized, no gaps. A deferred YNB without a dose is a
    gap; with every component defined and dosed the medium is characterized.
    """
    glucose = s.MediaComponent(
        compound=s.Compound(name="glucose"),
        role=s.MediaComponentRole.carbon_source,
        concentration=s.Concentration(value=20.0, unit=s.ConcentrationUnit.g_per_l),
    )
    ynb = s.MediaComponent(
        compound=s.Compound(name="YNB"),
        role=s.MediaComponentRole.complex_ingredient,
        definition=s.ComponentDefinition.composition_deferred,
    )
    assert (YEPD.is_fully_characterized, YEPD.open_gaps) == (False, [])
    partial = s.Media(
        name="SD", state="liquid", is_synthetic=True, components=[glucose, ynb]
    )
    assert (partial.is_fully_characterized, partial.open_gaps) == (False, ["YNB"])
    full = s.Media(name="G", state="liquid", is_synthetic=True, components=[glucose])
    assert (full.is_fully_characterized, full.open_gaps) == (True, [])
    with _refuses('state must be one of "solid", "liquid", or "gas"'):
        s.Media(name="X", state="plasma", is_synthetic=True)


def test_environment_aerobicity_vocabulary() -> None:
    """The three oxygen regimes pass; ``hypoxic`` is refused."""
    for regime in ["aerobic", "anaerobic", "microaerobic"]:
        assert s.Environment(media=YEPD, aerobicity=regime).aerobicity == regime
    with _refuses("aerobicity must be aerobic/anaerobic/microaerobic, got 'hypoxic'"):
        s.Environment(media=YEPD, aerobicity="hypoxic")


def test_phenotype_label_fields_must_be_own_fields_of_the_concrete_class() -> None:
    """``provenance_gaps`` is inherited, not in ``FitnessPhenotype.__annotations__``, so
    it is refused as a label and as a statistic name; ``__getitem__`` reads a field.
    """
    assert s.FitnessPhenotype(fitness=0.85)["fitness"] == 0.85
    with _refuses("label_name 'provenance_gaps' must be a class attribute"):
        s.FitnessPhenotype(fitness=0.85, label_name="provenance_gaps")
    with _refuses("label_statistic_name 'provenance_gaps' must be a class attribute"):
        s.FitnessPhenotype(fitness=0.85, label_statistic_name="provenance_gaps")


def test_graph_level_refusal_lists_every_level() -> None:
    """Finding: the message joins a SET (``schema.py:1617-1619``), so the order of the
    seven levels varies with the hash seed; the set of names is what is pinned.
    Pinned until the levels are joined in a fixed order.
    """
    with pytest.raises(ValidationError) as caught:
        s.FitnessPhenotype(fitness=0.85, graph_level="cell")
    listed = re.search(r"graph_level must be one of: ([a-z, ]+?) \[", str(caught.value))
    assert listed is not None
    assert set(listed.group(1).split(", ")) == {
        "edge",
        "node",
        "hyperedge",
        "subgraph",
        "global",
        "metabolism",
        "gene ontology",
    }


def test_fitness_value_clamping_nan_and_n_samples() -> None:
    """Negative fitness clamps to 0.0; NaN and ``n_samples = 0`` are refused."""
    assert s.FitnessPhenotype(fitness=-0.3).fitness == 0.0
    with _refuses("Fitness cannot be NaN"):
        s.FitnessPhenotype(fitness=math.nan)
    with _refuses("n_samples must be a positive integer or None, got: 0"):
        s.FitnessPhenotype(fitness=0.85, n_samples=0)


def _fitness_attributes(**overrides: Any) -> SimpleNamespace:
    fields: dict[str, Any] = {name: None for name in s.FitnessPhenotype.model_fields}
    fields.update(
        graph_level="global",
        label_name="fitness",
        label_statistic_name="fitness_se",
        provenance_gaps=[],
        fitness=0.5,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_fitness_se_is_derived_from_a_dict_but_not_from_attributes() -> None:
    """From a dict, a standard error of 0.1 becomes ``fitness_se`` 0.1.

    Finding: ``_fill_fitness_se`` returns non-dict input untouched (``schema.py:1785``),
    so the same record validated ``from_attributes`` keeps ``fitness_se`` None. Pinned
    until the derivation also reads attribute input.
    """
    labeled: dict[str, Any] = {
        "fitness_uncertainty": 0.1,
        "fitness_uncertainty_type": s.UncertaintyType.standard_error,
    }
    assert s.FitnessPhenotype(fitness=0.5, **labeled).fitness_se == 0.1
    from_object = s.FitnessPhenotype.model_validate(
        _fitness_attributes(**labeled), from_attributes=True
    )
    assert from_object.fitness_uncertainty == 0.1
    assert from_object.fitness_se is None


def _calmorph(**overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "calmorph": {"A101_A": 1.5},
        "calmorph_coefficient_of_variation": {"ACV101_A": 0.2},
    }
    kwargs.update(overrides)
    return s.CalMorphPhenotype(**kwargs)


def test_calmorph_labels_and_nan_refusals() -> None:
    """A known base label and a known CV label pass; empty, NaN and unknown keys are
    refused with the module's messages; a missing CV table is allowed.
    """
    assert _calmorph().calmorph == {"A101_A": 1.5}
    assert _calmorph(calmorph_coefficient_of_variation=None).label_name == "calmorph"
    with _refuses("calmorph measurements cannot be empty"):
        _calmorph(calmorph={})
    with _refuses("calmorph measurement A101_A cannot be NaN"):
        _calmorph(calmorph={"A101_A": math.nan})
    with _refuses(
        "Invalid CalMorph base parameter: ZZZ. Must be one of the 281 base parameters "
        "in CALMORPH_LABELS."
    ):
        _calmorph(calmorph={"ZZZ": 1.0})
    with _refuses(
        "Invalid CalMorph CV parameter: A101_A. Must be one of the 220 CV parameters "
        "in CALMORPH_STATISTICS."
    ):
        _calmorph(calmorph_coefficient_of_variation={"A101_A": 0.1})
    with _refuses("CV measurement ACV101_A cannot be NaN"):
        _calmorph(calmorph_coefficient_of_variation={"ACV101_A": math.nan})


def test_publication_needs_an_identifier_and_a_url() -> None:
    """A DOI with its URL is accepted; an identifier without any URL, and a URL without
    any identifier, are refused.
    """
    assert s.Publication(doi="10.1/x", doi_url="https://doi.org/10.1/x").doi == "10.1/x"
    with _refuses("At least one of PubMed URL or DOI URL must be provided"):
        s.Publication(pubmed_id="123")
    with _refuses("At least one of PubMed ID or DOI must be provided"):
        s.Publication(doi_url="https://doi.org/10.1/x")


def _microarray(**overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "expression": {"YAL002W": 90.0, "YAL001C": 100.0},
        "expression_log2_ratio": {g: 0.5 for g in GENES},
        "expression_log2_ratio_se": {g: 0.1 for g in GENES},
        "n_replicates": {g: 3 for g in GENES},
    }
    kwargs.update(overrides)
    return s.MicroarrayExpressionPhenotype(**kwargs)


def test_microarray_repr_sorting_and_nan_standard_error() -> None:
    """The repr counts genes per dict; a NaN SE (n = 1) and a NaN variance are accepted.

    Finding: the before-validators coerce to ``SortedDict``, but the ``dict[str, float]``
    field then validates it back into a plain ``dict`` (``schema.py:2160-2167``), so what
    is stored is a plain dict whose insertion order happens to be sorted, not the
    ``SortedDict`` the field descriptions promise. Pinned until the annotation or the
    descriptions agree.
    """
    phenotype = _microarray(
        expression_log2_ratio_se={"YAL001C": math.nan, "YAL002W": 0.1},
        expression_log2_ratio_variance={"YAL001C": math.nan, "YAL002W": 0.01},
    )
    assert repr(phenotype) == (
        "MicroarrayExpressionPhenotype(expression_genes=2, log2_ratio_genes=2, "
        "log2_se_genes=2, n_replicates_genes=2)"
    )
    assert type(phenotype.expression) is dict
    assert list(phenotype.expression.items()) == [("YAL001C", 100.0), ("YAL002W", 90.0)]
    assert math.isnan(phenotype.expression_log2_ratio_se["YAL001C"])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"expression": None}, "expression measurements cannot be None"),
        ({"expression": {}}, "expression measurements cannot be empty"),
        (
            {"expression": {"YAL001C": math.inf, "YAL002W": 1.0}},
            "Invalid expression value for gene YAL001C: inf",
        ),
        ({"expression_log2_ratio": None}, "expression_log2_ratio cannot be None"),
        ({"expression_log2_ratio": {}}, "expression_log2_ratio cannot be empty"),
        (
            {"expression_log2_ratio_se": {"YAL001C": -0.1, "YAL002W": 0.1}},
            "SE for YAL001C cannot be negative: -0.1",
        ),
        (
            {"expression_log2_ratio_variance": {"YAL001C": -0.01, "YAL002W": 0.01}},
            "Variance for YAL001C cannot be negative: -0.01",
        ),
        (
            {"expression_log2_ratio_variance": {"YAL001C": 0.01}},
            "expression_log2_ratio_variance must have the same keys as "
            "expression_log2_ratio",
        ),
        (
            {"expression_log2_ratio_se": {"YAL001C": 0.1}},
            "expression_log2_ratio_se must have the same keys as expression_log2_ratio",
        ),
        (
            {"n_replicates": None},
            "n_replicates cannot be None - it is required for SE interpretation",
        ),
        (
            {"n_replicates": 3},
            "n_replicates must be a per-gene dict for this phenotype, got int",
        ),
        ({"n_replicates": {}}, "n_replicates cannot be empty"),
        (
            {"n_replicates": {"YAL001C": 0, "YAL002W": 3}},
            "n_replicates for YAL001C must be a positive integer, got: 0",
        ),
        (
            {"n_replicates": {"YAL001C": 3}},
            "n_replicates must have the same keys as expression",
        ),
    ],
)
def test_microarray_refusals(overrides: dict[str, Any], message: str) -> None:
    """Each malformed per-gene dict is refused with its own message."""
    with _refuses(message):
        _microarray(**overrides)


def test_microarray_log2_ratio_keys_are_never_checked_against_expression() -> None:
    """Finding: ``validate_matching_keys`` ties ``n_replicates`` to ``expression`` and the
    SE and variance to ``expression_log2_ratio`` (``schema.py:2240-2262``) but never ties
    the two primary dicts together, so log2 ratios for YAL003W and YAL004C beside
    expression for YAL001C and YAL002W validate. Pinned until the ratio keys must match.
    """
    other = ["YAL003W", "YAL004C"]
    phenotype = _microarray(
        expression_log2_ratio={g: 0.5 for g in other},
        expression_log2_ratio_se={g: 0.1 for g in other},
    )
    assert list(phenotype.expression_log2_ratio) == other
    assert list(phenotype.expression) == GENES


def _rnaseq(**overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "expression_tpm": SortedDict({"YAL001C": 10.0, "YAL002W": 0.0}),
        "expression_count": SortedDict({"YAL001C": 50, "YAL002W": 0}),
    }
    kwargs.update(overrides)
    return s.RNASeqExpressionPhenotype(**kwargs)


def test_rnaseq_repr_and_sorted_dict_passthrough() -> None:
    """Already-sorted dicts pass through; zero TPM and zero counts are legal values."""
    phenotype = _rnaseq()
    assert repr(phenotype) == "RNASeqExpressionPhenotype(tpm_genes=2, count_genes=2)"
    assert dict(phenotype.expression_tpm) == {"YAL001C": 10.0, "YAL002W": 0.0}
    assert dict(phenotype.expression_count) == {"YAL001C": 50, "YAL002W": 0}
    assert phenotype.measurement_type == "rnaseq_tpm"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"expression_tpm": None}, "expression_tpm cannot be None"),
        ({"expression_tpm": {}}, "expression_tpm cannot be empty"),
        ({"expression_tpm": {"YAL001C": -1.0}}, "Invalid TPM for gene YAL001C: -1.0"),
        ({"expression_count": None}, "expression_count cannot be None"),
        (
            {"expression_count": [50, 0]},
            "expression_count must be a per-gene dict, got list",
        ),
        ({"expression_count": {}}, "expression_count cannot be empty"),
        (
            {"expression_count": {"YAL001C": 1.5, "YAL002W": 0}},
            "expression_count for YAL001C must be a non-negative integer, got: 1.5",
        ),
        (
            {"expression_count": {"YAL001C": 50}},
            "expression_count must have the same keys as expression_tpm",
        ),
    ],
)
def test_rnaseq_refusals(overrides: dict[str, Any], message: str) -> None:
    """Each malformed TPM or count dict is refused with its own message."""
    with _refuses(message):
        _rnaseq(**overrides)


def test_pseudobulk_repr_and_refusals() -> None:
    """The repr shows the gene count and the two scalars; None, empty and NaN ratios,
    a negative or infinite dispersion and zero cells are refused.
    """
    ratios = {"YAL001C": 0.4, "YAL002W": -0.2}
    phenotype = s.PseudobulkExpressionPhenotype(
        expression_log2_ratio=SortedDict(ratios), dispersion=1.2, n_cells=40
    )
    assert repr(phenotype) == (
        "PseudobulkExpressionPhenotype(log2_ratio_genes=2, dispersion=1.2, n_cells=40)"
    )
    cases: list[tuple[dict[str, Any], str]] = [
        ({"expression_log2_ratio": None}, "expression_log2_ratio cannot be None"),
        ({"expression_log2_ratio": {}}, "expression_log2_ratio cannot be empty"),
        (
            {"expression_log2_ratio": {"YAL001C": math.nan}},
            "Invalid log2 ratio for gene YAL001C: nan",
        ),
        (
            {"expression_log2_ratio": ratios, "dispersion": -0.1},
            "dispersion must be a non-negative finite float, got -0.1",
        ),
        (
            {"expression_log2_ratio": ratios, "dispersion": math.inf},
            "dispersion must be a non-negative finite float, got inf",
        ),
        (
            {"expression_log2_ratio": ratios, "n_cells": 0},
            "n_cells must be a positive integer, got 0",
        ),
    ]
    for kwargs, message in cases:
        with _refuses(message):
            s.PseudobulkExpressionPhenotype(**kwargs)


def _visual(**overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "visual_score": 3.0,
        "n_replicates": 2,
        "score_scale_min": 0,
        "score_scale_max": 5,
        "score_semantics": "higher = more product",
        "target_product": "betaxanthin",
    }
    kwargs.update(overrides)
    return s.VisualScorePhenotype(**kwargs)


def test_visual_score_scale_bounds() -> None:
    """A score and a minimum inside [0, 5] pass; a minimum of 6 is outside the scale,
    and an inverted scale, a score of 7 and zero replicates are refused.
    """
    assert _visual(visual_score_min=1.0).visual_score_min == 1.0
    with _refuses("visual_score_min outside the declared scale"):
        _visual(visual_score_min=6.0)
    with _refuses("score_scale_min must be < score_scale_max"):
        _visual(score_scale_min=5)
    with _refuses("visual_score 7.0 outside scale [0, 5]"):
        _visual(visual_score=7.0)
    with _refuses("n_replicates must be >= 1"):
        _visual(n_replicates=0)


@pytest.mark.parametrize(
    ("cls", "level", "se_field", "label"),
    [
        (
            s.MetabolitePhenotype,
            "metabolite_level",
            "metabolite_level_se",
            "metabolite_level",
        ),
        (
            s.ProteinAbundancePhenotype,
            "protein_abundance",
            "protein_abundance_se",
            "protein_abundance",
        ),
    ],
)
def test_per_key_abundance_replicates_and_standard_errors(
    cls: Any, level: str, se_field: str, label: str
) -> None:
    """Two keys a and b: a NaN SE is accepted; zero replicates, an SE for a key without
    a level, a negative SE, mismatched replicate keys and an empty level are refused.
    """

    def make(**overrides: Any) -> Any:
        kwargs: dict[str, Any] = {
            level: {"a": 1.0, "b": 2.0},
            "n_replicates": {"a": 3, "b": 3},
            "measurement_type": "fluorescence",
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    made = make(**{se_field: {"a": math.nan, "b": 0.1}})
    assert getattr(made, se_field)["b"] == 0.1
    with _refuses("n_replicates for a must be >= 1"):
        make(n_replicates={"a": 0, "b": 3})
    with _refuses(f"SE key c not in {label}"):
        make(**{se_field: {"c": 0.1}})
    with _refuses("SE for a must be non-negative"):
        make(**{se_field: {"a": -0.1}})
    with _refuses(f"n_replicates keys must match {label} keys"):
        make(n_replicates={"a": 3})
    with _refuses(f"{label} cannot be empty"):
        make(**{level: {}}, n_replicates={})


def _response(**overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "measurement_type": s.MeasurementType.log2_ratio,
        "environment_response": -0.7,
    }
    kwargs.update(overrides)
    return s.EnvironmentResponsePhenotype(**kwargs)


def test_environment_response_uncertainty_rules() -> None:
    """A sample SD of 0.2 over 4 biological replicates derives SE 0.1. Without a sample
    count, or without a sample unit, the SD is refused; an uncertainty without its type
    and ``n_samples = 0`` are refused too.
    """
    sd = {
        "environment_response_uncertainty": 0.2,
        "environment_response_uncertainty_type": s.UncertaintyType.sample_sd,
    }
    derived = _response(
        **sd, n_samples=4, sample_unit=s.SampleUnit.biological_replicate
    )
    assert derived.environment_response_se == pytest.approx(0.1, abs=1e-15)
    with _refuses("n_samples and sample_unit are required for sample_sd"):
        _response(**sd)
    with _refuses("n_samples and sample_unit are required for sample_sd"):
        _response(**sd, n_samples=4)
    with _refuses(
        "environment_response_uncertainty and its type must both be set or both be "
        "None (no unlabelled uncertainty)"
    ):
        _response(environment_response_uncertainty=0.2)
    with _refuses("n_samples must be a positive integer or None, got: 0"):
        _response(n_samples=0)


def test_environment_response_from_attributes_skips_the_se_derivation() -> None:
    """Finding: as for fitness, ``_fill_response_se`` passes non-dict input through
    (``schema.py:2990``), so a standard error of 0.3 read ``from_attributes`` leaves
    ``environment_response_se`` None while the dict form sets it to 0.3. Pinned until
    the derivation reads attribute input.
    """
    labeled: dict[str, Any] = {
        "environment_response_uncertainty": 0.3,
        "environment_response_uncertainty_type": s.UncertaintyType.standard_error,
    }
    assert _response(**labeled).environment_response_se == 0.3
    fields: dict[str, Any] = {
        name: None for name in s.EnvironmentResponsePhenotype.model_fields
    }
    fields.update(
        graph_level="global",
        label_name="environment_response",
        label_statistic_name="environment_response_se",
        provenance_gaps=[],
        measurement_type=s.MeasurementType.log2_ratio,
        environment_response=-0.7,
        **labeled,
    )
    read = s.EnvironmentResponsePhenotype.model_validate(
        SimpleNamespace(**fields), from_attributes=True
    )
    assert read.environment_response_uncertainty == 0.3
    assert read.environment_response_se is None


def test_environment_perturbation_union_resolves_by_tag() -> None:
    """Each ``perturbation_type`` tag selects its leaf; an unknown tag is refused by
    every member of the union.
    """
    adapter: TypeAdapter[Any] = TypeAdapter(s.EnvironmentPerturbationType)
    dose = {"value": 1.0, "unit": "mM"}
    cases = [
        (
            {
                "perturbation_type": "small_molecule",
                "compound": {"name": "caffeine"},
                "concentration": dose,
            },
            s.SmallMoleculePerturbation,
        ),
        (
            {"perturbation_type": "environment_physical", "factor": "pH"},
            s.EnvironmentPhysicalPerturbation,
        ),
        (
            {
                "perturbation_type": "biologic",
                "agent_class": "toxin",
                "name": "t",
                "concentration": {"basis": "IC30"},
            },
            s.BiologicPerturbation,
        ),
    ]
    for payload, expected in cases:
        assert type(adapter.validate_python(payload)) is expected
    with pytest.raises(ValidationError) as excinfo:
        adapter.validate_python(
            {"perturbation_type": "radioactive", "description": "d"}
        )
    assert excinfo.value.errors()[0]["type"] == "literal_error"


def test_environment_with_every_perturbation_leaf_round_trips_through_json() -> None:
    """Dump and reload keeps each leaf class and every value."""
    environment = s.Environment(
        media=YEPD,
        temperature=s.Temperature(value=30.0),
        perturbations=[
            s.SmallMoleculePerturbation(
                compound=s.Compound(name="caffeine", pubchem_cid=2519),
                concentration=ONE_MM,
            ),
            s.EnvironmentPhysicalPerturbation(
                factor=s.PhysicalFactor.osmolarity, agent=s.Compound(name="NaCl")
            ),
            s.BiologicPerturbation(
                agent_class=s.BiologicAgentClass.peptide,
                name="defensin",
                concentration=s.Concentration(basis=s.DoseBasis.IC50),
            ),
        ],
        duration_generations=5.0,
    )
    reloaded = s.Environment.model_validate_json(environment.model_dump_json())
    assert reloaded == environment
    assert [type(p).__name__ for p in reloaded.perturbations] == [
        "SmallMoleculePerturbation",
        "EnvironmentPhysicalPerturbation",
        "BiologicPerturbation",
    ]


def test_genotype_of_mixed_leaves_round_trips_and_sorts_by_gene() -> None:
    """Perturbations are sorted by systematic name on construction; JSON reload keeps
    each leaf class, and the relaxed-name addition sorts before the ``Y`` genes.
    """
    genotype = s.Genotype(
        perturbations=[
            _kanmx("YAL002W", "VPS8"),
            s.GeneAdditionPerturbation(
                systematic_gene_name="crtYB",
                perturbed_gene_name="crtYB",
                source_organism="Xanthophyllomyces dendrorhous",
                is_heterologous=True,
                localization="plasmid",
            ),
            s.SgaDampPerturbation(
                systematic_gene_name="YAL001C",
                perturbed_gene_name="TFC3",
                strain_id="D2",
            ),
        ]
    )
    reloaded = s.Genotype.model_validate_json(genotype.model_dump_json())
    assert reloaded.systematic_gene_names == ["YAL001C", "YAL002W", "crtYB"]
    assert [p.systematic_gene_name for p in reloaded.perturbations] == [
        "YAL001C",
        "YAL002W",
        "crtYB",
    ]
    assert [type(p).__name__ for p in reloaded.perturbations] == [
        "SgaDampPerturbation",
        "SgaKanMxDeletionPerturbation",
        "GeneAdditionPerturbation",
    ]


@pytest.mark.parametrize(
    ("enum", "values"),
    [
        (
            s.UncertaintyType,
            ["sample_sd", "standard_error", "bootstrap_se", "variance", "ci95"],
        ),
        (
            s.SampleUnit,
            [
                "colony",
                "screen",
                "biological_replicate",
                "technical_replicate",
                "pooled",
            ],
        ),
        (s.TemperatureUnit, ["Celsius", "Kelvin", "Fahrenheit"]),
        (
            s.ComponentDefinition,
            ["defined", "composition_deferred", "intrinsically_undefined"],
        ),
        (
            s.PhysicalFactor,
            [
                "pH",
                "osmolarity",
                "carbon_source",
                "nitrogen_source",
                "ionic_strength",
                "nutrient_dropout",
                "radiation",
            ],
        ),
        (s.BiologicAgentClass, ["peptide", "protein", "antibody", "toxin"]),
        (s.DoseBasis, ["IC30", "IC50", "MIC", "fixed", "reduced_from_standard"]),
    ],
)
def test_persisted_enum_vocabularies(enum: Any, values: list[str]) -> None:
    """The stored values, in declaration order; a rename would orphan served records."""
    assert [member.value for member in enum] == values


def test_categorical_measurement_types_are_exactly_categorical_and_ordinal() -> None:
    """Only these two readouts need ``category`` instead of a numeric response."""
    assert s.CATEGORICAL_MEASUREMENT_TYPES == frozenset(
        {s.MeasurementType.categorical, s.MeasurementType.ordinal}
    )


# Phase 24: strain-background, genomic-span and pre-culture validator messages


def _bg_allele(
    functional: bool, zygosity: s.Zygosity, allele_name: str
) -> s.BackgroundAllele:
    """A LYS2 allele sourced by a declared gap (no quote needed)."""
    return s.BackgroundAllele(
        systematic_gene_name="YBR115C",
        gene_name="LYS2",
        allele_name=allele_name,
        edit=s.AlleleEdit.full_deletion,
        functional=functional,
        zygosity=zygosity,
        provenance=None,
        provenance_gaps=[_gap("provenance")],
    )


def test_gapped_empty_list_is_refused_by_the_mixin_before_the_empty_list_check() -> (
    None
):
    """Finding: ``_require_value_or_gap``'s "is an empty list" branch is unreachable.

    It needs a gapped field holding ``[]``, but ``ProvenanceGapMixin``'s validator runs
    first and refuses ANY gapped field that is not None, so its message is what a caller
    sees.
    """
    with _refuses(
        "field 'provenance' has a ProvenanceGap but is not None (cannot both store a "
        "value and declare it missing)"
    ):
        s.BackgroundAllele(
            systematic_gene_name="YBR115C",
            gene_name="LYS2",
            allele_name="lys2Δ0",
            edit=s.AlleleEdit.full_deletion,
            functional=False,
            zygosity=s.Zygosity.haploid,
            provenance=[],
            provenance_gaps=[_gap("provenance")],
        )


def test_genomic_span_needs_a_chromosome_name() -> None:
    with _refuses("GenomicSpan needs a chromosome and an assembly name"):
        s.GenomicSpan(chromosome=" ", start=1, end=1, assembly="R64")


def test_strain_background_name_cannot_be_blank() -> None:
    with _refuses("StrainBackground.name cannot be empty"):
        s.StrainBackground(
            name=" ",
            mating_type=s.MatingType.a,
            ploidy="haploid",
            provenance=None,
            provenance_gaps=[_gap("provenance")],
        )


def test_functional_copies_skips_a_functional_allele_of_a_compound_heterozygote() -> (
    None
):
    """Diploid: 2 copies, minus 1 for the heterozygous null; the functional entry at
    the same locus subtracts nothing, so the count is 1 (it would be 0 if it counted).
    """
    background = s.StrainBackground(
        name="toy",
        mating_type=s.MatingType.a_alpha,
        ploidy="diploid",
        alleles=[
            _bg_allele(True, s.Zygosity.heterozygous, "LYS2"),
            _bg_allele(False, s.Zygosity.heterozygous, "lys2Δ0"),
        ],
        provenance=None,
        provenance_gaps=[_gap("provenance")],
    )
    assert background.functional_copies("YBR115C") == 1


@pytest.mark.parametrize(
    "field", ["generations", "duration_hours", "od600_at_transfer"]
)
def test_pre_culture_quantities_must_be_non_negative(field: str) -> None:
    with _refuses(f"PreCulture.{field} must be non-negative"):
        s.PreCulture.model_validate({"source": "log_phase_culture", field: -0.5})


# --------------------------------------------------------------------------- #
# Bacterial additions (plan 3a, 3b, 3c, 3e: [[plan.bacteria-ontology-genome]]).
#
# Three properties are what the whole bacterial expansion rests on and each is
# checked here rather than assumed:
#   1. the two identifier families stay SEPARATE -- a bacterial tag fails on a yeast
#      leaf and a yeast name fails on a bacterial one, so one id space cannot leak
#      into the other through ``Genotype``'s content identity;
#   2. the assembly pin is only honored where a field is re-annotated to the
#      subclass, which is a pydantic fact and therefore a test, not a comment;
#   3. the vocabularies (assembly set ids, locus-tag patterns, the BW25113 genotype)
#      agree with the deposited tier rather than with memory.
# --------------------------------------------------------------------------- #
_MG1655: dict[str, Any] = dict(
    systematic_gene_name="b0002",
    perturbed_gene_name="thrA",
    gene_namespace="ecoli_k12_mg1655_bnumber",
)

_BACTERIAL_LEAF_CASES: list[tuple[type[s.GenePerturbation], dict[str, Any]]] = [
    (s.BacterialDeletionPerturbation, {**_MG1655}),
    (
        s.TransposonInsertionPerturbation,
        {
            **_MG1655,
            "barcode": "ACGT" * 5,
            "insertion_position": 547,
            "insertion_strand": "-",
        },
    ),
    (
        s.BacterialCrisprInterferencePerturbation,
        {**_MG1655, "crispr": s.CrisprConstruct(effector="dCas9-Mxi1")},
    ),
    (
        s.PromoterReplacementPerturbation,
        {**_MG1655, "expression_direction": "decreased", "promoter_name": "Ptac"},
    ),
    (
        s.HeterologousPathwayPerturbation,
        dict(
            systematic_gene_name="Efa:mvaE",
            perturbed_gene_name="mvaE",
            gene_namespace="pputida_kt2440_locus_tag",
            source_organism="Enterococcus faecalis",
            is_heterologous=True,
            localization="chromosomal_integration",
            pathway_name="isoprenol via mevalonate",
        ),
    ),
]
_LEAF_IDS = [cls.__name__ for cls, _ in _BACTERIAL_LEAF_CASES]


def _assembly_reference(
    strain: str = "MG1655",
    assembly_set: Any = "ecoli_K12_MG1655_ASM584v2",
    accession: str = "GCA_000005845.2",
    **kw: Any,
) -> s.AssemblyReferenceGenome:
    return s.AssemblyReferenceGenome(
        species="Escherichia coli",
        strain=strain,
        assembly_set=assembly_set,
        assembly_accession=accession,
        **kw,
    )


def _lb() -> Media:
    return Media(name="LB", state="liquid", is_synthetic=False)


def _experiment_cls(kind: str) -> type[s.Experiment]:
    """``EXPERIMENT_TYPE_MAP[kind]``, narrowed to the class the schema guarantees."""
    cls = s.EXPERIMENT_TYPE_MAP[kind]
    assert isinstance(cls, type) and issubclass(cls, s.Experiment)
    return cls


def _reference_cls(kind: str) -> type[s.ExperimentReference]:
    """``EXPERIMENT_REFERENCE_TYPE_MAP[kind]``, narrowed the same way."""
    cls = s.EXPERIMENT_REFERENCE_TYPE_MAP[kind]
    assert isinstance(cls, type) and issubclass(cls, s.ExperimentReference)
    return cls


# --- 1. the two identifier families stay separate -------------------------- #
def test_the_yeast_validator_still_refuses_a_b_number() -> None:
    """The base yeast validator is UNTOUCHED, which is the point of the new leaves.

    Widening ``GenePerturbation.validate_sys_gene_name`` to admit bacterial tags would
    let ``b0002`` into an SGA deletion leaf, and it was measured to move 35 of 36
    served dataset closures. So the yeast leaves must keep refusing, and a bacterial
    record must be impossible to file as a yeast one.
    """
    with _refuses("Invalid systematic gene name format"):
        s.DeletionPerturbation(systematic_gene_name="b0002", perturbed_gene_name="thrA")
    with _refuses("Invalid systematic gene name format"):
        s.KanMxDeletionPerturbation(
            systematic_gene_name="b0002", perturbed_gene_name="thrA"
        )
    with _refuses("Invalid systematic gene name format"):
        _kanmx(name="PP_0002", pert="x")


def test_the_bacterial_leaf_validates_exactly_what_the_yeast_leaf_refuses() -> None:
    """The same identifier: refused as a yeast deletion, accepted as a bacterial one."""
    with _refuses("Invalid systematic gene name format"):
        s.DeletionPerturbation(systematic_gene_name="b0002", perturbed_gene_name="thrA")
    accepted = s.BacterialDeletionPerturbation(**_MG1655)
    assert accepted.systematic_gene_name == "b0002"
    assert accepted.gene_namespace == "ecoli_k12_mg1655_bnumber"
    # and it is still a deletion, so an "every knockout" filter keeps catching it
    assert isinstance(accepted, s.DeletionPerturbation)
    assert accepted.state == "absent"
    assert accepted.mechanism_so_id == "SO:0000159"


@pytest.mark.parametrize("cls,kwargs", _BACTERIAL_LEAF_CASES, ids=_LEAF_IDS)
def test_a_bacterial_leaf_refuses_a_yeast_systematic_name(
    cls: type[s.GenePerturbation], kwargs: dict[str, Any]
) -> None:
    """Symmetry: a yeast ORF name is not a locus tag.

    The gene-addition-derived leaf is the documented exception -- it relaxes the name
    validator because a heterologous gene has no host locus tag at all -- so it is
    checked for what it DOES refuse instead.
    """
    if cls is s.HeterologousPathwayPerturbation:
        with _refuses("systematic_gene_name must be non-empty"):
            cls(**{**kwargs, "systematic_gene_name": ""})
        return
    with pytest.raises(ValidationError, match="Invalid bacterial locus tag"):
        cls(**{**kwargs, "systematic_gene_name": "YAL001C"})


def test_the_namespace_must_agree_with_the_tag_it_carries() -> None:
    """A KT2440 tag cannot be filed under the MG1655 namespace.

    The namespace is what keeps the three hosts' tags in separate id spaces, so a
    record whose namespace disagrees with its own tag would defeat the mechanism while
    looking well-formed.
    """
    with _refuses(
        "'PP_0002' is a pputida_kt2440_locus_tag tag but gene_namespace is "
        "'ecoli_k12_mg1655_bnumber'"
    ):
        s.BacterialDeletionPerturbation(
            systematic_gene_name="PP_0002",
            perturbed_gene_name="x",
            gene_namespace="ecoli_k12_mg1655_bnumber",
        )
    # the matching namespace is accepted
    assert (
        s.BacterialDeletionPerturbation(
            systematic_gene_name="PP_0002",
            perturbed_gene_name="x",
            gene_namespace="pputida_kt2440_locus_tag",
        ).gene_namespace
        == "pputida_kt2440_locus_tag"
    )


def test_the_three_locus_tag_patterns_are_pairwise_disjoint() -> None:
    """No identifier can belong to two namespaces, which is what makes the id space safe."""
    samples = {
        "ecoli_k12_mg1655_bnumber": ["b0001", "b4651"],
        "ecoli_k12_bw25113_locus_tag": ["BW25113_0001", "BW25113_4490"],
        "pputida_kt2440_locus_tag": [
            "PP_0002",
            "PP_16SA",
            "PP_23SG",
            "PP_t01",
            "PP_mr01",
        ],
    }
    for owner, tags in samples.items():
        for tag in tags:
            matching = {
                namespace
                for namespace, pattern in s.BACTERIAL_LOCUS_TAG_PATTERNS.items()
                if re.match(pattern, tag)
            }
            assert matching == {owner}, (tag, matching)


def test_a_kt2440_structural_rna_tag_is_admitted() -> None:
    """KT2440's named RNA tags are real genes, so the pattern must not be digits-only.

    Measured from the deposited GenBank file: 165 of KT2440's 5,786 gene features carry
    a named tag rather than a four-digit one, so a digits-only pattern would silently
    refuse every rRNA, tRNA and ``PP_mr`` gene.
    """
    for tag in (
        "PP_16SA",
        "PP_5SG",
        "PP_23SB",
        "PP_t75",
        "PP_mr67",
        "PP_tm01",
        "PP_r01",
    ):
        assert s._namespace_of_locus_tag(tag) == "pputida_kt2440_locus_tag", tag
    assert s._namespace_of_locus_tag("PP_nope") is None


@pytest.mark.parametrize("cls,kwargs", _BACTERIAL_LEAF_CASES, ids=_LEAF_IDS)
def test_a_bacterial_leaf_round_trips_through_genotype(
    cls: type[s.GenePerturbation], kwargs: dict[str, Any]
) -> None:
    """Each leaf survives a ``Genotype`` dump + revalidate as its own class."""
    leaf: Any = cls(**kwargs)
    genotype = Genotype(perturbations=[leaf])
    back = Genotype.model_validate(genotype.model_dump())
    assert type(back.perturbations[0]) is cls
    assert back.perturbations[0].model_dump() == leaf.model_dump()
    assert back == genotype


def test_every_bacterial_leaf_round_trips_in_one_genotype() -> None:
    """All five in one genotype, so the union resolves each tag unambiguously."""
    leaves: list[Any] = [cls(**kwargs) for cls, kwargs in _BACTERIAL_LEAF_CASES]
    genotype = Genotype(
        perturbations=[
            leaf.model_copy(update={"perturbed_gene_name": f"g{index}"})
            for index, leaf in enumerate(leaves)
        ]
    )
    back = Genotype.model_validate(genotype.model_dump())
    assert {type(p) for p in back.perturbations} == {type(leaf) for leaf in leaves}
    assert len(back) == 5


# --- 2. the assembly pin, and where it survives ---------------------------- #
def test_assembly_reference_genome_refuses_an_unknown_assembly_set() -> None:
    """A set id outside the deposited vocabulary cannot be stored.

    The pin is only worth anything if it dereferences, so an invented set id is a
    validation error rather than a pointer that fails later at ``resolve``.
    """
    with pytest.raises(ValidationError):
        _assembly_reference(assembly_set="ecoli_K12_MG1655_ASM584v3")
    with pytest.raises(ValidationError):
        _assembly_reference(assembly_set="sgd_S288C_R64-4-1_20230830")


def test_assembly_reference_genome_refuses_a_foreign_or_malformed_accession() -> None:
    """The accession must be one of ITS OWN set's two, so the pair cannot disagree."""
    with _refuses(
        "'GCA_000750555.1' is not an accession of 'ecoli_K12_MG1655_ASM584v2' "
        "(expected one of ('GCA_000005845.2', 'GCF_000005845.2'))"
    ):
        _assembly_reference(accession="GCA_000750555.1")
    with _refuses(
        "invalid assembly accession 'ASM584v2'; expected 'GCA_NNNNNNNNN.N' or "
        "'GCF_NNNNNNNNN.N'"
    ):
        _assembly_reference(accession="ASM584v2")
    # either member of the pair is accepted
    assert _assembly_reference(accession="GCF_000005845.2").assembly_accession == (
        "GCF_000005845.2"
    )


def test_the_assembly_pin_survives_only_where_the_field_is_re_annotated() -> None:
    """THE serialization finding, as a test rather than a comment.

    pydantic v2 serializes a field by its DECLARED type. An ``AssemblyReferenceGenome``
    in a slot annotated ``ReferenceGenome`` keeps the subclass as a python attribute but
    DUMPS only the base's three fields, and revalidating that dump yields a plain
    ``ReferenceGenome`` -- the assembly pin is gone, silently. That is why every
    bacterial reference class re-annotates ``genome_reference``, and why adding the pin
    to the base class instead would have been the only alternative (at a measured cost
    of 36 of 36 served dataset closures).
    """
    reference = _assembly_reference()
    base_slot = s.FitnessExperimentReference(
        dataset_name="t",
        genome_reference=reference,
        environment_reference=Environment(media=_lb()),
        phenotype_reference=FitnessPhenotype(fitness=1.0),
    )
    # the attribute keeps the subclass ...
    assert isinstance(base_slot.genome_reference, s.AssemblyReferenceGenome)
    # ... and the dump silently loses the pin
    dumped = base_slot.model_dump()["genome_reference"]
    assert set(dumped) == {"species", "strain", "ploidy"}
    assert "assembly_set" not in dumped
    revalidated = s.FitnessExperimentReference.model_validate(base_slot.model_dump())
    assert type(revalidated.genome_reference) is s.ReferenceGenome

    # the narrowed slot keeps it, through dict AND json
    narrowed = s.BacterialFitnessExperimentReference(
        dataset_name="t",
        genome_reference=reference,
        environment_reference=Environment(media=_lb()),
        phenotype_reference=FitnessPhenotype(fitness=1.0),
    )
    kept = narrowed.model_dump()["genome_reference"]
    assert kept["assembly_set"] == "ecoli_K12_MG1655_ASM584v2"
    assert kept["assembly_accession"] == "GCA_000005845.2"
    back = s.BacterialFitnessExperimentReference.model_validate_json(
        narrowed.model_dump_json()
    )
    assert back.genome_reference == reference
    assert type(back.genome_reference) is s.AssemblyReferenceGenome


@pytest.mark.parametrize(
    "kind",
    [
        "product_titer",
        "protein_turnover",
        "flux",
        "bacterial_fitness",
        "bacterial_environment_response",
        "bacterial_gene_interaction",
        "bacterial_gene_essentiality",
        "bacterial_protein_abundance",
        "bacterial_metabolite",
        "bacterial_rnaseq_expression",
        "bacterial_visual_score",
    ],
)
def test_every_bacterial_reference_pins_an_assembly(kind: str) -> None:
    """Each new family's reference declares the subclass, not the base.

    This is the property the whole pair exists for; the reconstruction path reaches the
    class through ``EXPERIMENT_REFERENCE_TYPE_MAP[tag]``, so it is checked through the
    map the way the loaders use it.
    """
    reference_cls = _reference_cls(kind)
    assert (
        reference_cls.model_fields["genome_reference"].annotation
        is s.AssemblyReferenceGenome
    )
    assert _experiment_cls(kind).model_fields["experiment_type"].default == kind
    assert reference_cls.model_fields["experiment_reference_type"].default == kind


# --- 3. the vocabularies agree with the deposited tier --------------------- #
def test_the_assembly_set_vocabulary_equals_the_registry_ids() -> None:
    """``schema.py`` restates the set ids; the registry owns them.

    The restatement exists because ``torchcell.sequence`` imports
    ``torchcell.datamodels``, so ``schema.py`` importing the registry would be a cycle.
    This test is what makes the restatement safe: the two cannot drift without failing
    here.
    """
    from torchcell.sequence.genome import registry

    assert set(typing.get_args(s.BacterialAssemblySet)) == {
        registry.ECOLI_K12_MG1655,
        registry.ECOLI_K12_BW25113,
        registry.PPUTIDA_KT2440,
    }
    assert s.BACTERIAL_ASSEMBLY_SETS == {
        "MG1655": registry.ECOLI_K12_MG1655,
        "BW25113": registry.ECOLI_K12_BW25113,
        "KT2440": registry.PPUTIDA_KT2440,
    }
    # every strain of the background vocabulary has a set, and every set an accession pair
    assert set(typing.get_args(s.BacterialReferenceStrain)) == set(
        s.BACTERIAL_ASSEMBLY_SETS
    )
    assert set(s.ASSEMBLY_SET_ACCESSIONS) == set(
        typing.get_args(s.BacterialAssemblySet)
    )
    for genbank, refseq in s.ASSEMBLY_SET_ACCESSIONS.values():
        assert genbank.startswith("GCA_") and refseq.startswith("GCF_")
        assert genbank.split("_")[1] == refseq.split("_")[1]


def test_the_namespace_vocabulary_covers_every_assembly_set() -> None:
    """One identifier namespace per deposited strain set, and no orphan on either side."""
    assert len(typing.get_args(s.BacterialGeneNamespace)) == len(
        typing.get_args(s.BacterialAssemblySet)
    )
    assert set(s.BACTERIAL_LOCUS_TAG_PATTERNS) == set(
        typing.get_args(s.BacterialGeneNamespace)
    )


@pytest.mark.data
def test_the_bw25113_genotype_is_verbatim_from_the_deposited_genbank_bytes() -> None:
    """The background-genotype constant is a substring of the sha256-pinned file.

    The constant's provenance is the tier, not a paper: it is read from the ``/note``
    qualifier of the ``source`` feature of BW25113's deposited GenBank flat file, whose
    bytes ``registry.resolve`` verifies against the manifest's sha256 on every call. So
    this reads the real bytes rather than trusting the string in ``schema.py``.

    Data-gated: it needs the genomes tier, so it runs under ``--data`` with the real
    ``DATA_ROOT`` exported (``tests/conftest.py`` otherwise points ``DATA_ROOT`` at a
    sentinel path, and ``registry`` deliberately has no fallback to a legacy location).
    """
    import gzip

    from torchcell.sequence.genome import registry

    path = registry.resolve(
        registry.ECOLI_K12_BW25113, "GCA_000750555.1_ASM75055v1_genomic.gbff.gz"
    )
    with gzip.open(path, "rt") as handle:
        header = handle.read(200_000)
    flattened = " ".join(header.split())
    assert s.BW25113_BACKGROUND_GENOTYPE in flattened
    assert f"genotype: {s.BW25113_BACKGROUND_GENOTYPE}" in flattened
    # the split-out lesions reconstruct the statement exactly, in order
    assert " ".join(s.BW25113_BACKGROUND_LESIONS) == s.BW25113_BACKGROUND_GENOTYPE


# --- 4. the bacterial strain background ------------------------------------ #
def _bw25113_allele(**kw: Any) -> s.BacterialBackgroundAllele:
    fields: dict[str, Any] = dict(
        systematic_gene_name="BW25113_3643",
        gene_namespace="ecoli_k12_bw25113_locus_tag",
        gene_name="rph",
        allele_name="rph-1",
        edit=s.AlleleEdit.sequence_variant,
        functional=False,
        provenance_gaps=[_gap("provenance")],
    )
    fields.update(kw)
    return s.BacterialBackgroundAllele(**fields)


def _bw25113_background(**kw: Any) -> s.BacterialStrainBackground:
    fields: dict[str, Any] = dict(
        name="BW25113",
        reference_strain="BW25113",
        assembly_set="ecoli_K12_BW25113_ASM75055v1",
        genotype_statement=s.BW25113_BACKGROUND_GENOTYPE,
        alleles=[_bw25113_allele()],
        provenance_gaps=[_gap("provenance")],
    )
    fields.update(kw)
    return s.BacterialStrainBackground(**fields)


def test_the_bacterial_background_is_a_sibling_not_a_widening() -> None:
    """The yeast classes are untouched: no bacterial strain is admissible there.

    Widening ``StrainBackground.reference_strain`` was measured to move 4 of 36 served
    closures (the chemogenomic loaders), so the bacterial background is a separate
    class and the yeast Literal still admits only S288C.
    """
    assert typing.get_args(
        s.StrainBackground.model_fields["reference_strain"].annotation
    ) == ("S288C",)
    with pytest.raises(ValidationError):
        s.StrainBackground(
            name="BW25113",
            reference_strain="BW25113",  # type: ignore[arg-type]
            mating_type=None,
            ploidy="haploid",
            provenance_gaps=[_gap("mating_type"), _gap("provenance")],
        )
    assert _bw25113_background().reference_strain == "BW25113"


def test_the_background_strain_and_assembly_set_cannot_disagree() -> None:
    """A background that named another strain's assembly would pin the wrong bytes."""
    with _refuses(
        "reference strain 'BW25113' is assembly set 'ecoli_K12_BW25113_ASM75055v1', "
        "not 'ecoli_K12_MG1655_ASM584v2'"
    ):
        _bw25113_background(assembly_set="ecoli_K12_MG1655_ASM584v2")


def test_a_background_allele_is_validated_in_its_own_namespace() -> None:
    """An allele's locus tag belongs to the namespace the allele declares."""
    with pytest.raises(ValidationError, match="Invalid bacterial locus tag"):
        _bw25113_allele(systematic_gene_name="YOR202W")
    with _refuses(
        "'b3643' is a ecoli_k12_mg1655_bnumber tag but gene_namespace is "
        "'ecoli_k12_bw25113_locus_tag'"
    ):
        _bw25113_allele(systematic_gene_name="b3643")


def test_a_background_allele_needs_a_cassette_exactly_when_it_is_a_replacement() -> (
    None
):
    with _refuses(
        "cassette is required for a cassette_replacement and forbidden for "
        "edit=sequence_variant"
    ):
        _bw25113_allele(cassette="FRT-kan-FRT")
    replacement = _bw25113_allele(
        edit=s.AlleleEdit.cassette_replacement, cassette="FRT-kan-FRT"
    )
    assert replacement.mechanism_so == ("SO:0000159", "deletion")


def test_a_haploid_background_carries_one_allele_entry_per_locus() -> None:
    """No zygosity means no compound heterozygote, so a second entry is a conflict."""
    with _refuses(
        "BW25113_3643: a haploid background carries one allele entry per locus"
    ):
        _bw25113_background(alleles=[_bw25113_allele(), _bw25113_allele()])


def test_background_functional_copies_is_haploid() -> None:
    background = _bw25113_background()
    assert background.functional_copies("BW25113_3643") == 0
    assert background.functional_copies("BW25113_0344") == 1
    assert background.alleles_at("BW25113_3643")[0].allele_name == "rph-1"
    assert not background.is_fully_sourced  # every element here is a declared gap


def test_the_background_rides_on_the_assembly_reference() -> None:
    """The background reaches a record through the reference, and must agree with it."""
    reference = _assembly_reference(
        strain="BW25113",
        assembly_set="ecoli_K12_BW25113_ASM75055v1",
        accession="GCA_000750555.1",
        background=_bw25113_background(),
    )
    assert reference.background is not None
    assert reference.background.genotype_statement == s.BW25113_BACKGROUND_GENOTYPE
    with _refuses("background name 'BW25113' != strain 'MG1655'"):
        _assembly_reference(
            strain="MG1655",
            assembly_set="ecoli_K12_BW25113_ASM75055v1",
            accession="GCA_000750555.1",
            background=_bw25113_background(),
        )


# --- 5. the three new phenotypes ------------------------------------------- #
def _titer(**kw: Any) -> s.ProductTiterPhenotype:
    fields: dict[str, Any] = dict(
        product=s.Compound(name="isoprenol", inchikey="XHQZJYCNDZAGLW-UHFFFAOYSA-N"),
        titer=2.4,
        titer_unit=s.ConcentrationUnit.g_per_l,
    )
    fields.update(kw)
    return s.ProductTiterPhenotype(**fields)


def test_the_titer_standard_error_is_derived_from_the_reported_uncertainty() -> None:
    """A sample SD of 0.3 over 9 replicates is an SE of 0.1, computed not stored."""
    phenotype = _titer(
        titer_uncertainty=0.3,
        titer_uncertainty_type=s.UncertaintyType.sample_sd,
        n_samples=9,
        sample_unit=s.SampleUnit.biological_replicate,
    )
    assert phenotype.titer_se == pytest.approx(0.1)
    # a bootstrap SE is already an SE and is used as-is
    assert _titer(
        titer_uncertainty=0.25, titer_uncertainty_type=s.UncertaintyType.bootstrap_se
    ).titer_se == pytest.approx(0.25)


def test_a_titer_uncertainty_may_never_be_unlabelled() -> None:
    with _refuses(
        "titer_uncertainty and titer_uncertainty_type must both be set or both be None "
        "(no unlabelled uncertainty)"
    ):
        _titer(titer_uncertainty=0.3)
    with _refuses("n_samples and sample_unit are required for sample_sd"):
        _titer(
            titer_uncertainty=0.3, titer_uncertainty_type=s.UncertaintyType.sample_sd
        )


def test_a_yield_or_productivity_without_its_unit_is_refused() -> None:
    """A bare number with no unit is not a measurement."""
    with _refuses(
        "product_yield and product_yield_unit must both be set or both be None "
        "(a bare number with no unit is not a measurement)"
    ):
        _titer(product_yield=0.12)
    with _refuses(
        "productivity and productivity_unit must both be set or both be None "
        "(a bare number with no unit is not a measurement)"
    ):
        _titer(productivity=0.05)
    ok = _titer(
        product_yield=0.12,
        product_yield_unit=s.ProductYieldUnit.g_per_g_substrate,
        productivity=0.05,
        productivity_unit=s.ProductivityUnit.g_per_l_per_h,
    )
    assert ok.product_yield_unit is s.ProductYieldUnit.g_per_g_substrate
    assert ok.productivity_unit is s.ProductivityUnit.g_per_l_per_h


def test_a_titer_is_a_finite_non_negative_amount() -> None:
    with _refuses("titer must be a finite number"):
        _titer(titer=float("nan"))
    with _refuses("titer must be non-negative, got -1.0"):
        _titer(titer=-1.0)
    assert _titer(titer=0.0).titer == 0.0  # a strain that made none is a real datum


def test_the_product_is_a_typed_compound_so_a_titer_joins_the_compound_layer() -> None:
    """The product is the same entity a chemogenomic dataset dosing it would use."""
    phenotype = _titer()
    assert phenotype.product.inchikey == "XHQZJYCNDZAGLW-UHFFFAOYSA-N"
    assert phenotype.label_name == "titer"
    assert phenotype.label_statistic_name == "titer_se"
    assert phenotype.graph_level == "global"


def test_protein_turnover_keys_its_replicates_on_the_same_proteins() -> None:
    fields: dict[str, Any] = dict(
        degradation_rate={"b0002": 0.11, "b0003": 0.07},
        n_replicates={"b0002": 3, "b0003": 3},
        measurement_type="pulse_silac_degradation_rate_per_hour",
    )
    phenotype = s.ProteinTurnoverPhenotype(**fields)
    assert phenotype.label_name == "degradation_rate"
    with _refuses("n_replicates keys must match degradation_rate keys"):
        s.ProteinTurnoverPhenotype(**{**fields, "n_replicates": {"b0002": 3}})
    with _refuses("degradation_rate cannot be empty"):
        s.ProteinTurnoverPhenotype(
            **{**fields, "degradation_rate": {}, "n_replicates": {}}
        )
    with _refuses("degradation_rate for b0002 must be finite and non-negative"):
        s.ProteinTurnoverPhenotype(
            **{**fields, "degradation_rate": {"b0002": -0.1, "b0003": 0.07}}
        )
    with _refuses("half_life key b9999 not in degradation_rate"):
        s.ProteinTurnoverPhenotype(**{**fields, "half_life": {"b9999": 6.0}})


def test_a_flux_record_states_an_interval_rather_than_one_statistic() -> None:
    """A fitted flux's uncertainty is two-sided, so there is no single label statistic."""
    fields: dict[str, Any] = dict(
        net_flux={"PGI": -1.2},
        net_flux_lower={"PGI": -1.5},
        net_flux_upper={"PGI": -0.9},
        confidence_level=0.95,
        measurement_type="c13_mfa_net_flux_mmol_per_gdcw_per_h",
    )
    phenotype = s.FluxPhenotype(**fields)
    assert phenotype.label_name == "net_flux"
    assert phenotype.label_statistic_name is None
    assert phenotype.graph_level == "metabolism"
    # a net flux is SIGNED: nothing is clamped, because the sign is the direction
    assert phenotype.net_flux["PGI"] == -1.2


def test_flux_bounds_must_bracket_the_fitted_flux() -> None:
    fields: dict[str, Any] = dict(
        net_flux={"PGI": -1.2}, measurement_type="c13_mfa_net_flux_mmol_per_gdcw_per_h"
    )
    with _refuses("net_flux_lower for PGI exceeds the fitted flux (-1.0 > -1.2)"):
        s.FluxPhenotype(**{**fields, "net_flux_lower": {"PGI": -1.0}})
    with _refuses("net_flux_upper for PGI is below the fitted flux (-1.5 < -1.2)"):
        s.FluxPhenotype(**{**fields, "net_flux_upper": {"PGI": -1.5}})
    with _refuses("net_flux bounds for PGI are inverted: -0.9 > -1.5"):
        s.FluxPhenotype(
            **{
                **fields,
                "net_flux_lower": {"PGI": -0.9},
                "net_flux_upper": {"PGI": -1.5},
            }
        )
    with _refuses("net_flux_lower key TPI not in net_flux"):
        s.FluxPhenotype(**{**fields, "net_flux_lower": {"TPI": -2.0}})
    with _refuses("confidence_level is a fraction in (0, 1), got 95.0"):
        s.FluxPhenotype(**{**fields, "confidence_level": 95.0})


# --- 6. the new experiment families reconstruct through the maps ----------- #
def test_a_product_titer_record_round_trips_through_the_type_maps() -> None:
    """The reconstruction path the loaders use resolves the new tags to these classes."""
    experiment = s.ProductTiterExperiment(
        dataset_name="Toy",
        genotype=Genotype(
            perturbations=[
                s.HeterologousPathwayPerturbation(
                    systematic_gene_name="Efa:mvaE",
                    perturbed_gene_name="mvaE",
                    gene_namespace="pputida_kt2440_locus_tag",
                    source_organism="Enterococcus faecalis",
                    is_heterologous=True,
                    localization="chromosomal_integration",
                    pathway_name="isoprenol via mevalonate",
                )
            ]
        ),
        environment=Environment(media=_lb()),
        phenotype=_titer(),
    )
    reference = s.ProductTiterExperimentReference(
        dataset_name="Toy",
        genome_reference=_assembly_reference(
            strain="KT2440",
            assembly_set="pputida_KT2440_ASM756v2",
            accession="GCA_000007565.2",
        ),
        environment_reference=Environment(media=_lb()),
        phenotype_reference=_titer(titer=0.0),
    )
    kind = experiment.experiment_type
    assert kind == "product_titer"
    assert _experiment_cls(kind).model_validate(experiment.model_dump()) == experiment
    rebuilt = _reference_cls(kind).model_validate(reference.model_dump())
    assert rebuilt == reference
    assert isinstance(rebuilt.genome_reference, s.AssemblyReferenceGenome)
    assert rebuilt.genome_reference.assembly_set == "pputida_KT2440_ASM756v2"


def _bacterial_fitness_experiment() -> s.BacterialFitnessExperiment:
    return s.BacterialFitnessExperiment(
        dataset_name="Toy",
        genotype=Genotype(perturbations=[s.BacterialDeletionPerturbation(**_MG1655)]),
        environment=Environment(media=_lb()),
        phenotype=FitnessPhenotype(fitness=0.82),
    )


def test_a_reused_phenotype_family_reconstructs_through_the_tag_not_the_union() -> None:
    """The tag is the reconstruction key, and for these families it has to be.

    An assembly-pinned family whose phenotype is REUSED has exactly the same field set
    on the experiment side as its yeast sibling -- only the reference differs -- and
    ``experiment_type`` is a plain ``str`` on every experiment class rather than a
    ``Literal``. So the undiscriminated ``ExperimentType`` union cannot tell the two
    apart and pydantic's smart mode may return the sibling, while the tag survives
    unchanged. Every reconstruction path in the codebase (the raw query loader, the
    deduplicator, the aggregator, ``Neo4jCellDataset``) keys on
    ``EXPERIMENT_TYPE_MAP[experiment_type]`` for precisely this reason, so that is the
    path pinned here. The stored record is identical either way; what the map buys is
    the exact class.
    """
    experiment = _bacterial_fitness_experiment()
    dumped = experiment.model_dump()
    assert dumped["experiment_type"] == "bacterial_fitness"

    rebuilt = _experiment_cls("bacterial_fitness").model_validate(dumped)
    assert type(rebuilt) is s.BacterialFitnessExperiment
    assert rebuilt == experiment
    assert isinstance(rebuilt, s.BacterialFitnessExperiment)
    assert rebuilt.phenotype.fitness == pytest.approx(0.82)
    assert isinstance(rebuilt.genotype, Genotype)
    assert rebuilt.genotype.systematic_gene_names == ["b0002"]

    # the union keeps every value and the tag, and may widen the class
    adapter: TypeAdapter[Any] = TypeAdapter(ExperimentType)
    through_union: Any = adapter.validate_python(dumped)
    assert through_union.model_dump() == dumped
    assert through_union.experiment_type == "bacterial_fitness"


def test_a_new_phenotype_family_is_discriminated_by_its_own_fields() -> None:
    """A family with a NEW phenotype class does resolve through the union.

    ``ProductTiterExperiment`` carries fields no other experiment has, so unlike the
    reused families it is unambiguous even without a tag discriminator. Pinning both
    behaviors keeps the difference between them visible.
    """
    experiment = s.ProductTiterExperiment(
        dataset_name="Toy",
        genotype=Genotype(perturbations=[s.BacterialDeletionPerturbation(**_MG1655)]),
        environment=Environment(media=_lb()),
        phenotype=_titer(),
    )
    titer_adapter: TypeAdapter[Any] = TypeAdapter(ExperimentType)
    back: Any = titer_adapter.validate_python(experiment.model_dump())
    assert isinstance(back, s.ProductTiterExperiment)
    assert back.phenotype.titer == pytest.approx(2.4)
    assert back.phenotype.product.name == "isoprenol"


@pytest.mark.parametrize(
    "kind",
    [
        "product_titer",
        "protein_turnover",
        "flux",
        "bacterial_fitness",
        "bacterial_environment_response",
        "bacterial_gene_interaction",
        "bacterial_gene_essentiality",
        "bacterial_protein_abundance",
        "bacterial_metabolite",
        "bacterial_rnaseq_expression",
        "bacterial_visual_score",
    ],
)
def test_every_new_family_is_in_both_maps_and_both_unions(kind: str) -> None:
    """A family missing from a map or a union is unreconstructible from the store."""
    experiment_cls = _experiment_cls(kind)
    reference_cls = _reference_cls(kind)
    assert experiment_cls in typing.get_args(ExperimentType)
    assert reference_cls in typing.get_args(s.ExperimentReferenceType)
    # the pair measures the same phenotype class, or the control is not a control
    assert (
        experiment_cls.model_fields["phenotype"].annotation
        is reference_cls.model_fields["phenotype_reference"].annotation
    )
    # the medium stays the shared Environment, so the cross-host medium join still forms
    assert experiment_cls.model_fields["environment"].annotation is Environment
