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
