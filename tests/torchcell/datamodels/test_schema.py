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


# --- #901/#902/#903: a typed gap on the medium --------------------------------- #
def _label_gap(field: str) -> ProvenanceGap:
    return ProvenanceGap(
        field=field, reason=ProvenanceGapReason.not_carried_by_curation
    )


def test_a_medium_label_is_stated_or_gapped_never_silently_none() -> None:
    """``name``, ``state`` and ``is_synthetic`` are each stated or ``None`` beside a
    gap naming that field; a gap on another field does not cover it, and a gapped field
    cannot also carry a value.
    """
    unstated = s.Media(
        name=None,
        state=None,
        is_synthetic=None,
        provenance_gaps=[_label_gap(f) for f in ("name", "state", "is_synthetic")],
    )
    assert unstated.gapped_fields() == {"name", "state", "is_synthetic"}
    assert s.Media.model_validate(unstated.model_dump()) == unstated
    stateless = s.Media(
        name="YPD with 2% glucose",
        state=None,
        is_synthetic=False,
        provenance_gaps=[_label_gap("state")],
    )
    assert (stateless.name, stateless.state) == ("YPD with 2% glucose", None)
    with _refuses("Media.name is unset and carries no ProvenanceGap"):
        s.Media(name=None, state="solid", is_synthetic=False)
    with _refuses("Media.state is unset and carries no ProvenanceGap"):
        s.Media(
            name="YPD",
            state=None,
            is_synthetic=None,
            provenance_gaps=[_label_gap("is_synthetic")],
        )
    with _refuses("field 'state' has a ProvenanceGap but is not None"):
        s.Media(
            name="YPD",
            state="solid",
            is_synthetic=False,
            provenance_gaps=[_label_gap("state")],
        )
    with _refuses('state must be one of "solid", "liquid", or "gas"'):
        s.Media(name="X", state="plasma", is_synthetic=None)


def test_a_gapped_medium_has_no_stated_name_and_environment_media_stays_required() -> (
    None
):
    """``stated_name`` refuses a gapped name by name; ``Environment.media`` is still a
    required ``Media``, so an unstated medium is a gapped ``Media``, never ``None``.
    """
    assert YEPD.stated_name == "YEPD"
    unnamed = s.Media(
        name=None,
        state=None,
        is_synthetic=None,
        provenance_gaps=[_label_gap(f) for f in ("name", "state", "is_synthetic")],
    )
    with pytest.raises(ValueError, match="this medium's name is a typed ProvenanceGap"):
        _ = unnamed.stated_name
    assert s.Environment.model_fields["media"].annotation is s.Media
    assert s.Environment.model_fields["media"].is_required()
    with pytest.raises(
        ValidationError, match="Input should be a valid dictionary or instance of Media"
    ):
        s.Environment(media=None)  # type: ignore[arg-type]  # required, never None
    assert s.Environment(media=unnamed).media.gapped_fields() == {
        "name",
        "state",
        "is_synthetic",
    }


def test_environment_aerobicity_vocabulary() -> None:
    """The four oxygen regimes pass (#1045 adds ``transition``); ``hypoxic`` is
    refused.
    """
    for regime in ["aerobic", "anaerobic", "microaerobic", "transition"]:
        assert s.Environment(media=YEPD, aerobicity=regime).aerobicity == regime
    with _refuses(
        "aerobicity must be aerobic/anaerobic/microaerobic/transition, got 'hypoxic'"
    ):
        s.Environment(media=YEPD, aerobicity="hypoxic")


def test_an_unstated_oxygen_regime_is_a_typed_gap_never_a_silent_none() -> None:
    """#1045: ``aerobicity`` is a value or a typed gap; the default stays ``aerobic``."""
    assert s.Environment(media=YEPD).aerobicity == "aerobic"
    with _refuses("Environment.aerobicity is unset and carries no ProvenanceGap"):
        s.Environment(media=YEPD, aerobicity=None)
    gap = ProvenanceGap(
        field="aerobicity",
        reason=ProvenanceGapReason.not_reported_by_primary,
        note="the cell is blank",
    )
    gapped = s.Environment(media=YEPD, aerobicity=None, provenance_gaps=[gap])
    assert gapped.aerobicity is None and gapped.gapped_fields() == {"aerobicity"}
    assert s.Environment.model_validate(gapped.model_dump()) == gapped
    with _refuses("field 'aerobicity' has a ProvenanceGap but is not None"):
        s.Environment(media=YEPD, aerobicity="aerobic", provenance_gaps=[gap])


def test_a_growth_phase_and_a_dissolved_oxygen_setpoint_tell_environments_apart() -> (
    None
):
    """#1045: ``growth_phase`` is optional; a DO setpoint is a physical-factor edit."""
    base = s.Environment(media=YEPD)
    assert base.growth_phase is None
    held = s.Environment(
        media=YEPD, growth_phase=s.CultureGrowthPhase.stationary, duration_hours=72.0
    )
    assert held.model_dump_json() != base.model_dump_json()
    assert s.Environment.model_validate(held.model_dump()).growth_phase is (
        s.CultureGrowthPhase.stationary
    )
    with pytest.raises(
        ValidationError, match="Input should be 'exponential' or 'stationary'"
    ):
        s.Environment(media=YEPD, growth_phase="lag")  # type: ignore[arg-type]
    setpoint = s.EnvironmentPhysicalPerturbation(
        factor=s.PhysicalFactor.dissolved_oxygen,
        magnitude=s.Concentration(value=30.0, unit=s.ConcentrationUnit.percent),
        provenance_gaps=[
            ProvenanceGap(
                field="agent",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="the gas that held the setpoint is not stated",
            )
        ],
    )
    controlled = s.Environment(media=YEPD, perturbations=[setpoint])
    assert controlled.model_dump_json() != base.model_dump_json()
    assert s.PhysicalFactor("dissolved_oxygen") is s.PhysicalFactor.dissolved_oxygen


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


def test_a_publication_defaults_to_a_journal_article_with_nothing_else_stated() -> None:
    """The default keeps every record written before ``source_type`` existed valid."""
    publication = s.Publication(doi="10.1/x", doi_url="https://doi.org/10.1/x")
    assert publication.source_type is s.SourceType.journal_article
    assert publication.title is None
    assert publication.identifier is None
    assert publication.identifier_url is None


_DISSERTATION_ID = "si/thesis.pdf sha256:" + "a" * 64


@pytest.mark.parametrize(
    "source_type",
    [s.SourceType.dissertation, s.SourceType.preliminary_report, s.SourceType.in_house],
)
def test_a_non_journal_source_is_identified_by_title_plus_deposited_document(
    source_type: s.SourceType,
) -> None:
    """No DOI, no URL: the identity is the title plus the mirrored document's path+hash."""
    publication = s.Publication(
        source_type=source_type,
        title="Engineering furfural tolerance",
        identifier=_DISSERTATION_ID,
        identifier_url="https://example.invalid/thesis.pdf",
    )
    assert publication.doi is None
    assert publication.pubmed_id is None
    assert publication.identifier == _DISSERTATION_ID
    with _refuses(f"a {source_type.value} source requires a title"):
        s.Publication(source_type=source_type, identifier=_DISSERTATION_ID)
    with _refuses(f"a {source_type.value} source requires a title"):
        s.Publication(source_type=source_type, title="   ", identifier=_DISSERTATION_ID)
    with _refuses(f"a {source_type.value} source requires an identifier"):
        s.Publication(source_type=source_type, title="Engineering furfural tolerance")


def test_a_non_journal_source_may_also_carry_a_doi_without_being_required_to() -> None:
    """doi/pubmed go optional, not forbidden: a thesis later deposited with a DOI fits."""
    publication = s.Publication(
        source_type=s.SourceType.dissertation,
        title="Engineering furfural tolerance",
        identifier=_DISSERTATION_ID,
        doi="10.1/thesis",
        doi_url="https://doi.org/10.1/thesis",
    )
    assert publication.doi == "10.1/thesis"


def test_source_type_vocabulary_is_closed_and_free_of_aliases() -> None:
    """``a = "x"`` then ``b = "x"`` would collapse two kinds of source silently."""
    assert {t.value for t in s.SourceType} == {
        "journal_article",
        "dissertation",
        "preliminary_report",
        "in_house",
    }
    assert len(list(s.SourceType)) == len({t.value for t in s.SourceType})


def test_relative_growth_rate_is_its_own_measurement_type() -> None:
    """A wild-type-relative generation-time ratio is not an absolute growth rate."""
    assert s.MeasurementType.relative_growth_rate.value == "relative_growth_rate"
    # A rate and a ratio against the run's wild type are different numbers; keeping
    # them as separate members is what stops one being averaged with the other.
    assert {"growth_rate", "relative_growth_rate"} <= {
        m.value for m in s.MeasurementType
    }
    assert len(list(s.MeasurementType)) == len({m.value for m in s.MeasurementType})
    assert s.MeasurementType.relative_growth_rate not in s.CATEGORICAL_MEASUREMENT_TYPES


def test_condition_growth_ratio_is_a_same_strain_ratio_member() -> None:
    """Peter 2018's growth ratio over the same isolate's control plate is its own member.

    It is a RATIO (control = 1.0), so it sits in ``RATIO_MEASUREMENT_TYPES`` beside the
    wild-type-relative rate, and in neither the categorical nor the absolute set.
    """
    member = s.MeasurementType.condition_growth_ratio
    assert member.value == "condition_growth_ratio"
    assert member in s.RATIO_MEASUREMENT_TYPES
    assert member not in s.ABSOLUTE_MEASUREMENT_TYPES
    assert member not in s.CATEGORICAL_MEASUREMENT_TYPES
    phenotype = s.EnvironmentResponsePhenotype(
        measurement_type=member, environment_response=0.0
    )
    assert phenotype.environment_response == 0.0


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
        # #964: None is admitted only beside a typed gap on it.
        (
            {"expression_count": None},
            "expression_count is None without a ProvenanceGap",
        ),
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
                "cell",
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
                "dissolved_oxygen",
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
    (
        s.BacterialMarkedAllelePerturbation,
        {
            **_MG1655,
            "cassette": "kan",
            "insertion_site": "3'-UTR",
            "tag": "SPA",
            "terminus": "C",
            "allele_effect": "hypomorphic",
        },
    ),
    (s.BacterialDegronPerturbation, {**_MG1655, "degron": "DAS+4", "terminus": "C"}),
    (
        s.BacterialCrisprActivationPerturbation,
        {**_MG1655, "crispr": s.CrisprConstruct(effector="dCas9*-MCPSoxS")},
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


def test_the_rel606_namespace_owns_its_tags_and_no_other_namespace_does() -> None:
    """E. coli B REL606's five-digit ``ECB_`` tags (numbered, tRNA, rRNA) match only its
    own namespace, and no REL606 pattern match is a tag of the other three or a yeast
    name. A four-digit ``ECB_`` and a RefSeq ``ECB_RS`` tag are not GenBank tags.
    """
    pattern = s.BACTERIAL_LOCUS_TAG_PATTERNS["ecoli_b_rel606_locus_tag"]
    assert pattern == r"^ECB_[rt]?\d{5}$"
    for tag in ("ECB_00001", "ECB_04279", "ECB_t00085", "ECB_r00022"):
        matching = {
            namespace
            for namespace, other in s.BACTERIAL_LOCUS_TAG_PATTERNS.items()
            if re.match(other, tag)
        }
        assert matching == {"ecoli_b_rel606_locus_tag"}, tag
    for tag in (
        "b0001",
        "BW25113_0001",
        "PP_0001",
        "YAL001C",
        "ECB_0001",
        "ECB_RS00005",
    ):
        assert not re.match(pattern, tag), tag
    leaf = s.BacterialDeletionPerturbation(
        systematic_gene_name="ECB_00002",
        perturbed_gene_name="thrA",
        gene_namespace="ecoli_b_rel606_locus_tag",
    )
    assert leaf.gene_namespace == "ecoli_b_rel606_locus_tag"
    with pytest.raises(ValidationError, match="is a ecoli_b_rel606_locus_tag tag"):
        s.BacterialDeletionPerturbation(
            systematic_gene_name="ECB_00002",
            perturbed_gene_name="thrA",
            gene_namespace="ecoli_k12_mg1655_bnumber",
        )


def test_the_w3110_namespace_owns_its_refseq_tags_and_no_other_namespace_does() -> None:
    """W3110's ``Y75_RS`` tags (#910) match only its namespace, no other namespace's tag
    or a yeast name matches it, and a pre-2010 ``Y75_p`` tag is not one of them.
    """
    pattern = s.BACTERIAL_LOCUS_TAG_PATTERNS["ecoli_k12_w3110_refseq_locus_tag"]
    assert pattern == r"^Y75_RS\d{5}$"
    for tag in ("Y75_RS00005", "Y75_RS25635"):
        matching = {
            namespace
            for namespace, other in s.BACTERIAL_LOCUS_TAG_PATTERNS.items()
            if re.match(other, tag)
        }
        assert matching == {"ecoli_k12_w3110_refseq_locus_tag"}, tag
    for tag in (
        "b0001",
        "BW25113_0001",
        "PP_0001",
        "ECB_00001",
        "YAL001C",
        "Y75_p0001",
    ):
        assert not re.match(pattern, tag), tag


def test_a_pooled_tn_gene_call_is_a_released_strain_of_the_transposon_kind() -> None:
    """``transposon_insertion`` (#910) carries the SO pair the placed Tn leaf pins, and a
    W3110 symbol stored as released round-trips through a genotype.
    """
    kind = s.ReleasedStrainKind.transposon_insertion
    assert s.RELEASED_STRAIN_KIND_SO[kind] == ("SO:0001218", "transgenic_insertion")
    placed = s.TransposonInsertionPerturbation.model_fields["mechanism_so_id"].default
    assert s.RELEASED_STRAIN_KIND_SO[kind][0] == placed
    leaf = s.BacterialReleasedStrainPerturbation(
        systematic_gene_name="rhoL",
        perturbed_gene_name="rhoL",
        gene_namespace="ecoli_k12_w3110_refseq_locus_tag",
        strain_label="rhoL",
        identifier_kind="Gene name",
        released_kind=kind,
        cassette="kanamycin",
    )
    assert leaf.mechanism_so == ("SO:0001218", "transgenic_insertion")
    genotype = s.Genotype(perturbations=[leaf])
    assert s.Genotype.model_validate(genotype.model_dump()) == genotype
    with pytest.raises(ValidationError, match="is a bacterial locus tag"):
        leaf.model_validate(
            {**leaf.model_dump(), "systematic_gene_name": "Y75_RS00005"}
        )


def test_a_w3110_reference_names_the_refseq_accession_of_its_set() -> None:
    """The W3110 set's pair is GCA_000010245.1 / GCF_000010245.2; another is refused."""
    reference = s.AssemblyReferenceGenome(
        species="Escherichia coli",
        strain="W3110",
        assembly_set="ecoli_K12_W3110_ASM1024v1",
        assembly_accession="GCF_000010245.2",
    )
    assert reference.assembly_accession == "GCF_000010245.2"
    with pytest.raises(ValidationError, match="is not an accession of"):
        s.AssemblyReferenceGenome(
            species="Escherichia coli",
            strain="W3110",
            assembly_set="ecoli_K12_W3110_ASM1024v1",
            assembly_accession="GCF_000010245.1",
        )


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
    """All eight in one genotype, so the union resolves each tag unambiguously."""
    leaves: list[Any] = [cls(**kwargs) for cls, kwargs in _BACTERIAL_LEAF_CASES]
    genotype = Genotype(
        perturbations=[
            leaf.model_copy(update={"perturbed_gene_name": f"g{index}"})
            for index, leaf in enumerate(leaves)
        ]
    )
    back = Genotype.model_validate(genotype.model_dump())
    assert {type(p) for p in back.perturbations} == {type(leaf) for leaf in leaves}
    assert len(back) == 8


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
        "protein_synthesis_rate",
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
        registry.ECOLI_B_REL606,
        registry.ECOLI_K12_W3110,
    }
    assert s.BACTERIAL_ASSEMBLY_SETS == {
        "MG1655": registry.ECOLI_K12_MG1655,
        "BW25113": registry.ECOLI_K12_BW25113,
        "KT2440": registry.PPUTIDA_KT2440,
        "REL606": registry.ECOLI_B_REL606,
        "W3110": registry.ECOLI_K12_W3110,
    }
    # every strain of the background vocabulary has a set, and every set an accession pair
    assert set(typing.get_args(s.BacterialReferenceStrain)) == set(
        s.BACTERIAL_ASSEMBLY_SETS
    )
    assert set(s.ASSEMBLY_SET_ACCESSIONS) == set(
        typing.get_args(s.BacterialAssemblySet)
    )
    # The two accessions share the assembly number; the versions can differ (W3110 pairs
    # GCA_000010245.1 with GCF_000010245.2, read off its GCF assembly report, #910).
    for genbank, refseq in s.ASSEMBLY_SET_ACCESSIONS.values():
        assert genbank.startswith("GCA_") and refseq.startswith("GCF_")
        assert genbank.split("_")[1].split(".")[0] == refseq.split("_")[1].split(".")[0]
    assert s.ASSEMBLY_SET_ACCESSIONS["ecoli_K12_W3110_ASM1024v1"] == (
        "GCA_000010245.1",
        "GCF_000010245.2",
    )


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


# --- begin #867: a below-detection titer is a bound, not a zero ------------- #
def test_a_below_detection_titer_is_a_left_censored_floor() -> None:
    """The stored number is the floor; the side says the truth lies below it."""
    bound = _titer(titer=0.155, titer_censoring=s.Censoring.left)
    assert (bound.titer, bound.titer_censoring) == (0.155, s.Censoring.left)
    assert bound.titer_se is None
    assert _titer().titer_censoring is None  # unstated, distinct from 'uncensored'
    stated = _titer(titer_censoring=s.Censoring.uncensored)
    assert stated.titer_censoring is s.Censoring.uncensored


def test_a_left_censored_titer_of_zero_is_refused() -> None:
    """A floor of 0.0 states an exact zero, which is the measurement nobody made."""
    with _refuses(
        "a left-censored titer stores the detection floor it lies below; a floor of "
        "0.0 states an exact zero, which is a measurement, not a bound"
    ):
        _titer(titer=0.0, titer_censoring=s.Censoring.left)


@pytest.mark.parametrize("side", [s.Censoring.left, s.Censoring.right])
def test_a_censored_titer_carries_no_error(side: s.Censoring) -> None:
    """A bound has no spread: neither a reported uncertainty nor a derived SE."""
    with _refuses(
        f"a {side.value}-censored titer is a bound, so it carries no "
        "titer_uncertainty and no titer_se"
    ):
        _titer(
            titer_censoring=side,
            titer_uncertainty=0.25,
            titer_uncertainty_type=s.UncertaintyType.bootstrap_se,
        )
    with _refuses(
        f"a {side.value}-censored titer is a bound, so it carries no "
        "titer_uncertainty and no titer_se"
    ):
        _titer(titer_censoring=side, titer_se=0.1)


# --- end #867 ------------------------------------------------------------------ #


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


# --- begin #857: the protein synthesis-rate family ---
def _synthesis_fields() -> dict[str, Any]:
    return dict(
        synthesis_rate={"b0002": 120.0, "b0003": 0.0},
        rate_unit=s.SynthesisRateUnit.molecules_per_generation,
        generation_time_minutes=21.5,
        measurement_type="ribosome_profiling_footprint_density",
    )


def test_a_synthesis_rate_needs_no_degradation_rate() -> None:
    """The label is the synthesis rate; nothing about degradation is required."""
    phenotype = s.ProteinSynthesisRatePhenotype(**_synthesis_fields())
    assert phenotype.label_name == "synthesis_rate"
    assert phenotype.label_statistic_name == "synthesis_rate_se"
    assert phenotype.graph_level == "node"
    assert phenotype.synthesis_rate == {"b0002": 120.0, "b0003": 0.0}
    assert phenotype.n_replicates is None
    assert phenotype.synthesis_rate_se is None
    assert phenotype.censoring is None
    assert "degradation_rate" not in s.ProteinSynthesisRatePhenotype.model_fields


def test_a_synthesis_rate_record_refuses_what_it_cannot_mean() -> None:
    fields = _synthesis_fields()
    with _refuses("synthesis_rate cannot be empty"):
        s.ProteinSynthesisRatePhenotype(**{**fields, "synthesis_rate": {}})
    with _refuses("synthesis_rate for b0002 must be finite and non-negative"):
        s.ProteinSynthesisRatePhenotype(**{**fields, "synthesis_rate": {"b0002": -1.0}})
    with _refuses("synthesis_rate for b0002 must be finite and non-negative"):
        s.ProteinSynthesisRatePhenotype(
            **{**fields, "synthesis_rate": {"b0002": math.nan}}
        )
    with _refuses("synthesis_rate_se key b9999 not in synthesis_rate"):
        s.ProteinSynthesisRatePhenotype(
            **{**fields, "synthesis_rate_se": {"b9999": 1.0}}
        )
    with _refuses("synthesis_rate_se for b0002 must be non-negative"):
        s.ProteinSynthesisRatePhenotype(
            **{**fields, "synthesis_rate_se": {"b0002": -1.0}}
        )
    with _refuses("n_replicates keys must match synthesis_rate keys"):
        s.ProteinSynthesisRatePhenotype(**{**fields, "n_replicates": {"b0002": 2}})
    with _refuses("n_replicates for b0003 must be >= 1"):
        s.ProteinSynthesisRatePhenotype(
            **{**fields, "n_replicates": {"b0002": 2, "b0003": 0}}
        )
    with _refuses("censoring key b9999 not in synthesis_rate"):
        s.ProteinSynthesisRatePhenotype(
            **{**fields, "censoring": {"b9999": s.Censoring.right}}
        )
    with _refuses("generation_time_minutes must be finite and positive, got 0.0"):
        s.ProteinSynthesisRatePhenotype(**{**fields, "generation_time_minutes": 0.0})
    with _refuses("a rate per generation states its generation time"):
        s.ProteinSynthesisRatePhenotype(**{**fields, "generation_time_minutes": None})


def test_a_rate_per_hour_needs_no_generation_time_and_keeps_its_companions() -> None:
    phenotype = s.ProteinSynthesisRatePhenotype(
        **{
            **_synthesis_fields(),
            "rate_unit": s.SynthesisRateUnit.molecules_per_hour,
            "generation_time_minutes": None,
            "synthesis_rate_se": {"b0002": math.nan},
            "n_replicates": {"b0002": 2, "b0003": 1},
            "censoring": {"b0003": s.Censoring.left},
        }
    )
    assert phenotype.rate_unit is s.SynthesisRateUnit.molecules_per_hour
    assert phenotype.n_replicates == {"b0002": 2, "b0003": 1}
    assert phenotype.censoring == {"b0003": s.Censoring.left}


# --- end #857 ---


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
    assert phenotype.net_flux is not None
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
        environment=s.CultureEnvironment(
            media=_lb(),
            culture_format=s.CultureFormat(
                vessel="48-well BioLector flower plate",
                working_volume_ul=1500.0,
                shaking_rpm=1000.0,
            ),
        ),
        phenotype=_titer(),
    )
    reference = s.ProductTiterExperimentReference(
        dataset_name="Toy",
        genome_reference=_assembly_reference(
            strain="KT2440",
            assembly_set="pputida_KT2440_ASM756v2",
            accession="GCA_000007565.2",
        ),
        environment_reference=s.CultureEnvironment(media=_lb()),
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
        environment=s.CultureEnvironment(media=_lb()),
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
        "protein_synthesis_rate",
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
    # The medium stays the shared Environment, so the cross-host medium join still
    # forms. The product-titer family is the one narrowing: a titer is read with its
    # vessel, so it declares CultureEnvironment (a subclass of Environment, holding the
    # same `media` field, so the medium-level join is unchanged).
    annotation = experiment_cls.model_fields["environment"].annotation
    if kind == "product_titer":
        assert annotation is s.CultureEnvironment
    else:
        assert annotation is Environment
    assert annotation is not None and issubclass(annotation, Environment)
    assert annotation.model_fields["media"].annotation is s.Media


# --------------------------------------------------------------------------- #
# 7. The follow-ups the first bacterial loaders asked for
# --------------------------------------------------------------------------- #
def test_a_phage_is_an_environment_perturbation_of_its_own_leaf() -> None:
    """A virion is none of the other three leaves, so it has its own.

    A small molecule is keyed by InChIKey (a virion has none), ``PhysicalFactor`` is a
    scalar variable, and ``BiologicAgentClass`` is peptide / protein / antibody / toxin.
    The dose is a dimensionless particle-to-cell ratio in its own field, never a
    ``Concentration``.
    """
    phage = s.PhagePerturbation(
        name="T4",
        family="Myoviridae",
        genome_type="dsDNA",
        ncbi_taxid=10665,
        genome_accession="AF158101.6",
        multiplicity_of_infection=0.01875,
        titer_pfu_per_ml=1.2e9,
        host_of_propagation="E. coli K-12 BW25113",
    )
    assert phage.perturbation_type == "phage"
    assert s.PhagePerturbation in typing.get_args(s.EnvironmentPerturbationType)
    assert isinstance(phage, s.EnvironmentPerturbation)
    # the dose is not a Concentration and carries no ConcentrationUnit
    assert "concentration" not in s.PhagePerturbation.model_fields
    assert (
        s.PhagePerturbation.model_fields["multiplicity_of_infection"].annotation
        == float | None
    )
    # it round-trips inside an environment, through the union
    environment = Environment(media=_lb(), perturbations=[phage])
    adapter: TypeAdapter[Any] = TypeAdapter(Environment)
    back = adapter.validate_python(environment.model_dump())
    assert back.perturbations[0] == phage


def test_a_phage_dose_is_stated_or_gapped_and_never_zero() -> None:
    """A phage challenge always has a dose, so an unstated MOI is a typed gap."""
    with _refuses(
        "PhagePerturbation.multiplicity_of_infection is unset and carries no "
        "ProvenanceGap (an unsourced element must be a typed gap, never a silent None)"
    ):
        s.PhagePerturbation(name="T4")
    gapped = s.PhagePerturbation(
        name="T4",
        provenance_gaps=[
            ProvenanceGap(
                field="multiplicity_of_infection",
                reason=ProvenanceGapReason.not_reported_by_primary,
            )
        ],
    )
    assert gapped.multiplicity_of_infection is None
    with _refuses("PhagePerturbation.multiplicity_of_infection must be > 0, got 0.0"):
        s.PhagePerturbation(name="T4", multiplicity_of_infection=0.0)
    with _refuses("PhagePerturbation.titer_pfu_per_ml must be > 0, got -1.0"):
        s.PhagePerturbation(
            name="T4", multiplicity_of_infection=1.0, titer_pfu_per_ml=-1.0
        )
    with _refuses("ncbi_taxid must be positive, got 0"):
        s.PhagePerturbation(name="T4", multiplicity_of_infection=1.0, ncbi_taxid=0)
    with _refuses("phage name must be non-empty and verbatim, got ' T4'"):
        s.PhagePerturbation(name=" T4", multiplicity_of_infection=1.0)


def test_a_derived_identifier_mapping_records_the_route_it_was_reached_by() -> None:
    """A crosswalked tag says so on the record, and the route matches the identifier."""
    mapping = s.DerivedIdentifierMapping(
        source_identifier="b0002", route="eck_crosswalk"
    )
    deletion = s.BacterialDeletionPerturbation(
        systematic_gene_name="BW25113_0002",
        perturbed_gene_name="thrA",
        gene_namespace="ecoli_k12_bw25113_locus_tag",
        identifier_mapping=mapping,
    )
    assert deletion.identifier_mapping == mapping
    dumped = deletion.model_dump()
    assert dumped["identifier_mapping"] == {
        "source_identifier": "b0002",
        "route": "eck_crosswalk",
    }
    assert s.BacterialDeletionPerturbation.model_validate(dumped) == deletion

    # every bacterial leaf carries the field, and it defaults to None
    for cls, fields in _BACTERIAL_LEAF_CASES:
        assert "identifier_mapping" in cls.model_fields
        assert cls(**fields).model_dump()["identifier_mapping"] is None


def test_a_mapping_route_must_match_the_identifier_it_starts_from() -> None:
    with _refuses(
        "an eck_crosswalk route starts from another strain's locus tag, got 'thrA'"
    ):
        s.DerivedIdentifierMapping(source_identifier="thrA", route="eck_crosswalk")
    with _refuses(
        "a jw_synonym route starts from a Keio JW id (^JW[RS]?\\d{4}$), got 'b0002'"
    ):
        s.DerivedIdentifierMapping(source_identifier="b0002", route="jw_synonym")
    with _refuses(
        "'b0002' is a locus tag, not a gene symbol; a tag of another strain is an "
        "eck_crosswalk"
    ):
        s.DerivedIdentifierMapping(source_identifier="b0002", route="gene_symbol")
    with _refuses("source_identifier must be a non-empty verbatim id, got ' b0002'"):
        s.DerivedIdentifierMapping(source_identifier=" b0002", route="eck_crosswalk")
    # the JW forms the BW25113 annotation actually carries, measured on its GenBank file
    for jw in ("JW0001", "JWR0257", "JWS0001"):
        assert (
            s.DerivedIdentifierMapping(
                source_identifier=jw, route="jw_synonym"
            ).source_identifier
            == jw
        )


def test_a_mapping_cannot_claim_a_tag_was_derived_from_itself() -> None:
    """A tag the source released as stored carries no mapping, and an eck_crosswalk
    crosses namespaces by definition.
    """
    with _refuses(
        "identifier_mapping says 'BW25113_0002' was derived from itself; a tag the "
        "source released verbatim carries no mapping"
    ):
        s.BacterialDeletionPerturbation(
            systematic_gene_name="BW25113_0002",
            perturbed_gene_name="thrA",
            gene_namespace="ecoli_k12_bw25113_locus_tag",
            identifier_mapping=s.DerivedIdentifierMapping(
                source_identifier="BW25113_0002", route="eck_crosswalk"
            ),
        )
    with _refuses(
        "an eck_crosswalk crosses namespaces, but 'BW25113_0003' is already a "
        "ecoli_k12_bw25113_locus_tag tag"
    ):
        s.BacterialDeletionPerturbation(
            systematic_gene_name="BW25113_0002",
            perturbed_gene_name="thrA",
            gene_namespace="ecoli_k12_bw25113_locus_tag",
            identifier_mapping=s.DerivedIdentifierMapping(
                source_identifier="BW25113_0003", route="eck_crosswalk"
            ),
        )
    # a gene-symbol route is fine on the same namespace: it is not a cross-strain join
    symbol = s.BacterialDeletionPerturbation(
        systematic_gene_name="BW25113_0002",
        perturbed_gene_name="thrA",
        gene_namespace="ecoli_k12_bw25113_locus_tag",
        identifier_mapping=s.DerivedIdentifierMapping(
            source_identifier="thrA", route="gene_symbol"
        ),
    )
    assert symbol.identifier_mapping is not None


def test_a_culture_environment_survives_the_product_titer_slot() -> None:
    """The narrowing's whole point: the vessel is dumped, not silently dropped.

    pydantic v2 serializes a field by its DECLARED type, so a ``CultureEnvironment`` in
    an ``Environment``-typed slot keeps its protocol as a python attribute and dumps
    WITHOUT it. Both halves are pinned: the narrowed slot keeps the vessel, the base
    slot loses it.
    """
    culture = s.CultureEnvironment(
        media=_lb(),
        culture_format=s.CultureFormat(
            vessel="48-well BioLector flower plate",
            working_volume_ul=1500.0,
            shaking_rpm=1000.0,
        ),
    )
    titer = s.ProductTiterExperiment(
        dataset_name="Toy",
        genotype=Genotype(perturbations=[s.BacterialDeletionPerturbation(**_MG1655)]),
        environment=culture,
        phenotype=_titer(),
    )
    dumped = titer.model_dump()
    assert dumped["environment"]["culture_format"]["vessel"] == (
        "48-well BioLector flower plate"
    )
    assert dumped["environment"]["culture_format"]["working_volume_ul"] == 1500.0
    back = s.ProductTiterExperiment.model_validate(dumped)
    assert isinstance(back.environment, s.CultureEnvironment)
    assert back == titer

    # the same culture in a base-typed slot (the bacterial fitness family) loses it
    fitness = s.BacterialFitnessExperiment(
        dataset_name="Toy",
        genotype=Genotype(perturbations=[s.BacterialDeletionPerturbation(**_MG1655)]),
        environment=culture,
        phenotype=FitnessPhenotype(fitness=0.9),
    )
    assert "culture_format" not in fitness.model_dump()["environment"]

    # and a plain Environment is refused by the narrowed slot rather than silently kept
    with pytest.raises(ValidationError, match="valid dictionary or instance of"):
        s.ProductTiterExperiment(
            dataset_name="Toy",
            genotype=Genotype(
                perturbations=[s.BacterialDeletionPerturbation(**_MG1655)]
            ),
            # the refusal under test: mypy sees it too, which is the point
            environment=Environment(media=_lb()),  # type: ignore[arg-type]
            phenotype=_titer(),
        )


def test_the_product_titer_reference_narrows_its_environment_too() -> None:
    """A control culture states its vessel, or it is not the same experiment."""
    reference = s.ProductTiterExperimentReference(
        dataset_name="Toy",
        genome_reference=s.AssemblyReferenceGenome(
            species="Pseudomonas putida",
            strain="KT2440",
            assembly_set="pputida_KT2440_ASM756v2",
            assembly_accession="GCA_000007565.2",
        ),
        environment_reference=s.CultureEnvironment(
            media=_lb(), culture_format=s.CultureFormat(vessel="48-well plate")
        ),
        phenotype_reference=_titer(titer=0.0),
    )
    dumped = reference.model_dump()
    assert dumped["environment_reference"]["culture_format"]["vessel"] == (
        "48-well plate"
    )
    assert s.ProductTiterExperimentReference.model_validate(dumped) == reference


# #776 / #793: the two additive phenotype blocks.


def test_environment_response_interval_requires_a_level_and_stays_verbatim() -> None:
    """Both released limits plus the level, and NO bracketing constraint.

    Caglar 2017 Table S5's Glycerol replicate 1 releases ``95p = -1027.769034`` against
    a doubling time of 80.95212424, the image of a slope interval straddling zero under
    ``DT = log_e 2 / slope``. The schema stores it; the verifier counts it.
    """
    stored = _response(
        measurement_type=s.MeasurementType.growth_rate,
        environment_response=80.95212424,
        environment_response_lower=38.94241445,
        environment_response_upper=-1027.769034,
        confidence_level=0.95,
    )
    assert stored.environment_response_upper == -1027.769034
    assert stored.environment_response_se is None
    with _refuses("confidence_level is required when a confidence limit is stored"):
        _response(environment_response_lower=0.1)
    with _refuses("confidence_level is required when a confidence limit is stored"):
        _response(environment_response_upper=0.9)
    with _refuses("confidence_level is a fraction in (0, 1), got 95.0"):
        _response(environment_response_lower=0.1, confidence_level=95.0)
    with _refuses("environment_response_lower must be finite"):
        _response(environment_response_lower=float("inf"), confidence_level=0.95)


def test_absolute_measurement_types_are_exactly_the_three_unnormalized_readouts() -> (
    None
):
    """The gate the verifier's reference relief requires. A relative readout is 0 at its
    control by construction and is deliberately not a member. ``cell_yield`` joined for
    De Chiara 2022's colony cell yields, released in cells; ``optical_density`` for Baba
    2006 Table 3's end-point OD600 (#927).
    """
    assert s.ABSOLUTE_MEASUREMENT_TYPES == frozenset(
        {
            s.MeasurementType.growth_rate,
            s.MeasurementType.colony_size,
            s.MeasurementType.cell_yield,
            s.MeasurementType.optical_density,
        }
    )
    assert s.MeasurementType.relative_growth_rate not in s.ABSOLUTE_MEASUREMENT_TYPES
    assert s.MeasurementType.log2_ratio not in s.ABSOLUTE_MEASUREMENT_TYPES
    assert not s.ABSOLUTE_MEASUREMENT_TYPES & s.CATEGORICAL_MEASUREMENT_TYPES


def test_environment_response_replicate_id_is_stored_verbatim() -> None:
    """The source's own replicate label, kept as a string so '01' and '1' stay apart."""
    assert _response(replicate_id="01").replicate_id == "01"
    assert _response().replicate_id is None


# #863: the released test of an environment response.


def test_environment_response_p_value_pair_is_stored_and_names_its_correction() -> None:
    """Wang 2024 Table S2 releases a permutation-test p-value and its FDR value per cell,
    0 among them; both are stored verbatim and never touch the uncertainty or the SE.
    """
    stored = _response(
        environment_response_p_value=0.0,
        environment_response_p_value_adjusted=0.00042,
        p_value_adjustment_method="benjamini_hochberg",
    )
    assert (
        stored.environment_response_p_value,
        stored.environment_response_p_value_adjusted,
        stored.p_value_adjustment_method,
    ) == (0.0, 0.00042, "benjamini_hochberg")
    assert stored.environment_response_se is None
    assert stored.environment_response_uncertainty is None
    assert _response(environment_response_p_value=1.0).p_value_adjustment_method is None
    unset = _response()
    assert (
        unset.environment_response_p_value,
        unset.environment_response_p_value_adjusted,
        unset.p_value_adjustment_method,
    ) == (None, None, None)


def test_environment_response_p_values_are_probabilities_with_a_named_correction() -> (
    None
):
    with _refuses("environment_response_p_value is 1.5, not a probability in [0, 1]"):
        _response(environment_response_p_value=1.5)
    with _refuses("environment_response_p_value is -0.1, not a probability in [0, 1]"):
        _response(environment_response_p_value=-0.1)
    with _refuses("environment_response_p_value is nan, not a probability in [0, 1]"):
        _response(environment_response_p_value=float("nan"))
    with _refuses(
        "environment_response_p_value_adjusted is inf, not a probability in [0, 1]"
    ):
        _response(
            environment_response_p_value_adjusted=float("inf"),
            p_value_adjustment_method="benjamini_hochberg",
        )
    with _refuses("an adjusted p-value names its correction: set p_value_adjustment"):
        _response(environment_response_p_value_adjusted=0.05)
    with _refuses("p_value_adjustment_method describes a stored adjusted p-value"):
        _response(
            environment_response_p_value=0.01,
            p_value_adjustment_method="benjamini_hochberg",
        )


# Wetmore 2015's own release: the released moderated t.


def test_environment_response_t_statistic_is_stored_and_never_feeds_the_se() -> None:
    """The moderated t is stored verbatim beside the SE it was computed from, and a
    record without it reads None.
    """
    stored = _response(
        environment_response_t_statistic=-5.25,
        environment_response_uncertainty=0.12,
        environment_response_uncertainty_type=s.UncertaintyType.standard_error,
    )
    assert stored.environment_response_t_statistic == -5.25
    assert stored.environment_response_se == 0.12
    assert _response().environment_response_t_statistic is None
    assert (
        _response(environment_response_t_statistic=0.0).environment_response_se is None
    )


def test_environment_response_t_statistic_is_finite_and_tests_a_number() -> None:
    with _refuses("environment_response_t_statistic must be finite, got nan"):
        _response(environment_response_t_statistic=float("nan"))
    with _refuses("environment_response_t_statistic must be finite, got inf"):
        _response(environment_response_t_statistic=float("inf"))
    with _refuses("environment_response_t_statistic tests a numeric response"):
        _response(
            measurement_type=s.MeasurementType.categorical,
            environment_response=None,
            category=s.ResponseCategory.reduced,
            environment_response_t_statistic=2.0,
        )


def _interaction(**overrides: Any) -> Any:
    kwargs: dict[str, Any] = {"gene_interaction": -1.5}
    kwargs.update(overrides)
    return s.GeneInteractionPhenotype(**kwargs)


def test_gene_interaction_replicate_quartet_follows_the_other_phenotypes() -> None:
    """Babu 2014's sourced eight colonies per pair, on the record.

    The p-value field is untouched and stays the class's ``label_statistic_name``: a
    p-value is a TEST of the score, the quartet is its replicate design and dispersion,
    and the two absences read differently on purpose.
    """
    stored = _interaction(n_samples=8, sample_unit=s.SampleUnit.colony)
    assert (stored.n_samples, stored.sample_unit) == (8, s.SampleUnit.colony)
    assert stored.label_statistic_name == "gene_interaction_p_value"
    assert stored.gene_interaction_uncertainty is None
    with _refuses("n_samples must be a positive integer or None, got: 0"):
        _interaction(n_samples=0)


def test_gene_interaction_uncertainty_is_both_or_neither() -> None:
    """An unlabelled dispersion is not ingested, and one that divides by n states its n."""
    sd = {
        "gene_interaction_uncertainty": 0.2,
        "gene_interaction_uncertainty_type": s.UncertaintyType.sample_sd,
    }
    labeled = _interaction(**sd, n_samples=8, sample_unit=s.SampleUnit.colony)
    assert labeled.gene_interaction_uncertainty == 0.2
    with _refuses(
        "gene_interaction_uncertainty and its type must both be set or both be None"
    ):
        _interaction(gene_interaction_uncertainty=0.2)
    with _refuses(
        "gene_interaction_uncertainty and its type must both be set or both be None"
    ):
        _interaction(gene_interaction_uncertainty_type=s.UncertaintyType.sample_sd)
    with _refuses("n_samples and sample_unit are required for sample_sd"):
        _interaction(**sd)
    with _refuses("n_samples and sample_unit are required for sample_sd"):
        _interaction(**sd, n_samples=8)
    # bootstrap_se and standard_error are already an SE, so they need no n
    assert (
        _interaction(
            gene_interaction_uncertainty=0.05,
            gene_interaction_uncertainty_type=s.UncertaintyType.bootstrap_se,
        ).gene_interaction_uncertainty
        == 0.05
    )


# --------------------------------------------------------------------------- #
# The round-2 bacterial leaves (#749, #792, #799) and the exposure dose.
# --------------------------------------------------------------------------- #
def test_a_marked_allele_is_a_sequence_change_and_not_an_absence() -> None:
    """The one leaf #749 and #792 both need, and what it refuses to assert.

    The gene is still there and still makes a product, so a marked allele must not be
    catchable by an "every knockout" filter, and it must carry no presence/absence
    state at all. Its mechanism is the insertion of recombinant bases, which is what
    was done to the DNA.
    """
    allele = s.BacterialMarkedAllelePerturbation(
        **_MG1655,
        cassette="kan",
        insertion_site="3'-UTR",
        tag="SPA",
        terminus="C",
        allele_effect="hypomorphic",
        collection="SPA-tag essential",
    )
    assert allele.perturbation_type == "bacterial_marked_allele"
    assert isinstance(allele, s.SequencePerturbation)
    assert not isinstance(allele, s.PresenceAbsencePerturbation)
    assert not isinstance(allele, s.DeletionPerturbation)
    assert "state" not in s.BacterialMarkedAllelePerturbation.model_fields
    assert allele.mechanism_so_id == "SO:0001218"
    assert allele.mechanism_so_name == "transgenic_insertion"
    # no magnitude field: no bacterial source releases a knockdown fold change
    assert "expression_range" not in s.BacterialMarkedAllelePerturbation.model_fields
    assert "expression_direction" not in (
        s.BacterialMarkedAllelePerturbation.model_fields
    )


def test_a_marked_allele_must_say_what_the_source_says_about_its_function() -> None:
    """``allele_effect`` is required and three-valued, so silence is not a False."""
    assert s.BacterialMarkedAllelePerturbation.model_fields[
        "allele_effect"
    ].is_required()
    with pytest.raises(ValidationError):
        s.BacterialMarkedAllelePerturbation(**_MG1655, cassette="kan")
    unstated = s.BacterialMarkedAllelePerturbation(
        **_MG1655, cassette="kan", allele_effect="not_stated"
    )
    assert unstated.allele_effect == "not_stated"
    assert unstated.tag is None and unstated.terminus is None
    with pytest.raises(ValidationError):
        # an effect outside the vocabulary; typed as Any so mypy sees the runtime path
        bad: Any = "maybe"
        s.BacterialMarkedAllelePerturbation(
            **_MG1655, cassette="kan", allele_effect=bad
        )


def test_a_marked_allele_terminus_names_a_tag() -> None:
    """A terminus with no fusion names nothing, so it is refused."""
    with _refuses(
        "terminus 'C' is stated with no tag; a terminus without a fusion names nothing"
    ):
        s.BacterialMarkedAllelePerturbation(
            **_MG1655, cassette="kan", terminus="C", allele_effect="not_stated"
        )
    # a tag whose end the source does not state is legal: Shiver's '-SPA' columns
    tag_only = s.BacterialMarkedAllelePerturbation(
        **_MG1655, tag="SPA", allele_effect="not_stated"
    )
    assert tag_only.terminus is None
    assert tag_only.cassette is None


def test_a_degron_is_its_own_leaf_carrying_its_inducing_condition() -> None:
    """A degron asserts regulated proteolysis, which a marked allele does not."""
    degron = s.BacterialDegronPerturbation(
        **_MG1655,
        degron="DAS+4",
        terminus="C",
        protease="ClpXP",
        adaptor="SspB",
        inducing_condition="sspB expressed from an arabinose-inducible promoter",
    )
    assert degron.perturbation_type == "bacterial_degron"
    assert isinstance(degron, s.SequencePerturbation)
    assert not isinstance(degron, s.BacterialMarkedAllelePerturbation)
    # degron is the one required field; a release that names only the degron is legal,
    # which is Shiver 2016's '-DAS' columns
    assert s.BacterialDegronPerturbation.model_fields["degron"].is_required()
    token_only = s.BacterialDegronPerturbation(**_MG1655, degron="DAS+4")
    assert (token_only.terminus, token_only.protease, token_only.adaptor) == (
        None,
        None,
        None,
    )
    assert token_only.inducing_condition is None
    with pytest.raises(ValidationError):
        s.BacterialDegronPerturbation(**_MG1655)


def test_the_bacterial_crispra_leaf_states_increased_where_crispri_cannot() -> None:
    """The expression axis's missing direction, and the leaf that supplies it."""
    activation = s.BacterialCrisprActivationPerturbation(
        **_MG1655, crispr=s.CrisprConstruct(effector="dCas9*-MCPSoxS")
    )
    assert activation.perturbation_type == "bacterial_crispr_activation"
    assert activation.expression_direction == "increased"
    assert activation.state == "present"
    assert activation.mechanism_so_id == "SO:0001998"
    assert isinstance(activation, s.CrisprActivationPerturbation)
    assert isinstance(activation, s.ExpressionModulationPerturbation)
    # the interference leaf cannot state it, which is why this leaf exists
    interference = s.BacterialCrisprInterferencePerturbation(
        **_MG1655, crispr=s.CrisprConstruct(effector="dCas9-Mxi1")
    )
    assert interference.expression_direction == "decreased"
    # and the yeast CRISPRa leaf still refuses the identifier
    with _refuses("Invalid systematic gene name format"):
        s.CrisprActivationPerturbation(
            systematic_gene_name="b0002",
            perturbed_gene_name="thrA",
            crispr=s.CrisprConstruct(effector="dCas9*-MCPSoxS"),
        )


def test_an_exposure_dose_is_a_time_not_a_concentration() -> None:
    """Shiver 2016's UV dose is 12 seconds with no irradiance anywhere in the mirror."""
    exposure = s.PhysicalExposurePerturbation(
        factor=s.PhysicalFactor.radiation,
        exposure_duration_seconds=12.0,
        provenance_gaps=[
            ProvenanceGap(
                field="irradiance_w_per_m2",
                reason=ProvenanceGapReason.not_reported_by_primary,
            )
        ],
    )
    assert exposure.perturbation_type == "physical_exposure"
    assert exposure.irradiance_w_per_m2 is None
    assert exposure.fluence_j_per_m2 is None
    assert "concentration" not in s.PhysicalExposurePerturbation.model_fields
    assert "magnitude" not in s.PhysicalExposurePerturbation.model_fields
    assert s.PhysicalExposurePerturbation in typing.get_args(
        s.EnvironmentPerturbationType
    )
    environment = Environment(media=_lb(), perturbations=[exposure])
    adapter: TypeAdapter[Any] = TypeAdapter(Environment)
    assert (
        adapter.validate_python(environment.model_dump()).perturbations[0] == exposure
    )


def test_an_exposure_states_a_dose_or_declares_the_gap() -> None:
    """An exposure always has a dose, so none of the three silently stays None."""
    with _refuses("PhysicalExposurePerturbation states no dose"):
        s.PhysicalExposurePerturbation(factor=s.PhysicalFactor.radiation)
    with _refuses(
        "PhysicalExposurePerturbation.exposure_duration_seconds must be > 0, got 0.0"
    ):
        s.PhysicalExposurePerturbation(
            factor=s.PhysicalFactor.radiation, exposure_duration_seconds=0.0
        )
    with _refuses(
        "PhysicalExposurePerturbation.fluence_j_per_m2 must be > 0, got -1.0"
    ):
        s.PhysicalExposurePerturbation(
            factor=s.PhysicalFactor.radiation, fluence_j_per_m2=-1.0
        )
    gapped = s.PhysicalExposurePerturbation(
        factor=s.PhysicalFactor.radiation,
        provenance_gaps=[
            ProvenanceGap(
                field="exposure_duration_seconds",
                reason=ProvenanceGapReason.not_reported_by_primary,
            )
        ],
    )
    assert gapped.exposure_duration_seconds is None


# --- 7. #770 and #753: the protein fold-change family and five new fields -- #
def _fold_change(**kw: Any) -> s.ProteinFoldChangePhenotype:
    """One PP_ keyed fold-change record on the linear scale, overridable per test."""
    fields: dict[str, Any] = dict(
        protein_fold_change={"PP_4188": 0.2213},
        fold_change_scale=s.FoldChangeScale.linear,
        reference_basis="control strain carrying a non-targeting sgRNA",
        n_replicates={"PP_4188": 3},
        measurement_type="dia_top3_ratio_to_control",
    )
    fields.update(kw)
    return s.ProteinFoldChangePhenotype(**fields)


def test_a_fold_change_record_names_its_scale_and_its_denominator() -> None:
    """The ratio, the scale that fixes its neutral value, and the basis it is against."""
    phenotype = _fold_change()
    assert phenotype.label_name == "protein_fold_change"
    assert phenotype.label_statistic_name == "protein_fold_change_se"
    assert phenotype.graph_level == "node"
    assert phenotype.fold_change_scale.neutral_value == 1.0
    assert s.FoldChangeScale.log2.neutral_value == 0.0
    assert s.FoldChangeScale.log10.neutral_value == 0.0
    # the reference is the neutral value by definition, never a measured quantity
    assert phenotype.neutral_reference() == {"PP_4188": 1.0}
    log2 = _fold_change(
        protein_fold_change={"PP_4188": -2.1, "PP_0368": 0.0},
        n_replicates={"PP_4188": 3, "PP_0368": 3},
        fold_change_scale=s.FoldChangeScale.log2,
    )
    # a log scale is SIGNED, and its neutral value is 0.0 for every stored key
    assert log2.neutral_reference() == {"PP_0368": 0.0, "PP_4188": 0.0}


def test_a_linear_fold_change_is_non_negative_and_a_log_one_is_signed() -> None:
    """A negative ratio is refused; zero is a measured loss of signal, not an absence."""
    with _refuses(
        "protein_fold_change for PP_4188 is -2.1, which is not a ratio; a linear fold "
        "change is non-negative (a log scale is signed). Zero is a measured numerator "
        "below detection, not a missing key"
    ):
        _fold_change(protein_fold_change={"PP_4188": -2.1})
    # 0.0 on the linear scale IS a measurement: Yunus 2026 releases the verbatim cell
    # `0` for 37 of its 102 single-target rows and counts them in its own census
    assert _fold_change(protein_fold_change={"PP_4188": 0.0}).protein_fold_change == {
        "PP_4188": 0.0
    }
    assert _fold_change(
        protein_fold_change={"PP_4188": -2.1}, fold_change_scale=s.FoldChangeScale.log2
    ).protein_fold_change == {"PP_4188": -2.1}
    with _refuses("protein_fold_change for PP_4188 must be finite"):
        _fold_change(protein_fold_change={"PP_4188": math.inf})
    with _refuses("protein_fold_change cannot be empty"):
        _fold_change(protein_fold_change={}, n_replicates={})
    with _refuses("n_replicates keys must match protein_fold_change keys"):
        _fold_change(n_replicates={"PP_0368": 3})
    with _refuses("n_replicates for PP_4188 must be >= 1"):
        _fold_change(n_replicates={"PP_4188": 0})


def test_a_fold_change_p_value_is_a_probability_and_names_its_correction() -> None:
    """The per-protein test result the schema had nowhere to put before #770."""
    phenotype = _fold_change(
        protein_fold_change_se={"PP_4188": 0.01},
        protein_fold_change_p_value={"PP_4188": 4.87e-20},
        protein_fold_change_p_value_adjusted={"PP_4188": 1.2e-17},
        p_value_adjustment_method="benjamini_hochberg",
    )
    assert phenotype.protein_fold_change_p_value == {"PP_4188": 4.87e-20}
    assert phenotype.p_value_adjustment_method == "benjamini_hochberg"
    with _refuses(
        "protein_fold_change_p_value for PP_4188 is 1.5, not a probability in [0, 1]"
    ):
        _fold_change(protein_fold_change_p_value={"PP_4188": 1.5})
    with _refuses("protein_fold_change_p_value key PP_0368 not in protein_fold_change"):
        _fold_change(protein_fold_change_p_value={"PP_0368": 0.01})
    with _refuses(
        "adjusted p-values name their correction: set p_value_adjustment_method"
    ):
        _fold_change(protein_fold_change_p_value_adjusted={"PP_4188": 0.01})
    with _refuses(
        "p_value_adjustment_method describes stored adjusted p-values; set "
        "protein_fold_change_p_value_adjusted or leave the method None"
    ):
        _fold_change(p_value_adjustment_method="benjamini_hochberg")
    # NaN is "no test for this key", the convention the SE maps already use
    nan_p = _fold_change(
        protein_fold_change_p_value={"PP_4188": math.nan}
    ).protein_fold_change_p_value
    assert nan_p is not None and math.isnan(nan_p["PP_4188"])
    with _refuses("SE for PP_4188 must be non-negative"):
        _fold_change(protein_fold_change_se={"PP_4188": -0.01})


def test_the_fold_change_family_round_trips_through_the_type_maps() -> None:
    """Leaf plus experiment plus reference, reconstructed by tag as the loaders do."""
    assert _experiment_cls("protein_fold_change") is s.ProteinFoldChangeExperiment
    assert (
        _experiment_cls("bacterial_protein_fold_change")
        is s.BacterialProteinFoldChangeExperiment
    )
    assert (
        s.EXPERIMENT_REFERENCE_TYPE_MAP["bacterial_protein_fold_change"]
        is s.BacterialProteinFoldChangeExperimentReference
    )
    experiment = s.BacterialProteinFoldChangeExperiment(
        dataset_name="Toy",
        genotype=Genotype(perturbations=[s.BacterialDeletionPerturbation(**_MG1655)]),
        environment=Environment(media=_lb()),
        phenotype=_fold_change(),
    )
    assert experiment.experiment_type == "bacterial_protein_fold_change"
    dumped = experiment.model_dump()
    assert s.BacterialProteinFoldChangeExperiment.model_validate(dumped) == experiment
    assert s.ProteinFoldChangePhenotype in typing.get_args(s.PhenotypeType)


def test_a_turnover_record_stores_its_published_interval_and_its_censoring() -> None:
    """#753: the interval stays an interval, and a capped value says it is a bound."""
    fields: dict[str, Any] = dict(
        degradation_rate={"b0002": 0.08, "b0003": 0.02},
        n_replicates={"b0002": 2, "b0003": 1},
        measurement_type="pulse_silac_degradation_rate_per_hour",
    )
    phenotype = s.ProteinTurnoverPhenotype(
        **fields,
        degradation_rate_lower={"b0003": 0.015},
        degradation_rate_upper={"b0003": 0.027},
        confidence_level=0.95,
        interval_method="curve_fit_parameter_variance_t_quantile",
        censoring={"b0002": s.Censoring.uncensored, "b0003": s.Censoring.right},
    )
    # the SE is still the named label statistic: a replicate SE and a fitted interval
    # are different statistics, and a release may carry either
    assert phenotype.label_statistic_name == "degradation_rate_se"
    assert phenotype.censoring == {
        "b0002": s.Censoring.uncensored,
        "b0003": s.Censoring.right,
    }
    # ragged by design: b0002 has two replicates and no published interval
    assert set(phenotype.degradation_rate_lower or {}) == {"b0003"}
    assert [str(member) for member in s.Censoring] == ["uncensored", "right", "left"]
    with _refuses(
        "degradation_rate_lower for b0003 exceeds the stored rate (0.03 > 0.02)"
    ):
        s.ProteinTurnoverPhenotype(
            **fields,
            degradation_rate_lower={"b0003": 0.03},
            confidence_level=0.95,
            interval_method="curve_fit_parameter_variance_t_quantile",
        )
    with _refuses(
        "degradation_rate_upper for b0003 is below the stored rate (0.01 < 0.02)"
    ):
        s.ProteinTurnoverPhenotype(
            **fields,
            degradation_rate_upper={"b0003": 0.01},
            confidence_level=0.95,
            interval_method="curve_fit_parameter_variance_t_quantile",
        )
    with _refuses("censoring key b9999 not in degradation_rate"):
        s.ProteinTurnoverPhenotype(**fields, censoring={"b9999": s.Censoring.right})


def test_turnover_bounds_must_state_their_level_and_their_method() -> None:
    """Bounds nobody can interpret are refused, unless each absence is a typed gap."""
    fields: dict[str, Any] = dict(
        degradation_rate={"b0002": 0.08},
        n_replicates={"b0002": 1},
        measurement_type="pulse_silac_degradation_rate_per_hour",
        degradation_rate_lower={"b0002": 0.07},
        degradation_rate_upper={"b0002": 0.09},
    )
    with _refuses(
        "an interval states the level it covers: set confidence_level or carry a "
        "ProvenanceGap on it"
    ):
        s.ProteinTurnoverPhenotype(**fields)
    with _refuses(
        "an interval states how it was produced: set interval_method or carry a "
        "ProvenanceGap on it"
    ):
        s.ProteinTurnoverPhenotype(**fields, confidence_level=0.95)
    gapped = s.ProteinTurnoverPhenotype(
        **fields,
        provenance_gaps=[
            ProvenanceGap(
                field="confidence_level",
                reason=ProvenanceGapReason.not_reported_by_primary,
            ),
            ProvenanceGap(
                field="interval_method",
                reason=ProvenanceGapReason.not_reported_by_primary,
            ),
        ],
    )
    assert gapped.gapped_fields() == {"confidence_level", "interval_method"}
    with _refuses(
        "confidence_level and interval_method describe stored bounds; set "
        "degradation_rate_lower/upper or leave both None"
    ):
        s.ProteinTurnoverPhenotype(
            degradation_rate={"b0002": 0.08},
            n_replicates={"b0002": 1},
            measurement_type="pulse_silac_degradation_rate_per_hour",
            confidence_level=0.95,
        )


def test_an_environment_carries_a_chemostat_dilution_rate() -> None:
    """#753: the dilution rate is the controlled variable, so it is part of identity."""
    chemostat = Environment(media=_lb(), dilution_rate_per_hour=0.2)
    assert chemostat.dilution_rate_per_hour == 0.2
    assert Environment(media=_lb()).dilution_rate_per_hour is None
    # two cultures differing only here are two environments, which is the point
    assert chemostat != Environment(media=_lb(), dilution_rate_per_hour=0.7)
    with _refuses("dilution_rate_per_hour must be finite and positive, got 0.0"):
        Environment(media=_lb(), dilution_rate_per_hour=0.0)
    with _refuses("dilution_rate_per_hour must be finite and positive, got -0.2"):
        Environment(media=_lb(), dilution_rate_per_hour=-0.2)


def test_the_two_new_identifier_routes_check_the_form_they_read() -> None:
    """#753: a UniProt accession and a retired tag of the pinned strain's namespace."""
    header = s.DerivedIdentifierMapping(
        source_identifier="sp|P0A6E9|BIOD2_ECOLI", route="uniprot_db_xref"
    )
    assert header.uniprot_accession() == "P0A6E9"
    bare = s.DerivedIdentifierMapping(
        source_identifier="P0A6E9", route="uniprot_db_xref"
    )
    assert bare.uniprot_accession() == "P0A6E9"
    with _refuses(
        "a uniprot_db_xref route starts from a UniProtKB accession, bare or in a "
        "db|ACC|ENTRY header, got 'bioD'"
    ):
        s.DerivedIdentifierMapping(source_identifier="bioD", route="uniprot_db_xref")
    assert (
        s.DerivedIdentifierMapping(
            source_identifier="b4104", route="locus_tag_synonym"
        ).route
        == "locus_tag_synonym"
    )
    with _refuses(
        "a locus_tag_synonym route starts from a retired locus tag of the pinned "
        "strain's own namespace, got 'phnE'"
    ):
        s.DerivedIdentifierMapping(source_identifier="phnE", route="locus_tag_synonym")
    # the namespace rule lives on the leaf, where the stored tag's namespace is known
    assert (
        s.BacterialDeletionPerturbation(
            **_MG1655,
            identifier_mapping=s.DerivedIdentifierMapping(
                source_identifier="b4104", route="locus_tag_synonym"
            ),
        ).identifier_mapping
        is not None
    )
    with _refuses(
        "a locus_tag_synonym stays inside one namespace, but 'BW25113_0002' is a "
        "ecoli_k12_bw25113_locus_tag tag while gene_namespace is "
        "'ecoli_k12_mg1655_bnumber' (a tag of another strain is an eck_crosswalk)"
    ):
        s.BacterialDeletionPerturbation(
            **_MG1655,
            identifier_mapping=s.DerivedIdentifierMapping(
                source_identifier="BW25113_0002", route="locus_tag_synonym"
            ),
        )


def test_a_provenance_gap_can_name_keys_of_a_partially_populated_map() -> None:
    """#753: 244 unsourced keys beside 1,077 sourced ones, and still no silent None."""
    phenotype = s.MetabolitePhenotype(
        metabolite_level={"glucose": 1.0},
        n_replicates={"glucose": 3},
        measurement_type="lcms_relative_intensity",
        target_metabolite_ids={"glucose": "CHEBI:17234"},
        provenance_gaps=[
            ProvenanceGap(
                field="target_metabolite_ids",
                reason=ProvenanceGapReason.not_reported_by_primary,
                keys=["feature_0042"],
                note="a merged isobaric feature with several candidate identities",
            )
        ],
    )
    gap = phenotype.provenance_gaps[0]
    assert gap.keys == ["feature_0042"]
    # the honesty invariant gets STRONGER, not weaker: the map may not carry the key
    with _refuses(
        "field 'target_metabolite_ids' declares keys ['glucose'] missing but carries "
        "them (cannot both store a value and declare it missing)"
    ):
        s.MetabolitePhenotype(
            metabolite_level={"glucose": 1.0},
            n_replicates={"glucose": 3},
            measurement_type="lcms_relative_intensity",
            target_metabolite_ids={"glucose": "CHEBI:17234"},
            provenance_gaps=[
                ProvenanceGap(
                    field="target_metabolite_ids",
                    reason=ProvenanceGapReason.not_reported_by_primary,
                    keys=["glucose"],
                )
            ],
        )
    with _refuses(
        "field 'measurement_type' has a per-key ProvenanceGap but is str, not a "
        "mapping (a per-key gap names keys of a dict-valued field)"
    ):
        s.MetabolitePhenotype(
            metabolite_level={"glucose": 1.0},
            n_replicates={"glucose": 3},
            measurement_type="lcms_relative_intensity",
            provenance_gaps=[
                ProvenanceGap(
                    field="measurement_type",
                    reason=ProvenanceGapReason.not_reported_by_primary,
                    keys=["glucose"],
                )
            ],
        )
    with pytest.raises(ValidationError, match="ProvenanceGap.keys cannot repeat a key"):
        ProvenanceGap(
            field="target_metabolite_ids",
            reason=ProvenanceGapReason.not_reported_by_primary,
            keys=["a", "a"],
        )
    with pytest.raises(
        ValidationError, match="ProvenanceGap.keys cannot hold an empty key"
    ):
        ProvenanceGap(
            field="target_metabolite_ids",
            reason=ProvenanceGapReason.not_reported_by_primary,
            keys=[" "],
        )


# --------------------------------------------------------------------------- #
# Issue #854: MrnaNumberFractionPhenotype, the count-less transcriptome
# --------------------------------------------------------------------------- #
def _mrna_fraction(**kw: Any) -> s.MrnaNumberFractionPhenotype:
    fields: dict[str, Any] = dict(
        mrna_number_fraction={"b0002": 0.6, "b0001": 0.0},
        n_libraries=1,
        measurement_type="rnaseq_mrna_number_fraction",
    )
    fields.update(kw)
    return s.MrnaNumberFractionPhenotype(**fields)


def test_mrna_fraction_keeps_a_released_zero_and_sorts_its_keys() -> None:
    phenotype = _mrna_fraction()
    assert list(phenotype.mrna_number_fraction.items()) == [
        ("b0001", 0.0),
        ("b0002", 0.6),
    ]
    assert repr(phenotype) == "MrnaNumberFractionPhenotype(genes=2, n_libraries=1)"
    assert phenotype.label_name == "mrna_number_fraction"


@pytest.mark.parametrize(
    "fractions,message",
    [
        ({}, "cannot be empty"),
        ({"b0001": 1.2}, "finite value in \\[0, 1\\]"),
        ({"b0001": -0.1}, "finite value in \\[0, 1\\]"),
        ({"b0001": float("nan")}, "finite value in \\[0, 1\\]"),
        ({"b0001": True}, "is not a number"),
        ({"b0001": 0.7, "b0002": 0.4}, "sums to 1.1"),
    ],
)
def test_mrna_fraction_refuses_what_a_fraction_cannot_be(
    fractions: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _mrna_fraction(mrna_number_fraction=fractions)


def test_mrna_fraction_refuses_a_list_no_libraries_and_a_blank_type() -> None:
    with pytest.raises(ValueError, match="per-gene dict"):
        _mrna_fraction(mrna_number_fraction=[0.5])
    with pytest.raises(ValueError, match="positive integer"):
        _mrna_fraction(n_libraries=0)
    with pytest.raises(ValueError, match="cannot be blank"):
        _mrna_fraction(measurement_type="  ")


def test_mrna_fraction_family_is_in_every_union_and_map() -> None:
    assert s.MrnaNumberFractionPhenotype in typing.get_args(s.PhenotypeType)
    assert s.MrnaNumberFractionExperiment in typing.get_args(s.ExperimentType)
    assert s.MrnaNumberFractionExperimentReference in typing.get_args(
        s.ExperimentReferenceType
    )
    assert (
        s.EXPERIMENT_TYPE_MAP["mrna_number_fraction"] is s.MrnaNumberFractionExperiment
    )
    assert (
        s.EXPERIMENT_REFERENCE_TYPE_MAP["mrna_number_fraction"]
        is s.MrnaNumberFractionExperimentReference
    )


# --------------------------------------------------------------------------- #
# Boocock 2025: SingleCellExpressionPhenotype, UMI counts of a segregant's cells (the
# Jackson 2020 shape, cassette markers and one library per record, is pinned in
# test_single_cell_expression_schema.py)
# --------------------------------------------------------------------------- #
def _single_cell(**kw: Any) -> s.SingleCellExpressionPhenotype:
    fields: dict[str, Any] = dict(
        umi_count={"YAL002W": 3, "YAL001C": 1},
        marker_umi_count=None,
        total_umis=6,
        n_features=10,
        feature_universe_sha256="a" * 64,
        n_cells=2,
        cell_cycle_stage_counts={"G1": 1, "S": 1},
        library_cell_counts={"L1": 2},
        cell_barcodes=["AAAC-1", "GGGT-1"],
        cell_libraries=["L1", "L1"],
        cell_total_umis=[4, 2],
        provenance_gaps=[_gap("marker_umi_count")],
    )
    fields.update(kw)
    return s.SingleCellExpressionPhenotype(**fields)


def test_single_cell_counts_sort_and_the_total_may_exceed_the_stored_universe() -> None:
    phenotype = _single_cell()
    assert list(phenotype.umi_count.items()) == [("YAL001C", 1), ("YAL002W", 3)]
    assert repr(phenotype) == (
        "SingleCellExpressionPhenotype(genes=2, total_umis=6, n_cells=2)"
    )
    assert (phenotype.label_name, phenotype.measurement_type) == (
        "umi_count",
        "scrnaseq_umi_count",
    )
    assert _single_cell(total_umis=4, cell_total_umis=[2, 2]).total_umis == 4
    assert phenotype.marker_umi_count is None
    assert phenotype.gapped_fields() == {"marker_umi_count"}


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"umi_count": [1]}, "per-gene dict, got list"),
        ({"umi_count": {}}, "umi_count cannot be empty"),
        ({"umi_count": {"YAL001C": 0}}, "zeros are omitted, not stored"),
        ({"umi_count": {"YAL001C": True}}, "must be a positive integer"),
        ({"total_umis": 3}, "total_umis 3 is smaller than the stored counts 4"),
        ({"n_features": 1}, "n_features 1 is smaller than the 2 keyed features"),
        ({"feature_universe_sha256": "A" * 64}, "64 lowercase hex digits"),
        ({"n_cells": 0}, "n_cells must be a positive integer, got 0"),
        (
            {"cell_cycle_stage_counts": {"G1": 1}},
            "cell_cycle_stage_counts sums to 1, not n_cells 2",
        ),
        (
            {"library_cell_counts": {"L1": 3, "L2": -1}},
            "library_cell_counts holds a non-positive count",
        ),
        (
            {"measurement_type": "scrnaseq_pseudobulk_umi_count"},
            "Input should be 'scrnaseq_umi_count'",
        ),
    ],
)
def test_single_cell_refuses_what_a_count_record_cannot_be(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _single_cell(**overrides)


# --- begin nguessan2025: a source without stage labels gaps the partition --- #
def test_single_cell_stage_partition_may_be_a_typed_gap() -> None:
    """N'Guessan 2025 labels no cell-cycle stage: None, but only beside its gap."""
    phenotype = _single_cell(
        cell_cycle_stage_counts=None,
        provenance_gaps=[_gap("cell_cycle_stage_counts"), _gap("marker_umi_count")],
    )
    assert phenotype.cell_cycle_stage_counts is None
    assert phenotype.gapped_fields() == {"cell_cycle_stage_counts", "marker_umi_count"}
    assert phenotype.library_cell_counts == {"L1": 2}


def test_single_cell_refuses_an_absent_stage_partition_without_its_gap() -> None:
    with pytest.raises(ValueError, match="carries no ProvenanceGap"):
        _single_cell(cell_cycle_stage_counts=None)
    with pytest.raises(ValueError, match="sums to 0, not n_cells 2"):
        _single_cell(cell_cycle_stage_counts={})
    with pytest.raises(ValueError, match="cannot both store a value"):
        _single_cell(
            provenance_gaps=[_gap("cell_cycle_stage_counts"), _gap("marker_umi_count")]
        )


# --- end nguessan2025 ---


def test_segregant_expression_family_is_in_every_union_and_map() -> None:
    assert s.SingleCellExpressionPhenotype in typing.get_args(s.PhenotypeType)
    assert s.SegregantExpressionExperiment in typing.get_args(s.ExperimentType)
    assert s.SegregantExpressionExperimentReference in typing.get_args(
        s.ExperimentReferenceType
    )
    assert (
        s.EXPERIMENT_TYPE_MAP["segregant_expression"] is s.SegregantExpressionExperiment
    )
    assert (
        s.EXPERIMENT_REFERENCE_TYPE_MAP["segregant_expression"]
        is s.SegregantExpressionExperimentReference
    )


# --- begin muenzner2024: the yeast protein-turnover pair -------------------------- #
def _yeast_turnover_experiment() -> s.YeastProteinTurnoverExperiment:
    return s.YeastProteinTurnoverExperiment(
        dataset_name="Toy",
        genotype=Genotype(
            perturbations=[
                s.NaturalGeneAbsencePerturbation(
                    systematic_gene_name="YAL001C",
                    perturbed_gene_name="TFC3",
                    strain_id="AAA",
                    pangenome_orf_id="1-YAL001C",
                )
            ]
        ),
        environment=Environment(
            media=s.Media(name="SM", state="liquid", is_synthetic=True)
        ),
        phenotype=s.ProteinTurnoverPhenotype(
            degradation_rate={"YAL001C": 0.25},
            half_life={"YAL001C": math.log(2) / 0.25},
            n_replicates={"YAL001C": 1},
            measurement_type="dynamic_silac_lys8_kdp_per_hour_no_dilution_correction",
        ),
    )


def test_yeast_turnover_pair_is_in_both_maps_and_both_unions() -> None:
    """The yeast pair resolves by its own tag and measures the bacterial family's class."""
    experiment_cls = _experiment_cls("yeast_protein_turnover")
    reference_cls = _reference_cls("yeast_protein_turnover")
    assert experiment_cls is s.YeastProteinTurnoverExperiment
    assert reference_cls is s.YeastProteinTurnoverExperimentReference
    assert experiment_cls in typing.get_args(ExperimentType)
    assert reference_cls in typing.get_args(s.ExperimentReferenceType)
    assert (
        experiment_cls.model_fields["phenotype"].annotation
        is reference_cls.model_fields["phenotype_reference"].annotation
        is s.ProteinTurnoverPhenotype
    )
    assert experiment_cls.model_fields["experiment_type"].default == (
        "yeast_protein_turnover"
    )
    assert reference_cls.model_fields["experiment_reference_type"].default == (
        "yeast_protein_turnover"
    )


def test_yeast_turnover_reference_takes_the_plain_s288c_genome() -> None:
    """The bacterial pair refuses an S288C genome; the yeast pair stores it as given."""
    phenotype = _yeast_turnover_experiment().phenotype
    s288c = s.ReferenceGenome(species="Saccharomyces cerevisiae", strain="S288C")
    environment = Environment(
        media=s.Media(name="SM", state="liquid", is_synthetic=True)
    )
    with pytest.raises(ValidationError, match="instance of AssemblyReferenceGenome"):
        s.ProteinTurnoverExperimentReference(
            dataset_name="Toy",
            genome_reference=s288c,  # type: ignore[arg-type]  # the refusal under test
            environment_reference=environment,
            phenotype_reference=phenotype,
        )
    reference = s.YeastProteinTurnoverExperimentReference(
        dataset_name="Toy",
        genome_reference=s288c,
        environment_reference=environment,
        phenotype_reference=phenotype,
    )
    assert reference.model_dump()["genome_reference"] == {
        "species": "Saccharomyces cerevisiae",
        "strain": "S288C",
        "ploidy": "haploid",
    }


def test_yeast_turnover_experiment_reconstructs_through_its_tag() -> None:
    """Same field set as ``ProteinTurnoverExperiment``, so the tag picks the class."""
    experiment = _yeast_turnover_experiment()
    dumped = experiment.model_dump()
    rebuilt = _experiment_cls(dumped["experiment_type"]).model_validate(dumped)
    assert type(rebuilt) is s.YeastProteinTurnoverExperiment
    assert rebuilt == experiment
    assert rebuilt.phenotype.half_life is not None
    assert rebuilt.phenotype.half_life["YAL001C"] == pytest.approx(math.log(2) / 0.25)


# --- end muenzner2024 --------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Issue #1022: MicroarrayExpressionScorePhenotype, the error-model score family
# --------------------------------------------------------------------------- #
def _expression_score(**kw: Any) -> s.MicroarrayExpressionScorePhenotype:
    fields: dict[str, Any] = dict(
        expression_score={"YBR001C": 1.5, "YAL001C": -3.0},
        expression_score_p_value={"YBR001C": 0.133614, "YAL001C": 0.0027},
        measurement_type="error_model_x_score_weighted_duplicate_average",
        reference_basis="wild-type BY4741 culture",
        array_platform="GEO GPL3588",
        n_replicates=2,
        harvest_optical_density=0.82,
    )
    fields.update(kw)
    return s.MicroarrayExpressionScorePhenotype(**fields)


def test_expression_score_sorts_its_keys_and_names_its_statistic() -> None:
    phenotype = _expression_score()
    assert list(phenotype.expression_score.items()) == [
        ("YAL001C", -3.0),
        ("YBR001C", 1.5),
    ]
    assert phenotype.expression_score_p_value is not None
    assert list(phenotype.expression_score_p_value) == ["YAL001C", "YBR001C"]
    assert (phenotype.label_name, phenotype.label_statistic_name) == (
        "expression_score",
        "expression_score_p_value",
    )
    assert repr(phenotype) == (
        "MicroarrayExpressionScorePhenotype(genes=2, p_value_genes=2, n_replicates=2)"
    )
    reference = _expression_score(
        expression_score={"YAL001C": 0.0},
        expression_score_p_value=None,
        harvest_optical_density=None,
    )
    assert repr(reference) == (
        "MicroarrayExpressionScorePhenotype(genes=1, p_value_genes=0, n_replicates=2)"
    )


@pytest.mark.parametrize(
    "fields,message",
    [
        ({"expression_score": {}}, "expression_score cannot be empty"),
        ({"expression_score": [1.0]}, "must be a per-gene dict"),
        ({"expression_score": {"YAL001C": True}}, "is not a number"),
        ({"expression_score": {"YAL001C": math.inf}}, "is not finite"),
        ({"expression_score_p_value": [0.5]}, "must be a per-gene dict"),
        (
            {"expression_score_p_value": {"YAL001C": 1.5, "YBR001C": 0.1}},
            "not a probability in \\[0, 1\\]",
        ),
        (
            {"expression_score_p_value": {"YAL001C": "0.1", "YBR001C": 0.1}},
            "is not a number",
        ),
        ({"expression_score_p_value": {"YAL001C": 0.1}}, "same keys"),
        ({"measurement_type": " "}, "cannot be blank"),
        ({"array_platform": ""}, "cannot be blank"),
        ({"n_replicates": 0}, "positive integer"),
        ({"harvest_optical_density": 0.0}, "finite and positive"),
        ({"harvest_optical_density": math.nan}, "finite and positive"),
    ],
)
def test_expression_score_refuses_what_a_score_record_cannot_be(
    fields: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _expression_score(**fields)


def test_expression_score_family_is_in_every_union_and_map() -> None:
    assert s.MicroarrayExpressionScorePhenotype in typing.get_args(s.PhenotypeType)
    assert s.MicroarrayExpressionScoreExperiment in typing.get_args(s.ExperimentType)
    assert s.MicroarrayExpressionScoreExperimentReference in typing.get_args(
        s.ExperimentReferenceType
    )
    assert (
        s.EXPERIMENT_TYPE_MAP["microarray_expression_score"]
        is s.MicroarrayExpressionScoreExperiment
    )
    assert (
        s.EXPERIMENT_REFERENCE_TYPE_MAP["microarray_expression_score"]
        is s.MicroarrayExpressionScoreExperimentReference
    )


# --------------------------------------------------------------------------- #
# Issues #1022 #989: the two-channel microarray parent, the X-to-p conversion and the
# ratio-only leaf
# --------------------------------------------------------------------------- #
TWO_CHANNEL_LEAVES = (
    s.MicroarrayExpressionPhenotype,
    s.MicroarrayExpressionScorePhenotype,
    s.MicroarrayExpressionRatioPhenotype,
    s.InductionTimecourseExpressionPhenotype,
)


def test_two_channel_releases_are_siblings_under_one_field_less_parent() -> None:
    """One physical measurement, four releases: four leaves of one parent, none a
    subclass of another, the parent a direct child of ``Phenotype`` with no field of
    its own and never a union member.
    """
    parent = s.TwoChannelMicroarrayExpressionPhenotype
    assert parent.__bases__[0] is s.Phenotype
    assert set(parent.model_fields) == set(s.Phenotype.model_fields)
    assert {c.__name__ for c in parent.__subclasses__()} == {
        c.__name__ for c in TWO_CHANNEL_LEAVES
    }
    for leaf in TWO_CHANNEL_LEAVES:
        assert leaf.__bases__[0] is parent, leaf.__name__
        assert not any(
            issubclass(leaf, other) for other in TWO_CHANNEL_LEAVES if other is not leaf
        ), leaf.__name__
    members = typing.get_args(s.PhenotypeType)
    assert parent not in members
    assert all(leaf in members for leaf in TWO_CHANNEL_LEAVES)
    assert [leaf.model_fields["label_name"].default for leaf in TWO_CHANNEL_LEAVES] == [
        "expression_log2_ratio",
        "expression_score",
        "expression_log2_ratio",
        "log2_cleaned",
    ]


def test_the_served_microarray_leaf_keeps_its_contract_under_the_parent() -> None:
    """Reparenting moved the served class's MRO only: same fields, same required ones."""
    fields = s.MicroarrayExpressionPhenotype.model_fields
    assert set(fields) == {
        "provenance_gaps",
        "graph_level",
        "label_name",
        "label_statistic_name",
        "expression_log2_ratio",
        "expression_log2_ratio_se",
        "expression",
        "expression_log2_ratio_variance",
        "n_replicates",
    }
    assert {name for name, info in fields.items() if info.is_required()} == {
        "expression_log2_ratio",
        "expression",
        "n_replicates",
    }


def test_the_two_channel_parent_is_never_a_record() -> None:
    """It declares no field, so no ``label_name`` can name one of its own.

    Checked in the state CI's full run reaches: once ``ABCMeta.__annotations__`` has
    been read, a class with no annotations of its own resolves ``__annotations__``
    through its MRO to ``Phenotype``'s, and the parent accepted
    ``label_name="graph_level"`` (CI, head ``5a3196648``). Its own empty
    ``__annotations__`` keeps the lookup on it in both states.
    """
    import abc

    assert isinstance(abc.ABCMeta.__annotations__, dict)  # the read CI's run makes
    assert s.TwoChannelMicroarrayExpressionPhenotype.__annotations__ == {}
    assert "__annotations__" in vars(s.TwoChannelMicroarrayExpressionPhenotype)
    with pytest.raises(ValidationError, match="must be a class attribute"):
        s.TwoChannelMicroarrayExpressionPhenotype(
            graph_level="node", label_name="graph_level"
        )


@pytest.mark.parametrize(
    "score,p_value",
    [
        (0.0, 1.0),
        (1.959963984540054, 0.05),
        (-1.959963984540054, 0.05),
        (3.090232306167813, 0.002),
    ],
)
def test_p_value_of_score_is_the_two_sided_normal_tail(
    score: float, p_value: float
) -> None:
    """``p == 2 * (1 - Phi(|X|))``: the conversion GSE4654's 1,707,575 pairs obey."""
    assert s.MicroarrayExpressionScorePhenotype.p_value_of_score(
        score
    ) == pytest.approx(p_value, rel=1e-12)


def test_p_value_of_score_keeps_the_far_tail_and_refuses_non_numbers() -> None:
    """``1 - Phi(10)`` rounds to 0 in doubles; the erfc form keeps it (about 1.5e-23)."""
    from scipy.stats import norm

    convert = s.MicroarrayExpressionScorePhenotype.p_value_of_score
    assert 1.0 - float(norm.cdf(10.0)) == 0.0
    assert convert(10.0) == pytest.approx(2.0 * float(norm.sf(10.0)), rel=1e-12)
    for bad in (math.inf, math.nan, True):
        with pytest.raises(ValueError, match="finite number"):
            convert(bad)


def _expression_ratio(**kw: Any) -> s.MicroarrayExpressionRatioPhenotype:
    fields: dict[str, Any] = dict(
        expression_log2_ratio={"YBR001C": 1.5, "YAL001C": -0.25},
        measurement_type="scanalyze_rat2n_normalized_log2_ratio",
        reference_basis="the same culture before the shift",
        array_platform="spotted yeast ORF array",
        provenance_gaps=[
            ProvenanceGap(
                field="n_replicates", reason=ProvenanceGapReason.not_reported_by_primary
            )
        ],
    )
    fields.update(kw)
    return s.MicroarrayExpressionRatioPhenotype(**fields)


def test_expression_ratio_sorts_its_keys_and_names_its_label() -> None:
    phenotype = _expression_ratio()
    assert list(phenotype.expression_log2_ratio.items()) == [
        ("YAL001C", -0.25),
        ("YBR001C", 1.5),
    ]
    assert (phenotype.label_name, phenotype.label_statistic_name) == (
        "expression_log2_ratio",
        None,
    )
    assert phenotype.n_replicates is None
    assert phenotype.gapped_fields() == {"n_replicates"}
    assert repr(phenotype) == (
        "MicroarrayExpressionRatioPhenotype(genes=2, n_replicates_genes=0)"
    )
    counted = _expression_ratio(
        n_replicates={"YBR001C": 2, "YAL001C": 1}, provenance_gaps=[]
    )
    assert counted.n_replicates is not None
    assert list(counted.n_replicates.items()) == [("YAL001C", 1), ("YBR001C", 2)]
    assert repr(counted) == (
        "MicroarrayExpressionRatioPhenotype(genes=2, n_replicates_genes=2)"
    )


@pytest.mark.parametrize(
    "fields,message",
    [
        ({"expression_log2_ratio": {}}, "expression_log2_ratio cannot be empty"),
        ({"expression_log2_ratio": [1.0]}, "must be a per-gene dict"),
        ({"expression_log2_ratio": {"YAL001C": True}}, "is not a number"),
        ({"expression_log2_ratio": {"YAL001C": math.nan}}, "is not finite"),
        ({"measurement_type": " "}, "cannot be blank"),
        ({"reference_basis": ""}, "cannot be blank"),
        ({"array_platform": "\t"}, "cannot be blank"),
        ({"provenance_gaps": []}, "carries no ProvenanceGap"),
        (
            {"n_replicates": {"YAL001C": 1, "YBR001C": 1}},
            "has a ProvenanceGap but is not None",
        ),
        ({"n_replicates": {"YAL001C": 1}, "provenance_gaps": []}, "same keys"),
        (
            {"n_replicates": {"YAL001C": 0, "YBR001C": 1}, "provenance_gaps": []},
            "positive integer",
        ),
        (
            {"n_replicates": {"YAL001C": True, "YBR001C": 1}, "provenance_gaps": []},
            "positive integer",
        ),
        ({"n_replicates": [1], "provenance_gaps": []}, "must be a per-gene dict"),
        ({"n_replicates": {}, "provenance_gaps": []}, "n_replicates cannot be empty"),
        ({"expression_log2_ratio_se": {"YAL001C": 0.1}}, "Extra inputs"),
    ],
)
def test_expression_ratio_refuses_what_a_ratio_record_cannot_be(
    fields: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _expression_ratio(**fields)


def test_expression_ratio_family_is_in_every_union_and_reconstructs_by_its_tag() -> (
    None
):
    assert s.MicroarrayExpressionRatioPhenotype in typing.get_args(s.PhenotypeType)
    assert s.MicroarrayExpressionRatioExperiment in typing.get_args(s.ExperimentType)
    assert s.MicroarrayExpressionRatioExperimentReference in typing.get_args(
        s.ExperimentReferenceType
    )
    s288c = s.ReferenceGenome(species="Saccharomyces cerevisiae", strain="S288C")
    ypd = Environment(media=Media(name="YPD", state="liquid", is_synthetic=False))
    experiment = s.MicroarrayExpressionRatioExperiment(
        dataset_name="Toy",
        genotype=Genotype(perturbations=[]),
        environment=ypd,
        phenotype=_expression_ratio(),
    )
    reference = s.MicroarrayExpressionRatioExperimentReference(
        dataset_name="Toy",
        genome_reference=s288c,
        environment_reference=ypd,
        phenotype_reference=_expression_ratio(
            expression_log2_ratio={"YAL001C": 0.0, "YBR001C": 0.0}
        ),
    )
    kind = experiment.experiment_type
    assert kind == "microarray_expression_ratio"
    assert reference.experiment_reference_type == kind
    assert _experiment_cls(kind).model_validate(experiment.model_dump()) == experiment
    assert _reference_cls(kind).model_validate(reference.model_dump()) == reference


# Hackett 2020 (2026.10.11): the inducible promoter insertion leaf and the induction
# time-course phenotype. Two genes, YAL001C and YAL002W; every layer the same keys.
# --------------------------------------------------------------------------- #
def _insertion(**overrides: Any) -> s.InduciblePromoterInsertionPerturbation:
    fields: dict[str, Any] = {
        "systematic_gene_name": "YER045C",
        "perturbed_gene_name": "ACA1",
        "promoter_name": "Z3EVpr",
        "cassette": "KanMX-Z3EVpr-ACA1",
        "marker": "KanMX",
        "responsive_to": "ZEV",
        "native_promoter_retained": True,
        "strain_id": "SMY2049",
    }
    return s.InduciblePromoterInsertionPerturbation(**(fields | overrides))


def test_inducible_insertion_is_an_increased_expression_insertion() -> None:
    leaf = _insertion()
    assert (
        leaf.perturbation_type,
        leaf.mechanism_so_id,
        leaf.mechanism_so_name,
        leaf.expression_direction,
        leaf.state,
        leaf.crispr,
    ) == (
        "inducible_promoter_insertion",
        "SO:0000667",
        "insertion",
        "increased",
        "present",
        None,
    )
    parsed = TypeAdapter(Genotype).validate_python(
        {"perturbations": [leaf.model_dump()]}
    )
    assert parsed.perturbations == [leaf]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"marker": "  "}, "marker cannot be empty"),
        ({"strain_id": ""}, "strain_id cannot be empty"),
        ({"expression_direction": "decreased"}, "must be 'increased'"),
        (
            {"crispr": s.CrisprConstruct(effector="dCas9")},
            "carries no CRISPR construct",
        ),
    ],
)
def test_inducible_insertion_refusals(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _insertion(**overrides)


_TWO = {"YAL001C": 0.5, "YAL002W": -0.25}


def _timecourse(**overrides: Any) -> s.InductionTimecourseExpressionPhenotype:
    fields: dict[str, Any] = {
        "log2_cleaned": dict(_TWO),
        "log2_noise_sd": {"YAL001C": 0.1, "YAL002W": 0.2},
        "sample_channel_intensity": {"YAL001C": 300.0, "YAL002W": 50.0},
        "reference_channel_intensity": {"YAL001C": 200.0, "YAL002W": 60.0},
        "log2_ratio_unprocessed": dict(_TWO),
        "log2_thresholded": {"YAL001C": 0.5, "YAL002W": 0.0},
        "log2_selected": {"YAL001C": 0.5, "YAL002W": 0.0},
        "log2_shrunken": {"YAL001C": 0.4, "YAL002W": 0.0},
        "timecourse_id": "ACA1_SMY2049_02/29/2016_P_ZEV",
        "strain_label": "SMY2049",
        "experiment_date": "02/29/2016",
        "release_table": "idea_tall_expression_data.zip",
    }
    return s.InductionTimecourseExpressionPhenotype(**(fields | overrides))


def test_timecourse_phenotype_sorts_its_layers_and_names_its_label() -> None:
    phen = _timecourse(log2_cleaned={"YAL002W": -0.25, "YAL001C": 0.5})
    # the before-validator sorts; pydantic then stores the sorted mapping as a dict
    assert list(phen.log2_cleaned) == ["YAL001C", "YAL002W"]
    assert (phen.graph_level, phen.label_name, phen.label_statistic_name) == (
        "node",
        "log2_cleaned",
        "log2_noise_sd",
    )
    assert repr(phen) == (
        "InductionTimecourseExpressionPhenotype("
        "timecourse_id='ACA1_SMY2049_02/29/2016_P_ZEV', genes=2)"
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"log2_selected": {"YAL001C": 0.5}}, "log2_selected must have the genes"),
        (
            {"reference_channel_intensity": {"YAL001C": -1.0, "YAL002W": 1.0}},
            "reference_channel_intensity cannot be negative",
        ),
        ({"log2_noise_sd": {"YAL001C": -0.1, "YAL002W": 0.1}}, "log2_noise_sd cannot"),
        ({"log2_cleaned": {"YAL001C": math.inf, "YAL002W": 0.0}}, "non-finite"),
        ({"log2_shrunken": {}}, "cannot be empty"),
        ({"log2_shrunken": None}, "log2_shrunken is unset"),
        ({"experiment_date": None}, "experiment_date is unset"),
        ({"timecourse_id": " "}, "timecourse_id cannot be empty"),
    ],
)
def test_timecourse_phenotype_refusals(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _timecourse(**overrides)


def test_timecourse_phenotype_accepts_a_gapped_layer() -> None:
    phen = _timecourse(
        log2_shrunken=None,
        provenance_gaps=[
            {"field": "log2_shrunken", "reason": "not_reported_by_primary"}
        ],
    )
    assert phen.log2_shrunken is None
    assert phen.gapped_fields() == {"log2_shrunken"}


def test_timecourse_family_is_in_every_union_and_map() -> None:
    assert s.InductionTimecourseExpressionPhenotype in typing.get_args(s.PhenotypeType)
    assert s.InductionTimecourseExpressionExperiment in typing.get_args(
        s.ExperimentType
    )
    assert s.InductionTimecourseExpressionExperimentReference in typing.get_args(
        s.ExperimentReferenceType
    )
    assert s.InduciblePromoterInsertionPerturbation in typing.get_args(
        s.GenePerturbationType
    )
    assert (
        s.EXPERIMENT_TYPE_MAP["induction_timecourse_expression"]
        is s.InductionTimecourseExpressionExperiment
    )
    assert (
        s.EXPERIMENT_REFERENCE_TYPE_MAP["induction_timecourse_expression"]
        is s.InductionTimecourseExpressionExperimentReference
    )


# --------------------------------------------------------------------------- #
# #691: a bacterial strain under its released identifier
# --------------------------------------------------------------------------- #
_RELEASED: dict[str, Any] = dict(
    systematic_gene_name="ECK2331",
    perturbed_gene_name="ECK2331",
    gene_namespace="ecoli_k12_bw25113_locus_tag",
    strain_label="ECK2331",
    identifier_kind="ECK accession",
    released_kind=s.ReleasedStrainKind.deletion,
)


def test_released_strain_leaf_stores_an_identifier_no_annotation_carries() -> None:
    leaf = s.BacterialReleasedStrainPerturbation(**_RELEASED, cassette="kanamycin")
    assert leaf.perturbation_type == "bacterial_released_strain"
    assert leaf.mechanism_so == ("SO:0000159", "deletion")
    assert leaf.collection is None
    back = s.Genotype.model_validate(
        s.Genotype(perturbations=[leaf]).model_dump()
    ).perturbations[0]
    assert back == leaf and type(back) is s.BacterialReleasedStrainPerturbation


@pytest.mark.parametrize(
    ("kind", "so"),
    [
        (s.ReleasedStrainKind.deletion, ("SO:0000159", "deletion")),
        (s.ReleasedStrainKind.affinity_tag, ("SO:0001218", "transgenic_insertion")),
        (s.ReleasedStrainKind.degron, ("SO:0001218", "transgenic_insertion")),
        (s.ReleasedStrainKind.linked_marker, ("SO:0001218", "transgenic_insertion")),
        (s.ReleasedStrainKind.sequence_allele, ("SO:0001060", "sequence_variant")),
        (s.ReleasedStrainKind.truncation, ("SO:0000159", "deletion")),
    ],
)
def test_released_kind_names_its_sequence_ontology_pair(
    kind: s.ReleasedStrainKind, so: tuple[str, str]
) -> None:
    leaf = s.BacterialReleasedStrainPerturbation(**{**_RELEASED, "released_kind": kind})
    assert leaf.mechanism_so == so


@pytest.mark.parametrize(
    "identifier", ["BW25113_0001", "b0002", "PP_0001", " ECK2331", ""]
)
def test_released_strain_leaf_refuses_a_locus_tag_or_a_non_verbatim_identifier(
    identifier: str,
) -> None:
    with pytest.raises(ValueError):
        s.BacterialReleasedStrainPerturbation(
            **{**_RELEASED, "systematic_gene_name": identifier}
        )


@pytest.mark.parametrize("field", ["strain_label", "identifier_kind"])
def test_released_strain_leaf_refuses_an_empty_label(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be a non-empty verbatim"):
        s.BacterialReleasedStrainPerturbation(**{**_RELEASED, field: " "})


# --------------------------------------------------------------------------- #
# #1045: a yeast gene under the name its source released, and a named allele
# --------------------------------------------------------------------------- #
_RELEASED_NAME: dict[str, Any] = dict(
    systematic_gene_name="ADE2",
    perturbed_gene_name="ADE2",
    released_gene_name="ADE2",
    released_kind=s.ReleasedGeneKind.deletion,
)


def test_released_gene_name_leaf_stores_the_symbol_no_annotation_resolved() -> None:
    leaf = s.ReleasedGeneNamePerturbation(
        **_RELEASED_NAME, marker="KanMX", strain_id="BY4741"
    )
    assert leaf.perturbation_type == "released_gene_name"
    assert leaf.mechanism_so == ("SO:0000159", "deletion")
    assert leaf.released_gene_id is None and leaf.crispr is None
    back = s.Genotype.model_validate(
        s.Genotype(perturbations=[leaf]).model_dump()
    ).perturbations[0]
    assert back == leaf and type(back) is s.ReleasedGeneNamePerturbation


def test_released_gene_name_leaf_carries_a_released_id_with_its_namespace() -> None:
    leaf = s.ReleasedGeneNamePerturbation(
        **_RELEASED_NAME,
        released_gene_id="856882",
        released_gene_id_namespace=s.ReleasedGeneIdNamespace.ncbi_entrez_gene,
    )
    assert leaf.released_gene_id == "856882"
    with pytest.raises(ValueError, match="set together"):
        s.ReleasedGeneNamePerturbation(**_RELEASED_NAME, released_gene_id="856882")
    with pytest.raises(ValueError, match="set together"):
        s.ReleasedGeneNamePerturbation(
            **_RELEASED_NAME,
            released_gene_id_namespace=s.ReleasedGeneIdNamespace.ncbi_entrez_gene,
        )
    with pytest.raises(ValueError, match="non-empty verbatim"):
        s.ReleasedGeneNamePerturbation(
            **_RELEASED_NAME,
            released_gene_id=" 856882",
            released_gene_id_namespace=s.ReleasedGeneIdNamespace.ncbi_entrez_gene,
        )


def test_released_gene_name_leaf_crispr_interference_carries_its_construct() -> None:
    construct = s.CrisprConstruct(
        effector="dCas9-Mxi1", guide_sequence="ACGTACGTACGTACGTACGT"
    )
    leaf = s.ReleasedGeneNamePerturbation(
        **{**_RELEASED_NAME, "released_kind": s.ReleasedGeneKind.crispr_interference},
        crispr=construct,
    )
    assert leaf.mechanism_so == ("SO:0001998", "sgRNA")
    with pytest.raises(ValueError, match="crispr is set exactly when"):
        s.ReleasedGeneNamePerturbation(
            **{
                **_RELEASED_NAME,
                "released_kind": s.ReleasedGeneKind.crispr_interference,
            }
        )
    with pytest.raises(ValueError, match="crispr is set exactly when"):
        s.ReleasedGeneNamePerturbation(**_RELEASED_NAME, crispr=construct)


def test_released_gene_name_leaf_carries_a_systematic_name_whose_kind_is_a_gap() -> (
    None
):
    """SynLethDB releases names like YPR108W-A with no stated kind: the released leaf
    carries them, since no placed leaf claims a kind the source does not state.
    """
    gap = ProvenanceGap(
        field="released_kind", reason=ProvenanceGapReason.not_reported_by_primary
    )
    leaf = s.ReleasedGeneNamePerturbation(
        systematic_gene_name="YPR108W-A",
        perturbed_gene_name="YPR108W-A",
        released_gene_name="YPR108W-A",
        released_gene_id="1466522",
        released_gene_id_namespace=s.ReleasedGeneIdNamespace.ncbi_entrez_gene,
        released_kind=None,
        provenance_gaps=[gap],
    )
    assert leaf.systematic_gene_name == "YPR108W-A"


@pytest.mark.parametrize("identifier", ["b0002", "PP_0001", " ADE2", ""])
def test_released_gene_name_leaf_refuses_a_locus_tag_or_a_non_verbatim_name(
    identifier: str,
) -> None:
    with pytest.raises(ValueError):
        s.ReleasedGeneNamePerturbation(
            **{
                **_RELEASED_NAME,
                "systematic_gene_name": identifier,
                "released_gene_name": identifier,
            }
        )


def test_released_gene_name_leaf_kind_is_set_or_a_typed_gap() -> None:
    gap = ProvenanceGap(
        field="released_kind", reason=ProvenanceGapReason.not_reported_by_primary
    )
    leaf = s.ReleasedGeneNamePerturbation(
        **{**_RELEASED_NAME, "released_kind": None}, provenance_gaps=[gap]
    )
    assert leaf.mechanism_so is None and leaf.gapped_fields() == {"released_kind"}
    assert hash(leaf) == hash(leaf.model_copy())
    back = s.Genotype.model_validate(
        s.Genotype(perturbations=[leaf]).model_dump()
    ).perturbations[0]
    assert back == leaf and type(back) is s.ReleasedGeneNamePerturbation
    with pytest.raises(ValueError, match="released_kind is unset and carries no"):
        s.ReleasedGeneNamePerturbation(**{**_RELEASED_NAME, "released_kind": None})


def test_released_gene_name_leaf_states_the_name_once() -> None:
    with pytest.raises(ValueError, match="must equal"):
        s.ReleasedGeneNamePerturbation(
            **{**_RELEASED_NAME, "released_gene_name": "ade2"}
        )


def test_named_allele_leaf_keeps_two_alleles_of_one_gene_apart() -> None:
    a = s.NamedAllelePerturbation(
        systematic_gene_name="YLR229C",
        perturbed_gene_name="CDC42",
        allele_name="cdc42-111",
    )
    b = s.NamedAllelePerturbation(
        systematic_gene_name="YLR229C",
        perturbed_gene_name="CDC42",
        allele_name="cdc42-1",
    )
    assert a != b and a.perturbation_type == "named_allele"
    assert (a.mechanism_so_id, a.mechanism_so_name) == (
        "SO:0001060",
        "sequence_variant",
    )
    back = s.Genotype.model_validate(
        s.Genotype(perturbations=[a]).model_dump()
    ).perturbations[0]
    assert back == a and type(back) is s.NamedAllelePerturbation
    with pytest.raises(ValueError, match="non-empty verbatim"):
        s.NamedAllelePerturbation(
            systematic_gene_name="YLR229C", perturbed_gene_name="CDC42", allele_name=" "
        )


# --------------------------------------------------------------------------- #
# #991 / #990: the promoter-carrying genotype leaves and the two readout families
# --------------------------------------------------------------------------- #
def test_promoter_driven_integration_requires_its_locus_and_promoter() -> None:
    """The integration leaf narrows the parent's optional locus to a required one."""
    leaf = s.PromoterDrivenGeneIntegrationPerturbation(
        systematic_gene_name="YER090W",
        perturbed_gene_name="TRP2",
        source_organism="Saccharomyces cerevisiae",
        is_heterologous=False,
        localization="chromosomal_integration",
        promoter_name="pTEF1",
        integration_locus="EasyClone XII-5",
    )
    assert leaf.perturbation_type == "promoter_driven_gene_integration"
    assert (leaf.state, leaf.mechanism_so_id) == ("present", "SO:0000667")
    assert isinstance(leaf, s.GeneAdditionPerturbation)
    assert leaf.promoter_sequence is None and leaf.is_inducible is False
    with pytest.raises(ValidationError):
        s.PromoterDrivenGeneIntegrationPerturbation(  # type: ignore[call-arg]
            systematic_gene_name="YER090W",
            perturbed_gene_name="TRP2",
            source_organism="Saccharomyces cerevisiae",
            is_heterologous=False,
            localization="chromosomal_integration",
            promoter_name="pTEF1",
        )
    with pytest.raises(ValidationError, match="cannot be empty"):
        s.PromoterDrivenGeneIntegrationPerturbation(
            systematic_gene_name="YER090W",
            perturbed_gene_name="TRP2",
            source_organism="Saccharomyces cerevisiae",
            is_heterologous=False,
            localization="chromosomal_integration",
            promoter_name="   ",
            integration_locus="EasyClone XII-5",
        )


def test_promoter_driven_integration_refuses_a_deposited_sequence_that_is_not_dna() -> (
    None
):
    """A promoter sequence is an identity key, so a degenerate spelling is refused."""
    with pytest.raises(ValidationError, match="unambiguous DNA"):
        s.PromoterDrivenGeneIntegrationPerturbation(
            systematic_gene_name="YER090W",
            perturbed_gene_name="TRP2",
            source_organism="Saccharomyces cerevisiae",
            is_heterologous=False,
            localization="chromosomal_integration",
            promoter_name="pTEF1",
            promoter_sequence="acgtn",
            integration_locus="EasyClone XII-5",
        )


def test_variable_promoter_reporter_requires_the_sequence_and_a_whole_span() -> None:
    """The sequence IS the genotype here, and half a span does not locate the insert."""

    def reporter(
        sequence: str | None = "ACGTACGTAC", start: int | None = 3, end: int | None = 8
    ) -> s.VariablePromoterReporterPerturbation:
        return s.VariablePromoterReporterPerturbation(
            systematic_gene_name="addgene127546:YFP",
            perturbed_gene_name="YFP",
            source_organism="synthetic construct",
            is_heterologous=True,
            localization="episomal_cen_plasmid",
            promoter_variant_id="HQ_28",
            promoter_sequence=sequence,  # type: ignore[arg-type]
            scaffold_name="pTpA",
            variable_region_start=start,
            variable_region_end=end,
        )

    leaf = reporter()
    assert leaf.perturbation_type == "variable_promoter_reporter"
    assert leaf.integration_locus is None
    with pytest.raises(ValidationError):
        reporter(sequence=None)
    with pytest.raises(ValidationError, match="both be set or"):
        reporter(end=None)
    with pytest.raises(ValidationError, match="is before"):
        reporter(start=8, end=3)


def test_both_promoter_leaves_are_members_of_the_genotype_union() -> None:
    """A Genotype admits each leaf and keeps its own type through the union."""
    integration = s.PromoterDrivenGeneIntegrationPerturbation(
        systematic_gene_name="YER090W",
        perturbed_gene_name="TRP2",
        source_organism="Saccharomyces cerevisiae",
        is_heterologous=False,
        localization="chromosomal_integration",
        promoter_name="pTEF1",
        integration_locus="EasyClone XII-5",
    )
    reporter = s.VariablePromoterReporterPerturbation(
        systematic_gene_name="addgene127546:YFP",
        perturbed_gene_name="YFP",
        source_organism="synthetic construct",
        is_heterologous=True,
        localization="episomal_cen_plasmid",
        promoter_variant_id="HQ_28",
        promoter_sequence="ACGTACGTAC",
        scaffold_name="pTpA",
    )
    genotype = s.Genotype(perturbations=[integration, reporter])
    assert sorted(genotype.perturbation_types) == [
        "promoter_driven_gene_integration",
        "variable_promoter_reporter",
    ]
    round_tripped = s.Genotype.model_validate(genotype.model_dump())
    assert round_tripped == genotype


def test_pooled_sort_seq_phenotype_keys_on_the_sequence_and_rejects_blanks() -> None:
    """The pooled family's identity key is the sequence id, and it cannot be blank."""
    phenotype = s.PooledSortSeqExpressionPhenotype(
        expression_level=8.5,
        sequence_id="HQ_28",
        library_id="NBT_S288CdU_SCUra",
        read_count=647,
        readout=s.ReporterReadout.sort_seq_fluorescence,
        reporter_gene="YFP",
        expression_units="sort-bin index",
    )
    assert (phenotype.label_name, phenotype.label_statistic_name) == (
        "expression_level",
        "expression_level_se",
    )
    assert phenotype.expression_level_se is None
    with pytest.raises(ValidationError, match="sequence_id cannot be empty"):
        s.PooledSortSeqExpressionPhenotype(
            expression_level=8.5,
            sequence_id="  ",
            readout=s.ReporterReadout.sort_seq_fluorescence,
            reporter_gene="YFP",
            expression_units="sort-bin index",
        )
    with pytest.raises(ValidationError, match="read_count must be a positive integer"):
        s.PooledSortSeqExpressionPhenotype(
            expression_level=8.5,
            sequence_id="HQ_28",
            read_count=0,
            readout=s.ReporterReadout.sort_seq_fluorescence,
            reporter_gene="YFP",
            expression_units="sort-bin index",
        )
    with pytest.raises(ValidationError, match="no unlabelled uncertainty"):
        s.PooledSortSeqExpressionPhenotype(
            expression_level=8.5,
            expression_level_uncertainty=1.0,
            sequence_id="HQ_28",
            readout=s.ReporterReadout.sort_seq_fluorescence,
            reporter_gene="YFP",
            expression_units="sort-bin index",
        )


def test_biosensor_phenotype_names_the_sensed_metabolite() -> None:
    """The signal stays in its own units and the sensed compound is typed."""
    phenotype = s.BiosensorFluorescencePhenotype(
        biosensor_signal=1.25,
        biosensor_signal_uncertainty=0.5,
        biosensor_signal_uncertainty_type=s.UncertaintyType.sample_sd,
        n_samples=4,
        sample_unit=s.SampleUnit.biological_replicate,
        sensed_metabolite=s.Compound(name="L-tryptophan"),
        sensor_name="trpR-based tryptophan biosensor",
        reporter_gene="yEGFP",
        readout=s.ReporterReadout.flow_cytometry_fluorescence,
        signal_units="specific GFP synthesis rate (MFI/h)",
    )
    assert phenotype.label_name == "biosensor_signal"
    assert phenotype.biosensor_signal_se == 0.25
    assert phenotype.sensed_metabolite.name == "L-tryptophan"
    with pytest.raises(ValidationError, match="n_samples and sample_unit are required"):
        s.BiosensorFluorescencePhenotype(
            biosensor_signal=1.25,
            biosensor_signal_uncertainty=0.5,
            biosensor_signal_uncertainty_type=s.UncertaintyType.sample_sd,
            sensed_metabolite=s.Compound(name="L-tryptophan"),
            sensor_name="trpR-based tryptophan biosensor",
            reporter_gene="yEGFP",
            readout=s.ReporterReadout.flow_cytometry_fluorescence,
            signal_units="specific GFP synthesis rate (MFI/h)",
        )


def test_the_two_readout_families_are_siblings_not_promoter_activity() -> None:
    """Each new leaf is its own phenotype under Phenotype, and the served one is intact."""
    for leaf in (s.PooledSortSeqExpressionPhenotype, s.BiosensorFluorescencePhenotype):
        assert leaf.__bases__[0] is s.Phenotype
        assert not issubclass(leaf, s.PromoterActivityPhenotype)
    assert s.PromoterActivityPhenotype.model_fields["well_id"].is_required()
    assert "sort_seq_fluorescence" in {member.value for member in s.ReporterReadout}
    assert s.EXPERIMENT_TYPE_MAP["pooled_sort_seq_expression"] is (
        s.PooledSortSeqExpressionExperiment
    )
    assert s.EXPERIMENT_TYPE_MAP["biosensor_fluorescence"] is (
        s.BiosensorFluorescenceExperiment
    )


def test_the_pooled_leaf_takes_a_value_or_a_gap_on_its_label_never_a_silent_none() -> (
    None
):
    """A release row with no estimate is a record; the absence is typed, not dropped.

    The owner rule the leaf follows: what the release lists is what the store holds, so
    an unmeasured row is kept with ``expression_level=None`` beside a ``ProvenanceGap``
    on that field. A silent ``None`` is refused, as is a gap beside a stated value.
    """

    def phenotype(
        level: float | None, gaps: list[ProvenanceGap] | None = None
    ) -> s.PooledSortSeqExpressionPhenotype:
        return s.PooledSortSeqExpressionPhenotype(
            expression_level=level,
            sequence_id="GA.EDV.SCUra.high_248",
            readout=s.ReporterReadout.sort_seq_fluorescence,
            reporter_gene="YFP",
            expression_units="sort-bin index",
            provenance_gaps=gaps or [],
        )

    gap = ProvenanceGap(
        field="expression_level",
        reason=ProvenanceGapReason.not_reported_by_primary,
        note="the release writes 'NA' in this row's ExpressionLevel cell",
    )
    absent = phenotype(None, [gap])
    assert absent.expression_level is None
    assert absent.gapped_fields() == {"expression_level"}
    with pytest.raises(ValidationError, match="carries no ProvenanceGap"):
        phenotype(None)
    assert phenotype(8.5).expression_level == 8.5


# #691 round 2: a CRISPR guide and a transposon insertion against a released identifier
# --------------------------------------------------------------------------- #
_RELEASED_CRISPR: dict[str, Any] = dict(
    systematic_gene_name="sokE",
    perturbed_gene_name="sokE",
    gene_namespace="ecoli_k12_mg1655_bnumber",
    strain_label="sokEb4700_32",
    identifier_kind="gene symbol",
    expression_direction="decreased",
    crispr=s.CrisprConstruct(effector="dCas9", guide_sequence="CGTACTACCAGATAACCTAA"),
)
_RELEASED_TRANSPOSON: dict[str, Any] = dict(
    systematic_gene_name="insA",
    perturbed_gene_name="insA",
    gene_namespace="ecoli_k12_bw25113_locus_tag",
    strain_label="insA",
    identifier_kind="gene symbol",
)


def test_released_crispr_leaf_stores_the_construct_against_a_released_label() -> None:
    leaf = s.BacterialReleasedCrisprPerturbation(**_RELEASED_CRISPR)
    assert leaf.perturbation_type == "bacterial_released_crispr"
    assert leaf.mechanism_so == ("SO:0001998", "sgRNA")
    assert leaf.crispr.guide_sequence == "CGTACTACCAGATAACCTAA"
    back = s.Genotype.model_validate(
        s.Genotype(perturbations=[leaf]).model_dump()
    ).perturbations[0]
    assert back == leaf and type(back) is s.BacterialReleasedCrisprPerturbation


@pytest.mark.parametrize("direction", ["increased", "decreased"])
def test_released_crispr_leaf_takes_either_direction(direction: str) -> None:
    leaf = s.BacterialReleasedCrisprPerturbation(
        **{**_RELEASED_CRISPR, "expression_direction": direction}
    )
    assert leaf.expression_direction == direction


def test_released_crispr_leaf_requires_a_stated_direction() -> None:
    with pytest.raises(ValueError, match="expression_direction"):
        s.BacterialReleasedCrisprPerturbation(
            **{**_RELEASED_CRISPR, "expression_direction": "knockdown"}
        )
    with pytest.raises(ValueError):
        s.BacterialReleasedCrisprPerturbation(
            **{k: v for k, v in _RELEASED_CRISPR.items() if k != "expression_direction"}
        )


def test_released_transposon_leaf_stores_gene_level_calls_and_mapped_sites() -> None:
    gene_level = s.BacterialReleasedTransposonPerturbation(**_RELEASED_TRANSPOSON)
    assert gene_level.perturbation_type == "bacterial_released_transposon"
    assert gene_level.mechanism_so == ("SO:0001218", "transgenic_insertion")
    assert gene_level.insertion_position is None and gene_level.barcode is None
    mapped = s.BacterialReleasedTransposonPerturbation(
        **_RELEASED_TRANSPOSON,
        barcode="ACGTACGTACGTACGTACGT",
        insertion_position=12345,
        insertion_strand="+",
        transposon="mariner Himar1",
        library_pool="pool 1",
    )
    for leaf in (gene_level, mapped):
        back = s.Genotype.model_validate(
            s.Genotype(perturbations=[leaf]).model_dump()
        ).perturbations[0]
        assert back == leaf
        assert type(back) is s.BacterialReleasedTransposonPerturbation
    with pytest.raises(ValueError, match="1-based"):
        s.BacterialReleasedTransposonPerturbation(
            **_RELEASED_TRANSPOSON, insertion_position=0
        )


@pytest.mark.parametrize(
    "identifier", ["BW25113_0001", "b0002", "PP_0001", "ECB_00001", " sokE", ""]
)
def test_released_construct_leaves_refuse_a_locus_tag_or_a_non_verbatim_identifier(
    identifier: str,
) -> None:
    with pytest.raises(ValueError):
        s.BacterialReleasedCrisprPerturbation(
            **{**_RELEASED_CRISPR, "systematic_gene_name": identifier}
        )
    with pytest.raises(ValueError):
        s.BacterialReleasedTransposonPerturbation(
            **{**_RELEASED_TRANSPOSON, "systematic_gene_name": identifier}
        )


@pytest.mark.parametrize("field", ["strain_label", "identifier_kind"])
def test_released_construct_leaves_refuse_an_empty_label(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be a non-empty verbatim"):
        s.BacterialReleasedCrisprPerturbation(**{**_RELEASED_CRISPR, field: " "})
    with pytest.raises(ValueError, match=f"{field} must be a non-empty verbatim"):
        s.BacterialReleasedTransposonPerturbation(
            **{**_RELEASED_TRANSPOSON, field: " "}
        )


def test_released_construct_leaves_are_gene_perturbation_siblings_of_the_strain_leaf() -> (
    None
):
    """Parent GenePerturbation, not an axis class: the placed leaves of each technology
    keep their locus-tag contract and the three released leaves sit beside them.
    """
    members = typing.get_args(s.GenePerturbationType)
    for cls in (
        s.BacterialReleasedCrisprPerturbation,
        s.BacterialReleasedTransposonPerturbation,
    ):
        assert cls in members
        assert cls.__mro__[1] is s.GenePerturbation
        assert not issubclass(cls, s.ExpressionModulationPerturbation)
        assert not issubclass(cls, s.PresenceAbsencePerturbation)
    assert s.BacterialReleasedStrainPerturbation.__mro__[1] is s.GenePerturbation
    placed = s.BacterialCrisprInterferencePerturbation.model_fields
    assert "strain_label" not in placed and "identifier_kind" not in placed


def test_fold_change_is_a_dimensionless_concentration_unit() -> None:
    assert s.ConcentrationUnit.fold_change.value == "fold_change"
    assert s.ConcentrationUnit("fold_change") is s.ConcentrationUnit.fold_change
    assert s.ConcentrationUnit.mg_nitrogen_per_l.value == "mg_N/L"


# #879: the released Z score of the response.


def test_environment_response_z_score_is_stored_and_never_feeds_the_se() -> None:
    """Wang 2018's released Z score is stored verbatim and leaves the SE alone."""
    stored = _response(environment_response=-0.75, environment_response_z_score=-0.88)
    assert stored.environment_response_z_score == -0.88
    assert stored.environment_response_se is None
    assert _response().environment_response_z_score is None
    assert (
        _response(environment_response_z_score=0.0).environment_response_z_score == 0.0
    )


def test_environment_response_z_score_is_finite_and_scores_a_number() -> None:
    with _refuses("environment_response_z_score must be finite, got nan"):
        _response(environment_response_z_score=float("nan"))
    with _refuses("environment_response_z_score must be finite, got -inf"):
        _response(environment_response_z_score=float("-inf"))
    with _refuses("environment_response_z_score scores a numeric response"):
        _response(
            measurement_type=s.MeasurementType.categorical,
            environment_response=None,
            category=s.ResponseCategory.reduced,
            environment_response_z_score=1.0,
        )


# #915: a typed gap on a pathway part's source organism.


def _pathway_part(**overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "systematic_gene_name": "mvaS",
        "perturbed_gene_name": "mvaS",
        "gene_namespace": "pputida_kt2440_locus_tag",
        "is_heterologous": True,
        "localization": "chromosomal_integration",
        "pathway_name": "isoprenol via mevalonate",
    }
    kwargs.update(overrides)
    return s.HeterologousPathwayPerturbation(**kwargs)


def _organism_gap() -> Any:
    from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

    return ProvenanceGap(
        field="source_organism",
        reason=ProvenanceGapReason.not_reported_by_primary,
        note="the plasmid table names the part token and no organism",
    )


def test_a_pathway_part_states_its_organism_or_carries_a_typed_gap() -> None:
    """The organism is stated, or None beside a gap naming it; never a bare None."""
    stated = _pathway_part(source_organism="Enterococcus faecalis")
    assert stated.source_organism == "Enterococcus faecalis"
    assert stated.provenance_gaps == []
    gapped = _pathway_part(source_organism=None, provenance_gaps=[_organism_gap()])
    assert gapped.source_organism is None
    assert gapped.gapped_fields() == {"source_organism"}
    with _refuses(
        "HeterologousPathwayPerturbation.source_organism is unset and carries no "
        "ProvenanceGap"
    ):
        _pathway_part(source_organism=None)
    with _refuses("field 'source_organism' has a ProvenanceGap but is not None"):
        _pathway_part(
            source_organism="Enterococcus faecalis", provenance_gaps=[_organism_gap()]
        )
    with pytest.raises(ValidationError, match="source_organism"):
        s.HeterologousPathwayPerturbation(  # type: ignore[call-arg]  # the refusal under test
            systematic_gene_name="mvaS",
            perturbed_gene_name="mvaS",
            gene_namespace="pputida_kt2440_locus_tag",
            is_heterologous=True,
            localization="chromosomal_integration",
            pathway_name="isoprenol via mevalonate",
        )


def test_a_gapped_pathway_part_hashes_and_compares_inside_a_genotype() -> None:
    """``Genotype`` compares perturbations as a set, so the gap carrier must hash."""
    part = _pathway_part(source_organism=None, provenance_gaps=[_organism_gap()])
    twin = _pathway_part(source_organism=None, provenance_gaps=[_organism_gap()])
    assert hash(part) == hash(twin)
    assert Genotype(perturbations=[part]) == Genotype(perturbations=[twin])
    assert Genotype(perturbations=[part]) != Genotype(
        perturbations=[_pathway_part(source_organism="Enterococcus faecalis")]
    )
