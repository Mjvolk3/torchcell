# tests/torchcell/verification/test_protein_fold_change.py
# [[tests.torchcell.verification.test_protein_fold_change]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/verification/test_protein_fold_change.py
"""Unit tests for the protein FOLD-CHANGE verifier (#770), on synthetic records.

The sibling of ``test_protein_verification`` on the relative axis. Each of the
verifier's six own rules gets one exact pass and one exact refusal, because the rules
that matter for a ratio do not exist for an absolute abundance: a scale, a named
denominator, a reference that is the scale's neutral value, and a per-protein p-value.

The synthetic panel is two bacterial deletion strains (``PP_0812``, ``PP_0815``) over two
proteins (``PP_4188``, ``PP_0368``) on the log2 scale, so the neutral reference is 0.0
and a negative stored value is ordinary rather than refused. One test switches the panel
to the LINEAR scale, where the neutral reference is 1.0 instead, which is the whole
reason ``fold_change_scale`` is a required field.
"""

from __future__ import annotations

import math
from typing import Any

from torchcell.datamodels.schema import (
    AssemblyReferenceGenome,
    BacterialDeletionPerturbation,
    BacterialProteinFoldChangeExperiment,
    BacterialProteinFoldChangeExperimentReference,
    Environment,
    FoldChangeScale,
    Genotype,
    Media,
    ProteinFoldChangePhenotype,
    Temperature,
)
from torchcell.verification.protein_fold_change import (
    fold_change_p_value_round_trip,
    protein_fold_change_locus_set,
    verify_protein_fold_change_dataset,
)
from torchcell.verification.report import Level, Provenance, VerificationReport

PROV = Provenance(source_uri="test://synthetic", citation_key="test2025")
PROTEINS = ["PP_4188", "PP_0368"]
STRAINS = ["PP_0812", "PP_0815"]
BASIS = "control strain carrying a non-targeting sgRNA"
MTYPE = "dia_log2_fold_change_paired_two_tailed_t_test"
ASSEMBLY = AssemblyReferenceGenome(
    species="Pseudomonas putida",
    strain="KT2440",
    assembly_set="pputida_KT2440_ASM756v2",
    assembly_accession="GCA_000007565.2",
)


def _environment(temperature: float = 30.0) -> Environment:
    return Environment(
        media=Media(name="M9", state="liquid", is_synthetic=True),
        temperature=Temperature(value=temperature),
    )


def _phenotype(
    offset: float,
    *,
    scale: FoldChangeScale = FoldChangeScale.log2,
    basis: str = BASIS,
    measurement_type: str = MTYPE,
    p_values: dict[str, float] | None = None,
) -> ProteinFoldChangePhenotype:
    linear = scale is FoldChangeScale.linear
    values = {
        protein: (0.5 + offset + index if linear else -1.5 + offset + index)
        for index, protein in enumerate(PROTEINS)
    }
    return ProteinFoldChangePhenotype(
        protein_fold_change=values,
        fold_change_scale=scale,
        reference_basis=basis,
        protein_fold_change_p_value=(
            {protein: 0.01 for protein in PROTEINS} if p_values is None else p_values
        ),
        n_replicates={protein: 3 for protein in PROTEINS},
        measurement_type=measurement_type,
    )


def _reference_phenotype(
    experiment_phenotype: ProteinFoldChangePhenotype,
    *,
    scale: FoldChangeScale | None = None,
    values: dict[str, float] | None = None,
) -> ProteinFoldChangePhenotype:
    """The denominator: the scale's neutral value for every key, unless overridden."""
    resolved = scale or experiment_phenotype.fold_change_scale
    resolved_values = values or experiment_phenotype.neutral_reference()
    return ProteinFoldChangePhenotype(
        protein_fold_change=resolved_values,
        fold_change_scale=resolved,
        reference_basis=experiment_phenotype.reference_basis,
        n_replicates=dict.fromkeys(resolved_values, 3),
        measurement_type=experiment_phenotype.measurement_type,
    )


def _record(
    strain: str,
    offset: float,
    *,
    scale: FoldChangeScale = FoldChangeScale.log2,
    basis: str = BASIS,
    measurement_type: str = MTYPE,
    p_values: dict[str, float] | None = None,
    temperature: float = 30.0,
    reference_scale: FoldChangeScale | None = None,
    reference_values: dict[str, float] | None = None,
) -> dict[str, Any]:
    phenotype = _phenotype(
        offset,
        scale=scale,
        basis=basis,
        measurement_type=measurement_type,
        p_values=p_values,
    )
    environment = _environment(temperature)
    experiment = BacterialProteinFoldChangeExperiment(
        dataset_name="test",
        genotype=Genotype(
            perturbations=[
                BacterialDeletionPerturbation(
                    systematic_gene_name=strain,
                    perturbed_gene_name=strain,
                    gene_namespace="pputida_kt2440_locus_tag",
                )
            ]
        ),
        environment=environment,
        phenotype=phenotype,
    )
    reference = BacterialProteinFoldChangeExperimentReference(
        dataset_name="test",
        genome_reference=ASSEMBLY,
        environment_reference=environment.model_copy(),
        phenotype_reference=_reference_phenotype(
            phenotype, scale=reference_scale, values=reference_values
        ),
    )
    return {"experiment": experiment.model_dump(), "reference": reference.model_dump()}


def _records() -> list[dict[str, Any]]:
    return [_record(strain, offset) for strain, offset in zip(STRAINS, [0.0, 2.0])]


def _result(report: VerificationReport, name: str) -> Any:
    [match] = [result for result in report.results if result.name == name]
    return match


def test_a_clean_fold_change_panel_passes_every_level() -> None:
    """Six own rules plus L0 and L1 count, and the exact messages each one reports."""
    report = verify_protein_fold_change_dataset(
        _records(), dataset_name="good", provenance=PROV, expected_count=2
    )
    assert report.passed
    assert [result.name for result in report.results] == [
        "structural",
        "count",
        "contrast_uniqueness",
        "value_fidelity",
        "p_values_are_probabilities",
        "reference_is_the_scales_neutral_value",
        "fold_change_scale_consistent",
        "measurement_type_consistent",
    ]
    contrast = _result(report, "contrast_uniqueness")
    assert contrast.level is Level.L1
    assert contrast.message == "2 distinct contrasts, one record each"
    assert contrast.details == {"n_contrasts": 2, "n_duplicated": 0}
    p_values = _result(report, "p_values_are_probabilities")
    assert p_values.level is Level.L2
    assert p_values.message == "all 4 p-values lie in (0, 1]"
    assert p_values.details == {"n_values": 4, "n_bad": 0}
    assert _result(report, "fold_change_scale_consistent").passed
    assert _result(report, "measurement_type_consistent").passed


def test_two_records_of_one_contrast_fail_uniqueness() -> None:
    """Same genotype, same environment, same denominator is one measurement twice."""
    duplicated = [_record("PP_0812", 0.0), _record("PP_0812", 4.0)]
    report = verify_protein_fold_change_dataset(
        duplicated, dataset_name="dup", provenance=PROV, expected_count=2
    )
    result = _result(report, "contrast_uniqueness")
    assert not result.passed
    assert result.message == "1 contrasts appear in more than one record"
    assert result.details == {"n_contrasts": 1, "n_duplicated": 1}


def test_the_environment_separates_two_contrasts_of_one_genotype() -> None:
    """A wild-type panel varies the environment, so the environment is part of the key."""
    records = [
        _record("PP_0812", 0.0, temperature=30.0),
        _record("PP_0812", 0.0, temperature=37.0),
    ]
    result = _result(
        verify_protein_fold_change_dataset(
            records, dataset_name="env", provenance=PROV, expected_count=2
        ),
        "contrast_uniqueness",
    )
    assert result.passed
    assert result.details == {"n_contrasts": 2, "n_duplicated": 0}


def test_the_denominator_separates_two_contrasts_of_one_genotype() -> None:
    """One paper can vary the denominator within itself (POI:Control vs dCas9:Control)."""
    records = [
        _record("PP_0812", 0.0),
        _record("PP_0812", 0.0, basis="dCas9 over the same protein in the control"),
    ]
    result = _result(
        verify_protein_fold_change_dataset(
            records, dataset_name="basis", provenance=PROV, expected_count=2
        ),
        "contrast_uniqueness",
    )
    assert result.passed
    assert result.details == {"n_contrasts": 2, "n_duplicated": 0}


def test_a_p_value_of_exactly_zero_is_refused_although_the_schema_admits_it() -> None:
    """The verifier's (0, 1] gate is tighter than the class's [0, 1] on purpose."""
    records = [
        _record("PP_0812", 0.0, p_values={"PP_4188": 0.0, "PP_0368": 0.01}),
        _record("PP_0815", 2.0),
    ]
    result = _result(
        verify_protein_fold_change_dataset(
            records, dataset_name="zero_p", provenance=PROV, expected_count=2
        ),
        "p_values_are_probabilities",
    )
    assert not result.passed
    assert result.message == "1 p-values are not in (0, 1]"
    assert result.details == {"n_values": 4, "n_bad": 1}


def test_a_reference_that_is_not_the_scales_neutral_value_is_refused() -> None:
    """0.0 is neutral on log2 and 1.0 on linear, so 1.0 on a log2 panel is wrong."""
    records = [
        _record("PP_0812", 0.0, reference_values=dict.fromkeys(PROTEINS, 1.0)),
        _record("PP_0815", 2.0),
    ]
    result = _result(
        verify_protein_fold_change_dataset(
            records, dataset_name="bad_ref", provenance=PROV, expected_count=2
        ),
        "reference_is_the_scales_neutral_value",
    )
    assert not result.passed
    assert "2 reference values are not the scale's neutral value" in result.message


def test_a_linear_panel_wants_one_as_its_neutral_reference() -> None:
    """The same verifier, the other scale: ``neutral_reference()`` supplies 1.0."""
    records = [
        _record(
            strain,
            offset,
            scale=FoldChangeScale.linear,
            measurement_type="dia_top3_ratio_to_control",
        )
        for strain, offset in zip(STRAINS, [0.0, 2.0])
    ]
    report = verify_protein_fold_change_dataset(
        records, dataset_name="linear", provenance=PROV, expected_count=2
    )
    assert report.passed
    assert _result(report, "reference_is_the_scales_neutral_value").passed
    reference = records[0]["reference"]["phenotype_reference"]
    assert reference["protein_fold_change"] == dict.fromkeys(PROTEINS, 1.0)


def test_a_reference_on_the_other_scale_is_refused() -> None:
    """A log2 experiment with a linear reference is key-matched but still wrong."""
    records = [
        _record(
            "PP_0812",
            0.0,
            reference_scale=FoldChangeScale.linear,
            reference_values=dict.fromkeys(PROTEINS, 1.0),
        ),
        _record("PP_0815", 2.0),
    ]
    result = _result(
        verify_protein_fold_change_dataset(
            records, dataset_name="cross_scale", provenance=PROV, expected_count=2
        ),
        "reference_is_the_scales_neutral_value",
    )
    assert not result.passed


def test_a_reference_missing_a_measured_key_is_refused() -> None:
    """Experiment over reference needs a denominator for every stored key."""
    records = [
        _record("PP_0812", 0.0, reference_values={"PP_4188": 0.0}),
        _record("PP_0815", 2.0),
    ]
    result = _result(
        verify_protein_fold_change_dataset(
            records, dataset_name="short_ref", provenance=PROV, expected_count=2
        ),
        "reference_is_the_scales_neutral_value",
    )
    assert not result.passed


def test_two_scales_or_two_statistics_in_one_dataset_are_refused() -> None:
    """A linear ratio and a log2 ratio of one contrast pool into nothing."""
    mixed_scale = [
        _record("PP_0812", 0.0),
        _record("PP_0815", 2.0, scale=FoldChangeScale.linear),
    ]
    scale_result = _result(
        verify_protein_fold_change_dataset(
            mixed_scale, dataset_name="mixed_scale", provenance=PROV, expected_count=2
        ),
        "fold_change_scale_consistent",
    )
    assert not scale_result.passed
    assert "2 distinct fold_change_scales mixed" in scale_result.message
    mixed_type = [
        _record("PP_0812", 0.0),
        _record("PP_0815", 2.0, measurement_type="dia_top3_ratio_to_control"),
    ]
    type_result = _result(
        verify_protein_fold_change_dataset(
            mixed_type, dataset_name="mixed_type", provenance=PROV, expected_count=2
        ),
        "measurement_type_consistent",
    )
    assert not type_result.passed
    assert "2 distinct measurement_types mixed" in type_result.message


def test_the_count_oracle_is_exact() -> None:
    report = verify_protein_fold_change_dataset(
        _records(), dataset_name="count", provenance=PROV, expected_count=3
    )
    assert not _result(report, "count").passed
    assert not report.passed


def test_the_locus_set_is_the_tested_proteins_plus_the_perturbed_genes() -> None:
    """L4's universe for this family, over the union of both identifier sources."""
    assert protein_fold_change_locus_set(_records()) == {
        "PP_4188",
        "PP_0368",
        "PP_0812",
        "PP_0815",
    }


def test_the_p_value_round_trip_inverts_the_loaders_own_conversion() -> None:
    """The released column is -log10(p); the only honest check is the inverse."""
    assert fold_change_p_value_round_trip(2.0, 0.01)
    assert fold_change_p_value_round_trip(19.312533, 10**-19.312533)
    # 1.301034 is Carruthers Figure 5b's minimum, the p<0.05 cutoff the sheet filters at
    assert fold_change_p_value_round_trip(1.301034, 10**-1.301034)
    assert not fold_change_p_value_round_trip(2.0, 0.02)
    # a p-value outside (0, 1] cannot be the inverse of any finite -log10
    assert not fold_change_p_value_round_trip(2.0, 0.0)
    assert not fold_change_p_value_round_trip(2.0, 1.5)
    assert not fold_change_p_value_round_trip(math.inf, 0.01)
    assert not fold_change_p_value_round_trip(math.nan, 0.01)
    # the tolerance is two-sided and the default is tight
    assert fold_change_p_value_round_trip(2.0, 0.01 * (1 + 1e-15))
    assert not fold_change_p_value_round_trip(2.0, 0.01 * (1 + 1e-6))
