# tests/torchcell/verification/test_rnaseq.py
# [[tests.torchcell.verification.test_rnaseq]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/verification/test_rnaseq.py
"""The RNA-seq / pseudobulk expression verifier on hand-built, schema-valid records.

Two record families share the verifier. The absolute-TPM family (Caudal) is two natural
isolates, AAA and SACE_YAU, each carrying one ``SequenceVariantPerturbation`` with its
``strain_id`` and measuring YAL001C + YBR001W (TPM 2.0/4.0 and 3.0/5.0, counts 15/20 and
30/50) against a shared population-mean reference (TPM 2.5/4.0). The pseudobulk family
(Nadal-Ribelles) is two SGA KanMX deletions with a ``strain_id`` and per-gene log2 ratios
over the same two genes, reference 0.0.

Derived expectations: a TPM dataset emits seven results in order (structural, count,
strain_uniqueness, tpm_value_fidelity, count_value_fidelity, measurement_type_consistent,
reference_finite); a pseudobulk dataset omits count_value_fidelity and names its L2
``log2_ratio_value_fidelity``. Values are flattened record by record in the dumped
dict's insertion order (the schema's SortedDict makes that sorted-gene order here), so
2 records x 2 genes = 4 values, and the first gene of the second record is index 2. The reference is counted per RECORD, so two records sharing one two-gene
reference report 4 reference values.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from torchcell.datamodels.schema import (
    Environment,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    MrnaNumberFractionExperiment,
    MrnaNumberFractionPhenotype,
    PseudobulkExpressionExperiment,
    PseudobulkExpressionExperimentReference,
    PseudobulkExpressionPhenotype,
    ReferenceGenome,
    RNASeqExpressionExperiment,
    RNASeqExpressionExperimentReference,
    RNASeqExpressionPhenotype,
    SequenceVariantPerturbation,
    SgaKanMxDeletionPerturbation,
    Temperature,
)
from torchcell.verification.report import Level, LevelResult, Provenance
from torchcell.verification.rnaseq import rnaseq_gene_set, verify_rnaseq_dataset

PROV = Provenance(source_uri="test://synthetic", citation_key="caudalTest2024")
GENES = ["YAL001C", "YBR001W"]
TPM_NAMES = [
    "structural",
    "count",
    "strain_uniqueness",
    "tpm_value_fidelity",
    "count_value_fidelity",
    "measurement_type_consistent",
    "reference_finite",
]
LOG2_NAMES = [
    "structural",
    "count",
    "strain_uniqueness",
    "log2_ratio_value_fidelity",
    "measurement_type_consistent",
    "reference_finite",
]


def _environment(temperature: float = 30.0) -> Environment:
    return Environment(
        media=Media(name="SC", state="liquid", is_synthetic=True),
        temperature=Temperature(value=temperature),
    )


def _tpm_record(
    strain: str | None,
    tpm: dict[str, float],
    count: dict[str, int],
    *,
    temperature: float = 30.0,
    ref_tpm: dict[str, float] | None = None,
) -> dict[str, Any]:
    """One Caudal-style isolate record; ``strain=None`` uses a perturbation with no strain id."""
    perturbation: Any = (
        SequenceVariantPerturbation(
            systematic_gene_name="YAL001C", perturbed_gene_name="TFC3", strain_id=strain
        )
        if strain is not None
        else KanMxDeletionPerturbation(
            systematic_gene_name="YAL001C", perturbed_gene_name="TFC3"
        )
    )
    env = _environment(temperature)
    reference_tpm = ref_tpm if ref_tpm is not None else {"YAL001C": 2.5, "YBR001W": 4.0}
    experiment = RNASeqExpressionExperiment(
        dataset_name="test",
        genotype=Genotype(perturbations=[perturbation]),
        environment=env,
        phenotype=RNASeqExpressionPhenotype(expression_tpm=tpm, expression_count=count),
    )
    reference = RNASeqExpressionExperimentReference(
        dataset_name="test",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="S288C"
        ),
        environment_reference=env.model_copy(),
        phenotype_reference=RNASeqExpressionPhenotype(
            expression_tpm=reference_tpm,
            expression_count={g: 10 for g in reference_tpm},
        ),
    )
    return {"experiment": experiment.model_dump(), "reference": reference.model_dump()}


def _log2_record(
    gene: str, strain: str, log2: dict[str, float], *, temperature: float = 30.0
) -> dict[str, Any]:
    """One Nadal-Ribelles-style pseudobulk deletion record."""
    env = _environment(temperature)
    experiment = PseudobulkExpressionExperiment(
        dataset_name="test",
        genotype=Genotype(
            perturbations=[
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=gene,
                    perturbed_gene_name=gene,
                    strain_id=strain,
                )
            ]
        ),
        environment=env,
        phenotype=PseudobulkExpressionPhenotype(
            expression_log2_ratio=log2, dispersion=1.2, n_cells=100
        ),
    )
    reference = PseudobulkExpressionExperimentReference(
        dataset_name="test",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        ),
        environment_reference=env.model_copy(),
        phenotype_reference=PseudobulkExpressionPhenotype(
            expression_log2_ratio={g: 0.0 for g in log2}, dispersion=1.0, n_cells=500
        ),
    )
    return {"experiment": experiment.model_dump(), "reference": reference.model_dump()}


def _tpm_records() -> list[dict[str, Any]]:
    return [
        _tpm_record(
            "AAA", {"YAL001C": 2.0, "YBR001W": 4.0}, {"YAL001C": 15, "YBR001W": 20}
        ),
        _tpm_record(
            "SACE_YAU", {"YAL001C": 3.0, "YBR001W": 5.0}, {"YAL001C": 30, "YBR001W": 50}
        ),
    ]


def _log2_records() -> list[dict[str, Any]]:
    return [
        _log2_record("YAL001C", "bc-YAL001C", {"YAL001C": -3.0, "YBR001W": 0.4}),
        _log2_record("YBR001W", "bc-YBR001W", {"YAL001C": 0.1, "YBR001W": -2.0}),
    ]


def _verify(records: list[dict[str, Any]], expected: int | None = None) -> Any:
    return verify_rnaseq_dataset(
        records,
        dataset_name="rna",
        provenance=PROV,
        expected_count=len(records) if expected is None else expected,
    )


def _result(report: Any, name: str) -> LevelResult:
    matches = [r for r in report.results if r.name == name]
    assert len(matches) == 1, [r.name for r in report.results]
    result: LevelResult = matches[0]
    return result


def test_tpm_dataset_emits_seven_results_in_order_and_passes() -> None:
    report = _verify(_tpm_records())
    assert [r.name for r in report.results] == TPM_NAMES
    assert [r.level for r in report.results] == [
        Level.L0,
        Level.L1,
        Level.L1,
        Level.L2,
        Level.L2,
        Level.L3,
        Level.L3,
    ]
    assert report.passed is True
    assert _result(report, "structural").message == "2 records validated"
    uniqueness = _result(report, "strain_uniqueness")
    assert uniqueness.message == "2 unique (strain, condition) records over 2 strains"
    assert uniqueness.details == {
        "n_records": 2,
        "n_strains": 2,
        "n_duplicated": 0,
        "n_missing": 0,
    }
    assert _result(report, "tpm_value_fidelity").details == {
        "n_values": 4,
        "n_bad": 0,
        "bad": [],
    }
    assert _result(report, "count_value_fidelity").message == (
        "4 counts are non-negative integers"
    )
    assert _result(report, "measurement_type_consistent").message == (
        "single measurement_type: 'rnaseq_tpm'"
    )
    assert _result(report, "reference_finite").message == (
        "reference expression finite for all 4 values"
    )


def test_pseudobulk_dataset_skips_the_count_check_and_allows_negatives() -> None:
    report = _verify(_log2_records())
    assert [r.name for r in report.results] == LOG2_NAMES
    assert report.passed is True
    fidelity = _result(report, "log2_ratio_value_fidelity")
    assert fidelity.level is Level.L2
    assert fidelity.details == {"n_values": 4, "n_bad": 0, "bad": []}
    assert _result(report, "measurement_type_consistent").details == {
        "measurement_types": ["pseudobulk_scrnaseq_log2fc"]
    }
    assert _result(report, "strain_uniqueness").details["n_strains"] == 2


def test_strain_uniqueness_keys_on_strain_and_condition() -> None:
    two_conditions = _verify(
        [
            _tpm_record("AAA", {"YAL001C": 2.0}, {"YAL001C": 15}),
            _tpm_record("AAA", {"YAL001C": 1.0}, {"YAL001C": 8}, temperature=37.0),
        ]
    )
    result = _result(two_conditions, "strain_uniqueness")
    assert result.passed is True
    assert result.details == {
        "n_records": 2,
        "n_strains": 1,
        "n_duplicated": 0,
        "n_missing": 0,
    }

    duplicated = _verify(
        [
            _tpm_record("AAA", {"YAL001C": 2.0}, {"YAL001C": 15}),
            _tpm_record("AAA", {"YAL001C": 1.0}, {"YAL001C": 8}),
        ]
    )
    result = _result(duplicated, "strain_uniqueness")
    assert result.passed is False
    assert (
        result.message == "1 (strain, condition) duplicated, 0 records without a strain"
    )
    assert result.details == {
        "n_records": 1,
        "n_strains": 1,
        "n_duplicated": 1,
        "n_missing": 0,
    }

    no_strain = _verify(
        [
            _tpm_record("AAA", {"YAL001C": 2.0}, {"YAL001C": 15}),
            _tpm_record(None, {"YAL001C": 1.0}, {"YAL001C": 8}),
        ]
    )
    result = _result(no_strain, "strain_uniqueness")
    assert result.passed is False
    assert (
        result.message == "0 (strain, condition) duplicated, 1 records without a strain"
    )
    assert result.details == {
        "n_records": 1,
        "n_strains": 1,
        "n_duplicated": 0,
        "n_missing": 1,
    }
    assert result.details["n_records"] == 1


def test_tpm_value_fidelity_indexes_the_flattened_value_list() -> None:
    records = _tpm_records()
    records[1]["experiment"]["phenotype"]["expression_tpm"]["YAL001C"] = -1.0
    records[1]["experiment"]["phenotype"]["expression_tpm"]["YBR001W"] = math.inf
    fidelity = _result(_verify(records), "tpm_value_fidelity")
    assert fidelity.passed is False
    assert fidelity.message == "2/4 values invalid"
    assert fidelity.details["bad"] == [
        {"index": 2, "value": -1.0, "reason": "< 0.0"},
        {"index": 3, "value": "inf", "reason": "inf"},
    ]


def test_count_value_fidelity_rejects_bool_negative_and_float_counts() -> None:
    records = _tpm_records()
    records[0]["experiment"]["phenotype"]["expression_count"]["YAL001C"] = True
    records[0]["experiment"]["phenotype"]["expression_count"]["YBR001W"] = -1
    records[1]["experiment"]["phenotype"]["expression_count"]["YAL001C"] = 2.5
    counts = _result(_verify(records), "count_value_fidelity")
    assert counts.passed is False
    assert counts.message == "3/4 counts are not non-negative integers"
    assert counts.details == {"n_values": 4, "n_bad": 3}


def test_reference_finite_counts_every_record_reference() -> None:
    records = _tpm_records()
    records[0]["reference"]["phenotype_reference"]["expression_tpm"]["YBR001W"] = (
        math.inf
    )
    reference = _result(_verify(records), "reference_finite")
    assert reference.passed is False
    assert reference.message == "1/4 reference expression values non-finite"
    assert reference.details == {"n_values": 4, "n_bad": 1}

    log2 = _log2_records()
    log2[1]["reference"]["phenotype_reference"]["expression_log2_ratio"]["YAL001C"] = (
        math.nan
    )
    reference = _result(_verify(log2), "reference_finite")
    assert reference.details == {"n_values": 4, "n_bad": 1}


def test_mixed_families_report_the_mix_only_when_the_first_record_is_pseudobulk() -> (
    None
):
    """Finding: the family is decided by ``records[0]`` alone. Pseudobulk-first mixing
    reaches L3 and reports two measurement types; TPM-first mixing raises ``KeyError`` on
    ``expression_count`` inside the count check before the L3 mix check runs, so the
    "no silent cross-assay mixing" guard only fires for one ordering.
    """
    pseudobulk_first = [_log2_records()[0], _tpm_records()[0]]
    report = _verify(pseudobulk_first)
    assert [r.name for r in report.results] == LOG2_NAMES
    mix = _result(report, "measurement_type_consistent")
    assert mix.passed is False
    assert mix.message == (
        "2 distinct measurement_types mixed: ['pseudobulk_scrnaseq_log2fc', 'rnaseq_tpm']"
    )
    assert mix.details == {
        "measurement_types": ["pseudobulk_scrnaseq_log2fc", "rnaseq_tpm"]
    }

    tpm_first = [_tpm_records()[0], _log2_records()[0]]
    with pytest.raises(KeyError, match="expression_count"):
        _verify(tpm_first)


def test_wrong_count_and_an_invalid_record_fail_their_own_levels() -> None:
    records = _tpm_records()
    records[0]["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"] = (
        "not-a-gene"
    )
    report = _verify(records, expected=943)
    structural = _result(report, "structural")
    assert structural.passed is False
    assert structural.message == "1/2 records failed schema validation"
    assert structural.details["failures"][0]["index"] == 0
    count = _result(report, "count")
    assert count.passed is False
    assert count.details == {"observed": 2, "expected": 943}


def test_empty_dataset_defaults_to_the_tpm_family_and_passes_a_zero_oracle() -> None:
    report = _verify([], expected=0)
    assert [r.name for r in report.results] == TPM_NAMES
    assert report.passed is True
    assert _result(report, "structural").message == "0 records validated"
    assert _result(report, "count_value_fidelity").message == (
        "0 counts are non-negative integers"
    )
    assert _result(report, "measurement_type_consistent").message == (
        "single measurement_type: None"
    )
    assert _result(report, "reference_finite").message == (
        "reference expression finite for all 0 values"
    )


def test_rnaseq_gene_set_unions_measured_genes_of_both_families() -> None:
    tpm = _tpm_records()
    tpm.append(_tpm_record("BBB", {"YCR001W": 1.0}, {"YCR001W": 3}))
    assert rnaseq_gene_set(tpm) == {"YAL001C", "YBR001W", "YCR001W"}
    assert rnaseq_gene_set(_log2_records()) == set(GENES)
    assert rnaseq_gene_set([]) == set()


# --------------------------------------------------------------------------- #
# replicate_aware=True: a compendium whose rows are sequenced libraries
# --------------------------------------------------------------------------- #
REPLICATE_NAMES = [
    "structural",
    "count",
    "replicate_groups",
    "tpm_value_fidelity",
    "count_value_fidelity",
    "measurement_type_consistent",
    "reference_finite",
]


def _verify_replicate_aware(
    records: list[dict[str, Any]], expected: int | None = None
) -> Any:
    return verify_rnaseq_dataset(
        records,
        dataset_name="rna",
        provenance=PROV,
        expected_count=len(records) if expected is None else expected,
        replicate_aware=True,
    )


def _library(tpm: dict[str, float], count: dict[str, int]) -> dict[str, Any]:
    """One library of a wild-type condition: no perturbation, so no strain id anywhere."""
    env = _environment()
    experiment = RNASeqExpressionExperiment(
        dataset_name="test",
        genotype=Genotype(perturbations=[]),
        environment=env,
        phenotype=RNASeqExpressionPhenotype(expression_tpm=tpm, expression_count=count),
    )
    reference = RNASeqExpressionExperimentReference(
        dataset_name="test",
        genome_reference=ReferenceGenome(
            species="Escherichia coli", strain="MG1655", ploidy="haploid"
        ),
        environment_reference=env.model_copy(),
        phenotype_reference=RNASeqExpressionPhenotype(
            expression_tpm={g: 1.0 for g in tpm}, expression_count={g: 10 for g in tpm}
        ),
    )
    return {"experiment": experiment.model_dump(), "reference": reference.model_dump()}


def test_replicate_aware_swaps_the_l1_rule_for_the_group_rule() -> None:
    """Two libraries of ONE condition pass, where ``strain_uniqueness`` cannot.

    The records carry no perturbation at all, so the strain rule reports two records
    without a strain and fails; that is exactly the PRECISE-1K / putidaPRECISE321 shape.
    """
    records = [
        _library({"b0001": 2.0, "b0002": 4.0}, {"b0001": 15, "b0002": 20}),
        _library({"b0001": 2.1, "b0002": 3.9}, {"b0001": 16, "b0002": 19}),
    ]
    report = _verify_replicate_aware(records)
    assert [r.name for r in report.results] == REPLICATE_NAMES
    assert report.passed is True
    groups = _result(report, "replicate_groups")
    assert groups.level is Level.L1
    assert groups.message == (
        "2 records with distinct profiles over 1 (genotype, environment) groups, "
        "each measuring one gene set"
    )
    assert groups.details == {
        "n_records": 2,
        "n_groups": 1,
        "n_in_repeated_profiles": 0,
        "n_groups_with_mixed_gene_sets": 0,
        "group_size_histogram": {"2": 1},
    }

    # the same records under the default rule fail, which is why the swap exists
    strain_rule = _result(
        verify_rnaseq_dataset(
            records, dataset_name="rna", provenance=PROV, expected_count=2
        ),
        "strain_uniqueness",
    )
    assert strain_rule.passed is False
    assert strain_rule.details["n_missing"] == 2


def test_replicate_groups_fails_a_library_counted_twice() -> None:
    """Two records with an identical profile are one library read twice."""
    profile = ({"b0001": 2.0, "b0002": 4.0}, {"b0001": 15, "b0002": 20})
    report = _verify_replicate_aware([_library(*profile), _library(*profile)])
    groups = _result(report, "replicate_groups")
    assert groups.passed is False
    assert groups.message == (
        "2 records share an expression profile; 0 groups pool records measuring "
        "different genes"
    )
    assert groups.details["n_in_repeated_profiles"] == 2


def test_replicate_groups_fails_a_group_whose_members_measure_different_genes() -> None:
    """A group pooling two gene sets is two conditions on one environment identity."""
    report = _verify_replicate_aware(
        [
            _library({"b0001": 2.0, "b0002": 4.0}, {"b0001": 15, "b0002": 20}),
            _library({"b0001": 2.1, "b0003": 3.9}, {"b0001": 16, "b0003": 19}),
        ]
    )
    groups = _result(report, "replicate_groups")
    assert groups.passed is False
    assert groups.details == {
        "n_records": 2,
        "n_groups": 1,
        "n_in_repeated_profiles": 0,
        "n_groups_with_mixed_gene_sets": 1,
        "group_size_histogram": {"2": 1},
    }


def test_replicate_groups_separates_conditions_and_reports_the_sizes() -> None:
    """Group identity is (genotype, environment), and the sizes are the structure."""
    records = [
        _library({"b0001": 2.0}, {"b0001": 15}),
        _library({"b0001": 2.1}, {"b0001": 16}),
        _library({"b0001": 2.2}, {"b0001": 17}),
    ]
    # a third condition at another temperature: a different environment, its own group
    other = _library({"b0001": 9.0}, {"b0001": 90})
    other["experiment"]["environment"]["temperature"]["value"] = 42.0
    report = _verify_replicate_aware([*records, other])
    groups = _result(report, "replicate_groups")
    assert groups.passed is True
    assert groups.details["n_groups"] == 2
    assert groups.details["group_size_histogram"] == {"1": 1, "3": 1}


# --------------------------------------------------------------------------- #
# Issue #854: the count-less number-fraction family
# --------------------------------------------------------------------------- #
FRACTION_NAMES = [
    "structural",
    "count",
    "replicate_groups",
    "number_fraction_value_fidelity",
    "fraction_sum_at_most_one",
    "measurement_type_consistent",
    "reference_finite",
]


def _fraction_record(
    fractions: dict[str, float], reference: dict[str, float] | None = None
) -> dict[str, Any]:
    """One library; the reference dict is dumped unvalidated so a bad sum can reach L3."""
    experiment = MrnaNumberFractionExperiment(
        dataset_name="test",
        genotype=Genotype(perturbations=[]),
        environment=_environment(),
        phenotype=MrnaNumberFractionPhenotype(
            mrna_number_fraction=fractions,
            n_libraries=1,
            measurement_type="rnaseq_mrna_number_fraction",
        ),
    )
    phenotype_reference = experiment.phenotype.model_dump()
    if reference is not None:
        phenotype_reference["mrna_number_fraction"] = reference
    return {
        "experiment": experiment.model_dump(),
        "reference": {"phenotype_reference": phenotype_reference},
    }


def test_number_fraction_dataset_skips_counts_and_checks_the_sum() -> None:
    records = [
        _fraction_record({"b0001": 0.5, "b0002": 0.25}),
        _fraction_record({"b0001": 0.4, "b0002": 0.0}),
    ]
    report = _verify_replicate_aware(records)
    assert [r.name for r in report.results] == FRACTION_NAMES
    assert report.passed
    assert _result(report, "fraction_sum_at_most_one").message == (
        "4 stored profiles sum to 0.400000 .. 0.750000"
    )
    assert rnaseq_gene_set(records) == {"b0001", "b0002"}


def test_fraction_sum_fails_a_reference_above_one() -> None:
    records = [_fraction_record({"b0001": 0.5}, reference={"b0001": 0.7, "b0002": 0.4})]
    result = _result(_verify_replicate_aware(records), "fraction_sum_at_most_one")
    assert (result.passed, result.details) == (False, {"n_profiles": 2, "n_over": 1})


def test_number_fraction_fidelity_refuses_a_value_above_one() -> None:
    record = _fraction_record({"b0001": 0.5})
    record["experiment"]["phenotype"]["mrna_number_fraction"]["b0001"] = 1.5
    result = _result(
        _verify_replicate_aware([record]), "number_fraction_value_fidelity"
    )
    assert result.passed is False
