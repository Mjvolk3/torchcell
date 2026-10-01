# tests/torchcell/verification/test_environment_response.py
# [[tests.torchcell.verification.test_environment_response]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/verification/test_environment_response.py
"""The environment-response verifier's own levels, eager and streaming, on a hand-built
release of schema records.

2026.09.30 (Phase 14). The release is three deletion strains (YAL001C, YBR085W,
YJR155W) under hydroquinone at IC30 on the library medium ``YP_GALACTOSE`` at 30 C,
log2 responses -1.2, 0.8 and -0.3 with SEs 0.1, 0.2 and 0.3, references at 0. The
shared rules of ``torchcell.verification.common`` are pinned in the sibling
``test_environment_response_verification.py`` and in ``test_common.py``; here each of
this module's eight own results (``structural``, ``count``, ``pair_uniqueness``,
``value_fidelity``, ``se_nonnegative``, ``measurement_type_consistent``,
``reference_zero``, ``environment_perturbed``) is pinned as an exact
``(level, name, passed, message)`` row, first on the passing release and then on one
table per failure mode, for both entry points; plus the report's ``summary()`` lines,
the strain and condition signatures on hand-written perturbation dicts, and the gene
set helper. The module has no ``main``.

2026.10.01 (issue #529): the two entry points are semantically identical, as the
streaming docstring promises, and ``test_eager_and_streaming_reports_are_equal`` holds
them to it field by field on a release that fails every own rule at once:

- ``pair_uniqueness`` counts REDUNDANT RECORDS (three copies of one record are 2) with
  one wording that names the study, so ``n_pairs + n_duplicated`` is the record count.
- ``value_fidelity`` and ``se_nonnegative`` index a bad value by its RECORD and carry a
  ``reason``.
- ``measurement_type_consistent`` prints the enum VALUE (``'log2_ratio'``), not the
  member's repr, although records hold ``model_dump()`` output.
"""

from __future__ import annotations

import copy
from typing import Any

from torchcell.datamodels.media import YP_GALACTOSE
from torchcell.datamodels.schema import (
    Compound,
    Concentration,
    DoseBasis,
    Environment,
    EnvironmentPerturbationType,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Genotype,
    KanMxDeletionPerturbation,
    MeasurementType,
    Media,
    ReferenceGenome,
    ResponseCategory,
    SmallMoleculePerturbation,
    Temperature,
)
from torchcell.verification.environment_response import (
    _condition_signature,
    _genotype_signature,
    environment_response_gene_set,
    verify_environment_response_dataset,
    verify_environment_response_dataset_streaming,
)
from torchcell.verification.report import Provenance, VerificationReport

PROV = Provenance(source_uri="test://synthetic", citation_key="synthetic2026")
GENES = ["YAL001C", "YBR085W", "YJR155W"]
HYDROQUINONE = "QIGBRXMKCJKVMJ-UHFFFAOYSA-N"
PUBLICATION = {"pubmed_id": "12345678", "doi": None}
OWN = {
    "structural",
    "count",
    "pair_uniqueness",
    "value_fidelity",
    "se_nonnegative",
    "measurement_type_consistent",
    "reference_zero",
    "environment_perturbed",
}
LOG2 = "'log2_ratio'"
EDIT_OK = (
    "all 3 experiments carry an environmental edit (perturbation, non-baseline "
    "temperature, or non-baseline media; baseline temp=30.0, media='YP + 2% galactose')"
)


def _environment(
    *, perturbed: bool = True, temperature: float = 30, media: Media = YP_GALACTOSE
) -> Environment:
    perturbations: list[EnvironmentPerturbationType] = (
        [
            SmallMoleculePerturbation(
                compound=Compound(name="hydroquinone", inchikey=HYDROQUINONE),
                concentration=Concentration(basis=DoseBasis.IC30),
            )
        ]
        if perturbed
        else []
    )
    return Environment(
        media=media,
        temperature=Temperature(value=temperature),
        perturbations=perturbations,
    )


def _numeric(value: float, se: float | None = None) -> EnvironmentResponsePhenotype:
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.log2_ratio,
        environment_response=value,
        environment_response_se=se,
        units="log2(treatment/control)",
    )


def _categorical(call: str | None) -> EnvironmentResponsePhenotype:
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.categorical,
        category=ResponseCategory(call) if call is not None else None,
        units="spot-assay call",
    )


def _record(
    gene: str,
    phenotype: EnvironmentResponsePhenotype,
    reference: EnvironmentResponsePhenotype,
    environment: Environment | None = None,
) -> dict[str, Any]:
    env = environment if environment is not None else _environment()
    experiment = EnvironmentResponseExperiment(
        dataset_name="synthetic",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=gene, perturbed_gene_name=gene
                )
            ]
        ),
        environment=env,
        phenotype=phenotype,
    )
    ref = EnvironmentResponseExperimentReference(
        dataset_name="synthetic",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        ),
        environment_reference=env.model_copy(),
        phenotype_reference=reference,
    )
    return {
        "experiment": experiment.model_dump(),
        "reference": ref.model_dump(),
        "publication": dict(PUBLICATION),
    }


def _release() -> list[dict[str, Any]]:
    values = [(-1.2, 0.1), (0.8, 0.2), (-0.3, 0.3)]
    return [
        _record(gene, _numeric(v, se), _numeric(0.0))
        for gene, (v, se) in zip(GENES, values, strict=True)
    ]


def _both(
    records: list[dict[str, Any]], expected_count: int = 3
) -> tuple[VerificationReport, VerificationReport]:
    eager = verify_environment_response_dataset(
        records,
        dataset_name="release",
        provenance=PROV,
        expected_count=expected_count,
        sgd_genes=set(GENES),
    )
    streaming = verify_environment_response_dataset_streaming(
        iter(records),
        dataset_name="release",
        provenance=PROV,
        expected_count=expected_count,
        sgd_genes=set(GENES),
    )
    return eager, streaming


def _rows(report: VerificationReport) -> list[tuple[str, str, bool, str]]:
    """This module's own results, in report order."""
    return [
        (r.level.name, r.name, r.passed, r.message)
        for r in report.results
        if r.name in OWN
    ]


def _row(report: VerificationReport, name: str) -> tuple[str, str, bool, str]:
    [row] = [row for row in _rows(report) if row[1] == name]
    return row


def _details(report: VerificationReport, name: str) -> dict[str, Any]:
    [result] = [r for r in report.results if r.name == name]
    return result.details


def test_passing_release_rows_and_result_order() -> None:
    """Every own row passes with its exact message; the 16 results come in the same
    order from both entry points (the eight own rows, then the eight shared ones),
    and the two reports are equal.
    """
    eager, streaming = _both(_release())
    expected = [
        ("L0", "structural", True, "3 records validated"),
        ("L1", "count", True, "observed 3, expected 3"),
        (
            "L1",
            "pair_uniqueness",
            True,
            "3 unique (study, strain, condition) records, one each",
        ),
        ("L2", "value_fidelity", True, "3 values checked"),
        ("L2", "se_nonnegative", True, "3 values checked"),
        ("L3", "measurement_type_consistent", True, f"single measurement_type: {LOG2}"),
        (
            "L3",
            "reference_zero",
            True,
            "numeric rule: reference response == 0 for all 3 records",
        ),
        ("L3", "environment_perturbed", True, EDIT_OK),
    ]
    assert _rows(eager) == expected
    assert eager == streaming
    names = [
        "structural",
        "count",
        "pair_uniqueness",
        "value_fidelity",
        "se_nonnegative",
        "measurement_type_consistent",
        "reference_zero",
        "environment_perturbed",
        "provenance_gaps",
        "canonical_gene_names",
        "uncertainty_sanity",
        "compound_identity",
        "media_compound_identity",
        "media_membership",
        "gene_containment_sgd",
        "current_genome_genes",
    ]
    assert [r.name for r in eager.results] == names
    assert [r.name for r in streaming.results] == names
    assert eager.passed and streaming.passed
    assert _details(eager, "reference_zero") == {
        "rule": "numeric_zero",
        "n_values": 3,
        "worst_abs": 0.0,
    }


def test_summary_is_sorted_by_level_with_a_mark_per_row() -> None:
    """``summary()`` heads with the dataset and verdict, then one line per result
    sorted by level (stable within a level). With the count oracle set to 4 the report
    fails and the count line carries ``XX``.
    """
    eager, _ = _both(_release(), expected_count=4)
    lines = eager.summary().split("\n")
    assert lines[0] == "release: FAIL"
    assert lines[1:4] == [
        "  [ok] L0 structural: 3 records validated",
        "  [XX] L1 count: observed 3, expected 4",
        "  [ok] L1 pair_uniqueness: 3 unique (study, strain, condition) records, "
        "one each",
    ]
    assert len(lines) == 17
    assert [line.split("] ")[1].split(" ")[0] for line in lines[1:]] == [
        "L0",
        "L1",
        "L1",
        "L1",
        "L1",
        "L2",
        "L2",
        "L2",
        "L3",
        "L3",
        "L3",
        "L3",
        "L3",
        "L3",
        "L4",
        "L4",
    ]


def test_triplicate_record_counts_two_redundant_records() -> None:
    """YAL001C's record three times plus the other two: 3 unique triples and 2
    redundant records (every copy after the first), so ``n_pairs + n_duplicated`` is
    the 5 records the count row observed. Both entry points report exactly this.
    """
    records = _release()
    records += [copy.deepcopy(records[0]), copy.deepcopy(records[0])]
    eager, streaming = _both(records, expected_count=5)
    for report in (eager, streaming):
        assert _row(report, "pair_uniqueness") == (
            "L1",
            "pair_uniqueness",
            False,
            "2 records duplicate an earlier (study, strain, condition) triple; "
            "3 unique triples",
        )
        assert _details(report, "pair_uniqueness") == {"n_pairs": 3, "n_duplicated": 2}


def test_same_strain_and_condition_in_another_study_is_not_a_duplicate() -> None:
    """The study (``pubmed_id``, else ``doi``) joins the key: YAL001C's record copied
    under a different PubMed id is a fourth unique triple, and the same copy under a
    DOI-only publication is a fifth.
    """
    records = _release()
    other_pmid = copy.deepcopy(records[0])
    other_pmid["publication"] = {"pubmed_id": "87654321", "doi": None}
    doi_only = copy.deepcopy(records[0])
    doi_only["publication"] = {"pubmed_id": None, "doi": "10.1/x"}
    eager, streaming = _both([*records, other_pmid, doi_only], expected_count=5)
    for report in (eager, streaming):
        assert _row(report, "pair_uniqueness")[2:] == (
            True,
            "5 unique (study, strain, condition) records, one each",
        )


def test_schema_failure_is_an_l0_row_with_the_record_index() -> None:
    """Record 1's ``environment_response`` is overwritten with NaN after the dump; the
    schema validator rejects it, so both entry points report ``1/3 records failed
    schema validation`` at index 1. The error is validated against the whole
    ``ExperimentType`` union, so its text runs far past the 500-character cap and is
    stored cut to exactly 500.
    """
    records = _release()
    records[1]["experiment"]["phenotype"]["environment_response"] = float("nan")
    eager, streaming = _both(records)
    for report in (eager, streaming):
        assert _row(report, "structural") == (
            "L0",
            "structural",
            False,
            "1/3 records failed schema validation",
        )
        details = _details(report, "structural")
        assert (details["n_records"], details["n_failures"]) == (3, 1)
        assert details["failures"][0]["index"] == 1
        assert len(details["failures"][0]["error"]) == 500


def test_non_finite_responses_are_indexed_by_record() -> None:
    """Record 0's response is removed (None), record 1's is NaN and record 2's is
    +inf. Both entry points check 2 values, find both bad, and name them by RECORD
    index (1 and 2, not their positions 0 and 1 among the present values) with a
    reason.
    """
    records = _release()
    records[0]["experiment"]["phenotype"]["environment_response"] = None
    records[1]["experiment"]["phenotype"]["environment_response"] = float("nan")
    records[2]["experiment"]["phenotype"]["environment_response"] = float("inf")
    eager, streaming = _both(records)
    for report in (eager, streaming):
        assert _row(report, "value_fidelity") == (
            "L2",
            "value_fidelity",
            False,
            "2/2 values invalid",
        )
        assert _details(report, "value_fidelity")["bad"] == [
            {"index": 1, "value": "nan", "reason": "nan"},
            {"index": 2, "value": "inf", "reason": "inf"},
        ]


def test_negative_se_fails_and_nan_se_is_not_counted() -> None:
    """SEs become None, NaN and -0.5: the None and the NaN are skipped, so one value is
    checked and it is bad (``< 0.0``). Both entry points name record 2 with the
    reason.
    """
    records = _release()
    records[0]["experiment"]["phenotype"]["environment_response_se"] = None
    records[1]["experiment"]["phenotype"]["environment_response_se"] = float("nan")
    records[2]["experiment"]["phenotype"]["environment_response_se"] = -0.5
    eager, streaming = _both(records)
    for report in (eager, streaming):
        assert _row(report, "se_nonnegative") == (
            "L2",
            "se_nonnegative",
            False,
            "1/1 values invalid",
        )
        assert _details(report, "se_nonnegative")["bad"] == [
            {"index": 2, "value": -0.5, "reason": "< 0.0"}
        ]


def test_mixed_measurement_types_fail() -> None:
    """Record 2 is relabeled ``z_score``: two types, listed sorted, as enum values."""
    records = _release()
    records[2]["experiment"]["phenotype"]["measurement_type"] = MeasurementType.z_score
    eager, streaming = _both(records)
    for report in (eager, streaming):
        assert _row(report, "measurement_type_consistent") == (
            "L3",
            "measurement_type_consistent",
            False,
            "2 distinct measurement_types mixed: ['log2_ratio', 'z_score']",
        )
        assert _details(report, "measurement_type_consistent") == {
            "measurement_types": ["log2_ratio", "z_score"]
        }


def test_nonzero_reference_fails_the_numeric_rule() -> None:
    """References 0, -0.25 and 0.125: the worst absolute value is 0.25, printed with
    ``:.3g``.
    """
    records = _release()
    records[1]["reference"]["phenotype_reference"]["environment_response"] = -0.25
    records[2]["reference"]["phenotype_reference"]["environment_response"] = 0.125
    eager, streaming = _both(records)
    for report in (eager, streaming):
        assert _row(report, "reference_zero") == (
            "L3",
            "reference_zero",
            False,
            "numeric rule: reference response not identically 0: max|v|=0.25",
        )
        assert _details(report, "reference_zero") == {
            "rule": "numeric_zero",
            "n_values": 3,
            "worst_abs": 0.25,
        }


def _categorical_release(references: list[str | None]) -> list[dict[str, Any]]:
    calls = ["sensitive", "sensitive", "resistant"]
    records = [
        _record(gene, _categorical(call), _categorical("no_change"))
        for gene, call in zip(GENES, calls, strict=True)
    ]
    for record, reference in zip(records, references, strict=True):
        record["reference"]["phenotype_reference"]["category"] = (
            None if reference is None else ResponseCategory(reference)
        )
    return records


def test_categorical_rule_passes_on_one_shared_baseline() -> None:
    """All three references carry ``no_change`` and no experiment reports it."""
    eager, streaming = _both(_categorical_release(["no_change"] * 3))
    for report in (eager, streaming):
        assert _row(report, "reference_zero") == (
            "L3",
            "reference_zero",
            True,
            "categorical rule: all 3 references carry the baseline category "
            "'no_change', which no experiment record reports",
        )


def test_categorical_rule_fails_on_a_missing_and_a_second_baseline() -> None:
    """References ``no_change``, None and ``sensitive``: one reference has no category
    and two distinct categories remain; ``sensitive`` is also measured twice, which is
    reported in the message and details but is not what fails the rule.
    """
    eager, streaming = _both(_categorical_release(["no_change", None, "sensitive"]))
    for report in (eager, streaming):
        assert _row(report, "reference_zero") == (
            "L3",
            "reference_zero",
            False,
            "categorical rule: 1 references carry no category; 2 distinct reference "
            "categories ['no_change', 'sensitive']; baseline also used as a measured "
            "call in {'sensitive': 2}",
        )
        assert _details(report, "reference_zero") == {
            "rule": "categorical_baseline",
            "n_values": 2,
            "reference_categories": {"no_change": 1, "sensitive": 1},
            "n_reference_missing_category": 1,
            "baseline_used_as_measured_call": {"sensitive": 2},
        }


def test_unperturbed_record_at_baseline_is_flagged_and_shifts_are_edits() -> None:
    """Three perturbed records plus three with no perturbation: one at the baseline
    (30 C, YP + 2% galactose), one at 37 C (a temperature edit) and one on a medium
    named ``SC`` (a medium edit). Baselines are the modes (30.0 on 5 of 6 records;
    the galactose medium on 5 of 6), so exactly one record has no edit.
    """
    records = _release()
    sc = Media(name="SC", state="liquid", is_synthetic=True)
    records += [
        _record("YCR001W", _numeric(0.1), _numeric(0.0), _environment(perturbed=False)),
        _record(
            "YDR001C",
            _numeric(0.2),
            _numeric(0.0),
            _environment(perturbed=False, temperature=37),
        ),
        _record(
            "YER001W",
            _numeric(0.3),
            _numeric(0.0),
            _environment(perturbed=False, media=sc),
        ),
    ]
    eager, streaming = _both(records, expected_count=6)
    for report in (eager, streaming):
        assert _row(report, "environment_perturbed") == (
            "L3",
            "environment_perturbed",
            False,
            "1 experiments have no environmental edit (no perturbation, baseline "
            "temperature 30.0, baseline media)",
        )
        assert _details(report, "environment_perturbed") == {
            "n_records": 6,
            "n_missing": 1,
            "baseline_temperature": 30.0,
            "baseline_media": "YP + 2% galactose",
        }


def _deletion(gene: str, name: str | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "systematic_gene_name": gene,
        "perturbation_type": "kanmx_deletion",
        "perturbed_gene_name": name or gene,
        **extra,
    }


def test_genotype_signature_keys_on_allele_guide_pool_and_donor() -> None:
    """The identity is ``(systematic, type, perturbed name)`` plus, when present, the
    guide spacer, then the library pool, then the donor; the background gene is left
    out and the perturbations are sorted. An allelic series (``act1-101``, ``act1-3``)
    gives two signatures, as do one spacer in two pools and one spacer with two donors.
    """
    background = frozenset({"YRR001C"})

    def signature(*perturbations: dict[str, Any]) -> tuple[Any, ...]:
        experiment = {"genotype": {"perturbations": list(perturbations)}}
        return _genotype_signature(experiment, background)

    assert signature(_deletion("YRR001C"), _deletion("YFL039C", "act1-101")) == (
        ("YFL039C", "kanmx_deletion", "act1-101"),
    )
    assert signature(_deletion("YFL039C", "act1-3")) == (
        ("YFL039C", "kanmx_deletion", "act1-3"),
    )
    guide = {"guide_sequence": "ACGT", "library_pool": None}
    assert signature(_deletion("YAL001C", crispr=guide)) == (
        ("YAL001C", "kanmx_deletion", "YAL001C", "ACGT"),
    )
    pooled = {"guide_sequence": "ACGT", "library_pool": "pool-2"}
    assert signature(_deletion("YAL001C", crispr=pooled)) == (
        ("YAL001C", "kanmx_deletion", "YAL001C", "ACGT", "pool-2"),
    )
    assert signature(
        _deletion("YAL001C", crispr=guide, donor_sequence="TTTT"), _deletion("YAA001W")
    ) == (
        ("YAA001W", "kanmx_deletion", "YAA001W"),
        ("YAL001C", "kanmx_deletion", "YAL001C", "ACGT", "TTTT"),
    )


def test_condition_signature_reads_compound_agent_dose_and_scalars() -> None:
    """A compound perturbation reads ``concentration``; a physical agent with no
    concentration reads ``magnitude``; ``None`` fields become ``""``, the perturbation
    tuples are sorted, and temperature, media name, both durations and ``screen_id``
    follow.
    """
    experiment = {
        "environment": {
            "perturbations": [
                {
                    "perturbation_type": "small_molecule",
                    "compound": {"name": "hydroquinone"},
                    "concentration": {"value": 1.5, "unit": "mM", "basis": None},
                },
                {
                    "perturbation_type": "physical",
                    "agent": {"name": "UV"},
                    "factor": "radiation",
                    "magnitude": {"value": 20, "unit": "J/m2"},
                },
            ],
            "temperature": {"value": 30.0},
            "media": {"name": "YPD"},
            "duration_hours": 48.0,
            "duration_generations": None,
        },
        "phenotype": {"screen_id": "s1"},
    }
    assert _condition_signature(experiment) == (
        (
            ("physical", "UV", "radiation", "20", "J/m2", ""),
            ("small_molecule", "hydroquinone", "", "1.5", "mM", ""),
        ),
        30.0,
        "YPD",
        48.0,
        None,
        "s1",
    )


def test_gene_set_leaves_out_the_background() -> None:
    """The union of screened deletions over the release, minus the background gene."""
    records = _release()
    assert environment_response_gene_set(records, frozenset({"YBR085W"})) == {
        "YAL001C",
        "YJR155W",
    }


def test_eager_and_streaming_reports_are_equal() -> None:
    """The streaming verifier's contract: semantically identical to the eager one.

    One release fails every own rule at once: YAL001C's record three times, record 1's
    response NaN, record 2's SE -0.5, record 3 relabeled ``z_score``, record 4's
    reference at 0.5, plus an unperturbed record at the baseline with no SE (so 5 SEs
    are checked). Both reports are compared result by result and field by field, then
    as whole models; the failing own rows are pinned to their exact values.
    """
    records = _release()
    records += [copy.deepcopy(records[0]), copy.deepcopy(records[0])]
    records[1]["experiment"]["phenotype"]["environment_response"] = float("nan")
    records[2]["experiment"]["phenotype"]["environment_response_se"] = -0.5
    records[3]["experiment"]["phenotype"]["measurement_type"] = MeasurementType.z_score
    records[4]["reference"]["phenotype_reference"]["environment_response"] = 0.5
    records.append(
        _record("YCR001W", _numeric(0.1), _numeric(0.0), _environment(perturbed=False))
    )
    eager, streaming = _both(records, expected_count=6)
    assert len(eager.results) == len(streaming.results) == 16
    for left, right in zip(eager.results, streaming.results, strict=True):
        assert (left.level, left.name, left.passed, left.message) == (
            right.level,
            right.name,
            right.passed,
            right.message,
        )
        assert left.details == right.details
    assert eager.model_dump() == streaming.model_dump()
    assert [row for row in _rows(eager) if not row[2]] == [
        ("L0", "structural", False, "1/6 records failed schema validation"),
        (
            "L1",
            "pair_uniqueness",
            False,
            "2 records duplicate an earlier (study, strain, condition) triple; "
            "4 unique triples",
        ),
        ("L2", "value_fidelity", False, "1/6 values invalid"),
        ("L2", "se_nonnegative", False, "1/5 values invalid"),
        (
            "L3",
            "measurement_type_consistent",
            False,
            "2 distinct measurement_types mixed: ['log2_ratio', 'z_score']",
        ),
        (
            "L3",
            "reference_zero",
            False,
            "numeric rule: reference response not identically 0: max|v|=0.5",
        ),
        (
            "L3",
            "environment_perturbed",
            False,
            "1 experiments have no environmental edit (no perturbation, baseline "
            "temperature 30.0, baseline media)",
        ),
    ]
    assert _details(eager, "value_fidelity")["bad"] == [
        {"index": 1, "value": "nan", "reason": "nan"}
    ]
    assert _details(eager, "se_nonnegative")["bad"] == [
        {"index": 2, "value": -0.5, "reason": "< 0.0"}
    ]
