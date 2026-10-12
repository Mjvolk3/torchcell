# tests/torchcell/verification/test_synthetic_lethality.py
# [[tests.torchcell.verification.test_synthetic_lethality]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/verification/test_synthetic_lethality.py
"""The SynLethDB pair verifier on hand-built, schema-valid records.

Fixture: two synthetic-lethal pairs, (YAL001C, YBR085W) score 0.9 PMID 11 and
(YBR085W, YJR155W) score 0.5 PMID 12, on ``SC`` at 30 C against a negative reference.
The release holds four rows: the two kept ones (written with the sides in the other
order for the first, which the unordered key absorbs) plus two ledgered drops. A correct
store emits, in order: structural, reference_structural, count, two_distinct_genes,
unordered_pair_uniqueness, positive_label, score_in_unit_interval (or
score_defined_by_source), released_rows, release_accounting, reference_negative, then
the caller's extra rows and the six shared rules.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from torchcell.datamodels.media import SC
from torchcell.datamodels.schema import (
    Environment,
    Genotype,
    ReferenceGenome,
    SgaKanMxDeletionPerturbation,
    SyntheticLethalityExperiment,
    SyntheticLethalityExperimentReference,
    SyntheticLethalityPhenotype,
    SyntheticRescueExperiment,
    SyntheticRescueExperimentReference,
    SyntheticRescuePhenotype,
    Temperature,
)
from torchcell.verification.report import LevelResult, Provenance
from torchcell.verification.synthetic_lethality import (
    SYNTHETIC_LETHALITY,
    SYNTHETIC_RESCUE,
    ReleasedPairs,
    pair_row_key,
    verify_synthetic_pair_dataset,
)

PROV = Provenance(source_uri="test://synthetic", citation_key="wangSynLethDB2022")
FAMILY_NAMES = [
    "structural",
    "reference_structural",
    "count",
    "two_distinct_genes",
    "unordered_pair_uniqueness",
    "positive_label",
    "score_in_unit_interval",
    "released_rows",
    "release_accounting",
    "reference_negative",
]
SHARED_NAMES = [
    "provenance_gaps",
    "canonical_gene_names",
    "uncertainty_sanity",
    "compound_identity",
    "media_compound_identity",
    "media_membership",
]


def _record(
    genes: tuple[str, ...],
    score: float | None,
    pubmed_id: str,
    *,
    rescue: bool = False,
    label: bool = True,
    reference_score: float | None = None,
    graph_level: str = "edge",
) -> dict[str, Any]:
    environment = Environment(media=SC, temperature=Temperature(value=30.0))
    genotype = Genotype(
        perturbations=[
            SgaKanMxDeletionPerturbation(
                systematic_gene_name=gene, perturbed_gene_name=gene, strain_id="S288C"
            )
            for gene in genes
        ]
    )
    genome = ReferenceGenome(species="Saccharomyces cerevisiae", strain="S288C")
    experiment: SyntheticLethalityExperiment | SyntheticRescueExperiment
    reference: (
        SyntheticLethalityExperimentReference | SyntheticRescueExperimentReference
    )
    if rescue:
        experiment = SyntheticRescueExperiment(
            dataset_name="test",
            genotype=genotype,
            environment=environment,
            phenotype=SyntheticRescuePhenotype(
                graph_level=graph_level,
                is_synthetic_rescue=label,
                synthetic_rescue_statistic_score=score,
            ),
        )
        reference = SyntheticRescueExperimentReference(
            dataset_name="test",
            genome_reference=genome,
            environment_reference=environment,
            phenotype_reference=SyntheticRescuePhenotype(
                is_synthetic_rescue=False,
                synthetic_rescue_statistic_score=reference_score,
            ),
        )
    else:
        experiment = SyntheticLethalityExperiment(
            dataset_name="test",
            genotype=genotype,
            environment=environment,
            phenotype=SyntheticLethalityPhenotype(
                graph_level=graph_level,
                is_synthetic_lethal=label,
                synthetic_lethality_statistic_score=score,
            ),
        )
        reference = SyntheticLethalityExperimentReference(
            dataset_name="test",
            genome_reference=genome,
            environment_reference=environment,
            phenotype_reference=SyntheticLethalityPhenotype(
                is_synthetic_lethal=False,
                synthetic_lethality_statistic_score=reference_score,
            ),
        )
    return {
        "experiment": experiment.model_dump(),
        "reference": reference.model_dump(),
        "publication": {"pubmed_id": pubmed_id},
    }


def _records(
    *, rescue: bool = False, scores: tuple[Any, Any] = (0.9, 0.5)
) -> list[Any]:
    return [
        _record(("YAL001C", "YBR085W"), scores[0], "11", rescue=rescue),
        _record(("YBR085W", "YJR155W"), scores[1], "12", rescue=rescue),
    ]


def _released(
    rows: Counter[Any] | None = None,
    *,
    n_rows: int | None = 4,
    stated: int | None = 4,
    drift: dict[str, str] | None = None,
) -> ReleasedPairs:
    kept = Counter(
        [
            pair_row_key("YBR085W", "YAL001C", 0.9, "11"),
            pair_row_key("YBR085W", "YJR155W", 0.5, "12"),
        ]
    )
    return ReleasedPairs(
        files=["Yeast_SL.csv"],
        drift=drift or {},
        rows=None if drift else (kept if rows is None else rows),
        n_released_rows=None if drift else n_rows,
        n_dropped=2,
        stated_count=stated,
        stated_count_quote=None if stated is None else "4 of yeast",
    )


def _verify(records: list[Any], **kwargs: Any) -> Any:
    arguments: dict[str, Any] = {
        "kind": SYNTHETIC_LETHALITY,
        "dataset_name": "sl",
        "provenance": PROV,
        "expected_count": len(records),
        "released": _released(),
        "score_definition": "normalized confidence",
    }
    arguments.update(kwargs)
    return verify_synthetic_pair_dataset(records, **arguments)


def _result(report: Any, name: str) -> LevelResult:
    matches = [r for r in report.results if r.name == name]
    assert len(matches) == 1, [r.name for r in report.results]
    result: LevelResult = matches[0]
    return result


def test_a_correct_lethality_store_emits_every_row_in_order_and_passes() -> None:
    report = _verify(_records())
    assert [r.name for r in report.results] == FAMILY_NAMES + SHARED_NAMES
    assert report.passed is True
    assert _result(report, "release_accounting").message == (
        "4 released rows = 2 stored + 2 ledgered drops; the release paper states 4"
    )
    assert _result(report, "released_rows").details["n_released_kept"] == 2


def test_pair_row_key_orders_the_two_sides() -> None:
    assert pair_row_key("B", "A", None, "1") == (("A", "B"), None, "1")
    assert pair_row_key("A", "B", 0.5, "1") == (("A", "B"), 0.5, "1")


def test_a_rescue_store_with_undefined_scores_fails_score_defined_by_source() -> None:
    rescue = _records(rescue=True, scores=(-0.22, None))
    released = _released(
        Counter(
            [
                pair_row_key("YAL001C", "YBR085W", -0.22, "11"),
                pair_row_key("YBR085W", "YJR155W", None, "12"),
            ]
        ),
        stated=None,
    )
    report = _verify(
        rescue, kind=SYNTHETIC_RESCUE, released=released, score_definition=None
    )
    score = _result(report, "score_defined_by_source")
    assert score.passed is False
    assert score.message == (
        "1 of 2 records state a score the source never defines (1 of them outside "
        "[0, 1])"
    )
    assert _result(report, "release_accounting").message == (
        "4 released rows = 2 stored + 2 ledgered drops; the release paper states no "
        "count for this file"
    )
    assert _result(report, "structural").message == (
        "2 records validated as SyntheticRescueExperiment"
    )
    clean = _verify(
        _records(rescue=True, scores=(None, None)),
        kind=SYNTHETIC_RESCUE,
        released=_released(
            Counter(
                [
                    pair_row_key("YAL001C", "YBR085W", None, "11"),
                    pair_row_key("YBR085W", "YJR155W", None, "12"),
                ]
            ),
            stated=None,
        ),
        score_definition=None,
    )
    assert _result(clean, "score_defined_by_source").passed is True


def test_out_of_range_scores_a_false_label_and_a_scored_reference_fail() -> None:
    records = [
        _record(("YAL001C", "YBR085W"), 1.4, "11"),
        _record(("YBR085W", "YJR155W"), 0.5, "12", label=False, reference_score=0.1),
    ]
    report = _verify(records)
    assert _result(report, "score_in_unit_interval").details["bad"] == [
        {"index": 0, "value": "1.4"}
    ]
    assert _result(report, "positive_label").details == {
        "n_bad": 1,
        "example_indices": [1],
    }
    assert _result(report, "reference_negative").details == {
        "problems": {"reference carries a score": 1}
    }
    assert _result(report, "released_rows").details["only_stored"] == [
        "(('YAL001C', 'YBR085W'), 1.4, '11')"
    ]


def test_a_repeated_pair_and_a_non_pair_fail_their_l1_rows() -> None:
    records = [
        *_records(),
        _record(("YBR085W", "YAL001C"), 0.9, "13"),
        _record(("YAL001C", "YAL001C"), 0.9, "14"),
        _record(("YAL001C", "YBR085W", "YJR155W"), 0.9, "15"),
        _record(("YAL001C", "YJR155W"), 0.9, "16", graph_level="hyperedge"),
    ]
    report = _verify(records, released=_released(n_rows=8, stated=None))
    assert _result(report, "unordered_pair_uniqueness").details == {
        "n_pairs": 2,
        "n_repeated": 1,
        "examples": ["YAL001C+YBR085W"],
    }
    assert _result(report, "two_distinct_genes").details["n_bad"] == 3


def test_release_accounting_and_rows_fail_on_drift_and_on_counts() -> None:
    drift = _verify(_records(), released=_released(drift={"Yeast_SL.csv": "x"}))
    assert _result(drift, "released_rows").message == (
        "sha256 drift in ['Yeast_SL.csv']: the release was not read"
    )
    assert _result(drift, "release_accounting").message == "the release was not read"
    off = _verify(_records(), released=_released(n_rows=5, stated=6))
    assert _result(off, "release_accounting").message == (
        "5 released rows != 2 stored + 2 dropped; 5 released rows != the 6 the "
        "release paper states"
    )


def test_an_empty_store_fails_the_rows_that_need_a_record() -> None:
    report = _verify([], expected_count=0, released=_released(Counter(), n_rows=2))
    failed = {r.name for r in report.results if not r.passed}
    assert {
        "structural",
        "two_distinct_genes",
        "unordered_pair_uniqueness",
        "positive_label",
        "reference_negative",
    } <= failed
