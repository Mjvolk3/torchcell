# tests/torchcell/verification/test_construct_placement.py
# [[tests.torchcell.verification.test_construct_placement]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/verification/test_construct_placement.py
"""The construct-placement census: three typed states, counted, and reported not gated."""

from __future__ import annotations

from typing import Any

from torchcell.verification.construct_placement import (
    RULE_NAME,
    ConstructPlacementState,
    construct_placement_census,
    construct_placement_result,
    l1_construct_placement,
)
from torchcell.verification.report import (
    FindingClass,
    Level,
    Provenance,
    VerificationReport,
    finding_class_of,
)

CONTIG = "63f89c1eaa13f888610ed781c4de12c8ac34f4e20512b4eb2c4f32d31af096da"
OTHER = "bbaca60509e079222f7cdd1bb54b70d93445af3a699c65109d94b553fc0087f7"


def _addition(**kwargs: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "perturbation_type": "gene_addition",
        "placement": None,
        "provenance_gaps": [],
    }
    record.update(kwargs)
    return record


def _sequence_backed(sha256: str = CONTIG) -> dict[str, Any]:
    return _addition(
        placement={
            "contig_sha256": sha256,
            "host_placement": {
                "assembly_set": "sgd_S288C_R64-4-1_20230830",
                "site_label": "XII-5",
                "span": {
                    "chromosome": "XII",
                    "start": 1,
                    "end": 9,
                    "assembly": "sgd_S288C_R64-4-1_20230830",
                },
            },
        }
    )


def _host_placed_without_coordinate(sha256: str = CONTIG) -> dict[str, Any]:
    return _addition(
        placement={
            "contig_sha256": sha256,
            "host_placement": {
                "assembly_set": "sgd_S288C_R64-4-1_20230830",
                "site_label": "XII-5",
                "span": None,
            },
        }
    )


def _episomal(sha256: str = CONTIG) -> dict[str, Any]:
    return _addition(placement={"contig_sha256": sha256, "host_placement": None})


def _composition_only(reason: str = "not_released_by_primary") -> dict[str, Any]:
    return _addition(
        provenance_gaps=[
            {"field": "placement", "reason": reason, "looked_in": None},
            {"field": "n_samples", "reason": "not_reported_by_primary"},
        ]
    )


def test_the_three_states_are_counted_separately() -> None:
    """A deposited pointer, a named pad with no coordinate, and a typed absence differ.

    All three look identical today, because every served gene addition carries
    ``plasmid_contig_id=None``: 26 of 26 stores. Separating them is what turns the
    unfilled ones into a worklist rather than a pile of nulls.
    """
    census = construct_placement_census(
        [
            _sequence_backed(),
            _episomal(),
            _host_placed_without_coordinate(),
            _composition_only(),
        ]
    )
    assert census.n_additions == 4
    assert census.n_sequence_backed == 2
    assert census.n_host_placed_without_coordinate == 1
    assert census.n_composition_only == 1
    assert census.n_untyped == 0


def test_an_episomal_placement_is_sequence_backed() -> None:
    """A present plasmid has no host site, and that is not a missing coordinate."""
    census = construct_placement_census([_episomal()])
    assert census.n_sequence_backed == 1
    assert census.n_host_placed_without_coordinate == 0


def test_an_untyped_absence_is_counted_apart_from_a_declared_one() -> None:
    """A silent ``None`` and a typed gap are different statements; one is honest.

    Folding the untyped one into ``composition_only`` would report a defect of the
    record as a property of the source, which is exactly the confusion the typed states
    exist to end.
    """
    census = construct_placement_census([_addition(), _composition_only()])
    assert (census.n_untyped, census.n_composition_only) == (1, 1)
    result = construct_placement_result(census)
    assert result.passed is False
    assert "UNTYPED" in result.message


def test_only_a_gap_on_placement_makes_a_record_composition_only() -> None:
    """A gap on another field is not a statement about the construct's sequence."""
    census = construct_placement_census(
        [
            _addition(
                provenance_gaps=[
                    {"field": "n_samples", "reason": "not_reported_by_primary"}
                ]
            )
        ]
    )
    assert (census.n_composition_only, census.n_untyped) == (0, 1)


def test_non_addition_perturbations_are_not_counted() -> None:
    """A deletion has no construct, so it is not a composition-only addition."""
    census = construct_placement_census(
        [
            {"perturbation_type": "deletion", "systematic_gene_name": "YAL001C"},
            {
                "perturbation_type": "heterologous_pathway",
                "placement": None,
                "provenance_gaps": [
                    {"field": "placement", "reason": "not_released_by_primary"}
                ],
            },
            _sequence_backed(),
        ]
    )
    assert census.n_additions == 2
    assert census.n_composition_only == 1


def test_additions_are_counted_per_contig_by_handle() -> None:
    """The per-contig count is the cross-dataset join a shared plasmid makes possible."""
    census = construct_placement_census(
        [_sequence_backed(), _sequence_backed(), _episomal(OTHER)]
    )
    assert census.by_contig == {CONTIG[:12]: 2, OTHER[:12]: 1}


def test_the_gap_reasons_are_tallied() -> None:
    """Why a store has no construct sequence is as countable as how many lack one."""
    census = construct_placement_census(
        [
            _composition_only("not_released_by_primary"),
            _composition_only("not_released_by_primary"),
            _composition_only("deferred_pending_source_review"),
        ]
    )
    assert census.gap_reasons == {
        "deferred_pending_source_review": 1,
        "not_released_by_primary": 2,
    }


def test_a_store_with_no_additions_says_so_and_passes() -> None:
    """An empty count is not a failure; most stores have no gene addition at all."""
    result = l1_construct_placement([{"perturbation_type": "deletion"}])
    assert result.passed is True
    assert result.message == "no gene additions in this store"
    assert result.level is Level.L1


def test_the_row_is_report_class_and_never_fails_a_store() -> None:
    """A source that released no plasmid map must not fail its store's verdict.

    "The data is the data ... the genome is separate.. it doesn't gate anything" (owner,
    2026-10-12). The row's subject is the record against OUR sequence references, so it
    is classified ``report``: a store whose only failing row is this one still passes,
    with the row listed as an open finding.
    """
    assert finding_class_of(RULE_NAME) is FindingClass.report
    result = l1_construct_placement([_addition()])
    assert result.passed is False
    assert result.finding_class is FindingClass.report
    report = VerificationReport(
        dataset_name="toy", provenance=Provenance(source_uri="test://construct")
    )
    report.add(
        construct_placement_result(construct_placement_census([_sequence_backed()]))
    )
    report.add(
        type(result)(
            level=Level.L1, name="record_count", passed=True, message="counts hold"
        )
    )
    report.add(result)
    assert report.passed is True
    assert [row.name for row in report.open_findings] == [RULE_NAME]


def test_the_message_reports_the_statistic() -> None:
    """A report row carries its number, which is what makes it evidence."""
    result = l1_construct_placement(
        [_sequence_backed(), _host_placed_without_coordinate(), _composition_only()]
    )
    assert "1/3 additions are sequence-backed" in result.message
    assert result.details["n_composition_only"] == 1


def test_the_states_are_exactly_three() -> None:
    """Audit B 2.5's three states, pinned so a fourth is a deliberate change."""
    assert [state.value for state in ConstructPlacementState] == [
        "sequence_backed",
        "host_placed_without_coordinate",
        "composition_only",
    ]
