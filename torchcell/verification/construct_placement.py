# torchcell/verification/construct_placement.py
# [[torchcell.verification.construct_placement]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/construct_placement.py
# Test file: tests/torchcell/verification/test_construct_placement.py

"""Count the three typed states of a gene addition's material: the store's coverage row.

Today a hand-typed composition is indistinguishable from a sequence-backed one, because
both carry ``plasmid_contig_id=None``: 26 of 26 served gene-addition stores look
identical whether the construct's sequence is deposited or was never released. With a
``placement`` that is set or carries a typed ``ProvenanceGap`` the three states separate
and become countable:

1. ``sequence_backed`` -- ``placement`` is set and its ``contig_ref`` pins the bytes.
2. ``host_placed_without_coordinate`` -- ``placement`` is set and names a host site, but
   that site's ``span`` is itself gapped: a named landing pad with no published
   coordinate, which is most integrations (Cachera's ``XII-5``, Lian's ``X3``).
3. ``composition_only`` -- ``placement`` is ``None`` with a typed gap. The parts are
   still stored exactly as they are today, and the absence is explicit.

The row is REPORT-class, by the standing rule: its subject is the record compared with
OUR sequence references, not the record's fidelity to its own release, so it is listed
under the store's verdict with its statistic and never fails a build and never retires a
build manifest. A store whose every addition is ``composition_only`` is a correct store
of a source that released no sequence; what the count buys is a worklist that names
exactly what a retrieval campaign should go and get (1 of 15 datasets is fillable from
the mirror today, which is the honest number).

The census reads MAPPINGS, the JSON dicts a built LMDB yields, like
``provenance_gap_census``, so it runs over a built store without reconstructing models.
Wiring it into the family runners waits for the composition step that puts ``placement``
on the addition leaf; until then this is importable and tested, and nothing calls it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from torchcell.verification.report import Level, LevelResult

#: The rule name this module emits. Registered in ``report.GENOME_REFERENCE_RULES`` so
#: it classifies ``report``; renaming it here without renaming it there would silently
#: turn the row into a gate.
RULE_NAME = "construct_placement_states"

#: The perturbation types whose material this row is about.
ADDITION_PERTURBATION_TYPES: frozenset[str] = frozenset(
    {"gene_addition", "heterologous_pathway"}
)


class ConstructPlacementState(StrEnum):
    """The three typed states of one gene addition's material."""

    #: ``placement`` set, its bytes pinned by an ``ArtifactRef``.
    sequence_backed = "sequence_backed"
    #: ``placement`` set with a host site whose coordinate is itself a typed gap.
    host_placed_without_coordinate = "host_placed_without_coordinate"
    #: ``placement`` absent with a typed gap: the composition is stored, the sequence is not.
    composition_only = "composition_only"


class ConstructPlacementCensus(BaseModel):
    """How many additions in a store are sequence-backed, host-placed, composition-only.

    ``n_untyped`` counts additions that are neither: ``placement`` is ``None`` and no
    ``ProvenanceGap`` names it. That is the silent ``None`` the typed states exist to
    abolish, so it is counted separately and reported rather than folded into
    ``composition_only`` -- an untyped absence and a declared one are different
    statements, and only one of them is honest.
    """

    model_config = ConfigDict(extra="forbid")

    n_additions: int
    n_sequence_backed: int
    n_host_placed_without_coordinate: int
    n_composition_only: int
    n_untyped: int
    by_contig: dict[str, int] = Field(
        default_factory=dict,
        description="additions per construct contig handle (the 12-hex prefix)",
    )
    gap_reasons: dict[str, int] = Field(
        default_factory=dict,
        description="reasons the composition-only additions give for having no sequence",
    )


def _placement_gap_reason(perturbation: Mapping[str, Any]) -> str | None:
    """The reason of a ``ProvenanceGap`` on ``placement``, or None when there is none."""
    for gap in perturbation.get("provenance_gaps") or []:
        if str(gap.get("field")) == "placement":
            return str(gap.get("reason"))
    return None


def _state_of(
    perturbation: Mapping[str, Any],
) -> tuple[ConstructPlacementState | None, str | None]:
    """One addition's state, and its gap reason when it is composition-only."""
    placement = perturbation.get("placement")
    if placement is None:
        reason = _placement_gap_reason(perturbation)
        if reason is None:
            return None, None
        return ConstructPlacementState.composition_only, reason
    host = placement.get("host_placement")
    if host is not None and host.get("span") is None:
        return ConstructPlacementState.host_placed_without_coordinate, None
    return ConstructPlacementState.sequence_backed, None


def construct_placement_census(
    perturbations: Iterable[Mapping[str, Any]],
) -> ConstructPlacementCensus:
    """Tally the placement states over a stream of perturbation mappings.

    Reads only the addition perturbations (``perturbation_type`` in
    ``ADDITION_PERTURBATION_TYPES``); a deletion has no construct and is not a
    composition-only addition. Single-pass and memory-light: counters only.
    """
    counts: Counter[ConstructPlacementState] = Counter()
    by_contig: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    n_additions = 0
    n_untyped = 0
    for perturbation in perturbations:
        if (
            str(perturbation.get("perturbation_type"))
            not in ADDITION_PERTURBATION_TYPES
        ):
            continue
        n_additions += 1
        state, reason = _state_of(perturbation)
        if state is None:
            n_untyped += 1
            continue
        counts[state] += 1
        if reason is not None:
            reasons[reason] += 1
        placement = perturbation.get("placement")
        if placement is not None:
            sha256 = str(placement.get("contig_sha256", ""))
            if sha256:
                by_contig[sha256[:12]] += 1
    return ConstructPlacementCensus(
        n_additions=n_additions,
        n_sequence_backed=counts[ConstructPlacementState.sequence_backed],
        n_host_placed_without_coordinate=counts[
            ConstructPlacementState.host_placed_without_coordinate
        ],
        n_composition_only=counts[ConstructPlacementState.composition_only],
        n_untyped=n_untyped,
        by_contig=dict(sorted(by_contig.items())),
        gap_reasons=dict(sorted(reasons.items())),
    )


def construct_placement_result(census: ConstructPlacementCensus) -> LevelResult:
    """Wrap a census into the report-class ``construct_placement_states`` row.

    ``passed`` is False only when an addition's material is an UNTYPED absence, which is
    a defect of the record rather than of the source. Being report-class, even that does
    not fail the store: it is listed as an open finding under the verdict, which is what
    makes it visible without letting the genome gate a record.
    """
    if census.n_additions == 0:
        message = "no gene additions in this store"
    else:
        message = (
            f"{census.n_sequence_backed}/{census.n_additions} additions are "
            f"sequence-backed, {census.n_host_placed_without_coordinate} are placed at "
            f"a named host site with no coordinate, {census.n_composition_only} are "
            f"composition-only with a typed gap"
        )
        if census.n_untyped:
            message += f"; {census.n_untyped} carry an UNTYPED absence"
    return LevelResult(
        level=Level.L1,
        name=RULE_NAME,
        passed=census.n_untyped == 0,
        message=message,
        details=census.model_dump(),
    )


def l1_construct_placement(perturbations: Iterable[Mapping[str, Any]]) -> LevelResult:
    """The report-class placement-state row over a stream of perturbation mappings."""
    return construct_placement_result(construct_placement_census(perturbations))
