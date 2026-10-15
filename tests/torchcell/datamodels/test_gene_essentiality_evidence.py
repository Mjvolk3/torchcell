# tests/torchcell/datamodels/test_gene_essentiality_evidence.py
"""``GeneEssentialityEvidence`` and the ``evidence`` field of
``GeneEssentialityPhenotype`` (#924): the two-level type, the per-type statistics and
their refusals, the released-call agreement, and the round trip.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from tests.torchcell.datamodels._essentiality_evidence import (
    curated_evidence,
    statement,
    statements,
)
from torchcell.datamodels.schema import (
    ESSENTIALITY_EVIDENCE_CLASS,
    ESSENTIALITY_EVIDENCE_FIELDS,
    FOOTPRINTING_CALL_IS_ESSENTIAL,
    GRADED_ESSENTIALITY_CALLS,
    TRADIS_CALL_IS_ESSENTIAL,
    EssentialityEvidenceClass,
    EssentialityEvidenceType,
    EssentialityStatement,
    EssentialityStatementRole,
    FootprintingCall,
    GeneEssentialityEvidence,
    GeneEssentialityPhenotype,
    GradedGeneEssentialityPhenotype,
    HmmEssentialityState,
    TradisCall,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

T = EssentialityEvidenceType
C = EssentialityEvidenceClass


def _evidence(
    type_: EssentialityEvidenceType, **stats: Any
) -> GeneEssentialityEvidence:
    return GeneEssentialityEvidence(
        evidence_class=ESSENTIALITY_EVIDENCE_CLASS[type_],
        evidence_type=type_,
        statements=statements(),
        **stats,
    )


def _gap(field: str) -> ProvenanceGap:
    return ProvenanceGap(
        field=field,
        reason=ProvenanceGapReason.not_reported_by_primary,
        looked_in=Provenance(source_uri="table.xls", citation_key="k", sha256="0" * 64),
        note="the cell reads n.d.",
    )


#: One fully populated record per evidence type, as each loader writes it.
RECORDS: dict[EssentialityEvidenceType, dict[str, Any]] = {
    T.construction_outcome: {
        "colonies_verified": 1,
        "colonies_tested": 8,
        "percent_verified": 12.5,
    },
    T.genetic_footprinting: {
        "footprinting_call": FootprintingCall.essential,
        "footprint_insertions_aa": [],
        "assertion_error": 0.05,
        "assertion_error_is_upper_bound": True,
    },
    T.tradis_insertion_index: {
        "tradis_call": TradisCall.essential,
        "insertion_index": 0.002,
        "log_likelihood_ratio": -12.0,
    },
    T.rb_tnseq_coverage_and_fitness: {
        "tnseq_reads": 25,
        "normalized_read_density": 0.03,
        "central_insertion_positions": 0,
        "normalized_insertion_density": 0.0,
        "gene_gc_fraction": 0.52,
        "false_discovery_rate_lower": 0.06,
        "false_discovery_rate_upper": 0.16,
    },
    T.curated_null_phenotype: {
        "curated_experiment_type": "classical genetics",
        "curated_reference_pubmed_id": "12140549",
    },
    T.tnseq_hmm_state: {
        "hmm_state": HmmEssentialityState.essential,
        "ta_sites": 8,
        "hmm_sites_essential": 5,
        "hmm_sites_growth_defect": 0,
        "hmm_sites_non_essential": 3,
        "hmm_sites_growth_advantage": 0,
        "mean_insertion_rate": 0.5,
        "mean_read_count": 23.0,
    },
}
#: The evidence types whose call is graded, never a boolean (#909).
GRADED_ONLY = {T.tnseq_hmm_state}
BOOLEAN_TYPES = [t for t in T if t not in GRADED_ONLY]


# --------------------------------------------------------------------------- #
# The DAG of the two enums and the per-type field map
# --------------------------------------------------------------------------- #
def test_every_type_has_exactly_one_parent_and_the_parents_are_the_issue_table() -> (
    None
):
    assert set(ESSENTIALITY_EVIDENCE_CLASS) == set(T)
    assert ESSENTIALITY_EVIDENCE_CLASS == {
        T.construction_outcome: C.mutant_recovery,
        T.genetic_footprinting: C.insertion_depletion,
        T.tradis_insertion_index: C.insertion_depletion,
        T.rb_tnseq_coverage_and_fitness: C.insertion_depletion,
        T.curated_null_phenotype: C.curated_annotation,
        T.tnseq_hmm_state: C.insertion_depletion,
    }
    assert set(ESSENTIALITY_EVIDENCE_CLASS.values()) == set(C)


def test_the_field_map_names_every_statistic_once_and_each_defaults_to_none() -> None:
    names = [n for fields in ESSENTIALITY_EVIDENCE_FIELDS.values() for n in fields]
    assert set(ESSENTIALITY_EVIDENCE_FIELDS) == set(T)
    # 18 statistics of the five measured types, 8 of tnseq_hmm_state, and since lane V
    # (2026-10-15) the curated family's own reference, curated_reference_pubmed_id
    assert len(names) == len(set(names)) == 18 + 8 + 1
    shared = {"evidence_class", "evidence_type", "statements", "provenance_gaps"}
    model_fields = GeneEssentialityEvidence.model_fields
    assert set(model_fields) == shared | set(names)
    assert all(model_fields[n].default is None for n in names)
    assert {t: set(f) for t, f in ESSENTIALITY_EVIDENCE_FIELDS.items()} == {
        t: set(stats) for t, stats in RECORDS.items()
    }


def test_released_call_booleans() -> None:
    assert FOOTPRINTING_CALL_IS_ESSENTIAL == {
        FootprintingCall.essential: True,
        FootprintingCall.non_essential: False,
        FootprintingCall.non_essential_flagged: False,
        FootprintingCall.not_covered: None,
        FootprintingCall.ambiguous: None,
    }
    assert [c.value for c in FootprintingCall] == ["E", "N", "N*", "X", "?"]
    assert TRADIS_CALL_IS_ESSENTIAL == {
        TradisCall.essential: True,
        TradisCall.non_essential: False,
        TradisCall.unclear: None,
    }


# --------------------------------------------------------------------------- #
# Construction and round trip
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("type_", BOOLEAN_TYPES)
def test_each_type_round_trips_on_the_phenotype(
    type_: EssentialityEvidenceType,
) -> None:
    evidence = _evidence(type_, **RECORDS[type_])
    phenotype = GeneEssentialityPhenotype(is_essential=True, evidence=evidence)
    assert GeneEssentialityPhenotype.model_validate(phenotype.model_dump()) == phenotype
    assert (
        GeneEssentialityPhenotype.model_validate_json(phenotype.model_dump_json())
        == phenotype
    )
    assert phenotype.evidence.evidence_class == ESSENTIALITY_EVIDENCE_CLASS[type_]


@pytest.mark.parametrize("type_", BOOLEAN_TYPES)
def test_a_reference_carries_the_type_and_no_statistics(
    type_: EssentialityEvidenceType,
) -> None:
    reference = GeneEssentialityPhenotype(is_essential=False, evidence=_evidence(type_))
    dumped = reference.evidence.model_dump()
    assert all(
        dumped[n] is None for f in ESSENTIALITY_EVIDENCE_FIELDS.values() for n in f
    )
    assert [s["role"] for s in dumped["statements"]] == [
        "scoring_rule",
        "call_definition",
    ]


def test_the_phenotype_requires_evidence() -> None:
    with pytest.raises(ValidationError, match="evidence"):
        GeneEssentialityPhenotype(is_essential=True)  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# Refusals on the descriptor
# --------------------------------------------------------------------------- #
def test_a_class_that_is_not_the_types_parent_is_refused() -> None:
    with pytest.raises(ValidationError, match="is a insertion_depletion type"):
        GeneEssentialityEvidence(
            evidence_class=C.mutant_recovery,
            evidence_type=T.genetic_footprinting,
            statements=statements(),
        )


@pytest.mark.parametrize("type_", list(T))
def test_another_types_statistic_is_refused(type_: EssentialityEvidenceType) -> None:
    foreign = next(t for t in T if t != type_)
    name, value = next(iter(RECORDS[foreign].items()))
    with pytest.raises(ValidationError, match=f"{name} belongs to {foreign.value}"):
        _evidence(type_, **{name: value})


def test_a_gap_on_another_types_statistic_is_refused() -> None:
    with pytest.raises(ValidationError, match="assertion_error belongs to"):
        _evidence(T.construction_outcome, provenance_gaps=[_gap("assertion_error")])


@pytest.mark.parametrize(
    ("stats", "match"),
    [
        ({"colonies_verified": 1, "colonies_tested": 8}, "set together"),
        (
            {"colonies_verified": 9, "colonies_tested": 8, "percent_verified": 112.5},
            "9 of 8 colonies",
        ),
        (
            {"colonies_verified": 0, "colonies_tested": 0, "percent_verified": 0.0},
            "0 of 0 colonies",
        ),
        (
            {"colonies_verified": 2, "colonies_tested": 8, "percent_verified": 20.0},
            "is not 100 x 2 / 8",
        ),
    ],
)
def test_colony_counts_are_whole_and_consistent(
    stats: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        _evidence(T.construction_outcome, **stats)


@pytest.mark.parametrize(
    ("stats", "match"),
    [
        ({"assertion_error": 0.1}, "set together"),
        ({"assertion_error_is_upper_bound": False}, "set together"),
        (
            {"assertion_error": 1.5, "assertion_error_is_upper_bound": False},
            "no probability",
        ),
        ({"footprinting_call": FootprintingCall.essential}, "typed ProvenanceGap"),
        ({"footprint_insertions_aa": [3, -1]}, "negative position"),
    ],
)
def test_footprinting_statistics(stats: dict[str, Any], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        _evidence(T.genetic_footprinting, **stats)


def test_an_e_call_with_a_gapped_assertion_error_is_admitted() -> None:
    evidence = _evidence(
        T.genetic_footprinting,
        footprinting_call=FootprintingCall.essential,
        footprint_insertions_aa=[],
        provenance_gaps=[_gap("assertion_error")],
    )
    assert evidence.gapped_fields() == {"assertion_error"}


def test_a_gapped_assertion_error_cannot_also_hold_a_value() -> None:
    with pytest.raises(ValidationError, match="cannot both store a value"):
        _evidence(
            T.genetic_footprinting,
            assertion_error=0.1,
            assertion_error_is_upper_bound=False,
            provenance_gaps=[_gap("assertion_error")],
        )


def test_an_n_call_needs_no_assertion_error() -> None:
    evidence = _evidence(
        T.genetic_footprinting, footprinting_call=FootprintingCall.non_essential
    )
    assert evidence.assertion_error is None and not evidence.provenance_gaps


@pytest.mark.parametrize(
    ("type_", "stats", "match"),
    [
        (T.tradis_insertion_index, {"insertion_index": -0.1}, "insertion_index"),
        (T.rb_tnseq_coverage_and_fitness, {"tnseq_reads": -1}, "tnseq_reads -1 < 0"),
        (
            T.rb_tnseq_coverage_and_fitness,
            {"normalized_insertion_density": -0.5},
            "normalized_insertion_density",
        ),
        (T.rb_tnseq_coverage_and_fitness, {"gene_gc_fraction": 1.2}, "no fraction"),
        (
            T.rb_tnseq_coverage_and_fitness,
            {"false_discovery_rate_lower": 0.06},
            "set together",
        ),
        (
            T.rb_tnseq_coverage_and_fitness,
            {"false_discovery_rate_lower": 0.2, "false_discovery_rate_upper": 0.1},
            "0 <= low <= high <= 1",
        ),
    ],
)
def test_insertion_statistics_are_in_range(
    type_: EssentialityEvidenceType, stats: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        _evidence(type_, **stats)


# --------------------------------------------------------------------------- #
# The released call against the boolean
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("call", "is_essential"),
    [
        (FootprintingCall.essential, True),
        (FootprintingCall.non_essential, False),
        (FootprintingCall.non_essential_flagged, False),
    ],
)
def test_a_footprinting_call_with_a_boolean_is_stored(
    call: FootprintingCall, is_essential: bool
) -> None:
    stats: dict[str, Any] = {"footprinting_call": call}
    if call is FootprintingCall.essential:
        stats |= {"assertion_error": 0.3, "assertion_error_is_upper_bound": False}
    phenotype = GeneEssentialityPhenotype(
        is_essential=is_essential, evidence=_evidence(T.genetic_footprinting, **stats)
    )
    assert phenotype.evidence.footprinting_call is call


@pytest.mark.parametrize(
    "call", [FootprintingCall.not_covered, FootprintingCall.ambiguous]
)
def test_x_and_question_mark_are_refused_as_graded(call: FootprintingCall) -> None:
    for is_essential in (True, False):
        with pytest.raises(ValidationError, match="has no boolean"):
            GeneEssentialityPhenotype(
                is_essential=is_essential,
                evidence=_evidence(T.genetic_footprinting, footprinting_call=call),
            )


def test_a_footprinting_call_disagreeing_with_the_boolean_is_refused() -> None:
    with pytest.raises(ValidationError, match="is not is_essential=True"):
        GeneEssentialityPhenotype(
            is_essential=True,
            evidence=_evidence(
                T.genetic_footprinting, footprinting_call=FootprintingCall.non_essential
            ),
        )


def test_a_tradis_unclear_call_is_refused_and_a_disagreeing_call_too() -> None:
    with pytest.raises(ValidationError, match="TraDIS call unclear has no boolean"):
        GeneEssentialityPhenotype(
            is_essential=False,
            evidence=_evidence(
                T.tradis_insertion_index, tradis_call=TradisCall.unclear
            ),
        )
    with pytest.raises(ValidationError, match="is not is_essential=False"):
        GeneEssentialityPhenotype(
            is_essential=False,
            evidence=_evidence(
                T.tradis_insertion_index, tradis_call=TradisCall.essential
            ),
        )


def test_the_helper_evidence_is_curated() -> None:
    evidence = curated_evidence("systematic mutation set")
    assert evidence.evidence_class is C.curated_annotation
    assert evidence.curated_experiment_type == "systematic mutation set"


# --------------------------------------------------------------------------- #
# The role-typed statements
# --------------------------------------------------------------------------- #
def test_statements_are_read_back_by_role() -> None:
    caveat = statement("caveat", "a quoted caveat")
    definition = statement("statistic", "a quoted statistic")
    evidence = GeneEssentialityEvidence(
        evidence_class=C.mutant_recovery,
        evidence_type=T.construction_outcome,
        statements=statements(
            (EssentialityStatementRole.caveat, caveat),
            (EssentialityStatementRole.statistic_definition, definition),
        ),
    )
    assert evidence.scoring_rule == statement()
    assert evidence.call_definition.value == "definition"
    assert evidence.statements_of(EssentialityStatementRole.caveat) == [caveat]
    assert evidence.statements_of(EssentialityStatementRole.statistic_definition) == [
        definition
    ]


@pytest.mark.parametrize(
    ("roles", "match"),
    [
        ([EssentialityStatementRole.call_definition], "0 scoring_rule statements"),
        (
            [
                EssentialityStatementRole.scoring_rule,
                EssentialityStatementRole.scoring_rule,
                EssentialityStatementRole.call_definition,
            ],
            "2 scoring_rule statements",
        ),
        ([EssentialityStatementRole.scoring_rule], "0 call_definition statements"),
    ],
)
def test_exactly_one_rule_and_one_definition(
    roles: list[EssentialityStatementRole], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        GeneEssentialityEvidence(
            evidence_class=C.curated_annotation,
            evidence_type=T.curated_null_phenotype,
            statements=[
                EssentialityStatement(role=role, provenance=statement())
                for role in roles
            ],
        )


# --------------------------------------------------------------------------- #
# #909: the HMM evidence and the graded sibling
# --------------------------------------------------------------------------- #
def test_the_hmm_states_are_the_releases_spellings() -> None:
    assert [state.value for state in HmmEssentialityState] == [
        "ES",
        "GD",
        "NE",
        "GA",
        "N/A",
    ]
    assert GRADED_ESSENTIALITY_CALLS == {
        T.genetic_footprinting: (
            "footprinting_call",
            frozenset({"E", "N", "N*", "X", "?"}),
        ),
        T.tradis_insertion_index: (
            "tradis_call",
            frozenset({"essential", "non_essential", "unclear"}),
        ),
        T.tnseq_hmm_state: ("hmm_state", frozenset({"ES", "GD", "NE", "GA", "N/A"})),
    }


@pytest.mark.parametrize(
    ("stats", "match"),
    [
        ({"hmm_sites_essential": 1}, "set together"),
        (
            {
                "ta_sites": 8,
                "hmm_sites_essential": 5,
                "hmm_sites_growth_defect": 0,
                "hmm_sites_non_essential": 2,
                "hmm_sites_growth_advantage": 0,
            },
            "are not the gene's 8 TA sites",
        ),
        (
            {"hmm_state": HmmEssentialityState.not_applicable, "ta_sites": 3},
            "N/A is exactly the gene with no TA site",
        ),
        (
            {"hmm_state": HmmEssentialityState.essential, "ta_sites": 0},
            "N/A is exactly the gene with no TA site",
        ),
        ({"mean_insertion_rate": 1.2}, "is not a fraction"),
        ({"mean_read_count": -1.0}, "mean_read_count -1.0 < 0"),
    ],
)
def test_hmm_statistics_are_consistent(stats: dict[str, Any], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        _evidence(T.tnseq_hmm_state, **stats)


def test_hmm_evidence_is_refused_on_the_boolean_leaf() -> None:
    with pytest.raises(ValidationError, match="calls a graded state, not a boolean"):
        GeneEssentialityPhenotype(
            is_essential=True,
            evidence=_evidence(T.tnseq_hmm_state, **RECORDS[T.tnseq_hmm_state]),
        )


@pytest.mark.parametrize(
    ("type_", "stats", "call"),
    [
        (T.tnseq_hmm_state, RECORDS[T.tnseq_hmm_state], "ES"),
        (
            T.tnseq_hmm_state,
            {"hmm_state": HmmEssentialityState.not_applicable, "ta_sites": 0},
            "N/A",
        ),
        (
            T.tradis_insertion_index,
            {
                "tradis_call": TradisCall.unclear,
                "insertion_index": 0.05,
                "log_likelihood_ratio": 1.0,
            },
            "unclear",
        ),
        (
            T.genetic_footprinting,
            {"footprinting_call": FootprintingCall.ambiguous},
            "?",
        ),
    ],
)
def test_a_graded_call_round_trips(
    type_: EssentialityEvidenceType, stats: dict[str, Any], call: str
) -> None:
    phenotype = GradedGeneEssentialityPhenotype(
        essentiality_call=call, evidence=_evidence(type_, **stats)
    )
    assert phenotype.label_name == "essentiality_call"
    assert (
        GradedGeneEssentialityPhenotype.model_validate_json(phenotype.model_dump_json())
        == phenotype
    )


def test_a_graded_reference_states_the_types_non_essential_call() -> None:
    reference = GradedGeneEssentialityPhenotype(
        essentiality_call="NE", evidence=_evidence(T.tnseq_hmm_state)
    )
    assert reference.evidence.hmm_state is None


@pytest.mark.parametrize(
    ("type_", "stats", "call", "match"),
    [
        (T.construction_outcome, {}, "essential", "releases no categorical call"),
        (T.curated_null_phenotype, {}, "inviable", "releases no categorical call"),
        (T.tnseq_hmm_state, {}, "unclear", "is not a tnseq_hmm_state call"),
        (T.tradis_insertion_index, {}, "ES", "is not a tradis_insertion_index call"),
        (
            T.tnseq_hmm_state,
            {"hmm_state": HmmEssentialityState.growth_defect},
            "ES",
            "is not the evidence's hmm_state 'GD'",
        ),
    ],
)
def test_a_graded_call_off_its_types_vocabulary_is_refused(
    type_: EssentialityEvidenceType, stats: dict[str, Any], call: str, match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        GradedGeneEssentialityPhenotype(
            essentiality_call=call, evidence=_evidence(type_, **stats)
        )
