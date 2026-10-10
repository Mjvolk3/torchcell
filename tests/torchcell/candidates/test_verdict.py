# tests/torchcell/candidates/test_verdict.py
# [[tests.torchcell.candidates.test_verdict]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/candidates/test_verdict.py
"""The verdict models: field rules, gate order, the stop rule, and outcome derivation."""

from typing import Any

import pytest
from pydantic import ValidationError

from tests.torchcell.candidates._builders import (
    aggregation,
    evidence,
    gate,
    unmeasured_tail,
    verdict,
)
from torchcell.candidates.verdict import (
    AggregationRecord,
    G2Host,
    G2Record,
    G4KeyHit,
    G4KeyRecord,
    G4ValueMatch,
    G4ValueRecord,
    GateResult,
    InventoryFile,
    derive_outcome,
    verdict_json_schema,
)

READABLE = G2Host(name="MG1655", assembly_set="set", resolvable=True, readable=True)
UNREADABLE = G2Host(name="W3110", assembly_set="w", resolvable=True, readable=False)
ABSENT = G2Host(name="BL21", assembly_set=None, resolvable=False, readable=False)


def test_gap_requires_an_issue() -> None:
    """A gap without an issue number is refused; with one it validates."""
    with pytest.raises(ValidationError, match="a 'gap' must name its issue"):
        GateResult(gate="G5", outcome="gap", reason="no class")
    assert (
        GateResult(gate="G5", outcome="gap", reason="no class", issue=854).issue == 854
    )


@pytest.mark.parametrize(
    ("outcome", "kind"), [("blocked", None), ("pass", "wall"), ("fail", "provision")]
)
def test_blocked_kind_only_with_blocked(outcome: str, kind: str | None) -> None:
    """blocked_kind is required for, and only for, a blocked gate."""
    with pytest.raises(ValidationError, match="blocked_kind is required for"):
        GateResult(gate="G3", outcome=outcome, reason="r", blocked_kind=kind)  # type: ignore[arg-type]


def test_aggregation_counts_cannot_exceed_the_studies() -> None:
    """Mirrored or net-new studies above the study count are refused."""
    with pytest.raises(ValidationError, match="n_sources_mirrored=4 exceeds"):
        AggregationRecord(
            n_source_studies=3,
            attribution_field="f",
            value_origin="re_measured",
            n_sources_mirrored=4,
            net_new_vs_served=None,
            evidence=(evidence(),),
        )
    with pytest.raises(ValidationError, match="at least 1 item"):
        AggregationRecord(
            n_source_studies=3,
            attribution_field="f",
            value_origin="re_measured",
            n_sources_mirrored=None,
            net_new_vs_served=None,
            evidence=(),
        )


def test_g2_host_rules() -> None:
    """Readable implies resolvable; resolvable implies an assembly set."""
    with pytest.raises(ValidationError, match="readable but not resolvable"):
        G2Host(name="x", assembly_set="s", resolvable=False, readable=True)
    with pytest.raises(ValidationError, match="resolvable without an assembly set"):
        G2Host(name="x", assembly_set=None, resolvable=True, readable=False)


@pytest.mark.parametrize(
    ("hosts", "outcome"),
    [
        ((READABLE,), "resolvable_readable"),
        ((READABLE, UNREADABLE), "resolvable_not_ingestible"),
        ((UNREADABLE, ABSENT), "absent_provisionable"),
    ],
)
def test_g2_record_outcome_is_the_worst_host(
    hosts: tuple[G2Host, ...], outcome: Any
) -> None:
    """The record must state the outcome its hosts imply, and nothing else."""
    record = G2Record(branch="assembly_set", outcome=outcome, hosts=hosts)
    assert record.outcome == outcome
    with pytest.raises(ValidationError, match="hosts imply"):
        G2Record(
            branch="assembly_set",
            outcome="resolvable_readable"
            if outcome != "resolvable_readable"
            else "absent_provisionable",
            hosts=hosts,
        )


def test_g2_record_mosaic_needs_two_parents_unless_unnamed() -> None:
    """A mosaic with one named parent is only valid as absent_unnamed."""
    with pytest.raises(ValidationError, match="exactly two parents"):
        G2Record(
            branch="haplotype_mosaic", outcome="resolvable_readable", hosts=(READABLE,)
        )
    record = G2Record(
        branch="haplotype_mosaic", outcome="absent_unnamed", hosts=(READABLE,)
    )
    assert record.hosts == (READABLE,)
    with pytest.raises(ValidationError, match="needs at least one host"):
        G2Record(branch="assembly_set", outcome="absent_provisionable", hosts=())


def test_a_valid_refusal_derives_refused() -> None:
    """G1 fail with every later gate unmeasured is a refusal; it is not passing."""
    v = verdict()
    assert (v.outcome, v.passing) == ("refused", False)
    assert v.gate("G1").outcome == "fail"


def test_gates_must_be_g1_to_g5_in_order() -> None:
    """A reordered or short gate list is refused."""
    gates = (gate("G2"), gate("G1"), *unmeasured_tail(("G3", "G4", "G5")))
    with pytest.raises(ValidationError, match="gates must be exactly"):
        verdict(gates=gates)


def test_stop_rule_after_fail_and_blocked() -> None:
    """Any gate after a fail or a blocked must be unmeasured."""
    after_fail = (gate("G1", "fail"), gate("G2"), *unmeasured_tail(("G3", "G4", "G5")))
    with pytest.raises(ValidationError, match="G2 is 'pass' after G1 stopped the run"):
        verdict(gates=after_fail)


@pytest.mark.parametrize(
    ("g1", "kind", "message"),
    [
        ("pass", "transcription", "G1 'pass' with source_kind 'transcription'"),
        ("fail", "primary", "G1 'fail' with source_kind 'primary'"),
        ("unmeasured", "primary", "G1 'unmeasured' with source_kind 'primary'"),
        ("gap", "primary", "G1 is pass, fail or unmeasured"),
    ],
)
def test_g1_outcome_and_source_kind_agree(g1: str, kind: str, message: str) -> None:
    """G1's outcome constrains the source kind."""
    extra = {"issue": 1} if g1 == "gap" else {}
    gates = (gate("G1", g1, **extra), *unmeasured_tail(("G2", "G3", "G4", "G5")))
    with pytest.raises(ValidationError, match=message):
        verdict(gates=gates, source_kind=kind)


def test_aggregation_record_iff_aggregation() -> None:
    """An aggregation needs its record and a primary may not carry one."""
    gates = (gate("G1"), *unmeasured_tail(("G2", "G3", "G4", "G5")))
    with pytest.raises(
        ValidationError, match="required for, and only for, aggregations"
    ):
        verdict(gates=gates, source_kind="aggregation")
    with pytest.raises(
        ValidationError, match="required for, and only for, aggregations"
    ):
        verdict(gates=gates, source_kind="primary", aggregation=aggregation())
    ok = verdict(gates=gates, source_kind="aggregation", aggregation=aggregation(5))
    assert (ok.outcome, ok.aggregation.n_source_studies if ok.aggregation else 0) == (
        "pending",
        5,
    )


def test_records_follow_their_gates() -> None:
    """A measured gate carries its record; an unmeasured one carries none."""
    gates = (gate("G1"), gate("G2"), *unmeasured_tail(("G3", "G4", "G5")))
    with pytest.raises(ValidationError, match="G2 is 'pass' without its record"):
        verdict(gates=gates, source_kind="primary")
    record = G2Record(
        branch="assembly_set", outcome="resolvable_readable", hosts=(READABLE,)
    )
    with pytest.raises(ValidationError, match="G2 is unmeasured but carries a record"):
        verdict(
            gates=(gate("G1"), *unmeasured_tail(("G2", "G3", "G4", "G5"))),
            source_kind="primary",
            g2=record,
        )
    v = verdict(gates=gates, source_kind="primary", g2=record)
    assert v.g2 == record


def test_g2_record_and_gate_must_agree() -> None:
    """A blocked-provision record under a passing gate is refused."""
    record = G2Record(
        branch="assembly_set", outcome="absent_provisionable", hosts=(ABSENT,)
    )
    gates = (gate("G1"), gate("G2"), *unmeasured_tail(("G3", "G4", "G5")))
    with pytest.raises(ValidationError, match="implies 'blocked', gate says 'pass'"):
        verdict(gates=gates, source_kind="primary", g2=record)


def test_g4_value_needs_g4_key_and_commit_and_date_are_checked() -> None:
    """g4_value alone, a short commit, and a non-ISO date are refused."""
    value = G4ValueRecord(
        store="s", store_commit=None, records_read=0, tolerance=0.01, matches=()
    )
    with pytest.raises(ValidationError, match="g4_value without g4_key"):
        verdict(g4_value=value)
    with pytest.raises(ValidationError, match="40-hex commit"):
        verdict(torchcell_commit="abc")
    with pytest.raises(ValidationError, match="ISO date"):
        verdict(decided_at="10/10/2026")


@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [
        (("pass",) * 5, "admissible"),
        (("pass", "pass", "pass", "pass", "gap"), "admissible_with_gaps"),
        (("pass", "pass", "pass", "unmeasured", "gap"), "pending"),
        (("pass", "fail") + ("unmeasured",) * 3, "refused"),
        (("pass", "blocked") + ("unmeasured",) * 3, "blocked:provision"),
    ],
)
def test_derive_outcome(outcomes: tuple[str, ...], expected: str) -> None:
    """The five derivations, first fail or blocked winning."""
    results = tuple(
        GateResult(
            gate=f"G{i + 1}",  # type: ignore[arg-type]
            outcome=o,  # type: ignore[arg-type]
            reason="r",
            issue=7 if o == "gap" else None,
            blocked_kind="provision" if o == "blocked" else None,
        )
        for i, o in enumerate(outcomes)
    )
    assert derive_outcome(results) == expected


def test_schema_names_the_verdict_and_its_required_fields() -> None:
    """The JSON schema an agent prompt carries."""
    schema = verdict_json_schema()
    assert schema["title"] == "CandidateVerdict"
    assert schema["required"] == [
        "citation_key",
        "table",
        "row_name",
        "gates",
        "zotero_item",
        "source_kind",
        "aggregation",
        "g2",
        "g3",
        "g4_key",
        "g4_value",
        "g5",
        "decided_at",
        "torchcell_commit",
    ]


def test_record_properties() -> None:
    """overlapping, margin, subsumed and intact compute from the fields."""
    key = G4KeyRecord(
        doi="10.1/x",
        pmid=None,
        accessions=(),
        hits=(
            G4KeyHit(key_kind="doi", key="10.1/x", where="a.py", relation="own"),
            G4KeyHit(key_kind="doi", key="10.1/x", where="b.py", relation="mentioned"),
            G4KeyHit(
                key_kind="pmid", key="1", where="SL", relation="aggregator_source"
            ),
        ),
        pmid_checked=True,
    )
    assert [h.where for h in key.overlapping] == ["SL"]
    match = G4ValueMatch(
        released_column="c",
        best_sample="s1",
        best_max_abs_diff=0.046,
        runner_up_sample="s2",
        runner_up_max_abs_diff=0.58,
        keys_compared=39,
    )
    assert match.margin == pytest.approx(0.534)
    lone = match.model_copy(
        update={"runner_up_sample": None, "runner_up_max_abs_diff": None}
    )
    assert lone.margin is None
    value = G4ValueRecord(
        store="s", store_commit=None, records_read=10, tolerance=0.05, matches=(match,)
    )
    assert value.subsumed is True
    assert value.model_copy(update={"tolerance": 0.01}).subsumed is False
    assert value.model_copy(update={"matches": ()}).subsumed is False
    base: dict[str, Any] = {
        "tree": "raw",
        "path": "data/x.xlsx",
        "role": "raw_data",
        "bytes": 1,
        "sha256_manifest": "a",
        "retrieval_method": None,
        "is_zip": True,
        "leading_archive_sha256": None,
    }
    assert InventoryFile(**base, sha256_disk="a", zip_ok=True).intact is True
    assert InventoryFile(**base, sha256_disk="a", zip_ok=False).intact is False
    assert InventoryFile(**base, sha256_disk="b", zip_ok=None).intact is False
