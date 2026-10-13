# tests/torchcell/candidates/test_verdict.py
# [[tests.torchcell.candidates.test_verdict]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/candidates/test_verdict.py
"""The verdict models: field rules, gate order, the stop rule, and outcome derivation."""

from typing import Any

import pytest
from pydantic import ValidationError

from tests.torchcell.candidates._builders import (
    admissible_verdict,
    aggregation,
    compendium_paper,
    cross_check,
    evidence,
    gate,
    source,
    unmeasured_tail,
    verdict,
)
from torchcell.candidates.verdict import (
    AggregationRecord,
    CompendiumDerivation,
    G2Host,
    G2Record,
    G4KeyHit,
    G4KeyRecord,
    G4ValueMatch,
    G4ValueRecord,
    GateResult,
    InventoryFile,
    SourceCrossCheck,
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


# --------------------------------------------------------------------------- #
# Aggregation decomposition (plan decision 16)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("value", "tolerance", "statistic", "passes"),
    [
        (0.0004, 0.0005, "max_abs_diff", True),
        (0.0006, 0.0005, "max_abs_diff", False),
        (0.97, 0.95, "min_pearson_r", True),
        (0.94, 0.95, "min_r_squared", False),
        (0.98, None, "min_pearson_r", None),
    ],
)
def test_cross_check_passes_by_statistic_direction(
    value: float, tolerance: float | None, statistic: str, passes: bool | None
) -> None:
    """A difference passes at or below its bound, a correlation at or above; no bound, no verdict."""
    check = SourceCrossCheck(
        statistic=statistic,  # type: ignore[arg-type]
        value=value,
        tolerance=tolerance,
        tolerance_basis="b",
        n_samples=1,
        n_values=1,
        against="a",
    )
    assert check.passes is passes


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"role": "compendium_paper"},
            "the compendium's own paper has no release apart from it",
        ),
        ({"own_release_kind": "none"}, "own_release disagrees with own_release_kind"),
        (
            {"own_release_url": None},
            "own_release_url and own_release_access are required for",
        ),
        ({"citation_key": None}, "mirrored requires, and only with, citation_key"),
        (
            {"loaded_as_own_dataset": True},
            "loaded_as_own_dataset requires, and only with, own_dataset_class",
        ),
        ({"n_samples_served": 5}, "5 served of 4 compendium samples"),
    ],
)
def test_source_decomposition_rules(overrides: dict[str, Any], message: str) -> None:
    """Every field pair that must agree is checked."""
    with pytest.raises(ValidationError, match=message):
        source(**overrides)


def test_cross_check_needs_an_own_release() -> None:
    """Nothing to compare against when the compendium is the only artifact."""
    with pytest.raises(ValidationError, match="a cross-check needs an own release"):
        compendium_paper(cross_check=cross_check())


def test_loadable_and_compendium_only() -> None:
    """Loadable: own release, not the compendium's, not loaded; only-artifact otherwise."""
    loadable = source()
    walled = source("doi:10.1/w", own_release_access="walled")
    loaded = source(
        "doi:10.1/l",
        loaded_as_own_dataset=True,
        own_dataset_class="LoadedDataset",
        cross_check=cross_check(),
    )
    paper = compendium_paper()
    assert [s.loadable_from_own_release for s in (loadable, walled, loaded, paper)] == [
        True,
        True,
        False,
        False,
    ]
    assert [s.compendium_only for s in (loadable, walled, loaded, paper)] == [
        False,
        False,
        False,
        True,
    ]


def _decomposed(n: int, mirrored: int | None, *sources: Any) -> AggregationRecord:
    return AggregationRecord(
        n_source_studies=n,
        attribution_field="Publication",
        value_origin="re_measured",
        n_sources_mirrored=mirrored,
        net_new_vs_served=None,
        evidence=(evidence(),),
        sources=sources,
    )


def test_decomposition_is_the_study_count() -> None:
    """One study in two served classes counts once; the mirrored count must match."""
    a1 = source()
    a2 = source(dataset_class="OtherDataset")
    b = source("doi:10.1/b", citation_key=None, mirrored=False)
    record = _decomposed(2, 1, a1, a2, b)
    assert record.studies() == {"doi:10.1/a": (a1, a2), "doi:10.1/b": (b,)}
    assert aggregation().studies() == {}
    with pytest.raises(ValidationError, match="the decomposition names 2 studies"):
        _decomposed(3, None, a1, b)
    with pytest.raises(ValidationError, match="but 1 decomposed studies are mirrored"):
        _decomposed(2, 2, a1, b)


def test_decomposition_refuses_repeats_conflicts_and_empty() -> None:
    """A (label, class) pair once; one label, one identity; an empty tuple is not None."""
    with pytest.raises(ValidationError, match="appears twice in the decomposition"):
        _decomposed(1, None, source(), source())
    with pytest.raises(ValidationError, match="entries disagree on role"):
        _decomposed(
            1, None, source(), source(dataset_class="OtherDataset", doi="10.1/other")
        )
    with pytest.raises(ValidationError, match="an empty decomposition"):
        _decomposed(1, None)


def test_g4_unmeasured_carries_a_key_record_only_for_a_pending_shared_key() -> None:
    """#913: the one record-bearing unmeasured G4 is a shared key awaiting comparison."""
    from torchcell.candidates.verdict import (
        G4KeyHit,
        G4KeyRecord,
        G4ValueRecord,
        GateResult,
        _check_g4,
    )

    pending = GateResult(gate="G4", outcome="unmeasured", reason="compare first")
    shared = G4KeyRecord(
        doi=None,
        pmid=None,
        accessions=("PRJNA1",),
        hits=(
            G4KeyHit(
                key_kind="accession",
                key="PRJNA1",
                where="m.py",
                relation="primary_of_other",
            ),
        ),
        pmid_checked=False,
    )
    _check_g4(pending, shared, None)
    clean = shared.model_copy(update={"hits": ()})
    with pytest.raises(ValueError, match="awaits its content comparison"):
        _check_g4(pending, clean, None)
    value = G4ValueRecord(
        store="s", store_commit=None, records_read=0, tolerance=0.01, matches=()
    )
    with pytest.raises(ValueError, match="awaits its content comparison"):
        _check_g4(pending, shared, value)
    with pytest.raises(ValueError, match="without its record"):
        _check_g4(GateResult(gate="G4", outcome="pass", reason="r"), None, None)


def test_derived_from_compendium_is_optional_typed_and_decides_nothing() -> None:
    """Absent by default (old verdicts read unchanged); when set, both fields are needed."""
    plain = admissible_verdict()
    assert plain.derived_from_compendium is None
    assert "derived_from_compendium" not in verdict_json_schema()["required"]
    flagged = admissible_verdict(
        derived_from_compendium=CompendiumDerivation(
            citation_key="compendiumKey2021", note="re-keyed and filtered"
        )
    )
    assert flagged.outcome == plain.outcome == "admissible"
    dumped = flagged.model_dump()
    assert dumped["derived_from_compendium"] == {
        "citation_key": "compendiumKey2021",
        "note": "re-keyed and filtered",
    }
    assert type(flagged).model_validate(dumped) == flagged
    with pytest.raises(ValidationError, match="citation_key"):
        CompendiumDerivation(citation_key="", note="re-keyed")
    with pytest.raises(ValidationError, match="note"):
        CompendiumDerivation(citation_key="compendiumKey2021", note="")
