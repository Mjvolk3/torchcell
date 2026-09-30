# tests/torchcell/data/test_label_policy.py
# [[tests.torchcell.data.test_label_policy]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/data/test_label_policy
"""The read-time label policy: which measurement each label takes, and why.

2026.09.30 (Phase 14): hand-built ``LabelEntry`` lists and row dicts, no build. Expected
values, each derived in its test: ``standard_error`` = sd / sqrt(n) (0.2 / 2 = 0.1) with
the n-less and non-positive-sd fallbacks; inverse-variance replicates 1.00 (se 0.01) and
0.80 (se 0.10) give weights 10000 and 100, value 10080 / 10100 and se 1 / sqrt(10100);
one SD-less replicate drops the group to the plain mean with se sqrt(0.01^2) / 2 = 0.005;
Stouffer on two-sided p = 0.02 gives 2 sf(sqrt(2) isf(0.01)) = 0.0010020 at equal weight,
0.0101986 at weights 1 and 9 and 0.0398580 when the weight-1 screen disagrees; the
published trigenic identity 0.9497 - 0.5128 * 0.9875 + 0.0331 - 0.0220 = 0.45441 and
the measured one 0.455171. Findings pinned: truncating temperature keys, cross-source
combination of strain-matched doubles, every non-interaction type read as fitness, and
a ``temp=None`` key hiding ``temperature``.
"""

from __future__ import annotations

import json
import math
import os.path as osp
import random
from typing import Any

import pytest

from torchcell.data.label_policy import (
    CONVERTED_ZERO,
    LabelChoice,
    LabelEntry,
    LabelPolicy,
    entries_from_records,
    source_key,
    strain_token,
)
from torchcell.data.label_table import (
    TripleRoles,
    build_label_table,
    label_table_path,
    triple_roles,
)

KUZMIN_FIRST = LabelPolicy(name="kuzmin-first")
COSTANZO_FIRST = LabelPolicy(
    name="costanzo-first",
    fitness_precedence=[
        "costanzo2016@30",
        "costanzo2016@26",
        "kuzmin2018",
        "kuzmin2020",
        CONVERTED_ZERO,
    ],
    interaction_precedence=[
        "costanzo2016@30",
        "costanzo2016@26",
        "kuzmin2018",
        "kuzmin2020",
    ],
)
BUILD_029 = "/db/experiments/029-solid-growth-ko-001-ko-build"
NO_029 = not osp.exists(osp.join(BUILD_029, "processed", "lmdb"))
REASON = "the 029 no-merge build is not on this machine"


def _fit(source: str, value: float, **kw: Any) -> LabelEntry:
    return LabelEntry(source=source, label="fitness", value=value, **kw)


def _gi(value: float, p: float, source: str = "costanzo2016@30") -> LabelEntry:
    return LabelEntry(
        source=source, label="gene_interaction", value=value, p_value=p, n_samples=4
    )


def _pick(
    policy: LabelPolicy, entries: list[LabelEntry], label: str = "fitness"
) -> LabelChoice:
    """Select, asserting something was chosen so the type narrows for the assertions."""
    choice = policy.select(entries, label)
    assert choice is not None
    return choice


def _pick_double(
    policy: LabelPolicy, entries: list[LabelEntry], strain: str | None
) -> LabelChoice:
    choice = policy.select_double(entries, strain)
    assert choice is not None
    return choice


def _roles(perturbations: list[dict[str, Any]]) -> TripleRoles:
    roles = triple_roles(perturbations)
    assert roles is not None
    return roles


def _sample_indices(order: str, n: int, seed: int | None = None) -> list[int]:
    with open(osp.join(BUILD_029, "processed", "perturbation_count_index.json")) as f:
        by_order = json.load(f)
    pool: list[int] = [int(i) for i in by_order[order]]
    if seed is None:
        return pool[:n]
    random.seed(seed)
    return random.sample(pool, n)


def test_source_key_splits_costanzo_by_temperature_and_collapses_converted_zeros() -> (
    None
):
    assert source_key("DmfCostanzo2016Dataset", 30.0) == "costanzo2016@30"
    assert source_key("DmfCostanzo2016Dataset", 26.0) == "costanzo2016@26"
    assert source_key("TmfKuzmin2018Dataset", 26.0) == "kuzmin2018"
    assert source_key("GeneEssentialitySgdDataset", None) == CONVERTED_ZERO
    assert source_key("SynthLethalityYeastSynthLethDbDataset", 30.0) == CONVERTED_ZERO


def test_policy_id_is_stable_and_moves_with_any_rule() -> None:
    assert KUZMIN_FIRST.policy_id == LabelPolicy(name="kuzmin-first").policy_id
    assert KUZMIN_FIRST.policy_id != COSTANZO_FIRST.policy_id
    measured = KUZMIN_FIRST.model_copy(update={"source_convention": "measured"})
    assert measured.policy_id != KUZMIN_FIRST.policy_id


def test_precedence_decides_which_screen_supplies_the_value() -> None:
    entries = [_fit("costanzo2016@30", 0.90), _fit("kuzmin2018", 0.80)]
    assert _pick(KUZMIN_FIRST, entries).source == "kuzmin2018"
    assert _pick(COSTANZO_FIRST, entries).source == "costanzo2016@30"


def test_an_unlisted_source_is_never_chosen() -> None:
    policy = LabelPolicy(name="kuzmin-only", fitness_precedence=["kuzmin2018"])
    assert policy.select([_fit("costanzo2016@30", 0.9)], "fitness") is None


def test_the_converted_zero_yields_to_any_measurement_but_stands_alone() -> None:
    chosen = _pick(
        KUZMIN_FIRST, [_fit(CONVERTED_ZERO, 0.0), _fit("costanzo2016@30", 0.62)]
    )
    assert chosen.value == pytest.approx(0.62)
    assert chosen.source == "costanzo2016@30"
    assert chosen.n_entries_available == 2

    only_zero = _pick(KUZMIN_FIRST, [_fit(CONVERTED_ZERO, 0.0)])
    assert only_zero.value == 0.0
    assert only_zero.source == CONVERTED_ZERO


def test_a_policy_can_be_told_to_average_the_zero_in_instead() -> None:
    lenient = KUZMIN_FIRST.model_copy(
        update={
            "converted_zero_only_without_measurement": False,
            "fitness_precedence": [CONVERTED_ZERO, "costanzo2016@30"],
        }
    )
    chosen = _pick(lenient, [_fit(CONVERTED_ZERO, 0.0), _fit("costanzo2016@30", 0.62)])
    assert chosen.source == CONVERTED_ZERO


def test_same_source_replicates_combine_by_inverse_variance() -> None:
    precise = _fit("costanzo2016@30", 1.00, sd=0.02, n_samples=4)
    noisy = _fit("costanzo2016@30", 0.80, sd=0.20, n_samples=4)
    chosen = _pick(COSTANZO_FIRST, [precise, noisy])
    assert chosen.n_entries_combined == 2
    # se = sd / sqrt(n): 0.02 / 2 = 0.01 and 0.20 / 2 = 0.10, so the weights 1 / se^2
    # are 10000 and 100; value = (10000 * 1.00 + 100 * 0.80) / 10100 = 10080 / 10100
    assert chosen.value == pytest.approx(10080 / 10100, rel=1e-12)
    # combined se = 1 / sqrt(10100), below the better single measurement's 0.01
    assert chosen.standard_error == pytest.approx(1 / math.sqrt(10100), rel=1e-12)
    # the reported sd re-inflates the se by sqrt of the pooled n = 4 + 4
    assert chosen.sd == pytest.approx(math.sqrt(8) / math.sqrt(10100), rel=1e-12)
    assert chosen.n_samples == 8


def test_plain_mean_is_available_and_sits_midway() -> None:
    policy = COSTANZO_FIRST.model_copy(update={"replicate_combination": "mean"})
    chosen = _pick(
        policy,
        [
            _fit("costanzo2016@30", 1.00, sd=0.02, n_samples=4),
            _fit("costanzo2016@30", 0.80, sd=0.20, n_samples=4),
        ],
    )
    assert chosen.value == pytest.approx(0.90)


def test_stouffer_sharpens_agreement_and_dulls_disagreement() -> None:
    agree = _pick(
        COSTANZO_FIRST, [_gi(-0.2, 0.02), _gi(-0.2, 0.02)], "gene_interaction"
    )
    assert agree.p_value is not None
    assert agree.p_value < 0.02

    disagree = _pick(
        COSTANZO_FIRST, [_gi(-0.2, 0.02), _gi(0.2, 0.02)], "gene_interaction"
    )
    assert disagree.p_value == pytest.approx(1.0, abs=1e-6)


def test_a_single_p_value_passes_through_unchanged() -> None:
    entry = _gi(-0.3, 0.004, source="kuzmin2018")
    assert _pick(KUZMIN_FIRST, [entry], "gene_interaction").p_value == pytest.approx(
        0.004
    )


def test_a_triple_prefers_the_double_its_own_query_strain_measured() -> None:
    query_strain = "YAR002W+YML107C_tm2550"
    entries = [
        _fit("kuzmin2018", 0.5128, strain_id=query_strain),
        _fit("costanzo2016@30", 0.8900, strain_id="YAR002W_dma37"),
    ]
    assert _pick_double(KUZMIN_FIRST, entries, query_strain).value == pytest.approx(
        0.5128
    )

    # absent the strain-matched record it falls through to the pair's own screen, which
    # is what a build without the query-strain fitness is forced into
    only_screen = [_fit("costanzo2016@30", 0.8900, strain_id="YAR002W_dma37")]
    assert _pick_double(KUZMIN_FIRST, only_screen, query_strain).value == pytest.approx(
        0.89
    )

    off = KUZMIN_FIRST.model_copy(update={"prefer_strain_matched_double": False})
    assert _pick_double(off, entries, query_strain).source == "kuzmin2018"


def test_the_tm_token_joins_a_triple_to_its_query_strain_across_both_years() -> None:
    """2018 writes the whole pair on both query perturbations, 2020 writes it on one."""
    assert strain_token("YKL010C+YMR067C_tm2424") == "tm2424"
    assert strain_token("YDR003W_tm1501") == "tm1501"
    assert strain_token("YDR186C_dma966") is None
    assert strain_token(None) is None

    # the triple names tm1501 in the 2020 shape while the query strain's own fitness
    # record carries the full pair form, so an exact string match misses and a token does not
    entries = [
        _fit("kuzmin2020", 0.71, strain_id="YAL015C+YOL043C_tm1501"),
        _fit("costanzo2016@30", 0.93, strain_id="YAL015C_dma12"),
    ]
    assert _pick_double(KUZMIN_FIRST, entries, "YDR003W_tm1501").value == pytest.approx(
        0.71
    )


def test_the_triples_own_kuzmin_year_is_promoted_above_the_other() -> None:
    entries = [_fit("kuzmin2018", 0.7), _fit("kuzmin2020", 0.6)]
    assert _pick(KUZMIN_FIRST, entries).source == "kuzmin2018"
    assert (
        _pick(KUZMIN_FIRST.promoted_for_year("kuzmin2020"), entries).source
        == "kuzmin2020"
    )
    # a non-Kuzmin year leaves both precedences exactly as they were
    unchanged = KUZMIN_FIRST.promoted_for_year("costanzo2016@30")
    assert unchanged.fitness_precedence == KUZMIN_FIRST.fitness_precedence
    assert unchanged.interaction_precedence == KUZMIN_FIRST.interaction_precedence


def test_the_published_convention_is_the_identity_the_released_scores_used() -> None:
    f_triple, f_query_double, f_array = 0.9497, 0.5128, 0.9875
    eps_ik, eps_jk = -0.0331, 0.0220
    published = KUZMIN_FIRST.trigenic_tau(
        f_triple, f_query_double, f_array, eps_ik, eps_jk
    )
    # 0.9497 - 0.5128 * 0.9875 - (-0.0331) - 0.0220 = 0.9497 - 0.50639 + 0.0331 - 0.0220
    assert published == pytest.approx(0.45441, abs=1e-12)

    measured_policy = KUZMIN_FIRST.model_copy(update={"source_convention": "measured"})
    measured = measured_policy.trigenic_tau(
        f_triple,
        f_query_double,
        f_array,
        eps_ik,
        eps_jk,
        f_query_single_i=0.83,
        f_query_single_j=0.91,
    )
    # the control terms are weighted by the OTHER query single: eps_ik by f_j = 0.91 and
    # eps_jk by f_i = 0.83, so 0.44331 + 0.0331 * 0.91 - 0.0220 * 0.83 = 0.455171
    assert measured == pytest.approx(0.455171, abs=1e-12)


def test_the_measured_convention_refuses_to_guess_a_missing_query_single() -> None:
    measured_policy = KUZMIN_FIRST.model_copy(update={"source_convention": "measured"})
    with pytest.raises(ValueError, match="needs both query single fitnesses"):
        measured_policy.trigenic_tau(0.95, 0.51, 0.99, -0.03, 0.02)


def test_a_repeated_source_in_a_precedence_is_refused() -> None:
    with pytest.raises(ValueError, match="repeated source"):
        LabelPolicy(name="bad", fitness_precedence=["kuzmin2018", "kuzmin2018"])


def test_triple_roles_recovers_the_array_gene_and_query_pair() -> None:
    roles_2018 = _roles(
        [
            {"systematic_gene_name": "YJL098W", "strain_id": "YJL098W_dma2503"},
            {"systematic_gene_name": "YKL010C", "strain_id": "YKL010C+YMR067C_tm2424"},
            {"systematic_gene_name": "YMR067C", "strain_id": "YKL010C+YMR067C_tm2424"},
        ]
    )
    assert roles_2018.array_gene == "YJL098W"
    assert roles_2018.query_genes == ("YKL010C", "YMR067C")
    assert roles_2018.query_strain_id == "YKL010C+YMR067C_tm2424"

    roles_2020 = _roles(
        [
            {"systematic_gene_name": "YBR005W", "strain_id": "YBR005W"},
            {"systematic_gene_name": "YDR003W", "strain_id": "YDR003W_tm1501"},
            {"systematic_gene_name": "YDR186C", "strain_id": "YDR186C_dma966"},
        ]
    )
    assert roles_2020.array_gene == "YDR186C"
    assert roles_2020.query_genes == ("YBR005W", "YDR003W")

    # a record whose source never named its strains has no recoverable roles
    assert (
        triple_roles([{"systematic_gene_name": g, "strain_id": ""} for g in "ABC"])
        is None
    )
    assert triple_roles([{"systematic_gene_name": "A", "strain_id": "A_dma1"}]) is None


def test_source_key_names_kuzmin2020_and_refuses_what_it_cannot_rank() -> None:
    assert source_key("TmiKuzmin2020Dataset", 30.0) == "kuzmin2020"
    # the converted-zero tag is checked first, so it wins over a Kuzmin substring
    assert source_key("GeneEssentialitySgdKuzmin2018", 30.0) == CONVERTED_ZERO
    with pytest.raises(
        ValueError, match=r"^DmfCostanzo2016Dataset entry carries no temperature$"
    ):
        source_key("DmfCostanzo2016Dataset", None)
    with pytest.raises(
        ValueError, match=r"^no source key for dataset 'SmfOhya2005Dataset'$"
    ):
        source_key("SmfOhya2005Dataset", 30.0)


def test_source_key_truncates_a_fractional_costanzo_temperature() -> None:
    """Finding: ``int(temperature)`` truncates, so 29.9 lands in the 29 bin, not 30.

    ``source_key`` (label_policy.py:101) formats ``int(temperature)``; a temperature
    stored as 29.9 would silently become an unranked ``costanzo2016@29`` that no default
    precedence lists. Pinned until the key rounds or refuses non-integral temperatures.
    """
    assert source_key("DmfCostanzo2016Dataset", 29.9) == "costanzo2016@29"
    assert source_key("DmfCostanzo2016Dataset", 30.0) == "costanzo2016@30"


def test_standard_error_is_sd_over_root_n_with_each_fallback() -> None:
    # 0.2 / sqrt(4) = 0.1
    assert _fit("kuzmin2018", 1.0, sd=0.2, n_samples=4).standard_error == 0.1
    # no n, or an n below 1, leaves the SD itself as the error
    assert _fit("kuzmin2018", 1.0, sd=0.2).standard_error == 0.2
    assert _fit("kuzmin2018", 1.0, sd=0.2, n_samples=0).standard_error == 0.2
    # a zero, negative, non-finite or absent SD gives no error at all
    assert _fit("kuzmin2018", 1.0, sd=0.0, n_samples=4).standard_error is None
    assert _fit("kuzmin2018", 1.0, sd=-0.1, n_samples=4).standard_error is None
    assert _fit("kuzmin2018", 1.0, sd=math.nan, n_samples=4).standard_error is None
    assert _fit("kuzmin2018", 1.0, n_samples=4).standard_error is None


def test_select_reports_none_for_an_absent_label_and_refuses_an_unknown_one() -> None:
    # only fitness entries: there is nothing to choose for the interaction label
    assert KUZMIN_FIRST.select([_fit("kuzmin2018", 0.7)], "gene_interaction") is None
    growth = LabelEntry(source="kuzmin2018", label="growth", value=1.0)
    with pytest.raises(ValueError, match=r"^no precedence for label 'growth'$"):
        KUZMIN_FIRST.select([growth], "growth")


def test_precedence_walks_past_absent_sources_and_counts_only_the_label() -> None:
    """kuzmin2018 is absent, so the next listed source present (kuzmin2020) wins.

    ``n_entries_available`` counts the two fitness entries, not the interaction entry
    of the same record.
    """
    entries = [
        _fit("costanzo2016@26", 0.91),
        _fit("kuzmin2020", 0.74, sd=0.04, n_samples=16, p_value=0.2),
        _gi(-0.12, 0.01),
    ]
    chosen = _pick(KUZMIN_FIRST, entries)
    assert chosen == LabelChoice(
        value=0.74,
        source="kuzmin2020",
        sd=0.04,
        standard_error=0.01,
        n_samples=16,
        p_value=0.2,
        n_entries_combined=1,
        n_entries_available=2,
    )


def test_the_zero_is_dropped_before_the_count_is_taken_only_for_choosing() -> None:
    """With the strict rule a refused zero still counts as available.

    ``n_available`` is taken before the converted zeros are filtered out, so the choice
    reports 3 available entries while combining only the two measured replicates.
    """
    entries = [
        _fit(CONVERTED_ZERO, 0.0),
        _fit("costanzo2016@30", 0.6),
        _fit("costanzo2016@30", 0.8),
    ]
    chosen = _pick(KUZMIN_FIRST, entries)
    # neither replicate has an SD, so inverse variance falls back to the plain mean
    assert chosen.value == pytest.approx(0.7, abs=1e-12)
    assert chosen.source == "costanzo2016@30"
    assert chosen.n_entries_combined == 2
    assert chosen.n_entries_available == 3
    assert chosen.standard_error is None
    assert chosen.sd is None
    assert chosen.n_samples is None


def test_inverse_variance_falls_back_to_the_mean_when_one_replicate_lacks_an_sd() -> (
    None
):
    """One replicate without an SD disables weighting for the whole group.

    value = (1.00 + 0.80) / 2 = 0.90; se = sqrt(0.01^2) / 2 = 0.005 (the missing se is
    dropped from the root sum but the divisor is still the 2 entries); n pooled = 4 (the
    second has none), so sd = 0.005 * sqrt(4) = 0.01.
    """
    chosen = _pick(
        COSTANZO_FIRST,
        [
            _fit("costanzo2016@30", 1.00, sd=0.02, n_samples=4),
            _fit("costanzo2016@30", 0.80),
        ],
    )
    assert chosen.value == pytest.approx(0.90, abs=1e-12)
    assert chosen.standard_error == pytest.approx(0.005, abs=1e-15)
    assert chosen.sd == pytest.approx(0.01, abs=1e-15)
    assert chosen.n_samples == 4


def test_first_takes_one_replicate_and_reports_the_rest_as_available() -> None:
    policy = COSTANZO_FIRST.model_copy(update={"replicate_combination": "first"})
    chosen = _pick(
        policy,
        [
            _fit("costanzo2016@30", 1.00, sd=0.02, n_samples=4, p_value=0.3),
            _fit("costanzo2016@30", 0.80, sd=0.20, n_samples=4, p_value=0.01),
        ],
    )
    # the first entry verbatim, its own p (no Stouffer), its own se 0.02 / 2
    assert chosen == LabelChoice(
        value=1.00,
        source="costanzo2016@30",
        sd=0.02,
        standard_error=0.01,
        n_samples=4,
        p_value=0.3,
        n_entries_combined=1,
        n_entries_available=2,
    )


def test_min_and_none_p_combinations() -> None:
    entries = [_gi(-0.2, 0.03), _gi(-0.1, 0.01), _gi(-0.3, 0.5)]
    as_min = COSTANZO_FIRST.model_copy(update={"p_combination": "min"})
    assert _pick(as_min, entries, "gene_interaction").p_value == 0.01
    # min over entries that carry no p at all is None, not 0 or 1
    no_p = [
        LabelEntry(source="costanzo2016@30", label="gene_interaction", value=v)
        for v in (-0.2, -0.1)
    ]
    assert _pick(as_min, no_p, "gene_interaction").p_value is None
    as_none = COSTANZO_FIRST.model_copy(update={"p_combination": "none"})
    assert _pick(as_none, entries, "gene_interaction").p_value is None


def test_stouffer_closed_form_with_equal_and_unequal_weights() -> None:
    """Two-sided p = 0.02 is one-sided 0.01, z = isf(0.01) = 2.326347874...

    Equal weights (n 4 and 4): z_c = (4z + 4z) / sqrt(32) = sqrt(2) z, so
    p = 2 sf(sqrt(2) z) = 0.0010020422. Weights n 1 and 9, same sign:
    z_c = 10z / sqrt(82), p = 0.0101986142. Weights 1 and 9 with opposite signs:
    |z - 9z| / sqrt(82) = 8z / sqrt(82), p = 0.0398580337, the heavier screen's sign.
    """
    from scipy import stats

    z = float(stats.norm.isf(0.01))
    equal = _pick(
        COSTANZO_FIRST, [_gi(-0.2, 0.02), _gi(-0.3, 0.02)], "gene_interaction"
    )
    assert equal.p_value == pytest.approx(0.001002042203701685, rel=1e-9)
    assert equal.p_value == pytest.approx(
        2 * stats.norm.sf(math.sqrt(2) * z), rel=1e-12
    )

    def _w(value: float, n: int) -> LabelEntry:
        return LabelEntry(
            source="costanzo2016@30",
            label="gene_interaction",
            value=value,
            p_value=0.02,
            n_samples=n,
        )

    unequal = _pick(COSTANZO_FIRST, [_w(0.1, 1), _w(0.2, 9)], "gene_interaction")
    assert unequal.p_value == pytest.approx(0.010198614178960919, rel=1e-9)
    opposed = _pick(COSTANZO_FIRST, [_w(0.1, 1), _w(-0.2, 9)], "gene_interaction")
    assert opposed.p_value == pytest.approx(0.039858033702562454, rel=1e-9)


def test_stouffer_skips_unusable_p_values_and_passes_a_lone_survivor() -> None:
    """A missing p, a p of 0 and a p above 1 are dropped; one survivor passes as is."""
    entries = [
        LabelEntry(source="costanzo2016@30", label="gene_interaction", value=-0.1),
        _gi(-0.2, 0.0),
        _gi(-0.2, 1.5),
        _gi(-0.3, 0.04),
    ]
    chosen = _pick(COSTANZO_FIRST, entries, "gene_interaction")
    assert chosen.n_entries_combined == 4
    assert chosen.p_value == 0.04
    # every p unusable: no combined p rather than an invented one
    none_usable = _pick(
        COSTANZO_FIRST, [_gi(-0.2, 0.0), _gi(-0.2, 1.5)], "gene_interaction"
    )
    assert none_usable.p_value is None


def test_a_strain_matched_double_combines_every_match_and_counts_only_them() -> None:
    """Finding: strain-matched entries are combined as replicates across sources.

    ``select_double`` (label_policy.py:374-384) passes every strain-matched entry to
    ``_combine``, whose docstring says it combines "entries that share a source". Two
    matches from kuzmin2018 and kuzmin2020 average to (0.50 + 0.70) / 2 = 0.60 under
    the source of the first, and ``n_entries_available`` counts the 2 matches, not the
    3 fitness entries on the record. Pinned until mixed-source matches are ranked.
    """
    entries = [
        _fit("kuzmin2018", 0.50, strain_id="YAL015C+YOL043C_tm1501"),
        _fit("kuzmin2020", 0.70, strain_id="YDR003W_tm1501"),
        _fit("costanzo2016@30", 0.93, strain_id="YAL015C_dma12"),
    ]
    chosen = _pick_double(KUZMIN_FIRST, entries, "YAL015C+YOL043C_tm1501")
    assert chosen.value == pytest.approx(0.60, abs=1e-12)
    assert chosen.source == "kuzmin2018"
    assert chosen.n_entries_combined == 2
    assert chosen.n_entries_available == 2


def test_select_double_without_a_query_strain_is_plain_selection() -> None:
    entries = [
        _fit("costanzo2016@30", 0.93, strain_id="YAL015C+YOL043C_tm1501"),
        _fit("kuzmin2018", 0.50),
    ]
    # no query strain: the strain match is skipped and precedence picks kuzmin2018
    assert _pick_double(KUZMIN_FIRST, entries, None).value == 0.50
    # with the strain, the matched costanzo entry wins over the higher-ranked source
    assert _pick_double(KUZMIN_FIRST, entries, "YAL015C+YOL043C_tm1501").value == 0.93


def test_promotion_moves_the_year_to_the_front_only_where_it_is_listed() -> None:
    policy = LabelPolicy(
        name="costanzo-fitness",
        fitness_precedence=["costanzo2016@30", CONVERTED_ZERO],
        interaction_precedence=["costanzo2016@30", "kuzmin2018", "kuzmin2020"],
    )
    promoted = policy.promoted_for_year("kuzmin2020")
    # absent from fitness: untouched rather than inserted
    assert promoted.fitness_precedence == ["costanzo2016@30", CONVERTED_ZERO]
    assert promoted.interaction_precedence == [
        "kuzmin2020",
        "costanzo2016@30",
        "kuzmin2018",
    ]
    # switched off, even a Kuzmin year leaves the order alone
    off = policy.model_copy(update={"same_year_kuzmin_first": False})
    assert off.promoted_for_year("kuzmin2020").interaction_precedence == [
        "costanzo2016@30",
        "kuzmin2018",
        "kuzmin2020",
    ]


def test_entries_from_records_normalizes_both_key_spellings() -> None:
    """Short keys (dataset, exp_type, temp, p) and long keys (dataset_name,
    experiment_type, temperature, p_value) both parse; a NaN sd, n or p becomes None;
    a missing or NaN value drops the row.
    """
    rows: list[dict[str, Any]] = [
        {
            "dataset": "SmfKuzmin2018Dataset",
            "exp_type": "fitness",
            "value": 0.8,
            "temp": 30,
            "sd": 0.1,
            "n_samples": 12,
            "p": 0.2,
            "strain_id": "YAL001C_tm1",
        },
        {
            "dataset_name": "DmfCostanzo2016Dataset",
            "experiment_type": "gene_interaction",
            "value": -0.1,
            "temperature": 26,
            "sd": math.nan,
            "n_samples": math.nan,
            "p_value": 0.03,
        },
        {
            "dataset": "GeneEssentialitySgdDataset",
            "exp_type": "fitness",
            "value": 0,
            "p": math.nan,
        },
        {"dataset": "SmfKuzmin2018Dataset", "exp_type": "fitness", "value": None},
        {"dataset": "SmfKuzmin2018Dataset", "exp_type": "fitness", "value": math.nan},
    ]
    assert entries_from_records(rows) == [
        LabelEntry(
            source="kuzmin2018",
            label="fitness",
            value=0.8,
            sd=0.1,
            n_samples=12,
            p_value=0.2,
            strain_id="YAL001C_tm1",
        ),
        LabelEntry(
            source="costanzo2016@26", label="gene_interaction", value=-0.1, p_value=0.03
        ),
        LabelEntry(source=CONVERTED_ZERO, label="fitness", value=0.0),
    ]


def test_entries_from_records_reads_any_non_interaction_type_as_fitness() -> None:
    """Finding: every experiment type without "interaction" in it becomes fitness.

    ``label = "gene_interaction" if "interaction" in raw_type else "fitness"``
    (label_policy.py:446), so a morphology entry is ranked as a fitness value. Pinned
    until the parser refuses an unknown experiment type.
    """
    rows: list[dict[str, Any]] = [
        {"dataset": "SmfKuzmin2018Dataset", "exp_type": "morphology", "value": 2.5}
    ]
    assert entries_from_records(rows) == [
        LabelEntry(source="kuzmin2018", label="fitness", value=2.5)
    ]


def test_entries_from_records_short_key_present_as_none_hides_the_long_key() -> None:
    """Finding: a ``temp`` key holding None hides a populated ``temperature``.

    ``r.get("temp", r.get("temperature"))`` (label_policy.py:447) falls back only
    when ``temp`` is ABSENT, so a Costanzo row carrying ``temp=None`` and
    ``temperature=30`` is refused as having no temperature. Pinned until the lookup
    treats None as absent.
    """
    rows: list[dict[str, Any]] = [
        {
            "dataset": "DmfCostanzo2016Dataset",
            "exp_type": "fitness",
            "value": 0.9,
            "temp": None,
            "temperature": 30,
        }
    ]
    with pytest.raises(
        ValueError, match=r"^DmfCostanzo2016Dataset entry carries no temperature$"
    ):
        entries_from_records(rows)


# ------------------------------------------------------------------ integration
@pytest.mark.data
@pytest.mark.skipif(NO_029, reason=REASON)
def test_the_policy_reads_a_real_no_merge_build() -> None:
    """029 is the only build today that keeps every entry, so it is the real test."""
    sample = [i for order in ("1", "2", "3") for i in _sample_indices(order, 200)]
    table = build_label_table(
        KUZMIN_FIRST, BUILD_029, indices=sample, workers=8, cache=False
    )

    assert len(table) == len(sample)
    assert table["fitness"].notna().all(), (
        "every record in a growth build carries a fitness"
    )
    assert set(table["fitness_source"].dropna()) <= set(KUZMIN_FIRST.fitness_precedence)
    # a no-merge build keeps several entries per genotype, which is the whole point
    assert table["fitness_n_entries"].max() > 1


@pytest.mark.data
@pytest.mark.skipif(NO_029, reason=REASON)
def test_two_policies_disagree_on_the_same_records_and_cache_apart() -> None:
    sample = _sample_indices("2", 400)
    a = build_label_table(
        KUZMIN_FIRST, BUILD_029, indices=sample, workers=8, cache=False
    )
    b = build_label_table(
        COSTANZO_FIRST, BUILD_029, indices=sample, workers=8, cache=False
    )

    merged = a.merge(b, on="index", suffixes=("_k", "_c"))
    assert ((merged["fitness_k"] - merged["fitness_c"]).abs() > 1e-9).any(), (
        "the two conventions must part company on a real pool"
    )
    assert label_table_path(BUILD_029, KUZMIN_FIRST) != label_table_path(
        BUILD_029, COSTANZO_FIRST
    )


@pytest.mark.data
@pytest.mark.skipif(NO_029, reason=REASON)
def test_roles_resolve_on_every_sampled_triple_of_a_real_build() -> None:
    import lmdb

    sample = _sample_indices("3", 500, seed=0)
    env = lmdb.open(osp.join(BUILD_029, "processed", "lmdb"), readonly=True, lock=False)
    resolved = 0
    with env.begin() as txn:
        for i in sample:
            for item in json.loads(txn.get(str(i).encode())):
                experiment = item["experiment"]
                if experiment["experiment_type"] != "fitness":
                    continue
                roles = _roles(experiment["genotype"]["perturbations"])
                assert roles.array_gene not in roles.query_genes
                resolved += 1
                break
    assert resolved == len(sample)
