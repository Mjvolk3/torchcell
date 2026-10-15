# tests/torchcell/data/test_label_policy.py
"""The read-time label policy: which measurement each label takes, and why."""

from __future__ import annotations

import json
import os.path as osp
import random
from typing import Any

import pytest

from torchcell.data.label_policy import (
    CONVERTED_ZERO,
    LabelChoice,
    LabelEntry,
    LabelPolicy,
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
    # the precise entry carries a hundred times the weight, so the mean sits beside it
    assert chosen.value == pytest.approx(0.998, abs=2e-3)
    precise_se = precise.standard_error
    assert precise_se is not None
    assert chosen.standard_error is not None
    # combining cannot be less certain than the better single measurement
    assert chosen.standard_error < precise_se
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
    assert KUZMIN_FIRST.promoted_for_year("costanzo2016@30") is KUZMIN_FIRST


def test_the_published_convention_is_the_identity_the_released_scores_used() -> None:
    f_triple, f_query_double, f_array = 0.9497, 0.5128, 0.9875
    eps_ik, eps_jk = -0.0331, 0.0220
    published = KUZMIN_FIRST.trigenic_tau(
        f_triple, f_query_double, f_array, eps_ik, eps_jk
    )
    assert published == pytest.approx(
        f_triple - f_query_double * f_array - eps_ik - eps_jk
    )

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
    # the two conventions are different numbers by construction, not two estimates of one
    assert measured != pytest.approx(published)


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


# ------------------------------------------------------------------ integration
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
