# experiments/027-betaxanthin-metabolic/scripts/round2_summary_2026_09_12.py
# [[experiments.027-betaxanthin-metabolic.scripts.round2_summary_2026_09_12]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/round2_summary_2026_09_12.py

"""One table per question for the 2026-09-11 to 2026-09-12 round-2 metabolic-module run.

Reads only committed result files and writes ``results/round2_summary_2026_09_12.json``
plus a markdown twin whose tables are pasted verbatim into the round-2 note
([[experiments.027-betaxanthin-metabolic.round2-2026-09-12]]). Nothing here is computed
from a log or from a slurm queue; every number is read from a results JSON that its own
stream's script wrote.

Sources:

* ``026-metabolism-flux/results/thermo_curation_closure.json``
  (thermo_curation_closure_summary.py)
* ``026-metabolism-flux/results/reaction_thermo_coverage_final.json``
  (thermo_pipeline_yeast9.py --curation-dir)
* ``026-metabolism-flux/results/catpred_table_impact.json`` (catpred_table_impact.py)
* ``026-metabolism-flux/results/catpred_training_overlap.json``
  (catpred_training_overlap.py)
* ``026-metabolism-flux/results/catpred_uncertainty.json`` (catpred_uncertainty.py)
* ``026-metabolism-flux/results/within_class_plausibility_with_catpred.json``
  (within_class_plausibility.py --suffix _with_catpred)
* ``027-betaxanthin-metabolic/results/wave7_parity_summary.json``
  (analyze_wave7_parity.py)
* ``027-betaxanthin-metabolic/results/wave7_fluxforce_summary.json``
  (analyze_wave7_fluxforce.py)
* ``027-betaxanthin-metabolic/results/wave7_aa_summary.json`` (analyze_wave7_aa.py)
* ``027-betaxanthin-metabolic/results/wave7_kcat_table_verification.json``
  (verify_kcat_table.py)
* ``027-betaxanthin-metabolic/results/mulleder_concentration_window.json``
  (mulleder_concentration_window.py), the only source for the window check

The scoring rule is the 027 rule and it is stated in every table note: the pinned
Merzbacher test Spearman at the epoch chosen by the centered five-epoch moving average of
validation Pearson, never a maximum over epochs. Every paired mean carries its n and its
across-seed standard deviation. The Genesis criterion is the share of the top 100 predicted
strains that balance, sit inside their box and run downhill under the model's own reaction
energies; it is reported under all three balance rules, the trainer's median, the max over
independent rows, and the max over the rows that turn over above 1e-3 of the flux scale.
"""

from __future__ import annotations

import json
import math
import os
import os.path as osp
from datetime import UTC, datetime
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
R26 = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
R27 = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "results")
OUT_STEM = osp.join(R27, "round2_summary_2026_09_12")

#: Result stems the round submitted last. The script reports only whether each landed
#: before this run; it never invents a score for one that did not.
EXPECTED_LATE_STEMS = (
    "wave7_aa_pooledq_e200_s2702",
    "wave6_proj_flux_projection_s2703",
    "wave6_proj_flux_projection_s2704",
    "wave6_proj_flux_projection_s2705",
)

SCORING_BETAXANTHIN = (
    "pinned Merzbacher test Spearman on 629 genes at the epoch selected by the centered "
    "5-epoch moving average of validation Pearson, never a maximum over epochs"
)
SCORING_AMINO_ACID = (
    "mean per-feature test Spearman over the 19 Mulleder amino acids on 564 pinned "
    "genotypes at the epoch selected by the centered 5-epoch moving average of the mean "
    "per-feature validation Pearson, never a maximum over epochs"
)


def _load(root: str, name: str) -> dict[str, Any]:
    with open(osp.join(root, name)) as f:
        obj: dict[str, Any] = json.load(f)
    return obj


def wilson_lower(k: int, n: int, z: float = 1.959963984540054) -> float:
    """Wilson 95 percent lower bound, the statistic the admission rule ranks on."""
    if n == 0:
        return 0.0
    p = k / n
    d = 1.0 + z * z / n
    center = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n))
    return (center - half) / d


# --------------------------------------------------------------------------- models


class ThermoLadderRung(BaseModel):
    """One rung of the chemical-reaction coverage ladder."""

    name: str
    n_reactions_priced: int
    fraction_reactions_priced: float
    n_flux_carrying_priced: int
    fraction_flux_carrying_priced: float
    what_it_would_take: str


class WithdrawnPrice(BaseModel):
    """An accession-priced metabolite whose structure is not the GEM's molecule."""

    met_id: str
    name: str
    gem_formula: str
    structure_formula: str
    shipped_dgf_prime_kj_per_mol: float
    detail: str
    withdrawn: bool
    re_derived_by_record: str | None


class ThermoCoverage(BaseModel):
    """Coverage before and after the curation round, and where the gains came from."""

    n_reactions: int
    n_reactions_priced_before: int
    n_reactions_priced_after: int
    n_reactions_chemical: int
    n_reactions_chemical_priced: int
    n_reactions_non_chemical: int
    n_metabolites_priced_before: int
    n_metabolites_priced_after: int
    ladder: list[ThermoLadderRung]
    ladder_denominator: int
    ladder_partition_check: bool
    gains_by_route: dict[str, int]
    withdrawn_prices: list[WithdrawnPrice]
    n_accessions_checked: int
    n_accessions_matching: int
    n_accessions_repeat_unit: int
    permanently_closed_reactions_by_reason: dict[str, int]
    unpriced_metabolites_by_reason: dict[str, int]
    permanent_gap_classes: dict[str, int]
    n_reactions_imbalanced: int


class CurationRecord(BaseModel):
    """One curated thermodynamic record, one row."""

    slug: str
    kind: str
    admission: str
    admitted: bool
    n_metabolites_priced: int
    n_reactions_closed: int
    n_reactions_closed_by_cancellation: int
    what_it_would_take: str


class CatPredVerdict(BaseModel):
    """The admission read for one kinetic parameter."""

    parameter: str
    lookup_point_fraction: float
    n_scorable: int
    threshold_hits: int
    predictor_rows: list[dict[str, Any]]
    admitted_before: list[str]
    admitted_after: list[str]
    n_units: int
    n_units_changed: int
    median_abs_change_decades: float
    p90_abs_change_decades: float
    max_abs_change_decades: float
    n_units_over_factor_two: int
    n_units_over_one_decade: int


class CatPredOverlap(BaseModel):
    """How much of the scorable set the released weights were fit on."""

    parameter: str
    levels: dict[str, str]
    n_gem_accessions: int
    n_gem_accessions_in_db: int
    n_gem_units: int


class CatPredUncertainty(BaseModel):
    """What the per-prediction variance is worth, on the 58 measured pairs."""

    parameter: str
    n_pairs: int
    n_pairs_not_in_db: int
    sd_median_decades_total: float
    coverage_1_sigma: float
    coverage_2_sigma: float
    rms_z_total: float
    rms_z_aleatoric: float
    rms_z_epistemic: float
    spearman_sd_vs_abs_error: float
    spearman_p: float
    mae_by_sd_quartile: list[float]
    combination_rules: dict[str, float]


class ParityRow(BaseModel):
    """One seed of the betaxanthin parity contrast."""

    seed: int
    flux: float
    pooled: float
    difference: float


class ParityContrast(BaseModel):
    """A paired betaxanthin parity contrast with its n and its across-seed spread."""

    key: str
    name: str
    rows: list[ParityRow]
    n_pairs: int
    mean_difference: float
    sd_difference: float | None
    t_statistic: float | None
    flux_mean: float
    flux_sd: float | None
    pooled_mean: float
    pooled_sd: float | None
    one_kcat_table: bool
    note: str


class FeasibilityRow(BaseModel):
    """One audited cell, read under all three balance rules."""

    head: str
    cell: str
    label: str
    job_ids: list[int]
    group: str
    n: int
    flux_active_frac_median: float
    balance_median_ratio: float
    balance_max_ratio: float
    balance_active_max_ratio: float
    balance_active_p95_ratio: float
    balance_active_rows: float
    pass_median_rule: float
    pass_max_rule: float
    pass_active_rule: float
    genesis_median_rule: float | None
    genesis_max_rule: float | None
    genesis_active_rule: float | None
    agreement_max_abs_delta: float


class FluxForceArm(BaseModel):
    """One flux-force or control arm on the betaxanthin head."""

    arm: str
    channel: str
    n: int
    job_ids: list[int]
    test_spearman_mean: float
    test_spearman_sd: float | None
    escaped: int
    escaped_sustained: int
    genesis_top100_mean: float
    balance_le_1e_2_top100_mean: float
    second_law_eq_0_top100_mean: float
    sign_flip_frac_mean: float | None


class FluxForceContrast(BaseModel):
    """A paired flux-force contrast."""

    arm: str
    against: str
    n_pairs: int
    mean_diff: float
    sd_diff: float | None


class AminoAcidContrast(BaseModel):
    """A paired amino-acid contrast at one budget."""

    key: str
    budget: str
    arm_a: str
    arm_b: str
    seeds: list[int]
    a_values: list[float]
    b_values: list[float]
    differences: list[float]
    n_pairs: int
    mean_difference: float | None
    sd_difference: float | None
    t_statistic: float | None


class ConcentrationAgreement(BaseModel):
    """The learned whole-cell concentration against the measured one."""

    cell: str
    label: str
    seed: int
    supervised: bool
    mean_spearman: float
    mean_median_abs_error_decades: float
    min_per_column_spearman: float
    max_per_column_spearman: float
    frac_pred_at_ceiling: float
    frac_pred_at_floor: float


class WindowCheck(BaseModel):
    """The Mulleder measurements against the flux layer's concentration window."""

    dataset_name: str
    n_records: int
    n_measurements: int
    conc_lower_m: float
    conc_upper_m: float
    overall_min_m: float
    overall_max_m: float
    frac_measurements_above_upper: float
    frac_measurements_below_lower: float
    n_amino_acids_exceeding_upper: int
    amino_acids_exceeding_upper: list[str]
    recommended_conc_upper_m: float


class LedgerRow(BaseModel):
    """One wave-7 cell, from the result files' own metadata."""

    stream: str
    cell: str
    job_id: int | None
    run_id: str | None
    arm: str
    seed: int
    epochs_run: int
    select_epoch: int
    kcat_table: str
    score: float
    endpoint: str


class PendingCell(BaseModel):
    """A cell a stream expected and this run did not find."""

    stem: str
    landed: bool
    named_by: str


class Round2Summary(BaseModel):
    """Every number the round-2 note quotes, read from the stream result files."""

    generated_at: str
    kcat_table_expected: dict[str, int]
    kcat_table_matches_shipped: bool
    thermo: ThermoCoverage
    curation_records: list[CurationRecord]
    catpred_verdicts: list[CatPredVerdict]
    catpred_overlap: list[CatPredOverlap]
    catpred_uncertainty: list[CatPredUncertainty]
    parity_contrasts: list[ParityContrast]
    feasibility: list[FeasibilityRow]
    fluxforce_arms: list[FluxForceArm]
    fluxforce_contrasts: list[FluxForceContrast]
    amino_acid_contrasts: list[AminoAcidContrast]
    concentration_agreement: list[ConcentrationAgreement]
    window_check: WindowCheck
    ledger: list[LedgerRow]
    pending: list[PendingCell]


# ----------------------------------------------------------------------------- build


def _thermo(closure: dict[str, Any], coverage: dict[str, Any]) -> ThermoCoverage:
    rows = {r["slug"]: r for r in closure["rows"]}
    curated_met_ids = {
        met_id: slug for slug, r in rows.items() for met_id in r["met_ids"]
    }
    withdrawn: list[WithdrawnPrice] = []
    for o in closure["accession_sweep"]["other"]:
        slug = curated_met_ids.get(o["met_id"])
        withdrawn.append(
            WithdrawnPrice(
                met_id=o["met_id"],
                name=o["name"],
                gem_formula=o["gem_formula"],
                structure_formula=o["structure_formula"],
                shipped_dgf_prime_kj_per_mol=o["shipped_dgf_prime_kj_per_mol"],
                detail=o["detail"],
                withdrawn=slug is None,
                re_derived_by_record=slug,
            )
        )
    gains = {
        "cancellation": int(closure["n_reactions_cancelled"]),
        "carrier pseudo pair": int(rows["cytochrome_c"]["n_reactions_closed"]),
        "mannan residue value": int(rows["mannan"]["n_reactions_closed"]),
        "conserved moiety": int(closure["moiety_outcomes"].get("priced", 0)),
    }
    # The remedy sentence is the key, kept whole: three of the seven classes open with
    # "nothing is possible" and truncating at the colon would silently merge them.
    gaps = {
        str(k): int(v["n_metabolites"]) for k, v in coverage["remedy_summary"].items()
    }
    return ThermoCoverage(
        n_reactions=int(closure["n_reactions"]),
        n_reactions_priced_before=int(closure["n_reactions_priced_before"]),
        n_reactions_priced_after=int(closure["n_reactions_priced_after"]),
        n_reactions_chemical=int(closure["n_reactions_chemical"]),
        n_reactions_chemical_priced=int(closure["n_reactions_chemical_priced"]),
        n_reactions_non_chemical=int(closure["n_reactions_non_chemical"]),
        n_metabolites_priced_before=int(closure["n_metabolites_priced_before"]),
        n_metabolites_priced_after=int(closure["n_metabolites_priced_after"]),
        ladder=[
            ThermoLadderRung(
                name=str(r["name"]).replace("_", " "),
                n_reactions_priced=int(r["n_reactions_priced"]),
                fraction_reactions_priced=float(r["fraction_reactions_priced"]),
                n_flux_carrying_priced=int(r["n_flux_carrying_priced"]),
                fraction_flux_carrying_priced=float(r["fraction_flux_carrying_priced"]),
                what_it_would_take=str(r["what_it_would_take"]),
            )
            for r in closure["ladder_after"]
        ],
        ladder_denominator=int(closure["ladder_denominator"]),
        ladder_partition_check=bool(closure["ladder_partition_check"]),
        gains_by_route=gains,
        withdrawn_prices=withdrawn,
        n_accessions_checked=int(closure["accession_sweep"]["n_checked"]),
        n_accessions_matching=int(closure["accession_sweep"]["n_match"]),
        n_accessions_repeat_unit=int(
            closure["accession_sweep"]["n_repeat_unit_multiple"]
        ),
        permanently_closed_reactions_by_reason={
            str(k): int(v)
            for k, v in coverage["ceiling_ladder"]["permanently_closed_reasons"].items()
        },
        unpriced_metabolites_by_reason={
            str(k): int(v) for k, v in coverage["metabolite_reasons"].items()
        },
        permanent_gap_classes=gaps,
        n_reactions_imbalanced=int(closure["n_reactions_imbalanced"]),
    )


def _curation_records(closure: dict[str, Any]) -> list[CurationRecord]:
    return [
        CurationRecord(
            slug=str(r["slug"]),
            kind=str(r["kind"]),
            admission=str(r["admission"]),
            admitted=bool(r["admitted"]),
            n_metabolites_priced=int(r["n_metabolites_priced"]),
            n_reactions_closed=int(r["n_reactions_closed"]),
            n_reactions_closed_by_cancellation=int(
                r["n_reactions_closed_by_cancellation"]
            ),
            what_it_would_take=str(r["what_it_would_take"]),
        )
        for r in sorted(closure["rows"], key=lambda x: str(x["slug"]))
    ]


def _catpred(plaus: dict[str, Any], impact: dict[str, Any]) -> list[CatPredVerdict]:
    out: list[CatPredVerdict] = []
    impact_by_param = {p["parameter"]: p for p in impact["parameters"]}
    for parameter in ("k_cat", "K_M"):
        block = plaus["parameters"][parameter]
        band = block["own_class_band"]
        lookup = float(block["ec_median_lookup"]["own_class"]["frac"])
        n_scorable = int(band["n_pairs_scorable"])
        threshold = next(
            (k for k in range(n_scorable + 1) if wilson_lower(k, n_scorable) > lookup),
            n_scorable + 1,
        )
        rows: list[dict[str, Any]] = []
        for name, d in sorted(
            band["predictors"].items(), key=lambda kv: -float(kv[1]["frac"])
        ):
            lower = wilson_lower(int(d["k"]), int(d["n"]))
            rows.append(
                {
                    "predictor": name,
                    "k": int(d["k"]),
                    "n": int(d["n"]),
                    "fraction": float(d["frac"]),
                    "wilson_lower": lower,
                    "verdict": "admitted" if lower > lookup else "rejected",
                }
            )
        imp = impact_by_param[parameter]
        out.append(
            CatPredVerdict(
                parameter=parameter,
                lookup_point_fraction=lookup,
                n_scorable=n_scorable,
                threshold_hits=threshold,
                predictor_rows=rows,
                admitted_before=[str(x) for x in imp["selected_shipped"]],
                admitted_after=[str(x) for x in imp["selected_with_catpred"]],
                n_units=int(imp["n_units"]),
                n_units_changed=int(imp["n_units_changed"]),
                median_abs_change_decades=float(imp["median_abs_change_decades"]),
                p90_abs_change_decades=float(imp["p90_abs_change_decades"]),
                max_abs_change_decades=float(imp["max_abs_change_decades"]),
                n_units_over_factor_two=int(imp["n_units_over_factor_two"]),
                n_units_over_one_decade=int(imp["n_units_over_one_decade"]),
            )
        )
    return out


def _catpred_overlap(overlap: dict[str, Any]) -> list[CatPredOverlap]:
    return [
        CatPredOverlap(
            parameter=str(p["parameter"]),
            levels={
                str(x["level"]): f"{x['n_matched']}/{x['n_total']} "
                f"({100.0 * float(x['frac']):.1f} %)"
                for x in p["levels"]
            },
            n_gem_accessions=int(p["n_gem_accessions"]),
            n_gem_accessions_in_db=int(p["n_gem_accessions_in_db"]),
            n_gem_units=int(p["n_gem_units"]),
        )
        for p in overlap["parameters"]
    ]


def _catpred_uncertainty(unc: dict[str, Any]) -> list[CatPredUncertainty]:
    out: list[CatPredUncertainty] = []
    for p in unc["parameters"]:
        cov = {c["component"]: c for c in p["coverage_all"]}
        out.append(
            CatPredUncertainty(
                parameter=str(p["parameter"]),
                n_pairs=int(p["n_pairs"]),
                n_pairs_not_in_db=int(p["n_pairs_not_in_catpred_db"]),
                sd_median_decades_total=float(p["sd_median_decades"]["total"]),
                coverage_1_sigma=float(cov["total_1_sigma"]["empirical"]),
                coverage_2_sigma=float(cov["total_2_sigma"]["empirical"]),
                rms_z_total=float(cov["total_1_sigma"]["rms_z"]),
                rms_z_aleatoric=float(cov["aleatoric_1_sigma"]["rms_z"]),
                rms_z_epistemic=float(cov["epistemic_1_sigma"]["rms_z"]),
                spearman_sd_vs_abs_error=float(p["spearman_sd_vs_abs_error"]),
                spearman_p=float(p["spearman_p"]),
                mae_by_sd_quartile=[float(x) for x in p["mae_by_sd_quartile"]],
                combination_rules={
                    str(c["rule"]): float(c["median_abs_error_decades"])
                    for c in p["combinations"]
                },
            )
        )
    return out


PARITY_KEYS = (
    ("wave7", "the three wave-7 pairs, every flux cell on the shipped table", True),
    (
        "combined_with_wave6_same_table_group",
        "wave 7 pooled with the largest wave-6 same-table group",
        False,
    ),
    ("wave6_reference", "wave 6, its three kcat tables pooled", False),
)


def _parity(summary: dict[str, Any]) -> list[ParityContrast]:
    out: list[ParityContrast] = []
    for key, name, one_table in PARITY_KEYS:
        b = summary[key]
        if int(b["n_pairs"]) == 0:
            continue
        rows = [
            ParityRow(seed=int(s), flux=float(f), pooled=float(p), difference=float(d))
            for s, f, p, d in zip(
                b["seeds"],
                b["flux_values"],
                b["pooled_values"],
                b["differences"],
                strict=True,
            )
        ]
        out.append(
            ParityContrast(
                key=key,
                name=name,
                rows=rows,
                n_pairs=int(b["n_pairs"]),
                mean_difference=float(b["mean_difference"]),
                sd_difference=b["sd_difference"],
                t_statistic=b["t_statistic"],
                flux_mean=float(b["flux_mean"]),
                flux_sd=b["flux_sd"],
                pooled_mean=float(b["pooled_mean"]),
                pooled_sd=b["pooled_sd"],
                one_kcat_table=one_table,
                note=str(b["note"]),
            )
        )
    return out


def _head_of(label: str) -> str:
    """The head family of an audited cell, taken from its own label's seed suffix."""
    parts = label.split()
    return " ".join(parts[:-1]) if parts[-1].startswith("s27") else label


def _feasibility(summary: dict[str, Any]) -> list[FeasibilityRow]:
    rows = [
        FeasibilityRow(
            head=_head_of(str(f["label"])),
            cell=str(f["cell"]),
            label=str(f["label"]),
            job_ids=[int(j) for j in f["slurm_job_ids"]],
            group=str(f["group"]),
            n=int(f["n"]),
            flux_active_frac_median=float(f["flux_active_frac_median"]),
            balance_median_ratio=float(f["balance_median_ratio_median"]),
            balance_max_ratio=float(f["balance_max_ratio_median"]),
            balance_active_max_ratio=float(f["balance_active_max_ratio_median"]),
            balance_active_p95_ratio=float(f["balance_active_p95_ratio_median"]),
            balance_active_rows=float(f["balance_active_rows_median"]),
            pass_median_rule=float(f["balance_median_le_1e_2"]),
            pass_max_rule=float(f["balance_max_le_1e_2"]),
            pass_active_rule=float(f["balance_active_max_le_1e_2"]),
            genesis_median_rule=f["genesis_median_rule_model_thermo"],
            genesis_max_rule=f["genesis_max_rule_model_thermo"],
            genesis_active_rule=f["genesis_active_rule_model_thermo"],
            agreement_max_abs_delta=float(f["agreement_max_abs_delta"]),
        )
        for f in summary["feasibility"]
    ]
    return sorted(rows, key=lambda r: (r.head, r.label))


def _fluxforce(
    summary: dict[str, Any],
) -> tuple[list[FluxForceArm], list[FluxForceContrast]]:
    arms = [
        FluxForceArm(
            arm=str(a["arm"]),
            channel=str(a["channel"]),
            n=int(a["n"]),
            job_ids=[int(j) for j in a["job_ids"]],
            test_spearman_mean=float(a["test_spearman_mean"]),
            test_spearman_sd=a["test_spearman_sd"],
            escaped=int(a["escaped"]),
            escaped_sustained=int(a["escaped_sustained"]),
            genesis_top100_mean=float(a["genesis_top100_mean"]),
            balance_le_1e_2_top100_mean=float(a["balance_le_1e_02_top100_mean"]),
            second_law_eq_0_top100_mean=float(a["thermo_model_eq_0_top100_mean"]),
            sign_flip_frac_mean=a["sign_flip_frac_mean"],
        )
        for a in summary["arms"]
    ]
    contrasts = [
        FluxForceContrast(
            arm=str(c["arm"]),
            against=str(c["against"]),
            n_pairs=int(c["n_pairs"]),
            mean_diff=float(c["mean_diff"]),
            sd_diff=c["sd_diff"],
        )
        for c in summary["contrasts"]
    ]
    return arms, contrasts


AA_CONTRAST_KEYS = (
    "e200/conc_vs_pooled",
    "e200/conc_nocstr_vs_pooled",
    "e200/turnover_vs_pooled",
    "e200/conc_vs_turnover",
    "e200/conc_vs_conc_nocstr",
    "e200/pooledq_vs_pooled",
    "e200/conc_vs_pooledq",
    "e80/conc_vs_pooled",
)


def _amino_acid(summary: dict[str, Any]) -> list[AminoAcidContrast]:
    out: list[AminoAcidContrast] = []
    for key in AA_CONTRAST_KEYS:
        c = summary["contrasts"][key]
        if int(c["n_pairs"]) == 0:
            continue
        budget, _ = key.split("/", 1)
        out.append(
            AminoAcidContrast(
                key=key,
                budget=budget,
                arm_a=str(c["arm_a"]),
                arm_b=str(c["arm_b"]),
                seeds=[int(s) for s in c["seeds"]],
                a_values=[float(x) for x in c["a_values"]],
                b_values=[float(x) for x in c["b_values"]],
                differences=[float(x) for x in c["differences"]],
                n_pairs=int(c["n_pairs"]),
                mean_difference=float(c["mean_difference"]),
                sd_difference=c["sd_difference"],
                t_statistic=c["t_statistic"],
            )
        )
    return out


def _concentration(summary: dict[str, Any]) -> list[ConcentrationAgreement]:
    out: list[ConcentrationAgreement] = []
    for c in summary["concentration"]:
        errors = [float(x) for x in c["median_abs_error_log10"]]
        spearman = [float(x) for x in c["spearman_per_feature"]]
        out.append(
            ConcentrationAgreement(
                cell=str(c["cell"]),
                label=str(c["label"]),
                seed=int(c["seed"]),
                supervised=bool(c["supervised"]),
                mean_spearman=float(c["mean_spearman"]),
                mean_median_abs_error_decades=sum(errors) / len(errors),
                min_per_column_spearman=min(spearman),
                max_per_column_spearman=max(spearman),
                frac_pred_at_ceiling=float(c["frac_pred_at_ceiling"]),
                frac_pred_at_floor=float(c["frac_pred_at_floor"]),
            )
        )
    return out


def _window(w: dict[str, Any]) -> WindowCheck:
    return WindowCheck(
        dataset_name=str(w["dataset_name"]),
        n_records=int(w["n_records"]),
        n_measurements=int(w["n_measurements"]),
        conc_lower_m=float(w["conc_lower_m"]),
        conc_upper_m=float(w["conc_upper_m"]),
        overall_min_m=float(w["overall_min_m"]),
        overall_max_m=float(w["overall_max_m"]),
        frac_measurements_above_upper=float(w["frac_measurements_above_upper"]),
        frac_measurements_below_lower=float(w["frac_measurements_below_lower"]),
        n_amino_acids_exceeding_upper=len(w["amino_acids_exceeding_upper"]),
        amino_acids_exceeding_upper=[str(a) for a in w["amino_acids_exceeding_upper"]],
        recommended_conc_upper_m=float(w["recommended_conc_upper_m"]),
    )


def _kcat_fingerprint(counts: dict[str, Any] | None, expected: dict[str, int]) -> str:
    if not counts:
        return "none, no flux layer"
    fp = (
        f"{int(counts['measured'])}/{int(counts.get('homolog', 0))}/"
        f"{int(counts['predicted'])}/{int(counts['default'])}"
    )
    matches = all(int(counts.get(k, 0)) == v for k, v in expected.items())
    return f"{fp} shipped" if matches else f"{fp} off-table"


def _ledger(
    parity: dict[str, Any],
    ff: dict[str, Any],
    aa: dict[str, Any],
    expected: dict[str, int],
) -> list[LedgerRow]:
    rows: list[LedgerRow] = []
    for c in parity["cells"]:
        if str(c.get("wave")) != "wave7":
            continue
        jobs = [int(j) for j in c.get("slurm_job_ids", [])]
        rows.append(
            LedgerRow(
                stream="parity",
                cell=str(c["cell"]).removesuffix(".json"),
                job_id=jobs[0] if jobs else None,
                run_id=str(c["wandb_url"]).rsplit("/", 1)[-1]
                if c.get("wandb_url")
                else None,
                arm=str(c["arm"]),
                seed=int(c["seed"]),
                epochs_run=int(c["epochs_run"]),
                select_epoch=int(c["select_epoch"]),
                kcat_table=_kcat_fingerprint(c.get("kcat_tier_counts"), expected),
                score=float(c["pinned_test_spearman"]),
                endpoint="betaxanthin",
            )
        )
    for c in ff["cells"]:
        if not str(c["stem"]).startswith("wave7_"):
            continue
        rows.append(
            LedgerRow(
                stream="flux force",
                cell=str(c["stem"]),
                job_id=int(c["job_id"]) if c.get("job_id") is not None else None,
                run_id=str(c["wandb_run_id"]) if c.get("wandb_run_id") else None,
                arm=str(c["channel"]),
                seed=int(c["seed"]),
                epochs_run=int(c["epochs_run"]),
                select_epoch=int(c["select_epoch"]),
                kcat_table=_kcat_fingerprint(c.get("kcat_tier_counts"), expected),
                score=float(c["test_spearman_pinned"]),
                endpoint="betaxanthin",
            )
        )
    for c in aa["cells"]:
        rows.append(
            LedgerRow(
                stream="amino acid",
                cell=str(c["cell"]).removesuffix(".json"),
                job_id=None,
                run_id=str(c["wandb_url"]).rsplit("/", 1)[-1]
                if c.get("wandb_url")
                else None,
                arm=(
                    f"{c['arm']}, {c['aa_readout']}"
                    if str(c["arm"]) != "pooled"
                    else "pooled"
                ),
                seed=int(c["seed"]),
                epochs_run=int(c["epochs_run"]),
                select_epoch=int(c["select_epoch"]),
                kcat_table=_kcat_fingerprint(c.get("kcat_tier_counts"), expected),
                score=float(c["score_mean_spearman"]),
                endpoint="mulleder19",
            )
        )
    return rows


def _pending(
    parity: dict[str, Any], ff: dict[str, Any], aa: dict[str, Any]
) -> list[PendingCell]:
    named: dict[str, str] = {}
    for stem in parity["missing_files"]:
        named.setdefault(str(stem).removesuffix(".json"), "parity")
    for stem in ff["missing"]:
        named.setdefault(str(stem).removesuffix(".json"), "flux force")
    for stem in aa["missing_files"]:
        named.setdefault(str(stem).removesuffix(".json"), "amino acid")
    for stem in EXPECTED_LATE_STEMS:
        named.setdefault(stem, "driver fill")
    return [
        PendingCell(
            stem=stem,
            landed=osp.exists(osp.join(R27, stem + ".json")),
            named_by=named[stem],
        )
        for stem in sorted(named)
    ]


def build() -> Round2Summary:
    closure = _load(R26, "thermo_curation_closure.json")
    coverage = _load(R26, "reaction_thermo_coverage_final.json")
    impact = _load(R26, "catpred_table_impact.json")
    overlap = _load(R26, "catpred_training_overlap.json")
    unc = _load(R26, "catpred_uncertainty.json")
    plaus = _load(R26, "within_class_plausibility_with_catpred.json")
    parity = _load(R27, "wave7_parity_summary.json")
    ff = _load(R27, "wave7_fluxforce_summary.json")
    aa = _load(R27, "wave7_aa_summary.json")
    verify = _load(R27, "wave7_kcat_table_verification.json")
    window = _load(R27, "mulleder_concentration_window.json")

    expected = {str(k): int(v) for k, v in verify["expected_tier_counts"].items()}
    arms, contrasts = _fluxforce(ff)
    return Round2Summary(
        generated_at=datetime.now(UTC).isoformat(),
        kcat_table_expected=expected,
        kcat_table_matches_shipped=bool(verify["matches_shipped"]),
        thermo=_thermo(closure, coverage),
        curation_records=_curation_records(closure),
        catpred_verdicts=_catpred(plaus, impact),
        catpred_overlap=_catpred_overlap(overlap),
        catpred_uncertainty=_catpred_uncertainty(unc),
        parity_contrasts=_parity(parity),
        feasibility=_feasibility(parity),
        fluxforce_arms=arms,
        fluxforce_contrasts=contrasts,
        amino_acid_contrasts=_amino_acid(aa),
        concentration_agreement=_concentration(aa),
        window_check=_window(window),
        ledger=_ledger(parity, ff, aa, expected),
        pending=_pending(parity, ff, aa),
    )


# -------------------------------------------------------------------------- markdown

#: Display names, so a table column never carries a code identifier wider than its cell.
ARM_LABEL = {
    "fluxforce": "flux force, box clip last",
    "fluxforce_force_e200": "flux force, direction clip last",
    "fluxforce_boxforce": "flux force, direction in the box throughout",
    "tdg": "sign clamp",
    "projection": "plain projection",
    "flux_capacity": "box flux head",
    "pooled": "pooled control",
}
AA_ARM_LABEL = {
    "conc": "concentration",
    "conc_nocstr": "concentration, constraint scale 0",
    "turnover": "turnover",
    "pooled": "pooled, CRPS",
    "pooledq": "pooled, pinball",
}


def _f(x: float | None, nd: int = 4) -> str:
    if x is None:
        return "n/a"
    if x != 0 and abs(x) < 10.0 ** (-nd):
        return f"{x:.1e}"
    return f"{x:.{nd}f}"


def _i(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def markdown(s: Round2Summary) -> str:  # noqa: PLR0915
    t = s.thermo
    lines: list[str] = []

    lines.append(
        "**Table 1. Reaction energy coverage before and after the curation round.**\n"
    )
    lines.append("| quantity | before | after |")
    lines.append("|---|---:|---:|")
    lines.append(
        f"| reactions priced, all {t.n_reactions} | {t.n_reactions_priced_before} | "
        f"{t.n_reactions_priced_after} |"
    )
    lines.append(
        f"| reactions priced, chemical {t.n_reactions_chemical} | not reported | "
        f"{t.n_reactions_chemical_priced} |"
    )
    lines.append(
        f"| metabolites priced | {t.n_metabolites_priced_before} | "
        f"{t.n_metabolites_priced_after} |"
    )
    lines.append(
        f"| non-chemical reactions, outside the denominator | 0 | "
        f"{t.n_reactions_non_chemical} |"
    )
    lines.append(
        f"| chemical reactions failing the elemental or charge balance check | "
        f"not reported | {t.n_reactions_imbalanced} |"
    )
    lines.append(
        "\nThe chemical denominator exists only in the after run, so the two whole-model "
        "counts are the comparable pair and the chemical fraction is the honest one.\n"
    )

    lines.append(
        f"**Table 2. The chemical ladder, denominator {t.ladder_denominator} reactions; "
        f"the three states partition it exactly ({t.ladder_partition_check}).**\n"
    )
    lines.append(
        "| rung | reactions priced | fraction | flux-carrying priced | fraction | "
        "what it would take |"
    )
    lines.append("|---|---:|---:|---:|---:|---|")
    for rung in t.ladder:
        lines.append(
            f"| {rung.name} | {rung.n_reactions_priced} | {rung.fraction_reactions_priced:.4f} | "
            f"{rung.n_flux_carrying_priced} | {rung.fraction_flux_carrying_priced:.4f} | "
            f"{rung.what_it_would_take} |"
        )
    lines.append("")

    lines.append(
        "**Table 3. Where the gains came from, and what a permanent gap is.**\n"
    )
    lines.append("| route | reactions gained |")
    lines.append("|---|---:|")
    for k, v in t.gains_by_route.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("| permanently closed reactions, sole reason | count |")
    lines.append("|---|---:|")
    for k, v in sorted(
        t.permanently_closed_reactions_by_reason.items(), key=lambda kv: -kv[1]
    ):
        lines.append(f"| {k.replace('_', ' ')} | {v} |")
    lines.append("")
    lines.append("| unpriced metabolites, reason | count |")
    lines.append("|---|---:|")
    for k, v in sorted(t.unpriced_metabolites_by_reason.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k.replace('_', ' ')} | {v} |")
    lines.append("")
    lines.append("| what the unpriced metabolite would take | count |")
    lines.append("|---|---:|")
    for k, v in sorted(t.permanent_gap_classes.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} |")
    lines.append("")

    n_withdrawn = sum(1 for w in t.withdrawn_prices if w.withdrawn)
    lines.append(
        f"**Table 4. The accession sweep, {t.n_accessions_checked} priced species: "
        f"{t.n_accessions_matching} match the GEM's molecule, "
        f"{t.n_accessions_repeat_unit} are a consistent repeat-unit multiple, and "
        f"{len(t.withdrawn_prices)} were priced from a different molecule, of which "
        f"{n_withdrawn} are withdrawn.**\n"
    )
    lines.append(
        "| metabolite | name | GEM formula | structure priced | shipped dGf'0 kJ/mol | "
        "disposition |"
    )
    lines.append("|---|---|---|---|---:|---|")
    for wp in t.withdrawn_prices:
        disposition = (
            "withdrawn"
            if wp.withdrawn
            else f"re-derived by the {wp.re_derived_by_record} record"
        )
        lines.append(
            f"| {wp.met_id} | {wp.name} | {wp.gem_formula} | {wp.structure_formula} | "
            f"{wp.shipped_dgf_prime_kj_per_mol:.1f} | {disposition} |"
        )
    lines.append("")

    lines.append(
        f"**Table 5. The {len(s.curation_records)} curated records, one row each; only "
        "measured and tabulated admissions reach a table a model reads.**\n"
    )
    lines.append(
        "| slug | kind | admission | admitted | metabolites priced | reactions closed | "
        "closed by cancellation | what it would take |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---|")
    for rec in s.curation_records:
        lines.append(
            f"| {rec.slug.replace('_', ' ')} | {rec.kind.replace('_', ' ')} | "
            f"{rec.admission} | "
            f"{'yes' if rec.admitted else 'no'} | {rec.n_metabolites_priced} | "
            f"{rec.n_reactions_closed} | {rec.n_reactions_closed_by_cancellation} | "
            f"{rec.what_it_would_take.split(';')[0].split(':')[0]} |"
        )
    lines.append("")

    for cpv in s.catpred_verdicts:
        lines.append(
            f"**Table 6{'a' if cpv.parameter == 'k_cat' else 'b'}. CatPred admission for "
            f"{cpv.parameter}: the EC-median lookup scores "
            f"{cpv.lookup_point_fraction:.4f} on {cpv.n_scorable} scorable pairs, so a "
            f"predictor needs {cpv.threshold_hits} hits for its Wilson 95 percent lower "
            f"bound to clear that point.**\n"
        )
        lines.append(
            "| predictor | own-class hits | fraction | Wilson lower | verdict |"
        )
        lines.append("|---|---:|---:|---:|---|")
        for row in cpv.predictor_rows:
            lines.append(
                f"| {row['predictor']} | {row['k']}/{row['n']} | "
                f"{row['fraction']:.4f} | {row['wilson_lower']:.4f} | {row['verdict']} |"
            )
        lines.append(
            f"\nAdmitted before: {', '.join(cpv.admitted_before)}. Admitted after: "
            f"{', '.join(cpv.admitted_after)}. Of {cpv.n_units} catalytic units "
            f"{cpv.n_units_changed} change value, median move "
            f"{cpv.median_abs_change_decades:.3f} decades, p90 "
            f"{cpv.p90_abs_change_decades:.3f}, max {cpv.max_abs_change_decades:.3f}, "
            f"{cpv.n_units_over_factor_two} past a factor of two and "
            f"{cpv.n_units_over_one_decade} past one decade.\n"
        )

    lines.append(
        "**Table 7. The training overlap, matched on UniProt accession and on the "
        "InChIKey connectivity block against the table the production weights were fit "
        "on.**\n"
    )
    levels = ("enzyme", "pair", "value")
    lines.append(
        "| parameter | "
        + " | ".join(levels)
        + " | GEM proteins in CatPred-DB | GEM units |"
    )
    lines.append("|---|" + "---:|" * (len(levels) + 2))
    for o in s.catpred_overlap:
        lines.append(
            f"| {o.parameter} | "
            + " | ".join(o.levels.get(x, "n/a") for x in levels)
            + f" | {o.n_gem_accessions_in_db}/{o.n_gem_accessions} | {o.n_gem_units} |"
        )
    lines.append("")

    lines.append(
        "**Table 8. What the per-prediction variance is worth, on the measured pairs. "
        "The root mean square of the standardized residual is 1 for a calibrated "
        "interval.**\n"
    )
    lines.append(
        "| parameter | pairs | pairs absent from CatPred-DB | median predicted sd, "
        "decades | coverage at 1 sd (nominal 0.683) | coverage at 2 sd (nominal 0.954) | "
        "rms z total | rms z aleatoric | rms z epistemic | Spearman, sd vs absolute "
        "error | p |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for u in s.catpred_uncertainty:
        lines.append(
            f"| {u.parameter} | {u.n_pairs} | {u.n_pairs_not_in_db} | "
            f"{u.sd_median_decades_total:.3f} | {u.coverage_1_sigma:.3f} | "
            f"{u.coverage_2_sigma:.3f} | {u.rms_z_total:.2f} | "
            f"{u.rms_z_aleatoric:.2f} | {u.rms_z_epistemic:.2f} | "
            f"{u.spearman_sd_vs_abs_error:.3f} | {u.spearman_p:.1e} |"
        )
    lines.append("")
    rules = sorted({rk for u in s.catpred_uncertainty for rk in u.combination_rules})
    lines.append(
        "| combination rule, median absolute error in decades | "
        + " | ".join(u.parameter for u in s.catpred_uncertainty)
        + " |"
    )
    lines.append("|---|" + "---:|" * len(s.catpred_uncertainty))
    for rule in rules:
        lines.append(
            f"| {rule.replace('_', ' ')} | "
            + " | ".join(
                _f(u.combination_rules.get(rule), 3) for u in s.catpred_uncertainty
            )
            + " |"
        )
    lines.append(
        "\nMedian absolute error inside each quartile of the predicted standard "
        "deviation, lowest first: "
        + "; ".join(
            f"{u.parameter} " + ", ".join(f"{x:.3f}" for x in u.mae_by_sd_quartile)
            for u in s.catpred_uncertainty
        )
        + " decades.\n"
    )

    fp = "/".join(
        str(s.kcat_table_expected[k])
        for k in ("measured", "homolog", "predicted", "default")
    )
    for pc in s.parity_contrasts:
        lines.append(
            f"**Table 9{'abc'[s.parity_contrasts.index(pc)]}. Betaxanthin parity, "
            f"{pc.name}.**\n"
        )
        lines.append(
            "| seed | "
            + " | ".join(str(r.seed) for r in pc.rows)
            + " | mean | sd | n | t |"
        )
        lines.append("|---|" + "---:|" * len(pc.rows) + "---:|---:|---:|---:|")
        lines.append(
            "| box flux head | "
            + " | ".join(_f(r.flux) for r in pc.rows)
            + f" | {_f(pc.flux_mean)} | {_f(pc.flux_sd)} | | |"
        )
        lines.append(
            "| pooled | "
            + " | ".join(_f(r.pooled) for r in pc.rows)
            + f" | {_f(pc.pooled_mean)} | {_f(pc.pooled_sd)} | | |"
        )
        lines.append(
            "| difference | "
            + " | ".join(_f(r.difference) for r in pc.rows)
            + f" | {_f(pc.mean_difference)} | {_f(pc.sd_difference)} | {pc.n_pairs} | "
            f"{_f(pc.t_statistic, 2)} |"
        )
        table_line = (
            f"Every flux cell carries the shipped verified kcat table {fp}."
            if pc.one_kcat_table
            else "This pool spans more than one kcat table."
        )
        note = pc.note[:1].upper() + pc.note[1:] if pc.note else ""
        lines.append(f"\n{(table_line + ' ' + note).strip()}\n")
    lines.append(
        f"Scoring rule for Tables 9a to 9c: {SCORING_BETAXANTHIN}. The t is read for its "
        "sign and magnitude at this n and is not a significance claim.\n"
    )

    lines.append(
        f"**Table 10. Feasibility of the top 100 predicted strains for the "
        f"{len(s.feasibility)} audited cells, under the three balance rules.**\n"
    )
    lines.append(
        "| head | cell | jobs | flux-carrying | balance median | balance max | active "
        "max | active p95 | active rows | passes median | passes max | passes active | "
        "Genesis median | Genesis max | Genesis active |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for fr in s.feasibility:
        lines.append(
            f"| {fr.head} | {fr.label.split()[-1]} | "
            f"{', '.join(str(j) for j in fr.job_ids)} | "
            f"{fr.flux_active_frac_median:.3f} | {_f(fr.balance_median_ratio, 3)} | "
            f"{_f(fr.balance_max_ratio, 3)} | {_f(fr.balance_active_max_ratio, 3)} | "
            f"{_f(fr.balance_active_p95_ratio, 3)} | {fr.balance_active_rows:.0f} | "
            f"{fr.pass_median_rule:.2f} | {fr.pass_max_rule:.2f} | "
            f"{fr.pass_active_rule:.2f} | {_i(fr.genesis_median_rule)} | "
            f"{_i(fr.genesis_max_rule)} | {_i(fr.genesis_active_rule)} |"
        )
    lines.append(
        "\nThe active rule restricts the ratio to the rows whose turnover exceeds 1e-3 of "
        "the flux scale. A Genesis entry of n/a is an arm whose thermodynamics are off, so "
        "the layer carries no reaction energies of its own to test against. Every cell was "
        "recomputed from its checkpoint against the code tree it was trained with.\n"
    )

    lines.append(
        "**Table 11. The flux-force arms and their controls on the betaxanthin head, "
        "seeds 2700 to 2702.**\n"
    )
    lines.append(
        "| arm | channel | n | jobs | test Spearman | sd | escaped | sustained | second "
        "law exact | balance <= 1e-2 | Genesis, median rule | sign flips |"
    )
    lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in s.fluxforce_arms:
        lines.append(
            f"| {ARM_LABEL.get(arm.arm, arm.arm)} | `{arm.channel}` | {arm.n} | "
            f"{', '.join(str(j) for j in arm.job_ids)} | {_f(arm.test_spearman_mean)} | "
            f"{_f(arm.test_spearman_sd)} | {arm.escaped} | {arm.escaped_sustained} | "
            f"{arm.second_law_eq_0_top100_mean:.2f} | "
            f"{arm.balance_le_1e_2_top100_mean:.2f} | {arm.genesis_top100_mean:.2f} | "
            f"{_f(arm.sign_flip_frac_mean, 3)} |"
        )
    lines.append("")
    lines.append("| contrast | n | mean paired difference | sd |")
    lines.append("|---|---:|---:|---:|")
    for ffc in s.fluxforce_contrasts:
        lines.append(
            f"| {ARM_LABEL.get(ffc.arm, ffc.arm)} minus {ARM_LABEL.get(ffc.against, ffc.against)} "
            f"| {ffc.n_pairs} | {_f(ffc.mean_diff)} | {_f(ffc.sd_diff)} |"
        )
    lines.append(
        f"\nScoring rule: {SCORING_BETAXANTHIN}. Escape is the first epoch beginning "
        "three consecutive epochs with a validation prediction spread above 0.05 of the "
        "target's, and sustained means the last three epochs are also above it. Sign "
        "flips are the mean over the last five epochs of the fraction of flux-carrying "
        "priced reactions whose sign the projection reversed, which only the flux-force "
        "channels report.\n"
    )

    lines.append(
        "**Table 12. Amino-acid paired contrasts, mean per-feature test Spearman over "
        "the 19 Mulleder columns.**\n"
    )
    lines.append("| budget | contrast | seeds | differences | mean | sd | n | t |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|")
    for aac in s.amino_acid_contrasts:
        lines.append(
            f"| {aac.budget} | {AA_ARM_LABEL.get(aac.arm_a, aac.arm_a)} minus "
            f"{AA_ARM_LABEL.get(aac.arm_b, aac.arm_b)} | "
            f"{', '.join(str(x) for x in aac.seeds)} | "
            f"{', '.join(_f(x) for x in aac.differences)} | "
            f"{_f(aac.mean_difference)} | {_f(aac.sd_difference)} | {aac.n_pairs} | "
            f"{_f(aac.t_statistic, 2)} |"
        )
    lines.append(f"\nScoring rule: {SCORING_AMINO_ACID}.\n")

    win = s.window_check
    lines.append(
        "**Table 13. The Mulleder measurements against the shipped concentration "
        "window.**\n"
    )
    lines.append("| quantity | value |")
    lines.append("|---|---:|")
    lines.append(f"| dataset | {win.dataset_name} |")
    lines.append(f"| records, measurements | {win.n_records}, {win.n_measurements} |")
    lines.append(
        f"| shipped window, M | {win.conc_lower_m:.0e} to {win.conc_upper_m:.0e} |"
    )
    lines.append(
        f"| measured range, M | {win.overall_min_m:.3e} to {win.overall_max_m:.3e} |"
    )
    lines.append(
        f"| measurements above the ceiling | {win.frac_measurements_above_upper:.4f} |"
    )
    lines.append(
        f"| measurements below the floor | {win.frac_measurements_below_lower:.4f} |"
    )
    lines.append(
        f"| amino acids exceeding the ceiling at some strain | "
        f"{win.n_amino_acids_exceeding_upper} of 19 |"
    )
    lines.append(
        f"| window the wave-7 flux cells ran, M | {win.recommended_conc_upper_m:.0e} |"
    )
    lines.append(
        f"\nExceeding the ceiling: {', '.join(win.amino_acids_exceeding_upper)}.\n"
    )

    lines.append(
        "**Table 14. The learned whole-cell concentration against the measured one, per "
        "cell, on the pinned test genotypes.**\n"
    )
    lines.append(
        "| cell | seed | supervised | mean per-feature Spearman | mean median absolute "
        "error, decades | per-column Spearman range | at ceiling | at floor |"
    )
    lines.append("|---|---:|---|---:|---:|---|---:|---:|")
    for cg in s.concentration_agreement:
        lines.append(
            f"| {cg.label} | {cg.seed} | {'yes' if cg.supervised else 'no'} | "
            f"{_f(cg.mean_spearman)} | {cg.mean_median_abs_error_decades:.3f} | "
            f"{cg.min_per_column_spearman:+.3f} to {cg.max_per_column_spearman:+.3f} | "
            f"{cg.frac_pred_at_ceiling:.3f} | {cg.frac_pred_at_floor:.3f} |"
        )
    lines.append(
        "\nThe level is supervised in every cell whose supervised column reads yes, so "
        "matching it is fitting; the held-out quantity is the ordering across strains, "
        "which is the Spearman column.\n"
    )

    lines.append(f"**Table 15. Job ledger, {len(s.ledger)} wave-7 cells.**\n")
    lines.append(
        "| stream | cell | job or run | arm | seed | epochs | selected epoch | kcat "
        "table | score | endpoint |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---|---:|---|")
    for lr in s.ledger:
        who = str(lr.job_id) if lr.job_id is not None else f"W&B {lr.run_id}"
        lines.append(
            f"| {lr.stream} | {lr.cell.replace('_', ' ')} | {who} | "
            f"{lr.arm.replace('_', ' ')} | {lr.seed} | {lr.epochs_run} | "
            f"{lr.select_epoch} | {lr.kcat_table} | {_f(lr.score)} | {lr.endpoint} |"
        )
    lines.append(
        "\nBetaxanthin cells score the pinned Merzbacher test Spearman on 629 genes; "
        "mulleder19 cells score the mean per-feature test Spearman over 19 columns on "
        "564 genotypes. Both are read at the validation-selected epoch and neither is a "
        "maximum over epochs. The amino-acid summary carries no slurm id, so those rows "
        "are keyed by their W&B run instead.\n"
    )

    lines.append("**Table 16. Cells a stream expected and this run did not find.**\n")
    lines.append("| stem | named by | landed |")
    lines.append("|---|---|---|")
    for pend in s.pending:
        lines.append(
            f"| {pend.stem} | {pend.named_by} | {'yes' if pend.landed else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    s = build()
    with open(OUT_STEM + ".json", "w") as f:
        f.write(s.model_dump_json(indent=2))
    md = markdown(s)
    with open(OUT_STEM + ".md", "w") as f:
        f.write(md)
    print(md)
    print(f"wrote {OUT_STEM}.json and .md")


if __name__ == "__main__":
    main()
