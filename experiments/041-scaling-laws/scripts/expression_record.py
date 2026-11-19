# experiments/041-scaling-laws/scripts/expression_record.py
# [[experiments.041-scaling-laws.scripts.expression_record]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/041-scaling-laws/scripts/expression_record
"""What the expression and proteome rounds of experiment 019 already pin down for a
scaling study, read from their committed result files.

Nothing here is a new measurement. The script reads the 019 result files named below
and typesets the pieces that map onto the axes of the scaling-laws document: the
replicate floor per panel (the bound on E), the budget curve of the long-budget arms
(the compute axis, in Pearson), the loss-against-metric disagreement that rules the
validation loss out as the scaling objective as it stands, the joint-training null
(the multimodal data axis), the sequence-baseline bar and the best arm of the
twelve-seed round.

Reads experiments/019-simb-multimodal/results/
    expression_ceiling_all.json        (expression_ceiling_all.py)
    morphology_noise_ceiling.json      (morphology_noise_ceiling.py)
    short_budget_spread.json           (short_budget_spread.py)
    loss_min_vs_pearson_peak.json      (loss_min_vs_pearson_peak.py)
    joint_checkpoint_readout.json      (joint_checkpoint_readout.py)
    baselines_both_label.json          (baselines_both_label_table.py)
    v21_readout.json                   (v21_readout.py)
    prediction_shrinkage_probe.json    (prediction_shrinkage_probe.py)

Writes notes-tex/modeling/scaling-laws/tables/floors.tex, budget_curve.tex and
expression_numbers.tex (the macros the section's prose quotes).

    python experiments/041-scaling-laws/scripts/expression_record.py
"""

from __future__ import annotations

import json
from pathlib import Path

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[3]
R = REPO / "experiments" / "019-simb-multimodal" / "results"
TABLES = REPO / "notes-tex" / "modeling" / "scaling-laws" / "tables"
SOURCE = "%% SOURCE: experiments/041-scaling-laws/scripts/expression_record.py, reading experiments/019-simb-multimodal/results/"
HEAD = "%% GENERATED FILE -- do not hand-edit.\n"


def load(name: str) -> dict:
    with open(R / name) as f:
        return json.load(f)


def floors_table(ceil: dict, morph: dict) -> str:
    k, c, m = ceil["kemmeren2014"], ceil["caudal2024"], ceil["messner2023"]
    rows = [
        (
            "Kemmeren 2014, expression",
            "test-retest, 82 paired deletions",
            f"{k['test_retest_r_per_gene_mean']:.3f}",
            f"{k['ceiling_test_retest']:.3f}",
        ),
        ("", "decomposition", "--", f"{k['ceiling_decomposition']:.3f}"),
        (
            "Caudal 2024, expression",
            f"test-retest, {c['n_replicate_pairs']} re-cultured isolates",
            f"{c['test_retest']['mean_rel']:.3f}",
            f"{c['test_retest']['ceiling_mean_sqrt']:.3f}",
        ),
        (
            "",
            "decomposition",
            f"{c['decomposition']['mean_rel']:.3f}",
            f"{c['decomposition']['ceiling_mean_sqrt']:.3f}",
        ),
        (
            "Messner 2023, proteome",
            f"test-retest, {m['test_retest_duplicate_origin_strains']['n_duplicated_orfs']} duplicated-origin ORFs",
            f"{m['test_retest_duplicate_origin_strains']['r_per_protein_mean']:.3f}",
            f"{m['test_retest_duplicate_origin_strains']['ceiling']:.3f}",
        ),
        (
            "",
            f"decomposition, {m['decomposition_wt_replicates']['wt_replicates']} wild-type replicates",
            f"{m['decomposition_wt_replicates']['mean_rel']:.3f}",
            f"{m['decomposition_wt_replicates']['ceiling_mean_sqrt']:.3f}",
        ),
        (
            "Ohya 2005, morphology",
            f"decomposition, {morph['n_wt_replicates']} wild-type replicates",
            f"{morph['reliability_mean_model_features']:.3f}",
            f"{morph['ceiling_mean_model_features']:.3f}",
        ),
    ]
    lines = [
        HEAD + SOURCE + "expression_ceiling_all.json and morphology_noise_ceiling.json",
        r"\begin{table}[htbp]",
        r"\centering\footnotesize",
        r"\caption{The replicate ceiling of each 019 panel, the bound on $E$ in Pearson units. Reliability is the"
        r" per-feature fraction of across-strain variance that is signal (test-retest: the correlation of two"
        r" independent measurements across strains; decomposition: one minus replicate variance over total"
        r" variance), averaged over features; the ceiling is the mean of its square root, the best across-strain"
        r" Pearson a perfect predictor of the signal can reach against a single noisy measurement. Two estimators"
        r" are given where both exist because they disagree where few pairs determine either.}",
        r"\label{tab:floors}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"panel & estimator & reliability & ceiling, mean Pearson per feature \\",
        r"\midrule",
    ]
    for a, b, c_, d in rows:
        lines.append(f"{a} & {b} & {c_} & {d} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def budget_table(sb: dict) -> str:
    lines = [
        HEAD + SOURCE + "short_budget_spread.json (W&B " + sb["project"] + ")",
        r"\begin{table}[htbp]",
        r"\centering\footnotesize",
        r"\caption{The compute axis as the 019 campaign measured it: the same eight long-budget arms of the"
        r" v9 round rescored on the prefix of their own curves at each budget (the maximum of a centered"
        r" five-epoch rolling mean of the validation Pearson per feature over epochs up to the budget). Mean"
        r" and sd over the eight arms; the spread is an arm spread plus nondeterminism, so it bounds the"
        r" replicate spread from above. The statistic is Pearson, not the loss, for the reason given in"
        r" the text.}",
        r"\label{tab:budget-curve}",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"epochs & arms & mean Pearson per feature & sd & min & max \\",
        r"\midrule",
    ]
    for b in sb["by_budget"]:
        lines.append(
            f"{b['budget_epochs']:,} & {b['n']} & {b['mean']:.4f} & {b['sd']:.4f} & {b['min']:.4f} & {b['max']:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def macros(
    sb: dict, lm: dict, joint: dict, base: dict, v21: dict, shrink: dict, morph: dict
) -> str:
    live = lm["live"]
    t = joint["v19"]["tests"]
    ex, pr = t["H1a_expression_superiority"], t["H1b_proteome_noninferiority"]
    bar = max(r["val_mean"] for r in base["rows"] if r["head"] == "expression")
    bar_row = max(
        (r for r in base["rows"] if r["head"] == "expression"),
        key=lambda r: r["val_mean"],
    )
    arms = [
        a for a in v21["arms"] if a["head"] == "expression" and a["window_all"]["n"]
    ]
    best = max(arms, key=lambda a: a["window_all"]["mean"])
    ref = next(a for a in arms if a["arm"] == "S_ref")
    # The probe holds both heads; the expression checkpoints predict about 6,100
    # features, the proteome ones about 1,850. The heads disperse oppositely.
    rows = [r for r in shrink["rows"] if r["n_features"] > 3000]
    prot = [r for r in shrink["rows"] if r["n_features"] <= 3000]
    first, last = sb["by_budget"][0], sb["by_budget"][-1]
    at1000 = next(b for b in sb["by_budget"] if b["budget_epochs"] == 1000)
    m = {
        "budgetFirstEpochs": f"{first['budget_epochs']:,}",
        "budgetFirstMean": f"{first['mean']:.3f}",
        "budgetLastEpochs": f"{last['budget_epochs']:,}",
        "budgetLastMean": f"{last['mean']:.3f}",
        "budgetThousandMean": f"{at1000['mean']:.3f}",
        "budgetGainPastThousand": f"{last['mean'] - at1000['mean']:.3f}",
        "lossLiveRuns": str(lm["n_live"]),
        "lossMinEpochMedian": f"{live['epoch_loss_min_median']:,.0f}",
        "lossMinEpochMax": f"{live['epoch_loss_min_max']:,.0f}",
        "lossPeakEpochMedian": f"{live['epoch_pearson_peak_median']:,.0f}",
        "lossFinalExcessPct": f"{100 * live['loss_final_rel_excess_median']:.1f}",
        "lossAtPeakExcessPct": f"{100 * live['loss_at_pearson_peak_rel_excess_median']:.1f}",
        "lossGainAfterMin": f"{live['pearson_gain_after_loss_min_median']:+.3f}",
        "jointExprDiff": f"{ex['mean_diff']:+.4f}",
        "jointExprPos": str(ex["n_positive"]),
        "jointExprN": str(ex["n_partitions"]),
        "jointExprRef": f"{ex['ref_mean']:.3f}",
        "jointProtDiff": f"{pr['mean_diff']:+.4f}",
        "jointProtPos": str(pr["n_positive"]),
        "jointProtMargin": f"{pr['margin']:.3f}",
        "exprBar": f"{bar:.3f}",
        "exprBarSd": f"{bar_row['val_sd']:.3f}",
        "exprBarSeeds": str(bar_row["n_seeds"]),
        "vBestArm": best["arm"].replace("_", r"\_"),
        "vBestMean": f"{best['window_all']['mean']:.3f}",
        "vBestSd": f"{best['window_all']['sd']:.3f}",
        "vRefMean": f"{ref['window_all']['mean']:.3f}",
        "vRefSd": f"{ref['window_all']['sd']:.3f}",
        "shrinkSpreadMin": f"{min(r['spread_over_r'] for r in rows):.1f}",
        "shrinkSpreadMax": f"{max(r['spread_over_r'] for r in rows):.1f}",
        "shrinkRuns": str(len(rows)),
        "shrinkProtSpreadMin": f"{min(r['spread_over_r'] for r in prot):.1f}",
        "shrinkProtSpreadMax": f"{max(r['spread_over_r'] for r in prot):.1f}",
        "shrinkProtRuns": str(len(prot)),
        "morphRealized": f"{100 * morph['fraction_of_ceiling_realized']:.0f}",
        "morphObservedBest": f"{morph['observed_best']['roll_max']:.3f}",
    }
    out = [HEAD + SOURCE + "the files named in the script docstring"]
    for k, v in m.items():
        out.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    return "\n".join(out) + "\n"


def main() -> None:
    ceil, morph = (
        load("expression_ceiling_all.json"),
        load("morphology_noise_ceiling.json"),
    )
    sb, lm = load("short_budget_spread.json"), load("loss_min_vs_pearson_peak.json")
    joint, base = (
        load("joint_checkpoint_readout.json"),
        load("baselines_both_label.json"),
    )
    v21, shrink = load("v21_readout.json"), load("prediction_shrinkage_probe.json")
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, text in (
        ("floors.tex", floors_table(ceil, morph)),
        ("budget_curve.tex", budget_table(sb)),
        ("expression_numbers.tex", macros(sb, lm, joint, base, v21, shrink, morph)),
    ):
        (TABLES / name).write_text(text)
        print("wrote", TABLES / name)


if __name__ == "__main__":
    main()
