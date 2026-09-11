# experiments/027-betaxanthin-metabolic/scripts/overnight_summary_2026_09_11.py
# [[experiments.027-betaxanthin-metabolic.scripts.overnight_summary_2026_09_11]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/overnight_summary_2026_09_11.py

"""One table per question for the 2026-09-10 to 2026-09-11 overnight metabolic-module run.

Reads only committed result files and writes ``results/overnight_summary_2026_09_11.json``
plus a markdown twin whose tables are pasted verbatim into the overnight note
([[experiments.027-betaxanthin-metabolic.overnight-2026-09-11]]). Nothing here is computed
from a log; every number is read from a results JSON that its own script wrote.

Sources:

* ``026-metabolism-flux/results/kinetic_coverage_closure.json`` (kinetic_coverage_gaps.py,
  improve_kinetic_coverage.py)
* ``026-metabolism-flux/results/reaction_thermo_coverage_final.json`` (thermo_pipeline_yeast9.py)
* ``026-metabolism-flux/results/pka_method_selection.json`` (external_pka_method_selection.py)
* ``027-betaxanthin-metabolic/results/wave6_summary.json`` (analyze_wave6.py)
* ``027-betaxanthin-metabolic/results/wave6_proj_*.json`` (train_bx.py, one per cell)

The scoring rule is the 027 rule: test Spearman on the pinned Merzbacher genes at the
validation-selected epoch, never a maximum over epochs; a paired difference is reported with
its across-seed sd and n. The Genesis fraction is the share of the top 100 predicted strains
with balance median ratio <= 1e-2, no box violation, and no second-law violation under the
model's own reaction energies, as defined in the trainer's FeasibilityBlock.
"""

from __future__ import annotations

import glob
import json
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
OUT_STEM = osp.join(R27, "overnight_summary_2026_09_11")

GENESIS_KEY = "both_balance_le_1e-02_and_thermo_model_eq_0"


class ProjectionCell(BaseModel):
    """One projection-head cell, read from its own results JSON."""

    result_file: str
    arm: str
    seed: int
    constraint_scale: float
    thermo_source: str
    kcat_table: str
    select_epoch: int
    test_spearman_pinned: float
    genesis_fraction_top100: float | None
    balance_le_1e2_top100: float | None
    box_eq_0_top100: float | None
    thermo_model_eq_0_top100: float | None
    balance_median_top100: float | None


class OvernightSummary(BaseModel):
    """Every number the overnight note quotes, read from the stream result files."""

    generated_at: str
    kinetic_tiers: dict[str, dict[str, int]]
    kinetic_default_units_reachable_with_open_tools: int
    thermo_ceiling_ladder: list[dict[str, Any]]
    thermo_reasons_permanent: dict[str, int]
    pka_selection: dict[str, Any]
    betaxanthin_parity_six_seeds: dict[str, Any]
    betaxanthin_parity_same_table: dict[str, Any]
    amino_acid_parity: dict[str, Any]
    amino_acid_projection_vs_pooled: dict[str, Any]
    projection_cells: list[ProjectionCell]
    pooled_by_seed: dict[int, float]


def _kcat_table_label(tiers: dict[str, int] | None) -> str:
    """The fingerprint rule from kinetic_table_versions.json, by tier counts."""
    if not tiers:
        return "unknown"
    key = (
        tiers.get("measured"),
        tiers.get("homolog", 0),
        tiers.get("predicted"),
        tiers.get("default"),
    )
    return {
        (148, 0, 3570, 10): "round 1",
        (96, 782, 2820, 30): "interim",
        (97, 782, 2831, 18): "shipped",
    }.get(key, "other")


def _projection_cells() -> list[ProjectionCell]:
    cells: list[ProjectionCell] = []
    for path in sorted(glob.glob(osp.join(R27, "wave6_proj_*.json"))):
        if osp.basename(path).startswith("wave6_projection_summary"):
            continue
        with open(path) as f:
            d = json.load(f)
        if "test" not in d or "cell" not in d:
            continue
        top = ((d.get("feasibility") or {}).get("groups") or {}).get("all_top100") or {}
        fr = top.get("fractions") or {}
        st = top.get("stats") or {}
        cells.append(
            ProjectionCell(
                result_file=osp.basename(path),
                arm=d["cell"]["arm"],
                seed=int(d["cell"]["seed"]),
                constraint_scale=float(d["cell"].get("constraint_scale", 0.0)),
                thermo_source=str(d["cell"].get("thermo_source", "")),
                kcat_table=_kcat_table_label(
                    (d.get("flux_meta") or {}).get("kcat_tier_counts")
                ),
                select_epoch=int(d["select_epoch"]),
                test_spearman_pinned=float(d["test"]["pinned"]["spearman"]),
                genesis_fraction_top100=fr.get(GENESIS_KEY),
                balance_le_1e2_top100=fr.get("balance_median_ratio_le_1e-02"),
                box_eq_0_top100=fr.get("box_violation_frac_eq_0"),
                thermo_model_eq_0_top100=fr.get("thermo_model_violation_frac_eq_0"),
                balance_median_top100=(st.get("balance_median_ratio") or {}).get(
                    "median"
                ),
            )
        )
    return cells


def _pooled_by_seed(w6: dict[str, Any]) -> dict[int, float]:
    out: dict[int, float] = {}
    for name, cell in w6["per_cell"].items():
        if cell["arm"] == "pooled" and cell["head"] == "betaxanthin":
            out[int(cell["seed"])] = float(cell["score"])
    return out


def _contrast(block: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "arm_a",
        "arm_b",
        "seeds",
        "a_values",
        "b_values",
        "differences",
        "n_pairs",
        "mean_difference",
        "sd_difference",
        "t_statistic",
        "a_mean",
        "a_sd",
        "b_mean",
        "b_sd",
    )
    return {k: block[k] for k in keys if k in block}


def build() -> OvernightSummary:
    with open(osp.join(R26, "kinetic_coverage_closure.json")) as f:
        kin = json.load(f)
    with open(osp.join(R26, "reaction_thermo_coverage_final.json")) as f:
        thermo = json.load(f)
    with open(osp.join(R26, "pka_method_selection.json")) as f:
        pka = json.load(f)
    with open(osp.join(R27, "wave6_summary.json")) as f:
        w6 = json.load(f)

    tiers: dict[str, dict[str, int]] = {}
    for param, block in kin["parameters"].items():
        # The shipped variant is the last one the closure script records, the verified
        # tables read with the substrate-matched measured key and the homolog tier.
        chosen = block["variants"]["verified_substrate_key_with_homolog"]
        tiers[param] = dict(chosen["tier_counts"])

    permanent = {
        k: int(v)
        for k, v in thermo["reactions_by_sole_blocking_reason"].items()
        if k in ("pseudo_metabolite", "placeholder_formula", "placeholder_structure")
    }
    ladder = [
        {
            k: r[k]
            for k in (
                "name",
                "n_reactions_priced",
                "fraction_reactions_priced",
                "n_flux_carrying_priced",
                "fraction_flux_carrying_priced",
                "what_it_would_take",
            )
        }
        for r in thermo["ceiling_ladder"]["rungs"]
    ]
    pka_sel = {
        k: pka[k]
        for k in (
            "n_evaluated",
            "working_ph",
            "selected",
            "selected_kind",
            "selected_members",
            "selected_worst_working_p90",
            "selected_worst_working_ph",
            "selected_failure_rate",
            "selected_median_ph72",
        )
    }
    pka_sel["rule_statistic"] = pka["rule"]["statistic"]
    pka_sel["max_failure_rate"] = pka["rule"]["max_failure_rate"]

    return OvernightSummary(
        generated_at=datetime.now(UTC).isoformat(),
        kinetic_tiers=tiers,
        kinetic_default_units_reachable_with_open_tools=int(
            kin["ceiling"]["n_reachable_with_open_source_tools"]
        ),
        thermo_ceiling_ladder=ladder,
        thermo_reasons_permanent=permanent,
        pka_selection=pka_sel,
        betaxanthin_parity_six_seeds=_contrast(w6["betaxanthin_parity_heterogeneous"]),
        betaxanthin_parity_same_table=_contrast(w6["betaxanthin_parity"]),
        amino_acid_parity=_contrast(w6["amino_acid_parity"]),
        amino_acid_projection_vs_pooled=_contrast(w6["amino_acid_proj_vs_pooled"]),
        projection_cells=_projection_cells(),
        pooled_by_seed=_pooled_by_seed(w6),
    )


#: Display names for the arm strings, so a table column never carries a code identifier
#: wider than its cell. The config string stays in the JSON twin.
ARM_LABEL = {
    "flux_projection": "projection, balance end",
    "flux_projection_anchored": "anchored, balance end",
    "flux_projection_anchored_k20": "anchored K20, balance end",
    "flux_projection_anchored_k20_boxend": "anchored K20, box end",
    "flux_projection_anchored_tdg": "anchored, sign-tightened",
    "flux_projection_anchored_tdg_stochastic": "anchored, sign-tightened, stochastic",
    "flux_projection_anchored_tdgm5": "anchored, sign-tightened 5 kJ/mol margin",
    "flux_projection_stochastic": "projection, stochastic",
}


def _fmt(x: float | None, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float) and abs(x) < 1e-3 and x != 0:
        return f"{x:.1e}"
    return f"{x:.{nd}f}"


def markdown(s: OvernightSummary) -> str:
    lines: list[str] = []
    lines.append(
        "**Kinetic tiers, 3,728 catalytic units (kinetic_coverage_closure.json).**\n"
    )
    lines.append("| parameter | measured | homolog | predicted | default |")
    lines.append("|---|---:|---:|---:|---:|")
    for p, t in s.kinetic_tiers.items():
        lines.append(
            f"| {p} | {t['measured']} | {t.get('homolog', 0)} | {t['predicted']} | {t['default']} |"
        )
    lines.append(
        f"\nDefault units reachable with an open-source tool: {s.kinetic_default_units_reachable_with_open_tools}.\n"
    )

    lines.append(
        "**Reaction energy coverage ladder, yeast-GEM 9.0.2 (reaction_thermo_coverage_final.json).**\n"
    )
    lines.append(
        "| rung | reactions priced | fraction | flux-carrying priced | fraction | what it would take |"
    )
    lines.append("|---|---:|---:|---:|---:|---|")
    for r in s.thermo_ceiling_ladder:
        lines.append(
            f"| {r['name'].replace('_', ' ')} | {r['n_reactions_priced']} | {r['fraction_reactions_priced']:.4f} | "
            f"{r['n_flux_carrying_priced']} | {r['fraction_flux_carrying_priced']:.4f} | {r['what_it_would_take']} |"
        )
    perm = ", ".join(f"{k} {v}" for k, v in s.thermo_reasons_permanent.items())
    lines.append(f"\nPermanently closed reactions by sole reason: {perm}.\n")

    p = s.pka_selection
    lines.append("**pKa method selection (pka_method_selection.json).**\n")
    lines.append(
        f"Rule: {p['rule_statistic']}, failure rate at most {p['max_failure_rate']}, evaluated on "
        f"{p['n_evaluated']} compounds at pH {', '.join(str(x) for x in p['working_ph'])}. "
        f"Selected `{p['selected']}` ({p['selected_kind']}: {' then '.join(p['selected_members'])}); "
        f"worst working-pH p90 {p['selected_worst_working_p90']:.3f} kJ/mol at pH {p['selected_worst_working_ph']}, "
        f"failure rate {p['selected_failure_rate']:.4f}, median at pH 7.2 {p['selected_median_ph72']:.3f} kJ/mol.\n"
    )

    for title, c in (
        (
            "Betaxanthin parity, box flux head at constraint scale 0 vs pooled, six seeds (three kcat table versions)",
            s.betaxanthin_parity_six_seeds,
        ),
        (
            "Betaxanthin parity, seeds on one kcat table",
            s.betaxanthin_parity_same_table,
        ),
        (
            "Amino-acid parity, box flux head vs pooled (mean per-feature Spearman, 19 features)",
            s.amino_acid_parity,
        ),
        ("Amino-acid, projection head vs pooled", s.amino_acid_projection_vs_pooled),
    ):
        lines.append(f"**{title}.**\n")
        lines.append(
            "| seed | "
            + " | ".join(str(x) for x in c["seeds"])
            + " | mean | sd | n | t |"
        )
        lines.append("|---|" + "---:|" * len(c["seeds"]) + "---:|---:|---:|---:|")
        lines.append(
            f"| {c['arm_a']} | "
            + " | ".join(_fmt(x, 4) for x in c["a_values"])
            + f" | {_fmt(c['a_mean'], 4)} | {_fmt(c['a_sd'], 4)} | | |"
        )
        lines.append(
            f"| {c['arm_b']} | "
            + " | ".join(_fmt(x, 4) for x in c["b_values"])
            + f" | {_fmt(c['b_mean'], 4)} | {_fmt(c['b_sd'], 4)} | | |"
        )
        lines.append(
            "| difference | "
            + " | ".join(_fmt(x, 4) for x in c["differences"])
            + f" | {_fmt(c['mean_difference'], 4)} | {_fmt(c['sd_difference'], 4)} | {c['n_pairs']} | {_fmt(c['t_statistic'], 2)} |"
        )
        lines.append("")

    lines.append(
        "**Projection-head cells on betaxanthin (one results JSON each; pooled at the same seed for reference).**\n"
    )
    lines.append(
        "| arm | seed | scale | thermo source | kcat table | selected epoch | test Spearman | pooled at seed | Genesis top-100 | balance <= 1e-2 | box exact | second law exact | balance median |"
    )
    lines.append("|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in s.projection_cells:
        lines.append(
            f"| {ARM_LABEL.get(c.arm, c.arm)} | {c.seed} | {c.constraint_scale:g} | {c.thermo_source.replace('union_external', 'union+ext')} | {c.kcat_table} | {c.select_epoch} | "
            f"{c.test_spearman_pinned:.4f} | {_fmt(s.pooled_by_seed.get(c.seed), 4)} | {_fmt(c.genesis_fraction_top100, 2)} | "
            f"{_fmt(c.balance_le_1e2_top100, 2)} | {_fmt(c.box_eq_0_top100, 2)} | {_fmt(c.thermo_model_eq_0_top100, 2)} | "
            f"{_fmt(c.balance_median_top100)} |"
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
