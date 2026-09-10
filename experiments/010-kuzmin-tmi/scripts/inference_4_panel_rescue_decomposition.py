#!/usr/bin/env python
# experiments/010-kuzmin-tmi/scripts/inference_4_panel_rescue_decomposition.py
# [[experiments.010-kuzmin-tmi.scripts.inference_4_panel_rescue_decomposition]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/inference_4_panel_rescue_decomposition
r"""How much of the panel's rescue ordering is the model, and how much is published data.

WHY THIS EXISTS. The 20-strain panel is presented as a rescue experiment: against the
measured chassis double PDC1 + LAT1, which third deletion best restores growth. The
predicted recoveries order CUP9 > CAD1 > TOS8, and that ordering was described as the
claim the panel tests. But the predicted recovery is not the model's output. It is

    f_abc - f_ab = (f_ab f_c + f_ac f_b + f_bc f_a - 2 f_a f_b f_c - f_ab) + tau_abc
                    \_______________________ measured ______________________/   \model/

and the measured part varies across the three candidates through f_ac and f_bc, which
are published Costanzo and Kuzmin values. This splits the two contributions so the
report can say which one carries the ordering. It turns out the measured offsets span
0.178 while the predicted tau span 0.038, so the published lower rungs alone already
order the three regulators.

It also checks the ordering's robustness properly. Shrinking every predicted tau by a
common factor s does NOT automatically preserve the ordering, because the measured
offsets differ; it preserves it here because each adjacent gap stays positive for all
s >= 0, which is a statement about these particular numbers.

NO RECOMPUTE. This reads the committed design artifacts and does arithmetic on them, so
it never re-runs the 41.9-million-triple scoring or the design search.

Run from repo root:
  python experiments/010-kuzmin-tmi/scripts/inference_4_panel_rescue_decomposition.py

Outputs, under results/inference_4/:
  panel20_rescue_decomposition.csv   one row per chassis-pair triple
  panel20_rescue_decomposition.json  the spreads and the robustness check
"""

import json
import os
import os.path as osp

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results", "inference_4")

CHASSIS_PAIR = "chassis pair"


def main():
    panel = pd.read_csv(osp.join(RESULTS_DIR, "panel20_triples.csv"))
    cp = panel[panel.arm == CHASSIS_PAIR].copy()
    if cp.empty:
        raise SystemExit("no chassis-pair triples in panel20_triples.csv")

    # The chassis pair is the pair of genes common to every triple in this arm.
    chassis = set.intersection(*[{r.name1, r.name2, r.name3} for r in cp.itertuples()])
    if len(chassis) != 2:
        raise SystemExit(f"chassis pair is not two genes: {chassis}")
    cp["third_gene"] = [
        next(iter({r.name1, r.name2, r.name3} - chassis)) for r in cp.itertuples()
    ]

    # Rows do not share a gene ordering, so the chassis double sits in f_ab, f_ac or
    # f_bc depending on the row. Look it up by gene identity rather than by position.
    def chassis_double(r) -> float:
        by_pair = {
            frozenset((r.name1, r.name2)): r.f_ab,
            frozenset((r.name1, r.name3)): r.f_ac,
            frozenset((r.name2, r.name3)): r.f_bc,
        }
        return float(by_pair[frozenset(chassis)])

    f_ab_values = {chassis_double(r) for r in cp.itertuples()}
    if len(f_ab_values) != 1:
        raise SystemExit(f"chassis-pair triples disagree on the double: {f_ab_values}")
    f_ab = f_ab_values.pop()

    cp["measured_offset"] = cp["f_expected"] - f_ab
    cp["model_tau"] = cp["worst"]
    cp["rescue"] = cp["f_triple_predicted_worst"] - f_ab
    cp = cp.sort_values("rescue", ascending=False).reset_index(drop=True)

    residual = (cp.measured_offset + cp.model_tau - cp.rescue).abs().max()
    if residual > 1e-9:
        raise SystemExit(f"decomposition does not close: residual {residual:.2e}")

    offset_spread = float(cp.measured_offset.max() - cp.measured_offset.min())
    tau_spread = float(cp.model_tau.max() - cp.model_tau.min())
    rescue_spread = float(cp.rescue.max() - cp.rescue.min())

    # Adjacent gaps under a common shrinkage s applied to tau only. The ordering is
    # preserved for every s >= 0 exactly when each gap's intercept and slope are both
    # non-negative, and strictly preserved when the intercept is positive.
    gaps = []
    for i in range(len(cp) - 1):
        hi, lo = cp.iloc[i], cp.iloc[i + 1]
        gaps.append(
            {
                "above": hi.third_gene,
                "below": lo.third_gene,
                "intercept_measured": float(hi.measured_offset - lo.measured_offset),
                "slope_model": float(hi.model_tau - lo.model_tau),
            }
        )
    order_robust = all(
        g["intercept_measured"] > 0 and g["slope_model"] >= 0 for g in gaps
    )

    out = cp[
        [
            "triple", "third_gene", "arm", "f_expected", "measured_offset",
            "model_tau", "rescue", "f_triple_predicted_worst",
        ]
    ]
    out.to_csv(osp.join(RESULTS_DIR, "panel20_rescue_decomposition.csv"), index=False)
    print(f"chassis pair {sorted(chassis)}, measured double f_ab = {f_ab:.4f}\n")
    print(out.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    summary = {
        "chassis_pair": sorted(chassis),
        "f_ab": f_ab,
        "measured_offset_spread": offset_spread,
        "model_tau_spread": tau_spread,
        "rescue_spread": rescue_spread,
        "measured_share_of_separation": offset_spread / rescue_spread,
        "adjacent_gaps": gaps,
        "ordering_robust_to_any_tau_shrinkage": bool(order_robust),
        "ordering": list(cp.third_gene),
    }
    with open(osp.join(RESULTS_DIR, "panel20_rescue_decomposition.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(
        f"\nmeasured offsets span {offset_spread:.4f}, predicted tau span "
        f"{tau_spread:.4f}, so the published rungs carry "
        f"{100 * offset_spread / rescue_spread:.0f} percent of the separation"
    )
    print(f"ordering {' > '.join(cp.third_gene)}, robust to any tau shrinkage: "
          f"{order_robust}")
    print(f"\nwrote {RESULTS_DIR}/panel20_rescue_decomposition.{{csv,json}}")


if __name__ == "__main__":
    main()
