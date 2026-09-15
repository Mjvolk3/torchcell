# experiments/026-metabolism-flux/scripts/probe_cell_build_verify.py
# [[experiments.026-metabolism-flux.physics-gates-2026-09-14]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/probe_cell_build_verify
r"""Does the REAL build path reproduce the init check the cell was designed on?

WHY THIS EXISTS AND WHY IT IS NOT REDUNDANT. Every number behind the wave-9 warm-start cell
was measured by `probe_warmstart_init.py`, which reaches its k_cat rules by overwriting
`layer.kcat_per_h` after construction. A training cell cannot do that: it names an arm, and
026's `build_flux_layer` resolves the table from `parameters.resolve_kcat_table`. Those are
two different code paths to what should be the same table, and "should be" is not a
measurement.

This builds the layer the way the CELL builds it, through the arm dict the 027 trainer
assembles, and re-reads the five statistics and the three watched ceilings. Agreement is the
check; disagreement means the cell was designed on a table it will not run.

THE HASH SEED IS PART OF THE CHECK. The kinetics owner measured that the resolved table
depends on `PYTHONHASHSEED`, 2 of 3,728 values moving by up to 0.313 decades, so this script
refuses to run without `PYTHONHASHSEED=0` exactly as the kinetics scripts do. A verification
that passed under one seed and a cell that trained under another would be comparing two
tables.

Run from the repo root, on CPU::

    PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
        ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/probe_cell_build_verify.py

Writes ``experiments/026-metabolism-flux/results/probe_cell_build_verify.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time

import torch
from dotenv import load_dotenv

load_dotenv()

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
SCRIPTS_DIR = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "scripts")
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
BX_CONF = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "conf")
sys.path.insert(0, SCRIPTS_DIR)

from probe_warmstart_init import (  # noqa: E402
    TURNOVER_THRESHOLD,
    _gene_ids,
    genotype_boxes,
    reference_fit,
    statistics,
    watched_capacity,
)
from train_flux import ARMS, build_flux_layer  # noqa: E402

#: The row this must reproduce: job 1937's `sigma_plus_2` + gate fix.
REFERENCE_ROW = {
    "balance_active_max": 0.0306,
    "balance_active_p95": 0.0019,
    "balance_active_rows": 81.8,
    "flux_active_frac": 0.1774,
    "proposal_distance_rel": 0.018,
}
#: Relative tolerance on the comparison. The hash seed moves 2 of 3,728 k_cat values by up
#: to 0.313 decades even when it is pinned on both sides, and the probe row was measured
#: through the buffer-overwrite path, so an exact match is not the right test.
TOLERANCE = 0.05


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arm-file", default="arm_flux_warmstart_boxforce_nominal.yaml")
    parser.add_argument("--out", default="probe_cell_build_verify.json")
    args = parser.parse_args()

    seed = os.environ.get("PYTHONHASHSEED")
    if seed != "0":
        raise SystemExit(
            f"PYTHONHASHSEED is {seed!r}, not '0'. The resolved k_cat table depends on it "
            "(2 of 3,728 values move by up to 0.313 decades), so a verification run under "
            "a different seed is checking a different table."
        )

    started = time.time()
    torch.set_num_threads(8)
    import yaml

    with open(osp.join(BX_CONF, args.arm_file)) as handle:
        arm_file = yaml.safe_load(handle)
    # EXACTLY WHAT 027's TRAINER ASSEMBLES: the registry entry the arm file names, with the
    # channel string overridden and the run's own kcat tier and thermo route written in.
    arm = dict(ARMS[arm_file["base_arm"]])
    arm["parameterization"] = arm_file["flux_parameterization"]
    arm["kcat_tier"] = "predicted"
    arm["thermo_source"] = "union_external"

    gene_ids = _gene_ids()
    torch.manual_seed(args.seed)
    layer, meta = build_flux_layer(arm, gene_ids, args.hidden)
    assert layer is not None
    # The three switches the trainer sets on the built layer, in its order.
    layer.config.availability_fold = "normalized_softmin"
    layer.apply_thermo_exempt_list(
        osp.join(RESULTS_DIR, "thermo_exempt_candidates.json")
    )
    layer.apply_log_c_init(osp.join(RESULTS_DIR, "thermo_consistent_log_c.json"))

    torch.manual_seed(args.seed)
    h = torch.randn(args.batch, len(gene_ids), args.hidden)
    context = torch.randn(args.batch, args.hidden)
    empty = torch.zeros(0, dtype=torch.long)
    with torch.no_grad():
        out = layer(h, context, empty, empty)
    stats = statistics(layer, out)
    watched = watched_capacity(layer, h, context)
    fit = reference_fit(layer, h, context)
    with torch.no_grad():
        _, _, tight_lb, tight_ub, capacity, _ = genotype_boxes(layer, h, context)
        capacity_median = float(capacity[:, layer.has_gpr].median())

    deltas = {
        key: {
            "measured": stats[key],
            "expected": value,
            "relative_delta": (
                abs(stats[key] - value) / abs(value) if value else float("nan")
            ),
        }
        for key, value in REFERENCE_ROW.items()
    }
    agrees = all(row["relative_delta"] <= TOLERANCE for row in deltas.values())

    payload = {
        "definitions": {
            "why": (
                "the probe reaches its k_cat rules by overwriting layer.kcat_per_h after "
                "construction; a cell names an arm and resolves the table through "
                "parameters.resolve_kcat_table. This checks the two agree."
            ),
            "tolerance": (
                f"relative {TOLERANCE}. The resolved table depends on PYTHONHASHSEED and "
                "the reference row came from the other code path, so an exact match is not "
                "the right test."
            ),
        },
        "python_hash_seed": seed,
        "arm_file": args.arm_file,
        "base_arm": arm_file["base_arm"],
        "channel": arm_file["flux_parameterization"],
        "complex_aggregation": meta["complex_aggregation"],
        "kcat_ceiling_sigma_k": meta["kcat_ceiling_sigma_k"],
        "kcat_tier_counts": meta["kcat_tier_counts"],
        "coverage_thermo_exempt_file": layer.coverage_report()["thermo_exempt_file"],
        "coverage_log_c_init_file": layer.coverage_report()["log_c_init_file"],
        "availability_fold": layer.coverage_report()["availability_fold"],
        "capacity_median_over_gpr_reactions": capacity_median,
        "turnover_threshold": TURNOVER_THRESHOLD,
        "statistics": stats,
        "watched_reactions": watched,
        "reference_fit": fit,
        "comparison_to_job_1937": deltas,
        "agrees_within_tolerance": agrees,
    }
    out_path = osp.join(RESULTS_DIR, args.out)
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(stats, indent=2))
    for row in watched:
        print(
            f"  {row['reaction']:8s} v {row['reference_flux']:8.3f} "
            f"capacity {row['capacity']:10.4f} ceiling {row['box_ceiling']:9.4f} "
            f"excess {row['excess']:.3f} held {row['held']}"
        )
    for key, row in deltas.items():
        print(
            f"  {key:26s} measured {row['measured']:.4g} expected {row['expected']:.4g} "
            f"relative delta {row['relative_delta']:.4f}"
        )
    print(f"agrees within {TOLERANCE}: {agrees}")
    print(f"wrote {out_path} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
