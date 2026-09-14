#!/bin/bash
# experiments/025-solid-growth/scripts/delta_submit_sweep.sh
# [[experiments.025-solid-growth.scripts.delta_cgt]]
#
# Submit the graph-regularization sweep on Delta, on the 025 build, as one staggered chain.
# The figure it feeds is notes/assets/drawio/FigS-graph-regularization-sweep.drawio: how
# the nine gene-gene graphs enter training, from no penalty (lambda = 0) through the soft
# KL prior to the hard mask.
#
#   RUN ON A DELTA LOGIN NODE, FROM THE 025 WORKTREE ROOT, after delta_preflight_025.sh.
#
#   DRY=1 bash experiments/025-solid-growth/scripts/delta_submit_sweep.sh sweep   # print only
#         bash experiments/025-solid-growth/scripts/delta_submit_sweep.sh sweep   # submit
#   AFTER=<jobid> ...                              # chain the first job after that job
#
# Every arm is cgt_s0_r_kl_ctrl_013 (the constant-rate protocol: learnable table, 010's
# random split, normalizer fit on train, AdamW 2.5e-4, no schedule, 30 epochs) with ONE
# key changed; the three ctrl_013 seeds already queued (22034665 / 22034668 / 22034671,
# 2026-09-12) ARE the ladder's lambda 1e-3 point, so 1e-3 is not resubmitted here.
#
#   lambda in {0, 1e-5, 1e-4, 1e-2, 1e-1, 1}   model.graph_regularization.graph_reg_lambda
#   hard mask, layer 1, nine heads             cgt_s0_r_mask_028
#   random graphs, degree-matched              not yet: needs the rewiring option
#
# Order: seed-major, endpoints first inside a seed (0, mask, then the ladder), so the
# first completed jobs bracket the figure. Each job is chained `after` the previous one
# plus 30 s, the stagger the Taiga-era filelock timeouts needed; with NVMe staging the
# stagger costs nothing and keeps Delta from starting several 554 GB stage-ins on one
# Lustre path at the same instant. `after` (not `afterok`) so one failed job does not
# strand the rest of the chain.
#
# Clock: 24 h. Measured on the staged NVMe path, 27 min/epoch (job 22030924: 30 epochs
# in 16 h 01 m including the stage-in), so 24 h holds 30 epochs with margin and gets
# backfilled ahead of 48 h requests. GPU-hours: 21 jobs x up to 96 = at most 2,016,
# about 1,350 at the measured 16 h, against bfjt-delta-gpu.
set -euo pipefail

MODE="${1:?usage: $0 sweep}"
ACCOUNT="${ACCOUNT:-bfjt-delta-gpu}"
LAUNCHER="experiments/025-solid-growth/scripts/delta_cgt.slurm"
BASE_CONFIG="cgt_s0_r_kl_ctrl_013"
MASK_CONFIG="cgt_s0_r_mask_028"
SEEDS="${SEEDS:-1 2 3}"
LAMBDAS="${LAMBDAS:-1e-2 1e-1 1e-4 1e-5 1}"
HOURS="${HOURS:-24}"
DRY="${DRY:-0}"
AFTER="${AFTER:-}"

[[ -f "$LAUNCHER" ]] || { echo "run from the worktree root: $LAUNCHER not found" >&2; exit 2; }
[[ -f "experiments/025-solid-growth/conf/$MASK_CONFIG.yaml" ]] || { echo "missing $MASK_CONFIG" >&2; exit 2; }

prev="$AFTER"
submit() {  # submit <job-name> <config> [overrides...]
  local name="$1" cfg="$2"; shift 2
  local cmd=(sbatch --parsable --account="$ACCOUNT" --time="${HOURS}:00:00" -J "$name")
  [[ -n "$prev" ]] && cmd+=(--dependency="after:${prev}+30")
  cmd+=("$LAUNCHER" "$cfg" "$@")
  echo "${cmd[*]}"
  if [[ "$DRY" == "1" ]]; then
    prev="DRY-$name"
  else
    prev=$("${cmd[@]}")
    echo "  -> $prev"
  fi
}

case "$MODE" in
  sweep)
    for s in $SEEDS; do
      submit "025-kl-0-nvme-s${s}" "$BASE_CONFIG" model.graph_regularization.graph_reg_lambda=0 +seed="$s"
      submit "025-mask-nvme-s${s}" "$MASK_CONFIG" +seed="$s"
      for lam in $LAMBDAS; do
        submit "025-kl-${lam}-nvme-s${s}" "$BASE_CONFIG" model.graph_regularization.graph_reg_lambda="$lam" +seed="$s"
      done
    done
    ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
echo "last job: $prev"
