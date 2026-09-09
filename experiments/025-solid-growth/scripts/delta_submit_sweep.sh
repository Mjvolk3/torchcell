#!/bin/bash
# experiments/025-solid-growth/scripts/delta_submit_sweep.sh
# [[experiments.025-solid-growth.scripts.delta_cgt]]
#
# Submit the graph-regularization sweep on Delta: how the nine gene-gene graphs enter
# training, from no penalty (lambda = 0) through the soft KL prior to the hard mask.
#
#   RUN ON A DELTA LOGIN NODE, FROM THE REPO ROOT, after delta_preflight_025.sh passes.
#
#   bash experiments/025-solid-growth/scripts/delta_submit_sweep.sh canary   # ONE 24 h KL run, seed 1
#   bash experiments/025-solid-growth/scripts/delta_submit_sweep.sh sweep    # everything below
#   bash experiments/025-solid-growth/scripts/delta_submit_sweep.sh disjoint # KL on 010's disjoint split
#
# Arms (3 seeds each unless noted; every run is one gpuA40x4 node, 4-GPU DDP):
#   lambda in {0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1}   soft KL, layer 1     24 h  (0 and mask: 12 h)
#   hard mask, layer 1                               config 006           12 h
#   disjoint KL, lambda 1e-3                         config 007           24 h
# 1 is included because 010's coefficient carried a x367 defect, so its effective weight was
# about 0.37; the sweep has to bracket the run the paper's number came from.
#
# GPU-hours charge per GPU, 4 per node-hour: 21 KL runs x 96 h + 3 lambda-0 x 48 + 3 mask x 48
# = 2,304 GPU-h for the sweep, plus 288 for the disjoint arm.
#
# Run the CANARY first and read minutes per epoch off its log before submitting the rest:
# GilaHyper does a KL epoch in 19 min on RTX 6000 Ada, IGB in 58, and Delta A40s with zero
# dataloader workers have not been measured.
set -euo pipefail

MODE="${1:?usage: $0 canary|sweep|disjoint}"
ACCOUNT="${ACCOUNT:-bfjt-delta-gpu}"
LAUNCHER="experiments/025-solid-growth/scripts/delta_cgt.slurm"
SEEDS="${SEEDS:-1 2 3}"
LAMBDAS="${LAMBDAS:-0 1e-5 1e-4 1e-3 1e-2 1e-1 1}"
DRY="${DRY:-0}"

submit() {  # submit <job-name> <hours> <config> [overrides...]
  local name="$1" hours="$2" cfg="$3"; shift 3
  local cmd=(sbatch --account="$ACCOUNT" --time="${hours}:00:00" -J "$name" "$LAUNCHER" "$cfg" "$@")
  echo "${cmd[*]}"
  [[ "$DRY" == "1" ]] || "${cmd[@]}"
}

case "$MODE" in
  canary)
    submit 025-canary-kl-1e-3-s1 24 cgt_010b_r_kl_005 model.graph_regularization.graph_reg_lambda=1e-3 seed=1
    ;;
  sweep)
    for s in $SEEDS; do
      for lam in $LAMBDAS; do
        hours=24; [[ "$lam" == "0" ]] && hours=12
        submit "025-kl-${lam}-s${s}" "$hours" cgt_010b_r_kl_005 model.graph_regularization.graph_reg_lambda="$lam" seed="$s"
      done
      submit "025-mask-s${s}" 12 cgt_010b_r_mask_006 seed="$s"
    done
    ;;
  disjoint)
    for s in $SEEDS; do
      submit "025-qkl-1e-3-s${s}" 24 cgt_010b_q_kl_007 model.graph_regularization.graph_reg_lambda=1e-3 seed="$s"
    done
    ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
