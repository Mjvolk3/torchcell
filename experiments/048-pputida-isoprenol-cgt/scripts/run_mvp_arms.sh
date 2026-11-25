#!/usr/bin/env bash
# experiments/048-pputida-isoprenol-cgt/scripts/run_mvp_arms.sh
# [[experiments.048-pputida-isoprenol-cgt.scripts.run_mvp_arms]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/048-pputida-isoprenol-cgt/scripts/run_mvp_arms
#
# The MVP arms of experiment 048, in the order the end criterion names them.
# Every arm writes its own results JSON under
# experiments/048-pputida-isoprenol-cgt/results/.
#
# Run from the worktree root. The dataset is built by the first arm and reused by
# the rest, so the arms are deliberately serial rather than parallel.
#
#   bash experiments/048-pputida-isoprenol-cgt/scripts/run_mvp_arms.sh
#
# Criterion 1 is the random split by design, replicated over seeds, because a
# campaign with 44 test designs has a wide enough sampling distribution that one
# seed is not a number.
# Criterion 2 is cycle extrapolation: cycle 6 from cycles 0 to 4, with cycle 5 as
# the selection set.
# Criterion 4 is the fidelity-tier arm, the same splits trained only on the
# post-repair cycles 3 to 6.
# The ablations say what each part is worth: metabolism off, and the sequence
# feature swapped back to the one-hot floor.

set -euo pipefail

PY="${PY:-$HOME/miniconda3/envs/torchcell/bin/python}"
TRAIN="experiments/048-pputida-isoprenol-cgt/scripts/train_isoprenol_cgt.py"
SEEDS="${SEEDS:-42 43 44 45 46}"
EPOCHS="${EPOCHS:-300}"
PATIENCE="${PATIENCE:-40}"
# one_hot is the floor and the only encoder that runs today: 165 of the 5,729
# KT2440 loci are non-coding (75 tRNA, 67 misc RNA, 21 rRNA, 2 other) and carry
# no CDS and no protein, so every CDS-based encoder (codon frequency, CaLM,
# GenSLM, ESM2, ProtT5) needs an explicit policy for them before it can run.
EMB="${EMB:-one_hot}"
WANDB="${WANDB:---wandb}"
# ARM selects one group so the groups can queue as separate slurm jobs;
# "all" runs every group in series, which is the by-hand path.
ARM="${ARM:-all}"

export PYTHONPATH="${PYTHONPATH:-.}"
export NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
export NEO4J_USER="${NEO4J_USER:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-torchcell}"

run () {
  echo "=== $* ==="
  "$PY" "$TRAIN" --epochs "$EPOCHS" --patience "$PATIENCE" \
    --embeddings "$EMB" $WANDB "$@"
}

run_group () {
  local name="$1"
  if [[ "$ARM" != "all" && "$ARM" != "$name" ]]; then return 0; fi
  echo "### arm group: $name"
  shift
  "$@"
}

# Criterion 1: random split by design, over seeds. The seed re-rolls the split
# and the initialization together, which is deliberate: with 44 test designs the
# split is the larger source of variance and is what needs replicating.
arm_design () {
  for seed in $SEEDS; do run --split design --seed "$seed" --tag design; done
}

# The metabolism arm. This is the ablation that is actually an ablation: without
# --flux the cell graph's reaction and metabolite stores are carried but never
# read by the forward pass on this branch (measured: deleting them moves the
# head output by 5.2e-08, the same-input nondeterminism floor), so the only way
# metabolism enters a prediction is through the flux layer.
arm_design_flux () {
  for seed in $SEEDS; do run --split design --seed "$seed" --flux --tag design_flux; done
}

# Criterion 2: cycle extrapolation, cycle 6 from cycles 0 to 4 with cycle 5 for
# selection. The split is deterministic, so the seed re-rolls only the
# initialization.
arm_cycle () {
  for seed in $SEEDS; do run --split cycle --seed "$seed" --tag cycle; done
}

arm_cycle_flux () {
  for seed in $SEEDS; do run --split cycle --seed "$seed" --flux --tag cycle_flux; done
}

# A cycle-5 arm was here and is deliberately gone. Testing on cycle 5 while
# selecting on cycle 4 leaves cycle 6's 59 records in no split, so the partition
# check in `cycle_splits` refuses it, correctly: the arm as written asks for a
# split that does not exist. Cycle 6 is the campaign's last cycle and belongs in
# test, which is what `arm_cycle` already does, so the only ways to make a
# cycle-5 arm valid are to drop cycle 6 (discarding released records to score a
# redundant arm) or to train on it (which stops being extrapolation). Neither is
# worth a card. Job 7612 is the 23-second failure this replaces.

# Criterion 4: the fidelity tier, cycles 3 to 6 only. Note what this means under
# the cycle split, where training is already capped below cycle 5: the tier
# leaves cycles 3 and 4, which is 111 records, so the design split is where this
# arm carries most of its weight.
arm_design_fidelity () {
  for seed in $SEEDS; do
    run --split design --seed "$seed" --fidelity-tier --tag design_fidelity
  done
}

arm_cycle_fidelity () {
  for seed in $SEEDS; do
    run --split cycle --seed "$seed" --fidelity-tier --tag cycle_fidelity
  done
}

# The feature floor: random vectors in place of one-hot, to separate what the
# graph and the architecture contribute from what the gene identity alone does.
arm_design_random () {
  for seed in $SEEDS; do
    run --split design --seed "$seed" --embeddings random_64 --tag design_random
  done
}

run_group design arm_design
run_group design_flux arm_design_flux
run_group cycle arm_cycle
run_group cycle_flux arm_cycle_flux

run_group design_fidelity arm_design_fidelity
run_group cycle_fidelity arm_cycle_fidelity
run_group design_random arm_design_random

echo "arm group ${ARM} complete"
