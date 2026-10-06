#!/bin/bash
# experiments/019-simb-multimodal/scripts/delta_v21_inplace_node.sh
# [[experiments.019-simb-multimodal.scripts.delta_v21_inplace_node]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/delta_v21_inplace_node
#
# NODE SIDE of the v21 in-place swap (2026-10-06). Runs ON a Delta compute node, in a
# session adopted into ONE running v21 job (delta_v21_inplace_swap.sh gets it there), and
# replaces that job's four slow runs with the same pack on the fast code, inside the same
# allocation:
#
#   1. checks the session really is in job RAW's cgroup (a node can hold two of our jobs);
#   2. holds the job's batch script with SIGSTOP, so the allocation outlives its children;
#   3. stops every python process in that job's cgroup and waits for the card to empty;
#   4. starts delta_expr_v21_small.slurm as a plain script for pack TASK (zero workers,
#      splits in memory), detached, with `scancel RAW` after it so the job ends when the
#      pack does instead of idling to its time limit.
#
# Without --go it only prints what it found. Logs go to slurm-logs/019-expr-v21w2.
#
#   delta_v21_inplace_node.sh RAW TASK [--go]
set -euo pipefail
RAW="${1:?raw job id}"; TASK="${2:?array task (pack index)}"; GO="${3:-}"
R="${V21_PROJECT_ROOT:-/work/hdd/bbub/mjvolk3/torchcell-v11}"
L=/work/hdd/bbub/mjvolk3/slurm-logs/019-expr-v21w2

if ! grep -q "job_${RAW}/" /proc/self/cgroup; then
  echo "FATAL: this session is not in job $RAW's cgroup: $(cat /proc/self/cgroup)" >&2; exit 9
fi
TOP=""
for p in $(pgrep -u "$USER" -f "/var/spool/slurmd/job${RAW}/slurm_script"); do
  parent=$(ps -o ppid= -p "$p" | tr -d ' ')
  if ! ps -o args= -p "$parent" | grep -q "job${RAW}/slurm_script"; then TOP="$p"; fi
done
[ -n "$TOP" ] || { echo "FATAL: no batch script found for job $RAW" >&2; exit 8; }
PYS=()
for p in $(pgrep -u "$USER" -f "envs/torchcell" || true); do
  if grep -q "job_${RAW}/" "/proc/$p/cgroup" 2>/dev/null; then PYS+=("$p"); fi
done
echo "$(hostname -s) job $RAW pack $TASK: batch script pid $TOP ($(ps -o stat= -p "$TOP" | tr -d ' ')), ${#PYS[@]} python processes in the job, card $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
if [[ "$GO" != "--go" ]]; then echo "  dry run, nothing changed"; exit 0; fi

kill -STOP "$TOP"
for p in "${PYS[@]}"; do kill -TERM "$p" 2>/dev/null || true; done
sleep 25
for p in "${PYS[@]}"; do kill -KILL "$p" 2>/dev/null || true; done
for _ in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 500 ] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "$used" -lt 500 ] || { echo "FATAL: card still holds ${used} MiB after stopping the old runs" >&2; exit 7; }

mkdir -p "$L"; cd "$R"
export SLURM_JOB_ID="$RAW" SLURM_ARRAY_JOB_ID="$RAW" SLURM_ARRAY_TASK_ID="$TASK"
export SLURM_JOB_PARTITION=gpuA40x4 SLURM_JOB_ACCOUNT=bfjt-delta-gpu
export V21_STAGE=round V21_WORKERS=0 V21_PROJECT_ROOT="$R" V21_LOG_DIR="$L"
setsid nohup bash -c "bash experiments/019-simb-multimodal/scripts/delta_expr_v21_small.slurm; scancel $RAW" \
  > "$L/inplace_pack${TASK}.out" 2>&1 < /dev/null &
echo "  swapped: fast pack $TASK launching, log $L/inplace_pack${TASK}.out"
