#!/bin/bash
# experiments/019-simb-multimodal/scripts/delta_v21_inplace_swap.sh
# [[experiments.019-simb-multimodal.scripts.delta_v21_inplace_swap]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/delta_v21_inplace_swap
#
# LOGIN-NODE SIDE of the v21 in-place swap (2026-10-06). The first submission of v21 (jobs
# 22690625 to 22690651, one pack per job) runs at 177 s per epoch because the eval-mode
# train pass respawns a DataLoader worker every tenth epoch and a spawned worker re-imports
# the stack from Lustre; 1,200 epochs would take 59 h against a 48 h limit. The fast code
# runs the same pack at about 45 s per epoch (pack 0, swapped by hand at 03:58, W&B runs
# iuqwl2p8, tdnhpkid, fcq0xkke, 0jajrluv). Resubmitting through the queue had a start
# estimate a day out, so the swap happens INSIDE each running job.
#
# For each RAW:TASK it opens a step in that job (srun --jobid --overlap) and from there an
# ssh to the same node, which the node's pam_slurm_adopt places in THAT job's cgroup even
# when the node holds two of our jobs (verified read-only on gpub092), then runs
# delta_v21_inplace_node.sh there.
#
#   bash delta_v21_inplace_swap.sh            dry run over every running v21 pack but pack 0
#   bash delta_v21_inplace_swap.sh --go       do it
#   bash delta_v21_inplace_swap.sh --go 22690626:1 22690627:2     only these
set -euo pipefail
GO=""; [[ "${1:-}" == "--go" ]] && { GO="--go"; shift; }
R="${V21_PROJECT_ROOT:-/work/hdd/bbub/mjvolk3/torchcell-v11}"
NODE_SH="$R/experiments/019-simb-multimodal/scripts/delta_v21_inplace_node.sh"
if [[ $# -gt 0 ]]; then JOBS=("$@"); else
  mapfile -t JOBS < <(squeue -u "$USER" -h -t RUNNING -o "%A:%K %j" | awk '$2 ~ /^019-v21w1-small-p/ && $1 != "22690625:0" {print $1}' | sort)
fi
echo "${#JOBS[@]} packs: ${JOBS[*]}"
for jt in "${JOBS[@]}"; do
  RAW="${jt%%:*}"; TASK="${jt##*:}"
  srun --jobid="$RAW" --overlap --ntasks=1 bash -c \
    "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \$(hostname -s) bash $NODE_SH $RAW $TASK $GO" \
    || echo "  FAILED on $jt"
done
