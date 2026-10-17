#!/bin/bash
# experiments/042-hpc-smoke-tests/scripts/submit_smoke.sh
# [[experiments.042-hpc-smoke-tests.scripts.submit_smoke]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/042-hpc-smoke-tests/scripts/submit_smoke
#
# Submit a machine's smoke-test job from GilaHyper and print the job id, nothing else,
# so a dispatcher can capture it.  `status` lists the owner's jobs on that machine.
#
#   bash experiments/042-hpc-smoke-tests/scripts/submit_smoke.sh polaris
#   bash experiments/042-hpc-smoke-tests/scripts/submit_smoke.sh polaris status
#
# Scheduler commands only cross to the login node (submit, status): every other
# command, and every poll, belongs on GilaHyper.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/machines.sh" "${1:-}"
action="${2:-submit}"

case "$action" in
  submit)
    tc_ssh "cd '$TC_TREE' && $TC_SUBMIT '$TC_JOBSCRIPT'"
    ;;
  status)
    case "$TC_SUBMIT" in
      qsub) tc_ssh "qstat -u \$USER" ;;
      sbatch) tc_ssh "squeue -u \$USER -o '%.10i %.9P %.22j %.2t %.10M %R'" ;;
    esac
    ;;
  *)
    echo "usage: $0 <machine> [submit|status]" >&2
    exit 2
    ;;
esac
