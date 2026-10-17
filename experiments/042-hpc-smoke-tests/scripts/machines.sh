#!/bin/bash
# experiments/042-hpc-smoke-tests/scripts/machines.sh
# [[experiments.042-hpc-smoke-tests.scripts.machines]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/042-hpc-smoke-tests/scripts/machines
#
# One table of where torchcell lives on each remote machine, sourced by the other
# scripts in this directory:  `source machines.sh polaris` exports TC_*.
#
#   TC_SSH_HOST   ssh destination (an alias from ~/.ssh/config, or user@host)
#   TC_SSH_OPTS   extra ssh options (identity file, BatchMode); may be empty
#   TC_PREFIX     project-space dir that holds miniforge3/, the code tree, the data
#   TC_TREE       the rsynced torchcell checkout the jobs run from
#   TC_DATA_ROOT  DATA_ROOT on that machine
#   TC_ACCEL      wheel flavor: cu128 (NVIDIA) or xpu (Intel Max)
#   TC_SUBMIT     scheduler submit command
#   TC_JOBSCRIPT  the smoke-test job script for that machine (relative to TC_TREE)
#
# Access is the owner's: ALCF through a ControlMaster socket opened with a MobilePASS+
# passcode (10 h), NERSC through the 24 h sshproxy certificate in ~/.ssh/nersc.
set -euo pipefail

machine="${1:-}"
case "$machine" in
  polaris)
    export TC_SSH_HOST="polaris"
    export TC_SSH_OPTS=""
    export TC_PREFIX="/eagle/GenesisHackathonOct26/mjvolk3"
    export TC_ACCEL="cu128"
    export TC_SUBMIT="qsub"
    export TC_JOBSCRIPT="experiments/042-hpc-smoke-tests/scripts/polaris_smoke.pbs"
    ;;
  aurora)
    export TC_SSH_HOST="aurora"
    export TC_SSH_OPTS=""
    export TC_PREFIX="/flare/GenesisHackathonOct26/mjvolk3"
    export TC_ACCEL="xpu"
    export TC_SUBMIT="qsub"
    export TC_JOBSCRIPT="experiments/042-hpc-smoke-tests/scripts/aurora_smoke.pbs"
    ;;
  perlmutter)
    # The sshproxy cert is passed explicitly so a script never falls back to a
    # password prompt.
    export TC_SSH_HOST="mjvolk3@perlmutter.nersc.gov"
    export TC_SSH_OPTS="-i $HOME/.ssh/nersc -o BatchMode=yes"
    export TC_PREFIX="/global/cfs/cdirs/m5429/mjvolk3"
    export TC_ACCEL="cu128"
    export TC_SUBMIT="sbatch"
    export TC_JOBSCRIPT="experiments/042-hpc-smoke-tests/scripts/perlmutter_smoke.slurm"
    ;;
  *)
    echo "usage: source machines.sh {polaris|aurora|perlmutter}" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac
export TC_MACHINE="$machine"
export TC_TREE="$TC_PREFIX/torchcell"
export TC_DATA_ROOT="$TC_PREFIX/torchcell-data"
export TC_PY="$TC_PREFIX/miniforge3/envs/torchcell/bin/python"

# ssh wrapper: `tc_ssh <remote command>`
tc_ssh() {
  # shellcheck disable=SC2086
  ssh $TC_SSH_OPTS "$TC_SSH_HOST" "$@"
}
