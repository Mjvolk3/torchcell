#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/alcf/aurora_env_043_control.sh
# [[experiments.043-expression-proteome-morphology.scripts.alcf.aurora_env_043_control]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/alcf/aurora_env_043_control.sh
#
# The control arm of the round aurora_2026_10_25 stack test: the OLD pip env (torch
# 2.11.0+xpu under the default oneapi/2026.1 module, aurora_env_043.sh, unchanged) with two
# discriminators of the GPU page fault ("Segmentation fault from GPU ... NotPresent, PDE,
# Write, banned: 1", drm_neo.cpp:288):
#   NEOReadDebugKeys=1 DisableScratchPages=0  compute-runtime maps a scratch page under
#     unmapped GPU addresses, so a write to an unmapped page lands there instead of faulting
#     (the key is present in the installed libze_intel_gpu.so.1, 1.6.33578.77). Fewer faults
#     here = the faulting write goes to address space that is not mapped.
#   NUM_WORKERS=0  no DataLoader worker is forked after the XPU context exists (spawn is not
#     wired in batch_loader; data.num_workers is a placement key, so the resume is accepted).
# Sets STACK=pip_mixed (aurora_run_cell_043.sh writes it to wandb.stack).
HERE_CONTROL=/flare/GenesisHackathonOct26/mjvolk3/torchcell/experiments/043-expression-proteome-morphology/scripts/alcf
# shellcheck disable=SC1091
source "$HERE_CONTROL/aurora_env_043.sh"
export STACK=pip_mixed
export NEOReadDebugKeys=1 DisableScratchPages=0
export NUM_WORKERS=0
