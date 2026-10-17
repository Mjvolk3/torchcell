#!/bin/bash
# experiments/040-inhibitor-synergy-wetlab/scripts/r11_delta_gate.sh
# [[experiments.040-inhibitor-synergy-wetlab.scripts.r11_delta_gate]]
#
# SUBMIT ROUND 11 ON DELTA ONLY ONCE DELTA CAN RUN IT. Run from GilaHyper, in the 040
# worktree root.
#
# Why this exists: on 2026-10-16 the round-11 smoke (22937918) sat for its whole hour
# without printing a line past the header, and was killed. On the login node the same
# interpreter's `import torch` hung in Lustre's cl_sync_io_wait on the first standard
# library .pyc of /work/hdd/bbub/miniconda3/envs/torchcell, while a single stat of the
# same files returned in 0.2 s; `lfs getstripe` hung as well. A job submitted into that
# hangs to its twelve-hour limit, so 105 of them would spend about 1,260 A40 hours of
# bfjt-delta-gpu on nothing.
#
# Steps, each one gating the next:
#   1  every PROBE_EVERY seconds, ask Delta for `import torch, torchcell` under a
#      90 s timeout, from the tree with its PYTHONPATH, until it returns
#   2  rsync the 040 scripts and conf (the tree on Delta predates a formatting commit)
#   3  submit conf/mixture/r11_smoke.yaml to gpuA40x4-interactive and wait for it
#   4  require COMPLETED, no Traceback, and a "fold 0 seed 0" line for each of the six
#      smoke configs
#   5  run conf/mixture/r11_submit_delta.sh (105 jobs, one GPU each, no dependencies,
#      no -preempt partition, account bfjt-delta-gpu) and print the job ids
#
#   bash experiments/040-inhibitor-synergy-wetlab/scripts/r11_delta_gate.sh
set -u

PROJECT=/scratch/bbub/mjvolk3/torchcell-040
PY=/scratch/bbub/mjvolk3/envs/torchcell-r11/bin/python
EXP=experiments/040-inhibitor-synergy-wetlab
LOG_DIR=$PROJECT/$EXP/slurm/output
PROBE_EVERY=${PROBE_EVERY:-600}
MAX_HOURS=${MAX_HOURS:-36}
SMOKE_CONFIGS="r11smoke_displace_co_doff r11smoke_hill_loewe_anchors r11smoke_hill_doff r11smoke_vanhet_dev r11smoke_head_hsa r11smoke_displace_dev_gene"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=30"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))

# ---- 1. wait for an interpreter that imports ----------------------------------- #
# THE PROBE NEVER HOLDS A SESSION. The first version ran the import inside the ssh
# session; a process blocked in a Lustre read ignores the timeout's signal, so every
# hung probe kept its channel open, and after about twelve probes (2026-10-16, 11:29 to
# 13:17) the shared connection hit its session limit and refused every new session,
# the owner's included. Now the import runs DETACHED on the login node (setsid, stdio
# closed) and writes a marker on success; a separate short session reads the marker,
# and a new probe starts only when no earlier one is still in flight.
MARKER='$HOME/.r11probe_ok'
PROBE="cd $PROJECT && PYTHONPATH=$PROJECT timeout 90 $PY -c 'import torch, torchcell' && date > $MARKER"
timeout 60 $SSH delta "rm -f $MARKER" >/dev/null 2>&1
while true; do
  status=$(timeout 60 $SSH delta "
    if [ -f $MARKER ]; then echo ok
    elif pgrep -u \$USER -f r11probe_ok >/dev/null; then echo inflight
    else setsid nohup sh -c \"$PROBE\" >/dev/null 2>&1 < /dev/null & echo launched
    fi" 2>/dev/null | tail -1)
  if [ "$status" = "ok" ]; then
    echo "$(stamp) probe: import torch, torchcell returned on Delta"
    break
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "$(stamp) GAVE UP: Delta could not import torch within $MAX_HOURS h; nothing submitted"
    exit 2
  fi
  case "$status" in
    launched) echo "$(stamp) probe: launched a detached import" ;;
    inflight) echo "$(stamp) probe: the last import is still blocked" ;;
    *) echo "$(stamp) probe: Delta refused the session (no free session, or the login needs Duo)" ;;
  esac
  sleep "$PROBE_EVERY"
done

# ---- 2. bring the scripts and sweep files up to the committed tree ------------- #
if ! timeout 1200 rsync -rltR -e "$SSH" --exclude '__pycache__' \
    ./$EXP/scripts ./$EXP/conf delta:$PROJECT/; then
  echo "$(stamp) FAILED: rsync of scripts and conf; nothing submitted"
  exit 3
fi
echo "$(stamp) rsync: scripts and conf are current on Delta"

# ---- 3. the smoke --------------------------------------------------------------- #
smoke=$(timeout 150 $SSH delta "cd $PROJECT && mkdir -p $EXP/slurm/output && sbatch --parsable \
  --job-name=040-r11-smoke --partition=gpuA40x4-interactive --account=bfjt-delta-gpu \
  --qos=bfjt-delta-gpu --time=01:00:00 \
  --export=ALL,SWEEP=conf/mixture/r11_smoke.yaml,PARALLEL=1 \
  $EXP/scripts/delta_train_mixture.slurm" | tail -1)
if ! [[ "$smoke" =~ ^[0-9]+$ ]]; then
  echo "$(stamp) FAILED: smoke submission returned '$smoke'; nothing submitted"
  exit 4
fi
echo "$(stamp) smoke submitted as $smoke"
while true; do
  state=$(timeout 150 $SSH delta "sacct -j $smoke -X -n -o State%20 | head -1" 2>/dev/null | xargs)
  case "$state" in
    COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_ME*|NODE_FAIL*|PREEMPTED*) break ;;
  esac
  sleep 120
done
echo "$(stamp) smoke $smoke ended $state"

# ---- 4. every smoke config must have run --------------------------------------- #
log=$(timeout 150 $SSH delta "cat $LOG_DIR/${smoke}_040-r11-smoke.out" 2>/dev/null)
missing=""
for name in $SMOKE_CONFIGS; do
  grep -q "$name fold 0 seed 0:" <<<"$log" || missing="$missing $name"
done
tracebacks=$(grep -c "Traceback" <<<"$log")
if [ "$state" != "COMPLETED" ] || [ -n "$missing" ] || [ "$tracebacks" -ne 0 ]; then
  echo "$(stamp) SMOKE FAILED: state $state, $tracebacks tracebacks, configs without a seed line:$missing"
  grep -nE "Traceback|Error" <<<"$log" | tail -20
  echo "$(stamp) nothing submitted"
  exit 5
fi
echo "$(stamp) smoke passed: all six configs trained and scored"

# ---- 5. round 11 ---------------------------------------------------------------- #
out=$(timeout 1800 $SSH delta "bash $PROJECT/$EXP/conf/mixture/r11_submit_delta.sh" 2>&1)
ids=$(grep -oE "Submitted batch job [0-9]+" <<<"$out" | awk '{print $4}')
n=$(wc -w <<<"$ids")
echo "$(stamp) round 11 submitted: $n jobs, $(head -1 <<<"$ids") to $(tail -1 <<<"$ids")"
grep -vE "Submitted batch job" <<<"$out" | head -10
[ "$n" -eq 105 ] || { echo "$(stamp) WARNING: expected 105 jobs, got $n"; exit 6; }
exit 0
