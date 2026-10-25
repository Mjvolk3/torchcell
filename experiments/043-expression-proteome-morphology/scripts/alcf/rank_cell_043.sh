#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/alcf/rank_cell_043.sh
# [[experiments.043-expression-proteome-morphology.scripts.alcf.rank_cell_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/alcf/rank_cell_043.sh
#
# One MPI rank of a Polaris prod job: runs on card PALS_LOCAL_RANKID and CLAIMS cells from
# polaris_cells_043.tsv, in table order, until none is left to claim, then exits.
#
# Claim design (all on eagle, shared by every job of the lane):
#   $CLAIMS/<run id>.done   the cell is finished (every stage); never claimed again.
#   $CLAIMS/<run id>.lock/  a mkdir lock (atomic on Lustre) holding `owner` (job, host, rank,
#                           card) and `heartbeat`, touched every 5 min while the cell runs.
#   A lock whose heartbeat is older than 15 min is stale (its job was killed): it is renamed
#   aside into $CLAIMS/stale/ (rename is atomic, so two ranks cannot both take it) and the
#   cell is claimed again. A finished or failed cell's lock is left in place; its heartbeat
#   stops and it goes stale, so a failed cell is retried by a later job, never at once.
#   The W&B live check inside run_cell_043.sh is the second guard against a double start.
set -euo pipefail
: "${PALS_RANKID:?run under mpiexec (Cray PALS)}" "${PALS_LOCAL_RANKID:?}"
TC_PREFIX=/eagle/GenesisHackathonOct26/mjvolk3
TC_TREE=$TC_PREFIX/torchcell
export PY=$TC_PREFIX/miniforge3/envs/torchcell/bin/python
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export HTTP_PROXY=$http_proxy HTTPS_PROXY=$https_proxy
export no_proxy=localhost,127.0.0.1,.alcf.anl.gov
cd "$TC_TREE"
set -a; source .env; set +a
export DATA_ROOT=$TC_PREFIX/torchcell-data/043
export EXPERIMENT_ROOT=$TC_TREE/experiments
export PYTHONUNBUFFERED=1 PYTHONPATH="$TC_TREE"
export WANDB_MODE=online
SHIM=$(mktemp -d)
REAL_GIT=$(command -v git)
SYNCED=$(cut -c1-9 "$TC_TREE/.synced_commit")
cat > "$SHIM/git" <<SH
#!/bin/bash
if [ "\$*" = "rev-parse --short HEAD" ]; then echo "$SYNCED"; exit 0; fi
exec "$REAL_GIT" "\$@"
SH
chmod +x "$SHIM/git"
export PATH="$SHIM:$PATH"
export CUDA_VISIBLE_DEVICES=$PALS_LOCAL_RANKID

ALCF=experiments/043-expression-proteome-morphology/scripts/alcf
OUT=$TC_TREE/experiments/043-expression-proteome-morphology/slurm/output/onto_grid
CLAIMS=$DATA_ROOT/experiments/043-expression-proteome-morphology/claims_polaris_lane
mkdir -p "$OUT" "$CLAIMS/stale"
me="job ${PBS_JOBID:-?} host $(hostname) rank $PALS_RANKID card $CUDA_VISIBLE_DEVICES"
STALE_MIN=15

run_id_of() {  # run_id_of <cell> <overrides...>: the W&B run id run_cell_043.sh will use
  local cell=$1 profile=ontology prefix="" o
  shift
  for o in "$@"; do
    case "$o" in profile=*) profile=${o#profile=} ;; rid_prefix=*) prefix=${o#rid_prefix=} ;; esac
  done
  case "$profile" in
    ontology) echo "${prefix:-o1023}_${cell}" ;;
    aurora) echo "${prefix:-a1024b}_${cell}" ;;
  esac
}
claim() {  # claim <run id>: 0 = this rank now owns the cell
  local lock=$CLAIMS/$1.lock
  if mkdir "$lock" 2>/dev/null; then echo "$me $(date -u +%FT%TZ)" > "$lock/owner"; touch "$lock/heartbeat"; return 0; fi
  # stale only when neither the lock directory (created by mkdir, touched when owner and
  # heartbeat are written) nor its heartbeat changed within STALE_MIN: a lock seen between
  # another rank's mkdir and its first heartbeat is fresh by its own mtime
  if [ -d "$lock" ] && [ -z "$(find "$lock" "$lock/heartbeat" -maxdepth 0 -mmin -"$STALE_MIN" 2>/dev/null)" ]; then
    mv "$lock" "$CLAIMS/stale/$1.$(date +%s).$PALS_RANKID" 2>/dev/null || return 1
    mkdir "$lock" 2>/dev/null || return 1
    echo "$me $(date -u +%FT%TZ) (took a stale claim)" > "$lock/owner"; touch "$lock/heartbeat"
    return 0
  fi
  return 1
}

declare -A tried=()
while true; do
  mapfile -t ROWS < <(grep -v '^#' "$ALCF/polaris_cells_043.tsv")
  picked=""
  for row in "${ROWS[@]}"; do
    IFS=$'\t' read -r cell finetune overrides <<< "$row"
    # shellcheck disable=SC2086
    rid=$(run_id_of "$cell" $overrides)
    [ -e "$CLAIMS/$rid.done" ] && continue
    [ -n "${tried[$rid]:-}" ] && continue
    if claim "$rid"; then picked=$rid; break; fi
  done
  [ -n "$picked" ] || { echo "[$me] no claimable cell left at $(date)"; exit 0; }
  tried[$picked]=1
  # detached from stdout/stderr, so its sleep never holds mpiexec output open after a kill
  ( while sleep 300; do touch "$CLAIMS/$picked.lock/heartbeat"; done ) </dev/null >/dev/null 2>&1 &
  hb=$!
  echo "[$me] claimed $picked at $(date)" >> "$OUT/$cell.log"
  rc=0
  # shellcheck disable=SC2086
  CELL_DONE="$CLAIMS/$picked.done" bash "$ALCF/run_cell_043.sh" "$cell" "$finetune" $overrides \
    >> "$OUT/$cell.log" 2>&1 || rc=$?
  kill "$hb" 2>/dev/null || true
  echo "[$me] $picked ended rc $rc at $(date)" | tee -a "$OUT/$cell.log"
done
