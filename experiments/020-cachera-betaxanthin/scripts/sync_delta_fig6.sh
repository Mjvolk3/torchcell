#!/bin/bash
# experiments/020-cachera-betaxanthin/scripts/sync_delta_fig6.sh
# [[experiments.020-cachera-betaxanthin]]
#
# Mirror everything the Delta metabolism sweeps read, GilaHyper -> Delta (NCSA).
#
#   fig6_pigment_transfer dataset   ~293 MB
#   node-embedding trees            ~1.1 GB   (the sweep uses all EIGHT variants)
#   data/sgd/genome                  ~11 GB   genome + GO + the cached graph pickles
#   data/string                     ~326 MB   six of the nine graphs
#   data/tflink                      ~80 MB   one of the nine
#   data/go                          ~33 MB
#   ---------------------------------------
#   total                           ~12.8 GB
#
# RSYNC IS INCREMENTAL, so this is far cheaper than it looks if the July-2026 fig3_core
# transfer already put the shared trees on Delta -- only `fig6_pigment_transfer` and any
# missing embedding is genuinely new. Run with DRY_RUN=1 first to see what would move.
#
# ONE DUO PROMPT. Delta requires Duo 2FA per SSH connection, so every extra `ssh` is another
# push notification to approve. Two things keep it to one:
#   * `-o ControlMaster` multiplexing -- all rsyncs reuse a single authenticated connection
#   * `--rsync-path="mkdir -p ... && rsync"` -- the destination directory is created inside
#     the rsync's own remote session, instead of a separate `ssh mkdir`
# Approve the ONE prompt when it appears; the rest run over the same socket.
#
#   bash experiments/020-cachera-betaxanthin/scripts/sync_delta_fig6.sh
#   DELTA_HOST=dt-login02.delta.ncsa.illinois.edu bash .../sync_delta_fig6.sh
#   DRY_RUN=1 bash .../sync_delta_fig6.sh        # local inventory only -- no SSH, no Duo
set -euo pipefail

GH_DATA_ROOT="${DATA_ROOT:-/scratch/projects/torchcell-scratch}"
DELTA_USER="${DELTA_USER:-mjvolk3}"
DELTA_HOST="${DELTA_HOST:-login.delta.ncsa.illinois.edu}"
# Confirmed 2026-07-22: Delta's .env DATA_ROOT is the large /work/hdd space, NOT /projects
# (tight quota). run_training resolves the dataset from this.
DELTA_DATA_ROOT="${DELTA_DATA_ROOT:-/work/hdd/bbub/mjvolk3/torchcell}"

# What to move. The dataset tree, plus every embedding the eight-way `node_embeddings` axis
# can sample -- a missing one does not fail at submit time, it fails ~20 minutes into
# whichever trial first draws it.
PATHS=(
  # The dataset itself -- the only tree that is definitely new on Delta.
  "data/torchcell/experiments/019-simb-multimodal/fig6_pigment_transfer"
  # Every embedding the eight-way `node_embeddings` axis can sample. Directory names verified
  # against NodeEmbeddingBuilder's `root_path` entries, not guessed: several axis members
  # share a tree (fudt_upstream/downstream -> fudt_embedding, nt_window_5979 ->
  # nucleotide_transformer_embedding, random_1000 -> random_embedding).
  "data/scerevisiae/protT5_embedding"
  "data/scerevisiae/esm2_embedding"
  "data/scerevisiae/calm_embedding"
  "data/scerevisiae/codon_frequency_embedding"
  "data/scerevisiae/fudt_embedding"
  "data/scerevisiae/nucleotide_transformer_embedding"
  "data/scerevisiae/random_embedding"
  # Genome + GO, and the graph trees behind the nine-graph block: `string` supplies six of
  # the nine, `tflink` one, and physical/regulatory come from the sgd genome tree.
  "data/sgd/genome"
  "data/string"
  "data/tflink"
  "data/go"
)

echo "== fig6 metabolism sync: GilaHyper -> Delta =="
echo "   src root : $GH_DATA_ROOT"
echo "   dest     : $DELTA_USER@$DELTA_HOST:$DELTA_DATA_ROOT"
MISSING=0
for rel in "${PATHS[@]}"; do
  if [[ -e "$GH_DATA_ROOT/$rel" ]]; then
    printf '   %-72s %s\n' "$rel" "$(du -sh "$GH_DATA_ROOT/$rel" | cut -f1)"
  else
    printf '   %-72s %s\n' "$rel" "MISSING"
    MISSING=1
  fi
done
# Refuse rather than transfer a partial set: discovering a missing embedding 20 minutes into
# a trial on a 2-day job wastes far more than stopping here does.
[[ "$MISSING" == 1 ]] && { echo "ERROR: source paths missing (see MISSING above)" >&2; exit 1; }

# DRY_RUN exits HERE, before any SSH. `rsync --dry-run` would still open a connection and
# therefore still fire a Duo push, which defeats the point of a dry run -- the question
# DRY_RUN answers is "what is about to move and how big is it", and that is answerable
# entirely from the local side.
if [[ -n "${DRY_RUN:-}" ]]; then
  echo
  echo "   DRY_RUN set -- local inventory only, no connection opened, no Duo push."
  echo "   Total: $(du -sch "${PATHS[@]/#/$GH_DATA_ROOT/}" 2>/dev/null | tail -1 | cut -f1)"
  exit 0
fi

# Connection multiplexing: one Duo approval for the whole run.
CTL="${TMPDIR:-/tmp}/delta-ssh-%r@%h:%p"
SSH_OPTS=(-o "ControlMaster=auto" -o "ControlPath=$CTL" -o "ControlPersist=10m")
echo
echo "   Delta uses Duo 2FA -- approve the ONE push when it appears."
echo

for rel in "${PATHS[@]}"; do
  dest="$DELTA_DATA_ROOT/$rel"
  echo "-- $rel"
  rsync -aP --human-readable \
    -e "ssh ${SSH_OPTS[*]}" \
    --rsync-path="mkdir -p '$dest' && rsync" \
    "$GH_DATA_ROOT/$rel/" "$DELTA_USER@$DELTA_HOST:$dest/"
done

# Close the shared connection rather than leaving an authenticated socket open for 10m.
ssh "${SSH_OPTS[@]}" -O exit "$DELTA_USER@$DELTA_HOST" 2>/dev/null || true

echo
echo "== done. Verify on Delta:"
echo "   ls $DELTA_DATA_ROOT/data/torchcell/experiments/019-simb-multimodal/fig6_pigment_transfer"
echo "   du -sh $DELTA_DATA_ROOT/data/scerevisiae/*"
