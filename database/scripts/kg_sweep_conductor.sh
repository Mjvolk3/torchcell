#!/usr/bin/env bash
# database/scripts/kg_sweep_conductor.sh
# [[database.scripts.kg_sweep_conductor]]
# https://github.com/Mjvolk3/torchcell/tree/main/database/scripts/kg_sweep_conductor.sh
#
# The full-tree dev-store sweep before a full KG rebuild, as three memory-tiered slurm
# arrays sized from MEASURED peak RSS, under the owner's 384 G cap (2026-10-15: "apply
# the 384 cap", so the metabolic side keeps 129 G, its GPUs and jobs at 32 G or less
# beside the build; nothing is held).
#
#   usage: TORCHCELL_SRC=<checkout at the build tag> [DRY=1] bash database/scripts/kg_sweep_conductor.sh
#
# Tiers (reservation about 1.5x to 3x the measured peak; measured maxima in brackets):
#   xl  peak >= 50 G            -> 160 G, 8 CPU, %1, 24 h, FIRST        [77.0 G, Nadal-Ribelles, 9 min]
#   l   peak >= 14 G or unknown -> 96 G,  8 CPU, %2, 24 h               [38.5 G; longest 4 h 32 min]
#   s   the rest                -> 32 G,  4 CPU, %6, 8 h, after xl      [10.8 G; longest 52 min]
# Peak reservation: l 2 x 96 + s 6 x 32 = 384 G once xl has left; xl 160 + l 192 = 352 G
# before. A store with no measurement goes to l. Measured on arrays 4011, 4457, 4604,
# 4685, 4686, 4687, 4866 and 4897 (sacct MaxRSS and Elapsed, 2026-10-13), the files in
# database/slurm/sweep_measurements/. The 2026-10-13 sweep ran a scratchpad copy of this
# script at l %5 x 64 G and held every other pending job; both are gone under the cap.
#
# Every task is one `torchcell.database.build_dataset_lmdb` build plus `--verify`
# (database/slurm/scripts/gilahyper_build_dataset_lmdbs_array.slurm), independent of the
# others, so recovery is re-running the failed task alone. When every array has left the
# queue, `python -m torchcell.database.build_dataset_lmdb --list-stale --include-private
# --organisms $KG_ORGANISMS` must print nothing before the rebuild is submitted.
set -euo pipefail
: "${TORCHCELL_SRC:?set TORCHCELL_SRC to the checkout at the build tag (never the primary checkout)}"
PY=${PY:-$HOME/miniconda3/envs/torchcell/bin/python}
OUT=${OUT:-/scratch/projects/torchcell/database/slurm}
KG_ORGANISMS=${KG_ORGANISMS:-scerevisiae,ecoli,pputida}
MEAS="$TORCHCELL_SRC/database/slurm/sweep_measurements"
SCRIPT="$TORCHCELL_SRC/database/slurm/scripts/gilahyper_build_dataset_lmdbs_array.slurm"
SHA=$(git -C "$TORCHCELL_SRC" rev-parse --short HEAD)
[ -z "$(git -C "$TORCHCELL_SRC" status --short --untracked-files=no)" ] || { echo "$TORCHCELL_SRC is dirty"; exit 1; }
TAG=$(git -C "$TORCHCELL_SRC" describe --tags --exact-match HEAD 2>/dev/null || true)
if [ -z "$TAG" ]; then
    echo "HEAD $SHA of $TORCHCELL_SRC carries no exact tag; the build gate will refuse it"
    [ "${DRY:-0}" = 1 ] || exit 1
fi
mkdir -p "$OUT"
cd "$TORCHCELL_SRC"
PYTHONPATH="$TORCHCELL_SRC" "$PY" -m torchcell.database.build_dataset_lmdb --list-stale --include-private \
    --organisms "$KG_ORGANISMS" 2>/dev/null > "$OUT/stale_classes_$SHA.txt"
"$PY" - "$OUT/stale_classes_$SHA.txt" "$MEAS/store_rss_recent.tsv" "$MEAS/store_seconds_4011.tsv" "$OUT" "$SHA" <<'PYEOF'
import sys

stale_f, rss_f, sec_f, out, sha = sys.argv[1:6]
stale = [line.strip() for line in open(stale_f) if line.strip()]


def mb(value: str) -> int:
    value = value.strip()
    unit = value[-1] if value[-1] in "KMG" else "K"
    num = float(value[:-1]) if value[-1] in "KMG" else float(value)
    return int(num * {"K": 1 / 1024, "M": 1, "G": 1024}[unit])


rss: dict[str, int] = {}
for line in open(rss_f):
    parts = line.split()
    if len(parts) >= 3:
        rss[parts[0]] = max(rss.get(parts[0], 0), mb(parts[2]))
secs: dict[str, int] = {}
for line in open(sec_f):
    seconds, name = line.split()
    secs[name] = int(seconds)
tiers: dict[str, list[str]] = {"xl": [], "l": [], "s": []}
for name in stale:
    peak = rss.get(name)
    tier = "l" if peak is None else "xl" if peak >= 50_000 else "l" if peak >= 14_000 else "s"
    tiers[tier].append(name)
for tier, names in tiers.items():
    names.sort(key=lambda n: -secs.get(n, 1200))  # longest first, so the tail is short
    with open(f"{out}/sweep_{tier}_{sha}.txt", "w") as fh:
        fh.write("".join(n + "\n" for n in names))
    hours = sum(secs.get(n, 1200) for n in names) / 3600
    longest = max([secs.get(n, 1200) for n in names] + [0]) / 3600
    print(
        f"{tier}: {len(names)} stores, {hours:.1f} measured task-hours "
        f"(unmeasured charged 20 min), longest {longest:.1f} h"
    )
unknown = [n for n in stale if n not in rss]
print(f"unmeasured, placed in l: {len(unknown)} " + " ".join(unknown))
PYEOF
NXL=$(grep -c . "$OUT/sweep_xl_$SHA.txt" || true)
NL=$(grep -c . "$OUT/sweep_l_$SHA.txt" || true)
NS=$(grep -c . "$OUT/sweep_s_$SHA.txt" || true)
# KG_ORGANISMS is a comma list and sbatch splits --export on commas (job 4899 built
# yeast-only that way), so it rides the environment through --export=ALL instead.
export KG_ORGANISMS
EXP="--export=ALL,TORCHCELL_SRC=$TORCHCELL_SRC"
CMD_XL="sbatch --parsable --job-name=kg4-sweep-xl --array=1-${NXL}%1 --cpus-per-task=8 --mem=160G --time=24:00:00 $EXP,CLASSES_FILE=$OUT/sweep_xl_$SHA.txt $SCRIPT"
CMD_L="sbatch --parsable --job-name=kg4-sweep-l --array=1-${NL}%2 --cpus-per-task=8 --mem=96G --time=24:00:00 $EXP,CLASSES_FILE=$OUT/sweep_l_$SHA.txt $SCRIPT"
CMD_S="sbatch --parsable --job-name=kg4-sweep-s --array=1-${NS}%6 --cpus-per-task=4 --mem=32G --time=08:00:00 $EXP,CLASSES_FILE=$OUT/sweep_s_$SHA.txt $SCRIPT"
echo "$CMD_XL"
echo "$CMD_L"
echo "$CMD_S   # with --dependency=afterany:<xl>"
if [ "${DRY:-0}" = 1 ]; then
    echo "DRY=1, nothing submitted"
    exit 0
fi
# FIFO reorder, not a hold: the node schedules by submit order and backfill did not
# start CPU-only jobs behind a GPU-waiting pack (2026-10-16 00:50, lane Y's rebuilds
# planned for 18:03 with 341 G free). Every pending job of ours is held for the seconds
# the three sbatch calls take, then released; a job waiting on a GPU keeps waiting on
# the GPU and loses nothing, and the arrays sit ahead of it in the queue.
HELD="$OUT/sweep_reorder_held_$SHA.txt"; : > "$HELD"
for j in $(squeue -h -u "$USER" -t PD -o "%i %j %R" | grep -v ' kg4-sweep' | grep -v JobHeldUser | awk '{print $1}'); do
    scontrol hold "$j" && echo "$j" >> "$HELD"
done
release_held() { for j in $(cat "$HELD"); do scontrol release "$j" || true; done; echo "released $(wc -l < "$HELD") reordered jobs"; }
trap release_held EXIT
XL=0; L=0; SJ=0
[ "$NXL" -gt 0 ] && { XL=$($CMD_XL); echo "xl=$XL"; }
[ "$NL" -gt 0 ] && { L=$($CMD_L); echo "l=$L"; }
DEP=""
[ "$XL" != 0 ] && DEP="--dependency=afterany:$XL"
[ "$NS" -gt 0 ] && { SJ=$(sbatch --parsable --job-name=kg4-sweep-s --array=1-${NS}%6 --cpus-per-task=4 --mem=32G --time=08:00:00 $DEP $EXP,CLASSES_FILE=$OUT/sweep_s_$SHA.txt "$SCRIPT"); echo "s=$SJ"; }
echo "$XL $L $SJ" > "$OUT/sweep_jobs_$SHA.txt"
echo "sweep submitted at $(date) from $TORCHCELL_SRC ($SHA${TAG:+, $TAG}); job ids in $OUT/sweep_jobs_$SHA.txt"
