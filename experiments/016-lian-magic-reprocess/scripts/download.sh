#!/bin/bash
# Download + fastq-convert the 21 Lian MAGIC SRA runs (18 furfural + 3 plasmid libs).
# Idempotent: skips runs whose .fastq.gz already exists.
set -uo pipefail
SRA="$HOME/miniconda3/envs/lian-sra/bin"
WORK="${LIAN_WORK:?set LIAN_WORK}"
export PATH="$SRA:$PATH"
cd "$WORK/sra"

runs=$(tail -n +2 "$WORK/run_manifest.tsv" | cut -f1)
for r in $runs; do
  if compgen -G "$WORK/sra/${r}*.fastq.gz" > /dev/null; then
    echo "SKIP $r (fastq.gz exists)"; continue
  fi
  echo "=== $(date +%T) prefetch $r ==="
  if ! "$SRA/prefetch" -q -O "$WORK/sra" --max-size 12g "$r"; then
    echo "FAIL prefetch $r"; continue
  fi
  echo "=== $(date +%T) fasterq-dump $r ==="
  if ! "$SRA/fasterq-dump" -q -e 8 -O "$WORK/sra" "$WORK/sra/$r/$r.sra"; then
    echo "FAIL fasterq-dump $r"; continue
  fi
  gzip -f "$WORK/sra/$r"*.fastq
  rm -rf "$WORK/sra/$r/"
  echo "DONE $r  $(du -sh $WORK/sra/${r}*.fastq.gz 2>/dev/null | cut -f1)"
done
echo "ALL_DOWNLOADS_COMPLETE $(date +%T)"
