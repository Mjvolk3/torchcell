#!/bin/bash
#SBATCH --job-name=044-emapper
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH --time=10:00:00
#SBATCH --no-requeue
#SBATCH --output=experiments/044-organism-similarity/slurm/output/%x_%j.out
#SBATCH --error=experiments/044-organism-similarity/slurm/output/%x_%j.out
# experiments/044-organism-similarity/scripts/gh_04_emapper.slurm
# [[experiments.044-organism-similarity.scripts.gh_04_emapper]]
#
# eggNOG-mapper 2 (diamond mode) over every per-gene proteome in ONE run: the input is
# the concatenated FASTA 03_mmseqs_rbh.py wrote (headers <key>|<protein_id>), so every
# organism is annotated by the same search with the same settings; per-query results do
# not depend on the other queries in the file. 04_emapper.py parse splits by key.
# Needs the eggNOG 5.0.2 data from fetch_eggnog_db.sh. Submit from the worktree root:
#   sbatch experiments/044-organism-similarity/scripts/gh_04_emapper.slurm
echo "Running on $(hostname) at $(date)"
ROOT="$SLURM_SUBMIT_DIR"
cd "$ROOT" || exit 1
set -a; source "$ROOT/.env"; set +a
ORGSIM=/scratch/projects/torchcell-scratch/envs/orgsim
export PATH="$ORGSIM/bin:$PATH"
DB="$DATA_ROOT/data/eggnog"
IN="$DATA_ROOT/data/organism_similarity/mmseqs/all_proteomes.faa"
OUT="$DATA_ROOT/data/organism_similarity/emapper"
mkdir -p "$OUT" "$OUT/tmp"
"$ORGSIM/bin/python" "$ORGSIM/bin/emapper.py" --version --data_dir "$DB"
"$ORGSIM/bin/python" "$ORGSIM/bin/emapper.py" -m diamond --cpu 48 --dbmem \
  --data_dir "$DB" -i "$IN" -o all_proteomes --output_dir "$OUT" \
  --temp_dir "$OUT/tmp" --override || exit 1
echo "finished at $(date)"
