#!/bin/bash
#SBATCH --mem=64g # 64g works for a single-GPU ESM2 embedding job (matches compute_esm2)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bbub-delta-gpu
#SBATCH --job-name=embed-isolate
#SBATCH --time=48:00:00      # hh:mm:ss for the job, 48 hr max.
#SBATCH --constraint="projects&scratch"
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none     # <- or closest
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type="END"
# #SBATCH --output=/dev/null
#SBATCH --output=/projects/bbub/mjvolk3/torchcell/experiments/embeddings/slurm/output/%x_%j.out
#SBATCH --error=/projects/bbub/mjvolk3/torchcell/experiments/embeddings/slurm/output/%x_%j.out

# Per-isolate ESM2 embeddings for the Caudal 2024 natural isolates (WS9 / Fig 4).
#
# Pass the isolate id via the ISOLATE env var (keep any SACE_ prefix, e.g. SACE_YAV):
#   sbatch --export=ALL,ISOLATE=SACE_YAV experiments/embeddings/compute_isolate_embeddings.slurm
#
# Or drive a whole panel with a job array over an isolate list file
# (one canonical strain id per line), reading the ISOLATE from that line:
#   sbatch --array=1-943 --export=ALL,ISOLATE_LIST=experiments/embeddings/isolates.txt \
#          experiments/embeddings/compute_isolate_embeddings.slurm

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
source ~/.bashrc
cd /projects/bbub/mjvolk3/torchcell
pwd
lscpu
nvidia-smi
cat /proc/meminfo
module list  # job documentation and metadata
conda activate /projects/bbub/miniconda3/envs/torchcell

# Resolve the isolate: an explicit ISOLATE wins; otherwise read line
# $SLURM_ARRAY_TASK_ID from $ISOLATE_LIST (array driver).
if [ -z "${ISOLATE}" ] && [ -n "${ISOLATE_LIST}" ]; then
    ISOLATE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${ISOLATE_LIST}")
fi
if [ -z "${ISOLATE}" ]; then
    echo "ERROR: set ISOLATE=<strain_id> (or ISOLATE_LIST + --array)"; exit 1
fi

echo "Starting per-isolate embedding computation for ${ISOLATE}..."
python experiments/embeddings/compute_isolate_embeddings.py \
    --isolate "${ISOLATE}" \
    --esm2-model esm2_t33_650M_UR50D
