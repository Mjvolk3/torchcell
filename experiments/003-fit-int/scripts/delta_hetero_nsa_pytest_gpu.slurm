#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bbub-delta-gpu
#SBATCH --job-name=torchcell_tests
#SBATCH --time=2:00:00
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type="END"
#SBATCH --output=/projects/bbub/mjvolk3/torchcell/tests/slurm/output/%x_%j.out
#SBATCH --error=/projects/bbub/mjvolk3/torchcell/tests/slurm/output/%x_%j.out

module reset
source ~/.bashrc

cd /projects/bbub/mjvolk3/torchcell
# Create output directory
mkdir -p /projects/bbub/mjvolk3/torchcell/tests/slurm/output/
pwd
lscpu
nvidia-smi

conda activate /work/hdd/bbub/miniconda3/envs/torchcell

echo "Job starting on $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

mkdir -p /projects/bbub/mjvolk3/torchcell/tests/logs/$SLURM_JOB_ID

# Run the tests
python -m pytest -v tests/torchcell/nn/test_self_attention_block.py
python -m pytest -v tests/torchcell/nn/test_masked_attention_block.py
python -m pytest -v tests/torchcell/nn/test_nsa_encoder.py
python -m pytest -v tests/torchcell/nn/test_hetero_nsa.py
#python -m pytest -v tests/torchcell/models/test_hetero_cell_nsa.py

echo "All tests completed"