#!/bin/bash
#SBATCH -p cabbi
#SBATCH --mem=24g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=dango-abl-no_rezero
#SBATCH --time=5-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/a-m/mjvolk3/scratch/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/a-m/mjvolk3/scratch/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out

# experiments/006-kuzmin-tmi/scripts/igb_dango_ablation_no_rezero.slurm
# IGB cabbi variant of gh_dango_ablation_no_rezero.slurm. Written 2026-10-26, NOT submitted.
#
# Before submitting:
# - DATA_ROOT (/home/a-m/mjvolk3/scratch/torchcell) must hold copies of the GilaHyper
#   inputs (GilaHyper DATA_ROOT /scratch/projects/torchcell-scratch), checked by sha256:
#   the 006 side files data/torchcell/experiments/006-kuzmin-tmi/001-small-build/
#   {processed/label_df.parquet, processed/is_any_perturbed_gene_index.json,
#   data_module_cache/index_seed_42.json}, and the genome and graph inputs data/sgd,
#   data/go, data/string (v9.1 for the graphs, v9.1 and v11.0 for the lambda table) and
#   data/tflink. The 006 LMDB is not needed.
# - The checkout must carry fig/dango-si-completion (Dango residual option,
#   scripts/dango_ablation.py, conf/dango_ablation_*.yaml). CODE_ROOT below is the
#   checkout the served IGB DANGO jobs ran from (igb_dango-ddp_string9_1.slurm).
# - torchcell/models/dango.py imports torch_scatter, whose wheel needs GLIBC 2.32 while
#   the IGB hosts run 2.28, so the job runs inside rockylinux_9.sif as the served IGB
#   DANGO jobs did.
# - The compute nodes have no network: W&B runs offline; sync them once from the login
#   node with wandb sync.

CODE_ROOT=/home/a-m/mjvolk3/projects/torchcell

module purge
source ~/.bashrc
module load singularity

cd "$CODE_ROOT" || exit 1

singularity exec --nv rockylinux_9.sif bash -c "
source $HOME/miniconda3/bin/activate
conda activate torchcell
export PYTHONPATH=$CODE_ROOT:\$PYTHONPATH
export DATA_ROOT=/home/a-m/mjvolk3/scratch/torchcell
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
python experiments/006-kuzmin-tmi/scripts/dango_ablation.py --config-name dango_ablation_no_rezero
"
