#!/bin/bash
#SBATCH -p main
#SBATCH --mem=64g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=010-query-tables-inf3
#SBATCH --time=4:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/slurm/output/%x_%j.out

# INFERENCE_3: Query experimental SMF/DMF data for panel 12 gene tables
# Phase 1: Singles - queries SmfCostanzo2016, SmfKuzmin2018, SmfKuzmin2020
# Phase 2: Doubles - queries DmfCostanzo2016, DmfKuzmin2018, DmfKuzmin2020
#                    + DmiCostanzo2016, DmiKuzmin2018, DmiKuzmin2020

cd /home/michaelvolk/Documents/projects/torchcell || exit 1

source ~/miniconda3/bin/activate
conda activate torchcell

export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"
export PYTHONUNBUFFERED=1

~/miniconda3/envs/torchcell/bin/python \
  experiments/010-kuzmin-tmi/scripts/12_panel_inference_3_queried_data_tables.py
