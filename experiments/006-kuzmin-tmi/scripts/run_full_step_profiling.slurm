#!/bin/bash
#SBATCH -p main
#SBATCH --mem=128g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=PROFILE-FULL-STEP
#SBATCH --time=2:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out

# Change to torchcell root directory (use absolute path)
cd /home/michaelvolk/Documents/projects/torchcell || exit 1

# Set torchcell root for use in paths
TORCHCELL_ROOT="/home/michaelvolk/Documents/projects/torchcell"

# Create timestamped output directory (relative to torchcell root)
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
OUTPUT_DIR="experiments/006-kuzmin-tmi/profiling_results/full_step_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

echo "================================================================================"
echo "COMPLETE TRAINING STEP PROFILING"
echo "================================================================================"
echo "Working directory: $(pwd)"
echo "Output directory: ${OUTPUT_DIR}"
echo "Timestamp: ${TIMESTAMP}"
echo "================================================================================"

# Add project root to Python path
export PYTHONPATH="${TORCHCELL_ROOT}:${PYTHONPATH}"

# Set Python unbuffered for immediate output
export PYTHONUNBUFFERED=1

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate torchcell

echo ""
echo "================================================================================"
echo "STEP 1/3: Profiling DANGO Model"
echo "================================================================================"
echo "This will measure forward + loss + backward + optimizer for DANGO"
echo "Expected time: ~10 minutes"
echo ""

python experiments/006-kuzmin-tmi/scripts/profile_dango_full_step.py \
  --output_dir ${OUTPUT_DIR} \
  --batch_size 8 \
  --iterations 50 \
  --warmup 5

if [ $? -ne 0 ]; then
    echo "ERROR: DANGO profiling failed"
    exit 1
fi

echo ""
echo "================================================================================"
echo "STEP 2/3: Profiling Lazy Hetero Model"
echo "================================================================================"
echo "This will measure forward + loss + backward + optimizer for Lazy Hetero"
echo "Expected time: ~15-20 minutes (slower due to expanded graphs)"
echo ""

python experiments/006-kuzmin-tmi/scripts/profile_gene_interaction_dango_full_step.py \
  --output_dir ${OUTPUT_DIR} \
  --batch_size 8 \
  --iterations 50 \
  --warmup 5

if [ $? -ne 0 ]; then
    echo "ERROR: Lazy Hetero profiling failed"
    exit 1
fi

echo ""
echo "================================================================================"
echo "STEP 3/3: Generating Comparison Report"
echo "================================================================================"
echo "Creating side-by-side comparison of DANGO vs Lazy Hetero"
echo ""

# Find the generated profile files
DANGO_FILE=$(ls ${OUTPUT_DIR}/profile_dango_full_step_*.txt | head -n 1)
LAZY_FILE=$(ls ${OUTPUT_DIR}/profile_gene_interaction_dango_full_step_*.txt | head -n 1)

if [ -z "$DANGO_FILE" ] || [ -z "$LAZY_FILE" ]; then
    echo "ERROR: Could not find profile files"
    echo "DANGO file: $DANGO_FILE"
    echo "Lazy Hetero file: $LAZY_FILE"
    exit 1
fi

python experiments/006-kuzmin-tmi/scripts/compare_full_step_profiles.py \
  --dango_profile ${DANGO_FILE} \
  --lazy_profile ${LAZY_FILE} \
  --output_dir ${OUTPUT_DIR}

if [ $? -ne 0 ]; then
    echo "ERROR: Comparison generation failed"
    exit 1
fi

echo ""
echo "================================================================================"
echo "PROFILING COMPLETE!"
echo "================================================================================"
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Files generated:"
ls -lh ${OUTPUT_DIR}
echo ""
echo "To view the comparison:"
echo "  cat ${OUTPUT_DIR}/comparison_full_step_*.txt"
echo ""
echo "To view individual profiles:"
echo "  cat ${OUTPUT_DIR}/profile_dango_full_step_*.txt"
echo "  cat ${OUTPUT_DIR}/profile_gene_interaction_dango_full_step_*.txt"
echo "================================================================================"

# Exit with success
exit 0
