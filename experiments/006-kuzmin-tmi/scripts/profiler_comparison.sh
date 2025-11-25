#!/bin/bash
# profiler_comparison.sh
# Run profiling for both dango.py and hetero_cell_bipartite_dango_gi.py
# Then compare the results

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory and base directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="/home/michaelvolk/Documents/projects/torchcell"
EXPERIMENT_DIR="$BASE_DIR/experiments/006-kuzmin-tmi"
SCRIPTS_DIR="$EXPERIMENT_DIR/scripts"
CONF_DIR="$EXPERIMENT_DIR/conf"

# Change to scripts directory for execution
cd "$SCRIPTS_DIR"

# Load environment variables
source ~/.bashrc

# Initialize conda for this shell session
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate torchcell
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate torchcell
else
    echo "Warning: Could not find conda initialization script. Assuming environment is already active."
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}PyTorch Profiler Comparison Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Start time: $(date)"
echo ""

# Function to check if GPU is available
check_gpu() {
    echo -e "${YELLOW}Checking GPU availability...${NC}"
    python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
    echo ""
}

# Function to verify config settings
verify_configs() {
    echo -e "${YELLOW}Verifying configuration settings...${NC}"

    # Check hetero config
    echo "Checking hetero_cell_bipartite_dango_gi.yaml:"
    grep -A2 "profiler:" "$CONF_DIR/hetero_cell_bipartite_dango_gi.yaml"
    grep "max_epochs:" "$CONF_DIR/hetero_cell_bipartite_dango_gi.yaml" | head -1
    grep "devices:" "$CONF_DIR/hetero_cell_bipartite_dango_gi.yaml" | grep -v "#" | head -1
    grep "perturbation_subset_size:" "$CONF_DIR/hetero_cell_bipartite_dango_gi.yaml" | head -1
    echo ""

    # Check dango config
    echo "Checking dango_kuzmin2018_tmi.yaml:"
    grep -A2 "profiler:" "$CONF_DIR/dango_kuzmin2018_tmi.yaml"
    grep "max_epochs:" "$CONF_DIR/dango_kuzmin2018_tmi.yaml" | head -1
    grep "devices:" "$CONF_DIR/dango_kuzmin2018_tmi.yaml" | grep -v "#" | head -1
    grep "perturbation_subset_size:" "$CONF_DIR/dango_kuzmin2018_tmi.yaml" | head -1
    echo ""
}

# Function to run profiling
run_profiling() {
    local script_name=$1
    local description=$2

    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Running $description${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo "Start time: $(date)"

    # Set CUDA_VISIBLE_DEVICES to use only GPU 0 for clean profiling
    export CUDA_VISIBLE_DEVICES=0

    # Run the script
    ~/miniconda3/envs/torchcell/bin/python "$script_name" 2>&1 | tee "${script_name%.py}_profile.log"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $description completed successfully${NC}"
    else
        echo -e "${RED}✗ $description failed${NC}"
        return 1
    fi

    echo "End time: $(date)"
    echo ""

    # Give system a moment to write all files
    sleep 5
}

# Function to find latest profile outputs
find_profile_outputs() {
    echo -e "${YELLOW}Searching for profile outputs...${NC}"

    # Load dotenv to get DATA_ROOT
    if [ -f "/home/michaelvolk/Documents/projects/torchcell/.env" ]; then
        export $(grep -v '^#' /home/michaelvolk/Documents/projects/torchcell/.env | xargs)
    fi

    # Find the most recent profile directories
    DATA_ROOT="${DATA_ROOT:-/scratch/projects/torchcell}"
    PROFILE_DIR="$DATA_ROOT/data/torchcell/experiments/006-kuzmin-tmi/profiler_output"

    if [ -d "$PROFILE_DIR" ]; then
        echo "Profile directory: $PROFILE_DIR"

        # Find latest dango profile
        DANGO_PROFILE=$(find "$PROFILE_DIR" -type f -path "*dango_*" -name "*.json" ! -name "*.pt.trace.json" -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2)

        # Find latest hetero profile
        HETERO_PROFILE=$(find "$PROFILE_DIR" -type f -path "*hetero_dango_gi_*" -name "*.json" ! -name "*.pt.trace.json" -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2)

        if [ -n "$DANGO_PROFILE" ]; then
            echo -e "${GREEN}Found Dango profile: $DANGO_PROFILE${NC}"
        else
            echo -e "${RED}No Dango profile found${NC}"
        fi

        if [ -n "$HETERO_PROFILE" ]; then
            echo -e "${GREEN}Found HeteroCell profile: $HETERO_PROFILE${NC}"
        else
            echo -e "${RED}No HeteroCell profile found${NC}"
        fi
    else
        echo -e "${RED}Profile directory not found: $PROFILE_DIR${NC}"
    fi
    echo ""
}

# Function to run comparison
run_comparison() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Running Profile Comparison Analysis${NC}"
    echo -e "${BLUE}========================================${NC}"

    # Run the comparison script with auto-find
    ~/miniconda3/envs/torchcell/bin/python compare_profiler_outputs.py --auto-find 2>&1 | tee profile_comparison_results.txt

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Profile comparison completed successfully${NC}"
        echo -e "${GREEN}Results saved to: profile_comparison_results.txt${NC}"
    else
        echo -e "${RED}✗ Profile comparison failed${NC}"
    fi
    echo ""
}

# Function to cleanup old logs (optional)
cleanup_logs() {
    echo -e "${YELLOW}Cleaning up old log files...${NC}"
    # Keep only the 5 most recent log files for each type
    ls -t *_profile.log 2>/dev/null | tail -n +6 | xargs -r rm
    ls -t profile_comparison_results*.txt 2>/dev/null | tail -n +6 | xargs -r rm
    echo "Cleanup complete"
    echo ""
}

# Main execution
main() {
    echo -e "${BLUE}Starting profiler comparison pipeline...${NC}"
    echo "Working directory: $(pwd)"
    echo ""

    # Step 1: Check environment
    check_gpu

    # Step 2: Verify configurations
    verify_configs

    # Step 3: Optional cleanup of old logs
    read -t 10 -p "Clean up old log files? (y/N, auto-skip in 10s): " -n 1 -r || true
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cleanup_logs
    fi

    # Step 4: Run HeteroCell profiling
    echo -e "${YELLOW}Phase 1: Profile HeteroCell model${NC}"
    if ! run_profiling "hetero_cell_bipartite_dango_gi.py" "HeteroCell Bipartite Dango GI Profiling"; then
        echo -e "${RED}✗ HeteroCell profiling failed. Continue anyway? (y/N)${NC}"
        read -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    # Step 5: Run Dango profiling
    echo -e "${YELLOW}Phase 2: Profile Dango model${NC}"
    if ! run_profiling "dango.py" "Dango Profiling"; then
        echo -e "${RED}✗ Dango profiling failed. Continue anyway? (y/N)${NC}"
        read -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    # Step 6: Find generated profiles
    echo -e "${YELLOW}Phase 3: Locate profile outputs${NC}"
    find_profile_outputs

    # Step 7: Run comparison analysis
    echo -e "${YELLOW}Phase 4: Analyze and compare profiles${NC}"
    run_comparison

    # Step 8: Summary
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Profiling Pipeline Complete!${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo "End time: $(date)"
    echo ""
    echo "Output files:"
    echo "  - hetero_cell_bipartite_dango_gi_profile.log"
    echo "  - dango_profile.log"
    echo "  - profile_comparison_results.txt"
    echo "  - Profile JSONs in: $DATA_ROOT/profiler_output/"
    echo "  - Comparison plots in: $ASSET_IMAGES_DIR/"
    echo ""
    echo -e "${GREEN}To view profiles in Chrome:${NC}"
    echo "  1. Open Chrome and navigate to: chrome://tracing"
    echo "  2. Click 'Load' and select the JSON files from $DATA_ROOT/profiler_output/"
    echo ""
    echo -e "${GREEN}To view in TensorBoard:${NC}"
    echo "  tensorboard --logdir=$DATA_ROOT/profiler_output/"
}

# Error handling
trap 'echo -e "${RED}✗ Script interrupted or failed at line $LINENO${NC}"' ERR

# Run main function
main

echo -e "${GREEN}Script execution completed!${NC}"