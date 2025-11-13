#!/bin/bash
# launch_tensorboard.sh
# Launch TensorBoard for viewing PyTorch profiler results

set -e

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

# Load environment variables
if [ -f "$BASE_DIR/.env" ]; then
    export $(grep -v '^#' "$BASE_DIR/.env" | xargs)
fi

# Set profile directory
DATA_ROOT="${DATA_ROOT:-/scratch/projects/torchcell}"
PROFILE_DIR="$DATA_ROOT/data/torchcell/experiments/006-kuzmin-tmi/profiler_output"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}TensorBoard Launcher for PyTorch Profiler${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to launch TensorBoard
launch_tensorboard() {
    local port=$1
    local logdir=$2
    local description=$3

    echo -e "${YELLOW}Checking port $port...${NC}"

    if check_port $port; then
        echo -e "${RED}Port $port is already in use!${NC}"
        echo "Either kill the existing process or use a different port."
        echo "To kill existing TensorBoard on port $port:"
        echo "  lsof -ti:$port | xargs kill -9"
        return 1
    fi

    echo -e "${GREEN}Launching TensorBoard for: $description${NC}"
    echo "  Directory: $logdir"
    echo "  Port: $port"
    echo ""

    # Activate conda environment
    if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate torchcell
    elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
        source ~/anaconda3/etc/profile.d/conda.sh
        conda activate torchcell
    fi

    # Launch TensorBoard
    tensorboard --logdir="$logdir" --port=$port --bind_all &
    local tb_pid=$!

    sleep 2

    if ps -p $tb_pid > /dev/null; then
        echo -e "${GREEN}✓ TensorBoard started successfully (PID: $tb_pid)${NC}"
        echo ""
        echo -e "${BLUE}Access TensorBoard at:${NC}"
        echo "  Local: http://localhost:$port"
        echo "  Network: http://$(hostname -I | awk '{print $1}'):$port"
        echo ""
        return 0
    else
        echo -e "${RED}✗ Failed to start TensorBoard${NC}"
        return 1
    fi
}

# Function to stop all TensorBoard instances
stop_all_tensorboard() {
    echo -e "${YELLOW}Stopping all TensorBoard instances...${NC}"
    pkill -f tensorboard 2>/dev/null || true
    echo -e "${GREEN}All TensorBoard instances stopped${NC}"
}

# Main menu
main() {
    echo "Select an option:"
    echo "  1) Launch TensorBoard for all profiles (port 6006)"
    echo "  2) Launch separate TensorBoards for each model"
    echo "  3) Launch TensorBoard with custom directory"
    echo "  4) Stop all TensorBoard instances"
    echo "  5) Exit"
    echo ""

    read -p "Enter your choice (1-5): " choice
    echo ""

    case $choice in
        1)
            launch_tensorboard 6006 "$PROFILE_DIR" "All Profiles"
            ;;
        2)
            echo -e "${YELLOW}Launching separate TensorBoard instances...${NC}"

            # Find hetero and dango profile directories
            HETERO_DIR=$(find "$PROFILE_DIR" -type d -path "*hetero_dango_gi_*" -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2)
            DANGO_DIR=$(find "$PROFILE_DIR" -type d -path "*dango_*" ! -path "*hetero*" -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2)

            if [ -n "$HETERO_DIR" ]; then
                launch_tensorboard 6006 "$HETERO_DIR" "HeteroCell Profile"
            else
                echo -e "${RED}No HeteroCell profile directory found${NC}"
            fi

            if [ -n "$DANGO_DIR" ]; then
                launch_tensorboard 6007 "$DANGO_DIR" "Dango Profile"
            else
                echo -e "${RED}No Dango profile directory found${NC}"
            fi

            echo ""
            echo -e "${BLUE}If using SSH tunneling, ensure you have forwarded both ports:${NC}"
            echo "  ssh -L 6006:localhost:6006 -L 6007:localhost:6007 user@remote"
            ;;
        3)
            read -p "Enter custom directory path: " custom_dir
            read -p "Enter port number (default 6006): " custom_port
            custom_port=${custom_port:-6006}

            if [ -d "$custom_dir" ]; then
                launch_tensorboard $custom_port "$custom_dir" "Custom Directory"
            else
                echo -e "${RED}Directory not found: $custom_dir${NC}"
            fi
            ;;
        4)
            stop_all_tensorboard
            ;;
        5)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            ;;
    esac

    echo ""
    echo -e "${YELLOW}Important:${NC}"
    echo "If accessing from local machine, ensure SSH tunnel is active:"
    echo "  ssh -L 6006:localhost:6006 user@$(hostname)"
    echo ""
    echo "To view PyTorch profiles in Chrome (alternative to TensorBoard):"
    echo "  1. Open Chrome and navigate to: chrome://tracing"
    echo "  2. Click 'Load' and select JSON files from: $PROFILE_DIR"
}

# Run main function
main