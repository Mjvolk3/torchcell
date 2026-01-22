#!/bin/bash
# scripts/setup-worktree
# [[scripts.setup-worktree]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/setup-worktree


# One-command setup for new git worktrees

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up torchcell worktree...${NC}"

# Get the main repo path (assumes worktrees are in ../torchcell.worktrees/)
MAIN_REPO="/Users/michaelvolk/Documents/projects/torchcell"
WORKTREE_DIR="$(pwd)"

echo -e "\n${BLUE}1. Setting up .env file (worktree-specific)...${NC}"

# Function to update env var paths
update_env_paths() {
    local env_file="$1"

    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS sed requires '' after -i
        sed -i '' "s|^ASSET_IMAGES_DIR=.*|ASSET_IMAGES_DIR=\"$WORKTREE_DIR/notes/assets/images\"|" "$env_file"
        sed -i '' "s|^EXPERIMENT_ROOT=.*|EXPERIMENT_ROOT=\"$WORKTREE_DIR/experiments\"|" "$env_file"
        sed -i '' "s|^WORKSPACE_DIR=.*|WORKSPACE_DIR=\"$WORKTREE_DIR\"|" "$env_file"
        sed -i '' "s|^BIOCYPHER_CONFIG_PATH=.*|BIOCYPHER_CONFIG_PATH=\"$WORKTREE_DIR/biocypher/config/linux-arm_biocypher_config.yaml\"|" "$env_file"
        sed -i '' "s|^SCHEMA_CONFIG_PATH=.*|SCHEMA_CONFIG_PATH=\"$WORKTREE_DIR/biocypher/config/torchcell_schema_config.yaml\"|" "$env_file"
        sed -i '' "s|^MPLSTYLE_PATH=.*|MPLSTYLE_PATH=\"$WORKTREE_DIR/torchcell/torchcell.mplstyle\"|" "$env_file"
    else
        # Linux sed doesn't need '' after -i
        sed -i "s|^ASSET_IMAGES_DIR=.*|ASSET_IMAGES_DIR=\"$WORKTREE_DIR/notes/assets/images\"|" "$env_file"
        sed -i "s|^EXPERIMENT_ROOT=.*|EXPERIMENT_ROOT=\"$WORKTREE_DIR/experiments\"|" "$env_file"
        sed -i "s|^WORKSPACE_DIR=.*|WORKSPACE_DIR=\"$WORKTREE_DIR\"|" "$env_file"
        sed -i "s|^BIOCYPHER_CONFIG_PATH=.*|BIOCYPHER_CONFIG_PATH=\"$WORKTREE_DIR/biocypher/config/linux-arm_biocypher_config.yaml\"|" "$env_file"
        sed -i "s|^SCHEMA_CONFIG_PATH=.*|SCHEMA_CONFIG_PATH=\"$WORKTREE_DIR/biocypher/config/torchcell_schema_config.yaml\"|" "$env_file"
        sed -i "s|^MPLSTYLE_PATH=.*|MPLSTYLE_PATH=\"$WORKTREE_DIR/torchcell/torchcell.mplstyle\"|" "$env_file"
    fi
}

if [ -f .env ] && [ ! -L .env ]; then
    echo "  ✓ .env file already exists (not a symlink)"
    echo "  → Updating worktree-specific paths..."
    update_env_paths ".env"
    echo "  ✓ Updated paths to point to worktree"
elif [ -L .env ]; then
    echo "  ! .env is a symlink - removing and creating worktree-specific copy"
    rm .env
    cp "$MAIN_REPO/.env" .env
    update_env_paths ".env"
    echo "  ✓ Created worktree-specific .env"
else
    echo "  → Creating new .env from main repo template..."
    cp "$MAIN_REPO/.env" .env
    update_env_paths ".env"
    echo "  ✓ Created worktree-specific .env"
fi

echo "  → Worktree-specific paths configured:"
echo "    - ASSET_IMAGES_DIR → $WORKTREE_DIR/notes/assets/images"
echo "    - EXPERIMENT_ROOT → $WORKTREE_DIR/experiments"
echo "    - WORKSPACE_DIR → $WORKTREE_DIR"
echo "  → Shared paths (unchanged):"
echo "    - DATA_ROOT remains shared with main repo (large datasets)"

echo -e "\n${BLUE}2. Verifying VS Code configs...${NC}"
if [ -f .vscode/launch.json ]; then
    echo "  ✓ launch.json exists"
else
    echo "  ✗ launch.json missing (should be tracked in git)"
fi

if [ -f .vscode/tasks.json ]; then
    echo "  ✓ tasks.json exists"
else
    echo "  ✗ tasks.json missing (should be tracked in git)"
fi

if [ -f .vscode/settings.json ]; then
    echo "  ✓ settings.json exists"
else
    echo "  ✗ settings.json missing (should be tracked in git)"
fi

echo -e "\n${BLUE}3. Verifying Python environment...${NC}"
EXPECTED_PYTHON="/Users/michaelvolk/opt/miniconda3/envs/torchcell/bin/python"
if [ -f "$EXPECTED_PYTHON" ]; then
    echo "  ✓ torchcell Python environment found"
    $EXPECTED_PYTHON --version
else
    echo "  ✗ torchcell Python environment not found at $EXPECTED_PYTHON"
fi

echo -e "\n${BLUE}4. Configuring VS Code for worktree...${NC}"
# Use VS Code settings to prioritize worktree over installed package
# This is safer than pip install -e which would break other worktrees

if [ ! -d .vscode ]; then
    mkdir -p .vscode
fi

# Check if settings.json needs PYTHONPATH update
if [ -f .vscode/settings.json ]; then
    if grep -q "python.envFile" .vscode/settings.json; then
        echo "  ✓ VS Code settings already configured"
    else
        echo "  ℹ VS Code settings exist but may need PYTHONPATH configuration"
        echo "  → Check .vscode/settings.json has 'python.envFile': '\${workspaceFolder}/.env.vscode'"
    fi
else
    echo "  ✗ .vscode/settings.json missing"
fi

# Create .env.vscode for PYTHONPATH override (if not exists)
if [ ! -f .env.vscode ]; then
    echo "  Creating .env.vscode with PYTHONPATH..."
    echo "PYTHONPATH=$WORKTREE_DIR:\${PYTHONPATH}" > .env.vscode
    echo "  ✓ Created .env.vscode"
else
    echo "  ✓ .env.vscode already exists"
fi

echo -e "\n${GREEN}✓ Worktree setup complete!${NC}"
echo -e "\n${BLUE}How this works:${NC}"
echo "  - .env is COPIED (not symlinked) from main repo with worktree-specific overrides"
echo "  - Worktree-specific paths (tracked in git):"
echo "    • ASSET_IMAGES_DIR → worktree's notes/assets/images"
echo "    • EXPERIMENT_ROOT → worktree's experiments/"
echo "    • WORKSPACE_DIR → worktree root"
echo "    • Config paths (BioCypher, schema, mplstyle) → worktree versions"
echo "  - Shared paths (large data, not in git):"
echo "    • DATA_ROOT stays pointed at main repo (datasets are expensive to rebuild)"
echo "  - .env.vscode sets PYTHONPATH to prioritize this worktree's code"
echo "  - Other worktrees and main repo are NOT affected"
echo -e "\n${BLUE}Quick start:${NC}"
echo "  - Open this folder in VS Code (or reload window)"
echo "  - Press F5 to debug (uses 'Python: Workspace Folder' config)"
echo "  - Press Cmd+Shift+P -> 'Tasks: Run Task' to see all available tasks"
echo -e "\n${BLUE}Verify worktree is active:${NC}"
echo "  python -c 'import torchcell; print(torchcell.__file__)'"
echo "  (should show path to this worktree, not main repo)"
