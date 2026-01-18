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

echo -e "\n${BLUE}1. Setting up .env symlink...${NC}"
if [ -L .env ]; then
    echo "  ✓ .env symlink already exists"
elif [ -f .env ]; then
    echo "  ! .env file exists (not a symlink), backing up to .env.backup"
    mv .env .env.backup
    ln -s "$MAIN_REPO/.env" .env
    echo "  ✓ Created .env symlink"
else
    ln -s "$MAIN_REPO/.env" .env
    echo "  ✓ Created .env symlink"
fi

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

echo -e "\n${GREEN}✓ Worktree setup complete!${NC}"
echo -e "\n${BLUE}Quick start:${NC}"
echo "  - Open this folder in VS Code"
echo "  - Press F5 to debug (uses 'Python: Workspace Folder' config)"
echo "  - Press Cmd+Shift+P -> 'Tasks: Run Task' to see all available tasks"
