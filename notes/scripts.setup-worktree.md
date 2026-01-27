---
id: mrhkxcxh6oui8plh9b2nix8
title: Setup Worktree
desc: 'setup-worktree'
updated: 1769533754601
created: 1768688635122
---

## Git Worktree Setup Guide

This guide documents the streamlined setup process for creating new git worktrees in the torchcell project.

## Quick Start

**Option 1: Shared data (default)** - Use main repo's datasets (saves disk space)

```bash
./scripts/setup-worktree.sh
```

VS Code: `Cmd+Shift+P` → "Tasks: Run Task" → **"tcwt: setup worktree (data-main)"**

**Option 2: Local data** - Build datasets in this worktree (for dataset experimentation)

```bash
./scripts/setup-worktree.sh --data-local
```

VS Code: `Cmd+Shift+P` → "Tasks: Run Task" → **"tcwt: setup worktree (data-local)"**

**When to use which:**

- **Shared**: Model training, inference, most feature development
- **Local**: Testing dataset preprocessing changes, schema updates, data pipeline debugging

## What Gets Set Up

### 1. Environment Variables (`.env`)

- **Copies** `.env` from main repo with worktree-specific overrides
- **Worktree-specific paths**: `ASSET_IMAGES_DIR`, `EXPERIMENT_ROOT`, `WORKSPACE_DIR`
- **Data storage**:
  - **Shared (default)**: `DATA_ROOT` → main repo, `data/` symlinked
  - **Local (`--data-local`)**: `DATA_ROOT` → worktree, local `data/` directory

### 2. VS Code Launch Configurations (`.vscode/launch.json`)

- Tracked in git, appears automatically in all worktrees
- Configured with correct Python interpreter: `/Users/michaelvolk/opt/miniconda3/envs/torchcell/bin/python`
- Available debug configurations:
  - **Python: Workspace Folder** - Debug current file (F5)
  - **Debug Hydra Config** - Debug experiment configs
  - **Pytest Current Line** - Debug specific test
  - **Python: yeast8** - Use yeast8 environment
  - **Python: torchgeo** - Use torchgeo environment

### 3. VS Code Tasks (`.vscode/tasks.json`)

- Tracked in git, appears automatically in all worktrees
- All workspace tasks available via `Cmd+Shift+P` → "Tasks: Run Task"
- Key tasks:
  - **tcwt: setup worktree (data-main)** - Setup with shared data (default)
  - **tcwt: setup worktree (data-local)** - Setup with local data storage
  - **tcf: add frontmatter** - Add dendron frontmatter to files
  - **tcm: move file** - Move file with related note/test
  - **tcd: delete file** - Delete file and open related files
  - **tco: open related code** - Open code from dendron note
  - **black: file** - Format Python code
  - **mypy: file/workspace** - Type checking
  - **pytest: coverage** - Run tests with coverage
  - **Pandoc tasks** - Generate PDFs from markdown
  - **HPC tasks** - Submit jobs to Delta/nano clusters

## Creating a New Worktree

```bash
# Navigate to worktrees directory
cd /Users/michaelvolk/Documents/projects/torchcell.worktrees

# Create new worktree
git worktree add feature-name branch-name

# Navigate into worktree
cd feature-name

# Run setup script
./scripts/setup-worktree.sh

# Open in VS Code
code .
```

## Architecture Decisions

### Why Copy .env with Overrides?

- **Worktree-specific paths**: Each worktree needs its own `ASSET_IMAGES_DIR`, `EXPERIMENT_ROOT`
- **Flexible data storage**: Can choose shared or local datasets per worktree
- **No accidental overwrites**: Worktrees won't modify main repo's data accidentally

### Why Two Data Storage Modes?

- **Shared (default)**: Most worktrees share expensive datasets (saves GB of disk space)
- **Local (`--data-local`)**: For dataset experimentation (preprocessing changes, schema updates)
- **Symlink in shared mode**: Scripts use relative `data/` paths, transparently resolves to main repo

### Why Extract Tasks from Workspace File?

- **Git-tracked**: Tasks appear in all worktrees automatically
- **No workspace dependency**: Works when opening folder directly in VS Code
- **Standard VS Code pattern**: `.vscode/tasks.json` is the canonical location

### Why Hardcode Python Path in launch.json?

- **Prevents environment mismatch**: VS Code won't use wrong conda env
- **Explicit > Implicit**: Clear which Python is used for debugging
- **Avoids torch-scatter issues**: Ensures correct PyTorch environment

### How git rev-parse Makes Scripts Worktree-Aware

All dendron note scripts now work in worktrees using `git rev-parse --show-toplevel`:

```python
def get_git_root():
    """Get the git repository root, works in both main repo and worktrees."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True
    )
    return result.stdout.strip()
```

**How it works:**

- Main repo: Returns `/Users/michaelvolk/Documents/projects/torchcell`
- Worktree: Returns `/Users/michaelvolk/Documents/projects/torchcell.worktrees/feature-name`
- Both share same `.git` directory but have independent working trees

**Updated scripts:**

- `add_frontmatter.py` - Creates proper frontmatter paths (no ugly `../torchcell.worktrees/...`)
- `move_file_with_related.py` - Moves file + note + test
- `delete_file_and_open_related.py` - Deletes and opens related
- `from_note_open_related_file.py` - Opens code from note
- `from_note_open_related_src.py` - Opens source from note
- `from_src_open_related_note_and_test.py` - Opens note and test from source
- `create_python_test_file.py` - Creates test files
- `python_move_dendron_pytest.py` - Moves Python files with pytest

## Files Tracked in Git

These files are committed and automatically available in all worktrees:

```
.vscode/
├── launch.json       # Debug configurations
├── tasks.json        # VS Code tasks
├── settings.json     # Editor settings
└── dendron.code-snippets  # Code snippets

scripts/
└── setup-worktree.sh # This setup script
```

## Files NOT Tracked in Git

These files are created by the setup script:

```
.env                  # Copy of main repo's .env with worktree overrides
data/                 # Symlink to main repo (shared mode) or local dir (local mode)
.env.vscode           # PYTHONPATH override for VS Code
```

## Troubleshooting

### Issue: VS Code uses wrong Python environment

**Symptoms:**

```
torch_scatter._version_cpu.so: Symbol not found: __ZN3c1017RegisterOperatorsD1Ev
```

**Solution:**

The debug configurations in `launch.json` specify the correct Python path. Make sure you're using one of the provided debug configurations (F5) rather than running files directly.

## Comparison with Alternatives

### ❌ Manual Setup Every Time

- Tedious and error-prone
- Easy to forget steps
- Inconsistent across worktrees

### ❌ Copy Configuration Files

- Configuration drift between worktrees
- Hard to update all worktrees when config changes
- Wastes disk space

### ✅ This Approach (Script + Git-tracked Configs)

- One command setup
- Consistent across all worktrees
- Easy to update (just git pull)
- Industry standard pattern

## Related Documentation

- Main workspace: `/Users/michaelvolk/Documents/projects/torchcell/torchcell.code-workspace`
- Python env: `/Users/michaelvolk/opt/miniconda3/envs/torchcell`
- Project CLAUDE.md: Documents coding standards and experiment structure

## Using GitLens

GitLens in VS Code provides enhanced git functionality for worktrees:

- **Worktree visualization**: See all worktrees in the Source Control sidebar
- **Branch switching**: Easily switch between worktrees
- **File history**: View file history across main repo and worktrees
- **Compare**: Compare files between worktrees
