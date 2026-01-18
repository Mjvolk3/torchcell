---
id: mrhkxcxh6oui8plh9b2nix8
title: Setup Worktree
desc: 'setup-worktree'
updated: 1768705011439
created: 1768688635122
---

## Git Worktree Setup Guide

This guide documents the streamlined setup process for creating new git worktrees in the torchcell project.

## Quick Start

When creating a new worktree, run one command:

```bash
# From within your new worktree
./scripts/setup-worktree.sh
```

That's it! Your worktree is now ready for development.

**Or use the VS Code task:**

`Cmd+Shift+P` → "Tasks: Run Task" → "tcwt: setup worktree"

## What Gets Set Up

The setup script handles three critical configurations:

### 1. Environment Variables (`.env`)

- Creates symlink to main repo's `.env` file
- Ensures all worktrees use same environment configuration (DATA_ROOT, ASSET_IMAGES_DIR, etc.)
- No need to maintain multiple `.env` files

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
  - **tcwt: setup worktree** - Run this setup script
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

### Why Symlink .env Instead of Copying?

- **Single source of truth**: Changes propagate automatically to all worktrees
- **No sync issues**: Impossible for worktrees to have different env vars
- **Zero maintenance**: Set it once, forget it

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
.env                  # Symlink to main repo's .env
```

## Troubleshooting

### Issue: VS Code uses wrong Python environment

**Symptoms:**

```
torch_scatter._version_cpu.so: Symbol not found: __ZN3c1017RegisterOperatorsD1Ev
```

**Solution:**

The debug configurations in `launch.json` specify the correct Python path. Make sure you're using one of the provided debug configurations (F5) rather than running files directly.

### Issue: DATA_ROOT not found

**Symptoms:**

```
ValueError: DATA_ROOT environment variable is not set
```

**Solution:**

Run `./scripts/setup-worktree.sh` to create the `.env` symlink.

Verify with:

```bash
ls -la .env
# Should show: .env -> /Users/michaelvolk/Documents/projects/torchcell/.env
```

### Issue: Tasks not showing up

**Symptoms:**

VS Code task menu is empty or missing expected tasks

**Solution:**

Ensure `.vscode/tasks.json` exists and is tracked in git:

```bash
git ls-files .vscode/tasks.json
```

If missing, it should be added to git in the main repository.

### Issue: Frontmatter Has Wrong Paths

**Symptoms:**

Frontmatter shows `../torchcell.worktrees/...` paths

**Solution:**

The scripts have been updated to use `git rev-parse`. Regenerate frontmatter:

```bash
python notes/assets/scripts/add_frontmatter.py path/to/file.py
```

## Comparison with Alternatives

###❌ Manual Setup Every Time

- Tedious and error-prone
- Easy to forget steps
- Inconsistent across worktrees

###❌ Copy Configuration Files

- Configuration drift between worktrees
- Hard to update all worktrees when config changes
- Wastes disk space

###✅ This Approach (Script + Git-tracked Configs)

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
