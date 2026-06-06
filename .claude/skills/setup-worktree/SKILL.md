---
name: setup-worktree
description: Create a new git worktree or ensure an existing one is properly configured. Wraps scripts/setup-worktree.sh with validation.
---

# Setup Worktree

Create a new git worktree or ensure an existing one is configured. Wraps `scripts/setup-worktree.sh` with validation the bash script doesn't cover.

**Usage:**
- `/setup-worktree <branch-name>` -- create new worktree and set it up
- `/setup-worktree` -- set up the current worktree (if already inside one)

## Step 1: Determine worktree path

**If a branch name is provided:**

```bash
git worktree list
```

If it exists, use its path. If not, create it:

```bash
git worktree add ~/Documents/projects/torchcell.worktrees/<branch> -b <branch>
```

Naming convention: `plan/<name>` branch -> `~/Documents/projects/torchcell.worktrees/plan/<name>/`.

**If no argument and already in a worktree** (`pwd` contains `.worktrees/`): use the current directory.

**If no argument and in the main repo:** inform the user and stop -- need a branch name.

## Step 2: Run the bash setup script

```bash
cd <worktree_path>
bash scripts/setup-worktree.sh
```

This handles: worktree-specific `.env` (main->worktree path prefix swap), `data/` symlink to main (shared datasets), `.env.vscode` PYTHONPATH, Claude Code auto-memory symlink, pre-commit install, and the weekly-note merge driver registration.

Pass `--data-local` if this worktree needs its own `DATA_ROOT` (dataset experimentation) instead of sharing main's.

## Step 3: Validate .env sources cleanly

```bash
bash -c 'set -a; source <worktree_path>/.env 2>&1; echo "EXIT: $?"'
```

Non-zero exit means `.env` has lines bash interprets as commands (unquoted URLs, comments missing `#`). Fix: add `#` to comment-like lines, or quote `KEY=value with spaces`. Re-test and report what was fixed.

## Step 4: Verify import resolves to the worktree

```bash
PYTHONPATH=<worktree_path> ~/miniconda3/envs/torchcell/bin/python -c "import torchcell; print(torchcell.__file__)"
```

The path must point to this worktree, not main or another worktree. If it points elsewhere, a `pip install -e .` from another checkout is shadowing it -- the `.env.vscode` PYTHONPATH override (created by the script) is what prevents this in the editor.

## Step 5: Report

```
Worktree setup complete: <worktree_path>
  Branch:  <branch_name>
  .env:    OK (sourced cleanly) | FIXED (N lines)
  Data:    data/ symlinked to main | LOCAL
  Import:  torchcell loads from <path>
  Memory:  symlinked to main repo
  Merge:   merge.weeklynote registered
```

## Important Rules

- ALWAYS run this when creating or first working with a worktree.
- The bash script is the foundation -- run it first, then layer on validation.
- Never skip the `.env` validation -- unquoted URLs cause silent source failures.
- If the worktree is behind main, suggest rebasing AFTER setup (setup needs the current branch state).
- Do NOT ask extra approval questions -- tool approval prompts are the gates.
