---
id: d2f2t6402kkt8wgts89h4m7
title: worktree-setup
desc: 'One-command worktree setup with git-aware note scripts'
updated: 1768691050469
created: 1768690311298
---

## Context

Started from [[fitness-interaction-n_samples|user.Mjvolk3.torchcell.tasks.weekly.2026.03.fitness-interaction-n_samples]] but worktree setup insufficient - pivoted to build infrastructure first.

Branch: `fitness-interaction-n_samples`
Worktree: `/Users/michaelvolk/Documents/projects/torchcell.worktrees/fitness-interaction-n_samples`

## 2026.01.16

- [x] Debug `.env` symlink issue - `DATA_ROOT` not found in worktree
- [x] Debug torch-scatter error - VS Code using wrong Python environment
- [x] Create one-command setup script [[setup-worktree|scripts.setup-worktree]]
- [x] Extract tasks from workspace to `.vscode/tasks.json` (git-tracked)
- [x] Configure `launch.json` with correct Python interpreter for all debug configs
- [x] Update 8 note management scripts to be worktree-aware using `git rev-parse`:
  - [[add_frontmatter|notes.assets.scripts.add_frontmatter]]
  - `move_file_with_related.py`
  - `delete_file_and_open_related.py`
  - `from_note_open_related_file.py`
  - `from_note_open_related_src.py`
  - `from_src_open_related_note_and_test.py`
  - `create_python_test_file.py`
  - `python_move_dendron_pytest.py`
- [x] Add `tcwt: setup worktree` VS Code task
- [x] Create comprehensive documentation [[setup-worktree|scripts.setup-worktree]]
- [x] Test frontmatter generation in worktree (no more `../torchcell.worktrees/...` paths)
- [x] Test setup script - verifies `.env` symlink, configs, Python environment

## Files Created/Modified

**Git-tracked (appear in all worktrees):**

- `.vscode/launch.json` - Debug configs with correct Python
- `.vscode/tasks.json` - All tasks + new `tcwt` task
- `scripts/setup-worktree.sh` - One-command setup
- `notes/scripts.setup-worktree.md` - Documentation
- `notes/assets/scripts/*.py` - 8 scripts updated

**Per-worktree (created by setup script):**

- `.env` → symlink to main repo

## Future Workflow

```bash
git worktree add ../torchcell.worktrees/feature-name branch-name
cd ../torchcell.worktrees/feature-name
./scripts/setup-worktree.sh  # OR run task: tcwt: setup worktree
code .
```

## Summary for Main Weekly

```markdown
- [x] Worktree setup infrastructure - one-command setup, git-aware note scripts
  - Setup script: `./scripts/setup-worktree.sh` or `tcwt` task
  - 8 note management scripts now worktree-compatible (use `git rev-parse`)
  - Git-tracked configs: `.vscode/launch.json`, `.vscode/tasks.json`
  - Documentation: [[setup-worktree|scripts.setup-worktree]]
```
