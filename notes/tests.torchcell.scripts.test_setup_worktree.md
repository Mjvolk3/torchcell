---
id: k877xue2ghnjtcc12149asu
title: Test_setup_worktree
desc: ''
updated: 1790416749062
created: 1790416749062
---

## 2026.09.26 - setup-worktree.sh inside a worktree of a temporary repository

`PATH=/usr/bin:/bin` (no conda, no pre-commit), `HOME` and the global git config under `tmp_path`. The script finds the main repository through `git rev-parse --git-common-dir`, copies its `.env` with six paths rewritten to the worktree (`DATA_ROOT` untouched, `OTHER` kept), symlinks `data/` to the main repository, writes `.env.vscode` with the `PYTHONPATH` line, and registers the weekly-note merge driver in the shared `.git/config`; each is asserted exactly against the expected text. A second run keeps the copied `.env` (not a symlink) and the existing symlink; `--data-local` rewrites `DATA_ROOT` to the worktree and creates a real `data/torchcell`; an unknown option exits 1 with the usage before anything is written. Phase 4 of [[plan.test-suite-buildout.2026.09.25]].
