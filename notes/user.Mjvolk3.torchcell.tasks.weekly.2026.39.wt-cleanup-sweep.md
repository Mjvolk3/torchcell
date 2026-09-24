---
id: ijsfls6bfjqa5ovlemwd41v
title: wt-cleanup-sweep
desc: ''
updated: 1790209478753
created: 1790209478753
---

## 2026.09.23

- [x] Worktree sweep after landings: `scripts/wt_cleanup.py` removes landed clean worktrees with their local and remote branches, keeps dirty and unlanded ones, ends with the broom banner; hooked into `/enqueue-merge` wrap-up and exposed as `/wt-cleanup` [[scripts.wt_cleanup]]
