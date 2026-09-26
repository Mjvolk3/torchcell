---
id: z5wn4gmb0g0haubzdhkeubx
title: Test_drain_merge_queue
desc: ''
updated: 1790416741614
created: 1790416741614
---

## 2026.09.26 - Landing branches into a bare origin under tmp_path

The drainer once force-pushed a test's temporary history to the real `main` through an inherited `GIT_DIR` (Gotcha 9 of [[plan.test-suite-buildout.2026.09.25]]), so this file is built to make that impossible: `_cleanup_remote` is replaced by a recorder (it would call `gh`), `shutil.which` returns None, the Slack webhook and the four `GIT_*` variables are deleted, the working directory is the temporary main, `HOME` and the global git config point away from the developer's, and an autouse teardown asserts that every repository under `tmp_path` has its `origin` under `tmp_path`. The layout mirrors production (`<tmp>/torchcell` and `<tmp>/torchcell.worktrees/<branch>`). Covered: a clean landing (origin/main at the branch tip, row `landed` with that sha, worktree and local branch gone, remote cleanup asked once, heartbeat written), a missing and a dirty worktree blocked with their reasons, a rebase conflict blocked with the worktree left clean on its branch, the free-note sweep (committed and pushed; a paired note is not free; a diverged local main skips the sweep), `_is_free_note`, the exclusive landing lock, and `main()` stripping the inherited git environment.

Finding: `git status --porcelain` collapses an untracked directory to one `?? dir/` line, so a free note inside a directory git has never seen (a brand-new `notes/` or `.claude/rules/` tree) is not swept; pinned as behavior, and the sweep fixture commits a placeholder so `notes/` is tracked as in production.
