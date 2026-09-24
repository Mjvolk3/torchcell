---
id: 08828m1ied673r7j5i3zix6
title: Wt_cleanup
desc: ''
updated: 1790209470376
created: 1790209470376
---

## 2026.09.23 - The sweep after a landing, with its own banner

`scripts/wt_cleanup.py` is the counterpart of the drainer's cleanup. The drainer
removes the worktree, local branch and remote branch of the one branch it lands;
everything that lands another way or never lands accumulates. On this date the
repository carried 28 worktrees beyond the primary checkout. The sweep classifies
each one against `origin/main` after `git fetch --prune` (local `main` lags every
landing, [[vanished-worktree-check-origin-main]]) and acts on exactly one class:

| verdict | rule | action |
|---|---|---|
| landed | 0 commits ahead of `origin/main`, `git status --short` empty | worktree removed, local branch and `origin/<branch>` deleted, open PR closed |
| landed-dirty | 0 ahead, tree holds modified, staged or untracked files | kept, named under the yellow banner with its first three paths |
| unlanded | commits not on `origin/main` | kept |
| detached | no branch (`NNN-build`, bisect trees) | kept; `--detached` removes the landed, clean ones |
| cwd | the worktree the shell runs in | kept |
| branch-only | local or remote branch with no worktree, ancestor of `origin/main` | deleted; `main` never |

There is no force flag: a dirty landed tree is the only copy of that work, so it
is committed (then it lands normally) or moved out with `/deprecate`, and the sweep
is re-run. The landing flock from `scripts/merge_queue.py` is held for the whole
pass so a drainer and a sweep never remove the same tree.

The banner uses a broom in place of the dove, the same stoplight bars, and the
same exit discipline: 0 green (swept clean), 3 yellow (dirty landed trees need a
decision), 2 red (a removal failed), 4 the flock stayed busy. `/enqueue-merge`
runs the sweep in its wrap-up step, before the dove banner; `/wt-cleanup` runs it
on demand.

First dry run on the real repository, 2026-09-23: 1 worktree landed and clean
(`feat/update-kanban-skill`), 9 landed-dirty (three plan and docs trees with
uncommitted notes, five `worktree-wf_*` trees under `.claude/worktrees/` with
edited source, one agent tree mid-task), 8 unlanded, 10 detached (two build trees
and eight bisect trees), 2 branch-only refs (`inference-1-positive-panel` local,
`feat/neo4j-5-upgrade` remote). Tests in
`tests/torchcell/scripts/test_wt_cleanup.py` build a throwaway clone with a bare
origin and cover every verdict, the dry run, the detached flag and the cwd guard.
