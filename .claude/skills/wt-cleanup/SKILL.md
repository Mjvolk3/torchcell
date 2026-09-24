---
name: wt-cleanup
description: Sweep worktrees and branches whose work has already landed on origin/main (local worktree, local branch, remote branch, stale PR), keep everything in flight or dirty, and end with the broom banner. Runs automatically at the end of /enqueue-merge; run it by hand any time the worktree list has grown.
---

# Worktree Cleanup

The merge-queue drainer cleans up only the branch it lands. Everything that
lands another way, or never lands, stays behind: plan worktrees whose note went
in with a sibling, detached build and bisect trees, branches pushed for a PR that
was closed by hand, local branches whose worktree is long gone. This skill runs
`scripts/wt_cleanup.py`, the sweep for all of that, and ends with the **broom
banner**, the sweep's counterpart of the dove.

**Usage:** `/wt-cleanup [--dry-run] [--detached]`

## What the sweep does

Every judgment is against `origin/main` after `git fetch --prune`, never local
`main` (which lags every landing). It takes the landing flock for the whole
pass, so it cannot interleave with a drainer.

| verdict | meaning | action |
|---|---|---|
| `landed` | every commit is on `origin/main`, tree clean | worktree removed, local + remote branch deleted, any open PR closed with a comment |
| `landed-dirty` | landed, but uncommitted or untracked files | **kept**, listed under the yellow banner: those files are the only copy |
| `unlanded` | commits not on `origin/main` | kept (in flight) |
| `detached` | no branch (build, bisect trees) | kept; `--detached` removes the ones that are landed and clean |
| `cwd` | the worktree this shell is parked in | kept, always |
| `branch-only` | a local or remote branch with no worktree, already on `origin/main` | deleted (`main` never) |

Nothing dirty is ever removed, and there is no force flag. A dirty landed tree is
resolved by committing its work (then it is `unlanded` and lands normally) or by
moving the files out with `/deprecate`, then re-running the sweep.

## Steps

1. Park in the primary checkout so the sweep can remove any worktree but this
   one, then run it:
   ```bash
   MAIN="$HOME/Documents/projects/torchcell"
   PY="$HOME/miniconda3/envs/torchcell/bin/python"
   cd "$MAIN"
   $PY "$MAIN/scripts/wt_cleanup.py" [--dry-run] [--detached]
   ```
   Give the Bash call a long timeout (`timeout: 300000`): the pass runs one
   `gh pr list` per branch and waits up to 120 s for the landing flock.

2. Branch on the exit code (control flow only; do not print this table):

   | exit | banner | do |
   |---|---|---|
   | 0 | green, swept clean | nothing; the banner is the last output |
   | 3 | yellow, landed trees hold uncommitted work | list nothing extra; the banner already names each tree and its first paths. Fix them only if the user asks |
   | 2 | red, a removal failed | the banner names the failure; investigate the named worktree or branch |
   | 4 | yellow, flock busy | a landing is in progress; re-run after it finishes |

3. **The banner is the last thing in the pane.** Do not restate, summarize or
   describe it afterwards, and do not print the exit-code table. The same
   output discipline as `/enqueue-merge`.

## Reading the yellow list

The dirty trees the sweep keeps are usually one of three things:

- a plan or docs worktree whose note landed with another branch but whose
  scratch files never got committed: commit them on a fresh branch or `/deprecate`
  the files;
- a `worktree-wf_*` tree under `.claude/worktrees/` left by a workflow run: the
  edits inside are that run's uncommitted output, keep or discard by hand;
- a bisect or build tree under `/scratch/.../bisect/` or `NNN-build`: detached,
  usually one modified results file; `--detached` removes the clean ones only.
