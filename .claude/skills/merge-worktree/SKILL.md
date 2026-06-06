---
name: merge-worktree
description: Merge a worktree branch to main via rebase, then clean up worktree and branch. Takes branch name as argument.
---

# Merge Worktree

Merge a completed worktree branch into main via rebase (no PR merge), then remove the worktree and delete the branch.

**Usage:** `/merge-worktree <branch-name>`

If no branch name is given, list active worktrees and ask the user to pick.

## Path convention (set first; then cwd never matters)

The recurring failure is running the **ff-merge in the wrong directory**. If `git merge --ff-only` runs inside the worktree, `HEAD` there is already the branch tip, so git prints `Already up to date` and **main never advances** -- a silent no-op.

To make that impossible, **every git command targets its repo explicitly with `git -C <path>`.** Set these once (substitute the real branch):

```bash
MAIN="$HOME/Documents/projects/torchcell"
WT="$HOME/Documents/projects/torchcell.worktrees/<branch>"
```

- Main-repo operations use `git -C "$MAIN" ...`; the one worktree op (rebase) uses `git -C "$WT" ...`.
- **Run rebase and ff-merge as separate commands. Never chain `cd "$WT"` with a main-repo git command in the same `&&` line.**

## Serialize -- never run concurrently

Do NOT start while another `/merge-worktree` (or any landing that stashes on shared main) is in flight. All worktrees share one `.git` stash list; parallel runs corrupt each other. Check `git -C "$MAIN" stash list` is quiet first; if a landing is active, wait.

## Step 1: Validate

1. `git -C "$MAIN" worktree list` -- confirm the branch exists as a worktree. If not, show the list and stop.
2. Check the worktree is clean:
   ```bash
   git -C "$WT" status --short
   ```
   If there are uncommitted changes, warn and stop.

## Step 2: Stash main repo changes

If `git -C "$MAIN" status --short` shows uncommitted **tracked** changes:

```bash
git -C "$MAIN" stash push -m "merge-worktree: main WIP parked for <branch> merge"
git -C "$MAIN" rev-parse stash@{0}   # record as PARKED_STASH_SHA
```

Capture the SHA so Step 6 pops *this exact* stash (positional `stash@{0}` is unsafe). If the tree is clean (or only untracked scratch), skip.

## Step 3: Rebase onto main

```bash
git -C "$WT" rebase main
```

If conflicts, inform the user and stop. Do NOT force anything.

## Step 4: Fast-forward main

```bash
git -C "$MAIN" merge <branch> --ff-only
```

A correct ff-merge prints `Updating <old>..<new>` + `Fast-forward`. If it prints `Already up to date` right after a rebase that produced new commits, you ran it against the wrong repo -- re-run with `-C "$MAIN"` and confirm `git -C "$MAIN" rev-parse main` equals the branch tip. If `--ff-only` fails (non-fast-forward), the rebase didn't land on current main -- re-fetch and re-rebase; do not force.

## Step 5: Push main

```bash
git -C "$MAIN" push origin main
```

## Step 6: Close any PR, then clean up

A worktree branch from `/wt-implement` usually has an **open PR**. We already landed via rebase + ff-only, so the PR is not the merge vehicle -- but it must be **explicitly closed**.

1. Find and close any open PR (do NOT `gh pr merge` -- the commits already landed; merging would duplicate history):
   ```bash
   PR=$(gh pr list --repo Mjvolk3/torchcell --head <branch> --state open --json number --jq '.[0].number')
   ```
   If `$PR` is non-empty:
   ```bash
   gh pr close "$PR" --repo Mjvolk3/torchcell \
     --comment "Landed on \`main\` via rebase + ff-only (linear history; commits applied verbatim). Intentionally not merged through GitHub. The change is on main as of the latest push."
   ```
2. Remove the worktree: `git -C "$MAIN" worktree remove --force "$WT"`
3. Delete the local branch: `git -C "$MAIN" branch -d <branch>` (use `-D` if `-d` fails after a confirmed merge).
4. Delete the remote branch (if pushed): `gh api repos/Mjvolk3/torchcell/git/refs/heads/<branch> --method DELETE` (a `422 Reference does not exist` is harmless). This also auto-closes any PR that slipped past step 1.

## Step 7: Restore stashed changes

If Step 2 stashed (you have `PARKED_STASH_SHA`), restore now -- **always pop, never leave WIP parked**:

```bash
git -C "$MAIN" stash list --format='%gd %H' | awk -v s=<PARKED_STASH_SHA> '$2==s {print $1; exit}'
```

Pop that exact ref: `git -C "$MAIN" stash pop <ref>`. If it **conflicts**, do NOT drop the stash or auto-resolve -- report the conflicting files and stop (the stash stays intact). If it applies cleanly but later trips a format gate (e.g. weekly-note duplicate H2), that is a follow-up format pass on the now-visible WIP, not a reason to re-stash.

## Step 8: Verify

```bash
git -C "$MAIN" worktree list
git -C "$MAIN" log --oneline -3
```

The branch's top commit MUST appear in main's log (final catch for a no-op ff-merge). Print:

```
Merged:     <branch> -> main
Closed PR:  #<pr_number> (landed via rebase+ff) | none open
Cleaned up: worktree, local branch, remote branch
```

If an open PR existed and was not closed (and the remote-branch deletion didn't auto-close it), say so -- never report a clean landing while a PR is hanging.

## Important Rules

- **Every git command uses `git -C "$MAIN"` or `git -C "$WT"`.** Never depend on ambient cwd; never chain `cd "$WT"` with a main-repo git command on the same `&&` line.
- **The PR is never the merge vehicle, but it must always be closed.** Branches land via rebase + ff-only + push, never `gh pr merge`.
- NEVER force-push without `--force-with-lease`.
- If anything fails, stop and report -- do not auto-recover.
- The user's tool approval prompts are the gates -- do NOT ask extra confirmation questions.
