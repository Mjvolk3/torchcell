---
name: enqueue-merge
description: Enqueue a finished worktree branch and watch it land through the single-writer merge queue from this session -- resolving any rebase conflict here, and printing a distinct dove signal when it is merged + cleaned up + safe to close. Takes a branch name (defaults to the current worktree's branch).
---

# Enqueue Merge

Hand a finished worktree branch to the single-writer merge queue
(`scripts/merge_queue.py`, `scripts/drain_merge_queue.py`) and **watch it all the
way to landed from this session**. This replaces running `/merge-worktree`
inline, and is the mechanism that stops concurrent worktree landings from
blocking each other or racing the shared `.git`.

The landing critical section is deterministic and model-free -- the drainer
rebases `origin/main` -> `push HEAD:main` -> closes the PR -> cleans up, all
under a non-blocking `flock`, so only one branch lands at a time. What this skill
adds is the **session staying in the loop**: it watches *its* branch's queue row,
and if the branch can't fast-forward across something that landed ahead of it,
**this session resolves the rebase conflict** (the one step that needs a model),
requeues, and watches again. When the branch finally lands, you get one distinct
in-pane signal -- a dove + green stoplight -- that means *merged, cleaned up,
safe to close this session*.

**Usage:** `/enqueue-merge [<branch-name>]`

With no argument, use the current worktree's branch.

## Preconditions (the enqueue gate)

Enqueue means "this branch is final -- land it." Three gates, all checked before
the branch enters the queue:

1. **Fully committed** -- `git -C "$WT" status --short` is empty. The drainer
   blocks a branch with uncommitted changes rather than landing it.
2. **Has an open PR** -- every landed branch must have had a PR, so the drainer's
   cleanup always has one to close (audit/review symmetry). No open PR -> stop
   and tell the user to `gh pr create` first.
3. **No foreign commits** -- the branch must carry only *its own* commits, not a
   sibling feature's `plan:` note inherited from a stale local `main` (a
   "tangle"). Such commits don't conflict on rebase, so the drainer would land
   them silently as residue -- the one failure the conflict-resolution flow can
   never catch. The `foreign` subcommand is the deterministic detector; on a hit
   the agent strips them before queueing (see step 4).

## Output discipline -- the dove banner is the LAST thing in the pane

The dove close banner is the **final, bottom-most output** of this skill: the
user glances at the bottom of the pane and knows the branch is merged + cleaned
up + safe to close. Two commands cooperate to keep it there:

- `watch` is **control flow only** -- it blocks until the branch reaches a
  terminal state, prints a single terse line (`watch: landed <branch> @ <sha>`),
  and returns an exit code. It does **not** print the banner.
- `banner` prints the dove + stoplight banner and is run **last** -- after
  `watch` returns and after any wrap-up (closing the issue, one summary line).

So the ordering is fixed: terse `watch` line -> your wrap-up -> the `banner`
command. **Nothing may follow the banner** -- no summary, no sign-off, no "let me
know". Post-banner chatter pushes the signal off-screen, which is the bug this
design fixes. Do **not**:

- reproduce, pre-render, or describe the banner, its emoji, or the stoplight
  dots in the conversation -- the `banner` command renders them;
- print or paraphrase the exit-code table (below) as a "watching for..." key;
- narrate intermediate queue states while waiting.

Emit at most **one** short line before watching (e.g. `Enqueued <branch>;
watching for the landing.`). The exit-code table is **your control flow only**.

## Steps

1. **Resolve paths:**
   ```bash
   MAIN="$HOME/Documents/projects/torchcell"
   PY="$HOME/miniconda3/envs/torchcell/bin/python"
   BRANCH="${1:-$(git rev-parse --abbrev-ref HEAD)}"
   WT="$HOME/Documents/projects/torchcell.worktrees/$BRANCH"
   ```
   Confirm `$WT` is a real worktree (`git -C "$MAIN" worktree list`); if the
   branch has no worktree, show the list and stop.

2. **Gate 1 -- committed:** `git -C "$WT" status --short`. If non-empty, tell the
   user to commit or discard first, and stop.

3. **Gate 2 -- open PR exists:**
   ```bash
   gh pr list --repo Mjvolk3/torchcell --head "$BRANCH" --state open --json number --jq '.[0].number'
   ```
   If this prints nothing, there is no open PR -- stop and tell the user to
   `gh pr create` for `$BRANCH` before enqueuing. Do not enqueue without a PR.

4. **Gate 3 -- no foreign commits.** Fetch so the base is current, then run the
   deterministic detector over `origin/main..$BRANCH`:
   ```bash
   git -C "$WT" fetch origin main --quiet
   $PY "$MAIN/scripts/merge_queue.py" foreign "$BRANCH"
   ```
   Branch on the exit code:
   - **0** -- clean; every commit is the branch's own work. Proceed to step 5.
   - **1** -- foreign commits found *and* strippable with one rebase. Run the
     exact command it printed (in `$WT`), then re-run `foreign` and confirm it
     returns 0 before proceeding:
     ```bash
     git -C "$WT" rebase --onto origin/main <strip_point> "$BRANCH"
     $PY "$MAIN/scripts/merge_queue.py" foreign "$BRANCH"
     ```
     The strip rewrote history, so update the open PR to match what will land:
     `git -C "$WT" push --force-with-lease`.
   - **2** -- foreign commits are *not* a contiguous base prefix; stop and tell
     the user (an interactive rebase is needed). Do not enqueue.

5. **Enqueue** (idempotent -- a second enqueue of an active branch is a no-op):
   ```bash
   $PY "$MAIN/scripts/merge_queue.py" add "$BRANCH" \
     --worktree "$WT" --by "$CLAUDE_CODE_SESSION_ID"
   ```

6. **Trigger the drain in the background** (detached, so it runs to completion
   independent of this session -- a deep queue can take longer than a single
   foreground command may run, and the drainer self-flocks so this never races
   another drainer):
   ```bash
   nohup "$PY" "$MAIN/scripts/drain_merge_queue.py" \
     >> /tmp/torchcell-drain-merge-queue.log 2>&1 &
   ```

7. **Watch your branch to a terminal state.** First `cd "$MAIN"` -- on a
   successful landing the drainer deletes the worktree, so a session still
   parked inside it errors on a vanished cwd; parking in `$MAIN` keeps the shell
   valid afterwards. Then watch -- it blocks (up to ~570s) and returns a terse
   status line + an exit code; it does **not** print the banner (that is step 8):
   ```bash
   cd "$MAIN"
   $PY "$MAIN/scripts/merge_queue.py" watch "$BRANCH"
   ```
   **Run this Bash call with `timeout: 600000`** (the tool max). `watch` blocks
   up to 570s internally, just under the harness's 600s foreground cap; the
   default 120s Bash timeout would kill it early and you'd never see the clean
   exit code. Branch on the exit code -- this table is **your control flow
   only**, do not print it or its meanings to the user:

   | Exit | Meaning | Do |
   |------|---------|----|
   | 0 | Landed | Go to step 8 -- wrap up, then fire the banner last. |
   | 2 | Blocked, needs a human (e.g. push failed) | Go to step 8 -- the red banner is the final output; the branch stays `blocked`. |
   | 3 | Rebase conflict | Resolve it (next section), then loop back to step 7. Don't announce the conflict. |
   | 4 | Still queued after the window | **Auto-re-watch** (next paragraph) -- a deep queue just needs another window, not your attention. |
   | 5 | No queue entry | Stop (shouldn't happen right after step 5). |

   **Exit 4 -- auto-re-watch a deep queue.** A 570s window can elapse while
   branches ahead of you land one at a time; that is normal, not a stall. Do
   **not** stop or surface a banner yet. Re-trigger the drain and re-watch (back
   to the top of step 7):
   ```bash
   nohup "$PY" "$MAIN/scripts/drain_merge_queue.py" >> /tmp/torchcell-drain-merge-queue.log 2>&1 &
   $PY "$MAIN/scripts/merge_queue.py" watch "$BRANCH"
   ```
   Loop this up to ~3 windows. Only if it is *still* queued after that **and**
   the queue is not progressing (same position; check `merge_queue.py ls`) does
   it count as stalled -- then go to step 8 for the yellow banner. Silent waiting
   between windows is correct; emit nothing until a terminal banner.

8. **Wrap up, then fire the banner -- the LAST output.** Everything in this step
   except the final `banner` command happens *before* it, so the dove banner
   stays bottom-most (see Output discipline above):
   - **On a landing (exit 0):** if the branch closes a GitHub issue (the number
     is in the branch name, the PR, or the commit subjects, e.g. `(#21)`), close
     it now with a pointer to the landed SHA -- this is wrap-up and belongs
     *before* the banner. Add at most one short summary line only if it genuinely
     helps; otherwise none.
   - Then run the banner as the **final command**, and output nothing after it:
     ```bash
     $PY "$MAIN/scripts/merge_queue.py" banner "$BRANCH"
     ```
   - **On exit 2 or 4:** skip the wrap-up and run the same `banner "$BRANCH"`
     command -- it renders the red (needs-you) or yellow (leave-open) banner from
     the branch's row. It is still the last output; say nothing after it.

## Conflict resolution (exit 3 -- the only model-driven step)

Exit 3 means the branch is `blocked` because it couldn't fast-forward across a
branch that landed ahead of it in the queue. The drainer aborted its rebase and
left the worktree clean. Resolve it **here, in this session**:

1. `cd "$WT"`
2. `git fetch origin`
3. `git rebase origin/main`
4. For each conflicted file: read it, reconcile **both** sides correctly (this
   is the judgement step), then `git add <file>`.
5. `git rebase --continue`. Repeat 4-5 until the rebase reports it is complete.
6. Verify the tree is clean and the rebase is done:
   `git -C "$WT" status --short` must be empty (no `UU`, no `rebase in progress`).
7. Requeue, re-drain, then `cd "$MAIN"` (back out of the worktree, per step 7)
   and watch again:
   ```bash
   $PY "$MAIN/scripts/merge_queue.py" requeue "$BRANCH"
   nohup "$PY" "$MAIN/scripts/drain_merge_queue.py" \
     >> /tmp/torchcell-drain-merge-queue.log 2>&1 &
   cd "$MAIN"
   $PY "$MAIN/scripts/merge_queue.py" watch "$BRANCH"
   ```
   -- back to the exit-code table in step 7.

If the conflict is genuinely unresolvable, `git rebase --abort` and stop: the
branch remains `blocked` for a human. **Never** `--no-verify`, force-push, or
blind-discard to "unblock" -- fix the state, or leave it blocked.

## Important rules

- **Do not run `/merge-worktree` after enqueuing.** The drainer lands it; doing
  both reintroduces the concurrency the queue exists to avoid.
- The drainer lands **worktree -> origin** (`push HEAD:main`), so a stale or
  diverged local `main` never blocks it. The same flock guards the manual
  `/merge-worktree` fallback, so even a stray by-hand landing cannot interleave.
- **Only this session resolves its own branch's conflicts.** A conflict on
  another session's branch is that session's to resolve (or the cron leaves it
  `blocked` -- cron has no model and never resolves). Watch your own row.
- Reorder / drop / clear via `merge_queue.py mv|rm|clear`.
