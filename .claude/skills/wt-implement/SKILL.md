---
name: wt-implement
description: Implement an instruction file (plan note, scratch note, .claude/plans/ file) in an isolated git worktree. Creates branch, sets up env, executes the instructions, runs verification, rebases onto main, and optionally merges.
---

# Worktree Implement

Take an instruction file and execute it in an isolated git worktree. The instruction file can be a plan note, scratch note, `.claude/plans/` file, or any markdown with implementation specs.

**Usage:** `/wt-implement <freeform text>`

Examples:

```
/wt-implement notes/plan.coo-head.2026.06.04.md
/wt-implement plan.coo-head.2026.06.04
/wt-implement the coo classification head plan and merge when done
/wt-implement notes/scratch.2026.06.04.container-rebuild.md
```

---

## Phase 1: Parse Input and Setup

### Step 1: Find the instruction file

Extract it from the freeform input, in order: (1) literal `.md` path that exists; (2) dendron fname (dot-delimited) -> `notes/<fname>.md`; (3) keyword search over `notes/plan.*.md` and `.claude/plans/*.md`, present matches; (4) give up and ask.

Also extract modifiers: **"merge when done" / "merge after" / "to completion"** -> auto-merge after PR; **"no merge" / "just PR"** -> stop at PR (default).

### Step 2: Read the instruction file

Read in full: files to create/modify, verification steps, execution order.

### Step 3: Derive branch name

Strip `notes/`, `plan.`, `.claude/plans/`, date suffixes, `.md`; truncate to 50 chars; prefix `plan/`.

- `notes/plan.coo-head.2026.06.04.md` -> `plan/coo-head`
- `notes/scratch.2026.06.04.container-rebuild.md` -> `plan/container-rebuild`

### Step 4: Create worktree

```bash
git worktree list
```

If not present:

```bash
git worktree add ~/Documents/projects/torchcell.worktrees/plan/<slug> -b plan/<slug>
```

### Step 5: Setup worktree

```bash
cd ~/Documents/projects/torchcell.worktrees/plan/<slug>
bash scripts/setup-worktree.sh
```

Then layer on validation: confirm `.env` sources cleanly in bash, and verify `import torchcell` resolves to the worktree:

```bash
bash -c 'set -a; source .env 2>&1; echo "EXIT: $?"'
PYTHONPATH=$(pwd) ~/miniconda3/envs/torchcell/bin/python -c "import torchcell; print(torchcell.__file__)"
```

The import path must point at the worktree, not main.

### Step 6: cd into worktree

```bash
cd ~/Documents/projects/torchcell.worktrees/plan/<slug>
```

**CRITICAL:** ALL subsequent edits happen in the worktree. Verify `pwd` shows the worktree path before any edit.

---

## Phase 2: Implement

### Step 7: Load context

Re-read the instruction file from the worktree. For each referenced file, read its current worktree state and any context files mentioned.

### Step 8: Execute file specifications

For each spec in order: read current file; apply changes (Edit / Write / `dendron-cli` for new notes); verify syntax.

**Rules:** follow specs exactly, no extra scope; respect "Not fixing" / "Out of scope"; all project conventions apply (no fallback mechanisms, image-output + timestamp pattern, never edit dendron frontmatter).

### Step 9: Run verification

Execute the instruction file's verification. Common pattern:

1. **Tests**: `~/miniconda3/envs/torchcell/bin/python -m pytest tests/torchcell/<area> -xvs`
2. **Type check**: `~/miniconda3/envs/torchcell/bin/python -m mypy <files>`
3. **Lint**: `~/miniconda3/envs/torchcell/bin/python -m ruff check <files>` (once ruff is adopted; until then pre-commit runs black/isort/pyupgrade at commit time)
4. **Import check**: `~/miniconda3/envs/torchcell/bin/python -c "from torchcell.<module> import <thing>"`

If a check fails, fix and re-run. Do not proceed until all pass. If the instruction file specifies no verification, default to: mypy on changed `.py` files.

---

## Phase 3: Finalize

### Step 10: Update notes and commit

1. `/update-notes` -- dendron module notes + weekly task note
2. `/stage --all` -- stage all blocks without prompting (isolated feature branch)
3. `/commit` -- descriptive message

---

## Phase 4: Land on Main

### Step 11: Rebase onto main with retry

```bash
git fetch origin main
git rebase origin/main
```

**Retry logic** (concurrent worktree merges): on conflict -> `git rebase --abort`; wait 10s; `git fetch origin main`; retry. If conflicts again, attempt resolution:

- Weekly notes (`user.Mjvolk3.torchcell.tasks.weekly.*.md`): auto-resolved by `.gitattributes merge=weeklynote` (driver `scripts/git_merge_weekly_note.py`, registered by `setup-worktree.sh`).
- `__init__.py`: keep both sides (additive).
- Other files: inform the user and stop.

Max 3 attempts before stopping.

### Step 12: Push and create PR

```bash
git push -u origin plan/<slug>
gh pr create --title "<concise title>" --body "$(cat <<'EOF'
## Summary
<key changes as bullets>

## Instruction file
<dendron link or path>

## Verification
- Tests: PASS
- Mypy: PASS

Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

### Step 13: Merge (if requested)

Only if the user said "merge when done" / similar. Prefer handing off to `/merge-worktree <branch>`, which lands via rebase + ff-only, closes the PR, and cleans up worktree + branches. If doing it inline:

1. `gh pr merge <number> --merge`
2. `git checkout main && git pull --ff-only origin main`
3. `git worktree remove --force ~/Documents/projects/torchcell.worktrees/plan/<slug>`
4. `git branch -d plan/<slug>`
5. `gh api repos/Mjvolk3/torchcell/git/refs/heads/plan/<slug> --method DELETE`

If NOT merging, report the PR URL and stop.

---

## Important Rules

- ALL edits in the worktree directory, never main.
- Follow the instruction file exactly -- no extra scope.
- Respect "Not fixing" / "Out of scope" sections.
- Run the instruction file's verification, not a generic suite.
- Rebase retry logic is essential for concurrent workflows.
- Do NOT ask extra approval questions -- tool approval prompts are the gates.
- If the instruction file has no explicit verification, default to mypy on changed `.py` files.
