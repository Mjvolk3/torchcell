---
name: stage
description: Smart staging with auto-detected file blocks and interactive override. Use before /commit.
---

# Smart Stage

Stage files for commit using auto-detected logical blocks. The user picks which blocks to stage via interactive selection.

**Usage:** `/stage` -- interactive block selection
**Usage:** `/stage --all` -- stage all blocks without prompting (for autonomous pipelines like `/wt-implement`)

## Auto-stage mode (`--all`)

When invoked with `--all`, skip Steps 3--4 entirely. Stage **all** detected blocks without prompting. All other rules still apply (trio grouping, no secrets, no `git add -A`, scratch exclusion). Jump from Step 2 to Step 5.

## Workflow

```
/stage -> /commit
```

## Step 1: Scan unstaged/untracked files

Run `git status` (never `-uall`) for modified, deleted, and untracked files. If there are none, inform the user and stop.

## Step 2: Auto-detect file blocks

### Scratch exclusion (default-off, applied FIRST)

**Exclude every `notes/scratch.*` file from ALL blocks by default** (CLAUDE.md "Scratch Files"). A scratch file is staged ONLY when the user, in this same invocation, explicitly names that specific file. A bare `/stage`, `/stage --all`, or "stage my changes" never stages any `notes/scratch.*` -- not even under `--all`. When scratch content matures, graduate it into a `plan.*`/topic note and stage that.

**Hard rule, no exceptions even on explicit request:** never stage a scratch file with `delete` in its filename.

Drop excluded scratch files from the list before block detection; never surface them as a selectable block.

### The Commit Trio rule (bidirectional)

**Forward:** Any block containing a `.py` file under `torchcell/` or `tests/torchcell/` MUST also include:

1. The paired dendron note (`notes/<dendron-path>.md`) -- if it's in the changed list.
2. The paired test file (`tests/torchcell/.../test_*.py`) or source file -- if changed.
3. The current weekly note (`notes/user.Mjvolk3.torchcell.tasks.weekly.*.md`) -- always pulled into any block containing Python files.

**Reverse:** Any note matching `notes/torchcell.*.md` MUST be grouped with its paired source file. If the source has no changes, do NOT stage the note -- it waits for the source commit. Never commit notes orphaned from their source.

The weekly note appears in the FIRST Python block only. This enforces the atomic commit rule: source + note + weekly travel together.

### Block detection rules (priority order)

1. **Python trio**: a `.py` source + its `notes/<dendron-path>.md` + its `tests/torchcell/` counterpart + the weekly note. Related `.py` in one package can group (e.g. all under `torchcell/transforms/`).
2. **Skill bundles**: all files under a `.claude/skills/<name>/` directory grouped together.
3. **Config clusters**: related config (e.g. `pyproject.toml` + `.pre-commit-config.yaml` when both have lint changes).
4. **Experiment scripts + notes**: `experiments/<id>/scripts/*.py` paired with `notes/experiments.<id>.scripts.<name>.md`.
5. **Shell scripts + notes**: `.sh` paired with `notes/scripts.<name>.md`.
6. **Standalone notes**: `notes/*.md` not already paired.
7. **Other files**: anything else (configs, docs).

A file appears in exactly one block (highest-priority rule wins).

## Step 3: Present blocks to user

Numbered blocks with file lists. Mark the weekly note explicitly:

```
Detected file blocks:
[1] transforms package (4 files):
    torchcell/transforms/coo_regression.py
    tests/torchcell/transforms/test_coo_regression.py
    notes/torchcell.transforms.coo_regression.md
    notes/user.Mjvolk3.torchcell.tasks.weekly.2026.06.md  (weekly)
[2] Scripts: scripts/build.sh, notes/scripts.build.md
[3] Standalone notes: notes/plan.coo-head.2026.06.04.md
```

`notes/scratch.*` files are NEVER shown as a block.

## Step 4: User picks blocks

Use `AskUserQuestion` with `multiSelect: true`: each detected block as an option, plus "All".

## Step 5: Pre-stage reminders (informational, not blocking)

- If selected files include `.py` under `torchcell/` or `tests/torchcell/`: "Source files detected. Consider running `/update-notes`, `/mypy`, and `/ruff` first."
- If `torchcell/models/*.py` or `torchcell/sequence/*.py` changed: "Documented modules changed -- `/commit` will rebuild Sphinx autodoc to verify."

## Step 6: Stage selected blocks

- `git add <files>` for chosen blocks (explicit paths, never `git add -A` or `git add .`).
- `git rm` for deleted files.
- Never stage `.env`, credentials, or secrets -- warn if detected.

## Step 7: Confirm

`git diff --cached --name-status` to show what is staged.

## Important Rules

- NEVER `git add -A` or `git add .`.
- NEVER stage `.env`, credentials, or secrets -- warn the user.
- NEVER stage `notes/scratch.*` unless the user explicitly named that file (default-off). A scratch file with `delete` in its name is never staged, even on explicit request.
- Use `git rm` for deleted files.
- A file appears in exactly one block.
- The weekly note is always with the first Python block -- never staged alone when Python files are present.
- Do NOT ask extra approval questions -- tool approval prompts are the gates.

## Example Invocations

- `/stage` -- interactive staging
- "stage my changes"
