---
name: commit
description: Commit staged changes with an auto-generated bulleted commit message. Run /stage first to stage files.
---

When the user asks to commit changes, follow this workflow. The user's tool approval prompts serve as the approval gates -- do NOT ask extra confirmation questions.

## Step 1: Check what's staged

Run `git diff --cached --name-status`. If nothing is staged, tell the user to run `/stage` first and stop.

## Step 1.5: Sphinx autodoc verify (only if documented modules are staged)

torchcell's Sphinx docs are **autosummary/autodoc-generated** -- `docs/source/modules/models.rst` and `docs/source/modules/sequence.rst` iterate `torchcell.models.models` / `torchcell.sequence.data_classes` and pull docstrings via `sphinx.ext.autodoc`. There are **no hand-written narrative doc pages to edit.**

From the staged list, identify `.py` files under **`torchcell/models/`** or **`torchcell/sequence/`** (exclude `__init__.py`, `__pycache__`, tests).

- **If none are found: skip this step entirely.**
- **If found:** the source docstrings ARE the docs. Do not edit `.rst` by hand. Instead:
  1. Confirm new/changed public objects have docstrings (autodoc renders them verbatim).
  2. Rebuild to verify the autodoc build is clean:
     ```bash
     make -C docs html 2>&1 | tail -20
     ```
  3. If the build errors (import failure, missing object in the autosummary list), fix the source/docstring -- not the generated `.rst`. Do not stage `docs/generated/` (it is build output).
  4. Print a one-line note: which documented module changed and that the autodoc build is clean.

This is the full, honest docs mapping -- only `models` and `sequence` are documented; every other module has no doc page, so this step is usually a no-op.

## Step 1.6: Scratch backstop (default-off)

Scan the staged list for any `notes/scratch.*` file (defense-in-depth in case one slipped past `/stage`).

- If a staged `notes/scratch.*` is present and the user did NOT, in this same request, explicitly ask to commit that specific scratch file: **unstage it (`git restore --staged <file>`), tell the user it was excluded, and continue** with the rest. Per CLAUDE.md "Scratch Files", scratch is ephemeral and default-off; graduate durable content into a `plan.*`/topic note instead.
- **Hard rule, no exceptions even on explicit request:** if a staged scratch file has `delete` in its filename, unstage it and refuse.

## Step 1.7: Commit trio check (bidirectional)

Scan the staged list for orphans:

- **Orphaned notes:** If `notes/torchcell.*.md` is staged but its paired `.py` source is NOT staged, **warn and stop**. Notes must not be committed without their source.
- **Orphaned source:** If a `.py` file under `torchcell/` or `tests/torchcell/` is staged but its paired dendron note is NOT staged, **warn**. Suggest running `/update-notes` first.
- **Exceptions:** `notes/plan.*.md`, `notes/user.*.md` (incl. weekly), and `notes/experiments.*.md` for experiment scripts are exempt from the source-pairing check. `notes/scratch.*` is handled by Step 1.6 (default-off).

If violations are found, do NOT proceed. Print the violation and stop.

## Step 2: Match commit style

Run `git log -5 --oneline` to understand the repo's commit message style.

## Step 3: Draft and run commit

1. Draft a message: clear summary line (<70 chars), bulleted changes (max 10), WHAT + WHY not implementation detail. Format:

   ```
   Brief summary of changes

   - First change
   - Second change

   Co-Authored-By: Claude Code (commit) <noreply@anthropic.com>
   ```

2. Present the message, then run the commit (the user approves via the tool prompt):

   ```bash
   git commit -m "$(cat <<'EOF'
   <commit message here>

   Co-Authored-By: Claude Code (commit) <noreply@anthropic.com>
   EOF
   )"
   ```

3. Run `git status` after to verify success.

## Handle failures

- If pre-commit hooks fail (black/isort/pyupgrade reformat files), re-stage the reformatted files and create a NEW commit (never amend unless explicitly asked).
- If there is nothing to commit, inform the user.

## Important Rules

- NEVER amend commits unless explicitly requested.
- NEVER skip hooks or use `--no-verify`.
- NEVER commit without the co-authored-by line.
- DO NOT push unless explicitly asked.
- Keep bullets concise; group related changes to stay under 10.
- Do NOT ask extra approval questions -- the tool approval prompts are the gates.

## Example Invocations

- `/commit`
- "commit staged changes"
