---
name: update-notes
description: "Meta-skill: update dendron module notes for changed source files, then update the weekly task note. Combines /update-src-notes + /update-tasks-weekly into one command. Run before /stage and /commit."
---

# Update Notes

Thin meta-skill. It does **no weekly or source-note logic of its own** -- it runs the two canonical skills in order and prints a combined summary.

> **Why this is a delegator, not a copy.** Inlining copies of the child workflows causes drift (e.g. an inlined weekly half misses rollover / new-ISO-week creation). **Never re-inline the logic here.** The two child skills are the single source of truth.

## Workflow

```
(edit source) -> /update-notes -> /stage -> /commit
```

## Arguments

Optional file path arguments pass straight through to the source-notes step (Part 1):

- **With arguments** (e.g. `/update-notes torchcell/models/dcell.py scripts/build.sh`): update only those files.
- **No arguments**: source-notes step auto-discovers (staged first, then modified).

---

# Part 1: Source module notes

Read and execute the **entire** `update-src-notes` skill at `.claude/skills/update-src-notes/SKILL.md`, passing through any file-path arguments. That skill is authoritative for target-file discovery, dendron-note mapping/creation, the dated-section append, and the intentional-stance self-check.

Keep the list of created/updated module notes for the combined summary.

If that skill stops early (no source files found), do **not** stop -- continue to Part 2. Weekly rollover and format validation must still run on a source-empty invocation.

---

# Part 2: Weekly task note

Read and execute the **entire** `update-tasks-weekly` skill at `.claude/skills/update-tasks-weekly/SKILL.md`. That skill is authoritative for worktree detection, new-ISO-week note creation, the rollover policy, weekly entry authoring, dead-link/bad-anchor validation, strict-flat-format validation, done-above-pending reordering, and staging the weekly note.

Run it even when Part 1 found nothing.

---

# Part 3: Combined summary

```
Source: staged files (or: modified files -- nothing staged / no source files)

Created notes:
  - torchcell.models.dcell (new)

Updated source notes:
  - torchcell.transforms.coo_regression (added 2026.06.04 section)

Weekly note:
  - Week-boundary: created 2026.23 (new ISO week) [or: none]
  - Rolled forward: 3 open tasks from 2026.22 -> 2026.23
  - Format fixes: removed 4 blank lines between bullets; fixed 1 dead link
  - Added 2 entries under ## 2026.06.04

All notes staged. Run /stage -> /commit to finalize.
```

---

## Important rules

- Never edits dendron frontmatter, never inlines child-skill logic, never asks extra approval questions.
- The commit trio still applies: child skills stage notes/weekly; paired **source** files are staged by `/stage`. Do not commit here.

## Example invocations

- `/update-notes` -- auto-discover source; then rollover + weekly + format.
- `/update-notes torchcell/models/dcell.py` -- one module + weekly.
- "update notes for changed files"
