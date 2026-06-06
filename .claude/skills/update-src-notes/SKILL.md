---
name: update-src-notes
description: Update dendron module notes for changed source files (Python, Bash, etc.). Checks staged files first, then falls back to modified (unstaged) files. Appends dated sections and stages updated notes.
---

# Update Source Notes

Update dendron module notes for source files that have changed. Module notes are staged when done.

## Workflow

```
(edit source) -> /update-src-notes -> /update-tasks-weekly -> /stage -> /commit
```

Or use `/update-notes` (meta-skill) which runs this + `/update-tasks-weekly` in one command.

## Arguments

Optional file path arguments:

- **With arguments** (e.g. `/update-src-notes torchcell/models/dcell.py experiments/010-kuzmin-tmi/scripts/query.py`): update only those files. Use `git diff HEAD -- <file>` for the diff (staged + unstaged). If a file has no diff against HEAD (new untracked), read the full file instead.
- **No arguments**: auto-discover (Step 1 staged-then-modified fallback).

If no files are found (no args, nothing staged, nothing modified), inform the user and stop.

## Step 1: Determine target source files

- **If arguments provided**: use them directly; verify each exists.
- **If no arguments**, two-tier discovery:
  1. Staged first: `git diff --cached --name-only` for source files (`.py`, `.sh`, etc.).
  2. If nothing staged, fall back to modified: `git diff --name-only`.

Report which tier was used. If empty after both tiers, inform the user and stop.

## Step 2: Map files to dendron notes

Convert path separators to dots and drop the extension:

| Source file                                      | Dendron note                                            |
|--------------------------------------------------|---------------------------------------------------------|
| `torchcell/models/dcell.py`                      | `notes/torchcell.models.dcell.md`                       |
| `torchcell/transforms/coo_regression.py`         | `notes/torchcell.transforms.coo_regression.md`          |
| `experiments/010-kuzmin-tmi/scripts/query.py`    | `notes/experiments.010-kuzmin-tmi.scripts.query.md`     |
| `scripts/build.sh`                               | `notes/scripts.build.md`                                |
| `notes/assets/scripts/add_frontmatter.py`        | `notes/notes.assets.scripts.add_frontmatter.md`         |

**Rules:**

- Any source file can be synced -- package modules, experiment scripts, shell scripts.
- Convert path separators to dots, keep hyphens as-is, drop the extension.
- The dendron note is `notes/<dendron_path>.md`.
- If the note does not exist, **create it** with `dendron-cli note write` (Step 2b).

## Step 2b: Create missing notes

```bash
dendron-cli note write --fname "<dendron_path>"
```

e.g. `torchcell/models/dcell.py` with no note -> `dendron-cli note write --fname "torchcell.models.dcell"`.

This creates `notes/<dendron_path>.md` with proper dendron frontmatter, then treated as existing in Step 3.

**Important:** `dendron-cli` creates notes ending with `---\n` and no trailing blank line. When editing newly created notes, match on the unique `created:` timestamp line + `---` rather than `---` followed by a blank line.

Track newly created notes for the Step 5 summary.

## Step 3: Update each module note

For each existing module note:

1. **Read the note** to understand current content.
2. **Read the diff** for the source file:
   - explicit file arguments: `git diff HEAD -- <source_file>` (empty -> read full file)
   - staged (tier 1): `git diff --cached -- <source_file>`
   - modified (tier 2): `git diff -- <source_file>`
3. **Check for an existing H2 with today's date** (`## YYYY.MM.DD`).

### If today's date section already exists

- Append new subsections/bullets for changes not already covered. Do not duplicate.

### If no section for today's date exists

- **Append** a new H2 at the bottom (before trailing blank lines):

```markdown
## YYYY.MM.DD - Brief Title

One-paragraph summary describing what changed and why.

### Subsection (optional)

- Bullet points with specifics
```

**Writing guidelines (intentional stance -- "why the change, for what"):**

- Lead with **purpose**, not mechanics. The diff shows *what* changed; the note captures *why*.
- The brief title names the intent (e.g. "Route fitness through COO classification head", not "Add coo_regression function").
- Summary paragraph: 1-3 sentences on motivation and impact.
- Use bullets sparingly. Omit trivial details (import reorder, lint) unless deliberate.
- Short code snippets only when they clarify a new interface.
- No Unicode emojis (breaks xelatex PDF export).

**Intentional-stance self-check (mandatory).** After drafting each dated section, re-read its first sentence and title. If they describe *what changed* rather than *why it exists*, it is a changelog, not documentation -- **rewrite it.** A reader six months out must learn the motivation without reading the diff. Do not stage a section that fails this check.

## Step 4: Stage updated module notes

`git add <note_path>` for each note modified in Step 3.

## Step 5: Print summary

```
Source: staged files (or: modified files -- nothing was staged)

Created notes:
  - torchcell.models.dcell (new)

Updated module notes:
  - torchcell.transforms.coo_regression (added 2026.06.04 section)

All module notes staged.
```

## Important Rules

- Create missing notes with `dendron-cli note write` -- never the Write tool (it would lack frontmatter).
- NEVER modify dendron YAML frontmatter (the `---` block).
- NEVER remove or rewrite existing dated sections -- only append.
- Preserve each note's existing style.
- No Unicode emojis.
- Do NOT ask extra approval questions -- tool approval prompts are the gates.

## Example Invocations

- `/update-src-notes` -- auto-discover (staged then modified)
- `/update-src-notes torchcell/models/dcell.py`
- `/update-src-notes scripts/build.sh torchcell/cell.py`
- "update source notes for changed files"
