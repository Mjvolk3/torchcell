---
name: update-tasks-weekly
description: Update the weekly task note with a bullet point linking to reviewed/modified files, then auto-stage the weekly file.
---

# Update Weekly Tasks

Update the current weekly task note with progress and links. After updating, auto-stage the weekly file.

## Task Note Location

The weekly note lives at `notes/user.Mjvolk3.torchcell.tasks.weekly.YYYY.WW.md` where `WW` is the ISO week. Find the current one by globbing `notes/user.Mjvolk3.torchcell.tasks.weekly.2026.*.md`.

### Worktree Detection

**If `pwd` contains `.worktrees/`**, you are in a worktree. Worktrees write to a **child note** of the main weekly note to avoid merge conflicts.

1. If `pwd` does not contain `.worktrees/`, use the main weekly note.
2. If in a worktree, find the worktree-specific child note:
   - Read the main weekly note.
   - Find the line containing the branch name (from `git branch --show-current`) and a `[[...]]` dendron link.
   - The linked dendron path is the target for all updates.
3. If no matching line exists, inform the user and stop. They must first add a single-line entry in the main weekly (on `main`) linking to the worktree note.

**Example:** branch `inference-dataset-3`, main weekly has:

```markdown
- [ ] `inference-dataset-3` worktree [[user.Mjvolk3.torchcell.tasks.weekly.2026.06.inference-dataset-3]]
```

All updates go to `notes/user.Mjvolk3.torchcell.tasks.weekly.2026.06.inference-dataset-3.md`. (torchcell already uses this child-note pattern, e.g. `...2026.05.010-inferece-datsaet-2.md`.)

## Instructions

Run in order. Steps 0, 4, 6 are unconditional -- run them even when no source changed.

0. **Resolve the target weekly note (week-boundary handling).** Compute today in Chicago time: `TZ=America/Chicago date +%G.%V` gives `YYYY.WW` (ISO year + ISO week -- use `%G/%V`, never `%Y/%U`); `TZ=America/Chicago date +%Y.%m.%d` gives the date. Target = `notes/user.Mjvolk3.torchcell.tasks.weekly.<YYYY>.<WW>.md`. If it does **not** exist (new ISO week), create it with `dendron-cli note write --fname "user.Mjvolk3.torchcell.tasks.weekly.<YYYY>.<WW>"` -- never the Write tool. Then run rollover (Step 4) from the most recent prior weekly note into it. (Worktree runs target the child note instead.)
1. **Read the target weekly note.**
2. **Find today's date section** (`## YYYY.MM.DD`). Create it if missing (append below the last dated section -- newest at bottom).
3. **Author task entries** (Task Entry Format below).
   - **The one-sentence explainer is the product.** A reader scanning the list must understand what was done and why it matters without opening anything. One sentence, two at most. The dendron link carries every detail -- file lists, test counts, function names do **not** belong in the bullet.
   - Link with `[[dendron.path]]` (no aliases -- `dendron.yml` sets `aliasMode: none`).
   - If a linked script generated images, embed them in the linked note, never in the weekly note.
4. **Roll forward / reconcile every open `- [ ]` bullet** under any past H2 (strictly older than today) per the Rollover policy. The floor is absolute: no open task is left under a stale past date.
5. **Reorder within every H2 touched**: `- [x]` above `- [ ]`, relative order preserved within each group.
6. **Validate format** (Format Validation section) across the whole file.
7. **Auto-stage**: `git add` the target weekly note (and any newly created new-week note).

## Rollover policy

Compute today in Chicago time. Walk **every** `- [ ]` bullet under an H2 **strictly older** than today. Each gets exactly one outcome:

**A. Reconcile to done (check first).** If the work is demonstrably complete -- a later `- [x]` covers the same work (same `[[plan....]]` link or deliverable), or the linked note shows it landed -- flip it to `- [x]` **in place** (same H2, wording, links). Count as "reconciled". This stops implemented `Plan:` stubs from rolling forever.

**B. Roll forward.** Otherwise append it **verbatim** (every `[[wiki-link]]`/`#tag`) to today's H2 in the target note, then remove it from the source H2. On a week boundary the target is the new-ISO-week note from Step 0.

Priority tags do **not** gate rollover -- every open task rolls. They change only surfacing in the summary:

| Inline tag        | Extra surfacing after the roll                                                       |
|-------------------|-------------------------------------------------------------------------------------|
| `#high`           | Call out by name in the summary every run until closed.                             |
| `#medium`         | Call out on the Monday week-boundary run.                                           |
| `#low`            | Silent unless the task's original date is >= 4 weeks old -- then "consider closing".|
| (none)            | Silent. If open > 7 days from its original date, "consider tagging or closing".     |

Completed `- [x]` bullets are immutable history: never move, delete, or reorder across H2s (the within-H2 done-above-pending reorder is the sole exception). Past H2 date headers are immutable -- never rename or remove them, even when emptied.

### Worktree note

Inside a worktree, the target is the resolved child note; the rollover applies to it independently. Past-date H2s in the main weekly are not touched by worktree runs.

## Task Entry Format

```markdown
- [x] One-sentence explainer of completed work [[dendron.path.to.note]]
- [ ] One-sentence explainer of pending task [[dendron.path.to.note]]
```

## Linking Rules

- Python source: `[[torchcell.module_name]]` -> `notes/torchcell.module_name.md`
- Experiment scripts: `[[experiments.<id>.scripts.<name>]]`
- Scratch notes: `[[scratch.YYYY.MM.DD.HHMMSS-title]]`
- If a linked note references an image-generating script, embed images in that note: `![desc](assets/images/...)`

### Section Anchor Links

When a note has multiple dated sections, link to the specific one: `[[note.path#anchor-slug]]`.

**Always verify the H2 exists before computing an anchor:**

```bash
grep -n "^## " notes/torchcell.some_module.md
```

Compute the anchor from the exact header (GitHub-style slugification: strip `## `, lowercase, remove chars except `[\w\s-]`, spaces->hyphens, do NOT collapse hyphens):

```bash
python3 -c "import re,sys; h=sys.argv[1]; t=re.sub(r'^#+\s*','',h).lower(); t=re.sub(r'[^\w\s-]','',t); print(t.replace(' ','-'))" "## 2026.06.04 - COO classification head"
```

## Format Validation

Scan the **entire** file (every H2) and fix:

**Structure:**

- H3+ headings under date sections -- flatten into bullets.
- Non-`## YYYY.MM.DD` H2s -- move their bullets into the correct dated H2 and delete the stray H2.
- Horizontal rules (`***`, separator `---`) -- remove.
- Nested sub-lists -- flatten to one level.
- **Blank lines between bullets within a date section -- remove.** Consecutive bullets under one H2 are one tight flat list. (Exactly one blank line between the H2 and its first bullet, and before the next H2, is correct.)

**Bullet length (structural, not stylistic -- fix it):**

- One sentence, two at most, plus its `[[dendron.link(s)]]`. It states *what was done and why it matters*; it does not enumerate files, test counts, or mechanics.
- Condense any longer bullet in place. Preserve the `[x]`/`[ ]` state, every `[[wiki-link]]`/`#tag`, and the core claim. Push cut detail into the linked note (create/append it first if the detail exists nowhere else -- never delete unique information).

**Links (no dead or unlinked references):**

- Every `[[dendron.path]]` must resolve to an existing `notes/<dendron.path>.md` (strip any `#anchor`, map dots to path).
- Every `#anchor` must be a real H2 in the target note (verify with grep + the anchor one-liner). Fix or drop the anchor.

**Never touch:** the frontmatter `---` block.

Report every category of fix, with counts, in the summary.

## Files to Skip

Do not create weekly entries for:

- **Tag notes** (`notes/tags.*.md`) -- Dendron tag infrastructure.
- **Config/tooling files** (CLAUDE.md, README.md, .claude/skills/, .vscode/) -- mention inline in the commit summary, no `[[link]]` entry.

## Example

```markdown
## 2026.06.04

- [x] Investigated YLR313C-B coordinate mismatch in the SGD R64-4-1 GFF [[scratch.2026.06.04.112028-010-inference-dataset-3-table-investigate-YLR313C-B]]
- [ ] Inference dataset 3 panel tables
```
