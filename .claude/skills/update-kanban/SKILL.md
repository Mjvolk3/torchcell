---
name: update-kanban
description: Reconcile the torchcell GitHub Projects V2 kanban board (the board linked to Mjvolk3/torchcell). Empties the No-status column by giving every no-status item a default placement -- PRs go to Done or In review, issues go to Backlog. Never touches items that already have a status.
---

# Update Kanban (torchcell)

Ported from iBioFoundry-AI's `update-kanban` skill. The difference: that skill
hardcodes one board's field and option IDs, while this one resolves the board, the
Status field, and its option IDs by NAME on every run, so a renamed or rebuilt board
fails loudly instead of being written with stale IDs.

## Prerequisite: the `project` token scope

`gh` needs the `project` scope (read AND write). Check:

```bash
gh auth status 2>&1 | grep -i scopes
```

If `project` is missing, stop and ask the user to run (the `!` must be the first
character of the prompt; if it hangs on "Press Enter", run it in a tmux pane):

```
! gh auth refresh -h github.com -s project
```

## Goal

"No status" should always be empty. Every no-status item gets a default placement;
humans re-triage from there.

## Default placement for no-status items

| Item type                   | Default status |
|-----------------------------|----------------|
| Merged PR                   | Done           |
| Closed-unmerged PR          | Done           |
| Open PR (draft or not)      | In review      |
| Issue (any state)           | Backlog        |
| Draft issue (board-only)    | Backlog        |

Closed-unmerged PRs go to Done because torchcell lands by rebase + ff-only and then
CLOSES the PR (see CLAUDE.md "Git Worktrees"), so a landed branch shows as
closed-unmerged. Open PRs go to In review because an open PR awaits review by
definition.

## Steps

Work in the scratchpad directory (below: `$S`), not `/tmp`.

1. **Resolve the board.** Take the project(s) linked to the repo:

   ```bash
   gh repo view Mjvolk3/torchcell --json projectsV2 \
     -q '.projectsV2.Nodes[] | select(.closed | not) | {number, title, url}'
   ```

   Exactly one open board must come back. If zero or more than one, STOP and ask the
   user which board is torchcell's; do not pick one. Set `OWNER=Mjvolk3` and
   `NUMBER=<number>` (if the board's URL is under `/orgs/<org>/`, `OWNER` is that org).

2. **Resolve the Status field and option IDs by name.**

   ```bash
   PROJECT_ID=$(gh project view "$NUMBER" --owner "$OWNER" --format json -q .id)
   gh project field-list "$NUMBER" --owner "$OWNER" --format json > "$S/fields.json"
   FIELD_ID=$(jq -r '.fields[] | select(.name=="Status") | .id' "$S/fields.json")
   opt() { jq -r --arg n "$1" '.fields[] | select(.name=="Status") | .options[] | select(.name==$n) | .id' "$S/fields.json"; }
   BACKLOG=$(opt Backlog); IN_REVIEW=$(opt "In review"); DONE=$(opt Done)
   ```

   If `FIELD_ID` or any of the three option IDs is empty, STOP and report the
   Status options the board actually has (`jq '.fields[] | select(.name=="Status")'`).
   No guessing a near-match name.

3. **Dump the board and PR states** (always a large limit; the default page is small):

   ```bash
   gh project item-list "$NUMBER" --owner "$OWNER" --limit 2000 --format json > "$S/kanban.json"
   gh pr list --repo Mjvolk3/torchcell --state all --limit 2000 \
     --json number,state,mergedAt,url > "$S/prs.json"
   ```

   Sanity check: `jq '.items | length' "$S/kanban.json"` must be below the limit;
   if it equals it, raise the limit and redump.

4. **Classify no-status items** (`.status` null or missing). For each, read
   `.content.type` (`PullRequest` / `Issue` / `DraftIssue`) and `.content.url`.
   PR state comes from `$S/prs.json` joined on URL; a PR from another repo on the board
   is looked up individually with `gh pr view <url> --json state,mergedAt`. Buckets:
   merged PR, closed-unmerged PR, open PR, issue, draft issue.

5. **Show the plan before writing.** Print the counts per bucket and every item
   (number, title, URL, target status). Items that already have a status are not in
   the plan at all.

6. **Apply defaults**, serially (no bulk edit exists):

   ```bash
   gh project item-edit --project-id "$PROJECT_ID" --id "$ITEM_ID" \
     --field-id "$FIELD_ID" --single-select-option-id "$OPTION_ID"
   ```

   A failed edit stops the loop and is reported with the item; do not skip and continue.

7. **Verify and report.** Redump the board and confirm the no-status count is 0.
   Report counts per bucket plus number/title/URL of every item moved, so the user
   can spot-check and re-triage.

## Rules

- Never change the status of an item that already has one without explicit user
  instruction.
- Never add or remove items from the board; this skill only sets Status on items
  already there.
