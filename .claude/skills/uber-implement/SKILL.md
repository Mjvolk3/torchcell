---
name: uber-implement
description: End-to-end autonomous pipeline -- plans with /plan-4.8 (3 scouts, deliberator, plan-writer, reducer-critic), then implements with /wt-implement in an isolated worktree. No manual approval between plan and implementation.
---

# Uber Implement

End-to-end pipeline: plan a change with the full `/plan-4.8` pipeline, then immediately implement it in a worktree via `/wt-implement`. No manual approval gate between planning and implementation.

## Usage

`/uber-implement <request> [merge when done]`

The request is the same natural language you would pass to `/plan-4.8`. Append "merge when done" to auto-merge the PR after implementation.

## Phase A: Plan (plan-4.8 phases 0--5)

Run `/plan-4.8` phases 0 through 5 exactly as documented:

1. **Phase 0: Setup** -- create the plan note (`dendron-cli note write --fname "plan.<slug>.YYYY.MM.DD"`)
2. **Phase 1: Three Parallel Scouts**
3. **Phase 2: Deliberation**
4. **Phase 3: Plan Draft** (~300 lines, dense)
5. **Phase 4: Reducer-Critic Loop** (max 3 iterations)
6. **Phase 5: Weekly Task Note + Stage** -- append the pending bullet, stage plan note + weekly note. **Skip** the `code notes/<fname>.md` editor open and the printed `## Summary` block -- those hand control back to the user, and uber-implement does not hand back.

**Skip the "Interactive Revision" section entirely.** Hand off straight to Phase B. The reducer-critic in Phase 4 is the only quality gate.

## Phase B: Commit + Implement

1. **Commit the staged plan note + weekly note** so they exist in history even if implementation fails:

    ```bash
    git commit -m "plan: <slug>"
    ```

2. **Hand off to `/wt-implement`:**
   - Pass the plan note path: `notes/plan.<slug>.YYYY.MM.DD.md`
   - Include "merge when done" if the user said so; otherwise default to "just PR".

`/wt-implement` handles worktree creation, implementation, verification, rebase, PR, and optional merge from here.

## Phase C: Report

```text
## Uber Implement Complete

Plan:   notes/plan.<slug>.YYYY.MM.DD.md ([[plan.<slug>.YYYY.MM.DD]])
Branch: plan/<slug>
PR:     <PR URL>
Status: <merged | PR created>

Files changed:
  - torchcell/.../file1.py (NEW)
  - torchcell/.../file2.py (MODIFY)
```

## Important Rules

- **No approval gate between plan and implement.** The reducer-critic in `/plan-4.8` Phase 4 is the only quality gate.
- **All `/plan-4.8` rules apply** during Phase A; **all `/wt-implement` rules apply** during Phase B.
- **Commit the plan note before implementation.**
- **Do NOT use EnterPlanMode/ExitPlanMode** -- this skill replaces plan mode.
- **Do NOT ask extra approval questions** -- tool approval prompts are the gates.
- **Token budget**: significant context across both phases. For very large changes, scope scouts tightly.
- **Use high or xhigh effort** for the implementation phase.

## Example

```text
/uber-implement Add a COO classification head for fitness regression and wire it through the dcell model. merge when done
```
