---
name: plan-4.8
description: Three scouts explore the codebase in parallel, argue the approach, and a plan-writer agent synthesizes the deliberation into a concise Dendron note. A reducer-critic then smart-tightens the plan (flagging low-value content for cuts, not truncating) until the plan is dense and human-readable. Built for Claude Opus 4.8 -- literal instruction-following, self-verification, long-running coherence -- so plans describe what and why, not every line of code.
---

# Plan-4.8

A lightweight, high-level planning skill built for Opus 4.8 implementation agents.

## Why this skill exists

Opus 4.8 surfaces three behaviors that reshape planning:

1. **Takes instructions literally.** Over-constrained plans get copy-pasted into shallow implementations. Leave judgment room.
2. **Catches its own logical faults and verifies its own outputs.** Implementers self-correct; plans need not enumerate every edge case.
3. **Works coherently for hours.** Plans can describe broad strokes; the implementer sustains the multi-step work.

Together: a plan that reads like a design memo -- file paths, gotchas, key decisions, narrative approach -- outperforms a plan stuffed with class bodies and function skeletons.

**Target length: ~300 lines.** Enforced by a reducer-critic that identifies low-value content and proposes cuts. A dense 310-line plan beats a mutilated 250-line plan.

## Usage

`/plan-4.8 <request>`

Freeform request. Scouts read the codebase to ground the plan.

## Web-search policy (library currency)

The model's training cutoff is January 2026; today may be later. Any claim about a third-party library (current version, new API, deprecation, breaking change) can be stale from model memory alone. Agents **must** web-search before recommending library-specific patterns.

- **Repo-pinned version first.** Check `env/requirements.txt`, `env/requirements_dependent.txt`, `pyproject.toml` for the installed version. Recommendations must work with it.
- **Web-search if recommending a newer version/feature.** Confirm via WebSearch/WebFetch that the feature exists in a version available today. State the version.
- **Do not rely on model memory for release dates, versions, or deprecations.**
- **Cite sources briefly** inline: `torch_geometric 2.6+ (confirmed 2026-04-18 via pyg.org)`.

Applies to every agent (scouts, deliberator, plan-writer, reducer-critic).

## Paired-note policy (design intent of in-scope files)

Every source file in this repo has a paired dendron note: `torchcell/<module>.py` pairs with `notes/torchcell.<module>.md`; `experiments/<id>/scripts/<name>.py` pairs with `notes/experiments.<id>.scripts.<name>.md`; `scripts/<name>.sh` pairs with `notes/scripts.<name>.md`. Paired notes record dated design decisions, prior plan references, rejected alternatives, and open follow-ups.

**Required for every agent:** before recommending an edit to or deletion of a file, read its paired note. Source alone does not communicate which parts are intentional/stable versus provisional/awaiting replacement.

Classify each in-scope file:

| Classification | Indicators in the note                                                                  | Planning implication                                                        |
|----------------|------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| `stable`       | Multiple dated entries; invariants documented; referenced by other notes/plans          | Preserve contracts; enumerate downstream callers; justify signature changes |
| `provisional`  | Note marks code as first pass / stopgap / placeholder pending a named follow-up         | Reshape freely; state it is the redesign the note anticipated              |
| `in-flux`      | Recent entries contradict earlier ones; an open `plan.*` note targets the same file     | Coordinate with the in-flight work; do not race or duplicate it            |
| `undocumented` | Note empty, missing, or auto-generated stub only                                        | Flag it; treat as `provisional` unless code structure shows otherwise      |

The classification belongs in the plan's `## Relevant Files` table (Phase 3).

## Phase 0: Setup

1. Summarize the request into a 5-8 word title.
2. Slugify: lowercase, non-alphanumeric -> hyphen, collapse runs, max 60 chars.
3. Dendron fname: `plan.<slug>.YYYY.MM.DD` (today, Chicago time).
4. Create the note immediately:

    ```bash
    dendron-cli note write --fname "plan.<slug>.YYYY.MM.DD"
    ```

5. Announce the fname.

## Phase 1: Three Parallel Scouts

Launch **three** Agent calls (subagent_type: Explore, thoroughness: "medium") in a single message so they run concurrently. Each gets the same `<request>` with a distinct angle. No word caps.

### Scout A - Codebase reconnaissance

> You are Scout A. Gather codebase context for: `<request>`.
>
> Find: (1) files/modules in-scope; (2) integration points -- imports, callers; (3) patterns already used for this kind of change; (4) tests covering the area (`tests/torchcell/...`); (5) **pinned versions** of any third-party libs in the in-scope files (`env/requirements.txt`, `pyproject.toml`).
>
> For every MODIFY/DELETE candidate, read its paired dendron note (`notes/torchcell.<module>.md` for `torchcell/`; `notes/experiments.<id>.scripts.<name>.md` for experiment scripts; `notes/scripts.<name>.md` for `scripts/`) and record its stance (`stable`/`provisional`/`in-flux`/`undocumented`).
>
> Return: bulleted file paths with one-line purpose + stance for MODIFY/DELETE candidates; integration-point bullets; pinned-version block. Do NOT paste file contents.

### Scout B - Design history + conventions

> You are Scout B. Gather design context for: `<request>`.
>
> Read: (1) paired dendron notes for every in-scope file; (2) prior plan notes (`notes/plan.*.md`) touching the same subsystem; (3) CLAUDE.md + CLAUDE.local.md conventions that apply; (4) auto-memory feedback at `~/.claude/projects/-home-michaelvolk-Documents-projects-torchcell/memory/` relevant to this area.
>
> Reading the paired note for each in-scope file is required. Assign each `stable`/`provisional`/`in-flux`/`undocumented` from the note's dated entries and framing.
>
> Return: (1) per-file classification (`<path> -- <classification> -- <one-sentence justification citing the note + date>`); (2) prior design decisions that constrain this work, with dendron links + dates; (3) conventions the plan MUST follow (e.g. no fallback mechanisms, image-output + timestamp pattern, dendron frontmatter never edited); (4) user feedback rules with a pointer to the memory file.

### Scout C - Gotchas + failure modes

> You are Scout C. Find hazards for: `<request>`.
>
> Look for: (1) open GitHub issues touching affected files, best-effort (`gh issue list --state open --json number,title,labels,body --limit 50` -- skip if none/none configured); (2) infra quirks (pre-commit hooks, conda env, worktree setup, DATA_ROOT/`/scratch` paths, Slurm, Neo4j/biocypher, DDP/Lightning); (3) related in-flight work on other branches (`git branch -vv`, recent commits); (4) data-layout assumptions (DATA_ROOT, symlinks, lmdb); (5) **library-version hazards** -- web-search recent release notes for deprecations/breaking changes since Jan 2026; compare to the pinned version.
>
> Return: numbered gotchas, each with (a) what, (b) where it bites, (c) how to sidestep. Cite URLs+dates for library hazards.

## Phase 2: Deliberation (one agent)

Launch **one** Agent call (subagent_type: general-purpose) with all three scout reports.

> You are reading three scout reports for: `<request>`. Argue the approach: where scouts converge, where they conflict, the right decision at each conflict.
>
> Scout A: `<report A>`  Scout B: `<report B>`  Scout C: `<report C>`
>
> Think critically. If a scout recommended something conflicting with a documented rule or gotcha, say so.
>
> Produce: 1. **Agreements**; 2. **Disagreements + resolutions** (both sides one sentence each, then resolve with rationale); 3. **Uncovered ground** (flag `AMBIGUOUS -- ask user` if unresolvable); 4. **Recommended approach** (3-6 sentences, broad strokes -- what's new/modified/deleted, no code).

## Phase 3: Plan Draft

Launch **one** Agent call (subagent_type: general-purpose) with the request, all scout reports, and the deliberation. It edits the Phase 0 note.

> You are the plan-writer. Edit `notes/<fname>.md`. Aim for ~300 lines but prioritize density. A reducer-critic tightens afterward.
>
> Inputs: Request `<request>`; Scout A/B/C `<A>/<B>/<C>`; Deliberation `<D>`.
>
> Required sections (H2, in order, no H1):
>
> 1. `## Context` -- why this work, what it solves/replaces. Reference GitHub issues as backtick-wrapped `` `#N` `` if any.
> 2. `## Relevant Files` -- table: path, action (NEW/MODIFY/DELETE/REFERENCE), one-line purpose, stance (`stable`/`provisional`/`in-flux`/`undocumented`/`n/a` for NEW).
> 3. `## Key Design Decisions` -- numbered; decision first, then why; rejected alternatives when illuminating.
> 4. `## Approach` -- narrative, not a file-by-file spec. Libraries, patterns, execution order, out-of-scope. One 3-10 line snippet only when it disambiguates a subtle pattern; more is a failure.
> 5. `## Gotchas` -- numbered from Scout C + deliberation. Each: hazard + sidestep.
> 6. `## Verification` -- tests, commands, manual smoke checks (`~/miniconda3/envs/torchcell/bin/python -m pytest tests/torchcell/... -xvs`, mypy, ruff).
> 7. `## Open Questions` -- only if the deliberator flagged AMBIGUOUS. Omit if none.
>
> Do NOT include: per-file `**Purpose:**`/`**Depends on:**` sub-blocks; huge class bodies; multi-page code listings.
>
> Style: why before what; humans read this; reference specific paths/line numbers; backtick-wrap `` `#N` ``; no Unicode emojis (breaks xelatex export).
>
> **Before reporting done, self-verify:** every path exists or is tagged NEW; every decision has a rationale; approach matches the deliberator's resolutions; no section duplicates another; gotchas are specific; every library claim has a web-sourced citation matched to the pinned version. Fix problems before handing off.

## Phase 4: Reducer-Critic Loop

Launch **one** Agent call (subagent_type: general-purpose) with the current plan. Max 3 iterations. The critic does NOT truncate -- it identifies low-value content and applies cuts.

> You are the reducer-critic for `notes/<fname>.md`. Current line count: `<N>`. Target: ~300 lines; density matters more than raw count.
>
> Current plan: `<full plan text>`
>
> Identify **low-value content**: redundancy; convention restatement (already in CLAUDE.md); obvious context; template filler; code-as-prose candidates; implementation-level detail that belongs in code comments.
>
> Preserve **high-value content**: specific paths/line numbers; unique decisions with rationale; non-obvious gotchas with sidesteps; tradeoffs; constraints from prior decisions (with dendron links); cross-references.
>
> Produce: A. **Proposed cuts** (section + quote + one-sentence justification); B. **Apply cuts** (edit the file; keep the clearer of duplicates; convert dense code to prose; verify cross-references still resolve). Run `wc -l notes/<fname>.md` and report the new count.
>
> **Termination signal.** If no high-quality cuts remain, state exactly: `"No further cuts available without cost."`

Loop logic: termination phrase -> exit to Phase 5. Under ~310 with cuts applied -> exit. Over ~310 with cuts found -> another pass (max 3). Over after 3 -> exit, do NOT truncate; tell the user it came in long.

## Phase 5: Weekly Task Note + Present

1. Append one pending bullet to the current weekly note under today's `## YYYY.MM.DD`: `- [ ] <one-sentence plan summary> [[plan.<slug>.YYYY.MM.DD]]`. Create the date H2 if missing.
2. Stage the plan note + weekly note.
3. Try `code notes/<fname>.md` -- swallow IPC errors silently.
4. Print the summary block as your **last output**:

    ```text
    ## Summary

    <2-4 sentences: problem, approach, key files, primary risk>

    ## Files

    Plan note:    notes/<fname>.md
    Dendron link: [[plan.<slug>.YYYY.MM.DD]]
    Line count:   <wc -l result>

    Read the plan in your editor. Ask questions or request revisions here.
    When ready to implement: /wt-implement notes/<fname>.md
    (Use high or xhigh effort for implementation.)
    ```

Nothing after this block.

## Interactive Revision

After presenting, enter a revision loop: answer questions from context; make specific edits; revise (don't restart) a rejected decision; run another reducer-critic pass on request. Exit when the user approves or pivots.

## Rules (Opus 4.8 takes these literally)

- **3 scouts, 1 deliberator, 1 plan-writer, 1 reducer-critic (looped).** Do not add agents.
- **Scouts run in parallel** in a single message.
- **No artificial word caps on scouts.** The reducer-critic trims.
- **Reducer removes low-value categories, not line count.** After 3 passes, if still over, stop.
- **Self-verify before reporting** (plan-writer and reducer-critic).
- **No EnterPlanMode / ExitPlanMode.** This skill replaces plan mode.
- **Leave judgment room.**
- **Verify library currency from the web, not memory.**

## Example

```text
/plan-4.8 Add a COO classification head for fitness regression in torchcell/transforms and wire it through the dcell model and the 010-kuzmin-tmi inference scripts.
```
