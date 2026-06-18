---
name: wt-write
description: Draft and revise long-form prose (manuscripts, proposals, strategy docs, design memos) as Dendron notes inside an isolated git worktree, iterate with the user, then hand off to /merge-worktree. The writing counterpart to /plan-4.8 -- worktree-first, but for words instead of an implementation plan. Use when the deliverable is a document, not code.
---

# WT-Write

A worktree-first writing skill for substantial prose: a paper outline, a grant
or publishing plan, a design memo, a literature synthesis, an SOP. The note is
born in its own worktree, drafted and revised there with the user in the loop,
then landed via `/merge-worktree`. Code stays out of scope -- this is for words.

## When to use this vs the alternatives

- `/wt-write` -- the deliverable is a **document** (manuscript, proposal,
  strategy, synthesis, SOP). No code change implied.
- `/plan-4.8` -- the deliverable is an **implementation plan** for a code change
  (scouts fan out over the codebase, a reducer-critic tightens to ~300 lines).
- `/update-src-notes` / `/update-notes` -- you are documenting a code change that
  already happened in the paired module note.

If the request is "write a plan to publish X", "draft the methods section",
"synthesize these sources into a strategy note" -- that is `/wt-write`.

## Usage

- `/wt-write <topic-or-request>` -- new document; the skill derives a slug and fname.
- `/wt-write <dendron.fname> <request>` -- write into a specific named note.

Source material (conversations, papers, prior drafts) may be provided **inline in
chat** or pointed to as a file/scratch-note path. If the user references pasted
text you cannot actually see in the conversation, **say so and stop** -- do not
invent content. Ask them to re-paste or save it to a path you can read.

## Why worktree-first (same invariant as /plan-4.8)

A note committed to local `main` is never pushed (landings go worktree -> origin
via `/merge-worktree`), so it sits un-pushed and every later worktree inherits
it -- the tangle that lands one branch's note commit under another's merge.
Drafting in a worktree keeps `main` pristine and lets revision churn happen in
isolation; the finished document lands in one clean rebase + ff-merge.

## Phase 0: Setup (worktree-first)

1. Summarize the request into a 5-8 word title.
2. Slugify: lowercase, non-alphanumeric -> hyphen, collapse runs, max 60 chars.
3. Create the branch worktree off `main` and set it up:

    ```bash
    MAIN="$HOME/Documents/projects/torchcell"
    WT="$HOME/Documents/projects/torchcell.worktrees/write/<slug>"
    git -C "$MAIN" worktree add "$WT" -b write/<slug> main
    cd "$WT"
    bash scripts/setup-worktree.sh
    ```

   If the branch/worktree already exists (resuming), `cd "$WT"` into it instead.
4. Announce the branch and worktree path so the user knows where output lands.

**Every subsequent phase runs with the worktree as cwd.** Nothing touches local `main`.

## Phase 1: Create the note(s)

Decide the fname(s). Convention:

- General documents: `<descriptive.dotted.fname>` chosen for where it belongs in
  the Dendron tree (e.g. `torchcell.publishing.nature-biotech-plan`,
  `experiments.014-ecoli.manuscript-outline`).
- If the user gave an explicit fname, use it.

Create each NEW note with `dendron-cli` (NEVER the Write tool -- it skips the
frontmatter the rest of the toolchain depends on):

```bash
dendron-cli note write --fname "<dotted.fname>"
```

For an **existing** note, do not recreate it -- you will append with the Edit
tool in Phase 3 (preserve its frontmatter; never edit the `---` block).

A long document may span several linked notes (outline + per-section children).
Create them all here and cross-link with `[[dotted.fname]]`.

## Phase 2: Gather context (light, only what the document needs)

This is not a codebase scout fan-out. Read only what grounds the writing:

1. Source material the user supplied (inline or at a path).
2. Related Dendron notes the document should cite or build on -- find with
   `grep -rl "<keyword>" notes/` and read the relevant ones; cross-link them.
3. **Currency check for any external claim.** Training cutoff is January 2026;
   today may be later. Any factual claim about a paper, dataset, tool version,
   funding call, or external result that the document asserts as current must be
   web-confirmed (WebSearch / WebFetch), not stated from memory. Cite briefly
   inline: `(confirmed 2026-06-09 via <source>)`. If a claim cannot be verified,
   mark it `[UNVERIFIED]` in the draft rather than asserting it.

For a document that needs broad external research (many sources, fact-checking),
prefer running `/deep-research` first and writing its cited output into the note,
rather than improvising sources here.

## Phase 3: Draft

Write the document into the note(s) with the Edit tool. Follow Dendron
conventions exactly:

- **Date-stamped H2 sections.** Each work session is a `## YYYY.MM.DD - Title`
  block (`TZ=America/Chicago date +%Y.%m.%d`). Newest at the bottom; never
  overwrite a prior dated section -- a significant revision becomes a *new* dated
  H2 so history is preserved.
- **Intentional stance.** Lead with the problem / motivation / thesis, then the
  content. A document that only states conclusions is weaker than one that shows
  why they follow.
- **No Unicode emojis** anywhere -- they break the xelatex PDF export.
- **Markdownlint-clean:** 2-space indent for nested lists, one blank line around
  H2s, no trailing-punctuation worries (the workspace disables those rules, but
  keep it tidy so save-format is a no-op).
- Cross-link related notes with `[[dotted.fname]]` (no aliases needed unless the
  user wants display text; `dendron.yml` sets `aliasMode: none`).

Aim for the document the user asked for -- there is no artificial length target
here (unlike `/plan-4.8`). Density and clarity over word count.

## Phase 4: Revision loop (the core of this skill)

Present the draft and enter an interactive revision loop. Writing is iterative:

- Answer questions about the draft from your context.
- Make specific edits the user requests -- revise in place within the current
  dated H2 while still drafting; once a section is "landed" and the user later
  asks for a substantive rework, open a new dated H2 instead of mutating the old.
- Restructure (reorder sections, split into child notes, merge) on request.
- Re-run a currency check on any claim the user flags.

Stay in this loop until the user signals the document is ready or pivots. Do not
rush to merge -- the worktree exists precisely so revision can churn freely.

## Phase 5: Weekly note + stage + hand off

When the user approves:

1. Append one pending bullet under today's `## YYYY.MM.DD` H2 in the **worktree's
   weekly child note** `user.Mjvolk3.torchcell.tasks.weekly.<YYYY>.<WW>.<slug>.md`
   (`WW` = ISO week, `TZ=America/Chicago date +%G.%V`; create with `dendron-cli
   note write` if missing): `- [ ] <one-sentence summary> [[<dotted.fname>]]`.
   This child-note convention keeps worktree weekly edits from colliding on
   rebase (see `/update-tasks-weekly`).
2. Stage the document note(s) + weekly child note on the branch (cwd is the
   worktree, so plain `git add` targets it). Keep any source material that should
   stay private as untracked `notes/scratch.*` -- never `git add` it.
3. Try `code notes/<dotted.fname>.md` -- swallow IPC errors silently.
4. Print the summary block as your **last output**:

    ```text
    ## Summary

    <2-4 sentences: what the document is, its thesis, key sections, what is still open>

    ## Files

    Document:     notes/<dotted.fname>.md   (in the worktree, on the branch)
    Dendron link: [[<dotted.fname>]]
    Branch:       write/<slug>   (worktree at ~/Documents/projects/torchcell.worktrees/write/<slug>)

    Read it in your editor. Request more revisions here.
    When ready to land: /merge-worktree write/<slug>
    ```

Nothing after this block.

## Rules

- **Worktree-first.** The note is born in its own `write/<slug>` worktree, never
  on local `main`. Every phase runs with the worktree as cwd.
- **NEW notes via `dendron-cli`, never the Write tool.** Existing notes: append
  with Edit; never touch the frontmatter `---` block.
- **Date-stamped H2 sections; never overwrite prior dated sections.** Revisions
  are new dated blocks.
- **No fabricated sources.** External claims are web-confirmed or marked
  `[UNVERIFIED]`. If the user references material you cannot see, stop and ask.
- **No Unicode emojis** (xelatex export).
- **This is a writing skill, not a coding skill.** It does not edit `torchcell/`
  source or run code. If the work turns out to need code, hand off to
  `/plan-4.8`.
- **Land via `/merge-worktree`, never `gh pr merge`.** The user's tool-approval
  prompts are the gates -- do not ask extra confirmation questions.

## Example

```text
/wt-write torchcell.publishing.nature-biotech-plan Draft a staged plan to take
the TorchCell CGT work to a Nature Biotech submission: current state, the gap to
a publishable result, the experiments/analyses needed, and a timeline. Source
material pasted below.
```

Phase 0 cuts `write/nature-biotech-plan`. Phase 1 creates
`notes/torchcell.publishing.nature-biotech-plan.md` via `dendron-cli`. Phase 2
reads the pasted sources and the relevant existing TorchCell notes, web-confirms
any cited benchmark or competing-paper claim. Phase 3 drafts the staged plan.
Phase 4 iterates with the user. Phase 5 stages on the branch and points to
`/merge-worktree write/nature-biotech-plan`.
