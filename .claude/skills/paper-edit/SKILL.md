---
name: paper-edit
description: Edit the Nature Biotech manuscript (paper/nature-biotech) respecting the section status stoplight, then rebuild editing.pdf. Use for any change to the paper body, sections, or SI.
---

Workflow for editing the Nature Biotechnology manuscript in `paper/nature-biotech/`. The
edit-then-rebuild loop so the author can look at the PDF directly. Follow this exactly.

## Step 0: Edit the shared body, never the wrappers

Edit `content.tex` and `sections/*.tex` (the shared body). Do NOT put content in
`editing.tex`, `submission.tex`, or `twocolumn.tex` — those are thin per-view wrappers.
Figures come from assets → draw.io → `figures/` (see CLAUDE.md); never write image data.

## Step 1: Check the section status BEFORE editing (stoplight policy)

Every heading has an editing-only chip `\secstatus{todo|tent|final}`, and
`sections/outline.tex` is the status board. Notation:
`todo` = red ✗ (not done), `tent` = amber ■ (author-reviewed, keep stable), `final` = green ✓
(publication-ready). Symbols are editing-only (pifont `\ding`), suppressed in submission.

For each section you intend to change, read its `\secstatus`:

- **`todo`** → fair game. Edit freely.
- **`tent` or `final`** → the author has read and approved it. **Do NOT edit yet.** First
  state the exact change you propose and ask for an explicit go-ahead. Proceed only if the
  author confirms (they may override per-request). This is a hard rule from CLAUDE.md.

Do **not** self-promote a section's status. When you finish a `todo` section, leave it
`todo`; the author promotes `todo→tent→final` on review. If the author explicitly tells you
to change a status, update BOTH the inline `\secstatus{...}` and the matching line in
`sections/outline.tex` so the board stays in sync.

## Step 2: Make the edit

- Proofs / propositions / Supplementary Notes: follow `notes/paper.proof-writing-standard.md`
  ([[paper.proof-writing-standard]]) and the house format in `preamble.tex` (proposition/
  proof envs, `\pfstep`, `\notesec`; no bullets in proofs; no Theorems).
- Match the finalized notation (see [[cgt-paper-fig1-methods-state]] memory): $G=(N,E)$,
  $N=6607$ nodes, perturbation $p$-family, $D_{\mathrm{KL}}$, etc.
- Keep `\cb{...}` char-budget and `\wc{...}` word-count tags intact on headings.

## Step 3: Rebuild so the author can view the PDF

```bash
make -C paper/nature-biotech editing
```

This regenerates `editing.pdf` (the drafting view with the stoplight chips + outline board).
Then verify the build is clean:

```bash
grep -iE 'undefined control sequence|! LaTeX Error|! Undefined' paper/nature-biotech/editing.log | grep -iv warning | head
```

Empty output = clean. The `Object @figure.N already defined` / `PDF version 1.7` warnings
are pre-existing and harmless. If you changed a draw.io figure, `make -C paper/nature-biotech fig`
first (the build has a hard figure-size gate).

To also refresh the submission / two-column views: `make -C paper/nature-biotech paper`.

## Step 4: Report

State which sections changed, their status (and whether any `tent`/`final` edit was
author-approved), and that `editing.pdf` rebuilt clean. Do not commit unless asked; if
asked, follow `/stage` + `/commit` (scratch notes are never committed).
