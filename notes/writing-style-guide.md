---
id: 7ok4822jeg8c45dil486eww
title: Writing Style Guide
desc: ''
updated: 1787355864950
created: 1787355864950
---

Prose standards for anything written to be read. Record every new preference here,
under its topical heading, so the rule outlives the conversation that produced it.

## Scope -- two tiers, on purpose

Applying every rule everywhere would be wrong. A working note IS for the group, so
"we decided X" is correct in one and a violation in the other.

**Mechanical tier -- everywhere.** Dendron notes in `notes/*.md`, commit messages,
code comments, docstrings, the manuscript, `notes-tex/`. No em-dashes, American
spelling, verbatim quotes unaltered. No audience-dependent exception exists for
these, and `make check` fails a `notes-tex/` build on them.

**Audience tier -- anything a colleague reads end to end.** The manuscript,
`notes-tex/` documents, figure captions, table notes, and any Dendron note written
for someone else. Adds: no trailing restatement, no narrating the document, no
self-reference about the group, define-before-use. These relax for scratch
thinking, where getting the idea down beats phrasing it well.

The mechanical tier is checked. The audience tier is a read -- see the note on why
the trailing-restatement detector was removed.

## 2026.08.21 - Established from the first review round of microbe-perturb-seq

84 reviewer comments on a 30-page typeset note. Most were style, and most of
those were the same handful of habits. The rules below are written as
prohibitions because that is how they were flagged.

### Trailing restatement - the main one to avoid

**A clause appended to a sentence that re-says what the sentence already said, in
different words.** It reads as emphasis and adds nothing.

> Every molecule tagged inside that droplet shares a barcode **because the droplet
> wall keeps it that way.**

The droplet already did the work; the tail restates it. Write the mechanism once:

> A microfluidic device co-encapsulates one cell with one barcoded bead in an
> emulsion droplet, which isolates every molecule that cell releases.

The tell is a trailing `because ...`, `which is what ...`, `so it ...`, or
`and that is ...` whose content is already entailed by the main clause. It is
NOT the same as a trailing clause that adds a consequence or a number -- those
earn their place. Ask: does the tail introduce a fact the sentence did not
already contain? If not, cut it.

### Do not narrate the document

The reader knows they are reading a document. State what was done or found.

- "This document asks what such a screen costs" -> "What follows establishes what
  such a screen costs"
- "This document works out ..." -> "... is established in Sec. N"
- Do not explain who the intended readers are, or that it is not a manuscript.

### Do not be self-referential about the group

Write the property, not who wants it.

- "For a group whose goal is a predictive model of cell state, this is the
  highest-density source of training data" -> "It is the highest-density source
  of genotype-to-expression training data available."
- "Two families are live options **for us**" -> "Two method families are
  candidates."
- "Split-pool is the one **we currently prefer**" -> give the reason directly.

### No colloquialisms or flourish

Colleagues have to want to read it. Concision is what buys that.

- "Droplet is the incumbent" -> "Droplet is the established option"
- "would measure its own sample preparation" -> "would measure artifacts of its
  own sample preparation"
- "Fixation first is what makes the measurement about the perturbation" ->
  "Fixing first is what isolates the perturbation's signal from the handling"

### Spelling and typography

- **American spelling, by default and without exception.** organize, optimize,
  permeabilize, factorize, color, favor, neighbor, labor, catalog, analyze,
  installment, centered, gray, program, judgment, alphabetize. This holds even
  where a target journal is British and even where a neighboring source quote
  uses the other form: the quote keeps its spelling, the prose around it does
  not. It is not a per-document preference and should never be relitigated per
  document.
  - **It is checked as a class, not as a word list.**
    `notes-tex/common/check_doc.py` matches `-ise`/`-yse` by stem plus endings,
    so a conjugation cannot slip through the way `permeabilises` did while
    `permeabilised` was caught, plus explicit words for `-our`, `-re`,
    doubled-l and the one-offs. A new British form that gets past it is a bug in
    that list, so fix the list rather than only the sentence.
- **Never em-dashes** (`---`). Use a spaced en dash ` -- ` or a comma. Keep `--`
  for numeric ranges and markdown table rules.
- **In draw.io, `--` is literal.** LaTeX turns `--` into an en dash; a canvas does
  not. Use the real character (`&#8211;`) in `.drawio` sources.
- **Nature sets `in vivo`, `in vitro`, `in situ` ROMAN**, not italic. Organism
  names stay italic (`\org{}`).
- **Superscripts in figures** use real characters, not carets: `4^10` must be
  drawn as 4 with superscript 10, matching neighboring exponents.

### Verbatim quotes are verbatim

A quote from a paper keeps its own spelling, including British forms. Style
sweeps must exclude `quote=` fields and ``...'' spans. A find-and-replace that
Americanizes a source quote has falsified it.

### Define before use

Any term used in a technical sense goes in the controlled vocabulary BEFORE it
appears in prose or in a figure. A word that shows up first in a figure label is
the specific failure a vocabulary table exists to prevent.

Order a vocabulary table ALPHABETICALLY, not by theme. Thematic grouping needs
subheading rows, a subheading leaves the definition column blank, and the first
entry under each heading then reads as though it has slipped a line. Teaching the
scheme is what the surrounding prose is for; the table is for lookup.

A lookup entry still has to explain. An entry that names a thing without saying
why it is that way sends the reader to the section anyway, which is the one job
the table was meant to save them. Say what the term is, then the one property
that makes it behave surprisingly. And an entry that asserts a NUMBER carries its
citation in the table, not only in the generating script: a reader checking "the
field standard is 100 cells" has nowhere else to go.

### One symbol, one meaning, and rename rather than warn

Where two sources use the same letter for different quantities, rename one and
record the displaced source's notation at the point of definition. Carrying both
means warning about the collision at every use, and a warning that has to be
repeated is a naming decision that was not made. The same holds for words: if two
things in a document are called an "index", pick one and call the other something
else. Standardizing costs one sentence of attribution; not standardizing costs a
sentence every time either term appears.

### Answer the question that was asked

If a reviewer asks "why 48 of 96 wells?", the answer is either the reason or
"the source does not say, and here is the consequence". Do not restate the fact
they were querying. Where a source genuinely does not justify a choice, say so
and mark it as a candidate for measurement.

## Related

- [[paper.nature-biotech.figures]] -- figure sizing, palette, export
- Provenance flags and the section stoplight: `notes-tex/common/tcdoc.sty`
- Per-figure number provenance: `notes-tex/common/figure_provenance.py`
