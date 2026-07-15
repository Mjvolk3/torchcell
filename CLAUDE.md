## Vision -- The Virtual Cell (North Star)

The field's consensus goal (Qian et al. 2026, *Nature* Perspective "Towards the
construction of a virtual yeast" -- authored by much of the yeast-systems establishment:
Costanzo, Boone, Andrews, Ralser, Steinmetz, Zampieri, Cai, ...) is the **AI-driven
virtual cell**, with *S. cerevisiae* first. We share that goal. Note the paper ALSO calls
itself biology-first, so our differentiation is **layer, rigor, and operationalization**,
not philosophy: torchcell builds the **provenance-first, L0-L4-VERIFIED experimental-data
ontology / substrate** their predictive+generative modeling stack must stand on -- NOT the
AI agent itself. Distinctives to hold to (and clarify in the manuscript):

- **Provenance-first + rebuildable + verified** (absent from their roadmap): every value
  traces to a sha256-pinned artifact + verbatim source quote, with L0-L4 verification.
- **Ontologize the EXPERIMENT** (typed `genotype × environment -> phenotype` record with
  perturbations + provenance), not a top-down functional decomposition of the cell; let
  structure grow naturally from what real papers report, captured to the bit/byte, being
  MORE rigorous than the source (beyond published standards when necessary).
- **Sequence-level genotype fidelity**: perturbation as an edit to the total genomic
  content in the cell (deletions, integrations, alleles, present plasmids, heterologous
  cassettes) -- the enabler of inverse strain design.
- **Generalization axis** = single-cell bioproduction hosts (yeast first + manuscript
  target), not yeast -> other eukaryotes.

Sequence/annotation standards stance: own the pydantic object as a SUPERSET; ingest
GenBank first, then GFF3+FASTA; emit GenBank/GFF3/SBOL for interoperability;
Sequence-Ontology-aligned; grow toward SBOL-level design capture as plasmid/strain design
becomes first-class. Full analysis + differentiation for the paper:
`[[paper.north-star]]` (background: memory `virtual-cell-consensus-goal`).

## For Local Configs

`/Users/michaelvolk/Documents/projects/torchcell/CLAUDE.local.md`

- Includes how to run local python

## Git Worktrees

We develop on branches via git worktrees at
`~/Documents/projects/torchcell.worktrees/<branch>/`. **Everything goes through a
worktree by default -- code AND notes (roadmap/plan/weekly/dendron/`.claude/*`).**

- **The primary checkout stays on `main` and is NEVER used to `git checkout` another
  branch.** Flipping its HEAD drifts local `main` behind origin and makes files vanish
  from the tree (and from the editor pointed at it). The user opens worktrees in
  separate VS Code windows.
- **Setup is mandatory.** Run `/setup-worktree <branch>` (wraps
  `scripts/setup-worktree.sh`) before any work, then `cd` into the worktree and make
  ALL edits there. Scratch stays untracked; do not leave the primary tree dirty.
- **Why notes too:** a commit made directly on the primary checkout's `main` (even a
  notes edit) advances local `main` independently, so a worktree branch that was
  rebased onto the *old* `main` can no longer fast-forward -- the normal
  `rebase wt onto main -> ff-merge` flow breaks. Keeping the primary tree clean on
  `main` is what keeps ff-landings unblocked.
- **Land via `git rebase` + ff-only, NEVER `gh pr merge` / the GitHub "Merge pull
  request" button.** A merge commit diverges `main` from the rebased branch and blocks
  the next worktree's ff-land; it also buries the individual commits. Use
  `/merge-worktree` (rebase the branch onto `main`, then `git merge --ff-only` + push)
  or `/enqueue-merge` (the single-writer merge queue). If a PR is open, close it with a
  comment that the branch landed via rebase+ff. Result: linear history, every commit
  lands verbatim.
- **Shared `.git` hazard.** All worktrees share one object store + stash list, and
  pre-commit stashes/rolls back -- so parallel landings corrupt each other. Land one at
  a time (the merge queue serializes this). `MEMORY.md` is shared via symlink; keep
  writes additive.
- **On a blocked land, fix the state in the worktree and re-land.** NEVER `--no-verify`,
  force-push, or blind-discard to "unblock."

## Programming Guide

- Do NOT ever use fallback mechanisms unless we clearly tell you to. This means minimize try except blocks, unnecessary conditionals, etc.
- **STRICT RULE — artifacts come from scripts in the experiment folder.** Any artifact used
  in the paper or in `notes/` (figures, tables, derived CSVs, numbers) MUST be produced by a
  committed script that lives in the `experiments/<id>/` folder where that data is generated
  (e.g. `experiments/smf-dmf-tmf-001/traditional_ml-plot_paper.py`), reading the real result
  files. Never hand-author a figure/table/number into the paper or a note without a
  regenerating script. When you create such an artifact, also point to its generating script.
  (This is the paper-artifact half of the provenance principle below; the dataset half is
  "Provenance & Reproducibility".)
- **Pydantic-first.** torchcell models structured data with pydantic: schemas,
  configs, provenance/manifest records, and any typed record. Prefer pydantic
  `BaseModel` (validation + serialization + typed access) over dataclasses/dicts
  for structured data. New structured objects should be pydantic unless there is a
  clear reason not to.

## Provenance & Reproducibility

Core principle: for ANY artifact (paper PDF, supplementary file, OCR markdown,
extracted data, AND dataset raw files), we must be able to answer coherently
"**where did this come from and how was it done?**" from OUR own documentation.
The documentation is authoritative -- the buck stops there; we never re-chase
authors or trust a live URL. Model these records with pydantic.

- **The stored artifact + its `sha256` is canonical, NOT the URL.** URLs are
  historical *retrieval metadata*, not live dependencies. On rebuild we run the
  recorded retrieval command, then verify sha256. If upstream content changed
  (sha256 mismatch) or the URL rotted (command fails), we DETECT it and fall back
  to our mirror / Zotero -- we never silently follow drift. Upstream changes create
  a NEW versioned provenance record; they never overwrite the version we used.
  (A future "watcher" service re-fetches periodically to flag upstream changes.)
- **Every retrieved file records:** `source_url`, `retrieval_method` (enum, e.g.
  `springer_esm | zotero_attachment | pmc_oa_api | direct_url | manual_browser`),
  `retrieval_command` (the exact code, or a manual recipe for un-scriptable
  sources), `sha256`, `retrieved_at`, plus processing provenance (e.g.
  `mineru_version` + args + DPI) so OCR/extraction is deterministic.
- **Mirrors (each backed up):** Zotero = canonical **PDFs only** (we read from PDF;
  reproducibility backstop; large non-PDF supplements/software/data do NOT go in
  Zotero). `$DATA_ROOT/torchcell-library/<citation_key>/` = local PDF mirror + OCR
  artifacts + `manifest.json` + non-PDF supplements under `software/`, `si/`, etc.
  Dataset RAW files get the same treatment: a raw-data mirror with a per-file
  provenance record (source_url + retrieval_command + sha256) that dataset loaders
  reference, so every built LMDB traces to an exact, hash-pinned raw version.
- **Rebuild guarantee:** retrieval code + OCR/extraction recipes + backed-up mirrors
  (Zotero PDFs, raw-data mirror) + manual recipes for un-scriptable files =
  full rebuild-from-scratch, with sha256 verifying every rebuilt file matches.
- Retrieval-method reality (verified): Springer ESM (`static-content.springer.com`)
  and the PMC OA API are scriptable; PMC file downloads (JS proof-of-work) and
  `nature.com` (auth redirect) are not -- those become manual-once -> deposit ->
  reproducible via the mirror.

## Adding Datasets (Modular) -- Sourcing Values from the SI

When adding a dataset in the modular fashion (a new `torchcell/datasets/.../*.py`
loader), any statistical/metadata value the schema needs -- `n_samples`,
replicate/colony/screen counts, the uncertainty TYPE (sample SD vs bootstrap SE),
units, thresholds -- MUST be sourced from the paper/SI, recorded with a verbatim
quote + `sha256` + section/line, and never guessed.

- **Do not conclude "not in the SI" on a first pass.** If a value isn't obvious,
  RETRY and comb through thoroughly: read the full Methods/SI (not a summary),
  check the data-file COLUMN DESCRIPTIONS, and search for every synonym
  (`replicate`, `colony`, `screen`, `bootstrap`, `standard deviation`, `n =`,
  `measurements`). A wrong "not found" leads to a guessed value that silently
  corrupts training weights.
- **Follow deferrals.** SIs routinely defer method detail to a cited earlier paper
  (e.g. Kuzmin -> Baryshnikova 2010; Costanzo -> Baryshnikova 2010). The value is
  then sourced from THAT mirrored paper; the provenance chain (this SI -> ref N ->
  formula) is the answer, and is itself the citation.
- **Identify the exact column the loader consumes.** Different columns can have
  different replicate structures (e.g. Kuzmin query fitness = 12-24 colonies,
  bootstrap; combined-mutant SD = 4-8 colonies, sample SD). Map the statistic of
  the column you actually store, not a neighbouring one.
- **When a value genuinely varies per-record and isn't a released column, say so**
  in code + PR, pick a documented representative, and flag it for review -- do not
  present a guess as sourced. Record the finding in the dataset's dendron note.
- **Resolving a range with no per-record value (never assume the optimistic end).**
  When the SI gives only a range (e.g. "4-8 colonies") and no per-record column,
  resolve it in this order: (1) **back-solve** from another provided statistic --
  e.g. reconstruct the reported p-value/SE from the stored SD across all records
  and find the `n` whose central tendency matches (this is empirical, not an
  assumption); (2) if back-solve is precluded (unpublished term, no companion
  statistic), take the **conservative lower-end** of the replicate count (larger
  SE -- never overstate precision); (3) the **median** of the range only if a
  central estimate is explicitly wanted over conservatism. Document which rule was
  used and the evidence. Worked example: Kuzmin combined-mutant `n_samples`
  (back-solve + conservative both -> 4) in
  `[[torchcell.datasets.scerevisiae.costanzo2016.noise-computation]]`.

## Paper / Manuscript Workflow

The Nature Biotechnology manuscript lives in `paper/nature-biotech/`. It uses the
Springer Nature `sn-jnl` class (Nature Portfolio `sn-nature` style) with a shared
body (`content.tex`) compiled by thin wrappers, and builds via Tectonic.

- **Two tiers (workshop vs shared).** `paper/nature-biotech/` is the **workshop**:
  private, full, versioned in torchcell (`editing.tex`, `figure-proto.tex`,
  char-budget tags, draft scaffolding). `~/Documents/projects/torchcell-overleaf` is
  the **shared** copy: an Overleaf-backed git repo holding a curated subset that
  collaborators see. Edit in the workshop only.
- **Publish to collaborators:** `bash paper/nature-biotech/sync-overleaf.sh` copies
  the curated `SHARE_FILES` (submission.tex -> `main.tex`, plus content/preamble/
  cls/bst/bib, figures, and the figure guide), pulls collaborator changes first,
  then pushes to Overleaf. Edit `SHARE_FILES` to control what crosses over.
- **Build PDFs:** `make -C paper/nature-biotech paper` builds three views with
  Tectonic -- `submission.pdf` (journal submission; official single column),
  `editing.pdf` (our drafting/typeset look; print-approx margins + char budgets),
  `twocolumn.pdf` (published-like double column). `make figproto` builds the
  true-scale figure-sizing canvas. Install Tectonic via `conda install -c
  conda-forge tectonic`.
- **Figures come from assets; never write image data directly to Overleaf.**
  Generate plots to `ASSET_IMAGES_DIR` (the standard image-output convention),
  compose figures in draw.io at Nature print size (full 180 mm / column 88 mm,
  <=170 mm tall -- see `paper/nature-biotech/figures/README.md`), export vector PDF
  into `paper/nature-biotech/figures/`, then sync. Collaborators may add their own
  images in Overleaf; the sync pulls first and never deletes their files. The
  figure-prep guide and the figure-sizing canvas are synced so collaborators see them.
- **Proofs and formal claims: follow the proof-writing standard.** Before authoring or
  editing any proposition, proof, or Supplementary Note, read
  `notes/paper.proof-writing-standard.md` ([[paper.proof-writing-standard]]) and follow it.
  In short: skeleton is Setup -> Claim -> Proof -> Consequence -> Interpretation; use the
  LaTeX house format in `paper/nature-biotech/preamble.tex` (`proposition`/`lemma` +
  `proof` environments, `\pfstep{...}` bold run-in step headers -- NEVER `itemize`/bullets
  inside a proof, minimal `\emph`); prefer **Proposition** for main claims and
  **Corollary/Remark** for consequences; do NOT use **Theorem** (the paper is empirical).
  SI house order: Supplementary Notes (text + proofs) first, then all figures, then refs.
- **SI structure invariant:** in `sections/backmatter.tex`, `\section*{Supplementary
  Information}` + a Contents block, then the Supplementary *Notes* (`\subsection*`, each on a
  new page). A figure that supports a specific Note (e.g. an empirical-validation figure)
  sits WITH that Note; general figures that support the main text follow after all the Notes.
  S-numbering (`\setcounter{figure}{0}` + `\renewcommand{\thefigure}{S\arabic{figure}}`) is
  set ONCE at the start of the SI so both interleaved and trailing figures number in order.
  Keep main Figs 1--6 before Methods with `\FloatBarrier` (placeins) after each Results figure
  and before `\section{Methods}`.
- **Section status stoplight + agent-editing policy.** Every heading carries an editing-only
  status chip `\secstatus{todo|tent|final}` (defined in `editing.tex`; no-op in
  `submission.tex`/`twocolumn.tex`), and the manuscript opens with an editing-only outline/
  status board (`sections/outline.tex`, inserted by `\paperoutline`). Notation:
  **`todo` = red ✗** (not done), **`tent` = amber ■** (author-reviewed, keep stable),
  **`final` = green ✓** (publication-ready). The board (`sections/outline.tex`) also holds a
  per-Results-section paragraph plan (P1 high-level first) under each section's figure. **Editing rule for coding agents:** a `todo`
  section is fair game to edit freely; a `tent` or `final` section has been read and
  approved by the author, so **do NOT edit it without first surfacing the intended change
  and getting an explicit go-ahead** (the author may override per-request). When you finish
  a `todo` section, leave it `todo` for the author to promote -- do not self-promote to
  `tent`/`final`. Keep the outline board flags in sync when a section's state changes. Use
  the `paper-edit` skill for the edit-then-rebuild loop.

## Mermaid Diagrams with Math (KaTeX)

Mermaid (v11+ via `mmdc`) renders `$$...$$` labels with KaTeX. Two non-obvious
rules, learned the hard way -- follow them when editing any mermaid diagram with
math:

- **Multi-line labels need TRIPLE backslash `\\\`, not `\\`.** Mermaid collapses
  `\\` -> `\` before passing to KaTeX, so a normal `\begin{gathered}...\\...\end{gathered}`
  renders on ONE line. Write the row separator as `\\\` so KaTeX receives `\\` and
  breaks the line: `$$\begin{gathered}\text{Line 1}\\\ \text{Line 2}\end{gathered}$$`.
  (Known mermaid bug: mermaid-js/mermaid #7194, #5941.) This is the lever for
  controlling box/figure width -- wrap long boxes instead of letting them run wide.
- **Don't mix plain text and `$$...$$` on one line** -- mermaid eats the space at the
  HTML/KaTeX seam (`Cell Graph|G|`). Make the whole label one KaTeX block; use
  `\text{...}` for words. `<br>` only breaks between pure-text segments; it is
  dropped next to math.

**Worked example** (every box wraps via `\\\`, full palette):
`notes/torchcell.models.equivariant_cell_graph_transformer.mermaid.type-i-ii.md`.
**Render a standalone diagram to PDF/SVG/PNG** (md-matched name, outlined SVG for
draw.io): `bash notes/assets/publish/scripts/mermaid_pdf.sh <file.md>`. Full pipeline
notes in [[paper.nature-biotech.figures]]. Use `mmdc` directly, not the pandoc
`mermaid-filter` path (its bundled puppeteer times out).

## Dendron Paths

We use dendron paths in markdown files to link between project notes. These are very useful for giving additional context.

File of these patters:

[[2025.05.10 - Inspecting Data in GoGraph|dendron://torchcell/torchcell.models.dcell#20250510---inspecting-data-in-gograph]]

Can be found here:

notes/torchcell.models.dcell.md

**The General Pattern**

`notes/` dir is skipped in path description as it is default location

Dendron encode from `torchcell/torchcell.models.dcell` to `notes/torchcell.models.dcell.md`

When I tell you to write some output to a file that is in `notes/` then typially you just need to append or modify, we don't want you messing up dendron frontmatter.

### Scratch Notes Are Not Committed

`notes/scratch.*.md` (and any PDFs they render to, `notes/assets/pdf-output/scratch.*`)
are **working / throwaway notes** and must **not be staged or committed**. When scratch
content matures into something worth keeping, **move it into a properly named Dendron note**
(e.g. a `paper.*`, `torchcell.*`, or `experiments.*` note created with `dendron-cli`), fix
its cross-references and any rendered-PDF names to match, then delete the scratch original.
Do not `git add` a `scratch.*` note; if one is already tracked, flag it rather than
committing it.

### Creating New Dendron Notes

**For NEW notes in `notes/` directory, use `dendron-cli` instead of Write tool to ensure proper frontmatter:**

```bash
dendron-cli note write --fname "note.path.here"
```

**Pattern:**

- File path: `notes/note.path.here.md`
- Dendron fname: `note.path.here` (no `notes/` prefix, no `.md` extension)

**Examples:**

```bash
# Creates notes/torchcell.datamodels.expression-schema-discussion.md
dendron-cli note write --fname "torchcell.datamodels.expression-schema-discussion"

# Creates notes/experiments.012-sameith-kemmeren.analysis.md
dendron-cli note write --fname "experiments.012-sameith-kemmeren.analysis"
```

After creating with `dendron-cli`, you can append content using the Edit tool.

**Important:** Only use `dendron-cli` for NEW notes. Use Edit tool for modifying existing Dendron notes (append mode to preserve frontmatter).

### Dendron Note Organization - Date-Stamped Sections

**Pattern for organizing note content over time:**

Use H2 headings with date prefix to track changes and additions:

```markdown
## YYYY.MM.DD - Initial Analysis

Content from this date...

## YYYY.MM.DD - Updated Implementation Plan

Revised content from later date...

## YYYY.MM.DD - Final Results

Latest updates...
```

**Guidelines:**

- **Date format:** `YYYY.MM.DD` (e.g., `2026.01.19`)
- **Heading level:** Always use H2 (`##`) for date-stamped sections
- **Title:** Add descriptive title after date separator (` - `)
- **Scope:** All content below a dated H2 belongs to that date context until the next H2
- **Revisions:** If a note block needs significant revision, create a new dated H2 section rather than editing the old one (preserves history). We can remove as we see fit.
- **Chronology** Time goes from top to bottom. Newer notes should be below older notes.

**Example from actual notes:**

```markdown
## 2025.08.29 - Model Update for Torch Compile Optimization

### Problem Analysis - Torch Compile Optimization

The current DCell implementation experiences significant graph breaks...

### Implementation Plan - ModuleList Conversion

1. Phase 1: Create dcell_opt.py
2. Phase 2: Test torch.compile compatibility

## 2025.09.02 - Plan Revision

### Problem Analysis - Sequential Processing Bottleneck

After profiling, discovered sequential processing is the real issue...

### Implementation Plan - Parallel Stratum Processing

Updated approach based on new findings...
```

**Benefits:**

- ✓ Clear temporal tracking of note evolution
- ✓ Preserves historical context (don't overwrite old dates)
- ✓ Easy to find when changes were made
- ✓ Multiple H2 sections per note = multiple work sessions documented

## Structure of Python Program Outputs

`notes/assets/scripts/add_frontmatter.py` is used with vscode cmd to apply frontmatter to python files. This creates a linked dendron note file that I will use for natural language notes associated with this file. Here is an example.

Front matter:

```python
# experiments/013-uncharacterized-genes/scripts/triple_interaction_enrichment_of_uncharacterized_genes.py
# [[experiments.013-uncharacterized-genes.scripts.triple_interaction_enrichment_of_uncharacterized_genes]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/013-uncharacterized-genes/scripts/triple_interaction_enrichment_of_uncharacterized_genes
```

- python file: `experiments/013-uncharacterized-genes/scripts/triple_interaction_enrichment_of_uncharacterized_genes.py`
- image output: `notes/assets/images/013-uncharacterized-genes/gene_interaction_counts_2025-12-21-06-26-21.png`
  - Note that the path matches experimental dir, but is name such so we don't have to go back dirs in relative path for associated note file.
  - If we are outputting images, they must be references in some dendron note file. The most logical place to md image referens is in the note file that is associated with the python script.
- data output: `experiments/013-uncharacterized-genes/results`

```bash
michaelvolk@M1-MV-6 torchcell % tree experiments/013-uncharacterized-genes/results
experiments/013-uncharacterized-genes/results
├── dubious_genes.json
├── gene_interaction_counts.csv
├── only_dubious_genes.json
├── only_uncharacterized_genes.json
├── uncharacterized_genes.json
├── uncharacterized_triple_interactions.csv
└── union_all_genes.json

1 directory, 7 files
```

- note file: `notes/experiments.013-uncharacterized-genes.scripts.triple_interaction_enrichment_of_uncharacterized_genes.md`

Note files can then be linked to from our current weekly note where we track todos and progress. For example in.

`notes/user.Mjvolk3.torchcell.tasks.weekly.2026.02.md`

We have the following, which helps we track work easily.

```markdown
## 2026.01.06

- [x] notes on [[Triple_interaction_enrichment_of_uncharacterized_genes|experiments.013-uncharacterized-genes.scripts.triple_interaction_enrichment_of_uncharacterized_genes]]
```

### Markdown Formatting

All markdown files in `notes/` are automatically formatted on save using the markdownlint VSCode extension. Configuration is stored in `torchcell.code-workspace` (lines 79-98).

**Markdownlint Rules (from workspace config):**

- **MD007**: `{ "indent": 2 }` - Use 2-space indentation for nested lists
- **MD008**: `false` - Don't require consistent header style
- **MD013**: `false` - No line length limit
- **MD014**: `false` - Dollar signs in shell commands allowed
- **MD018**: `false` - No space required after hash on atx style headers
- **MD025**: `false` - Multiple top-level headings allowed
- **MD026**: `false` - Trailing punctuation in headings allowed
- **MD029**: `false` - Ordered list item prefix doesn't need to be ordered
- **MD033**: `false` - Inline HTML allowed
- **MD036**: `false` - Emphasis used instead of heading allowed
- **MD040**: `false` - Fenced code blocks without language allowed
- **MD045**: `false` - Images allowed without alt text
- **MD050**: `false` - Strong style doesn't need to be consistent

Write clean, well-structured markdown and VSCode will handle consistency automatically. Try to abide by these rules so we don't have to manually save to reformat.

### Saving Images in Python

- All images should be saved in `ASSET_IMAGES_DIR`
- Do this by using `load_dotenv`
- Time stamp the images with by using torchcell/timestamp.py

The common pattern is `(osp.join(ASSET_IMAGES_DIR, f"{title}_{timestamp()}.png"))`

- We do this to iterate on the plots until we are satisfied with them. Then we will typically ask for `timestamp` to be removed once we are satisfied.

The common pattern is `(osp.join(ASSET_IMAGES_DIR, f"{title}}.png"))`

## Figure & Plotting Standards (repo-wide)

Whenever you generate a figure/plot anywhere in this repo (paper figures especially,
but as a general rule everywhere), follow these. Read this section, the palette, and
ideally a couple of already-approved paper figures before writing plot code. Follow
them to the best of the plot's ability (some plots legitimately deviate — say so).

- **Figure-building reference (START HERE) — [[paper.nature-biotech.figures]]** (the
  "Figure-building reference assets" section). The single, machine-independent pointer for
  every figure: the draw.io **sizing template**
  `notes/assets/drawio/figure-sizing-template.drawio.svg` (compose panels at true Nature
  print size — full 180 mm / column 88 mm, ≤170 mm tall), the **color palette**
  (`color-palette.svg` + `torchcell.utils.PLOT_PALETTE`), panel widths + true-size export
  (`PANEL_WIDTHS_MM`, `mm_to_in`, `savefig_true_size_svg`), and the plot-label display map
  (`torchcell.utils.display_label`). This lives in the committed repo (CLAUDE.md + that
  note), NOT in any local config, so it is identical on every machine.
- **Palette — one ordered source of truth.** Colors come from
  `torchcell.utils.PLOT_PALETTE` / `PLOT_PALETTE_FILL` (code) and
  `notes/assets/images/color-palette.svg` (visual, generated by
  `notes/assets/scripts/generate_color_palette.py`). **18 colors = the hue order orange
  · red · purple · yellow · blue · gray, repeated 3 times.** A series of N takes the
  first N, so the four warm primaries are always spent before blue/gray — that is what
  maximizes the primaries and minimizes blue. Keep it green-free.
  - **1–6 primary** — the draw.io Fig 1 colors (orange `#D79B00`/`#FFE6CC`, red
    `#B85450`/`#F8CECC`, purple `#9673A6`/`#E1D5E7`, yellow `#D6B656`/`#FFF2CC`, blue
    `#6C8EBF`, gray `#666666`). **LOCKED — never change these**; the figures use them.
  - **7–12 dark** — a uniform `0.73` shade of the primary.
  - **13–18 muted earth** — chroma-reduced siblings: terracotta, rose-brown, dusty
    mauve, sand, slate, taupe.
- **Tiers differ by CHROMA, not lightness — and this is load-bearing.** Lightness is
  already spoken for: light-vs-dark encodes *validation vs test within a series* (that is
  what `PLOT_PALETTE_FILL` is for). If lightness also separated *categories*, the two
  encodings would collide. Desaturation is also what makes a color read as earth tone
  rather than fluorescent — a high-chroma color at high lightness goes neon (`#FC938C`).
  Cap new colors at chroma `C* <= 36`.
- **18 is the ceiling; do not "just add two more."** A 4th red has nowhere to go — its
  three slots already occupy the mid and dark range, so a 4th can only go lighter, i.e.
  pink. A 4th orange at low lightness turns olive-brown, breaking green-free. Verified by
  CIELAB search: every route past 18 reintroduces fluorescence. **If a plot needs more
  than 18 series, disambiguate with hatching, not with more color.** Any change to the
  palette must keep the minimum pairwise ΔE (CIE76) above ~15 for the non-locked colors.
- **Use the LINE/border colors for plot marks — we do NOT fill plot marks with the pale
  fill colors.** `PLOT_PALETTE` (line colors) is the plot series palette: bar faces,
  lines, markers. `PLOT_PALETTE_FILL` (the light fills) is the draw.io object companion
  (Fig 1), used only as the *lighter* member of a two-level bar (e.g. validation = line
  color, test = its fill) — never as the primary plot color.
- **Representation/patterns in solid black.** Hatches, bar edges, and pattern marks are
  black at full opacity. To make a bar lighter (test vs validation), use a lighter
  *solid* color (the fill), NOT `alpha=` on the whole bar — alpha fades the black
  hatch/edge too (use `to_rgba(color, a)` on the face only if you must).
- **Tenth gridlines** on 0–1 metric axes: a gridline every 0.1, labelled only every 0.2
  (`MultipleLocator(0.2)` major + `MultipleLocator(0.1)` minor, minor tick marks hidden).
- **Panel width is STRICT; height is LOOSE.** Design each panel to one of the standard
  widths so panels tile across Nature's 180 mm page. Use the constants in
  `torchcell.utils` (do not hardcode/eyeball):
  `PANEL_WIDTHS_MM = {full: 179, wide: 118.9, half_plus: 88.5, half: 88, third: 57.8, sixth: 28.3}`
  (mm). Width is what makes panels align across a row. Height follows the content but
  must stay `<= MAX_HEIGHT_MM` (170 mm). Convert with `mm_to_in(...)` for `figsize`.
- **Box the plot:** full black border — all four spines visible, ~0.5 pt (not the
  top/right-despined look).
- **Type:** Arial, 6 pt everywhere (Nature minimum); set `svg.fonttype: "none"` so
  text stays real/editable in the SVG.
- **Export true-size SVG** via `torchcell.utils.savefig_true_size_svg(fig, path)` so it
  imports at true mm in draw.io (it rescales matplotlib's 72-dpi points to draw.io's
  100-units/inch). Do NOT pass `bbox_inches="tight"` on a fixed-width panel — it
  defeats the width template (fine only for a standalone legend crop).
- **Reference prior work:** match the look of already-approved paper figures in
  `notes/assets/images/*_palette.svg` where reasonable.

## When Creating or Designing new Experiments

These env vars will be read in in python with:

```python
from dotenv import load_dotenv
```

These are the most relevant dirs. (These are just example paths, they can change so just read in the variable.):

```bash
DATA_ROOT="/scratch/projects/torchcell-scratch"
ASSET_IMAGES_DIR="/home/michaelvolk/Documents/projects/torchcell/notes/assets/images"
EXPERIMENT_ROOT="/home/michaelvolk/Documents/projects/torchcell/experiments"
```

**`DATA_ROOT`**

- Where all of the large data is saved becuase this dir is typically mapped to larger storage.
- Datasets for training
- Datasets used for inference
- Inference results too are often put here because the outputs can be very large.

**`ASSET_IMAGES_DIR`**

- Where we want to save all image output
- Often convenient if there are more than one image for that given experiment to put them into dir with something like `osp.join(ASSET_IMAGES_DIR, f"006-kuzmin-tmi-inferece_1_{timestamp}")`. Often we like to start by default putting time stamps on images and we iterate to a decent idea, then if we want things to stably overwrite we will ask to remove `timestamp` after. So add by default.

**`EXPERIMENT_ROOT`**

This is what a typical experiment looks like. There is configuration in `conf/` that parameterizes experiements written in `scripts`. Slurm scripts are also in scripts and they typically have matching names with their experiments. Inside naming is also typically a designation of which slurm machine or partition a given scripts belongs to. We try to copy slurm scripts are typically numbered in a fashion that matches yamls to have a very transparaent experiment ecosystem. If we are running sweeps with optuna or wandb. we break from this pattern because we only need one config specifying the script - so we can many outputs for one script instead of one-to-one. Sometimes we inspect models here and dump `profile_results/`. `slurm/outputs` is resereved for concatenated slurm output and error. `queries/` holds the queries that the experiment used for the study.

```bash
/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi
├── conf
├── profiling_results
├── queries
├── results
├── scripts
└── slurm

6 directories, 3 files
```

Here is more depth on a more recent experiment. We cannot show for previous becuase it is too deep.

```bash
(base) michaelvolk@gilahyper torchcell % tree /home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi                       16:57
/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi
├── conf
│   ├── default.yaml
│   ├── equivariant_cell_graph_transformer_cabbi_000_inference_r0.yaml
│   ├── equivariant_cell_graph_transformer_cabbi_000_inference_r1.yaml
│   ├── equivariant_cell_graph_transformer_cabbi_000_inference_r2.yaml
│   ├── equivariant_cell_graph_transformer_cabbi_000.yaml
│   ├── equivariant_cell_graph_transformer_cabbi_002.yaml
│   ├── equivariant_cell_graph_transformer_mmli_001.yaml
│   └── equivariant_cell_graph_transformer_mmli_003.yaml
├── queries
│   └── 001_small_build.cql
├── results
├── scripts
│   ├── equivariant_cell_graph_transformer_inference_1.py
│   ├── equivariant_cell_graph_transformer.py
│   ├── gh_equivariant_cell_graph_transformer_inference_1_E006_M01.slurm
│   ├── gh_equivariant_cell_graph_transformer_inference_1_E006_M02.slurm
│   ├── gh_equivariant_cell_graph_transformer_inference_1_E006_M03.slurm
│   ├── gh_select_12_and_24_gene_top_triples.slurm
│   ├── igb_cabbi_equivariant_cell_graph_transformer-ddp_000.slurm
│   ├── igb_cabbi_equivariant_cell_graph_transformer-ddp_002.slurm
│   ├── igb_mmli_equivariant_cell_graph_transformer-ddp_001.slurm
│   ├── igb_mmli_equivariant_cell_graph_transformer-ddp_003.slurm
│   ├── inference_dataset_1.py
│   ├── __init__.py
│   ├── __pycache__
│   │   └── inference_dataset_1.cpython-313.pyc
│   ├── query.py
│   └── select_12_and_24_genes_top_triples.py
└── slurm
    └── output
        ├── 010-inference-M01_686.out
        ├── 010-inference-M02_690.out
        ├── 010-inference-M02_696.out
        ├── 010-inference-M02_698.out
        ├── 010-inference-M03_693.out
        ├── 010-inference-M03_697.out
        ├── 010-inference-M03_699.out
        └── 010-select-12-genes_694.out
```

## Troubleshooting  Deep Learning Experiments

Before investigating complex distributed computing issues (DDP sync, TorchMetrics configuration, Lightning hooks), **verify your data paths first**.

Dendron note source: `notes/experiments.010-kuzmin-tmi.false-torchmetrics-bug-bc-wrong-dataset-path.md`

## Running Experiment Scripts

All experiment scripts should be run from the torchcell root directory:

```bash
# From /Users/michaelvolk/Documents/projects/torchcell/
python experiments/<experiment-id>/scripts/<script-name>.py
```

This ensures consistent path resolution for `DATA_ROOT`, `ASSET_IMAGES_DIR`, `EXPERIMENT_ROOT`.
