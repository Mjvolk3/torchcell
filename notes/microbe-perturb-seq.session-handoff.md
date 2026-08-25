---
id: 4oi2f8dyiptm0demh16ljs5
title: Session Handoff
desc: 'Handoff from the M1 Mac session (branch notes/pert-seq-method-review) to the gilahyper session: what notes-tex is, the two-tier Zotero bib policy, and the one analysis that must run on gilahyper.'
updated: 1787268314486
created: 1787268314486
---

## 2026.08.20 - Handoff to the gilahyper session

**Read this first if you are the gilahyper session.** It tells you what was built on
the Mac side, what you need from Zotero, and the one job that can only be done where
you are.

### Where the work is

- **Branch:** `notes/pert-seq-method-review`
- **Worktree (Mac):** `~/Documents/projects/torchcell.worktrees/notes-pert-seq-method-review`
- **Base commit at time of writing:** `6c70f426` -- nothing committed yet on this
  branch, everything below is uncommitted working-tree state. Sync before assuming.

### What is new: `notes-tex/`

A new top-level tree, the LaTeX counterpart to `notes/`. It is **not** `docs/`
(that is the Sphinx tree for the package) and **not** `paper/` (that is the Nature
Biotech manuscript, which syncs to Overleaf).

```
notes-tex/
  common/
    tcdoc.sty          shared style: status chips, tcfig, provenance markers
    Makefile.common    tectonic build, Zotero bib, draw.io export, check gate
    check_doc.py       formatting / provenance / citation gate
    build_bib.py       union-of-two-libraries bib builder  <- see below
  microbe-perturb-seq/ document #1: the perturb-seq method review + costing
```

Pairing is by **name**: `notes-tex/<slug>/` <-> `notes/<slug>.*.md` <-> the Zotero
collection called `<slug>`. Same rule that already pairs a `.py` with its note.

Build: `cd notes-tex/microbe-perturb-seq && make && make check`.
Full conventions are in `notes-tex/README.md` -- read that before editing any `.tex`.

### The two-tier Zotero bib policy (important, and new)

This is the part most likely to bite you if you regenerate a bib.

| Target | Source | Why |
| --- | --- | --- |
| `paper/nature-biotech/references.bib` | **group `paper` collection ONLY** | Publication guarantee: everything cited in the manuscript must live in a library the whole group holds, so citations stay recoverable after publication. Built by `paper/nature-biotech/zotero_export_bib.py`. **Do not point this at the personal library.** |
| `notes-tex/<slug>/references.bib` | **UNION of group + personal collections named `<slug>`** | Technical notes cite much more widely than the paper, and that reading lands in the personal library first. A promote-to-group round trip before every citation would stop notes being written. |

Collection keys currently in use for `microbe-perturb-seq`:

- group `torchcell` (id `6582362`), collection `microbe-perturb-seq` = `FE8DQKUH` (15 items)
- personal (library id `1`), collection `microbe-perturb-seq` = `AC8MFJXK` (23 items)
- union = 23 entries, 15 shared, 8 personal-only

Commands:

```bash
cd notes-tex/microbe-perturb-seq
make bib        # rebuild references.bib as the union
make bibcheck   # report-only; exits non-zero on a citekey conflict
```

**Requires Zotero running with Better BibTeX on `localhost:23119`.** If gilahyper
has no Zotero, you cannot regenerate a bib there -- use the committed
`references.bib` as-is and leave bib regeneration to the Mac. That is a real
constraint to plan around, not a bug.

### Citekey collisions: what to do when it fails

The union is where the two libraries can disagree. `build_bib.py` distinguishes:

- **Key conflict** -- one citekey, two *different* works. **Fatal**; `make bib`
  refuses to write, because whichever copy won, some citation would point at the
  wrong paper.
- **Duplicate work** -- one work under *two* citekeys. Not fatal, but it splits one
  reference into two numbered entries in the PDF.

Both are reported with the citekey, which library each side came from, both titles,
and the differing fields, so the item can be searched for directly in Zotero. Fix in
Zotero (re-pin a key, or merge the items), then re-run `make bib`. There is
currently **no CI check** -- it needs a live Zotero, which a GitHub runner does not
have. Options are noted at the end of `notes-tex/README.md`.

### The job that can only run on gilahyper

`experiments/024-perturb-seq-costing/scripts/effect_size_analysis.py`

It measures how loud a single yeast knockdown actually is, from the Kemmeren and
Sameith expression compendia, and whether the response grows with a second
deletion. It needs the built LMDBs:

- `$DATA_ROOT/data/torchcell/microarray_kemmeren2014`
- `$DATA_ROOT/data/torchcell/sm_microarray_sameith2015`
- `$DATA_ROOT/data/torchcell/dm_microarray_sameith2015`

It **fails fast with a clear message** anywhere those are absent -- no fallbacks, no
synthetic data, per the repo rule. On the Mac they do not exist, which is why it has
not been run.

```bash
python experiments/024-perturb-seq-costing/scripts/effect_size_analysis.py
```

Writes three files to `experiments/024-perturb-seq-costing/results/`:
`effect_size_per_strain.csv`, `effect_size_summary.csv`, `effect_size_multiplex.json`.

**Why it matters.** Every cells-per-perturbation number in the document currently
rests on a *nominal* two-fold change threshold. This replaces that with the observed
distribution of yeast regulatory responses. After it runs:

1. feed `median_abs_log2fc_responders` into
   `cost_model.umis_needed_for_fold_change()`,
2. re-run `render_tex_tables.py` and the plot scripts,
3. rebuild the document.

Design note already baked in: the 1-vs-2 perturbation slope is fitted **within
Sameith only** (Sm singles vs Dm doubles). Kemmeren has the larger singles sample
but a different platform and noise floor, so mixing them would confound perturbation
count with study.

### Also open

- Add `kemmerenLargeScaleGeneticPerturbations2014` and
  `sameithHighresolutionGeneExpression2015` to either `microbe-perturb-seq`
  collection so section 4.4 can cite them instead of carrying an `\external{}` flag.
- Figures reproduced from Brettner/Jariani/Yao in the method explainers need a
  copyright decision before the document goes anywhere public.

### Sync etiquette

The primary checkout stays on `main`; all work is in the worktree above. Land via
rebase + ff-only (`/enqueue-merge`), never a merge commit. **Land one at a time** --
worktrees share one object store and stash list, so two simultaneous landings
corrupt each other. If both sessions have uncommitted work on this branch, commit
and push from one, then pull in the other before touching the same files.
