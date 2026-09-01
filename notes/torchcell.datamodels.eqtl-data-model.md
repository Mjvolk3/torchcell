---
id: o0ch0r506usd2hvethzhidv
title: Eqtl Data Model
desc: ''
updated: 1787791712946
created: 1787791712946
---

## 2026.08.26 - eQTL data, and how a meiotic mosaic fits an ontology built for gene edits

Working note for `notes-tex/eqtl-data-model/`. The typeset document is the deliverable;
this holds the decisions and the follow-ups. Promoted from a scratch note written while
triaging the eQTL rows in [[experiments.database.expansion-100]].

- Document: `notes-tex/eqtl-data-model/eqtl-data-model.pdf`, published to Zotero under
  `torchcell / notes-tex / eqtl-data-model`.
- Figure: `notes/assets/drawio/eqtl-experiment-and-genotype-inference.drawio`.
- Schema class names were READ from `torchcell/datamodels/schema.py`, not recalled.
- Boocock numbers are verbatim from the mirror
  (`$DATA_ROOT/torchcell-library/boocockSinglecellEQTLMapping2025/paper.md`).

### Why this note exists

Albert 2018, Boocock 2025 and N'Guessan 2025 are the only three rows in the 155-candidate
list scoring high on BOTH Perturb-seq axes. They also break an assumption the schema rests
on: **the genotype is never observed, it is inferred**.

### The findings, shortest form

1. **The genotype matrix is `[0,1]`, not `{0,1}`.** Boocock fits on the standardized HMM
   posterior. Binarizing at 0.5 is only for cell-identity matching.
2. **Reconstruction still works, and the reason is the interesting part.** The segregant is
   not sequenced de novo, it is ASSIGNED to one of two deep-sequenced parents at each
   locus. ~90 crossovers per meiosis -> ~10^2 blocks -> one marker fixes every cataloged
   variant in its block. Locating ~10^2 boundaries, not determining ~10^4 values.
3. **Error budget ~0.8% ambiguous** (my arithmetic from published parameters, flagged
   `\external{}` in the doc). **The bigger error source is non-crossover gene conversion**
   (~46/meiosis, ~2 kb tracts), which sparse markers miss entirely.
4. **Per cell it flips: median genotype agreement 92.5%.** Fine for an association fit over
   27,744 cells, useless for writing down a genome. Pooling (median 17 cells/segregant)
   plus the pre-existing Bloom 2013 WGS is what rescues the strain-level genotype.
5. **Reconstruction and attribution come apart.** Which-allele succeeds at ~99%;
   which-variant-is-causal never does, because within a block variants are perfectly
   collinear. Pruning at r > 0.999 concedes exactly this.

### Verdict against the sequence gate

**PASS**, conditional on holding the parent assemblies. The gate asks whether total genomic
content can be estimated, not whether a causal variant can be identified, and those come
apart cleanly here. The `segregant-WGS` basis assigned in the candidate list is justified.

### What must change before a loader

- **Do NOT reuse `SequenceVariantPerturbation`.** A natural isolate is individually
  sequenced (allele dereferences to a real sequence + sha256); a segregant is not. That
  class promises a verified sequence that does not exist for a segregant, so using it would
  encode an inference as an observation.
- **Store a haplotype mosaic instead**: `{(chr, start, end, parent, p)}`, |G| ~ 10^1-10^2,
  against two sha256-pinned parent assemblies. Honest (stores the posterior), ~100x more
  compact (dissolves the cardinality problem), lossless (variant set is a derived view),
  and linkage-preserving.
- **The blocking gap: `GenePerturbation.systematic_gene_name` is REQUIRED** and regex-
  validated (`^(Y[A-P][LR]\d{3}[WC](-[A-Z])?|Q\d{4}|YNC[A-Q]\d{4}[WC])$`). Cis-eQTLs are
  mostly PROMOTER variants, which are intergenic and have no gene to attach to. **This same
  gap blocks the entire Regulatory DNA class** (de Boer 2020, Renganaath 2020, Keren 2013),
  so resolve both together.
- **The QTL table is NOT a `Phenotype`.** It is an estimate conditional on method,
  threshold and marker density, with no source value to verify against at L2-L4.
- **Parent assemblies are an ingestion DEPENDENCY**, not context. A VCF is not an assembly.

### Open items

1. Decide the intergenic representation. Options: non-gene-keyed interval leaf; gene-keyed
   plus `ProvenanceGap` on the attribution; or coding-only (honest short-term, worst
   long-term). Gates two classes.
2. Measure the cardinality cost on ONE Albert 2018 segregant before building anything.
3. **Correct the Boocock row in the candidate list** -- recorded DOI was
   `10.1038/s41586-025-09628-1`; the mirror says **eLife 2025, `10.7554/eLife.95566`**, and
   27,744 is CELLS across **393 segregants**, not 2.7e4 segregants.

### Toolchain note

`make figures` needs draw.io, which the Makefile expects at `/tmp/drawio.AppImage` -- and
`/tmp` is cleared on reboot, same silent-breakage pattern as
[[tectonic-lives-in-local-bin]]. Restored it this session. Two gotchas found:
`APPIMAGE_EXTRACT_AND_RUN=1` does NOT work here (extract with `--appimage-extract` and run
`squashfs-root/drawio` directly), and **an XML comment between `<diagram>` and
`<mxGraphModel>` makes drawio fail with "input file/directory not found"** -- put layout
comments in the prolog above `<mxfile>`.

### 2026.08.27 - figure revised to five panels

First version was too thin. Now generated rather than hand-typed, so the repetitive cells
are exact: `scratchpad/gen_eqtl_fig.py` emits the `.drawio`, which is committed as the
artifact. Panels:

- **a** cross and the two assays, ending in X and Y, with the asymmetry called out
- **b** one chromosome at FOUR levels: true variants, marker calls (with A/B shown),
  **the HMM posterior as a trace whose ramps ARE the uncertainty**, inferred blocks. Gene
  conversion tract drawn between markers producing no ramp.
- **c** both matrices drawn as grids: X shows haplotype BLOCKS (runs of 5-11 columns, short
  runs read as noise and defeated the caption), Y as a graded heatmap
- **d** the canonical eQTL map: cis on the diagonal, trans hotspots as vertical bands
- **e** what is stored vs not, and the intergenic gap

175.2 x 168.1 mm, just inside the 170 mm cap. Layout gotchas hit: a hard-coded panel-title
width overhung to 965 units (title widths must be passed per panel), and text cells wrap
at render time so legends need measured heights, not estimated ones.

Related: [[experiments.database.expansion-100]] ·
[[torchcell.datasets.scerevisiae.caudal2024]] ·
[[torchcell.sequence.plasmid-and-genomic-content-design]] · [[paper.north-star]]

## 2026.08.31 - Review round 1: comments pulled, revised, republished

Pulled 31 annotations (25 with written comments) off the published v1 PDF
(`eqtl-data-model_2026-08-27-17-50-23_49c19dc8.pdf`) with
`notes-tex/common/zotero_comments.py` (whose `DOCS_COLLECTION` import had gone stale
against the generalized `zotero_publish.py`; fixed to walk the repo-path collection
route). Ledger, by annotation key:

- `2DVTJH4H` (cite the cross + use the bib): the document now carries a real
  `references.bib`. Eight items live in the personal `eqtl-data-model` collection
  (`4VNJWJAW`, group twin `VNDH4NMX`), added by DOI via `zotero_add_ref.py` with PINNED
  citekeys; entries emitted with `--emit-bibtex` (keyword `zotero-pending`) because
  Better BibTeX is unreachable on GilaHyper. `make bib` on the Mac regenerates.
  The BY x RM sentence cites Brem 2002.
- `FYDGICYN` (Bloom 2013 vs the new eLife paper): the eLife 2025 paper IS Boocock 2025,
  already central to the doc; Sec 1.1 now says Bloom 2013 is the deliberate reference
  point because Boocock pooled "393 previously genotyped haploid segregants" from that
  same cross, and both are cited.
- `RQ4IVE3Y` (how well are boundaries defined / known a priori?): Sec 1 now states
  boundaries are inferred per strain, not known ahead of time, and points at Sec 2.
- `HX9R8PJC` (does "segregate" mean swaps?): Sec 1.1 defines Mendelian segregation and
  states explicitly nothing is swapped or edited.
- `3DXFXAEI` / `Z66Q24SU` / `AJ8PLK69` (LOD, LD, centimorgan undefined): all three now
  defined at first use.
- `36MKHU3J` (trans structure = engineering leverage): one sentence added to Hotspots.
- `W4WUZCVU` (figure: conversion tract sat ON a marker): tract moved into the gap
  between markers in panel b (was x=462 overlapping the marker at 464).
- `Q3JJY965` (figure: panel d not matrix-like; how is the map derived?): panel d redrawn
  square (90 x 90) with gridlines, an "each dot is one significant (gene, marker) pair:
  a row of the QTL table" note, and the caption + Sec 1.1 both tie it to the QTL table.
- `JR7TT52X` (aren't they haploid?): Sec 2.4 explains disomy in haploids; figure box
  says "aneuploidy (a haploid can carry a disomy)".
- `YYJNQIPS` (do the papers estimate aneuploidy?): checked the mirrored Boocock text
  (grep aneuploid/ploid/copy number/disom: no hits); the doc now states no
  aneuploidy/CNV screen is reported there, so it is an unmodeled error term. Bloom 2013
  is not mirrored, so no claim is made about it.
- `JWQJCW9D` (each study needs custom probabilistic-genotype extraction): stated after
  the Sec 2.2 table: shared output representation, no shared recovery path.
- `Y8RL5NT2` (coupling: nothing to be done): stated as inherent to the one-pot design,
  carried not corrected.
- `IFF7Q9FY` (what is a gene conversion event?): defined in Sec 2.3, and the ~90
  crossovers / ~46 non-crossovers now cite Mancera 2008 (10.1038/nature07135).
- `S3697X5E` (parents' reads suffice to attempt assemblies): added to the non-reference
  bullet; attemptable with known residual error, not a new sequencing campaign.
- `UN3GVIIM` (per-cell insufficiency vs how torchcell is built): Sec 2.5 now says the
  record stores the strain-level mosaic, the same reference-plus-differences form the
  schema already uses, so the ingestion pattern is not threatened.
- `DYNPS52M` (Sec 2.6 confusing): rewritten as three run-in blocks (Reconstruction
  succeeds / Attribution is blocked / What each answer decides).
- `IHL4D6B3` (botched LaTeX regex): the `\\d` mangling replaced with a verbatim span;
  renders as the actual schema.py pattern now. Also fixed `\file{...\_...}` literal
  backslashes (\file is a url command; raw underscores are correct inside it).
- `RSRX2SKH` (reference usually means phenotype baseline): the gap now opens by
  separating genomic reference from phenotype baseline and closes noting expression is
  stored absolute.
- `DYMNBQCK` (aren't genotypes ordered sets?): answered precisely: `Genotype` sorts by
  gene name/type for canonical equality, which is set semantics with stable iteration;
  lexical order is not chromosomal order (verified in `schema.py:926-939`).
- `2PE9MZIL` (new type accepted): recorded as a decision in Sec 3.4, and the what-to-do
  item 2 now treats the interval-keyed leaf as existing regardless.
- `FPK8BWWJ` (synthetic locus + flanking windows): added as a live option in Secs 3.3
  and 3.5, tied to the existing 1,000 bp up / 300 bp down window convention
  (`torchcell/datasets/fungal_up_down_transformer.py:30-31`), with the window
  assignment flagged as an interpretation.
- Highlights with no written comment (`HWK9EEM6`, `4EPSVR7Y`, `5DG79NHH`, `VU2HR3I4`,
  `MBRZEZHU`, `U7HFHTFK`, `BXPGKWBB`): no action; `BXPGKWBB` ("Agreed") and
  `36MKHU3J` were affirmations.

`make check` clean (8 citations resolve, 0 style violations); v2 published to the same
Zotero collection.
