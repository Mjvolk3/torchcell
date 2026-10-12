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

## 2026.09.12 - Figure and document updated for what was built

The mosaic is implemented ([[torchcell.datasets.scerevisiae.bloom2019]], PR #364) and it differs from the 2026.08.26 design in one respect: it is a sibling genotype class, `SegregantGenotype` with `HaplotypeBlock`s between two `SegregantParent`s, not a new perturbation leaf. A perturbation leaf would have to satisfy `GenePerturbation`'s gene-name regex, and `Genotype.sort_perturbations` walks the list on gene names; a `Genotype` with an empty list would read as wild-type S288C. The sibling is added only to the unions and maps, so no served closure moved (schema-impact gate: five added symbols, one stale enum, no served loader).

Panel e of the figure now states what is stored as built, and its title says so. The generator that emits the `.drawio` is now committed beside it as `notes/assets/drawio/eqtl-experiment-and-genotype-inference.gen.py`; the scratchpad copy it was run from had drifted behind the committed diagram (panel d width, the gene-conversion box position, the legend geometry, gridlines, dot jitter ranges), and it was reconciled cell by cell until regenerating reproduced the committed diagram exactly except panel e. Panels a to d are unchanged. Sec. 3 of the document gains "What was built": the sibling decision, the measured block counts (cross A median 83, the other 15 crosses 90 to 143 at the median with tails to 1,847, so "compact" holds in bytes but not as a property of meiosis alone), which of the what-to-do items closed, and that the intergenic question is decoupled from this class but still open.

## 2026.10.11 - Boocock 2025 built on the same genotype class; cardinality measured

Boocock 2025 is ingested ([[torchcell.datasets.scerevisiae.boocock2025]], 79,516 records), which closes two open items of the 2026.08.26 section. The candidate row is corrected: eLife 2025, doi 10.7554/eLife.95566; 7,124 cells of 393 pooled segregants plus 79,123 de novo segregant cells from three crosses, and no galactose condition.

The posterior the 2026.08.26 finding 1 describes is stored as released, not binarized: a de novo cell's blocks are runs of constant called parent and constant P(called parent) on a 0.05 grid, so each marker re-expands to its exact call and a posterior within 0.025 of the release (checked at L2 on every record). That needed one schema change: `SegregantGenotype` now refuses adjacent same-parent blocks only when they also share the posterior. The measured cost (`experiments/036-dataset-fixes-before-kg-build/scripts/boocock2025_release_inventory.py`): a called-parent run is 57 to 82.5 blocks a cell at the median (2.7 to 3.9 KB), the 0.05 grid 533 to 729 blocks (25 to 34 KB), and the built store 48 KB a record at the median, 3.79 GB in all. Per cell the posterior is uncertain at 25% to 47% of the LD-pruned markers (strictly inside 0.05 to 0.95), which is the spread finding 4 predicts; the pooled panel's per-cell agreement with whole-genome calls, recomputed on the 2,641 markers the release shares with the lookup genotypes, is 92.12% at the median against the paper's 92.5% on its full marker set.

Finding 4's rescue holds as stated: the 393 pooled segregants are stored with the whole-genome calls the cells were matched to, and the de novo cells, which have no other genotype, keep their own posterior.

## 2026.10.11 - N'Guessan 2025: the third eQTL row, held on its panel's genotypes

N'Guessan 2025 ([[torchcell.datasets.scerevisiae.nguessan2025]]), the third of the three rows the 2026.08.26 section named, is written as a loader and is not built. Its design is Boocock's pooled panel at a larger scale: 18,233 cells of batch 1 (4,489 strains) of the Nguyen Ba 2022 BB-QTL BY4741 x RM11-1a panel, 14,263 cells assigned to 2,951 strains, one record per strain with its cells summed. Two differences from Boocock bear on this note:

- **The genotype is not in the dataset's own release.** The release carries each cell's assigned strain as an index into batch 1; the genotypes are the panel's (Nguyen Ba 2022, 41,594 SNPs, a posterior P(RM) per strain), on a Dryad deposit that is walled (#1014). Finding 4's rescue (pool the cells, take the strain-level genotype) applies, but the strain-level genotype is a separate deposit joined through an index, so the verifier tests the join (MAT fixation against the panel's MATa expression, and cis concordance against permuted strain maps) instead of assuming it.
- **No cell-cycle stage is released**, so `SingleCellExpressionPhenotype.cell_cycle_stage_counts` became `dict | None`, required, with a None carrying a `ProvenanceGap`. Schema impact: Boocock 2025 and N'Guessan 2025 only (Boocock's dev store needs one rebuild).

The candidate row is corrected (authors N'Guessan, Tong, Heydari, Nguyen Ba; doi 10.7554/eLife.93906; not a Bloom-lineage panel) and its verdict is blocked:wall at G3. Not measured yet: the genotype cardinality at 41,594 markers, about 13 to 19 times Boocock's de novo marker counts.

## 2026.10.11 - One single-cell UMI-count leaf for Boocock 2025 and Jackson 2020 (#1016)

Owner decision on #1016: use one class for the two single-cell UMI-count leaves unless it loses data. Measured: it does not, so Jackson 2020's `PseudobulkUmiCountPhenotype` was merged into Boocock 2025's `SingleCellExpressionPhenotype` (the better name: 79,123 Boocock records and 12 Jackson records are single cells, which "pseudobulk" would mislabel) and N'Guessan 2025's loader moved with it.

### How it was decided

`experiments/036-dataset-fixes-before-kg-build/scripts/single_cell_leaf_merge_audit.py` (`round-trip` mode) streamed every stored phenotype of the two dev stores as built BEFORE the merge, through a candidate class carrying both leaves' fields (the source-specific ones nullable with a typed gap), and compared every stored field of the dump with the stored value and its JSON type. A merge counts as lossless only if no stored value is dropped, coerced, or given a second meaning.

| store (pre-merge) | records | references | round-tripped | dropped or changed | total = stored counts | total > stored counts |
|---|---|---|---|---|---|---|
| Boocock 2025 (`processed.superseded.20261011-220231`) | 79,516 | 4 | 79,516 + 4 | 0 | 35,156 | 44,360 (173,191 UMIs outside the stored universe) |
| Jackson 2020 (`processed.superseded.20261011-220009`) | 784 | 11 | 784 + 11 | 0 | 784 | 0 |

### Field by field

| field | Boocock 2025 leaf (before) | Jackson 2020 leaf (before) | merged `SingleCellExpressionPhenotype` | same type and meaning? |
|---|---|---|---|---|
| `umi_count` | nonzero counts over the analysis transcripts (`Yr`), S288C ids | nonzero counts over every released gene feature, R64 ids, markers excluded | nonzero counts over the record's pinned universe | yes: the universe is pinned per record by the next two fields |
| `n_features`, `feature_universe_sha256` | size and sha256 of the universe | same | same | yes |
| `n_cells` | cells summed (1 for a de novo cell) | cells summed | same | yes |
| `total_umis` | `nUMI` over every feature; at least the stored counts | every released feature; exactly genes plus markers | every feature the source counted; at least the stored counts | yes: the same all-feature library size. Jackson's equality is a property of its release (every feature stored), now checked by its L2 rather than the class |
| `marker_umi_count` | none | `KANMX`, `NATMX`, zeros kept | dict, or None with a typed gap | Jackson identical; Boocock gapped (reads counted against SGD R64-2-1, no cassette named) |
| `cell_cycle_stage_counts` | authors' five stages; None only with a gap (N'Guessan) | none | dict, or None with a typed gap | Boocock identical; Jackson gapped (no per-cell stage released) |
| `library_cell_counts` | cells per 10x library | none | cells per library, the count of `cell_libraries` | Boocock identical; Jackson filled: one 10x batch per flask, so the release `Condition` is the library |
| `cell_barcodes` | none (a de novo cell's barcode sits in its `segregant_id`) | droplet barcodes, unique | droplet barcodes as released | Jackson identical; Boocock filled from `barcode.features` |
| `cell_libraries` | none | none | each cell's library, aligned | new: Boocock reuses a barcode across a panel's libraries, so only (library, barcode) names a cell |
| `cell_total_umis` | none | per-cell totals, sum to `total_umis` | same | Jackson identical; Boocock filled from per-cell `nUMI` |
| `measurement_type` | `scrnaseq_umi_count` | `scrnaseq_pseudobulk_umi_count` | free tag, kept as each loader wrote it | yes; two tags for one assay remain (open decision below) |

What stays separate, because merging it would lose meaning: the experiment pairs and their references. Boocock's `SegregantExpressionExperiment` holds a haplotype mosaic and its reference is the panel pseudobulk on the S288C reference genome; Jackson's `PseudobulkUmiCountExperiment` holds a gene-keyed barcoded deletion with a `StrainReferenceGenome` (FY4/FY5) and a `CultureEnvironment`, and its reference is the pooled HO-deletion controls of the flask. Only the phenotype leaf is shared.

### Which cells were summed

The merged leaf requires the summed cells for every source, because all three releases carry them: Jackson stored them already, Boocock's `segData.RDS` has each cell's barcode, library and `nUMI`, and N'Guessan's matrix rows are cells. A bare barcode does not name a Boocock cell:

| Boocock panel (reference, every cell) | cells | distinct bare barcodes | distinct (library, barcode) pairs | reference phenotype JSON |
|---|---|---|---|---|
| Ap (2 libraries) | 7,124 | 7,105 | 7,124 | 437,686 B |
| A (2 libraries) | 27,744 | 27,713 | 27,744 | 1,329,914 B |
| B (4 libraries) | 44,784 | 44,603 | 44,784 | 2,162,352 B |
| 3004 (2 libraries) | 6,595 | 6,520 | 6,595 | 444,075 B |

So the merged leaf carries `cell_libraries` beside `cell_barcodes` and refuses a repeated (library, barcode) pair, not a repeated barcode; `library_cell_counts` must equal the count of `cell_libraries`. Jackson's eleven references are unchanged by this rule: one library each, every barcode distinct (976 of 976 in YPD, the condition with the most cells). The per-cell lists also answer which cells a pooled Boocock segregant summed, which the pre-merge store did not record.

### Schema impact and stores

`scripts/schema_impact_check.py --base origin/main`: 5 changed symbols, all breaking (`PseudobulkUmiCountPhenotype` removed and dropped from `PhenotypeType`; `SingleCellExpressionPhenotype` gains four required fields and a changed validator; the Jackson experiment pair's phenotype type changes), 3 impacted datasets: `Boocock2025Dataset`, `TfDeletionPseudobulkJackson2020Dataset`, `Nguessan2025Dataset`. Acceptable: none of the three is served (KG 3.0 holds none), and KG 4.0 is a full rebuild. The graph has one node class fewer (`pseudobulk umi count phenotype` removed; 44 node classes).

Both dev stores were rebuilt from `78ec49eb5` and verified L0 to L4 (tables in [[torchcell.datasets.scerevisiae.boocock2025]] and [[torchcell.datasets.scerevisiae.jackson2020]]). `old-vs-new` mode paired each retired store with its rebuild record by record:

| store | records paired | references paired | old values dropped or changed | added |
|---|---|---|---|---|
| Boocock 2025 | 79,516 | 4 | 0 | `cell_barcodes`, `cell_libraries`, `cell_total_umis`, `marker_umi_count` (None) and its gap, on every record and reference |
| Jackson 2020 | 784 | 11 | 0 | `cell_libraries`, `library_cell_counts`, `cell_cycle_stage_counts` (None) and its gap, on every record and reference |

### Cost

Requiring the summed cells on a reference makes Boocock's references large (the JSON sizes above: up to 2.16 MB for panel B, against Jackson's largest of 125,972 B, the RAPA flask), and the build re-serializes the whole reference for every record it interns. Measured from job 3973: the Boocock build took 2,046 s (34 min; peak RSS 11.2 GB) where the pre-merge build took about 13 min (its directories and manifest are timestamped 09:00 and 09:13), and `experiment_reference_index.json` grew from 2,315,245 to 11,514,915 bytes. The interned store grew from 3.7 to 3.8 GB, since each reference is stored once. Not measured: the per-record read cost of the larger reference through `__getitem__`. Hypothesis (untested): a consumer that validates the reference once per record pays that cost once per record, so a full pass over panel B is slower than before; the KG build's adapter is such a consumer.

### Open owner decision

`measurement_type` is still two tags for one assay (`scrnaseq_umi_count` on Boocock and N'Guessan, `scrnaseq_pseudobulk_umi_count` on Jackson). The class carries both verbatim; making them one is a value change for whichever dataset moves, so it was not done here.
