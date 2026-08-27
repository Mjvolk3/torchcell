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

- Document: `notes-tex/eqtl-data-model/main.pdf`, published to Zotero under
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

Related: [[experiments.database.expansion-100]] ·
[[torchcell.datasets.scerevisiae.caudal2024]] ·
[[torchcell.sequence.plasmid-and-genomic-content-design]] · [[paper.north-star]]
