---
id: hf026kinlayer2026090
title: Handoff 2026.09.02 - Metabolism Kinetics Layer
desc: 'Transfer document: state of the flux layer branch, and the kinetic-parameter infrastructure to build next'
updated: 1788400000000
created: 1788400000000
---

## 2026.09.02 - Transfer to a new session

Read this first, then [[experiments.026-metabolism-flux.enzyme-constrained-thermodynamic-flux-layer]].

**Branch:** `feat/metabolism-flux-module`, worktree
`~/Documents/projects/torchcell.worktrees/feat/metabolism-flux-module`.
**PR #301 is OPEN and NOT landed.** One commit, `bbb02a06`. Do not land it without the
author saying so; it is the reviewable record of the work below.

### What is already done and committed

| area | state |
| --- | --- |
| `torchcell/metabolism/constraints.py` | GEM to tensors: sparse `S`, bounds, GPR as catalytic units, thermodynamics, independent balance rows, null-space basis. Done. |
| `torchcell/metabolism/flux_layer.py` | The differentiable layer. Box, directionality, enzyme capacity exact; mass balance, second law, budget, dissipation as smooth penalties. No binary variables. Done. |
| `torchcell/metabolism/parameters.py` | Kinetic parameters as a database-first policy with per-value provenance. **Skeleton only, see below.** |
| `torchcell/metabolism/media.py` | Media ontology to exchange bounds, organism-agnostic. Done. |
| `torchcell/models/cell_graph_transformer_metabolism.py` | `FluxMetaboliteHead`, `FluxScalarHead`. Done. |
| `experiments/026-metabolism-flux/` | Audit, 5-arm sweep, FVA baseline, box-reachability, media audit, sampler. Done. |
| tests | `tests/torchcell/metabolism/test_flux_layer.py`, 13 property tests. ruff and mypy strict pass. |

Numbers worth not re-deriving: `rank(S)` 2,593, nullity 1,538, 3,728 catalytic units,
1,161 GEM genes, 274 exchanges, wild-type growth 0.0858 h^-1, FVA licensed set 1,532
reactions (2,818 of width <= 1 minus 1,286 blocked), betaxanthin noise ceiling 0.914.

### The four things measured that drive everything below

1. **k_cat coverage is 4.0 %**, 148 of 3,728 catalytic units, from the Open Enzyme
   Database *S. cerevisiae* slice (1,126 rows). This is the binding constraint.
2. **A sigmoid box cannot reach a sparse flux vector.** Zero flux is an asymptote, so the
   mass-balance residual pins at 1.99, its maximum. Null space balances to 1e-8 and leaves
   27 % of reactions out of bounds.
3. **All 441 reactions with a degenerate recomputed standard energy are transport
   reactions**, so the transport term is readable off the model's own curation.
   `use_shipped_transport_delta_g` implements this, is tested, and was OFF in every
   reported arm.
4. **All four datasets emit a name-only `Media` with zero components.** The dataset to
   medium join is a name string.

### WORK QUEUE, in the order the author asked for it

#### 1. The kinetic model layer: support all eight Wu Figure 3 predictors, plus three more

`wuSystematicallyExploringYeast2026` (Nature Catalysis, doi 10.1038/s41929-026-01523-w,
mirrored) Figure 3 panels a-h are **eight predictors**, and the author asked to support all
of them. **They are not the three named earlier in conversation**, so the target set is the
union of eleven:

| panel | model | predicts | inputs | repo |
| --- | --- | --- | --- | --- |
| a | DLKcat | k_cat | sequence + substrate SMILES | `SysBioChalmers/DLKcat` |
| b | UniKP-k_cat | k_cat | sequence + substrate SMILES | `Luo-SynBioLab/UniKP` |
| c | EITLEM-Kinetics-k_cat | k_cat | sequence + substrate SMILES | `XvesS/EITLEM-Kinetics` |
| d | TurNuP | k_cat | sequence + **full reaction SMILES** | `AlexanderKroll/kcat_prediction` |
| e | DeepEnzyme | k_cat | sequence + **AlphaFold structure** + SMILES | `hongzhonglu/DeepEnzyme` |
| f | Boost_KM | K_m | sequence + reactant SMILES | `AlexanderKroll/KM_prediction` |
| g | UniKP-K_m | K_m | sequence + reactant SMILES | same UniKP repo |
| h | EITLEM-Kinetics-K_m | K_m | sequence + reactant SMILES | same EITLEM repo |
| -- | KcatNet | k_cat | sequence + substrate SMILES | author-named, not in Wu |
| -- | RealKcat | k_cat **and** K_m | sequence + substrate SMILES | author-named, not in Wu |
| -- | DEKP | k_cat | sequence + optional structure | author-named, not in Wu |

Five of the eleven share exactly the `(sequence, substrate SMILES)` signature, which is the
one the `KcatPredictor` protocol in `parameters.py` already declares. TurNuP needs a
reaction SMILES and DeepEnzyme needs a structure, so the protocol needs widening to carry
an optional reaction SMILES and an optional structure path.

**Architectural decision the author made: precompute like an embedding dataset, then assign
in the model.** So this mirrors `torchcell/datasets/embedding.py` and the node-embedding
builder, not a live call inside the forward pass. Build:

- a `KineticParameterDataset` keyed by `(uniprot, substrate_id, predictor, parameter)`,
  materialized to LMDB under `$DATA_ROOT/data/torchcell/kinetics/<organism>/<predictor>/`,
  with the same provenance discipline as the literature mirror: sha256 per artifact, the
  exact retrieval or inference command, and the model checkpoint version;
- a resolver that assembles a per-catalytic-unit `k_cat` vector from that store using the
  existing database-first policy, so a run records which predictor supplied which value;
- **caching keyed on the ORF, not on the run.** The author was explicit: "if an ORF with a
  name is well defined then we shouldn't have to recompute it." A well-defined systematic
  name plus a substrate id plus a predictor version is a stable cache key forever.

**Wu reports no accuracy numbers for any of the eight.** Any ranking must come from
Supplementary Table 8, which is **not in our mirror** (the mirrored SI has only
Supplementary Figures 1-20), or from the primary model papers. Do not invent a ranking.

Wu's own procedure, for reference: everything is predicted, nothing is looked up, there is
no ensembling, and **no temperature or pH conditioning is applied at all**. That last point
is a real gap and our layer already does better, since `resolve_parameter` selects at the
assay temperature nearest 30 C.

#### 2. The Arnold-group database is the WRONG database, and this is now measured

`longEnzymeEngineeringDatabase2026` is **EnzEngDB** (Long et al., *Nucleic Acids Research*
database issue, doi 10.1093/nar/gkaf1142). It was read in full, and its own repositories
were queried read-only. **It cannot help the yeast-GEM coverage problem, and the answer to
"how many Yeast9 gene enzymes exist in it" is zero.** That figure does not need to be
plotted, because there is nothing to plot.

| what we needed | what EnzEngDB has |
| --- | --- |
| $k_{\mathrm{cat}}$, $K_M$ | **none.** Every assay output is flattened into one untyped `fitness_value` column, by explicit design |
| EC number | no field |
| organism / taxid | no field, 0 occurrences |
| UniProt accession | 13 of 1,342 legacy rows, **9 unique, none yeast**; the field does not exist at all in the live schema |
| *S. cerevisiae* content | **0 occurrences** of `yeast`, `cerevisiae` or `Saccharomyces` in the paper, in all 140 public experiment records, and in all 1,342 rows of the analysis table |

It is a **directed-evolution sequence-function corpus**: 6,234 gold-standard entries, 1,846
variants, 635 reactions, over six scaffolds that are bacterial and archaeal (P450-BM3 from
*Bacillus megaterium*, *Aeropyrum pernix* protoglobin, *Rhodothermus marinus* cytochrome c,
*Pyrococcus furiosus* tryptophan synthase) engineered for carbene and nitrene chemistry
**that does not occur in yeast**. The only join it offers is sequence-to-sequence
alignment, so any hit against a yeast-GEM gene would be a homology artifact rather than a
catalytic-unit match. 93 % of its measurements are whole-cell rather than purified enzyme,
and the authors explicitly disclaim cross-experiment comparison, so `fitness_value` is not
on a common scale across the 140 experiments.

**The paper hands us the right targets in its own literature review.** It names the two
pipelines it deliberately does not duplicate, describing them as having assembled
**"&gt;90,000 curated records"** of exactly $k_{\mathrm{cat}}$, $K_M$ and
$k_{\mathrm{cat}}/K_M$ at precision 0.8 to 0.9:

- **EnzyExtract**, Wei et al., *Protein Science* 2025;34:e70251
- **Enzyme Co-Scientist**, Jiang et al., bioRxiv 2025.03.03.641178

**These two are the next database extraction, and they should get the same treatment
EnzEngDB just got** before any client is written: schema, access route, join identifiers,
assay-condition metadata, and measured coverage against the 3,728 yeast-GEM catalytic
units. Do not assume they are scriptable.

**EnzEngDB is still worth having, on a different axis.** It is a clean, DOI-attributed,
sha256-checksummed, sequence-level genotype-to-phenotype corpus of designed protein
variants with reaction context, which fits the "perturbation as an edit to genomic content"
and inverse-design framing directly. It is cheap to mirror: 46 MB, no authentication,
CC-BY-4.0. Scope it as a bioproduction and enzyme-design dataset, never as an Open Enzyme
Database replacement.

Access, verified read-only, if it is ever wanted:

- **Zenodo, the citable immutable route**, DOI `10.5281/zenodo.17310823`,
  `Data.zip` 30,050,527 B md5 `c68ca58c1670ac57091bc5726b613510`.
- **GitHub** `ssec-jhu/levseq-dash`, 140 experiments under `levseq_dash/app/data/<id>/`,
  each a CSV plus a JSON carrying a `csv_checksum`, plus a CIF. Pin a commit, not `HEAD`.
- **Analysis table** `fhalab/EnzymeEngineeringDB`, `data/protein-evolution-database_V6.csv`,
  7,719,042 B, sha256 `abe6bd9f0c1dcc3487b6a2a595d427b70c6714b9a2bfd1c195e50b702ffe9245`.
- **There is no REST API, and the site is a trap for a client.** It is a Plotly Dash
  single-page app that returns the same 11,683-byte index page with **HTTP 200 for every
  path**, including invented ones. A 200 there means "catch-all", not "endpoint exists".

#### 3. Two paper-quality figures

- **kcat and Km distributions for all Yeast9 genes**, in Wu Figure 3 style. Read off the
  figure directly, the style is: **ECDF line plots, one facet per predictor**, two overlaid
  series, **log x** (kcat ticks 1e-4 to 1e4 s^-1; Km ticks 1e-3 to 1e1 mM), **linear y from
  0 to 1 labelled "Distribution"**, a **dashed horizontal line at y = 0.5**, and each
  curve's median annotated with a leader line and a numeric label in the series color. All
  four spines, no other gridlines. Under our standards: `PLOT_PALETTE` warm primaries
  first, Arial 6 pt, boxed, `third` or `half` panel width.
- **Yeast9 catalytic-unit coverage across every kinetics source**, same standards: Open
  Enzyme Database at 4.0 %, EnzEngDB at 0 % and why, and whatever EnzyExtract and Enzyme
  Co-Scientist turn out to give. The point of the figure is the gap that motivates the
  predictors, so a source contributing nothing still belongs on it.

Wu's medians, for a sanity check against ours: underground vs known, DLKcat 5.52/5.28,
UniKP 3.43/4.19, EITLEM 4.58/3.64, TurNuP 10.09/11.01, DeepEnzyme 6.59/5.88 s^-1;
Boost_KM 0.25/0.11, UniKP-Km 0.21/0.11, EITLEM-Km 0.25/0.07 mM. The kcat direction is
inconsistent across predictors, three up and two down; the Km direction is consistent
across all three, which is the paper's actual claim.

#### 4. Heterologous perturbation, and why it is the same problem as DNA embeddings

The author's framing: a future perturbation adds a heterologous gene or a new promoter, so
the entity universe is not fixed. There is no equivalent mechanism today for DNA
embeddings either. The sketch the author gave:

- the key set is **the union of all perturbation loci with the reference loci**;
- build the dictionary once over that union, and do not rebuild it continuously;
- an ORF with a well-defined name is stable, so it is cached permanently and never
  recomputed.

This is the same shape as the fixed-N problem already recorded as WS-NS1 in
[[plan.cgt-metabolism-flux-layer.2026.07.26]], and the near-term trick recorded there is to
**pre-extend the entity universe** with the union of cassette metabolites and reactions
across the pigment datasets and mask them off for strains lacking the cassette. Treat the
kinetic cache and the embedding cache as one design, not two.

#### 5. Redesign the Flux Cone Learning comparison

The author rejected the comparison currently written up. Do **not** compare 3-class
accuracy on their 811 genes. Instead:

> compare directly against their released sample vectors: fit one of our models, then ask
> whether the flux vector **distributions** are similar at all.

Merzbacher's samples are already mirrored at `databases/fcl/paper_merzbacher_2025/`
(Zenodo 10.5281/zenodo.15761895): 1,159 single deletions, 124 samples per cone,
143,716 samples x 4,130 fluxes, 4.43 GB. So the comparison is a distribution-to-distribution
one on an existing dataset, per reaction and per genotype. Rewrite the "GECKO: comparison,
not parity" and Merzbacher sections of the report accordingly.

#### 6. Open question the author raised

"Are we sure these models fit in the betaxanthin image." Betaxanthin sits at r ~ 0.1
against a 0.914 ceiling in every arm, so the honest reading is that nothing fits it yet.
Worth resolving whether that is capacity, epochs, the readout, or the target.

### Document state, and what is still owed

The report now has **20 numbered table captions and 5 numbered figure captions**, all with
detail. Tables were shrunk to `\footnotesize` with tighter column padding via
`notes/assets/publish/tex-templates/header-includes.tex` (a shared file, so this affects
every note PDF). The draw.io figure is now a **true-vector outlined-text SVG**, produced by
`notes/assets/publish/scripts/drawio_vector_svg.sh`, which round-trips draw.io's PDF export
through `pdftocairo -svg`; draw.io's own SVG uses HTML `foreignObject` labels that
`rsvg-convert` silently fails to draw.

Still owed on the document:

- the author flagged the mass-balance axis label as botched in typeset. It is
  `median |[Sv]_i| / omega_i`, the median over metabolites of the mass-balance residual as
  a fraction of that metabolite's turnover, and **2.0 is its maximum**, reached when a
  metabolite is only produced and never consumed. Spell that out in the caption rather
  than leaving the symbol bare.
- Table 11 and its caption straddle a page break.

### Environment and conventions, so the next session does not rediscover them

- Interpreter `~/miniconda3/envs/torchcell/bin/python`, always with `PYTHONPATH=$PWD` from
  the worktree root, or the primary checkout's torchcell wins.
- `DATA_ROOT=/scratch/projects/torchcell-scratch`. GEM at
  `$DATA_ROOT/data/torchcell/yeast-GEM/yeast-GEM-9.0.2`. OED mirror at
  `$DATA_ROOT/data/enzyme_kinetics/open_enzyme_database/scerevisiae`.
- Dataset `fig6_pigment_transfer` is built, 294 MB, 4,930 aggregated genotypes. Never
  rebuild it.
- **NEVER write to Zotero.** Papers go to the mirror.
- `trash` is not installed here; move to the scratchpad instead.
- A sweep run is ~16 min per (arm, seed) at 20 epochs, batch 128, on one GPU.
- The sweep loop is **seed-major on purpose**, so an interrupted prefix is still balanced
  across arms.

### Two decode traps that cost real time, do not hit them again

1. `phenotype_sample_indices` indexes the experiment **within** a genotype and is not
   offset across the batch. The batch row is `phenotype_values_batch`.
2. Betaxanthin and the metabolome share the label `metabolite_level`, so they are separated
   only by group **width**, 1 versus 19, never by position. Measured over 1,200 records the
   betaxanthin group comes first in only ~11 % of co-measured genotypes.

And one metric trap: an **absolute** variance floor reports a collapsed prediction as NaN
in some epochs and as exactly 0.0 in others. The guard in `masked_pearson` is now relative
to the target's spread.
