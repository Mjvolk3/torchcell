---
id: 1r7ety468vrx9ilm8lmy6m8
title: Paper Nature Biotech CGT Outline
desc: ''
updated: 1781054932460
created: 1781054932460
---

## 2026.06.09 - Nature Biotechnology Article Outline (CGT paper)

### Purpose and how to use this note

This is the figure-first outline for the TorchCell Cell Graph Transformer (CGT)
research paper, targeted at Nature Biotechnology as a research **Article**. It
exists to (1) lock the narrative spine to the SIMB 2026 abstract
([[conference.simb-2026.abstract]]) so nothing in the abstract is dropped, (2)
plan the main display items at the panel level with explicit plot-type options,
and (3) drive progress by tagging every panel as already-available versus
needs-generating. Markdown now; ported to the Nature Biotech LaTeX template in
Overleaf later (section -> .tex mapping is in the last subsection).

Prior outlines exist and are superseded by this one (they targeted a different
journal and had no panel breakdown): [[paper.outline]], [[paper.outline.02]],
[[paper.outline.03]]. Reuse their prose where useful; this note governs structure.

Status tags used below: `[HAVE]` asset exists in-repo; `[GEN]` must be generated;
`[PARTIAL]` partial / lives in the iBioFoundry-AI repo; `[DECIDE]` open choice.

### Nature Biotechnology Article format constraints (the box we draw in)

Hard limits for a research **Article** (confirmed 2026-06-09 via nature.com/nbt
submission guidelines; cross-check the live page before submission as limits
change):

| Item           | Limit                                                       | Notes                                                                                      |
|----------------|-------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| Title          | no hard Article limit stated; keep concise (aim <=15 words) | The 90-char / 10-word cap applies to Brief Communications, NOT Articles -- do not misapply |
| Abstract       | **<= 150 words**, unreferenced                              | reuse SIMB abstract                                                                        |
| Main text      | **<= 3,000 words**                                          | excludes abstract, Methods, references, figure legends                                     |
| Display items  | **<= 6** total (figures + tables combined)                  | everything else -> Extended Data / Supplementary                                           |
| References     | **<= 50** (guideline for Articles)                          | be selective in Intro/Discussion                                                           |
| Figure legends | not counted in the 3,000; keep concise                      | numeric cap varies by type -- check live page                                              |
| Methods        | no fixed word limit                                         | concise; full math/derivations live here                                                   |
| Format         | TeX/LaTeX accepted (submit compiled PDF)                    | Greek (tau, rho) via Symbol font                                                           |

Implication: we get **at most 6 main display items**. The plan below uses **5
main figures + 1 reserve/Extended-Data candidate**, with deep architecture math,
ablations, scaling curves, and per-dataset diagnostics pushed to Extended Data /
Supplementary. Per-section word budgets (below) are set to sum to ~3,000:
Intro ~500-650 + Results R1-R5 ~1,700-2,000 + Discussion ~500-650.

### SIMB abstract coverage map (top priority: nothing dropped)

Each abstract claim -> where it lands in the paper. Source sentences from
[[conference.simb-2026.abstract]].

| # | SIMB abstract claim                                                                                                                                                                        | Paper home                     | Figure   |
|---|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------|----------|
| 1 | DL advanced protein structure/function; systems-bio + metabolic engineering lagged for lack of unified phenotype data / shared ontology                                                    | Intro (motivation)             | -        |
| 2 | TorchCell = open-source framework aggregating heterogeneous ME data, predicting multimodal phenotypes                                                                                      | Intro (last para) + Results R1 | Fig 1    |
| 3 | Literature + public data annotated with shared experimental ontology -> Neo4j KG at NCSA (genotype/phenotype/environment)                                                                  | Results R1                     | Fig 1a   |
| 4 | Dataloaders build DL-ready datasets; strain = perturbation operator over shared wildtype reference (genome seq + gene networks + metabolism + environment); millions of strains at scale   | Results R1                     | Fig 1b-c |
| 5 | CGT = virtual-cell architecture; interaction graphs constrain multi-head attention via loss aligning attention to network priors; equivariant perturbation operator; multitask heads       | Results R2                     | Fig 2    |
| 6 | Trigenic GGI: CGT r=0.454+/-0.004, rho=0.421+/-0.003 (SEM); beats DANGO 0.367, DCell 0.157, Yeast9 FBA 0.0006                                                                                    | Results R3 (headline)          | Fig 3    |
| 7 | Same architecture predicts expression under single+double KO (r=0.543+/-0.023) and morphology under single KO (r=0.619+/-0.037); one latent embedding generalizes across phenotype classes | Results R4                     | Fig 4    |
| 8 | Applied CGT to recommend gene deletions for beta-carotene and betaxanthin in S. cerevisiae                                                                                                 | Results R5                     | Fig 5    |
| 9 | Unlike DBTL tools that don't represent the strain, TorchCell pairs with autonomous platforms (UIUC iBioFoundry) for iterative AI-guided strain engineering (chemicals, fuels, pigments)    | Results R5 + Discussion        | Fig 5    |

Every abstract sentence has a home. Claims 6 (GGI), 5 (CGT design), and 7
(multitask) are the load-bearing novelty; 8-9 are the translational hook.

### Working title

Primary (from SIMB): **"Accelerating metabolic engineering through multimodal
phenotype prediction with TorchCell and the Cell Graph Transformer."**
Alternatives to A/B test with co-authors:

- "A virtual-cell transformer predicts multimodal phenotypes across the genotype-environment space of yeast."
- "Graph-constrained attention links genotype to phenotype for AI-guided strain design."

### Abstract (<=150 words)

Reuse the SIMB abstract nearly verbatim (it is already ~150-word scale and
co-author-approved). One edit before submission: reconcile the error-bar
provenance (see Gaps) so all reported +/- are the same statistic.

### Introduction (target ~500-650 words)

Three beats, problem-first:

1. **The genotype-phenotype gap in metabolic engineering.** DL transformed
   protein structure/function; systems biology and strain design lag because
   phenotype data are scattered across small, heterogeneous studies with no
   shared schema. Cite the data-fragmentation problem; contrast with the
   protein-LM/AlphaFold trajectory. (Genesis-proposal framing.)
2. **Why prior approaches fall short.** Narrow task-specific ML; mechanistic FBA
   misses epistasis; recent perturbation transformers do not consistently beat
   linear baselines (Ahlmann-Eltze et al., Nat Methods 2025) -- motivates
   physically/biologically grounded representations. DCell, DANGO as DL prior art
   for genetic interactions. **Our own classical-ML baselines** (random forest,
   SVR, elastic net) predict knockout *fitness* reasonably but **fail on gene
   interactions** -- the direct motivation for a deep model (Suppl. Fig. S1; one
   sentence in main text, full figure in SI).
3. **This work.** TorchCell unifies heterogeneous ME data via a shared ontology
   into a KG; the CGT learns one cell representation that decodes multiple
   phenotype classes and reaches SOTA on trigenic epistasis; we close the loop by
   recommending strain designs for pigment production with an autonomous foundry.
   End on the contribution list (3-4 bullets).

`[DECIDE]` framing emphasis: lead with "first foundation-model-style multimodal
phenotype predictor for ME" vs lead with "SOTA epistasis prediction." Recommend
the multimodal/representation framing as primary, epistasis as the proof point.

### Results (target ~1,700-2,000 words across R1-R5)

**R1. A shared ontology and knowledge graph unify metabolic-engineering data.**
TorchCell experiment ontology -> BioCypher -> Neo4j KG at NCSA; dataloaders emit
DL-ready datasets; strains encoded as perturbation operators over a shared
wildtype reference cell (genome sequence + gene/regulatory/PPI networks +
metabolism + environment); scale to millions of strains. -> Fig 1.

**R2. The Cell Graph Transformer: a graph-constrained virtual cell.** Multimodal
gene embeddings + CLS token through a transformer encoder; biological interaction
graphs regularize multi-head attention via an attention-prior alignment loss; an
equivariant perturbation operator maps the reference embedding to the perturbed
strain state; multitask heads decode phenotypes. -> Fig 2.

**R3. CGT predicts trigenic gene-gene interactions at state of the art.** The
headline result: r=0.454+/-0.004, rho=0.421+/-0.003 (SEM); outperforms DANGO, DCell,
and Yeast9 FBA (which recovers no epistatic signal). Open by noting classical-ML
baselines fail on this task (Suppl. Fig. S1), then show CGT closes the gap. Note
increased noise at the extremes of the interaction range (expected). -> Fig 3.

**R4. One embedding, many phenotypes: expression and morphology.** Same
architecture predicts KO expression (r=0.543) and single-KO morphology
(r=0.619), showing the latent cell embedding generalizes across qualitatively
different phenotype classes relevant to strain design. -> Fig 4.

**R5. CGT-guided strain design with an autonomous foundry.** Recommend gene
deletions for beta-carotene and betaxanthin in S. cerevisiae; unlike strain-
agnostic DBTL tools, TorchCell represents the strain itself and pairs with UIUC
iBioFoundry for iterative AI-guided engineering. -> Fig 5.

### Discussion (target ~500-650 words)

Strain-aware representation as the differentiator; biological grounding (attention
priors + metabolism) as the answer to "DL doesn't beat linear baselines";
limitations (single organism today, morphology data maturity, error-bar
provenance, FBA baseline scope); outlook to cross-organism foundation models
(genesis-proposal MEFM vision) without over-claiming beyond this paper's data.

### Methods / Online Methods (no word limit; goes at end)

Ontology + KG construction (BioCypher, schema config); dataset construction and
the perturbation-operator formalism; CGT architecture and the graph-regularized
attention loss (full math here, not in main text); equivariant perturbation
operator; multitask training, splits, metrics; baselines (DANGO, DCell, Yeast9
FBA) and exact comparison protocol; inference at scale (465M+ triple candidates);
strain-selection procedure; statistics and error-bar definitions (fix provenance).

### Section work requirements (effort to submission-ready)

Rough effort to take each section from its current state to submission-ready.
Effort: LOW (hours) / MED (a day or two of writing + existing assets) / HIGH
(gated on data not yet in-repo or on external/wet-lab work).

| Section | Current state | Effort | What's needed |
|---|---|---|---|
| Title | 3 candidates drafted | LOW | pick one with co-authors |
| Abstract | shortened SIMB in place (~150 w) | LOW | restore uncertainties after the error-bar fix |
| Introduction | 3 beats outlined | MED | ~500-650 w new prose; citations; classical-ML motivation (S1) |
| R1 Ontology/KG | assets exist | MED | prose + Fig 1 (HAVE, redraw) |
| R2 CGT architecture | model + mermaid exist | MED-HIGH | prose + Fig 2 schematics (all GEN) |
| R3 GGI SOTA | headline result exists | MED | prose; Fig 3a fix error bars; 3b-d generate |
| R4 expression + morphology | partial | HIGH | morphology result not grounded (data in dev); expression performance plot needed |
| R5 strain design | partial / external | HIGH | recs partly in iBioFoundry-AI; decide scope; validation maybe future |
| Discussion | outlined | MED | ~500-650 w new prose |
| Methods | pipeline exists | MED-HIGH | document pipeline + full CGT/attention-loss math + stats |
| Figs 1-5 | mixed | per figure plan | see per-panel HAVE/GEN tags above |
| Suppl. S1-S6 | mixed | MED | S1 assemble (HAVE); S2-S6 mostly GEN |

The schedule-driving items are the **HIGH** rows (R4 morphology, R5 reportability)
plus the Fig 3a error-bar **BLOCKER** -- these gate submission more than prose
does. Everything else is writing time against assets that already exist.

---

### Figure preparation guidelines (Nature Biotechnology)

Confirmed 2026-06-10 via nature.com figure/artwork guides + a real example
(Fig. 4, DreaMS, Nat Biotechnol, doi:10.1038/s41587-025-02663-3 -- a full-width
10-panel a-j figure with a two-column caption). Design figures to the **figure
box**, NOT to the manuscript page margins (submission templates are wide; the
published two-column page is dense/typeset -- different stages, same paper).

Dimensions (design to these exactly):

- Single-column width: **88 mm**.
- Double-column / full-width: **180 mm**.
- Max height: **170 mm** (leaves room for the caption below).
- Column gutter: **~4 mm** (= 180 - 2x88; why published two-column text looks
  "close"). Pixel-measured 2026-06-10 from the DreaMS page (doi:10.1038/s41587-025-02663-3,
  anchoring text width = 180 mm): single column **87.7 mm**, gutter **4.7 mm** --
  matches the spec. (Outer page side-margins were not reliably extractable from a
  screenshot -- the near-white background defeats page-edge detection; use the
  source PDF if exact outer margins are needed.) NOTE: sn-jnl's own `iicol` is
  160/76/8 (submission two-column), narrower than the published 180/88/4;
  `figure-proto.tex` overrides geometry to the published numbers for true-scale sizing.

Layout: pack panels to content -- real Nature figures are **asymmetric by design**
(DreaMS Fig. 4 mixes a wide 3-scatter row, a large square UMAP, a 2x5 bar
mini-grid, ROC curves); a rigid uniform grid is the wrong model. Bold lowercase
panel letters (a, b, c, ...) top-left; caption spans full width below.

Type + lines: sans-serif (Helvetica/Arial), ~5-7 pt at final size, consistent
across panels; line weights >= 0.25 pt.

Files: line art/graphs as vector (EPS preferred; PDF/AI fine); photos as TIFF/JPEG
>= 300 dpi; one file per figure at final print size.

Workflow: build in matplotlib/Illustrator/Inkscape exported to PDF/SVG at exactly
88 or 180 mm wide, then `\includegraphics[width=88mm]{...}`. Sizing to mm up front
keeps text in the 5-7 pt range. (`paper/nature-biotech/figure-proto.tex` is an
optional LaTeX check that a full-width figure fills a real page.)

### Figure plan -- main display items (4 panels per figure; bet + options each)

Budget: **5 main figures (Figs 1-5) + one reserve display-item slot** (6th, held
for a summary table or a promoted SI panel) -- <=6 total per nbt. Each figure
targets **4 panels (a-d)**. Per panel: **Bet** = the plot we lead with; **Options**
= swap-in alternatives to decide during figure-making. Status: `[HAVE]` asset
exists / `[GEN]` generate / `[PARTIAL]` partly in iBioFoundry-AI repo / `[DECIDE]`.

**Figure 1 - TorchCell: ontology-unified knowledge graph + perturbation-operator cell.** 4 panels. The "what is TorchCell" overview.

- 1a `[HAVE]` Data-to-KG flow. Bet: schematic literature/public data -> experiment
  ontology -> Neo4j KG at NCSA. Options: add biolink triple-layer inset
  (`notes/assets/images/biocypher.config...biolink-core-triple-layers-of-association.png`);
  or a KG node/edge-count bar. Source: `notes/torchcell.ontology.mermaid_diagram*.md`,
  `biocypher/config/torchcell_schema_config.yaml`.
- 1b `[HAVE]` Multimodal reference cell. Bet: stacked graph-layers cartoon (genome
  seq + gene/regulatory/PPI networks + metabolism + environment). Options: the
  9-graph-type panel from the model mermaid; or a single integrated cell-graph.
- 1c `[HAVE/GEN]` Perturbation operator. Bet: schematic compressing a full
  experiment into an operator over the reference cell. Options: dataset-scale inset
  (`notes/experiments.smf-dmf-tmf-001.dataset_size_histograms.md`); or a
  log-scale "millions of strains" count bar.
- 1d `[GEN]` CGT teaser. Bet: one-line pipeline (multimodal in -> cell latent ->
  equivariant perturbation -> multitask heads). Options: drop if crowded and
  promote a KG-stats panel; or move the teaser into Fig 2a.

**Figure 2 - The Cell Graph Transformer architecture.** 4 panels. The methods novelty made visual.

- 2a `[GEN]` Encoder + graph-regularized attention. Bet: schematic of how
  interaction graphs mask/bias multi-head attention. Options: side-by-side vanilla
  vs graph-constrained attention. Source:
  `torchcell/models/equivariant_cell_graph_transformer.py`,
  `notes/torchcell.models.equivariant_cell_graph_transformer.mermaid.md`.
- 2b `[GEN]` Attention-prior alignment. Bet: attention-vs-prior heatmap (weights
  aligning to known network edges). Options: alignment-loss curve over training;
  or attention-rollout on a small subgraph.
- 2c `[GEN]` Equivariant perturbation operator. Bet: reference embedding ->
  perturbed-state diagram. Options: contrast Type I/II transforms; or an
  equivariance-check panel (permuted inputs -> permuted outputs).
- 2d `[GEN]` Multitask decoder heads. Bet: head fan-out to GGI / expression /
  morphology. Options: fold into 2a and use 2d for an attention-prior on/off
  ablation bar instead (`[DECIDE]`).

**Figure 3 - State-of-the-art trigenic gene-gene interaction prediction.** 4 panels. The headline result. (Classical-ML baselines that fail on this task live in Suppl. Fig. S1, cited from R3.)

- 3a `[HAVE]` Model comparison (the DCell / Yeast9 / DANGO / CGT panel). Bet:
  grouped bar of trigenic tau, CGT vs DANGO vs DCell vs Yeast9 FBA, error bars.
  Options: add Spearman as a second grouped series; or a dumbbell vs FBA baseline.
  Asset: `notes/assets/images/010-kuzmin-tmi/trigenic_tau_model_comparison_2026-04-13-00-57-40.png`
  via `experiments/010-kuzmin-tmi/scripts/trigenic_tau_model_comparison.py`. FIX
  error-bar provenance first (see Gaps);
  [[experiments.010-kuzmin-tmi.scripts.trigenic_tau_model_comparison]].
- 3b `[GEN]` Predicted-vs-actual. Bet: hexbin/density of predicted vs measured
  trigenic tau with r, rho annotated. Options: 2D-histogram; or scatter colored by
  local density.
- 3c `[GEN]` Where it works/breaks. Bet: performance stratified by interaction
  magnitude (binned r, or error vs |tau|) -- substantiates "noise at extremes."
  Options: residual-vs-magnitude; or per-quantile calibration.
- 3d `[GEN/DECIDE]` What drives performance. Bet: ablation (modalities / graph
  types / attention-prior removed). Options: calibration curve; or CGT-vs-best-
  classical-ML scatter on the same test set (ties Fig 3 to Suppl. Fig. S1).

**Figure 4 - One latent cell embedding generalizes across phenotype classes.** 4 panels. The multimodal-generalization claim.

- 4a `[GEN]` Expression (single+double KO), r=0.543. Bet: predicted-vs-actual
  scatter. Options: single- vs double-KO split as two sub-panels; or per-gene r
  distribution. NOTE: today only expression *distribution*/cross-study plots exist
  (`notes/assets/images/012-sameith-kemmeren-expression/*`); the performance plot
  must be generated.
- 4b `[GEN, MAJOR]` Morphology (single KO), r=0.619. Bet: predicted-vs-actual.
  Options: per-trait r bar (CalMorph traits); or example morphology-trait panel.
  Dataset in development (`.claude/commands/morphology_dataset.md`, Ohya2005) --
  highest-risk panel; mark preliminary if not landed.
- 4c `[GEN]` Cross-phenotype summary. Bet: Pearson for GGI / expression /
  morphology side by side (the "same embedding, many phenotypes" money panel).
  Options: radar across phenotype classes; or grouped bar with baselines.
- 4d `[DECIDE]` Shared representation. Bet: UMAP of cell embeddings colored by
  phenotype/condition. Options: drop and reclaim the slot for a baseline-vs-CGT
  bar across all three phenotypes; or a transfer panel (train on one phenotype,
  probe another).

**Figure 5 - CGT-guided strain design closes the DBTL loop with an autonomous foundry.** 4 panels. The translational payoff (claims 8-9).

- 5a `[HAVE]` Pathways. Bet: beta-carotene + betaxanthin biosynthesis diagrams
  with CGT-flagged deletion targets highlighted. Options: single combined pathway;
  or pathway + flux-context inset. Source: `notes/metabolism.beta-carotene-and-betaxanthin.md`.
- 5b `[PARTIAL]` Recommendations. Bet: ranked CGT gene-deletion recommendations
  (lollipop of predicted improvement). Options: top-N table; or predicted-vs-FBA
  scatter. Lives partly in iBioFoundry-AI repo +
  `experiments/010-kuzmin-tmi/results/inference_3/singles_table_panel12_*.csv`;
  decide reportable-here vs cross-referenced.
- 5c `[GEN]` DBTL loop. Bet: TorchCell + UIUC iBioFoundry cycle schematic (design
  -> build -> test -> learn), stressing strain-aware recommendation. Options: add a
  "model improves after feedback" arrow/inset.
- 5d `[PARTIAL/FUTURE]` Validation. Bet: titer vs base strain (n>=3, significance)
  if data exist by submission. Options: otherwise a clearly-labeled proposed-
  validation panel, or move to Discussion. `[DECIDE]` with wet-lab co-authors.

**Reserve (6th display item).** `[DECIDE]` Hold the 6th slot for either a summary
**Table 1** (datasets/phenotypes/metrics at a glance) or a promoted panel; do not
fill unless it earns its place against the <=6 limit.

### Supplementary figure plan (up to 6; ideas-stage)

Loose for now. Each ~4 panels where it helps. **S1 is load-bearing** (cited in
Results R3 + Intro as the DL motivation); the rest hold overflow from the main 6.

**Suppl. Fig. S1 - Classical ML predicts fitness but fails on gene interactions (motivates DL).** `[HAVE]` Cited in main text.

- S1a Fitness is tractable: RF / SVR / elastic net on KO fitness across embeddings
  and dataset sizes 1e3-1e5 (decent r). Source: `experiments/smf-dmf-tmf-001/`,
  assets `notes/assets/images/smf-dmf-tmf-001_node_embedding_performance_*`,
  `[[experiments.smf-dmf-tmf-001.results]]`.
- S1b Interactions fail: same models on DMI/TMI gene interactions -- poor/variable,
  no clear winner. Source: `experiments/002-dmi-tmi/`, assets
  `notes/assets/images/002-dmi-tmi_*_node_embedding_performance_*`, `[[experiments.002-dmi-tmi]]`.
- S1c Scaling: performance vs dataset size / num params (classical ML does not
  recover GI even with more data). Asset: `002-dmi-tmi_*_vs_num_params.png`.
- S1d Bet vs option: direct best-classical-ML vs CGT on the same GI test set
  (sharpest contrast); option: per-embedding performance heatmap.

**Suppl. Fig. S2 - Data composition and scale.** `[HAVE/GEN]` (a) smf/dmf/tmf
counts; (b) gene-count bars (`notes/experiments.smf-dmf-tmf-001.dataset_genes_bar_plot.md`);
(c) performance-vs-training-size scaling curve `[GEN]`; (d) split sizes/coverage.

**Suppl. Fig. S3 - CGT architecture detail + ablations.** `[GEN]` (a) full
attention-prior loss schematic; (b) attention-prior on/off; (c) modality / graph-
type ablation; (d) training curves. (Overflow target if Fig 2/3d run short.)

**Suppl. Fig. S4 - Knowledge graph + ontology spec.** `[HAVE]` (a) full schema/
ontology diagram (`notes/torchcell.ontology.mermaid_diagram*.md`); (b) biolink
triple layers; (c) KG node/edge statistics; (d) example query -> dataset.

**Suppl. Fig. S5 - Expression-prediction diagnostics.** `[HAVE/GEN]` (a) Kemmeren/
Sameith expression distributions; (b) cross-study correlation (pearson/spearman
dists); (c) single- vs double-KO breakdown; (d) predicted-vs-actual. Source:
`notes/assets/images/012-sameith-kemmeren-expression/*`,
`[[experiments.012-sameith-kemmeren.scripts.gene_by_gene_expression_correlation]]`.

**Suppl. Fig. S6 - Inference at scale.** `[HAVE/PARTIAL]` (a) candidate funnel
(465M+ triples -> selected); (b) gene-stability/coverage; (c) score distributions;
(d) final 12-gene construction panel. Source:
`[[experiments.010-kuzmin-tmi.inference-dataset-3]]`,
`experiments/010-kuzmin-tmi/results/inference_3/`.

### Supplementary text / tables (uncapped)

Full CGT math + attention-prior derivation; hyperparameter tables; exact baseline
protocols (DANGO, DCell, Yeast9 FBA); statistics + error-bar definitions.

---

### Gaps / progress-driving checklist (what to generate before submission)

Ordered by risk. This is the actionable backlog this outline exists to drive.

1. `[BLOCKER]` **Error-bar provenance** (Fig 3a). Confirm whether CGT +/-0.006 is
   SE/SEM (comparable to DANGO/DCell SEM) or SD (~sqrt(3)x too large); regenerate
   so all error bars are the same statistic. See the 2026.06.04 caveat in
   [[conference.simb-2026.abstract]] and
   [[experiments.010-kuzmin-tmi.scripts.trigenic_tau_model_comparison]].
2. `[HIGH]` **Morphology result** (Fig 4b). Dataset in development; the r=0.619
   panel is not yet reproducible in-repo. Either land the morphology training run
   or stage the panel as clearly preliminary.
3. `[HIGH]` **Expression performance plot** (Fig 4a). Generate predicted-vs-actual
   for the r=0.543 model (only distribution plots exist now).
4. `[MED]` **GGI scatter + magnitude-stratified performance** (Fig 3b-c) -- new plots.
5. `[MED]` **CGT + attention-prior schematics** (Fig 2) -- publication-grade redraws
   from the mermaid sources.
6. `[MED]` **Strain-design reportability** (Fig 5b/d): decide what crosses over
   from iBioFoundry-AI and whether validation titers exist by submission.
7. `[LOW]` **Scaling curve** (Suppl. Fig. S2c) if we want a data-scaling argument.
8. `[LOW]` **Classical-ML vs CGT contrast panel** (Suppl. Fig. S1d / Fig 3d option):
   re-run best classical baselines on the exact GGI test set for a direct scatter.

### Open questions for co-authors / next revision

- Lead framing: multimodal-representation-first vs epistasis-SOTA-first?
- Is morphology in-scope for v1, or held for a follow-up (de-risks Fig 4b)?
- How much of the beta-carotene/betaxanthin + iBioFoundry validation is reportable
  now vs "proposed validation"? This decides whether Fig 5 is results or outlook.
- Author list, contributions, and which results are co-owned with LBNL/ANL/iBioFoundry.
- Fold in the ChatGPT (#1) and Perplexity (#2) conversations (not yet provided) on
  the next pass -- likely refine Intro positioning and the related-work contrast.

### LaTeX template + Overleaf + Tectonic (the port)

The Overleaf-ready skeleton lives in this worktree at **`paper/nature-biotech/`**
(`main.tex`, `references.bib`, `README.md`, plus the official Springer Nature class
`sn-jnl.cls` + `sn-nature.bst`, and the upstream `sn-article.tex` manual). It uses
**`\documentclass[pdflatex,sn-nature,Numbered]{sn-jnl}`** -- `sn-jnl` is the single
Springer Nature class that covers all Nature Portfolio journals (incl. Nature
Biotechnology); `sn-nature` is the Nature reference style; `Numbered` gives Nature's
superscript numbered citations; the `pdflatex` option makes it compile under
pdflatex **and xelatex** engines. (Confirmed 2026-06-09 via nature.com/nbt; class
v3.1, Dec 2024.) Full how-to in `paper/nature-biotech/README.md`.

**Section -> TeX mapping** (already stubbed in `main.tex`):

- Title/authors -> `\title[short]{...}`, `\author*[1]{\fnm{}\sur{}}\email{}`, `\affil`.
- Abstract -> `\abstract{...}` (<=150 words, unreferenced).
- Introduction -> `\section{Introduction}`.
- Results R1-R5 -> `\section{Results}` with `\bmhead{...}` run-in subheads matching the R-titles.
- Discussion -> `\section{Discussion}`; Methods -> `\section{Methods}` (full math).
- Figures -> `figure` env, `figures/*.pdf` (export vectors from `notes/assets/images/...`).
- SI -> `\bmhead{Supplementary information}` + separate SI file (S1-S6).

**Getting it into Overleaf** (three paths, detailed in the README): (A) open the
official gallery template and paste our `main.tex`/`references.bib`; (B) zip
`paper/nature-biotech/` and upload; (C) git-sync (Overleaf Pro git remote, or
GitHub sync) so the repo and Overleaf stay linked.

**Tectonic (local builds):** expected to work, **not yet verified on this machine**
(tectonic is not installed here, so the skeleton has not been test-compiled). The
reasoning: Tectonic is a XeTeX engine, and our `pdflatex` documentclass option is
exactly the sn-jnl switch that enables xelatex-compatible compilation; Tectonic
auto-fetches the packages `sn-jnl.cls` pulls in and runs classic BibTeX for
`sn-nature.bst`. To install + verify: `conda install -c conda-forge tectonic` (or
`cargo install tectonic`), then `tectonic -X compile main.tex` (V2 CLI). The VS
Code "Overleaf Workshop" extension still edits/compiles against the Overleaf
project; Tectonic is the offline/CI alternative. Known gotchas to watch on first
build: a class option that hard-requires pdfTeX-only primitives, or BibTeX vs
biber mismatch -- neither is expected with sn-nature, but confirm on the real run.
