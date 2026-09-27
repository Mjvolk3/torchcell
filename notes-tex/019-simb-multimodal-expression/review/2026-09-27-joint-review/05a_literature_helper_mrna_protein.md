# Literature helper report (condensed by the coordinator, 2026-09-27 ~02:10)

Source: the "Survey yeast mRNA-protein literature" helper spawned by reviewer 5; full report
in its transcript. Mirror paths relative to `$DATA_ROOT/torchcell-library/`.

## Steady state, across genes (the strong correlation)
- Ghaemmaghami 2003: Spearman 0.57 all genes, 0.66 on 206 re-measured essentials; mean 4,800 protein per mRNA.
- Greenbaum 2003: Pearson 0.66 (n = 2,044). Lu/Marcotte 2007: 73% variance explained in yeast.
- Csardi 2015 (38 mRNA + 20 protein datasets): naive median r 0.54; noise-corrected mRNA explains > 85% of steady-state protein variance; log-log slope 1.69 (amplification).
- Vogel and Marcotte 2012's "~40%" is cross-organism, not yeast-specific.

## Across genotypes or conditions, within gene (the weak correlation, the one that matters here)
- Teyssonniere 2024 PNAS (942 isolates, 630 proteins): median within-gene Spearman 0.165 (across-gene 0.53); protein fold-changes 32% smaller than mRNA; SNP-pQTL vs SNP-eQTL overlap 3%.
- Albert 2014/2018 (biparental cross, high power): >= 53% of distant eQTL have a matching pQTL, 92% directional agreement; so the low overlap is partly a power artifact.
- Skelly 2013: median Spearman 0.33 across 22 strains and segregants.
- Lee 2011 (NaCl stress): R2 0.77 for genes whose mRNA INCREASES, 0.09 for genes whose mRNA DECREASES. Direction matters more than gene class; deletions are mostly decreases.
- McManus 2014 / Artieri 2014: translational buffering 5.5x more common than amplification.
- Taniguchi 2010 (E. coli): across genes r 0.77, within a single cell same gene r 0.01.

## Dosage compensation
- Muenzner 2024 (mirror): 70.5% of proteins on aneuploid chromosomes compensated at protein level; attenuation slope protein 0.65 vs mRNA 0.92; 63% of NON-complex proteins also attenuated; turnover rate predicts compensation.
- Messner 2023 (mirror, line 135): in BY4741 aneuploidies pass to proteome with "a minimum amount of gene-dosage buffering", i.e. the deletion collection background is the LOW-compensation case.

## Gene classes that decouple
- Ribosomal proteins and complex subunits (surplus degraded); Messner KO: 22% of 51 complexes show decrease of other subunits on deleting one, 18% increase.
- Long-half-life proteins more often differentially expressed and decreased (Messner line 163).
- The deleteome's dominant axis is the slow-growth/ESR signature (O'Duibhir 2014: r 0.73 to 0.93 with heat shock; ~900 genes); PC1 of the Kemmeren log2FC matrix = 22.6% of variance.

## Messner vs Kemmeren per deletion: NEVER PUBLISHED
- Messner cites Kemmeren but reports no per-deletion cross-omic correlation. Closest: Ozturk 2022 (S. pombe, 94 paired KO strains): strain-level R 0.51, per-gene R from -0.84 to 0.98, 750 of 1,706 genes significantly positive. Despres 2022: R2 0.62 but on 11 reporter proteins restricted to effects significant in both.
- Our own number (2-readouts.tex, sec. proteome): per-deletion median 0.036, per-protein 0.075, ridge either way 2 to 3.5% of held-out variance; gene-pair co-variation Spearman 0.31.

## Noise ceilings
- Messner: published CVs 8.1% technical, 11.3% WT biological, 16.2% KO; helper recomputed from the released matrix 7.9 / 11.0 / 15.5%. Derived per-protein reliability ceiling (helper's calculation, CV-based): median achievable Pearson 0.67, IQR 0.50 to 0.79, 24% of proteins below 0.5; technical-only ceiling 0.82. Script `scratchpad/ceiling2.py`.
- Kemmeren: 0.59% of (gene, mutant) cells robustly changed; median 4 significant genes per mutant; per-gene sd of log2FC 0.124 vs calling threshold 0.766; no per-gene reliability derivable from released M + p (helper tried two estimators, irreconcilable). Sparse-signal, noise-dominated at the cell level with a large shared growth axis.

## Implications the helper drew
1. Csardi's > 85% (across genes, steady state) and Teyssonniere's 0.165 (within gene, across genotypes) answer different questions; our target is the second.
2. The direction asymmetry (Lee 2011) maps onto deletions: decreases are where mRNA predicts protein worst.
3. Any joint-training synergy must come from shared gene-module structure (co-variation 0.31) and shared genotype effects on the growth axis, not from per-deletion agreement (0.04).
