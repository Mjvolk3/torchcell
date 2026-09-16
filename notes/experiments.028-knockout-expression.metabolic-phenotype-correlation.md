---
id: ks9y4ngnfa2ngjt0pvspg9y
title: Metabolic Phenotype Correlation
desc: ''
updated: 1789593926987
created: 1789593926987
---

## 2026.09.16 - Working note for the typeset document

Typeset document: `notes-tex/028-knockout-expression-metabolic/` (build with `make`, gate with `make check`, figure refresh with `make plots`). Script: [[experiments.028-knockout-expression.scripts.expression_metabolic_yko_correlation]]. The script writes the two tables the document inputs (`tables/reads_summary.tex`, `tables/amino_acids.tex`) and a stable-named SVG for `make plots`.

The question: does a deletion's gene-level readout (Kemmeren mRNA, Nadal-Ribelles perturb-seq mRNA, Messner protein) carry its metabolic phenotype (Mulleder and Cooper amino acids, Cachera betaxanthin, Ozaydin beta-carotene)? Three reads per (source, panel) on the shared deletions: single-gene maximum against its permutation null, ridge out-of-fold Pearson against label permutations, and a Mantel read of strain-pair profile similarity against strain-pair metabolite distance.

The document's second purpose is quality control by an independent route. The three sources barely agree with each other on the same deletions (Messner vs Kemmeren per-deletion median r 0.04; Messner vs Nadal 0.01; Nadal does not replicate across its own batches). A source that predicts the metabolic phenotype carries real signal whether or not it agrees with the others; one that predicts nothing has no such defense.

Numbers, sections, and the Nadal-Ribelles read are in the document and its results JSON; this note tracks decisions:

- Zelezniak 2018 and da Silveira 2014 share under 125 deletions with any source and are kept in the summary table as null reads, not in the figure.
- Messner proteins measured in fewer than 95% of the shared deletions are dropped, the rest mean-imputed, because the ridge needs a dense matrix.
- The single-gene maximum is reported but not relied on: its permutation null widens under heavy-tailed metabolite values (Kemmeren nulls 0.11 to 0.30).
- Caveat carried in the document: Mulleder and Messner are from one laboratory on one collection; a plate-aware permutation has not been run.
