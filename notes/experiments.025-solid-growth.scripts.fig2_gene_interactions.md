---
id: taow1y1tg0it8pdetsea2i1
title: Fig2_gene_interactions
desc: ''
updated: 1791912126236
created: 1791912126236
---

## 2026.10.13 - Figure 2 as one twelve-panel sheet in the gate style

Replaces the draw.io composition for review: no frames, every letter followed by the section mark (red cross, nothing reviewed yet) and the figure mark (full circle, every number on file). Primaries only: gray no graphs or a no-interaction null, red the cell graph transformer with its soft prior, orange the hard mask, purple degree-matched random graphs, blue the learned null, yellow other arms. Panels and sources: a definitions; b measured trigenic interaction by bin of the CGT M03 prediction on the 37,673 test triples (r 0.455; `cgt_predictions_M03_c7671wgj_test.npy`); c the state of the art with the SI's current numbers, FBA 0.0019 (the 0.0006 once printed does not reproduce), DCell 0.157 from three checkpoints of one run, DANGO 0.366 over five v9.1 runs, CGT 0.454 over three replicates, and the no-interaction ridge 0.400 as a line; d the additive null on both splits; e the four ways the graphs enter; f the training pool; g the weight ladder at epoch 29; h biological minus random by seed; i validation curves; j edge recall at degree; k the 60-epoch budget; l the composite embedding with and without the prior. Every number is read from a committed result file named in the script's docstring.

![](assets/images/025-solid-growth/fig2_gene_interactions.svg)

### Row 4, the wild-type genes (added the same day)

Panels m to p read `results/wt_gene_embeddings/` from [[experiments.025-solid-growth.scripts.wt_gene_embedding_clusters]] (S3 closure checkpoint 8fuwsk48, epoch 43, seed 2, fetched from mmli; load check re-scores the validation triples at 0.5007 against the logged 0.5009). The encoder's wild-type output splits into 12 Leiden clusters, 9 GO-enriched at q < 0.05 (ribonucleoprotein complex q 7e-106, Golgi vesicle transport q 2e-32, mitotic cell cycle q 9e-18); the long arc of the map is the 2,255 genes that appear in no training triple, and the three unenriched clusters all sit on it. The raw learnable table clusters into 7 with 6 enriched and agrees only weakly with the encoder (adjusted Rand index 0.14). The two 4.7 MB embedding matrices are not committed; the script regenerates them from the checkpoint under slurm.

### Review round, the same day

Panel a is the hand-drawn explainer again, exported from the draw.io figure by `notes/assets/drawio/Fig2-gene-interactions.explainer.py` (fifty-three cells of the first slot, six times true size, headless). Panel b took the DCell look back: filled boxes in the red fill, orange whiskers, bin edges from minus infinity to infinity. Panel j is red throughout, with the no-penalty point gray. Row 4 now reads m encoder map, n training coverage, o terms, p table, so the two maps sit side by side; the gray arc of n is the 2,255 genes that appear in no training triple, which is what the reviewer noticed first. Spacing tightened, the sheet is 190 mm tall.
