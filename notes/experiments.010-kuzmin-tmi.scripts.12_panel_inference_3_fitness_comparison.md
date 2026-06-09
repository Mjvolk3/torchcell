---
id: jauqyg06vf9pxfqufw6j9qk
title: 12_panel_inference_3_fitness_comparison
desc: ''
updated: 1781028958386
created: 1781028958386
---

## 2026.06.09 - Comparing Multi-Source Fitness and Reconstructing Triple Fitness Trajectories for the Inference_3 Panel

This script assesses whether the inference_3 12-gene panel's experimental fitness measurements are trustworthy and whether its predicted triples trace biologically interesting fitness paths. It quantifies how much different data sources disagree relative to their own measurement noise (between-source spread vs within-source std) and reconstructs each triple's WT to single to double to triple fitness trajectory from the model's predicted interaction, surfacing triples with a monotonically increasing path. It exists to vet measurement reliability and to identify "hero" triples worth the validation experiment.

### Specifics worth keeping

- Inputs: the queried singles/doubles CSVs from the queried-data script plus `triples_table_panel12_k200.csv`.
- Triple fitness uses f_ijk = tau_ijk + f_ij*f_k + f_ik*f_j + f_jk*f_i - 2*f_i*f_j*f_k, with SMF/DMF taken from the lowest-std source per gene/pair.
- Produces forest plots, Gaussian overlays, a between-source-spread summary CSV (`fitness_comparison_summary.csv`), gene/doubles summary tables, a best-path trajectory plot, and hero-triple plots showing all 6 mutation orderings.
- Figures saved to `ASSET_IMAGES_DIR/010-kuzmin-tmi/`; the monotonic-path flag connects directly to the Jonckheere-Terpstra validation rationale.
