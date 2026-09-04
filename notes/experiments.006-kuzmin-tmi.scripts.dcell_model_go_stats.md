---
id: cui93zh1fzmdrp8as3w7msg
title: Dcell_model_go_stats
desc: ''
updated: 1788478115147
created: 1788478115147
---

## 2026.09.03 - Measured GO DAG of the DCell baseline

Source for Supplementary Note `note:dcell-model` (`paper/nature-biotech/sections/si-note-dcell-model.tex`), `tab-dcell-model-go-filter.tex`, and panels b-d of `FigS-dcell-model`. The script rebuilds the filtered GO DAG exactly as `experiments/006-kuzmin-tmi/scripts/dcell.py` does for the trigenic DCell run (no `go_date_filter`, `go_min_genes` default 4, `subsystem_output_min` 20, `subsystem_output_max_mult` 0.3), from the cached `SCerevisiaeGraph.G_go` (`$DATA_ROOT/data/sgd/genome/graph/G_go.pkl`, built from SGD `go_details` over the 6,607-gene reference and `$DATA_ROOT/data/go/go.obo`, release 2025-03-16). Filter order: `filter_go_IGI` -> `filter_redundant_terms` -> `filter_by_contained_genes(n=4)`. Strata come from `torchcell.data.cell_data.compute_strata`, the same function the cell graph uses.

Frozen outputs under `experiments/006-kuzmin-tmi/results/dcell_model/`: `go_filter_stages.csv`, `go_terms_final.csv`, `go_strata.csv`, `go_genes_final.csv`, `go_evidence_codes.csv`, `dcell_model_size.csv`, and the wandb freeze `dcell_wandb_model_size.csv` (`model/*` summary fields of every run in `zhao-group/torchcell_006-kuzmin-tmi_dcell` and `zhao-group/torchcell_005-kuzmin2018-tmi_dcell`). `--from-csv` re-renders panels and table without touching the graph or wandb.

Measured (run 2026.09.03 on the M1, `go_filter_stages.csv` and `dcell_model_size.csv`):

| stage | terms | edges | annotations | genes covered | leaves |
|---|---|---|---|---|---|
| raw (three namespaces under GO:ROOT) | 5,660 | 6,674 | 66,404 | 6,607 | 3,826 |
| drop IGI annotations (119 terms emptied) | 5,541 | 6,540 | 64,483 | 6,607 | 3,763 |
| drop redundant terms (106) | 5,435 | 6,433 | 64,324 | 6,607 | 3,763 |
| contained genes >= 4 (2,780 terms dropped; the run) | 2,655 | 3,208 | 59,986 | 6,607 | 1,527 |
| reference only: date <= 2017-07-19, then the same filters | 1,408 | 1,714 | 22,978 | 6,409 | 911 |

- Final DAG: 2,655 subsystems, 3,208 edges, 13 strata (stratum 0 = GO:ROOT, 1 = the three namespace roots, 2 = 630 terms, ..., 12 = 2 terms), 1,527 leaves, 59,986 annotations (the `59986 rows` the model docstring mentions), annotation dates 2000-12-07 to 2023-10-04.
- Evidence codes retained: IEA 26,609; IBA 8,434; IDA 7,491; IMP 5,939; ND 5,487; HDA 4,108; IPI 711; RCA 517; IC 207; ISS 151; TAS 124; HMP 63; ISA 60; NAS 37; ISM 21; IEP 13; HGI 13; ISO 1. All 5,487 ND annotations sit on the three namespace roots; 675 genes are in the DAG only through them. Every gene is in at least 2 subsystems (median 8, max 38).
- Widths: direct-annotation median 6 (contained median 10), max 2,444 direct genes on `molecular_function` (width 734); width floor 20; 61,429 hidden units in total (paper: 97,181 over 2,526 subsystems).
- Parameters implied by the DAG with the `DCell` sizing rules: 20,548,953 (subsystems) + 64,084 (heads) = 20,613,037. This equals `model/params_total` logged by all 15 DCell runs that reached model construction (e.g. `dttu9dx2`, `c7248f86` in 006; `4ipeq1qh` in 005), so the rebuilt DAG is the trained one.
- Deviations from Ma et al. 2018 recorded in the note: IGI removal acts on annotations not terms; redundancy is judged against a parent's gene set; containment threshold 4 over the whole reference (paper: 6 over disrupted genes); no date cutoff; width from direct annotations rather than contained genes; BatchNorm before tanh; auxiliary losses averaged rather than summed; AdamW weight decay instead of a cross-validated L2 penalty.

![](./assets/images/006-kuzmin-tmi/dcell_model_terms_per_stratum.svg)

![](./assets/images/006-kuzmin-tmi/dcell_model_genes_per_term.svg)

![](./assets/images/006-kuzmin-tmi/dcell_model_terms_per_gene.svg)
