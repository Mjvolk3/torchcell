---
id: 2zjrj4zhq9q3v5gv9gxhbru
title: Dataset_043
desc: ''
updated: 1792450013817
created: 1792450013817
---

## 2026.10.19 - The torch Dataset over the raw store

`Raw043Dataset` serves one sample per raw record of the chosen datasets, addressed through
the head-only raw index `raw_043/raw/raw_dataset_name_index.json` that
[[experiments.043-expression-proteome-morphology.scripts.data_shapes_043]] writes (the
processed store's `dataset_name_index.json` is in processed order, which is not raw order:
processed index 82,729, listed under Caudal, is raw `data_82729`, a Boocock record). The
module docstring holds the sample schema and the marking rules; the diagram is
[[experiments.043-expression-proteome-morphology.mermaid.data-flow]].

What the config decides (`Dataset043Config`, pydantic):

- `datasets`: default `TRAINING_DATASETS`, the 13 gene-level readouts (owner, 2026.10.19:
  "include ALL gene-level readouts in training"). `SmMicroarraySameith2015Dataset` is
  listed under `DUPLICATE_DATASETS` (82 of 82 vectors identical to Kemmeren 2014) and
  `ScmdOhya2005Dataset` under `HOLDOUT_DATASETS` ("morphology is just a hold out for
  evals"); both can be added to `datasets`.
- `segregant_posterior_min` (0.9), `isolate_marking` (`all_gene_level` marks sequence
  variants as VARIANT, LOF as LOF, natural absences as ABSENT, the HO deletion as DELETION;
  `lof_absent_deletion` leaves the variants unmarked), `include_reference` (returns the
  record's `experiment_reference` label as `y_ref`), `label_keys` (a different phenotype
  field per dataset, e.g. `half_life` for the turnover set), `segregant_parent_vocab`
  (the eight parent names the two panels carry; an unknown name raises).
- Every `y` is the released value; `y_scale` names the scale. Conversion (log of a linear
  abundance, ratio to the reference, centering) is model-side.

Three definite choices the owner should read before training:

1. **Segregant reference parent.** The parent with `peter_strain_id` None (BY) is the
   reference; for Boocock crosses `3004` (CBS2888 x YJM981) and `B` (YJM145 x YPS163)
   neither parent is BY, so parent 1 stands as the background and parent-2 genes are
   marked. `allele_parent` carries the parent token per marked gene, so the model can
   still tell the two apart.
2. **Gene position inside blocks.** A gene is placed by its midpoint (SGD R64-4-1 GFF
   coordinates) in the block whose first and last marker enclose it; the block positions
   are the released marker positions on the S288C reference, taken as the same coordinate
   system (R64-2-1 and R64-4-1 share chromosome coordinates; not re-verified here). A
   gene in the gap between two blocks is `unassigned` and not marked. That gap count is
   large: 2,201 to 2,693 of 6,607 genes on the three Boocock records and 697 to 1,002 on
   the three N'Guessan records below. Hypothesis (untested): Boocock's LD-pruned marker
   set leaves longer unassigned runs between called blocks. A probabilistic or
   interpolated placement is the later step the owner named.
3. **Isolate marking.** Under `all_gene_level` a Caudal isolate marks about 5,000 to 6,000
   of 6,607 genes (sequence variants dominate); under `lof_absent_deletion` the same record
   marks 499. The sequence-variant mark says "not the reference allele" with no magnitude;
   the model has to learn what that means from `pert_type`.

Run output (login node, 45 samples = 3 records of each of the 15 datasets, 9.6 s wall,
`include_reference=True`):

```text
45 samples over 15 datasets; gene index 6607; isolate marking all_gene_level; segregant posterior >= 0.9
MicroarrayKemmeren2014Dataset data_0 [Genotype] exp_type MICROARRAY_EXPRESSION y_scale LOG2_RATIO env 042e9aca39c0 | gene_index (1,) {'DELETION': 1} outside index 0 accessory 0 | y (6607,) measured 6127 of 6169 keys (outside index 42)
DmMicroarraySameith2015Dataset data_1564 [Genotype] exp_type MICROARRAY_EXPRESSION y_scale LOG2_RATIO env 042e9aca39c0 | gene_index (2,) {'DELETION': 2} outside index 0 accessory 0 | y (6607,) measured 6127 of 6169 keys (outside index 42)
MicroarrayScoreHu2007Dataset data_1636 [Genotype] exp_type MICROARRAY_EXPRESSION_SCORE y_scale SCORE env a805d6f93770 | gene_index (1,) {'DELETION': 1} outside index 0 accessory 0 | y (6607,) measured 5726 of 6273 keys (outside index 547)
TfDeletionPseudobulkJackson2020Dataset data_1898 [Genotype] exp_type SINGLE_CELL_EXPRESSION y_scale UMI_COUNT env 57a078e53dee | gene_index (1,) {'DELETION': 1} outside index 0 accessory 0 | y (6607,) measured 3006 of 3021 keys (outside index 15)
NadalRibellesPerturbSeq2025Dataset data_2682 [Genotype] exp_type PSEUDOBULK_EXPRESSION y_scale LOG2_RATIO env 5f3603465471 | gene_index (1,) {'DELETION': 1} outside index 0 accessory 0 | y (6607,) measured 5482 of 5833 keys (outside index 351)
CaudalPanTranscriptome2024Dataset data_18199 [Genotype+natural_isolate] exp_type RNASEQ_EXPRESSION y_scale TPM env c7d8fd2babbb | gene_index (5995,) {'VARIANT': 5496, 'LOF': 390, 'ABSENT': 109} outside index 9 accessory 91 | y (6607,) measured 6014 of 6063 keys (outside index 49)
Boocock2025Dataset data_19992 [SegregantGenotype] exp_type SINGLE_CELL_EXPRESSION y_scale UMI_COUNT env f4497b2aeb0e | gene_index (1749,) {'VARIANT': 1749} outside index 0 accessory 0 | y (6607,) measured 926 of 926 keys (outside index 0) | segregant B ('YJM145', 'YPS163') ref parent 1: marked 1749, ref allele 2001, low posterior 480, unassigned 2377, blocks 736
Boocock2025Dataset data_19993 [SegregantGenotype] exp_type SINGLE_CELL_EXPRESSION y_scale UMI_COUNT env f4497b2aeb0e | gene_index (1415,) {'VARIANT': 1415} outside index 0 accessory 0 | y (6607,) measured 990 of 990 keys (outside index 0) | segregant B ('YJM145', 'YPS163') ref parent 1: marked 1415, ref allele 1795, low posterior 704, unassigned 2693, blocks 793
Boocock2025Dataset data_19994 [SegregantGenotype] exp_type SINGLE_CELL_EXPRESSION y_scale UMI_COUNT env f4497b2aeb0e | gene_index (2320,) {'VARIANT': 2320} outside index 0 accessory 0 | y (6607,) measured 1174 of 1174 keys (outside index 0) | segregant B ('YJM145', 'YPS163') ref parent 1: marked 2320, ref allele 1913, low posterior 173, unassigned 2201, blocks 712
Nguessan2025Dataset data_99508 [SegregantGenotype] exp_type SINGLE_CELL_EXPRESSION y_scale UMI_COUNT env bbf7b9f1e80e | gene_index (2743,) {'VARIANT': 2743} outside index 0 accessory 0 | y (6607,) measured 2517 of 2517 keys (outside index 0) | segregant BBQTL_BYxRM ('BY4741', 'RM11-1a') ref parent 1: marked 2743, ref allele 2245, low posterior 617, unassigned 1002, blocks 1056
Nguessan2025Dataset data_99509 [SegregantGenotype] exp_type SINGLE_CELL_EXPRESSION y_scale UMI_COUNT env bbf7b9f1e80e | gene_index (2917,) {'VARIANT': 2917} outside index 0 accessory 0 | y (6607,) measured 1797 of 1797 keys (outside index 0) | segregant BBQTL_BYxRM ('BY4741', 'RM11-1a') ref parent 1: marked 2917, ref allele 2836, low posterior 157, unassigned 697, blocks 655
Nguessan2025Dataset data_99510 [SegregantGenotype] exp_type SINGLE_CELL_EXPRESSION y_scale UMI_COUNT env bbf7b9f1e80e | gene_index (3158,) {'VARIANT': 3158} outside index 0 accessory 0 | y (6607,) measured 1683 of 1683 keys (outside index 0) | segregant BBQTL_BYxRM ('BY4741', 'RM11-1a') ref parent 1: marked 3158, ref allele 2571, low posterior 160, unassigned 718, blocks 653
PooledPromoterExpressionVaishnav2022Dataset data_102459 [Genotype] exp_type POOLED_SORT_SEQ_EXPRESSION y_scale SORT_BIN_INDEX env 61d90319663a | gene_index (0,) {} outside index 0 accessory 0 | y (6607,) measured 0 of 1 keys (outside index 0) | promoter 110 nt, scaffold pTpA, variable 18-97, y_scalar 10.437
ProteomeMessner2023Dataset data_8800 [Genotype] exp_type PROTEIN_ABUNDANCE y_scale LINEAR_ABUNDANCE env 8f819c781251 | gene_index (1,) {'DELETION': 1} outside index 0 accessory 0 | y (6607,) measured 1821 of 1821 keys (outside index 0)
ProteomeZelezniak2018Dataset data_13424 [Genotype] exp_type PROTEIN_ABUNDANCE y_scale LOG_ABUNDANCE env ed4dc0d4d175 | gene_index (1,) {'DELETION': 1} outside index 0 accessory 0 | y (6607,) measured 726 of 726 keys (outside index 0)
ProteomeMuenzner2024Dataset data_19142 [Genotype+natural_isolate] exp_type PROTEIN_ABUNDANCE y_scale LINEAR_ABUNDANCE env fbf37e55f920 | gene_index (5330,) {'VARIANT': 4861, 'LOF': 349, 'ABSENT': 120} outside index 7 accessory 95 | y (6607,) measured 1499 of 1499 keys (outside index 0)
ProteinTurnoverMuenzner2024Dataset data_19937 [Genotype+natural_isolate] exp_type PROTEIN_TURNOVER y_scale RATE_PER_HOUR env ab85e0823b60 | gene_index (5901,) {'VARIANT': 5430, 'LOF': 337, 'ABSENT': 134} outside index 5 accessory 88 | y (6607,) measured 1006 of 1006 keys (outside index 0)
ScmdOhya2005Dataset data_13521 [Genotype] exp_type CALMORPH y_scale CALMORPH env a9376e030b59 | gene_index (1,) {'DELETION': 1} outside index 0 accessory 0 | y (6607,) measured 0 of 281 keys (outside index 0) | y_global (281,) measured 281
SmMicroarraySameith2015Dataset data_1482 [Genotype] exp_type MICROARRAY_EXPRESSION y_scale LOG2_RATIO env 042e9aca39c0 | gene_index (1,) {'DELETION': 1} outside index 0 accessory 0 | y (6607,) measured 6127 of 6169 keys (outside index 42)

batch:
  gene_index     torch.int64    (21728,)
  gene_batch     torch.int64    (21728,)
  pert_type      torch.int64    (21728,)
  allele_parent  torch.int64    (21728,)
  n_marked       torch.int64    (15,)
  promoter_seq   list[15] e.g. 'None'
  scaffold       list[15] e.g. 'None'
  env_id         torch.int64    (15,)
  dataset_id     torch.int64    (15,)
  exp_type_id    torch.int64    (15,)
  y_scale        torch.int64    (15,)
  raw_index      torch.int64    (15,)
  y              torch.float32  (15, 6607)
  y_mask         torch.bool     (15, 6607)
  y_scalar       torch.float32  (15,)
  y_global       torch.float32  (15, 281)
  strain         list[15] e.g. 'None'
  y_ref          torch.float32  (15, 6607)
  y_ref_scalar   torch.float32  (15,)
  y measured per row: [6127, 6127, 5726, 3006, 5482, 6014, 926, 2517, 0, 1821, 726, 1499, 1006, 0, 6127]; marked genes per row: [1, 2, 1, 1, 1, 5995, 1749, 2743, 0, 1, 1, 5330, 5901, 1, 1]

Caudal under lof_absent_deletion: CaudalPanTranscriptome2024Dataset data_18199 [Genotype+natural_isolate] exp_type RNASEQ_EXPRESSION y_scale TPM env c7d8fd2babbb | gene_index (499,) {'LOF': 390, 'ABSENT': 109} outside index 5 accessory 91 | y (6607,) measured 6014 of 6063 keys (outside index 49)
```

(The second and third record of the single-deletion datasets repeat the first line's
shape and are left out above; the full 45 lines print on every run.)

Debugger entry points are the three `# BREAKPOINT:` comments in `__main__`: the config,
the per-sample dict, the mixed batch through `collate_043`, and the alternative isolate
rule on the same Caudal record.
