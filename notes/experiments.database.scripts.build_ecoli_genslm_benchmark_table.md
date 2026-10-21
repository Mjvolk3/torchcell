---
id: vsycanii0v3mvpu8lg6tg99
title: Build_ecoli_genslm_benchmark_table
desc: ''
updated: 1792613942263
created: 1792613942263
---

## 2026.10.21 - The E. coli datasets table and the gene-encoding benchmark short list

`experiments/database/scripts/build_ecoli_genslm_benchmark_table.py` is the generator behind
`notes-tex/database/ecoli-datasets/` ([[experiments.database.ecoli-datasets]]). It reads the
newest E. coli pre-build snapshot that `build_bacteria_datasets_table.py --organisms ecoli`
writes ([[experiments.database.scripts.build_bacteria_datasets_table]]) and emits four `.tex`
files under that document's `tables/`:

- `datasets.tex`: every served E. coli dataset as a landscape `longtable`, grouped by phenotype
  family, with the probe's sequence-only fit tier and label unit beside the yeast-table columns.
- `tiers.tex`: datasets and instances per sequence-only fit tier.
- `shortlist.tex`: the benchmark short list, with the measured columns.
- `numbers.tex`: a `\newcommand` per number the prose quotes, so a figure in a sentence is as
  generated as a table cell.

It also writes `experiments/database/results/ecoli_genslm_benchmark_shortlist.json` and the
generated block of the paired note.

**What is hand-authored.** Only the `PICKS` registry: which datasets, their tier (core or
second), the task each poses, what one label is, the held-out unit, the served datasets that
pose the same task, and the reason. Everything numeric comes from a store.

**What is measured, per shortlisted store, by a full scan of its LMDB.** *Genes* = distinct
gene ids carrying a label (the keys of a node-level label dict, or the perturbations'
`systematic_gene_name` for a strain-level or pair label); *Labels* = gene-attached numeric or
boolean values; *Env* = distinct stored environments (the interned environment's name);
*Gene identifier* = the `gene_namespace` of the perturbations, with the released
`identifier_kind` when the leaf is a released identifier, or the dominant key class of a
readout dict (b-number, BW25113 locus tag, ECK accession, gene symbol). The LMDB fingerprint
each measurement was read from is in the JSON; `--reuse-measurements` renders from it without
rescanning.

**The selection rule the short list follows.** The label must sit on one gene or one gene pair
(sequence-only fit *best*, or *good* only through a gene x condition cross); enough genes must
carry a label for a held-out-gene split; one primary per task family, with the same-task
datasets named beside it. Guide-level CRISPRi stores are excluded because the label depends on
the protospacer, not the gene's coding sequence; whole-strain readouts have no gene to attach
to.

Also landed with it: `PaperTable.to_latex(table_env="longtable")` in `torchcell/paper/tables.py`,
a page-breaking table with the caption in the first head and the header repeated on every
later page, which is what a sixty-row table needs on an A4 notes-tex page.

Run from the repo root:

```bash
python experiments/database/scripts/build_bacteria_datasets_table.py --organisms ecoli
python experiments/database/scripts/build_ecoli_genslm_benchmark_table.py
make -C notes-tex/database/ecoli-datasets && make -C notes-tex/database/ecoli-datasets check
```
