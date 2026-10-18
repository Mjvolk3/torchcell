---
id: v3kb5pb2l4fxltu7i4myzrt
title: Yeast_datasets_readout_table
desc: ''
updated: 1792303254302
created: 1792303254302
---

## 2026.10.18 - The yeast readout table

`experiments/database/scripts/yeast_datasets_readout_table.py` renders [[experiments.database.yeast-datasets-readout-table]]: every *S. cerevisiae* dataset a served KG release carries, with its experiment count, whether its readout is gene-level, what the readout is, and what genotype each record carries. The rows and counts are read from the committed release snapshot `database/releases/<release>.json` (default the newest), and the snapshot before it marks the rows that are new. The descriptive columns are the script's `CURATED` registry, the one hand-authored input; the run fails when the registry and the snapshot disagree in either direction, so a dataset added to the KG cannot go unlisted.

*Gene-level* follows the owner's rule of 2026-10-18: yes when the readout is a property of a gene or a named locus (an expression level, a protein abundance, a copy number; Vaishnav 2022's reporter expression and Puddu 2019's copy numbers of named loci included), no when it is a property of the whole cell (growth, viability, morphology, a metabolite or product level). Essentiality and synthetic lethality sit with the whole-cell readouts because their readout is strain viability.

The PDF is rendered by `notes/assets/publish/scripts/bib_tex_pdf.sh` with `--pdf`. Pandoc sizes a pipe table's columns by the dash counts of its separator line once a row is wider than the text block, so the script writes weighted separators; equal dashes ran the long dataset names into the counts. [[experiments.database.yeast-datasets-table]] is a different table (instances against gzip signal) from a different generator and is untouched.
