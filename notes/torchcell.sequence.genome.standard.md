---
id: p5nxp62jpjn9ucrs6vdkgtw
title: Standard
desc: ''
updated: 1795053661511
created: 1795053661511
---

## 2026.11.18 - Standard genomes by assembly set

`STANDARD_GENOMES` maps an assembly-set id to the genome class that reads it (five bacterial classes: E. coli K-12 BW25113, MG1655, W3110, B REL606, P. putida KT2440). `load_genome(assembly_set)` makes sure every set the class reads is in the genomes tier, downloading an absent one from `tc-data` only when `TC_DATA_URL` is set, and constructs the class. `genome_for(dataset)` reads the one `assembly_set` a dataset's reference index names and calls `load_genome`.

- A downloaded file is sha256-checked against its manifest row before the rename, and `manifest.json` is written last, so an interrupted fetch never looks deposited.
- The yeast classes are not in the table: `SCerevisiaeGenome` takes a `go_root` as well, so it needs its own row shape.
- Worked example for the four served E. coli datasets: `experiments/034-showcase-datasets/scripts/ecoli_genome_snippet.py`. Measured there (one run, GilaHyper, 2026-11-18): Campos 2018 names 3,664 genes, 3,573 current locus tags and 91 pseudogenes; Babu 2014 names 3,880 against MG1655, 17 not in the annotation; Fuhrer 2017 names 3,830 Keio JW numbers, 3,632 resolved through the gene synonym, 151 pseudogenes, 47 not in the annotation.
