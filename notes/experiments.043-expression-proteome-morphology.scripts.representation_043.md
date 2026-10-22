---
id: 22binkqtfp182bdkvw3hhjb
title: Representation_043
desc: ''
updated: 1792638765530
created: 1792638765530
---

## 2026.10.22 - Context tables built

Script: `experiments/043-expression-proteome-morphology/scripts/representation_043.py`; library [[torchcell.data.representation_043]]; launcher [[experiments.043-expression-proteome-morphology.scripts.gh_representation_043_build]].

`--stage build` ran as slurm job 6117 at commit `12acf06be` (44 s, MaxRSS 2.6 GB; a first run, job 6116, used the same code uncommitted). Output under `$DATA_ROOT/data/torchcell/experiments/043-expression-proteome-morphology/representation_043/` (`build_record.json`):

| Item | Value |
|---|---|
| Contexts | 1,012 (S288C + 1,011 isolates) over 6,607 genes |
| Vocabulary, FUDT upstream | 1,209,089 keys (isolate and S288C windows + 71,158 Vaishnav windows) |
| Vocabulary, FUDT downstream | 413,607 |
| Vocabulary, ProtT5 | 757,721 |
| Vocabulary, CaLM | 1,163,338 |
| Accessory ORF rows | 84,592 |
| Genes per isolate with no released allele | 596 (mitochondrial genes and SGD annotations the 2018 allele tarball does not carry) |
| Index reference alleles with no place in the 6,607 index | 4 features (YER109C, YFL056C, YFL057C, YOR031W), 4,044 rows |
| Isolate (gene, encoder) slots with a key / gapped | 23,530,350 / 3,188,358 |
| Vaishnav variants without a window key | 0 |

`resolve_genotype` maps one raw genotype onto the resolver (perturbation list -> deletion, isolate or promoter record; haplotype blocks -> segregant); a record's isolate name maps to its context directly or as `SACE_<name>` (93 isolates carry the prefix in the index). `gene_vectors` reads the vectors of a representation lazily, per encoder, and raises on a key with no row.
