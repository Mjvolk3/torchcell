---
id: uomyz882o470pxtqi8v6urq
title: Test_rel606
desc: ''
updated: 1791388882123
created: 1791388882123
---

## 2026.10.07 - REL606 genome tests

Covers [[torchcell.sequence.genome.ecoli.rel606]]. Synthetic tests (everywhere, the real
class over `REL606_LOCI` of `_bacterial_fixtures` with the network refusing): the members
read and the recorded `data.db` source, the assembly record, the gene set with tRNA and
rRNA tags, a minus-strand gene, the RefSeq inline GO route (and the obsolete term
leaving with `remove_deprecated_go_terms`), and resolution round trips (locus tags, a
symbol, a RefSeq tag, the shared tRNA symbol `metZ`, the pseudogene, a retired tag, a
b-number). Tier tests (`@pytest.mark.data`, skipped without the set): 4,383 / 4,316 / 67
loci, 85 tRNA and 22 rRNA tags, 4,209 CDS; GO coverage (2,325 rows, 2,232 genes, 1,649
terms); 2,230 genes and 1,599 terms after obsolete removal; round trips; the three
selenoprotein CDS.

Run: `DATA_ROOT=/scratch/projects/torchcell-scratch pytest tests/torchcell/sequence/genome/ecoli/test_rel606.py --data`.
