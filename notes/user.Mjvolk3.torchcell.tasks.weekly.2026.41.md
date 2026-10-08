---
id: byzr4ppu8xdozf9fox128ed
title: '41'
desc: ''
updated: 1791256185172
created: 1791256185172
---

## 2026.10.05

- [x] `w037-isobutanol-scrnaseq` worktree: picked six of the twelve isobutanol strains on hand for the first single-cell RNA-seq run, as three pairs that each differ by one plasmid [[user.Mjvolk3.torchcell.tasks.weekly.2026.41.w037-mirror-cite]]
- [ ] The table note and the SI caption still say "pre-build ... not yet a versioned Neo4j DB build"; all 51 have been served as release 1.0 since 2026.09.17, so that framing needs the author's call [[torchcell.knowledge_graphs.releases]]
- [ ] Radiant VM: request a ~2 T block volume from NCSA; the NFS-backed store faults on every read (system database included) and Neo4j does not support NFS [[plan.kg-releases.2026.09.19]]
- [ ] Delete `/bulk/deprecated/2026.09.19/kg-releases-partial-backup-workdir` (373 G, the killed first backup) and `/db/deprecated/2026.09.19/kg-releases` if still present

## 2026.10.07

- [x] `chore/notes-tex-groups` worktree: a group layer for `notes-tex/` and its Zotero tree, with the existing collections moved under their groups [[user.Mjvolk3.torchcell.tasks.weekly.2026.41.notes-tex-groups]]
- [x] `feat/bacteria-queue-table` worktree: the bacterial expansion document gains the 281-row discovery queue less the fifty recommended builds (whose papers are now filed in Zotero), and both generators write to the moved `notes-tex/database/` path again [[experiments.database.expansion-bacteria]]
- [x] `feat/bacterial-genome-hard-tests` worktree: cross-source tests of the four bacterial genomes against the feature table, GFF, FASTAs and NCBI's GAF for every locus; two findings pinned (wrapped-product spaces from Biopython, obsolete GO ids in the RefSeq inline route) [[tests.torchcell.sequence.genome.test_bacterial_tier]]
- [x] `docs/ecoli-si-phenotype-audit` worktree: audited all 14 loaded *E. coli* papers' SI for released phenotypes our loaders do not store; 21 loadable-now opportunities worth 840,752 records with a measured count, the largest being Shiver's 235 Nichols conditions at 835,337, and three new duplication risks [[plan.bacteria-si-phenotype-audit-ecoli]]
