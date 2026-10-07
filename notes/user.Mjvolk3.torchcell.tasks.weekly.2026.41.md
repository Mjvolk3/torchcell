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
- [x] `feat/genslm-embedding-dataset` worktree: GenSLM codon language model wrapper and embedding dataset for any genome with a CDS (KT2440, the three E. coli sets, S288C), vendored tokenizer and configs with provenance, Globus fetch script; the paper `zvyaginGenSLMsGenomescaleLanguage2023` captured into the mirror from `torchcell-in` [[torchcell.models.genslm]] [[torchcell.datasets.genslm]] [[scripts.genslm_fetch_weights]]
- [ ] Globus login + Connect Personal setup on GilaHyper (by hand), then fetch the 25M and 250M GenSLM checkpoints and build the KT2440 store [[scripts.genslm_fetch_weights]]
