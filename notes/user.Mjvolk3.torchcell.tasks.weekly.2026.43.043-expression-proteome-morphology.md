---
id: iplas96fdgd1kg2qgq7wlbp
title: 043-expression-proteome-morphology
desc: ''
updated: 1792393327949
created: 1792393327949
---

## 2026.10.19

- [x] Experiment 043 stage 1: one 15-dataset expression, proteome and morphology query over KG 4.0, registered as a supported query and validated on `2026.10.17-4295f7ca`, with a dev-tree census of all 173,805 records [[experiments.043-expression-proteome-morphology]]
- [x] The raw store build (job 5851) fails in `Neo4jQueryRaw.extract_systematic_gene_names` on the Boocock and Nguessan segregant genotypes; needs a torchcell fix or an owner decision before a rebuild [[experiments.043-expression-proteome-morphology.scripts.build_043_store]]
- [x] Fixed the raw stage for genotypes without a perturbation list (segregants), kept the zero-perturbation isolate, and rebuilt the 043 raw store: 173,617 samples, 188 short of the served 173,805 because the `all` gene clause drops records naming genes outside the genome gene set [[experiments.043-expression-proteome-morphology.scripts.build_043_store]]
- [x] Fixed the doubled shared label in the processors and the segregant reads in the index passes, recomputed the 043 indices without a rebuild, and added the co-location census [[experiments.043-expression-proteome-morphology.scripts.reindex_043_store]]
- [x] Measured correlations between same-readout datasets in the 043 raw store: Sameith Sm single-mutant profiles are identical to Kemmeren's, mRNA-protein agreement is 0.03 per deletion strain and 0.21 per isolate, and Kemmeren carries the most CalMorph signal [[experiments.043-expression-proteome-morphology.scripts.cross_dataset_correlation]]
- [ ] Owner decisions for 043: keep the 188 gene-set drops or re-query, a perturbation-type rule for isolates, and the `aggregator=None` torchcell bug [[experiments.043-expression-proteome-morphology]]
- [x] Within-dataset ceilings: Kemmeren own-culture ceiling 0.729 (114 mutants) replaces the void 019 number; Jackson construction pairs barely above null; Hu has no replicates; no segregant in two panels [[experiments.043-expression-proteome-morphology.scripts.within_dataset_ceilings]]
- [x] 043 datasets primer typeset and published for the batch decision [[experiments.043-expression-proteome-morphology.scripts.primer_043_tables]]
- [x] 043 design pass: data tensors and dataset module, input-side genotype tokens, output-side readout heads, three mermaid diagrams [[experiments.043-expression-proteome-morphology.mermaid.data-flow]]
- [x] 043 primer revised to the owner's plan (every gene-level readout together; splits A, B, C; Ohya probe), with the data-as-tensors section, the input and output designs and one owner-decision checklist; published to Zotero [[experiments.043-expression-proteome-morphology]]
- [ ] Owner decisions for the 043 design: the 20-row checklist, Table 12 of the primer [[experiments.043-expression-proteome-morphology]]

## 2026.10.20

- [x] 043 review pass: reconciled representation and objective decisions folded into the primer (composite token, isolates as reference contexts, heads per level, HL-Gauss, one fold file, linear ladder first, decision rule, data per model part, isolate embedding plan); 29-row decision table; diagrams redrawn [[experiments.043-expression-proteome-morphology]]
- [x] Distinct alleles of the Peter 2018 tarball: 775,856 proteins, 1,198,539 CDS over 1,011 isolates [[experiments.043-expression-proteome-morphology.scripts.count_distinct_alleles]]
- [ ] Owner answers to the seven 043 questions (primer Sec. 5) [[experiments.043-expression-proteome-morphology]]
