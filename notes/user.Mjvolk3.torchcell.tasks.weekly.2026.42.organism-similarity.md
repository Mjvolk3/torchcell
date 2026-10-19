---
id: 1po58kcv3it5or3he5f9k70
title: organism-similarity
desc: ''
updated: 1792450923275
created: 1792450923275
---

## 2026.10.19

- [x] `feat/pkudriavzevii-sd108-genome` worktree (PR #1146, landed 9b320d534): our *P. kudriavzevii* SD108 genome as a tier set (`pkudriavzevii_SD108_v1`, constant `PKUDRIAVZEVII_SD108`, deposited from the private release repository at commit 8ca6a106 with the new `RetrievalMethod.git_repository`) and the class `PKudriavzeviiSD108Genome` on the GFF3+FASTA route, GO stated absent, the S288C name crosswalk read-only [[torchcell.sequence.genome.pkudriavzevii.sd108]] [[scripts.provision_pkudriavzevii_genome]]
- [x] `exp/044-organism-similarity` worktree: the yeast-vs-mammal diversity claim checked against the mirror (a genus and subphylum statement, Peris 2023 Fig 3c; Shen 2018 via Aranguiz 2025), then measured over 34 organisms: isolate vs S288C protein identity 99.72% (median) or 98.04% (mean over ORFs, frameshifts charged); S288C/*S. paradoxus* AAI 92.1%, S288C/*S. eubayanus* 83.5% against human/chimp 98.1%, human/macaque 95.3%, human/mouse 83.7%, human/chicken 72.1%; the 75 Saccharomycotina pairs at median AAI 47.4%. Five figures (claim strip plot, cladogram with AAI, ortholog and EC-4 heatmaps, taxonomy vs AAI tanglegram, the 1,011-isolate NJ tree, ortholog fraction vs AAI). Open: *Z. bailii* IST302 stands in for CLIB 213, *A. limacinum* has no annotated assembly, BUSCO AAI is a single-copy eggNOG OG proxy, CLEAN not run [[experiments.044-organism-similarity]] [[experiments.044-organism-similarity.plan]]
