---
id: icprrx4bf34fktoarxrr7dq
title: Promoter Expression Gpra
desc: ''
updated: 1789437520906
created: 1789437520906
---

## 2026.09.14 - What is deposited for the million-scale random-promoter data

Two papers, both in the literature mirror, neither built: de Boer 2020 (*Nat Biotechnol*, doi 10.1038/s41587-019-0315-8, "Deciphering eukaryotic gene-regulatory logic with 100 million random promoters") and Vaishnav 2022 (*Nature*, doi 10.1038/s41586-022-04506-6), the same gigantic parallel reporter assay: 80 bp of random DNA in a fixed scaffold driving a fluorescent reporter against a constitutive second color, read by Sort-seq.

Deposits checked 2026-09-14 (GEO's web form is CAPTCHA-walled; the FTP listings are open):

| accession or record | contents |
|---|---|
| GEO GSE104878 (de Boer) | the bulk: average promoter expression per sequence, pTpA scaffold 883 MB and Abf1TATA scaffold 830 MB, plus galactose 548 MB and glycerol 451 MB and an OLS glucose set 402 MB, all gzipped; also the scaffold library sequences |
| GEO GSE104903 (de Boer) | MNase nucleosome data only, 1.4 MB |
| GEO GSE163045 (Vaishnav) | defined medium (SD-Ura) expression per sequence, 637 MB gzipped, plus three small files |
| GEO GSE163866 (Vaishnav) | log-phase RNA-seq RPKM, 168 KB |
| Zenodo 4436477 (Vaishnav) | complex-medium training data 3.59 GB and defined-medium 2.47 GB as text, trained models and fitness functions as `.h5`, orthologous promoter expression across the 1,011 genomes 31 MB, S288C reference promoter expression in both media 8 MB |

So the training corpora are deposited in full, not only summaries. **This is a sequence-to-expression corpus, not a genotype-to-phenotype record**: the perturbation is a synthetic 80 bp sequence on an episomal reporter, there is no modified native gene and no strain-level phenotype, so it would enter as an embedding-side resource rather than through the perturbation ontology. The revitalized supported-datasets table lists `vaishnav_2022` with the open action "Classify `vaishnav_2022` and `zhang_2020` (new datasets vs reference-only)"; `zhang_2020` is now resolved as a real dataset ([[torchcell.datasets.scerevisiae.zhang2020]]).
