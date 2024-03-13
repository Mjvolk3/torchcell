---
id: 119togmo1rze74fbueeywjd
title: S288c_gb
desc: ''
updated: 1695247042877
created: 1695246868534
---
## GeneBank for Constructing Genome

This file is useful and contains a lot of metadata but we have to loop over all sequence records to access features, instead of querying them directly. `gffutils` allows querying of a sqlite db, which will likley be faster, and lends itself to better readability.

The `ncbi dataset` cli is very useful for downloading genomic information. This tool is likely the best option for other genomes. We can download whichever files we choose, `fasta`, `gff`, or `gbff`.
