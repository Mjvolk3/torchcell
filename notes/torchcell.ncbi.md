---
id: pbf7j00r33hsb3pcc0szxhv
title: Ncbi
desc: ''
updated: 1690838428405
created: 1690829192479
---
## Downloading a Genome

- Go to [NCBI Assembly](https://www.ncbi.nlm.nih.gov/assembly/) and search for genome of interest.

![](./assets/images/src.torchcell.ncbi.md.NCBI-Assembly.png)

- Select a genome ![](./assets/images/src.torchcell.ncbi.md.R64-Genome.png)

- Select `datasets` ![](./assets/images/src.torchcell.ncbi.md.Genome-assembly-R64.png)

- ![](./assets/images/src.torchcell.ncbi.md.Datasets-command-line-query.png)

## Download S cerevisiae Genome Example

```bash
cd data/ncbi/s_cerevisiae/
datasets download genome accession GCF_000146045.2 --include gff3,rna,cds,protein,genome,seq-report --filename GCF_000146045.2.zip
```
