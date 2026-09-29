---
id: evuq3riimllacy3mx9p0b24
title: Genotype_environment_aggregate
desc: ''
updated: 1790549475620
created: 1790549475620
---

## 2026.09.27 - An aggregator keyed on the (genotype, environment) cell

`GenotypeAggregator` keys on the perturbed gene set alone. That is the right identity for a
solid-growth build, where every record of a genotype is the same condition; for a
chemogenomic record, a strain in one of 41 to 5,170 conditions, it would put every condition
of a gene into one processed entry, and `label_df` keeps the first non-NaN value it finds.
`GenotypeEnvironmentAggregator` keys on the cell instead:

- genotype: the sorted list of (gene, perturbation type, copy number, reference copy number)
  and the reference genome's ploidy. Type is part of the key here, unlike the gene-set
  aggregator, because a heterozygous deletion (`engineered_copy_number`, one of two copies)
  and a homozygous one (`kanmx_deletion`) of the same gene are different strains, and Hoepfner
  2014 carries both for 648,977 gene-by-compound cells.
- environment: `environment_identity` from `torchcell.datamodels.identity`, the same content
  address the graph adapter uses for its environment nodes, so provenance quotes and
  free-text names never enter the key and a compound renamed under one InChIKey does not
  split a cell.

`aggregate_key_raw` parses only the record's environment into pydantic (`Environment(**d)`),
at 0.12 to 0.36 ms per record on the four served chemogenomic datasets (scratch timing,
2026-09-27, 500 records each). Tests in
`tests/torchcell/data/test_genotype_environment_aggregate.py`: two doses, two compounds,
het versus hom, ploidy, and an undosed environment are separate cells; a repeated screen, a
rename under one InChIKey, and perturbation order are not; the raw and pydantic paths agree.
First used by [[experiments.033-env-chemgen-pooled.scripts.query]].
