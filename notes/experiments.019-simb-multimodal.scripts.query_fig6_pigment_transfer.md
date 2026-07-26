---
id: p4jmzzw267rmd5kopf2o4yw
title: Query_fig6_pigment_transfer
desc: ''
updated: 1785040436700
created: 1785040436700
---

## 2026.07.25 - Build + co-location census (deletion-keyed)

Results: `experiments/019-simb-multimodal/results/fig6_pigment_transfer_census.json`
LMDB: `$DATA_ROOT/data/torchcell/experiments/019-simb-multimodal/fig6_pigment_transfer`

Built with `DeletionKeyedGenotypeAggregator` + the `Perturbation` graph processor.

### The gate, measured

| pair | full gene-set key | deletion key (this build) |
| --- | ---: | ---: |
| betaxanthin n metabolome | 0 | **4,432** |
| beta-carotene n metabolome | 0 | **4,221** |

Both land exactly on the raw-LMDB prediction from
`verify_deletion_keyed_aggregation.py` (4,432 / 4,221).

| quantity | value |
| --- | ---: |
| aggregated genotype groups (= `len(dataset)`) | **4,930** |
| groups carrying >=2 modalities | 4,800 |
| groups carrying all three | 4,023 |
| betaxanthin n beta-carotene | 4,193 |
| rows: Mulleder / betaxanthin / beta-carotene | 4,678 / 4,669 / 4,406 |

### Two things the census surfaced

**The COO value counts are disjoint, which is what makes the decode possible.** Under label
`metabolite_level` there are 4,669 records contributing 1 value (betaxanthin) and 4,678
contributing 19 (Mulleder). The `Perturbation` processor drops the dict keys, so group SIZE
is the only separator - and here it is unambiguous.

**No deduplicator merge occurred.** The plan anticipated that the betaxanthin dARO4 and
dARO7 strains would collide, since the cassette carries ARO4/ARO7 as alleles and both gene
names appear in every strain. They do collide under a gene-name SET key (the aggregator's),
but `MeanExperimentDeduplicator` keys on a sorted LIST, which keeps the duplicate entry and
therefore distinguishes `[.., YBR249C, YBR249C, YPR060C]` from `[.., YBR249C, YPR060C,
YPR060C]`. `deduplicator_merged_dataset_names` is empty; deletion keying then separates
them at the aggregator too.

### Caveat carried forward

`Perturbation` marks the UNION of perturbed gene names across a genotype group as
`perturbation_indices`, and ARO4/ARO7/BTS1 are real S288C nodes. So a co-located genotype
is marked with 4 perturbed genes (deleted ORF + ARO4 + ARO7 + BTS1) while a metabolome-only
genotype is marked with 1. The marking therefore depends on which modalities were measured.
This is CONSTANT across conditions A1-A4 (same dataset, same splits), so it does not affect
the Delta contrast, but it would confound a cross-genotype generalization claim.
