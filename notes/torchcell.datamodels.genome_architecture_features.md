---
id: mzr7lbqng841w50ir40x5o3
title: Genome_architecture_features
desc: ''
updated: 1791732596182
created: 1791732596182
---

## 2026.10.11 - The genome-architecture vocabulary (Puddu 2019)

`torchcell/datamodels/genome_architecture_features.py` registers the feature vocabulary of a whole-genome-sequencing genome-architecture profile, the way `bacterial_morphology_features.py` registers an imaging assay's. One record of `GenomeArchitecturePhenotype` names its assay, and that assay's symbols are the only permitted keys.

What the module exists to prevent: the numbers in one record are not in one unit. 125 rDNA repeats per haploid genome, 17.5 mtDNA copies per cell, 2.0 copies of chromosome IV and 756 telomeric reads per million are four different quantities, and a consumer that averages across keys averages quantities the source never claimed were comparable. So `ArchitectureBasis` is a property of the FEATURE: `copies_per_haploid_genome`, `copies_per_cell`, `chromosome_copies_per_cell`, `reads_per_million`, `ploidy_deviation_units`, `rearrangement_count`. Each feature also records the genomic interval its number was read from, verbatim from the source's Methods, so a coverage ratio stays interpretable.

Registered assay: `puddu2019_wgs_coverage`, 28 features keyed by the exact column headers of Puddu 2019's Supplementary Table 3 (`rDNA`, `CUP1`, `mitochondria`, `2-micron`, `Ty1`-`Ty5`, `Telomeres`, `chr01`-`chr16`, `ANtot`, `GCR`), platform Illumina HiSeqX, reference assembly `S288c_R64-1-1_Ensembl`. Keys are the release's own symbols so a stored value is comparable to the released bytes without a translation table.

`validate_value` enforces what a basis implies: every value is finite and non-negative (a copy number, a read share and an absolute deviation are all bounded below by zero and none has an upper bound a definition supplies, since a strain can carry 250 rDNA repeats or 385 plasmid copies), and a `rearrangement_count` must be integral. A later sequencing screen registers its own assay rather than widening this one.

Measurements, design decisions and the L0-L4 table: [[torchcell.datasets.scerevisiae.puddu2019]].
