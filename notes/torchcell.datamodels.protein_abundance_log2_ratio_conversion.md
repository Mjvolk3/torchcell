---
id: uhde5k2pt68whzukh0z7w0n
title: Protein_abundance_log2_ratio_conversion
desc: ''
updated: 1789443196143
created: 1789443196143
---

## 2026.09.15 - Build-time log2 ratio of a protein-abundance experiment to its reference

`torchcell/datamodels/protein_abundance_log2_ratio_conversion.py`. A `Converter` for `Neo4jCellDataset` that rewrites every `ProteinAbundanceExperiment` so that `protein_abundance[k] = log2(experiment[k] / reference[k])`, the reference profile becomes zero, both standard errors move to the log2 scale by the delta method (`se / (value * ln 2)`), and `measurement_type` becomes `log2_ratio_to_reference(<original>)`. Other experiment types pass through unchanged, so the converter sits in a build that unions the proteome with expression datasets.

Why a converter and not a training-time transform: the expression datasets store `log2(mutant / reference)` per gene, and the `Perturbation` graph processor only reads `experiment.phenotype`, never the reference. Doing the ratio at build time gives the LMDB the same label form for both modalities, and every downstream reader (the per-gene head, the split-matched linear baselines, the census) sees one convention.

The base `Converter.convert` hands the conversion function the experiment alone, so this class overrides `convert` to read both halves of the record. A protein missing from the reference, a non-positive quantity on either side, or a record already on the ratio scale raises rather than yielding a NaN or a double conversion.

Tests: `tests/torchcell/datamodels/test_protein_abundance_log2_ratio_conversion.py` (5 tests). Verified on a real Messner record from the dev LMDB: 1,830 proteins, log2 ratio range -2.36 to 0.97, median 0.01, reference all zero. First use: [[experiments.019-simb-multimodal.scripts.build_fig3_proteome]].
