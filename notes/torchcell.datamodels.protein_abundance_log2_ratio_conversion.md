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

## 2026.09.15 - Fixed key union: every strain filled to the 1,850 proteins, NaN where unmeasured

The first build kept each record's own keys and the split baselines failed on the second record ("different gene key set"): Messner strains quantify 1,441 to 1,850 of the 1,850 proteins (only 465 in every strain, 1,536 in 95% of strains), and the `Perturbation` processor flattens a dict phenotype to a key-sorted vector without its keys, so two strains with different key sets are misaligned column by column, and the target decode keeps only value groups whose length equals the head's, so most strains would silently have gone unsupervised.

`ProteinAbundanceLog2RatioConverter.process` now scans the raw store first (`scan_reference_union`): the union of protein keys and the wild-type profile, which is one profile because every record carries the same reference values on its own keys (checked). Every converted record spans that union, `NaN` where the strain did not measure the protein, `n_replicates` 0 there (the schema admits 0 only where the value is NaN), the reference zero on every key with the delta-method SE from the full profile. `convert` without a scan keeps per-record keys.

Downstream NaN handling, all landed with this change: `_mean_float_dict` and `_rms_pool_float_dict` (mean deduplicator) average finite values only; `MaskedMultitaskLoss.forward` folds `isfinite(target)` into the feature mask and zeroes the target there, so the pinball loss and every distributional loss run on finite entries; `per_feature_pearson`, `per_strain_pearson` (a shared sparse branch over finite pairs, features with fewer than 3 pairs dropped), `_rank` (`nan_policy="omit"`), `pred_sd_ratio`, `mse`, `nmse` and the PIT cache in `train_cgt_multitask.py`; the masked-conditioning objective gives unmeasured entries an infinite reveal key and excludes them from the observed set; `_to_token_space` uses `where` so a NaN never reaches the observed-label encoder. Split baselines: per-gene train mean and both metrics over finite entries, B2 and B3 fit on residuals with NaN replaced by 0 (the per-gene mean).

Check on a random 200 x 50 matrix with 10% of target entries removed: sparse per-feature Pearson 0.45713 against a per-column manual 0.45713; loss with NaN rows and entries equals the manual masked mean and its gradient is finite.
