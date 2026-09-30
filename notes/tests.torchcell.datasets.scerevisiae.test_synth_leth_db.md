---
id: ulyeqqeebg3srdjy1av07ha
title: Test_synth_leth_db
desc: ''
updated: 1790481848881
created: 1790481848881
---

## 2026.09.26 - Hermetic builds of the SynLethDB SL and SR loaders with a stub genome

Test file: `tests/torchcell/datasets/scerevisiae/test_synth_leth_db.py` (8 test functions, 12 cases with parametrization). Source under test: `torchcell/datasets/scerevisiae/synth_leth_db.py`. Both loaders call `_build_gene_name_mapping()` before `super().__init__`, iterating `genome.db.all_features()`, so the genome is a duck-typed stub whose features carry only the three attributes the mapping reads (`featuretype`, `id`, `attributes`). The raw CSV is written into `<root>/raw/` so `download()` (Google Drive) is never called. No network, no `$DATA_ROOT`.

### Stub genome and the expected mapping

Eight features: six `gene` features and a `CDS` and an `mRNA` that must be ignored.

| featuretype | id | attributes |
| --- | --- | --- |
| gene | YAL001C | gene TFC3; Alias TSV115, FUN24 |
| CDS | YAL001C_CDS | gene TFC3 (ignored) |
| gene | YAL002W | gene VPS8; Alias FUN15 |
| mRNA | YAL002W_mRNA | gene VPS8 (ignored) |
| gene | YAL003W | gene EFB1; Alias TEF5 |
| gene | YAL005C | gene SSA1; Alias YG100 |
| gene | YAL008W | gene FUN14 (no Alias) |
| gene | YAL012W | no attributes (name-less ORF) |

`gene_name_to_systematic` is pinned as the exact 16-key dict (each gene id maps to itself, plus every `gene` value and every `Alias`); `YAL001C_CDS` is absent; the `genome` attribute is deleted after the mapping (`"genome" not in vars(ds)`), which is what keeps the dataset picklable.

### SL fixture and expected values

`Yeast_SL.csv` with header `n1.name,n2.name,r.statistic_score,r.pubmed_id`:

| row | n1 | n2 | score | pubmed | exercises |
| --- | --- | --- | --- | --- | --- |
| 0 | TFC3 | VPS8 | 0.85 | 12345678 | common names |
| 1 | TSV115 | EFB1 | 0.5 | 23456789 | alias |
| 2 | YAL005C | FUN14' | 0.1 | 34567890 | systematic passthrough; prime stripped for lookup, `perturbed_gene_name` normalized to `FUN14_prime` by the schema |
| 3 | YAL012W | TEF5 | 0.3 | 45678901 | name-less ORF maps to itself |
| 4 | SSA1 | VPS8 | (empty) | 56789012 | NaN score |

Record 0 is compared to hand-built `SyntheticLethalityExperiment` / `SyntheticLethalityExperimentReference` / `Publication` by `model_dump` equality: two `SgaKanMxDeletionPerturbation` with `strain_id="S288C"`, `Media(name="YEPD", state="solid", is_synthetic=False)` at 30 C, `is_synthetic_lethal=True` with score 0.85, reference `False` with score `None`, publication PMID only (no DOI). Genotypes sort by systematic name, so row 3 stores `[YAL003W, YAL012W]`. Side files: `preprocess/` is exactly `build_manifest.json, experiment_reference_index.json, gene_set.json` (no `data.csv`, so `ds.df is None`); one reference with members `[0..4]`; gene set = the six systematic names sorted; the manifest records `dataset_name = "sl"`, the loader class and module; `processed/` is `lmdb, pre_filter.pt, pre_transform.pt` with NO `interned` env, because these loaders write the records LMDB directly with `pickle` rather than through `_intern_record`.

### SR fixture and expected values

`Yeast_SR.csv` with two rows: `TFC3,VPS8,0.42,11111111` and `SSA1,YAL012W,,22222222`. Record 0 pinned by `model_dump` equality (`is_synthetic_rescue=True`, score 0.42; reference `False`, `None`); record 1's empty score becomes `None` through `pd.notna`. One reference with members `[0, 1]`, `experiment_reference_type = "synthetic rescue"`, gene set of four names.

### Findings (pinned as the code behaves)

- **SL stores NaN, SR stores None, for the same empty score cell.** The SL loader does `float(row["r.statistic_score"])` with no NaN guard and `SyntheticLethalityPhenotype` has no NaN validator, so an empty score is stored as `nan`; the SR loader wraps the same expression in `pd.notna(...)` and stores `None`. Pinned on SL record 4 with `math.isnan`.
- **The documented "fall back to the raw name" is always a crash.** `get_systematic_name` returns the input name when unmapped (after printing `Warning: No systematic name found for gene NOTAGENE`); that raw name then fails `GenePerturbation`'s systematic-name regex, so the build raises `pydantic.ValidationError("Invalid systematic gene name format")`. Pinned for both classes with `pytest.raises` plus the captured warning line. A common name that happens to look like a systematic name would slip through; not constructible with a real genome and not pinned.
- `strain_id="S288C"` on every perturbation is the reference strain name, not an SGA strain id as the field's description (`'Strain ID' in raw data`) states. Pinned via the record equality.

### Not covered, and why

- `download()` (Google Drive session with the `download_warning` cookie): network.
- A NaN in `r.pubmed_id` (pandas would upcast the column to float and `str()` would give `"12345678.0"`). Hypothesis, untested: needs a second build per class and the real files are not known to contain one.
- `main()` (needs a real `SCerevisiaeGenome` under `$DATA_ROOT`).

## 2026.09.30 - Phase 14: aliases, duplicates, the float PMID, downloads

Eight to sixteen tests, 81 to 100 percent. Findings: an alias listed on two genes resolves to the later one silently (line 77); a swapped duplicate pair is stored twice, and a synonym pair is stored as a double deletion of one ORF (157-170); one blank PMID makes pandas read the column as float, so every record stores `"111.0"` and a `.../111.0/` URL while the blank row stores `"nan"` (229); `main` builds both LMDBs under the working directory, not `$DATA_ROOT` (49, 243). Also pinned: `download` for both classes (a plain response, the Drive `download_warning` confirm round trip, an HTTP error writing no file) and the schema classes.
