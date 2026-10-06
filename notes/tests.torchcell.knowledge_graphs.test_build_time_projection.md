---
id: lax9g8c8dgn64x3939z7soa
title: Test_build_time_projection
desc: ''
updated: 1791321323186
created: 1791321323186
---

## 2026.10.06 - Calibration, projection and the gatherer, worked by hand

Fixture: a synthetic three-adapter calibration over hand-picked full counts, so every rate is a short decimal. Smf Costanzo 40 s over 2,000 records (0.02 s/rec), Dmi Costanzo 300 s over min(100,000 cap, 1,000,000) = 100,000 (0.003 s/rec), Smf Kuzmin 2018 10 s over 500 (0.02 s/rec).

- `kg_full` projection: 40 + 300 + 10 = 350 s, error 0 %, rows sorted Dmi (600/7 %), Smf (80/7 %), Kuzmin (20/7 %).
- Uncapped: Dmi 0.003 x 1,000,000 = 3,000 s, total 3,050 s; no measured time leaves `measured_sec` and `error_pct` at `None`.
- `subset_size=100` with Dmi `None`: 2 + 3,000 + 2 = 3,004 s; the 2 s tie keeps calibration order (stable sort); against 3,500 s the error is 100 x (3004 - 3500) / 3500 = -14.17 %.
- `subset_size=0`: total 0 and every share 0.0 % (the guard, not a division).
- `cap_for` / `effective_records`: eleven cases, including a per-dataset `None` that uncaps a dataset under a global cap and a cap of 0 that is a cap.
- Refusals: pydantic error types and locations (`int_parsing`, `missing`, `literal_error`, `int_from_float`, the adapter list index), `KeyError` for an unmapped adapter or an unknown dataset, `ZeroDivisionError` for a zero-record dataset.
- The committed `experiments/database/scripts/build966_timings.json`: 33 adapters in `ADAPTER_TO_DATASET` order summing to the measured 33,180 s, so `kg_full` self-checks at 0 %; uncapped is 32,172 + 1,008/100,000 x 20,705,612 = 240,884.57 s.
- `ADAPTER_TO_DATASET` is checked against the inverse of the served `dataset_adapter_map`; all 33 pairs agree.
- `gather_dataset_full_records` runs against a fake `lmdb` module in `sys.modules`: dev tree first, `<build_tree>/<leaf>/processed/lmdb` when the dev dir is absent, exact `open` kwargs (`readonly=True, lock=False, subdir=True, max_dbs=0`), one close per env.

Finding (pinned by `test_lmdb_subpaths_differ_from_loader_default_roots_only_for_synthleth`): the comment at `torchcell/knowledge_graphs/build_time_projection.py:159-161` says `DATASET_LMDB_SUBPATH` matches every loader default root except DmfCostanzo. Measured against `inspect.signature(cls.__init__)`: DmfCostanzo now matches (its root was fixed in 9eadc39e2), and the two SynLethDB datasets differ. The loaders default to `data/torchcell/syn_leth_db_yeast` and `data/torchcell/syn_rescue_db_yeast` (`torchcell/datasets/scerevisiae/synth_leth_db.py:488`, `:621`), while the table names `synth_lethality_yeast_synth_leth_db` and `synth_rescue_yeast_synth_leth_db`. Both trees exist in the dev `DATA_ROOT`, so the gatherer counts a tree the default loader does not build.

### Correction after audit 2 - the SynLethDB consequence

The consequence stated above was imprecise. Checked read-only on GilaHyper (2026.10.06): the dev-tree directories `synth_lethality_yeast_synth_leth_db` and `synth_rescue_yeast_synth_leth_db` hold only `raw/`, no `processed/lmdb`. Without `build_tree` the gatherer raises on `lmdb.open` for them; with `build_tree` it falls back to the database tree, whose LMDBs hold 14,000 and 6,948 entries (the committed constants), while the dev default-root LMDBs that the live rebuild reads (`build_dataset_lmdb.dataset_default_root`) hold 13,996 and 6,942. The table follows the KG conf path (`torchcell/knowledge_graphs/conf/scerevisiae_global_kg.yaml:72`). Reach: only `experiments/database/scripts/project_build_time.py`; latent.

Second finding (pinned in `test_adapter_to_dataset_is_the_inverse_of_the_served_adapter_map`): the projection covers 33 of the 51 datasets in `dataset_adapter_map`. The 18 absent: AminoAcidCooper2010, Bloom2019, CrisprMagicLian2019, CrispriChemgenSmith2016, CrispriMormino2022, the six EnvChemgen sets (Auesukaree2009, Costanzo2021, Hoepfner2014, Mota2024, Vanacloig2022, Wildenhain2015), FattyAcidSmith2006, Het/HomHillenmeyer2008, NadalRibellesPerturbSeq2025, ProteomeMessner2023, SmfBaryshnikova2010, SmfODuibhir2014. Their record counts were not measured here. The committed total of `DATASET_FULL_RECORDS` is 44,357,773 (2 x 20,705,612 Costanzo double-mutant plus 2,946,549).
