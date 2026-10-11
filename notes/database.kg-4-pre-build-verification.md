---
id: k3fiugv86mwczqlpo8udxdz
title: Kg 4 Pre Build Verification
desc: ''
updated: 1791584371342
created: 1791584371342
---
## 2026.10.09 - Every mapped store verified against the bytes KG 4.0 will be built from

`build_dataset_lmdb` builds a store; it does not verify one. Slurm array 3565 rebuilt 105
dev stores from `main` today and `python -m torchcell.database.build_dataset_lmdb
--list-stale --include-private` now reads 0, so the staleness gate the live rebuild runs is
satisfied. That gate compares schema-closure fingerprints; it says nothing about whether the
records inside a store still satisfy their own L0 to L4 rules. Measured before this sweep:
115 of the 123 mapped stores carried no `preprocess/verification_report.json` at all. The 8
that did were the stores whose loaders were worked on by hand during the wave
(`env_chemgen_hoepfner2014`, `gene_interaction_butland2008`, `growth_rate_lamoureux2023`,
`growth_rate_s23_schmidt2016`, `growth_wang2015` and the three Yunus 2026 CRISPRi arms), and
each of those reports is newer than its own `preprocess/build_manifest.json`, so it does
describe the bytes on disk. The gap was the other 115.

This sweep closes that gap. One script,
`experiments/036-dataset-fixes-before-kg-build/scripts/kg4_pre_build_verification_sweep.py`,
resolves how each mapped dataset is verified, runs that route against the store now on disk,
and records the verdict rule by rule into
`experiments/036-dataset-fixes-before-kg-build/results/kg4_pre_build_verification_sweep.json`.

```bash
PYTHONPATH=. python experiments/036-dataset-fixes-before-kg-build/scripts/\
kg4_pre_build_verification_sweep.py --list-routes
PYTHONPATH=. python experiments/036-dataset-fixes-before-kg-build/scripts/\
kg4_pre_build_verification_sweep.py            # every mapped dataset
PYTHONPATH=. python experiments/036-dataset-fixes-before-kg-build/scripts/\
kg4_pre_build_verification_sweep.py --table    # the tables below
```

### Three routes, and the route is a property of the dataset

| Route | Datasets | What owns the battery |
|---|---:|---|
| `registry` | 64 | a family runner in `torchcell/verification/runners.py`, reached through one of the 12 family registries or the three morphology root constants |
| `module` | 43 | the loader module's own entry point (`run_verification`, `verify_build`, `verify`, `verify_essentiality`, `verify_growth_build`, `verify_tf_build`, `verify_morphology_build`, `run_tolerance_verification`) |
| `none` | 16 | nothing |

A registry runner is called with its registry narrowed to the one dataset under
verification, so a crash in one store stops at that store instead of taking the family's
other stores with it. The per-dataset L4 rules a runner adds still run, because the runner
reads the stores those rules compare against (Ohya for the yeast containment rules, the
record's own pinned assembly for the bacterial ones). `run_expression` is the one runner
that is not narrowed: its L4 compares the three expression datasets' measured universes
against each other, so the family only means anything whole.

The module route's entry point, argument order and arm selector are recorded per dataset in
the script's `MODULE_ROUTES` table rather than inferred from a signature at runtime, so a
renamed parameter is a loud failure instead of a dataset that quietly verified another arm.
The arm selectors are the loader modules' own keys: `ishii2007.DATASET_ROOTS`,
`yunus2026.DATASETS`, and the `Family` literals of `carruthers2025`, `desiqueira2025`,
`menasalvas2025` and `caglar2017`.

Every route is read-only on `processed/`: the readers in `torchcell.verification.runners`
open each LMDB `readonly=True, lock=False`. The only bytes written are the
`preprocess/verification_report*.json` files each route writes by its own convention, in
the dev tree. `$DATA_ROOT/database/` is never touched and no store is rebuilt. Each dataset
runs in its own subprocess, so neither an OOM nor an exception can end the sweep.
