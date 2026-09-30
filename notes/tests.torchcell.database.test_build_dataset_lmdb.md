---
id: 6de26n76sm1gtb49c875gjy
title: Test_build_dataset_lmdb
desc: ''
updated: 1790777408770
created: 1790777408770
---

## 2026.09.30 - Phase 17: the CLI on toy loaders

Two to eleven tests, 45.5 to 100 percent. Exact stdout (the base class's streaming line, then `BUILT ... in 42s; gene_set size 3; references 2`, then the manifest path) with exit 0; a loader that writes no build manifest exiting 1 with the exact `ERROR:` line; a missing `--dataset` exiting 2 with argparse's message; an unknown class raising before anything is created with the known-class list sorted; an existing `processed/lmdb` refused with the exact `deprecate.sh` recipe; `$DATA_ROOT` read after `load_dotenv`; nothing ever created under `database/`; genome and graph injected only when the loader's `__init__` names them.
