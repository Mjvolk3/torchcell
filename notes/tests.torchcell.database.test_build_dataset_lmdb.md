---
id: 6de26n76sm1gtb49c875gjy
title: Test_build_dataset_lmdb
desc: ''
updated: 1790777408770
created: 1790777408770
---

## 2026.09.30 - Phase 17: the CLI on toy loaders

Two to eleven tests, 45.5 to 100 percent. Exact stdout (the base class's streaming line, then `BUILT ... in 42s; gene_set size 3; references 2`, then the manifest path) with exit 0; a loader that writes no build manifest exiting 1 with the exact `ERROR:` line; a missing `--dataset` exiting 2 with argparse's message; an unknown class raising before anything is created with the known-class list sorted; an existing `processed/lmdb` refused with the exact `deprecate.sh` recipe; `$DATA_ROOT` read after `load_dotenv`; nothing ever created under `database/`; genome and graph injected only when the loader's `__init__` names them.

## 2026.10.09 - The unreadable state and `--verify` (#833)

Eleven to thirty-two tests in the file. A toy store whose record 0 is re-pickled with an
instance of a class from a throwaway module (installed in `sys.modules`, then removed)
reads `unreadable: ModuleNotFoundError: No module named 'tc_gone'` although every
fingerprint matches, is named by `--list-stale` on stdout, and carries its reason on
stderr, so the class list an array job reads line by line stays bare.

`--verify` is tested through a toy whose `__module__` is a throwaway module carrying the
entry point (with a real one-line `__file__` on disk, because the build manifest is
computed from the loader file's import closure):

- `run_verification(data_root)` passing: the store keeps its manifest, stdout carries
  `verification PASSED: 1 report(s) by <module>.run_verification`.
- `verify_build(dataset_root, data_root)` passing: its report is written to
  `preprocess/verification_report.json`, which is the file the 105-store array rebuild of
  2026-10-09 left absent and each dataset's own `--data` test reads.
- raising, returning no report, or returning a report whose `passed` is False: exit 1 and
  the build manifest is RENAMED to `build_manifest.json.unverified.<stamp>`, after which
  `mapped_store_status` reads the store `no_manifest`.
- no entry point at all: `NO PER-DATASET VERIFIER -- ...` on stdout, manifest kept, exit
  0 (its family runner covers it; 38 of the 123 mapped datasets are in that position).
- an entry point requiring `family`, `arm` or a dataset `name` is refused with
  `LookupError` rather than guessed at.
- a bioproduction registry entry wins over the module's own entry point, since the
  registry path adds the family L4 containment.
- a build WITHOUT `--verify` runs no verification, and the array slurm script passes the
  flag (pinned on the script text).
