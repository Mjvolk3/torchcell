---
id: nklnr2h5vo1s86x5t0v95ag
title: Experiment_dataset
desc: ''
updated: 1709699945486
created: 1709699916365
---
Base class for the torchcell datasets... Calling `ExperimentDataset` out of a lack of creativity.

## 2026.09.29 - Downloading a packaged artifact instead of building

`ExperimentDataset._download` gained a third branch. With `processed/lmdb` present nothing happens (unchanged). With `TC_DATA_URL` set and no LMDB, `_download_artifact` asks the `tc-data` endpoint ([[torchcell.datasets.client]]) for the newest `supported` artifact whose slug is this root's basename and whose `torchcell_version` shares the installed `major.minor`, checks that the row's `dataset_class` is this class, downloads it to `<root>/artifacts/<archive>` with sha256 verification and resume, unpacks `processed/` and `preprocess/` into the root, and requires `preprocess/build_manifest.json` to exist; PyG's `_process` then finds `processed/lmdb` and skips `process()`. With the endpoint set and nothing compatible in the index, the loader raises a `RuntimeError` naming the endpoint, the slug and the installed version; it never falls back to the publisher download while the variable is set. With the variable unset, an un-built dataset downloads its raw files as before.

The client import is local to the method: `torchcell.datasets` eagerly imports every loader module, and those import this module, so a top-level import would be circular. Tests: `tests/torchcell/data/test_experiment_dataset_download.py` (a minimal subclass whose `download` and `process` both raise, driven through the real packager and app via `TestClient`).
