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

## 2026.09.30 - Raw-file sha256 pins verified at build time

Issues #518, #524, #537 and the hash items of #528. The provenance rule says every built LMDB traces to an exact, hash-pinned raw file, but the pin was checked only inside `download()`, which PyG skips whenever `raw/` is populated; several loaders also copied into `raw/` before hashing, and some deposits created the mirror directory before checking. This module now owns the one shared implementation:

- `RawSha256MismatchError(RuntimeError)`: carries `path`, `expected`, `observed`; message `sha256 mismatch for <path>: expected <pin>, observed <digest>`.
- `file_sha256`, `verify_sha256`, `verify_raw_files(raw_dir, {name: pin})` (called first thing in every pinned loader's `process()`).
- `write_verified(data, dest, pin, source)` (hash the payload, write `.partial`, rename), `copy_verified(src, dest, pin)` (hash the source, copy to `.partial`, rename), `link_verified(src, dest, pin)` (hash the source, symlink if absent). A refusal never leaves bytes at the destination.

Sweep of `grep -n "sha256" torchcell/datasets/scerevisiae/*.py` (31 files match; `kuzmin2018.py` matches only in a comment and has no pin). Columns: (a) the pin was checked only in `download()`, so a file in `raw/` was built from unchecked; (b) the copy into `raw/` happened before the check, so a refusal left unverified bytes in `raw/`; (c) a refused `deposit_raw_mirror` left a directory (or a partial deposit) behind. "n/a" in (c) means the loader has no deposit function, or its deposit records the source digest rather than refusing against a pin.

| Loader | Pin source in `process()` | (a) download-only | (b) copy before check | (c) refused deposit leaves dir |
|---|---|---|---|---|
| auesukaree2009 | `_PDF_SHA256` | yes | yes | yes (mkdir before hash) |
| baryshnikova2010 | `XLS_SHA256` | yes | no (symlink after hash) | no |
| bloom2019 | raw-mirror manifest | yes | no | n/a (records source digest) |
| cachera2023 | `DATA_SHA256` | yes (#528) | no (bytes hashed before write) | n/a |
| caudal2024 | zip and tarball pins + genomes-tier manifest for the two matrices | yes; the two matrices were never re-hashed once in `raw/` | no (symlink) | n/a |
| cooper2010 | `TABLE4_SHA256` | yes (#518) | no (symlink) | partial (a refused second file left the first deposited) |
| costanzo2021 | `_S1_SHA256` | yes (#524) | yes (#524) | yes (#524, mkdir before hash) |
| dasilveira2014 | `DATA_SHA256`, `CHEBI_SHA256` | yes | yes | n/a |
| hillenmeyer2008 | raw-mirror manifest | matrix already re-checked in `process()`; key and controls files download-only | no (symlink) | n/a (records source digest) |
| hoepfner2014 | `_DRYAD_FILES` | yes | yes (Dryad fetch streamed into `raw/` then hashed) | yes, and the deposit copied into the MIRROR before hashing |
| lian2019 | `TSV_SHA256`, `DESIGN_D_SHA256` | yes | no | partial |
| lopez2024 (both classes) | `_XLSX_SHA256` | yes | yes | n/a |
| messner2023 | `MATRIX_SHA256`, `METADATA_SHA256` | yes (#528, present file skipped) | no | n/a |
| mormino2022 | `PDF_SHA256`, `PAPER_MD_SHA256` | yes | no | yes (mkdir before hash) |
| mota2024 | `_ACID_SPECS` pins | yes | yes (mirror copy and ESM write before hash) | partial |
| mulleder2016 | `DATA_SHA256` | yes | no | n/a |
| nadal_ribelles2025 | `SHA256_EXPECTED` | yes | no (symlink) | n/a |
| oduibhir2014 | `_DATASET_S2_SHA256` | yes (#537) | yes (#537) | n/a |
| ohnuki2018 | `_RAW_FILES` | yes | yes | n/a |
| ohnuki2022 | `MUTANT_SHA256`, `WT_SHA256` | yes | no (source hashed first; a failed post-copy check left the copy) | n/a |
| ohya2005 | `_RAW_FILES` | yes | yes | n/a |
| ozaydin2013 | `_SI_SHA256` | yes | no | n/a |
| smith2006 | `XLS_SHA256` | yes | no (symlink) | no |
| smith2016 | `EFFECT_SHA256`, `GUIDE_SHA256` | yes | no | partial |
| vanacloig2022 | `DATA_SHA256` | yes | no (symlink) | no |
| wildenhain2015 | `DATA_SHA256`, `AID_SHA256` | yes | no | partial |
| xue2025 | `DATA_SHA256` | yes | yes | n/a |
| yeastphenome | per-screen `valuez_sha256` | yes | no | n/a |
| yoshida2012 | `PDF_SHA256` | yes | no | n/a |
| zelezniak2018 (both classes) | `DATA_SHA256`, `METABOLITE_DATA_SHA256` | yes | no | n/a |

Every row is fixed the same way: `process()` calls `verify_raw_files(self.raw_dir, pins)` before any record is read; `download()` stages through `copy_verified` / `write_verified` / `link_verified` (hash first, then `.partial` + rename, or symlink), so a refusal leaves nothing in `raw/`; every `deposit_raw_mirror` that refuses against a pin verifies all sources before creating any directory. The helpers and `RawSha256MismatchError` live once in `torchcell/data/experiment_dataset.py`, the base module every loader already imports.

Records are unchanged for a verified raw file: no record-building code moved; the check only runs before it. Tests: `tests/torchcell/data/test_experiment_dataset.py` (the helpers), one refusal test per loader test file, and the `raw_pin_calls` / `off_pin_raw` fixtures in `tests/torchcell/conftest.py`.

## 2026.09.30 - Single Reference Build and Recursive Hash Sort (Issue #532)

`transform_item` constructed the reference model twice and discarded the first; it now constructs it once (`test_transform_item_builds_the_reference_once` counts one `__init__`). `serialize_for_hashing` serialized a reference MODEL as `json.dumps(dict(sorted(dump.items())))`, sorting only the top-level keys, so a model and its `model_dump()` hashed differently. A model now serializes as `json.dumps(model_dump(), sort_keys=True)`, identical to its dump. The dict path the reference index uses is untouched: the toy fixture's three reference ids are pinned to the literal hashes computed from the pre-fix code (`test_serialize_for_hashing_sorts_a_model_at_every_depth_like_its_dump`), so no built reference index changes.
