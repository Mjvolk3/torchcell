---
id: r0ysbjjb8s5s1nri32an85h
title: Test_raw_pins
desc: ''
updated: 1790819173498
created: 1790819173498
---

## 2026.09.30 - Build-time pins across every loader (issue #561)

Covers every loader module that calls `verify_raw_files`, enumerated by `PINNED_LOADERS` in [[tests.torchcell.conftest]] (an AST scan of `torchcell/datasets/scerevisiae/`, 30 modules today), so a new pinned loader is parametrized with no list to edit.

- `test_process_verifies_every_raw_file_against_its_pin_before_reading[<loader>]`: for each concrete dataset class, `process()` on a `raw/` of empty files reaches `verify_raw_files` exactly once, before any read, with `self.raw_dir` and the full `{file: pin}` mapping. Asserted: the mapping equals the pins the module declares, its files are exactly `raw_file_names`, and every constant pin is 64-hex. Bloom 2019 and both Hillenmeyer 2008 classes pin from the raw-mirror manifest and Caudal 2024 pins two files from the genomes tier; their manifest reads are stubbed with stand-ins that name the path asked, so the assertion proves each file is checked against its own record. A missing table entry for a new class fails with `KeyError`.
- `test_a_real_data_test_meets_the_real_pin[slow|data|slow+data]` and `test_every_other_test_keeps_the_recorder`: `restore_real_pins` puts the real check back in all 30 modules for a `slow`/`data` test and leaves the recorder for any other marker.
- `test_the_recorder_checks_presence_and_records_without_hashing`: the module recorder records `(module, raw dir, pins)` and refuses an absent file by name.
- `test_download_refuses_a_manifest_digest_off_the_module_pin[<7 loaders>]`: a manifest digest other than the module constant raises `ManifestPinMismatchError` with the exact message, and nothing is linked.

Mutation checks run by hand before commit: dropping the design workbook from Lian 2019's mapping, reading a raw file before `verify_raw_files` in Ohnuki 2022, and removing `data` from `REAL_DATA_MARKERS` each failed the matching test.

## 2026.09.30 - Review fixes on PR #577

- `test_every_loader_is_pinned_or_on_the_debt_list` (new): (A) the modules under `torchcell/datasets/scerevisiae/` with a class defining `process()`, minus `PINNED_LOADERS`, equal the literal `UNPINNED_LOADERS` debt list (`costanzo2016`, `costanzo2016_deprecated`, `kemmeren2014`, `kuzmin2018`, `kuzmin2020`, `sameith2015`, `sgd`, `synth_leth_db`). (B) No module outside `PINNED_LOADERS` carries a pin, meaning a `sha256`-named identifier, attribute, argument or def, or a 64-hex string constant, found by AST so comments do not count. The `== 30` count is dropped.
- The process test no longer creates raw files. With `raw/` empty, any open before the check raises. The docstring now states what is proved: the first call carries the complete mapping before any raw file is opened. Later calls are not observed because the stand-in raises. The manifest and genomes-tier stand-ins encode the data root and the tier key in the digest they return. A new pinned class raises `KeyError` until it has a row in `_pins_from_constants`, which is the tripwire.
- The download refusal now covers all 11 (relpath, pin) pairs of the seven loaders. Earlier files have their pin repointed at the staged bytes, and exactly those are linked. `dotenv.load_dotenv` is stubbed.
- Mutation checks: an `open().readline()` before the call in Hillenmeyer, a wrong data root in Caudal, and removing Xue's call each failed.
