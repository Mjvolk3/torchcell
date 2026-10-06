---
id: 9eczbd2cz3578pib7z1bzzt
title: Test_signal
desc: ''
updated: 1791321330935
created: 1791321330935
---

## 2026.10.06 - Path resolution and the CLI front

`resolve_lmdb` on directories under `tmp_path` holding an empty `data.mdb`: a relative argument joins `DATA_ROOT` and appends `processed/lmdb`, an absolute one ignores `DATA_ROOT`, `--lmdb` appends nothing, and a missing `data.mdb` raises `FileNotFoundError("No data.mdb under <resolved dir>")`. With `--lmdb` pointing at a dataset root the message names that root.

`main` runs through a monkeypatched `sys.argv` with `load_dotenv`, `read_first_record`, `phenotype_descriptor` and `stream_gzip_signal` replaced by recorders, and the two formatters real. The order of calls is dotenv, first record, descriptor, gzip stream (label = the raw CLI argument). Printed signal lines worked by hand: 1,234,567 bytes is `1.2 MB  (1.2×10⁶ bytes, 1,234,567)`, 584 bytes is `584 B  (5.8×10² bytes, 584)`. An unset `DATA_ROOT` is a `KeyError` before any reader runs, and a missing positional is argparse exit 2.

Mutation note: replacing the absolute-path branch with `Path(data_root) / arg` survives because pathlib already discards the left operand when the right one is absolute, so that branch is redundant rather than untested.

The source header still names `tests/torchcell/paper/test_tables.py` as its test file (`torchcell/paper/signal.py:4`); this file is the mirrored one.

After audit 2: the absolute-path test was deleted, because `Path(data_root) / "/abs"` is already `Path("/abs")` and no input distinguishes the `os.path.isabs` branch; the absolute case stays covered in the `--lmdb` test with a nonexistent `DATA_ROOT`. The `--lmdb` `main` test now pins all five stdout lines and the full call order.
