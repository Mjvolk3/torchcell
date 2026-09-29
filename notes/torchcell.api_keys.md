---
id: 4unb5vdcx2aihv7q3oa0id2
title: Api_keys
desc: ''
updated: 1790716983879
created: 1790716983879
---

## 2026.09.29 - Shared hashed API keys for tc-lit and tc-data

`torchcell/api_keys.py` factors the key model out of `torchcell/literature/server.py` so the dataset endpoint uses the same scheme without a copy: `ApiKeys` (frozen pydantic, `hashes: {name: sha256hex}`) with `from_file`, `from_pairs(spec, env_name)`, `from_env_names(file_var, inline_var)` and constant-time `verify`; `hash_key`, `mint_key` and `print_minted_key` (the `--gen-key` output). Each server subclasses it to bind its variables: `LiteratureKeys` (`TC_LIT_KEYS_FILE` / `TC_LIT_API_KEYS`) and `DataKeys` (`TC_DATA_KEYS_FILE` / `TC_DATA_API_KEYS`). `tc-lit` behavior is unchanged: same error text for a bad pair, same `KeyError` when neither variable is set, same `--gen-key` lines; `tests/torchcell/literature` passes unchanged (131 passed, 3 skipped).
