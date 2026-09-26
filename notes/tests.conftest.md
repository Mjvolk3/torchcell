---
id: arpre85og14uih5smapr0yv
title: Conftest
desc: ''
updated: 1790409471700
created: 1790409471700
---

## 2026.09.26 - The safe-by-default contract for plain pytest

`tests/conftest.py` is the root of the test-suite build-out ([[plan.test-suite-buildout.2026.09.25]], Decisions 6 and 15). It sets a sentinel `DATA_ROOT` (`/tmp/torchcell-test-data-root`) with `os.environ.setdefault` before any `torchcell` import, plus `WANDB_MODE=disabled` and `MPLBACKEND=Agg`; defines the six opt-in flags `--gpu --slow --data --neo4j --network --wandb` and skips every marked test whose flag is absent, naming the flag in the reason; installs autouse guards that fail an unmarked test which runs `sbatch` (always), `gh`/`ssh`, resolves a hostname, opens a URL, sends a `requests` call, opens a Neo4j driver, or calls `wandb.init`; and asserts at session end that the sentinel root is still absent or empty.

What the guards found on the first full run (2026.09.26, `main` at e1db34e35 plus Phase 0a): `tests/torchcell/models/test_fungal_up_down_transformer.py` reaches the HuggingFace API even with the model cached (transformers lists repo templates on `from_pretrained`), so it is `network`-marked; `test_yeast_GEM.py` downloaded a 300 s yeast-GEM release into `data/torchcell/yeast-GEM` under cwd on every CI run, so its fixture now reads the `$DATA_ROOT` checkout, downloads into a tmp root only under `--network`, and skips otherwise. Under an empty-but-set `DATA_ROOT` (which the sentinel is), the twelve `os.getenv("DATA_ROOT") is None` gates produced 4 failures and 20 errors (the sgd genome and the sample-batch loaders opened a missing root); those became `isdir` gates on the exact mirror path plus `@pytest.mark.data`. Result: 1630 passed, 68 skipped, 3 xfailed in 40 s under both the sentinel and `DATA_ROOT=$(mktemp -d)`, same counts, sentinel directory never created (baseline before: 1653 passed, 49 skipped, 4 failed, 20 errors, 382 s).
