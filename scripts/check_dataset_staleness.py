#!/usr/bin/env python
"""Report built datasets under DATA_ROOT that are stale against the local schema.

Recomputes each built LMDB's schema-contract fingerprints (from its preprocess/build_manifest.json)
and flags any whose depended-on schema symbols have changed. Thin wrapper; logic lives in
``torchcell/provenance/build_manifest.py``. Usage: ``python scripts/check_dataset_staleness.py``
(reads $DATA_ROOT), or pass ``--data-root <path>``.
"""

import sys

from torchcell.provenance.build_manifest import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
