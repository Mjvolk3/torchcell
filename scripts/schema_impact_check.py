#!/usr/bin/env python
"""Pre-commit / CI entry for the static schema-impact gate.

Diffs the working-tree schema surface against a git ref and reports which dataset loaders a
change forces to rebuild, blocking (exit 1) on a breaking change unless TORCHCELL_SCHEMA_ACK
is set. Thin wrapper so pre-commit runs a plain script (no ``-m`` package double-import).
Logic lives in ``torchcell/provenance/schema_impact.py``.
"""

from torchcell.provenance.schema_impact import main

if __name__ == "__main__":
    raise SystemExit(main())
