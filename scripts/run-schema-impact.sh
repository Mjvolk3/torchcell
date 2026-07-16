#!/usr/bin/env bash
# pre-commit schema-impact wrapper.
#
# When a schema-surface file (torchcell/datamodels/{schema,pydant}.py) is staged, report which
# dataset loaders the change forces to rebuild and BLOCK the commit on a breaking change (unless
# TORCHCELL_SCHEMA_ACK is set). Resolves the torchcell conda env python by $HOME-relative path
# (pre-commit's `entry:` is not shell-interpreted, so ~/$HOME cannot expand there), matching
# scripts/run-mypy.sh. Analyzes the whole surface, so staged filenames are not needed.
# Logic lives in torchcell/provenance/schema_impact.py.
set -euo pipefail
exec "$HOME/miniconda3/envs/torchcell/bin/python" scripts/schema_impact_check.py "$@"
