#!/usr/bin/env bash
# pre-commit supported-query drift wrapper.
#
# When a schema-surface file, the BioCypher schema config, the cell adapter, a shipped query,
# the supported-query package, or a release snapshot is staged, check every registered query
# against the newest committed release snapshot and BLOCK the commit when a `supported` query
# drifts (a `deprecated` one is reported and never blocks). TORCHCELL_QUERY_DRIFT_ACK=1 lets a
# deliberate ontology revision through after the report is read; the query-drift CI job on
# main then files the before-next-kg-build issue. Resolves the torchcell conda env python by
# $HOME-relative path, like scripts/run-schema-impact.sh; PYTHONPATH is the checkout being
# committed, so a worktree checks its own torchcell rather than the primary checkout's.
# Logic lives in torchcell/knowledge_graphs/supported_queries/check.py.
set -euo pipefail
PYTHONPATH="$(git rev-parse --show-toplevel)"
export PYTHONPATH
status=0
"$HOME/miniconda3/envs/torchcell/bin/python" -m torchcell.knowledge_graphs.supported_queries \
  --repo-root "$PYTHONPATH" check || status=$?
if [ "$status" -eq 1 ] && [ "${TORCHCELL_QUERY_DRIFT_ACK:-}" = "1" ]; then
  echo "TORCHCELL_QUERY_DRIFT_ACK=1: supported-query drift acknowledged; commit allowed."
  exit 0
fi
exit "$status"
