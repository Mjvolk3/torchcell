#!/usr/bin/env bash
# scripts/zot_search.sh -- search the personal Zotero mirror from any torchcell checkout.
#
# Runs `zotlib.index search` (thesis repo) against the hybrid txtai index at
# /bulk/zotero-library/_index/bge-large. zotlib is NOT a torchcell dependency: the
# thesis checkout is put on PYTHONPATH and its conda env is used, so nothing is
# installed into the torchcell env. GilaHyper only (the index + the bge-large query
# model live here); see .claude/skills/zot-search/SKILL.md for the retrieval protocol.
#
# Usage: scripts/zot_search.sh "<query>" [-k N] [--scope KEY]... [--kind chunk] [--json] ...
# Every argument is passed through to `python -m zotlib.index search`.
#
# Overrides: ZOTLIB_ROOT (checkout holding zotlib/), ZOTLIB_PYTHON (interpreter),
# CUDA_VISIBLE_DEVICES (else the card with the least used memory is picked).
set -euo pipefail

ZOTLIB_ROOT="${ZOTLIB_ROOT:-}"
if [[ -z "$ZOTLIB_ROOT" ]]; then
    for candidate in \
        "$HOME/Documents/projects/thesis" \
        "$HOME/Documents/projects/thesis.worktrees/library-index"; do
        if [[ -f "$candidate/zotlib/index.py" ]]; then
            ZOTLIB_ROOT="$candidate"
            break
        fi
    done
fi
if [[ ! -f "$ZOTLIB_ROOT/zotlib/index.py" ]]; then
    echo "zot_search: no checkout with zotlib/index.py found (ZOTLIB_ROOT='$ZOTLIB_ROOT')" >&2
    exit 2
fi

ZOTLIB_PYTHON="${ZOTLIB_PYTHON:-$HOME/miniconda3/envs/thesis/bin/python}"
if [[ ! -x "$ZOTLIB_PYTHON" ]]; then
    echo "zot_search: interpreter not found: $ZOTLIB_PYTHON (set ZOTLIB_PYTHON)" >&2
    exit 2
fi

# Match CUDA's device numbering to nvidia-smi's so the pick below means what it says.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used \
        --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
fi
export CUDA_VISIBLE_DEVICES

export PYTHONPATH="$ZOTLIB_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$ZOTLIB_PYTHON" -m zotlib.index search "$@"
