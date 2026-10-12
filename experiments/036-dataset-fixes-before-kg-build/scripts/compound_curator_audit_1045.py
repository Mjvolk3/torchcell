# experiments/036-dataset-fixes-before-kg-build/scripts/compound_curator_audit_1045.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.compound_curator_audit_1045]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/compound_curator_audit_1045
r"""Curate the compound rows the #1045 media entries need, and merge them additively.

The removals audit (#1045) row 16 adds Price 2018's 'MOPS Rich Defined media_noCarbon' to
``MEDIA_LIBRARY``. Four of its single substances had no identity row with an InChIKey, so
they are curated here from ONE input list,
``compound_identity_inputs/audit_1045_media.txt``, with the committed curator
(``torchcell.datamodels.compound_identity_curate.curate``), and merged into the pinned
table with the same additive merge the #726/#860 pass used
(``compound_curator_726_860.merge``): no existing row's name, InChIKey, CID, ChEBI id or
status changes, nothing is removed, a label whose CID an existing row already holds is
folded into that row as a synonym, and a lookup key claimed twice stops the run.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/compound_curator_audit_1045.json``
(the adopted PubChem record per label, the table digests before and after). With
``--write`` it also rewrites ``torchcell/datamodels/compound_identity_table.json``; the
printed sha256 is then re-pinned into ``compound_identity.py::_TABLE_SHA256`` by hand.

Run from the repo root (the response cache makes a rerun offline)::

    python experiments/036-dataset-fixes-before-kg-build/scripts/compound_curator_audit_1045.py \
        --cache $DATA_ROOT/tmp/pubchem_curator_cache_audit_1045_2026-10-12.json --write
"""

import argparse
import hashlib
import importlib.util
import json
import os
import os.path as osp
from pathlib import Path
from typing import Any

from torchcell.datamodels.compound_identity_curate import CuratedRow, curate, serialize

RESULTS = "experiments/036-dataset-fixes-before-kg-build/results"
INPUT_LIST = Path("torchcell/datamodels/compound_identity_inputs/audit_1045_media.txt")
TABLE = Path("torchcell/datamodels/compound_identity_table.json")
MERGE_SCRIPT = Path(
    "experiments/036-dataset-fixes-before-kg-build/scripts/compound_curator_726_860.py"
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_merge() -> Any:
    """The #726/#860 pass's additive ``merge``, loaded from its committed script."""
    spec = importlib.util.spec_from_file_location(
        "compound_curator_726_860", MERGE_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {MERGE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.merge


def main() -> None:
    """Curate the list, merge it additively, and write the evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--min-interval", type=float, default=1.0)
    parser.add_argument("--write", action="store_true", help="rewrite the table")
    args = parser.parse_args()

    old_bytes = TABLE.read_bytes()
    existing = [CuratedRow(**r) for r in json.loads(old_bytes)["records"]]
    if serialize(existing).encode("utf-8") != old_bytes:
        raise RuntimeError("the pinned table does not round-trip through serialize()")

    new_rows, directives = curate([INPUT_LIST], [], args.cache, args.min_interval)
    merge = _load_merge()
    rows, folds = merge(
        existing, new_rows, {d.label.strip().lower(): d.label for d in directives}
    )
    before = {(r.name, r.inchikey, r.pubchem_cid, r.chebi_id) for r in existing}
    after = {(r.name, r.inchikey, r.pubchem_cid, r.chebi_id) for r in rows}
    if not before <= after:
        raise RuntimeError(f"an existing row changed: {sorted(before - after)[:3]}")
    text = serialize(rows)
    key_to_row = {
        k.strip().lower(): row for row in rows for k in [row.name, *row.synonyms]
    }
    labels = [
        {
            "label": d.label,
            "row_name": key_to_row[d.label.strip().lower()].name,
            "inchikey": key_to_row[d.label.strip().lower()].inchikey,
            "pubchem_cid": key_to_row[d.label.strip().lower()].pubchem_cid,
            "chebi_id": key_to_row[d.label.strip().lower()].chebi_id,
            "resolution_status": key_to_row[d.label.strip().lower()].resolution_status,
            "smiles": key_to_row[d.label.strip().lower()].smiles,
        }
        for d in directives
    ]
    summary = {
        "input_list": str(INPUT_LIST),
        "input_list_sha256": _sha256(INPUT_LIST.read_bytes()),
        "table_before": {"sha256": _sha256(old_bytes), "records": len(existing)},
        "table_after": {"sha256": _sha256(text.encode("utf-8")), "records": len(rows)},
        "rows_added": len(rows) - len(existing),
        "folds": folds,
        "table_written": args.write,
    }
    if args.write:
        TABLE.write_text(text, encoding="utf-8")
    os.makedirs(RESULTS, exist_ok=True)
    with open(osp.join(RESULTS, "compound_curator_audit_1045.json"), "w") as handle:
        json.dump({"summary": summary, "labels": labels}, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"summary": summary, "labels": labels}, indent=2))


if __name__ == "__main__":
    main()
