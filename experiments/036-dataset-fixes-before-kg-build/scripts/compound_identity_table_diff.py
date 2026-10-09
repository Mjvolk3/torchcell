# experiments/036-dataset-fixes-before-kg-build/scripts/compound_identity_table_diff.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.compound_identity_table_diff]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/compound_identity_table_diff
r"""Diff two compound-identity tables and name every row that would RENAME a served node (#726).

A compound row is NOT additive. Adding IPTG once changed what
``resolved_compound("IPTG").name`` returns and so renamed the IPTG node of already-served
Foo 2014 records, which is the full-rebuild case rather than an increment, and nothing in
the schema-impact gate or the admission check fires on it (the compound table is not part
of the schema closure). So every regeneration of the table has to be diffed against the
one it replaces, by LOOKUP KEY, before it is committed.

This compares an OLD table against a NEW one and reports, per lookup key (a row's name or
any of its synonyms, casefolded, which is what ``resolve_compound_identity`` matches on):

- **renames**: the key resolved to one canonical name before and a DIFFERENT one now. Each
  is a node rename in every record that passes that label, so the datasets using it need a
  full rebuild and their pinned tests move.
- **newly resolved**: the key had no row, or a row with no InChIKey, and now has one. This
  is the additive case the pass exists to produce.
- **lost**: the key resolved before and does not now, or lost its InChIKey. A regeneration
  should never do this; each one is a regression to explain before committing.
- **identifier changes**: same canonical name, different InChIKey, CID or ChEBI id, which
  means PubChem's own record moved under us.

``retrieved_at`` is ignored, because a re-query legitimately stamps every row with the run
date and that alone is not a content change.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/compound_identity_table_diff.json``.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/compound_identity_table_diff.py \\
        --old <old table.json> --new torchcell/datamodels/compound_identity_table.json
"""

import argparse
import hashlib
import json
import os
import os.path as osp
from pathlib import Path
from typing import Any

RESULTS = "experiments/036-dataset-fixes-before-kg-build/results"
#: The fields a diff reports a change in; `retrieved_at` is deliberately not one.
IDENTITY_FIELDS = ("inchikey", "pubchem_cid", "chebi_id", "smiles", "resolution_status")


def read_table(path: str) -> tuple[dict[str, dict[str, Any]], str, int]:
    """Lookup key -> its row, plus the file's sha256 and its record count.

    The key set is what ``resolve_compound_identity`` matches on: a row's name and each of
    its synonyms, stripped and lowercased.
    """
    text = Path(path).read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    records = json.loads(text)["records"]
    by_key: dict[str, dict[str, Any]] = {}
    for record in records:
        for key in [record["name"], *record["synonyms"]]:
            normalized = key.strip().lower()
            if normalized in by_key:
                raise ValueError(
                    f"{path}: lookup key {normalized!r} is claimed by two rows "
                    f"({by_key[normalized]['name']!r} and {record['name']!r})"
                )
            by_key[normalized] = record
    return by_key, digest, len(records)


def main() -> None:
    """Diff the two tables and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True, help="the table being replaced")
    parser.add_argument(
        "--new",
        default="torchcell/datamodels/compound_identity_table.json",
        help="the regenerated table",
    )
    args = parser.parse_args()

    old, old_sha, old_n = read_table(args.old)
    new, new_sha, new_n = read_table(args.new)

    renames: list[dict[str, Any]] = []
    newly_resolved: list[dict[str, Any]] = []
    lost: list[dict[str, Any]] = []
    identifier_changes: list[dict[str, Any]] = []
    for key in sorted(set(old) & set(new)):
        before, after = old[key], new[key]
        if before["name"] != after["name"]:
            renames.append(
                {"key": key, "before": before["name"], "after": after["name"]}
            )
        if before["inchikey"] is None and after["inchikey"] is not None:
            newly_resolved.append(
                {"key": key, "name": after["name"], "inchikey": after["inchikey"]}
            )
        if before["inchikey"] is not None and after["inchikey"] is None:
            lost.append(
                {"key": key, "name": before["name"], "inchikey": before["inchikey"]}
            )
        changed = {
            field: [before[field], after[field]]
            for field in IDENTITY_FIELDS
            if before[field] != after[field]
        }
        if changed and before["name"] == after["name"]:
            identifier_changes.append(
                {"key": key, "name": after["name"], "changed": changed}
            )

    result: dict[str, Any] = {
        "old": {
            "path": args.old,
            "sha256": old_sha,
            "records": old_n,
            "keys": len(old),
        },
        "new": {
            "path": args.new,
            "sha256": new_sha,
            "records": new_n,
            "keys": len(new),
        },
        "keys_added": sorted(set(new) - set(old)),
        "keys_removed": sorted(set(old) - set(new)),
        "renames": renames,
        "newly_resolved": newly_resolved,
        "lost": lost,
        "identifier_changes": identifier_changes,
    }
    result["verdict"] = {
        "bytes_identical": old_sha == new_sha,
        "no_key_renames_a_node": not renames,
        "nothing_lost": not lost,
        "no_key_removed": not result["keys_removed"],
        "keys_added": len(result["keys_added"]),
        "newly_resolved": len(newly_resolved),
        "identifier_changes": len(identifier_changes),
    }

    os.makedirs(RESULTS, exist_ok=True)
    out_path = osp.join(RESULTS, "compound_identity_table_diff.json")
    with open(out_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "keys_added"}, indent=2))
    print(f"keys added: {len(result['keys_added'])}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
