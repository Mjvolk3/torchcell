# experiments/036-dataset-fixes-before-kg-build/scripts/strain_background_fingerprint_impact.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.strain_background_fingerprint_impact]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/strain_background_fingerprint_impact
"""Which served datasets' schema closures the #507 strain-background schema moves.

Two comparisons, both static (AST contract fingerprints, ``schema_deps``), no git and
no database connection:

1. BASE vs WORKTREE schema surface: the symbols whose contract fingerprint this branch
   changes, adds or removes. BASE is the schema of a checkout at the branch point,
   passed as ``--base-schema`` (``schema.py`` and its sibling ``pydant.py``).
2. For every dataset in the SERVED manifest (``--manifest``), the changed symbols that
   lie in its loader's closure, i.e. the served classes whose stored closure
   fingerprint no longer matches (the ``kg_manifest`` admission check would BLOCK
   them and require a full rebuild).
3. The check ``kg_manifest`` itself runs: each served dataset's STORED closure
   fingerprints against the worktree's current ones (``stale_vs_served``). This also
   counts drift from commits between the served build and the base, so it is reported
   separately from (2).

Writes ``results/strain_background_fingerprint_impact.json`` and ``.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from torchcell.provenance.schema_deps import (
    load_default_surface,
    load_surface,
    loader_closure,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    """Diff the surfaces, map changes onto served datasets, write the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-schema", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    base_schema: Path = args.base_schema
    base = load_surface([base_schema, base_schema.with_name("pydant.py")])
    new = load_default_surface()
    changed = sorted(
        name
        for name in base.names & new.names
        if base.fingerprints[name] != new.fingerprints[name]
    )
    added = sorted(new.names - base.names)
    removed = sorted(base.names - new.names)
    moved = set(changed) | set(added) | set(removed)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows: list[dict[str, str | int]] = []
    for name, entry in sorted(manifest["datasets"].items()):
        loader = REPO_ROOT / entry["loader_relpath"]
        closure = loader_closure(loader, new) | loader_closure(loader, base)
        hit = sorted(closure & moved)
        stored: dict[str, str] = entry["closure"]
        current = {sym: new.fingerprints[sym] for sym in loader_closure(loader, new)}
        stale = sorted(
            sym
            for sym in set(stored) | set(current)
            if stored.get(sym) != current.get(sym)
        )
        rows.append(
            {
                "dataset_class": name,
                "loader": entry["loader_relpath"],
                "n_changed_symbols_in_closure": len(hit),
                "changed_symbols_in_closure": ";".join(hit),
                "stale_vs_served": ";".join(stale),
            }
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = {
        "base_schema": str(base_schema),
        "manifest": str(args.manifest),
        "manifest_release": manifest.get("release"),
        "changed_symbols": changed,
        "added_symbols": added,
        "removed_symbols": removed,
        "n_served_datasets": len(rows),
        "n_served_datasets_moved": sum(
            1 for r in rows if r["n_changed_symbols_in_closure"]
        ),
        "datasets": rows,
    }
    out_json = RESULTS / "strain_background_fingerprint_impact.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (RESULTS / "strain_background_fingerprint_impact.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"changed: {changed}")
    print(f"added: {added}")
    print(f"removed: {removed}")
    print(f"served datasets moved: {summary['n_served_datasets_moved']} of {len(rows)}")
    print(out_json)


if __name__ == "__main__":
    main()
