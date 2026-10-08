# scripts/kg_manifest_conf_audit.py
# [[scripts.kg_manifest_conf_audit]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/kg_manifest_conf_audit.py
"""Measure issue #743 against a served KG manifest, read-only.

For every served dataset whose adapter module serves more than one mapped dataset
class, print the conf yaml the manifest entry recorded (the module's FIRST conf, before
the fix), the conf the adapter class binds (after the fix), and three facts about the
bound conf:

- ``baseline``: whether ``manifest.adapter_files`` holds a hash for it and that hash
  equals its content at the served commit (``git show``), i.e. whether the manifest
  already carries the right baseline even though the entry named the wrong file;
- ``tree``: whether its working-tree content still equals that recorded hash;
- ``same list``: whether the recorded and the bound conf enable the same methods at the
  served commit (if not, the method-level drift check read the wrong enable-list).

Usage (from the repo root; the manifest is opened for reading only)::

    python scripts/kg_manifest_conf_audit.py --manifest <kg_manifest.json>
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from collections import Counter
from pathlib import Path

import yaml
from pydantic import BaseModel

from torchcell.knowledge_graphs.dataset_adapter_map import build_adapter_map
from torchcell.knowledge_graphs.kg_manifest import (
    _dataset_class,
    dataset_adapter_files,
    load_manifest,
)


class ConfAuditRow(BaseModel):
    """One served dataset of a multi-class adapter module."""

    module: str
    dataset: str
    recorded_conf: str
    bound_conf: str
    baseline_recorded: bool  # manifest.adapter_files hashes the bound conf
    baseline_matches_served_commit: bool
    tree_matches_recorded_hash: bool
    same_enable_list: bool


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_show(repo_root: Path, ref: str, relpath: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{ref}:{relpath}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _enable_list(text: str) -> list[str]:
    conf = yaml.safe_load(text)
    return [
        m["method_name"]
        for key in ("node_methods", "edge_methods")
        for m in conf["cell_adapter"][key]
    ]


def audit(manifest_path: Path, repo_root: Path) -> list[ConfAuditRow]:
    """Rows for every served dataset whose module serves several mapped classes."""
    manifest = load_manifest(manifest_path)
    assert manifest.torchcell_commit is not None
    commit = manifest.torchcell_commit
    adapter_map = build_adapter_map(include_private=True)
    per_module = Counter(a.__module__ for a in adapter_map.values())
    rows: list[ConfAuditRow] = []
    for name, entry in sorted(manifest.datasets.items()):
        cls = _dataset_class(name)
        adapter = adapter_map[cls]
        if per_module[adapter.__module__] < 2:
            continue
        module_rel, bound = dataset_adapter_files(cls, repo_root)
        recorded = entry.adapter_files[1]
        stored = manifest.adapter_files.get(bound)
        at_commit = _git_show(repo_root, commit, bound)
        rows.append(
            ConfAuditRow(
                module=Path(module_rel).name,
                dataset=name,
                recorded_conf=Path(recorded).name,
                bound_conf=Path(bound).name,
                baseline_recorded=stored is not None,
                baseline_matches_served_commit=stored == _sha256(at_commit),
                tree_matches_recorded_hash=stored
                == _sha256((repo_root / bound).read_text(encoding="utf-8")),
                same_enable_list=_enable_list(_git_show(repo_root, commit, recorded))
                == _enable_list(at_commit),
            )
        )
    return rows


def format_rows(rows: list[ConfAuditRow]) -> str:
    """A markdown table, one row per dataset, plus the count of mis-recorded entries."""

    def yn(flag: bool) -> str:
        return "yes" if flag else "NO"

    lines = [
        "| module | dataset | recorded conf (before) | bound conf (now) | "
        "baseline hashed | baseline = served commit | tree = recorded hash | "
        "same enable-list |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r.module}` | {r.dataset} | `{r.recorded_conf}` | `{r.bound_conf}` | "
            f"{yn(r.baseline_recorded)} | {yn(r.baseline_matches_served_commit)} | "
            f"{yn(r.tree_matches_recorded_hash)} | {yn(r.same_enable_list)} |"
        )
    wrong = sum(r.recorded_conf != r.bound_conf for r in rows)
    lines.append("")
    lines.append(
        f"{len(rows)} served datasets in {len({r.module for r in rows})} multi-class "
        f"modules; {wrong} recorded another class's conf."
    )
    return "\n".join(lines)


def main() -> int:
    """Print the audit table for ``--manifest``; the manifest is only read."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--repo", default=".", type=Path)
    args = parser.parse_args()
    print(format_rows(audit(args.manifest, args.repo.resolve())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
