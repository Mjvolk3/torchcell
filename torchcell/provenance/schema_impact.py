# torchcell/provenance/schema_impact.py
# [[torchcell.provenance.schema_impact]]
"""Static schema-impact gate: which datasets a schema change forces to rebuild.

Compares the working-tree schema surface against a git ref (default ``HEAD``), classifies each
changed symbol as ``breaking`` (loader build fails or stored records become invalid) or
``stale`` (build still works but records would differ), and maps each change to the dataset
LOADERS whose transitive closure contains it. Run as a pre-commit hook it stops a schema edit
from silently breaking an unchanged loader (the exact latent break that motivated this).

Needs no stored state -- git plus the recomputed dependency graph IS the state, so it runs
per-machine against local git with no coordination. The authoritative "which built LMDB is
stale" question is answered separately by ``build_manifest.py`` (retroactive, content-hashed).

Exit codes: ``0`` no breaking impact; ``1`` breaking impact present (unless ``TORCHCELL_SCHEMA_ACK``
is set to acknowledge). ``--strict`` also fails on stale-only impact.
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from torchcell.provenance.schema_deps import (
    ContractSpec,
    SchemaSurface,
    default_surface_modules,
    load_surface,
    load_surface_from_sources,
    loader_closure,
    loader_schema_deps,
)

__all__ = [
    "ChangeKind",
    "SymbolChange",
    "LoaderImpact",
    "ImpactReport",
    "classify_change",
    "diff_surfaces",
    "map_impacts",
    "build_impact_report",
]

_ACK_ENV = "TORCHCELL_SCHEMA_ACK"


class ChangeKind(StrEnum):
    """Severity of a schema-symbol change for its dependent datasets."""

    breaking = (
        "breaking"  # construction fails or stored records invalid -> must fix + rebuild
    )
    stale = "stale"  # build still works but records would differ -> rebuild to refresh


def _max_kind(kinds: Iterable[ChangeKind]) -> ChangeKind:
    return (
        ChangeKind.breaking if ChangeKind.breaking in set(kinds) else ChangeKind.stale
    )


class SymbolChange(BaseModel):
    """A schema symbol that changed contract between two surfaces, with its severity."""

    symbol: str
    kind: ChangeKind
    status: Literal["added", "removed", "modified"]
    reasons: list[str]


class LoaderImpact(BaseModel):
    """A dataset loader whose closure contains one or more changed symbols."""

    loader: str  # repo-relative path
    dataset_classes: list[str]
    changed_symbols: list[str]
    kind: ChangeKind


class ImpactReport(BaseModel):
    """The full result of diffing a schema change: changed symbols and impacted loaders."""

    base: str
    changed_symbols: list[SymbolChange] = Field(default_factory=list)
    impacted_loaders: list[LoaderImpact] = Field(default_factory=list)

    @property
    def has_breaking(self) -> bool:
        """True if any impacted loader is hit by a breaking change."""
        return any(
            impact.kind == ChangeKind.breaking for impact in self.impacted_loaders
        )

    @property
    def has_impact(self) -> bool:
        """True if any loader is impacted at all."""
        return bool(self.impacted_loaders)


def classify_change(
    old: ContractSpec, new: ContractSpec
) -> tuple[ChangeKind, list[str]]:
    """Classify a fingerprint-changed symbol into breaking/stale with human-readable reasons."""
    reasons: list[str] = []
    kinds: list[ChangeKind] = []

    if old.bases != new.bases:
        reasons.append(f"base classes changed: {list(old.bases)} -> {list(new.bases)}")
        kinds.append(ChangeKind.breaking)
    if old.config != new.config:
        reasons.append("model Config changed")
        kinds.append(ChangeKind.breaking)

    old_fields = dict(old.fields)
    new_fields = dict(new.fields)
    for name in sorted(set(new_fields) - set(old_fields)):
        if new_fields[name].required:
            reasons.append(f"added required field '{name}'")
            kinds.append(ChangeKind.breaking)
        else:
            reasons.append(f"added optional field '{name}'")
            kinds.append(ChangeKind.stale)
    for name in sorted(set(old_fields) - set(new_fields)):
        reasons.append(f"removed field '{name}'")
        kinds.append(ChangeKind.breaking)
    for name in sorted(set(old_fields) & set(new_fields)):
        old_field = old_fields[name]
        new_field = new_fields[name]
        if old_field == new_field:
            continue
        if old_field.annotation != new_field.annotation:
            reasons.append(
                f"field '{name}' type changed: {old_field.annotation} -> {new_field.annotation}"
            )
            kinds.append(ChangeKind.breaking)
        if old_field.required != new_field.required:
            if new_field.required:
                reasons.append(f"field '{name}' became required")
                kinds.append(ChangeKind.breaking)
            else:
                reasons.append(f"field '{name}' became optional")
                kinds.append(ChangeKind.stale)
        elif old_field.default != new_field.default:
            reasons.append(f"field '{name}' default changed")
            kinds.append(ChangeKind.stale)

    old_assigns = dict(old.assigns)
    new_assigns = dict(new.assigns)
    for name in sorted(set(new_assigns) - set(old_assigns)):
        reasons.append(f"added member/const '{name}'")
        kinds.append(ChangeKind.stale)
    for name in sorted(set(old_assigns) - set(new_assigns)):
        reasons.append(f"removed member/const '{name}'")
        kinds.append(ChangeKind.breaking)
    for name in sorted(set(old_assigns) & set(new_assigns)):
        if old_assigns[name] != new_assigns[name]:
            reasons.append(f"member/const '{name}' value changed")
            kinds.append(ChangeKind.breaking)

    old_methods = dict(old.methods)
    new_methods = dict(new.methods)
    changed_methods = sorted(
        name
        for name in set(old_methods) | set(new_methods)
        if old_methods.get(name) != new_methods.get(name)
    )
    if changed_methods:
        reasons.append(f"validator/serializer changed: {changed_methods}")
        kinds.append(ChangeKind.stale)

    if (
        not reasons
    ):  # fingerprint differed but no structural reason surfaced -> be conservative
        reasons.append("contract fingerprint changed")
        kinds.append(ChangeKind.stale)
    return _max_kind(kinds), reasons


def diff_surfaces(old: SchemaSurface, new: SchemaSurface) -> list[SymbolChange]:
    """Every symbol whose contract changed between two surfaces, classified."""
    changes: list[SymbolChange] = []
    for name in sorted(old.names | new.names):
        old_spec = old.specs.get(name)
        new_spec = new.specs.get(name)
        if old_spec is None:
            changes.append(
                SymbolChange(
                    symbol=name,
                    kind=ChangeKind.stale,
                    status="added",
                    reasons=["new symbol"],
                )
            )
        elif new_spec is None:
            changes.append(
                SymbolChange(
                    symbol=name,
                    kind=ChangeKind.breaking,
                    status="removed",
                    reasons=["symbol removed"],
                )
            )
        elif old.fingerprints[name] != new.fingerprints[name]:
            kind, reasons = classify_change(old_spec, new_spec)
            changes.append(
                SymbolChange(symbol=name, kind=kind, status="modified", reasons=reasons)
            )
    return changes


def _dataset_classes(loader_path: Path) -> list[str]:
    """Names of dataset classes declared in a loader (``@register_dataset`` or ``*Dataset``)."""
    names: list[str] = []
    for node in ast.parse(loader_path.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.ClassDef):
            continue
        decorated = any(
            (isinstance(d, ast.Name) and d.id == "register_dataset")
            or (isinstance(d, ast.Attribute) and d.attr == "register_dataset")
            for d in node.decorator_list
        )
        if decorated or node.name.endswith("Dataset"):
            names.append(node.name)
    return names


def _loader_paths(repo_root: Path) -> list[Path]:
    """All dataset-loader modules under ``torchcell/datasets`` (excluding dunder/deprecated)."""
    out: list[Path] = []
    for path in sorted((repo_root / "torchcell" / "datasets").rglob("*.py")):
        if path.name.startswith("__") or "deprecated" in path.name:
            continue
        out.append(path)
    return out


def _git_show(ref: str, relpath: str, repo_root: Path) -> str:
    """Contents of ``relpath`` at ``ref``; empty string if the path did not exist there."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{ref}:{relpath}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else ""


def map_impacts(
    changes: list[SymbolChange],
    loader_paths: list[Path],
    old_surface: SchemaSurface,
    new_surface: SchemaSurface,
    repo_root: Path,
) -> list[LoaderImpact]:
    """Map changed symbols to the loaders whose closure contains them (breaking first)."""
    change_by_symbol = {change.symbol: change for change in changes}
    changed_names = set(change_by_symbol)
    impacts: list[LoaderImpact] = []
    if not changed_names:
        return impacts
    for loader_path in loader_paths:
        # union of new+old closure so a REMOVED symbol still flags loaders that imported it
        if not (
            loader_schema_deps(loader_path, new_surface)
            or loader_schema_deps(loader_path, old_surface)
        ):
            continue
        closure = loader_closure(loader_path, new_surface) | loader_closure(
            loader_path, old_surface
        )
        hit = sorted(closure & changed_names)
        if not hit:
            continue
        impacts.append(
            LoaderImpact(
                loader=str(loader_path.relative_to(repo_root)),
                dataset_classes=_dataset_classes(loader_path),
                changed_symbols=hit,
                kind=_max_kind(change_by_symbol[symbol].kind for symbol in hit),
            )
        )
    impacts.sort(key=lambda impact: (impact.kind != ChangeKind.breaking, impact.loader))
    return impacts


def build_impact_report(base: str, repo_root: Path) -> ImpactReport:
    """Diff the working-tree schema surface against ``base`` and map changes to loaders."""
    module_paths = default_surface_modules()
    new_surface = load_surface(module_paths)
    old_surface = load_surface_from_sources(
        {
            str(path): _git_show(base, str(path.relative_to(repo_root)), repo_root)
            for path in module_paths
        }
    )
    changes = diff_surfaces(old_surface, new_surface)
    impacts = map_impacts(
        changes, _loader_paths(repo_root), old_surface, new_surface, repo_root
    )
    return ImpactReport(base=base, changed_symbols=changes, impacted_loaders=impacts)


def format_report(report: ImpactReport) -> str:
    if not report.changed_symbols:
        return f"No schema contract changes vs {report.base}."
    lines: list[str] = [f"Schema impact vs {report.base}", ""]
    lines.append(f"Changed symbols ({len(report.changed_symbols)}):")
    for change in report.changed_symbols:
        tag = "BREAKING" if change.kind == ChangeKind.breaking else "stale"
        lines.append(f"  [{tag}] {change.symbol} ({change.status})")
        for reason in change.reasons:
            lines.append(f"      - {reason}")
    lines.append("")
    if not report.impacted_loaders:
        lines.append(
            "Impacted datasets: none (no built loader depends on the changed symbols)."
        )
        return "\n".join(lines)
    breaking = [i for i in report.impacted_loaders if i.kind == ChangeKind.breaking]
    lines.append(
        f"Impacted datasets ({len(report.impacted_loaders)}; {len(breaking)} breaking) -> rebuild:"
    )
    for impact in report.impacted_loaders:
        tag = "BREAKING" if impact.kind == ChangeKind.breaking else "stale"
        classes = ", ".join(impact.dataset_classes) or impact.loader
        lines.append(f"  [{tag}] {classes}  via {', '.join(impact.changed_symbols)}")
    return "\n".join(lines)


def _repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    return Path(result.stdout.strip()) if result.returncode == 0 else Path.cwd()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m torchcell.provenance.schema_impact",
        description="Report which datasets a schema change forces to rebuild.",
    )
    parser.add_argument(
        "--base", default="HEAD", help="git ref to diff against (default: HEAD)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit nonzero on ANY impact, not just breaking",
    )
    # pre-commit passes staged filenames as positional args; accept and ignore them (we always
    # analyze the whole surface, since one staged schema hunk can affect the whole graph).
    parser.add_argument("files", nargs="*", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    report = build_impact_report(args.base, _repo_root())
    print(format_report(report))

    if not report.has_impact:
        return 0
    if os.environ.get(_ACK_ENV):
        print(f"\n{_ACK_ENV} set -> impact acknowledged; proceeding.")
        return 0
    if report.has_breaking:
        print(
            f"\nBREAKING schema change: update + rebuild the datasets above, "
            f"then re-run with {_ACK_ENV}=1 to acknowledge."
        )
        return 1
    if args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
