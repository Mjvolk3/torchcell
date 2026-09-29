# scripts/kg_compat_page.py
# [[scripts.kg_compat_page]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/kg_compat_page.py
# Test file: tests/scripts/test_kg_compat_page.py
"""Generate the package-version / knowledge-graph-release compatibility page.

Reads the committed release snapshots (``database/releases/<release>.json`` and their
``.closures.json`` companions, ``torchcell.knowledge_graphs.release_snapshot``) and the
package tags from ``v1.2.0`` on (``git tag --sort=v:refname``), and for every (tag,
release) pair runs the closure check of ``releases.closure_compatibility`` against the
schema surface AS IT WAS AT THAT TAG (``git show <tag>:torchcell/datamodels/schema.py``
and ``pydant.py`` parsed with ``schema_deps.load_surface_from_sources``, no checkout,
no temp files). The verdicts:

- ``compatible``: every served dataset's closure fingerprints match the surface at the
  tag, so that package version serializes the served records under the contract they
  were built with.
- ``partial (<n> datasets drifted: A, B)``: the named datasets would serialize
  differently under that package version; when every served dataset drifted the cell
  says so (``partial (all <n> datasets drifted)``) instead of listing all of them.
- ``unknown``: the tag predates the surface modules, so there is nothing to compare.

Writes ``docs/source/database/compatibility.md`` (MyST); ``--check`` exits 1 when the
file on disk differs from what would be written, so CI can hold the page to the
snapshots. Plan: [[plan.data-release-program.2026.09.29]], Decisions 1 and 2.

Usage::

    python scripts/kg_compat_page.py            # regenerate the page
    python scripts/kg_compat_page.py --check    # exit 1 if the page is stale
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from torchcell.knowledge_graphs.kg_manifest import SURFACE_RELPATHS
from torchcell.knowledge_graphs.release_snapshot import (
    KgReleaseSnapshot,
    load_closures,
    load_snapshots,
)
from torchcell.knowledge_graphs.releases import closure_compatibility
from torchcell.provenance.schema_deps import SchemaSurface, load_surface_from_sources

REPO = Path(__file__).resolve().parents[1]
FIRST_TAG = "v1.2.0"
OUTPUT_RELPATH = "docs/source/database/compatibility.md"
SCRIPT_RELPATH = "scripts/kg_compat_page.py"

HOW_DECIDED = (
    "Every served record is serialized under the schema contract of the commit that "
    "built it, and the release snapshot records, per dataset, the contract fingerprints "
    "of that dataset's schema closure. A package tag is compatible with a release when "
    "the schema surface at that tag (`torchcell/datamodels/schema.py` and `pydant.py`) "
    "reproduces every one of those fingerprints, which is the same drift check the "
    "admission gate runs before an incremental import, turned around to face a client. "
    "A partial verdict names the datasets whose records that package version would "
    "serialize differently; the rest read unchanged."
)


def package_tags(repo_root: Path, first: str = FIRST_TAG) -> list[str]:
    """Tags from ``first`` on, in version order (``git tag --sort=v:refname``)."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), "tag", "--sort=v:refname"],
        capture_output=True,
        text=True,
        check=True,
    )
    tags = result.stdout.split()
    if first not in tags:
        raise ValueError(f"tag {first} is not in {repo_root}; fetch the tags first")
    return tags[tags.index(first) :]


def surface_at_tag(repo_root: Path, tag: str) -> SchemaSurface | None:
    """The schema surface at ``tag``, or None when a surface module is absent there."""
    sources: dict[str, str] = {}
    for relpath in SURFACE_RELPATHS:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{tag}:{relpath}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        sources[relpath] = result.stdout
    return load_surface_from_sources(sources)


def verdict(
    snapshot: KgReleaseSnapshot,
    closures: dict[str, dict[str, str]],
    surface: SchemaSurface | None,
) -> str:
    """``compatible``, ``partial (...)`` naming the drifted datasets, or ``unknown``."""
    if surface is None:
        return "unknown"
    report = closure_compatibility(
        snapshot.release,
        snapshot.torchcell_commit,
        snapshot.datasets,
        closures,
        surface,
    )
    if report.ok:
        return "compatible"
    if not report.compatible and not report.unchecked:
        return f"partial (all {len(report.drifted)} datasets drifted)"
    names = ", ".join(drift.dataset_class for drift in report.drifted)
    return f"partial ({len(report.drifted)} datasets drifted: {names})"


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def render_page(
    snapshots: list[KgReleaseSnapshot],
    tags: list[str],
    verdicts: dict[tuple[str, str], str],
) -> str:
    """The page text: releases table, tag-by-release matrix, and the decision rule."""
    lines = [
        f"<!-- Generated by {SCRIPT_RELPATH}. Do not edit by hand. -->",
        "",
        "# Database releases and package compatibility",
        "",
        f"This page is generated by `{SCRIPT_RELPATH}` from the release snapshots "
        "committed under `database/releases/` and must not be edited by hand; "
        "regenerate it with `python scripts/kg_compat_page.py` after a release is "
        "stamped, and `--check` fails when it is stale.",
        "",
        "## Releases",
        "",
        _row(
            [
                "KG release",
                "KG version",
                "built",
                "commit",
                "package version at build",
                "tag",
            ]
        ),
        _row(["---"] * 6),
    ]
    for snapshot in snapshots:
        lines.append(
            _row(
                [
                    snapshot.release,
                    snapshot.version,
                    snapshot.built_at,
                    snapshot.torchcell_commit[:8],
                    snapshot.torchcell_version or "(not recorded)",
                    snapshot.torchcell_tag or "(untagged)",
                ]
            )
        )
    lines += [
        "",
        "## Compatibility",
        "",
        _row(["package tag", *(snapshot.release for snapshot in snapshots)]),
        _row(["---"] * (1 + len(snapshots))),
    ]
    for tag in tags:
        lines.append(
            _row([tag, *(verdicts[tag, snapshot.release] for snapshot in snapshots)])
        )
    lines += ["", "## How compatibility is decided", "", HOW_DECIDED, ""]
    return "\n".join(lines)


def build_page(repo_root: Path) -> str:
    """Load the snapshots and tags from ``repo_root`` and render the page."""
    snapshots = load_snapshots(repo_root)
    if not snapshots:
        raise ValueError(f"no release snapshots under {repo_root}/database/releases")
    tags = package_tags(repo_root)
    surfaces = {tag: surface_at_tag(repo_root, tag) for tag in tags}
    verdicts = {
        (tag, snapshot.release): verdict(
            snapshot, load_closures(repo_root, snapshot.release), surfaces[tag]
        )
        for tag in tags
        for snapshot in snapshots
    }
    return render_page(snapshots, tags, verdicts)


def main(argv: list[str] | None = None) -> int:
    """Write the page, or with ``--check`` report whether it is current."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo-root", default=str(REPO))
    parser.add_argument("--output", default=None, help=f"default {OUTPUT_RELPATH}")
    parser.add_argument(
        "--check", action="store_true", help="exit 1 if the page would change"
    )
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    output = Path(args.output) if args.output else repo_root / OUTPUT_RELPATH
    page = build_page(repo_root)
    current = output.read_text(encoding="utf-8") if output.exists() else None
    if args.check:
        if current == page:
            print(f"{output}: current")
            return 0
        print(f"{output}: stale; run python {SCRIPT_RELPATH}")
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
