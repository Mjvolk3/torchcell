# torchcell/artifacts/__main__.py
# [[torchcell.artifacts.__main__]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/artifacts/__main__.py
# Test file: tests/torchcell/artifacts/test_artifact_cli.py

"""``python -m torchcell.artifacts``: resolve, check and deposit from the shell.

- ``resolve <tc-uri> --sha256 <hex> [--no-materialize]`` prints the
  ``ResolvedArtifact`` as JSON.
- ``check <tc-uri> --sha256 <hex>`` prints ``resolves`` or ``unresolvable`` and exits
  0 or 1.
- ``deposit <dir> --key <k> [--tier objects|raw] [--processing <json>]
  [--allow-change]`` prints the written manifest's file count and path.

``--data-root`` overrides ``DATA_ROOT`` for every subcommand. Errors propagate with
their own messages (an unresolvable ref, an integrity disagreement).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from torchcell.artifacts.deposit import DEPOSIT_TIERS, deposit
from torchcell.artifacts.ref import ArtifactRef
from torchcell.artifacts.resolve import check, resolve
from torchcell.artifacts.tiers import key_dir
from torchcell.literature.manifest import MANIFEST_FILENAME, ProcessingRecord


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m torchcell.artifacts")
    parser.add_argument("--data-root", default=None, help="Overrides DATA_ROOT.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("resolve", "check"):
        cmd = sub.add_parser(name, help=f"{name} a tc:// artifact ref")
        cmd.add_argument("uri", help="tc://<tier>/<key>/<path>[#<member>]")
        cmd.add_argument("--sha256", required=True, help="sha256 of the file at path")
        if name == "resolve":
            cmd.add_argument(
                "--no-materialize",
                action="store_true",
                help="Consult manifests only; download and hash nothing.",
            )
    dep = sub.add_parser("deposit", help="deposit a directory into a tier")
    dep.add_argument("directory", type=Path)
    dep.add_argument("--key", required=True)
    dep.add_argument("--tier", default="objects", choices=sorted(DEPOSIT_TIERS))
    dep.add_argument(
        "--processing",
        type=Path,
        default=None,
        help="A ProcessingRecord JSON attached to every new or changed record.",
    )
    dep.add_argument("--allow-change", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand; return the process exit code."""
    args = _parser().parse_args(argv)
    if args.command in ("resolve", "check"):
        ref = ArtifactRef.parse(args.uri, sha256=args.sha256)
        if args.command == "check":
            ok = check(ref, data_root=args.data_root)
            print("resolves" if ok else "unresolvable")
            return 0 if ok else 1
        resolved = resolve(
            ref, materialize=not args.no_materialize, data_root=args.data_root
        )
        print(resolved.model_dump_json())
        return 0
    processing = (
        ProcessingRecord.model_validate_json(args.processing.read_text())
        if args.processing is not None
        else None
    )
    manifest = deposit(
        args.directory,
        tier=args.tier,
        key=args.key,
        processing=processing,
        data_root=args.data_root,
        allow_change=args.allow_change,
    )
    dest = key_dir(args.tier, args.key, args.data_root) / MANIFEST_FILENAME
    print(f"{len(manifest.files)} files -> {dest}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
