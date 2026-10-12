# experiments/036-dataset-fixes-before-kg-build/scripts/verification_overlay.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.verification_overlay]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/verification_overlay.py
"""Build a DATA_ROOT overlay so a verification rerun writes nothing into the dev tree.

Every verification route writes ``preprocess/verification_report*.json`` beside the store
it read. A rerun under unlanded code must not do that in the shared dev tree: other
branches' tests read those files, and a report carrying a field ``main`` does not know
(``LevelResult.finding_class``) fails to load there. The overlay mirrors ``DATA_ROOT``
with symlinks, except that every ``data/torchcell/<store>/preprocess`` is a real
directory whose entries are symlinks to the store's own, minus every
``verification_report*``: a report written into the overlay is a new file there, and
every other file is read through to the live store. A verifier writes nothing but its
reports (``runners._write_report`` and the loaders' ``verify*`` entry points; the
loaders' other ``_write_*`` helpers run at build time), so nothing else is written
through. Linking rather than copying keeps the store's ``build_manifest.json`` live, so
the sweep still sees a store rebuilt while it is verified.

Usage::

    python experiments/036-dataset-fixes-before-kg-build/scripts/verification_overlay.py \
        --source /scratch/projects/torchcell-scratch --overlay <dir>
"""

from __future__ import annotations

import argparse
import os
import os.path as osp

STORE_PARENT = osp.join("data", "torchcell")


def link_children(source: str, target: str, skip: frozenset[str]) -> None:
    """Symlink every entry of ``source`` into ``target`` except the names in ``skip``."""
    os.makedirs(target, exist_ok=True)
    for name in sorted(os.listdir(source)):
        if name not in skip:
            os.symlink(osp.join(source, name), osp.join(target, name))


def overlay_preprocess(source: str, target: str) -> int:
    """A real ``preprocess`` dir of links to the store's entries, its reports dropped."""
    os.makedirs(target)
    dropped = 0
    for name in sorted(os.listdir(source)):
        if name.startswith("verification_report"):
            dropped += 1
            continue
        os.symlink(osp.join(source, name), osp.join(target, name))
    return dropped


def build_overlay(source: str, overlay: str) -> dict[str, int]:
    """Mirror ``source`` at ``overlay``; return counts of stores and reports left out."""
    if osp.exists(overlay):
        raise FileExistsError(f"{overlay} exists; an overlay is built once, fresh")
    link_children(source, overlay, frozenset({"data"}))
    link_children(
        osp.join(source, "data"), osp.join(overlay, "data"), frozenset({"torchcell"})
    )
    stores_src = osp.join(source, STORE_PARENT)
    stores_dst = osp.join(overlay, STORE_PARENT)
    os.makedirs(stores_dst)
    n_stores = 0
    n_dropped = 0
    for slug in sorted(os.listdir(stores_src)):
        src = osp.join(stores_src, slug)
        if not osp.isdir(src) or osp.islink(src):
            os.symlink(src, osp.join(stores_dst, slug))
            continue
        link_children(src, osp.join(stores_dst, slug), frozenset({"preprocess"}))
        if osp.isdir(osp.join(src, "preprocess")):
            n_dropped += overlay_preprocess(
                osp.join(src, "preprocess"), osp.join(stores_dst, slug, "preprocess")
            )
        n_stores += 1
    return {"stores": n_stores, "reports_left_out": n_dropped}


def main(argv: list[str] | None = None) -> int:
    """Build the overlay and print its counts."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True)
    parser.add_argument("--overlay", required=True)
    args = parser.parse_args(argv)
    print(build_overlay(args.source, args.overlay))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
