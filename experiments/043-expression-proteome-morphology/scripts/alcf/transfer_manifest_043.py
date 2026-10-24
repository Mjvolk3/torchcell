# experiments/043-expression-proteome-morphology/scripts/alcf/transfer_manifest_043.py
# [[experiments.043-expression-proteome-morphology.scripts.alcf.transfer_manifest_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/alcf/transfer_manifest_043.py
"""What experiment 043 sends to ALCF (Polaris, Aurora), with sizes and sha256.

Two tiers, both relative to ``$DATA_ROOT`` so the remote tree mirrors the local layout
under ``<project>/mjvolk3/torchcell-data/043/``:

``train``    every file the ``train`` stage of ``train_043.py`` opens: the prepared store
             (``train_043_prep/prep_v2_routeA`` for store v1, ``prep_v3_routeA`` for
             v2; the prep_v1 store sent on 2026-10-23 stays on the remote side), the vocabularies (``representation_043``)
             and the four frozen allele tables (``isolate_embeddings/peter2018_1011/
             alleles/<encoder>.h5``, which also hold the Vaishnav promoter and reporter
             rows), plus the table manifest and verify report. The split file
             ``conf/splits/global_split_v1.json`` travels with the code tree.
``prepare``  what ``--stage prepare`` additionally reads, so the store can be re-prepared
             remotely: ``raw_043/raw`` (LMDB + indexes), ``raw_043/processed`` metadata
             (``gene_set.json`` and the indexes, not its LMDB), the isolate ``index`` and
             ``sequences``, and the genome / GO / STRING / TFLink roots of ``_edge_lists``.

Excluded and listed with the reason: ``isolate_embeddings/.../work`` (reclaimable embedding
intermediates), ``raw_043/aggregation/lmdb`` and ``raw_043/processed/lmdb`` (163 GB each;
no stage of the trainer opens them), and the analysis caches.

Files under 2 GB carry a sha256; larger files carry their size only. ``build`` also writes,
per tier, an rsync ``--files-from`` list and a ``sha256sum -c`` checklist beside the
manifest.

Store generation (``store_version_043``, env ``STORE_043_VERSION``): v1 lists
``prep_v2_routeA``, ``representation_043`` and ``raw_043`` and writes
``transfer_manifest_043.json`` and ``{files,sha256,sizes}_<tier>_043.txt``; v2 lists
``prep_v3_routeA``, ``representation_043_v2`` and ``raw_043_v2`` and writes the same
names with ``_v2`` (``transfer_manifest_043_v2.json``, ``files_train_043_v2.txt``), so
the v1 lists of a transfer still in flight are never rewritten.

    python experiments/043-expression-proteome-morphology/scripts/alcf/transfer_manifest_043.py build
    STORE_043_VERSION=v2 python experiments/043-expression-proteome-morphology/scripts/alcf/transfer_manifest_043.py build
"""

from __future__ import annotations

import argparse
import hashlib
import os
import os.path as osp
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.dirname(HERE))
from store_version_043 import StorePaths043  # noqa: E402
from store_version_043 import active as active_store  # noqa: E402

EXP = "data/torchcell/experiments/043-expression-proteome-morphology"
ISO = "data/scerevisiae/isolate_embeddings/peter2018_1011"
SHA_LIMIT = 2_000_000_000

Tier = Literal["train", "prepare"]


# (tier, path relative to DATA_ROOT, file or directory)
def sources(store: StorePaths043) -> list[tuple[Tier, str]]:
    """The tiers of one store generation."""
    return [
        ("train", f"{EXP}/train_043_prep/{store.prep_version}_routeA"),
        ("train", f"{EXP}/{store.representation_tag}"),
        ("train", f"{ISO}/alleles"),
        ("train", f"{ISO}/manifest.json"),
        ("train", f"{ISO}/verify_report.json"),
        ("prepare", f"{EXP}/{store.raw_tag}/raw"),
        ("prepare", f"{EXP}/{store.raw_tag}/processed"),
        ("prepare", f"{ISO}/index"),
        ("prepare", f"{ISO}/sequences"),
        ("prepare", "data/sgd/genome"),
        ("prepare", "data/go"),
        ("prepare", "data/string"),
        ("prepare", "data/tflink"),
    ]


def excluded(store: StorePaths043) -> dict[str, str]:
    """Paths under the sources that are not sent, with the reason."""
    return {
        f"{ISO}/work": "reclaimable embedding intermediates; no stage reads them",
        f"{EXP}/{store.raw_tag}/aggregation/lmdb": "LMDB of about 160 GB; neither the "
        "prepare nor the train stage opens it",
        f"{EXP}/{store.raw_tag}/processed/lmdb": "LMDB of about 160 GB; prepare reads "
        "raw/lmdb, not this one",
        f"{EXP}/cross_dataset_cache": "analysis cache of cross_dataset_correlation.py",
        f"{EXP}/{store.linear_evals_cache}": "analysis cache of linear_evals_043.py",
    }


class ManifestFile(BaseModel):
    """One file to transfer."""

    model_config = ConfigDict(frozen=True)

    tier: Tier
    path: str
    bytes: int
    sha256: str | None


class TierTotal(BaseModel):
    """Totals of one tier."""

    tier: Tier
    n_files: int
    bytes: int
    gb: float
    n_sha256: int


class TransferManifest043(BaseModel):
    """The whole manifest."""

    model_config = ConfigDict(frozen=True)

    generated_at: str
    data_root: str
    remote_subdir: str
    sha256_limit_bytes: int
    totals: list[TierTotal]
    excluded: dict[str, str]
    files: list[ManifestFile]


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _walk(data_root: str, rel: str, skip: dict[str, str]) -> list[str]:
    full = osp.join(data_root, rel)
    if osp.isfile(full):
        return [rel]
    if not osp.isdir(full):
        raise FileNotFoundError(full)
    out = []
    for dirpath, dirnames, filenames in os.walk(full):
        rel_dir = osp.relpath(dirpath, data_root)
        dirnames[:] = sorted(d for d in dirnames if osp.join(rel_dir, d) not in skip)
        for name in sorted(filenames):
            out.append(osp.join(rel_dir, name))
    return out


def build(
    data_root: str, out_dir: str, workers: int, store: StorePaths043
) -> TransferManifest043:
    """Walk the sources, hash the small files, write the manifest and the lists."""
    skip = excluded(store)
    entries: list[tuple[Tier, str, int]] = []
    for tier, rel in sources(store):
        for f in _walk(data_root, rel, skip):
            if f.endswith(".lock"):
                continue
            entries.append((tier, f, osp.getsize(osp.join(data_root, f))))

    def one(e: tuple[Tier, str, int]) -> ManifestFile:
        tier, rel, size = e
        sha = _sha256(osp.join(data_root, rel)) if size < SHA_LIMIT else None
        return ManifestFile(tier=tier, path=rel, bytes=size, sha256=sha)

    with ThreadPoolExecutor(workers) as pool:
        files = list(pool.map(one, entries))

    totals = []
    for tier in ("train", "prepare"):
        sel = [f for f in files if f.tier == tier]
        b = sum(f.bytes for f in sel)
        totals.append(
            TierTotal(
                tier=tier,
                n_files=len(sel),
                bytes=b,
                gb=round(b / 1e9, 3),
                n_sha256=sum(f.sha256 is not None for f in sel),
            )
        )
        with open(osp.join(out_dir, f"files_{tier}_043{store.suffix}.txt"), "w") as fh:
            fh.writelines(f"{f.path}\n" for f in sel)
        with open(osp.join(out_dir, f"sha256_{tier}_043{store.suffix}.txt"), "w") as fh:
            fh.writelines(f"{f.sha256}  {f.path}\n" for f in sel if f.sha256)
        with open(osp.join(out_dir, f"sizes_{tier}_043{store.suffix}.txt"), "w") as fh:
            fh.writelines(f"{f.bytes} {f.path}\n" for f in sel)

    manifest = TransferManifest043(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        data_root=data_root,
        remote_subdir="torchcell-data/043",
        sha256_limit_bytes=SHA_LIMIT,
        totals=totals,
        excluded=skip,
        files=files,
    )
    with open(
        osp.join(out_dir, f"transfer_manifest_043{store.suffix}.json"), "w"
    ) as fh:
        fh.write(manifest.model_dump_json(indent=1))
    return manifest


def main() -> None:
    """CLI."""
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build"])
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    m = build(os.environ["DATA_ROOT"], HERE, args.workers, active_store())
    for t in m.totals:
        print(f"{t.tier}: {t.n_files} files, {t.gb} GB, {t.n_sha256} with sha256")


if __name__ == "__main__":
    main()
