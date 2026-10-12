# experiments/027-betaxanthin-metabolic/scripts/verify_kcat_table.py
# [[experiments.027-betaxanthin-metabolic.wave7-parity-one-table]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/verify_kcat_table.py

r"""Resolve the kcat table the trainer would build RIGHT NOW, and hash what it read.

WHY. Wave 6's six betaxanthin parity seeds ran on three different kcat tables, because the
resolver reads predicted-parameter parquets off the ``$DATA_ROOT`` mirror and that mirror
was regenerated while the queue drained. Freezing the CODE did not catch it: two of the
three tables came out of a byte-identical ``train_flux.py``. The only thing that separates
them is the table itself, so wave 7 resolves the table BEFORE it submits anything, checks
the tier counts against the shipped verified table, and records the path plus sha256 of
every file the resolution read.

WHAT IT DOES NOT DO. It does not redirect the resolver. The trainer still reads the live
mirror; this script measures what is there and copies it aside, so a later mismatch is
detectable and the run is rebuildable. Detection, not a fallback.

Run from the repo root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/verify_kcat_table.py

Writes ``results/wave7_kcat_table_verification.json`` and copies the hashed files into
``$DATA_ROOT/frozen-027-wave7-kinetics/``. Exits non-zero when the resolved tier counts are
not the shipped ones, which is the gate before a cell is submitted.
"""

import argparse
import hashlib
import os
import os.path as osp
import shutil
from datetime import UTC, datetime

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from torchcell.metabolism.constraints import build_gem_tensors
from torchcell.metabolism.parameters import (
    KINETICS_MIRROR,
    PLAUSIBILITY_RESULT,
    VERIFIED_PAIRS_FEATURE,
    SubstrateMatch,
    load_predicted_unit_table,
    resolve_kcat_table,
    resolve_km_table,
)
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.timestamp import timestamp

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "results")
FROZEN_DIR = osp.join(DATA_ROOT, "frozen-027-wave7-kinetics")

#: The shipped verified table, as reported in
#: [[experiments.026-metabolism-flux.kinetic-coverage-closure]] and carried by the wave-6
#: projection cells. Every wave-7 cell must resolve exactly this.
SHIPPED_TIER_COUNTS = {"measured": 97, "homolog": 782, "predicted": 2831, "default": 18}

#: The Open Enzyme Database mirrors the measured and homolog tiers read, relative to
#: ``$DATA_ROOT``. Both directories hold one records file and one manifest.
OED_SCEREVISIAE = "data/enzyme_kinetics/open_enzyme_database/scerevisiae"
OED_ALL = "data/enzyme_kinetics/open_enzyme_database/all"


class MirrorFile(BaseModel):
    """One file the resolution read, pinned by content rather than by path."""

    model_config = ConfigDict(extra="forbid")

    role: str
    path: str
    sha256: str
    bytes: int
    mtime_utc: str
    frozen_copy: str | None = None


class KcatTableVerification(BaseModel):
    """What the resolver produced, and everything it read to produce it."""

    model_config = ConfigDict(extra="forbid")

    generated_at: str
    kcat_tier: str
    kcat_homolog_tier: bool
    kinetics_version: str
    n_catalytic_units: int
    tier_counts: dict[str, int]
    expected_tier_counts: dict[str, int]
    matches_shipped: bool
    km_tier_counts: dict[str, int]
    predicted_kcat_predictors: list[str]
    predicted_km_predictors: list[str]
    kcat_notes: str
    source_files: list[MirrorFile] = Field(default_factory=list)
    frozen_copy_dir: str


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(role: str, path: str, freeze: bool) -> MirrorFile:
    stat = os.stat(path)
    copy_path: str | None = None
    if freeze:
        rel = (
            osp.relpath(path, DATA_ROOT)
            if path.startswith(DATA_ROOT)
            else osp.basename(path)
        )
        copy_path = osp.join(FROZEN_DIR, rel)
        os.makedirs(osp.dirname(copy_path), exist_ok=True)
        shutil.copy2(path, copy_path)
    return MirrorFile(
        role=role,
        path=path,
        sha256=_sha256(path),
        bytes=stat.st_size,
        mtime_utc=datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
        frozen_copy=copy_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-freeze",
        action="store_true",
        help="Hash the mirror files without copying them into the frozen directory.",
    )
    parser.add_argument("--out", default="wave7_kcat_table_verification.json")
    args = parser.parse_args()
    freeze = not args.no_freeze

    source = YeastGEM()
    gem = build_gem_tensors(source.model, model_dir=source.model_dir)
    units = gem.catalytic_units
    kinetics_dir = osp.join(DATA_ROOT, KINETICS_MIRROR)
    verified_pairs = osp.join(kinetics_dir, VERIFIED_PAIRS_FEATURE)
    plausibility = osp.join(EXPERIMENT_ROOT, PLAUSIBILITY_RESULT)
    oed_scerevisiae = osp.join(DATA_ROOT, OED_SCEREVISIAE)
    oed_all = osp.join(DATA_ROOT, OED_ALL)

    substrates = SubstrateMatch.from_verified_pairs(verified_pairs)
    predicted_kcat = load_predicted_unit_table(
        "k_cat", units.n_units, kinetics_dir, plausibility
    )
    predicted_km = load_predicted_unit_table(
        "K_M", units.n_units, kinetics_dir, plausibility
    )
    km = resolve_km_table(
        units,
        source.model_dir,
        oed_scerevisiae,
        predicted=predicted_km,
        substrates=substrates,
        homolog_mirror_dir=oed_all,
    )
    kcat = resolve_kcat_table(
        units,
        source.model_dir,
        oed_scerevisiae,
        predicted=predicted_kcat,
        substrates=substrates,
        homolog_mirror_dir=oed_all,
    )
    counts = kcat.tier_counts()

    files: list[MirrorFile] = [
        _record("verified_pairs", verified_pairs, freeze),
        _record("plausibility_study", plausibility, freeze),
    ]
    for name in ("oed_records.json", "manifest.json"):
        files.append(
            _record(f"oed_scerevisiae/{name}", osp.join(oed_scerevisiae, name), freeze)
        )
        files.append(_record(f"oed_all/{name}", osp.join(oed_all, name), freeze))
    for parameter, table in (("k_cat", predicted_kcat), ("K_M", predicted_km)):
        for predictor, path in sorted(table.source_paths.items()):
            record = _record(f"predicted/{parameter}/{predictor}", path, freeze)
            assert record.sha256 == table.sha256[predictor], (
                f"{path} changed between the resolver reading it and this hash; the mirror "
                "is being written while the table is resolved."
            )
            files.append(record)

    report = KcatTableVerification(
        generated_at=timestamp(),
        kcat_tier="predicted",
        kcat_homolog_tier=True,
        kinetics_version=predicted_kcat.version,
        n_catalytic_units=int(units.n_units),
        tier_counts=counts,
        expected_tier_counts=SHIPPED_TIER_COUNTS,
        matches_shipped=counts == SHIPPED_TIER_COUNTS,
        km_tier_counts=km.tier_counts(),
        predicted_kcat_predictors=list(predicted_kcat.selection.selected),
        predicted_km_predictors=list(predicted_km.selection.selected),
        kcat_notes=kcat.notes,
        source_files=files,
        frozen_copy_dir=FROZEN_DIR if freeze else "",
    )
    out_path = osp.join(RESULTS, args.out)
    with open(out_path, "w") as fh:
        fh.write(report.model_dump_json(indent=2))
    print(f"wrote {out_path}")
    print(f"resolved kcat tier counts: {counts}")
    print(f"shipped verified table:    {SHIPPED_TIER_COUNTS}")
    for record in files:
        print(f"  {record.role:34s} {record.sha256[:16]} {record.path}")
    if not report.matches_shipped:
        raise SystemExit(
            "the resolved kcat table is NOT the shipped verified one. Submitting a cell "
            "now would put a fourth table into the parity contrast."
        )
    print("MATCH: the trainer will build the shipped verified table.")


if __name__ == "__main__":
    main()
