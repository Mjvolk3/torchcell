# experiments/010-kuzmin-tmi/scripts/migrate_010_build_schema.py
# [[experiments.010-kuzmin-tmi.scripts.migrate_010_build_schema]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/migrate_010_build_schema
"""Carry the frozen 010 build onto the current schema, position for position.

The 010 build (``001-small-build``, 376,732 trigenic records) was made in
December 2025 and no longer validates. Two later changes broke it:

    933ee5ef3  2026-07-08  perturbation ontology refactor renamed the deletion
                           literals: ``deletion`` became ``sga_kanmx_deletion``
                           or ``mean_deletion``
    1cf60cdc5  2026-07-14  ``Media.is_synthetic`` became a required field

Both are pure vocabulary. Neither changes what was measured, which gene was
perturbed, or what the label is, so a record can be carried across by rewriting
those fields and nothing else. Rebuilding from the served graph instead would
change the record ORDER, and every published 010 artifact is position-keyed:
``index_seed_42.json``, ``label_df.parquet``,
``is_any_perturbed_gene_index.json`` and the saved per-record prediction files
all address records by position. So this migration is a key-preserving copy, and
the check that it worked is that every position still carries the same three
perturbed genes and the same label.

What this does NOT do, deliberately:

* It does not correct the environment. The frozen build records YEPD at 30 C.
  The Kuzmin loaders now emit SGA triple-mutant selection medium at 26 C, which
  is the corrected reading of the screen. Applying that here would change the
  experimental content of records whose published results are being preserved.
  A build with the corrected environment is a REBUILD, and a different data
  version, not this.
* It does not touch the source. The frozen build's files are owned by the
  knowledge-graph build user and are hardlinked into the graph tree, so the
  output goes to a new sibling directory and the original is opened read-only.

The migrated build carries a ``build_provenance.json`` naming its source, the
source LMDB sha256, the exact field rewrites applied, and the schema fingerprint
it validates against, so a later reader can tell which data version a run used.
"""

import argparse
import hashlib
import json
import os
import os.path as osp
import shutil
import socket
import subprocess
from datetime import UTC, datetime

import lmdb
from dotenv import load_dotenv

from torchcell.datamodels.schema import (
    EXPERIMENT_REFERENCE_TYPE_MAP,
    EXPERIMENT_TYPE_MAP,
)

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

SOURCE_BUILD = osp.join(
    DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/001-small-build"
)
DEST_BUILD = osp.join(
    DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/001-small-build-schema-v2"
)
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")

# Position-keyed sidecars that must be carried across unchanged. Copying rather
# than regenerating them is the point: a regenerated index could reorder.
SIDECARS = [
    "processed/label_df.parquet",
    "processed/is_any_perturbed_gene_index.json",
    "processed/dataset_name_index.json",
    "processed/perturbation_count_index.json",
    "processed/phenotype_label_index.json",
    "processed/experiment_types.json",
    "processed/gene_set.json",
    "data_module_cache/index_seed_42.json",
    "data_module_cache/index_details_seed_42.json",
]

# The whole migration, stated as data. Every rewrite the script performs is here
# and nothing else is touched.
DELETION_TYPE_TO_LITERAL = {
    "KanMX": "sga_kanmx_deletion",
    "mean": "mean_deletion",
}
# YEPD is a complex medium, not chemically defined (torchcell/datamodels/media.py:243).
MEDIA_IS_SYNTHETIC = {"YEPD": False}

MAP_SIZE = 8 * 1024**3


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def migrate_perturbation(pert: dict) -> tuple[dict, bool]:
    """Rewrite a perturbation's type literal. Returns (perturbation, changed)."""
    if pert.get("perturbation_type") != "deletion":
        return pert, False
    deletion_type = pert.get("deletion_type")
    assert deletion_type in DELETION_TYPE_TO_LITERAL, (
        f"unmapped deletion_type {deletion_type!r}; the migration table must "
        "cover every value present in the build"
    )
    pert["perturbation_type"] = DELETION_TYPE_TO_LITERAL[deletion_type]
    return pert, True


def migrate_media(env: dict) -> bool:
    """Add the required is_synthetic to a media dict. Returns whether it changed."""
    media = env["media"]
    if "is_synthetic" in media:
        return False
    name = media["name"]
    assert name in MEDIA_IS_SYNTHETIC, (
        f"unmapped media {name!r}; is_synthetic must be sourced, never guessed"
    )
    media["is_synthetic"] = MEDIA_IS_SYNTHETIC[name]
    return True


def migrate_record(data_list: list[dict]) -> tuple[list[dict], dict[str, int]]:
    counts = {"perturbation_type": 0, "media_is_synthetic": 0}
    for item in data_list:
        for side in ("experiment", "experiment_reference"):
            block = item[side]
            env_key = "environment" if side == "experiment" else "environment_reference"
            if env_key not in block:
                env_key = "environment"
            counts["media_is_synthetic"] += int(migrate_media(block[env_key]))
            genotype_key = "genotype" if side == "experiment" else "genotype_reference"
            if genotype_key not in block:
                continue
            genotype = block[genotype_key]
            for pert in genotype.get("perturbations", []):
                _, changed = migrate_perturbation(pert)
                counts["perturbation_type"] += int(changed)
    return data_list, counts


def genes_and_label(data_list: list[dict]) -> tuple[tuple[str, ...], float]:
    """The identity check: perturbed gene set and label at this position."""
    item = data_list[0]
    genes = tuple(
        sorted(
            p["systematic_gene_name"]
            for p in item["experiment"]["genotype"]["perturbations"]
        )
    )
    phen = item["experiment"]["phenotype"]
    label = phen.get("gene_interaction")
    return genes, label


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="migrate only N records")
    ap.add_argument(
        "--validate-every",
        type=int,
        default=1,
        help="validate every Nth record through the current pydantic schema",
    )
    args = ap.parse_args()

    src_lmdb = osp.join(SOURCE_BUILD, "processed", "lmdb")
    src_mdb = osp.join(src_lmdb, "data.mdb")
    print(f"source {src_lmdb}")
    src_sha = sha256_file(src_mdb)
    print(f"source data.mdb sha256 {src_sha}")

    dst_lmdb = osp.join(DEST_BUILD, "processed", "lmdb")
    os.makedirs(dst_lmdb, exist_ok=True)

    src_env = lmdb.open(src_lmdb, readonly=True, lock=False, subdir=True)
    dst_env = lmdb.open(dst_lmdb, map_size=MAP_SIZE, subdir=True)

    totals = {"perturbation_type": 0, "media_is_synthetic": 0}
    n_records = 0
    n_validated = 0
    identity: dict[int, tuple[tuple[str, ...], float]] = {}

    with src_env.begin() as rtxn, dst_env.begin(write=True) as wtxn:
        cursor = rtxn.cursor()
        for key, value in cursor:
            data_list = json.loads(value.decode())
            before = genes_and_label(data_list)
            migrated, counts = migrate_record(data_list)
            for k, v in counts.items():
                totals[k] += v
            after = genes_and_label(migrated)
            assert before == after, (
                f"record {key!r} identity changed during migration: {before} -> {after}"
            )
            identity[int(key.decode())] = after

            if n_records % args.validate_every == 0:
                for item in migrated:
                    exp_cls = EXPERIMENT_TYPE_MAP[item["experiment"]["experiment_type"]]
                    ref_cls = EXPERIMENT_REFERENCE_TYPE_MAP[
                        item["experiment_reference"]["experiment_reference_type"]
                    ]
                    exp_cls(**item["experiment"])
                    ref_cls(**item["experiment_reference"])
                n_validated += 1

            wtxn.put(key, json.dumps(migrated).encode())
            n_records += 1
            if n_records % 25000 == 0:
                print(f"  {n_records} records")
            if args.limit is not None and n_records >= args.limit:
                break

    src_env.close()
    dst_env.close()

    print(f"\nmigrated {n_records} records, validated {n_validated}")
    print(f"  perturbation_type rewrites  {totals['perturbation_type']}")
    print(f"  media is_synthetic added    {totals['media_is_synthetic']}")

    if args.limit is not None:
        print("\npartial run, sidecars and provenance not written")
        return

    for rel in SIDECARS:
        src = osp.join(SOURCE_BUILD, rel)
        dst = osp.join(DEST_BUILD, rel)
        os.makedirs(osp.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  copied {rel}")

    # The order guarantee, checked rather than asserted: the split file's
    # positions must still name the same genes in the migrated build.
    with open(osp.join(DEST_BUILD, "data_module_cache", "index_seed_42.json")) as f:
        split = json.load(f)
    n_split = sum(len(v) for v in split.values())
    assert n_split == n_records, f"split covers {n_split} of {n_records} records"
    assert max(max(v) for v in split.values()) == n_records - 1

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    provenance = {
        "build": "010-kuzmin-tmi/001-small-build-schema-v2",
        "derived_from": {
            "path": SOURCE_BUILD,
            "processed_lmdb_sha256": src_sha,
            "note": (
                "frozen December 2025 build; owned by the knowledge-graph build "
                "user and hardlinked into the graph tree, opened read-only here"
            ),
        },
        "migration": {
            "reason": (
                "the frozen build predates the perturbation-ontology refactor "
                "(933ee5ef3) and the required Media.is_synthetic field "
                "(1cf60cdc5), so it no longer validates"
            ),
            "perturbation_type_map": DELETION_TYPE_TO_LITERAL,
            "media_is_synthetic": MEDIA_IS_SYNTHETIC,
            "fields_rewritten": totals,
            "not_changed": [
                "record order and LMDB keys",
                "perturbed genes and labels at every position",
                "environment name, state and temperature (YEPD, 30 C), which "
                "the current Kuzmin loaders would emit as SGA triple-mutant "
                "selection at 26 C; correcting that is a rebuild, not this",
            ],
        },
        "records": n_records,
        "records_validated": n_validated,
        "migrated_at": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "torchcell_commit": commit,
    }
    prov_path = osp.join(DEST_BUILD, "build_provenance.json")
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"\nwrote {prov_path}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report = osp.join(RESULTS_DIR, "migrate_010_build_schema_report.json")
    with open(report, "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
