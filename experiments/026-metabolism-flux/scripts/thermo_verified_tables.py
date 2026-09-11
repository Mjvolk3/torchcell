# experiments/026-metabolism-flux/scripts/thermo_verified_tables.py
# [[experiments.026-metabolism-flux.scripts.thermo_verified_tables]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/thermo_verified_tables.py

r"""Write the curated metabolite and reaction tables as NEW versioned files.

``metabolite_dgf_prime_with_external.parquet`` is the table the flux layer reads under
``ThermoSource.UNION_EXTERNAL``, and runs are training against it right now. This script
therefore writes NOTHING over it. It writes a new version, under ``verified/``, alongside a
manifest that names the sha256 of the file each one supersedes, which is the same pattern
``improve_kinetic_coverage.write_verified_tables`` uses for the kinetic tables: both
versions stay on disk, and a run records which it consumed rather than inheriting whichever
was last written.

Reads ``results/thermo_pipeline_yeast9.json`` (whatever the last
``thermo_pipeline_yeast9.py`` run wrote, with or without ``--curation-dir``) and writes,
under ``$DATA_ROOT/data/torchcell/thermo_equilibrator/verified/``:

* ``metabolite_dgf_prime_with_curation.parquet`` -- one row per PRICED metabolite, the same
  columns ``thermo_union_reprice.py`` writes so a reader can swap the file and nothing else;
* ``reaction_drg_prime_with_curation.parquet`` -- one row per reaction, priced or not, with
  the route, the reason, the cancelled participants and the non-chemical flag, which the
  older reaction tables have no column for;
* ``manifest.json`` -- what was built, from what, and what it supersedes.

Run from the repository root under the torchcell env::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/thermo_verified_tables.py
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
from datetime import UTC, datetime
from typing import cast

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from torchcell.metabolism.thermo_pipeline import (
    ThermoPipelineResult,
    read_result,
    sha256_of,
)

load_dotenv()

DATA_ROOT = cast(str, os.environ["DATA_ROOT"])
EXPERIMENT_ROOT = cast(str, os.environ["EXPERIMENT_ROOT"])
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
THERMO_DIR = osp.join(DATA_ROOT, "data", "torchcell", "thermo_equilibrator")
VERIFIED_DIR = osp.join(THERMO_DIR, "verified")
PIPELINE_JSON = osp.join(RESULTS, "thermo_pipeline_yeast9.json")

#: The files these supersede. Named rather than overwritten: three training runs are
#: reading the first of them as this is written.
SUPERSEDES = {
    "metabolite_dgf_prime_with_curation.parquet": osp.join(
        THERMO_DIR, "metabolite_dgf_prime_with_external.parquet"
    ),
    "reaction_drg_prime_with_curation.parquet": osp.join(
        THERMO_DIR, "reaction_drg_prime_yeast9_conditions.parquet"
    ),
}


class VerifiedTableManifest(BaseModel):
    """What the verified tables contain, what built them, and what they replace."""

    built_at: str
    pipeline_result_path: str
    pipeline_result_sha256: str
    gem_path: str
    gem_sha256: str
    stages: list[str] = Field(default_factory=list)
    curation_dir: str = ""
    curation_record_sha256: dict[str, str] = Field(default_factory=dict)
    tables: list[dict[str, object]] = Field(default_factory=list)
    n_metabolites_priced: int = 0
    n_reactions_priced: int = 0
    n_reactions_chemical: int = 0
    n_reactions_chemical_priced: int = 0
    n_reactions_non_chemical: int = 0
    n_reactions_cancelled: int = 0
    routes: dict[str, int] = Field(default_factory=dict)
    note: str = (
        "A NEW version rather than an overwrite. The shipped tables named under "
        "supersedes_path are unchanged and every run that read them still reproduces."
    )


def metabolite_frame(result: ThermoPipelineResult) -> pd.DataFrame:
    """One row per priced metabolite, in the shape the flux layer's reader expects.

    Only priced rows: an unpriced row with a NaN is read as a value by anything that
    forgets to check the mask, and this table is read by something that does exactly that
    unless it is handed only real values.
    """
    return pd.DataFrame(
        [
            {
                "met_id": price.met_id,
                "dgf_prime_kj_per_mol": price.dgf_prime_kj_per_mol,
                "sigma_kj_per_mol": price.sigma_kj_per_mol,
                "route": str(price.route),
                "p_h": price.p_h,
                "ionic_strength_m": price.ionic_strength_m,
                "p_mg": price.p_mg,
                "temperature_k": price.temperature_k,
                "detail": price.detail,
                "curation_slug": str(price.provenance.get("slug", "")),
            }
            for price in result.metabolites
            if price.priced
        ]
    )


def reaction_frame(result: ThermoPipelineResult) -> pd.DataFrame:
    """One row per reaction, priced or not, with why and by which route."""
    return pd.DataFrame(
        [
            {
                "reaction_id": reaction.reaction_id,
                "delta_r_g0_kj_per_mol": reaction.delta_r_g0_kj_per_mol,
                "sigma_kj_per_mol": reaction.sigma_kj_per_mol,
                "route": str(reaction.route),
                "reason": str(reaction.reason) if reaction.reason else "",
                "n_participants": reaction.n_participants,
                "n_cancelled_participants": len(reaction.cancelled_met_ids),
                "cancelled_met_ids": ";".join(reaction.cancelled_met_ids),
                "non_chemical": reaction.non_chemical,
                "blocking_met_ids": ";".join(reaction.blocking_met_ids),
                "curation_slug": str(reaction.provenance.get("slug", "")),
            }
            for reaction in result.reactions
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-result", default=PIPELINE_JSON)
    parser.add_argument("--out-dir", default=VERIFIED_DIR)
    parser.add_argument(
        "--curation-dir",
        default=osp.join(RESULTS, "curation"),
        help="hashed into the manifest so the tables name the records behind them",
    )
    args = parser.parse_args()

    result = read_result(args.pipeline_result)
    os.makedirs(args.out_dir, exist_ok=True)

    tables: list[dict[str, object]] = []
    for name, frame in (
        ("metabolite_dgf_prime_with_curation.parquet", metabolite_frame(result)),
        ("reaction_drg_prime_with_curation.parquet", reaction_frame(result)),
    ):
        path = osp.join(args.out_dir, name)
        frame.to_parquet(path, index=False)
        superseded = SUPERSEDES[name]
        tables.append(
            {
                "path": path,
                "sha256": sha256_of(path),
                "n_rows": int(len(frame)),
                "columns": list(frame.columns),
                "supersedes_path": superseded,
                "supersedes_sha256": (
                    sha256_of(superseded) if osp.isfile(superseded) else ""
                ),
                "supersedes_exists": osp.isfile(superseded),
            }
        )
        print(f"wrote {path}: {len(frame)} rows")

    record_hashes: dict[str, str] = {}
    if osp.isdir(args.curation_dir):
        for entry in sorted(os.listdir(args.curation_dir)):
            if entry.endswith(".json"):
                record_hashes[entry] = sha256_of(osp.join(args.curation_dir, entry))

    manifest = VerifiedTableManifest(
        built_at=datetime.now(UTC).isoformat(),
        pipeline_result_path=args.pipeline_result,
        pipeline_result_sha256=sha256_of(args.pipeline_result),
        gem_path=result.config.gem_path,
        gem_sha256=result.config.gem_sha256,
        stages=result.config.stage_names,
        curation_dir=args.curation_dir if record_hashes else "",
        curation_record_sha256=record_hashes,
        tables=tables,
        n_metabolites_priced=result.metabolite_counts.n_priced,
        n_reactions_priced=result.reaction_counts.n_priced,
        n_reactions_chemical=result.reaction_counts_chemical.n_total,
        n_reactions_chemical_priced=result.reaction_counts_chemical.n_priced,
        n_reactions_non_chemical=result.n_non_chemical_reactions,
        n_reactions_cancelled=sum(1 for r in result.reactions if r.cancelled_met_ids),
        routes=result.reaction_counts.by_route,
    )
    manifest_path = osp.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as handle:
        handle.write(manifest.model_dump_json(indent=1))
    print(f"wrote {manifest_path}")
    print(
        f"  metabolites priced {manifest.n_metabolites_priced}, reactions priced "
        f"{manifest.n_reactions_priced}, chemical "
        f"{manifest.n_reactions_chemical_priced}/{manifest.n_reactions_chemical}"
    )
    for table in tables:
        print(
            f"  {osp.basename(str(table['path']))} supersedes {table['supersedes_path']}"
        )


if __name__ == "__main__":
    main()
