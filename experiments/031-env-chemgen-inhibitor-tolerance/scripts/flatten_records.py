# experiments/031-env-chemgen-inhibitor-tolerance/scripts/flatten_records.py
# [[experiments.031-env-chemgen-inhibitor-tolerance.scripts.flatten_records]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-env-chemgen-inhibitor-tolerance/scripts/flatten_records
"""Flatten the built env-chemgen LMDB stores into one row per record (parquet).

Reads the DEV-tree builds (``$DATA_ROOT/data/torchcell/env_chemgen_*``) through their
loaders, so every value is the served pydantic record, not the raw matrix. One row per
record carries every axis a cell/environment representation has to encode: the genotype
(gene, perturbation type, ploidy), the environment (medium, temperature, aerobicity,
duration, dosed compounds with InChIKey / dose / basis, physical factors), the readout
(measurement + assay type, replicate structure, uncertainty), and the reference. The
downstream scripts (``dataset_axes_comparison.py``, ``cross_dataset_similarity.py``)
read these parquet files only.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import time
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from torchcell.datasets.scerevisiae.hillenmeyer2008 import (
    HetHillenmeyer2008Dataset,
    HomHillenmeyer2008Dataset,
)
from torchcell.datasets.scerevisiae.vanacloig2022 import EnvChemgenVanacloig2022Dataset

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(
    EXPERIMENT_ROOT, "031-env-chemgen-inhibitor-tolerance", "results"
)

DATASETS: dict[str, tuple[type, str]] = {
    "vanacloig2022": (EnvChemgenVanacloig2022Dataset, "env_chemgen_vanacloig2022"),
    "hillenmeyer2008_hom": (
        HomHillenmeyer2008Dataset,
        "env_chemgen_hillenmeyer2008_hom",
    ),
    "hillenmeyer2008_het": (
        HetHillenmeyer2008Dataset,
        "env_chemgen_hillenmeyer2008_het",
    ),
}


def _env_axes(env: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Flatten one Environment dict into prefixed scalar columns."""
    media = env["media"]
    temp = env.get("temperature")
    compounds, inchikeys, cids, values, units, bases, solvents, physical = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for p in env.get("perturbations", []):
        if p["perturbation_type"] == "small_molecule":
            c, conc = p["compound"], p["concentration"]
            compounds.append(c["name"])
            inchikeys.append(c.get("inchikey") or "")
            cids.append(str(c.get("pubchem_cid") or ""))
            values.append("" if conc.get("value") is None else str(conc["value"]))
            units.append(conc.get("unit") or "")
            bases.append(conc.get("basis") or "")
            solvents.append(
                (p.get("solvent") or {}).get("name", "") if p.get("solvent") else ""
            )
        elif p["perturbation_type"] == "environment_physical":
            mag = p.get("magnitude") or {}
            agent = p.get("agent") or {}
            physical.append(
                f"{p['factor']}={mag.get('value')}{mag.get('unit') or ''}"
                + (f"[{agent.get('name')}]" if agent else "")
            )
    return {
        f"{prefix}media_name": media["name"],
        f"{prefix}media_base": media.get("base_medium"),
        f"{prefix}media_state": media["state"],
        f"{prefix}media_synthetic": media["is_synthetic"],
        f"{prefix}media_n_components": len(media.get("components", [])),
        f"{prefix}media_dropouts": "|".join(
            d["name"] for d in media.get("dropouts", [])
        ),
        f"{prefix}temperature_c": None if temp is None else temp["value"],
        f"{prefix}aerobicity": env.get("aerobicity"),
        f"{prefix}duration_hours": env.get("duration_hours"),
        f"{prefix}duration_generations": env.get("duration_generations"),
        f"{prefix}n_small_molecules": len(compounds),
        f"{prefix}compound": "|".join(compounds),
        f"{prefix}inchikey": "|".join(inchikeys),
        f"{prefix}pubchem_cid": "|".join(cids),
        f"{prefix}dose_value": "|".join(values),
        f"{prefix}dose_unit": "|".join(units),
        f"{prefix}dose_basis": "|".join(bases),
        f"{prefix}solvent": "|".join(solvents),
        f"{prefix}physical": "|".join(physical),
        f"{prefix}n_gaps": len(env.get("provenance_gaps", [])),
    }


def _row(name: str, idx: int, rec: dict[str, Any]) -> dict[str, Any]:
    """One flat row for one stored record."""
    exp, ref = rec["experiment"], rec["reference"]
    genos = exp["genotype"] if isinstance(exp["genotype"], list) else [exp["genotype"]]
    perts = [p for g in genos for p in g["perturbations"]]
    ph = exp["phenotype"]
    row: dict[str, Any] = {
        "dataset": name,
        "idx": idx,
        "gene": "|".join(p["systematic_gene_name"] for p in perts),
        "gene_common": "|".join(p.get("perturbed_gene_name") or "" for p in perts),
        "n_genes": len(perts),
        "perturbation_type": "|".join(p["perturbation_type"] for p in perts),
        "ref_species": ref["genome_reference"]["species"],
        "ref_strain": ref["genome_reference"]["strain"],
        "ref_ploidy": ref["genome_reference"].get("ploidy", "haploid"),
        "measurement_type": ph["measurement_type"],
        "assay_type": ph.get("assay_type"),
        "response": ph.get("environment_response"),
        "response_se": ph.get("environment_response_se"),
        "uncertainty": ph.get("environment_response_uncertainty"),
        "uncertainty_type": ph.get("environment_response_uncertainty_type"),
        "n_samples": ph.get("n_samples"),
        "sample_unit": ph.get("sample_unit"),
        "category": ph.get("category"),
        "screen_id": ph.get("screen_id"),
        "units": ph.get("units"),
        "ref_response": ref["phenotype_reference"].get("environment_response"),
    }
    row.update(_env_axes(exp["environment"], ""))
    row.update(_env_axes(ref["environment_reference"], "ref_"))
    return row


def flatten(name: str, limit: int | None) -> str:
    """Write ``results/records_<name>.parquet`` and return its path."""
    cls, sub = DATASETS[name]
    ds = cls(root=osp.join(DATA_ROOT, "data", "torchcell", sub))
    n = len(ds) if limit is None else min(limit, len(ds))
    t0 = time.time()
    rows = []
    for i in range(n):
        rows.append(_row(name, i, ds[i]))
        if (i + 1) % 200_000 == 0:
            print(f"{name}: {i + 1}/{n} rows ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = osp.join(RESULTS_DIR, f"records_{name}.parquet")
    df.to_parquet(out, index=False)
    print(
        f"{name}: wrote {len(df)} rows -> {out} ({time.time() - t0:.0f}s)", flush=True
    )
    return out


def main() -> None:
    """CLI: ``--datasets`` (default all three) and an optional ``--limit`` for smoke runs."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS)
    )
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    for name in args.datasets:
        flatten(name, args.limit)


if __name__ == "__main__":
    main()
