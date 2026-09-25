# experiments/031-betaxanthin-module/scripts/target_pool_store.py
# [[experiments.031-betaxanthin-module.scripts.target_pool_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-betaxanthin-module/scripts/target_pool_store

r"""The measured amino-acid pool of a deletion strain as an affine store.

The cross-endpoint read (``experiments/032-amino-acid-module/scripts/
betaxanthin_amino_acid_correlation.py``, 2026-09-24) found that a deletion's measured
M\"ulleder amino-acid pool reads its betaxanthin level out of fold at 0.226 through a ridge
on its own strain folds, and appended to the gene's GO multi-hot adds 0.072 beyond the
annotation. That read lives on its own folds and its own model. This script writes the
pool as an affine store over the same gene axis the flux and GO stores use, so the same
columns are read in the same column as every other arm: the CPU ladder
(``target_store_baselines.py``, ridge, cosine nearest neighbors and a forest on the
trainer's pinned split), ``target_store_concat.py`` (the pool appended to GO and to GO
with the HiGHS vector), and the transformer's affine slot.

Columns: the 19 M\"ulleder 2016 concentrations in mM, stored as released. The trainer's
``log1p`` encoding and the ladder's ``log(1 + v)`` then map them, and since mM are
positive the map is monotone, unlike a log of a log. Rows: every ORF of the loader's
``preprocess/data.csv``, one row per strain. The wild-type row, what a gene the screen
did not measure reads, is the loader's reference: the population robust mean per amino
acid (the MCD estimate of the paper's ``robust_summary_statistics`` sheet), carried in
``preprocess/experiment_reference_index.json``. A missing cell in the released table is
filled with that reference for its amino acid and counted in the provenance.

What this store is: phenotype to phenotype. It carries a measurement of the same strain
on minimal medium, not the strain's sequence, so a model over it demonstrates that the
amino-acid data carries betaxanthin signal beyond the gene's identity, not that the
module generalizes from genotype; the split by gene still holds, since a held-out gene's
pool row is that strain's own measurement in a different screen, never the betaxanthin
target. Stores are never overwritten.

Run from the repo root::

    PYTHONHASHSEED=0 PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/031-betaxanthin-module/scripts/target_pool_store.py --out YPD_bx_aapool
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
TABLE_DIR = osp.join(DATA_ROOT, "data/torchcell/reference_flux")
MULLEDER = osp.join(DATA_ROOT, "data/torchcell/amino_acid_mulleder2016")
STUDY = "031-betaxanthin-module"


class PoolStoreProvenance(BaseModel):
    """Where the store's rows come from, and what was filled."""

    model_config = ConfigDict(extra="forbid")
    source_csv: str
    source_csv_sha256: str
    source_build_manifest: dict
    reference_source: str
    reference_levels_mM: dict[str, float]
    n_genes: int
    n_columns: int
    n_missing_cells_filled_with_reference: int
    column_layout: str
    method: str
    flux_sha256: str
    written_by: str = "experiments/031-betaxanthin-module/scripts/target_pool_store.py"


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="YPD_bx_aapool", help="store stem, ess_solve_encodings_<out>.npz")
    ap.add_argument("--out-dir", default=TABLE_DIR)
    args = ap.parse_args()
    started = time.time()
    csv_path = osp.join(MULLEDER, "preprocess", "data.csv")
    table = pd.read_csv(csv_path).set_index("orf")
    assert not table.index.duplicated().any(), "duplicated ORF in the released table"
    columns = [str(c) for c in table.columns]
    assert len(columns) == 19, columns
    with open(osp.join(MULLEDER, "preprocess", "experiment_reference_index.json")) as fh:
        refs = json.load(fh)
    assert len(refs) == 1, f"{len(refs)} reference groups; expected one"
    reference = refs[0]["reference"]["phenotype_reference"]["metabolite_level"]
    assert set(reference) == set(columns), set(reference) ^ set(columns)
    wild = np.array([float(reference[c]) for c in columns], np.float32)
    values = table[columns].to_numpy(np.float64)
    missing = ~np.isfinite(values)
    n_missing = int(missing.sum())
    values = np.where(missing, wild[None, :], values).astype(np.float32)
    assert np.isfinite(values).all() and (values >= 0).all(), "a negative concentration"
    with open(osp.join(MULLEDER, "preprocess", "build_manifest.json")) as fh:
        manifest = json.load(fh)
    provenance = PoolStoreProvenance(
        source_csv=csv_path,
        source_csv_sha256=sha256_of(csv_path),
        source_build_manifest=manifest,
        reference_source=osp.join(MULLEDER, "preprocess", "experiment_reference_index.json"),
        reference_levels_mM={c: float(reference[c]) for c in columns},
        n_genes=int(len(table)),
        n_columns=len(columns),
        n_missing_cells_filled_with_reference=n_missing,
        column_layout="Mulleder 2016 intracellular amino-acid concentrations (mM, as released), 19 columns",
        method=(
            "the strain's own measured amino-acid pool as affine columns; the wild-type row is "
            "the loader's reference (population robust mean per amino acid); written by "
            "target_pool_store.py"
        ),
        flux_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
    )
    os.makedirs(args.out_dir, exist_ok=True)
    path = osp.join(args.out_dir, f"ess_solve_encodings_{args.out}.npz")
    assert not osp.exists(path), f"{path} exists; stores are never overwritten"
    np.savez(
        path,
        genes=np.array([str(g) for g in table.index]),
        v_deletion=values,
        v_wild_type=wild,
        rxn_ids=np.array(columns),
        provenance=np.array(provenance.model_dump_json()),
    )
    print(f"{path}: {values.shape}; {n_missing} missing cells filled with the reference", flush=True)
    results_dir = osp.join(EXPERIMENT_ROOT, STUDY, "results")
    os.makedirs(results_dir, exist_ok=True)
    summary = {
        "store": path,
        "n_genes": int(len(table)),
        "n_columns": len(columns),
        "columns": columns,
        "n_missing_cells_filled_with_reference": n_missing,
        "reference_levels_mM": provenance.reference_levels_mM,
        "source_csv_sha256": provenance.source_csv_sha256,
        "seconds": time.time() - started,
    }
    out = osp.join(results_dir, f"target_pool_store_{args.out}.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(out)


if __name__ == "__main__":
    main()
