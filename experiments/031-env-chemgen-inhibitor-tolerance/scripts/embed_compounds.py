# experiments/031-env-chemgen-inhibitor-tolerance/scripts/embed_compounds.py
# [[experiments.031-env-chemgen-inhibitor-tolerance.scripts.embed_compounds]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-env-chemgen-inhibitor-tolerance/scripts/embed_compounds
"""Embed every dosed compound of the three env-chemgen datasets with every registered
molecule encoder, and report coverage.

Reads the flattened parquet files written by ``flatten_records.py``
(``results/records_<dataset>.parquet``; ``inchikey`` and ``compound`` are ``|``-joined
when a record doses two compounds), resolves each InChIKey to a SMILES through the
committed ``torchcell/datamodels/compound_identity_table.json``, and writes

- ``results/embeddings/<encoder>.npz`` with ``inchikey`` (str) and ``X`` (float32,
  one row per InChIKey across the union of the three datasets), and
- ``results/embedding_coverage.csv`` with one row per (dataset, encoder).

A compound is one distinct InChIKey. A record whose compound carries no InChIKey (a
mixture, a proprietary code, or an unresolved name) is not a compound here; the count
of such names is printed per dataset so the gap is visible. A compound whose InChIKey
has no SMILES in the table is counted in ``n_compounds`` and left out of
``n_with_smiles``; nothing is guessed. ``n_failed`` counts compounds the encoder's
``check`` rejects (unparsable, or for Uni-Mol no real 3D conformer); each is named in
``results/embeddings/failures.json``.

Timing: every encoder embeds the UNION of the three compound sets once.
``seconds_union`` is that measured wall time (check + encode, model already loaded);
``seconds`` prorates it by the dataset's share of the union; ``seconds_per_100`` is
``100 * seconds_union / n_embedded``. ``results/encoder_timing.md`` holds the
per-encoder rows (plus model load time) as the markdown table the encoders note
embeds.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv

from torchcell.molecule import ENCODERS, MoleculeEncoder

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(
    EXPERIMENT_ROOT, "031-env-chemgen-inhibitor-tolerance", "results"
)
EMBED_DIR = osp.join(RESULTS_DIR, "embeddings")
# the five served chemogenomic stores the representation design spans: three for record
# count, Wildenhain for compound diversity, Vanacloig as the application target
DATASETS = [
    "vanacloig2022",
    "hillenmeyer2008_hom",
    "hillenmeyer2008_het",
    "hoepfner2014",
    "wildenhain2015",
]


def load_identity_table() -> dict[str, dict[str, object]]:
    """InChIKey -> record for every table row that carries an InChIKey, read from
    the sha256-pinned table the resolver module owns.
    """
    from torchcell.datamodels.compound_identity import _TABLE_PATH

    with open(_TABLE_PATH) as f:
        records = json.load(f)["records"]
    return {r["inchikey"]: r for r in records if r.get("inchikey")}


def dataset_compounds(name: str) -> tuple[list[str], int]:
    """Sorted distinct InChIKeys dosed in ``name`` and the count of distinct compound
    names dosed with no InChIKey.
    """
    df = pd.read_parquet(
        osp.join(RESULTS_DIR, f"records_{name}.parquet"),
        columns=["inchikey", "compound"],
    )
    keys: set[str] = set()
    no_key: set[str] = set()
    for ik, comp in (
        df[["inchikey", "compound"]].drop_duplicates().itertuples(index=False)
    ):
        # The InChIKey field is the authority on how many compounds a record dosed: a key
        # is alphanumeric with dashes and can never contain the pipe the flattener joins
        # on. A compound NAME can: Wildenhain carries PubChem synonym strings with a
        # literal pipe in them, so a name that does not split into the same number of
        # parts is treated as one opaque label rather than aligned position by position.
        ik_parts = str(ik).split("|")
        comp_parts = str(comp).split("|")
        aligned = len(ik_parts) == len(comp_parts)
        for i, k in enumerate(ik_parts):
            if k:
                keys.add(k)
            elif aligned and comp_parts[i]:
                no_key.add(comp_parts[i])
            elif str(comp):
                no_key.add(str(comp))
    return sorted(keys), len(no_key)


def served_structures(datasets: list[str]) -> dict[str, dict[str, str]]:
    """InChIKey -> ``{"smiles", "name"}`` as the SERVED record carries them.

    The flattened records carry the ``Compound.smiles`` and ``Compound.name`` of every dosed
    compound. This is the only structure source for a key the curated identity table has no
    row for, and the name is what a failure report for such a key can print.
    """
    out: dict[str, dict[str, str]] = {}
    for name in datasets:
        path = osp.join(RESULTS_DIR, f"records_{name}.parquet")
        # a parquet written before the smiles column was added simply has nothing to give
        if "smiles" not in pq.ParquetFile(path).schema.names:
            print(
                f"  {name}: no smiles column, re-flatten to use the served structures"
            )
            continue
        df = pd.read_parquet(
            path, columns=["inchikey", "smiles", "compound"]
        ).drop_duplicates()
        for ik, sm, comp in df.itertuples(index=False):
            ik_parts = str(ik).split("|")
            sm_parts = str(sm).split("|")
            comp_parts = str(comp).split("|")
            # same rule as dataset_compounds: the key field sets the compound count, and a
            # name only aligns to a key when the name field splits into the same number
            aligned = len(ik_parts) == len(comp_parts)
            for i, (k, s) in enumerate(zip(ik_parts, sm_parts, strict=True)):
                if k and s and k not in out:
                    out[k] = {
                        "smiles": s,
                        "name": comp_parts[i] if aligned else str(comp),
                    }
    return out


def embed_all(
    encoder: MoleculeEncoder, smiles_by_key: dict[str, str]
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Coverage accounting, molecule by molecule, then one batched encode of the
    survivors. ``encoder.check`` is the encoder's own per-molecule precondition
    (parsability; for Uni-Mol also a real 3D conformer), so a rejected molecule is
    named and counted in ``n_failed`` and never embedded from a stand-in.
    Returns (key -> vector, key -> error message).
    """
    accepted: list[str] = []
    failed: dict[str, str] = {}
    for k, smi in smiles_by_key.items():
        try:
            encoder.check(smi)
        except ValueError as e:
            failed[k] = f"{type(e).__name__}: {e}"
        else:
            accepted.append(k)
    x = encoder.encode([smiles_by_key[k] for k in accepted])
    ok = dict(zip(accepted, x, strict=True))
    return ok, failed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--encoders", nargs="+", default=list(ENCODERS), choices=list(ENCODERS)
    )
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    args = ap.parse_args()

    table = load_identity_table()
    per_dataset: dict[str, list[str]] = {}
    for name in args.datasets:
        keys, n_no_key = dataset_compounds(name)
        per_dataset[name] = keys
        n_smiles = sum(1 for k in keys if table.get(k, {}).get("smiles"))
        missing_rows = [k for k in keys if k not in table]
        print(
            f"{name}: {len(keys)} InChIKeys, {n_smiles} with SMILES, "
            f"{len(missing_rows)} not in identity table, "
            f"{n_no_key} dosed compound names with no InChIKey"
        )
    union = sorted(set().union(*per_dataset.values()))
    # STRUCTURE SOURCE PRECEDENCE, not a fallback. The curated identity table WINS where it
    # has a row, because the curated and SMILES-derived routes measurably disagree on the
    # stereo block for some compounds. Where the table has no row at all the served record
    # is the only source, and it is the provenance authority: Hoepfner derives 88 of its 148
    # InChIKeys from the released Table S1 SMILES and those keys were never curated.
    served = served_structures(args.datasets)
    smiles_by_key: dict[str, str] = {}
    name_by_key: dict[str, str] = {}
    n_from_served = 0
    for k in union:
        curated = table.get(k, {}).get("smiles")
        if curated:
            smiles_by_key[k] = str(curated)
            name_by_key[k] = str(table[k]["name"])
        elif served.get(k):
            smiles_by_key[k] = served[k]["smiles"]
            name_by_key[k] = served[k]["name"]
            n_from_served += 1
    print(
        f"union: {len(union)} InChIKeys, {len(smiles_by_key)} with SMILES "
        f"({n_from_served} taken from the served record because the identity table has "
        f"no row)"
    )

    os.makedirs(EMBED_DIR, exist_ok=True)
    rows = []
    timing_rows = []
    failures: dict[str, dict[str, str]] = {}
    for enc_name in args.encoders:
        t0 = time.time()
        encoder = ENCODERS[enc_name]()
        t_load = time.time() - t0
        t0 = time.time()
        ok, failed = embed_all(encoder, smiles_by_key)
        seconds = time.time() - t0
        failures[enc_name] = failed
        keys_ok = sorted(ok)
        np.savez(
            osp.join(EMBED_DIR, f"{enc_name}.npz"),
            inchikey=np.array(keys_ok, dtype=str),
            X=np.stack([ok[k] for k in keys_ok]).astype(np.float32),
        )
        print(
            f"{enc_name}: dim {encoder.dim}, load {t_load:.1f}s, embedded {len(ok)} "
            f"in {seconds:.1f}s, failed {len(failed)}"
        )
        for k, msg in failed.items():
            print(f"  FAILED {k} {name_by_key[k]!r}: {msg[:160]}")
        timing_rows.append(
            {
                "encoder": enc_name,
                "dim": encoder.dim,
                "n_embedded": len(ok),
                "n_failed": len(failed),
                "load_seconds": round(t_load, 2),
                "seconds_union": round(seconds, 2),
                "seconds_per_100": round(100 * seconds / max(len(ok), 1), 3),
            }
        )
        for name in args.datasets:
            keys = per_dataset[name]
            with_smiles = [k for k in keys if k in smiles_by_key]
            n_emb = sum(1 for k in with_smiles if k in ok)
            rows.append(
                {
                    "dataset": name,
                    "encoder": enc_name,
                    "n_compounds": len(keys),
                    "n_with_smiles": len(with_smiles),
                    "n_embedded": n_emb,
                    "n_failed": len(with_smiles) - n_emb,
                    "dim": encoder.dim,
                    "seconds": round(
                        seconds * len(with_smiles) / max(len(smiles_by_key), 1), 3
                    ),
                    "seconds_union": round(seconds, 3),
                    "seconds_per_100": round(100 * seconds / max(len(ok), 1), 3),
                }
            )
    cov = pd.DataFrame(rows)
    cov.to_csv(osp.join(RESULTS_DIR, "embedding_coverage.csv"), index=False)
    with open(osp.join(EMBED_DIR, "failures.json"), "w") as f:
        json.dump(failures, f, indent=2, sort_keys=True)
    # One row per encoder over the union of compounds, as a markdown table for the
    # encoders note ([[torchcell.molecule.encoders]]).
    timing = pd.DataFrame(timing_rows)
    with open(osp.join(RESULTS_DIR, "encoder_timing.md"), "w") as f:
        f.write(markdown_table(timing))
    print()
    print(cov.to_string(index=False))
    print()
    print(timing.to_string(index=False))


def markdown_table(df: pd.DataFrame) -> str:
    """Pipe-delimited markdown for a small frame (no tabulate dependency)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
