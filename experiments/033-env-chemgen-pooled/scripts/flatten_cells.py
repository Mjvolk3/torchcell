# experiments/033-env-chemgen-pooled/scripts/flatten_cells.py
# [[experiments.033-env-chemgen-pooled.scripts.flatten_cells]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/033-env-chemgen-pooled/scripts/flatten_cells
"""Flatten the processed store of build 001 into one row per (genotype, environment) cell.

The processed LMDB holds 6,042,771 entries, each a JSON list of the measurements the
aggregator folded into one cell. Everything the 031 plan asks of the store, the ploidy
pairs, the dose bases, the environment vocabulary, the repeated measurements, lives inside
those 106 GB of JSON, and the processed directory's own indexes carry none of it. This
script reads every entry once and writes the axes as columns, so every later analysis
reads a table instead of the store.

One row is one entry. The genotype columns are the perturbed genes, the queried gene
(the one that is not a host deletion), the perturbation type, the copy numbers and the
functional dose ``copy_number / reference_copy_number`` (zero for a deletion). The
environment columns are the content address the aggregator keyed on
(``environment_identity``), the dosed compounds with their InChIKeys and concentrations,
and the physical and medium fields. The phenotype columns keep EVERY measurement of the
cell as a list, because which one a trainer reads is a read-time label policy.

A record whose perturbation is neither a deletion nor a copy-number variant raises; so
does a concentration unit outside the ones listed in ``MOLAR_FACTOR`` and
``NOT_MOLAR_UNITS``. Nothing is defaulted.

Run under slurm (``gh_flatten_cells.slurm``): the full pass parses 106 GB of JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import os.path as osp
from multiprocessing import Pool
from typing import Any

import lmdb
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

from torchcell.datamodels.identity import environment_identity, identity_sha256
from torchcell.datamodels.schema import environment_class_for

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]

#: Which build to flatten. Build 001 is the table of the 2026.09.21 store (every 031 and
#: 035 result stands on it); build 002 is the table of the 2026.10.06 store, after the
#: dataset fixes of #500, #501 and #504 to #506. Each build's table keeps its own
#: directory so the two records can be compared.
BUILD = os.environ.get("BUILD_NAME", "002-pooled-build")
BUILD_ROOT = f"/db/experiments/033-env-chemgen-pooled-{BUILD}"
LMDB_PATH = osp.join(BUILD_ROOT, "processed", "lmdb")
DEFAULT_OUT = osp.join(
    DATA_ROOT,
    "experiments",
    "033-env-chemgen-pooled",
    "cell_table" if BUILD == "001-pooled-build" else f"cell_table_{BUILD[:3]}",
)

#: The three efflux-regulator deletions of the 3DeltaAlpha host (PDR1, PDR3, SNQ2).
HOST_GENES: frozenset[str] = frozenset({"YGL013C", "YBL005W", "YDR011W"})

#: Units that convert to molar without a molecular weight, as a factor to mol/L.
MOLAR_FACTOR: dict[str, float] = {"M": 1.0, "mM": 1e-3, "uM": 1e-6, "nM": 1e-9}
#: Units that are a dose and do NOT convert to molar without a molecular weight or density.
NOT_MOLAR_UNITS: frozenset[str] = frozenset(
    {"ug/mL", "mg/mL", "percent_v/v", "percent_w/v", "g/L"}
)


class CellRow(BaseModel):
    """One (genotype, environment) cell of the processed store."""

    index: int
    dataset: str
    n_measurements: int
    # genotype
    genes: str
    n_perturbations: int
    query_gene: str
    perturbation_type: str
    copy_number: float | None
    reference_copy_number: float | None
    functional_dose: float
    ploidy: str
    strain: str
    # environment
    environment_id: str
    n_compounds: int
    compound_names: str
    inchikeys: str
    n_with_inchikey: int
    conc_values: str
    conc_units: str
    conc_bases: str
    log10_molar: float | None
    environment_perturbation_types: str
    physical_perturbations: str
    base_medium: str
    temperature_c: float | None
    aerobicity: str | None
    duration_hours: float | None
    duration_generations: float | None
    # phenotype, one element per folded measurement
    measurement_type: str
    assay_type: str
    responses: list[float]
    response_ses: list[float | None]
    n_samples: list[int | None]
    sample_units: list[str | None]
    screen_ids: list[str | None]


def functional_dose(perturbation: dict[str, Any]) -> float:
    """Copies present over copies in the reference; zero for a deletion."""
    if "copy_number" in perturbation:
        return float(perturbation["copy_number"]) / float(
            perturbation["reference_copy_number"]
        )
    assert perturbation["state"] == "absent", (
        f"perturbation is neither a copy-number variant nor a deletion: {perturbation}"
    )
    return 0.0


def log10_molar(concentration: dict[str, Any] | None) -> float | None:
    """log10 of the molar concentration, or None where no molar value is stated."""
    if concentration is None or concentration["value"] is None:
        return None
    unit = concentration["unit"]
    if unit in NOT_MOLAR_UNITS:
        return None
    value = float(concentration["value"])
    if value <= 0.0:
        return None
    return math.log10(value * MOLAR_FACTOR[unit])


def strip_provenance(perturbation: dict[str, Any]) -> dict[str, Any]:
    """A physical perturbation without its provenance blocks, for a compact column."""
    return {
        k: v
        for k, v in perturbation.items()
        if k not in ("provenance_gaps", "provenance", "description")
    }


def flatten(index: int, entry: list[dict[str, Any]]) -> CellRow:
    """The row of one store entry. Genotype and environment are read off the first
    measurement; the aggregator's key guarantees every measurement shares them.
    """
    first = entry[0]
    experiment = first["experiment"]
    genome = first["experiment_reference"]["genome_reference"]
    perturbations = experiment["genotype"]["perturbations"]
    environment = experiment["environment"]

    query = [p for p in perturbations if p["systematic_gene_name"] not in HOST_GENES]
    if len(perturbations) == 1:
        query = perturbations
    assert len(query) == 1, f"entry {index} has {len(query)} queried genes"
    q = query[0]

    small = [
        p
        for p in environment["perturbations"]
        if p["perturbation_type"] == "small_molecule"
    ]
    physical = [
        p
        for p in environment["perturbations"]
        if p["perturbation_type"] != "small_molecule"
    ]
    concentrations = [p["concentration"] for p in small]
    keys = [p["compound"]["inchikey"] for p in small]
    temperature = environment["temperature"]

    phenotypes = [m["experiment"]["phenotype"] for m in entry]
    return CellRow(
        index=index,
        dataset=experiment["dataset_name"],
        n_measurements=len(entry),
        genes=";".join(sorted(p["systematic_gene_name"] for p in perturbations)),
        n_perturbations=len(perturbations),
        query_gene=q["systematic_gene_name"],
        perturbation_type=q["perturbation_type"],
        copy_number=q.get("copy_number"),
        reference_copy_number=q.get("reference_copy_number"),
        functional_dose=functional_dose(q),
        ploidy=genome["ploidy"],
        strain=genome["strain"],
        environment_id=identity_sha256(
            environment_identity(
                environment_class_for(experiment["experiment_type"])(**environment)
            )
        ),
        n_compounds=len(small),
        compound_names="|".join(p["compound"]["name"] for p in small),
        inchikeys="|".join(k if k is not None else "" for k in keys),
        n_with_inchikey=sum(k is not None for k in keys),
        conc_values="|".join(
            "" if c is None or c["value"] is None else repr(float(c["value"]))
            for c in concentrations
        ),
        conc_units="|".join(
            "" if c is None or c["unit"] is None else c["unit"] for c in concentrations
        ),
        conc_bases="|".join(
            "" if c is None or c["basis"] is None else c["basis"]
            for c in concentrations
        ),
        log10_molar=log10_molar(concentrations[0]) if len(small) == 1 else None,
        environment_perturbation_types="|".join(
            sorted(p["perturbation_type"] for p in environment["perturbations"])
        ),
        physical_perturbations=json.dumps(
            [strip_provenance(p) for p in physical], sort_keys=True
        ),
        base_medium=environment["media"]["base_medium"],
        temperature_c=None if temperature is None else temperature["value"],
        aerobicity=environment["aerobicity"],
        duration_hours=environment["duration_hours"],
        duration_generations=environment["duration_generations"],
        measurement_type=phenotypes[0]["measurement_type"],
        assay_type=phenotypes[0]["assay_type"],
        responses=[p["environment_response"] for p in phenotypes],
        response_ses=[p["environment_response_se"] for p in phenotypes],
        n_samples=[p["n_samples"] for p in phenotypes],
        sample_units=[p["sample_unit"] for p in phenotypes],
        screen_ids=[p["screen_id"] for p in phenotypes],
    )


def flatten_chunk(args: tuple[list[int], str]) -> str:
    """Read one chunk of indices and write it as a parquet part."""
    indices, part_path = args
    env = lmdb.open(
        LMDB_PATH, readonly=True, lock=False, readahead=False, meminit=False
    )
    rows = []
    with env.begin() as txn:
        for index in indices:
            value = txn.get(str(index).encode())
            assert value is not None, f"store has no entry {index}"
            rows.append(flatten(index, json.loads(value)).model_dump())
    env.close()
    pd.DataFrame(rows).to_parquet(part_path, index=False)
    return part_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--chunk", type=int, default=20_000)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="read every stride-th entry; 1 is the full store, larger is a smoke run",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--stop",
        type=int,
        default=None,
        help="with --start, a contiguous range of entries for a smoke run",
    )
    args = parser.parse_args()

    env = lmdb.open(
        LMDB_PATH, readonly=True, lock=False, readahead=False, meminit=False
    )
    length = env.stat()["entries"]
    env.close()
    stop = length if args.stop is None else args.stop
    indices = list(range(args.start, stop, args.stride))
    print(f"store length {length:,}; reading {len(indices):,} entries", flush=True)

    parts_dir = osp.join(args.out, "parts")
    os.makedirs(parts_dir, exist_ok=True)
    jobs = [
        (indices[i : i + args.chunk], osp.join(parts_dir, f"part_{n:05d}.parquet"))
        for n, i in enumerate(range(0, len(indices), args.chunk))
    ]
    with Pool(args.workers) as pool:
        for done, path in enumerate(pool.imap_unordered(flatten_chunk, jobs), start=1):
            if done % 20 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} parts, last {path}", flush=True)

    table = pd.concat(
        [pd.read_parquet(path) for _, path in jobs], ignore_index=True
    ).sort_values("index", ignore_index=True)
    assert len(table) == len(indices), "a part is missing rows"
    assert table["index"].is_unique, "an entry was read twice"
    out_path = osp.join(args.out, "cell_table.parquet")
    table.to_parquet(out_path, index=False)
    print(f"wrote {out_path}: {len(table):,} rows, {table.shape[1]} columns")
    print(table["dataset"].value_counts().to_string())


if __name__ == "__main__":
    main()
