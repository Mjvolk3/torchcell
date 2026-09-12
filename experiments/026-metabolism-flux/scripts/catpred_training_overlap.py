# experiments/026-metabolism-flux/scripts/catpred_training_overlap.py
# [[experiments.026-metabolism-flux.scripts.catpred_training_overlap]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/catpred_training_overlap.py

r"""How much of the measured yeast set CatPred was trained on, counted rather than assumed.

WHY THIS RUNS BEFORE ANY SCORE IS QUOTED
------------------------------------------
Every predictor in this comparison trained on BRENDA and/or SABIO-RK, and the Open Enzyme
Database is assembled from the same literature, so ``compare_predictors_to_measured.py``
already carries a leakage caveat. For six of the seven that caveat is all there is: their
training tables either were never released at the row level or do not carry the accession,
so the overlap is unknown. CatPred released **CatPred-DB**, one CSV per parameter with a
UniProt accession, a substrate SMILES, an EC number and a taxonomy id on every row, so for
this one predictor the overlap is a **measurement**.

It matters more here than it would for a held-out benchmark, because of what the released
checkpoints are. ``pretrained/production/kcat/args.json`` records
``data_path: kcat-random_trainvaltest.csv``: the production ensembles were fit on
train **plus** validation **plus** test. The paper's held-out :math:`R^2` describes models
trained on the 80 percent split; it does not describe the weights mirrored here. So a pair
that appears anywhere in CatPred-DB is a pair these weights were fit on.

THREE LEVELS, BECAUSE THEY MEAN DIFFERENT THINGS
--------------------------------------------------
``enzyme`` -- the accession appears in CatPred-DB. The model has seen this protein, on
some substrate. Its sequence branch is not being asked to generalize.

``pair`` -- the accession appears with a substrate whose InChIKey connectivity block
matches ours. The model has seen this enzyme acting on this molecule, so its prediction is
a partly memorized value. Connectivity blocks rather than exact SMILES, because a database
writes the neutral acid where the GEM writes the anion, and matching on the exact string
would report a false zero.

``value`` -- the pair matches AND the value CatPred-DB recorded is within a tenth of a
decade of the value the Open Enzyme Database recorded. This is the same measurement
reaching both databases from the same paper, which is leakage in its strongest form.

For :math:`k_{cat}` a CatPred-DB row carries the CONCATENATED reactant SMILES, so a match
is tested against every dot-separated fragment. For :math:`K_M` the row carries the single
relevant substrate.

Writes ``results/catpred_training_overlap.json``. Run from the worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/catpred_training_overlap.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from torchcell.metabolism.kinetics_pipeline import inchikey_block, sha256_of

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
MIRROR = osp.join(DATA_ROOT, "models", "kinetics", "catpred")
DB_DIR = osp.join(MIRROR, "CatPred-DB", "data")
MEASURED_CSV = osp.join(RESULTS, "predictor_vs_measured.csv")
PAIR_TABLE = osp.join(RESULTS, "kinetic_verified_pairs.parquet")

#: The CatPred-DB table per parameter, its substrate column, and its target column. The
#: ``trainvaltest`` split is the one the PRODUCTION weights were fit on, which is what
#: makes it the right table to test leakage against.
DB_SPEC = {
    "k_cat": ("kcat/kcat-random_trainvaltest.csv", "reactant_smiles", "log10kcat_max"),
    "K_M": ("km/km-random_trainvaltest.csv", "substrate_smiles", "log10km_mean"),
}
#: NCBI taxonomy ids for *S. cerevisiae*: the species and the S288C reference strain.
SCEREVISIAE_TAXA = (4932, 559292)
#: How close two log10 values must be to be called the same measurement. A tenth of a
#: decade is well inside the spread between two literature reports of one constant, and
#: far below the decade at which predictors are judged here.
VALUE_TOLERANCE_DECADES = 0.1


class OverlapLevel(BaseModel):
    """One of the three overlap readings, with its count and its denominator."""

    model_config = ConfigDict(extra="forbid")

    level: str
    n_matched: int
    n_total: int
    frac: float
    examples: list[str] = Field(default_factory=list)


class ParameterOverlap(BaseModel):
    """Everything measured for one parameter."""

    model_config = ConfigDict(extra="forbid")

    parameter: str
    db_path: str
    db_sha256: str
    n_db_rows: int
    n_db_accessions: int
    n_db_scerevisiae_rows: int
    n_db_scerevisiae_accessions: int
    n_measured_pairs: int
    levels: list[OverlapLevel]
    #: Of the GEM's own proteins, how many CatPred has seen at all. A wider reading than
    #: the 57 scorable pairs: it says how much of the 3,728-unit table is a prediction for
    #: a protein the model already knows.
    n_gem_accessions: int
    n_gem_accessions_in_db: int
    n_gem_units: int
    n_gem_units_all_genes_in_db: int


class OverlapReport(BaseModel):
    """The whole check, one object."""

    model_config = ConfigDict(extra="forbid")

    built_at: str
    mirror: str
    production_data_path: str
    value_tolerance_decades: float
    measured_csv: str
    measured_csv_sha256: str
    parameters: list[ParameterOverlap]
    caveat: str


def _blocks(smiles: str, cache: dict[str, str | None]) -> set[str]:
    """The InChIKey connectivity blocks of every dot-separated fragment of a SMILES."""
    out: set[str] = set()
    for fragment in str(smiles).split("."):
        if not fragment:
            continue
        if fragment not in cache:
            cache[fragment] = inchikey_block(fragment)
        block = cache[fragment]
        if block:
            out.add(block)
    return out


def _measured(parameter: str) -> pd.DataFrame:
    """The 57 measured yeast pairs for one parameter, one row each."""
    frame = pd.read_csv(MEASURED_CSV)
    frame = frame[frame["parameter"] == parameter]
    return frame.drop_duplicates(["uniprot", "substrate"])[
        ["uniprot", "substrate", "smiles", "measured"]
    ].reset_index(drop=True)


def _gem_accessions() -> tuple[dict[int, set[str]], set[str]]:
    """Per catalytic unit the accessions of its genes, and the set of all of them."""
    pairs = pd.read_parquet(PAIR_TABLE, columns=["unit_id", "uniprot"]).dropna()
    per_unit: dict[int, set[str]] = {}
    for unit_id, accession in pairs.drop_duplicates().itertuples(index=False):
        per_unit.setdefault(int(unit_id), set()).add(str(accession))
    return per_unit, {a for accessions in per_unit.values() for a in accessions}


def overlap_for(parameter: str) -> ParameterOverlap:
    """Measure the three overlap levels for one parameter."""
    relative, substrate_column, target_column = DB_SPEC[parameter]
    db_path = osp.join(DB_DIR, relative)
    database = pd.read_csv(
        db_path,
        usecols=["uniprot", substrate_column, target_column, "taxonomy_id", "ec"],
    )
    measured = _measured(parameter)

    cache: dict[str, str | None] = {}
    db_by_accession: dict[str, list[tuple[set[str], float]]] = {}
    for row in database.itertuples(index=False):
        accession = str(getattr(row, "uniprot"))
        db_by_accession.setdefault(accession, []).append(
            (
                _blocks(getattr(row, substrate_column), cache),
                float(getattr(row, target_column)),
            )
        )

    enzyme_hits: list[str] = []
    pair_hits: list[str] = []
    value_hits: list[str] = []
    targets = np.log10(measured["measured"].to_numpy(dtype=float))
    for position, row in enumerate(measured.itertuples(index=False)):
        accession = str(row.uniprot)
        entries = db_by_accession.get(accession)
        if not entries:
            continue
        label = f"{accession}/{row.substrate}"
        enzyme_hits.append(label)
        ours = _blocks(row.smiles, cache) if isinstance(row.smiles, str) else set()
        matched = [value for blocks, value in entries if ours & blocks]
        if not matched:
            continue
        pair_hits.append(label)
        target = float(targets[position])
        if any(abs(value - target) <= VALUE_TOLERANCE_DECADES for value in matched):
            value_hits.append(label)

    per_unit, gem_accessions = _gem_accessions()
    in_db = {a for a in gem_accessions if a in db_by_accession}
    units_all_in_db = sum(
        1 for accessions in per_unit.values() if accessions and accessions <= in_db
    )
    yeast = database[database["taxonomy_id"].isin(SCEREVISIAE_TAXA)]
    n_measured = int(len(measured))
    return ParameterOverlap(
        parameter=parameter,
        db_path=db_path,
        db_sha256=sha256_of(db_path),
        n_db_rows=int(len(database)),
        n_db_accessions=int(database["uniprot"].nunique()),
        n_db_scerevisiae_rows=int(len(yeast)),
        n_db_scerevisiae_accessions=int(yeast["uniprot"].nunique()),
        n_measured_pairs=n_measured,
        levels=[
            OverlapLevel(
                level="enzyme",
                n_matched=len(enzyme_hits),
                n_total=n_measured,
                frac=len(enzyme_hits) / n_measured,
                examples=sorted(enzyme_hits)[:8],
            ),
            OverlapLevel(
                level="pair",
                n_matched=len(pair_hits),
                n_total=n_measured,
                frac=len(pair_hits) / n_measured,
                examples=sorted(pair_hits)[:8],
            ),
            OverlapLevel(
                level="value",
                n_matched=len(value_hits),
                n_total=n_measured,
                frac=len(value_hits) / n_measured,
                examples=sorted(value_hits)[:8],
            ),
        ],
        n_gem_accessions=len(gem_accessions),
        n_gem_accessions_in_db=len(in_db),
        n_gem_units=len(per_unit),
        n_gem_units_all_genes_in_db=units_all_in_db,
    )


def main() -> None:
    """Measure the overlap for both parameters and write the report."""
    with open(osp.join(MIRROR, "pretrained", "production", "kcat", "args.json")) as h:
        production_data_path = str(json.load(h)["data_path"])

    report = OverlapReport(
        built_at=datetime.now(UTC).isoformat(),
        mirror=MIRROR,
        production_data_path=production_data_path,
        value_tolerance_decades=VALUE_TOLERANCE_DECADES,
        measured_csv=MEASURED_CSV,
        measured_csv_sha256=sha256_of(MEASURED_CSV),
        parameters=[overlap_for(p) for p in ("k_cat", "K_M")],
        caveat=(
            "The same overlap almost certainly exists for the other predictors, which "
            "trained on the same two databases. It is measured here and not there "
            "because CatPred released its training table with a UniProt accession on "
            "every row; a zero is not claimed for any other predictor."
        ),
    )
    out_path = osp.join(RESULTS, "catpred_training_overlap.json")
    with open(out_path, "w") as handle:
        handle.write(report.model_dump_json(indent=2))

    print(f"production weights were fit on {production_data_path}")
    for block in report.parameters:
        print(
            f"\n{block.parameter}: CatPred-DB {block.n_db_rows} rows, "
            f"{block.n_db_accessions} accessions, "
            f"{block.n_db_scerevisiae_rows} S. cerevisiae rows"
        )
        for level in block.levels:
            print(
                f"  {level.level:8s} {level.n_matched:3d}/{level.n_total} "
                f"({level.frac:.1%})"
            )
        print(
            f"  GEM proteins in CatPred-DB: {block.n_gem_accessions_in_db}/"
            f"{block.n_gem_accessions}; catalytic units whose every gene is in it: "
            f"{block.n_gem_units_all_genes_in_db}/{block.n_gem_units}"
        )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
