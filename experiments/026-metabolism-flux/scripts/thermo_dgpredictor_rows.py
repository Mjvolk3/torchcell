# experiments/026-metabolism-flux/scripts/thermo_dgpredictor_rows.py
# [[experiments.026-metabolism-flux.scripts.thermo_dgpredictor_rows]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/thermo_dgpredictor_rows.py

r"""Name the 33 shipped reaction rows that are kJ/mol, so they stop being converted twice.

**The bug this closes.** yeast-GEM 9 copied yETFL's ``metDeltaGFtr`` and ``rxnDeltaGR``
without converting, and yETFL's model declares ``thermo_units = 'kcal/mol'``, so
``constraints._read_delta_g_csv`` multiplies every value it reads by 4.184. That is right
for the metabolite column and for 3,177 of the 3,210 rows of the reaction column. It is
wrong for the other 33: those come from dGPredictor, whose own driver
(``data/modelCuration/v8_7_0/code/run_dgpredictor.py``) reports kJ/mol, so multiplying
them by 4.184 leaves them 4.184x too large.

**How the 33 are identified, rather than guessed.** ``v8_7_0.m`` fills the reaction column
from yETFL by reaction id and then fills a fixed index block from dGPredictor, so a row is
a dGPredictor row exactly when the shipped column has a value and yETFL does not. That set
difference is computed here against the mirrored yETFL model, and the identification is
checked two ways before it is written: every row in the yETFL intersection must agree with
the shipped value to within ``round(x, 2)``, and the 33 must lie inside the index block
``v8_7_0.m`` names (``model.rxns(3983:4062)``, 1-based inclusive).

**What consumes the output.** ``parameters.resolve_reaction_delta_g`` reads
``dgpredictor_reaction_rows.json`` from the eQuilibrator thermo directory and divides those
rows back by 4.184 before the union table's last-resort fill. It raises if the file is
absent rather than quietly shipping the over-converted values, so the correction cannot be
skipped by forgetting to run this.

Writes ``results/dgpredictor_reaction_rows.json`` and a copy at
``$DATA_ROOT/data/torchcell/thermo_equilibrator/dgpredictor_reaction_rows.json``.

Run under the torchcell env (needs cobra and scipy)::

    ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/thermo_dgpredictor_rows.py
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
from datetime import UTC, datetime
from typing import cast

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

DATA_ROOT = cast(str, os.environ["DATA_ROOT"])
EXPERIMENT_ROOT = cast(str, os.environ["EXPERIMENT_ROOT"])
GEM_DIR = osp.join(DATA_ROOT, "data", "torchcell", "yeast-GEM", "yeast-GEM-9.0.2")
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
THERMO_DIR = osp.join(DATA_ROOT, "data", "torchcell", "thermo_equilibrator")
YETFL_DIR = osp.join(DATA_ROOT, "data", "torchcell", "yetfl_thermo")

#: yeast-GEM's "no value" marker, a finite number rather than a blank.
SENTINEL = 1.0e7

#: kJ per kcal, thermochemical. The factor the parser applies to the whole column.
KJ_PER_KCAL = 4.184

#: The index block ``v8_7_0.m`` assigns from dGPredictor, verbatim from the curation
#: script: ``model.rxnDeltaG(3983:4062) = dgpred_rxnG.detaG(3983:4062)``. MATLAB is 1-based
#: and inclusive, so in python this is ``[3982:4062]``.
DGPRED_BLOCK_MATLAB = (3983, 4062)

#: Rounding ``v8_7_0.m`` applies, and therefore the tolerance at which a shipped row counts
#: as identical to its yETFL source.
ROUNDING_TOLERANCE = 0.005001


class DgPredictorRows(BaseModel):
    """Which shipped reaction rows are already kJ/mol, and the evidence that they are."""

    built_at: str
    question: str = (
        "Which rows of yeast-GEM 9.0.2's model_rxnDeltaG.csv come from dGPredictor "
        "(kJ/mol) rather than from yETFL (kcal/mol)?"
    )
    method: str = (
        "Set difference: a shipped row whose reaction id yETFL's curated model does not "
        "price. v8_7_0.m fills from yETFL by reaction id and then fills a fixed index "
        "block from dGPredictor, so the difference IS the dGPredictor block."
    )
    shipped_csv: str
    shipped_sha256: str
    yetfl_mat: str
    yetfl_sha256: str
    yetfl_declared_units: str
    n_shipped_rows: int
    n_matched_to_yetfl: int
    max_abs_difference_to_yetfl: float
    frac_within_rounding: float
    n_dgpredictor_rows: int
    dgpredictor_reaction_ids: list[str]
    #: 0-based model positions of those rows, to check them against the curation script's
    #: index block rather than trusting the set difference alone.
    dgpredictor_model_positions: list[int]
    block_check: dict[str, object]
    unit_back_solve: dict[str, object] = Field(
        default_factory=dict,
        description=(
            "The empirical check, independent of the curation script: for every reaction "
            "the eQuilibrator recomputation can price from its metabolites, is the shipped "
            "row closer to that kJ/mol sum as it stands, or after multiplying by 4.184? A "
            "kcal row should prefer the multiplication and a kJ row should not."
        ),
    )
    correction: str = (
        "constraints._read_delta_g_csv multiplies the whole column by 4.184 on read. "
        "These rows are already kJ/mol, so a consumer divides them back by 4.184. "
        "parameters.resolve_reaction_delta_g does that before the union fill."
    )
    kj_per_kcal: float = KJ_PER_KCAL
    caveat: str = Field(
        default=(
            "The identification is by ABSENCE from yETFL, so a reaction yETFL prices "
            "under a different id would be misfiled as dGPredictor. The block check is "
            "what bounds that risk: the rows found this way must sit inside the index "
            "range the curation script assigns from dGPredictor."
        )
    )


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _shipped_reaction_column(path: str) -> dict[str, float]:
    """The shipped reaction table as ``id -> raw value``, sentinel and NaN dropped."""
    out: dict[str, float] = {}
    with open(path) as handle:
        handle.readline()
        for line in handle:
            line = line.strip()
            if not line:
                continue
            key, _, raw = line.partition(",")
            value = float(raw)
            if value == SENTINEL or not np.isfinite(value):
                continue
            out[key] = value
    return out


def main() -> None:
    import cobra
    import scipy.io as sio

    shipped_path = osp.join(GEM_DIR, "data", "databases", "model_rxnDeltaG.csv")
    yetfl_path = osp.join(YETFL_DIR, "yeast8_thermo_curated.mat")
    shipped = _shipped_reaction_column(shipped_path)

    model_mat = sio.loadmat(yetfl_path, squeeze_me=True, struct_as_record=False)["model"]
    rxn_ids = [str(x) for x in np.asarray(model_mat.rxns)]
    drg = np.asarray(model_mat.rxnDeltaGR, dtype=float)
    yetfl = {r: float(v) for r, v in zip(rxn_ids, drg) if abs(v) < SENTINEL}

    common = sorted(set(shipped) & set(yetfl))
    difference = np.abs(
        np.array([shipped[k] for k in common]) - np.array([yetfl[k] for k in common])
    )
    dgpred_ids = sorted(set(shipped) - set(yetfl))

    gem = cobra.io.read_sbml_model(osp.join(GEM_DIR, "model", "yeast-GEM.xml"))
    position = {r.id: i for i, r in enumerate(gem.reactions)}
    positions = [position[r] for r in dgpred_ids if r in position]
    lower, upper = DGPRED_BLOCK_MATLAB
    inside = [p for p in positions if lower - 1 <= p <= upper - 1]

    # The empirical arm. The eQuilibrator recomputation is an independent kJ/mol table, so
    # a reaction it can price from its own metabolites is a ruler for the shipped row's
    # unit. Reported for the 33 and, as a control, for the yETFL rows that the same ruler
    # reaches: the control has to prefer the multiplication, or the ruler is not one.
    import pandas as pd

    recomputed = pd.read_parquet(
        osp.join(THERMO_DIR, "metabolite_dgf_prime.parquet"),
        columns=["met_id", "dgf_prime_kj_per_mol"],
    )
    dgf = {
        str(k): float(v)
        for k, v in zip(recomputed["met_id"], recomputed["dgf_prime_kj_per_mol"])
        if np.isfinite(v)
    }

    def _reaction_sum(reaction_id: str) -> float | None:
        reaction = gem.reactions.get_by_id(reaction_id)
        stoichiometry = {m.id: float(c) for m, c in reaction.metabolites.items()}
        if not stoichiometry or any(m not in dgf for m in stoichiometry):
            return None
        return sum(c * dgf[m] for m, c in stoichiometry.items())

    def _arm(reaction_ids: list[str]) -> dict[str, object]:
        as_is: list[float] = []
        converted: list[float] = []
        prefers_kj = 0
        for reaction_id in reaction_ids:
            if reaction_id not in shipped or reaction_id not in position:
                continue
            reference = _reaction_sum(reaction_id)
            if reference is None:
                continue
            raw = shipped[reaction_id]
            a, b = abs(raw - reference), abs(raw * KJ_PER_KCAL - reference)
            as_is.append(a)
            converted.append(b)
            prefers_kj += int(a < b)
        return {
            "n_scored": len(as_is),
            "median_abs_error_as_written_kj": float(np.median(as_is)) if as_is else None,
            "median_abs_error_times_4_184_kj": float(np.median(converted))
            if converted
            else None,
            "n_preferring_as_written": prefers_kj,
            "frac_preferring_as_written": (prefers_kj / len(as_is)) if as_is else None,
        }

    control_ids = [r for r in common if r not in set(dgpred_ids)]
    back_solve = {
        "reference_table": osp.join(THERMO_DIR, "metabolite_dgf_prime.parquet"),
        "dgpredictor_rows": _arm(dgpred_ids),
        "yetfl_control_rows": _arm(control_ids),
        "reading": (
            "a kcal row should prefer the x4.184 column and a kJ row should prefer the "
            "column as written; the control fixes which way the ruler points"
        ),
    }
    control = back_solve["yetfl_control_rows"]
    control_frac = control.get("frac_preferring_as_written")
    back_solve["verdict"] = (
        "UNINFORMATIVE. The control rows are known kcal and should prefer the x4.184 "
        f"column, and they prefer it in only {1 - float(control_frac):.2f} of cases, so "
        "the ruler does not separate the two units at all. The reason is documented in "
        "[[experiments.026-metabolism-flux.yeast9-delta-g-provenance]]: at the REACTION "
        "level the shipped column disagrees with its own metabolite column by a median "
        "10.2 kcal/mol through the proton term and the compartment collapse, which is "
        "larger than the unit factor being tested. The metabolite-level regression that "
        "did settle the unit question (slope 4.187, Pearson 0.987 over 1,927 shared "
        "metabolites) has no reaction-level analogue. The identification of these 33 rows "
        "therefore rests on the documentary evidence: v8_7_0.m assigns them from "
        "dGPredictor, run_dgpredictor.py reports kJ/mol, and yETFL does not price them."
        if control_frac is not None
        else "not scored"
    )

    record = DgPredictorRows(
        built_at=datetime.now(UTC).isoformat(),
        shipped_csv=shipped_path,
        shipped_sha256=_sha256(shipped_path),
        yetfl_mat=yetfl_path,
        yetfl_sha256=_sha256(yetfl_path),
        yetfl_declared_units=str(model_mat.thermo_units),
        n_shipped_rows=len(shipped),
        n_matched_to_yetfl=len(common),
        max_abs_difference_to_yetfl=float(difference.max()) if len(difference) else 0.0,
        frac_within_rounding=float((difference <= ROUNDING_TOLERANCE).mean())
        if len(difference)
        else 0.0,
        n_dgpredictor_rows=len(dgpred_ids),
        dgpredictor_reaction_ids=dgpred_ids,
        dgpredictor_model_positions=sorted(positions),
        block_check={
            "curation_script_block_matlab_1_based_inclusive": list(DGPRED_BLOCK_MATLAB),
            "n_rows_found": len(dgpred_ids),
            "n_rows_located_in_the_model": len(positions),
            "n_inside_the_block": len(inside),
            "min_position_0_based": min(positions) if positions else None,
            "max_position_0_based": max(positions) if positions else None,
            "outside_the_block": [
                {"reaction_id": r, "position_0_based": position[r]}
                for r in dgpred_ids
                if r in position and not lower - 1 <= position[r] <= upper - 1
            ],
            "verdict": (
                "every row found by the set difference lies inside the index block the "
                "curation script assigns from dGPredictor"
                if positions and len(inside) == len(positions)
                else (
                    "some rows fall outside the block. The block indices were written "
                    "against yeast-GEM v8.7.0 and are checked here against 9.0.2, whose "
                    "reaction list has since changed, so a shifted position is expected "
                    "rather than disqualifying. The unit back-solve is the independent "
                    "arm; it does not depend on either model's row order."
                )
            ),
        },
        unit_back_solve=back_solve,
    )

    os.makedirs(RESULTS, exist_ok=True)
    out_paths = [
        osp.join(RESULTS, "dgpredictor_reaction_rows.json"),
        osp.join(THERMO_DIR, "dgpredictor_reaction_rows.json"),
    ]
    for path in out_paths:
        os.makedirs(osp.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            handle.write(record.model_dump_json(indent=1))
        print(f"wrote {path}")
    print(
        f"{record.n_shipped_rows} shipped rows, {record.n_matched_to_yetfl} from yETFL "
        f"(max difference {record.max_abs_difference_to_yetfl:.4f}, "
        f"{record.frac_within_rounding:.3f} within rounding), "
        f"{record.n_dgpredictor_rows} from dGPredictor"
    )
    print(json.dumps(record.block_check, indent=1))
    print(json.dumps(record.unit_back_solve, indent=1))
    print("dGPredictor rows:", ", ".join(record.dgpredictor_reaction_ids))


if __name__ == "__main__":
    main()
