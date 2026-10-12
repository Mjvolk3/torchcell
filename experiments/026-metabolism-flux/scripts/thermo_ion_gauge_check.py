# experiments/026-metabolism-flux/scripts/thermo_ion_gauge_check.py
# [[experiments.026-metabolism-flux.scripts.thermo_ion_gauge_check]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/thermo_ion_gauge_check.py

r"""Prove which reaction energies the ion table's absolute values can and cannot change.

The ion stage puts a measured absolute formation energy on potassium, sodium, calcium,
zinc, copper and cadmium. A reader is entitled to ask how much of the resulting reaction
table depends on those particular numbers, because a value nobody can check would be a
liability wherever it does not cancel.

The claim is that in a reaction elementally balanced in the ion's element, the ion's
coefficients sum to zero and its absolute energy drops out exactly, as does its
ionic-strength term whenever the compartments share an ionic strength. That is an
arithmetic claim, so it is MEASURED rather than argued: the whole pipeline is run twice,
once with the table as built and once with every ion shifted by a large constant, and the
reactions whose energy moves are listed against the reactions that are elementally
unbalanced.

If the claim holds, the two sets are the same set, and the reactions that move are exactly
the exchange, sink and biomass pseudo-reactions where a standard reaction energy has no
physical meaning in the first place.

Writes ``results/thermo_ion_gauge_check.json``.

Run from the repository root under the torchcell env::

    ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/thermo_ion_gauge_check.py
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
from datetime import UTC, datetime
from typing import cast

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from torchcell.metabolism.thermo_pipeline import ThermoRoute, run_thermo_pipeline

load_dotenv()

EXPERIMENT_ROOT = cast(str, os.environ["EXPERIMENT_ROOT"])
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
OUT_PATH = osp.join(RESULTS, "thermo_ion_gauge_check.json")

#: Large enough that any dependence is unmistakable and small enough to stay exact in
#: double precision against reaction energies of order 1e3 kJ/mol.
GAUGE_SHIFT_KJ = 1000.0

#: Below this, a difference is float noise rather than a dependence.
MOVED_TOLERANCE_KJ = 1e-6


class MovedReaction(BaseModel):
    """One reaction whose energy depends on the ion gauge."""

    reaction_id: str
    delta_r_g0_kj_per_mol: float
    change_under_gauge_shift_kj_per_mol: float
    elementally_balanced: bool
    imbalance: dict[str, float]
    reaction_string: str


class GaugeCheckReport(BaseModel):
    """What the ion table's absolute values do and do not change."""

    built_at: str
    gauge_shift_kj_per_mol: float = GAUGE_SHIFT_KJ
    claim: str = (
        "in a reaction elementally balanced in the ion's element the ion's absolute "
        "formation energy cancels exactly, so only elementally unbalanced reactions can "
        "move under a change of gauge"
    )
    n_reactions: int
    n_priced_baseline: int
    n_priced_shifted: int
    n_reactions_touching_a_tabled_ion: int
    n_moved: int
    n_moved_and_unbalanced: int
    n_moved_but_balanced: int
    claim_holds: bool
    max_change_among_balanced_kj_per_mol: float
    moved: list[MovedReaction] = Field(default_factory=list)
    reading: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surrogate-spread", type=float, default=5.0)
    parser.add_argument("--no-surrogate", action="store_true")
    parser.add_argument("--no-repeat-unit", action="store_true")
    args = parser.parse_args()

    import cobra

    # The driver owns the stage list and the config; importing it keeps the two runs
    # identical in every respect except the gauge.
    import thermo_pipeline_yeast9 as driver

    model = cobra.io.read_sbml_model(osp.join(driver.GEM_DIR, "model", "yeast-GEM.xml"))
    config = driver.build_config(model)

    baseline = run_thermo_pipeline(
        model, config, driver.build_stages(args), driver.formation_sigma()
    )
    shifted = run_thermo_pipeline(
        model,
        config,
        driver.build_stages(args, ion_gauge_shift_kj_per_mol=GAUGE_SHIFT_KJ),
        driver.formation_sigma(),
    )

    ion_met_ids = {
        price.met_id
        for price in baseline.metabolites
        if price.route is ThermoRoute.LITERATURE_TABLE
    }
    touching = {
        reaction.id
        for reaction in model.reactions
        if {m.id for m in reaction.metabolites} & ion_met_ids
    }

    before = {r.reaction_id: r.delta_r_g0_kj_per_mol for r in baseline.reactions}
    after = {r.reaction_id: r.delta_r_g0_kj_per_mol for r in shifted.reactions}

    moved: list[MovedReaction] = []
    max_balanced_change = 0.0
    for reaction_id, value in before.items():
        other = after.get(reaction_id)
        if value is None or other is None:
            continue
        change = other - value
        if abs(change) <= MOVED_TOLERANCE_KJ:
            continue
        reaction = model.reactions.get_by_id(reaction_id)
        imbalance = {
            k: float(v)
            for k, v in reaction.check_mass_balance().items()
            if abs(v) > 1e-9
        }
        balanced = not imbalance
        if balanced:
            max_balanced_change = max(max_balanced_change, abs(change))
        moved.append(
            MovedReaction(
                reaction_id=reaction_id,
                delta_r_g0_kj_per_mol=float(value),
                change_under_gauge_shift_kj_per_mol=float(change),
                elementally_balanced=balanced,
                imbalance=imbalance,
                reaction_string=reaction.reaction[:160],
            )
        )

    n_moved_unbalanced = sum(1 for m in moved if not m.elementally_balanced)
    n_moved_balanced = sum(1 for m in moved if m.elementally_balanced)
    report = GaugeCheckReport(
        built_at=datetime.now(UTC).isoformat(),
        n_reactions=len(model.reactions),
        n_priced_baseline=baseline.reaction_counts.n_priced,
        n_priced_shifted=shifted.reaction_counts.n_priced,
        n_reactions_touching_a_tabled_ion=len(touching),
        n_moved=len(moved),
        n_moved_and_unbalanced=n_moved_unbalanced,
        n_moved_but_balanced=n_moved_balanced,
        claim_holds=n_moved_balanced == 0,
        max_change_among_balanced_kj_per_mol=max_balanced_change,
        moved=sorted(moved, key=lambda m: -abs(m.change_under_gauge_shift_kj_per_mol)),
        reading=(
            f"{len(touching)} reactions touch a tabled ion and {len(moved)} of them move "
            f"when every ion is shifted by {GAUGE_SHIFT_KJ:g} kJ/mol. "
            + (
                "Every one that moves is elementally unbalanced, so the absolute values "
                "change no balanced reaction's energy and the table's contribution to the "
                "usable part of the reaction table is exactly its coverage, not its "
                "numbers."
                if n_moved_balanced == 0
                else "SOME BALANCED REACTIONS MOVE, so the cancellation claim is false as "
                "stated and the absolute values matter more widely than argued."
            )
        ),
    )
    with open(OUT_PATH, "w") as handle:
        handle.write(report.model_dump_json(indent=1))
    print(f"wrote {OUT_PATH}")
    print(f"  {report.reading}")
    for row in report.moved[:12]:
        print(
            f"    {row.reaction_id:8s} {row.change_under_gauge_shift_kj_per_mol:+10.1f}"
            f" balanced={row.elementally_balanced} imbalance={row.imbalance}"
        )


if __name__ == "__main__":
    main()
