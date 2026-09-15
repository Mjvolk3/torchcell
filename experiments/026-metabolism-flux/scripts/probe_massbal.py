# experiments/026-metabolism-flux/scripts/probe_massbal.py
# [[experiments.026-metabolism-flux.physics-gates-2026-09-14]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/probe_massbal
r"""Is the GEM itself elementally and charge balanced?

WHY IT MATTERS TO THE FLUX LAYER. Mass balance in the layer is ``Sv = 0``, which is
balance of METABOLITE COUNTS and says nothing about whether each reaction conserves atoms
or charge. A charge-unbalanced reaction has no well-defined transformed Gibbs energy, so
its tabulated standard energy is not a measurement of anything and pricing the second law
on it asserts a direction the chemistry does not support. A reaction with a formula-less
participant cannot be checked at all.

Counted here with cobra's own ``check_mass_balance`` over the internal reactions, i.e.
excluding the exchanges and the biomass reaction, which cross the system boundary or are
lumped and are not expected to balance.

Run from the repo root::

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/probe_massbal.py

Writes ``experiments/026-metabolism-flux/results/probe_massbal.json``.
"""

from __future__ import annotations

import collections
import json
import os
import os.path as osp
import time

from dotenv import load_dotenv

load_dotenv()

from torchcell.metabolism.yeast_GEM import YeastGEM  # noqa: E402

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")


def main() -> None:
    started = time.time()
    source = YeastGEM()
    model = source.model
    exchanges = {r.id for r in model.reactions if len(r.metabolites) == 1}
    biomass = {r.id for r in model.reactions if abs(r.objective_coefficient) > 0}

    n_unbalanced = 0
    n_charge = 0
    n_element_entries = 0
    n_no_formula = 0
    by_key: collections.Counter[str] = collections.Counter()
    examples: list[dict[str, object]] = []
    n_internal = 0
    for reaction in model.reactions:
        if reaction.id in exchanges or reaction.id in biomass:
            continue
        n_internal += 1
        if any(m.formula in (None, "") for m in reaction.metabolites):
            n_no_formula += 1
            continue
        imbalance = reaction.check_mass_balance()
        if not imbalance:
            continue
        n_unbalanced += 1
        for key in imbalance:
            by_key[key] += 1
            if key == "charge":
                n_charge += 1
            else:
                n_element_entries += 1
        examples.append(
            {
                "reaction": reaction.id,
                "name": reaction.name,
                "imbalance": {k: round(float(v), 4) for k, v in imbalance.items()},
            }
        )

    payload = {
        "definitions": {
            "internal_reactions": (
                "every reaction that is neither a single-metabolite exchange nor the "
                "objective (biomass) reaction."
            ),
            "no_formula": (
                "reactions with at least one participant carrying no chemical formula, so "
                "the balance cannot be checked at all."
            ),
        },
        "internal_reactions": n_internal,
        "no_formula": n_no_formula,
        "unbalanced": n_unbalanced,
        "charge_imbalance_reactions": n_charge,
        "element_imbalance_entries": n_element_entries,
        "by_key": dict(by_key.most_common(15)),
        "examples": examples[:25],
    }
    print(
        json.dumps(
            {k: v for k, v in payload.items() if k not in {"examples", "definitions"}},
            indent=2,
        )
    )
    out_path = osp.join(RESULTS_DIR, "probe_massbal.json")
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote {out_path} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
