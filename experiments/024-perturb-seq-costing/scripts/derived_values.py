# experiments/024-perturb-seq-costing/scripts/derived_values.py
# [[experiments.024-perturb-seq-costing.scripts.derived_values]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/024-perturb-seq-costing/scripts/derived_values
"""Numbers the review states in prose that no other script emitted.

The repo rule is that every number in a document comes from a committed script.
An audit found several that were hand arithmetic in the ``.tex``: the value of
rRNA depletion, the sublibrary counts behind the fourth-ligation-round argument,
the cost of a fourth barcode plate, and the transcript-content sensitivity of
Sec. 4.1. Each was individually defensible and none was reproducible, which is
exactly the state the rule exists to prevent.

They live here rather than in the four scripts they draw from because they are
all *compositions* of values those scripts already publish. Putting a
"subtract these two budget rows" helper inside ``cost_model`` would imply the
model owns the claim; it does not, the prose does.

The Sec. 4.1 entry is a correction rather than a transcription. The published
137 / 359 / 699 were evaluated at the two-fold coefficient A = 16.3 together
with phi = 1.1, and phi = 1.1 is the value calibrated to the MEASURED 1.34-fold
coefficient A = 90.9. That pairing implies a floor of A*phi = 18 cells, not the
100 the same section relies on. Both self-consistent pairings are computed here
so the mismatch cannot recur silently.

Run:  python experiments/024-perturb-seq-costing/scripts/derived_values.py
"""

from __future__ import annotations

import json
import math
import os
import os.path as osp

from dotenv import load_dotenv
from pydantic import BaseModel

import cost_data as CD
import cost_model as CM
import design_equation as DE
import method_data as MD

load_dotenv()
RESULTS = osp.join(
    os.environ["EXPERIMENT_ROOT"], "024-perturb-seq-costing", "results"
)

# Total mRNA per cell, the three candidate denominators of table 3.
TRANSCRIPT_TOTALS = {"review (10,500)": 10_500, "Brettner (30,000)": 30_000,
                     "Jackson (60,000)": 60_000}
MOLECULES_PER_GENE = 3.5  # the review's per-gene figure, table 3
SPLITPOOL_DEPTH = 410.0  # mRNA UMIs per cell, Brettner


class Pairing(BaseModel):
    """One self-consistent (effect size, overdispersion) pair for Eq. (5).

    A and phi are NOT independent here. phi is pinned by requiring that the
    100-cell convention come out of the equation's floor, A*phi = 100, so each
    effect size implies its own phi. Mixing an A from one pair with the phi from
    the other is the error this class exists to make visible.
    """

    label: str
    delta_log2: float
    fold: float
    A: float
    phi: float
    floor_cells: float
    cells_per_perturbation: dict[str, float]

    @property
    def spread(self) -> float:
        v = list(self.cells_per_perturbation.values())
        return max(v) / min(v)


def pairing(label: str, delta_log2: float) -> Pairing:
    A = DE.power_coefficient(delta_log2)
    phi = 100.0 / A
    cells = {}
    for name, total in TRANSCRIPT_TOTALS.items():
        p_j = MOLECULES_PER_GENE / total
        cells[name] = A * (1.0 / (SPLITPOOL_DEPTH * p_j) + phi)
    return Pairing(label=label, delta_log2=delta_log2, fold=2.0**delta_log2,
                   A=A, phi=phi, floor_cells=A * phi,
                   cells_per_perturbation=cells)


def transcript_sensitivity() -> dict:
    """Sec. 4.1: how much the total-mRNA denominator moves cells per perturbation."""
    measured = pairing("measured 1.34-fold", DE.DELTA_MEASURED)
    nominal = pairing("nominal two-fold", 1.0)
    # The pairing the document used to print, kept so the correction is legible.
    A_two = DE.power_coefficient(1.0)
    bad = {n: A_two * (1.0 / (SPLITPOOL_DEPTH * (MOLECULES_PER_GENE / t)) + measured.phi)
           for n, t in TRANSCRIPT_TOTALS.items()}
    return {
        "measured": measured.model_dump(),
        "nominal_two_fold": nominal.model_dump(),
        "spread_measured": measured.spread,
        "spread_nominal": nominal.spread,
        "superseded_incoherent_pairing": {
            "note": "A from the two-fold pair with phi from the measured pair; "
                    "implied floor is only %.0f cells" % (A_two * measured.phi),
            "values": bad,
        },
    }


def depletion_lever() -> dict:
    """Sec. 3.5 / 5.5: what rRNA depletion is worth at genome scale.

    A difference of two rows of the budget table rather than a figure any source
    states, and it is quoted in four places, so it gets computed once.
    """
    design = CM.ScreenDesign(cells_per_gene=250)
    plain = CM.budget_for(design, CM.SPLITSEQ_PUBLISHED)
    depleted = CM.budget_for(design, CM.SPLITSEQ_DEPLETED)
    return {
        "cells_per_gene": 250,
        "splitpool_total_usd": plain.recurring_usd,
        "splitpool_depleted_total_usd": depleted.recurring_usd,
        "saving_usd": plain.recurring_usd - depleted.recurring_usd,
    }


def sublibrary_counts(cells_per_run: float = 480_000.0,
                      collision_target: float = 0.01) -> dict:
    """Sec. 5.4: how many sublibraries a collision target forces, and its cost.

    Splitting into S sublibraries divides the cells drawing from one barcode
    space, because the Illumina index resolves two cells that took identical
    plate paths into different sublibraries. So the constraint is cells PER
    SUBLIBRARY, and S is what delivers it.
    """
    out = {}
    for rounds in (3, 4):
        space = 96**rounds
        # 1 - ((B-1)/B)^(n-1) = target  ->  n
        n_max = 1.0 + math.log1p(-collision_target) / math.log((space - 1) / space)
        needed = max(1, math.ceil(cells_per_run / n_max))
        out[f"{rounds}_rounds"] = {
            "barcode_space": space,
            "cells_per_sublibrary_at_target": n_max,
            "sublibraries_needed": needed,
            "cost_usd": needed * CD.BRETTNER_PER_SUBLIBRARY,
        }
    # With four rounds the collision target stops binding, so the real floor is
    # the protocol's own working limit on cells per sublibrary.
    pcr_limited = math.ceil(cells_per_run / MD.CELLS_PER_SUBLIBRARY_AT_CEILING)
    out["four_rounds_pcr_limited"] = {
        "cells_per_sublibrary": MD.CELLS_PER_SUBLIBRARY_AT_CEILING,
        "sublibraries_needed": pcr_limited,
        "cost_usd": pcr_limited * CD.BRETTNER_PER_SUBLIBRARY,
    }
    return out


def fourth_plate() -> dict:
    """Sec. 5.4: what adding a fourth barcode plate actually costs.

    One-time is the three-plate purchase divided evenly. The per-run term is the
    part that was never written down: a fourth BARCODE round is a third
    LIGATION round, so the ligase line -- which covers rounds 2 and 3 -- grows by
    half, and one more plate is drawn down.
    """
    items = {i.name: i for i in CD.BRETTNER_ITEMS}
    plates_total = items["IDT barcode plates (RT r1 + ligation r2 + r3)"].usd
    oligos_per_run = items["Barcoding oligos drawn from the plates"].usd
    ligase_per_run = items["T4 DNA Ligase (NEB M0202M)"].usd
    one_time = plates_total / 3.0
    # 3 plates -> 3 draws per run; ligase covers the 2 existing ligations.
    extra_oligo = oligos_per_run / 3.0
    extra_ligase = ligase_per_run / 2.0
    return {
        "one_time_usd": one_time,
        "per_run_oligo_usd": extra_oligo,
        "per_run_ligase_usd": extra_ligase,
        "per_run_usd": extra_oligo + extra_ligase,
        "plate_uses": 215,
    }


def plate_set_lifetime(cells_per_run: float = 480_000.0) -> dict:
    """Sec. 5.1: how much screening one $7,699 plate set actually buys.

    Raised in review: the start-up cost is amortized over 215 protocol runs, but
    the plates are retired by freeze--thaw rather than by depletion, so if a plate
    degrades before 215 uses the amortization is fiction and the per-run cost is
    higher than quoted. Worth putting a number on rather than asserting.

    The two failure modes are separable and the arithmetic says so:

    * DEPLETION is a volume budget. 215 round-3 withdrawals is the binding one of
      the three capacities Brettner et al.\\ state, and it is a hard count.
    * FREEZE--THAW is a cycle budget on whichever plate is being pipetted from,
      and aliquoting decouples the two. Split the source into ``w`` working
      plates in ONE thaw and the source sees one cycle whatever happens next,
      while each working plate serves 215/w runs and therefore sees 215/w cycles.
      So the working-plate count needed to keep every plate under a tolerance of
      ``f`` cycles is ceil(215/f) -- 22 plates at f=10, 3 at f=100.

    The consequence for the cost argument is the useful part: aliquoting is the
    lever, it is nearly free (empty plates and one thaw), and it converts a
    freeze--thaw limit into a plate-count decision rather than into a shorter
    plate life. The tolerance ``f`` itself is NOT measured here and is not in the
    mirror; the tolerances swept are illustrative, and the point that survives
    whichever value is true is the shape, not the number.
    """
    uses = 215  # the round-3 plate, the binding capacity of the three
    # Recomputed rather than transcribed, so it tracks the budget model: the
    # depleted split-pool row at the 250-cells-per-gene tier.
    runs_per_screen = CM.budget_for(
        CM.ScreenDesign(cells_per_gene=250), CM.SPLITSEQ_DEPLETED
    ).n_batches
    return {
        "plate_uses": uses,
        "cells_per_run": cells_per_run,
        "cells_over_plate_life": uses * cells_per_run,
        "runs_per_genome_screen_250": runs_per_screen,
        "genome_screens_per_plate_set": uses / runs_per_screen,
        "working_plates_needed": {
            f"tolerance_{f}_cycles": math.ceil(uses / f) for f in (5, 10, 25, 100)
        },
        "freeze_thaw_tolerance_measured": False,
    }


def sublibrary_item_reconciliation() -> dict:
    """Sec. 3.1: why $55 per sublibrary is not the sum of the itemized lines.

    $55 is Brettner et al.'s own rolled-up figure and the line items sum to $53.
    The gap is the leeway their sentence says it includes, so this is a quoted
    total against an itemized subtotal rather than a discrepancy.
    """
    items = {i.name: i for i in CD.BRETTNER_ITEMS}
    itemized = sum(i.usd for i in CD.BRETTNER_ITEMS if i.scaling == "per_sublibrary")
    return {
        "itemized_usd": itemized,
        "quoted_usd": CD.BRETTNER_PER_SUBLIBRARY,
        "leeway_usd": CD.BRETTNER_PER_SUBLIBRARY - itemized,
        "quote": CD.BRETTNER_TOTALS_QUOTE,
    }


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    ts = transcript_sensitivity()

    print("--- Sec. 4.1: transcript-content sensitivity -------------------")
    for key in ("measured", "nominal_two_fold"):
        p = ts[key]
        print(f"  {p['label']:22s} A={p['A']:6.1f}  phi={p['phi']:.2f}  "
              f"floor={p['floor_cells']:.0f} cells")
        for name, v in p["cells_per_perturbation"].items():
            print(f"      {name:20s} {v:8.0f} cells/perturbation")
    print(f"  spread, measured pairing: {ts['spread_measured']:.1f}x")
    bad = ts["superseded_incoherent_pairing"]
    print(f"  SUPERSEDED (do not quote): "
          f"{', '.join(f'{v:.0f}' for v in bad['values'].values())}  -- {bad['note']}")

    dl = depletion_lever()
    print("\n--- Sec. 3.5/5.5: value of rRNA depletion ----------------------")
    print(f"  ${dl['splitpool_total_usd']:,.0f} undepleted less "
          f"${dl['splitpool_depleted_total_usd']:,.0f} depleted = "
          f"${dl['saving_usd']:,.0f}")

    sc = sublibrary_counts()
    print("\n--- Sec. 5.4: sublibraries forced by a 1% collision target ------")
    for k, v in sc.items():
        print(f"  {k:26s} {v['sublibraries_needed']:>4d} sublibraries  "
              f"${v['cost_usd']:,.0f} per run")

    fp = fourth_plate()
    print("\n--- Sec. 5.4: cost of a fourth barcode plate -------------------")
    print(f"  one time ${fp['one_time_usd']:,.2f}; per run "
          f"${fp['per_run_usd']:.2f} "
          f"(oligo ${fp['per_run_oligo_usd']:.2f} + ligase "
          f"${fp['per_run_ligase_usd']:.2f})")

    pl = plate_set_lifetime()
    print("\n--- Sec. 5.1: what one plate set buys, and freeze-thaw -----------")
    print(f"  {pl['plate_uses']} runs x {pl['cells_per_run']:,.0f} cells = "
          f"{pl['cells_over_plate_life']:,.0f} cells barcoded")
    print(f"  {pl['runs_per_genome_screen_250']} runs per genome-scale screen at "
          f"250 cells/gene -> {pl['genome_screens_per_plate_set']:.1f} screens")
    for k, v in pl["working_plates_needed"].items():
        print(f"  {k:24s} needs {v:>3d} working plates")

    si = sublibrary_item_reconciliation()
    print("\n--- Sec. 3.1: $55 per sublibrary vs the itemized lines ----------")
    print(f"  itemized ${si['itemized_usd']:.0f}, quoted ${si['quoted_usd']:.0f}, "
          f"leeway ${si['leeway_usd']:.0f}")

    out = {
        "transcript_sensitivity": ts,
        "depletion_lever": dl,
        "sublibrary_counts": sc,
        "fourth_plate": fp,
        "plate_set_lifetime": pl,
        "sublibrary_item_reconciliation": si,
    }
    path = osp.join(RESULTS, "derived_values.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
