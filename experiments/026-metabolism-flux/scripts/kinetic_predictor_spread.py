# experiments/026-metabolism-flux/scripts/kinetic_predictor_spread.py
# [[experiments.026-metabolism-flux.scripts.kinetic_predictor_spread]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/kinetic_predictor_spread.py

r"""Where the admitted predictors disagree, and whether the disagreement is concentrated.

THE QUESTION
------------
The predicted tier is a median over the predictors the within-class study admits, and those
predictors disagree by more than a decade on a large minority of catalytic units. A median
over three numbers spanning two decades is not the same object as a median over three
numbers spanning a tenth of one, so the flux layer needs to know whether the wide units are
scattered uniformly, in which case the spread is a global uncertainty, or concentrated in
identifiable chemistry, in which case it is a targeted one.

Three groupings, each a property of the unit that is known before any predictor runs:

**EC class**, the first digit of the reaction's EC number. Oxidoreductases, transferases and
hydrolases have very different amounts of training data behind them.

**Substrate size**, the heavy-atom count of the unit's substrate, taken as the median over
the substrates the unit's reaction consumes. A predictor's molecular encoder is trained on a
size distribution, and a lipid with 60 heavy atoms is outside it.

**Complex size**, the number of genes in the catalytic unit. A unit's value is the min over
its subunits, so a complex takes a min over several predictions and its spread is a
different statistic from a monomer's.

WHAT IS COMPARED, AND WHAT IS NOT
----------------------------------
Spread is the range in :math:`\log_{10}` across the ADMITTED predictors for that parameter,
per unit, on units where all of them emit a value. For :math:`k_{cat}` that is three
predictors. For :math:`K_M` the code path
(:meth:`~torchcell.metabolism.parameters.PredictorSelection.from_study`) admits EZKinetics
alone, so a spread over the admitted set does not exist; the :math:`K_M` panel is therefore
computed over EZKinetics and EITLEM and labeled as such, since that is the pair the coverage
figure's alternative selection rule keeps. The two rules are not reconciled here; that is a
separate decision and it is flagged rather than papered over.

Writes ``results/kinetic_predictor_spread.json`` plus the figure. Run from the worktree root
after ``improve_kinetic_coverage.py``, so the new units are included.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
from datetime import UTC, datetime
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator
from pydantic import BaseModel, ConfigDict

from torchcell.metabolism.constraints import build_gem_tensors
from torchcell.metabolism.kinetics_pipeline import sha256_of
from torchcell.metabolism.parameters import (
    KINETICS_MIRROR,
    KINETICS_VERSION,
    PLAUSIBILITY_RESULT,
    PredictorSelection,
    _per_unit_from_predictor,
)
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
KINETICS_ROOT = osp.join(DATA_ROOT, KINETICS_MIRROR)
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "026-metabolism-flux")

#: The K_M pair the coverage figure's alternative selection rule keeps. Used only because
#: the code path's rule admits one predictor for K_M and a spread needs two.
KM_PAIR = ["ezkinetics", "eitlem"]

#: Heavy-atom bins. The edges are where the chemistry changes rather than round numbers:
#: under 12 is a central-metabolism intermediate, 12 to 30 a cofactor-sized molecule, 30 to
#: 60 a nucleotide cofactor or short acyl species, and over 60 a lipid or a CoA thioester.
SIZE_BINS = [(0, 12), (12, 30), (30, 60), (60, 10_000)]
SIZE_LABELS = ["<12", "12-30", "30-60", ">60"]

#: EC classes are labeled by their digit alone. Spelling them out on the tick collides at
#: this panel width, and the names belong in the caption where they are typeset at body
#: size: 1 oxidoreductase, 2 transferase, 3 hydrolase, 4 lyase, 5 isomerase, 6 ligase,
#: 7 translocase.
EC_LABELS = {
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "none": "none",
}


class GroupSpread(BaseModel):
    """Spread statistics for one group of catalytic units."""

    model_config = ConfigDict(extra="forbid")

    group: str
    n_units: int
    median_decades: float
    p90_decades: float
    frac_over_1_decade: float
    ci95_over_1_decade: tuple[float, float]


class SpreadReport(BaseModel):
    """The whole analysis, one object."""

    model_config = ConfigDict(extra="forbid")

    built_at: str
    sources: dict[str, str]
    parameters: dict[str, dict[str, Any]]


def wilson(k: int, n: int) -> tuple[float, float]:
    """Wilson 95 percent interval, so a group of 30 is not read like a group of 900."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.959964
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (float(center - half), float(center + half))


def summarize(spread: np.ndarray, label: str) -> GroupSpread:
    over = spread > 1.0
    return GroupSpread(
        group=label,
        n_units=int(spread.size),
        median_decades=float(np.median(spread)) if spread.size else float("nan"),
        p90_decades=float(np.percentile(spread, 90)) if spread.size else float("nan"),
        frac_over_1_decade=float(over.mean()) if spread.size else float("nan"),
        ci95_over_1_decade=wilson(int(over.sum()), int(spread.size)),
    )


def heavy_atoms(smiles: str) -> int | None:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else mol.GetNumHeavyAtoms()


def unit_substrate_size(parameter: str, n_units: int) -> np.ndarray:
    """Median heavy-atom count of each unit's substrates, NaN where unknown.

    Read from the predictor table itself rather than re-resolved from the GEM, so the size
    is the size of the molecule the predictor was actually shown.
    """
    frame = pd.read_parquet(
        osp.join(KINETICS_ROOT, "eitlem", KINETICS_VERSION, f"{parameter}.parquet"),
        columns=["unit_id", "smiles"],
    ).drop_duplicates()
    sizes = {s: heavy_atoms(s) for s in frame["smiles"].unique()}
    frame["n_heavy"] = frame["smiles"].map(sizes)
    per_unit = frame.groupby("unit_id")["n_heavy"].median()
    out = np.full(n_units, np.nan)
    out[per_unit.index.to_numpy().astype(int)] = per_unit.to_numpy()
    return out


def main() -> None:
    # The admitted set is read from a plausibility file, and scoring a new predictor writes
    # a versioned sibling of that file rather than replacing the one a running training
    # cell resolves against. `--suffix` selects the sibling and names the output to match,
    # so a spread computed over a different admitted set can never be mistaken for the
    # shipped one.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suffix",
        default="",
        help="Appended to the plausibility input and the outputs, e.g. `_with_catpred`.",
    )
    args = parser.parse_args()
    source = YeastGEM()
    model = source.model
    gem = build_gem_tensors(model, model_dir=source.model_dir)
    units = gem.catalytic_units
    n_units = units.n_units

    genes_per_unit = np.zeros(n_units, dtype=int)
    for unit, _ in units.unit_gene_index.t().tolist():
        genes_per_unit[int(unit)] += 1
    ec_class = []
    for unit_id in range(n_units):
        reaction = model.reactions[int(units.unit_reaction[unit_id])]
        codes = (reaction.annotation or {}).get("ec-code")
        codes = codes if isinstance(codes, list) else ([codes] if codes else [])
        ec_class.append(str(codes[0]).split(".")[0] if codes else "none")
    ec_class = np.array(ec_class)

    plausibility = osp.join(
        EXPERIMENT_ROOT, PLAUSIBILITY_RESULT.replace(".json", f"{args.suffix}.json")
    )
    sources: dict[str, str] = {plausibility: sha256_of(plausibility)}
    report: dict[str, Any] = {}

    for parameter in ("k_cat", "K_M"):
        selection = PredictorSelection.from_study(parameter, plausibility)
        names = selection.selected if len(selection.selected) > 1 else KM_PAIR
        rule = (
            "predictors the code path admits"
            if len(selection.selected) > 1
            else "the coverage figure's alternative pair; the code path admits one "
            "predictor for this parameter, and one predictor has no spread"
        )
        stacked = []
        for name in names:
            path = osp.join(KINETICS_ROOT, name, KINETICS_VERSION, f"{parameter}.parquet")
            values, _ = _per_unit_from_predictor(path, parameter, n_units)
            stacked.append(values)
            sources[path] = sha256_of(path)
        matrix = np.vstack(stacked)
        complete = np.isfinite(matrix).all(axis=0)
        spread = np.full(n_units, np.nan)
        spread[complete] = matrix[:, complete].max(axis=0) - matrix[:, complete].min(
            axis=0
        )
        size = unit_substrate_size(parameter, n_units)

        by_ec = [
            summarize(spread[complete & (ec_class == code)], EC_LABELS[code])
            for code in sorted(set(ec_class))
            if int((complete & (ec_class == code)).sum()) >= 20
        ]
        by_size = []
        for (low, high), label in zip(SIZE_BINS, SIZE_LABELS):
            selector = complete & (size >= low) & (size < high)
            if int(selector.sum()) >= 20:
                by_size.append(summarize(spread[selector], label))
        by_complex = []
        for label, selector in (
            ("1", genes_per_unit == 1),
            ("2", genes_per_unit == 2),
            ("3+", genes_per_unit >= 3),
        ):
            joint = complete & selector
            if int(joint.sum()) >= 20:
                by_complex.append(summarize(spread[joint], label))

        report[parameter] = {
            "predictors": names,
            "selection_note": rule,
            "n_units_all_predictors": int(complete.sum()),
            "overall": summarize(spread[complete], "all").model_dump(),
            "by_ec_class": [g.model_dump() for g in by_ec],
            "by_substrate_heavy_atoms": [g.model_dump() for g in by_size],
            "by_complex_size": [g.model_dump() for g in by_complex],
        }

    out_path = osp.join(RESULTS_DIR, f"kinetic_predictor_spread{args.suffix}.json")
    with open(out_path, "w") as handle:
        handle.write(
            SpreadReport(
                built_at=datetime.now(UTC).isoformat(),
                sources=sources,
                parameters=report,
            ).model_dump_json(indent=2)
        )

    plot(report, args.suffix)
    for parameter, block in report.items():
        print(f"{parameter}: {block['predictors']} n={block['n_units_all_predictors']}")
        print(f"  overall {block['overall']}")
        for key in ("by_ec_class", "by_substrate_heavy_atoms", "by_complex_size"):
            for group in block[key]:
                print(
                    f"  {key:26} {group['group']:14} n={group['n_units']:5} "
                    f"median={group['median_decades']:.2f} "
                    f"frac>1={group['frac_over_1_decade']:.3f}"
                )
    print(f"-> {out_path}")


def panel(axis: Any, groups: list[dict], color: str, xlabel: str) -> None:
    """One grouping as a bar of the fraction above a decade, with Wilson intervals."""
    positions = np.arange(len(groups))
    fractions = [g["frac_over_1_decade"] for g in groups]
    lower = [f - g["ci95_over_1_decade"][0] for f, g in zip(fractions, groups)]
    upper = [g["ci95_over_1_decade"][1] - f for f, g in zip(fractions, groups)]
    axis.bar(positions, fractions, width=0.7, color=color, edgecolor="black", linewidth=0.4)
    axis.errorbar(
        positions, fractions, yerr=[lower, upper], fmt="none", ecolor="black",
        elinewidth=0.5, capsize=1.2,
    )
    axis.set_xticks(positions)
    # Group sizes are NOT written on the tick. Seven EC classes at panel width put
    # "n=500n=1413n=888" on one line, and a count belongs in the caption and the result
    # table where it is typeset at body size and can be read.
    axis.set_xticklabels([g["group"] for g in groups], rotation=0)
    axis.set_xlabel(xlabel)
    axis.set_ylim(0, 1)
    axis.yaxis.set_major_locator(MultipleLocator(0.2))
    axis.yaxis.set_minor_locator(MultipleLocator(0.1))
    axis.tick_params(which="minor", length=0)
    axis.grid(axis="y", which="both", linewidth=0.3, color="0.85")
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)


def plot(report: dict[str, Any], suffix: str = "") -> None:
    """One row of three panels for k_cat, one for K_M."""
    apply_paper_style()
    width = mm_to_in(PANEL_WIDTHS_MM["full"])
    # Explicit margins rather than a constrained layout: the panel letters sit OUTSIDE the
    # axes, a constrained layout does not reserve room for anything outside them, and the
    # letters of the top row were clipped off the canvas.
    figure, axes = plt.subplots(2, 3, figsize=(width, mm_to_in(80.0)))
    figure.subplots_adjust(
        left=0.085, right=0.995, top=0.92, bottom=0.13, wspace=0.22, hspace=0.55
    )
    for row, parameter in enumerate(("k_cat", "K_M")):
        block = report[parameter]
        color = PLOT_PALETTE[row]
        panel(axes[row][0], block["by_ec_class"], color, "EC class")
        panel(
            axes[row][1],
            block["by_substrate_heavy_atoms"],
            color,
            "substrate heavy atoms",
        )
        panel(axes[row][2], block["by_complex_size"], color, "genes in the unit")
        axes[row][0].set_ylabel(
            f"{'k' if parameter == 'k_cat' else 'K'}$_{{{'cat' if parameter == 'k_cat' else 'M'}}}$"
            "\nfraction above one decade"
        )
    # Panel letters at each panel's OUTER top-left, above the axes and flush with the
    # y-label, so nothing is drawn inside the data area.
    for index, axis in enumerate(axes.flat):
        box = axis.get_position()
        figure.text(
            box.x0 - 0.055,
            box.y1 + 0.012,
            "abcdef"[index],
            fontsize=8,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
    stamp = timestamp()
    os.makedirs(IMAGE_DIR, exist_ok=True)
    base = osp.join(IMAGE_DIR, f"kinetic_predictor_spread{suffix}_{stamp}")
    figure.savefig(f"{base}.png", dpi=300)
    savefig_true_size_svg(figure, f"{base}.svg")
    plt.close(figure)
    print(f"-> {base}.svg")


if __name__ == "__main__":
    main()
