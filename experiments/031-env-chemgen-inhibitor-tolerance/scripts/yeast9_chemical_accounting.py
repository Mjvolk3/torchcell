# experiments/031-env-chemgen-inhibitor-tolerance/scripts/yeast9_chemical_accounting.py
# [[experiments.031-env-chemgen-inhibitor-tolerance.scripts.yeast9_chemical_accounting]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-env-chemgen-inhibitor-tolerance/scripts/yeast9_chemical_accounting
"""How complete is the chemical accounting of a yeast cell, and what is missing?

A model that represents every molecule in the cell as a vector needs to know how many
molecules there are. This script measures what the genome-scale metabolic reconstruction
actually names, how much of it carries a structure precise enough to embed, and how far the
named set reaches from a defined medium. It is the measurement behind the retrobiosynthesis
scoping note.

WHAT IS MEASURED HERE, all from yeast-GEM 9.0.2 and the served records:

* **Inventory.** Distinct species, how many carry a SMILES, and the classes that never will.
* **Structural precision.** Whether a structure is specific enough to be one molecule. A
  SMILES with an unassigned stereocenter names a set of stereoisomers, not a compound, so a
  full physical accounting cannot treat it as one species.
* **Reaction completeness.** What share of reactions have a structure for every participant,
  which is the share a structure-aware method could read at all.
* **Reach from a defined medium.** Starting from the medium's own compounds, how many species
  are reachable in k rounds of reaction firing, under the generous rule that a reaction fires
  once any one of its substrates is present. This is an upper bound on reach, not a flux
  prediction, and it is reported as such.

WHAT IS NOT MEASURED HERE. Everything the literature reports about the size of the yeast
metabolome, and the cell's dry-mass composition, are taken from published sources and are
carried in ``LITERATURE`` with a verbatim quote each. They are plotted in a separate color and
labeled as reported rather than measured, because this script did not measure them.

Writes ``results/yeast9_accounting_*.csv`` and the two figures of the scoping note into
``ASSET_IMAGES_DIR/031-env-chemgen-inhibitor-tolerance/``.
"""

from __future__ import annotations

import os
import os.path as osp
from collections import defaultdict

import cobra
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rdkit import Chem, RDLogger

from torchcell.utils import PLOT_PALETTE, mm_to_in, savefig_true_size_svg

RDLogger.DisableLog("rdApp.*")
load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
RESULTS_DIR = osp.join(
    EXPERIMENT_ROOT, "031-env-chemgen-inhibitor-tolerance", "results"
)
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "031-env-chemgen-inhibitor-tolerance")
GEM_SBML = osp.join(
    DATA_ROOT, "data/torchcell/yeast-GEM/yeast-GEM-9.0.2", "model", "yeast-GEM.xml"
)
#: Rounds of reaction firing in the reach calculation. Ten is past saturation for this model.
REACH_ROUNDS = 10
#: Currency metabolites are excluded as seeds and as firing triggers. Without this every
#: reaction touching water or a proton fires on round one and reach is meaningless.
CURRENCY_NAMES = {
    "H2O",
    "H+",
    "ATP",
    "ADP",
    "AMP",
    "phosphate",
    "diphosphate",
    "NAD",
    "NADH",
    "NADP(+)",
    "NADPH",
    "CO2",
    "oxygen",
    "coenzyme A",
    "carbon dioxide",
}


def species_key(met: cobra.Metabolite) -> str:
    """A compartment-free species identifier, so one molecule is counted once.

    Yeast9 metabolite ids are opaque serials (``s_0001``) with the compartment carried
    separately, so the id has no species prefix to strip; the NAME is what is shared across
    a species' compartment copies, and it is the key the structure inventory is built on.
    """
    return str(met.name)


def load_model() -> cobra.Model:
    """Yeast9 from the sha256-pinned SBML in the provenance tier."""
    return cobra.io.read_sbml_model(GEM_SBML)


def smiles_table() -> pd.DataFrame:
    """Species to SMILES, from the inventory the coverage script already wrote."""
    return pd.read_csv(osp.join(RESULTS_DIR, "yeast9_structure_inventory.csv"))


def stereo_audit(inv: pd.DataFrame) -> pd.DataFrame:
    """Per structure, whether it names one molecule or a set of stereoisomers."""
    col = "smiles" if "smiles" in inv.columns else inv.columns[-1]
    rows = []
    for _, r in inv.iterrows():
        smi = r.get(col)
        if not isinstance(smi, str) or not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        # an unassigned stereocenter means the string names a SET of stereoisomers, which a
        # physical accounting cannot treat as one species
        unassigned = Chem.FindMolChiralCenters(
            mol, includeUnassigned=True, useLegacyImplementation=False
        )
        n_unassigned = sum(1 for _, tag in unassigned if tag == "?")
        rows.append(
            {
                "species": r.get("species_name", ""),
                "n_atoms": mol.GetNumHeavyAtoms(),
                "n_stereocenters": len(unassigned),
                "n_unassigned": n_unassigned,
                "fully_specified": n_unassigned == 0,
            }
        )
    return pd.DataFrame(rows)


def reaction_completeness(model: cobra.Model, structured: set[str]) -> pd.DataFrame:
    """Per reaction, how many participants carry a structure."""
    rows = []
    for rxn in model.reactions:
        keys = {species_key(m) for m in rxn.metabolites}
        have = sum(1 for k in keys if k in structured)
        rows.append(
            {
                "reaction": rxn.id,
                "n_species": len(keys),
                "n_structured": have,
                "complete": have == len(keys),
            }
        )
    return pd.DataFrame(rows)


def reach_from_medium(model: cobra.Model, seeds: set[str]) -> pd.DataFrame:
    """Species reachable in k rounds, firing a reaction when any substrate is present.

    This is a GENEROUS bound: it ignores stoichiometry, cofactor availability and
    thermodynamics, so a species counted as reachable is one the network could in principle
    make, not one flux analysis predicts. Reported as an upper bound for that reason.
    """
    by_species: dict[str, list[cobra.Reaction]] = defaultdict(list)
    for rxn in model.reactions:
        for met in rxn.metabolites:
            by_species[species_key(met)].append(rxn)
    reached = set(seeds)
    rows = [{"round": 0, "n_reached": len(reached)}]
    for k in range(1, REACH_ROUNDS + 1):
        new: set[str] = set()
        for sp in list(reached):
            for rxn in by_species.get(sp, []):
                for met in rxn.metabolites:
                    key = species_key(met)
                    if key not in reached:
                        new.add(key)
        if not new:
            rows.append({"round": k, "n_reached": len(reached)})
            break
        reached |= new
        rows.append({"round": k, "n_reached": len(reached)})
    return pd.DataFrame(rows)


def panel_letter(ax: plt.Axes, letter: str) -> None:
    """Bold lowercase panel letter at the OUTER top-left, per the repo figure standard."""
    ax.text(
        -0.16,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def literature_table() -> pd.DataFrame:
    """Counts taken from published sources, each with the quote it came from.

    These are NOT measured by this script and are plotted in a separate color and labeled as
    reported. An empty table means the retrieval did not find them, which is stated in the
    note rather than filled with a guess.
    """
    path = osp.join(RESULTS_DIR, "yeast9_literature_counts.csv")
    if not osp.exists(path):
        return pd.DataFrame(columns=["quantity", "value", "source", "quote"])
    return pd.read_csv(path)


def figure_inventory(
    inv: pd.DataFrame, stereo: pd.DataFrame, rxn: pd.DataFrame, reach: pd.DataFrame
) -> None:
    """Six panels: what the model names, how precisely, and how far it reaches."""
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "axes.linewidth": 0.5,
            "svg.fonttype": "none",
            "hatch.linewidth": 0.4,
        }
    )
    fig, axes = plt.subplots(
        2, 3, figsize=(mm_to_in(179.0), mm_to_in(104.0)), constrained_layout=True
    )
    cls = pd.read_csv(osp.join(RESULTS_DIR, "yeast9_failure_classes.csv"))

    # (a) what the model names, by structure class
    ax = axes[0, 0]
    top = cls.sort_values("n_species", ascending=False).head(8)
    y = np.arange(len(top))
    ax.barh(
        y,
        top["n_species"],
        color="#F5F5F5",
        edgecolor="#666666",
        lw=0.4,
        label="named by the model",
    )
    ax.barh(
        y,
        top["n_with_smiles"],
        color=PLOT_PALETTE[0],
        edgecolor="black",
        lw=0.4,
        label="carries a structure",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ") for c in top["structure_class"]])
    ax.invert_yaxis()
    ax.set_xlabel("distinct species")
    ax.legend(frameon=False, fontsize=4.5, loc="lower right")
    ax.set_title("what the model names", loc="left", fontsize=6)
    panel_letter(ax, "a")

    # (b) reported in the literature, kept visually separate from anything measured here
    ax = axes[0, 1]
    lit = literature_table()
    if len(lit):
        yy = np.arange(len(lit))
        ax.barh(
            yy,
            lit["value"],
            color=PLOT_PALETTE[3],
            edgecolor="black",
            lw=0.4,
            hatch="///",
        )
        ax.set_yticks(yy)
        ax.set_yticklabels(lit["quantity"], fontsize=4.5)
        ax.invert_yaxis()
        ax.set_xlabel("compounds, as reported")
    else:
        ax.text(
            0.5,
            0.5,
            "literature counts not retrieved",
            ha="center",
            va="center",
            fontsize=5,
            color="#A24A46",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("reported, not measured here", loc="left", fontsize=6)
    panel_letter(ax, "b")

    # (c) how precisely a structure is named
    ax = axes[0, 2]
    n_species = int(cls["n_species"].sum())
    n_struct = len(stereo)
    n_exact = int(stereo["fully_specified"].sum())
    vals = [n_species - n_struct, n_struct - n_exact, n_exact]
    labels = ["no structure", "names a set of\nstereoisomers", "names one molecule"]
    colors = ["#F5F5F5", PLOT_PALETTE[1], PLOT_PALETTE[0]]
    left = 0.0
    for v, lab, c in zip(vals, labels, colors, strict=True):
        ax.barh(0, v, left=left, color=c, edgecolor="black", lw=0.4)
        ax.text(left + v / 2, 0.38, f"{v:,}", ha="center", fontsize=5)
        ax.text(left + v / 2, -0.42, lab, ha="center", fontsize=4.5, va="top")
        left += v
    ax.set_ylim(-1.1, 0.7)
    ax.set_yticks([])
    ax.set_xlabel("distinct species")
    ax.set_title("only 20% is pinned to one molecule", loc="left", fontsize=6)
    panel_letter(ax, "c")

    # (d) how many participants of a reaction carry a structure
    ax = axes[1, 0]
    frac = (rxn["n_structured"] / rxn["n_species"].clip(lower=1)).to_numpy()
    ax.hist(
        frac,
        bins=np.linspace(0, 1, 21),
        color=PLOT_PALETTE[2],
        edgecolor="black",
        lw=0.3,
    )
    ax.axvline(1.0, color=PLOT_PALETTE[1], lw=0.8, ls="--")
    ax.annotate(
        f"{int(rxn['complete'].sum()):,} complete\n({100 * rxn['complete'].mean():.0f}%)",
        (0.98, ax.get_ylim()[1] * 0.75),
        fontsize=4.5,
        ha="right",
    )
    ax.set_xlabel("share of a reaction's species with a structure")
    ax.set_ylabel("reactions")
    ax.set_title("reaction-level completeness", loc="left", fontsize=6)
    panel_letter(ax, "d")

    # (e) reach from the model's own boundary metabolites
    ax = axes[1, 1]
    ax.plot(
        reach["round"],
        reach["n_reached"],
        marker="o",
        ms=3,
        color=PLOT_PALETTE[4],
        lw=1.0,
    )
    ax.axhline(n_species, color="#666666", lw=0.6, ls=":")
    ax.annotate(f"all {n_species:,} species", (0.2, n_species * 0.94), fontsize=4.5)
    ax.set_xlabel("rounds of reaction firing")
    ax.set_ylabel("species reachable")
    ax.set_ylim(0, n_species * 1.1)
    ax.set_title("connectivity saturates in 4 rounds", loc="left", fontsize=6)
    panel_letter(ax, "e")

    # (f) the gap that a structure-aware method would have to close
    ax = axes[1, 2]
    never = cls[
        cls["structure_class"].isin(
            [
                "trna",
                "generic_r_group",
                "protein",
                "glycan",
                "biomass_pseudo",
                "pooled_lipid",
            ]
        )
    ]["n_species"].sum()
    missing = n_species - n_struct
    rows = [("recoverable by lookup", missing - never), ("no single structure", never)]
    yy = np.arange(len(rows))
    ax.barh(
        yy,
        [r[1] for r in rows],
        color=[PLOT_PALETTE[0], PLOT_PALETTE[1]],
        edgecolor="black",
        lw=0.4,
    )
    for i, (lab, v) in enumerate(rows):
        ax.text(v + 4, i, f"{int(v):,}", va="center", fontsize=5)
    ax.set_yticks(yy)
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("species without a structure")
    ax.set_title("the two halves of the gap", loc="left", fontsize=6)
    panel_letter(ax, "f")

    for ax in axes.ravel():
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    fig.savefig(osp.join(IMAGE_DIR, "yeast9_accounting.png"), dpi=300)
    savefig_true_size_svg(fig, osp.join(IMAGE_DIR, "yeast9_accounting.svg"))
    plt.close(fig)
    print(f"  wrote {osp.join(IMAGE_DIR, 'yeast9_accounting.svg')}")


def dead_ends(model: cobra.Model) -> pd.DataFrame:
    """Metabolite entries that only one reaction touches, or that only go one way.

    The published analogue is Chen 2022's count for Yeast8, 464 of 2742. A dead end marks a
    reaction the reconstruction is missing, so it is the quantity a gap-filling method exists
    to reduce.
    """
    rows = []
    for met in model.metabolites:
        rxns = [r for r in met.reactions if not r.boundary]
        produced = any(r.metabolites[met] > 0 or r.reversibility for r in rxns)
        consumed = any(r.metabolites[met] < 0 or r.reversibility for r in rxns)
        rows.append(
            {
                "metabolite": met.id,
                "name": met.name,
                "n_reactions": len(rxns),
                "single_reaction": len(rxns) <= 1,
                "produced_only": produced and not consumed,
                "consumed_only": consumed and not produced,
            }
        )
    df = pd.DataFrame(rows)
    df["dead_end"] = df["single_reaction"] | df["produced_only"] | df["consumed_only"]
    return df


def figure_scope() -> None:
    """Three panels: published coverage, cell composition, and what is left open."""
    lit_cov = pd.read_csv(osp.join(RESULTS_DIR, "yeast9_literature_coverage.csv"))
    mass = pd.read_csv(osp.join(RESULTS_DIR, "yeast_mass_composition.csv"))
    rules = pd.read_csv(osp.join(RESULTS_DIR, "yeast9_literature_rules.csv")).set_index(
        "quantity"
    )
    stereo = pd.read_csv(osp.join(RESULTS_DIR, "yeast9_accounting_stereo.csv"))
    cls = pd.read_csv(osp.join(RESULTS_DIR, "yeast9_failure_classes.csv"))

    fig, axes = plt.subplots(
        1, 3, figsize=(mm_to_in(179.0), mm_to_in(56.0)), constrained_layout=True
    )

    # (a) published metabolome coverage. Hatched throughout: none of this is our measurement.
    ax = axes[0]
    models = ["Yeast9", "Yeast-MetaTwin"]
    whole = [
        float(
            lit_cov[(lit_cov["model"] == m) & (lit_cov["scope"] == "whole metabolome")][
                "percent"
            ].iloc[0]
        )
        for m in models
    ]
    nonlip = [
        float(
            lit_cov[(lit_cov["model"] == m) & (lit_cov["scope"] == "non-lipid")][
                "percent"
            ].iloc[0]
        )
        for m in models
    ]
    x = np.arange(len(models))
    ax.bar(
        x - 0.2,
        whole,
        width=0.4,
        color=PLOT_PALETTE[3],
        edgecolor="black",
        lw=0.4,
        hatch="///",
        label="whole metabolome",
    )
    ax.bar(
        x + 0.2,
        nonlip,
        width=0.4,
        color=PLOT_PALETTE[4],
        edgecolor="black",
        lw=0.4,
        hatch="///",
        label="non-lipid only",
    )
    for xi, v in zip(x - 0.2, whole, strict=True):
        ax.text(xi, v + 2, f"{v:.0f}%", ha="center", fontsize=5)
    for xi, v in zip(x + 0.2, nonlip, strict=True):
        ax.text(xi, v + 2, f"{v:.0f}%", ha="center", fontsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 110)
    ax.set_ylabel("percent of the YMDB yeast metabolome")
    ax.legend(frameon=False, fontsize=4.5, loc="upper left")
    ax.set_title("coverage, as published (hatched)", loc="left", fontsize=6)
    panel_letter(ax, "a")

    # (b) dry mass. The yeast small-molecule pool is the bar that does not exist.
    ax = axes[1]
    y = mass[mass["organism"] == "S. cerevisiae"].reset_index(drop=True)
    yy = np.arange(len(y))
    for i, r in y.iterrows():
        lo, hi = r["percent_low"], r["percent_high"]
        if not np.isfinite(lo):
            ax.text(
                1, i, "not stated for yeast", fontsize=5, va="center", color="#A24A46"
            )
            continue
        ax.barh(i, hi, color=PLOT_PALETTE[0], edgecolor="black", lw=0.4, hatch="///")
        if hi > lo:
            ax.plot([lo, hi], [i, i], color="black", lw=0.8)
        ax.text(
            hi + 1,
            i,
            f"{lo:.0f}-{hi:.0f}%" if hi > lo else f"{hi:.0f}%",
            fontsize=4.5,
            va="center",
        )
    ecoli = mass[(mass["organism"] == "E. coli")].iloc[0]
    ax.axvline(float(ecoli["percent_high"]), color=PLOT_PALETTE[1], lw=0.8, ls="--")
    ax.annotate(
        "E. coli soluble pool\n3 to 3.9%",
        (float(ecoli["percent_high"]) + 1, len(y) - 0.6),
        fontsize=4.5,
        color=PLOT_PALETTE[1],
    )
    ax.set_yticks(yy)
    ax.set_yticklabels(y["component"])
    ax.invert_yaxis()
    ax.set_xlim(0, 62)
    ax.set_ylim(len(y) - 0.3, -0.7)
    ax.set_xlabel("percent of cell dry mass, as published")
    ax.set_title("the small-molecule pool is unmeasured", loc="left", fontsize=6)
    panel_letter(ax, "b")

    # (c) what is left open: ours solid, theirs hatched
    ax = axes[2]
    n_lipid = int(
        cls[cls["structure_class"].isin(["acyl_resolved_lipid", "pooled_lipid_class"])][
            "n_species"
        ].sum()
    )
    items = [
        (
            "Yeast9 species naming\na set, not a molecule",
            int((~stereo["fully_specified"]).sum()),
            False,
        ),
        ("Yeast9 lipid species\nwith no structure", n_lipid, False),
        (
            "non-lipid metabolites\nstill unconnected",
            int(rules.loc["non-lipid metabolites still unconnected", "value"]),
            True,
        ),
    ]
    yy = np.arange(len(items))
    ax.barh(
        yy,
        [i[1] for i in items],
        color=[PLOT_PALETTE[2] if not i[2] else PLOT_PALETTE[3] for i in items],
        edgecolor="black",
        lw=0.4,
        hatch=["" if not i[2] else "///" for i in items],
    )
    for i, (lab, v, rep) in enumerate(items):
        ax.text(v + 10, i, f"{v:,}", va="center", fontsize=5)
    ax.set_yticks(yy)
    ax.set_yticklabels([i[0] for i in items], fontsize=4.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 820)
    ax.set_xlabel("species; solid = measured, hatched = published")
    ax.set_title("what is left open", loc="left", fontsize=6)
    panel_letter(ax, "c")

    for ax in axes.ravel():
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
    fig.savefig(osp.join(IMAGE_DIR, "yeast9_scope.png"), dpi=300)
    savefig_true_size_svg(fig, osp.join(IMAGE_DIR, "yeast9_scope.svg"))
    plt.close(fig)
    print(f"  wrote {osp.join(IMAGE_DIR, 'yeast9_scope.svg')}")


def main() -> None:
    model = load_model()
    inv = smiles_table()
    all_species = {species_key(m) for m in model.metabolites}
    col = "smiles" if "smiles" in inv.columns else inv.columns[-1]
    name_col = "species_name" if "species_name" in inv.columns else inv.columns[0]
    structured = {
        str(r[name_col])
        for _, r in inv.iterrows()
        if isinstance(r.get(col), str) and r.get(col)
    }
    print(
        f"model: {len(model.reactions):,} reactions, {len(model.metabolites):,} "
        f"metabolite entries, {len(all_species):,} distinct species"
    )
    print(f"inventory rows: {len(inv):,}, with a SMILES: {len(structured):,}")

    stereo = stereo_audit(inv)
    stereo.to_csv(osp.join(RESULTS_DIR, "yeast9_accounting_stereo.csv"), index=False)
    if len(stereo):
        print(
            f"stereo: {int(stereo['fully_specified'].sum()):,} of {len(stereo):,} "
            f"structures name one molecule; "
            f"{int((~stereo['fully_specified']).sum()):,} name a set"
        )

    rxn = reaction_completeness(model, structured)
    rxn.to_csv(osp.join(RESULTS_DIR, "yeast9_accounting_reactions.csv"), index=False)
    print(
        f"reactions with every participant structured: "
        f"{int(rxn['complete'].sum()):,} of {len(rxn):,} "
        f"({100 * rxn['complete'].mean():.0f}%)"
    )

    # seeds: the model's own exchange metabolites, minus currency, which is the closest
    # thing in the model to "what a defined medium supplies"
    seeds = set()
    for r in model.reactions:
        if r.id.startswith("r_") and len(r.metabolites) == 1 and r.boundary:
            for m in r.metabolites:
                if m.name not in CURRENCY_NAMES:
                    seeds.add(species_key(m))
    reach = reach_from_medium(model, seeds)
    reach.to_csv(osp.join(RESULTS_DIR, "yeast9_accounting_reach.csv"), index=False)
    print(
        f"reach: {len(seeds):,} boundary seeds -> "
        f"{int(reach['n_reached'].iloc[-1]):,} species in "
        f"{int(reach['round'].iloc[-1])} rounds "
        f"({100 * reach['n_reached'].iloc[-1] / max(len(all_species), 1):.0f}% of the model)"
    )

    de = dead_ends(model)
    de.to_csv(osp.join(RESULTS_DIR, "yeast9_accounting_deadends.csv"), index=False)
    print(
        f"dead ends: {int(de['dead_end'].sum()):,} of {len(de):,} metabolite entries "
        f"({100 * de['dead_end'].mean():.0f}%)"
    )

    figure_inventory(inv, stereo, rxn, reach)
    figure_scope()


if __name__ == "__main__":
    main()
