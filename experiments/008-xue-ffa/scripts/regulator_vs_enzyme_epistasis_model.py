# experiments/008-xue-ffa/scripts/regulator_vs_enzyme_epistasis_model.py
# [[experiments.008-xue-ffa.scripts.regulator_vs_enzyme_epistasis_model]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/regulator_vs_enzyme_epistasis_model
#
# DOES A REGULATOR PERTURBATION MANUFACTURE MORE APPARENT EPISTASIS AT ONE METABOLIC
# READOUT THAN AN ENZYME PERTURBATION? A model-only test.
#
# WHAT THIS IS, AND WHAT IT IS NOT. Everything this script produces is a property of the
# model it defines: a Yeast9 flux-balance model, a capacity-scaling perturbation rule, and
# a single scalar readout. It is NOT evidence about the measured strains. Nothing here
# was measured in a cell, no number here is a titer, and no statement here supports or
# refutes any claim about the ten deleted transcription factors of the source study. The
# only thing under test is whether a perturbation model that contains NO interaction term
# of any kind can still produce pervasive interaction at a single measured locus, and
# whether it does so more when the perturbation acts on a regulator than on an enzyme.
#
# THE HYPOTHESIS UNDER TEST (in the model, not in the biology). A perturbation that acts
# on a REGULATOR produces more, and larger, apparent interaction at a single metabolic
# readout than a perturbation that acts directly on a pathway ENZYME, because the
# regulator sits more reaction steps away from the readout and because it modulates many
# reactions at once, most of which are not measured.
#
# THE MODEL, in five pieces.
#   1. Network. Yeast9 (yeast-GEM 9.0.2), read from the pinned SBML, hashed into the
#      summary so the run is traceable to an exact file.
#   2. Base strain. The chassis of the source study: POX1, FAA1 and FAA4 knocked out.
#      Every perturbation below is applied on top of that chassis.
#   3. Readout. The maximum of the summed export flux of the five measured species
#      (palmitate, palmitoleate, stearate, oleate, myristate) with growth held at
#      GROWTH_FRACTION of the base strain's maximum growth. The base strain's value is
#      the denominator, so the base strain reads exactly 1, as in the document's
#      normalization. A strain that cannot reach the growth floor exports nothing and
#      reads 0; that is the definition, not a fallback.
#   4. Two arms of ten genes. REGULATOR: the ten deleted transcription factors, none of
#      which is a Yeast9 gene, so each acts only through its Yeast9 targets taken from the
#      SGD regulatory graph and TFLink, factor -> target direction only. ENZYME: ten of
#      the thirteen core fatty acid pathway genes, each acting on the reactions it
#      catalyzes itself.
#   5. Perturbation. LINEAR: deleting a gene multiplies both bounds of every reaction in
#      its reaction set by alpha, preserving sign. BINARY: it sets both bounds to zero.
#
# COMPOSITION, and why it is the composition-free rule. On a reaction hit by more than one
# deleted gene the factors MULTIPLY in the linear arm and the MINIMUM is taken in the
# binary arm (on {0, 1} the two rules agree). Either way a combination's factor on a
# reaction is a fixed function of the single-gene factors with no combination-specific
# term, so the PERTURBATION model contains no interaction parameter at all. Any eps or tau
# the readout shows is therefore produced by the network and by the choice to measure one
# locus.
#
# INTERACTION SCORES, exactly as the document defines them:
#   eps_ij  = f_ij  - f_i f_j
#   tau_ijk = f_ijk - f_ij f_k - f_ik f_j - f_jk f_i + 2 f_i f_j f_k
#
# HOPS. The distance half of the hypothesis. On the undirected reaction-metabolite
# bipartite graph of Yeast9 with currency metabolites dropped, a gene's hop count is the
# shortest path in reactions from any reaction in its reaction set to any of the five
# readout exchanges. A regulator's reaction set is the union over its Yeast9 targets.
#
# OUTPUT. singles.csv, pairs.csv, triples.csv, genes.csv and summary.json under
# results/regulator_vs_enzyme/, plus one four-panel figure.

import hashlib
import itertools
import json
import os
import os.path as osp
from collections import Counter

import cobra
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from cobra.manipulation import knock_out_model_genes
from dotenv import load_dotenv
from scipy.stats import spearmanr

from torchcell.graph.graph import SCerevisiaeGraph
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    apply_paper_style,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

MODEL_PATH = osp.join(
    DATA_ROOT, "data/torchcell/yeast-GEM/yeast-GEM-9.0.2/model/yeast-GEM.xml"
)
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "008-xue-ffa/results/regulator_vs_enzyme")
IMAGES_DIR = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
FIG_STEM = "regulator_vs_enzyme_epistasis"

# The five measured species leave the model through these exchanges.
READOUT_EXCHANGES = {
    "r_1993": "palmitate",
    "r_1994": "palmitoleate",
    "r_2055": "stearate",
    "r_2189": "oleate",
    "r_2193": "myristate",
}
BIOMASS = "r_2111"
# The chassis of the source study, applied to the base strain before anything else.
CHASSIS = {"POX1": "YGL205W", "FAA1": "YOR317W", "FAA4": "YMR246W"}
GROWTH_FRACTION = 0.5

# The ten deleted transcription factors of the source study.
REGULATOR_ARM = {
    "FKH1": "YIL131C",
    "GCN5": "YGR252W",
    "MED4": "YOR174W",
    "OPI1": "YHL020C",
    "RFX1": "YLR176C",
    "RGR1": "YLR071C",
    "RPD3": "YNL330C",
    "SPT3": "YDR392W",
    "TFC7": "YOR110W",
    "YAP6": "YDR259C",
}
# The thirteen core fatty acid pathway genes, in GENE_ORDER of ffa_network_overlay_panel.
PATHWAY_GENES = [
    "ACC1",
    "FAS1",
    "FAS2",
    "ELO1",
    "ELO2",
    "ELO3",
    "OLE1",
    "FAA1",
    "FAA2",
    "FAA3",
    "FAA4",
    "POX1",
    "SLC1",
]
N_ENZYME_ARM = 10

# Capacity multipliers for the linear arm. The three the question asks for come first;
# the rest extend the sweep downward because Yeast9 leaves about 70% of its bounds at the
# +-1000 placeholder, where scaling by 0.75 to 0.25 changes nothing that binds. Every
# setting tried is reported, and none of them was chosen after looking at the answer.
ALPHAS = (0.75, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.002)
# A score is called nonzero above this. The readout is an LP optimal value, so exact
# ties are the rule rather than the exception and the tolerance is a numerical one.
TOLERANCE = 1e-6

# Currency metabolites, dropped before the bipartite graph is built, so a path may not
# run through a proton or a phosphate. Names are Yeast9 metabolite names.
CURRENCY = {
    "H+",
    "H2O",
    "ATP",
    "ADP",
    "AMP",
    "phosphate",
    "diphosphate",
    "NAD",
    "NADH",
    "NADP(+)",
    "NADPH",
    "carbon dioxide",
    "oxygen",
    "coenzyme A",
}

C_REGULATOR = PLOT_PALETTE[0]
C_ENZYME = PLOT_PALETTE[1]
ARM_COLOR = {"regulator": C_REGULATOR, "enzyme": C_ENZYME}

CAVEATS = [
    "Everything reported here is a property of this model. It is not evidence about "
    "the measured strains and no number here is a titer.",
    "FBA maximizes a readout under a growth floor. The optimum is what the network "
    "COULD export, not what a strain does export, and the map from capacity to titer "
    "is not modeled.",
    "Capacity scaling is not transcription. Multiplying a reaction's bounds by alpha "
    "is a stand-in for lowered expression that ignores enzyme kinetics, protein cost, "
    "isozyme redundancy and every regulatory response to the deletion.",
    "About 70% of Yeast9 bounds sit at the +-1000 placeholder, which is not a measured "
    "capacity. Scaling a placeholder is scaling nothing until alpha is small enough to "
    "bring it below the flux the optimum wants.",
    "The regulatory graphs are incomplete. SGD regulatory and TFLink disagree on which "
    "targets a factor has, neither is a complete target list, and a factor with no "
    "Yeast9 target would act on nothing here for a reason about the graph rather than "
    "about the cell.",
    "A regulator's targets are treated as all-or-nothing and equal. Activation versus "
    "repression, which the edges carry, is ignored, so a deletion always lowers "
    "capacity even where the factor represses.",
    "An LP optimal value is piecewise linear in the bounds, so exact zeros in eps and "
    "tau are expected and are not the same statement as a small measured interaction.",
    "Three enzyme-arm genes, POX1, FAA1 and FAA4, are the chassis knockouts. Their "
    "own-reaction capacity is already zero or shared with an isozyme in the base "
    "strain, so their perturbation is weaker than the arm's other members by "
    "construction.",
    "Hops are counted on a structural graph. A reaction with zero capacity in the "
    "chassis still carries a path, so hops measure network topology and not flux.",
]


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_base_model() -> tuple[cobra.Model, float, float]:
    """Read Yeast9, knock out the chassis, and set up the readout LP.

    Returns the model with the growth floor and the readout objective already in place,
    the base strain's maximum growth, and the base strain's readout value.
    """
    print(f"reading {MODEL_PATH}")
    model = cobra.io.read_sbml_model(MODEL_PATH)
    print(
        f"  Yeast9: {len(model.reactions)} reactions, "
        f"{len(model.metabolites)} metabolites, {len(model.genes)} genes"
    )
    deactivated = knock_out_model_genes(model, sorted(CHASSIS.values()))
    print(
        f"  chassis {sorted(CHASSIS)} knocked out, "
        f"{len(deactivated)} reactions deactivated"
    )
    base_growth = model.slim_optimize()
    if np.isnan(base_growth):
        raise ValueError("the chassis strain has no feasible growth solution")
    model.reactions.get_by_id(BIOMASS).lower_bound = GROWTH_FRACTION * base_growth
    model.objective = {
        model.reactions.get_by_id(r): 1.0 for r in sorted(READOUT_EXCHANGES)
    }
    base_readout = model.slim_optimize()
    if np.isnan(base_readout) or base_readout <= 0.0:
        raise ValueError(f"the chassis strain has no positive readout: {base_readout}")
    print(
        f"  base growth {base_growth:.6f} /h, growth floor "
        f"{GROWTH_FRACTION} x that, base readout {base_readout:.6f} mmol/gDW/h"
    )
    return model, base_growth, base_readout


def regulator_targets(
    graphs: tuple[nx.DiGraph, ...], orf: str, model_genes: set[str]
) -> tuple[set[str], set[str]]:
    """Targets of a factor, and the subset that are Yeast9 genes.

    Both graphs are directed, so only edges leaving the factor are taken.
    """
    targets: set[str] = set()
    for g in graphs:
        if not g.is_directed():
            raise ValueError("a regulatory graph is undirected; direction is required")
        if orf in g:
            targets |= set(g.successors(orf))
    return targets, targets & model_genes


def reactions_of_genes(model: cobra.Model, orfs: set[str]) -> set[str]:
    rxns: set[str] = set()
    for orf in orfs:
        rxns |= {r.id for r in model.genes.get_by_id(orf).reactions}
    return rxns


def bipartite_hops(model: cobra.Model) -> dict[str, int]:
    """Shortest path in REACTIONS from every reaction to the nearest readout exchange.

    The graph is reaction-metabolite bipartite with currency metabolites dropped, so one
    reaction step is two edges. A reaction that cannot reach a readout exchange is absent
    from the returned mapping.
    """
    g = nx.Graph()
    dropped = 0
    for rxn in model.reactions:
        rnode = ("R", rxn.id)
        g.add_node(rnode)
        for met in rxn.metabolites:
            if met.name in CURRENCY:
                dropped += 1
                continue
            g.add_edge(rnode, ("M", met.id))
    print(
        f"  bipartite graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges, "
        f"{dropped} reaction-metabolite incidences dropped as currency"
    )
    print(f"  currency metabolites: {sorted(CURRENCY)}")
    source = ("SOURCE",)
    for rid in READOUT_EXCHANGES:
        g.add_edge(source, ("R", rid))
    dist = nx.single_source_shortest_path_length(g, source)
    # One edge from the virtual source, then two edges per reaction step.
    return {
        node[1]: (d - 1) // 2 for node, d in dist.items() if node[0] == "R"
    }


def readout_ratio(
    model: cobra.Model, factors: dict[str, float], base_readout: float
) -> float:
    """Readout of a perturbed strain, as a ratio to the base strain."""
    with model:
        for rid, factor in factors.items():
            rxn = model.reactions.get_by_id(rid)
            lb, ub = rxn.bounds
            rxn.bounds = (lb * factor, ub * factor)
        value = model.slim_optimize()
    if np.isnan(value):
        # The strain cannot reach the growth floor, so it exports nothing.
        return 0.0
    return value / base_readout


def combine(
    rxn_sets: list[set[str]], alpha: float, modulation: str
) -> dict[str, float]:
    """Per-reaction factor for a combination of deletions.

    Linear: the single-gene factors multiply. Binary: the minimum is taken. Neither rule
    carries a combination-specific term, so the perturbation model has no interaction.
    """
    factors: dict[str, float] = {}
    for rxns in rxn_sets:
        for rid in rxns:
            if modulation == "linear":
                factors[rid] = factors.get(rid, 1.0) * alpha
            elif modulation == "binary":
                factors[rid] = min(factors.get(rid, 1.0), 0.0)
            else:
                raise ValueError(f"unknown modulation {modulation}")
    return factors


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    model, base_growth, base_readout = build_base_model()
    model_genes = {g.id for g in model.genes}
    # No gene maps to the biomass reaction or to an exchange in Yeast9, so no perturbation
    # can touch the growth floor or the readout itself. Check rather than assume.
    for rid in [BIOMASS, *READOUT_EXCHANGES]:
        rule = model.reactions.get_by_id(rid).gene_reaction_rule
        if rule:
            raise ValueError(f"{rid} carries a gene rule ({rule}); the LP would be cyclic")

    print("\nloading the genome and the regulatory graphs")
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    graph_store = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )
    reg_graphs = (graph_store.G_regulatory.graph, graph_store.G_tflink.graph)
    print(
        f"  SGD regulatory: {reg_graphs[0].number_of_edges()} edges, "
        f"TFLink: {reg_graphs[1].number_of_edges()} edges"
    )

    print("\nhop counts")
    hops_of_reaction = bipartite_hops(model)

    # --- arm definitions -------------------------------------------------------------
    print("\nREGULATOR arm, ten deleted transcription factors")
    gene_rows = []
    rxn_set_of: dict[tuple[str, str], set[str]] = {}
    for name in sorted(REGULATOR_ARM):
        orf = REGULATOR_ARM[name]
        resolved = genome.resolve_gene_name(name).systematic_name
        if resolved != orf:
            raise ValueError(f"{name} resolves to {resolved}, not the stated {orf}")
        if orf in model_genes:
            raise ValueError(f"{name} is a Yeast9 gene; the regulator arm assumes it is not")
        targets, y9_targets = regulator_targets(reg_graphs, orf, model_genes)
        rxns = reactions_of_genes(model, y9_targets)
        rxn_set_of[("regulator", name)] = rxns
        reachable = [hops_of_reaction[r] for r in rxns if r in hops_of_reaction]
        hops = min(reachable) if reachable else None
        n_live = sum(
            1 for r in rxns if model.reactions.get_by_id(r).bounds != (0.0, 0.0)
        )
        gene_rows.append(
            {
                "arm": "regulator",
                "gene": name,
                "orf": orf,
                "n_targets": len(targets),
                "n_yeast9_targets": len(y9_targets),
                "n_reactions": len(rxns),
                "n_live_reactions": n_live,
                "hops": hops,
            }
        )
        print(
            f"  {name:5s} {orf}  targets {len(targets):5d}  Yeast9 targets "
            f"{len(y9_targets):4d}  reactions {len(rxns):5d}  live {n_live:5d}  hops {hops}"
        )

    print(
        "\nENZYME arm: the ten of the thirteen core pathway genes with the most Yeast9\n"
        "reactions, ties broken by pathway order in GENE_ORDER"
    )
    counted = []
    for idx, name in enumerate(PATHWAY_GENES):
        orf = genome.resolve_gene_name(name).systematic_name
        if orf not in model_genes:
            raise ValueError(f"pathway gene {name} ({orf}) is not a Yeast9 gene")
        rxns = reactions_of_genes(model, {orf})
        counted.append((len(rxns), idx, name, orf, rxns))
    counted.sort(key=lambda row: (-row[0], row[1]))
    for rank, (n_rxn, _, name, orf, _rxns) in enumerate(counted):
        mark = "kept" if rank < N_ENZYME_ARM else "dropped"
        print(f"  {name:5s} {orf}  reactions {n_rxn:3d}  {mark}")
    for n_rxn, _, name, orf, rxns in counted[:N_ENZYME_ARM]:
        rxn_set_of[("enzyme", name)] = rxns
        reachable = [hops_of_reaction[r] for r in rxns if r in hops_of_reaction]
        hops = min(reachable) if reachable else None
        n_live = sum(
            1 for r in rxns if model.reactions.get_by_id(r).bounds != (0.0, 0.0)
        )
        gene_rows.append(
            {
                "arm": "enzyme",
                "gene": name,
                "orf": orf,
                "n_targets": 0,
                "n_yeast9_targets": 1,
                "n_reactions": len(rxns),
                "n_live_reactions": n_live,
                "hops": hops,
            }
        )
    genes_df = pd.DataFrame(gene_rows)
    arm_genes = {
        arm: sorted(g for a, g in rxn_set_of if a == arm)
        for arm in ("regulator", "enzyme")
    }
    for arm in ("regulator", "enzyme"):
        sub = genes_df[genes_df["arm"] == arm]
        print(
            f"  {arm} hops: {sorted(sub['hops'].tolist())}, "
            f"median {sub['hops'].median()}"
        )

    for arm, genes in arm_genes.items():
        for gene in genes:
            bad = set(rxn_set_of[(arm, gene)]) & ({BIOMASS} | set(READOUT_EXCHANGES))
            if bad:
                raise ValueError(f"{arm} {gene} would scale {sorted(bad)}")

    # --- the sweep -------------------------------------------------------------------
    settings = [("linear", a) for a in ALPHAS] + [("binary", 0.0)]
    single_rows, pair_rows, triple_rows = [], [], []
    hops_of_gene = {
        (r["arm"], r["gene"]): r["hops"] for r in gene_rows
    }
    print(
        f"\nsolving {len(settings)} settings x 2 arms x (10 singles + 45 pairs + "
        f"120 triples) = {len(settings) * 2 * 175} LPs"
    )
    for modulation, alpha in settings:
        for arm in ("regulator", "enzyme"):
            genes = arm_genes[arm]
            f_single = {}
            for gene in genes:
                f = readout_ratio(
                    model,
                    combine([rxn_set_of[(arm, gene)]], alpha, modulation),
                    base_readout,
                )
                f_single[gene] = f
                single_rows.append(
                    {
                        "arm": arm,
                        "modulation": modulation,
                        "alpha": alpha,
                        "gene": gene,
                        "orf": genes_df.loc[
                            (genes_df["arm"] == arm) & (genes_df["gene"] == gene), "orf"
                        ].item(),
                        "f": f,
                    }
                )
            f_pair = {}
            for gi, gj in itertools.combinations(genes, 2):
                f = readout_ratio(
                    model,
                    combine(
                        [rxn_set_of[(arm, gi)], rxn_set_of[(arm, gj)]], alpha, modulation
                    ),
                    base_readout,
                )
                f_pair[(gi, gj)] = f
                pair_rows.append(
                    {
                        "arm": arm,
                        "modulation": modulation,
                        "alpha": alpha,
                        "gene_i": gi,
                        "gene_j": gj,
                        "f_i": f_single[gi],
                        "f_j": f_single[gj],
                        "f_ij": f,
                        "eps": f - f_single[gi] * f_single[gj],
                    }
                )
            for gi, gj, gk in itertools.combinations(genes, 3):
                f = readout_ratio(
                    model,
                    combine(
                        [rxn_set_of[(arm, g)] for g in (gi, gj, gk)], alpha, modulation
                    ),
                    base_readout,
                )
                fi, fj, fk = f_single[gi], f_single[gj], f_single[gk]
                fij, fik, fjk = f_pair[(gi, gj)], f_pair[(gi, gk)], f_pair[(gj, gk)]
                tau = f - fij * fk - fik * fj - fjk * fi + 2 * fi * fj * fk
                gene_hops = [hops_of_gene[(arm, g)] for g in (gi, gj, gk)]
                triple_rows.append(
                    {
                        "arm": arm,
                        "modulation": modulation,
                        "alpha": alpha,
                        "gene_i": gi,
                        "gene_j": gj,
                        "gene_k": gk,
                        "f_i": fi,
                        "f_j": fj,
                        "f_k": fk,
                        "f_ijk": f,
                        "tau": tau,
                        "mean_hops": float(np.mean(gene_hops)),
                        "max_hops": float(np.max(gene_hops)),
                    }
                )
        print(f"  {modulation:6s} alpha {alpha:<6g} done")

    singles_df = pd.DataFrame(single_rows)
    pairs_df = pd.DataFrame(pair_rows)
    triples_df = pd.DataFrame(triple_rows)

    # --- summary ---------------------------------------------------------------------
    def sign_split(values: np.ndarray) -> dict[str, int]:
        return {
            "positive": int(np.sum(values > TOLERANCE)),
            "negative": int(np.sum(values < -TOLERANCE)),
            "zero": int(np.sum(np.abs(values) <= TOLERANCE)),
        }

    def spearman_or_none(x: np.ndarray, y: np.ndarray) -> dict[str, float] | None:
        if len(x) < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
            return None
        rho, p = spearmanr(x, y)
        return {"rho": float(rho), "p": float(p), "n": int(len(x))}

    per_setting = []
    for modulation, alpha in settings:
        for arm in ("regulator", "enzyme"):
            p = pairs_df[
                (pairs_df["modulation"] == modulation)
                & (pairs_df["alpha"] == alpha)
                & (pairs_df["arm"] == arm)
            ]
            t = triples_df[
                (triples_df["modulation"] == modulation)
                & (triples_df["alpha"] == alpha)
                & (triples_df["arm"] == arm)
            ]
            s = singles_df[
                (singles_df["modulation"] == modulation)
                & (singles_df["alpha"] == alpha)
                & (singles_df["arm"] == arm)
            ]
            eps = p["eps"].to_numpy()
            tau = t["tau"].to_numpy()
            # THE DECISIVE COUNT. A trigenic score is arithmetic on seven readouts, and
            # if any of the seven is a zero the score is arithmetic on a feasibility
            # cliff rather than a measure of how two deletions combine. A triple counts
            # as graded only when all three singles, all three pairs and the triple
            # itself clear the growth floor. Without this an all-or-nothing sweep reports
            # 120 of 120 triples "interacting" at |tau| exactly 1 or exactly 2.
            pair_f = {
                frozenset((r.gene_i, r.gene_j)): r.f_ij for r in p.itertuples()
            }
            single_f = dict(zip(s["gene"], s["f"], strict=True))
            graded = np.array([
                all(single_f[g] > 0.0 for g in (r.gene_i, r.gene_j, r.gene_k))
                and all(pair_f[frozenset(c)] > 0.0
                        for c in itertools.combinations((r.gene_i, r.gene_j, r.gene_k), 2))
                and r.f_ijk > 0.0
                for r in t.itertuples()
            ], dtype=bool)
            per_setting.append(
                {
                    "arm": arm,
                    "modulation": modulation,
                    "alpha": alpha,
                    # A zero readout means the strain misses the growth floor. These
                    # fractions say whether an interaction score comes from a graded
                    # flux change or from a feasibility cliff.
                    "frac_singles_zero": float(np.mean(s["f"].to_numpy() == 0.0)),
                    "frac_pairs_zero": float(np.mean(p["f_ij"].to_numpy() == 0.0)),
                    "frac_triples_zero": float(np.mean(t["f_ijk"].to_numpy() == 0.0)),
                    "n_pairs": int(len(eps)),
                    "n_pairs_interacting": int(np.sum(np.abs(eps) > TOLERANCE)),
                    "frac_pairs_interacting": float(np.mean(np.abs(eps) > TOLERANCE)),
                    "median_abs_eps": float(np.median(np.abs(eps))),
                    "p90_abs_eps": float(np.percentile(np.abs(eps), 90)),
                    "eps_sign_split": sign_split(eps),
                    "n_triples": int(len(tau)),
                    "n_triples_interacting": int(np.sum(np.abs(tau) > TOLERANCE)),
                    "n_triples_graded": int(np.sum(graded)),
                    "n_triples_interacting_graded": int(
                        np.sum((np.abs(tau) > TOLERANCE) & graded)
                    ),
                    "frac_triples_interacting": float(
                        np.mean(np.abs(tau) > TOLERANCE)
                    ),
                    "median_abs_tau": float(np.median(np.abs(tau))),
                    "p90_abs_tau": float(np.percentile(np.abs(tau), 90)),
                    "tau_sign_split": sign_split(tau),
                    "median_f_single": float(s["f"].median()),
                    "spearman_mean_hops_vs_abs_tau": spearman_or_none(
                        t["mean_hops"].to_numpy(), np.abs(tau)
                    ),
                    "spearman_max_hops_vs_abs_tau": spearman_or_none(
                        t["max_hops"].to_numpy(), np.abs(tau)
                    ),
                }
            )
    setting_df = pd.DataFrame(per_setting)

    hop_distribution = {
        arm: dict(
            Counter(
                int(h) for h in genes_df[genes_df["arm"] == arm]["hops"].tolist()
            )
        )
        for arm in ("regulator", "enzyme")
    }

    summary = {
        "what_this_is": (
            "A property of the model defined in this script, not evidence about the "
            "measured strains. No number here is a titer."
        ),
        "model_path": MODEL_PATH,
        "model_sha256": sha256_of_file(MODEL_PATH),
        "yeast9": {
            "n_reactions": len(model.reactions),
            "n_metabolites": len(model.metabolites),
            "n_genes": len(model.genes),
        },
        "chassis": CHASSIS,
        "readout_exchanges": READOUT_EXCHANGES,
        "biomass_reaction": BIOMASS,
        "growth_fraction": GROWTH_FRACTION,
        "base_growth": float(base_growth),
        "base_readout": float(base_readout),
        "tolerance": TOLERANCE,
        "alphas": list(ALPHAS),
        "currency_metabolites": sorted(CURRENCY),
        "arm_genes": arm_genes,
        "enzyme_arm_rule": (
            "the ten of the thirteen core pathway genes with the most Yeast9 reactions, "
            "ties broken by pathway order in GENE_ORDER"
        ),
        "hop_distribution": hop_distribution,
        "per_setting": per_setting,
        "caveats": CAVEATS,
    }

    singles_df.to_csv(osp.join(RESULTS_DIR, "singles.csv"), index=False)
    pairs_df.to_csv(osp.join(RESULTS_DIR, "pairs.csv"), index=False)
    triples_df.to_csv(osp.join(RESULTS_DIR, "triples.csv"), index=False)
    genes_df.to_csv(osp.join(RESULTS_DIR, "genes.csv"), index=False)
    with open(osp.join(RESULTS_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # The setting the figure's middle panel shows. Not the setting with the largest
    # |tau|: most of this sweep is all-or-nothing, so at the largest-|tau| setting every
    # triple carries the SAME score and a distribution plot of it is a flat line. The
    # panel instead shows the setting whose |tau| values are most SPREAD, measured as the
    # p90 minus the median summed over both arms, which is the only place the readout
    # responds in a graded way. The largest-|tau| setting is reported in the text.
    spread = (
        setting_df["p90_abs_tau"] - setting_df["median_abs_tau"]
    ).groupby([setting_df["modulation"], setting_df["alpha"]]).sum()
    focus_modulation, focus_alpha = spread.idxmax()
    total_tau = (
        triples_df.assign(abs_tau=triples_df["tau"].abs())
        .groupby(["modulation", "alpha"])["abs_tau"]
        .sum()
    )
    loudest_modulation, loudest_alpha = total_tau.idxmax()

    make_figure(genes_df, triples_df, setting_df, settings, focus_modulation, focus_alpha)

    print_summary(
        genes_df,
        setting_df,
        settings,
        hop_distribution,
        focus_modulation,
        focus_alpha,
        loudest_modulation,
        loudest_alpha,
    )


def setting_label(modulation: str, alpha: float) -> str:
    return "binary" if modulation == "binary" else f"{alpha:g}"


def make_figure(
    genes_df: pd.DataFrame,
    triples_df: pd.DataFrame,
    setting_df: pd.DataFrame,
    settings: list[tuple[str, float]],
    focus_modulation: str,
    focus_alpha: float,
) -> None:
    apply_paper_style()
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(52.0)),
    )
    # Room above the titles for the panel letters: at top=0.88 the letters sat on the
    # canvas edge and printed cut in half (author review, 2026.09.18).
    fig.subplots_adjust(left=0.045, right=0.995, bottom=0.30, top=0.84, wspace=0.42)

    # (a) hop count from the perturbed reactions to the readout, per arm.
    ax = axes[0]
    all_hops = sorted({int(h) for h in genes_df["hops"].tolist()})
    width = 0.38
    for k, arm in enumerate(("regulator", "enzyme")):
        counts = Counter(int(h) for h in genes_df[genes_df["arm"] == arm]["hops"])
        ax.bar(
            [i + (k - 0.5) * width for i in range(len(all_hops))],
            [counts.get(h, 0) for h in all_hops],
            width=width,
            color=ARM_COLOR[arm],
            edgecolor="black",
            linewidth=0.5,
            label=arm,
        )
    ax.set_xticks(range(len(all_hops)))
    ax.set_xticklabels([str(h) for h in all_hops])
    ax.set_xlabel("hops from perturbed reactions to the readout")
    ax.set_ylabel("genes")
    ax.set_title("perturbation distance", fontsize=6)
    ax.legend(fontsize=6, loc="upper right", handlelength=1.2)
    panel_label(ax, "a")

    # (b, c) WHERE EVERY NONZERO SCORE COMES FROM: the fraction of combinations at each
    # order that miss the growth floor and read exactly 0. A panel of |tau| values stood
    # here and was misleading, because at every setting where this model scores anything
    # the score is arithmetic on those zeros rather than a graded response.
    orders = [
        ("1 deletion", "frac_singles_zero", PLOT_PALETTE[2]),
        ("2 deletions", "frac_pairs_zero", PLOT_PALETTE[3]),
        ("3 deletions", "frac_triples_zero", PLOT_PALETTE[4]),
    ]
    bar_w = 0.27
    for ax, arm, letter in ((axes[1], "regulator", "b"), (axes[2], "enzyme", "c")):
        for k, (label, column, color) in enumerate(orders):
            fracs = [
                setting_df[
                    (setting_df["modulation"] == m)
                    & (setting_df["alpha"] == a)
                    & (setting_df["arm"] == arm)
                ][column].item()
                for m, a in settings
            ]
            ax.bar(
                [i + (k - 1) * bar_w for i in range(len(settings))],
                fracs,
                width=bar_w,
                color=color,
                edgecolor="black",
                linewidth=0.4,
                label=label,
            )
        ax.set_xticks(range(len(settings)))
        ax.set_xticklabels([setting_label(m, a) for m, a in settings], rotation=90)
        ax.set_ylim(0, 1.32)
        ax.set_xlabel("capacity multiplier")
        ax.set_ylabel("fraction missing the growth floor")
        ax.set_title(f"{arm} arm, strains that cannot grow", fontsize=6)
        panel_label(ax, letter)
    axes[1].legend(fontsize=5, loc="upper right", handlelength=1.0, labelspacing=0.25)

    # (d) fraction of triples with |tau| above tolerance, every setting, both arms.
    ax = axes[3]
    labels = [setting_label(m, a) for m, a in settings]
    for k, arm in enumerate(("regulator", "enzyme")):
        fracs = [
            setting_df[
                (setting_df["modulation"] == m)
                & (setting_df["alpha"] == a)
                & (setting_df["arm"] == arm)
            ]["frac_triples_interacting"].item()
            for m, a in settings
        ]
        ax.bar(
            [i + (k - 0.5) * width for i in range(len(settings))],
            fracs,
            width=width,
            color=ARM_COLOR[arm],
            edgecolor="black",
            linewidth=0.5,
            label=arm,
        )
    ax.set_xticks(range(len(settings)))
    ax.set_xticklabels(labels, rotation=90)
    # Headroom above the full-height bars so the legend does not cover one.
    ax.set_ylim(0, 1.32)
    ax.set_xlabel("capacity multiplier")
    ax.set_ylabel(r"fraction of triples with $|\tau| >$ tol")
    ax.set_title("settings where a score appears", fontsize=6)
    ax.legend(fontsize=5, loc="upper right", handlelength=1.0, labelspacing=0.25)
    panel_label(ax, "d")

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)

    png = osp.join(IMAGES_DIR, f"{FIG_STEM}.png")
    svg = osp.join(IMAGES_DIR, f"{FIG_STEM}.svg")
    fig.savefig(png, dpi=600)
    savefig_true_size_svg(fig, svg)
    plt.close(fig)
    print(f"\nfigure written\n  {png}\n  {svg}")


def print_summary(
    genes_df: pd.DataFrame,
    setting_df: pd.DataFrame,
    settings: list[tuple[str, float]],
    hop_distribution: dict,
    focus_modulation: str,
    focus_alpha: float,
    loudest_modulation: str,
    loudest_alpha: float,
) -> None:
    print("\n" + "=" * 92)
    print("REGULATOR VERSUS ENZYME EPISTASIS, IN A MODEL")
    print("=" * 92)
    print(
        "Everything below is a property of the model defined in this script. It is NOT\n"
        "evidence about the measured strains, and no number here is a titer."
    )

    print("\nhop count from the perturbed reactions to the readout")
    for arm in ("regulator", "enzyme"):
        sub = genes_df[genes_df["arm"] == arm]
        print(
            f"  {arm:10s} {dict(sorted(hop_distribution[arm].items()))}  "
            f"median {sub['hops'].median():g}  "
            f"reactions per gene, median {sub['n_reactions'].median():g}"
        )

    print(
        "\n'f=0' is the fraction of strains that miss the growth floor and so export\n"
        "nothing. A high f=0 with a median single f of 1 means the score comes from a\n"
        "feasibility cliff, not from a graded flux change."
    )
    header = (
        f"{'setting':>8s} {'arm':>10s} {'med f':>7s} {'f=0 1/2/3':>15s} "
        f"{'pairs>tol':>10s} {'med|eps|':>9s} {'p90|eps|':>9s} {'trip>tol':>9s} "
        f"{'med|tau|':>9s} {'p90|tau|':>9s} {'tau +/-/0':>14s}"
    )
    print("\n" + header)
    print("-" * len(header))
    for modulation, alpha in settings:
        for arm in ("regulator", "enzyme"):
            r = setting_df[
                (setting_df["modulation"] == modulation)
                & (setting_df["alpha"] == alpha)
                & (setting_df["arm"] == arm)
            ].iloc[0]
            split = r["tau_sign_split"]
            print(
                f"{setting_label(modulation, alpha):>8s} {arm:>10s} "
                f"{r['median_f_single']:7.4f} "
                f"{r['frac_singles_zero']:4.2f}/{r['frac_pairs_zero']:4.2f}/"
                f"{r['frac_triples_zero']:4.2f} "
                f"{r['n_pairs_interacting']:4d}/{r['n_pairs']:<5d} "
                f"{r['median_abs_eps']:9.4f} {r['p90_abs_eps']:9.4f} "
                f"{r['n_triples_interacting']:4d}/{r['n_triples']:<4d} "
                f"{r['median_abs_tau']:9.4f} {r['p90_abs_tau']:9.4f} "
                f"{split['positive']:4d}/{split['negative']:4d}/{split['zero']:4d}"
            )

    print("\nSpearman correlation of a triple's hop count with |tau|")
    for modulation, alpha in settings:
        for arm in ("regulator", "enzyme"):
            r = setting_df[
                (setting_df["modulation"] == modulation)
                & (setting_df["alpha"] == alpha)
                & (setting_df["arm"] == arm)
            ].iloc[0]
            parts = []
            for key, tag in (
                ("spearman_mean_hops_vs_abs_tau", "mean"),
                ("spearman_max_hops_vs_abs_tau", "max"),
            ):
                v = r[key]
                parts.append(
                    f"{tag} rho {v['rho']:+.3f} (p {v['p']:.3g})"
                    if v is not None
                    else f"{tag} undefined, one side is constant"
                )
            print(
                f"  {setting_label(modulation, alpha):>8s} {arm:>10s}  " + "; ".join(parts)
            )

    for tag, modulation, alpha in (
        ("largest |tau|", loudest_modulation, loudest_alpha),
        ("most spread in |tau|", focus_modulation, focus_alpha),
    ):
        sub = setting_df[
            (setting_df["modulation"] == modulation) & (setting_df["alpha"] == alpha)
        ]
        reg = sub[sub["arm"] == "regulator"].iloc[0]
        enz = sub[sub["arm"] == "enzyme"].iloc[0]
        print(
            f"\n{tag}: setting {setting_label(modulation, alpha)}, regulator "
            f"{reg['n_triples_interacting']}/{reg['n_triples']} triples interacting at "
            f"median |tau| {reg['median_abs_tau']:.4f} and p90 {reg['p90_abs_tau']:.4f}, "
            f"enzyme {enz['n_triples_interacting']}/{enz['n_triples']} at median "
            f"{enz['median_abs_tau']:.4f} and p90 {enz['p90_abs_tau']:.4f}"
        )

    reg_all = setting_df[setting_df["arm"] == "regulator"]
    enz_all = setting_df[setting_df["arm"] == "enzyme"]
    n_reg = int(reg_all["n_triples_interacting"].sum())
    n_enz = int(enz_all["n_triples_interacting"].sum())
    p90_reg = float(reg_all["p90_abs_tau"].max())
    p90_enz = float(enz_all["p90_abs_tau"].max())
    print(
        f"\nacross all {len(settings)} settings: regulator {n_reg} interacting triples "
        f"(largest p90 |tau| {p90_reg:.4f}), enzyme {n_enz} (largest p90 |tau| "
        f"{p90_enz:.4f})"
    )

    print("\ncaveats a Methods paragraph would have to state")
    for c in CAVEATS:
        print(f"  - {c}")

    graded_reg = int(reg_all["n_triples_interacting_graded"].sum())
    graded_enz = int(enz_all["n_triples_interacting_graded"].sum())
    print(
        f"\nof those, interacting triples whose seven readouts are ALL above the growth "
        f"floor: regulator {graded_reg}, enzyme {graded_enz}"
    )

    for _, r in setting_df[setting_df["n_triples_interacting_graded"] > 0].iterrows():
        print(
            f"  graded interacting triples: {r['n_triples_interacting_graded']} in the "
            f"{r['arm']} arm at setting {setting_label(r['modulation'], r['alpha'])}"
        )

    if graded_reg == 0 and graded_enz == 0:
        verdict = (
            "This model does NOT settle the comparison. Every nonzero interaction score "
            "in either arm belongs to a triple in which at least one of the seven "
            "readouts the score is computed from missed the growth floor and read "
            "exactly 0, so every score is arithmetic on a feasibility cliff rather than "
            "a measure of how two deletions combine. No graded interaction was produced "
            "in either arm at any setting tried."
        )
    elif n_reg > n_enz and p90_reg > p90_enz:
        verdict = (
            "In this model the regulator arm produces MORE and LARGER apparent "
            "interaction at the single readout than the enzyme arm."
        )
    elif n_reg < n_enz and p90_reg < p90_enz:
        verdict = (
            "In this model the ENZYME arm produces more and larger apparent interaction "
            "at the single readout than the regulator arm, the opposite of the "
            "hypothesis."
        )
    elif n_reg == n_enz and p90_reg == p90_enz:
        verdict = (
            "In this model the two arms produce the same apparent interaction at the "
            "single readout, so the model shows NO difference between them."
        )
    else:
        verdict = (
            f"In this model the two measures disagree: the regulator arm has "
            f"{n_reg} interacting triples against the enzyme arm's {n_enz} while the "
            f"largest p90 |tau| is {p90_reg:.4f} against {p90_enz:.4f}, so the model "
            f"shows no consistent difference between the arms."
        )
    if not (graded_reg == 0 and graded_enz == 0):
        verdict += (
            f" That comparison is a comparison of feasibility cliffs: of the {n_reg} "
            f"interacting triples in the regulator arm and {n_enz} in the enzyme arm, "
            f"{graded_reg} and {graded_enz} are graded, meaning all seven readouts the "
            f"score is computed from clear the growth floor. Every other score is "
            f"arithmetic on a readout of exactly zero."
        )
    print("\nVERDICT")
    print(f"  {verdict}")


if __name__ == "__main__":
    main()
