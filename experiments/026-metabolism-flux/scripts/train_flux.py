# experiments/026-metabolism-flux/scripts/train_flux.py
# [[experiments.026-metabolism-flux.scripts.train_flux]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/train_flux.py

r"""Diagnostic runs for the enzyme-constrained thermodynamic flux layer.

Run from the worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/train_flux.py --arms pooled,flux_anchored

Two phenotypes, both single-deletion, both already built into
``fig6_pigment_transfer``:

* **betaxanthin** (Cachera 2023) -- 4,735 strains, a heterologous product whose precursor
  is tyrosine. Noise ceiling ``r = 0.914``.
* **mulleder19** (Mulleder 2016) -- 4,678 strains x 19 intracellular amino acids in mM.
  Every one of the 19 resolves to a cytosolic yeast-GEM metabolite, so the flux head reads
  the model's own turnover at exactly the species that were measured.

WHY A PURPOSE-BUILT LOOP RATHER THAN THE 019 HARNESS
-----------------------------------------------------
The 019 multitask trainer is 3,264 lines and its loss is assembled from head outputs plus
one graph-regularization scalar. The flux layer adds **five** more loss terms and, more
importantly, five per-epoch **feasibility diagnostics** that are the actual object of this
experiment -- a flux vector that fits betaxanthin while violating mass balance has not
learned metabolism. Threading those through the big harness would put the diagnostics in
the position of an afterthought. Here they are first-class and logged every epoch.

THE ARMS
--------
Each arm changes exactly one thing relative to the one above it, so a difference is
attributable:

=================  ==========================================================
arm                what it adds
=================  ==========================================================
``pooled``         baseline: the existing MLP-over-pooled-tokens readout
``flux_off``       flux layer, GPR availability, mass balance. No thermo, no kcat
``flux_free``      + learned potential (loop-freedom, no tabulated energies)
``flux_anchored``  + tabulated delta_f G', enzyme capacity, budget, dissipation
=================  ==========================================================

``pooled`` -> ``flux_off`` asks whether routing the prediction through a stoichiometric
network helps at all. ``flux_free`` -> ``flux_anchored`` asks whether the **measured**
energies carry information beyond the structural loop-freedom that any potential gives.
That second contrast is the one that decides whether the thermodynamic table earns its
place, and it is why the two thermo modes are separate arms rather than one flag.
"""

import argparse
import json
import os
import os.path as osp
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv

from torchcell.data import DeletionKeyedGenotypeAggregator, MeanExperimentDeduplicator
from torchcell.data.graph_processor import Perturbation
from torchcell.data.neo4j_cell import Neo4jCellDataset
from torchcell.datamodules import CellDataModule
from torchcell.graph import SCerevisiaeGraph
from torchcell.graph.graph import build_gene_multigraph
from torchcell.metabolism.constraints import (
    ThermoMode,
    build_gem_tensors,
    null_space_basis,
)
from torchcell.metabolism.flux_layer import FluxLayer, FluxLayerConfig
from torchcell.metabolism.parameters import (
    KINETICS_MIRROR,
    PLAUSIBILITY_RESULT,
    THERMO_EQUILIBRATOR_MIRROR,
    VERIFIED_PAIRS_FEATURE,
    SubstrateMatch,
    ThermoSource,
    load_predicted_unit_table,
    molecular_weight_table,
    resolve_kcat_table,
    resolve_km_table,
    resolve_reaction_delta_g,
)
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.models.cell_graph_transformer_metabolism import (
    CellGraphTransformerMetabolism,
)
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
DATASET_ROOT = osp.join(
    DATA_ROOT, "data/torchcell/experiments/019-simb-multimodal/fig6_pigment_transfer"
)
QUERY_PATH = osp.join(
    EXPERIMENT_ROOT, "019-simb-multimodal/queries/fig6_pigment_transfer.cql"
)
OED_MIRROR = osp.join(
    DATA_ROOT, "data/enzyme_kinetics/open_enzyme_database/scerevisiae"
)
#: The whole Open Enzyme Database, every organism. It supplies the EC-plus-substrate
#: homolog tier, which is measured chemistry from a protein that is NOT this organism's and
#: is tagged apart from the measured tier for exactly that reason.
HOMOLOG_MIRROR = osp.join(DATA_ROOT, "data/enzyme_kinetics/open_enzyme_database/all")

#: The 19 Mulleder columns in the key-sorted order the Perturbation graph processor emits
#: (``sorted(metabolite_level)``), paired with the cytosolic yeast-GEM species each one is.
#: Resolving these is what the Mulleder loader deferred as ``target_metabolite_ids``.
AMINO_ACID_TO_GEM_METABOLITE = {
    "alanine": "s_0955",
    "arginine": "s_0965",
    "asparagine": "s_0969",
    "aspartate": "s_0973",
    "glutamate": "s_0991",
    "glutamine": "s_0999",
    "glycine": "s_1003",
    "histidine": "s_1006",
    "isoleucine": "s_1016",
    "leucine": "s_1021",
    "lysine": "s_1025",
    "methionine": "s_1029",
    "phenylalanine": "s_1032",
    "proline": "s_1035",
    "serine": "s_1039",
    "threonine": "s_1045",
    "tryptophan": "s_1048",
    "tyrosine": "s_1051",
    "valine": "s_1056",
}

#: Betaxanthin's route is tyrosine -> L-DOPA -> betalamic acid, and the Cachera cassette
#: additionally carries feedback-resistant ARO4 and ARO7, i.e. it deregulates the shikimate
#: pathway upstream of tyrosine. The named precursors are therefore cytosolic tyrosine and
#: the two aromatic-pathway intermediates the cassette's alleles act on.
BETAXANTHIN_PRECURSORS = ["s_1051", "s_0188", "s_1032"]

ARMS: dict[str, dict[str, Any]] = {
    "pooled": {"flux": False},
    "flux_off": {
        "flux": True,
        "thermo_mode": ThermoMode.OFF,
        "use_enzyme_capacity": False,
        "use_protein_budget": False,
        "use_dissipation_limit": False,
    },
    "flux_free": {
        "flux": True,
        "thermo_mode": ThermoMode.FREE,
        "use_enzyme_capacity": False,
        "use_protein_budget": False,
        "use_dissipation_limit": False,
    },
    "flux_anchored": {
        "flux": True,
        "thermo_mode": ThermoMode.ANCHORED,
        "use_enzyme_capacity": True,
        "use_protein_budget": True,
        "use_dissipation_limit": True,
    },
    # THE PROTEIN NODE (2026-09-09). The abundance head replaces the nominal E_bar with a
    # predicted per-gene enzyme abundance; the capacity bound is kcat * E_u and the
    # protein budget is the forward pool constraint on E. Two arms: capacity alone, and
    # capacity plus the anchored thermodynamics. Dissipation is a readout in both.
    "flux_capacity": {
        "flux": True,
        "thermo_mode": ThermoMode.OFF,
        "use_enzyme_capacity": True,
        "use_protein_budget": True,
        "use_dissipation_limit": False,
        "use_abundance_head": True,
    },
    "flux_abundance": {
        "flux": True,
        "thermo_mode": ThermoMode.ANCHORED,
        "use_enzyme_capacity": True,
        "use_protein_budget": True,
        "use_dissipation_limit": False,
        "use_abundance_head": True,
    },
    # The exactness-budget arm: mass balance exact by construction, box soft.
    "flux_nullspace": {
        "flux": True,
        "parameterization": "nullspace",
        "thermo_mode": ThermoMode.ANCHORED,
        "use_enzyme_capacity": True,
        "use_protein_budget": True,
        "use_dissipation_limit": True,
    },
    # WAVE 6 (2026-09-10). THE PROJECTION HEAD. `box` and `nullspace` each buy one exact
    # constraint by giving up the other; `projection` starts at v0 = N z and alternates the
    # balance projection with the box clip, so both hold, with the one named by
    # `projection_end` (default balance) exact and the other's residual reported per sample.
    # The motivation is the Genesis metabolic-grounding target, which asks what fraction of
    # NOMINATED strains are feasible: a penalty that is small on average says nothing about
    # the top 100, and a parameterization does.
    #
    # `flux_projection` is the structural analog of `flux_off`: no thermodynamics, no
    # capacity, no budget, so pooled -> flux_projection isolates what routing the prediction
    # through a feasible flux buys, with no measured parameter involved.
    "flux_projection": {
        "flux": True,
        "parameterization": "projection",
        "thermo_mode": ThermoMode.OFF,
        "use_enzyme_capacity": False,
        "use_protein_budget": False,
        "use_dissipation_limit": False,
    },
    # EVERYTHING ON. Anchored thermodynamics, the predicted-kcat capacity bound through the
    # abundance head, the forward protein-pool budget, and the dissipation limit, on top of
    # a flux that is balanced and in-box by construction. This is the arm that answers
    # whether the physics can be satisfied at once rather than one term at a time; the
    # dissipation limit is trained against here rather than held out as a readout, so a
    # model that respects it in this arm has been told to, not discovered it.
    "flux_projection_anchored": {
        "flux": True,
        "parameterization": "projection",
        "thermo_mode": ThermoMode.ANCHORED,
        "use_enzyme_capacity": True,
        "use_protein_budget": True,
        "use_dissipation_limit": True,
        "use_abundance_head": True,
    },
    # THE DISSIPATION LIMIT HELD OUT (2026-09-14). The Gibbs dissipation limit is a
    # verification readout, not a training target: a model that stays under 3,700 J/gDW/h
    # because it was penalized for exceeding it has been told to, not discovered it, and its
    # dissipation feasibility then says nothing. Every 027 cell on `flux_projection_anchored`
    # through wave 8 trained against the limit, so this arm is the same everything-on recipe
    # with that one term off; `g_diss_mean` is still recorded every epoch as a readout.
    "flux_projection_anchored_nodiss": {
        "flux": True,
        "parameterization": "projection",
        "thermo_mode": ThermoMode.ANCHORED,
        "use_enzyme_capacity": True,
        "use_protein_budget": True,
        "use_dissipation_limit": False,
        "use_abundance_head": True,
    },
    # The same, sampled. `stochastic` on a projection parameterization is read as the LATENT
    # draw z = mu + sigma eps with a KL against N(0, I), which is the only new boolean an
    # arm can reach through 026's build_flux_layer; the KL weight is FluxLayerConfig's
    # `kl_weight` default of 1e-3, since build_flux_layer writes the weights dict itself.
    "flux_projection_stochastic": {
        "flux": True,
        "parameterization": "projection",
        "thermo_mode": ThermoMode.ANCHORED,
        "use_enzyme_capacity": True,
        "use_protein_budget": True,
        "use_dissipation_limit": True,
        "use_abundance_head": True,
        "stochastic": True,
    },
    # THE SECOND LAW AS A BOX (2026-09-11). Cells 1689/1690/1708 made mass balance exact
    # and left the thermodynamics soft, and the soft term closed nothing: on the top 100
    # predicted strains of job 1708, at a constraint scale of 0.1, the balance criterion was
    # met by every strain and `thermo_model_violation_frac` still read a median of 0.405, so
    # the Genesis criterion failed on the deltaG half in every arm run to that point.
    #
    # The fix is not a bigger weight. For a reaction whose energy has a known sign, the
    # second law IS a bound: dG > 0 forbids forward flux, dG < 0 forbids reverse. Writing it
    # as a bound puts it where the box and the balance already are, and the alternating
    # projection then ends on a clip that satisfies all three. `tdg` in the parameterization
    # string is that switch, and `_box` is what makes the clip last, without which the
    # directional bound would be reported as exact while the balance projection moved the
    # flux back off it. Twenty alternations, because the balance residual is now the only
    # loose end and it is the one K buys down.
    "flux_projection_anchored_tdg": {
        "flux": True,
        "parameterization": "projection_k20_box_tdg",
        "thermo_mode": ThermoMode.ANCHORED,
        "use_enzyme_capacity": True,
        "use_protein_budget": True,
        "use_dissipation_limit": True,
        "use_abundance_head": True,
    },
    # The same, sampled, so the feasibility question can be asked of a DISTRIBUTION over a
    # nominated strain's flux rather than of one point.
    "flux_projection_anchored_tdg_stochastic": {
        "flux": True,
        "parameterization": "projection_k20_box_tdg",
        "thermo_mode": ThermoMode.ANCHORED,
        "use_enzyme_capacity": True,
        "use_protein_budget": True,
        "use_dissipation_limit": True,
        "use_abundance_head": True,
        "stochastic": True,
    },
    # K = 20 alternations instead of 5. The step count rides in the parameterization string
    # because that field is the only one build_flux_layer forwards; everything else matches
    # `flux_projection_anchored`, so the pair is a clean read on whether the residual the
    # projection leaves is large enough to matter to the phenotype.
    # THE BOX END WITHOUT THE SIGN TIGHTENING (2026-09-11). The control that separates two
    # explanations of why the tdg head lost the phenotype signal. Measured at initialization
    # on yeast-GEM with the REAL gene universe, untrained: ending on the clip with NO
    # thermodynamics in the box takes the flux from 11.9 percent zeros to 44.1 percent and
    # the balance ratio from 5.4e-9 to 0.27; the sign tightening then takes it to 68.0
    # percent and collapses 17.1 percent of boxes to a point, against 0.2 percent before it.
    # So respecting the bounds already costs most of the density on its own, and this arm is
    # what says whether that alone accounts for the lost signal.
    "flux_projection_anchored_k20_boxend": {
        "flux": True,
        "parameterization": "projection_k20_box",
        "thermo_mode": ThermoMode.ANCHORED,
        "use_enzyme_capacity": True,
        "use_protein_budget": True,
        "use_dissipation_limit": True,
        "use_abundance_head": True,
    },
    # THE MARGIN. The directional box at its default margin pins 68.0 percent of reactions
    # below 1e-6 at initialization and 70.6 percent on the trained top-100 strains of job
    # 1725, whose head scores at chance. `tdgm5` constrains only reactions at least 5 kJ/mol
    # from equilibrium: at initialization that recovers the mean absolute flux from 0.067 to
    # 0.184 and costs a second-law violation fraction of 0.123, so it trades exactness for
    # density rather than buying both. Implemented and tested, NOT run: the box-end control
    # above is the cheaper discriminator and took the cells.
    "flux_projection_anchored_tdgm5": {
        "flux": True,
        "parameterization": "projection_k20_box_tdgm5",
        "thermo_mode": ThermoMode.ANCHORED,
        "use_enzyme_capacity": True,
        "use_protein_budget": True,
        "use_dissipation_limit": True,
        "use_abundance_head": True,
    },
    # THE NOMINAL ABUNDANCE PATH WITH THE WIDENED KCAT CEILING. `flux_nullspace`'s switches
    # with a 2-sigma widening of the k_cat table, which is the only rule that lifts all
    # three reactions the pFBA reference carries above their box ceiling: no subunit
    # aggregation rule reaches r_0961, a single-gene unit. The channel is overridden by the
    # 027 arm file, so the parameterization named here is not what the warm-start cells run.
    #
    # It exists as a registry entry rather than as a field in the 027 arm file because
    # ArmConfig forbids extra keys, so `base_arm` is the only channel an 027 cell has for a
    # k_cat rule, and `base_arm` is serialized into the result file with the cell.
    "flux_nullspace_kcat_sigma2": {
        "flux": True,
        "parameterization": "nullspace",
        "thermo_mode": ThermoMode.ANCHORED,
        "use_enzyme_capacity": True,
        "use_protein_budget": True,
        "use_dissipation_limit": True,
        "kcat_ceiling_sigma_k": 2.0,
    },
    "flux_projection_anchored_k20": {
        "flux": True,
        "parameterization": "projection_k20",
        "thermo_mode": ThermoMode.ANCHORED,
        "use_enzyme_capacity": True,
        "use_protein_budget": True,
        "use_dissipation_limit": True,
        "use_abundance_head": True,
    },
}


def build_dataset() -> Neo4jCellDataset:
    """Open the already-built pigment/metabolome dataset. Never rebuilds."""
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        # See the note in 027's train_bx.py: the default `overwrite=True` rebuilds
        # data.db under every worker, which is a write race when workers share a node.
        overwrite=False,
    )
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )
    with open(QUERY_PATH) as f:
        query = f.read()
    return Neo4jCellDataset(
        root=DATASET_ROOT,
        query=query,
        gene_set=genome.gene_set,
        graphs=build_gene_multigraph(
            graph=graph, graph_names=["physical", "regulatory"]
        ),
        incidence_graphs=None,
        node_embeddings={},
        converter=None,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=DeletionKeyedGenotypeAggregator,
        graph_processor=Perturbation(),
    )


#: Media whose recipe object carries NO provenance and NO concentrations: the composition
#: is the standard Difco/Sigma formulation named in `torchcell/datamodels/media.py` and
#: only the uptake magnitudes are sourced (Suthers). A cell bounded by one of these is
#: running on an UNSOURCED medium, and its result file says so rather than leaving a reader
#: to assume the bound vector is as well sourced as the kcat table.
UNSOURCED_MEDIA = frozenset({"SC", "SC-URA"})


def media_report(model: Any, media_name: str | None) -> dict[str, Any]:
    """Apply a medium to ``model``'s exchange bounds, or record that none was applied.

    MUTATES ``model`` when a medium is named: ``MediaBounds.apply`` closes every exchange
    and opens exactly the named recipe, which is what makes a medium a statement about the
    whole boundary rather than an additive patch. The wild-type growth rate is measured
    before and after, because that single number is what says whether the change was a
    no-op, a loosening or a starvation, and it is the first thing to check when an arm
    moves after a medium is applied.

    No fallback: a name the ontology does not carry is an error here rather than a silent
    return to the shipped bounds, which would be a cell that reports a medium it never ran.
    """
    if media_name is None:
        return {
            "applied": False,
            "note": (
                "shipped yeast-GEM 9.0.2 exchange bounds, where all 20 amino-acid "
                "exchanges have lower_bound 0, i.e. effectively synthetic minimal. This "
                "is what every 026 and 027 arm before 2026-09-14 ran on."
            ),
        }
    from torchcell.metabolism.media import FBA_MEDIA, media_to_bounds

    assert media_name in FBA_MEDIA, (
        f"medium {media_name!r} is not in FBA_MEDIA ({sorted(FBA_MEDIA)}). The bound "
        "vector comes from the ontology recipe or the cell does not run; it is never "
        "quietly replaced by the shipped bounds."
    )
    media = FBA_MEDIA[media_name]
    growth_before = float(model.slim_optimize())
    bounds = media_to_bounds(media, model)
    bounds.apply(model)
    growth_after = float(model.slim_optimize())
    # The bound vector is data: written once under DATA_ROOT, hash-named, so every
    # store and cell that ran on it can name the exact file rather than a code constant.
    artifact = bounds.write_artifact(
        osp.join(os.environ["DATA_ROOT"], "data/torchcell/media_bounds")
    )
    return {
        "applied": True,
        "media_name": media_name,
        "recipe": media.name,
        "bounds_artifact": artifact,
        "bounds_sha256": bounds.sha256(),
        "state": media.state,
        "n_components": bounds.n_components,
        "n_resolved": bounds.n_resolved,
        "n_exchanges_opened": len(bounds.bounds),
        "unresolved": bounds.unresolved_names,
        "excluded_by_role": bounds.excluded_names,
        "wild_type_growth_before": growth_before,
        "wild_type_growth_after": growth_after,
        "provenance": (
            "UNSOURCED: the recipe object carries no provenance and no concentrations; "
            "its composition is the standard Difco/Sigma formulation named in "
            "torchcell/datamodels/media.py and only the uptake magnitudes are Suthers'."
            if media_name in UNSOURCED_MEDIA
            else "see the recipe's docstring in torchcell/metabolism/media.py"
        ),
    }


def build_flux_layer(
    arm: dict[str, Any], model_gene_ids: list[str], hidden_dim: int
) -> tuple[FluxLayer | None, dict[str, Any]]:
    """Construct the flux layer for one arm, with real MW and OED-resolved kcat.

    Four things an arm may now ask for beyond its constraint switches, all keyed off the arm
    dict so the registry stays the single source of truth:

    * ``kcat_tier``: ``"measured"`` keeps the database tiers and the organism default only;
      ``"predicted"`` inserts the sequence-predictor tier between them.
    * ``kcat_homolog_tier``: whether a cross-organism EC-plus-substrate measurement may fill
      a unit the organism's own assays do not reach. ON by default since 2026-09-11: it is
      the tier a chassis with no assays of its own would live on, and turning it off is what
      a yeast-only ablation does. It is tagged apart from the measured tier and the flux
      layer's coarse split counts it with the predictions, never as a measurement.
    * ``thermo_source``: which route supplies the standard reaction energies.
    * ``constraint_warmup_epochs`` / ``constraint_final_scale``: the ramp on the physics.

    **The measured tier is substrate-matched.** ``SubstrateMatch`` is built here for every
    arm, predicted tier or not, because without it no database row can be checked against
    the reaction it is being attached to and the resolver accepts none. Keying on the
    accession alone attached 74 of 148 turnover numbers to a substrate the reaction does not
    consume; see [[experiments.026-metabolism-flux.kinetic-coverage-closure]].
    """
    if not arm["flux"]:
        return None, {}
    source = YeastGEM()
    # THE GROWTH MEDIUM, applied BEFORE the tensors are taken and reported either way.
    #
    # Every 026 and 027 arm to date has run on the SHIPPED yeast-GEM boundary, where all 20
    # amino-acid exchanges have lower_bound 0, i.e. effectively synthetic minimal. That is
    # right for Mulleder, whose strains are prototrophs on minimal agar, and wrong for
    # Cachera (solid YPD) and Montano Lopez (SC liquid), both of which supply the very
    # amino acids their readouts depend on. `media_name` keys
    # `torchcell.metabolism.media.FBA_MEDIA`; absent, nothing is touched and the layer is
    # bit-identical to every earlier build.
    media_meta = media_report(source.model, arm.get("media_name"))
    gem = build_gem_tensors(source.model, model_dir=source.model_dir)
    units = gem.catalytic_units
    mw = molecular_weight_table(source.model_dir, units.gene_ids)
    kcat_tier = arm.get("kcat_tier", "measured")
    kinetics_dir = osp.join(DATA_ROOT, KINETICS_MIRROR)
    substrates = SubstrateMatch.from_verified_pairs(
        osp.join(kinetics_dir, VERIFIED_PAIRS_FEATURE)
    )
    homolog_mirror = HOMOLOG_MIRROR if arm.get("kcat_homolog_tier", True) else None
    # THE TWO KCAT TABLE RULES, forwarded from the arm so they are serialized with the cell.
    # Both default to what every arm before 2026-09-14 resolved, so an arm that names
    # neither builds a bit-identical table.
    #
    # `complex_aggregation` is a CONVENTION: min is "a complex cannot turn over faster than
    # its slowest subunit", max is GECKO 3's reading. It moves complexes and leaves the
    # table's center alone, which is why the median k_cat is 4.27 under min and 4.38 under
    # max while ATP synthase moves 65x.
    #
    # `kcat_ceiling_sigma_k` is an UNCERTAINTY statement on the shipped convention:
    # k_cat <- k_cat * 10^(k sigma_u) with sigma from parameters.KCAT_SIGMA_LADDER. It is
    # the only rule that reaches a single-gene unit, which is why r_0961 needs it.
    #
    # Applied to the k_cat side only. K_M keeps its own reduction, the maximum, because the
    # affinity argument is a different one and the parameter review does not contest it.
    complex_aggregation = arm.get("complex_aggregation", "min")
    kcat_ceiling_sigma_k = float(arm.get("kcat_ceiling_sigma_k", 0.0))
    predicted_kcat = predicted_km = None
    km = None
    if kcat_tier == "predicted":
        plausibility = osp.join(EXPERIMENT_ROOT, PLAUSIBILITY_RESULT)
        predicted_kcat = load_predicted_unit_table(
            "k_cat",
            units.n_units,
            kinetics_dir,
            plausibility,
            complex_aggregation=complex_aggregation,
        )
        predicted_km = load_predicted_unit_table(
            "K_M", units.n_units, kinetics_dir, plausibility
        )
        km = resolve_km_table(
            units,
            source.model_dir,
            OED_MIRROR,
            predicted=predicted_km,
            substrates=substrates,
            homolog_mirror_dir=homolog_mirror,
        )
    elif kcat_tier != "measured":
        raise ValueError(
            f"kcat_tier must be 'measured' or 'predicted', got {kcat_tier!r}"
        )
    kcat = resolve_kcat_table(
        units,
        source.model_dir,
        OED_MIRROR,
        predicted=predicted_kcat,
        substrates=substrates,
        homolog_mirror_dir=homolog_mirror,
        complex_aggregation=complex_aggregation,
        kcat_ceiling_sigma_k=kcat_ceiling_sigma_k,
    )
    # THE SHUFFLED-PARAMETER CONTROL. Permuting the resolved kcat vector ACROSS catalytic
    # units keeps the architecture, the parameter count, the coverage tiers and the value
    # DISTRIBUTION exactly as they are, and destroys only the correspondence between a
    # turnover number and the reaction it was measured for. An arm that still beats its
    # pooled control under this permutation gained something from the architecture rather
    # than from the measured kinetics, which is the question the control exists to ask.
    # Seeded and recorded, so the permutation is reproducible from the result file.
    kcat_shuffle_seed = arm.get("shuffle_kcat_seed")
    kcat_shuffle_meta: dict[str, Any] = {"applied": False}
    if kcat_shuffle_seed is not None:
        generator = np.random.default_rng(int(kcat_shuffle_seed))
        order = generator.permutation(kcat.values.shape[0])
        before = float(np.nansum(kcat.values))
        kcat.values = kcat.values[order]
        kcat_shuffle_meta = {
            "applied": True,
            "seed": int(kcat_shuffle_seed),
            "n_units": int(order.shape[0]),
            "n_moved": int((order != np.arange(order.shape[0])).sum()),
            # The multiset is preserved by construction; the sum is the cheap witness.
            "value_sum_before": before,
            "value_sum_after": float(np.nansum(kcat.values)),
            "note": (
                "kcat values permuted across catalytic units. Architecture, parameter "
                "count, tier counts and value distribution are unchanged; only the "
                "unit-to-value correspondence is destroyed."
            ),
        }

    thermo_source = ThermoSource(arm.get("thermo_source", "shipped"))
    reaction_thermo = None
    if thermo_source is not ThermoSource.SHIPPED:
        reaction_thermo = resolve_reaction_delta_g(
            gem, osp.join(DATA_ROOT, THERMO_EQUILIBRATOR_MIRROR), thermo_source
        )
    parameterization = arm.get("parameterization", "box")
    cfg = FluxLayerConfig(
        hidden_dim=hidden_dim,
        parameterization=parameterization,
        thermo_mode=arm["thermo_mode"],
        thermo_source=thermo_source,
        constraint_warmup_epochs=int(arm.get("constraint_warmup_epochs", 0)),
        constraint_final_scale=float(arm.get("constraint_final_scale", 1.0)),
        constraint_delay_epochs=int(arm.get("constraint_delay_epochs", 0)),
        use_enzyme_capacity=arm["use_enzyme_capacity"],
        use_protein_budget=arm["use_protein_budget"],
        use_dissipation_limit=arm["use_dissipation_limit"],
        use_abundance_head=bool(arm.get("use_abundance_head", False)),
        # Construction-time: the head's depth (028 round 19); 1 is the shipped linear map.
        abundance_head_layers=int(arm.get("abundance_head_layers", 1)),
        stochastic=bool(arm.get("stochastic", False)),
        # CONSTRUCTION-TIME SWITCHES 027's trainer writes into the arm dict before this
        # call (wave 8). `latent_stochastic` was assigned onto the built layer's config
        # until now, which works on 'projection' because its spread head is always built;
        # `log_c_from_latent` creates a module, so it has to be known here. No 026 arm
        # sets either, and both default to what every earlier build ran.
        latent_stochastic=bool(arm.get("latent_stochastic", False)),
        log_c_from_latent=bool(arm.get("log_c_from_latent", False)),
        # THE WARM START'S THREE CONSTRUCTION-TIME FIELDS. `FluxLayer.__init__` solves or
        # loads the reference and creates `warm_log_alpha` from these, so they have to be
        # on the config BEFORE the layer is built; an assignment afterwards is ignored.
        # All three default to the shipped values, so an arm that names none builds a
        # bit-identical layer, and a non-warmstart arm never reads them at all.
        warm_start_reference_table_path=arm.get("warm_start_reference_table_path"),
        warm_start_alpha_learnable=bool(arm.get("warm_start_alpha_learnable", True)),
        warm_start_init_rel=float(arm.get("warm_start_init_rel", 0.01)),
        # The physics terms are dimensionless but not automatically COMMENSURATE with the
        # data term. Measured at init on a real batch, the unweighted constraint sum is
        # ~255 against a data loss of ~2, so at weight 1 the model spends every parameter
        # on feasibility and none on the phenotype. These weights put the two within an
        # order of magnitude of each other at initialization.
        weights={
            "balance": 0.05,
            "thermo": 0.5,
            "capacity": 1.0,
            "budget": 0.5,
            "dissipation": 0.05,
            "parsimony": 1.0e-3,
            "thermo_prior": 1.0e-2,
            "box": 0.5,
        },
    )
    basis = None
    if parameterization == "nullspace":
        basis = null_space_basis(
            gem.s,
            cache_path=osp.join(
                DATA_ROOT, "data/torchcell/yeast-GEM/null_space_basis_9_0_2.npy"
            ),
        )
    layer = FluxLayer(
        gem,
        model_gene_ids,
        config=cfg,
        kcat_per_s=kcat.values,
        molecular_weight_kda=mw.values,
        null_space=basis,
        reaction_thermo=reaction_thermo,
        kcat_provenance=[str(p) for p in kcat.provenance],
        km_mm=None if km is None else km.values,
        km_provenance=None if km is None else [str(p) for p in km.provenance],
    )
    meta = {
        "coverage": layer.coverage_report(),
        # UNCHANGED BY THE PREDICTED TIER, deliberately. `experimental_coverage` counts
        # measurements only, so the preflight floor in 027's `run_cell` still fires when the
        # Open Enzyme Database mirror is missing, even though the predicted tier would fill
        # every unit it left empty.
        "kcat_experimental_fraction": float(kcat.experimental_coverage.fraction),
        "kcat_tier": kcat_tier,
        "complex_aggregation": complex_aggregation,
        "kcat_ceiling_sigma_k": kcat_ceiling_sigma_k,
        "kcat_tier_counts": kcat.tier_counts(),
        "kcat_notes": kcat.notes,
        "km_tier_counts": None if km is None else km.tier_counts(),
        "predicted_kcat_selection": (
            None if predicted_kcat is None else predicted_kcat.selection.model_dump()
        ),
        "predicted_km_selection": (
            None if predicted_km is None else predicted_km.selection.model_dump()
        ),
        "thermo_source": str(thermo_source),
        "thermo_route_counts": (
            None if reaction_thermo is None else reaction_thermo.counts
        ),
        "thermo_sha256": None if reaction_thermo is None else reaction_thermo.sha256,
        "mw_experimental_fraction": float(mw.experimental_coverage.fraction),
        "media": media_meta,
        "kcat_shuffle": kcat_shuffle_meta,
        "met_index": {m: i for i, m in enumerate(gem.met_ids)},
    }
    return layer, meta


def extract_targets(
    batch: Any, head_specs: dict[str, dict[str, Any]]
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Pull ``(target, mask)`` per head out of the COO phenotype encoding.

    THREE COLLATION FACTS, and getting any of them wrong yields a plausible number
    rather than an error. This decode follows the one validated in the 019 harness
    (``train_cgt_multitask.py:_extract_targets_and_masks``).

    1. **The batch row is ``phenotype_values_batch``, not ``phenotype_sample_indices``.**
       ``phenotype_sample_indices`` indexes the EXPERIMENT within a genotype group and is
       not offset across the batch, so every graph's values collide at 0, 1, 2. Using it as
       the row index writes a 64-strain batch's targets into the first three rows.
    2. **``phenotype_types`` collates to a list-of-lists**, one name list per genotype, and
       the type index is LOCAL to its graph. A value's name is
       ``phenotype_types[b][type_idx]``.
    3. **The dict keys are dropped, so two heads sharing a label are separated only by
       group SIZE.** Betaxanthin and the Mulleder metabolome are both a
       ``MetabolitePhenotype`` and both arrive as ``metabolite_level``. Within one
       genotype they are two experiments, hence two groups under
       ``phenotype_sample_indices``, of size 1 and 19. **Their order is not fixed** --
       measured over 1,200 records, the 1-value betaxanthin group comes first in only
       ~11 % of the co-measured genotypes -- so splitting a 20-value block by position is
       wrong roughly nine times in ten. Grouping and matching on width is what makes the
       assignment identity-based.

    THE LIMIT OF WIDTH AS AN IDENTITY, and the guard that makes it audible. Width separates
    two groups only while no two groups under one label have the SAME width. Two width-1
    ``metabolite_level`` groups on one genotype -- betaxanthin beside an isobutanol
    biosensor fold change, say -- both match a width-1 head and are AVERAGED by the
    ``torch.stack(matched).mean`` below, producing a plausible number from two different
    assays. A width-1 head that meets such a genotype is therefore refused rather than
    decoded, with the group sizes in the message.

    THE GUARD IS SCOPED TO WIDTH 1, and deliberately: averaging several groups of the SAME
    wider width is the intended behavior for genuine replicate experiments, and a head that
    does not read the colliding groups averages nothing. Measured 2026-09-14 over both
    builds: ``fig6_pigment_transfer`` carries group-size multisets (1, 19) on 4,432
    genotypes, (19,) on 246, (1,) on 237 and () on 15, so no 026 or 027 cell run to date
    meets the condition; ``fig6_build`` carries exactly ONE colliding genotype, record 3363,
    where two Cachera strains (the ARO4 and the ARO7 deletion, each inside the betaxanthin
    cassette) aggregate to one deletion key and contribute two width-1 betaxanthin values.
    A betaxanthin head on that build is refused, which is right; the isobutanol head reads a
    width-20 column there and is unaffected.
    """
    gene = batch["gene"]
    values = gene.phenotype_values
    type_idx = gene.phenotype_type_indices
    val_batch = gene.phenotype_values_batch
    samp_idx = gene.phenotype_sample_indices
    raw_types = gene["phenotype_types"]
    per_graph_types: list[list[str]] = (
        [list(raw_types)]
        if len(raw_types) > 0 and isinstance(raw_types[0], str)
        else [list(t) for t in raw_types]
    )
    batch_size = int(batch.num_graphs)

    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, spec in head_specs.items():
        label = spec["label"]
        width = int(spec["width"])
        target = values.new_zeros(batch_size, width)
        mask = torch.zeros(batch_size, width, dtype=torch.bool, device=values.device)
        for b in range(batch_size):
            sel_b = val_batch == b
            if not bool(sel_b.any()):
                continue
            names_b = per_graph_types[b]
            tb = type_idx[sel_b].tolist()
            keep = torch.tensor(
                [names_b[t] == label for t in tb],
                dtype=torch.bool,
                device=values.device,
            )
            if not bool(keep.any()):
                continue
            cand_vals = values[sel_b][keep]
            cand_samp = samp_idx[sel_b][keep]
            groups = [
                cand_vals[cand_samp == s]
                for s in cand_samp.unique(sorted=True).tolist()
            ]
            matched = [g for g in groups if g.numel() == width]
            if width == 1 and len(matched) > 1:
                raise AssertionError(
                    f"genotype {b} of this batch carries {len(matched)} width-1 {label!r} "
                    f"groups (group sizes {[int(g.numel()) for g in groups]}) and head "
                    f"{name!r} is width 1, so the two would be averaged into one target -- "
                    "a betaxanthin colony-color z-score with an isobutanol biosensor fold "
                    "change, say. Heads sharing a label are separated by group WIDTH "
                    "because the phenotype dict keys are dropped in collation, and width "
                    "cannot separate these. Build the dataset from a query carrying at "
                    "most one width-1 dataset for this label, or decode this head as a "
                    "column of a wider block."
                )
            if not matched:
                continue
            target[b] = torch.stack(matched, dim=0).mean(dim=0)
            mask[b] = True
        out[name] = (target, mask)
    return out


def masked_pearson(
    pred: np.ndarray, true: np.ndarray, mask: np.ndarray
) -> tuple[float, float]:
    """Per-feature Pearson over observed rows, plus the fraction of predictions that are
    not finite.

    **Non-finite predictions are excluded and COUNTED, not silently propagated.** The
    earlier version tested ``p.std() < 1e-12`` first, and a NaN fails that comparison, so a
    single NaN prediction turned the whole feature's correlation into NaN and the arm's
    metric into NaN with no indication of why. Worse, it was indistinguishable from a
    collapsed constant prediction, which the same branch reports as exactly 0.0: the
    flux_nullspace seed-1234 run alternated between ``nan`` and ``0.0000`` across epochs
    and neither value said which failure was happening.

    Returning the non-finite fraction alongside the score makes the distinction visible,
    which matters because "the model predicted a constant" and "the model produced NaN" are
    different defects with different fixes.

    Returns:
        ``(mean per-feature Pearson, fraction of observed predictions that are non-finite)``.
    """
    scores: list[float] = []
    n_obs = 0
    n_bad = 0
    for c in range(pred.shape[1]):
        sel = mask[:, c]
        if sel.sum() < 10:
            continue
        p, t = pred[sel, c], true[sel, c]
        finite = np.isfinite(p) & np.isfinite(t)
        n_obs += int(finite.size)
        n_bad += int((~finite).sum())
        p, t = p[finite], t[finite]
        # RELATIVE variance guard. An absolute floor of 1e-12 is the wrong test: a head
        # that has collapsed onto a constant does not emit a bitwise-identical value, it
        # emits one with a residual spread of order 1e-11, which clears an absolute floor
        # and then makes `corrcoef` divide by a number indistinguishable from zero. The
        # result is NaN, reported as if the metric were unavailable, when what actually
        # happened is the documented mean-collapse failure mode. Measured on the
        # flux_nullspace seed-1234 rerun: 8 of 20 epochs returned NaN and 3 returned
        # exactly 0.0, and both are the same event seen through different sides of the
        # floor. Scaling by the target's spread makes "collapsed" scale-free.
        if p.size < 10 or t.std() < 1e-12 or p.std() < 1e-8 * t.std():
            scores.append(0.0)
            continue
        score = float(np.corrcoef(p, t)[0, 1])
        scores.append(score if np.isfinite(score) else 0.0)
    return (
        float(np.mean(scores)) if scores else float("nan"),
        (n_bad / n_obs) if n_obs else 0.0,
    )


def run_arm(
    arm_name: str,
    seed: int,
    dataset: Neo4jCellDataset,
    args: argparse.Namespace,
    arm_override: dict[str, Any] | None = None,
    return_model: bool = False,
    permute_train_targets: bool = False,
    on_epoch: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train one arm at one seed and return its metrics and feasibility trace.

    Args:
        arm_name: Key into :data:`ARMS`, or a label when ``arm_override`` is given.
        seed: Split seed and weight-init seed, held identical across arms.
        dataset: The already-built pigment/metabolome dataset.
        args: Parsed hyperparameters.
        arm_override: An arm dict used instead of the registry entry, for one-off
            configurations such as the stochastic sampler.
        return_model: Include the trained model under ``"model"``. The sampling demo needs
            a FITTED posterior; sampling an untrained prior would make its width comparison
            against flux variability analysis meaningless.
        permute_train_targets: Permute each batch's training targets and their mask
            together, leaving validation untouched, which turns the run into a
            label-permutation null. An explicit parameter rather than an ``args``
            attribute, so a caller that builds its own argument namespace cannot fail on
            a missing field: ``flux_sampling_demo.py`` does exactly that.
        on_epoch: Called with each epoch's metric row as it is produced. This is what
            makes a run watchable while it is still running, and it takes the row rather
            than a metric name so a caller logs the feasibility diagnostics on the same
            step as the score. An unattended overnight sweep whose only output is a JSON
            file written after each 24-minute cell is not observable in time to act on.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    arm = arm_override if arm_override is not None else ARMS[arm_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gene_ids = list(dataset.cell_graph["gene"].node_ids)
    flux_layer, flux_meta = build_flux_layer(arm, gene_ids, args.hidden)

    aa_keys = sorted(AMINO_ACID_TO_GEM_METABOLITE)
    # Both phenotypes carry the label `metabolite_level`; `width` is what tells them apart.
    head_specs: dict[str, dict[str, Any]] = {
        "betaxanthin": {"label": "metabolite_level", "width": 1},
        "mulleder19": {"label": "metabolite_level", "width": len(aa_keys)},
    }

    heads_config: dict[str, Any] = {}
    if arm["flux"]:
        met_index = flux_meta["met_index"]
        heads_config["betaxanthin"] = {
            "kind": "flux_scalar",
            "output_dim": 1,
            "precursor_indices": [
                met_index[m] for m in BETAXANTHIN_PRECURSORS if m in met_index
            ],
        }
        heads_config["mulleder19"] = {
            "kind": "flux_metabolite",
            "output_dim": 19,
            "metabolite_indices": [
                met_index[AMINO_ACID_TO_GEM_METABOLITE[k]] for k in aa_keys
            ],
        }
    else:
        heads_config["betaxanthin"] = {"kind": "scalar", "output_dim": 1}
        heads_config["mulleder19"] = {"kind": "vector", "output_dim": 19}

    model = CellGraphTransformerMetabolism(
        cell_graph=dataset.cell_graph,
        gene_num=len(gene_ids),
        hidden_channels=args.hidden,
        num_transformer_layers=args.layers,
        num_attention_heads=4,
        dropout=0.1,
        heads_config=heads_config,
        learnable_embedding_config={
            "enabled": True,
            "size": args.hidden,
            "preprocessor": {"num_layers": 2, "dropout": 0.1},
        },
        flux_layer=flux_layer,
    ).to(device)

    dm = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(DATASET_ROOT, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=args.batch_size,
        random_seed=seed,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch=False,
        follow_batch=["perturbation_indices", "deletion_indices", "phenotype_values"],
    )
    dm.setup()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = nn.MSELoss(reduction="none")

    # Target standardization from the TRAIN split only: a metric computed against
    # test-informed statistics is not a held-out metric.
    stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    acc: dict[str, list[torch.Tensor]] = {k: [] for k in head_specs}
    for batch in dm.train_dataloader():
        for name, (t, m) in extract_targets(batch, head_specs).items():
            acc[name].append(torch.where(m, t, torch.nan))
    for name, chunks in acc.items():
        cat = torch.cat(chunks, dim=0)
        mean = torch.nanmean(cat, dim=0)
        centered = cat - mean
        var = torch.nanmean(centered * centered, dim=0)
        stats[name] = (mean.to(device), var.sqrt().clamp(min=1e-6).to(device))

    history: list[dict[str, Any]] = []
    best = {"val_betaxanthin": -np.inf, "val_mulleder19": -np.inf, "epoch": -1}
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        feas: dict[str, float] = {}
        for batch in dm.train_dataloader():
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, reps = model(dataset.cell_graph.to(device), batch)
            head_outputs = reps["head_outputs"]
            targets = extract_targets(batch, head_specs)
            loss = torch.zeros((), device=device)
            for name in head_specs:
                t, m = targets[name]
                t, m = t.to(device), m.to(device)
                if permute_train_targets:
                    # THE NULL. Permuting rows of the target AND its mask together
                    # destroys the genotype-to-phenotype association while leaving the
                    # target distribution, the missingness rate and every other moving
                    # part identical. Validation targets are NOT permuted, so
                    # `val_betaxanthin` is still scored against real labels and its
                    # maximum over epochs is a draw from the null distribution of the
                    # exact statistic the arms are reported with. Permuting target
                    # without mask would pair a value with another row's observed flag.
                    perm = torch.randperm(t.shape[0], device=device)
                    t, m = t[perm], m[perm]
                mean, sd = stats[name]
                z = (t - mean) / sd
                pred = head_outputs[name]
                if pred.dim() == 3:
                    pred = pred[..., 0]
                per = loss_fn(pred, z) * m
                denom = m.sum().clamp(min=1)
                loss = loss + per.sum() / denom
            loss = loss + reps["graph_reg_loss"]
            if "flux" in reps:
                flux = reps["flux"]
                loss = loss + model.flux_layer.constraint_loss(flux)
                for k, v in flux.items():
                    # Scalars only: `c_j` is [B, r] reaction availability, not a residual,
                    # and shares the `c_` prefix with the constraint terms.
                    if v.dim() == 0 and k.startswith(
                        ("c_", "feas_", "g_diss", "protein_used")
                    ):
                        feas[k] = feas.get(k, 0.0) + float(v.detach())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            train_loss += float(loss.detach())
            n_batches += 1

        model.eval()
        preds: dict[str, list[np.ndarray]] = {k: [] for k in head_specs}
        trues: dict[str, list[np.ndarray]] = {k: [] for k in head_specs}
        masks: dict[str, list[np.ndarray]] = {k: [] for k in head_specs}
        with torch.no_grad():
            for batch in dm.val_dataloader():
                batch = batch.to(device)
                _, reps = model(dataset.cell_graph.to(device), batch)
                targets = extract_targets(batch, head_specs)
                for name in head_specs:
                    t, m = targets[name]
                    mean, sd = stats[name]
                    pred = reps["head_outputs"][name]
                    if pred.dim() == 3:
                        pred = pred[..., 0]
                    preds[name].append((pred * sd + mean).cpu().numpy())
                    trues[name].append(t.cpu().numpy())
                    masks[name].append(m.cpu().numpy())

        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss / max(n_batches, 1),
        }
        for name in head_specs:
            mask_all = np.concatenate(masks[name])
            score, nonfinite = masked_pearson(
                np.concatenate(preds[name]), np.concatenate(trues[name]), mask_all
            )
            row[f"val_{name}"] = score
            row[f"nonfinite_{name}"] = nonfinite
            # The sample size the correlation was computed on. Without it a Pearson
            # cannot be compared against its own null width, which is 1/sqrt(n-3), and
            # `extract_targets` sets the whole row's mask together so column 0 is the
            # count of observed rows.
            row[f"n_val_{name}"] = int(mask_all[:, 0].sum())
        for k, v in feas.items():
            row[k] = v / max(n_batches, 1)
        history.append(row)
        if on_epoch is not None:
            on_epoch(row)
        # `nan > x` is False, so a NaN epoch can never become the best. That is the
        # intended behavior: an epoch whose metric could not be computed is not a result.
        if row["val_betaxanthin"] > best["val_betaxanthin"]:
            best = {
                "val_betaxanthin": row["val_betaxanthin"],
                "val_mulleder19": row["val_mulleder19"],
                "epoch": epoch,
            }
        # Print from `row`, which holds the per-batch MEANS; `feas` holds the running sums
        # and printing it reports a violation fraction of 23.8, which is not a fraction.
        print(
            f"[{arm_name} s{seed}] ep {epoch:3d} loss {row['train_loss']:.4f} "
            f"bx {row['val_betaxanthin']:+.4f} aa {row['val_mulleder19']:+.4f} "
            + " ".join(
                f"{k.removeprefix('feas_')} {v:.3g}"
                for k, v in row.items()
                if k.startswith("feas_")
            ),
            flush=True,
        )

    out: dict[str, Any] = {
        "arm": arm_name,
        "seed": seed,
        "best": best,
        "final": history[-1] if history else {},
        "history": history,
        "flux_meta": {k: v for k, v in flux_meta.items() if k != "met_index"},
        "n_parameters": model.num_parameters,
        "wall_time_s": time.time() - t0,
    }
    if return_model:
        out["model"] = model
    return out


def main() -> None:
    """Run the requested arms across the requested seeds, checkpointing after each."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="pooled,flux_off,flux_free,flux_anchored")
    parser.add_argument("--seeds", default="42,7,1234")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--out", default="flux_arms.json")
    parser.add_argument(
        "--permute-train-targets",
        action="store_true",
        help=(
            "Permute training targets within each batch, leaving validation untouched. "
            "Turns the run into a label-permutation null for the reported statistic."
        ),
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    dataset = build_dataset()
    print(f"dataset: {len(dataset)} aggregated genotypes", flush=True)

    out_path = osp.join(RESULTS_DIR, args.out)
    runs: list[dict[str, Any]] = []
    # SEED-MAJOR, and the ordering is load-bearing under a deadline. Arm-major finishes
    # every seed of arm 1 before starting arm 2, so an interrupted sweep yields complete
    # data for some arms and none for others, which supports no comparison at all.
    # Seed-major finishes one seed of EVERY arm first, so any prefix of the run is a
    # usable, balanced experiment and each further pass just adds a replicate.
    for seed in [int(s) for s in args.seeds.split(",")]:
        for arm_name in args.arms.split(","):
            runs.append(
                run_arm(
                    arm_name,
                    seed,
                    dataset,
                    args,
                    permute_train_targets=args.permute_train_targets,
                )
            )
            with open(out_path, "w") as f:
                json.dump({"args": vars(args), "runs": runs}, f, indent=2, default=str)
            print(f"checkpointed {len(runs)} runs -> {out_path}", flush=True)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
