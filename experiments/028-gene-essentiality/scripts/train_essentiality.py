# experiments/028-gene-essentiality/scripts/train_essentiality.py
# [[experiments.028-gene-essentiality.scripts.train_essentiality]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/train_essentiality.py

r"""Gene-essentiality training cell: the CGT backbone with or without the metabolic module.

MOVED FROM 027 (2026-09-16). The essentiality line began as wave 9 of
``experiments/027-betaxanthin-metabolic`` (``train_bx.py``, ``--head essentiality``) and
is its own experiment from here on: its own W&B project (``torchcell_028_gene_essentiality``),
dataset root, checkpoint and dump directories, and notes. This file is that trainer with the
paths repointed and the head fixed to ``essentiality``; the betaxanthin, amino-acid and
isobutanol code paths it inherited are inert and are pruned as the line settles. The wave-9
to wave-11 result files were copied across unchanged so the record is continuous.

WHAT THIS DECIDES
-----------------
Whether a cell-graph transformer over the perturbed gene graph ranks essential genes above
nonessential ones better than (a) the published FBA comparator and (b) Flux Cone Learning
(Merzbacher 2025), on the released yeast split, and whether routing the prediction THROUGH
the enzyme-constrained thermodynamic flux layer (the biomass column of its flux vector) helps
or costs, and where the gradient goes when it does.

THE OBJECTIVE IS A BINARY CROSS ENTROPY ON ONE LOGIT. There is no fitness objective anywhere
in this cell: the union build carries single-mutant fitness for 4,540 genotypes, and every one
of them arrives with the essentiality mask False and contributes no loss. The label comes from
the released FCL split (inversion applied once, class counts asserted), the epoch is selected
on validation AUROC under a centered 5-epoch moving average, and the number reported is the
AUROC on the pinned held-out genes at that epoch, with per-class recall beside the majority
rate.

THE PROTOCOL, inherited from 027 and kept
------------------------------------------
1. **One head.** Nothing co-trains with the label.
2. **The split is pinned; the seed varies weight initialization only.** ``split_seed`` fixes
   the train/validation draw for every arm and seed; the 223 released test genes are forced
   into TEST, and 195 of them resolve in the build.
3. **Selection and reporting use DISJOINT data.** Validation picks the epoch; the pinned test
   genes are scored once at it. A max over epochs of the validation curve is never reported.

Run one cell by hand from the repo root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/028-gene-essentiality/scripts/train_essentiality.py \
        --arm pooled --head essentiality --seed 0 --epochs 2 --smoke
"""

import argparse
import csv
import hashlib
import json
import math
import os
import os.path as osp
import socket
import sys
import time
import uuid
from collections import deque
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import pearsonr, spearmanr

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
EXP_DIR = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality")
CONF_DIR = osp.join(EXP_DIR, "conf")
RESULTS_DIR = osp.join(EXP_DIR, "results")
#: Per-gene test predictions land beside 020's, so the existing FCL head-to-head scripts
#: (`plot_merzbacher_comparison.py`, `evaluate_merzbacher_head_to_head.py`) read them without
#: modification. Re-deriving the FCL labels here would fork their provenance.
TEST_PRED_DIR = os.getenv(
    "TC_TEST_PRED_DIR", osp.join(DATA_ROOT, "test-predictions-028")
)

# 026 owns the flux layer's construction; this experiment owns only the protocol.
#
# `TC_FLUX_SCRIPTS_DIR` exists for one situation and it is worth naming: 027 was authored in a
# worktree whose branch predates the 026 landing, so `$EXPERIMENT_ROOT/026-metabolism-flux/`
# there has no `train_flux.py` and `torchcell/metabolism/` has no `flux_layer.py`. The
# override lets a smoke test run against the primary checkout. On Delta and after the rebase
# it is unset and the default path is the right one.
FLUX_SCRIPTS_DIR = os.getenv(
    "TC_FLUX_SCRIPTS_DIR", osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "scripts")
)
assert osp.isfile(osp.join(FLUX_SCRIPTS_DIR, "train_flux.py")), (
    f"026's train_flux.py is not at {FLUX_SCRIPTS_DIR}. This branch predates the 026 "
    "landing; rebase onto main, or set TC_FLUX_SCRIPTS_DIR for a local smoke test."
)
sys.path.insert(0, FLUX_SCRIPTS_DIR)
from train_flux import (  # noqa: E402
    AMINO_ACID_TO_GEM_METABOLITE,
    ARMS,
    BETAXANTHIN_PRECURSORS,
    build_flux_layer,
    extract_targets,
)

# 026's `build_dataset` is deliberately NOT imported. It opens the dataset with two graphs
# and no node embeddings, which is the configuration every weak betaxanthin run used
# (0.04-0.16 val Pearson). The 0.43 runs used nine graphs and prot_T5. See README.md,
# "Design review before launch".
from torchcell.data import (  # noqa: E402
    DeletionKeyedGenotypeAggregator,
    MeanExperimentDeduplicator,
)
from torchcell.data.graph_processor import Perturbation  # noqa: E402
from torchcell.data.neo4j_cell import Neo4jCellDataset  # noqa: E402
from torchcell.datamodules import CellDataModule  # noqa: E402
from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder  # noqa: E402
from torchcell.graph import SCerevisiaeGraph, build_gene_multigraph  # noqa: E402
from torchcell.losses.distributional import (  # noqa: E402
    DistHead,
    coverage,
    dist_param_dim,
    gaussian_crps,
    make_dist_head,
    pinball,
    pit_ks,
    pit_values,
)
from torchcell.metabolism.constraints import (  # noqa: E402
    CONC_LOWER_M,
    CONC_UPPER_M,
    ThermoMode,
)
from torchcell.metabolism.parameters import (  # noqa: E402
    ThermoSource,
    yeast_compartment_volume_fractions,
)
from torchcell.models.cell_graph_transformer_metabolism import (  # noqa: E402
    CellGraphTransformerMetabolism,
    gene_removal_indices,
    perturbed_gene_pool,
)
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402

HEAD = "betaxanthin"

#: WAVE 9 (2026-09-14). THE ISOBUTANOL BIOSENSOR (Montano Lopez 2024), the third product
#: target and the only one whose measured quantity is a single NAMED yeast-GEM metabolite
#: pool. The dissertation states the ligand verbatim (mirrored `paper.md` line 425): "This
#: biosensor works by expressing green fluorescent protein (GFP) proportionally to the
#: amount α- isopropylmalate (α-IPM) in the cell, which is a precursor of leucine and a
#: byproduct in the isobutanol pathway ... As more α-IPM accumulates, it binds to Leu3p,
#: both of which activate transcription of GFP which is under control of the Leu1
#: promoter." So the reporter reads the 2-isopropylmalate POOL, which yeast-GEM 9.0.2
#: carries as `s_0162` [c] and `s_0164` [m], and the product isobutanol has a native
#: secretion exchange `r_1866`: this target can be read as a pool AND as a product flux,
#: which neither betaxanthin (a product with no pool) nor the amino acids (pools with no
#: product) can do.
ISOBUTANOL_HEAD = "isobutanol"

#: WAVE 9 (2026-09-14). GENE ESSENTIALITY, the endpoint where the published methods are far
#: enough apart to decide something. The amino-acid target's ceiling is about 0.17 Pearson
#: and every arm sits at 0.05 to 0.07, which is why it cannot; on YEAST essentiality the
#: mirrored Merzbacher 2025 Supplementary Table S5 puts FBA at ROC AUC 0.69 and Flux Cone
#: Learning at 0.73 on the released test split (0.78 +/- 0.065 across its 5 CV folds).
ESSENTIALITY_HEAD = "essentiality"
#: The phenotype label `GeneEssentialityPhenotype` emits (schema.py: `label_name` default).
ESSENTIALITY_LABEL = "is_essential"
#: yeast-GEM 9.0.2's objective column, verified on the model 2026-09-14: `r_2111` ("growth"),
#: reaction index 1499, and `build_gem_tensors` resolves `gem.biomass_index` to exactly it.
#: `r_4041` is the biomass PSEUDOREACTION and is not the objective.
BIOMASS_REACTION = "r_2111"

#: The heads whose phenotype is ONE number per genotype, so the scalar scoring, dumping and
#: reporting branches take them together. `mulleder19` is the vector head.
SCALAR_HEADS = (HEAD, ISOBUTANOL_HEAD, ESSENTIALITY_HEAD)
#: The heads whose target is a CLASS rather than a magnitude. They are scored with a
#: log-loss and selected on AUROC, and their target is never standardized: subtracting a
#: mean from a 0/1 label and dividing by its sd would make the log-loss meaningless.
BINARY_HEADS = (ESSENTIALITY_HEAD,)

#: HOW A FLUX ARM READS ESSENTIALITY (028, 2026-09-16). `biomass` is the wave-9 readout: the
#: logit is an affine of the layer's growth column alone, so every gradient reaching the
#: backbone passes through the flux layer. `context_biomass` keeps the pooled arm's own head
#: on the perturbed context AND adds the growth affine, so the backbone receives both the
#: direct gradient and the one routed through the module; the per-module gradient norms
#: then say how much of the update the module carries when it is not the only path.
ESS_READOUT_BIOMASS = "biomass"
ESS_READOUT_CONTEXT_BIOMASS = "context_biomass"
#: `gated` is `context_biomass` with the growth affine behind a ZERO-initialized gate: at
#: initialization the cell is the pooled model exactly, so the gate's trajectory (logged per
#: epoch as `ess_gate`) says whether the optimizer chose the flux path, and the per-module
#: gradient report says whether gradient reached the layer, which at gate 0 it cannot.
ESS_READOUT_GATED = "gated"
#: `fba` is a POOLED-arm readout (round 5, 2026-09-16): the context head's logit plus an
#: affine of FBA's four knockout-to-wild-type growth ratios for the deleted gene, one per
#: medium, read from `fba_essentiality_per_gene.csv` (the file the classical ladder appends
#: to every encoding). The ladder's rungs cross 0.80 only with those four numbers appended,
#: so this is the transformer given the same complement; the four weights are logged per
#: epoch as `ess_fba_w_<medium>`.
ESS_READOUT_FBA = "fba"
#: Round 6: the same FBA affine beside a FLUX arm's growth column, so the learned layer is
#: tested on top of the solve rather than instead of it. `biomass_fba` is the forced growth
#: column plus the affine (no context head); `gated_fba` is the gated readout plus the
#: affine (context head, gated growth column, FBA ratios).
ESS_READOUT_BIOMASS_FBA = "biomass_fba"
ESS_READOUT_GATED_FBA = "gated_fba"
#: The gated readout plus a linear head over the solved FLUX VECTOR (round 21): the
#: solve's flux carries more of the ranking than its growth column on the ladder, and
#: the dual-overlap probe says the growth column cannot be moved between genotypes
#: through the ceilings, so this reads what the solve returns rather than one number.
ESS_READOUT_GATED_FLUX = "gated_flux"
ESS_READOUTS = (
    ESS_READOUT_BIOMASS,
    ESS_READOUT_CONTEXT_BIOMASS,
    ESS_READOUT_GATED,
    ESS_READOUT_FBA,
    ESS_READOUT_BIOMASS_FBA,
    ESS_READOUT_GATED_FBA,
    ESS_READOUT_GATED_FLUX,
)
#: The readouts that keep the pooled arm's own context head beside the growth affine.
ESS_READOUTS_WITH_CONTEXT = (
    ESS_READOUT_CONTEXT_BIOMASS,
    ESS_READOUT_GATED,
    ESS_READOUT_GATED_FBA,
    ESS_READOUT_GATED_FLUX,
)
#: Which reactions the flux head of `gated_flux` reads.
ESS_FLUX_HEAD_REACTIONS = ("all", "exchange", "gene")
#: The readouts that carry the FBA affine, on either arm.
ESS_READOUTS_WITH_FBA = (
    ESS_READOUT_FBA,
    ESS_READOUT_BIOMASS_FBA,
    ESS_READOUT_GATED_FBA,
)
#: The per-gene FBA table (`fba_essentiality_baseline.py`) and its medium columns, in the
#: order the ladder uses them.
FBA_PER_GENE_CSV = osp.join(RESULTS_DIR, "fba_essentiality_per_gene.csv")
FBA_MEDIA = ("shipped", "SM", "SC", "YPD")
#: Per-gene solve-derived features from `ess_reference_flux_table.py`, appended to the
#: affine's table under `--ess-solve-features`: the eight statistics of the wild-type pFBA
#: solve and its variability on the gene's own reactions (`stats`), the deletion's own
#: pFBA flux vector (`pfba`, 4,131 columns), or both. Each column is standardized over the
#: genes the file covers; a gene it does not cover reads the column mean (0).
#: `ESS_SOLVE_TABLE` names another table, such as the same program solved by another
#: solver (`ess_reference_flux_table.py --solver ipm`, round 36); the path is recorded in
#: the result's `flux_meta.essentiality.fba.solve.source`.
SOLVE_ENCODINGS_NPZ = os.environ.get(
    "ESS_SOLVE_TABLE",
    osp.join(DATA_ROOT, "data/torchcell/reference_flux/ess_solve_encodings_YPD.npz"),
)
ESS_SOLVE_FEATURES = ("none", "stats", "pfba", "stats_pfba")
#: How the pFBA vector enters the affine (round 33). 'standardized' is round 31: each of
#: the 4,131 columns to mean 0 and sd 1 over the covered genes. 'log1p' is the flux
#: head's own encoding (round 21 onward): log(1 + |v|) with v in mmol/gDW/h, unscaled,
#: so a zero flux reads 0 and the affine sees what GatedFluxReadout sees. The
#: statistics block stays standardized under both.
ESS_SOLVE_ENCODINGS = ("standardized", "log1p")

#: Round 32 (2026-09-18). The fitting transition of the flux-head cells is a chance event
#: of the RUN and not of the seed (round 29: the same seed and program fit at epoch 70 in
#: one run and never in another, the two diverging from the first epoch, with torch and
#: numpy seeded and no loader workers). `ESS_DETERMINISTIC=1` pins what a seed leaves
#: free on the card: cuDNN's autotuner (`benchmark`), its nondeterministic kernels, and
#: every torch operation that has a deterministic implementation (`warn_only=True`, so an
#: operation with none warns once and the run continues; the pack sets PYTHONWARNINGS so
#: the warning reaches the log and names the kernel). cuBLAS needs
#: `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the environment before CUDA initializes, which is
#: the pack line's job, not this script's. Recorded in every result and dump as
#: `deterministic`.
DETERMINISTIC = os.environ.get("ESS_DETERMINISTIC") == "1"

#: WAVE 7 (2026-09-11). The two mechanistic amino-acid readouts, named so a result file says
#: which one it ran rather than leaving it to be inferred from the arm.
AA_READOUT_TURNOVER = "flux_metabolite"
AA_READOUT_CONCENTRATION = "flux_concentration"
#: WAVE 8 (2026-09-14). THE ABLATION THAT ASKS WHETHER THE MODULE IS ON. The same
#: concentration head as `flux_concentration` (per-metabolite base + Linear(d, m) from the
#: genotype context into the same sigmoid window, the same volume-weighted mixture, the same
#: standardization and CRPS) built directly on the pooled backbone, with NO flux layer: no
#: stoichiometry, no kcat, no thermodynamics, no constraint term. `thermo_log_c` is a
#: function of the context alone ([[experiments.027-betaxanthin-metabolic.wave8-sampling]],
#: Finding 5), so the flux enters the concentration cells only through the thermodynamic
#: penalty; this readout removes that coupling and keeps everything else.
AA_READOUT_CONTEXT_CONCENTRATION = "context_concentration"
AA_READOUTS = (
    AA_READOUT_TURNOVER,
    AA_READOUT_CONCENTRATION,
    AA_READOUT_CONTEXT_CONCENTRATION,
)
#: The readouts whose prediction IS a log10 molar concentration with no affine on the
#: mean, so their target is standardized and scored in that space.
AA_LOG10_MOLAR_READOUTS = (AA_READOUT_CONCENTRATION, AA_READOUT_CONTEXT_CONCENTRATION)
#: Where `collect_concentration` reads the learned natural-log concentration from.
CONC_SOURCE_FLUX = "flux"
CONC_SOURCE_CONTEXT = "context"
#: Mulleder 2016 reports `intracellular_concentration_mM`, so a stored 1.0 is 1 mM.
MM_PER_M = 1.0e3
LN10 = math.log(10.0)

#: How the betaxanthin scalar is read off the layer.
BX_READOUT_FLUX_SCALAR = "flux_scalar"
BX_READOUT_PRECURSOR_CONCENTRATION = "flux_precursor_concentration"
#: WAVE 9. THE BETAXANTHIN IS-THE-MODULE-ON CONTROL, the exact analogue of
#: `context_concentration` for the precursor readout: the same five-precursor whole-cell
#: mixture and the same affine, computed from a concentration the POOLED backbone
#: parameterizes, with no flux layer. The established wave-8 result is that the precursor
#: readout beats 026's scalar flux head by +0.0399 (sd 0.0123, n = 3); this arm says
#: whether that is aromatic-precursor physics or a five-dimensional bottleneck on the
#: genotype context.
BX_READOUT_CONTEXT_PRECURSOR_CONCENTRATION = "context_precursor_concentration"
BX_READOUTS = (
    BX_READOUT_FLUX_SCALAR,
    BX_READOUT_PRECURSOR_CONCENTRATION,
    BX_READOUT_CONTEXT_PRECURSOR_CONCENTRATION,
)

#: How the isobutanol scalar is read. The three arms of the wave-9 three-way; `pooled` is
#: implied by `--arm pooled`, which builds no layer and falls to the ordinary scalar head.
IB_READOUT_FLUX_SCALAR = "flux_scalar"
IB_READOUT_PRECURSOR_CONCENTRATION = "flux_precursor_concentration"
IB_READOUTS = (IB_READOUT_FLUX_SCALAR, IB_READOUT_PRECURSOR_CONCENTRATION)
#: Which anchors the isobutanol precursor readout reads. `one` is the PRIMARY arm and is
#: literally what the biosensor binds; `three` adds the two pools upstream and downstream
#: of it on the valine branch, and is the secondary.
IB_ANCHORS_ONE = "one"
IB_ANCHORS_THREE = "three"
IB_ANCHOR_SETS = (IB_ANCHORS_ONE, IB_ANCHORS_THREE)

#: Whether the dataset's growth medium is written onto the GEM's exchange bounds.
MEDIA_NONE = "none"
MEDIA_DATASET = "dataset"
APPLY_MEDIA_CHOICES = (MEDIA_NONE, MEDIA_DATASET)

#: Distributional losses this trainer can score a head with. 'crps' is the Gaussian CRPS
#: every 027 cell before 2026-09-11 ran and stays the default. 'quantile' is the 019
#: pinball loss over the K = 19 tau grid, which 019 measured as its best distributional
#: loss on expression; the point estimate is then the predictive MEDIAN.
DISTS = ("crps", "quantile")
#: Central intervals the calibration block reports coverage for.
COVERAGE_ALPHAS = (0.5, 0.8, 0.9)

#: How the 19 measured amino acids are read off the layer's per-compartment concentration.
CONC_SPACE_MIXTURE = "mixture"
CONC_SPACE_CYTOSOL = "cytosol"
CONC_SPACES = (CONC_SPACE_MIXTURE, CONC_SPACE_CYTOSOL)
#: Which fractional-volume table weights the whole-cell average.
VOLUMES_ASSUMED = "assumed"
VOLUMES_SOURCED = "sourced"
VOLUME_TABLES = (VOLUMES_ASSUMED, VOLUMES_SOURCED)

#: yeast-GEM's extracellular compartment is not part of the cell, and the membrane
#: pseudo-compartments have no aqueous volume, so neither enters a whole-cell average.
NON_CELL_COMPARTMENTS = frozenset({"e", "ce", "erm", "vm", "gm", "mm", "lp"})

#: Selected-epoch trainable parameters land here, one file per cell, so a finished run can
#: be re-opened and re-audited. Buffers are excluded (see `_params_snapshot`), which is what
#: keeps the file a few MB rather than ~50 MB of constant stoichiometric matrix.
CHECKPOINT_DIR = os.getenv("TC_CHECKPOINT_DIR", osp.join(DATA_ROOT, "checkpoints-028"))

# 026's module globals came in with its `build_dataset`; this file defines its own, from
# the same config fields the arms already declare.
DATASET_ROOT = osp.join(
    DATA_ROOT, "data/torchcell/experiments/019-simb-multimodal/fig6_pigment_transfer"
)
QUERY_PATH = osp.join(
    os.environ["EXPERIMENT_ROOT"],
    "019-simb-multimodal/queries/fig6_pigment_transfer.cql",
)

#: WAVE 9. THE ISOBUTANOL BUILD, and it is a DIFFERENT on-disk dataset. The biosensor
#: screen is not in `fig6_pigment_transfer`: `generate_fig6_pigment_transfer_cql.py` drops
#: it by name. It IS in `fig6_build` (13,948 aggregated genotypes, built 2026-07-22), which
#: is the seven-dataset union. Nothing is rebuilt; both roots are opened read-only and the
#: dataset class skips `process()` when `processed/` exists.
ISOBUTANOL_DATASET_ROOT = osp.join(
    DATA_ROOT, "data/torchcell/experiments/019-simb-multimodal/fig6_build"
)
ISOBUTANOL_QUERY_PATH = osp.join(
    os.environ["EXPERIMENT_ROOT"], "019-simb-multimodal/queries/fig6_build.cql"
)


#: WAVE 9. THE ESSENTIALITY BUILD, a dedicated root because no existing one carries the
#: label: both builds holding `GeneEssentialitySgdDataset` converted `is_essential` into
#: `fitness` (measured 2026-09-14). Built by `gh_cpu_build_essentiality.slurm` from
#: `queries/essentiality_build.cql` with `converter=None` and single-deletion genotypes
#: only. It is NOT under `019-simb-multimodal/` because it is not a 019 figure build.
ESSENTIALITY_DATASET_ROOT = osp.join(
    DATA_ROOT, "data/torchcell/experiments/028-gene-essentiality/essentiality_build"
)
ESSENTIALITY_QUERY_PATH = osp.join(
    os.environ["EXPERIMENT_ROOT"],
    "028-gene-essentiality/queries/essentiality_build.cql",
)


def dataset_paths(head: str = HEAD) -> tuple[str, str]:
    """``(root, query path)`` for the build a head's phenotype lives in.

    Keyed off the HEAD rather than off a flag, because the phenotype decides which build
    contains it and a cell that named the wrong one would train on a dataset that does not
    carry its target at all. Every betaxanthin and amino-acid cell keeps the wave-1 to
    wave-8 pair unchanged.
    """
    if head == ISOBUTANOL_HEAD:
        return ISOBUTANOL_DATASET_ROOT, ISOBUTANOL_QUERY_PATH
    if head == ESSENTIALITY_HEAD:
        return ESSENTIALITY_DATASET_ROOT, ESSENTIALITY_QUERY_PATH
    return DATASET_ROOT, QUERY_PATH


# The nine graphs and the protein embedding that separate the 0.43 runs from the 0.13 ones.
# Copied from experiments/019-simb-multimodal/conf/gh_metabolism_000.yaml rather than
# retyped, so the backbone is the one that was actually measured.
STRONG_GRAPHS: list[str] = [
    "physical",
    "regulatory",
    "tflink",
    "string12_0_neighborhood",
    "string12_0_fusion",
    "string12_0_cooccurence",
    "string12_0_coexpression",
    "string12_0_experimental",
    "string12_0_database",
]
STRONG_NODE_EMBEDDINGS: tuple[str, ...] = ("prot_T5_all",)
#: The embeddings the run's dataset is opened with; `--node-embeddings` replaces the
#: default (round 6: codon frequency, and the species-LM composite the expression line
#: uses). The names are `NodeEmbeddingBuilder`'s; the preprocessor projects their
#: concatenation, so the input width follows the choice.
NODE_EMBEDDING_NAMES: list[str] = list(STRONG_NODE_EMBEDDINGS)

# The model infers its input width by indexing the embedding dataset, so it needs the same
# objects the dataset was opened with, not just their names. Built once and held here
# because `build_dataset` owns the genome and graph they are constructed from.
_NODE_EMBEDDINGS: dict[str, Any] = {}


def build_dataset(head: str = HEAD) -> "Neo4jCellDataset":
    """Open the head's dataset with the nine-graph, prot_T5 backbone. Never rebuilds.

    The default is the pigment build every wave-1 to wave-8 cell and every audit script
    opens; ``--head isobutanol`` opens ``fig6_build`` instead (see :func:`dataset_paths`).
    """
    root, query_path = dataset_paths(head)
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        # NOT the default. `overwrite` defaults to True, which calls
        # `gffutils.create_db(force=True)` and so UNLINKS and rebuilds the 15 MB
        # `data/sgd/genome/data.db` on every construction. With 8 workers per node all
        # building the genome at once, seven of them open the file inside another's rebuild
        # window and die on `FileNotFoundError: .../data.db` -- which reads as a missing
        # file and is in fact a write race. This killed 7 of 8 workers on Delta node
        # gpub071 (21813442_0) and every worker of three tasks in array 21797666. The
        # database is a build artifact and is never edited, so nothing here should rebuild
        # it; the older experiments all pass overwrite=False for the same reason.
        overwrite=False,
    )
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )
    _NODE_EMBEDDINGS.update(
        NodeEmbeddingBuilder.build(
            embedding_names=list(NODE_EMBEDDING_NAMES),
            data_root=DATA_ROOT,
            genome=genome,
            graph=graph,
        )
    )
    with open(query_path) as handle:
        query = handle.read()
    return Neo4jCellDataset(
        root=root,
        query=query,
        gene_set=genome.gene_set,
        graphs=build_gene_multigraph(graph=graph, graph_names=STRONG_GRAPHS),
        incidence_graphs=None,
        node_embeddings=_NODE_EMBEDDINGS,
        converter=None,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=DeletionKeyedGenotypeAggregator,
        graph_processor=Perturbation(),
    )


class ScoringConfig(BaseModel):
    """The scoring rule, fixed before the first launch."""

    model_config = ConfigDict(extra="forbid")

    primary: str
    select_metric: str
    select_smoothing_window: int
    aggregate: str


class BaselineConfig(BaseModel):
    """The published numbers the decision is made against, read from 020's results JSON."""

    model_config = ConfigDict(extra="forbid")

    fcl_rf_test_spearman: float
    fcl_rf_test_auc_high: float
    cgt_020_v4_test_spearman_mean: float
    cgt_020_v4_test_spearman_sd: float


class DataConfig(BaseModel):
    """Dataset location, the external test split to pin, and the partition seed."""

    model_config = ConfigDict(extra="forbid")

    dataset_subdir: str
    query_path: str
    pinned_test_split_file: str
    split_seed: int


class ModelConfig(BaseModel):
    """The 026 incumbent hyperparameters, identical across all six arms."""

    model_config = ConfigDict(extra="forbid")

    hidden: int
    layers: int
    attention_heads: int
    batch_size: int
    lr: float
    weight_decay: float


class TrainConfig(BaseModel):
    """Epoch ceiling and the two knobs the launcher may override."""

    model_config = ConfigDict(extra="forbid")

    epochs: int
    early_stopping: bool
    num_workers: int


class FluxConfig(BaseModel):
    """The kcat-coverage floor that stops a flux arm running on default parameters."""

    model_config = ConfigDict(extra="forbid")

    min_kcat_experimental_fraction: float


class BxConfig(BaseModel):
    """``conf/base.yaml``, validated.

    ``extra="forbid"`` throughout, so a typo in a YAML key is an error at load rather than a
    silently ignored hyperparameter -- the failure mode that makes two arms differ by
    something nobody chose.
    """

    model_config = ConfigDict(extra="forbid")

    scoring: ScoringConfig
    baselines: BaselineConfig
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    flux: FluxConfig


class ArmConfig(BaseModel):
    """``conf/arm_<name>.yaml``: what an arm file is allowed to change.

    The flux CHANNEL is here because a parameterization is a property of the arm rather than
    of the run, and because 026's ARMS registry stays the single source of truth for the
    physics switches: an 027 arm names a base arm for the switches and, optionally, the
    channel string the layer parses.
    """

    model_config = ConfigDict(extra="forbid")

    base_arm: str
    permute_train_targets: bool
    flux_parameterization: str | None = None
    flux_balance_active_row_set: str = "dynamic"


class BxCell(BaseModel):
    """One training run, fully specified. Serialized into the results file verbatim."""

    model_config = ConfigDict(extra="forbid")

    arm: str
    base_arm: str
    seed: int
    permute_train_targets: bool
    epochs: int
    hidden: int
    layers: int
    attention_heads: int
    batch_size: int
    num_workers: int
    lr: float
    weight_decay: float
    split_seed: int
    select_smoothing_window: int = Field(ge=1)
    split_file: str | None = Field(
        default=None,
        description=(
            "A split file under results/splits/ (build_essentiality_splits.py): the "
            "per-gene labels and the pinned train/val/test record indices. When set, the "
            "labels come from it rather than from the released CSV, the data module pins "
            "all three splits and DROPS every record the file does not name, and the "
            "wave-9 seeded 80/10/10 draw is not made. None reproduces the wave-9 to "
            "wave-11 protocol exactly."
        ),
    )
    preprocessor_layers: int = Field(
        default=2,
        ge=1,
        description=(
            "Depth of the MLP that maps the concatenated gene embedding onto the "
            "transformer width. THIS IS WHERE THE PARAMETERS ARE: at two layers the first "
            "Linear is input_dim by (input_dim + hidden)/2, which does not shrink when the "
            "width does, so the preprocessor is 65 percent of the 90-wide model and 97 "
            "percent of the 24-wide one. At one layer it is a single input_dim by hidden "
            "projection, so the width finally controls the size."
        ),
    )
    # RECIPE SWITCHES, added 2026-09-09 after the collapse audit. Both default to what the
    # 144-cell grid ran, so a grid cell deserializes unchanged; both are ON in the 019 run
    # that reached 0.434 (job 1344), and the grid had dropped them without recording it.
    crps: bool = Field(
        default=False,
        description=(
            "Score the scalar head as a Gaussian CRPS with a (mu, log_sigma) output. The "
            "grid's loop sliced log_sigma away before the loss and fell back to MSE."
        ),
    )
    graph_reg: bool = Field(
        default=False,
        description=(
            "Pass 019's graph-structure regularizer (lambda 6.5e-5, one attention head per "
            "graph) to the model. The grid never passed it, so graph_reg_loss was 0."
        ),
    )
    abort_on_collapse: bool = Field(
        default=False,
        description=(
            "Stop the run when the validation prediction spread stays under 1 percent of "
            "the target spread past epoch 10, which is the collapse the grid ran 200 epochs "
            "of without noticing."
        ),
    )
    aux_mulleder: bool = Field(
        default=False,
        description=(
            "Train the 19-column Mulleder amino-acid head jointly, as the 019 run that "
            "reached 0.434 did (active_heads: [betaxanthin, mulleder19]). The 027 protocol "
            "dropped it to make the comparison betaxanthin-only, and the four recipe "
            "diagnostics of 2026-09-09 collapsed with and without CRPS and the graph "
            "regularizer, which leaves the co-head as the remaining recipe difference. "
            "Selection and scoring stay betaxanthin-only."
        ),
    )

    attention_mask: bool = Field(
        default=False,
        description=(
            "Hard graph masking: burn each of the nine relations into one attention head "
            "at layer 1, as the 020 Delta grid (delta_grid_betaxanthin.yaml) did. The 019 "
            "KL regularizer was replaced by this after the July runs."
        ),
    )
    learnable_embedding: bool = Field(
        default=False,
        description=(
            "MEMORIZATION CONTROL. A free per-gene vector lets the model fit the training "
            "targets by lookup (019 measured train r 0.86 with it), so a run that cannot "
            "fit even with it has a dead signal path, not weak features."
        ),
    )
    pert_heads: int = Field(
        default=6,
        description=(
            "Cross-attention heads of the perturbation TRANSFORM (the readout head that "
            "shares the config key is not built in 027). 6 is what 019's gh_metabolism_000 "
            "and the 027 grid used; 9 is what the 020 grid that trained used. Must divide "
            "hidden."
        ),
    )
    # WAVE 5 (2026-09-10). Three knobs on the module itself, all defaulting to what waves 1
    # to 4 ran, so an earlier result file deserializes unchanged.
    kcat_tier: str = Field(
        default="measured",
        description=(
            "'measured' resolves kcat from the Open Enzyme Database and from homologs and "
            "defaults the rest (of 3,728 catalytic units: 97 assayed, 782 homolog, 2,849 "
            "default). 'predicted' inserts the sequence-predictor tier between them (97 "
            "assayed, 782 homolog, 2,831 predicted, 18 default) using the predictors the "
            "within-class study admits for yeast. The counts are what the built layer "
            "reports (capacity_calibration.py --kcat-tier); the tier sets the capacity "
            "ceiling, and the measured tier's is the looser box, since the organism "
            "default turnover is above the predictors' median."
        ),
    )
    thermo_source: str = Field(
        default="shipped",
        description=(
            "Which route prices the standard reaction energies: 'shipped' (the GEM's own "
            "metabolite table summed over S, 3,505 of 4,131 reactions), 'equilibrator' "
            "(2,741) or 'union' (3,722)."
        ),
    )
    constraint_warmup_epochs: int = Field(
        default=0,
        description=(
            "Ramp the flux layer's constraint weights linearly from 0 over this many epochs."
        ),
    )
    constraint_scale: float = Field(
        default=1.0,
        description="Final multiplier on the whole weighted constraint sum after warmup.",
    )
    constraint_delay_epochs: int = Field(
        default=0,
        description=(
            "Hold the constraint scale at exactly zero for this many epochs before the ramp "
            "begins, so the module is applied to a head that has already left the plateau."
        ),
    )
    head_normalize: bool = Field(
        default=False,
        description=(
            "Divide the flux head's two inputs by the layer's working flux magnitude before "
            "reading them. MEASURED 2026-09-10 (diagnose_head_scale.py, results/"
            "head_scale_diagnosis.json): unnormalized, the flux head starts at output mean "
            "+7.030 against a target standardized to mean 0 sd 1, while the pooled head "
            "starts at -0.573. The head spends its first epochs unlearning a seven-sigma "
            "offset. Off by default so every wave-1 to wave-5 run deserializes unchanged."
        ),
    )
    # WAVE 6 (2026-09-10 night). The phenotype the cell trains, selects and scores on.
    head: str = Field(
        default=HEAD,
        description=(
            "'betaxanthin' is the wave-1 to wave-5 protocol and the default, so every "
            "earlier result file deserializes unchanged. 'mulleder19' moves the whole "
            "protocol to the 19-column Mulleder amino-acid phenotype: it is the only "
            "supervised head, selection is the mean per-feature validation Pearson over "
            "the 19 columns, and the test metrics are per-feature. On a flux arm the "
            "amino-acid head is 026's mechanistic `flux_metabolite` readout, which reads "
            "the turnover of the 19 named GEM metabolites off the predicted flux vector, "
            "so the arm contrast stays on the readout rather than on the trunk alone."
        ),
    )
    # WAVE 7 (2026-09-11). WHICH QUANTITY OF THE LAYER THE AMINO-ACID PREDICTION IS.
    aa_readout: str = Field(
        default=AA_READOUT_TURNOVER,
        description=(
            "'flux_metabolite' is wave 6's readout and the default: a_k log(1 + omega_k) + "
            "b_k on the TURNOVER of each amino acid's GEM metabolite, two free parameters "
            "per column, and it measured -0.039 mean per-feature Spearman against pooled at "
            "three seeds. 'flux_concentration' predicts the layer's own learned log "
            "concentration of that metabolite instead: the ANCHORED reaction potential "
            "already carries log_c = lo + (hi - lo) sigmoid(base + W context) per strain, "
            "and Mulleder 2016 measures exactly that quantity in absolute units. The "
            "prediction is then log10(exp(log_c)) with NO parameters of its own on the "
            "mean, so the same tensor the second law is computed from is the tensor the "
            "amino-acid loss trains, which is what makes the tie mechanistic rather than "
            "correlational. Requires an arm whose thermo_mode is ANCHORED. "
            "'context_concentration' (wave 8) is the ablation of that readout: the SAME "
            "head, base + Linear(d, m) from the genotype context into the same sigmoid "
            "window, the same mixture, standardization and CRPS, built on the pooled "
            "backbone with no flux layer at all. If it matches 'flux_concentration' the "
            "module is not on for this target; if it falls back to pooled, the module's "
            "thermodynamic penalty is what carries the gain. Requires --arm pooled."
        ),
    )
    target_log10_molar: bool = Field(
        default=False,
        description=(
            "Standardize and score the amino-acid target as log10 of its molar "
            "concentration rather than as the stored mM value. Set by `build_cell` for the "
            "'flux_concentration' and 'context_concentration' readouts and only for them, "
            "because that readout's "
            "prediction IS a log10 molar concentration and comparing it against a linear mM "
            "target would need a per-feature affine, which is the tie this arm removes. "
            "Spearman is invariant to the monotone change of variable, so the per-feature "
            "Spearman of a log10-molar arm and of an mM arm are directly comparable; "
            "Pearson is not, and both are reported."
        ),
    )
    dist: str = Field(
        default="crps",
        description=(
            "Which distributional loss scores the head. 'crps' is the Gaussian CRPS on a "
            "(mu, log_sigma) output, what every 027 cell before 2026-09-11 ran, and the "
            "default, so an earlier result file deserializes unchanged. 'quantile' is the "
            "019 pinball loss over the K = 19 tau grid from 0.05 to 0.95 "
            "(`torchcell.losses.distributional`), whose point estimate is the predictive "
            "MEDIAN rather than a mean. It applies to the MLP heads only: the flux readouts "
            "predict the layer's own quantity, so a quantile fan would have to come from a "
            "separate head and would break the tie that arm exists to make."
        ),
    )
    crps_sigma_fixed: float | None = Field(
        default=None,
        description=(
            "Hold the CRPS predictive spread at this value, in standardized target units, "
            "instead of letting the head predict it. Under a learned spread, widening sigma "
            "lowers the loss without fitting signal, which is one route to the constant "
            "predictor. The value that is not an escape comes from the label's own "
            "reliability rho as sqrt(1 - rho): 0.40 for betaxanthin at rho 0.84, 0.76 for "
            "the amino acids, 0.94 for the isobutanol screen "
            "(results/target_reliability.json)."
        ),
    )
    bx_readout: str = Field(
        default=BX_READOUT_FLUX_SCALAR,
        description=(
            "How the betaxanthin scalar is read off a flux arm. 'flux_scalar' is 026's "
            "head, w^T v plus the log turnover of the named precursors, and the default. "
            "'flux_precursor_concentration' reads the layer's learned whole-cell "
            "CONCENTRATION of the precursors instead, the same quantity the amino-acid "
            "arm supervises, through one weight per precursor and an intercept; that "
            "affine is required because betaxanthin is a fluorescence intensity and no "
            "unit-free map takes log10 molar to it. Needs an ANCHORED arm."
        ),
    )
    volume_table: str = Field(
        default=VOLUMES_ASSUMED,
        description=(
            "Which fractional-volume table weights the whole-cell concentration average. "
            "'assumed' is wave 7's textbook table (one of six entries sourced) and stays "
            "the default so every wave-7 cell deserializes unchanged. 'sourced' is Uchida "
            "et al. 2011's soft X-ray tomography measurement of S. cerevisiae, four of six "
            "entries measured with the quote and the mirrored sha256 attached per entry."
        ),
    )
    conc_space: str = Field(
        default=CONC_SPACE_MIXTURE,
        description=(
            "Which concentration the amino-acid prediction is. 'mixture' is the WHOLE-CELL "
            "average, sum_n f_n [A]_n over every intracellular compartment the GEM carries "
            "that amino acid in, with f_n the compartment's fractional volume: that is what "
            "Mulleder measures, and Thermo-Flux states the relation. It matters rather than "
            "being a refinement, because arginine, lysine and histidine are largely "
            "vacuolar in yeast, so a cytosol-only readout predicts the wrong pool for them. "
            "'cytosol' keeps only the cytosolic species and is the control that says what "
            "the mixture is worth."
        ),
    )
    conc_lower_m: float = Field(
        default=CONC_LOWER_M,
        gt=0.0,
        description=(
            "Floor of the flux layer's physiological concentration window, in molar, "
            "written onto the built layer's config. Default is the shipped constant."
        ),
    )
    conc_upper_m: float = Field(
        default=CONC_UPPER_M,
        gt=0.0,
        description=(
            "Ceiling of that window, in molar. The shipped 1e-2 does NOT cover the "
            "phenotype this arm supervises: 10.6 percent of Mulleder's 88,882 measurements "
            "are above it, and glutamate and glutamine are above it in 99.8 percent of "
            "strains, up to 9.5e-2 M (mulleder_concentration_window.json). A supervised "
            "value above the ceiling is unreachable and the sigmoid saturates silently, so "
            "the concentration cells raise this. It also widens the concentration term of "
            "every reaction energy, which is a real loosening of the thermodynamic anchor."
        ),
    )
    flux_parameterization: str | None = Field(
        default=None,
        description=(
            "The flux layer's channel string, overriding the one 026's ARMS entry carries. "
            "SERIALIZED HERE ON PURPOSE. Wave 7 passed it through the environment variable "
            "TC_FLUX_PARAMETERIZATION, which the layer records in "
            "flux_meta.coverage.parameterization_raw but the CELL does not, so anything "
            "that rebuilt a cell from its result file alone reconstructed the BASE arm: the "
            "first sampler audit of job 1755 died on "
            "KeyError 'flux_layer.latent_mlp.0.weight', a parameter that exists only on the "
            "arm the rebuild wrongly chose. With the channel in the cell, a rebuild is "
            "correct without an environment variable. The variable is still read when this "
            "field is None, so the wave-7 scripts and their reruns keep working."
        ),
    )
    flux_balance_active_row_set: str = Field(
        default="dynamic",
        description=(
            "Which rows the active-row balance term is priced on, from the arm file. "
            "'dynamic' recomputes the set from the flux under test, which job 1795 "
            "escaped by killing rows: 422 rows clearing the turnover threshold on the "
            "top 100 became 5. 'frozen' fixes the set at the first forward, so a "
            "killed row keeps its ratio of 2 in the sum."
        ),
    )
    flux_balance_active_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Weight on the flux layer's ACTIVE-ROW balance term, the relative residual over "
            "the independent rows of S that clear the turnover threshold. 0.0 is off and is "
            "what every cell before wave 8 ran. Like the density weight it is written onto "
            "the built layer's config and applied OUTSIDE the constraint scale, which is "
            "the whole point: at the scale this recipe runs, 0.0, the ordinary `c_balance` "
            "is multiplied away and nothing in the loss asks the flux to balance."
        ),
    )
    flux_density_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Weight on the flux layer's turnover-density term, which penalizes independent "
            "rows of S that do not turn over. 0.0 is off and is what every cell before "
            "wave 8 ran. Written onto the built layer's config, and NOT multiplied by the "
            "constraint scale: the wave-7 recipe runs the physics at scale 0.0, where a "
            "term inside the ramp is exactly zero and the cell would be a silent duplicate "
            "of the arm it is supposed to test."
        ),
    )
    flux_parsimony_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Weight on the flux layer's parsimony term, added OUTSIDE the constraint "
            "scale. The `weights['parsimony']` entry is inside the ramp, and this recipe "
            "holds that ramp at 0.0, so parsimony has never actually been applied."
        ),
    )
    flux_thermo_exempt_path: str | None = Field(
        default=None,
        description=(
            "JSON list of reaction ids (with route, window and table sha256) that no "
            "concentration in the window can make downhill, exempted from the second-law "
            "tightening after construction. Written by thermo_exempt_gate.py; None is off."
        ),
    )
    flux_log_c_init_path: str | None = Field(
        default=None,
        description=(
            "JSON log-concentration vector under which the reference flux is sign "
            "consistent, used to initialize log_c_base after construction. Written by "
            "thermo_exempt_gate.py; None is off."
        ),
    )
    flux_availability_fold: str = Field(
        default="softmin",
        description=(
            "How gene gates fold up an AND in the GPR: 'softmin' (the shipped fold, which "
            "reads 1 - log(n)/beta on an intact n-subunit complex) or 'normalized_softmin' "
            "(divided by its all-ones value, so an intact complex reads exactly 1)."
        ),
    )
    # THE GENOTYPE-KEYED WARM START (2026-09-14). All three are CONSTRUCTION-TIME: the flux
    # layer reads them in `FluxLayer.__init__`, where it solves or loads the reference and
    # creates `warm_log_alpha`, so they travel through the arm dict into `build_flux_layer`
    # the way `log_c_from_latent` does and CANNOT be assigned onto a built layer's config.
    # An assignment after construction would be silently ignored, which is the failure mode
    # worth naming: the cell would report a reference table it never loaded.
    flux_shuffle_kcat_seed: int | None = Field(
        default=None,
        description=(
            "Permute the resolved kcat vector ACROSS catalytic units with this seed before "
            "the layer is built, which is the shuffled-parameter control the review asks "
            "for. Architecture, parameter count, tier counts and the value DISTRIBUTION "
            "are unchanged; only the correspondence between a turnover number and its "
            "reaction is destroyed. An arm that still beats its pooled control under the "
            "permutation gained something from the architecture rather than from the "
            "measured kinetics. Construction-time, so it travels through the arm dict. "
            "None is off and is what every cell before wave 9 ran."
        ),
    )
    flux_reference_table: str | None = Field(
        default=None,
        description=(
            "Path to the per-genotype reference flux table (npz) written by "
            "`probe_reference_flux_table.py`, setting "
            "`FluxLayerConfig.warm_start_reference_table_path`. The layer looks each batch "
            "row's reference up by its sorted deleted-gene SET through "
            "`read_reference_flux_table` and `FluxLayer.genotype_reference`, falling back "
            "to the single wild-type reference for a key the table misses and reporting "
            "the fallback RATE in `coverage_report`. None keeps the one-reference behavior "
            "job 1946 ran, where the proposal `v_ref + alpha N z` was very nearly the same "
            "vector for every genotype and the head reading it was close to a constant. "
            "A table indexed by another GEM's reaction list is refused rather than applied "
            "by position."
        ),
    )
    flux_alpha_frozen: bool = Field(
        default=False,
        description=(
            "Freeze the warm start's residual scale: sets "
            "`FluxLayerConfig.warm_start_alpha_learnable = False`, which holds "
            "`warm_log_alpha` at `warm_start_init_rel` with no gradient. It stays a "
            "PARAMETER and stays in the state dict, so a frozen cell and a learned cell "
            "load each other's checkpoints and the arm is a one-boolean diff. This is the "
            "control for job 1946's failure mode, where alpha grew 0.0099 to 0.087 and "
            "took the active-row balance from about 1.0 to 2.0 with it: with alpha fixed, "
            "genotype information has to come from the reference rather than from the "
            "residual's size."
        ),
    )
    flux_warm_start_init_rel: float = Field(
        default=0.01,
        gt=0.0,
        description=(
            "`FluxLayerConfig.warm_start_init_rel`, the proposal's initial relative "
            "distance from the reference. Alpha is learned in log space so it stays "
            "positive without a clamp; under `--flux-alpha-frozen` this is the value it is "
            "held at for the whole run. The default equals the shipped constant, so a cell "
            "that does not name one builds a bit-identical layer."
        ),
    )
    # WAVE 8 (2026-09-12). K draws per genotype per forward pass, the simplest form of the
    # question Flux Cone Learning answers with 124 OptGPSampler samples per deletion cone.
    # All three default to what every earlier cell ran, so an older result file deserializes
    # unchanged. Read [[experiments.027-betaxanthin-metabolic.wave8-sampling]] first: the
    # Step 0 diagnostic is why `flux_latent_stochastic` is a cell field at all.
    flux_draws: int = Field(
        default=1,
        ge=1,
        description=(
            "Latent draws per genotype per forward pass. The batch is expanded to B*draws "
            "rows inside the flux layer, so the data loss becomes the mean over draws of "
            "the per-draw CRPS and every constraint residual the mean over draws of its own "
            "value. At 1 the forward pass is bit-for-bit the one every earlier cell ran. "
            "Above 1 the layer refuses to build unless the latent actually draws, since K "
            "copies of mu are not K samples."
        ),
    )
    flux_latent_stochastic: bool = Field(
        default=False,
        description=(
            "Turn the flux layer's latent draw on, z = mu + sigma eps with a KL against "
            "N(0, I). MEASURED (wave8_latent_spread.json): no arm in 026's ARMS registry "
            "except `flux_projection_stochastic` sets this, so every wave-6 and wave-7 "
            "projection cell trained a deterministic latent whose spread head never "
            "received a gradient and kept sigma at exp(-3) exactly. A K-draw contrast has "
            "to hold this fixed across both of its arms, or it confounds the draw count "
            "with whether the model samples at all."
        ),
    )
    flux_kl_weight: float | None = Field(
        default=None,
        description=(
            "Override the flux layer's `kl_weight`, the coefficient on out['c_kl']. None "
            "keeps FluxLayerConfig's 1e-3. The KL sits INSIDE the ramped constraint sum, so "
            "the weight that is actually applied is this times `constraint_scale`, which is "
            "recorded as `effective_kl_weight` in the result's flux_meta. It matters "
            "because the KL is the only term pushing sigma up: at an effective 1e-4 the "
            "objective shrank the median posterior sd 130x over 80 epochs on the one 027 "
            "cell that trained it."
        ),
    )
    # WAVE 9 (2026-09-14). The isobutanol target, the two missing is-the-module-on controls,
    # the held-out amino-acid columns and the medium. Every field defaults to what wave 8
    # ran, so an earlier result file deserializes unchanged and an earlier cell is
    # bit-identical.
    pos_weight_balanced: bool = Field(
        default=False,
        description=(
            "Weight the essential class in the binary cross entropy by the released "
            "split's training-side nonessential-to-essential ratio (772/126), so both "
            "classes contribute equally to the loss. Off is the wave-9 to wave-13 "
            "protocol, where 16 percent positives leave the loss dominated by the "
            "majority class."
        ),
    )
    ess_readout: str = Field(
        default=ESS_READOUT_BIOMASS,
        description=(
            "On a flux arm with the essentiality head: `biomass` reads the growth column "
            "alone (wave 9); `context_biomass` adds the growth affine to the pooled arm's "
            "own context head; `gated` puts it behind a zero-initialized gate. On a pooled "
            "arm: `fba` adds an affine of FBA's four growth ratios for the deleted gene; "
            "`biomass` (the default) is inert there."
        ),
    )
    node_embeddings: list[str] = Field(
        default=list(STRONG_NODE_EMBEDDINGS),
        description="The node embeddings the dataset was opened with, in order.",
    )
    ess_fba_media: list[str] = Field(
        default=list(FBA_MEDIA),
        description=(
            "The FBA media whose growth ratios the fba readouts carry, in this order; "
            "the default is all four."
        ),
    )
    flux_capacity_scale: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Multiplies the flux layer's enzyme-capacity ceilings before they enter the "
            "box (`FluxLayerConfig.capacity_scale`). 1.0 is the ceiling as parameterized, "
            "under which the growth LP is infeasible for every genotype (round 15); "
            "capacity_calibration.py finds the smallest scale that admits wild-type growth."
        ),
    )
    flux_lp_slack_penalty: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Under --flux-parameterization lp, the price per unit of unmet forced flux in "
            "the elastic program (`FluxLayerConfig.lp_slack_penalty`); 0 is the hard "
            "program, whose infeasible state absorbed the forced cells of rounds 15b to "
            "15d. With it every genotype's program is feasible by construction."
        ),
    )
    flux_pool_by_construction: bool = Field(
        default=False,
        description=(
            "Project the abundance head's output onto the protein budget before the "
            "capacity ceilings are formed (`FluxLayerConfig.pool_by_construction`), so "
            "the GECKO pool constraint holds exactly instead of as the soft hinge."
        ),
    )
    flux_lr_mult: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Learning-rate multiplier on the flux layer's parameters (a second AdamW "
            "group); 1.0 is one group. Round 19: the forced column under the solve "
            "underfits at the backbone's rate."
        ),
    )
    flux_abundance_head_layers: int = Field(
        default=1,
        ge=1,
        le=2,
        description=(
            "Depth of the abundance head E_g(h_g): 1 is the shipped linear map (one "
            "weight vector shared by every gene), 2 a hidden layer of the token width "
            "(`FluxLayerConfig.abundance_head_layers`). Round 19."
        ),
    )
    flux_abundance_log_range: float = Field(
        default=3.0,
        gt=0.0,
        description=(
            "The abundance head's reach r in E_g = E0_g exp(r tanh(.)), e^r either side "
            "of the prior (`FluxLayerConfig.abundance_log_range`, read at call time); "
            "3.0 is the shipped value. Round 20."
        ),
    )
    ess_flux_head_reactions: str = Field(
        default="all",
        description=(
            "Which reactions the `gated_flux` readout's flux head reads: 'all', "
            "'exchange' (uptake and secretion), or 'gene' (gene-owned). Round 21."
        ),
    )
    ess_solve_features: str = Field(
        default="none",
        description=(
            "Solve-derived per-gene columns appended to the fba readouts' affine table "
            "after the media ratios: 'stats' (eight wild-type pFBA and FVA statistics of "
            "the gene's reactions), 'pfba' (the deletion's own pFBA flux vector, 4,131 "
            "columns), 'stats_pfba' (both), from ess_reference_flux_table.py. 'none' is "
            "the four ratios alone, bit-identical to round 5."
        ),
    )
    ess_solve_encoding: str = Field(
        default="standardized",
        description=(
            "How the pFBA vector's columns enter the affine: 'standardized' (round 31) or "
            "'log1p' (log(1 + |v|), the flux head's encoding; round 33)."
        ),
    )
    ess_solve_zero_init: bool = Field(
        default=False,
        description=(
            "Zero the affine's weights on the solve-feature columns at construction, as "
            "the flux head is zero-initialized, so the cell starts as the four-ratio "
            "affine and the vector enters by gradient only (round 33). The media ratios' "
            "weights and the bias keep their default initialization."
        ),
    )
    ess_solve_head_hidden: int = Field(
        default=0,
        description=(
            "Width of one GELU hidden layer between the affine's input (the media ratios "
            "and the solve columns) and its logit; 0 is the linear affine of every round "
            "through 40. Round 41 asks whether the linear read is the vector's ceiling."
        ),
    )
    ib_readout: str = Field(
        default=IB_READOUT_FLUX_SCALAR,
        description=(
            "How the isobutanol biosensor fold change is read off a FLUX arm. "
            "'flux_scalar' is 026's head, w^T v over all 4,131 reactions plus the log "
            "turnover of the named precursors and of the product species whose turnover is "
            "the r_1866 secretion flux. 'flux_precursor_concentration' reads the layer's "
            "learned whole-cell CONCENTRATION of the anchors instead, through one weight "
            "per anchor and an intercept; that affine is required because the target is a "
            "GFP fold change and no unit-free map takes log10 molar to it. Ignored on "
            "--arm pooled, which has no layer and uses the ordinary scalar head."
        ),
    )
    ib_anchors: str = Field(
        default=IB_ANCHORS_ONE,
        description=(
            "Which anchors the isobutanol precursor readout reads. 'one' is the PRIMARY "
            "arm, alpha-isopropylmalate alone, which is literally what Leu3p binds; "
            "'three' adds 2-keto-isovalerate and L-valine and is the secondary."
        ),
    )
    target_log10: bool = Field(
        default=False,
        description=(
            "Standardize and score the target as log10 of its STORED value, with no unit "
            "conversion. Set by `build_cell` for --head isobutanol and only for it: the "
            "biosensor number is a fold change, which is multiplicative, so a Gaussian "
            "CRPS is well posed in log10 and not in linear FC. Distinct from "
            "`target_log10_molar`, which additionally divides by 1000 to turn Mulleder's "
            "stored mM into molar."
        ),
    )
    aa_supervise_columns: list[str] | None = Field(
        default=None,
        description=(
            "Which of the 19 amino-acid columns the LOSS is charged on. None supervises "
            "all 19 and is what every wave-6 to wave-8 cell ran. A subset holds the rest "
            "out: a physical concentration latent is ONE object linked by stoichiometry "
            "and thermodynamics, so supervision at 13 columns should reach the other 6 "
            "through S and the reaction energies, while a bank of independent per-metabolite "
            "sigmoids cannot -- the rows the mixture never reads receive no gradient and "
            "stay at their initialization (asserted in "
            "tests/torchcell/metabolism/test_context_concentration_readout.py). The epoch "
            "is selected on the SUPERVISED columns' validation Pearson so the held-out "
            "columns never enter selection; the test scorer is unchanged and reports all "
            "19 per-column, with the two groups summarized in `aa_column_split`."
        ),
    )
    apply_media: str = Field(
        default=MEDIA_NONE,
        description=(
            "Whether the dataset's recorded GROWTH MEDIUM is written onto the GEM's "
            "exchange bounds before the flux layer is built. 'none' is the shipped "
            "yeast-GEM boundary every 026 and 027 arm has run on, where all 20 amino-acid "
            "exchanges have lower_bound 0, i.e. effectively synthetic minimal; it is the "
            "default so every earlier cell is bit-identical. 'dataset' applies "
            "`media_to_bounds(FBA_MEDIA[name], model).apply(model)` for the medium the "
            "head's dataset was grown in (SM for Mulleder, YPD for Cachera, SC for Lopez), "
            "and records the wild-type growth rate before and after in flux_meta. The "
            "shipped bounds are RIGHT for Mulleder by accident and wrong for the other two."
        ),
    )
    flux_log_c_from_latent: bool = Field(
        default=False,
        description=(
            "Make the flux layer's anchored concentration read the DRAWN latent as well as "
            "the context, log c = lo + (hi - lo) sigmoid(base + W context + U z) with U "
            "zero-initialized, so each of the K draws carries its own concentration and "
            "the `flux_concentration` amino-acid readout becomes a function of the sample. "
            "Without it that readout is a function of the genotype context alone and a "
            "K-draw contrast on it is null by construction (wave 8 Finding 5). Requires "
            "--flux-latent-stochastic; a construction-time switch, forwarded through the "
            "arm dict into 026's build_flux_layer. Default False is bit-identical to every "
            "earlier cell."
        ),
    )


AUX_HEAD = "mulleder19"
#: The measured amino-acid columns, in the order 026 fixed. Sorted, so the column order is
#: a property of the key set and not of a dict literal's typing order.
AA_KEYS: list[str] = sorted(AMINO_ACID_TO_GEM_METABOLITE)
AUX_WIDTH = len(AA_KEYS)

#: The 020 grid's hard mask: relation -> attention head at layer 1, in edge-type names.
ATTENTION_MASK_CONFIG: dict[str, Any] = {
    "enabled": True,
    "layers": [1],
    "head_graphs": {
        0: "physical_interaction",
        1: "regulatory_interaction",
        2: "tflink",
        3: "string12_0_neighborhood",
        4: "string12_0_fusion",
        5: "string12_0_cooccurence",
        6: "string12_0_coexpression",
        7: "string12_0_experimental",
        8: "string12_0_database",
    },
}


#: The 019 graph-structure prior, copied from gh_metabolism_000.yaml: one regularized
#: attention head per graph at layer 1, lambda 6.5e-5, a quarter of the rows sampled.
GRAPH_REG_LAMBDA = 6.5e-5
GRAPH_REG_CONFIG: dict[str, Any] = {
    "graph_reg_lambda": GRAPH_REG_LAMBDA,
    "graph_reg_layer": [1],
    "row_sampling_rate": 0.25,
    "regularized_heads": {
        name: {"layer": [1], "head": i, "lambda": GRAPH_REG_LAMBDA}
        for i, name in enumerate(STRONG_GRAPHS)
    },
}
#: Collapse detector threshold: prediction spread over target spread on the validation
#: split. The grid's cells sat at 8e-3 and below. The patience is NOT ten epochs: on the
#: current code the mean plateau is the normal start and lasts 117 to 157 epochs before the
#: spread appears (020 Delta grid, W&B history of runs w7fuxsvr, azh3tstz, 74sde13z,
#: dhmloet1), while the July 26 code left it at epoch 23 (run gilahyper-1346). A ten-epoch
#: abort, which the 2026-09-09 diagnostics ran with, kills every run that would have
#: escaped; the 027 grid's 200 epochs on Delta were also inside that window for most seeds.
COLLAPSE_SD_RATIO = 0.01
COLLAPSE_PATIENCE_EPOCHS = int(os.getenv("COLLAPSE_PATIENCE_EPOCHS", "200"))
#: The clip on the global gradient norm, unchanged since 026; logged beside the pre-clip
#: norm so `grad_clipped_frac` says how often it binds.
GRAD_CLIP_NORM = 10.0


def point_estimate(
    raw: torch.Tensor, width: int, dist_head: "DistHead | None"
) -> torch.Tensor:
    """The head's POINT prediction ``[B, F]``, whatever distributional form it emitted.

    Gaussian CRPS reports ``mu``; the quantile head reports the predictive MEDIAN, which is
    the knot at tau nearest 0.5 and is what ``DistHead.point`` returns. Reading the first
    parameter of a quantile fan would report the 5th percentile as the prediction.
    """
    if raw.dim() != 3:
        return raw[:, :width]
    if dist_head is not None:
        return dist_head.point(raw)
    return raw[..., 0]


def _head_params(pred: torch.Tensor) -> torch.Tensor:
    """``[B, param_dim]`` from a scalar head, whichever shape it emitted.

    A distributional scalar head emits ``[B, 1, param_dim]`` and a point head ``[B, 1]``.
    The grid's loop did ``pred[..., 0]`` on the former, which keeps mu and throws away
    log_sigma, so the CRPS branch of the loss was unreachable. Squeezing the feature axis
    keeps both parameters.
    """
    return pred.squeeze(1) if pred.dim() == 3 else pred


class CompartmentVolumeFractions(BaseModel):
    r"""Fractional volumes :math:`f_n`, with per-compartment provenance.

    WHY THIS EXISTS. Mulleder 2016 measures a WHOLE-CELL concentration, while the flux
    layer learns one concentration per compartment species. Thermo-Flux (Smith et al. 2026,
    mirrored ``paper.md`` sha256
    ``496019ea07d9d95a63c1f0dbeada666d1e775a0dfcd50e53fe0235b87ead2593``) states the
    relation verbatim: "The whole-cell concentration of a metabolite is a weighted average
    of the concentration in the different subcellular compartments", ``[A]_total = [A]_1
    f_1 + ... + [A]_n f_n``, "where ``[A]_total`` is the concentration of a metabolite in
    the whole cell, ``[A]_n`` is the concentration of the metabolite in compartment ``n``
    and ``f_n`` is the fractional volume of compartment ``n``". It gives no yeast values.

    This matters and is not a refinement: arginine, lysine and histidine are largely
    vacuolar in yeast, so a cytosol-only readout predicts the wrong pool for them.

    PROVENANCE, stated per entry rather than as one blanket claim. Only the nucleus is
    sourced; the rest are textbook exponential-phase values carried here as ASSUMED, in the
    same way ``transport.pipeline_conditions`` carries assumed organellar pH. They are
    recorded in every result file and flagged for curation.
    """

    model_config = ConfigDict(extra="forbid")

    fractions: dict[str, float]
    sourced: dict[str, bool]
    source: str
    quote: str
    sha256: str
    note: str


def compartment_volume_fractions(
    table: str = VOLUMES_ASSUMED,
) -> CompartmentVolumeFractions:
    """The fractional-volume table a cell runs on.

    ``sourced`` reads `torchcell.metabolism.parameters.yeast_compartment_volume_fractions`,
    which carries Uchida et al. 2011's soft X-ray tomography measurements of *S.
    cerevisiae* with the quote and the mirrored sha256 per entry: cytosol 0.70 (the
    paper's own remainder, an upper bound), vacuole 0.085 and nucleus 0.085 and
    mitochondrion 0.015 as midpoints of the reported ranges, with the endoplasmic
    reticulum and the peroxisome still flagged unsourced because that study does not
    resolve them. ``assumed`` is wave 7's textbook table, kept so the two can be paired
    seed for seed and so every wave-7 result file deserializes unchanged.
    """
    assert table in VOLUME_TABLES, (
        f"--volume-table must be one of {VOLUME_TABLES}, got {table!r}."
    )
    if table == VOLUMES_SOURCED:
        measured = yeast_compartment_volume_fractions()
        entry = {e.compartment: e for e in measured.entries}
        return CompartmentVolumeFractions(
            fractions=measured.fractions(),
            sourced=measured.sourced(),
            source=measured.source,
            quote=next(e.quote for e in measured.entries if e.quote),
            sha256=next(e.sha256 for e in measured.entries if e.sha256),
            note=(
                measured.note
                + " Per-entry notes: "
                + "; ".join(
                    f"{k}: {v.note}" for k, v in sorted(entry.items()) if v.note
                )
            ),
        )
    return CompartmentVolumeFractions(
        fractions={"c": 0.65, "v": 0.18, "n": 0.10, "m": 0.05, "er": 0.015, "p": 0.005},
        sourced={
            "c": False,
            "v": False,
            "n": True,
            "m": False,
            "er": False,
            "p": False,
        },
        source=(
            "Nucleus: Milo and Phillips 2016, Cell Biology by the Numbers, mirrored "
            "paper.md sha256 "
            "a0b71fc2ec024108cc0d11a6ba08b5b32d6d9174386a73ac377f377364946eae, "
            "'How big is a yeast cell?', citing BNID 100491 and 103952. Every other "
            "compartment: textbook exponential-phase values, ASSUMED, not sourced."
        ),
        quote=(
            "Prominent components of the cell volume include the nucleus, which takes up "
            "about $1 0 \\%$ of the total cell volume (BNID 100491, 103952), the cell "
            "wall, often ignored but making up $1 0 { - } 2 5 \\%$ (BNID 104593, 104592) "
            "of the total dry mass, and the endoplasmic reticulum and vacuole, which are "
            "usually the largest organelles."
        ),
        sha256="a0b71fc2ec024108cc0d11a6ba08b5b32d6d9174386a73ac377f377364946eae",
        note=(
            "Five of the six entries are assumed and NEED CURATION. The weights set which "
            "compartment dominates a whole-cell average, so they change the physical "
            "reading of the prediction; they do not limit what the model can fit, because "
            "each compartment's concentration is free inside the window and can compensate "
            "for a wrong weight. The fractions sum to 1 over the intracellular "
            "compartments, and a compartment where the GEM has no species for an amino "
            "acid contributes nothing to that amino acid's average."
        ),
    )


def compartment_species_for(
    anchors: dict[str, str],
) -> dict[str, list[tuple[str, str]]]:
    """``{label -> [(compartment, species id), ...]}`` for the whole cell.

    Derived from the GEM at run time rather than hardcoded, so it cannot drift from the
    model the layer was built on: every species whose name matches the anchor species'
    base name, minus the extracellular and membrane pseudo-compartments.

    Args:
        anchors: ``{label -> one GEM species id}``; the id names the chemical and the
            search finds the rest of its compartments.
    """
    from torchcell.metabolism.yeast_GEM import YeastGEM

    model = YeastGEM().model
    by_id = {s.id: s for s in model.metabolites}
    out: dict[str, list[tuple[str, str]]] = {}
    for label, anchor in anchors.items():
        assert anchor in by_id, f"{label!r} names {anchor!r}, which is not in the GEM."
        base = by_id[anchor].name.split(" [")[0]
        hits = sorted(
            (str(s.compartment), str(s.id))
            for s in model.metabolites
            if s.name.split(" [")[0] == base
            and str(s.compartment) not in NON_CELL_COMPARTMENTS
        )
        assert hits, f"no intracellular GEM species found for {label!r} ({base!r})."
        out[label] = hits
    return out


def amino_acid_compartment_species() -> dict[str, list[tuple[str, str]]]:
    """``{amino acid -> [(compartment, species id), ...]}`` for the whole cell."""
    return compartment_species_for(
        {aa: AMINO_ACID_TO_GEM_METABOLITE[aa] for aa in AA_KEYS}
    )


def gem_metabolite_index() -> dict[str, int]:
    """``{species id -> row}`` in the GEM's own order, WITHOUT building a flux layer.

    The flux layer's metabolite axis is ``GemTensors.met_ids``, which
    ``build_gem_tensors`` derives as ``[m.id for m in model.metabolites]`` and 026's
    ``build_flux_layer`` publishes as ``flux_meta["met_index"]``. The context-only
    concentration readout needs that same axis (its head has one row per GEM metabolite,
    so its parameter count equals the flux layer's concentration head) and needs nothing
    else from the layer, so the map is read from the same cobra model directly. Checked
    against the layer's own index in ``tests/torchcell/metabolism``.
    """
    from torchcell.metabolism.yeast_GEM import YeastGEM

    return {str(m.id): i for i, m in enumerate(YeastGEM().model.metabolites)}


#: WAVE 8 BLOCK 3. Betaxanthin is NOT a GEM metabolite: the Cachera cassette makes it
#: outside the native network, L-tyrosine -> (CYP76AD1) L-DOPA -> (DOD) betalamic acid ->
#: betaxanthin, and yeast-GEM carries no L-DOPA, betalamic acid or betaxanthin species
#: (searched by name, 2026-09-12). What the GEM does carry is the SUPPLY of the route: the
#: tyrosine pool the cassette draws on, and the shikimate nodes the cassette's own
#: feedback-resistant ARO4 and ARO7 alleles act on. ARO4 is DAHP synthase, so its product
#: DAHP is the node it deregulates; ARO7 is chorismate mutase, so chorismate is its
#: substrate and prephenate its product.
#:
#: NOT the same list as 026's `BETAXANTHIN_PRECURSORS`, and the difference is a defect
#: found here rather than a preference: that list is
#: ``["s_1051", "s_0188", "s_1032"]``, and **s_0188 is 2-phospho-D-glyceric acid**, a
#: glycolysis intermediate one enolase step before phosphoenolpyruvate, not an aromatic
#: intermediate. 026's own comment describes the entry as one of "the two aromatic-pathway
#: intermediates the cassette's alleles act on", so the id does not match the intent.
#: 026 owns that registry and is not edited from here; this list is the corrected one and
#: the turnover-based `flux_scalar` head still reads 026's.
BETAXANTHIN_PRECURSOR_ANCHORS: dict[str, str] = {
    "tyrosine": "s_1051",
    "phenylalanine": "s_1032",
    "chorismate": "s_0515",
    "prephenate": "s_1377",
    "dahp": "s_0349",
}


#: WAVE 9. THE ISOBUTANOL ANCHORS, all three native yeast-GEM 9.0.2 species, verified
#: against the model on 2026-09-14 (ids, names and compartments read from
#: `yeast-GEM.xml`): `s_0162` 2-isopropylmalate [c] (also [m] `s_0164`; [e] `s_0163` is
#: dropped by `NON_CELL_COMPARTMENTS`), `s_0232` 3-methyl-2-oxobutanoate [c] -- the GEM's
#: name for 2-keto-isovalerate -- (also [m] `s_0233`), and `s_1056` L-valine [c] (also [m]
#: `s_1058`). As with the betaxanthin anchors, the id NAMES the chemical and
#: `compartment_species_for` derives the rest of its compartments from the GEM at run time,
#: so the readout cannot drift from the model the layer was built on.
#:
#: THE PRIMARY ARM READS alpha-isopropylmalate ALONE, because that is literally the ligand
#: Leu3p binds; the three-anchor set is the secondary and adds the pool one step upstream
#: (the keto acid both the leucine and the isobutanol branch draw on) and the transaminase
#: product that competes for it.
ISOBUTANOL_PRECURSOR_ANCHORS: dict[str, str] = {
    "alpha_isopropylmalate": "s_0162",
    "keto_isovalerate": "s_0232",
    "valine": "s_1056",
}
ISOBUTANOL_PRIMARY_ANCHOR = "alpha_isopropylmalate"

#: The yeast-GEM secretion exchange for the PRODUCT, `isobutanol [e] -->`. Unlike
#: betaxanthin, this route needs no heterologous cassette: the full pathway exists with
#: real ORFs on every step (ILV2/ILV6 r_0097, ILV5 r_0096, ILV3 r_0352, BAT1/BAT2
#: r_1087/r_1088, PDC1/5/6 r_0062, ADH r_0182/r_0183/r_0184), so the flux arm has a real
#: product column to read rather than a proxy.
ISOBUTANOL_PRODUCT_EXCHANGE = "r_1866"

#: The metabolite rows 026's `FluxScalarHead` reads the TURNOVER of on an isobutanol flux
#: arm, the analogue of `BETAXANTHIN_PRECURSORS`. Two groups, and the difference matters:
#: the first six are the intracellular PRECURSOR pools, and the last three are the PRODUCT.
#: `s_0933` is isobutanol [e], and the only two reactions touching it are the transport
#: `r_1868` and the exchange `r_1866` (measured on the GEM, 2026-09-14), so its turnover
#: IS the secretion flux this arm is supposed to read. The head also carries a dense
#: `w^T v` term over all 4,131 reactions, in which `r_1866` is column 1307, so the product
#: flux is reachable twice over; naming it here is what makes the arm a hypothesis rather
#: than an unrestricted read.
ISOBUTANOL_PRECURSORS: list[str] = [
    "s_0162",  # 2-isopropylmalate [c], the biosensor ligand
    "s_0164",  # 2-isopropylmalate [m]
    "s_0232",  # 3-methyl-2-oxobutanoate (2-keto-isovalerate) [c]
    "s_0233",  # 3-methyl-2-oxobutanoate [m]
    "s_1056",  # L-valine [c]
    "s_1058",  # L-valine [m]
    "s_0932",  # isobutanol [c]
    "s_0934",  # isobutanol [m]
    "s_0933",  # isobutanol [e]; its turnover is the r_1866 secretion flux
]

#: HOW THE ISOBUTANOL TARGET IS DECODED, and it is NOT by width 1. Measured on
#: `fig6_build` (2026-09-14, over all 13,948 genotypes): `MeanExperimentDeduplicator` has
#: already MERGED the Lopez screen into the Mulleder record wherever both were measured, so
#: 4,244 genotypes carry ONE `metabolite_level` group of width 20 -- the 19 amino acids
#: plus `isobutanol`, key-sorted by the graph processor -- while the width-1 groups in that
#: build are 4,669 betaxanthin records against 189 isobutanol-only ones. A width-1
#: isobutanol head would therefore have read betaxanthin colony color on 96 percent of the
#: rows it matched. The head is decoded as one COLUMN of the width-20 block instead, which
#: is identity-based: no other combination of datasets in that build produces width 20.
ISOBUTANOL_BLOCK_KEYS: list[str] = sorted([*AA_KEYS, ISOBUTANOL_HEAD])
ISOBUTANOL_BLOCK_WIDTH = len(ISOBUTANOL_BLOCK_KEYS)
ISOBUTANOL_COLUMN = ISOBUTANOL_BLOCK_KEYS.index(ISOBUTANOL_HEAD)

#: The dataset whose records are HELD BACK from the isobutanol loss. The 224-strain
#: validated re-screen (n = 3) is the external direction check, and on `fig6_build` its
#: value is not separable from the screen's: the deduplicator averaged the two into one
#: `isobutanol` entry on every genotype where both were measured (208 of the 4,244 width-20
#: blocks). Supervising those rows would train on a number that partly IS the external
#: check, so every genotype deleting a gene the validated panel measured is masked out of
#: the loss and out of the test score, and its prediction is what the check reads.
ISOBUTANOL_HELDOUT_DATASET = "IsobutanolValidatedLopez2024Dataset"

#: The medium each head's dataset was GROWN IN, as a key of
#: `torchcell.metabolism.media.FBA_MEDIA`, used only under `--apply-media dataset`.
#:
#: PROVENANCE IS NOT UNIFORM ACROSS THE THREE, and a result file has to say so.
#: * Mulleder 2016 is synthetic MINIMAL agar, and `SM_FBA` is YNB + ammonium + glucose with
#:   no amino acids. That is what the shipped bounds already are, so this entry is the NULL
#:   control: applying it should leave the amino-acid arm where it was.
#: * Cachera 2023 is solid YPD + G418. `YPD_APPROX_FBA` is Suthers' defined stand-in, whose
#:   quote is carried in `torchcell/metabolism/media.py`; peptone is not modeled by anyone.
#: * Montano Lopez 2024 is SC liquid at 30 C. `SC_FBA` carries NO provenance and NO
#:   concentrations: its composition is the standard Difco/Sigma formulation named in
#:   `torchcell/datamodels/media.py` and only the uptake magnitudes are sourced (Suthers).
#:   An SC-bounded cell is therefore running on an UNSOURCED medium and is labeled as such
#:   in `flux_meta.media`.
#: * Essentiality is scored on RICH medium: the deletion collection's viability calls come
#:   from growth on YPD, and Merzbacher's split inherits those literature calls. This is
#:   deliberately NOT the medium of the published FBA comparator, which Merzbacher ran on
#:   glucose minimal aerobic; the phenotype and that baseline are different experiments, and
#:   the mismatch is recorded rather than split the difference. A cell that wants to
#:   reproduce the FBA number runs on the shipped bounds instead, with `--apply-media none`.
HEAD_MEDIUM: dict[str, str] = {
    HEAD: "YPD",
    AUX_HEAD: "SM",
    ISOBUTANOL_HEAD: "SC",
    ESSENTIALITY_HEAD: "YPD",
}


def isobutanol_anchors(anchor_set: str) -> dict[str, str]:
    """``{label -> GEM species id}`` for the isobutanol precursor readout.

    The one-anchor set is the PRIMARY arm and holds only alpha-isopropylmalate, which is
    the metabolite Leu3p binds and therefore the quantity the reporter is proportional to.
    """
    assert anchor_set in IB_ANCHOR_SETS, (
        f"--ib-anchors must be one of {IB_ANCHOR_SETS}, got {anchor_set!r}."
    )
    if anchor_set == IB_ANCHORS_ONE:
        return {
            ISOBUTANOL_PRIMARY_ANCHOR: ISOBUTANOL_PRECURSOR_ANCHORS[
                ISOBUTANOL_PRIMARY_ANCHOR
            ]
        }
    return dict(ISOBUTANOL_PRECURSOR_ANCHORS)


def isobutanol_heldout_flags(dataset: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    """``([n_genes] bool, report)``: which genes the validated re-screen measured.

    The panel is held back as an EXTERNAL direction check, and on ``fig6_build`` that has
    to be done by gene rather than by value: ``MeanExperimentDeduplicator`` averaged the
    screen and the re-screen into one ``isobutanol`` entry wherever both were measured, so
    those records carry a label that partly IS the check. Every genotype deleting a gene
    the panel measured is therefore masked out of the loss, the validation curve and the
    test score, and only its PREDICTION is read.

    The membership is taken from the dataset's own ``dataset_name_index``, which records
    which source datasets each aggregated genotype came from, so no second copy of the
    Lopez gene list is kept anywhere.
    """
    held_records: set[int] = set()
    for combination, rows in dataset.dataset_name_index.items():
        if ISOBUTANOL_HELDOUT_DATASET in str(combination).split("+"):
            held_records.update(int(r) for r in rows)
    assert held_records, (
        f"no record of {ISOBUTANOL_HELDOUT_DATASET!r} in this build's dataset_name_index "
        f"({sorted(dataset.dataset_name_index)[:3]} ...). The isobutanol head is defined "
        "on fig6_build, which carries the validated re-screen; without it the external "
        "direction check has no strains and the hold-out is silently a no-op."
    )
    fraction = len(held_records) / len(dataset)
    assert fraction < 0.1, (
        f"the held-back panel is {fraction:.1%} of the dataset "
        f"({len(held_records)}/{len(dataset)}); the validated re-screen is 224 strains and "
        "should be a small minority, so this is resolving something else."
    )
    node_ids = list(dataset.cell_graph["gene"].node_ids)
    position = {name: i for i, name in enumerate(node_ids)}
    flags = torch.zeros(len(node_ids), dtype=torch.bool)
    held_genes: list[str] = []
    absent: list[str] = []
    for gene, rows in dataset.is_any_deletion_gene_index.items():
        if not held_records.intersection(int(r) for r in rows):
            continue
        held_genes.append(str(gene))
        if str(gene) in position:
            flags[position[str(gene)]] = True
        else:
            absent.append(str(gene))
    report = {
        "dataset": ISOBUTANOL_HELDOUT_DATASET,
        "records": len(held_records),
        "genes": len(held_genes),
        "genes_absent_from_cell_graph": absent,
        "gene_nodes_flagged": int(flags.sum()),
        "fraction_of_dataset": fraction,
        "why": (
            "the validated re-screen is the external direction check; on this build its "
            "value was averaged into the screen's by MeanExperimentDeduplicator, so the "
            "strains are held out by gene rather than by value."
        ),
    }
    print(
        f"[isobutanol-holdout] {len(held_records)} records / {len(held_genes)} genes of "
        f"{ISOBUTANOL_HELDOUT_DATASET} masked out of the loss and the score "
        f"({fraction:.1%} of {len(dataset)}); {len(absent)} absent from the cell graph",
        flush=True,
    )
    return flags, report


def mixture_weights(
    met_index: dict[str, int], space: str, volume_table: str = VOLUMES_ASSUMED
) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[tuple[str, str, float]]]]:
    r"""``(indices, log weights, per-amino-acid map)`` for the whole-cell average.

    The prediction is ``log10 sum_n f_n c_n``, computed as a log-sum-exp over the
    compartments so it is stable when one compartment sits at the window floor. Padding
    entries carry a log weight of ``-inf``, which contributes exactly zero.

    ``space='cytosol'`` keeps only the cytosolic species at weight 1, which is the
    comparison arm rather than the physical quantity: it answers what the mixture is worth.
    """
    return species_mixture_weights(
        amino_acid_compartment_species(), AA_KEYS, met_index, space, volume_table
    )


def species_mixture_weights(
    species: dict[str, list[tuple[str, str]]],
    order: list[str],
    met_index: dict[str, int],
    space: str,
    volume_table: str = VOLUMES_ASSUMED,
    bx_readout: str = BX_READOUT_FLUX_SCALAR,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[tuple[str, str, float]]]]:
    r"""The same whole-cell average for ANY set of chemicals, in a fixed column order.

    Shared by the 19 amino acids and by the betaxanthin precursors, so the two readouts
    cannot come to disagree about what a whole-cell concentration is.
    """
    assert space in CONC_SPACES, (
        f"conc space must be one of {CONC_SPACES}, got {space!r}."
    )
    table = compartment_volume_fractions(volume_table)
    rows: list[list[tuple[int, float]]] = []
    described: dict[str, list[tuple[str, str, float]]] = {}
    for aa in order:
        entries: list[tuple[int, float]] = []
        described[aa] = []
        for compartment, sid in species[aa]:
            if space == CONC_SPACE_CYTOSOL and compartment != "c":
                continue
            weight = (
                1.0 if space == CONC_SPACE_CYTOSOL else table.fractions[compartment]
            )
            assert sid in met_index, (
                f"{aa} species {sid} ({compartment}) is not in the layer's metabolite "
                "index, so the flux layer and the GEM disagree about the model."
            )
            entries.append((met_index[sid], weight))
            described[aa].append((compartment, sid, weight))
        assert entries, f"no compartment left for {aa!r} under space {space!r}."
        rows.append(entries)
    width = max(len(r) for r in rows)
    indices = torch.zeros(len(rows), width, dtype=torch.long)
    log_w = torch.full((len(rows), width), float("-inf"))
    for i, entries in enumerate(rows):
        for j, (idx, weight) in enumerate(entries):
            indices[i, j] = idx
            log_w[i, j] = math.log(weight)
    return indices, log_w, described


def mixture_log10_molar(
    log_c: torch.Tensor, indices: torch.Tensor, log_w: torch.Tensor
) -> torch.Tensor:
    """``[B, m]`` natural-log concentrations -> ``[B, 19]`` log10 whole-cell molar."""
    per_compartment = log_c[:, indices.reshape(-1)].reshape(
        log_c.shape[0], indices.shape[0], indices.shape[1]
    )
    return torch.logsumexp(per_compartment + log_w.unsqueeze(0), dim=-1) / LN10


#: Merzbacher 2025's released yeast essentiality split, the file the comparator numbers were
#: computed on. Columns `knockout`, `essential`, `test`.
FCL_ESSENTIALITY_SPLIT = (
    "data/merzbacher2025_fcl/deletionprediction-main/data/"
    "yeast_essentiality_test_split.csv"
)
#: THE COLUMN NAMED `essential` CARRIES 1.0 FOR A NONESSENTIAL GENE, and nothing in the file
#: says so. Counted on the released CSV 2026-09-14: 1,159 rows, of which `test=0` holds 772
#: at 1.0 and 126 at 0.0, and `test=1` holds 192 at 1.0 and 31 at 0.0. The paper's Methods
#: describe exactly those splits with the labels the other way round: "a classstratified
#: $20 \%$ of deletions (192 nonessential; 31 essential) was held out as a test set" and
#: "The remaining $80 \%$ of deletions (772 nonessential; 126 essential)". Supplementary
#: Table S4 agrees at the total, N=964 nonessential and N=157 essential. So the column is a
#: VIABILITY flag and reading it at face value inverts every label, which a classifier will
#: happily fit and report as a plausible AUROC.
FCL_ESSENTIAL_VALUE = 0.0
#: What the counts must be, so a re-released file with different contents is caught rather
#: than silently changing the comparison.
FCL_EXPECTED_COUNTS = {
    ("0", "nonessential"): 772,
    ("0", "essential"): 126,
    ("1", "nonessential"): 192,
    ("1", "essential"): 31,
}


def fcl_train_pos_weight() -> float:
    """Nonessential over essential on the released TRAINING side, 772 / 126."""
    return (
        FCL_EXPECTED_COUNTS[("0", "nonessential")]
        / FCL_EXPECTED_COUNTS[("0", "essential")]
    )


class FclEssentialityComparators(BaseModel):
    r"""Merzbacher 2025's published YEAST numbers, read from the mirrored paper.

    Recorded here as data rather than as prose so a result file carries the comparison it
    claims to make. Source: the mirrored `si/si1.md`, Supplementary Table S5 (the table is
    at line 33 and its caption at line 35, "Classification performance metrics for Flux Cone
    Learning (FCL) and Flux Balance Analysis (FBA) in yeast (S. cerevisiae) and Chinese
    Hamster Ovary (CHO) cells (Figure 3)"), with the protocol from `paper.md` line 122.

    NOT the 93.5 and 95 percent figures: those are the E. COLI accuracies from `paper.md`
    line 54 ("FBA delivers a maximal accuracy of $9 3 . 5 \\%$ correctly predicted genes for
    E. coli growing aerobically in glucose" and "achieving an average $9 5 \\%$ accuracy for
    all test genes across training repeats"). Yeast is a harder target and the gap is much
    narrower there.
    """

    model_config = ConfigDict(extra="forbid")

    source: str = (
        "$DATA_ROOT/torchcell-library/merzbacherAccuratePredictionGene2025/si/si1.md, "
        "Supplementary Table S5 (line 33; caption line 35); protocol paper.md line 122"
    )
    fba_accuracy: float = 0.81
    fba_precision: float = 0.91
    fba_recall: float = 0.97
    fba_roc_auc: float = 0.69
    fcl_cv_accuracy: float = 0.90
    fcl_cv_roc_auc: float = 0.78
    fcl_cv_roc_auc_sd: float = 0.065
    fcl_test_accuracy: float = 0.89
    fcl_test_precision: float = 0.89
    fcl_test_recall: float = 0.99
    fcl_test_roc_auc: float = 0.73
    #: 192 of the 223 released test genes are nonessential.
    majority_class_rate_test: float = 192.0 / 223.0
    note: str = (
        "FBA's accuracy of 0.81 is BELOW the 0.861 majority-class rate on the same test "
        "genes, so accuracy is the wrong headline here and ROC AUC is the comparator: "
        "0.69 for FBA against 0.73 for FCL on the released test split. The FBA baseline is "
        "cobrapy `single_gene_deletion` for all genes with glucose as the carbon source "
        "and the standard biomass reaction (paper.md line 122)."
    )


def read_fcl_essentiality_split(path: str) -> tuple[dict[str, int], dict[str, Any]]:
    """``({gene -> 1 if essential else 0}, report)`` from the released FCL split.

    The label inversion described at :data:`FCL_ESSENTIAL_VALUE` is applied here, ONCE, and
    the resulting class counts are checked against the paper's own Methods sentence. A file
    whose counts disagree is refused rather than used, because every number downstream is a
    comparison against a table computed on these exact genes.
    """
    import csv

    labels: dict[str, int] = {}
    test_genes: set[str] = set()
    counts: dict[tuple[str, str], int] = {}
    unlabeled = 0
    with open(path) as handle:
        for row in csv.DictReader(handle):
            gene = str(row["knockout"]).strip()
            raw = str(row["essential"]).strip()
            if not raw:
                unlabeled += 1
                continue
            essential = int(float(raw) == FCL_ESSENTIAL_VALUE)
            labels[gene] = essential
            split = str(row["test"]).strip()
            if split == "1":
                test_genes.add(gene)
            key = (split, "essential" if essential else "nonessential")
            counts[key] = counts.get(key, 0) + 1
    assert counts == FCL_EXPECTED_COUNTS, (
        f"the released split at {path} has class counts {counts}, not "
        f"{FCL_EXPECTED_COUNTS}. Those counts are the paper's own Methods sentence (192 "
        "nonessential and 31 essential held out; 772 and 126 kept), and every comparator "
        "number is computed on exactly these genes, so a file that disagrees is a "
        "different experiment."
    )
    report = {
        "path": path,
        "n_labeled": len(labels),
        "n_unlabeled_rows": unlabeled,
        "n_test_genes": len(test_genes),
        "n_essential": sum(labels.values()),
        "n_nonessential": len(labels) - sum(labels.values()),
        "label_convention": (
            "the CSV column `essential` is a VIABILITY flag: 1.0 marks a NONESSENTIAL "
            "gene. It is inverted here once, and the counts are asserted against the "
            "paper's Methods."
        ),
    }
    return labels, report


def load_split_file(name: str) -> dict[str, Any]:
    """A split file written by ``build_essentiality_splits.py``, by name or path."""
    import gzip

    path = name if osp.isabs(name) else osp.join(RESULTS_DIR, "splits", name)
    assert osp.isfile(path), f"no split file at {path}"
    with gzip.open(path, "rt") as fh:
        split = json.load(fh)
    for key in ("labels", "train", "val", "test", "report"):
        assert key in split, f"{path} has no `{key}`"
    split["path"] = path
    return split


def essentiality_gene_labels(
    dataset: Any, split_file: str | None = None
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """``(value, known, report)`` over the cell graph's gene nodes, from the released split.

    The label is Merzbacher 2025's, inverted once by `read_fcl_essentiality_split` and
    asserted against the paper's Methods counts. A gene the split does not name is `known`
    False and is masked out of the loss, the selection metric and the test score, so the
    arms are scored on exactly the genes the comparator table was computed on.
    """
    if split_file is None:
        path = osp.join(DATA_ROOT, FCL_ESSENTIALITY_SPLIT)
        labels, split_report = read_fcl_essentiality_split(path)
    else:
        # THE SPLIT FILE'S LABELS, so a cell's labels and its partition are one artifact
        # and the released held-out genes carry the released labels either way (the
        # builder lets the released label win on every disagreement).
        split = load_split_file(split_file)
        labels = {str(g): int(v) for g, v in split["labels"].items()}
        split_report = {"path": split["path"], **split["report"]}
    node_ids = list(dataset.cell_graph["gene"].node_ids)
    position = {name: i for i, name in enumerate(node_ids)}
    value = torch.zeros(len(node_ids))
    known = torch.zeros(len(node_ids), dtype=torch.bool)
    found: list[str] = []
    absent: list[str] = []
    for gene, essential in labels.items():
        index = position.get(gene)
        if index is None:
            absent.append(gene)
            continue
        found.append(gene)
        value[index] = float(essential)
        known[index] = True
    found_essential = sum(1 for g in found if labels[g] == 1)
    absent_essential = sum(1 for g in absent if labels[g] == 1)
    report = {
        "split": split_report,
        "requested": len(labels),
        "found": len(found),
        "absent": len(absent),
        "found_essential": found_essential,
        "found_nonessential": len(found) - found_essential,
        "absent_essential": absent_essential,
        "absent_nonessential": len(absent) - absent_essential,
        "note": (
            "the label is joined per genotype by its deleted gene; a gene the split does "
            "not name is masked out of the loss, the selection metric and the test score."
        ),
    }
    print(
        f"[essentiality] split labels on {len(found)}/{len(labels)} cell-graph genes "
        f"({found_essential} essential, {len(found) - found_essential} nonessential); "
        f"{len(absent)} absent ({absent_essential} essential)",
        flush=True,
    )
    return value, known, report


def resolve_fcl_test_indices(dataset: Any) -> tuple[set[int], dict[str, Any]]:
    """Record indices of the genes Merzbacher held out, pinned as OUR test split.

    The comparator numbers are computed on those 223 genes, so the comparison is only a
    comparison if they are the held-out set here too. Train and validation come from the
    remaining labeled genes through the data module's own seeded draw.
    """
    import csv

    path = osp.join(DATA_ROOT, FCL_ESSENTIALITY_SPLIT)
    test_genes = {
        str(row["knockout"]).strip()
        for row in csv.DictReader(open(path))
        if str(row["test"]).strip() == "1" and str(row["essential"]).strip()
    }
    gene_index = dataset.is_any_deletion_gene_index
    indices: set[int] = set()
    found: list[str] = []
    for gene in sorted(test_genes):
        hits = gene_index.get(gene)
        if hits:
            found.append(gene)
            indices.update(int(h) for h in hits)
    fraction = len(indices) / len(dataset)
    assert indices, (
        "none of the released test genes resolve in this build, so the comparison has no "
        "held-out set."
    )
    assert fraction < 0.5, (
        f"the FCL test genes select {fraction:.1%} of the dataset "
        f"({len(indices)}/{len(dataset)}), which cannot be right for 223 genes."
    )
    report = {
        "path": path,
        "requested": len(test_genes),
        "found": len(found),
        "absent": sorted(test_genes - set(found)),
        "records": len(indices),
        "fraction_of_dataset": fraction,
    }
    print(
        f"[fcl-test] {len(found)}/{len(test_genes)} released test genes present -> "
        f"{len(indices)} records pinned to TEST ({fraction:.1%} of {len(dataset)})",
        flush=True,
    )
    return indices, report


def binary_auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank-based AUROC over the observed rows; ``nan`` when one class is absent.

    The Mann-Whitney form, so ties are handled by mid-ranks rather than by a sweep over
    thresholds, and the number matches the probability that a random essential gene scores
    above a random nonessential one.
    """
    from scipy.stats import rankdata

    keep = np.isfinite(score) & np.isfinite(label)
    s, y = score[keep], label[keep]
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0.0 or n_neg == 0.0:
        return float("nan")
    ranks = rankdata(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def scored_width(spec: dict[str, Any]) -> int:
    """How many columns a head's target has AFTER the decode, which is what is scored.

    A head that names a ``column`` reads one column of a wider decoded block, so its
    prediction and its target are width 1 whatever the block's width.
    """
    return 1 if spec.get("column") is not None else int(spec["width"])


def heldout_rows(batch: Any, flags: torch.Tensor) -> torch.Tensor:
    """``[B]`` bool: which genotypes of this batch delete a HELD-BACK gene.

    ``flags`` is one bool per gene node of the cell graph, so the membership test runs on
    the same integer indices the model perturbs with, rather than on gene-name strings that
    the batch does not carry. A genotype is held back if ANY of its perturbed genes is.
    """
    gene = batch["gene"]
    indices = gene.perturbation_indices
    hit = flags.to(indices.device)[indices]
    out = torch.zeros(int(batch.num_graphs), dtype=torch.bool, device=indices.device)
    out[gene.perturbation_indices_batch[hit]] = True
    return out


def gene_keyed_target(
    batch: Any, value: torch.Tensor, known: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``([B, 1], [B, 1])`` target and mask from a PER-GENE label, by deleted gene.

    The essentiality target is not in the graph. It comes from Merzbacher 2025's released
    split, which is keyed on one knockout per row, so the label is looked up per genotype
    through the gene its deletion names. `value` and `known` are one entry per gene node of
    the cell graph, so the lookup runs on the same integer indices the model perturbs with
    rather than on gene-name strings the batch does not carry.

    THE DELETION INDEX, not the perturbation index: an addition is not a knockout, and a
    genotype is keyed by what it removes. Genotypes whose gene is absent from the split, or
    which do not delete exactly one gene, arrive masked False and contribute no loss, no
    selection metric and no test score.
    """
    gene = batch["gene"]
    indices = gene.deletion_indices
    assignment = gene.deletion_indices_batch
    n = int(batch.num_graphs)
    device = indices.device
    counts = torch.zeros(n, dtype=torch.long, device=device)
    counts.index_add_(0, assignment, torch.ones_like(assignment))
    # On the BATCH's device, not the label's. `value` and `known` are built once on the host
    # from the released split and never move; the indices that address them come off the
    # batch, so a target allocated beside the labels is a CPU tensor being indexed by CUDA
    # rows. Every other head's target is already on the batch device, which is where the
    # loss reads it.
    target = torch.zeros(n, 1, device=device)
    mask = torch.zeros(n, 1, dtype=torch.bool, device=device)
    single = counts == 1
    if bool(single.any()):
        # The one deleted gene of each single-deletion genotype, in batch-row order.
        rows = assignment[single[assignment]]
        genes = indices[single[assignment]]
        known_here = known.to(device)[genes]
        target[rows[known_here], 0] = value.to(device)[genes[known_here]]
        mask[rows[known_here], 0] = True
    return target, mask


def targets_in_space(
    batch: Any, head_specs: dict[str, dict[str, Any]], log10_molar: bool
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """``extract_targets``, then the per-head column slice and change of variable.

    Mulleder's values are stored as ``intracellular_concentration_mM``. The concentration
    readout predicts ``log10`` of a molar concentration, so its target has to live there
    too, and the change of variable belongs at the ONE place targets enter the run rather
    than at each of the five that consume them.

    Unobserved entries arrive as exact zeros under a False mask, and ``log10(0)`` is
    ``-inf``, which multiplied by a zero mask is NaN rather than zero. They are replaced by
    1 before the log, which the mask then discards. Observed entries must be strictly
    positive; a non-positive measured concentration is a data error and is asserted, not
    clipped.

    TWO PER-HEAD KEYS, both absent from every head spec before wave 9, so the decode of an
    earlier cell is untouched:

    ``column``
        take one column of the decoded block as the head's whole target. This is how the
        isobutanol biosensor is read: on ``fig6_build`` it was merged into the Mulleder
        record, so it is column ``ISOBUTANOL_COLUMN`` of a width-20 ``metabolite_level``
        group rather than a width-1 group of its own (see ``ISOBUTANOL_BLOCK_KEYS``).

    ``transform``
        ``"log10"`` puts the target in log10 of its STORED unit, with no unit conversion.
        The biosensor number is a fold change, which is multiplicative (sd 0.213 in log10
        over the screen), so CRPS on a Gaussian is well posed there and not in linear FC.
        Spearman is invariant to the monotone change, so a log10 arm and a linear arm are
        directly comparable on the primary endpoint.
    """
    # A GENE-KEYED head reads no phenotype from the batch at all, so `extract_targets` is
    # not called for it: its target is a per-gene label joined by the deleted gene name.
    gene_keyed = {
        name: spec for name, spec in head_specs.items() if "gene_label" in spec
    }
    decoded = {
        name: spec for name, spec in head_specs.items() if "gene_label" not in spec
    }
    raw: dict[str, tuple[torch.Tensor, torch.Tensor]] = (
        extract_targets(batch, decoded) if decoded else {}
    )
    for name, spec in gene_keyed.items():
        raw[name] = gene_keyed_target(batch, spec["gene_label"], spec["gene_known"])
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, (t, m) in raw.items():
        spec = head_specs[name]
        column = spec.get("column")
        if column is not None:
            t, m = t[:, column : column + 1], m[:, column : column + 1]
        flags = spec.get("heldout_gene_flags")
        if flags is not None:
            # THE HELD-BACK STRAINS, masked at the one place targets enter the run so no
            # consumer can forget: the loss, the validation curve, the test score and the
            # calibration pass all read this mask.
            m = m & ~heldout_rows(batch, flags).unsqueeze(1)
        transform = spec.get("transform")
        if transform == "log10":
            assert bool((t[m] > 0).all()), (
                f"head {name!r} carries a non-positive observed value, which has no "
                "log10. The Lopez biosensor column is a fold change and is positive by "
                "construction (measured minimum 0.0386 over the 4,244 fig6_build blocks)."
            )
            out[name] = (torch.log10(torch.where(m, t, torch.ones_like(t))), m)
            continue
        assert transform is None, (
            f"head {name!r} names transform {transform!r}, which is not implemented."
        )
        if not log10_molar:
            out[name] = (t, m)
            continue
        assert bool((t[m] > 0).all()), (
            f"head {name!r} carries a non-positive observed concentration, which has no "
            "log10. The Mulleder column is an absolute concentration in mM."
        )
        safe = torch.where(m, t, torch.ones_like(t))
        out[name] = (torch.log10(safe / MM_PER_M), m)
    return out


def arm_names() -> list[str]:
    """Every arm this experiment defines, in a FIXED order.

    The order is load-bearing: the shard rule assigns worker ``w`` the setting at index
    ``w % n_settings``, so a reordering silently reassigns work between workers mid-experiment.
    Sorted, so it cannot drift with filesystem order.
    """
    return sorted(
        f[len("arm_") : -len(".yaml")]
        for f in os.listdir(CONF_DIR)
        if f.startswith("arm_") and f.endswith(".yaml")
    )


def load_config() -> BxConfig:
    with open(osp.join(CONF_DIR, "base.yaml")) as fh:
        return BxConfig(**yaml.safe_load(fh))


def build_cell(
    arm: str,
    seed: int,
    cfg: BxConfig,
    epochs: int | None = None,
    crps: bool = False,
    graph_reg: bool = False,
    abort_on_collapse: bool = False,
    aux_mulleder: bool = False,
    attention_mask: bool = False,
    pert_heads: int = 6,
    layers: int | None = None,
    hidden: int | None = None,
    attention_heads: int | None = None,
    preprocessor_layers: int = 2,
    learnable_embedding: bool = False,
    kcat_tier: str = "measured",
    thermo_source: str = "shipped",
    constraint_warmup_epochs: int = 0,
    constraint_scale: float = 1.0,
    constraint_delay_epochs: int = 0,
    head_normalize: bool = False,
    head: str = HEAD,
    aa_readout: str = AA_READOUT_TURNOVER,
    dist: str = "crps",
    conc_space: str = CONC_SPACE_MIXTURE,
    volume_table: str = VOLUMES_ASSUMED,
    crps_sigma_fixed: float | None = None,
    bx_readout: str = BX_READOUT_FLUX_SCALAR,
    conc_lower_m: float = CONC_LOWER_M,
    conc_upper_m: float = CONC_UPPER_M,
    flux_density_weight: float = 0.0,
    flux_balance_active_weight: float = 0.0,
    flux_parsimony_weight: float = 0.0,
    flux_thermo_exempt_path: str | None = None,
    flux_log_c_init_path: str | None = None,
    flux_availability_fold: str = "softmin",
    flux_shuffle_kcat_seed: int | None = None,
    flux_parameterization: str | None = None,
    flux_reference_table: str | None = None,
    flux_alpha_frozen: bool = False,
    flux_warm_start_init_rel: float = 0.01,
    flux_draws: int = 1,
    flux_latent_stochastic: bool = False,
    flux_kl_weight: float | None = None,
    flux_log_c_from_latent: bool = False,
    ib_readout: str = IB_READOUT_FLUX_SCALAR,
    ib_anchors: str = IB_ANCHORS_ONE,
    aa_supervise_columns: list[str] | None = None,
    apply_media: str = MEDIA_NONE,
    lr: float | None = None,
    weight_decay: float | None = None,
    batch_size: int | None = None,
    split_file: str | None = None,
    ess_readout: str = ESS_READOUT_BIOMASS,
    ess_fba_media: list[str] | None = None,
    ess_solve_features: str = "none",
    ess_solve_encoding: str = "standardized",
    ess_solve_zero_init: bool = False,
    ess_solve_head_hidden: int = 0,
    flux_capacity_scale: float = 1.0,
    flux_lp_slack_penalty: float = 0.0,
    flux_pool_by_construction: bool = False,
    flux_lr_mult: float = 1.0,
    flux_abundance_head_layers: int = 1,
    flux_abundance_log_range: float = 3.0,
    ess_flux_head_reactions: str = "all",
    node_embeddings: list[str] | None = None,
    pos_weight_balanced: bool = False,
) -> BxCell:
    """Merge ``base.yaml`` with one arm file into a fully specified cell."""
    with open(osp.join(CONF_DIR, f"arm_{arm}.yaml")) as fh:
        armcfg = ArmConfig(**yaml.safe_load(fh))
    assert aa_readout in AA_READOUTS, (
        f"--aa-readout must be one of {AA_READOUTS}, got {aa_readout!r}."
    )
    assert ib_readout in IB_READOUTS, (
        f"--ib-readout must be one of {IB_READOUTS}, got {ib_readout!r}."
    )
    assert ib_anchors in IB_ANCHOR_SETS, (
        f"--ib-anchors must be one of {IB_ANCHOR_SETS}, got {ib_anchors!r}."
    )
    assert apply_media in APPLY_MEDIA_CHOICES, (
        f"--apply-media must be one of {APPLY_MEDIA_CHOICES}, got {apply_media!r}."
    )
    if aa_supervise_columns is not None:
        unknown = sorted(set(aa_supervise_columns) - set(AA_KEYS))
        assert not unknown, (
            f"--aa-supervise-columns names {unknown}, which are not Mulleder columns. "
            f"The 19 are {AA_KEYS}."
        )
        assert len(set(aa_supervise_columns)) == len(aa_supervise_columns), (
            "--aa-supervise-columns repeats a column."
        )
        assert 0 < len(aa_supervise_columns) < len(AA_KEYS), (
            f"--aa-supervise-columns holds out "
            f"{len(AA_KEYS) - len(aa_supervise_columns)} of {len(AA_KEYS)} columns; a "
            "mask that holds out none or all is not an experiment."
        )
        assert head == AUX_HEAD, (
            f"--aa-supervise-columns masks the amino-acid loss and needs --head "
            f"{AUX_HEAD}; this cell is on head {head!r}."
        )
    assert conc_space in CONC_SPACES, (
        f"--conc-space must be one of {CONC_SPACES}, got {conc_space!r}."
    )
    assert volume_table in VOLUME_TABLES, (
        f"--volume-table must be one of {VOLUME_TABLES}, got {volume_table!r}."
    )
    assert bx_readout in BX_READOUTS, (
        f"--bx-readout must be one of {BX_READOUTS}, got {bx_readout!r}."
    )
    assert dist in DISTS, f"--dist must be one of {DISTS}, got {dist!r}."
    width = hidden if hidden is not None else cfg.model.hidden
    n_heads = (
        attention_heads if attention_heads is not None else cfg.model.attention_heads
    )
    # BOTH HEAD COUNTS DIVIDE THE WIDTH, and the second one is easy to forget. The
    # transformer's self-attention uses `attention_heads`; the perturbation TRANSFORM's
    # cross-attention (the equivariant operator that injects the deletion, not the readout
    # head, which 027 no longer builds) uses `pert_heads`, which defaults to 6 and is NOT
    # the same number. Jobs 2039 to 2041 asked for width 45 with 9 attention heads, which
    # divides, and died 17 seconds in on `embed_dim must be divisible by num_heads` from
    # that cross-attention, because 45 is not a multiple of 6. A width must be a multiple
    # of the least common multiple of the two: with the defaults 9 and 6 that is 18, so
    # the ladder runs 90, 54, 36 and 18 unless the head counts move with it.
    for label, count in (("attention head count", n_heads), ("pert_heads", pert_heads)):
        assert width % count == 0, (
            f"the transformer width {width} is not divisible by the {label} {count}. "
            f"Both attention head counts must divide the width: --attention-heads "
            f"({n_heads}) for the transformer and --pert-heads ({pert_heads}) for the "
            "perturbation transform's cross-attention. With the defaults the width must "
            "be a multiple of 18."
        )
    assert armcfg.base_arm in ARMS, (
        f"arm {arm!r} names base_arm={armcfg.base_arm!r}, which is not in 026's ARMS "
        f"registry ({sorted(ARMS)}). The registry is the single source of truth for what "
        "the flux layer does; this file never redefines it."
    )
    return BxCell(
        arm=arm,
        base_arm=armcfg.base_arm,
        seed=seed,
        permute_train_targets=armcfg.permute_train_targets,
        epochs=epochs if epochs is not None else cfg.train.epochs,
        hidden=hidden if hidden is not None else cfg.model.hidden,
        layers=layers if layers is not None else cfg.model.layers,
        attention_heads=(
            attention_heads
            if attention_heads is not None
            else cfg.model.attention_heads
        ),
        preprocessor_layers=preprocessor_layers,
        batch_size=batch_size if batch_size is not None else cfg.model.batch_size,
        num_workers=int(os.getenv("NUM_WORKERS", str(cfg.train.num_workers))),
        lr=lr if lr is not None else cfg.model.lr,
        weight_decay=(
            weight_decay if weight_decay is not None else cfg.model.weight_decay
        ),
        split_seed=cfg.data.split_seed,
        split_file=split_file,
        ess_readout=ess_readout,
        ess_fba_media=list(ess_fba_media) if ess_fba_media else list(FBA_MEDIA),
        ess_solve_features=ess_solve_features,
        ess_solve_encoding=ess_solve_encoding,
        ess_solve_zero_init=ess_solve_zero_init,
        ess_solve_head_hidden=ess_solve_head_hidden,
        flux_capacity_scale=flux_capacity_scale,
        flux_lp_slack_penalty=flux_lp_slack_penalty,
        flux_pool_by_construction=flux_pool_by_construction,
        flux_lr_mult=flux_lr_mult,
        flux_abundance_head_layers=flux_abundance_head_layers,
        flux_abundance_log_range=flux_abundance_log_range,
        ess_flux_head_reactions=ess_flux_head_reactions,
        node_embeddings=(
            list(node_embeddings) if node_embeddings else list(STRONG_NODE_EMBEDDINGS)
        ),
        pos_weight_balanced=pos_weight_balanced,
        select_smoothing_window=cfg.scoring.select_smoothing_window,
        crps=crps,
        graph_reg=graph_reg,
        abort_on_collapse=abort_on_collapse,
        aux_mulleder=aux_mulleder,
        attention_mask=attention_mask,
        pert_heads=pert_heads,
        learnable_embedding=learnable_embedding,
        kcat_tier=kcat_tier,
        thermo_source=thermo_source,
        constraint_warmup_epochs=constraint_warmup_epochs,
        constraint_scale=constraint_scale,
        constraint_delay_epochs=constraint_delay_epochs,
        head_normalize=head_normalize,
        head=head,
        aa_readout=aa_readout,
        # DERIVED, not a switch of its own: the concentration readout emits a log10 molar
        # concentration, so the target it is scored against has to be in that space, and no
        # other readout may silently change the space its numbers are reported in.
        target_log10_molar=aa_readout in AA_LOG10_MOLAR_READOUTS,
        # DERIVED for the same reason: the biosensor target is a fold change and is trained
        # and scored in log10 of it, on every isobutanol cell and no other.
        target_log10=head == ISOBUTANOL_HEAD,
        dist=dist,
        conc_space=conc_space,
        volume_table=volume_table,
        crps_sigma_fixed=crps_sigma_fixed,
        bx_readout=bx_readout,
        conc_lower_m=conc_lower_m,
        conc_upper_m=conc_upper_m,
        # The channel comes from the arm FILE, which is where a parameterization belongs,
        # unless the command line names one (round 7 moves it per cell to test the box-
        # ending projection against the balance-ending one); the environment variable is
        # read only when neither names one.
        flux_parameterization=flux_parameterization or armcfg.flux_parameterization,
        flux_balance_active_row_set=armcfg.flux_balance_active_row_set,
        flux_density_weight=flux_density_weight,
        flux_balance_active_weight=flux_balance_active_weight,
        flux_parsimony_weight=flux_parsimony_weight,
        flux_thermo_exempt_path=flux_thermo_exempt_path,
        flux_log_c_init_path=flux_log_c_init_path,
        flux_availability_fold=flux_availability_fold,
        flux_shuffle_kcat_seed=flux_shuffle_kcat_seed,
        flux_reference_table=flux_reference_table,
        flux_alpha_frozen=flux_alpha_frozen,
        flux_warm_start_init_rel=flux_warm_start_init_rel,
        flux_draws=flux_draws,
        flux_latent_stochastic=flux_latent_stochastic,
        flux_kl_weight=flux_kl_weight,
        flux_log_c_from_latent=flux_log_c_from_latent,
        ib_readout=ib_readout,
        ib_anchors=ib_anchors,
        aa_supervise_columns=aa_supervise_columns,
        apply_media=apply_media,
    )


def resolve_pinned_test(dataset: Any, cfg: BxConfig) -> tuple[set[int], dict[str, Any]]:
    """Merzbacher 2025's 639 test ORFs -> record indices in our build.

    Lifted from ``train_cgt_multitask._pinned_test_indices`` including its two guards, both
    of which were found the hard way there:

    * the mapping goes through ``is_any_deletion_gene_index``, NOT
      ``is_any_perturbed_gene_index``. The pigment cassettes are emitted as perturbations on
      every strain and three of their members are native ORFs (ARO4/YBR249C and ARO7/YPR060C
      in Cachera's betaxanthin cassette), two of which are in Merzbacher's test list. Under
      the perturbed-gene index the pin resolved 639 genes to 4,885 of 4,930 records and left
      28 training examples -- and trained anyway.
    * a requested gene absent from the built LMDB is COUNTED AND REPORTED. The Cachera build
      is stale w.r.t. the shared name resolver (issue #195) and the split file names the 10
      it loses; a comparison that quietly shrank its own test set would be reporting on a
      different gene set than it claims.
    """
    path = osp.join(EXPERIMENT_ROOT, cfg.data.pinned_test_split_file)
    with open(path) as fh:
        payload = json.load(fh)
    genes = payload["split"]["test"]
    gene_index = dataset.is_any_deletion_gene_index
    indices: set[int] = set()
    found: list[str] = []
    absent: list[str] = []
    for gene in genes:
        hits = gene_index.get(gene)
        if hits:
            found.append(gene)
            indices.update(int(h) for h in hits)
        else:
            absent.append(gene)
    frac = len(indices) / len(dataset)
    assert frac < 0.5, (
        f"pinned test set is {frac:.1%} of the dataset ({len(indices)}/{len(dataset)}). "
        "639 genes should select a small minority of records; this means the gene index is "
        "resolving cassette membership rather than deletions."
    )
    report = {
        "path": path,
        "requested": len(genes),
        "found": len(found),
        "absent": absent,
        "records": len(indices),
        "fraction_of_dataset": frac,
    }
    print(
        f"[pinned-split] {len(found)}/{len(genes)} genes present -> {len(indices)} records "
        f"pinned to TEST ({frac:.1%} of {len(dataset)}); {len(absent)} absent (issue #195)",
        flush=True,
    )
    return indices, report


def code_provenance() -> dict[str, str]:
    """sha256 of the files whose content decides what a cell computed.

    Git-independent on purpose: this worktree is edited concurrently by several agents and
    is not committed between launches, so a commit hash would name a tree that never
    existed on disk. Hashing the two files that define the protocol and the physics is what
    tells a cell launched before tonight's edits apart from one launched after.
    """
    import torchcell.metabolism.flux_layer as flux_layer_module

    paths = {
        "train_bx.py": osp.abspath(__file__),
        # From the imported module, not from a path built by hand: the hash then names the
        # file this process actually ran, even when PYTHONPATH points at another checkout.
        "flux_layer.py": osp.abspath(flux_layer_module.__file__),
    }
    out: dict[str, str] = {}
    for name, path in paths.items():
        with open(path, "rb") as fh:
            out[name] = hashlib.sha256(fh.read()).hexdigest()
        out[f"{name}_path"] = path
    return out


def _params_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    """CPU copy of the TRAINABLE parameters only.

    Not ``state_dict()``: the flux layer registers the stoichiometric matrix and the
    thermodynamic tables as buffers, and the selection ring holds five snapshots, so copying
    buffers would carry ~46 MB of constant S matrix five times per worker for nothing.
    """
    return {k: v.detach().to("cpu", copy=True) for k, v in model.named_parameters()}


#: Attribute name every readout uses to hold the backbone it wraps. Stripped from a
#: parameter path before grouping, because otherwise the wrapper IS the group.
WRAPPER_PREFIX = "model."


def _module_path(name: str) -> str:
    """The parameter's module key with the readout wrappers stripped off the front.

    EVERY flux readout stores the backbone as ``self.model``, so on a flux arm the raw
    parameter names all begin ``model.`` and grouping on the first component puts the
    embedding preprocessor, the transformer, the perturbation transform AND THE FLUX LAYER into
    one bucket called ``model``. Measured 2026-09-15 on a synthetic wrapper: the groups came
    back as ``{'model': ..., 'readout': ...}``, two numbers for a seven-module network. The
    pooled cells are unwrapped, which is why wave 11 looked correct and hid this.

    Nested wrappers are stripped repeatedly, so a readout wrapping a readout still resolves.
    """
    while name.startswith(WRAPPER_PREFIX):
        name = name[len(WRAPPER_PREFIX) :]
    return name.split(".", 1)[0]


def _module_grad_norms(model: nn.Module) -> dict[str, float]:
    """L2 norm of the gradient per TOP-LEVEL module, keyed the way `n_parameters` is.

    The key is the module component of the parameter name (``transformer_layers``,
    ``essentiality_head``, ``flux_layer``, ...) after the wrapper prefix is removed, so the
    logged curve lines up with the parameter-count block in the result file and the flux
    layer gets a line of its own. Parameters with no gradient this step (a frozen block, an
    unused head) are skipped rather than counted as zero, so a module that is absent from
    the report received no gradient at all, which is itself worth seeing.
    """
    squares: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        module = _module_path(name)
        sq = float(param.grad.detach().pow(2).sum())
        squares[module] = squares.get(module, 0.0) + sq
        if module == "flux_layer":
            # THE LAYER'S OWN PARTS, one level down (round 18): the abundance head, the
            # availability gate, the concentration parameters, so a module arm can say
            # which of them the gradient through the solve reaches.
            stripped = name
            while stripped.startswith(WRAPPER_PREFIX):
                stripped = stripped[len(WRAPPER_PREFIX) :]
            part = stripped.split(".", 2)[1]
            key = f"flux_layer.{part}"
            squares[key] = squares.get(key, 0.0) + sq
    return {k: v**0.5 for k, v in squares.items()}


def _restore(model: nn.Module, snap: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(snap[name].to(param.device))


def _select_epoch(val_curve: list[float], window: int) -> tuple[int, str]:
    """Epoch chosen by argmax of a CENTERED moving average of validation Pearson.

    Returns ``(epoch, rule)``. The rule string is written into the result so a short run --
    a canary, or a cell the wall clock truncated -- is never silently scored by a different
    rule than the one the design fixed.
    """
    finite = [v if np.isfinite(v) else -np.inf for v in val_curve]
    if len(finite) < window:
        return int(np.argmax(finite)), "raw_argmax_short_run"
    half = window // 2
    smoothed = [
        float(np.mean(finite[i - half : i + half + 1]))
        for i in range(half, len(finite) - half)
    ]
    return int(np.argmax(smoothed)) + half, f"centered_ma_{window}"


def _masked_pearson(pred: np.ndarray, true: np.ndarray) -> float:
    """Pearson on a 1-D pair, with the RELATIVE variance guard 026 arrived at.

    An absolute floor is the wrong test: a head collapsed onto a constant emits a residual
    spread of order 1e-11, which clears any absolute floor and then makes ``corrcoef`` divide
    by a number indistinguishable from zero, yielding NaN -- reported as "metric unavailable"
    when what happened is the documented mean-collapse failure. Scaling by the target's spread
    makes "collapsed" scale-free, and a collapsed head scores 0.0, which is what it earned.
    """
    finite = np.isfinite(pred) & np.isfinite(true)
    p, t = pred[finite], true[finite]
    if p.size < 10 or t.std() < 1e-12 or p.std() < 1e-8 * t.std():
        return 0.0
    r = float(pearsonr(p, t)[0])
    return r if np.isfinite(r) else 0.0


def _masked_spearman(pred: np.ndarray, true: np.ndarray) -> float:
    finite = np.isfinite(pred) & np.isfinite(true)
    p, t = pred[finite], true[finite]
    if p.size < 10 or t.std() < 1e-12 or p.std() < 1e-8 * t.std():
        return 0.0
    rho = float(spearmanr(p, t)[0])
    return rho if np.isfinite(rho) else 0.0


def _top_k_enrichment(
    pred: np.ndarray, true: np.ndarray, k: int, top_frac: float = 0.1
) -> dict[str, float]:
    """Of the k genes the model ranks highest, how many are truly in the top ``top_frac``.

    The imbalance-immune half of the 020 recipe, and the metric the task is actually about:
    accuracy on a 67%-majority problem actively selects against finding high producers.
    ``top_frac`` defines "high" on OUR measured values so the metric needs no external label
    file; the 3-class head-to-head against Merzbacher's released labels is a separate step,
    run by 020's own scripts on the per-gene dump this file writes.
    """
    finite = np.isfinite(pred) & np.isfinite(true)
    p, t = pred[finite], true[finite]
    if p.size < k:
        return {"hits": float("nan"), "enrichment": float("nan")}
    n_high = max(1, int(round(top_frac * t.size)))
    high = set(np.argsort(-t)[:n_high].tolist())
    top = np.argsort(-p)[:k].tolist()
    hits = sum(1 for i in top if i in high)
    base = n_high / t.size
    return {
        "hits": float(hits),
        "enrichment": float((hits / k) / base) if base else float("nan"),
    }


# ---------------------------------------------------------------------------------------
# FEASIBILITY (wave 6, 2026-09-10). The Genesis "Metabolic Grounding" endpoint: of the top
# 100 strains the model nominates as high producers, what fraction carries a flux vector
# that does not violate mass balance or the second law.
#
# Everything below is computed from the flux layer's OWN forward output and its OWN buffers
# and methods, so the per-sample numbers reduce to the scalars the training loop already
# logs (`c_balance`, `feas_balance_median`, `feas_box_violation_frac`, `c_thermo`,
# `feas_thermo_violation_frac`). Nothing here re-derives a definition.
# ---------------------------------------------------------------------------------------

#: The balance ladder, as a median relative residual over the independent rows of S.
BALANCE_TOLERANCES: tuple[float, ...] = (1e-1, 1e-2, 1e-3)
#: The second-law ladder, as a fraction of active reactions running uphill.
THERMO_TOLERANCES: tuple[float, ...] = (0.0, 0.01)
#: The single balance tolerance the "both" criterion is built on.
BOTH_BALANCE_TOLERANCE = 1e-2
#: How many top-ranked strains the Genesis metric is read on.
TOP_K_FEASIBLE = 100


class FeasibilityStats(BaseModel):
    """The distribution of one per-sample feasibility scalar over a group of genotypes."""

    model_config = ConfigDict(extra="forbid")

    median: float
    mean: float
    p90: float
    max: float


class FeasibilityGroup(BaseModel):
    """Feasibility over one group of genotypes: a whole split, or its top-ranked 100."""

    model_config = ConfigDict(extra="forbid")

    n: int
    rank_by: str | None = None
    stats: dict[str, FeasibilityStats]
    fractions: dict[str, float]


class FeasibilityBlock(BaseModel):
    """The feasibility audit of one cell at its selected epoch.

    ``available`` is False for an arm with no flux layer, and ``reason`` says so. A pooled
    arm has no flux vector at all, so writing zeros here would report perfect feasibility
    for a model that never made a feasibility claim.
    """

    model_config = ConfigDict(extra="forbid")

    available: bool
    reason: str | None = None
    definitions: dict[str, str] = Field(default_factory=dict)
    tolerances: dict[str, list[float]] = Field(default_factory=dict)
    n_independent_balance_rows: int | None = None
    n_reactions_second_law_applied: int | None = None
    thermo_mode: str | None = None
    thermo_source: str | None = None
    groups: dict[str, FeasibilityGroup] = Field(default_factory=dict)


FEASIBILITY_DEFINITIONS: dict[str, str] = {
    "balance_median_ratio": (
        "median over the independent rows i of S of |(Sv)_i| / (omega_i + balance_epsilon), "
        "with omega_i = 0.5 sum_j |S_ij v_j| the layer's own turnover. This is "
        "FluxLayer.forward's feas_balance_median, per sample instead of batch-reduced."
    ),
    "balance_max_ratio": "the same ratio, maximized over the independent rows.",
    "box_violation_frac": (
        "fraction of the 4,131 reactions with v > ub + 1e-4 or v < lb - 1e-4, where "
        "(lb, ub) is FluxLayer.dynamic_box at this sample's availability and abundance. "
        "The box parameterization satisfies this by construction, so a nonzero value is a "
        "bug rather than an infeasibility."
    ),
    "thermo_model_violation_frac": (
        "fraction of active reactions (delta_r_g0_mask and not thermo_exempt) with "
        "v * delta_r_G > thermo_epsilon, using the layer's OWN delta_r_G from "
        "reaction_potential, which includes the learned concentration term. Available only "
        "when the arm's thermo_mode is not OFF."
    ),
    "thermo_model_uphill_frac": (
        "mean over active reactions of relu(v dG + eps) / (|v||dG| + eps), the per-sample "
        "form of the layer's c_thermo. Available only when thermo_mode is not OFF."
    ),
    "thermo_standard_state_violation_frac": (
        "fraction of active reactions with v * (delta_r_G0 + delta_r_G_transport) > "
        "thermo_epsilon. An AUDIT REFERENCE, not the layer's own term: it evaluates the "
        "second law at standard state because an arm with thermo_mode OFF learns no "
        "concentrations, so the concentration term RT S^T ln c is absent rather than zero."
    ),
    "thermo_standard_state_uphill_frac": (
        "the same standard-state driving force in the c_thermo normalization."
    ),
    "fractions": (
        "fraction of the genotypes in the group meeting each named criterion. "
        "'both_balance_le_1e-02_and_<thermo>_eq_0' is the Genesis criterion: mass balance "
        "within 1 percent and no uphill active reaction."
    ),
    "rank_by": (
        "top-100 groups are the 100 genotypes with the HIGHEST predicted betaxanthin "
        "(denormalized head mean), i.e. the strains the model nominates as producers."
    ),
}


def _per_sample_feasibility(
    layer: Any, flux: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Per-sample feasibility scalars from one flux-layer forward output. ``[B]`` each."""
    cfg = layer.config
    v = flux["v"]
    residual = layer._s_matmul(v)[:, layer.independent_rows].abs()
    omega = layer.turnover(v)[:, layer.independent_rows]
    ratio = residual / (omega + cfg.balance_epsilon)
    out: dict[str, torch.Tensor] = {
        "balance_median_ratio": ratio.median(dim=1).values,
        "balance_max_ratio": ratio.max(dim=1).values,
    }
    # The box is genotype-dependent, so it has to be rebuilt at this sample's availability
    # and abundance rather than read off the static lb/ub buffers.
    e_u = (
        layer.unit_abundance(flux["e_g"])
        if (cfg.use_abundance_head and "e_g" in flux)
        else None
    )
    lb, ub, _ = layer.dynamic_box(flux["c_j"], e_u)
    out["box_violation_frac"] = ((v > ub + 1e-4) | (v < lb - 1e-4)).float().mean(dim=1)

    active = layer.delta_r_g0_mask & (~layer.thermo_exempt)
    if bool(active.any()):
        va = v[:, active]
        if "delta_r_g" in flux:
            ga = flux["delta_r_g"][:, active]
            drive = va * ga
            out["thermo_model_violation_frac"] = (
                (drive > cfg.thermo_epsilon).float().mean(dim=1)
            )
            out["thermo_model_uphill_frac"] = (
                torch.relu(drive + cfg.thermo_epsilon)
                / (va.abs() * ga.abs() + cfg.thermo_epsilon)
            ).mean(dim=1)
        ss = (layer.delta_r_g0 + layer.delta_r_g_transport)[active].unsqueeze(0)
        drive_ss = va * ss
        out["thermo_standard_state_violation_frac"] = (
            (drive_ss > cfg.thermo_epsilon).float().mean(dim=1)
        )
        out["thermo_standard_state_uphill_frac"] = (
            torch.relu(drive_ss + cfg.thermo_epsilon)
            / (va.abs() * ss.abs() + cfg.thermo_epsilon)
        ).mean(dim=1)
    return out


def collect_feasibility(
    model: nn.Module,
    loader: Any,
    dataset: Any,
    head_name: str,
    mean: torch.Tensor,
    sd: torch.Tensor,
) -> dict[str, np.ndarray]:
    """One forward pass over a loader collecting per-sample feasibility and prediction.

    The prediction is the DENORMALIZED head mean of the first output column, which for the
    betaxanthin head is the predicted production level the ranking is defined on.
    """
    device = next(model.parameters()).device
    layer: Any = model.flux_layer
    assert layer is not None, "collect_feasibility called on an arm with no flux layer."
    was_training = model.training
    model.eval()
    parts: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, reps = model(dataset.cell_graph.to(device), batch)
            per = _per_sample_feasibility(layer, reps["flux"])
            if head_name in SCALAR_HEADS:
                # The scalar the ranking is defined on: the head's mean, denormalized back
                # into the units the phenotype is reported in.
                #
                # ONE ROW PER DRAW, not per genotype, when the arm draws. Feasibility is a
                # property of a FLUX and there are K of those per genotype, while the
                # prediction is the draw ensemble and there is one; repeating it aligns the
                # two so that "the top 100 predicted strains" selects every draw of those
                # 100 genotypes. That is the claim a sampled arm has to answer, since a
                # genotype feasible on average and one feasible on every draw are different
                # statements.
                pred = _head_params(reps["head_outputs"][head_name])
                scalar = pred[:, 0] * sd[0] + mean[0]
                draws = int(layer.config.draws)
                per["prediction"] = (
                    scalar.repeat_interleave(draws, dim=0) if draws > 1 else scalar
                )
            for key, value in per.items():
                parts.setdefault(key, []).append(value.detach().cpu().numpy())
    model.train(was_training)
    return {k: np.concatenate(v) for k, v in parts.items()}


def _stats(x: np.ndarray) -> FeasibilityStats:
    return FeasibilityStats(
        median=float(np.median(x)),
        mean=float(np.mean(x)),
        p90=float(np.percentile(x, 90)),
        max=float(np.max(x)),
    )


def criteria_flags(per: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """PER-SAMPLE boolean feasibility under each named criterion.

    Separate from :func:`_fractions` because the stochastic audit needs the flags
    themselves: a genotype that is feasible on average and a genotype that is feasible on
    every draw are different claims, and only the flags can tell them apart.
    """
    out: dict[str, np.ndarray] = {}
    balance = per["balance_median_ratio"]
    for tol in BALANCE_TOLERANCES:
        out[f"balance_median_ratio_le_{tol:.0e}"] = balance <= tol
    out["box_violation_frac_eq_0"] = per["box_violation_frac"] <= 0.0
    for key, label in (
        ("thermo_model_violation_frac", "thermo_model"),
        ("thermo_standard_state_violation_frac", "thermo_standard_state"),
    ):
        if key not in per:
            continue
        for tol in THERMO_TOLERANCES:
            name = (
                f"{label}_violation_frac_eq_0"
                if tol == 0.0
                else f"{label}_violation_frac_le_{tol:.0e}"
            )
            out[name] = per[key] <= tol
        out[f"both_balance_le_{BOTH_BALANCE_TOLERANCE:.0e}_and_{label}_eq_0"] = (
            balance <= BOTH_BALANCE_TOLERANCE
        ) & (per[key] <= 0.0)
    return out


def _fractions(per: dict[str, np.ndarray]) -> dict[str, float]:
    """Fraction of the group meeting each tolerance. Explicit names, explicit thresholds."""
    return {k: float(v.mean()) for k, v in criteria_flags(per).items()}


def _group(
    per: dict[str, np.ndarray], sel: np.ndarray, rank_by: str | None
) -> FeasibilityGroup:
    sub = {k: v[sel] for k, v in per.items() if k not in {"prediction", "observed"}}
    return FeasibilityGroup(
        n=int(sel.sum()),
        rank_by=rank_by,
        stats={k: _stats(v) for k, v in sub.items()},
        fractions=_fractions(sub),
    )


def _top_k_mask(prediction: np.ndarray, sel: np.ndarray, k: int) -> np.ndarray:
    """Mask of the ``k`` highest-predicted members of ``sel``, descending."""
    idx = np.flatnonzero(sel)
    order = idx[np.argsort(-prediction[idx], kind="stable")][:k]
    mask = np.zeros_like(sel)
    mask[order] = True
    return mask


def build_feasibility_block(
    per_all: dict[str, np.ndarray],
    per_test: dict[str, np.ndarray],
    test_pinned: np.ndarray,
    layer: Any,
    head_name: str,
) -> FeasibilityBlock:
    """Assemble the feasibility JSON block from two completed passes."""
    rank_note = (
        f"predicted {head_name}, descending" if head_name in SCALAR_HEADS else None
    )
    groups: dict[str, FeasibilityGroup] = {
        "all": _group(
            per_all, np.ones(per_all["balance_median_ratio"].size, bool), None
        ),
        "pinned_test": _group(per_test, test_pinned, None),
    }
    if rank_note is not None:
        all_sel = np.ones(per_all["balance_median_ratio"].size, bool)
        groups["all_top100"] = _group(
            per_all,
            _top_k_mask(per_all["prediction"], all_sel, TOP_K_FEASIBLE),
            rank_note,
        )
        groups["pinned_test_top100"] = _group(
            per_test,
            _top_k_mask(per_test["prediction"], test_pinned, TOP_K_FEASIBLE),
            rank_note,
        )
    return FeasibilityBlock(
        available=True,
        reason=None,
        definitions=FEASIBILITY_DEFINITIONS,
        tolerances={
            "balance_median_ratio": list(BALANCE_TOLERANCES),
            "thermo_violation_frac": list(THERMO_TOLERANCES),
            "both_balance_median_ratio": [BOTH_BALANCE_TOLERANCE],
        },
        n_independent_balance_rows=int(layer.independent_rows.numel()),
        n_reactions_second_law_applied=int(
            (layer.delta_r_g0_mask & (~layer.thermo_exempt)).sum()
        ),
        thermo_mode=str(layer.config.thermo_mode),
        thermo_source=str(layer.config.thermo_source),
        groups=groups,
    )


def flatten_feasibility_for_wandb(block: FeasibilityBlock) -> dict[str, float]:
    """The feasibility fractions as flat scalars, for a W&B run summary."""
    out: dict[str, float] = {}
    for group, payload in block.groups.items():
        out[f"feas/{group}/n"] = float(payload.n)
        for name, value in payload.fractions.items():
            out[f"feas/{group}/{name}"] = value
        for name, stat in payload.stats.items():
            out[f"feas/{group}/{name}_median"] = stat.median
    return out


class ConcentrationReadoutBase(nn.Module):
    r"""What the concentration readouts share: ``[B, m]`` natural-log concentration in,
    ``[B, 19]`` (or ``[B, 19, 2]`` under CRPS) standardized prediction out.

    ONE PATH, TWO SOURCES OF ``log c``. ``FluxConcentrationReadout`` takes the flux layer's
    ``thermo_log_c``; ``ContextConcentrationReadout`` computes the same parameterization on
    the pooled backbone with no layer. Everything after the ``log c`` tensor is this class:
    the volume-weighted whole-cell mixture in log10 molar, the train-split standardization,
    and the per-column ``log_sigma``. Factored here rather than copied so the ablation
    cannot differ from the arm it ablates in anything but the source of ``log c``; the
    operations and their order are exactly those the wave-7 readout ran.
    """

    metabolite_indices: torch.Tensor
    mixture_log_weights: torch.Tensor
    target_mean: torch.Tensor
    target_sd: torch.Tensor
    #: Annotated at class level so mypy reads the concrete types rather than
    #: `nn.Module.__getattr__`'s `Tensor | Module`, which is not callable.
    model: CellGraphTransformerMetabolism
    log_sigma: torch.Tensor | None

    def __init__(
        self,
        model: CellGraphTransformerMetabolism,
        metabolite_indices: torch.Tensor,
        mixture_log_weights: torch.Tensor,
        crps: bool,
    ) -> None:
        """Wrap a built model and bind the readout to the amino acids' compartments.

        Args:
            model: The metabolism transformer.
            metabolite_indices: ``[19, C]`` long, the GEM metabolite row of every
                intracellular compartment species of each measured amino acid, in the
                column order the target uses.
            mixture_log_weights: ``[19, C]``, the natural log of each compartment's
                fractional volume, ``-inf`` on padding.
            crps: Whether the loss is a Gaussian CRPS, in which case a per-column
                ``log_sigma`` is created for the spread and the mean stays parameter-free.
        """
        super().__init__()
        width = int(metabolite_indices.shape[0])
        self.model = model
        self.register_buffer("metabolite_indices", metabolite_indices)
        self.register_buffer("mixture_log_weights", mixture_log_weights)
        # The trainer compares a head's output against the STANDARDIZED target, so the
        # readout has to speak that space. These are the train-split mean and sd of the
        # target in log10 molar, written in by `set_target_standardization` once the
        # trainer has computed them; the identity here is only what the module is built
        # with, and a run that forgot to set them would be a raw log10 prediction against
        # a standardized target, i.e. the offset defect wave 6 measured on the turnover
        # head. They are CONSTANTS from the training split, not parameters, so the
        # prediction is still a parameter-free image of the concentration.
        self.register_buffer("target_mean", torch.zeros(width))
        self.register_buffer("target_sd", torch.ones(width))
        self.register_parameter(
            "log_sigma", nn.Parameter(torch.zeros(width)) if crps else None
        )

    def set_target_standardization(self, mean: torch.Tensor, sd: torch.Tensor) -> None:
        """Write the train-split per-column mean and sd of the log10 molar target."""
        assert mean.shape == self.target_mean.shape, (
            f"target mean has shape {tuple(mean.shape)}, expected "
            f"{tuple(self.target_mean.shape)}."
        )
        with torch.no_grad():
            self.target_mean.copy_(mean.to(self.target_mean.device))
            self.target_sd.copy_(sd.to(self.target_sd.device))

    @property
    def flux_layer(self) -> Any:
        """The wrapped model's layer, so every caller that reaches for it is unchanged."""
        return self.model.flux_layer

    @property
    def num_parameters(self) -> dict[str, int]:
        """The wrapped model's counts, plus every parameter the readout itself owns.

        For the flux readout that is the per-column ``log_sigma`` when there is one; for
        the context readout it is ``log_sigma`` plus the concentration head.
        """
        counts: dict[str, int] = dict(self.model.num_parameters)
        extra = sum(int(p.numel()) for p in self.parameters()) - sum(
            int(p.numel()) for p in self.model.parameters()
        )
        counts[f"{AUX_HEAD}_head"] = extra
        counts["total"] = counts["total"] + extra
        return counts

    def _write_prediction(self, log_c: torch.Tensor, reps: dict[str, Any]) -> None:
        """``[B, m]`` natural-log concentration -> the standardized head output."""
        log10_c = mixture_log10_molar(
            log_c, self.metabolite_indices, self.mixture_log_weights
        )
        pred = (log10_c - self.target_mean.unsqueeze(0)) / self.target_sd.unsqueeze(0)
        if self.log_sigma is not None:
            pred = torch.stack(
                [pred, self.log_sigma.unsqueeze(0).expand_as(pred)], dim=-1
            )
        reps["head_outputs"][AUX_HEAD] = pred


class FluxConcentrationReadout(ConcentrationReadoutBase):
    r"""The amino-acid prediction IS the flux layer's own learned log concentration.

    WHAT IT REPLACES. ``FluxMetaboliteHead`` reads the TURNOVER of each amino acid's GEM
    metabolite and maps it through a free per-column affine, ``a_k log(1 + omega_k) + b_k``.
    Turnover is a flux, not a pool, so the affine is where the missing residence time is
    absorbed, and the readout can only ever be scored as a correlation. Mulleder 2016
    measures the pool itself, in mM.

    WHAT IT READS INSTEAD. In ``ThermoMode.ANCHORED`` the layer already learns a per-strain
    concentration for every metabolite, ``log c = lo + (hi - lo) sigmoid(base + W context)``
    with ``[lo, hi]`` the physiological window, and it is that tensor, through
    ``RT S^T ln c``, that sets every reaction energy and therefore the second law and the
    dissipation limit. This module emits ``log10(exp(log_c))`` at the 19 named metabolite
    rows and nothing else, so a gradient from the amino-acid loss lands on the same
    parameters the physics reads.

    NO AFFINE ON THE MEAN, deliberately. A learned per-column scale and offset would make
    the prediction an arbitrary monotone image of the concentration, which is the tie this
    arm exists to remove: with the affine, the arm measures whether the layer's context
    vector carries amino-acid information; without it, it measures whether the layer's
    CONCENTRATION is the measured concentration. The only parameter here is a per-column
    ``log_sigma``, created only under CRPS, which sets the width of the predictive
    distribution and never moves its mean, so the two arms it is compared against keep the
    same loss function.

    The model it wraps must have been built with an ANCHORED flux layer.
    """

    def forward(
        self, cell_graph: Any, batch: Any, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run the wrapped model, then write the concentration into ``head_outputs``.

        THE LAYER HAS TO BE CALLED HERE, and the reason is worth naming.
        ``CellGraphTransformerMetabolism.forward`` runs its flux layer only when at least
        one head reads it (``if self.flux_layer is not None and self.flux_head_names``),
        and this arm declares no head, so nothing would run the layer and ``reps`` would
        carry no ``flux`` at all. The call below is the model's own call, with the model's
        own layer and the same four arguments in the same order, so the layer's output --
        including every constraint residual the trainer then charges to the loss -- is
        identical to what a declared flux head would have produced.
        """
        predictions, reps = self.model(cell_graph, batch, *args, **kwargs)
        if "flux" not in reps:
            layer = self.model.flux_layer
            assert layer is not None, (
                "the concentration readout was built on a model with no flux layer."
            )
            # THE GATE READS THE DELETION-ONLY INDEX, the pool reads every perturbed
            # token: the context is a representation of what was changed, the gate is a
            # statement about what is gone. See `gene_removal_indices`.
            gate_indices, gate_batch, _ = gene_removal_indices(batch)
            reps["flux"] = layer(
                reps["H_genes_pert"],
                perturbed_gene_pool(
                    reps["H_genes_pert"],
                    batch["gene"].perturbation_indices,
                    batch["gene"].perturbation_indices_batch,
                ),
                gate_indices,
                gate_batch,
            )
        flux = reps["flux"]
        assert "thermo_log_c" in flux, (
            "the flux layer emitted no `thermo_log_c`, so its thermo_mode is not ANCHORED "
            "and there is no learned concentration to predict. This arm is only defined on "
            "an anchored layer."
        )
        self._write_prediction(flux["thermo_log_c"], reps)
        return predictions, reps


def anchored_log_concentration(
    base: torch.Tensor,
    from_context: nn.Linear,
    context: torch.Tensor,
    conc_lower_m: float,
    conc_upper_m: float,
) -> torch.Tensor:
    r"""``log c = lo + (hi - lo) sigmoid(base + W context)``, ``[B, d] -> [B, m]``.

    Term for term the ANCHORED branch of ``FluxLayer.reaction_potential`` with the latent
    term absent, in natural log of molar. The two context-only readouts own their own
    parameters and share this FORMULA, so the ablation and the arm it ablates cannot come
    to disagree about what the concentration parameterization is.
    """
    lo, hi = math.log(conc_lower_m), math.log(conc_upper_m)
    return lo + (hi - lo) * torch.sigmoid(base.unsqueeze(0) + from_context(context))


class ContextConcentrationReadout(ConcentrationReadoutBase):
    r"""The flux layer's concentration head, on the pooled backbone, with no flux layer.

    THE QUESTION IT ANSWERS (user, 2026-09-14): how do we know the metabolic module is ON
    for the amino-acid target? The concentration readout's gain over pooled (+0.015 mean
    pinned Spearman, n = 3 seeds, 200 epochs) is the module's only positive result there,
    and Finding 5 of [[experiments.027-betaxanthin-metabolic.wave8-sampling]] showed the
    readout predicts ``thermo_log_c = lo + (hi - lo) sigmoid(base + W context)``, a function
    of the genotype context alone: the flux never enters the prediction, and the layer
    touches the concentration only through the thermodynamic penalty that prices sign
    consistency between ``v`` and ``dG'(log c)``. At constraint scale 0 the gain was
    +0.020, so that coupling does not help. Hypothesis (untested until this arm runs): the
    whole gain is the concentration PARAMETERIZATION, and the module contributes nothing
    beyond it.

    THE CONSTRUCTION. ``log_c_base`` (zeros, one per GEM metabolite) and
    ``log_c_from_context = nn.Linear(d, m)`` (PyTorch default init), exactly as
    ``FluxLayer.__init__`` builds them in ``ThermoMode.ANCHORED``; ``log c = lo + (hi - lo)
    sigmoid(base + W context)`` with the window read at call time exactly as
    ``FluxLayer.reaction_potential`` reads it; the context is ``perturbed_gene_pool`` of the
    perturbed gene tokens, which is the tensor the metabolism model hands its layer as
    ``context``. ``m`` is the FULL GEM metabolite count (2,806), not the 19 amino acids'
    species, so the head is parameter-for-parameter the layer's concentration head; the
    rows the mixture never reads are trained by nothing and stay at their initialization.
    From ``log c`` on, the path is the shared base class, so the two readouts differ ONLY
    in whether a flux layer stands between the context and the concentration.

    WHAT IS ABSENT, by design: stoichiometry, kcat, thermodynamics, every constraint term,
    the flux vector. ``reps`` carries no ``"flux"`` key, so the trainer charges no
    constraint loss and the feasibility block reports null, as it does for ``pooled``.

    HOW TO READ THE RESULT. Paired by seed against the wave-7 concentration cells and the
    wave-7 pooled cells. Matching the concentration cells means the module is not on for
    this target; falling back to pooled means the module, through ``c_thermo`` at constraint
    scale 0.1, is what carries the gain.
    """

    log_c_base: torch.Tensor
    log_c_from_context: nn.Linear

    def __init__(
        self,
        model: CellGraphTransformerMetabolism,
        metabolite_indices: torch.Tensor,
        mixture_log_weights: torch.Tensor,
        crps: bool,
        hidden: int,
        n_metabolites: int,
        conc_lower_m: float,
        conc_upper_m: float,
    ) -> None:
        """Bind the head to the GEM's metabolite axis and the window.

        Args:
            model: The pooled backbone, built with NO flux layer.
            metabolite_indices: As in the base class, rows into the ``n_metabolites`` axis.
            mixture_log_weights: As in the base class.
            crps: As in the base class.
            hidden: ``d``, the width of the context vector.
            n_metabolites: ``m``, the GEM metabolite count the head has one row for.
            conc_lower_m: Floor of the concentration window, in molar.
            conc_upper_m: Ceiling of the window, in molar.
        """
        super().__init__(model, metabolite_indices, mixture_log_weights, crps)
        assert model.flux_layer is None, (
            "the context-only concentration readout was built on a model that carries a "
            "flux layer; the ablation is defined on the pooled backbone."
        )
        assert int(metabolite_indices.max()) < n_metabolites, (
            f"a mixture index {int(metabolite_indices.max())} is outside the head's "
            f"{n_metabolites} metabolite rows."
        )
        assert 0.0 < conc_lower_m < conc_upper_m, (
            f"window [{conc_lower_m}, {conc_upper_m}] M is not an increasing positive pair."
        )
        self.conc_lower_m = float(conc_lower_m)
        self.conc_upper_m = float(conc_upper_m)
        # The same two objects `FluxLayer.__init__` creates under ThermoMode.ANCHORED:
        # a zero base per metabolite and a default-initialized Linear from the context.
        self.log_c_base = nn.Parameter(torch.zeros(n_metabolites))
        self.log_c_from_context = nn.Linear(hidden, n_metabolites)

    def log_concentration(self, context: torch.Tensor) -> torch.Tensor:
        """``[B, d]`` context -> ``[B, m]`` natural-log molar, inside the window.

        Term for term the ANCHORED branch of ``FluxLayer.reaction_potential`` with the
        latent term absent.
        """
        return anchored_log_concentration(
            self.log_c_base,
            self.log_c_from_context,
            context,
            self.conc_lower_m,
            self.conc_upper_m,
        )

    def forward(
        self, cell_graph: Any, batch: Any, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run the pooled backbone, then write the concentration into ``head_outputs``.

        The context is the same tensor ``CellGraphTransformerMetabolism.forward`` hands
        its flux layer, built the same way from the same tokens. The natural-log
        concentration is left in ``reps["context_log_c"]`` so the concentration audit
        reads the readout's own tensor, as it reads ``thermo_log_c`` on a flux arm.
        """
        predictions, reps = self.model(cell_graph, batch, *args, **kwargs)
        assert "flux" not in reps, (
            "the context-only concentration readout received a flux output; the model "
            "was built with a layer, which this ablation excludes."
        )
        context = perturbed_gene_pool(
            reps["H_genes_pert"],
            batch["gene"].perturbation_indices,
            batch["gene"].perturbation_indices_batch,
        )
        log_c = self.log_concentration(context)
        reps["context_log_c"] = log_c
        self._write_prediction(log_c, reps)
        return predictions, reps


def _sampling_meta(model: nn.Module, cell: BxCell) -> dict[str, Any]:
    """What the arm sampled with, recorded beside the coverage block.

    The KL weight on its own is misleading, because the KL sits inside the ramped constraint
    sum: the coefficient that was actually applied is the product with the constraint scale.
    Both are written so a reader of the result file never has to reconstruct it.
    """
    layer: Any = getattr(model, "flux_layer", None)
    if layer is None:
        return {"sampling": {"flux_draws": cell.flux_draws, "latent_stochastic": False}}
    kl_weight = float(layer.config.kl_weight)
    return {
        "sampling": {
            "flux_draws": int(layer.config.draws),
            "latent_stochastic": bool(layer.latent_stochastic),
            "log_c_from_latent": bool(layer.log_c_from_latent_enabled),
            "latent_log_std_init": float(layer.config.latent_log_std_init),
            "kl_weight": kl_weight,
            "constraint_final_scale": float(layer.config.constraint_final_scale),
            "effective_kl_weight": kl_weight
            * float(layer.config.constraint_final_scale),
        }
    }


class FluxDrawEnsemble(nn.Module):
    r"""Reduce a K-draw model's ``B*K`` head outputs to ``B`` predictions, and report both.

    WHAT THE FLUX LAYER DOES AND WHAT IS LEFT OVER. With ``config.draws = K`` the layer
    expands the batch to ``B*K`` rows, one per draw, and the readout produces one prediction
    per draw. Everything downstream of the head, the trainer's loss, the validation Pearson,
    the test score, the calibration pass, expects ``B`` rows. This wrapper is the single
    place that bridges the two, so no eval consumer needs to know the arm samples.

    THE TWO PREDICTIONS ARE DIFFERENT OBJECTS AND BOTH ARE KEPT.

    ``reps['head_draws'][name]``
        the raw ``[B*K, ...]`` per-draw output. The TRAINING loss reads this, against a
        target repeated ``K`` times, which makes the data term the mean over draws of the
        per-draw score. Scoring the averaged prediction instead would train the model to
        make the AVERAGE good and leave the individual draws free, which is the opposite of
        what a sampler is for.

    ``reps['head_outputs'][name]``
        the ``[B, ...]`` reduction, which is what is SCORED. Flux Cone Learning averages its
        per-sample predictions into one call per gene, and this is the same operation. For a
        point head the reduction is the mean over draws. For the Gaussian CRPS head, whose
        output is ``(mu, log sigma)`` per draw, the reduction is the moment match of the
        mixture,

        .. math::
            \bar\mu=\frac1K\sum_k\mu_k,\qquad
            \bar\sigma^2=\frac1K\sum_k\sigma_k^2+\frac1K\sum_k(\mu_k-\bar\mu)^2,

        so the disagreement BETWEEN draws widens the predictive distribution instead of
        being discarded. Taking the mean of ``log sigma`` and dropping the second term would
        report a K-draw ensemble as more confident than a single draw, which is backwards.

    ``reps['flux_draw_pred_sd']``
        the across-draw sd of the mean parameter, averaged over the batch. This is the
        number that says whether K draws are doing anything at all; at K = 1 it is exactly
        zero and is reported as such.
    """

    model: nn.Module

    def __init__(self, model: nn.Module, draws: int, head_names: list[str]) -> None:
        """Wrap a built model whose flux layer is configured for ``draws`` draws."""
        super().__init__()
        assert draws >= 1, f"draws must be at least 1, got {draws}."
        self.model = model
        self.draws = int(draws)
        self.head_names = list(head_names)

    @property
    def flux_layer(self) -> Any:
        """The wrapped model's layer, so every caller that reaches for it is unchanged."""
        return self.model.flux_layer

    @property
    def num_parameters(self) -> dict[str, int]:
        """The wrapped model's counts. This wrapper holds no parameters of its own."""
        return cast(dict[str, int], self.model.num_parameters)

    @staticmethod
    def reduce(pred: torch.Tensor, draws: int) -> torch.Tensor:
        """``[B*K, ...] -> [B, ...]``, moment-matching a ``(mu, log sigma)`` last axis."""
        rest = pred.shape[1:]
        grouped = pred.view(-1, draws, *rest)
        if pred.shape[-1] == 2:
            mu, log_sigma = grouped[..., 0], grouped[..., 1]
            mean = mu.mean(dim=1)
            variance = torch.exp(2.0 * log_sigma).mean(dim=1) + mu.var(
                dim=1, unbiased=False
            )
            return torch.stack(
                [mean, 0.5 * torch.log(variance.clamp(min=1e-30))], dim=-1
            )
        return grouped.mean(dim=1)

    def forward(
        self, cell_graph: Any, batch: Any, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run the wrapped model, then split its head outputs into draws and a reduction."""
        predictions, reps = self.model(cell_graph, batch, *args, **kwargs)
        head_outputs = cast(dict[str, torch.Tensor], reps["head_outputs"])
        draws_out: dict[str, torch.Tensor] = {}
        spreads: list[torch.Tensor] = []
        for name in self.head_names:
            pred = head_outputs[name]
            draws_out[name] = pred
            if self.draws == 1:
                spreads.append(pred.new_zeros(()))
                continue
            head_outputs[name] = self.reduce(pred, self.draws)
            grouped = pred.view(-1, self.draws, *pred.shape[1:])
            mu = grouped[..., 0] if pred.shape[-1] == 2 else grouped
            spreads.append(mu.detach().std(dim=1, unbiased=False).mean())
        reps["head_draws"] = draws_out
        reps["flux_draw_pred_sd"] = (
            torch.stack(spreads).mean() if spreads else predictions.new_zeros(())
        )
        return predictions, reps


class FluxPrecursorConcentrationReadout(nn.Module):
    r"""Betaxanthin from the learned CONCENTRATIONS of the precursors that feed it.

    WHY THIS IS NOT THE AMINO-ACID READOUT. Betaxanthin is not a GEM metabolite. The
    Cachera cassette makes it outside the native network, L-tyrosine -> (CYP76AD1) L-DOPA
    -> (DOD) betalamic acid -> betaxanthin, and yeast-GEM carries no species for L-DOPA,
    betalamic acid or betaxanthin. What the model can carry is the SUPPLY: the tyrosine
    pool the cassette draws on, and the shikimate nodes its feedback-resistant ARO4 and
    ARO7 alleles act on.

    THE AFFINE IS REQUIRED HERE, and that is a real difference from the amino-acid arm.
    There the target IS a concentration, so the prediction could be the layer's own
    concentration with no parameter on the mean. Betaxanthin is a corrected fluorescence
    intensity in arbitrary units, and no unit-free map takes log10 molar to it, so this
    head learns ``y = w^T log10 c_precursors + b``: one weight per precursor and one
    intercept. That is the smallest map that can cross the units, and it is still a
    mechanistic tie, because the only thing it is allowed to read is the concentration of
    the named precursors.
    """

    metabolite_indices: torch.Tensor
    mixture_log_weights: torch.Tensor
    model: CellGraphTransformerMetabolism

    def __init__(
        self,
        model: CellGraphTransformerMetabolism,
        metabolite_indices: torch.Tensor,
        mixture_log_weights: torch.Tensor,
        param_dim: int,
        head_name: str = HEAD,
    ) -> None:
        """Bind the readout to the precursors' compartment species.

        Args:
            model: The metabolism transformer, built with an ANCHORED flux layer.
            metabolite_indices: ``[P, C]`` long, every intracellular compartment species
                of each precursor.
            mixture_log_weights: ``[P, C]``, log fractional volumes, ``-inf`` on padding.
            param_dim: Distributional parameters the head emits per output.
            head_name: Which head's output this writes. ``betaxanthin`` by default, so
                every wave-8 cell is unchanged; ``isobutanol`` for the wave-9 ``ibprec``
                arm, whose target is the same shape (one arbitrary-unit number per
                genotype read off named pools) and whose only difference is which pools.
        """
        super().__init__()
        self.model = model
        self.register_buffer("metabolite_indices", metabolite_indices)
        self.register_buffer("mixture_log_weights", mixture_log_weights)
        self.param_dim = param_dim
        self.head_name = head_name
        self.readout = nn.Linear(int(metabolite_indices.shape[0]), param_dim)

    @property
    def flux_layer(self) -> Any:
        """The wrapped model's layer, so every caller that reaches for it is unchanged."""
        return self.model.flux_layer

    @property
    def num_parameters(self) -> dict[str, int]:
        """The wrapped model's counts, plus this head's weight and intercept."""
        counts: dict[str, int] = dict(self.model.num_parameters)
        extra = sum(p.numel() for p in self.readout.parameters() if p.requires_grad)
        counts[f"{self.head_name}_head"] = extra
        counts["total"] = counts["total"] + extra
        return counts

    def forward(
        self, cell_graph: Any, batch: Any, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run the wrapped model, then read betaxanthin off the precursor pools."""
        predictions, reps = self.model(cell_graph, batch, *args, **kwargs)
        if "flux" not in reps:
            layer = self.model.flux_layer
            assert layer is not None, (
                "the precursor readout was built on a model with no flux layer."
            )
            # THE GATE READS THE DELETION-ONLY INDEX, the pool reads every perturbed
            # token: the context is a representation of what was changed, the gate is a
            # statement about what is gone. See `gene_removal_indices`.
            gate_indices, gate_batch, _ = gene_removal_indices(batch)
            reps["flux"] = layer(
                reps["H_genes_pert"],
                perturbed_gene_pool(
                    reps["H_genes_pert"],
                    batch["gene"].perturbation_indices,
                    batch["gene"].perturbation_indices_batch,
                ),
                gate_indices,
                gate_batch,
            )
        flux = reps["flux"]
        assert "thermo_log_c" in flux, (
            "the flux layer emitted no `thermo_log_c`, so its thermo_mode is not ANCHORED "
            "and there is no learned concentration to read the precursors from."
        )
        log10_c = mixture_log10_molar(
            flux["thermo_log_c"], self.metabolite_indices, self.mixture_log_weights
        )
        out = self.readout(log10_c)
        # THE READOUT'S OWN INPUT, kept so the audit can ask whether the concentration
        # VARIES across genotypes. Wave 9 measured an isobutanol precursor arm whose
        # validation prediction spread stayed at 1e-4 of the target's for 200 epochs while
        # its concentration-head weights were healthy, and nothing in the result file said
        # whether the pool was constant or the affine was dead.
        reps["precursor_log10_c"] = log10_c
        reps["head_outputs"][self.head_name] = out.unsqueeze(1)
        return predictions, reps


class FluxBiomassReadout(nn.Module):
    r"""Essentiality from the layer's own predicted GROWTH rate, through one affine.

    WHY THIS READOUT AND NOT ``FluxScalarHead``. Essentiality is the statement that a
    deletion stops growth, so the physically meaningful quantity is the flux through the
    GEM's objective column, ``r_2111`` ("growth"), on the medium the phenotype was scored
    on. ``FluxScalarHead`` reads a dense ``w^T v`` over all 4,131 reactions plus named
    turnovers, which can reach growth but is not a statement about it. This head reads ONE
    column and maps it to a logit with a scale and an intercept, so what is being tested is
    whether the layer's growth prediction ranks essential genes below nonessential ones,
    which is exactly what FBA's `single_gene_deletion` baseline does with a threshold.

    The affine is two parameters and is required: the layer's flux is in mmol/gDW/h and the
    target is a class, so a threshold has to be learned rather than assumed. It cannot
    manufacture a ranking that the growth column does not already carry, which is what
    keeps the arm a test of the module.
    """

    model: CellGraphTransformerMetabolism

    def __init__(
        self,
        model: CellGraphTransformerMetabolism,
        biomass_index: int,
        head_name: str = ESSENTIALITY_HEAD,
    ) -> None:
        """Bind the readout to the GEM's objective column."""
        super().__init__()
        self.model = model
        self.biomass_index = int(biomass_index)
        self.head_name = head_name
        self.readout = nn.Linear(1, 1)

    @property
    def flux_layer(self) -> Any:
        """The wrapped model's layer, so every caller that reaches for it is unchanged."""
        return self.model.flux_layer

    @property
    def num_parameters(self) -> dict[str, int]:
        """The wrapped model's counts, plus this head's scale and intercept."""
        counts: dict[str, int] = dict(self.model.num_parameters)
        extra = sum(p.numel() for p in self.readout.parameters() if p.requires_grad)
        counts[f"{self.head_name}_head"] = extra
        counts["total"] = counts["total"] + extra
        return counts

    def growth_column(self, batch: Any, reps: dict[str, Any]) -> torch.Tensor:
        """``[B, 1]`` growth flux, running the layer when no model head already did."""
        if "flux" not in reps:
            layer = self.model.flux_layer
            assert layer is not None, (
                "the biomass readout was built on a model with no flux layer."
            )
            # THE GATE READS THE DELETION-ONLY INDEX, the pool reads every perturbed
            # token: the context is a representation of what was changed, the gate is a
            # statement about what is gone. See `gene_removal_indices`.
            gate_indices, gate_batch, _ = gene_removal_indices(batch)
            reps["flux"] = layer(
                reps["H_genes_pert"],
                perturbed_gene_pool(
                    reps["H_genes_pert"],
                    batch["gene"].perturbation_indices,
                    batch["gene"].perturbation_indices_batch,
                ),
                gate_indices,
                gate_batch,
            )
        growth = reps["flux"]["v"][:, self.biomass_index : self.biomass_index + 1]
        reps["biomass_flux"] = growth
        return growth

    def forward(
        self, cell_graph: Any, batch: Any, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run the wrapped model, then read the growth column off the predicted flux."""
        predictions, reps = self.model(cell_graph, batch, *args, **kwargs)
        growth = self.growth_column(batch, reps)
        reps["head_outputs"][self.head_name] = self.readout(growth).unsqueeze(1)
        return predictions, reps


class ContextBiomassReadout(FluxBiomassReadout):
    r"""Essentiality from the context head PLUS the layer's growth column.

    The pooled arm's logit is an MLP of the perturbed context; the wave-9 flux arm's logit
    is an affine of the predicted growth flux and nothing else, so on that arm the module
    is the ONLY path a gradient can take to the backbone. This readout keeps both:
    ``logit = head(context) + a * v_biomass + b``. The backbone then receives the direct
    gradient and the one routed through the flux layer, and the per-module gradient norms
    logged every epoch (``grad_norm_flux_layer_mean`` against the transformer's) say how
    much of the update the module carries when it is not forced to carry all of it.
    """

    def forward(
        self, cell_graph: Any, batch: Any, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run the wrapped model with its context head, then add the growth affine."""
        predictions, reps = self.model(cell_graph, batch, *args, **kwargs)
        assert self.head_name in reps["head_outputs"], (
            "the context head was not built; `context_biomass` needs it in heads_config."
        )
        growth = self.growth_column(batch, reps)
        context_logit = reps["head_outputs"][self.head_name]
        # The model's scalar head emits [B, 1]; the growth affine is one number per
        # genotype too, so it takes the head's shape whatever the head's convention.
        assert context_logit.numel() == growth.shape[0], (
            context_logit.shape,
            growth.shape,
        )
        growth_logit = self.readout(growth).reshape(context_logit.shape)
        reps["head_outputs"][self.head_name] = context_logit + growth_logit
        return predictions, reps


class GatedBiomassReadout(ContextBiomassReadout):
    r"""``context_biomass`` with the growth affine behind a zero-initialized gate.

    .. math::

        \ell = h(z) + g\,(a\,v_{\text{growth}} + b), \qquad g_0 = 0

    ``ContextBiomassReadout`` hands the flux path a nonzero weight from the first step, so a
    cell that ends below the pooled arm cannot say whether the module cost it or whether the
    optimizer simply started from a worse point. Here the cell IS the pooled model at
    initialization (the ReZero pattern the perturbation propagation already uses): whatever
    it does worse it chose to do through the gate, and whatever it does better it did with
    the flux. ``ess_gate`` is written into every epoch row.
    """

    def __init__(
        self,
        model: CellGraphTransformerMetabolism,
        biomass_index: int,
        head_name: str = ESSENTIALITY_HEAD,
    ) -> None:
        """Bind the readout to the GEM's objective column, with the gate closed."""
        super().__init__(model, biomass_index, head_name)
        self.gate = nn.Parameter(torch.zeros(()))

    @property
    def num_parameters(self) -> dict[str, int]:
        """The wrapped model's counts, plus the affine and the gate."""
        counts = super().num_parameters
        counts[f"{self.head_name}_head"] += self.gate.numel()
        counts["total"] += self.gate.numel()
        return counts

    def forward(
        self, cell_graph: Any, batch: Any, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Context logit plus the GATED growth affine."""
        predictions, reps = self.model(cell_graph, batch, *args, **kwargs)
        assert self.head_name in reps["head_outputs"], (
            "the context head was not built; `gated` needs it in heads_config."
        )
        growth = self.growth_column(batch, reps)
        context_logit = reps["head_outputs"][self.head_name]
        assert context_logit.numel() == growth.shape[0], (
            context_logit.shape,
            growth.shape,
        )
        growth_logit = self.readout(growth).reshape(context_logit.shape)
        reps["ess_gate"] = self.gate.detach()
        reps["head_outputs"][self.head_name] = context_logit + self.gate * growth_logit
        return predictions, reps


class GatedFluxReadout(GatedBiomassReadout):
    r"""``gated`` plus a zero-initialized linear head over the solved flux vector.

    .. math::

        \ell = h(z) + g\,(a\,z^* + b) + w^\top \log(1 + |v^*_{\mathcal{R}}|),
        \qquad w_0 = 0

    ``v^*`` is the flux the layer that solves returns (the argmax, piecewise constant in
    the bounds, so it carries no gradient to the backbone and the head reads it as a
    fixed feature of the current solve), restricted to the reactions
    :math:`\mathcal{R}` named by ``reactions``: every reaction, the exchange reactions
    (the uptake and secretion profile), or the gene-owned ones. The growth column keeps
    its gated affine so the cell is the round-18 gated cell at initialization; the flux
    head's weight norm is written into every epoch row as ``ess_flux_head_w_norm``.
    The ladder measured the deletion pFBA vector at 0.77 against its growth ratio at
    0.65 (round 11), and the dual-overlap probe measured that the ceilings cannot move
    the growth column between genotypes; this reads what the solve returns.
    """

    flux_mask: torch.Tensor

    def __init__(
        self,
        model: CellGraphTransformerMetabolism,
        biomass_index: int,
        flux_mask: torch.Tensor,
        head_name: str = ESSENTIALITY_HEAD,
    ) -> None:
        """Bind the growth column, the gate, and the flux head over ``flux_mask``."""
        super().__init__(model, biomass_index, head_name)
        self.register_buffer("flux_mask", flux_mask.bool())
        self.flux_head = nn.Linear(int(flux_mask.sum()), 1)
        nn.init.zeros_(self.flux_head.weight)
        nn.init.zeros_(self.flux_head.bias)

    @property
    def num_parameters(self) -> dict[str, int]:
        """The gated cell's counts plus the flux head."""
        counts = super().num_parameters
        extra = sum(p.numel() for p in self.flux_head.parameters())
        counts[f"{self.head_name}_head"] += extra
        counts["total"] += extra
        return counts

    def forward(
        self, cell_graph: Any, batch: Any, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """The gated logit plus the flux head over the solved flux."""
        predictions, reps = super().forward(cell_graph, batch, *args, **kwargs)
        v = reps["flux"]["v"].detach()
        features = torch.log1p(v[:, self.flux_mask].abs())
        flux_logit = self.flux_head(features).reshape(
            reps["head_outputs"][self.head_name].shape
        )
        reps["ess_flux_head_w_norm"] = self.flux_head.weight.detach().norm()
        reps["head_outputs"][self.head_name] = (
            reps["head_outputs"][self.head_name] + flux_logit
        )
        return predictions, reps


def fba_growth_table(
    gene_ids: list[str], media: tuple[str, ...] = FBA_MEDIA
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """``([n_genes, m], [n_genes] bool, meta)``: FBA's growth ratios per cell-graph gene.

    One row per gene node in cell-graph order, so the lookup runs on the same integer
    indices the model perturbs with. The ratio is knockout growth over wild-type growth on
    each medium of ``FBA_MEDIA``, from the table ``fba_essentiality_baseline.py`` wrote for
    the 1,121 labeled genes; a gene the table does not cover gets 1.0 (no growth defect)
    under a False flag, and the flag count is written into the result so a cell states how
    many of its genes had a solver value.
    """
    assert media and all(m in FBA_MEDIA for m in media), media
    with open(FBA_PER_GENE_CSV, newline="") as fh:
        ratios = {
            row["gene"]: [float(row[m]) for m in media] for row in csv.DictReader(fh)
        }
    table = torch.ones(len(gene_ids), len(media))
    known = torch.zeros(len(gene_ids), dtype=torch.bool)
    for i, gene in enumerate(gene_ids):
        if gene in ratios:
            table[i] = torch.tensor(ratios[gene], dtype=torch.float32)
            known[i] = True
    meta = {
        "source": FBA_PER_GENE_CSV,
        "media": list(media),
        "n_genes": len(gene_ids),
        "n_genes_with_fba": int(known.sum()),
        "note": (
            "knockout-to-wild-type growth ratio per medium; genes absent from the table "
            "read 1.0 (no defect) and are flagged, and every labeled gene is in the table."
        ),
    }
    return table, known, meta


def solve_feature_table(
    gene_ids: list[str], kind: str, encoding: str = "standardized"
) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...], dict[str, Any]]:
    """``([n_genes, k], [n_genes] bool, column names, meta)``: solve-derived features.

    Read from ``SOLVE_ENCODINGS_NPZ`` in cell-graph gene order and standardized per column
    over the genes the file covers, as the ladder standardizes them; a gene the file does
    not cover reads 0 in every column under a False flag.
    """
    assert kind in ESS_SOLVE_FEATURES and kind != "none", kind
    assert encoding in ESS_SOLVE_ENCODINGS, encoding
    payload = np.load(SOLVE_ENCODINGS_NPZ, allow_pickle=False)
    genes = [str(g) for g in payload["genes"]]
    blocks: list[np.ndarray] = []
    names: list[str] = []

    def standardize(raw: np.ndarray) -> np.ndarray:
        mean = raw.mean(axis=0, keepdims=True)
        std = raw.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        return cast(np.ndarray, (raw - mean) / std)

    if kind in ("stats", "stats_pfba"):
        blocks.append(standardize(payload["gene_flux_stats"].astype(np.float32)))
        names += [f"stat_{s}" for s in payload["stat_names"]]
    if kind in ("pfba", "stats_pfba"):
        v = payload["v_deletion"].astype(np.float32)
        # The flux head's encoding (round 21 onward) or the ladder's; the flux is in
        # mmol/gDW/h clamped to the layer's scale, exactly what the head reads.
        blocks.append(np.log1p(np.abs(v)) if encoding == "log1p" else standardize(v))
        names += [f"pfba_{r}" for r in payload["rxn_ids"]]
    encoded = np.concatenate(blocks, axis=1).astype(np.float32)
    row_of = {g: i for i, g in enumerate(genes)}
    table = torch.zeros(len(gene_ids), encoded.shape[1])
    known = torch.zeros(len(gene_ids), dtype=torch.bool)
    for i, gene in enumerate(gene_ids):
        if gene in row_of:
            table[i] = torch.from_numpy(encoded[row_of[gene]])
            known[i] = True
    meta = {
        "source": SOLVE_ENCODINGS_NPZ,
        "kind": kind,
        "encoding": encoding,
        "n_columns": int(encoded.shape[1]),
        "n_genes_with_solve": int(known.sum()),
        "note": (
            "statistics standardized over the genes the file covers; the pFBA vector "
            f"{'as log(1 + |v|)' if encoding == 'log1p' else 'standardized the same way'}; "
            "uncovered genes read 0 in every column and are flagged."
        ),
    }
    return table, known, tuple(names), meta


def affine_table(
    cell: BxCell, gene_ids: list[str]
) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...], dict[str, Any]]:
    """The fba readouts' per-gene table: the media ratios, then any solve features."""
    media = tuple(cell.ess_fba_media)
    table, known, meta = fba_growth_table(gene_ids, media)
    columns: tuple[str, ...] = media
    if cell.ess_solve_features != "none":
        s_table, s_known, s_names, s_meta = solve_feature_table(
            gene_ids, cell.ess_solve_features, cell.ess_solve_encoding
        )
        table = torch.cat([table, s_table], dim=1)
        known = known & s_known
        columns = media + s_names
        meta = {**meta, "solve": s_meta, "n_columns": len(columns)}
    return table, known, columns, meta


def deleted_gene_ratios(batch: Any, table: torch.Tensor) -> torch.Tensor:
    """``[B, m]`` rows of ``table`` for each genotype's single deleted gene; ones otherwise.

    The lookup runs on the deletion index exactly as the essentiality label does
    (:func:`gene_keyed_target`); a genotype with no single deleted gene reads the wild-type
    row (all ones), which the affine's intercept absorbs.
    """
    gene = batch["gene"]
    indices = gene.deletion_indices
    assignment = gene.deletion_indices_batch
    n = int(batch.num_graphs)
    counts = torch.zeros(n, dtype=torch.long, device=indices.device)
    counts.index_add_(0, assignment, torch.ones_like(assignment))
    ratios = torch.ones(n, table.shape[1], device=table.device)
    single = counts == 1
    if bool(single.any()):
        rows = assignment[single[assignment]]
        genes = indices[single[assignment]]
        ratios[rows] = table[genes]
    return ratios


class FbaAffine(nn.Module):
    """``Linear(m, 1)`` over the deleted gene's FBA growth ratios, with its table."""

    fba_table: torch.Tensor
    fba_known: torch.Tensor

    def __init__(
        self, fba_table: torch.Tensor, fba_known: torch.Tensor, media: tuple[str, ...]
    ) -> None:
        """Hold the ``[n_genes, m]`` table and one weight per medium."""
        super().__init__()
        assert fba_table.shape[1] == len(media), (fba_table.shape, media)
        self.register_buffer("fba_table", fba_table)
        self.register_buffer("fba_known", fba_known)
        self.media = tuple(media)
        self.linear = nn.Linear(len(media), 1)

    def forward(self, batch: Any, like: torch.Tensor) -> torch.Tensor:
        """The affine's logit, in the shape of ``like``."""
        ratios = deleted_gene_ratios(batch, self.fba_table)
        assert like.numel() == ratios.shape[0], (like.shape, ratios.shape)
        return cast(torch.Tensor, self.linear(ratios).reshape(like.shape))

    def weights(self) -> dict[str, float]:
        """``ess_fba_w_<column>`` -> weight, for the epoch row.

        With more than 16 columns (the pFBA vector) only the first 16 are written by
        name, plus the norm of the whole weight vector as ``ess_fba_w_norm``.
        """
        if not hasattr(self.linear, "weight"):
            # A hidden layer (round 41): no per-column weight to name; the norm of the
            # whole head stands in.
            params = [p.detach().reshape(-1) for p in self.linear.parameters()]
            return {"ess_fba_w_norm": float(torch.cat(params).norm())}
        w = self.linear.weight.detach().reshape(-1)
        out = {
            f"ess_fba_w_{m}": float(v)
            for m, v in zip(self.media[:16], w[:16].tolist(), strict=True)
        }
        if len(self.media) > 16:
            out["ess_fba_w_norm"] = float(w.norm())
        return out


def apply_solve_zero_init(cell: BxCell, affine: FbaAffine) -> None:
    """Round 33: zero the affine's weights on every column after the media ratios.

    The flux head over the solve is zero-initialized (round 21 onward), so the cell
    starts as the backbone plus the four-ratio affine and the vector enters by gradient
    alone; the round-31 affine started with a default ``Linear(4135, 1)`` over 4,131
    standardized columns, a logit of about 0.6 sd before any training, and four of its
    six pFBA-vector cells selected an epoch before 14. No-op unless the cell asks.
    """
    if not cell.ess_solve_zero_init:
        return
    n_media = len(cell.ess_fba_media)
    with torch.no_grad():
        affine.linear.weight[:, n_media:].zero_()
    print(
        f"[solve-zero-init] affine weights zeroed on {affine.linear.weight.shape[1] - n_media} "
        f"solve-feature columns; the {n_media} media ratios keep their initialization"
    )


def apply_solve_head(cell: BxCell, affine: FbaAffine) -> None:
    """Round 41: one GELU hidden layer in the affine, when the cell asks for it.

    The affine's ``Linear(m, 1)`` becomes ``Linear(m, h) -> GELU -> Linear(h, 1)`` over
    the same input (the media ratios and the solve columns), so the readout's forward
    and its place in the logit are unchanged and only the read of the columns is
    nonlinear. Applied after ``apply_solve_zero_init``, which it supersedes: the hidden
    layer starts at its default initialization. No-op at width 0.
    """
    if cell.ess_solve_head_hidden <= 0:
        return
    m = affine.linear.in_features
    h = cell.ess_solve_head_hidden
    affine.linear = nn.Sequential(nn.Linear(m, h), nn.GELU(), nn.Linear(h, 1)).to(
        affine.fba_table.device
    )
    print(
        f"[solve-head] affine over {m} columns replaced by Linear({m}, {h}) -> GELU -> Linear({h}, 1)"
    )


class FbaContextReadout(nn.Module):
    r"""The pooled arm's context head PLUS an affine of FBA's growth ratios.

    ``fba_readout`` is declared on the class so the attribute is typed as the affine
    rather than as ``nn.Module``'s ``Tensor | Module`` lookup.

    .. math::

        \ell = h(z) + w^\top r_{\text{gene}} + b, \qquad r \in [0, 1]^4

    ``r`` is the knockout-to-wild-type growth ratio of the genotype's ONE deleted gene on
    each of the four media (``FBA_MEDIA``), looked up through the deletion index exactly as
    the essentiality label is (:func:`gene_keyed_target`). This is the transformer given the
    complement the classical ladder shows carrying every encoding past 0.80: logistic
    regression on a frozen embedding scores 0.73 to 0.78, and 0.81 to 0.83 with these four
    numbers appended. A genotype with no single deleted gene reads the wild-type row (all
    ones), which the intercept absorbs.

    The four weights are read into every epoch row (``ess_fba_w_<medium>``), so the cell
    states whether the optimizer used the solver's numbers and on which medium. No flux
    layer: this is a POOLED-arm readout, asserted at construction.
    """

    model: CellGraphTransformerMetabolism
    fba_readout: FbaAffine

    def __init__(
        self,
        model: CellGraphTransformerMetabolism,
        fba_table: torch.Tensor,
        fba_known: torch.Tensor,
        media: tuple[str, ...] = FBA_MEDIA,
        head_name: str = ESSENTIALITY_HEAD,
    ) -> None:
        """Bind the readout to the per-gene ratio table, in cell-graph gene order."""
        super().__init__()
        assert model.flux_layer is None, (
            "the FBA-ratio readout is defined on the pooled backbone; a flux arm carries "
            "the affine through `biomass_fba` or `gated_fba`."
        )
        self.model = model
        self.head_name = head_name
        self.fba_readout = FbaAffine(fba_table, fba_known, media)

    @property
    def flux_layer(self) -> Any:
        """``None``: the pooled backbone carries no layer."""
        return self.model.flux_layer

    @property
    def num_parameters(self) -> dict[str, int]:
        """The backbone's counts, plus the ratio weights and the intercept."""
        counts: dict[str, int] = dict(self.model.num_parameters)
        extra = sum(int(p.numel()) for p in self.fba_readout.parameters())
        counts[f"{self.head_name}_head"] = (
            counts.get(f"{self.head_name}_head", 0) + extra
        )
        counts["total"] = counts["total"] + extra
        return counts

    def forward(
        self, cell_graph: Any, batch: Any, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Context logit plus the affine of the deleted gene's four growth ratios."""
        predictions, reps = self.model(cell_graph, batch, *args, **kwargs)
        assert self.head_name in reps["head_outputs"], (
            "the context head was not built; `fba` needs it in heads_config."
        )
        context_logit = reps["head_outputs"][self.head_name]
        fba_logit = self.fba_readout(batch, context_logit)
        reps["head_outputs"][self.head_name] = context_logit + fba_logit
        return predictions, reps


class BiomassFbaReadout(FluxBiomassReadout):
    r"""The forced growth column PLUS the FBA affine: ``a v_growth + b + w^T r``.

    No context head. The learned layer's growth column and the solver's growth ratios are
    the only two paths to the logit, so the cell says whether the layer adds a ranking the
    solve does not already carry (Round 5 put the solve alone at 0.82 beside the context
    head; Rounds 2 and 4 put the layer alone level with the head).
    """

    fba_readout: FbaAffine

    def __init__(
        self,
        model: CellGraphTransformerMetabolism,
        biomass_index: int,
        fba_table: torch.Tensor,
        fba_known: torch.Tensor,
        media: tuple[str, ...] = FBA_MEDIA,
        head_name: str = ESSENTIALITY_HEAD,
    ) -> None:
        """Bind the growth column and the per-gene ratio table."""
        super().__init__(model, biomass_index, head_name)
        self.fba_readout = FbaAffine(fba_table, fba_known, media)

    @property
    def num_parameters(self) -> dict[str, int]:
        """The wrapped counts plus the ratio weights and intercept."""
        counts = super().num_parameters
        extra = sum(int(p.numel()) for p in self.fba_readout.parameters())
        counts[f"{self.head_name}_head"] += extra
        counts["total"] += extra
        return counts

    def forward(
        self, cell_graph: Any, batch: Any, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Growth affine plus the ratio affine; no context head."""
        predictions, reps = self.model(cell_graph, batch, *args, **kwargs)
        growth = self.growth_column(batch, reps)
        growth_logit = self.readout(growth).unsqueeze(1)
        reps["head_outputs"][self.head_name] = growth_logit + self.fba_readout(
            batch, growth_logit
        )
        return predictions, reps


class GatedFbaReadout(GatedBiomassReadout):
    r"""The gated readout PLUS the FBA affine: ``h(z) + g (a v_growth + b) + w^T r``."""

    fba_readout: FbaAffine

    def __init__(
        self,
        model: CellGraphTransformerMetabolism,
        biomass_index: int,
        fba_table: torch.Tensor,
        fba_known: torch.Tensor,
        media: tuple[str, ...] = FBA_MEDIA,
        head_name: str = ESSENTIALITY_HEAD,
    ) -> None:
        """Bind the growth column and the per-gene ratio table."""
        super().__init__(model, biomass_index, head_name)
        self.fba_readout = FbaAffine(fba_table, fba_known, media)

    @property
    def num_parameters(self) -> dict[str, int]:
        """The wrapped counts plus the ratio weights and intercept."""
        counts = super().num_parameters
        extra = sum(int(p.numel()) for p in self.fba_readout.parameters())
        counts[f"{self.head_name}_head"] += extra
        counts["total"] += extra
        return counts

    def forward(
        self, cell_graph: Any, batch: Any, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """The gated readout's logit plus the ratio affine."""
        predictions, reps = super().forward(cell_graph, batch, *args, **kwargs)
        logit = reps["head_outputs"][self.head_name]
        reps["head_outputs"][self.head_name] = logit + self.fba_readout(batch, logit)
        return predictions, reps


class ContextPrecursorConcentrationReadout(nn.Module):
    r"""The precursor readout's head, on the pooled backbone, with no flux layer.

    THE QUESTION IT ANSWERS. The only ESTABLISHED betaxanthin contrast of wave 8 is that
    the precursor-concentration readout beats 026's scalar flux head by +0.0399 (per seed
    +0.0377, +0.0532, +0.0288; sd 0.0123, n = 3), and the scalar head is itself BELOW the
    no-module control by -0.0233. So the established statement is that the concentration
    readout recovers what the flux head lost, and whether the flux layer contributes
    anything to it has never been measured. `FluxPrecursorConcentrationReadout` reads
    ``thermo_log_c``, which is ``lo + (hi - lo) sigmoid(base + W context)``, a function of
    the pooled genotype context alone; the flux touches it only through the thermodynamic
    penalty. Hypothesis (untested until this arm runs): the whole +0.0399 is the
    five-dimensional concentration bottleneck and the affine on top of it.

    THE CONSTRUCTION is the same pair of objects ``ContextConcentrationReadout`` builds --
    a zero base per GEM metabolite and a default-initialized ``Linear(d, m)`` from the
    perturbed-gene pool, squashed into the window by
    :func:`anchored_log_concentration` -- followed by the SAME whole-cell mixture over the
    precursors' compartment species and the SAME ``Linear(P, param_dim)`` affine. The two
    readouts differ only in whether a flux layer stands between the context and the
    concentration.

    NOT a ``ConcentrationReadoutBase``, deliberately. That base standardizes a
    parameter-free prediction against the train-split statistics of a target measured in
    the same units; this head's target is an arbitrary-unit intensity, so its affine does
    that job and the trainer's standardization pass must not also write into it.
    """

    metabolite_indices: torch.Tensor
    mixture_log_weights: torch.Tensor
    log_c_base: torch.Tensor
    model: CellGraphTransformerMetabolism

    def __init__(
        self,
        model: CellGraphTransformerMetabolism,
        metabolite_indices: torch.Tensor,
        mixture_log_weights: torch.Tensor,
        param_dim: int,
        hidden: int,
        n_metabolites: int,
        conc_lower_m: float,
        conc_upper_m: float,
        head_name: str = HEAD,
        sigma_fixed: float | None = None,
    ) -> None:
        """Bind the head to the GEM's metabolite axis, the window and the precursors.

        Args:
            model: The pooled backbone, built with NO flux layer.
            metabolite_indices: ``[P, C]`` rows into the ``n_metabolites`` axis.
            mixture_log_weights: ``[P, C]`` log fractional volumes, ``-inf`` on padding.
            param_dim: Distributional parameters the head emits.
            hidden: ``d``, the width of the context vector.
            n_metabolites: ``m``, the GEM metabolite count the head has one row for.
            conc_lower_m: Floor of the concentration window, in molar.
            conc_upper_m: Ceiling of the window, in molar.
            head_name: Which head's output this writes.
            sigma_fixed: Predictive spread to HOLD CONSTANT, in standardized target units,
                or None to let the affine predict it. Under CRPS the affine emits
                ``(mu, log_sigma)`` and widening ``sigma`` lowers the loss without fitting
                anything, which is one route to the constant predictor this experiment keeps
                meeting. Classical test theory gives the value that is not an escape: with
                reliability ``rho`` the irreducible residual spread is ``sqrt(1 - rho)``, so
                betaxanthin's 0.84 fixes it at 0.40. The affine then emits ``mu`` alone.
        """
        super().__init__()
        assert model.flux_layer is None, (
            "the context-only precursor readout was built on a model that carries a flux "
            "layer; the ablation is defined on the pooled backbone."
        )
        assert int(metabolite_indices.max()) < n_metabolites, (
            f"a mixture index {int(metabolite_indices.max())} is outside the head's "
            f"{n_metabolites} metabolite rows."
        )
        assert 0.0 < conc_lower_m < conc_upper_m, (
            f"window [{conc_lower_m}, {conc_upper_m}] M is not an increasing positive pair."
        )
        self.model = model
        self.register_buffer("metabolite_indices", metabolite_indices)
        self.register_buffer("mixture_log_weights", mixture_log_weights)
        self.param_dim = param_dim
        self.head_name = head_name
        self.conc_lower_m = float(conc_lower_m)
        self.conc_upper_m = float(conc_upper_m)
        self.log_c_base = nn.Parameter(torch.zeros(n_metabolites))
        self.log_c_from_context = nn.Linear(hidden, n_metabolites)
        assert sigma_fixed is None or sigma_fixed > 0.0, (
            f"--crps-sigma-fixed must be a positive spread, got {sigma_fixed}."
        )
        assert sigma_fixed is None or param_dim == 2, (
            f"--crps-sigma-fixed fixes the spread of a (mu, log_sigma) pair, but this head "
            f"emits {param_dim} parameter(s); it is meaningful only under CRPS."
        )
        self.sigma_fixed = sigma_fixed
        # With the spread held constant the affine emits the MEAN only, and the constant is
        # concatenated in `forward`. Predicting a value the loss cannot move would leave a
        # dead column in the head and a gradient path that does nothing.
        self.readout = nn.Linear(
            int(metabolite_indices.shape[0]),
            1 if sigma_fixed is not None else param_dim,
        )
        if sigma_fixed is not None:
            self.register_buffer(
                "log_sigma_fixed", torch.tensor(float(math.log(sigma_fixed)))
            )

    @property
    def flux_layer(self) -> Any:
        """``None``: this ablation is defined on the layer-free backbone."""
        return self.model.flux_layer

    @property
    def num_parameters(self) -> dict[str, int]:
        """The backbone's counts, plus the concentration head and the affine."""
        counts: dict[str, int] = dict(self.model.num_parameters)
        extra = sum(int(p.numel()) for p in self.parameters()) - sum(
            int(p.numel()) for p in self.model.parameters()
        )
        counts[f"{self.head_name}_head"] = extra
        counts["total"] = counts["total"] + extra
        return counts

    def log_concentration(self, context: torch.Tensor) -> torch.Tensor:
        """``[B, d]`` context -> ``[B, m]`` natural-log molar, inside the window."""
        return anchored_log_concentration(
            self.log_c_base,
            self.log_c_from_context,
            context,
            self.conc_lower_m,
            self.conc_upper_m,
        )

    def forward(
        self, cell_graph: Any, batch: Any, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run the pooled backbone, then read the phenotype off the precursor pools."""
        predictions, reps = self.model(cell_graph, batch, *args, **kwargs)
        assert "flux" not in reps, (
            "the context-only precursor readout received a flux output; the model was "
            "built with a layer, which this ablation excludes."
        )
        context = perturbed_gene_pool(
            reps["H_genes_pert"],
            batch["gene"].perturbation_indices,
            batch["gene"].perturbation_indices_batch,
        )
        log_c = self.log_concentration(context)
        reps["context_log_c"] = log_c
        log10_c = mixture_log10_molar(
            log_c, self.metabolite_indices, self.mixture_log_weights
        )
        reps["precursor_log10_c"] = log10_c
        out = self.readout(log10_c)
        if self.sigma_fixed is not None:
            out = torch.cat([out, self.log_sigma_fixed.expand_as(out)], dim=-1)
        reps["head_outputs"][self.head_name] = out.unsqueeze(1)
        return predictions, reps


class ConcentrationGroup(BaseModel):
    """How well the layer's learned concentration matches the measured one, per column."""

    model_config = ConfigDict(extra="forbid")

    n_records: int
    n_measurements: int
    mean_spearman: float
    mean_pearson: float
    columns: list[str]
    spearman_per_feature: list[float]
    pearson_per_feature: list[float]
    n_per_feature: list[int]
    median_pred_log10_m: list[float]
    median_true_log10_m: list[float]
    median_abs_error_log10: list[float]
    frac_pred_at_ceiling: float
    frac_pred_at_floor: float


class CalibrationBlock(BaseModel):
    """Is the head's predictive DISTRIBUTION calibrated, not just its point estimate?

    Both losses here are proper scoring rules over a distribution, so a comparison on the
    point metric alone leaves out half of what they were trained for. The probability
    integral transform is the one diagnostic that reads the same way for both: PIT is the
    predictive CDF evaluated at the observation, uniform on ``[0, 1]`` when the predictive
    distribution is right, U-shaped when it is overconfident and bell-shaped when it is too
    wide. `coverage` and `pit_ks` are scalar read-offs of the same transform.
    """

    model_config = ConfigDict(extra="forbid")

    available: bool
    reason: str | None = None
    dist: str | None = None
    group: str | None = None
    n_measurements: int | None = None
    columns: list[str] = Field(default_factory=list)
    #: Empirical coverage of the central alpha interval, whole group then per feature.
    coverage: dict[str, float] = Field(default_factory=dict)
    coverage_per_feature: dict[str, list[float]] = Field(default_factory=dict)
    #: Kolmogorov-Smirnov distance of the PIT sample from uniform; 0 is calibrated.
    pit_ks: float | None = None
    pit_ks_per_feature: list[float] = Field(default_factory=list)
    pit_mean: float | None = None
    pit_histogram: list[float] = Field(default_factory=list)


def collect_pit(
    model: nn.Module,
    loader: Any,
    dataset: Any,
    head_specs: dict[str, dict[str, Any]],
    head_name: str,
    mean: torch.Tensor,
    sd: torch.Tensor,
    cell: BxCell,
    dist_head: "DistHead | None",
) -> tuple[np.ndarray, np.ndarray]:
    """PIT values and their observation mask over a loader, ``[N, F]`` each.

    The PIT is computed in the STANDARDIZED space the head predicts in, which is the space
    the loss was computed in; a monotone change of variable does not move a PIT anyway.
    The Gaussian branch uses this trainer's own ``sigma = exp(log_sigma)`` rather than
    `DistHead`'s softplus, because that is the sigma the CRPS was trained with and a
    calibration number computed under a different link would not describe the fitted model.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    pits: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, reps = model(dataset.cell_graph.to(device), batch)
            t, m = targets_in_space(batch, head_specs, cell.target_log10_molar)[
                head_name
            ]
            t, m = t.to(device), m.to(device)
            z = (t - mean) / sd
            raw = _head_params(reps["head_outputs"][head_name])
            if raw.dim() != 3:
                continue
            if dist_head is not None:
                pit = pit_values(
                    "quantile", z, quantiles=raw, taus=dist_head.taus.to(raw.device)
                )
            else:
                mu, log_sigma = raw[..., 0], raw[..., 1]
                pit = pit_values(
                    "gaussian", z, mu=mu, scale=torch.exp(log_sigma.clamp(-6.0, 3.0))
                )
            pits.append(pit.cpu().numpy())
            masks.append(m.cpu().numpy())
    model.train(was_training)
    if not pits:
        return np.zeros((0, 0)), np.zeros((0, 0), dtype=bool)
    return np.concatenate(pits), np.concatenate(masks)


def build_calibration_block(
    pit: np.ndarray, mask: np.ndarray, cell: BxCell, columns: list[str], group: str
) -> CalibrationBlock:
    """Coverage, the KS distance from uniform, and the PIT histogram, over one group."""
    if pit.size == 0:
        return CalibrationBlock(
            available=False,
            reason=(
                "the head emits a point prediction, so it has no predictive distribution "
                "to calibrate."
            ),
            dist=cell.dist,
        )
    flat = torch.tensor(pit[mask], dtype=torch.float32)
    cov = {f"alpha_{a:.2f}": float(coverage(flat, a)) for a in COVERAGE_ALPHAS}
    per_feature: dict[str, list[float]] = {
        f"alpha_{a:.2f}": [] for a in COVERAGE_ALPHAS
    }
    ks_per_feature: list[float] = []
    for f in range(pit.shape[1]):
        col = torch.tensor(pit[mask[:, f], f], dtype=torch.float32)
        for a in COVERAGE_ALPHAS:
            per_feature[f"alpha_{a:.2f}"].append(float(coverage(col, a)))
        ks_per_feature.append(float(pit_ks(col)))
    hist, _ = np.histogram(pit[mask], bins=20, range=(0.0, 1.0), density=True)
    return CalibrationBlock(
        available=True,
        dist=cell.dist,
        group=group,
        n_measurements=int(mask.sum()),
        columns=columns,
        coverage=cov,
        coverage_per_feature=per_feature,
        pit_ks=float(pit_ks(flat)),
        pit_ks_per_feature=ks_per_feature,
        pit_mean=float(flat.mean()),
        pit_histogram=[float(x) for x in hist],
    )


class ConcentrationMixtureEntry(BaseModel):
    """One compartment species contributing to an amino acid's whole-cell average."""

    model_config = ConfigDict(extra="forbid")

    compartment: str
    species: str
    fractional_volume: float


class ConcentrationAudit(BaseModel):
    """The learned concentration against Mulleder, for ANY arm whose layer is ANCHORED.

    On a ``flux_concentration`` cell this is the supervised quantity and duplicates the test
    block, which is the cross-check. On every other anchored arm it is UNSUPERVISED: the
    concentrations were fitted only to make the reaction energies work, and this says
    whether they landed anywhere near the measured pools. The projection-head note asks for
    exactly this number and it had never been computed.
    """

    model_config = ConfigDict(extra="forbid")

    available: bool
    reason: str | None = None
    supervised: bool = False
    conc_space: str | None = None
    conc_lower_m: float | None = None
    conc_upper_m: float | None = None
    volume_fractions: CompartmentVolumeFractions | None = None
    mixture: dict[str, list[ConcentrationMixtureEntry]] = Field(default_factory=dict)
    groups: dict[str, ConcentrationGroup] = Field(default_factory=dict)


def collect_concentration(
    model: nn.Module,
    loader: Any,
    dataset: Any,
    head_specs: dict[str, dict[str, Any]],
    indices: torch.Tensor,
    log_weights: torch.Tensor,
    source: str = CONC_SOURCE_FLUX,
) -> tuple[np.ndarray, np.ndarray]:
    """``(learned log10 molar, measured log10 molar)``, ``[N, 19]`` each, over a loader.

    The learned side is the WHOLE-CELL average over the amino acid's compartments, which
    is the quantity Mulleder measures; see :func:`mixture_weights`. ``source`` names the
    tensor: the flux layer's ``thermo_log_c`` on a flux arm, or the
    ``ContextConcentrationReadout``'s own ``context_log_c`` on the ablation.
    """
    assert source in (CONC_SOURCE_FLUX, CONC_SOURCE_CONTEXT), (
        f"source must be {CONC_SOURCE_FLUX!r} or {CONC_SOURCE_CONTEXT!r}, got {source!r}."
    )
    device = next(model.parameters()).device
    layer: Any = model.flux_layer
    was_training = model.training
    model.eval()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, reps = model(dataset.cell_graph.to(device), batch)
            log_c = (
                reps["flux"]["thermo_log_c"]
                if source == CONC_SOURCE_FLUX
                else reps["context_log_c"]
            )
            log10_c = mixture_log10_molar(log_c, indices, log_weights)
            t, m = targets_in_space(batch, head_specs, True)[AUX_HEAD]
            if log10_c.shape[0] != t.shape[0]:
                # One concentration per DRAW under `log_c_from_latent`, one target per
                # genotype: the audited value is the mean over draws in log10 M, the same
                # mean the draw ensemble scores (its moment match averages mu, which is an
                # affine image of this).
                draws = int(layer.config.draws)
                assert log10_c.shape[0] == t.shape[0] * draws, (
                    f"thermo_log_c has {log10_c.shape[0]} rows against {t.shape[0]} "
                    f"targets and {draws} draws."
                )
                log10_c = log10_c.view(t.shape[0], draws, -1).mean(dim=1)
            preds.append(log10_c.cpu().numpy())
            trues.append(torch.where(m, t, torch.nan).cpu().numpy())
    model.train(was_training)
    return np.concatenate(preds), np.concatenate(trues)


def _concentration_group(
    pred: np.ndarray, true: np.ndarray, sel: np.ndarray, lo: float, hi: float
) -> ConcentrationGroup:
    p, t = pred[sel], true[sel]
    scored = _score_test_vector(p, t, np.ones(len(p), dtype=bool), AA_KEYS)
    med_pred = [float(np.median(p[:, f])) for f in range(p.shape[1])]
    med_true = [
        float(np.nanmedian(t[:, f]))
        if bool(np.isfinite(t[:, f]).any())
        else float("nan")
        for f in range(t.shape[1])
    ]
    med_err = [
        float(np.nanmedian(np.abs(p[:, f] - t[:, f])))
        if bool(np.isfinite(t[:, f]).any())
        else float("nan")
        for f in range(t.shape[1])
    ]
    # Within 0.01 of a decade of an edge, the sigmoid is saturated and the head cannot move
    # the value further, which is the failure the window check exists to prevent.
    return ConcentrationGroup(
        n_records=int(sel.sum()),
        n_measurements=int(np.isfinite(t).sum()),
        mean_spearman=float(scored["mean_spearman"]),
        mean_pearson=float(scored["mean_pearson"]),
        columns=AA_KEYS,
        spearman_per_feature=list(scored["spearman_per_feature"]),
        pearson_per_feature=list(scored["pearson_per_feature"]),
        n_per_feature=list(scored["n_per_feature"]),
        median_pred_log10_m=med_pred,
        median_true_log10_m=med_true,
        median_abs_error_log10=med_err,
        frac_pred_at_ceiling=float((p >= math.log10(hi) - 0.01).mean()),
        frac_pred_at_floor=float((p <= math.log10(lo) + 0.01).mean()),
    )


def build_model_for_cell(
    cell: BxCell, cfg: BxConfig, dataset: Any, device: torch.device
) -> tuple[nn.Module, dict[str, dict[str, Any]], str, dict[str, Any]]:
    """Build the flux layer, the heads and the model exactly as a training run does.

    Factored out of :func:`run_cell` so ``audit_top_strain_feasibility.py`` rebuilds the
    SAME architecture from a result file's cell record rather than a second copy of this
    construction that could drift from it. Seeds are set here, in the order the training
    run set them, so a rebuild draws the same initialization before the saved weights
    overwrite it.

    Returns ``(model, head_specs, main_head, flux_meta)``.
    """
    torch.manual_seed(cell.seed)
    np.random.seed(cell.seed)
    # 026's registry stays the single source of truth for WHAT the module does; the cell adds
    # only the parameter-source and schedule knobs on top of it. A copy, so a second cell in
    # the same process cannot inherit the first one's overrides.
    arm = {
        **ARMS[cell.base_arm],
        "kcat_tier": cell.kcat_tier,
        "thermo_source": cell.thermo_source,
        "constraint_warmup_epochs": cell.constraint_warmup_epochs,
        "constraint_final_scale": cell.constraint_scale,
        "constraint_delay_epochs": cell.constraint_delay_epochs,
        # Construction-time (round 19): the abundance head's depth.
        "abundance_head_layers": cell.flux_abundance_head_layers,
    }
    # THE CHANNEL OVERRIDE. `fluxforce` is a parameterization of
    # torchcell.metabolism.flux_layer, not a new physics switch, and 026's ARMS stays the
    # single source of truth for the switches; this passes the string through without
    # editing that registry.
    #
    # THE CELL WINS OVER THE ENVIRONMENT (wave 8). Wave 7 had only the variable, so a cell's
    # own record did not carry its channel and a rebuild from the result file alone
    # reconstructed the base arm; `audit_top_strain_feasibility.py` on job 1755 died on
    # KeyError 'flux_layer.latent_mlp.0.weight' for exactly that reason. `cell.flux_parameterization`
    # is serialized with the cell, so a rebuild is now correct on its own. The variable is
    # still honored when the cell names nothing, which is what keeps every wave-7 script and
    # its reruns working unchanged.
    channel = cell.flux_parameterization or os.getenv("TC_FLUX_PARAMETERIZATION")
    if channel:
        arm["parameterization"] = channel
    # THE TWO CONSTRUCTION-TIME LATENT SWITCHES go into the arm dict, the way the channel
    # does, because `log_c_from_latent` builds a module and cannot be assigned onto a
    # built layer. `latent_stochastic` travels with it so the layer's constructor can
    # check the pairing; it is also assigned onto the config below, as before, so the
    # call-time read is the same value.
    arm["latent_stochastic"] = cell.flux_latent_stochastic
    arm["log_c_from_latent"] = cell.flux_log_c_from_latent
    # THE WARM-START SWITCHES, in the arm dict for the same reason and NOT assigned onto
    # the built config: `FluxLayer.__init__` reads all three while it solves the reference
    # and creates `warm_log_alpha`, so a post-construction assignment would be ignored and
    # the cell would claim a reference table it never opened.
    arm["shuffle_kcat_seed"] = cell.flux_shuffle_kcat_seed
    arm["warm_start_reference_table_path"] = cell.flux_reference_table
    arm["warm_start_alpha_learnable"] = not cell.flux_alpha_frozen
    arm["warm_start_init_rel"] = cell.flux_warm_start_init_rel
    # THE MEDIUM, written into the arm dict for the same reason the channel is: 026's
    # `build_flux_layer` owns the cobra model (it constructs its own `YeastGEM()` and reads
    # the exchange bounds into the box through `build_gem_tensors`), so the bounds have to
    # be applied there, before the tensors are taken. `None` is the shipped boundary and is
    # the default, so every cell before wave 9 builds a bit-identical layer.
    assert cell.apply_media in APPLY_MEDIA_CHOICES, (
        f"apply_media must be one of {APPLY_MEDIA_CHOICES}, got {cell.apply_media!r}."
    )
    if cell.apply_media == MEDIA_DATASET:
        assert cell.head in HEAD_MEDIUM, (
            f"--apply-media dataset needs a recorded medium for head {cell.head!r}; "
            f"HEAD_MEDIUM covers {sorted(HEAD_MEDIUM)}."
        )
        arm["media_name"] = HEAD_MEDIUM[cell.head]
    gene_ids = list(dataset.cell_graph["gene"].node_ids)
    flux_layer, flux_meta = build_flux_layer(arm, gene_ids, cell.hidden)
    if cell.apply_media == MEDIA_DATASET:
        assert arm["flux"] and flux_meta["media"]["applied"], (
            "--apply-media dataset is a statement about the GEM's exchange bounds, which "
            f"only a FLUX arm has; arm {cell.base_arm!r} built no layer."
        )

    if flux_layer is not None:
        # THE CONCENTRATION WINDOW, written onto the built layer rather than passed into
        # its construction: 026's `build_flux_layer` owns that construction and forwards
        # only the fields its own registry names, and this experiment does not edit it.
        # `reaction_potential` reads the window off `config` at call time, so the
        # assignment takes effect. Both defaults equal the shipped constants, so a cell
        # that does not ask for a different window is unchanged.
        # The turnover-density weight, written the same way and for the same reason:
        # `constraint_loss` reads it off `config` at call time. 0.0 leaves the loss exactly
        # as every cell before wave 8 computed it.
        flux_layer.config.density_weight = cell.flux_density_weight
        flux_layer.config.balance_active_weight = cell.flux_balance_active_weight
        flux_layer.config.parsimony_outside_ramp_weight = cell.flux_parsimony_weight
        # The layer validates this at construction, and it is assigned after, so the
        # check is repeated here rather than letting a typo pass as 'dynamic'.
        assert cell.flux_balance_active_row_set in {"dynamic", "frozen"}, (
            f"flux_balance_active_row_set={cell.flux_balance_active_row_set!r} is not "
            "'dynamic' or 'frozen'."
        )
        flux_layer.config.balance_active_row_set = cell.flux_balance_active_row_set
        flux_layer.config.conc_lower_m = cell.conc_lower_m
        flux_layer.config.conc_upper_m = cell.conc_upper_m
        # The capacity scale is read by dynamic_box at call time, so it is written onto
        # the built layer the same way.
        flux_layer.config.capacity_scale = cell.flux_capacity_scale
        # So are the elastic program's price and the pool projection (round 18).
        flux_layer.config.lp_slack_penalty = cell.flux_lp_slack_penalty
        flux_layer.config.pool_by_construction = cell.flux_pool_by_construction
        flux_layer.config.abundance_log_range = cell.flux_abundance_log_range
        # THE WARM-START GEOMETRY SWITCHES (wave 9). The fold is read off config at call
        # time. ORDER MATTERS for the two files: apply_log_c_init reads conc_lower_m and
        # conc_upper_m, assigned just above, and refuses a window that disagrees with the
        # one the witness was solved in. Both default to None, so every earlier cell is
        # unchanged.
        flux_layer.config.availability_fold = cell.flux_availability_fold
        if cell.flux_thermo_exempt_path is not None:
            flux_layer.apply_thermo_exempt_list(cell.flux_thermo_exempt_path)
        if cell.flux_log_c_init_path is not None:
            flux_layer.apply_log_c_init(cell.flux_log_c_init_path)
        # THE K-DRAW SWITCHES, written before the layer is used but AFTER it is built, for
        # the same reason as the three above: 026's `build_flux_layer` owns the construction
        # and forwards only the fields its own registry names. `draws` and `kl_weight` are
        # read at call time so the assignment takes effect; `latent_stochastic` is read by
        # the layer's `latent_stochastic` property, also at call time.
        #
        # The K > 1 assertion the layer makes in its constructor cannot fire here, since the
        # layer was already built, so it is repeated as the explicit check below.
        flux_layer.config.latent_stochastic = cell.flux_latent_stochastic
        flux_layer.config.draws = cell.flux_draws
        if cell.flux_kl_weight is not None:
            flux_layer.config.kl_weight = cell.flux_kl_weight
        assert cell.flux_draws == 1 or flux_layer.latent_stochastic, (
            f"--flux-draws {cell.flux_draws} needs --flux-latent-stochastic: K copies of a "
            "deterministic mu are not K samples, and the arm would pay K times the compute "
            "to report the K = 1 prediction."
        )
        assert flux_layer.log_c_from_latent_enabled == cell.flux_log_c_from_latent, (
            "the built layer's log_c_from_latent does not match the cell's; the switch "
            "is construction-time and travels through the arm dict."
        )
        assert not (
            cell.flux_draws > 1
            and cell.aa_readout == AA_READOUT_CONCENTRATION
            and not cell.flux_log_c_from_latent
        ), (
            "the concentration readout predicts a slice of the layer's `thermo_log_c`, "
            "which without --flux-log-c-from-latent is a function of the genotype "
            "context and NOT of the drawn flux, so --flux-draws above 1 buys it nothing "
            "and its B rows would not align with the B*K flux rows. Measured and written "
            "up in [[experiments.027-betaxanthin-metabolic.wave8-sampling]] (Finding 5); "
            "set --flux-log-c-from-latent so each draw carries its own concentration."
        )

    if arm["flux"]:
        frac = float(flux_meta["kcat_experimental_fraction"])
        # NO FALLBACK. `resolve_kcat_table` catches a missing Open Enzyme Database mirror and
        # returns an empty record list, which replaces every measured kcat with the organism
        # default -- an "enzyme-constrained" arm with no enzyme constraints, running to
        # completion and reporting a plausible number. The mirror is 512 KB and is NOT on
        # Delta as of 2026-09-03; see README.md, "Prerequisites".
        assert frac >= cfg.flux.min_kcat_experimental_fraction, (
            f"kcat experimental fraction is {frac:.4f}, below the floor "
            f"{cfg.flux.min_kcat_experimental_fraction}. The Open Enzyme Database mirror at "
            f"$DATA_ROOT/data/enzyme_kinetics/open_enzyme_database/scerevisiae is missing or "
            "empty, so every kcat has silently become the organism default."
        )

    heads = (HEAD, AUX_HEAD, ISOBUTANOL_HEAD, ESSENTIALITY_HEAD)
    assert cell.head in heads, f"--head must be one of {heads}, got {cell.head!r}."
    assert not (cell.head == AUX_HEAD and cell.aux_mulleder), (
        "--head mulleder19 makes the amino-acid head the PRIMARY supervised head; "
        "--aux-mulleder would then declare it twice."
    )
    assert not (cell.dist == "quantile" and arm["flux"]), (
        "--dist quantile is defined for the MLP heads only. A flux readout predicts the "
        "layer's own quantity (its turnover or its concentration), so a quantile fan would "
        "have to come from a separate parameterized head, which is exactly the tie the "
        "mechanistic arm removes. The principled distributional form there is an energy "
        "score over the flux sampler's draws, which is not built."
    )
    assert not (
        cell.bx_readout == BX_READOUT_PRECURSOR_CONCENTRATION
        and not (cell.head == HEAD and arm["flux"])
    ), (
        f"bx_readout {BX_READOUT_PRECURSOR_CONCENTRATION!r} is defined only for a FLUX "
        f"arm on --head {HEAD}; this cell is arm {cell.base_arm!r} on head {cell.head!r}."
    )
    # THE BETAXANTHIN ABLATION IS DEFINED ON THE POOLED BACKBONE ONLY, exactly as the
    # amino-acid one is: a flux arm would put the layer back and the contrast would be gone.
    assert not (
        cell.bx_readout == BX_READOUT_CONTEXT_PRECURSOR_CONCENTRATION
        and not (cell.head == HEAD and cell.base_arm == "pooled" and not arm["flux"])
    ), (
        f"bx_readout {BX_READOUT_CONTEXT_PRECURSOR_CONCENTRATION!r} is defined only for "
        f"--arm pooled with --head {HEAD}; this cell is arm {cell.base_arm!r} on head "
        f"{cell.head!r}."
    )
    assert not (
        cell.ib_readout == IB_READOUT_PRECURSOR_CONCENTRATION
        and cell.head == ISOBUTANOL_HEAD
        and not arm["flux"]
    ), (
        f"ib_readout {IB_READOUT_PRECURSOR_CONCENTRATION!r} reads the layer's learned "
        f"concentration and needs a FLUX arm; arm {cell.base_arm!r} builds none. The "
        "no-module isobutanol control is --arm pooled, which uses the scalar head."
    )
    assert cell.aa_readout in AA_READOUTS, (
        f"aa_readout must be one of {AA_READOUTS}, got {cell.aa_readout!r}."
    )
    assert not (
        cell.aa_readout == AA_READOUT_CONCENTRATION
        and not (cell.head == AUX_HEAD and arm["flux"])
    ), (
        f"aa_readout {AA_READOUT_CONCENTRATION!r} is defined only for a FLUX arm with "
        f"--head {AUX_HEAD}; this cell is arm {cell.base_arm!r} on head {cell.head!r}."
    )
    # THE ABLATION IS DEFINED ON THE POOLED BACKBONE ONLY. A flux arm would put the layer
    # back, and a betaxanthin head has no 19-column concentration to supervise.
    assert not (
        cell.aa_readout == AA_READOUT_CONTEXT_CONCENTRATION
        and not (
            cell.head == AUX_HEAD and cell.base_arm == "pooled" and not arm["flux"]
        )
    ), (
        f"aa_readout {AA_READOUT_CONTEXT_CONCENTRATION!r} is defined only for --arm pooled "
        f"with --head {AUX_HEAD}; this cell is arm {cell.base_arm!r} on head {cell.head!r}."
    )
    # A FIXED SPREAD MUST REACH A HEAD THAT HONORS IT. Only the context-precursor readout
    # reads `sigma_fixed`; on any other head the flag would be accepted, recorded in the
    # result file and do nothing, which is the kind of silent no-op that makes a contrast
    # look null.
    assert cell.crps_sigma_fixed is None or (
        cell.crps and cell.bx_readout == BX_READOUT_CONTEXT_PRECURSOR_CONCENTRATION
    ), (
        "--crps-sigma-fixed fixes the spread of the CRPS pair emitted by the "
        f"{BX_READOUT_CONTEXT_PRECURSOR_CONCENTRATION!r} readout, so it needs --crps and that readout; "
        f"this cell has crps={cell.crps} and bx_readout={cell.bx_readout!r}."
    )
    assert cell.target_log10_molar == (cell.aa_readout in AA_LOG10_MOLAR_READOUTS), (
        "target_log10_molar is derived from the readout and must not be set by hand: the "
        "concentration readouts predict log10 molar and every other readout predicts the "
        "target in its stored units."
    )
    assert cell.target_log10 == (cell.head == ISOBUTANOL_HEAD), (
        "target_log10 is derived from the head and must not be set by hand: the biosensor "
        "fold change is trained and scored in log10 and every other target is not."
    )
    main_head = cell.head
    head_specs: dict[str, dict[str, Any]]
    if main_head == ESSENTIALITY_HEAD:
        # THE TARGET IS NOT IN THE GRAPH. It is Merzbacher 2025's released label, joined
        # per genotype by its deleted gene, which is what FCL itself trained on. The build
        # supplies the genotype and the cell graph; every phenotype column it carries is
        # ignored here, which is also why the union build's fitness merge is harmless.
        gene_value, gene_known, essentiality_report = essentiality_gene_labels(
            dataset, split_file=cell.split_file
        )
        head_specs = {
            main_head: {
                "label": ESSENTIALITY_LABEL,
                "width": 1,
                "gene_label": gene_value,
                "gene_known": gene_known,
            }
        }
        flux_meta["essentiality_labels"] = essentiality_report
    elif main_head == ISOBUTANOL_HEAD:
        # THE BIOSENSOR IS ONE COLUMN OF THE MERGED 20-VALUE BLOCK, not a width-1 group;
        # see `ISOBUTANOL_BLOCK_KEYS` for the measurement that forces it. The held-out
        # flags mask every genotype the validated re-screen measured, so the 224-strain
        # panel stays an external check rather than a training label.
        heldout_flags, heldout_report = isobutanol_heldout_flags(dataset)
        head_specs = {
            main_head: {
                "label": "metabolite_level",
                "width": ISOBUTANOL_BLOCK_WIDTH,
                "column": ISOBUTANOL_COLUMN,
                "transform": "log10",
                "heldout_gene_flags": heldout_flags,
            }
        }
        flux_meta["isobutanol"] = {
            "dataset_root": dataset_paths(ISOBUTANOL_HEAD)[0],
            "block_keys": ISOBUTANOL_BLOCK_KEYS,
            "block_width": ISOBUTANOL_BLOCK_WIDTH,
            "column": ISOBUTANOL_COLUMN,
            "target_space": "log10_fold_change",
            "held_out": heldout_report,
            "note": (
                "the Lopez biosensor fold change was merged into the Mulleder record by "
                "MeanExperimentDeduplicator wherever both were measured, so it is read as "
                "column 8 of a key-sorted width-20 metabolite_level block rather than as "
                "a width-1 group; in fig6_build the width-1 metabolite_level groups are "
                "4,669 betaxanthin records against 189 isobutanol-only ones."
            ),
        }
    else:
        head_specs = {
            # Betaxanthin and the Mulleder metabolome both carry the label
            # `metabolite_level`; WIDTH is what separates them. Declaring only the width-1
            # spec supervises betaxanthin alone, and a Mulleder-only genotype simply
            # matches nothing and arrives masked False.
            main_head: {
                "label": "metabolite_level",
                "width": 1 if main_head == HEAD else AUX_WIDTH,
            }
        }
    # THE HEAD'S OUTPUT WIDTH follows the distributional loss: 2 for the Gaussian CRPS's
    # (mu, log_sigma) and K = 19 for the quantile grid. `dist_param_dim` is the same
    # function 019's trainer uses, so the two experiments cannot disagree about the width.
    param_dim = dist_param_dim(cell.dist) if cell.crps else 1
    assert not (cell.dist == "quantile" and not cell.crps), (
        "--dist quantile is a distributional loss and needs --crps, which is the flag that "
        "makes the head emit more than a point."
    )
    heads_config: dict[str, Any] = {}
    met_index = flux_meta["met_index"] if arm["flux"] else {}
    if main_head == ESSENTIALITY_HEAD:
        assert not cell.aux_mulleder, (
            "--head essentiality skips the train-split standardization pass, so the "
            "amino-acid co-head would have no mean and sd to standardize against. The "
            "two are refused together rather than failing on a KeyError mid-epoch."
        )
        assert not cell.crps, (
            "--head essentiality is a CLASSIFIER scored with a binary cross entropy on one "
            "logit. CRPS is a proper scoring rule for a CONTINUOUS target and a Gaussian "
            "over a 0/1 label is not the right object, so --crps is refused here rather "
            "than quietly producing a (mu, log_sigma) pair nothing reads."
        )
        if arm["flux"]:
            # THE BIOMASS READOUT declares no model head: the prediction is an affine of
            # the layer's own growth column, injected by `FluxBiomassReadout` below.
            # The layer publishes `rxn_ids`, not the GEM's own `biomass_index`: that
            # attribute belongs to the gem tensors `build_flux_layer` consumes and does not
            # survive onto the module. Resolving the column by reaction id is also what
            # makes the readout state which column it took.
            assert flux_layer is not None, (
                "--head essentiality on a flux arm reads the GEM's objective column, so it "
                "needs a flux layer."
            )
            assert BIOMASS_REACTION in flux_layer.rxn_ids, (
                f"--head essentiality on a flux arm reads {BIOMASS_REACTION}, which this "
                f"layer's {len(flux_layer.rxn_ids)} reactions do not include."
            )
            if cell.ess_readout in ESS_READOUTS_WITH_CONTEXT:
                # THE CONTEXT HEAD IS BUILT TOO: `ContextBiomassReadout` and its gated
                # variant add the growth affine to its logit rather than replacing it.
                heads_config[ESSENTIALITY_HEAD] = {
                    "kind": "scalar",
                    "output_dim": 1,
                    "param_dim": 1,
                }
        else:
            # A pooled arm has no growth column, so a flux readout here would be built
            # as nothing and the cell would silently run the plain head (the round-6 smoke
            # did exactly that with biomass_fba on a pooled arm).
            assert cell.ess_readout in (ESS_READOUT_BIOMASS, ESS_READOUT_FBA), (
                f"--ess-readout {cell.ess_readout} needs a flux arm; a pooled arm takes "
                f"the default or `fba`."
            )
            heads_config[ESSENTIALITY_HEAD] = {
                "kind": "scalar",
                "output_dim": 1,
                "param_dim": 1,
            }
    elif main_head == ISOBUTANOL_HEAD:
        if arm["flux"] and cell.ib_readout == IB_READOUT_PRECURSOR_CONCENTRATION:
            # THE PRECURSOR-CONCENTRATION READOUT declares no model head; the prediction is
            # the affine of the layer's own `thermo_log_c` at the anchor rows, injected by
            # `FluxPrecursorConcentrationReadout` below.
            assert flux_layer is not None and (
                flux_layer.thermo_mode is ThermoMode.ANCHORED
            ), (
                f"--ib-readout {IB_READOUT_PRECURSOR_CONCENTRATION} needs an arm whose "
                f"thermo_mode is ANCHORED; arm {cell.base_arm!r} is "
                f"{'not a flux arm' if flux_layer is None else str(flux_layer.thermo_mode)}."
            )
        elif arm["flux"]:
            missing = [m for m in ISOBUTANOL_PRECURSORS if m not in met_index]
            assert not missing, (
                f"the isobutanol precursor species {missing} are not in the layer's "
                "metabolite index, so the flux layer and the GEM disagree about the model."
            )
            heads_config[ISOBUTANOL_HEAD] = {
                "kind": "flux_scalar",
                "output_dim": 1,
                "param_dim": param_dim,
                "precursor_indices": [met_index[m] for m in ISOBUTANOL_PRECURSORS],
                "normalize_inputs": cell.head_normalize,
            }
            flux_meta["isobutanol"]["precursor_species"] = ISOBUTANOL_PRECURSORS
            flux_meta["isobutanol"]["product_exchange"] = ISOBUTANOL_PRODUCT_EXCHANGE
        else:
            heads_config[ISOBUTANOL_HEAD] = {
                "kind": "scalar",
                "output_dim": 1,
                "param_dim": param_dim,
            }
    elif main_head == HEAD:
        if arm["flux"] and cell.bx_readout == BX_READOUT_PRECURSOR_CONCENTRATION:
            # THE PRECURSOR-CONCENTRATION READOUT declares no model head: the prediction
            # is a linear map of the layer's own `thermo_log_c` at the precursor rows,
            # injected by `FluxPrecursorConcentrationReadout` below.
            assert flux_layer is not None and (
                flux_layer.thermo_mode is ThermoMode.ANCHORED
            ), (
                f"--bx-readout {BX_READOUT_PRECURSOR_CONCENTRATION} needs an arm whose "
                f"thermo_mode is ANCHORED; arm {cell.base_arm!r} is "
                f"{'not a flux arm' if flux_layer is None else str(flux_layer.thermo_mode)}."
            )
        elif arm["flux"]:
            heads_config[HEAD] = {
                "kind": "flux_scalar",
                "output_dim": 1,
                "param_dim": param_dim,
                "precursor_indices": [
                    met_index[m] for m in BETAXANTHIN_PRECURSORS if m in met_index
                ],
                "normalize_inputs": cell.head_normalize,
            }
        elif cell.bx_readout == BX_READOUT_CONTEXT_PRECURSOR_CONCENTRATION:
            # THE CONTEXT-ONLY PRECURSOR READOUT declares no head either: the prediction is
            # injected by `ContextPrecursorConcentrationReadout` below, so the only
            # parameterized map from the context to the loss is the concentration head and
            # its affine. The `pooled` arm's own scalar head is deliberately NOT built.
            assert flux_layer is None, (
                f"--bx-readout {BX_READOUT_CONTEXT_PRECURSOR_CONCENTRATION} excludes the "
                f"flux layer; arm {cell.base_arm!r} built one."
            )
        else:
            heads_config[HEAD] = {
                "kind": "scalar",
                "output_dim": 1,
                "param_dim": param_dim,
            }
    elif arm["flux"] and cell.aa_readout == AA_READOUT_CONCENTRATION:
        # THE CONCENTRATION READOUT declares NO head at all: the prediction is a slice of
        # the layer's own `thermo_log_c`, injected into `head_outputs` by
        # `FluxConcentrationReadout` below. Declaring a head here would put a parameterized
        # map between the concentration and the loss, which is the tie this arm removes.
        assert (
            flux_layer is not None and flux_layer.thermo_mode is ThermoMode.ANCHORED
        ), (
            f"--aa-readout {AA_READOUT_CONCENTRATION} needs an arm whose thermo_mode is "
            f"ANCHORED, since the learned concentration exists only there; arm "
            f"{cell.base_arm!r} is "
            f"{'not a flux arm' if flux_layer is None else str(flux_layer.thermo_mode)}."
        )
    elif cell.aa_readout == AA_READOUT_CONTEXT_CONCENTRATION:
        # THE CONTEXT-ONLY CONCENTRATION READOUT declares no head either: the prediction
        # is injected by `ContextConcentrationReadout` below, on the pooled backbone. The
        # `pooled` arm's own `vector` head is deliberately NOT built, so the only
        # parameterized map from the context to the loss is the concentration head.
        assert flux_layer is None, (
            f"--aa-readout {AA_READOUT_CONTEXT_CONCENTRATION} excludes the flux layer; "
            f"arm {cell.base_arm!r} built one."
        )
    elif arm["flux"]:
        # 026's mechanistic readout: each measured amino acid is read off the turnover of
        # its own GEM metabolite, so the flux arm's amino-acid prediction goes through the
        # flux vector exactly as its betaxanthin prediction does.
        heads_config[AUX_HEAD] = {
            "kind": "flux_metabolite",
            "output_dim": AUX_WIDTH,
            "param_dim": param_dim,
            "metabolite_indices": [
                met_index[AMINO_ACID_TO_GEM_METABOLITE[k]] for k in AA_KEYS
            ],
        }
    else:
        heads_config[AUX_HEAD] = {
            "kind": "vector",
            "output_dim": AUX_WIDTH,
            "param_dim": param_dim,
        }
    if cell.aux_mulleder:
        # The co-head reads the pooled representation whatever the arm, so the contrast
        # between arms stays on the betaxanthin readout alone.
        head_specs[AUX_HEAD] = {"label": "metabolite_level", "width": AUX_WIDTH}
        heads_config[AUX_HEAD] = {"kind": "vector", "output_dim": AUX_WIDTH}

    graph_reg_kwargs: dict[str, Any] = (
        {
            "graph_regularization_config": GRAPH_REG_CONFIG,
            "graph_reg_lambda": GRAPH_REG_LAMBDA,
        }
        if cell.graph_reg
        else {}
    )
    model = CellGraphTransformerMetabolism(
        cell_graph=dataset.cell_graph,
        gene_num=len(gene_ids),
        hidden_channels=cell.hidden,
        num_transformer_layers=cell.layers,
        num_attention_heads=cell.attention_heads,
        dropout=0.1,
        heads_config=heads_config,
        # OFF. With it on, the model learns a free per-gene vector and stops needing the
        # protein embedding, which 020 measured at 0.279 (off) against 0.151 (on) and
        # 019 recorded as making generalization impossible by construction.
        learnable_embedding_config={
            "enabled": cell.learnable_embedding,
            "size": cell.hidden,
            "preprocessor": {"num_layers": cell.preprocessor_layers, "dropout": 0.1},
        },
        # `num_heads` feeds the perturbation TRANSFORM's cross-attention: 6 from 019's
        # conf, or 9 as the 020 grid ran; the default 8 does not divide 90. `enabled: False`
        # builds no perturbation READOUT head. Every 027 readout lives in `heads_config` or
        # in a wrapper, every call site here discards the first return value, and the head
        # received no gradient on any cell (per-module gradient report, wave 11,
        # 2026-09-15), so it was 16,381 dead parameters at width 90 and 1,201 at width 24
        # (job 2054: the width-90 tally went from 958,501 to 942,120). Removing it shifts the
        # RNG stream at construction: a cell built after 2026-09-16 is not bit-identical at
        # the same seed to one built before, which is why the wave-11 cells are not mixed
        # with later ones in a seed-paired comparison.
        perturbation_head_config={
            "num_heads": cell.pert_heads,
            "dropout": 0.1,
            "enabled": False,
        },
        attention_mask_config=ATTENTION_MASK_CONFIG if cell.attention_mask else None,
        # The model reads cell_graph["gene"].x only when it is TOLD it has precomputed
        # embeddings; without this it looks for a learnable table that is now disabled and
        # raises "No gene embeddings available".
        node_embeddings=_NODE_EMBEDDINGS,
        flux_layer=flux_layer,
        **graph_reg_kwargs,
    ).to(device)

    if main_head == ESSENTIALITY_HEAD and arm["flux"]:
        assert flux_layer is not None
        biomass_index = int(flux_layer.rxn_ids.index(BIOMASS_REACTION))
        flux_meta["essentiality"] = {
            "readout": cell.ess_readout,
            "biomass_reaction": BIOMASS_REACTION,
            "biomass_index": biomass_index,
            "note": (
                "essentiality is read off the flux through the GEM's objective column, "
                "which is the quantity FBA thresholds in its own baseline"
                + {
                    ESS_READOUT_BIOMASS: "",
                    ESS_READOUT_CONTEXT_BIOMASS: ", added to the pooled arm's context head",
                    ESS_READOUT_GATED: (
                        ", added to the pooled arm's context head through a gate "
                        "initialized at zero"
                    ),
                    ESS_READOUT_BIOMASS_FBA: (
                        ", plus an affine of FBA's growth ratios for the deleted gene"
                    ),
                    ESS_READOUT_GATED_FBA: (
                        ", added to the pooled arm's context head through a gate "
                        "initialized at zero, plus an affine of FBA's growth ratios for "
                        "the deleted gene"
                    ),
                    ESS_READOUT_GATED_FLUX: (
                        ", added to the pooled arm's context head through a gate "
                        "initialized at zero, plus a zero-initialized linear head over "
                        "the solved flux vector"
                    ),
                }[cell.ess_readout]
                + "."
            ),
        }
        assert cell.ess_readout != ESS_READOUT_FBA, (
            "--ess-readout fba is a pooled-arm readout (the context head plus FBA's four "
            "growth ratios); a flux arm carries the affine through biomass_fba or gated_fba."
        )
        wrapped_ess: FluxBiomassReadout
        if cell.ess_readout in ESS_READOUTS_WITH_FBA:
            fba_table, fba_known, fba_media, fba_meta = affine_table(cell, gene_ids)
            flux_meta["essentiality"]["fba"] = fba_meta
            if cell.ess_readout == ESS_READOUT_BIOMASS_FBA:
                wrapped_ess = BiomassFbaReadout(
                    model, biomass_index, fba_table, fba_known, fba_media
                ).to(device)
            else:
                wrapped_ess = GatedFbaReadout(
                    model, biomass_index, fba_table, fba_known, fba_media
                ).to(device)
            apply_solve_zero_init(cell, wrapped_ess.fba_readout)
            apply_solve_head(cell, wrapped_ess.fba_readout)
        elif cell.ess_readout == ESS_READOUT_GATED_FLUX:
            assert flux_layer.parameterization == "lp", (
                "gated_flux reads the flux the layer that solves returns; on a projection "
                f"layer ({flux_layer.parameterization!r}) the vector is off the polytope."
            )
            flux_mask = {
                "all": torch.ones(flux_layer.n_reactions, dtype=torch.bool),
                "exchange": flux_layer.is_exchange.detach().cpu().bool(),
                "gene": flux_layer.has_gpr.detach().cpu().bool(),
            }[cell.ess_flux_head_reactions]
            flux_meta["essentiality"]["flux_head"] = {
                "reactions": cell.ess_flux_head_reactions,
                "n_features": int(flux_mask.sum()),
            }
            wrapped_ess = GatedFluxReadout(model, biomass_index, flux_mask).to(device)
        else:
            readout_cls: type[FluxBiomassReadout] = {
                ESS_READOUT_BIOMASS: FluxBiomassReadout,
                ESS_READOUT_CONTEXT_BIOMASS: ContextBiomassReadout,
                ESS_READOUT_GATED: GatedBiomassReadout,
            }[cell.ess_readout]
            wrapped_ess = readout_cls(model, biomass_index).to(device)
        print(
            f"[{cell.ess_readout}-readout] essentiality from the predicted growth flux at "
            f"column {biomass_index} ({BIOMASS_REACTION})",
            flush=True,
        )
        return wrapped_ess, head_specs, main_head, flux_meta

    if main_head == ESSENTIALITY_HEAD and cell.ess_readout == ESS_READOUT_FBA:
        assert not arm["flux"]
        fba_table, fba_known, fba_media, fba_meta = affine_table(cell, gene_ids)
        flux_meta["essentiality"] = {
            "readout": cell.ess_readout,
            "fba": fba_meta,
            "note": (
                "essentiality is the pooled arm's context logit plus an affine of FBA's "
                "knockout-to-wild-type growth ratios for the deleted gene on four media, "
                "the feature the classical ladder appends to every encoding."
            ),
        }
        wrapped_fba = FbaContextReadout(model, fba_table, fba_known, fba_media).to(
            device
        )
        apply_solve_zero_init(cell, wrapped_fba.fba_readout)
        apply_solve_head(cell, wrapped_fba.fba_readout)
        print(
            f"[fba-readout] essentiality from the context head plus an affine of "
            f"{len(fba_media)} columns ({', '.join(fba_media[:4])}, ...) for "
            f"{fba_meta['n_genes_with_fba']} of {len(gene_ids)} genes",
            flush=True,
        )
        return wrapped_fba, head_specs, main_head, flux_meta

    if (
        main_head == ISOBUTANOL_HEAD
        and cell.ib_readout == IB_READOUT_PRECURSOR_CONCENTRATION
        and arm["flux"]
    ):
        assert flux_layer is not None
        anchors = isobutanol_anchors(cell.ib_anchors)
        indices, log_weights, described = species_mixture_weights(
            compartment_species_for(anchors),
            list(anchors),
            met_index,
            cell.conc_space,
            cell.volume_table,
        )
        flux_meta["isobutanol"]["anchor_set"] = cell.ib_anchors
        flux_meta["isobutanol"]["anchors"] = anchors
        flux_meta["isobutanol_precursor_mixture"] = {
            name: [
                {"compartment": c, "species": sid, "fractional_volume": w}
                for c, sid, w in entries
            ]
            for name, entries in described.items()
        }
        flux_meta["compartment_volume_fractions"] = compartment_volume_fractions(
            cell.volume_table
        ).model_dump()
        wrapped_ib = FluxPrecursorConcentrationReadout(
            model,
            indices.to(device),
            log_weights.to(device),
            param_dim,
            head_name=ISOBUTANOL_HEAD,
        ).to(device)
        print(
            "[precursor-readout] isobutanol from the whole-cell concentrations of "
            f"{ {n: [c for c, _, _ in v] for n, v in described.items()} }, window "
            f"[{flux_layer.config.conc_lower_m:.0e}, "
            f"{flux_layer.config.conc_upper_m:.0e}] M, volume table {cell.volume_table}",
            flush=True,
        )
        return wrapped_ib, head_specs, main_head, flux_meta

    if main_head == HEAD and cell.bx_readout == BX_READOUT_PRECURSOR_CONCENTRATION:
        assert flux_layer is not None
        indices, log_weights, described = species_mixture_weights(
            compartment_species_for(BETAXANTHIN_PRECURSOR_ANCHORS),
            list(BETAXANTHIN_PRECURSOR_ANCHORS),
            met_index,
            cell.conc_space,
            cell.volume_table,
        )
        flux_meta["betaxanthin_precursor_mixture"] = {
            name: [
                {"compartment": c, "species": sid, "fractional_volume": w}
                for c, sid, w in entries
            ]
            for name, entries in described.items()
        }
        flux_meta["compartment_volume_fractions"] = compartment_volume_fractions(
            cell.volume_table
        ).model_dump()
        wrapped_bx = FluxPrecursorConcentrationReadout(
            model, indices.to(device), log_weights.to(device), param_dim
        ).to(device)
        print(
            "[precursor-readout] betaxanthin from the whole-cell concentrations of "
            f"{ {n: [c for c, _, _ in v] for n, v in described.items()} }, window "
            f"[{flux_layer.config.conc_lower_m:.0e}, "
            f"{flux_layer.config.conc_upper_m:.0e}] M, volume table {cell.volume_table}",
            flush=True,
        )
        return wrapped_bx, head_specs, main_head, flux_meta

    if (
        main_head == HEAD
        and cell.bx_readout == BX_READOUT_CONTEXT_PRECURSOR_CONCENTRATION
    ):
        assert flux_layer is None and cell.flux_draws == 1, (
            "the context-only precursor readout is deterministic and layer-free; "
            f"got flux_draws={cell.flux_draws} on arm {cell.base_arm!r}."
        )
        # THE SAME METABOLITE AXIS THE LAYER WOULD HAVE, read from the GEM without building
        # the layer, exactly as the amino-acid ablation reads it.
        met_index = gem_metabolite_index()
        flux_meta["met_index"] = met_index
        indices, log_weights, described = species_mixture_weights(
            compartment_species_for(BETAXANTHIN_PRECURSOR_ANCHORS),
            list(BETAXANTHIN_PRECURSOR_ANCHORS),
            met_index,
            cell.conc_space,
            cell.volume_table,
        )
        flux_meta["betaxanthin_precursor_mixture"] = {
            name: [
                {"compartment": c, "species": sid, "fractional_volume": w}
                for c, sid, w in entries
            ]
            for name, entries in described.items()
        }
        flux_meta["compartment_volume_fractions"] = compartment_volume_fractions(
            cell.volume_table
        ).model_dump()
        flux_meta["context_concentration"] = {
            "n_metabolites": len(met_index),
            "conc_lower_m": cell.conc_lower_m,
            "conc_upper_m": cell.conc_upper_m,
            "note": (
                "the flux layer's ANCHORED concentration head on the pooled backbone with "
                "no flux layer, no stoichiometry, no kcat, no thermodynamics and no "
                "constraint term, read at the five betaxanthin precursor rows through the "
                "same whole-cell mixture and the same affine; the ablation of "
                "`flux_precursor_concentration`."
            ),
        }
        wrapped_ctx_bx = ContextPrecursorConcentrationReadout(
            model,
            indices.to(device),
            log_weights.to(device),
            param_dim,
            cell.hidden,
            len(met_index),
            cell.conc_lower_m,
            cell.conc_upper_m,
            sigma_fixed=cell.crps_sigma_fixed,
        ).to(device)
        print(
            "[context-precursor-readout] NO flux layer; betaxanthin from the whole-cell "
            f"concentrations of { {n: [c for c, _, _ in v] for n, v in described.items()} }"
            f", window [{cell.conc_lower_m:.0e}, {cell.conc_upper_m:.0e}] M over "
            f"{len(met_index)} GEM metabolite rows",
            flush=True,
        )
        return wrapped_ctx_bx, head_specs, main_head, flux_meta

    if main_head == AUX_HEAD and cell.aa_readout == AA_READOUT_CONCENTRATION:
        assert flux_layer is not None
        indices, log_weights, described = mixture_weights(
            met_index, cell.conc_space, cell.volume_table
        )
        flux_meta["concentration_mixture"] = {
            aa: [
                {"compartment": c, "species": sid, "fractional_volume": w}
                for c, sid, w in entries
            ]
            for aa, entries in described.items()
        }
        flux_meta["compartment_volume_fractions"] = compartment_volume_fractions(
            cell.volume_table
        ).model_dump()
        wrapped = FluxConcentrationReadout(
            model, indices.to(device), log_weights.to(device), cell.crps
        ).to(device)
        print(
            f"[concentration-readout] whole-cell mixture over "
            f"{ {aa: [c for c, _, _ in v] for aa, v in described.items()} }, window "
            f"[{flux_layer.config.conc_lower_m:.0e}, "
            f"{flux_layer.config.conc_upper_m:.0e}] M, "
            f"{'learned per-column log_sigma' if cell.crps else 'point prediction'}",
            flush=True,
        )
        if cell.flux_draws > 1:
            # The readout writes one [19, 2] row per row of `thermo_log_c`, which under
            # `log_c_from_latent` is one per draw, so the same reduction the betaxanthin
            # heads get applies here: `head_draws` for the per-draw loss, the moment-matched
            # mixture for the score. The head list is given explicitly because this arm
            # declares no head on the model.
            print(
                f"[flux-draws] {cell.flux_draws} latent draws per genotype on the "
                f"concentration readout; each draw carries its own log c",
                flush=True,
            )
            return (
                FluxDrawEnsemble(wrapped, cell.flux_draws, [AUX_HEAD]).to(device),
                head_specs,
                main_head,
                flux_meta,
            )
        return wrapped, head_specs, main_head, flux_meta

    if main_head == AUX_HEAD and cell.aa_readout == AA_READOUT_CONTEXT_CONCENTRATION:
        assert flux_layer is None and cell.flux_draws == 1, (
            "the context-only concentration readout is deterministic and layer-free; "
            f"got flux_draws={cell.flux_draws} on arm {cell.base_arm!r}."
        )
        # THE SAME METABOLITE AXIS THE LAYER WOULD HAVE, read from the GEM without building
        # the layer. Stored under the key the flux arms use so the concentration audit in
        # `run_cell` reads it the same way; it is filtered out of the result file as there.
        met_index = gem_metabolite_index()
        flux_meta["met_index"] = met_index
        indices, log_weights, described = mixture_weights(
            met_index, cell.conc_space, cell.volume_table
        )
        flux_meta["concentration_mixture"] = {
            aa: [
                {"compartment": c, "species": sid, "fractional_volume": w}
                for c, sid, w in entries
            ]
            for aa, entries in described.items()
        }
        flux_meta["compartment_volume_fractions"] = compartment_volume_fractions(
            cell.volume_table
        ).model_dump()
        flux_meta["context_concentration"] = {
            "n_metabolites": len(met_index),
            "conc_lower_m": cell.conc_lower_m,
            "conc_upper_m": cell.conc_upper_m,
            "note": (
                "the flux layer's ANCHORED concentration head (zero base per GEM "
                "metabolite + Linear(hidden, n_metabolites) from the perturbed-gene pool, "
                "sigmoid into the window) on the pooled backbone with no flux layer, no "
                "stoichiometry, no kcat, no thermodynamics and no constraint term; the "
                "ablation of `flux_concentration`."
            ),
        }
        wrapped_ctx = ContextConcentrationReadout(
            model,
            indices.to(device),
            log_weights.to(device),
            cell.crps,
            cell.hidden,
            len(met_index),
            cell.conc_lower_m,
            cell.conc_upper_m,
        ).to(device)
        print(
            f"[context-concentration-readout] NO flux layer; whole-cell mixture over "
            f"{ {aa: [c for c, _, _ in v] for aa, v in described.items()} }, window "
            f"[{cell.conc_lower_m:.0e}, {cell.conc_upper_m:.0e}] M over "
            f"{len(met_index)} GEM metabolite rows, "
            f"{'learned per-column log_sigma' if cell.crps else 'point prediction'}",
            flush=True,
        )
        return wrapped_ctx, head_specs, main_head, flux_meta

    if main_head == AUX_HEAD and arm["flux"] and cell.head_normalize:
        # THE HEAD FIX FOR THE AMINO-ACID READOUT. `FluxMetaboliteHead` emits
        # `a_k log(1 + omega_k) + b_k` with `a_k` initialized to 1, so against a target
        # standardized to mean 0 and sd 1 it starts at whatever `log1p(omega)` happens to be.
        # MEASURED on this dataset (smoke of 2026-09-10, the same recipe at seed 9001): the
        # flux arm's epoch-0 data loss is 2.4478 against the pooled arm's, which is the same
        # offset defect `--head-normalize` fixes for `FluxScalarHead`, and wave 5 measured
        # that defect holding the betaxanthin head on the mean plateau for 80 epochs.
        #
        # `FluxScalarHead` takes `normalize_inputs` and divides its input by
        # `log1p(flux_scale)`; `FluxMetaboliteHead` has no such switch. Since its slope is
        # learnable and multiplies exactly that input, dividing the INITIAL slope by
        # `log1p(flux_scale)` gives the identical forward pass at initialization without
        # touching the shared model file. It is an initialization, not an architecture
        # change: the slope is free to move anywhere from there.
        head = getattr(model, f"{AUX_HEAD}_head")
        with torch.no_grad():
            head.scale.div_(math.log1p(float(flux_layer.config.flux_scale)))
        print(
            f"[head-fix] {AUX_HEAD} slope initialized to "
            f"1/log1p({flux_layer.config.flux_scale}) = "
            f"{1.0 / math.log1p(float(flux_layer.config.flux_scale)):.4f}",
            flush=True,
        )
    if cell.flux_draws > 1:
        # ONE PLACE reduces the B*K head outputs to B, so no eval consumer has to know the
        # arm samples. The wrapper holds no parameters, so a checkpoint written through it
        # differs from the unwrapped one only by the `model.` prefix on every name.
        ensemble = FluxDrawEnsemble(model, cell.flux_draws, model.flux_head_names).to(
            device
        )
        print(
            f"[flux-draws] {cell.flux_draws} latent draws per genotype, reduced over heads "
            f"{model.flux_head_names}; the training loss is the mean over draws of the "
            f"per-draw score and the reported prediction is the draw ensemble",
            flush=True,
        )
        return ensemble, head_specs, main_head, flux_meta
    return model, head_specs, main_head, flux_meta


def run_cell(
    cell: BxCell,
    cfg: BxConfig,
    dataset: Any,
    pinned_test: set[int],
    pinned_report: dict[str, Any],
    on_epoch: Callable[[dict[str, Any]], None] | None = None,
    dump_predictions: bool = True,
    compute_feasibility: bool = True,
    checkpoint_stem: str | None = None,
    dump_stem: str | None = None,
    pinned_splits: dict[str, set[int]] | None = None,
) -> dict[str, Any]:
    """Train one cell, select an epoch on validation, score it on the pinned test genes.

    ``pinned_splits`` (from a split file) pins all three splits and drops every other
    record; ``pinned_test`` is then the file's test block and is passed for the report.

    ``dump_stem`` names the per-gene prediction dump under ``TEST_PRED_DIR`` and is the
    result file's stem, so it is required whenever ``dump_predictions`` is True.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, head_specs, main_head, flux_meta = build_model_for_cell(
        cell, cfg, dataset, device
    )

    dm = CellDataModule(
        dataset=dataset,
        # The split cache belongs to the BUILD, not to the config: a head whose phenotype
        # lives in another build opens another root, and a shared cache would hand it the
        # first build's split indices. For every head but isobutanol this is exactly
        # `cfg.data.dataset_subdir` under DATA_ROOT, so the path is unchanged.
        cache_dir=osp.join(dataset_paths(cell.head)[0], "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=cell.batch_size,
        # THE PINNED PARTITION. `random_seed` selects train/val/test, so it is held at
        # `split_seed` for every arm and every seed; `cell.seed` reached torch above and
        # varies weight INITIALIZATION only.
        random_seed=cell.split_seed,
        num_workers=cell.num_workers,
        pin_memory=True,
        prefetch=False,
        # `deletion_indices` joins follow_batch so `deletion_indices_batch` exists: the
        # flux layer's availability gate reads the DELETION-only index, because a
        # GeneAddition perturbation is not a removal (the Cachera cassette's ARO4 and ARO7
        # alleles are additions the strain overproduces through).
        follow_batch=["perturbation_indices", "deletion_indices", "phenotype_values"],
        pinned_test_indices=pinned_test if pinned_splits is None else None,
        pinned_split_indices=pinned_splits,
        drop_unpinned=pinned_splits is not None,
    )
    dm.setup()

    # THE FLUX LAYER'S OWN LEARNING RATE (round 19). Under the layer that solves the
    # backbone's path to the prediction is the bound duals, sparse and small, and the
    # forced column's training loss barely moves at the backbone's rate (round 18: 1.41
    # to 1.25 in 200 epochs against 1.33 to 0.31 for the free coordinate). A multiplier
    # on the layer's parameters is the smallest change that tests whether the step, not
    # the path, is the limit; 1.0 is one group and every earlier cell byte-identical.
    flux_params = [
        p for n, p in model.named_parameters() if _module_path(n) == "flux_layer"
    ]
    flux_ids = {id(p) for p in flux_params}
    other_params = [p for p in model.parameters() if id(p) not in flux_ids]
    if cell.flux_lr_mult != 1.0 and flux_params:
        groups: list[dict[str, Any]] = [
            {"params": other_params},
            {"params": flux_params, "lr": cell.lr * cell.flux_lr_mult},
        ]
        optimizer = torch.optim.AdamW(
            groups, lr=cell.lr, weight_decay=cell.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cell.lr, weight_decay=cell.weight_decay
        )

    # CRPS, not MSE. The head emits (mu, log_sigma) and is scored as a distribution;
    # 019 measured 0.413 -> 0.434 from this single change, and 020 measured crps 0.275
    # against point 0.172 across its sweep.
    # THE QUANTILE HEAD, when asked for. `DistHead` owns the tau grid and the median index,
    # so this trainer and 019's read the same grid; it holds no parameters in this mode, and
    # the CRPS path below is untouched, which is what keeps every earlier cell byte-identical.
    dist_head: DistHead | None = (
        make_dist_head(cell.dist).to(device) if cell.dist == "quantile" else None
    )

    binary = main_head in BINARY_HEADS
    # THE CLASS WEIGHT IS THE RELEASED SPLIT'S TRAINING-SIDE RATIO, 772 nonessential to
    # 126 essential, not our own train rows': the two differ only by the genes the cell
    # graph does not resolve and by the validation draw, and a constant read off the paper
    # is reproducible without a pass over the loader. Recorded in the result's cell.
    pos_weight: torch.Tensor | None = (
        torch.tensor([fcl_train_pos_weight()], device=device)
        if cell.pos_weight_balanced
        else None
    )

    def loss_fn(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if binary:
            # ONE LOGIT AGAINST A 0/1 LABEL. `binary_cross_entropy_with_logits` is the
            # numerically stable form, and `reduction="none"` keeps the elementwise shape
            # the caller's masked mean expects.
            return nn.functional.binary_cross_entropy_with_logits(
                prediction, target, reduction="none", pos_weight=pos_weight
            )
        if dist_head is not None:
            # Pinball, elementwise over the K quantiles and then averaged over them, so the
            # caller's `(loss * mask).sum() / mask.sum()` keeps meaning the masked mean.
            taus = dist_head.taus.to(prediction.device)
            if prediction.dim() == 3:
                return pinball(prediction, target, taus).mean(dim=-1)
            return (
                pinball(prediction, target.squeeze(-1), taus).mean(dim=-1).unsqueeze(-1)
            )
        if prediction.dim() == 3 and prediction.shape[-1] == 2:
            # A DISTRIBUTIONAL VECTOR head, `[B, F, 2]` against a `[B, F]` target. The
            # scalar branch below squeezes the target's feature axis and restores it after,
            # which on a width-19 head would broadcast `[B, 19, 1]` against a `[B, 19]`
            # mask into `[B, 19, 19]` and average a loss over pairs of amino acids.
            mu, log_sigma = prediction[..., 0], prediction[..., 1]
            sigma = torch.exp(log_sigma.clamp(-6.0, 3.0))
            return gaussian_crps(mu, sigma, target)
        if prediction.shape[-1] == 2:
            mu, log_sigma = prediction[..., 0], prediction[..., 1]
            # `gaussian_crps` takes sigma, not log sigma; the clamp keeps exp finite.
            sigma = torch.exp(log_sigma.clamp(-6.0, 3.0))
            return gaussian_crps(mu, sigma, target.squeeze(-1)).unsqueeze(-1)
        return nn.functional.mse_loss(prediction, target, reduction="none")

    # Target standardization from the TRAIN split only. A metric computed against
    # test-informed statistics is not a held-out metric.
    # A CLASS LABEL IS NEVER STANDARDIZED. Subtracting a train-split mean from a 0/1 label
    # and dividing by its sd would leave the cross entropy scoring a rescaled variable that
    # is no longer a probability's target, so the binary head is handed the identity and the
    # pass below is skipped. Every other head is unchanged.
    stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    if main_head in BINARY_HEADS:
        width = scored_width(head_specs[main_head])
        stats[main_head] = (torch.zeros(width).to(device), torch.ones(width).to(device))
        print(
            f"[binary-head] {main_head}: target left unstandardized (mean 0, sd 1) and "
            "scored with a binary cross entropy on one logit",
            flush=True,
        )
    else:
        chunks: dict[str, list[torch.Tensor]] = {name: [] for name in head_specs}
        for batch in dm.train_dataloader():
            for name, (t, m) in targets_in_space(
                batch, head_specs, cell.target_log10_molar
            ).items():
                chunks[name].append(torch.where(m, t, torch.nan))
        for name, parts in chunks.items():
            cat = torch.cat(parts, dim=0)
            mu = torch.nanmean(cat, dim=0)
            centered = cat - mu
            s = torch.nanmean(centered * centered, dim=0).sqrt().clamp(min=1e-6)
            stats[name] = (mu.to(device), s.to(device))
    mean, sd = stats[main_head]
    # THE SUPERVISED COLUMNS. None is all of them and is what every cell before wave 9 ran,
    # so `supervised` is then the identity and every tensor below is unchanged. A subset
    # masks the LOSS and the epoch-selection metric; the test scorer still reports all 19
    # per-column, with the two groups summarized in `aa_column_split`.
    supervised_columns = (
        list(range(scored_width(head_specs[main_head])))
        if cell.aa_supervise_columns is None
        else [AA_KEYS.index(name) for name in cell.aa_supervise_columns]
    )
    held_out_columns = [
        i
        for i in range(scored_width(head_specs[main_head]))
        if i not in supervised_columns
    ]
    column_mask = torch.zeros(scored_width(head_specs[main_head]), dtype=torch.bool)
    column_mask[supervised_columns] = True
    column_mask = column_mask.to(device)
    if held_out_columns:
        print(
            f"[aa-columns] supervising {len(supervised_columns)} of {len(AA_KEYS)} "
            f"columns; held out {[AA_KEYS[i] for i in held_out_columns]}",
            flush=True,
        )
    # A K-draw concentration cell is the readout inside the draw ensemble; the
    # standardization belongs to the readout either way.
    readout = model.model if isinstance(model, FluxDrawEnsemble) else model
    if isinstance(readout, ConcentrationReadoutBase):
        # The readout emits a log10 molar concentration and the loss is computed against
        # the standardized target, so the readout is handed the SAME train-split statistics
        # rather than a learned affine. Set here rather than in `build_model_for_cell`
        # because the statistics come from a pass over the training loader, which the model
        # has to exist for.
        readout.set_target_standardization(mean, sd)
        print(
            "[concentration-readout] target standardization in log10 M: mean "
            f"{[round(float(x), 3) for x in mean]}, sd "
            f"{[round(float(x), 3) for x in sd]}",
            flush=True,
        )

    window = cell.select_smoothing_window
    half = window // 2
    #: ``(epoch, trainable-parameter snapshot)``, `window` deep. Keyed by epoch rather than
    #: indexed by position: a run whose validation curve is entirely non-finite selects epoch
    #: 0, which position arithmetic would resolve to a snapshot the ring no longer holds.
    ring: deque[tuple[int, dict[str, torch.Tensor]]] = deque(maxlen=window)
    val_curve: list[float] = []
    history: list[dict[str, Any]] = []
    best_smoothed = -np.inf
    best_snap: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    t0 = time.time()

    for epoch in range(cell.epochs):
        model.train()
        train_loss, train_data_loss, n_batches = 0.0, 0.0, 0
        # THE CONSTRAINT RAMP, set once per epoch before any batch. Epoch 0 carries no
        # physics at all when a warmup is configured, so the data term is the only thing
        # moving the head while it is on the mean plateau.
        constraint_scale = 1.0
        epoch_layer: Any = model.flux_layer
        if epoch_layer is not None:
            constraint_scale = epoch_layer.set_constraint_epoch(epoch)
        train_constraint_loss = 0.0
        feas: dict[str, float] = {}
        # Per-epoch extrema and counts that must not be divided by the batch count the way
        # `feas` sums are: the largest pre-clip gradient norm, and how many batches clipped.
        grad_max: dict[str, float] = {}
        for batch in dm.train_dataloader():
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, reps = model(dataset.cell_graph.to(device), batch)
            t, m = targets_in_space(batch, head_specs, cell.target_log10_molar)[
                main_head
            ]
            t, m = t.to(device), m.to(device)
            if cell.permute_train_targets:
                # THE NULL: target and mask permuted TOGETHER, so the genotype-to-phenotype
                # association is destroyed while the target distribution and the missingness
                # rate are untouched. Validation and test are never permuted.
                perm = torch.randperm(t.shape[0], device=device)
                t, m = t[perm], m[perm]
            z = (t - mean) / sd
            # THE PER-DRAW SCORE, when the arm draws. `head_draws` carries the unreduced
            # B*K rows and the target is repeated to match, so the masked mean below is
            # exactly the mean over draws of the per-draw CRPS. Scoring the reduced
            # prediction instead would train the ensemble average and leave each draw free.
            if cell.flux_draws > 1:
                pred = _head_params(reps["head_draws"][main_head])
                z = z.repeat_interleave(cell.flux_draws, dim=0)
                m = m.repeat_interleave(cell.flux_draws, dim=0)
            else:
                pred = _head_params(reps["head_outputs"][main_head])
            # THE COLUMN MASK, applied to the LOSS only. An all-True mask leaves `m`
            # unchanged, so a cell that supervises every column computes the same number
            # the wave-8 cells did.
            m = m & column_mask.unsqueeze(0)
            data_loss = (loss_fn(pred, z) * m).sum() / m.sum().clamp(min=1)
            loss = data_loss + reps["graph_reg_loss"]
            if cell.aux_mulleder:
                # Masked per-feature MSE on the standardized amino-acid vector, weight 1,
                # as 019's head_weights had it. Never permuted: the null is on betaxanthin.
                t_aux, m_aux = targets_in_space(
                    batch, head_specs, cell.target_log10_molar
                )[AUX_HEAD]
                t_aux, m_aux = t_aux.to(device), m_aux.to(device)
                mu_aux, sd_aux = stats[AUX_HEAD]
                z_aux = (t_aux - mu_aux) / sd_aux
                p_aux = reps["head_outputs"][AUX_HEAD]
                if p_aux.dim() == 3:
                    p_aux = p_aux[..., 0]
                aux_loss = (
                    nn.functional.mse_loss(p_aux, z_aux, reduction="none") * m_aux
                ).sum() / m_aux.sum().clamp(min=1)
                loss = loss + aux_loss
                feas["aux_loss"] = feas.get("aux_loss", 0.0) + float(aux_loss.detach())
            train_data_loss += float(data_loss.detach())
            if "flux" in reps:
                # `reps` carries "flux" only when the model was built WITH a layer, so this
                # cannot be None here. Asserted rather than type-ignored: if it ever were
                # None the constraint terms would silently vanish and the arm would still
                # report a plausible number.
                batch_layer: Any = model.flux_layer
                assert batch_layer is not None
                constraint_term = batch_layer.constraint_loss(reps["flux"])
                loss = loss + constraint_term
                train_constraint_loss += float(constraint_term.detach())
                for k, v in reps["flux"].items():
                    # 'latent_' joins the prefixes at wave 8: a sampled arm's posterior sd
                    # is the number that separates a sampler from a point estimate, and the
                    # KL alone cannot report it because it mixes the mu and sigma halves.
                    if v.dim() == 0 and k.startswith(
                        ("c_", "feas_", "g_diss", "protein_used", "latent_")
                    ):
                        feas[k] = feas.get(k, 0.0) + float(v.detach())
            if "flux_draw_pred_sd" in reps:
                # The across-draw sd of the head's own prediction, in standardized target
                # units. Logged beside the posterior sd because the two can disagree: a
                # wide latent whose draws all clip onto the same box face moves the flux
                # and not the phenotype.
                feas["flux_draw_pred_sd"] = feas.get("flux_draw_pred_sd", 0.0) + float(
                    reps["flux_draw_pred_sd"].detach()
                )
            loss.backward()
            # GRADIENT TRACKING. The global norm BEFORE clipping (clip_grad_norm_ returns
            # it), accumulated as a per-epoch mean and max, and the same norm per top-level
            # module (embedding_preprocessor, transformer_layers, the head, the flux layer)
            # so a spiky validation curve can be read against whether the update itself is
            # spiky and where it lands. Requested 2026-09-15 for the essentiality cells.
            total_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            )
            feas["grad_norm_mean"] = feas.get("grad_norm_mean", 0.0) + total_norm
            grad_max["grad_norm_max"] = max(
                grad_max.get("grad_norm_max", 0.0), total_norm
            )
            grad_max["grad_clipped_frac"] = grad_max.get(
                "grad_clipped_frac", 0.0
            ) + float(total_norm > GRAD_CLIP_NORM)
            for module_name, module_norm in _module_grad_norms(model).items():
                key = f"grad_norm_{module_name}_mean"
                feas[key] = feas.get(key, 0.0) + module_norm
            optimizer.step()
            train_loss += float(loss.detach())
            n_batches += 1

        preds, trues = _predict(
            model,
            dm.val_dataloader(),
            dataset,
            head_specs,
            main_head,
            mean,
            sd,
            cell.target_log10_molar,
            dist_head,
        )
        # SELECTION READS THE SUPERVISED COLUMNS ONLY. With no mask this is every column
        # and the curve is the wave-8 curve; with one, selecting on a held-out column's
        # validation Pearson would be choosing the epoch by the very quantity the arm is
        # asking about.
        if binary:
            # THE SELECTION RULE'S BINARY ANALOG. Pearson between a logit and a 0/1 label
            # is not the endpoint; AUROC is what the comparator table reports, so the epoch
            # is chosen on validation AUROC under the SAME centered 5-epoch moving average.
            # A validation split carrying one class returns nan, which `_select_epoch`
            # already treats as unselectable.
            val_r = binary_auroc(preds[:, 0], trues[:, 0])
        else:
            val_r = _mean_pearson(
                preds[:, supervised_columns], trues[:, supervised_columns]
            )
        val_curve.append(val_r)
        ring.append((epoch, _params_snapshot(model)))
        # THE COLLAPSE DIAGNOSTIC. Pearson and Spearman are scale-free, so a head that
        # emits a constant plus jitter still returns a number; the spread ratio is what
        # says whether there is a prediction to correlate at all.
        pred_sd_ratio = _sd_ratio(preds, trues)
        if (
            cell.abort_on_collapse
            and epoch >= COLLAPSE_PATIENCE_EPOCHS
            and pred_sd_ratio < COLLAPSE_SD_RATIO
        ):
            raise SystemExit(
                f"[{cell.arm} s{cell.seed}] COLLAPSED: validation prediction spread is "
                f"{pred_sd_ratio:.2e} of the target spread at epoch {epoch}. Stopping "
                "rather than spending the budget on a constant predictor."
            )

        # The centered window can only score epoch `t - half` once epoch `t` exists, which is
        # exactly why the ring is `window` deep: the snapshot for that epoch is still in it.
        if len(val_curve) >= window:
            centered_epoch = epoch - half
            score = float(np.mean(val_curve[-window:]))
            if score > best_smoothed:
                best_smoothed = score
                best_snap = ring[-(half + 1)][1]
                best_epoch = centered_epoch

        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(n_batches, 1),
            # The data term alone, which is the one comparable with `val_loss`.
            "train_data_loss": train_data_loss / max(n_batches, 1),
            "val_loss": _masked_loss(
                model,
                dm.val_dataloader(),
                dataset,
                head_specs,
                main_head,
                mean,
                sd,
                loss_fn,
                cell.target_log10_molar,
                column_mask,
            ),
            "val_pearson": val_r,
            "val_pred_sd_ratio": pred_sd_ratio,
            "n_val": int(np.isfinite(trues).sum()),
            "graph_reg_loss": float(reps["graph_reg_loss"]),
            # The two numbers the warmup experiment is read on, logged BESIDE the spread
            # ratio so the epoch at which the physics binds and the epoch at which the head
            # leaves the plateau are on the same axis.
            "constraint_scale": constraint_scale,
            "train_constraint_loss": train_constraint_loss / max(n_batches, 1),
        }
        for k, v in feas.items():
            row[k] = v / max(n_batches, 1)
        if grad_max:
            row["grad_norm_max"] = grad_max["grad_norm_max"]
            row["grad_clipped_frac"] = grad_max["grad_clipped_frac"] / max(n_batches, 1)
        if isinstance(model, GatedBiomassReadout):
            # The gate IS the usage readout: 0 is the pooled model exactly.
            row["ess_gate"] = float(model.gate.detach())
        if isinstance(model, GatedFluxReadout):
            row["ess_flux_head_w_norm"] = float(model.flux_head.weight.detach().norm())
        if isinstance(model, FbaContextReadout | BiomassFbaReadout | GatedFbaReadout):
            # The weights say whether, and on which medium, the solver's number is used.
            row.update(model.fba_readout.weights())
        history.append(row)
        if on_epoch is not None:
            on_epoch(row)
        print(
            f"[{cell.arm} s{cell.seed}] ep {epoch:3d} loss {row['train_loss']:.4f} "
            f"data {row['train_data_loss']:.4f} cstr {row['train_constraint_loss']:.4f} "
            f"scale {constraint_scale:.3f} "
            f"val_loss {row['val_loss']:.4f} val_r {val_r:+.4f} "
            f"sd_ratio {pred_sd_ratio:.4f} best_ep {best_epoch}",
            flush=True,
        )

    sel_epoch, sel_rule = _select_epoch(val_curve, window)
    snap_by_epoch = dict(ring)
    if best_snap is not None and sel_rule.startswith("centered_ma"):
        # The ring selects online (it must -- a centered window needs a snapshot that is
        # about to fall out of memory) and `_select_epoch` recomputes offline from the saved
        # curve. They are two implementations of ONE rule, so they must agree, and a silent
        # divergence would mean the reported epoch is not the epoch that was scored.
        assert sel_epoch == best_epoch, (
            f"selection disagreement: online ring picked {best_epoch}, offline recompute "
            f"picked {sel_epoch}. The ring and `_select_epoch` must implement one rule."
        )
        _restore(model, best_snap)
    else:
        # A run shorter than the smoothing window (a canary, or a wall-clock truncation)
        # falls to the raw argmax, and `select_rule` in the result says so.
        assert sel_epoch in snap_by_epoch, (
            f"selected epoch {sel_epoch} has no snapshot; the ring holds "
            f"{sorted(snap_by_epoch)}. This means every validation epoch was non-finite."
        )
        _restore(model, snap_by_epoch[sel_epoch])

    test = _score_test(
        model,
        dm,
        dataset,
        head_specs,
        main_head,
        mean,
        sd,
        cell,
        dump_predictions,
        dist_head,
        dump_stem,
    )
    result: dict[str, Any] = {
        "cell": cell.model_dump(),
        "deterministic": DETERMINISTIC,
        "select_epoch": sel_epoch,
        "select_rule": sel_rule,
        "select_val_pearson": val_curve[sel_epoch] if val_curve else float("nan"),
        "test": test,
        "history": history,
        "flux_meta": {k: v for k, v in flux_meta.items() if k != "met_index"}
        | _sampling_meta(model, cell),
        # WHICH INDEX THE AVAILABILITY GATE READ. A GeneAddition perturbation is not a
        # removal, so the gate reads `deletion_indices` when the batch carries it. Recorded
        # because a batch built before that field existed silently falls back to
        # `perturbation_indices`, which is the behavior that shut DAHP synthase and
        # chorismate mutase in every Cachera genotype.
        "gene_removal_source": getattr(
            model.model if hasattr(model, "model") else model,
            "last_gene_removal_source",
            "not_recorded",
        ),
        "pinned_split": pinned_report,
        # THE TARGET SCALE THE HEAD WAS TRAINED AGAINST. Every head is standardized on the
        # TRAIN split (the pass above), and the loss is charged on `(t - mean) / sd`, so a
        # CRPS sigma head always sees a unit-variance target whatever the phenotype's raw
        # units. It lived only in the checkpoint until wave 9, which meant answering "was
        # the log10 fold change standardized?" needed a 2 GB file rather than the result.
        "target_standardization": {
            "head": main_head,
            "columns": AA_KEYS if main_head == AUX_HEAD else [main_head],
            "space": (
                "log10_molar"
                if cell.target_log10_molar
                else ("log10" if cell.target_log10 else "stored_units")
            ),
            "mean": [float(x) for x in mean],
            "sd": [float(x) for x in sd],
            "note": (
                "train-split mean and sd; the loss and every reported metric are computed "
                "on the standardized target, so the head is never asked to match a raw "
                "scale."
            ),
        },
        "n_parameters": model.num_parameters,
        "wall_time_s": time.time() - t0,
        "code_provenance": code_provenance(),
    }

    if main_head == AUX_HEAD:
        # NOT a held-out number, and labeled as one. The epoch was selected on this split,
        # so it is reported only because the pinned Merzbacher test genes carry few
        # amino-acid measurements and the reader needs to see the larger sample too.
        vp, vt = _predict(
            model,
            dm.val_dataloader(),
            dataset,
            head_specs,
            main_head,
            mean,
            sd,
            cell.target_log10_molar,
            dist_head,
        )
        result["validation_selection_split"] = {
            "note": (
                "the split the epoch was SELECTED on, so this is not held out; reported "
                "beside the test block because the pinned test genes carry few amino-acid "
                "measurements."
            ),
            **_score_test_vector(vp, vt, np.ones(len(vp), dtype=bool), AA_KEYS),
        }

    if main_head == AUX_HEAD:
        # WHICH COLUMNS WERE SUPERVISED AND WHICH WERE NOT, summarized from the per-feature
        # lists the scorer already produced, so the held-out-column arm is read without
        # re-deriving anything. On a cell that supervises all 19 the held-out group is
        # empty and both means are the group mean.
        result["aa_column_split"] = {
            "supervised": [AA_KEYS[i] for i in supervised_columns],
            "held_out": [AA_KEYS[i] for i in held_out_columns],
            "note": (
                "the loss and the epoch-selection metric were charged on the supervised "
                "columns only; the test scorer is unchanged and reports all 19."
            ),
            **{
                f"{group}_{name}_mean_spearman": (
                    float(
                        np.mean(
                            [
                                result["test"][group]["spearman_per_feature"][i]
                                for i in idx
                            ]
                        )
                    )
                    if idx
                    else float("nan")
                )
                for group in ("pinned", "all")
                for name, idx in (
                    ("supervised", supervised_columns),
                    ("held_out", held_out_columns),
                )
            },
        }

    if isinstance(
        readout,
        (FluxPrecursorConcentrationReadout, ContextPrecursorConcentrationReadout),
    ):
        # THE PRECURSOR POOL ITSELF, so a flat prediction can be attributed to a flat
        # concentration rather than to the affine on top of it.
        if main_head == ISOBUTANOL_HEAD:
            anchor_names = list(isobutanol_anchors(cell.ib_anchors))
        else:
            anchor_names = list(BETAXANTHIN_PRECURSOR_ANCHORS)
        layer_for_window: Any = model.flux_layer
        lo_c = (
            float(layer_for_window.config.conc_lower_m)
            if layer_for_window is not None
            else float(readout.conc_lower_m)
        )
        hi_c = (
            float(layer_for_window.config.conc_upper_m)
            if layer_for_window is not None
            else float(readout.conc_upper_m)
        )
        spread = collect_precursor_spread(
            model, dm.test_dataloader(), dataset, anchor_names, lo_c, hi_c
        )
        result["precursor_concentration"] = spread.model_dump()
        print(
            f"[precursor-spread] across-genotype sd of log10 c per anchor "
            f"{dict(zip(spread.anchors, [round(x, 6) for x in spread.sd_log10_c]))}; "
            f"frac at ceiling {spread.frac_at_ceiling:.3f}, floor "
            f"{spread.frac_at_floor:.3f}",
            flush=True,
        )

    if main_head == ISOBUTANOL_HEAD:
        result["isobutanol_external_check"] = isobutanol_external_check(
            model, dm, dataset, head_specs, mean, sd, cell, dist_head
        )

    # CALIBRATION at the selected epoch, on the pinned Merzbacher test genes. Reported for
    # every distributional head, so the CRPS and pinball arms are compared on what their
    # distribution does as well as on where its point estimate lands.
    pit, pit_mask = collect_pit(
        model,
        dm.test_dataloader(),
        dataset,
        head_specs,
        main_head,
        mean,
        sd,
        cell,
        dist_head,
    )
    if pit.size:
        pinned_rows = held_out_mask(dm, dataset, main_head)
        result["calibration"] = build_calibration_block(
            pit[pinned_rows],
            pit_mask[pinned_rows],
            cell,
            AA_KEYS if main_head == AUX_HEAD else [main_head],
            "pinned_test",
        ).model_dump()
    else:
        result["calibration"] = CalibrationBlock(
            available=False,
            reason="the head emitted a point prediction; there is no PIT for it.",
            dist=cell.dist,
        ).model_dump()

    layer_for_conc: Any = model.flux_layer
    context_readout = isinstance(readout, ContextConcentrationReadout)
    if main_head == AUX_HEAD and (
        context_readout
        or (
            layer_for_conc is not None
            and layer_for_conc.thermo_mode is ThermoMode.ANCHORED
        )
    ):
        # THE MECHANISTIC READOUT, computed for every anchored arm rather than only for the
        # supervised one: an arm that predicts amino acids from turnover still learns a
        # concentration, and whether that concentration is the measured concentration is a
        # separate question from whether the phenotype head scored well. The context-only
        # ablation is audited the same way on its own tensor, so its `conc/pinned_test/*`
        # numbers sit beside the flux cells' in the same views.
        indices, log_weights, described = mixture_weights(
            flux_meta["met_index"], cell.conc_space, cell.volume_table
        )
        c_pred, c_true = collect_concentration(
            model,
            dm.test_dataloader(),
            dataset,
            head_specs,
            indices.to(device),
            log_weights.to(device),
            source=CONC_SOURCE_CONTEXT if context_readout else CONC_SOURCE_FLUX,
        )
        pinned_mask = held_out_mask(dm, dataset, main_head)
        if context_readout:
            lo = float(readout.conc_lower_m)
            hi = float(readout.conc_upper_m)
        else:
            lo = float(layer_for_conc.config.conc_lower_m)
            hi = float(layer_for_conc.config.conc_upper_m)
        audit = ConcentrationAudit(
            available=True,
            supervised=cell.aa_readout in AA_LOG10_MOLAR_READOUTS,
            conc_space=cell.conc_space,
            conc_lower_m=lo,
            conc_upper_m=hi,
            volume_fractions=compartment_volume_fractions(cell.volume_table),
            mixture={
                aa: [
                    ConcentrationMixtureEntry(
                        compartment=c, species=sid, fractional_volume=w
                    )
                    for c, sid, w in entries
                ]
                for aa, entries in described.items()
            },
            groups={
                "pinned_test": _concentration_group(
                    c_pred, c_true, pinned_mask, lo, hi
                ),
                "test_all": _concentration_group(
                    c_pred, c_true, np.ones(len(c_pred), dtype=bool), lo, hi
                ),
            },
        )
    else:
        audit = ConcentrationAudit(
            available=False,
            reason=(
                "the learned log concentration exists only on a flux arm whose thermo_mode "
                "is ANCHORED, scored against the amino-acid phenotype; this cell is arm "
                f"{cell.arm!r} on head {main_head!r}."
            ),
        )
    result["concentration_audit"] = audit.model_dump()

    if compute_feasibility:
        layer: Any = model.flux_layer
        if layer is None:
            block = FeasibilityBlock(
                available=False,
                reason=(
                    f"arm {cell.arm!r} has no flux layer, so there is no flux vector and "
                    "neither mass balance nor the second law is defined for it. Reported "
                    "as null rather than as zero violations."
                ),
                definitions=FEASIBILITY_DEFINITIONS,
            )
        else:
            per_test = collect_feasibility(
                model, dm.test_dataloader(), dataset, main_head, mean, sd
            )
            per_all = collect_feasibility(
                model, dm.all_dataloader(), dataset, main_head, mean, sd
            )
            # The mask indexes the rows of `per_all`, and a K-draw arm has K of those per
            # genotype, so it is repeated the same way the prediction is. `np.repeat`
            # matches `repeat_interleave`: it duplicates each element in place, which is the
            # order the draws of one genotype arrive in.
            block = build_feasibility_block(
                per_all,
                per_test,
                np.repeat(
                    held_out_mask(dm, dataset, main_head), int(layer.config.draws)
                ),
                layer,
                main_head,
            )
        result["feasibility"] = block.model_dump()

    if checkpoint_stem is not None:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        ckpt_path = osp.join(CHECKPOINT_DIR, f"{checkpoint_stem}.pt")
        torch.save(
            {
                "experiment": "028-gene-essentiality",
                "cell": cell.model_dump(),
                "deterministic": DETERMINISTIC,
                "head": main_head,
                "select_epoch": sel_epoch,
                "select_rule": sel_rule,
                # The target standardization, without which the saved weights cannot be
                # turned back into a prediction in the phenotype's own units.
                "target_mean": mean.detach().cpu(),
                "target_sd": sd.detach().cpu(),
                "parameters": _params_snapshot(model),
                "n_parameters": model.num_parameters,
                "torch_version": torch.__version__,
            },
            ckpt_path,
        )
        result["checkpoint_path"] = ckpt_path
        print(f"[checkpoint] wrote {ckpt_path}", flush=True)
    return result


class PrecursorConcentrationSpread(BaseModel):
    """Does the precursor readout's INPUT vary across genotypes, or is it a constant?

    WHY THIS EXISTS (wave 9, 2026-09-14). Job 1928's isobutanol precursor arm held a
    validation prediction spread of 1e-4 of the target's for all 200 epochs while its
    concentration-head weight row at the anchor was 0.844 against a global median of 0.583,
    i.e. the head was not dead. Nothing in the result file could tell a flat CONCENTRATION
    apart from a dead AFFINE, and answering it needed the checkpoint plus a rebuild. One
    forward pass over the test split answers it directly: a near-zero `sd_log10_c` means the
    layer's pool is constant across strains and no affine on top of it can rank anything.
    """

    model_config = ConfigDict(extra="forbid")

    anchors: list[str]
    #: Across-genotype sd of the whole-cell log10 molar concentration, per anchor. This is
    #: the number that says whether there is anything for the affine to read.
    sd_log10_c: list[float]
    median_log10_c: list[float]
    min_log10_c: list[float]
    max_log10_c: list[float]
    #: Saturation of the layer's sigmoid window, which is the mechanism that would flatten
    #: the pool: within 0.01 of a decade of an edge the head cannot move the value further.
    frac_at_ceiling: float
    frac_at_floor: float
    conc_lower_m: float
    conc_upper_m: float
    n_records: int


def collect_precursor_spread(
    model: nn.Module,
    loader: Any,
    dataset: Any,
    anchors: list[str],
    conc_lower_m: float,
    conc_upper_m: float,
) -> PrecursorConcentrationSpread:
    """One forward pass, reading the precursor readout's own ``precursor_log10_c``."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, reps = model(dataset.cell_graph.to(device), batch)
            assert "precursor_log10_c" in reps, (
                "the model did not emit `precursor_log10_c`; this audit is defined only "
                "for the precursor-concentration readouts."
            )
            parts.append(reps["precursor_log10_c"].cpu().numpy())
    model.train(was_training)
    values = np.concatenate(parts)
    return PrecursorConcentrationSpread(
        anchors=anchors,
        sd_log10_c=[float(values[:, i].std()) for i in range(values.shape[1])],
        median_log10_c=[float(np.median(values[:, i])) for i in range(values.shape[1])],
        min_log10_c=[float(values[:, i].min()) for i in range(values.shape[1])],
        max_log10_c=[float(values[:, i].max()) for i in range(values.shape[1])],
        frac_at_ceiling=float((values >= math.log10(conc_upper_m) - 0.01).mean()),
        frac_at_floor=float((values <= math.log10(conc_lower_m) + 0.01).mean()),
        conc_lower_m=conc_lower_m,
        conc_upper_m=conc_upper_m,
        n_records=int(values.shape[0]),
    )


EXTERNAL_CHECK_K = 30


def isobutanol_external_check(
    model: nn.Module,
    dm: Any,
    dataset: Any,
    head_specs: dict[str, dict[str, Any]],
    mean: torch.Tensor,
    sd: torch.Tensor,
    cell: BxCell,
    dist_head: "DistHead | None",
) -> dict[str, Any]:
    """The model's prediction for every strain the validated re-screen measured.

    THE PRE-REGISTERED READ, in two halves. This half produces the predictions: one pass
    over the WHOLE dataset (``all_dataloader`` iterates ``self.dataset`` with
    ``shuffle=False``, so row ``i`` is record ``i``), restricted to the records the
    validated panel measured, which carry no label in this run because they are held out.
    The other half joins them to the independent re-screen's fold changes and reports what
    fraction of the predicted top and bottom 30 reproduced at FC >= 2 and FC <= 0.5; that
    lives in an analysis script, so this cell takes no dependency on the Lopez validated
    LMDB (which is being rebuilt as of 2026-09-14) and cannot die at the end of a 3-hour
    run because a build moved.

    The empirical ceilings the fractions are read against are the re-screen's own
    reproducibility, measured over the 222 genes that appear in both rounds: 59 of 60
    screen-up strains stayed at FC >= 2 and 150 of 150 screen-down strains stayed at
    FC <= 0.5.
    """
    preds, _ = _predict(
        model,
        dm.all_dataloader(),
        dataset,
        head_specs,
        ISOBUTANOL_HEAD,
        mean,
        sd,
        cell.target_log10_molar,
        dist_head,
    )
    assert len(preds) == len(dataset), (
        f"the external-check pass covered {len(preds)}/{len(dataset)} records; a short "
        "count would silently rank a subset."
    )
    held_records: set[int] = set()
    for combination, rows in dataset.dataset_name_index.items():
        if ISOBUTANOL_HELDOUT_DATASET in str(combination).split("+"):
            held_records.update(int(r) for r in rows)
    idx_to_genes: dict[int, list[str]] = {}
    for gene, rows in dataset.is_any_deletion_gene_index.items():
        for row in rows:
            idx_to_genes.setdefault(int(row), []).append(str(gene))
    rows_out = [
        {
            "record_index": rec,
            "genes": idx_to_genes.get(rec, []),
            "pred_log10_fold_change": float(preds[rec, 0]),
        }
        for rec in sorted(held_records)
    ]
    ranked = sorted(rows_out, key=lambda r: r["pred_log10_fold_change"], reverse=True)
    return {
        "note": (
            "predictions only; the fractions are computed by "
            "analyze_wave9_ib_external.py against the validated re-screen's own fold "
            "changes. These strains carry no label in this run: they are masked out of "
            "the loss, the validation curve and the test score."
        ),
        "held_out_dataset": ISOBUTANOL_HELDOUT_DATASET,
        "n_records": len(rows_out),
        "top_k": EXTERNAL_CHECK_K,
        "top": ranked[:EXTERNAL_CHECK_K],
        "bottom": ranked[-EXTERNAL_CHECK_K:],
        "predictions": rows_out,
        "reproducibility_ceiling": {
            "screen_up_stayed_up": "59/60",
            "screen_down_stayed_down": "150/150",
            "source": "the 222 genes measured in both the screen and the re-screen",
        },
    }


def _predict(
    model: nn.Module,
    loader: Any,
    dataset: Any,
    head_specs: dict[str, dict[str, Any]],
    head_name: str,
    mean: torch.Tensor,
    sd: torch.Tensor,
    log10_molar: bool = False,
    dist_head: "DistHead | None" = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Denormalized predictions and targets over a loader, ``[N, W]``.

    Unobserved entries of the target become NaN. The array is two-dimensional whatever the
    head's width, so a width-1 phenotype and a width-19 one are read the same way; the
    betaxanthin callers take column 0 and are unchanged by that.

    ``log10_molar`` puts BOTH sides in log10 molar, which is the space the concentration
    readout predicts in; the denormalized prediction and the target then have the same
    units, and the per-feature Spearman is unchanged by the monotone change of variable.
    """
    device = next(model.parameters()).device
    model.eval()
    ps: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    width = scored_width(head_specs[head_name])
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, reps = model(dataset.cell_graph.to(device), batch)
            t, m = targets_in_space(batch, head_specs, log10_molar)[head_name]
            raw = _head_params(reps["head_outputs"][head_name])
            pred = point_estimate(raw, width, dist_head)
            ps.append((pred * sd + mean).cpu().numpy().reshape(-1, width))
            ts.append(torch.where(m, t, torch.nan).cpu().numpy().reshape(-1, width))
    return np.concatenate(ps), np.concatenate(ts)


def _per_feature_pearson(preds: np.ndarray, trues: np.ndarray) -> list[float]:
    return [_masked_pearson(preds[:, f], trues[:, f]) for f in range(preds.shape[1])]


def _mean_pearson(preds: np.ndarray, trues: np.ndarray) -> float:
    """Mean per-feature Pearson. Identical to the scalar Pearson at width 1."""
    return float(np.mean(_per_feature_pearson(preds, trues)))


def _sd_ratio(preds: np.ndarray, trues: np.ndarray) -> float:
    """Mean per-feature prediction spread over target spread, the collapse diagnostic."""
    ratios = []
    for f in range(preds.shape[1]):
        obs = np.isfinite(trues[:, f])
        ratios.append(float(np.std(preds[obs, f]) / max(np.std(trues[obs, f]), 1e-12)))
    return float(np.mean(ratios))


def _masked_loss(
    model: nn.Module,
    loader: Any,
    dataset: Any,
    head_specs: dict[str, dict[str, Any]],
    head_name: str,
    mean: torch.Tensor,
    sd: torch.Tensor,
    loss_fn: Any,
    log10_molar: bool = False,
    column_mask: torch.Tensor | None = None,
) -> float:
    """The DATA-FIT loss on a held-out loader, in the same units as training.

    Separate from ``_predict`` because that returns the point prediction only: it slices
    ``pred[..., 0]`` and discards ``log_sigma``, so a CRPS cannot be reconstructed from what
    it hands back. The loss is computed on normalized targets, exactly as in training.

    This is the DATA term alone. The training loop's ``train_loss`` also carries the graph
    regularizer and, on a flux arm, the constraint terms, and neither depends on the labels,
    so adding them to a validation number would make the two curves incomparable in a way
    that is invisible on a plot. ``train_data_loss`` is logged beside it for that comparison.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    total, count = 0.0, 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, reps = model(dataset.cell_graph.to(device), batch)
            t, m = targets_in_space(batch, head_specs, log10_molar)[head_name]
            t, m = t.to(device), m.to(device)
            if column_mask is not None:
                # The same columns the training loss was charged on, so `val_loss` and
                # `train_data_loss` stay comparable.
                m = m & column_mask.to(m.device).unsqueeze(0)
            z = (t - mean) / sd
            pred = _head_params(reps["head_outputs"][head_name])
            total += float((loss_fn(pred, z) * m).sum())
            count += float(m.sum())
    model.train(was_training)
    return total / max(count, 1.0)


#: The test records of an active split file, filled by ``main`` when ``--split-file``
#: is given. Empty otherwise, and then essentiality pins the released FCL held-out
#: genes. A split whose test block is not the released genes (the genome held-out
#: split of round 30) scored an EMPTY pinned block before this existed, and the
#: summary step died on it after the dump was written (jobs 2396 and 2397).
SPLIT_TEST_RECORDS: set[int] = set()


def held_out_mask(dm: Any, dataset: Any, head: str) -> np.ndarray:
    """The rows the reported number is computed on, per head.

    Betaxanthin, the amino acids and isobutanol are scored on Merzbacher's 639-gene
    betaxanthin pin; essentiality is scored on the test block of the split file when
    one is active, else on the 223 genes the FCL essentiality split held out, because
    those are the genes its published table was computed on.
    """
    if head == ESSENTIALITY_HEAD:
        if SPLIT_TEST_RECORDS:
            order = [int(i) for i in dm.test_dataset.indices]
            return np.array([i in SPLIT_TEST_RECORDS for i in order], dtype=bool)
        return fcl_test_mask(dm, dataset)
    return pinned_test_mask(dm, dataset)


def fcl_test_mask(dm: Any, dataset: Any) -> np.ndarray:
    """Which rows of the TEST loader belong to a released FCL test gene."""
    import csv

    order = [int(i) for i in dm.test_dataset.indices]
    path = osp.join(DATA_ROOT, FCL_ESSENTIALITY_SPLIT)
    test_genes = {
        str(row["knockout"]).strip()
        for row in csv.DictReader(open(path))
        if str(row["test"]).strip() == "1" and str(row["essential"]).strip()
    }
    idx_to_genes: dict[int, list[str]] = {}
    for gene, rows in dataset.is_any_deletion_gene_index.items():
        for row in rows:
            idx_to_genes.setdefault(int(row), []).append(str(gene))
    mask = np.zeros(len(order), dtype=bool)
    for i, rec in enumerate(order):
        mask[i] = any(g in test_genes for g in idx_to_genes.get(rec, []))
    return mask


def pinned_test_mask(dm: Any, dataset: Any) -> np.ndarray:
    """Which rows of the TEST loader's fixed order belong to a pinned Merzbacher gene.

    ``test_dataloader`` is built with ``shuffle=False``, so the mask lines up row for row
    with any pass over that loader, which is what lets the feasibility pass and the scoring
    pass name the same subset.
    """
    order = [int(i) for i in dm.test_dataset.indices]
    idx_to_genes: dict[int, list[str]] = {}
    for gene, rows in dataset.is_any_deletion_gene_index.items():
        for row in rows:
            idx_to_genes.setdefault(int(row), []).append(str(gene))
    pin_path = osp.join(
        EXPERIMENT_ROOT, "020-cachera-betaxanthin/results/merzbacher_nested_split.json"
    )
    with open(pin_path) as fh:
        pinned_genes = set(json.load(fh)["split"]["test"])
    mask = np.zeros(len(order), dtype=bool)
    for i, rec in enumerate(order):
        mask[i] = any(g in pinned_genes for g in idx_to_genes.get(rec, []))
    return mask


def _score_test_vector(
    preds: np.ndarray, trues: np.ndarray, sel: np.ndarray, columns: list[str]
) -> dict[str, Any]:
    """Per-feature test metrics for a width-``F`` phenotype.

    The headline numbers are the MEAN over features of the per-feature Spearman and
    Pearson, and ``n_per_feature`` is reported beside them because the amino-acid columns
    are not measured on the same strains: a mean over features whose sample sizes differ by
    an order of magnitude is not a mean over a common set of strains, and the per-feature
    lists are what make that visible.
    """
    p, t = preds[sel], trues[sel]
    spearman = [_masked_spearman(p[:, f], t[:, f]) for f in range(p.shape[1])]
    pearson = [_masked_pearson(p[:, f], t[:, f]) for f in range(p.shape[1])]
    n_per_feature = [int(np.isfinite(t[:, f]).sum()) for f in range(p.shape[1])]
    any_observed = np.isfinite(t).any(axis=1)
    return {
        "n_records": int(sel.sum()),
        "n_records_with_any_feature": int(any_observed.sum()),
        "n_measurements": int(np.isfinite(t).sum()),
        "mean_spearman": float(np.mean(spearman)),
        "mean_pearson": float(np.mean(pearson)),
        "columns": columns,
        "spearman_per_feature": spearman,
        "pearson_per_feature": pearson,
        "n_per_feature": n_per_feature,
    }


def binary_block(logit: np.ndarray, label: np.ndarray) -> dict[str, Any]:
    """Deletion-level classification metrics at the 0.5 probability threshold.

    Reported per class because the yeast essentiality classes are 86/14 imbalanced, and
    reported beside the majority-class rate because FBA's published accuracy on these
    genes, 0.81, is BELOW the 0.861 a constant "nonessential" predictor scores. AUROC is
    the threshold-free comparator and is what the note reads.
    """
    if label.size == 0:
        return {"n": 0, "auroc": float("nan")}
    predicted = (logit > 0.0).astype(int)
    positive = label == 1
    negative = label == 0
    true_pos = float((predicted[positive] == 1).sum())
    true_neg = float((predicted[negative] == 0).sum())
    pred_pos = float((predicted == 1).sum())
    pred_neg = float((predicted == 0).sum())
    return {
        "n": int(label.size),
        "n_essential": int(positive.sum()),
        "n_nonessential": int(negative.sum()),
        "auroc": binary_auroc(logit, label),
        "accuracy": float((predicted == label).mean()),
        "majority_class_rate": float(max(positive.mean(), negative.mean())),
        "recall_essential": true_pos / max(float(positive.sum()), 1.0),
        "recall_nonessential": true_neg / max(float(negative.sum()), 1.0),
        "precision_essential": true_pos / pred_pos if pred_pos else float("nan"),
        "precision_nonessential": true_neg / pred_neg if pred_neg else float("nan"),
        "threshold": "logit > 0, i.e. predicted probability > 0.5",
    }


def _score_test(
    model: nn.Module,
    dm: Any,
    dataset: Any,
    head_specs: dict[str, dict[str, Any]],
    head_name: str,
    mean: torch.Tensor,
    sd: torch.Tensor,
    cell: BxCell,
    dump_predictions: bool,
    dist_head: "DistHead | None" = None,
    dump_stem: str | None = None,
) -> dict[str, Any]:
    """The prespecified test metrics, on the pinned genes and on the whole test split.

    ``test_dataloader`` is built with ``shuffle=False``, so batches arrive in the order of
    ``test_dataset.indices`` -- that is what lets a running cursor map row -> record -> gene.
    The gene name comes from the DELETION index for the reason spelled out in
    :func:`resolve_pinned_test`.
    """
    order = [int(i) for i in dm.test_dataset.indices]
    preds, trues = _predict(
        model,
        dm.test_dataloader(),
        dataset,
        head_specs,
        head_name,
        mean,
        sd,
        cell.target_log10_molar,
        dist_head,
    )
    assert len(preds) == len(order), (
        f"test pass covered {len(preds)}/{len(order)} records -- a short count is a silent "
        "truncation of the comparison set."
    )

    idx_to_genes: dict[int, list[str]] = {}
    for gene, rows in dataset.is_any_deletion_gene_index.items():
        for row in rows:
            idx_to_genes.setdefault(int(row), []).append(str(gene))

    is_pinned = held_out_mask(dm, dataset, head_name)
    all_rows = np.ones(len(order), dtype=bool)

    records: list[dict[str, Any]]
    if head_name in SCALAR_HEADS:
        p1, t1 = preds[:, 0], trues[:, 0]
        records = [
            {
                "record_index": rec,
                "genes": idx_to_genes.get(rec, []),
                "pred": float(p1[i]),
                "target": None if not np.isfinite(t1[i]) else float(t1[i]),
                "pinned": bool(is_pinned[i]),
            }
            for i, rec in enumerate(order)
        ]

        def block(sel: np.ndarray) -> dict[str, Any]:
            p, t = p1[sel], t1[sel]
            obs = np.isfinite(t)
            if head_name in BINARY_HEADS:
                # DELETION-LEVEL metrics, the level the comparator table reports at, and
                # per class because the classes are 86/14 imbalanced.
                return binary_block(p[obs], t[obs])
            return {
                "n": int(obs.sum()),
                "spearman": _masked_spearman(p, t),
                "pearson": _masked_pearson(p, t),
                **{
                    f"top{k}": _top_k_enrichment(p[obs], t[obs], k)
                    for k in (10, 25, 50)
                },
            }

        out: dict[str, Any] = {"pinned": block(is_pinned), "all": block(all_rows)}
    else:
        records = [
            {
                "record_index": rec,
                "genes": idx_to_genes.get(rec, []),
                "pred": [float(x) for x in preds[i]],
                "target": [None if not np.isfinite(x) else float(x) for x in trues[i]],
                "pinned": bool(is_pinned[i]),
            }
            for i, rec in enumerate(order)
        ]
        out = {
            "pinned": _score_test_vector(preds, trues, is_pinned, AA_KEYS),
            "all": _score_test_vector(preds, trues, all_rows, AA_KEYS),
        }

    if dump_predictions:
        assert dump_stem is not None, (
            "dump_predictions=True needs dump_stem, the result file's stem, to name the dump."
        )
        os.makedirs(TEST_PRED_DIR, exist_ok=True)
        # THE DUMP IS NAMED AFTER THE RESULT FILE, so two cells whose results are distinct
        # cannot share a dump. The stem used to be arm + head + readout + seed, which
        # carries no architecture: the wave-11 size ladder ran widths 12, 24, 54 and 90 on
        # the pooled essentiality arm, every one of them wrote
        # `pooled_essentiality_s<seed>.json`, and the last to finish won. Measured
        # 2026-09-16: the file fcl_comparison.py read as the wave-9 width-90 arm held the
        # width-12 cell, and the width-24 and wave-9 width-90 per-gene predictions were
        # gone (the result files keep their metrics; only the per-gene dumps were lost).
        # Wave 8 had already hit the same collision between `bxprec` and `bxscalar` on one
        # base arm and patched the stem with the readout name; naming by the result closes
        # every variant of it. `test.dump_path` records the file, and a reader resolves a
        # dump THROUGH the result that produced it, checking the dump's `cell` against the
        # result's, rather than by guessing a name.
        path = osp.join(TEST_PRED_DIR, f"{dump_stem}.json")
        with open(path, "w") as fh:
            json.dump(
                {
                    "experiment": "028-gene-essentiality",
                    "cell": cell.model_dump(),
                    "deterministic": DETERMINISTIC,
                    "head": head_name,
                    # WHICH SPACE the numbers below are in. The concentration readout
                    # predicts and is scored in log10 molar; every other arm reports the
                    # target in its stored units, which for the amino acids is mM. A dump
                    # that did not say so would put two different units under one key.
                    "target_space": (
                        "log10_molar" if cell.target_log10_molar else "stored_units"
                    ),
                    "n_test_records": len(order),
                    "metrics": out,
                    "predictions": records,
                },
                fh,
            )
        out["dump_path"] = path
        print(f"[test-dump] wrote {path} ({len(records)} records)", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=arm_names())
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--lr", type=float, default=None, help="AdamW learning rate (base.yaml: 1e-3)."
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="AdamW weight decay (base.yaml: 1e-4). A ladder knob for the 028 cells.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size (base.yaml: 128)."
    )
    parser.add_argument(
        "--ess-readout",
        choices=ESS_READOUTS,
        default=ESS_READOUT_BIOMASS,
        help=(
            "Flux arms: `biomass` reads the growth column alone; `context_biomass` "
            "adds the growth affine to the pooled arm's context head; `gated` does the "
            "same behind a zero-initialized gate, logged per epoch as ess_gate. Pooled "
            "arm: `fba` adds an affine of FBA's four per-medium growth ratios for the "
            "deleted gene to the context head, weights logged as ess_fba_w_<medium>. "
            "Flux arms: `biomass_fba` and `gated_fba` add the same affine beside the "
            "growth column."
        ),
    )
    parser.add_argument(
        "--node-embeddings",
        nargs="+",
        default=None,
        help=(
            "NodeEmbeddingBuilder names the dataset is opened with, replacing prot_T5_all; "
            "e.g. codon_frequency, or fudt_upstream fudt_downstream prot_T5_all calm."
        ),
    )
    parser.add_argument(
        "--ess-fba-media",
        nargs="+",
        choices=FBA_MEDIA,
        default=None,
        help=(
            "The media whose FBA growth ratios the fba readouts carry (default all four); "
            "one medium tests whether the four columns add anything beyond one."
        ),
    )
    parser.add_argument(
        "--flux-capacity-scale",
        type=float,
        default=1.0,
        help=(
            "Multiply the flux layer's enzyme-capacity ceilings by this factor; 1.0 is "
            "the ceiling as parameterized, which admits no growing flux (round 15)."
        ),
    )
    parser.add_argument(
        "--flux-lp-slack-penalty",
        type=float,
        default=0.0,
        help=(
            "Under --flux-parameterization lp, price per mmol/gDW/h of unmet forced flux "
            "in the elastic program; 0 is the hard program (round 18)."
        ),
    )
    parser.add_argument(
        "--flux-lr-mult",
        type=float,
        default=1.0,
        help="Learning-rate multiplier on the flux layer's parameters (round 19).",
    )
    parser.add_argument(
        "--flux-abundance-head-layers",
        type=int,
        default=1,
        choices=(1, 2),
        help="Depth of the abundance head: 1 linear (shipped), 2 with a hidden layer.",
    )
    parser.add_argument(
        "--flux-abundance-log-range",
        type=float,
        default=3.0,
        help="The abundance head's reach r (e^r either side of the prior); 3.0 shipped.",
    )
    parser.add_argument(
        "--ess-flux-head-reactions",
        default="all",
        choices=ESS_FLUX_HEAD_REACTIONS,
        help="Reactions the gated_flux readout's flux head reads (round 21).",
    )
    parser.add_argument(
        "--flux-pool-by-construction",
        action="store_true",
        help=(
            "Project the abundance head's output onto the protein budget before the "
            "capacity ceilings are formed, so the GECKO pool holds exactly (round 18)."
        ),
    )
    parser.add_argument(
        "--ess-solve-features",
        default="none",
        choices=ESS_SOLVE_FEATURES,
        help=(
            "Solve-derived per-gene columns appended to the fba readouts' affine after "
            "the media ratios: stats (eight wild-type pFBA/FVA statistics of the gene's "
            "reactions), pfba (the deletion's own pFBA flux vector), or both."
        ),
    )
    parser.add_argument(
        "--ess-solve-encoding",
        default="standardized",
        choices=ESS_SOLVE_ENCODINGS,
        help=(
            "How the pFBA vector's columns enter the affine: standardized (round 31) or "
            "log1p, the flux head's own encoding (round 33)."
        ),
    )
    parser.add_argument(
        "--ess-solve-zero-init",
        action="store_true",
        help=(
            "Zero the affine's weights on the solve-feature columns at construction, as "
            "the flux head is zero-initialized (round 33)."
        ),
    )
    parser.add_argument(
        "--ess-solve-head-hidden",
        type=int,
        default=0,
        help=(
            "Width of one GELU hidden layer in the fba readout's affine over the media "
            "ratios and the solve columns; 0 is the linear affine (round 41)."
        ),
    )
    parser.add_argument(
        "--pos-weight-balanced",
        action="store_true",
        help=(
            "Weight the essential class by the released split's training-side 772/126 in "
            "the binary cross entropy, so both classes carry equal loss."
        ),
    )
    parser.add_argument(
        "--split-file",
        default=None,
        help=(
            "A split file under results/splits/ from build_essentiality_splits.py: labels "
            "plus pinned train/val/test; unnamed records are dropped."
        ),
    )
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Skip the per-gene dump. For a local build/shape check, never for a result.",
    )
    parser.add_argument(
        "--crps", action="store_true", help="Gaussian CRPS on the head."
    )
    parser.add_argument(
        "--graph-reg", action="store_true", help="019's graph-structure regularizer."
    )
    parser.add_argument(
        "--abort-on-collapse",
        action="store_true",
        help="Stop once the validation prediction spread stays under 1%% of the target's.",
    )
    parser.add_argument(
        "--aux-mulleder",
        action="store_true",
        help="Train the 19-amino-acid Mulleder head jointly, as the 019 recipe did.",
    )
    parser.add_argument(
        "--attention-mask",
        action="store_true",
        help="The 020 grid's hard graph masking, one relation per attention head.",
    )
    parser.add_argument("--pert-heads", type=int, default=6)
    parser.add_argument(
        "--learnable-embedding",
        action="store_true",
        help="Memorization control: a free per-gene vector.",
    )
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument(
        "--hidden",
        type=int,
        default=None,
        help=(
            "Override the transformer width (base.yaml: 90). The model-size ladder on "
            "essentiality runs 90, 45 and 24; the width must be divisible by the head count."
        ),
    )
    parser.add_argument(
        "--attention-heads",
        type=int,
        default=None,
        help="Override the attention head count (base.yaml: 9); pair with --hidden.",
    )
    parser.add_argument(
        "--preprocessor-layers",
        type=int,
        default=2,
        help=(
            "Depth of the embedding preprocessor MLP (default 2). At 2 its first Linear is "
            "input_dim by (input_dim + hidden)/2 and carries most of the model whatever the "
            "width; 1 makes it a single input_dim by hidden projection, which is what a "
            "genuinely small cell needs."
        ),
    )
    parser.add_argument(
        "--kcat-tier",
        choices=("measured", "predicted"),
        default="measured",
        help="Insert the sequence-predictor kcat/K_M tier between measured and default.",
    )
    parser.add_argument(
        "--thermo-source",
        # THE ENUM IS THE LIST. A hardcoded tuple silently hid `verified` when it was added
        # to ThermoSource, so the flag reads the enum and cannot fall behind it again.
        choices=tuple(s.value for s in ThermoSource),
        default="shipped",
        help=(
            "Which route prices the standard reaction energies. 'union_external' is the "
            "same merge as 'union' with the external eQuilibrator table on the second side, "
            "which prices 3,738 of 4,131 reactions against union's 3,722 and un-double-"
            "converts the 33 dGPredictor rows. The default is unchanged."
        ),
    )
    parser.add_argument(
        "--constraint-warmup-epochs",
        type=int,
        default=0,
        help="Ramp the flux constraint weights linearly from 0 over this many epochs.",
    )
    parser.add_argument(
        "--constraint-scale",
        type=float,
        default=1.0,
        help="Final multiplier on the weighted constraint sum after warmup.",
    )
    parser.add_argument(
        "--constraint-delay-epochs",
        type=int,
        default=0,
        help="Hold the constraint scale at zero for this many epochs before ramping.",
    )
    parser.add_argument(
        "--head-normalize",
        action="store_true",
        help="Read the flux head's inputs in units of the layer's working flux magnitude.",
    )
    parser.add_argument(
        "--head",
        choices=(ESSENTIALITY_HEAD,),
        default=ESSENTIALITY_HEAD,
        help=(
            "Which phenotype the cell trains, selects and scores on. 'betaxanthin' is the "
            "wave-1 to wave-5 protocol and the default; 'mulleder19' moves all three to "
            "the 19 amino acids; 'isobutanol' moves them to the Montano Lopez 2024 "
            "biosensor fold change, which lives in the fig6_build dataset rather than in "
            "fig6_pigment_transfer and is trained on log10 of the fold change."
        ),
    )
    parser.add_argument(
        "--ib-readout",
        choices=IB_READOUTS,
        default=IB_READOUT_FLUX_SCALAR,
        help=(
            "Isobutanol from 026's flux scalar head (default) or from the learned "
            "whole-cell concentrations of its anchors. Ignored on --arm pooled."
        ),
    )
    parser.add_argument(
        "--ib-anchors",
        choices=IB_ANCHOR_SETS,
        default=IB_ANCHORS_ONE,
        help=(
            "'one' reads alpha-isopropylmalate alone, which is what Leu3p binds, and is "
            "the primary arm; 'three' adds 2-keto-isovalerate and L-valine."
        ),
    )
    parser.add_argument(
        "--aa-supervise-columns",
        default=None,
        help=(
            "Comma-separated amino-acid columns the LOSS is charged on; the rest are held "
            "out and reported separately. Omitted supervises all 19, which is what every "
            "earlier cell ran."
        ),
    )
    parser.add_argument(
        "--apply-media",
        choices=APPLY_MEDIA_CHOICES,
        default=MEDIA_NONE,
        help=(
            "Write the head's dataset medium onto the GEM's exchange bounds before the "
            "flux layer is built (SM for Mulleder, YPD for Cachera, SC for Lopez), and "
            "record the wild-type growth before and after. 'none' is the shipped boundary "
            "every earlier cell ran on."
        ),
    )
    parser.add_argument(
        "--aa-readout",
        choices=AA_READOUTS,
        default=AA_READOUT_TURNOVER,
        help=(
            "Which quantity of the flux layer the 19 amino acids are read off: "
            "'flux_metabolite' is wave 6's turnover readout with a free affine per column; "
            "'flux_concentration' predicts the layer's own learned log10 molar "
            "concentration with no parameters on the mean, and needs an ANCHORED arm; "
            "'context_concentration' is that same head built on the pooled backbone with "
            "no flux layer (the is-the-module-on ablation) and needs --arm pooled."
        ),
    )
    parser.add_argument(
        "--dist",
        choices=DISTS,
        default="crps",
        help=(
            "Distributional loss on the head: Gaussian CRPS (default) or the 019 pinball "
            "loss over 19 quantiles, whose point estimate is the predictive median. "
            "'quantile' applies to the MLP heads only, not to a flux readout."
        ),
    )
    parser.add_argument(
        "--crps-sigma-fixed",
        type=float,
        default=None,
        help=(
            "Hold the CRPS spread at this value in standardized units rather than "
            "predicting it; sqrt(1 - reliability) is the value the label supports."
        ),
    )
    parser.add_argument(
        "--bx-readout",
        choices=BX_READOUTS,
        default=BX_READOUT_FLUX_SCALAR,
        help=(
            "Betaxanthin from 026's flux scalar head (default) or from the learned "
            "whole-cell concentrations of its precursors."
        ),
    )
    parser.add_argument(
        "--volume-table",
        choices=VOLUME_TABLES,
        default=VOLUMES_ASSUMED,
        help=(
            "Fractional volumes for the whole-cell average: wave 7's 'assumed' textbook "
            "table, or 'sourced', Uchida et al. 2011's soft X-ray tomography measurement."
        ),
    )
    parser.add_argument(
        "--conc-space",
        choices=CONC_SPACES,
        default=CONC_SPACE_MIXTURE,
        help=(
            "Whole-cell 'mixture' over every intracellular compartment species of each "
            "amino acid, weighted by fractional volume, or cytosol-only as the control."
        ),
    )
    parser.add_argument(
        "--conc-lower-m",
        type=float,
        default=CONC_LOWER_M,
        help="Floor of the flux layer's concentration window, in molar.",
    )
    parser.add_argument(
        "--conc-upper-m",
        type=float,
        default=CONC_UPPER_M,
        help=(
            "Ceiling of that window, in molar. The shipped 1e-2 leaves 10.6 percent of "
            "Mulleder's measurements unreachable; see mulleder_concentration_window.json."
        ),
    )
    parser.add_argument(
        "--no-feasibility",
        action="store_true",
        help=(
            "Skip the two feasibility passes at the selected epoch. They add one forward "
            "pass over the test split and one over the whole dataset."
        ),
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Do not write the selected-epoch trainable parameters to $DATA_ROOT.",
    )
    parser.add_argument(
        "--wandb-group",
        default=None,
        help="Log this cell to W&B under the given group (project from WANDB_PROJECT).",
    )
    parser.add_argument(
        "--tag",
        default="",
        help=(
            "Suffix for the result file, a W&B tag, and (leading underscore stripped) the "
            "W&B GROUP, so every seed of one arm lands on one group page."
        ),
    )
    parser.add_argument(
        "--flux-balance-active-weight",
        type=float,
        default=0.0,
        help=(
            "Weight on the flux layer's active-row balance term, the relative residual over "
            "the rows that clear the turnover threshold. Also outside the constraint scale."
        ),
    )
    parser.add_argument(
        "--flux-density-weight",
        type=float,
        default=0.0,
        help=(
            "Weight on the flux layer's turnover-density term, which penalizes independent "
            "rows of S that do not turn over. It is NOT multiplied by the constraint "
            "scale, because the recipe these cells run holds that scale at 0.0."
        ),
    )
    parser.add_argument(
        "--flux-parsimony-weight",
        type=float,
        default=0.0,
        help=(
            "Weight on the flux layer's parsimony term, added outside the constraint "
            "scale. Parsimony is what makes an FBA flux sparse and it has never been "
            "applied: the recipe these cells run holds the ramp at 0.0."
        ),
    )
    parser.add_argument(
        "--flux-thermo-exempt-path",
        type=str,
        default=None,
        help=(
            "JSON exemption list from thermo_exempt_gate.py: reactions no concentration "
            "in the window can make downhill, exempted from the second-law tightening."
        ),
    )
    parser.add_argument(
        "--flux-log-c-init-path",
        type=str,
        default=None,
        help=(
            "JSON log-concentration witness from thermo_exempt_gate.py used to "
            "initialize log_c_base so the tightening licenses the reference flux."
        ),
    )
    parser.add_argument(
        "--flux-parameterization",
        default=None,
        help=(
            "Override the arm file's flux parameterization for this cell: box, nullspace, "
            "projection[_k<N>][_balance|_box][_tdg][_dykstra], ... (see "
            "torchcell.metabolism.flux_layer.parse_parameterization). The arm file's value "
            "stands when unset."
        ),
    )
    parser.add_argument(
        "--flux-shuffle-kcat",
        dest="flux_shuffle_kcat_seed",
        type=int,
        default=None,
        help=(
            "Permute the kcat vector across catalytic units with this seed: the "
            "shuffled-parameter control. Same architecture, same parameter count, same "
            "value distribution, no unit-to-value correspondence."
        ),
    )
    parser.add_argument(
        "--flux-reference-table",
        default=None,
        help=(
            "Per-genotype reference flux table (npz) from probe_reference_flux_table.py. "
            "The layer looks each row's reference up by its sorted deleted-gene set and "
            "reports the wild-type fallback rate; without it every genotype shares one "
            "reference, which is what job 1946 ran."
        ),
    )
    parser.add_argument(
        "--flux-alpha-frozen",
        action="store_true",
        help=(
            "Hold the warm start's residual scale at --flux-warm-start-init-rel with no "
            "gradient. It stays in the state dict, so frozen and learned checkpoints "
            "interchange."
        ),
    )
    parser.add_argument(
        "--flux-warm-start-init-rel",
        type=float,
        default=0.01,
        help=(
            "Initial relative distance of the proposal from the reference, and the value "
            "alpha is held at under --flux-alpha-frozen. Default is the shipped constant."
        ),
    )
    parser.add_argument(
        "--flux-availability-fold",
        type=str,
        default="softmin",
        choices=["softmin", "normalized_softmin"],
        help=(
            "GPR AND fold: 'softmin' (shipped) or 'normalized_softmin' (an intact "
            "complex reads exactly 1 instead of 1 - log(n)/beta)."
        ),
    )
    parser.add_argument(
        "--flux-draws",
        type=int,
        default=1,
        help=(
            "Latent draws per genotype per forward pass. Above 1 needs "
            "--flux-latent-stochastic; the data loss becomes the mean over draws of the "
            "per-draw score and the reported prediction is the draw ensemble."
        ),
    )
    parser.add_argument(
        "--flux-latent-stochastic",
        action="store_true",
        help=(
            "Draw the flux layer's latent, z = mu + sigma eps, with a KL against N(0, I). "
            "Held FIXED across both arms of a K-draw contrast, or the contrast confounds "
            "the draw count with whether the model samples at all."
        ),
    )
    parser.add_argument(
        "--flux-kl-weight",
        type=float,
        default=None,
        help=(
            "Override the coefficient on the latent KL (default 1e-3). It is multiplied by "
            "the constraint scale, so the applied weight is the product."
        ),
    )
    parser.add_argument(
        "--flux-log-c-from-latent",
        action="store_true",
        help=(
            "Let the anchored concentration read the drawn latent (zero-initialized U z "
            "term), so K draws give K concentrations per genotype and the "
            "flux_concentration readout can answer a K-draw contrast. Needs "
            "--flux-latent-stochastic."
        ),
    )
    args = parser.parse_args()

    if DETERMINISTIC:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        print(
            "[deterministic] cudnn.benchmark False, cudnn.deterministic True, "
            "use_deterministic_algorithms(True, warn_only=True); "
            f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'unset')}"
        )

    cfg = load_config()
    cell = build_cell(
        args.arm,
        args.seed,
        cfg,
        epochs=args.epochs,
        crps=args.crps,
        graph_reg=args.graph_reg,
        abort_on_collapse=args.abort_on_collapse,
        aux_mulleder=args.aux_mulleder,
        attention_mask=args.attention_mask,
        pert_heads=args.pert_heads,
        layers=args.layers,
        hidden=args.hidden,
        attention_heads=args.attention_heads,
        preprocessor_layers=args.preprocessor_layers,
        learnable_embedding=args.learnable_embedding,
        kcat_tier=args.kcat_tier,
        thermo_source=args.thermo_source,
        constraint_warmup_epochs=args.constraint_warmup_epochs,
        constraint_scale=args.constraint_scale,
        constraint_delay_epochs=args.constraint_delay_epochs,
        head_normalize=args.head_normalize,
        head=args.head,
        aa_readout=args.aa_readout,
        dist=args.dist,
        conc_space=args.conc_space,
        volume_table=args.volume_table,
        crps_sigma_fixed=args.crps_sigma_fixed,
        bx_readout=args.bx_readout,
        conc_lower_m=args.conc_lower_m,
        conc_upper_m=args.conc_upper_m,
        flux_density_weight=args.flux_density_weight,
        flux_balance_active_weight=args.flux_balance_active_weight,
        flux_parsimony_weight=args.flux_parsimony_weight,
        flux_thermo_exempt_path=args.flux_thermo_exempt_path,
        flux_log_c_init_path=args.flux_log_c_init_path,
        flux_availability_fold=args.flux_availability_fold,
        flux_shuffle_kcat_seed=args.flux_shuffle_kcat_seed,
        flux_parameterization=args.flux_parameterization,
        flux_reference_table=args.flux_reference_table,
        flux_alpha_frozen=args.flux_alpha_frozen,
        flux_warm_start_init_rel=args.flux_warm_start_init_rel,
        flux_draws=args.flux_draws,
        flux_latent_stochastic=args.flux_latent_stochastic,
        flux_kl_weight=args.flux_kl_weight,
        flux_log_c_from_latent=args.flux_log_c_from_latent,
        ib_readout=args.ib_readout,
        ib_anchors=args.ib_anchors,
        aa_supervise_columns=(
            None
            if args.aa_supervise_columns is None
            else [c.strip() for c in args.aa_supervise_columns.split(",") if c.strip()]
        ),
        apply_media=args.apply_media,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        split_file=args.split_file,
        ess_readout=args.ess_readout,
        ess_fba_media=args.ess_fba_media,
        ess_solve_features=args.ess_solve_features,
        ess_solve_encoding=args.ess_solve_encoding,
        ess_solve_zero_init=args.ess_solve_zero_init,
        ess_solve_head_hidden=args.ess_solve_head_hidden,
        flux_capacity_scale=args.flux_capacity_scale,
        flux_lp_slack_penalty=args.flux_lp_slack_penalty,
        flux_pool_by_construction=args.flux_pool_by_construction,
        flux_lr_mult=args.flux_lr_mult,
        flux_abundance_head_layers=args.flux_abundance_head_layers,
        flux_abundance_log_range=args.flux_abundance_log_range,
        ess_flux_head_reactions=args.ess_flux_head_reactions,
        node_embeddings=args.node_embeddings,
        pos_weight_balanced=args.pos_weight_balanced,
    )
    NODE_EMBEDDING_NAMES[:] = list(cell.node_embeddings)
    dataset = build_dataset(cell.head)
    print(f"dataset: {len(dataset)} aggregated genotypes", flush=True)
    pinned_splits: dict[str, set[int]] | None = None
    if cell.head == ESSENTIALITY_HEAD and cell.split_file is not None:
        # THE SPLIT FILE names every record's split; its test block is the comparison
        # set. For the fcl and genome splits that block is the released held-out genes'
        # records, the same set `resolve_fcl_test_indices` pins; for the genome held-out
        # split it is the 4,674 genes the released file never labeled, so the pinned
        # mask must follow the file rather than the released list.
        split = load_split_file(cell.split_file)
        pinned_splits = {
            k: {int(i) for i in split[k]} for k in ("train", "val", "test")
        }
        pinned = set(pinned_splits["test"])
        SPLIT_TEST_RECORDS.clear()
        SPLIT_TEST_RECORDS.update(pinned)
        report = {"path": split["path"], **split["report"]}
        print(
            f"[split-file] {split['path']}: records train {len(pinned_splits['train'])} "
            f"val {len(pinned_splits['val'])} test {len(pinned)}; unnamed records dropped",
            flush=True,
        )
    elif cell.head == ESSENTIALITY_HEAD:
        # THE COMPARATOR'S OWN HELD-OUT SET. Merzbacher's numbers are computed on those 223
        # genes; pinning anything else would compare two different experiments.
        pinned, report = resolve_fcl_test_indices(dataset)
    else:
        pinned, report = resolve_pinned_test(dataset, cfg)

    run = None
    if args.wandb_group:
        import wandb

        # The run NAME is `<host>-<slurm job id>_<sha256 of the config>` (019's standard):
        # unique, reproducible, nothing to parse. The run GROUP is the ARM-LEVEL TAG the
        # launcher passes (`--tag`, falling back to `--wandb-group`), so every seed of one
        # arm shares a group and W&B's group page
        # (`https://wandb.ai/<entity>/<project>/groups/<tag>`) shows the arm's runs
        # together with their aggregated curves. Until 2026-09-15 the group was the
        # per-job name, which put exactly one run in every group and made the group page
        # useless; the user reviews by grouped runs, and wandb_views.py rewrites the group
        # of older runs to the same rule. Everything else that distinguishes a run (wave,
        # machine, head, arm, seed) stays in TAGS.
        wandb_config = cell.model_dump()
        job_id = os.environ.get("SLURM_JOB_ID") or str(uuid.uuid4())
        hostname = socket.gethostname()
        hashed_cfg = hashlib.sha256(
            json.dumps(wandb_config, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        wandb_name = f"{hostname}-{job_id}_{hashed_cfg}"
        wandb_group = args.tag.lstrip("_") if args.tag else str(args.wandb_group)
        wandb_tags = [
            "028",
            "essentiality",
            args.wandb_group,
            hostname,
            cell.head,
            cell.arm,
            f"seed{cell.seed}",
            # THE DRAW COUNT AS ITS OWN TAG, derived from the cell rather than passed by
            # the launcher. A K-draw contrast whose arms share one hand-written tag cannot
            # be grouped by K in the W&B view, which is what the first wave-8 pairs did:
            # every cell carried `w8samp` whatever its K. Deriving it means the tag cannot
            # disagree with the value the run actually used.
            f"k{cell.flux_draws}",
        ]
        if args.tag:
            wandb_tags.append(args.tag.lstrip("_"))
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "torchcell_028_gene_essentiality"),
            group=wandb_group,
            name=f"run_{wandb_name}",
            tags=wandb_tags,
            config=wandb_config,
            mode=cast(Any, os.getenv("WANDB_MODE", "online")),
        )
    out = args.out or f"single_{cell.arm}{args.tag}_s{cell.seed}.json"
    result = run_cell(
        cell,
        cfg,
        dataset,
        pinned,
        report,
        dump_predictions=not args.smoke,
        on_epoch=(lambda row: wandb.log(row, step=int(row["epoch"]))) if run else None,
        compute_feasibility=not args.no_feasibility,
        checkpoint_stem=None if args.no_checkpoint else osp.splitext(out)[0],
        dump_stem=osp.splitext(out)[0],
        pinned_splits=pinned_splits,
    )
    if run is not None:
        pin = result["test"]["pinned"]
        summary: dict[str, Any] = {
            "select_epoch": result["select_epoch"],
            "select_val_pearson": result["select_val_pearson"],
            "wall_time_s": result["wall_time_s"],
        }
        if cell.head in BINARY_HEADS:
            # A CLASSIFIER REPORTS A CLASSIFIER'S METRICS. The scalar branch below reads
            # `spearman`, which `binary_block` does not emit and should not: a rank
            # correlation between a logit and a 0/1 label is not the endpoint, AUROC is.
            summary.update(
                {
                    "test_auroc_pinned": pin["auroc"],
                    "test_accuracy_pinned": pin["accuracy"],
                    "test_majority_rate_pinned": pin["majority_class_rate"],
                    "test_recall_essential_pinned": pin["recall_essential"],
                    "test_recall_nonessential_pinned": pin["recall_nonessential"],
                    "test_auroc_all": result["test"]["all"]["auroc"],
                    "n_test_pinned": pin["n"],
                    "n_test_pinned_essential": pin["n_essential"],
                }
            )
        elif cell.head in SCALAR_HEADS:
            summary.update(
                {
                    "test_spearman_pinned": pin["spearman"],
                    "test_pearson_pinned": pin["pearson"],
                    "n_test_pinned": pin["n"],
                }
            )
        else:
            summary.update(
                {
                    "test_mean_spearman_pinned": pin["mean_spearman"],
                    "test_mean_pearson_pinned": pin["mean_pearson"],
                    "test_mean_spearman_all": result["test"]["all"]["mean_spearman"],
                    "test_mean_pearson_all": result["test"]["all"]["mean_pearson"],
                    "n_test_pinned_measurements": pin["n_measurements"],
                    "n_test_pinned_records_with_any_feature": pin[
                        "n_records_with_any_feature"
                    ],
                }
            )
        if "feasibility" in result and result["feasibility"]["available"]:
            summary.update(
                flatten_feasibility_for_wandb(FeasibilityBlock(**result["feasibility"]))
            )
        if result["concentration_audit"]["available"]:
            for group, payload in result["concentration_audit"]["groups"].items():
                summary[f"conc/{group}/mean_spearman"] = payload["mean_spearman"]
                summary[f"conc/{group}/mean_pearson"] = payload["mean_pearson"]
                summary[f"conc/{group}/frac_pred_at_ceiling"] = payload[
                    "frac_pred_at_ceiling"
                ]
        run.summary.update(summary)
        result["wandb_run_id"] = run.id
        result["wandb_url"] = run.url
        run.finish()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = osp.join(RESULTS_DIR, out)
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    pinned_block = result["test"]["pinned"]
    if cell.head in BINARY_HEADS:
        headline = (
            f"  held-out AUROC {pinned_block['auroc']:+.4f} "
            f"(accuracy {pinned_block['accuracy']:.4f} against a majority rate of "
            f"{pinned_block['majority_class_rate']:.4f}; n={pinned_block['n']}, "
            f"{pinned_block['n_essential']} essential)"
        )
    else:
        headline = (
            f"  pinned test spearman {result['test']['pinned']['spearman']:+.4f} "
            f"(n={result['test']['pinned']['n']})"
            if cell.head in SCALAR_HEADS
            else (
                f"  pinned test mean per-feature spearman "
                f"{result['test']['pinned']['mean_spearman']:+.4f} "
                f"({result['test']['pinned']['n_measurements']} measurements on "
                f"{result['test']['pinned']['n_records_with_any_feature']} genotypes)"
            )
        )
    conc_line = ""
    if result["concentration_audit"]["available"]:
        pinned = result["concentration_audit"]["groups"]["pinned_test"]
        conc_line = (
            f"\n  learned concentration vs Mulleder, pinned test: mean spearman "
            f"{pinned['mean_spearman']:+.4f}, mean pearson {pinned['mean_pearson']:+.4f}, "
            f"frac at window ceiling {pinned['frac_pred_at_ceiling']:.3f}"
        )
    feas_line = ""
    if "feasibility" in result and result["feasibility"]["available"]:
        groups = result["feasibility"]["groups"]
        key = "all_top100" if "all_top100" in groups else "all"
        feas_line = f"\n  feasibility[{key}] {groups[key]['fractions']}"
    print(
        f"wrote {path}\n"
        f"  select_epoch {result['select_epoch']} ({result['select_rule']})\n"
        f"{headline}{conc_line}{feas_line}",
        flush=True,
    )


if __name__ == "__main__":
    main()
