# experiments/028-gene-essentiality/scripts/wave12_readout.py
# [[experiments.028-gene-essentiality.scripts.wave12_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/wave12_readout.py

r"""Read one round of 028 cells: per-arm AUROC with gene bootstrap, curve stability, figure.

One row per finished result file, one summary per arm (mean, sd, n over seeds), and one
multipanel figure:

- **a**, validation AUROC per epoch, one thin line per seed and the seed mean per arm,
  which is where the wave-9 spikiness is read against the validation size;
- **b**, the median absolute epoch-to-epoch step of that curve per arm, against the
  number of labeled validation genes the arm read it on;
- **c**, pinned-test AUROC per cell (points) with the per-arm seed mean and the gene
  bootstrap interval of the pooled per-gene predictions, beside the comparators computed
  on the same 195 genes: FBA on each medium (``fba_essentiality_baseline.py``) and Flux
  Cone Learning (``fcl_comparison.py``);
- **d**, per-module gradient norm per epoch (seed mean) for the arms that logged it.

Every number comes from ``results/<stem>.json`` and the per-gene dumps those results
name; a cell whose dump is missing is reported without an interval, never skipped.

    python experiments/028-gene-essentiality/scripts/wave12_readout.py --round w12
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils import (  # noqa: E402
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
RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "028-gene-essentiality")
PRED_DIR = os.getenv("TC_TEST_PRED_DIR", osp.join(DATA_ROOT, "test-predictions-028"))
FBA_JSON = osp.join(RESULTS, "fba_essentiality_by_medium.json")
FCL_JSON = osp.join(RESULTS, "fcl_comparison.json")
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0

#: Round -> (result globs, arm order for the figure). The arm of a result is the part of
#: its stem between the wave prefix and the architecture suffix; a round may pull a cell
#: from another wave in as its paired comparator.
ROUNDS: dict[str, tuple[list[str], list[str]]] = {
    "w12": (
        ["w12_*_e80_s*.json"],
        [
            "base",
            "fcl",
            "fcl_wd0",
            "fcl_wd1e-2",
            "fcl_wd1e-1",
            "fcl_b32",
            "gen",
            "fcl_lr3e-4",
            "fcl_lr3e-3",
            "fcl_posw",
            "fcl_noreg",
        ],
    ),
    # Wave 14: the two round-1 levers that moved three of three seeds (batch 32 and the
    # class weight) combined, the epoch budget raised to 200 since every fcl cell selected
    # its last smoothable epoch, and five-fold cross-validation on the 787 labeled
    # non-test genes whose fold models are averaged on the held-out genes, which is FCL's
    # own test protocol (`cv_ensemble_readout.py` scores the ensemble).
    "w14": (
        ["w14_*_s*.json"],
        ["b32posw", "b32posw200", "b32e200", "cv0", "cv1", "cv2", "cv3", "cv4"],
    ),
    # Wave 13: the prediction routed through the metabolic module, on the fcl split at
    # the base hyperparameters (batch 128, wd 1e-4, 80 epochs), paired by seed against the
    # w12 fcl pooled cell, which is the first arm of the round.
    "w13": (
        ["w12_fcl_h12l1p1_e80_s*.json", "w13_*_e80_s*.json"],
        [
            "fcl",
            "fluxbio",
            "fluxbio_s0",
            "fluxshuf_s0",
            "fluxctx",
            "fluxctx_s0",
            "fluxgate",
            "fluxgate_s0",
            "fluxoff",
            "fluxoffctx",
        ],
    ),
    # Wave 15 (round 4, jobs 2237-2245): the round-2 module readouts on top of the round-3
    # levers (batch 32, class weight, 200 epochs), paired by seed against the round-3
    # b32posw200 pooled cell, which is the first arm of the round.
    "w15": (
        ["w14_b32posw200_h12l1p1_e200_s*.json", "w15_*_e200_s*.json"],
        ["b32posw200", "gate_s0", "bio_s0", "shuf_s0"],
    ),
    # Wave 16 (round 5, packs): the pooled arm with FBA's four growth ratios added to the
    # context head (`--ess-readout fba`) on the round-3 levers, paired by seed against the
    # round-3 pooled cell, plus the same readout on the five cross-validation folds.
    "w16": (
        ["w14_b32posw200_h12l1p1_e200_s*.json", "w16_*_e200_s*.json"],
        ["b32posw200", "fba", "fbacv0", "fbacv1", "fbacv2", "fbacv3", "fbacv4"],
    ),
    # Wave 17 (round 6, packs): one medium against four, the flux layer on top of the
    # solve (forced and gated growth column plus the FBA affine), and two other node
    # embeddings under the backbone (codon frequency; the species-LM composite), all on
    # the round-3 levers and paired against the round-3 and round-5 pooled cells.
    "w17": (
        [
            "w14_b32posw200_h12l1p1_e200_s*.json",
            "w16_fba_h12l1p1_e200_s*.json",
            "w17_*_e200_s*.json",
        ],
        [
            "b32posw200",
            "fba",
            "fbaypd",
            "biofba",
            "gatefba",
            "codonfba",
            "compfba",
            "comp",
        ],
    ),
    # Wave 18 (round 7): the forced growth column on a layer whose bounds hold exactly
    # (projection ending on the box; the pure box form), against the round-4 forced cell
    # whose projection ends on the balance and leaves the box open, and the pooled cell.
    "w18": (
        [
            "w14_b32posw200_h12l1p1_e200_s*.json",
            "w15_bio_s0_h12l1p1_e200_s*.json",
            "w18_*_e200_s*.json",
        ],
        ["b32posw200", "bio_s0", "bioprojbox", "biobox"],
    ),
    # Wave 19 (round 8): width and depth on the FBA-readout cell, paired against the
    # round-5 width-12 cell.
    "w19": (
        ["w16_fba_h12l1p1_e200_s*.json", "w19_*_e200_s*.json"],
        ["fba", "fbah24", "fbah48", "fbal2", "fbap2"],
    ),
    # Round 9: the FBA affine on the ladder's other encodings (chromosome-pathway
    # indicator, ESM2, the seven-encoding concatenation, chromosome-pathway + ProtT5).
    "w20": (
        ["w16_fba_h12l1p1_e200_s*.json", "w20_*_e200_s*.json"],
        ["fba", "chromfba", "chromprotfba", "esmfba", "mixfba"],
    ),
    # Round 10: the flux layer warm-started from the deletion's own pFBA solution (one
    # reference per genotype), against the pooled cell, the round-4 forced and gated
    # columns, and the round-5 FBA affine.
    "w21": (
        [
            "w14_b32posw200_h12l1p1_e200_s*.json",
            "w15_bio_s0_h12l1p1_e200_s*.json",
            "w15_gate_s0_h12l1p1_e200_s*.json",
            "w16_fba_h12l1p1_e200_s*.json",
            "w21_*_e200_s*.json",
        ],
        [
            "b32posw200",
            "bio_s0",
            "gate_s0",
            "fba",
            "warmbio",
            "warmgate",
            "warmgatealpha",
            "warmgate10",
        ],
    ),
    # Round 11: the affine widened with the solve-derived columns (eight statistics of
    # the deleted gene's reactions under the wild-type solve, the deletion's pFBA vector).
    "w22": (
        ["w16_fba_h12l1p1_e200_s*.json", "w22_*_e200_s*.json"],
        ["fba", "fbastats", "fbapfba", "fbastatspfba"],
    ),
    # Round 13: the warm start on the stoichiometry-only layer, whose box is the GEM's
    # own bounds, so the deletion's pFBA reference sits inside it.
    "w24": (
        [
            "w14_b32posw200_h12l1p1_e200_s*.json",
            "w15_bio_s0_h12l1p1_e200_s*.json",
            "w15_gate_s0_h12l1p1_e200_s*.json",
            "w21_warmbio_h12l1p1_e200_s*.json",
            "w21_warmgate_h12l1p1_e200_s*.json",
            "w24_*_e200_s*.json",
        ],
        [
            "b32posw200",
            "bio_s0",
            "gate_s0",
            "warmbio",
            "warmgate",
            "warmoffbio",
            "warmoffgate",
        ],
    ),
    # Round 12: the round-9 passing cell on four more seeds (seven with round 9's) and on
    # the five folds, beside the round-5 cell and its folds.
    "w23": (
        [
            "w16_fba*_h12l1p1_e200_s*.json",
            "w20_chromprotfba_h12l1p1_e200_s*.json",
            "w23_*_e200_s*.json",
        ],
        [
            "fba",
            "chromprotfba",
            "fbacv0",
            "fbacv1",
            "fbacv2",
            "fbacv3",
            "fbacv4",
            "chromprotcv0",
            "chromprotcv1",
            "chromprotcv2",
            "chromprotcv3",
            "chromprotcv4",
        ],
    ),
    # Round 14: attribution of the round-9 gain inside the chromosome-pathway indicator,
    # dropping its pathway columns (normalized_chrom keeps the five properties and the
    # chromosome one-hot), with and without ProtT5.
    "w25": (
        [
            "w16_fba_h12l1p1_e200_s*.json",
            "w20_chromfba_h12l1p1_e200_s*.json",
            "w20_chromprotfba_h12l1p1_e200_s270[0-2].json",
            "w25_*_e200_s*.json",
        ],
        ["fba", "chromfba", "chromprotfba", "nchromfba", "nchromprotfba"],
    ),
    # Round 15: the layer that solves (growth LP under the layer's bounds, duals as the
    # gradient), forced and gated, with and without the second-law box, against the
    # pooled cell, the round-4 forced and gated columns, and the round-5 affine.
    "w26": (
        [
            "w14_b32posw200_h12l1p1_e200_s*.json",
            "w15_bio_s0_h12l1p1_e200_s*.json",
            "w15_gate_s0_h12l1p1_e200_s*.json",
            "w16_fba_h12l1p1_e200_s*.json",
            "w26_*_e200_s*.json",
        ],
        ["b32posw200", "bio_s0", "gate_s0", "fba", "lpbio", "lpbiotdg", "lpgate"],
    ),
    # Round 15b: the layer that solves on the stoichiometry-only box, beside the
    # round-13 warm-start cells on the same box and the round-5 affine.
    "w29": (
        [
            "w14_b32posw200_h12l1p1_e200_s*.json",
            "w16_fba_h12l1p1_e200_s*.json",
            "w24_*_e200_s*.json",
            "w29_*_e200_s*.json",
        ],
        ["b32posw200", "fba", "warmoffbio", "warmoffgate", "lpoffbio", "lpoffgate"],
    ),
    # Round 15c: the layer that solves on the anchored box with the capacity ceilings
    # scaled so the program admits growth, beside the round-15b cells on the GEM box.
    "w30": (
        [
            "w14_b32posw200_h12l1p1_e200_s*.json",
            "w16_fba_h12l1p1_e200_s*.json",
            "w29_*_e200_s*.json",
            "w30_*_e200_s*.json",
        ],
        [
            "b32posw200",
            "fba",
            "lpoffbio",
            "lpoffgate",
            "lpcapbio100",
            "lpcapgate100",
            "lpcapgate1000",
        ],
    ),
    # Round 15d: the forced layer that solves against the absorbing infeasible state,
    # with more capacity slack and with a third of the learning rate.
    "w31": (
        [
            "w14_b32posw200_h12l1p1_e200_s*.json",
            "w16_fba_h12l1p1_e200_s*.json",
            "w30_lpcapbio100_h12l1p1_e200_s*.json",
            "w31_*_e200_s*.json",
        ],
        ["b32posw200", "fba", "lpcapbio100", "lpcapbio1000", "lpcapbio100lr"],
    ),
    # Round 18, the forced column: the ELASTIC layer that solves, feasible for every
    # genotype by construction, against the round-4 forced column (the paired
    # comparator), the affine and the hard program's forced cells of rounds 15c and 15d.
    "w32": (
        [
            "w15_bio_s0_h12l1p1_e200_s*.json",
            "w16_fba_h12l1p1_e200_s*.json",
            "w30_lpcapbio100_h12l1p1_e200_s*.json",
            "w31_lpcapbio1000_h12l1p1_e200_s*.json",
            "w32_lpelbio*_h12l1p1_e200_s*.json",
            "w32_lpelpoolbio100_h12l1p1_e200_s*.json",
        ],
        [
            "bio_s0",
            "fba",
            "lpcapbio100",
            "lpcapbio1000",
            "lpelbio100",
            "lpelbio1000",
            "lpelpoolbio100",
        ],
    ),
    # Round 18, the gated column: the elastic layer that solves behind the gate against
    # the round-4 gated cell (paired), the affine, and the hard program's gated cells.
    "w33": (
        [
            "w15_gate_s0_h12l1p1_e200_s*.json",
            "w16_fba_h12l1p1_e200_s*.json",
            "w29_lpoffgate_h12l1p1_e200_s*.json",
            "w30_lpcapgate100_h12l1p1_e200_s*.json",
            "w32_lpelgate100_h12l1p1_e200_s*.json",
        ],
        ["gate_s0", "fba", "lpoffgate", "lpcapgate100", "lpelgate100"],
    ),
    # Round 19, the forced column: the elastic solve with the layer at ten times the
    # learning rate and with a two-layer abundance head, against the round-18 elastic
    # cell (paired), the round-4 forced column and the affine.
    "w34": (
        [
            "w32_lpelbio100_h12l1p1_e200_s*.json",
            "w15_bio_s0_h12l1p1_e200_s*.json",
            "w16_fba_h12l1p1_e200_s*.json",
            "w34_lpelbio100*_h12l1p1_e200_s*.json",
        ],
        [
            "lpelbio100",
            "bio_s0",
            "fba",
            "lpelbio100lr10",
            "lpelbio100mlp",
            "lpelbio100mlplr10",
        ],
    ),
    # Round 19, the gated column: the same two changes behind the gate against the
    # round-18 gated elastic cell (paired), the round-4 gated cell and the affine.
    "w35": (
        [
            "w32_lpelgate100_h12l1p1_e200_s*.json",
            "w15_gate_s0_h12l1p1_e200_s*.json",
            "w16_fba_h12l1p1_e200_s*.json",
            "w34_lpelgate100mlplr10_h12l1p1_e200_s*.json",
        ],
        ["lpelgate100", "gate_s0", "fba", "lpelgate100mlplr10"],
    ),
    # Round 20: the two gated forms on six seeds (rounds 15b and 18 plus seeds 2703 to
    # 2705), and the last two probes of the forced elastic column (scale 30; reach e^6).
    "w36": (
        [
            "w15_gate_s0_h12l1p1_e200_s*.json",
            "w16_fba_h12l1p1_e200_s*.json",
            "w29_lpoffgate_h12l1p1_e200_s*.json",
            "w36_lpoffgate_h12l1p1_e200_s*.json",
            "w32_lpelgate100_h12l1p1_e200_s*.json",
            "w36_lpelgate100_h12l1p1_e200_s*.json",
            "w32_lpelbio100_h12l1p1_e200_s*.json",
            "w36_lpelbio30lr10_h12l1p1_e200_s*.json",
            "w36_lpelbio100r6lr10_h12l1p1_e200_s*.json",
        ],
        [
            "gate_s0",
            "fba",
            "lpoffgate",
            "lpelgate100",
            "lpelbio100",
            "lpelbio30lr10",
            "lpelbio100r6lr10",
        ],
    ),
    # Round 21: the gated elastic cell plus a head over the solved flux vector.
    "w37": (
        [
            "w32_lpelgate100_h12l1p1_e200_s*.json",
            "w16_fba_h12l1p1_e200_s*.json",
            "w37_*_e200_s*.json",
        ],
        ["lpelgate100", "fba", "lpelgateflux", "lpelgatefluxexch"],
    ),
    # Round 22: the chromosome-pathway indicator split into its blocks on the round-9
    # recipe, paired against round 9 and beside round 14 (properties + chromosome).
    "w38": (
        [
            "w20_chromprotfba_h12l1p1_e200_s270[0-2].json",
            "w16_fba_h12l1p1_e200_s*.json",
            "w25_nchromprotfba_h12l1p1_e200_s*.json",
            "w38_*_e200_s*.json",
        ],
        [
            "chromprotfba",
            "fba",
            "nchromprotfba",
            "protpathfba",
            "protchromfba",
            "protpropsfba",
            "pathfba",
        ],
    ),
    # Round 23: the gated module forms on the ladder's top input (ProtT5 + the pathway
    # multi-hot), paired against the round-22 affine on that input and against the
    # pooled cell on that input with no readout.
    "w39": (
        [
            "w38_protpathfba_h12l1p1_e200_s*.json",
            "w39_*_e200_s*.json",
        ],
        [
            "pathpool",
            "protpathfba",
            "pathlpoffgate",
            "pathlpelgate100",
            "pathlpelgateflux",
        ],
    ),
    # Round 24: what carries round 21. The flux head over the GEM-box solve (no learned
    # ceilings), the head over the gene-owned reactions only, and the passing cell on
    # seeds 2703 to 2705 (its stem stays w37_, so the arm pools to six seeds).
    "w40": (
        [
            "w32_lpelgate100_h12l1p1_e200_s270[0-2].json",
            "w16_fba_h12l1p1_e200_s*.json",
            "w20_chromprotfba_h12l1p1_e200_s270[0-2].json",
            "w37_lpelgateflux_h12l1p1_e200_s*.json",
            "w37_lpelgatefluxexch_h12l1p1_e200_s*.json",
            "w40_*_e200_s*.json",
        ],
        [
            "lpelgate100",
            "fba",
            "chromprotfba",
            "lpelgateflux",
            "lpelgatefluxexch",
            "lpelgatefluxgene",
            "lpoffgateflux",
        ],
    ),
    # Round 26: the GO multi-hot as a backbone input on the round-9 recipe, against the
    # round-5 affine on ProtT5, the round-22 pathway cell, and the round-9 indicator.
    "w42": (
        [
            "w16_fba_h12l1p1_e200_s*.json",
            "w20_chromprotfba_h12l1p1_e200_s270[0-2].json",
            "w38_protpathfba_h12l1p1_e200_s*.json",
            "w42_*_e200_s*.json",
        ],
        [
            "fba",
            "chromprotfba",
            "protpathfba",
            "protgofba",
            "protgobpfba",
            "gofba",
            "protgopathfba",
        ],
    ),
    # Round 27: the scalable form, the flux head over the cached GEM-box solve on the
    # pathway input, six seeds, against the elastic form on that input, the control
    # and the affine.
    "w43": (
        [
            "w39_pathpool_h12l1p1_e200_s*.json",
            "w38_protpathfba_h12l1p1_e200_s*.json",
            "w39_pathlpelgateflux_h12l1p1_e200_s*.json",
            "w43_pathlpoffgatefluxfirst_h12l1p1_e200_s*.json",
            "w43_pathlpoffgateflux_h12l1p1_e200_s*.json",
        ],
        [
            "pathpool",
            "protpathfba",
            "pathlpelgateflux",
            "pathlpoffgatefluxfirst",
            "pathlpoffgateflux",
        ],
    ),
    # Round 29: the round-23 seed that never fit, rerun with a 400-epoch budget, beside
    # its 200-epoch run; the same program and seed, so the read is whether the fitting
    # transition is delayed or absent on that seed.
    "w45": (
        [
            "w39_pathlpelgateflux_h12l1p1_e200_s2701.json",
            "w45_pathlpelgateflux_h12l1p1_e400_s2701.json",
        ],
        ["pathlpelgateflux", "pathlpelgateflux400"],
    ),
    # Round 31: the layer-free form. The deletion's own pFBA vector precomputed and
    # appended to the FBA affine (no flux layer in the graph), with and without the
    # eight wild-type solve statistics, against the affine and the cached-solve flux
    # head on the same pathway input; six seeds each.
    "w47": (
        [
            "w38_protpathfba_h12l1p1_e200_s*.json",
            "w43_pathlpoffgateflux_h12l1p1_e200_s*.json",
            "w47_pathfbapfba_h12l1p1_e200_s*.json",
            "w47_pathfbastatspfba_h12l1p1_e200_s*.json",
        ],
        ["protpathfba", "pathlpoffgateflux", "pathfbapfba", "pathfbastatspfba"],
    ),
    # Round 32: the round-27 cell on seed 2701 twice with the card's kernels pinned
    # (ESS_DETERMINISTIC=1) and twice free; the read is per pair.
    "w48": (
        [
            "w43_pathlpoffgateflux_h12l1p1_e200_s*.json",
            "w48_pathlpoffgateflux*_h12l1p1_e200_s2701.json",
            "w48_pathlpoffgatefluxdet_h12l1p1_e200_s*.json",
        ],
        [
            "pathlpoffgateflux",
            "pathlpoffgatefluxdet1",
            "pathlpoffgatefluxdet2",
            "pathlpoffgatefluxfree1",
            "pathlpoffgatefluxfree2",
            "pathlpoffgatefluxdet",
        ],
    ),
    # Round 33: the layer-free form's factorial of encoding and initialization, paired
    # by seed against round 31's cell, with the cached-solve head as the comparator.
    "w49": (
        [
            "w43_pathlpoffgateflux_h12l1p1_e200_s*.json",
            "w47_pathfbapfba_h12l1p1_e200_s*.json",
            "w49_pathfbapfbazero_h12l1p1_e200_s*.json",
            "w49_pathfbapfbalogfree_h12l1p1_e200_s*.json",
            "w49_pathfbapfbalog_h12l1p1_e200_s*.json",
        ],
        [
            "pathlpoffgateflux",
            "pathfbapfba",
            "pathfbapfbazero",
            "pathfbapfbalogfree",
            "pathfbapfbalog",
        ],
    ),
    # Round 35: the layer-free log1p form on ProtT5 alone, against the affine on ProtT5
    # (round 5), the cached-solve head on ProtT5 (round 24) and the same form with the
    # pathway block (round 33): does the vector substitute for the annotation.
    "w51": (
        [
            "w16_fba_h12l1p1_e200_s*.json",
            "w40_lpoffgateflux_h12l1p1_e200_s*.json",
            "w49_pathfbapfbalogfree_h12l1p1_e200_s*.json",
            "w51_protfbapfbalog_h12l1p1_e200_s*.json",
        ],
        ["fba", "lpoffgateflux", "pathfbapfbalogfree", "protfbapfbalog"],
    ),
    # Round 36: the layer-free log1p cell over a store built by another solver of the
    # same growth program (dual simplex vertex; interior point without crossover),
    # paired by seed against the pFBA store.
    "w52": (
        [
            "w49_pathfbapfbalogfree_h12l1p1_e200_s*.json",
            "w52_pathlogsimplex_h12l1p1_e200_s*.json",
            "w52_pathlogipm_h12l1p1_e200_s*.json",
        ],
        ["pathfbapfbalogfree", "pathlogsimplex", "pathlogipm"],
    ),
    # Round 38: the same cell over the PDLP store (the cuOpt algorithm's own point),
    # beside the pFBA and interior-point stores.
    "w54": (
        [
            "w49_pathfbapfbalogfree_h12l1p1_e200_s*.json",
            "w52_pathlogipm_h12l1p1_e200_s*.json",
            "w54_pathlogpdlp_h12l1p1_e200_s*.json",
        ],
        ["pathfbapfbalogfree", "pathlogipm", "pathlogpdlp"],
    ),
    # Round 40: the eight wild-type solve statistics beside the log1p vector, against
    # the vector alone (round 33) and the statistics beside the standardized vector
    # (round 31).
    "w56": (
        [
            "w47_pathfbastatspfba_h12l1p1_e200_s*.json",
            "w49_pathfbapfbalogfree_h12l1p1_e200_s*.json",
            "w56_pathlogstats_h12l1p1_e200_s*.json",
        ],
        ["pathfbastatspfba", "pathfbapfbalogfree", "pathlogstats"],
    ),
    # Round 41: a GELU hidden layer in the affine over the log1p vector, at two
    # widths, against the linear affine of round 33.
    "w57": (
        [
            "w49_pathfbapfbalogfree_h12l1p1_e200_s*.json",
            "w57_pathloghead64_h12l1p1_e200_s*.json",
            "w57_pathloghead256_h12l1p1_e200_s*.json",
        ],
        ["pathfbapfbalogfree", "pathloghead64", "pathloghead256"],
    ),
    # Round 42: the round-33 cell over the pFBA store built on synthetic complete
    # medium, against the YPD store.
    "w58": (
        [
            "w49_pathfbapfbalogfree_h12l1p1_e200_s*.json",
            "w58_pathlogsc_h12l1p1_e200_s*.json",
        ],
        ["pathfbapfbalogfree", "pathlogsc"],
    ),
    # Round 43: the solved flux as a node embedding (the representation) with the
    # four-ratio affine, against the vector appended to the affine (round 33) and
    # the affine alone (round 22).
    "w59": (
        [
            "w38_protpathfba_h12l1p1_e200_s*.json",
            "w49_pathfbapfbalogfree_h12l1p1_e200_s*.json",
            "w59_pathsolvenode_h12l1p1_e200_s*.json",
            "w59_pathsolvenodeaff_h12l1p1_e200_s*.json",
        ],
        ["protpathfba", "pathfbapfbalogfree", "pathsolvenode", "pathsolvenodeaff"],
    ),
    # Round 44: the vector reduced by PCA on the node, at 64 and 16 components, alone
    # and with the vector in the affine, against the raw node embedding (round 43)
    # and the vector in the affine (round 33).
    "w60": (
        [
            "w49_pathfbapfbalogfree_h12l1p1_e200_s*.json",
            "w59_pathsolvenode_h12l1p1_e200_s*.json",
            "w60_pathsolvepca64_h12l1p1_e200_s*.json",
            "w60_pathsolvepca16_h12l1p1_e200_s*.json",
            "w60_pathsolvepca64aff_h12l1p1_e200_s*.json",
        ],
        [
            "pathfbapfbalogfree",
            "pathsolvenode",
            "pathsolvepca64",
            "pathsolvepca16",
            "pathsolvepca64aff",
        ],
    ),
    # Round 50: GO on the logit instead of the node, through the solve store path
    # (go_affine_store.py): GO alone as affine columns, GO appended to the pFBA vector,
    # the same on ProtT5 alone, and the round-47 GO-on-node cell at 400 epochs; beside
    # the round-33 cell and the round-47 cells.
    "w66": (
        [
            "w49_pathfbapfbalogfree_h12l1p1_e200_s*.json",
            "w63_pathgofba_h12l1p1_e200_s*.json",
            "w63_pathgolog_h12l1p1_e200_s*.json",
            "w66_pathgolog400_h12l1p1_e400_s*.json",
            "w66_pathgoaff_h12l1p1_e200_s*.json",
            "w66_pathgopfbaaff_h12l1p1_e200_s*.json",
            "w66_protgopfbaaff_h12l1p1_e200_s*.json",
        ],
        [
            "pathfbapfbalogfree",
            "pathgofba",
            "pathgolog",
            "pathgolog400",
            "pathgoaff",
            "pathgopfbaaff",
            "protgopfbaaff",
        ],
    ),
    # Round 49: the second bracket of the pooled cell's divergence (round 46 placed it
    # with the pathway multi-hot in the pooled arm): the plain pooled head on pathway +
    # ProtT5, the affine on the pathway multi-hot alone, and the affine on GO + ProtT5,
    # each pinned twice on seed 2700.
    "w65": (
        ["w65_*_h12l1p1_e200_s2700.json"],
        [
            "pathpooldet1",
            "pathpooldet2",
            "pathfbadet1",
            "pathfbadet2",
            "protgofbadet1",
            "protgofbadet2",
        ],
    ),
    # Round 48: the capacity of the layer-free cell: width 24; two transformer layers;
    # width 24 with the two-layer preprocessor (the round-28 wide recipe); beside the
    # round-33 cell.
    "w64": (
        [
            "w49_pathfbapfbalogfree_h12l1p1_e200_s*.json",
            "w64_pathlogh24_h24l1p1_e200_s*.json",
            "w64_pathlogh48_h48l1p1_e200_s*.json",
            "w64_pathlogl2_h12l2p1_e200_s*.json",
            "w64_pathlogwide_h24l1p2_e200_s*.json",
        ],
        ["pathfbapfbalogfree", "pathlogh24", "pathlogh48", "pathlogl2", "pathlogwide"],
    ),
    # Round 47: the GO multi-hot beside the layer-free solve in one transformer: GO +
    # pathway + ProtT5 with the log1p vector in the affine; GO in place of the pathway
    # block with the vector; GO + pathway + ProtT5 without the vector (the control);
    # beside the round-33 cell and the round-26 GO cells.
    "w63": (
        [
            "w49_pathfbapfbalogfree_h12l1p1_e200_s*.json",
            "w42_protgofba_h12l1p1_e200_s*.json",
            "w42_protgopathfba_h12l1p1_e200_s*.json",
            "w63_pathgofba_h12l1p1_e200_s*.json",
            "w63_protgolog_h12l1p1_e200_s*.json",
            "w63_pathgolog_h12l1p1_e200_s*.json",
        ],
        [
            "pathfbapfbalogfree",
            "protgofba",
            "protgopathfba",
            "pathgofba",
            "protgolog",
            "pathgolog",
        ],
    ),
    # Round 46: bracketing the pooled cell's divergence under the pinned kernels: the
    # affine cell (no solve columns) pinned twice, and the layer-free cell on ProtT5
    # alone (no pathway block) pinned twice.
    "w62": (
        ["w62_*_h12l1p1_e200_s2700.json"],
        ["protpathfbadet1", "protpathfbadet2", "protlogdet1", "protlogdet2"],
    ),
    # Round 45: the layer-free pooled cell on seed 2700 twice with the card's kernels
    # pinned and twice free, as round 32 read the LP form.
    "w61": (
        [
            "w49_pathfbapfbalogfree_h12l1p1_e200_s*.json",
            "w61_pathlog*_h12l1p1_e200_s*.json",
        ],
        [
            "pathfbapfbalogfree",
            "pathlogdet1",
            "pathlogdet2",
            "pathlogfree1",
            "pathlogfree2",
            "pathlogdet",
        ],
    ),
    # Round 28: the round-26 GO cells with a wider cell (width 24, two-layer
    # preprocessor), against the round-26 cells, the pathway cell and the affine.
    "w44": (
        [
            "w16_fba_h12l1p1_e200_s*.json",
            "w38_protpathfba_h12l1p1_e200_s*.json",
            "w42_protgofba_h12l1p1_e200_s*.json",
            "w42_protgopathfba_h12l1p1_e200_s*.json",
            "w44_*_e200_s*.json",
        ],
        [
            "fba",
            "protpathfba",
            "protgofba",
            "protgopathfba",
            "protgofbawide",
            "protgopathfbawide",
        ],
    ),
    # Round 16: the round-9 backbone with the round-11 affine.
    "w27": (
        [
            "w16_fba_h12l1p1_e200_s*.json",
            "w20_chromprotfba_h12l1p1_e200_s270[0-2].json",
            "w22_fbastatspfba_h12l1p1_e200_s*.json",
            "w27_*_e200_s*.json",
        ],
        ["fba", "chromprotfba", "fbastatspfba", "chromprotpfba", "chromprotstatspfba"],
    ),
    # Round 17: the round-16 cell on the five folds, beside the round-12 folds.
    "w28": (
        [
            "w20_chromprotfba_h12l1p1_e200_s270[0-2].json",
            "w23_chromprotcv*_h12l1p1_e200_s*.json",
            "w27_chromprotstatspfba_h12l1p1_e200_s*.json",
            "w28_*_e200_s*.json",
        ],
        [
            "chromprotfba",
            "chromprotstatspfba",
            "chromprotcv0",
            "chromprotcv1",
            "chromprotcv2",
            "chromprotcv3",
            "chromprotcv4",
            "cspcv0",
            "cspcv1",
            "cspcv2",
            "cspcv3",
            "cspcv4",
        ],
    ),
    # The 027 wave-11 width-12 and width-24 pooled cells, for a dry run of this script and
    # as the pre-move reference (same protocol as `base`).
    "wave11": (["wave11_ess_pool_h*_e80_s*.json"], ["ess_pool"]),
}
ARM_LABELS = {
    "base": "wave-11 protocol (75-gene val)",
    "fcl": "released labels, 158-gene val",
    "fcl_wd0": "weight decay 0",
    "fcl_wd1e-2": "weight decay 1e-2",
    "fcl_wd1e-1": "weight decay 1e-1",
    "fcl_b32": "batch 32",
    "gen": "genome labels (5,795 genes)",
    "fcl_lr3e-4": "lr 3e-4",
    "fcl_lr3e-3": "lr 3e-3",
    "fcl_posw": "class weight 772/126",
    "fcl_noreg": "graph regularization off",
    "b32posw": "batch 32 + class weight, 80 epochs",
    "b32posw200": "batch 32 + class weight, 200 epochs",
    "b32e200": "batch 32, 200 epochs",
    "cv0": "5-fold, fold 0",
    "cv1": "5-fold, fold 1",
    "cv2": "5-fold, fold 2",
    "cv3": "5-fold, fold 3",
    "cv4": "5-fold, fold 4",
    "fluxbio": "growth column only, constraints 0.1",
    "fluxbio_s0": "growth column only, constraints 0",
    "fluxctx": "context + growth column, constraints 0.1",
    "fluxctx_s0": "context + growth column, constraints 0",
    "fluxoff": "growth column only, stoichiometry only",
    "fluxoffctx": "context + growth column, stoichiometry only",
    "fluxgate": "gated growth column, constraints 0.1",
    "fluxgate_s0": "gated growth column, constraints 0",
    "fluxshuf_s0": "growth column only, kcat shuffled, constraints 0",
    "gate_s0": "gated growth column, constraints 0 (round-3 levers)",
    "bio_s0": "growth column only, constraints 0 (round-3 levers)",
    "shuf_s0": "growth column only, kcat shuffled (round-3 levers)",
    "fba": "context head + FBA growth ratios (round-3 levers)",
    "fbacv0": "context head + FBA ratios, fold 0",
    "fbacv1": "context head + FBA ratios, fold 1",
    "fbacv2": "context head + FBA ratios, fold 2",
    "fbacv3": "context head + FBA ratios, fold 3",
    "fbacv4": "context head + FBA ratios, fold 4",
    "fbaypd": "context head + FBA ratio, YPD only",
    "biofba": "growth column + FBA ratios, no context head",
    "gatefba": "gated growth column + FBA ratios",
    "codonfba": "codon-frequency backbone + FBA ratios",
    "compfba": "species-LM composite backbone + FBA ratios",
    "comp": "species-LM composite backbone",
    "bioprojbox": "growth column only, projection ending on the box",
    "biobox": "growth column only, box parameterization",
    "fbah24": "context head + FBA ratios, width 24",
    "fbah48": "context head + FBA ratios, width 48",
    "fbal2": "context head + FBA ratios, two blocks",
    "fbap2": "context head + FBA ratios, two-layer preprocessor",
    "chromfba": "chrom-pathway backbone + FBA ratios",
    "chromprotfba": "chrom-pathway + ProtT5 backbone + FBA ratios",
    "esmfba": "ESM2 backbone + FBA ratios",
    "mixfba": "seven-encoding backbone + FBA ratios",
    "warmbio": "growth column, pFBA warm start, alpha 0.01 frozen",
    "warmgate": "gated column + context, pFBA warm start, alpha 0.01 frozen",
    "warmgatealpha": "gated column + context, pFBA warm start, alpha learned",
    "warmgate10": "gated column + context, pFBA warm start, alpha 0.1 frozen",
    "fbastats": "context head + FBA ratios + eight reaction statistics",
    "fbapfba": "context head + FBA ratios + deletion pFBA vector",
    "fbastatspfba": "context head + FBA ratios + statistics + pFBA vector",
    "warmoffbio": "growth column, GEM box only, pFBA warm start, alpha 0.01 frozen",
    "warmoffgate": "gated column + context, GEM box only, pFBA warm start, alpha 0.01 frozen",
    "chromprotcv0": "chrom-pathway + ProtT5 + FBA ratios, fold 0",
    "chromprotcv1": "chrom-pathway + ProtT5 + FBA ratios, fold 1",
    "chromprotcv2": "chrom-pathway + ProtT5 + FBA ratios, fold 2",
    "chromprotcv3": "chrom-pathway + ProtT5 + FBA ratios, fold 3",
    "chromprotcv4": "chrom-pathway + ProtT5 + FBA ratios, fold 4",
    "nchromfba": "gene properties + chromosome backbone + FBA ratios",
    "nchromprotfba": "gene properties + chromosome + ProtT5 backbone + FBA ratios",
    "lpbio": "growth column, the layer that solves",
    "lpbiotdg": "growth column, the layer that solves, second-law box",
    "lpgate": "gated column + context, the layer that solves",
    "lpoffbio": "growth column, the layer that solves, GEM box",
    "lpoffgate": "gated column + context, the layer that solves, GEM box",
    "lpcapbio100": "growth column, the layer that solves, capacity x100",
    "lpcapgate100": "gated column + context, the layer that solves, capacity x100",
    "lpcapgate1000": "gated column + context, the layer that solves, capacity x1000",
    "lpcapbio1000": "growth column, the layer that solves, capacity x1000",
    "lpcapbio100lr": "growth column, the layer that solves, capacity x100, lr 3e-4",
    "lpelbio100": "growth column, the elastic layer that solves, capacity x100",
    "lpelbio1000": "growth column, the elastic layer that solves, capacity x1000",
    "lpelpoolbio100": "growth column, elastic solve, capacity x100, pool by construction",
    "lpelgate100": "gated column + context, the elastic layer that solves, capacity x100",
    "lpelbio100lr10": "growth column, elastic solve, x100, flux layer lr x10",
    "lpelbio100mlp": "growth column, elastic solve, x100, two-layer abundance head",
    "lpelbio100mlplr10": "growth column, elastic solve, x100, two-layer head + lr x10",
    "lpelgate100mlplr10": "gated column + context, elastic solve, x100, two-layer head + lr x10",
    "protpathfba": "pathway multi-hot + ProtT5 backbone + FBA ratios",
    "protchromfba": "chromosome one-hot + ProtT5 backbone + FBA ratios",
    "protpropsfba": "seven gene properties + ProtT5 backbone + FBA ratios",
    "pathfba": "pathway multi-hot backbone + FBA ratios",
    "pathpool": "pathway multi-hot + ProtT5 backbone, no readout",
    "pathlpoffgatefluxfirst": "the same cell, first attempt (three survivors of the six-cell pack; per-gene dumps overwritten)",
    "pathlpoffgateflux": "pathway multi-hot + ProtT5; gated + a head over the GEM-box solve's flux (cached, nothing learned in the layer)",
    "pathfbapfba": "pathway multi-hot + ProtT5; FBA affine + the deletion's pFBA vector as an input (no flux layer)",
    "pathfbastatspfba": "the same, with the eight wild-type solve statistics beside the pFBA vector",
    "pathfbapfbazero": "the same vector, the affine's solve columns zero-initialized",
    "pathfbapfbalogfree": "the vector as log(1 + |v|), default initialization",
    "pathfbapfbalog": "the vector as log(1 + |v|), solve columns zero-initialized (the flux head's construction, no layer)",
    "protfbapfbalog": "ProtT5 alone; FBA affine + the deletion's pFBA vector as log(1 + |v|) (no pathway block, no layer)",
    "pathlogsimplex": "the round-33 cell over the growth program's dual-simplex vertex (store by HiGHS simplex)",
    "pathlogipm": "the round-33 cell over the growth program's interior point without crossover (store by HiGHS ipm, KKT 1e-4)",
    "pathlogpdlp": "the round-33 cell over the growth program's PDLP point (store by HiGHS PDLP, KKT 1e-4; the cuOpt algorithm)",
    "pathlogstats": "the round-33 cell with the eight wild-type solve statistics beside the log1p vector",
    "pathloghead64": "the round-33 cell with a GELU hidden layer of width 64 in the affine",
    "pathloghead256": "the round-33 cell with a GELU hidden layer of width 256 in the affine",
    "pathlogsc": "the round-33 cell over the pFBA store built on synthetic complete medium (SC) instead of YPD",
    "pathsolvenode": "pathway multi-hot + ProtT5 + the deletion's log1p flux vector as a NODE embedding; the four-ratio affine only",
    "pathsolvenodeaff": "the same, with the log1p vector appended to the affine as well",
    "pathsolvepca64": "the vector reduced by PCA to 64 components as the node embedding; the four-ratio affine only",
    "pathsolvepca16": "the vector reduced by PCA to 16 components as the node embedding; the four-ratio affine only",
    "pathsolvepca64aff": "the 64-component node embedding, with the full log1p vector in the affine as well",
    "pathlogdet1": "round-33 cell (log1p vector in the affine, no layer), seed 2700, the card's kernels pinned, first run",
    "pathlogdet2": "the same, pinned, second run",
    "pathlogfree1": "the same, kernels free, first run",
    "pathlogfree2": "the same, free, second run",
    "pathlogdet": "the round-33 cell, the card's kernels pinned, one run per seed (seeds 2701 to 2705)",
    "protpathfbadet1": "round-22 affine cell (pathway + ProtT5, the four ratios, no solve columns), seed 2700, kernels pinned, first run",
    "protpathfbadet2": "the same, pinned, second run",
    "protlogdet1": "layer-free cell on ProtT5 alone (no pathway block), seed 2700, kernels pinned, first run",
    "protlogdet2": "the same, pinned, second run",
    "pathpooldet1": "round-9 pooled cell (pathway + ProtT5, no affine readout), seed 2700, kernels pinned, first run",
    "pathpooldet2": "the same, pinned, second run",
    "pathfbadet1": "affine cell on the pathway multi-hot alone (no ProtT5), seed 2700, kernels pinned, first run",
    "pathfbadet2": "the same, pinned, second run",
    "protgofbadet1": "affine cell on GO + ProtT5 (no pathway block), seed 2700, kernels pinned, first run",
    "protgofbadet2": "the same, pinned, second run",
    "pathgolog": "GO multi-hot + pathway multi-hot + ProtT5, the log(1 + |v|) vector in the affine, no layer",
    "protgolog": "GO multi-hot + ProtT5 (no pathway block), the log(1 + |v|) vector in the affine",
    "pathgofba": "GO multi-hot + pathway multi-hot + ProtT5, the four ratios only (round 47 control)",
    "pathgolog400": "the round-47 GO + pathway + ProtT5 layer-free cell at 400 epochs",
    "pathgoaff": "pathway + ProtT5 on the node; the GO multi-hot as affine columns (no solve)",
    "pathgopfbaaff": "pathway + ProtT5 on the node; the log(1 + |v|) vector and the GO multi-hot as affine columns",
    "protgopfbaaff": "ProtT5 alone on the node; the log(1 + |v|) vector and the GO multi-hot as affine columns",
    "pathlogh24": "the round-33 cell at width 24",
    "pathlogh48": "the round-33 cell at width 48",
    "pathlogl2": "the round-33 cell with two transformer blocks (graph regularization off)",
    "pathlogwide": "the round-33 cell at width 24 with the two-layer preprocessor",
    "pathlpoffgatefluxdet1": "round-27 cell, seed 2701, the card's kernels pinned, first run",
    "pathlpoffgatefluxdet2": "the same, pinned, second run",
    "pathlpoffgatefluxfree1": "the same, kernels free, first run",
    "pathlpoffgatefluxfree2": "the same, free, second run",
    "pathlpoffgatefluxdet": "round-27 cell, the card's kernels pinned, one run per seed (seeds 2700, 2702 to 2705)",
    "protgofba": "GO multi-hot (propagated, 2,362) + ProtT5 backbone + FBA ratios",
    "protgobpfba": "GO biological process (1,580) + ProtT5 backbone + FBA ratios",
    "gofba": "GO multi-hot backbone + FBA ratios",
    "protgopathfba": "GO + pathway multi-hots + ProtT5 backbone + FBA ratios",
    "protgofbawide": "GO + ProtT5 + FBA ratios, width 24, two-layer preprocessor",
    "protgopathfbawide": "GO + pathways + ProtT5 + FBA ratios, width 24, two-layer preprocessor",
    "pathlpoffgate": "pathway multi-hot + ProtT5; gated column + context, the layer that solves, GEM box",
    "pathlpelgate100": "pathway multi-hot + ProtT5; gated column + context, elastic solve, x100",
    "pathlpelgateflux": "pathway multi-hot + ProtT5; gated + a head over the solved flux (all reactions)",
    "pathlpelgateflux400": "the same cell, 400 epochs",
    "lpelbio30lr10": "growth column, elastic solve, x30, flux layer lr x10",
    "lpelbio100r6lr10": "growth column, elastic solve, x100, reach $e^{6}$, flux layer lr x10",
    "lpelgateflux": "gated column + context + a head over the solved flux (all reactions)",
    "lpelgatefluxexch": "gated column + context + a head over the solved exchange fluxes",
    "lpelgatefluxgene": "gated column + context + a head over the solved gene-owned fluxes",
    "lpoffgateflux": "gated column + context + a head over the GEM-box solve's flux (all reactions)",
    "chromprotpfba": "chrom-pathway + ProtT5 backbone + FBA ratios + pFBA vector",
    "chromprotstatspfba": "chrom-pathway + ProtT5 backbone + ratios + statistics + pFBA",
    "cspcv0": "chrom-pathway + ProtT5 + widened affine, fold 0",
    "cspcv1": "chrom-pathway + ProtT5 + widened affine, fold 1",
    "cspcv2": "chrom-pathway + ProtT5 + widened affine, fold 2",
    "cspcv3": "chrom-pathway + ProtT5 + widened affine, fold 3",
    "cspcv4": "chrom-pathway + ProtT5 + widened affine, fold 4",
}


class CellRow(BaseModel):
    """One finished cell."""

    stem: str
    arm: str
    seed: int
    hidden: int
    weight_decay: float
    batch_size: int
    split_file: str | None
    n_parameters: int
    n_val: int
    select_epoch: int
    select_val_auroc: float
    test_auroc_pinned: float
    test_auroc_ci_low: float | None
    test_auroc_ci_high: float | None
    test_accuracy_pinned: float
    test_recall_essential_pinned: float
    median_abs_step: float
    val_curve_range: float
    wall_time_min: float
    wandb_url: str | None


class ArmSummary(BaseModel):
    """Seed mean, sd and n per arm, plus the pooled-prediction bootstrap."""

    arm: str
    label: str
    n_seeds: int
    seeds: list[int]
    test_auroc_mean: float
    test_auroc_sd: float
    test_auroc_values: list[float]
    pooled_auroc: float | None
    pooled_ci_low: float | None
    pooled_ci_high: float | None
    median_abs_step_mean: float
    n_val: int
    select_epochs: list[int]


def binary_auroc(score: np.ndarray, label: np.ndarray) -> float:
    pos = label == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_ci(score: np.ndarray, label: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(score)
    draws = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        v = binary_auroc(score[idx], label[idx])
        if not np.isnan(v):
            draws.append(v)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def arm_of(stem: str) -> str:
    body = stem.split("_", 1)[1]
    arm = body.split("_h")[0]
    # A cell run past the 200-epoch budget is its own arm, so a 400-epoch rerun of one
    # seed reads beside the 200-epoch cell instead of pooling into it.
    if "_e400_" in stem:
        arm += "400"
    return arm


def pinned_predictions(
    result: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    path = result.get("test", {}).get("dump_path")
    if not path:
        return None
    if not osp.isfile(path):
        alt = osp.join(PRED_DIR, osp.basename(path))
        if not osp.isfile(alt):
            return None
        path = alt
    with open(path) as fh:
        dump = json.load(fh)
    # THE DUMP MUST BE THIS CELL'S. Until 2026-09-16 every width wrote the same dump name,
    # so a result can point at a file another cell overwrote (the wave-11 width-24 dumps
    # hold the width-12 cell). A dump whose cell disagrees is treated as missing.
    identity = (
        "arm",
        "head",
        "seed",
        "hidden",
        "layers",
        "preprocessor_layers",
        "epochs",
    )
    if any(dump["cell"].get(k) != result["cell"].get(k) for k in identity):
        return None
    recs = [r for r in dump["predictions"] if r["pinned"]]
    genes = [r["genes"][0] for r in recs]
    return (
        np.array([r["pred"] for r in recs], dtype=float),
        np.array([r["target"] for r in recs], dtype=int),
        genes,
    )


def load_round(
    round_key: str,
) -> tuple[list[CellRow], dict[str, list[dict[str, Any]]], dict[str, list[np.ndarray]]]:
    patterns, _ = ROUNDS[round_key]
    rows: list[CellRow] = []
    histories: dict[str, list[dict[str, Any]]] = {}
    preds: dict[str, list[np.ndarray]] = {}
    paths = [p for pat in patterns for p in sorted(glob.glob(osp.join(RESULTS, pat)))]
    for path in paths:
        with open(path) as fh:
            result = json.load(fh)
        stem = osp.splitext(osp.basename(path))[0]
        cell = result["cell"]
        hist = result["history"]
        curve = np.array([h["val_pearson"] for h in hist], dtype=float)
        steps = np.abs(np.diff(curve))
        arm = arm_of(stem)
        pinned = result["test"]["pinned"]
        ci: tuple[float | None, float | None] = (None, None)
        pp = pinned_predictions(result)
        if pp is not None:
            ci = bootstrap_ci(pp[0], pp[1])
            preds.setdefault(arm, []).append(np.stack([pp[0], pp[1]]))
        rows.append(
            CellRow(
                stem=stem,
                arm=arm,
                seed=int(cell["seed"]),
                hidden=int(cell["hidden"]),
                weight_decay=float(cell["weight_decay"]),
                batch_size=int(cell["batch_size"]),
                split_file=cell.get("split_file"),
                n_parameters=int(result["n_parameters"]["total"]),
                n_val=int(hist[0]["n_val"]),
                select_epoch=int(result["select_epoch"]),
                select_val_auroc=float(result["select_val_pearson"]),
                test_auroc_pinned=float(pinned["auroc"]),
                test_auroc_ci_low=ci[0],
                test_auroc_ci_high=ci[1],
                test_accuracy_pinned=float(pinned["accuracy"]),
                test_recall_essential_pinned=float(pinned["recall_essential"]),
                median_abs_step=float(np.median(steps)),
                val_curve_range=float(curve.max() - curve.min()),
                wall_time_min=float(result["wall_time_s"]) / 60.0,
                wandb_url=result.get("wandb_url"),
            )
        )
        histories.setdefault(arm, []).append(
            {"seed": int(cell["seed"]), "history": hist}
        )
    return rows, histories, preds


def summarize(
    rows: list[CellRow], preds: dict[str, list[np.ndarray]], order: list[str]
) -> list[ArmSummary]:
    out: list[ArmSummary] = []
    arms = [a for a in order if any(r.arm == a for r in rows)]
    arms += sorted({r.arm for r in rows} - set(arms))
    for arm in arms:
        cells = sorted([r for r in rows if r.arm == arm], key=lambda r: r.seed)
        values = [r.test_auroc_pinned for r in cells]
        pooled: tuple[float | None, float | None, float | None] = (None, None, None)
        if arm in preds:
            # Per-gene predictions averaged over seeds (a logit mean), then one AUROC with
            # its gene bootstrap: the interval on the arm's prediction rather than on one
            # seed's.
            stack = np.stack(preds[arm])  # [n_seeds, 2, n_genes]
            score = stack[:, 0, :].mean(axis=0)
            label = stack[0, 1, :].astype(int)
            lo, hi = bootstrap_ci(score, label)
            pooled = (binary_auroc(score, label), lo, hi)
        out.append(
            ArmSummary(
                arm=arm,
                label=ARM_LABELS.get(arm, arm),
                n_seeds=len(cells),
                seeds=[r.seed for r in cells],
                test_auroc_mean=float(np.mean(values)),
                test_auroc_sd=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                test_auroc_values=values,
                pooled_auroc=pooled[0],
                pooled_ci_low=pooled[1],
                pooled_ci_high=pooled[2],
                median_abs_step_mean=float(np.mean([r.median_abs_step for r in cells])),
                n_val=cells[0].n_val,
                select_epochs=[r.select_epoch for r in cells],
            )
        )
    return out


def comparators() -> dict[str, tuple[float, float, float]]:
    """Name -> (AUROC, CI low, CI high) on the 195 genes."""
    out: dict[str, tuple[float, float, float]] = {}
    if osp.isfile(FBA_JSON):
        with open(FBA_JSON) as fh:
            fba = json.load(fh)
        for row in fba["rows"]:
            s = next(x for x in row["scores"] if x["gene_set"] == "ours_195")
            out[f"FBA {row['medium']}"] = (
                s["auroc"],
                s["auroc_ci_low"],
                s["auroc_ci_high"],
            )
    if osp.isfile(FCL_JSON):
        with open(FCL_JSON) as fh:
            fcl = json.load(fh)
        # `scored` rows: method x gene_set; ours_scored is the 195 genes every cell scores.
        for row in fcl["scored"]:
            if row["gene_set"] != "ours_scored" or row.get("seed") is not None:
                continue
            name = {"FCL": "FCL", "FBA": "FBA (released)"}.get(row["method"])
            if name is None:
                continue
            out[name] = (row["auroc"], row["ci_low"], row["ci_high"])
    return out


def style() -> None:
    apply_paper_style()
    plt.rcParams.update({"legend.fontsize": 6, "lines.linewidth": 0.8})


def figure(
    round_key: str,
    summaries: list[ArmSummary],
    histories: dict[str, list[dict[str, Any]]],
    rows: list[CellRow],
    comps: dict[str, tuple[float, float, float]],
) -> str:
    style()
    arms = [s.arm for s in summaries]
    # Six hues per tier (orange, red, purple, blue, gray, yellow), and a second tier of
    # arms takes the same six hues DASHED with hollow markers: the palette's darker tiers
    # of the same hues (brown beside orange, maroon beside red) were not tellable apart
    # in a ten-arm round. Yellow is last so a round of five arms never reaches it.
    hues = [PLOT_PALETTE[i] for i in (0, 1, 2, 4, 5, 3)]
    color = {arm: hues[i % len(hues)] for i, arm in enumerate(arms)}
    tier = {arm: i // len(hues) for i, arm in enumerate(arms)}
    dash = {arm: "-" if tier[arm] == 0 else "--" for arm in arms}
    width_in = mm_to_in(PANEL_WIDTHS_MM["full"])
    fig, axes = plt.subplots(2, 2, figsize=(width_in, mm_to_in(120)))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # a: validation curves
    for arm in arms:
        curves = []
        for h in histories.get(arm, []):
            c = np.array([e["val_pearson"] for e in h["history"]], dtype=float)
            curves.append(c)
            ax_a.plot(np.arange(len(c)), c, color=color[arm], alpha=0.25, linewidth=0.5)
        if curves:
            n = min(len(c) for c in curves)
            ax_a.plot(
                np.arange(n),
                np.mean([c[:n] for c in curves], axis=0),
                color=color[arm],
                linestyle=dash[arm],
                label=ARM_LABELS.get(arm, arm),
            )
    ax_a.set_xlabel("epoch")
    ax_a.set_ylabel("validation AUROC")
    # The axis runs to 1.0 whatever the round reached, so the panels of every round read
    # on one scale; the floor drops to the lowest curve in tenths.
    floor = min(
        0.3,
        float(
            np.floor(
                10
                * np.nanmin(
                    [
                        np.nanmin([e["val_pearson"] for e in h["history"]])
                        for hs in histories.values()
                        for h in hs
                    ]
                )
            )
            / 10
        ),
    )
    ax_a.set_ylim(floor, 1.0)
    ax_a.yaxis.set_major_locator(MultipleLocator(0.2))
    ax_a.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax_a.grid(axis="y", which="both", linewidth=0.3, color="0.85")
    # The arm legend is a FIGURE legend across the top, not an axes legend on a: with two
    # columns of long arm labels an axes legend ran over panel b.
    legend_handles, legend_labels = ax_a.get_legend_handles_labels()
    # Three columns of 6 pt labels fit the 179 mm width up to about 46 characters per
    # label; longer labels (the warm-start arms) wrap to two columns rather than run off
    # the right edge, which is what the three-column round-10 figure did.
    legend_ncol = 3 if max(len(label) for label in legend_labels) <= 46 else 2
    legend_rows = int(np.ceil(len(legend_labels) / legend_ncol))

    # b: stability vs validation size
    for s in summaries:
        cells = [r for r in rows if r.arm == s.arm]
        ax_b.scatter(
            [r.n_val for r in cells],
            [r.median_abs_step for r in cells],
            facecolors=color[s.arm] if tier[s.arm] == 0 else "white",
            edgecolors=color[s.arm] if tier[s.arm] else "black",
            s=8,
            linewidths=0.5,
        )
    ax_b.set_xlabel("labeled validation genes")
    ax_b.set_ylabel("median |step|, validation AUROC")
    n_vals = [r.n_val for r in rows]
    # A log axis only when the round spans validation sizes (75 against 1,093 in round 1);
    # a round on one split file gets a linear axis with the one or two sizes it has.
    if max(n_vals) > 2 * min(n_vals):
        ax_b.set_xscale("log")
    else:
        ax_b.set_xticks(sorted(set(n_vals)))

    # c: test AUROC per arm with comparators
    x = 0
    ticks, labels = [], []
    for s in summaries:
        cells = [r for r in rows if r.arm == s.arm]
        ax_c.scatter(
            [x] * len(cells),
            [r.test_auroc_pinned for r in cells],
            facecolors=color[s.arm] if tier[s.arm] == 0 else "white",
            edgecolors=color[s.arm] if tier[s.arm] else "black",
            s=10,
            linewidths=0.5,
            zorder=3,
        )
        if (
            s.pooled_auroc is not None
            and s.pooled_ci_low is not None
            and s.pooled_ci_high is not None
        ):
            ax_c.errorbar(
                [x + 0.25],
                [s.pooled_auroc],
                yerr=[
                    [s.pooled_auroc - s.pooled_ci_low],
                    [s.pooled_ci_high - s.pooled_auroc],
                ],
                fmt="s",
                color=color[s.arm],
                markersize=3,
                capsize=1.5,
                elinewidth=0.6,
                markeredgecolor="black",
                markeredgewidth=0.3,
                zorder=3,
            )
        ticks.append(x)
        labels.append(s.arm)
        x += 1
    for name, (v, lo, hi) in comps.items():
        ax_c.errorbar(
            [x],
            [v],
            yerr=[[v - lo], [hi - v]],
            fmt="D",
            color=PLOT_PALETTE[5],
            markersize=3,
            capsize=1.5,
            elinewidth=0.6,
            markeredgecolor="black",
            markeredgewidth=0.3,
            zorder=3,
        )
        ticks.append(x)
        labels.append(name)
        x += 1
    ax_c.set_xticks(ticks)
    ax_c.set_xticklabels(labels, rotation=45, ha="right")
    ax_c.set_ylabel("held-out AUROC (195 genes)")
    ax_c.yaxis.set_major_locator(MultipleLocator(0.2))
    ax_c.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax_c.grid(axis="y", which="both", linewidth=0.3, color="0.85")
    ax_c.set_ylim(0.4, 1.0)

    # d: gradient norms per module, seed mean, for arms that logged them
    keys = [
        "grad_norm_embedding_preprocessor_mean",
        "grad_norm_transformer_layers_mean",
        "grad_norm_essentiality_head_mean",
        "grad_norm_flux_layer_mean",
    ]
    styles = ["-", "--", ":", "-."]
    for arm in arms:
        hs = histories.get(arm, [])
        for key, ls in zip(keys, styles, strict=True):
            curves = [
                np.array([e.get(key, np.nan) for e in h["history"]], dtype=float)
                for h in hs
            ]
            if not curves or all(np.all(np.isnan(c)) for c in curves):
                continue
            n = min(len(c) for c in curves)
            ax_d.plot(
                np.arange(n),
                np.nanmean([c[:n] for c in curves], axis=0),
                color=color[arm],
                linestyle=ls,
                linewidth=0.7,
            )
    ax_d.set_xlabel("epoch")
    ax_d.set_ylabel("gradient norm, seed mean")
    ax_d.set_yscale("log")
    from matplotlib.lines import Line2D

    ax_d.legend(
        [Line2D([0], [0], color="black", linestyle=ls) for ls in styles],
        ["preprocessor", "transformer", "essentiality head", "flux layer"],
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        ncol=2,
        frameon=False,
    )

    for ax, letter in zip(axes.flat, "abcd", strict=True):
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
        ax.text(
            -0.17,
            1.10,
            letter,
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            va="top",
            ha="left",
        )
    # Fixed margins rather than tight_layout: the figure legend needs `legend_rows` lines
    # above the top row (about 3.2 mm each at 6 pt plus the row's own letter), and the
    # two columns sit at one gap whatever the length of b's axis label.
    height_mm = 120.0
    top = 1.0 - (legend_rows * 3.2 + 6.0) / height_mm
    fig.subplots_adjust(
        left=0.07, right=0.99, bottom=0.12, top=top, wspace=0.22, hspace=0.55
    )
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(0.07, 1.0),
        ncol=legend_ncol,
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.6,
        columnspacing=1.2,
    )
    os.makedirs(IMAGE_DIR, exist_ok=True)
    base = osp.join(IMAGE_DIR, f"{round_key}_readout_{timestamp()}")
    fig.savefig(f"{base}.png", dpi=300)
    savefig_true_size_svg(fig, f"{base}.svg")
    plt.close(fig)
    return f"{base}.svg"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default="w12", choices=list(ROUNDS))
    args = parser.parse_args()
    rows, histories, preds = load_round(args.round)
    assert rows, f"no finished {args.round} result files under {RESULTS}"
    summaries = summarize(rows, preds, ROUNDS[args.round][1])
    comps = comparators()
    out = {
        "round": args.round,
        "n_cells": len(rows),
        "cells": [r.model_dump() for r in rows],
        "arms": [s.model_dump() for s in summaries],
        "comparators_on_195": {
            k: {"auroc": v[0], "ci_low": v[1], "ci_high": v[2]}
            for k, v in comps.items()
        },
        "bootstrap": {"n": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED},
    }
    path = osp.join(RESULTS, f"{args.round}_readout.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    table = pd.DataFrame([r.model_dump() for r in rows])
    table.to_csv(osp.join(RESULTS, f"{args.round}_readout.csv"), index=False)
    print(
        table[
            [
                "stem",
                "n_val",
                "select_epoch",
                "test_auroc_pinned",
                "test_auroc_ci_low",
                "test_auroc_ci_high",
                "median_abs_step",
                "wall_time_min",
            ]
        ].to_string(index=False)
    )
    for s in summaries:
        print(
            f"{s.arm:14s} n={s.n_seeds} mean {s.test_auroc_mean:.4f} sd {s.test_auroc_sd:.4f} "
            f"pooled {s.pooled_auroc if s.pooled_auroc is None else round(s.pooled_auroc, 4)} "
            f"[{s.pooled_ci_low}, {s.pooled_ci_high}] step {s.median_abs_step_mean:.4f} n_val {s.n_val} sel {s.select_epochs}"
        )
    for k, v in comps.items():
        print(f"{k:14s} {v[0]:.4f} [{v[1]:.3f}, {v[2]:.3f}]")
    print(figure(args.round, summaries, histories, rows, comps))
    print(path)


if __name__ == "__main__":
    main()
