# experiments/028-gene-essentiality/scripts/ess_kcat_bounded_store.py
# [[experiments.028-gene-essentiality.scripts.ess_kcat_bounded_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/ess_kcat_bounded_store

r"""Enzyme-capacity-bounded parsimonious deletion flux stores, solved offline (round 113).

The plain store ``ess_solve_encodings_YPD_nuc.npz`` is one two-stage pFBA per labeled gene
on the stoichiometric box with the corrected YPD medium. This script writes the SAME store
shape on a box whose forward and reverse bounds are additionally capped by an enzyme
capacity ceiling per reaction,

.. math::
    \bar v^u_j = \min\Big(v^u_j,\ s \sum_{u:\rho(u)=j} k_{cat,u}\,E_u\Big),

with :math:`k_{cat,u}` the resolved catalytic-unit turnover (``resolve_kcat_table``, the
predicted tier the 028 capacity calibration ran on), :math:`E_u` a MEASURED wild-type
abundance in mmol/gDW, and :math:`s` a capacity scale chosen per variant as the smallest
value on the calibration's geometric grid at which the wild type reaches at least
``WILD_TYPE_GROWTH_FRACTION`` of its unconstrained growth. The scale exists because the
un-scaled ceilings were measured infeasible for every genotype
(``results/capacity_calibration_predicted.json``: feasible fraction 0 at scale 1 to 10).

THE ABUNDANCE. No absolute proteome loader exists in ``torchcell.metabolism``; the flux
layer's ``dynamic_box`` uses a nominal 1e-3 mmol/gDW per unit. The measured proteome used
here is the Messner 2023 wild-type reference (the 388 ``HIS3`` control columns of
``proteome_messner2023/raw/yeast5k_noimpute_wide.csv``, MaxLFQ linear quantities, mean over
the non-missing replicates per protein, UniProt mapped to the systematic ORF through the
SGD GFF exactly as the dataset loader does). MaxLFQ is a relative quantity, so it is turned
into mmol/gDW by the total protein approach: the mass fraction of each detected protein
among the detected proteins, times the total cellular protein ``P_tot = 0.5`` g/gDW
(``GECKO3_YEAST_PROTEIN_POOL.total_protein_g_per_gdw``, sourced), divided by the SwissProt
molecular weight. The denominator covers only the detected proteins, so each fraction is an
overestimate by the undetected mass; this is recorded, not hidden. A unit's abundance is the
MINIMUM over its member genes (the scarcest subunit, the same convention as the layer's
softmin) and exists only when EVERY member gene is measured; a reaction gets a ceiling only
when EVERY one of its units has an abundance, otherwise it keeps the GEM bound. Nothing is
defaulted.

VARIANTS (``--variant``):

* ``point``: predicted-tier point values (``kcat_ceiling_sigma_k=0``).
* ``sigma1`` / ``sigma2``: each unit widened to :math:`10^{\mu_u + k\sigma_u}` with the
  ``KCAT_SIGMA_LADDER`` (k = 1, 2).
* ``sampled``: ``K`` draws of :math:`\log_{10}k_{cat,u}\sim N(\mu_u,\sigma_u)`, independent
  across units, seed 0; one pFBA per draw per gene, the store carries the MEAN vector and a
  second block of per-reaction SD over draws (``rxn_ids`` = ids + ``sd_<id>``), the shape
  ``ess_sampled_flux_store.py`` documents.
* ``moma``: NO ceilings, the plain GEM box on the corrected YPD. Objective: linear
  minimization of metabolic adjustment, :math:`\min \sum_j |v_j - v^{wt}_j|` subject to
  :math:`Sv = 0` and the knocked-out box, one LP through scipy's HiGHS with the absolute
  value split into positive and negative parts, against the pFBA wild type.

Writes ``$DATA_ROOT/data/torchcell/reference_flux/ess_solve_encodings_YPD_kcat<variant>.npz``
(``genes``, ``v_deletion`` float32, ``v_wild_type``, ``rxn_ids``, ``provenance`` = a
``StoreProvenance`` record the derived-store scripts validate, ``kcat_provenance`` = the
``KcatStoreProvenance`` subclass record with the ceiling fields, ``growth_ratio``) and
``results/ess_kcat_bounded_store_<variant>.json``. Run from the repo root::

    PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
        ~/miniconda3/envs/torchcell/bin/python \
        experiments/028-gene-essentiality/scripts/ess_kcat_bounded_store.py \
        --variant point --processes 32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import sys
import time
from multiprocessing import Pool
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from cobra.manipulation import knock_out_model_genes
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
TABLE_DIR = osp.join(DATA_ROOT, "data/torchcell/reference_flux")
PLAIN_STORE = osp.join(TABLE_DIR, "ess_solve_encodings_YPD_nuc.npz")
MESSNER_RAW = osp.join(DATA_ROOT, "data/torchcell/proteome_messner2023/raw")
MESSNER_MATRIX = "yeast5k_noimpute_wide.csv"
MESSNER_METADATA = "yeast5k_metadata.csv"
OED_SCEREVISIAE = osp.join(
    DATA_ROOT, "data/enzyme_kinetics/open_enzyme_database/scerevisiae"
)
OED_ALL = osp.join(DATA_ROOT, "data/enzyme_kinetics/open_enzyme_database/all")

sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "scripts"))
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from ess_reference_flux_table import (  # noqa: E402
    FLUX_SCALE,
    MediaApplied,
    StoreProvenance,
    labeled_genes,
)
from train_flux import media_report  # noqa: E402

from torchcell.datasets.scerevisiae.messner2023 import (  # noqa: E402
    build_uniprot_to_orf_map,
)
from torchcell.metabolism.constraints import build_gem_tensors  # noqa: E402
from torchcell.metabolism.flux_layer import (  # noqa: E402
    genotype_key,
    parsimonious_reference_flux,
)
from torchcell.metabolism.parameters import (  # noqa: E402
    GECKO3_YEAST_PROTEIN_POOL,
    KCAT_SIGMA_LADDER,
    KINETICS_MIRROR,
    PLAUSIBILITY_RESULT,
    VERIFIED_PAIRS_FEATURE,
    ParameterProvenance,
    SubstrateMatch,
    load_predicted_unit_table,
    molecular_weight_table,
    resolve_kcat_table,
)
from torchcell.metabolism.yeast_GEM import YeastGEM  # noqa: E402

Variant = Literal["point", "sigma1", "sigma2", "sampled", "moma"]
VARIANTS: tuple[Variant, ...] = ("point", "sigma1", "sigma2", "sampled", "moma")
SIGMA_K = {"point": 0.0, "sigma1": 1.0, "sigma2": 2.0}
#: ``capacity_calibration.py``'s geometric grid, reused verbatim.
SCALES = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)
#: The scale rule: the smallest grid scale at which the wild type keeps this fraction of
#: its unconstrained pFBA growth.
WILD_TYPE_GROWTH_FRACTION = 0.9
#: A deletion whose growth is below this is lethal and stored as the zero row.
LETHAL_GROWTH = 1e-6
SAMPLED_DRAWS = 10
SAMPLED_SEED = 0
SPLIT_FILE = "ess_fcl_val0.2_s0.json.gz"


class ProteomeAbundance(BaseModel):
    """Where the per-gene abundance came from and how far it reaches into the GEM."""

    model_config = {"extra": "forbid"}

    source_file: str
    source_sha256: str
    metadata_file: str
    n_wild_type_columns: int
    n_proteins_measured: int
    n_gem_genes: int
    n_gem_genes_with_abundance: int
    total_protein_g_per_gdw: float
    conversion: str
    abundance_mmol_per_gdw_quantiles: dict[str, float]


class KcatStoreProvenance(StoreProvenance):
    """``StoreProvenance`` plus the ceiling fields of a capacity-bounded store."""

    variant: str
    kcat_tier_counts: dict[str, int]
    kcat_sigma_k: float | None
    kcat_sigma_ladder_version: str
    capacity_scale: float | None
    scale_rule: str
    scale_grid: list[float]
    wild_type_growth_unconstrained: float
    n_units: int
    n_units_with_abundance: int
    n_reactions_with_gpr: int
    n_reactions_with_ceiling: int
    n_ceilings_binding_wild_type: int
    proteome: ProteomeAbundance | None
    sampled_draws: int | None = None
    sampled_seed: int | None = None
    ceiling_rule: str


class GeneRow(BaseModel):
    gene: str
    growth: float
    growth_ratio: float
    plain_growth_ratio: float
    lethal: bool
    relative_l2_log1p_to_plain: float


# --------------------------------------------------------------------------------------
# worker state (inherited by fork; set before the Pool is created)
# --------------------------------------------------------------------------------------
_S: torch.Tensor | None = None
_BIOMASS: int = -1
_KNOCKED: dict[str, np.ndarray] = {}
_BOXES: list[tuple[np.ndarray, np.ndarray]] = []
_V_WT: np.ndarray | None = None


def _knocked_box(gene: str, box: int) -> tuple[torch.Tensor, torch.Tensor]:
    lb, ub = _BOXES[box]
    k_lb, k_ub = lb.copy(), ub.copy()
    idx = _KNOCKED[gene]
    k_lb[idx] = 0.0
    k_ub[idx] = 0.0
    return torch.from_numpy(k_lb).float(), torch.from_numpy(k_ub).float()


def solve_pfba(task: tuple[str, int]) -> tuple[str, int, np.ndarray, float]:
    """One deletion's two-stage pFBA on box ``task[1]``; lethal is the zero row."""
    gene, box = task
    assert _S is not None
    k_lb, k_ub = _knocked_box(gene, box)
    try:
        record = parsimonious_reference_flux(_S, k_lb, k_ub, _BIOMASS)
        flux, growth = record.v.numpy().astype(np.float32), record.objective_value
    except ValueError:
        # The growth LP did not solve under the knocked-out box: a lethal deletion is
        # a result, the zero flux, as in the plain store.
        return gene, box, np.zeros(k_lb.numel(), dtype=np.float32), 0.0
    if growth < LETHAL_GROWTH:
        return gene, box, np.zeros(k_lb.numel(), dtype=np.float32), float(growth)
    return gene, box, flux, float(growth)


def solve_moma(task: tuple[str, int]) -> tuple[str, int, np.ndarray, float]:
    r"""Linear MOMA: :math:`\min\sum_j|v_j-v^{wt}_j|` s.t. :math:`Sv=0`, knocked box.

    One LP through scipy's HiGHS, the same path ``parsimonious_reference_flux`` takes.
    Variables ``[v, p, n]`` with ``v - p + n = v_wt`` and ``p, n >= 0``.
    """
    from scipy.optimize import linprog
    from scipy.sparse import bmat, csr_matrix, identity

    gene, box = task
    assert _S is not None and _V_WT is not None
    k_lb, k_ub = _knocked_box(gene, box)
    dense = (_S.to_dense() if _S.is_sparse else _S).double().numpy()
    m, r = dense.shape
    s = csr_matrix(dense)
    eye = identity(r, format="csr")
    a_eq = bmat([[s, None, None], [eye, -eye, eye]], format="csr")
    b_eq = np.concatenate([np.zeros(m), _V_WT.astype(np.float64)])
    c = np.concatenate([np.zeros(r), np.ones(2 * r)])
    bounds = list(zip(k_lb.double().numpy(), k_ub.double().numpy(), strict=True)) + [
        (0.0, None)
    ] * (2 * r)
    result = linprog(c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        return gene, box, np.zeros(r, dtype=np.float32), 0.0
    v = result.x[:r].astype(np.float32)
    growth = float(v[_BIOMASS])
    if growth < LETHAL_GROWTH:
        return gene, box, np.zeros(r, dtype=np.float32), growth
    return gene, box, v, growth


# --------------------------------------------------------------------------------------
# ceilings
# --------------------------------------------------------------------------------------
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def measured_abundance(
    model_dir: str, gene_ids: list[str]
) -> tuple[np.ndarray, ProteomeAbundance]:
    """``[n_gem_genes]`` mmol/gDW from the Messner 2023 wild type, nan where unmeasured."""
    matrix_path = osp.join(MESSNER_RAW, MESSNER_MATRIX)
    meta = pd.read_csv(osp.join(MESSNER_RAW, MESSNER_METADATA))
    wt_files = list(meta[meta["sampletype"] == "HIS3"]["Filename"])
    header = pd.read_csv(matrix_path, nrows=0).columns
    wt_cols = [f for f in wt_files if f in set(header)]
    assert wt_cols, "no HIS3 wild-type columns in the Messner matrix"
    matrix = pd.read_csv(matrix_path, usecols=[header[0], *wt_cols]).set_index(
        header[0]
    )
    up2orf = build_uniprot_to_orf_map()
    unmapped = [p for p in matrix.index.astype(str) if p not in up2orf]
    assert not unmapped, f"{len(unmapped)} Messner proteins have no UniProt->ORF map"
    wt_mean = matrix.mean(axis=1)
    wt_mean.index = matrix.index.astype(str).map(up2orf)
    wt_mean = wt_mean.dropna()
    # Two protein groups on one ORF are two quantified forms of the same gene product;
    # their mass adds.
    wt_mean = wt_mean.groupby(level=0).sum()
    mass_fraction = wt_mean / float(wt_mean.sum())
    mw = molecular_weight_table(model_dir, gene_ids)
    p_tot = GECKO3_YEAST_PROTEIN_POOL.total_protein_g_per_gdw
    assert p_tot is not None
    out = np.full(len(gene_ids), np.nan)
    for i, gene in enumerate(gene_ids):
        if gene not in mass_fraction.index:
            continue
        if mw.provenance[i] is not ParameterProvenance.SWISSPROT:
            continue
        # g/gDW divided by g/mmol (kDa = kg/mol = g/mmol) is mmol/gDW.
        out[i] = float(mass_fraction[gene]) * p_tot / float(mw.values[i])
    have = out[np.isfinite(out)]
    quantiles = (0.05, 0.25, 0.5, 0.75, 0.95)
    record = ProteomeAbundance(
        source_file=matrix_path,
        source_sha256=_sha256(matrix_path),
        metadata_file=osp.join(MESSNER_RAW, MESSNER_METADATA),
        n_wild_type_columns=len(wt_cols),
        n_proteins_measured=int(len(wt_mean)),
        n_gem_genes=len(gene_ids),
        n_gem_genes_with_abundance=int(np.isfinite(out).sum()),
        total_protein_g_per_gdw=float(p_tot),
        conversion=(
            "total protein approach: E_g = (MaxLFQ_g / sum over detected proteins of "
            "MaxLFQ) x P_tot / MW_g, with MaxLFQ the mean over the non-missing HIS3 "
            "wild-type columns, P_tot from GECKO3_YEAST_PROTEIN_POOL and MW_g from "
            "SwissProt (kDa). The denominator spans the detected proteins only, so every "
            "fraction overstates its share by the undetected mass."
        ),
        abundance_mmol_per_gdw_quantiles={
            str(q): float(v) for q, v in zip(quantiles, np.quantile(have, quantiles))
        },
    )
    return out, record


class Ceilings(BaseModel):
    """Per-reaction capacity at scale 1, mmol/gDW/h; nan where the GEM bound is kept."""

    model_config = {"arbitrary_types_allowed": True}

    per_reaction: np.ndarray
    n_units_with_abundance: int
    n_reactions_with_gpr: int
    n_reactions_with_ceiling: int


def reaction_ceilings(
    kcat_per_s: np.ndarray,
    unit_reaction: np.ndarray,
    unit_abundance: np.ndarray,
    n_reactions: int,
) -> Ceilings:
    """``sum_u kcat_u E_u`` over each reaction's units, the layer's ``dynamic_box`` sum."""
    has_abundance = np.isfinite(unit_abundance)
    per_unit = np.where(has_abundance, kcat_per_s * 3600.0 * unit_abundance, 0.0)
    cap = np.zeros(n_reactions)
    np.add.at(cap, unit_reaction, per_unit)
    n_units_rxn = np.zeros(n_reactions, dtype=np.int64)
    np.add.at(n_units_rxn, unit_reaction, 1)
    n_covered_rxn = np.zeros(n_reactions, dtype=np.int64)
    np.add.at(n_covered_rxn, unit_reaction, has_abundance.astype(np.int64))
    has_gpr = n_units_rxn > 0
    covered = has_gpr & (n_covered_rxn == n_units_rxn)
    out = np.where(covered, cap, np.nan)
    return Ceilings(
        per_reaction=out,
        n_units_with_abundance=int(has_abundance.sum()),
        n_reactions_with_gpr=int(has_gpr.sum()),
        n_reactions_with_ceiling=int(covered.sum()),
    )


def apply_ceilings(
    lb: np.ndarray, ub: np.ndarray, ceiling: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """``min(GEM ub, scale x ceiling)`` forward and the mirror on the reverse side."""
    cap = np.where(np.isfinite(ceiling), scale * ceiling, np.inf)
    return np.maximum(lb, -cap), np.minimum(ub, cap)


def wild_type_growth(lb: np.ndarray, ub: np.ndarray) -> float:
    assert _S is not None
    try:
        record = parsimonious_reference_flux(
            _S, torch.from_numpy(lb).float(), torch.from_numpy(ub).float(), _BIOMASS
        )
    except ValueError:
        return 0.0
    return float(record.objective_value)


def choose_scale(
    lb: np.ndarray, ub: np.ndarray, ceilings: list[np.ndarray], target: float
) -> tuple[float, list[dict[str, float]]]:
    """The smallest grid scale at which the wild type reaches ``target``.

    With one ceiling vector this is the wild type's own growth. With several (the sampled
    variant's draws) it is the MEDIAN over draws: the draws are independent across 3,728
    units, so the minimum over draws is set by whichever unit drew lowest and does not
    reach the target on this grid (measured: 0.131 of 0.543 at scale 10,000).
    """
    rows: list[dict[str, float]] = []
    for scale in SCALES:
        growths = [wild_type_growth(*apply_ceilings(lb, ub, c, scale)) for c in ceilings]
        median = float(np.median(growths))
        rows.append(
            {
                "scale": scale,
                "wild_type_growth_min": min(growths),
                "wild_type_growth_median": median,
                "wild_type_growth_max": max(growths),
            }
        )
        print(
            f"  scale {scale:g}: wild-type growth min {min(growths):.4f} median "
            f"{median:.4f} max {max(growths):.4f}",
            flush=True,
        )
        if median >= target:
            return scale, rows
    raise SystemExit(
        f"no scale on {SCALES} reaches {target:.4f} wild-type growth; the box is not "
        "usable and no larger scale is invented."
    )


def main() -> None:
    global _S, _BIOMASS, _KNOCKED, _BOXES, _V_WT
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--processes", type=int, default=32)
    parser.add_argument(
        "--limit-genes",
        type=int,
        default=0,
        help="Smoke test: solve only the first N genes (0 = every labeled gene).",
    )
    parser.add_argument(
        "--out-dir",
        default=TABLE_DIR,
        help="Smoke test: write the npz here instead of the reference_flux directory.",
    )
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    args = parser.parse_args()
    variant: Variant = args.variant
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("PYTHONHASHSEED must be 0, as for the plain store.")
    # The workers are forked; a threaded torch op in the parent before the fork can hang
    # the children, so every dense conversion here runs single-threaded.
    torch.set_num_threads(1)
    started = time.time()

    plain = np.load(PLAIN_STORE, allow_pickle=False)
    plain_provenance = StoreProvenance.model_validate_json(str(plain["provenance"]))
    genes = [str(g) for g in plain["genes"]]
    assert genes == labeled_genes(SPLIT_FILE), "plain store genes differ from the split"
    plain_v = plain["v_deletion"].astype(np.float32)
    if args.limit_genes:
        genes = genes[: args.limit_genes]
        plain_v = plain_v[: args.limit_genes]
    plain_ratio = plain["gene_flux_stats"][
        :, list(plain["stat_names"]).index("deletion_growth_ratio")
    ].astype(np.float64)[: len(genes)]

    source = YeastGEM()
    model = source.model
    media = media_report(model, "YPD")
    assert media["bounds_sha256"] == plain_provenance.media_report.bounds_sha256
    gem = build_gem_tensors(model, source.model_dir)
    rxn_ids = list(gem.rxn_ids)
    assert rxn_ids == [str(r) for r in plain["rxn_ids"]], "rxn_ids differ from the plain store"
    n_rxn = len(rxn_ids)
    _S = gem.s
    _BIOMASS = int(gem.biomass_index)
    base_lb = gem.lb.clamp(-FLUX_SCALE, FLUX_SCALE).double().numpy()
    base_ub = gem.ub.clamp(-FLUX_SCALE, FLUX_SCALE).double().numpy()
    gem_genes = {g.id for g in model.genes}
    assert all(g in gem_genes for g in genes)

    rxn_index = {r: i for i, r in enumerate(rxn_ids)}
    for gene in genes:
        with model:
            knocked = knock_out_model_genes(model, [gene])
            _KNOCKED[gene] = np.array([rxn_index[r.id] for r in knocked], dtype=np.int64)
    print(f"{len(genes)} genes, knockouts indexed, {time.time() - started:.0f}s", flush=True)

    unconstrained = wild_type_growth(base_lb, base_ub)
    assert abs(unconstrained - plain_provenance.wild_type_growth) < 1e-6, (
        unconstrained,
        plain_provenance.wild_type_growth,
    )
    target = WILD_TYPE_GROWTH_FRACTION * unconstrained

    units = gem.catalytic_units
    unit_reaction = units.unit_reaction.numpy()
    proteome: ProteomeAbundance | None = None
    tier_counts: dict[str, int] = {}
    sigma_k: float | None = None
    scale: float | None = None
    scale_rows: list[dict[str, float]] = []
    n_units_with_abundance = 0
    n_rxn_gpr = int(len(set(unit_reaction.tolist())))
    n_rxn_ceiling = 0
    ceiling_rule = (
        "ceiling_j = sum over units u of reaction j of kcat_u[1/s] x 3600 x E_u[mmol/gDW], "
        "E_u = min over the unit's member genes of the measured abundance, defined only "
        "when every member gene is measured; a reaction takes a ceiling only when every "
        "one of its units has an abundance, else it keeps the GEM bound; applied as "
        "ub <- min(ub, scale x ceiling) and lb <- max(lb, -scale x ceiling)"
    )
    if variant == "moma":
        boxes = [(base_lb, base_ub)]
        ceiling_rule = "none: the plain GEM box on the corrected YPD, no capacity ceilings"
        wt_vectors = [
            parsimonious_reference_flux(
                _S, torch.from_numpy(base_lb).float(), torch.from_numpy(base_ub).float(), _BIOMASS
            ).v.numpy().astype(np.float32)
        ]
        assert np.allclose(wt_vectors[0], plain["v_wild_type"], atol=1e-5)
        _V_WT = wt_vectors[0]
        wt_growths = [float(_V_WT[_BIOMASS])]
        n_binding = 0
    else:
        gene_abundance, proteome = measured_abundance(source.model_dir, units.gene_ids)
        unit_of = units.unit_gene_index[0].numpy()
        gene_of = units.unit_gene_index[1].numpy()
        unit_abundance = np.full(units.n_units, np.inf)
        np.minimum.at(unit_abundance, unit_of, gene_abundance[gene_of])
        unit_abundance[~np.isfinite(unit_abundance)] = np.nan
        print(
            f"proteome: {proteome.n_proteins_measured} proteins, "
            f"{proteome.n_gem_genes_with_abundance}/{proteome.n_gem_genes} GEM genes, "
            f"{int(np.isfinite(unit_abundance).sum())}/{units.n_units} units",
            flush=True,
        )
        kinetics_dir = osp.join(DATA_ROOT, KINETICS_MIRROR)
        substrates = SubstrateMatch.from_verified_pairs(
            osp.join(kinetics_dir, VERIFIED_PAIRS_FEATURE)
        )
        predicted = load_predicted_unit_table(
            "k_cat", units.n_units, kinetics_dir, osp.join(EXPERIMENT_ROOT, PLAUSIBILITY_RESULT)
        )
        sigma_k = SIGMA_K.get(variant, 0.0)
        table = resolve_kcat_table(
            units,
            source.model_dir,
            OED_SCEREVISIAE,
            predicted=predicted,
            substrates=substrates,
            homolog_mirror_dir=OED_ALL,
            kcat_ceiling_sigma_k=sigma_k,
        )
        tier_counts = table.tier_counts()
        print(f"kcat tiers {tier_counts}, sigma_k {sigma_k}", flush=True)
        kcat = table.values.numpy().astype(np.float64)
        if variant == "sampled":
            assert table.log10_sigma is not None
            sigma = table.log10_sigma.numpy().astype(np.float64)
            rng = np.random.default_rng(SAMPLED_SEED)
            draws = [
                10.0 ** (np.log10(kcat) + sigma * rng.standard_normal(units.n_units))
                for _ in range(SAMPLED_DRAWS)
            ]
        else:
            draws = [kcat]
        ceilings = [reaction_ceilings(d, unit_reaction, unit_abundance, n_rxn) for d in draws]
        n_units_with_abundance = ceilings[0].n_units_with_abundance
        n_rxn_gpr = ceilings[0].n_reactions_with_gpr
        n_rxn_ceiling = ceilings[0].n_reactions_with_ceiling
        print(
            f"ceilings on {n_rxn_ceiling}/{n_rxn_gpr} GPR reactions; choosing the scale",
            flush=True,
        )
        scale, scale_rows = choose_scale(
            base_lb, base_ub, [c.per_reaction for c in ceilings], target
        )
        boxes = [apply_ceilings(base_lb, base_ub, c.per_reaction, scale) for c in ceilings]
        wt_vectors, wt_growths, n_binding_draws = [], [], []
        for (lb, ub), c in zip(boxes, ceilings, strict=True):
            record = parsimonious_reference_flux(
                _S, torch.from_numpy(lb).float(), torch.from_numpy(ub).float(), _BIOMASS
            )
            v = record.v.numpy().astype(np.float32)
            wt_vectors.append(v)
            wt_growths.append(float(record.objective_value))
            cap = scale * c.per_reaction
            binding = np.isfinite(cap) & (np.abs(v.astype(np.float64)) >= cap * (1 - 1e-6) - 1e-9)
            n_binding_draws.append(int(binding.sum()))
        n_binding = int(round(float(np.mean(n_binding_draws))))
        print(
            f"scale {scale:g}: wild-type growth {np.mean(wt_growths):.4f}, "
            f"{n_binding_draws} ceilings binding", flush=True
        )
    _BOXES = boxes

    tasks = [(gene, b) for b in range(len(boxes)) for gene in genes]
    solve = solve_moma if variant == "moma" else solve_pfba
    with Pool(args.processes) as pool:
        results = pool.map(solve, tasks, chunksize=4)
    print(f"{len(tasks)} solves, {time.time() - started:.0f}s", flush=True)
    order = {g: i for i, g in enumerate(genes)}
    per_box = np.zeros((len(boxes), len(genes), n_rxn), dtype=np.float32)
    growth_box = np.zeros((len(boxes), len(genes)), dtype=np.float64)
    for gene, box, v, growth in results:
        per_box[box, order[gene]] = v
        growth_box[box, order[gene]] = growth

    wt_growth = float(np.mean(wt_growths))
    if variant == "sampled":
        mean = per_box.mean(axis=0)
        sd = per_box.std(axis=0, ddof=1)
        v_deletion = np.concatenate([mean, sd], axis=1).astype(np.float32)
        wt_stack = np.stack(wt_vectors)
        v_wild_type = np.concatenate([wt_stack.mean(axis=0), wt_stack.std(axis=0, ddof=1)])
        names = rxn_ids + [f"sd_{r}" for r in rxn_ids]
        growth = growth_box.mean(axis=0)
        lethal = (growth_box < LETHAL_GROWTH).all(axis=0)
    else:
        v_deletion = per_box[0]
        v_wild_type = wt_vectors[0]
        names = rxn_ids
        growth = growth_box[0]
        lethal = growth < LETHAL_GROWTH
    ratio = growth / wt_growth

    # Distances to the plain store in the head's encoding, log(1 + |v|), on the flux
    # columns only.
    enc_new = np.log1p(np.abs(v_deletion[:, :n_rxn].astype(np.float64)))
    enc_plain = np.log1p(np.abs(plain_v.astype(np.float64)))
    rel = np.linalg.norm(enc_new - enc_plain, axis=1) / np.maximum(
        np.linalg.norm(enc_plain, axis=1), 1e-12
    )
    rows = [
        GeneRow(
            gene=g,
            growth=float(growth[i]),
            growth_ratio=float(ratio[i]),
            plain_growth_ratio=float(plain_ratio[i]),
            lethal=bool(lethal[i]),
            relative_l2_log1p_to_plain=float(rel[i]),
        )
        for i, g in enumerate(genes)
    ]

    flux_sha = hashlib.sha256(v_deletion.tobytes()).hexdigest()
    base = dict(
        gem=osp.basename(source.model_dir),
        growth_fraction=1.0,
        flux_scale=FLUX_SCALE,
        method=(
            (
                "linear MOMA, min sum_j |v_j - v_wt_j| s.t. S v = 0 on the knocked-out "
                "plain GEM box (NO enzyme-capacity ceilings), one scipy HiGHS LP with |.| "
                "split into positive and negative parts, v_wt the two-stage pFBA wild type"
            )
            if variant == "moma"
            else (
                f"pFBA (parsimonious_reference_flux, two scipy HiGHS LPs) on the GEM box "
                f"capped by enzyme-capacity ceilings (variant {variant}, kcat "
                f"sigma_k {sigma_k}, capacity scale {scale:g}, {n_rxn_ceiling} reactions "
                f"ceilinged from the Messner 2023 wild-type proteome)"
                + (
                    f"; mean over {SAMPLED_DRAWS} kcat draws (seed {SAMPLED_SEED}) with "
                    "the per-reaction sd over draws as a second block"
                    if variant == "sampled"
                    else ""
                )
            )
        )
        + "; gene deleted through cobra.manipulation.knock_out_model_genes; written by "
        "ess_kcat_bounded_store.py",
        solver="pfba" if variant != "moma" else None,
        n_nonoptimal_zero_flux=0,
        medium="YPD",
        media_report=MediaApplied.model_validate(media),
        split_file=SPLIT_FILE,
        wild_type_growth=wt_growth,
        wild_type_key=genotype_key([]),
        n_sets=len(genes) + 1,
        n_lethal_zero_flux=int(lethal.sum()),
        n_genes_outside_the_gem=0,
        background_genes=[],
        python_hash_seed=os.environ.get("PYTHONHASHSEED"),
        flux_sha256=flux_sha,
    )
    provenance = StoreProvenance(**base)
    kcat_provenance = KcatStoreProvenance(
        **base,
        variant=variant,
        kcat_tier_counts=tier_counts,
        kcat_sigma_k=sigma_k,
        kcat_sigma_ladder_version=KCAT_SIGMA_LADDER.version,
        capacity_scale=scale,
        scale_rule=(
            f"the smallest scale on the calibration grid at which the wild type (the "
            f"median over draws, for sampled) reaches at least {WILD_TYPE_GROWTH_FRACTION} of its "
            "unconstrained pFBA growth; the un-scaled box was measured infeasible in "
            "results/capacity_calibration_predicted.json"
        ),
        scale_grid=list(SCALES),
        wild_type_growth_unconstrained=unconstrained,
        n_units=int(units.n_units),
        n_units_with_abundance=n_units_with_abundance,
        n_reactions_with_gpr=n_rxn_gpr,
        n_reactions_with_ceiling=n_rxn_ceiling,
        n_ceilings_binding_wild_type=n_binding,
        proteome=proteome,
        sampled_draws=SAMPLED_DRAWS if variant == "sampled" else None,
        sampled_seed=SAMPLED_SEED if variant == "sampled" else None,
        ceiling_rule=ceiling_rule,
    )
    path = osp.join(args.out_dir, f"ess_solve_encodings_YPD_kcat{variant}.npz")
    assert not osp.exists(path), f"{path} exists; outputs are never overwritten"
    np.savez(
        path,
        genes=np.array(genes),
        v_deletion=v_deletion,
        v_wild_type=v_wild_type.astype(np.float32),
        rxn_ids=np.array(names),
        provenance=np.array(provenance.model_dump_json()),
        kcat_provenance=np.array(kcat_provenance.model_dump_json()),
        growth_ratio=ratio.astype(np.float32),
    )
    moved_ratio = int((np.abs(ratio - plain_ratio) > 0.01).sum())
    moved_vector = int((rel > 1e-6).sum())
    summary: dict[str, Any] = {
        "store": path,
        "shape": list(v_deletion.shape),
        "variant": variant,
        "capacity_scale": scale,
        "scale_rows": scale_rows,
        "wild_type_growth": wt_growth,
        "wild_type_growth_unconstrained": unconstrained,
        "wild_type_growth_per_draw": wt_growths,
        "n_ceilings_binding_wild_type": n_binding,
        "n_reactions_with_ceiling": n_rxn_ceiling,
        "n_reactions_with_gpr": n_rxn_gpr,
        "n_units_with_abundance": n_units_with_abundance,
        "n_units": int(units.n_units),
        "kcat_tier_counts": tier_counts,
        "n_genes": len(genes),
        "n_lethal": int(lethal.sum()),
        "n_lethal_plain": int((plain_ratio * plain_provenance.wild_type_growth < LETHAL_GROWTH).sum()),
        "n_genes_ratio_moved_gt_0.01": moved_ratio,
        "n_genes_log1p_vector_moved": moved_vector,
        "median_relative_l2_log1p_to_plain": float(np.median(rel)),
        "median_relative_l2_log1p_to_plain_viable": float(np.median(rel[~lethal]))
        if (~lethal).any()
        else None,
        "seconds": time.time() - started,
        "processes": args.processes,
        "kcat_provenance": kcat_provenance.model_dump(),
        "rows": [r.model_dump() for r in rows],
    }
    out = osp.join(args.results_dir, f"ess_kcat_bounded_store_{variant}.json")
    assert not osp.exists(out), f"{out} exists; outputs are never overwritten"
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(
        json.dumps({k: v for k, v in summary.items() if k not in ("rows", "kcat_provenance", "scale_rows")}, indent=2)
    )
    print(f"{path}\n{out}", flush=True)


if __name__ == "__main__":
    main()
