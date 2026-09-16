# experiments/027-betaxanthin-metabolic/scripts/cbm_baseline_bx.py
# [[experiments.027-betaxanthin-metabolic.cpu-baselines-2026-09-14]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/cbm_baseline_bx
"""The per-genotype constraint-based baseline for the three metabolic targets.

WHY THIS EXISTS. 027 is a metabolic module scored against Merzbacher's FCL RandomForest,
and neither side has ever been read against the obvious prior: what a genome-scale
stoichiometric model says about the same knockouts, with no learning at all. Without that
row, "the flux module helps" and "flux helps" are the same sentence.

WHAT A GENOTYPE IS, AND WHY THE MARKERS ARE IN IT. Cachera's screen is the BY4741 knockout
collection, so every strain already carries four marker deletions, ``his3D1 leu2D0
met15D0 ura3D0``, before the screened gene is touched. A baseline that knocks out only the
screened gene is scoring a prototroph that does not exist. All four markers resolve in
yeast-GEM 9.0.2 and are applied through cobra's GPR-aware knockout, so an isozyme keeps its
reaction and a sole catalyst loses it.

THE MEDIUM FINDING, MEASURED HERE AND RECORDED IN THE RESULT JSON. Cachera grew the screen
on YPD + G418; ``torchcell/datasets/scerevisiae/cachera2023.py:321`` emits
``Media(name="SC", state="solid")``, which is wrong about the bench. The GEM stand-in for
YPD (``YPD_APPROX_FBA``, Suthers' "YNB plus 20 amino acids") carries no uracil and no
adenine, because yeast extract is never modeled. MEASURED: under that bound vector the
``ura3D0`` marker alone drops growth from 0.5352 to 0.0000 /h, so the whole BY4741
background is dead in silico while the bench strain grows perfectly well. The bound vector
that keeps both the YPD reading and a viable background is exactly YPD-approx plus those two
nucleobases, which is ``SC_FBA``. Both are run and both are reported; the primary condition
is named on the command line and is asserted to have a live background.

READOUTS, all at 90 % growth floors and all pre-registered in the review note:

* ``growth`` of the knockout,
* production at 90 % of the REFERENCE strain's growth (comparable across genotypes),
* production at 90 % of the knockout's OWN growth (the headroom reading),
* the shadow price of cytosolic L-tyrosine (betaxanthin's precursor), free from the same LP,
* pFBA total flux and per-demand FVA width at 0.9, on the primary condition only.

Run from the worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \\
        experiments/027-betaxanthin-metabolic/scripts/cbm_baseline_bx.py \\
        --target betaxanthin --primary-condition sc_markers
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import pickle
import platform
import sys
import time
from typing import Any

import cobra
import lmdb
import numpy as np
from cobra.flux_analysis import flux_variability_analysis, pfba
from cobra.manipulation import knock_out_model_genes
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from cbm_scoring import assess, assess_per_feature  # noqa: E402

from torchcell.metabolism.betaxanthin import (  # noqa: E402
    betaxanthin_demand_ids,
    build_betaxanthin_pathway,
)
from torchcell.metabolism.media import FBA_MEDIA, media_to_bounds  # noqa: E402
from torchcell.metabolism.pathway import apply_pathway  # noqa: E402
from torchcell.metabolism.yeast_GEM import YeastGEM  # noqa: E402

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "results")
DUMP_DIR = osp.join(DATA_ROOT, "test-predictions-027-wave7")
PINNED_SPLIT = osp.join(
    EXPERIMENT_ROOT, "020-cachera-betaxanthin/results/merzbacher_nested_split.json"
)
ISOBUTANOL_LMDB = osp.join(
    DATA_ROOT, "data/torchcell/isobutanol_screen_lopez2024/processed/lmdb"
)

#: BY4741's four marker deletions, as systematic ORFs. MET15 and MET17 are the same gene
#: (YLR303W); the collection's name for it is MET15 and the GEM's is MET17.
MARKER_KNOCKOUTS: dict[str, str] = {
    "HIS3": "YOR202W",
    "LEU2": "YCL018W",
    "MET15/MET17": "YLR303W",
    "URA3": "YEL021W",
}
#: Cytosolic L-tyrosine, betaxanthin's precursor (train_flux.AMINO_ACID_TO_GEM_METABOLITE).
TYROSINE_C = "s_1051"
#: 026's amino-acid to GEM metabolite map, duplicated here rather than imported so this
#: script does not pull the trainer's torch stack onto a CPU node. Any divergence is caught
#: by the assertion below against the dump's own column order.
AMINO_ACID_TO_GEM_METABOLITE: dict[str, str] = {
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
AA_KEYS = sorted(AMINO_ACID_TO_GEM_METABOLITE)
#: Isobutanol exchange, native to yeast-GEM 9.0.2; no cassette is needed for this target.
ISOBUTANOL_EXCHANGE = "r_1866"
#: 2-isopropylmalate, the Leu3p ligand the Lopez biosensor actually reports, in its two
#: compartments.
IPM_METABOLITES = {"ipm_c": "s_0162", "ipm_m": "s_0164"}

GROWTH_FRACTION = 0.9

#: (condition name -> medium key, whether the BY4741 markers are deleted) per target. Every
#: condition is computed; the primary one additionally gets pFBA and FVA.
CONDITIONS: dict[str, dict[str, tuple[str, bool]]] = {
    "betaxanthin": {
        "ypd_markers": ("YPD", True),
        "ypd_none": ("YPD", False),
        "sc_markers": ("SC", True),
        "sc_none": ("SC", False),
    },
    "amino_acids": {"sm_none": ("SM", False)},
    "isobutanol": {"sc_markers": ("SC", True), "sc_none": ("SC", False)},
}
DEFAULT_PRIMARY = {
    "betaxanthin": "sc_markers",
    "amino_acids": "sm_none",
    "isobutanol": "sc_markers",
}


# --------------------------------------------------------------------------------------
# targets
# --------------------------------------------------------------------------------------
def _dump_targets(head: str, arm: str) -> dict[str, Any]:
    """{gene -> measured target} over the PINNED test records of the wave-7 dumps.

    Read from the dumps rather than re-derived, so the gene set is the one the trainer
    scored, byte for byte. Every dump of the head is read and the targets are required to
    agree; a disagreement means two dumps were written against different builds and the
    comparison would be silently across gene sets.
    """
    # Selected by CONTENT, not by file name: since 2026-09-16 a dump is named after its
    # result file, so the arm no longer appears as a prefix.
    files = sorted(f for f in os.listdir(DUMP_DIR) if f.endswith(".json"))
    picked: list[str] = []
    targets: dict[str, Any] | None = None
    for name in files:
        path = osp.join(DUMP_DIR, name)
        with open(path) as fh:
            dump = json.load(fh)
        if dump["head"] != head or dump["cell"]["arm"] != arm:
            continue
        here = {
            rec["genes"][0]: rec["target"]
            for rec in dump["predictions"]
            if rec["pinned"] and len(rec["genes"]) == 1 and rec["target"] is not None
        }
        if targets is None:
            targets = here
        elif here != targets:
            raise ValueError(f"pinned targets in {path} differ from {picked[0]}")
        picked.append(path)
    if targets is None:
        raise FileNotFoundError(f"no {head} dumps for arm {arm!r} under {DUMP_DIR}")
    return {"targets": targets, "source_dumps": picked}


def _isobutanol_targets() -> dict[str, Any]:
    """{ORF -> log10 fold change} from the Lopez 2024 screen LMDB, on the pinned genes.

    log10 is a monotone transform of the released fold change, so it changes no rank
    statistic; it is taken because a ratio is read on a log scale and because the review
    named it.
    """
    with open(PINNED_SPLIT) as fh:
        pinned = set(json.load(fh)["split"]["test"])
    env = lmdb.open(ISOBUTANOL_LMDB, readonly=True, lock=False, subdir=True)
    values: dict[str, float] = {}
    with env.begin() as txn:
        n_entries = txn.stat()["entries"]
        for _, raw in txn.cursor():
            # The Lopez loader pickles its records (lopez2024.py:384), unlike the
            # Neo4j-built datasets, which store JSON.
            recs = pickle.loads(raw)
            if isinstance(recs, dict):
                recs = [recs]
            for rec in recs:
                genes = [
                    p["systematic_gene_name"]
                    for p in rec["experiment"]["genotype"]["perturbations"]
                ]
                if len(genes) != 1:
                    raise ValueError(f"record names {len(genes)} genes: {genes[:5]}")
                fc = rec["experiment"]["phenotype"]["metabolite_level"]["isobutanol"]
                values[genes[0]] = float(np.log10(float(fc)))
    env.close()
    return {
        "targets": {g: v for g, v in values.items() if g in pinned},
        "source_dumps": [ISOBUTANOL_LMDB],
        "lmdb_entries": n_entries,
        "lmdb_mtime": time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(osp.getmtime(ISOBUTANOL_LMDB))
        ),
        "n_screen_genes": len(values),
        "n_pinned_requested": len(pinned),
    }


# --------------------------------------------------------------------------------------
# model assembly
# --------------------------------------------------------------------------------------
def build_bench_model(media_key: str, cassette: bool) -> tuple[Any, list[str], dict[str, Any]]:
    """Yeast-GEM 9.0.2, optionally + the Cachera betaxanthin cassette, on one medium."""
    base = YeastGEM().model
    provenance: dict[str, Any] = {
        "base_reactions": len(base.reactions),
        "base_metabolites": len(base.metabolites),
        "base_genes": len(base.genes),
    }
    demands: list[str] = []
    model = base
    if cassette:
        pathway = build_betaxanthin_pathway(base)
        model = apply_pathway(base, pathway)
        demands = betaxanthin_demand_ids(pathway)
        provenance.update(
            cassette_reactions=len(pathway.reactions),
            cassette_metabolites=len(pathway.metabolites),
            cassette_evidence_census=pathway.evidence_census(),
            betaxanthin_demands=demands,
        )
    bounds = media_to_bounds(FBA_MEDIA[media_key], model)
    unresolved = [r.component_name for r in bounds.resolutions if r.outcome != "resolved"]
    if unresolved:
        raise ValueError(f"medium {media_key} has unresolved components: {unresolved}")
    bounds.apply(model)
    provenance.update(
        media_key=media_key,
        media_name=FBA_MEDIA[media_key].name,
        media_n_bounds=len(bounds.bounds),
        model_reactions=len(model.reactions),
        model_genes=len(model.genes),
    )
    return model, demands, provenance


def add_demand(model: Any, metabolite_id: str, reaction_id: str) -> Any:
    """A zero-lower-bound demand on a metabolite, so its synthesis flux can be maximized."""
    return model.add_boundary(
        model.metabolites.get_by_id(metabolite_id), type="demand", reaction_id=reaction_id
    )


# --------------------------------------------------------------------------------------
# per-genotype readouts
# --------------------------------------------------------------------------------------
def _maximize(model: Any, objective: Any, growth_rxn: Any, floor: float) -> float:
    """Max of ``objective`` subject to growth >= ``floor``; NaN when infeasible."""
    growth_rxn.lower_bound = floor
    model.objective = objective
    solution = model.optimize()
    return (
        float(solution.objective_value) if solution.status == "optimal" else float("nan")
    )


def run_condition(
    target: str,
    media_key: str,
    markers: bool,
    genes: list[str],
    full_readouts: bool,
) -> dict[str, Any]:
    """Every readout for one (medium, background) cell, over ``genes``."""
    model, btx_demands, provenance = build_bench_model(
        media_key, cassette=(target == "betaxanthin")
    )
    gem_genes = {g.id for g in model.genes}
    provenance["markers_applied"] = markers
    provenance["marker_genes"] = {
        name: {"orf": orf, "in_gem": orf in gem_genes}
        for name, orf in MARKER_KNOCKOUTS.items()
    }
    growth_rxn = next(r for r in model.reactions if r.objective_coefficient != 0)
    provenance["growth_reaction"] = growth_rxn.id
    growth_lb0 = growth_rxn.lower_bound
    provenance["growth_reaction_lower_bound"] = growth_lb0

    # Extra demand reactions, added ONCE. A demand with a zero lower bound forces no flux,
    # so the growth optimum is unchanged; this is asserted below.
    extra: dict[str, Any] = {}
    if target == "amino_acids":
        for key in AA_KEYS:
            extra[key] = add_demand(
                model, AMINO_ACID_TO_GEM_METABOLITE[key], f"DM_cbm_{key}"
            )
    elif target == "isobutanol":
        extra["isobutanol"] = model.reactions.get_by_id(ISOBUTANOL_EXCHANGE)
        for key, mid in IPM_METABOLITES.items():
            extra[key] = add_demand(model, mid, f"DM_cbm_{key}")

    wt = model.optimize()
    assert wt.status == "optimal", "wild-type LP is infeasible before any knockout"
    wt_growth = float(wt.objective_value)
    provenance["wild_type_growth_per_h"] = wt_growth

    marker_orfs = [MARKER_KNOCKOUTS[k] for k in MARKER_KNOCKOUTS] if markers else []
    background: Any = None
    if markers:
        knock_out_model_genes(model, marker_orfs)
        background = model.optimize()
        reference_growth = (
            float(background.objective_value)
            if background.status == "optimal"
            else 0.0
        )
    else:
        reference_growth = wt_growth
    provenance["background_growth_per_h"] = reference_growth
    provenance["background_knockouts"] = marker_orfs

    rows: dict[str, dict[str, Any]] = {}
    t0 = time.time()
    for i, gene in enumerate(genes):
        with model:
            if gene in gem_genes:
                knock_out_model_genes(model, [gene])
            solution = model.optimize()
            mu = (
                float(solution.objective_value)
                if solution.status == "optimal"
                else 0.0
            )
            row: dict[str, Any] = {"growth": mu, "in_gem": gene in gem_genes}
            if solution.status == "optimal":
                row["shadow_price_tyrosine_c"] = float(
                    solution.shadow_prices[TYROSINE_C]
                )
                if target == "betaxanthin":
                    demand_rxns = {
                        model.reactions.get_by_id(d): 1.0 for d in btx_demands
                    }
                    row["btx_at_90pct_reference_growth"] = _maximize(
                        model, demand_rxns, growth_rxn, GROWTH_FRACTION * reference_growth
                    )
                    row["btx_at_90pct_own_growth"] = _maximize(
                        model, demand_rxns, growth_rxn, GROWTH_FRACTION * mu
                    )
                elif target == "amino_acids":
                    row["aa_at_90pct_reference_growth"] = [
                        _maximize(
                            model,
                            extra[key],
                            growth_rxn,
                            GROWTH_FRACTION * reference_growth,
                        )
                        for key in AA_KEYS
                    ]
                else:
                    row["isobutanol_at_90pct_reference_growth"] = _maximize(
                        model, extra["isobutanol"], growth_rxn,
                        GROWTH_FRACTION * reference_growth,
                    )
                    row["isobutanol_at_90pct_own_growth"] = _maximize(
                        model, extra["isobutanol"], growth_rxn, GROWTH_FRACTION * mu
                    )
                    for key in IPM_METABOLITES:
                        row[f"{key}_at_90pct_reference_growth"] = _maximize(
                            model, extra[key], growth_rxn,
                            GROWTH_FRACTION * reference_growth,
                        )
                if full_readouts:
                    growth_rxn.lower_bound = growth_lb0
                    model.objective = growth_rxn
                    if mu > 1e-9:
                        row["pfba_total_flux"] = float(pfba(model).fluxes.abs().sum())
                        if target == "betaxanthin":
                            fva = flux_variability_analysis(
                                model,
                                reaction_list=btx_demands,
                                fraction_of_optimum=GROWTH_FRACTION,
                                loopless=False,
                                processes=1,
                            )
                            width = fva["maximum"] - fva["minimum"]
                            row["fva_width_sum_btx_demands"] = float(width.sum())
                            row["fva_width_max_btx_demand"] = float(width.max())
        # The context manager restores bounds, objective and knockouts; check the two that
        # a silent leak would corrupt every later genotype through.
        assert growth_rxn.lower_bound == growth_lb0, (
            "growth floor leaked out of the context"
        )
        rows[gene] = row
        if i % 100 == 0:
            print(
                f"    [{media_key}/{'markers' if markers else 'none'}] "
                f"{i}/{len(genes)}  {time.time() - t0:.0f}s",
                flush=True,
            )
    provenance["wall_clock_s"] = round(time.time() - t0, 1)
    provenance["s_per_genotype"] = round((time.time() - t0) / max(len(genes), 1), 4)
    return {"provenance": provenance, "rows": rows}


# --------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------
def score_condition(
    target: str,
    cell: dict[str, Any],
    genes: list[str],
    targets: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Every readout of one condition, scored against the measured target."""
    rows = cell["rows"]
    out: dict[str, Any] = {}
    scalar_keys = [
        k
        for k in (
            "growth",
            "shadow_price_tyrosine_c",
            "btx_at_90pct_reference_growth",
            "btx_at_90pct_own_growth",
            "isobutanol_at_90pct_reference_growth",
            "isobutanol_at_90pct_own_growth",
            "ipm_c_at_90pct_reference_growth",
            "ipm_m_at_90pct_reference_growth",
            "pfba_total_flux",
            "fva_width_sum_btx_demands",
            "fva_width_max_btx_demand",
        )
        if any(k in rows[g] for g in genes)
    ]
    if target == "amino_acids":
        pred = np.array(
            [
                rows[g].get("aa_at_90pct_reference_growth", [np.nan] * len(AA_KEYS))
                for g in genes
            ],
            dtype=float,
        )
        true = np.array(
            [
                [
                    np.nan if v is None else float(v)
                    for v in targets[g]
                ]
                for g in genes
            ],
            dtype=float,
        )
        out["aa_at_90pct_reference_growth"] = assess_per_feature(
            pred, true, AA_KEYS, rng
        )
        # A single scalar stand-in for the 19 columns, so the growth and pFBA readouts have
        # something to correlate against. It is the mean of the measured columns a genotype
        # HAS, and is reported as such rather than as the amino-acid endpoint.
        y = np.array(
            [
                np.mean(true[i][np.isfinite(true[i])])
                if np.isfinite(true[i]).any()
                else np.nan
                for i in range(len(genes))
            ]
        )
    else:
        y = np.array([float(targets[g]) for g in genes], dtype=float)
    for key in scalar_keys:
        x = np.array([rows[g].get(key, np.nan) for g in genes], dtype=float)
        out[key] = assess(x, y, rng)
    # CONTROL. A wild-type-only readout is the same number for every genotype, so it has no
    # rank information by construction and the scoring rule must refuse it rather than
    # return a correlation.
    constant = np.full(len(genes), cell["provenance"]["wild_type_growth_per_h"])
    control = assess(constant, y, rng)
    assert control["spearman"] is None and control["n_distinct_values"] == 1, (
        "the constant wild-type control produced a correlation; the scoring rule is wrong"
    )
    out["control_constant_readout"] = control
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--target", required=True, choices=["betaxanthin", "amino_acids", "isobutanol"]
    )
    ap.add_argument("--primary-condition", default=None)
    ap.add_argument(
        "--conditions",
        default=None,
        help=(
            "comma-separated subset of the target's conditions to run; the default runs "
            "all of them. Used when one condition cannot be completed inside a time bound "
            "and the run has to be narrowed to the primary."
        ),
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--max-genes",
        type=int,
        default=0,
        help="score only the first N genes; a SMOKE-TEST knob, never a reported run",
    )
    args = ap.parse_args()

    primary = args.primary_condition or DEFAULT_PRIMARY[args.target]
    assert primary in CONDITIONS[args.target], (
        f"--primary-condition {primary!r} is not one of "
        f"{sorted(CONDITIONS[args.target])}"
    )
    requested = (
        list(CONDITIONS[args.target])
        if args.conditions is None
        else args.conditions.split(",")
    )
    unknown = [c for c in requested if c not in CONDITIONS[args.target]]
    assert not unknown, f"--conditions names {unknown}, not in {sorted(CONDITIONS[args.target])}"
    assert primary in requested, "the primary condition must be among --conditions"
    rng = np.random.default_rng(args.seed)

    if args.target == "betaxanthin":
        loaded = _dump_targets("betaxanthin", "pooled")
    elif args.target == "amino_acids":
        loaded = _dump_targets("mulleder19", "pooled")
    else:
        loaded = _isobutanol_targets()
    targets = loaded["targets"]

    # The GEM's gene universe, read from the primary condition's own model so the scored
    # set is the one the readouts were computed on.
    media_key, markers = CONDITIONS[args.target][primary]
    probe, _, _ = build_bench_model(media_key, cassette=(args.target == "betaxanthin"))
    gem_genes = {g.id for g in probe.genes}
    genes = sorted(set(targets) & gem_genes)
    n_in_gem = len(genes)
    if args.max_genes:
        genes = genes[: args.max_genes]
    print(
        f"target {args.target}: {len(targets)} pinned genes with a finite measurement, "
        f"{n_in_gem} of them in the GEM, {len(genes)} scored",
        flush=True,
    )

    payload: dict[str, Any] = {
        "generated_by": "experiments/027-betaxanthin-metabolic/scripts/cbm_baseline_bx.py",
        "target": args.target,
        "primary_condition": primary,
        "growth_fraction": GROWTH_FRACTION,
        "solver": probe.solver.interface.__name__,
        "cobra_version": cobra.__version__,
        "python": platform.python_version(),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "target_source": {
            "dumps": loaded["source_dumps"],
            **{
                k: v
                for k, v in loaded.items()
                if k not in ("targets", "source_dumps")
            },
        },
        "n_pinned_with_measurement": len(targets),
        "n_pinned_in_gem": n_in_gem,
        "n_scored": len(genes),
        "max_genes_smoke_limit": args.max_genes,
        "medium_note": (
            "Cachera 2023 grew the screen on YPD + G418. "
            "torchcell/datasets/scerevisiae/cachera2023.py:321 emits Media(name='SC', "
            "state='solid'), which does not match the paper. The GEM stand-in for YPD "
            "(YPD_APPROX_FBA) omits uracil and adenine because yeast extract is not "
            "modeled, which makes the ura3D0 marker lethal in silico; SC_FBA is that same "
            "bound vector plus the two nucleobases. Both are run and reported."
        ),
        "conditions": {},
    }
    if args.target == "amino_acids":
        dump_columns = json.load(open(loaded["source_dumps"][0]))["metrics"]["pinned"][
            "columns"
        ]
        assert dump_columns == AA_KEYS, (
            f"amino-acid column order disagrees with the dumps: {dump_columns} vs {AA_KEYS}"
        )

    payload["conditions_requested"] = requested
    for name in requested:
        medium, use_markers = CONDITIONS[args.target][name]
        print(f"\ncondition {name}: medium {medium}, markers {use_markers}", flush=True)
        cell = run_condition(
            args.target, medium, use_markers, genes, full_readouts=(name == primary)
        )
        if name == primary:
            assert cell["provenance"]["background_growth_per_h"] > 1e-6, (
                f"the primary condition {name!r} has a dead background "
                f"({cell['provenance']['background_growth_per_h']} /h); pick a "
                "--primary-condition whose background strain grows"
            )
        payload["conditions"][name] = {
            "provenance": cell["provenance"],
            "scores": score_condition(args.target, cell, genes, targets, rng),
        }
        prov = cell["provenance"]
        print(
            f"  wild type {prov['wild_type_growth_per_h']:.6f} /h, background "
            f"{prov['background_growth_per_h']:.6f} /h, "
            f"{prov['s_per_genotype']:.3f} s/genotype",
            flush=True,
        )
        for key, cellscore in payload["conditions"][name]["scores"].items():
            rho = cellscore.get("spearman") if isinstance(cellscore, dict) else None
            if rho is None:
                mean = (
                    cellscore.get("mean_spearman")
                    if isinstance(cellscore, dict)
                    else None
                )
                if mean is not None:
                    print(f"    {key:<40s} mean per-feature rho {mean:+.4f}")
                continue
            ci = cellscore["bootstrap_ci95"]
            print(
                f"    {key:<40s} rho {rho:+.4f}  perm-p {cellscore['permutation_p']:.3f}"
                f"  CI [{ci[0]:+.4f},{ci[1]:+.4f}]  ties {cellscore['frac_at_mode']:.3f}"
            )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = args.out or osp.join(RESULTS_DIR, f"cbm_baseline_{args.target}.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
