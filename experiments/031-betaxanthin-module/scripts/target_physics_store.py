# experiments/031-betaxanthin-module/scripts/target_physics_store.py
# [[experiments.031-betaxanthin-module.scripts.target_physics_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-betaxanthin-module/scripts/target_physics_store

r"""Physically constrained deletion-flux stores for the betaxanthin and amino-acid targets.

The store affine of rounds B5 and A4 reads one parsimonious flux vector per deleted gene,
so the module carries mass balance and nothing else. The essentiality study's store ladder
(028, rounds 113 to 116) built the same vector under other programs on the plain GEM and
the labeled genes: linear MOMA, kcat ceilings, GECKO pools, a thermodynamic direction
rule. None of them is on the target models: the betaxanthin strain carries the Cachera
cassette with the BY4741 markers deleted on YPD, the amino-acid target is the prototroph
on synthetic minimal medium, and both stores cover every GEM gene. This script writes the
two cheapest physically constrained programs on those models, in the store shape the
trainer reads through ``ESS_SOLVE_TABLE``:

* ``moma``: linear minimization of metabolic adjustment,
  :math:`\min\sum_j|v_j-v^{wt}_j|` s.t. :math:`Sv=0` on the knocked-out box, one HiGHS
  LP per gene, :math:`v^{wt}` the source store's parsimonious wild-type growth vector.
  The deletion's flux is the nearest feasible flux to the parent's, not a re-optimized
  one, which is the assumption a knockout does not re-tune its enzyme levels;
* ``pfba``: the two-stage parsimonious FBA of the essentiality stores
  (``parsimonious_reference_flux``, HiGHS) on the plain box. The betaxanthin source store
  was solved by cobra's pFBA through GLPK and is degenerate (1,025 of its 1,031
  no-effect deletions carry a vector that differs from the wild type's, median relative
  log1p distance 0.38, measured 2026-09-23); this rebuild anchors every solve to one
  solver path, so a pair against the source isolates vertex noise;
* ``sign10`` / ``sign30``: two-stage parsimonious FBA on a box whose reaction directions
  are fixed by the standard reaction energies (eQuilibrator union-external route, the
  028 flux layer's table): a reaction with :math:`\Delta_r G'^\circ\le -T` loses its
  reverse part, one with :math:`\ge +T` its forward part, :math:`T` 10 or 30 kJ/mol.
  Unpriced reactions, the cassette's included, keep the GEM box. The second law enters
  the program rather than a loss.

The source store's gene list, reaction order and wild-type vector are reused, so a store
here differs from its source ONLY in the program; the growth-ratio CSV the trainer reads
(``ESS_FBA_CSV``) stays the source's, so a paired cell isolates the vector. Lethal (growth
below ``LETHAL_GROWTH``, or the program infeasible) is the zero row, as every store has it.
Writes ``$DATA_ROOT/data/torchcell/reference_flux/ess_solve_encodings_<YPD_bx|SM_gem>_<variant>.npz``
(never overwritten) and ``results/target_physics_store_<target>_<variant>.json`` in the
target's study. Run from the repo root::

    PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
        ~/miniconda3/envs/torchcell/bin/python \
        experiments/031-betaxanthin-module/scripts/target_physics_store.py \
        --target betaxanthin --variant moma --processes 8
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
import torch
from cobra.manipulation import knock_out_model_genes
from dotenv import load_dotenv
from pydantic import BaseModel
from scipy.optimize import linprog
from scipy.sparse import bmat, csr_matrix, identity

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
TABLE_DIR = osp.join(DATA_ROOT, "data/torchcell/reference_flux")
for study in ("028-gene-essentiality", "027-betaxanthin-metabolic", "026-metabolism-flux"):
    sys.path.insert(0, osp.join(EXPERIMENT_ROOT, study, "scripts"))
from bx_reference_flux_table import build_strain_model  # noqa: E402
from ess_gecko_store import ThermoApplied, apply_sign_rule  # noqa: E402
from ess_reference_flux_table import (  # noqa: E402
    FLUX_SCALE,
    MediaApplied,
    StoreProvenance,
)
from train_flux import media_report  # noqa: E402

from torchcell.metabolism.constraints import build_gem_tensors  # noqa: E402
from torchcell.metabolism.flux_layer import (  # noqa: E402
    genotype_key,
    parsimonious_reference_flux,
)
from torchcell.metabolism.parameters import (  # noqa: E402
    THERMO_EQUILIBRATOR_MIRROR,
    ThermoSource,
    resolve_reaction_delta_g,
)
from torchcell.metabolism.yeast_GEM import YeastGEM  # noqa: E402

Target = Literal["betaxanthin", "amino_acids"]
Variant = Literal["moma", "pfba", "sign10", "sign30"]
TARGETS: dict[str, dict[str, str]] = {
    "betaxanthin": {
        "source_stem": "YPD_bx",
        "medium": "YPD",
        "study": "031-betaxanthin-module",
    },
    "amino_acids": {
        "source_stem": "SM_gem",
        "medium": "SM",
        "study": "032-amino-acid-module",
    },
}
SIGN_THRESHOLD: dict[str, float] = {"sign10": 10.0, "sign30": 30.0}
LETHAL_GROWTH = 1e-6
BIOMASS_REACTION = "r_2111"


class TargetPhysicsProvenance(StoreProvenance):
    """The store record plus the program and the source it is paired against."""

    target: str
    variant: str
    source_store: str
    source_flux_sha256: str
    thermo: ThermoApplied | None
    wild_type_growth_unconstrained: float
    wild_type_growth_fraction: float
    column_layout: str


class GeneRow(BaseModel):
    """One gene's row of the summary."""

    gene: str
    growth: float
    growth_ratio: float
    source_growth_ratio: float
    lethal: bool
    converged: bool
    relative_l2_log1p_to_source: float


# worker state, inherited by fork
_S: torch.Tensor | None = None
_LB: np.ndarray | None = None
_UB: np.ndarray | None = None
_BIOMASS: int = -1
_V_WT: np.ndarray | None = None
_KNOCKED: dict[str, np.ndarray] = {}
_S_CSR: Any = None


def _knocked_box(gene: str) -> tuple[np.ndarray, np.ndarray]:
    assert _LB is not None and _UB is not None
    lb, ub = _LB.copy(), _UB.copy()
    lb[_KNOCKED[gene]] = 0.0
    ub[_KNOCKED[gene]] = 0.0
    return lb, ub


def solve_moma(gene: str) -> tuple[str, np.ndarray, float, bool]:
    r"""Linear MOMA: variables ``[v, p, n]``, ``v - p + n = v_wt``, ``p, n >= 0``."""
    assert _S_CSR is not None and _V_WT is not None
    lb, ub = _knocked_box(gene)
    m, r = _S_CSR.shape
    eye = identity(r, format="csr")
    a_eq = bmat([[_S_CSR, None, None], [eye, -eye, eye]], format="csr")
    b_eq = np.concatenate([np.zeros(m), _V_WT.astype(np.float64)])
    c = np.concatenate([np.zeros(r), np.ones(2 * r)])
    bounds = list(zip(lb.tolist(), ub.tolist(), strict=True)) + [(0.0, None)] * (2 * r)
    result = linprog(c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        return gene, np.zeros(r, dtype=np.float32), 0.0, False
    v = result.x[:r].astype(np.float32)
    growth = float(v[_BIOMASS])
    if growth < LETHAL_GROWTH:
        return gene, np.zeros(r, dtype=np.float32), growth, True
    return gene, v, growth, True


def solve_pfba(gene: str) -> tuple[str, np.ndarray, float, bool]:
    """Two-stage parsimonious FBA on the (direction-fixed) knocked-out box."""
    assert _S is not None
    lb, ub = _knocked_box(gene)
    try:
        record = parsimonious_reference_flux(
            _S, torch.from_numpy(lb).float(), torch.from_numpy(ub).float(), _BIOMASS
        )
    except ValueError:
        # The growth program did not solve under the knocked-out box: lethal, the zero
        # row, as the plain stores record it.
        return gene, np.zeros(lb.size, dtype=np.float32), 0.0, True
    growth = float(record.objective_value)
    if growth < LETHAL_GROWTH:
        return gene, np.zeros(lb.size, dtype=np.float32), growth, True
    return gene, record.v.numpy().astype(np.float32), growth, True


def main() -> None:
    global _S, _LB, _UB, _BIOMASS, _V_WT, _KNOCKED, _S_CSR
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=sorted(TARGETS))
    parser.add_argument(
        "--variant", required=True, choices=["moma", "pfba", "sign10", "sign30"]
    )
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--limit-genes", type=int, default=0, help="Smoke test.")
    parser.add_argument("--out-dir", default=TABLE_DIR)
    parser.add_argument("--results-dir", default="")
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("PYTHONHASHSEED must be 0, as for every store.")
    torch.set_num_threads(1)
    started = time.time()
    spec = TARGETS[args.target]
    results_dir = args.results_dir or osp.join(EXPERIMENT_ROOT, spec["study"], "results")

    source_path = osp.join(TABLE_DIR, f"ess_solve_encodings_{spec['source_stem']}.npz")
    source = np.load(source_path, allow_pickle=False)
    source_provenance = json.loads(str(source["provenance"]))
    genes = [str(g) for g in source["genes"]]
    if args.limit_genes:
        genes = genes[: args.limit_genes]

    gem_source = YeastGEM()
    if args.target == "betaxanthin":
        model, _demands, info = build_strain_model()
        media = info["media"]
        for r in model.reactions:
            r.lower_bound = max(r.lower_bound, -FLUX_SCALE)
            r.upper_bound = min(r.upper_bound, FLUX_SCALE)
        gem_label = source_provenance["gem"]
    else:
        model = gem_source.model
        media = media_report(model, spec["medium"])
        gem_label = osp.basename(gem_source.model_dir)
    gem = build_gem_tensors(model, gem_source.model_dir)
    rxn_ids = list(gem.rxn_ids)
    n_rxn = len(rxn_ids)
    assert rxn_ids == [str(r) for r in source["rxn_ids"]][:n_rxn], (
        "the model's reaction order differs from the source store's"
    )
    assert rxn_ids[int(gem.biomass_index)] == BIOMASS_REACTION
    _BIOMASS = int(gem.biomass_index)
    _S = gem.s
    s_dense = (gem.s.to_dense() if gem.s.is_sparse else gem.s).double().numpy()
    _S_CSR = csr_matrix(s_dense)
    base_lb = gem.lb.clamp(-FLUX_SCALE, FLUX_SCALE).double().numpy()
    base_ub = gem.ub.clamp(-FLUX_SCALE, FLUX_SCALE).double().numpy()
    gem_genes = {g.id for g in model.genes}
    assert all(g in gem_genes for g in genes), "a source-store gene is not in the model"

    # The source store's wild type: its first n_rxn columns are the growth vector (the
    # betaxanthin store appends the production vector and the total demand after them).
    source_wt = source["v_wild_type"].astype(np.float64)[:n_rxn]
    source_v = source["v_deletion"].astype(np.float32)[: len(genes), :n_rxn]
    unconstrained = parsimonious_reference_flux(
        gem.s,
        torch.from_numpy(base_lb).float(),
        torch.from_numpy(base_ub).float(),
        _BIOMASS,
    ).objective_value
    assert abs(unconstrained - source_provenance["wild_type_growth"]) < 1e-4, (
        unconstrained,
        source_provenance["wild_type_growth"],
    )
    print(
        f"{args.target}: {n_rxn} reactions, {len(genes)} genes, medium {spec['medium']}, "
        f"wild-type growth {unconstrained:.4f}; {time.time() - started:.0f}s",
        flush=True,
    )

    thermo: ThermoApplied | None = None
    lb, ub = base_lb, base_ub
    if args.variant in SIGN_THRESHOLD:
        table = resolve_reaction_delta_g(
            gem, osp.join(DATA_ROOT, THERMO_EQUILIBRATOR_MIRROR), ThermoSource.UNION_EXTERNAL
        )
        lb, ub, thermo = apply_sign_rule(
            base_lb,
            base_ub,
            table.delta_r_g0.numpy().astype(np.float64),
            table.mask.numpy().astype(bool),
            SIGN_THRESHOLD[args.variant],
            rxn_ids,
            source_wt,
            table,
        )
        print(
            f"thermo {thermo.source}: {thermo.n_reactions_with_energy}/{n_rxn} priced; "
            f"T {thermo.threshold_kj_per_mol} kJ/mol fixes {thermo.n_fixed_forward} "
            f"forward, {thermo.n_fixed_reverse} reverse; {thermo.n_bound_conflicts} bound "
            f"conflicts; {thermo.n_conflicts_with_plain_wild_type_flux} conflicts with "
            "the source wild-type flux",
            flush=True,
        )
    _LB, _UB = lb, ub

    rxn_index = {r: i for i, r in enumerate(rxn_ids)}
    for gene in genes:
        with model:
            knocked = knock_out_model_genes(model, [gene])
            _KNOCKED[gene] = np.array([rxn_index[r.id] for r in knocked], dtype=np.int64)

    if args.variant == "moma":
        _V_WT = source_wt.astype(np.float32)
        wt_v = _V_WT.copy()
        wt_growth = float(wt_v[_BIOMASS])
        solve = solve_moma
    else:
        try:
            record = parsimonious_reference_flux(
                gem.s, torch.from_numpy(lb).float(), torch.from_numpy(ub).float(), _BIOMASS
            )
        except ValueError as err:
            assert thermo is not None, "the plain box admits growth (asserted above)"
            # The direction rule admits no growth for the parent strain: a measured
            # outcome of the rule, recorded in the summary, and NO store is written (a
            # store of zero rows would be read as 1,155 lethal deletions).
            os.makedirs(results_dir, exist_ok=True)
            out = osp.join(
                results_dir, f"target_physics_store_{args.target}_{args.variant}.json"
            )
            summary = {
                "store": None,
                "target": args.target,
                "variant": args.variant,
                "source_store": source_path,
                "wild_type_infeasible": True,
                "wild_type_growth_unconstrained": float(unconstrained),
                "solver_message": str(err),
                "thermo": json.loads(thermo.model_dump_json()),
                "seconds": time.time() - started,
            }
            with open(out, "w") as fh:
                json.dump(summary, fh, indent=2)
            print(
                f"wild type infeasible under {args.variant}: no store written; "
                f"{thermo.n_conflicts_with_plain_wild_type_flux} fixed directions run "
                "against the source wild-type flux"
            )
            print(out)
            return
        wt_v = record.v.numpy().astype(np.float32)
        wt_growth = float(record.objective_value)
        solve = solve_pfba
    print(
        f"wild type under {args.variant}: growth {wt_growth:.4f} "
        f"({wt_growth / unconstrained:.3f} of unconstrained)",
        flush=True,
    )

    with Pool(args.processes) as pool:
        results = pool.map(solve, genes, chunksize=4)
    print(f"{len(genes)} solves, {time.time() - started:.0f}s", flush=True)
    order = {g: i for i, g in enumerate(genes)}
    v_deletion = np.zeros((len(genes), n_rxn), dtype=np.float32)
    growth = np.zeros(len(genes))
    converged = np.zeros(len(genes), dtype=bool)
    for gene, v, mu, ok in results:
        v_deletion[order[gene]] = v
        growth[order[gene]] = mu
        converged[order[gene]] = ok
    lethal = growth < LETHAL_GROWTH
    ratio = growth / wt_growth
    source_ratio = source["gene_flux_stats"][
        : len(genes), list(source["stat_names"]).index("deletion_growth_ratio")
    ].astype(np.float64)
    enc_new = np.log1p(np.abs(v_deletion.astype(np.float64)))
    enc_src = np.log1p(np.abs(source_v.astype(np.float64)))
    rel = np.linalg.norm(enc_new - enc_src, axis=1) / np.maximum(
        np.linalg.norm(enc_src, axis=1), 1e-12
    )
    rows = [
        GeneRow(
            gene=g,
            growth=float(growth[i]),
            growth_ratio=float(ratio[i]),
            source_growth_ratio=float(source_ratio[i]),
            lethal=bool(lethal[i]),
            converged=bool(converged[i]),
            relative_l2_log1p_to_source=float(rel[i]),
        )
        for i, g in enumerate(genes)
    ]

    method = (
        (
            "linear MOMA, min sum_j |v_j - v_wt_j| s.t. S v = 0 on the knocked-out box, "
            "one scipy HiGHS LP with |.| split into positive and negative parts, v_wt the "
            "source store's parsimonious wild-type growth vector"
        )
        if args.variant == "moma"
        else (
            "pFBA (parsimonious_reference_flux, two scipy HiGHS LPs) on the plain box, "
            "the essentiality stores' solver path"
        )
        if args.variant == "pfba"
        else (
            f"pFBA (parsimonious_reference_flux, two scipy HiGHS LPs) on the box with "
            f"reaction directions fixed by the standard reaction energies at "
            f"{SIGN_THRESHOLD[args.variant]:g} kJ/mol (eQuilibrator union-external route)"
        )
    ) + (
        "; the target model and gene list of the source store; gene deleted through "
        "cobra.manipulation.knock_out_model_genes; written by target_physics_store.py"
    )
    provenance = TargetPhysicsProvenance(
        gem=gem_label,
        growth_fraction=1.0,
        flux_scale=FLUX_SCALE,
        method=method,
        solver="scipy HiGHS (linprog)" if args.variant == "moma" else "pfba",
        n_nonoptimal_zero_flux=int((~converged).sum()),
        medium=spec["medium"],
        media_report=MediaApplied.model_validate(media),
        split_file=source_provenance["split_file"],
        wild_type_growth=wt_growth,
        wild_type_key=genotype_key([]),
        n_sets=len(genes) + 1,
        n_lethal_zero_flux=int(lethal.sum()),
        n_genes_outside_the_gem=0,
        background_genes=list(source_provenance["background_genes"]),
        python_hash_seed=os.environ.get("PYTHONHASHSEED"),
        flux_sha256=hashlib.sha256(v_deletion.tobytes()).hexdigest(),
        target=args.target,
        variant=args.variant,
        source_store=source_path,
        source_flux_sha256=source_provenance["flux_sha256"],
        thermo=thermo,
        wild_type_growth_unconstrained=float(unconstrained),
        wild_type_growth_fraction=float(wt_growth / unconstrained),
        column_layout="growth vector under the program (the source's first block only)",
    )
    os.makedirs(args.out_dir, exist_ok=True)
    path = osp.join(
        args.out_dir, f"ess_solve_encodings_{spec['source_stem']}_{args.variant}.npz"
    )
    assert not osp.exists(path), f"{path} exists; stores are never overwritten"
    np.savez(
        path,
        genes=np.array(genes),
        v_deletion=v_deletion,
        v_wild_type=wt_v.astype(np.float32),
        rxn_ids=np.array(rxn_ids),
        provenance=np.array(provenance.model_dump_json()),
        growth_ratio=ratio.astype(np.float32),
    )
    summary = {
        "store": path,
        "target": args.target,
        "variant": args.variant,
        "source_store": source_path,
        "n_genes": len(genes),
        "n_lethal": int(lethal.sum()),
        "n_lethal_source": int((source_ratio < LETHAL_GROWTH).sum()),
        "n_not_converged": int((~converged).sum()),
        "wild_type_growth": wt_growth,
        "wild_type_growth_unconstrained": float(unconstrained),
        "n_genes_ratio_moved_gt_0.01": int((np.abs(ratio - source_ratio) > 0.01).sum()),
        "n_genes_log1p_vector_moved": int((rel > 1e-6).sum()),
        "median_relative_l2_log1p_to_source": float(np.median(rel)),
        "median_relative_l2_log1p_to_source_viable": float(
            np.median(rel[~lethal]) if (~lethal).any() else float("nan")
        ),
        "thermo": None if thermo is None else json.loads(thermo.model_dump_json()),
        "provenance": json.loads(provenance.model_dump_json()),
        "rows": [r.model_dump() for r in rows],
        "seconds": time.time() - started,
        "processes": args.processes,
    }
    os.makedirs(results_dir, exist_ok=True)
    out = osp.join(results_dir, f"target_physics_store_{args.target}_{args.variant}.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(
        json.dumps({k: v for k, v in summary.items() if k not in ("rows", "provenance", "thermo")}, indent=2)
    )
    print(path)
    print(out)


if __name__ == "__main__":
    main()
