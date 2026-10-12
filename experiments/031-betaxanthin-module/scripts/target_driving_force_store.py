# experiments/031-betaxanthin-module/scripts/target_driving_force_store.py
# [[experiments.031-betaxanthin-module.scripts.target_driving_force_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-betaxanthin-module/scripts/target_driving_force_store

r"""A metabolite-concentration store: the max-min driving force of each deletion's flux.

The store affine reads fluxes, the right-hand side of the stoichiometric map. The
concentrations are the other side: with chemical potentials
:math:`\mu = \mu^\circ + RT\ln c` the driving force of reaction :math:`j` is
:math:`-\Delta_r G'_j = -(\Delta_r G'^\circ_j + RT\,(S^\top \ln c)_j)`, and the second law
requires it to be positive wherever flux flows. Fixing directions from the standard
energies alone admits no growth on any of these models (``target_physics_store.py``,
2026-09-23); with the concentrations free the question becomes a linear program per
genotype, the max-min driving force of Noor 2014: given the deletion's parsimonious flux
pattern, find the log concentrations, within physiological bounds, that make the SMALLEST
driving force over the carried, priced reactions as large as possible. The optimum
:math:`B` is a thermodynamic verdict on that flux pattern (negative means some carried
reaction must run uphill within the bounds), and the concentration vector is the
metabolite-side representation of the strain.

Per gene, from the source store's HiGHS parsimonious vector: the active set is every
reaction carrying more than ``FLUX_TOL`` in absolute value that is priced by the
eQuilibrator union-external table, not a boundary reaction and not the biomass
pseudo-reaction; stage 1 maximizes :math:`B`; stage 2 holds :math:`B` at its optimum and
takes the concentrations nearest (in :math:`\ell_1` of log concentration) to the
geometric center of the bounds, so the vector is determinate. Water and protons are fixed
at unit activity (the standard energies are transformed at the compartment pH). Every
other metabolite, extracellular ones included, ranges over ``CONC_LB`` to ``CONC_UB``.
A lethal deletion (zero flux) is the zero row, as every store has it.

Writes ``ess_solve_encodings_<stem>_mdf.npz`` in the solve store's shape: one column
per metabolite (mM; the fixed species 0) followed by ``mdf_kj_per_mol``, ``n_active``,
``n_binding``, and ``results/target_driving_force_store_<target>.json`` with the per-gene
verdicts. Run from the repo root::

    PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
        ~/miniconda3/envs/torchcell/bin/python \
        experiments/031-betaxanthin-module/scripts/target_driving_force_store.py \
        --target betaxanthin --processes 8
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
from typing import Any

import numpy as np
from dotenv import load_dotenv
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, identity

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
TABLE_DIR = osp.join(DATA_ROOT, "data/torchcell/reference_flux")
for study in ("028-gene-essentiality", "027-betaxanthin-metabolic", "026-metabolism-flux"):
    sys.path.insert(0, osp.join(EXPERIMENT_ROOT, study, "scripts"))
from bx_reference_flux_table import build_strain_model  # noqa: E402
from ess_reference_flux_table import (  # noqa: E402
    FLUX_SCALE,
    MediaApplied,
    StoreProvenance,
)
from train_flux import media_report  # noqa: E402

from torchcell.metabolism.constraints import build_gem_tensors  # noqa: E402
from torchcell.metabolism.flux_layer import genotype_key  # noqa: E402
from torchcell.metabolism.parameters import (  # noqa: E402
    THERMO_EQUILIBRATOR_MIRROR,
    ThermoSource,
    resolve_reaction_delta_g,
)
from torchcell.metabolism.yeast_GEM import YeastGEM  # noqa: E402

TARGETS: dict[str, dict[str, str]] = {
    "betaxanthin": {
        "source_stem": "YPD_bx_pfba", "out_stem": "YPD_bx", "medium": "YPD",
        "study": "031-betaxanthin-module",
    },
    "amino_acids": {
        "source_stem": "SM_gem", "out_stem": "SM_gem", "medium": "SM",
        "study": "032-amino-acid-module",
    },
}
#: Gas constant times 298.15 K, kJ/mol; the eQuilibrator energies are at 298.15 K.
RT = 8.314462618e-3 * 298.15
#: Physiological concentration bounds, M (Noor 2014's max-min driving force defaults).
CONC_LB, CONC_UB = 1e-6, 1e-2
FLUX_TOL = 1e-5
BIOMASS_REACTION = "r_2111"
FIXED_SPECIES = ("H2O", "H+")
STAT_COLUMNS = ["mdf_kj_per_mol", "n_active", "n_binding"]


class DrivingForceProvenance(StoreProvenance):
    """The store record plus the program's settings and the wild-type verdict."""

    target: str
    source_store: str
    thermo_source: str
    n_reactions_priced: int
    n_reactions_transport_excluded: int
    max_abs_dg0_kj_per_mol: float
    n_reactions_over_cap: int
    n_reactions_constrained: int
    n_metabolites_variable: int
    n_metabolites_fixed: int
    conc_lb_molar: float
    conc_ub_molar: float
    rt_kj_per_mol: float
    wild_type_mdf_kj_per_mol: float
    wild_type_n_active: int
    wild_type_n_binding: int
    column_layout: str


def is_transport(reaction: Any) -> bool:
    """Same species (by name) on both sides, water and protons aside, or a proton pump."""
    from collections import Counter

    def base(met: Any) -> str:
        return str(met.name).split(" [")[0]

    lhs = Counter(base(m) for m, c in reaction.metabolites.items() if c < 0 and base(m) not in FIXED_SPECIES)
    rhs = Counter(base(m) for m, c in reaction.metabolites.items() if c > 0 and base(m) not in FIXED_SPECIES)
    if len(lhs) > 0 and lhs == rhs:
        return True
    # Proton translocation (ATP synthase, the respiratory complexes, symporters): H+ on
    # both sides in different compartments. The driving force is the proton-motive force,
    # which the standard-energy table does not carry, so these are not constrained either.
    protons = {m.compartment for m, c in reaction.metabolites.items() if base(m) == "H+" and c < 0}
    protons_out = {m.compartment for m, c in reaction.metabolites.items() if base(m) == "H+" and c > 0}
    return len(protons) > 0 and len(protons_out) > 0 and protons != protons_out


# worker state, inherited by fork
_ST: Any = None  # S^T restricted to variable metabolites, csr [r, m_var]
_DELTA: np.ndarray | None = None
_PRICED: np.ndarray | None = None
_EXCLUDED: np.ndarray | None = None
_X_MID: np.ndarray | None = None
_V: np.ndarray | None = None


def solve_mdf(row: int) -> tuple[int, np.ndarray, float, int, int, bool]:
    """``(row, ln c, B, n_active, n_binding, ok)`` for the source store's row ``row``."""
    assert _ST is not None and _DELTA is not None and _PRICED is not None
    assert _EXCLUDED is not None and _X_MID is not None and _V is not None
    v = _V[row].astype(np.float64)
    m = _ST.shape[1]
    active = np.where((np.abs(v) > FLUX_TOL) & _PRICED & ~_EXCLUDED)[0]
    if active.size == 0:
        return row, np.zeros(m), 0.0, 0, 0, False
    d = np.sign(v[active])
    # d_j RT (S^T x)_j + B <= -d_j dG0_j
    a_rows = _ST[active]
    a_ub = hstack([csr_matrix(d[:, None]) .multiply(a_rows) * RT, csr_matrix(np.ones((active.size, 1)))], format="csr")
    b_ub = -d * _DELTA[active]
    bounds = [(np.log(CONC_LB), np.log(CONC_UB))] * m + [(None, None)]
    c = np.zeros(m + 1)
    c[-1] = -1.0
    s1 = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not s1.success:
        return row, np.zeros(m), float("nan"), int(active.size), 0, False
    b_opt = float(s1.x[-1])
    # Stage 2: B held at its optimum, x nearest the center of the bounds in l1.
    eye = identity(m, format="csr")
    a_eq = hstack([eye, -eye, eye], format="csr")  # x - p + n = x_mid
    a_ub2 = hstack([a_ub[:, :m], csr_matrix((active.size, 2 * m))], format="csr")
    b_ub2 = b_ub - (b_opt - 1e-6)
    c2 = np.concatenate([np.zeros(m), np.ones(2 * m)])
    bounds2 = [(np.log(CONC_LB), np.log(CONC_UB))] * m + [(0.0, None)] * (2 * m)
    s2 = linprog(
        c2, A_ub=a_ub2, b_ub=b_ub2, A_eq=a_eq, b_eq=_X_MID, bounds=bounds2, method="highs"
    )
    x = s2.x[:m] if s2.success else s1.x[:m]
    force = -d * (_DELTA[active] + RT * (a_rows @ x))
    n_binding = int((force <= b_opt + 1e-4).sum())
    return row, x, b_opt, int(active.size), n_binding, True


def main() -> None:
    global _ST, _DELTA, _PRICED, _EXCLUDED, _X_MID, _V
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=sorted(TARGETS))
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument(
        "--max-abs-dg0",
        type=float,
        default=0.0,
        help=(
            "Constrain only reactions whose |standard energy| is at most this (kJ/mol); 0 "
            "constrains every priced reaction. MEASURED 2026-09-23 on yeast-GEM 9.0.2: 728 "
            "priced non-transport reactions exceed 150 kJ/mol, lipid and cofactor species "
            "mostly, and the wild type's max-min driving force is set by them at -281; the "
            "capped variant (suffix mdfcap<N>) reads the second law over the rest."
        ),
    )
    parser.add_argument("--limit-genes", type=int, default=0, help="Smoke test.")
    parser.add_argument("--out-dir", default=TABLE_DIR)
    parser.add_argument("--results-dir", default="")
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("PYTHONHASHSEED must be 0, as for every store.")
    spec = TARGETS[args.target]
    results_dir = args.results_dir or osp.join(EXPERIMENT_ROOT, spec["study"], "results")
    started = time.time()

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
    else:
        model = gem_source.model
        media = media_report(model, spec["medium"])
    gem = build_gem_tensors(model, gem_source.model_dir)
    rxn_ids = list(gem.rxn_ids)
    n_rxn = len(rxn_ids)
    assert rxn_ids == [str(r) for r in source["rxn_ids"]][:n_rxn]
    met_ids = list(gem.met_ids)
    table = resolve_reaction_delta_g(
        gem, osp.join(DATA_ROOT, THERMO_EQUILIBRATOR_MIRROR), ThermoSource.UNION_EXTERNAL
    )
    delta = table.delta_r_g0.numpy().astype(np.float64)
    priced = table.mask.numpy().astype(bool)
    boundary = {r.id for r in model.reactions if r.boundary}
    excluded = np.array([r in boundary or r == BIOMASS_REACTION for r in rxn_ids])
    # A transport reaction moves the same species between compartments; its energy is the
    # membrane potential and the pH difference, which the standard-energy table does not
    # carry (MEASURED: 505 of 1,041 priced transports exceed 50 kJ/mol), so it is never
    # constrained.
    transport = np.array([is_transport(r) for r in model.reactions])
    excluded = excluded | transport
    capped = np.zeros(n_rxn, dtype=bool)
    if args.max_abs_dg0 > 0:
        capped = priced & (np.abs(delta) > args.max_abs_dg0)
        excluded = excluded | capped
    fixed = np.array(
        [model.metabolites.get_by_id(mid).name.split(" [")[0] in FIXED_SPECIES for mid in met_ids]
    )
    s_dense = (gem.s.to_dense() if gem.s.is_sparse else gem.s).double().numpy()
    _ST = csr_matrix(s_dense.T[:, ~fixed])
    _DELTA, _PRICED, _EXCLUDED = delta, priced, excluded
    m_var = int((~fixed).sum())
    _X_MID = np.full(m_var, 0.5 * (np.log(CONC_LB) + np.log(CONC_UB)))
    v_source = source["v_deletion"].astype(np.float32)[:, :n_rxn]
    wt_source = source["v_wild_type"].astype(np.float32)[:n_rxn]
    _V = np.vstack([wt_source[None, :], v_source[: len(genes)]])
    print(
        f"{args.target}: {n_rxn} reactions ({int(priced.sum())} priced, "
        f"{int(transport.sum())} transport, {int(capped.sum())} over the cap, "
        f"{int((priced & ~excluded).sum())} constrained), {len(met_ids)} metabolites "
        f"({m_var} variable, {int(fixed.sum())} fixed), {len(genes)} genes; "
        f"{time.time() - started:.0f}s",
        flush=True,
    )
    _, x_wt, b_wt, n_wt, nb_wt, ok_wt = solve_mdf(0)
    assert ok_wt, "the wild-type program did not solve"
    wt_active = np.where((np.abs(_V[0]) > FLUX_TOL) & priced & ~excluded)[0]
    wt_force = -np.sign(_V[0][wt_active]) * (delta[wt_active] + RT * (_ST[wt_active] @ x_wt))
    wt_binding = [
        {
            "reaction": rxn_ids[j],
            "name": model.reactions.get_by_id(rxn_ids[j]).name,
            "delta_r_g0_kj_per_mol": float(delta[j]),
            "flux": float(_V[0][j]),
            "driving_force_kj_per_mol": float(f),
        }
        for j, f in sorted(zip(wt_active.tolist(), wt_force.tolist()), key=lambda t: t[1])[:15]
    ]
    print(
        f"wild type: MDF {b_wt:+.3f} kJ/mol over {n_wt} carried priced reactions, "
        f"{nb_wt} binding; {time.time() - started:.0f}s",
        flush=True,
    )

    with Pool(args.processes) as pool:
        results = pool.map(solve_mdf, range(1, len(genes) + 1), chunksize=4)
    print(f"{len(genes)} solves, {time.time() - started:.0f}s", flush=True)
    conc = np.zeros((len(genes), m_var))
    mdf = np.full(len(genes), np.nan)
    n_active = np.zeros(len(genes), dtype=int)
    n_binding = np.zeros(len(genes), dtype=int)
    solved = np.zeros(len(genes), dtype=bool)
    for row, x, b, na, nb, ok in results:
        conc[row - 1], mdf[row - 1], n_active[row - 1], n_binding[row - 1], solved[row - 1] = x, b, na, nb, ok
    # Lethal by the source's own growth ratio: the HiGHS stores keep a zero-growth
    # knockout's flux vector (exchanges and cycles), so a zero row is not the test.
    if "growth_ratio" in source.files:
        source_ratio = source["growth_ratio"].astype(float)
    else:
        stat_names = [str(n) for n in source["stat_names"]]
        source_ratio = source["gene_flux_stats"][:, stat_names.index("deletion_growth_ratio")]
    lethal = source_ratio[: len(genes)] < 1e-6

    def encode(x: np.ndarray, b: float, na: int, nb: int, ok: bool) -> np.ndarray:
        out = np.zeros(len(met_ids) + len(STAT_COLUMNS), dtype=np.float32)
        if ok:
            out[np.where(~fixed)[0]] = np.exp(x) * 1e3  # mM
            out[len(met_ids):] = [b, na, nb]
        return out

    v_deletion = np.vstack(
        [encode(conc[i], mdf[i], n_active[i], n_binding[i], solved[i]) for i in range(len(genes))]
    )
    v_wild_type = encode(x_wt, b_wt, n_wt, nb_wt, True)
    provenance = DrivingForceProvenance(
        gem=source_provenance["gem"],
        growth_fraction=1.0,
        flux_scale=FLUX_SCALE,
        method=(
            "max-min driving force (Noor 2014) per deletion over the source store's "
            "parsimonious flux pattern: stage 1 maximizes the smallest driving force over "
            "the carried, priced, non-boundary reactions with log concentrations free "
            "within physiological bounds; stage 2 holds it and takes the l1-nearest "
            "concentrations to the center of the bounds; two scipy HiGHS LPs per gene; "
            "written by target_driving_force_store.py"
        ),
        solver="scipy HiGHS (linprog)",
        n_nonoptimal_zero_flux=int((~solved & ~lethal).sum()),
        medium=spec["medium"],
        media_report=MediaApplied.model_validate(media),
        split_file=source_provenance["split_file"],
        wild_type_growth=source_provenance["wild_type_growth"],
        wild_type_key=genotype_key([]),
        n_sets=len(genes) + 1,
        n_lethal_zero_flux=int(lethal.sum()),
        n_genes_outside_the_gem=0,
        background_genes=list(source_provenance["background_genes"]),
        python_hash_seed=os.environ.get("PYTHONHASHSEED"),
        flux_sha256=hashlib.sha256(v_deletion.tobytes()).hexdigest(),
        target=args.target,
        source_store=source_path,
        thermo_source=str(table.source),
        n_reactions_priced=int(priced.sum()),
        n_reactions_transport_excluded=int((transport & priced).sum()),
        max_abs_dg0_kj_per_mol=args.max_abs_dg0,
        n_reactions_over_cap=int(capped.sum()),
        n_reactions_constrained=int((priced & ~excluded).sum()),
        n_metabolites_variable=m_var,
        n_metabolites_fixed=int(fixed.sum()),
        conc_lb_molar=CONC_LB,
        conc_ub_molar=CONC_UB,
        rt_kj_per_mol=RT,
        wild_type_mdf_kj_per_mol=b_wt,
        wild_type_n_active=n_wt,
        wild_type_n_binding=nb_wt,
        column_layout="metabolite concentrations (mM; fixed species 0) | mdf, n_active, n_binding",
    )
    os.makedirs(args.out_dir, exist_ok=True)
    variant = "mdf" if args.max_abs_dg0 <= 0 else f"mdfcap{args.max_abs_dg0:g}"
    path = osp.join(args.out_dir, f"ess_solve_encodings_{spec['out_stem']}_{variant}.npz")
    assert not osp.exists(path), f"{path} exists; stores are never overwritten"
    np.savez(
        path,
        genes=np.array(genes),
        v_deletion=v_deletion,
        v_wild_type=v_wild_type,
        rxn_ids=np.array(met_ids + STAT_COLUMNS),
        provenance=np.array(provenance.model_dump_json()),
        mdf=mdf.astype(np.float32),
    )
    viable = solved & ~lethal
    summary = {
        "store": path,
        "target": args.target,
        "variant": variant,
        "source_store": source_path,
        "wild_type_most_uphill": wt_binding,
        "n_genes": len(genes),
        "n_lethal": int(lethal.sum()),
        "n_solved": int(solved.sum()),
        "wild_type": {"mdf_kj_per_mol": b_wt, "n_active": n_wt, "n_binding": nb_wt},
        "n_mdf_negative": int((mdf[viable] < 0).sum()),
        "n_mdf_moved_gt_0.1": int((np.abs(mdf[viable] - b_wt) > 0.1).sum()),
        "mdf_quantiles_viable": {
            q: float(np.quantile(mdf[viable], float(q))) for q in ("0.0", "0.1", "0.5", "0.9", "1.0")
        },
        "median_rel_l2_log1p_conc_to_wild_type": float(
            np.median(
                np.linalg.norm(
                    np.log1p(v_deletion[viable, : len(met_ids)]) - np.log1p(v_wild_type[: len(met_ids)]),
                    axis=1,
                )
                / max(float(np.linalg.norm(np.log1p(v_wild_type[: len(met_ids)]))), 1e-12)
            )
        ),
        "n_conc_moved": int(
            (
                np.abs(
                    np.log1p(v_deletion[viable, : len(met_ids)]) - np.log1p(v_wild_type[: len(met_ids)])
                ).max(axis=1)
                > 1e-6
            ).sum()
        ),
        "seconds": time.time() - started,
        "processes": args.processes,
        "provenance": json.loads(provenance.model_dump_json()),
        "rows": [
            {"gene": g, "mdf_kj_per_mol": float(mdf[i]), "n_active": int(n_active[i]),
             "n_binding": int(n_binding[i]), "lethal": bool(lethal[i]), "solved": bool(solved[i])}
            for i, g in enumerate(genes)
        ],
    }
    os.makedirs(results_dir, exist_ok=True)
    out = osp.join(results_dir, f"target_driving_force_store_{args.target}_{variant}.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("rows", "provenance", "wild_type_most_uphill")}, indent=2))
    for b in wt_binding[:6]:
        print(f"  uphill {b['reaction']} {b['driving_force_kj_per_mol']:+.1f} dG0 {b['delta_r_g0_kj_per_mol']:+.1f} v {b['flux']:+.3f} {b['name'][:50]}")
    print(path)
    print(out)


if __name__ == "__main__":
    main()
