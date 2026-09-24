# experiments/031-betaxanthin-module/scripts/target_concentration_polytope_store.py
# [[experiments.031-betaxanthin-module.scripts.target_concentration_polytope_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-betaxanthin-module/scripts/target_concentration_polytope_store

r"""A probabilistic concentration store: the driving-force polytope sampled, not its point.

The max-min driving-force store (``target_driving_force_store.py``) represents a deletion
by ONE concentration vector, the l1-nearest to the center of the bounds among those that
reach the max-min optimum $B^*$. The second law does not single that point out: every
log-concentration vector within the physiological bounds whose carried reactions all run
downhill by at least the optimum is as consistent, and that set is a polytope (linear in
$\ln c$). This script samples it. Per deletion, over the source store's parsimonious flux
pattern and the same reaction set the point store constrains (priced, non-boundary,
non-transport, under the energy cap):

* stage 1 of the point program gives $B^*$ (and its stage-2 point, the sampler's start);
* the region is $\{x \in [\ln c_{lb}, \ln c_{ub}]^m : \text{force}_j(x) \ge B^* - s\}$ for
  every carried reaction $j$, with ``--slack`` $s$ kJ/mol so the region has volume even
  where several reactions bind at the optimum;
* metabolites that appear in no constrained reaction are free within the bounds and are
  drawn uniformly; the rest are sampled by coordinate hit-and-run (one metabolite per
  step, the chord along its axis clipped by the halfspaces and the box, the step uniform
  on the chord), with a burn-in from the stage-2 point and thinning between draws.

Two stores in the point store's shape, both read through ``ESS_SOLVE_TABLE``:

* ``ess_solve_encodings_<stem>_<variant>polymean.npz``: the mean concentration over draws
  (mM; fixed species 0) with the point store's three statistics and the count of
  constrained metabolites, a like-for-like replacement for the point;
* ``ess_solve_encodings_<stem>_<variant>polyspread.npz``: that mean with the per-metabolite
  standard deviation of $\ln c$ over draws appended, the width of the region along each
  metabolite, which one point cannot carry.

Run from the repo root::

    PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
        ~/miniconda3/envs/torchcell/bin/python \
        experiments/031-betaxanthin-module/scripts/target_concentration_polytope_store.py \
        --target betaxanthin --max-abs-dg0 100 --processes 8
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
from scipy.sparse import csr_matrix

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
TABLE_DIR = osp.join(DATA_ROOT, "data/torchcell/reference_flux")
for study in (
    "031-betaxanthin-module",
    "028-gene-essentiality",
    "027-betaxanthin-metabolic",
    "026-metabolism-flux",
):
    sys.path.insert(0, osp.join(EXPERIMENT_ROOT, study, "scripts"))
import target_driving_force_store as dfs  # noqa: E402
from bx_reference_flux_table import build_strain_model  # noqa: E402
from ess_reference_flux_table import FLUX_SCALE, MediaApplied  # noqa: E402
from train_flux import media_report  # noqa: E402

from torchcell.metabolism.constraints import build_gem_tensors  # noqa: E402
from torchcell.metabolism.flux_layer import genotype_key  # noqa: E402
from torchcell.metabolism.parameters import (  # noqa: E402
    THERMO_EQUILIBRATOR_MIRROR,
    ThermoSource,
    resolve_reaction_delta_g,
)
from torchcell.metabolism.yeast_GEM import YeastGEM  # noqa: E402

STAT_COLUMNS = dfs.STAT_COLUMNS + ["n_constrained_metabolites"]


class ConcentrationPolytopeProvenance(dfs.DrivingForceProvenance):
    """The driving-force record plus the sampler's settings."""

    samples: int
    burn_in: int
    thinning: int
    slack_kj_per_mol: float
    sampler_seed: int


_ARGS: argparse.Namespace | None = None


def sample_row(row: int) -> dict[str, Any]:
    """The point program, then hit-and-run over its polytope, for source row ``row``."""
    assert _ARGS is not None and dfs._ST is not None and dfs._DELTA is not None
    assert dfs._V is not None and dfs._PRICED is not None and dfs._EXCLUDED is not None
    _, x0, b_opt, n_active, n_binding, ok = dfs.solve_mdf(row)
    m = dfs._ST.shape[1]
    if not ok:
        return {"row": row, "ok": False, "b": b_opt, "n_active": n_active, "n_binding": n_binding}
    v = dfs._V[row].astype(np.float64)
    active = np.where((np.abs(v) > dfs.FLUX_TOL) & dfs._PRICED & ~dfs._EXCLUDED)[0]
    d = np.sign(v[active])
    a_all = csr_matrix(d[:, None]).multiply(dfs._ST[active]) * dfs.RT  # force_j = -(d dG0 + A x)
    b_all = -d * dfs._DELTA[active] - (b_opt - _ARGS.slack)  # A x <= b  <=>  force >= B* - s
    lb, ub = np.log(dfs.CONC_LB), np.log(dfs.CONC_UB)
    constrained = np.unique(a_all.nonzero()[1])
    free = np.setdiff1d(np.arange(m), constrained)
    a = a_all[:, constrained].toarray()
    # The chain is seeded from the region itself (its constrained reactions, their
    # directions and offsets), not from the row: two deletions with the same polytope then
    # draw the same points and read the same row, as two deletions with the same flux
    # vector read the same point. Seeding by row gave every no-effect deletion its own
    # sampling noise, the degeneracy the GLPK vertices had (MEASURED 2026-09-24: 1,062 of
    # 1,062 viable rows differed from the wild type's).
    digest = hashlib.sha256(
        active.tobytes() + d.tobytes() + np.round(b_all, 6).tobytes() + str(_ARGS.seed).encode()
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    x = x0[constrained].copy()
    x = np.clip(x, lb, ub)
    # The stage-2 point satisfies force >= B*, so it sits inside the slackened region.
    assert np.all(a @ x <= b_all + 1e-7), "start point outside its own polytope"
    n_c = constrained.size
    draws = np.empty((_ARGS.samples, m))
    total = _ARGS.burn_in + _ARGS.samples * _ARGS.thinning
    # Coordinate hit-and-run: one metabolite per step, the chord along its axis clipped
    # by the box and by every halfspace, the step uniform on the chord. A random
    # direction stalls when the point has coordinates on the box (the stage-2 point often
    # does), because the chord collapses to zero; one axis at a time does not.
    r = b_all - a @ x
    k = 0
    n_stuck = 0
    for step in range(total):
        i = int(rng.integers(n_c))
        lo, hi = lb - x[i], ub - x[i]
        col = a[:, i]
        cpos, cneg = col > 1e-12, col < -1e-12
        if cpos.any():
            hi = min(hi, np.min(r[cpos] / col[cpos]))
        if cneg.any():
            lo = max(lo, np.max(r[cneg] / col[cneg]))
        if hi - lo > 1e-12:
            lam = rng.uniform(lo, hi)
            x[i] += lam
            r -= lam * col
        else:
            n_stuck += 1
        if step >= _ARGS.burn_in and (step - _ARGS.burn_in) % _ARGS.thinning == 0:
            full = np.empty(m)
            full[constrained] = x
            full[free] = rng.uniform(lb, ub, size=free.size)
            draws[k] = full
            k += 1
            if k == _ARGS.samples:
                break
    assert k == _ARGS.samples
    return {
        "row": row,
        "ok": True,
        "b": b_opt,
        "n_active": n_active,
        "n_binding": n_binding,
        "n_constrained": int(n_c),
        "n_stuck_steps": int(n_stuck),
        "constrained": constrained,
        "mean_ln": draws.mean(axis=0),
        "sd_ln": draws.std(axis=0, ddof=1),
        "mean_mm": (np.exp(draws) * 1e3).mean(axis=0),
        "point_ln": x0,
    }


def main() -> None:
    global _ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=sorted(dfs.TARGETS))
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--max-abs-dg0", type=float, default=100.0)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--burn-in", type=int, default=20000)
    parser.add_argument("--thinning", type=int, default=2000)
    parser.add_argument("--slack", type=float, default=1.0, help="kJ/mol below B*.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-genes", type=int, default=0, help="Smoke test.")
    parser.add_argument("--out-dir", default=TABLE_DIR)
    parser.add_argument("--results-dir", default="")
    args = parser.parse_args()
    _ARGS = args
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("PYTHONHASHSEED must be 0, as for every store.")
    spec = dfs.TARGETS[args.target]
    results_dir = args.results_dir or osp.join(EXPERIMENT_ROOT, spec["study"], "results")
    started = time.time()

    source_path = osp.join(TABLE_DIR, f"ess_solve_encodings_{spec['source_stem']}.npz")
    source = np.load(source_path, allow_pickle=False)
    source_provenance = json.loads(str(source["provenance"]))
    genes = [str(g) for g in source["genes"]]
    if args.limit_genes:
        genes = genes[: args.limit_genes]

    # The same model, pricing and exclusions as the point store.
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
    excluded = np.array([r in boundary or r == dfs.BIOMASS_REACTION for r in rxn_ids])
    transport = np.array([dfs.is_transport(r) for r in model.reactions])
    excluded = excluded | transport
    capped = np.zeros(n_rxn, dtype=bool)
    if args.max_abs_dg0 > 0:
        capped = priced & (np.abs(delta) > args.max_abs_dg0)
        excluded = excluded | capped
    fixed = np.array(
        [model.metabolites.get_by_id(mid).name.split(" [")[0] in dfs.FIXED_SPECIES for mid in met_ids]
    )
    s_dense = (gem.s.to_dense() if gem.s.is_sparse else gem.s).double().numpy()
    dfs._ST = csr_matrix(s_dense.T[:, ~fixed])
    dfs._DELTA, dfs._PRICED, dfs._EXCLUDED = delta, priced, excluded
    m_var = int((~fixed).sum())
    dfs._X_MID = np.full(m_var, 0.5 * (np.log(dfs.CONC_LB) + np.log(dfs.CONC_UB)))
    v_source = source["v_deletion"].astype(np.float32)[:, :n_rxn]
    wt_source = source["v_wild_type"].astype(np.float32)[:n_rxn]
    dfs._V = np.vstack([wt_source[None, :], v_source[: len(genes)]])
    print(
        f"{args.target}: {int((priced & ~excluded).sum())} constrained reactions, {m_var} variable "
        f"metabolites, {len(genes)} genes; burn-in {args.burn_in}, thinning {args.thinning}, "
        f"{args.samples} draws, slack {args.slack} kJ/mol; {time.time() - started:.0f}s",
        flush=True,
    )
    wt = sample_row(0)
    assert wt["ok"], "the wild-type program did not solve"
    print(
        f"wild type: B* {wt['b']:+.3f} over {wt['n_active']} reactions, {wt['n_binding']} binding, "
        f"{wt['n_constrained']} constrained metabolites of {m_var}; median sd ln c over them "
        f"{np.median(wt['sd_ln'][wt['constrained']]):.2f}; "
        f"{time.time() - started:.0f}s",
        flush=True,
    )
    with Pool(args.processes) as pool:
        results = pool.map(sample_row, range(1, len(genes) + 1), chunksize=4)
    print(f"{len(genes)} rows sampled, {time.time() - started:.0f}s", flush=True)

    if "growth_ratio" in source.files:
        source_ratio = source["growth_ratio"].astype(float)
    else:
        stat_names = [str(n) for n in source["stat_names"]]
        source_ratio = source["gene_flux_stats"][:, stat_names.index("deletion_growth_ratio")]
    lethal = source_ratio[: len(genes)] < 1e-6
    n_met = len(met_ids)
    var_idx = np.where(~fixed)[0]

    def encode(res: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        mean = np.zeros(n_met + len(STAT_COLUMNS), dtype=np.float32)
        spread = np.zeros(n_met, dtype=np.float32)
        if res["ok"]:
            mean[var_idx] = res["mean_mm"]
            mean[n_met:] = [res["b"], res["n_active"], res["n_binding"], res["n_constrained"]]
            spread[var_idx] = res["sd_ln"]
        return mean, np.concatenate([mean, spread])

    by_row = {r["row"]: r for r in results}
    ordered = [by_row[i] for i in range(1, len(genes) + 1)]
    enc = [encode(r) for r in ordered]
    v_mean = np.vstack([e[0] for e in enc])
    v_spread = np.vstack([e[1] for e in enc])
    wt_mean, wt_spread = encode(wt)
    solved = np.array([r["ok"] for r in ordered])
    variant = ("mdf" if args.max_abs_dg0 <= 0 else f"mdfcap{args.max_abs_dg0:g}") + "poly"
    provenance = ConcentrationPolytopeProvenance(
        gem=source_provenance["gem"],
        growth_fraction=1.0,
        flux_scale=FLUX_SCALE,
        method=(
            "the max-min driving-force polytope sampled per deletion: stage 1 of the point "
            "program gives B*, the region is every log concentration within the bounds "
            "whose carried priced reactions all run downhill by at least B* minus the "
            "slack, metabolites in no constrained reaction are uniform in the bounds and "
            "the rest are drawn by hit-and-run from the stage-2 point; written by "
            "target_concentration_polytope_store.py"
        ),
        solver="scipy HiGHS (linprog) for B*; numpy hit-and-run",
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
        flux_sha256=hashlib.sha256(v_mean.tobytes()).hexdigest(),
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
        conc_lb_molar=dfs.CONC_LB,
        conc_ub_molar=dfs.CONC_UB,
        rt_kj_per_mol=dfs.RT,
        wild_type_mdf_kj_per_mol=float(wt["b"]),
        wild_type_n_active=int(wt["n_active"]),
        wild_type_n_binding=int(wt["n_binding"]),
        column_layout=(
            "mean metabolite concentration over draws (mM; fixed species 0) | mdf, n_active, "
            "n_binding, n_constrained_metabolites [| sd of ln c per metabolite in the spread store]"
        ),
        samples=args.samples,
        burn_in=args.burn_in,
        thinning=args.thinning,
        slack_kj_per_mol=args.slack,
        sampler_seed=args.seed,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    written = []
    for suffix, block, names in (
        ("mean", v_mean, met_ids + STAT_COLUMNS),
        ("spread", v_spread, met_ids + STAT_COLUMNS + [f"sd_ln_{m}" for m in met_ids]),
    ):
        path = osp.join(args.out_dir, f"ess_solve_encodings_{spec['out_stem']}_{variant}{suffix}.npz")
        assert not osp.exists(path), f"{path} exists; stores are never overwritten"
        np.savez(
            path,
            genes=np.array(genes),
            v_deletion=block,
            v_wild_type=wt_mean if suffix == "mean" else wt_spread,
            rxn_ids=np.array(names),
            provenance=np.array(provenance.model_dump_json()),
            growth_ratio=source_ratio[: len(genes)].astype(np.float32),
        )
        print(f"{path}: {block.shape}", flush=True)
        written.append(path)
    viable = solved & ~lethal
    sd_c = np.array([np.median(r["sd_ln"][r["constrained"]]) if r["ok"] else np.nan for r in ordered])
    move = np.array(
        [np.linalg.norm(r["mean_ln"] - r["point_ln"]) / max(np.linalg.norm(r["point_ln"]), 1e-12) if r["ok"] else np.nan for r in ordered]
    )
    summary = {
        "stores": written,
        "target": args.target,
        "variant": variant,
        "source_store": source_path,
        "n_genes": len(genes),
        "n_lethal": int(lethal.sum()),
        "n_solved": int(solved.sum()),
        "wild_type": {
            "mdf_kj_per_mol": float(wt["b"]),
            "n_active": int(wt["n_active"]),
            "n_binding": int(wt["n_binding"]),
            "n_constrained_metabolites": int(wt["n_constrained"]),
            "median_sd_ln_c_constrained": float(np.median(wt["sd_ln"][wt["constrained"]])),
            "median_sd_ln_c_free": float(np.median(np.delete(wt["sd_ln"], wt["constrained"]))) if wt["n_constrained"] < m_var else float("nan"),
        },
        "median_n_constrained_metabolites_viable": float(np.median([r["n_constrained"] for r in ordered if r["ok"]])),
        "median_sd_ln_c_constrained_viable": float(np.nanmedian(sd_c[viable])),
        "median_rel_distance_mean_to_point_viable": float(np.nanmedian(move[viable])),
        "median_stuck_step_fraction_viable": float(
            np.nanmedian([r["n_stuck_steps"] / (args.burn_in + args.samples * args.thinning) for r in ordered if r["ok"]])
        ),
        "n_mean_differs_from_wild_type": int(
            (np.abs(np.log1p(v_mean[viable, :n_met]) - np.log1p(wt_mean[:n_met])).max(axis=1) > 1e-6).sum()
        ),
        "samples": args.samples,
        "burn_in": args.burn_in,
        "thinning": args.thinning,
        "slack_kj_per_mol": args.slack,
        "seconds": time.time() - started,
        "processes": args.processes,
        "provenance": json.loads(provenance.model_dump_json()),
        "rows": [
            {"gene": g, "mdf_kj_per_mol": float(r["b"]), "n_active": int(r["n_active"]),
             "n_binding": int(r["n_binding"]), "n_constrained": int(r.get("n_constrained", 0)),
             "median_sd_ln_c": float(sd_c[i]), "lethal": bool(lethal[i]), "solved": bool(r["ok"])}
            for i, (g, r) in enumerate(zip(genes, ordered))
        ],
    }
    os.makedirs(results_dir, exist_ok=True)
    out = osp.join(results_dir, f"target_concentration_polytope_store_{args.target}_{variant}.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("rows", "provenance")}, indent=2))
    print(out)


if __name__ == "__main__":
    main()
