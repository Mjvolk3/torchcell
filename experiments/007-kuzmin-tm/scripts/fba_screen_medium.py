# experiments/007-kuzmin-tm/scripts/fba_screen_medium.py
# [[experiments.007-kuzmin-tm.scripts.fba_screen_medium]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/007-kuzmin-tm/scripts/fba_screen_medium
"""The Yeast9 FBA baseline rerun on the trigenic screens' own medium (panel g of FigS-yeast9-fba).

The frozen baseline (``results/cobra-fba-growth_backup_20250923_134447/``, run 2025-09-15
by ``targeted_fba_growth_fast.py``) solved every deletion set on yeast-GEM 9.0.2's default
medium as distributed: ammonium nitrogen, glucose at 1 mmol/gDW/h, no amino acid. The
screens were scored on SD/MSG -His/Arg/Lys/Ura + canavanine/thialysine/G418/clonNAT
(Kuzmin 2018 SI; Tong & Boone 2006 recipe #16), which ``torchcell.datamodels.media``
records as ``SGA_TM_SELECTION`` and ``torchcell.metabolism.media`` maps onto exchange
bounds as ``SGA_TM_SELECTION_FBA`` (glucose 3.3, monosodium glutamate at the 0.165
supplement rate, YNB vitamins, the SC supplement minus the four dropouts, agar and the
selection agents excluded by role).

Three arms, the same deletion sets (``unique_perturbations.json`` of the frozen run: 4,036
singles, 651,181 doubles, 332,313 triples), the same solver and 60 s limit, the same
fitness proxy and interaction formulas (imported from ``targeted_fba_growth_fast.py``):

  screen_medium    SGA_TM_SELECTION_FBA under the default UptakePolicy         (the rerun)
  sm_glucose_3.3   SM_FBA: ammonium, YNB vitamins, glucose 3.3, no amino acid  (control:
                   isolates the nitrogen source and supplements from the carbon rate)
  yeast9_default   model.medium as distributed, glucose 1                      (control:
                   reproduces the frozen run with today's cobrapy)

Per arm (``results/fba_screen_medium/<arm>/``): ``medium_bounds.json`` (the MediaBounds
record: every exchange opened, its magnitude and source string, and the resolution of
every ontology component), ``wt_growth.csv``, ``{singles,doubles,triples}_deletions.parquet``,
``{digenic,trigenic}_interactions.parquet``, ``fba_metadata.json`` (versions, runtime, the
sha256 of the perturbation file, solver statuses).

Run from the repo root (``gh_fba_screen_medium.slurm``):
    python experiments/007-kuzmin-tm/scripts/fba_screen_medium.py [--arms screen_medium,...] [--limit N]
"""

import argparse
import hashlib
import json
import os
import os.path as osp
import platform
import subprocess
import sys
from datetime import datetime

import cobra
import optlang
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from targeted_fba_growth_fast import (  # noqa: E402
    calculate_genetic_interactions,
    get_cpu_count,
    perform_targeted_fba_fast,
)

import torchcell  # noqa: E402
from torchcell.metabolism.media import (  # noqa: E402
    SGA_TM_SELECTION_FBA,
    SM_FBA,
    MediaBounds,
    UptakePolicy,
    media_to_bounds,
)
from torchcell.metabolism.yeast_GEM import YeastGEM  # noqa: E402

load_dotenv()
SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
EXP_DIR = osp.dirname(SCRIPT_DIR)
FROZEN = osp.join(EXP_DIR, "results", "cobra-fba-growth_backup_20250923_134447")
OUT = osp.join(EXP_DIR, "results", "fba_screen_medium")
PERTURBATIONS = osp.join(FROZEN, "unique_perturbations.json")
SOLVER_TIMEOUT_S = 60

ARMS: dict[str, str] = {
    "screen_medium": "SGA triple-mutant selection medium (SD/MSG -His/Arg/Lys/Ura), SGA_TM_SELECTION_FBA, default UptakePolicy",
    "sm_glucose_3.3": "SM (ammonium + YNB vitamins + glucose 3.3), SM_FBA, default UptakePolicy",
    "yeast9_default": "yeast-GEM 9.0.2 model.medium as distributed (glucose 1, ammonium), unchanged",
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    repo = osp.dirname(osp.dirname(osp.abspath(torchcell.__file__)))
    return subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()


def arm_model(arm: str, base: cobra.Model) -> tuple[cobra.Model, MediaBounds | None]:
    """A copy of the base model with the arm's medium applied, plus the bounds record."""
    model = base.copy()
    if arm == "yeast9_default":
        return model, None
    recipe = {"screen_medium": SGA_TM_SELECTION_FBA, "sm_glucose_3.3": SM_FBA}[arm]
    bounds = media_to_bounds(recipe, model, policy=UptakePolicy())
    if bounds.unresolved_names:
        raise SystemExit(f"{arm}: unresolved medium components {bounds.unresolved_names}")
    bounds.apply(model)
    return model, bounds


def medium_record(model: cobra.Model, bounds: MediaBounds | None) -> dict:
    """What the solver saw: the open exchanges of the model after the medium was applied."""
    open_exchanges = {
        r.id: {"metabolite": list(r.metabolites)[0].name, "uptake_bound": -r.lower_bound}
        for r in model.exchanges
        if r.lower_bound < 0
    }
    return {
        "n_open_exchanges": len(open_exchanges),
        "open_exchanges": open_exchanges,
        "media_bounds": None if bounds is None else json.loads(bounds.model_dump_json()),
    }


def run_arm(arm: str, base: cobra.Model, perturbations: dict, n_processes: int, limit: int | None) -> None:
    out_dir = osp.join(OUT, arm)
    os.makedirs(out_dir, exist_ok=True)
    model, bounds = arm_model(arm, base)
    sol = model.optimize()
    assert sol.status == "optimal", (arm, sol.status)
    wt_growth = float(sol.objective_value)
    fluxes = sol.fluxes
    print(f"\n=== arm {arm}: {ARMS[arm]}")
    print(f"WT growth {wt_growth:.4f}; open exchanges {sum(r.lower_bound < 0 for r in model.exchanges)}; "
          f"glucose {fluxes['r_1714']:.3f}, oxygen {fluxes['r_1992']:.3f}, ammonium {fluxes['r_1654']:.3f}, "
          f"glutamate {fluxes['r_1889']:.3f}", flush=True)
    json.dump(medium_record(model, bounds), open(osp.join(out_dir, "medium_bounds.json"), "w"), indent=2)
    pd.DataFrame({"genotype": ["WT"], "growth": [wt_growth], "fitness": [1.0],
                  "glucose_uptake": [-fluxes["r_1714"]], "oxygen_uptake": [-fluxes["r_1992"]],
                  "ammonium_exchange": [fluxes["r_1654"]], "glutamate_exchange": [fluxes["r_1889"]]}
                 ).to_csv(osp.join(out_dir, "wt_growth.csv"), index=False)

    perts = perturbations if limit is None else {k: v[:limit] for k, v in perturbations.items()}
    start = datetime.now()
    files = perform_targeted_fba_fast(model=model, perturbations=perts, output_dir=out_dir,
                                      wt_growth=wt_growth, processes=n_processes, write_buffer_size=10000)
    fba_seconds = (datetime.now() - start).total_seconds()
    singles = pd.read_parquet(files["singles"])
    doubles = pd.read_parquet(files["doubles"])
    triples = pd.read_parquet(files["triples"])
    digenic, trigenic = calculate_genetic_interactions(singles, doubles, triples, wt_growth)
    digenic.to_parquet(osp.join(out_dir, "digenic_interactions.parquet"), index=False)
    trigenic.to_parquet(osp.join(out_dir, "trigenic_interactions.parquet"), index=False)

    statuses = pd.concat([singles.status, doubles.status, triples.status]).value_counts().to_dict()
    meta = {
        "arm": arm,
        "description": ARMS[arm],
        "timestamp": datetime.now().isoformat(),
        "host": platform.node(),
        "torchcell_commit": git_commit(),
        "cobra_version": cobra.__version__,
        "optlang_version": optlang.__version__,
        "solver": model.solver.interface.__name__,
        "solver_timeout_s": SOLVER_TIMEOUT_S,
        "model_id": model.id,
        "wt_growth": wt_growth,
        "perturbations_file": osp.relpath(PERTURBATIONS, EXP_DIR),
        "perturbations_sha256": sha256(PERTURBATIONS),
        "limit": limit,
        "n_singles": len(singles),
        "n_doubles": len(doubles),
        "n_triples": len(triples),
        "n_processes": n_processes,
        "solver_status_counts": {str(k): int(v) for k, v in statuses.items()},
        "fba_runtime_seconds": fba_seconds,
        "total_runtime_seconds": (datetime.now() - start).total_seconds(),
        "lethal_fraction": {
            "singles": float((singles.fitness < 0.01).mean()),
            "doubles": float((doubles.fitness < 0.01).mean()),
            "triples": float((triples.fitness < 0.01).mean()),
        },
        "tau_abs_above_1e-3": int((trigenic.tau.abs() > 1e-3).sum()),
    }
    json.dump(meta, open(osp.join(out_dir, "fba_metadata.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in meta.items() if k != "description"}, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS), help="comma-separated subset of " + ",".join(ARMS))
    ap.add_argument("--limit", type=int, default=None, help="first N deletion sets of each order (smoke test)")
    args = ap.parse_args()
    arms = args.arms.split(",")
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}")
    print(f"torchcell {torchcell.__file__} @ {git_commit()[:8]}; cobra {cobra.__version__}; optlang {optlang.__version__}")
    perturbations = json.load(open(PERTURBATIONS))
    print({k: len(v) for k, v in perturbations.items()})
    base = YeastGEM().model
    n_processes = get_cpu_count()
    for arm in arms:
        run_arm(arm, base, perturbations, n_processes, args.limit)


if __name__ == "__main__":
    main()
