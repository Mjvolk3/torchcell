# experiments/048-pputida-isoprenol-cgt/scripts/carruthers_campaign_audit.py
# [[experiments.048-pputida-isoprenol-cgt.scripts.carruthers_campaign_audit]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/048-pputida-isoprenol-cgt/scripts/carruthers_campaign_audit
"""Audit the Carruthers 2025 P. putida isoprenol CRISPRi campaign.

Regenerates every number quoted in
[[experiments.048-pputida-isoprenol-cgt.p-putida-cgt-mvp-goal]]:

1. Campaign shape: rows, cycles, designs, targets, replicate structure.
2. The cycle 2 / cycle 3 vector swap: shared designs, paired agreement, and the
   within-cycle split-half replicate ceiling that the agreement is read against.
3. Provenance: the shared analysis CSV's titers against the published SI, which
   is what licenses the loader to source from the mirror instead of the CSV.
4. The metabolic model diff, iJN1463 against the isoprenol-augmented model.

Writes JSON + CSV into experiments/048-pputida-isoprenol-cgt/results/.
"""

import json
import os
import os.path as osp
from itertools import combinations

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

RESULTS = osp.join(EXPERIMENT_ROOT, "048-pputida-isoprenol-cgt", "results")

# The folder shared by Juan Camilo (LBNL/JBEI), Genesis collaboration, received
# 2026.11.17. Not a mirror path: the audit's job is to prove which parts of it
# are reproducible from the mirror and which are owed a raw-mirror entry.
SHARED = os.environ.get("CARRUTHERS_SHARED_DIR", "/tmp/screenshots/04_Data")

MIRROR_LIB = osp.join(
    DATA_ROOT, "torchcell-library", "carruthersAutomationMachineLearning2025"
)

TARGET_COLS = [f"target_{i}" for i in range(1, 5)]


def design_key(row: pd.Series) -> tuple[str, ...]:
    """The sorted set of CRISPRi gene targets, which identifies a design."""
    return tuple(sorted(t for t in (row[c] for c in TARGET_COLS) if isinstance(t, str)))


def load_campaign() -> pd.DataFrame:
    df = pd.read_csv(osp.join(SHARED, "Carruthers_etal_2025.csv"), index_col=0)
    df["design"] = df.apply(design_key, axis=1)
    return df


def campaign_shape(df: pd.DataFrame) -> dict:
    per_cell = df.groupby(["cycle", "design"]).size()
    return {
        "rows": int(len(df)),
        "cycles": sorted(int(c) for c in df.cycle.unique()),
        "distinct_designs": int(df.design.nunique()),
        "distinct_targets": int(
            len({t for t in df[TARGET_COLS].to_numpy().ravel() if isinstance(t, str)})
        ),
        "design_cycle_cells": int(len(per_cell)),
        "replicate_counts": {
            int(k): int(v) for k, v in per_cell.value_counts().items()
        },
        "controls_per_cycle": {
            int(c): int(n) for c, n in df[df.is_control].groupby("cycle").size().items()
        },
        "sgrna_counts_per_cycle": {
            f"c{int(c)}_n{int(n)}": int(k)
            for (c, n), k in df.groupby(["cycle", "number_of_grna"]).size().items()
        },
    }


def design_reuse(df: pd.DataFrame) -> pd.DataFrame:
    sets = {c: set(g.design) for c, g in df.groupby("cycle")}
    rows = []
    for a, b in combinations(sorted(sets), 2):
        rows.append(
            {
                "cycle_a": int(a),
                "cycle_b": int(b),
                "n_a": len(sets[a]),
                "n_b": len(sets[b]),
                "shared": len(sets[a] & sets[b]),
            }
        )
    return pd.DataFrame(rows).sort_values("shared", ascending=False)


def split_half_reliability(
    df: pd.DataFrame, cycle: int, iters: int = 400, seed: int = 0
) -> dict:
    """Spearman-Brown corrected split-half reliability of design mean titer.

    This is the replicate ceiling. Any cross-cycle agreement has to be read
    against it, otherwise a loss of correlation cannot be attributed.
    """
    rng = np.random.default_rng(seed)
    d = df[df.cycle == cycle]
    vals = []
    n_designs = 0
    for _ in range(iters):
        a, b = [], []
        for _, sub in d.groupby("design"):
            if len(sub) < 2:
                continue
            idx = rng.permutation(len(sub))
            half = len(sub) // 2
            a.append(sub.isoprenol.values[idx[:half]].mean())
            b.append(sub.isoprenol.values[idx[half:]].mean())
        n_designs = len(a)
        r = np.corrcoef(a, b)[0, 1]
        vals.append(2 * r / (1 + r))
    vals = np.asarray(vals)
    return {
        "cycle": cycle,
        "reliability": float(vals.mean()),
        "ci90": [float(np.percentile(vals, 5)), float(np.percentile(vals, 95))],
        "n_designs": int(n_designs),
        "iters": iters,
    }


def vector_swap(df: pd.DataFrame, cycle_a: int = 2, cycle_b: int = 3) -> dict:
    """Paired agreement across the pIY989 -> pDBTL3-6_Vector swap."""
    per = df.groupby(["cycle", "design"]).isoprenol.mean().rename("iso").reset_index()
    piv = per.pivot(index="design", columns="cycle", values="iso")
    sub = piv[[cycle_a, cycle_b]].dropna()
    ctrl = df[df.is_control].groupby("cycle").isoprenol.mean()
    fold_a = sub[cycle_a] / ctrl.loc[cycle_a]
    fold_b = sub[cycle_b] / ctrl.loc[cycle_b]
    return {
        "cycle_a": cycle_a,
        "cycle_b": cycle_b,
        "n_shared_designs": int(len(sub)),
        "pearson": float(np.corrcoef(sub[cycle_a], sub[cycle_b])[0, 1]),
        "spearman": float(sub.corr(method="spearman").iloc[0, 1]),
        "pearson_fold_over_control": float(np.corrcoef(fold_a, fold_b)[0, 1]),
        "mean_a": float(sub[cycle_a].mean()),
        "mean_b": float(sub[cycle_b].mean()),
        "mean_paired_difference": float((sub[cycle_b] - sub[cycle_a]).mean()),
        "mean_dcas9_a": float(df[df.cycle == cycle_a].dCas9.mean()),
        "mean_dcas9_b": float(df[df.cycle == cycle_b].dCas9.mean()),
        "control_mean_a": float(ctrl.loc[cycle_a]),
        "control_mean_b": float(ctrl.loc[cycle_b]),
    }


def si_titer_agreement(df: pd.DataFrame) -> dict:
    """Prove the shared CSV's titers are the published ones, bit for bit.

    If this holds, the loader sources the phenotype from the mirrored SI and the
    shared CSV is only a convenience join, not a measurement of its own.
    """
    si_path = osp.join(MIRROR_LIB, "si", "si9.xlsx")
    f1 = pd.read_excel(si_path, sheet_name="Figure 1D").rename(
        columns={"isoprenol (mg/L)": "iso_si"}
    )
    f4 = pd.read_excel(si_path, sheet_name="Figure 4b").rename(
        columns={"isoprenoli titer (mg/L)": "iso_si"}
    )
    si = pd.concat([f1, f4], ignore_index=True)
    merged = df.merge(
        si[["Line Name", "cycle", "iso_si"]], on=["Line Name", "cycle"], how="left"
    )
    matched = merged.dropna(subset=["iso_si"])
    diff = (matched.isoprenol - matched.iso_si).abs()
    return {
        "si_file": si_path,
        "si_rows": int(len(si)),
        "comparisons": int(len(matched)),
        "unmatched_rows": int(merged.iso_si.isna().sum()),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "exact_within_1e-6": int((diff < 1e-6).sum()),
    }


def vector_per_cycle() -> pd.DataFrame:
    """SI Table 4: which vector each cycle's CRISPRi arrays were built in."""
    t4 = pd.read_excel(osp.join(MIRROR_LIB, "si", "si7.xlsx"), header=1).iloc[1:]
    t4.columns = [
        "plasmid",
        "jbei_accession",
        "cycle",
        "construct",
        "n_sgrna",
        "vector",
    ]
    return t4.groupby(["cycle", "vector"]).size().rename("n_arrays").reset_index()


def model_diff() -> dict:
    base = json.load(open(osp.join(SHARED, "iJN1463.json")))
    aug = json.load(open(osp.join(SHARED, "pputida_isoprenol.json")))

    def ids(model, key):
        return {x["id"] for x in model[key]}

    def objective(model):
        return [r["id"] for r in model["reactions"] if r.get("objective_coefficient")]

    return {
        "base": {
            "id": base.get("id"),
            "reactions": len(base["reactions"]),
            "metabolites": len(base["metabolites"]),
            "genes": len(base["genes"]),
            "objective": objective(base),
        },
        "augmented": {
            "id": aug.get("id"),
            "reactions": len(aug["reactions"]),
            "metabolites": len(aug["metabolites"]),
            "genes": len(aug["genes"]),
            "objective": objective(aug),
        },
        "reactions_added": sorted(ids(aug, "reactions") - ids(base, "reactions")),
        "reactions_removed": sorted(ids(base, "reactions") - ids(aug, "reactions")),
        "metabolites_added": sorted(ids(aug, "metabolites") - ids(base, "metabolites")),
        "genes_added": sorted(ids(aug, "genes") - ids(base, "genes")),
        "heterologous_genes_added": sorted(
            g for g in ids(aug, "genes") - ids(base, "genes") if not g.startswith("PP_")
        ),
    }


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    df = load_campaign()

    report = {
        "shared_dir": SHARED,
        "mirror_key": "carruthersAutomationMachineLearning2025",
        "campaign": campaign_shape(df),
        "vector_swap_2_vs_3": vector_swap(df),
        "replicate_ceiling": [split_half_reliability(df, c) for c in (2, 3)],
        "si_titer_agreement": si_titer_agreement(df),
        "metabolic_model_diff": model_diff(),
    }

    with open(osp.join(RESULTS, "carruthers_campaign_audit.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    design_reuse(df).to_csv(
        osp.join(RESULTS, "design_reuse_across_cycles.csv"), index=False
    )
    vector_per_cycle().to_csv(osp.join(RESULTS, "vector_per_cycle.csv"), index=False)
    df.groupby("cycle").isoprenol.describe().to_csv(
        osp.join(RESULTS, "titer_by_cycle.csv")
    )

    swap = report["vector_swap_2_vs_3"]
    ceil = report["replicate_ceiling"]
    print(
        f"rows {report['campaign']['rows']}, designs "
        f"{report['campaign']['distinct_designs']}, targets "
        f"{report['campaign']['distinct_targets']}"
    )
    print(
        f"cycle 2 vs 3: {swap['n_shared_designs']} shared designs, "
        f"pearson {swap['pearson']:.3f}, paired diff "
        f"{swap['mean_paired_difference']:.1f} mg/L"
    )
    for c in ceil:
        print(
            f"  replicate ceiling cycle {c['cycle']}: "
            f"{c['reliability']:.3f} {c['ci90']}"
        )
    si = report["si_titer_agreement"]
    print(
        f"SI titer agreement: {si['comparisons']} comparisons, "
        f"max abs diff {si['max_abs_diff']:.3g}, "
        f"unmatched {si['unmatched_rows']}"
    )
    print(f"results -> {RESULTS}")


if __name__ == "__main__":
    main()
