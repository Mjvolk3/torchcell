# experiments/019-simb-multimodal/scripts/morphology_input_screen.py
# [[experiments.019-simb-multimodal.scripts.morphology_input_screen]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/morphology_input_screen
"""Which served per-deletion measurement predicts the moving CalMorph features?

Ridge from each served (or dev-built) strain-by-feature matrix to the 116 moving Ohya 2005
CalMorph features, the test proteome_morphology_covariation.py ran for the proteome and
expression_morphology_covariation.py for expression, applied to every deletion-indexed
dataset with enough strains in common with Ohya. Each dataset is scored alone and as an
addition to Messner on the strains both cover ("Messner + X" against "Messner alone"), and
the three growth-response panels (Hoepfner 2014 HOP, Hillenmeyer 2008 HOP, YeastPhenome)
are scored together on the strains all four cover.

METHOD. 5-fold out-of-fold ridge, penalty chosen on an inner split of the training fold
from {1, 10, 100, 1e3, 1e4}, per-feature Pearson across held-out strains, the median over
the 116 features, one row-permutation null, seed 0, one fold split. Differences from the
committed proteome screen, which reads 0.2795 there and 0.282 here: inputs z-scored and
clipped at plus or minus 8, a feature kept when measured in at least half the shared
strains, condition-indexed datasets flattened to one column per condition, duplicate
records averaged, heterozygous rows dropped, background genes (present in more than half
the records, the Ohnuki 2022 three-deletion background) removed. Messner abundances are
log2 ratios to the record's reference. ONE SEED: Messner alone ranged 0.214 to 0.289 across
the strain subsets of the contrasts, so a difference under about 0.04 is not resolved. The
growth-axis removal of the committed proteome screen is not applied.

Reads $DATA_ROOT/data/torchcell/<dataset>/processed/lmdb read-only and the raw CalMorph
table of scmd_ohya2005; writes

  results/morphology_input_screen.json   per-dataset rows and the combined panels
  notes-tex/figure-3-gate/tables/morphology_input_screen.tex

    python experiments/019-simb-multimodal/scripts/morphology_input_screen.py           # screen, then table
    python experiments/019-simb-multimodal/scripts/morphology_input_screen.py --table   # table from the json

First run 2026-10-11 as a scratch screen during the dataset survey; this file is that
screen with the repository's paths and the table writer added.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import pickle
import sys
from collections import Counter, defaultdict
from typing import Any

import lmdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from torchcell.utils.paths import experiment_results_dir

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
D = osp.join(DATA_ROOT, "data", "torchcell")
RESULTS = experiment_results_dir("019-simb-multimodal", __file__)
OUT_JSON = osp.join(RESULTS, "morphology_input_screen.json")
TEX_DIR = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "notes-tex", "figure-3-gate", "tables")
OUT_TEX = osp.join(TEX_DIR, "morphology_input_screen.tex")

FOLDS, LAMBDAS, SEED = 5, [1.0, 10.0, 100.0, 1000.0, 10000.0], 0
MIN_COVER = 0.5
ZCLIP = 8.0
MIN_SHARED = 500

#: name, dataset directory under $DATA_ROOT/data/torchcell, extraction mode, label for the table
SOURCES = [
    ("messner", "proteome_messner2023", "messner", "Messner 2023 proteome (SWATH, log2 to reference)"),
    ("kemmeren", "microarray_kemmeren2014", "generic", "Kemmeren 2014 expression (array log2 ratio)"),
    ("ohnuki2022", "scmd_ohnuki2022", "generic", "Ohnuki 2022 CalMorph (three-deletion background)"),
    ("yeastphenome", "yeastphenome", "generic", "YeastPhenome growth z (dev build, 80 screens)"),
    ("hoepfner2014_hop", "env_chemgen_hoepfner2014", "generic", "Hoepfner 2014 HOP chemogenomics"),
    ("hillenmeyer_hom", "env_chemgen_hillenmeyer2008_hom", "generic", "Hillenmeyer 2008 HOP chemogenomics"),
    ("nadal2025", "nadal_ribelles_perturbseq2025", "generic", "Nadal-Ribelles 2025 pseudobulk expression"),
    ("mulleder", "amino_acid_mulleder2016", "generic", "Mulleder 2016 amino-acid pools"),
    ("cooper", "amino_acid_cooper2010", "generic", "Cooper 2010 amino-acid peaks"),
    ("costanzo2021", "env_chemgen_costanzo2021", "generic", "Costanzo 2021 differential fitness, 14 conditions"),
    ("vanacloig2022", "env_chemgen_vanacloig2022", "generic", "Vanacloig-Pedros 2022 inhibitor response, 34"),
    ("smf_costanzo2016", "smf_costanzo2016", "generic", "Costanzo 2016 single-mutant fitness"),
    ("smith2006", "env_chemgen_smith2006", "generic", "Smith 2006 fatty-acid clear zone"),
    ("cachera2023", "betaxanthin_cachera2023", "generic", "Cachera 2023 betaxanthin"),
    ("lopez2024", "isobutanol_screen_lopez2024", "generic", "Lopez 2024 isobutanol biosensor"),
    ("ozaydin2013", "carotenoid_ozaydin2013", "generic", "Ozaydin 2013 carotenoid color"),
]
COMBO = ["messner", "yeastphenome", "hillenmeyer_hom", "hoepfner2014_hop"]
GROWTH = ["yeastphenome", "hillenmeyer_hom", "hoepfner2014_hop"]


def ridge_fit_predict(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray, lam: float) -> np.ndarray:
    mx, my = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = Xtr - mx, Ytr - my
    if Xc.shape[1] > Xc.shape[0]:
        K = Xc @ Xc.T
        a = np.linalg.solve(K + lam * np.eye(K.shape[0]), Yc)
        return (Xte - mx) @ (Xc.T @ a) + my
    A = Xc.T @ Xc + lam * np.eye(Xc.shape[1])
    return (Xte - mx) @ np.linalg.solve(A, Xc.T @ Yc) + my


def col_pearson(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
    Pc, Yc = P - P.mean(0), Y - Y.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (Pc * Yc).sum(0) / np.sqrt((Pc**2).sum(0) * (Yc**2).sum(0))


def ridge_oof(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    folds = np.array_split(rng.permutation(n), FOLDS)
    pred = np.full_like(Y, np.nan, dtype=float)
    for k in range(FOLDS):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(FOLDS) if j != k])
        itel, itr = tr[: len(tr) // 5], tr[len(tr) // 5 :]
        best, bl = -np.inf, LAMBDAS[0]
        for lam in LAMBDAS:
            s = np.nanmean(col_pearson(ridge_fit_predict(X[itr], Y[itr], X[itel], lam), Y[itel]))
            if s > best:
                best, bl = s, lam
        pred[te] = ridge_fit_predict(X[tr], Y[tr], X[te], bl)
    return pred


def z(df: pd.DataFrame) -> pd.DataFrame:
    out = (df - df.mean()) / df.std(ddof=0)
    return out.clip(-ZCLIP, ZCLIP)


def morphology() -> pd.DataFrame:
    mt = pd.read_csv(osp.join(D, "scmd_ohya2005", "raw", "mt4718data.tsv"), sep="\t", index_col=0)
    mt.index = [str(i).upper() for i in mt.index]
    with open(osp.join(RESULTS, "calmorph_moving_features.json")) as fh:
        moving = json.load(fh)["moving"]
    mt = mt[[f for f in moving if f in mt.columns]].astype(float)
    return mt.groupby(level=0).mean()


def extract(dsdir: str, mode: str) -> tuple[pd.DataFrame, int]:
    """Strain (single deletion ORF) by feature matrix, mean over duplicate records."""
    env = lmdb.open(osp.join(D, dsdir, "processed", "lmdb"), readonly=True, lock=False)
    # pass 1: background genes, present in more than half of the first 2,000 records
    freq: Counter[str] = Counter()
    n = 0
    with env.begin() as txn:
        for i, (_, v) in enumerate(txn.cursor()):
            if i >= 2000:
                break
            r = pickle.loads(v)
            freq.update({p["systematic_gene_name"] for p in r["experiment"]["genotype"]["perturbations"]})
            n += 1
    background = {g for g, c in freq.items() if c > 0.5 * n}
    sums: dict[str, dict[tuple[str, ...], float]] = defaultdict(lambda: defaultdict(float))
    cnts: dict[str, dict[tuple[str, ...], int]] = defaultdict(lambda: defaultdict(int))
    ref_log: dict[str, float] | None = None
    with env.begin() as txn:
        for _, v in txn.cursor():
            r = pickle.loads(v)
            e = r["experiment"]
            perts = [p for p in e["genotype"]["perturbations"] if p["systematic_gene_name"] not in background]
            if len(perts) != 1:
                continue
            pt = perts[0]["perturbation_type"]
            if not pt.endswith("deletion") or "heterozygous" in pt:
                continue
            g = perts[0]["systematic_gene_name"]
            ph = e["phenotype"]
            lab = ph["label_name"]
            val = ph[lab]
            if mode == "messner":
                if ref_log is None:
                    ref = r["reference"]["phenotype_reference"]["protein_abundance"]
                    ref_log = {k: float(np.log2(x)) for k, x in ref.items() if x and x > 0}
                val = {k: float(np.log2(x)) - ref_log[k] for k, x in val.items() if x and x > 0 and k in ref_log}
            envd = e.get("environment") or {}
            cond = repr(sorted((k, repr(vv)) for k, vv in envd.items() if k != "provenance_gaps"))
            cond += "|" + str(ph.get("screen_id", "")) + "|" + lab
            if isinstance(val, dict):
                for k, x in val.items():
                    if x is None or not np.isfinite(x):
                        continue
                    sums[g][(cond, k)] += x
                    cnts[g][(cond, k)] += 1
            elif val is not None and np.isfinite(val):
                sums[g][(cond,)] += val
                cnts[g][(cond,)] += 1
    env.close()
    feats = sorted({f for g in sums for f in sums[g]})
    fidx = {f: i for i, f in enumerate(feats)}
    genes = sorted(sums)
    M = np.full((len(genes), len(feats)), np.nan, dtype=np.float32)
    for i, g in enumerate(genes):
        for f, s in sums[g].items():
            M[i, fidx[f]] = s / cnts[g][f]
    return pd.DataFrame(M, index=genes, columns=range(len(feats))), len({f[0] for f in feats})


def prep(X: pd.DataFrame, strains: list[str]) -> pd.DataFrame:
    X = X.reindex(strains)
    X = X.loc[:, X.notna().mean() >= MIN_COVER]
    X = X.loc[:, X.std(ddof=0) > 0]
    return z(X).fillna(0.0)


def score(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    r = col_pearson(ridge_oof(X, Y, rng), Y)
    return float(np.nanmedian(r)), float(np.nanmean(r))


def screen() -> dict[str, Any]:
    mt = morphology()
    rng = np.random.default_rng(SEED)
    mats: dict[str, pd.DataFrame] = {}
    rows: dict[str, dict[str, Any]] = {}
    prot = None
    for name, dsdir, mode, label in SOURCES:
        X, n_cond = extract(dsdir, mode)
        if name in COMBO:
            mats[name] = X
        if name == "messner":
            prot = X
        strains = sorted(set(X.index) & set(mt.index))
        Xp = prep(X, strains)
        Y = z(mt.loc[strains]).fillna(0.0).to_numpy()
        med, mean = score(Xp.to_numpy(), Y, rng)
        perm = rng.permutation(len(strains))
        nmed, _ = score(Xp.to_numpy()[perm], Y, rng)
        rec: dict[str, Any] = {
            "label": label,
            "dataset_dir": dsdir,
            "n_strains_with_morph": len(strains),
            "n_features_kept": int(Xp.shape[1]),
            "n_conditions": n_cond,
            "median_r_moving116": med,
            "mean_r_moving116": mean,
            "null_median_r": nmed,
        }
        if name != "messner":
            assert prot is not None
            s2 = sorted(set(strains) & set(prot.index))
            if len(s2) >= MIN_SHARED:
                Y2 = z(mt.loc[s2]).fillna(0.0).to_numpy()
                P2 = prep(prot, s2).to_numpy()
                X2 = prep(X, s2).to_numpy()
                pm, _ = score(P2, Y2, rng)
                xm, _ = score(X2, Y2, rng)
                cm, _ = score(np.hstack([P2, X2]), Y2, rng)
                rec.update(
                    {
                        "n_shared_with_messner": len(s2),
                        "messner_alone_median": pm,
                        "x_alone_median": xm,
                        "messner_plus_x_median": cm,
                        "gain_over_messner": cm - pm,
                    }
                )
        rows[name] = rec
        print(name, json.dumps(rec), flush=True)
    # the combined panels on the strains all four cover
    s = sorted(set(mt.index).intersection(*[set(m.index) for m in mats.values()]))
    Y = z(mt.loc[s]).fillna(0.0).to_numpy()
    P = {k: prep(m, s).to_numpy() for k, m in mats.items()}
    combo: dict[str, Any] = {"n_strains": len(s)}
    for k in P:
        combo[k] = score(P[k], Y, rng)[0]
    growth = np.hstack([P[k] for k in GROWTH])
    combo["three_growth_panels"] = score(growth, Y, rng)[0]
    both = np.hstack([P["messner"], growth])
    combo["messner_plus_three"] = score(both, Y, rng)[0]
    perm = rng.permutation(len(s))
    combo["null_messner_plus_three"] = score(both[perm], Y, rng)[0]
    print("combo", json.dumps(combo), flush=True)
    return {
        "method": {
            "folds": FOLDS,
            "lambdas": LAMBDAS,
            "seed": SEED,
            "statistic": "median over the 116 moving features of the out-of-fold Pearson across held-out strains",
            "z_clip": ZCLIP,
            "min_feature_cover": MIN_COVER,
            "note": "one seed, one fold split; Messner alone ranges 0.214 to 0.289 across the strain subsets, so differences under about 0.04 are not resolved",
        },
        "datasets": rows,
        "combined": combo,
    }


def table(res: dict[str, Any]) -> str:
    rows = res["datasets"]
    order = sorted(rows, key=lambda k: -rows[k]["median_r_moving116"])
    best_alone = max(rows[k]["median_r_moving116"] for k in rows)
    gains = {k: rows[k].get("gain_over_messner") for k in rows if "gain_over_messner" in rows[k]}
    best_gain = max(gains.values())
    lines = [
        "%% GENERATED by experiments/019-simb-multimodal/scripts/morphology_input_screen.py. Do not edit by hand.",
        "%% SOURCE: experiments/019-simb-multimodal/results/morphology_input_screen.json",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\footnotesize",
        "  \\setlength{\\tabcolsep}{3pt}",
        "  \\caption[Which served measurement predicts morphology]{Ridge from each served per-deletion "
        "measurement to the 116 moving CalMorph features, out of fold: the median over features of the "
        "Pearson across held-out strains, alone on the strains the dataset shares with Ohya 2005, and the "
        "gain of Messner plus the dataset over Messner alone on the strains both cover. One seed and one "
        "fold split; Messner alone ranges 0.214 to 0.289 across the strain subsets of the contrasts, so a "
        "difference under about 0.04 is not resolved. The null is a row permutation of the input. Bold, "
        "the best alone and the best gain. "
        f"Combined panels on the {res['combined']['n_strains']:,} strains all four cover: Messner "
        f"{res['combined']['messner']:.3f}, the three growth panels together {res['combined']['three_growth_panels']:.3f}, "
        f"Messner plus the three {res['combined']['messner_plus_three']:.3f}, permuted null "
        f"{res['combined']['null_messner_plus_three']:.3f}. "
        "\\src{experiments/019-simb-multimodal/scripts/morphology_input_screen.py}}",
        "  \\label{tab:morph-input-screen}",
        "  \\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{50mm}*{6}{>{\\raggedleft\\arraybackslash}p{18mm}}@{}}",
        "    \\toprule",
        "    input (served dataset) & strains shared with Ohya (n) & features kept (n) & median held-out Pearson, alone & permuted null, median Pearson & strains shared with Messner (n) & gain over Messner alone, median Pearson \\\\",
        "    \\midrule",
    ]
    for k in order:
        r = rows[k]
        alone = f"{r['median_r_moving116']:.3f}"
        if r["median_r_moving116"] == best_alone:
            alone = f"\\textbf{{{alone}}}"
        if "gain_over_messner" in r:
            gain = f"{r['gain_over_messner']:+.3f}"
            if r["gain_over_messner"] == best_gain:
                gain = f"\\textbf{{{gain}}}"
            shared = f"{r['n_shared_with_messner']:,}"
        else:
            gain, shared = "reference", f"{r['n_strains_with_morph']:,}"
        lines.append(
            f"    {r['label']} & {r['n_strains_with_morph']:,} & {r['n_features_kept']:,} & {alone} & "
            f"{r['null_median_r']:.3f} & {shared} & {gain} \\\\"
        )
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="store_true", help="write the table from the json without rerunning")
    args = ap.parse_args()
    if args.table:
        with open(OUT_JSON) as fh:
            res = json.load(fh)
    else:
        res = screen()
        with open(OUT_JSON, "w") as fh:
            json.dump(res, fh, indent=1)
    os.makedirs(TEX_DIR, exist_ok=True)
    with open(OUT_TEX, "w") as fh:
        fh.write(table(res))
    print(OUT_JSON, OUT_TEX, sep="\n")


if __name__ == "__main__":
    sys.exit(main())
