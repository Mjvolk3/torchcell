# experiments/W019-echo-crispr-array/scripts/published_source_audit.py
# [[experiments.W019-echo-crispr-array.scripts.published_source_audit]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/W019-echo-crispr-array/scripts/published_source_audit
"""Re-derive the published-source numbers the W019 notes quote.

Three things are produced, each small enough for `verify_triple_build_list.py` to read
without re-scanning a multi-million-record store.

1. `published_source_kuzmin_tau_se.csv`
   Kuzmin's own SE(tau), back-solved from the stored tau and P-value in
   `tmi_kuzmin{2018,2020}`. The released P-value column tops out at exactly 0.5, so it is
   a ONE-SIDED tail probability and the inversion is SE = |tau| / Phi^-1(1 - p). The
   two-sided inversion is reported alongside so the difference is visible; it is NOT the
   right one. The one-sidedness is confirmed independently in
   `published_source_pvalue_calibration.csv`.

2. `published_source_costanzo_panel.csv`
   Costanzo 2016 coverage of the eleven panel genes: single-mutant fitness records, and
   digenic records / distinct partners / significant partners at Costanzo's own
   intermediate threshold (P < 0.05 and |eps| > 0.08).

3. `published_source_inpanel_digenics.csv`
   Every one of the 55 gene pairs inside the eleven-gene basis, whether Costanzo measured
   it, and whether it is significant. This is what settles "0 of its N in-panel digenics
   is significant" for `YLR312C-B`.

The digenic pass reads each dataset's flat `preprocess/data.csv` rather than
deserializing the 20.7M-record LMDB, the same route the committed
`experiments/010-kuzmin-tmi/scripts/investigate_YLR313C_smf_and_interactions.py` takes.
Counts were checked against a full LMDB pass and agree exactly.

Run from repo root:
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/W019-echo-crispr-array/scripts/published_source_audit.py
"""

from __future__ import annotations

import csv
import itertools
import os
import os.path as osp
import pickle

import lmdb
import numpy as np
from dotenv import load_dotenv
from scipy import stats

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXP_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
RESULTS = osp.join(EXP_DIR, "results")

GENES = [
    "YBR203W", "YDR057W", "YER079W", "YGL087C", "YJR060W", "YKL033W-A",
    "YLL012W", "YLR104W", "YLR312C-B", "YPL046C", "YPL081W",
]
GENE_SET = frozenset(GENES)

# Costanzo 2016 intermediate confidence threshold, quoted in their SI (sha256
# 1828703b0ff739fdf1c0d9232fe4fd81a3ce95a1b111780f55ef63bfa676880e, "Genetic interaction
# data files" / data-release section): "We suggest three different thresholds [lenient
# (P<0.05), intermediate (P<0.05 and |eps|>0.08), and stringent confidence (P<0.05 and
# eps>0.16 or eps<-0.12)]".
P_THRESH = 0.05
EPS_THRESH = 0.08

# Kuzmin's raw release, used only to identify the reference distribution behind the
# released P-value column. Column 12 is the combined-mutant fitness standard deviation.
KUZMIN_RAW = osp.join(
    DATA_ROOT, "torchcell-library/kuzminSystematicAnalysisComplex2018/si/si_data",
    "Data File S1_Raw genetic interaction dataset.tsv",
)


def num(cell: str) -> float | None:
    """Float, or None for the release's three empty markers. Anything else fails loudly."""
    s = cell.strip()
    if s in ("", "NaN", "nan"):
        return None
    return float(s)


# ---------------------------------------------------------------------------------------
# 1. Kuzmin SE(tau)
# ---------------------------------------------------------------------------------------
def kuzmin_tau_se() -> None:
    rows = []
    all_tau, all_p = [], []
    for name in ("tmi_kuzmin2018", "tmi_kuzmin2020"):
        env = lmdb.open(osp.join(DATA_ROOT, "data/torchcell", name, "processed/lmdb"),
                        readonly=True, lock=False, subdir=True)
        tau, pv = [], []
        with env.begin() as tx:
            for _, v in tx.cursor():
                ph = pickle.loads(v)["experiment"]["phenotype"]
                tau.append(ph["gene_interaction"])
                pv.append(ph["gene_interaction_p_value"])
        env.close()
        rows.append(summarize_se(name, np.asarray(tau, float), np.asarray(pv, float)))
        all_tau.extend(tau)
        all_p.extend(pv)
        print(f"    {name}: {len(tau)} records")
    rows.append(summarize_se("both", np.asarray(all_tau, float), np.asarray(all_p, float)))

    out = osp.join(RESULTS, "published_source_kuzmin_tau_se.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote -> {out}")


def summarize_se(name: str, tau: np.ndarray, p: np.ndarray) -> dict:
    """SE back-solved per record. p is one-sided, so z = Phi^-1(1 - p) and SE = |tau|/z.

    Records at p == 0 (z infinite) and p == 0.5 (z zero) are not invertible and are
    excluded; both counts are reported so the exclusion is visible.
    """
    ok = np.isfinite(tau) & np.isfinite(p) & (p > 0) & (p < 0.5)
    se1 = np.abs(tau[ok]) / stats.norm.isf(p[ok])
    se2 = np.abs(tau[ok]) / stats.norm.isf(p[ok] / 2.0)
    return {
        "dataset": name,
        "n_records": int(len(tau)),
        "n_usable": int(ok.sum()),
        "n_p_zero": int((p == 0).sum()),
        "n_p_half": int((p == 0.5).sum()),
        "p_max": round(float(np.nanmax(p)), 6),
        "median_abs_tau": round(float(np.median(np.abs(tau))), 6),
        "se_one_sided_p25": round(float(np.percentile(se1, 25)), 6),
        "se_one_sided_median": round(float(np.median(se1)), 6),
        "se_one_sided_p75": round(float(np.percentile(se1, 75)), 6),
        "se_two_sided_median": round(float(np.median(se2)), 6),
    }


# ---------------------------------------------------------------------------------------
# 2. Which reference distribution produced the released P-value
# ---------------------------------------------------------------------------------------
def pvalue_calibration() -> None:
    """The digenic arm of Kuzmin's raw release carries the combined-mutant fitness SD.

    If the released P is a one-sided normal tail of |eps|/SE, then |eps|/Phi^-1(1-p)
    divided by that SD is nearly constant across records. Under a two-sided reading the
    same ratio is not. Rank correlation and relative IQR make the choice, so the
    one-sidedness is a measured property of the release, not an assumption.
    """
    eps, p, sd = [], [], []
    with open(KUZMIN_RAW, newline="") as fh:
        header = next(csv.reader([fh.readline()], delimiter="\t"))
        ci = {c: header.index(c) for c in (
            "Combined mutant type",
            "Adjusted genetic interaction score (epsilon or tau)",
            "P-value",
            "Combined mutant fitness standard deviation",
        )}
        for line in fh:
            rec = next(csv.reader([line], delimiter="\t"))
            if rec[ci["Combined mutant type"]] != "digenic":
                continue
            eps.append(num(rec[ci["Adjusted genetic interaction score (epsilon or tau)"]]))
            p.append(num(rec[ci["P-value"]]))
            sd.append(num(rec[ci["Combined mutant fitness standard deviation"]]))
    e = np.array([np.nan if v is None else v for v in eps], float)
    q = np.array([np.nan if v is None else v for v in p], float)
    s = np.array([np.nan if v is None else v for v in sd], float)
    ok = np.isfinite(e) & np.isfinite(q) & np.isfinite(s) & (q > 0) & (q < 0.5) & (s > 0)
    e, q, s = np.abs(e[ok]), q[ok], s[ok]

    rows = []
    for label, quant in (("one_sided_normal", stats.norm.isf(q)),
                         ("two_sided_normal", stats.norm.isf(q / 2.0))):
        r = (e / quant) / s
        lo, med, hi = np.percentile(r, [25, 50, 75])
        rows.append({
            "reference_distribution": label,
            "n": int(len(r)),
            "median_se_over_sd": round(float(med), 6),
            "relative_iqr": round(float((hi - lo) / med), 6),
            "spearman_se_vs_sd": round(float(stats.spearmanr(e / quant, s).statistic), 6),
        })
    out = osp.join(RESULTS, "published_source_pvalue_calibration.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote -> {out}")


# ---------------------------------------------------------------------------------------
# 3. Costanzo coverage of the panel
# ---------------------------------------------------------------------------------------
def costanzo_panel() -> None:
    smf_records = {g: 0 for g in GENES}
    smf_strains = {g: set() for g in GENES}
    smf_value = {g: set() for g in GENES}
    env = lmdb.open(osp.join(DATA_ROOT, "data/torchcell/smf_costanzo2016/processed/lmdb"),
                    readonly=True, lock=False, subdir=True)
    with env.begin() as tx:
        for _, v in tx.cursor():
            perts = pickle.loads(v)["experiment"]["genotype"]["perturbations"]
            g = perts[0]["systematic_gene_name"]
            if g in GENE_SET:
                smf_records[g] += 1
                smf_strains[g].add(perts[0]["strain_id"])
                ph = pickle.loads(v)["experiment"]["phenotype"]
                smf_value[g].add((ph["fitness"], ph["fitness_std"]))
    env.close()

    dmi_records = {g: 0 for g in GENES}
    partners = {g: set() for g in GENES}
    sig_partners = {g: set() for g in GENES}
    pair_rows = {frozenset(pr): [] for pr in itertools.combinations(GENES, 2)}

    path = osp.join(DATA_ROOT, "data/torchcell/dmi_costanzo2016/preprocess/data.csv")
    with open(path, newline="") as fh:
        header = next(csv.reader([fh.readline()]))
        ci = {c: header.index(c) for c in (
            "Query Systematic Name", "Array Systematic Name",
            "Genetic interaction score (ε)", "P-value",
        )}
        for line in fh:
            if not any(g in line for g in GENE_SET):
                continue
            rec = next(csv.reader([line]))
            a = rec[ci["Query Systematic Name"]]
            b = rec[ci["Array Systematic Name"]]
            if a == b:
                continue
            hit = GENE_SET.intersection((a, b))
            if not hit:
                continue
            e = num(rec[ci["Genetic interaction score (ε)"]])
            q = num(rec[ci["P-value"]])
            sig = e is not None and q is not None and q < P_THRESH and abs(e) > EPS_THRESH
            for g in hit:
                other = b if a == g else a
                dmi_records[g] += 1
                partners[g].add(other)
                if sig:
                    sig_partners[g].add(other)
            if len(hit) == 2:
                pair_rows[frozenset((a, b))].append((e, q, sig))

    out = osp.join(RESULTS, "published_source_costanzo_panel.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gene", "smf_records", "smf_strains", "smf_distinct_values",
                    "smf_fitness", "smf_std", "dmi_records",
                    "dmi_partners", "dmi_significant_partners"])
        for g in GENES:
            # every panel gene's records agree on one (fitness, std) pair; a second pair
            # would mean the strains disagree, which the CSV would then show
            vals = sorted(smf_value[g])
            w.writerow([g, smf_records[g], len(smf_strains[g]), len(vals),
                        vals[0][0], vals[0][1], dmi_records[g],
                        len(partners[g]), len(sig_partners[g])])
    print(f"  wrote -> {out}")

    out = osp.join(RESULTS, "published_source_inpanel_digenics.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "measured", "n_records", "n_significant",
                    "min_p", "max_abs_eps"])
        for pr in sorted(pair_rows, key=lambda s: " + ".join(sorted(s))):
            rows = pair_rows[pr]
            ps = [q for _, q, _ in rows if q is not None]
            es = [abs(e) for e, _, _ in rows if e is not None]
            w.writerow([
                " + ".join(sorted(pr)),
                int(bool(rows)),
                len(rows),
                sum(1 for _, _, s in rows if s),
                round(min(ps), 6) if ps else "",
                round(max(es), 6) if es else "",
            ])
    print(f"  wrote -> {out}")


def main() -> None:
    print("1. Kuzmin SE(tau) back-solve (reading tmi LMDBs)")
    kuzmin_tau_se()
    print("2. P-value reference-distribution calibration (Kuzmin raw release)")
    pvalue_calibration()
    print("3. Costanzo panel coverage (SMF LMDB + Dmi preprocess CSV)")
    costanzo_panel()


if __name__ == "__main__":
    main()
