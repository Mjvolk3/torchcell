# experiments/017-hoepfner-background-mutations/scripts/hoepfner_crossvalidate_table_s5.py
# [[experiments.017-hoepfner-background-mutations.scripts.hoepfner_crossvalidate_table_s5]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/017-hoepfner-background-mutations/scripts/hoepfner_crossvalidate_table_s5
"""Cross-validate the empirical background-mutation detector against Hoepfner Table_S5.

Table_S5 (sha256 b123dc3e...; library mirror si/Table_S5.xls) is the AUTHORITATIVE list. Its
`Data` sheet assigns every collection strain a Cluster / Lab / Batch; its `Table` sheet maps
each of the 4 clusters to its mutation. Cluster labels (Legend sheet): `CLn` = strain is
POSITIONALLY in cluster n (the paper's 157); `CLn+` = strain CORRELATES with cluster n but is
NOT positionally in it (31 extra -> 188 total flagged).

This turns Tier A of the risk audit from an ESTIMATE into the exact strain- and record-precise
footprint, and scores how well the promiscuity+adjacency detector (which used none of Table_S5)
recovered it. Writes results/table_s5_affected_strains.csv + results/table_s5_crossvalidation.json.
"""

import json
import os
import os.path as osp
import re

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "017-hoepfner-background-mutations", "results")
S5 = osp.join(
    DATA_ROOT,
    "torchcell-library/hoepfnerHighresolutionChemicalDissection2014/si/Table_S5.xls",
)
TOTAL, HIP_TOTAL = 29_996_238, 16_939_418
CLUSTER_MUTATION = {
    "CL1": "Chromosome XI aneuploidy", "CL2": "Chromosome XI aneuploidy",
    "CL3": "WHI2 nonsense mutation", "CL4": "Chromosome V locus amplification",
}


def parse_cluster(x: str) -> tuple[bool, str]:
    """(is_positional, base_cluster CLn) from a label like 'CL4', 'CL3+', 'CL4, CL3+'."""
    toks = [t.strip() for t in str(x).split(",")]
    positional = [t for t in toks if re.match(r"^CL[1-4]$", t)]
    if positional:
        return True, positional[0]
    return False, toks[0].rstrip("+")


def footprint(hip: pd.DataFrame, orfs: set) -> tuple[int, int]:
    m = hip[hip["orf"].isin(orfs)]
    return len(m), int(m["n_exp"].sum())


def main() -> None:
    base_hip = json.load(open(osp.join(RESULTS_DIR, "summary_stats.json")))[
        "HIP"]["baseline_neg_rate"]["neg_rate_4"]

    data = pd.read_excel(S5, sheet_name="Data", header=0)
    data = data[data["ORF"].astype(str).str.match(r"^Y[A-P][LR]\d")].copy()
    aff = data[data["Cluster"].notna()].copy()
    aff["ORF"] = aff["ORF"].astype(str).str.strip().str.upper()
    aff[["is_positional", "base_cluster"]] = aff["Cluster"].apply(
        lambda x: pd.Series(parse_cluster(x))
    )
    positional = aff[aff["is_positional"]]
    print(f"Table_S5 flagged strains: {len(aff)} total | {len(positional)} positional "
          f"(paper states 157)")
    print("positional per-cluster:", dict(positional["base_cluster"].value_counts()))
    print("lab (all flagged):", dict(aff["Lab"].value_counts()))

    hip = pd.read_csv(osp.join(RESULTS_DIR, "hip_strain_metrics.csv"))
    hip["orf"] = hip["orf"].str.upper()
    hip["enr"] = hip["frac_hyper_4"] / base_hip
    hip_orfs = set(hip["orf"])
    flagged_orfs = set(pd.read_csv(
        osp.join(RESULTS_DIR, "flagged_hip_strains.csv"))["orf"].str.upper())

    pos_orfs, all_orfs = set(positional["ORF"]), set(aff["ORF"])
    absent = all_orfs - hip_orfs
    n_pos, rec_pos = footprint(hip, pos_orfs & hip_orfs)
    n_all, rec_all = footprint(hip, all_orfs & hip_orfs)
    print(f"\nabsent from our HIP LMDB: {len(absent)} {sorted(absent)}")
    print(f"=== AUTHORITATIVE Tier A ===")
    print(f"157 positional  -> {n_pos} in LMDB, {rec_pos:,} rec "
          f"({rec_pos/TOTAL*100:.2f}% all / {rec_pos/HIP_TOTAL*100:.2f}% HIP)")
    print(f"188 incl. corr. -> {n_all} in LMDB, {rec_all:,} rec "
          f"({rec_all/TOTAL*100:.2f}% all / {rec_all/HIP_TOTAL*100:.2f}% HIP)")

    # detector vs authoritative (restricted to strains we actually measure)
    truth = all_orfs & hip_orfs
    tp, fp, fn = flagged_orfs & truth, flagged_orfs - truth, truth - flagged_orfs
    prec = len(tp) / len(flagged_orfs) if flagged_orfs else 0.0
    rec = len(tp) / len(truth) if truth else 0.0
    hip["enr_pct"] = hip["enr"].rank(pct=True)
    miss_pct = hip[hip["orf"].isin(fn)]["enr_pct"].median()
    print(f"\n=== detector (107 flagged @>=4x) vs 188 authoritative ===")
    print(f"TP={len(tp)} FP={len(fp)} FN={len(fn)}  precision={prec:.2f} recall={rec:.2f}")
    print(f"missed strains' HIP-enrichment percentile (median): {miss_pct:.2f} "
          f"(many affected strains individually look clean; Table_S5 flags by position+lab)")

    out = hip[hip["orf"].isin(all_orfs)].merge(
        aff[["ORF", "Cluster", "base_cluster", "is_positional", "Lab", "Batch",
             "Chromosome Arm", "Validation Result"]],
        left_on="orf", right_on="ORF", how="left").drop(columns=["ORF"])
    out["mutation"] = out["base_cluster"].map(CLUSTER_MUTATION)
    out["empirically_flagged"] = out["orf"].isin(flagged_orfs)
    out = out[["orf", "gene", "chrom", "pos", "Cluster", "base_cluster", "is_positional",
               "mutation", "Lab", "Batch", "Chromosome Arm", "Validation Result", "n_exp",
               "enr", "whi2_corr", "empirically_flagged"]].sort_values(
        ["base_cluster", "chrom", "pos"])
    out.to_csv(osp.join(RESULTS_DIR, "table_s5_affected_strains.csv"), index=False)

    json.dump(dict(
        table_s5=dict(
            sha256="b123dc3e87fc10d3b4256f449fcd2eb38c91d1779000278af5a1a788356624a2",
            source="Dryad doi:10.5061/dryad.v5m8v file id 4834604 (si/Table_S5.xls)",
            n_positional=len(positional), n_total_flagged=len(aff),
            cluster_mutation=CLUSTER_MUTATION,
            lab_counts={k: int(v) for k, v in aff["Lab"].value_counts().items()}),
        tier_a_positional_157=dict(n_in_lmdb=n_pos, n_records=rec_pos,
                                   pct_all=rec_pos / TOTAL * 100, pct_hip=rec_pos / HIP_TOTAL * 100),
        tier_a_total_188=dict(n_in_lmdb=n_all, n_records=rec_all,
                              pct_all=rec_all / TOTAL * 100, pct_hip=rec_all / HIP_TOTAL * 100),
        n_absent_from_lmdb=len(absent), absent=sorted(absent),
        crossvalidation=dict(n_flagged=len(flagged_orfs), TP=len(tp), FP=len(fp),
                             FN=len(fn), precision=prec, recall=rec),
    ), open(osp.join(RESULTS_DIR, "table_s5_crossvalidation.json"), "w"), indent=2)
    print(f"\nwrote table_s5_affected_strains.csv ({len(out)} rows) + table_s5_crossvalidation.json")


if __name__ == "__main__":
    main()
