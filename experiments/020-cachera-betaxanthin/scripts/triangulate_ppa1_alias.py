# experiments/020-cachera-betaxanthin/scripts/triangulate_ppa1_alias.py
# [[experiments.020-cachera-betaxanthin.scripts.triangulate_ppa1_alias]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/020-cachera-betaxanthin/scripts/triangulate_ppa1_alias
"""Triangulate which ORF the Cachera screen's ``PPA1`` row is: IPP1 (YBR011C) or VMA16 (YHR026W).

``PPA1`` is a two-way alias in the yeast namespace: SGD lists it as a secondary name of BOTH
``IPP1``/YBR011C (inorganic pyrophosphatase, essential) and ``VMA16``/YHR026W (V-ATPase V0
subunit c'', non-essential). ``build_merzbacher_split.py`` resolves it to VMA16 on an
essentiality argument. This script does NOT restate that argument; it builds seven
independent lines and reports each one's verdict separately, so a conflict is visible.

L1  What the TRAINING data actually contains (loader LMDB + the Neo4j fig6 build).
L2  What the raw screen file carries for that row (any systematic id / plate / barcode?).
L3  Collection composition: how many SGD-essential genes appear in the screen at all,
    and whether the apparent hits survive inspection.
L4  Neighbor evidence: does IPP1 or VMA16 appear separately under any other name, and how
    complete is the V-ATPase family in the screen?
L5  VIABILITY, from the source file: colony area and replicate colony count for the PPA1
    row. An ipp1-delta haploid is inviable and yields no colonies.
L6  Phenotype plausibility (SOFT): where the PPA1 value sits among VMA* deletions.
L7  Independent third-party mapping: Merzbacher 2025's released ORF list and labels.

Read-only. Writes a JSON report to ``results/ppa1_alias_triangulation.json``.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import pickle
import re
from typing import Any

import lmdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

CACHERA_RAW = osp.join(
    DATA_ROOT, "data/torchcell/betaxanthin_cachera2023/raw/GA1_2_4_6.csv"
)
CACHERA_LMDB = osp.join(
    DATA_ROOT, "data/torchcell/betaxanthin_cachera2023/processed/lmdb"
)
CACHERA_PREPROCESS = osp.join(
    DATA_ROOT, "data/torchcell/betaxanthin_cachera2023/preprocess/data.csv"
)
# The Neo4j-query build the 020 runs actually trained on (conf/base.yaml dataset_tag).
FIG6_LMDB = osp.join(
    DATA_ROOT,
    "data/torchcell/experiments/019-simb-multimodal/fig6_pigment_transfer/processed/lmdb",
)
ESSENTIAL_LMDB = osp.join(
    DATA_ROOT, "data/torchcell/gene_essentiality_sgd/processed/lmdb"
)
MERZ_DIR = osp.join(DATA_ROOT, "data/merzbacher2025_fcl/deletionprediction-main/data")
SPLIT_JSON = osp.join(
    EXPERIMENT_ROOT, "020-cachera-betaxanthin/results/merzbacher_nested_split.json"
)
RESULTS = osp.join(EXPERIMENT_ROOT, "020-cachera-betaxanthin", "results")

LEVEL = "corrected_mean_intensity.24_mean"
AREA = "area.24_mean"
IPP1, VMA16 = "YBR011C", "YHR026W"

report: dict[str, Any] = {}


def hr(t: str) -> None:
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def deletion_orfs(path: str, json_records: bool = False) -> dict[str, list[str]]:
    """Map single-deletion ORF -> record keys, for a pickle- or JSON-valued experiment LMDB."""
    out: dict[str, list[str]] = {}
    env = lmdb.open(path, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        for key, raw in txn.cursor():
            payload = json.loads(raw) if json_records else [pickle.loads(raw)]
            for rec in payload:
                exp = rec["experiment"]
                for p in exp["genotype"]["perturbations"]:
                    if p["perturbation_type"].endswith("deletion"):
                        out.setdefault(p["systematic_gene_name"], []).append(
                            f"{key.decode()}:{exp['dataset_name']}"
                        )
    env.close()
    return out


genome = SCerevisiaeGenome(
    genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
    go_root=osp.join(DATA_ROOT, "data/go"),
    overwrite=False,
)
idx = genome.feature_index
df = pd.read_csv(CACHERA_RAW)
names = df["gene"].astype(str).str.strip().str.upper()

#: screen ORF -> the source name it came in under (first winner, matching the loader).
screen_orf_to_name: dict[str, str] = {}
ambiguous: dict[str, list[str]] = {}
for raw_name in names.unique():
    if raw_name in ("0", "NAN", ""):
        continue
    res = genome.resolve_gene_name(raw_name)
    if res.systematic_name is not None:
        screen_orf_to_name.setdefault(res.systematic_name, raw_name)
    else:
        ambiguous[raw_name] = res.candidates

# --------------------------------------------------------------- L0: what SGD says
hr("L0 -- the alias itself, from the repo's own resolver")
for name in ["PPA1", "IPP1", "VMA16", "FEN1", "RAD27", "ELO2"]:
    r = genome.resolve_gene_name(name)
    print(f"  {name:6s} -> {r.status:10s} {r.systematic_name} {r.candidates}")
    report.setdefault("resolver", {})[name] = {
        "status": str(r.status),
        "systematic_name": r.systematic_name,
        "candidates": r.candidates,
    }
for target in (IPP1, VMA16):
    std = sorted(k for k, v in idx["standard_to_ids"].items() if target in v)
    ali = sorted(k for k, v in idx["alias_to_ids"].items() if target in v)
    print(f"  {target}: GFF standard={std}  aliases={ali}")
    report.setdefault("gff_names", {})[target] = {"standard": std, "alias": ali}

# ------------------------------------------------------------ L2: the raw source file
hr("L2 -- what the RAW Cachera file carries for that row")
print(f"  columns: {list(df.columns)}")
print(f"  rows: {len(df)}   AMBIGUOUS screen names: {ambiguous}")
report["raw_columns"] = list(df.columns)
report["screen_ambiguous_names"] = ambiguous
for probe in ["PPA1", "IPP1", "VMA16", IPP1, VMA16, "FEN1", "RAD27", "ELO2"]:
    hit = df[names == probe]
    lvl = float(hit.iloc[0][LEVEL]) if len(hit) else None
    print(f"  row {probe:8s}: n={len(hit)}" + (f"  {LEVEL}={lvl:.4f}" if lvl else ""))
    report.setdefault("raw_rows", {})[probe] = {"n": int(len(hit)), "level": lvl}

# ------------------------------------------------------ L1: what the builds contain
hr("L1 -- what the TRAINING builds actually contain")
builds: dict[str, Any] = {}
for label, path, as_json in [
    ("cachera loader LMDB", CACHERA_LMDB, False),
    ("fig6_pigment_transfer (what 020 trained on)", FIG6_LMDB, True),
]:
    orfs = deletion_orfs(path, json_records=as_json)
    print(f"  {label}: {len(orfs)} distinct deletion ORFs")
    for orf, common in ((IPP1, "IPP1"), (VMA16, "VMA16")):
        print(f"      {orf} ({common:5s}): {orfs.get(orf, 'ABSENT')}")
    builds[label] = {
        "n_orfs": len(orfs),
        "IPP1_records": orfs.get(IPP1, []),
        "VMA16_records": orfs.get(VMA16, []),
        "path": path,
    }
report["builds"] = builds

# The stale build resolved ambiguous aliases with `alias_to_systematic[...][0]` -- the FIRST
# candidate (567fa6aa^ cachera2023.py:205-207). That picks YCR034W for FEN1 (right, by luck)
# and YBR011C for PPA1 (wrong). Show both so the mechanism is visible, not inferred.
print("\n  betaxanthin levels stored in the fig6 build for the two ambiguous pairs:")
env = lmdb.open(FIG6_LMDB, readonly=True, lock=False, readahead=False)
stored: dict[str, float] = {}
with env.begin() as txn:
    for _, blob in txn.cursor():
        for rec in json.loads(blob):
            exp = rec["experiment"]
            if exp["dataset_name"] != "BetaxanthinCachera2023Dataset":
                continue
            for p in exp["genotype"]["perturbations"]:
                if p["perturbation_type"].endswith("deletion"):
                    stored[p["systematic_gene_name"]] = exp["phenotype"][
                        "metabolite_level"
                    ]["betaxanthin"]
env.close()
for orf, common, src in (
    (IPP1, "IPP1", "PPA1"),
    (VMA16, "VMA16", "PPA1"),
    ("YCR034W", "ELO2", "FEN1"),
    ("YKL113C", "RAD27", "RAD27"),
):
    raw_hit = df[names == src]
    raw_val = float(raw_hit.iloc[0][LEVEL]) if len(raw_hit) else None
    print(
        f"      {orf} ({common:5s}) stored={stored.get(orf)}  "
        f"raw row {src!r}={raw_val}"
    )
    report.setdefault("stored_vs_raw", {})[orf] = {
        "common": common,
        "stored": stored.get(orf),
        "raw_source_name": src,
        "raw_value": raw_val,
    }

pre = pd.read_csv(CACHERA_PREPROCESS)
print(f"\n  preprocess/data.csv rows={len(pre)}")
for orf in (IPP1, VMA16):
    print(f"      {orf}: {pre[pre['orf'] == orf].to_dict('records')}")
    report.setdefault("preprocess", {})[orf] = pre[pre["orf"] == orf].to_dict("records")

# ------------------------------------------------------- L3: collection composition
hr("L3 -- collection composition (is this a non-essential-only collection?)")
ess: dict[str, str] = {}
env = lmdb.open(ESSENTIAL_LMDB, readonly=True, lock=False, readahead=False)
with env.begin() as txn:
    for _, raw in txn.cursor():
        exp = pickle.loads(raw)["experiment"]
        for p in exp["genotype"]["perturbations"]:
            ess[p["systematic_gene_name"]] = p["perturbed_gene_name"]
env.close()
genes = set(idx["genes"])
ess_live = set(ess) & genes
noness = genes - ess_live
in_screen = set(screen_orf_to_name)
hits = sorted(ess_live & in_screen)
print(f"  live R64 genes {len(genes)}; SGD inviable-null {len(ess_live)}")
print(
    f"  non-essential covered: {len(noness & in_screen)}/{len(noness)} "
    f"({len(noness & in_screen) / len(noness):.2%})"
)
print(
    f"  essential covered:     {len(hits)}/{len(ess_live)} ({len(hits) / len(ess_live):.2%})"
)
ess_tbl = pd.DataFrame(
    [
        (o, ess[o], screen_orf_to_name[o], float(df[names == screen_orf_to_name[o]].iloc[0][AREA]))
        for o in hits
    ],
    columns=["orf", "sgd_name", "screen_name", AREA],
).sort_values(AREA)
print(ess_tbl.to_string(index=False))
print(f"  screen median {AREA} = {df[AREA].median():.1f}")
print(f"  IPP1 essential? {IPP1 in ess}   VMA16 essential? {VMA16 in ess}")
report["essentiality"] = {
    "n_essential_live": len(ess_live),
    "noness_coverage": len(noness & in_screen) / len(noness),
    "ess_coverage": len(hits) / len(ess_live),
    "essential_in_screen": ess_tbl.to_dict("records"),
    "IPP1_essential": IPP1 in ess,
    "VMA16_essential": VMA16 in ess,
}

# ------------------------------------------------------------- L4: neighbor evidence
hr("L4 -- neighbor evidence (V-ATPase family completeness)")
vma_std = sorted(
    (k for k in idx["standard_to_ids"] if re.match(r"^VMA\d+$", k)),
    key=lambda s: int(s[3:]),
)
vma_rows = []
for std in vma_std:
    orf = idx["standard_to_ids"][std][0]
    src = screen_orf_to_name.get(orf)
    row = df[names == src].iloc[0] if src else None
    vma_rows.append(
        {
            "std": std,
            "orf": orf,
            "screen_name": src,
            "area24": float(row[AREA]) if src else None,
            "bx24": float(row[LEVEL]) if src else None,
        }
    )
vma_df = pd.DataFrame(vma_rows)
print(vma_df.to_string(index=False))
print(f"\n  IPP1 {IPP1} in screen under any name: {IPP1 in in_screen}")
print(f"  VMA16 {VMA16} in screen under any name: {VMA16 in in_screen}")
report["vma_family"] = vma_rows

# ----------------------------------------------------------------- L5: viability
hr("L5 -- VIABILITY of the PPA1 strain, from the source file")
ppa1 = df[names == "PPA1"].iloc[0]
area_ok = df[df[AREA].notna()][AREA]
print(
    f"  PPA1: {AREA}={ppa1[AREA]:.1f} over {ppa1['area.24_count']:.0f} colonies at 24 h; "
    f"area.48_mean={ppa1['area.48_mean']:.1f} over {ppa1['area.48_count']:.0f} colonies"
)
print(
    f"  screen {AREA}: median={area_ok.median():.1f}  p1={np.percentile(area_ok, 1):.1f}  "
    f"min={area_ok.min():.1f}"
)
print(f"  PPA1 area percentile = {(area_ok < ppa1[AREA]).mean():.2%} (small but VIABLE)")
report["viability"] = {
    "area24": float(ppa1[AREA]),
    "n_colonies_24h": float(ppa1["area.24_count"]),
    "area48": float(ppa1["area.48_mean"]),
    "n_colonies_48h": float(ppa1["area.48_count"]),
    "area_percentile": float((area_ok < ppa1[AREA]).mean()),
    "screen_median_area": float(area_ok.median()),
}

# ------------------------------------------------------ L6: soft phenotype placement
hr("L6 -- SOFT phenotype placement among V-ATPase deletions")
present = vma_df.dropna(subset=["bx24"])
lvl = df[LEVEL].dropna()
v = float(ppa1[LEVEL])
print(present[["std", "screen_name", "area24", "bx24"]].sort_values("bx24").to_string(index=False))
print(f"\n  PPA1 bx={v:.4f}; VMA* median bx={present.bx24.median():.4f}; "
      f"screen median={lvl.median():.4f}")
print(f"  P(random screen gene bx <= PPA1) = {(lvl <= v).mean():.4f}")
print(f"  PPA1 area {ppa1[AREA]:.1f} vs VMA* median area {present.area24.median():.1f}")
report["soft_phenotype"] = {
    "ppa1_bx": v,
    "vma_median_bx": float(present.bx24.median()),
    "screen_median_bx": float(lvl.median()),
    "p_random_le_ppa1": float((lvl <= v).mean()),
    "vma_median_area": float(present.area24.median()),
}

# --------------------------------------------- L7: independent third-party mapping
hr("L7 -- Merzbacher 2025's independent ORF mapping")
val = pd.read_csv(osp.join(MERZ_DIR, "yeast_production_validation_split.csv"))
test = pd.read_csv(osp.join(MERZ_DIR, "yeast_production_test_split.csv"))["name"]
merz = {}
for orf, common in ((IPP1, "IPP1"), (VMA16, "VMA16"), ("YCR034W", "ELO2"), ("YKL113C", "RAD27")):
    row = val[val.knockout == orf]
    merz[orf] = {
        "common": common,
        "in_test_list": bool((test == orf).any()),
        "label": int(row.label.iloc[0]) if len(row) else None,
    }
    print(f"  {orf} ({common:5s}): in their 640 = {merz[orf]['in_test_list']}, "
          f"released label = {merz[orf]['label']}")
print(f"  their label distribution: {val.label.value_counts().to_dict()}")
print(f"  their genes that are SGD inviable-null: {len(set(val.knockout) & ess_live)}/640")
report["merzbacher"] = merz

with open(SPLIT_JSON) as fh:
    split = json.load(fh)
print(f"\n  our nested split: {IPP1} in test={IPP1 in split['split']['test']}, "
      f"pool={IPP1 in split['split']['train_val_pool']}")
print(f"  our nested split: {VMA16} in test={VMA16 in split['split']['test']}, "
      f"pool={VMA16 in split['split']['train_val_pool']}")
print(f"  split test genes missing from the build: "
      f"{split['availability_in_current_build']['split_test_genes_missing_from_build']}")
report["nested_split"] = {
    "IPP1_in_test": IPP1 in split["split"]["test"],
    "VMA16_in_test": VMA16 in split["split"]["test"],
    "missing_from_build": split["availability_in_current_build"][
        "split_test_genes_missing_from_build"
    ],
}

os.makedirs(RESULTS, exist_ok=True)
out = osp.join(RESULTS, "ppa1_alias_triangulation.json")
with open(out, "w") as fh:
    json.dump(report, fh, indent=2, default=str)
print(f"\nwrote {out}")
