# experiments/020-cachera-betaxanthin/scripts/fcl_resplit_sizing_check.py
# [[experiments.020-cachera-betaxanthin.scripts.fcl_resplit_sizing_check]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/020-cachera-betaxanthin/scripts/fcl_resplit_sizing_check
"""Counting only. Sizes the proposed FCL re-split and audits the "537 training strains" claim.

Reads, never writes: the released Merzbacher/FCL split CSVs from the sha256-pinned Zenodo
mirror, the built Cachera LMDB, the `fig6_pigment_transfer` training dataset's precomputed
indices, and the `data_module_cache` split indices that the pinned and unpinned runs actually
used. Emits numbers to stdout. Builds no split and writes no file.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import pickle

import lmdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]

FCL_DATA = osp.join(
    DATA_ROOT, "data/merzbacher2025_fcl/deletionprediction-main/data"
)
CACHERA_LMDB = osp.join(
    DATA_ROOT, "data/torchcell/betaxanthin_cachera2023/processed/lmdb"
)
FIG6 = osp.join(
    DATA_ROOT,
    "data/torchcell/experiments/019-simb-multimodal/fig6_pigment_transfer",
)

#: CellDataModule.train_ratio / val_ratio (torchcell/datamodules/cell.py:259-260); the
#: "ordinary seeded 80/10/10 random split" named in conf/base.yaml:95.
TRAIN_RATIO, VAL_RATIO = 0.8, 0.1


def fcl_sets() -> tuple[set[str], set[str]]:
    test = set(pd.read_csv(osp.join(FCL_DATA, "yeast_production_test_split.csv"))["name"])
    val = set(
        pd.read_csv(osp.join(FCL_DATA, "yeast_production_validation_split.csv"))[
            "knockout"
        ]
    )
    return test, val


def cachera_lmdb_deletions() -> set[str]:
    """Single-deletion systematic ORFs in the standalone Cachera loader build."""
    env = lmdb.open(CACHERA_LMDB, readonly=True, lock=False, readahead=False)
    out: set[str] = set()
    with env.begin() as txn:
        for _, raw in txn.cursor():
            exp = pickle.loads(raw)["experiment"]
            genes = [
                p["systematic_gene_name"]
                for p in exp["genotype"]["perturbations"]
                if p["perturbation_type"].endswith("deletion")
            ]
            if len(genes) == 1:
                out.add(genes[0])
    env.close()
    return out


def fig6_indices() -> dict[str, object]:
    p = osp.join(FIG6, "processed")
    gene_idx = json.load(open(osp.join(p, "is_any_deletion_gene_index.json")))
    heads = json.load(open(osp.join(p, "head_target_presence_index.json")))
    dsets = json.load(open(osp.join(p, "dataset_name_index.json")))
    return {"gene_idx": gene_idx, "heads": heads, "dsets": dsets}


def split_cache() -> dict[str, dict[str, int]]:
    p = osp.join(FIG6, "data_module_cache")
    out = {}
    for f in sorted(os.listdir(p)):
        if f.startswith("index_seed") and f.endswith(".json"):
            d = json.load(open(osp.join(p, f)))
            out[f] = {k: len(v) for k, v in d.items()}
    return out


def ratio_split(n: int, train: float = TRAIN_RATIO, val: float = VAL_RATIO) -> tuple[int, int, int]:
    tr = int(n * train)
    va = int(n * val)
    return tr, va, n - tr - va


def main() -> None:
    test640, val640 = fcl_sets()
    print("=" * 78)
    print("A. FCL RELEASED SPLIT FILES (Zenodo record 15761895)")
    print("=" * 78)
    print(f"  |test|            {len(test640)}")
    print(f"  |val|             {len(val640)}")
    print(f"  |val & test|      {len(val640 & test640)}")
    print(f"  |val - test|      {len(val640 - test640)}")
    print(f"  |test - val|      {len(test640 - val640)}")
    print(f"  sets identical    {val640 == test640}")

    built = cachera_lmdb_deletions()
    idx = fig6_indices()
    gene_idx: dict = idx["gene_idx"]  # type: ignore[assignment]
    heads: dict = idx["heads"]  # type: ignore[assignment]
    dsets: dict = idx["dsets"]  # type: ignore[assignment]

    bx_records = set(heads["betaxanthin"])
    rec_to_gene = {v[0]: g for g, v in gene_idx.items() if len(v) == 1}
    multi = {g: v for g, v in gene_idx.items() if len(v) != 1}
    bx_genes = {rec_to_gene[r] for r in bx_records if r in rec_to_gene}
    cachera_records = set(dsets["BetaxanthinCachera2023Dataset"])

    print()
    print("=" * 78)
    print("B. CACHERA DELETION COUNTS, BY BUILD")
    print("=" * 78)
    print(f"  standalone loader LMDB          {len(built)}")
    print(f"    {CACHERA_LMDB}")
    print(f"  fig6_pigment_transfer records   {len(gene_idx)} (one per deletion gene)")
    print(f"    genes mapping to >1 record    {len(multi)}")
    print(f"    with a betaxanthin target     {len(bx_genes)}")
    print(f"    BetaxanthinCachera2023Dataset {len(cachera_records)} records")
    print(f"    {FIG6}")

    for label, ours in [
        ("standalone Cachera LMDB", built),
        ("fig6 betaxanthin-target genes", bx_genes),
    ]:
        shared = test640 & ours
        print()
        print("=" * 78)
        print(f"C. FCL MAPPING + RE-SPLIT SIZING against {label} (N = {len(ours)})")
        print("=" * 78)
        print(f"  |FCL test|                    {len(test640)}")
        print(f"  |FCL test & ours| = test_FCL  {len(shared)}")
        print(f"  FCL test genes we lack        {len(test640 - shared)}"
              f"  {sorted(test640 - shared)[:12]}")
        pool = ours - test640
        print(f"  non-FCL pool                  {len(pool)}")
        tr, va, te = ratio_split(len(pool))
        print(f"  -> at {TRAIN_RATIO}/{VAL_RATIO}/{round(1 - TRAIN_RATIO - VAL_RATIO, 2)}"
              f" over the non-FCL pool:")
        print(f"       train      {tr}")
        print(f"       val        {va}")
        print(f"       test_ours  {te}")
        print(f"       test_FCL   {len(shared)}")
        print(f"       check      {tr + va + te + len(shared)} = {len(ours)}")
        # The REALIZED fractions the stratified assignment produces, measured off
        # index_seed_42.json (86.11 / 7.00 / 6.90), not the nominal 0.8/0.1/0.1.
        rtr, rva, rte = ratio_split(len(pool), 0.8611, 0.0700)
        print("  -> at the REALIZED 0.8611/0.0700/0.0690 (index_seed_42.json):")
        print(f"       train      {rtr}")
        print(f"       val        {rva}")
        print(f"       test_ours  {rte}")
        print(f"       test_FCL   {len(shared)}")

    print()
    print("=" * 78)
    print("C2. PIN-FILE RECONCILIATION (merzbacher_nested_split.json split.test)")
    print("=" * 78)
    pin_genes = set(
        json.load(
            open(
                osp.join(
                    os.environ["EXPERIMENT_ROOT"],
                    "020-cachera-betaxanthin/results/merzbacher_nested_split.json",
                )
            )
        )["split"]["test"]
    )
    all_genes = set(gene_idx)
    print(f"  |pin file split.test|                 {len(pin_genes)}")
    print(f"  |FCL 640 & fig6 deletion genes|       {len(test640 & all_genes)}")
    print(f"  |FCL 640 & fig6 betaxanthin-labeled|  {len(test640 & bx_genes)}")
    print(f"  |pin & fig6 deletion genes|           {len(pin_genes & all_genes)}  (the 'pin639' tag)")
    print(f"  |pin & fig6 betaxanthin-labeled|      {len(pin_genes & bx_genes)}")
    print(f"  pin genes with a record but no bx     {sorted((pin_genes & all_genes) - bx_genes)}")
    print(f"  YBR011C (IPP1): in build {'YBR011C' in all_genes}, "
          f"bx-labeled {'YBR011C' in bx_genes}, in pin file {'YBR011C' in pin_genes}")

    print()
    print("=" * 78)
    print("D. SPLIT INDICES THE RUNS ACTUALLY USED (fig6 data_module_cache)")
    print("=" * 78)
    for f, sizes in split_cache().items():
        tot = sum(sizes.values())
        frac = {k: round(v / tot, 4) for k, v in sizes.items()}
        print(f"  {f}")
        print(f"    {sizes}  total {tot}  fractions {frac}")

    sc = split_cache()
    unp = sc["index_seed_42.json"]
    pin = sc["index_seed_42_pin639-e069f15e.json"]
    print()
    print("  seed 42, same dataset, pinned vs unpinned:")
    print(f"    train {unp['train']} -> {pin['train']}   delta {unp['train'] - pin['train']}")
    print(f"    val   {unp['val']} -> {pin['val']}   delta {unp['val'] - pin['val']}")
    print(f"    test  {unp['test']} -> {pin['test']}   delta {pin['test'] - unp['test']}")

    print()
    print("=" * 78)
    print("E. THE 537 CLAIM (round_leaderboards_summary.json, n_train_supervised)")
    print("=" * 78)
    summ = json.load(
        open(
            osp.join(
                os.environ["EXPERIMENT_ROOT"],
                "019-simb-multimodal/results/round_leaderboards_summary.json",
            )
        )
    )
    bx = summ["betaxanthin"]["by_project"]
    for proj, rec in bx.items():
        print(f"  {proj:42s} n_train_supervised {rec['n_train_supervised']:.0f}")
    lb = pd.read_csv(
        osp.join(
            os.environ["EXPERIMENT_ROOT"],
            "019-simb-multimodal/results/round_leaderboards.csv",
        )
    )
    sub = lb[lb["project"].astype(str).str.contains("betaxanthin|020_metabolism", na=False)]
    grp = (
        sub.groupby(
            ["project", "n_train_supervised", "n_val_supervised", "n_test_supervised"]
        )
        .size()
        .reset_index(name="n_runs")
    )
    print()
    print("  per-run logged split sizes (round_leaderboards.csv):")
    print(grp.to_string(index=False))
    print()
    print("  unpinned 4235/340/340 vs pinned _v4 3698/286/931:")
    print(f"    train lost {4235 - 3698}   val lost {340 - 286}   test gained {931 - 340}")
    print(f"    {4235 - 3698} + {340 - 286} = {(4235 - 3698) + (340 - 286)} records moved")

    print()
    print("=" * 78)
    print("F. AUTHOR'S SCHEME, SIZED IN THE SAME SUPERVISED-RECORD SPACE")
    print("=" * 78)
    total_sup = 4235 + 340 + 340
    # The pin file's split.test. All 639 are supervised: 537 came out of train and 54 out of
    # val (591 moved), and the remaining 48 were already in the unpinned test set.
    n_pin = len(pin_genes)
    pool_sup = total_sup - n_pin
    print(f"  total supervised records          {total_sup}")
    print(f"  test_FCL (pinned, out of train)   {n_pin}")
    print(f"  non-FCL pool                      {pool_sup}")
    for lbl, tr_r, va_r in [
        ("nominal 0.8 / 0.1 / 0.1", 0.8, 0.1),
        ("realized 0.8611 / 0.0700 / 0.0690", 0.8611, 0.0700),
    ]:
        tr, va, te = ratio_split(pool_sup, tr_r, va_r)
        print(f"  -> {lbl}:")
        print(f"       train      {tr}")
        print(f"       val        {va}")
        print(f"       test_ours  {te}")
        print(f"       test_FCL   {n_pin}")
        print(f"       train delta vs unpinned 4235: {4235 - tr}")


if __name__ == "__main__":
    main()
