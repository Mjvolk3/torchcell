# experiments/043-expression-proteome-morphology/scripts/make_global_split_043.py
# [[experiments.043-expression-proteome-morphology.scripts.make_global_split_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/make_global_split_043
"""Write the 043 fold file ``conf/splits/global_split_v1.json`` (``GlobalSplit043``).

Store generation v2 (``STORE_043_VERSION=v2``, ``store_version_043``) draws
``conf/splits/global_split_v2.json`` from ``raw_043_v2`` under the same seed and
stratification, and its summary ``results/global_split_v2_summary.json`` compares the
gene folds with v1 (genes that changed fold, the pinned genes, Jackson 2020 per fold). A
deleted gene of a ``released_identifier`` perturbation (``released_kind`` deletion) is
keyed like any other: its ORF when the name resolves, its released spelling otherwise.

The draw follows the primer ("The evaluation protocol", "The fold file"), one
``numpy.random.default_rng(0)`` stream consumed in a fixed order (genes, pinned genes,
isolates, segregants, promoters):

* Genes. Every gene deleted in a record of Kemmeren 2014, Hu 2007, Jackson 2020,
  Nadal-Ribelles 2025, Messner 2023, Zelezniak 2018 or Ohya 2005, as its systematic ORF in
  the store's 6,607 gene index (the dataset module's ``_gene_idx``: a standard name such
  as ``ENA1`` resolves to its ORF; a name with no index position keeps its released
  spelling and is counted). Stratum = the gene's dataset-membership pattern over those
  seven datasets. Within a stratum the genes are permuted and dealt round-robin into ten
  folds, the deal continuing across strata so each fold gets a tenth of every pattern.
  Pinned genes are dealt the same way into folds 2..9 only: the 82 Sameith 2015 single
  deletions (decision P2; they are Kemmeren profiles and the constituents of split B) and
  YDL227C (HO), the reference every Jackson 2020 effect is taken against.
* Isolates. The strain names of Caudal 2024 and Muenzner 2024 (protein and turnover),
  stratum = membership in (Caudal, Muenzner protein, Muenzner turnover), five folds.
* Segregants. Key ``f"{dataset}:{cross}:{segregant_id}"``, five folds dealt within each
  (dataset, cross, panel) with panel the record's medium (Boocock's pooled SM panel and
  its de novo FACS cells are separate panels; de novo cells are records of their own).
* Promoters. The distinct Vaishnav 2022 80-mers (the released variable region,
  ``promoter_sequence[variable_region_start - 1 : variable_region_end]``), five folds.

Records are read by raw index through ``dataset_043.Raw043Dataset.record`` (the raw-order
index ``raw/raw_dataset_name_index.json``), in a process pool; the 163 GB LMDB is never
scanned with a cursor.

    python experiments/043-expression-proteome-morphology/scripts/make_global_split_043.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from typing import Any

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from dataset_043 import (  # noqa: E402
    DELETION_TYPES,
    RAW_INDEX_NAME,
    RELEASED_DELETION_KINDS,
    Dataset043Config,
    Raw043Dataset,
)
from global_split_043 import (  # noqa: E402
    DELETION_DATASETS,
    EXCLUDED_DATASET,
    ISOLATE_DATASETS,
    N_GENE_FOLDS,
    N_OTHER_FOLDS,
    PROMOTER_DATASETS,
    SEGREGANT_DATASETS,
    SPLIT_B_DATASET,
    GlobalSplit043,
    Role,
    SplitCounts,
)
from store_version_043 import active as active_store  # noqa: E402

EXPERIMENT = "043-expression-proteome-morphology"
HO = "YDL227C"
ALL_DATASETS = (
    DELETION_DATASETS
    + [SPLIT_B_DATASET, EXCLUDED_DATASET]
    + ISOLATE_DATASETS
    + SEGREGANT_DATASETS
    + PROMOTER_DATASETS
)
_DS: Raw043Dataset | None = None


def isolate_key(record: dict[str, Any]) -> str:
    """The isolate's strain name; the one Muenzner 2024 record with an empty genotype is
    the reference strain and is keyed ``reference:<genome_reference.strain>`` (FY4).
    """
    g = record["experiment"]["genotype"]
    if "natural_isolate" in g:
        return g["natural_isolate"]["isolate"]
    if g["perturbations"]:
        raise ValueError(
            "isolate-dataset record with perturbations but no natural_isolate"
        )
    return "reference:" + record["experiment_reference"]["genome_reference"]["strain"]


def _key_of(item: tuple[str, int]) -> tuple[str, int, Any]:
    """The split key of one record (runs in a pool worker)."""
    assert _DS is not None
    name, raw_index = item
    record = _DS.record(raw_index)
    ex = record["experiment"]
    if ex["dataset_name"] != name:
        raise ValueError(f"data_{raw_index} is {ex['dataset_name']}, index says {name}")
    g = ex["genotype"]
    if name in SEGREGANT_DATASETS:
        return (
            name,
            raw_index,
            (g["cross"], g["segregant_id"], ex["environment"]["media"]["name"]),
        )
    if name in ISOLATE_DATASETS:
        return name, raw_index, isolate_key(record)
    if name in PROMOTER_DATASETS:
        (p,) = g["perturbations"]
        seq = p["promoter_sequence"][
            p["variable_region_start"] - 1 : p["variable_region_end"]
        ]
        if len(seq) != 80:
            raise ValueError(f"data_{raw_index}: variable region of length {len(seq)}")
        return name, raw_index, seq
    genes = []
    for p in g["perturbations"]:
        released_deletion = (
            p["perturbation_type"] == "released_identifier"
            and p["released_kind"] in RELEASED_DELETION_KINDS
        )
        if p["perturbation_type"] not in DELETION_TYPES and not released_deletion:
            raise ValueError(f"{name} data_{raw_index}: {p['perturbation_type']}")
        idx = _DS._gene_idx(p["systematic_gene_name"])
        genes.append(p["systematic_gene_name"] if idx is None else _DS.genes[idx])
    return name, raw_index, sorted(genes)


def read_keys(ds: Raw043Dataset, workers: int) -> dict[str, list[tuple[int, Any]]]:
    """Dataset -> [(raw_index, key)] for every record of ``ALL_DATASETS``."""
    global _DS
    _DS = ds
    keys: dict[str, list[tuple[int, Any]]] = defaultdict(list)
    t0 = time.time()
    with Pool(workers) as pool:
        for n, (name, raw_index, key) in enumerate(
            pool.imap(_key_of, ds.items, chunksize=256), start=1
        ):
            keys[name].append((raw_index, key))
            if n % 20000 == 0:
                print(
                    f"{n} / {len(ds.items)} records, {time.time() - t0:.0f} s",
                    flush=True,
                )
    print(f"read {len(ds.items)} records in {time.time() - t0:.0f} s", flush=True)
    return keys


def deal(
    strata: dict[str, list[str]], folds: list[int], rng: np.random.Generator
) -> dict[str, int]:
    """Permute each stratum and deal round-robin over ``folds``, continuing across strata."""
    out: dict[str, int] = {}
    offset = 0
    for stratum in sorted(strata):
        members = sorted(strata[stratum])
        for i, j in enumerate(rng.permutation(len(members))):
            out[members[j]] = folds[(offset + i) % len(folds)]
        offset += len(members)
    return out


def compare_to_v1(
    v1_path: str, split: GlobalSplit043, keys: dict[str, list[tuple[int, Any]]]
) -> dict[str, Any]:
    """Gene folds of this split against ``global_split_v1``: genes that changed fold,
    genes new or gone, the pinned genes' folds, and the Jackson 2020 genes per fold.
    """
    v1 = GlobalSplit043.model_validate_json(open(v1_path).read())
    common = sorted(set(v1.gene_folds) & set(split.gene_folds))
    changed = [g for g in common if v1.gene_folds[g] != split.gene_folds[g]]
    moved_role = Counter(
        f"{v1._fold_role(v1.gene_folds[g]).value}->"
        f"{split._fold_role(split.gene_folds[g]).value}"
        for g in changed
    )
    jackson = sorted(
        {
            g
            for _, genes in keys["TfDeletionPseudobulkJackson2020Dataset"]
            for g in genes
        }
    )
    return {
        "v1_store_sha256": v1.store_sha256,
        "n_genes_v1": len(v1.gene_folds),
        "n_genes_v2": len(split.gene_folds),
        "n_genes_common": len(common),
        "n_genes_changed_fold": len(changed),
        "genes_changed_fold_by_role": dict(sorted(moved_role.items())),
        "genes_new": sorted(set(split.gene_folds) - set(v1.gene_folds)),
        "genes_gone": sorted(set(v1.gene_folds) - set(split.gene_folds)),
        "pinned_same_set": sorted(v1.pinned_training_genes)
        == sorted(split.pinned_training_genes),
        "pinned_in_training_folds": all(
            split.gene_folds[g] >= 2 for g in split.pinned_training_genes
        ),
        "jackson_genes_by_fold": dict(
            sorted(Counter(split.gene_folds[g] for g in jackson).items())
        ),
    }


def deal_new(
    strata: dict[str, list[str]],
    folds: list[int],
    rng: np.random.Generator,
    previous: dict[str, int],
) -> dict[str, int]:
    """Inheritance: every key ``previous`` holds keeps its fold; only the NEW keys of
    each stratum are permuted and dealt round-robin over ``folds``, the deal continuing
    across strata (``deal`` restricted to the new keys).
    """
    out = {
        k: previous[k] for members in strata.values() for k in members if k in previous
    }
    new = {
        stratum: [k for k in members if k not in previous]
        for stratum, members in strata.items()
    }
    out.update(deal({k: v for k, v in new.items() if v}, folds, rng))
    return out


def main() -> None:
    """Read the keys, draw the folds, count, validate, write."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--inherit-from",
        default=None,
        help="split version to inherit (e.g. global_split_v1): its keys keep their "
        "folds and only new keys are drawn; omitted = full redraw",
    )
    parser.add_argument(
        "--max-records", type=int, default=None, help="smoke test: records per dataset"
    )
    args = parser.parse_args()
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    exp_dir = osp.join(os.environ["EXPERIMENT_ROOT"], EXPERIMENT)
    ds = Raw043Dataset(
        Dataset043Config(
            data_root=data_root,
            datasets=ALL_DATASETS,
            max_records_per_dataset=args.max_records,
        )
    )
    index_path = osp.join(ds.config.store_root, "raw", RAW_INDEX_NAME)
    with open(index_path, "rb") as f:
        store_sha = hashlib.sha256(f.read()).hexdigest()
    with open(osp.join(ds.config.store_root, "processed", "gene_set.json"), "rb") as f:
        gene_set_sha = hashlib.sha256(f.read()).hexdigest()

    keys = read_keys(ds, args.workers)
    rng = np.random.default_rng(args.seed)
    prev: GlobalSplit043 | None = None
    if args.inherit_from is not None:
        prev = GlobalSplit043.model_validate_json(
            open(
                osp.join(exp_dir, "conf", "splits", f"{args.inherit_from}.json")
            ).read()
        )

    def draw(
        strata: dict[str, list[str]], folds: list[int], previous: dict[str, int]
    ) -> dict[str, int]:
        if prev is None:
            return deal(strata, folds, rng)
        return deal_new(strata, folds, rng, previous)

    # ---- genes
    membership: dict[str, set[str]] = defaultdict(set)
    for name in DELETION_DATASETS:
        for _, genes in keys[name]:
            for g in genes:
                membership[g].add(name)
    sameith_singles = sorted({g for _, genes in keys[EXCLUDED_DATASET] for g in genes})
    double_genes = sorted({g for _, genes in keys[SPLIT_B_DATASET] for g in genes})
    missing_singles = sorted(set(double_genes) - set(sameith_singles))
    if missing_singles:
        raise ValueError(f"Sameith double genes without a single: {missing_singles}")
    pinned = sorted(set(sameith_singles) | {HO})
    pinned_absent = [g for g in pinned if g not in membership]
    if pinned_absent:
        raise ValueError(f"pinned genes deleted in no split-A dataset: {pinned_absent}")
    short = {n: n.replace("Dataset", "") for n in DELETION_DATASETS}
    pattern = {
        g: "+".join(short[n] for n in DELETION_DATASETS if n in s)
        for g, s in membership.items()
    }
    free: dict[str, list[str]] = defaultdict(list)
    pinned_strata: dict[str, list[str]] = defaultdict(list)
    for g, p in pattern.items():
        (pinned_strata if g in pinned else free)[p].append(g)
    prev_genes = {} if prev is None else prev.gene_folds
    gene_folds = draw(free, list(range(N_GENE_FOLDS)), prev_genes)
    gene_folds.update(draw(pinned_strata, list(range(2, N_GENE_FOLDS)), prev_genes))
    outside_index = sorted(g for g in membership if g not in ds.gene_to_idx)

    # ---- isolates
    iso_member: dict[str, set[str]] = defaultdict(set)
    for name in ISOLATE_DATASETS:
        for _, strain in keys[name]:
            iso_member[strain].add(name)
    iso_pattern = {
        s: "".join("1" if n in m else "0" for n in ISOLATE_DATASETS)
        for s, m in iso_member.items()
    }
    iso_strata: dict[str, list[str]] = defaultdict(list)
    for s, p in iso_pattern.items():
        iso_strata[p].append(s)
    isolate_folds = draw(
        iso_strata,
        list(range(N_OTHER_FOLDS)),
        {} if prev is None else prev.isolate_folds,
    )

    # ---- segregants
    seg_strata: dict[str, list[str]] = defaultdict(list)
    seg_key_of: dict[tuple[str, int], str] = {}
    for name in SEGREGANT_DATASETS:
        for raw_index, (cross, sid, media) in keys[name]:
            key = f"{name}:{cross}:{sid}"
            seg_key_of[(name, raw_index)] = f"{cross}:{sid}"
            seg_strata[f"{name}|{cross}|{media}"].append(key)
    n_seg = sum(len(v) for v in seg_strata.values())
    if len({k for v in seg_strata.values() for k in v}) != n_seg:
        raise ValueError("segregant keys are not unique per record")
    segregant_folds = draw(
        seg_strata,
        list(range(N_OTHER_FOLDS)),
        {} if prev is None else prev.segregant_folds,
    )

    # ---- promoters
    mers = sorted({seq for _, seq in keys[PROMOTER_DATASETS[0]]})
    promoter_folds = draw(
        {"all": mers},
        list(range(N_OTHER_FOLDS)),
        {} if prev is None else prev.promoter_folds,
    )

    store = active_store()
    split = GlobalSplit043(
        version=store.split_version,
        mode="redraw" if prev is None else "inherit",
        inherits_from=args.inherit_from,
        seed=args.seed,
        store_sha256=store_sha,
        store_index_path=index_path,
        gene_set_sha256=gene_set_sha,
        pinned_training_genes=pinned,
        pinned_reason={
            **{g: "Sameith 2015 single deletion (P2)" for g in sameith_singles},
            HO: "HO deletion, the Jackson 2020 reference",
        },
        gene_folds=gene_folds,
        gene_strata=pattern,
        isolate_folds=isolate_folds,
        isolate_strata=iso_pattern,
        segregant_folds=segregant_folds,
        promoter_folds=promoter_folds,
        counts={},
        generated_by=f"experiments/{EXPERIMENT}/scripts/make_global_split_043.py",
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
    )

    # ---- counts per dataset
    unit = {n: "gene" for n in DELETION_DATASETS}
    unit.update({n: "isolate" for n in ISOLATE_DATASETS})
    unit.update({n: "segregant record" for n in SEGREGANT_DATASETS})
    unit.update({n: "distinct 80-mer" for n in PROMOTER_DATASETS})
    unit.update(
        {SPLIT_B_DATASET: "double deletion", EXCLUDED_DATASET: "single deletion"}
    )
    counts: dict[str, SplitCounts] = {}
    for name in ALL_DATASETS:
        c: Counter[str] = Counter()
        units: dict[str, set[str]] = defaultdict(set)
        for raw_index, key in keys[name]:
            if name in SEGREGANT_DATASETS:
                k: Any = seg_key_of[(name, raw_index)]
            else:
                k = key
            role = split.role(name, k)
            c[role.value] += 1
            units[role.value].add("|".join(k) if isinstance(k, list) else k)
        counts[name] = SplitCounts(
            **dict(c),
            units_train=len(units[Role.TRAIN.value]),
            units_val=len(units[Role.VAL.value]),
            units_test=len(units[Role.TEST.value]) or len(units[Role.SPLIT_B.value]),
            unit=unit[name],
        )
    split.counts = counts

    out_dir = osp.join(exp_dir, "conf", "splits")
    os.makedirs(out_dir, exist_ok=True)
    out = osp.join(out_dir, f"{store.split_version}.json")
    with open(out, "w") as f:
        f.write(split.model_dump_json(indent=1))
    GlobalSplit043.model_validate_json(open(out).read())

    fold_sizes = {
        "gene": Counter(gene_folds.values()),
        "isolate": Counter(isolate_folds.values()),
        "segregant": Counter(segregant_folds.values()),
        "promoter": Counter(promoter_folds.values()),
    }
    summary = {
        "out": out,
        "mode": split.mode,
        "inherits_from": split.inherits_from,
        "store_sha256": store_sha,
        "n_genes": len(gene_folds),
        "n_gene_strata": len(set(pattern.values())),
        "n_pinned": len(pinned),
        "genes_outside_index": outside_index,
        "n_isolates": len(isolate_folds),
        "isolate_strata": dict(Counter(iso_pattern.values())),
        "n_segregant_records": len(segregant_folds),
        "segregant_strata": {k: len(v) for k, v in seg_strata.items()},
        "n_promoter_80mers": len(promoter_folds),
        "n_vaishnav_records": len(keys[PROMOTER_DATASETS[0]]),
        "fold_sizes": {k: dict(sorted(v.items())) for k, v in fold_sizes.items()},
        "counts": {k: v.model_dump() for k, v in counts.items()},
    }
    if prev is not None:
        new_genes = sorted(set(gene_folds) - set(prev.gene_folds))
        summary["new_gene_folds"] = {g: gene_folds[g] for g in new_genes}
        summary["new_gene_fold_counts"] = dict(
            sorted(Counter(gene_folds[g] for g in new_genes).items())
        )
        summary["new_gene_strata"] = {g: pattern[g] for g in new_genes}
        summary["inherited_keys_unchanged"] = {
            name: all(now[k] == f for k, f in before.items() if k in now)
            and set(before) <= set(now)
            for name, before, now in (
                ("gene", prev.gene_folds, gene_folds),
                ("isolate", prev.isolate_folds, isolate_folds),
                ("segregant", prev.segregant_folds, segregant_folds),
                ("promoter", prev.promoter_folds, promoter_folds),
            )
        }
    if store.version != "v1":
        summary["against_global_split_v1"] = compare_to_v1(
            osp.join(out_dir, "global_split_v1.json"), split, keys
        )
    summary_path = osp.join(exp_dir, "results", f"{store.split_version}_summary.json")
    summary["record_counts"] = {
        k: {
            r: v.model_dump().get(r, 0)
            for r in ("train", "val", "test", "split_b_test", "probe", "excluded")
        }
        for k, v in counts.items()
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
