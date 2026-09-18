# experiments/019-simb-multimodal/scripts/split_gene_overlap_audit.py
# [[experiments.019-simb-multimodal.scripts.split_gene_overlap_audit]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/split_gene_overlap_audit
"""Audit the v13 / v14 / v16 partitions for deleted-gene overlap between train and held-out.

The partition (`torchcell/datamodules/cell.py`, `CellDataModule`) shuffles record indices
per index key after `random.seed(split_seed)`; it never looks at gene identity. One record
is one deletion gene SET (`GenotypeAggregator` hashes the sorted systematic names), so an
exact-genotype duplicate cannot cross a split, but a Sameith double and the singles of its
two genes are three records assigned independently. This script counts, per split seed and
per side of the split, how many held-out genotypes share a deleted gene with a training
genotype and of what kind (a double whose parent single is in train; a single whose gene is
deleted in train only inside a double), the modality composition of each side, the
genotype-level overlap between the v13 partition (fig3_core) and the v16 one (fig3_proteome)
at the same seed, and the connected components of the "shares a deleted gene" relation,
which is what a gene-set-identity split would have to move as units.

Reads only the cached index JSONs under
`$DATA_ROOT/data/torchcell/experiments/019-simb-multimodal/<store>/{processed,data_module_cache}`;
no LMDB is opened. Seconds of CPU.

    python experiments/019-simb-multimodal/scripts/split_gene_overlap_audit.py
"""

import json
import os
import os.path as osp
from collections import Counter, defaultdict

from dotenv import load_dotenv

SEEDS = (0, 1, 2, 3)
EXPR = "expression_log2_ratio"
PROT = "protein_abundance"


def load_store(root: str, tag: str) -> dict:
    d = osp.join(root, tag, "processed")
    with open(osp.join(d, "phenotype_label_index.json")) as fh:
        pli = {k: set(v) for k, v in json.load(fh).items()}
    with open(osp.join(d, "dataset_name_index.json")) as fh:
        dni = json.load(fh)
    with open(osp.join(d, "is_any_deletion_gene_index.json")) as fh:
        dgi = json.load(fh)
    genes: dict[int, set[str]] = defaultdict(set)
    for g, ids in dgi.items():
        for i in ids:
            genes[i].add(g)
    name: dict[int, str] = {}
    for nm, ids in dni.items():
        for i in ids:
            name[i] = nm
    return {"pli": pli, "genes": dict(genes), "name": name, "dgi": dgi}


def load_index(root: str, tag: str, seed: int) -> dict[str, list[int]]:
    with open(
        osp.join(root, tag, "data_module_cache", f"index_seed_{seed}.json")
    ) as fh:
        idx = json.load(fh)
    return {k: list(idx[k]) for k in ("train", "val", "test")}


def modality(i: int, pli: dict[str, set[int]]) -> str:
    e = i in pli[EXPR]
    p = i in pli.get(PROT, set())
    if e and p:
        return "both"
    if e:
        return "expr_only"
    if p:
        return "prot_only"
    return "other"


def split_counts(store: dict, splits: dict[str, list[int]]) -> dict[str, dict]:
    genes = store["genes"]
    pli = store["pli"]
    train = set(splits["train"])
    train_genes: set[str] = set()
    train_sets: set[frozenset[str]] = set()
    train_singles: set[str] = set()
    train_double_genes: set[str] = set()
    for i in train:
        train_genes |= genes[i]
        train_sets.add(frozenset(genes[i]))
        if len(genes[i]) == 1:
            train_singles |= genes[i]
        if len(genes[i]) == 2:
            train_double_genes |= genes[i]
    out: dict[str, dict] = {}
    for side in ("train", "val", "test"):
        ids = splits[side]
        mods = Counter(modality(i, pli) for i in ids)
        held = side != "train"
        share = [i for i in ids if held and genes[i] & train_genes]
        out[side] = {
            "n": len(ids),
            "prot_only": mods["prot_only"],
            "expr_only": mods["expr_only"],
            "both": mods["both"],
            "share_gene_w_train": len(share),
            "exact_dup": sum(
                1 for i in ids if held and frozenset(genes[i]) in train_sets
            ),
            "dbl_share1_w_train_single": sum(
                1
                for i in ids
                if held and len(genes[i]) == 2 and genes[i] & train_singles
            ),
            "sgl_w_train_dbl": sum(
                1
                for i in ids
                if held and len(genes[i]) == 1 and genes[i] & train_double_genes
            ),
            "share_by_dataset": dict(Counter(store["name"][i] for i in share)),
        }
    return out


def analyse(root: str, tag: str, filt: set[int] | None, label: str) -> dict:
    store = load_store(root, tag)
    res: dict[str, dict] = {}
    print(f"\n=== {tag} {label} ===")
    print(
        f"{'seed':>4} {'split':>5} {'n':>5} {'prot':>5} {'expr':>5} {'both':>5} "
        f"{'share':>5} {'dup':>4} {'dbl':>4} {'sgl':>4}"
    )
    for s in SEEDS:
        splits = load_index(root, tag, s)
        if filt is not None:
            splits = {k: [i for i in v if i in filt] for k, v in splits.items()}
        res[str(s)] = split_counts(store, splits)
        for side, r in res[str(s)].items():
            print(
                f"{s:>4} {side:>5} {r['n']:>5} {r['prot_only']:>5} {r['expr_only']:>5} "
                f"{r['both']:>5} {r['share_gene_w_train']:>5} {r['exact_dup']:>4} "
                f"{r['dbl_share1_w_train_single']:>4} {r['sgl_w_train_dbl']:>4}"
            )
    return res


def v13_v16_overlap(root: str) -> dict:
    core = load_store(root, "fig3_core")
    prot = load_store(root, "fig3_proteome")
    ec = sorted(core["pli"][EXPR])
    ep = sorted(prot["pli"][EXPR])
    gc = [frozenset(core["genes"][i]) for i in ec]
    gp = [frozenset(prot["genes"][i]) for i in ep]
    out = {
        "n_expression_records": {"fig3_core": len(ec), "fig3_proteome": len(ep)},
        "same_genotype_set": set(gc) == set(gp),
        "same_record_order": gc == gp,
        "positions_with_identical_genotype": sum(1 for a, b in zip(gc, gp) if a == b),
        "seeds": {},
    }
    print("\n=== v13 (fig3_core) vs v16 (fig3_proteome) expression genotypes ===")
    print(
        f"records {len(ec)} / {len(ep)}; same set {out['same_genotype_set']}; "
        f"same order {out['same_record_order']}; identical positions "
        f"{out['positions_with_identical_genotype']}"
    )
    ecs, eps = set(ec), set(ep)
    for s in SEEDS:
        ic = load_index(root, "fig3_core", s)
        ip = load_index(root, "fig3_proteome", s)
        row = {}
        for side in ("train", "val", "test"):
            a = {frozenset(core["genes"][i]) for i in ic[side] if i in ecs}
            b = {frozenset(prot["genes"][i]) for i in ip[side] if i in eps}
            row[side] = {
                "v13_n": len(a),
                "v16_n": len(b),
                "overlap": len(a & b),
                "frac_of_v13": len(a & b) / len(a),
            }
            print(
                f"seed {s} {side:5s} v13 n={len(a):4d} v16 n={len(b):4d} "
                f"overlap={len(a & b):4d} frac={len(a & b) / len(a):.3f}"
            )
        out["seeds"][str(s)] = row
    return out


def gene_share_components(root: str, tag: str) -> dict:
    store = load_store(root, tag)
    genes = store["genes"]
    dgi = store["dgi"]
    parent = {i: i for i in genes}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for ids in dgi.values():
        for i in ids[1:]:
            a, b = find(ids[0]), find(i)
            if a != b:
                parent[a] = b
    comp: dict[int, list[int]] = defaultdict(list)
    for i in genes:
        comp[find(i)].append(i)
    multi = [v for v in comp.values() if len(v) > 1]
    pli = store["pli"]
    mods = Counter(modality(i, pli) for v in multi for i in v)
    dbl_genes: set[str] = set()
    for i in genes:
        if len(genes[i]) == 2:
            dbl_genes |= genes[i]
    per_gene_doubles = Counter(g for i in genes if len(genes[i]) == 2 for g in genes[i])
    out = {
        "genotypes": len(genes),
        "components": len(comp),
        "size_distribution": {
            str(k): v for k, v in sorted(Counter(len(v) for v in comp.values()).items())
        },
        "genotypes_in_multi_member_components": sum(len(v) for v in multi),
        "doubles": sum(1 for i in genes if len(genes[i]) == 2),
        "genes_in_a_double": len(dbl_genes),
        "double_genes_also_a_single": sum(
            1 for g in dbl_genes if any(len(genes[i]) == 1 for i in dgi[g])
        ),
        "genes_in_two_or_more_doubles": sum(
            1 for n in per_gene_doubles.values() if n >= 2
        ),
        "max_doubles_per_gene": max(per_gene_doubles.values())
        if per_gene_doubles
        else 0,
        "modality_of_multi_member_genotypes": dict(mods),
    }
    print(f"\n=== {tag}: 'shares a deleted gene' components ===")
    for k, v in out.items():
        print(f"{k}: {v}")
    return out


def main() -> None:
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    root = osp.join(
        data_root, "data", "torchcell", "experiments", "019-simb-multimodal"
    )
    exp_root = os.environ.get(
        "EXPERIMENT_ROOT", osp.join(osp.dirname(osp.abspath(__file__)), "..", "..")
    )
    out_path = osp.join(
        exp_root, "019-simb-multimodal", "results", "split_gene_overlap_audit.json"
    )

    prot = load_store(root, "fig3_proteome")
    core = load_store(root, "fig3_core")
    res = {
        "generated_by": "experiments/019-simb-multimodal/scripts/split_gene_overlap_audit.py",
        "fig3_proteome_full": analyse(
            root, "fig3_proteome", None, "(v16: require_modalities [])"
        ),
        "fig3_proteome_prot": analyse(
            root,
            "fig3_proteome",
            prot["pli"][PROT],
            "(v14: require_modalities [protein_abundance])",
        ),
        "fig3_proteome_expr": analyse(
            root, "fig3_proteome", prot["pli"][EXPR], "(v16 expression-carrying rows)"
        ),
        "fig3_core_expr": analyse(
            root,
            "fig3_core",
            core["pli"][EXPR],
            "(v13: require_modalities [expression_log2_ratio])",
        ),
        "v13_v16_overlap": v13_v16_overlap(root),
        "gene_share_components": {
            "fig3_proteome": gene_share_components(root, "fig3_proteome")
        },
    }
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
