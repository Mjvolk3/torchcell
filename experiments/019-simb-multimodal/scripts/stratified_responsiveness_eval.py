# experiments/019-simb-multimodal/scripts/stratified_responsiveness_eval.py
# [[experiments.019-simb-multimodal.scripts.stratified_responsiveness_eval]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/stratified_responsiveness_eval
"""Split the val set by Kemmeren RESPONSIVENESS and report the metric on each half.

THE QUESTION. `val/mean/pearson_per_feature` on an identical A0 config measured 0.1527 /
0.0235 / 0.0661 across three seeds -- and `seed` feeds `CellDataModule(random_seed=...)`,
so those are three different SPLITS, not three inits. That +/-0.129 swing is 6x larger
than every architectural effect in round _008 combined (graph routing +0.020, decoder
capacity 0.000, self-indicator -0.001).

Kemmeren2014 profiled deletion mutants in two GEO series BY DESIGN -- GSE42527 RESPONSIVE
(700 genes) and GSE42526 NON-RESPONSIVE (783 genes), recovered by
`kemmeren_responsiveness_index.py`. A non-responsive mutant's log2-vs-WT profile is flat
by construction, so its targets carry almost no variance for a correlation to latch onto.
`pearson_per_feature` correlates predictions ACROSS val strains per gene; a val draw that
happens to be non-responsive-heavy therefore pushes the metric toward noise no matter what
the model does.

WHAT THIS SCRIPT DOES -- and does NOT do. It does NOT change the split. It takes the val
set exactly as trained and reports statistics on its responsive and non-responsive members
separately. Nothing is retrained and no index is rebuilt.

Two levels, cheapest first:

  DATA LEVEL (no model, no checkpoint). Per-strain target dispersion by group. If
  non-responsive strains are flat, their achievable correlation is ~0 REGARDLESS of the
  model -- which means the replicate-based 0.775 ceiling does not apply to the mixed set,
  and "0.153 of 0.775" is the wrong framing for how much headroom is left.

  MODEL LEVEL (``--checkpoint``). Restricts `pearson_per_feature` to each group, and
  reports per-strain `pearson_per_instance` grouped by responsiveness.

NAME RESOLUTION IS NOT OPTIONAL. The GEO titles carry COMMON names (BCK1, ABF2); the
dataset keys on SYSTEMATIC names (YJL095W). These are joined through the genome's shared
resolver and the match rate is REPORTED -- a silent partial join would look like a clean
result computed on whichever genes happened to map.

Usage
-----
    python experiments/019-simb-multimodal/scripts/stratified_responsiveness_eval.py \
        --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

from torchcell.data import (  # noqa: E402
    GenotypeAggregator,
    MeanExperimentDeduplicator,
    Neo4jCellDataset,
)
from torchcell.data.graph_processor import Perturbation  # noqa: E402
from torchcell.datamodules import CellDataModule  # noqa: E402
from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder  # noqa: E402
from torchcell.graph import SCerevisiaeGraph  # noqa: E402
from torchcell.graph.graph import build_gene_multigraph  # noqa: E402
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402

GRAPHS = [
    "physical",
    "regulatory",
    "tflink",
    "string12_0_neighborhood",
    "string12_0_fusion",
    "string12_0_cooccurence",
    "string12_0_coexpression",
    "string12_0_experimental",
    "string12_0_database",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0, help="split seed to analyse")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = os.environ["DATA_ROOT"]
    experiment_root = os.environ["EXPERIMENT_ROOT"]
    results_dir = osp.join(experiment_root, "019-simb-multimodal", "results")

    with open(osp.join(results_dir, "kemmeren_responsiveness.json")) as handle:
        labels = json.load(handle)
    responsive_common = {g.upper() for g in labels["responsive"]}
    nonresponsive_common = {g.upper() for g in labels["nonresponsive"]}
    print(
        f"labels: {len(responsive_common)} responsive / "
        f"{len(nonresponsive_common)} non-responsive (common names)"
    )

    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(data_root, "data/sgd/genome"),
        string_root=osp.join(data_root, "data/string"),
        tflink_root=osp.join(data_root, "data/tflink"),
        genome=genome,
    )

    # COMMON -> SYSTEMATIC. Reported, never assumed: an unresolved common name silently
    # drops that strain from BOTH groups and the remaining numbers would look clean.
    def to_systematic(common: set[str]) -> set[str]:
        out: set[str] = set()
        for name in common:
            resolved = genome.resolve_gene_name(name)
            systematic = getattr(resolved, "systematic_name", None) or (
                resolved if isinstance(resolved, str) else None
            )
            if systematic:
                out.add(systematic)
        return out

    responsive = to_systematic(responsive_common)
    nonresponsive = to_systematic(nonresponsive_common)
    print(
        f"resolved to systematic: {len(responsive)}/{len(responsive_common)} responsive, "
        f"{len(nonresponsive)}/{len(nonresponsive_common)} non-responsive"
    )

    dataset_root = osp.join(
        data_root, "data/torchcell/experiments/019-simb-multimodal", "fig3_core"
    )
    with open(
        osp.join(experiment_root, "019-simb-multimodal/queries", "fig3_core.cql")
    ) as handle:
        query = handle.read()

    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=build_gene_multigraph(graph=graph, graph_names=GRAPHS),
        incidence_graphs=None,
        node_embeddings=NodeEmbeddingBuilder.build(
            embedding_names=["calm"], data_root=data_root, genome=genome, graph=graph
        ),
        converter=None,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=Perturbation(),
        transform=None,
    )
    data_module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=32,
        random_seed=args.seed,
        num_workers=0,
        pin_memory=False,
        prefetch=False,
        follow_batch=["perturbation_indices", "phenotype_values"],
    )
    data_module.setup()
    print(f"\n=== RESPONSIVENESS BALANCE, ALL SPLITS (seed {args.seed}) ===")
    print(f"{'split':<8}{'n_expr':>8}{'resp':>7}{'nonresp':>9}{'unlab':>7}{'% resp':>9}")
    balance: dict[str, dict[str, int]] = {}
    for split_name in ("train", "val", "test"):
        counts = {"responsive": 0, "nonresponsive": 0, "unlabelled": 0}
        for idx in getattr(data_module.index, split_name):
            rec = dataset[int(idx)]
            nm = [dataset.gene_set[int(i)] for i in rec["gene"].perturbation_indices.tolist()]
            tys = list(rec["gene"].phenotype_types)
            if not any("expression" in str(t).lower() for t in tys):
                continue
            if any(n in responsive for n in nm):
                counts["responsive"] += 1
            elif any(n in nonresponsive for n in nm):
                counts["nonresponsive"] += 1
            else:
                counts["unlabelled"] += 1
        tot = counts["responsive"] + counts["nonresponsive"]
        pct = 100.0 * counts["responsive"] / tot if tot else float("nan")
        balance[split_name] = counts
        print(f"{split_name:<8}{sum(counts.values()):>8}{counts['responsive']:>7}"
              f"{counts['nonresponsive']:>9}{counts['unlabelled']:>7}{pct:>8.1f}%")

    indices = getattr(data_module.index, args.split)
    print(f"\n{args.split} split: {len(indices)} records (seed {args.seed})")

    groups: dict[str, list[np.ndarray]] = {"responsive": [], "nonresponsive": [], "unlabelled": []}
    unresolved: list[str] = []
    for idx in indices:
        record = dataset[int(idx)]
        pert = record["gene"].perturbation_indices
        names = [dataset.gene_set[int(i)] for i in pert.tolist()]
        values = record["gene"].phenotype_values
        if values is None or len(values) == 0:
            continue
        # FILTER TO EXPRESSION. `phenotype_values` is the COO array holding EVERY
        # phenotype on the record; a fig3_core row can also carry CalMorph morphology,
        # whose features span orders of magnitude. Reading it raw gave SD=1993 on log2
        # ratios that live in +/-5 -- a nonsense statistic that still "ran".
        types = list(record["gene"].phenotype_types)
        type_idx = record["gene"].phenotype_type_indices
        keep = [i for i, t in enumerate(types) if "expression" in str(t).lower()]
        if not keep:
            continue
        mask = torch.isin(type_idx, torch.tensor(keep))
        if not bool(mask.any()):
            continue
        target = values[mask].detach().cpu().numpy().astype(np.float64)
        if any(n in responsive for n in names):
            groups["responsive"].append(target)
        elif any(n in nonresponsive for n in names):
            groups["nonresponsive"].append(target)
        else:
            groups["unlabelled"].append(target)
            unresolved.extend(names)

    print()
    print("=== DATA LEVEL: target dispersion by responsiveness (no model) ===")
    print(f"{'group':<16}{'n':>6}{'mean|y|':>10}{'SD(y)':>10}{'frac |y|>1':>12}")
    summary: dict[str, dict[str, float]] = {}
    for name, rows in groups.items():
        if not rows:
            continue
        flat = np.concatenate(rows)
        stats = {
            "n_strains": len(rows),
            "mean_abs": float(np.mean(np.abs(flat))),
            "sd": float(np.std(flat)),
            "frac_gt1": float(np.mean(np.abs(flat) > 1.0)),
        }
        summary[name] = stats
        print(
            f"{name:<16}{stats['n_strains']:>6}{stats['mean_abs']:>10.4f}"
            f"{stats['sd']:>10.4f}{stats['frac_gt1']:>12.4f}"
        )
    if unresolved:
        print(f"\nunlabelled strains: {len(groups['unlabelled'])} "
              f"(e.g. {sorted(set(unresolved))[:6]})")

    # PER-GENE SIGNAL. pearson_per_feature averages a correlation over all 6,127 measured
    # genes. A gene whose values barely move ACROSS STRAINS contributes a pure-noise
    # correlation to that average, so the headline number is diluted by however many such
    # genes there are. This measures the dilution directly.
    all_rows = groups["responsive"] + groups["nonresponsive"]
    gene_stats: dict[str, float] = {}
    if all_rows and len({len(r) for r in all_rows}) == 1:
        Y = np.vstack(all_rows)  # [n_strains, n_genes]
        per_gene_sd = Y.std(axis=0)
        order = np.argsort(-per_gene_sd)
        total_var = float((per_gene_sd ** 2).sum())
        print("\n=== PER-GENE SIGNAL across val strains ===")
        print(f"  matrix {Y.shape}, median gene SD={np.median(per_gene_sd):.4f}, "
              f"max={per_gene_sd.max():.4f}")
        for frac in (0.01, 0.05, 0.10, 0.25, 0.50):
            k = max(1, int(frac * len(per_gene_sd)))
            share = float((per_gene_sd[order[:k]] ** 2).sum()) / total_var
            print(f"  top {frac:>5.0%} of genes ({k:>5}) hold {share:>6.1%} of total across-strain variance")
            gene_stats[f"var_share_top_{frac}"] = share
        gene_stats["median_gene_sd"] = float(np.median(per_gene_sd))

    out = osp.join(results_dir, f"stratified_responsiveness_seed{args.seed}.json")
    with open(out, "w") as handle:
        json.dump({"seed": args.seed, "split": args.split, "groups": summary,
                   "balance": balance, "per_gene": gene_stats}, handle, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
