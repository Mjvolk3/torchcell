# experiments/028-gene-essentiality/scripts/degree_by_class.py
# [[experiments.028-gene-essentiality.scripts.degree_by_class]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/degree_by_class
"""Raw degree as an essentiality score, with nothing trained.

Round 30 measured that degree and clustering through two layers of convolution over
the physical graph read 0.753 on the 4,674 genome-wide held-out genes, level with the
best model on the set. This scores the gene's degree itself, per graph, on three gene
sets: the genome held-out test genes, the released 195 held-out genes, and the 787
released genes the models train and validate on. A graph whose degree ranks the
genome-wide genes but not the released ones says the graph carries something the
metabolic genes do not vary in.

Writes ``results/degree_by_class.json`` for ``notes_tex_tables.py``.
"""

import gzip
import json
import os
import os.path as osp
import sys

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel
from sklearn.metrics import roc_auc_score

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "scripts"))

from graph_baselines import gene_graphs  # noqa: E402

RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
SPLITS = osp.join(RESULTS, "splits")


class DegreeRead(BaseModel):
    """Degree's AUROC on one gene set for one graph."""

    graph: str
    gene_set: str
    n_genes: int
    n_essential: int
    auroc: float
    ci: tuple[float, float]
    median_degree_essential: float
    median_degree_nonessential: float
    zero_degree_essential: int
    zero_degree_nonessential: int


def load(name: str) -> dict:
    with gzip.open(osp.join(SPLITS, name), "rt") as fh:
        return json.load(fh)


def gene_sets() -> dict[str, dict[str, int]]:
    genome = load("ess_genome_heldout_val0.2_s0.json.gz")
    fcl = load("ess_fcl_val0.2_s0.json.gz")
    return {
        "genome held-out (4,674)": {
            g: int(genome["labels"][g]) for g in genome["genes"]["test"]
        },
        "released held-out (195)": {g: int(fcl["labels"][g]) for g in fcl["genes"]["test"]},
        "released train + val (787)": {
            g: int(fcl["labels"][g])
            for k in ("train", "val")
            for g in fcl["genes"][k]
        },
    }


def bootstrap(score: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            draws.append(float(roc_auc_score(y[idx], score[idx])))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    gene_ids, graphs = gene_graphs()
    known = set(gene_ids)
    reads: list[DegreeRead] = []
    for graph_name, g in graphs.items():
        deg = dict(g.degree())
        for set_name, labels in gene_sets().items():
            genes = sorted(x for x in labels if x in known)
            score = np.array([float(deg.get(x, 0)) for x in genes])
            y = np.array([labels[x] for x in genes])
            ess = score[y == 1]
            non = score[y == 0]
            reads.append(
                DegreeRead(
                    graph=graph_name,
                    gene_set=set_name,
                    n_genes=len(genes),
                    n_essential=int(y.sum()),
                    auroc=float(roc_auc_score(y, score)),
                    ci=bootstrap(score, y),
                    median_degree_essential=float(np.median(ess)),
                    median_degree_nonessential=float(np.median(non)),
                    zero_degree_essential=int((ess == 0).sum()),
                    zero_degree_nonessential=int((non == 0).sum()),
                )
            )
            r = reads[-1]
            print(
                f"{graph_name:28s} {set_name:28s} auroc {r.auroc:.3f} "
                f"[{r.ci[0]:.3f}, {r.ci[1]:.3f}] median degree ess {r.median_degree_essential:.0f} "
                f"non {r.median_degree_nonessential:.0f} zero {r.zero_degree_essential}/"
                f"{r.zero_degree_nonessential}",
                flush=True,
            )
    out = osp.join(RESULTS, "degree_by_class.json")
    with open(out, "w") as fh:
        json.dump({"reads": [r.model_dump() for r in reads]}, fh, indent=2)
    print(out)


if __name__ == "__main__":
    main()
