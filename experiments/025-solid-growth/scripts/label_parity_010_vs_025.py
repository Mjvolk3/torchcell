# experiments/025-solid-growth/scripts/label_parity_010_vs_025.py
# [[experiments.025-solid-growth.scripts.recapitulate_tmi_from_fitness]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/label_parity_010_vs_025
"""Compare tmi/tmf label values on the shared triples between the 010 and 025 builds.

The 025 build re-aggregated every genotype group from the uncapped graph, so the label a
shared triple carries may have shifted relative to 010. Any S0-vs-010 architecture
comparison inherits that shift; this measures it. Genotypes are matched by sorted
gene-name set, the same identity the split transfer used.
"""

import gzip
import json
import os
import os.path as osp

import lmdb
import numpy as np
from dotenv import load_dotenv
from scipy import stats

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

LMDB_010 = osp.join(
    DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/001-small-build/processed/lmdb"
)
RECAP_TABLE = osp.join(
    DATA_ROOT,
    "data/torchcell/experiments/025-solid-growth/recapitulation/recapitulation_per_triple.csv.gz",
)
OUT = osp.join(EXPERIMENT_ROOT, "025-solid-growth/results/label_parity_010_vs_025.json")


def labels_010() -> dict[tuple, dict]:
    env = lmdb.open(LMDB_010, readonly=True, lock=False)
    out: dict[tuple, dict] = {}
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            entries = json.loads(value.decode())
            genes = tuple(
                sorted(
                    {
                        p["systematic_gene_name"]
                        for p in entries[0]["experiment"]["genotype"]["perturbations"]
                    }
                )
            )
            fit, gi = [], []
            for item in entries:
                e = item["experiment"]
                if e["experiment_type"] == "fitness":
                    v = e["phenotype"]["fitness"]
                    if v is not None:
                        fit.append(v)
                elif e["experiment_type"] == "gene interaction":
                    v = e["phenotype"]["gene_interaction"]
                    if v is not None:
                        gi.append(v)
            out[genes] = {
                "gi": float(np.mean(gi)) if gi else None,
                "fit": float(np.mean(fit)) if fit else None,
            }
    env.close()
    return out


def main() -> None:
    old = labels_010()
    print(f"010 genotypes: {len(old)}", flush=True)

    pairs_gi, pairs_fit = [], []
    n_matched = 0
    with gzip.open(RECAP_TABLE, "rt") as f:
        header = f.readline().strip().split(",")
        col = {c: i for i, c in enumerate(header)}
        for line in f:
            v = line.rstrip("\n").split(",")
            genes = (v[col["gene_a"]], v[col["gene_b"]], v[col["gene_c"]])
            o = old.get(genes)
            if o is None:
                continue
            n_matched += 1
            tmi = v[col["tmi_stored"]]
            tmf = v[col["tmf_stored"]]
            if tmi and o["gi"] is not None:
                pairs_gi.append((o["gi"], float(tmi)))
            if tmf and o["fit"] is not None:
                pairs_fit.append((o["fit"], float(tmf)))

    def report(pairs: list[tuple]) -> dict:
        if not pairs:
            return {"n": 0}
        a = np.array(pairs)
        d = a[:, 1] - a[:, 0]
        return {
            "n": len(pairs),
            "identical": int((d == 0).sum()),
            "pearson": float(stats.pearsonr(a[:, 0], a[:, 1])[0]),
            "max_abs_diff": float(np.abs(d).max()),
            "mean_abs_diff": float(np.abs(d).mean()),
            "frac_within_1e-6": float((np.abs(d) < 1e-6).mean()),
        }

    summary = {
        "n_010": len(old),
        "n_matched": n_matched,
        "gene_interaction": report(pairs_gi),
        "fitness": report(pairs_fit),
    }
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
