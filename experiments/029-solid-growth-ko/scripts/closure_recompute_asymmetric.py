# experiments/029-solid-growth-ko/scripts/closure_recompute_asymmetric.py
# [[experiments.029-solid-growth-ko.scripts.closure_recompute_asymmetric]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/029-solid-growth-ko/scripts/closure_recompute_asymmetric

"""Recompute the trigenic score on the 029 build with the identity the source used.

``closure_recompute.py`` evaluates the SYMMETRIC form of the trigenic identity,

    tau = f_ijk - f_ij f_k - f_ik f_j - f_jk f_i + 2 f_i f_j f_k,

because a record does not say which of its three genes came from the array, and without
that the asymmetric form cannot be written down. It reaches r = 0.517 on the Kuzmin 2018
screen under a Kuzmin-first policy.

The roles are recoverable after all. Exactly one perturbation of a triple carries an
array strain (``_dma`` or ``_tsa``) and the other two carry the query strain's ``tm``
token, and ``torchcell.data.label_table.triple_roles`` reads them; measured on 5,000
sampled triples of this build, both resolve in all 5,000. So the published form can be
evaluated here:

    tau = f_ijk - f_ij f_k - eps_ik - eps_jk

the triple's own fitness, minus the DOUBLE-MUTANT QUERY strain's fitness times the ARRAY
single's fitness, minus the two single-mutant control queries' adjusted scores against
that same array gene, with the query singles entering as 1, which is what the released
scores do.

What this separates. The symmetric form differs from the published one in shape AND in
which measurements it draws on. If the asymmetric form scores about the same, the
remaining gap is the one term the build cannot supply, the double-mutant query strain's
own fitness, which falls through to the pair's digenic array screen on this build because
those records enter the loaders only as of 4c4a4f950. If it scores much better, the shape
was also costing us. Either way it is a sharper statement than the symmetric number.

    python experiments/029-solid-growth-ko/scripts/closure_recompute_asymmetric.py

Reads the 029 build read-only plus the entry cache written by ``closure_recompute.py
scan``. Writes results/closure_asymmetric_summary.json and
results/closure_asymmetric_by_screen.csv.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from multiprocessing import Pool

import lmdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats

from torchcell.data.label_policy import LabelEntry, LabelPolicy
from torchcell.data.label_table import triple_roles

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
BUILD = "/db/experiments/029-solid-growth-ko-001-ko-build"
CACHE = osp.join(DATA_ROOT, "data/torchcell/experiments/029-solid-growth-ko/closure")
RESULTS = osp.join(EXPERIMENT_ROOT, "029-solid-growth-ko/results")
SCREEN = {"TmiKuzmin2018Dataset": "kuzmin2018", "TmiKuzmin2020Dataset": "kuzmin2020"}

_ENV = None


def _init() -> None:
    global _ENV
    _ENV = lmdb.open(
        osp.join(BUILD, "processed", "lmdb"), readonly=True, lock=False, max_readers=256
    )


def _roles_chunk(idxs: list[int]) -> list[dict[str, object]]:
    """The query pair, the array gene and the query strain of each triple."""
    assert _ENV is not None
    out: list[dict[str, object]] = []
    with _ENV.begin() as txn:
        for i in idxs:
            raw = txn.get(str(i).encode())
            if raw is None:
                continue
            for item in json.loads(raw):
                experiment = item["experiment"]
                if experiment["experiment_type"] != "fitness":
                    continue
                roles = triple_roles(experiment["genotype"]["perturbations"])
                if roles is not None:
                    out.append(
                        {
                            "idx": i,
                            "array_gene": roles.array_gene,
                            "qi": roles.query_genes[0],
                            "qj": roles.query_genes[1],
                            "query_strain_id": roles.query_strain_id,
                        }
                    )
                break
    return out


def scan_roles(workers: int = 40) -> pd.DataFrame:
    cached = osp.join(CACHE, "triple_roles.parquet")
    if osp.exists(cached):
        return pd.read_parquet(cached)
    with open(osp.join(BUILD, "processed", "perturbation_count_index.json")) as f:
        triples = json.load(f)["3"]
    chunks = [triples[i : i + 2000] for i in range(0, len(triples), 2000)]
    rows: list[dict[str, object]] = []
    with Pool(workers, initializer=_init) as pool:
        for n, part in enumerate(pool.imap_unordered(_roles_chunk, chunks)):
            rows.extend(part)
            if n % 25 == 0:
                print(f"  roles: chunk {n}/{len(chunks)}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(cached, index=False)
    return frame


def _chosen(entries: pd.DataFrame, policy: LabelPolicy, label: str) -> pd.Series:
    """One value per gene set for ``label``, under ``policy``."""
    want = "fitness" if label == "fitness" else "gene interaction"
    sub = entries[entries["exp_type"] == want]
    values: dict[str, float] = {}
    for genes, group in sub.groupby("genes", sort=False):
        pool = [
            LabelEntry(
                source=src,
                label=label,
                value=float(v),
                sd=None if pd.isna(sd) else float(sd),
                n_samples=None if pd.isna(n) else int(n),
                p_value=None if pd.isna(p) else float(p),
            )
            for src, v, sd, n, p in zip(
                group["source"],
                group["value"],
                group["sd"],
                group["n_samples"],
                group["p"],
            )
        ]
        choice = policy.select(pool, label)
        if choice is not None:
            values[str(genes)] = choice.value
    return pd.Series(values, dtype=float)


def _stats(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return {"n": int(len(x))}
    lr = stats.linregress(x, y)
    return {
        "n": int(len(x)),
        "pearson": float(stats.pearsonr(x, y)[0]),
        "spearman": float(stats.spearmanr(x, y)[0]),
        "slope": float(lr.slope),
        "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
        "median_abs_residual": float(np.median(np.abs(x - y))),
    }


def main() -> None:
    from torchcell.data.label_policy import source_key

    os.makedirs(RESULTS, exist_ok=True)
    print("reading the entry cache ...", flush=True)
    entries = pd.read_parquet(osp.join(CACHE, "entries.parquet"))
    entries["source"] = [
        source_key(d, t) for d, t in zip(entries["dataset"], entries["temp"])
    ]

    print("recovering the query/array roles of every triple ...", flush=True)
    roles = scan_roles()
    print(f"  roles resolved for {len(roles):,} triples", flush=True)

    singles = entries[entries["order"] == 1]
    doubles = entries[entries["order"] == 2]
    triples = entries[entries["order"] == 3]

    rows = []
    for screen_dataset, screen in SCREEN.items():
        policy = LabelPolicy(name=f"kuzmin-first::{screen}").promoted_for_year(screen)
        print(f"selecting values under {policy.name} ...", flush=True)
        f_single = _chosen(singles, policy, "fitness")
        f_double = _chosen(doubles, policy, "fitness")
        eps_double = _chosen(doubles, policy, "gene_interaction")
        f_triple = _chosen(triples, policy, "fitness")

        stored = triples[
            (triples["exp_type"] == "gene interaction")
            & (triples["dataset"] == screen_dataset)
        ].merge(roles, on="idx", how="inner")
        if not len(stored):
            continue

        def pair(a: pd.Series, b: pd.Series) -> list[str]:
            return ["|".join(sorted((x, y))) for x, y in zip(a, b)]

        qpair = pair(stored["qi"], stored["qj"])
        ik = pair(stored["qi"], stored["array_gene"])
        jk = pair(stored["qj"], stored["array_gene"])

        f_ijk = f_triple.reindex(stored["genes"]).to_numpy()
        f_ij = f_double.reindex(qpair).to_numpy()
        f_k = f_single.reindex(stored["array_gene"]).to_numpy()
        eps_ik = eps_double.reindex(ik).to_numpy()
        eps_jk = eps_double.reindex(jk).to_numpy()

        # the published identity: the query singles enter as 1
        tau_asym = f_ijk - f_ij * f_k - eps_ik - eps_jk

        # the symmetric form on the same records, for a like-for-like comparison
        f_i = f_single.reindex(stored["qi"]).to_numpy()
        f_j = f_single.reindex(stored["qj"]).to_numpy()
        f_ik = f_double.reindex(ik).to_numpy()
        f_jk = f_double.reindex(jk).to_numpy()
        tau_sym = f_ijk - f_ij * f_k - f_ik * f_j - f_jk * f_i + 2.0 * f_i * f_j * f_k

        y = stored["value"].to_numpy()
        rows.append(
            {"screen": screen, "form": "asymmetric (published)", **_stats(y, tau_asym)}
        )
        rows.append({"screen": screen, "form": "symmetric", **_stats(y, tau_sym)})
        rows.append(
            {
                "screen": screen,
                "form": "asymmetric, array single set to 1",
                **_stats(y, f_ijk - f_ij - eps_ik - eps_jk),
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(osp.join(RESULTS, "closure_asymmetric_by_screen.csv"), index=False)
    with open(osp.join(RESULTS, "closure_asymmetric_summary.json"), "w") as f:
        json.dump(table.to_dict(orient="records"), f, indent=2)
    print()
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
