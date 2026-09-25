# experiments/030-solid-growth-multi/scripts/closure_recompute_030.py
# [[experiments.030-solid-growth-multi.scripts.closure_recompute_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/closure_recompute_030

"""Recompute the trigenic scores of the 030 build from its own fitness, beside 025 and 029.

The 030 build is the 025 query (every allele) on the 029 store shape (no merge, every
source entry kept) plus the record both lacked: the double-mutant QUERY strain's own
fitness, which the Kuzmin Dmf loaders emit as of main 4c4a4f950 with the query strain's
``GENE1+GENE2_tmNNNN`` identifier. On 029 the published identity

    tau = f_ijk - f_ij f_k - eps_ik - eps_jk

reached r 0.511 (Kuzmin 2018) and 0.325 (2020) with f_ij taken from the pair's digenic
ARRAY screen, and the within-screen recompute on the raw tables showed that this one
term is the whole gap (0.985 / 0.976 with the query strain's fitness, 0.538 / 0.419
without it). This script asks whether the build now closes that gap: the same identity,
the same Kuzmin-first policy, with f_ij chosen by ``LabelPolicy.select_double``, which
prefers the entry whose strain token equals the triple's query strain, and beside it
the 029 reading with the query-strain entries excluded, on identical records.

Stages (cache under $DATA_ROOT/data/torchcell/experiments/030-solid-growth-multi/closure/):

    python .../closure_recompute_030.py scan     # build -> entries.parquet, triple_roles.parquet
    python .../closure_recompute_030.py analyze  # results, table

``--build`` and ``--cache`` point the same two stages at another build (the 029 store is
the smoke test: it carries no query-strain double, so the strain-matched reading must
equal the array-screen one and reproduce closure_recompute_asymmetric.py's 0.511).
``--limit-triples N`` scans the first N triples and the doubles inside them only.

Outputs (experiments/030-solid-growth-multi/results/):
- closure_030_by_screen.csv       one row per screen x stratum x form
- closure_030_summary.json        the rows, the coverage counts, the reference rows of
                                  025 and 029 read from their result files (path + sha256)
- t10-030-closure.tex             the comparison table for notes-tex/025-s3-closure
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
from collections.abc import Callable
from multiprocessing import Pool
from typing import Any

import lmdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats

from torchcell.data.label_policy import (
    LabelEntry,
    LabelPolicy,
    source_key,
    strain_token,
)
from torchcell.data.label_table import triple_roles

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

DEFAULT_BUILD = "/db/experiments/030-solid-growth-multi-001-multi-build"
DEFAULT_CACHE = osp.join(
    DATA_ROOT, "data/torchcell/experiments/030-solid-growth-multi/closure"
)
RESULTS = osp.join(EXPERIMENT_ROOT, "030-solid-growth-multi/results")
SCREENS = {"TmiKuzmin2018Dataset": "kuzmin2018", "TmiKuzmin2020Dataset": "kuzmin2020"}
ENTRY_COLS = [
    "idx",
    "order",
    "genes",
    "dataset",
    "exp_type",
    "temp",
    "marker",
    "value",
    "sd",
    "n_samples",
    "p",
    "strain_id",
]
DELETION = "deletion"

_ENV: lmdb.Environment | None = None
_BUILD = ""


def _init(build: str) -> None:
    global _ENV, _BUILD
    _BUILD = build
    _ENV = lmdb.open(
        osp.join(build, "processed", "lmdb"), readonly=True, lock=False, max_readers=256
    )


def _entries(idx: int, raw: bytes) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for item in json.loads(raw):
        e = item["experiment"]
        ph = e["phenotype"]
        perts = sorted(
            e["genotype"]["perturbations"], key=lambda p: p["systematic_gene_name"]
        )
        genes = "|".join(p["systematic_gene_name"] for p in perts)
        marker = "|".join(
            DELETION if "deletion" in p["perturbation_type"] else p["perturbation_type"]
            for p in perts
        )
        temp = float(e["environment"]["temperature"]["value"])
        strain_ids = {p["strain_id"] for p in perts if p.get("strain_id")}
        strain_id = strain_ids.pop() if len(strain_ids) == 1 else None
        if e["experiment_type"] == "fitness":
            rows.append(
                (
                    idx,
                    len(perts),
                    genes,
                    e["dataset_name"],
                    "fitness",
                    temp,
                    marker,
                    ph["fitness"],
                    ph.get("fitness_std"),
                    ph.get("n_samples"),
                    None,
                    strain_id,
                )
            )
        elif e["experiment_type"] == "gene interaction":
            rows.append(
                (
                    idx,
                    len(perts),
                    genes,
                    e["dataset_name"],
                    "gene interaction",
                    temp,
                    marker,
                    ph["gene_interaction"],
                    None,
                    None,
                    ph.get("gene_interaction_p_value"),
                    strain_id,
                )
            )
    return rows


def _genes_of(raw: bytes) -> str:
    first = json.loads(raw)[0]["experiment"]["genotype"]["perturbations"]
    return "|".join(sorted(p["systematic_gene_name"] for p in first))


def _keys_chunk(idxs: list[int]) -> list[tuple[int, str]]:
    assert _ENV is not None
    with _ENV.begin() as txn:
        return [(i, _genes_of(txn.get(str(i).encode()))) for i in idxs]


def _entries_chunk(idxs: list[int]) -> list[tuple[Any, ...]]:
    assert _ENV is not None
    out: list[tuple[Any, ...]] = []
    with _ENV.begin() as txn:
        for i in idxs:
            out.extend(_entries(i, txn.get(str(i).encode())))
    return out


def _roles_chunk(idxs: list[int]) -> list[dict[str, Any]]:
    """The query pair, the array gene and the query strain of each triple."""
    assert _ENV is not None
    out: list[dict[str, Any]] = []
    with _ENV.begin() as txn:
        for i in idxs:
            for item in json.loads(txn.get(str(i).encode())):
                e = item["experiment"]
                if e["experiment_type"] != "fitness":
                    continue
                roles = triple_roles(e["genotype"]["perturbations"])
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


def _run(
    fn: Callable[[list[int]], list[Any]],
    idxs: list[int],
    label: str,
    build: str,
    workers: int,
    chunk: int = 2000,
) -> list[Any]:
    chunks = [idxs[i : i + chunk] for i in range(0, len(idxs), chunk)]
    rows: list[Any] = []
    with Pool(workers, initializer=_init, initargs=(build,)) as pool:
        for n, part in enumerate(pool.imap_unordered(fn, chunks, chunksize=1)):
            rows.extend(part)
            if n % 200 == 0:
                print(f"  {label}: chunks {n}/{len(chunks)}", flush=True)
    return rows


def scan(build: str, cache: str, workers: int, limit_triples: int | None) -> None:
    os.makedirs(cache, exist_ok=True)
    with open(osp.join(build, "processed", "perturbation_count_index.json")) as f:
        by_order = json.load(f)
    print({k: len(v) for k, v in by_order.items()}, flush=True)
    triple_idx = by_order["3"][:limit_triples] if limit_triples else by_order["3"]

    triples = pd.DataFrame(
        _run(_entries_chunk, triple_idx, "triples", build, workers), columns=ENTRY_COLS
    )
    roles = pd.DataFrame(_run(_roles_chunk, triple_idx, "roles", build, workers))
    print(
        f"triple gene sets {triples['genes'].nunique():,}; roles resolved {len(roles):,} of {len(triple_idx):,}",
        flush=True,
    )

    pairs: set[str] = set()
    for g in triples["genes"].unique():
        a, b, c = g.split("|")
        pairs.update({f"{a}|{b}", f"{a}|{c}", f"{b}|{c}"})
    keys = _run(_keys_chunk, by_order["2"], "double keys", build, workers)
    keep = [i for i, k in keys if k in pairs]
    print(f"doubles in the closure: {len(keep):,} of {len(keys):,}", flush=True)
    doubles = pd.DataFrame(
        _run(_entries_chunk, keep, "closure doubles", build, workers),
        columns=ENTRY_COLS,
    )
    singles = pd.DataFrame(
        _run(_entries_chunk, by_order["1"], "singles", build, workers),
        columns=ENTRY_COLS,
    )

    df = pd.concat([singles, doubles, triples], ignore_index=True)
    df.to_parquet(osp.join(cache, "entries.parquet"), index=False)
    roles.to_parquet(osp.join(cache, "triple_roles.parquet"), index=False)
    print(df.groupby(["order", "exp_type"]).size().to_string(), flush=True)
    print(df.groupby(["order", "dataset"]).size().to_string(), flush=True)
    n_query_doubles = int(
        (
            (df["order"] == 2)
            & df["strain_id"].map(lambda s: strain_token(s) is not None)
        ).sum()
    )
    print(
        f"double entries carrying a query strain token: {n_query_doubles:,}", flush=True
    )


# --------------------------------------------------------------------------- analyze
def _label_entries(frame: pd.DataFrame, label: str) -> dict[str, list[LabelEntry]]:
    """LabelEntry objects of every gene set in ``frame``, for one label."""
    want = "fitness" if label == "fitness" else "gene interaction"
    sub = frame[frame["exp_type"] == want]
    out: dict[str, list[LabelEntry]] = {}
    for genes, src, v, sd, n, p, sid in zip(
        sub["genes"],
        sub["source"],
        sub["value"],
        sub["sd"],
        sub["n_samples"],
        sub["p"],
        sub["strain_id"],
    ):
        out.setdefault(str(genes), []).append(
            LabelEntry(
                source=src,
                label=label,
                value=float(v),
                sd=None if pd.isna(sd) else float(sd),
                n_samples=None if pd.isna(n) else int(n),
                p_value=None if pd.isna(p) else float(p),
                strain_id=None
                if sid is None or (isinstance(sid, float) and np.isnan(sid))
                else str(sid),
            )
        )
    return out


def _chosen(
    pools: dict[str, list[LabelEntry]], policy: LabelPolicy, label: str
) -> pd.Series:
    values = {
        g: c.value
        for g, pool in pools.items()
        if (c := policy.select(pool, label)) is not None
    }
    return pd.Series(values, dtype=float)


def _arr(series: pd.Series, keys: Any) -> np.ndarray:
    """``series`` looked up at ``keys``, NaN where a key is missing, as float64."""
    return series.reindex(keys).to_numpy(dtype=float)


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


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _reference_rows(
    ref_025: str, ref_029: str
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """The published numbers of the two earlier builds, read from their result files."""
    with open(ref_025) as f:
        s025 = json.load(f)
    with open(ref_029) as f:
        s029 = json.load(f)
    rows: list[dict[str, Any]] = []
    tri = s025["trigenic_strength"]["all"]
    rows.append(
        {
            "build": "025",
            "screen": "both",
            "stratum": "all",
            "form": "symmetric, mean over entries",
            **{
                k: tri[k]
                for k in (
                    "n",
                    "pearson",
                    "spearman",
                    "slope",
                    "rmse",
                    "median_abs_residual",
                )
            },
        }
    )
    for r in s029:
        if r["form"] == "asymmetric (published)":
            rows.append(
                {
                    "build": "029",
                    "screen": r["screen"],
                    "stratum": "deletion",
                    "form": "asymmetric, array-screen double",
                    **{
                        k: r[k]
                        for k in (
                            "n",
                            "pearson",
                            "spearman",
                            "slope",
                            "rmse",
                            "median_abs_residual",
                        )
                    },
                }
            )
        elif r["form"] == "symmetric":
            rows.append(
                {
                    "build": "029",
                    "screen": r["screen"],
                    "stratum": "deletion",
                    "form": "symmetric",
                    **{
                        k: r[k]
                        for k in (
                            "n",
                            "pearson",
                            "spearman",
                            "slope",
                            "rmse",
                            "median_abs_residual",
                        )
                    },
                }
            )
    provenance = {
        "reference_025": ref_025,
        "reference_025_sha256": _sha256(ref_025),
        "reference_029": ref_029,
        "reference_029_sha256": _sha256(ref_029),
    }
    return rows, provenance


def analyze(cache: str, results: str, ref_025: str, ref_029: str) -> None:
    os.makedirs(results, exist_ok=True)
    print("reading the entry cache ...", flush=True)
    entries = pd.read_parquet(osp.join(cache, "entries.parquet"))
    entries["source"] = [
        source_key(d, t) for d, t in zip(entries["dataset"], entries["temp"])
    ]
    roles = pd.read_parquet(osp.join(cache, "triple_roles.parquet"))

    singles = entries[entries["order"] == 1]
    doubles = entries[entries["order"] == 2]
    triples = entries[entries["order"] == 3]
    double_fit = _label_entries(doubles, "fitness")
    double_gi = _label_entries(doubles, "gene_interaction")
    single_fit = _label_entries(singles, "fitness")
    triple_fit = _label_entries(triples, "fitness")
    # the array-screen reading: the same pools without the query-strain entries (029)
    double_fit_array = {
        g: [e for e in pool if strain_token(e.strain_id) is None]
        for g, pool in double_fit.items()
    }
    triple_marker = triples.drop_duplicates("idx").set_index("idx")["marker"]

    rows: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {}
    for screen_dataset, screen in SCREENS.items():
        policy = LabelPolicy(name=f"kuzmin-first::{screen}").promoted_for_year(screen)
        print(
            f"selecting values under {policy.name} ({policy.policy_id}) ...", flush=True
        )
        f_single = _chosen(single_fit, policy, "fitness")
        eps_double = _chosen(double_gi, policy, "gene_interaction")
        f_double_array = _chosen(double_fit_array, policy, "fitness")
        f_triple = _chosen(triple_fit, policy, "fitness")

        stored = triples[
            (triples["exp_type"] == "gene interaction")
            & (triples["dataset"] == screen_dataset)
        ]
        stored = stored.merge(roles, on="idx", how="inner")
        if not len(stored):
            continue

        def pair(a: pd.Series, b: pd.Series) -> list[str]:
            return ["|".join(sorted((x, y))) for x, y in zip(a, b)]

        qpair = pair(stored["qi"], stored["qj"])
        ik = pair(stored["qi"], stored["array_gene"])
        jk = pair(stored["qj"], stored["array_gene"])

        # f_ij from the double-mutant query strain the screen used, where the build
        # carries it (select_double), and the 029 reading beside it
        f_ij_matched = np.full(len(stored), np.nan)
        matched = np.zeros(len(stored), dtype=bool)
        for n, (pk, sid) in enumerate(zip(qpair, stored["query_strain_id"])):
            pool = double_fit.get(pk)
            if not pool:
                continue
            token = strain_token(sid)
            matched[n] = any(strain_token(e.strain_id) == token for e in pool if token)
            choice = policy.select_double(pool, sid, "fitness")
            if choice is not None:
                f_ij_matched[n] = choice.value
        f_ij_array = _arr(f_double_array, qpair)

        f_ijk = _arr(f_triple, stored["genes"])
        f_k = _arr(f_single, stored["array_gene"])
        eps_ik = _arr(eps_double, ik)
        eps_jk = _arr(eps_double, jk)
        f_i = _arr(f_single, stored["qi"])
        f_j = _arr(f_single, stored["qj"])
        f_ik = _arr(f_double_array, ik)
        f_jk = _arr(f_double_array, jk)
        y = stored["value"].to_numpy(dtype=float)

        forms = {
            "asymmetric, query-strain double": f_ijk
            - f_ij_matched * f_k
            - eps_ik
            - eps_jk,
            "asymmetric, array-screen double": f_ijk
            - f_ij_array * f_k
            - eps_ik
            - eps_jk,
            "symmetric": f_ijk
            - f_ij_array * f_k
            - f_ik * f_j
            - f_jk * f_i
            + 2.0 * f_i * f_j * f_k,
        }
        markers = triple_marker.reindex(stored["idx"]).to_numpy()
        strata = {
            "all": np.ones(len(stored), dtype=bool),
            "deletion": np.array([m == "|".join([DELETION] * 3) for m in markers]),
            "query double in build": matched,
        }
        coverage[screen] = {
            "n_stored": int(len(stored)),
            "n_roles_resolved": int(len(stored)),
            "n_deletion_only": int(strata["deletion"].sum()),
            "n_query_double_in_build": int(matched.sum()),
            "n_query_double_and_deletion": int((matched & strata["deletion"]).sum()),
            "policy_id": policy.policy_id,
        }
        for stratum, sel in strata.items():
            for form, tau in forms.items():
                rows.append(
                    {
                        "build": "030",
                        "screen": screen,
                        "stratum": stratum,
                        "form": form,
                        **_stats(y[sel], tau[sel]),
                    }
                )

    reference, provenance = _reference_rows(ref_025, ref_029)
    table = pd.DataFrame(reference + rows)
    table.to_csv(osp.join(results, "closure_030_by_screen.csv"), index=False)
    summary = {
        "rows": table.to_dict(orient="records"),
        "coverage": coverage,
        **provenance,
    }
    with open(osp.join(results, "closure_030_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _write_table(table, osp.join(results, "t10-030-closure.tex"))
    print()
    print(table.to_string(index=False))
    print(json.dumps(coverage, indent=2))


def _write_table(table: pd.DataFrame, path: str) -> None:
    """The comparison table in the style of notes-tex/025-s3-closure/tables/t9-asymmetric.tex."""
    lines = [
        "%% SOURCE: experiments/030-solid-growth-multi/scripts/closure_recompute_030.py "
        "(results/closure_030_by_screen.csv) -- GENERATED, do not edit",
        "",
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"build & screen, stratum & form & $n$ & $r$ & slope & rmse \\",
        r"\midrule",
    ]
    for (build, screen, stratum), group in table.groupby(
        ["build", "screen", "stratum"], sort=False
    ):
        label = f"{screen} {stratum}" if stratum != "all" else screen
        best = group["pearson"].max()
        for k, (_, r) in enumerate(group.iterrows()):
            if pd.isna(r.get("pearson")):
                continue
            rr = f"{r['pearson']:.3f}"
            rr = rf"\textbf{{{rr}}}" if r["pearson"] == best else rr
            first = (
                f"{build} & {label.replace('kuzmin', 'Kuzmin ')}" if k == 0 else " & "
            )
            lines.append(
                f"{first} & {r['form']} & {int(r['n']):,} & {rr} & {r['slope']:.2f} & {r['rmse']:.3f} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("stage", choices=["scan", "analyze"])
    ap.add_argument("--build", default=DEFAULT_BUILD)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit-triples", type=int, default=None)
    ap.add_argument(
        "--reference-025",
        default=osp.join(
            EXPERIMENT_ROOT,
            "025-solid-growth/results/s3_closure_recompute_summary.json",
        ),
    )
    ap.add_argument(
        "--reference-029",
        default=osp.join(
            EXPERIMENT_ROOT,
            "029-solid-growth-ko/results/closure_asymmetric_summary.json",
        ),
    )
    args = ap.parse_args()
    if args.stage == "scan":
        scan(args.build, args.cache, args.workers, args.limit_triples)
    else:
        analyze(args.cache, args.results, args.reference_025, args.reference_029)


if __name__ == "__main__":
    main()
