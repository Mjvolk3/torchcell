# experiments/031-env-chemgen-inhibitor-tolerance/scripts/dataset_axes_comparison.py
# [[experiments.031-env-chemgen-inhibitor-tolerance.scripts.dataset_axes_comparison]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-env-chemgen-inhibitor-tolerance/scripts/dataset_axes_comparison
"""Axis-by-axis comparison of the served Vanacloig 2022 and Hillenmeyer 2008 records.

Reads the flattened parquet files written by ``flatten_records.py`` and writes, under
``results/``:

- ``axes_table.md`` / ``axes_table.csv``: one row per cell/environment axis (background
  strain, ploidy, perturbation type, medium, temperature, aerobicity, duration, dose
  basis, readout, replicate structure, ...) with each dataset's distinct values.
- ``environment_catalog_<dataset>.csv``: one row per distinct environment (compound +
  dose + physical factor + temperature + duration + screen) with its record and gene
  counts.
- ``overlap.md`` / ``gene_overlap.csv`` / ``compound_overlap.csv``: gene-set and
  compound-set (InChIKey) overlaps between the datasets, and the doses each dataset
  used for every shared compound.

Every number in the dendron note comes from these files.
"""

from __future__ import annotations

import os
import os.path as osp
from collections.abc import Iterable
from itertools import combinations

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(
    EXPERIMENT_ROOT, "031-env-chemgen-inhibitor-tolerance", "results"
)

DATASETS = ["vanacloig2022", "hillenmeyer2008_hom", "hillenmeyer2008_het"]
LABELS = {
    "vanacloig2022": "Vanacloig 2022",
    "hillenmeyer2008_hom": "Hillenmeyer 2008 HOM",
    "hillenmeyer2008_het": "Hillenmeyer 2008 HET",
}
ENV_KEYS = [
    "compound",
    "inchikey",
    "dose_value",
    "dose_unit",
    "dose_basis",
    "solvent",
    "physical",
    "temperature_c",
    "duration_generations",
    "duration_hours",
    "media_name",
    "aerobicity",
    "screen_id",
]


def _md(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table (no tabulate dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                str(v).replace("|", "\\|").replace("\n", " ") for v in r.values
            )
            + " |"
        )
    return "\n".join(lines)


def _load() -> dict[str, pd.DataFrame]:
    """Read the three flattened parquet files."""
    return {
        n: pd.read_parquet(osp.join(RESULTS_DIR, f"records_{n}.parquet"))
        for n in DATASETS
    }


def _values(s: pd.Series, top: int = 6) -> str:
    """Distinct values of a column as a compact ``value (count)`` string."""
    vc = s.fillna("None").astype(str).value_counts()
    parts = [f"{v} ({c:,})" for v, c in vc.head(top).items()]
    if len(vc) > top:
        parts.append(f"... {len(vc)} distinct")
    return "; ".join(parts)


def _background_genes(df: pd.DataFrame) -> set[str]:
    """Genes present in EVERY genotype of the dataset (a constant background)."""
    sets = [set(g.split("|")) for g in df["gene"].unique()]
    return set.intersection(*sets) if len(sets) > 1 else set()


def _query_genes(df: pd.DataFrame, background: set[str]) -> pd.Series:
    """The per-record queried gene(s) with the constant background removed."""
    return df["gene"].map(
        lambda g: "|".join(x for x in g.split("|") if x not in background)
    )


def axes_table(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per axis, one column per dataset."""
    rows: list[dict[str, str]] = []

    def add(axis: str, fn) -> None:  # noqa: ANN001
        rows.append({"axis": axis, **{LABELS[n]: fn(df) for n, df in dfs.items()}})

    bg = {n: _background_genes(df) for n, df in dfs.items()}
    qg = {n: _query_genes(df, bg[n]) for n, df in dfs.items()}
    env_id = {
        n: df[ENV_KEYS].fillna("").astype(str).agg("|".join, axis=1)
        for n, df in dfs.items()
    }
    coarse = {
        n: df["compound"].where(
            df["compound"] != "",
            df["physical"].where(
                df["physical"] != "", "T=" + df["temperature_c"].astype(str)
            ),
        )
        for n, df in dfs.items()
    }

    add("records", lambda d: f"{len(d):,}")
    add("distinct genotypes (full strain)", lambda d: f"{d['gene'].nunique():,}")
    add(
        "constant background genes",
        lambda d: ", ".join(sorted(bg[d["dataset"].iat[0]])) or "none",
    )
    add("distinct queried genes", lambda d: f"{qg[d['dataset'].iat[0]].nunique():,}")
    add("genes per genotype", lambda d: _values(d["n_genes"]))
    add(
        "perturbation type",
        lambda d: _values(
            d["perturbation_type"].map(lambda s: "|".join(sorted(s.split("|"))))
        ),
    )
    add("reference strain", lambda d: _values(d["ref_strain"]))
    add("ploidy", lambda d: _values(d["ref_ploidy"]))
    add(
        "distinct environments (compound + dose + physical + T + duration + screen)",
        lambda d: f"{env_id[d['dataset'].iat[0]].nunique():,}",
    )
    add(
        "distinct conditions (compound / physical / temperature)",
        lambda d: f"{coarse[d['dataset'].iat[0]].nunique():,}",
    )
    add(
        "distinct compounds (InChIKey)",
        lambda d: f"{d.loc[d['inchikey'] != '', 'inchikey'].nunique():,}",
    )
    add(
        "records with no small molecule",
        lambda d: f"{(d['n_small_molecules'] == 0).sum():,}",
    )
    add("small molecules per record", lambda d: _values(d["n_small_molecules"]))
    add("dose basis", lambda d: _values(d["dose_basis"]))
    add("dose unit", lambda d: _values(d["dose_unit"]))
    add("solvent", lambda d: _values(d["solvent"]))
    add("physical factor", lambda d: _values(d["physical"]))
    add("medium", lambda d: _values(d["media_name"]))
    add("medium base", lambda d: _values(d["media_base"]))
    add("medium state (liquid / solid)", lambda d: _values(d["media_state"]))
    add("medium synthetic", lambda d: _values(d["media_synthetic"]))
    add("medium dropouts", lambda d: _values(d["media_dropouts"]))
    add("temperature (C)", lambda d: _values(d["temperature_c"]))
    add("aerobicity", lambda d: _values(d["aerobicity"]))
    add("duration (hours)", lambda d: _values(d["duration_hours"]))
    add("duration (generations)", lambda d: _values(d["duration_generations"]))
    add("measurement type", lambda d: _values(d["measurement_type"]))
    add("assay type", lambda d: _values(d["assay_type"]))
    add("n_samples", lambda d: _values(d["n_samples"]))
    add("sample unit", lambda d: _values(d["sample_unit"]))
    add("uncertainty type", lambda d: _values(d["uncertainty_type"]))
    add("distinct screens (screen_id)", lambda d: f"{d['screen_id'].nunique():,}")
    add(
        "response median [5th, 95th pct]",
        lambda d: (
            f"{d['response'].median():.3f} [{d['response'].quantile(0.05):.3f}, {d['response'].quantile(0.95):.3f}]"
        ),
    )
    add(
        "response SE median",
        lambda d: (
            "None"
            if d["response_se"].isna().all()
            else f"{d['response_se'].median():.3f}"
        ),
    )
    add("reference response", lambda d: _values(d["ref_response"]))
    add(
        "reference medium == experiment medium",
        lambda d: _values(d["media_name"] == d["ref_media_name"]),
    )
    add("environment provenance gaps per record", lambda d: _values(d["n_gaps"]))
    return pd.DataFrame(rows)


def environment_catalog(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct environment with record and gene counts."""
    g = df.groupby(ENV_KEYS, dropna=False)
    cat = g.agg(
        records=("idx", "size"),
        genes=("gene", "nunique"),
        response_median=("response", "median"),
    ).reset_index()
    cat.insert(0, "dataset", name)
    return cat.sort_values("records", ascending=False)


def overlaps(
    dfs: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Gene-set and compound-set overlaps, plus per-shared-compound dose tables."""
    bg = {n: _background_genes(df) for n, df in dfs.items()}
    genes = {
        n: set(_query_genes(df, bg[n]).str.split("|").explode().unique()) - {""}
        for n, df in dfs.items()
    }
    keys = {
        n: set(
            df.loc[df["inchikey"] != "", "inchikey"].str.split("|").explode().unique()
        )
        for n, df in dfs.items()
    }

    def pair_rows(sets: dict[str, set[str]], what: str) -> Iterable[dict[str, object]]:
        for a, b in combinations(sets, 2):
            inter = sets[a] & sets[b]
            yield {
                "what": what,
                "a": LABELS[a],
                "b": LABELS[b],
                "n_a": len(sets[a]),
                "n_b": len(sets[b]),
                "shared": len(inter),
                "jaccard": len(inter) / len(sets[a] | sets[b]),
            }
        yield {
            "what": what,
            "a": "all three",
            "b": "",
            "n_a": "",
            "n_b": "",
            "shared": len(set.intersection(*sets.values())),
            "jaccard": "",
        }

    gene_df = pd.DataFrame(list(pair_rows(genes, "queried genes")))
    comp_df = pd.DataFrame(list(pair_rows(keys, "compounds (InChIKey)")))

    name_of = (
        pd.concat(
            [
                df.loc[df["inchikey"] != "", ["inchikey", "compound"]]
                for df in dfs.values()
            ]
        )
        .drop_duplicates("inchikey")
        .set_index("inchikey")["compound"]
    )
    rows = []
    for a, b in combinations(dfs, 2):
        if "vanacloig2022" not in (a, b):
            continue  # the HOM vs HET overlap is counted above; its 99 rows are not the question
        for k in sorted(keys[a] & keys[b]):
            row: dict[str, object] = {
                "inchikey": k,
                "compound": name_of[k],
                "a": LABELS[a],
                "b": LABELS[b],
            }
            for side, n in (("a", a), ("b", b)):
                sub = dfs[n][dfs[n]["inchikey"] == k]
                doses = sorted(
                    {
                        f"{v} {u}".strip() if v else (bs or "?")
                        for v, u, bs in zip(
                            sub["dose_value"],
                            sub["dose_unit"],
                            sub["dose_basis"],
                            strict=True,
                        )
                    }
                )
                row[f"{side}_records"] = len(sub)
                row[f"{side}_doses"] = "; ".join(doses)
                row[f"{side}_genes"] = sub["gene"].nunique()
            rows.append(row)
    shared_df = pd.DataFrame(rows)
    return gene_df, comp_df, shared_df


def main() -> None:
    """Write every table under ``results/``."""
    dfs = _load()
    axes = axes_table(dfs)
    axes.to_csv(osp.join(RESULTS_DIR, "axes_table.csv"), index=False)
    with open(osp.join(RESULTS_DIR, "axes_table.md"), "w") as f:
        f.write(axes.pipe(_md))
    for n, df in dfs.items():
        environment_catalog(n, df).to_csv(
            osp.join(RESULTS_DIR, f"environment_catalog_{n}.csv"), index=False
        )
    gene_df, comp_df, shared_df = overlaps(dfs)
    gene_df.to_csv(osp.join(RESULTS_DIR, "gene_overlap.csv"), index=False)
    comp_df.to_csv(osp.join(RESULTS_DIR, "compound_overlap.csv"), index=False)
    shared_df.to_csv(osp.join(RESULTS_DIR, "shared_compounds.csv"), index=False)
    with open(osp.join(RESULTS_DIR, "overlap.md"), "w") as f:
        f.write("## Queried-gene overlap\n\n" + gene_df.pipe(_md) + "\n\n")
        f.write("## Compound overlap (InChIKey)\n\n" + comp_df.pipe(_md) + "\n\n")
        f.write(
            "## Shared compounds and the doses each dataset used\n\n"
            + shared_df.pipe(_md)
            + "\n"
        )
    print(axes.pipe(_md))
    print()
    print(gene_df.pipe(_md))
    print()
    print(comp_df.pipe(_md))
    print()
    print(shared_df.pipe(_md))


if __name__ == "__main__":
    main()
