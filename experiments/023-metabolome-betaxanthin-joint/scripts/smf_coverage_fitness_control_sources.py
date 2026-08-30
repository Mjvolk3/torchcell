# experiments/023-metabolome-betaxanthin-joint/scripts/smf_coverage_fitness_control_sources.py
# [[experiments.023-metabolome-betaxanthin-joint.scripts.smf_coverage_fitness_control_sources]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/023-metabolome-betaxanthin-joint/scripts/smf_coverage_fitness_control_sources
"""Can a second single-mutant-fitness source shrink the 724 uncontrolled deletions?

THE CHALLENGE. `betaxanthin_amino_acid_predictivity.py` controls the betaxanthin /
amino-acid coupling for growth using Costanzo 2016 single-mutant fitness (SMF), KanMX
deletions at 30 C. That control covers 3,708 of the 4,432 deletions the Cachera and
Mulleder screens share, so 724 deletions carry no control, and those 724 are exactly where
the amino-acid profile predicts best. The question asked of this script is whether ANOTHER
fitness source in the repo covers those 724, since a wider control would let the
fitness-residualized fit run on more of the panel.

WHAT COUNTS AS A SOURCE. A usable source must yield a per-single-deletion fitness SCALAR
under a standard growth condition. Three classes of near-miss are excluded and named in the
inventory the script writes: `environment_response` screens (every chemogenomic dataset in
the repo scores growth under a named stress relative to its own control, so there is no
absolute unstressed fitness to regress out), binary essentiality, and double/triple mutant
fitness. The one deliberate borderline inclusion is the array-SMF COLUMN carried inside the
Kuzmin double-mutant tables, which is a single-deletion fitness even though the repo does
not ontologize it as an SMF record; it is measured here because the author's question was
specifically about Kuzmin SMF coverage.

SCALE. The sources do not share a normalization (Baryshnikova sets the fitness-distribution
MODE to 1, Costanzo normalizes to the wild-type control, O'Duibhir reports a relative growth
rate). A union control is therefore built by fitting each donor source onto the Costanzo
scale by least squares on the genes both cover, and the r of that map is reported so the
quality of the transfer is visible rather than assumed. Costanzo's own value is always kept
where it exists; a donor only fills a hole.

Sources read (all already built on disk; no build is triggered):
  $DATA_ROOT/data/torchcell/betaxanthin_cachera2023/preprocess/data.csv
  $DATA_ROOT/data/torchcell/amino_acid_mulleder2016/preprocess/data.csv
  $DATA_ROOT/data/torchcell/smf_costanzo2016/preprocess/data.csv
  $DATA_ROOT/data/torchcell/smf_kuzmin2018/preprocess/data.csv
  $DATA_ROOT/data/torchcell/smf_kuzmin2020/preprocess/data.csv
  $DATA_ROOT/data/torchcell/dmf_kuzmin2018/preprocess/data.csv   (array-SMF column only)
  $DATA_ROOT/data/torchcell/dmf_kuzmin2020/preprocess/data.csv   (array-SMF column only)
  $DATA_ROOT/data/torchcell/smf_baryshnikova2010/processed/lmdb
  $DATA_ROOT/data/torchcell/smf_oduibhir2014/processed/lmdb

Run from repo root:
  PYTHONPATH=. python experiments/023-metabolome-betaxanthin-joint/scripts/smf_coverage_fitness_control_sources.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import pickle
from typing import Any

import lmdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from torchcell.timestamp import timestamp

EXPERIMENT = "023-metabolome-betaxanthin-joint"

# Matched to betaxanthin_amino_acid_predictivity.py so the recomputed numbers are
# comparable to the ones the document quotes rather than merely similar. The penalty grid
# is EXTENDED past that script's [1e-2, 1e4] at the same 10^0.25 spacing, so the incumbent
# grid is a strict SUBSET of this one and every fit whose optimum was interior before
# selects the identical alpha now. The extension is needed because the pooled union
# control's one-feature fit wants unbounded shrinkage, which is a property of that control
# rather than of the grid (see ALPHA_EDGE handling in _cv_score).
ALPHAS = np.logspace(-2, 8, 41)
N_FOLDS = 5
CV_SEEDS = (0, 1, 2)

# Donor priority for the union control. Costanzo KanMX at 30 C is the incumbent and always
# wins; the rest are ordered by how many of the 4,432 shared deletions they cover.
UNION_PRIORITY = (
    "costanzo2016_kanmx_30c",
    "costanzo2016_natmx_30c",
    "baryshnikova2010_kanmx",
    "kuzmin2018_dmf_array_smf",
    "kuzmin2020_dmf_array_smf",
    "kuzmin2018_smf",
    "kuzmin2020_smf",
    "oduibhir2014",
)


def _z(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return (a - a.mean(axis=0)) / a.std(axis=0)


def _residualize(y: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Residual of ``y`` after least-squares removal of ``c`` plus an intercept."""
    design = np.hstack([np.ones((len(c), 1)), c])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return y - design @ beta


def _cv_score(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Out-of-fold Pearson/Spearman of a ridge fit, matching the incumbent script.

    The three spreads are the same three that script emits and mean the same things:
    ``pearson_sd_across_shuffles`` is fold-assignment noise at fixed n, and is the
    smallest; ``pearson_sd_across_folds`` is the spread of a one-fifth-sample estimate;
    ``pearson_se_fisher`` is the analytic 1/sqrt(n-3) standard error and is the one to read
    when comparing two fits at different n, which is exactly what this script does.
    """
    scores = []
    fold_r: list[float] = []
    alphas_chosen = []
    for seed in CV_SEEDS:
        folds = KFold(N_FOLDS, shuffle=True, random_state=seed)
        oof = np.zeros_like(y)
        for train_idx, test_idx in folds.split(x):
            model = RidgeCV(alphas=ALPHAS).fit(x[train_idx], y[train_idx])
            alphas_chosen.append(float(model.alpha_))
            oof[test_idx] = model.predict(x[test_idx])
            fold_r.append(float(pearsonr(oof[test_idx], y[test_idx])[0]))
        scores.append((pearsonr(oof, y)[0], spearmanr(oof, y)[0]))
    arr = np.asarray(scores)
    lo, hi = float(ALPHAS[0]), float(ALPHAS[-1])
    # The incumbent script RAISES on an endpoint alpha, which is right when an endpoint
    # means the grid is too narrow for a real signal. Here one design reaches the ceiling
    # for the opposite reason: the pooled union control has almost no out-of-fold value, so
    # ridge asks for unbounded shrinkage and no finite grid ends the search. Widening
    # forever would not fix that, so the selection is RECORDED instead. A ceiling flag on a
    # one-feature control is the finding, not a configuration error; a ceiling flag on the
    # 19-feature designs would be, and none of them raise it.
    alpha_at_edge = bool(min(alphas_chosen) <= lo or max(alphas_chosen) >= hi)
    r_mean = float(arr[:, 0].mean())
    n = int(len(y))
    z = np.arctanh(np.clip(r_mean, -0.999999, 0.999999))
    se_z = 1.0 / np.sqrt(n - 3)
    se_fisher = float((np.tanh(z + se_z) - np.tanh(z - se_z)) / 2.0)
    return {
        "pearson": r_mean,
        "pearson_sd_across_shuffles": float(arr[:, 0].std(ddof=1)),
        "pearson_sd_across_folds": float(np.std(fold_r, ddof=1)),
        "pearson_se_fisher": se_fisher,
        "spearman": float(arr[:, 1].mean()),
        "spearman_sd_across_shuffles": float(arr[:, 1].std(ddof=1)),
        "n": n,
        "n_features": int(x.shape[1]),
        "n_folds_scored": len(fold_r),
        "alpha_min_selected": float(min(alphas_chosen)),
        "alpha_max_selected": float(max(alphas_chosen)),
        "alpha_at_grid_edge": alpha_at_edge,
    }


def _lmdb_fitness(path: str, keep_perturbation_types: set[str] | None) -> pd.Series:
    """Per-ORF mean fitness from a built experiment LMDB, no dataset class needed.

    Reading the store directly avoids instantiating the loader, which for Baryshnikova
    would pull in a genome for ORF resolution. The stored records already carry the
    resolved systematic name, so the resolution has happened once, at build time.
    """
    values: dict[str, list[float]] = {}
    env = lmdb.open(path, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        for _, blob in txn.cursor():
            record = pickle.loads(blob)["experiment"]
            perturbations = record["genotype"]["perturbations"]
            if len(perturbations) != 1:
                continue
            perturbation = perturbations[0]
            if (
                keep_perturbation_types is not None
                and perturbation["perturbation_type"] not in keep_perturbation_types
            ):
                continue
            values.setdefault(perturbation["systematic_gene_name"], []).append(
                float(record["phenotype"]["fitness"])
            )
    env.close()
    return pd.Series({k: float(np.mean(v)) for k, v in values.items()}, dtype=float)


def _mean_by(df: pd.DataFrame, gene_col: str, value_col: str) -> pd.Series:
    sub = df[[gene_col, value_col]].dropna()
    return sub.groupby(gene_col)[value_col].mean()


def load_sources(data_root: str) -> dict[str, pd.Series]:
    """Every per-single-deletion fitness scalar the repo carries, keyed by systematic ORF."""
    tc = osp.join(data_root, "data", "torchcell")
    sources: dict[str, pd.Series] = {}

    costanzo = pd.read_csv(osp.join(tc, "smf_costanzo2016", "preprocess", "data.csv"))
    # The incumbent control: the array arm of the SGA deletion collection at 30 C.
    kanmx = costanzo[
        (costanzo["perturbation_type"] == "KanMX_deletion")
        & (costanzo["Temperature"] == 30)
    ]
    sources["costanzo2016_kanmx_30c"] = _mean_by(
        kanmx, "Systematic gene name", "Single mutant fitness"
    )
    # The NatMX arm is the SAME deletion carried as an SGA query strain. The incumbent
    # script excludes it to avoid pooling two perturbation classes, which is right when
    # both arms exist; where only the NatMX arm exists it is still a single-deletion
    # fitness for that gene, so it is measured here as a candidate gap filler.
    natmx = costanzo[
        (costanzo["perturbation_type"] == "NatMX_deletion")
        & (costanzo["Temperature"] == 30)
    ]
    sources["costanzo2016_natmx_30c"] = _mean_by(
        natmx, "Systematic gene name", "Single mutant fitness"
    )

    k18 = pd.read_csv(osp.join(tc, "smf_kuzmin2018", "preprocess", "data.csv"))
    array18 = k18[
        (k18["smf_type"] == "array_smf")
        & (k18["array_perturbation_type"] == "KanMX_deletion")
    ]
    query18 = k18[
        (k18["smf_type"] == "query_smf")
        & (k18["query_perturbation_type"] == "KanMX_deletion")
    ]
    sources["kuzmin2018_smf"] = pd.concat(
        [
            _mean_by(array18, "Array systematic name", "Array single mutant fitness"),
            _mean_by(
                query18,
                "Query systematic name no ho",
                "Query single/double mutant fitness",
            ),
        ]
    ).groupby(level=0).mean()

    k20 = pd.read_csv(osp.join(tc, "smf_kuzmin2020", "preprocess", "data.csv"))
    # Kuzmin 2020 releases its single mutants as a query-strain table whose two ORF columns
    # are the HO marker locus and the real deletion; both are recorded so the digenic query
    # is reconstructable, and the fitness is the strain's, so it applies to both entries.
    singles = k20[k20["Mutant type"] == "Single mutant"]
    sources["kuzmin2020_smf"] = pd.concat(
        [_mean_by(singles, "ORF1", "Fitness"), _mean_by(singles, "ORF2", "Fitness")]
    ).groupby(level=0).mean()

    for year in ("2018", "2020"):
        dmf = pd.read_csv(
            osp.join(tc, f"dmf_kuzmin{year}", "preprocess", "data.csv"),
            usecols=[
                "Array systematic name",
                "Array single mutant fitness",
                "array_perturbation_type",
            ],
        )
        dmf = dmf[dmf["array_perturbation_type"] == "KanMX_deletion"]
        sources[f"kuzmin{year}_dmf_array_smf"] = _mean_by(
            dmf, "Array systematic name", "Array single mutant fitness"
        )

    sources["baryshnikova2010_kanmx"] = _lmdb_fitness(
        osp.join(tc, "smf_baryshnikova2010", "processed", "lmdb"),
        {"sga_kanmx_deletion"},
    )
    sources["oduibhir2014"] = _lmdb_fitness(
        osp.join(tc, "smf_oduibhir2014", "processed", "lmdb"), None
    )
    return sources


def coverage_table(
    sources: dict[str, pd.Series], shared: set[str], gaps: set[str]
) -> pd.DataFrame:
    rows = []
    for name in UNION_PRIORITY:
        genes = set(sources[name].index)
        rows.append(
            {
                "source": name,
                "n_genes_in_source": len(genes),
                "n_of_4432_covered": len(genes & shared),
                "n_of_724_gaps_filled": len(genes & gaps),
            }
        )
    return pd.DataFrame(rows)


def union_control(
    sources: dict[str, pd.Series], shared: set[str]
) -> tuple[pd.Series, pd.Series, list[dict[str, Any]]]:
    """Costanzo where it exists, then each donor mapped onto the Costanzo scale.

    Returns the union fitness, the source name used per gene, and the per-donor transfer
    record (overlap size, the linear map, and its Pearson r on the overlap).
    """
    base = sources[UNION_PRIORITY[0]]
    fitness = base.reindex(sorted(shared)).dropna()
    origin = pd.Series(UNION_PRIORITY[0], index=fitness.index, dtype=object)
    transfers: list[dict[str, Any]] = []

    for name in UNION_PRIORITY[1:]:
        donor = sources[name]
        overlap = donor.index.intersection(base.index)
        if len(overlap) < 30:
            raise ValueError(f"{name}: overlap with Costanzo is {len(overlap)}, too thin")
        slope, intercept = np.polyfit(
            donor.loc[overlap].to_numpy(), base.loc[overlap].to_numpy(), 1
        )
        r = float(pearsonr(donor.loc[overlap], base.loc[overlap])[0])
        missing = sorted((shared - set(fitness.index)) & set(donor.index))
        transfers.append(
            {
                "source": name,
                "n_overlap_with_costanzo": int(len(overlap)),
                "slope_onto_costanzo": float(slope),
                "intercept_onto_costanzo": float(intercept),
                "pearson_on_overlap": r,
                "n_genes_contributed": len(missing),
            }
        )
        if missing:
            filled = donor.loc[missing] * slope + intercept
            fitness = pd.concat([fitness, filled])
            origin = pd.concat(
                [origin, pd.Series(name, index=missing, dtype=object)]
            )
    return fitness.sort_index(), origin.sort_index(), transfers


def fits_on(
    df: pd.DataFrame, names: list[str], mask: np.ndarray, fitness: np.ndarray
) -> dict[str, Any]:
    """The four cross-validated fits the incumbent script reports, on a chosen gene set."""
    pool = _z(np.log(df.loc[mask, names].to_numpy(dtype=float) + 1e-6))
    level = df.loc[mask, "level"].to_numpy(dtype=float)
    control = _z(fitness.reshape(-1, 1))
    return {
        "nineteen_amino_acids": _cv_score(pool, level),
        "fitness_only": _cv_score(control, level),
        "nineteen_plus_fitness": _cv_score(np.hstack([pool, control]), level),
        "nineteen_given_fitness": _cv_score(
            _residualize(pool, control),
            _residualize(level.reshape(-1, 1), control).ravel(),
        ),
        "betaxanthin_vs_fitness_pearson": float(pearsonr(level, control.ravel())[0]),
        "betaxanthin_sd": float(level.std(ddof=1)),
    }


def main() -> None:
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    experiment_root = os.environ["EXPERIMENT_ROOT"]
    tc = osp.join(data_root, "data", "torchcell")

    betaxanthin = pd.read_csv(
        osp.join(tc, "betaxanthin_cachera2023", "preprocess", "data.csv")
    )
    amino_acid = pd.read_csv(
        osp.join(tc, "amino_acid_mulleder2016", "preprocess", "data.csv")
    )
    names = [c for c in amino_acid.columns if c != "orf"]
    merged = betaxanthin.merge(amino_acid, on="orf", how="inner").set_index("orf")
    shared = set(merged.index)

    sources = load_sources(data_root)
    costanzo_genes = set(sources["costanzo2016_kanmx_30c"].index)
    gaps = shared - costanzo_genes

    coverage = coverage_table(sources, shared, gaps)
    union, origin, transfers = union_control(sources, shared)

    # Costanzo + Kuzmin only, the specific question that was asked.
    kuzmin_genes = set().union(
        *(
            set(sources[k].index)
            for k in (
                "kuzmin2018_smf",
                "kuzmin2020_smf",
                "kuzmin2018_dmf_array_smf",
                "kuzmin2020_dmf_array_smf",
            )
        )
    )
    costanzo_plus_kuzmin = (costanzo_genes | kuzmin_genes) & shared

    # The recompute. Three gene sets: the incumbent 3,708, the enlarged union, and the
    # whole 4,432 without any control, so the two effects the incumbent script warns about
    # (the control, and the gene set the control forces) stay separable.
    incumbent_mask = merged.index.isin(costanzo_genes)
    union_mask = merged.index.isin(set(union.index))
    incumbent_fitness = (
        sources["costanzo2016_kanmx_30c"].reindex(merged.index[incumbent_mask]).to_numpy()
    )
    union_fitness = union.reindex(merged.index[union_mask]).to_numpy()

    # The union pools measurements whose donor sets differ in more than scale: the 724
    # deletions Costanzo's array arm misses carry roughly twice the betaxanthin spread, so
    # a single pooled linear control mixes populations with different variances and its
    # coupling to betaxanthin is diluted. Standardizing the control WITHIN its donor
    # removes the between-donor offset, which is what a pooled control should have had in
    # the first place. Both variants are reported, because the difference between them is
    # the size of the pooling artifact and should not be hidden inside one number.
    union_fitness_within = union.copy()
    for source_name in origin.unique():
        idx = origin.index[origin == source_name]
        block = union.loc[idx]
        union_fitness_within.loc[idx] = (block - block.mean()) / block.std()
    union_fitness_within = union_fitness_within.reindex(
        merged.index[union_mask]
    ).to_numpy()

    fits = {
        "incumbent_costanzo_kanmx_30c": fits_on(
            merged, names, incumbent_mask, incumbent_fitness
        ),
        "enlarged_union": fits_on(merged, names, union_mask, union_fitness),
        "enlarged_union_within_source_z": fits_on(
            merged, names, union_mask, union_fitness_within
        ),
        "all_4432_no_control": {
            "nineteen_amino_acids": _cv_score(
                _z(np.log(merged[names].to_numpy(dtype=float) + 1e-6)),
                merged["level"].to_numpy(dtype=float),
            ),
            "betaxanthin_sd": float(merged["level"].std(ddof=1)),
        },
    }
    still_uncovered = sorted(shared - set(union.index))
    if still_uncovered:
        fits["still_uncontrolled"] = {
            "nineteen_amino_acids": _cv_score(
                _z(
                    np.log(
                        merged.loc[still_uncovered, names].to_numpy(dtype=float) + 1e-6
                    )
                ),
                merged.loc[still_uncovered, "level"].to_numpy(dtype=float),
            ),
            "betaxanthin_sd": float(merged.loc[still_uncovered, "level"].std(ddof=1)),
        }

    results_dir = osp.join(experiment_root, EXPERIMENT, "results")
    os.makedirs(results_dir, exist_ok=True)
    coverage.to_csv(
        osp.join(results_dir, "smf_coverage_by_source.csv"), index=False
    )
    payload = {
        "n_shared_deletions": len(shared),
        "n_with_costanzo_kanmx_30c": len(shared & costanzo_genes),
        "n_costanzo_gaps": len(gaps),
        "coverage_by_source": coverage.to_dict("records"),
        "costanzo_plus_kuzmin": {
            "n_of_4432_covered": len(costanzo_plus_kuzmin),
            "n_of_724_gaps_filled": len(kuzmin_genes & gaps),
        },
        "best_union": {
            "sources_in_priority_order": list(UNION_PRIORITY),
            "n_of_4432_covered": int(len(union)),
            "n_of_724_gaps_filled": int(len(gaps) - len(still_uncovered)),
            "n_still_uncovered": len(still_uncovered),
            "genes_per_source": origin.value_counts().to_dict(),
            "scale_transfers": transfers,
        },
        "cross_validated": fits,
        "written_at": timestamp(),
    }
    with open(
        osp.join(results_dir, "smf_coverage_fitness_control_sources.json"), "w"
    ) as fh:
        json.dump(payload, fh, indent=2)

    print(coverage.to_string(index=False))
    print(json.dumps({k: v for k, v in payload.items() if k != "coverage_by_source"}, indent=2))


if __name__ == "__main__":
    main()
