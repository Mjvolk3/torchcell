# experiments/026-metabolism-flux/scripts/ec_class_variance.py
# [[experiments.026-metabolism-flux.scripts.ec_class_variance]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/ec_class_variance.py

r"""How much do enzymes of the SAME function differ, and does a predictor beat knowing that?

A correlation computed across a quantity with a six-decade range is easy to earn and hard to
interpret. If most of that range sits BETWEEN EC classes, then a predictor that has learned
only which class an enzyme belongs to scores a respectable Pearson while being useless for
the question a flux layer asks, which is what THIS enzyme's turnover number is. The
statistics that separate those two cases are computed here.

**Three quantities, and the comparison between them is the point.**

1. **Within-class biological spread.** Across enzymes sharing a full EC4 number, the
   standard deviation of :math:`\log_{10} k_{cat}` in decades. This is how much real enzymes
   of the same function differ, and it is the floor below which no predictor conditioned on
   function alone can go.
2. **Condition spread.** For one (enzyme, substrate) pair measured more than once, the
   spread across the reported assay conditions. This is the measurement floor. Reported
   separately inside the physiological window (20 to 40 C, pH 6 to 8), because a
   :math:`k_{cat}` at 4 C is a different quantity rather than a noisy copy of the same one.
3. **Across-predictor spread**, already measured elsewhere and read back here for the
   comparison.

**The decisive test is the EC-class baseline.** For each scored yeast pair, take the median
:math:`\log_{10} k_{cat}` of every OTHER enzyme sharing its EC number and call that the
prediction. It uses no sequence, no structure and no learning. If the published predictors
do not clearly beat it, their correlation is largely the between-class range restated.
Leave-one-out is on the UniProt accession, so the enzyme never prices itself, and the EC
level backs off (EC4 to EC3 to EC2 to EC1 to global) when a class has no other member; the
level actually used is recorded per pair.

**Within-class accuracy is the same test from the other side.** Centering both measured and
predicted values on their class mean removes the between-class range by construction, and
what survives is whether the predictor orders enzymes WITHIN a function.

Needs the whole database rather than the yeast slice: yeast alone has 90 EC4 classes but
only four with three or more distinct enzymes, which cannot estimate a within-class spread.
``mirror_oed_all.py`` fetches it. Writes ``results/ec_class_variance.json``.
"""

import json
import os
import os.path as osp
from typing import cast

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats

load_dotenv()

DATA_ROOT = cast(str, os.environ["DATA_ROOT"])
EXPERIMENT_ROOT = cast(str, os.environ["EXPERIMENT_ROOT"])
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
OED_DIR = osp.join(DATA_ROOT, "data", "enzyme_kinetics", "open_enzyme_database")
ALL_RECORDS = osp.join(OED_DIR, "all", "oed_records.json")
YEAST_RECORDS = osp.join(OED_DIR, "scerevisiae", "oed_records.json")
PAIRS = osp.join(RESULTS, "predictor_vs_measured.csv")

#: The window the GEM's own physiology sits in. Outside it a turnover number is a different
#: quantity, not a noisy version of the same one.
TEMP_MIN_C, TEMP_MAX_C = 20.0, 40.0
PH_MIN, PH_MAX = 6.0, 8.0

#: A class needs this many distinct enzymes before its within-class spread is estimated.
#: Below three, a standard deviation is dominated by which two enzymes happened to be
#: measured.
MIN_ENZYMES_PER_CLASS = 3


def load_records(path: str, parameter: str) -> pd.DataFrame:
    """Wild-type rows carrying this parameter, one row per literature report."""
    column = {"k_cat": "kcat_value", "K_M": "km_value"}[parameter]
    rows = []
    for record in json.load(open(path)):
        value = record.get(column)
        if value is None or not np.isfinite(value) or value <= 0:
            continue
        if (record.get("enzymetype") or "").lower() != "wildtype":
            continue
        if not record.get("ec") or not record.get("uniprot"):
            continue
        rows.append(
            {
                "ec": record["ec"],
                "uniprot": record["uniprot"],
                "organism": record.get("organism"),
                "substrate": (record.get("substrate") or "").strip().lower(),
                "log10": float(np.log10(value)),
                "temperature": record.get("temperature"),
                "ph": record.get("ph"),
            }
        )
    return pd.DataFrame(rows)


def physiological(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows assayed inside the physiological window, conditions required to be present."""
    return frame[
        frame["temperature"].between(TEMP_MIN_C, TEMP_MAX_C)
        & frame["ph"].between(PH_MIN, PH_MAX)
    ]


def per_pair(frame: pd.DataFrame) -> pd.DataFrame:
    """One value per (enzyme, substrate): the mean in log space over its reports.

    Collapsing to the pair first stops a heavily studied enzyme from dominating its class's
    spread with forty reports of the same number.
    """
    return frame.groupby(["ec", "uniprot", "substrate"], as_index=False).agg(
        log10=("log10", "mean"), n_reports=("log10", "size")
    )


def within_class_spread(pairs: pd.DataFrame, level: int = 4) -> tuple[pd.DataFrame, dict]:
    """Standard deviation of log10 value across the distinct enzymes of each EC class.

    The statistic is a standard deviation rather than a max-minus-min range, because a range
    grows with the number of enzymes measured and would report a well-studied class as more
    variable than a rare one for no biological reason.

    ``level`` is how many digits of the EC number define a class. At four digits a class is
    one reaction; at three it is a chemical family, so a class holds enzymes doing related
    but different chemistry and the spread can only grow. Reading the spread at each depth
    is what separates "these enzymes catalyze the same reaction and still differ" from "we
    grouped unlike reactions together".
    """
    per_enzyme = pairs.groupby(["ec", "uniprot"], as_index=False)["log10"].mean()
    if level < 4:
        per_enzyme = per_enzyme.assign(
            ec=per_enzyme["ec"].str.split(".").str[:level].str.join(".")
        )
        per_enzyme = per_enzyme.groupby(["ec", "uniprot"], as_index=False)["log10"].mean()
    grouped = per_enzyme.groupby("ec")["log10"]
    table = pd.DataFrame(
        {
            "n_enzymes": grouped.size(),
            "sd_decades": grouped.std(ddof=1),
            "range_decades": grouped.max() - grouped.min(),
            "median_log10": grouped.median(),
        }
    ).reset_index()
    table = table[table["n_enzymes"] >= MIN_ENZYMES_PER_CLASS].dropna(subset=["sd_decades"])
    summary = {
        "ec_level": level,
        "n_classes_scored": int(len(table)),
        "n_enzymes_in_scored_classes": int(table["n_enzymes"].sum()),
        "median_within_class_sd_decades": float(table["sd_decades"].median()),
        "p90_within_class_sd_decades": float(table["sd_decades"].quantile(0.9)),
        "median_within_class_range_decades": float(table["range_decades"].median()),
        "frac_classes_sd_above_half_decade": float((table["sd_decades"] > 0.5).mean()),
        "frac_classes_range_above_1_decade": float((table["range_decades"] > 1.0).mean()),
        "sd_of_class_medians_decades": float(table["median_log10"].std(ddof=1)),
        "sd_across_all_enzymes_decades": float(per_enzyme["log10"].std(ddof=1)),
        # A max-minus-min over 4,555 enzymes is set by two outliers, so the total range is
        # reported as a 10th-to-90th-percentile span. Every other range in this report is a
        # max-minus-min over a handful of values, where the robust version would discard
        # most of the sample; the caption says which is which.
        "p10_p90_range_across_all_enzymes_decades": float(
            per_enzyme["log10"].quantile(0.9) - per_enzyme["log10"].quantile(0.1)
        ),
    }
    return table, summary


def variance_decomposition(pairs: pd.DataFrame, level: int) -> dict:
    """How much of the variance in log10 value lies BETWEEN EC classes at this depth.

    Two numbers, because they answer different questions. ``r_squared`` is the fraction of
    observed variance the class labels account for in this sample, and it rises toward 1 as
    classes get smaller whether or not the class means differ. ``icc`` is the one-way
    random-effects intraclass correlation, which subtracts the within-class mean square and
    so does not reward slicing the data more finely. The ICC is the honest one.
    """
    per_enzyme = pairs.groupby(["ec", "uniprot"], as_index=False)["log10"].mean()
    per_enzyme = per_enzyme.assign(
        klass=per_enzyme["ec"].str.split(".").str[:level].str.join(".")
    )
    counts = per_enzyme["klass"].value_counts()
    kept = per_enzyme[per_enzyme["klass"].isin(counts[counts >= 2].index)]
    groups = [g["log10"].to_numpy() for _, g in kept.groupby("klass")]
    n_total = sum(len(g) for g in groups)
    k = len(groups)
    grand = float(np.mean(np.concatenate(groups)))
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n_total - k)
    # k0 is the effective group size for unbalanced designs (Snedecor and Cochran).
    sizes = np.array([len(g) for g in groups], dtype=float)
    k0 = (n_total - (sizes**2).sum() / n_total) / (k - 1)
    icc = (ms_between - ms_within) / (ms_between + (k0 - 1) * ms_within)
    return {
        "level": level,
        "n_classes": int(k),
        "n_enzymes": int(n_total),
        "r_squared_between": float(ss_between / (ss_between + ss_within)),
        "icc": float(icc),
        "within_class_sd_decades": float(np.sqrt(ms_within)),
        "total_sd_decades": float(np.std(np.concatenate(groups), ddof=1)),
    }


def condition_spread(frame: pd.DataFrame) -> dict:
    """Spread of one (enzyme, substrate) pair across the conditions it was measured at.

    This is the measurement floor: no predictor given a sequence and a SMILES can resolve
    below the disagreement between two literature values for the same enzyme.
    """

    def table_of(subset: pd.DataFrame) -> pd.DataFrame:
        grouped = subset.groupby(["uniprot", "substrate"])
        table = pd.DataFrame(
            {
                "n": grouped.size(),
                "sd": grouped["log10"].std(ddof=1),
                "range": grouped["log10"].max() - grouped["log10"].min(),
                "n_conditions": grouped.apply(
                    lambda g: len(set(zip(g["temperature"], g["ph"]))), include_groups=False
                ),
            }
        ).reset_index()
        return table[table["n"] >= 2].dropna(subset=["sd"])

    def spread_of(subset: pd.DataFrame) -> dict:
        repeated = table_of(subset)
        multi = repeated[repeated["n_conditions"] >= 2]
        return {
            "n_pairs_repeated": int(len(repeated)),
            "median_sd_decades": float(repeated["sd"].median()),
            "median_range_decades": float(repeated["range"].median()),
            "p90_range_decades": float(repeated["range"].quantile(0.9)),
            "frac_range_above_1_decade": float((repeated["range"] > 1.0).mean()),
            "n_pairs_multi_condition": int(len(multi)),
            "median_range_multi_condition_decades": (
                float(multi["range"].median()) if len(multi) else None
            ),
        }

    return {
        "all_conditions": spread_of(frame),
        "physiological_window": spread_of(physiological(frame)),
        "window": {
            "temperature_c": [TEMP_MIN_C, TEMP_MAX_C],
            "ph": [PH_MIN, PH_MAX],
        },
        # The per-pair table itself, so the measurement floor can be drawn as a
        # distribution beside the biological one rather than compared as two summary bars.
        "_pairs_physiological": table_of(physiological(frame)),
    }


def ec_class_baseline(
    scored: pd.DataFrame, reference: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Predict each scored enzyme as the median of the OTHER enzymes in its EC class.

    No sequence, no structure, no training. The EC level backs off until the class holds
    another enzyme, and the level used is recorded so a prediction made at EC1 is not read
    as a class-level one.
    """
    per_enzyme = reference.groupby(["ec", "uniprot"], as_index=False)["log10"].mean()
    global_median = float(per_enzyme["log10"].median())
    predictions, levels, class_sizes = [], [], []
    for _, row in scored.iterrows():
        parts = row["ec"].split(".")
        chosen, used_level, size = global_median, 0, 0
        for level in (4, 3, 2, 1):
            prefix = ".".join(parts[:level])
            siblings = per_enzyme[
                per_enzyme["ec"].str.startswith(prefix + ("." if level < 4 else ""))
                if level < 4
                else (per_enzyme["ec"] == prefix)
            ]
            siblings = siblings[siblings["uniprot"] != row["uniprot"]]
            if len(siblings) >= 1:
                chosen, used_level, size = (
                    float(siblings["log10"].median()),
                    level,
                    int(len(siblings)),
                )
                break
        predictions.append(chosen)
        levels.append(used_level)
        class_sizes.append(size)
    out = scored.assign(
        baseline_log10=predictions, baseline_ec_level=levels, baseline_class_n=class_sizes
    )
    residual = out["baseline_log10"] - np.log10(out["measured"])
    summary = {
        "n_pairs": int(len(out)),
        "spearman": float(
            stats.spearmanr(np.log10(out["measured"]), out["baseline_log10"]).statistic
        ),
        "pearson_log10": float(
            stats.pearsonr(np.log10(out["measured"]), out["baseline_log10"]).statistic
        ),
        "median_abs_error_decades": float(np.median(np.abs(residual))),
        "median_bias_decades": float(np.median(residual)),
        "frac_within_1_decade": float(np.mean(np.abs(residual) < 1.0)),
        "n_priced_at_ec4": int(sum(1 for level in levels if level == 4)),
        "median_class_n": float(np.median(class_sizes)),
    }
    return out, summary


def _centered_pearson(frame: pd.DataFrame) -> float:
    """Pearson between measured and predicted after both are centered on the class mean."""
    centered = frame.assign(
        m=frame["m"] - frame.groupby("ec")["m"].transform("mean"),
        p=frame["p"] - frame.groupby("ec")["p"].transform("mean"),
    )
    return float(stats.pearsonr(centered["m"], centered["p"]).statistic)


def within_class_accuracy(
    scored: pd.DataFrame, predictors: list[str], n_boot: int = 2000, seed: int = 0
) -> dict:
    """Predictor accuracy after the between-class range is removed by centering.

    **Only classes holding at least two distinct PAIRS may contribute.** A class of one
    centers to exactly zero on both axes, which adds a point sitting on the origin: it
    cannot disagree, so it inflates the sample size and drags the correlation toward
    whatever the multi-pair classes say. Counting must therefore be done on unique pairs,
    not on the joined table, where each pair appears once per predictor and a singleton
    class looks like six.

    The interval is a CLUSTER bootstrap that resamples whole EC classes rather than pairs,
    because two pairs in one class are not independent draws once the class mean has been
    subtracted from both.
    """
    unique_pairs = scored.drop_duplicates(["uniprot", "substrate"])
    counts = unique_pairs["ec"].value_counts()
    multi = set(counts[counts >= 2].index)
    usable = scored[scored["ec"].isin(multi)]
    usable_pairs = unique_pairs[unique_pairs["ec"].isin(multi)]
    out: dict = {
        "n_pairs_available": int(len(unique_pairs)),
        "n_pairs_used": int(len(usable_pairs)),
        "n_classes": int(usable_pairs["ec"].nunique()),
        "note": (
            "Classes with a single scored pair are excluded: centering sends them to the "
            "origin, where they cannot disagree."
        ),
        "predictors": {},
    }
    rng = np.random.default_rng(seed)
    for predictor in predictors:
        subset = usable[usable["predictor"] == predictor]
        if subset["ec"].nunique() < 3:
            continue
        frame = pd.DataFrame(
            {
                "ec": subset["ec"].to_numpy(),
                "m": np.log10(subset["measured"].to_numpy()),
                "p": np.log10(subset["predicted"].to_numpy()),
            }
        )
        classes = frame["ec"].unique()
        by_class = {name: group for name, group in frame.groupby("ec")}
        draws = []
        for _ in range(n_boot):
            picked = rng.choice(classes, size=len(classes), replace=True)
            # Relabel each draw so a class picked twice becomes two independent clusters
            # rather than one class of doubled size, which would halve its centering.
            resampled = pd.concat(
                [by_class[name].assign(ec=f"{name}__{i}") for i, name in enumerate(picked)]
            )
            usable_classes = resampled.groupby("ec").size()
            if (usable_classes >= 2).sum() < 3:
                continue
            value = _centered_pearson(resampled)
            # A draw whose centered values are constant on one axis gives an undefined
            # correlation. Dropping those rather than propagating them is why TurNuP,
            # which covers 53 of the 57 pairs and so leaves some classes with a single
            # usable pair, returned a NaN interval on the first run.
            if np.isfinite(value):
                draws.append(value)
        out["predictors"][predictor] = {
            "n_pairs": int(len(subset)),
            "n_classes": int(len(classes)),
            "pearson_raw": float(stats.pearsonr(frame["m"], frame["p"]).statistic),
            "pearson_within_class": _centered_pearson(frame),
            "n_bootstrap_draws": int(len(draws)),
            "pearson_within_class_ci95": [
                float(np.percentile(draws, 2.5)),
                float(np.percentile(draws, 97.5)),
            ],
            "spearman_raw": float(stats.spearmanr(frame["m"], frame["p"]).statistic),
        }
    return out


def main() -> None:
    if not osp.exists(ALL_RECORDS):
        raise SystemExit(
            f"{ALL_RECORDS} missing. Run mirror_oed_all.py first; the yeast slice alone "
            "cannot estimate a within-class spread."
        )
    report: dict = {"parameters": {}}

    for parameter in ("k_cat", "K_M"):
        records = load_records(ALL_RECORDS, parameter)
        pairs = per_pair(records)
        classes, spread_summary = within_class_spread(pairs)
        block = {
            "n_records": int(len(records)),
            "n_pairs": int(len(pairs)),
            "n_enzymes": int(records["uniprot"].nunique()),
            "n_ec4_classes": int(records["ec"].nunique()),
            "within_class": spread_summary,
            "variance_decomposition": [
                variance_decomposition(pairs, level) for level in (1, 2, 3, 4)
            ],
            # The same within-class spread read at each EC depth. A class defined on fewer
            # digits pools unlike chemistry, so the spread must rise as digits are dropped;
            # what the numbers settle is HOW MUCH, and therefore how much of the EC4 spread
            # is same-reaction variation rather than an artifact of coarse grouping.
            "within_class_by_level": [
                within_class_spread(pairs, level)[1] for level in (1, 2, 3, 4)
            ],
            "condition_spread": condition_spread(records),
        }
        # Pull the frame out of the report before it is serialized; JSON holds summaries,
        # the CSV holds the rows a figure needs.
        block["condition_spread"].pop("_pairs_physiological").to_csv(
            osp.join(RESULTS, f"condition_spread_pairs_{parameter}.csv"), index=False
        )
        # Physiological-window within-class spread, which is the number that bounds a
        # prediction the flux layer would actually consume.
        physio_pairs = per_pair(physiological(records))
        _, physio_summary = within_class_spread(physio_pairs)
        block["within_class_physiological"] = physio_summary
        classes.to_csv(
            osp.join(RESULTS, f"ec_class_spread_{parameter}.csv"), index=False
        )
        report["parameters"][parameter] = block

    # The baseline and the within-class accuracy, on the pairs the predictors were scored on.
    yeast = load_records(YEAST_RECORDS, "k_cat")
    ec_of_pair = (
        yeast.groupby(["uniprot", "substrate"])["ec"].first().to_dict()
    )
    scored = pd.read_csv(PAIRS)
    scored = scored[scored["parameter"] == "k_cat"].copy()
    scored["ec"] = [
        ec_of_pair.get((u, s)) for u, s in zip(scored["uniprot"], scored["substrate"])
    ]
    scored = scored[scored["ec"].notna()]
    reference = load_records(ALL_RECORDS, "k_cat")
    unique_pairs = scored.drop_duplicates(["uniprot", "substrate"])[
        ["uniprot", "substrate", "measured", "ec"]
    ]
    baseline_table, baseline_summary = ec_class_baseline(unique_pairs, per_pair(reference))
    baseline_table.to_csv(osp.join(RESULTS, "ec_class_baseline_k_cat.csv"), index=False)
    predictors = sorted(scored["predictor"].unique())
    report["ec_class_baseline_k_cat"] = baseline_summary
    report["within_class_accuracy_k_cat"] = within_class_accuracy(scored, predictors)

    with open(osp.join(RESULTS, "ec_class_variance.json"), "w") as handle:
        json.dump(report, handle, indent=2)

    for parameter, block in report["parameters"].items():
        w = block["within_class"]
        print(
            f"\n{parameter}: {block['n_records']} records, {block['n_enzymes']} enzymes, "
            f"{block['n_ec4_classes']} EC4 classes"
        )
        print(
            f"  within-class sd  median {w['median_within_class_sd_decades']:.2f} dec "
            f"over {w['n_classes_scored']} classes; "
            f"all-enzyme sd {w['sd_across_all_enzymes_decades']:.2f} dec"
        )
        by_level = {b["ec_level"]: b for b in block["within_class_by_level"]}
        for decomposition in block["variance_decomposition"]:
            spread = by_level[decomposition["level"]]
            print(
                f"  EC{decomposition['level']}  classes {decomposition['n_classes']:5d}  "
                f"R2 {decomposition['r_squared_between']:.3f}  ICC {decomposition['icc']:.3f}"
                f"  median within-class sd {spread['median_within_class_sd_decades']:.2f} dec"
                f"  (p90 {spread['p90_within_class_sd_decades']:.2f})"
            )
        condition = block["condition_spread"]["physiological_window"]
        print(
            f"  condition floor (physiological): median range "
            f"{condition['median_range_decades']:.2f} dec over "
            f"{condition['n_pairs_repeated']} repeated pairs"
        )
    print("\nEC-class baseline on the scored k_cat pairs:")
    print(json.dumps(report["ec_class_baseline_k_cat"], indent=2))
    accuracy = report["within_class_accuracy_k_cat"]
    print(
        f"\nWithin-class accuracy, {accuracy['n_pairs_used']} of "
        f"{accuracy['n_pairs_available']} pairs in {accuracy['n_classes']} classes:"
    )
    for name, block in accuracy["predictors"].items():
        low, high = block["pearson_within_class_ci95"]
        print(
            f"  {name:12s} pearson raw {block['pearson_raw']:+.3f} -> "
            f"within class {block['pearson_within_class']:+.3f} "
            f"[{low:+.2f}, {high:+.2f}]"
        )
    print(f"\nwrote {osp.join(RESULTS, 'ec_class_variance.json')}")


if __name__ == "__main__":
    main()
