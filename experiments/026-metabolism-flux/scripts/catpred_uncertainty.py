# experiments/026-metabolism-flux/scripts/catpred_uncertainty.py
# [[experiments.026-metabolism-flux.scripts.catpred_uncertainty]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/catpred_uncertainty.py

r"""What CatPred's per-prediction variance buys, measured on the 57 measured yeast pairs.

THE QUESTION
------------
CatPred is the only predictor in the mirror that returns a variance with every value. Two
things follow only if the variance is any good, and both are tested here rather than
assumed.

**Is it calibrated?** If the predicted standard deviation is the standard deviation of the
error, then standardizing each residual by it gives a unit-variance quantity, and the
fraction of pairs inside :math:`\pm 1` and :math:`\pm 2` predicted standard deviations
should be 68.3 and 95.4 percent. Three readings are computed, against the total, the
aleatoric and the epistemic standard deviation, because they are different claims: the
aleatoric term says the LABEL is noisy and the epistemic term says the MODEL is unsure,
and only their sum is offered as a predictive interval.

**Does a variance-weighted combination beat the plain median?** The predicted tier the flux
layer consumes is a median over the admitted predictors in :math:`\log_{10}` space. If the
variance ranks predictions by reliability, then down-weighting CatPred where it is unsure
should beat weighting it equally. Three rules are compared on the same pairs, all in
:math:`\log_{10}` space:

``plain_median``
    the median over the predictors, which is what the resolver does today.
``inverse_variance``
    a weighted mean in which CatPred carries weight :math:`1/\sigma^2` and every other
    predictor carries :math:`1/\bar{\sigma}^2`, the median CatPred variance. **This is the
    only weighting the available data supports**: no other predictor emits a variance, so
    the alternative to giving them a common weight is inventing one.
``sd_gated``
    the plain median with CatPred's value dropped wherever its predicted standard
    deviation exceeds its own median, which is the cheapest way a resolver could use the
    variance without changing the aggregation.

THE CONTAMINATION, STATED BEFORE THE NUMBERS
----------------------------------------------
50 of the 57 :math:`k_{cat}` pairs and 43 of the 57 :math:`K_M` pairs are in CatPred-DB on
a structure-matching substrate, and the released production weights were fit on train plus
validation plus test (``catpred_training_overlap.py``). So these residuals are mostly
residuals on training data, and a calibration measured on training data is optimistic about
the aleatoric term and, for the epistemic term, measures the wrong thing entirely: the
ensemble agrees precisely where it has been fit. Every number here is reported for the full
57 and again for the pairs with no structure match in CatPred-DB, which is the only
uncontaminated reading available and is small enough to say so.

Writes ``results/catpred_uncertainty.json``. Run from the worktree root after the CatPred
tables are built and after ``compare_predictors_to_measured.py``::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/catpred_uncertainty.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats

from torchcell.metabolism.kinetics_pipeline import inchikey_block, sha256_of

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
KINETICS = osp.join(DATA_ROOT, "data", "torchcell", "kinetics")
MEASURED_CSV = osp.join(RESULTS, "predictor_vs_measured.csv")
OVERLAP_JSON = osp.join(RESULTS, "catpred_training_overlap.json")

#: The nominal two-sided coverages a Gaussian predictive interval claims, and the
#: multipliers that produce them.
NOMINAL = {"1_sigma": (1.0, 0.6826895), "2_sigma": (2.0, 0.9544997)}
#: The predictors whose median the flux layer's predicted tier is, per parameter, as the
#: within-class study admitted them BEFORE CatPred was scored. The combination rules are
#: compared against this set plus CatPred, which is the change under consideration.
INCUMBENT = {"k_cat": ("ezkinetics", "eitlem", "unikp"), "K_M": ("ezkinetics",)}


class Coverage(BaseModel):
    """Empirical coverage of one predictive interval against its nominal rate."""

    model_config = ConfigDict(extra="forbid")

    component: str
    nominal: float
    empirical: float
    n: int
    k: int
    ci95: tuple[float, float]
    mean_abs_z: float
    rms_z: float


class Combination(BaseModel):
    """One rule for combining predictors, scored on the same pairs."""

    model_config = ConfigDict(extra="forbid")

    rule: str
    n_pairs: int
    median_abs_error_decades: float
    mean_abs_error_decades: float
    frac_within_1_decade: float
    #: Signed-rank p against ``plain_median`` on the paired absolute errors. ``None`` for
    #: the rule being compared against.
    wilcoxon_p_vs_plain: float | None = None


class ParameterUncertainty(BaseModel):
    """Everything measured for one parameter."""

    model_config = ConfigDict(extra="forbid")

    parameter: str
    n_pairs: int
    n_pairs_not_in_catpred_db: int
    sd_median_decades: dict[str, float]
    coverage_all: list[Coverage]
    coverage_unseen: list[Coverage]
    #: Spearman rank correlation between the predicted total standard deviation and the
    #: absolute residual. A variance that ranks nothing is a variance worth nothing.
    spearman_sd_vs_abs_error: float
    spearman_p: float
    #: Median absolute error inside each quartile of the predicted total standard
    #: deviation, lowest quartile first. The paper's own reading of its uncertainty.
    mae_by_sd_quartile: list[float]
    n_by_sd_quartile: list[int]
    combinations: list[Combination]


class UncertaintyReport(BaseModel):
    """The whole analysis, one object."""

    model_config = ConfigDict(extra="forbid")

    built_at: str
    measured_csv: str
    measured_csv_sha256: str
    catpred_table_sha256: dict[str, str]
    contamination: dict[str, int] = Field(
        description="Pairs of each parameter that appear in CatPred-DB with a matching "
        "substrate, from catpred_training_overlap.json."
    )
    parameters: list[ParameterUncertainty]


def wilson(k: int, n: int) -> tuple[float, float]:
    """Wilson 95 percent interval, so a coverage on 57 pairs is not read like one on 5,700."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.959964
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (float(center - half), float(center + half))


def _catpred_table(parameter: str) -> pd.DataFrame:
    """CatPred's processed table reduced to one row per (accession, substrate name)."""
    path = osp.join(KINETICS, "catpred", "processed", f"{parameter}.parquet")
    frame = pd.read_parquet(path)
    frame = frame[frame[parameter] > 0].copy()
    frame["substrate"] = frame["substrate_name"].str.strip().str.lower()
    frame["log10_value"] = np.log10(frame[parameter])
    columns = {
        "log10_value": "mean",
        f"{parameter}_log10_sd_total": "mean",
        f"{parameter}_log10_sd_aleatoric": "mean",
        f"{parameter}_log10_sd_epistemic": "mean",
    }
    return frame.groupby(["uniprot", "substrate"], as_index=False).agg(columns)


def _measured(parameter: str) -> pd.DataFrame:
    """The measured pairs and every predictor's value on them, one row per pair."""
    frame = pd.read_csv(MEASURED_CSV)
    frame = frame[frame["parameter"] == parameter]
    wide = frame.pivot_table(
        index=["uniprot", "substrate", "measured", "smiles"],
        columns="predictor",
        values="predicted",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


def _seen_in_db(parameter: str, pairs: pd.DataFrame) -> np.ndarray:
    """Per pair, whether CatPred-DB holds it with a structure-matching substrate.

    Recomputed here rather than read from the overlap report, because the report stores
    counts and this needs the per-pair flag to split the calibration.
    """
    relative = {
        "k_cat": ("kcat/kcat-random_trainvaltest.csv", "reactant_smiles"),
        "K_M": ("km/km-random_trainvaltest.csv", "substrate_smiles"),
    }[parameter]
    path = osp.join(
        DATA_ROOT, "models", "kinetics", "catpred", "CatPred-DB", "data", relative[0]
    )
    database = pd.read_csv(path, usecols=["uniprot", relative[1]])
    cache: dict[str, str | None] = {}

    def blocks(smiles: str) -> set[str]:
        out: set[str] = set()
        for fragment in str(smiles).split("."):
            if not fragment:
                continue
            if fragment not in cache:
                cache[fragment] = inchikey_block(fragment)
            block = cache[fragment]
            if block:
                out.add(block)
        return out

    by_accession: dict[str, set[str]] = {}
    for accession, smiles in database.itertuples(index=False):
        by_accession.setdefault(str(accession), set()).update(blocks(smiles))
    return np.array(
        [
            bool(by_accession.get(str(u), set()) & blocks(s))
            for u, s in zip(pairs["uniprot"], pairs["smiles"])
        ]
    )


def coverage_of(residual: np.ndarray, sd: np.ndarray, component: str) -> list[Coverage]:
    """Empirical coverage at one and two predicted standard deviations."""
    keep = np.isfinite(residual) & np.isfinite(sd) & (sd > 0)
    z = residual[keep] / sd[keep]
    out: list[Coverage] = []
    for name, (multiplier, nominal) in NOMINAL.items():
        inside = np.abs(z) <= multiplier
        n, k = int(inside.size), int(inside.sum())
        out.append(
            Coverage(
                component=f"{component}_{name}",
                nominal=nominal,
                empirical=k / n if n else float("nan"),
                n=n,
                k=k,
                ci95=wilson(k, n),
                mean_abs_z=float(np.mean(np.abs(z))) if n else float("nan"),
                rms_z=float(np.sqrt(np.mean(z**2))) if n else float("nan"),
            )
        )
    return out


def _score(errors: np.ndarray, rule: str) -> Combination:
    return Combination(
        rule=rule,
        n_pairs=int(errors.size),
        median_abs_error_decades=float(np.median(np.abs(errors))),
        mean_abs_error_decades=float(np.mean(np.abs(errors))),
        frac_within_1_decade=float(np.mean(np.abs(errors) < 1.0)),
    )


def combinations_for(
    parameter: str, joined: pd.DataFrame, predictors: list[str]
) -> list[Combination]:
    """Score the three combination rules on the pairs every named predictor prices."""
    target = np.log10(joined["measured"].to_numpy(dtype=float))
    matrix = np.vstack([np.log10(joined[p].to_numpy(dtype=float)) for p in predictors])
    catpred = np.log10(joined["catpred"].to_numpy(dtype=float))
    sd = joined[f"{parameter}_log10_sd_total"].to_numpy(dtype=float)
    stacked = np.vstack([matrix, catpred])

    plain = np.median(stacked, axis=0) - target

    typical_variance = float(np.median(sd**2))
    weights = np.vstack(
        [
            np.full_like(matrix, 1.0 / typical_variance),
            (1.0 / np.maximum(sd**2, 1e-12))[None, :],
        ]
    )
    weighted = (stacked * weights).sum(axis=0) / weights.sum(axis=0) - target

    gate = sd > float(np.median(sd))
    gated_values = []
    for column in range(stacked.shape[1]):
        members = matrix[:, column] if gate[column] else stacked[:, column]
        gated_values.append(float(np.median(members)))
    gated = np.asarray(gated_values) - target

    scored = [
        _score(plain, "plain_median"),
        _score(weighted, "inverse_variance"),
        _score(gated, "sd_gated"),
        _score(np.median(matrix, axis=0) - target, "incumbent_median_without_catpred"),
    ]
    for record, errors in zip(
        scored[1:], (weighted, gated, np.median(matrix, axis=0) - target)
    ):
        # A paired test, because every rule is scored on the same pairs; the alternative
        # of two independent samples would throw away the pairing and understate the power.
        record.wilcoxon_p_vs_plain = float(
            stats.wilcoxon(np.abs(errors), np.abs(plain)).pvalue
        )
    return scored


def analyze(parameter: str) -> ParameterUncertainty:
    """Everything for one parameter."""
    table = _catpred_table(parameter)
    measured = _measured(parameter)
    joined = measured.merge(table, on=["uniprot", "substrate"], how="inner")
    residual = joined["log10_value"].to_numpy(dtype=float) - np.log10(
        joined["measured"].to_numpy(dtype=float)
    )
    seen = _seen_in_db(parameter, joined)

    components = {
        "total": joined[f"{parameter}_log10_sd_total"].to_numpy(dtype=float),
        "aleatoric": joined[f"{parameter}_log10_sd_aleatoric"].to_numpy(dtype=float),
        "epistemic": joined[f"{parameter}_log10_sd_epistemic"].to_numpy(dtype=float),
    }
    coverage_all = [
        c for k, v in components.items() for c in coverage_of(residual, v, k)
    ]
    coverage_unseen = [
        c
        for k, v in components.items()
        for c in coverage_of(residual[~seen], v[~seen], k)
    ]

    total = components["total"]
    rho = stats.spearmanr(total, np.abs(residual))
    quartiles = np.quantile(total, [0.25, 0.5, 0.75])
    bins = np.digitize(total, quartiles)
    mae = [
        float(np.median(np.abs(residual[bins == b])))
        if (bins == b).any()
        else float("nan")
        for b in range(4)
    ]
    counts = [int((bins == b).sum()) for b in range(4)]

    incumbent = [p for p in INCUMBENT[parameter] if p in joined.columns]
    complete = joined.dropna(
        subset=[*incumbent, "catpred", f"{parameter}_log10_sd_total"]
    )
    return ParameterUncertainty(
        parameter=parameter,
        n_pairs=int(len(joined)),
        n_pairs_not_in_catpred_db=int((~seen).sum()),
        sd_median_decades={k: float(np.median(v)) for k, v in components.items()},
        coverage_all=coverage_all,
        coverage_unseen=coverage_unseen,
        spearman_sd_vs_abs_error=float(rho.statistic),
        spearman_p=float(rho.pvalue),
        mae_by_sd_quartile=mae,
        n_by_sd_quartile=counts,
        combinations=combinations_for(parameter, complete, incumbent),
    )


def main() -> None:
    """Measure the calibration and the combination rules, and write the report."""
    with open(OVERLAP_JSON) as handle:
        overlap = json.load(handle)
    contamination = {
        block["parameter"]: next(
            level["n_matched"] for level in block["levels"] if level["level"] == "pair"
        )
        for block in overlap["parameters"]
    }
    report = UncertaintyReport(
        built_at=datetime.now(UTC).isoformat(),
        measured_csv=MEASURED_CSV,
        measured_csv_sha256=sha256_of(MEASURED_CSV),
        catpred_table_sha256={
            p: sha256_of(osp.join(KINETICS, "catpred", "processed", f"{p}.parquet"))
            for p in ("k_cat", "K_M")
        },
        contamination=contamination,
        parameters=[analyze(p) for p in ("k_cat", "K_M")],
    )
    out_path = osp.join(RESULTS, "catpred_uncertainty.json")
    with open(out_path, "w") as handle:
        handle.write(report.model_dump_json(indent=2))

    for block in report.parameters:
        print(
            f"\n{block.parameter}: {block.n_pairs} pairs, "
            f"{block.n_pairs_not_in_catpred_db} of them absent from CatPred-DB"
        )
        print(f"  median SD (decades): {block.sd_median_decades}")
        for record in block.coverage_all:
            print(
                f"  {record.component:22s} nominal {record.nominal:.3f} "
                f"empirical {record.empirical:.3f} ({record.k}/{record.n}) "
                f"rms z {record.rms_z:.2f}"
            )
        print(
            f"  spearman(SD, |error|) = {block.spearman_sd_vs_abs_error:+.3f} "
            f"(p = {block.spearman_p:.3g})"
        )
        print(f"  median |error| by SD quartile: {block.mae_by_sd_quartile}")
        for rule in block.combinations:
            suffix = (
                ""
                if rule.wilcoxon_p_vs_plain is None
                else f", p = {rule.wilcoxon_p_vs_plain:.3g} vs plain"
            )
            print(
                f"  {rule.rule:34s} median |error| "
                f"{rule.median_abs_error_decades:.3f} decades on {rule.n_pairs}{suffix}"
            )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
