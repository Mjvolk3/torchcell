# experiments/019-simb-multimodal/scripts/morphology_noise_ceiling.py
# [[experiments.019-simb-multimodal.scripts.morphology_noise_ceiling]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/morphology_noise_ceiling
"""Estimate the ACHIEVABLE per-gene Pearson ceiling for Ohya-2005 CalMorph morphology.

Ohya 2005 gives ONE biological sample per deletion mutant (population-averaged over
cells) but 122 independent his3 wild-type replicates. So each mutant's 501-trait vector
is a single noisy measurement. We estimate, per feature, the reliability

    reliability_k = signal_var_k / total_var_k = 1 - noise_var_k / total_var_k

where noise_var_k = variance across the 122 WT replicates (measurement + replicate noise
of one genotype) and total_var_k = variance across the 4,718 mutants (biological signal +
that same single-replicate noise). This is broad-sense reliability (H^2). The maximum
achievable across-strain Pearson for a *perfect* predictor of the true signal against the
noisy target is

    ceiling_k = corr(signal, signal + noise) = sqrt(reliability_k).

Our metric `val/global/pearson_per_gene` = mean over features of the across-strain Pearson,
so its ceiling = mean_k ceiling_k over the features the model actually predicts (the 281
CALMORPH_LABELS minus the 3 dropped = 278). We compare that ceiling to the observed ~0.04.

Reads the sha256-pinned SCMD mirror (per manifest):
  $DATA_ROOT/torchcell-library/ohyaHighdimensionalLargescalePhenotyping2005a/data/
    mt4718data.tsv  (4,718 mutants x 501)   wt122data.tsv  (122 WT replicates x 501)

The human-readable parameter descriptions in the scalar shortlist are read verbatim
from the Ohya 2005 SI, not written here: Table 2 "parameter statistics"
(si/si6.pdf, sha256 037a50927bc5b978ab38b207b264da87550feeeeaaf21bc311f65a5ef13e9638)
carries a "parameter description" column for all 501 ids, and its born-digital text
layer is parsed with `pdftotext -layout`, the same recipe torchcell.literature.calmorph
uses on Table 1. A parameter with no row in that table gets the literal string
"not sourced"; nothing is inferred from the id.

Run from repo root:  python experiments/019-simb-multimodal/scripts/morphology_noise_ceiling.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import re
from collections.abc import Hashable
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from torchcell.datamodels.calmorph_labels import CALMORPH_LABELS
from torchcell.literature.extract import pdf_text

DROPPED = [
    "A113_A",
    "D203",
    "D205",
]  # experiments/019 delta config multitask.drop_features.global

LIBRARY_KEY = "ohyaHighdimensionalLargescalePhenotyping2005"
DESCRIPTION_SI = "si/si6.pdf"  # Ohya 2005 Table 2, "parameter statistics"
DESCRIPTION_SI_SHA256 = (
    "037a50927bc5b978ab38b207b264da87550feeeeaaf21bc311f65a5ef13e9638"
)
NOT_SOURCED = "not sourced"

# A CalMorph id: channel letter (C cell wall / A actin / D DNA / T total), an optional
# CV variant, a parameter number with an optional -n subindex, and an optional nuclear
# stage suffix (_A, _A1B, _B, _C). Same grammar as torchcell.literature.calmorph.
_DESC_ROW_RE = re.compile(r"^([ACDT](?:CV)?\d+(?:-\d)?(?:_(?:A1B|A|B|C))?)\s\s+(.+)$")
_COL_GAP_RE = re.compile(r"\s{2,}")

# Cell-size and cell-shape features quoted in the write-up. Whole-cell / mother / bud
# area, the two axis lengths, the outline lengths, the size ratio, and the axis ratio
# (roundness) that the shortlist caption contrasts the shortlist against.
SIZE_FEATURES = [
    "C11-1_A",
    "C101_A1B",
    "C101_C",
    "C11-1_A1B",
    "C11-1_C",
    "C11-2_A1B",
    "C11-2_C",
    "C118_A1B",
    "C118_C",
    "C103_A",
    "C104_A",
    "C12-1_A",
    "C102_A1B",
    "C102_C",
    "C115_A",
    "C115_A1B",
    "C115_C",
]


def _load(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    mt = pd.read_csv(osp.join(data_dir, "mt4718data.tsv"), sep="\t").set_index("ORF")
    wt = pd.read_csv(osp.join(data_dir, "wt122data.tsv"), sep="\t").set_index("NAME")
    return mt, wt


def _reliability(mt: pd.DataFrame, wt: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    noise_var = wt[feats].var(ddof=1)  # across 122 WT replicates
    total_var = mt[feats].var(ddof=1)  # across 4,718 mutants (signal + noise)
    rel = (1.0 - noise_var / total_var).clip(lower=0.0, upper=1.0)
    ceiling = np.sqrt(rel)
    return pd.DataFrame(
        {
            "noise_var": noise_var,
            "total_var": total_var,
            "reliability": rel,
            "ceiling": ceiling,
        }
    )


def _summary(name: str, df: pd.DataFrame) -> None:
    c = df["ceiling"]
    print(f"\n== {name} (n={len(df)}) ==")
    print(f"  mean ceiling (=max achievable mean per-gene Pearson): {c.mean():.4f}")
    print(
        f"  median ceiling: {c.median():.4f}   |  ceiling IQR: [{c.quantile(0.25):.3f}, {c.quantile(0.75):.3f}]"
    )
    print(f"  reliability mean: {df['reliability'].mean():.4f}")
    for thr in (0.05, 0.10, 0.20, 0.30, 0.50):
        print(
            f"    features with ceiling > {thr:.2f}: {(c > thr).sum():4d}  ({100 * (c > thr).mean():.1f}%)"
        )


def _si_descriptions(data_root: str) -> dict[str, str]:
    """Parameter id -> the SI's own plain-English description, read from the SI PDF.

    Ohya 2005 Table 2 (`si/si6.pdf`) lists every parameter as
    `ID  parameter description  <statistics...>`, columns separated by runs of two or
    more spaces. Anchoring on the id at the start of the line and taking the first
    column-gap-delimited field after it gives the description verbatim, and the trailing
    numeric columns are ignored. Descriptions are NOT normalized: a source typo stays a
    source typo, so the table can be checked against the PDF character for character.
    """
    path = osp.join(data_root, "torchcell-library", LIBRARY_KEY, DESCRIPTION_SI)
    out: dict[str, str] = {}
    for line in pdf_text(path, layout=True).splitlines():
        match = _DESC_ROW_RE.match(line.strip())
        if match is None:
            continue
        out[match.group(1)] = _COL_GAP_RE.split(match.group(2).strip())[0]
    return out


def _annotate(
    rel: pd.DataFrame, mt: pd.DataFrame, descriptions: dict[str, str]
) -> pd.DataFrame:
    """Attach robust CV, the CalMorph label, the SI description, and the rank key.

    robust_cv = IQR / |median| across the 4,718 mutants, a scale-free measure of how far
    deletions move the feature that does not assume the feature is Gaussian. score =
    ceiling * robust_cv is the shortlist rank key: a scalar target is only worth training
    on if the measurement is reliable AND deletions move it, and ranking on ceiling alone
    selects features that are precisely measured and nearly constant across mutants,
    which is the wrong end of the trade.
    """
    feats = [str(k) for k in rel.index]
    q75 = mt[feats].quantile(0.75)
    q25 = mt[feats].quantile(0.25)
    median = mt[feats].median().abs()
    robust_cv = ((q75 - q25) / median.replace(0.0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )
    out = rel.assign(
        robust_cv=robust_cv,
        label=[CALMORPH_LABELS.get(k, "") for k in feats],
        description=[descriptions.get(k, NOT_SOURCED) for k in feats],
        score=rel["ceiling"] * robust_cv,
    )
    return out.assign(rank=out["score"].rank(ascending=False, method="min"))


def _records(df: pd.DataFrame) -> list[dict[Hashable, Any]]:
    return df.reset_index().rename(columns={"index": "feature"}).to_dict("records")


_TEX_HEADER = r"""%% GENERATED by experiments/019-simb-multimodal/scripts/morphology_noise_ceiling.py
%% SOURCE: experiments/019-simb-multimodal/results/morphology_noise_ceiling.json, scalar_shortlist
%% Descriptions are verbatim from Ohya 2005 SI Table 2 "parameter statistics"
%%   $DATA_ROOT/torchcell-library/{key}/{si}
%%   sha256 {sha}
\begin{{table}}[htbp]
\centering\footnotesize
%% Ragged-right paragraph columns: a narrow cell holding one long underscored token
%% cannot be justified, and justifying it emits an underfull hbox on every row.
\begin{{tabular}}{{l >{{\raggedright\arraybackslash}}p{{44mm}} >{{\raggedright\arraybackslash}}p{{72mm}} r r r}}
\toprule
feature & CalMorph label & SI description & ceiling & robust CV & score \\
\midrule
"""

_TEX_FOOTER = r"""\bottomrule
\end{tabular}
\caption[]{The twenty best scalar morphology targets, ranked by ceiling multiplied by
robust CV. The list is dominated by bounded class ratios, actin localization class,
bud-size class and nuclear-stage class, which is what a deletion actually moves. Cell
size is the opposite trade and cell shape more so: cell size at the unbudded stage
\file{C11-1_A} has a ceiling of $0.928$ but a robust CV of $0.097$, ranking 101st, and
mother-cell roundness \file{C115_A} has a ceiling of $0.972$ with a robust CV of $0.024$,
ranking 245th of the 275 features with a defined rank key. That last id was described as a
whole-cell axis ratio in the previous revision; SI Table 2 calls it roundness of the mother
cell, and it is a shape parameter rather than a size one. Descriptions are quoted
verbatim from Ohya 2005 SI Table 2, source typography included, and the label column is
the same parameter's name in SI Table 1; the two tables word some parameters
differently, and both wordings are the paper's. In a CalMorph id the
leading letter names the stain the parameter is measured from, C the FITC-ConA cell-wall
image, A the rhodamine phalloidin actin image and D the DAPI nuclear image, and the
suffix names the nuclear stage the cells were binned into, \texttt{\_A}, \texttt{\_A1B}
or \texttt{\_C}, with no suffix meaning all stages pooled. The single-letter classes
inside a description, actin $a$ to $f$ and nuclear $A$ to $F$, are defined by the
drawings in SI Fig.~5B rather than by any text, so those rows rest on that figure.}
\label{tab:morph-scalar}
\end{table}
"""


def _escape(text: str) -> str:
    return text.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def _shortlist_tex(shortlist: pd.DataFrame) -> str:
    body = "".join(
        f"\\file{{{k}}} & \\file{{{row['label']}}} & {_escape(str(row['description']))} "
        f"& ${row['ceiling']:.3f}$ & ${row['robust_cv']:.3f}$ & ${row['score']:.3f}$ \\\\\n"
        for k, row in shortlist.iterrows()
    )
    header = _TEX_HEADER.format(
        key=LIBRARY_KEY, si=DESCRIPTION_SI, sha=DESCRIPTION_SI_SHA256
    )
    return header + body + _TEX_FOOTER


def _observed_best(experiment_root: str) -> dict[str, float | str | None]:
    """Best morphology score on the committed leaderboard, or nothing if it is absent.

    Read rather than hardcoded: the earlier fixed 0.040 was the morph_002 control run and
    silently stopped being the best score once morph_v5 landed.
    """
    path = osp.join(
        experiment_root, "019-simb-multimodal", "results", "round_leaderboards.csv"
    )
    if not osp.exists(path):
        return {"source": None}
    board = pd.read_csv(path)
    morph = board[board["strand"] == "morphology"]
    morph = morph[~morph["is_collapsed"].fillna(False)]
    row = morph.loc[morph["primary_roll_max"].idxmax()]
    return {
        "source": path,
        "run_id": str(row["run_id"]),
        "roll_max": float(row["primary_roll_max"]),
        "epochs": float(row["epochs"]),
        "epoch_at_roll_max": float(row["primary_epoch_at_roll_max"]),
    }


def main() -> None:
    load_dotenv()
    data_dir = osp.join(
        os.environ["DATA_ROOT"],
        "torchcell-library/ohyaHighdimensionalLargescalePhenotyping2005a/data",
    )
    mt, wt = _load(data_dir)
    cols = list(mt.columns)
    base = [k for k in CALMORPH_LABELS if k in cols]  # 281 base labels
    model_feats = [k for k in base if k not in DROPPED]  # the 278 the model predicts
    cv = [c for c in cols if c not in set(base)]  # 220 CV statistics
    print(
        f"loaded mt={mt.shape} wt={wt.shape} | base={len(base)} model={len(model_feats)} cv={len(cv)}"
    )

    rel_all = _reliability(mt, wt, cols)
    rel_model = rel_all.loc[model_feats]
    _summary("ALL 501 CalMorph features", rel_all)
    _summary("MODEL 278 base features (what pearson_per_gene scores)", rel_model)
    _summary("CV 220 statistics (not modeled)", rel_all.loc[cv])

    ceil = rel_model["ceiling"].mean()
    observed = _observed_best(os.environ["EXPERIMENT_ROOT"])
    obs = observed.get("roll_max")
    print("\n" + "=" * 64)
    if obs is None:
        print("OBSERVED morph per-gene: no leaderboard; run pull_round_leaderboards.py")
    else:
        print(
            f"OBSERVED morph per-feature (run {observed['run_id']}, peak epoch "
            f"{observed['epoch_at_roll_max']:.0f} of {observed['epochs']:.0f}): {obs:.4f}"
        )
    print(f"CEILING  morph per-gene (278 feats, target noise) : {ceil:.3f}")
    if obs is not None:
        print(f"fraction of ceiling realized: {obs / ceil:.1%}")
        print(f"headroom (ceiling - observed): {ceil - obs:.3f}")
    print("=" * 64)
    # top predictable features (where signal is real)
    top = rel_model.sort_values("ceiling", ascending=False).head(12)
    print("\nMost-reliable modeled features (ceiling | reliability | label):")
    for k, row in top.iterrows():
        print(
            f"  {k:10s} {row['ceiling']:.3f} | {row['reliability']:.3f} | {CALMORPH_LABELS.get(k, '')[:52]}"
        )

    descriptions = _si_descriptions(os.environ["DATA_ROOT"])
    annotated = _annotate(rel_model, mt, descriptions)
    n_not_sourced = int((annotated["description"] == NOT_SOURCED).sum())
    print(
        f"\nSI descriptions: {len(descriptions)} rows parsed from {DESCRIPTION_SI}; "
        f"{n_not_sourced} of {len(annotated)} model features not sourced"
    )

    shortlist = annotated.sort_values("score", ascending=False).head(20)
    print("\nScalar-target shortlist (score | ceiling | robust CV | SI description):")
    for k, row in shortlist.iterrows():
        print(
            f"  {k:10s} {row['score']:.3f} | {row['ceiling']:.3f} | {row['robust_cv']:.3f} | "
            f"{str(row['description'])[:60]}"
        )

    size = annotated.loc[SIZE_FEATURES].sort_values("rank")
    print(
        "\nCell size and shape features (score | ceiling | robust CV | rank | SI description):"
    )
    for k, row in size.iterrows():
        print(
            f"  {k:10s} {row['score']:.4f} | {row['ceiling']:.3f} | {row['robust_cv']:.4f} | "
            f"{int(row['rank']):3d} | {str(row['description'])[:48]}"
        )

    results_dir = osp.join(
        os.environ["EXPERIMENT_ROOT"], "019-simb-multimodal", "results"
    )
    os.makedirs(results_dir, exist_ok=True)
    annotated.to_csv(osp.join(results_dir, "morphology_feature_ceiling.csv"))
    repo_root = osp.dirname(  # experiments/019-simb-multimodal/scripts -> repo root
        osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
    )
    tex_path = osp.join(
        repo_root, "notes-tex", "019-simb-multimodal", "tables", "t-morph-scalar.tex"
    )
    with open(tex_path, "w") as fh:
        fh.write(_shortlist_tex(shortlist))
    print(f"-> {tex_path}")
    payload = {
        "n_mutants": int(len(mt)),
        "n_wt_replicates": int(len(wt)),
        "n_base_features": len(base),
        "n_model_features": len(model_feats),
        "dropped_features": DROPPED,
        "ceiling_mean_model_features": float(ceil),
        "ceiling_median_model_features": float(rel_model["ceiling"].median()),
        "reliability_mean_model_features": float(rel_model["reliability"].mean()),
        "n_model_features_ceiling_above_0p5": int((rel_model["ceiling"] > 0.5).sum()),
        "ceiling_mean_all_501": float(rel_all["ceiling"].mean()),
        "observed_best": observed,
        "fraction_of_ceiling_realized": (None if obs is None else float(obs / ceil)),
        "description_source": {
            "citation_key": LIBRARY_KEY,
            "doi": "10.1073/pnas.0509436102",
            "artifact": DESCRIPTION_SI,
            "table": 'Table 2 "parameter statistics", column "parameter description"',
            "sha256": DESCRIPTION_SI_SHA256,
            "retrieval": "pdftotext -layout (born-digital text layer)",
            "n_rows_parsed": len(descriptions),
            "n_model_features_not_sourced": n_not_sourced,
        },
        "n_features_with_rank": int(annotated["score"].notna().sum()),
        "scalar_shortlist": _records(shortlist),
        "size_features": _records(size),
    }
    with open(osp.join(results_dir, "morphology_noise_ceiling.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n-> {osp.join(results_dir, 'morphology_noise_ceiling.json')}")


if __name__ == "__main__":
    main()
