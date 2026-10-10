# experiments/036-dataset-fixes-before-kg-build/scripts/hawkins2020_release_inventory.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.hawkins2020_release_inventory]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/hawkins2020_release_inventory

r"""What Hawkins 2020 (mismatch-CRISPRi) released, measured off the pinned workbook.

Everything the loader, the PR and the note claim about the release is regenerated here
from ``si/si4.xlsx`` (Table S3, sha256-pinned in the torchcell-library mirror) and the
pinned MG1655 / BW25113 genomes:

1. **Shape.** Rows, columns, distinct spacers, parent spacers, genes, series sizes.
2. **Measured versus predicted.** Which columns are a measurement (the replicate mean and
   SD of relative fitness) and which are a model output (``relative fitness (predicted)``,
   which is the linear model's predicted sgRNA ACTIVITY: exactly 1.0 for every fully
   complementary spacer). The predicted activity is the schema finding.
3. **The 10-to-15-doubling sheet.** Its SD column is byte-equal to the 10-doubling sheet's,
   and its non-targeting controls spread 2.3-fold wider; the loader refuses it.
4. **Identifiers.** The released b-numbers through MG1655 and the ECK crosswalk to the
   BW25113 host, checked against the authors' own ``BW25113_`` tags on the 10-15 sheet.
5. **Duplication.** Shared fully complementary spacers against the Wang 2018 and Cui 2018
   dev stores, and the two prior-study columns the workbook itself carries.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/hawkins2020_release_inventory.py
"""

import json
import os
import os.path as osp
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from torchcell.datasets.bacteria_common import (  # noqa: E402
    bacterial_genome,
    eck_crosswalk,
    reconcile_locus_tags,
)
from torchcell.sequence.genome.ecoli.k12 import (  # noqa: E402
    EcoliK12BW25113Genome,
    EcoliK12MG1655Genome,
)
from torchcell.verification.runners import stream_records  # noqa: E402

DATA_ROOT = os.environ["DATA_ROOT"]
KEY = "hawkinsMismatchCRISPRiRevealsCovarying2020"
WORKBOOK = osp.join(DATA_ROOT, "torchcell-library", KEY, "si", "si4.xlsx")
WORKBOOK_SHA256 = "a9aa39f576e0d240e353e104e0cf94af7a2c5d9e8a375ef8d56cf07694d7412a"
RESULTS = osp.join(
    os.environ["EXPERIMENT_ROOT"], "036-dataset-fixes-before-kg-build", "results"
)
OUT = osp.join(RESULTS, "hawkins2020_release_inventory.json")

MEAN = "relative fitness (mean)"
SD = "relative fitness (stddev)"
PRED = "relative fitness (predicted)"
WANG_COL = "Log2FC from Wang et al., 2018"
ROUSSET_COL = "relative fitness from Rousset et al., 2018"

#: Dev stores of the two E. coli CRISPRi guide screens already on main.
DEV_STORES: dict[str, str] = {
    "CrispriGuideFitnessWang2018Dataset": "crispri_guide_fitness_wang2018",
    "CrispriKnockdownCui2018Dataset": "crispri_knockdown_cui2018",
}


def _sha256(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pearson(a: pd.Series, b: pd.Series) -> float:
    return float(np.corrcoef(a.to_numpy(float), b.to_numpy(float))[0, 1])


def _mismatch(variant: str, original: str) -> list[int]:
    return [i for i, (v, o) in enumerate(zip(variant, original, strict=True)) if v != o]


def shape(sheets: dict[str, pd.DataFrame]) -> dict[str, Any]:
    eco = sheets["relative fitness (eco data)"]
    series = eco.groupby("original").size()
    mm = [_mismatch(v, o) for v, o in zip(eco["variant"], eco["original"], strict=True)]
    positions = Counter(p[0] for p in mm if len(p) == 1)
    return {
        "sheets": {name: list(frame.shape) for name, frame in sheets.items()},
        "eco_columns": list(eco.columns),
        "eco_rows": len(eco),
        "distinct_variants": int(eco["variant"].nunique()),
        "distinct_parent_spacers": int(eco["original"].nunique()),
        "distinct_locus_tags": int(eco["locus_tag"].nunique()),
        "mismatch_count": dict(Counter(len(p) for p in mm)),
        "single_mismatch_position_index_0_based_from_5prime": dict(
            sorted(positions.items())
        ),
        "rows_per_series": {
            str(size): int(count) for size, count in series.value_counts().items()
        },
        "series_per_gene": {
            str(size): int(count)
            for size, count in eco.groupby("locus_tag")["original"]
            .nunique()
            .value_counts()
            .sort_index()
            .items()
        },
        "pam": dict(eco["pam"].value_counts()),
    }


def measured_vs_predicted(eco: pd.DataFrame) -> dict[str, Any]:
    parent = eco["variant"] == eco["original"]
    mean = eco[MEAN].notna()
    sd = eco[SD].notna()
    retained = eco["family_retained"].astype(bool)
    return {
        "predicted_on_parent_spacers": eco.loc[parent, PRED].describe().to_dict(),
        "predicted_on_single_mismatch": eco.loc[~parent, PRED].describe().to_dict(),
        "family_retained": {
            "true": int(retained.sum()),
            "false": int((~retained).sum()),
        },
        "mean_present": int(mean.sum()),
        "mean_absent_read_floor": int((~mean).sum()),
        "sd_present": int(sd.sum()),
        "mean_present_sd_absent": int((mean & ~sd).sum()),
        "sd_present_mean_absent": int((sd & ~mean).sum()),
        "mean_present_by_family_retained": {
            "true": int((mean & retained).sum()),
            "false": int((mean & ~retained).sum()),
        },
        "mean_negative": int((eco[MEAN] < 0).sum()),
        "mean_min": float(eco[MEAN].min()),
        "mean_max": float(eco[MEAN].max()),
        "parent_spacers_with_mean": int((mean & parent).sum()),
        "single_mismatch_with_mean": int((mean & ~parent).sum()),
    }


def window_10_15(sheets: dict[str, pd.DataFrame]) -> dict[str, Any]:
    eco = sheets["relative fitness (eco data)"]
    late = sheets["10-15 relfit (eco)"]
    ctrl = sheets["relative fitness (eco controls)"]
    late_ctrl = sheets["10-15 relfit (eco controls)"]
    if not (eco["variant"].to_numpy() == late["variant"].to_numpy()).all():
        raise RuntimeError("the two windows are not row-aligned")
    both_sd = eco[SD].notna() & late[SD].notna()
    both_mean = eco[MEAN].notna() & late[MEAN].notna()
    both_ctrl_sd = ctrl[SD].notna() & late_ctrl[SD].notna()
    return {
        "rows_aligned": True,
        "late_mean_present": int(late[MEAN].notna().sum()),
        "sd_present_in_both": int(both_sd.sum()),
        "sd_equal_to_10_doubling_sd": int(
            np.isclose(eco.loc[both_sd, SD], late.loc[both_sd, SD]).sum()
        ),
        "control_sd_equal": int(
            np.isclose(
                ctrl.loc[both_ctrl_sd, SD], late_ctrl.loc[both_ctrl_sd, SD]
            ).sum()
        ),
        "control_sd_present_in_both": int(both_ctrl_sd.sum()),
        "control_mean_sd_10_doubling": float(ctrl[MEAN].std()),
        "control_mean_sd_10_to_15": float(late_ctrl[MEAN].std()),
        "control_mean_present_10_doubling": int(ctrl[MEAN].notna().sum()),
        "control_mean_present_10_to_15": int(late_ctrl[MEAN].notna().sum()),
        "means_in_both": int(both_mean.sum()),
        "pearson_10_vs_10_to_15": _pearson(
            eco.loc[both_mean, MEAN], late.loc[both_mean, MEAN]
        ),
        "authors_bw25113_tags": late["locus_tag"].nunique(),
    }


def identifiers(sheets: dict[str, pd.DataFrame]) -> dict[str, Any]:
    eco = sheets["relative fitness (eco data)"]
    late = sheets["10-15 relfit (eco)"]
    mg = bacterial_genome("ecoli", "MG1655", DATA_ROOT)
    bw = bacterial_genome("ecoli", "BW25113", DATA_ROOT)
    assert isinstance(mg, EcoliK12MG1655Genome)
    assert isinstance(bw, EcoliK12BW25113Genome)
    tags = pd.Series(sorted(eco["locus_tag"].unique()))
    stored, report = reconcile_locus_tags(mg, tags, label="hawkins2020 b-numbers")
    pair_of = {pair.mg1655: pair for pair in eck_crosswalk(mg, bw).pairs}
    authors = dict(zip(eco["locus_tag"], late["locus_tag"], strict=True))
    crossed: dict[str, str] = {}
    no_partner: list[str] = []
    for source, mg_tag in zip(tags, stored, strict=True):
        pair = pair_of.get(str(mg_tag))
        if pair is None:
            no_partner.append(str(source))
        else:
            crossed[str(source)] = pair.bw25113
    disagree = {s: (t, authors[s]) for s, t in crossed.items() if authors[s] != t}
    return {
        "released_b_numbers": len(tags),
        "mg1655_resolved_fraction": float(report.resolved_fraction),
        "mg1655_retired": list(report.retired_kept),
        "mg1655_remapped": report.remapped,
        "eck_crossed": len(crossed),
        "no_eck_partner": no_partner,
        "authors_bw25113_tag_disagreements": disagree,
        "numerics_disagree": sorted(
            s for s, t in crossed.items() if t.split("_")[1] != s[1:]
        ),
    }


def duplication(eco: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    measured = eco[eco[MEAN].notna()].set_index("variant")[MEAN]
    for name, dirname in DEV_STORES.items():
        root = osp.join(DATA_ROOT, "data", "torchcell", dirname)
        per_screen: dict[str, dict[str, float]] = {}
        for record in stream_records(root):
            experiment = record["experiment"]
            (pert,) = experiment["genotype"]["perturbations"][:1]
            spacer = pert["crispr"]["guide_sequence"]
            if spacer in measured.index:
                screen = str(experiment["phenotype"]["screen_id"])
                per_screen.setdefault(screen, {})[spacer] = float(
                    experiment["phenotype"]["environment_response"]
                )
        out[name] = {
            screen: {
                "shared_spacers": len(values),
                "pearson": _pearson(
                    measured.loc[list(values)], pd.Series(list(values.values()))
                )
                if len(values) > 2
                else None,
            }
            for screen, values in sorted(per_screen.items())
        }
    for column in (WANG_COL, ROUSSET_COL):
        both = eco[MEAN].notna() & eco[column].notna()
        out[f"workbook column {column!r}"] = {
            "rows": int(eco[column].notna().sum()),
            "rows_with_hawkins_mean": int(both.sum()),
            "pearson": _pearson(eco.loc[both, MEAN], eco.loc[both, column]),
            "single_mismatch_rows": int(
                (eco[column].notna() & (eco["variant"] != eco["original"])).sum()
            ),
        }
    return out


def main() -> None:
    got = _sha256(WORKBOOK)
    if got != WORKBOOK_SHA256:
        raise RuntimeError(
            f"{WORKBOOK} sha256 {got} is not the pinned {WORKBOOK_SHA256}"
        )
    sheets = pd.read_excel(WORKBOOK, sheet_name=None, engine="openpyxl")
    eco = sheets["relative fitness (eco data)"]
    result = {
        "workbook": {"path": f"torchcell-library/{KEY}/si/si4.xlsx", "sha256": got},
        "shape": shape(sheets),
        "measured_vs_predicted": measured_vs_predicted(eco),
        "window_10_15": window_10_15(sheets),
        "identifiers": identifiers(sheets),
        "duplication": duplication(eco),
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(
            result,
            handle,
            indent=2,
            default=lambda o: int(o) if isinstance(o, np.integer) else str(o),
        )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
