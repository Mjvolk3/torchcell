# experiments/036-dataset-fixes-before-kg-build/scripts/hackett2020_repeat_t0_check.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.hackett2020_repeat_t0_check]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/hackett2020_repeat_t0_check.py
"""Test whether Hackett 2020's three repeat t = 0 hybridizations are inside the IDEA t0.

GEO GSE142864 carries 1,768 samples; three are a second hybridization at t = 0 of a
time course the IDEA tall table carries with ONE t = 0 array (``SMY57_HAP1_t0_rep``,
``SMY26_MSN2__rep_t0``, ``SMY27_MSN4_rep_t0``). The per-replicate sweep of 2026-11-24
(#1178) left three questions open, and this script measures each one on sha256-pinned
bytes only (the two GEO series matrices and the tall table of the raw mirror):

1. What the series-matrix VALUE is. Each GSM column is keyed by 6,316 Agilent probe
   ids, not genes. Every probe is paired with the IDEA gene whose log2(red / green)
   correlates best with it across the GPL16244 arrays that match one IDEA array
   uniquely by (strain, TF, minutes); the slope of VALUE on the IDEA ratio over the
   probes paired at r > 0.95 says which scale it is.
2. How closely a single GEO array reproduces its IDEA array: the median over genes of
   |VALUE - IDEA log2(red / green)| for every uniquely matched array (the calibration).
3. Whether the IDEA t0 of the three courses is the primary t0 array, the repeat, or a
   combination: the same statistic against each array alone and against their mean.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/hackett2020_repeat_t0_check.json``.

    PYTHONPATH=<worktree> python experiments/036-dataset-fixes-before-kg-build/scripts/hackett2020_repeat_t0_check.py
"""

from __future__ import annotations

import csv
import gzip
import os
import os.path as osp
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from torchcell.datasets.scerevisiae import hackett2020 as h
from torchcell.literature.manifest import sha256_file

RESULTS = osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))),
    "results",
    "hackett2020_repeat_t0_check.json",
)
#: A GEO title: strain, TF (``DAL80_P``, ``GCN4_4ZEV``, ``CBF1_M`` carry a suffix the
#: tall table's TF does not), an optional repeat mark, minutes.
_TITLE = re.compile(
    r"^(?P<strain>[^_]+)_(?P<tf>.+?)_+(?P<rep>rep_)?t(?P<t>[0-9.]+)(?P<rep2>_rep)?$"
)
#: Probes paired to a gene at this correlation or above enter the comparisons; a gene
#: paired with more than one such probe is left out, so each gene is one probe.
PAIR_MIN_R = 0.99
SLOPE_MIN_R = 0.95
COURSES = (
    ("SMY57", "HAP1", "GSM4245039", "GSM4245040"),
    ("SMY26", "MSN2", "GSM4245183", "GSM4245184"),
    ("SMY27", "MSN4", "GSM4245195", "GSM4245196"),
)


class Sample(BaseModel):
    """One GEO sample's header fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gsm: str
    platform: str
    title: str
    source: str
    strain: str
    tf: str
    minutes: float
    repeat: bool
    supplementary_file: str


class CourseCheck(BaseModel):
    """One course's IDEA t0 against its two GEO t0 hybridizations."""

    model_config = ConfigDict(extra="forbid")

    strain: str
    tf: str
    primary: Sample
    repeat: Sample
    n_genes: int
    median_abs_diff_primary: float
    median_abs_diff_repeat: float
    median_abs_diff_mean_of_both: float
    pearson_primary_vs_repeat: float
    later_arrays_median_abs_diff: dict[str, float]


class Result(BaseModel):
    """Everything the 2026.11.24 note section cites."""

    model_config = ConfigDict(extra="forbid")

    sha256: dict[str, str]
    n_geo_samples: int
    n_repeat_samples: int
    n_probes: int
    n_paired_arrays_gpl16244: int
    n_probes_paired_r95: int
    n_genes_one_probe_r99: int
    value_slope_on_idea_log2_ratio: float
    value_intercept: float
    value_pearson: float
    calibration_n_arrays: int
    calibration_quantiles: dict[str, float]
    calibration_t0_n_arrays: int
    calibration_t0_quantiles: dict[str, float]
    calibration_t0_at_or_above_smallest_course_diff: list[str]
    courses: list[CourseCheck]


def read_matrix(path: Path) -> tuple[list[Sample], pd.DataFrame]:
    """A series matrix's sample headers and its probe x GSM VALUE table."""
    meta: dict[str, list[str]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("!series_matrix_table_begin"):
                break
            parts = next(csv.reader([line.rstrip("\n")], delimiter="\t"), [])
            # The first of a repeated key is kept (the strain row of the two
            # ``!Sample_characteristics_ch1`` lines; titles carry the strain anyway).
            if parts and parts[0] not in meta:
                meta[parts[0]] = parts[1:]
        table = pd.read_csv(handle, sep="\t", comment="!", index_col=0)
    samples: list[Sample] = []
    for i, gsm in enumerate(meta["!Sample_geo_accession"]):
        title = meta["!Sample_title"][i]
        match = _TITLE.fullmatch(title)
        if match is None:
            raise ValueError(f"unparseable title {title!r}")
        samples.append(
            Sample(
                gsm=gsm,
                platform=meta["!Sample_platform_id"][i],
                title=title,
                source=meta["!Sample_source_name_ch1"][i],
                strain=match["strain"],
                tf=re.sub(r"_(?:P|N|M|4ZEV)$", "", match["tf"]),
                minutes=float(match["t"]),
                repeat=bool(match["rep"] or match["rep2"]),
                supplementary_file=meta["!Sample_supplementary_file"][i].rsplit("/")[
                    -1
                ],
            )
        )
    return samples, table


def _z(a: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    a = a - a.mean(axis=1, keepdims=True)
    out: np.ndarray[Any, Any] = a / np.linalg.norm(a, axis=1, keepdims=True)
    return out


def _quantiles(values: np.ndarray[Any, Any]) -> dict[str, float]:
    return {
        f"q{int(q * 100):02d}": float(np.quantile(values, q))
        for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
    }


def main() -> None:
    """Measure and write the JSON."""
    load_dotenv()
    raw = Path(os.environ["DATA_ROOT"]) / h.RAW_DIR_REL
    pins = {
        h.GEO_GPL16244_REL: h.GEO_GPL16244_SHA256,
        h.GEO_GPL20785_REL: h.GEO_GPL20785_SHA256,
        h.TALL_REL: h.TALL_ZIP_SHA256,
    }
    for rel, pin in pins.items():
        if sha256_file(raw / rel) != pin:
            raise RuntimeError(f"{rel} does not match its pin")
    s1, m1 = read_matrix(raw / h.GEO_GPL16244_REL)
    s2, m2 = read_matrix(raw / h.GEO_GPL20785_REL)
    if list(m1.index) != list(m2.index):
        raise RuntimeError("the two platforms list different probes")
    samples = s1 + s2
    values = pd.concat([m1, m2], axis=1)

    tall = h.read_tall(raw / h.TALL_REL)
    tall["lr2"] = np.log2(tall["red_median"] / tall["green_median"])
    tall["minutes"] = tall["time"].astype(float)
    # One row per array: the course key (with its date) and the minutes.
    arrays = tall[[*h.TALL_COLUMNS[:5], "minutes"]].drop_duplicates()
    n_idea = Counter(
        zip(
            arrays["strain"].astype(str),
            arrays["TF"].astype(str),
            arrays["minutes"],
            strict=True,
        )
    )
    ratio = tall.pivot_table(
        index="GeneName",
        columns=["strain", "TF", "minutes"],
        values="lr2",
        aggfunc="first",
        observed=True,
    )

    def unique(s: Sample) -> bool:
        return not s.repeat and n_idea.get((s.strain, s.tf, s.minutes), 0) == 1

    paired = [s for s in s1 if unique(s)]
    gene_matrix = ratio[[(s.strain, s.tf, s.minutes) for s in paired]].to_numpy()
    probe_matrix = values[[s.gsm for s in paired]].to_numpy()
    corr = _z(probe_matrix) @ _z(gene_matrix).T
    best = corr.argmax(axis=1)
    pairs = pd.DataFrame(
        {"probe": values.index, "gene": ratio.index[best], "r": corr.max(axis=1)}
    )
    r95 = pairs[pairs["r"] > SLOPE_MIN_R]
    x = ratio.loc[r95["gene"], [(s.strain, s.tf, s.minutes) for s in paired]]
    y = values.loc[r95["probe"], [s.gsm for s in paired]]
    slope, intercept = np.polyfit(x.to_numpy().ravel(), y.to_numpy().ravel(), 1)
    pearson = np.corrcoef(x.to_numpy().ravel(), y.to_numpy().ravel())[0, 1]
    one = pairs[pairs["r"] > PAIR_MIN_R].drop_duplicates("gene", keep=False)

    def diff(gsm_values: np.ndarray[Any, Any], key: tuple[str, str, float]) -> float:
        idea = ratio[key].loc[one["gene"]].to_numpy()
        return float(np.median(np.abs(gsm_values - idea)))

    calibration = {
        s.gsm: diff(
            values.loc[one["probe"], s.gsm].to_numpy(), (s.strain, s.tf, s.minutes)
        )
        for s in samples
        if unique(s)
    }
    # The calibration leaves out the three courses' own primary t0 arrays.
    t0_of = {
        s.gsm: calibration[s.gsm]
        for s in samples
        if unique(s)
        and s.minutes == 0.0
        and s.gsm not in {primary for _, _, primary, _ in COURSES}
    }
    t0 = list(t0_of.values())
    by_gsm = {s.gsm: s for s in samples}
    courses: list[CourseCheck] = []
    for strain, tf, primary, repeat in COURSES:
        a = values.loc[one["probe"], primary].to_numpy(dtype=float)
        b = values.loc[one["probe"], repeat].to_numpy(dtype=float)
        courses.append(
            CourseCheck(
                strain=strain,
                tf=tf,
                primary=by_gsm[primary],
                repeat=by_gsm[repeat],
                n_genes=len(one),
                median_abs_diff_primary=diff(a, (strain, tf, 0.0)),
                median_abs_diff_repeat=diff(b, (strain, tf, 0.0)),
                median_abs_diff_mean_of_both=diff((a + b) / 2, (strain, tf, 0.0)),
                pearson_primary_vs_repeat=float(np.corrcoef(a, b)[0, 1]),
                later_arrays_median_abs_diff={
                    f"{s.minutes:g}": calibration[s.gsm]
                    for s in samples
                    if s.strain == strain and s.minutes > 0 and s.gsm in calibration
                },
            )
        )
    result = Result(
        sha256=pins,
        n_geo_samples=len(samples),
        n_repeat_samples=sum(s.repeat for s in samples),
        n_probes=len(values),
        n_paired_arrays_gpl16244=len(paired),
        n_probes_paired_r95=len(r95),
        n_genes_one_probe_r99=len(one),
        value_slope_on_idea_log2_ratio=float(slope),
        value_intercept=float(intercept),
        value_pearson=float(pearson),
        calibration_n_arrays=len(calibration),
        calibration_quantiles=_quantiles(np.array(list(calibration.values()))),
        calibration_t0_n_arrays=len(t0),
        calibration_t0_quantiles=_quantiles(np.array(t0)),
        calibration_t0_at_or_above_smallest_course_diff=sorted(
            f"{by_gsm[g].title} ({g}) {d:.4f}"
            for g, d in t0_of.items()
            if d
            >= min(
                min(c.median_abs_diff_primary, c.median_abs_diff_repeat)
                for c in courses
            )
        ),
        courses=courses,
    )
    os.makedirs(osp.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as handle:
        handle.write(result.model_dump_json(indent=2) + "\n")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
