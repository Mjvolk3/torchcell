# experiments/036-dataset-fixes-before-kg-build/scripts/hackett2020_release_inventory.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.hackett2020_release_inventory]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/hackett2020_release_inventory.py
"""Measure the Hackett 2020 IDEA releases, their strains and their cross-sources.

Every number in the 2026.10.11 section of ``[[torchcell.datasets.scerevisiae.hackett2020]]``
that is not a verifier row comes from this script. It reads only sha256-pinned bytes:
the IDEA tall table, Table EV1, Dataset EV2 and the two GEO series matrices from the
raw mirror, Dataset EV1 from the library mirror.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/hackett2020_release_inventory.json``.

    PYTHONPATH=<worktree> python experiments/036-dataset-fixes-before-kg-build/scripts/hackett2020_release_inventory.py
"""

from __future__ import annotations

import os
import os.path as osp
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from torchcell.datasets.scerevisiae import hackett2020 as h
from torchcell.literature.manifest import sha256_file
from torchcell.verification.induction_timecourse import geo_samples

RESULTS = osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))),
    "results",
    "hackett2020_release_inventory.json",
)
KEY = list(h.TALL_COLUMNS[:5])


class TallShape(BaseModel):
    """The IDEA site table as the loader reads it."""

    model_config = ConfigDict(extra="forbid")

    sha256: str
    n_rows: int
    n_timecourses: int
    n_arrays: int
    n_genes: int
    n_tfs: int
    n_strain_tokens: int
    arrays_per_timecourse: dict[str, int]
    sampling_minutes: list[float]
    restriction_courses: dict[str, int]
    mechanism_courses: dict[str, int]
    repeated_tfs: dict[str, int]
    t0_nonzero_by_layer: dict[str, int]
    nonzero_series_by_layer: dict[str, int]
    noise_sd_min: float
    reference_channel_min: float
    sample_channel_min: float
    n_reference_channel_below_2: int


class StrainShape(BaseModel):
    """Table EV1 and how the release's strain tokens map onto it."""

    model_config = ConfigDict(extra="forbid")

    sha256: str
    n_rows: int
    rows_by_type: dict[str, int]
    rows_by_background: dict[str, int]
    release_tokens_not_plain: dict[str, str]
    unlisted_strains: dict[str, int]
    table_rows_without_a_course: list[str]


class ValidationShape(BaseModel):
    """Dataset EV2."""

    model_config = ConfigDict(extra="forbid")

    sha256: str
    n_rows: int
    tfs: list[str]
    arrays_per_timecourse: dict[str, int]
    date_cells: int


class GeoShape(BaseModel):
    """The GEO GSE142864 series matrices' samples."""

    model_config = ConfigDict(extra="forbid")

    sha256: dict[str, str]
    n_samples: int
    replicate_t0_titles: list[str]
    idea_arrays_without_sample: int
    samples_not_in_idea: int


class Ev1Shape(BaseModel):
    """Dataset EV1 against the tall table's shrunken column."""

    model_config = ConfigDict(extra="forbid")

    sha256: str
    n_arrays: int
    n_cells: int
    max_abs_diff: float
    cells_not_9_decimal_rounding: int
    zero_pattern_mismatches: int


class Inventory(BaseModel):
    """Everything the dendron note cites, measured."""

    model_config = ConfigDict(extra="forbid")

    tall: TallShape
    strains: StrainShape
    validation: ValidationShape
    geo: GeoShape
    ev1: Ev1Shape


def _course_ids(frame: pd.DataFrame) -> pd.Series:
    parts = [frame[c].astype(str) for c in KEY]
    out = parts[0]
    for part in parts[1:]:
        out = out + "_" + part
    return out


def tall_shape(path: Path, tall: pd.DataFrame) -> TallShape:
    courses = tall.groupby(KEY, observed=True)["time"].nunique()
    exps = tall[KEY].drop_duplicates()
    by_tf = Counter(str(t) for t in exps["TF"])
    layers = [
        "log2_ratio",
        "log2_cleaned_ratio",
        "log2_cleaned_ratio_zth2d",
        "log2_selected_timecourses",
        "log2_shrunken_timecourses",
    ]
    t0 = tall[tall["time"] == 0]
    nonzero_series = {}
    for layer in layers:
        rows = tall.loc[tall[layer] != 0, [*KEY, "GeneName"]].drop_duplicates()
        nonzero_series[layer] = len(rows)
    return TallShape(
        sha256=sha256_file(path),
        n_rows=len(tall),
        n_timecourses=len(courses),
        n_arrays=int(tall.groupby([*KEY, "time"], observed=True).ngroups),
        n_genes=int(tall["GeneName"].nunique()),
        n_tfs=int(tall["TF"].nunique()),
        n_strain_tokens=int(tall["strain"].nunique()),
        arrays_per_timecourse={
            str(k): v for k, v in sorted(Counter(courses.tolist()).items())
        },
        sampling_minutes=sorted(float(t) for t in tall["time"].unique()),
        restriction_courses=dict(Counter(str(r) for r in exps["restriction"])),
        mechanism_courses=dict(Counter(str(m) for m in exps["mechanism"])),
        repeated_tfs={tf: n for tf, n in sorted(by_tf.items()) if n > 1},
        t0_nonzero_by_layer={c: int((t0[c] != 0).sum()) for c in layers},
        nonzero_series_by_layer=nonzero_series,
        noise_sd_min=float(tall["log2_noise_model"].min()),
        reference_channel_min=float(tall["green_median"].min()),
        sample_channel_min=float(tall["red_median"].min()),
        n_reference_channel_below_2=int((tall["green_median"] < 2).sum()),
    )


def strain_shape(path: Path, tall: pd.DataFrame) -> StrainShape:
    rows = h.read_strain_table(path)
    specs = {name: h.parse_strain(row) for name, row in rows.items()}
    tokens = sorted(str(s) for s in tall["strain"].unique())
    not_plain = {t: h.table_strain_of(t) for t in tokens if h.table_strain_of(t) != t}
    unlisted = {
        t: int(tall.loc[tall["strain"] == t, "time"].nunique())
        for t in tokens
        if h.table_strain_of(t) not in rows
    }
    used = {h.table_strain_of(t) for t in tokens} | {
        name for name, spec in specs.items() if spec.row.strain_type == "Prediction"
    }
    return StrainShape(
        sha256=sha256_file(path),
        n_rows=len(rows),
        rows_by_type=dict(Counter(r.strain_type for r in rows.values())),
        rows_by_background=dict(Counter(s.background for s in specs.values())),
        release_tokens_not_plain=not_plain,
        unlisted_strains=unlisted,
        table_rows_without_a_course=sorted(set(rows) - used),
    )


def validation_shape(path: Path, genes: list[str]) -> ValidationShape:
    raw = pd.read_excel(path, sheet_name=h.VALIDATION_SHEET)
    dates = int(raw["GeneName"].map(lambda v: not isinstance(v, str)).sum())
    frame = h.read_validation(path, genes)
    per = frame.groupby(["TF", "restriction", "mechanism"])["time"].nunique()
    return ValidationShape(
        sha256=sha256_file(path),
        n_rows=len(frame),
        tfs=sorted(str(t) for t in frame["TF"].unique()),
        arrays_per_timecourse={str(k): v for k, v in Counter(per.tolist()).items()},
        date_cells=dates,
    )


def geo_shape(raw: Path, tall: pd.DataFrame, validation: pd.DataFrame) -> GeoShape:
    paths = {
        h.GEO_GPL16244_REL: raw / h.GEO_GPL16244_REL,
        h.GEO_GPL20785_REL: raw / h.GEO_GPL20785_REL,
    }
    samples = geo_samples(paths.values())
    pairs = {(s, t) for s, t, _ in samples}
    idea = {
        (str(r.strain), float(r.time))
        for r in tall[["strain", "time"]].drop_duplicates().itertuples(index=False)
    }
    rows = h.read_strain_table(raw / h.STRAINS_REL)
    prediction = {
        h.parse_strain(row).insertion.gene_token.upper(): name  # type: ignore[union-attr]
        for name, row in rows.items()
        if row.strain_type == "Prediction"
    }
    valid = {
        (prediction[str(r.TF).upper()], float(r.time))
        for r in validation[["TF", "time"]].drop_duplicates().itertuples(index=False)
    }
    return GeoShape(
        sha256={rel: sha256_file(p) for rel, p in paths.items()},
        n_samples=sum(samples.values()),
        replicate_t0_titles=sorted(t for _, _, t in samples if "rep" in t),
        idea_arrays_without_sample=len(idea - pairs),
        samples_not_in_idea=len(pairs - idea - valid),
    )


def ev1_shape(path: Path, tall: pd.DataFrame) -> Ev1Shape:
    ev1 = pd.read_excel(path, sheet_name="Dataset EV1")
    names = [h.EXCEL_DATE_GENE if not isinstance(v, str) else v for v in ev1["GENE"]]
    course = _course_ids(tall).to_numpy()
    index = pd.DataFrame({"course": course, "time": tall["time"].to_numpy()})
    blocks = index.groupby(["course", "time"]).indices
    genes = tall["GeneName"].astype(str).to_numpy()
    values = tall["log2_shrunken_timecourses"].to_numpy()
    position = {g: i for i, g in enumerate(names)}
    worst = 0.0
    not9 = 0
    zero = 0
    cells = 0
    for column in ev1.columns[1:]:
        tc, time = str(column).rsplit("_", 1)
        rows = blocks[(tc, float(time))]
        stored = np.empty(len(names))
        stored[[position[g] for g in genes[rows]]] = values[rows]
        released = ev1[column].to_numpy(dtype=float)
        cells += len(released)
        worst = max(worst, float(np.max(np.abs(released - stored))))
        not9 += int((released != np.round(stored, 9)).sum())
        zero += int(((released == 0) != (stored == 0)).sum())
    return Ev1Shape(
        sha256=sha256_file(path),
        n_arrays=len(ev1.columns) - 1,
        n_cells=cells,
        max_abs_diff=worst,
        cells_not_9_decimal_rounding=not9,
        zero_pattern_mismatches=zero,
    )


def main() -> None:
    """Measure every release and write the inventory JSON."""
    load_dotenv()
    data_root = Path(os.environ["DATA_ROOT"])
    raw = data_root / h.RAW_DIR_REL
    library = data_root / "torchcell-library" / h.CITATION_KEY
    tall = h.read_tall(raw / h.TALL_REL)
    genes = sorted(str(g) for g in tall["GeneName"].unique())
    validation = h.read_validation(raw / h.VALIDATION_REL, genes)
    inventory = Inventory(
        tall=tall_shape(raw / h.TALL_REL, tall),
        strains=strain_shape(raw / h.STRAINS_REL, tall),
        validation=validation_shape(raw / h.VALIDATION_REL, genes),
        geo=geo_shape(raw, tall, validation),
        ev1=ev1_shape(library / h.SHRUNKEN_EV1_LIBRARY, tall),
    )
    os.makedirs(osp.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as handle:
        handle.write(inventory.model_dump_json(indent=2) + "\n")
    print(inventory.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
