# torchcell/verification/induction_timecourse
# [[torchcell.verification.induction_timecourse]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/induction_timecourse
# Test file: tests/torchcell/verification/test_induction_timecourse.py
"""L0-L4 verifier for induction time-course expression datasets (Hackett 2020 IDEA).

A record is one two-colour array of a chemostat time course after a regulator was
induced. The microarray verifier assumes a deletion screen against a common reference
(``deletion_downregulates``, one gene universe shared with Sameith), which does not hold
here, so this family has its own rules:

- L0 ``structural``: every experiment and reference validates as its concrete class,
  and ``EXPERIMENT_TYPE_MAP`` reconstructs the class from the record's tag.
- L1 ``count``; ``array_completeness`` (the stored (time course, minutes) set is the
  release's arrays minus the refused courses, each stored once); ``gene_completeness``
  (every layer of every record carries the same 6,175 keys, a bijection with the
  release's gene names); ``layer_gaps`` (a layer the release does not carry is a typed
  gap, and only then); plus the family-agnostic census of
  :class:`~torchcell.verification.common.SharedRecordRules`.
- L2 ``release_fidelity`` (every stored value of every layer equals its released cell
  bit for bit, re-read from ``raw/``), ``value_fidelity`` (intensities and noise
  non-negative, everything finite) and ``sampling_time`` (``duration_hours`` x 60 is
  the release's minutes).
- L3 ``t0_zero`` (every log2 layer is 0 at t = 0 and in every reference),
  ``layer_nesting`` (shrunken support within selected within thresholded, and the
  thresholded value equals the cleaned one), ``inducer_timing`` (no estradiol at t = 0,
  1 uM after), ``induction_upregulates`` (the induced gene's unprocessed log2 ratio is
  positive late in its course), ``genotype_matches_release`` and ``sourced_values``.
- L4 ``shrunken_matches_dataset_ev1`` (the stored shrunken layer equals the paper's
  Dataset EV1, a separately hosted file), ``release_matches_paper`` (the release
  reproduces the paper's counts: 217 courses, 1,693 arrays, 6,175 genes, 118,134
  thresholded and 100,036 selected time series), ``geo_sample_coverage`` (every array
  is a GEO GSE142864 sample of the same strain and time) and
  ``induced_gene_containment`` (the induced genes are S288C genes).
"""

from __future__ import annotations

import csv
import gzip
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from statistics import median
from typing import Any, cast

import numpy as np
import pandas as pd

from torchcell.datamodels.schema import (
    EXPERIMENT_TYPE_MAP,
    InductionTimecourseExpressionExperiment,
    InductionTimecourseExpressionExperimentReference,
)
from torchcell.datasets.scerevisiae import hackett2020 as h
from torchcell.verification.common import SharedRecordRules
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
    sha256_file,
)
from torchcell.verification.sourced import SourcedValue

Record = dict[str, Any]

#: Layers that are log2 ratios relative to t = 0 (zero there by construction).
LOG2_LAYERS = (
    "log2_cleaned",
    "log2_ratio_unprocessed",
    "log2_thresholded",
    "log2_selected",
    "log2_shrunken",
)
NON_NEGATIVE_LAYERS = (
    "log2_noise_sd",
    "sample_channel_intensity",
    "reference_channel_intensity",
)
ALL_LAYERS = tuple(h.TALL_LAYERS)
#: The paper's and Appendix's counts the release must reproduce (see the quotes).
PAPER_TIMECOURSES = 217
PAPER_ARRAYS = 1693
PAPER_GENES = 6175
PAPER_THRESHOLDED_SERIES = 118_134
PAPER_SELECTED_SERIES = 100_036
#: The EV1 workbook rounds what the tall TSV writes in full: to 9 decimals, and below
#: about 1e-4 to 3 significant digits in scientific form (-3.485000141e-05 is EV1's
#: -3.49e-05). Measured 2026-10-11 over all 10,454,275 cells: max |EV1 - tall|
#: 4.99986e-08, 10,906 cells not equal to the 9-decimal rounding, and the zero pattern
#: identical. Half a unit of the coarser rounding is the bound.
EV1_TOLERANCE = 5e-8
#: An induced gene counts as up when its unprocessed log2 ratio at the course's last
#: array is above 0; the oracle is that more than this share of courses are up.
MIN_UP_FRACTION = 0.9


def _result(
    level: Level, name: str, passed: bool, message: str, **details: Any
) -> LevelResult:
    return LevelResult(
        level=level, name=name, passed=passed, message=message, details=details
    )


def _minutes(experiment: Mapping[str, Any]) -> float:
    """The release minutes a record's ``duration_hours`` encodes (rounded to 1e-9)."""
    return round(float(experiment["environment"]["duration_hours"]) * 60.0, 9)


# --------------------------------------------------------------------------- #
# Release re-read
# --------------------------------------------------------------------------- #
class ReleaseIndex:
    """The two released tables keyed by (time course, minutes) -> gene-ordered rows."""

    def __init__(self, raw_dir: Path) -> None:
        """Re-read the tall table and Dataset EV2 from the dataset's ``raw/``."""
        tall = h.read_tall(raw_dir / h.TALL_ZIP_NAME)
        genes = sorted(str(g) for g in tall["GeneName"].unique())
        validation = h.read_validation(raw_dir / h.VALIDATION_NAME, genes)
        self.genes = genes
        self.tables: dict[str, pd.DataFrame] = {
            h.TALL_ZIP_NAME: tall,
            h.VALIDATION_NAME: validation,
        }
        self.blocks: dict[tuple[str, float], tuple[str, np.ndarray[Any, Any]]] = {}
        key = list(h.TALL_COLUMNS[:5])
        tall_indices = tall.groupby([*key, "time"], sort=False, observed=True).indices
        for raw, rows in tall_indices.items():
            group = cast(tuple[Any, ...], raw)
            tc = h.timecourse_id(*(str(v) for v in group[:5]))
            self.blocks[(tc, float(group[5]))] = (h.TALL_ZIP_NAME, np.asarray(rows))
        valid_indices = validation.groupby(
            ["TF", "restriction", "mechanism", "time"], sort=False
        ).indices
        for raw, rows in valid_indices.items():
            group = cast(tuple[Any, ...], raw)
            tc = h.validation_timecourse_id(*(str(v) for v in group[:3]))
            self.blocks[(tc, float(group[3]))] = (h.VALIDATION_NAME, np.asarray(rows))
        self.course_strain: dict[str, str] = {}
        for raw in tall.groupby(key, sort=False, observed=True).indices:
            group = cast(tuple[Any, ...], raw)
            tc = h.timecourse_id(*(str(v) for v in group))
            self.course_strain[tc] = str(group[1])
        self.minutes_of: dict[str, set[float]] = {}
        for tc, minutes in self.blocks:
            self.minutes_of.setdefault(tc, set()).add(minutes)
        self.names = {
            name: frame["GeneName"].astype(str).to_numpy()
            for name, frame in self.tables.items()
        }
        self.columns = {
            name: {
                column: frame[column].to_numpy(dtype=float)
                for column in self.layers()[name].values()
            }
            for name, frame in self.tables.items()
        }

    def layers(self) -> Mapping[str, Mapping[str, str]]:
        """Stored field -> released column, per table."""
        return {h.TALL_ZIP_NAME: h.TALL_LAYERS, h.VALIDATION_NAME: h.VALIDATION_LAYERS}

    def values(
        self, tc: str, minutes: float, field: str
    ) -> tuple[list[str], np.ndarray[Any, Any]] | None:
        """The released column of one array in its row order, or None if not released."""
        table, rows = self.blocks[(tc, minutes)]
        column = self.layers()[table].get(field)
        if column is None:
            return None
        return list(self.names[table][rows]), self.columns[table][column][rows]

    def tall_signal_series(self, column: str, courses: Iterable[str] | None) -> int:
        """Time series (course x gene) with a non-zero value in ``column``."""
        tall = self.tables[h.TALL_ZIP_NAME]
        key = list(h.TALL_COLUMNS[:5])
        nonzero = tall[column].to_numpy() != 0
        frame = tall.loc[nonzero, [*key, "GeneName"]].drop_duplicates()
        if courses is not None:
            wanted = set(courses)
            ids = [
                h.timecourse_id(*(str(v) for v in row))
                for row in frame[key].itertuples(index=False)
            ]
            return sum(1 for i in ids if i in wanted)
        return len(frame)


# --------------------------------------------------------------------------- #
# Sourced values
# --------------------------------------------------------------------------- #
def _text_of(path: Path) -> str:
    """The text a quote is checked against: gunzipped for a ``.gz`` file."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8", errors="replace")


def _strain_rows_text(path: Path) -> set[str]:
    """Table EV1 rendered as the loader quotes it: one tab-joined line per row."""
    sheet = pd.read_excel(path, sheet_name=h.STRAINS_SHEET, header=None, dtype=str)
    return {"\t".join(str(v) for v in row) for row in sheet.itertuples(index=False)}


def audit_sourced(
    values: Iterable[SourcedValue], *, library_root: Path, data_root: Path
) -> LevelResult:
    """Every quote: its file hashes to the pin and holds the quote verbatim.

    A raw-mirror source (``torchcell-raw/...``) resolves under ``data_root``, a library
    one under ``library_root/<citation key>/``; a gzip is read decompressed and the
    Table EV1 workbook as its row rendering.
    """
    checked = 0
    failures: list[dict[str, Any]] = []
    cache: dict[tuple[Path, str | None], Any] = {}
    seen: set[tuple[str, str]] = set()
    for sv in values:
        uri = sv.provenance.source_uri
        if (uri, sv.quote) in seen:
            continue
        seen.add((uri, sv.quote))
        path = (
            data_root / uri
            if uri.startswith("torchcell-raw/")
            else library_root / str(sv.provenance.citation_key) / uri
        )
        checked += 1
        key = (path, sv.provenance.sha256)
        if key not in cache:
            ok = path.exists() and sha256_file(path) == sv.provenance.sha256
            text: Any = None
            if ok:
                text = (
                    _strain_rows_text(path)
                    if path.suffix in {".xls", ".xlsx"}
                    else _text_of(path)
                )
            cache[key] = (ok, text)
        ok, text = cache[key]
        if not ok:
            failures.append({"source": uri, "problem": "missing or sha256 drift"})
        elif sv.quote not in text:
            failures.append({"source": uri, "quote": sv.quote[:120]})
    return _result(
        Level.L3,
        "sourced_values",
        not failures,
        f"{checked} quotes verbatim in their pinned sources"
        if not failures
        else f"{len(failures)}/{checked} quotes failed",
        n_checked=checked,
        failed_sources=sorted({f["source"] for f in failures}),
        failures=failures[:20],
    )


def _record_sourced_values(record: Record) -> Iterator[SourcedValue]:
    """Every SourcedValue a stored record carries (background, media, culture)."""

    def walk(obj: Any) -> Iterator[SourcedValue]:
        if isinstance(obj, dict):
            if {"value", "provenance", "quote"} <= set(obj) and isinstance(
                obj.get("provenance"), dict
            ):
                yield SourcedValue.model_validate(obj)
                return
            for key, value in obj.items():
                if key.startswith("log2_") or key.endswith("_intensity"):
                    continue
                yield from walk(value)
        elif isinstance(obj, list):
            for value in obj:
                yield from walk(value)

    yield from walk(record["experiment"]["environment"])
    yield from walk(record["reference"]["genome_reference"])


# --------------------------------------------------------------------------- #
# GEO
# --------------------------------------------------------------------------- #
_GEO_TIME = re.compile(r"_(?:rep_)?t(?P<t>[0-9.]+)(?:_rep)?$")


def geo_samples(paths: Iterable[Path]) -> Counter[tuple[str, float, str]]:
    """(strain, minutes, title) of every GEO sample, from the series-matrix headers."""
    out: Counter[tuple[str, float, str]] = Counter()
    for path in paths:
        titles: list[str] = []
        strains: list[str] = []
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("!series_matrix_table_begin"):
                    break
                if line.startswith("!Sample_title"):
                    titles = next(csv.reader([line.rstrip("\n")], delimiter="\t"))[1:]
                elif (
                    line.startswith("!Sample_characteristics_ch1") and "strain:" in line
                ):
                    strains = next(csv.reader([line.rstrip("\n")], delimiter="\t"))[1:]
        if len(titles) != len(strains):
            raise ValueError(f"{path}: {len(titles)} titles, {len(strains)} strains")
        for title, strain in zip(titles, strains, strict=True):
            match = _GEO_TIME.search(title)
            if match is None:
                raise ValueError(f"{path}: unparseable GEO title {title!r}")
            out[(strain.removeprefix("strain: "), float(match.group("t")), title)] += 1
    return out


# --------------------------------------------------------------------------- #
# The verifier
# --------------------------------------------------------------------------- #
def verify_induction_timecourse(
    records: Callable[[], Iterator[Record]],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
    raw_dir: Path,
    preprocess_dir: Path,
    library_root: Path,
    data_root: Path,
    sgd_genes: set[str] | None,
) -> VerificationReport:
    """Run the L0-L4 gate over a built store (``records`` yields it, twice)."""
    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    release = ReleaseIndex(raw_dir)
    ledger = h.ReleaseLedger.model_validate_json(
        (preprocess_dir / "release_ledger.json").read_text()
    )
    # Every layer is keyed by the release's GeneName token verbatim (2026.10.12).
    key_of = {name: name for name in release.genes}
    shared = SharedRecordRules()

    n = 0
    l0_failures: list[dict[str, Any]] = []
    stored: Counter[tuple[str, float]] = Counter()
    gene_short: list[dict[str, Any]] = []
    gap_problems: list[dict[str, Any]] = []
    fidelity_bad: list[dict[str, Any]] = []
    n_values = 0
    value_bad: list[dict[str, Any]] = []
    time_bad: list[dict[str, Any]] = []
    t0_bad: list[dict[str, Any]] = []
    nesting_bad: list[dict[str, Any]] = []
    inducer_bad: list[dict[str, Any]] = []
    genotype_bad: list[dict[str, Any]] = []
    last_unprocessed: dict[str, tuple[float, float, str]] = {}
    induced_genes: set[str] = set()
    sourced: dict[tuple[str, str], SourcedValue] = {}
    shrunken_of: dict[tuple[str, float], dict[str, float]] = {}
    seen_refs: set[str] = set()
    expected_keys = set(key_of.values())

    for record in records():
        n += 1
        experiment, reference = record["experiment"], record["reference"]
        try:
            InductionTimecourseExpressionExperiment.model_validate(experiment)
            InductionTimecourseExpressionExperimentReference.model_validate(reference)
            if EXPERIMENT_TYPE_MAP[experiment["experiment_type"]] is not (
                InductionTimecourseExpressionExperiment
            ):
                raise ValueError("experiment_type does not reconstruct the class")
        except (ValueError, TypeError, KeyError) as err:
            l0_failures.append({"index": n - 1, "error": str(err)[:300]})
        shared.add(record)
        phen = experiment["phenotype"]
        tc, minutes = phen["timecourse_id"], _minutes(experiment)
        stored[(tc, minutes)] += 1
        # gene completeness + gaps
        gapped = {g["field"] for g in phen.get("provenance_gaps") or []}
        table = phen["release_table"]
        released = set(release.layers()[table])
        for field in ALL_LAYERS:
            layer = phen.get(field)
            if layer is None:
                if field in released or field not in gapped:
                    gap_problems.append({"tc": tc, "t": minutes, "field": field})
                continue
            if field in gapped or field not in released:
                gap_problems.append({"tc": tc, "t": minutes, "field": field})
            if set(layer) != expected_keys:
                gene_short.append({"tc": tc, "t": minutes, "field": field})
            values = release.values(tc, minutes, field)
            if values is not None:
                names, arr = values
                got = np.array([layer[key_of[g]] for g in names], dtype=float)
                n_values += len(got)
                if not np.array_equal(got, arr):
                    fidelity_bad.append(
                        {
                            "tc": tc,
                            "t": minutes,
                            "field": field,
                            "n_diff": int((got != arr).sum()),
                        }
                    )
            vals = np.fromiter(layer.values(), dtype=float, count=len(layer))
            if not np.isfinite(vals).all() or (
                field in NON_NEGATIVE_LAYERS and (vals < 0).any()
            ):
                value_bad.append({"tc": tc, "t": minutes, "field": field})
            if field in LOG2_LAYERS and minutes == 0.0 and (vals != 0).any():
                t0_bad.append({"tc": tc, "field": field, "n": int((vals != 0).sum())})
        _check_nesting(phen, tc, minutes, nesting_bad)
        if phen.get("log2_shrunken") is not None:
            shrunken_of[(tc, minutes)] = phen["log2_shrunken"]
        # time + inducer
        if minutes not in release.minutes_of.get(tc, set()):
            time_bad.append({"tc": tc, "t": minutes})
        perturbations = experiment["environment"]["perturbations"]
        if minutes == 0.0:
            if perturbations:
                inducer_bad.append({"tc": tc, "t": minutes, "problem": "inducer at t0"})
        elif not (
            len(perturbations) == 1
            and perturbations[0]["compound"]["name"] == "estradiol"
            and perturbations[0]["concentration"]["value"] == h.INDUCER_UM
            and perturbations[0]["concentration"]["unit"] == "uM"
        ):
            inducer_bad.append(
                {"tc": tc, "t": minutes, "problem": "not 1 uM estradiol"}
            )
        media_name = experiment["environment"]["media"]["name"]
        restriction = tc.split("_")[-2]
        if media_name != h.MEDIA_BY_RESTRICTION[restriction].name:
            inducer_bad.append({"tc": tc, "problem": f"medium {media_name}"})
        # genotype
        perts = experiment["genotype"]["perturbations"]
        tf = tc.split("_")[0]
        if tf in {"GEV", "Z3EV"}:
            if perts:
                genotype_bad.append(
                    {"tc": tc, "problem": "control with a perturbation"}
                )
        elif (
            len(perts) != 1
            or perts[0]["perturbation_type"] != "inducible_promoter_insertion"
            or perts[0]["systematic_gene_name"] != h.induced_orf(tf)
        ):
            genotype_bad.append({"tc": tc, "problem": "insertion does not name the TF"})
        else:
            gene = perts[0]["systematic_gene_name"]
            induced_genes.add(gene)
            # the layers are keyed by the release's own token (the TF column)
            value = phen["log2_ratio_unprocessed"][tf]
            if tc not in last_unprocessed or minutes > last_unprocessed[tc][0]:
                last_unprocessed[tc] = (minutes, value, gene)
        # reference
        ref_id = reference["phenotype_reference"]["timecourse_id"]
        if ref_id != tc:
            genotype_bad.append({"tc": tc, "problem": f"reference course {ref_id}"})
        if ref_id not in seen_refs:
            seen_refs.add(ref_id)
            ref_phen = reference["phenotype_reference"]
            for field in LOG2_LAYERS:
                layer = ref_phen.get(field)
                if layer is not None and any(v != 0 for v in layer.values()):
                    t0_bad.append({"reference": ref_id, "field": field})
            if reference["environment_reference"]["perturbations"]:
                inducer_bad.append({"reference": ref_id, "problem": "inducer"})
        for sv in _record_sourced_values(record):
            sourced.setdefault((sv.provenance.source_uri, sv.quote), sv)

    expected_arrays = {
        k
        for k in release.blocks
        if k[0] not in {r.timecourse_id for r in ledger.refused}
    }
    # ---- L0 / L1 ----------------------------------------------------------- #
    report.add(
        _result(
            Level.L0,
            "structural",
            not l0_failures,
            f"{n} records validated"
            if not l0_failures
            else f"{len(l0_failures)} failed",
            n_records=n,
            failures=l0_failures[:10],
        )
    )
    report.add(
        _result(
            Level.L1,
            "count",
            n == expected_count,
            f"observed {n}, expected {expected_count}",
            observed=n,
            expected=expected_count,
        )
    )
    dup = [k for k, v in stored.items() if v > 1]
    missing = sorted(expected_arrays - set(stored))
    extra = sorted(set(stored) - expected_arrays)
    report.add(
        _result(
            Level.L1,
            "array_completeness",
            not dup and not missing and not extra,
            f"{len(stored)} of {len(expected_arrays)} released arrays stored once "
            f"({len(release.blocks) - len(expected_arrays)} arrays of "
            f"{len(ledger.refused)} refused courses excluded)",
            n_stored=len(stored),
            n_expected=len(expected_arrays),
            duplicated=[list(k) for k in dup[:10]],
            missing=[list(k) for k in missing[:10]],
            extra=[list(k) for k in extra[:10]],
        )
    )
    bijection = len(set(key_of.values())) == len(key_of) == len(release.genes)
    report.add(
        _result(
            Level.L1,
            "gene_completeness",
            not gene_short and bijection,
            f"every layer of every record carries the {len(expected_keys)} keys, a "
            "bijection with the release's gene names"
            if not gene_short and bijection
            else f"{len(gene_short)} layers short; bijection {bijection}",
            n_genes=len(expected_keys),
            short=gene_short[:10],
        )
    )
    report.add(
        _result(
            Level.L1,
            "layer_gaps",
            not gap_problems,
            "a layer is absent exactly when its release does not carry it, and then "
            "as a typed gap"
            if not gap_problems
            else f"{len(gap_problems)} layer/gap mismatches",
            problems=gap_problems[:10],
        )
    )
    for result in shared.results():
        report.add(result)
    # ---- L2 ---------------------------------------------------------------- #
    report.add(
        _result(
            Level.L2,
            "release_fidelity",
            not fidelity_bad,
            f"{n_values} stored values equal their released cells bit for bit"
            if not fidelity_bad
            else f"{len(fidelity_bad)} arrays differ from the release",
            n_values=n_values,
            bad=fidelity_bad[:10],
        )
    )
    report.add(
        _result(
            Level.L2,
            "value_fidelity",
            not value_bad,
            "every value finite; intensities and noise SDs non-negative"
            if not value_bad
            else f"{len(value_bad)} layers invalid",
            bad=value_bad[:10],
        )
    )
    report.add(
        _result(
            Level.L2,
            "sampling_time",
            not time_bad,
            "duration_hours x 60 is a released time of the record's course"
            if not time_bad
            else f"{len(time_bad)} records at an unreleased time",
            bad=time_bad[:10],
        )
    )
    # ---- L3 ---------------------------------------------------------------- #
    report.add(
        _result(
            Level.L3,
            "t0_zero",
            not t0_bad,
            "every log2 layer is 0 at t = 0 and in every reference"
            if not t0_bad
            else f"{len(t0_bad)} t = 0 layers non-zero",
            bad=t0_bad[:10],
        )
    )
    report.add(
        _result(
            Level.L3,
            "layer_nesting",
            not nesting_bad,
            "shrunken support within selected within thresholded; thresholded equals "
            "cleaned where non-zero"
            if not nesting_bad
            else f"{len(nesting_bad)} arrays break the nesting",
            bad=nesting_bad[:10],
        )
    )
    report.add(
        _result(
            Level.L3,
            "inducer_timing",
            not inducer_bad,
            "no estradiol at t = 0 or in a reference, 1 uM estradiol after; medium is "
            "the course's limitation"
            if not inducer_bad
            else f"{len(inducer_bad)} environments wrong",
            bad=inducer_bad[:10],
        )
    )
    ups = [v for (_, v, _) in last_unprocessed.values()]
    up_fraction = sum(1 for v in ups if v > 0) / len(ups) if ups else 0.0
    med = median(ups) if ups else math.nan
    report.add(
        _result(
            Level.L3,
            "induction_upregulates",
            up_fraction > MIN_UP_FRACTION,
            f"the induced gene is up at the last array of {up_fraction:.3f} of "
            f"{len(ups)} courses (median log2 {med:.2f}, {2**med:.1f}-fold; the paper's "
            "median saturation is 53-fold)",
            n_courses=len(ups),
            up_fraction=up_fraction,
            median_log2=med,
            down=sorted(
                [tc, v] for tc, (_, v, _) in last_unprocessed.items() if v <= 0
            )[:20],
        )
    )
    report.add(
        _result(
            Level.L3,
            "genotype_matches_release",
            not genotype_bad,
            "every course's genotype is the release's induced gene (none for a control)"
            if not genotype_bad
            else f"{len(genotype_bad)} genotypes disagree",
            bad=genotype_bad[:10],
        )
    )
    quotes = list(h.SOURCED_VALUES.values()) + list(sourced.values())
    for media in h.MEDIA_BY_RESTRICTION.values():
        quotes.extend(media.provenance)
        quotes.extend(sv for c in media.components for sv in c.provenance)
    report.add(audit_sourced(quotes, library_root=library_root, data_root=data_root))
    # ---- L4 ---------------------------------------------------------------- #
    report.add(_ev1_result(shrunken_of, key_of, library_root))
    report.add(_paper_counts_result(release, ledger))
    report.add(_geo_result(stored, release, data_root))
    if sgd_genes is not None:
        outside = sorted(induced_genes - sgd_genes)
        report.add(
            _result(
                Level.L4,
                "induced_gene_containment",
                not outside,
                f"all {len(induced_genes)} induced genes are S288C genes"
                if not outside
                else f"{len(outside)} induced genes outside S288C",
                outside=outside,
            )
        )
    return report


def _check_nesting(
    phen: Mapping[str, Any], tc: str, minutes: float, bad: list[dict[str, Any]]
) -> None:
    thresholded = phen.get("log2_thresholded")
    selected = phen.get("log2_selected")
    shrunken = phen.get("log2_shrunken")
    if thresholded is None or selected is None or shrunken is None:
        return
    cleaned = phen["log2_cleaned"]
    for gene, value in shrunken.items():
        if value != 0 and selected[gene] == 0:
            bad.append({"tc": tc, "t": minutes, "gene": gene, "rule": "shrunken"})
            return
    for gene, value in selected.items():
        if value != 0 and thresholded[gene] == 0:
            bad.append({"tc": tc, "t": minutes, "gene": gene, "rule": "selected"})
            return
    for gene, value in thresholded.items():
        if value != 0 and value != cleaned[gene]:
            bad.append({"tc": tc, "t": minutes, "gene": gene, "rule": "thresholded"})
            return


def _ev1_result(
    shrunken_of: Mapping[tuple[str, float], Mapping[str, float]],
    key_of: Mapping[str, str],
    library_root: Path,
) -> LevelResult:
    """The stored shrunken layer against the paper's Dataset EV1 workbook."""
    path = library_root / h.CITATION_KEY / h.SHRUNKEN_EV1_LIBRARY
    if sha256_file(path) != h.SHRUNKEN_EV1_SHA256:
        return _result(
            Level.L4, "shrunken_matches_dataset_ev1", False, "EV1 sha256 drift"
        )
    ev1 = pd.read_excel(path, sheet_name="Dataset EV1")
    names = [
        h.EXCEL_DATE_GENE if not isinstance(v, str) else v for v in ev1["GENE"].tolist()
    ]
    columns: dict[tuple[str, float], str] = {}
    for column in ev1.columns[1:]:
        tc, time = str(column).rsplit("_", 1)
        columns[(tc, float(time))] = column
    missing = sorted(k for k in shrunken_of if k not in columns)
    worst = 0.0
    n = 0
    zero_mismatch = 0
    for key, layer in shrunken_of.items():
        if key not in columns:
            continue
        values = ev1[columns[key]].to_numpy(dtype=float)
        stored = np.array([layer[key_of[g]] for g in names], dtype=float)
        n += len(values)
        worst = max(worst, float(np.max(np.abs(values - stored))))
        zero_mismatch += int(((values == 0) != (stored == 0)).sum())
    passed = (
        not missing
        and worst <= EV1_TOLERANCE
        and zero_mismatch == 0
        and len(set(names)) == len(names)
    )
    return _result(
        Level.L4,
        "shrunken_matches_dataset_ev1",
        passed,
        f"{n} stored shrunken values match Dataset EV1 within {EV1_TOLERANCE:g} "
        f"(max |diff| {worst:.2e}), zeros where EV1 has zeros; EV1 holds "
        f"{len(columns)} arrays"
        if passed
        else f"{len(missing)} arrays not in EV1; max |diff| {worst:.2e}; "
        f"{zero_mismatch} zero-pattern mismatches",
        n_values=n,
        max_abs_diff=worst,
        zero_pattern_mismatches=zero_mismatch,
        missing=[list(k) for k in missing[:10]],
        n_ev1_arrays=len(columns),
    )


def _paper_counts_result(release: ReleaseIndex, ledger: h.ReleaseLedger) -> LevelResult:
    """The release reproduces the paper's and the Appendix's counts."""
    tall = release.tables[h.TALL_ZIP_NAME]
    key = list(h.TALL_COLUMNS[:5])
    observed = {
        "timecourses": int(tall.groupby(key, observed=True).ngroups),
        "arrays": int(tall.groupby([*key, "time"], observed=True).ngroups),
        "genes": len(release.genes),
        "thresholded_series": release.tall_signal_series(
            "log2_cleaned_ratio_zth2d", None
        ),
        "selected_series": release.tall_signal_series(
            "log2_selected_timecourses", None
        ),
        "shrunken_series": release.tall_signal_series(
            "log2_shrunken_timecourses", None
        ),
    }
    expected = {
        "timecourses": PAPER_TIMECOURSES,
        "arrays": PAPER_ARRAYS,
        "genes": PAPER_GENES,
        "thresholded_series": PAPER_THRESHOLDED_SERIES,
        "selected_series": PAPER_SELECTED_SERIES,
        "shrunken_series": PAPER_SELECTED_SERIES,
    }
    passed = observed == expected and ledger.n_tall_arrays == PAPER_ARRAYS
    return _result(
        Level.L4,
        "release_matches_paper",
        passed,
        "the release reproduces 217 courses, 1,693 arrays, 6,175 genes, 118,134 "
        "thresholded and 100,036 selected and shrunken time series"
        if passed
        else "the release departs from the paper's counts",
        observed=observed,
        expected=expected,
    )


def _geo_result(
    stored: Mapping[tuple[str, float], int], release: ReleaseIndex, data_root: Path
) -> LevelResult:
    """Every stored array is a GEO GSE142864 sample of its strain at its time."""
    raw = data_root / h.RAW_DIR_REL
    pins = {
        h.GEO_GPL16244_REL: h.GEO_GPL16244_SHA256,
        h.GEO_GPL20785_REL: h.GEO_GPL20785_SHA256,
    }
    for rel, pin in pins.items():
        if sha256_file(raw / rel) != pin:
            return _result(
                Level.L4, "geo_sample_coverage", False, f"{rel} sha256 drift"
            )
    samples = geo_samples(raw / rel for rel in pins)
    by_strain_time = {(s, t) for s, t, _ in samples}
    titles_of: dict[tuple[str, float], set[str]] = {}
    for strain, minutes, title in samples:
        titles_of.setdefault((strain, minutes), set()).add(title)
    tables = h.read_strain_table(raw / h.STRAINS_REL)
    prediction = {
        h.parse_strain(row).insertion.gene_token.upper(): name  # type: ignore[union-attr]
        for name, row in tables.items()
        if row.strain_type == "Prediction"
    }
    missing: list[list[Any]] = []
    for tc, minutes in stored:
        if tc in release.course_strain:
            if (release.course_strain[tc], minutes) not in by_strain_time:
                missing.append([tc, minutes])
            continue
        tf = tc.split("_")[0]
        strain = prediction[tf.upper()]
        titles = titles_of.get((strain, minutes), set())
        if not any(t.split("_")[1].upper() == tf.upper() for t in titles):
            missing.append([tc, minutes])
    return _result(
        Level.L4,
        "geo_sample_coverage",
        not missing,
        f"all {len(stored)} arrays are GEO GSE142864 samples of the same strain and "
        f"time ({sum(samples.values())} GEO samples)"
        if not missing
        else f"{len(missing)} arrays have no GEO sample",
        missing=missing[:20],
        n_geo_samples=sum(samples.values()),
    )


__all__ = [
    "ReleaseIndex",
    "audit_sourced",
    "geo_samples",
    "verify_induction_timecourse",
]
