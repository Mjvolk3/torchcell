# experiments/036-dataset-fixes-before-kg-build/scripts/gasch2000_release_inventory.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.gasch2000_release_inventory]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/gasch2000_release_inventory.py
"""Measure the Gasch 2000 release and classify every array column by its reference.

Every number in the 2026.10.12 section of ``[[torchcell.datasets.scerevisiae.gasch2000]]``
comes from this script. It reads only sha256-pinned bytes from the raw mirror
(``$DATA_ROOT/torchcell-raw/gaschGenomicExpressionPrograms2000/``: the authors'
``complete_dataset.txt`` and the text layer of their web-supplement Materials and
Methods) and the paper's OCR in the library mirror.

Each of the 173 array columns is assigned to one released series by its header, and
each series to the reference its methods name, with the verbatim sentence that names
it (re-read from the pinned bytes here, so a drifted quote fails the run):

- ``same_culture_before_change``: the same culture sampled before the shift, as a
  time-zero aliquot, half of the culture, or the default "unstressed, asynchronous
  culture" the supplement assigns to every series it does not otherwise describe.
- ``other_culture``: a different culture of known identity (cells grown at 33 C; the
  strain carrying the empty vector).
- ``reference_pool``: an equal-mass pool of every sample in the series. The release
  carries these arrays against the pool, not "zero transformed" (measured below: the
  time-zero and 33 C arrays of those series are not 0).
- ``other_publication``: the DeRisi 1997 diauxic-shift arrays the release carries.
- ``unresolved``: a header the methods do not describe or contradict.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/gasch2000_release_inventory.json``.

    PYTHONPATH=<worktree> python experiments/036-dataset-fixes-before-kg-build/scripts/gasch2000_release_inventory.py
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from torchcell.data.experiment_dataset import verify_sha256

RESULTS = Path(__file__).resolve().parent.parent / "results"
CITATION_KEY = "gaschGenomicExpressionPrograms2000"
RELEASE = (
    "data/complete_dataset.txt",
    "3050664e0deba976b70b954676675cce74d3288198d850abb6fb123bac58daa7",
)
METHODS = (
    "si/materials.txt",
    "5402523a710262509ab1cc7bfbf84b09f4a74f145b41fd3c2c715aed73ecaf73",
)
PAPER = ("paper.md", "e22b3e01f5664a00b9a6fdcd4c78bacd5803e5c31f285f45b98c47a871a98fbb")
N_LEADING = 3  # UID, NAME, GWEIGHT

ReferenceKind = Literal[
    "same_culture_before_change",
    "other_culture",
    "reference_pool",
    "other_publication",
    "unresolved",
]


class Series(BaseModel):
    """One released series: the headers that belong to it and the reference it names."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    series: str
    header: str  # regex over the stripped column header
    reference_kind: ReferenceKind
    source: Literal["methods", "paper"]
    quote: str
    note: str = ""


SERIES: tuple[Series, ...] = (
    Series(
        series="hs-1",
        header=r"^Heat Shock \d+ minutes hs-1$",
        reference_kind="same_culture_before_change",
        source="methods",
        quote="isolated from the unstressed, asynchronous culture was labeled with Cy3-dUTP, and",
        note="release labels 5, 10, 15, 20, 30, 40, 60, 80 min; the methods list "
        "5, 10, 15, 20, 30, 40, 50, and 60 minutes",
    ),
    Series(
        series="hs-2",
        header=r"^Heat Shock \d+ ?(minutes|inutes)\s+hs-2$",
        reference_kind="reference_pool",
        source="methods",
        quote="pool was prepared by combining an equal mass of total RNA from each sample, and this",
    ),
    Series(
        series="37C to 25C",
        header=r"^37C to 25C shock - \d+ min$",
        reference_kind="same_culture_before_change",
        source="methods",
        quote="an aliquot was frozen for use as the reference sample",
    ),
    Series(
        series="variable heat to 37C",
        header=r"^heat shock \d+ to 37, 20 minutes$",
        reference_kind="same_culture_before_change",
        source="methods",
        quote="Half of the culture was collected to serve as the unstressed",
    ),
    Series(
        series="29C to 33C",
        header=r"^29C to 33C - \d+ minutes$",
        reference_kind="same_culture_before_change",
        source="methods",
        quote="isolated from the unstressed, asynchronous culture was labeled with Cy3-dUTP, and",
    ),
    Series(
        series="33C vs 30C",
        header=r"^33C vs\. 30C - 90 minutes$",
        reference_kind="unresolved",
        source="methods",
        quote="media at 33oC, and samples were collected at 5, 15, 30 minutes.",
        note="the header names a 30 C reference; the series it sits in starts at 29 C "
        "and the methods sample it at 5, 15 and 30 minutes only",
    ),
    Series(
        series="29C sorbitol to 33C sorbitol",
        header=r"^29C \+1M sorbitol to 33C \+ 1M sorbitol - \d+ minutes$",
        reference_kind="same_culture_before_change",
        source="methods",
        quote="isolated from the unstressed, asynchronous culture was labeled with Cy3-dUTP, and",
    ),
    Series(
        series="29C sorbitol to 33C no sorbitol",
        header=r"^29C \+1M sorbitol to 33C \+ \*NO sorbitol - \d+ minutes$",
        reference_kind="same_culture_before_change",
        source="methods",
        quote="isolated from the unstressed, asynchronous culture was labeled with Cy3-dUTP, and",
    ),
    Series(
        series="H2O2 constant",
        header=r"^constant 0\.32 mM H2O2 \(\d+ min\) (redo|rescan)$",
        reference_kind="same_culture_before_change",
        source="methods",
        quote="Following the removal of 100 mL as a",
    ),
    Series(
        series="menadione",
        header=r"^1 ?mM Menadione \(\d+ min\) ?redo$",
        reference_kind="same_culture_before_change",
        source="methods",
        quote="isolated from the unstressed, asynchronous culture was labeled with Cy3-dUTP, and",
    ),
    Series(
        series="dtt-1",
        header=r"^2\.5mM DTT \d+ min dtt-1$",
        reference_kind="same_culture_before_change",
        source="methods",
        quote="isolated from the unstressed, asynchronous culture was labeled with Cy3-dUTP, and",
    ),
    Series(
        series="dtt-2",
        header=r"^dtt \d+ min\s+dtt-2$",
        reference_kind="reference_pool",
        source="methods",
        quote="Cy5-dUTP and compared to a reference pool, consisting of equal mass of total RNA from",
    ),
    Series(
        series="diamide",
        header=r"^1\.5 mM diamide \(\d+ min\)$",
        reference_kind="same_culture_before_change",
        source="methods",
        quote="isolated from the unstressed, asynchronous culture was labeled with Cy3-dUTP, and",
    ),
    Series(
        series="hyper-osmotic",
        header=r"^1M sorbitol - \d+ min$",
        reference_kind="same_culture_before_change",
        source="methods",
        quote="isolated from the unstressed, asynchronous culture was labeled with Cy3-dUTP, and",
    ),
    Series(
        series="hypo-osmotic",
        header=r"^Hypo-osmotic shock - \d+ min$",
        reference_kind="same_culture_before_change",
        source="methods",
        quote="isolated from the unstressed, asynchronous culture was labeled with Cy3-dUTP, and",
    ),
    Series(
        series="hypo-osmotic steady state",
        header=r"^steady-state 1M sorbitol$",
        reference_kind="unresolved",
        source="methods",
        quote="sorbitol were also collected and resuspended in identical media, containing 1 M sorbitol.",
        note="the array list calls it 'hypo-osmotic steady-state'; the methods describe a "
        "handling control in identical sorbitol medium, and neither names its reference",
    ),
    Series(
        series="amino acid starvation",
        header=r"^aa starv [\d.]+ h$",
        reference_kind="same_culture_before_change",
        source="methods",
        quote="isolated from the unstressed, asynchronous culture was labeled with Cy3-dUTP, and",
    ),
    Series(
        series="nitrogen depletion",
        header=r"^Nitrogen Depletion [\d]+ (min\.|h|d)$",
        reference_kind="same_culture_before_change",
        source="methods",
        quote="isolated from the unstressed, asynchronous culture was labeled with Cy3-dUTP, and",
    ),
    Series(
        series="diauxic shift",
        header=r"^(Diauxic Shift Timecourse - 0 h|diauxic shift timecourse ?[\d.]+( h)?)$",
        reference_kind="other_publication",
        source="paper",
        quote="along with previously-published data (DeRisi et al., 1997)",
    ),
    Series(
        series="ypd-2",
        header=r"^YPD \d+ (h|d)\s+ypd-2$",
        reference_kind="same_culture_before_change",
        source="methods",
        quote="collected to serve as the time = 0 reference.",
    ),
    Series(
        series="ypd-1",
        header=r"^YPD stationary phase \d+ (h|d) ypd-1$",
        reference_kind="same_culture_before_change",
        source="methods",
        quote="collected to serve as the time = 0 reference.",
        note="release labels run to 28 d; the methods list 2 h to 5 d",
    ),
    Series(
        series="mutant heat",
        header=r"^DBY.*(37degree|37degrees).*$",
        reference_kind="same_culture_before_change",
        source="methods",
        quote="isolated from the unstressed, asynchronous culture was labeled with Cy3-dUTP, and",
        note="one msn2 msn4 array is labeled '(real strain)' and its partner is not",
    ),
    Series(
        series="mutant H2O2",
        header=r"^DBY.*H2O2.*$",
        reference_kind="same_culture_before_change",
        source="methods",
        quote="to OD600  0.5, and 50 mL was removed as a reference.",
    ),
    Series(
        series="MSN2 MSN4 overexpression",
        header=r"^Msn[24] overexpression( \(repeat\))?$",
        reference_kind="other_culture",
        source="methods",
        quote="Cy5-dUTP and compared to total RNA from control cells harboring pRS416.",
    ),
    Series(
        series="YAP1 overexpression",
        header=r"^YAP1 overexpression$",
        reference_kind="unresolved",
        source="methods",
        quote="Overexpression constructs pRS-MSN2 and pRS-MSN4 as well as the parent vector",
        note="the methods and Table 1 describe MSN2 and MSN4 constructs only",
    ),
    Series(
        series="car-1",
        header=r"^\w+ vs\. reference pool\s+car-1$",
        reference_kind="reference_pool",
        source="methods",
        quote="dUTP.  For both experiments, a reference pool was created by combining an equal mass",
    ),
    Series(
        series="car-2",
        header=r"^YP \w+ vs reference pool car-2$",
        reference_kind="reference_pool",
        source="methods",
        quote="dUTP.  For both experiments, a reference pool was created by combining an equal mass",
    ),
    Series(
        series="ct-1",
        header=r"^\d+ deg growth ct-1$",
        reference_kind="other_culture",
        source="methods",
        quote="the 33oC sample was used as a reference and labeled in Cy3-dUTP while all other",
    ),
    Series(
        series="ct-2",
        header=r"^steady state \d+ dec C ct-2( \(repeat hyb\))?$",
        reference_kind="reference_pool",
        source="methods",
        quote="pool was created by mixing an equal mass of total RNA from each sample, and this pool",
    ),
)

#: The arrays of the pool-referenced series that WOULD read 0 everywhere if the release
#: were "zero transformed" against them (the time-zero, YP glucose and 33 C arrays).
POOL_ANCHORS = (
    "Heat Shock 000 minutes hs-2",
    "dtt 000 min  dtt-2",
    "YP glucose vs reference pool car-2",
    "steady state 33 dec C ct-2",
)


class ColumnRecord(BaseModel):
    """One released array column, its series and its non-null ratio count."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int
    header: str
    series: str
    reference_kind: ReferenceKind
    n_values: int


def _columns(release: Path) -> tuple[list[str], list[list[str]]]:
    with release.open(newline="", encoding="latin-1") as fh:
        rows = list(csv.reader(fh, delimiter="\t"))
    return rows[0], rows[1:]


def _series_of(header: str) -> Series:
    matches = [s for s in SERIES if re.match(s.header, header.strip())]
    if len(matches) != 1:
        raise ValueError(f"{header!r} matches {[m.series for m in matches]}")
    return matches[0]


def main() -> None:
    """Classify every column, re-read every quote, and write the inventory JSON."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    raw = Path(data_root) / "torchcell-raw" / CITATION_KEY
    library = Path(data_root) / "torchcell-library" / CITATION_KEY
    verify_sha256(raw / RELEASE[0], RELEASE[1])
    verify_sha256(raw / METHODS[0], METHODS[1])
    verify_sha256(library / PAPER[0], PAPER[1])
    texts = {
        "methods": (raw / METHODS[0]).read_text(encoding="utf-8"),
        "paper": (library / PAPER[0]).read_text(encoding="utf-8"),
    }
    for s in SERIES:
        if s.quote not in texts[s.source]:
            raise ValueError(f"{s.series}: quote not found in {s.source}: {s.quote!r}")

    header, rows = _columns(raw / RELEASE[0])
    if any(len(r) != len(header) for r in rows):
        raise ValueError("ragged release rows")
    columns: list[ColumnRecord] = []
    for j in range(N_LEADING, len(header)):
        s = _series_of(header[j])
        n = sum(1 for r in rows if r[j].strip() != "")
        columns.append(
            ColumnRecord(
                index=j,
                header=header[j],
                series=s.series,
                reference_kind=s.reference_kind,
                n_values=n,
            )
        )
    anchors = {}
    for name in POOL_ANCHORS:
        j = header.index(name)
        values = [float(r[j]) for r in rows if r[j].strip() != ""]
        anchors[name] = {
            "n_values": len(values),
            "n_zero": sum(1 for v in values if v == 0.0),
            "mean_abs": sum(abs(v) for v in values) / len(values),
        }
    by_kind: dict[str, dict[str, int]] = {}
    for c in columns:
        entry = by_kind.setdefault(c.reference_kind, {"columns": 0, "values": 0})
        entry["columns"] += 1
        entry["values"] += c.n_values
    by_series: dict[str, int] = {}
    for c in columns:
        by_series[c.series] = by_series.get(c.series, 0) + 1
    out = {
        "release": {
            "path": RELEASE[0],
            "sha256": RELEASE[1],
            "n_genes": len(rows),
            "n_array_columns": len(columns),
            "n_values": sum(c.n_values for c in columns),
        },
        "by_reference_kind": by_kind,
        "pool_anchor_arrays": anchors,
        "series": [
            {**s.model_dump(), "n_columns": by_series[s.series]} for s in SERIES
        ],
        "columns": [c.model_dump() for c in columns],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "gasch2000_release_inventory.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps(
            {k: out[k] for k in ("release", "by_reference_kind", "pool_anchor_arrays")},
            indent=2,
        )
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
