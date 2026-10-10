# experiments/036-dataset-fixes-before-kg-build/scripts/thompsonMassivelyParallelFitness2019_release_inventory.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.thompsonMassivelyParallelFitness2019_release_inventory]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/thompsonMassivelyParallelFitness2019_release_inventory
"""Measure the Thompson 2019 lysine row against the Borchert 2024 compendium by content.

Schedule row "Thompson 2019 lysine" (Thompson MG et al. 2019, mBio 10:e02577-18,
``thompsonMassivelyParallelFitness2019``) grew the JBEI-1 RB-TnSeq library of P. putida
KT2440 on four sole carbon sources. The paper is not in the literature mirror, so its
open-access copy and released tables are deposited into the RAW mirror from the PMC
Article Datasets bucket (``pmc_cloud``, the Mohiuddin 2022 route), sha256-pinned.

The decisive question is duplication. The paper's own per-gene release is Table S1, 39
genes x 4 conditions. Each Table S1 column is compared with EVERY sample column of the
compendium release (``fModule_Metadata.xlsx``, 332 samples) and then with the records
the served ``RbTnseqBorchert2024Dataset`` store holds. A sample is identified with a
Table S1 column only by the values, never by its name or metadata, and the runner-up
sample is recorded so the margin of the identification is visible.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/thompsonMassivelyParallelFitness2019_release_inventory.py
    python experiments/036-dataset-fixes-before-kg-build/scripts/thompsonMassivelyParallelFitness2019_release_inventory.py --deposit --write-record

``--deposit`` retrieves the release files from the PMC bucket into
``$DATA_ROOT/torchcell-raw/thompsonMassivelyParallelFitness2019/`` and writes its
``manifest.json`` (an existing file with another sha256 raises). Without it the script
reads the files already deposited and checks their pins. ``--write-record`` writes the
``subsumption_record.json`` beside them, the Wetmore 2015 / Thompson 2020 pattern.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import os.path as osp
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
    sha256_file,
)
from torchcell.literature.retrieve import pmc_cloud_object, pmc_cloud_url
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import SourcedValue, audit_sourced_value

RESULTS: Final = "experiments/036-dataset-fixes-before-kg-build/results"
SCRIPT: Final = (
    "experiments/036-dataset-fixes-before-kg-build/scripts/"
    "thompsonMassivelyParallelFitness2019_release_inventory.py"
)


def _load_shared() -> ModuleType:
    """The sibling script that defines the subsumption-record types and helpers.

    Loaded by path so this row's record is the SAME pydantic type the three earlier
    rows wrote (``SubsumptionRecord``, ``ReleaseFile``, ``ServedCoverage``), not a
    look-alike copy that could drift.
    """
    path = Path(__file__).with_name("bacteria_subsumed_rows.py")
    spec = importlib.util.spec_from_file_location("bacteria_subsumed_rows", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["bacteria_subsumed_rows"] = module
    spec.loader.exec_module(module)
    return module


shared = _load_shared()

KEY: Final = "thompsonMassivelyParallelFitness2019"
DOI: Final = "10.1128/mBio.02577-18"
TITLE: Final = (
    "Massively Parallel Fitness Profiling Reveals Multiple Novel Enzymes in "
    "Pseudomonas putida Lysine Metabolism"
)
ROW_NAME: Final = "Thompson 2019 lysine"
PMC_PREFIX: Final = "PMC6509195.1"
RETRIEVED_AT: Final = "2026-10-10"
_PMC_TEXT: Final = "PMC open-access plain text of the article (PMC Article Datasets)"
_CSV: Final = "CSV cell of the publisher's released Table S1"


class PmcFile(BaseModel):
    """One file deposited from the PMC Article Datasets bucket, with its pin."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    relpath: str
    sha256: str
    bytes: int
    purpose: str

    @property
    def bucket_key(self) -> str:
        """The bucket key, ``<PMCID>.<version>/<file>``."""
        return f"{PMC_PREFIX}/{self.name}"

    def release_file(self) -> Any:
        """The file as the shared ``ReleaseFile`` type the record lists."""
        return shared.ReleaseFile(
            relpath=self.relpath,
            mirror="torchcell-raw",
            citation_key=KEY,
            role=ROLE_RAW_DATA,
            bytes=self.bytes,
            sha256=self.sha256,
            source_url=pmc_cloud_url(self.bucket_key),
            retrieval_method=RetrievalMethod.pmc_cloud,
            retrieval_command=(
                "torchcell.literature.retrieve.pmc_cloud_object("
                f"key='{self.bucket_key}')"
            ),
            retrieved_at=RETRIEVED_AT,
            purpose=self.purpose,
        )


PAPER_TEXT: Final = PmcFile(
    name=f"{PMC_PREFIX}.txt",
    relpath=f"paper/{PMC_PREFIX}.txt",
    sha256="aedb5fb086aade7b80f848427a90af5478646b9d362b015b6a74a4410fb8dd4c",
    bytes=73809,
    purpose="the article text every quote below is read from: Methods 'RB-TnSeq' "
    "states the library, conditions, vessel and fitness definition; Results states "
    "the 39-gene selection behind Table S1",
)
TABLE_S1: Final = PmcFile(
    name="mBio.02577-18-st001.csv",
    relpath="si/mBio.02577-18-st001.csv",
    sha256="b4718373fdcd6637de7a25eee4a17aedd47c1ad1a07262b1502e2cb0c4d2a0e5",
    bytes=4059,
    purpose="Table S1, the paper's only per-gene fitness release: 39 genes x 4 carbon "
    "sources, compared cell by cell with the compendium and the served store",
)
TABLE_S2: Final = PmcFile(
    name="mBio.02577-18-st002.csv",
    relpath="si/mBio.02577-18-st002.csv",
    sha256="907aed18cfbcc5cd8114636b373107a321263d7a0f10026a8f5fba0f3da6c8af",
    bytes=50030,
    purpose="Table S2, pairwise statistics of wild-type targeted proteomics across "
    "carbon sources; read to measure that it carries test statistics, not abundances",
)
PMC_FILES: Final = (PAPER_TEXT, TABLE_S1, TABLE_S2)

#: Table S1 column -> the compendium carbon source that names the same compound.
TABLE_S1_CONDITIONS: Final[dict[str, str]] = {
    "D-Lysine": "D-Lysine",
    "L-Lysine": "L-Lysine",
    "5-AVA": "5-Aminovaleric acid",
    "Glucose": "D-Glucose",
}
#: The compendium rounds every fitness to three decimals, so a cell copied unchanged
#: differs from Table S1 by at most half of the last place.
ROUNDING_HALF_ULP: Final = 0.0005


def _quote(value: Any, quote: str, *, page: str) -> SourcedValue:
    """Bind a value to a verbatim quote of the pinned PMC article text."""
    return SourcedValue(
        value=value,
        quote=quote,
        provenance=Provenance(
            source_uri=PAPER_TEXT.relpath,
            citation_key=KEY,
            sha256=PAPER_TEXT.sha256,
            method=_PMC_TEXT,
            page=page,
        ),
    )


SOURCED: Final[dict[str, SourcedValue]] = {
    "library": _quote(
        "JBEI-1, a regrown working stock of the Rand 2017 library (Putida_ML5)",
        "P. putida RB-TnSeq library JBEI-1 was created by diluting a 1-ml aliquot of "
        "the previously described P. putida RB-TnSeq library (16) in 500\u2009ml of LB "
        "media supplemented with kanamycin",
        page="Materials and Methods, 'RB-TnSeq'",
    ),
    "time_zero": _quote(
        3,
        "three 1-ml aliquots were removed, pelleted, decanted, and then stored at "
        "−80°C for use as a time zero control",
        page="Materials and Methods, 'RB-TnSeq'",
    ),
    "conditions": _quote(
        {
            "carbon_sources": ["glucose", "5AVA", "d-lysine", "l-lysine"],
            "concentration_mM": 10,
            "medium": "MOPS minimal medium",
            "vessel": "50-ml culture tubes",
            "hours": 48,
            "temperature_C": 30,
        },
        "diluted 1:50 into 10\u2009ml MOPS minimal medium supplemented with 10\u2009mM "
        "glucose or "
        "5AVA or d-lysine or l-lysine. Cells were grown in 50-ml culture tubes for "
        "48\u2009h "
        "at 30°C with shaking at 200\u2009rpm.",
        page="Materials and Methods, 'RB-TnSeq'",
    ),
    "fitness": _quote(
        "log2 ratio of barcode reads, experimental sample over time zero",
        "The fitness of a strain is defined here as the normalized log2 ratio of "
        "barcode reads in the experimental sample to barcode reads in the time zero "
        "sample.",
        page="Materials and Methods, 'RB-TnSeq'",
    ),
    "release": _quote(
        "http://fit.genomics.lbl.gov",
        "All fitness data in this work is publicly available at "
        "http://fit.genomics.lbl.gov.",
        page="Materials and Methods, 'RB-TnSeq'",
    ),
    "table_s1_selection": _quote(
        39,
        "Fitness profiling revealed 39 genes with significant fitness values below "
        "−2 for 5AVA, d-lysine, or l-lysine and no lower than −0.5 for glucose "
        "(Fig.\u00a01A; see also Table\u00a0S1 in the supplemental material).",
        page="Results, 'Identification of lysine catabolism genes via RB-TnSeq'",
    ),
    "table_s1_caption": _quote(
        "Table S1",
        "TABLE\u00a0S1 Quantitative results of RB-TnSeq screen.",
        page="Results, supplemental material legend",
    ),
    "table_s2_caption": _quote(
        "Table S2",
        "All pairwise statistical comparisons of different carbon sources for each "
        "protein can be found in Table\u00a0S2.",
        page="Results, targeted proteomics",
    ),
}


# --------------------------------------------------------------------------- #
# Deposit and pin
# --------------------------------------------------------------------------- #
def raw_root() -> Path:
    """``$DATA_ROOT/torchcell-raw/<KEY>``."""
    return shared.data_root() / shared.RAW_REL / KEY


def deposit() -> Path:
    """Retrieve every PMC file into the raw mirror and write its manifest.

    The retrieved bytes are checked against the pin before they are written, so a
    changed upstream object raises instead of replacing what was measured.
    """
    root = raw_root()
    records: list[ArtifactRecord] = []
    for item in PMC_FILES:
        target = root / item.relpath
        if target.exists():
            got = sha256_file(target)
            if got != item.sha256:
                raise RuntimeError(f"{target} sha256 {got}, pinned {item.sha256}")
        else:
            payload = pmc_cloud_object(item.bucket_key)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            got = sha256_file(target)
            if got != item.sha256:
                raise RuntimeError(
                    f"retrieved {item.bucket_key} sha256 {got}, pinned {item.sha256}"
                )
        records.append(
            ArtifactRecord(
                path=item.relpath,
                role=ROLE_RAW_DATA,
                bytes=target.stat().st_size,
                sha256=item.sha256,
                source=pmc_cloud_url(item.bucket_key),
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.pmc_cloud,
                    source_url=pmc_cloud_url(item.bucket_key),
                    retriever="torchcell.literature.retrieve.pmc_cloud_object",
                    params={"key": item.bucket_key},
                    sha256=item.sha256,
                    retrieved_at=RETRIEVED_AT,
                ),
            )
        )
    path = root / "manifest.json"
    manifest = (
        Manifest.model_validate_json(path.read_text())
        if path.exists()
        else Manifest(citation_key=KEY, doi=DOI, title=TITLE)
    )
    held = {record.path: record.sha256 for record in manifest.files}
    for record in records:
        if record.path not in held:
            manifest.files.append(record)
        elif held[record.path] != record.sha256:
            raise RuntimeError(f"{path} holds {record.path} at another sha256")
    manifest.files.sort(key=lambda record: record.path)
    for url in (pmc_cloud_url(PMC_PREFIX), shared.FITNESS_BROWSER):
        if url not in manifest.si_data_sources:
            manifest.si_data_sources.append(url)
    manifest.si_expected = SI_EXPECTED
    manifest.created_at = manifest.created_at or datetime.now(UTC).isoformat()
    path.write_text(manifest.model_dump_json(indent=2))
    return path


def pinned(item: PmcFile) -> Path:
    """The deposited file's path after checking its bytes against the pin."""
    path = raw_root() / item.relpath
    got = sha256_file(path)
    if got != item.sha256:
        raise RuntimeError(f"{path} sha256 {got}, pinned {item.sha256}")
    return path


SI_EXPECTED: Final = [
    "Text S1 (DOCX, additional methods), Figs S1-S7 (TIF) and the article PDF are not "
    "deposited: no measurement reads them. Table S1 (39 genes x 4 carbon sources) and "
    "Table S2 (proteomics pairwise statistics) are deposited under si/, the PMC plain "
    "text under paper/. The genome-wide fitness this paper reports is released only on "
    "the Fitness Browser; the Borchert 2024 compendium carries the same four samples "
    "(subsumption_record.json)"
]


# --------------------------------------------------------------------------- #
# Measurements
# --------------------------------------------------------------------------- #
class ColumnMatch(BaseModel):
    """One Table S1 column identified with a compendium sample by its values."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    table_s1_column: str
    compendium_sample: str
    compendium_desc: str
    compendium_set: str
    compendium_vessel: str
    compendium_media: str
    compendium_concentration_mM: float
    compendium_total_rep: int
    genes_compared: int
    max_abs_difference: float
    median_abs_difference: float
    cells_within_rounding: int
    pearson_r: float
    runner_up_sample: str
    runner_up_max_abs_difference: float
    same_condition_samples: tuple[str, ...]


class TableS1Shape(BaseModel):
    """What Table S1 holds, read off its pinned bytes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rows: int
    columns: tuple[str, ...]
    distinct_loci: int
    both_locus_columns_equal: bool
    cells: int
    genes_meeting_stated_selection: int


class TableS2Shape(BaseModel):
    """What Table S2 holds: pairwise test statistics, not per-sample abundances."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rows: int
    columns: tuple[str, ...]
    proteins: int
    carbon_sources: int


class ServedSample(BaseModel):
    """One matched sample as the served store holds it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample: str
    served_records: int
    table_s1_genes_served: int
    max_abs_served_minus_compendium: float
    max_abs_served_minus_table_s1: float


class LysineVersusCompendium(BaseModel):
    """Thompson 2019 lysine measured against the compendium and the served store."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    compendium_citation_key: str
    compendium_samples_total: int
    compendium_loci: int
    table_s1: TableS1Shape
    table_s2: TableS2Shape
    matches: tuple[ColumnMatch, ...]
    served: tuple[ServedSample, ...]
    served_store_records: int
    table_s1_cells_served: int
    unmatched_same_condition_samples: tuple[str, ...]


class LysineRecord(shared.SubsumptionRecord):  # type: ignore[name-defined, misc]
    """This row's record: the Table S1 identification and what the store serves."""

    versus_superset: LysineVersusCompendium


def table_s1() -> pd.DataFrame:
    """Table S1, columns renamed: the header repeats ``gene`` twice."""
    frame = pd.read_csv(pinned(TABLE_S1))
    frame.columns = ["locus", "locus_2", "desc", *TABLE_S1_CONDITIONS]
    return frame


def table_s1_shape(frame: pd.DataFrame) -> TableS1Shape:
    """Count Table S1 and re-apply the selection rule the Results state."""
    raw = pd.read_csv(pinned(TABLE_S1))
    lowest = frame[["D-Lysine", "L-Lysine", "5-AVA"]].min(axis=1)
    selected = (lowest < -2) & (frame["Glucose"] >= -0.5)
    return TableS1Shape(
        rows=len(frame),
        columns=tuple(str(c) for c in raw.columns),
        distinct_loci=int(frame["locus"].nunique()),
        both_locus_columns_equal=bool((frame["locus"] == frame["locus_2"]).all()),
        cells=len(frame) * len(TABLE_S1_CONDITIONS),
        genes_meeting_stated_selection=int(selected.sum()),
    )


def table_s2_shape() -> TableS2Shape:
    """Count Table S2's rows, proteins and carbon sources."""
    frame = pd.read_csv(pinned(TABLE_S2), index_col=0)
    return TableS2Shape(
        rows=len(frame),
        columns=tuple(str(c) for c in frame.columns),
        proteins=int(frame["Prot"].nunique()),
        carbon_sources=int(frame["Carbon1"].nunique()),
    )


def compendium_matrix() -> pd.DataFrame:
    """The compendium's gene by sample fitness matrix, indexed by locus."""
    frame = pd.read_excel(
        shared.verified(shared.COMPENDIUM_RELEASE), sheet_name="fitness_measurements"
    )
    return frame.set_index("locusId")


def match_columns(
    s1: pd.DataFrame, matrix: pd.DataFrame, metadata: pd.DataFrame
) -> tuple[list[ColumnMatch], pd.DataFrame]:
    """Identify each Table S1 column with the compendium sample whose values match.

    Every sample column is scored by the maximum absolute difference over Table S1's
    loci; the best and the runner-up are both kept. Returns the matches and a per-cell
    table (locus, column, Table S1 value, compendium value).
    """
    loci = s1["locus"].tolist()
    missing = [locus for locus in loci if locus not in matrix.index]
    if missing:
        raise RuntimeError(f"Table S1 loci absent from the compendium: {missing}")
    sample_columns = [c for c in matrix.columns if str(c).startswith("set")]
    by_column = {f"{r.expName} {r.expDesc}": r for r in metadata.itertuples()}
    matches: list[ColumnMatch] = []
    cells: list[dict[str, Any]] = []
    for column, condition in TABLE_S1_CONDITIONS.items():
        reported = s1.set_index("locus")[column].astype(float)
        scores = sorted(
            (float(np.max(np.abs(matrix.loc[loci, c].astype(float) - reported))), c)
            for c in sample_columns
        )
        (best, name), (runner, runner_name) = scores[0], scores[1]
        meta = by_column[name]
        served = matrix.loc[loci, name].astype(float)
        diff = served - reported
        same = metadata[
            (metadata["expGroup"] == "carbon source")
            & (metadata["condition_1"].astype(str).str.lower() == condition.lower())
            & (metadata["condition_2"].isna())
        ]
        matches.append(
            ColumnMatch(
                table_s1_column=column,
                compendium_sample=str(meta.expName),
                compendium_desc=str(meta.expDesc),
                compendium_set=str(meta.set),
                compendium_vessel=str(meta.vessel),
                compendium_media=str(meta.media),
                compendium_concentration_mM=float(meta.concentration_1),
                compendium_total_rep=int(meta.total_rep),
                genes_compared=len(loci),
                max_abs_difference=round(best, 6),
                median_abs_difference=round(float(diff.abs().median()), 6),
                cells_within_rounding=int(
                    (diff.abs() <= ROUNDING_HALF_ULP + 1e-12).sum()
                ),
                pearson_r=round(float(np.corrcoef(served, reported)[0, 1]), 6),
                runner_up_sample=str(runner_name).split(" ")[0],
                runner_up_max_abs_difference=round(runner, 6),
                same_condition_samples=tuple(sorted(str(n) for n in same["expName"])),
            )
        )
        for locus in loci:
            cells.append(
                {
                    "locus": locus,
                    "table_s1_column": column,
                    "compendium_sample": str(meta.expName),
                    "table_s1": float(reported[locus]),
                    "compendium": float(served[locus]),
                    "difference": float(served[locus] - reported[locus]),
                }
            )
    return matches, pd.DataFrame(cells)


def served_values(samples: set[str]) -> tuple[int, dict[str, dict[str, float]]]:
    """Read the served Borchert 2024 store once: its size and the matched samples.

    1.37 million records, so this is the slow step. The handle is closed before
    returning, because a held handle makes the next reader fail.
    """
    from torchcell.datasets.pputida.borchert2024 import RbTnseqBorchert2024Dataset

    dataset = RbTnseqBorchert2024Dataset(
        root=osp.join(os.environ["DATA_ROOT"], shared.COMPENDIUM_SLUG)
    )
    values: dict[str, dict[str, float]] = {name: {} for name in samples}
    for index in range(len(dataset)):
        experiment = dataset[index]["experiment"]
        sample = experiment["phenotype"]["screen_id"]
        if sample not in values:
            continue
        locus = experiment["genotype"]["perturbations"][0]["systematic_gene_name"]
        values[sample][locus] = float(experiment["phenotype"]["environment_response"])
    total = len(dataset)
    dataset.close_lmdb()
    return total, values


def served_samples(
    matches: list[ColumnMatch],
    s1: pd.DataFrame,
    matrix: pd.DataFrame,
    values: dict[str, dict[str, float]],
) -> list[ServedSample]:
    """Compare each matched sample's served records with the compendium and Table S1."""
    out: list[ServedSample] = []
    indexed = s1.set_index("locus")
    for match in matches:
        held = values[match.compendium_sample]
        column = f"{match.compendium_sample} {match.compendium_desc}"
        vs_compendium = max(
            abs(value - float(matrix.loc[locus, column]))
            for locus, value in held.items()
        )
        s1_loci = [locus for locus in indexed.index if locus in held]
        vs_s1 = max(
            abs(held[locus] - float(indexed.loc[locus, match.table_s1_column]))
            for locus in s1_loci
        )
        out.append(
            ServedSample(
                sample=match.compendium_sample,
                served_records=len(held),
                table_s1_genes_served=len(s1_loci),
                max_abs_served_minus_compendium=round(vs_compendium, 9),
                max_abs_served_minus_table_s1=round(vs_s1, 6),
            )
        )
    return out


def build_record() -> tuple[LysineRecord, pd.DataFrame]:
    """Measure the row and assemble its provenance record."""
    s1 = table_s1()
    metadata = shared.compendium_metadata()
    matrix = compendium_matrix()
    matches, cells = match_columns(s1, matrix, metadata)
    total, values = served_values({m.compendium_sample for m in matches})
    served = served_samples(matches, s1, matrix, values)
    matched = {m.compendium_sample for m in matches}
    unmatched = tuple(
        sorted(
            name
            for m in matches
            for name in m.same_condition_samples
            if name not in matched
        )
    )
    s1_served = sum(s.table_s1_genes_served for s in served)
    evidence = LysineVersusCompendium(
        compendium_citation_key=shared.COMPENDIUM_KEY,
        compendium_samples_total=len(metadata),
        compendium_loci=len(matrix),
        table_s1=table_s1_shape(s1),
        table_s2=table_s2_shape(),
        matches=tuple(matches),
        served=tuple(served),
        served_store_records=total,
        table_s1_cells_served=s1_served,
        unmatched_same_condition_samples=unmatched,
    )
    coverage = shared.ServedCoverage(
        served_dataset="RbTnseqBorchert2024Dataset",
        served_store=f"$DATA_ROOT/{shared.COMPENDIUM_SLUG}",
        served_records=total,
        records_from_this_paper=s1_served,
        released_instances=evidence.table_s1.cells,
        released_instances_basis="the cells of Table S1, the paper's only per-gene "
        "file (39 genes x 4 carbon sources). This is an independent release count: "
        "each cell is found in the served store under the sample its values identify. "
        "The four matched samples are genome-wide, so the store holds far more of this "
        "paper's experiment than Table S1 does; that count is in versus_superset.served",
        attribution_field="EnvironmentResponsePhenotype.screen_id",
        attribution_value="the compendium sample name the Table S1 column matched",
    )
    gap = max(m.max_abs_difference for m in matches)
    margin = min(m.runner_up_max_abs_difference for m in matches)
    served_records = sum(s.served_records for s in served)
    conclusion = (
        f"SUBSUMED. Each of Table S1's {len(matches)} columns is identified, by its "
        f"{evidence.table_s1.rows} values alone, with exactly one of the compendium's "
        f"{evidence.compendium_samples_total} samples: "
        + ", ".join(f"{m.table_s1_column} = {m.compendium_sample}" for m in matches)
        + f". The largest difference over all {evidence.table_s1.cells} cells is "
        f"{gap}, while the closest runner-up sample of any column differs by "
        f"{margin}. The served RbTnseqBorchert2024Dataset holds all four samples as "
        f"{served_records} records, so a loader for this row would store the same "
        "experiment a second time. No loader."
    )
    loadable = (
        "Nothing new. Table S1's cells are a 39-gene subset of four samples the store "
        "serves genome-wide; the store's values are the compendium's (three decimals) "
        "and differ from Table S1 by no more than max_abs_difference, which exceeds "
        "the 0.0005 rounding bound, so the compendium values were not copied from "
        "Table S1 unchanged. Table S2 is pairwise statistics of wild-type proteomics "
        "(Prot, Carbon1, Carbon2, Stat, pval), with no per-sample abundance and no "
        "perturbation, so it is not a record of any existing class. The deletion-"
        "strain growth curves (Figs S3, S5-S7) are released only as TIF images."
    )
    record = LysineRecord(
        citation_key=KEY,
        doi=DOI,
        title=TITLE,
        row_name=ROW_NAME,
        decision="subsumed_no_loader",
        served_coverage=coverage,
        release_files=(
            *(item.release_file() for item in PMC_FILES),
            shared.COMPENDIUM_RELEASE,
        ),
        release_probes=(
            shared.probe_release(
                shared.FITNESS_BROWSER,
                "the paper's genome-wide release; a non-200 is the measured form of "
                "the schedule row's Cloudflare-blocked claim",
            ),
        ),
        sourced_values=SOURCED,
        versus_superset=evidence,
        conclusion=conclusion,
        loadable_slice=loadable,
        measured_at=datetime.now(UTC).isoformat(),
        script=SCRIPT,
    )
    return record, cells


def audit_quotes() -> dict[str, str]:
    """Re-open the pinned article text and confirm every quote is still verbatim."""
    root = shared.data_root() / shared.RAW_REL
    audited: dict[str, str] = {}
    for name, value in SOURCED.items():
        result = audit_sourced_value(value, root)
        if not result.passed:
            raise RuntimeError(f"{name}: {result.message}")
        audited[name] = result.message
    return audited


def main(argv: list[str] | None = None) -> None:
    """Deposit (optionally), measure, write the results, and optionally the record."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--deposit", action="store_true")
    parser.add_argument("--write-record", action="store_true")
    args = parser.parse_args(argv)
    load_dotenv()
    results: dict[str, Any] = {"script": SCRIPT}
    if args.deposit:
        results["manifest"] = str(deposit())
    results["quote_audit"] = audit_quotes()
    record, cells = build_record()
    results["record"] = record.model_dump(mode="json")
    if args.write_record:
        results["written"] = str(shared.write_record(record))
    os.makedirs(RESULTS, exist_ok=True)
    cells.to_csv(
        osp.join(RESULTS, "thompsonMassivelyParallelFitness2019_table_s1_cells.csv"),
        index=False,
    )
    path = osp.join(
        RESULTS, "thompsonMassivelyParallelFitness2019_release_inventory.json"
    )
    with open(path, "w") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps({k: v for k, v in results.items() if k != "record"}, indent=2))
    print(record.conclusion)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
