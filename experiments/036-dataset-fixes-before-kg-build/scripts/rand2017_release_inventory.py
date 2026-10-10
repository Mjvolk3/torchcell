# experiments/036-dataset-fixes-before-kg-build/scripts/rand2017_release_inventory.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.rand2017_release_inventory]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/rand2017_release_inventory
r"""Rand 2017 (the Putida_ML5 library paper): what it released, and where its data sits.

Row "Rand 2017 Putida_ML5 library" of the bacterial schedule
(``experiments/database/scripts/build_bacteria_candidate_datasets_table.py``, class
"Modality / backbone") is the KT2440 RB-TnSeq library every Putida fitness row is scored
on, plus the library's first fitness assays. Two questions decide it, and this script
answers both from sha256-pinned bytes:

(a) **Are its fitness experiments already in the Borchert 2024 compendium?** Measured by
    CONTENT, not by metadata. Supplementary Table 1 prints, for 60 genes, two contrasts
    the Methods define as ``Fitness(LA) - Fitness(Glucose)`` and the same for 4HV, each
    the average of two replicates. Every compendium sample column is tried as the
    treatment and every glucose sample as the control; the pair the SI's numbers pick out
    is the attribution, and the runner-up is the null it beats.

(b) **Is the library description something the genotype model stores?**
    ``TransposonInsertionPerturbation`` has ``barcode`` and ``insertion_position`` slots,
    so a per-strain release WOULD be stored. The paper's only data pointer is the Fitness
    Browser; its HTTP status is probed and recorded. The paper and SI print no per-strain
    table, so the description is a provenance record the fitness datasets reference.

The paper is NOT in the Zotero-backed literature mirror, so ``--deposit`` writes the PMC
Article Datasets objects (``PMC5705400.1``, the NIH author manuscript) into
``$DATA_ROOT/torchcell-raw/randMetabolicPathwayCatabolizing2017/`` with a
``manifest.json``, plus a ``pdftotext -layout`` rendering of Supplementary Information
(the SI is a PDF; the table is read from the rendering). The Mohiuddin 2022 deposit is
the precedent. Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/rand2017_release_inventory.py --deposit
    python experiments/036-dataset-fixes-before-kg-build/scripts/rand2017_release_inventory.py
    python experiments/036-dataset-fixes-before-kg-build/scripts/rand2017_release_inventory.py --network

``--network`` adds the Fitness Browser probe; without it the probe recorded in the
committed results JSON is carried forward unchanged. Results go to
``experiments/036-dataset-fixes-before-kg-build/results/rand2017_release_inventory.json``
and, as the raw-mirror provenance record, ``subsumption_record.json`` beside the
manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from torchcell.data.experiment_dataset import write_verified
from torchcell.literature.manifest import (
    ROLE_PAPER_PDF,
    ROLE_PAPER_TEXT,
    ROLE_SI_PDF,
    ROLE_SI_TEXT,
    ArtifactRecord,
    Manifest,
    ProcessingRecord,
    RetrievalMethod,
    RetrievalRecord,
    sha256_file,
)
from torchcell.literature.provenance import run_retriever
from torchcell.literature.retrieve import pmc_cloud_url
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import SourcedValue, audit_sourced_value

SCRIPT: Final = "experiments/036-dataset-fixes-before-kg-build/scripts/rand2017_release_inventory.py"
RESULTS: Final = (
    "experiments/036-dataset-fixes-before-kg-build/results/"
    "rand2017_release_inventory.json"
)
CITATION_KEY: Final = "randMetabolicPathwayCatabolizing2017"
DOI: Final = "10.1038/s41564-017-0028-z"
TITLE: Final = "A metabolic pathway for catabolizing levulinic acid in bacteria"
ROW_NAME: Final = "Rand 2017 Putida_ML5 library"
PMC_PREFIX: Final = "PMC5705400.1"
RETRIEVED_AT: Final = "2026-10-10"
RECORD_NAME: Final = "subsumption_record.json"

COMPENDIUM_KEY: Final = "borchertMachineLearningAnalysis2024"
COMPENDIUM_XLSX: Final = "data/fModule_Metadata.xlsx"
COMPENDIUM_SHA256: Final = (
    "4d649385ac06684482396a125f135df22a2a5060da73485b2cd14468f8cc8be1"
)
COMPENDIUM_STORE: Final = "data/torchcell/rbtnseq_borchert2024"
FITNESS_BROWSER: Final = (
    "http://fit.genomics.lbl.gov/cgi-bin/exps.cgi?orgId=Putida&expGroup=carbon%20source"
)


class MirrorFile(BaseModel):
    """One file of the raw-mirror deposit, retrieved or derived."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    relpath: str
    object_name: str
    role: str
    sha256: str
    derived_from: str | None = None

    @property
    def source_url(self) -> str:
        """The PMC Article Datasets URL of the retrieved object."""
        return pmc_cloud_url(f"{PMC_PREFIX}/{self.object_name}")

    def retrieval(self) -> RetrievalRecord:
        """The re-runnable retrieval that produced the bytes."""
        return RetrievalRecord(
            method=RetrievalMethod.pmc_cloud,
            source_url=self.source_url,
            retriever="torchcell.literature.retrieve.pmc_cloud_object",
            params={"key": f"{PMC_PREFIX}/{self.object_name}"},
            sha256=self.sha256,
            retrieved_at=RETRIEVED_AT,
        )


PAPER_TEXT: Final = MirrorFile(
    relpath="paper/PMC5705400.1.txt",
    object_name="PMC5705400.1.txt",
    role=ROLE_PAPER_TEXT,
    sha256="9f49ab51f4c179b7652cff75a581f1d00319656dc82daf50c42db26807d55c26",
)
SI_PDF: Final = MirrorFile(
    relpath="si/NIHMS900347-supplement-1.pdf",
    object_name="NIHMS900347-supplement-1.pdf",
    role=ROLE_SI_PDF,
    sha256="1e41d7e8e8e2a9a48273210fd6092be265528b549988b88fc0c7d52de2ce3dcd",
)
RETRIEVED: Final = (
    PAPER_TEXT,
    MirrorFile(
        relpath="paper/PMC5705400.1.xml",
        object_name="PMC5705400.1.xml",
        role=ROLE_PAPER_TEXT,
        sha256="e43714862c6e7dc95474e38ba27ee5f1ce5fac6fa052dc48241e88ccca54a42c",
    ),
    MirrorFile(
        relpath="paper/PMC5705400.1.pdf",
        object_name="PMC5705400.1.pdf",
        role=ROLE_PAPER_PDF,
        sha256="458007790ed6675c510490190ba39e69d2ddc239df8771c76adb6780b840be4a",
    ),
    SI_PDF,
    MirrorFile(
        relpath="si/NIHMS900347-supplement-2.pdf",
        object_name="NIHMS900347-supplement-2.pdf",
        role=ROLE_SI_PDF,
        sha256="a3d954bc944bd3dd135c58b51a4c35e4a7ee94f786de792533a7a790d5430e46",
    ),
)
SI_TEXT_RELPATH: Final = "si/NIHMS900347-supplement-1.txt"
PDFTOTEXT_ARGS: Final = ("-layout",)


# --------------------------------------------------------------------------- #
# Verbatim quotes (PMC plain text of the author manuscript)
# --------------------------------------------------------------------------- #
_PAPER: Final = Provenance(
    source_uri=PAPER_TEXT.relpath,
    citation_key=CITATION_KEY,
    sha256=PAPER_TEXT.sha256,
    method="PMC Article Datasets plain text of the NIH author manuscript",
    page="Methods",
    retrieved=RETRIEVED_AT,
)


def _q(value: object, quote: str, page: str = "Methods") -> SourcedValue:
    return SourcedValue(
        value=value, provenance=_PAPER.model_copy(update={"page": page}), quote=quote
    )


SOURCED: Final[dict[str, SourcedValue]] = {
    "library_name": _q(
        "Putida_ML5",
        "We named the final, sequenced mapped transposon mutant library Putida_ML5.",
    ),
    "library_size": _q(
        "thousands of colonies (no count)",
        "We combined thousands of kanamycin-resistant P. putida colonies into a single "
        "tube",
    ),
    "vector": _q(
        "pKMW3 mariner, random 20mer barcodes",
        "pKMW3 is a mariner class transposon vector library containing a kanamycin "
        "resistance marker and millions of random 20mer DNA barcodes.",
    ),
    "conditions": _q(
        ("4HV 40 mM", "LA 40 mM", "potassium acetate 20 mM", "glucose 40 mM", 2),
        "The carbon sources tested were 40mM 4HV (pH adjusted to 7 with NaOH), 40mM LA "
        "(pH adjusted to 7 with NaOH), 20mM potassium acetate, and 40mM glucose, each "
        "with two replicates.",
    ),
    "days": _q(
        "two days, each with its own glucose control",
        "The 4HV and acetate experiments were performed one day and the LA experiments "
        "were performed on a different day, each day with its own 40 mM Glucose "
        "control.",
    ),
    "la_vessel": _q(
        "24-well microplate, Multitron, 700 rpm",
        "For LA, the samples were grown in a 24-well microplate in a Multitron shaker "
        "set to 30°C and 700 rpm.",
    ),
    "si_mean_of_two": _q(
        2,
        "The fitness values reported in Supplementary Table 1 are the average of 2 "
        "replicates.",
    ),
    "si_contrast": _q(
        "Fitness(LA) - Fitness(Glucose)",
        "Fitness scores for LA and 4HV relative to glucose were calculated using the "
        "following equation:",
    ),
    "data_availability": _q(
        FITNESS_BROWSER,
        "All data from the P. putida transposon sequencing experiments is available "
        "through the fitness browser at "
        "http://fit.genomics.lbl.gov/cgi-bin/exps.cgi?orgId=Putida&expGroup=carbon%20source.",
        page="Data Availability",
    ),
    "assay_authors": _q(
        "K.M.W., R.L.C, J.R., A.M.D.",
        "K.M.W., R.L.C, J.R. and A.M.D. performed the fitness assays with the "
        "Putida_ML5 library.",
        page="Author Contribution",
    ),
}

#: The Methods' ten samples, as the compendium's ``expName`` that carries each. The
#: glucose, 4HV and LA pairs are ATTRIBUTED BY CONTENT below; acetate has no printed
#: value (the SI dropped acetate-shared phenotypes), so its pair rests on the Methods
#: naming 20 mM acetate on the 4HV day and on set1 being that day by content.
RAND_SAMPLES: Final[dict[str, tuple[str, str]]] = {
    "set1IT078": ("glucose 40 mM, 4HV/acetate day", "content"),
    "set1IT079": ("glucose 40 mM, 4HV/acetate day", "content"),
    "set1IT082": ("4HV 40 mM", "content"),
    "set1IT083": ("4HV 40 mM", "content"),
    "set1IT084": ("potassium acetate 20 mM", "methods_quote"),
    "set1IT085": ("potassium acetate 20 mM", "methods_quote"),
    "set5IT075": ("glucose, LA day", "content"),
    "set5IT076": ("glucose, LA day", "content"),
    "set5IT081": ("LA 40 mM", "content"),
    "set5IT082": ("LA 40 mM", "content"),
}


# --------------------------------------------------------------------------- #
# Typed results
# --------------------------------------------------------------------------- #
class SiTable1(BaseModel):
    """Supplementary Table 1 as read from the pinned SI rendering."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    genes: int
    la_values: int
    hv_values: int
    na_cells: int
    genes_in_compendium: int
    genes_absent_from_compendium: tuple[str, ...]


class ContrastMatch(BaseModel):
    """How well one compendium treatment-minus-control reproduces one SI column."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    si_column: str
    treatment: tuple[str, ...]
    control: tuple[str, ...]
    n_genes: int
    pearson_r: float
    median_abs_diff: float
    max_abs_diff: float
    within_half_unit: int
    runner_up_treatment: str
    runner_up_median_abs_diff: float
    runner_up_control: str
    runner_up_control_median_abs_diff: float


class FitnessBrowserProbe(BaseModel):
    """The measured answer of the paper's only data pointer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str
    status_code: int
    cloudflare_challenge: bool
    probed_at: str


class ServedStore(BaseModel):
    """What the served Borchert 2024 dev store holds of the Rand samples."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    store: str
    kept_samples: int
    kept_records: int
    rand_samples_in_release: int
    rand_samples_dropped: int
    drop_rule: str
    drop_rule_samples: int
    drop_rule_records: int
    rand_records_served: int
    rand_records_if_medium_lands: int


class Inventory(BaseModel):
    """The whole measurement, written as the results JSON and the mirror record."""

    model_config = ConfigDict(extra="forbid")

    citation_key: str
    doi: str
    title: str
    row_name: str
    decision: Literal["subsumed_no_loader"]
    mirror_files: list[dict[str, object]]
    compendium_putida_ml5_samples: dict[str, int]
    si_table_1: SiTable1
    matches: list[ContrastMatch]
    rand_samples: dict[str, tuple[str, str]]
    fitness_browser: FitnessBrowserProbe
    served: ServedStore
    genotype_verdict: str
    other_content: str
    sourced_values: dict[str, SourcedValue]
    quote_audit: dict[str, str]
    measured_at: str
    script: str = SCRIPT


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def data_root() -> Path:
    """``DATA_ROOT`` from the environment."""
    return Path(os.environ["DATA_ROOT"])


def mirror_dir() -> Path:
    """``$DATA_ROOT/torchcell-raw/randMetabolicPathwayCatabolizing2017``."""
    return data_root() / "torchcell-raw" / CITATION_KEY


def pdftotext_version() -> str:
    """The poppler version string ``pdftotext -v`` prints (it prints to stderr)."""
    out = subprocess.run(
        ["pdftotext", "-v"], capture_output=True, text=True, check=True
    )
    return (out.stderr or out.stdout).splitlines()[0].strip()


def render_si(pdf: Path, dest: Path) -> None:
    """Render the SI PDF to text with ``pdftotext -layout``."""
    subprocess.run(["pdftotext", *PDFTOTEXT_ARGS, str(pdf), str(dest)], check=True)


# --------------------------------------------------------------------------- #
# Deposit
# --------------------------------------------------------------------------- #
def deposit() -> Path:
    """Retrieve every PMC object, verify its sha256, and write the raw mirror.

    Idempotent by sha256: an existing file with the pinned hash is left alone and one
    with any other hash raises rather than being overwritten.
    """
    root = mirror_dir()
    records: list[ArtifactRecord] = []
    for item in RETRIEVED:
        dest = root / item.relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if sha256_file(dest) != item.sha256:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            write_verified(
                run_retriever(item.retrieval()), dest, item.sha256, item.source_url
            )
        records.append(
            ArtifactRecord(
                path=item.relpath,
                role=item.role,
                bytes=dest.stat().st_size,
                sha256=item.sha256,
                source=item.source_url,
                original_filename=item.object_name,
                retrieval=item.retrieval(),
            )
        )
    text = root / SI_TEXT_RELPATH
    with tempfile.TemporaryDirectory() as tmp:
        rendered = Path(tmp) / "si.txt"
        render_si(root / SI_PDF.relpath, rendered)
        if text.exists() and sha256_file(text) != sha256_file(rendered):
            raise RuntimeError(f"{text} exists with a different sha256; refusing")
        if not text.exists():
            shutil.copy2(rendered, text)
    records.append(
        ArtifactRecord(
            path=SI_TEXT_RELPATH,
            role=ROLE_SI_TEXT,
            bytes=text.stat().st_size,
            sha256=sha256_file(text),
            source=f"derived from {SI_PDF.relpath}",
            original_filename=Path(SI_TEXT_RELPATH).name,
            processing=ProcessingRecord(
                processor="experiments/036-dataset-fixes-before-kg-build/scripts/"
                "rand2017_release_inventory.render_si",
                tool="pdftotext",
                version=pdftotext_version(),
                params={"args": list(PDFTOTEXT_ARGS), "input_sha256": SI_PDF.sha256},
            ),
        )
    )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=TITLE,
        files=records,
        si_data_sources=[item.source_url for item in RETRIEVED],
        si_expected=[
            "Supplementary Information PDF (supplement-1: Note, Figs, Tables 1-10)",
            "Life Sciences Reporting Summary (supplement-2)",
            "Supplementary Files: plasmid sequences and Jupyter notebook (named in the "
            "Methods; NOT in the PMC object set, not retrieved)",
        ],
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return root


def load_manifest() -> Manifest:
    """The raw mirror's manifest."""
    return Manifest.model_validate_json((mirror_dir() / "manifest.json").read_text())


def verified_path(manifest: Manifest, relpath: str) -> Path:
    """A mirror file's path after checking its bytes against the manifest."""
    (record,) = [r for r in manifest.files if r.path == relpath]
    path = mirror_dir() / relpath
    if sha256_file(path) != record.sha256:
        raise RuntimeError(f"{path} does not match its manifest sha256")
    return path


# --------------------------------------------------------------------------- #
# Supplementary Table 1
# --------------------------------------------------------------------------- #
_ROW = re.compile(
    r"^\s*(PP_\d{4})\s.*?\s(-?\d*\.?\d+|NA)\s+(-?\d*\.?\d+|NA)\s*$", re.MULTILINE
)


def si_table_1(text: str) -> pd.DataFrame:
    """The table's rows: locus, the two printed cells, and each cell's half-unit.

    The table sits between its own caption and Supplementary Table 2's; every row is a
    locus tag followed by name/annotation and two numeric (or ``NA``) cells.
    """
    start = text.index("Supplementary Table 1. Genes of Interest Identified")
    end = text.index("Supplementary Table 2. Target MS/MS")
    rows = []
    for m in _ROW.finditer(text[start:end]):
        rows.append({"locus": m.group(1), "LA": m.group(2), "HV": m.group(3)})
    frame = pd.DataFrame(rows).set_index("locus")
    for col in ("LA", "HV"):
        frame[f"{col}_half"] = frame[col].map(_half_unit)
        frame[col] = pd.to_numeric(frame[col].replace("NA", np.nan))
    return frame


def _half_unit(cell: str) -> float:
    """Half of the last printed decimal place (``-4.5`` -> 0.05, ``0.003`` -> 0.0005)."""
    if cell == "NA":
        return np.nan
    decimals = len(cell.split(".")[1]) if "." in cell else 0
    return 0.5 * 10.0 ** (-decimals)


# --------------------------------------------------------------------------- #
# The compendium
# --------------------------------------------------------------------------- #
def compendium() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The release's metadata sheet and its gene-by-sample fitness sheet."""
    path = data_root() / "torchcell-raw" / COMPENDIUM_KEY / COMPENDIUM_XLSX
    if sha256_file(path) != COMPENDIUM_SHA256:
        raise RuntimeError(f"{path} does not match the pinned compendium sha256")
    meta = pd.read_excel(path, sheet_name="metadata")
    fit = pd.read_excel(path, sheet_name="fitness_measurements").set_index("sysName")
    fit = fit[[c for c in fit.columns if c.startswith("set")]]
    fit.columns = [c.split()[0] for c in fit.columns]
    return meta, fit


def match(
    fit: pd.DataFrame,
    meta: pd.DataFrame,
    table: pd.DataFrame,
    column: str,
    treatment: tuple[str, ...],
    control: tuple[str, ...],
) -> ContrastMatch:
    """Score one treatment-minus-control against an SI column, with both nulls.

    Null 1 swaps the treatment for every other single sample column (control fixed).
    Null 2 swaps the control for every other glucose-only sample (treatment fixed).
    """
    si = table[column].dropna()
    half = table.loc[si.index, f"{column}_half"]
    loci = si.index.intersection(fit.index)
    si, half = si[loci], half[loci]
    ctrl = fit.loc[loci, list(control)].mean(axis=1)
    diff = fit.loc[loci, list(treatment)].mean(axis=1) - ctrl - si
    nulls = {
        s: float((fit.loc[loci, s] - ctrl - si).abs().median())
        for s in fit.columns
        if s not in treatment
    }
    runner = min(nulls, key=nulls.__getitem__)
    glucose = meta[(meta["condition_1"] == "D-Glucose") & meta["condition_2"].isna()]
    treat = fit.loc[loci, list(treatment)].mean(axis=1)
    cnulls = {
        s: float((treat - fit.loc[loci, s] - si).abs().median())
        for s in glucose["expName"]
        if s not in control
    }
    crunner = min(cnulls, key=cnulls.__getitem__)
    return ContrastMatch(
        si_column=column,
        treatment=treatment,
        control=control,
        n_genes=len(loci),
        pearson_r=round(float(np.corrcoef(si, diff + si)[0, 1]), 5),
        median_abs_diff=round(float(diff.abs().median()), 4),
        max_abs_diff=round(float(diff.abs().max()), 4),
        within_half_unit=int((diff.abs() <= half + 1e-9).sum()),
        runner_up_treatment=runner,
        runner_up_median_abs_diff=round(nulls[runner], 4),
        runner_up_control=crunner,
        runner_up_control_median_abs_diff=round(cnulls[crunner], 4),
    )


def served_store() -> ServedStore:
    """The dev store's own drop ledger, read for the Rand samples."""
    ledger = json.loads(
        (
            data_root() / COMPENDIUM_STORE / "preprocess" / "dropped_records.json"
        ).read_text()
    )
    rule = "medium_not_in_media_library"
    dropped = {s.split()[0] for s in ledger["by_rule"][rule]["samples"]}
    rand_dropped = sorted(set(RAND_SAMPLES) & dropped)
    per_sample = ledger["by_rule"][rule]["n_records_not_written"] // len(dropped)
    return ServedStore(
        store=COMPENDIUM_STORE,
        kept_samples=ledger["kept_samples"],
        kept_records=ledger["kept_records"],
        rand_samples_in_release=len(RAND_SAMPLES),
        rand_samples_dropped=len(rand_dropped),
        drop_rule=rule,
        drop_rule_samples=len(dropped),
        drop_rule_records=ledger["by_rule"][rule]["n_records_not_written"],
        rand_records_served=(len(RAND_SAMPLES) - len(rand_dropped)) * per_sample,
        rand_records_if_medium_lands=len(RAND_SAMPLES) * per_sample,
    )


# --------------------------------------------------------------------------- #
# The Fitness Browser
# --------------------------------------------------------------------------- #
def probe_fitness_browser() -> FitnessBrowserProbe:
    """GET the paper's data pointer and record what answers."""
    request = urllib.request.Request(
        FITNESS_BROWSER, headers={"User-Agent": "torchcell-provenance/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            status, body = response.status, response.read(4096)
    except urllib.error.HTTPError as error:
        status, body = error.code, error.read(4096)
    return FitnessBrowserProbe(
        url=FITNESS_BROWSER,
        status_code=status,
        cloudflare_challenge=b"Just a moment" in body,
        probed_at=datetime.now(UTC).isoformat(),
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
GENOTYPE_VERDICT: Final = (
    "PROVENANCE RECORD, not stored genotypes. TransposonInsertionPerturbation has barcode "
    "and insertion_position slots, so a per-strain release would be stored, but this "
    "paper releases none: the Methods name the library and its vector and say only that "
    "'thousands' of colonies were pooled, the SI prints no strain table, and the only "
    "data pointer (Data Availability) is the Fitness Browser, which answers the probe "
    "below. Every fitness release of the library (this paper's included) is gene-level, "
    "so the records stay one gene-level insertion genotype per locus with barcode and "
    "insertion_position None, which is what RbTnseqBorchert2024Dataset already stores."
)
OTHER_CONTENT: Final = (
    "Net-new content outside the Putida_ML5 row, measured and NOT loaded here: main-text "
    "Table 1 maps 10 Tn5 (pBAM1) insertion isolates of a SEPARATE library to a position "
    "on the KT2440 chromosome, each selected for no growth on LA plates; Table 2 scores "
    "6 clean deletions (lvaR, lvaA-E) with empty vector or complementation as -, + or ++ "
    "growth on LA and 4HV; Supplementary Tables 3 and 4 are E. coli LS5218 (not in the "
    "genomes tier). Filed as a separate dataset issue."
)


def audit(values: dict[str, SourcedValue]) -> dict[str, str]:
    """Re-open every quote in the raw mirror; a failed audit raises."""
    root = data_root() / "torchcell-raw"
    out: dict[str, str] = {}
    for name, value in values.items():
        result = audit_sourced_value(value, root)
        if not result.passed:
            raise RuntimeError(f"{name}: {result.message}")
        out[name] = result.message
    return out


def measure(probe: FitnessBrowserProbe) -> Inventory:
    """Every measurement, off the pinned mirror and the dev store."""
    manifest = load_manifest()
    text = verified_path(manifest, SI_TEXT_RELPATH).read_text(encoding="utf-8")
    table = si_table_1(text)
    meta, fit = compendium()
    present = table.index.intersection(fit.index)
    matches = [
        match(
            fit,
            meta,
            table,
            "LA",
            ("set5IT081", "set5IT082"),
            ("set5IT075", "set5IT076"),
        ),
        match(
            fit,
            meta,
            table,
            "HV",
            ("set1IT082", "set1IT083"),
            ("set1IT078", "set1IT079"),
        ),
    ]
    return Inventory(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=TITLE,
        row_name=ROW_NAME,
        decision="subsumed_no_loader",
        mirror_files=[
            {"path": r.path, "role": r.role, "bytes": r.bytes, "sha256": r.sha256}
            for r in manifest.files
        ],
        compendium_putida_ml5_samples={
            str(k): int(v)
            for k, v in meta.loc[meta["mutantLibrary"] == "Putida_ML5", "set"]
            .value_counts()
            .sort_index()
            .items()
        },
        si_table_1=SiTable1(
            genes=len(table),
            la_values=int(table["LA"].notna().sum()),
            hv_values=int(table["HV"].notna().sum()),
            na_cells=int(table[["LA", "HV"]].isna().sum().sum()),
            genes_in_compendium=len(present),
            genes_absent_from_compendium=tuple(sorted(set(table.index) - set(present))),
        ),
        matches=matches,
        rand_samples=RAND_SAMPLES,
        fitness_browser=probe,
        served=served_store(),
        genotype_verdict=GENOTYPE_VERDICT,
        other_content=OTHER_CONTENT,
        sourced_values=SOURCED,
        quote_audit=audit(SOURCED),
        measured_at=datetime.now(UTC).isoformat(),
    )


def main() -> None:
    """Deposit (optionally), measure, and write the results and the mirror record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deposit", action="store_true", help="write the raw mirror")
    parser.add_argument("--network", action="store_true", help="probe the browser")
    args = parser.parse_args()
    load_dotenv()
    if args.deposit:
        print(f"deposited {deposit()}")
    if args.network:
        probe = probe_fitness_browser()
    else:
        prior = json.loads(Path(RESULTS).read_text())
        probe = FitnessBrowserProbe.model_validate(prior["fitness_browser"])
    inventory = measure(probe)
    payload = inventory.model_dump_json(indent=2)
    Path(RESULTS).write_text(payload + "\n")
    (mirror_dir() / RECORD_NAME).write_text(payload + "\n")
    for m in inventory.matches:
        print(m.model_dump_json())
    print(inventory.served.model_dump_json())
    print(inventory.fitness_browser.model_dump_json())


if __name__ == "__main__":
    main()
