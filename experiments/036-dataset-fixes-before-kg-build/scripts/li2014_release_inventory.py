# experiments/036-dataset-fixes-before-kg-build/scripts/li2014_release_inventory.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.li2014_release_inventory]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/li2014_release_inventory

r"""What Li 2014 released, where its bytes now live, and what the schema can hold.

Li GW, Burkhardt D, Gross C, Weissman JS (2014) "Quantifying absolute protein synthesis
rates reveals principles underlying allocation of cellular resources", Cell 157:624-635,
doi:10.1016/j.cell.2014.02.033 (PMC4006352, an NIH author manuscript; PII
S0092867414002323). Row 61 of the bacterial candidate table.

The paper is not in the literature mirror, so its release is deposited in the RAW mirror
(``$DATA_ROOT/torchcell-raw/liQuantifyingAbsoluteProtein2014/``) on the Mohiuddin 2022 /
Rapp 2026 pattern, and never in Zotero:

- the five supplementary workbooks and the published Extended Experimental Procedures
  PDF, served directly by the Elsevier CDN (``retrieve.elsevier_mmc``, recorded as
  ``RetrievalMethod.direct_url`` as Rapp 2026 does);
- PMC's own plain text and JATS XML of the author manuscript, from the PMC Article
  Datasets bucket (``retrieve.pmc_cloud_object``); the bucket holds no PDF and no
  supplement for this article (its listing has exactly the .json, .txt and .xml);
- a ``pdftotext -layout`` rendering of the SI PDF, so a quote from it is auditable as
  text, with a ``ProcessingRecord`` pinning the PDF's sha256 as its input.

Measurements, all off the deposited bytes:

1. **Table S1** (mmc1): genes, per-condition cells, and how many are plain integers vs
   ``[n]`` bracketed integers. The bracket is not defined anywhere in the paper or SI
   text; its meaning is BACK-SOLVED from a count: the main text says 3,041 genes were
   evaluated in rich defined medium, all with >128 footprints, and exactly 3,041 of the
   MOPS-complete cells are plain integers.
2. **Identifier route**: Table S1's gene symbols through ``reconcile_locus_tags`` on the
   pinned MG1655 assembly (the strain the SI names), with the resolved fraction.
3. **Schema fit**: the one released statistic is a synthesis rate in molecules per
   generation. ``ProteinTurnoverPhenotype`` requires a non-empty ``degradation_rate``
   and admits ``synthesis_rate`` only on keys that carry one; this script constructs
   the phenotype both ways to record the validator's verdict rather than asserting it.
4. **Duplication vs Gupta 2024**: the served dev store's b-number keys against Li's
   resolved b-numbers, and Spearman/Pearson r between Li's synthesis rate and Gupta's
   degradation rate / half-life in Gupta's one batch wild-type condition (MOPS glucose
   minimal, 42 min doubling).
5. ``--network``: the GEO series GSE53767 sample list (title, strain, medium, files), the
   replicate structure behind the paper's "error of less than 1.3-fold across
   biological replicates".

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/li2014_release_inventory.py --deposit
    python experiments/036-dataset-fixes-before-kg-build/scripts/li2014_release_inventory.py --network
"""

import argparse
import hashlib
import json
import math
import os
import os.path as osp
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import openpyxl
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, ValidationError
from scipy.stats import pearsonr, spearmanr

load_dotenv()

from torchcell.datamodels.schema import ProteinTurnoverPhenotype  # noqa: E402
from torchcell.datasets.bacteria_common import (  # noqa: E402
    bacterial_genome,
    reconcile_locus_tags,
)
from torchcell.literature.manifest import (  # noqa: E402
    ROLE_PAPER_TEXT,
    ROLE_RAW_DATA,
    ROLE_SI_DATA,
    ROLE_SI_PDF,
    ROLE_SI_TEXT,
    ArtifactRecord,
    Manifest,
    ProcessingRecord,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.literature.provenance import run_retriever  # noqa: E402
from torchcell.literature.retrieve import elsevier_mmc_url, pmc_cloud_url  # noqa: E402

CITATION_KEY = "liQuantifyingAbsoluteProtein2014"
PAPER_DOI = "10.1016/j.cell.2014.02.033"
PAPER_TITLE = (
    "Quantifying absolute protein synthesis rates reveals principles underlying "
    "allocation of cellular resources"
)
PII = "S0092867414002323"
PMC_PREFIX = "PMC4006352.1"
RETRIEVED_AT = "2026-10-10"
GEO_SERIES = "GSE53767"
GEO_SAMPLES = ("GSM1300279", "GSM1300280", "GSM1300281", "GSM1300282")

TABLE_S1 = "1-s2.0-S0092867414002323-mmc1.xlsx"
SI_PDF = "1-s2.0-S0092867414002323-mmc6.pdf"
SI_TEXT = "1-s2.0-S0092867414002323-mmc6.pdftotext.txt"
PAPER_TEXT = "PMC4006352.1.txt"

RESULTS = osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))),
    "results",
    "li2014_release_inventory.json",
)

#: The Gupta 2024 condition compared against: the one wild-type BATCH record (no
#: dilution rate), 40 mM MOPS glucose minimal medium at a 42 min doubling time.
GUPTA_ROOT_REL = "data/torchcell/protein_turnover_gupta2024"


class RawFile(BaseModel):
    """One deposited file: pinned bytes, role, and the retrieval that reproduces it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    relpath: str
    role: str
    sha256: str
    bytes: int
    source: str  # "elsevier" | "pmc_cloud"
    description: str

    @property
    def source_url(self) -> str:
        """The URL the retriever GETs."""
        if self.source == "elsevier":
            return elsevier_mmc_url(PII, self.name.removeprefix(f"1-s2.0-{PII}-"))
        return pmc_cloud_url(f"{PMC_PREFIX}/{self.name}")

    @property
    def retrieval(self) -> RetrievalRecord:
        """The re-runnable retrieval of these bytes."""
        if self.source == "elsevier":
            return RetrievalRecord(
                method=RetrievalMethod.direct_url,
                source_url=self.source_url,
                retriever="torchcell.literature.retrieve.elsevier_mmc",
                params={
                    "pii": PII,
                    "filename": self.name.removeprefix(f"1-s2.0-{PII}-"),
                },
                sha256=self.sha256,
                retrieved_at=RETRIEVED_AT,
            )
        return RetrievalRecord(
            method=RetrievalMethod.pmc_cloud,
            source_url=self.source_url,
            retriever="torchcell.literature.retrieve.pmc_cloud_object",
            params={"key": f"{PMC_PREFIX}/{self.name}"},
            sha256=self.sha256,
            retrieved_at=RETRIEVED_AT,
        )


RAW_FILES: tuple[RawFile, ...] = (
    RawFile(
        name=TABLE_S1,
        relpath=f"data/{TABLE_S1}",
        role=ROLE_RAW_DATA,
        sha256="f493023e9b4bf7a154fd12b4c1fd617047af46153fffde7f1215932d35d5a661",
        bytes=141947,
        source="elsevier",
        description="Table S1: 4,095 gene rows x 3 media (MOPS complete, MOPS minimal, "
        "MOPS complete without methionine), absolute synthesis rate in molecules per "
        "generation; plain integers and [n] bracketed integers",
    ),
    RawFile(
        name="1-s2.0-S0092867414002323-mmc2.xlsx",
        relpath="si/1-s2.0-S0092867414002323-mmc2.xlsx",
        role=ROLE_SI_DATA,
        sha256="ffd05c218c3f7b9d60f96b454e10f072b369a0a3538ba91115694b455beef17f",
        bytes=12026,
        source="elsevier",
        description="Table S2: 62 literature copy numbers (gene, reported copy number, "
        "strain, medium, temperature, growth phase, PubMed ID) used for validation",
    ),
    RawFile(
        name="1-s2.0-S0092867414002323-mmc3.xlsx",
        relpath="si/1-s2.0-S0092867414002323-mmc3.xlsx",
        role=ROLE_SI_DATA,
        sha256="28a68899b6eac61657f5b2eb386d6aee3a7f7ae374f2bbfe9e279e946190cd22",
        bytes=12573,
        source="elsevier",
        description="Table S3: 64 complexes with subunit stoichiometry (curated)",
    ),
    RawFile(
        name="1-s2.0-S0092867414002323-mmc4.xlsx",
        relpath="si/1-s2.0-S0092867414002323-mmc4.xlsx",
        role=ROLE_SI_DATA,
        sha256="877ba750e657a8d81b3e2d54a7dd05788866f2b37951afe7e676a665176b8f21",
        bytes=111211,
        source="elsevier",
        description="sheet 'TableS4': 4,095 genes, mRNA level (RPKM) and translation "
        "efficiency (AU), rich defined medium only",
    ),
    RawFile(
        name="1-s2.0-S0092867414002323-mmc5.xlsx",
        relpath="si/1-s2.0-S0092867414002323-mmc5.xlsx",
        role=ROLE_SI_DATA,
        sha256="47b501c945ff53cb24b5887fedca823780a5e18341fb49496063447937c04f28",
        bytes=11809,
        source="elsevier",
        description="sheet 'TableS5': 102 transcription factors, ligand dependence and "
        "autoregulation (curated)",
    ),
    RawFile(
        name=SI_PDF,
        relpath=f"si/{SI_PDF}",
        role=ROLE_SI_PDF,
        sha256="a96e712d91dea29c6f1072fe2319d6e38ce3103ac7728477cf3b962aa1784828",
        bytes=2780857,
        source="elsevier",
        description="the published article with Extended Experimental Procedures and "
        "supplemental figures (Cell 157, 624-635, S1-S...)",
    ),
    RawFile(
        name=PAPER_TEXT,
        relpath=f"paper/{PAPER_TEXT}",
        role=ROLE_PAPER_TEXT,
        sha256="b3cc01fb5111bef6ae40d758d5e89e69cd198c77bcf33419da7578d27c605a72",
        bytes=62723,
        source="pmc_cloud",
        description="PMC plain text of the NIH author manuscript (NIHMS570024)",
    ),
    RawFile(
        name="PMC4006352.1.xml",
        relpath="paper/PMC4006352.1.xml",
        role=ROLE_PAPER_TEXT,
        sha256="03ac4643bbf8bb28b269645144f8903f26d57d08dce9cf0da69c8be9c28c09a9",
        bytes=126393,
        source="pmc_cloud",
        description="PMC JATS XML of the NIH author manuscript",
    ),
)

#: Quotes that ground the inventory, each checked to occur verbatim in the named file.
QUOTES: dict[str, tuple[str, str]] = {
    "strain": (f"si/{SI_TEXT}", "E. coli K-12 strain MG1655 was used for this study."),
    "media": (
        f"si/{SI_TEXT}",
        "All cultures were based on MOPS media with 0.2% glucose (Teknova), with either",
    ),
    "doubling_times": (
        f"si/{SI_TEXT}",
        # pdftotext renders the degree sign as the control byte \x01; kept verbatim.
        "The doubling time at 37\x01 C is 21.5 ± 0.4 min in fully supplemented MOPS media, "
        "26.5 ± 1.1 min in the methionine",
    ),
    "unit": (f"si/{SI_TEXT}", "where ki has the unit of molecules per generation."),
    "stable_equals_copy_number": (
        f"si/{SI_TEXT}",
        "proteins, ki is also the copy number. The results are listed in Table S1.",
    ),
    "evaluated_3041": (
        f"paper/{PAPER_TEXT}",
        "For growth in a rich defined medium (Neidhardt et al., 1974), we evaluated "
        "3,041 genes which account for >96% of total proteins synthesized.",
    ),
    "threshold_and_error": (
        f"paper/{PAPER_TEXT}",
        "All of these genes have >128 ribosome footprint fragments sequenced, with an "
        "error of less than 1.3-fold across biological replicates.",
    ),
    "upper_bound_for_degraded": (
        f"paper/{PAPER_TEXT}",
        "Our measures based on synthesis rates thus provide an upper bound for the "
        "protein levels for the small subset of proteins that are rapidly degraded.",
    ),
    "geo": (
        f"paper/{PAPER_TEXT}",
        "Data are available at Gene Expression Omnibus with accession number GSE53767.",
    ),
}


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_mirror_dir() -> Path:
    """``$DATA_ROOT/torchcell-raw/liQuantifyingAbsoluteProtein2014``."""
    return Path(os.environ["DATA_ROOT"]) / "torchcell-raw" / CITATION_KEY


def _pdftotext_version() -> str:
    """The poppler ``pdftotext`` version the SI text was rendered with."""
    out = subprocess.run(
        ["pdftotext", "-v"], capture_output=True, text=True, check=True
    )
    return (out.stderr or out.stdout).splitlines()[0].strip()


def render_si_text(pdf: Path, dest: Path) -> None:
    """``pdftotext -layout`` of the SI PDF, the auditable text a quote is read from."""
    subprocess.run(["pdftotext", "-layout", str(pdf), str(dest)], check=True)


def deposit() -> Path:
    """Run every recorded retriever, verify the bytes, and write mirror + manifest.

    Idempotent by sha256: a mirror file already at its pinned hash is left alone; any
    other hash raises instead of being overwritten.
    """
    root = raw_mirror_dir()
    records: list[ArtifactRecord] = []
    with tempfile.TemporaryDirectory() as tmp:
        for raw in RAW_FILES:
            dest = root / raw.relpath
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                if _sha256(dest) != raw.sha256:
                    raise RuntimeError(f"{dest} exists with another sha256; refusing")
            else:
                data = run_retriever(raw.retrieval)
                got = hashlib.sha256(data).hexdigest()
                if got != raw.sha256:
                    raise RuntimeError(
                        f"{raw.source_url}: sha256 {got}, expected {raw.sha256}"
                    )
                staged = Path(tmp) / raw.name
                staged.write_bytes(data)
                shutil.move(str(staged), dest)
            records.append(
                ArtifactRecord(
                    path=raw.relpath,
                    role=raw.role,
                    bytes=dest.stat().st_size,
                    sha256=raw.sha256,
                    source=raw.source_url,
                    original_filename=raw.name,
                    retrieval=raw.retrieval,
                )
            )
    si_pdf = root / f"si/{SI_PDF}"
    si_text = root / f"si/{SI_TEXT}"
    render_si_text(si_pdf, si_text)
    records.append(
        ArtifactRecord(
            path=f"si/{SI_TEXT}",
            role=ROLE_SI_TEXT,
            bytes=si_text.stat().st_size,
            sha256=_sha256(si_text),
            source=f"derived from si/{SI_PDF}",
            original_filename=SI_TEXT,
            processing=ProcessingRecord(
                processor="experiments/036-dataset-fixes-before-kg-build/scripts/"
                "li2014_release_inventory.py:render_si_text",
                tool="pdftotext",
                version=_pdftotext_version(),
                params={"flags": ["-layout"]},
                input_sha256=[_sha256(si_pdf)],
            ),
        )
    )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title=PAPER_TITLE,
        files=records,
        si_data_sources=[raw.source_url for raw in RAW_FILES],
        si_expected=["Table S1", "Table S2", "Table S3", "Table S4", "Table S5"],
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return root


def verify_mirror(root: Path) -> dict[str, str]:
    """Every manifest file re-hashed against its record; raise on any mismatch."""
    manifest = Manifest.model_validate_json((root / "manifest.json").read_text())
    out: dict[str, str] = {}
    for record in manifest.files:
        got = _sha256(root / record.path)
        if got != record.sha256:
            raise RuntimeError(f"{record.path}: sha256 {got} != {record.sha256}")
        out[record.path] = got
    return out


def check_quotes(root: Path, hashes: dict[str, str]) -> dict[str, dict[str, str]]:
    """Each grounding quote, located verbatim in its file (whitespace-normalized)."""
    out: dict[str, dict[str, str]] = {}
    for name, (relpath, quote) in QUOTES.items():
        text = (root / relpath).read_text(encoding="utf-8")
        flat = re.sub(r"\s+", " ", text)
        if quote not in flat:
            raise AssertionError(f"quote {name!r} not found in {relpath}")
        out[name] = {"file": relpath, "sha256": hashes[relpath], "quote": quote}
    return out


_BRACKET = re.compile(r"^\[(\d+)\]$")


def read_table_s1(path: Path) -> tuple[list[str], list[tuple[Any, ...]]]:
    """Header and body rows of Table S1, as openpyxl reads them."""
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = list(book["TableS1"].iter_rows(values_only=True))
    book.close()
    return [str(h) for h in rows[0]], rows[1:]


def table_s1_inventory(
    header: list[str], body: list[tuple[Any, ...]]
) -> dict[str, Any]:
    """Per-condition plain vs bracketed cells, and the value ranges of each."""
    conditions: dict[str, Any] = {}
    for col, name in enumerate(header[1:], start=1):
        kinds: Counter[str] = Counter()
        plain: list[int] = []
        bracketed: list[int] = []
        for row in body:
            value = row[col]
            if isinstance(value, int) and not isinstance(value, bool):
                kinds["plain_integer"] += 1
                plain.append(value)
            elif isinstance(value, str) and _BRACKET.match(value):
                kinds["bracketed_integer"] += 1
                bracketed.append(int(value[1:-1]))
            else:
                kinds[f"other:{value!r}"] += 1
        conditions[name] = {
            "cells": dict(kinds),
            "plain_min": min(plain),
            "plain_max": max(plain),
            "plain_sum": sum(plain),
            "bracketed_min": min(bracketed),
            "bracketed_max": max(bracketed),
            "plain_values_below_bracket_max": sum(v <= max(bracketed) for v in plain),
        }
    names = [str(row[0]) for row in body]
    merged = [n for n in names if "+" in n]
    return {
        "columns": header,
        "gene_rows": len(body),
        "unique_gene_names": len(set(names)),
        "merged_pair_rows": merged,
        "split_rows": [n for n in names if n.startswith("dnaX")],
        "conditions": conditions,
    }


def resolve_names(body: list[tuple[Any, ...]]) -> tuple[dict[str, str], dict[str, Any]]:
    """Gene symbol -> MG1655 locus tag via ``reconcile_locus_tags``, with the report."""
    genome = bacterial_genome("ecoli", "MG1655")
    single = [str(row[0]) for row in body if "+" not in str(row[0])]
    stored, report = reconcile_locus_tags(genome, pd.Series(single), label="li2014")
    locus = dict(zip(single, (str(t) for t in stored), strict=True))
    tag = re.compile(r"^b\d{4}$")
    resolved = {name: t for name, t in locus.items() if tag.match(t)}
    return resolved, {
        "names_offered": len(single),
        "merged_pair_rows_not_offered": len(body) - len(single),
        "resolved_to_b_number": len(resolved),
        "resolved_fraction": round(len(resolved) / len(single), 4),
        "unresolved": sorted(set(single) - set(resolved)),
        "distinct_b_numbers": len(set(resolved.values())),
        "report": str(report),
    }


def loadable_keys(
    header: list[str], body: list[tuple[Any, ...]], resolved: dict[str, str]
) -> dict[str, dict[str, int]]:
    """Per condition: the keys a synthesis-rate record would carry, and every drop.

    A key is a plain-integer cell (evaluated, >128 footprints) on a single gene name
    that resolves to one MG1655 b-number. Bracketed cells, merged-pair rows (one value
    for two loci) and unresolved names are counted as the drops they would be.
    """
    out: dict[str, dict[str, int]] = {}
    for col, condition in enumerate(header[1:], start=1):
        counts: Counter[str] = Counter()
        for row in body:
            name, value = str(row[0]), row[col]
            plain = isinstance(value, int) and not isinstance(value, bool)
            if not plain:
                counts["drop_bracketed_below_128_footprints"] += 1
            elif "+" in name:
                counts["drop_merged_pair_one_value_two_loci"] += 1
            elif name not in resolved:
                counts["drop_name_not_resolved_to_one_b_number"] += 1
            else:
                counts["loadable_keys"] += 1
        out[condition] = dict(counts)
    return out


def schema_fit(sample_key: str, rate: float) -> dict[str, str]:
    """What ``ProteinTurnoverPhenotype`` says to a synthesis-rate-only record."""
    verdicts: dict[str, str] = {}
    try:
        ProteinTurnoverPhenotype(
            degradation_rate={},
            synthesis_rate={sample_key: rate},
            n_replicates={},
            measurement_type="ribosome_profiling_synthesis_molecules_per_generation",
        )
        verdicts["empty_degradation_rate"] = "accepted"
    except ValidationError as error:
        verdicts["empty_degradation_rate"] = error.errors()[0]["msg"]
    return verdicts


def gupta_overlap(
    header: list[str], body: list[tuple[Any, ...]], resolved: dict[str, str]
) -> dict[str, Any]:
    """Shared b-numbers and r against Gupta 2024's batch wild-type record."""
    from torchcell.datasets.ecoli.gupta2024 import ProteinTurnoverGupta2024Dataset

    root = osp.join(os.environ["DATA_ROOT"], GUPTA_ROOT_REL)
    dataset = ProteinTurnoverGupta2024Dataset(root=root)
    gupta_keys: set[str] = set()
    batch: dict[str, Any] | None = None
    for index in range(len(dataset)):
        raw = dataset[index]["experiment"]
        dump = raw if isinstance(raw, dict) else raw.model_dump()
        gupta_keys |= set(dump["phenotype"]["degradation_rate"])
        if (
            not dump["genotype"]["perturbations"]
            and dump["environment"]["dilution_rate_per_hour"] is None
        ):
            if batch is not None:
                raise RuntimeError("more than one batch wild-type Gupta record")
            batch = dump["phenotype"]
    dataset.close_lmdb()
    if batch is None:
        raise RuntimeError("no batch wild-type Gupta record")
    li_keys = set(resolved.values())
    out: dict[str, Any] = {
        "gupta_records": len(dataset),
        "gupta_union_keys": len(gupta_keys),
        "li_resolved_keys": len(li_keys),
        "shared_union_keys": len(li_keys & gupta_keys),
        "gupta_batch_keys": len(batch["degradation_rate"]),
        "per_condition": {},
    }
    by_tag = {tag: name for name, tag in resolved.items()}
    name_col = {str(row[0]): row for row in body}
    for col, condition in enumerate(header[1:], start=1):
        rows = []
        for tag, name in by_tag.items():
            if tag not in batch["degradation_rate"]:
                continue
            value = name_col[name][col]
            if not (isinstance(value, int) and not isinstance(value, bool)):
                continue
            rows.append(
                (value, batch["degradation_rate"][tag], batch["half_life"][tag])
            )
        synth = [math.log10(r[0]) for r in rows]
        deg = [r[1] for r in rows]
        out["per_condition"][condition] = {
            "n_shared_plain": len(rows),
            "spearman_synthesis_vs_degradation_rate": round(
                float(spearmanr([r[0] for r in rows], deg).statistic), 4
            ),
            "pearson_log10_synthesis_vs_degradation_rate": round(
                float(pearsonr(synth, deg).statistic), 4
            ),
        }
    return out


def geo_samples() -> list[dict[str, Any]]:
    """Title, characteristics and files of every GSE53767 sample (network)."""
    out = []
    for gsm in GEO_SAMPLES:
        url = (
            "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
            f"?acc={gsm}&targ=self&form=text&view=brief"
        )
        text = httpx.get(url, timeout=60.0).raise_for_status().text
        fields: dict[str, list[str]] = {}
        for line in text.splitlines():
            if " = " not in line:
                continue
            key, value = line.split(" = ", 1)
            if key in (
                "!Sample_title",
                "!Sample_characteristics_ch1",
                "!Sample_description",
            ):
                fields.setdefault(key.removeprefix("!Sample_"), []).append(value)
        out.append({"gsm": gsm, **fields})
    return out


def main() -> None:
    """Deposit (optional), then measure everything off the mirror bytes."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--deposit", action="store_true")
    parser.add_argument("--network", action="store_true")
    args = parser.parse_args()
    root = deposit() if args.deposit else raw_mirror_dir()
    hashes = verify_mirror(root)
    header, body = read_table_s1(root / f"data/{TABLE_S1}")
    resolved, resolution = resolve_names(body)
    first = next(iter(resolved.values()))
    result: dict[str, Any] = {
        "citation_key": CITATION_KEY,
        "doi": PAPER_DOI,
        "mirror": str(root),
        "mirror_sha256": hashes,
        "quotes": check_quotes(root, hashes),
        "table_s1": table_s1_inventory(header, body),
        "identifier_resolution_mg1655": resolution,
        "loadable_keys_if_schema_held_a_synthesis_rate": loadable_keys(
            header, body, resolved
        ),
        "schema_fit_protein_turnover": schema_fit(first, 62.0),
        "gupta2024_overlap": gupta_overlap(header, body, resolved),
    }
    if args.network:
        result["geo_samples"] = geo_samples()
    elif osp.exists(RESULTS):
        with open(RESULTS) as handle:
            previous = json.load(handle)
        if "geo_samples" in previous:
            result["geo_samples"] = previous["geo_samples"]
    os.makedirs(osp.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps({k: v for k, v in result.items() if k != "quotes"}, indent=2)[:6000]
    )


if __name__ == "__main__":
    main()
