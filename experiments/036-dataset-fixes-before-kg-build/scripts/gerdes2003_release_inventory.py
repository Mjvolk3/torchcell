# experiments/036-dataset-fixes-before-kg-build/scripts/gerdes2003_release_inventory.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.gerdes2003_release_inventory]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/gerdes2003_release_inventory

r"""What Gerdes 2003 released, where its bytes now live, and what the schema can hold.

Gerdes SY, Scholle MD, Campbell JW, et al., Osterman AL (2003) "Experimental
determination and system level analysis of essential genes in Escherichia coli MG1655",
J Bacteriol 185:5673-5684, doi:10.1128/JB.185.19.5673-5684.2003 (PMC193955, PubMed
13129938). Row "Gerdes 2003" of the bacterial candidate table.

The paper is in neither Zotero library (searched 2026-10-10, read-only: no item for
"Gerdes" or the title in the group or the personal library), so the literature mirror
holds nothing for it. Its body is walled for scripts, measured 2026-10-10: the PMC
Article Datasets bucket has no ``PMC193955.`` prefix (the article is free on PMC but
not in the open-access subset), ``journals.asm.org`` answers HTTP 403 with a challenge
page, Europe PMC's ``?pdf=render`` answers 403 and its full-text XML service 500. The
SUPPLEMENT is not walled: the authors' Supplementary Online Material page at
``https://www.genome.wisc.edu/Gerdes2003/`` serves Table S1 (Excel, PDF, TXT), Table S2,
Table S6, the results PDF and the two procedure PDFs over plain HTTPS. So the deposit is
the supplement plus PubMed's own record of the article (title, abstract, ids), into the
RAW mirror (``$DATA_ROOT/torchcell-raw/gerdesExperimentalDeterminationSystem2003/``),
never Zotero. The citation key is the three-word Better BibTeX form every raw-only key
uses (``mohiuddinHighthroughputScreeningPromoter2022``).

Consumed files: Table S1 (``table_s1.xls``), the Table S1 page with its column comments
(``supplementary_table.html``), the genetic footprinting procedure (``genetic.pdf``) and
the essentiality-assertion criteria (``assertion_of_essentiality.pdf``), each PDF with a
``pdftotext -layout`` rendering so its quotes are auditable text, and the PubMed XML.

Measurements, all off the deposited bytes:

1. **Table S1**: rows, header, the call histogram (E / N / N* / X / ?), Blattner ids
   (blank, repeated), and the call counts against the abstract's and the procedure
   overview's.
2. **Is the Wisconsin copy the published Table S1?** Baba 2006's Supplementary Table 3
   re-serves "gene essentiality results as determined by genetic footprinting ... in
   Supplementary Table S1 (Gerdes et al., 2003)" per ORF (Baba's legend); its column is
   compared, by b-number, with this table's call (Baba's raw mirror is read).
3. **Identifier route**: the Blattner ids through ``reconcile_locus_tags`` on the pinned
   MG1655 assembly (the strain the procedure names).
4. **Schema fit**: ``TransposonInsertionPerturbation`` + ``GeneEssentialityPhenotype``
   constructed, so the verdict is the validators'.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/gerdes2003_release_inventory.py --deposit
    python experiments/036-dataset-fixes-before-kg-build/scripts/gerdes2003_release_inventory.py
"""

import argparse
import hashlib
import json
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

import pandas as pd
import xlrd
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

load_dotenv()

from torchcell.datamodels.schema import (  # noqa: E402
    GeneEssentialityPhenotype,
    TransposonInsertionPerturbation,
)
from torchcell.datasets.bacteria_common import (  # noqa: E402
    STRAIN_GENE_NAMESPACES,
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

CITATION_KEY = "gerdesExperimentalDeterminationSystem2003"
PAPER_DOI = "10.1128/JB.185.19.5673-5684.2003"
PAPER_TITLE = (
    "Experimental determination and system level analysis of essential genes in "
    "Escherichia coli MG1655"
)
PUBMED_ID = "13129938"
SOM_BASE = "https://www.genome.wisc.edu/Gerdes2003"
RETRIEVED_AT = "2026-10-10"

TABLE_S1 = "table_s1.xls"
TABLE_PAGE = "supplementary_table.html"
GENETIC_PDF = "genetic.pdf"
ASSERTION_PDF = "assertion_of_essentiality.pdf"
PUBMED_XML = f"pubmed_{PUBMED_ID}.xml"
GENETIC_TEXT = "genetic.pdftotext.txt"
ASSERTION_TEXT = "assertion_of_essentiality.pdftotext.txt"
BABA_KEY = "babaConstructionEscherichiaColi2006"
BABA_TABLE_S3 = "data/msb4100050-s5.xls"

RESULTS = osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))),
    "results",
    "gerdes2003_release_inventory.json",
)


class RawFile(BaseModel):
    """One deposited file: pinned bytes, role, and the URL its retrieval GETs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    relpath: str
    role: str
    sha256: str
    bytes: int
    url: str
    description: str

    @property
    def retrieval(self) -> RetrievalRecord:
        """The re-runnable retrieval of these bytes (a plain GET)."""
        return RetrievalRecord(
            method=RetrievalMethod.direct_url,
            source_url=self.url,
            retriever="torchcell.literature.retrieve.direct_url",
            params={"url": self.url},
            sha256=self.sha256,
            retrieved_at=RETRIEVED_AT,
        )


RAW_FILES: tuple[RawFile, ...] = (
    RawFile(
        name=TABLE_S1,
        relpath=f"data/{TABLE_S1}",
        role=ROLE_RAW_DATA,
        sha256="2c4c24570ef1fa40303ed1c1fe24f0c02d7fa5975bd4560d48b29c9e3790b56e",
        bytes=1158144,
        url=f"{SOM_BASE}/{TABLE_S1}",
        description="Table S1 (Excel format): 4,291 ORF rows, each with its ERGO start "
        "and id, gene, the E/N/X/? assertion, CDS length, transposon insertion "
        "positions (amino-acid coordinates), assertion error, Swiss-Prot id and role, "
        "Blattner id, functional category and ERI",
    ),
    RawFile(
        name=TABLE_PAGE,
        relpath=f"si/{TABLE_PAGE}",
        role=ROLE_SI_DATA,
        sha256="f29982272aab87de1c402e2fd1b2a0c5d6cd54cbfbcd259816a2245681100510",
        bytes=8898,
        url=f"{SOM_BASE}/{TABLE_PAGE}",
        description="the Table S1 page of the Supplementary Online Material: the "
        "'Comments to Table S1' defining E, N, X, ? and the assertion error",
    ),
    RawFile(
        name=GENETIC_PDF,
        relpath=f"si/{GENETIC_PDF}",
        role=ROLE_SI_PDF,
        sha256="2e8cb6f0a7cb62220b3ab8b762228b6c0fc2df81104885605d64c98bf2f16d8e",
        bytes=1119035,
        url=f"{SOM_BASE}/{GENETIC_PDF}",
        description="Experimental and Analytical Procedures 1: the genetic "
        "footprinting procedure (strain, transposon, fermentor medium and conditions)",
    ),
    RawFile(
        name=ASSERTION_PDF,
        relpath=f"si/{ASSERTION_PDF}",
        role=ROLE_SI_PDF,
        sha256="cf73ff78b52a021226ea6a8cccbd57f744fc4a16ae2cbe2f034e7b626da9976e",
        bytes=22627,
        url=f"{SOM_BASE}/{ASSERTION_PDF}",
        description="Experimental and Analytical Procedures 2: the criteria behind each "
        "E / N / X / ? assertion and their sources of error",
    ),
    RawFile(
        name=PUBMED_XML,
        relpath=f"paper/{PUBMED_XML}",
        role=ROLE_PAPER_TEXT,
        sha256="294205fd9da3bab725f0c8f398a0c8ddaecc7e79d569d544f3e23249c84b5d5b",
        bytes=22998,
        url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pubmed&id={PUBMED_ID}&retmode=xml",
        description="PubMed's XML record of the article (title, abstract, journal, "
        "ids); the only scriptable text of the article itself",
    ),
)
DERIVED_TEXTS: dict[str, str] = {
    GENETIC_PDF: GENETIC_TEXT,
    ASSERTION_PDF: ASSERTION_TEXT,
}

#: Quotes that ground the inventory: (file, quote), each found verbatim.
QUOTES: dict[str, tuple[str, str]] = {
    "abstract_counts": (
        f"paper/{PUBMED_XML}",
        "We identified 620 genes as essential and 3,126 genes as dispensable for "
        "growth under these conditions.",
    ),
    "abstract_conditions": (
        f"paper/{PUBMED_XML}",
        "genes required for robust aerobic growth of Escherichia coli in rich media.",
    ),
    "overview_unambiguous": (
        f"si/{GENETIC_TEXT}",
        "Unambiguous (non)essentiality assertions made for 3,746 (87%) E. coli ORFs",
    ),
    "strain": (
        f"si/{GENETIC_TEXT}",
        "Strain. Escherichia coli strain MG1655 (F- λ- ilvG rfb50 rph1) (Jensen, 1993) "
        "was used.",
    ),
    "transposon": (
        f"si/{GENETIC_TEXT}",
        "artificial transposon EZ::TN<KAN-2> (Epicentre Technologies, Madison, WI)",
    ),
    "outgrowth": (
        f"si/{GENETIC_TEXT}",
        "Cells were grown in batch culture for 23 population doublings (12 hrs) to a "
        "cell density of 1.4x109.",
    ),
    "conditions": (
        f"si/{GENETIC_TEXT}",
        "Throughout the fermentation temperature was held at 37o C, dissolved oxygen at "
        "30-50% of saturation, and the pH at 6.95 (via titration with 5% H3PO4).",
    ),
    "medium_base": (
        f"si/{GENETIC_TEXT}",
        "tryptone 10 g/L, yeast extract 5 g/L, NaCl 50 mM, NH4Cl 9.5 mM, MgCl2 0.528 "
        "mM, K2SO4 0.276 mM, FeSO4 0.01 mM, CaCl2 5x10-4 mM, and K2HPO4 1.32 mM;",
    ),
    "table_conditions": (
        f"si/{TABLE_PAGE}",
        "under conditions of aerobic logarithmic cell growth in rich medium.",
    ),
    "accd_called_nonessential": (
        f"si/{TABLE_PAGE}",
        "is called nonessential and <i>pdxJ</i> is called essential regardless of "
        "convincing arguments to the contrary.",
    ),
    "essential_rule": (
        f"si/{ASSERTION_TEXT}",
        "All ORFs longer then 240 base pairs in length (>80 AA) free",
    ),
}


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_root() -> Path:
    """``DATA_ROOT`` from the environment."""
    return Path(os.environ["DATA_ROOT"])


def raw_mirror_dir() -> Path:
    """``$DATA_ROOT/torchcell-raw/gerdesExperimentalDeterminationSystem2003``."""
    return data_root() / "torchcell-raw" / CITATION_KEY


def _pdftotext_version() -> str:
    """The poppler ``pdftotext`` version the SI text was rendered with."""
    out = subprocess.run(
        ["pdftotext", "-v"], capture_output=True, text=True, check=True
    )
    return (out.stderr or out.stdout).splitlines()[0].strip()


def render_text(pdf: Path, dest: Path) -> None:
    """``pdftotext -layout`` of one SI PDF, the auditable text a quote is read from."""
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
                        f"{raw.url}: sha256 {got}, expected {raw.sha256}"
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
                    source=raw.url,
                    original_filename=raw.name,
                    retrieval=raw.retrieval,
                )
            )
    for pdf_name, text_name in DERIVED_TEXTS.items():
        pdf = root / f"si/{pdf_name}"
        text = root / f"si/{text_name}"
        render_text(pdf, text)
        records.append(
            ArtifactRecord(
                path=f"si/{text_name}",
                role=ROLE_SI_TEXT,
                bytes=text.stat().st_size,
                sha256=_sha256(text),
                source=f"derived from si/{pdf_name}",
                original_filename=text_name,
                processing=ProcessingRecord(
                    processor="experiments/036-dataset-fixes-before-kg-build/scripts/"
                    "gerdes2003_release_inventory.py:render_text",
                    tool="pdftotext",
                    version=_pdftotext_version(),
                    params={"flags": ["-layout"]},
                    input_sha256=[_sha256(pdf)],
                ),
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title=PAPER_TITLE,
        files=records,
        si_data_sources=[raw.url for raw in RAW_FILES],
        si_expected=[
            "the article body (PDF or full text): walled for scripts on 2026-10-10 "
            "(no PMC open-access copy, journals.asm.org 403, Europe PMC 403); a manual "
            "deposit is the by-hand issue's recipe",
            "Table S1 PDF and TXT renditions, Table S2, Table S6 and results.pdf on the "
            "Supplementary Online Material page: not consumed by the loader",
        ],
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
        flat = re.sub(r"\s+", " ", (root / relpath).read_text(encoding="utf-8"))
        if quote not in flat:
            raise AssertionError(f"quote {name!r} not found in {relpath}")
        out[name] = {"file": relpath, "sha256": hashes[relpath], "quote": quote}
    return out


def table_s1_rows(root: Path) -> tuple[list[Any], list[list[Any]]]:
    """Table S1's column header (row 3) and its data rows, as xlrd reads them."""
    sheet = xlrd.open_workbook(str(root / f"data/{TABLE_S1}")).sheet_by_index(0)
    rows = [
        [sheet.cell_value(r, c) for c in range(sheet.ncols)] for r in range(sheet.nrows)
    ]
    return rows[2], rows[3:]


def table_inventory(root: Path) -> dict[str, Any]:
    """Counts off the deposited Table S1, and the call counts the texts state."""
    header, rows = table_s1_rows(root)
    calls = Counter(r[3] for r in rows)
    with_b = [r for r in rows if r[9] != ""]
    repeated = {b: n for b, n in Counter(r[9] for r in with_b).items() if n > 1}
    essential = calls["E"]
    non_essential = calls["N"] + calls["N*"]
    return {
        "sheet_name": xlrd.open_workbook(str(root / f"data/{TABLE_S1}"))
        .sheet_by_index(0)
        .name,
        "header": header,
        "rows": len(rows),
        "call_histogram": dict(sorted(calls.items())),
        "n_star_rows": [f"{r[2]} {r[9]}" for r in rows if r[3] == "N*"],
        "rows_without_blattner_id": len(rows) - len(with_b),
        "rows_without_blattner_id_by_call": dict(
            sorted(Counter(r[3] for r in rows if r[9] == "").items())
        ),
        "blattner_ids_on_two_or_more_rows": dict(sorted(repeated.items())),
        "rows_on_repeated_blattner_ids": sum(repeated.values()),
        "essential_rows_without_assertion_error": [
            f"{r[2]} {r[9]}" for r in rows if r[3] == "E" and r[6] == ""
        ],
        "counts_vs_texts": {
            "table_essential": essential,
            "table_non_essential_n_plus_n_star": non_essential,
            "table_x": calls["X"],
            "table_ambiguous": calls["?"],
            "abstract_essential": 620,
            "abstract_dispensable": 3126,
            "overview_unambiguous": 3746,
            "table_unambiguous": essential + non_essential,
        },
    }


def baba_reserved_column(root: Path) -> dict[str, Any]:
    """Baba 2006 Table S3's 'genetic' column against this table's call, by b-number."""
    baba = data_root() / "torchcell-raw" / BABA_KEY / BABA_TABLE_S3
    sheet = xlrd.open_workbook(str(baba)).sheet_by_index(0)
    baba_rows = [
        [sheet.cell_value(r, c) for c in range(sheet.ncols)]
        for r in range(4, sheet.nrows)
    ]
    _, rows = table_s1_rows(root)
    gerdes: dict[str, set[str]] = {}
    for r in rows:
        if r[9] != "":
            gerdes.setdefault(str(r[9]), set()).add(str(r[3]))
    reserved: dict[str, set[str]] = {}
    for r in baba_rows:
        if r[8] != "none":
            reserved.setdefault(str(r[8]), set()).add(str(r[20]))
    shared = sorted(set(gerdes) & set(reserved))
    disagree = [b for b in shared if gerdes[b] != reserved[b]]
    return {
        "baba_table_s3_sha256": _sha256(baba),
        "baba_b_numbers": len(reserved),
        "shared_b_numbers": len(shared),
        "identical_call": len(shared) - len(disagree),
        "differing": {b: [sorted(reserved[b]), sorted(gerdes[b])] for b in disagree},
        "differing_where_baba_says_absent": sum(
            1 for b in disagree if reserved[b] == {"-"}
        ),
        "differing_on_a_repeated_gerdes_b_number": sum(
            1 for b in disagree if len(gerdes[b]) > 1
        ),
    }


def identifier_route(root: Path) -> dict[str, Any]:
    """The Blattner ids through ``reconcile_locus_tags`` on MG1655."""
    _, rows = table_s1_rows(root)
    with_b = [str(r[9]) for r in rows if r[9] != ""]
    genome = bacterial_genome("ecoli", "MG1655")
    stored, report = reconcile_locus_tags(
        genome, pd.Series(with_b, dtype=object), label="Gerdes 2003 Blattner ids"
    )
    changed = sorted(
        {(b, str(t)) for b, t in zip(with_b, stored.tolist(), strict=True) if b != t}
    )
    return {
        "assembly_set": report.assembly_set,
        "unique_blattner_ids": report.unique_names,
        "status_histogram": {k.value: v for k, v in report.status_histogram.items()},
        "resolved_fraction": report.resolved_fraction,
        "retired": list(report.retired_kept),
        "ambiguous": {k: list(v) for k, v in report.ambiguous_kept.items()},
        "kept_on_collision": list(report.kept_on_collision),
        "outside_namespace": list(report.outside_namespace),
        "stored_under_another_tag": [f"{b} -> {t}" for b, t in changed],
    }


def schema_fit() -> dict[str, Any]:
    """Construct the phenotype and the transposon leaf the loader would write."""
    leaf = TransposonInsertionPerturbation(
        systematic_gene_name="b0014",
        perturbed_gene_name="dnaK",
        gene_namespace=STRAIN_GENE_NAMESPACES["MG1655"],
        transposon="EZ::TN<KAN-2>",
    )
    return {
        "essential": GeneEssentialityPhenotype(is_essential=True).model_dump(),
        "transposon_leaf": leaf.model_dump(mode="json"),
        "no_field_for": [
            "the evidence type (no insertion recovered after 23 doublings of a pool)",
            "the assertion error (a per-gene probability of a chance miss)",
            "the insertion positions behind an N call",
            "the X (not covered) and ? (ambiguous) categories",
        ],
    }


def main() -> None:
    """Deposit (optional), then measure everything off the mirror bytes."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--deposit", action="store_true")
    args = parser.parse_args()
    root = deposit() if args.deposit else raw_mirror_dir()
    hashes = verify_mirror(root)
    result: dict[str, Any] = {
        "citation_key": CITATION_KEY,
        "doi": PAPER_DOI,
        "mirror": str(root),
        "mirror_sha256": hashes,
        "quotes": check_quotes(root, hashes),
        "table_s1": table_inventory(root),
        "baba2006_reserved_column": baba_reserved_column(root),
        "identifier_resolution_mg1655": identifier_route(root),
        "schema_fit": schema_fit(),
    }
    os.makedirs(osp.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    print(
        json.dumps({k: v for k, v in result.items() if k != "quotes"}, indent=2)[:7000]
    )


if __name__ == "__main__":
    main()
