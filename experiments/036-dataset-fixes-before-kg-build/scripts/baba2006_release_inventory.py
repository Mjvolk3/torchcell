# experiments/036-dataset-fixes-before-kg-build/scripts/baba2006_release_inventory.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.baba2006_release_inventory]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/baba2006_release_inventory

r"""What Baba 2006 released, where its bytes now live, and what the schema can hold.

Baba T, Ara T, Hasegawa M, Takai Y, Okumura Y, Baba M, Datsenko KA, Tomita M, Wanner BL,
Mori H (2006) "Construction of Escherichia coli K-12 in-frame, single-gene knockout
mutants: the Keio collection", Mol Syst Biol 2:2006.0008, doi:10.1038/msb4100050
(PMC1681482, open access). Row "Baba 2006" of the bacterial candidate table.

The paper is in the literature mirror (``babaConstructionEscherichiaColi2006``), but
that mirror's ``si/si1.pdf`` is a second copy of the article, not the supplement: the
Supplementary Tables are not in it. They are in the PMC Article Datasets bucket
(``PMC1681482.1/msb4100050-s1.pdf`` to ``-s10.xls``), so the three tables the loader
reads, plus PMC's plain text of the article (the quote anchor), are deposited in the RAW
mirror (``$DATA_ROOT/torchcell-raw/babaConstructionEscherichiaColi2006/``) through
``retrieve.pmc_cloud_object``, on the Li 2014 / Mohiuddin 2022 pattern, and never in
Zotero. Which bucket object is which table is read from the article XML's
``<supplementary-material>`` captions (s4 = Supplementary Table 2, s5 = Table 3, s8 =
Table 6).

Measurements, all off the deposited bytes:

1. **Supplementary Table 2** (s4, the deletion primers): its JW ids are the 4,288
   targeted ORFs.
2. **Supplementary Table 3** (s5): every targeted ORF with its PCR verification
   (``5. verified``, ``6. colonies tested``, ``7. percent``), the OD600 growth of the
   obtained mutants (LB 22 h, MOPS 24 h and 48 h), and four re-served essentiality
   columns (Gerdes 2003 genetic footprinting, PEC, the Wisconsin Tn5 set, a score).
   The paper's own scoring rule ("Mutants were scored as correct if two or more colonies
   had the expected structure") is applied to ``5. verified`` and compared with the
   3,985 / 303 split and with Table 6.
3. **Supplementary Table 6** (s8): the 303 essential gene candidates.
4. **Identifier route**: the JW ids through ``reconcile_locus_tags`` on the pinned
   BW25113 assembly (the strain the paper names), whose GenBank file carries JW ids as
   ``gene_synonym``.
5. **Schema fit**: the call is a construction outcome; ``GeneEssentialityPhenotype`` is
   constructed both ways and ``DeletionStrainPerturbation`` with the derived
   ``jw_synonym`` mapping, so the verdict is the validators', not an assertion.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/baba2006_release_inventory.py --deposit
    python experiments/036-dataset-fixes-before-kg-build/scripts/baba2006_release_inventory.py
"""

import argparse
import hashlib
import json
import os
import os.path as osp
import re
import shutil
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
    DeletionStrainPerturbation,
    DerivedIdentifierMapping,
    GeneEssentialityPhenotype,
)
from torchcell.datasets.bacteria_common import (  # noqa: E402
    LOCUS_TAG_PATTERNS,
    STRAIN_GENE_NAMESPACES,
    bacterial_genome,
    reconcile_locus_tags,
)
from torchcell.literature.manifest import (  # noqa: E402
    ROLE_PAPER_TEXT,
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.literature.provenance import run_retriever  # noqa: E402
from torchcell.literature.retrieve import pmc_cloud_url  # noqa: E402

CITATION_KEY = "babaConstructionEscherichiaColi2006"
PAPER_DOI = "10.1038/msb4100050"
PAPER_TITLE = (
    "Construction of Escherichia coli K-12 in-frame, single-gene knockout mutants: "
    "the Keio collection"
)
PMC_PREFIX = "PMC1681482.1"
RETRIEVED_AT = "2026-10-10"

TABLE_S2 = "msb4100050-s4.xls"
TABLE_S3 = "msb4100050-s5.xls"
TABLE_S6 = "msb4100050-s8.xls"
PAPER_TEXT = "PMC1681482.1.txt"

RESULTS = osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))),
    "results",
    "baba2006_release_inventory.json",
)


class RawFile(BaseModel):
    """One deposited file: pinned bytes, role, and the retrieval that reproduces it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    relpath: str
    role: str
    sha256: str
    bytes: int
    description: str

    @property
    def source_url(self) -> str:
        """HTTPS URL of the bucket object."""
        return pmc_cloud_url(f"{PMC_PREFIX}/{self.name}")

    @property
    def retrieval(self) -> RetrievalRecord:
        """The re-runnable retrieval of these bytes."""
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
        name=TABLE_S2,
        relpath=f"data/{TABLE_S2}",
        role=ROLE_RAW_DATA,
        sha256="897599a5401367356aa2823ed939524561894a9120ef86bad923200b56770bac",
        bytes=1434624,
        description="Supplementary Table 2: the deletion primer extensions of the "
        "4,288 targeted ORFs (ECK, gene, W3110 JW id and coordinates, MG1655 b-number "
        "and coordinates, both primers)",
    ),
    RawFile(
        name=TABLE_S3,
        relpath=f"data/{TABLE_S3}",
        role=ROLE_RAW_DATA,
        sha256="a1a9c30b6cd703ff835f2809b420afe38b1c139865c9af87566a410669a30d2e",
        bytes=1677312,
        description="Supplementary Table 3: 4,378 ORF rows (the 4,288 targeted plus 90 "
        "'N.A.' rows not targeted), PCR-verified colonies of those tested and the "
        "percent correct, OD600 growth in LB (22 h) and glucose MOPS (24 h, 48 h), and "
        "the genetic / PEC / MG_Tn5 / score essentiality columns",
    ),
    RawFile(
        name=TABLE_S6,
        relpath=f"data/{TABLE_S6}",
        role=ROLE_RAW_DATA,
        sha256="ded72d1f890d47b0544fa52edcb491b187a010fa6b938437fc04418e08e31638",
        bytes=107008,
        description="Supplementary Table 6: the 303 essential gene candidates with the "
        "comparison columns and orthologs",
    ),
    RawFile(
        name=PAPER_TEXT,
        relpath=f"paper/{PAPER_TEXT}",
        role=ROLE_PAPER_TEXT,
        sha256="6e39bc2f12aecceac568f91dfcd068e109433132df618dc68624d317737bf3cd",
        bytes=54484,
        description="PMC plain text of the article (the quote anchor; cleaner than the "
        "literature mirror's OCR in the table footnotes)",
    ),
)

#: Quotes that ground the inventory, each checked to occur verbatim in the named file.
QUOTES: dict[str, str] = {
    "targeted_and_obtained": "Of 4288 genes targeted, deletions were obtained for 3985 "
    "ORFs (Supplementary Table 3).",
    "essential_candidates": "these 3985 genes are (probably) nonessential, while the "
    "303 genes (including 37 genes of unknown function), for which no mutants were "
    "found, are candidates for essential genes",
    "scoring_rule": "Mutants were scored as correct if two or more colonies had the "
    "expected structure based on PCR tests for both junction fragments.",
    "colonies_tested": "From every gene deletion experiment, four or eight KmR colonies "
    "were chosen and checked for ones with the correct structure by PCR",
    "secm": "The exceptional case, secM, has a translational arrest sequence",
    "table_6": "Essential gene candidates are given in Supplementary Table 6.",
    "strain": "The Keio collection is comprised of 3985 deletions in duplicate (7970 "
    "total) of E. coli K-12 strain BW25113",
    "selection": "Our standard protocol usually yielded 10–1000 KmR colonies when cells "
    "were incubated aerobically at 37°C on Luria broth (LB) agar containing 30 μg/ml "
    "kanamycin.",
    "lb": "Cells were routinely grown in LB medium containing 1% Bacto Tryptone (Difco), "
    "0.5% yeast extract (Difco), and 0.5% NaCl",
    "percent_footnote": "bPercent of the four or eight KmR colonies shown to have the "
    "correct structure by PCR as described in text is given.",
}

#: Sheet header cells the loader reads (row 1 of each table).
HEADERS: dict[str, tuple[str, ...]] = {
    TABLE_S3: (
        "1. ECK number",
        "2. gene",
        "3. W3110 information",
        "4. MG1655 information",
        "5. verified",
        "6. colonies tested",
        "7. percent",
        "8. cell growth (OD600)",
        "9. genetic",
    )
}


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_mirror_dir() -> Path:
    """``$DATA_ROOT/torchcell-raw/babaConstructionEscherichiaColi2006``."""
    return Path(os.environ["DATA_ROOT"]) / "torchcell-raw" / CITATION_KEY


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
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title=PAPER_TITLE,
        files=records,
        si_data_sources=[raw.source_url for raw in RAW_FILES],
        si_expected=[
            "Supplementary Tables 1, 4, 5, 7 and 8 (s3.pdf, s6.xls, s7.xls, s9.xls, "
            "s10.xls) and the legend document (s2.doc): in the PMC bucket under the "
            "same prefix, not consumed by the loader"
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
    """Each grounding quote, located verbatim in the PMC text (whitespace-normalized)."""
    relpath = f"paper/{PAPER_TEXT}"
    flat = re.sub(r"\s+", " ", (root / relpath).read_text(encoding="utf-8"))
    out: dict[str, dict[str, str]] = {}
    for name, quote in QUOTES.items():
        if quote not in flat:
            raise AssertionError(f"quote {name!r} not found in {relpath}")
        out[name] = {"file": relpath, "sha256": hashes[relpath], "quote": quote}
    return out


def sheet_rows(path: Path, skip: int) -> tuple[list[Any], list[list[Any]]]:
    """Header row 1 and the data rows after ``skip`` header rows, as xlrd reads them."""
    sheet = xlrd.open_workbook(str(path)).sheet_by_index(0)
    rows = [
        [sheet.cell_value(r, c) for c in range(sheet.ncols)] for r in range(sheet.nrows)
    ]
    return rows[1], rows[skip:]


def table_inventory(root: Path) -> dict[str, Any]:
    """Counts, overlaps and the scoring rule, read off the three deposited tables."""
    _, st2 = sheet_rows(root / f"data/{TABLE_S2}", 3)
    header3, st3 = sheet_rows(root / f"data/{TABLE_S3}", 4)
    _, st6 = sheet_rows(root / f"data/{TABLE_S6}", 4)
    labels = [h for h in header3 if h]
    targeted = [r for r in st3 if r[16] != "N.A."]
    untargeted = [r for r in st3 if r[16] == "N.A."]
    obtained = [r for r in targeted if r[14] >= 2]
    not_obtained = [r for r in targeted if r[14] < 2]
    jw2 = Counter(r[2] for r in st2)
    jw3 = Counter(r[2] for r in st3)
    jw6 = {r[2] for r in st6}
    od_cols = (17, 18, 19)
    return {
        "table_s2_rows": len(st2),
        "table_s2_distinct_jw": len(jw2),
        "table_s3_rows": len(st3),
        "table_s3_distinct_jw": len(jw3),
        "table_s3_header_labels": labels,
        "table_s3_targeted_rows": len(targeted),
        "table_s3_untargeted_na_rows": len(untargeted),
        "table_s6_rows": len(st6),
        "targeted_jw_equal_table_s2": {r[2] for r in targeted} == set(jw2),
        "untargeted_jw_in_table_s2": len({r[2] for r in untargeted} & set(jw2)),
        "verified_histogram": {
            str(k): v for k, v in sorted(Counter(r[14] for r in targeted).items())
        },
        "colonies_tested_histogram": {
            str(k): v for k, v in sorted(Counter(r[15] for r in targeted).items())
        },
        "percent_equals_verified_over_tested": sum(
            1 for r in targeted if abs(r[16] - 100.0 * r[14] / r[15]) < 1e-9
        ),
        "scoring_rule_two_or_more_verified": {
            "obtained": len(obtained),
            "not_obtained": len(not_obtained),
            "not_obtained_equals_table_s6": {r[2] for r in not_obtained} == jw6,
            "not_obtained_with_one_verified": [
                f"{r[1]} {r[2]} ({r[14]:.0f} of {r[15]:.0f})"
                for r in not_obtained
                if r[14] == 1
            ],
        },
        "obtained_with_all_three_od600": sum(
            1 for r in obtained if all(isinstance(r[c], float) for c in od_cols)
        ),
        "not_obtained_od600_cells": {
            " | ".join(key): n
            for key, n in Counter(
                tuple(str(r[c]) for c in od_cols) for r in not_obtained
            ).items()
        },
        "reserved_genetic_column_targeted": dict(
            sorted(Counter(r[20] for r in targeted).items())
        ),
        "eck_values_on_more_than_one_table_s3_row": sum(
            1 for n in Counter(r[0] for r in st3).values() if n > 1
        ),
    }


def identifier_route(root: Path) -> dict[str, Any]:
    """The targeted JW ids through ``reconcile_locus_tags`` on BW25113."""
    _, st3 = sheet_rows(root / f"data/{TABLE_S3}", 4)
    targeted = [r for r in st3 if r[16] != "N.A."]
    genome = bacterial_genome("ecoli", "BW25113")
    stored, report = reconcile_locus_tags(
        genome, pd.Series([r[2] for r in targeted], dtype=object), label="Baba 2006 JW"
    )
    pattern = LOCUS_TAG_PATTERNS[report.gene_namespace]
    on = [pattern.match(str(tag)) is not None for tag in stored]
    off = [r for r, ok in zip(targeted, on, strict=True) if not ok]
    return {
        "assembly_set": report.assembly_set,
        "unique_jw": report.unique_names,
        "status_histogram": {k.value: v for k, v in report.status_histogram.items()},
        "resolved_fraction": report.resolved_fraction,
        "on_namespace": sum(on),
        "retired": len(report.retired_kept),
        "kept_on_collision": len(report.kept_on_collision),
        "ambiguous": len(report.ambiguous_kept),
        "off_namespace_by_call": {
            "obtained": sum(1 for r in off if r[14] >= 2),
            "not_obtained": sum(1 for r in off if r[14] < 2),
        },
        "off_namespace_not_obtained": [
            f"{r[1]} {r[2]} {r[8]}" for r in off if r[14] < 2
        ],
    }


def schema_fit(root: Path) -> dict[str, Any]:
    """Construct the phenotype and the deletion leaf the loader would write (thrA), and
    an essential call's phenotype with its #924 evidence (secM, one of eight verified).
    """
    from torchcell.datasets.ecoli import baba2006 as loader

    rows = {r.jw_id: r for r in loader.read_table_s3(root / "data" / loader.TABLE_S3)}
    thra, secm = rows["JW0001"], next(r for r in rows.values() if r.gene == "secM")
    tag = bacterial_genome("ecoli", "BW25113").resolve_gene_name("JW0001")
    if tag.systematic_name is None:
        raise RuntimeError("JW0001 does not resolve on BW25113")
    leaf = DeletionStrainPerturbation(
        systematic_gene_name=tag.systematic_name,
        perturbed_gene_name="thrA",
        gene_namespace=STRAIN_GENE_NAMESPACES["BW25113"],
        identifier_mapping=DerivedIdentifierMapping(
            source_identifier="JW0001", route="jw_synonym"
        ),
        collection="Keio collection",
        cassette="kanamycin cassette",
    )
    essential = GeneEssentialityPhenotype(
        is_essential=not secm.obtained, evidence=loader.evidence(secm)
    )
    non_essential = GeneEssentialityPhenotype(
        is_essential=not thra.obtained, evidence=loader.evidence(thra)
    )
    return {
        "phenotype_fields": sorted(GeneEssentialityPhenotype.model_fields),
        "essential": essential.model_dump(mode="json"),
        "non_essential": non_essential.model_dump(mode="json"),
        "deletion_leaf": leaf.model_dump(mode="json"),
        "field_for": {
            "the evidence type (a construction outcome, not a growth measurement)": (
                "evidence.evidence_type construction_outcome (evidence_class "
                "mutant_recovery)"
            ),
            "the PCR counts behind the call (verified, colonies tested, percent)": (
                "evidence.colonies_verified / colonies_tested / percent_verified"
            ),
            "'candidate' status of the 303 (the paper's own word)": (
                "evidence.call_definition ('candidates for essential genes')"
            ),
        },
        "no_field_for": [],
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
        "tables": table_inventory(root),
        "identifier_resolution_bw25113": identifier_route(root),
        "schema_fit": schema_fit(root),
    }
    os.makedirs(osp.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    print(
        json.dumps({k: v for k, v in result.items() if k != "quotes"}, indent=2)[:6000]
    )


if __name__ == "__main__":
    main()
