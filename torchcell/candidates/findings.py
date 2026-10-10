# torchcell/candidates/findings.py
# [[torchcell.candidates.findings]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/candidates/findings.py
# Test file: tests/torchcell/candidates/test_findings.py
"""Recorded G1 judgments: the source kind of each settled aggregation row, with its quotes.

The deterministic G1 (:func:`torchcell.candidates.gates.evaluate_g1`) leaves a row that is
marked as an aggregation ``unmeasured``, because aggregation versus transcription is a
reading judgment. The rows below were already read: each finding is composed from the
``SourcedValue`` constants of the row's settled-row module (file + sha256 + verbatim
quote), so the evidence a verdict carries is the evidence the module's tests audit.

Owner decision 2026-10-10 (plan decision 15): an aggregation passes G1 when it carries an
:class:`~torchcell.candidates.verdict.AggregationRecord`; a transcription (LLM or review
tables) fails. Counts here are read off the modules' committed pins or the papers' quoted
statements; a count nobody measured is ``None``, never a guess.

Not every aggregation row has a finding yet: Borchert 2024, Lim 2022 and CeCaFDB are the
re-measured and copied precedents, D2Cell and MCF2Chem the two transcriptions. The rest
(Oyetunde 2019, and the yeast side's served SynthLethDB) are the backfill of piece 3 of
[[plan.dataset-admission-pipeline.2026.10.10]].
"""

from __future__ import annotations

from typing import Final

from torchcell.candidates.gates import SourceKindFinding
from torchcell.candidates.verdict import AggregationRecord
from torchcell.datasets.ecoli import cai2023, li2024, zhang2015
from torchcell.datasets.pputida import borchert2024, lim2022
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import SourcedValue

_BORCHERT_PROVENANCE: Final = Provenance(
    source_uri="paper.md",
    citation_key=borchert2024.CITATION_KEY,
    sha256=borchert2024.PAPER_MD_SHA256,
)

BORCHERT2024_FB_SHARE: Final = SourcedValue(
    value={"fitness_browser": 254, "samples": 332},
    provenance=_BORCHERT_PROVENANCE,
    quote=borchert2024.Q_FB_SHARE,
    note="Gene fitness was computed by the compendium from barcode counts; the release "
    "is one workbook keyed by sample.",
)

BORCHERT2024_COMPENDIUM: Final = SourcedValue(
    value={"samples": 332, "conditions": 183},
    provenance=_BORCHERT_PROVENANCE,
    quote=borchert2024.Q_COMPENDIUM,
)

#: Distinct source DOIs of the in-scope CeCaFDB workbooks, off the committed pins.
CECAFDB_SOURCE_DOIS: Final = tuple(sorted({w.doi for w in zhang2015.WORKBOOKS}))

FINDINGS: Final[tuple[SourceKindFinding, ...]] = (
    SourceKindFinding(
        row_name="D2Cell 2026",
        citation_key=li2024.CITATION_KEY,
        source_kind="transcription",
        reason="every database value is the output of an LLM relation-extraction step "
        "over abstracts and full texts; no row carries the sentence it was read from",
        evidence=(li2024.RE_MODEL, li2024.CORPUS, li2024.TEXT_ONLY),
        aggregation=None,
    ),
    SourceKindFinding(
        row_name="MCF2Chem 2023",
        citation_key=cai2023.CITATION_KEY,
        source_kind="transcription",
        reason="values were read from review tables, and the per-record citation was "
        "recovered from the review's reference column",
        evidence=(cai2023.EXTRACTION_FROM_REVIEWS, cai2023.PER_RECORD_REFERENCE_RULE),
        aggregation=None,
    ),
    SourceKindFinding(
        row_name="CeCaFDB flux compendium",
        citation_key=zhang2015.CITATION_KEY,
        source_kind="aggregation",
        reason="one released workbook per source reference, each attributed to its "
        "paper; the flux values are curator re-mapped onto KEGG reactions and "
        "renormalized to uptake = 100, so they are derived, not re-measured",
        evidence=(zhang2015.LUMPED_SPLIT, zhang2015.RELATIVE_FLUX),
        aggregation=AggregationRecord(
            n_source_studies=len(CECAFDB_SOURCE_DOIS),
            attribution_field="one workbook per source reference (WorkbookPin.doi, "
            "resolved through the Download page's PubMed link)",
            value_origin="derived_by_aggregator",
            n_sources_mirrored=None,
            net_new_vs_served=None,
            evidence=(zhang2015.REFERENCE_COUNT, zhang2015.ECOLI_TABLE_ROW),
        ),
    ),
    SourceKindFinding(
        row_name="Lim 2022 putidaPRECISE321",
        citation_key=lim2022.CITATION_KEY,
        source_kind="aggregation",
        reason="321 RNA-seq profiles re-quantified as log2(TPM + 1) from read counts "
        "across 21 projects; every record names its source study",
        evidence=(lim2022.COMPENDIUM_STRUCTURE, lim2022.COLLECTION),
        aggregation=AggregationRecord(
            n_source_studies=lim2022.COMPENDIUM_STRUCTURE.value["projects"],
            attribution_field="Publication (the sample sheet's DOI, else PMID, else "
            "Lim 2022); preprocess/sample_ledger.json per sample",
            value_origin="re_measured",
            n_sources_mirrored=None,
            net_new_vs_served=None,
            evidence=(lim2022.COMPENDIUM_STRUCTURE,),
        ),
    ),
    SourceKindFinding(
        row_name="Borchert 2024 fModules",
        citation_key=borchert2024.CITATION_KEY,
        source_kind="aggregation",
        reason="332 RB-TnSeq samples whose gene fitness the compendium computed from "
        "barcode counts; a sample is attributed to a source study only when that "
        "study's mirrored Methods name its condition",
        evidence=(BORCHERT2024_COMPENDIUM, BORCHERT2024_FB_SHARE),
        aggregation=AggregationRecord(
            n_source_studies=len(borchert2024.SOURCE_STUDIES),
            attribution_field="Publication via attribute_sample (source_study_quote, "
            "else compendium_release)",
            value_origin="re_measured",
            n_sources_mirrored=len(borchert2024.SOURCE_STUDIES),
            net_new_vs_served=None,
            evidence=(BORCHERT2024_FB_SHARE,),
        ),
    ),
)

FINDINGS_BY_ROW: Final[dict[str, SourceKindFinding]] = {f.row_name: f for f in FINDINGS}


def finding_for(row_name: str) -> SourceKindFinding | None:
    """The recorded finding for a row, or None when G1 has none to read."""
    return FINDINGS_BY_ROW.get(row_name)
