# tests/torchcell/candidates/test_findings.py
# [[tests.torchcell.candidates.test_findings]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/candidates/test_findings.py
"""The recorded G1 findings: kinds, counts read off the module pins, and the quotes."""

import pytest

from torchcell.candidates import findings
from torchcell.datasets.ecoli import (
    cai2023,
    goodall2018,
    lamoureux2023,
    li2024,
    price2018,
    wetmore2015,
    zhang2015,
)
from torchcell.datasets.pputida import borchert2024, lim2022


@pytest.mark.parametrize(
    ("row", "key", "kind"),
    [
        ("D2Cell 2026", li2024.CITATION_KEY, "transcription"),
        ("MCF2Chem 2023", cai2023.CITATION_KEY, "transcription"),
        ("CeCaFDB flux compendium", zhang2015.CITATION_KEY, "aggregation"),
        ("Lim 2022 putidaPRECISE321", lim2022.CITATION_KEY, "aggregation"),
        ("Borchert 2024 fModules", borchert2024.CITATION_KEY, "aggregation"),
        ("Price 2018", price2018.CITATION_KEY, "aggregation"),
        ("PRECISE-1K", lamoureux2023.CITATION_KEY, "aggregation"),
        ("Goodall 2018", goodall2018.CITATION_KEY, "primary"),
    ],
)
def test_each_finding_kind_and_key(row: str, key: str, kind: str) -> None:
    """Eight rows are recorded, each under its settled module's citation key."""
    finding = findings.finding_for(row)
    assert finding is not None
    assert (finding.citation_key, finding.source_kind) == (key, kind)


def test_unrecorded_row_has_no_finding() -> None:
    """Oyetunde 2019 is an aggregation row with no finding yet (backfill).

    Eight findings: the five of the gate's landing plus Price 2018 and PRECISE-1K,
    added with plan decision 16 (2026-10-10), and Goodall 2018's primary reading (#909).
    """
    assert findings.finding_for("Oyetunde 2019") is None
    assert len(findings.FINDINGS) == len(findings.FINDINGS_BY_ROW) == 5 + 2 + 1


def test_aggregation_records() -> None:
    """Counts come off the module pins: 33 CeCaFDB workbooks, 21 Lim projects, 5 studies."""
    records = {
        f.row_name: f.aggregation
        for f in findings.FINDINGS
        if f.aggregation is not None
    }
    cecafdb = records["CeCaFDB flux compendium"]
    lim = records["Lim 2022 putidaPRECISE321"]
    borchert = records["Borchert 2024 fModules"]
    assert (
        cecafdb.n_source_studies,
        cecafdb.value_origin,
        cecafdb.n_sources_mirrored,
    ) == (33, "derived_by_aggregator", None)
    assert (
        len(findings.CECAFDB_SOURCE_DOIS)
        == len(set(findings.CECAFDB_SOURCE_DOIS))
        == 33
    )
    assert (lim.n_source_studies, lim.value_origin) == (21, "re_measured")
    # 5 + 1 (#908: Thompson 2019 lysine, raw-mirrored, attributed its four samples)
    assert (borchert.n_source_studies, borchert.n_sources_mirrored) == (6, 6)


def test_evidence_is_the_modules_own_sourced_values() -> None:
    """The quotes are the settled modules' constants, not copies."""
    d2cell = findings.finding_for("D2Cell 2026")
    mcf = findings.finding_for("MCF2Chem 2023")
    assert d2cell is not None and mcf is not None
    assert d2cell.evidence == (li2024.RE_MODEL, li2024.CORPUS, li2024.TEXT_ONLY)
    assert mcf.evidence[0] is cai2023.EXTRACTION_FROM_REVIEWS
    assert findings.BORCHERT2024_FB_SHARE.quote == borchert2024.Q_FB_SHARE
    assert (
        findings.BORCHERT2024_FB_SHARE.provenance.sha256 == borchert2024.PAPER_MD_SHA256
    )


def test_decision_16_findings_pin_their_counts_and_quotes() -> None:
    """Price 2018: two sample sources; PRECISE-1K: 45 projects; quotes from the modules."""
    price = findings.finding_for("Price 2018")
    precise = findings.finding_for("PRECISE-1K")
    assert price is not None and price.aggregation is not None
    assert precise is not None and precise.aggregation is not None
    assert (price.aggregation.n_source_studies, price.aggregation.value_origin) == (
        2,
        "re_measured",
    )
    assert price.evidence == (
        wetmore2015.SUPERSET_INCLUDES_THIS_PAPER,
        price2018.NEW_EXPERIMENTS,
    )
    assert precise.aggregation.n_source_studies == 45
    assert precise.aggregation.sources is None
    for value in (
        findings.PRECISE1K_PROJECTS,
        findings.PRECISE1K_ONE_LAB,
        findings.PRECISE1K_ACCESSIONS,
    ):
        assert value.provenance.sha256 == lamoureux2023.PAPER_MD_SHA256
        assert value.provenance.citation_key == lamoureux2023.CITATION_KEY
