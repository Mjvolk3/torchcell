# tests/torchcell/datasets/scerevisiae/test_cachera2023.py
# [[tests.torchcell.datasets.scerevisiae.test_cachera2023]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_cachera2023.py
"""Cachera 2023 betaxanthin CRI-SPA loader: one record per colony (#1177).

The data-gated tests at the end build from the sha256-pinned raw-mirror files
(``--data --slow``). Everything above them is hermetic.

A synthetic ``QD1_2_4_6.csv`` under ``tmp_path`` in the released 14-column layout (a
blank index header, then ``gene``, ``plate``, ``row``, ``column``, ``repeat`` and the
eight ``[corrected_]<area|mean_intensity>.<24|48>`` columns; only ``mean_intensity.24``
and ``corrected_mean_intensity.24`` are consumed, every other cell is 9.0), beside a
synthetic ``GA1_2_4_6.csv`` whose ``WT`` row (0.35, std 0.68, count 12) is the reference.
No genome: every colony is stored under the name its ``gene`` cell releases (#1045).
Twelve colony positions, as (gene, plate, row, column, repeat, raw, corrected):

0. AAC1 (0,0,0,1, 0.4, 0.13) -> record 0 on the released-gene-name leaf
1. AAC1 (0,0,1,1, 0.4, -0.2) -> record 1, a second colony of AAC1
2. ``0`` (0,0,2,1, 0.3, -0.22) -> an empty library well, no record
3. WT (0,0,3,1, 0.4, 0.35) -> record 2, a control colony: the cassette, no deletion
4. BUD16 (0,0,4,1, blank, blank) -> no colony measured, no record
5. ACN9 (0,0,5,1, 0.5, blank) -> withdrawn by the authors, no record
6. YBR001C (0,1,0,1, 0.4, 0.3) -> record 3 on ``KanMxDeletionPerturbation``
7. blank gene (0,1,1,1, 0.4, 0.9) -> ``str(nan)`` is ``"nan"``, no record
8. `` AAD3 `` (0,1,2,1, 0.4, 0.0) -> stripped to AAD3, record 4; 0.0 is a value
9. ACN9 (0,1,3,2, 0.5, 0.5) -> record 5 under ``ACN9`` as released, repeat 2
10. SDH7 (0,1,4,2, 0.6, 0.9) -> record 6, a strain of its own beside ACN9
11. ycr001w (0,1,5,2, 0.4, -0.25) -> record 7, YCR001W on the KanMX leaf

So 8 records (1 control, 2 under a systematic name, 5 under a released symbol) and 4
positions without one (1 no colony, 1 withdrawn, 2 naming no strain), ledgered in
``preprocess/dropped_records.json``. Each record is one colony: its value verbatim,
``n_replicates`` 1, ``metabolite_level_se`` None with a typed
``not_applicable_single_measurement`` gap, ``replicate_id`` the colony position. Every
record shares one reference, the GA ``WT`` summary.

Issue #509 (fixed 2026.10.02): every record and its reference store the paper's final
screen plate, ``CACHERA_YPD_G418``, ``temperature=None`` with a typed
``not_reported_by_primary`` gap, and ``measurement_type``
``cri_spa_corrected_hsv_yellowness_24h``. The dataset gene set carries the non-S288C
names ``CYP76AD1`` and ``DOD`` (the cassette additions).

The sha256 contract (issue #528): ``process`` verifies both files in ``raw/`` against
their pins before reading a row; the tests that build from synthetic files run under the
conftest recorder in place of that byte check.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import os.path as osp
import shutil
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from torchcell.data import RawSha256MismatchError
from torchcell.datamodels.schema import (
    Environment,
    GeneAdditionPerturbation,
    Genotype,
    KanMxDeletionPerturbation,
    Localization,
    MetaboliteExperiment,
    MetaboliteExperimentReference,
    MetabolitePhenotype,
    Publication,
    ReferenceGenome,
    ReleasedIdentifierPerturbation,
    ReleasedKind,
)
from torchcell.datasets.scerevisiae import cachera2023 as c
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

# The released header of QD1_2_4_6.csv, verbatim (14 columns, the first one blank).
_QD_HEADER = (
    ",gene,plate,row,column,repeat,area.24,mean_intensity.24,area.48,"
    "mean_intensity.48,corrected_area.24,corrected_mean_intensity.24,"
    "corrected_area.48,corrected_mean_intensity.48"
).split(",")
# The released header of GA1_2_4_6.csv, verbatim (27 columns, the first one blank).
_GA_HEADER = (
    ",Unnamed: 0,gene,area.24_mean,area.24_std,area.24_count,mean_intensity.24_mean,"
    "mean_intensity.24_std,mean_intensity.24_count,area.48_mean,area.48_std,"
    "area.48_count,mean_intensity.48_mean,mean_intensity.48_std,"
    "mean_intensity.48_count,corrected_area.24_mean,corrected_area.24_std,"
    "corrected_area.24_count,corrected_mean_intensity.24_mean,"
    "corrected_mean_intensity.24_std,corrected_mean_intensity.24_count,"
    "corrected_area.48_mean,corrected_area.48_std,corrected_area.48_count,"
    "corrected_mean_intensity.48_mean,corrected_mean_intensity.48_std,"
    "corrected_mean_intensity.48_count"
).split(",")

_Row = tuple[str | None, int, int, int, int, float | None, float | None]
# (gene, plate, row, column, repeat, raw 24 h, corrected 24 h); None is a blank cell.
_ROWS: list[_Row] = [
    ("AAC1", 0, 0, 0, 1, 0.4, 0.13),
    ("AAC1", 0, 0, 1, 1, 0.4, -0.2),
    ("0", 0, 0, 2, 1, 0.3, -0.22),
    ("WT", 0, 0, 3, 1, 0.4, 0.35),
    ("BUD16", 0, 0, 4, 1, None, None),
    ("ACN9", 0, 0, 5, 1, 0.5, None),
    ("YBR001C", 0, 1, 0, 1, 0.4, 0.3),
    (None, 0, 1, 1, 1, 0.4, 0.9),
    (" AAD3 ", 0, 1, 2, 1, 0.4, 0.0),
    ("ACN9", 0, 1, 3, 2, 0.5, 0.5),
    ("SDH7", 0, 1, 4, 2, 0.6, 0.9),
    ("ycr001w", 0, 1, 5, 2, 0.4, -0.25),
]
# (gene, level, std, count) rows of the synthetic GA file.
_GA_ROWS: list[tuple[str, float | None, float | None, float | None]] = [
    ("AAC1", -0.035, 0.233, 2.0),
    ("WT", 0.35, 0.68, 12.0),
]
_WT_SE = 0.68 / math.sqrt(12)


def _cell(value: float | str | None) -> str:
    return "" if value is None else str(value)


def _write_qd(path: Path, rows: list[_Row] = _ROWS) -> None:
    """The released per-colony layout, written by hand so blank cells stay blank."""
    lines = [",".join(_QD_HEADER)]
    for i, (gene, plate, row, column, repeat, raw, level) in enumerate(rows):
        cells = {name: "9.0" for name in _QD_HEADER[6:]}
        cells[c._COLONY_RAW] = _cell(raw)
        cells[c._COLONY_LEVEL] = _cell(level)
        lines.append(
            ",".join(
                [str(i), _cell(gene), str(plate), str(row), str(column), str(repeat)]
                + [cells[n] for n in _QD_HEADER[6:]]
            )
        )
    path.write_text("\n".join(lines) + "\n")


def _write_ga(
    path: Path,
    rows: list[tuple[str, float | None, float | None, float | None]] = _GA_ROWS,
) -> None:
    """The released per-gene layout; only the three 24 h corrected columns matter."""
    assert _GA_HEADER.index(c._LEVEL) == 18 and _GA_HEADER.index(c._COUNT) == 20
    lines = [",".join(_GA_HEADER)]
    for i, (gene, level, std, count) in enumerate(rows):
        cells: dict[str, str] = {name: "9.0" for name in _GA_HEADER[3:]}
        cells[c._LEVEL] = _cell(level)
        cells[c._STD] = _cell(std)
        cells[c._COUNT] = _cell(count)
        lines.append(
            ",".join([str(i), str(i), gene] + [cells[n] for n in _GA_HEADER[3:]])
        )
    path.write_text("\n".join(lines) + "\n")


def _stage(root: Path, rows: list[_Row] = _ROWS) -> None:
    (root / "raw").mkdir(parents=True)
    _write_qd(root / "raw" / c.QD_FILENAME, rows)
    _write_ga(root / "raw" / c.DATA_FILENAME)


@pytest.fixture
def built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> c.BetaxanthinCachera2023Dataset:
    """The twelve-position file of the module docstring, built end to end."""
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    _stage(tmp_path)
    return c.BetaxanthinCachera2023Dataset(root=str(tmp_path))


def _cassette() -> list[GeneAdditionPerturbation]:
    """The four Btx-cassette additions, each placed on the deposited pBTX1 contig.

    The placement is built from the loader's own pinned constants rather than copied,
    because the point of P4 is that the pointer is one composed object: a contig named
    by its sha256 with an ``ArtifactRef`` pinning the same bytes, the gene's feature key
    on that contig, the XII-5 host site with both of its spans typed absent, and the two
    designed alleles as contig coordinates.
    """
    return [
        GeneAdditionPerturbation(
            systematic_gene_name=systematic,
            gene_namespace="scerevisiae_s288c_sgd_systematic",
            perturbed_gene_name=name,
            source_organism=organism,
            is_heterologous=heterologous,
            localization=Localization.chromosomal_integration,
            integration_locus="XII-5",
            construct_name="Btx-cassette",
            variant=variant,
            placement=c._placement(systematic),
        )
        for systematic, name, organism, heterologous, variant in [
            ("CYP76AD1", "CYP76AD1", "Beta vulgaris", True, None),
            ("DOD", "DOD", "Mirabilis jalapa", True, None),
            ("YBR249C", "ARO4", "Saccharomyces cerevisiae", False, "K229L"),
            ("YPR060C", "ARO7", "Saccharomyces cerevisiae", False, "G141S"),
        ]
    ]


def test_the_btx_cassette_points_at_the_deposited_pbtx1_contig() -> None:
    """P4: each addition names pBTX1 by sha256, with its feature and its allele.

    Measured by
    ``experiments/036-dataset-fixes-before-kg-build/scripts/cachera2023_btx_cassette_placement.py``:
    the paper builds the CRI-SPA donor with pBTX1 ("the integrative plasmid pBTX1
    harboring the betaxanthin-synthesizing genes CYP76AD1 and DOD"), that contig indexes
    cleanly while pBTX2 is one of the 11 maps blocked for an origin-wrapping location,
    and codon 229 of its ARO4 feature reads ``TTA`` (Leu) at contig 3368-3370 while
    codon 141 of its ARO7 feature reads ``AGT`` (Ser) at 820-822.
    """
    by_gene = {p.systematic_gene_name: p for p in c._betaxanthin_cassette()}
    assert set(by_gene) == {"CYP76AD1", "DOD", "YBR249C", "YPR060C"}
    for gene, placement in ((g, p.placement) for g, p in by_gene.items()):
        assert placement is not None, gene
        assert placement.contig_sha256 == c.PBTX1_SHA256
        assert placement.contig_ref.sha256 == c.PBTX1_SHA256
        assert placement.contig_ref.path == c.PBTX1_PATH
        assert placement.feature_key.so_id == "SO:0000704"
        host = placement.host_placement
        assert host is not None and host.site_label == "XII-5"
        assert host.assembly_set == "sgd_S288C_R64-4-1_20230830"
        assert host.span is None and host.insert_span is None
        assert host.gapped_fields() == {"span", "insert_span"}
    aro4_placement = by_gene["YBR249C"].placement
    aro7_placement = by_gene["YPR060C"].placement
    assert aro4_placement is not None and aro7_placement is not None
    assert aro4_placement.feature_key.token == "ARO4-K229L@SO:0000704:2684-3796+"
    aro4 = aro4_placement.variants
    aro7 = aro7_placement.variants
    assert [(v.variant_statement, v.position_start, v.position_end) for v in aro4] == [
        ("K229L", 3368, 3370)
    ]
    assert [(v.variant_statement, v.position_start, v.position_end) for v in aro7] == [
        ("G141S", 820, 822)
    ]
    assert aro4[0].codon_number == 229 and aro7[0].codon_number == 141
    assert [v.reference_contig_sha256 for v in aro4 + aro7] == [c.PBTX1_SHA256] * 2
    # the heterologous genes carry no designed allele, so no call
    for gene in ("CYP76AD1", "DOD"):
        plant = by_gene[gene].placement
        assert plant is not None and plant.variants == []


def _environment() -> Environment:
    return Environment(
        media=c.CACHERA_YPD_G418, temperature=None, provenance_gaps=[c.TEMPERATURE_GAP]
    )


def _deletion(gene: str) -> KanMxDeletionPerturbation | ReleasedIdentifierPerturbation:
    if gene.startswith("Y"):
        return KanMxDeletionPerturbation(
            systematic_gene_name=gene, perturbed_gene_name=gene
        )
    return ReleasedIdentifierPerturbation(
        systematic_gene_name=gene,
        gene_namespace="scerevisiae_s288c_sgd_systematic",
        perturbed_gene_name=gene,
        identifier_kind="gene column entry, as released",
        released_kind=ReleasedKind.deletion,
        cassette="KanMX",
    )


def _experiment(gene: str | None, level: float, rid: str) -> dict[str, Any]:
    deletion = [] if gene is None else [_deletion(gene)]
    return MetaboliteExperiment(
        dataset_name="BetaxanthinCachera2023Dataset",
        genotype=Genotype(perturbations=[*deletion, *_cassette()]),
        environment=_environment(),
        phenotype=MetabolitePhenotype(
            metabolite_level={"betaxanthin": level},
            metabolite_level_se=None,
            n_replicates={"betaxanthin": 1},
            measurement_type="cri_spa_corrected_hsv_yellowness_24h",
            target_metabolite_ids=None,
            replicate_id=rid,
            provenance_gaps=[
                ProvenanceGap(
                    field="metabolite_level_se",
                    reason=ProvenanceGapReason.not_applicable_single_measurement,
                    note=c._SINGLE_COLONY_SE_NOTE,
                )
            ],
        ),
    ).model_dump()


def _reference() -> dict[str, Any]:
    return MetaboliteExperimentReference(
        dataset_name="BetaxanthinCachera2023Dataset",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain="BY4741",
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        ),
        environment_reference=_environment(),
        phenotype_reference=MetabolitePhenotype(
            metabolite_level={"betaxanthin": 0.35},
            metabolite_level_se={"betaxanthin": _WT_SE},
            n_replicates={"betaxanthin": 12},
            measurement_type="cri_spa_corrected_hsv_yellowness_24h",
        ),
    ).model_dump()


_PUBLICATION = Publication(
    pubmed_id="37572348",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/37572348/",
    doi="10.1093/nar/gkad656",
    doi_url="https://doi.org/10.1093/nar/gkad656",
).model_dump()

#: The eight records of the module docstring: (gene or None, level, replicate_id).
_RECORDS: list[tuple[str | None, float, str]] = [
    ("AAC1", 0.13, "repeat1/plate0/row0/column0"),
    ("AAC1", -0.2, "repeat1/plate0/row0/column1"),
    (None, 0.35, "repeat1/plate0/row0/column3"),
    ("YBR001C", 0.3, "repeat1/plate0/row1/column0"),
    ("AAD3", 0.0, "repeat1/plate0/row1/column2"),
    ("ACN9", 0.5, "repeat2/plate0/row1/column3"),
    ("SDH7", 0.9, "repeat2/plate0/row1/column4"),
    ("YCR001W", -0.25, "repeat2/plate0/row1/column5"),
]


def test_replicate_id_is_the_released_colony_position() -> None:
    """``repeat``, ``plate``, ``row`` and ``column``, in the release's column order."""
    assert c.replicate_id((4, 12, 31, 47)) == "repeat4/plate12/row31/column47"
    assert c._POSITION == ("repeat", "plate", "row", "column")


def test_build_keeps_every_measured_colony_in_row_order_as_released(
    built: c.BetaxanthinCachera2023Dataset,
) -> None:
    """Eight records, one per measured colony naming a strain, each exactly as
    hand-built: the value verbatim, n 1, no SE with a typed gap, the position as
    ``replicate_id``. AAC1's two colonies are two records; ACN9 and SDH7 stay apart and
    are never renamed; the WT colony carries the cassette and no deletion.
    """
    assert len(built) == len(_RECORDS)
    for i, (gene, level, rid) in enumerate(_RECORDS):
        assert built[i]["experiment"] == _experiment(gene, level, rid), i
        assert built[i]["reference"] == _reference(), i
        assert built[i]["publication"] == _PUBLICATION, i


def test_the_control_colony_is_the_btx_background_with_no_deletion(
    built: c.BetaxanthinCachera2023Dataset,
) -> None:
    """Record 2 (WT): the four cassette additions only, so its genotype is the
    reference's background and its value is one of the colonies the reference sums.
    """
    perturbations = built[2]["experiment"]["genotype"]["perturbations"]
    assert [p["perturbation_type"] for p in perturbations] == ["gene_addition"] * 4
    assert built[2]["experiment"]["phenotype"]["metabolite_level"] == {
        "betaxanthin": 0.35
    }


def test_a_colony_record_states_one_measurement(
    built: c.BetaxanthinCachera2023Dataset,
) -> None:
    """Every record: n 1, SE None, one ``not_applicable_single_measurement`` gap on
    ``metabolite_level_se``, and a distinct ``replicate_id``.
    """
    phenotypes = [built[i]["experiment"]["phenotype"] for i in range(len(built))]
    assert {json.dumps(p["n_replicates"]) for p in phenotypes} == {'{"betaxanthin": 1}'}
    assert {p["metabolite_level_se"] is None for p in phenotypes} == {True}
    assert {
        tuple((g["field"], g["reason"]) for g in p["provenance_gaps"])
        for p in phenotypes
    } == {(("metabolite_level_se", "not_applicable_single_measurement"),)}
    assert len({p["replicate_id"] for p in phenotypes}) == len(phenotypes)


def test_the_reference_is_the_authors_wt_summary(
    built: c.BetaxanthinCachera2023Dataset,
) -> None:
    """The GA ``WT`` row: level 0.35, SE 0.68 / sqrt(12), n 12, no replicate id."""
    reference = built[0]["reference"]["phenotype_reference"]
    assert reference["metabolite_level"] == {"betaxanthin": 0.35}
    assert reference["metabolite_level_se"] == {"betaxanthin": _WT_SE}
    assert reference["n_replicates"] == {"betaxanthin": 12}
    assert reference["replicate_id"] is None


def test_environment_and_readout_match_the_paper_issue_509(
    built: c.BetaxanthinCachera2023Dataset,
) -> None:
    """Issue #509: every record and its reference store the final CRI-SPA screen plate
    (solid YPD + 2% agar + 200 ug/ml G418, complex), no temperature with a typed
    ``not_reported_by_primary`` gap, and an HSV yellowness ``measurement_type``. The
    loader used to store synthetic SC at 30 C with a fluorescence measurement type.
    """
    environments = {
        json.dumps(built[i]["experiment"]["environment"], sort_keys=True)
        for i in range(len(built))
    }
    assert environments == {json.dumps(_environment().model_dump(), sort_keys=True)}
    record = built[0]
    env = record["experiment"]["environment"]
    media = env["media"]
    assert (media["state"], media["is_synthetic"], media["base_medium"]) == (
        "solid",
        False,
        "YPD",
    )
    assert media["name"].startswith("YPD + G418")
    assert [
        (
            comp["compound"]["name"],
            comp["role"],
            comp["concentration"]["value"],
            comp["concentration"]["unit"],
        )
        for comp in media["components"]
    ] == [
        ("yeast extract", "complex_ingredient", 1.0, "percent_w/v"),
        ("peptone", "complex_ingredient", 2.0, "percent_w/v"),
        ("D-glucose", "carbon_source", 2.0, "percent_w/v"),
        ("agar", "gelling_agent", 2.0, "percent_w/v"),
        ("G418 (geneticin)", "selection_agent", 200.0, "ug/mL"),
    ]
    assert media["components"][-1]["provenance"][0]["quote"] == (
        "For selection on solid medium, plates were supplemented with "
        "$2 0 0 ~ \\mathrm { { m g / l } }$ geneticin (G418)"
    )
    assert env["temperature"] is None
    assert [(g["field"], g["reason"]) for g in env["provenance_gaps"]] == [
        ("temperature", "not_reported_by_primary")
    ]
    assert record["reference"]["environment_reference"] == _environment().model_dump()
    assert {
        built[i]["experiment"]["phenotype"]["measurement_type"]
        for i in range(len(built))
    } | {
        built[i]["reference"]["phenotype_reference"]["measurement_type"]
        for i in range(len(built))
    } == {"cri_spa_corrected_hsv_yellowness_24h"}


def test_every_record_shares_one_reference_and_the_gene_set_has_the_cassette(
    built: c.BetaxanthinCachera2023Dataset,
) -> None:
    """Finding: the gene set is every perturbation's ``systematic_gene_name``, so the
    heterologous ``CYP76AD1`` and ``DOD`` sit beside the S288C ids. Pinned until the gene
    set is restricted to the genome's ids or the cassette carries separate identifiers.
    """
    index = built.experiment_reference_index
    assert index is not None
    (entry,) = index
    assert entry.member_indices == list(range(8))
    assert entry.reference.model_dump() == _reference()
    preprocess = Path(built.preprocess_dir)
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "AAC1",
        "AAD3",
        "ACN9",
        "CYP76AD1",
        "DOD",
        "SDH7",
        "YBR001C",
        "YBR249C",
        "YCR001W",
        "YPR060C",
    ]
    stored = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert [e["member_indices"] for e in stored] == [list(range(8))]


def test_preprocess_csv_holds_the_kept_colonies_exactly(
    built: c.BetaxanthinCachera2023Dataset,
) -> None:
    """``preprocess/data.csv``: gene (blank for a control colony), replicate id, the
    four position fields and the level.
    """
    text = (Path(built.preprocess_dir) / "data.csv").read_text()
    assert text.splitlines() == [
        "gene,replicate_id,repeat,plate,row,column,level",
        "AAC1,repeat1/plate0/row0/column0,1,0,0,0,0.13",
        "AAC1,repeat1/plate0/row0/column1,1,0,0,1,-0.2",
        ",repeat1/plate0/row0/column3,1,0,0,3,0.35",
        "YBR001C,repeat1/plate0/row1/column0,1,0,1,0,0.3",
        "AAD3,repeat1/plate0/row1/column2,1,0,1,2,0.0",
        "ACN9,repeat2/plate0/row1/column3,2,0,1,3,0.5",
        "SDH7,repeat2/plate0/row1/column4,2,0,1,4,0.9",
        "YCR001W,repeat2/plate0/row1/column5,2,0,1,5,-0.25",
    ]


def test_the_ledger_and_the_log_account_for_every_released_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Position 4 has no colony, position 5 was withdrawn by the authors, positions 2
    and 7 name no strain; the ledger lists each by its colony position under its rule,
    and every other position is a record.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    _stage(tmp_path)
    with caplog.at_level(logging.INFO, logger=c.log.name):
        dataset = c.BetaxanthinCachera2023Dataset(root=str(tmp_path))
    assert len(dataset) == 8
    messages = [r.getMessage() for r in caplog.records if r.name == c.log.name]
    assert (
        "Cachera: 8 colony records (1 control, 2 under a systematic name, 5 under a "
        "released symbol); 1 positions without a colony, 1 colonies withdrawn by the "
        "authors, 2 colonies on a position naming no strain"
    ) in messages
    assert "Wrote 8 Cachera betaxanthin experiments to LMDB" in messages
    ledger = c.DropLog.model_validate_json(
        (Path(dataset.preprocess_dir) / "dropped_records.json").read_text()
    )
    assert ledger == c.DropLog(
        dataset="BetaxanthinCachera2023Dataset",
        source_records=12,
        kept_records=8,
        kept_control_records=1,
        kept_under_systematic_name=2,
        kept_under_released_symbol=5,
        dropped_records=4,
        rules=[
            c.DropRule(
                rule="no_colony_measured",
                description=c._RULE_DESCRIPTIONS[c.RULE_NO_COLONY],
                n_records=1,
                items=["repeat1/plate0/row0/column4"],
            ),
            c.DropRule(
                rule="withdrawn_by_the_authors",
                description=c._RULE_DESCRIPTIONS[c.RULE_WITHDRAWN],
                n_records=1,
                items=["repeat1/plate0/row0/column5"],
            ),
            c.DropRule(
                rule="gene_cell_is_0_or_blank",
                description=c._RULE_DESCRIPTIONS[c.RULE_NO_GENE],
                n_records=2,
                items=["repeat1/plate0/row0/column2", "repeat1/plate0/row1/column1"],
            ),
        ],
    )


def test_a_repeated_colony_position_refuses_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two rows at one (repeat, plate, row, column) are two claims on one colony."""
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    _stage(tmp_path, _ROWS + [("AAC1", 0, 0, 0, 1, 0.4, 0.2)])
    with pytest.raises(
        c.RepeatedPositionError,
        match="names colony repeat1/plate0/row0/column0 on two rows",
    ):
        c.BetaxanthinCachera2023Dataset(root=str(tmp_path))
    assert not (tmp_path / "processed" / "lmdb").exists()


def test_two_tokens_meeting_on_one_upper_cased_name_refuse_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ycr001w`` and ``YCR001W`` would merge two released strains; refused by name."""
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    _stage(tmp_path, _ROWS + [("YCR001W", 0, 2, 0, 2, 0.4, 0.1)])
    with pytest.raises(
        c.RepeatedGeneError, match="releases 'ycr001w' and 'YCR001W', both 'YCR001W'"
    ):
        c.BetaxanthinCachera2023Dataset(root=str(tmp_path))


def test_a_per_gene_file_without_a_wt_row_refuses_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reference is the authors' WT summary; without it there is no reference."""
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    _stage(tmp_path)
    _write_ga(tmp_path / "raw" / c.DATA_FILENAME, [("AAC1", 0.1, 0.2, 2.0)])
    with pytest.raises(c.MissingControlSummaryError, match="has 0 WT rows"):
        c.BetaxanthinCachera2023Dataset(root=str(tmp_path))


def test_a_raw_file_off_the_pin_is_refused_at_build_time(
    off_pin_raw: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (issue #528): with both files already in ``raw/`` PyG skips
    ``download``, so ``process`` verifies them against their pins first and raises
    ``RawSha256MismatchError`` naming the first file and both digests; no store is
    written and the file is left as found.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    staged = off_pin_raw(c, [c.DATA_FILENAME, c.QD_FILENAME])
    raw = staged.root / "raw" / c.DATA_FILENAME
    with pytest.raises(RawSha256MismatchError) as excinfo:
        c.BetaxanthinCachera2023Dataset(root=str(staged.root))
    assert str(excinfo.value) == (
        f"sha256 mismatch for {raw}: expected {c.DATA_SHA256}, "
        f"observed {staged.observed}"
    )
    assert list((staged.root / "processed").iterdir()) == []
    assert not (staged.root / "preprocess").exists()
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == staged.observed


class _Response:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._data


def _bare(tmp_path: Path) -> c.BetaxanthinCachera2023Dataset:
    """An uninitialized instance whose ``raw_dir`` is ``tmp_path/raw`` (not created)."""
    dataset = c.BetaxanthinCachera2023Dataset.__new__(c.BetaxanthinCachera2023Dataset)
    dataset.root = str(tmp_path)
    return dataset


def _fake_urlopen(
    monkeypatch: pytest.MonkeyPatch,
    payloads: dict[str, bytes],
    calls: list[tuple[str, str, int]],
) -> None:
    def urlopen(req: Any, timeout: int) -> _Response:
        calls.append((req.full_url, req.get_header("User-agent"), timeout))
        return _Response(payloads[req.full_url])

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)


def test_the_per_colony_url_is_pinned_to_the_repository_commit() -> None:
    """The QD file is fetched from the commit the raw mirror records, not ``main``."""
    assert c.QD_URL == (
        "https://raw.githubusercontent.com/pc2912/CRI-SPA_repo/"
        "2c800ab7b9811456f353d325be75b24e938440c6/QD1_2_4_6.csv"
    )


def test_download_refuses_a_response_under_ten_thousand_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """9999 bytes is under the floor; the request carries the URL, UA and 120 s timeout."""
    calls: list[tuple[str, str, int]] = []
    _fake_urlopen(monkeypatch, {c.DATA_URL: b"x" * 9999}, calls)
    dataset = _bare(tmp_path)
    with pytest.raises(RuntimeError) as excinfo:
        dataset.download()
    assert str(excinfo.value) == "CRI-SPA download too small: 9999 bytes"
    assert calls == [(c.DATA_URL, "Mozilla/5.0", 120)]
    assert os.listdir(tmp_path / "raw") == []


def test_download_refuses_a_response_off_the_pin_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = b"y" * 10000
    _fake_urlopen(monkeypatch, {c.DATA_URL: data}, [])
    dataset = _bare(tmp_path)
    with pytest.raises(RawSha256MismatchError) as excinfo:
        dataset.download()
    assert str(excinfo.value) == (
        f"sha256 mismatch for {c.DATA_URL}: expected {c.DATA_SHA256}, "
        f"observed {hashlib.sha256(data).hexdigest()}"
    )
    assert os.listdir(tmp_path / "raw") == []


def test_download_writes_both_files_on_their_pins_then_accepts_them_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With each pin set to its payload's digest both files land verbatim, GA first; a
    second call finds them and returns without a request (``process`` verifies them).
    """
    ga, qd = b"z" * 10000, b"q" * 20000
    monkeypatch.setattr(c, "DATA_SHA256", hashlib.sha256(ga).hexdigest())
    monkeypatch.setattr(c, "QD_SHA256", hashlib.sha256(qd).hexdigest())
    calls: list[tuple[str, str, int]] = []
    _fake_urlopen(monkeypatch, {c.DATA_URL: ga, c.QD_URL: qd}, calls)
    dataset = _bare(tmp_path)
    dataset.download()
    assert (tmp_path / "raw" / c.DATA_FILENAME).read_bytes() == ga
    assert (tmp_path / "raw" / c.QD_FILENAME).read_bytes() == qd
    dataset.download()
    assert [url for url, _, _ in calls] == [c.DATA_URL, c.QD_URL]


def test_items_retype_through_the_metabolite_classes_and_the_hooks_are_inert(
    built: c.BetaxanthinCachera2023Dataset,
) -> None:
    """``transform_item`` rebuilds a stored item as a ``MetaboliteExperiment`` with its
    reference, dumping to exactly the stored dictionaries; the raw file list is the two
    released CSVs, and ``preprocess_raw`` is the documented identity.
    """
    item = built[0]
    typed = built.transform_item(item)
    assert type(typed["experiment"]) is MetaboliteExperiment
    assert type(typed["reference"]) is MetaboliteExperimentReference
    assert typed["experiment"].model_dump() == item["experiment"]
    assert typed["reference"].model_dump() == item["reference"]
    assert built.raw_file_names == ["GA1_2_4_6.csv", "QD1_2_4_6.csv"]
    frame = pd.DataFrame({"a": [1]})
    assert built.preprocess_raw(frame) is frame


def test_main_builds_under_data_root_without_a_genome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main`` builds ``$DATA_ROOT/data/torchcell/betaxanthin_cachera2023`` with no
    genome, then prints the length and record 0.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    root = tmp_path / "data" / "torchcell" / "betaxanthin_cachera2023"
    _stage(root)
    c.main()
    lines = capsys.readouterr().out.splitlines()
    assert lines[-2] == "len = 8"
    rebuilt = c.BetaxanthinCachera2023Dataset(root=str(root))
    assert lines[-1] == str(rebuilt[0])


# --- data-gated: the real raw-mirror files -----------------------------------------

_DATA_ROOT = os.environ["DATA_ROOT"]
_MIRROR_DIR = osp.join(_DATA_ROOT, c.RAW_MIRROR_REL, "data")
_needs_mirror = pytest.mark.skipif(
    not osp.exists(osp.join(_MIRROR_DIR, c.QD_FILENAME)),
    reason="requires the Cachera raw mirror at $DATA_ROOT (absent in CI)",
)


@pytest.mark.data
@pytest.mark.slow
@_needs_mirror
def test_cachera_build_stores_every_measured_colony_as_released(tmp_path: Path) -> None:
    """The pinned per-colony file (79,872 positions = 4 repeats x 19,968) builds to
    68,834 records: 68,822 colonies of 4,759 gene tokens plus the 12 WT control colonies
    (8 in repeat 4, 4 in repeat 6). The other positions are 10,825 with no colony, 126
    the authors withdrew and 87 on library wells labeled ``0``. Grouping the colony
    records by gene and averaging reproduces every ``GA1_2_4_6.csv`` level to 1e-12 with
    the same colony count, so the old per-gene store is the mean of the new one. The
    61 tokens on two or three library wells keep every well's colonies apart by
    position (ACO2: 8 positions per repeat). Measured 2026.11.24 on the scratch build.
    """
    root = tmp_path / "betaxanthin_cachera2023"
    (root / "raw").mkdir(parents=True)
    for name in (c.DATA_FILENAME, c.QD_FILENAME):
        shutil.copy(osp.join(_MIRROR_DIR, name), root / "raw" / name)
    dataset = c.BetaxanthinCachera2023Dataset(root=str(root))
    assert len(dataset) == 68834

    levels: dict[str | None, list[float]] = defaultdict(list)
    per_repeat: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    rids: set[str] = set()
    for i in range(len(dataset)):
        experiment = dataset[i]["experiment"]
        deletions = [
            p
            for p in experiment["genotype"]["perturbations"]
            if p["perturbation_type"] != "gene_addition"
        ]
        gene = deletions[0]["systematic_gene_name"] if deletions else None
        kinds[deletions[0]["perturbation_type"] if deletions else "control"] += 1
        phenotype = experiment["phenotype"]
        levels[gene].append(phenotype["metabolite_level"]["betaxanthin"])
        rids.add(phenotype["replicate_id"])
        per_repeat[phenotype["replicate_id"].split("/")[0]] += 1
    assert len(rids) == 68834
    assert kinds == {
        "control": 12,
        "kanmx_deletion": 17045,
        "released_identifier": 51777,
    }
    assert per_repeat == {
        "repeat1": 17074,
        "repeat2": 18077,
        "repeat4": 15256,
        "repeat6": 18427,
    }
    assert len(levels) == 4760

    ga = pd.read_csv(osp.join(_MIRROR_DIR, c.DATA_FILENAME))
    ga["token"] = ga["gene"].astype(str).str.strip().str.upper()
    ga = ga[ga[c._LEVEL].notna() & ~ga["token"].isin(["0", "NAN"])].set_index("token")
    assert len(ga) == 4760
    counts = dict(zip(ga.index.tolist(), ga[c._COUNT].tolist(), strict=True))
    means = dict(zip(ga.index.tolist(), ga[c._LEVEL].tolist(), strict=True))
    worst = 0.0
    for gene, values in levels.items():
        token = "WT" if gene is None else gene
        assert len(values) == int(counts[token]), gene
        worst = max(worst, abs(sum(values) / len(values) - float(means[token])))
    assert worst < 1e-12

    reference = dataset[0]["reference"]["phenotype_reference"]
    assert reference["n_replicates"] == {"betaxanthin": 12}
    assert reference["metabolite_level"]["betaxanthin"] == pytest.approx(
        sum(levels[None]) / 12, abs=1e-12
    )
    assert len([r for r in rids if r.startswith("repeat1/")]) == 17074

    ledger = c.DropLog.model_validate_json(
        (Path(dataset.preprocess_dir) / "dropped_records.json").read_text()
    )
    assert (
        ledger.source_records,
        ledger.kept_records,
        ledger.kept_control_records,
        ledger.kept_under_systematic_name,
        ledger.kept_under_released_symbol,
        ledger.dropped_records,
    ) == (79872, 68834, 12, 17045, 51777, 11038)
    assert [(r.rule, r.n_records) for r in ledger.rules] == [
        ("no_colony_measured", 10825),
        ("withdrawn_by_the_authors", 126),
        ("gene_cell_is_0_or_blank", 87),
    ]

    data = pd.read_csv(Path(dataset.preprocess_dir) / "data.csv")
    aco2 = data[data["gene"] == "ACO2"]
    assert len(aco2[["plate", "row", "column"]].drop_duplicates()) == 8
    assert len(aco2) == 32


_PAPER_MD = osp.join(_DATA_ROOT, "torchcell-library", c.CITATION_KEY, c.PAPER_MD)


@pytest.mark.data
@pytest.mark.skipif(
    not osp.exists(_PAPER_MD), reason="requires the Cachera paper.md mirror"
)
def test_every_sourced_quote_is_on_its_line_of_the_pinned_paper_issue_509() -> None:
    """Each sourced quote is a verbatim substring of the pinned OCR, on its line."""
    data = Path(_PAPER_MD).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c.PAPER_MD_SHA256
    lines = data.decode("utf-8").split("\n")
    found = {
        key: [n for n, line in enumerate(lines, 1) if sv.quote in line]
        for key, sv in c.SOURCED_VALUES.items()
    }
    assert found == {
        "measurement_type": [58],
        "screen_medium": [117],
        "readout_medium": [58],
        "ypd_recipe": [31],
        "agar": [31],
        "g418": [31],
        "replicate_unit": [126],
        "replicates_per_gene": [145],
        "authors_withdrawal": [64],
        "pooled_withdrawal": [68],
        "control_cassette": [131],
        "control_is_by_btx": [148],
        "data_availability": [175],
    }
    assert not [
        line
        for line in lines
        if "°" in line or "\\circ" in line or "temperature" in line.lower()
    ]


@pytest.mark.data
@_needs_mirror
def test_the_raw_mirror_manifest_pins_the_files_the_loader_reads() -> None:
    """``manifest.json`` records both consumed files at the loader's pins, each fetched
    from the repository commit through the ``direct_url`` retriever, and the bytes on
    disk hash to them.
    """
    from torchcell.literature.manifest import Manifest

    root = osp.join(_DATA_ROOT, c.RAW_MIRROR_REL)
    manifest = Manifest.model_validate_json(Path(root, "manifest.json").read_text())
    by_path = {f.path: f for f in manifest.files}
    for name, pin in ((c.QD_FILENAME, c.QD_SHA256), (c.DATA_FILENAME, c.DATA_SHA256)):
        record = by_path[f"data/{name}"]
        assert record.sha256 == pin
        assert record.retrieval is not None
        assert record.retrieval.source_url == (
            f"https://raw.githubusercontent.com/pc2912/CRI-SPA_repo/{c.REPO_COMMIT}/{name}"
        )
        digest = hashlib.sha256(Path(root, record.path).read_bytes()).hexdigest()
        assert digest == pin
    assert by_path[f"data/{c.QD_FILENAME}"].bytes == c.QD_BYTES
