# tests/torchcell/datasets/ecoli/test_li2014.py
# [[tests.torchcell.datasets.ecoli.test_li2014]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_li2014.py
"""Tests for the Li 2014 protein synthesis-rate loader.

Everything outside the last block runs with NO network and NO real ``$DATA_ROOT``: Table
S1 is synthesized to the pinned workbook's exact shape, the raw mirror and its manifest
are written under ``tmp_path`` with the module's pins monkeypatched to the synthetic
digests, and the MG1655 annotation is the real ``EcoliK12MG1655Genome`` class over a
synthetic assembly. The synthetic rows exercise every typed refusal once. The last block,
skipped without the mirror, pins the numbers measured on the REAL bytes.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, cast

import openpyxl
import pytest

import torchcell.datasets.ecoli.li2014 as li
from tests.torchcell.sequence.genome._bacterial_fixtures import (
    SyntheticLocus,
    forbid_network,
    gaf_row,
    serve_tier,
    write_assembly,
)
from torchcell.datamodels.schema import (
    AssemblyReferenceGenome,
    ComponentDefinition,
    ProteinSynthesisRateExperiment,
    ProteinSynthesisRateExperimentReference,
    SynthesisRateUnit,
)
from torchcell.literature.manifest import ArtifactRecord, Manifest
from torchcell.sequence.genome.ecoli.k12 import MG1655_ASSEMBLY, EcoliK12MG1655Genome
from torchcell.verification.levels import l3_convention
from torchcell.verification.sourced import SourcedValue

# --------------------------------------------------------------------------- #
# A synthetic MG1655 assembly. ``yaaX`` is a synonym of thrC (a second Table S1 row
# for the same locus) and ``ambA`` is a synonym of two loci (ambiguous).
# --------------------------------------------------------------------------- #
LOCI = [
    SyntheticLocus(
        tag="b0001", parts=((1, 9),), strand="+", symbol="thrL", protein="MK"
    ),
    SyntheticLocus(
        tag="b0002",
        parts=((12, 26),),
        strand="-",
        symbol="thrA",
        synonyms=("ambA",),
        protein="MRVLK",
    ),
    SyntheticLocus(
        tag="b0003", parts=((30, 41),), strand="+", symbol="thrB", protein="MQQ"
    ),
    SyntheticLocus(
        tag="b0004",
        parts=((45, 56),),
        strand="+",
        symbol="thrC",
        synonyms=("yaaX", "ambA"),
        protein="MSY",
    ),
]
GAF = [
    gaf_row(cast(str, locus.symbol), f"{locus.symbol}|{locus.tag}", "GO:0000001")
    for locus in LOCI
]

#: ``(gene, MOPS complete, MOPS minimal, MOPS complete without methionine)``.
ROWS: tuple[tuple[str, Any, Any, Any], ...] = (
    ("thrL", 10, 5, "[3]"),
    ("thrA", 20, "[1]", 7),
    ("thrB", "[2]", 4, 0),
    ("thrC", 30, 6, 8),
    ("yaaX", 31, 7, "[1]"),
    ("thrA+thrB", 50, 60, "[9]"),
    ("insZ", 9, 9, 9),
    ("ambA", 3, 3, 3),
)
#: Plain cells per column, and the keys left once every typed refusal is applied.
PLAIN = {"complete": 7, "minimal": 7, "complete_without_methionine": 5}
STORED = {
    "complete": {"b0001": 10.0, "b0002": 20.0},
    "minimal": {"b0001": 5.0, "b0003": 4.0},
    "complete_without_methionine": {"b0002": 7.0, "b0003": 0.0},
}


def write_table_s1(path: Path, rows: tuple[tuple[str, Any, Any, Any], ...]) -> Path:
    """Table S1 with the pinned sheet name and header."""
    book = openpyxl.Workbook()
    sheet = book.active
    assert sheet is not None
    sheet.title = li.TABLE_S1_SHEET
    sheet.append(["Gene", *(spec.column for spec in li.MEDIA)])
    for row in rows:
        sheet.append(list(row))
    book.save(path)
    return path


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("cell", "expected"), [(12, 12), (0, 0), ("[3]", None), ("[0]", None)]
)
def test_a_cell_is_a_plain_integer_or_a_bracketed_one(
    cell: int | str, expected: int | None
) -> None:
    assert li.plain_value(cell) == expected


@pytest.mark.parametrize("cell", ["NA", "[x]", "3", 1.5, True, None])
def test_a_cell_of_any_other_shape_is_refused(cell: Any) -> None:
    with pytest.raises(ValueError, match="neither an integer nor"):
        li.plain_value(cell)


def test_the_three_media_are_the_si_media_with_their_doubling_times() -> None:
    by_key = {spec.key: spec for spec in li.MEDIA}
    assert {k: v.generation_time_minutes for k, v in by_key.items()} == {
        "complete": 21.5,
        "minimal": 56.3,
        "complete_without_methionine": 26.5,
    }
    assert {k: v.column for k, v in by_key.items()} == {
        "complete": "MOPS complete",
        "minimal": "MOPS minimal",
        "complete_without_methionine": "MOPS complete without methionine",
    }
    assert {k: (v.plain_cells, v.stored_keys) for k, v in by_key.items()} == {
        "complete": (3041, 3025),
        "minimal": (3362, 3346),
        "complete_without_methionine": (2241, 2226),
    }
    assert li.EVALUATED_GENES.value == by_key["complete"].plain_cells
    for spec in li.MEDIA:
        assert f"{spec.generation_time_minutes} ±" in spec.generation_time_source.quote


def test_each_medium_is_mops_glucose_with_its_own_supplement() -> None:
    media = {spec.key: li.medium(spec) for spec in li.MEDIA}
    assert len({m.name for m in media.values()}) == 3
    deferred = {
        key: [
            c.compound.name
            for c in m.components
            if c.definition is ComponentDefinition.composition_deferred
        ]
        for key, m in media.items()
    }
    assert deferred == {
        "complete": ["full supplement (Neidhardt et al., 1974)"],
        "minimal": [],
        "complete_without_methionine": [
            "full supplement without L-methionine (Neidhardt et al., 1974)"
        ],
    }
    for m in media.values():
        glucose = [c for c in m.components if c.role.value == "carbon_source"]
        assert len(glucose) == 1
        assert glucose[0].concentration is not None
        assert glucose[0].concentration.value == pytest.approx(0.2)
        assert m.base_medium == "MOPS_MINIMAL"
    environment = li.environment(li.MEDIA[0])
    assert environment.temperature is not None
    assert environment.temperature.value == pytest.approx(37.0)
    assert environment.aerobicity == "aerobic"
    assert environment.perturbations == []


def test_the_phenotype_is_per_generation_and_declares_what_is_not_released() -> None:
    values = li.MediumValues(synthesis_rate={"b0001": 3.0}, refused=[])
    phenotype = li.phenotype(li.MEDIA[1], values)
    assert phenotype.rate_unit is SynthesisRateUnit.molecules_per_generation
    assert phenotype.generation_time_minutes == pytest.approx(56.3)
    assert phenotype.measurement_type == li.MEASUREMENT_TYPE
    assert phenotype.n_replicates is None
    assert phenotype.synthesis_rate_se is None
    assert sorted(gap.field for gap in phenotype.provenance_gaps) == [
        "n_replicates",
        "synthesis_rate_se",
    ]


def test_the_publication_is_this_paper() -> None:
    publication = li.publication()
    assert (publication.doi, publication.pubmed_id) == (li.DOI, "24766808")


def test_a_name_outside_the_namespace_without_a_typed_reason_stops_the_build() -> None:
    class Reconciliation:
        retired_kept = ("insZ",)
        ambiguous_kept: dict[str, tuple[str, ...]] = {}
        kept_on_collision: tuple[str, ...] = ()
        outside_namespace = ("insZ", "mystery")

    with pytest.raises(RuntimeError, match="no typed reason"):
        li.name_refusals(cast(Any, Reconciliation()))


# --------------------------------------------------------------------------- #
# Synthetic: the loader end to end
# --------------------------------------------------------------------------- #
@pytest.fixture
def genome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EcoliK12MG1655Genome:
    """The real MG1655 genome class over the synthetic assembly; network refuses."""
    files = write_assembly(tmp_path / "tier", MG1655_ASSEMBLY, LOCI, GAF)
    forbid_network(monkeypatch)
    serve_tier(monkeypatch, files)
    return EcoliK12MG1655Genome(genome_root=str(tmp_path / "mg1655"), overwrite=True)


def _pin(strain: Any, **_: Any) -> AssemblyReferenceGenome:
    """The assembly pin, without reading the genomes tier's assembly report."""
    return AssemblyReferenceGenome(
        species="Escherichia coli",
        strain=strain,
        assembly_set=cast(Any, li.MG1655_ASSEMBLY_SET),
        assembly_accession="GCA_000005845.2",
    )


@pytest.fixture
def mirrored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, genome: EcoliK12MG1655Genome
) -> Path:
    """A tmp ``DATA_ROOT`` holding the synthetic mirror, with the synthetic pins."""
    data_root = tmp_path / "data_root"
    mirror = data_root / li.RAW_DIR_REL
    (mirror / "data").mkdir(parents=True)
    table = write_table_s1(mirror / li.TABLE_S1_REL, ROWS)
    digest = hashlib.sha256(table.read_bytes()).hexdigest()
    manifest = Manifest(
        citation_key=li.CITATION_KEY,
        files=[
            ArtifactRecord(
                path=li.TABLE_S1_REL,
                role="raw_data",
                bytes=table.stat().st_size,
                sha256=digest,
                source="synthetic",
            )
        ],
    )
    (mirror / "manifest.json").write_text(manifest.model_dump_json())
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    monkeypatch.setattr(li, "TABLE_S1_SHA256", digest)
    monkeypatch.setattr(li, "SOURCE_ROWS", len(ROWS))
    monkeypatch.setattr(
        li,
        "MEDIA",
        tuple(
            spec.model_copy(
                update={
                    "plain_cells": PLAIN[spec.key],
                    "stored_keys": len(STORED[spec.key]),
                }
            )
            for spec in li.MEDIA
        ),
    )
    monkeypatch.setattr(
        li,
        "EVALUATED_GENES",
        li.EVALUATED_GENES.model_copy(update={"value": PLAIN["complete"]}),
    )
    monkeypatch.setattr(li, "assembly_reference", _pin)
    monkeypatch.setattr(
        li, "bacterial_genome", lambda host, strain, data_root=None: genome
    )
    return data_root


def _by_medium(dataset: li.ProteinSynthesisRateLi2014Dataset) -> dict[str, Any]:
    records = [dataset[index] for index in range(len(dataset))]
    return {li._medium_of(record): record for record in records}


def test_the_loader_stores_plain_cells_and_refuses_the_rest_by_type(
    tmp_path: Path, mirrored: Path, genome: EcoliK12MG1655Genome
) -> None:
    root = tmp_path / "dataset"
    dataset = li.ProteinSynthesisRateLi2014Dataset(root=str(root), ecoli_genome=genome)
    assert len(dataset) == li.EXPECTED_RECORDS
    assert dataset.gene_set == set()
    by_medium = _by_medium(dataset)
    for key, expected in STORED.items():
        experiment = by_medium[key]["experiment"]
        assert experiment["phenotype"]["synthesis_rate"] == expected
        assert experiment["genotype"]["perturbations"] == []
        reference = by_medium[key]["reference"]
        assert reference["phenotype_reference"]["synthesis_rate"] == STORED["complete"]

    refused = (root / "preprocess" / "refused_cells.csv").read_text().splitlines()
    assert refused[0] == "medium,gene,cell,reason"
    assert sorted(refused[1:]) == sorted(
        [
            "complete,thrB,[2],below_read_count_gate",
            "complete,thrC,30,two_rows_one_locus",
            "complete,yaaX,31,two_rows_one_locus",
            "complete,thrA+thrB,50,merged_pair_one_value_two_loci",
            "complete,insZ,9,name_retired_in_assembly",
            "complete,ambA,3,name_ambiguous_in_assembly",
            "minimal,thrA,[1],below_read_count_gate",
            "minimal,thrC,6,two_rows_one_locus",
            "minimal,yaaX,7,two_rows_one_locus",
            "minimal,thrA+thrB,60,merged_pair_one_value_two_loci",
            "minimal,insZ,9,name_retired_in_assembly",
            "minimal,ambA,3,name_ambiguous_in_assembly",
            "complete_without_methionine,thrL,[3],below_read_count_gate",
            "complete_without_methionine,thrC,8,two_rows_one_locus",
            "complete_without_methionine,yaaX,[1],below_read_count_gate",
            "complete_without_methionine,thrA+thrB,[9],below_read_count_gate",
            "complete_without_methionine,insZ,9,name_retired_in_assembly",
            "complete_without_methionine,ambA,3,name_ambiguous_in_assembly",
        ]
    )
    accounting = li.BuildAccounting.model_validate_json(
        (root / "preprocess" / "build_accounting.json").read_text()
    )
    assert accounting.per_medium_keys == {k: len(v) for k, v in STORED.items()}
    assert accounting.per_medium_zero_rates == {
        "complete": 0,
        "minimal": 0,
        "complete_without_methionine": 1,
    }
    assert (root / "raw" / li.TABLE_S1_FILENAME).is_symlink()


def test_the_verifier_passes_on_the_synthetic_build_and_fails_a_wrong_count(
    tmp_path: Path,
    mirrored: Path,
    genome: EcoliK12MG1655Genome,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "dataset"
    li.ProteinSynthesisRateLi2014Dataset(root=str(root), ecoli_genome=genome)
    monkeypatch.setattr(
        li, "_l3_quotes_are_verbatim", lambda data_root: l3_convention("quotes", True)
    )
    report = li.verify_build(str(root), genome=genome)
    assert report.passed, report.summary()
    assert {r.name for r in report.results} >= {
        "per_medium_key_counts_are_the_pinned_ones",
        "generation_time_is_the_si_doubling_time",
        "assembly_pin_is_mg1655_genbank",
        "stored_protein_keys_are_loci_of_the_pinned_assembly",
    }
    assert (root / "preprocess" / "verification_report.json").is_file()
    assert not li.verify_build(str(root), genome=genome, expected_count=2).passed


def test_a_plain_cell_count_off_its_pin_stops_the_build(
    tmp_path: Path,
    mirrored: Path,
    genome: EcoliK12MG1655Genome,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        li, "MEDIA", (li.MEDIA[0].model_copy(update={"plain_cells": 6}), *li.MEDIA[1:])
    )
    with pytest.raises(RuntimeError, match="7 plain cells, not the pinned 6"):
        li.ProteinSynthesisRateLi2014Dataset(
            root=str(tmp_path / "dataset"), ecoli_genome=genome
        )


def test_a_header_that_is_not_the_pinned_one_stops_the_build(
    tmp_path: Path,
    mirrored: Path,
    genome: EcoliK12MG1655Genome,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        li,
        "MEDIA",
        (li.MEDIA[0].model_copy(update={"column": "MOPS rich"}), *li.MEDIA[1:]),
    )
    with pytest.raises(RuntimeError, match="Table S1 header"):
        li.ProteinSynthesisRateLi2014Dataset(
            root=str(tmp_path / "dataset"), ecoli_genome=genome
        )


def test_a_direct_run_opens_the_mg1655_genome_itself(
    mirrored: Path, genome: EcoliK12MG1655Genome
) -> None:
    dataset = li.ProteinSynthesisRateLi2014Dataset.__new__(
        li.ProteinSynthesisRateLi2014Dataset
    )
    dataset.ecoli_genome = None
    assert dataset._genome() is genome


def test_the_loader_declares_its_schema_pair_and_its_raw_file() -> None:
    dataset = li.ProteinSynthesisRateLi2014Dataset.__new__(
        li.ProteinSynthesisRateLi2014Dataset
    )
    assert dataset.experiment_class is ProteinSynthesisRateExperiment
    assert dataset.reference_class is ProteinSynthesisRateExperimentReference
    assert dataset.raw_file_names == [li.TABLE_S1_FILENAME]
    assert li.ProteinSynthesisRateLi2014Dataset.has_gene_perturbations is False
    with pytest.raises(NotImplementedError):
        dataset.create_experiment()
    assert dataset.preprocess_raw("x") == "x"


# --------------------------------------------------------------------------- #
# Synthetic: the verbatim-quote audit
# --------------------------------------------------------------------------- #
def _synthetic_text_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drop: str | None = None
) -> Path:
    """Text files holding every quote (less ``drop``), re-pinned to their digests."""
    data_root = tmp_path / "quotes_root"
    mirror = data_root / li.RAW_DIR_REL
    texts: dict[str, list[str]] = {}
    for sv in li.TEXT_SOURCED_VALUES:
        if sv.quote != drop:
            texts.setdefault(sv.provenance.source_uri, []).append(sv.quote)
    digests: dict[str, str] = {}
    for relpath in {sv.provenance.source_uri for sv in li.TEXT_SOURCED_VALUES}:
        path = mirror / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(texts.get(relpath, [])), encoding="utf-8")
        digests[relpath] = hashlib.sha256(path.read_bytes()).hexdigest()
    repinned: list[SourcedValue] = [
        sv.model_copy(
            update={
                "provenance": sv.provenance.model_copy(
                    update={"sha256": digests[sv.provenance.source_uri]}
                )
            }
        )
        for sv in li.TEXT_SOURCED_VALUES
    ]
    monkeypatch.setattr(li, "TEXT_SOURCED_VALUES", tuple(repinned))
    (mirror / "data").mkdir(parents=True)
    table = write_table_s1(mirror / li.TABLE_S1_REL, ROWS)
    monkeypatch.setattr(
        li, "TABLE_S1_SHA256", hashlib.sha256(table.read_bytes()).hexdigest()
    )
    return data_root


def test_the_quote_audit_passes_when_every_quote_is_in_its_pinned_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _synthetic_text_mirror(tmp_path, monkeypatch)
    result = li._l3_quotes_are_verbatim(str(data_root))
    assert result.passed, result.message


def test_the_quote_audit_fails_a_quote_that_is_not_in_its_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _synthetic_text_mirror(
        tmp_path, monkeypatch, drop=li.READ_COUNT_GATE.quote
    )
    result = li._l3_quotes_are_verbatim(str(data_root))
    assert not result.passed
    assert "1 failed" in result.message


# --------------------------------------------------------------------------- #
# The REAL released bytes: skipped unless the raw mirror is on this machine
# --------------------------------------------------------------------------- #
def _real_mirror() -> Path | None:
    data_root = os.environ.get("DATA_ROOT")
    if not data_root:
        return None
    path = Path(data_root) / li.RAW_DIR_REL / li.TABLE_S1_REL
    return path if path.is_file() else None


requires_mirror = pytest.mark.skipif(
    _real_mirror() is None, reason="the Li 2014 raw mirror is not on this machine"
)


@requires_mirror
def test_the_released_table_has_the_pinned_plain_cell_counts() -> None:
    path = _real_mirror()
    assert path is not None
    assert hashlib.sha256(path.read_bytes()).hexdigest() == li.TABLE_S1_SHA256
    header, rows = li.read_table_s1(path)
    assert header == ("Gene", *(spec.column for spec in li.MEDIA))
    assert len(rows) == li.SOURCE_ROWS
    for column, spec in enumerate(li.MEDIA):
        plain = sum(1 for row in rows if li.plain_value(row.cells[column]) is not None)
        assert plain == spec.plain_cells


@requires_mirror
def test_every_released_quote_is_verbatim_in_the_mirror() -> None:
    result = li._l3_quotes_are_verbatim(None)
    assert result.passed, result.message
