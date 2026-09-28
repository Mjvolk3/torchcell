# tests/torchcell/datasets/scerevisiae/test_bloom2019_synthetic.py
# [[tests.torchcell.datasets.scerevisiae.test_bloom2019_synthetic]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_bloom2019_synthetic.py
"""Bloom 2019 segregant loader built end to end on two synthetic crosses.

``process()`` is pinned to the release (16 crosses, 13,950 segregants, 530,100 records),
so the four release constants ``CROSSES``, ``EXPECTED_SEGREGANTS``, ``N_SEGREGANTS`` and
``EXPECTED_RECORDS`` are repointed with ``monkeypatch`` at a two-cross fixture (the
YeastPhenome precedent). ``read_cross_table`` hardcodes ``engine="xlrd"`` (bloom2019.py
line 737) and no BIFF writer is installed, so ``pandas.read_excel`` is wrapped to read the
openpyxl-written sheet with openpyxl when ``xlrd`` is requested; every other line of the
xls join runs unchanged. The build also needs ``$DATA_ROOT``: the raw mirror's
``manifest.json`` (each genotype matrix's sha256 is stored on the record) and the genomes
tier's ``peter2018_1011_assemblies`` manifest (the tarball pin plus the member index), both
written under ``tmp_path``.

Crosses: 375 = BYa x M22 (xls lists them M22 first, README BYa first), A = BYa x RMx.
Peter ids M22 -> ADR, RMx -> AAA; BY is the S288C reference. Genotype 375 has six markers
in the file order chrI_100, chrI_300, chrI_200, chrII_50, chrII_60, chrIII_10 (sorted:
100, 200, 300 | 50, 60 | 10, a permutation of the header):

    375_seg1  calls 1 2 2 2 2 1 -> sorted 1 2 2 2 2 1 -> chrI(100,100,p1,1) chrI(200,300,p2,2)
                                                          chrII(50,60,p2,2) chrIII(10,10,p1,1)   4 blocks
    375_seg2  calls 2 2 2 1 1 1 -> sorted 2 2 2 1 1 1 -> chrI(100,300,p2,3) chrII(50,60,p1,2)
                                                          chrIII(10,10,p1,1)                       3 blocks
    A_seg1    markers chrI_150, chrIV_500; calls 1 2 -> chrI(150,150,p1,1) chrIV(500,500,p2,1)  2 blocks

Phenotypes: 40 columns written as YPD;;2, the 38 served columns in ``build_conditions``
order, then YPD;;3; the value of segregant s (0, 1, 2 in file order) at written position
p is ``s * 100 + p + 0.5``, so served column k of segregant s stores ``s * 100 + k + 1.5``
(6-azauracil of 375_seg1 is 1.5, YPD;;1 (k = 34) of 375_seg1 is 35.5). Records are one
per (segregant, served column) in that order: 3 x 38 = 114. References: residual on a
YPD control (36 columns less YNB;ph3/ph8 = 34 columns), absolute YNB;;1, residual on the
YNB control (2 columns), absolute YPD;;1, first seen at k = 0, 31, 32, 34.

Gene set: stub genes YAL001C chrI 120-180 and YBL001C chrII 55-58 overlap cross 375's
spans, YDL001W chrIV 490-510 overlaps cross A's; YAL002W chrI 400-500 and YCL001W
chrIII 20-30 overlap none.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import lmdb
import openpyxl
import pandas as pd
import pytest

from torchcell.datamodels.media import YNB_GLUCOSE_SOLID, YP_GALACTOSE, YPD
from torchcell.datamodels.schema import (
    AssayType,
    Compound,
    Concentration,
    ConcentrationUnit,
    Environment,
    EnvironmentPhysicalPerturbation,
    EnvironmentResponsePhenotype,
    HaplotypeBlock,
    MeasurementType,
    PhysicalFactor,
    Publication,
    ReferenceGenome,
    SampleUnit,
    SegregantGenotype,
    SegregantGrowthExperiment,
    SegregantGrowthExperimentReference,
    SegregantParent,
    SmallMoleculePerturbation,
    Temperature,
)
from torchcell.datasets.scerevisiae import bloom2019 as m
from torchcell.literature.manifest import ROLE_RAW_DATA, ArtifactRecord, Manifest
from torchcell.sequence.genome.registry import GenomeManifest
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

_CITATION_KEY = "bloomRareVariantsContribute2019"
_DOI = "10.7554/eLife.49212"
_TAR_SHA = "53540d095958ae8c32509c04485f0d2d0948069c7647f828698d611899a9b4da"
_BY_GENOTYPE = "BY MATa his3d1 leu2d0 ura3d0 ho::KanMX"
_M22_GENOTYPE = "M22 MATalpha ho::HygMX"
_RM_GENOTYPE = "RM MATalpha ho::HygMX"

_CROSSES = ["375", "A"]
_EXPECTED_SEGREGANTS = {"375": 2, "A": 1}
_README_ROWS: list[tuple[str, ...]] = [("375", "BYa", "M22"), ("A", "BYa", "RMx")]
_DEFAULT_GENOME = object()
# (diploid parent, xls Parent 1, xls Parent 2, Peter id of parent 1, cross, n segregants)
_XLS_ROWS: list[tuple[str, str, str, str, str, int]] = [
    ("BYxM22", _M22_GENOTYPE, _BY_GENOTYPE, "ADR", "375", 2),
    ("BYxRM", _BY_GENOTYPE, _RM_GENOTYPE, "AAA", "A", 1),
]
_MATRICES: dict[str, tuple[list[str], list[tuple[str, list[int]]]]] = {
    "375": (
        [
            "chrI_100_A_T_1",
            "chrI_300_A_T_2",
            "chrI_200_A_T_3",
            "chrII_50_G_C_4",
            "chrII_60_GA_G_5",
            "chrIII_10_A_T,C_6",
        ],
        [("375_seg1", [1, 2, 2, 2, 2, 1]), ("375_seg2", [2, 2, 2, 1, 1, 1])],
    ),
    "A": (["chrI_150_A_T_1", "chrIV_500_C_G_2"], [("A_seg1", [1, 2])]),
}
_SERVED = list(m.build_conditions())
_PHENOTYPE_COLUMNS = ["YPD;;2", *_SERVED, "YPD;;3"]
_SEGREGANTS = ["375_seg1", "375_seg2", "A_seg1"]
_INDEX_LINES = "ADR\t1011Assemblies/ADR.re.fa\nAAA\t1011Assemblies/AAA.re.fa\n"


class _Feature(SimpleNamespace):
    chrom: str
    start: int
    end: int


class _StubGenome:
    """``db`` (chrom/start/end per gene) and ``gene_set``, all ``compute_gene_set`` reads."""

    gene_set = ["YAL001C", "YAL002W", "YBL001C", "YCL001W", "YDL001W"]
    db = {
        "YAL001C": _Feature(chrom="chrI", start=120, end=180),
        "YAL002W": _Feature(chrom="chrI", start=400, end=500),
        "YBL001C": _Feature(chrom="chrII", start=55, end=58),
        "YCL001W": _Feature(chrom="chrIII", start=20, end=30),
        "YDL001W": _Feature(chrom="chrIV", start=490, end=510),
    }


def _genome() -> SCerevisiaeGenome:
    return cast(SCerevisiaeGenome, _StubGenome())


def _write_gzip_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with gzip.open(path, "wt") as handle:
        handle.write("\t".join(["", *header]) + "\n")
        for row in rows:
            handle.write("\t".join(row) + "\n")


def _write_readme(path: Path, rows: list[tuple[str, ...]]) -> None:
    lines = [
        "Genotype files for the crosses.",
        "The allele coding (1,2) corresponding to Parent 1 or Parent 2, is indicated "
        "in the table below",
        "cross\tParent1\tParent2",
        *["\t".join(row) + "\t" for row in rows],
        "",
        "Marker names are <chr>_<pos>_<ref>_<alt>_<index>.",
    ]
    path.write_text("\n".join(lines) + "\n")


def _write_xls(path: Path, rows: list[tuple[str, str, str, str, str, int]]) -> None:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Crosses and Strains"
    sheet.append(
        [
            "Diploid Parent",
            "Parent 1",
            "Parent 2",
            "Strain ID of Parent 1 in Peter et al. 2018",
            "name of cross used in provided code",
            "Number of Segregants Analyzed",
        ]
    )
    for row in rows:
        sheet.append(list(row))
    workbook.save(path)


def _phenotype_rows(segregants: list[str], columns: list[str]) -> list[list[str]]:
    return [
        [seg, *[str(s * 100 + p + 0.5) for p in range(len(columns))]]
        for s, seg in enumerate(segregants)
    ]


def _write_data_files(
    directory: Path,
    *,
    matrices: dict[str, tuple[list[str], list[tuple[str, list[int]]]]] = _MATRICES,
    phenotype_columns: list[str] = _PHENOTYPE_COLUMNS,
    phenotype_rows: list[list[str]] | None = None,
    readme_rows: list[tuple[str, ...]] = _README_ROWS,
    xls_rows: list[tuple[str, str, str, str, str, int]] = _XLS_ROWS,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for cross, (header, rows) in matrices.items():
        _write_gzip_tsv(
            directory / f"genotype_{cross}.tsv.gz",
            header,
            [[seg, *map(str, calls)] for seg, calls in rows],
        )
    _write_gzip_tsv(
        directory / "phenotypes.tsv.gz",
        phenotype_columns,
        phenotype_rows or _phenotype_rows(_SEGREGANTS, phenotype_columns),
    )
    _write_readme(directory / "cross_genotypes_README", readme_rows)
    _write_xls(directory / "elife-49212-fig1-data1-v2.xls", xls_rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_mirror_manifest(
    data_root: Path, files: dict[str, Path], sha_override: dict[str, str] | None = None
) -> Path:
    """``torchcell-raw/<key>/manifest.json`` listing ``files`` (rel path -> file)."""
    mirror = data_root / "torchcell-raw" / _CITATION_KEY
    mirror.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(
        citation_key=_CITATION_KEY,
        files=[
            ArtifactRecord(
                path=rel,
                role=ROLE_RAW_DATA,
                bytes=path.stat().st_size,
                sha256=(sha_override or {}).get(rel, _sha256(path)),
            )
            for rel, path in files.items()
        ],
        provenance_complete=True,
    )
    (mirror / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return mirror


def _write_genomes_tier(
    data_root: Path, *, tar_sha: str = _TAR_SHA, index_lines: str = _INDEX_LINES
) -> None:
    tier = data_root / "torchcell-genomes" / "peter2018_1011_assemblies"
    tier.mkdir(parents=True, exist_ok=True)
    index = tier / "1011Assemblies.tar.gz.member_index.tsv"
    index.write_text(index_lines)
    manifest = GenomeManifest(
        assembly_set="peter2018_1011_assemblies",
        organism="Saccharomyces cerevisiae",
        strain_or_population="1,011 isolates",
        source="Peter et al. 2018",
        release="2018",
        files=[
            ArtifactRecord(
                path="1011Assemblies.tar.gz", role="container", bytes=1, sha256=tar_sha
            ),
            ArtifactRecord(
                path=index.name,
                role="index",
                bytes=index.stat().st_size,
                sha256=_sha256(index),
            ),
        ],
        provenance_complete=True,
        created_at="2026-09-27T00:00:00+00:00",
    )
    (tier / "manifest.json").write_text(manifest.model_dump_json(indent=2))


def _patch_release(
    monkeypatch: pytest.MonkeyPatch, *, expected_records: int = 114
) -> None:
    monkeypatch.setattr(m, "CROSSES", list(_CROSSES))
    monkeypatch.setattr(m, "EXPECTED_SEGREGANTS", dict(_EXPECTED_SEGREGANTS))
    monkeypatch.setattr(m, "N_SEGREGANTS", 3)
    monkeypatch.setattr(m, "EXPECTED_RECORDS", expected_records)
    original = pd.read_excel

    def read_excel(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("engine") == "xlrd":
            kwargs["engine"] = "openpyxl"
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_excel", read_excel)


def _raw_files(directory: Path) -> dict[str, Path]:
    names = [
        "phenotypes.tsv.gz",
        "cross_genotypes_README",
        "elife-49212-fig1-data1-v2.xls",
    ]
    names += [f"genotype_{cross}.tsv.gz" for cross in _CROSSES]
    return {f"data/{name}": directory / name for name in names}


def _build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    slug: str = "bloom2019",
    genome: Any = _DEFAULT_GENOME,
    tar_sha: str = _TAR_SHA,
    index_lines: str = _INDEX_LINES,
    **data_kwargs: Any,
) -> m.Bloom2019Dataset:
    """Raw files under ``<root>/raw`` (so ``download()`` is skipped), the mirror manifest
    and the genomes tier under a ``tmp_path`` ``DATA_ROOT``.
    """
    _patch_release(monkeypatch)
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    root = tmp_path / slug
    _write_data_files(root / "raw", **data_kwargs)
    _write_mirror_manifest(data_root, _raw_files(root / "raw"))
    _write_genomes_tier(data_root, tar_sha=tar_sha, index_lines=index_lines)
    return m.Bloom2019Dataset(
        root=str(root), genome=_genome() if genome is _DEFAULT_GENOME else genome
    )


@pytest.fixture
def dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> m.Bloom2019Dataset:
    return _build(tmp_path, monkeypatch)


_CALL_METHOD = (
    "R/qtl argmax.geno hard call written by analysis/mapping.R "
    "(`g=pull.argmaxgeno(cross)`), released as genotype_<cross>.tsv.gz "
    "(1 = parent 1, 2 = parent 2)"
)
_BY = SegregantParent(
    name="BYa",
    peter_strain_id=None,
    assembly_member="S288C_reference_genome_R64-4-1_20230830 (SGD; torchcell reference)",
    assembly_sha256="S288C reference (SGD R64-4-1); see ReferenceGenome",
    engineered_background=_BY_GENOTYPE,
)
_M22 = SegregantParent(
    name="M22",
    peter_strain_id="ADR",
    assembly_member="1011Assemblies.tar.gz::1011Assemblies/ADR.re.fa",
    assembly_sha256=_TAR_SHA,
    engineered_background=_M22_GENOTYPE,
)
_RM = SegregantParent(
    name="RMx",
    peter_strain_id="AAA",
    assembly_member="1011Assemblies.tar.gz::1011Assemblies/AAA.re.fa",
    assembly_sha256=_TAR_SHA,
    engineered_background=_RM_GENOTYPE,
)


def _block(
    chrom: str, start: int, end: int, parent: Literal[1, 2], n: int
) -> HaplotypeBlock:
    return HaplotypeBlock(
        chromosome=chrom,
        start=start,
        end=end,
        parent=parent,
        posterior=1.0,
        n_markers=n,
    )


def _genotype(dataset: m.Bloom2019Dataset, segregant: str) -> SegregantGenotype:
    cross = segregant.split("_")[0]
    matrix_sha = _sha256(Path(dataset.raw_dir) / f"genotype_{cross}.tsv.gz")
    blocks = {
        "375_seg1": [
            _block("chrI", 100, 100, 1, 1),
            _block("chrI", 200, 300, 2, 2),
            _block("chrII", 50, 60, 2, 2),
            _block("chrIII", 10, 10, 1, 1),
        ],
        "375_seg2": [
            _block("chrI", 100, 300, 2, 3),
            _block("chrII", 50, 60, 1, 2),
            _block("chrIII", 10, 10, 1, 1),
        ],
        "A_seg1": [_block("chrI", 150, 150, 1, 1), _block("chrIV", 500, 500, 2, 1)],
    }[segregant]
    return SegregantGenotype(
        cross=cross,
        segregant_id=segregant,
        parent_1=_BY,
        parent_2=_M22 if cross == "375" else _RM,
        blocks=blocks,
        call_method=_CALL_METHOD,
        marker_matrix_sha256=matrix_sha,
    )


def _environment(
    media: Any, temperature: float, perturbations: list[Any]
) -> Environment:
    return Environment(
        media=media,
        temperature=Temperature(value=temperature),
        perturbations=perturbations,
        aerobicity="aerobic",
        duration_hours=48.0,
    )


_SE_GAP = ProvenanceGap(
    field="environment_response_se",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="the release carries the duplicate-averaged value only",
)


def _residual_phenotype(value: float, control: str) -> EnvironmentResponsePhenotype:
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.control_regression_residual,
        assay_type=AssayType.colony_size_array,
        environment_response=value,
        n_samples=2,
        sample_unit=SampleUnit.technical_replicate,
        units=(
            "residual of colony mean radius regressed on the same segregant's radius on "
            f"the matched control plate {control} (same cross, batch and layout; "
            "residuals(lm(s.radius.mean ~ ctrl.s.radius.mean)), process_images.R), "
            "duplicate plates averaged and missing cells mean-imputed per trait within a "
            "cross (mapping.R)"
        ),
        provenance_gaps=[_SE_GAP],
    )


def _absolute_phenotype(value: float) -> EnvironmentResponsePhenotype:
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.colony_size,
        assay_type=AssayType.colony_size_array,
        environment_response=value,
        n_samples=2,
        sample_unit=SampleUnit.technical_replicate,
        units=(
            "end-point colony mean radius in image pixels (s.radius.mean, "
            "process_images.R), duplicate plates averaged and missing cells mean-imputed "
            "per trait within a cross (mapping.R); the control plate of its batch"
        ),
        provenance_gaps=[_SE_GAP],
    )


def _reference(media: Any, absolute: bool) -> dict[str, Any]:
    if absolute:
        note = (
            "absolute colony size: the parents' radii are not released; 0 is the scale's "
            "physical zero (no colony), flagged for review"
        )
        gaps = [
            ProvenanceGap(
                field="environment_response_se",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note=note,
            )
        ]
        measurement = MeasurementType.colony_size
    else:
        note = (
            "residual reference: a segregant growing exactly as predicted from its "
            "control-plate growth has residual 0"
        )
        gaps = []
        measurement = MeasurementType.control_regression_residual
    return SegregantGrowthExperimentReference(
        dataset_name="Bloom2019Dataset",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="S288C", ploidy="haploid"
        ),
        environment_reference=_environment(media, 30.0, []),
        phenotype_reference=EnvironmentResponsePhenotype(
            measurement_type=measurement,
            assay_type=AssayType.colony_size_array,
            environment_response=0.0,
            units=note,
            provenance_gaps=gaps,
        ),
    ).model_dump()


def test_three_segregants_times_38_conditions_in_cross_then_column_order(
    dataset: m.Bloom2019Dataset,
) -> None:
    """114 records; segregant s's served column k stores ``s * 100 + k + 1.5`` and
    every record of one segregant carries that segregant's mosaic (4, 3 and 2 blocks).
    """
    assert len(dataset) == 114
    seen = []
    for i in range(114):
        experiment = dataset[i]["experiment"]
        seen.append(
            (
                experiment["genotype"]["segregant_id"],
                len(experiment["genotype"]["blocks"]),
                experiment["phenotype"]["environment_response"],
            )
        )
    assert seen == [
        (seg, n_blocks, s * 100 + k + 1.5)
        for s, (seg, n_blocks) in enumerate(
            [("375_seg1", 4), ("375_seg2", 3), ("A_seg1", 2)]
        )
        for k in range(38)
    ]


def test_first_record_is_6_azauracil_on_ypd_with_the_batch_3_control_reference(
    dataset: m.Bloom2019Dataset,
) -> None:
    """Record 0: 375_seg1 (BY x M22, four blocks, matrix sha256 of genotype_375.tsv.gz)
    on YPD + 6-azauracil 50 ug/mL at 30 C for 48 h, residual 1.5 over duplicate plates;
    the reference is the YPD;;1 environment (the batch-3 control plate is the same
    medium) with residual 0 and no gap.
    """
    assert (
        dataset[0]["experiment"]
        == SegregantGrowthExperiment(
            dataset_name="Bloom2019Dataset",
            genotype=_genotype(dataset, "375_seg1"),
            environment=_environment(
                YPD,
                30.0,
                [
                    SmallMoleculePerturbation(
                        compound=Compound(
                            name="6-azauracil",
                            inchikey="SSPYSWLZOPCOLO-UHFFFAOYSA-N",
                            smiles="C1=NNC(=O)NC1=O",
                            pubchem_cid=68037,
                            chebi_id="CHEBI:53745",
                        ),
                        concentration=Concentration(
                            value=50.0, unit=ConcentrationUnit.ug_per_ml
                        ),
                    )
                ],
            ),
            phenotype=_residual_phenotype(1.5, "YPD;;3"),
        ).model_dump()
    )
    assert dataset[0]["reference"] == _reference(YPD, absolute=False)
    assert (
        dataset[0]["publication"]
        == Publication(doi=_DOI, doi_url=f"https://doi.org/{_DOI}").model_dump()
    )


def test_absolute_ph_temperature_and_carbon_source_conditions(
    dataset: m.Bloom2019Dataset,
) -> None:
    """375_seg2 (records 38..75): YPD;;1 (k = 34) is an absolute colony size 135.5 with the
    absolute YPD reference; YNB;ph3;1 (k = 32) is a pH 3 physical perturbation on the YNB
    plate with the YNB residual reference; YPD;15;1 (k = 35) puts 15 C on the environment
    with an empty gap list (the temperature is the sourced one); Galactose;;1 (k = 13) is
    the YP + 2% galactose medium with no perturbation.
    """
    genotype = _genotype(dataset, "375_seg2")
    assert (
        dataset[38 + 34]["experiment"]
        == SegregantGrowthExperiment(
            dataset_name="Bloom2019Dataset",
            genotype=genotype,
            environment=_environment(YPD, 30.0, []),
            phenotype=_absolute_phenotype(135.5),
        ).model_dump()
    )
    assert dataset[38 + 34]["reference"] == _reference(YPD, absolute=True)
    assert (
        dataset[38 + 32]["experiment"]
        == SegregantGrowthExperiment(
            dataset_name="Bloom2019Dataset",
            genotype=genotype,
            environment=_environment(
                YNB_GLUCOSE_SOLID,
                30.0,
                [
                    EnvironmentPhysicalPerturbation(
                        factor=PhysicalFactor.ph,
                        magnitude=Concentration(value=3.0, unit=ConcentrationUnit.ph),
                    )
                ],
            ),
            phenotype=_residual_phenotype(133.5, "YNB;;1"),
        ).model_dump()
    )
    assert dataset[38 + 32]["reference"] == _reference(
        YNB_GLUCOSE_SOLID, absolute=False
    )
    assert (
        dataset[38 + 35]["experiment"]["environment"]
        == _environment(YPD, 15.0, []).model_dump()
    )
    assert (
        dataset[38 + 13]["experiment"]["environment"]
        == _environment(YP_GALACTOSE, 30.0, []).model_dump()
    )


def test_unsourced_30c_plates_carry_no_provenance_gap(
    dataset: m.Bloom2019Dataset,
) -> None:
    """Finding: the module docstring (lines 57-59) says the 30 C of every non-temperature
    plate is a documented representative "with a ProvenanceGap", but ``_environment``
    (lines 861-873) stores the value with ``provenance_gaps == []`` for every plate, and
    ``ConditionSpec.temperature_sourced`` is never read by the build. Caffeine;15mM;2
    (k = 2, unsourced) and YPD;37;1 (k = 36, sourced) both store an empty gap list.
    """
    assert dataset[2]["experiment"]["environment"]["provenance_gaps"] == []
    assert (
        dataset[36]["experiment"]["environment"]["temperature"]
        == Temperature(value=37.0).model_dump()
    )
    assert dataset[36]["experiment"]["environment"]["provenance_gaps"] == []
    assert dataset.conditions["Caffeine;15mM;2"].temperature_sourced is False
    assert dataset.conditions["YPD;37;1"].temperature_sourced is True


def test_cross_a_genotype_pins_the_rm_parent_to_its_assembly_member(
    dataset: m.Bloom2019Dataset,
) -> None:
    assert (
        dataset[76]["experiment"]["genotype"]
        == _genotype(dataset, "A_seg1").model_dump()
    )
    assert (
        dataset[38]["experiment"]["genotype"]
        == _genotype(dataset, "375_seg2").model_dump()
    )


def test_reference_index_has_four_entries_first_seen_at_k_0_31_32_34(
    dataset: m.Bloom2019Dataset,
) -> None:
    """Residual/YPD-control (34 columns per segregant), absolute YNB;;1, residual/YNB
    control (YNB;ph3, YNB;ph8), absolute YPD;;1.
    """
    index = json.loads(
        (
            Path(dataset.root) / "preprocess" / "experiment_reference_index.json"
        ).read_text()
    )
    ynb_residual = {32, 33}
    absolute = {31, 34}
    assert [entry["member_indices"] for entry in index] == [
        [
            s * 38 + k
            for s in range(3)
            for k in range(38)
            if k not in ynb_residual | absolute
        ],
        [s * 38 + 31 for s in range(3)],
        [s * 38 + k for s in range(3) for k in sorted(ynb_residual)],
        [s * 38 + 34 for s in range(3)],
    ]
    assert [entry["reference"] for entry in index] == [
        _reference(YPD, absolute=False),
        _reference(YNB_GLUCOSE_SOLID, absolute=True),
        _reference(YNB_GLUCOSE_SOLID, absolute=False),
        _reference(YPD, absolute=True),
    ]


def test_side_files_gene_set_block_counts_and_segregant_csv(
    dataset: m.Bloom2019Dataset,
) -> None:
    """The gene set is the three stub genes whose span overlaps a haplotype block;
    ``block_counts.json`` lists the per-segregant block counts by cross; ``data.csv`` is
    the sorted segregant ids; the build manifest names the root slug and the loader.
    """
    preprocess = Path(dataset.root) / "preprocess"
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "YAL001C",
        "YBL001C",
        "YDL001W",
    ]
    assert json.loads((preprocess / "block_counts.json").read_text()) == {
        "375": [4, 3],
        "A": [2],
    }
    assert (preprocess / "data.csv").read_text() == (
        "segregant_id\n375_seg1\n375_seg2\nA_seg1\n"
    )
    manifest = json.loads((preprocess / "build_manifest.json").read_text())
    assert manifest["dataset_name"] == "bloom2019"
    assert manifest["loader_class"] == "Bloom2019Dataset"
    assert {"SegregantGrowthExperiment", "SegregantGenotype", "HaplotypeBlock"} <= set(
        manifest["closure"]
    )
    assert dataset.experiment_class is SegregantGrowthExperiment
    assert dataset.reference_class is SegregantGrowthExperimentReference
    assert dataset.raw_file_names == [
        "phenotypes.tsv.gz",
        "cross_genotypes_README",
        "elife-49212-fig1-data1-v2.xls",
        "genotype_375.tsv.gz",
        "genotype_A.tsv.gz",
    ]
    with pytest.raises(
        NotImplementedError, match=r"Bloom2019Dataset builds its records in process\(\)"
    ):
        dataset.create_experiment()
    with pytest.raises(
        NotImplementedError,
        match=(
            "a SegregantGenotype carries no gene perturbations; "
            "Bloom2019Dataset.compute_gene_set returns the S288C genes its haplotype "
            "blocks span"
        ),
    ):
        m.Bloom2019Dataset.extract_systematic_gene_names({})
    frame = pd.DataFrame({"untouched": [True]})
    assert dataset.preprocess_raw(frame) is frame


def test_gene_set_is_recomputed_from_the_raw_marker_headers_on_a_built_store(
    dataset: m.Bloom2019Dataset,
) -> None:
    """A second instance on the built root skips ``process()`` and holds no ``_markers``;
    ``compute_gene_set`` then re-reads the sorted marker names from ``raw/`` and yields
    the same three genes.
    """
    dataset.close_lmdb()
    second = m.Bloom2019Dataset(root=dataset.root, genome=_genome())
    assert not hasattr(second, "_markers")
    assert second.compute_gene_set() == {"YAL001C", "YBL001C", "YDL001W"}
    assert second._marker_spans() == {
        "chrI": [(100, 300), (150, 150)],
        "chrII": [(50, 60)],
        "chrIII": [(10, 10)],
        "chrIV": [(500, 500)],
    }


def test_condition_table_count_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m, "N_CONDITIONS", 37)
    with pytest.raises(ValueError, match="expected 37 conditions, built 38"):
        m.build_conditions()


def test_one_mosaic_is_interned_per_segregant_not_per_condition(
    dataset: m.Bloom2019Dataset,
) -> None:
    """The raw LMDB record stores the genotype as a ``{"$ref", "name"}`` pointer named
    by the segregant id: records 0..37 share 375_seg1's pointer, record 38 points at
    375_seg2's, so three genotypes are interned for 114 records.
    """
    dataset.close_lmdb()
    env = lmdb.open(
        str(Path(dataset.processed_dir) / "lmdb"), readonly=True, lock=False
    )
    with env.begin() as txn:
        values = [txn.get(f"{i}".encode()) for i in range(39)]
    env.close()
    assert all(value is not None for value in values)
    raw = [pickle.loads(value) for value in values if value is not None]
    pointers = {r["experiment"]["genotype"]["name"] for r in raw[:38]}
    assert pointers == {"375_seg1"}
    assert len({r["experiment"]["genotype"]["$ref"] for r in raw[:38]}) == 1
    assert raw[38]["experiment"]["genotype"]["name"] == "375_seg2"
    assert (
        raw[38]["experiment"]["genotype"]["$ref"]
        != raw[0]["experiment"]["genotype"]["$ref"]
    )
    assert raw[0]["experiment"]["environment"]["name"] == YPD.name
    assert raw[0]["reference"]["name"] == "Bloom2019Dataset"


def test_gene_set_needs_a_genome_and_at_least_one_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``compute_gene_set`` runs inside ``post_process`` after the LMDB is written: with
    no genome it raises naming the requirement; with a genome whose genes miss every
    block it raises ``no S288C gene overlaps any haplotype block``.
    """

    class _NoOverlap(_StubGenome):
        gene_set = ["YAL002W", "YCL001W"]

    with pytest.raises(
        RuntimeError, match="Bloom2019Dataset requires a genome to compute its gene set"
    ):
        _build(tmp_path, monkeypatch, slug="none", genome=None)
    with pytest.raises(ValueError, match="no S288C gene overlaps any haplotype block"):
        _build(tmp_path / "x", monkeypatch, slug="miss", genome=_NoOverlap())


def test_genomes_tier_pin_and_member_index_gate_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tier manifest pinning the tarball at another digest raises with both digests; a
    member index without ADR raises when the M22 parent is built.
    """
    with pytest.raises(
        RuntimeError,
        match=(
            f"genomes tier pins 1011Assemblies.tar.gz at {'1' * 64}; the stored records "
            f"pin {_TAR_SHA}"
        ),
    ):
        _build(tmp_path, monkeypatch, slug="pin", tar_sha="1" * 64)
    with pytest.raises(
        RuntimeError,
        match="parent M22: Peter id ADR is not in the assembly member index",
    ):
        _build(
            tmp_path / "x",
            monkeypatch,
            slug="index",
            index_lines="AAA\t1011Assemblies/AAA.re.fa\n",
        )


def test_readme_and_xls_join_reject_a_mismatched_cross_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A README row with two fields raises with the line; a README missing cross A
    raises with both sets; an xls whose parents are not the README pair raises naming
    the cross, the cells and the pair; an xls missing a cross raises.
    """
    with pytest.raises(
        ValueError, match=r"unexpected README table line: '375\\tBYa\\t'"
    ):
        _build(
            tmp_path / "a",
            monkeypatch,
            readme_rows=[("375", "BYa"), ("A", "BYa", "RMx")],
        )
    with pytest.raises(
        ValueError, match=r"README crosses \['375'\] != expected \['375', 'A'\]"
    ):
        _build(tmp_path / "b", monkeypatch, readme_rows=[("375", "BYa", "M22")])
    rows = list(_XLS_ROWS)
    rows[0] = ("BYxM22", _M22_GENOTYPE, _RM_GENOTYPE, "ADR", "375", 2)
    with pytest.raises(
        ValueError,
        match=(
            r"cross 375: xls parents \['M22 MATalpha ho::HygMX', "
            r"'RM MATalpha ho::HygMX'\] do not match README BYa/M22"
        ),
    ):
        _build(tmp_path / "c", monkeypatch, xls_rows=rows)
    with pytest.raises(
        ValueError, match="xls cross sheet does not list every README cross"
    ):
        _build(tmp_path / "d", monkeypatch, xls_rows=_XLS_ROWS[:1])


def test_matrix_and_phenotype_release_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A call of 3 raises ``calls outside {1, 2}``; a third row in cross 375 raises the
    segregant count; an unknown phenotype column, a missing served column, a fourth
    phenotype row, a segregant id in two crosses and a NaN cell each raise their message.
    """
    header, rows = _MATRICES["375"]
    with pytest.raises(ValueError, match=r"genotype_375: calls outside \{1, 2\}"):
        _build(
            tmp_path / "a",
            monkeypatch,
            matrices={
                **_MATRICES,
                "375": (header, [("375_seg1", [1, 2, 3, 2, 2, 1]), rows[1]]),
            },
        )
    with pytest.raises(ValueError, match="genotype_375: 3 segregants, expected 2"):
        _build(
            tmp_path / "b",
            monkeypatch,
            matrices={
                **_MATRICES,
                "375": (header, [*rows, ("375_seg3", [1, 1, 1, 1, 1, 1])]),
            },
        )
    with pytest.raises(
        ValueError,
        match=r"phenotype columns not in the condition table: \['Bogus;;1'\]",
    ):
        _build(
            tmp_path / "c",
            monkeypatch,
            phenotype_columns=[*_PHENOTYPE_COLUMNS, "Bogus;;1"],
        )
    with pytest.raises(
        ValueError,
        match=r"condition table columns absent from the release: \['Zeocin;25ug/mL;3'\]",
    ):
        _build(tmp_path / "d", monkeypatch, phenotype_columns=_PHENOTYPE_COLUMNS[:-2])
    with pytest.raises(ValueError, match="4 phenotype rows, expected 3"):
        _build(
            tmp_path / "e",
            monkeypatch,
            phenotype_rows=_phenotype_rows([*_SEGREGANTS, "extra"], _PHENOTYPE_COLUMNS),
        )
    with pytest.raises(
        ValueError, match="segregant id 375_seg1 appears in two crosses"
    ):
        _build(
            tmp_path / "f",
            monkeypatch,
            matrices={**_MATRICES, "A": (_MATRICES["A"][0], [("375_seg1", [1, 2])])},
        )
    nan_rows = _phenotype_rows(_SEGREGANTS, _PHENOTYPE_COLUMNS)
    nan_rows[0][1 + 3] = "NA"  # 375_seg1; Caffeine;15mM;2 is k = 2, written position 3
    with pytest.raises(
        ValueError, match="375_seg1/Caffeine;15mM;2: NaN in the release"
    ):
        _build(tmp_path / "g", monkeypatch, phenotype_rows=nan_rows)


def test_record_count_oracle_raises_after_the_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_release(monkeypatch)
    monkeypatch.setattr(m, "EXPECTED_RECORDS", 100)
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    root = tmp_path / "count"
    _write_data_files(root / "raw")
    _write_mirror_manifest(data_root, _raw_files(root / "raw"))
    _write_genomes_tier(data_root)
    with pytest.raises(ValueError, match="wrote 114 records, expected 100"):
        m.Bloom2019Dataset(root=str(root), genome=_genome())
    assert (root / "processed" / "lmdb").is_dir()


def test_download_symlinks_every_raw_file_from_the_verified_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``raw/`` empty, ``download()`` links the five data files from
    ``$DATA_ROOT/torchcell-raw/<key>/data/`` after checking each digest against the
    manifest: a file absent from the mirror raises naming it, a manifest digest that
    disagrees raises with both digests, and a clean mirror links all five and builds.
    """
    _patch_release(monkeypatch)
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    _write_genomes_tier(data_root)
    staged = tmp_path / "staged"
    _write_data_files(staged)
    mirror = _write_mirror_manifest(data_root, _raw_files(staged))
    with pytest.raises(
        RuntimeError,
        match=f"required raw artifact missing from mirror: {mirror / 'data' / 'phenotypes.tsv.gz'}",
    ):
        m.Bloom2019Dataset(root=str(tmp_path / "a"), genome=_genome())
    (mirror / "data").mkdir()
    for path in staged.iterdir():
        (mirror / "data" / path.name).write_bytes(path.read_bytes())
    _write_mirror_manifest(
        data_root,
        _raw_files(mirror / "data"),
        sha_override={"data/cross_genotypes_README": "2" * 64},
    )
    readme_sha = _sha256(mirror / "data" / "cross_genotypes_README")
    with pytest.raises(
        RuntimeError,
        match=f"cross_genotypes_README sha256 mismatch: got {readme_sha}, expected {'2' * 64}",
    ):
        m.Bloom2019Dataset(root=str(tmp_path / "b"), genome=_genome())
    _write_mirror_manifest(data_root, _raw_files(mirror / "data"))
    raw = tmp_path / "c" / "raw"
    raw.mkdir(parents=True)
    (raw / "cross_genotypes_README").write_bytes(
        (mirror / "data" / "cross_genotypes_README").read_bytes()
    )
    dataset = m.Bloom2019Dataset(root=str(tmp_path / "c"), genome=_genome())
    assert not (raw / "cross_genotypes_README").is_symlink()
    for name in dataset.raw_file_names:
        if name == "cross_genotypes_README":
            continue
        link = raw / name
        assert link.is_symlink() and os.readlink(link) == str(mirror / "data" / name)
    assert len(dataset) == 114
    assert dataset[0]["experiment"]["genotype"]["marker_matrix_sha256"] == _sha256(
        mirror / "data" / "genotype_375.tsv.gz"
    )


def test_deposit_raw_mirror_records_every_file_with_its_retrieval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deposit copies the zip members, the xls, the paper files and the four R files
    and writes a manifest: zip members carry the Dropbox URL with the container digest,
    the eLife files their CDN URLs, the R files the raw.githubusercontent URL at the
    pinned commit; the PDF is the one ``paper_pdf`` role. An existing copy with another
    digest is refused.
    """
    _patch_release(monkeypatch)
    members = tmp_path / "members"
    _write_data_files(members)
    xls = members / "elife-49212-fig1-data1-v2.xls"
    xml = tmp_path / "elife-49212-v2.xml"
    xml.write_text("<article/>")
    pdf = tmp_path / "elife-49212-v2.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")
    code = tmp_path / "code"
    code.mkdir()
    for name in ("mapping.R", "mapping_fx.R", "process_images.R", "rr_fx.R"):
        (code / name).write_text(f"# {name}\n")
    data_root = tmp_path / "data_root"
    root = m.deposit_raw_mirror(
        zip_members_dir=members,
        xls_path=xls,
        xml_path=xml,
        pdf_path=pdf,
        code_dir=code,
        retrieved_at_data="2026-09-01",
        retrieved_at_paper="2026-09-02",
        data_root=str(data_root),
    )
    assert root == data_root / "torchcell-raw" / _CITATION_KEY
    manifest = m.load_manifest(str(data_root))
    assert manifest.doi == _DOI
    assert manifest.si_expected == [
        "Figure 1 source data 1 (elife-49212-fig1-data1-v2.xls)"
    ]
    by_path = {record.path: record for record in manifest.files}
    assert sorted(by_path) == [
        "code/mapping.R",
        "code/mapping_fx.R",
        "code/process_images.R",
        "code/rr_fx.R",
        "data/cross_genotypes_README",
        "data/elife-49212-fig1-data1-v2.xls",
        "data/genotype_375.tsv.gz",
        "data/genotype_A.tsv.gz",
        "data/phenotypes.tsv.gz",
        "paper/elife-49212-v2.pdf",
        "paper/elife-49212-v2.xml",
    ]
    for record in manifest.files:
        assert record.sha256 == _sha256(root / record.path)
        assert record.bytes == (root / record.path).stat().st_size
        assert (
            record.retrieval is not None
            and record.retrieval.method.value == "direct_url"
        )
    member = by_path["data/genotype_A.tsv.gz"]
    assert member.retrieval is not None
    assert member.retrieval.retriever == "torchcell.literature.retrieve.zip_member"
    assert member.retrieval.params == {
        "url": m.DROPBOX_URL,
        "member": "genotype_A.tsv.gz",
        "container_sha256": m.DROPBOX_ZIP_SHA256,
    }
    assert member.retrieval.retrieved_at == "2026-09-01"
    assert by_path["paper/elife-49212-v2.pdf"].role == "paper_pdf"
    assert by_path["paper/elife-49212-v2.xml"].role == "raw_data"
    assert by_path["paper/elife-49212-v2.xml"].retrieval is not None
    assert by_path["paper/elife-49212-v2.xml"].retrieval.retrieved_at == "2026-09-02"
    r_file = by_path["code/rr_fx.R"]
    assert r_file.retrieval is not None
    assert r_file.retrieval.source_url == (
        "https://raw.githubusercontent.com/joshsbloom/yeast-16-parents/"
        "c913c9ae7fd237329f639de02e7ec511b048730f/phenotyping/code/rr_fx.R"
    )
    assert r_file.retrieval.params["path"] == "phenotyping/code/rr_fx.R"
    assert m.manifest_sha256(manifest, "data/phenotypes.tsv.gz") == _sha256(
        members / "phenotypes.tsv.gz"
    )
    with pytest.raises(
        KeyError, match="data/missing is not in the raw-mirror manifest"
    ):
        m.manifest_sha256(manifest, "data/missing")
    (code / "rr_fx.R").write_text("# changed\n")
    with pytest.raises(
        RuntimeError,
        match=f"{root / 'code' / 'rr_fx.R'} exists with a different sha256; refusing",
    ):
        m.deposit_raw_mirror(
            zip_members_dir=members,
            xls_path=xls,
            xml_path=xml,
            pdf_path=pdf,
            code_dir=code,
            retrieved_at_data="2026-09-01",
            retrieved_at_paper="2026-09-02",
            data_root=str(data_root),
        )
