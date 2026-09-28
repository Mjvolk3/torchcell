# tests/torchcell/datasets/scerevisiae/test_hillenmeyer2008_synthetic.py
# [[tests.torchcell.datasets.scerevisiae.test_hillenmeyer2008_synthetic]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_hillenmeyer2008_synthetic.py
"""Hillenmeyer 2008 HET and HOM loaders built end to end on synthetic FitDb matrices.

``process()`` reads, besides ``raw/``: the S288C gene universe from the genomes tier
(``resolve(SGD_S288C_R64, ...)`` on the ORF and RNA FASTAs), the raw-mirror
``manifest.json`` (the matrix sha256 is re-checked at build time), and the injected
genome's ``resolve_gene_name``. ``DATA_ROOT`` is pointed into ``tmp_path``; the genomes
tier is written with a real ``GenomeManifest`` (no monkeypatch of ``resolve``, so its
sha256 check stays on the path) and the raw mirror is deposited with the loader's own
``deposit_raw_mirror`` (``verify_source=False``, no network). The dataset root starts
empty, so ``download()`` symlinks the three mirror files and ``process()`` runs once.
The resolver is a dict-backed stub returning real ``GeneNameResolution`` objects.

HET matrix columns (filename: condition, dose, generations -> control set): A1 and A2
benomyl 6.9 uM 20gen -> CS1 (one group of two arrays), A3 benomyl 6.9 uM -5gen -> CS2 (a
second group), A4 amphotericin (no structure identifier: dropped), A5 ``37c, 45c``
(heat-shock cycle: dropped). CS1 has 3 control arrays, CS2 has 2.

HET rows (A1..A5), ``-`` for a missing token:

    "YAL001C:chr00_1"  1.0   3.0   0.5  2.0  NA     suspicious batch chr00_1
    YAL001C:chr1_2     2.0   NA    ""   ""   ""     second construction of YAL001C
    YBR001C:chr3_9     0.25  0.25  NaN  null 1.0
    YOLD01W:chr5_1     4.0   (line ends)           RENAMED -> YBR002W (current)
    YDUB01C, YRET01W, YOLD02W                       dropped (see the drop test)

Per-array values average an ORF's construction rows first: YAL001C in CS1 is A1
(1.0 + 2.0)/2 = 1.5 and A2 3.0, so mean 2.25 and sample SD sqrt((0.75^2 + 0.75^2)/1) =
sqrt(1.125); in CS2 A3 0.5 alone (n = 1, no SD). YBR001C in CS1 is 0.25 twice, a sample
SD of exactly 0 that is stored as a typed gap. YBR002W is 4.0 alone. Records: 0
YAL001C/CS1, 1 YAL001C/CS2, 2 YBR001C/CS1, 3 YBR002W/CS1. Dropped cells: amphotericin
has YAL001C 2.0 (1 record), the heat-shock column YBR001C 1.0 (1 record).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

import pytest

from torchcell.datamodels.media import YPD_LIQUID
from torchcell.datamodels.schema import (
    AssayType,
    Compound,
    Concentration,
    ConcentrationUnit,
    EngineeredCopyNumberPerturbation,
    Environment,
    EnvironmentPerturbationType,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Genotype,
    KanMxDeletionPerturbation,
    MeasurementType,
    Publication,
    ReferenceGenome,
    SampleUnit,
    SmallMoleculePerturbation,
    UncertaintyType,
)
from torchcell.datasets.scerevisiae import hillenmeyer2008 as m
from torchcell.literature.manifest import ArtifactRecord
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    GenomeIntegrityError,
    GenomeManifest,
)
from torchcell.sequence.genome.scerevisiae.s288c import (
    GeneNameResolution,
    GeneNameStatus,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

_HET = "HetHillenmeyer2008Dataset"
_HOM = "HomHillenmeyer2008Dataset"
_CS1 = "het_01::old scanner::20::tag3::YPD::dmso::0"
_CS2 = "het_02::old scanner::-5::tag3::YPD::dmso::0"
_HOM_CS = "hom_01::old scanner::20::tag3::YPD::dmso::0"

_RESOLUTIONS: dict[str, tuple[GeneNameStatus, str]] = {
    "YAL001C": (GeneNameStatus.CURRENT, "YAL001C"),
    "YBR001C": (GeneNameStatus.CURRENT, "YBR001C"),
    "YBR002W": (GeneNameStatus.CURRENT, "YBR002W"),
    "YOLD01W": (GeneNameStatus.RENAMED, "YBR002W"),
    "YDUB01C": (GeneNameStatus.NON_GENE_FEATURE, "YDUB01C"),
    "YRET01W": (GeneNameStatus.CURRENT, "YRET01W"),
    "YOLD02W": (GeneNameStatus.RENAMED, "YBR003W"),
    "YBR003W": (GeneNameStatus.RETIRED, "YBR003W"),
}


class _StubGenome:
    """Only ``resolve_gene_name`` is read from the injected genome."""

    def resolve_gene_name(self, name: str) -> GeneNameResolution:
        status, systematic = _RESOLUTIONS[name]
        return GeneNameResolution(
            input_name=name, status=status, systematic_name=systematic
        )


_ORF_FASTA = (
    ">YAL001C TFC3\nATG\n>YBR001C NTH2\nATG\n>YBR002W RER2\nATG\n>YBR003W COQ1\nATG\n"
)
_RNA_FASTA = ">YNCA0001W tA(UGC)A\nGGG\n"

_HET_HEADER = [
    "Orf",
    "A1:benomyl:6.9:um::::20gen:het_01:old scanner",
    "A2:benomyl:6.9:um::::20gen:het_01:old scanner",
    "A3:benomyl:6.9:um::::-5gen:het_02:old scanner",
    "A4:amphotericin:10:um::::20gen:het_01:old scanner",
    "A5:37c, 45c::::::20gen:het_01:old scanner",
]
_HET_ROWS = [
    ['"YAL001C:chr00_1"', "1.0", "3.0", "0.5", "2.0", "NA"],
    ["YAL001C:chr1_2", "2.0", "NA", "", "", ""],
    ["YBR001C:chr3_9", "0.25", "0.25", "NaN", "null", "1.0"],
    ["YOLD01W:chr5_1", "4.0"],
    ["YDUB01C:chr1_1", "9", "9", "9", "9", "9"],
    ["YRET01W:chr1_1", "9", "9", "9", "9", "9"],
    ["YOLD02W:chr1_1", "9", "9", "9", "9", "9"],
]


def _tsv(rows: list[list[str]]) -> str:
    return "".join("\t".join(row) + "\n" for row in rows)


def _source_files() -> dict[str, str]:
    return {
        "het.ratio_result_nm.pub": _tsv([_HET_HEADER, *_HET_ROWS]),
        "hom.z_result_nm.pub": _tsv(
            [["Orf", "B1:benomyl:6.9:um::::20gen"], ["YAL001C:chr00_1", "1.5"]]
        ),
        "het.txt": _tsv(
            [
                ["filename", "condition", "control_set"],
                ["A1", "benomyl", _CS1],
                ["A2", "benomyl", _CS1],
                ["A3", "benomyl", _CS2],
                ["A4", "amphotericin", _CS1],
                ["A5", "37c, 45c", _CS1],
            ]
        ),
        "hom.txt": _tsv(
            [["filename", "condition", "control_set"], ["B1", "benomyl", _HOM_CS]]
        ),
        "het_controls.txt": _tsv(
            [
                ["control_set", "filename"],
                [_CS1, "C1"],
                [_CS1, "C2"],
                [_CS1, "C3"],
                [_CS2, "C4"],
                [_CS2, "C5"],
            ]
        ),
        "hom_controls.txt": _tsv(
            [["control_set", "filename"], [_HOM_CS, "D1"], [_HOM_CS, "D2"]]
        ),
        "README_CEL_files.txt": "key-file semantics\n",
    }


def _write_genomes_tier(data_root: Path) -> None:
    tier = data_root / "torchcell-genomes" / SGD_S288C_R64
    tier.mkdir(parents=True)
    records = []
    for name, text in (
        ("orf_coding_all_R64-4-1_20230830.fasta", _ORF_FASTA),
        ("rna_coding_R64-4-1_20230830.fasta", _RNA_FASTA),
    ):
        (tier / name).write_text(text)
        records.append(
            ArtifactRecord(
                path=name,
                role="sequence",
                bytes=len(text.encode()),
                sha256=hashlib.sha256(text.encode()).hexdigest(),
            )
        )
    manifest = GenomeManifest(
        assembly_set=SGD_S288C_R64,
        organism="Saccharomyces cerevisiae",
        strain_or_population="S288C",
        source="SGD",
        release="R64-4-1_20230830",
        files=records,
        provenance_complete=True,
        created_at="2026-09-27T00:00:00+00:00",
    )
    (tier / "manifest.json").write_text(manifest.model_dump_json(indent=2))


def _deposit(tmp_path: Path, data_root: Path) -> Path:
    source = tmp_path / "staging"
    source.mkdir()
    for name, text in _source_files().items():
        (source / name).write_text(text)
    return m.deposit_raw_mirror(
        source, data_root=str(data_root), retrieved_at="2026-09-27"
    )


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    _write_genomes_tier(data_root)
    _deposit(tmp_path, data_root)
    return data_root


@pytest.fixture
def het(tmp_path: Path, data_root: Path) -> m.HetHillenmeyer2008Dataset:
    return m.HetHillenmeyer2008Dataset(root=str(tmp_path / "het"), genome=_StubGenome())


_BENOMYL = SmallMoleculePerturbation(
    compound=Compound(
        name="benomyl",
        inchikey="RIOXQFHNBCKOKP-UHFFFAOYSA-N",
        smiles="CCCCNC(=O)N1C2=CC=CC=C2N=C1NC(=O)OC",
        pubchem_cid=28780,
        chebi_id="CHEBI:3015",
    ),
    concentration=Concentration(value=6.9, unit=ConcentrationUnit.micromolar),
)
_SOM = Provenance(
    source_uri="paper.md",
    citation_key="hillenmeyerChemicalGenomicPortrait2008",
    sha256="cf4759f00083de78dd953b12dd66d4360a2f645321305f768403f26b451c1df0",
)
_TEMPERATURE_GAP = ProvenanceGap(
    field="temperature",
    reason=ProvenanceGapReason.deferred_pending_source_review,
    looked_in=_SOM,
    resolve_with=Provenance(
        source_uri="paper.pdf",
        citation_key="pierceGenomewideAnalysisBarcoded2006",
        sha256="not-mirrored",
        method="S. E. Pierce et al., Nat Methods 3, 601 (Aug, 2006) -- ref (2) of the "
        "Hillenmeyer SOM, the declared source of the growth protocol",
    ),
    note="the SOM states no growth temperature for the non-temperature conditions "
    "and defers the whole pooled-growth protocol to Pierce 2006, which is not "
    "mirrored; 30 C is the community default but this paper never states it",
)


def _environment(
    perturbations: list[EnvironmentPerturbationType], generations: float
) -> Environment:
    return Environment(
        media=YPD_LIQUID,
        temperature=None,
        perturbations=perturbations,
        aerobicity="aerobic",
        duration_generations=generations,
        provenance_gaps=[_TEMPERATURE_GAP],
    )


def _het_genotype(orf: str) -> Genotype:
    return Genotype(
        perturbations=[
            EngineeredCopyNumberPerturbation(
                systematic_gene_name=orf,
                perturbed_gene_name=orf,
                copy_number=1,
                reference_copy_number=2,
                marker="KanMX",
            )
        ]
    )


def _phenotype(
    mean: float,
    n: int,
    sd: float | None,
    control_set: str,
    gaps: list[ProvenanceGap] | None = None,
    measurement: MeasurementType = MeasurementType.log2_ratio,
    units: str = m.UNITS_HET,
) -> EnvironmentResponsePhenotype:
    return EnvironmentResponsePhenotype(
        measurement_type=measurement,
        assay_type=AssayType.pooled_competitive_growth_barcode,
        environment_response=mean,
        n_samples=n,
        sample_unit=SampleUnit.biological_replicate,
        environment_response_uncertainty=sd,
        environment_response_uncertainty_type=(
            UncertaintyType.sample_sd if sd is not None else None
        ),
        screen_id=control_set,
        units=units,
        provenance_gaps=gaps or [],
    )


def _reference(
    dataset: str,
    control_set: str,
    n: int,
    generations: float,
    strain: str = "heterozygous diploid deletion collection (Giaever 2002)",
    measurement: MeasurementType = MeasurementType.log2_ratio,
) -> EnvironmentResponseExperimentReference:
    return EnvironmentResponseExperimentReference(
        dataset_name=dataset,
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain=strain, ploidy="diploid"
        ),
        environment_reference=_environment([], generations),
        phenotype_reference=EnvironmentResponsePhenotype(
            measurement_type=measurement,
            assay_type=AssayType.pooled_competitive_growth_barcode,
            environment_response=0.0,
            n_samples=n,
            sample_unit=SampleUnit.biological_replicate,
            screen_id=control_set,
            units=(
                f"no-drug control set {control_set!r}: {n} control arrays on the same "
                "deletion pool, generation count and scanner, run in YPD with a DMSO "
                "vehicle at concentration 0 as the release spells it. The score is 0 by "
                "construction -- it is the control mean this set's treatment arrays are "
                "scored against"
            ),
        ),
    )


_ZERO_SD_GAP = ProvenanceGap(
    field="environment_response_uncertainty",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=_SOM,
    note="the 2 arrays of this group print the identical score to the release's last "
    "decimal, so the replicate dispersion is below the released precision rather than "
    "measured to be zero",
)

_HET_EXPECTED = [
    ("YAL001C", _phenotype(2.25, 2, math.sqrt(1.125), _CS1), 20.0),
    ("YAL001C", _phenotype(0.5, 1, None, _CS2), 5.0),
    ("YBR001C", _phenotype(0.25, 2, None, _CS1, [_ZERO_SD_GAP]), 20.0),
    ("YBR002W", _phenotype(4.0, 1, None, _CS1), 20.0),
]
_REF_CS1 = _reference(_HET, _CS1, 3, 20.0)
_REF_CS2 = _reference(_HET, _CS2, 2, 5.0)


def test_het_records_average_constructions_then_arrays(
    het: m.HetHillenmeyer2008Dataset,
) -> None:
    """Four records in ORF-then-group order, each a heterozygous CNV in YPD with the
    temperature gap; the renamed row is stored under YBR002W.
    """
    assert len(het) == 4
    for i, (orf, phenotype, generations) in enumerate(_HET_EXPECTED):
        assert (
            het[i]["experiment"]
            == EnvironmentResponseExperiment(
                dataset_name=_HET,
                genotype=_het_genotype(orf),
                environment=_environment([_BENOMYL], generations),
                phenotype=phenotype,
            ).model_dump()
        )
    assert [het[i]["reference"] for i in range(4)] == [
        _REF_CS1.model_dump(),
        _REF_CS2.model_dump(),
        _REF_CS1.model_dump(),
        _REF_CS1.model_dump(),
    ]
    doi = "10.1126/science.1150021"
    assert (
        het[0]["publication"]
        == Publication(doi=doi, doi_url=f"https://doi.org/{doi}").model_dump()
    )


def test_het_side_files_gene_set_reference_index_and_manifest(
    het: m.HetHillenmeyer2008Dataset,
) -> None:
    preprocess = Path(het.root) / "preprocess"
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "YAL001C",
        "YBR001C",
        "YBR002W",
    ]
    index = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert sorted(
        (
            entry["member_indices"],
            entry["reference"]["phenotype_reference"]["screen_id"],
        )
        for entry in index
    ) == [([0, 2, 3], _CS1), ([1], _CS2)]
    manifest = json.loads((preprocess / "build_manifest.json").read_text())
    assert manifest["dataset_name"] == "het"
    assert manifest["loader_class"] == _HET
    assert sorted(manifest["closure"]) == [
        "AssayType",
        "ComponentDefinition",
        "Compound",
        "Concentration",
        "ConcentrationUnit",
        "DeletionPerturbation",
        "DoseBasis",
        "EngineeredCopyNumberPerturbation",
        "Environment",
        "EnvironmentPerturbation",
        "EnvironmentPhysicalPerturbation",
        "EnvironmentResponseExperiment",
        "EnvironmentResponseExperimentReference",
        "EnvironmentResponsePhenotype",
        "Experiment",
        "ExperimentReference",
        "GenePerturbation",
        "Genotype",
        "KanMxDeletionPerturbation",
        "MeasurementType",
        "Media",
        "MediaComponent",
        "MediaComponentRole",
        "ModelStrict",
        "Phenotype",
        "PhysicalFactor",
        "PresenceAbsencePerturbation",
        "ProvenanceGapMixin",
        "Publication",
        "ReferenceGenome",
        "ResponseCategory",
        "SampleUnit",
        "SmallMoleculePerturbation",
        "Solvent",
        "Temperature",
        "TemperatureUnit",
        "UncertaintyType",
    ]
    assert het.raw_file_names == [
        "het.ratio_result_nm.pub",
        "het.txt",
        "het_controls.txt",
    ]
    assert het.experiment_class is EnvironmentResponseExperiment
    assert het.reference_class is EnvironmentResponseExperimentReference
    with pytest.raises(
        NotImplementedError,
        match="Hillenmeyer2008 builds its records in process\\(\\); see _array_values",
    ):
        het.create_experiment()
    frame = {"untouched": True}
    assert het.preprocess_raw(frame) is frame


def test_het_drop_reports_count_columns_and_genes(
    het: m.HetHillenmeyer2008Dataset,
) -> None:
    """Each dropped column is its own environment with one array and one record not
    written; the three dropped row ids are the non-gene feature, a current name absent
    from the FASTA, and a rename onto a retired target.
    """
    preprocess = Path(het.root) / "preprocess"
    report = json.loads((preprocess / "dropped_records.json").read_text())
    assert report["dataset"] == _HET
    assert report["matrix"] == "het.ratio_result_nm.pub"
    assert report["kept_records"] == 4
    assert sorted(report["rules"]) == [
        "heat_shock_cycle_not_representable",
        "unidentifiable_agent",
        "unnamed_agent_dosed_into_a_media_swap",
    ]
    assert report["by_rule"] == {
        "unidentifiable_agent": {
            "n_environments": 1,
            "n_arrays": 1,
            "n_records_not_written": 1,
        },
        "heat_shock_cycle_not_representable": {
            "n_environments": 1,
            "n_arrays": 1,
            "n_records_not_written": 1,
        },
    }
    assert report["environments"] == [
        {
            "rule": "heat_shock_cycle_not_representable",
            "detail": "37c, 45c",
            "n_arrays": 1,
            "headers": [_HET_HEADER[5]],
            "n_records_not_written": 1,
        },
        {
            "rule": "unidentifiable_agent",
            "detail": "amphotericin",
            "n_arrays": 1,
            "headers": [_HET_HEADER[4]],
            "n_records_not_written": 1,
        },
    ]
    genes = json.loads((preprocess / "dropped_genes.json").read_text())
    assert genes["n_dropped"] == 3
    assert genes["dropped"] == {
        "YDUB01C": {
            "status": "non_gene_feature",
            "resolved_to": "YDUB01C",
            "in_sgd_fasta": "False",
        },
        "YOLD02W": {
            "status": "renamed",
            "resolved_to": "YBR003W",
            "in_sgd_fasta": "True",
        },
        "YRET01W": {
            "status": "current",
            "resolved_to": "YRET01W",
            "in_sgd_fasta": "False",
        },
    }


def test_het_batch_reports_flag_the_suspicious_construction(
    het: m.HetHillenmeyer2008Dataset,
) -> None:
    """YAL001C has two construction rows (chr00_1, chr1_2); chr00_1 is one of the ten
    SOM batches, so YAL001C is the one flagged ORF.
    """
    preprocess = Path(het.root) / "preprocess"
    batches = json.loads((preprocess / "strain_batches.json").read_text())
    assert (
        batches["n_orfs"],
        batches["n_rows"],
        batches["n_orfs_with_multiple_batches"],
    ) == (3, 4, 1)
    assert batches["batches_by_orf"] == {
        "YAL001C": ["chr00_1", "chr1_2"],
        "YBR001C": ["chr3_9"],
        "YBR002W": ["chr5_1"],
    }
    suspicious = json.loads((preprocess / "suspicious_batch_strains.json").read_text())
    assert suspicious["collection"] == "heterozygous"
    assert suspicious["n_flagged_orfs"] == 1
    assert suspicious["flagged_orfs"] == {"YAL001C": ["chr00_1"]}


def test_download_symlinks_the_three_mirror_files(
    tmp_path: Path, het: m.HetHillenmeyer2008Dataset
) -> None:
    mirror = tmp_path / "data_root" / "torchcell-raw" / m.CITATION_KEY / "data"
    raw = tmp_path / "het" / "raw"
    assert {name: os.readlink(raw / name) for name in het.raw_file_names} == {
        name: str(mirror / name) for name in het.raw_file_names
    }


def test_hom_record_is_a_kanmx_deletion_scored_as_a_z_score(
    tmp_path: Path, data_root: Path
) -> None:
    hom = m.HomHillenmeyer2008Dataset(root=str(tmp_path / "hom"), genome=_StubGenome())
    assert len(hom) == 1
    assert (
        hom[0]["experiment"]
        == EnvironmentResponseExperiment(
            dataset_name=_HOM,
            genotype=Genotype(
                perturbations=[
                    KanMxDeletionPerturbation(
                        systematic_gene_name="YAL001C", perturbed_gene_name="YAL001C"
                    )
                ]
            ),
            environment=_environment([_BENOMYL], 20.0),
            phenotype=_phenotype(
                1.5,
                1,
                None,
                _HOM_CS,
                measurement=MeasurementType.z_score,
                units=m.UNITS_HOM,
            ),
        ).model_dump()
    )
    assert (
        hom[0]["reference"]
        == _reference(
            _HOM,
            _HOM_CS,
            2,
            20.0,
            strain="homozygous diploid deletion collection (Giaever 2002)",
            measurement=MeasurementType.z_score,
        ).model_dump()
    )
    suspicious = json.loads(
        (tmp_path / "hom" / "preprocess" / "suspicious_batch_strains.json").read_text()
    )
    assert (suspicious["collection"], suspicious["flagged_orfs"]) == (
        "homozygous",
        {"YAL001C": ["chr00_1"]},
    )


def test_download_refuses_a_mirror_file_whose_bytes_changed(
    tmp_path: Path, data_root: Path
) -> None:
    matrix = data_root / "torchcell-raw" / m.CITATION_KEY / "data" / "het.txt"
    expected = hashlib.sha256(matrix.read_bytes()).hexdigest()
    matrix.write_text("tampered\n")
    got = hashlib.sha256(b"tampered\n").hexdigest()
    with pytest.raises(
        RuntimeError,
        match=re.escape(f"het.txt sha256 mismatch: got {got}, expected {expected}"),
    ):
        m.HetHillenmeyer2008Dataset(root=str(tmp_path / "het"), genome=_StubGenome())


def test_download_refuses_a_file_missing_from_the_mirror(
    tmp_path: Path, data_root: Path
) -> None:
    missing = data_root / "torchcell-raw" / m.CITATION_KEY / "data" / "het_controls.txt"
    missing.unlink()
    with pytest.raises(
        RuntimeError,
        match=re.escape(f"required raw artifact missing from mirror: {missing}"),
    ):
        m.HetHillenmeyer2008Dataset(root=str(tmp_path / "het"), genome=_StubGenome())


def test_a_tampered_genome_tier_is_refused_at_build_time(
    tmp_path: Path, data_root: Path
) -> None:
    """The S288C ORF FASTA in the genome tier gains a byte after its manifest was
    written; the build's gene-universe read (``resolve`` with sha256 verification)
    raises ``GenomeIntegrityError`` naming the file and both digests.
    """
    name = "orf_coding_all_R64-4-1_20230830.fasta"
    fasta = data_root / "torchcell-genomes" / SGD_S288C_R64 / name
    pinned = hashlib.sha256(fasta.read_bytes()).hexdigest()
    fasta.write_text(fasta.read_text() + "N")
    got = hashlib.sha256(fasta.read_bytes()).hexdigest()
    with pytest.raises(
        GenomeIntegrityError,
        match=re.escape(
            f"{SGD_S288C_R64}/{name}: sha256 {got} on disk, manifest pins {pinned}"
        ),
    ):
        m.HetHillenmeyer2008Dataset(root=str(tmp_path / "het"), genome=_StubGenome())


def test_process_rechecks_the_matrix_sha256_at_build_time(
    tmp_path: Path, data_root: Path
) -> None:
    """A raw matrix written directly (so ``download()`` is skipped) whose bytes differ
    from the manifest pin is refused by ``process()`` itself.
    """
    raw = tmp_path / "het" / "raw"
    raw.mkdir(parents=True)
    files = _source_files()
    for name in ("het.txt", "het_controls.txt"):
        (raw / name).write_text(files[name])
    (raw / "het.ratio_result_nm.pub").write_text(
        files["het.ratio_result_nm.pub"] + "\n"
    )
    got = hashlib.sha256((files["het.ratio_result_nm.pub"] + "\n").encode()).hexdigest()
    expected = hashlib.sha256(files["het.ratio_result_nm.pub"].encode()).hexdigest()
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"het.ratio_result_nm.pub sha256 mismatch at build time: got {got}, "
            f"expected {expected}"
        ),
    ):
        m.HetHillenmeyer2008Dataset(root=str(tmp_path / "het"), genome=_StubGenome())


def test_deposit_manifest_records_each_wayback_retrieval(data_root: Path) -> None:
    """Seven files under ``data/``, each pinned by its own digest with the archived URL
    as source and ``direct_url`` retrieval dated as passed. The URLs are written out
    literally: ``http://web.archive.org/web/<timestamp>id_/`` followed by the original
    chemogenomics.stanford.edu download path.
    """
    manifest = m.load_manifest(str(data_root))
    files = _source_files()
    archived = [
        (
            "het.ratio_result_nm.pub",
            "http://web.archive.org/web/20151207003548id_/"
            "http://chemogenomics.stanford.edu/supplements/global/download/data/het.ratio_result_nm.pub",
        ),
        (
            "hom.z_result_nm.pub",
            "http://web.archive.org/web/20151207003024id_/"
            "http://chemogenomics.stanford.edu/supplements/global/download/data/hom.z_result_nm.pub",
        ),
        (
            "het.txt",
            "http://web.archive.org/web/20151207063659id_/"
            "http://chemogenomics.stanford.edu/supplements/global/download/data/het.txt",
        ),
        (
            "hom.txt",
            "http://web.archive.org/web/20151207011015id_/"
            "http://chemogenomics.stanford.edu/supplements/global/download/data/hom.txt",
        ),
        (
            "het_controls.txt",
            "http://web.archive.org/web/20151207003758id_/"
            "http://chemogenomics.stanford.edu/supplements/global/download/data/het_controls.txt",
        ),
        (
            "hom_controls.txt",
            "http://web.archive.org/web/20151207004537id_/"
            "http://chemogenomics.stanford.edu/supplements/global/download/data/hom_controls.txt",
        ),
        (
            "README_CEL_files.txt",
            "http://web.archive.org/web/20151207004902id_/"
            "http://chemogenomics.stanford.edu/supplements/global/download/data/README_CEL_files.txt",
        ),
    ]
    assert [(f.path, f.sha256, f.source) for f in manifest.files] == [
        (f"data/{name}", hashlib.sha256(files[name].encode()).hexdigest(), url)
        for name, url in archived
    ]
    for record in manifest.files:
        assert record.retrieval is not None
        assert (record.retrieval.method.value, record.retrieval.retrieved_at) == (
            "direct_url",
            "2026-09-27",
        )
    with pytest.raises(
        KeyError, match="data/nope is not in the Hillenmeyer raw-mirror manifest"
    ):
        m.manifest_sha256(manifest, "data/nope")


def test_deposit_refuses_to_overwrite_a_mirror_file_with_other_bytes(
    tmp_path: Path, data_root: Path
) -> None:
    source = tmp_path / "staging2"
    source.mkdir()
    for name, text in _source_files().items():
        (source / name).write_text(text)
    (source / "het.txt").write_text("changed\n")
    dest = data_root / "torchcell-raw" / m.CITATION_KEY / "data" / "het.txt"
    with pytest.raises(
        RuntimeError,
        match=re.escape(f"{dest} exists with a different sha256; refusing"),
    ):
        m.deposit_raw_mirror(source, data_root=str(data_root))
    (source / "het.txt").write_text(_source_files()["het.txt"])
    (source / "README_CEL_files.txt").unlink()
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"required raw file missing from {source}: README_CEL_files.txt"
        ),
    ):
        m.deposit_raw_mirror(source, data_root=str(data_root))


def test_key_file_headers_are_checked(tmp_path: Path) -> None:
    keyfile = tmp_path / "k.txt"
    keyfile.write_text("file\tcond\tcs\n")
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"unexpected key-file header in {keyfile}: ['file', 'cond', 'cs']"
        ),
    ):
        m.read_control_set_map(keyfile)
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"unexpected controls-file header in {keyfile}: ['file', 'cond', 'cs']"
        ),
    ):
        m.read_control_set_sizes(keyfile)
