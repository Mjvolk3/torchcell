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
benomyl 6.9 uM 20gen -> CS1 (pool het_01; one group of two arrays), A3 benomyl 6.9 uM
-5gen -> CS2 (pool het_02; a second group), A4 amphotericin (no structure identifier:
dropped), A5 ``37c, 45c`` (heat-shock cycle: dropped), A6 colchicine whose key file says
colchiceine (compound conflict: dropped), A7 minimal media at 0gen (no exposure:
dropped). CS1 has 3 control arrays, CS2 has 2.

HET rows (A1..A7):

    "YAL001C:chr00_1"  1.0   3.0   0.5  2.0  NA  1.0  1.0   suspicious batch chr00_1
    YAL001C:chr1_2     2.0   NA    ""   ""   ""               second construction
    YBR001C:chr3_9     0.25  0.25  NaN  null 1.0
    YOLD01W:chr5_1     4.0   (line ends)                      RENAMED -> YBR002W, alone
    YOLD03W:chr6_1     1.0   NA                               RENAMED -> YBR001C (merge)
    YOR202W:chr15_3    1.0                                    HIS3, a BY4743 marker locus
    YDL227C:ctrl_1     5.0   5.0                              HO control strain: dropped
    YDUB01C, YRET01W, YOLD02W                                 dropped (see the drop test)

Each row is its own strain, so nothing averages across rows: YAL001C chr00_1 in CS1 is
A1 1.0 and A2 3.0, mean 2.0, sample SD sqrt(2); in CS2 A3 0.5 alone. YAL001C chr1_2 in
CS1 is 2.0 alone. YBR001C in CS1 is 0.25 twice, a sample SD of exactly 0 that is stored
as a typed gap. YBR002W (built against YOLD01W) is 4.0; YBR001C built against YOLD03W is
1.0 (merged: two source ORFs land on YBR001C); HIS3 is 1.0. Records: 0 YAL001C/chr00_1
CS1, 1 YAL001C/chr00_1 CS2, 2 YAL001C/chr1_2 CS1, 3 YBR001C/chr3_9 CS1, 4 YBR002W CS1,
5 YBR001C/YOLD03W CS1, 6 YOR202W CS1.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

import pytest

from torchcell.data import RawSha256MismatchError
from torchcell.data.experiment_dataset import verify_raw_files
from torchcell.datamodels.media import YPD_LIQUID
from torchcell.datamodels.schema import (
    BarcodedKanMxDeletionPerturbation,
    Compound,
    Concentration,
    ConcentrationUnit,
    CultureEnvironment,
    Genotype,
    HeterozygousDeletionPerturbation,
    MeasurementType,
    OrfHistoryRelation,
    PreCultureSource,
    Publication,
    SmallMoleculePerturbation,
    StrainConstruction,
    StrainEnvironmentResponseExperiment,
    StrainEnvironmentResponseExperimentReference,
    heterozygous_deletion_functional_copies,
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
    "YOLD03W": (GeneNameStatus.RENAMED, "YBR001C"),
    "YOR202W": (GeneNameStatus.CURRENT, "YOR202W"),
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
    ">YOR202W HIS3\nATG\n"
)
_RNA_FASTA = ">YNCA0001W tA(UGC)A\nGGG\n"

_HET_HEADER = [
    "Orf",
    "A1:benomyl:6.9:um::::20gen:het_01:old scanner",
    "A2:benomyl:6.9:um::::20gen:het_01:old scanner",
    "A3:benomyl:6.9:um::::-5gen:het_02:old scanner",
    "A4:amphotericin:10:um::::20gen:het_01:old scanner",
    "A5:37c, 45c::::::20gen:het_01:old scanner",
    "A6:colchicine:250:um::::20gen:het_01:old scanner",
    "A7:minimal media::::::0gen:het_02:old scanner",
]
_HET_ROWS = [
    ['"YAL001C:chr00_1"', "1.0", "3.0", "0.5", "2.0", "NA", "1.0", "1.0"],
    ["YAL001C:chr1_2", "2.0", "NA", "", "", ""],
    ["YBR001C:chr3_9", "0.25", "0.25", "NaN", "null", "1.0"],
    ["YOLD01W:chr5_1", "4.0"],
    ["YOLD03W:chr6_1", "1.0", "NA"],
    ["YOR202W:chr15_3", "1.0"],
    ["YDL227C:ctrl_1", "5.0", "5.0"],
    ["YDUB01C:chr1_1", "9", "9", "9", "9", "9"],
    ["YRET01W:chr1_1", "9", "9", "9", "9", "9"],
    ["YOLD02W:chr1_1", "9", "9", "9", "9", "9"],
]
_CS0 = "het_02::old scanner::0::tag3::YPD::dmso::0"


def _tsv(rows: list[list[str]]) -> str:
    return "".join("\t".join(row) + "\n" for row in rows)


def _source_files() -> dict[str, str]:
    return {
        "het.ratio_result_nm.pub": _tsv([_HET_HEADER, *_HET_ROWS]),
        "hom.z_result_nm.pub": _tsv(
            [
                ["Orf", "B1:benomyl:6.9:um::::20gen"],
                ["YAL001C:chr00_1", "1.5"],
                ["YOR153W:chr15_2", "2.5"],
            ]
        ),
        "het.txt": _tsv(
            [
                ["filename", "condition", "control_set"],
                ["A1", "benomyl", _CS1],
                ["A2", "benomyl", _CS1],
                ["A3", "benomyl", _CS2],
                ["A4", "amphotericin", _CS1],
                ["A5", "37c, 45c", _CS1],
                ["A6", "colchiceine", _CS1],
                ["A7", "no drug minimal media", _CS0],
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
                [_CS0, "C6"],
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


_BENOMYL_COMPOUND = Compound(
    name="benomyl",
    inchikey="RIOXQFHNBCKOKP-UHFFFAOYSA-N",
    smiles="CCCCNC(=O)N1C2=CC=CC=C2N=C1NC(=O)OC",
    pubchem_cid=28780,
    chebi_id="CHEBI:3015",
)


def _experiment(
    dataset: m._Hillenmeyer2008Base, i: int
) -> StrainEnvironmentResponseExperiment:
    return StrainEnvironmentResponseExperiment.model_validate(dataset[i]["experiment"])


def _genotype(experiment: StrainEnvironmentResponseExperiment) -> Genotype:
    """The record's single genotype (Hillenmeyer never emits a genotype list)."""
    genotype = experiment.genotype
    assert isinstance(genotype, Genotype)
    return genotype


def _reference(
    dataset: m._Hillenmeyer2008Base, i: int
) -> StrainEnvironmentResponseExperimentReference:
    return StrainEnvironmentResponseExperimentReference.model_validate(
        dataset[i]["reference"]
    )


#: (row id, current gene, source ORF if renamed, mean, n, sd, control set, pool)
_HET_EXPECTED = [
    ("YAL001C", "chr00_1", None, 2.0, 2, math.sqrt(2.0), _CS1, "het_01"),
    ("YAL001C", "chr00_1", None, 0.5, 1, None, _CS2, "het_02"),
    ("YAL001C", "chr1_2", None, 2.0, 1, None, _CS1, "het_01"),
    ("YBR001C", "chr3_9", None, 0.25, 2, None, _CS1, "het_01"),
    ("YBR002W", "chr5_1", "YOLD01W", 4.0, 1, None, _CS1, "het_01"),
    ("YBR001C", "chr6_1", "YOLD03W", 1.0, 1, None, _CS1, "het_01"),
    ("YOR202W", "chr15_3", None, 1.0, 1, None, _CS1, "het_01"),
]


def test_het_records_are_one_per_constructed_strain_and_control_set(
    het: m.HetHillenmeyer2008Dataset,
) -> None:
    """Seven records in row-then-group order; two rows of YAL001C (chr00_1, chr1_2) and
    the two strains on YBR001C never average (#505 G3, n_samples counts arrays of ONE
    construction).
    """
    assert len(het) == 7
    for i, (orf, batch, source, mean, n, sd, cs, pool) in enumerate(_HET_EXPECTED):
        experiment = _experiment(het, i)
        assert experiment.experiment_type == "strain_environment_response"
        (pert,) = _genotype(experiment).perturbations
        assert isinstance(pert, HeterozygousDeletionPerturbation)
        assert (pert.systematic_gene_name, pert.cassette, pert.collection) == (
            orf,
            "kanMX4",
            pool,
        )
        assert pert.construction == StrainConstruction(batch=batch)
        if source is None:
            assert pert.constructed_orf is None
        else:
            assert pert.constructed_orf is not None
            assert pert.constructed_orf.source_systematic_name == source
        phenotype = experiment.phenotype
        assert (phenotype.screen_id, phenotype.n_samples) == (cs, n)
        assert phenotype.environment_response == pytest.approx(mean)
        if sd is None:
            assert phenotype.environment_response_uncertainty is None
        else:
            assert phenotype.environment_response_uncertainty == pytest.approx(sd)
        assert phenotype.units == m.UNITS_HET
    # the zero-SD group is a typed gap, not a dispersion of 0
    assert _experiment(het, 3).phenotype.gapped_fields() == {
        "environment_response_uncertainty"
    }
    # a merge (two source ORFs on YBR001C) vs a lone rename (YOLD01W -> YBR002W)
    merged = _genotype(_experiment(het, 5)).perturbations[0]
    lone = _genotype(_experiment(het, 4)).perturbations[0]
    assert isinstance(merged, HeterozygousDeletionPerturbation)
    assert isinstance(lone, HeterozygousDeletionPerturbation)
    assert merged.constructed_orf is not None and lone.constructed_orf is not None
    assert merged.constructed_orf.relation is OrfHistoryRelation.merged
    assert lone.constructed_orf.relation is None
    assert lone.constructed_orf.gapped_fields() == {"relation", "deleted_span"}
    # G2: HIS3 is null in BY4743 before and after the deletion
    his3 = _genotype(_experiment(het, 6)).perturbations[0]
    assert isinstance(his3, HeterozygousDeletionPerturbation)
    assert his3.gapped_fields() == {"barcode", "downtag_barcode", "replaced_allele"}
    background = _reference(het, 6).genome_reference.background
    assert heterozygous_deletion_functional_copies(background, his3) == 0
    doi = "10.1126/science.1150021"
    assert (
        het[0]["publication"]
        == Publication(doi=doi, doi_url=f"https://doi.org/{doi}").model_dump()
    )


def test_het_environment_carries_the_pre_culture_its_generation_sign_implies(
    het: m.HetHillenmeyer2008Dataset,
) -> None:
    """20gen and -5gen of one drug: same compound, different pre-culture (#505 E3)."""
    grown, frozen = _experiment(het, 0).environment, _experiment(het, 1).environment
    for environment, generations in ((grown, 20.0), (frozen, 5.0)):
        assert isinstance(environment, CultureEnvironment)
        assert environment.media == YPD_LIQUID
        assert environment.duration_generations == generations
        assert environment.gapped_fields() == {"temperature", "culture_format"}
        (drug,) = environment.perturbations
        assert isinstance(drug, SmallMoleculePerturbation)
        assert drug.compound == _BENOMYL_COMPOUND
        assert drug.concentration == Concentration(
            value=6.9, unit=ConcentrationUnit.micromolar
        )
        assert drug.gapped_fields() == {"solvent"}
    assert grown.pre_culture is not None and frozen.pre_culture is not None
    assert grown.pre_culture.source is PreCultureSource.log_phase_culture
    assert grown.pre_culture.medium == YPD_LIQUID
    assert frozen.pre_culture.source is PreCultureSource.frozen_stock
    assert frozen.pre_culture.source_label == "-5gen"


def test_het_reference_is_by4743_with_the_matched_control_set(
    het: m.HetHillenmeyer2008Dataset,
) -> None:
    reference = _reference(het, 0)
    genome = reference.genome_reference
    assert (genome.strain, genome.ploidy, genome.background.name) == (
        "BY4743",
        "diploid",
        "BY4743",
    )
    assert genome.background == m.hillenmeyer_background()
    assert reference.phenotype_reference.screen_id == _CS1
    assert reference.phenotype_reference.n_samples == 3
    assert reference.environment_reference.perturbations == []
    assert reference.environment_reference.duration_generations == 20.0
    assert reference.environment_reference.pre_culture is not None
    assert (
        _reference(het, 1).environment_reference.pre_culture
        == _experiment(het, 1).environment.pre_culture
    )
    assert [
        het[i]["reference"]["phenotype_reference"]["screen_id"] for i in range(7)
    ] == [_CS1, _CS2, _CS1, _CS1, _CS1, _CS1, _CS1]


def test_het_side_files_gene_set_reference_index_and_manifest(
    het: m.HetHillenmeyer2008Dataset,
) -> None:
    preprocess = Path(het.root) / "preprocess"
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "YAL001C",
        "YBR001C",
        "YBR002W",
        "YOR202W",
    ]
    index = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert sorted(
        (
            entry["member_indices"],
            entry["reference"]["phenotype_reference"]["screen_id"],
        )
        for entry in index
    ) == [([0, 2, 3, 4, 5, 6], _CS1), ([1], _CS2)]
    manifest = json.loads((preprocess / "build_manifest.json").read_text())
    assert manifest["dataset_name"] == "het"
    assert manifest["loader_class"] == _HET
    # the old EngineeredCopyNumberPerturbation leaf has left the closure (#505 G1)
    assert sorted(manifest["closure"]) == [
        "AlleleEdit",
        "AssayType",
        "BackgroundAllele",
        "BarcodedKanMxDeletionPerturbation",
        "ComponentDefinition",
        "Compound",
        "Concentration",
        "ConcentrationUnit",
        "ConstructedOrf",
        "CultureEnvironment",
        "CultureFormat",
        "DeletionPerturbation",
        "DoseBasis",
        "EndpointRule",
        "Environment",
        "EnvironmentPerturbation",
        "EnvironmentPhysicalPerturbation",
        "EnvironmentResponseExperiment",
        "EnvironmentResponseExperimentReference",
        "EnvironmentResponsePhenotype",
        "Experiment",
        "ExperimentReference",
        "GenePerturbation",
        "GenomicSpan",
        "Genotype",
        "HashableProvenanceGapMixin",
        "HeterozygousDeletionPerturbation",
        "KanMxDeletionPerturbation",
        "MatingType",
        "MeasurementType",
        "Media",
        "MediaComponent",
        "MediaComponentRole",
        "ModelStrict",
        "OrfHistoryRelation",
        "Phenotype",
        "PhysicalFactor",
        "PreCulture",
        "PreCultureSource",
        "PresenceAbsencePerturbation",
        "ProvenanceGapMixin",
        "Publication",
        "ReferenceGenome",
        "ResponseCategory",
        "SampleUnit",
        "SmallMoleculePerturbation",
        "Solvent",
        "StrainBackground",
        "StrainConstruction",
        "StrainEnvironmentResponseExperiment",
        "StrainEnvironmentResponseExperimentReference",
        "StrainReferenceGenome",
        "Temperature",
        "TemperatureUnit",
        "UncertaintyType",
        "Zygosity",
    ]
    assert het.raw_file_names == [
        "het.ratio_result_nm.pub",
        "het.txt",
        "het_controls.txt",
    ]
    assert het.experiment_class is StrainEnvironmentResponseExperiment
    assert het.reference_class is StrainEnvironmentResponseExperimentReference
    with pytest.raises(
        NotImplementedError,
        match="Hillenmeyer2008 builds its records in process\\(\\); see iter_records",
    ):
        het.create_experiment()
    frame = {"untouched": True}
    assert het.preprocess_raw(frame) is frame


def test_het_drop_reports_count_columns_strains_and_genes(
    het: m.HetHillenmeyer2008Dataset,
) -> None:
    """Each dropped column is its own environment; records-not-written are counted over
    the kept strain rows. The HO control row is a dropped strain, not a YDL227C record.
    """
    preprocess = Path(het.root) / "preprocess"
    report = json.loads((preprocess / "dropped_records.json").read_text())
    assert report["dataset"] == _HET
    assert report["matrix"] == "het.ratio_result_nm.pub"
    assert report["kept_records"] == 7
    assert sorted(report["rules"]) == sorted(m.DROP_RULES)
    assert report["by_rule"] == {
        rule: {"n_environments": 1, "n_arrays": 1, "n_records_not_written": n}
        for rule, n in (
            ("unidentifiable_agent", 1),
            ("heat_shock_cycle_not_representable", 1),
            ("key_header_compound_conflict_no_tie_breaker", 1),
            ("zero_generations_no_exposure", 1),
        )
    }
    details = {(e["rule"], e["detail"]) for e in report["environments"]}
    assert details == {
        ("heat_shock_cycle_not_representable", "37c, 45c"),
        ("unidentifiable_agent", "amphotericin"),
        (
            "key_header_compound_conflict_no_tie_breaker",
            "header 'colchicine' vs key file 'colchiceine'",
        ),
        ("zero_generations_no_exposure", "minimal media: 0gen"),
    }
    strains = json.loads((preprocess / "dropped_strains.json").read_text())
    assert strains["by_rule"] == {
        "ho_control_strain_unknown_construction": {
            "row_ids": ["YDL227C:ctrl_1"],
            "n_records_not_written": 1,
        }
    }
    checks = json.loads((preprocess / "key_header_checks.json").read_text())
    assert checks["n_by_outcome"] == {"agree": 5, "spelling_variant": 1, "conflict": 1}
    constructed = json.loads((preprocess / "constructed_orfs.json").read_text())
    assert (
        constructed["n_renamed_source_orfs"],
        constructed["n_merged_source_orfs"],
    ) == (2, 1)
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
    ) == (4, 6, 2)
    assert batches["batches_by_orf"] == {
        "YAL001C": ["chr00_1", "chr1_2"],
        "YBR001C": ["chr3_9", "chr6_1"],
        "YBR002W": ["chr5_1"],
        "YOR202W": ["chr15_3"],
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


def test_hom_record_is_a_barcoded_kanmx4_deletion_and_pdr5_is_dropped(
    tmp_path: Path, data_root: Path
) -> None:
    """The SOM says the homozygous PDR5 strain had the wrong gene deleted (#505 G4)."""
    hom = m.HomHillenmeyer2008Dataset(root=str(tmp_path / "hom"), genome=_StubGenome())
    assert len(hom) == 1
    experiment = _experiment(hom, 0)
    (pert,) = _genotype(experiment).perturbations
    assert isinstance(pert, BarcodedKanMxDeletionPerturbation)
    assert (pert.systematic_gene_name, pert.cassette, pert.collection) == (
        "YAL001C",
        "kanMX4",
        "hom_01",
    )
    assert pert.construction == StrainConstruction(batch="chr00_1")
    assert pert.gapped_fields() == {"barcode", "downtag_barcode"}
    phenotype = experiment.phenotype
    assert (phenotype.measurement_type, phenotype.environment_response) == (
        MeasurementType.z_score,
        1.5,
    )
    assert (phenotype.screen_id, phenotype.units) == (_HOM_CS, m.UNITS_HOM)
    reference = _reference(hom, 0)
    assert reference.genome_reference.strain == "BY4743"
    assert reference.phenotype_reference.n_samples == 2
    strains = json.loads(
        (tmp_path / "hom" / "preprocess" / "dropped_strains.json").read_text()
    )
    assert strains["by_rule"] == {
        "som_wrong_gene_deleted": {
            "row_ids": ["YOR153W:chr15_2"],
            "n_records_not_written": 1,
        }
    }
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
    with pytest.raises(RawSha256MismatchError) as err:
        m.HetHillenmeyer2008Dataset(root=str(tmp_path / "het"), genome=_StubGenome())
    assert str(err.value) == (
        f"sha256 mismatch for {matrix}: expected {expected}, observed {got}"
    )
    assert not (tmp_path / "het" / "raw" / "het.txt").exists()


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


def test_process_rechecks_the_raw_files_sha256_at_build_time(
    tmp_path: Path, data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raw matrix written directly (so ``download()`` is skipped) whose bytes differ
    from the manifest pin is refused by ``process()`` itself, with
    ``RawSha256MismatchError`` naming the raw file and both digests, before any store.
    """
    monkeypatch.setattr(m, "verify_raw_files", verify_raw_files)
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
    with pytest.raises(RawSha256MismatchError) as err:
        m.HetHillenmeyer2008Dataset(root=str(tmp_path / "het"), genome=_StubGenome())
    assert str(err.value) == (
        f"sha256 mismatch for {raw / 'het.ratio_result_nm.pub'}: expected {expected}, "
        f"observed {got}"
    )
    assert list((tmp_path / "het" / "processed").iterdir()) == []


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
