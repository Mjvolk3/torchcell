# tests/torchcell/datasets/scerevisiae/test_hillenmeyer2008_synthetic.py
# [[tests.torchcell.datasets.scerevisiae.test_hillenmeyer2008_synthetic]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_hillenmeyer2008_synthetic.py
"""Hillenmeyer 2008 HET and HOM loaders built end to end on synthetic FitDb matrices.

``process()`` reads, besides ``raw/``, only the raw-mirror ``manifest.json`` (the matrix
sha256 is re-checked at build time); it consults no genome. ``DATA_ROOT`` is pointed into
``tmp_path`` and the raw mirror is deposited with the loader's own ``deposit_raw_mirror``
(``verify_source=False``, no network). The dataset root starts empty, so ``download()``
symlinks the three mirror files and ``process()`` runs once.

HET matrix columns (filename: condition, dose, generations -> control set): A1 and A2
benomyl 6.9 uM 20gen -> CS1 (pool het_01; one group of two arrays), A3 benomyl 6.9 uM
-5gen -> CS2 (pool het_02), A4 amphotericin (no curated structure: STORED with the typed
inchikey gap), A5 ``37c, 45c`` (heat-shock cycle: the one array rule left, needs schema),
A6 colchicine whose key file says colchiceine (STORED under the header's compound), A7
minimal media at 0gen -> CS0 (STORED with duration 0 and a typed pre-culture gap), A8
``chemical diversity labs 14a`` (a ChemDiv library code, identity-table status
PROPRIETARY: HELD OUT under UNIDENTIFIED_COMPOUND_RULE). CS1 has 3 control arrays, CS2
2, CS0 1. Groups in column order: G1 (A1, A2), G2 (A3), G3 (A4), G4 (A6), G5 (A7).

HET rows (A1..A8):

    "YAL001C:chr00_1"  1.0   3.0   0.5  2.0  NA  1.0  1.0  3.0  suspicious batch chr00_1
    YAL001C:chr1_2     2.0   NA    ""   ""   ""                  second construction
    YBR001C:chr3_9     0.25  0.25  NaN  null 1.0
    YAR044W:chr5_1     4.0   (line ends)                         merged by a later SGD release
    YOR202W:chr15_3    1.0                                       HIS3, a BY4743 marker locus
    YDL227C:ctrl_1     5.0   5.0                                 'ctrl_' batch, stored
    YPL999W:chr1_1     9     9     9    9    9   NA   NA   4.0  no current gene, stored

Every row is stored under its released ORF and batch, and nothing averages across rows.
Records: 0-4 YAL001C/chr00_1 in G1 (1.0, 3.0: mean 2.0, sample SD sqrt(2)), G2 0.5, G3
2.0, G4 1.0, G5 1.0; 5 YAL001C/chr1_2 G1 2.0; 6 YBR001C G1 (0.25 twice: an SD of exactly
0, stored as a typed gap); 7 YAR044W G1 4.0; 8 YOR202W G1 1.0; 9 YDL227C/ctrl_1 G1 (5.0
twice); 10-12 YPL999W G1 (9 twice), G2 9, G3 9.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import pytest

from torchcell.data import RawSha256MismatchError
from torchcell.data.experiment_dataset import verify_raw_files
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import SD, YPD_LIQUID
from torchcell.datamodels.schema import (
    BarcodedKanMxDeletionPerturbation,
    Compound,
    Concentration,
    ConcentrationUnit,
    CultureEnvironment,
    EukaryoticStrainBackground,
    Genotype,
    HeterozygousDeletionPerturbation,
    MeasurementType,
    PreCultureSource,
    Publication,
    SmallMoleculePerturbation,
    StrainConstruction,
    StrainEnvironmentResponseExperiment,
    StrainEnvironmentResponseExperimentReference,
    heterozygous_deletion_functional_copies,
)
from torchcell.datasets.scerevisiae import hillenmeyer2008 as m
from torchcell.literature.manifest import RetrievalMethod
from torchcell.verification.sourced import ProvenanceGapReason

_HET = "HetHillenmeyer2008Dataset"
_HOM = "HomHillenmeyer2008Dataset"
_CS1 = "het_01::old scanner::20::tag3::YPD::dmso::0"
_CS2 = "het_02::old scanner::-5::tag3::YPD::dmso::0"
_CS0 = "het_02::old scanner::0::tag3::YPD::dmso::0"
_HOM_CS = "hom_01::old scanner::20::tag3::YPD::dmso::0"

_HET_HEADER = [
    "Orf",
    "A1:benomyl:6.9:um::::20gen:het_01:old scanner",
    "A2:benomyl:6.9:um::::20gen:het_01:old scanner",
    "A3:benomyl:6.9:um::::-5gen:het_02:old scanner",
    "A4:amphotericin:10:um::::20gen:het_01:old scanner",
    "A5:37c, 45c::::::20gen:het_01:old scanner",
    "A6:colchicine:250:um::::20gen:het_01:old scanner",
    "A7:minimal media::::::0gen:het_02:old scanner",
    "A8:chemical diversity labs 14a:10:um::::20gen:het_01:old scanner",
]
_HET_ROWS = [
    ['"YAL001C:chr00_1"', "1.0", "3.0", "0.5", "2.0", "NA", "1.0", "1.0", "3.0"],
    ["YAL001C:chr1_2", "2.0", "NA", "", "", ""],
    ["YBR001C:chr3_9", "0.25", "0.25", "NaN", "null", "1.0"],
    ["YAR044W:chr5_1", "4.0"],
    ["YOR202W:chr15_3", "1.0"],
    ["YDL227C:ctrl_1", "5.0", "5.0"],
    ["YPL999W:chr1_1", "9", "9", "9", "9", "9", "NA", "NA", "4.0"],
]


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
                ["A8", "chemical diversity labs 14a", _CS1],
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
    _deposit(tmp_path, data_root)
    return data_root


@pytest.fixture
def het(tmp_path: Path, data_root: Path) -> m.HetHillenmeyer2008Dataset:
    return m.HetHillenmeyer2008Dataset(root=str(tmp_path / "het"))


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


#: (released ORF, batch, mean, n, sd, control set, pool), in record order
_HET_EXPECTED = [
    ("YAL001C", "chr00_1", 2.0, 2, math.sqrt(2.0), _CS1, "het_01"),
    ("YAL001C", "chr00_1", 0.5, 1, None, _CS2, "het_02"),
    ("YAL001C", "chr00_1", 2.0, 1, None, _CS1, "het_01"),
    ("YAL001C", "chr00_1", 1.0, 1, None, _CS1, "het_01"),
    ("YAL001C", "chr00_1", 1.0, 1, None, _CS0, "het_02"),
    ("YAL001C", "chr1_2", 2.0, 1, None, _CS1, "het_01"),
    ("YBR001C", "chr3_9", 0.25, 2, None, _CS1, "het_01"),
    ("YAR044W", "chr5_1", 4.0, 1, None, _CS1, "het_01"),
    ("YOR202W", "chr15_3", 1.0, 1, None, _CS1, "het_01"),
    ("YDL227C", "ctrl_1", 5.0, 2, None, _CS1, "het_01"),
    ("YPL999W", "chr1_1", 9.0, 2, None, _CS1, "het_01"),
    ("YPL999W", "chr1_1", 9.0, 1, None, _CS2, "het_02"),
    ("YPL999W", "chr1_1", 9.0, 1, None, _CS1, "het_01"),
]


def test_het_records_are_one_per_released_strain_and_control_set(
    het: m.HetHillenmeyer2008Dataset,
) -> None:
    """Thirteen records in row-then-group order, each keyed on the released ORF and
    batch with no ``ConstructedOrf``: the merged-by-SGD ORF (YAR044W), the ORF with no
    current gene (YPL999W) and the ``ctrl_`` row of YDL227C are stored like the rest.
    """
    assert len(het) == 13
    for i, (orf, batch, mean, n, sd, cs, pool) in enumerate(_HET_EXPECTED):
        experiment = _experiment(het, i)
        assert experiment.experiment_type == "strain_environment_response"
        (pert,) = _genotype(experiment).perturbations
        assert isinstance(pert, HeterozygousDeletionPerturbation)
        assert (
            pert.systematic_gene_name,
            pert.perturbed_gene_name,
            pert.cassette,
            pert.collection,
        ) == (orf, orf, "kanMX4", pool)
        assert pert.construction == StrainConstruction(batch=batch)
        assert pert.constructed_orf is None
        phenotype = experiment.phenotype
        assert (phenotype.screen_id, phenotype.n_samples) == (cs, n)
        assert phenotype.environment_response == pytest.approx(mean)
        if sd is None:
            assert phenotype.environment_response_uncertainty is None
        else:
            assert phenotype.environment_response_uncertainty == pytest.approx(sd)
        assert phenotype.units == m.UNITS_HET
    # the zero-SD groups are a typed gap, not a dispersion of 0
    for i in (6, 9, 10):
        assert _experiment(het, i).phenotype.gapped_fields() == {
            "environment_response_uncertainty"
        }
    # G2: HIS3 is null in BY4743 before and after the deletion
    his3 = _genotype(_experiment(het, 8)).perturbations[0]
    assert isinstance(his3, HeterozygousDeletionPerturbation)
    assert his3.gapped_fields() == {"barcode", "downtag_barcode", "replaced_allele"}
    background = _reference(het, 8).genome_reference.background
    assert isinstance(background, EukaryoticStrainBackground)
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


def test_an_unidentified_compound_is_stored_with_the_inchikey_gap(
    het: m.HetHillenmeyer2008Dataset,
) -> None:
    """A4 amphotericin: the curated table gives no structure, so the record keeps the
    released label and the typed ``inchikey`` gap ``resolved_compound`` attaches.
    """
    environment = _experiment(het, 2).environment
    (drug,) = environment.perturbations
    assert isinstance(drug, SmallMoleculePerturbation)
    assert drug.compound == resolved_compound("amphotericin")
    assert drug.compound.name == "amphotericin"
    assert drug.compound.inchikey is None
    assert "inchikey" in drug.compound.gapped_fields()
    assert drug.concentration == Concentration(
        value=10.0, unit=ConcentrationUnit.micromolar
    )


def test_a_key_header_compound_disagreement_is_served_under_the_header(
    het: m.HetHillenmeyer2008Dataset,
) -> None:
    """A6: header colchicine, key file colchiceine; the header's compound is served."""
    (drug,) = _experiment(het, 3).environment.perturbations
    assert isinstance(drug, SmallMoleculePerturbation)
    assert drug.compound == resolved_compound("colchicine")
    assert drug.concentration == Concentration(
        value=250.0, unit=ConcentrationUnit.micromolar
    )


def test_a_zero_generation_array_is_stored_with_a_typed_pre_culture_gap(
    het: m.HetHillenmeyer2008Dataset,
) -> None:
    """A7 minimal media at 0gen: duration 0.0, no pre-culture, and the gap says why;
    its 0gen control set gets the same treatment on the reference.
    """
    environment = _experiment(het, 4).environment
    assert isinstance(environment, CultureEnvironment)
    assert environment.media == SD
    assert (environment.duration_generations, environment.pre_culture) == (0.0, None)
    assert environment.gapped_fields() == {
        "temperature",
        "culture_format",
        "auxotroph_supplements",
        "pre_culture",
    }
    gap = next(g for g in environment.provenance_gaps if g.field == "pre_culture")
    assert gap.reason is ProvenanceGapReason.not_reported_by_primary
    assert gap.note is not None and "'0gen'" in gap.note
    reference = _reference(het, 4).environment_reference
    assert isinstance(reference, CultureEnvironment)
    assert (reference.duration_generations, reference.pre_culture) == (0.0, None)
    assert "pre_culture" in reference.gapped_fields()
    assert _reference(het, 4).phenotype_reference.n_samples == 1


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
        het[i]["reference"]["phenotype_reference"]["screen_id"] for i in range(13)
    ] == [entry[5] for entry in _HET_EXPECTED]


def test_het_side_files_gene_set_reference_index_and_manifest(
    het: m.HetHillenmeyer2008Dataset,
) -> None:
    preprocess = Path(het.root) / "preprocess"
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "YAL001C",
        "YAR044W",
        "YBR001C",
        "YDL227C",
        "YOR202W",
        "YPL999W",
    ]
    index = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert sorted(
        (
            entry["member_indices"],
            entry["reference"]["phenotype_reference"]["screen_id"],
        )
        for entry in index
    ) == [([0, 2, 3, 5, 6, 7, 8, 9, 10, 12], _CS1), ([1, 11], _CS2), ([4], _CS0)]
    manifest = json.loads((preprocess / "build_manifest.json").read_text())
    assert manifest["dataset_name"] == "het"
    assert manifest["loader_class"] == _HET
    # the old EngineeredCopyNumberPerturbation leaf has left the closure (#505 G1)
    # Since #734 the closure also lists the module-level bindings (Literal vocabularies,
    # id patterns, aliases, helper functions) the record classes reach; each is its own
    # surface node with its own fingerprint, so a narrowed Literal or a changed pattern
    # is a schema change the manifest can see.
    assert sorted(manifest["closure"]) == [
        "ALLELE_EDIT_SO",
        "ARTIFACT_MEMBER_SEPARATOR",
        "ARTIFACT_TIERS",
        "ARTIFACT_URI_SCHEME",
        "ASSEMBLY_ACCESSION_PATTERN",
        "ASSEMBLY_SET_ACCESSIONS",
        "AlleleEdit",
        "ArtifactRef",
        "ArtifactTier",
        "AssayType",
        "AssemblySet",
        "BACTERIAL_ASSEMBLY_SETS",
        "BACTERIAL_LOCUS_TAG_PATTERNS",
        "BackgroundAllele",
        "BacterialGeneNamespace",
        "BarcodedKanMxDeletionPerturbation",
        "CASSETTE_INTEGRATION_SO",
        "CATEGORICAL_MEASUREMENT_TYPES",
        "CHEBI_ID_PATTERN",
        "ChromosomeCopyChange",
        "ComponentDefinition",
        "Compound",
        "Concentration",
        "ConcentrationUnit",
        "ConstructPlacement",
        "ConstructSpan",
        "ConstructVariantCall",
        "ConstructedOrf",
        "CultureEnvironment",
        "CultureFormat",
        "CultureGrowthPhase",
        "DeletionPerturbation",
        "DerivedIdentifierMapping",
        "DerivedIdentifierRoute",
        "DoseBasis",
        "EndpointRule",
        "Environment",
        "EnvironmentPerturbation",
        "EnvironmentPerturbationType",
        "EnvironmentPhysicalPerturbation",
        "EnvironmentResponseExperiment",
        "EnvironmentResponseExperimentReference",
        "EnvironmentResponsePhenotype",
        "EukaryoticStrainBackground",
        "Experiment",
        "ExperimentReference",
        "FeatureKey",
        "GENE_ID_NAMESPACE_PATTERNS",
        "GeneIdNamespace",
        "GenePerturbation",
        "GenePerturbationType",
        "GenomeZygosity",
        "GenomicSpan",
        "Genotype",
        "HashableProvenanceGapMixin",
        "HeterozygousDeletionPerturbation",
        "HostPlacement",
        "INCHIKEY_PATTERN",
        "IntegratedCassette",
        "IsolateManipulation",
        "JW_IDENTIFIER_PATTERN",
        "KanMxDeletionPerturbation",
        "Localization",
        "MatingType",
        "MeasurementType",
        "Media",
        "MediaComponent",
        "MediaComponentRole",
        "ModelStrict",
        "NaturalIsolateDescriptor",
        "OrfHistoryRelation",
        "PLACEMENT_TARGET_REFUSED_SO",
        "Phenotype",
        "PhysicalFactor",
        "PreCulture",
        "PreCultureSource",
        "PresenceAbsencePerturbation",
        "ProvenanceGapMixin",
        "Publication",
        "REFERENCE_STRAIN_ASSEMBLY_SETS",
        "ReferenceGenome",
        "ReferenceStrain",
        "ResponseCategory",
        "SGD_S288C_NAMESPACE",
        "SGD_SYSTEMATIC_GENE_PATTERN",
        "SOTerm",
        "SO_ID_PATTERN",
        "SampleUnit",
        "SgaPerturbationType",
        "SmallMoleculePerturbation",
        "Solvent",
        "SourceType",
        "SpanComposition",
        "SpanPart",
        "StrainBackground",
        "StrainConstruction",
        "StrainEnvironmentResponseExperiment",
        "StrainEnvironmentResponseExperimentReference",
        "StrainReferenceGenome",
        "Temperature",
        "TemperatureUnit",
        "UNIPROT_ACCESSION_PATTERN",
        "UncertaintyType",
        "VariantCallMode",
        "VariantFrequencyBasis",
        "VariantType",
        "Zygosity",
        "_ARTIFACT_SHA256",
        "_Z95",
        "_check_gene_namespace",
        "_namespace_of_identifier",
        "_namespace_of_locus_tag",
        "_require_value_or_gap",
        "_validate_host_identifier",
        "_validate_so_id",
        "derive_se",
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


def test_het_ledgers_count_the_remaining_rule_the_gaps_and_every_cell(
    het: m.HetHillenmeyer2008Dataset,
) -> None:
    """The heat-shock array (needs schema) and the ChemDiv-code array (unidentified
    agent) are not stored; each is counted over the stored rows. The compound gap ledger
    lists amphotericin with its two records, and the 21 released value cells split into
    17 stored + 2 in each dropped array.
    """
    preprocess = Path(het.root) / "preprocess"
    report = json.loads((preprocess / "dropped_records.json").read_text())
    assert report["dataset"] == _HET
    assert report["matrix"] == "het.ratio_result_nm.pub"
    assert report["kept_records"] == 13
    assert (
        sorted(report["rules"])
        == sorted(m.DROP_RULES)
        == [
            "compound_has_no_released_name_or_structure_not_representable",
            "heat_shock_cycle_not_representable",
            "unnamed_agent_dosed_into_a_media_swap",
        ]
    )
    one_array = {"n_environments": 1, "n_arrays": 1, "n_records_not_written": 2}
    assert report["by_rule"] == {
        "heat_shock_cycle_not_representable": one_array,
        m.UNIDENTIFIED_COMPOUND_RULE: one_array,
    }
    assert [(e["rule"], e["detail"]) for e in report["environments"]] == [
        (m.UNIDENTIFIED_COMPOUND_RULE, "chemical diversity labs 14a"),
        ("heat_shock_cycle_not_representable", "37c, 45c"),
    ]
    assert report["cell_accounting"] == {
        "n_value_cells": 21,
        "n_in_stored_records": 17,
        "n_in_dropped_arrays": {
            "heat_shock_cycle_not_representable": 2,
            m.UNIDENTIFIED_COMPOUND_RULE: 2,
        },
        "n_in_withdrawn_strains": {},
    }
    gaps = json.loads((preprocess / "compound_identity_gaps.json").read_text())
    assert (gaps["dataset"], gaps["n_compounds"], gaps["n_records"]) == (_HET, 1, 2)
    assert gaps["compounds"] == {
        "amphotericin": {"n_environments": 1, "n_arrays": 1, "n_records": 2}
    }
    strains = json.loads((preprocess / "dropped_strains.json").read_text())
    assert strains["by_rule"] == {}
    checks = json.loads((preprocess / "key_header_checks.json").read_text())
    assert checks["n_by_outcome"] == {"agree": 6, "spelling_variant": 1, "conflict": 1}
    (conflict,) = [c for c in checks["disagreements"] if c["outcome"] == "conflict"]
    assert conflict == {
        "filename": "A6",
        "header_condition": "colchicine",
        "key_condition": "colchiceine",
        "outcome": "conflict",
        "rule": "header_served_key_disagrees",
    }
    assert not (preprocess / "constructed_orfs.json").exists()
    assert not (preprocess / "dropped_genes.json").exists()


def test_an_unidentified_compound_is_held_out_of_the_store_and_listed_in_the_ledger(
    het: m.HetHillenmeyer2008Dataset,
) -> None:
    """A8 doses a ChemDiv library code (status PROPRIETARY, no released structure): no
    stored record carries it, the ledger lists it with its two value cells and the
    SOM quotes, and the cells still balance (stored + every held-out rule = released).
    """
    names = {
        p.compound.name
        for i in range(len(het))
        for p in _experiment(het, i).environment.perturbations
        if isinstance(p, SmallMoleculePerturbation)
    }
    assert names == {"benomyl", "amphotericin", "colchicine"}
    preprocess = Path(het.root) / "preprocess"
    report = json.loads((preprocess / "dropped_records.json").read_text())
    assert report["unidentified_compound_rule"] == m.UNIDENTIFIED_COMPOUND_RULE
    assert report["unidentified_compound_policy"] == m.UNIDENTIFIED_COMPOUND_POLICY
    assert report["unidentified_compounds_held_out"] == [
        {
            "label": "chemical diversity labs 14a",
            "n_environments": 1,
            "n_arrays": 1,
            "n_records_not_written": 2,
            "n_value_cells": 2,
            "headers": [_HET_HEADER[8]],
            "key_file_conditions": ["chemical diversity labs 14a"],
        }
    ]
    sources = report["unidentified_compound_sources"]
    assert [s["quote"] for s in sources] == [
        m.QUOTE_TABLE_S2,
        m.QUOTE_TABLE_S1_CHEMDIV,
        m.QUOTE_TABLE_S1_TELOMERASE,
    ]
    assert {s["provenance"]["sha256"] for s in sources} == {m.SOM_SHA256}
    accounting = report["cell_accounting"]
    assert accounting["n_value_cells"] == accounting["n_in_stored_records"] + sum(
        accounting["n_in_dropped_arrays"].values()
    ) + sum(accounting["n_in_withdrawn_strains"].values())
    gaps = json.loads((preprocess / "compound_identity_gaps.json").read_text())
    assert "chemical diversity labs 14a" not in gaps["compounds"]


def test_het_batch_reports_flag_the_suspicious_construction(
    het: m.HetHillenmeyer2008Dataset,
) -> None:
    """YAL001C has two construction rows (chr00_1, chr1_2); chr00_1 is one of the ten
    SOM batches, so YAL001C is the one flagged ORF. The ``ctrl_1`` batch is listed
    verbatim.
    """
    preprocess = Path(het.root) / "preprocess"
    batches = json.loads((preprocess / "strain_batches.json").read_text())
    assert (
        batches["n_orfs"],
        batches["n_rows"],
        batches["n_orfs_with_multiple_batches"],
    ) == (6, 7, 1)
    assert batches["batches_by_orf"] == {
        "YAL001C": ["chr00_1", "chr1_2"],
        "YAR044W": ["chr5_1"],
        "YBR001C": ["chr3_9"],
        "YDL227C": ["ctrl_1"],
        "YOR202W": ["chr15_3"],
        "YPL999W": ["chr1_1"],
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
    hom = m.HomHillenmeyer2008Dataset(root=str(tmp_path / "hom"))
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
    report = json.loads(
        (tmp_path / "hom" / "preprocess" / "dropped_records.json").read_text()
    )
    assert report["cell_accounting"] == {
        "n_value_cells": 2,
        "n_in_stored_records": 1,
        "n_in_dropped_arrays": {},
        "n_in_withdrawn_strains": {"som_wrong_gene_deleted": 1},
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
        m.HetHillenmeyer2008Dataset(root=str(tmp_path / "het"))
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
        m.HetHillenmeyer2008Dataset(root=str(tmp_path / "het"))


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
        m.HetHillenmeyer2008Dataset(root=str(tmp_path / "het"))
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


# ---------------------------------------------------------------------------
# 2026.10.06 (Phase 21): the verify_source re-run and small branches.
# ---------------------------------------------------------------------------


class _FixedDate:
    """``date`` stand-in whose ``today()`` is 2026-10-06."""

    @staticmethod
    def today() -> Any:
        import datetime as _dt

        return _dt.date(2026, 10, 6)


def _staging(tmp_path: Path, name: str) -> Path:
    source = tmp_path / name
    source.mkdir()
    for file_name, text in _source_files().items():
        (source / file_name).write_text(text)
    return source


def test_deposit_with_verify_source_reruns_each_archived_retrieval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``verify_source=True`` calls ``torchcell.literature.retrieve.direct_url`` once
    per file on its Wayback URL, in ``raw_relpaths`` order, and records on each
    retrieval ``last_check = SourceCheck(checked_at=<today>, produced_sha256=<sha256
    of the re-fetched bytes>, matches=True)``; the stub serves the staged bytes, so
    each produced digest equals the deposited one. The rest of each retrieval record
    is the ``direct_url`` method, the Wayback URL as ``source_url`` and as the
    retriever's ``url`` param, the file's sha256, and ``retrieved_at`` at its default
    ``RAW_RETRIEVED_AT`` = 2026-07-11.
    """
    import torchcell.literature.retrieve as retrieve

    data_root = tmp_path / "dr"
    source = _staging(tmp_path, "staging")
    by_url = {
        m.wayback_url(name): text.encode() for name, text in _source_files().items()
    }
    fetched: list[str] = []

    def direct_url(url: str) -> bytes:
        fetched.append(url)
        return by_url[url]

    monkeypatch.setattr(retrieve, "direct_url", direct_url)
    monkeypatch.setattr(m, "date", _FixedDate)
    m.deposit_raw_mirror(source, data_root=str(data_root), verify_source=True)
    assert fetched == [m.wayback_url(name) for name in m.raw_relpaths()]
    manifest = m.load_manifest(str(data_root))
    for record in manifest.files:
        name = Path(record.path).name
        assert record.retrieval is not None
        check = record.retrieval.last_check
        assert check is not None
        digest = hashlib.sha256(_source_files()[name].encode()).hexdigest()
        assert check.model_dump() == {
            "checked_at": "2026-10-06",
            "produced_sha256": digest,
            "matches": True,
        }
        assert record.retrieval.model_dump(exclude={"last_check"}) == {
            "method": RetrievalMethod.direct_url,
            "source_url": m.wayback_url(name),
            "retriever": "torchcell.literature.retrieve.direct_url",
            "params": {"url": m.wayback_url(name)},
            "sha256": digest,
            "retrieved_at": "2026-07-11",
        }


def test_deposit_with_verify_source_refuses_a_changed_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the archived URL now yields other bytes, the deposit raises naming both
    digests before copying that file, and writes no manifest.
    """
    import torchcell.literature.retrieve as retrieve

    data_root = tmp_path / "dr"
    source = _staging(tmp_path, "staging")
    monkeypatch.setattr(retrieve, "direct_url", lambda url: b"drifted")
    first = next(iter(m.raw_relpaths()))
    deposited = hashlib.sha256(_source_files()[first].encode()).hexdigest()
    produced = hashlib.sha256(b"drifted").hexdigest()
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"{first}: the archived URL now yields sha256 {produced}, the deposited "
            f"bytes are {deposited}"
        ),
    ):
        m.deposit_raw_mirror(source, data_root=str(data_root), verify_source=True)
    mirror = m.raw_mirror_dir(str(data_root))
    assert not (mirror / "manifest.json").exists()
    assert not (mirror / m.raw_relpaths()[first]).exists()


def test_canonical_concentration_below_one_nanomolar_stays_nanomolar() -> None:
    """0.5 nM = 5e-10 M: no unit of the family leaves a value >= 1 (M 5e-10, mM 5e-7,
    uM 5e-4, nM 0.5), so the fallback stores 5e-10 / 1e-9 = 0.5 nM.
    """
    dose = m.canonical_concentration("0.5", "nM")
    assert (dose.value, dose.unit) == (0.5, ConcentrationUnit.nanomolar)


def test_read_key_conditions_maps_filename_to_condition_and_checks_the_header(
    tmp_path: Path,
) -> None:
    keyfile = tmp_path / "het.txt"
    keyfile.write_text(
        "filename\tcondition\tcontrol_set\nA1\tbenomyl\tCS1\nA2\tx\tCS2\n"
    )
    assert m.read_key_conditions(keyfile) == {"A1": "benomyl", "A2": "x"}
    keyfile.write_text("filename\tcondition\tcs\n")
    message = (
        f"unexpected key-file header in {keyfile}: ['filename', 'condition', 'cs']"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        m.read_key_conditions(keyfile)
