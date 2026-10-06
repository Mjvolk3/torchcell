# tests/torchcell/datasets/scerevisiae/test_hoepfner2014.py
# [[tests.torchcell.datasets.scerevisiae.test_hoepfner2014]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_hoepfner2014.py
"""Unit tests for the Hoepfner 2014 HIP-HOP loader's record construction.

The loader's whole surface is how it turns one deposited column header into a typed
condition, so these tests drive ``_column_meta`` / ``_environment`` / ``_reference`` /
``_genotype`` directly on a synthetic header plus a synthetic Table S1 mapping. What is
pinned is exactly what the review found broken: the compound is keyed on its CLEAN name
and carries a structure identifier, the screen id is a typed field rather than text inside
that name, the medium is the shared library object on BOTH arms, the vehicle carries its
own identity, the two assays record their own exposure durations, and a compound that
resolves to no identifier takes its column out of the build.

2026.09.30 (Phase 16): the end-to-end build is pinned in ``test_hoepfner2014_synthetic.py``;
this file adds the pieces that build does not reach, on the same synthetic header:

- ``_dryad_get`` against a recording fake session (no network): a non-HTML body is
  returned from the first GET (``timeout=180, stream=True``); an HTML page carrying the
  Anubis challenge ``{"id": "c1", "randomData": "abc", "difficulty": 1}`` is closed,
  solved (nonce 26, digest ``0d56d5ce...``: the first nonce whose sha256("abc" + nonce)
  starts with a zero nibble) and passed to
  ``https://datadryad.org<base>/.within.website/x/cmd/anubis/api/pass-challenge`` with
  ``redir=<url>, elapsedTime=100`` (``<base>`` is ``/base`` when the prefix script is
  present, empty when not); a throttled page with no challenge sleeps 15 s and retries;
  a cleared response that is still HTML is closed and retried; 80 throttled pages raise
  "could not clear Anubis challenge for <url>".
- ``_fetch_from_dryad`` with chunks ``b"ab"``, ``b""``, ``b"c"``: the empty keep-alive
  chunk is skipped, the file holds ``abc`` (sha256 ba7816bf...), the session carries the
  browser User-Agent, and a body off the pin raises "<name> sha256 mismatch".
- the sign convention: a deposited HIP cell and HOP cell of -3.0 are both stored as
  -3.0 with ``MEASUREMENT_UNITS`` declaring "negative = hypersensitive, positive =
  resistant" (negative-is-sick, the opposite of Hillenmeyer's positive-is-sick; memory
  ``chemogenomic-response-polarity-splits``), so the loader does not orient.
- the dosage-duration confound (memory ``033-env-chemgen-pooled-build``): for one
  compound, dose and study, the HIP and HOP environments differ in exactly their
  durations (HIP gapped, HOP 16 h / 5 generations), their duration gaps, their culture
  format and their pre-culture, while both references sit on the same ``BY4743``
  ``StrainReferenceGenome``, so no cell is paired across the two arms in one environment.
- #506: the BY4743 background as typed gaps, the heterozygous / barcoded kanMX4 leaves
  with Table S5 construction and ``constructed_orf``, the detection rule, the
  glucose-starvation exclusion, the solvent and pH gaps, and (when ``DATA_ROOT`` and the
  raw mirror are present) the deposited-header and HOP detection counts.
- ``transform_item`` on a spliced record, the inert hooks, the default-genome resolver
  (``load_dotenv`` stubbed, ``SCerevisiaeGenome`` a recorder built once with
  ``overwrite=False``) and ``main`` with the dataset class as a recorder.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os.path as osp
import pickle
import re
import time
from pathlib import Path
from typing import Any

import pytest
import requests

from torchcell.data import RawSha256MismatchError
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import YPD_LIQUID
from torchcell.datamodels.schema import (
    AssayType,
    BarcodedKanMxDeletionPerturbation,
    ConcentrationUnit,
    DoseBasis,
    EndpointRule,
    EnvironmentPhysicalPerturbation,
    HeterozygousDeletionPerturbation,
    MeasurementType,
    PhysicalFactor,
    PreCultureSource,
    SampleUnit,
    SmallMoleculePerturbation,
    heterozygous_deletion_functional_copies,
)
from torchcell.datamodels.strain_background import BRACHMANN_1998, GIAEVER_2002
from torchcell.datasets.scerevisiae import hoepfner2014 as module
from torchcell.datasets.scerevisiae.hoepfner2014 import (
    DROP_RULE,
    SOURCED_VALUES,
    EnvChemgenHoepfner2014Dataset,
    _has_identifier,
    load_table_s5_strains,
)
from torchcell.sequence.genome.scerevisiae.s288c import GeneNameStatus


@pytest.fixture(autouse=True)
def _no_tc_data_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every build here reads ``raw/``; an inherited ``TC_DATA_URL`` would send it to
    the tc-data endpoint instead (``ExperimentDataset._download``).
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)


REPO_ROOT = Path(__file__).resolve().parents[4]

# Two real Table S1 rows: amitriptyline resolves, boromycin is the one compound whose
# released SMILES RDKit cannot parse and whose curated row carries no identifier.
AMITRIPTYLINE_SMILES = "CN(C)CCC=C2c1ccccc1CCc3ccccc23"
BOROMYCIN_SMILES = (
    "CC(C)C(N)C(=O)OC(C)C1C/C=C\\CCC(O)C(C)(C)C7CCC(C)C4(OB25OC(C(=O)O1)C3(O2)"
    "OC(CCC3C)C(C)(C)C(O)CCCC6CC(OC(=O)C4O5)C(C)O6)O7"
)

META: dict[str, dict[str, str | None]] = {
    "3": {"common_name": "Amitriptyline", "smiles": AMITRIPTYLINE_SMILES},
    "409": {"common_name": "Boromycin", "smiles": BOROMYCIN_SMILES},
    "777": {"common_name": None, "smiles": AMITRIPTYLINE_SMILES},
    "888": {"common_name": "Unreleased", "smiles": None},
}

HEADER = [
    '"Systematic Name"',
    '"Ad. scores for Exp. 3_50_HIP_0077"',
    '"Ad. scores for Exp. 3_50_HIP_0077 z-score"',
    '"MADL scores for Exp. 3_50_HIP_0091"',
    '"Ad. scores for Exp. 409_1.2_HIP_0077"',
    '"Ad. scores for Exp. 777_10_HIP_0077"',
    '"Ad. scores for Exp. 888_10_HIP_0077"',
    '"Ad. scores for Exp. 3_50_HOP_0077"',
]


class _FakeTxn:
    """A dict-backed stand-in for the interned LMDB write transaction."""

    def __init__(self) -> None:
        self.store: dict[bytes, bytes] = {}

    def get(self, key: bytes) -> bytes | None:
        return self.store.get(key)

    def put(self, key: bytes, value: bytes) -> None:
        self.store[key] = value


def _dataset() -> EnvChemgenHoepfner2014Dataset:
    """The loader without its PyG init: these tests never touch disk or an LMDB."""
    dataset = EnvChemgenHoepfner2014Dataset.__new__(EnvChemgenHoepfner2014Dataset)
    dataset.name = "EnvChemgenHoepfner2014Dataset"
    return dataset


def _columns(assay: str = "HIP") -> tuple[list[Any], list[tuple[int, str]]]:
    return _dataset()._column_meta(HEADER, assay, META, _FakeTxn())


# ---- compound identity and the drop rule ---------------------------------------- #
def test_compound_name_is_clean_and_carries_no_experiment_tag() -> None:
    """The released common name when there is one, else ``CMB<id>``; an id absent from
    Table S1 also falls back to ``CMB<id>``.
    """
    assert {
        cmb: EnvChemgenHoepfner2014Dataset._compound_name(cmb, META)
        for cmb in [*META, "5"]
    } == {
        "3": "Amitriptyline",
        "409": "Boromycin",
        "777": "CMB777",
        "888": "Unreleased",
        "5": "CMB5",
    }


def test_kept_compound_carries_a_structure_identifier() -> None:
    kept, _ = _columns()
    compound = kept[0].env_dump["perturbations"][0]["compound"]
    assert compound["inchikey"] == "KRMDCWKBEZIMAB-UHFFFAOYSA-N"
    assert compound["name"] == "amitriptyline"  # the table's canonical spelling
    assert compound["smiles"] == AMITRIPTYLINE_SMILES


def test_structure_only_proprietary_compound_resolves_from_its_smiles() -> None:
    """A CMB id with no released name still gets an InChIKey from its SMILES."""
    kept, _ = _columns()
    by_cmb = {col.cmb: col for col in kept}
    compound = by_cmb["777"].env_dump["perturbations"][0]["compound"]
    assert compound["name"] == "CMB777"
    assert compound["inchikey"] == "KRMDCWKBEZIMAB-UHFFFAOYSA-N"


def test_unidentifiable_compound_column_is_dropped_by_the_rule() -> None:
    """Only column 4 (CMB409, boromycin) is dropped, reported as (index, CMB id)."""
    kept, dropped = _columns()
    assert dropped == [(4, "409")]
    assert [col.cmb for col in kept] == ["3", "3", "777"]
    assert DROP_RULE == (
        "drop every record whose compound carries no structure identifier after "
        "resolution: no curated compound_identity_table row with an InChIKey / ChEBI "
        "id / PubChem CID, and no RDKit-parseable released Table S1 SMILES to derive an "
        "InChIKey from"
    )


def test_compound_without_a_released_smiles_is_the_encodable_filter_not_a_drop() -> (
    None
):
    """CMB888 has no structure at all: it never enters the build and is not a 'drop'."""
    kept, dropped = _columns()
    assert "888" not in {col.cmb for col in kept}
    assert "888" not in {cmb for _, cmb in dropped}


def test_z_score_and_other_assay_columns_are_not_kept() -> None:
    kept, _ = _columns("HIP")
    assert [col.index for col in kept] == [1, 3, 5]
    hop_kept, _ = _columns("HOP")
    assert [col.index for col in hop_kept] == [7]


# ---- screen id ------------------------------------------------------------------ #
def test_screen_id_is_a_typed_field_and_separates_two_screens_of_one_dose() -> None:
    kept, _ = _columns()
    by_study = {col.pheno_base["screen_id"]: col for col in kept if col.cmb == "3"}
    assert set(by_study) == {"0077", "0091"}
    doses = {
        col.env_dump["perturbations"][0]["concentration"]["value"]
        for col in by_study.values()
    }
    assert doses == {
        50.0
    }  # same compound, same dose, two screens: only screen_id parts


def test_n_samples_follows_the_column_prefix() -> None:
    kept, _ = _columns()
    by_study = {col.pheno_base["screen_id"]: col for col in kept if col.cmb == "3"}
    assert by_study["0077"].pheno_base["n_samples"] == 2  # 'Ad.'
    assert by_study["0091"].pheno_base["n_samples"] == 1  # 'MADL'
    assert by_study["0077"].pheno_base["sample_unit"] == SampleUnit.technical_replicate


# ---- environment ---------------------------------------------------------------- #
def test_both_arms_use_the_shared_media_library_object() -> None:
    dataset = _dataset()
    kept, _ = _columns()
    assert kept[0].env_dump["media"] == YPD_LIQUID.model_dump()
    reference = dataset._reference("HIP", "0077")
    assert reference.environment_reference.media == YPD_LIQUID


def test_dose_is_typed_and_the_vehicle_carries_its_own_identity() -> None:
    kept, _ = _columns()
    perturbation = kept[0].env_dump["perturbations"][0]
    assert perturbation["concentration"] == {
        "value": 50.0,
        "unit": ConcentrationUnit.micromolar,
        "basis": DoseBasis.IC30,
    }
    solvent = perturbation["solvent"]
    assert solvent["name"] == "DMSO" and solvent["percent"] == 2.0
    assert solvent["compound"]["inchikey"] == "IAZDPXIOMUYVGZ-UHFFFAOYSA-N"
    assert solvent["compound"]["pubchem_cid"] == 679


def test_hip_duration_fields_are_both_gapped_and_hop_states_both_durations() -> None:
    """#506 item 5: HIP hours, generations and endpoint are all gaps (Fig. S2)."""
    dataset = _dataset()
    hip = dataset._environment("HIP", [])
    assert (hip.duration_hours, hip.duration_generations) == (None, None)
    assert [gap.field for gap in hip.provenance_gaps] == [
        "duration_hours",
        "duration_generations",
        "auxotroph_supplements",
    ]
    assert [gap.resolve_with for gap in hip.provenance_gaps[:2]] == [
        module.HOEPFNER_SUPPLEMENT
    ] * 2
    assert hip.culture_format is not None and hip.culture_format.endpoint is None
    assert [g.field for g in hip.culture_format.provenance_gaps] == ["endpoint"]
    hop = dataset._environment("HOP", [])
    assert (hop.duration_hours, hop.duration_generations) == (16.0, 5.0)
    assert [gap.field for gap in hop.provenance_gaps] == ["auxotroph_supplements"]
    assert hip.temperature is not None and hip.temperature.value == 30.0


def test_culture_protocol_is_the_24_well_format_and_its_pre_culture() -> None:
    dataset = _dataset()
    hip, hop = dataset._environment("HIP", []), dataset._environment("HOP", [])
    assert hip.culture_format is not None and hop.culture_format is not None
    for env in (hip, hop):
        assert env.culture_format is not None
        assert env.culture_format.vessel == "24-well plate (Greiner 662102)"
        assert env.culture_format.working_volume_ul == 1600.0
        assert env.culture_format.shaking_rpm == 550.0
    assert hip.culture_format.inoculum_cells_per_strain == 250.0
    assert hop.culture_format.inoculum_cells_per_strain == 320.0
    assert hop.culture_format.endpoint is EndpointRule.fixed_duration
    assert hip.pre_culture is not None and hop.pre_culture is not None
    assert hip.pre_culture.source is PreCultureSource.log_phase_culture
    assert hop.pre_culture.source is PreCultureSource.thaw_recovery
    assert hop.pre_culture.duration_hours == 3.0


def test_dose_above_200_um_has_a_solvent_gap_and_an_acid_a_ph_gap() -> None:
    """#506 item 4: the 2% DMSO vehicle covers doses up to 200 uM only."""
    dataset = _dataset()
    hcl = resolved_compound("Hydrochloric Acid", smiles="Cl", derive_from_smiles=True)
    at_ceiling = dataset._treatment("3", hcl, 200.0)
    assert len(at_ceiling) == 1
    assert isinstance(at_ceiling[0], SmallMoleculePerturbation)
    assert at_ceiling[0].solvent is not None and at_ceiling[0].solvent.percent == 2.0
    above = dataset._treatment("4016", hcl, 60000.0)
    small, ph = above
    assert isinstance(small, SmallMoleculePerturbation)
    assert small.solvent is None
    assert [gap.field for gap in small.provenance_gaps] == ["solvent"]
    assert isinstance(ph, EnvironmentPhysicalPerturbation)
    assert ph.factor is PhysicalFactor.ph and ph.magnitude is None
    assert [gap.field for gap in ph.provenance_gaps] == ["magnitude"]
    assert set(module.PH_AGENT_CONDITIONS) == {"4013", "4016", "4017"}


# ---- reference ------------------------------------------------------------------ #
def test_reference_is_a_vehicle_control_on_the_joinable_strain_token() -> None:
    reference = _dataset()._reference("HOP", "0077")
    genome = reference.genome_reference
    assert genome.strain == "BY4743" and genome.ploidy == "diploid"
    background = genome.background
    assert background.name == "BY4743" and background.provenance is None
    name_gap, construction_gap = background.provenance_gaps
    assert name_gap.field == "provenance"
    assert name_gap.resolve_with == GIAEVER_2002
    assert name_gap.looked_in is not None
    assert name_gap.looked_in.sha256 == module.PAPER_MD_SHA256
    assert "line 102" in str(name_gap.looked_in.page)
    assert construction_gap.field == "construction"
    assert construction_gap.resolve_with == module.HOEPFNER_SUPPLEMENT
    assert {a.allele_name for a in background.alleles} == {
        "his3Δ1",
        "leu2Δ0",
        "lys2Δ0",
        "met15Δ0",
        "ura3Δ0",
    }
    assert all(
        a.provenance_gaps[0].resolve_with == BRACHMANN_1998 for a in background.alleles
    )
    vehicles = [
        p
        for p in reference.environment_reference.perturbations
        if isinstance(p, SmallMoleculePerturbation)
    ]
    assert len(vehicles) == 1
    vehicle = vehicles[0]
    assert vehicle.compound.name == "dimethyl sulfoxide"
    assert vehicle.concentration.value == 2.0
    assert vehicle.concentration.unit is ConcentrationUnit.percent_v_v
    phenotype = reference.phenotype_reference
    assert phenotype.environment_response == 0.0
    assert phenotype.n_samples == 4  # conservative lower end of "four to eight"
    assert phenotype.screen_id == "0077"
    assert phenotype.assay_type is AssayType.pooled_competitive_growth_barcode


def test_every_phenotype_declares_its_uncertainty_absences() -> None:
    kept, _ = _columns()
    gapped = {gap["field"] for gap in kept[0].pheno_base["provenance_gaps"]}
    assert gapped == {
        "environment_response_se",
        "environment_response_uncertainty",
        "environment_response_uncertainty_type",
    }
    assert kept[0].pheno_base["measurement_type"] == MeasurementType.sensitivity_score
    assert (
        kept[0].pheno_base["assay_type"] is AssayType.pooled_competitive_growth_barcode
    )


# ---- genotype ------------------------------------------------------------------- #
def test_hip_is_a_heterozygous_kanmx4_deletion_and_hop_a_barcoded_deletion() -> None:
    dataset = _dataset()
    hip = dataset._genotype("HIP", "YAL001C").perturbations[0]
    assert isinstance(hip, HeterozygousDeletionPerturbation)
    assert (hip.cassette, hip.collection) == ("kanMX4", "YSC1055")
    assert hip.state == "present" and hip.constructed_orf is None
    assert [g.field for g in hip.provenance_gaps] == ["barcode", "construction"]
    hop = dataset._genotype("HOP", "YAL001C").perturbations[0]
    assert isinstance(hop, BarcodedKanMxDeletionPerturbation)
    assert (hop.cassette, hop.collection) == ("kanMX4", "YSC1056")
    assert [g.field for g in hop.provenance_gaps] == ["barcode"]


def test_marker_locus_dose_is_derived_against_the_background() -> None:
    """#506 item 1: HIS3 is 0 copies before and after; LYS2 / MET17 are undetermined."""
    dataset = _dataset()
    background = module.hoepfner_background()
    doses = {}
    for orf in ("YOR202W", "YBR115C", "YLR303W", "YAL001C"):
        leaf = dataset._genotype("HIP", orf).perturbations[0]
        assert isinstance(leaf, HeterozygousDeletionPerturbation)
        doses[orf] = heterozygous_deletion_functional_copies(background, leaf)
        gapped = [g.field for g in leaf.provenance_gaps]
        assert ("replaced_allele" in gapped) is (orf != "YAL001C")
    assert doses == {"YOR202W": 0, "YBR115C": None, "YLR303W": None, "YAL001C": 1}


def test_table_s5_construction_is_carried_per_hip_strain() -> None:
    """Lab / Batch / Plate / Well verbatim; a duplicated ORF keeps lab and batch only."""
    strains = load_table_s5_strains(REPO_ROOT)
    dataset = _dataset()
    single = dataset._genotype("HIP", "YBR271W", "YBR271W", strains["YBR271W"])
    leaf = single.perturbations[0]
    assert isinstance(leaf, HeterozygousDeletionPerturbation)
    assert leaf.construction is not None
    assert (leaf.construction.lab, leaf.construction.batch) == ("Lab 14", "chr00_1")
    assert leaf.construction.plate is not None and leaf.construction.well is not None
    assert [g.field for g in leaf.provenance_gaps] == ["barcode"]
    assert len(strains["YCL023C"]) == 2
    duplicate = dataset._genotype("HIP", "YCL023C", "YCL023C", strains["YCL023C"])
    dup = duplicate.perturbations[0]
    assert isinstance(dup, HeterozygousDeletionPerturbation)
    assert dup.construction is not None
    assert (dup.construction.lab, dup.construction.batch) == ("Lab 5", "chr3_1")
    assert (dup.construction.plate, dup.construction.well) == (None, None)


def test_renamed_orf_records_what_was_built_not_the_whole_current_gene() -> None:
    """#506 item 8: a RENAMED source ORF carries a ConstructedOrf with gapped history."""
    dataset = _dataset()
    for assay in ("HIP", "HOP"):
        leaf = dataset._genotype(assay, "YJL019W", "YJL018W").perturbations[0]
        assert isinstance(
            leaf, (HeterozygousDeletionPerturbation, BarcodedKanMxDeletionPerturbation)
        )
        assert leaf.perturbed_gene_name == "YJL018W"
        constructed = leaf.constructed_orf
        assert constructed is not None
        assert constructed.source_systematic_name == "YJL018W"
        assert (constructed.relation, constructed.deleted_span) == (None, None)
        assert [g.field for g in constructed.provenance_gaps] == [
            "relation",
            "deleted_span",
        ]
        assert constructed.provenance_gaps[0].resolve_with == module.SGD_LOCUS_HISTORY


# ---- provenance ----------------------------------------------------------------- #
def test_table_s5_strain_list_is_sha256_pinned() -> None:
    """#506 item 5: the regenerated list carries every HIP entry and the four strains
    the 017 CSV missed (YCR061W CL2, YJL017W CL4, YIL015C-A CL4, YCR035C MUT).
    """
    strains = load_table_s5_strains(REPO_ROOT)
    assert sum(len(entries) for entries in strains.values()) == 5996
    flagged = {
        orf for orf, entries in strains.items() if any(e.flagged for e in entries)
    }
    assert len(flagged) == 189
    assert {"YCR061W", "YJL017W", "YIL015C-A", "YCR035C"} <= flagged
    assert strains["YCR035C"][0].cluster == ""
    assert strains["YCR035C"][0].validation_result == "MUT"
    assert strains["YBR271W"][0].mutation == "Chromosome XI Aneuploidy"
    # case-insensitive key: Table S5 writes 'YAL035C-a'
    assert strains["YAL035C-A"][0].orf == "YAL035C-a"


def test_table_s5_csv_hash_mismatch_raises(tmp_path: Path, monkeypatch: Any) -> None:
    fake = tmp_path / module.TABLE_S5_STRAINS_CSV
    fake.parent.mkdir(parents=True)
    fake.write_text("orf,gene\nYAL001C,TFC3\n")
    digest = hashlib.sha256(b"orf,gene\nYAL001C,TFC3\n").hexdigest()
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"{fake} sha256 {digest} != pinned {module.TABLE_S5_STRAINS_CSV_SHA256}"
        ),
    ):
        load_table_s5_strains(tmp_path)


def test_every_sourced_value_quote_is_verbatim_in_the_mirrored_paper() -> None:
    """The audit anchor is only real if the quote is still in the sha256-pinned file."""
    import os

    data_root = os.environ.get("DATA_ROOT")
    if data_root is None:
        pytest.skip("DATA_ROOT is not set")
    paper = osp.join(
        data_root, "torchcell-library", module.CITATION_KEY, module.PAPER_MD
    )
    if not osp.exists(paper):
        pytest.skip("the literature mirror is not mounted on this machine")
    texts: dict[str, str] = {}
    for key, sourced in SOURCED_VALUES.items():
        citation_key = sourced.provenance.citation_key
        assert citation_key is not None, key
        if citation_key not in texts:
            path = osp.join(data_root, "torchcell-library", citation_key, "paper.md")
            texts[citation_key] = Path(path).read_text()
            assert hashlib.sha256(texts[citation_key].encode()).hexdigest() == (
                sourced.provenance.sha256
            ), key
        assert sourced.quote in texts[citation_key], key
    assert {sv.provenance.citation_key for sv in SOURCED_VALUES.values()} == {
        module.CITATION_KEY,
        module.GIAEVER_2014_KEY,
    }


def test_identifier_rule_reads_every_identity_field() -> None:
    from torchcell.datamodels.schema import Compound

    assert not _has_identifier(Compound(name="x"))
    assert _has_identifier(Compound(name="x", pubchem_cid=1))
    assert _has_identifier(Compound(name="x", chebi_id="CHEBI:1"))
    assert _has_identifier(Compound(name="x", inchikey="KRMDCWKBEZIMAB-UHFFFAOYSA-N"))


# ---- the ORF retention rule ------------------------------------------------------ #
class _Resolution:
    """The three fields of ``GeneNameResolution`` the loader reads."""

    def __init__(self, status: GeneNameStatus, name: str | None, feature: str | None):
        self.status = status
        self.systematic_name = name
        self.feature_type = feature


_RESOLVER: dict[str, _Resolution] = {
    "YAL001C": _Resolution(GeneNameStatus.CURRENT, "YAL001C", None),
    "YAL034C-B": _Resolution(GeneNameStatus.CURRENT, "YAL034C-B", None),
    "YAL035C-A": _Resolution(GeneNameStatus.RENAMED, "YAL034C-B", None),
    "YCL074W": _Resolution(GeneNameStatus.NON_GENE_FEATURE, "YCL074W", "pseudogene"),
    "R0010W": _Resolution(GeneNameStatus.RETIRED, "R0010W", None),
}


def _matrix(tmp_path: Path) -> str:
    """A HIP matrix with one row per resolver outcome; every kept cell non-empty."""
    rows = [
        "\t".join(HEADER),
        "\t".join(['"YAL001C"'] + ['"0.1"'] * 7),
        "\t".join(['"YAL034C-B"'] + ['"0.2"'] * 7),
        "\t".join(['"YAL035C-A"'] + ['"0.3"'] * 7),
        # column 5 (777_10, a kept HIP column) empty; column 7 is the HOP column
        "\t".join(['"YCL074W"'] + ['"0.4"'] * 4 + ['""'] + ['"0.4"'] * 2),
        "\t".join(['"R0010W"'] + ['"0.5"'] * 7),
    ]
    path = tmp_path / "HIP_scores.txt"
    path.write_text("\n".join(rows) + "\n")
    return str(path)


def _flagged_entry(orf: str) -> module.TableS5Entry:
    return module.TableS5Entry(
        orf=orf,
        table_s5_id="1",
        plate="201",
        row_column="A2",
        cluster="CL1",
        base_cluster="CL1",
        is_positional=True,
        mutation="Chromosome XI Aneuploidy",
        validation_result="",
        lab="Lab 14",
        batch="chr00_1",
        chromosome_arm="AL",
        flagged=True,
    )


def _records(tmp_path: Path) -> tuple[list[dict[str, Any]], Any]:
    import pickle

    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType

    columns, dropped = _columns("HIP")
    counts = module._BuildCounts()
    refs = {("HIP", study): {"$ref": study} for study in {c.study for c in columns}}
    records = [
        pickle.loads(value)
        for value in _dataset()._iter_records(
            _matrix(tmp_path),
            "HIP",
            {"YAL001C", "YAL034C-B", "YCL074W", "R0010W"},
            columns,
            dropped,
            refs,
            {"$ref": "pub"},
            counts,
            {"YAL034C-B": [_flagged_entry("YAL034C-B")]},
            TypeAdapter(ExperimentType).validate_python,
            lambda name: _RESOLVER[name],
            _dataset()._column_census(HEADER, "HIP", META),
        )
    ]
    return records, counts


def test_non_gene_and_retired_rows_are_dropped_with_their_status(
    tmp_path: Path,
) -> None:
    records, counts = _records(tmp_path)
    kept_rows = {
        r["experiment"]["genotype"]["perturbations"][0]["perturbed_gene_name"]
        for r in records
    }
    assert kept_rows == {"YAL001C", "YAL034C-B", "YAL035C-A"}
    dropped = {d.source_name: d for d in counts.dropped_orfs["HIP"]}
    assert set(dropped) == {"R0010W", "YCL074W"}
    assert (dropped["YCL074W"].status, dropped["YCL074W"].feature_type) == (
        "non_gene_feature",
        "pseudogene",
    )
    assert dropped["R0010W"].status == "retired"
    # the pseudogene row loses its non-empty cells in the kept HIP columns only
    n_kept_columns = len(_columns("HIP")[0])
    assert dropped["R0010W"].n_records == n_kept_columns
    assert dropped["YCL074W"].n_records == n_kept_columns - 1


def test_renamed_row_is_a_distinct_strain_of_the_current_gene(tmp_path: Path) -> None:
    records, counts = _records(tmp_path)
    assert counts.renamed_orfs["HIP"] == {"YAL035C-A": "YAL034C-B"}
    by_source = {
        r["experiment"]["genotype"]["perturbations"][0]["perturbed_gene_name"]: r[
            "experiment"
        ]["genotype"]["perturbations"][0]["systematic_gene_name"]
        for r in records
    }
    assert by_source["YAL035C-A"] == "YAL034C-B"
    assert by_source["YAL034C-B"] == "YAL034C-B"
    # Table S5 names physical strains: the listed gene's own row is flagged, the
    # renamed merged-ORF strain that now maps to that gene is not
    assert counts.flagged == {"YAL034C-B": len(_columns("HIP")[0])}


def test_detection_rule_drops_a_row_scored_in_under_half_its_columns(
    tmp_path: Path,
) -> None:
    """#506 item 2: the HIP arm of HEADER has five sensitivity columns (1, 3, 4, 5, 6);
    a row scored in 2 of them is dropped with its kept-column cells counted, a row
    scored in 3 is kept, and a row scored nowhere is counted as empty, not as a drop.
    """
    rows = [
        "\t".join(HEADER),
        "\t".join(['"YAL001C"', '"0.1"', '""', '"0.1"', '""', '""', '""', '"9"']),
        "\t".join(['"YAL034C-B"', '"0.2"', '""', '"0.2"', '""', '"0.2"', '""', '""']),
        "\t".join(['"R0010W"'] + ['""'] * 7),
    ]
    path = tmp_path / "HIP_scores.txt"
    path.write_text("\n".join(rows) + "\n")
    import pickle

    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType

    columns, dropped = _columns("HIP")
    census = _dataset()._column_census(HEADER, "HIP", META)
    assert census.detection == [1, 3, 4, 5, 6]
    counts = module._BuildCounts()
    resolver = {
        **_RESOLVER,
        "R0010W": _Resolution(GeneNameStatus.CURRENT, "R0010W", None),
    }
    records = [
        pickle.loads(value)
        for value in _dataset()._iter_records(
            str(path),
            "HIP",
            {"YAL001C", "YAL034C-B", "R0010W"},
            columns,
            dropped,
            {("HIP", s): {"$ref": s} for s in {c.study for c in columns}},
            {"$ref": "pub"},
            counts,
            {},
            TypeAdapter(ExperimentType).validate_python,
            lambda name: resolver[name],
            census,
        )
    ]
    kept = {
        r["experiment"]["genotype"]["perturbations"][0]["perturbed_gene_name"]
        for r in records
    }
    assert kept == {"YAL034C-B"}
    (undetected,) = counts.undetected["HIP"]
    assert (undetected.source_name, undetected.n_scored_columns) == ("YAL001C", 2)
    assert (undetected.n_columns, undetected.n_records) == (5, 2)
    assert counts.empty_rows["HIP"] == 1
    assert counts.detected_rows["HIP"] == 1
    # a HIP row absent from Table S5 carries a construction gap
    pert = records[0]["experiment"]["genotype"]["perturbations"][0]
    assert pert["construction"] is None
    assert "construction" in {g["field"] for g in pert["provenance_gaps"]}
    assert counts.construction == {"not_listed": 1}


def test_glucose_starvation_is_excluded_and_counted_never_built() -> None:
    """#506 item 3: CMB4019 is neither kept nor a compound drop; its columns are
    EXCLUDED under the named rule, and a Table S1 name other than the pinned one
    fails the build instead of excluding a renamed compound blind.
    """
    header = [*HEADER, '"Ad. scores for Exp. 4019_0.5_HIP_0125"']
    meta: dict[str, dict[str, str | None]] = {
        **META,
        "4019": {
            "common_name": "D-Glucose (starvation)",
            "smiles": "OCC1OC(O)C(O)C(O)C1O",
        },
    }
    kept, dropped = _dataset()._column_meta(header, "HIP", meta, _FakeTxn())
    assert "4019" not in {c.cmb for c in kept} | {c for _, c in dropped}
    census = _dataset()._column_census(header, "HIP", meta)
    assert census.excluded == [(8, "4019")]
    assert 8 in census.detection
    renamed: dict[str, dict[str, str | None]] = {
        **meta,
        "4019": {"common_name": "D-Glucose", "smiles": "C"},
    }
    with pytest.raises(ValueError, match="CMB4019 is excluded as"):
        _dataset()._column_census(header, "HIP", renamed)
    assert "never describes" in module.EXCLUSION_RULE


def _real_raw() -> Path:
    import os

    data_root = os.environ.get("DATA_ROOT")
    if data_root is None:
        pytest.skip("DATA_ROOT is not set")
    raw = Path(data_root) / "torchcell-raw" / module.CITATION_KEY
    if not (raw / "HOP_scores.txt").exists():
        pytest.skip("the Hoepfner raw mirror is not mounted on this machine")
    return raw


def test_real_header_solvent_ph_and_ic30_columns() -> None:
    """#506 items 4 and 7 on the deposited headers: 33 kept columns per arm above
    200 uM (17 CMB ids, 66 columns), 6 pH-agent columns per arm, and 64 HIP / 60 HOP of
    the kept columns with a Table S1 IC30 (293 / 291) outside 0.5x to 2x it.
    """
    raw = _real_raw()
    meta = module._load_compound_meta(str(raw / "Table_S1.xls"))
    ic30 = module._load_ic30(str(raw / "Table_S1.xls"))
    over: set[str] = set()
    for filename, assay, n_over, n_ic30, n_out in (
        ("HIP_scores.txt", "HIP", 33, 293, 64),
        ("HOP_scores.txt", "HOP", 33, 291, 60),
    ):
        with open(raw / filename) as handle:
            header = handle.readline().rstrip("\n").split("\t")
        kept, _ = _dataset()._column_meta(header, assay, meta, _FakeTxn())
        above = [c for c in kept if c.conc > module.DMSO_CEILING_UM]
        over |= {c.cmb for c in above}
        assert len(above) == n_over
        for col in above:
            assert col.env_dump["perturbations"][0]["solvent"] is None
        assert sum(1 for c in kept if c.cmb in module.PH_AGENT_CONDITIONS) == 6
        with_ic30 = [c for c in kept if c.cmb in ic30]
        assert len(with_ic30) == n_ic30
        assert (
            sum(1 for c in with_ic30 if not 0.5 <= c.conc / ic30[c.cmb] <= 2.0) == n_out
        )
        census = _dataset()._column_census(header, assay, meta)
        assert len(census.excluded) == 3  # CMB4019 at 0.25, 0.5, 0.75
        assert census.positive_control == []  # Benomyl is in the separate files
    assert len(over) == 17


def test_real_hop_detection_rule_drops_328_sgd_essential_rows() -> None:
    """#506 item 2 pinned on the deposited HOP matrix with the real S288C resolver:
    452 scored rows fall under half of the 2,923 HOP sensitivity columns, 328 of them
    SGD-essential genes (``gene_essentiality_sgd`` gene set), and 4,495 rows pass.
    The audit's 321 essential / 444 genes counted the 2026-09 cell table over the 280
    kept environments; the rule here uses every deposited column of the arm.
    """
    import json
    import os

    raw = _real_raw()
    data_root = os.environ["DATA_ROOT"]
    essential_path = (
        Path(data_root)
        / "data/torchcell/gene_essentiality_sgd/preprocess/gene_set.json"
    )
    if not essential_path.exists():
        pytest.skip("the SGD essentiality build is not on this machine")
    essential = set(json.loads(essential_path.read_text()))
    from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    meta = module._load_compound_meta(str(raw / "Table_S1.xls"))
    with open(raw / "HOP_scores.txt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
    census = _dataset()._column_census(header, "HOP", meta)
    counts = module._BuildCounts()
    out = list(
        _dataset()._iter_records(
            str(raw / "HOP_scores.txt"),
            "HOP",
            module._load_sgd_genes(data_root),
            [],
            [],
            {},
            {"$ref": "pub"},
            counts,
            {},
            lambda _: None,
            genome.resolve_gene_name,
            census,
        )
    )
    assert out == []
    undetected = counts.undetected["HOP"]
    assert len(census.detection) == 2923
    assert len(undetected) == 452
    assert sum(1 for row in undetected if row.systematic_name in essential) == 328
    assert counts.detected_rows["HOP"] == 4495


# ---- Dryad retrieval (no network) ------------------------------------------------ #
_URL = "https://datadryad.org/downloads/file_stream/4834609"


class _Response(requests.Response):
    """A streamed response: its content type, its HTML text and whether it was closed."""

    def __init__(self, content_type: str, text: str = "", label: str = "") -> None:
        super().__init__()
        self.headers["content-type"] = content_type
        self._content = text.encode()
        self.encoding = "utf-8"
        self.label = label
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _Session(requests.Session):
    """Records every GET and serves the queued responses in order."""

    def __init__(self, responses: list[_Response]) -> None:
        super().__init__()
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, *args: Any, **kwargs: Any) -> requests.Response:
        self.calls.append((args[0], kwargs))
        return self.responses.pop(0)


def _label(response: requests.Response) -> str:
    """The label of a served fake response (a guard, not a cast)."""
    assert isinstance(response, _Response)
    return response.label


def _challenge_page(with_base: bool) -> str:
    challenge = {"challenge": {"id": "c1", "randomData": "abc", "difficulty": 1}}
    page = (
        '<script id="anubis_challenge" type="application/json">'
        f"{json.dumps(challenge)}</script>"
    )
    if with_base:
        page += (
            '<script id="anubis_base_prefix" type="application/json">"/base"</script>'
        )
    return page


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """``time.sleep`` replaced by a recorder, so the back-off costs nothing."""
    calls: list[float] = []
    monkeypatch.setattr(time, "sleep", calls.append)
    return calls


def test_dryad_get_returns_a_non_html_body_from_the_first_request(
    sleeps: list[float],
) -> None:
    """A file body (not ``text/html``) is returned as is: one GET, no sleep."""
    session = _Session([_Response("application/octet-stream", label="body")])
    response = module._dryad_get(session, _URL)
    assert _label(response) == "body"
    assert session.calls == [(_URL, {"timeout": 180, "stream": True})]
    assert sleeps == []


@pytest.mark.parametrize(
    ("with_base", "pass_url"),
    [
        (
            True,
            "https://datadryad.org/base/.within.website/x/cmd/anubis/api/pass-challenge",
        ),
        (
            False,
            "https://datadryad.org/.within.website/x/cmd/anubis/api/pass-challenge",
        ),
    ],
)
def test_dryad_get_solves_the_anubis_challenge_and_passes_it(
    with_base: bool, pass_url: str, sleeps: list[float]
) -> None:
    """The challenge page is closed, the proof of work solved (nonce 26 is the first whose
    sha256("abc26") starts with a zero nibble) and sent to the pass-challenge endpoint
    under the page's base prefix, redirecting to the file URL; the cleared body returns.
    """
    page = _Response("text/html; charset=utf-8", _challenge_page(with_base))
    session = _Session([page, _Response("application/octet-stream", label="body")])
    response = module._dryad_get(session, _URL)
    assert _label(response) == "body"
    assert page.closed
    assert session.calls == [
        (_URL, {"timeout": 180, "stream": True}),
        (
            pass_url,
            {
                "params": {
                    "id": "c1",
                    "response": (
                        "0d56d5ce616422904a584cf3735a45ae611817d96c4717b17369ef6778025848"
                    ),
                    "nonce": 26,
                    "redir": _URL,
                    "elapsedTime": 100,
                },
                "stream": True,
                "allow_redirects": True,
                "timeout": 1800,
            },
        ),
    ]
    assert sleeps == []


def test_dryad_get_retries_a_throttled_page_and_a_cleared_page_that_is_still_html(
    sleeps: list[float],
) -> None:
    """A page with no challenge is a throttle: close, sleep 15 s, GET again. A challenge
    whose pass request still answers HTML is closed, then sleep 15 s and GET again; the
    third file GET returns the body. Five requests, two sleeps.
    """
    throttled = _Response("text/html", "<html>slow down</html>")
    page = _Response("text/html", _challenge_page(False))
    still_html = _Response("text/html", "<html>again</html>")
    session = _Session(
        [throttled, page, still_html, _Response("application/octet-stream", label="b")]
    )
    response = module._dryad_get(session, _URL)
    assert _label(response) == "b"
    assert [throttled.closed, page.closed, still_html.closed] == [True, True, True]
    assert [url for url, _ in session.calls] == [
        _URL,
        _URL,
        "https://datadryad.org/.within.website/x/cmd/anubis/api/pass-challenge",
        _URL,
    ]
    assert sleeps == [15, 15]


def test_dryad_get_gives_up_after_eighty_throttled_pages(sleeps: list[float]) -> None:
    """80 throttled pages (80 GETs, 80 sleeps of 15 s) end in a named refusal."""
    session = _Session([_Response("text/html", "<html>busy</html>") for _ in range(80)])
    with pytest.raises(
        RuntimeError, match=f"^could not clear Anubis challenge for {re.escape(_URL)}$"
    ):
        module._dryad_get(session, _URL)
    assert len(session.calls) == 80
    assert sleeps == [15] * 80


class _ChunkedResponse:
    """A streamed body served as fixed chunks, one of them an empty keep-alive chunk."""

    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.closed = False
        self.chunk_sizes: list[int] = []

    def iter_content(self, chunk_size: int) -> list[bytes]:
        self.chunk_sizes.append(chunk_size)
        return self.chunks

    def close(self) -> None:
        self.closed = True


def test_fetch_from_dryad_skips_empty_chunks_and_verifies_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """``b"ab"``, ``b""``, ``b"c"`` write ``abc`` (the empty chunk is skipped) in 1 MiB
    reads through a session carrying the browser User-Agent; the digest ba7816bf... is
    the pin, so the file is kept and "Wrote <dest> (sha256 verified)" is logged. The
    same body against another pin raises ``RawSha256MismatchError`` naming the URL and
    both digests, and the refused body never reaches ``dest`` (it streams into a
    ``.partial`` sibling that is removed), so the earlier verified file is untouched.
    """
    sessions: list[Any] = []
    responses: list[_ChunkedResponse] = []

    def fake_get(session: Any, url: str) -> _ChunkedResponse:
        sessions.append((session.headers["User-Agent"], url))
        responses.append(_ChunkedResponse([b"ab", b"", b"c"]))
        return responses[-1]

    monkeypatch.setattr(module, "_dryad_get", fake_get)
    abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    dest = tmp_path / "HOP_scores.txt"
    with caplog.at_level(logging.INFO, logger=module.log.name):
        _dataset()._fetch_from_dryad(
            "HOP_scores.txt", {"url": _URL, "sha256": abc}, str(dest)
        )
    assert dest.read_bytes() == b"abc"
    assert sessions == [(module._UA, _URL)]
    assert responses[0].chunk_sizes == [1 << 20]
    assert responses[0].closed
    assert [r.getMessage() for r in caplog.records if r.name == module.log.name] == [
        f"Downloading Hoepfner2014 HOP_scores.txt from {_URL}",
        f"Wrote {dest} (sha256 verified)",
    ]
    with pytest.raises(RawSha256MismatchError) as err:
        _dataset()._fetch_from_dryad(
            "HOP_scores.txt", {"url": _URL, "sha256": "0" * 64}, str(dest)
        )
    assert str(err.value) == (
        f"sha256 mismatch for {_URL}: expected {'0' * 64}, observed {abc}"
    )
    assert dest.read_bytes() == b"abc"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["HOP_scores.txt"]


# ---- sign convention and the dosage-duration confound ----------------------------- #
def _signed_matrix(tmp_path: Path) -> str:
    """One YAL001C row: -3.0 in the HIP column 1 and the HOP column 7; the HIP
    columns 4 (dropped compound) and 6 (unencodable) carry 1.0 so the row is scored in 3
    of 5 HIP columns and passes the detection rule without adding a kept-column record.
    """
    cells = ['""'] * 7
    cells[0] = '"-3.0"'
    cells[3] = '"1.0"'
    cells[5] = '"1.0"'
    cells[6] = '"-3.0"'
    path = tmp_path / "signed.txt"
    path.write_text("\n".join(["\t".join(HEADER), "\t".join(['"YAL001C"', *cells])]))
    return str(path)


def _stream(tmp_path: Path, assay: str) -> list[dict[str, Any]]:
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType

    columns, dropped = _columns(assay)
    return [
        pickle.loads(value)
        for value in _dataset()._iter_records(
            _signed_matrix(tmp_path),
            assay,
            {"YAL001C"},
            columns,
            dropped,
            {(assay, c.study): {"$ref": c.study} for c in columns},
            {"$ref": "pub"},
            module._BuildCounts(),
            {},
            TypeAdapter(ExperimentType).validate_python,
            lambda name: _RESOLVER[name],
            _dataset()._column_census(HEADER, assay, META),
        )
    ]


def test_both_assays_store_the_deposited_sign_negative_is_hypersensitive(
    tmp_path: Path,
) -> None:
    """HIP and HOP cells of -3.0 are both stored as -3.0: no reorientation. The declared
    units put the sick side at negative ("negative = hypersensitive"), the opposite of
    Hillenmeyer's positive-is-sick, so a pooled target must flip one of them (memory
    ``chemogenomic-response-polarity-splits``); the loader itself does not.
    """
    hip = _stream(tmp_path, "HIP")
    hop = _stream(tmp_path, "HOP")
    assert [r["experiment"]["phenotype"]["environment_response"] for r in hip] == [-3.0]
    assert [r["experiment"]["phenotype"]["environment_response"] for r in hop] == [-3.0]
    assert [r["experiment"]["phenotype"]["units"] for r in hip + hop] == [
        module.MEASUREMENT_UNITS
    ] * 2
    assert module.MEASUREMENT_UNITS == (
        "adjusted MADL sensitivity score = (r_L - med(r_L)) / MAD(r_L) over all pool "
        "strains, r_L = log ratio of treated vs control strain abundance (Hoepfner "
        "2014); negative = hypersensitive, positive = resistant; 0 = growth equal to "
        "control. A companion gene-wise z-score column is deposited per experiment but "
        "not stored here."
    )
    assert [
        r["experiment"]["genotype"]["perturbations"][0]["perturbation_type"]
        for r in hip + hop
    ] == ["heterozygous_deletion", "barcoded_kanmx_deletion"]


def test_hip_and_hop_environments_differ_only_in_exposure_and_culture() -> None:
    """For CMB 3 at 50 uM in study 0077, the HIP and HOP environments differ in exactly
    the exposure and culture fields: ``duration_hours`` (None vs 16.0),
    ``duration_generations`` (None vs 5.0), ``provenance_gaps`` (the HIP duration gaps),
    ``culture_format`` (inoculum and endpoint) and ``pre_culture`` (log phase vs thaw
    recovery). Both references sit on the same BY4743 ``StrainReferenceGenome``, so the
    dosage arm is confounded with the exposure: no cell is paired across the arms within
    one environment (memory ``033-env-chemgen-pooled-build``).
    """
    hip = next(col for col in _columns("HIP")[0] if col.index == 1).env_dump
    hop = next(col for col in _columns("HOP")[0] if col.index == 7).env_dump
    differing = sorted(key for key in hip if hip[key] != hop[key])
    assert differing == [
        "culture_format",
        "duration_generations",
        "duration_hours",
        "pre_culture",
        "provenance_gaps",
    ]
    assert (hip["duration_hours"], hop["duration_hours"]) == (None, 16.0)
    assert (hip["duration_generations"], hop["duration_generations"]) == (None, 5.0)
    dataset = _dataset()
    assert (
        dataset._reference("HIP", "0077").genome_reference.model_dump()
        == dataset._reference("HOP", "0077").genome_reference.model_dump()
        == module.hoepfner_reference_genome().model_dump()
    )


# ---- typed round trip, hooks, default genome, main ------------------------------- #
def test_a_spliced_record_retypes_through_the_environment_response_classes(
    tmp_path: Path,
) -> None:
    """A streamed HIP record with its environment pointer spliced back to the column's
    inline environment, its reference and publication resolved, rebuilds through
    ``StrainEnvironmentResponseExperiment`` / ``...Reference`` and dumps back unchanged.
    """
    from torchcell.datamodels.schema import (
        Publication,
        StrainEnvironmentResponseExperiment,
        StrainEnvironmentResponseExperimentReference,
    )

    record = _stream(tmp_path, "HIP")[0]
    column = next(col for col in _columns("HIP")[0] if col.index == 1)
    dataset = _dataset()
    item = {
        "experiment": {**record["experiment"], "environment": column.env_dump},
        "reference": dataset._reference("HIP", "0077").model_dump(),
        "publication": Publication(
            doi=module.DOI, doi_url=f"https://doi.org/{module.DOI}"
        ).model_dump(),
    }
    typed = dataset.transform_item(item)
    assert type(typed["experiment"]) is StrainEnvironmentResponseExperiment
    assert type(typed["reference"]) is StrainEnvironmentResponseExperimentReference
    assert typed["experiment"].model_dump() == item["experiment"]
    assert typed["reference"].model_dump() == item["reference"]
    assert typed["publication"].model_dump() == item["publication"]


def test_generic_hooks_are_inert() -> None:
    """``preprocess_raw`` returns what it was given; ``create_experiment`` raises a bare
    ``NotImplementedError`` (records are built inline in ``process``).
    """
    dataset = _dataset()
    frame = {"Systematic Name": ["YAL001C"]}
    assert dataset.preprocess_raw(frame) == {"Systematic Name": ["YAL001C"]}
    with pytest.raises(NotImplementedError) as info:
        dataset.create_experiment()
    assert str(info.value) == ""


def test_resolver_without_a_genome_opens_one_read_only_s288c_genome_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``genome=None`` the resolver loads ``.env`` (stubbed), builds one
    ``SCerevisiaeGenome`` under ``$DATA_ROOT`` with ``overwrite=False`` and returns its
    ``resolve_gene_name``; a second call reuses the stored genome.
    """
    import torchcell.sequence.genome.scerevisiae as scerevisiae_package

    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    built: list[dict[str, Any]] = []

    class _Genome:
        def __init__(self, **kwargs: Any) -> None:
            built.append(kwargs)

        def resolve_gene_name(self, name: str) -> str:
            return f"resolved:{name}"

    monkeypatch.setattr(scerevisiae_package, "SCerevisiaeGenome", _Genome)
    dataset = _dataset()
    dataset.genome = None
    first = dataset._resolver()
    second = dataset._resolver()
    assert built == [
        {
            "genome_root": f"{tmp_path}/data/sgd/genome",
            "go_root": f"{tmp_path}/data/go",
            "overwrite": False,
        }
    ]
    assert (first("YAL001C"), second("YBR001C")) == (
        "resolved:YAL001C",
        "resolved:YBR001C",
    )


def test_main_builds_the_dataset_under_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main`` constructs the dataset at ``$DATA_ROOT/data/torchcell/
    env_chemgen_hoepfner2014`` with no genome (the resolver opens one at build time) and
    prints its length and first item; the class is a recorder.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    calls: list[dict[str, Any]] = []

    class _Recorder:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

        def __len__(self) -> int:
            return 13

        def __getitem__(self, index: int) -> str:
            return f"item[{index}]"

    monkeypatch.setattr(module, "EnvChemgenHoepfner2014Dataset", _Recorder)
    module.main()
    assert calls == [{"root": f"{tmp_path}/data/torchcell/env_chemgen_hoepfner2014"}]
    assert capsys.readouterr().out == "len = 13\nitem[0]\n"


# ---- 2026.10.06 (Phase 21): the remaining one-line branches ----------------------- #
def _entry(lab: str, batch: str, plate: str = "201", well: str = "A2") -> Any:
    return module.TableS5Entry(
        orf="YAL001C",
        table_s5_id="1",
        plate=plate,
        row_column=well,
        cluster="",
        base_cluster="",
        is_positional=False,
        mutation="",
        validation_result="",
        lab=lab,
        batch=batch,
        chromosome_arm="",
        flagged=False,
    )


def test_strain_construction_refuses_entries_that_disagree_on_lab_or_batch() -> None:
    """One entry keeps plate and well; two agreeing entries drop them (the deposited
    row does not say which copy was pooled); entries whose lab differs, and entries
    whose batch differs, each raise with the sorted sets.
    """
    one = module._strain_construction([_entry("Lab 1", "b1")])
    assert one.model_dump() == {
        "lab": "Lab 1",
        "batch": "b1",
        "plate": "201",
        "well": "A2",
        "strain_accession": None,
    }
    two = module._strain_construction(
        [_entry("Lab 1", "b1"), _entry("Lab 1", "b1", "202", "B3")]
    )
    assert two.model_dump() == {
        "lab": "Lab 1",
        "batch": "b1",
        "plate": None,
        "well": None,
        "strain_accession": None,
    }
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Table S5 entries for YAL001C disagree on lab / batch: "
            "['Lab 1', 'Lab 2'] / ['b1']"
        ),
    ):
        module._strain_construction([_entry("Lab 2", "b1"), _entry("Lab 1", "b1")])
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Table S5 entries for YAL001C disagree on lab / batch: "
            "['Lab 1'] / ['b1', 'b2']"
        ),
    ):
        module._strain_construction([_entry("Lab 1", "b2"), _entry("Lab 1", "b1")])


def test_load_ic30_reads_digit_ids_from_the_two_moa_sheets(tmp_path: Path) -> None:
    """Both MoA sheets contribute and any other sheet is ignored (``Other``'s CMB 77
    never appears). Known-MoA rows: CMB 12 (int) -> "12"; " 45 " (string) -> "45";
    "CMB7" (not digits), a blank id and a blank IC30 are skipped. Novel-MoA: 99 -> 4.0.
    Expected {"12": 3.0, "45": 2.5, "99": 4.0}. With the novel sheet's IC30 column
    renamed, that sheet is skipped whole and only the known-MoA ids remain.
    """
    import openpyxl

    workbook = openpyxl.Workbook()
    known = workbook.active
    known.title = "Reference Substances known MoA"
    known.append(["CMB ID", "IC30 (uM)"])
    for row in ([12, 3.0], [" 45 ", 2.5], ["CMB7", 1.0], [None, 9.0], [13, None]):
        known.append(row)
    novel = workbook.create_sheet("Substances novel MoA")
    novel.append(["CMB ID", "IC30 (uM)"])
    novel.append([99, 4.0])
    other = workbook.create_sheet("Other")
    other.append(["CMB ID", "IC30 (uM)"])
    other.append([77, 7.0])
    path = tmp_path / "Table_S1.xlsx"
    workbook.save(path)
    assert module._load_ic30(str(path)) == {"12": 3.0, "45": 2.5, "99": 4.0}
    novel.cell(row=1, column=2, value="MoA")
    workbook.save(path)
    assert module._load_ic30(str(path)) == {"12": 3.0, "45": 2.5}


def test_column_census_records_the_positive_control_columns() -> None:
    """A CMB991 column (``POSITIVE_CONTROL_CMB``) is a detection column AND a positive
    control; its z-score companion and the other assay's column are not counted.
    """
    header = [
        *HEADER,
        '"Ad. scores for Exp. 991_5_HIP_0077"',
        '"Ad. scores for Exp. 991_5_HIP_0077 z-score"',
        '"Ad. scores for Exp. 991_5_HOP_0077"',
    ]
    meta: dict[str, dict[str, str | None]] = {
        **META,
        "991": {"common_name": "Control", "smiles": AMITRIPTYLINE_SMILES},
    }
    census = _dataset()._column_census(header, "HIP", meta)
    assert census.positive_control == [8]
    assert census.detection == [1, 3, 4, 5, 6, 8]
    assert census.cmbs == {"3", "409", "777", "888", "991"}
    assert census.unencodable == [(6, "888")]


def test_iter_records_counts_excluded_cells_and_skips_columns_past_a_short_row(
    tmp_path: Path,
) -> None:
    """HEADER plus an excluded CMB4019 column at index 8. Kept HIP columns are 1, 3, 5
    (409 is a compound drop, 888 unencodable). Detection columns: 1, 3, 4, 5, 6, 8.

    YAL001C fills every cell: three records (0.1 each, screens 0077, 0091, 0077), and
    its index-4, index-6 and index-8 cells count as one dropped (409), one unencodable
    and one excluded (4019) cell. YAL034C-B is a SHORT row of five fields (indices 0-4:
    0.2, "", 0.3, 0.4): it is scored in 1, 3 and 4, i.e. 3 >= 0.5 * 6, so it is kept;
    columns 1 and 3 give records 0.2 and 0.3 and column 5 lies past the row and is
    skipped (line 2147), so no record is invented for it; its 409 cell counts as a
    drop and it has no excluded cell.
    """
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType

    header = [*HEADER, '"Ad. scores for Exp. 4019_0.5_HIP_0125"']
    meta: dict[str, dict[str, str | None]] = {
        **META,
        "4019": {
            "common_name": "D-Glucose (starvation)",
            "smiles": "OCC1OC(O)C(O)C(O)C1O",
        },
    }
    columns, dropped = _dataset()._column_meta(header, "HIP", meta, _FakeTxn())
    assert ([c.index for c in columns], dropped) == ([1, 3, 5], [(4, "409")])
    census = _dataset()._column_census(header, "HIP", meta)
    path = tmp_path / "HIP_scores.txt"
    path.write_text(
        "\n".join(
            [
                "\t".join(header),
                "\t".join(['"YAL001C"'] + ['"0.1"'] * 7 + ['"0.7"']),
                "\t".join(['"YAL034C-B"', '"0.2"', '""', '"0.3"', '"0.4"']),
            ]
        )
        + "\n"
    )
    counts = module._BuildCounts()
    records = [
        pickle.loads(value)
        for value in _dataset()._iter_records(
            str(path),
            "HIP",
            {"YAL001C", "YAL034C-B"},
            columns,
            dropped,
            {("HIP", s): {"$ref": s} for s in {c.study for c in columns}},
            {"$ref": "pub"},
            counts,
            {},
            TypeAdapter(ExperimentType).validate_python,
            lambda name: _RESOLVER[name],
            census,
        )
    ]
    assert [
        (
            r["experiment"]["genotype"]["perturbations"][0]["perturbed_gene_name"],
            r["experiment"]["phenotype"]["environment_response"],
            r["experiment"]["phenotype"]["screen_id"],
        )
        for r in records
    ] == [
        ("YAL001C", 0.1, "0077"),
        ("YAL001C", 0.1, "0091"),
        ("YAL001C", 0.1, "0077"),
        ("YAL034C-B", 0.2, "0077"),
        ("YAL034C-B", 0.3, "0091"),
    ]
    assert dict(counts.excluded) == {("HIP", "4019"): 1}
    assert dict(counts.dropped) == {("HIP", "409"): 2}
    assert counts.unencodable["HIP"] == 1
    assert counts.detected_rows["HIP"] == 2
