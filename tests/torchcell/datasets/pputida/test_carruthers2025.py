# tests/torchcell/datasets/pputida/test_carruthers2025.py
# [[tests.torchcell.datasets.pputida.test_carruthers2025]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/pputida/test_carruthers2025.py
"""The Carruthers 2025 P. putida CRISPRi isoprenol loaders.

Synthetic tests run everywhere: they exercise the construct parser, the titer and
proteome aggregations, the chassis and pathway builders against a recording fake genome,
the retention arithmetic, and the two schema behaviors this loader had to design around
(``ConcentrationUnit`` has no ``mg/L`` member, and a ``CultureEnvironment`` placed in
``ProductTiterExperiment.environment`` is dumped without its culture-protocol slots).

The ``@pytest.mark.data`` tests read the real ``$DATA_ROOT``: they pin the raw mirror's
recorded digests against the module constants, reconcile the chassis symbols and the
guide targets against the deposited KT2440 annotation, and run L0 to L4 over both built
LMDBs. They are skipped unless the mirror, the KT2440 tier cache and the built stores
are all present.

Derived expectations for the pinned Source Data workbook, every one measured on
sha256 ``1b3a7ab5...``: 1,506 released cultures, of which 90 are controls (18 in DBTL0
and 12 in each of DBTL1-6); 465 ``(construct, cycle)`` strains, 458 with three
replicates and 7 with six; 121 distinct guide targets, all current KT2440 locus tags;
and 20 proteome samples over 1,501 protein keys, of which 1,424 survive into a record's
abundance map.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import os.path as osp
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd
import pytest
from pydantic import ValidationError

import torchcell.datasets.pputida.carruthers2025 as c25
from torchcell.datamodels.media import M9_NREL_CARRUTHERS2025
from torchcell.datamodels.schema import (
    AlleleEdit,
    BacterialDeletionPerturbation,
    BacterialProteinAbundanceExperiment,
    ConcentrationUnit,
    CultureEnvironment,
    EndpointRule,
    Genotype,
    ProductTiterExperiment,
    ProductTiterExperimentReference,
    ProteinAbundancePhenotype,
    SampleUnit,
    SmallMoleculePerturbation,
    UncertaintyType,
)
from torchcell.datasets.bacteria_common import LocusTagResolutionError
from torchcell.literature.manifest import ROLE_SI_DATA, Manifest, RetrievalMethod
from torchcell.verification.report import Level, VerificationReport


# --------------------------------------------------------------------------- #
# Synthetic: the construct parser
# --------------------------------------------------------------------------- #
def test_parse_construct_splits_a_multi_guide_array_into_its_locus_tags() -> None:
    """An array name is its ``PP_`` tags in released order, with no residue."""
    tags, extras = c25.parse_construct("PP_0528_PP_0751_PP_0812_PP_0815")
    assert tags == ("PP_0528", "PP_0751", "PP_0812", "PP_0815")
    assert extras == ()


def test_parse_construct_keeps_a_non_targeting_filler_out_of_the_tags() -> None:
    """``NT1`` occupies an array position but perturbs no gene, so it is not a target."""
    assert c25.parse_construct("PP_1607_NT1") == (("PP_1607",), ("NT1",))
    assert c25.parse_construct("PP_4194_NT2") == (("PP_4194",), ("NT2",))


def test_parse_construct_on_a_single_guide_name() -> None:
    """A single-guide construct is one tag and nothing else."""
    assert c25.parse_construct("PP_0815") == (("PP_0815",), ())


# --------------------------------------------------------------------------- #
# Synthetic: the titer phenotype
# --------------------------------------------------------------------------- #
def test_titer_phenotype_stores_the_mean_with_the_sample_sd_and_derives_the_se() -> (
    None
):
    """The mean, the sample SD as ``sample_sd``, and SE = SD / sqrt(n)."""
    pheno = c25.titer_phenotype([100.0, 110.0, 120.0], is_reference=False)
    assert pheno.titer == pytest.approx(110.0)
    assert pheno.titer_uncertainty == pytest.approx(10.0)
    assert pheno.titer_uncertainty_type is UncertaintyType.sample_sd
    assert pheno.n_samples == 3
    assert pheno.sample_unit is SampleUnit.biological_replicate
    assert pheno.titer_se == pytest.approx(10.0 / math.sqrt(3))
    assert pheno.quantification_method == "GC-FID"


def test_titer_phenotype_stores_the_source_number_in_the_identical_microgram_unit() -> (
    None
):
    """mg/L is stored verbatim as ug/mL, which is the same quantity exactly."""
    pheno = c25.titer_phenotype([818.6499, 773.4597, 762.3653], is_reference=False)
    assert pheno.titer_unit is ConcentrationUnit.ug_per_ml
    assert "mg/L" not in {unit.value for unit in ConcentrationUnit}
    assert pheno.titer == pytest.approx((818.6499 + 773.4597 + 762.3653) / 3)


def test_titer_phenotype_gaps_the_yield_and_productivity_the_campaign_never_released() -> (
    None
):
    """Four typed absences, not four silent Nones."""
    pheno = c25.titer_phenotype([1.0, 2.0], is_reference=True)
    assert pheno.gapped_fields() == {
        "product_yield",
        "product_yield_unit",
        "productivity",
        "productivity_unit",
    }
    assert pheno.product_yield is None
    assert pheno.productivity is None


def test_titer_phenotype_refuses_a_single_replicate() -> None:
    """One culture has no sample SD, so it cannot carry the paper's uncertainty type."""
    with pytest.raises(RuntimeError, match="at least two replicates"):
        c25.titer_phenotype([100.0], is_reference=False)


def test_the_product_is_the_compound_layers_isoprenol_with_its_identity() -> None:
    """The product goes through the shared resolver, whose table now carries isoprenol.

    The row (PubChem CID 12988, curated on
    ``compound_identity_inputs/bioproduction.txt``) is what makes this titer and a
    tolerance screen dosing the same molecule one compound node, rather than two
    joinable only by name. Before it, the product carried a typed gap on ``inchikey``.
    """
    product = c25.isoprenol_product()
    assert product.name == "isoprenol"
    assert product.inchikey == "CPJRRXSHAYUTGL-UHFFFAOYSA-N"
    assert (product.pubchem_cid, product.chebi_id) == (12988, "CHEBI:62898")
    assert product.gapped_fields() == set()


# --------------------------------------------------------------------------- #
# Synthetic: the pathway and the guides
# --------------------------------------------------------------------------- #
def test_the_five_pathway_genes_carry_the_verbatim_plasmid_tokens_and_their_organisms() -> (
    None
):
    """Identifier, host namespace, organism and promoter, all from mirrored bytes."""
    by_token = {p.systematic_gene_name: p for p in c25.pathway_perturbations()}
    assert set(by_token) == {"MvaSEf", "MvaEEf", "MKMm", "PMDScHKQ", "AphA"}
    assert by_token["PMDScHKQ"].source_organism == "Saccharomyces cerevisiae"
    assert by_token["PMDScHKQ"].variant == "HKQ"
    assert by_token["MvaSEf"].source_organism == "Enterococcus faecalis"
    assert by_token["MKMm"].source_organism == "Methanosarcina mazei"
    assert by_token["AphA"].source_organism == "Escherichia coli"
    assert {p.promoter_name for p in (by_token["MvaSEf"], by_token["MvaEEf"])} == {
        "PBAD"
    }
    assert {by_token[t].promoter_name for t in ("MKMm", "PMDScHKQ", "AphA")} == {
        "Ptrc1-O"
    }
    for pert in by_token.values():
        assert pert.is_heterologous is True
        assert pert.localization == "episomal_plasmid"
        assert pert.construct_name == "pIY670"
        assert pert.copy_number == 1.0
        assert pert.gene_namespace == "pputida_kt2440_locus_tag"
        assert pert.pathway_name == "isoprenol via the IPP-bypass mevalonate pathway"


def test_every_pathway_token_appears_in_the_quoted_plasmid_description() -> None:
    """The stored identifiers cannot drift away from the quote they come from."""
    parts = str(c25.PIY670_PARTS.value)
    for gene in c25.PATHWAY_GENES:
        assert str(gene["token"]) in parts


def test_pathway_perturbations_refuses_a_token_the_quoted_description_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A quote edit that drops a part is caught, not silently stored."""
    monkeypatch.setattr(
        c25,
        "PATHWAY_GENES",
        (*c25.PATHWAY_GENES, {**c25.PATHWAY_GENES[0], "token": "Nope"}),
    )
    with pytest.raises(RuntimeError, match="is not in the quoted pIY670 description"):
        c25.pathway_perturbations()


def test_a_crispri_perturbation_carries_dcas9_one_guide_and_no_released_spacer() -> (
    None
):
    """The spacers were never released, so ``guide_sequence`` stays None."""
    pert = c25.crispri_perturbation("PP_0815", "PP_0815")
    assert pert.perturbation_type == "bacterial_crispr_interference"
    assert pert.gene_namespace == "pputida_kt2440_locus_tag"
    assert pert.crispr is not None
    assert pert.crispr.effector == "dCas9"
    assert pert.crispr.guide_sequence is None
    assert pert.crispr.n_guides == 1
    assert pert.expression_direction == "decreased"


def test_a_crispri_perturbation_refuses_a_yeast_systematic_name() -> None:
    """The bacterial leaf keeps the identifier families apart."""
    with pytest.raises(ValidationError):
        c25.crispri_perturbation("YAL001C", "TFC3")


# --------------------------------------------------------------------------- #
# Synthetic: the environment, and why it is not a CultureEnvironment
# --------------------------------------------------------------------------- #
def test_the_production_environment_is_the_sourced_medium_temperature_and_duration() -> (
    None
):
    """M9-NREL at 24 C for 48 h, aerobic, with the 2 g/L L-arabinose inducer.

    A ``CultureEnvironment`` since the titer slot was narrowed, so the vessel it was
    grown in travels on the record instead of only in ``CULTURE_FORMAT``.
    """
    env = c25.production_environment()
    assert type(env) is CultureEnvironment
    assert env.culture_format is not None
    assert env.culture_format.vessel == "48-well BioLector flower plate"
    assert env.culture_format.working_volume_ul == pytest.approx(1500.0)
    assert env.culture_format.shaking_rpm == pytest.approx(1000.0)
    assert env.culture_format.endpoint is EndpointRule.fixed_duration
    assert [sv.quote for sv in env.culture_format.provenance] == [
        c25.CULTURE_FORMAT.quote
    ]
    assert env.media is M9_NREL_CARRUTHERS2025
    assert env.temperature is not None
    assert env.temperature.value == pytest.approx(24.0)
    assert env.duration_hours == pytest.approx(48.0)
    assert env.aerobicity == "aerobic"
    (inducer,) = env.perturbations
    assert isinstance(inducer, SmallMoleculePerturbation)
    assert inducer.compound.name == "L-arabinose"
    assert inducer.concentration.value == pytest.approx(2.0)
    assert inducer.concentration.unit is ConcentrationUnit.g_per_l


def test_the_titer_slot_keeps_the_protocol_and_the_proteome_slot_still_drops_it() -> (
    None
):
    """Both halves of the serialization rule, on the loader's own environment.

    pydantic serializes a field by its DECLARED type.
    ``ProductTiterExperiment.environment`` is now ``CultureEnvironment``, so the vessel,
    volume and shaking are dumped; ``BacterialProteinAbundanceExperiment.environment``
    is still ``Environment``, so the same object there keeps them as python attributes
    and dumps without them, silently. That is why the narrowing had to be per family.
    """
    env = c25.production_environment()
    perturbations: list[Any] = list(c25.pathway_perturbations())
    titer = ProductTiterExperiment(
        dataset_name="probe",
        genotype=Genotype(perturbations=perturbations),
        environment=env,
        phenotype=c25.titer_phenotype([1.0, 2.0, 3.0], is_reference=False),
    )
    culture = titer.model_dump()["environment"]["culture_format"]
    assert culture["vessel"] == "48-well BioLector flower plate"
    assert culture["working_volume_ul"] == pytest.approx(1500.0)
    assert ProductTiterExperiment.model_validate(titer.model_dump()) == titer

    proteome = BacterialProteinAbundanceExperiment(
        dataset_name="probe",
        genotype=Genotype(perturbations=perturbations),
        environment=env,
        phenotype=ProteinAbundancePhenotype(
            protein_abundance={"PP_0001": 1.0},
            n_replicates={"PP_0001": 3},
            measurement_type="dia_top3_peptide_counts_mean",
        ),
    )
    assert env.culture_format is not None
    assert "culture_format" not in proteome.model_dump()["environment"]


# --------------------------------------------------------------------------- #
# Synthetic: the proteome aggregation
# --------------------------------------------------------------------------- #
def test_the_proteome_aggregation_gives_the_mean_the_se_and_the_replicate_count() -> (
    None
):
    """SE is the sample SD over sqrt(n); a single replicate has no SE."""
    abundance, se, n_reps = c25.ProteomeCarruthers2025Dataset._aggregate(
        {"PP_0001": [10.0, 20.0, 30.0], "PP_0002": [5.0]}, "sample"
    )
    assert abundance == {"PP_0001": pytest.approx(20.0), "PP_0002": pytest.approx(5.0)}
    assert se["PP_0001"] == pytest.approx(10.0 / math.sqrt(3))
    assert math.isnan(se["PP_0002"])
    assert n_reps == {"PP_0001": 3, "PP_0002": 1}


def test_a_released_zero_abundance_is_kept_rather_than_imputed() -> None:
    """A 0 is the number the source reports for an undetected protein."""
    abundance, _, n_reps = c25.ProteomeCarruthers2025Dataset._aggregate(
        {"PP_0001": [0.0, 0.0, 0.0]}, "sample"
    )
    assert abundance["PP_0001"] == 0.0
    assert n_reps["PP_0001"] == 3


def test_the_proteome_sample_pattern_reads_a_target_out_of_every_released_sample_name() -> (
    None
):
    """The three released spellings of an off-target sample all yield their tag."""
    for sample, tag in (
        ("JBEI_OTS_PP_0378_48hr", "PP_0378"),
        ("JBEI_OTS_PP_0977_1_P4_48hr", "PP_0977"),
        ("JBEI_OTS_PP_3416_P4_48hr", "PP_3416"),
    ):
        match = c25.PROTEOME_SAMPLE_RE.match(sample)
        assert match is not None
        assert match.group("tag") == tag


# --------------------------------------------------------------------------- #
# Synthetic: retention arithmetic and the mirror helpers
# --------------------------------------------------------------------------- #
def test_build_accounting_refuses_a_drop_no_rule_accounts_for() -> None:
    """A record missing from the build with no rule naming it is a bug, not a build."""
    accounting = c25.BuildAccounting(
        dataset="d",
        source_rows=10,
        control_rows=0,
        candidate_records=5,
        kept_records=4,
        dropped_records=1,
        rules=[],
    )
    with pytest.raises(RuntimeError, match="records are missing from the build"):
        accounting.check()


def test_build_accounting_refuses_counts_that_do_not_add_up() -> None:
    """Kept + dropped must equal the candidate count."""
    accounting = c25.BuildAccounting(
        dataset="d",
        source_rows=10,
        control_rows=0,
        candidate_records=5,
        kept_records=4,
        dropped_records=0,
        rules=[],
    )
    with pytest.raises(RuntimeError, match="!= 5 candidates"):
        accounting.check()


def test_the_seven_pride_accessions_are_one_per_dbtl_cycle_and_all_distinct() -> None:
    """The campaign's proteomics sits in seven accessions; all seven are enumerated."""
    assert list(c25.PRIDE_ACCESSIONS) == [f"DBTL{n}" for n in range(7)]
    assert len(set(c25.PRIDE_ACCESSIONS.values())) == 7
    assert c25.PRIDE_ACCESSIONS["DBTL0"] == "PXD063733"
    assert c25.PRIDE_ACCESSIONS["DBTL6"] == "PXD063746"


def test_the_pmc_cloud_key_names_this_articles_bucket_prefix() -> None:
    """The retrieval is a bucket key, which is what makes it re-runnable."""
    assert c25.pmc_cloud_key(c25.SOURCE_DATA_FILENAME) == (
        "PMC12748988.1/41467_2025_66304_MOESM9_ESM.xlsx"
    )


def test_manifest_sha256_raises_on_a_path_the_manifest_does_not_list() -> None:
    """A mirror file with no manifest row cannot be pinned, so it refuses."""
    manifest = Manifest(citation_key=c25.CITATION_KEY)
    with pytest.raises(KeyError, match="not in the raw-mirror manifest"):
        c25.manifest_sha256(manifest, "si/nope.xlsx")


def _write_titer_workbook(path: Path, alt_titer: float) -> None:
    """A two-sheet workbook in the released shape, with a settable second-sheet value."""
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = c25.SHEET_TITER
    sheet.append(
        ["Line Name", "cycle", "is_control", "isoprenoli titer (mg/L)", "pass filter?"]
    )
    sheet.append(["PP_0815-R1", 0, "False", 100.0, "True"])
    alt = book.create_sheet(c25.SHEET_TITER_ALT)
    alt.append(["Line Name", "cycle", "is_control", "isoprenol (mg/L)", "category"])
    alt.append(["PP_0815-R1", 0, "False", alt_titer, "other"])
    book.save(path)


def test_read_titer_rows_parses_the_replicate_suffix_and_the_filter_flag(
    tmp_path: Path,
) -> None:
    """One released culture becomes one typed row."""
    path = tmp_path / "titer.xlsx"
    _write_titer_workbook(path, 100.0)
    (row,) = c25.read_titer_rows(str(path))
    assert row.construct_name == "PP_0815"
    assert row.cycle == 0
    assert row.replicate == 1
    assert row.is_control is False
    assert row.titer_mg_per_l == pytest.approx(100.0)
    assert row.passed_filter is True


def test_read_titer_rows_refuses_two_sheets_that_disagree_on_a_titer(
    tmp_path: Path,
) -> None:
    """The two released exports are one measurement; a mismatch stops the build."""
    path = tmp_path / "titer.xlsx"
    _write_titer_workbook(path, 101.0)
    with pytest.raises(RuntimeError, match="disagree at row 0"):
        c25.read_titer_rows(str(path))


def test_read_titer_rows_refuses_a_line_name_with_no_replicate_suffix(
    tmp_path: Path,
) -> None:
    """Without ``-R<n>`` a row cannot be assigned to a replicate."""
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = c25.SHEET_TITER
    sheet.append(
        ["Line Name", "cycle", "is_control", "isoprenoli titer (mg/L)", "pass filter?"]
    )
    sheet.append(["PP_0815", 0, "False", 100.0, "True"])
    alt = book.create_sheet(c25.SHEET_TITER_ALT)
    alt.append(["Line Name", "cycle", "is_control", "isoprenol (mg/L)", "category"])
    alt.append(["PP_0815", 0, "False", 100.0, "other"])
    book.save(tmp_path / "t.xlsx")
    with pytest.raises(RuntimeError, match="no -R<n> replicate suffix"):
        c25.read_titer_rows(str(tmp_path / "t.xlsx"))


def test_a_sheet_the_workbook_does_not_carry_is_refused(tmp_path: Path) -> None:
    """A renamed sheet means the pinned workbook is not the one this loader reads."""
    book = openpyxl.Workbook()
    book.save(tmp_path / "empty.xlsx")
    with pytest.raises(RuntimeError, match=f"has no sheet {c25.SHEET_TITER!r}"):
        c25.read_titer_rows(str(tmp_path / "empty.xlsx"))


# --------------------------------------------------------------------------- #
# Data-gated: the real mirror, the real annotation, the built stores
# --------------------------------------------------------------------------- #
DATA_ROOT = os.environ.get("DATA_ROOT", "")
TITER_ROOT = osp.join(DATA_ROOT, "data/torchcell/isoprenol_titer_carruthers2025")
PROTEOME_ROOT = osp.join(DATA_ROOT, "data/torchcell/proteome_carruthers2025")
MIRROR_PRESENT = bool(DATA_ROOT) and osp.isfile(
    osp.join(DATA_ROOT, c25.RAW_DIR_REL, "manifest.json")
)
KT2440_PRESENT = bool(DATA_ROOT) and osp.isfile(
    osp.join(DATA_ROOT, "data/pputida/kt2440/genome/data.db")
)
BUILT = osp.isdir(osp.join(TITER_ROOT, "processed/lmdb")) and osp.isdir(
    osp.join(PROTEOME_ROOT, "processed/lmdb")
)

requires_mirror = pytest.mark.skipif(
    not MIRROR_PRESENT,
    reason="requires the Carruthers 2025 raw mirror under $DATA_ROOT",
)
requires_genome = pytest.mark.skipif(
    not KT2440_PRESENT, reason="requires the built KT2440 genome cache under $DATA_ROOT"
)
requires_built = pytest.mark.skipif(
    not BUILT, reason="requires both built Carruthers 2025 LMDBs under $DATA_ROOT"
)


@pytest.fixture(scope="module")
def kt2440() -> Any:
    """The deposited KT2440 annotation, reopened read-only."""
    from torchcell.datasets.bacteria_common import bacterial_genome

    return bacterial_genome("pputida", "KT2440")


@pytest.mark.data
@requires_mirror
def test_the_mirror_manifest_records_the_digests_the_module_pins() -> None:
    """The retrieval record and the loader's pins are the same bytes."""
    manifest = c25.load_manifest()
    assert manifest.citation_key == c25.CITATION_KEY
    assert manifest.doi == c25.DOI
    assert c25.manifest_sha256(manifest, c25.SOURCE_DATA_REL) == c25.SOURCE_DATA_SHA256
    assert c25.manifest_sha256(manifest, c25.TARGETS_REL) == c25.TARGETS_SHA256
    for record in manifest.files:
        assert record.retrieval is not None
        assert record.retrieval.method.value == "pmc_cloud"
        assert record.retrieval.sha256 == record.sha256


@pytest.mark.data
@requires_mirror
def test_the_mirror_enumerates_every_external_data_location() -> None:
    """Two deposited PMC objects, the Dryad deposit, and all seven PRIDE projects."""
    manifest = c25.load_manifest()
    sources = manifest.si_data_sources
    assert (
        sum(
            accession in " ".join(sources)
            for accession in c25.PRIDE_ACCESSIONS.values()
        )
        == 7
    )
    assert any(c25.DRYAD_DOI in source for source in sources)
    assert len(sources) == 10
    assert any("Anubis" in item for item in manifest.si_expected)


@pytest.mark.data
@requires_genome
def test_the_chassis_background_types_eight_alleles_against_the_real_annotation(
    kt2440: Any,
) -> None:
    """Each allele's locus comes from the genome's resolution of the source's symbol."""
    background = c25.chassis_background(kt2440)
    assert background.name == "IY1449b"
    assert background.reference_strain == "KT2440"
    by_symbol = {allele.gene_name: allele for allele in background.alleles}
    assert by_symbol.keys() == {
        "phaA",
        "phaB",
        "mvaB",
        "hbdH",
        "ldhA",
        "zwfB",
        "gntZ",
        "liuC",
    }
    assert by_symbol["phaA"].systematic_gene_name == "PP_5003"
    assert by_symbol["phaB"].systematic_gene_name == "PP_5004"
    assert by_symbol["mvaB"].systematic_gene_name == "PP_3540"
    assert by_symbol["hbdH"].systematic_gene_name == "PP_3073"
    assert by_symbol["ldhA"].systematic_gene_name == "PP_1649"
    assert by_symbol["zwfB"].systematic_gene_name == "PP_4042"
    assert by_symbol["gntZ"].systematic_gene_name == "PP_4043"
    assert by_symbol["liuC"].systematic_gene_name == "PP_4066"
    for allele in background.alleles:
        assert allele.edit is AlleleEdit.full_deletion
        assert allele.functional is False
        assert allele.provenance


@pytest.mark.data
@requires_genome
def test_the_stated_span_contains_the_three_loci_it_names_and_not_the_bare_zwf(
    kt2440: Any,
) -> None:
    """What identifies the Methods' ``zwf`` as ``zwfB``: the span, measured.

    ``4,538,575 + 86,812`` covers PP_4042, PP_4043 and PP_4066 entirely, while the bare
    symbol ``zwf`` resolves to PP_5351 well outside it.
    """
    assert c25.SPAN_END == 4_625_386
    for tag in ("PP_4042", "PP_4043", "PP_4066"):
        locus = kt2440[tag]
        assert locus is not None
        assert c25.SPAN_START <= locus.start and locus.end <= c25.SPAN_END
    outside = kt2440.resolve_gene_name("zwf")
    assert outside.systematic_name == "PP_5351"
    bare = kt2440["PP_5351"]
    assert bare is not None
    assert bare.start > c25.SPAN_END
    background = c25.chassis_background(kt2440)
    spans = {a.gene_name: a.deleted_span for a in background.alleles}
    assert {name for name, span in spans.items() if span is not None} == {
        "zwfB",
        "gntZ",
        "liuC",
    }
    assert spans["zwfB"] is not None
    assert (spans["zwfB"].start, spans["zwfB"].end) == (c25.SPAN_START, c25.SPAN_END)


@pytest.mark.data
@requires_genome
def test_phac_carries_no_locus_of_this_assembly(kt2440: Any) -> None:
    """The documented gap in ``dphaABC``, asserted rather than asserted-in-prose."""
    for symbol in c25.CHASSIS_UNMAPPED_SYMBOLS:
        assert kt2440.resolve_gene_name(symbol).status.value == "retired"


@pytest.mark.data
@requires_genome
def test_the_reference_pins_the_genbank_assembly_and_survives_serialization(
    kt2440: Any,
) -> None:
    """The assembly pin is only real where the field is narrowed; here it is."""
    reference = c25.chassis_reference(kt2440)
    assert reference.species == "Pseudomonas putida"
    assert reference.strain == "IY1449b"
    assert reference.assembly_set == "pputida_KT2440_ASM756v2"
    assert reference.assembly_accession == "GCA_000007565.2"
    dumped = ProductTiterExperimentReference(
        dataset_name="probe",
        genome_reference=reference,
        environment_reference=c25.production_environment(),
        phenotype_reference=c25.titer_phenotype([1.0, 2.0], is_reference=True),
    ).model_dump()
    assert dumped["genome_reference"]["assembly_accession"] == "GCA_000007565.2"
    assert dumped["genome_reference"]["background"]["name"] == "IY1449b"


@pytest.mark.data
@requires_mirror
@requires_genome
def test_every_guide_target_is_a_current_kt2440_locus_tag(kt2440: Any) -> None:
    """The reconciliation histogram the PR reports: 121 of 121 current, nothing kept."""
    from torchcell.datasets.bacteria_common import reconcile_locus_tags

    rows = c25.read_titer_rows(
        osp.join(DATA_ROOT, c25.RAW_DIR_REL, c25.SOURCE_DATA_REL)
    )
    tags = sorted(
        {
            tag
            for row in rows
            if not row.is_control
            for tag in c25.parse_construct(row.construct_name)[0]
        }
    )
    assert len(tags) == 121
    stored, report = reconcile_locus_tags(kt2440, pd.Series(tags), label="test")
    assert report.resolved_fraction == 1.0
    assert {status.value: n for status, n in report.status_histogram.items() if n} == {
        "current": 121
    }
    assert report.remapped == 0
    assert report.retired_kept == ()
    assert report.ambiguous_kept == {}
    assert report.outside_namespace == ()
    assert list(stored) == tags


@pytest.mark.data
@requires_mirror
def test_the_released_cultures_match_the_papers_own_control_and_strain_counts() -> None:
    """1,506 cultures; 18 DBTL0 controls and 12 per later cycle; 465 strains."""
    rows = c25.read_titer_rows(
        osp.join(DATA_ROOT, c25.RAW_DIR_REL, c25.SOURCE_DATA_REL)
    )
    assert len(rows) == 1506
    controls = [row for row in rows if row.is_control]
    assert len(controls) == 90
    per_cycle = {
        cycle: sum(1 for row in controls if row.cycle == cycle) for cycle in range(7)
    }
    assert per_cycle == {0: 18, 1: 12, 2: 12, 3: 12, 4: 12, 5: 12, 6: 12}
    strains: dict[tuple[str, int], int] = {}
    for row in rows:
        if row.is_control:
            continue
        key = (row.construct_name, row.cycle)
        strains[key] = strains.get(key, 0) + 1
    assert len(strains) == 465
    assert sorted(strains.values()) == [3] * 458 + [6] * 7


@pytest.mark.data
@requires_built
def test_the_built_titer_store_has_one_record_per_strain_and_passes_l0_to_l4() -> None:
    """L0 schema, L1 count + the 472 reconciliation, L2 fidelity and the SE identity,
    L3 unit and pathway pins, L4 against Supplementary Data 1.
    """
    from torchcell.verification.runners import load_records

    report = c25.titer_report(load_records(TITER_ROOT), DATA_ROOT)
    assert report.passed, [
        (r.level, r.name, r.message) for r in report.results if not r.passed
    ]
    assert report.levels_covered == {Level.L0, Level.L1, Level.L2, Level.L3, Level.L4}


@pytest.mark.data
@requires_built
def test_the_built_proteome_store_has_one_record_per_sample_and_passes_l0_to_l4() -> (
    None
):
    """The protein family's own five levels over the PP_0815 off-target panel."""
    from torchcell.verification.runners import load_records

    report = c25.proteome_report(load_records(PROTEOME_ROOT), DATA_ROOT)
    assert report.passed, [
        (r.level, r.name, r.message) for r in report.results if not r.passed
    ]
    assert report.levels_covered == {Level.L0, Level.L1, Level.L2, Level.L3, Level.L4}


@pytest.mark.data
@requires_built
def test_the_titer_build_accounting_records_zero_drops() -> None:
    """Every released non-control culture is in a record."""
    accounting = c25.BuildAccounting.model_validate_json(
        Path(osp.join(TITER_ROOT, "preprocess/build_accounting.json")).read_text()
    )
    accounting.check()
    assert accounting.source_rows == 1506
    assert accounting.control_rows == 90
    assert accounting.candidate_records == 465
    assert accounting.kept_records == 465
    assert accounting.dropped_records == 0


@pytest.mark.data
@requires_built
def test_the_proteome_build_accounting_drops_keys_but_never_a_sample() -> None:
    """77 of 1,501 protein keys are dropped; all 19 samples are kept."""
    accounting = c25.BuildAccounting.model_validate_json(
        Path(osp.join(PROTEOME_ROOT, "preprocess/build_accounting.json")).read_text()
    )
    accounting.check()
    assert accounting.kept_records == 19
    assert accounting.dropped_records == 0
    dropped = pd.read_csv(
        osp.join(PROTEOME_ROOT, "preprocess/dropped_protein_keys.csv")
    )
    assert len(dropped) == 77
    assert set(dropped["reason"]) == {
        "merged_accessions",
        "not_a_locus_of_the_pinned_assembly",
    }
    merged = set(dropped.loc[dropped["reason"] == "merged_accessions", "protein_key"])
    assert merged == {"Apha", "Asd"}


@pytest.mark.data
@requires_built
def test_the_build_manifests_name_this_loader_module() -> None:
    """The freshness gate reads exactly these files."""
    for root, cls in (
        (TITER_ROOT, "IsoprenolTiterCarruthers2025Dataset"),
        (PROTEOME_ROOT, "ProteomeCarruthers2025Dataset"),
    ):
        manifest = json.loads(
            Path(osp.join(root, "preprocess/build_manifest.json")).read_text()
        )
        assert manifest["loader_class"] == cls
        assert manifest["loader_module"] == "torchcell.datasets.pputida.carruthers2025"
        assert manifest["closure"]


# --------------------------------------------------------------------------- #
# The L0-L4 batteries moved INTO the loader module (``c25.titer_report`` and
# ``c25.proteome_report``), so ``run_product_titer`` and
# ``run_bacterial_protein_abundance`` run them from ``run_all``. What stays here is
# the data-gated assertion that they pass on the real stores; regenerate the note's
# table with ``python -m torchcell.verification.runners``.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Hermetic end to end: a synthetic KT2440 assembly, a synthetic raw mirror, and
# both loaders built under tmp_path. No network, no $DATA_ROOT.
#
# The real tier cannot be used on a CI runner, so the real KT2440 class is built
# over a 600 nt synthetic replicon carrying the eighteen loci these two loaders
# need: the eight chassis symbols the background types, the guide targets, and a
# locus whose symbol is ``apha`` so the merged-key drop has something to resolve.
# ``phaC`` is deliberately absent, which is what the chassis builder asserts.
# --------------------------------------------------------------------------- #
LOCUS_SPECS: tuple[tuple[str, str | None], ...] = (
    ("PP_3073", "hbdH"),
    ("PP_1649", "ldhA"),
    ("PP_3540", "mvaB"),
    ("PP_4042", "zwfB"),
    ("PP_4043", "gntZ"),
    ("PP_4066", "liuC"),
    ("PP_5003", "phaA"),
    ("PP_5004", "phaB"),
    ("PP_0368", None),
    ("PP_0378", None),
    ("PP_0528", None),
    ("PP_0812", None),
    ("PP_0813", None),
    ("PP_0814", None),
    ("PP_0815", None),
    ("PP_0977", None),
    ("PP_1593", None),
    ("PP_2664", None),
    ("PP_3379", None),
    ("PP_5313", None),
    ("PP_5424", "apha"),
    # The overexpression panel's two operons. PP_2208 is annotated phnX and PP_2209
    # phnW, as the pinned assembly annotates them, so the fixture reproduces the one
    # crosswalk the Source Data's own Protein.Description column contradicts.
    ("PP_2208", "phnX"),
    ("PP_2209", "phnW"),
    ("PP_2791", None),
    ("PP_2792", None),
    ("PP_2793", None),
    ("PP_2794", None),
)
#: The loci the overexpression panel quantifies, and the KEYS the released sheet files
#: them under. They are excluded from :data:`PROTEOME_TAGS` so no literal ``PP_`` key
#: collides with the symbol or locus-tag layer that reaches the same locus.
OVEREXPRESSION_LOCI: tuple[str, ...] = (
    "PP_2208",
    "PP_2209",
    "PP_2791",
    "PP_2792",
    "PP_2793",
    "PP_2794",
)
OVEREXPRESSION_KEYS: tuple[str, ...] = (
    "Phnw",
    "Phnx",
    "Pp_2791",
    "Pp_2792",
    "Pp_2793",
    "Pp_2794",
)
#: What the released ``Supplementary Figure 12ac`` puts in ``Protein.Description`` for
#: the two Phn keys: the locus tag, and the WRONG one of the pair.
OVEREXPRESSION_KEY_DESCRIPTIONS: dict[str, str] = {"Phnw": "PP_2208", "Phnx": "PP_2209"}
#: The keys the synthetic proteome sheet uses as locus tags; ``PP_5424`` is reached
#: only through the symbol ``Apha`` and the overexpression loci only through their own
#: keys, so none of them ever collides.
PROTEOME_TAGS: tuple[str, ...] = tuple(
    tag for tag, _ in LOCUS_SPECS if tag != "PP_5424" and tag not in OVEREXPRESSION_LOCI
)
#: One key no layer resolves and one the sheet files under two accessions.
PROTEOME_UNRESOLVED = "Krt1"
PROTEOME_MERGED = "Apha"
#: Single-guide DBTL0 constructs, which the two per-target oracles join against.
SINGLE_GUIDE_TARGETS: tuple[str, ...] = (
    "PP_0368",
    "PP_0378",
    "PP_0812",
    "PP_0815",
    "PP_0977",
)
#: The DBTL0 construct with six replicates rather than three.
SIX_REPLICATE_CONSTRUCT = "PP_4042"
#: The DBTL0 construct carrying a non-targeting filler guide.
FILLER_CONSTRUCT = "PP_3073_NT1"
#: Multi-guide constructs, one per later cycle.
COMBINATION_CONSTRUCTS: dict[int, tuple[str, ...]] = {
    1: ("PP_0815_PP_0812", "PP_0368_PP_0815"),
    2: ("PP_0378_PP_0815", "PP_0528_PP_0815"),
    3: ("PP_0812_PP_0977",),
    4: ("PP_0368_PP_0378_PP_0815",),
    5: ("PP_1593_PP_2664",),
    6: ("PP_3379_PP_5313_PP_0815",),
}
#: The proteome samples the synthetic sheet carries: the reference, the positive
#: control, and two off-target samples (one with a plate suffix).
PROTEOME_SAMPLES: tuple[str, ...] = (
    c25.PROTEOME_REFERENCE_SAMPLE,
    c25.PROTEOME_TARGET_SAMPLE,
    "JBEI_OTS_PP_0378_48hr",
    "JBEI_OTS_PP_0977_1_P4_48hr",
)
#: The KO backgrounds of the synthetic ``Figure 6a``: one single-gene deletion with a
#: CRISPRi analogue in ``Figure 4b``, and the one multi-gene designation whose Target
#: sgRNA only the Results text names.
KO_PANEL_BACKGROUNDS: tuple[str, ...] = ("PP_0815", "PP_0812-15")
#: The synthetic ``Supplementary Figure 13d`` off-target samples; ``PP_0977`` appears
#: twice as a repeated culture, so the panel screens two distinct genes.
OFFTARGET_SAMPLES: tuple[str, ...] = ("PP_0378", "PP_0977_1", "PP_0977_2")
OFFTARGET_DISTINCT_TARGETS = 2
#: The ``Figure 6d`` line the Results sentence prints a titer for, and its cultures.
KO_ARRAY_PROSE_LINE = "PP_0368_PP_0812-15_KO with PP_0528_PP_0815"
KO_ARRAY_PROSE_TITERS: tuple[float, ...] = (980.0, 981.0, 982.0)
#: What the four synthetic panels build: 2 Figure 6a + 2 Figure 6d + 4 off-target +
#: 2 uninduced overexpression strains.
SYNTHETIC_PANEL_TITER_RECORDS = 2 + 3 + 1 + len(OFFTARGET_SAMPLES) + 2

ASSEMBLY_REPORT = """# Assembly name:  ASM756v2
# Organism name:  Pseudomonas putida KT2440 (g-proteobacteria)
# Infraspecific name:  strain=KT2440
# Taxid:          160488
# GenBank assembly accession: GCA_000007565.2
# RefSeq assembly accession: GCF_000007565.2
# RefSeq assembly and GenBank assemblies identical: yes
#
## Assembly-Units:
AE015451.2\tassembled-molecule\tna\tChromosome\tAE015451.2\t=\tNC_002947.4
"""
ASSEMBLY_REPORT_MEMBER = "GCA_000007565.2_ASM756v2_assembly_report.txt"


def _synthetic_loci() -> list[Any]:
    """One :class:`SyntheticLocus` per :data:`LOCUS_SPECS` entry, laid end to end."""
    from tests.torchcell.sequence.genome._bacterial_fixtures import SyntheticLocus

    loci = []
    cursor = 1
    for index, (tag, symbol) in enumerate(LOCUS_SPECS):
        start, end = cursor, cursor + 11
        cursor = end + 3
        loci.append(
            SyntheticLocus(
                tag=tag,
                parts=((start, end),),
                strand="+" if index % 2 == 0 else "-",
                symbol=symbol,
                product=f"synthetic product {tag}",
                protein_id=f"AAN{index:05d}.1",
                protein="MKV",
            )
        )
    return loci


@pytest.fixture
def synthetic_kt2440(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The real KT2440 class over a synthetic assembly, with the network refused."""
    import tests.torchcell.sequence.genome._bacterial_fixtures as fixtures
    from torchcell.sequence.genome.pputida.kt2440 import (
        KT2440_ASSEMBLY,
        PPutidaKT2440Genome,
    )

    monkeypatch.setattr(fixtures, "SEQUENCE", fixtures.SEQUENCE * 6)
    files = fixtures.write_assembly(
        tmp_path / "tier",
        KT2440_ASSEMBLY,
        _synthetic_loci(),
        [fixtures.gaf_row("hbdH", f"hbdH|{PROTEOME_TAGS[0]}", "GO:0000001")],
    )
    fixtures.forbid_network(monkeypatch)
    fixtures.serve_tier(monkeypatch, files)
    report = tmp_path / "tier" / ASSEMBLY_REPORT_MEMBER
    report.write_text(ASSEMBLY_REPORT)

    def serve_report(assembly_set: str, filename: str, **_: Any) -> str:
        if filename != ASSEMBLY_REPORT_MEMBER:
            raise FileNotFoundError(f"{assembly_set}/{filename} is not in the fixture")
        return str(report)

    import torchcell.datasets.bacteria_common as bacteria_common

    monkeypatch.setattr(bacteria_common, "resolve", serve_report)
    root = tmp_path / "kt2440"
    root.mkdir()
    return PPutidaKT2440Genome(genome_root=str(root), overwrite=False)


def _titer_rows() -> list[tuple[str, int, str, float, str]]:
    """The synthetic ``Figure 4b`` rows: 90 controls plus the strains above."""
    rows: list[tuple[str, int, str, float, str]] = []
    base = 150.0
    for cycle in range(7):
        n = 18 if cycle == 0 else 12
        for replicate in range(1, n + 1):
            rows.append(
                (
                    f"Control-R{replicate}",
                    cycle,
                    "True",
                    base + cycle + replicate * 0.25,
                    "True",
                )
            )
    offset = 200.0
    for index, tag in enumerate(SINGLE_GUIDE_TARGETS):
        for replicate in (1, 2, 3):
            rows.append(
                (
                    f"{tag}-R{replicate}",
                    0,
                    "False",
                    offset + index * 10 + replicate * 1.5,
                    "True" if index % 2 == 0 else "False",
                )
            )
    for replicate in range(1, 7):
        rows.append(
            (
                f"{SIX_REPLICATE_CONSTRUCT}-R{replicate}",
                0,
                "False",
                300.0 + replicate,
                "True",
            )
        )
    for replicate in (1, 2, 3):
        rows.append(
            (f"{FILLER_CONSTRUCT}-R{replicate}", 0, "False", 255.0 + replicate, "False")
        )
    value = 400.0
    for cycle, constructs in COMBINATION_CONSTRUCTS.items():
        for construct in constructs:
            for replicate in (1, 2, 3):
                value += 1.25
                rows.append(
                    (f"{construct}-R{replicate}", cycle, "False", value, "True")
                )
    return rows


def _target_means(rows: list[tuple[str, int, str, float, str]]) -> dict[str, float]:
    """Per DBTL0 construct, the mean over its replicates (the oracles' own number)."""
    groups: dict[str, list[float]] = {}
    for line, cycle, is_control, titer, _ in rows:
        if cycle != 0 or is_control == "True":
            continue
        base = c25.REPLICATE_RE.match(line)
        assert base is not None
        groups.setdefault(base.group("base"), []).append(titer)
    return {name: sum(v) / len(v) for name, v in groups.items()}


def _write_source_data(path: Path) -> None:
    """A Source Data workbook in the released shape, small enough to read by eye."""
    rows = _titer_rows()
    means = _target_means(rows)
    book = openpyxl.Workbook()

    titer = book.active
    titer.title = c25.SHEET_TITER
    titer.append(
        ["Line Name", "cycle", "is_control", "isoprenoli titer (mg/L)", "pass filter?"]
    )
    alt = book.create_sheet(c25.SHEET_TITER_ALT)
    alt.append(["Line Name", "cycle", "is_control", "isoprenol (mg/L)", "category"])
    for line, cycle, is_control, value, passed in rows:
        titer.append([line, cycle, is_control, value, passed])
        alt.append(
            [
                line,
                cycle,
                is_control,
                value,
                "control" if is_control == "True" else "other",
            ]
        )

    figure3a = book.create_sheet(c25.SHEET_TARGET_MEANS)
    figure3a.append(
        ["Strain", "Target", "cog_base_function", "Isoprenol mean", "Target:Control"]
    )
    for name, mean in sorted(means.items()):
        figure3a.append([f"IY{name}", name, "Energy production", mean, 0.5])

    controls = book.create_sheet(c25.SHEET_CONTROLS)
    controls.append(["Line Name", "Cycle", "Titer"])
    for line, cycle, is_control, value, _ in rows:
        if is_control == "True":
            controls.append([line, f"DBTL-{cycle}", value])

    proteome = book.create_sheet(c25.SHEET_PROTEOME)
    proteome.append(
        [
            "Protein.Group",
            "Protein.Names",
            "Protein",
            "Protein.Description",
            "Sample",
            "Replicate",
            "Top_3pep_counts_mean",
            "%_of protein_abundance_Top3-method",
            "log10_%_abundance",
        ]
    )
    keys: list[tuple[str, tuple[str, ...]]] = [
        (tag, (f"Q{tag}",)) for tag in PROTEOME_TAGS
    ]
    # The overexpression panel's keys are in BOTH matrices, as they are in the release:
    # the loader refuses a key one sheet carries and the other does not, so the two
    # panels can never key one protein two ways.
    keys.extend((key, (f"Q{key}",)) for key in OVEREXPRESSION_KEYS)
    keys.append((PROTEOME_UNRESOLVED, ("P04264",)))
    keys.append((PROTEOME_MERGED, ("P0AE22", "Q88C43")))
    for sample_index, sample in enumerate(PROTEOME_SAMPLES):
        for key_index, (key, accessions) in enumerate(keys):
            for accession in accessions:
                for replicate_index, replicate in enumerate(("R1", "R2", "R3")):
                    signal = (
                        0.0
                        if key_index == 0
                        else 1000.0 * (sample_index + 1)
                        + key_index * 10
                        + replicate_index
                    )
                    proteome.append(
                        [
                            accession,
                            f"{accession}_PSEPK",
                            key,
                            f"synthetic {key}",
                            sample,
                            replicate,
                            signal,
                            1e-05,
                            -5,
                        ]
                    )

    _write_source_data_panels(book, rows)
    book.save(path)


def _figure_4b_values(
    rows: list[tuple[str, int, str, float, str]], base: str, cycle: int
) -> list[float]:
    """The synthetic ``Figure 4b`` titers of one construct in one cycle."""
    out = [
        titer
        for line, row_cycle, _, titer, _ in rows
        if row_cycle == cycle
        and (match := c25.REPLICATE_RE.match(line)) is not None
        and match.group("base") == base
    ]
    assert out, f"{base} has no cycle {cycle} rows in the synthetic Figure 4b"
    return out


def _write_source_data_panels(
    book: Any, rows: list[tuple[str, int, str, float, str]]
) -> None:
    """The four unstored titer panels and the overexpression proteome, in released shape.

    Every arm the loader reads as a re-export carries a verbatim ``Figure 4b`` titer and
    every arm it reads as new carries a value no other sheet holds, so the partition
    proof is exercised in both directions. The ``PP_0815`` arms are wired so the two
    documented duplications are reproduced: the non-targeting triplicate is identical
    across the two sheets and the targeting triplicates share exactly one value.
    """
    ko_panel = book.create_sheet(c25.SHEET_KO_PANEL)
    ko_alt = book.create_sheet(c25.SHEET_KO_PANEL_ALT)
    panel_rows: list[list[Any]] = [["Strain", "Type", "Titer"]]
    crispri = _figure_4b_values(rows, "PP_0815", 0)
    target = {"PP_0815": [900.0, 901.0, 902.0], "PP_0812-15": [910.0, 911.0, 912.0]}
    non_target = {"PP_0815": [920.0, 921.0, 922.0], "PP_0812-15": [930.0, 931.0, 932.0]}
    for value in crispri:
        panel_rows.append(["PP_0815", c25.ARM_CRISPRI, value])
    for background in KO_PANEL_BACKGROUNDS:
        for value in target[background]:
            panel_rows.append([background, c25.ARM_KO_TARGET, value])
        for value in non_target[background]:
            panel_rows.append([background, c25.ARM_KO_NONTARGET, value])
    for row in panel_rows:
        ko_panel.append(row)
        ko_alt.append(row)

    arrays = book.create_sheet(c25.SHEET_KO_ARRAYS)
    arrays.append(["Line Name", "Type", "Replicate", "Isoprenol"])
    control_cycle = _figure_4b_values(rows, "Control", c25.KO_ARRAY_CONTROL_CYCLE)[:3]
    for index, value in enumerate(control_cycle, start=1):
        arrays.append([c25.KO_ARRAY_CONTROL_LINE, c25.ARM_CRISPRI, f"R{index}", value])
    reexported_array = COMBINATION_CONSTRUCTS[1][1]
    for index, value in enumerate(_figure_4b_values(rows, reexported_array, 1), 1):
        arrays.append([reexported_array, c25.ARM_CRISPRI, f"R{index}", value])
    for index, value in enumerate((940.0, 941.0, 942.0), start=1):
        arrays.append(["PP_0812-15_KO", c25.ARM_KO_ONLY, f"R{index}", value])
    for index, value in enumerate((950.0, 951.0, 952.0), start=1):
        arrays.append(
            ["PP_0812-15_KO with PP_0815", c25.ARM_KO_PLUS_CRISPRI, f"R{index}", value]
        )
    # The exact genotype the Results sentence prints a titer for, so the prose L4 runs
    # against a real record rather than being skipped on the fixture.
    for index, value in enumerate(KO_ARRAY_PROSE_TITERS, start=1):
        arrays.append(
            [KO_ARRAY_PROSE_LINE, c25.ARM_KO_PLUS_CRISPRI, f"R{index}", value]
        )

    offtarget = book.create_sheet(c25.SHEET_OFFTARGET_TITER)
    offtarget.append(["Sample", "Replicate", "titer"])
    # The reference arm is bit-identical to Figure 6a's, and the targeting arm shares
    # exactly its first value: both of the release's own internal duplications.
    offtarget_target = [target["PP_0815"][0], 960.0, 961.0]
    for index in range(3):
        offtarget.append(
            [
                c25.OFFTARGET_REFERENCE_SAMPLE,
                f"R{index + 1}",
                non_target["PP_0815"][index],
            ]
        )
        offtarget.append(
            [c25.OFFTARGET_TARGET_SAMPLE, f"R{index + 1}", offtarget_target[index]]
        )
        for sample_index, sample in enumerate(OFFTARGET_SAMPLES):
            offtarget.append(
                [sample, f"R{index + 1}", 970.0 + sample_index * 10 + index]
            )

    titer_panel = book.create_sheet(c25.SHEET_OVEREXPRESSION_TITER)
    titer_panel.append(["Strain", "Replicate", "Inducer concentration", "Isoprenol"])
    levels = (c25.OVEREXPRESSION_UNINDUCED_LEVEL, *c25.OVEREXPRESSION_INDUCED_LEVELS)
    for label_index, label in enumerate(c25.OVEREXPRESSION_SHEET_LABELS.values()):
        for level_index, level in enumerate(levels):
            for replicate in (1, 2, 3):
                titer_panel.append(
                    [
                        label,
                        f"R{replicate}",
                        level,
                        1000.0 + label_index * 100 + level_index * 10 + replicate,
                    ]
                )
        # The sheet releases the control as two uninduced blocks under one name.
        for replicate in (1, 2, 3):
            titer_panel.append(
                [
                    c25.OVEREXPRESSION_CONTROL_STRAIN,
                    f"R{replicate}",
                    c25.OVEREXPRESSION_UNINDUCED_LEVEL,
                    1200.0 + label_index * 10 + replicate,
                ]
            )

    proteome_panel = book.create_sheet(c25.SHEET_OVEREXPRESSION_PROTEOME)
    proteome_panel.append(
        [
            "Protein.Group",
            "Protein.Names",
            "Protein",
            "Protein.Description",
            "Sample",
            "Strain",
            "Inducer concentration",
            "Replicate",
            "Top_3pep_counts_mean",
            "%_of protein_abundance_Top3-method",
            "log10_%_abundance",
        ]
    )
    operons = {
        "pSTABL1": ("Phnw", "Phnx"),
        "pSTABL2": ("Pp_2791", "Pp_2792", "Pp_2793", "Pp_2794"),
    }
    for label, keys in operons.items():
        samples = [
            (
                f"{label}_Condition_{index + 1}",
                level,
                c25.OVEREXPRESSION_SHEET_LABELS[label],
            )
            for index, level in enumerate(reversed(c25.OVEREXPRESSION_INDUCED_LEVELS))
        ]
        samples.append(
            (
                f"{label}_Condition_{len(c25.OVEREXPRESSION_INDUCED_LEVELS) + 1}",
                c25.OVEREXPRESSION_UNINDUCED_LEVEL,
                c25.OVEREXPRESSION_SHEET_LABELS[label],
            )
        )
        samples.append(
            (
                f"{label}_Control",
                c25.OVEREXPRESSION_UNINDUCED_LEVEL,
                c25.OVEREXPRESSION_CONTROL_STRAIN,
            )
        )
        for sample_index, (sample, level, strain) in enumerate(samples):
            for key_index, key in enumerate(keys):
                for replicate in (1, 2, 3):
                    proteome_panel.append(
                        [
                            f"Q{key}",
                            f"Q{key}_PSEPK",
                            key,
                            OVEREXPRESSION_KEY_DESCRIPTIONS.get(
                                key, f"synthetic {key}"
                            ),
                            sample,
                            strain,
                            level,
                            f"R{replicate}",
                            2000.0 + sample_index * 100 + key_index * 10 + replicate,
                            1e-05,
                            -5,
                        ]
                    )


def _write_targets(path: Path, means: dict[str, float]) -> None:
    """A Supplementary Data 1 workbook: two preamble rows, then the header."""
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["Supplementary Table 1: List of gene targets from this study"])
    sheet.append([None])
    sheet.append(
        [
            "Locus number",
            "Protein name",
            "Source (Flux-RETAP or Heuristic) 1",
            "RbTn-Seq Data?",
            "Mean isoprenol titer (mg/L)",
            "dCas9/control",
            "POI/control",
            "Passed DBTL0 filter?",
            "Used after DBTL2/3?",
        ]
    )
    for name, mean in sorted(means.items()):
        sheet.append(
            [
                name,
                f"synthetic {name}",
                "FR",
                "N",
                round(mean, 2),
                0.2,
                0.3,
                "True",
                "False",
            ]
        )
    book.save(path)


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def synthetic_mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A raw mirror under ``tmp_path`` built by the module's own deposit function.

    The two workbooks are written here, their real digests are monkeypatched over the
    module's pins, and `deposit_raw_mirror` writes the mirror and its manifest. That
    exercises the deposit, the manifest round trip and both loaders' `download`
    without a network call or a read of the real ``$DATA_ROOT``.
    """
    staging = tmp_path / "staging"
    staging.mkdir()
    source_data = staging / c25.SOURCE_DATA_FILENAME
    targets = staging / c25.TARGETS_FILENAME
    _write_source_data(source_data)
    _write_targets(targets, _target_means(_titer_rows()))
    monkeypatch.setattr(c25, "SOURCE_DATA_SHA256", _sha256_bytes(source_data))
    monkeypatch.setattr(c25, "TARGETS_SHA256", _sha256_bytes(targets))
    # The two counts the module pins to the REAL workbook. The fixture is a miniature
    # of the released shape, not of its size, so the pins are re-pointed at what it
    # holds; the released numbers are asserted by the dev-tree build and its report.
    monkeypatch.setattr(c25, "OFFTARGET_CANDIDATE_TARGETS", OFFTARGET_DISTINCT_TARGETS)
    monkeypatch.setattr(
        c25, "EXPECTED_PANEL_TITER_RECORDS", SYNTHETIC_PANEL_TITER_RECORDS
    )
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    root = c25.deposit_raw_mirror(
        source_data_path=source_data, targets_path=targets, data_root=str(data_root)
    )
    assert root == c25.raw_mirror_dir(str(data_root))
    return data_root


# --- the deposit itself ---------------------------------------------------- #
def test_the_deposit_writes_both_files_and_a_manifest_that_pins_them(
    synthetic_mirror: Path,
) -> None:
    """Two `si_data` records, each with a re-runnable `pmc_cloud` retrieval."""
    manifest = c25.load_manifest(str(synthetic_mirror))
    assert manifest.citation_key == c25.CITATION_KEY
    assert manifest.doi == c25.DOI
    assert manifest.title == c25.TITLE
    assert [record.path for record in manifest.files] == [
        c25.SOURCE_DATA_REL,
        c25.TARGETS_REL,
    ]
    for record in manifest.files:
        assert record.role == ROLE_SI_DATA
        assert record.retrieval is not None
        assert record.retrieval.method is RetrievalMethod.pmc_cloud
        assert record.retrieval.retriever == (
            "torchcell.literature.retrieve.pmc_cloud_object"
        )
        assert record.retrieval.params["key"].startswith("PMC12748988.1/")
        assert record.retrieval.sha256 == record.sha256
    assert c25.manifest_sha256(manifest, c25.SOURCE_DATA_REL) == c25.SOURCE_DATA_SHA256
    assert len(manifest.si_data_sources) == 10
    assert manifest.provenance_complete is True


def test_the_deposit_is_idempotent_by_sha256(
    synthetic_mirror: Path, tmp_path: Path
) -> None:
    """Re-depositing the same bytes leaves the mirror alone."""
    deposited = c25.raw_mirror_dir(str(synthetic_mirror)) / c25.SOURCE_DATA_REL
    before = deposited.stat().st_mtime_ns
    c25.deposit_raw_mirror(
        source_data_path=tmp_path / "staging" / c25.SOURCE_DATA_FILENAME,
        targets_path=tmp_path / "staging" / c25.TARGETS_FILENAME,
        data_root=str(synthetic_mirror),
    )
    assert deposited.stat().st_mtime_ns == before


def test_the_deposit_refuses_a_mirror_file_whose_bytes_differ(
    synthetic_mirror: Path, tmp_path: Path
) -> None:
    """A differing mirror file raises rather than being overwritten."""
    deposited = c25.raw_mirror_dir(str(synthetic_mirror)) / c25.SOURCE_DATA_REL
    deposited.write_bytes(b"not the workbook")
    with pytest.raises(RuntimeError, match="exists with a different sha256; refusing"):
        c25.deposit_raw_mirror(
            source_data_path=tmp_path / "staging" / c25.SOURCE_DATA_FILENAME,
            targets_path=tmp_path / "staging" / c25.TARGETS_FILENAME,
            data_root=str(synthetic_mirror),
        )


def test_the_deposit_refuses_a_source_file_off_its_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both files are verified BEFORE anything is written, so nothing is deposited."""
    staging = tmp_path / "staging"
    staging.mkdir()
    source_data = staging / c25.SOURCE_DATA_FILENAME
    targets = staging / c25.TARGETS_FILENAME
    _write_source_data(source_data)
    _write_targets(targets, _target_means(_titer_rows()))
    monkeypatch.setattr(c25, "SOURCE_DATA_SHA256", "0" * 64)
    monkeypatch.setattr(c25, "TARGETS_SHA256", _sha256_bytes(targets))
    data_root = tmp_path / "data_root"
    with pytest.raises(Exception, match="sha256"):
        c25.deposit_raw_mirror(
            source_data_path=source_data, targets_path=targets, data_root=str(data_root)
        )
    assert not (data_root / c25.RAW_DIR_REL / c25.SOURCE_DATA_REL).exists()


def test_raw_mirror_dir_reads_data_root_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mirror path is derived, never configured per call."""
    monkeypatch.setenv("DATA_ROOT", "/from_env")
    assert c25.raw_mirror_dir() == Path("/from_env") / c25.RAW_DIR_REL


# --- the readers' refusals ------------------------------------------------- #
def test_each_oracle_reader_refuses_a_changed_header(tmp_path: Path) -> None:
    """A renamed column means the pinned workbook is not the one these readers read."""
    path = tmp_path / "source.xlsx"
    _write_source_data(path)
    book = openpyxl.load_workbook(path)
    book[c25.SHEET_TARGET_MEANS]["B1"] = "Gene"
    book[c25.SHEET_CONTROLS]["C1"] = "Isoprenol"
    book[c25.SHEET_PROTEOME]["G1"] = "Signal"
    book.save(path)
    with pytest.raises(RuntimeError, match=f"{c25.SHEET_TARGET_MEANS} header changed"):
        c25.read_target_means(str(path))
    with pytest.raises(RuntimeError, match=f"{c25.SHEET_CONTROLS} header changed"):
        c25.read_control_titers(str(path))
    with pytest.raises(RuntimeError, match=f"{c25.SHEET_PROTEOME} header changed"):
        c25.read_proteome_rows(str(path))


def test_read_control_titers_refuses_a_cycle_label_that_is_not_dbtl(
    tmp_path: Path,
) -> None:
    """The cycle column is `DBTL-<n>`; anything else cannot be assigned to a cycle."""
    path = tmp_path / "source.xlsx"
    _write_source_data(path)
    book = openpyxl.load_workbook(path)
    book[c25.SHEET_CONTROLS]["B2"] = "round zero"
    book.save(path)
    with pytest.raises(RuntimeError, match="is not DBTL-<n>"):
        c25.read_control_titers(str(path))


def test_read_si_target_means_refuses_a_changed_header(tmp_path: Path) -> None:
    """Supplementary Data 1's two preamble rows and its header are both load bearing."""
    path = tmp_path / "targets.xlsx"
    _write_targets(path, {"PP_0815": 1.0})
    book = openpyxl.load_workbook(path)
    book.worksheets[0]["A3"] = "Gene"
    book.save(path)
    with pytest.raises(RuntimeError, match="Supplementary Data 1 header changed"):
        c25.read_si_target_means(str(path))


def test_read_proteome_rows_types_every_released_cell(tmp_path: Path) -> None:
    """Accession, entry name, symbol, description, sample, replicate and signal."""
    path = tmp_path / "source.xlsx"
    _write_source_data(path)
    rows = c25.read_proteome_rows(str(path))
    expected = (
        len(PROTEOME_SAMPLES)
        * 3
        * (len(PROTEOME_TAGS) + len(OVEREXPRESSION_KEYS) + 1 + 2)
    )
    assert len(rows) == expected
    first = rows[0]
    assert first.sample == c25.PROTEOME_REFERENCE_SAMPLE
    assert first.replicate == "R1"
    assert first.entry_name.endswith("_PSEPK")
    assert first.top3_signal == 0.0


def test_the_aggregation_refuses_a_protein_with_no_replicate_values() -> None:
    """An empty cell list has no mean, so it is refused rather than imputed."""
    with pytest.raises(RuntimeError, match="no replicate values"):
        c25.ProteomeCarruthers2025Dataset._aggregate({"PP_0001": []}, "sample")


# --- the chassis builder over the synthetic annotation --------------------- #
def test_the_chassis_background_resolves_every_symbol_it_types(
    synthetic_kt2440: Any,
) -> None:
    """Eight alleles, three carrying the stated span, all sourced."""
    background = c25.chassis_background(synthetic_kt2440)
    assert background.name == c25.CHASSIS_STRAIN
    assert background.parents == ["KT2440"]
    assert background.genotype_statement == c25._Q_SI_TABLE2
    assert len(background.provenance or []) == 4
    assert {a.gene_name: a.systematic_gene_name for a in background.alleles} == {
        "phaA": "PP_5003",
        "phaB": "PP_5004",
        "mvaB": "PP_3540",
        "hbdH": "PP_3073",
        "ldhA": "PP_1649",
        "zwfB": "PP_4042",
        "gntZ": "PP_4043",
        "liuC": "PP_4066",
    }
    spanned = {a.gene_name for a in background.alleles if a.deleted_span is not None}
    assert spanned == {"zwfB", "gntZ", "liuC"}
    for allele in background.alleles:
        assert allele.edit is AlleleEdit.full_deletion
        assert allele.functional is False
        assert allele.gene_namespace == "pputida_kt2440_locus_tag"
    assert background.functional_copies("PP_4042") == 0
    assert background.functional_copies("PP_0815") == 1


def test_the_chassis_builder_stops_when_a_symbol_it_types_stops_resolving(
    synthetic_kt2440: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A symbol the annotation drops cannot be given a locus, so the build stops."""
    monkeypatch.setattr(
        c25,
        "CHASSIS_ALLELES",
        (
            *c25.CHASSIS_ALLELES,
            {
                "symbol": "nope",
                "allele": "Δnope",
                "in_span": False,
                "both_sources": True,
            },
        ),
    )
    with pytest.raises(RuntimeError, match="'nope' does not resolve to a"):
        c25.chassis_background(synthetic_kt2440)


def test_the_chassis_builder_stops_when_a_documented_gap_starts_resolving(
    synthetic_kt2440: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If `phaC` ever gains a locus, the note's gap is wrong and the build says so."""
    monkeypatch.setattr(c25, "CHASSIS_UNMAPPED_SYMBOLS", ("hbdH",))
    with pytest.raises(RuntimeError, match="it is documented as carrying no locus"):
        c25.chassis_background(synthetic_kt2440)


def test_the_reference_is_built_from_the_served_assembly_report(
    synthetic_kt2440: Any,
) -> None:
    """Species, strain and the GenBank accession, read from the deposited report."""
    reference = c25.chassis_reference(synthetic_kt2440)
    assert reference.species == "Pseudomonas putida"
    assert reference.strain == c25.CHASSIS_STRAIN
    assert reference.assembly_set == c25.KT2440_ASSEMBLY_SET
    assert reference.assembly_accession == "GCA_000007565.2"
    assert reference.background is not None


def test_standard_names_prefers_the_annotations_symbol_and_falls_back_to_the_tag(
    synthetic_kt2440: Any,
) -> None:
    """One gene carries one spelling; a symbol-free locus keeps its tag."""
    names = c25._standard_names(synthetic_kt2440, ["PP_3073", "PP_0815"])
    assert names == {"PP_3073": "hbdH", "PP_0815": "PP_0815"}


# --- both loaders, built end to end under tmp_path ------------------------- #
@pytest.fixture
def built_titer(synthetic_mirror: Path, synthetic_kt2440: Any, tmp_path: Path) -> Any:
    """The titer loader built over the synthetic mirror and annotation."""
    return c25.IsoprenolTiterCarruthers2025Dataset(
        root=str(tmp_path / "build" / "titer"), pputida_genome=synthetic_kt2440
    )


@pytest.fixture
def built_proteome(
    synthetic_mirror: Path, synthetic_kt2440: Any, tmp_path: Path
) -> Any:
    """The proteome loader built over the synthetic mirror and annotation."""
    return c25.ProteomeCarruthers2025Dataset(
        root=str(tmp_path / "build" / "proteome"), pputida_genome=synthetic_kt2440
    )


def test_the_titer_loader_writes_one_record_per_construct_and_cycle(
    built_titer: Any,
) -> None:
    """Strains, not cultures: the 465-shaped arithmetic on a small fixture."""
    expected = (
        len(SINGLE_GUIDE_TARGETS)
        + 2
        + sum(len(v) for v in COMBINATION_CONSTRUCTS.values())
    )
    assert len(built_titer) == expected + SYNTHETIC_PANEL_TITER_RECORDS
    assert built_titer.experiment_class is ProductTiterExperiment
    assert built_titer.reference_class is ProductTiterExperimentReference
    assert built_titer.raw_file_names == [
        c25.SOURCE_DATA_FILENAME,
        c25.TARGETS_FILENAME,
    ]
    record = built_titer[0]
    experiment = record["experiment"]
    dumped = experiment if isinstance(experiment, dict) else experiment.model_dump()
    kinds = {pert["perturbation_type"] for pert in dumped["genotype"]["perturbations"]}
    assert kinds == {"heterologous_pathway", "bacterial_crispr_interference"}
    assert dumped["phenotype"]["titer_unit"] == ConcentrationUnit.ug_per_ml.value
    assert dumped["phenotype"]["sample_unit"] == SampleUnit.biological_replicate.value


def test_the_titer_loader_writes_its_accounting_the_controls_and_the_filter(
    built_titer: Any,
) -> None:
    """One reference per cycle, the panels' drops, and the two side tables."""
    accounting = c25.BuildAccounting.model_validate_json(
        Path(osp.join(built_titer.root, "preprocess/build_accounting.json")).read_text()
    )
    accounting.check()
    assert accounting.source_rows == len(_titer_rows())
    assert accounting.control_rows == 90
    # The only drops are the induced overexpression groups, whose dose has no unit.
    assert accounting.dropped_records == len(c25.OVEREXPRESSION_INDUCED_LEVELS) * len(
        c25.OVEREXPRESSION_SHEET_LABELS
    )
    assert [rule.rule for rule in accounting.rules] == [
        "inducer_concentration_has_no_released_unit"
    ]
    assert accounting.kept_records == len(built_titer)
    assert accounting.reconciliation is not None
    assert accounting.reconciliation.resolved_fraction == 1.0
    assert len(accounting.notes) == 4 + len(
        json.loads(
            Path(osp.join(built_titer.root, "preprocess/panel_proofs.json")).read_text()
        )
    )

    controls = pd.read_csv(osp.join(built_titer.root, "preprocess/cycle_controls.csv"))
    assert controls["cycle"].tolist() == list(range(7))
    assert controls["n_control_cultures"].tolist() == [18] + [12] * 6
    assert (controls["cv_percent"] > 0).all()

    passes = pd.read_csv(osp.join(built_titer.root, "preprocess/pass_filter.csv"))
    assert len(passes) == len(built_titer) - SYNTHETIC_PANEL_TITER_RECORDS
    panels = pd.read_csv(
        osp.join(built_titer.root, "preprocess/source_data_panels.csv")
    )
    assert len(panels) == SYNTHETIC_PANEL_TITER_RECORDS
    assert set(panels["sheet"]) == {
        c25.SHEET_KO_PANEL,
        c25.SHEET_KO_ARRAYS,
        c25.SHEET_OFFTARGET_TITER,
        c25.SHEET_OVEREXPRESSION_TITER,
    }
    assert passes["n_replicates"].max() == 6
    filler = passes.loc[passes["construct"] == FILLER_CONSTRUCT].iloc[0]
    assert filler["non_targeting_tokens"] == "NT1"
    assert filler["n_targets"] == 1


def test_the_titer_loader_keeps_a_six_replicate_strain_as_one_record(
    built_titer: Any,
) -> None:
    """The replicate count stored is the one the strain has, not the paper's 3."""
    records = [built_titer[i] for i in range(len(built_titer))]
    samples = sorted(
        (
            record["experiment"]
            if isinstance(record["experiment"], dict)
            else record["experiment"].model_dump()
        )["phenotype"]["n_samples"]
        for record in records
    )
    assert samples.count(6) == 1
    assert samples.count(3) == len(records) - 1


def test_the_titer_loader_builds_one_reference_per_cycle(built_titer: Any) -> None:
    """Seven cycle controls plus the panels' own, each with its own replicate count.

    The panels add one reference per ``Figure 6a`` KO background and one for the
    overexpression RFP control, whose six uninduced cultures the sheet releases as two
    blocks under one name. ``Figure 6d`` adds none: its control rows are DBTL6 cultures
    already stored, so those records reuse the DBTL6 reference.
    """
    index = json.loads(
        Path(
            osp.join(built_titer.root, "preprocess/experiment_reference_index.json")
        ).read_text()
    )
    assert len(index) == 7 + len(KO_PANEL_BACKGROUNDS) + 1
    counts = sorted(
        entry["reference"]["phenotype_reference"]["n_samples"] for entry in index
    )
    assert counts == [3] * len(KO_PANEL_BACKGROUNDS) + [6] + [12] * 6 + [18]


def test_the_proteome_loader_writes_one_record_per_sample_and_drops_two_keys(
    built_proteome: Any,
) -> None:
    """The PP_0815 panel's records plus the two uninduced overexpression samples."""
    assert len(built_proteome) == (
        len(PROTEOME_SAMPLES) - 1 + c25.EXPECTED_OVEREXPRESSION_PROTEOME_RECORDS
    )
    assert built_proteome.raw_file_names == [c25.SOURCE_DATA_FILENAME]
    record = built_proteome[0]
    experiment = record["experiment"]
    dumped = experiment if isinstance(experiment, dict) else experiment.model_dump()
    abundance = dumped["phenotype"]["protein_abundance"]
    assert set(abundance) == set(PROTEOME_TAGS) | set(
        c25.OVEREXPRESSION_OPERONS["pSTABL1"]
    ) | set(c25.OVEREXPRESSION_OPERONS["pSTABL2"])
    assert dumped["phenotype"]["measurement_type"] == c25.PROTEOME_MEASUREMENT_TYPE
    assert set(dumped["phenotype"]["n_replicates"].values()) == {3}
    kinds = Counter(
        pert["perturbation_type"] for pert in dumped["genotype"]["perturbations"]
    )
    assert kinds["heterologous_pathway"] == 5
    assert kinds["bacterial_deletion"] == 1
    assert kinds["bacterial_crispr_interference"] == 1

    dropped = pd.read_csv(
        osp.join(built_proteome.root, "preprocess/dropped_protein_keys.csv")
    )
    assert set(dropped["protein_key"]) == {PROTEOME_UNRESOLVED, PROTEOME_MERGED}
    merged = dropped.loc[dropped["protein_key"] == PROTEOME_MERGED].iloc[0]
    assert merged["reason"] == "merged_accessions"
    assert merged["accessions"] == "P0AE22;Q88C43"
    unresolved = dropped.loc[dropped["protein_key"] == PROTEOME_UNRESOLVED].iloc[0]
    assert unresolved["reason"] == "not_a_locus_of_the_pinned_assembly"


def test_the_proteome_loader_records_the_target_control_and_the_off_target_samples(
    built_proteome: Any,
) -> None:
    """The PP_0815 guide for the positive control, each sample's own tag otherwise."""
    targets: list[str] = []
    for index in range(len(built_proteome)):
        experiment = built_proteome[index]["experiment"]
        dumped = experiment if isinstance(experiment, dict) else experiment.model_dump()
        targets.extend(
            pert["systematic_gene_name"]
            for pert in dumped["genotype"]["perturbations"]
            if pert["perturbation_type"] == "bacterial_crispr_interference"
        )
    assert sorted(targets) == ["PP_0378", "PP_0815", "PP_0977"]
    samples = pd.read_csv(osp.join(built_proteome.root, "preprocess/samples.csv"))
    assert sorted(samples["sample"]) == sorted(
        [*PROTEOME_SAMPLES[1:], "pSTABL1_Condition_7", "pSTABL2_Condition_7"]
    )
    by_sheet = dict(zip(samples["sample"], samples["sheet"], strict=True))
    assert by_sheet["pSTABL1_Condition_7"] == c25.SHEET_OVEREXPRESSION_PROTEOME
    n_keys = len(PROTEOME_TAGS) + len(OVEREXPRESSION_KEYS)
    assert set(samples["n_proteins"]) == {n_keys, 2, 4}
    # The overexpression records carry only the operon their own strain adds, and the
    # extra native copies are typed as gene additions rather than pathway genes.
    additions: list[str] = []
    for index in range(len(built_proteome)):
        experiment = built_proteome[index]["experiment"]
        dumped = experiment if isinstance(experiment, dict) else experiment.model_dump()
        additions.extend(
            pert["systematic_gene_name"]
            for pert in dumped["genotype"]["perturbations"]
            if pert["perturbation_type"] == "gene_addition"
        )
    assert sorted(additions) == sorted(
        [*c25.OVEREXPRESSION_OPERONS["pSTABL1"], *c25.OVEREXPRESSION_OPERONS["pSTABL2"]]
    )


def test_the_proteome_loaders_accounting_names_both_drop_rules(
    built_proteome: Any,
) -> None:
    """No sample is dropped, and each key rule lists the keys it removed."""
    accounting = c25.BuildAccounting.model_validate_json(
        Path(
            osp.join(built_proteome.root, "preprocess/build_accounting.json")
        ).read_text()
    )
    accounting.check()
    # The only dropped SAMPLES are the induced overexpression ones.
    assert accounting.dropped_records == len(c25.OVEREXPRESSION_INDUCED_LEVELS) * len(
        c25.OVEREXPRESSION_SHEET_LABELS
    )
    by_rule = {rule.rule: rule for rule in accounting.rules}
    assert by_rule["protein_key_is_not_a_locus_of_the_pinned_assembly"].items == [
        PROTEOME_UNRESOLVED
    ]
    assert by_rule["protein_key_merges_two_accessions"].items == [PROTEOME_MERGED]
    assert by_rule["inducer_concentration_has_no_released_unit"].n_records == len(
        by_rule["inducer_concentration_has_no_released_unit"].items
    )
    assert accounting.reconciliation is not None
    assert accounting.reconciliation.resolved_fraction >= 0.94
    assert len(accounting.notes) == 6 + len(
        json.loads(
            Path(
                osp.join(built_proteome.root, "preprocess/panel_proofs.json")
            ).read_text()
        )
    )


def _synthetic_key_set_sizes(built_proteome: Any) -> tuple[int, ...]:
    """The fixture's own per-record protein key-set sizes, in record order."""
    sizes: list[int] = []
    for index in range(len(built_proteome)):
        experiment = built_proteome[index]["experiment"]
        dumped = experiment if isinstance(experiment, dict) else experiment.model_dump()
        sizes.append(len(dumped["phenotype"]["protein_abundance"]))
    return tuple(sizes)


def test_both_loaders_pass_their_full_level_batteries_on_the_synthetic_build(
    built_titer: Any,
    built_proteome: Any,
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L0 to L4 hold on the fixture, both cross-source joins included.

    The module's record and overlap counts are pinned to the real released bytes, so
    the fixture's own shape is patched over them; everything else, including the two
    L4 joins against the synthetic workbooks, is the code the real stores run. The
    fixture's one six-replicate construct is what reconciles its strain count, and its
    filler construct is released under a name no single tag matches, so the oracle
    joins the five single-guide targets plus the six-replicate one.
    """
    monkeypatch.setattr(c25, "bacterial_genome", lambda *a, **k: synthetic_kt2440)
    monkeypatch.setattr(c25, "EXPECTED_TITER_RECORDS", len(built_titer))
    monkeypatch.setattr(c25, "EXPECTED_PROTEOME_RECORDS", len(built_proteome))
    monkeypatch.setattr(
        c25, "PROTEOME_KEY_SET_SIZES", _synthetic_key_set_sizes(built_proteome)
    )
    monkeypatch.setattr(
        c25,
        "EXPECTED_CRISPRI_TITER_RECORDS",
        len(built_titer) - SYNTHETIC_PANEL_TITER_RECORDS,
    )
    monkeypatch.setattr(
        c25, "PAPER_STRAIN_COUNT", len(built_titer) - SYNTHETIC_PANEL_TITER_RECORDS + 1
    )
    monkeypatch.setattr(c25, "SI_TARGET_OVERLAP", len(SINGLE_GUIDE_TARGETS) + 1)
    monkeypatch.setattr(
        c25, "OFFTARGET_SHARED_REFERENCE_RECORDS", 1 + 1 + len(OFFTARGET_SAMPLES)
    )
    monkeypatch.setattr(
        c25,
        "KO_ARRAY_RESULT",
        c25.KO_ARRAY_RESULT.model_copy(
            update={
                "value": {
                    KO_ARRAY_PROSE_LINE: sum(KO_ARRAY_PROSE_TITERS)
                    / len(KO_ARRAY_PROSE_TITERS),
                    "PP_0528_PP_0815": sum(
                        _figure_4b_values(_titer_rows(), "PP_0528_PP_0815", 2)
                    )
                    / 3,
                }
            }
        ),
    )
    # `verify_build` reads each store with its own reader, and LMDB refuses a second
    # open of one environment in a process, so both builders hand their handle back
    # before the batteries run.
    built_titer.close_lmdb()
    built_proteome.close_lmdb()
    families: tuple[tuple[str, c25.Family], ...] = (
        (built_titer.root, "titer"),
        (built_proteome.root, "proteome"),
    )
    for root, family in families:
        report = c25.verify_build(root, str(synthetic_mirror), family=family)
        assert report.passed, [
            (r.level, r.name, r.message) for r in report.results if not r.passed
        ]
        assert report.levels_covered == {
            Level.L0,
            Level.L1,
            Level.L2,
            Level.L3,
            Level.L4,
        }
        written = VerificationReport.model_validate_json(
            Path(root, "preprocess/verification_report.json").read_text()
        )
        assert written.results == report.results


def test_the_titer_l4_refuses_an_oracle_overlap_that_is_not_the_pinned_one(
    built_titer: Any, synthetic_mirror: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The join is pinned, so a target that stops joining stops the report."""
    from torchcell.verification.runners import load_records

    monkeypatch.setattr(c25, "SI_TARGET_OVERLAP", len(SINGLE_GUIDE_TARGETS) + 2)
    with pytest.raises(AssertionError, match="targets join a single-guide record"):
        c25._l4_titer_vs_supplementary_data_1(
            load_records(built_titer.root), str(synthetic_mirror)
        )


def test_the_proteome_l4_refuses_a_stored_protein_the_sheet_does_not_carry(
    built_proteome: Any,
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A store that drifted from its source is caught per protein, never averaged over."""
    from torchcell.verification.runners import load_records

    monkeypatch.setattr(c25, "bacterial_genome", lambda *a, **k: synthetic_kt2440)
    records = load_records(built_proteome.root)
    for record in records:
        record["experiment"]["phenotype"]["protein_abundance"]["PP_9999"] = 1.0
    with pytest.raises(AssertionError, match="are not in the released sheet"):
        c25._l4_proteome_vs_released_sheet(records, str(synthetic_mirror))


def test_the_titer_battery_fails_a_store_whose_se_is_not_sd_over_sqrt_n(
    built_titer: Any, synthetic_mirror: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The derived standard error is checked, not assumed: break one and L2 fails.

    Only the SE rule may fail: breaking a derived statistic must not disturb the counts,
    the partition or either cross-source join, all of which read other fields.
    """
    from torchcell.verification.runners import load_records

    monkeypatch.setattr(c25, "EXPECTED_TITER_RECORDS", len(built_titer))
    monkeypatch.setattr(
        c25,
        "EXPECTED_CRISPRI_TITER_RECORDS",
        len(built_titer) - SYNTHETIC_PANEL_TITER_RECORDS,
    )
    monkeypatch.setattr(
        c25, "PAPER_STRAIN_COUNT", len(built_titer) - SYNTHETIC_PANEL_TITER_RECORDS + 1
    )
    monkeypatch.setattr(c25, "SI_TARGET_OVERLAP", len(SINGLE_GUIDE_TARGETS) + 1)
    monkeypatch.setattr(
        c25, "OFFTARGET_SHARED_REFERENCE_RECORDS", 1 + 1 + len(OFFTARGET_SAMPLES)
    )
    monkeypatch.setattr(
        c25,
        "KO_ARRAY_RESULT",
        c25.KO_ARRAY_RESULT.model_copy(
            update={
                "value": {
                    KO_ARRAY_PROSE_LINE: sum(KO_ARRAY_PROSE_TITERS)
                    / len(KO_ARRAY_PROSE_TITERS),
                    "PP_0528_PP_0815": sum(
                        _figure_4b_values(_titer_rows(), "PP_0528_PP_0815", 2)
                    )
                    / 3,
                }
            }
        ),
    )
    records = load_records(built_titer.root)
    records[0]["experiment"]["phenotype"]["titer_se"] += 1.0
    report = c25.titer_report(records, str(synthetic_mirror))
    broken = [r for r in report.results if not r.passed]
    assert [r.name for r in broken] == ["se_is_the_uncertainty_over_sqrt_n"]
    assert broken[0].details["worst"][0]["index"] == 0


# --- the build-time refusals, each driven by a doctored workbook ----------- #
def _doctor(data_root: Path, mutate: Any) -> None:
    """Rewrite the deposited Source Data workbook through ``mutate``, re-pinning it."""
    path = c25.raw_mirror_dir(str(data_root)) / c25.SOURCE_DATA_REL
    book = openpyxl.load_workbook(path)
    mutate(book)
    book.save(path)
    digest = _sha256_bytes(path)
    manifest = c25.load_manifest(str(data_root))
    for record in manifest.files:
        if record.path == c25.SOURCE_DATA_REL:
            record.sha256 = digest
            assert record.retrieval is not None
            record.retrieval.sha256 = digest
    (c25.raw_mirror_dir(str(data_root)) / "manifest.json").write_text(
        manifest.model_dump_json(indent=2)
    )


@pytest.mark.parametrize(
    "mutate,message",
    [
        (
            lambda book: book[c25.SHEET_CONTROLS].delete_rows(2),
            "control cultures in Figure 4b",
        ),
        (
            lambda book: book[c25.SHEET_TITER].cell(row=2, column=2, value=3),
            "control cultures; the Fig. 2 caption states",
        ),
        (
            lambda book: book[c25.SHEET_CONTROLS].cell(row=2, column=3, value=1.0),
            "control titers disagree by",
        ),
        (
            lambda book: book[c25.SHEET_TARGET_MEANS].cell(row=2, column=4, value=1.0),
            "per-target means disagree",
        ),
    ],
    ids=[
        "control-row-count",
        "control-count-vs-caption",
        "control-value",
        "target-mean",
    ],
)
def test_the_build_refuses_a_workbook_whose_cross_source_checks_fail(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Any,
    message: str,
) -> None:
    """Each assertion in `process` is reachable and names what disagreed."""
    _doctor(synthetic_mirror, mutate)
    monkeypatch.setattr(
        c25,
        "SOURCE_DATA_SHA256",
        _sha256_bytes(c25.raw_mirror_dir(str(synthetic_mirror)) / c25.SOURCE_DATA_REL),
    )
    with pytest.raises(RuntimeError, match=message):
        c25.IsoprenolTiterCarruthers2025Dataset(
            root=str(tmp_path / "refused"), pputida_genome=synthetic_kt2440
        )


def test_the_build_refuses_a_guide_target_that_does_not_resolve(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stated 1.0 threshold stops the build; no record is dropped to pass it.

    The renamed construct is a LATER-cycle one, so the DBTL0 per-target oracles still
    join and the reconciliation threshold is the check that fires.
    """
    doomed = COMBINATION_CONSTRUCTS[5][0]
    renamed = doomed.replace("PP_1593", "PP_9999")

    def rename(book: Any) -> None:
        sheet = book[c25.SHEET_TITER]
        alt = book[c25.SHEET_TITER_ALT]
        for row in range(2, sheet.max_row + 1):
            value = str(sheet.cell(row=row, column=1).value)
            if value.startswith(f"{doomed}-R"):
                swapped = value.replace(doomed, renamed)
                sheet.cell(row=row, column=1, value=swapped)
                alt.cell(row=row, column=1, value=swapped)

    _doctor(synthetic_mirror, rename)
    monkeypatch.setattr(
        c25,
        "SOURCE_DATA_SHA256",
        _sha256_bytes(c25.raw_mirror_dir(str(synthetic_mirror)) / c25.SOURCE_DATA_REL),
    )
    with pytest.raises(LocusTagResolutionError, match="resolve to pputida_KT2440"):
        c25.IsoprenolTiterCarruthers2025Dataset(
            root=str(tmp_path / "refused"), pputida_genome=synthetic_kt2440
        )


def test_the_build_refuses_a_missing_mirror_file(
    synthetic_mirror: Path, synthetic_kt2440: Any, tmp_path: Path
) -> None:
    """A mirror the manifest pins but the disk lacks is named, not skipped."""
    path = c25.raw_mirror_dir(str(synthetic_mirror)) / c25.TARGETS_REL
    path.rename(path.with_suffix(".moved"))
    with pytest.raises(RuntimeError, match="required raw artifact missing from mirror"):
        c25.IsoprenolTiterCarruthers2025Dataset(
            root=str(tmp_path / "refused"), pputida_genome=synthetic_kt2440
        )


def test_the_proteome_build_refuses_a_sheet_with_no_non_targeting_reference(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the control sample there is no reference profile to key records to."""

    def drop_reference(book: Any) -> None:
        sheet = book[c25.SHEET_PROTEOME]
        for row in range(sheet.max_row, 1, -1):
            if sheet.cell(row=row, column=5).value == c25.PROTEOME_REFERENCE_SAMPLE:
                sheet.delete_rows(row)

    _doctor(synthetic_mirror, drop_reference)
    monkeypatch.setattr(
        c25,
        "SOURCE_DATA_SHA256",
        _sha256_bytes(c25.raw_mirror_dir(str(synthetic_mirror)) / c25.SOURCE_DATA_REL),
    )
    with pytest.raises(RuntimeError, match="is not in the released matrix"):
        c25.ProteomeCarruthers2025Dataset(
            root=str(tmp_path / "refused"), pputida_genome=synthetic_kt2440
        )


def test_the_proteome_genotype_refuses_a_sample_name_of_no_known_form() -> None:
    """A sample the released naming does not cover cannot be keyed to a genotype."""
    dataset = c25.ProteomeCarruthers2025Dataset.__new__(
        c25.ProteomeCarruthers2025Dataset
    )
    deletion = BacterialDeletionPerturbation(
        systematic_gene_name="PP_0815",
        perturbed_gene_name="PP_0815",
        gene_namespace="pputida_kt2440_locus_tag",
    )
    with pytest.raises(RuntimeError, match="is neither the non-targeting control"):
        dataset._genotype(
            "JBEI_mystery_48hr", c25.pathway_perturbations(), deletion, {}
        )


def test_a_loader_opens_the_genome_itself_when_nothing_injects_one(
    synthetic_kt2440: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A direct run falls back to `bacterial_genome`; the build entry points inject."""
    calls: list[tuple[str, str]] = []

    def fake(host: str, strain: str, *args: Any, **kwargs: Any) -> Any:
        calls.append((host, strain))
        return synthetic_kt2440

    monkeypatch.setattr(c25, "bacterial_genome", fake)
    dataset = c25.IsoprenolTiterCarruthers2025Dataset.__new__(
        c25.IsoprenolTiterCarruthers2025Dataset
    )
    dataset.pputida_genome = None
    assert dataset._genome() is synthetic_kt2440
    assert calls == [("pputida", "KT2440")]
    assert dataset._genome() is synthetic_kt2440
    assert len(calls) == 1


def test_both_loaders_refuse_the_interface_they_do_not_implement(
    synthetic_mirror: Path, synthetic_kt2440: Any, tmp_path: Path
) -> None:
    """`create_experiment` is inline in `process`, and `preprocess_raw` is a no-op."""
    dataset = c25.IsoprenolTiterCarruthers2025Dataset(
        root=str(tmp_path / "iface"), pputida_genome=synthetic_kt2440
    )
    frame = pd.DataFrame({"a": [1]})
    assert dataset.preprocess_raw(frame) is frame
    with pytest.raises(NotImplementedError):
        dataset.create_experiment()
    proteome = c25.ProteomeCarruthers2025Dataset(
        root=str(tmp_path / "iface_proteome"), pputida_genome=synthetic_kt2440
    )
    assert proteome.preprocess_raw(frame) is frame
    with pytest.raises(NotImplementedError):
        proteome.create_experiment()


def test_main_builds_both_families_and_prints_their_accounting(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The module's interactive entry point, over the synthetic mirror."""
    monkeypatch.setattr(c25, "bacterial_genome", lambda *a, **k: synthetic_kt2440)
    monkeypatch.setattr(c25, "load_dotenv", lambda *a, **k: None, raising=False)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "main_root"))
    (tmp_path / "main_root").mkdir()
    for relative in (c25.RAW_DIR_REL,):
        shutil.copytree(
            c25.raw_mirror_dir(str(synthetic_mirror)), tmp_path / "main_root" / relative
        )
    c25.main()
    out = capsys.readouterr().out
    assert "IsoprenolTiterCarruthers2025Dataset: len =" in out
    assert "ProteomeCarruthers2025Dataset: len =" in out
    # Both families drop exactly the induced overexpression groups, so both print it.
    dropped = len(c25.OVEREXPRESSION_INDUCED_LEVELS) * len(
        c25.OVEREXPRESSION_SHEET_LABELS
    )
    assert out.count(f'"dropped_records": {dropped}') == 2


# --- the four Source Data panels: parsers, proofs and refusals ------------- #
def test_a_locus_designation_expands_to_the_inclusive_range_it_names() -> None:
    """``PP_0812-15`` is four loci; a bare tag is itself; a bad range refuses."""
    assert c25.expand_locus_designation("PP_0815") == ("PP_0815",)
    assert c25.expand_locus_designation("PP_0812-15") == (
        "PP_0812",
        "PP_0813",
        "PP_0814",
        "PP_0815",
    )
    with pytest.raises(RuntimeError, match="does not ascend"):
        c25.expand_locus_designation("PP_0815-12")
    with pytest.raises(RuntimeError, match="neither a PP_ locus tag nor"):
        c25.expand_locus_designation("PP_0812-0815")


def test_a_ko_array_line_splits_into_its_deletions_and_its_knockdowns() -> None:
    """Both released shapes parse, and anything else refuses rather than guessing."""
    assert c25.parse_ko_array_line("PP_0812-15_KO") == (("PP_0812-15",), ())
    assert c25.parse_ko_array_line("PP_0368_PP_0812-15_KO with PP_0528_PP_0815") == (
        ("PP_0368", "PP_0812-15"),
        ("PP_0528", "PP_0815"),
    )
    with pytest.raises(RuntimeError, match="is neither '<designations>_KO'"):
        c25.parse_ko_array_line("PP_0815")
    with pytest.raises(RuntimeError, match="is not PP_ designations joined"):
        c25.parse_ko_array_line("PP_0368-and-PP_0815_KO")


def test_the_partition_proof_refuses_a_reexport_that_is_no_longer_stored() -> None:
    """A ``CRISPRi`` culture missing from ``Figure 4b`` means the sheets diverged."""
    with pytest.raises(RuntimeError, match="are NOT Figure 4b values"):
        c25._assert_panel_partition(
            reexported={c25.SHEET_KO_PANEL: [1.0, 2.0]}, novel={}, stored={1.0}
        )


def test_the_partition_proof_refuses_a_new_culture_that_is_already_stored() -> None:
    """A KO culture that matches a stored titer would store one measurement twice."""
    with pytest.raises(RuntimeError, match="would store a titer twice"):
        c25._assert_panel_partition(
            reexported={}, novel={c25.SHEET_OFFTARGET_TITER: [5.0]}, stored={5.0}
        )


def _ko_panel_row(background: str, arm: Any, titer: float) -> Any:
    return c25.KoPanelRow(background=background, arm=arm, titer_mg_per_l=titer)


def test_the_ko_panel_refuses_a_multi_gene_background_with_no_sourced_sgrna() -> None:
    """A new multi-gene KO background stops the build instead of getting a guess."""
    rows = [
        _ko_panel_row("PP_0812-13", arm, value)
        for arm, value in ((c25.ARM_KO_TARGET, 1.0), (c25.ARM_KO_NONTARGET, 2.0))
    ]
    with pytest.raises(RuntimeError, match="KO_MULTI_GENE_GUIDE must gain a sourced"):
        c25._ko_panel_groups(rows)


def test_the_ko_panel_refuses_a_background_that_is_not_a_pair() -> None:
    """The panel is pairs: a Target arm with no Non-target arm has no reference."""
    with pytest.raises(RuntimeError, match="the panel is pairs"):
        c25._ko_panel_groups([_ko_panel_row("PP_0815", c25.ARM_KO_TARGET, 1.0)])


def test_the_ko_panel_names_the_deleted_gene_as_its_own_target_sgrna() -> None:
    """A single-gene background's Target arm knocks down the gene it deleted."""
    rows = [
        _ko_panel_row("PP_0815", c25.ARM_KO_TARGET, 1.0),
        _ko_panel_row("PP_0815", c25.ARM_KO_NONTARGET, 2.0),
    ]
    records, references, proofs = c25._ko_panel_groups(rows)
    assert [(r.deletions, r.knockdowns) for r in records] == [
        (("PP_0815",), ("PP_0815",))
    ]
    assert references[0].key == "non_target:PP_0815"
    assert references[0].titers_mg_per_l == (2.0,)
    assert proofs == []


def _ko_array_row(line: str, arm: Any, replicate: str, titer: float) -> Any:
    return c25.KoArrayRow(
        line_name=line, arm=arm, replicate=replicate, titer_mg_per_l=titer
    )


def test_the_ko_array_panel_refuses_a_control_that_is_not_a_stored_cycle_culture() -> (
    None
):
    """The panel's reference must BE the stored per-cycle one, not a copy of part of it."""
    rows = [_ko_array_row(c25.KO_ARRAY_CONTROL_LINE, c25.ARM_CRISPRI, "R1", 7.0)]
    with pytest.raises(RuntimeError, match="are not DBTL6 control titers"):
        c25._ko_array_groups(rows, {c25.KO_ARRAY_CONTROL_CYCLE: [1.0, 2.0]})


def test_the_ko_array_panel_refuses_a_missing_control_row() -> None:
    """With no control row the records have no reference to point at."""
    with pytest.raises(RuntimeError, match="has no 'Control' / 'CRISPRi' row"):
        c25._ko_array_groups([], {c25.KO_ARRAY_CONTROL_CYCLE: [1.0]})


def test_the_ko_array_panel_refuses_an_arm_its_line_name_contradicts() -> None:
    """A ``KO Only`` line that parses to a knockdown is a mislabeled release."""
    rows = [
        _ko_array_row(c25.KO_ARRAY_CONTROL_LINE, c25.ARM_CRISPRI, "R1", 1.0),
        _ko_array_row("PP_0812-15_KO with PP_0815", c25.ARM_KO_ONLY, "R1", 9.0),
    ]
    with pytest.raises(RuntimeError, match="but parses to 1 knockdowns"):
        c25._ko_array_groups(rows, {c25.KO_ARRAY_CONTROL_CYCLE: [1.0]})


def _offtarget_reference(values: tuple[float, ...]) -> Any:
    return c25.PanelTiterReference(
        key=f"non_target:{c25.PROTEOME_BACKGROUND_DELETION}",
        sheets=(c25.SHEET_KO_PANEL,),
        group=f"{c25.PROTEOME_BACKGROUND_DELETION}/{c25.ARM_KO_NONTARGET}",
        titers_mg_per_l=values,
        note="the fixture's non-targeting arm",
    )


def _offtarget_rows(pairs: tuple[tuple[str, float], ...]) -> list[Any]:
    return [
        c25.OffTargetTiterRow(sample=sample, replicate="R1", titer_mg_per_l=value)
        for sample, value in pairs
    ]


def test_the_off_target_panel_refuses_a_reference_that_is_not_bit_identical() -> None:
    """The documented deduplication is asserted, not assumed."""
    rows = _offtarget_rows(
        ((c25.OFFTARGET_REFERENCE_SAMPLE, 1.0), (c25.OFFTARGET_TARGET_SAMPLE, 2.0))
    )
    with pytest.raises(RuntimeError, match="is not bit-identical to Figure 6a"):
        c25._offtarget_groups(rows, _offtarget_reference((9.0,)), [2.0])


def test_the_off_target_panel_refuses_a_changed_target_disagreement() -> None:
    """Two triplicates sharing no value, or two, is a different finding to re-decide."""
    rows = _offtarget_rows(
        ((c25.OFFTARGET_REFERENCE_SAMPLE, 1.0), (c25.OFFTARGET_TARGET_SAMPLE, 2.0))
    )
    with pytest.raises(RuntimeError, match="share 0 values, not the 1 this loader"):
        c25._offtarget_groups(rows, _offtarget_reference((1.0,)), [5.0])


def test_the_off_target_panel_refuses_a_sample_it_cannot_read() -> None:
    """An off-target sample must be a locus tag with an optional culture index."""
    rows = _offtarget_rows(
        (
            (c25.OFFTARGET_REFERENCE_SAMPLE, 1.0),
            (c25.OFFTARGET_TARGET_SAMPLE, 2.0),
            ("PP_0378_extra_suffix", 3.0),
        )
    )
    with pytest.raises(RuntimeError, match="nor a PP_xxxx\\[_n\\] off-target"):
        c25._offtarget_groups(rows, _offtarget_reference((1.0,)), [2.0])


def test_the_off_target_panel_refuses_a_changed_candidate_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The screened gene count is Supplementary Table 1's, and it is checked."""
    rows = _offtarget_rows(
        (
            (c25.OFFTARGET_REFERENCE_SAMPLE, 1.0),
            (c25.OFFTARGET_TARGET_SAMPLE, 2.0),
            ("PP_0378", 3.0),
        )
    )
    monkeypatch.setattr(c25, "OFFTARGET_CANDIDATE_TARGETS", 14)
    with pytest.raises(RuntimeError, match="screens 1 distinct off-target genes"):
        c25._offtarget_groups(rows, _offtarget_reference((1.0,)), [2.0])


def test_the_off_target_panel_shares_one_reference_and_keeps_both_triplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dedup and the disagreement, on the arms the release actually ships."""
    monkeypatch.setattr(c25, "OFFTARGET_CANDIDATE_TARGETS", 1)
    non_target = (293.9713, 298.0681, 309.4453)
    ko_target = [422.4448, 464.3335, 495.3173]
    rows = _offtarget_rows(
        tuple((c25.OFFTARGET_REFERENCE_SAMPLE, value) for value in non_target)
        + (
            (c25.OFFTARGET_TARGET_SAMPLE, 464.3335),
            (c25.OFFTARGET_TARGET_SAMPLE, 450.0),
        )
        + (("PP_0378", 1.0),)
    )
    records, reference, proofs = c25._offtarget_groups(
        rows, _offtarget_reference(non_target), ko_target
    )
    assert reference.sheets == (c25.SHEET_KO_PANEL, c25.SHEET_OFFTARGET_TITER)
    assert all(record.reference_key == reference.key for record in records)
    target = next(r for r in records if r.group == c25.OFFTARGET_TARGET_SAMPLE)
    assert target.titers_mg_per_l == (464.3335, 450.0)
    assert target.note is not None
    assert "neither triplicate is stored as the other" in target.note
    assert any("1 shared value (464.3335)" in proof for proof in proofs)


def _overexpression_titer_rows(
    levels: tuple[float, ...], labels: tuple[str, ...]
) -> list[Any]:
    rows = [
        c25.OverexpressionTiterRow(
            strain=label, replicate="R1", inducer_level=level, titer_mg_per_l=1.0
        )
        for label in labels
        for level in levels
    ]
    rows.append(
        c25.OverexpressionTiterRow(
            strain=c25.OVEREXPRESSION_CONTROL_STRAIN,
            replicate="R1",
            inducer_level=c25.OVEREXPRESSION_UNINDUCED_LEVEL,
            titer_mg_per_l=2.0,
        )
    )
    return rows


def test_the_overexpression_panel_refuses_a_changed_induction_series() -> None:
    """A level this loader was not written against stops the build."""
    labels = tuple(c25.OVEREXPRESSION_SHEET_LABELS.values())
    rows = _overexpression_titer_rows((0.0, 2000.0), labels)
    with pytest.raises(RuntimeError, match="releases levels"):
        c25._overexpression_titer_groups(rows)


def test_the_overexpression_panel_refuses_a_changed_strain_label() -> None:
    """The two plotting labels are pinned, because the operon map is keyed on them."""
    levels = (c25.OVEREXPRESSION_UNINDUCED_LEVEL, *c25.OVEREXPRESSION_INDUCED_LEVELS)
    rows = _overexpression_titer_rows(levels, ("pSTABL3 (PP_9999)",))
    with pytest.raises(RuntimeError, match="strain labels .* are not the pinned"):
        c25._overexpression_titer_groups(rows)


def test_the_overexpression_panel_keeps_only_the_uninduced_arm() -> None:
    """One record per label at level 0, and the induced groups in a typed drop rule."""
    levels = (c25.OVEREXPRESSION_UNINDUCED_LEVEL, *c25.OVEREXPRESSION_INDUCED_LEVELS)
    labels = tuple(c25.OVEREXPRESSION_SHEET_LABELS.values())
    records, reference, proofs, drops = c25._overexpression_titer_groups(
        _overexpression_titer_rows(levels, labels)
    )
    assert len(records) == len(labels)
    assert {record.group.split("/")[1] for record in records} == {"0"}
    assert all(record.overexpression_environment for record in records)
    assert {record.native_copies for record in records} == {
        c25.OVEREXPRESSION_OPERONS["pSTABL1"],
        c25.OVEREXPRESSION_OPERONS["pSTABL2"],
    }
    assert reference.key == "overexpression_control"
    assert drops[0].rule == "inducer_concentration_has_no_released_unit"
    assert drops[0].n_records == len(c25.OVEREXPRESSION_INDUCED_LEVELS) * len(labels)
    assert any("uninduced records kept" in proof for proof in proofs)


def _overexpression_proteome_row(
    sample: str, key: str, level: float, description: str, strain: str
) -> Any:
    return c25.OverexpressionProteomeRow(
        sample=sample,
        strain=strain,
        inducer_level=level,
        replicate="R1",
        protein=key,
        accession=f"Q{key}",
        entry_name=f"Q{key}_PSEPK",
        description=description,
        top3_signal=1.0,
    )


def test_the_phn_crosswalk_refuses_a_changed_resolution() -> None:
    """The function-backed crosswalk is asserted, so a swap can never go unnoticed."""
    rows = [
        _overexpression_proteome_row(
            "pSTABL1_Condition_7", key, 0.0, description, "pSTABL1 (PP_2208-09)"
        )
        for key, description in c25.PHN_SHEET_DESCRIPTION.items()
    ]
    with pytest.raises(RuntimeError, match="its FUNCTION puts it at"):
        c25.assert_phn_crosswalk(rows, {"Phnw": "PP_2208", "Phnx": "PP_2209"})


def test_the_phn_crosswalk_refuses_a_corrected_sheet_cell() -> None:
    """If the sheet stops contradicting the assembly, the decision is re-made by hand."""
    rows = [
        _overexpression_proteome_row(
            "pSTABL1_Condition_7", key, 0.0, locus, "pSTABL1 (PP_2208-09)"
        )
        for key, locus in c25.PHN_FUNCTION_CROSSWALK.items()
    ]
    with pytest.raises(RuntimeError, match="not the 'PP_2208' this loader records"):
        c25.assert_phn_crosswalk(rows, dict(c25.PHN_FUNCTION_CROSSWALK))


def test_the_operon_proof_refuses_samples_that_quantify_another_operon() -> None:
    """The overexpressed operon is proved from the proteome of its own samples."""
    rows = [
        _overexpression_proteome_row(
            "pSTABL1_Condition_7", "Phnw", 0.0, "PP_2208", "pSTABL1 (PP_2208-09)"
        )
    ]
    with pytest.raises(RuntimeError, match="samples quantify"):
        c25.assert_overexpression_operons(rows, {"Phnw": "PP_9999"})


def test_the_operon_proof_refuses_a_sample_name_it_cannot_read() -> None:
    """A sample outside the released naming would be silently unattributed."""
    rows = [
        _overexpression_proteome_row(
            "pSTABL1_Round_7", "Phnw", 0.0, "PP_2208", "pSTABL1 (PP_2208-09)"
        )
    ]
    with pytest.raises(RuntimeError, match="is neither '<label>_Condition_<n>'"):
        c25.assert_overexpression_operons(rows, {"Phnw": "PP_2209"})


def test_the_overexpression_proteome_pairs_each_record_with_its_own_control() -> None:
    """Each uninduced sample is referenced against its own label's Control sample."""
    rows: list[Any] = []
    for label, keys in (("pSTABL1", ("Phnw",)), ("pSTABL2", ("Pp_2791",))):
        for sample, level, strain in (
            (f"{label}_Condition_1", 1000.0, f"{label} (x)"),
            (f"{label}_Condition_7", 0.0, f"{label} (x)"),
            (f"{label}_Control", 0.0, c25.OVEREXPRESSION_CONTROL_STRAIN),
        ):
            rows.extend(
                _overexpression_proteome_row(sample, key, level, "d", strain)
                for key in keys
            )
    pairs, proofs = c25.overexpression_proteome_samples(rows)
    assert pairs == {
        "pSTABL1_Condition_7": "pSTABL1_Control",
        "pSTABL2_Condition_7": "pSTABL2_Control",
    }
    assert any("12 induced samples dropped" not in proof for proof in proofs)


def test_the_overexpression_proteome_refuses_a_sample_with_two_inducer_levels() -> None:
    """One sample is one condition; two levels under one name is unreadable."""
    rows = [
        _overexpression_proteome_row("pSTABL1_Condition_7", "Phnw", 0.0, "d", "s"),
        _overexpression_proteome_row("pSTABL1_Condition_7", "Phnw", 500.0, "d", "s"),
    ]
    with pytest.raises(RuntimeError, match="carries two"):
        c25.overexpression_proteome_samples(rows)


def test_the_overexpression_proteome_refuses_a_control_that_is_induced() -> None:
    """A Control at a nonzero level is not the uninduced reference these records need."""
    rows = [
        _overexpression_proteome_row("pSTABL1_Condition_7", "Phnw", 0.0, "d", "s"),
        _overexpression_proteome_row(
            "pSTABL1_Control", "Phnw", 500.0, "d", c25.OVEREXPRESSION_CONTROL_STRAIN
        ),
    ]
    with pytest.raises(RuntimeError, match="is at inducer level"):
        c25.overexpression_proteome_samples(rows)


def test_the_overexpression_proteome_refuses_a_control_of_the_wrong_strain() -> None:
    """The reference sample must be the RFP control, not another overexpression arm."""
    rows = [
        _overexpression_proteome_row("pSTABL1_Condition_7", "Phnw", 0.0, "d", "s"),
        _overexpression_proteome_row(
            "pSTABL1_Control", "Phnw", 0.0, "d", "pSTABL1 (x)"
        ),
    ]
    with pytest.raises(RuntimeError, match="is strain"):
        c25.overexpression_proteome_samples(rows)


def test_the_overexpression_proteome_refuses_a_missing_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A record with no control sample has no reference, so the build stops."""
    rows = [_overexpression_proteome_row("pSTABL1_Condition_7", "Phnw", 0.0, "d", "s")]
    with pytest.raises(RuntimeError, match="has no 'pSTABL1_Control'"):
        c25.overexpression_proteome_samples(rows)


def test_the_overexpression_cells_refuse_a_repeated_replicate() -> None:
    """A doubled replicate would shrink the standard error the record carries."""
    rows = [
        _overexpression_proteome_row("pSTABL1_Condition_7", "Phnw", 0.0, "d", "s")
        for _ in range(2)
    ]
    with pytest.raises(RuntimeError, match="appears twice"):
        c25._overexpression_cells(rows, "pSTABL1_Condition_7", {"Phnw": "PP_2209"})


def test_the_overexpression_cells_refuse_a_sample_the_sheet_does_not_carry() -> None:
    """An empty aggregation is a lookup error, not an empty record."""
    with pytest.raises(RuntimeError, match="has no rows for sample"):
        c25._overexpression_cells([], "pSTABL1_Condition_7", {})


def test_the_overexpression_environment_gaps_its_endpoint_and_keeps_the_rest() -> None:
    """The one slot the panel's own Methods sentence contradicts is a typed gap."""
    base = c25.production_environment()
    panel = c25.overexpression_environment()
    assert base.duration_hours == 48.0
    assert panel.duration_hours is None
    assert [gap.field for gap in panel.provenance_gaps] == ["duration_hours"]
    assert panel.media == base.media
    assert panel.temperature == base.temperature
    assert panel.culture_format == base.culture_format
    assert len(panel.perturbations) == len(base.perturbations)


def test_a_native_extra_copy_is_a_gene_addition_of_the_host_species() -> None:
    """``source_organism`` is the host's, so the containment gate checks the locus."""
    addition = c25.native_copy_perturbation("PP_2208", "phnX")
    assert addition.perturbation_type == "gene_addition"
    assert addition.is_heterologous is False
    assert addition.source_organism == c25.SPECIES
    assert addition.construct_name is None
    deletion = c25.deletion_perturbation("PP_0815", "PP_0815")
    assert deletion.perturbation_type == "bacterial_deletion"
    assert deletion.gene_namespace == c25.KT2440_NAMESPACE
    assert deletion.collection is None


def test_the_two_panel_sheets_must_be_one_export_of_one_panel(tmp_path: Path) -> None:
    """``Supplementary Figure 11`` and ``Figure 6a`` are the same rows, and that holds."""
    path = tmp_path / "book.xlsx"
    book = openpyxl.Workbook()
    panel = book.active
    panel.title = c25.SHEET_KO_PANEL
    panel.append(["Strain", "Type", "Titer"])
    panel.append(["PP_0815", c25.ARM_KO_TARGET, 1.0])
    alt = book.create_sheet(c25.SHEET_KO_PANEL_ALT)
    alt.append(["Strain", "Type", "Titer"])
    alt.append(["PP_0815", c25.ARM_KO_TARGET, 2.0])
    book.save(path)
    with pytest.raises(RuntimeError, match="are not the same rows"):
        c25.read_ko_panel_rows(str(path))


def test_each_new_panel_reader_refuses_a_changed_header(tmp_path: Path) -> None:
    """Every panel reader asserts its released header before it reads a value."""
    cases: tuple[tuple[str, Any, str], ...] = (
        (c25.SHEET_KO_ARRAYS, c25.read_ko_array_rows, "header changed"),
        (c25.SHEET_OFFTARGET_TITER, c25.read_offtarget_titer_rows, "header changed"),
        (
            c25.SHEET_OVEREXPRESSION_TITER,
            c25.read_overexpression_titer_rows,
            "header changed",
        ),
        (
            c25.SHEET_OVEREXPRESSION_PROTEOME,
            c25.read_overexpression_proteome_rows,
            "header changed",
        ),
    )
    for sheet, reader, message in cases:
        path = tmp_path / f"{sheet}.xlsx"
        book = openpyxl.Workbook()
        active = book.active
        active.title = sheet
        active.append(["wrong", "header", "cells", "here"])
        active.append([1, 2, 3, 4])
        book.save(path)
        with pytest.raises(RuntimeError, match=message):
            reader(str(path))


def test_the_ko_panel_reader_refuses_a_changed_header(tmp_path: Path) -> None:
    """The KO panel's own header check covers both of its two sheets."""
    path = tmp_path / "ko.xlsx"
    book = openpyxl.Workbook()
    panel = book.active
    panel.title = c25.SHEET_KO_PANEL
    panel.append(["wrong", "header", "cells"])
    panel.append([1, 2, 3])
    alt = book.create_sheet(c25.SHEET_KO_PANEL_ALT)
    alt.append(["wrong", "header", "cells"])
    alt.append([1, 2, 3])
    book.save(path)
    with pytest.raises(RuntimeError, match="header changed"):
        c25.read_ko_panel_rows(str(path))


def test_the_off_target_panel_refuses_a_missing_arm() -> None:
    """Both anchors are required: each one carries one of the two duplications."""
    reference = _offtarget_reference((1.0,))
    with pytest.raises(RuntimeError, match="has no 'Non-Target' arm"):
        c25._offtarget_groups(_offtarget_rows((("PP_0378", 1.0),)), reference, [2.0])
    with pytest.raises(RuntimeError, match="has no 'Target' arm"):
        c25._offtarget_groups(
            _offtarget_rows(((c25.OFFTARGET_REFERENCE_SAMPLE, 1.0),)), reference, [2.0]
        )


def test_the_overexpression_proteome_refuses_a_changed_record_count() -> None:
    """The uninduced arm is two samples on the pinned bytes, and that is asserted."""
    rows = [
        _overexpression_proteome_row("pSTABL1_Condition_7", "Phnw", 0.0, "d", "s"),
        _overexpression_proteome_row(
            "pSTABL1_Control", "Phnw", 0.0, "d", c25.OVEREXPRESSION_CONTROL_STRAIN
        ),
    ]
    with pytest.raises(RuntimeError, match="yields 1 uninduced records"):
        c25.overexpression_proteome_samples(rows)


def test_the_proteome_build_refuses_a_panel_key_the_other_sheet_lacks(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One protein must key one way across both matrices, so a stray key refuses."""
    _doctor(
        synthetic_mirror,
        lambda book: book[c25.SHEET_OVEREXPRESSION_PROTEOME].cell(
            row=2, column=3, value="Phnz"
        ),
    )
    monkeypatch.setattr(
        c25,
        "SOURCE_DATA_SHA256",
        c25.manifest_sha256(
            c25.load_manifest(str(synthetic_mirror)), c25.SOURCE_DATA_REL
        ),
    )
    with pytest.raises(RuntimeError, match="which Supplementary Figure 13abc does not"):
        c25.ProteomeCarruthers2025Dataset(
            root=str(tmp_path / "refuse" / "proteome"), pputida_genome=synthetic_kt2440
        )


def test_the_prose_l4_refuses_a_genotype_the_store_does_not_hold_uniquely(
    built_titer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sentence names two exact strains; a store without them cannot be joined."""
    from torchcell.verification.runners import load_records

    records = load_records(built_titer.root)
    monkeypatch.setattr(
        c25,
        "KO_ARRAY_RESULT",
        c25.KO_ARRAY_RESULT.model_copy(update={"value": {"PP_9999_PP_9998": 1.0}}),
    )
    with pytest.raises(AssertionError, match="each must be unique"):
        c25._l4_ko_array_titer_vs_results_text(records)
