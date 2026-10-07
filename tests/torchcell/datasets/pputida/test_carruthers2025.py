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
    ("PP_0812", None),
    ("PP_0815", None),
    ("PP_0977", None),
    ("PP_1593", None),
    ("PP_2664", None),
    ("PP_3379", None),
    ("PP_5313", None),
    ("PP_5424", "apha"),
)
#: The seventeen keys the synthetic proteome sheet uses as locus tags; ``PP_5424``
#: is reached only through the symbol ``Apha``, so the two never collide.
PROTEOME_TAGS: tuple[str, ...] = tuple(
    tag for tag, _ in LOCUS_SPECS if tag != "PP_5424"
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
    2: ("PP_0378_PP_0815",),
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
    book.save(path)


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
    expected = len(PROTEOME_SAMPLES) * 3 * (len(PROTEOME_TAGS) + 1 + 2)
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
    assert len(built_titer) == expected
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
    """Zero drops, one reference per cycle, and the two side tables."""
    accounting = c25.BuildAccounting.model_validate_json(
        Path(osp.join(built_titer.root, "preprocess/build_accounting.json")).read_text()
    )
    accounting.check()
    assert accounting.source_rows == len(_titer_rows())
    assert accounting.control_rows == 90
    assert accounting.dropped_records == 0
    assert accounting.kept_records == len(built_titer)
    assert accounting.reconciliation is not None
    assert accounting.reconciliation.resolved_fraction == 1.0
    assert len(accounting.notes) == 3

    controls = pd.read_csv(osp.join(built_titer.root, "preprocess/cycle_controls.csv"))
    assert controls["cycle"].tolist() == list(range(7))
    assert controls["n_control_cultures"].tolist() == [18] + [12] * 6
    assert (controls["cv_percent"] > 0).all()

    passes = pd.read_csv(osp.join(built_titer.root, "preprocess/pass_filter.csv"))
    assert len(passes) == len(built_titer)
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
    """Seven cycles, seven control phenotypes, each with its own replicate count."""
    index = json.loads(
        Path(
            osp.join(built_titer.root, "preprocess/experiment_reference_index.json")
        ).read_text()
    )
    assert len(index) == 7
    counts = sorted(
        entry["reference"]["phenotype_reference"]["n_samples"] for entry in index
    )
    assert counts == [12] * 6 + [18]


def test_the_proteome_loader_writes_one_record_per_sample_and_drops_two_keys(
    built_proteome: Any,
) -> None:
    """Three records (the non-targeting control is the reference) over 17 keys."""
    assert len(built_proteome) == len(PROTEOME_SAMPLES) - 1
    assert built_proteome.raw_file_names == [c25.SOURCE_DATA_FILENAME]
    record = built_proteome[0]
    experiment = record["experiment"]
    dumped = experiment if isinstance(experiment, dict) else experiment.model_dump()
    abundance = dumped["phenotype"]["protein_abundance"]
    assert set(abundance) == set(PROTEOME_TAGS)
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
    assert sorted(samples["sample"]) == sorted(PROTEOME_SAMPLES[1:])
    assert set(samples["n_proteins"]) == {len(PROTEOME_TAGS)}


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
    assert accounting.dropped_records == 0
    by_rule = {rule.rule: rule for rule in accounting.rules}
    assert by_rule["protein_key_is_not_a_locus_of_the_pinned_assembly"].items == [
        PROTEOME_UNRESOLVED
    ]
    assert by_rule["protein_key_merges_two_accessions"].items == [PROTEOME_MERGED]
    assert accounting.reconciliation is not None
    assert accounting.reconciliation.resolved_fraction >= 0.94
    assert len(accounting.notes) == 4


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
    monkeypatch.setattr(c25, "PROTEOME_KEYS_PER_RECORD", len(PROTEOME_TAGS))
    monkeypatch.setattr(c25, "PAPER_STRAIN_COUNT", len(built_titer) + 1)
    monkeypatch.setattr(c25, "SI_TARGET_OVERLAP", len(SINGLE_GUIDE_TARGETS) + 1)
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


def test_the_titer_battery_fails_a_store_whose_se_is_not_sd_over_sqrt_n(
    built_titer: Any, synthetic_mirror: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The derived standard error is checked, not assumed: break one and L2 fails."""
    from torchcell.verification.runners import load_records

    monkeypatch.setattr(c25, "EXPECTED_TITER_RECORDS", len(built_titer))
    monkeypatch.setattr(c25, "PAPER_STRAIN_COUNT", len(built_titer) + 1)
    monkeypatch.setattr(c25, "SI_TARGET_OVERLAP", len(SINGLE_GUIDE_TARGETS) + 1)
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
    assert '"dropped_records": 0' in out
