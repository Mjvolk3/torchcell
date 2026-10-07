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

import json
import math
import os
import os.path as osp
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
    ConcentrationUnit,
    CultureEnvironment,
    CultureFormat,
    Environment,
    Genotype,
    ProductTiterExperiment,
    ProductTiterExperimentReference,
    SampleUnit,
    SmallMoleculePerturbation,
    UncertaintyType,
)
from torchcell.literature.manifest import Manifest
from torchcell.verification.levels import (
    l0_structural,
    l1_count,
    l2_cross_method,
    l2_value_fidelity,
    l3_convention,
    l4_cross_source,
)
from torchcell.verification.report import Level


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


def test_the_product_is_the_compound_layers_isoprenol_with_its_identity_gap() -> None:
    """The product goes through the shared resolver, which has no isoprenol row yet."""
    product = c25.isoprenol_product()
    assert product.name == "isoprenol"
    assert product.inchikey is None
    assert product.gapped_fields() == {"inchikey"}


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
    """M9-NREL at 24 C for 48 h, aerobic, with the 2 g/L L-arabinose inducer."""
    env = c25.production_environment()
    assert type(env) is Environment
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


def test_a_culture_environment_loses_its_protocol_inside_a_product_titer_experiment() -> (
    None
):
    """The measured reason the loader stores a plain ``Environment``.

    ``Experiment.environment`` is annotated ``Environment`` and pydantic serializes by
    the declared type, so the vessel, volume and shaking of a ``CultureEnvironment``
    are dropped on dump with no error. When ``ProductTiterExperiment`` narrows that
    slot, this test is what says so.
    """
    env = CultureEnvironment(
        media=M9_NREL_CARRUTHERS2025,
        culture_format=CultureFormat(
            vessel="48-well BioLector flower plate",
            working_volume_ul=1500.0,
            shaking_rpm=1000.0,
        ),
    )
    perturbations: list[Any] = list(c25.pathway_perturbations())
    experiment = ProductTiterExperiment(
        dataset_name="probe",
        genotype=Genotype(perturbations=perturbations),
        environment=env,
        phenotype=c25.titer_phenotype([1.0, 2.0, 3.0], is_reference=False),
    )
    assert env.culture_format is not None
    assert "culture_format" not in experiment.model_dump()["environment"]


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
    """L0 schema, L1 count, L2 value fidelity and cross-method, L3 units, L4 oracle."""
    from torchcell.verification.runners import load_records

    records = load_records(TITER_ROOT)
    report = _titer_levels(records)
    assert all(result.passed for result in report), [
        (r.level, r.name, r.message) for r in report if not r.passed
    ]


@pytest.mark.data
@requires_built
def test_the_built_proteome_store_has_one_record_per_sample_and_passes_l0_to_l4() -> (
    None
):
    """The same five levels over the protein-abundance family."""
    from torchcell.verification.runners import load_records

    records = load_records(PROTEOME_ROOT)
    report = _proteome_levels(records)
    assert all(result.passed for result in report), [
        (r.level, r.name, r.message) for r in report if not r.passed
    ]


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
# The L0-L4 level batteries (shared by the data-gated tests and ``main``)
# --------------------------------------------------------------------------- #
def _titer_levels(records: list[dict[str, Any]]) -> list[Any]:
    """L0 to L4 over the product-titer family."""
    experiments = [record["experiment"] for record in records]
    titers = [exp["phenotype"]["titer"] for exp in experiments]
    uncertainties = [exp["phenotype"]["titer_uncertainty"] for exp in experiments]
    sqrt_n = [
        exp["phenotype"]["titer_uncertainty"] / math.sqrt(exp["phenotype"]["n_samples"])
        for exp in experiments
    ]
    derived = [exp["phenotype"]["titer_se"] for exp in experiments]
    return [
        l0_structural(experiments, ProductTiterExperiment.model_validate),
        l1_count(len(records), 465),
        l2_value_fidelity(titers, minimum=0.0),
        l2_value_fidelity(uncertainties, minimum=0.0),
        l2_cross_method(derived, sqrt_n, tol=1e-9),
        l3_convention(
            "titer_unit_is_the_sources_mg_per_l_as_ug_per_ml",
            all(
                exp["phenotype"]["titer_unit"] == ConcentrationUnit.ug_per_ml.value
                for exp in experiments
            ),
            detail="1 mg/L == 1 ug/mL exactly, so the released number is stored verbatim",
        ),
        l3_convention(
            "every_genotype_carries_the_five_pathway_genes",
            all(
                sum(
                    1
                    for pert in exp["genotype"]["perturbations"]
                    if pert["perturbation_type"] == "heterologous_pathway"
                )
                == 5
                for exp in experiments
            ),
            detail="IY1449b + pIY670 is the production strain IY1452b",
        ),
        _titer_l4(experiments),
    ]


def _titer_l4(experiments: list[dict[str, Any]]) -> Any:
    """L4: the store's single-guide titers against Supplementary Data 1's own means.

    The oracle is a DIFFERENT released file from the one the loader reads, so this
    joins the built records to an independent statement of the same measurement. Its
    means are printed to two decimals, hence the 0.005 tolerance.
    """
    si_path = osp.join(DATA_ROOT, c25.RAW_DIR_REL, c25.TARGETS_REL)
    released = c25.read_si_target_means(si_path)
    by_tag: dict[str, list[float]] = {}
    for exp in experiments:
        targets = [
            pert["systematic_gene_name"]
            for pert in exp["genotype"]["perturbations"]
            if pert["perturbation_type"] == "bacterial_crispr_interference"
        ]
        if len(targets) == 1:
            by_tag.setdefault(targets[0], []).append(exp["phenotype"]["titer"])
    shared = [
        (tag, min(by_tag[tag], key=lambda t: abs(t - mean)), mean)
        for tag, mean in sorted(released.items())
        if tag in by_tag
    ]
    if len(shared) != 120:
        raise AssertionError(
            f"{len(shared)} of Supplementary Data 1's {len(released)} targets join a "
            "single-guide record; all 120 do on the pinned bytes. The record-level "
            "join reaches two the construct-name join cannot (PP_1607 and PP_4194 are "
            "released only as the NT-filler names PP_1607_NT1 and PP_4194_NT2, whose "
            "filler guide is not a perturbation)"
        )
    return l4_cross_source(shared, tol=5e-3).model_copy(
        update={"name": "single_guide_titer_vs_supplementary_data_1"}
    )


def _proteome_levels(records: list[dict[str, Any]]) -> list[Any]:
    """L0 to L4 over the protein-abundance family."""
    from torchcell.datamodels.schema import BacterialProteinAbundanceExperiment

    experiments = [record["experiment"] for record in records]
    abundances = [
        value
        for exp in experiments
        for value in exp["phenotype"]["protein_abundance"].values()
    ]
    key_counts = [len(exp["phenotype"]["protein_abundance"]) for exp in experiments]
    replicate_counts = [
        n for exp in experiments for n in exp["phenotype"]["n_replicates"].values()
    ]
    return [
        l0_structural(experiments, BacterialProteinAbundanceExperiment.model_validate),
        l1_count(len(records), 19),
        l2_value_fidelity(abundances, minimum=0.0),
        l2_cross_method(key_counts, [1424] * len(key_counts), tol=0.0),
        l3_convention(
            "every_protein_key_is_a_kt2440_locus_tag",
            all(
                key.startswith("PP_")
                for exp in experiments
                for key in exp["phenotype"]["protein_abundance"]
            ),
            detail="the 77 non-host keys are dropped by a sourced rule",
        ),
        l3_convention(
            "every_sample_is_a_biological_triplicate",
            set(replicate_counts) == {3},
            detail="Supplementary Fig. 13: 'All strains were cultured in triplicate'",
        ),
        _proteome_l4(experiments),
    ]


def _proteome_l4(experiments: list[dict[str, Any]]) -> Any:
    """L4: the store's PP_0815-target profile against the released sheet, re-read.

    Every stored abundance must be reproducible from the deposited bytes by the same
    aggregation, so a store that drifted from its source is caught per protein.
    """
    from torchcell.datasets.bacteria_common import (
        bacterial_genome,
        reconcile_locus_tags,
    )

    rows = c25.read_proteome_rows(
        osp.join(DATA_ROOT, c25.RAW_DIR_REL, c25.SOURCE_DATA_REL)
    )
    genome = bacterial_genome("pputida", "KT2440")
    keys = sorted({row.protein for row in rows})
    stored_keys, _ = reconcile_locus_tags(genome, pd.Series(keys), label="l4")
    key_map = dict(zip(keys, stored_keys, strict=True))
    cells: dict[str, list[float]] = {}
    for row in rows:
        if row.sample != c25.PROTEOME_TARGET_SAMPLE:
            continue
        cells.setdefault(key_map[row.protein], []).append(row.top3_signal)
    target = next(
        exp
        for exp in experiments
        if any(
            pert["systematic_gene_name"] == c25.PROTEOME_BACKGROUND_DELETION
            and pert["perturbation_type"] == "bacterial_crispr_interference"
            for pert in exp["genotype"]["perturbations"]
        )
    )
    abundance = target["phenotype"]["protein_abundance"]
    shared = [
        (key, value, sum(cells[key]) / len(cells[key]))
        for key, value in sorted(abundance.items())
        if key in cells
    ]
    if len(shared) != len(abundance):
        raise AssertionError(
            f"{len(abundance) - len(shared)} stored proteins are not in the released "
            "sheet under their reconciled key"
        )
    return l4_cross_source(shared, tol=1e-6).model_copy(
        update={"name": "stored_target_profile_vs_released_sheet"}
    )


def main() -> int:
    """Print the L0-L4 report for both built stores (the PR's verifier numbers)."""
    from dotenv import load_dotenv

    load_dotenv()
    from torchcell.verification.runners import load_records

    failures = 0
    for label, root, battery in (
        ("isoprenol titer", TITER_ROOT, _titer_levels),
        ("proteome", PROTEOME_ROOT, _proteome_levels),
    ):
        print(f"=== {label} ({root}) ===")
        for result in battery(load_records(root)):
            flag = "PASS" if result.passed else "FAIL"
            level = (
                result.level.value if isinstance(result.level, Level) else result.level
            )
            print(f"  [{flag}] {level} {result.name}: {result.message}")
            failures += not result.passed
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
