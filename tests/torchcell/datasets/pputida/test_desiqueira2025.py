# tests/torchcell/datasets/pputida/test_desiqueira2025.py
# [[tests.torchcell.datasets.pputida.test_desiqueira2025]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/pputida/test_desiqueira2025.py
"""The de Siqueira 2025 P. putida acetate-tolerization loaders.

Synthetic tests run everywhere: they exercise the two released-file readers, the variant
ledger that types what cannot be written, the Table S2 census, the PT background and
genotype builders against a recording fake genome, both environments, the two
cross-source assertions, the retention arithmetic, and both loaders built end to end
under ``tmp_path`` with no network and no ``$DATA_ROOT``.

The ``@pytest.mark.data`` tests read the real ``$DATA_ROOT``: they assert that every
module quote is a verbatim substring of the sha256-pinned mirrored bytes, pin the raw
mirror's recorded digests against the module constants, and run L0 to L4 over both built
LMDBs.

Derived expectations for the pinned released files, every one measured: Data Set S1
holds 34,600 cells over 1,730 protein groups and 20 samples, of which 5 are of a
writable strain and 1,531 protein keys survive into a record's abundance map; Data Set
S2 holds 173 variant calls over 5 sequenced clones at 83 distinct sites, 58 with a
GenBank locus tag, 10 with a RefSeq tag only and 105 with none; Table S2 holds 24 cells,
of which 6 are ``n.d.``, 14 belong to an unwritable strain, 2 sit in a medium the
library does not hold and 2 become records.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import os.path as osp
from pathlib import Path
from typing import Any

import openpyxl
import pytest
from pydantic import ValidationError

import torchcell.datasets.pputida.desiqueira2025 as ds
from torchcell.datamodels.media import M9_NREL_DESIQUEIRA2025
from torchcell.datamodels.schema import (
    AlleleEdit,
    BacterialProteinAbundanceExperiment,
    BacterialProteinAbundanceExperimentReference,
    ConcentrationUnit,
    EnvironmentPhysicalPerturbation,
    PhysicalFactor,
    ProductTiterExperiment,
    ProductTiterExperimentReference,
    ProductTiterPhenotype,
    SampleUnit,
    SmallMoleculePerturbation,
)
from torchcell.datasets.pputida.carruthers2025 import PATHWAY_GENES, PIY670_PARTS
from torchcell.literature.manifest import ROLE_SI_DATA, RetrievalMethod

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


def _serve_assembly_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve the KT2440 assembly report from ``tmp_path``, with no tier on disk."""
    import torchcell.datasets.bacteria_common as bacteria_common

    report = tmp_path / ASSEMBLY_REPORT_MEMBER
    report.write_text(ASSEMBLY_REPORT)

    def serve(assembly_set: str, filename: str, **_: Any) -> str:
        if filename != ASSEMBLY_REPORT_MEMBER:
            raise FileNotFoundError(f"{assembly_set}/{filename} is not in the fixture")
        return str(report)

    monkeypatch.setattr(bacteria_common, "resolve", serve)


@pytest.fixture
def served_assembly_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The assembly report the reference builder reads, with no genomes tier."""
    _serve_assembly_report(tmp_path, monkeypatch)


# --------------------------------------------------------------------------- #
# Synthetic: the genotype finding, which is what decides the record set
# --------------------------------------------------------------------------- #
def test_the_pt_background_types_one_deletion_event_under_two_designations() -> None:
    """PP_2675 is a full deletion; the Delta14 half is a PARTIAL deletion."""
    background = ds.pt_background()
    assert background.name == ds.PT_STRAIN
    assert background.reference_strain == ds.WT_STRAIN
    assert background.assembly_set == ds.KT2440_ASSEMBLY_SET
    edits = {allele.systematic_gene_name: allele.edit for allele in background.alleles}
    assert edits == {
        ds.PT_FULL_DELETION: AlleleEdit.full_deletion,
        ds.PT_PARTIAL_DELETION: AlleleEdit.partial_deletion,
    }
    assert all(not allele.functional for allele in background.alleles)
    assert background.genotype_statement == str(ds.PT_GENOTYPE.value)


def test_the_partial_deletion_carries_a_typed_gap_on_its_span() -> None:
    """The source writes 'Delta14' with no unit and no coordinates, so span is gapped."""
    background = ds.pt_background()
    partial = next(
        a
        for a in background.alleles
        if a.systematic_gene_name == ds.PT_PARTIAL_DELETION
    )
    assert partial.deleted_span is None
    assert partial.gapped_fields() == {"deleted_span"}
    assert not partial.is_sourced
    full = next(
        a for a in background.alleles if a.systematic_gene_name == ds.PT_FULL_DELETION
    )
    assert full.is_sourced
    assert full.deleted_span is None


def test_the_full_deletion_cites_the_mirrored_paper_the_source_defers_to() -> None:
    """The PP_2675 half is sourced from Thompson 2020, not guessed from the delta."""
    assert ds.PT_FULL_DELETION_SOURCE.provenance.citation_key == ds.THOMPSON_KEY
    assert ds.PT_FULL_DELETION_SOURCE.provenance.sha256 == ds.THOMPSON_SHA256
    assert "complete internal in-frame deletion" in ds.PT_FULL_DELETION_SOURCE.quote


def test_the_wild_type_reference_carries_no_background_and_pt_carries_one(
    served_assembly_report: None,
) -> None:
    """A WT record IS the reference assembly; a PT record names its own strain."""
    wild = ds.strain_reference(ds.WT_LABEL)
    assert wild.background is None
    assert wild.strain == ds.WT_STRAIN
    assert wild.assembly_set == ds.KT2440_ASSEMBLY_SET
    assert wild.assembly_accession == "GCA_000007565.2"
    pt = ds.strain_reference(ds.PT_STRAIN)
    assert pt.background is not None
    assert pt.strain == ds.PT_STRAIN


@pytest.mark.parametrize("strain", list(ds.SIGMA_STRAINS))
def test_a_tolerized_isolate_has_no_reference_because_it_cannot_be_written(
    strain: str,
) -> None:
    """Asking for a Sigma strain's reference refuses rather than inventing one."""
    with pytest.raises(RuntimeError, match="not a writable strain"):
        ds.strain_reference(strain)
    with pytest.raises(RuntimeError, match="not a writable strain"):
        ds.strain_genotype(strain, with_pathway=False)


def test_the_two_writable_genotypes_are_distinguishable() -> None:
    """WT carries nothing, PT carries its deletion, and a titer strain adds pIY670."""
    assert ds.strain_genotype(ds.WT_LABEL, with_pathway=False).perturbations == []
    pt = ds.strain_genotype(ds.PT_STRAIN, with_pathway=False)
    assert pt.systematic_gene_names == [ds.PT_FULL_DELETION]
    assert pt.perturbation_types == ["bacterial_deletion"]
    with_plasmid = ds.strain_genotype(ds.PT_STRAIN, with_pathway=True)
    assert len(with_plasmid) == 1 + len(PATHWAY_GENES)
    assert with_plasmid != pt


def test_the_pt_perturbation_carries_the_registry_accession() -> None:
    """The JBEI ICE code is the strain discriminator the source gives."""
    perturbation = ds.pt_perturbation()
    assert perturbation.construction is not None
    assert perturbation.construction.strain_accession == str(ds.PT_ICE_ACCESSION.value)
    assert perturbation.gene_namespace == ds.KT2440_NAMESPACE


# --------------------------------------------------------------------------- #
# Synthetic: the typed gap for the called variants
# --------------------------------------------------------------------------- #
VARIANT_HEADER_ROW: tuple[str, ...] = (
    "Protein Effect",
    "old_locus_tag",
    "Name",
    "Type",
    "Sequence",
    "Minimum",
    "Maximum",
    "Length",
    "# Intervals",
    "Coverage",
    "Polymorphism Type",
    "Variant Frequency",
    "Track Name",
    "Strain",
    "Document Name",
    "Sequence Name",
    "Sequence (with extension)",
    "Change",
    "Amino Acid Change",
    "CDS",
    "CDS Codon Number",
    "CDS Position",
    "Codon Change",
    "gene",
    "locus_tag",
    "note",
    "product",
)

#: (strain, old_locus_tag, refseq_locus_tag, protein effect) of the synthetic calls.
VARIANT_SPECS: tuple[tuple[str, str | None, str | None, str], ...] = (
    ("PT", None, None, "None"),
    ("PT", "PP_0180", "PP_RS00965", "None"),
    (ds.SIGMA_STRAINS[0], "PP_1656", "PP_RS08530", "Substitution"),
    (ds.SIGMA_STRAINS[0], "PP_1656", "PP_RS08530", "Substitution"),
    (ds.SIGMA_STRAINS[0], None, "PP_RS21780", "Substitution"),
    (ds.SIGMA_STRAINS[3], None, None, "None"),
)


def _variant_row(
    index: int, strain: str, tag: str | None, refseq: str | None, effect: str
) -> list[Any]:
    """One synthetic Data Set S2 row, with the released column order and cell types."""
    row: list[Any] = [None] * len(VARIANT_HEADER_ROW)
    position = 100 + index
    row[0] = effect
    row[1] = tag
    row[2] = "G"
    row[3] = "Polymorphism"
    row[4] = "A"
    row[5] = position
    row[6] = position
    row[7] = 1
    row[8] = 1
    row[9] = "61 -> 63" if index == 0 else str(20 + index)
    row[10] = "SNP (transition)"
    row[11] = "0.5 -> 0.6" if index == 0 else "0.97"
    row[12] = f"Variants: GS_{index}"
    row[13] = strain
    row[17] = "A -> G"
    row[20] = 12 if effect != "None" else None
    row[22] = "GGT -> GAT" if effect != "None" else None
    row[23] = "relA" if tag == "PP_1656" else None
    row[24] = refseq
    row[26] = "GTP diphosphokinase" if tag == "PP_1656" else None
    return row


def _write_variants(path: Path) -> None:
    """A synthetic Data Set S2 workbook over :data:`VARIANT_SPECS`."""
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(list(VARIANT_HEADER_ROW))
    for index, (strain, tag, refseq, effect) in enumerate(VARIANT_SPECS):
        sheet.append(_variant_row(index, strain, tag, refseq, effect))
    book.save(path)
    book.close()


def test_read_variant_calls_types_every_released_cell(tmp_path: Path) -> None:
    """Coverage and frequency stay verbatim strings; a range would break a float."""
    path = tmp_path / "variants.xlsx"
    _write_variants(path)
    calls = ds.read_variant_calls(str(path))
    assert len(calls) == len(VARIANT_SPECS)
    assert calls[0].coverage == "61 -> 63"
    assert calls[0].variant_frequency == "0.5 -> 0.6"
    assert calls[1].genbank_locus_tag == "PP_0180"
    assert calls[1].refseq_locus_tag == "PP_RS00965"
    assert calls[2].gene_symbol == "relA"
    assert calls[2].cds_codon_number == 12
    assert calls[0].protein_effect is None


def test_every_call_names_the_missing_variant_leaf_and_its_own_blocker(
    tmp_path: Path,
) -> None:
    """The four typed blockers, each assigned from the row it is measured on."""
    path = tmp_path / "variants.xlsx"
    _write_variants(path)
    calls = ds.read_variant_calls(str(path))
    assert all(ds.REASON_NO_VARIANT_LEAF in c.blocking_reasons for c in calls)
    assert ds.REASON_NO_LOCUS in calls[0].blocking_reasons
    assert ds.REASON_FUNCTIONAL_UNSTATED in calls[1].blocking_reasons
    assert ds.REASON_LOCUS_SEEN_TWICE not in calls[1].blocking_reasons
    assert ds.REASON_LOCUS_SEEN_TWICE in calls[2].blocking_reasons
    assert ds.REASON_LOCUS_SEEN_TWICE in calls[3].blocking_reasons
    assert ds.REASON_REFSEQ_ONLY in calls[4].blocking_reasons


def test_the_variant_ledger_counts_sites_strains_and_unsequenced_isolates(
    tmp_path: Path,
) -> None:
    """The ledger is what the additive schema proposal in the PR rests on."""
    path = tmp_path / "variants.xlsx"
    _write_variants(path)
    ledger = ds.variant_ledger(ds.read_variant_calls(str(path)))
    assert ledger.n_calls == len(VARIANT_SPECS)
    assert ledger.n_distinct_sites == len(VARIANT_SPECS)
    assert ledger.n_sites_in_more_than_one_strain == 0
    assert ledger.calls_with_genbank_locus == 3
    assert ledger.calls_with_refseq_locus_only == 1
    assert ledger.calls_with_no_locus == 2
    assert ledger.loci_claimed_twice == {ds.SIGMA_STRAINS[0]: ["PP_1656 x2"]}
    assert ledger.unsequenced_strains == [
        ds.SIGMA_STRAINS[1],
        ds.SIGMA_STRAINS[2],
        ds.SIGMA_STRAINS[4],
    ]
    assert ledger.reasons[ds.REASON_NO_VARIANT_LEAF] == len(VARIANT_SPECS)


def test_read_variant_calls_refuses_a_changed_header(tmp_path: Path) -> None:
    """A released column order this reader was not written against stops the build."""
    path = tmp_path / "variants.xlsx"
    book = openpyxl.Workbook()
    book.active.append(["Something", "Else"])
    book.active.append([1, 2])
    book.save(path)
    book.close()
    with pytest.raises(RuntimeError, match="Data Set S2 header changed"):
        ds.read_variant_calls(str(path))


def test_read_variant_calls_refuses_an_empty_release(tmp_path: Path) -> None:
    """A header with no rows is not a release."""
    path = tmp_path / "variants.xlsx"
    book = openpyxl.Workbook()
    book.active.append(list(VARIANT_HEADER_ROW))
    book.save(path)
    book.close()
    with pytest.raises(RuntimeError, match="released no variant calls"):
        ds.read_variant_calls(str(path))


# --------------------------------------------------------------------------- #
# Synthetic: Table S2 and the two cross-source assertions
# --------------------------------------------------------------------------- #
def test_the_table_s2_census_partitions_every_released_cell() -> None:
    """24 cells: 6 n.d., 14 unwritable, 2 in an absent medium, 2 loaded."""
    census = ds.titer_census()
    assert census.n_cells == len(ds.TITER_COLUMNS) * (1 + len(ds.SIGMA_STRAINS))
    assert census.n_not_determined == 6
    assert census.n_unwritable_strain == 14
    assert census.n_medium_not_in_library == 2
    assert census.n_loaded == len(ds.TITER_COLUMNS_LOADED) == 2


def test_the_census_refuses_a_partition_that_loses_a_cell() -> None:
    """The partition is checked, not assumed."""
    census = ds.TiterCensus(
        n_cells=24,
        n_not_determined=6,
        n_unwritable_strain=14,
        n_medium_not_in_library=2,
        n_loaded=1,
    )
    with pytest.raises(RuntimeError, match="does not partition"):
        census.check()


def test_the_released_titer_table_keeps_not_determined_as_none() -> None:
    """'n.d.' is not a measurement and is never read as a zero."""
    table = ds.released_titers()
    assert table[ds.PT_STRAIN][ds.TITER_REFERENCE_COLUMN] == 3.29
    assert table[ds.PT_STRAIN][ds.TITER_COLUMNS[3]] == 0.05
    assert table[ds.SIGMA_STRAINS[1]][ds.TITER_COLUMNS[0]] is None
    assert table[ds.SIGMA_STRAINS[0]][ds.TITER_COLUMNS[0]] == 0.0


def test_the_table_s2_glucose_cell_agrees_with_the_results_text_in_mg_per_l() -> None:
    """3.29 mM and 283 mg/L are the same measurement to within Table S2's rounding."""
    difference = ds.assert_titer_cross_source()
    assert difference < ds.TITER_CROSS_SOURCE_TOL_MM
    assert difference == pytest.approx(
        abs(3.29 - 283.0 / ds.ISOPRENOL_G_PER_MOL), abs=1e-12
    )


def test_the_cross_source_check_refuses_a_table_that_no_longer_agrees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed released number stops the build rather than being stored."""
    table = ds.released_titers()
    table[ds.PT_STRAIN][ds.TITER_REFERENCE_COLUMN] = 9.99
    monkeypatch.setattr(ds, "released_titers", lambda: table)
    with pytest.raises(RuntimeError, match="disagree by"):
        ds.assert_titer_cross_source()


def test_the_cross_source_check_refuses_an_undetermined_reference_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An 'n.d.' reference cell is not something to check against."""
    table = ds.released_titers()
    table[ds.PT_STRAIN][ds.TITER_REFERENCE_COLUMN] = None
    monkeypatch.setattr(ds, "released_titers", lambda: table)
    with pytest.raises(RuntimeError, match="not determined"):
        ds.assert_titer_cross_source()


def test_the_piy670_part_strings_agree_except_for_the_known_ocr_loss() -> None:
    """What licenses reusing the Carruthers-sourced organisms for the five genes."""
    ds.assert_piy670_matches_carruthers()
    mine = str(ds.PIY670_DESIQUEIRA_PARTS.value).casefold()
    theirs = str(PIY670_PARTS.value).casefold()
    assert mine != theirs
    assert mine.replace("pmdhkq", "pmdschkq") == theirs


def test_the_piy670_assertion_refuses_a_second_difference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any drift beyond the recorded 'Sc' loss stops the build."""
    broken = str(ds.PIY670_DESIQUEIRA_PARTS.value).replace("AphA", "AphB")
    monkeypatch.setattr(
        ds,
        "PIY670_DESIQUEIRA_PARTS",
        ds.PIY670_DESIQUEIRA_PARTS.model_copy(update={"value": broken}),
    )
    with pytest.raises(RuntimeError, match="no longer agree"):
        ds.assert_piy670_matches_carruthers()


def test_the_piy670_assertion_refuses_a_different_part_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A construct with another part is another construct."""
    broken = str(ds.PIY670_DESIQUEIRA_PARTS.value) + "-Extra"
    monkeypatch.setattr(
        ds,
        "PIY670_DESIQUEIRA_PARTS",
        ds.PIY670_DESIQUEIRA_PARTS.model_copy(update={"value": broken}),
    )
    with pytest.raises(RuntimeError, match="part counts differ"):
        ds.assert_piy670_matches_carruthers()


# --------------------------------------------------------------------------- #
# Synthetic: the phenotypes and the environments
# --------------------------------------------------------------------------- #
def test_the_titer_is_stored_in_the_released_millimolar_with_no_arithmetic() -> None:
    """ConcentrationUnit has mM, so Table S2's number is stored unchanged."""
    phenotype = ds.titer_phenotype(3.29)
    assert phenotype.titer == 3.29
    assert phenotype.titer_unit is ConcentrationUnit.millimolar
    assert phenotype.n_samples == int(ds.TITER_N_REPLICATES.value)
    assert phenotype.sample_unit is SampleUnit.biological_replicate
    assert phenotype.quantification_method == "GC-FID"


def test_the_titer_gaps_the_uncertainty_the_table_never_released() -> None:
    """A maximum with no uncertainty is gapped, never given an invented SD."""
    phenotype = ds.titer_phenotype(0.05)
    assert phenotype.titer_uncertainty is None
    assert phenotype.titer_uncertainty_type is None
    assert phenotype.titer_se is None
    assert phenotype.gapped_fields() == {
        "titer_uncertainty",
        "titer_uncertainty_type",
        "product_yield",
        "product_yield_unit",
        "productivity",
        "productivity_unit",
    }


def test_the_titer_phenotype_refuses_a_non_finite_value() -> None:
    """A titer is a measured amount."""
    with pytest.raises(RuntimeError, match="finite and non-negative"):
        ds.titer_phenotype(float("nan"))
    with pytest.raises(RuntimeError, match="finite and non-negative"):
        ds.titer_phenotype(-1.0)


def test_the_product_is_isoprenol_with_the_table_s_own_identity() -> None:
    """The compound table carries the isoprenol row, and the module's key cross-checks it."""
    product = ds.isoprenol_product()
    assert product.name == "isoprenol"
    assert ds.ISOPRENOL_INCHIKEY == "CPJRRXSHAYUTGL-UHFFFAOYSA-N"
    assert product.inchikey == ds.ISOPRENOL_INCHIKEY
    assert product.pubchem_cid == 12988
    assert product.gapped_fields() == set()


@pytest.mark.parametrize(
    ("condition", "n_carbon"),
    [(ds.CONDITION_ACETATE, 1), (ds.CONDITION_GLUCOSE, 1), (ds.CONDITION_MIXED, 2)],
)
def test_each_proteomics_condition_carries_its_carbon_sources_as_typed_factors(
    condition: str, n_carbon: int
) -> None:
    """The medium object has no carbon source; the loader carries it, as media.py says."""
    environment = ds.proteome_environment(condition)
    assert environment.media is M9_NREL_DESIQUEIRA2025
    assert len(environment.perturbations) == n_carbon
    carbon = [
        p
        for p in environment.perturbations
        if isinstance(p, EnvironmentPhysicalPerturbation)
    ]
    assert len(carbon) == n_carbon
    assert all(p.factor is PhysicalFactor.carbon_source for p in carbon)
    assert environment.temperature is not None
    assert environment.temperature.value == 30.0
    assert environment.duration_hours is None
    assert environment.gapped_fields() == {"duration_hours"}


def test_the_acetate_condition_uses_the_sourced_millimolar_default() -> None:
    """The Methods' sole-carbon default is 50 mM acetate, and it is stored as mM."""
    environment = ds.proteome_environment(ds.CONDITION_ACETATE)
    acetate = environment.perturbations[0]
    assert isinstance(acetate, EnvironmentPhysicalPerturbation)
    dose = acetate.magnitude
    assert dose is not None
    assert dose.value == 50.0
    assert dose.unit is ConcentrationUnit.millimolar


def test_proteome_environment_refuses_an_unreleased_condition() -> None:
    """Only the three released conditions exist."""
    with pytest.raises(RuntimeError, match="not a released proteomics condition"):
        ds.proteome_environment("Xylose")


def test_the_titer_environment_adds_the_inducer_and_the_selection() -> None:
    """Arabinose induces the pathway and kanamycin maintains the plasmid."""
    environment = ds.titer_environment(ds.TITER_COLUMNS[3])
    kinds = [p.perturbation_type for p in environment.perturbations]
    assert kinds.count("environment_physical") == 2
    assert kinds.count("small_molecule") == 2
    names = {
        p.compound.name
        for p in environment.perturbations
        if isinstance(p, SmallMoleculePerturbation)
    }
    assert names == {"L-arabinose", "kanamycin"}
    assert environment.duration_hours is None
    assert environment.gapped_fields() == {"duration_hours"}


def test_the_glucose_only_titer_environment_has_one_carbon_source() -> None:
    """The reference column is glucose alone."""
    environment = ds.titer_environment(ds.TITER_REFERENCE_COLUMN)
    carbon = [
        p
        for p in environment.perturbations
        if isinstance(p, EnvironmentPhysicalPerturbation)
    ]
    assert len(carbon) == 1
    magnitude = carbon[0].magnitude
    assert magnitude is not None
    assert magnitude.unit is ConcentrationUnit.percent_w_v


def test_the_titer_environment_refuses_a_column_whose_medium_is_absent() -> None:
    """Fig. S3's media A and C are not the served M9 object."""
    with pytest.raises(RuntimeError, match="not a Table S2 column this loader writes"):
        ds.titer_environment(ds.TITER_COLUMNS[0])


def test_the_pathway_perturbations_are_the_five_plasmid_genes() -> None:
    """Each is heterologous, episomal and on the pIY670 construct."""
    pathway = ds.pathway_perturbations()
    assert len(pathway) == len(PATHWAY_GENES)
    assert {p.construct_name for p in pathway} == {"pIY670"}
    assert all(p.is_heterologous for p in pathway)
    assert all(p.localization == "episomal_plasmid" for p in pathway)
    assert all(p.gene_namespace == ds.KT2440_NAMESPACE for p in pathway)
    assert {p.systematic_gene_name for p in pathway} == {
        str(gene["token"]) for gene in PATHWAY_GENES
    }


def test_the_publication_carries_the_doi_and_no_invented_pubmed_id() -> None:
    """The mirror records no PubMed id for this paper, so none is written."""
    publication = ds.publication()
    assert publication.doi == ds.DOI
    assert publication.pubmed_id is None


# --------------------------------------------------------------------------- #
# Synthetic: the retention arithmetic
# --------------------------------------------------------------------------- #
def test_the_accounting_refuses_an_arithmetic_that_loses_a_record() -> None:
    """Kept plus dropped must be the candidate count."""
    accounting = ds.BuildAccounting(
        dataset="x",
        source_rows=10,
        candidate_records=5,
        kept_records=2,
        dropped_records=2,
    )
    with pytest.raises(RuntimeError, match="!= 5 candidates"):
        accounting.check()


def test_the_accounting_refuses_drops_no_rule_accounts_for() -> None:
    """A dropped record with no rule naming it is an unexplained loss."""
    accounting = ds.BuildAccounting(
        dataset="x",
        source_rows=10,
        candidate_records=5,
        kept_records=2,
        dropped_records=3,
        rules=[ds.DropRule(rule="r", scope="sample", description="d", n_records=1)],
    )
    with pytest.raises(RuntimeError, match="record-scoped rules total 1"):
        accounting.check()


def test_protein_key_rules_do_not_count_toward_the_record_arithmetic() -> None:
    """A dropped KEY removes a column, not a record."""
    accounting = ds.BuildAccounting(
        dataset="x",
        source_rows=10,
        candidate_records=5,
        kept_records=5,
        dropped_records=0,
        rules=[
            ds.DropRule(
                rule="k",
                scope="protein_key",
                description="d",
                n_records=0,
                items=["Krt1"],
            )
        ],
    )
    accounting.check()
    assert accounting.rules[0].items == ["Krt1"]


# --------------------------------------------------------------------------- #
# Hermetic end to end: a synthetic KT2440 assembly, a synthetic raw mirror, and
# both loaders built under tmp_path. No network, no $DATA_ROOT.
# --------------------------------------------------------------------------- #
#: The loci the synthetic assembly carries: PT's two, plus loci whose symbols the
#: synthetic proteome sheet keys on.
LOCUS_SPECS: tuple[tuple[str, str | None], ...] = (
    (ds.PT_FULL_DELETION, None),
    (ds.PT_PARTIAL_DELETION, None),
    ("PP_0154", "spcC"),
    ("PP_0988", "gcvP-I"),
    ("PP_1656", "relA"),
    ("PP_2037", None),
    ("PP_4264", "hemN"),
    ("PP_4872", "ygiQ"),
    ("PP_0168", None),
    ("PP_0673", None),
    ("PP_1458", None),
    ("PP_1623", "rpoS"),
    ("PP_1649", "ldhA"),
    ("PP_1743", "actP-I"),
    ("PP_2025", None),
    ("PP_2674", "pedE"),
    ("PP_3827", None),
    ("PP_4387", None),
)
#: Keys the synthetic sheet files as protein identifiers: six locus tags, one symbol
#: that resolves, one key no layer resolves, and one filed under two accessions.
PROTEOME_RESOLVING_TAGS: tuple[str, ...] = (
    "PP_0154",
    "PP_0988",
    "PP_1656",
    "PP_2037",
    "PP_4264",
    "PP_0168",
    "PP_0673",
    "PP_1458",
    "PP_1623",
    "PP_1649",
    "PP_1743",
    "PP_2025",
    "PP_2674",
    "PP_3827",
    "PP_4387",
)
PROTEOME_SYMBOL_KEY = "Ygiq"
PROTEOME_UNRESOLVED_KEY = "Krt1"
PROTEOME_MERGED_KEY = "Pyrc"
PROTEOME_KEYS: tuple[str, ...] = (
    *PROTEOME_RESOLVING_TAGS,
    PROTEOME_SYMBOL_KEY,
    PROTEOME_UNRESOLVED_KEY,
    PROTEOME_MERGED_KEY,
)
#: The (strain, condition) samples the synthetic sheet releases: the real 20-sample
#: shape in miniature, with PT missing the mixed-carbon sample it never had.
PROTEOME_SAMPLES: tuple[tuple[str, str], ...] = (
    *((ds.WT_LABEL, condition) for condition in ds.PROTEOME_CONDITIONS),
    (ds.PT_STRAIN, ds.CONDITION_ACETATE),
    (ds.PT_STRAIN, ds.CONDITION_GLUCOSE),
    *(
        (strain, condition)
        for strain in ds.SIGMA_STRAINS
        for condition in ds.PROTEOME_CONDITIONS
    ),
)


def _synthetic_loci() -> list[Any]:
    """One ``SyntheticLocus`` per :data:`LOCUS_SPECS` entry, laid end to end."""
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
        [fixtures.gaf_row("hemN", f"hemN|{LOCUS_SPECS[6][0]}", "GO:0000001")],
    )
    fixtures.forbid_network(monkeypatch)
    fixtures.serve_tier(monkeypatch, files)
    _serve_assembly_report(tmp_path, monkeypatch)
    root = tmp_path / "kt2440"
    root.mkdir()
    return PPutidaKT2440Genome(genome_root=str(root), overwrite=False)


def _write_proteome(path: Path) -> None:
    """A synthetic Data Set S1 over :data:`PROTEOME_SAMPLES` and :data:`PROTEOME_KEYS`."""
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append([*ds.PROTEOME_HEADER, "extra"])
    for strain, condition in PROTEOME_SAMPLES:
        for index, key in enumerate(PROTEOME_KEYS):
            accessions = (
                ("Q00001", "Q00002")
                if key == PROTEOME_MERGED_KEY
                else (f"Q{index:05d}",)
            )
            for accession in accessions:
                sheet.append(
                    [
                        accession,
                        f"{key.upper()}_PSEPK",
                        key,
                        f"synthetic {key}",
                        strain,
                        condition,
                        f"{strain}_{condition}",
                        1000.0 + index + len(strain) + len(condition),
                        30.0 + index,
                        None,
                    ]
                )
    book.save(path)
    book.close()


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def synthetic_mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A raw mirror under ``tmp_path`` built by the module's own deposit function."""
    staging = tmp_path / "staging"
    staging.mkdir()
    proteome = staging / ds.PROTEOME_FILENAME
    variants = staging / ds.VARIANTS_FILENAME
    _write_proteome(proteome)
    _write_variants(variants)
    monkeypatch.setattr(ds, "PROTEOME_SHA256", _sha256_bytes(proteome))
    monkeypatch.setattr(ds, "VARIANTS_SHA256", _sha256_bytes(variants))
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    root = ds.deposit_raw_mirror(
        proteome_path=proteome, variants_path=variants, data_root=str(data_root)
    )
    assert root == ds.raw_mirror_dir(str(data_root))
    return data_root


# --- the deposit itself ---------------------------------------------------- #
def test_the_deposit_writes_both_files_and_a_manifest_that_pins_them(
    synthetic_mirror: Path,
) -> None:
    """Two ``si_data`` records, each with a re-runnable ``pmc_cloud`` retrieval."""
    manifest = ds.load_manifest(str(synthetic_mirror))
    assert manifest.citation_key == ds.CITATION_KEY
    assert manifest.doi == ds.DOI
    assert manifest.title == ds.TITLE
    assert [record.path for record in manifest.files] == [
        ds.PROTEOME_REL,
        ds.VARIANTS_REL,
    ]
    for record in manifest.files:
        assert record.role == ROLE_SI_DATA
        assert record.retrieval is not None
        assert record.retrieval.method is RetrievalMethod.pmc_cloud
        assert record.retrieval.retriever == (
            "torchcell.literature.retrieve.pmc_cloud_object"
        )
        assert record.retrieval.params["key"].startswith(f"{ds.PMCID}.1/")
        assert record.retrieval.sha256 == record.sha256
    assert ds.manifest_sha256(manifest, ds.PROTEOME_REL) == ds.PROTEOME_SHA256
    assert len(manifest.si_data_sources) == 5
    assert manifest.provenance_complete is True


def test_the_manifest_enumerates_the_deposits_it_deliberately_does_not_hold(
    synthetic_mirror: Path,
) -> None:
    """The SRA BioProject and the PRIDE project are named, not silently skipped."""
    manifest = ds.load_manifest(str(synthetic_mirror))
    expected = " ".join(manifest.si_expected)
    assert ds.SRA_BIOPROJECT in expected
    assert ds.PRIDE_ACCESSION in expected
    assert "Fig. 3B and 3C" in expected


def test_the_deposit_is_idempotent_by_sha256(
    synthetic_mirror: Path, tmp_path: Path
) -> None:
    """Re-depositing identical bytes leaves the mirror alone."""
    target = ds.raw_mirror_dir(str(synthetic_mirror)) / ds.PROTEOME_REL
    before = target.stat().st_mtime_ns
    ds.deposit_raw_mirror(
        proteome_path=tmp_path / "staging" / ds.PROTEOME_FILENAME,
        variants_path=tmp_path / "staging" / ds.VARIANTS_FILENAME,
        data_root=str(synthetic_mirror),
    )
    assert target.stat().st_mtime_ns == before


def test_the_deposit_refuses_a_mirror_file_whose_bytes_differ(
    synthetic_mirror: Path, tmp_path: Path
) -> None:
    """A differing mirror file raises rather than being overwritten."""
    target = ds.raw_mirror_dir(str(synthetic_mirror)) / ds.VARIANTS_REL
    target.write_bytes(b"not the released workbook")
    with pytest.raises(RuntimeError, match="different sha256; refusing"):
        ds.deposit_raw_mirror(
            proteome_path=tmp_path / "staging" / ds.PROTEOME_FILENAME,
            variants_path=tmp_path / "staging" / ds.VARIANTS_FILENAME,
            data_root=str(synthetic_mirror),
        )


def test_the_deposit_refuses_a_source_file_off_its_pin(tmp_path: Path) -> None:
    """Both files are verified BEFORE anything is written."""
    staging = tmp_path / "s"
    staging.mkdir()
    proteome = staging / ds.PROTEOME_FILENAME
    variants = staging / ds.VARIANTS_FILENAME
    _write_proteome(proteome)
    _write_variants(variants)
    with pytest.raises(Exception):
        ds.deposit_raw_mirror(
            proteome_path=proteome,
            variants_path=variants,
            data_root=str(tmp_path / "out"),
        )
    assert not (tmp_path / "out").exists()


def test_raw_mirror_dir_reads_data_root_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The mirror path is derived, never configured twice."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert ds.raw_mirror_dir() == tmp_path / ds.RAW_DIR_REL


def test_manifest_sha256_refuses_a_path_the_manifest_does_not_hold(
    synthetic_mirror: Path,
) -> None:
    """A pin can only be read for a file the manifest records."""
    manifest = ds.load_manifest(str(synthetic_mirror))
    with pytest.raises(KeyError, match="not in the raw-mirror manifest"):
        ds.manifest_sha256(manifest, "si/nope.xlsx")


# --- the proteome reader --------------------------------------------------- #
def test_read_proteome_rows_types_every_released_cell(tmp_path: Path) -> None:
    """The nine consumed columns, typed; extra columns are ignored."""
    path = tmp_path / "p.xlsx"
    _write_proteome(path)
    rows = ds.read_proteome_rows(str(path))
    assert len(rows) == len(PROTEOME_SAMPLES) * (len(PROTEOME_KEYS) + 1)
    assert {row.strain for row in rows} == {
        ds.WT_LABEL,
        ds.PT_STRAIN,
        *ds.SIGMA_STRAINS,
    }
    assert rows[0].top3_mean > 0
    assert rows[0].top3_sd > 0


def test_read_proteome_rows_refuses_a_changed_header(tmp_path: Path) -> None:
    """A released column order this reader was not written against stops the build."""
    path = tmp_path / "p.xlsx"
    book = openpyxl.Workbook()
    book.active.append(["Protein.Group", "Nope"])
    book.active.append(["Q1", 2])
    book.save(path)
    book.close()
    with pytest.raises(RuntimeError, match="Data Set S1 header changed"):
        ds.read_proteome_rows(str(path))


# --- both loaders, built end to end under tmp_path ------------------------- #
@pytest.fixture
def built_proteome(
    synthetic_mirror: Path, synthetic_kt2440: Any, tmp_path: Path
) -> Any:
    """The proteome loader built over the synthetic mirror and annotation."""
    return ds.ProteomeDeSiqueira2025Dataset(
        root=str(tmp_path / "build" / "proteome"), pputida_genome=synthetic_kt2440
    )


@pytest.fixture
def built_titer(synthetic_mirror: Path, synthetic_kt2440: Any, tmp_path: Path) -> Any:
    """The titer loader built over the synthetic mirror."""
    return ds.IsoprenolTiterDeSiqueira2025Dataset(
        root=str(tmp_path / "build" / "titer"), pputida_genome=synthetic_kt2440
    )


def test_the_proteome_loader_writes_only_the_writable_strains(
    built_proteome: Any,
) -> None:
    """Five records out of the released samples, and the Sigma samples left out."""
    assert len(built_proteome) == ds.EXPECTED_PROTEOME_RECORDS
    assert built_proteome.experiment_class is BacterialProteinAbundanceExperiment
    assert (
        built_proteome.reference_class is BacterialProteinAbundanceExperimentReference
    )
    assert built_proteome.raw_file_names == [ds.PROTEOME_FILENAME, ds.VARIANTS_FILENAME]
    strains = set()
    for index in range(len(built_proteome)):
        record = built_proteome[index]
        strains.add(str(record["reference"]["genome_reference"]["strain"]))
    assert strains == {ds.WT_STRAIN, ds.PT_STRAIN}


def test_the_proteome_records_drop_the_unresolvable_and_merged_keys(
    built_proteome: Any,
) -> None:
    """Six keys survive: five tags plus the one symbol that resolves."""
    record = built_proteome[0]
    abundance = record["experiment"]["phenotype"]["protein_abundance"]
    assert set(abundance) == {*PROTEOME_RESOLVING_TAGS, LOCUS_SPECS[7][0]}
    assert PROTEOME_UNRESOLVED_KEY not in abundance
    assert PROTEOME_MERGED_KEY not in abundance
    reference = record["reference"]["phenotype_reference"]["protein_abundance"]
    assert set(reference) == set(abundance)


def test_the_proteome_se_is_the_released_sd_over_the_root_of_three(
    built_proteome: Any,
) -> None:
    """The released column is a sample SD, so it is divided before being stored."""
    record = built_proteome[0]
    phenotype = record["experiment"]["phenotype"]
    assert phenotype["measurement_type"] == ds.PROTEOME_MEASUREMENT_TYPE
    assert set(phenotype["n_replicates"].values()) == {3}
    for tag, se in phenotype["protein_abundance_se"].items():
        assert se == pytest.approx(se)
        assert se > 0
        assert tag in phenotype["protein_abundance"]
    index = PROTEOME_RESOLVING_TAGS.index("PP_0154")
    assert phenotype["protein_abundance_se"]["PP_0154"] == pytest.approx(
        (30.0 + index) / math.sqrt(3)
    )


def test_the_proteome_loader_writes_its_accounting_and_its_variant_ledger(
    built_proteome: Any,
) -> None:
    """Every sample left out is accounted for by the one genotype rule."""
    preprocess = Path(built_proteome.preprocess_dir)
    accounting = json.loads((preprocess / "build_accounting.json").read_text())
    assert accounting["candidate_records"] == len(PROTEOME_SAMPLES)
    assert accounting["kept_records"] == ds.EXPECTED_PROTEOME_RECORDS
    assert accounting["dropped_records"] == (
        len(PROTEOME_SAMPLES) - ds.EXPECTED_PROTEOME_RECORDS
    )
    rules = {rule["rule"]: rule for rule in accounting["rules"]}
    assert (
        rules["strain_genotype_is_not_representable"]["n_records"]
        == (accounting["dropped_records"])
    )
    assert (
        PROTEOME_UNRESOLVED_KEY
        in (rules["protein_key_is_not_a_locus_of_the_pinned_assembly"]["items"])
    )
    assert rules["protein_key_merges_two_accessions"]["items"] == [PROTEOME_MERGED_KEY]
    ledger = json.loads((preprocess / "called_variants.json").read_text())
    assert ledger["n_calls"] == len(VARIANT_SPECS)
    dropped = (preprocess / "dropped_protein_keys.csv").read_text()
    assert PROTEOME_UNRESOLVED_KEY in dropped
    assert (preprocess / "samples.csv").exists()


def test_the_proteome_reference_is_the_wild_type_in_the_glucose_condition(
    built_proteome: Any,
) -> None:
    """The reference the Fig. 4 caption names, and it stays a record too."""
    baselines = set()
    for index in range(len(built_proteome)):
        record = built_proteome[index]
        reference = record["reference"]
        assert reference["environment_reference"]["media"]["name"] == (
            M9_NREL_DESIQUEIRA2025.name
        )
        baselines.add(
            json.dumps(
                reference["phenotype_reference"]["protein_abundance"], sort_keys=True
            )
        )
    assert len(baselines) == 1


def test_the_titer_loader_writes_one_record_per_loaded_table_cell(
    built_titer: Any,
) -> None:
    """Two records, both PT carrying pIY670."""
    assert len(built_titer) == len(ds.TITER_COLUMNS_LOADED)
    assert built_titer.experiment_class is ProductTiterExperiment
    assert built_titer.reference_class is ProductTiterExperimentReference
    assert built_titer.raw_file_names == [ds.VARIANTS_FILENAME]
    record = built_titer[0]
    experiment = record["experiment"]
    kinds = {p["perturbation_type"] for p in experiment["genotype"]["perturbations"]}
    assert kinds == {"bacterial_deletion", "heterologous_pathway"}
    assert experiment["phenotype"]["titer_unit"] == ConcentrationUnit.millimolar.value
    assert record["reference"]["genome_reference"]["strain"] == ds.PT_STRAIN


def test_the_titer_accounting_names_all_three_reasons_a_cell_is_left_out(
    built_titer: Any,
) -> None:
    """The unwritable strains, the n.d. cells and the absent medium, each counted."""
    preprocess = Path(built_titer.preprocess_dir)
    accounting = json.loads((preprocess / "build_accounting.json").read_text())
    assert accounting["candidate_records"] == 24
    assert accounting["kept_records"] == 2
    assert accounting["dropped_records"] == 22
    rules = {rule["rule"]: rule["n_records"] for rule in accounting["rules"]}
    assert rules == {
        "strain_genotype_is_not_representable": 14,
        "cell_is_not_determined": 6,
        "medium_is_not_in_the_media_library": 2,
    }
    notes = " ".join(accounting["notes"])
    assert "DISAGREEMENT" in notes
    assert (preprocess / "titers.csv").exists()
    assert (preprocess / "called_variants.json").exists()


def test_both_loaders_pass_their_level_batteries_on_the_synthetic_build(
    built_proteome: Any, built_titer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L0 to L4 over the hermetic stores, through the module's own runner."""
    genome = built_proteome.pputida_genome
    monkeypatch.setattr(ds, "bacterial_genome", lambda *a, **k: genome)
    for dataset, family in ((built_proteome, "proteome"), (built_titer, "titer")):
        report = ds.verify_build(dataset.root, family=family)
        assert report.passed, report.summary()
        assert {result.level.value for result in report.results} == {0, 1, 2, 3, 4}
        written = Path(dataset.root, "preprocess", "verification_report.json")
        assert json.loads(written.read_text())["dataset_name"] == report.dataset_name


def test_verify_build_refuses_a_family_it_does_not_know(
    built_titer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the two released families exist."""
    monkeypatch.setattr(
        ds, "bacterial_genome", lambda *a, **k: built_titer.pputida_genome
    )
    with pytest.raises(RuntimeError, match="is neither 'proteome' nor 'titer'"):
        ds.verify_build(built_titer.root, family="growth")


def test_the_supplementary_uniqueness_row_catches_a_repeated_pair(
    built_proteome: Any,
) -> None:
    """Two records of one (strain, environment) would be a duplicated measurement."""
    from torchcell.verification.runners import load_records

    records = load_records(built_proteome.root)
    assert ds.strain_condition_uniqueness(records).passed
    assert not ds.strain_condition_uniqueness([*records, records[0]]).passed


def test_the_gene_containment_row_exempts_the_heterologous_pathway_parts(
    built_titer: Any,
) -> None:
    """A pIY670 part is not a KT2440 locus, which is what its leaf exists to say."""
    from torchcell.verification.runners import load_records

    records = load_records(built_titer.root)
    universe = {tag for tag, _ in LOCUS_SPECS}
    assert ds.gene_containment_rule(records, universe).passed
    assert not ds.gene_containment_rule(records, set()).passed


def test_the_assembly_pin_row_fails_on_a_record_with_no_pin(built_titer: Any) -> None:
    """The pin survives only where the field is narrowed, so it is checked."""
    from torchcell.verification.runners import load_records

    records = load_records(built_titer.root)
    assert ds.assembly_pin_rule(records).passed
    stripped = json.loads(json.dumps(records[0]))
    stripped["reference"]["genome_reference"].pop("assembly_set")
    assert not ds.assembly_pin_rule([stripped]).passed


# --- the build-time refusals ----------------------------------------------- #
def test_the_proteome_build_refuses_a_missing_mirror_file(
    synthetic_mirror: Path, synthetic_kt2440: Any, tmp_path: Path
) -> None:
    """A mirror the manifest pins but that is gone stops the build."""
    (ds.raw_mirror_dir(str(synthetic_mirror)) / ds.PROTEOME_REL).unlink()
    with pytest.raises(RuntimeError, match="missing from mirror"):
        ds.ProteomeDeSiqueira2025Dataset(
            root=str(tmp_path / "b"), pputida_genome=synthetic_kt2440
        )


def test_the_proteome_build_refuses_a_sample_set_that_changed(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The released writable-sample count is an oracle, not an assumption."""
    monkeypatch.setattr(ds, "EXPECTED_PROTEOME_RECORDS", 4)
    with pytest.raises(RuntimeError, match="writable samples, not the 4"):
        ds.ProteomeDeSiqueira2025Dataset(
            root=str(tmp_path / "b"), pputida_genome=synthetic_kt2440
        )


def test_the_proteome_build_refuses_keys_that_resolve_below_the_threshold(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A keying change stops and reports rather than silently dropping records."""
    from torchcell.datasets.bacteria_common import LocusTagResolutionError

    monkeypatch.setattr(
        ds.ProteomeDeSiqueira2025Dataset, "MIN_RESOLVED_FRACTION", 0.999
    )
    with pytest.raises(LocusTagResolutionError):
        ds.ProteomeDeSiqueira2025Dataset(
            root=str(tmp_path / "b"), pputida_genome=synthetic_kt2440
        )


def test_a_loader_opens_the_genome_itself_when_nothing_injects_one(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A direct run resolves its own genome; the build entry points inject one."""
    monkeypatch.setattr(ds, "bacterial_genome", lambda *a, **k: synthetic_kt2440)
    dataset = ds.ProteomeDeSiqueira2025Dataset(root=str(tmp_path / "b"))
    assert len(dataset) == ds.EXPECTED_PROTEOME_RECORDS
    assert dataset.pputida_genome is synthetic_kt2440


def test_standard_names_prefers_the_annotations_symbol_and_falls_back_to_the_tag(
    synthetic_kt2440: Any,
) -> None:
    """A locus the annotation gives no symbol keeps its tag as the common name."""
    names = ds._standard_names(
        synthetic_kt2440, [ds.PT_FULL_DELETION, LOCUS_SPECS[6][0]]
    )
    assert names[ds.PT_FULL_DELETION] == ds.PT_FULL_DELETION
    assert names[LOCUS_SPECS[6][0]] == "hemN"


def test_both_loaders_refuse_the_interface_they_do_not_implement(
    built_proteome: Any, built_titer: Any
) -> None:
    """``create_experiment`` is not this loader's path; ``preprocess_raw`` is a no-op."""
    for dataset in (built_proteome, built_titer):
        assert dataset.preprocess_raw("df") == "df"
        with pytest.raises(NotImplementedError):
            dataset.create_experiment()


def test_main_builds_both_families_and_verifies_them(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The module's own entry point builds, prints its accounting and verifies."""
    monkeypatch.setattr(ds, "bacterial_genome", lambda *a, **k: synthetic_kt2440)
    monkeypatch.setattr(ds, "load_dotenv", lambda *a, **k: None, raising=False)
    monkeypatch.setenv("DATA_ROOT", str(synthetic_mirror))
    ds.main()
    out = capsys.readouterr().out
    assert "ProteomeDeSiqueira2025Dataset: len = 5" in out
    assert "IsoprenolTiterDeSiqueira2025Dataset: len = 2" in out
    assert "PASS" in out


def test_the_phenotype_builder_refuses_a_sample_with_no_measured_protein() -> None:
    """An empty abundance map is not a measurement."""
    with pytest.raises(RuntimeError, match="no measured protein"):
        ds.ProteomeDeSiqueira2025Dataset._phenotype({}, 3)


def test_the_titer_phenotype_refuses_an_unlabelled_uncertainty() -> None:
    """The schema itself forbids a number with no statistic name."""
    with pytest.raises(ValidationError):
        ProductTiterPhenotype(
            product=ds.isoprenol_product(),
            titer=1.0,
            titer_unit=ConcentrationUnit.millimolar,
            titer_uncertainty=0.1,
        )


# --------------------------------------------------------------------------- #
# Data-gated: the real mirror and the real built stores
# --------------------------------------------------------------------------- #
def _library_root() -> Path:
    return Path(os.environ["DATA_ROOT"]) / "torchcell-library"


def _have(*paths: Path) -> bool:
    return "DATA_ROOT" in os.environ and all(p.exists() for p in paths)


@pytest.mark.data
def test_every_module_quote_is_verbatim_in_the_pinned_mirrored_bytes() -> None:
    """The audit anchor: a quote that drifted would silently unsource a value."""
    library = _library_root()
    paper = library / ds.CITATION_KEY / ds.PAPER_MD
    si3 = library / ds.CITATION_KEY / ds.SI3_MD
    thompson = library / ds.THOMPSON_KEY / "paper.md"
    if not _have(paper, si3, thompson):
        pytest.skip("the torchcell-library mirror is not available")
    texts = {
        ds.PAPER_MD_SHA256: paper.read_text(),
        ds.SI3_MD_SHA256: si3.read_text(),
        ds.THOMPSON_SHA256: thompson.read_text(),
    }
    for path, expected in (
        (paper, ds.PAPER_MD_SHA256),
        (si3, ds.SI3_MD_SHA256),
        (thompson, ds.THOMPSON_SHA256),
    ):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    quotes = [getattr(ds, name) for name in dir(ds) if name.startswith("_Q_")]
    assert len(quotes) >= 25
    missing = [q for q in quotes if not any(q in text for text in texts.values())]
    assert missing == []


@pytest.mark.data
def test_the_raw_mirror_records_the_digests_the_module_pins() -> None:
    """The manifest is the audit anchor for the two consumed workbooks."""
    if "DATA_ROOT" not in os.environ:
        pytest.skip("DATA_ROOT is not set")
    mirror = ds.raw_mirror_dir()
    if not (mirror / "manifest.json").exists():
        pytest.skip("the raw mirror has not been deposited on this machine")
    manifest = ds.load_manifest()
    assert ds.manifest_sha256(manifest, ds.PROTEOME_REL) == ds.PROTEOME_SHA256
    assert ds.manifest_sha256(manifest, ds.VARIANTS_REL) == ds.VARIANTS_SHA256
    for record in manifest.files:
        assert hashlib.sha256((mirror / record.path).read_bytes()).hexdigest() == (
            record.sha256
        )


@pytest.mark.data
def test_the_released_variant_table_has_the_shape_the_module_states() -> None:
    """173 calls, 83 sites, and the locus arithmetic the PR's proposal rests on."""
    if "DATA_ROOT" not in os.environ:
        pytest.skip("DATA_ROOT is not set")
    path = ds.raw_mirror_dir() / ds.VARIANTS_REL
    if not path.exists():
        pytest.skip("the raw mirror has not been deposited on this machine")
    ledger = ds.variant_ledger(ds.read_variant_calls(str(path)))
    assert ledger.n_calls == 173
    assert ledger.n_distinct_sites == 83
    assert ledger.n_sites_in_more_than_one_strain == 33
    assert ledger.calls_with_genbank_locus == 58
    assert ledger.calls_with_refseq_locus_only == 10
    assert ledger.calls_with_no_locus == 105
    assert ledger.unsequenced_strains == [ds.SIGMA_STRAINS[2]]
    assert set(ledger.loci_claimed_twice) == {
        ds.SIGMA_STRAINS[0],
        ds.SIGMA_STRAINS[1],
        ds.SIGMA_STRAINS[3],
    }


@pytest.mark.data
def test_the_built_stores_pass_l0_to_l4() -> None:
    """The acceptance gate, run against the dev-tree builds."""
    if "DATA_ROOT" not in os.environ:
        pytest.skip("DATA_ROOT is not set")
    data_root = os.environ["DATA_ROOT"]
    for rel, family, expected in (
        ("data/torchcell/proteome_desiqueira2025", "proteome", 5),
        ("data/torchcell/isoprenol_titer_desiqueira2025", "titer", 2),
    ):
        root = osp.join(data_root, rel)
        if not osp.exists(osp.join(root, "processed", "lmdb")):
            pytest.skip(f"{rel} has not been built on this machine")
        report = ds.verify_build(root, data_root, family=family)
        assert report.passed, report.summary()
        from torchcell.verification.runners import load_records

        assert len(load_records(root)) == expected
