# tests/torchcell/datasets/pputida/test_desiqueira2025.py
# [[tests.torchcell.datasets.pputida.test_desiqueira2025]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/pputida/test_desiqueira2025.py
"""The de Siqueira 2025 P. putida acetate-tolerization loaders.

Synthetic tests run everywhere: they exercise the two released-file readers, the variant
ledger and the three perturbation leaves one released call becomes, the Table S2 census,
the PT background and genotype builders against a recording fake genome, both
environments, the two cross-source assertions, the retention arithmetic, and both loaders
built end to end under ``tmp_path`` with no network and no ``$DATA_ROOT``.

The ``@pytest.mark.data`` tests read the real ``$DATA_ROOT``: they assert that every
module quote is a verbatim substring of the sha256-pinned mirrored bytes, pin the raw
mirror's recorded digests against the module constants, and run L0 to L4 over both built
LMDBs.

Derived expectations for the pinned released files, every one measured: Data Set S1
holds 34,600 cells over 1,730 protein groups and 20 samples, of which 17 are of a
writable strain and 1,531 protein keys survive into a record's abundance map; Data Set
S2 holds 173 variant calls over 5 sequenced clones at 83 distinct sites, 58 with a
GenBank locus tag, 10 with a RefSeq tag only and 105 with none, encoded as 53 in-locus
sequence variants, 115 site-keyed variants and 5 restatements of PT's designed deletion;
Table S2 holds 24 cells, of which 6 are ``n.d.``, 4 belong to the isolate that was never
sequenced, 4 sit in a medium the library does not hold and 10 become records.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import os.path as osp
from collections import Counter
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
    BacterialVariantType,
    ConcentrationUnit,
    EnvironmentPhysicalPerturbation,
    PhysicalFactor,
    ProductTiterExperiment,
    ProductTiterExperimentReference,
    ProductTiterPhenotype,
    SampleUnit,
    SmallMoleculePerturbation,
    VariantCallMode,
    VariantFrequencyBasis,
    VariantSiteKind,
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


@pytest.mark.parametrize("strain", list(ds.SEQUENCED_SIGMA_STRAINS))
def test_a_sequenced_isolate_is_referenced_against_pts_background(
    strain: str, served_assembly_report: None
) -> None:
    """PT's deletion is the content an isolate HOLDS CONSTANT, so it is the reference."""
    reference = ds.strain_reference(strain)
    assert reference.background is not None
    assert reference.background == ds.pt_background()
    assert reference.strain == ds.PT_STRAIN
    assert reference.assembly_set == ds.KT2440_ASSEMBLY_SET


def test_the_unsequenced_isolate_has_no_reference_and_no_genotype() -> None:
    """Data Set S2 releases no row for Sigma3, so its genotype is unknown."""
    assert ds.UNSEQUENCED_SIGMA not in ds.WRITABLE_STRAINS
    with pytest.raises(RuntimeError, match="was never sequenced"):
        ds.strain_reference(ds.UNSEQUENCED_SIGMA)
    with pytest.raises(RuntimeError, match="not a writable strain"):
        ds.strain_genotype(ds.UNSEQUENCED_SIGMA, with_pathway=False, calls=[])


def test_the_writable_genotypes_are_the_deletion_plus_each_strains_own_calls(
    tmp_path: Path,
) -> None:
    """WT carries nothing, PT its deletion plus its calls, and pIY670 adds five genes."""
    calls = ds.read_variant_calls(str(_variants_workbook(tmp_path)))
    assert (
        ds.strain_genotype(ds.WT_LABEL, with_pathway=False, calls=calls).perturbations
        == []
    )
    pt = ds.strain_genotype(ds.PT_STRAIN, with_pathway=False, calls=calls)
    assert len(pt) == 1 + VARIANT_PERTURBATIONS_PER_STRAIN["PT"]
    assert pt.perturbation_types.count("bacterial_deletion") == 1
    assert ds.PT_FULL_DELETION in pt.systematic_gene_names
    with_plasmid = ds.strain_genotype(ds.PT_STRAIN, with_pathway=True, calls=calls)
    assert len(with_plasmid) == len(pt) + len(PATHWAY_GENES)
    assert with_plasmid != pt


def test_two_sequenced_isolates_have_unequal_genotypes(tmp_path: Path) -> None:
    """``Genotype.__eq__`` compares the perturbation SET, and the call sets differ."""
    calls = ds.read_variant_calls(str(_variants_workbook(tmp_path)))
    genotypes = {
        strain: ds.strain_genotype(strain, with_pathway=False, calls=calls)
        for strain in (ds.PT_STRAIN, *ds.SEQUENCED_SIGMA_STRAINS)
    }
    assert {strain: len(genotype) for strain, genotype in genotypes.items()} == {
        strain: 1 + VARIANT_PERTURBATIONS_PER_STRAIN[strain] for strain in genotypes
    }
    for left in genotypes:
        for right in genotypes:
            if left != right:
                assert genotypes[left] != genotypes[right], (left, right)


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

#: (strain, old_locus_tag, refseq_locus_tag, protein effect, Polymorphism Type) of the
#: synthetic calls. Every sequenced clone of the real release carries at least one, so
#: the ledger's assertion that Data Set S2 releases calls for PT and the four sequenced
#: isolates holds on the synthetic workbook too; the GenBank tags are loci of
#: :data:`LOCUS_SPECS`, so a written in-locus variant is a locus of the pinned assembly.
VARIANT_SPECS: tuple[tuple[str, str | None, str | None, str, str], ...] = (
    ("PT", None, None, "None", "SNP (transition)"),
    ("PT", "PP_0168", "PP_RS00965", "None", "SNP (transversion)"),
    ("PT", ds.PT_FULL_DELETION, None, "None", "Deletion"),
    (ds.SIGMA_STRAINS[0], "PP_1656", "PP_RS08530", "Substitution", "Substitution"),
    (ds.SIGMA_STRAINS[0], "PP_1656", "PP_RS08530", "Substitution", "Substitution"),
    (ds.SIGMA_STRAINS[0], None, "PP_RS21780", "Substitution", "SNP (transition)"),
    (ds.SIGMA_STRAINS[1], "PP_0673", None, "None", "Insertion"),
    (ds.SIGMA_STRAINS[3], None, None, "None", "Deletion (tandem repeat)"),
    (ds.SIGMA_STRAINS[4], "PP_4264", "PP_RS21360", "Substitution", "Insertion"),
)
#: The encoding each :data:`VARIANT_SPECS` row is read as, in the same order.
VARIANT_ENCODINGS: tuple[str, ...] = (
    ds.ENCODING_INTERGENIC,
    ds.ENCODING_IN_LOCUS,
    ds.ENCODING_RESTATES_DESIGNED_DELETION,
    ds.ENCODING_IN_LOCUS,
    ds.ENCODING_IN_LOCUS,
    ds.ENCODING_LOCUS_NOT_IN_ASSEMBLY,
    ds.ENCODING_IN_LOCUS,
    ds.ENCODING_INTERGENIC,
    ds.ENCODING_IN_LOCUS,
)
#: ``{strain: the perturbations its written calls become}``, derived from the specs: one
#: per call less the row that restates PT's designed deletion.
VARIANT_PERTURBATIONS_PER_STRAIN: dict[str, int] = {
    "PT": 2,
    ds.SIGMA_STRAINS[0]: 3,
    ds.SIGMA_STRAINS[1]: 1,
    ds.SIGMA_STRAINS[3]: 1,
    ds.SIGMA_STRAINS[4]: 1,
}


def _variant_position(index: int) -> int:
    """The coordinate the synthetic row at ``index`` is called at."""
    return 100 + index


def _variant_row(
    index: int,
    strain: str,
    tag: str | None,
    refseq: str | None,
    effect: str,
    polymorphism: str,
) -> list[Any]:
    """One synthetic Data Set S2 row, with the released column order and cell types."""
    row: list[Any] = [None] * len(VARIANT_HEADER_ROW)
    position = _variant_position(index)
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
    row[10] = polymorphism
    row[11] = "0.5 -> 0.6" if index == 0 else "0.97"
    row[12] = f"Variants: GS_{index}"
    row[13] = strain
    row[15] = ds.VARIANT_REPLICON
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
    for index, spec in enumerate(VARIANT_SPECS):
        sheet.append(_variant_row(index, *spec))
    book.save(path)
    book.close()


def _variants_workbook(tmp_path: Path) -> Path:
    """The synthetic Data Set S2, written once per test that reads its calls."""
    path = tmp_path / "variants.xlsx"
    _write_variants(path)
    return path


def test_read_variant_calls_types_every_released_cell(tmp_path: Path) -> None:
    """Coverage and frequency stay verbatim strings; a range would break a float."""
    path = tmp_path / "variants.xlsx"
    _write_variants(path)
    calls = ds.read_variant_calls(str(path))
    assert len(calls) == len(VARIANT_SPECS)
    assert calls[0].coverage == "61 -> 63"
    assert calls[0].variant_frequency == "0.5 -> 0.6"
    assert calls[0].reference_sequence == ds.VARIANT_REPLICON
    assert calls[0].reference_allele == "A"
    assert calls[0].alternate_allele == "G"
    assert calls[1].genbank_locus_tag == "PP_0168"
    assert calls[1].refseq_locus_tag == "PP_RS00965"
    assert calls[3].gene_symbol == "relA"
    assert calls[3].cds_codon_number == 12
    assert calls[0].protein_effect is None


def test_read_variant_calls_refuses_a_row_on_another_replicon(tmp_path: Path) -> None:
    """A coordinate on another sequence is not a coordinate on this assembly."""
    path = tmp_path / "variants.xlsx"
    book = openpyxl.Workbook()
    book.active.append(list(VARIANT_HEADER_ROW))
    row = _variant_row(0, *VARIANT_SPECS[0])
    row[15] = "NC_002947 (1)"
    book.active.append(row)
    book.save(path)
    book.close()
    with pytest.raises(RuntimeError, match="names replicon 'NC_002947 \\(1\\)'"):
        ds.read_variant_calls(str(path))


def test_read_variant_calls_refuses_a_polymorphism_type_it_cannot_type(
    tmp_path: Path,
) -> None:
    """A released spelling outside the map would otherwise default to a wrong kind."""
    path = tmp_path / "variants.xlsx"
    book = openpyxl.Workbook()
    book.active.append(list(VARIANT_HEADER_ROW))
    row = _variant_row(0, *VARIANT_SPECS[0])
    row[10] = "Rearrangement"
    book.active.append(row)
    book.save(path)
    book.close()
    with pytest.raises(RuntimeError, match="unmapped Polymorphism Type"):
        ds.read_variant_calls(str(path))


def test_every_call_is_encoded_as_the_leaf_its_released_shape_names(
    tmp_path: Path,
) -> None:
    """The four encodings, each assigned from the row it is measured on."""
    path = tmp_path / "variants.xlsx"
    _write_variants(path)
    calls = ds.read_variant_calls(str(path))
    assert [call.encoding for call in calls] == list(VARIANT_ENCODINGS)
    assert calls[2].genbank_locus_tag == ds.PT_FULL_DELETION
    assert calls[5].genbank_locus_tag is None
    assert calls[5].refseq_locus_tag == "PP_RS21780"
    assert calls[7].refseq_locus_tag is None


def test_an_in_locus_call_becomes_a_sequence_variant_of_that_locus(
    tmp_path: Path,
) -> None:
    """Every released cell reaches the call, and the locus keys the perturbation."""
    path = tmp_path / "variants.xlsx"
    _write_variants(path)
    calls = ds.read_variant_calls(str(path))
    perturbations = ds.called_variant_perturbations(calls, ds.SIGMA_STRAINS[0])
    in_locus = [
        p for p in perturbations if p.perturbation_type == "bacterial_sequence_variant"
    ]
    assert len(in_locus) == 2
    first = in_locus[0]
    assert first.systematic_gene_name == "PP_1656"
    assert first.perturbed_gene_name == "relA"
    assert first.gene_namespace == ds.KT2440_NAMESPACE
    call = first.call
    assert call.variant_type is BacterialVariantType.substitution
    assert call.type_statement == "Substitution"
    assert call.reference_sequence == ds.VARIANT_REPLICON
    assert call.position_start == _variant_position(3)
    assert call.position_end == _variant_position(3)
    assert call.sequence_change == "A -> G"
    assert call.reference_allele == "A"
    assert call.alternate_allele == "G"
    assert call.annotation == "Substitution"
    assert call.codon_change == "GGT -> GAT"
    assert call.codon_number == 12
    assert call.call_mode is VariantCallMode.clone
    assert call.frequency_statement == "0.97"
    assert call.frequency == 0.97
    assert call.frequency_basis is VariantFrequencyBasis.fraction
    assert call.caller == str(dict(ds.WGS_METHOD.value)["caller"])


def test_an_intergenic_call_is_keyed_on_its_site_and_names_no_locus(
    tmp_path: Path,
) -> None:
    """A call the release places in no gene carries the derived site id, not a tag."""
    path = tmp_path / "variants.xlsx"
    _write_variants(path)
    calls = ds.read_variant_calls(str(path))
    (site,) = ds.called_variant_perturbations(calls, ds.SIGMA_STRAINS[3])
    assert site.perturbation_type == "bacterial_site_variant"
    assert site.site_kind is VariantSiteKind.intergenic
    expected = f"{ds.VARIANT_REPLICON}:{_variant_position(7)}"
    assert site.systematic_gene_name == expected
    assert site.perturbed_gene_name == expected
    assert site.released_locus_statement is None
    assert site.flanking_systematic_gene_names == ()


def test_a_refseq_only_call_keeps_its_released_tag_and_claims_no_neighbor(
    tmp_path: Path,
) -> None:
    """The refusal the ``locus_not_in_assembly`` kind exists to state."""
    path = tmp_path / "variants.xlsx"
    _write_variants(path)
    calls = ds.read_variant_calls(str(path))
    perturbations = ds.called_variant_perturbations(calls, ds.SIGMA_STRAINS[0])
    (unmapped,) = [
        p for p in perturbations if p.perturbation_type == "bacterial_site_variant"
    ]
    assert unmapped.site_kind is VariantSiteKind.locus_not_in_assembly
    assert unmapped.released_locus_statement == "PP_RS21780"
    assert unmapped.perturbed_gene_name == "PP_RS21780"
    assert unmapped.systematic_gene_name == (
        f"{ds.VARIANT_REPLICON}:{_variant_position(5)}"
    )
    assert unmapped.flanking_systematic_gene_names == ()


def test_the_designed_deletion_is_not_written_a_second_time_as_a_call(
    tmp_path: Path,
) -> None:
    """PT's PP_2675 row restates the lesion ``pt_perturbation`` already writes."""
    path = tmp_path / "variants.xlsx"
    _write_variants(path)
    calls = ds.read_variant_calls(str(path))
    perturbations = ds.called_variant_perturbations(calls, ds.PT_STRAIN)
    assert len(perturbations) == VARIANT_PERTURBATIONS_PER_STRAIN["PT"]
    assert ds.PT_FULL_DELETION not in {p.systematic_gene_name for p in perturbations}


def test_the_released_frequency_is_a_fraction_or_nothing_at_all() -> None:
    """A percent range has no single value, so it carries no number and no basis."""
    assert ds.released_frequency("0.96") == (0.96, VariantFrequencyBasis.fraction)
    assert ds.released_frequency("95.1% -> 97.6%") == (None, None)
    with pytest.raises(RuntimeError, match="outside \\(0, 1\\]"):
        ds.released_frequency("97.6")


def test_the_variant_ledger_counts_sites_strains_and_unsequenced_isolates(
    tmp_path: Path,
) -> None:
    """The ledger every build writes beside its store."""
    path = tmp_path / "variants.xlsx"
    _write_variants(path)
    ledger = ds.variant_ledger(ds.read_variant_calls(str(path)))
    assert ledger.n_calls == len(VARIANT_SPECS)
    assert ledger.n_distinct_sites == len(VARIANT_SPECS)
    assert ledger.n_sites_in_more_than_one_strain == 0
    assert ledger.calls_with_genbank_locus == 6
    assert ledger.calls_with_refseq_locus_only == 1
    assert ledger.calls_with_no_locus == 2
    assert ledger.loci_claimed_twice == {ds.SIGMA_STRAINS[0]: ["PP_1656 x2"]}
    assert ledger.unsequenced_strains == [ds.UNSEQUENCED_SIGMA]
    assert ledger.encodings == {
        ds.ENCODING_IN_LOCUS: 5,
        ds.ENCODING_INTERGENIC: 2,
        ds.ENCODING_LOCUS_NOT_IN_ASSEMBLY: 1,
        ds.ENCODING_RESTATES_DESIGNED_DELETION: 1,
    }
    assert ledger.perturbations_per_strain == VARIANT_PERTURBATIONS_PER_STRAIN


def test_the_ledger_refuses_a_count_of_encodings_that_loses_a_call(
    tmp_path: Path,
) -> None:
    """Every call is encoded, and a written call is a perturbation of its strain."""
    path = tmp_path / "variants.xlsx"
    _write_variants(path)
    ledger = ds.variant_ledger(ds.read_variant_calls(str(path)))
    short = ledger.model_copy(update={"encodings": {ds.ENCODING_IN_LOCUS: 5}})
    with pytest.raises(RuntimeError, match="5 encodings over 9 calls"):
        short.check()
    miscounted = ledger.model_copy(
        update={"perturbations_per_strain": {ds.PT_STRAIN: 2}}
    )
    with pytest.raises(RuntimeError, match="2 perturbations over 9 calls less 1"):
        miscounted.check()


def test_the_ledger_refuses_a_release_that_sequenced_another_strain_set(
    tmp_path: Path,
) -> None:
    """The four sequenced isolates are asserted against the workbook, not assumed."""
    path = tmp_path / "variants.xlsx"
    _write_variants(path)
    calls = [
        call
        for call in ds.read_variant_calls(str(path))
        if call.strain != ds.SIGMA_STRAINS[4]
    ]
    with pytest.raises(RuntimeError, match="releases calls for"):
        ds.variant_ledger(calls)


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
    """24 cells: 6 n.d., 4 of the unsequenced isolate, 4 in an absent medium, 10 loaded."""
    census = ds.titer_census()
    assert census.n_cells == len(ds.TITER_COLUMNS) * (1 + len(ds.SIGMA_STRAINS))
    assert census.n_not_determined == 6
    assert census.n_unwritable_strain == 4
    assert census.n_medium_not_in_library == 4
    assert census.n_loaded == 10
    assert census.n_loaded == len(ds.TITER_STRAINS_LOADED) * len(
        ds.TITER_COLUMNS_LOADED
    )


def test_the_census_refuses_a_partition_that_loses_a_cell() -> None:
    """The partition is checked, not assumed."""
    census = ds.TiterCensus(
        n_cells=24,
        n_not_determined=6,
        n_unwritable_strain=4,
        n_medium_not_in_library=4,
        n_loaded=9,
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


#: The one synthetic protein whose constant-SEM column is NOT single-valued, which is
#: the pinned workbook's own shape (1,728 of 1,729 proteins carry one value).
PROTEOME_SEM_MULTIVALUED_KEY = PROTEOME_KEYS[0]


def _write_proteome(path: Path) -> None:
    """A synthetic Data Set S1 over :data:`PROTEOME_SAMPLES` and :data:`PROTEOME_KEYS`.

    All 15 released columns, and the two derived ones satisfy the build oracles the way
    the pinned workbook does: the CV is exactly ``100 * pct_sd / pct_mean``, and the SEM
    is one value per protein for every key but
    :data:`PROTEOME_SEM_MULTIVALUED_KEY`. The log10 column is a MEAN OF LOGS, so it sits
    strictly below ``log10`` of the released percent, as it does in the real bytes.
    """
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
            pct_mean = 0.5 + index / 100.0
            pct_sd = 0.01 * (index + 1)
            sem = 0.004 * (index + 1)
            if key == PROTEOME_SEM_MULTIVALUED_KEY:
                sem += len(condition) / 1000.0
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
                        pct_mean,
                        pct_sd,
                        math.log10(pct_mean) - 0.02,
                        0.003 * (index + 1),
                        100.0 * pct_sd / pct_mean,
                        sem,
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
    """All 15 consumed columns, typed; extra columns are ignored."""
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
    assert rows[0].pct_mean == 0.5
    assert rows[0].pct_sd == 0.01
    assert rows[0].log10_pct_mean == pytest.approx(math.log10(0.5) - 0.02)
    assert rows[0].log10_pct_sd == pytest.approx(0.003)
    assert rows[0].cv_percent == pytest.approx(2.0)
    assert len(ds.PROTEOME_HEADER) == 15


def test_every_asserted_header_cell_is_read_by_a_row_or_an_oracle() -> None:
    """The finding this revision closes: no column is asserted and then parsed past."""
    normalized = {n.mean_column for n in ds.PROTEOME_NORMALIZATIONS} | {
        n.sd_column for n in ds.PROTEOME_NORMALIZATIONS
    }
    oracles = {"CV%_of_%_protein_abundance", "%_of protein_abundance_Top3_rep_mean_sem"}
    keys = {
        "Protein.Group",
        "Protein.Names",
        "Protein",
        "Protein.Description",
        "Strain",
        "Condition",
        "Sample",
    }
    assert normalized | oracles | keys == set(ds.PROTEOME_HEADER)
    assert len(normalized) == 6
    assert [n.measurement_type for n in ds.PROTEOME_NORMALIZATIONS] == [
        ds.PROTEOME_MEASUREMENT_TYPE,
        ds.PERCENT_MEASUREMENT_TYPE,
        ds.LOG10_PERCENT_MEASUREMENT_TYPE,
    ]
    assert [n.recoverable_from_top3 for n in ds.PROTEOME_NORMALIZATIONS] == [
        True,
        False,
        False,
    ]


def test_a_row_serves_the_pair_of_the_normalization_it_is_asked_for(
    tmp_path: Path,
) -> None:
    """``normalized`` is the only place a class's scale reaches a record."""
    path = tmp_path / "p.xlsx"
    _write_proteome(path)
    row = ds.read_proteome_rows(str(path))[0]
    top3, percent, log10 = ds.PROTEOME_NORMALIZATIONS
    assert row.normalized(top3) == (row.top3_mean, row.top3_sd)
    assert row.normalized(percent) == (row.pct_mean, row.pct_sd)
    assert row.normalized(log10) == (row.log10_pct_mean, row.log10_pct_sd)


def test_a_cv_column_the_percent_pair_cannot_reproduce_is_refused(
    tmp_path: Path,
) -> None:
    """The oracle that makes the CV a derived column rather than a phenotype."""
    path = tmp_path / "p.xlsx"
    _write_proteome(path)
    rows = ds.read_proteome_rows(str(path))
    ds.assert_percent_cv_is_derived(rows)
    rows[3] = rows[3].model_copy(update={"cv_percent": 42.0})
    with pytest.raises(RuntimeError, match="the CV is not derived"):
        ds.assert_percent_cv_is_derived(rows)


def test_a_sem_column_that_varies_with_the_sample_is_refused(tmp_path: Path) -> None:
    """The oracle that keeps the constant per-protein SEM out of a per-sample record."""
    path = tmp_path / "p.xlsx"
    _write_proteome(path)
    rows = ds.read_proteome_rows(str(path))
    ds.assert_sem_is_constant_per_protein(rows)
    assert ds.SEM_MULTIVALUED_PROTEINS == 1
    assert rows[0].protein == PROTEOME_SEM_MULTIVALUED_KEY
    index = next(
        i for i, row in enumerate(rows) if row.protein != PROTEOME_SEM_MULTIVALUED_KEY
    )
    rows[index] = rows[index].model_copy(update={"constant_sem": 999.0})
    with pytest.raises(RuntimeError, match="carry more than one SEM value"):
        ds.assert_sem_is_constant_per_protein(rows)


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
def built_percent(synthetic_mirror: Path, synthetic_kt2440: Any, tmp_path: Path) -> Any:
    """The percent-of-total normalization built over the same synthetic mirror."""
    return ds.ProteomePercentDeSiqueira2025Dataset(
        root=str(tmp_path / "build" / "percent"), pputida_genome=synthetic_kt2440
    )


@pytest.fixture
def built_log10(synthetic_mirror: Path, synthetic_kt2440: Any, tmp_path: Path) -> Any:
    """The mean-log10-percent normalization built over the same synthetic mirror."""
    return ds.ProteomeLog10PercentDeSiqueira2025Dataset(
        root=str(tmp_path / "build" / "log10"), pputida_genome=synthetic_kt2440
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
    """17 of the 20 released samples; the unsequenced isolate's three are left out."""
    assert len(built_proteome) == ds.EXPECTED_PROTEOME_RECORDS == 17
    assert built_proteome.experiment_class is BacterialProteinAbundanceExperiment
    assert (
        built_proteome.reference_class is BacterialProteinAbundanceExperimentReference
    )
    assert built_proteome.raw_file_names == [ds.PROTEOME_FILENAME, ds.VARIANTS_FILENAME]
    strains = set()
    genotypes = set()
    for index in range(len(built_proteome)):
        record = built_proteome[index]
        strains.add(str(record["reference"]["genome_reference"]["strain"]))
        genotypes.add(json.dumps(record["experiment"]["genotype"], sort_keys=True))
    assert strains == {ds.WT_STRAIN, ds.PT_STRAIN}
    assert len(genotypes) == len(ds.WRITABLE_STRAINS) == 6


def test_a_site_keyed_call_stays_out_of_the_gene_set(built_proteome: Any) -> None:
    """A site id names a place that is not a gene, so it is no gene node."""
    site_ids = {
        str(perturbation["systematic_gene_name"])
        for index in range(len(built_proteome))
        for perturbation in built_proteome[index]["experiment"]["genotype"][
            "perturbations"
        ]
        if perturbation["perturbation_type"] == "bacterial_site_variant"
    }
    assert site_ids == {
        f"{ds.VARIANT_REPLICON}:{_variant_position(index)}"
        for index, encoding in enumerate(VARIANT_ENCODINGS)
        if encoding in (ds.ENCODING_INTERGENIC, ds.ENCODING_LOCUS_NOT_IN_ASSEMBLY)
    }
    gene_set = set(built_proteome.gene_set)
    assert not site_ids & gene_set
    assert "PP_1656" in gene_set


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


def test_each_normalization_class_stores_its_own_released_column(
    built_proteome: Any, built_percent: Any, built_log10: Any
) -> None:
    """Three classes over one workbook, each on the pair its NORMALIZATION names."""
    assert built_percent.NORMALIZATION.measurement_type == ds.PERCENT_MEASUREMENT_TYPE
    assert (
        built_log10.NORMALIZATION.measurement_type == ds.LOG10_PERCENT_MEASUREMENT_TYPE
    )
    assert len(built_percent) == len(built_log10) == ds.EXPECTED_PROTEOME_RECORDS
    index = PROTEOME_RESOLVING_TAGS.index("PP_0154")
    pct_mean = 0.5 + index / 100.0
    for dataset, expected_mean, expected_sd in (
        (built_percent, pct_mean, 0.01 * (index + 1)),
        (built_log10, math.log10(pct_mean) - 0.02, 0.003 * (index + 1)),
    ):
        phenotype = dataset[0]["experiment"]["phenotype"]
        assert phenotype["measurement_type"] == dataset.NORMALIZATION.measurement_type
        assert phenotype["protein_abundance"]["PP_0154"] == pytest.approx(expected_mean)
        assert phenotype["protein_abundance_se"]["PP_0154"] == pytest.approx(
            expected_sd / math.sqrt(3)
        )
        reference = dataset[0]["reference"]["phenotype_reference"]
        assert reference["measurement_type"] == dataset.NORMALIZATION.measurement_type
    top3 = built_proteome[0]["experiment"]["phenotype"]
    assert top3["measurement_type"] == ds.PROTEOME_MEASUREMENT_TYPE
    assert top3["protein_abundance"]["PP_0154"] > 100.0


def test_each_normalization_class_has_its_own_dataset_name_and_root(
    built_proteome: Any, built_percent: Any, built_log10: Any
) -> None:
    """A record names the class that wrote it, so the three scales never merge."""
    names = {
        dataset[0]["experiment"]["dataset_name"]
        for dataset in (built_proteome, built_percent, built_log10)
    }
    assert names == {
        "ProteomeDeSiqueira2025Dataset",
        "ProteomePercentDeSiqueira2025Dataset",
        "ProteomeLog10PercentDeSiqueira2025Dataset",
    }
    assert {cls.__name__ for cls in ds.PROTEOME_CLASSES.values()} == names
    assert {cls.NORMALIZATION.root_slug for cls in ds.PROTEOME_CLASSES.values()} == {
        "proteome_desiqueira2025",
        "proteome_percent_desiqueira2025",
        "proteome_log10_percent_desiqueira2025",
    }


def test_the_normalization_level_refuses_a_record_on_another_scale(
    built_percent: Any,
) -> None:
    """The supplementary L3 that pairs a store to the column its class reads."""
    records = [built_percent[index] for index in range(len(built_percent))]
    percent = ds.PROTEOME_NORMALIZATIONS[1]
    passing = ds.stored_normalization_rule(records, percent)
    assert passing.passed
    assert passing.details["mean_column"] == percent.mean_column
    assert passing.details["recoverable_from_top3"] is False
    failing = ds.stored_normalization_rule(records, ds.PROTEOME_NORMALIZATIONS[0])
    assert not failing.passed


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
        rules["strain_was_never_sequenced"]["n_records"]
        == (accounting["dropped_records"])
    )
    assert rules["strain_was_never_sequenced"]["items"] == [ds.UNSEQUENCED_SIGMA]
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
    """Ten records: PT and the four sequenced isolates, each in two media."""
    assert len(built_titer) == len(ds.TITER_STRAINS_LOADED) * len(
        ds.TITER_COLUMNS_LOADED
    )
    assert len(built_titer) == 10
    assert built_titer.experiment_class is ProductTiterExperiment
    assert built_titer.reference_class is ProductTiterExperimentReference
    assert built_titer.raw_file_names == [ds.VARIANTS_FILENAME]
    record = built_titer[0]
    experiment = record["experiment"]
    kinds = {p["perturbation_type"] for p in experiment["genotype"]["perturbations"]}
    assert kinds == {
        "bacterial_deletion",
        "heterologous_pathway",
        "bacterial_sequence_variant",
        "bacterial_site_variant",
    }
    assert len(experiment["genotype"]["perturbations"]) == (
        1 + len(PATHWAY_GENES) + VARIANT_PERTURBATIONS_PER_STRAIN["PT"]
    )
    assert experiment["phenotype"]["titer_unit"] == ConcentrationUnit.millimolar.value
    assert record["reference"]["genome_reference"]["strain"] == ds.PT_STRAIN


def test_the_titer_accounting_names_all_three_reasons_a_cell_is_left_out(
    built_titer: Any,
) -> None:
    """The unsequenced isolate, the n.d. cells and the absent medium, each counted."""
    preprocess = Path(built_titer.preprocess_dir)
    accounting = json.loads((preprocess / "build_accounting.json").read_text())
    assert accounting["candidate_records"] == 24
    assert accounting["kept_records"] == 10
    assert accounting["dropped_records"] == 14
    rules = {rule["rule"]: rule["n_records"] for rule in accounting["rules"]}
    assert rules == {
        "strain_was_never_sequenced": 4,
        "cell_is_not_determined": 6,
        "medium_is_not_in_the_media_library": 4,
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
    """Only the titer family and the three released normalizations exist."""
    monkeypatch.setattr(
        ds, "bacterial_genome", lambda *a, **k: built_titer.pputida_genome
    )
    assert sorted(ds.PROTEOME_FAMILIES) == [
        "proteome",
        "proteome_log10_percent",
        "proteome_percent",
    ]
    with pytest.raises(RuntimeError, match="is neither 'titer' nor one of"):
        ds.verify_build(built_titer.root, family="growth")


def test_the_supplementary_uniqueness_row_catches_a_repeated_pair(
    built_proteome: Any,
) -> None:
    """Two records of one (strain, environment) would be a duplicated measurement."""
    from torchcell.verification.runners import load_records

    records = load_records(built_proteome.root)
    assert ds.strain_condition_uniqueness(records).passed
    assert not ds.strain_condition_uniqueness([*records, records[0]]).passed


def test_the_gene_containment_row_exempts_the_pathway_parts_and_the_sites(
    built_titer: Any,
) -> None:
    """A pIY670 part and a site id are not KT2440 loci, which is what their leaves say."""
    from torchcell.verification.runners import load_records

    records = load_records(built_titer.root)
    universe = {tag for tag, _ in LOCUS_SPECS}
    result = ds.gene_containment_rule(records, universe)
    assert result.passed
    assert result.details["outside"] == []
    site_ids = {
        str(p["systematic_gene_name"])
        for record in records
        for p in record["experiment"]["genotype"]["perturbations"]
        if p["perturbation_type"] == "bacterial_site_variant"
    }
    assert site_ids
    assert not site_ids & universe
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
    assert "ProteomeDeSiqueira2025Dataset: len = 17" in out
    assert "IsoprenolTiterDeSiqueira2025Dataset: len = 10" in out
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
    assert ledger.encodings == {
        ds.ENCODING_IN_LOCUS: 53,
        ds.ENCODING_INTERGENIC: 105,
        ds.ENCODING_LOCUS_NOT_IN_ASSEMBLY: 10,
        ds.ENCODING_RESTATES_DESIGNED_DELETION: 5,
    }
    assert ledger.perturbations_per_strain == {
        ds.PT_STRAIN: 32,
        ds.SIGMA_STRAINS[0]: 33,
        ds.SIGMA_STRAINS[1]: 27,
        ds.SIGMA_STRAINS[3]: 42,
        ds.SIGMA_STRAINS[4]: 34,
    }
    assert ledger.calls_per_strain == {
        ds.PT_STRAIN: 33,
        ds.SIGMA_STRAINS[0]: 34,
        ds.SIGMA_STRAINS[1]: 28,
        ds.SIGMA_STRAINS[3]: 43,
        ds.SIGMA_STRAINS[4]: 35,
    }


@pytest.mark.data
def test_the_released_refseq_only_calls_name_the_two_retired_tags() -> None:
    """The 10 ``locus_not_in_assembly`` calls, and the tags they keep verbatim."""
    if "DATA_ROOT" not in os.environ:
        pytest.skip("DATA_ROOT is not set")
    path = ds.raw_mirror_dir() / ds.VARIANTS_REL
    if not path.exists():
        pytest.skip("the raw mirror has not been deposited on this machine")
    calls = ds.read_variant_calls(str(path))
    unmapped = [
        call for call in calls if call.encoding == ds.ENCODING_LOCUS_NOT_IN_ASSEMBLY
    ]
    assert len(unmapped) == 10
    assert Counter(call.refseq_locus_tag for call in unmapped) == {
        "PP_RS21780": 9,
        "PP_RS19075": 1,
    }
    for strain in {call.strain for call in unmapped}:
        for perturbation in ds.called_variant_perturbations(calls, strain):
            if perturbation.perturbation_type != "bacterial_site_variant":
                continue
            if perturbation.site_kind is not VariantSiteKind.locus_not_in_assembly:
                continue
            assert perturbation.released_locus_statement in ("PP_RS21780", "PP_RS19075")
            assert perturbation.flanking_systematic_gene_names == ()


@pytest.mark.data
def test_the_built_stores_pass_l0_to_l4() -> None:
    """The acceptance gate, run against the dev-tree builds."""
    if "DATA_ROOT" not in os.environ:
        pytest.skip("DATA_ROOT is not set")
    data_root = os.environ["DATA_ROOT"]
    for rel, family, expected, n_genes in (
        ("data/torchcell/proteome_desiqueira2025", "proteome", 17, 30),
        ("data/torchcell/proteome_percent_desiqueira2025", "proteome_percent", 17, 30),
        (
            "data/torchcell/proteome_log10_percent_desiqueira2025",
            "proteome_log10_percent",
            17,
            30,
        ),
        ("data/torchcell/isoprenol_titer_desiqueira2025", "titer", 10, 35),
    ):
        root = osp.join(data_root, rel)
        if not osp.exists(osp.join(root, "processed", "lmdb")):
            pytest.skip(f"{rel} has not been built on this machine")
        report = ds.verify_build(root, data_root, family=family)
        assert report.passed, report.summary()
        from torchcell.verification.runners import load_records

        assert len(load_records(root)) == expected
        gene_set = json.loads(Path(root, "preprocess", "gene_set.json").read_text())
        assert len(gene_set) == n_genes
        assert not [tag for tag in gene_set if ":" in tag]


@pytest.mark.data
def test_the_two_new_normalizations_are_not_recoverable_from_the_stored_top3() -> None:
    """The whole justification for storing a second and third scale, re-measured here.

    Over every one of the 34,600 released cells of the sha256-pinned Data Set S1, not a
    sample of them. Two independent findings, each the reason one class exists:

    1. the released percent is NOT ``100 * top3_mean / sum(top3_mean)`` for ANY cell,
       and the per-cell ratio spans 0.914506 to 1.083554, so it is a replicate-wise mean
       of per-replicate percentages;
    2. the released log10 is at or below ``log10`` of the released percent for every
       cell and strictly below it for 33,914, which is Jensen's inequality for a mean of
       logarithms and rules out the log of the mean.
    """
    if "DATA_ROOT" not in os.environ:
        pytest.skip("DATA_ROOT is not set")
    path = Path(os.environ["DATA_ROOT"], ds.RAW_DIR_REL, "si", ds.PROTEOME_FILENAME)
    if not path.exists():
        pytest.skip("the de Siqueira 2025 raw mirror is not deposited under $DATA_ROOT")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == ds.PROTEOME_SHA256
    rows = ds.read_proteome_rows(str(path))
    assert len(rows) == 34_600
    assert len({row.protein for row in rows}) == 1_729
    assert len({(row.strain, row.condition) for row in rows}) == 20

    by_sample: dict[tuple[str, str], list[Any]] = {}
    for row in rows:
        by_sample.setdefault((row.strain, row.condition), []).append(row)
    assert len(by_sample) == 20
    ratios: list[float] = []
    for group in by_sample.values():
        total = sum(row.top3_mean for row in group)
        assert sum(row.pct_mean for row in group) == pytest.approx(100.0, abs=1e-3)
        ratios.extend(
            row.pct_mean / (100.0 * row.top3_mean / total)
            for row in group
            if row.top3_mean > 0
        )
    assert len(ratios) == 34_600
    assert sum(1 for r in ratios if abs(r - 1.0) <= 1e-9) == 0
    assert min(ratios) == pytest.approx(0.914506, abs=5e-7)
    assert max(ratios) == pytest.approx(1.083554, abs=5e-7)

    below = above = equal = 0
    for row in rows:
        gap = row.log10_pct_mean - math.log10(row.pct_mean)
        if abs(gap) <= 5e-5:
            equal += 1
        elif gap < 0:
            below += 1
        else:
            above += 1
    assert (below, equal, above) == (33_914, 686, 0)
    assert rows[0].protein == "Csda"
    assert rows[0].log10_pct_mean == pytest.approx(-2.23385590492606)
    assert math.log10(rows[0].pct_mean) == pytest.approx(-2.19547, abs=5e-6)

    delta_method = sum(
        1
        for row in rows
        if abs(row.pct_sd / (row.pct_mean * math.log(10)) - row.log10_pct_sd)
        > 1e-6 * max(1.0, abs(row.log10_pct_sd))
    )
    assert delta_method == 34_496

    ds.assert_percent_cv_is_derived(rows)
    ds.assert_sem_is_constant_per_protein(rows)
