# tests/torchcell/datasets/scerevisiae/test_puddu2019.py
# [[tests.torchcell.datasets.scerevisiae.test_puddu2019]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_puddu2019.py
"""Puddu 2019 loader: its pure builders, and its readers against the mirrored workbook.

The pure half runs anywhere (the environment, the background, the perturbation builder,
the n-denominator SD). The loader consults no genome: every colony is stored under the
deletion name the release gives it. The ``data``-marked
half reads the real Supplementary Tables workbook out of the raw mirror and asserts the
counts the module docstring states, so a re-released workbook fails here rather than
silently changing the store.
"""

from __future__ import annotations

import os
import os.path as osp
from pathlib import Path

import pytest

from torchcell.datamodels.genome_architecture_features import PUDDU2019_WGS_COVERAGE
from torchcell.datamodels.schema import Zygosity
from torchcell.datasets.scerevisiae import puddu2019 as p


def _data_root() -> str:
    from dotenv import load_dotenv

    load_dotenv()
    return os.environ.get("DATA_ROOT", "")


_RAW = osp.join(_data_root(), p.RAW_DIR_REL)
_WORKBOOK = osp.join(_RAW, p.SI_TABLES_XLSX)
_LIBRARY = osp.join(_data_root(), "torchcell-library")


# --------------------------------------------------------------------------- #
# Pure builders
# --------------------------------------------------------------------------- #
def test_the_medium_is_liquid_ypad_deriving_from_the_library_object() -> None:
    media = p.puddu_media()
    assert (media.state, media.base_medium) == ("liquid", "YPAD")
    assert [c.compound.name for c in media.components] == [
        "yeast extract",
        "peptone",
        "D-glucose",
        "adenine",
    ]
    assert media.provenance[-1] is p.PAPER_SOURCED_VALUES["culture"]
    assert media.is_synthetic is False


def test_the_environment_is_30c_with_a_typed_gap_on_the_duration() -> None:
    environment = p.puddu_environment()
    assert environment.temperature is not None
    assert environment.temperature.value == 30.0
    assert environment.duration_hours is None
    assert [gap.field for gap in environment.provenance_gaps] == ["duration_hours"]
    assert environment.culture_format is not None
    assert environment.culture_format.working_volume_ul == 1800.0
    assert environment.aerobicity == "aerobic"


def test_the_background_is_a_diploid_by4743_with_a_gapped_mating_type() -> None:
    background = p.puddu_background()
    assert (background.name, background.ploidy) == ("BY4743", "diploid")
    assert background.mating_type is None
    assert [gap.field for gap in background.provenance_gaps] == ["mating_type"]
    assert {allele.allele_name for allele in background.alleles} == {
        "his3Δ1",
        "leu2Δ0",
        "lys2Δ0",
        "met15Δ0",
        "ura3Δ0",
    }
    assert background.functional_copies("YLR303W") == 1  # MET17, met15Δ0 heterozygous
    assert all(
        allele.zygosity in {Zygosity.homozygous, Zygosity.heterozygous}
        for allele in background.alleles
    )
    assert p.reference_genome().ploidy == "diploid"


def test_a_detected_tag_is_kept_verbatim_and_an_absent_one_is_a_typed_gap() -> None:
    assert p._tags("ACGT") == "ACGT"
    assert p._tags("ACGT, TTTT") == "ACGT, TTTT"
    assert p._tags(None) is None
    assert p._tags("NA") is None
    assert p._tags("nan") is None
    assert p._tags("  ") is None
    both = p.deletion_perturbation(
        systematic_gene_name="YBR273C",
        perturbed_gene_name="UBX7",
        strain_id="Del5_UBX7",
        uptag="ATTTGTGTCCCTGCGCCTG",
        dntag="GGCGATATCCAGCCTAG",
        has_barcode_row=True,
    )
    assert both.barcode == "ATTTGTGTCCCTGCGCCTG"
    assert both.systematic_gene_name == "YBR273C"
    assert both.perturbed_gene_name == "UBX7"
    assert both.constructed_orf is None
    assert both.cassette == "kanMX4"
    assert both.construction is not None
    assert both.construction.strain_accession == "Del5_UBX7"
    assert both.provenance_gaps == []
    assert both.perturbation_type == "barcoded_kanmx_deletion"
    assert both.state == "absent"

    none = p.deletion_perturbation(
        systematic_gene_name="YCR082W",
        perturbed_gene_name="AHC2",
        strain_id="Del55_AHC2",
        uptag=None,
        dntag="AATTGCCGCTGAGCTGATGT",
        has_barcode_row=True,
    )
    assert [gap.field for gap in none.provenance_gaps] == ["barcode"]
    assert none.provenance_gaps[0].note is not None
    assert "No tags detected" in none.provenance_gaps[0].note

    no_row = p.deletion_perturbation(
        systematic_gene_name="YDR140W",
        perturbed_gene_name="MTQ2",
        strain_id="Del1821_MTQ2",
        uptag=None,
        dntag=None,
        has_barcode_row=False,
    )
    assert [gap.field for gap in no_row.provenance_gaps] == [
        "barcode",
        "downtag_barcode",
    ]
    assert no_row.provenance_gaps[0].note is not None
    assert "carries no row for this colony" in no_row.provenance_gaps[0].note


def test_a_retired_orf_is_stored_under_its_released_name() -> None:
    """YAR037W is retired in R64; the release names it, so it is stored as released."""
    retired = p.deletion_perturbation(
        systematic_gene_name="YAR037W",
        perturbed_gene_name="YAR037W",
        strain_id="Del2_YAR037W",
        uptag="ACGT",
        dntag="TTTT",
        has_barcode_row=True,
    )
    assert retired.systematic_gene_name == "YAR037W"
    assert retired.perturbed_gene_name == "YAR037W"
    assert retired.constructed_orf is None


def test_the_loader_takes_no_genome_and_the_ledger_has_no_refusal() -> None:
    import inspect

    params = inspect.signature(p.GenomeArchitecturePuddu2019Dataset.__init__).parameters
    assert "genome" not in params
    assert not hasattr(p, "resolve_released_genes")
    assert not hasattr(p, "constructed_orf")
    assert not hasattr(p, "ResolvedGenes")
    assert not hasattr(p, "N_RETIRED_ORF_COLONIES")
    assert p.N_SERVED_RECORDS == p.N_KNOCKOUT_COLONIES == 8835
    assert "refused" not in p.ReleaseLedger.model_fields


def test_the_published_wild_type_sd_is_an_n_denominator_sd() -> None:
    """Table 2's SD is the population SD of its 8 colonies, not the sample SD."""
    assert p._population_sd([1.0, 1.0, 1.0]) == 0.0
    assert p._population_sd([0.0, 2.0]) == 1.0
    with pytest.raises(ValueError, match="at least one value"):
        p._population_sd([])


def test_a_strain_id_that_is_not_del_number_label_is_refused() -> None:
    cls = p.GenomeArchitecturePuddu2019Dataset
    assert cls._strain_label("Del5_UBX7") == "UBX7"
    assert cls._strain_label("Del4733_WT-1") == "WT-1"
    with pytest.raises(p.ReleaseShapeError, match="is not Del<number>_<label>"):
        cls._strain_label("UBX7")


def test_every_pinned_quote_names_the_citation_key_and_a_pinned_artifact() -> None:
    for name, sv in p.PAPER_SOURCED_VALUES.items():
        assert sv.provenance.citation_key == p.CITATION_KEY, name
        assert sv.provenance.sha256 == p.PAPER_MD_SHA256, name
        assert sv.provenance.source_uri == p.PAPER_MD, name
    for name, sv in p.WORKBOOK_SOURCED_VALUES.items():
        assert sv.provenance.citation_key == p.CITATION_KEY, name
        assert sv.provenance.sha256 == p.SI_TABLES_XLSX_SHA256, name
        assert sv.provenance.source_uri == p.SI_TABLES_XLSX, name
    assert p.PAPER_SOURCED_VALUES["rho0"].value == 1.0
    assert p.PAPER_SOURCED_VALUES["ena"].value == "ERP109205"
    assert p.PAPER_SOURCED_VALUES["replicates"].value == {
        1: 258,
        2: 4097,
        3: 30,
        4: 72,
        5: 1,
    }


# --------------------------------------------------------------------------- #
# The mirrored workbook
# --------------------------------------------------------------------------- #
@pytest.mark.data
@pytest.mark.skipif(
    not osp.isfile(_WORKBOOK), reason="Puddu 2019 raw mirror not on disk"
)
def test_the_mirror_holds_the_pinned_workbook_with_its_pmc_retrieval() -> None:
    manifest = p.load_manifest(_data_root())
    assert p.manifest_sha256(manifest, p.SI_TABLES_XLSX) == p.SI_TABLES_XLSX_SHA256
    record = manifest.files[0]
    assert (record.role, record.bytes) == ("si_data", 3098936)
    assert record.retrieval is not None
    assert record.retrieval.method == "pmc_cloud"
    assert record.retrieval.source_url == p.SI_SOURCE_URL


@pytest.mark.data
@pytest.mark.skipif(
    not osp.isfile(_WORKBOOK), reason="Puddu 2019 raw mirror not on disk"
)
def test_table_3_is_the_8843_colony_matrix_the_assay_vocabulary_names() -> None:
    colonies = p.read_colonies(_WORKBOOK)
    assert len(colonies) == p.N_COLONIES_RELEASED == 8843
    assert set(colonies.columns) == {
        "SDname",
        "ERSno",
        "Deletion",
        *PUDDU2019_WGS_COVERAGE.by_symbol,
    }
    first = colonies.iloc[0]
    assert (first["SDname"], first["ERSno"], first["Deletion"]) == (
        "SD0863b",
        "ERS838232",
        "Del1_TDA8",
    )
    assert first["rDNA"] == 117.19869745199
    assert first["ANtot"] == 0.0
    wild_type = colonies[colonies["Deletion"].str.match(p.WILD_TYPE_LABEL)]
    assert len(wild_type) == p.N_WILD_TYPE_COLONIES == 8
    assert sorted(set(wild_type["Deletion"])) == [
        "Del4733_WT-1",
        "Del4734_WT-2",
        "Del4735_WT-3",
        "Del4736_WT-4",
    ]
    assert int((colonies["mitochondria"] < 1.0).sum()) == p.PAPER_RHO0_COLONIES == 303


@pytest.mark.data
@pytest.mark.skipif(
    not osp.isfile(_WORKBOOK), reason="Puddu 2019 raw mirror not on disk"
)
def test_the_other_tables_have_the_released_shapes_and_agree_with_table_3() -> None:
    accessions = p.read_accessions(_WORKBOOK)
    assert len(accessions) == p.N_ACCESSION_ROWS == 9014
    colonies = p.read_colonies(_WORKBOOK)
    assert set(colonies["SDname"]) <= set(accessions)
    assert len(set(accessions) - set(colonies["SDname"])) == 171
    barcodes = p.read_confirmed_deletions(_WORKBOOK)
    assert len(barcodes) == p.N_BARCODE_ROWS == 8831
    assert barcodes.set_index("SDname").loc["SD0867b", "Confirmed_deletion"] == (
        "YBR273C"
    )
    genes = p.read_gene_names(_WORKBOOK)
    assert len(genes) == p.N_GENE_ROWS == 4436
    assert genes["UBX7"] == "YBR273C"
    assert genes["MTQ2"] == "YDR140W"


@pytest.mark.data
@pytest.mark.skipif(
    not osp.isfile(_WORKBOOK), reason="Puddu 2019 raw mirror not on disk"
)
def test_every_knockout_colony_has_a_released_deletion_name_the_retired_included() -> (
    None
):
    """8,835 colonies, 4,437 released names, the 9 on the 5 retired ORFs among them."""
    colonies = p.read_colonies(_WORKBOOK)
    served = colonies[~colonies["Deletion"].str.match(p.WILD_TYPE_LABEL)].reset_index(
        drop=True
    )
    barcodes = p.read_confirmed_deletions(_WORKBOOK).set_index("SDname")
    genes = p.read_gene_names(_WORKBOOK)
    dataset = p.GenomeArchitecturePuddu2019Dataset.__new__(
        p.GenomeArchitecturePuddu2019Dataset
    )
    released = dataset._released_gene_of_colony(served, barcodes, genes)
    assert len(released) == p.N_SERVED_RECORDS == 8835
    assert len(set(released.values())) == 4437
    retired = {"YAR037W", "YAR040C", "YAR043C", "YCL013W", "YCL026C"}
    on_retired = sorted(c for c, g in released.items() if g in retired)
    assert len(on_retired) == 9
    assert {released[c] for c in on_retired} == retired
    common_of_gene = {systematic: common for common, systematic in genes.items()}
    assert set(released.values()) - set(common_of_gene) == {"YKL075C"}
    assert common_of_gene["YBR273C"] == "UBX7"


@pytest.mark.data
@pytest.mark.skipif(
    not osp.isfile(_WORKBOOK), reason="Puddu 2019 raw mirror not on disk"
)
def test_the_wild_type_reference_is_table_2_and_equals_the_8_colonies() -> None:
    distribution = p.read_wild_type_distribution(_WORKBOOK)
    colonies = p.read_colonies(_WORKBOOK)
    wild_type = colonies[colonies["Deletion"].str.match(p.WILD_TYPE_LABEL)]
    for feature, column in p.WILD_TYPE_FEATURE_COLUMN.items():
        published_mean = p.wild_type_statistic(distribution, feature, "WT_mean")
        published_sd = p.wild_type_statistic(distribution, feature, "WT_StDev")
        values = [float(v) for v in wild_type[column]]
        assert published_mean == pytest.approx(sum(values) / len(values), abs=0.006)
        assert published_sd == pytest.approx(p._population_sd(values), abs=0.006)
    profile = p.wild_type_profile(distribution)
    assert profile.genome_architecture["rDNA"] == 124.99
    assert profile.genome_architecture_sd is not None
    assert profile.genome_architecture_sd["rDNA"] == 7.85
    assert profile.n_samples == 8
    assert all(profile.genome_architecture[f"chr{i:02d}"] == 2.0 for i in range(1, 17))
    gapped = {key for gap in profile.provenance_gaps for key in gap.keys}
    assert gapped == {"Ty4", "Ty5", "ANtot"}
    assert "ANtot" not in profile.genome_architecture


@pytest.mark.data
@pytest.mark.skipif(
    not osp.isdir(osp.join(_LIBRARY, p.CITATION_KEY)),
    reason="Puddu 2019 library mirror not on disk",
)
def test_every_quote_reads_back_from_the_sha256_pinned_sources() -> None:
    from torchcell.verification.sourced import audit_sourced_value

    for name, sv in p.PAPER_SOURCED_VALUES.items():
        audit = audit_sourced_value(sv, Path(_LIBRARY))
        assert audit.passed, f"{name}: {audit.message}"
    cells = p.workbook_cell_strings(_WORKBOOK)
    for name, sv in p.WORKBOOK_SOURCED_VALUES.items():
        assert sv.quote in cells, name
