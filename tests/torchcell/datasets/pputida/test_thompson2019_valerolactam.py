# tests/torchcell/datasets/pputida/test_thompson2019_valerolactam.py
# [[tests.torchcell.datasets.pputida.test_thompson2019_valerolactam]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/pputida/test_thompson2019_valerolactam.py
"""The Thompson 2019 valerolactam loaders: the titers and the Table S1 growth rates.

Synthetic tests run everywhere: they exercise the Table S1 text-layer reader and its
refusals, the two phenotype builders with the typed gaps this paper forces, the two
environments, the genotype of each arm, the symbol resolution that reaches ``davT``
through the pinned GOA proteome file, the raw-mirror deposit and quote audit, and a full
end-to-end build of BOTH families over a synthetic KT2440 assembly and a synthetic
mirror -- no network and no ``$DATA_ROOT``.

The synthetic Table S1 reproduces the released numbers exactly, because the L4 that
table feeds is a join back onto its own released bytes; a fixture with invented numbers
would exercise the plumbing while retiring the check.

The ``@pytest.mark.data`` tests read the real ``$DATA_ROOT``: they pin the raw mirror's
recorded digests against the module constants, re-read every quote out of the two pinned
artifacts, resolve every gene symbol against the deposited KT2440 annotation, and run
L0 to L4 over both built stores. They are skipped unless the mirror, the KT2440 tier
cache and the stores are present.

Derived expectations for the pinned bytes: 4 titer records (the 24 h column; the four
48 h cells are refused for want of a reference) and 9 growth records (Table S1 in full,
three strains x three carbon sources, three of them released zeros). Six gene symbols
reach a locus: five through the annotation and ``davT`` through the GOA file, at
``PP_0214``.
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import torchcell.datasets.pputida.thompson2019_valerolactam as vl
from torchcell.datamodels.media import MEDIA_LIBRARY
from torchcell.datamodels.schema import (
    BacterialEnvironmentResponseExperiment,
    ConcentrationUnit,
    MeasurementType,
    PhysicalFactor,
    ProductTiterExperiment,
    SampleUnit,
)
from torchcell.literature.manifest import (
    ROLE_PAPER_TEXT,
    ROLE_SI_TEXT,
    Manifest,
    RetrievalMethod,
)

DATA_ROOT = os.environ.get("DATA_ROOT", "")
_MIRROR = Path(DATA_ROOT, "torchcell-raw", vl.CITATION_KEY) if DATA_ROOT else None
_TITER_STORE = (
    Path(DATA_ROOT, "data/torchcell/valerolactam_titer_thompson2019")
    if DATA_ROOT
    else None
)
_GROWTH_STORE = (
    Path(DATA_ROOT, "data/torchcell/lactam_growth_rate_thompson2019")
    if DATA_ROOT
    else None
)
_TIER = Path(DATA_ROOT, "data/pputida/kt2440/genome") if DATA_ROOT else None

requires_mirror = pytest.mark.skipif(
    _MIRROR is None or not (_MIRROR / vl.PAPER_TEXT.relpath).exists(),
    reason="the Thompson 2019 raw mirror is not deposited under $DATA_ROOT",
)
requires_genome = pytest.mark.skipif(
    _TIER is None or not _TIER.exists(),
    reason="the KT2440 genome tier cache is not present under $DATA_ROOT",
)
requires_stores = pytest.mark.skipif(
    _TITER_STORE is None
    or not (_TITER_STORE / "processed" / "lmdb").exists()
    or _GROWTH_STORE is None
    or not (_GROWTH_STORE / "processed" / "lmdb").exists(),
    reason="the Thompson 2019 dev-tree LMDBs are not built",
)

# --------------------------------------------------------------------------- #
# Synthetic artifacts, written the way the loader reads them
# --------------------------------------------------------------------------- #
#: The nine Table S1 cells as released, in the order the PDF's text layer renders them;
#: the strain is printed only where it changes, which is how the merged cell comes out.
TABLE_S1_TEXT = """Table S1: Specific growth rates of P. putida and valerolactam catabolic mutants on various
carbon sources.


 Strain        Carbon Source       Maximal Growth Rate (1/hr)

 WT            5AVA                0.304339

               Glucose             0.561722

               Valerolactam        0.492629

 ΔdavT         5AVA                0

               Glucose             0.568278

               Valerolactam        0

 ΔoplBA        5AVA                0.283833

               Glucose             0.567119

               Valerolactam        0
"""


def _paper_text() -> str:
    """A synthetic PMC full text carrying every quote the loader binds a value to."""
    return "\n\n".join(
        ("A synthetic stand-in for the PMC full text.", *vl.PAPER_QUOTES)
    )


def _si_text() -> str:
    """A synthetic SI text layer carrying Table S1 and every SI quote."""
    return "\n\n".join((TABLE_S1_TEXT, *vl.SI_QUOTES))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_sources(root: Path, *, si_text: str | None = None) -> dict[str, str]:
    """Write a local copy of every mirror file and return the deposit mapping."""
    bodies = {
        vl.PAPER_TEXT.relpath: _paper_text(),
        vl.PAPER_XML.relpath: "<article>synthetic</article>",
        vl.PAPER_PDF.relpath: "%PDF-1.4 synthetic paper",
        vl.SI_PDF.relpath: "%PDF-1.4 synthetic SI",
        vl.SI_TEXT.relpath: _si_text() if si_text is None else si_text,
    }
    sources: dict[str, str] = {}
    for relpath, body in bodies.items():
        path = root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        sources[relpath] = str(path)
    return sources


def repin(monkeypatch: pytest.MonkeyPatch, sources: dict[str, str]) -> None:
    """Point every ``RawFile`` pin at the synthetic bytes."""
    repinned = tuple(
        raw.model_copy(
            update={
                "sha256": _sha256(Path(sources[raw.relpath])),
                "bytes": Path(sources[raw.relpath]).stat().st_size,
            }
        )
        for raw in vl.RAW_FILES
    )
    by_relpath = {raw.relpath: raw for raw in repinned}
    monkeypatch.setattr(vl, "RAW_FILES", repinned)
    monkeypatch.setattr(
        vl, "RETRIEVED_FILES", tuple(r for r in repinned if r.bucket_name is not None)
    )
    for name in ("PAPER_TEXT", "PAPER_XML", "PAPER_PDF", "SI_PDF", "SI_TEXT"):
        monkeypatch.setattr(vl, name, by_relpath[getattr(vl, name).relpath])


# --------------------------------------------------------------------------- #
# Table S1
# --------------------------------------------------------------------------- #
def test_the_reader_carries_the_strain_down_its_three_carbon_sources(
    tmp_path: Path,
) -> None:
    """A merged strain cell is printed once; the three rows beneath it are that strain."""
    path = tmp_path / "mmc1.txt"
    path.write_text(TABLE_S1_TEXT, encoding="utf-8")
    rows = vl.read_table_s1(path)
    assert len(rows) == vl.EXPECTED_GROWTH_ROWS
    assert [(row.strain, row.carbon_source, row.rate_per_hour) for row in rows] == [
        ("KT2440", "5AVA", 0.304339),
        ("KT2440", "Glucose", 0.561722),
        ("KT2440", "Valerolactam", 0.492629),
        ("ΔdavT", "5AVA", 0.0),
        ("ΔdavT", "Glucose", 0.568278),
        ("ΔdavT", "Valerolactam", 0.0),
        ("ΔoplBA", "5AVA", 0.283833),
        ("ΔoplBA", "Glucose", 0.567119),
        ("ΔoplBA", "Valerolactam", 0.0),
    ]


def test_a_missing_row_stops_the_reader(tmp_path: Path) -> None:
    """Eight rows is not Table S1, and a short parse is a changed release."""
    path = tmp_path / "mmc1.txt"
    path.write_text(
        TABLE_S1_TEXT.replace("               Valerolactam        0.492629\n", ""),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="parsed 8 rows, expected 9"):
        vl.read_table_s1(path)


def test_an_unknown_strain_label_stops_the_reader(tmp_path: Path) -> None:
    """A strain the SI does not name cannot be mapped onto a Table 1 strain."""
    path = tmp_path / "mmc1.txt"
    path.write_text(TABLE_S1_TEXT.replace(" WT  ", " ΔoplA  "), encoding="utf-8")
    with pytest.raises(RuntimeError, match="names strain"):
        vl.read_table_s1(path)


def test_a_rate_before_any_strain_label_stops_the_reader(tmp_path: Path) -> None:
    """The first row must name its strain; nothing is attributed by position."""
    path = tmp_path / "mmc1.txt"
    path.write_text(
        TABLE_S1_TEXT.replace(" WT            5AVA", "               5AVA"),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="precedes any strain label"):
        vl.read_table_s1(path)


# --------------------------------------------------------------------------- #
# The released titers, and the four that are refused
# --------------------------------------------------------------------------- #
def test_the_stated_titers_are_eight_cells_of_which_four_are_stored() -> None:
    """Four strains x two sampling times; the 24 h column is the stored family."""
    assert len(vl.TITER_CELLS) == 8
    assert vl.EXPECTED_TITER_RECORDS == 4
    stored = [c for c in vl.TITER_CELLS if c.hours in vl.TITER_HOURS_STORED]
    assert [(c.strain, c.titer_mg_per_l) for c in stored] == [
        ("KT2440", 0.43),
        ("ΔoplBA", 4.47),
        ("ΔoplBAΔdavT", 19.29),
        ("ΔoplBAΔdavTΔalr", 63.66),
    ]
    refused = [c for c in vl.TITER_CELLS if c.hours not in vl.TITER_HOURS_STORED]
    assert [(c.strain, c.titer_mg_per_l) for c in refused] == [
        ("KT2440", None),
        ("ΔoplBA", 9.27),
        ("ΔoplBAΔdavT", 85.19),
        ("ΔoplBAΔdavTΔalr", 91.97),
    ]
    assert vl.WT_48H_NOT_DETECTED.value == "not detected"


def test_every_stated_titer_is_in_the_quote_its_cell_cites() -> None:
    """A number and the sentence that states it travel together, or neither is sourced."""
    for cell in vl.TITER_CELLS:
        if cell.titer_mg_per_l is None:
            assert "no valerolactam could be detected" in cell.quote
            continue
        assert f"{cell.titer_mg_per_l} mg/L" in cell.quote


def test_the_titer_phenotype_gaps_every_statistic_the_paper_withholds() -> None:
    """mg/L is stored verbatim as ug/mL, and no uncertainty is invented."""
    phenotype = vl.titer_phenotype(4.47)
    assert phenotype.titer == 4.47
    assert phenotype.titer_unit is ConcentrationUnit.ug_per_ml
    assert phenotype.n_samples == 3
    assert phenotype.sample_unit is SampleUnit.biological_replicate
    assert phenotype.quantification_method == "HPLC-QTOF MS"
    assert phenotype.titer_uncertainty is None
    assert phenotype.titer_uncertainty_type is None
    assert phenotype.titer_se is None
    assert {gap.field for gap in phenotype.provenance_gaps} == {
        "titer_uncertainty",
        "titer_uncertainty_type",
        "product_yield",
        "product_yield_unit",
        "productivity",
        "productivity_unit",
    }
    assert phenotype.product.name == "valerolactam"
    assert [gap.field for gap in phenotype.product.provenance_gaps] == ["inchikey"]


def test_the_growth_phenotype_is_an_absolute_rate_with_no_replicate_count() -> None:
    """Table S1 releases one rate per cell and no dispersion, so n is a typed gap."""
    phenotype = vl.growth_phenotype(0.492629)
    assert phenotype.measurement_type is MeasurementType.growth_rate
    assert phenotype.environment_response == 0.492629
    assert phenotype.units == vl.GROWTH_UNITS
    assert phenotype.n_samples is None
    assert phenotype.environment_response_se is None
    assert {gap.field for gap in phenotype.provenance_gaps} == {
        "environment_response_uncertainty",
        "environment_response_uncertainty_type",
        "environment_response_se",
        "n_samples",
        "sample_unit",
    }


def test_a_released_zero_rate_is_stored_as_a_zero() -> None:
    """Both mutants fail to grow on valerolactam; that is a measurement, not a gap."""
    assert vl.growth_phenotype(0.0).environment_response == 0.0


# --------------------------------------------------------------------------- #
# The environments and this paper's own MOPS recipe
# --------------------------------------------------------------------------- #
def test_the_production_environment_is_lb_with_the_three_stated_additions() -> None:
    """Lysine, arabinose and kanamycin at their stated doses, in a 10 mL culture."""
    env = vl.production_environment(24.0)
    assert env.media.name.startswith("LB, Miller")
    assert env.duration_hours == 24.0
    assert env.temperature is not None and env.temperature.value == 30.0
    dosed = [p.model_dump() for p in env.perturbations]
    assert [
        (p["compound"]["name"], p["concentration"]["value"], p["concentration"]["unit"])
        for p in dosed
    ] == [
        ("L-lysine", 25.0, "mM"),
        ("L-arabinose", 0.2, "percent_w/v"),
        ("kanamycin", 50.0, "ug/mL"),
    ]
    assert env.culture_format is not None
    assert env.culture_format.working_volume_ul == 10_000.0
    assert env.culture_format.shaking_rpm == 250.0
    assert env.culture_format.vessel is None
    assert [gap.field for gap in env.culture_format.provenance_gaps] == ["vessel"]


def test_the_growth_environment_carries_its_carbon_source_as_a_typed_edit() -> None:
    """The medium is carbon-source free, so the 10 mM carbon source is the edit."""
    env = vl.growth_environment("Valerolactam")
    assert env.media is vl.MOPS_MODIFIED_THOMPSON2019
    assert env.duration_hours == 48.0
    (edit,) = (p.model_dump() for p in env.perturbations)
    assert edit["factor"] == PhysicalFactor.carbon_source.value
    assert edit["magnitude"]["value"] == 10.0
    assert edit["agent"]["name"] == "valerolactam"
    (other,) = (p.model_dump() for p in vl.growth_environment("5AVA").perturbations)
    assert other["agent"]["name"] == "5-aminovaleric acid"


def test_this_papers_mops_is_its_own_object_and_differs_from_the_library_recipe() -> (
    None
):
    """Serving MOPS_MINIMAL would misstate the amounts this paper writes out."""
    assert vl.MOPS_MODIFIED_THOMPSON2019.base_medium == "MOPS_MINIMAL"
    assert vl.MOPS_MODIFIED_THOMPSON2019 not in MEDIA_LIBRARY.values()
    library = MEDIA_LIBRARY["MOPS_MINIMAL"]

    def doses(media: Any) -> dict[str, tuple[float, str]]:
        dumped = media.model_dump()
        return {
            component["compound"]["name"]: (
                component["concentration"]["value"],
                component["concentration"]["unit"],
            )
            for component in dumped["components"]
            if component["concentration"] is not None
        }

    mine, theirs = doses(vl.MOPS_MODIFIED_THOMPSON2019), doses(library)
    calcium = "calcium chloride"
    assert mine[calcium] == (32.5, "uM")
    assert theirs[calcium] == (0.0005, "mM")
    assert "iron(II) chloride" in mine and "iron(II) chloride" not in theirs
    assert not any(
        component.role.value == "carbon_source"
        for component in vl.MOPS_MODIFIED_THOMPSON2019.components
    )


# --------------------------------------------------------------------------- #
# Genotypes
# --------------------------------------------------------------------------- #
LOCUS_TAGS = {
    "oplB": "PP_3514",
    "oplA": "PP_3515",
    "davT": "PP_0214",
    "alr": "PP_3722",
    "davB": "PP_0383",
    "davA": "PP_0382",
}


def test_a_titer_genotype_is_its_deletions_plus_the_production_plasmid() -> None:
    """Every titer record carries the three pBADT-davBA-ORF26 genes."""
    genotype = vl.strain_genotype("ΔoplBAΔdavTΔalr", LOCUS_TAGS)
    kinds = [p.perturbation_type for p in genotype.perturbations]
    assert kinds.count("bacterial_deletion") == 4
    assert kinds.count("heterologous_pathway") == 3
    plasmid = {
        p.perturbed_gene_name: p
        for p in genotype.perturbations
        if p.perturbation_type == "heterologous_pathway"
    }
    assert sorted(plasmid) == ["ORF26", "davA", "davB"]
    assert plasmid["davB"].systematic_gene_name == "PP_0383"
    assert plasmid["davB"].is_heterologous is False
    assert plasmid["davB"].source_organism == vl.NATIVE_ORGANISM
    assert plasmid["ORF26"].is_heterologous is True
    assert plasmid["ORF26"].source_organism == vl.ORF26_ORGANISM
    assert all(p.construct_name == vl.PLASMID_NAME for p in plasmid.values())
    assert all(p.promoter_name == vl.PROMOTER_NAME for p in plasmid.values())


def test_a_growth_genotype_carries_no_plasmid_and_the_wild_type_carries_nothing() -> (
    None
):
    """Figs. 2A to 2C grow plasmid-free strains, and the wild type has no edit."""
    assert vl.deletion_genotype("KT2440", LOCUS_TAGS).perturbations == []
    genotype = vl.deletion_genotype("ΔoplBA", LOCUS_TAGS)
    assert [p.systematic_gene_name for p in genotype.perturbations] == [
        "PP_3514",
        "PP_3515",
    ]
    assert {p.perturbation_type for p in genotype.perturbations} == {
        "bacterial_deletion"
    }


def test_the_strain_table_is_the_four_titer_strains_and_the_davt_single() -> None:
    """Table 1's P. putida rows, with the JBEI part id of each strain made here."""
    assert {name: spec.jbei_part_id for name, spec in vl.STRAINS.items()} == {
        "KT2440": None,
        "ΔdavT": None,
        "ΔoplBA": "JPUB_013576",
        "ΔoplBAΔdavT": "JPUB_013577",
        "ΔoplBAΔdavTΔalr": "JPUB_013578",
    }
    assert vl.all_symbols() == ("alr", "davT", "oplA", "oplB", "davB", "davA")


# --------------------------------------------------------------------------- #
# A synthetic KT2440, and the symbol that only the GOA file places
# --------------------------------------------------------------------------- #
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
#: Every locus the synthetic assembly carries: the five symbols the annotation resolves,
#: plus ``PP_0214`` with NO symbol, which is what makes ``davT`` need the GOA file.
LOCUS_SPECS: tuple[tuple[str, str | None], ...] = (
    ("PP_0214", None),
    ("PP_0382", "davA"),
    ("PP_0383", "davB"),
    ("PP_3514", "oplB"),
    ("PP_3515", "oplA"),
    ("PP_3722", "alr"),
)
#: The GOA row that names ``davT``: the symbol in column 3 and ``PP_0214`` in the
#: synonym column, exactly as the deposited proteome file carries it.
DAVT_GAF_SYNONYMS = "davT|PP_0214"


def _synthetic_loci() -> list[Any]:
    from tests.torchcell.sequence.genome._bacterial_fixtures import SyntheticLocus

    loci = []
    cursor = 1
    for index, (tag, symbol) in enumerate(LOCUS_SPECS):
        start, end = cursor, cursor + 299
        cursor = end + 100
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
        [
            fixtures.gaf_row("davT", DAVT_GAF_SYNONYMS, "GO:0000001"),
            fixtures.gaf_row("oplB", "oplB|PP_3514", "GO:0000002"),
        ],
    )
    fixtures.forbid_network(monkeypatch)
    fixtures.serve_tier(monkeypatch, files)
    report = tmp_path / "tier" / ASSEMBLY_REPORT_MEMBER
    report.write_text(ASSEMBLY_REPORT)

    def serve(assembly_set: str, filename: str, **_: Any) -> str:
        if filename == ASSEMBLY_REPORT_MEMBER:
            return str(report)
        if (assembly_set, filename) in files:
            return str(files[(assembly_set, filename)])
        raise FileNotFoundError(f"{assembly_set}/{filename} is not in the fixture")

    import torchcell.datasets.bacteria_common as bacteria_common

    monkeypatch.setattr(bacteria_common, "resolve", serve)
    monkeypatch.setattr(vl, "resolve", serve)
    monkeypatch.setattr(
        vl,
        "load_genome_manifest",
        lambda assembly_set, data_root=None: SimpleNamespace(
            record=lambda member: SimpleNamespace(sha256="f" * 64)
        ),
    )
    root = tmp_path / "kt2440"
    root.mkdir()
    return PPutidaKT2440Genome(genome_root=str(root), overwrite=False)


def test_davt_is_absent_from_the_annotation_and_present_in_the_goa_file(
    synthetic_kt2440: Any,
) -> None:
    """The documented reason the GOA file is read, asserted in both directions."""
    assert synthetic_kt2440.resolve_gene_name("davT").status.value == "retired"
    resolution = vl.goa_symbol_locus(synthetic_kt2440, "davT")
    assert resolution.locus_tag == "PP_0214"
    assert resolution.accession == "UP_davT"
    assert resolution.assembly_set == "pputida_KT2440_ASM756v2"
    assert resolution.sha256 == "f" * 64


def test_a_symbol_the_goa_file_does_not_carry_is_refused(synthetic_kt2440: Any) -> None:
    """Exactly one protein, or the mapping is not stated and nothing is guessed."""
    with pytest.raises(RuntimeError, match="is the symbol of 0 proteins"):
        vl.goa_symbol_locus(synthetic_kt2440, "davD")


def test_symbol_resolution_reaches_every_symbol_the_paper_names(
    synthetic_kt2440: Any,
) -> None:
    """Five through the annotation, one through the GOA file, none left as a symbol."""
    resolution = vl.resolve_symbols(synthetic_kt2440)
    assert resolution.locus_tags == LOCUS_TAGS
    assert resolution.reconciliation.resolved_fraction == 1.0
    assert [entry.symbol for entry in resolution.goa] == ["davT"]


def test_a_symbol_the_annotation_drops_stops_the_resolution(
    synthetic_kt2440: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A symbol that reaches no locus cannot be stored: the paper names no tag."""
    monkeypatch.setattr(vl, "all_symbols", lambda: ("oplB", "notAGene"))
    with pytest.raises(RuntimeError, match="resolves no locus for"):
        vl.resolve_symbols(synthetic_kt2440)


# --------------------------------------------------------------------------- #
# The raw mirror and the quote audit
# --------------------------------------------------------------------------- #
def test_the_deposit_writes_every_file_with_its_retrieval_or_processing_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four retrieved PMC objects and one derived text layer, all pinned."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "root"))
    sources = write_sources(tmp_path / "source")
    repin(monkeypatch, sources)
    root = vl.deposit_raw_mirror(dict(sources))
    manifest = Manifest.model_validate_json((root / "manifest.json").read_text())
    assert [record.path for record in manifest.files] == [
        raw.relpath for raw in vl.RAW_FILES
    ]
    by_path = {record.path: record for record in manifest.files}
    paper = by_path[vl.PAPER_TEXT.relpath]
    assert paper.role == ROLE_PAPER_TEXT
    assert paper.retrieval is not None
    assert paper.retrieval.method is RetrievalMethod.pmc_cloud
    assert paper.retrieval.params == {"key": "PMC6838509.1/PMC6838509.1.txt"}
    derived = by_path[vl.SI_TEXT.relpath]
    assert derived.role == ROLE_SI_TEXT
    assert derived.retrieval is None
    assert derived.processing is not None
    assert derived.processing.tool == "pypdf"
    assert derived.processing.input_sha256 == [vl.SI_PDF.sha256]
    assert vl.FITNESS_BROWSER in manifest.si_data_sources
    assert (
        vl.manifest_sha256(vl.load_manifest(), vl.SI_TEXT.relpath) == vl.SI_TEXT.sha256
    )


def test_the_deposit_refuses_a_source_whose_bytes_do_not_match_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing is written when any source has drifted: the refusal is before the copy."""
    from torchcell.data import RawSha256MismatchError

    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "root"))
    sources = write_sources(tmp_path / "source")
    repin(monkeypatch, sources)
    Path(sources[vl.SI_TEXT.relpath]).write_text("drifted", encoding="utf-8")
    with pytest.raises(RawSha256MismatchError):
        vl.deposit_raw_mirror(dict(sources))
    assert not (tmp_path / "root" / vl.RAW_DIR_REL / "manifest.json").exists()


def test_the_deposit_refuses_a_mirror_file_that_already_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing mirror file is never overwritten; a disagreement is upstream drift."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "root"))
    sources = write_sources(tmp_path / "source")
    repin(monkeypatch, sources)
    held = tmp_path / "root" / vl.RAW_DIR_REL / vl.SI_TEXT.relpath
    held.parent.mkdir(parents=True, exist_ok=True)
    held.write_text("another revision", encoding="utf-8")
    with pytest.raises(RuntimeError, match="exists with a different sha256"):
        vl.deposit_raw_mirror(dict(sources))


def test_the_quote_audit_checks_every_quote_against_the_pinned_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both consumed artifacts are hashed, then every quote must be a substring."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "root"))
    sources = write_sources(tmp_path / "source")
    repin(monkeypatch, sources)
    vl.deposit_raw_mirror(dict(sources))
    assert vl.verify_quotes() == {
        vl.PAPER_TEXT.relpath: len(vl.PAPER_QUOTES),
        vl.SI_TEXT.relpath: len(vl.SI_QUOTES),
    }


def test_a_drifted_artifact_hash_stops_the_quote_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed mirror is drift, and every quote has to be re-read before it is used."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "root"))
    sources = write_sources(tmp_path / "source")
    repin(monkeypatch, sources)
    vl.deposit_raw_mirror(dict(sources))
    drifted = tuple(
        raw.model_copy(update={"sha256": "0" * 64})
        if raw.relpath == vl.PAPER_TEXT.relpath
        else raw
        for raw in vl.RAW_FILES
    )
    monkeypatch.setattr(vl, "RAW_FILES", drifted)
    with pytest.raises(RuntimeError, match="not the pinned"):
        vl.verify_quotes()


def test_a_quote_the_artifact_no_longer_carries_stops_the_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A value whose sentence is gone is no longer sourced, so the build stops."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "root"))
    sources = write_sources(tmp_path / "source")
    repin(monkeypatch, sources)
    vl.deposit_raw_mirror(dict(sources))
    monkeypatch.setattr(
        vl, "PAPER_QUOTES", (*vl.PAPER_QUOTES, "a sentence nobody wrote")
    )
    with pytest.raises(RuntimeError, match="are not verbatim in"):
        vl.verify_quotes()


def test_a_missing_source_file_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deposit is explicit about which artifact it was not given."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "root"))
    sources = write_sources(tmp_path / "source")
    repin(monkeypatch, sources)
    del sources[vl.PAPER_XML.relpath]
    with pytest.raises(RuntimeError, match="no source given for paper/"):
        vl.deposit_raw_mirror(dict(sources))


# --------------------------------------------------------------------------- #
# Build accounting
# --------------------------------------------------------------------------- #
def test_the_accounting_refuses_arithmetic_that_does_not_close() -> None:
    """Kept plus dropped is the candidate count, and the rules explain every drop."""
    vl.BuildAccounting(
        dataset="probe",
        source_rows=8,
        candidate_records=8,
        kept_records=4,
        dropped_records=4,
        rules=[vl.DropRule(rule="r", description="d", n_records=4)],
    ).check()
    with pytest.raises(RuntimeError, match="!= 8 candidates"):
        vl.BuildAccounting(
            dataset="probe",
            source_rows=8,
            candidate_records=8,
            kept_records=4,
            dropped_records=3,
            rules=[vl.DropRule(rule="r", description="d", n_records=3)],
        ).check()
    with pytest.raises(RuntimeError, match="records are missing from the build"):
        vl.BuildAccounting(
            dataset="probe",
            source_rows=8,
            candidate_records=8,
            kept_records=4,
            dropped_records=4,
            rules=[vl.DropRule(rule="r", description="d", n_records=1)],
        ).check()


# --------------------------------------------------------------------------- #
# End to end, over the synthetic assembly and mirror
# --------------------------------------------------------------------------- #
@pytest.fixture
def built_stores(
    synthetic_kt2440: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[str, str]:
    """Deposit a synthetic mirror and build both families end to end under ``tmp_path``."""
    data_root = tmp_path / "root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    sources = write_sources(tmp_path / "source")
    repin(monkeypatch, sources)
    vl.deposit_raw_mirror(dict(sources))
    titer_root = str(data_root / "data/torchcell/valerolactam_titer_thompson2019")
    growth_root = str(data_root / "data/torchcell/lactam_growth_rate_thompson2019")
    titer = vl.ValerolactamTiterThompson2019Dataset(
        root=titer_root, pputida_genome=synthetic_kt2440
    )
    titer.close_lmdb()
    growth = vl.LactamGrowthRateThompson2019Dataset(
        root=growth_root, pputida_genome=synthetic_kt2440
    )
    growth.close_lmdb()
    return titer_root, growth_root


def test_the_titer_store_holds_the_24_hour_column_and_ledgers_the_refusals(
    built_stores: tuple[str, str],
) -> None:
    """Four records, four refused cells, and the arithmetic written out."""
    from torchcell.verification.runners import load_records

    titer_root, _ = built_stores
    records = load_records(titer_root)
    assert len(records) == vl.EXPECTED_TITER_RECORDS
    for record in records:
        ProductTiterExperiment.model_validate(record["experiment"])
        assert record["experiment"]["environment"]["duration_hours"] == 24.0
    ledger = pd.read_csv(osp.join(titer_root, "preprocess", "titer_rows.csv"))
    assert sorted(ledger["titer_mg_per_l"]) == [0.43, 4.47, 19.29, 63.66]
    refused = pd.read_csv(osp.join(titer_root, "preprocess", "refused_titers.csv"))
    assert list(refused["hours"]) == [48.0] * 4
    accounting = json.loads(
        Path(titer_root, "preprocess", "build_accounting.json").read_text()
    )
    assert (accounting["kept_records"], accounting["dropped_records"]) == (4, 4)
    assert accounting["rules"][0]["rule"] == "no_released_reference_titer_at_this_time"
    assert accounting["symbol_locus_tags"]["davT"] == "PP_0214"
    assert accounting["goa_resolved"][0]["locus_tag"] == "PP_0214"


def test_every_titer_record_is_measured_against_the_wild_type_at_the_same_time(
    built_stores: tuple[str, str],
) -> None:
    """The reference is the 0.43 mg/L wild-type culture at 24 h, for all four records."""
    from torchcell.verification.runners import load_records

    titer_root, _ = built_stores
    for record in load_records(titer_root):
        reference = record["reference"]
        assert reference["phenotype_reference"]["titer"] == 0.43
        assert reference["environment_reference"]["duration_hours"] == 24.0
        assert reference["genome_reference"]["strain"] == "KT2440"
        assert reference["genome_reference"]["assembly_accession"] == "GCA_000007565.2"


def test_the_growth_store_holds_every_released_table_s1_cell(
    built_stores: tuple[str, str],
) -> None:
    """Nine records, nothing dropped, and each one's reference is its glucose rate."""
    from torchcell.verification.runners import load_records

    _, growth_root = built_stores
    records = load_records(growth_root)
    assert len(records) == vl.EXPECTED_GROWTH_RECORDS
    for record in records:
        BacterialEnvironmentResponseExperiment.model_validate(record["experiment"])
    ledger = pd.read_csv(osp.join(growth_root, "preprocess", "growth_rows.csv"))
    assert dict(ledger["carbon_source"].value_counts()) == {
        "5AVA": 3,
        "Glucose": 3,
        "Valerolactam": 3,
    }
    assert sorted(set(ledger["reference_rate_per_hour"])) == [
        0.561722,
        0.567119,
        0.568278,
    ]
    accounting = json.loads(
        Path(growth_root, "preprocess", "build_accounting.json").read_text()
    )
    assert (accounting["kept_records"], accounting["dropped_records"]) == (9, 0)
    assert accounting["rules"] == []


def test_both_built_stores_pass_l0_to_l4(built_stores: tuple[str, str]) -> None:
    """Each module verifier, over the store it just built."""
    titer_root, growth_root = built_stores
    data_root = os.environ["DATA_ROOT"]
    titer = vl.verify_titer_build(titer_root, data_root)
    assert titer.passed, titer.summary()
    assert {result.level.name for result in titer.results} == {
        "L0",
        "L1",
        "L2",
        "L3",
        "L4",
    }
    growth = vl.verify_growth_build(growth_root, data_root)
    assert growth.passed, growth.summary()
    assert "stored_growth_rate_vs_table_s1" in {r.name for r in growth.results}
    for root in (titer_root, growth_root):
        assert Path(root, "preprocess", "verification_report.json").exists()


def test_the_titer_l4_refuses_a_store_whose_value_left_its_quote(
    built_stores: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The join is onto the sentence, so a cell whose number moved is caught."""
    titer_root, _ = built_stores
    moved = tuple(
        cell.model_copy(update={"titer_mg_per_l": 1.0})
        if cell.strain == "ΔoplBA" and cell.hours == 24.0
        else cell
        for cell in vl.TITER_CELLS
    )
    monkeypatch.setattr(vl, "TITER_CELLS", moved)
    with pytest.raises(AssertionError, match="is not in the quote"):
        vl.verify_titer_build(titer_root, os.environ["DATA_ROOT"])


def test_the_growth_l4_refuses_a_record_table_s1_does_not_release(
    built_stores: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record whose (strain, carbon source) is not a released row fails the join."""
    _, growth_root = built_stores
    monkeypatch.setitem(vl.TABLE_S1_CARBON, "Glucose", "xylose")
    with pytest.raises(AssertionError, match="is not a row Table S1 releases"):
        vl.verify_growth_build(growth_root, os.environ["DATA_ROOT"])


# --------------------------------------------------------------------------- #
# The pinned bytes under $DATA_ROOT
# --------------------------------------------------------------------------- #
@pytest.mark.data
@requires_mirror
def test_the_mirror_manifest_records_the_digests_this_module_pins() -> None:
    """The deposited bytes, the manifest and the module constants must all agree."""
    manifest = vl.load_manifest(DATA_ROOT)
    assert manifest.doi == vl.DOI
    for raw in vl.RAW_FILES:
        path = vl.raw_mirror_dir(DATA_ROOT) / raw.relpath
        assert vl.manifest_sha256(manifest, raw.relpath) == raw.sha256
        assert _sha256(path) == raw.sha256
        assert path.stat().st_size == raw.bytes


@pytest.mark.data
@requires_mirror
def test_every_quote_is_verbatim_in_the_pinned_mirror() -> None:
    """The real audit, over the real bytes."""
    assert vl.verify_quotes(DATA_ROOT) == {
        vl.PAPER_TEXT.relpath: len(vl.PAPER_QUOTES),
        vl.SI_TEXT.relpath: len(vl.SI_QUOTES),
    }


@pytest.mark.data
@requires_mirror
def test_the_real_table_s1_releases_the_nine_rates_this_module_expects() -> None:
    """Read off the deposited text layer, not off the fixture."""
    rows = vl.read_table_s1(vl.raw_mirror_dir(DATA_ROOT) / vl.SI_TEXT.relpath)
    assert len(rows) == vl.EXPECTED_GROWTH_ROWS
    assert sum(1 for row in rows if row.rate_per_hour == 0.0) == 3


@pytest.mark.data
@requires_genome
def test_the_deposited_annotation_and_goa_file_place_every_symbol() -> None:
    """``davT`` is at PP_0214 in the real GOA file, and the other five in the GenBank."""
    from torchcell.datasets.bacteria_common import bacterial_genome

    genome = bacterial_genome("pputida", "KT2440", DATA_ROOT)
    resolution = vl.resolve_symbols(genome, DATA_ROOT)
    assert resolution.locus_tags == LOCUS_TAGS
    (davt,) = resolution.goa
    assert davt.product_name == "5-aminovalerate aminotransferase DavT"
    assert davt.member == "109.P_putida_KT2440.goa"


@pytest.mark.data
@requires_stores
def test_both_dev_stores_pass_l0_to_l4() -> None:
    """The gate over the real builds."""
    titer = vl.verify_titer_build(str(_TITER_STORE), DATA_ROOT)
    assert titer.passed, titer.summary()
    growth = vl.verify_growth_build(str(_GROWTH_STORE), DATA_ROOT)
    assert growth.passed, growth.summary()
