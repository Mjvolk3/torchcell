# tests/torchcell/datasets/ecoli/test_brunk2016.py
# [[tests.torchcell.datasets.ecoli.test_brunk2016]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_brunk2016.py
"""The Brunk 2016 E. coli multi-omics loader, in four arms.

Synthetic tests run everywhere: they exercise the Table S1 OCR reader and its sourced
one-row repair, the plasmid-name gene parser, the two released routes that key an SRM
protein to a locus and the refusal of the third, the phenotype builders with the typed
gaps this release forces, the environment the overlay only some strains carry, and a
full end-to-end build of all four arms over synthetic workbooks, a synthetic OCR and a
synthetic MG1655 assembly -- no network and no real ``$DATA_ROOT``.

The synthetic fixtures reproduce the released SHAPES exactly (an OCR whose last cells
sit one row low, a protein with a multi-gene GPR, a metabolite column with no unit)
because the checks they feed are refusals of those shapes; where a check joins two
statements of the same thing, the fixture carries both statements.

The ``@pytest.mark.data`` tests read the real ``$DATA_ROOT``: the deposited mirror's
digests against the module pins, every quote re-read from the pinned bytes, the counts
the module claims, and L0 to L4 over the four built stores. They are skipped unless the
mirror and the stores are present.
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

import tests.torchcell.sequence.genome._bacterial_fixtures as fixtures
import torchcell.datasets.ecoli.brunk2016 as brunk
from tests.torchcell.sequence.genome._bacterial_fixtures import SyntheticLocus
from torchcell.datamodels.schema import (
    ConcentrationUnit,
    EndpointRule,
    SmallMoleculePerturbation,
)
from torchcell.sequence.genome.ecoli.k12 import MG1655_ASSEMBLY, EcoliK12MG1655Genome
from torchcell.verification.report import Level
from torchcell.verification.sourced import ProvenanceGapReason

DATA_ROOT = os.environ.get("DATA_ROOT", "")
_MIRROR = Path(DATA_ROOT, "torchcell-raw", brunk.CITATION_KEY) if DATA_ROOT else None
_STORES = (
    {family: Path(DATA_ROOT, rel) for family, (_, rel) in brunk.FAMILY_BUILDS.items()}
    if DATA_ROOT
    else {}
)

requires_mirror = pytest.mark.skipif(
    _MIRROR is None or not (_MIRROR / brunk.SI_OCR_REL).exists(),
    reason="the Brunk 2016 raw mirror is not deposited under $DATA_ROOT",
)
requires_stores = pytest.mark.skipif(
    not _STORES
    or not all((store / "processed" / "lmdb").exists() for store in _STORES.values()),
    reason="the four Brunk 2016 dev stores are not built under $DATA_ROOT",
)


# --------------------------------------------------------------------------- #
# Synthetic release: an OCR with the released shift, and two small workbooks
# --------------------------------------------------------------------------- #
#: The released plasmid block, trimmed to the four plasmids the fixture strains use.
PLASMID_ROWS: tuple[tuple[str, str], ...] = (
    ("JPUB_006200", "pBbA5c-MevTo-MK-PMK"),
    ("JPUB 004937", "pBbA5c-MevTsa-MK-PMK"),
    ("JPUB 004938", "pTrc99A-nudB-PMD"),
    ("JPUB_002464", "pTrc99A-GPPS-LS"),
    ("JPUB_004921", "pBbA5c-MevTo-MK-PMK-PMD-idi"),
    ("JBx_000323", "pBbA5c-MevTo-MK-PMK-PMD-idi-ispA"),
    ("JPUB_002466", "pTrc99A-BIS"),
    ("JPUB_002460", "pBbA5c-MevTco-trc-MK-PMK-PMD-idi-ispA"),
    ("JPUB_002470", "pBbA5c-atoB-HMGSsa-HMGRsa-trc-MK-PMK-PMD-idi-trc-GPPS-LS"),
    ("JPUB_002473", "pTrc99A-LS"),
    ("JPUB_006210", "pBbA5c-MevTco-PMK-MK"),
    ("JPUB_002471", "pBbE1a-GPPS-LS-atoB-HMGSsa-HMGRsa-trc-MK-PMK-PMD-idi"),
)
#: The strain block as the OCR renders it: B2 empty, its pair on the DH1 row.
SHIFTED_STRAIN_ROWS: tuple[tuple[str, str], ...] = (
    ("I1", "JPUB_006200 + JPUB_004938"),
    ("I2", "JPUB_006210 + JPUB_004938"),
    ("I3", "JPUB 004937 + JPUB 004938"),
    ("L1", "JPUB_004921 + JPUB_002464"),
    ("L2", "JPUB_002471"),
    ("L3", "JPUB_002470 + JPUB_002473"),
    ("B1", "JBx_000323 + JPUB_002466"),
    ("B2", ""),
    ("DH1", "JPUB_002460 + JPUB_002466"),
)


def table_s1_markdown(
    strain_rows: tuple[tuple[str, str], ...] = SHIFTED_STRAIN_ROWS,
    plasmid_rows: tuple[tuple[str, str], ...] = PLASMID_ROWS,
) -> str:
    """Table S1 as the OCR writes it, plus the SI sentences the loader quotes."""
    cells = ["<tr><td>Plasmids</td><td>Description</td><td>Reference</td></tr>"]
    cells += [
        f"<tr><td>{name}</td><td>{composition}</td><td>ref</td></tr>"
        for name, composition in plasmid_rows
    ]
    cells.append("<tr><td>Strains</td><td>Description</td><td>Reference</td></tr>")
    cells += [
        f"<tr><td>{strain}</td><td>{composition}</td><td>ref</td></tr>"
        for strain, composition in strain_rows
    ]
    table = "<table>" + "".join(cells) + "</table>"
    return "# Table S1\n\n" + "\n\n".join(brunk.SI_QUOTES) + "\n\n" + table + "\n"


def paper_text() -> str:
    """A stand-in article text carrying every sentence the loader quotes."""
    return "\n\n".join(brunk.PAPER_QUOTES) + "\n"


#: The fixture's metabolite columns: two in uM, one in g/L, two with no unit, and the
#: three fuel columns, each with the COBRA id the identifier sheet gives it.
METABOLITE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("ATP (uM)", "atp_c"),
    ("citrate (uM)", "cit_c"),
    ("Acetate g/L", "ac_e"),
    ("Glycine", "gly_c"),
    ("Alanine", "ala__L_c"),
    ("Isopentenol g/L", "ipoh_e"),
    ("Limonene g/L", "lim_c"),
    ("Bisabolene g/L", "bis_e"),
)
HOURS: tuple[float, ...] = (0.0, 4.0)


def _metabolite_value(strain: str, hour: float, column: str) -> float | None:
    """A released cell: distinct per strain and hour, blank where the fuel is absent."""
    index = brunk.STRAINS.index(strain) + 1
    if column.endswith("g/L") and column in brunk.TITER_COLUMN.values():
        product = brunk.PRODUCT_OF_STRAIN.get(strain)
        if product is None or brunk.TITER_COLUMN[product] != column:
            return None
        return round(0.1 * index + hour, 4)
    return round(index + hour / 10.0 + len(column) / 100.0, 4)


def write_metabolomics_workbook(path: Path) -> Path:
    """A workbook with the two sheets the metabolite and titer arms read."""
    names = pd.DataFrame(
        {
            "COBRA_met": [cobra for _, cobra in METABOLITE_COLUMNS],
            "JBEI_id": [name for name, _ in METABOLITE_COLUMNS],
            "metabolite_name": [name for name, _ in METABOLITE_COLUMNS],
        }
    )
    rows: list[dict[str, Any]] = []
    for strain in brunk.STRAINS:
        for hour in HOURS:
            row: dict[str, Any] = {"Hour": hour, "Strain": strain, "Sample": len(rows)}
            row["OD600"] = 1.0
            row["Intracellular volume / sample"] = 1e-6
            for column, _ in METABOLITE_COLUMNS:
                row[column] = _metabolite_value(strain, hour, column)
            rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as writer:
        names.to_excel(writer, sheet_name=brunk.SHEET_METABOLITE_IDS)
        pd.DataFrame(rows).to_excel(
            writer, sheet_name=brunk.SHEET_METABOLITES, index=False
        )
    return path


#: ``(protein, organism, gpr)``: one protein per released mapping route, plus the two
#: pathway proteins the organism check reads and one normalization standard.
PROTEIN_ROWS: tuple[tuple[str, str, str], ...] = (
    ("ENO", "Escherichia coli", "b2779"),
    ("MDH", "Escherichia coli", "b3236"),
    ("FRDA", "Escherichia coli", "(b4151 and b4152 and b4153 and b4154)"),
    ("AtoB", "Escherichia coli", "b2224"),
    ("HMGS", "Saccharomyces cerevisiae", "not in model/not found"),
    ("HMGS", "Staphylococcus aureus", "not in model/not found"),
    ("AmpR", "", "not in model/not found"),
)
#: The triplicate sheet's UniProt route, for the one protein it covers.
UNIPROT_ROWS: tuple[tuple[str, str], ...] = (
    (
        "sp|P0A6P9|ENO_ECOLI",
        "Enolase OS=Escherichia coli (strain K12) GN=eno PE=1 SV=2",
    ),
)
PROTEIN_HOURS: tuple[float, ...] = (0.0, 4.0)


def write_proteomics_workbook(
    path: Path, *, with_unexplained_hour: bool = True
) -> Path:
    """A workbook with the three sheets the proteome arm reads."""
    identifiers = pd.DataFrame(
        {
            "COBRA_react": [protein for protein, _, _ in PROTEIN_ROWS],
            "GPR": [gpr for _, _, gpr in PROTEIN_ROWS],
            "JBEI_id": [protein for protein, _, _ in PROTEIN_ROWS],
            "reaction_name": [protein for protein, _, _ in PROTEIN_ROWS],
        }
    )
    triplicate = pd.DataFrame(
        {
            "Uniprot ID": ["P0A6P9"] * len(UNIPROT_ROWS),
            "TotalArea": [1.0] * len(UNIPROT_ROWS),
            "Replicate": ["1A.1"] * len(UNIPROT_ROWS),
            "Strain": ["I1"] * len(UNIPROT_ROWS),
            "Hour": [0] * len(UNIPROT_ROWS),
            "Pathway": ["Glycolysis I"] * len(UNIPROT_ROWS),
            "ProteinDescription": [description for _, description in UNIPROT_ROWS],
            "ProteinName": [name for name, _ in UNIPROT_ROWS],
        }
    )
    hours: list[Any] = list(PROTEIN_HOURS)
    if with_unexplained_hour:
        hours.append(brunk.UNEXPLAINED_HOUR_LABEL)
    rows: list[dict[str, Any]] = []
    for strain in brunk.STRAINS:
        for hour in hours:
            if hour == brunk.UNEXPLAINED_HOUR_LABEL and strain == brunk.WILD_TYPE:
                continue
            for protein, organism, _ in PROTEIN_ROWS:
                numeric = 0.0 if isinstance(hour, str) else float(hour)
                rows.append(
                    {
                        "Sample Name": f"{strain}-{hour}",
                        "Hour": hour,
                        "Organism": organism or None,
                        "Strain": strain,
                        "Replicate": 1,
                        "Protein": protein,
                        "Pathway": (
                            "Mevalonate Pathway"
                            if protein in ("AtoB", "HMGS")
                            else "Glycolysis I"
                        ),
                        "Peptide": f"{protein}PEPTIDE",
                        # The wild type expresses no pathway protein, which is what
                        # the OCR repair's L4 reads; the host proteins are measured
                        # in every strain.
                        "ProteinArea": round(
                            (100.0 * (brunk.STRAINS.index(strain) + 1) + numeric)
                            * (
                                0.001
                                if strain == brunk.WILD_TYPE
                                and protein in ("AtoB", "HMGS")
                                else 1.0
                            ),
                            3,
                        ),
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(rows).to_excel(
            writer, sheet_name=brunk.SHEET_PROTEOMICS, index=False
        )
        identifiers.to_excel(writer, sheet_name=brunk.SHEET_PROTEIN_IDS)
        triplicate.to_excel(writer, sheet_name=brunk.SHEET_PROTEOMICS_TRIPLICATE)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Table S1: the reader and its sourced repair
# --------------------------------------------------------------------------- #
def test_the_ocr_plasmid_id_spelling_is_normalized() -> None:
    """The OCR drops or spaces the underscore of an id; the reader restores it."""
    assert brunk.normalize_plasmid_ids("JPUB 004937 + JPUB _006210") == (
        "JPUB_004937 + JPUB_006210"
    )
    assert brunk.normalize_plasmid_ids("JBx 000323") == "JBx_000323"
    assert brunk.normalize_plasmid_ids("pBbA5c-MevTo") == "pBbA5c-MevTo"


def test_the_wild_type_row_is_repaired_onto_the_one_empty_strain(
    tmp_path: Path,
) -> None:
    """The article calls DH1 the wild type, so its composition belongs to empty B2."""
    path = tmp_path / "mmc1.md"
    path.write_text(table_s1_markdown(), encoding="utf-8")
    table = brunk.read_table_s1(path)
    assert (table.repaired_from, table.repaired_to) == ("DH1", "B2")
    assert table.composition["B2"] == ("JPUB_002460", "JPUB_002466")
    assert table.composition["DH1"] == ()
    assert table.plasmids["JPUB_004937"] == "pBbA5c-MevTsa-MK-PMK"


def test_an_unshifted_table_is_read_with_no_repair(tmp_path: Path) -> None:
    """A reading that already puts the pair on B2 is taken as it stands."""
    rows = tuple(
        (strain, "JPUB_002460 + JPUB_002466" if strain == "B2" else "")
        if strain in ("B2", "DH1")
        else (strain, composition)
        for strain, composition in SHIFTED_STRAIN_ROWS
    )
    path = tmp_path / "mmc1.md"
    path.write_text(table_s1_markdown(strain_rows=rows), encoding="utf-8")
    table = brunk.read_table_s1(path)
    assert (table.repaired_from, table.repaired_to) == (None, None)
    assert table.composition["B2"] == ("JPUB_002460", "JPUB_002466")


def test_a_shift_with_no_empty_strain_row_is_refused(tmp_path: Path) -> None:
    """Two compositions and no empty row is not the one-row shift, so it raises."""
    rows = tuple(
        (strain, "JPUB_002460 + JPUB_002466") if strain == "B2" else (strain, comp)
        for strain, comp in SHIFTED_STRAIN_ROWS
    )
    path = tmp_path / "mmc1.md"
    path.write_text(table_s1_markdown(strain_rows=rows), encoding="utf-8")
    with pytest.raises(RuntimeError, match="the one-row OCR shift is not what this is"):
        brunk.read_table_s1(path)


def test_a_strain_block_that_is_not_the_nine_released_strains_is_refused(
    tmp_path: Path,
) -> None:
    """A renamed or dropped strain row stops the read."""
    path = tmp_path / "mmc1.md"
    path.write_text(
        table_s1_markdown(strain_rows=SHIFTED_STRAIN_ROWS[:-1]), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="strain block is"):
        brunk.read_table_s1(path)


def test_a_plasmid_no_strain_uses_is_refused(tmp_path: Path) -> None:
    """Every listed plasmid must be on a strain; an orphan means a misread table."""
    path = tmp_path / "mmc1.md"
    path.write_text(
        table_s1_markdown(
            plasmid_rows=PLASMID_ROWS + (("JPUB_999999", "pTrc99A-XYZ"),)
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="unused"):
        brunk.read_table_s1(path)


def test_a_markdown_with_no_table_s1_is_refused(tmp_path: Path) -> None:
    """The reader refuses a document that does not hold exactly one Table S1."""
    path = tmp_path / "mmc1.md"
    path.write_text("# no table here\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Table S1 candidates"):
        brunk.read_table_s1(path)


# --------------------------------------------------------------------------- #
# The plasmid-name gene parser
# --------------------------------------------------------------------------- #
def test_a_mevt_token_expands_to_the_three_genes_the_si_names() -> None:
    """``MevTo`` is atoB, HMGS and HMGR, with the SI's own variant word."""
    parts = brunk.parse_plasmid_genes("JPUB_006200", "pBbA5c-MevTo-MK-PMK")
    assert [(p.gene.token, p.variant) for p in parts] == [
        ("atoB", None),
        ("HMGS", "original"),
        ("HMGR", "original"),
        ("MK", None),
        ("PMK", None),
    ]
    codon = brunk.parse_plasmid_genes("JPUB_006210", "pBbA5c-MevTco-PMK-MK")
    assert {p.variant for p in codon} == {None, "codon_optimized"}
    aureus = brunk.parse_plasmid_genes("JPUB_004937", "pBbA5c-MevTsa-MK-PMK")
    assert [p.gene.source_organism for p in aureus if p.gene.token.endswith("sa")] == [
        "Staphylococcus aureus",
        "Staphylococcus aureus",
    ]


def test_the_backbone_and_the_trc_promoter_are_not_genes() -> None:
    """A vector name and the supplemental promoter contribute no perturbation."""
    parts = brunk.parse_plasmid_genes(
        "JPUB_002460", "pBbA5c-MevTco-trc-MK-PMK-PMD-idi-ispA"
    )
    assert "trc" not in {p.gene.token for p in parts}
    assert not any(token in {p.gene.token for p in parts} for token in ("pBbA5c",))


def test_an_unknown_token_in_a_plasmid_name_stops_the_build() -> None:
    """A renamed construct cannot silently lose a gene."""
    with pytest.raises(RuntimeError, match="neither a known pathway gene"):
        brunk.parse_plasmid_genes("JPUB_000001", "pBbA5c-MevTo-XYZ")


def test_a_plasmid_name_with_no_gene_is_refused() -> None:
    """A backbone alone is not a construct this loader can read."""
    with pytest.raises(RuntimeError, match="lists no gene"):
        brunk.parse_plasmid_genes("JPUB_000002", "pTrc99A")


def test_every_strains_genes_come_from_its_own_plasmids(tmp_path: Path) -> None:
    """The wild type carries none and a two-plasmid strain carries both lists."""
    path = tmp_path / "mmc1.md"
    path.write_text(table_s1_markdown(), encoding="utf-8")
    parts = brunk.strain_parts(brunk.read_table_s1(path))
    assert parts["DH1"] == []
    assert [p.gene.token for p in parts["I1"]] == [
        "atoB",
        "HMGS",
        "HMGR",
        "MK",
        "PMK",
        "nudB",
        "PMD",
    ]
    assert {p.plasmid_name for p in parts["B2"]} == {
        "pBbA5c-MevTco-trc-MK-PMK-PMD-idi-ispA",
        "pTrc99A-BIS",
    }


# --------------------------------------------------------------------------- #
# The protein key routes
# --------------------------------------------------------------------------- #
def test_the_two_released_routes_key_a_protein_and_the_third_shape_is_refused(
    tmp_path: Path,
) -> None:
    """UniProt ``GN=`` first, then a single-gene GPR; a multi-gene GPR is refused."""
    workbook = write_proteomics_workbook(tmp_path / "mmc3.xlsx")
    read = brunk.read_protein_keys(workbook)
    keys = {key.protein: key for key in read if key.protein != "HMGS"}
    assert keys["ENO"].route == brunk.ROUTE_UNIPROT
    assert keys["ENO"].gene_name == "eno"
    assert keys["MDH"].route == brunk.ROUTE_SINGLE_GENE_GPR
    assert keys["MDH"].gene_name == "b3236"
    assert keys["FRDA"].gene_name is None
    assert keys["FRDA"].reason == brunk.REFUSAL_MULTI_GENE_GPR
    assert {key.organism for key in read if key.protein == "HMGS"} == {
        "Saccharomyces cerevisiae",
        "Staphylococcus aureus",
    }
    assert keys["AmpR"].reason == brunk.REFUSAL_NO_MAPPING


def test_a_protein_area_that_disagrees_across_its_peptides_is_refused(
    tmp_path: Path,
) -> None:
    """The sheet repeats one area per peptide row; two different areas is a misread."""
    workbook = tmp_path / "mmc3.xlsx"
    write_proteomics_workbook(workbook)
    frame = pd.read_excel(workbook, sheet_name=brunk.SHEET_PROTEOMICS)
    extra = frame.iloc[[0]].copy()
    extra["Peptide"] = "SECOND"
    extra["ProteinArea"] = float(cast("float", frame.loc[0, "ProteinArea"])) + 1.0
    with pd.ExcelWriter(
        workbook, mode="a", engine="openpyxl", if_sheet_exists="replace"
    ) as writer:
        pd.concat([frame, extra]).to_excel(
            writer, sheet_name=brunk.SHEET_PROTEOMICS, index=False
        )
    with pytest.raises(RuntimeError, match="more than one ProteinArea"):
        brunk.read_protein_areas(workbook)


def test_a_pathway_gene_organism_must_be_the_one_the_workbook_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The organism is read from the released column, never from the token suffix."""
    workbook = write_proteomics_workbook(tmp_path / "mmc3.xlsx")
    assert brunk.check_pathway_organisms(workbook) == 3
    wrong = brunk.PATHWAY_GENES["HMGS"].model_copy(
        update={"source_organism": "Homo sapiens"}
    )
    monkeypatch.setitem(brunk.PATHWAY_GENES, "HMGS", wrong)
    with pytest.raises(RuntimeError, match="the workbook states"):
        brunk.check_pathway_organisms(workbook)


def test_a_gene_the_workbook_never_measures_keeps_the_unreported_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``GPPS`` has no organism anywhere, so an asserted one stops the build."""
    workbook = write_proteomics_workbook(tmp_path / "mmc3.xlsx")
    assert brunk.PATHWAY_GENES["GPPS"].source_organism == (
        brunk.SOURCE_ORGANISM_UNREPORTED
    )
    invented = brunk.PATHWAY_GENES["GPPS"].model_copy(
        update={"protein_name": "AmpR", "source_organism": "Abies grandis"}
    )
    monkeypatch.setitem(brunk.PATHWAY_GENES, "GPPS", invented)
    with pytest.raises(RuntimeError, match="states no organism"):
        brunk.check_pathway_organisms(workbook)


# --------------------------------------------------------------------------- #
# The metabolite column split
# --------------------------------------------------------------------------- #
def test_the_columns_are_split_by_the_unit_in_their_own_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two uM columns, one g/L acid, the three fuels and two columns with no unit."""
    workbook = write_metabolomics_workbook(tmp_path / "mmc2.xlsx")
    monkeypatch.setattr(brunk, "EXPECTED_UM_COLUMNS", 2)
    monkeypatch.setattr(brunk, "EXPECTED_UNITLESS_COLUMNS", 2)
    columns = brunk.read_metabolite_columns(workbook)
    assert columns.micromolar == ("ATP (uM)", "citrate (uM)")
    assert columns.grams_per_litre == ("Acetate g/L",)
    assert set(columns.product) == set(brunk.TITER_COLUMN.values())
    assert columns.unitless == ("Glycine", "Alanine")
    assert columns.cobra_id["ATP (uM)"] == "atp_c"


def test_a_column_count_that_moved_stops_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The measured column counts are pinned, so a changed release is caught."""
    workbook = write_metabolomics_workbook(tmp_path / "mmc2.xlsx")
    monkeypatch.setattr(brunk, "EXPECTED_UM_COLUMNS", 51)
    with pytest.raises(RuntimeError, match=r"\(uM\) columns"):
        brunk.read_metabolite_columns(workbook)


def test_a_repeated_strain_hour_sample_is_refused(tmp_path: Path) -> None:
    """One sample per strain and hour; a duplicate would collide on identity."""
    workbook = tmp_path / "mmc2.xlsx"
    write_metabolomics_workbook(workbook)
    frame = pd.read_excel(workbook, sheet_name=brunk.SHEET_METABOLITES)
    with pd.ExcelWriter(
        workbook, mode="a", engine="openpyxl", if_sheet_exists="replace"
    ) as writer:
        pd.concat([frame, frame.iloc[[0]]]).to_excel(
            writer, sheet_name=brunk.SHEET_METABOLITES, index=False
        )
    with pytest.raises(RuntimeError, match="repeats a strain-hour sample"):
        brunk.read_metabolite_samples(workbook)


# --------------------------------------------------------------------------- #
# Phenotypes, environment and genotype
# --------------------------------------------------------------------------- #
def test_a_metabolite_phenotype_stores_one_replicate_and_no_invented_dispersion() -> (
    None
):
    """The sheet releases one number per sample, so the SE map is a typed gap."""
    phenotype = brunk.metabolite_phenotype(
        {"atp_c": 1.5}, brunk.MEASUREMENT_TYPE_METABOLITE
    )
    assert phenotype.metabolite_level == {"atp_c": 1.5}
    assert phenotype.metabolite_level_se is None
    assert phenotype.n_replicates == {"atp_c": 1}
    assert phenotype.target_metabolite_ids is None
    gaps = {gap.field: gap.reason for gap in phenotype.provenance_gaps}
    assert gaps == {
        "metabolite_level_se": ProvenanceGapReason.not_reported_by_primary,
        "target_metabolite_ids": ProvenanceGapReason.not_reported_by_primary,
    }


def test_a_protein_phenotype_keeps_the_released_peak_area_verbatim() -> None:
    """A negative peak area is a released number and is stored as it stands."""
    phenotype = brunk.protein_phenotype({"b2779": -104.0})
    assert phenotype.protein_abundance == {"b2779": -104.0}
    assert phenotype.protein_abundance_se is None
    assert phenotype.measurement_type == brunk.MEASUREMENT_TYPE_PROTEIN
    assert [gap.field for gap in phenotype.provenance_gaps] == ["protein_abundance_se"]


def test_the_isopentenol_titer_is_the_canonical_isoprenol_entity() -> None:
    """Resolved through the compound table's own synonym, in the sheet's g/L."""
    phenotype = brunk.titer_phenotype("isopentenol", 1.2358)
    assert phenotype.titer == 1.2358
    assert phenotype.titer_unit is ConcentrationUnit.g_per_l
    assert phenotype.product.inchikey == "CPJRRXSHAYUTGL-UHFFFAOYSA-N"
    assert phenotype.quantification_method == "GC-FID"
    assert phenotype.n_samples is None
    assert {gap.field for gap in phenotype.provenance_gaps} >= {
        "titer_uncertainty",
        "titer_uncertainty_type",
        "n_samples",
        "sample_unit",
        "product_yield",
        "productivity",
    }


def test_a_terpene_titer_names_the_overlay_method_the_methods_state() -> None:
    """Limonene and bisabolene come off the dodecane overlay by GC-MS."""
    assert brunk.titer_phenotype("limonene", 0.5).quantification_method == "GC-MS"
    assert brunk.titer_phenotype("bisabolene", 0.5).product.name == "bisabolene"


def test_the_overlay_is_on_the_terpene_cultures_and_on_no_others() -> None:
    """The 10% dodecane overlay travels with the limonene and bisabolene records."""
    isopentenol = [
        cast("SmallMoleculePerturbation", p)
        for p in brunk.environment("I1", 24.0).perturbations
    ]
    limonene = [
        cast("SmallMoleculePerturbation", p)
        for p in brunk.environment("L1", 24.0).perturbations
    ]
    assert [p.compound.name for p in isopentenol] == ["IPTG"]
    assert [p.compound.name for p in limonene] == ["IPTG", "dodecane"]
    overlay = limonene[1]
    assert overlay.concentration.value == 10.0
    assert overlay.concentration.unit is ConcentrationUnit.percent_v_v


def test_the_environment_carries_the_sampling_hour_and_the_flask() -> None:
    """``duration_hours`` is what makes two samples of one strain two records."""
    culture = brunk.environment("DH1", 36.0)
    assert culture.duration_hours == 36.0
    assert culture.temperature is not None
    assert culture.temperature.value == 30.0
    assert culture.aerobicity == "aerobic"
    assert culture.culture_format is not None
    assert culture.culture_format.vessel == "1 L Erlenmeyer flask"
    assert culture.culture_format.working_volume_ul == 100_000.0
    assert culture.culture_format.shaking_rpm == 200.0
    assert culture.culture_format.endpoint is EndpointRule.fixed_duration
    assert culture.media.base_medium == "EZ_RICH"


def test_the_medium_defers_its_recipe_rather_than_inventing_one() -> None:
    """The paper names the medium and its glucose, and nothing else."""
    components = {c.compound.name: c for c in brunk.EZ_RICH_BRUNK2016.components}
    base = components["EZ-Rich defined medium base (no recipe stated)"]
    assert base.definition.value == "composition_deferred"
    assert base.concentration is None
    assert base.defers_to == ["neidhardtCultureMediumEnterobacteria1974"]
    glucose = components["D-glucose"]
    assert glucose.concentration is not None
    assert glucose.concentration.value == 1.0
    assert glucose.concentration.unit is ConcentrationUnit.percent_w_v
    assert brunk.EZ_RICH_BRUNK2016.is_synthetic is True


def test_the_host_background_asserts_no_allele_the_paper_never_writes() -> None:
    """DH1 is named and its lesions are not, so the background carries none."""
    background = brunk.host_background()
    assert background.name == "DH1"
    assert background.reference_strain == "MG1655"
    assert background.alleles == []
    assert background.genotype_statement is None
    assert background.provenance is not None


def test_a_native_pathway_gene_is_stored_under_its_b_number() -> None:
    """An extra copy of a native gene names its host locus; a foreign one its token."""
    parts = brunk.parse_plasmid_genes("JPUB_004938", "pTrc99A-nudB-PMD")
    locus_of = {"nudB": "b1865"}
    native = brunk.pathway_perturbation(parts[0], locus_of, "isopentenol")
    assert native.systematic_gene_name == "b1865"
    assert native.is_heterologous is False
    assert native.source_organism == "Escherichia coli"
    assert native.construct_name == "pTrc99A-nudB-PMD"
    assert native.pathway_name == "heterologous mevalonate pathway to isopentenol"
    foreign = brunk.pathway_perturbation(parts[1], locus_of, "isopentenol")
    assert foreign.systematic_gene_name == "PMD"
    assert foreign.is_heterologous is True


def test_the_wild_type_genotype_is_empty_and_a_plasmid_on_it_is_refused() -> None:
    """A wild type carries no perturbation, which is what the repair rests on."""
    assert brunk.genotype("DH1", [], {}).perturbations == []
    parts = brunk.parse_plasmid_genes("JPUB_004938", "pTrc99A-nudB-PMD")
    with pytest.raises(RuntimeError, match="carries no plasmid"):
        brunk.genotype("DH1", parts, {"nudB": "b1865"})


def test_the_build_accounting_refuses_a_ledger_that_does_not_add_up() -> None:
    """Kept plus dropped must be the rows the build read."""
    accounting = brunk.BuildAccounting(
        dataset="probe",
        source_rows=10,
        kept_records=4,
        dropped_records=4,
        distinct_targets=1,
        quotes_checked=1,
    )
    with pytest.raises(RuntimeError, match="is not the 10 source rows"):
        accounting.check()


# --------------------------------------------------------------------------- #
# The raw mirror and the quote audit
# --------------------------------------------------------------------------- #
def test_the_mirror_paths_hang_off_data_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """The deposit writes under ``$DATA_ROOT/torchcell-raw/<citation key>``."""
    monkeypatch.setenv("DATA_ROOT", "/tmp/probe-root")
    assert brunk.raw_mirror_dir() == Path(
        "/tmp/probe-root/torchcell-raw", brunk.CITATION_KEY
    )


def test_every_retrieved_file_records_a_rerunnable_retrieval() -> None:
    """Three Elsevier components and one PMC object, each with its own retriever."""
    retrievals = {raw.name: raw.retrieval for raw in brunk.RETRIEVED_FILES}
    assert set(retrievals) == {
        brunk.METABOLOMICS_FILE,
        brunk.PROTEOMICS_FILE,
        brunk.SI_PDF_FILE,
        brunk.PAPER_TEXT_FILE,
    }
    elsevier = retrievals[brunk.METABOLOMICS_FILE]
    assert elsevier.retriever == "torchcell.literature.retrieve.elsevier_mmc"
    assert elsevier.params == {
        "pii": brunk.ELSEVIER_PII,
        "filename": brunk.METABOLOMICS_FILE,
    }
    pmc = retrievals[brunk.PAPER_TEXT_FILE]
    assert pmc.retriever == "torchcell.literature.retrieve.pmc_cloud_object"
    assert pmc.params == {"key": f"{brunk.PMC_PREFIX}/{brunk.PAPER_TEXT_FILE}"}
    ocr = next(raw for raw in brunk.RAW_FILES if raw.derived)
    assert ocr.name == brunk.SI_OCR_FILE
    assert brunk.ocr_processing().input_sha256 == [brunk.SI_PDF_SHA256]


@pytest.fixture
def synthetic_mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Deposit a synthetic mirror whose pins the module is pointed at."""
    data_root = tmp_path / "root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    source = tmp_path / "source"
    source.mkdir()
    metabolomics = write_metabolomics_workbook(source / brunk.METABOLOMICS_FILE)
    proteomics = write_proteomics_workbook(source / brunk.PROTEOMICS_FILE)
    si_pdf = source / brunk.SI_PDF_FILE
    si_pdf.write_bytes(b"%PDF-1.7 synthetic\n")
    ocr = source / brunk.SI_OCR_FILE
    ocr.write_text(table_s1_markdown(), encoding="utf-8")
    paper = source / brunk.PAPER_TEXT_FILE
    paper.write_text(paper_text(), encoding="utf-8")
    for attribute, path in (
        ("METABOLOMICS_SHA256", metabolomics),
        ("PROTEOMICS_SHA256", proteomics),
        ("SI_PDF_SHA256", si_pdf),
        ("SI_OCR_SHA256", ocr),
        ("PAPER_TEXT_SHA256", paper),
    ):
        monkeypatch.setattr(brunk, attribute, _sha256(path))
    monkeypatch.setattr(
        brunk,
        "RAW_FILES",
        tuple(
            raw.model_copy(
                update={
                    "sha256": getattr(brunk, f"{_PIN_OF[raw.name]}"),
                    "bytes": (source / raw.name).stat().st_size,
                }
            )
            for raw in brunk.RAW_FILES
        ),
    )
    monkeypatch.setattr(
        brunk, "RETRIEVED_FILES", tuple(f for f in brunk.RAW_FILES if not f.derived)
    )
    monkeypatch.setattr(
        brunk,
        "DATA_SHA256",
        {
            brunk.METABOLOMICS_FILE: brunk.METABOLOMICS_SHA256,
            brunk.PROTEOMICS_FILE: brunk.PROTEOMICS_SHA256,
        },
    )
    brunk.deposit_raw_mirror(
        sources={raw.name: source / raw.name for raw in brunk.RAW_FILES}
    )
    return data_root


_PIN_OF = {
    "mmc2.xlsx": "METABOLOMICS_SHA256",
    "mmc3.xlsx": "PROTEOMICS_SHA256",
    "mmc1.pdf": "SI_PDF_SHA256",
    "mmc1.md": "SI_OCR_SHA256",
    "PMC4882250.1.txt": "PAPER_TEXT_SHA256",
}


def test_the_deposit_writes_every_artifact_with_its_record(
    synthetic_mirror: Path,
) -> None:
    """Five files land under their roles, and the OCR carries its processing record."""
    manifest = brunk.load_manifest()
    assert manifest.citation_key == brunk.CITATION_KEY
    assert manifest.doi == brunk.DOI
    paths = {record.path: record for record in manifest.files}
    assert set(paths) == {
        brunk.METABOLOMICS_REL,
        brunk.PROTEOMICS_REL,
        brunk.SI_PDF_REL,
        brunk.SI_OCR_REL,
        brunk.PAPER_TEXT_REL,
    }
    assert paths[brunk.SI_OCR_REL].retrieval is None
    assert paths[brunk.SI_OCR_REL].processing is not None
    assert paths[brunk.METABOLOMICS_REL].retrieval is not None
    assert manifest.si_expected == list(brunk.NOT_MIRRORED)


def test_the_deposit_is_idempotent_and_never_overwrites_other_bytes(
    synthetic_mirror: Path, tmp_path: Path
) -> None:
    """A file already at its pin is left alone; a different one raises."""
    sources = {raw.name: tmp_path / "source" / raw.name for raw in brunk.RAW_FILES}
    brunk.deposit_raw_mirror(sources=sources)
    deposited = brunk.raw_mirror_dir() / brunk.PAPER_TEXT_REL
    deposited.write_text("drifted", encoding="utf-8")
    with pytest.raises(RuntimeError, match="exists with a different sha256"):
        brunk.deposit_raw_mirror(sources=sources)


def test_a_source_that_does_not_match_its_pin_is_refused_before_any_write(
    synthetic_mirror: Path, tmp_path: Path
) -> None:
    """A retrieved file whose bytes moved is never deposited."""
    sources = {raw.name: tmp_path / "source" / raw.name for raw in brunk.RAW_FILES}
    drifted = tmp_path / "drifted.txt"
    drifted.write_text("not the article", encoding="utf-8")
    sources[brunk.PAPER_TEXT_FILE] = drifted
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        brunk.deposit_raw_mirror(sources=sources)


def test_the_deposit_refuses_a_missing_source(
    synthetic_mirror: Path, tmp_path: Path
) -> None:
    """Every artifact the manifest names must be handed to the deposit."""
    sources = {
        raw.name: tmp_path / "source" / raw.name
        for raw in brunk.RAW_FILES
        if raw.name != brunk.SI_OCR_FILE
    }
    with pytest.raises(KeyError, match=brunk.SI_OCR_FILE):
        brunk.deposit_raw_mirror(sources=sources)


def test_the_quote_audit_reads_the_pinned_bytes_and_stops_on_drift(
    synthetic_mirror: Path,
) -> None:
    """A hash pin is checked first, then every quote is searched in those bytes."""
    assert brunk.verify_quotes() == len(brunk.PAPER_QUOTES) + len(brunk.SI_QUOTES)
    paper = brunk.raw_mirror_dir() / brunk.PAPER_TEXT_REL
    paper.write_text(paper_text().replace("wild-type E.", "WT E."), encoding="utf-8")
    with pytest.raises(RuntimeError, match="is not the pinned bytes"):
        brunk.verify_quotes()


def test_a_quote_the_mirror_no_longer_carries_stops_the_audit(
    synthetic_mirror: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transcription that drifted from the bytes fails before a value is used."""
    monkeypatch.setattr(
        brunk, "PAPER_QUOTES", brunk.PAPER_QUOTES + ("a sentence nobody wrote",)
    )
    with pytest.raises(RuntimeError, match="quote not verbatim"):
        brunk.verify_quotes()


# --------------------------------------------------------------------------- #
# Hermetic end to end: all four arms over the synthetic release
# --------------------------------------------------------------------------- #
#: The loci the synthetic assembly needs: the four native pathway genes and the two
#: proteins the fixture keys to a locus.
LOCUS_SPECS: tuple[tuple[str, str], ...] = (
    ("b0421", "ispA"),
    ("b1865", "nudB"),
    ("b2224", "atoB"),
    ("b2779", "eno"),
    ("b2889", "idi"),
    ("b3236", "mdh"),
)
ASSEMBLY_REPORT = """# Assembly name:  ASM584v2
# Organism name:  Escherichia coli str. K-12 substr. MG1655 (E. coli)
# Infraspecific name:  strain=K-12 substr. MG1655
# Taxid:          511145
# GenBank assembly accession: GCA_000005845.2
# RefSeq assembly accession: GCF_000005845.2
# RefSeq assembly and GenBank assemblies identical: yes
#
## Assembly-Units:
"""
ASSEMBLY_REPORT_MEMBER = "GCA_000005845.2_ASM584v2_assembly_report.txt"


def _synthetic_loci() -> list[SyntheticLocus]:
    """One locus per :data:`LOCUS_SPECS` entry, laid end to end."""
    loci: list[SyntheticLocus] = []
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
                synonyms=(),
                product=f"synthetic product {tag}",
                protein_id=f"AAC{index:05d}.1",
                protein="MKV",
            )
        )
    return loci


@pytest.fixture
def synthetic_mg1655(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The real MG1655 class over a synthetic assembly, with the network refused."""
    monkeypatch.setattr(fixtures, "SEQUENCE", fixtures.SEQUENCE * 3)
    files = fixtures.write_assembly(
        tmp_path / "tier",
        MG1655_ASSEMBLY,
        _synthetic_loci(),
        [fixtures.gaf_row("eno", "eno|b2779", "GO:0000001")],
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
    root = tmp_path / "mg1655"
    root.mkdir()
    return EcoliK12MG1655Genome(genome_root=str(root), overwrite=False)


@pytest.fixture
def built_stores(
    synthetic_mirror: Path, synthetic_mg1655: Any, monkeypatch: pytest.MonkeyPatch
) -> dict[str, str]:
    """Build all four arms over the synthetic release; return their roots."""
    for attribute, value in (
        ("EXPECTED_UM_COLUMNS", 2),
        ("EXPECTED_UNITLESS_COLUMNS", 2),
        ("EXPECTED_METABOLOME_RECORDS", len(brunk.STRAINS) * len(HOURS)),
        ("EXPECTED_EXOMETABOLITE_RECORDS", len(brunk.STRAINS) * len(HOURS)),
        ("EXPECTED_PROTEOME_RECORDS", len(brunk.STRAINS) * len(PROTEIN_HOURS)),
        ("EXPECTED_HOST_PROTEINS", 4),
        ("EXPECTED_TITER_RECORDS", 8 * len(HOURS)),
        ("EXPECTED_PROTEIN_KEYS", 3),
    ):
        monkeypatch.setattr(brunk, attribute, value)
    monkeypatch.setattr(
        brunk,
        "EXPECTED_BY_FAMILY",
        {
            "metabolome": len(brunk.STRAINS) * len(HOURS),
            "exometabolite": len(brunk.STRAINS) * len(HOURS),
            "proteome": len(brunk.STRAINS) * len(PROTEIN_HOURS),
            "titer": 8 * len(HOURS),
        },
    )
    # The two metabolite arms read their count off a ClassVar bound at import.
    monkeypatch.setattr(
        brunk.MetabolomeBrunk2016Dataset,
        "EXPECTED_RECORDS",
        len(brunk.STRAINS) * len(HOURS),
    )
    monkeypatch.setattr(
        brunk.ExometaboliteBrunk2016Dataset,
        "EXPECTED_RECORDS",
        len(brunk.STRAINS) * len(HOURS),
    )
    roots: dict[str, str] = {}
    for family, (cls, rel) in brunk.FAMILY_BUILDS.items():
        root = str(synthetic_mirror / rel)
        dataset = cls(root=root, ecoli_genome=synthetic_mg1655)  # type: ignore[call-arg]
        if family == "metabolome":
            assert len(dataset) == brunk.EXPECTED_METABOLOME_RECORDS
        dataset.close_lmdb()
        roots[family] = root
    return roots


def test_every_arm_builds_one_record_per_released_sample(
    built_stores: dict[str, str],
) -> None:
    """The four stores hold the samples their own sheet releases."""
    from torchcell.verification.runners import load_records

    counts = {family: len(load_records(root)) for family, root in built_stores.items()}
    assert counts == {
        "metabolome": 18,
        "exometabolite": 18,
        "proteome": 18,
        "titer": 16,
    }


def test_the_proteome_build_drops_the_unexplained_hour_and_the_unkeyed_protein(
    built_stores: dict[str, str],
) -> None:
    """The ``72C`` samples get no hour and ``FRDA`` gets no locus, each with a count."""
    accounting = json.loads(
        Path(
            built_stores["proteome"], "preprocess", "build_accounting.json"
        ).read_text()
    )
    rules = {rule["rule"]: rule for rule in accounting["rules"]}
    assert rules["sample_hour_label_is_not_a_stated_time"]["n_items"] == 8
    assert rules["protein_key_is_not_resolvable_to_one_host_locus"]["items"] == ["FRDA"]
    assert rules["protein_is_not_a_host_protein"]["items"] == ["AmpR", "HMGS", "HMGS"]
    keys = pd.read_csv(Path(built_stores["proteome"], "preprocess", "protein_keys.csv"))
    assert set(keys["protein"]) == {p for p, _, _ in PROTEIN_ROWS}


def test_the_metabolome_build_declines_the_columns_with_no_unit(
    built_stores: dict[str, str],
) -> None:
    """The unitless columns are counted, listed and never stored."""
    from torchcell.verification.runners import load_records

    accounting = json.loads(
        Path(
            built_stores["metabolome"], "preprocess", "build_accounting.json"
        ).read_text()
    )
    rules = {rule["rule"]: rule for rule in accounting["rules"]}
    assert rules["column_header_states_no_unit"]["items"] == ["Glycine", "Alanine"]
    stored: set[str] = set()
    for record in load_records(built_stores["metabolome"]):
        stored |= set(record["experiment"]["phenotype"]["metabolite_level"])
    assert stored == {"atp_c", "cit_c"}


def test_a_titer_record_references_the_non_optimized_variant_of_its_product(
    built_stores: dict[str, str],
) -> None:
    """I1, L1 and B1 are the baselines the SI names, and are their own reference."""
    ledger = pd.read_csv(Path(built_stores["titer"], "preprocess", "titer_rows.csv"))
    assert set(ledger["product"]) == {"isopentenol", "limonene", "bisabolene"}
    assert dict(zip(ledger["strain"], ledger["baseline_strain"]))["I3"] == "I1"
    own = ledger[ledger["strain"] == ledger["baseline_strain"]]
    assert set(own["strain"]) == {"I1", "L1", "B1"}
    assert (own["titer_g_per_l"] == own["baseline_titer_g_per_l"]).all()


def test_a_sample_the_wild_type_shares_no_metabolite_with_is_dropped(
    synthetic_mirror: Path, synthetic_mg1655: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record whose hour-matched control measures nothing has no reference."""
    workbook = brunk.raw_mirror_dir() / brunk.METABOLOMICS_REL
    frame = pd.read_excel(workbook, sheet_name=brunk.SHEET_METABOLITES)
    blanked = (frame["Strain"] == brunk.WILD_TYPE) & (frame["Hour"] == 0.0)
    frame.loc[blanked, ["ATP (uM)", "citrate (uM)"]] = None
    with pd.ExcelWriter(
        workbook, mode="a", engine="openpyxl", if_sheet_exists="replace"
    ) as writer:
        frame.to_excel(writer, sheet_name=brunk.SHEET_METABOLITES, index=False)
    monkeypatch.setattr(brunk, "METABOLOMICS_SHA256", _sha256(workbook))
    monkeypatch.setattr(
        brunk, "DATA_SHA256", {brunk.METABOLOMICS_FILE: brunk.METABOLOMICS_SHA256}
    )
    monkeypatch.setattr(brunk, "EXPECTED_UM_COLUMNS", 2)
    monkeypatch.setattr(brunk, "EXPECTED_UNITLESS_COLUMNS", 2)
    monkeypatch.setattr(
        brunk.MetabolomeBrunk2016Dataset, "EXPECTED_RECORDS", len(brunk.STRAINS)
    )
    root = str(synthetic_mirror / "data/torchcell/metabolome_probe")
    dataset = brunk.MetabolomeBrunk2016Dataset(root=root, ecoli_genome=synthetic_mg1655)
    assert len(dataset) == len(brunk.STRAINS)
    dataset.close_lmdb()
    accounting = json.loads(
        Path(root, "preprocess", "build_accounting.json").read_text()
    )
    rules = {rule["rule"]: rule for rule in accounting["rules"]}
    assert rules["sample_measures_no_column_of_this_unit_block"]["items"] == ["DH1@0h"]
    unshared = rules["sample_shares_no_metabolite_with_the_wild_type_of_its_hour"]
    assert unshared["n_items"] == 8
    assert all(item.endswith("@0h") for item in unshared["items"])


def test_every_arm_passes_its_own_l0_to_l4_gate(built_stores: dict[str, str]) -> None:
    """The four reports pass over the synthetic build, levels and all."""
    for family, root in built_stores.items():
        report = brunk.verify_build(root, None, family=family)  # type: ignore[arg-type]
        assert report.passed, report.summary()
        levels = {result.level for result in report.results}
        assert {Level.L0, Level.L1, Level.L3, Level.L4} <= levels
        assert Path(root, "preprocess", "verification_report.json").exists()


def test_an_unknown_family_is_refused(built_stores: dict[str, str]) -> None:
    """``verify_build`` serves exactly the four families this module builds."""
    with pytest.raises(RuntimeError, match="is not one of the four families"):
        brunk.verify_build(built_stores["titer"], None, family="rnaseq")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# The real mirror and the real stores
# --------------------------------------------------------------------------- #
@requires_mirror
def test_the_raw_mirror_records_the_digests_the_module_pins() -> None:
    """Every deposited artifact is the bytes the loader reads."""
    manifest = brunk.load_manifest()
    recorded = {record.path: record.sha256 for record in manifest.files}
    assert recorded[brunk.METABOLOMICS_REL] == brunk.METABOLOMICS_SHA256
    assert recorded[brunk.PROTEOMICS_REL] == brunk.PROTEOMICS_SHA256
    assert recorded[brunk.SI_OCR_REL] == brunk.SI_OCR_SHA256
    assert recorded[brunk.PAPER_TEXT_REL] == brunk.PAPER_TEXT_SHA256


@requires_mirror
def test_every_quote_is_verbatim_in_the_deposited_bytes() -> None:
    """18 quotes, re-read from the mirror rather than trusted."""
    assert brunk.verify_quotes() == len(brunk.PAPER_QUOTES) + len(brunk.SI_QUOTES)


@requires_mirror
def test_the_released_table_s1_reads_as_the_nine_strains_and_twelve_plasmids() -> None:
    """The real OCR carries the shift the repair is written for."""
    table = brunk.read_table_s1(brunk.raw_mirror_dir() / brunk.SI_OCR_REL)
    assert len(table.plasmids) == 12
    assert (table.repaired_from, table.repaired_to) == ("DH1", "B2")
    assert table.composition["DH1"] == ()


@requires_mirror
def test_the_released_protein_keys_split_as_the_module_counts_them() -> None:
    """68 host proteins, 44 keyed to a locus by the two released routes."""
    keys = brunk.read_protein_keys(brunk.raw_mirror_dir() / brunk.PROTEOMICS_REL)
    host = [key for key in keys if key.organism == brunk.HOST_SPECIES]
    assert len(host) == brunk.EXPECTED_HOST_PROTEINS
    mapped = [key for key in host if key.gene_name is not None]
    assert len(mapped) == brunk.EXPECTED_PROTEIN_KEYS
    routes = {key.route for key in mapped}
    assert routes == {brunk.ROUTE_UNIPROT, brunk.ROUTE_SINGLE_GENE_GPR}


@requires_stores
def test_the_built_dev_stores_hold_the_measured_counts_and_pass_l0_to_l4() -> None:
    """The four stores under ``$DATA_ROOT`` carry 117, 126, 81 and 72 records."""
    from torchcell.verification.runners import load_records

    expected = {
        "metabolome": brunk.EXPECTED_METABOLOME_RECORDS,
        "exometabolite": brunk.EXPECTED_EXOMETABOLITE_RECORDS,
        "proteome": brunk.EXPECTED_PROTEOME_RECORDS,
        "titer": brunk.EXPECTED_TITER_RECORDS,
    }
    for family, (_, rel) in brunk.FAMILY_BUILDS.items():
        root = osp.join(DATA_ROOT, rel)
        assert len(load_records(root)) == expected[family]
        report = brunk.verify_build(root, DATA_ROOT, family=family)
        assert report.passed, report.summary()
