# tests/torchcell/sequence/genome/scerevisiae/test_ncbi.py
# [[tests.torchcell.sequence.genome.scerevisiae.test_ncbi]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/sequence/genome/scerevisiae/test_ncbi.py
"""The NCBI yeast genome reader: the replicon table, the keys, and the real set (#990).

The hermetic tests cover what decides whether a gene lands on the right chromosome and
in the right identifier space: the assembly report's table, the chromosome keys the
module assigns, the member names the set is read by, and the refusals. The one test that
reads the deposited CEN.PK113-7D set is marked ``data``, so CI stays hermetic while the
machine that holds the genomes tier checks the literal against the deposit.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from torchcell.sequence.genome.scerevisiae.ncbi import (
    CENPK113_7D_ASSEMBLY,
    MITOCHONDRION,
    PLASMID_2MICRON,
    CenPk1137DGene,
    CenPk1137DGenome,
    read_assembly_report,
)

HEADER = "# Assembly name:  ASM257140v2\n# Assembly level: Chromosome\n"
ROW = "{name}\tassembled-molecule\t{molecule}\tChromosome\t{accession}\tna\tna\tPrimary\t100\tna\n"


def _report(tmp_path: Path, rows: list[tuple[str, str, str]]) -> str:
    path = tmp_path / "report.txt"
    body = "".join(
        ROW.format(name=name, molecule=molecule, accession=accession)
        for name, molecule, accession in rows
    )
    path.write_text(HEADER + body)
    return str(path)


def test_the_report_table_keys_chromosomes_the_mitochondrion_and_the_plasmid(
    tmp_path: Path,
) -> None:
    """A roman numeral is its own number, MT is 0 and p2uM is 17, keyed by accession."""
    path = _report(
        tmp_path,
        [
            ("I", "I", "CP000001.1"),
            ("XVI", "XVI", "CP000016.1"),
            ("MT", "MT", "CP000017.1"),
            ("p2uM", "p2uM", "CP000018.1"),
        ],
    )
    assert read_assembly_report(path) == {
        "CP000001.1": 1,
        "CP000016.1": 16,
        "CP000017.1": MITOCHONDRION,
        "CP000018.1": PLASMID_2MICRON,
    }


def test_the_report_table_refuses_a_molecule_it_has_no_key_for(tmp_path: Path) -> None:
    """A wrong key silently moves a gene to another chromosome, so it is refused."""
    path = _report(tmp_path, [("XVII", "XVII", "CP000019.1")])
    with pytest.raises(ValueError, match="has no chromosome key"):
        read_assembly_report(path)


def test_the_report_table_refuses_a_report_with_no_replicon(tmp_path: Path) -> None:
    """An empty table is a deposit problem, not an empty genome."""
    path = tmp_path / "report.txt"
    path.write_text(HEADER)
    with pytest.raises(ValueError, match="lists no replicon"):
        read_assembly_report(str(path))


def test_the_cenpk_assembly_names_every_member_it_is_read_by() -> None:
    """The five members of the deposited set, under the names the reader resolves."""
    stem = "GCA_002571405.2_ASM257140v2"
    assert CENPK113_7D_ASSEMBLY.genbank_member == f"{stem}_genomic.gbff.gz"
    assert CENPK113_7D_ASSEMBLY.gff_member == f"{stem}_genomic.gff.gz"
    assert CENPK113_7D_ASSEMBLY.dna_fasta_member == f"{stem}_genomic.fna.gz"
    assert CENPK113_7D_ASSEMBLY.protein_fasta_member == f"{stem}_protein.faa.gz"
    assert CENPK113_7D_ASSEMBLY.assembly_report_member == f"{stem}_assembly_report.txt"
    assert CENPK113_7D_ASSEMBLY.assembly_set == "scerevisiae_CENPK113-7D_ASM257140v2"


def test_the_cenpk_replicon_literal_is_eighteen_distinct_keyed_replicons() -> None:
    """16 chromosomes, the mitochondrion and the 2-micron plasmid, one key each."""
    replicons = CENPK113_7D_ASSEMBLY.replicons
    assert len(replicons) == 18
    assert sorted(replicons.values()) == list(range(17)) + [PLASMID_2MICRON]
    assert len(set(replicons)) == 18


def test_the_gene_class_maps_a_replicon_and_refuses_any_other() -> None:
    """The gene's chromosome comes from the assembly's own table, never from a guess."""
    accession = next(a for a, key in CENPK113_7D_ASSEMBLY.replicons.items() if key == 1)
    assert CenPk1137DGene.seqid_to_chromosome(accession) == 1
    with pytest.raises(ValueError, match="is not one of the replicons"):
        CenPk1137DGene.seqid_to_chromosome("chrI")


def test_the_genome_class_declares_its_set_release_and_files() -> None:
    """What the class reads, and that it asks for no CDS FASTA (the flat file has them)."""
    assert CenPk1137DGenome.ASSEMBLY_SET == CENPK113_7D_ASSEMBLY.assembly_set
    assert CenPk1137DGenome.GENOME_VERSION == "ASM257140v2"
    assert CenPk1137DGenome.ANNOTATION_RELEASE == "GCA_002571405.2"
    files = CenPk1137DGenome.release_files()
    assert files.gff == CENPK113_7D_ASSEMBLY.gff_member
    assert files.dna_fasta == CENPK113_7D_ASSEMBLY.dna_fasta_member
    assert files.protein_fasta == CENPK113_7D_ASSEMBLY.protein_fasta_member
    assert files.cds_fasta is None
    assert CenPk1137DGenome.LOCUS_FEATURE_TYPES == frozenset({"gene", "pseudogene"})
    assert "no GO release is keyed" in CenPk1137DGenome.GO_ABSENCE


@pytest.mark.data
def test_the_deposited_set_reads_as_this_strains_own_identifiers(
    tmp_path: Path,
) -> None:
    """The real deposit: the literal is the report's, and the genome reads SCEN_* genes.

    MEASURED on 2026-10-12: 6,171 genes over the 18 replicons, every tag in the
    ``SCEN_*`` namespace, and no GO, which is the point of the class.
    """
    from torchcell.sequence.genome.registry import resolve

    data_root = os.environ["DATA_ROOT"]
    report = resolve(
        CENPK113_7D_ASSEMBLY.assembly_set,
        CENPK113_7D_ASSEMBLY.assembly_report_member,
        data_root=data_root,
    )
    assert read_assembly_report(report) == CENPK113_7D_ASSEMBLY.replicons

    root = tmp_path / "genome"
    root.mkdir()
    genome = CenPk1137DGenome(genome_root=str(root))
    assert len(genome.gene_set) == 6171
    assert all(tag.startswith("SCEN_") for tag in genome.gene_set)
    assert genome.go_annotations == {}
    gene = genome[sorted(genome.gene_set)[0]]
    assert gene is not None
    assert gene.id == "SCEN_A00120"
    assert gene.chromosome == 1
    assert gene.go is None
