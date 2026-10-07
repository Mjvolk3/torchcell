# tests/torchcell/sequence/genome/_bacterial_fixtures.py
# [[tests.torchcell.sequence.genome._bacterial_fixtures]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/sequence/genome/_bacterial_fixtures.py
"""Synthetic NCBI assembly sets for the bacterial genome tests, and the tier stub.

:func:`write_assembly` writes, for one :class:`BacterialAssembly` and a list of
:class:`SyntheticLocus`, every member the genome class reads under the member names the
real class asks for: the GenBank flat file (written by Biopython), the GenBank GFF3, the
replicon and protein FASTAs (gzipped like NCBI's), the RefSeq GFF3 with its
``old_locus_tag`` crosswalk and ``Ontology_term`` rows, and the GAF when the assembly's GO
route is a GAF. :func:`serve_tier` stubs ``resolve`` at every import site so the real
genome classes read those files; :func:`forbid_network` makes every network entry point
raise. Coordinates are 1-based inclusive, as in GFF.
"""

import gzip
import socket
import urllib.request
from pathlib import Path
from typing import Any, Literal

import pytest
import torch_geometric.data
import torch_geometric.data.download
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord
from pydantic import BaseModel, ConfigDict

import torchcell.sequence.genome.bacterial as bacterial
import torchcell.sequence.genome.base as base
import torchcell.sequence.genome.ecoli.k12 as k12
from torchcell.sequence.genome.bacterial import GO_BASIC_OBO, BacterialAssembly
from torchcell.sequence.genome.registry import GO_RELEASE_20260805

#: The synthetic replicon every fixture annotates (100 nt).
SEQUENCE = (
    "ATGAAATAAGGCTTATCATGACGCCCTGAAACCATGGTTTAGCCATGCCTTGA"
    "TTCATGCCGCTGATAGCCGCATGGCTTACCAATGCGTTGAATGCCCG"
)
assert len(SEQUENCE) == 100
GO_OBO = """format-version: 1.2
data-version: releases/synthetic

[Term]
id: GO:0000001
name: live process
namespace: biological_process

[Term]
id: GO:0000002
name: retired process
namespace: biological_process
is_obsolete: true

[Term]
id: GO:0000003
name: live function
namespace: molecular_function

[Term]
id: GO:0000004
name: live component
namespace: cellular_component
"""


class SyntheticLocus(BaseModel):
    """One locus of a synthetic assembly: its gene feature and product."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tag: str
    parts: tuple[tuple[int, int], ...]
    strand: Literal["+", "-"]
    symbol: str | None = None
    synonyms: tuple[str, ...] = ()
    xref: str | None = None
    product_type: str | None = "CDS"
    product: str | None = None
    protein_id: str | None = None
    protein: str | None = None
    isoform: tuple[int, int, str, str] | None = None
    pseudo: bool = False
    refseq_tag: str | None = None
    refseq_go: tuple[str, ...] = ()

    @property
    def start(self) -> int:
        """First base of the first part."""
        return self.parts[0][0]

    @property
    def end(self) -> int:
        """Last base of the last part."""
        return self.parts[-1][1]


def _location(
    parts: tuple[tuple[int, int], ...], strand: str
) -> SimpleLocation | CompoundLocation:
    sign = 1 if strand == "+" else -1
    pieces = [SimpleLocation(s - 1, e, sign) for s, e in parts]
    return pieces[0] if len(pieces) == 1 else CompoundLocation(pieces)


def _genbank(assembly: BacterialAssembly, loci: list[SyntheticLocus]) -> SeqRecord:
    record = SeqRecord(
        Seq(SEQUENCE),
        id=assembly.replicon,
        name=assembly.replicon.split(".")[0],
        description="synthetic replicon",
        annotations={"molecule_type": "DNA", "topology": "circular"},
    )
    for locus in loci:
        location = _location(locus.parts, locus.strand)
        gene: dict[str, list[Any]] = {"locus_tag": [locus.tag]}
        if locus.symbol:
            gene["gene"] = [locus.symbol]
        if locus.synonyms:
            gene["gene_synonym"] = ["; ".join(locus.synonyms)]
        if locus.xref:
            gene["db_xref"] = [locus.xref]
        if locus.pseudo:
            gene["pseudo"] = [None]
        record.features.append(SeqFeature(location, type="gene", qualifiers=gene))
        if locus.product_type is None:
            continue
        product: dict[str, list[Any]] = {"locus_tag": [locus.tag]}
        if locus.product:
            product["product"] = [locus.product]
        if locus.protein_id:
            product["protein_id"] = [locus.protein_id]
            product["db_xref"] = [f"NCBI_GP:{locus.protein_id}"]
        if locus.pseudo:
            product["pseudo"] = [None]
        record.features.append(
            SeqFeature(location, type=locus.product_type, qualifiers=product)
        )
        if locus.isoform is not None:
            start, end, protein_id, _ = locus.isoform
            record.features.append(
                SeqFeature(
                    _location(((start, end),), locus.strand),
                    type="CDS",
                    qualifiers={"locus_tag": [locus.tag], "protein_id": [protein_id]},
                )
            )
    return record


def _gff_row(
    seqid: str, ftype: str, start: int, end: int, strand: str, attrs: str
) -> str:
    return "\t".join(
        [seqid, "Genbank", ftype, str(start), str(end), ".", strand, ".", attrs]
    )


def _genbank_gff(
    assembly: BacterialAssembly, loci: list[SyntheticLocus], omit: set[str]
) -> str:
    rows = [
        "##gff-version 3",
        f"##sequence-region {assembly.replicon} 1 {len(SEQUENCE)}",
    ]
    rows.append(
        _gff_row(
            assembly.replicon,
            "region",
            1,
            len(SEQUENCE),
            "+",
            f"ID={assembly.replicon}:1..{len(SEQUENCE)};Is_circular=true",
        )
    )
    for locus in loci:
        if locus.tag in omit:
            continue
        ftype = "pseudogene" if locus.pseudo else "gene"
        attrs = f"ID=gene-{locus.tag};gbkey=Gene;locus_tag={locus.tag}"
        if locus.symbol:
            attrs += f";Name={locus.symbol};gene={locus.symbol}"
        if locus.pseudo:
            attrs += ";pseudo=true"
        for start, end in locus.parts:
            rows.append(
                _gff_row(assembly.replicon, ftype, start, end, locus.strand, attrs)
            )
        if locus.protein_id:
            rows.append(
                _gff_row(
                    assembly.replicon,
                    "CDS",
                    locus.start,
                    locus.end,
                    locus.strand,
                    f"ID=cds-{locus.protein_id};Parent=gene-{locus.tag};"
                    f"locus_tag={locus.tag};protein_id={locus.protein_id}",
                )
            )
    return "\n".join(rows) + "\n"


def _refseq_gff(assembly: BacterialAssembly, loci: list[SyntheticLocus]) -> str:
    """Every locus as a RefSeq gene (its own tag, or ``refseq_tag`` with the GenBank tag
    as ``old_locus_tag``), its ``refseq_go`` on a CDS row, and one RefSeq-only gene
    without ``old_locus_tag`` whose CDS carries ``GO:0000004``.
    """
    seqid = "NZ_" + assembly.replicon
    rows = ["##gff-version 3"]
    for locus in loci:
        tag = locus.refseq_tag or locus.tag
        old = f";old_locus_tag={locus.tag}" if locus.refseq_tag else ""
        rows.append(
            _gff_row(
                seqid,
                "gene",
                locus.start,
                locus.end,
                locus.strand,
                f"ID=gene-{tag};locus_tag={tag}{old}",
            )
        )
        if locus.refseq_go:
            rows.append(
                _gff_row(
                    seqid,
                    "CDS",
                    locus.start,
                    locus.end,
                    locus.strand,
                    f"ID=cds-{tag};Parent=gene-{tag};locus_tag={tag};"
                    f"Ontology_term={','.join(locus.refseq_go)};"
                    "go_process=some process|0000001||IEA",
                )
            )
    rows.append(
        _gff_row(seqid, "gene", 98, 100, "+", "ID=gene-X_RS99999;locus_tag=X_RS99999")
    )
    rows.append(
        _gff_row(
            seqid,
            "CDS",
            98,
            100,
            "+",
            "ID=cds-X;Parent=gene-X_RS99999;locus_tag=X_RS99999;Ontology_term=GO:0000004",
        )
    )
    return "\n".join(rows) + "\n"


def gaf_row(symbol: str, synonyms: str, go_id: str, qualifier: str = "enables") -> str:
    """One 17-column GAF 2.2 row for a UniProt object."""
    return "\t".join(
        [
            "UniProtKB",
            f"UP_{symbol}",
            symbol,
            qualifier,
            go_id,
            "GO_REF:0000002",
            "IEA",
            "InterPro:IPR000001",
            "F",
            "a protein",
            synonyms,
            "protein",
            "taxon:83333",
            "20260727",
            "InterPro",
            "",
            "",
        ]
    )


def write_assembly(
    root: Path,
    assembly: BacterialAssembly,
    loci: list[SyntheticLocus],
    gaf_rows: list[str] | None = None,
    omit_from_gff: frozenset[str] = frozenset(),
) -> dict[tuple[str, str], Path]:
    """Write the members of one synthetic assembly set (and the GO release) under
    ``root``; returns ``(assembly_set, member) -> path`` for :func:`serve_tier`.
    """
    directory = root / assembly.assembly_set
    directory.mkdir(parents=True, exist_ok=True)
    files: dict[tuple[str, str], Path] = {}

    def put(member: str, text: str, set_id: str = assembly.assembly_set) -> None:
        path = root / set_id / member
        path.parent.mkdir(parents=True, exist_ok=True)
        if member.endswith(".gz"):
            with gzip.open(path, "wt") as handle:
                handle.write(text)
        else:
            path.write_text(text)
        files[(set_id, member)] = path

    gbff = directory / assembly.genbank_member
    with gzip.open(gbff, "wt") as handle:
        SeqIO.write(_genbank(assembly, loci), handle, "genbank")
    files[(assembly.assembly_set, assembly.genbank_member)] = gbff
    put(assembly.gff_member, _genbank_gff(assembly, loci, set(omit_from_gff)))
    put(assembly.dna_fasta_member, f">{assembly.replicon} synthetic\n{SEQUENCE}\n")
    proteins = []
    for locus in loci:
        if locus.protein_id and locus.protein:
            proteins.append(f">{locus.protein_id} {locus.product}\n{locus.protein}\n")
        if locus.isoform is not None:
            proteins.append(f">{locus.isoform[2]} isoform\n{locus.isoform[3]}\n")
    put(assembly.protein_fasta_member, "".join(proteins))
    put(assembly.refseq_gff_member, _refseq_gff(assembly, loci))
    spec = assembly.go_source
    if spec.route == "gaf_synonym_column":
        assert gaf_rows is not None, "a GAF-route assembly needs gaf_rows"
        header = "!gaf-version: 2.2\n!generated-by: synthetic\n"
        put(spec.member, header + "\n".join(gaf_rows) + "\n", spec.assembly_set)
    put(GO_BASIC_OBO, GO_OBO, GO_RELEASE_20260805)
    return files


def serve_tier(
    monkeypatch: pytest.MonkeyPatch, files: dict[tuple[str, str], Path]
) -> list[tuple[str, str]]:
    """Stub ``resolve`` wherever the genome code imported it; record every call."""
    calls: list[tuple[str, str]] = []

    def serve(assembly_set: str, filename: str, **_: Any) -> str:
        calls.append((assembly_set, filename))
        if (assembly_set, filename) not in files:
            raise FileNotFoundError(f"{assembly_set}/{filename} is not in the fixture")
        return str(files[(assembly_set, filename)])

    for module in (base, bacterial, k12):
        monkeypatch.setattr(module, "resolve", serve)
    return calls


def forbid_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every network entry point raises, including the download helper the SGD genome
    uses (``torch_geometric.data.download_url``) and raw socket connections.
    """

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"network access during genome construction: {args!r}")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setattr(urllib.request, "urlretrieve", refuse)
    monkeypatch.setattr("requests.sessions.Session.request", refuse)
    monkeypatch.setattr(torch_geometric.data, "download_url", refuse)
    monkeypatch.setattr(torch_geometric.data.download, "download_url", refuse)


#: Synthetic MG1655: a + CDS, a - CDS with an isoform, a tRNA, a pseudogene with a
#: /pseudo CDS, two genes whose synonyms differ only in case (``pro2`` / ``Pro2``), and a
#: pseudogene joined around an insertion.
MG1655_LOCI = [
    SyntheticLocus(
        tag="b0001",
        parts=((1, 9),),
        strand="+",
        symbol="thrL",
        synonyms=("ECK0001",),
        xref="ECOCYC:EG11277",
        product="thr operon leader peptide",
        protein_id="AAC73112.1",
        protein="MK",
    ),
    SyntheticLocus(
        tag="b0002",
        parts=((12, 26),),
        strand="-",
        symbol="thrA",
        synonyms=("ECK0002", "Hs", "thrA1"),
        product="aspartokinase I",
        protein_id="AAC73113.1",
        protein="MRVLK",
        isoform=(12, 20, "QNV50512.1", "MRV"),
    ),
    SyntheticLocus(
        tag="b0003",
        parts=((30, 41),),
        strand="+",
        symbol="thrW",
        synonyms=("ECK0003",),
        product_type="tRNA",
        product="tRNA-Thr",
    ),
    SyntheticLocus(
        tag="b0004",
        parts=((44, 52),),
        strand="+",
        symbol="yaaP",
        synonyms=("ECK0004",),
        pseudo=True,
    ),
    SyntheticLocus(
        tag="b0005",
        parts=((55, 66),),
        strand="+",
        symbol="proB",
        synonyms=("ECK0005", "pro2"),
        product="glutamate 5-kinase",
        protein_id="AAC73116.1",
        protein="MSDS",
    ),
    SyntheticLocus(
        tag="b0006",
        parts=((70, 81),),
        strand="-",
        symbol="proC",
        synonyms=("ECK0006", "Pro2"),
        product="pyrroline-5-carboxylate reductase",
        protein_id="AAC73117.1",
        protein="MEKK",
    ),
    SyntheticLocus(
        tag="b0007",
        parts=((84, 88), (92, 97)),
        strand="+",
        symbol="insZ",
        synonyms=("ECK0007",),
        product_type=None,
        pseudo=True,
    ),
]

#: Synthetic ECOLI-uniprot GAF: a NOT row, a ``/``-joined pair naming a pseudogene and
#: an absent b-number, a versioned ``b0005.1`` that must not read as b0005, and an
#: obsolete term on b0005.
MG1655_GAF = [
    gaf_row("thrL", "thrL|ECK0001|b0001", "GO:0000001"),
    gaf_row("thrA", "thrA|Hs|b0002|JW0001", "GO:0000003"),
    gaf_row("thrA", "thrA|Hs|b0002|JW0001", "GO:0000004", qualifier="NOT|enables"),
    gaf_row("insZ", "insZ|ychG|b0004/b0099", "GO:0000002"),
    gaf_row("hokC", "hokC|gef|b0005.1", "GO:0000003"),
    gaf_row("proB", "proB|pro2|b0005", "GO:0000001"),
    gaf_row("proB", "proB|pro2|b0005", "GO:0000002"),
]

#: Synthetic BW25113: RefSeq retags every locus (``BW25113_RS...`` with the GenBank tag as
#: ``old_locus_tag``); ECK0003 sits on ``BW25113_4412`` (its numerics disagree with
#: b0003); ECK0005 is on two loci (not one-to-one); ECK0008 and ECK0099 are BW25113's own.
BW25113_LOCI = [
    SyntheticLocus(
        tag="BW25113_0001",
        parts=((1, 9),),
        strand="+",
        symbol="thrL",
        synonyms=("ECK0001", "JW4367"),
        product="thr operon leader peptide",
        protein_id="AIN30539.1",
        protein="MK",
        refseq_tag="BW25113_RS00005",
        refseq_go=("GO:0000001",),
    ),
    SyntheticLocus(
        tag="BW25113_0002",
        parts=((12, 26),),
        strand="-",
        symbol="thrA",
        synonyms=("ECK0002", "JW0001"),
        product="aspartokinase I",
        protein_id="AIN30540.1",
        protein="MRVLK",
        refseq_tag="BW25113_RS00010",
        refseq_go=("GO:0000003", "GO:0000002"),
    ),
    SyntheticLocus(
        tag="BW25113_4412",
        parts=((30, 41),),
        strand="+",
        symbol="hokC",
        synonyms=("ECK0003",),
        product="toxin HokC",
        protein_id="AIN30541.1",
        protein="MKQ",
        refseq_tag="BW25113_RS00015",
    ),
    SyntheticLocus(
        tag="BW25113_0004",
        parts=((44, 52),),
        strand="+",
        symbol="yaaP",
        synonyms=("ECK0004",),
        pseudo=True,
        refseq_tag="BW25113_RS00020",
        refseq_go=("GO:0000001",),
    ),
    SyntheticLocus(
        tag="BW25113_0005",
        parts=((55, 66),),
        strand="+",
        symbol="proB",
        synonyms=("ECK0005", "ECK0099"),
        product="glutamate 5-kinase",
        protein_id="AIN30542.1",
        protein="MSDS",
        refseq_tag="BW25113_RS00025",
    ),
    SyntheticLocus(
        tag="BW25113_0008",
        parts=((70, 81),),
        strand="-",
        symbol="yaaX",
        synonyms=("ECK0008", "ECK0005"),
        product="protein YaaX",
        protein_id="AIN30543.1",
        protein="MEKK",
    ),
]

#: Synthetic KT2440: numbered and named (rRNA, tRNA) tags, a symbol shared by two genes,
#: a pseudogene, and RefSeq ``PP_RS`` retagging.
KT2440_LOCI = [
    SyntheticLocus(
        tag="PP_0001",
        parts=((1, 9),),
        strand="+",
        symbol="parB",
        product="chromosome-partitioning protein",
        protein_id="AAN65635.1",
        protein="MK",
        refseq_tag="PP_RS00005",
    ),
    SyntheticLocus(
        tag="PP_0002",
        parts=((12, 26),),
        strand="-",
        product="partition protein",
        protein_id="AAN65636.1",
        protein="MAKVF",
        refseq_tag="PP_RS00010",
    ),
    SyntheticLocus(
        tag="PP_16SA",
        parts=((30, 41),),
        strand="+",
        product_type="rRNA",
        product="16S ribosomal RNA",
        refseq_tag="PP_RS00015",
    ),
    SyntheticLocus(
        tag="PP_t01",
        parts=((44, 52),),
        strand="+",
        product_type="tRNA",
        product="tRNA-Ala",
    ),
    SyntheticLocus(
        tag="PP_0005",
        parts=((55, 66),),
        strand="+",
        symbol="asd",
        product="aspartate-semialdehyde dehydrogenase",
        protein_id="AAN65637.1",
        protein="MSD",
    ),
    SyntheticLocus(
        tag="PP_0006",
        parts=((70, 81),),
        strand="-",
        symbol="asd",
        product="aspartate-semialdehyde dehydrogenase",
        protein_id="AAN65638.1",
        protein="MEK",
    ),
    SyntheticLocus(tag="PP_0007", parts=((84, 97),), strand="+", pseudo=True),
]

#: Synthetic GOA proteome rows: ``PP0002`` (no underscore) is never read as a tag.
KT2440_GAF = [
    gaf_row("parB", "parB|PP_0001", "GO:0000001"),
    gaf_row("parA", "PP0002|PP_0002", "GO:0000003"),
    gaf_row("asd", "asd|PP_0005", "GO:0000001"),
]
