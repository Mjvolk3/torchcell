# tests/torchcell/sequence/genome/test_bacterial_tier.py
# [[tests.torchcell.sequence.genome.test_bacterial_tier]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/sequence/genome/test_bacterial_tier.py

"""Cross-source checks of the four bacterial genomes against the deposited tier.

The per-strain modules pin what one genome reads from its own route. This module checks
the GenBank-first ingest against the OTHER members of each assembly set, which NCBI
generates independently of the flat file, and against the GAF the yeast genome taught us
to trust:

* the GCA ``_feature_table.txt.gz``: every locus's coordinates, strand, symbol,
  pseudogene class, protein accession, product name and product length;
* the GCA GFF3 in ``data.db``: every locus's coordinates and every coding product;
* the GCA ``_genomic.fna.gz``: every gene's sequence against the CDS cut from the flat
  file's own sequence, and the translation of every CDS against ``_protein.faa.gz``;
* the GCF ``_gene_ontology.gaf.gz`` (NCBI's own GAF, keyed by ``WP_`` accession)
  against the RefSeq inline ``Ontology_term`` route BW25113 and REL606 store;
* the GO Consortium ``ECOLI-uniprot.gaf.gz`` against MG1655's ``UniProtKB`` xrefs;
* ``go-basic.obo``: which stored ids are obsolete, and what dropping them costs.

Every number is a measurement on the deposited bytes (2026.10.07), so a change in any
member, or in the parser, fails by name. Two findings are pinned as they stand rather
than hidden: (1) Biopython joins a quoted qualifier that the flat file wrapped after a
hyphen or a comma with a space, so 14 MG1655, 11 REL606 and 8 KT2440 products carry a
space the GFF and the feature table do not (``PRODUCT_WRAPPED``), and one REL606 product
differs because NCBI's own GFF and feature table carry a stray space the flat file lacks;
(2) the RefSeq inline route carries 49 (BW25113) and 50 (REL606) GO ids that are obsolete
in the pinned release, 31 of them with a ``replaced_by`` successor, and
``remove_deprecated_go_terms`` drops them instead of remapping.

Everything here is ``@pytest.mark.data`` and skipped when the four sets and the GO release
are not under ``$DATA_ROOT/torchcell-genomes``; nothing reaches the network.
"""

import gzip
import os
import os.path as osp
import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from Bio.Seq import Seq

from tests.torchcell.sequence.genome._bacterial_fixtures import forbid_network
from torchcell.literature.manifest import sha256_file
from torchcell.sequence.genome.bacterial import BacterialGenome, _gff_attributes
from torchcell.sequence.genome.base import GeneNameStatus
from torchcell.sequence.genome.ecoli.k12 import (
    EcoliK12BW25113Genome,
    EcoliK12MG1655Genome,
)
from torchcell.sequence.genome.ecoli.rel606 import EcoliBREL606Genome
from torchcell.sequence.genome.pputida.kt2440 import PPutidaKT2440Genome
from torchcell.sequence.genome.registry import (
    ECOLI_B_REL606,
    ECOLI_K12_BW25113,
    ECOLI_K12_MG1655,
    GO_RELEASE_20260805,
    PPUTIDA_KT2440,
    resolve,
)

DATA_ROOT = os.environ.get("DATA_ROOT", "")
TIER_PRESENT = all(
    osp.isfile(osp.join(DATA_ROOT, "torchcell-genomes", s, "manifest.json"))
    for s in (
        ECOLI_K12_MG1655,
        ECOLI_K12_BW25113,
        ECOLI_B_REL606,
        PPUTIDA_KT2440,
        GO_RELEASE_20260805,
    )
)
pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not TIER_PRESENT,
        reason="requires the four bacterial assembly sets and the GO release under "
        "$DATA_ROOT/torchcell-genomes",
    ),
]

GENOME_CLASSES: dict[str, type[BacterialGenome[Any]]] = {
    "MG1655": EcoliK12MG1655Genome,
    "BW25113": EcoliK12BW25113Genome,
    "REL606": EcoliBREL606Genome,
    "KT2440": PPutidaKT2440Genome,
}


@pytest.fixture(scope="module", params=list(GENOME_CLASSES), ids=list(GENOME_CLASSES))
def genome(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[BacterialGenome[Any]]:
    """Each strain from the tier into a temporary cache root, the network refusing."""
    strain = request.param
    with pytest.MonkeyPatch.context() as mp:
        forbid_network(mp)
        built = GENOME_CLASSES[strain](
            genome_root=str(tmp_path_factory.mktemp(strain.lower())), overwrite=True
        )
    yield built


def strain_of(genome: BacterialGenome[Any]) -> str:
    """The key of ``GENOME_CLASSES`` a genome was built from."""
    return next(k for k, v in GENOME_CLASSES.items() if type(genome) is v)


def feature_table(genome: BacterialGenome[Any]) -> list[dict[str, str]]:
    """The GCA feature table as rows keyed by its header (``# feature`` -> ``feature``)."""
    path = resolve(
        genome.ASSEMBLY_SET, genome.ASSEMBLY.genbank_assembly + "_feature_table.txt.gz"
    )
    rows = []
    with gzip.open(path, "rt") as fh:
        header = fh.readline().lstrip("# ").rstrip("\n").split("\t")
        for line in fh:
            rows.append(dict(zip(header, line.rstrip("\n").split("\t"), strict=True)))
    return rows


def ncbi_gaf_by_locus(genome: BacterialGenome[Any]) -> dict[str, set[str]]:
    """NCBI's GCF GAF (``WP_`` accession -> GO) crosswalked to GenBank locus tags through
    the RefSeq GFF's CDS rows (``protein_id`` -> ``locus_tag``) and ``old_locus_tag``.
    """
    member = genome.ASSEMBLY.refseq_assembly + "_gene_ontology.gaf.gz"
    wp_go: dict[str, set[str]] = {}
    with gzip.open(resolve(genome.ASSEMBLY_SET, member), "rt") as fh:
        for line in fh:
            if line.startswith("!"):
                continue
            columns = line.rstrip("\n").split("\t")
            assert "NOT" not in columns[3].split("|")
            wp_go.setdefault(columns[1], set()).add(columns[4])
    wp_to_refseq: dict[str, set[str]] = {}
    with gzip.open(
        resolve(genome.ASSEMBLY_SET, genome.ASSEMBLY.refseq_gff_member), "rt"
    ) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            columns = line.rstrip("\n").split("\t")
            if columns[2] != "CDS":
                continue
            attributes = _gff_attributes(columns[8])
            if "protein_id" in attributes:
                wp_to_refseq.setdefault(attributes["protein_id"][0], set()).add(
                    attributes["locus_tag"][0]
                )
    by_locus: dict[str, set[str]] = {}
    for accession, terms in wp_go.items():
        for refseq_tag in wp_to_refseq[accession]:
            for old in genome.refseq.old_locus_tags.get(refseq_tag, ()):
                by_locus.setdefault(old, set()).update(terms)
    return by_locus


def obo_replaced_by() -> dict[str, list[str]]:
    """``replaced_by`` of every obsolete term in the pinned ``go-basic.obo``."""
    replaced: dict[str, list[str]] = {}
    current: str | None = None
    with open(resolve(GO_RELEASE_20260805, "go-basic.obo")) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line == "[Term]":
                current = None
            elif line.startswith("id: GO:"):
                current = line[4:]
            elif line.startswith("replaced_by: ") and current is not None:
                replaced.setdefault(current, []).append(line[13:])
    return replaced


# --------------------------------------------------------------------------------------
# Measured on the deposited bytes, 2026.10.07.

REPLICON = {
    "MG1655": ("U00096.3", 4_641_652),
    "BW25113": ("CP009273.1", 4_631_469),
    "REL606": ("CP000819.1", 4_629_812),
    "KT2440": ("AE015451.2", 6_181_873),
}
COUNTS = {  # loci, genes, pseudogenes, coding (protein_id) loci
    "MG1655": (4651, 4506, 145, 4290),
    "BW25113": (4490, 4303, 187, 4131),
    "REL606": (4383, 4316, 67, 4209),
    "KT2440": (5786, 5729, 57, 5564),
}
FEATURE_TABLE_CLASSES = {
    "MG1655": {
        "protein_coding": 4290,
        "ncRNA": 107,
        "rRNA": 22,
        "tRNA": 86,
        "pseudogene": 145,
        "other": 1,
    },
    "BW25113": {
        "protein_coding": 4131,
        "antisense_RNA": 43,
        "pseudogene": 182,
        "ncRNA": 19,
        "rRNA": 22,
        "tRNA": 85,
        "tRNA_pseudogene": 3,
        "SRP_RNA": 1,
        "tmRNA": 1,
        "tmRNA_pseudogene": 1,
        "RNase_P_RNA": 1,
        "ncRNA_pseudogene": 1,
    },
    "REL606": {"protein_coding": 4209, "pseudogene": 67, "rRNA": 22, "tRNA": 85},
    "KT2440": {
        "protein_coding": 5564,
        "snoRNA": 64,
        "rRNA": 22,
        "tRNA": 75,
        "pseudogene": 57,
        "ncRNA": 3,
        "tmRNA": 1,
    },
}
#: Coding loci whose flat-file product carries a space Biopython inserted where the
#: file wrapped the value after a hyphen or a comma; the GFF and the feature table have
#: the unbroken string.
PRODUCT_WRAPPED = {
    "MG1655": [
        "b0085", "b0086", "b0087", "b0142", "b0181", "b0353", "b2024", "b2055",
        "b2059", "b2263", "b2264", "b3628", "b3939", "b4233",
    ],
    "BW25113": [],
    "REL606": [
        "ECB_00086", "ECB_00087", "ECB_00088", "ECB_00089", "ECB_00582", "ECB_01821",
        "ECB_01900", "ECB_01926", "ECB_02368", "ECB_03708", "ECB_04101",
    ],
    "KT2440": [
        "PP_0547", "PP_1332", "PP_1334", "PP_1337", "PP_1603", "PP_2698", "PP_4016",
        "PP_4637",
    ],
}  # fmt: skip
#: Feature-table ``with_protein`` CDS rows beyond the loci's product CDS: the isoform
#: proteins (alternative starts) the flat file lists as extra CDS features. MG1655 only.
ISOFORM_ROWS = {
    "MG1655": [
        ("b0149", "QNV50512.1"), ("b0470", "AYC08179.1"), ("b0484", "AYC08180.1"),
        ("b1120", "QNV50513.1"), ("b1888", "AAC74958.2"), ("b2592", "QNV50514.1"),
        ("b3168", "UMR55123.1"), ("b3168", "UMR55124.1"), ("b4346", "QNV50517.1"),
        ("b4795", "QNV50548.1"),
    ],
    "BW25113": [],
    "REL606": [],
    "KT2440": [],
}  # fmt: skip
#: The one product where NCBI's GFF and feature table disagree with the flat file for a
#: reason other than wrapping: they carry a space before the second comma.
PRODUCT_NCBI_DISAGREES = {
    "ECB_00636": (
        "fused N-acetyl glucosamine specific PTS enzyme: IIC, IIB, and IIA components",
        "fused N-acetyl glucosamine specific PTS enzyme: IIC, IIB , and IIA components",
    )
}
#: Joined pseudogenes (segments > 1); no joined gene exists in any of the four.
JOINED_PSEUDOGENES = {
    "MG1655": [],
    "BW25113": [
        ("BW25113_0349", 2, "mhpC"), ("BW25113_0542", 2, "renD"),
        ("BW25113_0553", 2, "nmpC"), ("BW25113_2190", 2, "yejO"),
        ("BW25113_3046", 2, "yqiG"), ("BW25113_3443", 2, "yrhA"),
        ("BW25113_3504", 2, "yhiS"), ("BW25113_4492", 3, "ydbA"),
        ("BW25113_4498", 2, "gatR"), ("BW25113_4569", 2, "yhcE"),
        ("BW25113_4570", 2, "lomR"), ("BW25113_4571", 2, "wbbL"),
        ("BW25113_4579", 2, "yaiX"), ("BW25113_4580", 2, "yaiT"),
        ("BW25113_4582", 2, "yoeA"), ("BW25113_4587", 2, "insN"),
        ("BW25113_4600", 2, "ydfJ"), ("BW25113_4623", 2, "insO"),
    ],
    "REL606": [],
    "KT2440": [],
}  # fmt: skip
STRANDS = {
    "MG1655": {"+": 2216, "-": 2290},
    "BW25113": {"+": 2119, "-": 2184},
    "REL606": {"+": 2079, "-": 2237},
    "KT2440": {"+": 2974, "-": 2755},
}
NON_CODING_PRODUCT_TYPES = {
    "MG1655": {"rRNA": 22, "tRNA": 86, "ncRNA": 107, None: 1},
    "BW25113": {"rRNA": 22, "tRNA": 85, "ncRNA": 64, "tmRNA": 1},
    "REL606": {"rRNA": 22, "tRNA": 85},
    "KT2440": {"rRNA": 22, "tRNA": 75, "ncRNA": 67, "tmRNA": 1},
}
START_CODONS = {
    "MG1655": {"GTG": 336, "TTG": 80, "ATT": 4, "CTG": 2},
    "BW25113": {"GTG": 317, "TTG": 71, "ATT": 2, "CTG": 2},
    "REL606": {"GTG": 399, "TTG": 108, "CTG": 5},
    "KT2440": {"GTG": 634, "TTG": 317, "CTG": 4},
}
#: prfB: the gene is one interval, the CDS a join across the programmed +1 frameshift,
#: so the CDS is one base shorter than the gene and still translates to the protein.
PRFB = {
    "MG1655": ("b2891", 1099, 1098, 365),
    "BW25113": ("BW25113_2891", 1099, 1098, 365),
    "REL606": ("ECB_02723", 1099, 1098, 365),
    "KT2440": ("PP_1495", 1096, 1095, 364),
}
#: Selenoproteins: an internal UGA reads Sec, so table-11 translation has an internal
#: stop and differs from the protein at that one position.
SELENOPROTEINS = {
    "MG1655": [("b1474", "fdnG"), ("b3894", "fdoG"), ("b4079", "fdhF")],
    "BW25113": [
        ("BW25113_1474", "fdnG"),
        ("BW25113_3894", "fdoG"),
        ("BW25113_4079", "fdhF"),
    ],
    "REL606": [("ECB_01432", "fdnG"), ("ECB_03779", "fdoG"), ("ECB_03951", "fdhF")],
    "KT2440": [("PP_0489", "fdoG")],
}
FIRST_PSEUDOGENE = {
    "MG1655": "b0218",
    "BW25113": "BW25113_4659",
    "REL606": "ECB_00042",
    "KT2440": "PP_5434",
}
#: (tag, start, end, strand) of the first and last gene on the replicon.
FIRST_GENE = {
    "MG1655": ("b0001", 190, 255, "+"),
    "BW25113": ("BW25113_0001", 190, 255, "+"),
    "REL606": ("ECB_00001", 190, 255, "+"),
    "KT2440": ("PP_0001", 147, 1019, "-"),
}
LAST_GENE = {
    "MG1655": ("b4403", 4_640_942, 4_641_628, "+"),
    "BW25113": ("BW25113_4403", 4_630_759, 4_631_445, "+"),
    "REL606": ("ECB_04279", 4_629_102, 4_629_788, "+"),
    "KT2440": ("PP_5420", 6_181_463, 6_181_870, "-"),
}
GO_TERMS_TOTAL = {"MG1655": 4084, "BW25113": 1648, "REL606": 1656, "KT2440": 2619}
#: Stored GO ids absent from the pinned DAG (obsolete), the count with a replaced_by,
#: the terms and loci remove_deprecated_go_terms drops, and go_genes' size after.
OBSOLETE_GO = {
    "MG1655": (0, 0, 0, [], 4082),
    "BW25113": (49, 31, 96, ["BW25113_1302", "BW25113_3215", "BW25113_3471"], 1583),
    "REL606": (50, 31, 95, ["ECB_01279", "ECB_03320"], 1599),
    "KT2440": (0, 0, 0, [], 2619),
}
#: NCBI's GCF GAF against the stored inline route: loci only in the GAF, only stored,
#: in both, with equal sets, with the GAF a subset of the stored set.
NCBI_GAF_VS_STORED = {
    "BW25113": (8, 75, 2175, 2029, 2164),
    "REL606": (11, 58, 2195, 2037, 2184),
    "KT2440": (328, 821, 3091, 186, 1702),
}
#: Loci where NCBI's GAF carries a term the inline route lacks, with those terms.
NCBI_GAF_EXCEEDS_INLINE = {
    "BW25113": [
        ("BW25113_2607", "trmD", ["GO:0006400"]),
        ("BW25113_3166", "truB", ["GO:0004730", "GO:0006400"]),
        ("BW25113_3283", "yrdD", ["GO:0003917"]),
        ("BW25113_3485", "yhhJ", ["GO:0043190"]),
        ("BW25113_3520", "yhjB", ["GO:0003700"]),
        ("BW25113_3651", "trmH", ["GO:0008173"]),
        ("BW25113_3741", "mnmG", ["GO:0050660"]),
        ("BW25113_3954", "yijO", ["GO:0003677"]),
        ("BW25113_3965", "trmA", ["GO:0006400"]),
        ("BW25113_4013", "metA", ["GO:0004414", "GO:0009086"]),
        ("BW25113_4150", "ampC", ["GO:0030288"]),
    ],
    "REL606": [
        ("ECB_02496", "trmD", ["GO:0006400"]),
        ("ECB_02816", None, ["GO:0000271"]),
        ("ECB_03033", "truB", ["GO:0004730", "GO:0006400"]),
        ("ECB_03134", "yrdD", ["GO:0003917"]),
        ("ECB_03335", "yhhJ", ["GO:0043190"]),
        ("ECB_03368", "yhjB", ["GO:0003700"]),
        ("ECB_03436", None, ["GO:0003677"]),
        ("ECB_03508", "trmH", ["GO:0008173"]),
        ("ECB_03839", "yijO", ["GO:0003677"]),
        ("ECB_03850", "trmA", ["GO:0006400"]),
        ("ECB_03885", "metA", ["GO:0004414", "GO:0009086"]),
    ],
}
#: Symbols carried by more than one gene of the gene set, hence AMBIGUOUS, with the
#: candidates (BW25113: the IS element symbols, as gene counts).
DUPLICATE_SYMBOLS: dict[str, dict[str, Any]] = {
    "MG1655": {},
    "BW25113": {
        "insA": 6, "insB1": 5, "insC1": 6, "insD1": 6, "insE1": 5, "insF1": 5,
        "insH1": 10, "insI1": 3, "insL1": 3,
    },
    "REL606": {"metZ": ["ECB_t00051", "ECB_t00057"]},
    "KT2440": {"asd": ["PP_1989", "PP_1992"]},
}  # fmt: skip
#: Symbols shared by a gene and a pseudogene; the gene wins, RENAMED. MG1655 only.
GENE_PSEUDOGENE_SYMBOLS = {
    "MG1655": {"insI2": ("b1404", ["b4708"])},
    "BW25113": {},
    "REL606": {},
    "KT2440": {},
}
#: synonyms, RefSeq tags total, RefSeq tags with no old_locus_tag, RefSeq tags that
#: reach a GenBank locus, GenBank loci no RefSeq tag reaches, loci with no symbol.
NAME_LAYERS = {
    "MG1655": (8179, 4651, 4651, 0, 4651, 0),
    "BW25113": (12041, 4519, 151, 4368, 122, 0),
    "REL606": (0, 4507, 254, 4252, 131, 283),
    "KT2440": (0, 5719, 158, 5561, 225, 3702),
}
SYMBOL_RESOLUTIONS = {
    "MG1655": {"renamed": 4506, "non_gene_feature": 144},
    "BW25113": {"renamed": 4254, "ambiguous": 9, "non_gene_feature": 183},
    "REL606": {"renamed": 4036, "ambiguous": 1, "non_gene_feature": 62},
    "KT2440": {"renamed": 2071, "ambiguous": 1, "non_gene_feature": 11},
}
XREF_PREFIXES = {
    "MG1655": {"ASAP": 4449, "ECOCYC": 4651, "UniProtKB/Swiss-Prot": 4275},
    "BW25113": {},
    "REL606": {},
    "KT2440": {},
}
LOCUS_TABLE_COLUMNS = [
    "locus_tag", "symbol", "synonyms", "old_locus_tags", "db_xrefs", "pseudo",
    "replicon", "start", "end", "strand", "segments", "product_feature_type",
    "product", "protein_id", "isoform_protein_ids",
]  # fmt: skip


# --------------------------------------------------------------------------------------


def test_the_replicon_length_agrees_across_flat_file_fasta_and_cache(
    genome: BacterialGenome[Any],
) -> None:
    """One circular replicon; the flat file, the FASTA and ``chr_to_len`` give the same
    length, and the gene set is sorted with the pinned size.
    """
    accession, length = REPLICON[strain_of(genome)]
    (replicon,) = genome.genbank.replicons
    assert (replicon.accession, replicon.length, replicon.topology) == (
        accession,
        length,
        "circular",
    )
    assert len(genome.fasta_dna[accession].seq) == length
    assert dict(genome.chr_to_len) == {1: length}
    loci, genes, pseudogenes, coding = COUNTS[strain_of(genome)]
    assert len(genome.genbank.loci) == loci
    assert len(genome.gene_set) == genes
    assert list(genome.gene_set) == sorted(genome.gene_set)
    assert len(genome.genbank.pseudogene_tags) == pseudogenes
    assert (
        sum(1 for locus_ in genome.genbank.loci.values() if locus_.protein_id) == coding
    )
    assert len(genome.fasta_protein) == len(genome.fasta_cds) == coding


def test_the_feature_table_agrees_with_every_genbank_locus(
    genome: BacterialGenome[Any],
) -> None:
    """NCBI's feature table names exactly the GenBank loci, in the pinned classes, and
    agrees on coordinates, strand, symbol and pseudogene status for every one of them,
    and on protein accession and protein length for every coding locus.
    """
    loci = genome.genbank.loci
    rows = feature_table(genome)
    gene_rows = {r["locus_tag"]: r for r in rows if r["feature"] == "gene"}
    assert gene_rows.keys() == loci.keys()
    classes: dict[str, int] = {}
    for r in gene_rows.values():
        classes[r["class"]] = classes.get(r["class"], 0) + 1
    assert classes == FEATURE_TABLE_CLASSES[strain_of(genome)]
    disagreeing = [
        tag
        for tag, locus in loci.items()
        if (
            int(gene_rows[tag]["start"]),
            int(gene_rows[tag]["end"]),
            gene_rows[tag]["strand"],
            gene_rows[tag]["symbol"] or None,
            gene_rows[tag]["class"].endswith("pseudogene"),
        )
        != (locus.start, locus.end, locus.strand, locus.symbol, locus.pseudo)
    ]
    assert disagreeing == []
    with_protein = {
        (r["locus_tag"], r["product_accession"]): r
        for r in rows
        if r["feature"] == "CDS" and r["class"] == "with_protein"
    }
    coding = {
        (tag, locus_.protein_id) for tag, locus_ in loci.items() if locus_.protein_id
    }
    assert coding <= with_protein.keys()
    isoforms = {
        (tag, accession)
        for tag, locus_ in loci.items()
        for accession in locus_.isoform_protein_ids
    }
    assert with_protein.keys() - coding == isoforms
    assert sorted(isoforms) == ISOFORM_ROWS[strain_of(genome)]
    assert [
        tag
        for tag, accession in coding
        if int(with_protein[(tag, accession)]["product_length"])
        != len(genome.fasta_protein[tag].seq)
    ] == []


def test_products_differ_from_the_gff_only_where_the_flat_file_wrapped(
    genome: BacterialGenome[Any],
) -> None:
    """The GFF (in ``data.db``) and the feature table carry the same product for every
    coding locus. The flat file's product differs for exactly ``PRODUCT_WRAPPED``, each
    by one space Biopython inserted where the file wrapped the value after a hyphen or
    comma, and for ``PRODUCT_NCBI_DISAGREES``, where NCBI's derived files carry a stray
    space the flat file does not.
    """
    loci = genome.genbank.loci
    coding = {
        tag: locus_.protein_id for tag, locus_ in loci.items() if locus_.protein_id
    }
    table = {
        (r["locus_tag"], r["product_accession"]): r["name"]
        for r in feature_table(genome)
        if r["feature"] == "CDS" and r["class"] == "with_protein"
    }
    gff: dict[tuple[str, str], list[str]] = {}
    for feature in genome.db.features_of_type("CDS"):
        tag = feature.attributes["locus_tag"][0]
        for accession in feature.attributes.get("protein_id", []):
            gff[(tag, accession)] = list(feature.attributes["product"])
    assert [
        tag for tag, acc in coding.items() if gff[(tag, acc)] != [table[(tag, acc)]]
    ] == []
    differing = sorted(
        tag for tag, acc in coding.items() if loci[tag].product != table[(tag, acc)]
    )
    strain = strain_of(genome)
    expected_ncbi = [t for t in PRODUCT_NCBI_DISAGREES if t in loci]
    assert differing == sorted(PRODUCT_WRAPPED[strain] + expected_ncbi)
    for tag in PRODUCT_WRAPPED[strain]:
        flat, derived = loci[tag].product, table[(tag, coding[tag])]
        assert flat is not None
        assert flat.replace("- ", "-").replace(", ", ",") == derived.replace(", ", ",")
        assert len(flat) - len(derived) == (2 if tag == "b2024" else 1)
    for tag in expected_ncbi:
        assert (loci[tag].product, table[(tag, coding[tag])]) == PRODUCT_NCBI_DISAGREES[
            tag
        ]


def test_the_cache_holds_every_locus_at_the_flat_file_coordinates(
    genome: BacterialGenome[Any],
) -> None:
    """Every ``data.db`` gene or pseudogene row sits at its GenBank locus's interval; a
    joined pseudogene is several rows spanning the locus. The joined loci are pinned and
    every one is a pseudogene, which ``__getitem__`` refuses by name.
    """
    loci = genome.genbank.loci
    rows: dict[str, list[tuple[int, int, str]]] = {}
    for feature in genome.db.features_of_type(("gene", "pseudogene")):
        rows.setdefault(feature.attributes["locus_tag"][0], []).append(
            (feature.start, feature.end, feature.strand)
        )
    assert rows.keys() == loci.keys()
    single = [
        tag
        for tag, locus in loci.items()
        if locus.segments == 1 and rows[tag] != [(locus.start, locus.end, locus.strand)]
    ]
    assert single == []
    joined = sorted(
        (tag, locus.segments, locus.symbol)
        for tag, locus in loci.items()
        if locus.segments > 1
    )
    assert joined == JOINED_PSEUDOGENES[strain_of(genome)]
    for tag, segments, _ in joined:
        assert loci[tag].pseudo
        assert len(rows[tag]) == segments
        assert (min(r[0] for r in rows[tag]), max(r[1] for r in rows[tag])) == (
            loci[tag].start,
            loci[tag].end,
        )
        with pytest.raises(ValueError, match=f"{tag} is a joined pseudogene"):
            genome[tag]


def test_every_gene_constructs_and_its_three_sequences_agree(
    genome: BacterialGenome[Any],
) -> None:
    """Every gene of the gene set constructs. Its ``seq`` is cut from the FASTA and its
    ``cds`` from the flat file's own sequence, and they are equal for every coding gene
    but prfB, whose CDS joins across the programmed frameshift. Every CDS is a whole
    number of codons; the start codons are pinned; every CDS translates (table 11, start
    read as M) to its protein except the selenoproteins. Non-coding genes carry neither
    a CDS nor a protein, in the pinned product types.
    """
    strain = strain_of(genome)
    loci = genome.genbank.loci
    strands: dict[str, int] = {}
    non_coding: dict[str | None, int] = {}
    starts: dict[str, int] = {}
    seq_differs, not_codons, internal_stop, mismatched = [], [], [], []
    for tag in genome.gene_set:
        gene = genome[tag]
        assert gene is not None and gene.id == tag and gene.chromosome == 1
        locus = loci[tag]
        strands[gene.strand] = strands.get(gene.strand, 0) + 1
        assert len(gene.seq) == locus.end - locus.start + 1
        if gene.cds is None:
            assert gene.protein is None and locus.protein_id is None
            kind = locus.product_feature_type
            non_coding[kind] = non_coding.get(kind, 0) + 1
            continue
        cds = str(gene.cds.seq)
        if cds != gene.seq:
            seq_differs.append((tag, len(gene.seq), len(cds), len(gene.protein.seq)))
        if len(cds) % 3:
            not_codons.append(tag)
        if cds[:3] != "ATG":
            starts[cds[:3]] = starts.get(cds[:3], 0) + 1
        translated = str(Seq(cds).translate(table=11))
        assert translated.endswith("*")
        body = translated[:-1]
        if "*" in body:
            internal_stop.append((tag, locus.symbol))
        if "M" + body[1:] != str(gene.protein.seq):
            mismatched.append((tag, locus.symbol))
    assert strands == STRANDS[strain]
    assert non_coding == NON_CODING_PRODUCT_TYPES[strain]
    assert seq_differs == [PRFB[strain]]
    assert loci[PRFB[strain][0]].symbol == "prfB"
    assert not_codons == []
    assert starts == START_CODONS[strain]
    assert internal_stop == mismatched == SELENOPROTEINS[strain]


def test_a_pseudogene_is_a_locus_without_product_and_a_codon_table_sums_to_one(
    genome: BacterialGenome[Any],
) -> None:
    """A pseudogene resolves to a gene object with ``pseudo`` set and no CDS, protein or
    product; no pseudogene carries a protein id. A coding gene's codon frequencies sum
    to one, with the pinned ATG share for the second gene.
    """
    tag = FIRST_PSEUDOGENE[strain_of(genome)]
    assert genome.genbank.pseudogene_tags[0] == tag
    pseudogene = genome[tag]
    assert pseudogene is not None
    assert (
        pseudogene.pseudo,
        pseudogene.cds,
        pseudogene.protein,
        pseudogene.product,
    ) == (True, None, None, None)
    assert [
        t for t in genome.genbank.pseudogene_tags if genome.genbank.loci[t].protein_id
    ] == []
    second = genome[genome.gene_set[1]]
    assert second is not None
    frequency = second.codon_frequency
    assert sum(frequency.values()) == pytest.approx(1.0)
    atg = 0.030303 if strain_of(genome) == "KT2440" else 0.028015
    assert frequency["ATG"] == pytest.approx(atg, abs=5e-7)


def test_windows_stop_at_the_replicon_ends_and_never_wrap(
    genome: BacterialGenome[Any],
) -> None:
    """The replicon is circular but a window is linear: upstream of the first gene, or
    downstream of the last, a window that runs off the end is refused by name unless
    ``allow_undersize`` clips it to the end; ``window`` and the symmetric window clip.
    Measured on the pinned first and last genes of each strain.
    """
    strain = strain_of(genome)
    first_tag, first_start, first_end, first_strand = FIRST_GENE[strain]
    last_tag, last_start, last_end, last_strand = LAST_GENE[strain]
    length = REPLICON[strain][1]
    loci = genome.genbank.loci
    assert min(genome.gene_set, key=lambda t: loci[t].start) == first_tag
    assert max(genome.gene_set, key=lambda t: loci[t].end) == last_tag
    first, last = genome[first_tag], genome[last_tag]
    assert first is not None and last is not None
    assert (first.start, first.end, first.strand) == (
        first_start,
        first_end,
        first_strand,
    )
    assert (last.start, last.end, last.strand) == (last_start, last_end, last_strand)

    def bounds(result: Any) -> tuple[int, int, int]:
        return (result.start_window, result.end_window, len(result.seq))

    if first_strand == "+":  # the three E. coli: thrL at 190..255
        with pytest.raises(ValueError, match=r"five prime size \(1000\) too large"):
            first.window_five_prime(1000)
        assert bounds(first.window_five_prime(1000, allow_undersize=True)) == (
            0,
            189,
            189,
        )
        assert bounds(first.window_five_prime(100)) == (89, 189, 100)
        assert bounds(first.window_three_prime(1000)) == (255, 1255, 1000)
        assert bounds(first.window(5000)) == (0, 5000, 5000)
        assert bounds(first.window(5000, is_max_size=False)) == (0, 444, 444)
        assert bounds(last.window_five_prime(1000)) == (
            last_start - 1001,
            last_start - 1,
            1000,
        )
        with pytest.raises(ValueError, match=r"3utr size \(1000\) too large"):
            last.window_three_prime(1000)
        assert bounds(last.window_three_prime(1000, allow_undersize=True)) == (
            last_end,
            length,
            length - last_end,
        )
        assert length - last_end == 24
        assert bounds(last.window(5000)) == (length - 5000, length, 5000)
        assert bounds(last.window(5000, is_max_size=False)) == (
            length - 735,
            length,
            735,
        )
    else:  # KT2440: parB on the minus strand at 147..1019; PP_5420 ends 3 bp from the end
        assert bounds(first.window_five_prime(1000)) == (1019, 2019, 1000)
        with pytest.raises(ValueError, match=r"3utr size \(1000\) too large"):
            first.window_three_prime(1000)
        assert bounds(first.window_three_prime(1000, allow_undersize=True)) == (
            0,
            146,
            146,
        )
        assert bounds(first.window(5000)) == (0, 5000, 5000)
        assert bounds(first.window(5000, is_max_size=False)) == (0, 1165, 1165)
        with pytest.raises(ValueError, match=r"five prime size \(100\) too large"):
            last.window_five_prime(100)
        assert bounds(last.window_five_prime(1000, allow_undersize=True)) == (
            last_end,
            length,
            3,
        )
        assert bounds(last.window_three_prime(1000)) == (
            last_start - 1001,
            last_start - 1,
            1000,
        )
        assert bounds(last.window(5000)) == (length - 5000, length, 5000)
        assert bounds(last.window(5000, is_max_size=False)) == (
            length - 414,
            length,
            414,
        )


def test_obsolete_go_ids_are_stored_and_dropped_rather_than_remapped(
    genome: BacterialGenome[Any],
) -> None:
    """The GAF routes (MG1655, KT2440) carry no id outside the pinned DAG. The RefSeq
    inline route (BW25113, REL606) carries 49 and 50 obsolete ids, 31 of each with a
    ``replaced_by`` successor; ``remove_deprecated_go_terms`` drops them (96 and 95
    locus-term pairs, emptying the pinned loci) and does not remap to the successor.
    """
    strain = strain_of(genome)
    missing, replaced, dropped, emptied, genes_after = OBSOLETE_GO[strain]
    dag = genome.go_dag
    terms = {t for ts in genome.go_annotations.values() for t in ts}
    assert len(terms) == GO_TERMS_TOTAL[strain]
    obsolete = sorted(t for t in terms if t not in dag)
    assert len(obsolete) == missing
    assert all(dag[t].is_obsolete is False for t in terms if t in dag)
    successors = obo_replaced_by()
    assert sum(1 for t in obsolete if t in successors) == replaced
    stored = genome.go_annotations
    before = {tag: set(ts) for tag, ts in stored.items()}
    try:
        genome.remove_deprecated_go_terms()
        after = genome.go_annotations
        assert sorted(set(before) - set(after)) == emptied
        assert sum(len(before[t]) - len(after.get(t, ())) for t in before) == dropped
        assert len(genome.go) == genes_after
        for tag in emptied:
            assert before[tag] <= set(obsolete)
        kept = {t for ts in after.values() for t in ts}
        assert kept == terms - set(obsolete)
        added = {s for t in obsolete for s in successors.get(t, [])} - terms
        assert bool(added) == (replaced > 0)
        assert not added & kept
    finally:  # the fixture is shared by the module: put the stored GO back
        genome.go_annotations = stored
        genome._go = None
        genome._go_genes = None


def test_ncbis_own_gaf_against_the_stored_go(genome: BacterialGenome[Any]) -> None:
    """For BW25113 and REL606 the inline ``Ontology_term`` route is a superset of NCBI's
    GCF GAF on all but 11 loci, which are pinned with the terms the GAF adds. For KT2440
    the GOA proteome file and NCBI's GAF are different annotations (equal on 186 of
    3,091 shared loci). MG1655's set has no GCF GAF; its RefSeq GFF retags nothing and
    carries no ``Ontology_term`` row, so the GAF is its only GO.
    """
    strain = strain_of(genome)
    if strain == "MG1655":
        with pytest.raises(KeyError, match="_gene_ontology.gaf.gz"):
            resolve(
                genome.ASSEMBLY_SET,
                genome.ASSEMBLY.refseq_assembly + "_gene_ontology.gaf.gz",
            )
        assert set(genome.refseq.old_locus_tags) == set(genome.genbank.loci)
        assert all(olds == () for olds in genome.refseq.old_locus_tags.values())
        assert genome.refseq.ontology_term_rows == ()
        return
    ncbi = ncbi_gaf_by_locus(genome)
    stored = {tag: set(ts) for tag, ts in genome.go_annotations.items()}
    both = ncbi.keys() & stored.keys()
    assert (
        len(ncbi.keys() - stored.keys()),
        len(stored.keys() - ncbi.keys()),
        len(both),
        sum(1 for t in both if ncbi[t] == stored[t]),
        sum(1 for t in both if ncbi[t] <= stored[t]),
    ) == NCBI_GAF_VS_STORED[strain]
    if strain == "KT2440":
        return
    exceeding = sorted(
        (tag, genome.genbank.loci[tag].symbol, sorted(ncbi[tag] - stored[tag]))
        for tag in both
        if not ncbi[tag] <= stored[tag]
    )
    assert exceeding == NCBI_GAF_EXCEEDS_INLINE[strain]


def test_the_uniprot_gaf_names_mg1655_loci_by_the_accession_the_flat_file_carries(
    genome: BacterialGenome[Any],
) -> None:
    """MG1655 only: every b-number the ECOLI-uniprot GAF reaches whose locus carries a
    ``UniProtKB/Swiss-Prot`` xref is reached under that accession (3,890 of 3,890), while
    the GAF's gene symbol (column 3) is the GenBank symbol for only 3,775 of them, so
    column 11 is the join and column 3 is not.
    """
    if strain_of(genome) != "MG1655":
        pytest.skip("the UniProt GAF is MG1655's route")
    accessions: dict[str, set[str]] = {}
    symbols: dict[str, set[str]] = {}
    spec = genome.ASSEMBLY.go_source
    path = resolve(spec.assembly_set, spec.member)
    assert sha256_file(Path(path)) == genome.go_source.sha256
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("!"):
                continue
            columns = line.rstrip("\n").split("\t")
            if "NOT" in columns[3].split("|"):
                continue
            for token in (t for v in columns[10].split("|") for t in v.split("/")):
                if token in genome.genbank.loci:
                    accessions.setdefault(token, set()).add(columns[1])
                    symbols.setdefault(token, set()).add(columns[2])
    assert len(accessions) == 4014 - 52
    xrefs = {
        tag: {x.split(":", 1)[1] for x in locus_.db_xrefs if x.startswith("UniProtKB")}
        for tag, locus_ in genome.genbank.loci.items()
        if any(x.startswith("UniProtKB") for x in locus_.db_xrefs)
    }
    assert len(xrefs) == 4275
    both = accessions.keys() & xrefs.keys()
    assert len(both) == 3890
    assert [tag for tag in both if not accessions[tag] & xrefs[tag]] == []
    agreeing = sorted(t for t in both if genome.genbank.loci[t].symbol in symbols[t])
    assert len(agreeing) == 3775
    differing = sorted(t for t in both if t not in agreeing)
    assert [
        (t, sorted(symbols[t]), genome.genbank.loci[t].symbol) for t in differing[:3]
    ] == [
        ("b0116", ["lpdA"], "lpd"),
        ("b0159", ["mtnN"], "mtn"),
        ("b0262", ["fbpC"], "afuC"),
    ]


def test_the_name_layers_are_pinned_and_duplicate_symbols_are_ambiguous(
    genome: BacterialGenome[Any],
) -> None:
    """Layer sizes per strain; every symbol resolves RENAMED, AMBIGUOUS (the pinned
    duplicates, with their candidates) or NON_GENE_FEATURE; no two symbols differ only
    by case; a blank name is RETIRED as the empty string; a lower-cased locus tag
    resolves CURRENT and says the match was case-insensitive.
    """
    strain = strain_of(genome)
    index = genome.feature_index
    (
        synonyms,
        refseq_total,
        refseq_without_old,
        refseq_reaching,
        unreached,
        no_symbol,
    ) = NAME_LAYERS[strain]
    assert len(index["synonym"][0]) == synonyms
    assert len(index["old_locus_tag"][0]) == 0
    assert len(genome.refseq.old_locus_tags) == refseq_total
    assert (
        sum(1 for v in genome.refseq.old_locus_tags.values() if not v)
        == refseq_without_old
    )
    assert len(index["refseq_locus_tag"][0]) == refseq_reaching
    reached = {old for olds in genome.refseq.old_locus_tags.values() for old in olds}
    assert len(genome.genbank.loci.keys() - reached) == unreached
    assert (
        sum(1 for locus_ in genome.genbank.loci.values() if locus_.symbol is None)
        == no_symbol
    )
    exact_symbols = index["symbol"][0]
    folded: dict[str, int] = {}
    for symbol in exact_symbols:
        folded[symbol.upper()] = folded.get(symbol.upper(), 0) + 1
    assert [s for s in exact_symbols if folded[s.upper()] > 1] == []
    statuses: dict[str, int] = {}
    for symbol in exact_symbols:
        status = genome.resolve_gene_name(symbol).status.value
        statuses[status] = statuses.get(status, 0) + 1
    assert statuses == SYMBOL_RESOLUTIONS[strain]
    genes = set(genome.gene_set)
    duplicates = {
        s: sorted({t for t in tags if t in genes})
        for s, tags in exact_symbols.items()
        if len({t for t in tags if t in genes}) > 1
    }
    expected = DUPLICATE_SYMBOLS[strain]
    assert duplicates.keys() == expected.keys()
    for symbol, tags in duplicates.items():
        resolution = genome.resolve_gene_name(symbol)
        assert resolution.status is GeneNameStatus.AMBIGUOUS
        assert resolution.candidates == tags
        assert expected[symbol] == (
            len(tags) if isinstance(expected[symbol], int) else tags
        )
    shared = {
        s: (
            sorted({t for t in tags if t in genes}),
            sorted({t for t in tags if t not in genes}),
        )
        for s, tags in exact_symbols.items()
        if len(set(tags)) > 1 and s not in duplicates
    }
    assert {s: (g[0], p) for s, (g, p) in shared.items()} == GENE_PSEUDOGENE_SYMBOLS[
        strain
    ]
    for symbol, (gene, _) in GENE_PSEUDOGENE_SYMBOLS[strain].items():
        resolution = genome.resolve_gene_name(symbol)
        assert (resolution.status, resolution.systematic_name) == (
            GeneNameStatus.RENAMED,
            gene,
        )
    blank = genome.resolve_gene_name("   ")
    assert (blank.status, blank.systematic_name) == (GeneNameStatus.RETIRED, "")
    tag = genome.gene_set[1]
    lowered = genome.resolve_gene_name(tag.lower())
    assert (lowered.status, lowered.systematic_name) == (GeneNameStatus.CURRENT, tag)
    assert lowered.note == (
        None if tag.lower() == tag else "locus tag (case-insensitive match)"
    )


def test_xrefs_locus_table_and_pickle(genome: BacterialGenome[Any]) -> None:
    """Only MG1655's GenBank file carries ``db_xref`` (ASAP, ECOCYC, Swiss-Prot); the
    locus table has one row per locus with the ``GenBankLocus`` columns; a pickled
    genome round-trips its gene set, sequences and GO source.
    """
    strain = strain_of(genome)
    prefixes: dict[str, int] = {}
    for locus in genome.genbank.loci.values():
        for xref in locus.db_xrefs:
            prefix = xref.split(":")[0]
            prefixes[prefix] = prefixes.get(prefix, 0) + 1
    assert prefixes == XREF_PREFIXES[strain]
    table = genome.locus_table
    assert table.shape == (COUNTS[strain][0], len(LOCUS_TABLE_COLUMNS))
    assert list(table.columns) == LOCUS_TABLE_COLUMNS
    assert list(table["locus_tag"]) == list(genome.genbank.loci)
    restored = pickle.loads(pickle.dumps(genome))
    assert list(restored.gene_set) == list(genome.gene_set)
    tag = genome.gene_set[5]
    original, copy = genome[tag], restored[tag]
    assert original is not None and copy is not None
    assert (copy.seq, copy.go, copy.product) == (
        original.seq,
        original.go,
        original.product,
    )
    assert restored.go_source == genome.go_source


def test_carruthers_chassis_symbols_and_span_on_kt2440(
    genome: BacterialGenome[Any],
) -> None:
    """KT2440 only: the eight deletion symbols of the Carruthers 2025 chassis IY1449b
    resolve RENAMED to their loci, ``phaC`` and ``glZ`` are RETIRED, the off-target
    PP_0815 is CURRENT, and the stated 86,812 bp span (4,538,575 to 4,625,386) holds 61
    loci (57 genes, PP_4029 to PP_mr45) with two loci straddling its ends.
    """
    if strain_of(genome) != "KT2440":
        pytest.skip("the Carruthers chassis is KT2440")
    renamed = {
        "phaA": "PP_5003",
        "phaB": "PP_5004",
        "mvaB": "PP_3540",
        "hbdH": "PP_3073",
        "ldhA": "PP_1649",
        "zwfB": "PP_4042",
        "gntZ": "PP_4043",
        "liuC": "PP_4066",
    }
    for symbol, tag in renamed.items():
        resolution = genome.resolve_gene_name(symbol)
        assert (resolution.status, resolution.systematic_name) == (
            GeneNameStatus.RENAMED,
            tag,
        ), symbol
    for symbol in ("phaC", "glZ"):
        assert genome.resolve_gene_name(symbol).status is GeneNameStatus.RETIRED
    assert genome.resolve_gene_name("PP_0815").status is GeneNameStatus.CURRENT
    assert genome.resolve_gene_name("zwf").systematic_name == "PP_5351"
    start, end = 4_538_575, 4_625_386
    loci = genome.genbank.loci
    inside = sorted(
        t for t, locus_ in loci.items() if start <= locus_.start and locus_.end <= end
    )
    assert (len(inside), inside[0], inside[-1]) == (61, "PP_4029", "PP_mr45")
    assert sum(1 for t in inside if not loci[t].pseudo) == 57
    assert {"PP_4042", "PP_4043", "PP_4066"} <= set(inside)
    assert sorted(
        t
        for t, locus_ in loci.items()
        if locus_.start < start <= locus_.end or locus_.start <= end < locus_.end
    ) == ["PP_4027", "PP_4090"]
