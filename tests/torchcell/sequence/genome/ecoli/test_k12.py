# tests/torchcell/sequence/genome/ecoli/test_k12.py
# [[tests.torchcell.sequence.genome.ecoli.test_k12]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/sequence/genome/ecoli/test_k12.py
"""E. coli K-12 MG1655 and BW25113 genomes.

Two kinds of test. The synthetic ones run everywhere: the real genome classes read a
synthetic assembly set written by ``_bacterial_fixtures`` through a stubbed ``resolve``,
with every network entry point raising. The tier ones (``@pytest.mark.data``, skipped
when the bacterial sets are absent from ``$DATA_ROOT/torchcell-genomes``) build each
genome from the deposited sets into a temporary cache root and pin the measured counts
of [[plan.bacteria-ontology-genome]] step 3.
"""

import os
import os.path as osp
import pickle
from collections.abc import Iterator
from pathlib import Path

import pytest
from Bio.Seq import Seq
from sortedcontainers import SortedSet

from tests.torchcell.sequence.genome._bacterial_fixtures import (
    BW25113_LOCI,
    MG1655_GAF,
    MG1655_LOCI,
    SEQUENCE,
    forbid_network,
    serve_tier,
    write_assembly,
)
from torchcell.sequence.genome.bacterial import (
    GenomeAnnotationMismatchError,
    GoAnnotationSource,
)
from torchcell.sequence.genome.base import (
    GeneNameStatus,
    GenomeDatabaseSource,
    read_genome_database_record,
)
from torchcell.sequence.genome.ecoli.k12 import (
    BW25113_ASSEMBLY,
    MG1655_ASSEMBLY,
    EckPair,
    EcoliK12BW25113Genome,
    EcoliK12Genome,
    EcoliK12MG1655Genome,
    eck_crosswalk,
)
from torchcell.sequence.genome.registry import (
    ECOLI_K12_BW25113,
    ECOLI_K12_MG1655,
    GO_RELEASE_20260805,
)

# --------------------------------------------------------------------------------------
# Synthetic assembly sets (run everywhere)


@pytest.fixture
def tier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Both synthetic K-12 sets served through ``resolve``; the network refuses."""
    files = write_assembly(tmp_path / "tier", MG1655_ASSEMBLY, MG1655_LOCI, MG1655_GAF)
    files |= write_assembly(tmp_path / "tier", BW25113_ASSEMBLY, BW25113_LOCI)
    forbid_network(monkeypatch)
    return serve_tier(monkeypatch, files)


@pytest.fixture
def mg1655(tmp_path: Path, tier: list[tuple[str, str]]) -> EcoliK12MG1655Genome:
    """The synthetic MG1655 genome, its ``data.db`` built by the constructor."""
    return EcoliK12MG1655Genome(genome_root=str(tmp_path / "mg1655"), overwrite=True)


@pytest.fixture
def bw25113(tmp_path: Path, tier: list[tuple[str, str]]) -> EcoliK12BW25113Genome:
    """The synthetic BW25113 genome."""
    return EcoliK12BW25113Genome(genome_root=str(tmp_path / "bw25113"), overwrite=True)


def test_mg1655_reads_only_its_tier_members_and_records_its_set(
    tmp_path: Path, tier: list[tuple[str, str]]
) -> None:
    """Construction resolves the GenBank-route members, the GO release and the GAF from
    the tier, in order, with the network refusing; ``data.db`` records the GCA GFF.
    """
    root = tmp_path / "mg1655"
    EcoliK12MG1655Genome(genome_root=str(root), overwrite=True)
    gca = "GCA_000005845.2_ASM584v2"
    assert tier == [
        (ECOLI_K12_MG1655, f"{gca}_genomic.fna.gz"),
        (ECOLI_K12_MG1655, f"{gca}_genomic.gff.gz"),
        (ECOLI_K12_MG1655, f"{gca}_protein.faa.gz"),
        (ECOLI_K12_MG1655, f"{gca}_genomic.gbff.gz"),
        (GO_RELEASE_20260805, "go-basic.obo"),
        (ECOLI_K12_MG1655, "GCF_000005845.2_ASM584v2_genomic.gff.gz"),
        (ECOLI_K12_MG1655, "ECOLI-uniprot.gaf.gz"),
    ]
    record = read_genome_database_record(str(root / "data.db"))
    assert record is not None
    assert record.source.model_dump(include={"assembly_set", "gff_filename"}) == {
        "assembly_set": ECOLI_K12_MG1655,
        "gff_filename": f"{gca}_genomic.gff.gz",
    }
    assert isinstance(record.source, GenomeDatabaseSource)
    assert EcoliK12MG1655Genome.database_untrusted_reason(str(root)) is None


def test_mg1655_gene_set_is_the_genbank_non_pseudo_loci(
    mg1655: EcoliK12MG1655Genome,
) -> None:
    """Genes are the non-pseudo locus tags; pseudogenes stay loci of the annotation."""
    assert list(mg1655.gene_set) == ["b0001", "b0002", "b0003", "b0005", "b0006"]
    assert mg1655.genbank.pseudogene_tags == ["b0004", "b0007"]
    assert mg1655.chr_to_nc == {1: "U00096.3"}
    assert mg1655.chr_to_len == {1: 100}
    assert mg1655.strain == "K-12 MG1655"


def test_mg1655_gene_carries_coordinates_sequences_names_and_go(
    mg1655: EcoliK12MG1655Genome,
) -> None:
    """A minus-strand gene: reverse-complemented sequence, its CDS and re-keyed protein,
    GenBank names and product, and the GAF's GO.
    """
    gene = mg1655["b0002"]
    assert gene is not None
    expected = str(Seq(SEQUENCE[11:26]).reverse_complement())
    assert (gene.id, gene.chromosome, gene.start, gene.end, gene.strand) == (
        "b0002",
        1,
        12,
        26,
        "-",
    )
    assert gene.seq == expected
    assert gene.cds is not None and gene.protein is not None
    assert str(gene.cds.seq) == expected
    assert str(gene.protein.seq) == "MRVLK"
    assert (gene.symbol, gene.synonyms, gene.alias) == (
        "thrA",
        ["ECK0002", "Hs", "thrA1"],
        ["ECK0002", "Hs", "thrA1"],
    )
    assert (gene.product, gene.protein_id, gene.pseudo) == (
        "aspartokinase I",
        "AAC73113.1",
        False,
    )
    assert gene.name == ["thrA"]
    assert gene.go == SortedSet(["GO:0000003"])
    trna = mg1655["b0003"]
    assert trna is not None
    assert (trna.go, trna.protein, trna.cds) == (None, None, None)


def test_mg1655_resolve_gene_name_round_trips(mg1655: EcoliK12MG1655Genome) -> None:
    """Locus tag, symbol, ECK synonym, case-folded tag, pseudogene, case-exact
    synonyms, a case-folded collision and a fabricated tag.
    """
    expected = {
        "b0002": (GeneNameStatus.CURRENT, "b0002", [], None),
        "thrA": (
            GeneNameStatus.RENAMED,
            "b0002",
            [],
            "gene symbol of current gene b0002",
        ),
        "ECK0002": (
            GeneNameStatus.RENAMED,
            "b0002",
            [],
            "gene synonym of current gene b0002",
        ),
        " B0002 ": (
            GeneNameStatus.CURRENT,
            "b0002",
            [],
            "locus tag (case-insensitive match)",
        ),
        "b0004": (
            GeneNameStatus.NON_GENE_FEATURE,
            "b0004",
            [],
            "valid GenBank GCA_000005845.2 pseudogene, not a gene feature",
        ),
        "yaaP": (
            GeneNameStatus.NON_GENE_FEATURE,
            "b0004",
            [],
            "gene symbol of pseudogene b0004 (not a gene feature)",
        ),
        "pro2": (
            GeneNameStatus.RENAMED,
            "b0005",
            [],
            "gene synonym of current gene b0005",
        ),
        "Pro2": (
            GeneNameStatus.RENAMED,
            "b0006",
            [],
            "gene synonym of current gene b0006",
        ),
        "PRO2": (
            GeneNameStatus.AMBIGUOUS,
            None,
            ["b0005", "b0006"],
            "gene synonym of multiple current genes (case-insensitive match)",
        ),
        "b9999": (
            GeneNameStatus.RETIRED,
            "b9999",
            [],
            "not found in GCA_000005845.2_ASM584v2; retained as given",
        ),
    }
    for name, (status, systematic, candidates, note) in expected.items():
        r = mg1655.resolve_gene_name(name)
        assert (r.input_name, r.status, r.systematic_name, r.candidates, r.note) == (
            name,
            status,
            systematic,
            candidates,
            note,
        ), name
    assert mg1655.resolve_gene_name("b0004").feature_type == "pseudogene"
    assert mg1655.alias_to_systematic["thrA"] == ["b0002"]
    assert mg1655.alias_to_systematic["ECK0005"] == ["b0005"]


def test_mg1655_go_route_records_what_it_reached(mg1655: EcoliK12MG1655Genome) -> None:
    """GAF column 11 reaches three genes and one pseudogene; ``b0099`` is no locus."""
    source = mg1655.go_source
    assert source.model_dump(exclude={"sha256"}) == {
        "route": "gaf_synonym_column",
        "assembly_set": ECOLI_K12_MG1655,
        "member": "ECOLI-uniprot.gaf.gz",
        "identifier": MG1655_ASSEMBLY.go_source.identifier,
        "rows": 7,
        "not_rows_excluded": 1,
        "rows_without_identifier": 1,
        "identifiers": 5,
        "identifiers_not_in_annotation": ("b0099",),
        "pseudogenes_annotated": 1,
        "genes_annotated": 3,
        "terms": 3,
    }
    assert isinstance(source, GoAnnotationSource)
    assert dict(mg1655.go_annotations) == {
        "b0001": SortedSet(["GO:0000001"]),
        "b0002": SortedSet(["GO:0000003"]),
        "b0004": SortedSet(["GO:0000002"]),
        "b0005": SortedSet(["GO:0000001", "GO:0000002"]),
    }
    assert list(mg1655.go) == ["GO:0000001", "GO:0000002", "GO:0000003"]
    assert dict(mg1655.go_genes) == {
        "GO:0000001": SortedSet(["b0001", "b0005"]),
        "GO:0000002": SortedSet(["b0005"]),
        "GO:0000003": SortedSet(["b0002"]),
    }
    # The DAG is the pinned go-basic.obo; goatools leaves obsolete terms out of it.
    assert ("GO:0000001" in mg1655.go_dag, "GO:0000002" in mg1655.go_dag) == (
        True,
        False,
    )


def test_mg1655_remove_deprecated_go_terms_filters_in_memory(
    mg1655: EcoliK12MG1655Genome,
) -> None:
    """The obsolete term leaves b0005 and empties b0004, whose entry goes."""
    mg1655.remove_deprecated_go_terms()
    assert dict(mg1655.go_annotations) == {
        "b0001": SortedSet(["GO:0000001"]),
        "b0002": SortedSet(["GO:0000003"]),
        "b0005": SortedSet(["GO:0000001"]),
    }
    assert list(mg1655.go) == ["GO:0000001", "GO:0000003"]


def test_mg1655_drop_empty_go_drops_by_feature_id(mg1655: EcoliK12MG1655Genome) -> None:
    """The tRNA and b0006 (no GO) leave the gene set and the database copy; they then
    resolve as retired, like a dropped yeast gene.
    """
    mg1655.drop_empty_go()
    assert list(mg1655.gene_set) == ["b0001", "b0002", "b0005"]
    assert {f.id for f in mg1655.db.features_of_type("gene")} == {
        "gene-b0001",
        "gene-b0002",
        "gene-b0005",
    }
    assert mg1655.resolve_gene_name("b0006").status == GeneNameStatus.RETIRED
    assert mg1655["b0006"] is None


def test_mg1655_refuses_a_joined_pseudogene_and_absent_tags(
    mg1655: EcoliK12MG1655Genome,
) -> None:
    """A joined pseudogene cannot be one interval; an unknown tag is None."""
    with pytest.raises(ValueError) as err:
        mg1655["b0007"]
    assert str(err.value) == (
        "b0007 is a joined pseudogene (2 segments in "
        "GCA_000005845.2_ASM584v2_genomic.gbff.gz); a gene with one start/end "
        "interval cannot represent it"
    )
    assert mg1655["b9999"] is None
    pseudo = mg1655["b0004"]
    assert pseudo is not None
    assert (pseudo.pseudo, pseudo.seq) == (True, SEQUENCE[43:52])


def test_mg1655_pickles_with_its_annotation(mg1655: EcoliK12MG1655Genome) -> None:
    """An unpickled genome reopens the same cache and keeps GenBank and GO state."""
    copy = pickle.loads(pickle.dumps(mg1655))
    assert list(copy.gene_set) == list(mg1655.gene_set)
    assert dict(copy.go_annotations) == dict(mg1655.go_annotations)
    assert copy.resolve_gene_name("thrA").systematic_name == "b0002"


def test_construction_refuses_a_gff_that_disagrees_with_the_genbank_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A GFF missing a GenBank gene is named, not papered over."""
    files = write_assembly(
        tmp_path, MG1655_ASSEMBLY, MG1655_LOCI, MG1655_GAF, frozenset({"b0005"})
    )
    serve_tier(monkeypatch, files)
    with pytest.raises(GenomeAnnotationMismatchError) as err:
        EcoliK12MG1655Genome(genome_root=str(tmp_path / "root"), overwrite=True)
    assert "only in the GenBank file ['b0005']" in str(err.value)


def test_for_strain_picks_the_strain_class(
    tmp_path: Path, tier: list[tuple[str, str]]
) -> None:
    """``for_strain`` returns the strain class; an unknown strain is refused."""
    genome = EcoliK12Genome.for_strain(
        "BW25113", genome_root=str(tmp_path / "bw"), overwrite=True
    )
    assert type(genome) is EcoliK12BW25113Genome
    assert genome.ASSEMBLY_SET == ECOLI_K12_BW25113
    with pytest.raises(ValueError) as err:
        EcoliK12Genome.for_strain("W3110", genome_root=str(tmp_path))  # type: ignore[arg-type]  # the refusal under test
    assert str(err.value) == (
        "unknown E. coli K-12 strain 'W3110'; known: ['BW25113', 'MG1655']"
    )


def test_bw25113_stores_refseq_inline_go_by_default(
    bw25113: EcoliK12BW25113Genome,
) -> None:
    """D9: the stored GO is BW25113's RefSeq ``Ontology_term`` through
    ``old_locus_tag``; the RefSeq-only gene's row reaches no locus.
    """
    assert bw25113.go_source.model_dump(exclude={"sha256", "identifier"}) == {
        "route": "refseq_gff_ontology_term",
        "assembly_set": ECOLI_K12_BW25113,
        "member": "GCF_000750555.1_ASM75055v1_genomic.gff.gz",
        "rows": 4,
        "not_rows_excluded": 0,
        "rows_without_identifier": 1,
        "identifiers": 3,
        "identifiers_not_in_annotation": (),
        "pseudogenes_annotated": 1,
        "genes_annotated": 2,
        "terms": 3,
    }
    assert dict(bw25113.go_annotations) == {
        "BW25113_0001": SortedSet(["GO:0000001"]),
        "BW25113_0002": SortedSet(["GO:0000002", "GO:0000003"]),
        "BW25113_0004": SortedSet(["GO:0000001"]),
    }


def test_bw25113_resolves_refseq_tags_jw_numbers_and_refuses_b_numbers(
    bw25113: EcoliK12BW25113Genome,
) -> None:
    """A RefSeq tag and a JW synonym resolve; a b-number is another namespace."""
    rs = bw25113.resolve_gene_name("BW25113_RS00010")
    assert (rs.status, rs.systematic_name, rs.note) == (
        GeneNameStatus.RENAMED,
        "BW25113_0002",
        "RefSeq locus tag of current gene BW25113_0002",
    )
    jw = bw25113.resolve_gene_name("JW0001")
    assert (jw.status, jw.systematic_name) == (GeneNameStatus.RENAMED, "BW25113_0002")
    assert bw25113.resolve_gene_name("b0002").status == GeneNameStatus.RETIRED
    shared = bw25113.resolve_gene_name("ECK0005")
    assert (shared.status, shared.candidates) == (
        GeneNameStatus.AMBIGUOUS,
        ["BW25113_0005", "BW25113_0008"],
    )


def test_bw25113_eck_crosswalk_and_derived_view(bw25113: EcoliK12BW25113Genome) -> None:
    """The ECK join and the MG1655-derived GO view, which leaves the stored GO alone."""
    crosswalk = bw25113.mg1655_eck_crosswalk()
    assert crosswalk.shared == ("ECK0001", "ECK0002", "ECK0003", "ECK0004", "ECK0005")
    assert [p.eck for p in crosswalk.pairs] == [
        "ECK0001",
        "ECK0002",
        "ECK0003",
        "ECK0004",
    ]
    assert crosswalk.numeric_disagreements == (
        EckPair(eck="ECK0003", mg1655="b0003", bw25113="BW25113_4412"),
    )
    assert crosswalk.not_one_to_one == ("ECK0005",)
    assert crosswalk.mg1655_only == ("ECK0006", "ECK0007")
    assert crosswalk.bw25113_only == ("ECK0008", "ECK0099")
    stored = dict(bw25113.go_annotations)
    view = bw25113.go_annotations_via_mg1655_eck()
    assert view.view == "mg1655_gaf_via_eck"
    assert view.annotations == {
        "BW25113_0001": ("GO:0000001",),
        "BW25113_0002": ("GO:0000003",),
        "BW25113_0004": ("GO:0000002",),
    }
    assert (view.crosswalk_pairs, view.genes_annotated, view.terms) == (4, 2, 2)
    assert (view.basis_assembly_set, view.basis_member) == (
        ECOLI_K12_MG1655,
        "ECOLI-uniprot.gaf.gz",
    )
    assert dict(bw25113.go_annotations) == stored


def test_eck_crosswalk_refuses_swapped_strains(
    mg1655: EcoliK12MG1655Genome, bw25113: EcoliK12BW25113Genome
) -> None:
    """The MG1655 argument must hold b-numbers."""
    with pytest.raises(ValueError) as err:
        eck_crosswalk(bw25113.genbank, mg1655.genbank)
    assert str(err.value) == (
        "GCA_000750555.1_ASM75055v1_genomic.gbff.gz is not a MG1655 annotation "
        r"(locus tags outside b\d{4})"
    )


# --------------------------------------------------------------------------------------
# The deposited tier (data-gated)

DATA_ROOT = os.environ.get("DATA_ROOT", "")
TIER_PRESENT = all(
    osp.isfile(osp.join(DATA_ROOT, "torchcell-genomes", s, "manifest.json"))
    for s in (ECOLI_K12_MG1655, ECOLI_K12_BW25113, GO_RELEASE_20260805)
)
on_tier = [
    pytest.mark.data,
    pytest.mark.skipif(
        not TIER_PRESENT,
        reason="requires the E. coli K-12 assembly sets under $DATA_ROOT/torchcell-genomes",
    ),
]


@pytest.fixture(scope="module")
def tier_mg1655(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[EcoliK12MG1655Genome]:
    """MG1655 from the tier into a temporary cache root, with the network refusing."""
    with pytest.MonkeyPatch.context() as mp:
        forbid_network(mp)
        genome = EcoliK12MG1655Genome(
            genome_root=str(tmp_path_factory.mktemp("mg1655")), overwrite=True
        )
    yield genome


@pytest.fixture(scope="module")
def tier_bw25113(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[EcoliK12BW25113Genome]:
    """BW25113 from the tier into a temporary cache root, with the network refusing."""
    with pytest.MonkeyPatch.context() as mp:
        forbid_network(mp)
        genome = EcoliK12BW25113Genome(
            genome_root=str(tmp_path_factory.mktemp("bw25113")), overwrite=True
        )
    yield genome


class TestTierMG1655:
    """GCA_000005845.2 as deposited."""

    pytestmark = on_tier

    def test_gene_features_from_the_genbank_route(
        self, tier_mg1655: EcoliK12MG1655Genome
    ) -> None:
        """4,651 gene features: 4,506 genes and 145 pseudogenes, none joined."""
        loci = tier_mg1655.genbank.loci
        assert len(loci) == 4651
        assert len(tier_mg1655.gene_set) == 4506
        assert len(tier_mg1655.genbank.pseudogene_tags) == 145
        assert sum(locus.segments > 1 for locus in loci.values()) == 0
        assert (len(tier_mg1655.fasta_cds), len(tier_mg1655.fasta_protein)) == (
            4290,
            4290,
        )

    def test_go_coverage_of_ecoli_uniprot_gaf(
        self, tier_mg1655: EcoliK12MG1655Genome
    ) -> None:
        """4,014 b-numbers reached after the 20 NOT rows (the plan's 4,016 counted a
        substring match, which reads ``b0691.1`` as b0691 and keeps the 20 NOT rows);
        52 are no locus of this annotation; 3,902 genes carry 4,082 terms.
        """
        assert tier_mg1655.go_source.model_dump(
            exclude={"identifiers_not_in_annotation", "identifier"}
        ) == {
            "route": "gaf_synonym_column",
            "assembly_set": ECOLI_K12_MG1655,
            "member": "ECOLI-uniprot.gaf.gz",
            "sha256": "ad338c31d8114ce5579a43be4b8e3b78ab366541968cf76c3a91f82b353a9cdf",
            "rows": 56262,
            "not_rows_excluded": 20,
            "rows_without_identifier": 1729,
            "identifiers": 4014,
            "pseudogenes_annotated": 60,
            "genes_annotated": 3902,
            "terms": 4082,
        }
        assert len(tier_mg1655.go_source.identifiers_not_in_annotation) == 52
        assert len(tier_mg1655.go) == 4082

    def test_every_gaf_term_is_live_in_the_pinned_go_release(
        self, tier_mg1655: EcoliK12MG1655Genome
    ) -> None:
        """go-basic 2026-07-26 holds every term, none obsolete: nothing is removed."""
        before = {k: list(v) for k, v in tier_mg1655.go_annotations.items()}
        tier_mg1655.remove_deprecated_go_terms()
        assert {k: list(v) for k, v in tier_mg1655.go_annotations.items()} == before

    def test_resolve_gene_name_round_trips(
        self, tier_mg1655: EcoliK12MG1655Genome
    ) -> None:
        """b0002 current; thrA and ECK0002 renamed to it; b9999, a JW number (absent
        from MG1655's GenBank file) and a BW25113 tag retired.
        """
        resolve = tier_mg1655.resolve_gene_name
        assert (resolve("b0002").status, resolve("b0002").systematic_name) == (
            GeneNameStatus.CURRENT,
            "b0002",
        )
        for name in ("thrA", "ECK0002"):
            r = resolve(name)
            assert (r.status, r.systematic_name) == (GeneNameStatus.RENAMED, "b0002")
        for name in ("b9999", "JW0001", "BW25113_0002"):
            r = resolve(name)
            assert (r.status, r.systematic_name) == (GeneNameStatus.RETIRED, name)

    def test_cds_translates_to_the_rekeyed_protein(
        self, tier_mg1655: EcoliK12MG1655Genome
    ) -> None:
        """Every CDS translates (table 11, start read as M) to its protein except the
        three selenoproteins, whose UGA reads as Sec.
        """
        mismatches = [
            tag
            for tag, cds in tier_mg1655.fasta_cds.items()
            if "M" + str(cds.seq.translate(table=11)).rstrip("*")[1:]
            != str(tier_mg1655.fasta_protein[tag].seq)
        ]
        assert mismatches == ["b1474", "b3894", "b4079"]
        assert [tier_mg1655.genbank.loci[t].symbol for t in mismatches] == [
            "fdnG",
            "fdoG",
            "fdhF",
        ]


class TestTierBW25113:
    """GCA_000750555.1 as deposited, and its ECK join with MG1655."""

    pytestmark = on_tier

    def test_gene_features_from_the_genbank_route(
        self, tier_bw25113: EcoliK12BW25113Genome
    ) -> None:
        """4,490 gene features: 4,303 genes, 187 pseudogenes, 18 joined pseudogenes."""
        loci = tier_bw25113.genbank.loci
        assert len(loci) == 4490
        assert len(tier_bw25113.gene_set) == 4303
        assert len(tier_bw25113.genbank.pseudogene_tags) == 187
        assert sum(locus.segments > 1 for locus in loci.values()) == 18

    def test_go_coverage_of_the_refseq_inline_route(
        self, tier_bw25113: EcoliK12BW25113Genome
    ) -> None:
        """D9 default: 2,278 Ontology_term rows, 11 on RefSeq-only genes; 2,250 GenBank
        loci reached, 2,188 of them genes, with 1,632 terms.
        """
        assert tier_bw25113.go_source.model_dump(exclude={"identifier"}) == {
            "route": "refseq_gff_ontology_term",
            "assembly_set": ECOLI_K12_BW25113,
            "member": "GCF_000750555.1_ASM75055v1_genomic.gff.gz",
            "sha256": "b5d361ed256bf30f5e2522c54239b241cf2787b7b5a03d4ebf2c95734ee3b1bd",
            "rows": 2278,
            "not_rows_excluded": 0,
            "rows_without_identifier": 11,
            "identifiers": 2250,
            "identifiers_not_in_annotation": (),
            "pseudogenes_annotated": 62,
            "genes_annotated": 2188,
            "terms": 1632,
        }

    def test_resolve_gene_name_round_trips(
        self, tier_bw25113: EcoliK12BW25113Genome
    ) -> None:
        """BW25113_0002 current; thrA, ECK0002, JW0001 and its RefSeq tag renamed to it;
        a fabricated tag and a b-number retired.
        """
        resolve = tier_bw25113.resolve_gene_name
        assert resolve("BW25113_0002").status == GeneNameStatus.CURRENT
        for name in ("thrA", "ECK0002", "JW0001", "BW25113_RS00010"):
            r = resolve(name)
            assert (r.status, r.systematic_name) == (
                GeneNameStatus.RENAMED,
                "BW25113_0002",
            ), name
        for name in ("BW25113_9999", "b0002"):
            assert resolve(name).status == GeneNameStatus.RETIRED

    def test_eck_crosswalk_is_pinned(self, tier_bw25113: EcoliK12BW25113Genome) -> None:
        """4,435 shared ECK ids, 4,423 one-to-one, 11 with disagreeing numerics; 192
        only in MG1655 and 16 only in BW25113.
        """
        crosswalk = tier_bw25113.mg1655_eck_crosswalk()
        assert len(crosswalk.shared) == 4435
        assert len(crosswalk.pairs) == 4423
        assert len(crosswalk.not_one_to_one) == 12
        assert (len(crosswalk.mg1655_only), len(crosswalk.bw25113_only)) == (192, 16)
        assert [
            (p.eck, p.mg1655, p.bw25113) for p in crosswalk.numeric_disagreements
        ] == [
            ("ECK0018", "b0018", "BW25113_4412"),
            ("ECK0057", "b0056", "BW25113_4659"),
            ("ECK0281", "b0282", "BW25113_4694"),
            ("ECK1313", "b1318", "BW25113_4524"),
            ("ECK1536", "b1543", "BW25113_4600"),
            ("ECK2646", "b2649", "BW25113_2650"),
            ("ECK2853", "b2855", "BW25113_2856"),
            ("ECK2858", "b2862", "BW25113_2863"),
            ("ECK3674", "b3682", "BW25113_3683"),
            ("ECK4096", "b4583", "BW25113_4104"),
            ("ECK4329", "b4584", "BW25113_4339"),
        ]

    def test_mg1655_derived_view_is_never_stored(
        self, tier_bw25113: EcoliK12BW25113Genome
    ) -> None:
        """MG1655's GAF through ECK reaches 3,877 loci (3,799 genes, 4,066 terms); the
        stored RefSeq GO is unchanged.
        """
        view = tier_bw25113.go_annotations_via_mg1655_eck()
        assert (len(view.annotations), view.genes_annotated, view.terms) == (
            3877,
            3799,
            4066,
        )
        assert tier_bw25113.go_source.genes_annotated == 2188
        assert len(tier_bw25113.go_annotations) == 2250
