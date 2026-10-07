# torchcell/sequence/genome/scerevisiae/s288c
# [[torchcell.sequence.genome.scerevisiae.s288c]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/sequence/genome/scerevisiae/s288c
# Test file: tests/torchcell/sequence/genome/scerevisiae/test_s288c.py

"""S. cerevisiae S288C genome access over SGD FASTA/GFF with GO and sequence windows.

:class:`SCerevisiaeGenome` is the SGD subclass of
:class:`~torchcell.sequence.genome.base.AnnotatedGenome`, and :class:`SCerevisiaeGene`
of :class:`~torchcell.sequence.genome.base.AnnotatedGene`. This module keeps only the
SGD specifics: the roman-numeral chromosomes with the mitochondrion at 0
(:data:`CHROMOSOMES`), the SGD locus feature types (:data:`SGD_LOCUS_FEATURE_TYPES`),
GO terms from the GFF ``Ontology_term`` attribute, ``orf_classification`` and the
five-prime-UTR-intron CDS selection, :meth:`SCerevisiaeGenome.drop_chrmt`, ``go.obo``
under ``go_root``, and the ``data/sgd/genome`` default root.

Everything organism-agnostic moved to :mod:`torchcell.sequence.genome.base` on
2026.10.07 (the ``data.db`` machinery, :class:`GeneNameStatus` /
:class:`GeneNameResolution` and the layered resolver, the GO properties, the windows).
Every name this module defined before is re-exported here, so
``from torchcell.sequence.genome.scerevisiae.s288c import ...`` and ``s288c.<name>``
keep working, and rebinding one of those names on this module rebinds it in ``base``
too (:class:`_BaseMirroringModule`).
"""

# Re-exported for callers that rebind them on this module (the synthetic suite
# replaces s288c.filecmp, s288c.resolve and s288c.GffutilsConnectionManager to reach
# the moved machinery); the code that uses them is in base.
import filecmp as filecmp
import os
import os.path as osp
import sys
import types
from typing import Any, ClassVar, cast

from attrs import define, field
from gffutils.feature import Feature
from torch_geometric.data import download_url

from torchcell.sequence import get_chr_from_description, roman_to_int
from torchcell.sequence.db_connection import (
    GffutilsConnectionManager as GffutilsConnectionManager,
)
from torchcell.sequence.genome import base
from torchcell.sequence.genome.base import _ACCESS_CODES as _ACCESS_CODES
from torchcell.sequence.genome.base import _BUILD_TEMP as _BUILD_TEMP
from torchcell.sequence.genome.base import _COMPANION_SUFFIXES as _COMPANION_SUFFIXES
from torchcell.sequence.genome.base import _DAMAGE_CODES as _DAMAGE_CODES
from torchcell.sequence.genome.base import _LOCK_CODES as _LOCK_CODES
from torchcell.sequence.genome.base import _PID_LIMIT as _PID_LIMIT
from torchcell.sequence.genome.base import _PRIVATE_COPY as _PRIVATE_COPY
from torchcell.sequence.genome.base import _UPDATE_CHECKOUT as _UPDATE_CHECKOUT
from torchcell.sequence.genome.base import CREATE_DB_KWARGS as CREATE_DB_KWARGS
from torchcell.sequence.genome.base import GENOME_DB_FILENAME as GENOME_DB_FILENAME
from torchcell.sequence.genome.base import RECORD_VERSION as RECORD_VERSION
from torchcell.sequence.genome.base import SOURCE_TABLE as SOURCE_TABLE
from torchcell.sequence.genome.base import (
    UNTRUSTED_DB_FILENAME as UNTRUSTED_DB_FILENAME,
)
from torchcell.sequence.genome.base import AnnotatedGene as AnnotatedGene
from torchcell.sequence.genome.base import AnnotatedGenome as AnnotatedGenome
from torchcell.sequence.genome.base import GeneNameResolution as GeneNameResolution
from torchcell.sequence.genome.base import GeneNameStatus as GeneNameStatus
from torchcell.sequence.genome.base import (
    GenomeDatabaseInstallError as GenomeDatabaseInstallError,
)
from torchcell.sequence.genome.base import GenomeDatabaseRecord as GenomeDatabaseRecord
from torchcell.sequence.genome.base import (
    GenomeDatabaseRecordError as GenomeDatabaseRecordError,
)
from torchcell.sequence.genome.base import GenomeDatabaseSource as GenomeDatabaseSource
from torchcell.sequence.genome.base import (
    GenomeDatabaseSourceError as GenomeDatabaseSourceError,
)
from torchcell.sequence.genome.base import (
    GenomeDatabaseUnavailableError as GenomeDatabaseUnavailableError,
)
from torchcell.sequence.genome.base import (
    GenomeDatabaseVersionError as GenomeDatabaseVersionError,
)
from torchcell.sequence.genome.base import GenomeReleaseFiles as GenomeReleaseFiles
from torchcell.sequence.genome.base import (
    GenomeRootNotFoundError as GenomeRootNotFoundError,
)
from torchcell.sequence.genome.base import (
    GenomeRootNotWritableError as GenomeRootNotWritableError,
)
from torchcell.sequence.genome.base import _change_counter as _change_counter
from torchcell.sequence.genome.base import (
    _committed_record_json as _committed_record_json,
)
from torchcell.sequence.genome.base import (
    _content_digest_or_none as _content_digest_or_none,
)
from torchcell.sequence.genome.base import (
    _copy_preserving_mode as _copy_preserving_mode,
)
from torchcell.sequence.genome.base import _database_counts as _database_counts
from torchcell.sequence.genome.base import _has_hot_journal as _has_hot_journal
from torchcell.sequence.genome.base import _identity as _identity
from torchcell.sequence.genome.base import (
    _journal_moved_to_kept as _journal_moved_to_kept,
)
from torchcell.sequence.genome.base import _keep_copy as _keep_copy
from torchcell.sequence.genome.base import _kept_wording as _kept_wording
from torchcell.sequence.genome.base import _meta_rows as _meta_rows
from torchcell.sequence.genome.base import _meta_rows_or_zero as _meta_rows_or_zero
from torchcell.sequence.genome.base import _pid_alive as _pid_alive
from torchcell.sequence.genome.base import _read_record_json as _read_record_json
from torchcell.sequence.genome.base import (
    _read_record_json_checked as _read_record_json_checked,
)
from torchcell.sequence.genome.base import _remove_if_present as _remove_if_present
from torchcell.sequence.genome.base import _remove_private_copy as _remove_private_copy
from torchcell.sequence.genome.base import (
    _restore_annotated_genome as _restore_annotated_genome,
)
from torchcell.sequence.genome.base import _ro_uri as _ro_uri
from torchcell.sequence.genome.base import _rollback_equals as _rollback_equals
from torchcell.sequence.genome.base import _root_lock as _root_lock
from torchcell.sequence.genome.base import _sweep_dead as _sweep_dead
from torchcell.sequence.genome.base import _temp_in as _temp_in
from torchcell.sequence.genome.base import _vanished as _vanished
from torchcell.sequence.genome.base import all_codons as all_codons
from torchcell.sequence.genome.base import check_record as check_record
from torchcell.sequence.genome.base import (
    database_content_digest as database_content_digest,
)
from torchcell.sequence.genome.base import (
    genome_database_source as genome_database_source,
)
from torchcell.sequence.genome.base import (
    install_genome_database as install_genome_database,
)
from torchcell.sequence.genome.base import (
    migrate_genome_database as migrate_genome_database,
)
from torchcell.sequence.genome.base import nucleotides as nucleotides
from torchcell.sequence.genome.base import (
    read_genome_database_record as read_genome_database_record,
)
from torchcell.sequence.genome.base import (
    rebuild_genome_database as rebuild_genome_database,
)
from torchcell.sequence.genome.base import record_version as record_version
from torchcell.sequence.genome.base import refuse_newer_record as refuse_newer_record
from torchcell.sequence.genome.base import require_damage as require_damage
from torchcell.sequence.genome.base import untrusted_reason as untrusted_reason
from torchcell.sequence.genome.base import validation_summary as validation_summary
from torchcell.sequence.genome.base import (
    write_genome_database as write_genome_database,
)
from torchcell.sequence.genome.registry import SGD_S288C_R64
from torchcell.sequence.genome.registry import resolve as resolve


class _BaseMirroringModule(types.ModuleType):
    """The type of this module: a name it shares with ``base`` is rebound in both.

    The names re-exported above are second bindings of base's objects. Rebinding one
    of them here alone (``monkeypatch.setattr(s288c, "write_genome_database", ...)``,
    which the synthetic suite does to intercept the machinery mid-flight) would leave
    the code in ``base``, which looks the name up in its own globals, calling the
    original. So an assignment to, or deletion of, a shared name on this module is
    applied to ``base`` as well, and the two modules keep one binding per name.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        """Bind ``name`` here, and in ``base`` when the two share it."""
        if name in _SHARED_WITH_BASE:
            setattr(base, name, value)
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        """Delete ``name`` here, and in ``base`` when the two share it."""
        if name in _SHARED_WITH_BASE:
            delattr(base, name)
        super().__delattr__(name)


#: Every name bound here to the same object as in ``base`` (the re-exports above).
_SHARED_WITH_BASE = frozenset(
    name
    for name, value in list(globals().items())
    if not name.startswith("__") and name in vars(base) and vars(base)[name] is value
)
sys.modules[__name__].__class__ = _BaseMirroringModule

# We put MT at 0, because it is circular, and this preserves arabic to roman
CHROMOSOMES = [
    "chrmt",
    "chrI",
    "chrII",
    "chrIII",
    "chrIV",
    "chrV",
    "chrVI",
    "chrVII",
    "chrVIII",
    "chrIX",
    "chrX",
    "chrXI",
    "chrXII",
    "chrXIII",
    "chrXIV",
    "chrXV",
    "chrXVI",
]


# Gene-like LOCUS feature types in the SGD R64 GFF (a deletion/perturbation can target
# these). "gene" is the protein-coding-ORF universe (== gene_set); the rest are RNA genes,
# transposon genes, pseudogenes, and blocked_reading_frame pseudogenes. Every OTHER GFF
# featuretype (region, CDS, mRNA, ARS, intron, telomere, ...) is deliberately excluded so a
# non-locus feature id can never shadow a real gene name during resolution.
SGD_LOCUS_FEATURE_TYPES = frozenset(
    {
        "gene",
        "tRNA_gene",
        "snoRNA_gene",
        "snRNA_gene",
        "ncRNA_gene",
        "rRNA_gene",
        "telomerase_RNA_gene",
        "transposable_element_gene",
        "pseudogene",
        "blocked_reading_frame",
    }
)


# repr=False: keep AnnotatedGene's __repr__ (attrs would generate one for this class).
@define(repr=False)
class SCerevisiaeGene(AnnotatedGene):
    """A single S. cerevisiae gene resolved from the SGD GFF database and FASTA files."""

    #: SGD carries a gene's GO terms (beside its SO terms) in ``Ontology_term``.
    GO_ATTRIBUTE: ClassVar[str] = "Ontology_term"

    @classmethod
    def seqid_to_chromosome(cls, seqid: str) -> int:
        """``chrI`` .. ``chrXVI`` to 1 .. 16 (roman numerals), ``chrmt`` to 0."""
        maybe_roman_numeral = seqid.split("chr")[-1]
        if maybe_roman_numeral == "mt":
            return 0
        return roman_to_int(maybe_roman_numeral)

    def coding_feature(self) -> Feature:
        """The feature whose coordinates give the gene its sequence.

        The gene row, except for a nuclear gene that has a five-prime-UTR intron at its
        start (no intron strictly inside the gene, no 1 bp CDS at its 5' end): then its
        CDS, the ``Verified`` one when there are several (``orf_classification``), and
        the min..max span when several are ``Verified``.
        """
        # process the feature region and produce a feature
        feature_region = self.db.region(
            region=(
                self.db[self.id].chrom,
                self.db[self.id].start,
                self.db[self.id].end,
            ),
            completely_within=True,
        )
        features = [feature for feature in feature_region]
        contains_five_prime_UTR_intron = False
        no_middle_intron = True
        not_chrmt = self.db[self.id].chrom != "chrmt"
        for some_feature in features:
            if some_feature.featuretype == "five_prime_UTR_intron":
                contains_five_prime_UTR_intron = True
                five_prime_UTR_intron_feature = some_feature

        # 4 genes with single bp CDS 5prime
        no_five_prime_one_bp_cds = True
        for some_feature in features:
            if some_feature.featuretype == "CDS":
                cds_difference = some_feature.start - some_feature.end
                if (
                    some_feature.strand == "+"
                    and cds_difference == 0
                    and self.db[self.id].start == some_feature.start
                ):
                    no_five_prime_one_bp_cds = False

                elif (
                    some_feature.strand == "-"
                    and cds_difference == 0
                    and self.db[self.id].end == some_feature.end
                ):
                    no_five_prime_one_bp_cds = False

            self.db[self.id].start
        # No guarantee introns are same as five_prime_UTR_intron
        if contains_five_prime_UTR_intron:
            if (
                five_prime_UTR_intron_feature.start > self.db[self.id].start
                and five_prime_UTR_intron_feature.end < self.db[self.id].end
            ):
                no_middle_intron = False

        # TODO logic is bit complicated, might want to abstract away.
        if (
            contains_five_prime_UTR_intron
            and no_middle_intron
            and not_chrmt
            and no_five_prime_one_bp_cds
        ):
            cds_features = [
                feature for feature in features if feature.featuretype == "CDS"
            ]
            if len(cds_features) == 0:
                raise ValueError(
                    f"Gene {self.id} has a five_prime_UTR_intron but no CDS feature"
                )
            if len(cds_features) == 1:
                feature = cds_features[0]
            # sometimes we have more than one CDS, we need to select the one we have most confidence in with "Verified" ORF
            else:
                for cds in cds_features:
                    # A ValueError, not KeyError: __getitem__ reads KeyError as
                    # "gene not found", which would hide the malformed CDS.
                    if "orf_classification" not in cds.attributes:
                        raise ValueError(
                            f"Gene {self.id}: CDS {cds.id} at {cds.start}..{cds.end} "
                            "has no orf_classification attribute"
                        )
                verified_orfs = [
                    feature
                    for feature in cds_features
                    if feature.attributes["orf_classification"][0] == "Verified"
                ]
                if len(verified_orfs) == 0:
                    raise ValueError(
                        f"Gene {self.id} has a five_prime_UTR_intron and "
                        f"{len(cds_features)} CDS features, none Verified"
                    )
                if len(verified_orfs) == 1:
                    feature = verified_orfs[0]
                if len(verified_orfs) > 1:
                    feature = Feature()
                    feature.chrom = self.db[self.id].chrom
                    feature.strand = self.db[self.id].strand
                    feature.start = min([feature.start for feature in verified_orfs])
                    feature.end = max([feature.end for feature in verified_orfs])
            assert isinstance(feature, Feature), "feature is not a gffutils Feature"
            # log.warning(f"{self.id} - Using CDS Sequence")
        else:
            feature = self.db[self.id]
        return feature

    def annotate(self, gene_feature: Feature) -> None:
        """The GFF3 reserved attributes and GO terms, then the SGD-only attributes."""
        super().annotate(gene_feature)
        # TODO consider adding these to ABC...
        # Some might be too specific to S. cerevisiae, but so they could be optional
        self.ontology_term = gene_feature.attributes.get("Ontology_term", None)
        self.display = gene_feature.attributes.get("display", None)
        self.dbxref = gene_feature.attributes.get("dbxref", None)
        self.orf_classification = gene_feature.attributes.get(
            "orf_classification", None
        )


def _restore_genome(
    cls: type["SCerevisiaeGenome"],
    genome_root: str,
    go_root: str,
    private_db_path: str | None,
) -> "SCerevisiaeGenome":
    """Unpickling target named by a genome pickled before 2026.10.07, when
    ``__reduce_ex__`` lived here: reopen with ``overwrite=False``, on the same database
    file, through :func:`~torchcell.sequence.genome.base._restore_annotated_genome`
    (the target pickles name now). It keeps a worker that re-imports this module from
    a newer checkout able to load a genome its parent pickled with older code.
    """
    return _restore_annotated_genome(
        cls,
        {"genome_root": genome_root, "go_root": go_root, "overwrite": False},
        private_db_path,
    )


def genome_database_untrusted_reason(genome_root: str) -> str | None:
    """Why ``<genome_root>/data.db`` would be built or migrated by a construction, or
    None when a construction would open it as is. Reads only; data-gated tests call
    it so that a test never performs the first migration of a real root.
    """
    return SCerevisiaeGenome.database_untrusted_reason(genome_root)


@define(eq=False)
class SCerevisiaeGenome(AnnotatedGenome[SCerevisiaeGene]):
    """S288C genome wrapper exposing genes, GO annotations, and sequence queries.

    The SGD R64-4-1 release of the genomes tier (:data:`SGD_S288C_R64`) read under the
    SGD conventions: chromosomes keyed by roman numeral with the mitochondrion at 0,
    :data:`SGD_LOCUS_FEATURE_TYPES` as the locus types, GO terms from the GFF
    ``Ontology_term`` attribute, genes built as :class:`SCerevisiaeGene`,
    :meth:`drop_chrmt`, and GO's ``go.obo`` kept under ``go_root`` (downloaded when
    absent). The ``data.db`` cache contract (the build, the open-time trust check, the
    one-time migration, the private write copy, the root lock and the known limits) is
    the base class's: see :class:`~torchcell.sequence.genome.base.AnnotatedGenome`.
    """

    #: The assembly set in the genomes tier this class reads its release files from.
    ASSEMBLY_SET: ClassVar[str] = SGD_S288C_R64
    #: The release whose files the constructor resolves.
    GENOME_VERSION: ClassVar[str] = "R64-4-1_20230830"
    LOCUS_FEATURE_TYPES: ClassVar[frozenset[str]] = SGD_LOCUS_FEATURE_TYPES
    ANNOTATION_NAME: ClassVar[str] = "R64"
    ANNOTATION_RELEASE: ClassVar[str] = "R64-4-1"

    genome_root: str = field(init=True, repr=False, default="data/sgd/genome")
    go_root: str = field(init=True, repr=False, default="data/go")
    overwrite: bool = field(init=True, repr=True, default=False)

    @classmethod
    def gene_class(cls) -> type[SCerevisiaeGene]:
        """SGD genes are :class:`SCerevisiaeGene`."""
        return SCerevisiaeGene

    @classmethod
    def release_files(cls) -> GenomeReleaseFiles:
        """The SGD release files of :attr:`GENOME_VERSION`."""
        return GenomeReleaseFiles(
            dna_fasta="S288C_reference_sequence_" + cls.GENOME_VERSION + ".fsa",
            gff="saccharomyces_cerevisiae_" + cls.GENOME_VERSION + ".gff",
            protein_fasta="orf_trans_all_" + cls.GENOME_VERSION + ".fasta",
            cds_fasta="orf_coding_all_" + cls.GENOME_VERSION + ".fasta",
        )

    @classmethod
    def fasta_chromosome(cls, record: Any) -> int:
        """The record's ``[chromosome=<roman>]`` tag, or 0 for its
        ``[location=mitochondrion]`` tag (the SGD FASTA description).
        """
        return get_chr_from_description(record.description)

    def _prepare_go_obo(self) -> str:
        """``<go_root>/go.obo``, downloaded from current.geneontology.org when absent."""
        # TODO Not sure if this is now to tightly coupled to GO
        # We do want to remove inaccurate info as early as possible
        # Store GO path for lazy loading
        obo_path = osp.join(self.go_root, "go.obo")
        if not osp.exists(obo_path):
            os.makedirs(self.go_root, exist_ok=True)
            download_url(
                "http://current.geneontology.org/ontology/go.obo", self.go_root
            )
        # GO DAG will be loaded lazily via property
        return obo_path

    def drop_chrmt(self) -> None:
        """Remove all chrmt features from this instance's database copy and cache."""
        mitochondrial_features = [
            f for f in self.db.all_features() if f.seqid == "chrmt"
        ]

        # Remove these features from the gene set cache if it existsc
        if self._gene_set is not None:
            for feature in mitochondrial_features:
                self._gene_set.discard(feature.id)

        # Remove these features from this instance's private copy of the database
        self._write("delete", [feature.id for feature in mitochondrial_features])

        # The locus index and the GO-to-genes map were built from the pre-drop
        # database; reset them so the next access rebuilds without chrmt.
        self._feature_index = None
        self._go_genes = None


def main() -> None:
    """Build the genome from DATA_ROOT and print its gene set."""
    import os

    from dotenv import load_dotenv

    load_dotenv()
    DATA_ROOT = cast(str, os.getenv("DATA_ROOT"))

    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    print(f"genome.gene_set: {genome.gene_set}")
    # orf_classes = []
    # lengths = []
    # for gene in genome.gene_set:
    #     orf_classes.append(genome[gene].orf_classification[0])
    #     lengths.append(len(genome[gene].protein.seq))
    # print(pd.Series(orf_classes).value_counts())
    # genome.go
    # genome.drop_chrmt()
    # print(len(genome.gene_set))
    # genome.drop_empty_go()

    # # genes_not_divisible_by_3 = [
    # #     gene for gene in genome.gene_set if len(genome[gene]) % 3 != 0
    # # ]
    # # print(len(genes_not_divisible_by_3))
    # # print(genes_not_divisible_by_3)
    # # genes_no_start = [
    # #     gene for gene in genes_not_divisible_by_3 if genome[gene].seq[:3] != "ATG"
    # # ]
    # # print(len(genes_no_start))
    # # print(genes_no_start)
    # # print()

    # not_divisible_by_3 = []
    # for gene in genome.gene_set:
    #     if len(str(genome["YIL111W"].cds.seq)) % 3 != 0:
    #         not_divisible_by_3.append(gene)
    # print(len(not_divisible_by_3))
    # print(compute_codon_frequency(str(genome["YIL111W"].cds.seq)))
    print()


if __name__ == "__main__":
    main()
