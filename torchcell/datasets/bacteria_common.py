# torchcell/datasets/bacteria_common
# [[torchcell.datasets.bacteria_common]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/bacteria_common
# Test file: tests/torchcell/datasets/test_bacteria_common.py
"""The shared skeleton of the bacterial dataset loaders (E. coli K-12, P. putida KT2440).

The bacterial counterpart of ``scerevisiae/gene_name_reconcile.py``, plus what a
bacterial loader needs that a yeast one does not. Section 4 of
[[plan.bacteria-ontology-genome]] is the design; the dendron note
[[torchcell.datasets.bacteria_common]] carries the per-dataset checklist.

* :func:`bacterial_genome` returns the genome of one reference strain from its default
  ``data.db`` cache root under ``DATA_ROOT`` (read-only reopen, ``overwrite=False``),
  the analogue of ``gene_name_reconcile.default_genome``.
* :func:`reconcile_locus_tags` maps a dataset's source gene names to the strain's
  GenBank locus tags with the retain-all policy of ``reconcile_systematic_names``, in
  the bacterial resolver's order (locus tag, ``old_locus_tag``, RefSeq locus tag, gene
  symbol, ``gene_synonym`` such as ECK, retired), and returns the remapped names with a
  :class:`LocusTagReconciliation` (the status and layer histograms).
* :func:`assembly_reference` builds the ``AssemblyReferenceGenome`` a record stores,
  with the set id and the GenBank accession read from the assembly report deposited in
  the set, so a loader cannot mistype either.
* The namespace vocabulary and the locus-tag patterns are imported from
  ``torchcell.datamodels.schema`` (never restated); :data:`LOCUS_TAG_PATTERNS` compiles
  them and :data:`STRAIN_GENE_NAMESPACES` names the namespace of each strain.
* :func:`eck_crosswalk` joins MG1655 and BW25113 on their shared ECK synonym, for a paper
  that reports one background's identifiers for an experiment done in the other.
* :class:`BacterialGenomeInjector` is the host-aware genome injection the build entry
  points share: a loader names ``ecoli_genome`` or ``pputida_genome`` in ``__init__``
  and states its strain in ``REFERENCE_STRAIN``; the yeast ``genome`` parameter keeps
  receiving ``SCerevisiaeGenome``. Genomes are built only when a loader asks for one,
  so a yeast-only build never touches the bacterial tier.
"""

import inspect
import logging
import os
import os.path as osp
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal, overload

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from torchcell.datamodels.schema import (
    ASSEMBLY_SET_ACCESSIONS,
    BACTERIAL_ASSEMBLY_SETS,
    BACTERIAL_LOCUS_TAG_PATTERN,
    BACTERIAL_LOCUS_TAG_PATTERNS,
    AssemblyReferenceGenome,
    BacterialAssemblySet,
    BacterialGeneNamespace,
    BacterialReferenceStrain,
    BacterialStrainBackground,
)
from torchcell.literature.manifest import sha256_file
from torchcell.sequence.genome.bacterial import (
    BacterialAssembly,
    BacterialGenome,
    GenomeAnnotationMismatchError,
)
from torchcell.sequence.genome.base import GeneNameResolution, GeneNameStatus
from torchcell.sequence.genome.ecoli.k12 import (
    EckCrosswalk,
    EckPair,
    EcoliK12BW25113Genome,
    EcoliK12Genome,
    EcoliK12MG1655Genome,
    EcoliK12StrainName,
)
from torchcell.sequence.genome.ecoli.k12 import eck_crosswalk as _annotation_crosswalk
from torchcell.sequence.genome.pputida.kt2440 import PPutidaKT2440Genome
from torchcell.sequence.genome.registry import resolve

log = logging.getLogger(__name__)

__all__ = [
    "BACTERIAL_ASSEMBLY_SETS",
    "BACTERIAL_GENOME_CLASSES",
    "BACTERIAL_GENOME_PARAMETERS",
    "BACTERIAL_LOCUS_TAG_PATTERN",
    "BACTERIAL_LOCUS_TAG_PATTERNS",
    "HOST_STRAINS",
    "LAYER_LOCUS_TAG",
    "LAYER_NOT_FOUND",
    "LOCUS_TAG_PATTERNS",
    "REFERENCE_STRAIN_ATTRIBUTE",
    "STRAIN_GENE_NAMESPACES",
    "AssemblyReport",
    "BacterialGeneNamespace",
    "BacterialGenomeInjector",
    "BacterialHost",
    "BacterialReferenceStrain",
    "EckCrosswalk",
    "EckPair",
    "LocusTagReconciliation",
    "LocusTagResolutionError",
    "assembly_reference",
    "bacterial_genome",
    "declared_reference_strain",
    "eck_crosswalk",
    "gene_namespace_of",
    "host_of_strain",
    "read_assembly_report",
    "reconcile_locus_tags",
    "resolution_layer",
    "strain_of_assembly_set",
]

# --------------------------------------------------------------------------- #
# Hosts, strains, namespaces
# --------------------------------------------------------------------------- #
BacterialHost = Literal["ecoli", "pputida"]
"""A bacterial host; each names the loader parameter its genome is injected through."""

#: The reference strains of each host. Two for E. coli, never one: a ``BW25113_`` number
#: is not an MG1655 b-number, so an E. coli loader states which strain it is written
#: against (plan D5).
HOST_STRAINS: dict[BacterialHost, tuple[BacterialReferenceStrain, ...]] = {
    "ecoli": ("MG1655", "BW25113"),
    "pputida": ("KT2440",),
}

#: The identifier namespace of each strain's GenBank locus tags. Values are typed by the
#: schema's ``BacterialGeneNamespace`` Literal, so a value outside it fails type checking.
STRAIN_GENE_NAMESPACES: dict[BacterialReferenceStrain, BacterialGeneNamespace] = {
    "MG1655": "ecoli_k12_mg1655_bnumber",
    "BW25113": "ecoli_k12_bw25113_locus_tag",
    "KT2440": "pputida_kt2440_locus_tag",
}

#: The schema's per-namespace locus-tag patterns (``BACTERIAL_LOCUS_TAG_PATTERNS``),
#: compiled. Each is anchored, so ``match`` is a full match.
LOCUS_TAG_PATTERNS: dict[str, re.Pattern[str]] = {
    namespace: re.compile(pattern)
    for namespace, pattern in BACTERIAL_LOCUS_TAG_PATTERNS.items()
}

#: The genome class of each reference strain.
BACTERIAL_GENOME_CLASSES: dict[
    BacterialReferenceStrain, type[EcoliK12Genome] | type[PPutidaKT2440Genome]
] = {
    "MG1655": EcoliK12MG1655Genome,
    "BW25113": EcoliK12BW25113Genome,
    "KT2440": PPutidaKT2440Genome,
}

_STRAIN: TypeAdapter[BacterialReferenceStrain] = TypeAdapter(BacterialReferenceStrain)
_ASSEMBLY_SET: TypeAdapter[BacterialAssemblySet] = TypeAdapter(BacterialAssemblySet)


def host_of_strain(strain: BacterialReferenceStrain) -> BacterialHost:
    """The host a reference strain belongs to."""
    for host, strains in HOST_STRAINS.items():
        if strain in strains:
            return host
    raise ValueError(f"{strain!r} is the strain of no host in {HOST_STRAINS}")


def strain_of_assembly_set(assembly_set: str) -> BacterialReferenceStrain:
    """The reference strain whose deposited assembly set is ``assembly_set``."""
    for strain, set_id in BACTERIAL_ASSEMBLY_SETS.items():
        if set_id == assembly_set:
            return _STRAIN.validate_python(strain)
    raise KeyError(
        f"{assembly_set!r} is not a bacterial assembly set; known: "
        f"{sorted(BACTERIAL_ASSEMBLY_SETS.values())}"
    )


def gene_namespace_of(genome: BacterialGenome[Any]) -> BacterialGeneNamespace:
    """The identifier namespace of a bacterial genome's locus tags."""
    return STRAIN_GENE_NAMESPACES[strain_of_assembly_set(genome.ASSEMBLY_SET)]


# --------------------------------------------------------------------------- #
# The genome of a strain
# --------------------------------------------------------------------------- #
@overload
def bacterial_genome(
    host: Literal["ecoli"], strain: EcoliK12StrainName, data_root: str | None = None
) -> EcoliK12Genome: ...


@overload
def bacterial_genome(
    host: Literal["pputida"], strain: Literal["KT2440"], data_root: str | None = None
) -> PPutidaKT2440Genome: ...


@overload
def bacterial_genome(
    host: BacterialHost, strain: BacterialReferenceStrain, data_root: str | None = None
) -> EcoliK12Genome | PPutidaKT2440Genome: ...


def bacterial_genome(
    host: BacterialHost, strain: BacterialReferenceStrain, data_root: str | None = None
) -> EcoliK12Genome | PPutidaKT2440Genome:
    """The genome of ``strain`` from its default cache root (read-only reference use).

    The cache root is ``<data_root>/<ASSEMBLY.default_genome_root>``
    (``data/ecoli/mg1655/genome``, ``data/ecoli/bw25113/genome``,
    ``data/pputida/kt2440/genome``) with ``data_root`` defaulting to ``$DATA_ROOT``, and
    the genome is opened with ``overwrite=False``: an existing ``data.db`` is reopened,
    never rebuilt in place. The sequence and annotation files come from the genomes tier
    under ``$DATA_ROOT`` through ``registry.resolve``. A strain of another host is refused.
    """
    if strain not in HOST_STRAINS[host]:
        raise ValueError(
            f"{strain!r} is not a {host} strain; {host} strains: {HOST_STRAINS[host]}"
        )
    root = os.environ["DATA_ROOT"] if data_root is None else data_root
    genome_class = BACTERIAL_GENOME_CLASSES[strain]
    return genome_class(
        genome_root=osp.join(root, genome_class.ASSEMBLY.default_genome_root),
        overwrite=False,
    )


# --------------------------------------------------------------------------- #
# The assembly pin
# --------------------------------------------------------------------------- #
class AssemblyReport(BaseModel):
    """The header of an NCBI ``_assembly_report.txt`` deposited in an assembly set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assembly_set: BacterialAssemblySet
    member: str
    sha256: str
    assembly_name: str = Field(description='"# Assembly name", e.g. "ASM584v2".')
    organism_name: str
    infraspecific_name: str
    taxid: int
    genbank_accession: str = Field(description='"# GenBank assembly accession".')
    refseq_accession: str = Field(description='"# RefSeq assembly accession".')


def _assembly(strain: BacterialReferenceStrain) -> BacterialAssembly:
    return BACTERIAL_GENOME_CLASSES[strain].ASSEMBLY


def read_assembly_report(
    strain: BacterialReferenceStrain, data_root: str | None = None
) -> AssemblyReport:
    """Read the deposited GenBank assembly report of ``strain``'s assembly set.

    The member is ``<GenBank assembly>_assembly_report.txt``, resolved (sha256-verified)
    from the tier. The header runs to the first bare ``#`` line; each ``# key: value``
    line in it is read once (a repeated key is refused). The accessions must name the
    assembly the genome class reads, and the pair must be the schema's
    ``ASSEMBLY_SET_ACCESSIONS`` entry for the set.
    """
    assembly = _assembly(strain)
    member = f"{assembly.genbank_assembly}_assembly_report.txt"
    path = Path(resolve(assembly.assembly_set, member, data_root=data_root))
    header: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if line == "#":
            break
        key, separator, value = line.removeprefix("# ").partition(":")
        if not line.startswith("# ") or not separator:
            raise ValueError(f"{member}: unexpected header line {line!r}")
        if key in header:
            raise ValueError(f"{member}: header key {key!r} is repeated")
        header[key] = value.strip()
    report = AssemblyReport(
        assembly_set=_ASSEMBLY_SET.validate_python(assembly.assembly_set),
        member=member,
        sha256=sha256_file(path),
        assembly_name=header["Assembly name"],
        organism_name=header["Organism name"],
        infraspecific_name=header["Infraspecific name"],
        taxid=int(header["Taxid"]),
        genbank_accession=header["GenBank assembly accession"],
        refseq_accession=header["RefSeq assembly accession"],
    )
    named = (
        f"{report.genbank_accession}_{report.assembly_name}",
        f"{report.refseq_accession}_{report.assembly_name}",
    )
    if named != (assembly.genbank_assembly, assembly.refseq_assembly):
        raise GenomeAnnotationMismatchError(
            f"{member} names {named}, but the {strain} genome reads "
            f"{(assembly.genbank_assembly, assembly.refseq_assembly)}"
        )
    pair = (report.genbank_accession, report.refseq_accession)
    if pair != ASSEMBLY_SET_ACCESSIONS[assembly.assembly_set]:
        raise GenomeAnnotationMismatchError(
            f"{member} pairs {pair}; the schema pins "
            f"{ASSEMBLY_SET_ACCESSIONS[assembly.assembly_set]} for {assembly.assembly_set}"
        )
    return report


def assembly_reference(
    strain: BacterialReferenceStrain,
    *,
    background: BacterialStrainBackground | None = None,
    data_root: str | None = None,
) -> AssemblyReferenceGenome:
    """The ``AssemblyReferenceGenome`` a record written against ``strain`` stores.

    ``species`` is the genome class's organism, ``assembly_set`` the strain's set, and
    ``assembly_accession`` the GenBank (``GCA_``) accession read from the deposited
    assembly report: the records' identifiers are GenBank locus tags (plan D1), which
    RefSeq retags for BW25113 and KT2440, so the GenBank assembly is the one they mean.
    ``strain`` is the reference strain, or the background's ``name`` when a background
    is given (the schema requires the two to agree); the background must be an edit of
    ``strain``'s assembly.
    """
    if background is not None and background.reference_strain != strain:
        raise ValueError(
            f"background {background.name!r} is an edit of "
            f"{background.reference_strain!r}, not {strain!r}"
        )
    report = read_assembly_report(strain, data_root)
    return AssemblyReferenceGenome(
        species=_assembly(strain).organism,
        strain=strain if background is None else background.name,
        assembly_set=report.assembly_set,
        assembly_accession=report.genbank_accession,
        background=background,
    )


# --------------------------------------------------------------------------- #
# Retain-all locus-tag reconciliation
# --------------------------------------------------------------------------- #
#: Statuses whose resolved locus tag is preferred over the source name: a current gene,
#: a name of exactly one current gene, or a valid pseudogene locus.
_REMAP_STATUSES = (
    GeneNameStatus.CURRENT,
    GeneNameStatus.RENAMED,
    GeneNameStatus.NON_GENE_FEATURE,
)
#: The layer of a name that is itself a locus tag of the annotation.
LAYER_LOCUS_TAG = "locus tag"
#: The layer of a name no layer resolves (``RETIRED``).
LAYER_NOT_FOUND = "not found"
_CASE_INSENSITIVE = "(case-insensitive match)"


class LocusTagResolutionError(ValueError):
    """A dataset's source names resolve to locus tags below the loader's threshold."""


class LocusTagReconciliation(BaseModel):
    """What :func:`reconcile_locus_tags` did to one dataset's source names.

    Every count is over the distinct source names. ``status_histogram`` and
    ``layer_histogram`` list every status and every resolver layer, zeros included.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    assembly_set: str
    gene_namespace: BacterialGeneNamespace
    unique_names: int
    status_histogram: dict[GeneNameStatus, int]
    layer_histogram: dict[str, int] = Field(
        description="Resolver layer that decided each name: 'locus tag', the genome's "
        "NAME_LAYERS in order, then 'not found'."
    )
    remapped: int = Field(
        description="Names stored as a locus tag other than their own."
    )
    kept_on_collision: tuple[str, ...] = Field(
        description="Names that resolve to a locus another name also resolves to; each "
        "is kept as given so the records stay distinct."
    )
    retired_kept: tuple[str, ...]
    ambiguous_kept: dict[str, tuple[str, ...]] = Field(
        description="Ambiguous name -> its candidate locus tags; kept as given."
    )
    case_insensitive: tuple[str, ...] = Field(
        description="Names resolved only by a case-insensitive match."
    )
    outside_namespace: tuple[str, ...] = Field(
        description="Stored names that are not locus tags of gene_namespace; a "
        "bacterial perturbation leaf refuses them."
    )

    @property
    def resolved(self) -> int:
        """Names that resolved to one locus (CURRENT, RENAMED or NON_GENE_FEATURE)."""
        return sum(self.status_histogram[status] for status in _REMAP_STATUSES)

    @property
    def resolved_fraction(self) -> float:
        """``resolved`` over ``unique_names``."""
        return self.resolved / self.unique_names

    def require_resolved(self, min_fraction: float) -> None:
        """Refuse the dataset when fewer than ``min_fraction`` of its names resolve.

        Checklist item 4: a dataset below its stated threshold stops and reports; its
        records are never dropped to pass.
        """
        if self.resolved_fraction < min_fraction:
            statuses = {s.value: n for s, n in self.status_histogram.items()}
            raise LocusTagResolutionError(
                f"{self.label}: {self.resolved} of {self.unique_names} names "
                f"({self.resolved_fraction:.3f}) resolve to {self.assembly_set} locus "
                f"tags, below {min_fraction}; statuses {statuses}"
            )


def resolution_layer(
    genome: BacterialGenome[Any], resolution: GeneNameResolution
) -> str:
    """The resolver layer that decided ``resolution``.

    :data:`LAYER_LOCUS_TAG` for a locus tag (gene or pseudogene), the description of one
    of ``genome.NAME_LAYERS`` (``"old locus tag"``, ``"RefSeq locus tag"``,
    ``"gene symbol"``, ``"gene synonym"``), or :data:`LAYER_NOT_FOUND`. It is read from
    the resolution's ``note``, which ``BacterialGenome.resolve_gene_name`` writes; a note
    of no known form is refused rather than guessed at.
    """
    if resolution.status is GeneNameStatus.RETIRED:
        return LAYER_NOT_FOUND
    note = resolution.note
    if note is None or note.startswith((LAYER_LOCUS_TAG, "valid ")):
        return LAYER_LOCUS_TAG
    for _, layer in genome.NAME_LAYERS:
        if note.startswith(f"{layer} of "):
            return layer
    raise ValueError(
        f"{resolution.input_name!r}: resolution note {note!r} names no layer of "
        f"{type(genome).__name__}"
    )


def reconcile_locus_tags(
    genome: BacterialGenome[Any], names: pd.Series, *, label: str
) -> tuple[pd.Series, LocusTagReconciliation]:
    """Map source gene names to the genome's locus tags, retaining every record.

    Each distinct name goes through ``genome.resolve_gene_name``: locus tag, then
    ``old_locus_tag``, RefSeq locus tag, gene symbol, ``gene_synonym`` (ECK in E. coli),
    else retired. A name that resolves to one locus (current gene, renamed, or a
    pseudogene) is stored as that locus tag, UNLESS another distinct name resolves to the
    same locus: then both are kept as given, so two source strains never merge into one
    record identity. Retired and ambiguous names are kept as given. Nothing is dropped
    for a naming reason.

    Returns the stored names (a new Series aligned to ``names``) and the
    :class:`LocusTagReconciliation`, which is also logged.
    """
    unique = list(names.unique())
    if not unique:
        raise ValueError(f"{label}: no names to reconcile")
    namespace = gene_namespace_of(genome)
    resolutions = {name: genome.resolve_gene_name(name) for name in unique}
    proposed = {
        name: (
            res.systematic_name
            if res.status in _REMAP_STATUSES and res.systematic_name is not None
            else name
        )
        for name, res in resolutions.items()
    }
    proposed_counts = Counter(proposed.values())
    final = {
        name: (name if proposed_counts[prop] > 1 else prop)
        for name, prop in proposed.items()
    }
    statuses = Counter(res.status for res in resolutions.values())
    layers = Counter(resolution_layer(genome, res) for res in resolutions.values())
    layer_order = [
        LAYER_LOCUS_TAG,
        *(d for _, d in genome.NAME_LAYERS),
        LAYER_NOT_FOUND,
    ]
    pattern = LOCUS_TAG_PATTERNS[namespace]
    report = LocusTagReconciliation(
        label=label,
        assembly_set=genome.ASSEMBLY_SET,
        gene_namespace=namespace,
        unique_names=len(resolutions),
        status_histogram={status: statuses[status] for status in GeneNameStatus},
        layer_histogram={layer: layers[layer] for layer in layer_order},
        remapped=sum(1 for name, stored in final.items() if stored != name),
        kept_on_collision=tuple(
            sorted(n for n, prop in proposed.items() if proposed_counts[prop] > 1)
        ),
        retired_kept=tuple(
            sorted(
                n
                for n, res in resolutions.items()
                if res.status is GeneNameStatus.RETIRED
            )
        ),
        ambiguous_kept={
            n: tuple(res.candidates)
            for n, res in sorted(resolutions.items())
            if res.status is GeneNameStatus.AMBIGUOUS
        },
        case_insensitive=tuple(
            sorted(
                n
                for n, res in resolutions.items()
                if res.note is not None and _CASE_INSENSITIVE in res.note
            )
        ),
        outside_namespace=tuple(
            sorted({s for s in final.values() if pattern.match(s) is None})
        ),
    )
    log.info(
        "%s locus-tag reconciliation against %s: %d unique names, statuses %s, layers "
        "%s; %d remapped; %d kept as given on collision %s; %d retired kept %s; %d "
        "ambiguous kept %s; %d outside %s",
        label,
        report.assembly_set,
        report.unique_names,
        {s.value: n for s, n in report.status_histogram.items()},
        report.layer_histogram,
        report.remapped,
        len(report.kept_on_collision),
        list(report.kept_on_collision),
        len(report.retired_kept),
        list(report.retired_kept),
        len(report.ambiguous_kept),
        report.ambiguous_kept,
        len(report.outside_namespace),
        namespace,
    )
    return names.map(final), report


# --------------------------------------------------------------------------- #
# The ECK crosswalk between the two K-12 backgrounds
# --------------------------------------------------------------------------- #
def eck_crosswalk(
    mg1655_genome: EcoliK12MG1655Genome, bw25113_genome: EcoliK12BW25113Genome
) -> EckCrosswalk:
    """The ECK synonym join of the two K-12 GenBank annotations.

    ``pairs`` are the ECK ids carried by exactly one locus in each strain (4,423 on the
    deposited sets); ``numeric_disagreements`` flags the pairs whose b-number and
    ``BW25113_`` number differ (11), which is why no string surgery relates the two
    namespaces. Use it only where a paper reports one background's identifiers for an
    experiment done in the other, and record the mapping on the record as derived.
    """
    return _annotation_crosswalk(mg1655_genome.genbank, bw25113_genome.genbank)


# --------------------------------------------------------------------------- #
# Host-aware genome injection (the build entry points)
# --------------------------------------------------------------------------- #
#: A loader ``__init__`` parameter that receives a bacterial genome, and its host. The
#: yeast ``genome`` parameter keeps receiving ``SCerevisiaeGenome``: injection is by
#: parameter NAME, so a bacterial loader is never handed S288C.
BACTERIAL_GENOME_PARAMETERS: dict[str, BacterialHost] = {
    "ecoli_genome": "ecoli",
    "pputida_genome": "pputida",
}
#: The class attribute in which a bacterial loader states its reference strain.
REFERENCE_STRAIN_ATTRIBUTE = "REFERENCE_STRAIN"
#: The yeast loader parameter (``SCerevisiaeGenome``).
YEAST_GENOME_PARAMETER = "genome"


def declared_reference_strain(dataset_class: type) -> BacterialReferenceStrain:
    """The ``REFERENCE_STRAIN`` a bacterial loader class declares; absent is refused."""
    if not hasattr(dataset_class, REFERENCE_STRAIN_ATTRIBUTE):
        raise TypeError(
            f"{dataset_class.__name__} declares a bacterial genome parameter but no "
            f"{REFERENCE_STRAIN_ATTRIBUTE}; a bacterial loader names its reference "
            f"strain, one of {list(BACTERIAL_ASSEMBLY_SETS)}"
        )
    return _STRAIN.validate_python(getattr(dataset_class, REFERENCE_STRAIN_ATTRIBUTE))


class BacterialGenomeInjector:
    """The bacterial genomes a build hands to the loaders that declare one.

    One instance per build. :meth:`genome_kwargs` inspects a loader's ``__init__``: a
    loader naming ``ecoli_genome`` or ``pputida_genome`` gets the genome of its
    ``REFERENCE_STRAIN`` under that name, built on first request (``bacterial_genome``
    with this build's ``data_root``) and shared by every later loader of that strain. A
    loader naming neither gets nothing, so a yeast-only build constructs no bacterial
    genome and never reads the bacterial tier.
    """

    def __init__(self, data_root: str) -> None:
        """Build genomes under ``data_root``'s default cache roots."""
        self.data_root = data_root
        self.built: dict[
            BacterialReferenceStrain, EcoliK12Genome | PPutidaKT2440Genome
        ] = {}

    def genome_kwargs(
        self, dataset_class: type
    ) -> dict[str, EcoliK12Genome | PPutidaKT2440Genome]:
        """The bacterial genome keyword ``dataset_class.__init__`` declares, if any.

        Refused: a loader naming both bacterial parameters, a bacterial loader (one
        naming a bacterial parameter or stating ``REFERENCE_STRAIN``) that also names
        the yeast ``genome``, and a strain of the other host.
        """
        params = inspect.signature(dataset_class.__init__).parameters  # type: ignore[misc]  # inspecting a class's __init__
        declared = [name for name in BACTERIAL_GENOME_PARAMETERS if name in params]
        bacterial = bool(declared) or hasattr(dataset_class, REFERENCE_STRAIN_ATTRIBUTE)
        if bacterial and YEAST_GENOME_PARAMETER in params:
            raise TypeError(
                f"{dataset_class.__name__} is a bacterial loader that names "
                f"{YEAST_GENOME_PARAMETER!r}, which receives SCerevisiaeGenome; name "
                f"one of {list(BACTERIAL_GENOME_PARAMETERS)} instead"
            )
        if not declared:
            return {}
        if len(declared) > 1:
            raise TypeError(
                f"{dataset_class.__name__} names {declared}; a loader serves one host"
            )
        (name,) = declared
        host = BACTERIAL_GENOME_PARAMETERS[name]
        strain = declared_reference_strain(dataset_class)
        if strain not in HOST_STRAINS[host]:
            raise TypeError(
                f"{dataset_class.__name__} names {name!r} but its "
                f"{REFERENCE_STRAIN_ATTRIBUTE} {strain!r} is not a {host} strain "
                f"{HOST_STRAINS[host]}"
            )
        if strain not in self.built:
            self.built[strain] = bacterial_genome(host, strain, self.data_root)
        return {name: self.built[strain]}
