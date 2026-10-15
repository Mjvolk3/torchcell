# torchcell/datasets/scerevisiae/sgd
# [[torchcell.datasets.scerevisiae.sgd]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/sgd
# Test file: tests/torchcell/datasets/scerevisiae/test_sgd.py

"""SGD-derived gene essentiality dataset for S. cerevisiae.

The source is SGD's per-gene API dump, ``$DATA_ROOT/data/sgd/genome/genes/<ORF>.json``
(the locus ``phenotype_details`` responses ``torchcell.graph.sgd.main_get_all_genes``
wrote, 6,607 files), pinned by one sha256 over the sorted ``"<file name> <file
sha256>"`` lines (:data:`SGD_GENES_DIGEST`). The build reads those files, not a graph
built from them, and refuses bytes off the pin.

One record per SGD annotation with ``mutant_type`` "null", strain "S288C" and phenotype
"inviable" (1,329 in the pinned dump), under two rules (#901):

- **Allele.** Every annotation names one allele and every annotation is a record
  (#1045). When the allele is the whole-gene deletion SGD writes ``<gene>-Δ`` the
  genotype is a deletion of the gene (``SgaKanMxDeletionPerturbation``); the other 42
  annotations name a point, truncation or internal-deletion allele (``cdc42-111``,
  ``sac1Δ114-503``, ``hht2-L61P``) and are stored on ``NamedAllelePerturbation`` with
  the allele name verbatim, which is what keeps two alleles of one gene two records.
  Before #1045 those 42 were ledgered and never stored, because storing them as a
  whole-gene deletion made 24 records exact duplicates of a sibling. The ledger
  ``preprocess/dropped_records.json`` (:class:`SgdDropLog`) now records a build that
  drops nothing.
- **Condition.** An annotation's condition properties are carried, never dropped: the
  pinned dump holds four (class, text) pairs, each mapped by :data:`SGD_CONDITIONS` and
  any other refused by name. "nitrogen starvation" and "2% glucose" are nutrient edits
  (``EnvironmentPhysicalPerturbation``), "elevated temperature, 37 °C" is the
  temperature, and "YPD with 2% glucose" names a medium without its physical state, so
  it is stored by its stated name on the library YPD base with only ``state`` a typed
  gap (:data:`SGD_STATELESS_YPD`).
- **Medium.** ``Environment.media`` is required, so an annotation that names no medium
  still stores a ``Media``, whose ``name``, ``state`` and ``is_synthetic`` are typed
  gaps (``not_carried_by_curation``) quoting what the annotation carries in place of
  one (:func:`sgd_unstated_medium`); the temperature it does not state is a typed gap
  on the environment. The old loader guessed YEPD solid at 30 C for every record.
- **Evidence (#924).** Every phenotype carries a ``GeneEssentialityEvidence`` of type
  ``curated_null_phenotype`` (class ``curated_annotation``): the annotation fields a
  call reads and the call's definition, both anchored to SGD's own description of a
  phenotype annotation (Engel 2022, :data:`SGD_SCORING_RULE`,
  :data:`SGD_CALL_DEFINITION`); the allele rule above is this loader's, not SGD's, so
  the evidence does not carry it; and
  per record the annotation's SGD experiment type verbatim (1,121 "systematic mutation
  set", 166 "classical genetics" of the 1,287 stored in the pinned dump). The mirrored
  SGD literature does not define the "inviable" term itself.
- **The annotation's reference is on the evidence (lane V, 2026-10-15).** SGD curates
  one gene's inviable null phenotype from more than one paper: 17 (gene, allele,
  condition) groups of the pinned dump carry two or three annotations with different
  ``pubmed_id`` (MRS6 from 7750142, 8132658 and 7882424; USO1, SRP14, UBC1 and 11
  others from Giaever's 12140549 and Dowell's 20413493). The PubMed id was only on the
  record's ``Publication``, which the ``Experiment`` dump does not include, so those 35
  annotations dumped to 17 experiments and the graph held one ``Experiment`` node per
  group with two or three ``Publication`` nodes pointing at it: 18 curated annotations
  were in the LMDB and absent from the served graph. ``GeneEssentialityEvidence.
  curated_reference_pubmed_id`` carries it, because what a curated call rests on is one
  annotation of one report, and that is part of what the call IS.
"""

import hashlib
import json
import logging
import os
import os.path as osp
import pickle
import random
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import lmdb
import pandas as pd
from Bio import Entrez
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from torchcell.data import ExperimentDataset, Organism, file_sha256, post_process
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.schema import (
    Concentration,
    ConcentrationUnit,
    Environment,
    EnvironmentPhysicalPerturbation,
    EssentialityEvidenceClass,
    EssentialityEvidenceType,
    EssentialityStatement,
    EssentialityStatementRole,
    GeneEssentialityEvidence,
    GeneEssentialityExperiment,
    GeneEssentialityExperimentReference,
    GeneEssentialityPhenotype,
    Genotype,
    Media,
    NamedAllelePerturbation,
    PhysicalFactor,
    Publication,
    ReferenceGenome,
    SgaKanMxDeletionPerturbation,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.graph.sgd import data_root
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

#: Where the SGD per-gene API dump sits under ``$DATA_ROOT``.
SGD_GENES_RELPATH = "data/sgd/genome/genes"
#: sha256 over the sorted ``"<file name> <file sha256>"`` lines of the 6,607 gene JSONs
#: the served build consumed (the #889 verifier's pin of the same dump). Any changed,
#: added or removed file changes it.
SGD_GENES_DIGEST = "1cb4ed59556b1534e28eb79ad21135cdb5a149f78dca8df4924f473e39eef409"

#: The pinned dump, as the place every gap of this dataset looked in.
SGD_DUMP = Provenance(
    source_uri=f"$DATA_ROOT/{SGD_GENES_RELPATH}/<systematic name>.json",
    citation_key="cherrySGDSaccharomycesGenome1998",
    sha256=SGD_GENES_DIGEST,
    method="SGD locus phenotype_details API responses (torchcell.graph.sgd."
    "main_get_all_genes); sha256 over the sorted '<file name> <file sha256>' lines",
    page="phenotype_details[].properties",
)

# --- begin #924: the curated-annotation evidence on every record ------------- #
#: SGD's own description of a phenotype annotation (Engel 2022, the literature mirror).
SGD_CURATION_CITATION_KEY = "engelNewDataCollaborations2022"
SGD_CURATION_PAPER = "paper.md"
SGD_CURATION_PAPER_SHA256 = (
    "ef55aeb36bb3d86d1dcff9f5a053ca4e6c11812b6a4bf8a465196c2d28877152"
)
_ANNOTATION_QUOTE = (
    "The phenotype annotations for an allele include an observable feature (e.g., "
    "“cell shape”), a qualifier (e.g., “abnormal”), a mutant type "
    "(e.g., null), strain background, and a reference."
)


def _engel(value: Any, quote: str, *, note: str) -> SourcedValue:
    """A value bound to a verbatim quote of the pinned Engel 2022 OCR."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=SGD_CURATION_PAPER,
            citation_key=SGD_CURATION_CITATION_KEY,
            sha256=SGD_CURATION_PAPER_SHA256,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
            page="New allele pages",
        ),
    )


SGD_SCORING_RULE = _engel(
    {"observable": "inviable", "mutant_type": "null", "strain_background": "S288C"},
    _ANNOTATION_QUOTE,
    note="the annotation fields a stored call reads: the observable 'inviable', the "
    "mutant type 'null' and the strain background 'S288C'",
)
SGD_CALL_DEFINITION = _engel(
    "SGD annotates the null allele's observable feature as 'inviable'",
    _ANNOTATION_QUOTE,
    note="is_essential=True is the curated observable of a null allele; the mirrored "
    "SGD literature does not define the 'inviable' term itself",
)
SGD_EXPERIMENT_TYPES = _engel(
    "classical genetics or high-throughput (large-scale survey, systematic mutation "
    "set)",
    "In addition, annotations are classified as classical genetics or highthroughput "
    "(e.g., large-scale survey, systematic mutation set).",
    note="the definition of curated_experiment_type, the annotation's 'experiment' "
    "display name",
)
#: Every sourced statement of this loader, audited against the literature mirror.
SOURCED_VALUES: dict[str, SourcedValue] = {
    "scoring_rule": SGD_SCORING_RULE,
    "call_definition": SGD_CALL_DEFINITION,
    "experiment_types": SGD_EXPERIMENT_TYPES,
}


#: The source's statements every record's evidence carries, by role (#924).
EVIDENCE_STATEMENTS: tuple[EssentialityStatement, ...] = (
    EssentialityStatement(
        role=EssentialityStatementRole.scoring_rule, provenance=SGD_SCORING_RULE
    ),
    EssentialityStatement(
        role=EssentialityStatementRole.call_definition, provenance=SGD_CALL_DEFINITION
    ),
    EssentialityStatement(
        role=EssentialityStatementRole.statistic_definition,
        provenance=SGD_EXPERIMENT_TYPES,
    ),
)


def sgd_evidence(
    experiment_type: str | None = None, pubmed_id: str | None = None
) -> GeneEssentialityEvidence:
    """The curated-annotation evidence (#924), with the annotation's SGD experiment
    type and the PubMed id of the reference it was curated from when given; the
    reference (both ``None``) carries the rule and definition only.

    The PubMed id is on the evidence because a curated call rests on one report and
    which report it is is part of what the call IS: 17 (gene, allele, condition) groups
    of the pinned dump are curated from two or three different papers each, and without
    it those 35 annotations dumped to 17 experiments (lane V, 2026-10-15).
    """
    return GeneEssentialityEvidence(
        evidence_class=EssentialityEvidenceClass.curated_annotation,
        evidence_type=EssentialityEvidenceType.curated_null_phenotype,
        statements=list(EVIDENCE_STATEMENTS),
        curated_experiment_type=experiment_type,
        curated_reference_pubmed_id=pubmed_id,
    )


# --- end #924 ---------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Environment: the medium and temperature SGD does not state are typed gaps (#901)
# --------------------------------------------------------------------------- #
#: What every SGD medium gap says first: an annotation is a phenotype on an allele
#: under zero or one condition properties, and none of them names a growth medium.
_SGD_SILENCE = (
    "the SGD phenotype annotation states a phenotype ('inviable', mutant_type "
    "'null', strain 'S288C') and an allele ('<gene>-Δ') but no growth medium: "
)
#: Where the medium would be, so the gap reads as a worklist entry.
_SGD_RESOLVE = (
    "; the medium, if stated, is in the annotation's own reference "
    "(Publication.pubmed_id)"
)


# --- begin #901/#902/#903: a typed gap on the medium --------------------------- #
def sgd_unstated_medium(silence: str) -> Media:
    """The ``Media`` of an SGD annotation that names none: every label a typed gap.

    ``silence`` says what the annotation carries in place of a medium (no condition
    property, or one that is a nutrient edit or a temperature). The three gaps share
    one reason (``not_carried_by_curation``: SGD curates the primary paper and does not
    carry its medium) and one place looked (the pinned dump), so every such record
    stores one gapped medium, never a guessed YEPD.
    """
    note = _SGD_SILENCE + silence + _SGD_RESOLVE
    return Media(
        name=None,
        state=None,
        is_synthetic=None,
        provenance_gaps=[
            ProvenanceGap(
                field=field,
                reason=ProvenanceGapReason.not_carried_by_curation,
                looked_in=SGD_DUMP,
                note=note,
            )
            for field in ("name", "state", "is_synthetic")
        ],
    )


#: An annotation with no condition property (1,299 of the 1,329 in the pinned dump).
SGD_UNSTATED_MEDIUM = sgd_unstated_medium(
    "phenotype_details[].properties carries no condition property"
)
#: "YPD with 2% glucose" names a medium, so its name is stated, verbatim, and its base
#: is the library's YPD (yeast extract, peptone, 2% dextrose: complex by construction,
#: hence ``is_synthetic`` False). SGD states no physical state, and ``Media.state`` is
#: solid, liquid or gas, so the state alone is the gap: neither plate nor broth is
#: guessed.
SGD_STATELESS_YPD = Media(
    name="YPD with 2% glucose",
    state=None,
    is_synthetic=False,
    base_medium="YPD",
    provenance_gaps=[
        ProvenanceGap(
            field="state",
            reason=ProvenanceGapReason.not_carried_by_curation,
            looked_in=SGD_DUMP,
            note="SGD states the medium as 'YPD with 2% glucose' (condition property, "
            "class_type 'media') without its physical state; the state, if stated, "
            "is in the annotation's own reference (Publication.pubmed_id)",
        )
    ],
)
# --- end #901/#902/#903 -------------------------------------------------------- #

SGD_TEMPERATURE_GAP = ProvenanceGap(
    field="temperature",
    reason=ProvenanceGapReason.not_carried_by_curation,
    looked_in=SGD_DUMP,
    note="the SGD phenotype annotation carries no temperature condition property",
)

SGD_ENVIRONMENT = Environment(
    media=SGD_UNSTATED_MEDIUM, temperature=None, provenance_gaps=[SGD_TEMPERATURE_GAP]
)
"""An annotation with no condition property: neither medium nor temperature is stated."""

#: The verbatim (class_type, note) condition pairs of the pinned dump, and the
#: environment each one is. Any other pair refuses the build (:class:`SgdConditionError`).
SGD_CONDITIONS: dict[tuple[str, str], Environment] = {
    ("media", "nitrogen starvation"): Environment(
        media=sgd_unstated_medium(
            "its one condition property (class_type 'media') reads 'nitrogen "
            "starvation', a nutrient edit to a base medium it does not name"
        ),
        temperature=None,
        perturbations=[
            EnvironmentPhysicalPerturbation(
                factor=PhysicalFactor.nitrogen_source, description="nitrogen starvation"
            )
        ],
        provenance_gaps=[SGD_TEMPERATURE_GAP],
    ),
    ("media", "2% glucose"): Environment(
        media=sgd_unstated_medium(
            "its one condition property (class_type 'media') reads '2% glucose', a "
            "carbon-source edit to a base medium it does not name"
        ),
        temperature=None,
        perturbations=[
            EnvironmentPhysicalPerturbation(
                factor=PhysicalFactor.carbon_source,
                description="2% glucose",
                magnitude=Concentration(value=2.0, unit=ConcentrationUnit.percent),
                agent=resolved_compound("glucose"),
            )
        ],
        provenance_gaps=[SGD_TEMPERATURE_GAP],
    ),
    ("media", "YPD with 2% glucose"): Environment(
        media=SGD_STATELESS_YPD, temperature=None, provenance_gaps=[SGD_TEMPERATURE_GAP]
    ),
    ("temperature", "elevated temperature, 37 °C"): Environment(
        media=sgd_unstated_medium(
            "its one condition property is a temperature ('elevated temperature, "
            "37 °C', class_type 'temperature'), which names no medium"
        ),
        temperature=Temperature(value=37.0),
    ),
}

#: The retention rule of this dataset (#1045): nothing is dropped.
RULE_NONE = "none"
_RULE_DESCRIPTION = (
    "every inviable null S288C annotation is a record: the whole-gene deletion "
    "'<gene>-Δ' on SgaKanMxDeletionPerturbation, any other allele (a point, truncation "
    "or internal-deletion allele) on NamedAllelePerturbation under its name verbatim"
)


class SgdDumpDigestError(ValueError):
    """The SGD per-gene dump under ``$DATA_ROOT`` is not the pinned bytes."""


class SgdConditionError(ValueError):
    """An annotation carries a condition property outside :data:`SGD_CONDITIONS`."""


class SgdAlleleError(ValueError):
    """An annotation does not name exactly one allele."""


class DuplicateSgdAnnotationError(ValueError):
    """Two annotations are one (gene, allele, condition, PubMed id) record."""


class SgdAnnotation(BaseModel):
    """One inviable null S288C phenotype annotation, as SGD's API states it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sgd_annotation_id: int
    systematic_name: str
    locus_display_name: str
    allele: str
    pubmed_id: str
    experiment: str
    conditions: tuple[tuple[str, str], ...]
    note: str | None


class SgdDroppedAnnotation(BaseModel):
    """One annotation the allele rule left out of the store, verbatim."""

    model_config = ConfigDict(extra="forbid")

    sgd_annotation_id: int
    systematic_name: str
    locus_display_name: str
    allele: str
    pubmed_id: str
    conditions: list[str]
    note: str | None


class SgdDropLog(BaseModel):
    """The ledger of a build: what the dump holds, what was stored on which leaf.

    ``rows`` lists what a rule dropped and is empty since #1045; the verifier reads it
    to compare the store with the release, less the ledgered annotations.
    """

    model_config = ConfigDict(extra="forbid")

    dataset: str
    source_annotations: int
    kept_records: int
    dropped_records: int
    n_whole_gene_deletion: int
    n_named_allele: int
    sgd_genes_digest: str
    rule: str
    description: str
    rows: list[SgdDroppedAnnotation]


def sgd_genes_digest(genes_dir: str) -> str:
    """One sha256 over every ``<gene>.json`` in ``genes_dir`` (see the module pin)."""
    lines = [
        f"{name} {file_sha256(osp.join(genes_dir, name))}"
        for name in sorted(os.listdir(genes_dir))
        if name.endswith(".json")
    ]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def verify_sgd_dump(genes_dir: str) -> None:
    """Refuse a missing dump or bytes off :data:`SGD_GENES_DIGEST`.

    A fresh download (``torchcell.graph.sgd.main_get_all_genes``) is a new release with
    its own digest: it needs a new pin and a new store, so it is never fetched here.
    """
    if not osp.isdir(genes_dir):
        raise SgdDumpDigestError(
            f"{genes_dir} does not exist; the SGD per-gene dump pinned at "
            f"{SGD_GENES_DIGEST} is the source of this dataset"
        )
    observed = sgd_genes_digest(genes_dir)
    if observed != SGD_GENES_DIGEST:
        raise SgdDumpDigestError(
            f"{genes_dir}: digest {observed} is not the pinned {SGD_GENES_DIGEST}"
        )


def _allele(detail: Mapping[str, Any], where: str) -> str:
    alleles = [
        prop["bioitem"]["display_name"]
        for prop in detail["properties"]
        if prop["class_type"] == "BIOITEM" and prop["role"] == "Allele"
    ]
    if len(alleles) != 1:
        raise SgdAlleleError(f"{where}: {len(alleles)} alleles {alleles}")
    return str(alleles[0])


def inviable_null_annotations(genes_dir: str) -> list[SgdAnnotation]:
    """Every annotation reading ``null`` / ``S288C`` / ``inviable``, in file order.

    Files are read in sorted name order and annotations in the order SGD lists them;
    the systematic name is the file name, as the dump writes one file per locus.
    """
    out: list[SgdAnnotation] = []
    names = sorted(n for n in os.listdir(genes_dir) if n.endswith(".json"))
    for name in tqdm(names):
        with open(osp.join(genes_dir, name), encoding="utf-8") as handle:
            payload = json.load(handle)
        for detail in payload.get("phenotype_details") or []:
            if not (
                detail["mutant_type"] == "null"
                and detail["strain"]["display_name"] == "S288C"
                and detail["phenotype"]["display_name"] == "inviable"
            ):
                continue
            where = f"{name} annotation {detail['id']}"
            out.append(
                SgdAnnotation(
                    sgd_annotation_id=int(detail["id"]),
                    systematic_name=name[: -len(".json")],
                    locus_display_name=detail["locus"]["display_name"],
                    allele=_allele(detail, where),
                    pubmed_id=str(detail["reference"]["pubmed_id"]),
                    experiment=detail["experiment"]["display_name"],
                    conditions=tuple(
                        (str(prop["class_type"]), str(prop["note"]))
                        for prop in detail["properties"]
                        if prop["class_type"] != "BIOITEM"
                    ),
                    note=detail["note"] or None,
                )
            )
    return out


def is_whole_gene_deletion(allele: str, locus_display_name: str) -> bool:
    """True when ``allele`` is SGD's name for the deletion of the whole gene.

    SGD writes it ``<gene>-Δ`` (U+0394); one annotation of the pinned dump spells the
    sign U+2206 (``yil102c-A-∆``), which is the same name, so both signs and letter case
    are folded. Anything after the sign (``sac1Δ114-503``, ``cdc24p-∆CH``) or any other
    form (``cdc42-111``, ``Nog1-(1-426)``) is a different allele.
    """
    folded = allele.replace("∆", "Δ").casefold()
    return folded == f"{locus_display_name}-Δ".casefold()


def condition_environment(annotation: SgdAnnotation) -> Environment:
    """The environment an annotation's condition properties state (module docstring)."""
    if not annotation.conditions:
        return SGD_ENVIRONMENT
    if len(annotation.conditions) != 1 or (
        annotation.conditions[0] not in SGD_CONDITIONS
    ):
        raise SgdConditionError(
            f"{annotation.systematic_name} annotation {annotation.sgd_annotation_id}: "
            f"conditions {list(annotation.conditions)} are not in SGD_CONDITIONS"
        )
    return SGD_CONDITIONS[annotation.conditions[0]]


def split_by_allele(
    annotations: Iterable[SgdAnnotation],
) -> tuple[list[SgdAnnotation], list[SgdAnnotation]]:
    """``(whole-gene deletions, named alleles)``; both halves are stored (#1045)."""
    whole: list[SgdAnnotation] = []
    named: list[SgdAnnotation] = []
    for annotation in annotations:
        if is_whole_gene_deletion(annotation.allele, annotation.locus_display_name):
            whole.append(annotation)
        else:
            named.append(annotation)
    return whole, named


def refuse_duplicate_annotations(annotations: Iterable[SgdAnnotation]) -> None:
    """Raise when two annotations share gene, allele, condition and PubMed id.

    Since the evidence carries ``curated_reference_pubmed_id`` this key is a subset of
    what the stored experiment dumps, so a build that passes here has one content id
    per record: the dump holds the gene, the allele, the condition's environment, the
    SGD experiment type and the PubMed id, and no two annotations agree on all five.
    Measured on the pinned dump: 1,329 annotations, 1,329 distinct dumps (lane V).
    """
    keys = Counter(
        (a.systematic_name, a.allele, a.conditions, a.pubmed_id) for a in annotations
    )
    repeated = sorted(str(key) for key, n in keys.items() if n > 1)
    if repeated:
        raise DuplicateSgdAnnotationError(
            f"{len(repeated)} (gene, allele, conditions, PubMed id) keys repeat: "
            f"{repeated}"
        )


def drop_log(
    dataset_name: str,
    n_source: int,
    whole: list[SgdAnnotation],
    named: list[SgdAnnotation],
) -> SgdDropLog:
    """The ledger of one build: every annotation a record, none dropped."""
    return SgdDropLog(
        dataset=dataset_name,
        source_annotations=n_source,
        kept_records=len(whole) + len(named),
        dropped_records=0,
        n_whole_gene_deletion=len(whole),
        n_named_allele=len(named),
        sgd_genes_digest=SGD_GENES_DIGEST,
        rule=RULE_NONE,
        description=_RULE_DESCRIPTION,
        rows=[],
    )


def sgd_perturbation(
    annotation: SgdAnnotation,
) -> SgaKanMxDeletionPerturbation | NamedAllelePerturbation:
    """The perturbation an annotation states (#1045).

    The whole-gene deletion ``<gene>-Δ`` is the deletion of the gene; any other allele
    is stored under its name, verbatim, on the gene's systematic name.
    """
    if is_whole_gene_deletion(annotation.allele, annotation.locus_display_name):
        return SgaKanMxDeletionPerturbation(
            systematic_gene_name=annotation.systematic_name,
            perturbed_gene_name=annotation.locus_display_name,
            strain_id="S288C",
        )
    return NamedAllelePerturbation(
        systematic_gene_name=annotation.systematic_name,
        perturbed_gene_name=annotation.locus_display_name,
        allele_name=annotation.allele,
        strain_id="S288C",
    )


def get_publication_info(pubmed_id: str) -> dict[str, str | None] | None:
    """Fetch publication metadata (PubMed URL, DOI) for a PubMed ID via Entrez."""
    Entrez.email = "mvjolk3@illinois.edu"  # type: ignore[assignment]  # Bio.Entrez.email stub typed as None
    max_retries = 5
    base_delay = 1  # seconds

    for attempt in range(max_retries):
        try:
            handle = Entrez.efetch(  # type: ignore[no-untyped-call]  # Bio.Entrez has no stubs
                db="pubmed", id=pubmed_id, rettype="xml", retmode="text"
            )
            records = Entrez.read(handle)  # type: ignore[no-untyped-call]  # Bio.Entrez has no stubs
            handle.close()

            article = records["PubmedArticle"][0]["MedlineCitation"]["Article"]

            info: dict[str, str | None] = {
                "pubmed_id": pubmed_id,
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/",
                "doi": None,
                "doi_url": None,
            }

            # Try to get DOI from ELocationID
            doi = next(
                (
                    eid
                    for eid in article.get("ELocationID", [])
                    if eid.attributes["EIdType"] == "doi"
                ),
                None,
            )

            # If DOI not found in ELocationID, try ArticleId
            if not doi:
                article_id_list = records["PubmedArticle"][0]["PubmedData"][
                    "ArticleIdList"
                ]
                doi = next(
                    (
                        article_id
                        for article_id in article_id_list
                        if article_id.attributes["IdType"] == "doi"
                    ),
                    None,
                )

            if doi:
                info["doi"] = str(doi)
                info["doi_url"] = f"https://doi.org/{info['doi']}"

            return info

        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                print(
                    f"Attempt {attempt + 1} failed. Retrying in {delay:.2f} seconds..."
                )
                time.sleep(delay)
                continue
            else:
                print(f"Error fetching info for PubMed ID {pubmed_id}: {str(e)}")
                return None

    return None


@register_dataset
class GeneEssentialitySgdDataset(ExperimentDataset):
    """Gene essentiality experiments built from SGD inviable null phenotypes."""

    organism = Organism.scerevisiae

    def __init__(
        self,
        root: str = "data/torchcell/gene_essentiality_sgd",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the experiment dataset (the source is the pinned SGD dump)."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[GeneEssentialityExperiment]:
        """Return the experiment model class for this dataset."""
        return GeneEssentialityExperiment

    @property
    def reference_class(self) -> type[GeneEssentialityExperimentReference]:
        """Return the experiment reference model class for this dataset."""
        return GeneEssentialityExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """No files in ``raw/``: the source is the pinned dump under ``$DATA_ROOT``."""
        return []

    def download(self) -> None:
        """Do nothing; the pinned dump is verified in :meth:`process`, never fetched."""
        pass

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Return the DataFrame unchanged (no raw preprocessing needed)."""
        return df

    @post_process
    def process(self) -> None:
        """Verify the SGD dump, apply the allele rule, and write the LMDB and ledger."""
        log.info("Processing SGD Gene Essentiality Data...")
        genes_dir = osp.join(data_root(), SGD_GENES_RELPATH)
        verify_sgd_dump(genes_dir)
        annotations = inviable_null_annotations(genes_dir)
        whole, named = split_by_allele(annotations)
        refuse_duplicate_annotations(annotations)
        environments = [condition_environment(a) for a in annotations]

        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        ledger = drop_log(self.name, len(annotations), whole, named)
        with open(osp.join(self.preprocess_dir, "dropped_records.json"), "w") as f:
            f.write(ledger.model_dump_json(indent=2))

        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e12))
        with env.begin(write=True) as txn:
            for index, (annotation, environment) in enumerate(
                zip(annotations, environments, strict=True)
            ):
                experiment, reference, publication = self.create_experiment(
                    self.name, annotation, environment
                )
                serialized_data = pickle.dumps(
                    {
                        "experiment": experiment.model_dump(),
                        "reference": reference.model_dump(),
                        "publication": publication.model_dump(),
                    }
                )
                txn.put(f"{index}".encode(), serialized_data)
        env.close()
        log.info(
            "%s: %d of %d annotations stored (%d whole-gene deletions, %d named "
            "alleles), %d ledgered",
            self.name,
            ledger.kept_records,
            ledger.source_annotations,
            ledger.n_whole_gene_deletion,
            ledger.n_named_allele,
            ledger.dropped_records,
        )

    @staticmethod
    def create_experiment(  # type: ignore[override]  # dataset-specific signature
        dataset_name: str, annotation: SgdAnnotation, environment: Environment
    ) -> tuple[
        GeneEssentialityExperiment, GeneEssentialityExperimentReference, Publication
    ]:
        """Build the experiment, reference, and publication for one annotation."""
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain="S288C",
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        )

        genotype = Genotype(perturbations=[sgd_perturbation(annotation)])

        phenotype = GeneEssentialityPhenotype(
            is_essential=True,
            evidence=sgd_evidence(annotation.experiment, annotation.pubmed_id),
        )
        phenotype_reference = GeneEssentialityPhenotype(
            is_essential=False, evidence=sgd_evidence()
        )

        pub_info = get_publication_info(annotation.pubmed_id)

        if pub_info is None:
            raise ValueError(
                "Unable to retrieve publication information for PubMed ID: "
                f"{annotation.pubmed_id}"
            )

        experiment = GeneEssentialityExperiment(
            dataset_name=dataset_name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )

        reference = GeneEssentialityExperimentReference(
            dataset_name=dataset_name,
            genome_reference=genome_reference,
            environment_reference=environment,
            phenotype_reference=phenotype_reference,
        )

        publication = Publication(
            pubmed_id=pub_info["pubmed_id"],
            pubmed_url=pub_info["pubmed_url"],
            doi=pub_info["doi"],
            doi_url=pub_info["doi_url"],
        )

        return experiment, reference, publication


def main() -> None:
    """Build and inspect the SGD gene essentiality dataset under ``$DATA_ROOT``."""
    from dotenv import load_dotenv

    load_dotenv()
    dataset = GeneEssentialitySgdDataset(
        root=osp.join(data_root(), "data/torchcell/gene_essentiality_sgd")
    )
    print(dataset)


if __name__ == "__main__":
    main()
