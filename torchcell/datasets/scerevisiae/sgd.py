# torchcell/datasets/scerevisiae/sgd
# [[torchcell.datasets.scerevisiae.sgd]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/sgd
# Test file: tests/torchcell/datasets/scerevisiae/test_sgd.py
"""SGD phenotype annotations: a provenance record of a compendium that states no medium.

SGD's locus ``phenotype_details`` API (Cherry et al. 1998, citation key
``cherrySGDSaccharomycesGenome1998``) holds 1,329 annotations reading ``mutant_type``
"null", strain "S288C" and phenotype "inviable" across the 6,607 per-gene JSON files
``torchcell.graph.sgd.main_get_all_genes`` wrote, pinned by :data:`SGD_GENES_DIGEST`.
Those annotations were served as ``GeneEssentialitySgdDataset`` until 2026-10-11.

DECISION: NOT LOADED. This module registers no dataset, in the MCF2Chem 2023
(:mod:`torchcell.datasets.ecoli.cai2023`) and CeCaFDB 2015
(:mod:`torchcell.datasets.ecoli.zhang2015`) pattern. ``Environment.media`` is required,
because an inviable call is a growth measurement and a growth measurement happened in
some medium; a source that does not name the medium cannot produce a record, and the
schema is not weakened to admit one. Every clause below is a measurement on the pinned
dump, not a preference.

**1. No annotation states a medium the schema can carry.** Measured by
:func:`condition_inventory` over all 1,329 annotations: 1,299 carry no condition
property at all and 30 carry exactly one, and those 30 are the four pairs of
:data:`SGD_CONDITION_PAIRS`. Twenty-seven read "nitrogen starvation" and one "2%
glucose", which are nutrient edits to an unnamed base medium; one reads "elevated
temperature, 37 °C", which is a temperature. Exactly one annotation names a medium at
all (YIL051C / MMF1 / ``mmf1-Δ``, "YPD with 2% glucose"), and it names no physical
state, while ``Media.state`` is required and validated against solid, liquid and gas.
No annotation anywhere in the dump carries any of "solid", "liquid", "agar", "plate",
"broth" or "culture". So the storable-record count is **0 of 1,329**, and 1 only if the
physical state of that one medium were guessed. The medium of an SGD annotation lives
in the annotation's own reference, which is a per-annotation primary paper: 208 of the
1,329 annotations come from "classical genetics" and 1,121 from a "systematic mutation
set" (:func:`experiment_type_inventory`), spanning hundreds of papers. Loading those
papers is a different act from loading SGD, and it is the loadable path.

**2. What the served store therefore was.** It gave every one of its 1,329 records a
``Media(name="YEPD", state="solid", is_synthetic=False)`` and a ``Temperature(30)``
that SGD never states, under a comment that said so ("for this dataset all meta data is
guessed"). The 30 conditions SGD does state were dropped, so the 18 whole-gene-deletion
annotations measured under nitrogen starvation were stored as if grown in rich medium,
which inverts their meaning.

**3. The allele layer was also wrong, independently of the medium.** Every annotation
names exactly one allele (:func:`allele_inventory`). 1,287 name the whole-gene deletion
SGD writes ``<gene>-Δ``; the other 42 name a point, truncation or internal-deletion
allele (``cdc42-111``, ``sac1Δ114-503``, ``hht2-L61P``), which no schema leaf carries
verbatim (#984). The served store stored all 42 as whole-gene deletions, which made 24
of them exact duplicates of a sibling record across 9 (gene, PubMed id) keys
(:func:`collapse_duplicates`), the largest being SAC1 with ten alleles under one paper.

**4. The calls contradict a served fitness store, and the conditions are why.** 24 of
the genes SGD calls inviable have a viable full-deletion single-mutant fitness in
``SmfCostanzo2016Dataset`` (4,581 viable full deletions). Keeping only whole-gene
deletion alleles removes one (YNL031C, three ``hht2`` point alleles), and of the
remaining 23, **all 16 conditional ones are annotated under nitrogen starvation**
(:data:`CONDITIONAL_DISAGREEMENTS`; 14 of the 16 loci are ATG-named, the other two SAC1
and VPS30), and SGD's own note on those annotations reads "essential for autophagy;
inviable after 3 day nitrogen starvation followed by transfer to rich media plates".
The remaining 7 disagreements are unconditional (:data:`UNCONDITIONAL_DISAGREEMENTS`)
and remain unexplained. A store that discarded the condition could not tell the two
groups apart.

Script and committed results:
``experiments/036-dataset-fixes-before-kg-build/scripts/sgd_synlethdb_findings.py``,
``results/sgd_synlethdb_findings.json``.
Finding: [[torchcell.datasets.scerevisiae.sgd]]. Issues #901, #984.
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from torchcell.data import file_sha256
from torchcell.graph.sgd import data_root
from torchcell.verification.report import Provenance

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
#: Where the SGD per-gene API dump sits under ``$DATA_ROOT``.
SGD_GENES_RELPATH: Final = "data/sgd/genome/genes"

#: sha256 over the sorted ``"<file name> <file sha256>"`` lines of the 6,607 gene JSONs
#: the served build consumed (the #889 verifier's pin of the same dump). Any changed,
#: added or removed file changes it.
SGD_GENES_DIGEST: Final = (
    "1cb4ed59556b1534e28eb79ad21135cdb5a149f78dca8df4924f473e39eef409"
)

#: The pinned dump, as the place every measurement of this module was made.
SGD_DUMP: Final = Provenance(
    source_uri=f"$DATA_ROOT/{SGD_GENES_RELPATH}/<systematic name>.json",
    citation_key="cherrySGDSaccharomycesGenome1998",
    sha256=SGD_GENES_DIGEST,
    method="SGD locus phenotype_details API responses (torchcell.graph.sgd."
    "main_get_all_genes); sha256 over the sorted '<file name> <file sha256>' lines",
    page="phenotype_details[].properties",
)

#: The annotation triple this module reads: an inviable null mutant in the S288C
#: reference background, which is what an essentiality call is.
ANNOTATION_SELECTOR: Final[dict[str, str]] = {
    "mutant_type": "null",
    "strain": "S288C",
    "phenotype": "inviable",
}

#: Every verbatim ``(class_type, note)`` condition pair the pinned dump carries on an
#: annotation of :data:`ANNOTATION_SELECTOR`, and what each one is. None of them is a
#: medium the schema can carry, which is clause 1 of the decision.
SGD_CONDITION_PAIRS: Final[dict[tuple[str, str], str]] = {
    ("media", "nitrogen starvation"): "a nutrient edit to an unnamed base medium",
    ("media", "2% glucose"): "a carbon-source concentration, with no base medium",
    (
        "media",
        "YPD with 2% glucose",
    ): "the only annotation that names a medium; it names no physical state, which "
    "Media requires",
    ("temperature", "elevated temperature, 37 °C"): "a temperature, not a medium",
}

#: Words that would let a physical state be read off a condition rather than guessed.
#: Measured absent from every condition property of the dump.
PHYSICAL_STATE_WORDS: Final[tuple[str, ...]] = (
    "solid",
    "liquid",
    "agar",
    "plate",
    "broth",
    "culture",
)

#: The 7 genes SGD calls inviable with no condition at all that have a viable full
#: deletion in ``SmfCostanzo2016Dataset``. Unexplained, and not explained by a
#: condition; recorded so the disagreement is not lost with the store.
UNCONDITIONAL_DISAGREEMENTS: Final[tuple[str, ...]] = (
    "YHR072W-A",
    "YHR099W",
    "YJL012C",
    "YLR293C",
    "YNL138W-A",
    "YOR167C",
    "YOR182C",
)

#: The 16 genes whose disagreement IS explained by the annotation's condition. All 16
#: are annotated under nitrogen starvation; 14 of the loci are ATG-named, the other two
#: are SAC1 and VPS30.
CONDITIONAL_DISAGREEMENTS: Final[tuple[str, ...]] = (
    "YBL078C",
    "YBR128C",
    "YBR217W",
    "YCR068W",
    "YDL149W",
    "YFR021W",
    "YGL180W",
    "YHR171W",
    "YKL212W",
    "YLL042C",
    "YLR423C",
    "YMR159C",
    "YNR007C",
    "YPL120W",
    "YPL149W",
    "YPR185W",
)

#: Viable full deletions in the served ``SmfCostanzo2016Dataset`` store, the denominator
#: of clause 4.
VIABLE_FULL_DELETIONS_SMF_COSTANZO2016: Final = 4581


class SgdDumpDigestError(ValueError):
    """The SGD per-gene dump under ``$DATA_ROOT`` is not the pinned bytes."""


class SgdAlleleError(ValueError):
    """An annotation does not name exactly one allele."""


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


class ConditionInventory(BaseModel):
    """What the dump states about the growth condition of every annotation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_annotations: int
    n_without_condition: int = Field(
        description="annotations with no condition property"
    )
    pairs: dict[str, int] = Field(
        description="count per verbatim '<class_type>: <note>' condition pair"
    )
    n_naming_a_medium: int = Field(
        description="annotations whose condition names a medium by name"
    )
    n_stating_a_physical_state: int = Field(
        description="annotations whose condition carries a PHYSICAL_STATE_WORDS term, "
        "which Media.state would need"
    )
    n_storable: int = Field(
        description="annotations that could be stored with a sourced Media: a named "
        "medium AND its physical state"
    )


class AlleleInventory(BaseModel):
    """The allele layer: whole-gene deletions, every other allele, and the collapse."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_annotations: int
    n_whole_gene_deletion: int
    n_other_allele: int
    collapse_keys: dict[str, list[str]] = Field(
        description="(gene, PubMed id) keys where storing every allele as a whole-gene "
        "deletion made several annotations one record, and the alleles involved"
    )
    n_collapse_extra_records: int = Field(
        description="records beyond one per collapse key, i.e. the duplicates the "
        "served store held"
    )


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
    its own digest: the measurements here are stated of these bytes, so it is never
    fetched.
    """
    if not osp.isdir(genes_dir):
        raise SgdDumpDigestError(
            f"{genes_dir} does not exist; the SGD per-gene dump pinned at "
            f"{SGD_GENES_DIGEST} is what this module's measurements are stated of"
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


def inviable_null_annotations(
    genes_dir: str, *, progress: bool = True
) -> list[SgdAnnotation]:
    """Every annotation reading ``null`` / ``S288C`` / ``inviable``, in file order.

    Files are read in sorted name order and annotations in the order SGD lists them;
    the systematic name is the file name, as the dump writes one file per locus.
    """
    out: list[SgdAnnotation] = []
    names = sorted(n for n in os.listdir(genes_dir) if n.endswith(".json"))
    for name in tqdm(names, disable=not progress):
        with open(osp.join(genes_dir, name), encoding="utf-8") as handle:
            payload = json.load(handle)
        for detail in payload.get("phenotype_details") or []:
            if not (
                detail["mutant_type"] == ANNOTATION_SELECTOR["mutant_type"]
                and detail["strain"]["display_name"] == ANNOTATION_SELECTOR["strain"]
                and detail["phenotype"]["display_name"]
                == ANNOTATION_SELECTOR["phenotype"]
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


def names_a_medium(conditions: Iterable[tuple[str, str]]) -> bool:
    """True when a condition property names a growth medium, not an edit to one.

    The rule is the ``media`` class plus a medium NAME, which in this dump is the one
    "YPD with 2% glucose" annotation. "nitrogen starvation" and "2% glucose" are the
    ``media`` class too, so the class alone does not answer the question.
    """
    return any(
        kind == "media"
        and text.split()[0].upper() in {"YPD", "YPAD", "YEPD", "SC", "SD"}
        for kind, text in conditions
    )


def states_a_physical_state(conditions: Iterable[tuple[str, str]]) -> bool:
    """True when a condition property carries a word ``Media.state`` could be read from."""
    return any(
        word in text.casefold()
        for _, text in conditions
        for word in PHYSICAL_STATE_WORDS
    )


def condition_inventory(annotations: Iterable[SgdAnnotation]) -> ConditionInventory:
    """Clause 1, measured: what the dump states about each annotation's condition.

    ``n_storable`` is the count a loader could honestly store: a named medium AND the
    physical state ``Media`` requires. It is the number the decision rests on.
    """
    rows = list(annotations)
    pairs = Counter(f"{kind}: {text}" for row in rows for kind, text in row.conditions)
    return ConditionInventory(
        n_annotations=len(rows),
        n_without_condition=sum(1 for row in rows if not row.conditions),
        pairs=dict(sorted(pairs.items())),
        n_naming_a_medium=sum(1 for row in rows if names_a_medium(row.conditions)),
        n_stating_a_physical_state=sum(
            1 for row in rows if states_a_physical_state(row.conditions)
        ),
        n_storable=sum(
            1
            for row in rows
            if names_a_medium(row.conditions)
            and states_a_physical_state(row.conditions)
        ),
    )


def experiment_type_inventory(annotations: Iterable[SgdAnnotation]) -> dict[str, int]:
    """Count per SGD ``experiment`` display name, the shape of the primary layer."""
    return dict(sorted(Counter(row.experiment for row in annotations).items()))


def collapse_duplicates(
    annotations: Iterable[SgdAnnotation],
) -> tuple[dict[str, list[str]], int]:
    """The duplicates that storing every allele as a whole-gene deletion produced.

    Returns the ``(gene, PubMed id)`` keys holding more than one annotation and the
    alleles under each, plus the number of records beyond one per key, which is what the
    served store carried twice.
    """
    by_key: dict[str, list[str]] = {}
    for row in annotations:
        by_key.setdefault(f"{row.systematic_name} PMID {row.pubmed_id}", []).append(
            row.allele
        )
    repeated = {key: alleles for key, alleles in by_key.items() if len(alleles) > 1}
    extra = sum(len(alleles) - 1 for alleles in repeated.values())
    return dict(sorted(repeated.items())), extra


def allele_inventory(annotations: Iterable[SgdAnnotation]) -> AlleleInventory:
    """Clause 3, measured: the allele layer and the records the collapse duplicated."""
    rows = list(annotations)
    whole = [
        row
        for row in rows
        if is_whole_gene_deletion(row.allele, row.locus_display_name)
    ]
    keys, extra = collapse_duplicates(rows)
    return AlleleInventory(
        n_annotations=len(rows),
        n_whole_gene_deletion=len(whole),
        n_other_allele=len(rows) - len(whole),
        collapse_keys=keys,
        n_collapse_extra_records=extra,
    )


def main() -> None:
    """Re-measure every clause of the decision from the pinned dump."""
    from dotenv import load_dotenv

    load_dotenv()
    genes_dir = osp.join(data_root(), SGD_GENES_RELPATH)
    verify_sgd_dump(genes_dir)
    annotations = inviable_null_annotations(genes_dir)
    print(condition_inventory(annotations).model_dump_json(indent=2))
    print(allele_inventory(annotations).model_dump_json(indent=2))
    print(experiment_type_inventory(annotations))


if __name__ == "__main__":
    main()
