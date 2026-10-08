# torchcell/datasets/ecoli/butland2008
# [[torchcell.datasets.ecoli.butland2008]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/butland2008
# Test file: tests/torchcell/datasets/ecoli/test_butland2008.py
"""Butland 2008: the UNFILTERED eSGA digenic interaction matrix of E. coli K-12.

Butland, Babu, Diaz-Mejia et al. 2008 (Nature Methods 5:789-795,
doi:10.1038/nmeth.1239, citation key ``butlandESGAColiSynthetic2008``) ran 39
genome-wide conjugation screens, crossing one Hfr Cavalli query deletion each into an
arrayed F- recipient collection of 8,073 strains, and released the resulting
interaction (S) scores **in full**. Supplementary Table 4's fourth sheet is a
39 x 8,073 matrix whose banner reads, verbatim, "without any filtering parameters".

RECORD = one ``BacterialGeneInteractionExperiment`` per CELL of that matrix: a two-gene
``Genotype`` of ``BacterialDeletionPerturbation`` leaves (the ``cat``-marked query and
the ``kan``-marked recipient isolate) and a ``GeneInteractionPhenotype`` holding that
cell's S score verbatim.

WHY THIS IS A LOADER AND NOT A PROVENANCE RECORD. Two earlier passes concluded the
opposite and both were wrong, so the measurement is restated here and re-run at build
time. Issue #794 recommended a provenance record on the premise that Babu 2014 subsumes
these screens; Babu's Table S1 does attribute 39 of its 163 donors to ``Butland et al.``
and the served ``GeneInteractionBabu2014Dataset`` does tag 727 records
``screen_id="Butland et al."``. But 727 is **0.23 percent** of the 314,847 S scores this
release prints, so Babu publishes the high-confidence tail of a re-analysis rather than a
superset: it releases 1,129 rows over these same 39 donors and omits 490 of Butland's own
1,270 high-confidence gene pairs, 321 of them non-essential. Where the two overlap the
number is the SAME number -- 793 of 873 same-orientation rows carry an identical score --
so loading this release adds records, never a competing score definition. The earlier
"blocked on a paywalled retrieval" status was also stale: the paper and all five SI files
are mirrored. The measurement is
``experiments/036-dataset-fixes-before-kg-build/scripts/bacteria_subsumed_rows.py`` and
``$DATA_ROOT/torchcell-raw/butlandESGAColiSynthetic2008/subsumption_record.json``.

SCOPE, AND WHY IT IS THE WHOLE MATRIX. The cell is the release's own unit of
observation, every one of the 314,847 cells is populated, and every cell has a home on
the existing schema: a signed float goes to ``GeneInteractionPhenotype.gene_interaction``
(unclamped, which is why ``FitnessPhenotype`` is wrong here), the two loci to two
bacterial deletion leaves, and the "Strain Versions" token to the recipient leaf's
``StrainConstruction.batch``. Supplementary Table 3, the |Z|>=4 high-confidence set, is a
measured PROPER SUBSET: all 1,379 of its rows are located at their (query, recipient,
isolate) cell of sheet 4 and all 1,379 S scores are identical, so loading Table 3 instead
would store a selection-biased tail (730 of its 799 non-essential pairs are aggravating)
and loading both would duplicate every row. The narrower scopes were each considered and
rejected: "the 321 pairs Babu omits" is defined by another paper's editorial filter
rather than by anything this release states, and "all 799 non-essential high-confidence
pairs" keeps a sourced |Z| threshold but discards the null distribution, which is the one
thing no other bacterial interaction store carries. The restriction to non-essential
recipients is forced rather than chosen (see RETENTION rule 1).

PHENOTYPE, AND THE THREE THINGS IT CANNOT HOLD.

- ``n_samples`` / ``sample_unit``. The release states the replicate design exactly
  (``SOURCED_VALUES["replicate_design"]``) AND prints the per-cell colony measurements,
  so the count is measured per record rather than assumed: 258,364 of the stored records
  rest on the documented four colonies and the rest on 8 to 182, over 2, 8 or 13
  replicate screens. ``GeneInteractionPhenotype`` has no ``n_samples`` field and it is a
  SERVED graph class, so adding one would force a full knowledge-graph rebuild. The
  measured distribution goes to ``preprocess/replicate_structure.json`` (issue #793).
- a per-record p-value. The paper states ``P < 0.0001`` for the |Z|>=4 cut as a whole,
  never per cell, and sheet 4 releases no p-value column, so
  ``gene_interaction_p_value`` carries a typed ``not_reported_by_primary`` gap. Sheet 3's
  |Z| score is a released per-cell statistic with no slot on the class either -- the same
  #793 constraint as ``n_samples``, so it is recorded in
  ``preprocess/high_confidence_pairs.json`` for the high-confidence rows rather than
  derived into a p-value through a normal CDF.
- ``screen_id`` stays ``None``. Its documented purpose is disambiguating one
  publication's repeated measurement of the same (genotype, environment); here each cell
  is released once, and the query gene -- which IS the screen -- is already the
  ``cat``-marked leaf of the genotype.

THE TWO KEIO ISOLATES ARE TWO STRAINS, AND THE SCHEMA SAYS WHERE THAT GOES. 3,956 of the
3,968 non-essential genes are arrayed as two independently constructed isolates, each
with its own S score against each query, so collapsing them would mean averaging (a
derivation) or picking one (unsourced). ``StrainConstruction.batch`` is the field for
exactly this: its docstring cites Hillenmeyer's "some gene deletions were constructed
more than once, in different batches" and states that two records of one ORF whose
construction differs are two strains, not two replicates. Butland's footnote d calls the
column "an in house ID given for the two independent isolates" and the Results report
that the two isolates "occasionally" disagree markedly, "suggesting a defect in one
strain". So the verbatim token ``Isolate 1`` / ``Isolate 2`` is stored as that leaf's
``batch`` and nothing is averaged.

THE S-SCORE FORMULA DEFERS TO COLLINS 2006, WHICH IS NOT MIRRORED. Both Supplementary
Table 3 and Supplementary Table 4 footnote the score to "an algorithm implemented for
yeast SGA (Collins et al., 2006)" = Genome Biol 7:R63. That paper is NOT in the
literature mirror, so the operational definition in ``SOURCED_VALUES["score_definition"]``
is as far as the provenance chain reaches, and no other source is substituted for it.
The chain is: this release's own workbook footnote -> Collins 2006 -> unmirrored.

EXACT ZEROS ARE RELEASED VALUES, NOT A SENTINEL, AND THAT WAS MEASURED. 8,559 of the
309,036 non-essential cells carry an S of exactly 0, and the release documents no zero
convention, so the question had to be settled on the bytes. Three measurements say the
zeros are the small-magnitude end of the distribution: the mean |Z| of the zero-S cells
is 0.1116 against 0.1214 for the cells whose |S| is nonzero but under 0.05; 8,071 of the
8,559 carry at least one non-zero raw colony measurement; and an absent colony does NOT
map to zero, since 310 of the 798 cells whose every raw colony reads 0 carry a nonzero S,
down to -17.8. So a zero is stored as released. The 395 cells that are BOTH all-zero-colony
and exactly-zero-score are the one case where the release contradicts itself, and they are
dropped under rule 5 rather than stored as "no interaction" for a strain that never grew.

RETENTION (rules in order; counts, reasons and items in
``preprocess/dropped_records.json``). Each rule counts only the cells no earlier rule
removed.

1. ``spa_tag_recipient_has_no_bacterial_perturbation_leaf`` -- 5,811 cells (the 149
   SPA-tag essential recipient rows x 39 queries). A ``kan``-marked C-terminal SPA tag on
   an essential gene lowers transcript abundance and is neither a deletion, a mapped
   transposon insertion, a CRISPRi knockdown nor a promoter replacement, so no bacterial
   gene-perturbation leaf can type it (issue #792). This is the same blocker the Babu
   loader filed, and it is why the scope is the non-essential half of the array.
2. ``b_number_is_not_a_locus_tag_of_the_pinned_annotation`` -- 6,318 cells on one of 82
   released recipient ids GCA_000005845.2 does not carry under any layer (78 ``JW`` Keio
   ids, ``CSCR``, and the ids the annotation resolves to more than one locus). A leaf's
   name validator refuses most outright and storing any would put a record on a locus the
   assembly does not have.
3. ``b_number_remapped_by_the_annotation`` -- 4,407 cells on one of 57 b-numbers the
   assembly carries as a ``/gene_synonym`` of a DIFFERENT locus. Storing the merged locus
   needs a ``DerivedIdentifierMapping`` and ``DerivedIdentifierRoute`` has no member for a
   retired tag of the pinned strain's own namespace (issue #753), so the cell is dropped
   rather than remapped silently.
4. ``self_pair_is_not_a_digenic_genotype`` -- 78 cells (each of the 39 query genes is
   itself in the recipient array, x 2 isolates). The Supplementary Methods say these exist
   ("we observed small numbers of self double mutants"), but a ``Genotype`` of two leaves
   on ONE locus would assert deleting the same gene twice.
5. ``released_score_contradicts_its_own_raw_colonies`` -- 395 cells whose every raw colony
   measurement reads 0, so no double mutant grew, yet whose released S is exactly 0.0, so
   no interaction. The release gives no rule for reconciling the two.
6. ``already_served_by_gene_interaction_babu2014`` -- 1,448 cells whose oriented
   (query, recipient) gene pair the served Babu store already holds, dropped so nothing is
   stored twice (see THE PARTITION).

5,811 + 6,318 + 4,407 + 78 + 395 + 1,448 = 18,457 dropped; 314,847 - 18,457 = 296,390
records, 143,651 aggravating, 144,953 alleviating and 7,786 at exactly zero, over 39
query genes and 3,829 recipient genes.

THE PARTITION AGAINST THE SERVED BABU STORE, PROVED BOTH WAYS AT BUILD TIME
(``assert_served_partition``, ``preprocess/served_partition.json``). The unit of the
proof is the ORIENTED gene pair, because that is what identifies a strain in this family:
swapping query and recipient swaps the ``cat`` and ``kan`` cassettes, which the Babu
loader already relies on to keep its 102 reciprocal pairs as two records each.

- FORWARD: not one stored cell shares an oriented (query, recipient) gene pair with a
  served record, so nothing enters the graph twice. Dropping a pair drops BOTH its isolate
  cells, since Babu's single score is derived from the same colonies.
- REVERSE: 725 of the 727 served ``Butland et al.`` records ARE cells of this release, so
  the overlap is accounted for rather than assumed. The 2 exceptions are pinned by name:
  ``b2528 -> b4486`` and ``b2531 -> b4486``, which this release names ``b2528 -> b4344``
  and ``b2531 -> b4344`` (gene name ``*``, absent from Genobase ver. 6) and rule 3 drops,
  because the assembly carries ``b4344`` as a synonym of ``b4486``. No record of Babu's
  other screen set (``This Study``) collides with any cell of this matrix.

Both counts are constants (:data:`SERVED_OVERLAP_CELLS`, :data:`SERVED_BUTLAND_PAIRS`,
:data:`SERVED_PAIRS_NOT_IN_THIS_RELEASE`), so a drift in either store stops the build.

STRAIN. The same conjugant chassis as Babu 2014, for the same reason: every released
identifier is an MG1655 b-number, so records pin
``assembly_reference("MG1655", background=...)``, and the physical strain is a
``BacterialStrainBackground`` whose ``parents`` name Hfr Cavalli and the Keio collection.

MEDIUM AND CONDITION. One condition: the second double-drug selection plate, LB agar with
kanamycin and chloramphenicol, 24 h at 32 C, which is the plate that was photographed and
scored. No dose is asserted for either drug. The SI prints kanamycin at 50 ug/mL for the
recipient pre-culture plate and 25 ug/mL "throughout" the custom mini-array protocol, and
prints nothing at all for the genome-wide selection plates, which is a disagreement within
one mirrored source rather than a value. ``base_medium`` is the ``LB`` library key.

NO PUBMED ID. The mirror carries no PubMed id for this DOI and the schedule row prints
none, so ``Publication.pubmed_id`` is ``None``; a journal article identified by its DOI
plus a URL satisfies the class, and no id is invented or fetched live.

REFERENCE. One per record set: the unperturbed conjugant chassis at a gene-interaction
score of 0, which on the S axis is "no epistasis".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import os.path as osp
import re
import shutil
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Final

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, TypeAdapter
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
    write_verified,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import LB
from torchcell.datamodels.schema import (
    BACTERIAL_ASSEMBLY_SETS,
    AssemblyReferenceGenome,
    BacterialAssemblySet,
    BacterialDeletionPerturbation,
    BacterialGeneInteractionExperiment,
    BacterialGeneInteractionExperimentReference,
    BacterialGeneNamespace,
    BacterialStrainBackground,
    Environment,
    Experiment,
    ExperimentReference,
    GeneInteractionPhenotype,
    Genotype,
    Media,
    MediaComponent,
    MediaComponentRole,
    Publication,
    StrainConstruction,
    Temperature,
)
from torchcell.datasets.bacteria_common import (
    LOCUS_TAG_PATTERNS,
    STRAIN_GENE_NAMESPACES,
    LocusTagReconciliation,
    assembly_reference,
    bacterial_genome,
    reconcile_locus_tags,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.literature.provenance import run_retriever
from torchcell.sequence.genome.base import GeneNameStatus
from torchcell.sequence.genome.ecoli.k12 import EcoliK12Genome, EcoliK12StrainName
from torchcell.verification.levels import l0_structural, l1_count
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
    audit_sourced_value,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY = "butlandESGAColiSynthetic2008"
PAPER_DOI = "10.1038/nmeth.1239"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
LIBRARY_DIR_REL = f"torchcell-library/{CITATION_KEY}"

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "a3da20a90b56b85e1cd7e23e784c318e78cec115b5bc3e94afcd89a5f94f8eef"
METHODS_MD = "si/si1.md"
METHODS_MD_SHA256 = "a5e1648bea298f26ab23cb80de96c1c68f7aa7d5d6420aa56608991bc65aa278"

#: The already-served Babu 2014 store this build is partitioned against.
BABU_KEY = "babuQuantitativeGenomeWideGenetic2014"
BABU_ROOT_REL = "data/torchcell/gene_interaction_babu2014"
BABU_SCREEN_TAG = "Butland et al."

TABLE_S1 = "si2.xls"
TABLE_S2 = "si3.xls"
TABLE_S3 = "si4.xls"
TABLE_S4 = "si5.xls"
#: When the literature mirror captured this paper's SI (from its ``manifest.json``).
DATA_RETRIEVED_AT = "2026-10-07T09:58:24.878215+00:00"
_ESM = "https://static-content.springer.com/esm/art%3A10.1038%2Fnmeth.1239/MediaObjects"

_ASSEMBLY_SET: TypeAdapter[BacterialAssemblySet] = TypeAdapter(BacterialAssemblySet)
REFERENCE_STRAIN_NAME: Final[EcoliK12StrainName] = "MG1655"
MG1655_NAMESPACE: Final[BacterialGeneNamespace] = STRAIN_GENE_NAMESPACES["MG1655"]
MG1655_ASSEMBLY_SET: Final[BacterialAssemblySet] = _ASSEMBLY_SET.validate_python(
    BACTERIAL_ASSEMBLY_SETS["MG1655"]
)
B_NUMBER_PATTERN = LOCUS_TAG_PATTERNS[MG1655_NAMESPACE]


class RawFile(BaseModel):
    """One file the loader consumes: its pinned bytes and how they were retrieved."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    sha256: str
    bytes: int
    description: str
    retrieval: RetrievalRecord

    @property
    def mirror_relpath(self) -> str:
        """Path inside the raw mirror (``data/<name>``)."""
        return f"data/{self.name}"


def _springer(
    name: str, moesm: int, sha256: str, size: int, description: str
) -> RawFile:
    """A publisher SI workbook of this article's Springer ESM endpoint."""
    url = f"{_ESM}/41592_2008_BFnmeth1239_MOESM{moesm}_ESM.xls"
    return RawFile(
        name=name,
        sha256=sha256,
        bytes=size,
        description=description,
        retrieval=RetrievalRecord(
            method=RetrievalMethod.springer_esm,
            source_url=url,
            retriever="torchcell.literature.retrieve.springer_esm",
            params={"url": url},
            sha256=sha256,
            retrieved_at=DATA_RETRIEVED_AT,
        ),
    )


RAW_FILES: tuple[RawFile, ...] = (
    _springer(
        TABLE_S1,
        306,
        "7e088873d486307183fbc0eff98b2a29b452c3e3abe8a636979966fd7b49deea",
        717824,
        "Supplementary Table 1: the recipient array roster, one row per strain with its "
        "b-number, its 'Non-essential' / 'SPA-tag essential' label and its strain "
        "version. Read to confirm the matrix's own row block is this roster",
    ),
    _springer(
        TABLE_S2,
        307,
        "67886f46c2b6561d40f9c7fc24b56b6e6cfdf199d1c13cfa4acfa59fd75d9ef1",
        32768,
        "Supplementary Table 2: the 39 query deletion strains of the genome-wide "
        "screens. Read for the query b-numbers the matrix's columns must be",
    ),
    _springer(
        TABLE_S3,
        308,
        "a2ecd1071d665af77d4f7ae32c10b335b1f5ed8616de0ceb95f0f53c9ad24024",
        290816,
        "Supplementary Table 3: the |Z|>=4 high-confidence interactions with their "
        "S-score, |Z score| and Log2(Q/R). A measured proper subset of Table 4, read "
        "to prove that and to write the high-confidence ledger, never stored as records",
    ),
    _springer(
        TABLE_S4,
        309,
        "74a6ea3a0373fa6e1776b0becb4212f29b9876647b96327db5222dcd68a8822b",
        36617216,
        "Supplementary Table 4: the unfiltered 39-screen matrix in four sheets. Sheet "
        "'S scores' is the authoritative source of every stored value; sheet 'Raw "
        "Colony Sizes' is read ONLY to count each cell's colony measurements and to "
        "find the cells whose score contradicts them",
    ),
)
DATA_SHA256: dict[str, str] = {raw.name: raw.sha256 for raw in RAW_FILES}

NOT_LOADED = (
    "Supplementary Table 4 sheet 'Raw Colony Sizes' (si5.xls) and sheet 'Normalized "
    "Colony Sizes': the per-colony pixel areas and their plate-normalized medians. "
    "Both are a colony-size fitness readout and torchcell has no colony-size phenotype "
    "class, so neither can be typed; the raw sheet IS read, for the per-record "
    "replicate count and for rule 5, but no value of it is stored. The normalized "
    "sheet additionally carries sentinel values (-100000 and 0 among otherwise "
    "positive sizes) that the release never documents",
    "Supplementary Table 4 sheet 'Z scores': the per-cell |Z| statistic. "
    "GeneInteractionPhenotype's only statistic field is gene_interaction_p_value and a "
    "|Z| is not a p-value; converting one through a normal CDF would be a derivation "
    "this release never published. The class is SERVED, so it cannot gain a field "
    "without a full rebuild (the same constraint as issue #793)",
    "Supplementary Table 3 (si4.xls) as records: measured to be a PROPER SUBSET of the "
    "Table 4 S-score matrix, all 1,379 rows located at their own cell with an identical "
    "score, so storing it would duplicate 1,379 records. Its 'Functional association' "
    "column is a STRING annotation of the gene pair rather than a measurement of it, "
    "and its Log2(Q/R) column is a second readout with no phenotype class. The file is "
    "read for the high-confidence ledger and for the Collins 2006 deferral quote",
    "Supplementary Table 2 (si3.xls) beyond the query b-numbers: the knockout primer "
    "sequences, the transfer direction and the minute coordinate of each query. "
    "Construction detail of the donor strain, not a genotype-phenotype record",
    "Supplementary Figures 1-6 and the Supplementary Methods (si1.pdf): the plate "
    "images, the linkage and reproducibility analyses and the scoring procedure. "
    "Quoted from the OCR in the LITERATURE mirror (si/si1.md), which is where those "
    "SourcedValues point; none of it is a build input",
)
"""Every released file or sheet the loader read and did NOT store, with the reason."""


# --------------------------------------------------------------------------- #
# Quote helpers (verbatim substrings of sha256-pinned artifacts)
# --------------------------------------------------------------------------- #
_OCR = "MinerU OCR of the publisher PDF (torchcell-library mirror)"
_SHEET = "spreadsheet cell of the publisher's released workbook"


def _sourced(
    value: Any,
    quote: str,
    *,
    uri: str,
    sha256: str,
    method: str,
    page: str | None = None,
    note: str | None = None,
) -> SourcedValue:
    """A value bound to a verbatim quote in one sha256-pinned mirror artifact."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=uri,
            citation_key=CITATION_KEY,
            sha256=sha256,
            method=method,
            page=page,
        ),
    )


def _paper(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """A value bound to a verbatim quote in the pinned ``paper.md``."""
    return _sourced(
        value,
        quote,
        uri=PAPER_MD,
        sha256=PAPER_MD_SHA256,
        method=_OCR,
        page=page,
        note=note,
    )


def _methods(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """A value bound to a verbatim quote in the pinned Supplementary Methods OCR."""
    return _sourced(
        value,
        quote,
        uri=METHODS_MD,
        sha256=METHODS_MD_SHA256,
        method=_OCR,
        page=page,
        note=note,
    )


def _workbook(
    value: Any, quote: str, *, uri: str, sha256: str, page: str, note: str | None = None
) -> SourcedValue:
    """A value bound to a verbatim cell of one pinned released workbook."""
    return _sourced(
        value, quote, uri=uri, sha256=sha256, method=_SHEET, page=page, note=note
    )


# --------------------------------------------------------------------------- #
# Sourced values
# --------------------------------------------------------------------------- #
_Q_ARRAY = (
    "The recipient mutant strains (Supplementary Table 1 online) arrayed in twenty-five "
    "384-well plates included the Keio single gene deletion mutant strain collection8 "
    "covering 3,968 nonessential single gene replacements marked with a "
    "kanamycin-resistance cassette (kan). Of these, 3,956 were represented by two "
    "independent single gene deletion isolates, for a total of 7,924 single gene "
    "deletion mutants8."
)
_Q_SCREENS = (
    "Thirty-nine $E$ . coli genome-wide screens are processed in batch scoring mode "
    "using colony scorer program4."
)
_Q_UNFILTERED = (
    "Supplementary Table 4. Raw colony sizes (see Sheet 1), normalized median colony "
    "sizes (see Sheet 2), |Z scores| (see Sheet 3) and interaction (S) scores (see "
    "Sheet 4) of each mutant gene pair from 39 genome-wide screens without any "
    "filtering parameters."
)
_Q_BANNER = "Number of Keio deletion strains = 7924\nSPA-tag essential genes = 149"
#: Sheet 4's footnote f and Table 3's footnote g, which differ only in that letter. Each
#: is split over two consecutive cells of the footnote block, so the quote is their
#: newline join, verbatim down to the footnote letter and the trailing space.
_Q_S_SCORE_TAIL = (
    "A quantitative interaction score (S), is calculated to quantify the strength and "
    "confidence of epistasis determined for each mutant gene pair based on an algorithm "
    "\nimplemented for yeast SGA (Collins et al., 2006). Negative S-scores correspond "
    "to aggravating interactions, while positive S-scores to alleviating interactions. "
)
_Q_S_SCORE = f"f{_Q_S_SCORE_TAIL}"
_Q_S_SCORE_S3 = f"g{_Q_S_SCORE_TAIL}"
_Q_REPLICATES = (
    "g In the genome-wide screen, each recipient deletion mutant is pinned twice "
    'leaving four replicate recipient colonies representing two "Isolate 1" and two '
    '"Isolate 2" versions of the strain. The number "1" \nrepresent the first replicate '
    'of the genome-wide screen, while the number "2" represent the second replicate of '
    "the same screen. "
)
_Q_ISOLATES = (
    "dAn in house ID given for the two independent isolates of the mutant strains from "
    "KEIO collection."
)
_Q_ISOLATE_DEFECT = (
    "As expected, the results obtained for isolate-1 and isolate-2 deletion strains "
    "were highly correlated (data not shown), although occasionally the scores differed "
    "markedly (Supplementary Fig. 3 online), suggesting a defect in one strain."
)
_Q_SPA = (
    "a Hypomorphic, KanR- marked, strains with C-terminal Sequential Peptide Affinity "
    "(SPA)-tags on essential genes were included in the array collection (Butland et "
    "al., 2005)."
)
_Q_HIGH_CONFIDENCE = (
    "we reclustered the profiles after accounting for linkage (30-kbp window) and used "
    "a stringent |Z score| cut-off of $\\geq 4$ $P < 0 . 0 0 0 1$ to create a "
    "high-confidence dataset of 1,288 genetic interactions (Fig. 4b and Supplementary "
    "Table 3 online). Among these, we detected 799 genetic interactions for nonessential "
    "genes, with 730 categorized as aggravating interactions and only 69 as alleviating "
    "interactions."
)
_Q_CONJUGATION = (
    "For each genome-wide screen, a query gene deletion mutation is first constructed by "
    "homologous recombination in an $E$ . coli Hfr Cavalli donor strain bearing an "
    "integrated temperature-inducible $\\lambda$ -Red high efficiency recombination "
    "system2."
)
_Q_MARKERS = (
    "(a) An Hfr strain with a query gene deletion mutation marked with cat "
    "(chloramphenicol-resistance gene; pink box) is grown overnight in liquid LB with "
    "chloramphenicol (Cm) and pinned onto LB-Cm plates. Simultaneously, the recipient "
    "$\\mathsf { F } ^ { - }$ mutant array strains marked with kan "
    "(kanamycin-resistance gene; red box) are pinned onto LB-kanamycin (Kan) plates."
)
_Q_SELECTION = (
    "(E) Fifth day, the colonies from the first double drug selection plate is pinned "
    "onto a second double drug selection plate."
)
_Q_SCORED_PLATE = (
    "(F) Sixth day, the colonies from each double drug plate is photographed using a "
    "Kaiser $\\mathrm { R } S _ { 1 }$ camera stand (product code no. 5510) and a "
    "digital camera (Canon Powershot A640, 10 Megapixels) with illumination from two "
    "Testrite 16 x 24 light boxes (Freestyle Photographic supplies product #1624). The "
    "captured images were saved as jpeg files and growth phenotype of the double "
    "mutants was quantitatively assayed using an in-house automated image processing "
    "system originally devised for yeast4. In each step of the above process, the "
    "plates were incubated for 24 hrs at $3 2 ^ { \\circ } \\mathrm { C }$ ."
)
_Q_AGAR = (
    "The solid donor and recipient plates are pinned onto a LB-agar conjugation plate "
    "which is incubated for 24 hrs at $3 2 ~ ^ { \\circ } \\mathrm { C }$ . This "
    "conjugation plate is pinned onto a LB-Kan-Cm agar plate for a first selection of "
    "48 hrs at $3 2 ~ ^ { \\circ } \\mathrm { C }$ ."
)
_Q_SELF_PAIRS = (
    "Third, while gene duplication can be a factor in that we observed small numbers of "
    "self double mutants, we found that this was not a vital issue with the two-round "
    "selection protocol we used in our current eSGA screening procedure."
)
_Q_KAN_50 = (
    "(B) Second day, the overnight culture of the query deletion mutant, and the frozen "
    "glycerol stock culture from an ordered array of recipient deletion mutant, marked "
    "with ${ \\mathrm { K a n } } ^ { \\mathrm { R } }$ , were pinned onto a LB medium "
    "containing $3 4 ~ \\mu \\mathrm { g / m l }$ chloramphenicol and 50 $\\mu \\mathrm "
    "{ g / m l }$ kanamycin, respectively at 384 density (24 column x 16 row)"
)
_Q_KAN_25 = (
    "Antibiotics were used at the following concentrations throughout, Ampicillin "
    "(Amp): $1 0 0 ~ \\mathrm { \\mu g / m l }$ ; Chloramphenicol $\\mathrm { ( C m ) }$ "
    ": 34 $\\mu \\mathrm { g / m l }$ and Kanamycin (Kan): $2 5 ~ \\mu \\mathrm { g / m "
    "l }$ ."
)

SOURCED_VALUES: dict[str, SourcedValue] = {
    "recipient_array": _paper(
        {"non_essential_genes": 3968, "isolate_2": 3956, "strains": 7924},
        _Q_ARRAY,
        page="Results, 'Global profiling with 39 genome-wide screens'",
        note="the non-essential half of the array: 3,968 Keio deletions of which 3,956 "
        "are arrayed as two independent isolates, giving the 7,924 recipient strain "
        "rows the matrix carries under the 'Non-essential' label",
    ),
    "spa_tag_recipients": _workbook(
        149,
        _Q_BANNER,
        uri=f"si/{TABLE_S4}",
        sha256=DATA_SHA256[TABLE_S4],
        page="Supplementary Table 4, every sheet, header cell A3",
        note="the other 149 recipient rows. A SPA-tagged essential gene is a hypomorph "
        "with no bacterial perturbation leaf (issue #792), so its 5,811 cells are "
        "dropped rather than typed as deletions",
    ),
    "spa_tag_definition": _workbook(
        "C-terminal SPA tag on an essential gene, kan-marked",
        _Q_SPA,
        uri=f"si/{TABLE_S4}",
        sha256=DATA_SHA256[TABLE_S4],
        page="Supplementary Table 4, every sheet, footnote a",
        note="what the untypable allele IS: a kan-marked C-terminal tag on an essential "
        "gene, not a deletion of it",
    ),
    "screens": _methods(
        39,
        _Q_SCREENS,
        page="Supplementary Methods, colony scoring",
        note="the 39 query columns of the matrix, asserted against Supplementary "
        "Table 2's query roster at build time",
    ),
    "unfiltered_matrix": _workbook(
        {"sheets": 4, "screens": 39},
        _Q_UNFILTERED,
        uri=f"si/{TABLE_S4}",
        sha256=DATA_SHA256[TABLE_S4],
        page="Supplementary Table 4, every sheet, title cells A1 and A2 joined by one "
        "space",
        note="the scope decision in one sentence: the release's own unit of observation "
        "is the cell of an unfiltered matrix, so the loader stores cells",
    ),
    "score_definition": _workbook(
        "interaction score (S), signed",
        _Q_S_SCORE,
        uri=f"si/{TABLE_S4}",
        sha256=DATA_SHA256[TABLE_S4],
        page="Supplementary Table 4, sheet 'S scores', footnote f",
        note="the stored axis and its sign convention. The FORMULA defers one step "
        "further, to Collins et al. 2006 (Genome Biol 7:R63), which is NOT in the "
        "literature mirror, so this operational definition is as far as the chain "
        "reaches and no other source is substituted for it",
    ),
    "score_definition_table_s3": _workbook(
        "interaction score (S), signed",
        _Q_S_SCORE_S3,
        uri=f"si/{TABLE_S3}",
        sha256=DATA_SHA256[TABLE_S3],
        page="Supplementary Table 3, footnote g",
        note="the same definition and the same Collins 2006 deferral, printed again in "
        "the high-confidence table, which is why the two tables' scores are one axis",
    ),
    "replicate_design": _workbook(
        {"colonies": 4, "colonies_per_isolate": 2, "replicate_screens": 2},
        _Q_REPLICATES,
        uri=f"si/{TABLE_S4}",
        sha256=DATA_SHA256[TABLE_S4],
        page="Supplementary Table 4, sheet 'Raw Colony Sizes', footnote g",
        note="the DOCUMENTED design of one stored score: two colonies of this isolate "
        "in each of two replicate screens. It is not uniform -- the raw sheet prints "
        "the colony measurements per cell and they range from 4 to 182 over 2, 8 or 13 "
        "replicate screens -- so the build counts each record's own colonies rather "
        "than asserting four. GeneInteractionPhenotype has no n_samples field (issue "
        "#793), so the measured distribution goes to "
        "preprocess/replicate_structure.json",
    ),
    "isolate_is_a_strain": _workbook(
        ("Isolate 1", "Isolate 2"),
        _Q_ISOLATES,
        uri=f"si/{TABLE_S1}",
        sha256=DATA_SHA256[TABLE_S1],
        page="Supplementary Table 1, footnote d",
        note="what the 'Strain Versions' column is, verbatim. The token is stored as "
        "the recipient leaf's StrainConstruction.batch, which is the schema's field for "
        "two independent constructions of one ORF deletion",
    ),
    "isolates_can_disagree": _paper(
        True,
        _Q_ISOLATE_DEFECT,
        page="Results, 'Global profiling with 39 genome-wide screens'",
        note="why the two isolates are two strains rather than two replicates of one, "
        "and so why their scores are two records rather than one average",
    ),
    "high_confidence_set": _paper(
        {"pairs": 1288, "non_essential": 799, "aggravating": 730, "alleviating": 69},
        _Q_HIGH_CONFIDENCE,
        page="Results, clustering and the high-confidence dataset",
        note="the |Z|>=4 selection this loader does NOT store as records: measured to "
        "be a proper subset of the matrix, written to "
        "preprocess/high_confidence_pairs.json instead",
    ),
    "conjugation": _methods(
        "Hfr Cavalli donor x Keio recipient conjugant",
        _Q_CONJUGATION,
        page="Supplementary Methods, genome-wide eSGA screening procedure",
        note="how the measured double mutant was made: the query's cat-marked deletion "
        "is transferred into the kan-marked F- Keio recipient by conjugation",
    ),
    "markers": _paper(
        ("cat", "kan"),
        _Q_MARKERS,
        page="Figure 1 legend, eSGA outline",
        note="which cassette each side of the cross carries, which is what makes the "
        "query leaf and the recipient leaf two different strain roles",
    ),
    "scored_plate": _methods(
        ("LB with kanamycin and chloramphenicol", 32.0, 24.0),
        _Q_SCORED_PLATE,
        page="Supplementary Methods, genome-wide eSGA screening procedure, step (F)",
        note="the one condition: the double-drug plate that was photographed and "
        "scored, incubated 24 h at 32 C. No component amount is stated for it, so every "
        "component of the stored medium carries concentration=None",
    ),
    "second_selection": _methods(
        2,
        _Q_SELECTION,
        page="Supplementary Methods, genome-wide eSGA screening procedure, step (E)",
        note="which plate the stored score is of: the SECOND double-drug selection "
        "plate, not the conjugation plate and not the first selection",
    ),
    "agar": _methods(
        "agar",
        _Q_AGAR,
        page="Supplementary Methods, custom mini-array eSGA screening",
        note="the selection plates are LB-Kan-Cm agar plates; no percentage is printed "
        "for the gelling agent anywhere, so it is named and unquantified",
    ),
    "self_double_mutants": _methods(
        True,
        _Q_SELF_PAIRS,
        page="Supplementary Methods, discussion of mating conditions",
        note="the release's own statement that a query gene's cell against itself "
        "exists. A Genotype of two leaves on ONE locus would assert deleting the same "
        "gene twice, so those 78 cells are dropped",
    ),
    "kanamycin_dose_50": _methods(
        50.0,
        _Q_KAN_50,
        page="Supplementary Methods, genome-wide eSGA screening procedure, step (B)",
        note="the kanamycin amount of the recipient PRE-CULTURE plate, not of the "
        "scored double-drug plate. Recorded because it is half of a disagreement, never "
        "asserted onto the medium",
    ),
    "kanamycin_dose_25": _methods(
        25.0,
        _Q_KAN_25,
        page="Supplementary Methods, custom mini-array query mutant construction",
        note="the other half: 25 ug/ml 'throughout' the mini-array protocol against 50 "
        "ug/ml in the genome-wide one, and nothing at all for the genome-wide selection "
        "plates. A disagreement within one mirrored source is not a value, so the "
        "stored medium asserts no kanamycin dose",
    ),
}


# --------------------------------------------------------------------------- #
# The one medium and the one environment
# --------------------------------------------------------------------------- #
_NO_AMOUNT = (
    "the paper names the medium and its two marker drugs and prints no amount for any "
    "component of the scored double-drug plate, so none is asserted here"
)
_NO_DOSE = (
    "no dose is asserted: the SI prints kanamycin at 50 ug/ml for the recipient "
    "pre-culture plate and 25 ug/ml 'throughout' the mini-array protocol, and nothing "
    "for the genome-wide selection plates"
)
TEMPERATURE_C = 32.0
DURATION_HOURS = 24.0
AEROBICITY = "aerobic"


def selection_medium() -> Media:
    """The LB-Kan-Cm plate the double-mutant colonies were photographed on."""
    statement = SOURCED_VALUES["scored_plate"]
    doses = [SOURCED_VALUES["kanamycin_dose_50"], SOURCED_VALUES["kanamycin_dose_25"]]
    return Media(
        name="LB with kanamycin and chloramphenicol, amounts not stated "
        "(Butland 2008), solid",
        state="solid",
        is_synthetic=False,
        base_medium="LB",
        components=[
            *(
                MediaComponent(
                    compound=component.compound,
                    role=component.role,
                    concentration=None,
                    definition=component.definition,
                    provenance=[statement],
                    note=_NO_AMOUNT,
                )
                for component in LB.components
            ),
            MediaComponent(
                compound=resolved_compound("agar"),
                role=MediaComponentRole.gelling_agent,
                concentration=None,
                provenance=[SOURCED_VALUES["agar"]],
                note="the selection plates are agar plates; no percentage is printed",
            ),
            MediaComponent(
                compound=resolved_compound("kanamycin"),
                role=MediaComponentRole.selection_agent,
                concentration=None,
                provenance=[statement, SOURCED_VALUES["markers"], *doses],
                note=f"selects the recipient's kan cassette; {_NO_DOSE}",
            ),
            MediaComponent(
                compound=resolved_compound("chloramphenicol"),
                role=MediaComponentRole.selection_agent,
                concentration=None,
                provenance=[statement, SOURCED_VALUES["markers"]],
                note="selects the query's cat cassette, so a colony on this medium is a "
                "double mutant; the SI prints 34 ug/ml for the query pre-culture and "
                "nothing for the genome-wide selection plates",
            ),
        ],
        provenance=[statement, SOURCED_VALUES["agar"]],
    )


SELECTION_MEDIUM = selection_medium()
"""Built once: one condition, so one medium object every record joins on."""


def screen_environment() -> Environment:
    """The one environment: LB-Kan-Cm plates, 32 C, 24 h, aerobic."""
    return Environment(
        media=SELECTION_MEDIUM,
        temperature=Temperature(value=TEMPERATURE_C),
        perturbations=[],
        aerobicity=AEROBICITY,
        duration_hours=DURATION_HOURS,
    )


SCREEN_ENVIRONMENT = screen_environment()
"""The single environment, built once."""

PUBLICATION = Publication(
    pubmed_id=None,
    pubmed_url=None,
    doi=PAPER_DOI,
    doi_url=f"https://doi.org/{PAPER_DOI}",
)


# --------------------------------------------------------------------------- #
# Raw mirror (the loader reads the mirror, never a live URL)
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirrors and build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/butlandESGAColiSynthetic2008``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def library_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-library/butlandESGAColiSynthetic2008``."""
    return Path(data_root or _data_root()) / LIBRARY_DIR_REL


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retrieve_raw_files(
    dest_dir: str | Path, names: Sequence[str] | None = None
) -> dict[str, Path]:
    """Run each file's recorded retriever and write the verified bytes to ``dest_dir``.

    The recorded ``RetrievalRecord`` is what runs (``run_retriever``), so this is the
    re-runnable retrieval itself; a byte mismatch raises before anything is written.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for raw in RAW_FILES:
        if names is not None and raw.name not in names:
            continue
        path = dest / raw.name
        write_verified(
            run_retriever(raw.retrieval),
            path,
            raw.sha256,
            raw.retrieval.source_url or raw.name,
        )
        out[raw.name] = path
    return out


MIRROR_EXPECTATION = (
    "Supplementary Tables 1-4 (data/si2.xls .. data/si5.xls) are what "
    "GeneInteractionButland2008Dataset consumes; Supplementary Table 1 (si1.pdf, the "
    "Supplementary Methods and Figures) stays in the literature mirror because no build "
    "input reads its bytes, only its OCR"
)


def deposit_raw_mirror(
    *, sources: Mapping[str, str | Path], data_root: str | None = None
) -> Path:
    """Add this loader's four workbooks to the existing raw mirror, ADDITIVELY.

    The mirror is shared: the subsumption measurement deposited these same files and
    wrote ``subsumption_record.json`` beside them. So nothing here is rewritten from
    scratch -- a manifest entry another run wrote is kept byte-identical, one whose
    sha256 disagrees raises, and ``si_expected`` gains this loader's sentence instead of
    replacing whatever is there.
    """
    missing = sorted(set(DATA_SHA256) - set(sources))
    if missing:
        raise KeyError(f"no source given for {missing}")
    root = raw_mirror_dir(data_root)
    root.mkdir(parents=True, exist_ok=True)
    records: list[ArtifactRecord] = []
    for raw in RAW_FILES:
        src = Path(sources[raw.name])
        got = _sha256(src)
        if got != raw.sha256:
            raise RuntimeError(
                f"{src} sha256 mismatch: got {got}, expected {raw.sha256}"
            )
        dest = root / raw.mirror_relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if _sha256(dest) != raw.sha256:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            shutil.copy2(src, dest)
        records.append(
            ArtifactRecord(
                path=raw.mirror_relpath,
                role=ROLE_RAW_DATA,
                bytes=dest.stat().st_size,
                sha256=raw.sha256,
                source=raw.retrieval.source_url,
                retrieval=raw.retrieval,
            )
        )
    path = root / "manifest.json"
    manifest = (
        Manifest.model_validate_json(path.read_text())
        if path.exists()
        else Manifest(
            citation_key=CITATION_KEY,
            doi=PAPER_DOI,
            title="eSGA: E. coli synthetic genetic array analysis",
        )
    )
    held = {record.path: record for record in manifest.files}
    for record in records:
        existing = held.get(record.path)
        if existing is None:
            manifest.files.append(record)
        elif existing.sha256 != record.sha256:
            raise RuntimeError(
                f"{path} holds {record.path} at {existing.sha256}, this loader pins "
                f"{record.sha256}; refusing"
            )
    manifest.files.sort(key=lambda record: record.path)
    for record in records:
        if record.source is not None and record.source not in manifest.si_data_sources:
            manifest.si_data_sources.append(record.source)
    if MIRROR_EXPECTATION not in manifest.si_expected:
        manifest.si_expected.append(MIRROR_EXPECTATION)
    manifest.provenance_complete = True
    manifest.created_at = manifest.created_at or datetime.now(UTC).isoformat()
    path.write_text(manifest.model_dump_json(indent=2))
    return root


def load_manifest(data_root: str | None = None) -> Manifest:
    """Read the raw mirror's ``manifest.json``."""
    return Manifest.model_validate_json(
        (raw_mirror_dir(data_root) / "manifest.json").read_text()
    )


def manifest_sha256(manifest: Manifest, relpath: str) -> str:
    """The recorded sha256 of one mirror file."""
    for record in manifest.files:
        if record.path == relpath:
            return record.sha256
    raise KeyError(f"{relpath} is not in the raw-mirror manifest")


# --------------------------------------------------------------------------- #
# Reading the released sheets
# --------------------------------------------------------------------------- #
class SheetFormatError(ValueError):
    """A released sheet whose title, sheet list or layout is not what we read."""


class ReleaseContentError(ValueError):
    """A released sheet whose contents contradict a sourced count."""


#: The two labels of the roster's first column; any other row is a banner or a footnote.
LABEL_NON_ESSENTIAL = "Non-essential"
LABEL_SPA_TAG = "SPA-tag essential"
ARRAY_LABELS: Final[tuple[str, str]] = (LABEL_NON_ESSENTIAL, LABEL_SPA_TAG)
#: Where the matrix sheets put their pieces: row 0+1 the title, row 2 the banner, row 3
#: the query b-numbers, row 4 the query gene names, rows 6+ the recipient block, and
#: columns 0-3 the label, gene name, b-number and strain version of each recipient.
TITLE_ROWS: Final[tuple[int, int]] = (0, 1)
BANNER_ROW: Final = 2
QUERY_TAG_ROW: Final = 3
QUERY_GENE_ROW: Final = 4
BODY_ROW: Final = 6
LABEL_COL, GENE_COL, TAG_COL, VERSION_COL = 0, 1, 2, 3
FIRST_VALUE_COL: Final = 4

SHEET_RAW = "Raw Colony Sizes"
SHEET_NORMALIZED = "Normalized Colony Sizes"
SHEET_Z = "Z scores"
SHEET_S = "S scores"
MATRIX_SHEETS: Final[tuple[str, ...]] = (SHEET_RAW, SHEET_NORMALIZED, SHEET_Z, SHEET_S)
ROSTER_SHEET = "st1"
QUERY_SHEET = "donor_st2"
HIGH_CONFIDENCE_SHEET = "st3"

#: The released counts the build asserts, each the ``value`` of its ``SOURCED_VALUES``
#: entry. Named here so a hermetic test can build a small synthetic release.
N_SCREENS: int = int(SOURCED_VALUES["screens"].value)
N_NON_ESSENTIAL_STRAINS: int = int(SOURCED_VALUES["recipient_array"].value["strains"])
N_SPA_TAG_STRAINS: int = int(SOURCED_VALUES["spa_tag_recipients"].value)
N_HIGH_CONFIDENCE_PAIRS: int = int(SOURCED_VALUES["high_confidence_set"].value["pairs"])

#: Supplementary Table 3's columns, in release order, renamed for use.
_S3_COLUMNS: Final[tuple[str, ...]] = (
    "query_gene",
    "query_tag",
    "recipient_gene",
    "recipient_tag",
    "version",
    "essentiality",
    "functional_association",
    "s_score",
    "abs_z",
    "log2_qr",
)
#: Supplementary Table 3's own label for a non-essential recipient (lower case there).
S3_NON_ESSENTIAL = "non-essential"


class Matrix(BaseModel):
    """One sheet of Supplementary Table 4, read as a rectangle plus its two headers.

    ``values`` is the raw object block, kept untyped because the sheets differ: the S,
    normalized and |Z| sheets hold numbers (with one footnote-marked cell each) and the
    raw sheet holds the per-colony strings.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    sheet: str
    title: str
    banner: str
    query_tags: tuple[str, ...]
    query_genes: tuple[str, ...]
    labels: tuple[str, ...]
    recipient_genes: tuple[str, ...]
    recipient_tags: tuple[str, ...]
    versions: tuple[str, ...]
    values: Any


def read_matrix_sheet(path: str | Path, sheet: str) -> Matrix:
    """One Table 4 sheet, refusing a layout that is not the one we read.

    The title is joined from the two cells the workbook splits it over and the banner is
    the cell beneath, so a re-released file with a shifted block fails here rather than
    silently loading a header row as data.
    """
    frame = pd.read_excel(path, sheet_name=sheet, header=None)
    if frame.shape[1] <= FIRST_VALUE_COL:
        raise SheetFormatError(f"{sheet}: {frame.shape[1]} columns")
    title = " ".join(str(frame.iat[row, 0]).strip() for row in TITLE_ROWS)
    body = frame.iloc[BODY_ROW:, :]
    rows = body[body.iloc[:, LABEL_COL].astype(str).isin(ARRAY_LABELS)]
    tags = tuple(str(value) for value in frame.iloc[QUERY_TAG_ROW, FIRST_VALUE_COL:])
    bad = [tag for tag in tags if not re.match(B_NUMBER_PATTERN, tag)]
    if bad:
        raise SheetFormatError(f"{sheet}: query header holds {bad[:5]}")
    return Matrix(
        sheet=sheet,
        title=title,
        banner=str(frame.iat[BANNER_ROW, 0]),
        query_tags=tags,
        query_genes=tuple(
            str(value) for value in frame.iloc[QUERY_GENE_ROW, FIRST_VALUE_COL:]
        ),
        labels=tuple(rows.iloc[:, LABEL_COL].astype(str)),
        recipient_genes=tuple(rows.iloc[:, GENE_COL].astype(str)),
        recipient_tags=tuple(rows.iloc[:, TAG_COL].astype(str)),
        versions=tuple(rows.iloc[:, VERSION_COL].astype(str)),
        values=rows.iloc[:, FIRST_VALUE_COL:].to_numpy(),
    )


def _check_quote(text: str, name: str, what: str) -> None:
    """A sheet cell must still read exactly as the sourced quote says it does."""
    quote = str(SOURCED_VALUES[name].quote).strip()
    if text.strip() != quote:
        raise SheetFormatError(
            f"{what} reads {text.strip()!r}, the quote says {quote!r}"
        )


def read_s_scores(path: str | Path) -> Matrix:
    """The authoritative sheet: the S score of every (query, recipient strain) cell.

    Its title cell is the "without any filtering parameters" quote and its banner cell
    is the array census, so both sourced values are re-checked against the bytes they
    were taken from before any number is read.
    """
    sheets = tuple(str(name) for name in pd.ExcelFile(path).sheet_names)
    if sheets != MATRIX_SHEETS:
        raise SheetFormatError(
            f"{TABLE_S4}: sheets {list(sheets)}, not {MATRIX_SHEETS}"
        )
    matrix = read_matrix_sheet(path, SHEET_S)
    _check_quote(matrix.title, "unfiltered_matrix", f"{SHEET_S} title")
    _check_quote(matrix.banner, "spa_tag_recipients", f"{SHEET_S} banner")
    counts = Counter(matrix.labels)
    if (
        counts[LABEL_NON_ESSENTIAL] != N_NON_ESSENTIAL_STRAINS
        or counts[LABEL_SPA_TAG] != N_SPA_TAG_STRAINS
    ):
        raise ReleaseContentError(
            f"{SHEET_S}: {dict(counts)} recipient rows; the release states "
            f"{N_NON_ESSENTIAL_STRAINS} non-essential and {N_SPA_TAG_STRAINS} SPA-tag"
        )
    if len(matrix.query_tags) != N_SCREENS:
        raise ReleaseContentError(
            f"{SHEET_S}: {len(matrix.query_tags)} query columns, the paper states "
            f"{N_SCREENS}"
        )
    keys = list(zip(matrix.recipient_tags, matrix.versions, strict=True))
    if len(set(keys)) != len(keys):
        raise ReleaseContentError(
            f"{SHEET_S}: a (b-number, strain version) row key appears twice"
        )
    return matrix


def score_block(matrix: Matrix) -> Any:
    """The S sheet's values as floats, with the one footnote-marked cell unmarked.

    Exactly one cell of each numeric sheet carries the footnote letter that binds the
    sheet's score definition to Collins 2006; it is the only non-numeric cell, which is
    asserted rather than assumed.
    """
    frame = pd.DataFrame(matrix.values)
    numeric = frame.apply(lambda column: pd.to_numeric(column, errors="coerce"))
    marked = [
        (int(row), int(column))
        for row, column in zip(*numeric.isna().to_numpy().nonzero(), strict=True)
    ]
    if len(marked) != 1:
        raise SheetFormatError(
            f"{matrix.sheet}: {len(marked)} non-numeric cells, expected the one "
            f"footnote-marked cell: {marked[:5]}"
        )
    row, column = marked[0]
    text = str(frame.iat[row, column])
    if not re.fullmatch(r"-?\d+(\.\d+)?[a-z]", text):
        raise SheetFormatError(f"{matrix.sheet}: non-numeric cell {text!r}")
    numeric.iat[row, column] = float(text[:-1])
    return numeric.to_numpy(dtype=float)


_COLONY_GROUP = re.compile(r"(\d+):\(([-\d,]+)\)")


def colony_counts(matrix: Matrix) -> tuple[Any, Any, Any]:
    """Per cell of the raw sheet: colonies measured, colonies at zero, replicate screens.

    The raw sheet is read for these three counts and for nothing else: no colony size is
    stored, because torchcell has no colony-size phenotype class. A cell that does not
    parse raises, so a re-released sheet cannot silently produce a count of zero.
    """
    values = matrix.values
    total = np.zeros(values.shape, dtype=np.int32)
    zeros = np.zeros(values.shape, dtype=np.int32)
    screens = np.zeros(values.shape, dtype=np.int32)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            text = re.sub(r"[a-z]", "", str(values[row, column]))
            groups = _COLONY_GROUP.findall(text)
            if ", ".join(f"{a}:({b})" for a, b in groups) != text:
                raise SheetFormatError(
                    f"{matrix.sheet}: cell ({row}, {column}) reads "
                    f"{values[row, column]!r}, not '<n>:(<size>,...)' groups"
                )
            sizes = [int(size) for _, body in groups for size in body.split(",")]
            total[row, column] = len(sizes)
            zeros[row, column] = sum(1 for size in sizes if size == 0)
            screens[row, column] = len(groups)
    return total, zeros, screens


def read_query_roster(path: str | Path) -> dict[str, str]:
    """Supplementary Table 2's 39 query deletion strains, as ``b-number -> gene``."""
    frame = pd.read_excel(path, sheet_name=QUERY_SHEET, header=None)
    rows = frame.iloc[3:, :]
    rows = rows[rows.iloc[:, 1].astype(str).str.match(B_NUMBER_PATTERN)]
    roster = {str(row.iloc[1]): str(row.iloc[0]) for _, row in rows.iterrows()}
    if len(roster) != N_SCREENS:
        raise ReleaseContentError(
            f"{TABLE_S2}: {len(roster)} query strains, the paper states {N_SCREENS}"
        )
    return roster


def read_array_roster(path: str | Path) -> dict[tuple[str, str], str]:
    """Supplementary Table 1's roster, as ``(b-number, strain version) -> label``."""
    frame = pd.read_excel(path, sheet_name=ROSTER_SHEET, header=None)
    body = frame.iloc[BODY_ROW:, :]
    rows = body[body.iloc[:, LABEL_COL].astype(str).isin(ARRAY_LABELS)]
    roster = {
        (str(row.iloc[TAG_COL]), str(row.iloc[VERSION_COL])): str(row.iloc[LABEL_COL])
        for _, row in rows.iterrows()
    }
    counts = Counter(roster.values())
    if (
        counts[LABEL_NON_ESSENTIAL] != N_NON_ESSENTIAL_STRAINS
        or counts[LABEL_SPA_TAG] != N_SPA_TAG_STRAINS
    ):
        raise ReleaseContentError(
            f"{TABLE_S1}: {dict(counts)} roster rows; the release states "
            f"{N_NON_ESSENTIAL_STRAINS} non-essential and {N_SPA_TAG_STRAINS} SPA-tag"
        )
    return roster


def read_high_confidence(path: str | Path) -> pd.DataFrame:
    """Supplementary Table 3's |Z|>=4 rows, typed, with the banner rows dropped."""
    frame = pd.read_excel(path, sheet_name=HIGH_CONFIDENCE_SHEET, header=2)
    if frame.shape[1] != len(_S3_COLUMNS):
        raise SheetFormatError(f"{TABLE_S3}: {frame.shape[1]} columns")
    frame.columns = list(_S3_COLUMNS)
    keep = frame["query_tag"].astype(str).str.match(B_NUMBER_PATTERN) & frame[
        "recipient_tag"
    ].astype(str).str.match(B_NUMBER_PATTERN)
    data = frame[keep].copy()
    for column in ("s_score", "abs_z", "log2_qr"):
        data[column] = data[column].astype(float)
    for column in ("query_tag", "recipient_tag", "version", "essentiality"):
        data[column] = data[column].astype(str)
    ordered = data.drop_duplicates(["query_tag", "recipient_tag"])
    if len(ordered) != N_HIGH_CONFIDENCE_PAIRS:
        raise ReleaseContentError(
            f"{TABLE_S3}: {len(ordered)} ordered pairs, the paper states "
            f"{N_HIGH_CONFIDENCE_PAIRS}"
        )
    return data.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Auditing a quote that lives in a workbook cell rather than in OCR text
# --------------------------------------------------------------------------- #
class WorkbookQuote(BaseModel):
    """One ``SOURCED_VALUES`` entry whose quote is a cell of a released workbook.

    ``audit_sourced_value`` reads its artifact as UTF-8 text, which an OLE2 ``.xls``
    is not, so a value quoted from a spreadsheet needs its own audit: the same two
    checks (the pinned sha256 still matches, and the quote is still present) against
    the cells instead of against the bytes. ``join`` is how the workbook splits the
    sentence: a footnote runs over consecutive cells of column A and is read with a
    newline between them, while the sheet title is two cells read with one space.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    file: str
    sheet: str
    join: str
    strip: bool
    where: str


WORKBOOK_QUOTES: tuple[WorkbookQuote, ...] = (
    WorkbookQuote(
        name="unfiltered_matrix",
        file=TABLE_S4,
        sheet=SHEET_S,
        join=" ",
        strip=True,
        where="title cells A1 and A2",
    ),
    WorkbookQuote(
        name="spa_tag_recipients",
        file=TABLE_S4,
        sheet=SHEET_S,
        join="\n",
        strip=False,
        where="banner cell A3",
    ),
    WorkbookQuote(
        name="spa_tag_definition",
        file=TABLE_S4,
        sheet=SHEET_S,
        join="\n",
        strip=False,
        where="footnote a of the block below the matrix",
    ),
    WorkbookQuote(
        name="score_definition",
        file=TABLE_S4,
        sheet=SHEET_S,
        join="\n",
        strip=False,
        where="footnote f, over two consecutive cells",
    ),
    WorkbookQuote(
        name="replicate_design",
        file=TABLE_S4,
        sheet=SHEET_RAW,
        join="\n",
        strip=False,
        where="footnote g, over two consecutive cells",
    ),
    WorkbookQuote(
        name="score_definition_table_s3",
        file=TABLE_S3,
        sheet=HIGH_CONFIDENCE_SHEET,
        join="\n",
        strip=False,
        where="footnote g, over two consecutive cells",
    ),
    WorkbookQuote(
        name="isolate_is_a_strain",
        file=TABLE_S1,
        sheet=ROSTER_SHEET,
        join="\n",
        strip=False,
        where="footnote d of the block below the roster",
    ),
)
"""Every sourced value quoted from a spreadsheet cell, with how its cells are joined."""

TEXT_QUOTED: tuple[str, ...] = tuple(
    name
    for name in SOURCED_VALUES
    if name not in {entry.name for entry in WORKBOOK_QUOTES}
)
"""Every sourced value quoted from OCR text, which ``audit_sourced_value`` can read."""


def audit_workbook_quote(
    entry: WorkbookQuote, data_root: str | None = None
) -> LevelResult:
    """Verify one workbook-quoted value: the file's sha256, then the quote's presence."""
    path = raw_mirror_dir(data_root) / f"data/{entry.file}"
    if not path.exists():
        raise FileNotFoundError(f"source artifact not found: {path}")
    sourced = SOURCED_VALUES[entry.name]
    integrity = _sha256(path) == DATA_SHA256[entry.file]
    present = False
    if integrity:
        column = pd.read_excel(path, sheet_name=entry.sheet, header=None).iloc[:, 0]
        cells = [
            str(value).strip() if entry.strip else str(value)
            for value in column
            if isinstance(value, str)
        ]
        present = str(sourced.quote) in entry.join.join(cells)
    if not integrity:
        message = f"sha256 drift: workbook re-released or edited ({entry.file})"
    elif not present:
        message = f"quote no longer found in {entry.file} sheet {entry.sheet!r}"
    else:
        message = f"value backed by verbatim cells of {entry.file} ({entry.where})"
    return LevelResult(
        level=Level.L3,
        name="provenance_audit",
        passed=integrity and present,
        message=message,
        details={
            "citation_key": CITATION_KEY,
            "source": sourced.provenance.source_uri,
            "value": repr(sourced.value),
            "sha256_ok": integrity,
            "quote_present": present,
            "where": entry.where,
        },
    )


# --------------------------------------------------------------------------- #
# Retention ledger
# --------------------------------------------------------------------------- #
RULE_SPA_TAG = "spa_tag_recipient_has_no_bacterial_perturbation_leaf"
RULE_NOT_A_TAG = "b_number_is_not_a_locus_tag_of_the_pinned_annotation"
RULE_REMAPPED = "b_number_remapped_by_the_annotation"
RULE_SELF_PAIR = "self_pair_is_not_a_digenic_genotype"
RULE_CONTRADICTION = "released_score_contradicts_its_own_raw_colonies"
RULE_SERVED = "already_served_by_gene_interaction_babu2014"
#: Not a drop rule: how the one locus this release names twice is stored.
RULE_ROSTER_NAME_WINS = "common_name_comes_from_the_array_roster_not_the_query_header"
#: The roster's token for a strain whose gene name Genobase ver. 6 does not carry.
UNNAMED_IN_GENOBASE = "*"

#: The rules in the order the build applies them; each counts only the cells no earlier
#: rule removed, which is what makes the six counts sum to the dropped total.
DROP_RULES: Final[tuple[str, ...]] = (
    RULE_SPA_TAG,
    RULE_NOT_A_TAG,
    RULE_REMAPPED,
    RULE_SELF_PAIR,
    RULE_CONTRADICTION,
    RULE_SERVED,
)

DROP_RULE_DESCRIPTIONS: dict[str, str] = {
    RULE_SPA_TAG: "the recipient is one of the 149 strains the roster labels 'SPA-tag "
    "essential': a kan-marked C-terminal SPA tag on an essential gene, which lowers "
    "transcript abundance. None of the five bacterial gene-perturbation leaves can type "
    "it -- it is not a deletion, not a mapped transposon insertion, not a CRISPRi "
    "knockdown and not a promoter replacement -- and typing it as a deletion would "
    "assert an absence the paper never claims, so the cell is dropped and the missing "
    "leaf is filed as issue #792",
    RULE_NOT_A_TAG: "a released id is not a locus tag of the pinned annotation at all "
    "(a Keio JW id, a gene symbol, or an id the annotation resolves to more than one "
    "locus). A bacterial perturbation leaf's name validator refuses it, and storing it "
    "would put a record on a locus GCA_000005845.2 does not have",
    RULE_REMAPPED: "a released b-number that the pinned annotation carries as a "
    "/gene_synonym of a DIFFERENT locus. Storing the merged locus needs a "
    "DerivedIdentifierMapping and DerivedIdentifierRoute has no member for a retired tag "
    "of the pinned strain's own namespace (issue #753), so the cell is dropped rather "
    "than remapped silently",
    RULE_SELF_PAIR: "the query gene IS the recipient gene. The Supplementary Methods "
    "record that such cells exist ('we observed small numbers of self double mutants'), "
    "but a Genotype of two leaves on one locus would assert deleting the same gene "
    "twice, which is not what was measured",
    RULE_CONTRADICTION: "every raw colony measurement of this cell reads 0, so no "
    "double mutant grew, yet the released S score is exactly 0.0, so no interaction. "
    "The release documents no rule for reconciling the two, and storing the score would "
    "label a strain that never grew as having no interaction. Measured separately: an "
    "exact zero is NOT a sentinel in general (310 of the 798 all-zero-colony cells carry "
    "a nonzero score, down to -17.8), so only the contradictory cells are dropped",
    RULE_SERVED: "the oriented (query, recipient) gene pair is already a record of the "
    "served GeneInteractionBabu2014Dataset, which re-released 727 of this screen's "
    "measurements under screen_id 'Butland et al.'. Both isolate cells of such a pair "
    "are dropped, because Babu's single score is derived from the same colonies",
}


class DropRule(BaseModel):
    """One retention rule, its count and the items it removed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule: str
    description: str
    n_records: int
    items: list[str]


class DropLog(BaseModel):
    """The retention ledger of one build."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str
    source_records: int
    kept_records: int
    dropped_records: int
    source_queries: int
    source_recipients: int
    kept_queries: int
    kept_recipients: int
    kept_isolates: dict[str, int]
    n_aggravating: int
    n_alleviating: int
    n_zero: int
    rules: list[DropRule]


class ReplicateStructure(BaseModel):
    """The replicate design the schema cannot store, measured per record.

    ``GeneInteractionPhenotype`` has no ``n_samples`` / ``sample_unit`` field and adding
    one would change a served graph class, so the design is recorded here (issue #793).
    It is measured rather than asserted: the documented four colonies are the mode, not
    the rule, and the raw sheet prints each cell's own colony measurements.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_unit: str
    documented_n_samples: int
    documented_colonies_per_isolate: int
    documented_replicate_screens: int
    measured_n_samples_counts: dict[str, int]
    measured_replicate_screen_counts: dict[str, int]
    modal_n_samples: int
    records_at_modal_n_samples: int
    quote: str
    citation_key: str
    source_uri: str
    sha256: str
    note: str


class HighConfidenceLedger(BaseModel):
    """Supplementary Table 3 against the matrix: the subset proof, and what it adds.

    Every row of the released |Z|>=4 table is located at its own cell of the S-score
    matrix and carries the same score, which is why that table is not a second record
    set. ``pairs`` lists the stored high-confidence cells so the selection survives
    without a field on the phenotype to hold it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rows: int
    ordered_pairs: int
    located_in_matrix: int
    identical_score: int
    non_essential_pairs: int
    spa_tag_pairs: int
    stored_rows: int
    not_stored_rows: int
    z_cutoff_note: str
    pairs: list[str]


class GeneNameLedger(BaseModel):
    """Which spelling of a common name each locus is stored under, and why.

    A gene is one entity, so one locus carries one spelling (the shared
    ``canonical_gene_names`` rule). This release prints two for one locus: the matrix's
    query header names ``b1922`` ``rpoF`` while the recipient roster names it ``fliA``,
    and both resolve to ``b1922`` in the pinned annotation, so neither is wrong and
    picking by hand would be arbitrary. The ROSTER spelling wins, because the roster is
    the release's own per-strain name column and the matrix's recipient block IS that
    roster (asserted at build time), while the header is one cell per screen.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule: str
    disagreements: dict[str, dict[str, str]]
    n_records_renamed: int
    unnamed_rows_dropped: int
    unnamed_note: str


class ServedPartition(BaseModel):
    """The partition against the served Babu 2014 store, proved in both directions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    served_root: str
    served_records: int
    served_butland_records: int
    served_butland_pairs: int
    served_fraction_of_this_release: float
    stored_records: int
    stored_pairs: int
    shared_pairs: int
    overlap_cells_dropped: int
    served_pairs_not_in_this_release: list[str]


# --------------------------------------------------------------------------- #
# The record
# --------------------------------------------------------------------------- #
QUERY_COLLECTION = "Hfr Cavalli (Hfr C) query deletion strains"
RECIPIENT_COLLECTION = "Keio collection"
QUERY_CASSETTE = "cat"
RECIPIENT_CASSETTE = "kan"

P_VALUE_GAP = ProvenanceGap(
    field="gene_interaction_p_value",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=Provenance(
        source_uri=METHODS_MD,
        citation_key=CITATION_KEY,
        sha256=METHODS_MD_SHA256,
        method="full Supplementary Methods and the Results' significance section read, "
        "plus every column of Supplementary Tables 3 and 4",
        page="Supplementary Methods; Results, 'Evaluation of data quality and "
        "statistical significance of interaction (S) scores'",
    ),
    note="the paper states P<0.0001 for the |Z|>=4 cut as a whole, never a p-value per "
    "cell, and the S-score sheet releases no p-value column. The |Z| sheet's per-cell "
    "statistic is not a p-value and converting it through a normal CDF would be a "
    "derivation this release never published",
)


def query_perturbation(locus_tag: str, symbol: str) -> BacterialDeletionPerturbation:
    """The query side: the Hfr C deletion transferred by conjugation, marked ``cat``."""
    return BacterialDeletionPerturbation(
        systematic_gene_name=locus_tag,
        perturbed_gene_name=symbol,
        gene_namespace=MG1655_NAMESPACE,
        collection=QUERY_COLLECTION,
        cassette=QUERY_CASSETTE,
    )


def recipient_perturbation(
    locus_tag: str, symbol: str, version: str
) -> BacterialDeletionPerturbation:
    """The recipient side: one arrayed Keio isolate, marked ``kan``.

    ``version`` is the "Strain Versions" token verbatim and is stored as the
    construction ``batch``, because the two isolates are two independent constructions
    of the same deletion whose scores can disagree.
    """
    return BacterialDeletionPerturbation(
        systematic_gene_name=locus_tag,
        perturbed_gene_name=symbol,
        gene_namespace=MG1655_NAMESPACE,
        collection=RECIPIENT_COLLECTION,
        cassette=RECIPIENT_CASSETTE,
        construction=StrainConstruction(batch=version),
    )


def pair_genotype(
    query_tag: str,
    query_gene: str,
    recipient_tag: str,
    recipient_gene: str,
    version: str,
) -> Genotype:
    """The double mutant of one matrix cell: the query allele and one recipient isolate."""
    return Genotype(
        perturbations=[
            query_perturbation(query_tag, query_gene),
            recipient_perturbation(recipient_tag, recipient_gene, version),
        ]
    )


def phenotype(score: float) -> GeneInteractionPhenotype:
    """One released S score of the unfiltered matrix."""
    return GeneInteractionPhenotype(
        gene_interaction=score,
        gene_interaction_p_value=None,
        screen_id=None,
        provenance_gaps=[P_VALUE_GAP],
    )


def reference_phenotype() -> GeneInteractionPhenotype:
    """The unperturbed chassis of the same screen: no epistasis, a score of 0."""
    return GeneInteractionPhenotype(
        gene_interaction=0.0,
        gene_interaction_p_value=None,
        screen_id=None,
        provenance_gaps=[P_VALUE_GAP],
    )


def chassis_background() -> BacterialStrainBackground:
    """The conjugant chassis: an Hfr C query crossed into a Keio (BW25113) recipient."""
    return BacterialStrainBackground(
        name="Hfr Cavalli x Keio (K-12 BW25113) eSGA conjugant",
        reference_strain=REFERENCE_STRAIN_NAME,
        assembly_set=MG1655_ASSEMBLY_SET,
        parents=["Hfr Cavalli", "Keio collection (K-12 BW25113)"],
        construction="the query's cat-marked deletion is transferred from the "
        "hyper-recombinant Hfr Cavalli donor into the F- Keio recipient by conjugation "
        "and homologous recombination, and the conjugant is selected twice on LB with "
        "kanamycin and chloramphenicol",
        genotype_statement=None,
        alleles=[],
        provenance=[
            SOURCED_VALUES["conjugation"],
            SOURCED_VALUES["markers"],
            SOURCED_VALUES["recipient_array"],
        ],
        provenance_gaps=[
            ProvenanceGap(
                field="genotype_statement",
                reason=ProvenanceGapReason.not_reported_by_primary,
                looked_in=Provenance(
                    source_uri=METHODS_MD,
                    citation_key=CITATION_KEY,
                    sha256=METHODS_MD_SHA256,
                    method="full Supplementary Methods read",
                    page="Supplementary Methods, query mutant construction and the "
                    "genome-wide screening procedure",
                ),
                note="the methods name Hfr Cavalli, the lambda-Red cassette of DY330 "
                "and the Keio collection but write no genotype string for either "
                "parent and none for the conjugant",
            )
        ],
    )


def reference_genome(data_root: str | None = None) -> AssemblyReferenceGenome:
    """MG1655 pinned to its GenBank assembly, carrying the conjugant chassis."""
    return assembly_reference(
        REFERENCE_STRAIN_NAME, background=chassis_background(), data_root=data_root
    )


def build_experiment(
    dataset_name: str,
    query_tag: str,
    query_gene: str,
    recipient_tag: str,
    recipient_gene: str,
    version: str,
    score: float,
) -> BacterialGeneInteractionExperiment:
    """The record of one matrix cell."""
    return BacterialGeneInteractionExperiment(
        dataset_name=dataset_name,
        genotype=pair_genotype(
            query_tag, query_gene, recipient_tag, recipient_gene, version
        ),
        environment=SCREEN_ENVIRONMENT,
        phenotype=phenotype(score),
    )


def build_reference(
    dataset_name: str, genome: AssemblyReferenceGenome
) -> BacterialGeneInteractionExperimentReference:
    """The unperturbed chassis of the screen, scoring 0."""
    return BacterialGeneInteractionExperimentReference(
        dataset_name=dataset_name,
        genome_reference=genome,
        environment_reference=SCREEN_ENVIRONMENT.model_copy(),
        phenotype_reference=reference_phenotype(),
    )


# --------------------------------------------------------------------------- #
# The partition against the served Babu 2014 store
# --------------------------------------------------------------------------- #
#: The served store's records tagged ``screen_id="Butland et al."``, measured on the dev
#: tree on 2026-10-08. A drift in either store moves this and stops the build.
SERVED_BUTLAND_RECORDS: Final = 727
#: Those records as oriented (query, recipient) gene pairs: no two of the 727 share one.
SERVED_BUTLAND_PAIRS: Final = 727
#: Cells of this release dropped because their oriented pair is one of those served. It
#: is not 2 x 725 because two of the shared pairs have only one isolate row.
SERVED_OVERLAP_CELLS: Final = 1448
#: The served pairs this release does NOT carry under those names, pinned by name
#: because the reverse direction of the partition is a measurement, not an assumption.
#: This release names both recipients ``b4344`` (gene name ``*``, absent from Genobase
#: ver. 6), which the pinned annotation carries as a synonym of ``b4486``, so rule 3
#: drops them before the partition is taken.
SERVED_PAIRS_NOT_IN_THIS_RELEASE: Final[tuple[str, ...]] = (
    "b2528 -> b4486",
    "b2531 -> b4486",
)


def read_served_babu(served_root: str) -> tuple[dict[tuple[str, str], str], int]:
    """Stream the served Babu 2014 LMDB once: its oriented pairs and its record count.

    The generator is fully consumed, so ``stream_records`` closes the environment before
    this returns; a held handle makes the next open of the same path fail with "already
    open in this process", and that surfaces only in a full-suite run.
    """
    from torchcell.verification.runners import stream_records

    pairs: dict[tuple[str, str], str] = {}
    records = 0
    for record in stream_records(served_root):
        experiment = record["experiment"]
        by_cassette = {
            str(leaf["cassette"]): str(leaf["systematic_gene_name"])
            for leaf in experiment["genotype"]["perturbations"]
        }
        pairs[(by_cassette[QUERY_CASSETTE], by_cassette[RECIPIENT_CASSETTE])] = str(
            experiment["phenotype"]["screen_id"]
        )
        records += 1
    return pairs, records


def assert_served_partition(
    served_root: str,
    served: Mapping[tuple[str, str], str],
    served_records: int,
    stored_pairs: Sequence[tuple[str, str]],
    release_pairs: Sequence[tuple[str, str]],
    *,
    stored_records: int,
    released_cells: int,
    overlap_cells: int,
) -> ServedPartition:
    """Prove the partition against the served store, in both directions.

    FORWARD: not one stored oriented pair is a pair the served store holds, so nothing
    enters the graph twice. REVERSE: every served record tagged with this screen IS a
    pair of this release, except the ones pinned in
    :data:`SERVED_PAIRS_NOT_IN_THIS_RELEASE`, so the overlap is accounted for rather
    than assumed and a drift in either store raises.
    """
    stored = set(stored_pairs)
    release = set(release_pairs)
    shared = sorted(stored & set(served))
    if shared:
        raise RuntimeError(
            f"{len(shared)} oriented pairs are already served by Babu 2014, so storing "
            f"them would duplicate a record: {shared[:5]}"
        )
    butland = {pair for pair, screen in served.items() if screen == BABU_SCREEN_TAG}
    if len(butland) != SERVED_BUTLAND_PAIRS:
        raise RuntimeError(
            f"{served_root} holds {len(butland)} oriented pairs under screen_id "
            f"{BABU_SCREEN_TAG!r}, this build was measured against "
            f"{SERVED_BUTLAND_PAIRS}"
        )
    outside = tuple(
        f"{query} -> {recipient}"
        for query, recipient in sorted(butland)
        if (query, recipient) not in release
    )
    return ServedPartition(
        served_root=served_root,
        served_records=served_records,
        served_butland_records=len(butland),
        served_butland_pairs=len(butland),
        served_fraction_of_this_release=len(butland) / released_cells,
        stored_records=stored_records,
        stored_pairs=len(stored),
        shared_pairs=0,
        overlap_cells_dropped=overlap_cells,
        served_pairs_not_in_this_release=list(outside),
    )


# --------------------------------------------------------------------------- #
# Retention
# --------------------------------------------------------------------------- #
class StoredCell(BaseModel):
    """One cell of the matrix as the record builder's arguments."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_tag: str
    query_gene: str
    recipient_tag: str
    recipient_gene: str
    version: str
    score: float
    n_samples: int
    replicate_screens: int


class Retention(BaseModel):
    """Which matrix cells are stored, and the ledgers that explain the rest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cells: list[StoredCell]
    drop_log: DropLog
    reconciliation: LocusTagReconciliation
    replicate_structure: ReplicateStructure
    gene_names: GeneNameLedger
    partition: ServedPartition


def _cell_label(query_tag: str, recipient_tag: str, version: str) -> str:
    """``<query b-number> -> <recipient b-number> (<strain version>)``."""
    return f"{query_tag} -> {recipient_tag} ({version})"


def retain(
    scores: Matrix,
    *,
    dataset_name: str,
    genome: EcoliK12Genome,
    query_roster: Mapping[str, str],
    colonies: Any,
    zero_colonies: Any,
    screens: Any,
    served: Mapping[tuple[str, str], str],
    served_records: int,
    served_root: str,
) -> Retention:
    """Apply the six retention rules in order, then prove the partition."""
    if tuple(sorted(scores.query_tags)) != tuple(sorted(query_roster)):
        raise ReleaseContentError(
            "the matrix's query columns are not Supplementary Table 2's query roster"
        )
    block = score_block(scores)
    names = pd.Series(sorted(set(scores.query_tags) | set(scores.recipient_tags)))
    _, reconciliation = reconcile_locus_tags(
        genome, names, label=f"{dataset_name} Table 4 query and recipient b-numbers"
    )
    resolutions = {name: genome.resolve_gene_name(name) for name in names}
    not_a_tag = {
        name
        for name, resolved in resolutions.items()
        if resolved.status in (GeneNameStatus.RETIRED, GeneNameStatus.AMBIGUOUS)
    }
    remapped = {
        name: str(resolved.systematic_name)
        for name, resolved in resolutions.items()
        if name not in not_a_tag
        and resolved.systematic_name is not None
        and resolved.systematic_name != name
    }
    bad_queries = sorted((not_a_tag | set(remapped)) & set(scores.query_tags))
    if bad_queries:
        raise ReleaseContentError(
            "every query strain of this release is a locus tag of the pinned "
            f"annotation in its own right, but {bad_queries} are not"
        )
    roster_names: dict[str, str] = {}
    unnamed = 0
    for row, tag in enumerate(scores.recipient_tags):
        name = scores.recipient_genes[row]
        if name == UNNAMED_IN_GENOBASE:
            unnamed += 1
            continue
        held = roster_names.setdefault(tag, name)
        if held != name:
            raise ReleaseContentError(
                f"the roster gives locus {tag} two names, {held!r} and {name!r}, so "
                "the stored common name of one gene would depend on the row"
            )
    stored_unnamed = sorted(
        tag
        for row, tag in enumerate(scores.recipient_tags)
        if scores.recipient_genes[row] == UNNAMED_IN_GENOBASE
        and scores.labels[row] == LABEL_NON_ESSENTIAL
        and tag not in not_a_tag
        and tag not in remapped
    )
    if stored_unnamed:
        raise ReleaseContentError(
            f"{len(stored_unnamed)} rows the identifier rules keep carry no gene name "
            f"({UNNAMED_IN_GENOBASE!r}), so a record would store one: {stored_unnamed[:5]}"
        )
    query_names = {
        tag: roster_names.get(tag, scores.query_genes[column])
        for column, tag in enumerate(scores.query_tags)
    }
    disagreements = {
        tag: {
            "query_header": scores.query_genes[column],
            "array_roster": query_names[tag],
        }
        for column, tag in enumerate(scores.query_tags)
        if query_names[tag] != scores.query_genes[column]
    }

    dropped: dict[str, list[str]] = {rule: [] for rule in DROP_RULES}
    cells: list[StoredCell] = []
    sign = Counter[str]()
    isolates = Counter[str]()
    sample_counts = Counter[int]()
    screen_counts = Counter[int]()
    stored_pairs: list[tuple[str, str]] = []
    for row, recipient_tag in enumerate(scores.recipient_tags):
        label = scores.labels[row]
        version = scores.versions[row]
        recipient_gene = scores.recipient_genes[row]
        for column, query_tag in enumerate(scores.query_tags):
            item = _cell_label(query_tag, recipient_tag, version)
            if label != LABEL_NON_ESSENTIAL:
                dropped[RULE_SPA_TAG].append(item)
                continue
            if recipient_tag in not_a_tag:
                dropped[RULE_NOT_A_TAG].append(item)
                continue
            if recipient_tag in remapped:
                dropped[RULE_REMAPPED].append(item)
                continue
            if recipient_tag == query_tag:
                dropped[RULE_SELF_PAIR].append(item)
                continue
            score = float(block[row, column])
            if colonies[row, column] == zero_colonies[row, column] and score == 0.0:
                dropped[RULE_CONTRADICTION].append(item)
                continue
            if (query_tag, recipient_tag) in served:
                dropped[RULE_SERVED].append(item)
                continue
            cells.append(
                StoredCell(
                    query_tag=query_tag,
                    query_gene=query_names[query_tag],
                    recipient_tag=recipient_tag,
                    recipient_gene=recipient_gene,
                    version=version,
                    score=score,
                    n_samples=int(colonies[row, column]),
                    replicate_screens=int(screens[row, column]),
                )
            )
            stored_pairs.append((query_tag, recipient_tag))
            sign["neg" if score < 0 else ("pos" if score > 0 else "zero")] += 1
            isolates[version] += 1
            sample_counts[int(colonies[row, column])] += 1
            screen_counts[int(screens[row, column])] += 1

    released_cells = len(scores.recipient_tags) * len(scores.query_tags)
    release_pairs = [
        (query_tag, recipient_tag)
        for row, recipient_tag in enumerate(scores.recipient_tags)
        if scores.labels[row] == LABEL_NON_ESSENTIAL
        and recipient_tag not in not_a_tag
        and recipient_tag not in remapped
        for query_tag in scores.query_tags
        if query_tag != recipient_tag
    ]
    rules = [
        DropRule(
            rule=rule,
            description=DROP_RULE_DESCRIPTIONS[rule],
            n_records=len(dropped[rule]),
            items=sorted(dropped[rule]),
        )
        for rule in DROP_RULES
    ]
    total_dropped = sum(rule.n_records for rule in rules)
    if total_dropped + len(cells) != released_cells:
        raise RuntimeError("drop rules do not account for every released cell")
    drop_log = DropLog(
        dataset=dataset_name,
        source_records=released_cells,
        kept_records=len(cells),
        dropped_records=total_dropped,
        source_queries=len(set(scores.query_tags)),
        source_recipients=len(set(scores.recipient_tags)),
        kept_queries=len({cell.query_tag for cell in cells}),
        kept_recipients=len({cell.recipient_tag for cell in cells}),
        kept_isolates=dict(sorted(isolates.items())),
        n_aggravating=sign["neg"],
        n_alleviating=sign["pos"],
        n_zero=sign["zero"],
        rules=rules,
    )
    partition = assert_served_partition(
        served_root,
        served,
        served_records,
        stored_pairs,
        release_pairs,
        stored_records=len(cells),
        released_cells=released_cells,
        overlap_cells=len(dropped[RULE_SERVED]),
    )
    if partition.overlap_cells_dropped != SERVED_OVERLAP_CELLS:
        raise RuntimeError(
            f"{partition.overlap_cells_dropped} cells overlap the served store, this "
            f"build was measured against {SERVED_OVERLAP_CELLS}"
        )
    if tuple(partition.served_pairs_not_in_this_release) != (
        SERVED_PAIRS_NOT_IN_THIS_RELEASE
    ):
        raise RuntimeError(
            "the served pairs this release does not carry are "
            f"{partition.served_pairs_not_in_this_release}, measured "
            f"{list(SERVED_PAIRS_NOT_IN_THIS_RELEASE)}"
        )
    return Retention(
        cells=cells,
        drop_log=drop_log,
        reconciliation=reconciliation,
        replicate_structure=_replicate_structure(sample_counts, screen_counts),
        gene_names=GeneNameLedger(
            rule=RULE_ROSTER_NAME_WINS,
            disagreements=disagreements,
            n_records_renamed=sum(
                1 for cell in cells if cell.query_tag in disagreements
            ),
            unnamed_rows_dropped=unnamed,
            unnamed_note="the roster writes '*' for a strain whose gene name Genobase "
            "ver. 6 does not carry (footnote of every sheet). Every such row is already "
            "dropped by an identifier rule, which is asserted rather than assumed, so "
            "no record stores '*' as a common name",
        ),
        partition=partition,
    )


def _replicate_structure(
    sample_counts: Mapping[int, int], screen_counts: Mapping[int, int]
) -> ReplicateStructure:
    """The measured colony design of the stored records, with its verbatim source."""
    sourced = SOURCED_VALUES["replicate_design"]
    modal, at_modal = max(sample_counts.items(), key=lambda item: item[1])
    return ReplicateStructure(
        sample_unit="colony",
        documented_n_samples=int(sourced.value["colonies"]),
        documented_colonies_per_isolate=int(sourced.value["colonies_per_isolate"]),
        documented_replicate_screens=int(sourced.value["replicate_screens"]),
        measured_n_samples_counts={
            str(key): value for key, value in sorted(sample_counts.items())
        },
        measured_replicate_screen_counts={
            str(key): value for key, value in sorted(screen_counts.items())
        },
        modal_n_samples=modal,
        records_at_modal_n_samples=at_modal,
        quote=str(sourced.quote),
        citation_key=CITATION_KEY,
        source_uri=str(sourced.provenance.source_uri),
        sha256=str(sourced.provenance.sha256),
        note="the documented design is four colonies -- two of this isolate in each of "
        "two replicate screens -- and it is the mode rather than the rule, so each "
        "record's own colony measurements are counted off the raw sheet instead. "
        "GeneInteractionPhenotype has no n_samples or sample_unit field and it is a "
        "SERVED graph class, so the design is recorded here rather than forced onto the "
        "record (issue #793)",
    )


def high_confidence_ledger(
    table_s3: pd.DataFrame, scores: Matrix, stored: Sequence[StoredCell]
) -> HighConfidenceLedger:
    """Prove Table 3 is a subset of the matrix, and list the stored cells it selects.

    Every row must be locatable at its own (query, recipient, strain version) cell and
    must carry the same score, which is what makes Table 3 a selection rather than a
    second release. A row that is not raises, because a Table 3 the matrix does not
    contain would mean the two files are not one screen.
    """
    block = score_block(scores)
    index = {
        (tag, version): row
        for row, (tag, version) in enumerate(
            zip(scores.recipient_tags, scores.versions, strict=True)
        )
    }
    column_of = {tag: column for column, tag in enumerate(scores.query_tags)}
    stored_keys = {
        (cell.query_tag, cell.recipient_tag, cell.version) for cell in stored
    }
    located = identical = 0
    selected: list[str] = []
    for row in table_s3.itertuples(index=False):
        key = (str(row.recipient_tag), str(row.version))
        if key not in index or str(row.query_tag) not in column_of:
            raise ReleaseContentError(
                f"{TABLE_S3} row {row.query_tag} -> {row.recipient_tag} ({row.version}) "
                "is not a cell of the S-score matrix"
            )
        located += 1
        cell = float(block[index[key], column_of[str(row.query_tag)]])
        released = float(str(row.s_score))
        if cell != released:
            raise ReleaseContentError(
                f"{TABLE_S3} row {row.query_tag} -> {row.recipient_tag} scores "
                f"{released}, the matrix cell reads {cell}"
            )
        identical += 1
        label = _cell_label(
            str(row.query_tag), str(row.recipient_tag), str(row.version)
        )
        if (
            str(row.query_tag),
            str(row.recipient_tag),
            str(row.version),
        ) in stored_keys:
            selected.append(label)
    ordered = table_s3.drop_duplicates(["query_tag", "recipient_tag"])
    essentiality = Counter(ordered["essentiality"])
    return HighConfidenceLedger(
        rows=len(table_s3),
        ordered_pairs=len(ordered),
        located_in_matrix=located,
        identical_score=identical,
        non_essential_pairs=essentiality[S3_NON_ESSENTIAL],
        spa_tag_pairs=essentiality[LABEL_SPA_TAG],
        stored_rows=len(selected),
        not_stored_rows=len(table_s3) - len(selected),
        z_cutoff_note="the |Z|>=4 (P<0.0001) selection of Supplementary Table 3. "
        "GeneInteractionPhenotype has no slot for a per-cell |Z| or for a significance "
        "flag, and converting |Z| to a p-value would be a derivation this release never "
        "published, so the selection is recorded here rather than on the record",
        pairs=sorted(selected),
    )


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
DATASET_ROOT_REL = "data/torchcell/gene_interaction_butland2008"
#: Records of the full build: 314,847 released cells minus 5,811 SPA-tag recipient
#: cells, 6,318 on an id the annotation does not carry, 4,407 on an annotation-remapped
#: b-number, 78 self pairs, 395 whose score contradicts their own raw colonies and 1,448
#: whose oriented pair the served Babu 2014 store already holds.
EXPECTED_RECORDS = 296390


@register_dataset
class GeneInteractionButland2008Dataset(ExperimentDataset):
    """Unfiltered eSGA interaction scores of E. coli K-12 double deletion mutants."""

    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = REFERENCE_STRAIN_NAME

    def __init__(
        self,
        root: str = "data/torchcell/gene_interaction_butland2008",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        ecoli_genome: EcoliK12Genome | None = None,
        served_root: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; ``served_root`` points the partition at the Babu 2014 store."""
        self.ecoli_genome = ecoli_genome
        self.served_root = served_root
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return BacterialGeneInteractionExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return BacterialGeneInteractionExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Every consumed workbook, linked from this key's raw mirror."""
        return [raw.name for raw in RAW_FILES]

    def download(self) -> None:
        """Link each mirror file into ``raw/`` after checking its manifest and sha256.

        Every file is found and pinned BEFORE ``raw/`` is populated, so an incomplete
        mirror leaves no half-populated raw directory behind.
        """
        data_root = _data_root()
        manifest = load_manifest(data_root)
        mirror = raw_mirror_dir(data_root)
        found: dict[str, Path] = {}
        for raw in RAW_FILES:
            check_manifest_pin(
                raw.mirror_relpath,
                manifest_sha256(manifest, raw.mirror_relpath),
                raw.sha256,
            )
            src = mirror / raw.mirror_relpath
            if not src.exists():
                raise RuntimeError(f"required raw artifact missing from mirror: {src}")
            found[raw.name] = src
        os.makedirs(self.raw_dir, exist_ok=True)
        for raw in RAW_FILES:
            link_verified(found[raw.name], osp.join(self.raw_dir, raw.name), raw.sha256)
        log.info(
            "Butland 2008 raw files linked into %s (sha256 verified)", self.raw_dir
        )

    def _genome(self) -> EcoliK12Genome:
        """The injected genome, or the reference strain's default cache."""
        if self.ecoli_genome is None:
            self.ecoli_genome = bacterial_genome("ecoli", self.REFERENCE_STRAIN)
        expected = BACTERIAL_ASSEMBLY_SETS[self.REFERENCE_STRAIN]
        if self.ecoli_genome.ASSEMBLY_SET != expected:
            raise ValueError(
                f"{type(self).__name__} needs the {expected} genome, got "
                f"{self.ecoli_genome.ASSEMBLY_SET}"
            )
        return self.ecoli_genome

    def _served_root(self) -> str:
        """Where the already-served Babu 2014 store lives (dev tree by default)."""
        return self.served_root or osp.join(_data_root(), BABU_ROOT_REL)

    @post_process
    def process(self) -> None:
        """Parse the S-score matrix into one record per retained cell, plus the ledgers."""
        verify_raw_files(self.raw_dir, DATA_SHA256)
        matrix_path = osp.join(self.raw_dir, TABLE_S4)
        scores = read_s_scores(matrix_path)
        raw_sizes = read_matrix_sheet(matrix_path, SHEET_RAW)
        if sorted(raw_sizes.query_tags) != sorted(scores.query_tags):
            raise SheetFormatError(
                "the raw sheet's query columns are not the S sheet's query columns"
            )
        order = [raw_sizes.query_tags.index(tag) for tag in scores.query_tags]
        colonies, zero_colonies, screens = colony_counts(raw_sizes)
        colonies, zero_colonies, screens = (
            colonies[:, order],
            zero_colonies[:, order],
            screens[:, order],
        )
        query_roster = read_query_roster(osp.join(self.raw_dir, TABLE_S2))
        roster = read_array_roster(osp.join(self.raw_dir, TABLE_S1))
        matrix_rows = dict(
            zip(
                zip(scores.recipient_tags, scores.versions, strict=True),
                scores.labels,
                strict=True,
            )
        )
        if matrix_rows != roster:
            raise ReleaseContentError(
                "the S-score matrix's recipient rows are not Supplementary Table 1's "
                "array roster"
            )
        served_root = self._served_root()
        served, served_records = read_served_babu(served_root)
        retention = retain(
            scores,
            dataset_name=self.name,
            genome=self._genome(),
            query_roster=query_roster,
            colonies=colonies,
            zero_colonies=zero_colonies,
            screens=screens,
            served=served,
            served_records=served_records,
            served_root=served_root,
        )
        drop_log = retention.drop_log
        log.info(
            "Butland 2008: %d released cells -> %d records (%d aggravating, %d "
            "alleviating, %d zero); dropped %s",
            drop_log.source_records,
            drop_log.kept_records,
            drop_log.n_aggravating,
            drop_log.n_alleviating,
            drop_log.n_zero,
            {rule.rule: rule.n_records for rule in drop_log.rules},
        )

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        ledger = high_confidence_ledger(
            read_high_confidence(osp.join(self.raw_dir, TABLE_S3)),
            scores,
            retention.cells,
        )
        self._write_ledgers(retention, ledger)

        reference = build_reference(self.name, reference_genome())
        env_out, interned_env = self._open_write_lmdb(
            osp.join(self.processed_dir, "lmdb")
        )
        index = 0
        with env_out.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for cell in tqdm(
                retention.cells, total=drop_log.kept_records, desc="butland2008"
            ):
                experiment = build_experiment(
                    self.name,
                    cell.query_tag,
                    cell.query_gene,
                    cell.recipient_tag,
                    cell.recipient_gene,
                    cell.version,
                    cell.score,
                )
                txn.put(
                    f"{index}".encode(),
                    self._intern_record(experiment, reference, PUBLICATION, itxn),
                )
                index += 1
        env_out.close()
        interned_env.close()
        if index != drop_log.kept_records:
            raise RuntimeError(
                f"wrote {index} records, the ledger counted {drop_log.kept_records}"
            )
        log.info("Wrote %d Butland 2008 gene-interaction experiments to LMDB", index)

    def _write_ledgers(
        self, retention: Retention, ledger: HighConfidenceLedger
    ) -> None:
        """The drop log, the identifier reconciliation, the measured replicate design,
        the partition proof, the high-confidence selection and the files not loaded.
        """
        out = Path(self.preprocess_dir)
        (out / "dropped_records.json").write_text(
            retention.drop_log.model_dump_json(indent=2)
        )
        (out / "identifier_reconciliation.json").write_text(
            retention.reconciliation.model_dump_json(indent=2)
        )
        (out / "replicate_structure.json").write_text(
            retention.replicate_structure.model_dump_json(indent=2)
        )
        (out / "gene_name_disagreements.json").write_text(
            retention.gene_names.model_dump_json(indent=2)
        )
        (out / "served_partition.json").write_text(
            retention.partition.model_dump_json(indent=2)
        )
        (out / "high_confidence_pairs.json").write_text(
            ledger.model_dump_json(indent=2)
        )
        (out / "files_not_loaded.json").write_text(
            json.dumps(list(NOT_LOADED), indent=2)
        )

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled by ``build_experiment``."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Verification (L0-L4) of a built tree
# --------------------------------------------------------------------------- #
Record = Mapping[str, Any]

VERIFIER_PROVENANCE = Provenance(
    source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/data/{TABLE_S4}",
    citation_key=CITATION_KEY,
    sha256=DATA_SHA256[TABLE_S4],
    method="Supplementary Table 4, sheet 'S scores': the interaction score of every "
    "(query strain, recipient strain) cell of the unfiltered 39 x 8,073 matrix. One "
    "BacterialGeneInteractionExperiment per cell, a two-leaf "
    "BacterialDeletionPerturbation genotype against MG1655 whose query carries cat and "
    "whose recipient carries kan and its isolate token, reference = the unperturbed "
    "conjugant chassis at 0",
    page="Supplementary Table 4 (41592_2008_BFnmeth1239_MOESM309_ESM.xls), sheet "
    "'S scores'",
    retrieved=DATA_RETRIEVED_AT,
)


def _validate_record(record: Record) -> None:
    """L0: the experiment and its reference validate as the assembly-pinned pair."""
    BacterialGeneInteractionExperiment.model_validate(record["experiment"])
    BacterialGeneInteractionExperimentReference.model_validate(record["reference"])


def _l1_two_distinct_genes(records: Sequence[Record]) -> LevelResult:
    """L1: every record is a digenic pair of two DIFFERENT loci, one query one recipient."""
    bad: list[str] = []
    for record in records:
        perturbations = record["experiment"]["genotype"]["perturbations"]
        tags = [p["systematic_gene_name"] for p in perturbations]
        cassettes = sorted(str(p["cassette"]) for p in perturbations)
        if (
            len(perturbations) != 2
            or len(set(tags)) != 2
            or cassettes != sorted((QUERY_CASSETTE, RECIPIENT_CASSETTE))
        ):
            bad.append("+".join(tags))
    return LevelResult(
        level=Level.L1,
        name="digenic_pair_of_one_query_and_one_recipient",
        passed=bool(records) and not bad,
        message=f"{len(records) - len(bad)} of {len(records)} records are two distinct "
        "loci, one cat-marked query and one kan-marked recipient",
        details={"n_bad": len(bad), "bad_examples": bad[:20]},
    )


def _l1_isolate_is_on_the_recipient(records: Sequence[Record]) -> LevelResult:
    """L1: the recipient leaf carries its strain version and the query leaf carries none.

    The isolate is a property of the arrayed recipient strain, and it is the only thing
    distinguishing the two records of one gene pair, so a record missing it would be an
    unexplained duplicate of its sibling.
    """
    versions = Counter[str]()
    bad: list[str] = []
    for record in records:
        for perturbation in record["experiment"]["genotype"]["perturbations"]:
            construction = perturbation.get("construction")
            batch = None if construction is None else construction.get("batch")
            if perturbation["cassette"] == RECIPIENT_CASSETTE:
                if batch is None:
                    bad.append(str(perturbation["systematic_gene_name"]))
                else:
                    versions[str(batch)] += 1
            elif batch is not None:
                bad.append(str(perturbation["systematic_gene_name"]))
    return LevelResult(
        level=Level.L1,
        name="recipient_leaf_carries_its_keio_isolate",
        passed=bool(records) and not bad and len(versions) > 1,
        message=f"{sum(versions.values())} recipient leaves carry a strain version "
        f"{dict(sorted(versions.items()))}; {len(bad)} leaves are wrong",
        details={"versions": dict(sorted(versions.items())), "bad_examples": bad[:20]},
    )


def _l2_scores_match_the_matrix(
    records: Sequence[Record], released: Mapping[tuple[str, str, str], float]
) -> LevelResult:
    """L2: every stored score is the released cell of its own (query, recipient, isolate)."""
    mismatched: list[str] = []
    for record in records:
        experiment = record["experiment"]
        by_cassette = {
            str(p["cassette"]): p for p in experiment["genotype"]["perturbations"]
        }
        recipient = by_cassette[RECIPIENT_CASSETTE]
        key = (
            str(by_cassette[QUERY_CASSETTE]["systematic_gene_name"]),
            str(recipient["systematic_gene_name"]),
            str(recipient["construction"]["batch"]),
        )
        stored = float(experiment["phenotype"]["gene_interaction"])
        if key not in released or released[key] != stored:
            mismatched.append(f"{key} stored {stored}")
    return LevelResult(
        level=Level.L2,
        name="gene_interaction_equals_its_matrix_cell",
        passed=bool(records) and not mismatched,
        message=f"{len(records) - len(mismatched)} of {len(records)} stored scores "
        "equal their Supplementary Table 4 cell",
        details={"n_mismatched": len(mismatched), "examples": mismatched[:20]},
    )


def _l3_sign_convention(records: Sequence[Record]) -> LevelResult:
    """L3: the stored axis is signed and unclamped, with a zero reference.

    Unlike the released high-confidence tail this matrix is unfiltered, so an exact zero
    is an ordinary value of it and the check asserts both signs rather than no zeros.
    """
    scores = [
        float(record["experiment"]["phenotype"]["gene_interaction"])
        for record in records
    ]
    negative = sum(1 for score in scores if score < 0)
    positive = sum(1 for score in scores if score > 0)
    zero = sum(1 for score in scores if score == 0)
    references = {
        float(record["reference"]["phenotype_reference"]["gene_interaction"])
        for record in records
    }
    return LevelResult(
        level=Level.L3,
        name="signed_unclamped_interaction_score_with_zero_reference",
        passed=negative > 0 and positive > 0 and references == {0.0},
        message=f"{negative} aggravating and {positive} alleviating stored scores, "
        f"{zero} at exactly zero, reference scores {sorted(references)}",
        details={
            "n_negative": negative,
            "n_positive": positive,
            "n_zero": zero,
            "reference_scores": sorted(references),
        },
    )


def _l3_partition_against_babu(
    records: Sequence[Record], served: Mapping[tuple[str, str], str]
) -> LevelResult:
    """L3: no stored record's oriented gene pair is one the served Babu store holds."""
    shared: list[str] = []
    for record in records:
        by_cassette = {
            str(p["cassette"]): str(p["systematic_gene_name"])
            for p in record["experiment"]["genotype"]["perturbations"]
        }
        pair = (by_cassette[QUERY_CASSETTE], by_cassette[RECIPIENT_CASSETTE])
        if pair in served:
            shared.append(f"{pair[0]} -> {pair[1]}")
    return LevelResult(
        level=Level.L3,
        name="partitioned_from_the_served_babu2014_store",
        passed=bool(records) and not shared,
        message=f"{len(records)} stored records against {len(served)} served oriented "
        f"pairs; {len(shared)} share one",
        details={"n_shared": len(shared), "examples": shared[:20]},
    )


def _l4_containment(records: Sequence[Record], universe: set[str]) -> LevelResult:
    """L4: every perturbed locus is a GenBank locus of the pinned MG1655 assembly."""
    perturbed = {
        p["systematic_gene_name"]
        for record in records
        for p in record["experiment"]["genotype"]["perturbations"]
    }
    missing = sorted(perturbed - universe)
    return LevelResult(
        level=Level.L4,
        name="gene_containment_mg1655_locus_tags",
        passed=bool(perturbed) and not missing,
        message=f"{len(perturbed) - len(missing)} of {len(perturbed)} perturbed loci "
        "are MG1655 GenBank loci",
        details={
            "n_perturbed": len(perturbed),
            "n_universe": len(universe),
            "missing_examples": missing[:20],
        },
    )


def verify_records(
    records: Sequence[Record],
    *,
    released: Mapping[tuple[str, str, str], float],
    served: Mapping[tuple[str, str], str],
    universe: set[str],
    expected_count: int,
    dataset_name: str = "gene_interaction_butland2008",
) -> VerificationReport:
    """The L0-L4 gate over built records, given the re-read matrix, the served store's
    oriented pairs and the MG1655 gene universe.
    """
    from torchcell.verification.common import shared_rule_results

    report = VerificationReport(
        dataset_name=dataset_name, provenance=VERIFIER_PROVENANCE
    )
    report.add(l0_structural(records, _validate_record))
    report.add(l1_count(len(records), expected_count))
    report.add(_l1_two_distinct_genes(records))
    report.add(_l1_isolate_is_on_the_recipient(records))
    report.add(_l2_scores_match_the_matrix(records, released))
    report.add(_l3_sign_convention(records))
    report.add(_l3_partition_against_babu(records, served))
    for result in shared_rule_results(records):
        report.add(result)
    report.add(_l4_containment(records, universe))
    return report


def released_scores(path: str | Path) -> dict[tuple[str, str, str], float]:
    """The matrix re-read as ``(query, recipient, strain version) -> S score``."""
    scores = read_s_scores(path)
    block = score_block(scores)
    return {
        (query, recipient, version): float(block[row, column])
        for row, (recipient, version) in enumerate(
            zip(scores.recipient_tags, scores.versions, strict=True)
        )
        for column, query in enumerate(scores.query_tags)
    }


def _gene_universe(records: Sequence[Record], base: str) -> set[str]:
    """The L4 universe: every GenBank locus of the assembly the records themselves pin."""
    from torchcell.verification.runners import _gene_set_for_reference

    references = {
        json.dumps(record["reference"]["genome_reference"], sort_keys=True)
        for record in records
    }
    universe: set[str] = set()
    for reference in references:
        universe |= _gene_set_for_reference(json.loads(reference), base)
    return universe


def run_verification(data_root: str | None = None) -> VerificationReport:
    """Verify the built dev-tree LMDB (L0-L4) plus the provenance audit of every
    ``SOURCED_VALUES`` entry, and write ``preprocess/verification_report.json``.
    """
    from torchcell.verification.runners import _write_report, load_records

    base = data_root or _data_root()
    abs_root = osp.join(base, DATASET_ROOT_REL)
    records = load_records(abs_root)
    drops = DropLog.model_validate_json(
        Path(abs_root, "preprocess", "dropped_records.json").read_text()
    )
    released = released_scores(osp.join(abs_root, "raw", TABLE_S4))
    served, _ = read_served_babu(osp.join(base, BABU_ROOT_REL))
    report = verify_records(
        records,
        released=released,
        served=served,
        universe=_gene_universe(records, base),
        expected_count=drops.kept_records,
    )
    library_root = Path(base) / "torchcell-library"
    for name in TEXT_QUOTED:
        report.add(audit_sourced_value(SOURCED_VALUES[name], library_root))
    for entry in WORKBOOK_QUOTES:
        report.add(audit_workbook_quote(entry, base))
    _write_report(report, osp.join(abs_root, "preprocess"))
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI: ``deposit`` the raw mirror, ``build`` the dev LMDB, or ``verify`` it."""
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(
        prog="python -m torchcell.datasets.ecoli.butland2008"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    deposit = sub.add_parser("deposit", help="extend the raw mirror")
    deposit.add_argument(
        "--retrieve-into",
        default=None,
        help="re-run every recorded retrieval into this directory and deposit those "
        "bytes; without it the literature mirror's captured SI files are used",
    )
    sub.add_parser("build", help="build (or load) the dev-tree LMDB")
    sub.add_parser("verify", help="run L0-L4 on the built dev-tree LMDB")
    args = parser.parse_args(argv)

    load_dotenv()
    data_root = _data_root()
    if args.command == "deposit":
        sources: dict[str, str | Path]
        if args.retrieve_into is not None:
            sources = dict(retrieve_raw_files(args.retrieve_into))
        else:
            sources = {
                raw.name: library_dir(data_root) / "si" / raw.name for raw in RAW_FILES
            }
        print(deposit_raw_mirror(sources=sources, data_root=data_root))
        return 0
    if args.command == "build":
        dataset = GeneInteractionButland2008Dataset(
            root=osp.join(data_root, DATASET_ROOT_REL)
        )
        print(f"len = {len(dataset)}")
        dataset.close_lmdb()
        return 0
    report = run_verification(data_root)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
