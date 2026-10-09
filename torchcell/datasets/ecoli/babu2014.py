# torchcell/datasets/ecoli/babu2014
# [[torchcell.datasets.ecoli.babu2014]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/babu2014
# Test file: tests/torchcell/datasets/ecoli/test_babu2014.py
"""Babu 2014: the genome-wide eSGA digenic genetic-interaction map of E. coli K-12.

Babu, Arnold, Bundalovic-Torma et al. 2014 (PLoS Genetics 10(2):e1004120,
doi:10.1371/journal.pgen.1004120, PMC3930520, citation key
``babuQuantitativeGenomeWideGenetic2014``) crossed 163 Hfr-Cavalli donor query mutants
by conjugation into an arrayed F- recipient collection and scored double-mutant colony
sizes into a signed interaction score (S). Only the high-confidence tail is released:
Table S2 holds 42,705 (donor, recipient) pairs, 25,239 aggravating and 17,466
alleviating, which is the instance count the bacterial schedule row states.

RECORD = one ``BacterialGeneInteractionExperiment`` per Table S2 row: a two-gene
``Genotype`` of ``BacterialDeletionPerturbation`` leaves (the donor allele and the
recipient allele, distinguished by their ``collection`` and ``cassette``) and a
``GeneInteractionPhenotype`` holding that row's ``GI score`` verbatim.

WHICH OF THE 37 RELEASED SI FILES IS AUTHORITATIVE, AND WHY. The release is 21 PDFs
(Figures S1-S5 = ``si1``-``si5``, Protocols S1-S16 = ``si6``-``si21``) and 16
spreadsheets (Tables S1-S16 = ``si22``-``si37``); with the OCR artifacts the mirror
carries 122 SI files. **``si23.xls``, sheet ``WG_GI_Score_Mar_06_2013``, column
``GI score`` is the only authoritative source of the stored value**: it is the one
released table whose unit of observation IS a (donor, recipient) pair with its own
score, its row count is exactly the 42,705 the paper and Protocol S14 both state, and
its sign split is exactly the 25,239 / 17,466 the Results state. ``si22.xls``
(Table S1, the donor catalog) is the second build input, read for two columns the
record needs and nothing else: ``Essentiality`` (which donors are hypomorphs rather
than deletions) and ``Donors screened in this and previous study`` (which screen set a
pair came from, stored as ``screen_id``). Every other file is read-and-not-loaded; the
reasons are in :data:`NOT_LOADED`, and the two that also carry a ``GI score`` column
(Table S12 ``si33.xls`` sheet 2, Table S13 ``si34.xls``) are PROPER SUBSETS of Table S2
selected for a figure, so loading them would duplicate records.

STRAIN, AND WHY MG1655 IS THE PIN. The colony measured is a conjugant: an Hfr Cavalli
donor transfers its ``cat``-marked deletion into an F- Keio (K-12 BW25113) recipient
carrying a ``kan``-marked deletion. Every identifier the release keys on is an MG1655
b-number (``b0002__thrA``), and 3,863 of the 3,880 distinct names ARE locus tags of
GCA_000005845.2 (3,726 current genes, 1 renamed, 136 pseudogene loci). A b-number is
not derivable from a ``BW25113_`` tag by string surgery ([[plan.bacteria-ontology-genome]]
measured 45 numeric disagreements and 230 BW25113 genes absent from MG1655), and the ECK
crosswalk would make every record carry a derived mapping. So records pin
``assembly_reference("MG1655", background=...)`` and the physical chassis is carried as
a ``BacterialStrainBackground`` whose ``parents`` name Hfr Cavalli and the Keio
collection, the same shape Rapp 2026 uses for its BW25993-derived host.

PHENOTYPE, AND THE ONE THING IT STILL CANNOT HOLD. ``GeneInteractionPhenotype`` takes
the signed S score as ``gene_interaction``; it is NOT a fitness value and must never
become one, because ``FitnessPhenotype`` clamps non-positive values and 25,239 of these
42,705 numbers are negative. One sourced fact has no slot on the class:

- a per-record p-value. The paper states ``P<=0.05`` for the released set as a whole,
  not per pair, so ``gene_interaction_p_value`` carries a typed
  ``not_reported_by_primary`` gap.

``n_samples`` / ``sample_unit`` USED to be the other one. Protocol S2 states eight
colonies per pair exactly (``SOURCED_VALUES["n_samples"]``) and
``GeneInteractionPhenotype`` had no field for it, so the value lived in
``preprocess/replicate_structure.json`` and the gap was filed as #793. The class now
carries the replicate-design quartet, so every record stores ``n_samples=8`` and
``sample_unit=colony``; the json stays as the provenance copy (the verbatim quote, the
citation key, the source path and its sha256), which is what makes the field on the
record auditable rather than asserted.

``screen_id`` IS the superset record. Table S1 attributes 124 of the 163 donors to
``This Study`` and 39 to ``Butland et al.``, and Protocol S2 says the new scores were
"combined with our previously published GI datasets from 39 genome-wide screens". So
Table S2 SUBSUMES Butland 2008's screens, which is why schedule row 43 (Butland 2008)
is a provenance record rather than a second loader: every record stores its screen set
verbatim in ``screen_id``, so the 727 records carried over from the 39 earlier screens
stay distinguishable from the 37,852 of the 124 new ones.

THE HYPOMORPHS HAVE NO PERTURBATION LEAF, AND THAT IS A FILED GAP, NOT A GUESS. The
recipient array is "3,968 non-essential single gene deletions ... and 149 hypomorphic
mutant strains ... in which a Kan-R marker was integrated into the 3'-UTR", and 7 of the
163 donors are essential-gene hypomorphs (Table S1 ``Essentiality``). A 3'-end cassette
that alters transcript abundance is neither a deletion nor a CRISPRi knockdown nor a
promoter replacement nor a mapped transposon insertion, so none of the five bacterial
gene-perturbation leaves can type it; the yeast ``DampPerturbation`` is the right
concept but carries no ``gene_namespace`` and is not a bacterial leaf. Rather than type
a hypomorph as a deletion, the 3,420 pairs that involve one are dropped under
``hypomorphic_allele_has_no_bacterial_perturbation_leaf`` and the missing leaf is
filed as issue #792.

WHICH RECIPIENTS ARE HYPOMORPHS IS A DEFERRAL, FOLLOWED. Babu releases the count (149)
and not the list, deferring to refs [13,16] = Butland 2008 and Babu 2011. Babu 2011 is
not in the mirror; Butland 2008 IS (``butlandESGAColiSynthetic2008``), and its
Supplementary Table 1 (``si2.xls``, sheet ``st1``) is the array roster, stating
"Number of KEIO mutant strains = 7924 / SPA-tag essential genes = 149" and labeling each
b-number ``Non-essential`` or ``SPA-tag essential``. Its 149 ``SPA-tag essential`` rows
are exactly the 149 Babu's count names, so that file is the third build input, read from
ITS OWN raw mirror under ITS OWN citation key (the shape Rousset 2018 uses to read Cui
2018's guide table). The provenance chain is Babu Results -> refs [13,16] -> Butland
Supplementary Table 1.

RETENTION (rules, counts and items in ``preprocess/dropped_records.json``).

1. ``hypomorphic_allele_has_no_bacterial_perturbation_leaf`` -- 3,420 pairs: 1,024 with
   one of the 7 essential (hypomorphic) donors, 2,479 with one of the 149 SPA-tagged
   recipients, 83 with both.
2. ``b_number_is_not_a_locus_tag_of_the_pinned_annotation`` -- 183 pairs naming one of
   17 released ids GCA_000005845.2 does not carry at all, under any layer: ``JW5447``,
   ``JW5661``, ``cscR``, nine ``bNNNN.1`` sub-numbered Keio entries, and the five
   retired b-numbers ``b0370``, ``b0510``, ``b4091``, ``b4223``, ``b4274``, ``b4574``.
   A bacterial leaf's validator refuses the first twelve outright, and storing any of
   them would put a record on a locus the assembly does not have.
3. ``b_number_remapped_by_the_annotation`` -- 519 pairs naming one of 52 b-numbers that
   GCA_000005845.2 carries as a ``/gene_synonym`` of a DIFFERENT locus: 46 of a
   pseudogene (``b0359``->``b4579``, ``b4103``->``b4583``, ...), 5 of another pseudogene
   through the remap rule (``b0500``->``b0501``, ``b0625``->``b4581``,
   ``b1371``->``b4570``, ``b4338``->``b4584``, ``b4103``->``b4583``) and 1 of a current
   gene (``b0519``->``b4572``). Storing the merged locus needs a
   ``DerivedIdentifierMapping`` and ``DerivedIdentifierRoute`` has no member for a
   retired tag of the pinned strain's OWN namespace (issue #753), so the pairs are
   dropped rather than remapped silently. This is the same finding Girgis 2009 and Rapp
   2026 record, at this release's scale.
4. ``contradictory_duplicate_pair`` -- 4 rows, the two ordered (donor, recipient) pairs
   Table S2 releases TWICE with different scores (``b1396__paaI``/``b2863__ygeQ`` at
   -5.19537 and +4.33814; ``b2675__nrdE``/``b4462__ygaR`` at -7.12262 and -5.67666).
   The release gives no rule for choosing, and the two scores of the paaI pair disagree
   in SIGN, so both rows of both pairs are dropped.

3,420 + 183 + 519 + 4 = 4,126 dropped; 42,705 - 4,126 = 38,579 records, 22,732
aggravating and 15,847 alleviating, over 155 donors and 3,658 recipients.

RECIPROCAL PAIRS ARE KEPT, AND THEY ARE NOT DUPLICATES. 102 unordered gene pairs appear
twice because each gene was a donor in its own screen, and the two scores often disagree
in sign (``b0436__tig`` / ``b0957__ompA``: -7.63085 as donor-tig, +6.96912 as
donor-ompA). These are two different strains, not two measurements of one: in the first
``tig`` carries ``cat`` and ``ompA`` carries ``kan``, in the second the markers are
swapped, so the two ``Genotype`` objects differ and both records stand. The count and
every pair are written to ``preprocess/reciprocal_pairs.json``.

MEDIUM. One condition, so one medium: LB plates with kanamycin and chloramphenicol, the
double-drug selection the conjugants were scored on, at 32 C for 36 h. Babu names the
medium and the two drugs and prints no amount for any component, so every component
carries ``concentration=None``; the gelling agent is named by Butland 2008's own
protocol ("LB-Kan-Cm agar plate") and likewise unquantified. No dose is asserted for
either drug: Butland's SI prints kanamycin at 50 ug/mL for the array pre-culture and
25 ug/mL "throughout", which is a disagreement within one mirrored source, not a value.
``base_medium`` is the ``LB`` library key, which already carries this paper's own LB
amounts from Protocol S8 as corroboration, so the medium joins there without asserting
that the screen plates used them.

BUILD-TIME CHECKS (no fallbacks; each raises). Every sheet's title cell, sheet-name list
and header row must read as expected; Table S1 must carry exactly 163 donors and Table
S2's donors must be exactly that set; the sign split of the released scores must be
25,239 / 17,466; Butland's roster must carry exactly 149 ``SPA-tag essential`` rows; and
the drop rules must account for every dropped row.

REFERENCE. One per record set: the unperturbed conjugant chassis at a gene-interaction
score of 0, which on the S axis is "no epistasis beyond the product of the two single
mutants".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import os.path as osp
import shutil
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Final

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
    SampleUnit,
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
CITATION_KEY = "babuQuantitativeGenomeWideGenetic2014"
PAPER_DOI = "10.1371/journal.pgen.1004120"
PMCID = "PMC3930520"
#: The PubMed id of doi:10.1371/journal.pgen.1004120 (the bacterial schedule row states
#: it beside the DOI, and the PLoS Genetics article page carries the same id).
PUBMED_ID = "24586182"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
LIBRARY_DIR_REL = f"torchcell-library/{CITATION_KEY}"

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "b59b39946b2bcaff5a8f608aaf733c09482e0557a8be0f4546ae8218fcf2dd97"
PROTOCOL_S2_MD = "si/si7.md"
PROTOCOL_S2_MD_SHA256 = (
    "b060ce3f5277f708675e7f26239d5ee2d357ffb2306a04da326417caad135db9"
)
PROTOCOL_S14_MD = "si/si19.md"
PROTOCOL_S14_MD_SHA256 = (
    "e6e4c214b857296d3a5db90a305d78f2b815dbf8f59bfb3793cc8743a72c3cd2"
)
PROTOCOL_S16_MD = "si/si21.md"
PROTOCOL_S16_MD_SHA256 = (
    "be1fa4b58d75de8aed1964f261d8e9ee119eeadcddebdeb30519540d18c5f58f"
)

#: Butland 2008, the eSGA method paper Protocol S2 derives the S score from and the
#: paper the 149 hypomorphic recipients defer to (refs [13,16] of the Results).
BUTLAND_KEY = "butlandESGAColiSynthetic2008"
BUTLAND_DOI = "10.1038/nmeth.1239"
BUTLAND_RAW_DIR_REL = f"torchcell-raw/{BUTLAND_KEY}"
BUTLAND_LIBRARY_DIR_REL = f"torchcell-library/{BUTLAND_KEY}"
BUTLAND_PAPER_MD = "paper.md"
BUTLAND_PAPER_MD_SHA256 = (
    "a3da20a90b56b85e1cd7e23e784c318e78cec115b5bc3e94afcd89a5f94f8eef"
)
BUTLAND_METHODS_MD = "si/si1.md"
BUTLAND_METHODS_MD_SHA256 = (
    "a5e1648bea298f26ab23cb80de96c1c68f7aa7d5d6420aa56608991bc65aa278"
)

TABLE_S1 = "si22.xls"
TABLE_S2 = "si23.xls"
BUTLAND_TABLE_S1 = "si2.xls"
#: When the literature mirror captured Babu's SI (copied from its ``manifest.json``).
DATA_RETRIEVED_AT = "2026-10-07T09:17:40.083033+00:00"
#: When the literature mirror captured Butland's SI (from its own ``manifest.json``).
BUTLAND_RETRIEVED_AT = "2026-10-07T09:58:24.878215+00:00"

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


def _pmc_file(name: str, obj: str, sha256: str, size: int, description: str) -> RawFile:
    """A publisher SI file of the PMC Article Datasets bucket (``pmc_cloud``)."""
    key = f"{PMCID}.1/{obj}"
    return RawFile(
        name=name,
        sha256=sha256,
        bytes=size,
        description=description,
        retrieval=RetrievalRecord(
            method=RetrievalMethod.pmc_cloud,
            source_url=f"https://pmc-oa-opendata.s3.amazonaws.com/{key}",
            retriever="torchcell.literature.retrieve.pmc_cloud_object",
            params={"key": key},
            sha256=sha256,
            retrieved_at=DATA_RETRIEVED_AT,
        ),
    )


RAW_FILES: tuple[RawFile, ...] = (
    _pmc_file(
        TABLE_S1,
        "pgen.1004120.s022.xls",
        "7506c2edc9975d0e331d152e47fa17d4ef7203999b5a8f6a6cd136ef76f7dda7",
        63488,
        "Table S1: the catalog of the 163 donor query strains. Read for exactly two "
        "columns -- 'Essentiality', which says which donors are essential-gene "
        "hypomorphs rather than deletions, and 'Donors screened in this and previous "
        "study', which becomes the record's screen_id",
    ),
    _pmc_file(
        TABLE_S2,
        "pgen.1004120.s023.xls",
        "0789563ada0db3e349bb7e5396311f9d705ea165090733803c009acfa7c95b96",
        3015168,
        "Table S2: the 42,705 high-confidence (donor, recipient) pairs with their GI "
        "score. The authoritative source of every stored value",
    ),
)

BUTLAND_RAW_FILE = RawFile(
    name=BUTLAND_TABLE_S1,
    sha256="7e088873d486307183fbc0eff98b2a29b452c3e3abe8a636979966fd7b49deea",
    bytes=717824,
    description="Butland 2008 Supplementary Table 1: the recipient array roster, "
    "labeling each b-number 'Non-essential' (a Keio deletion) or 'SPA-tag essential' "
    "(one of the 149 hypomorphic strains Babu 2014 defers to refs [13,16] for). Read "
    "for that one label column",
    retrieval=RetrievalRecord(
        method=RetrievalMethod.springer_esm,
        source_url="https://static-content.springer.com/esm/art%3A10.1038%2Fnmeth.1239"
        "/MediaObjects/41592_2008_BFnmeth1239_MOESM306_ESM.xls",
        retriever="torchcell.literature.retrieve.springer_esm",
        params={
            "url": "https://static-content.springer.com/esm/art%3A10.1038%2Fnmeth.1239"
            "/MediaObjects/41592_2008_BFnmeth1239_MOESM306_ESM.xls"
        },
        sha256="7e088873d486307183fbc0eff98b2a29b452c3e3abe8a636979966fd7b49deea",
        retrieved_at=BUTLAND_RETRIEVED_AT,
    ),
)

DATA_SHA256: dict[str, str] = {raw.name: raw.sha256 for raw in RAW_FILES}
DATA_SHA256[BUTLAND_RAW_FILE.name] = BUTLAND_RAW_FILE.sha256

NOT_LOADED = (
    "Table S12 (si33.xls, sheet 'sig gis - intra and intermod') and Table S13 "
    "(si34.xls, sheet 'Module pairs - overview network'): the only other released "
    "sheets carrying a per-pair GI score. Both are PROPER SUBSETS of Table S2, "
    "re-listed with the module or bioprocess annotation the figure they support needs, "
    "so loading either would duplicate records under a second provenance",
    "Tables S3 and S4 (si24.xls, si25.xls): the literature-curated comparison set and "
    "the monochromatic process-pair scores. S3's 'S-Score' column is a score of gene "
    "pairs drawn from OTHER publications' low-throughput experiments (its 'Pubmed ID' "
    "and 'Source' columns say which), so it is a benchmark, not this screen's output; "
    "S4's score is a ratio over a process pair, not a genotype's phenotype",
    "Tables S6, S7, S8, S10 (si27.xls, si28.xls, si29.xls, si31.xls): per-gene and "
    "per-complex COUNTS of interactions (totals, aggravating, alleviating, J-indices). "
    "Every one is an aggregate of Table S2, recomputable from the stored records",
    "Table S9 (si30.xlsx): Pearson correlation of GI profiles between recipient gene "
    "pairs. A derived similarity between two PROFILES, not a measurement of a double "
    "mutant, and it has no phenotype class",
    "Tables S5, S11, S14, S15 (si26.xls, si32.xls, si35.xls, si36.xlsx): the YaiF "
    "ortholog table, the functional-module membership list, the 233-species "
    "phylogenetic profiles and the mutual-information clusters. Annotation and "
    "comparative-genomics layers, none of them a genotype-phenotype record",
    "Table S16 (si37.xls): the strain and plasmid list. Its two parental genotype "
    "strings corroborate the background but are released only as spreadsheet cells, "
    "which audit_sourced_value cannot read as text, so the background's sourced values "
    "quote Protocol S16 and the Results instead",
    "Figures S1-S5 (si1-si5.pdf) and Protocols S1, S3-S13, S15, S16 "
    "(si6.pdf, si8-si18.pdf, si20-si21.pdf): the network figures and the remaining "
    "methods. Protocols S2, S14 and S16 are quoted from their OCR in the LITERATURE "
    "mirror (si/si7.md, si/si19.md, si/si21.md), which is where those SourcedValues "
    "point; none of them is a build input",
    "Butland 2008 Supplementary Tables 2-4 (si3.xls, si4.xls, si5.xls): the 39-screen "
    "query primer list, that paper's own 1,288 high-confidence interactions and its "
    "raw and normalized colony sizes. The 39 screens are already inside Babu's Table "
    "S2 and are marked there by screen_id, so loading Butland's own release would "
    "duplicate them under a second score definition",
)
"""Every released file the loader read and did NOT load, with the reason."""


# --------------------------------------------------------------------------- #
# Quote helpers (verbatim substrings of sha256-pinned OCR)
# --------------------------------------------------------------------------- #
def _sourced(
    value: Any,
    quote: str,
    *,
    uri: str,
    sha256: str,
    citation_key: str,
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
            citation_key=citation_key,
            sha256=sha256,
            method=method,
            page=page,
        ),
    )


def _paper(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """A value bound to a verbatim quote in the pinned Babu 2014 ``paper.md``."""
    return _sourced(
        value,
        quote,
        uri=PAPER_MD,
        sha256=PAPER_MD_SHA256,
        citation_key=CITATION_KEY,
        method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
        note=note,
    )


def _protocol_s2(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """A value bound to a verbatim quote in the pinned OCR of Protocol S2."""
    return _sourced(
        value,
        quote,
        uri=PROTOCOL_S2_MD,
        sha256=PROTOCOL_S2_MD_SHA256,
        citation_key=CITATION_KEY,
        method="MinerU OCR of Protocol S2 (torchcell-library mirror)",
        page="Protocol S2 (pgen.1004120.s007.pdf)",
        note=note,
    )


def _protocol_s14(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """A value bound to a verbatim quote in the pinned OCR of Protocol S14."""
    return _sourced(
        value,
        quote,
        uri=PROTOCOL_S14_MD,
        sha256=PROTOCOL_S14_MD_SHA256,
        citation_key=CITATION_KEY,
        method="MinerU OCR of Protocol S14 (torchcell-library mirror)",
        page="Protocol S14 (pgen.1004120.s019.pdf)",
        note=note,
    )


def _protocol_s16(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """A value bound to a verbatim quote in the pinned OCR of Protocol S16."""
    return _sourced(
        value,
        quote,
        uri=PROTOCOL_S16_MD,
        sha256=PROTOCOL_S16_MD_SHA256,
        citation_key=CITATION_KEY,
        method="MinerU OCR of Protocol S16 (torchcell-library mirror)",
        page="Protocol S16 (pgen.1004120.s021.pdf)",
        note=note,
    )


def _butland(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """A value bound to a verbatim quote in the pinned Butland 2008 ``paper.md``."""
    return _sourced(
        value,
        quote,
        uri=BUTLAND_PAPER_MD,
        sha256=BUTLAND_PAPER_MD_SHA256,
        citation_key=BUTLAND_KEY,
        method="MinerU OCR of the publisher PDF (torchcell-library mirror); the eSGA "
        "method paper Babu 2014 Protocol S2 derives its S score from",
        note=note,
    )


def _butland_methods(
    value: Any, quote: str, *, note: str | None = None
) -> SourcedValue:
    """A value bound to a verbatim quote in the pinned OCR of Butland's SI methods."""
    return _sourced(
        value,
        quote,
        uri=BUTLAND_METHODS_MD,
        sha256=BUTLAND_METHODS_MD_SHA256,
        citation_key=BUTLAND_KEY,
        method="MinerU OCR of the Butland 2008 Supplementary Information "
        "(torchcell-library mirror)",
        page="Supplementary Methods, eSGA screening procedure",
        note=note,
    )


# --------------------------------------------------------------------------- #
# Sourced values
# --------------------------------------------------------------------------- #
_Q_CONJUGATION = (
    "After generating selectable mutants in a hyper-recombinant Hfr-Cavalli (Hfr C) "
    "‘donor’ strain background marked with a chloramphenicol-resistance "
    "cassette $( \\mathrm { C m } ^ { \\breve { \\bf R } } )$ , the corresponding "
    "deletion alleles were transferred by conjugation into a near genome-wide mutant "
    "collection of $\\mathrm { F } _ { \\mathrm { - } }$ ‘recipient’ mutant "
    "strains, arrayed in duplicate at 384-colony density."
)
_Q_ARRAY = (
    "This collection, contains 3,968 non-essential single gene deletions in which the "
    "open reading frame was replaced and marked by a kanamycin resistance $( \\bar { "
    "\\mathrm { K a n } } ^ { \\mathrm { R } } )$ cassette (i.e., the Keio collection) "
    "[19], and 149 hypomorphic mutant strains [13,16], in which a $\\bar { \\mathrm { "
    "K a n } } ^ { \\mathrm { R } }$ marker was integrated into the $3 ^ { \\prime }$ "
    "-UTR to alter transcript abundance or stability [13]"
)
_Q_DONORS = (
    "In total, a set of 163 query ‘donor’ genes with evidence of expression "
    "and whose products had high physical interaction degree were selected for "
    "screening (Protocol S1)."
)
_Q_EIGHT_MAIN = (
    "we performed two independent replicate screens such that each donor-recipient "
    "mutant gene pair was tested eight times to account for experimental variation "
    "(see Protocol S2)."
)
_Q_MEDIUM_TEMP = (
    "Following genetic transfer, the double mutants were selected on rich medium "
    "(Luria Broth) containing both marker drugs $\\left( \\mathrm { K a n + C m } "
    "\\right)$ . After outgrowth for $3 6 ~ \\mathrm { h r s }$ at $3 2 ^ { \\circ } "
    "\\mathrm { C }$ , the plates were imaged digitally."
)
_Q_THRESHOLD_COUNTS = (
    "After filtering, the network encompassed GI with $S$ -scores of $- 3$ or lower "
    "(25,239 in total) that indicate aggravating (i.e., SSL) relationships, and GIs "
    "with $S _ { \\mathbf { \\lambda } }$ -scores of $+ 3$ or higher (17,466) "
    "representing alleviating relationships (Figure 2B, Table S2)"
)
_Q_EIGHT_S2 = (
    "Each genome-wide screen was performed twice by replica pinning the conjugants "
    "arrayed in a 384 density format using four biological replicate recipient "
    "colonies, which is selected further on double antibiotics at 1,536 colony "
    "density, to generate double mutant colonies. These eight replicate measurements "
    "of each gene pair were subsequently averaged into a single GI S-score to account "
    "for colony plate variance."
)
_Q_DEFERRAL = (
    "The raw colony size fitness measurements from the newly performed 124 eSGA "
    "genetic screens were normalized and processed into $S$ -scores essentially as "
    "described in our previous genomewide study [1]."
)
_Q_COMBINED39 = (
    "This newly derived GI score was then combined with our previously published GI "
    "datasets from 39 genome-wide screens to generate a comprehensive GI network."
)
_Q_CUTOFFS = (
    "The filtered GI interaction data was $Z .$ -score normalized and thresholded at a "
    "cut-off value of |2|, corresponding to a $p$ -value $\\le 0 . 0 5$ and a GI "
    "S-score of $\\leq - 3 . 3$ for aggravating and $\\geq 3 . 1$ for alleviating "
    "interactions."
)
_Q_STRAINS = (
    "The F- ‘recipient’ single gene deletion knock-out strain marked with ${ "
    "\\mathrm { K a n } } ^ { \\mathrm { R } }$ were from the Keio mutant library [1]. "
    "The Hfr C non-essential donor gene deletion mutant strains or essential gene "
    "hypomorphic mutations were constructed using the $\\lambda$ -Red recombination "
    "[2-4] or P1 phage transduction [5] system."
)
_Q_42705 = (
    "We mapped 42,705 putative digenic interactions from our eSGA screen with $S "
    "\\mathrm { \\cdot }$ -scores $| 2 3 |$ at a threshold of 2 standard deviations of "
    "significance"
)
_Q_BUTLAND_SPA = (
    "we also added $1 4 9 \\mathrm { ~ \\textit ~ { ~ F ~ } ~ }$ potentially "
    "hypomorphic kan-marked strains with essential genes involved in conserved "
    "bacterial processes13 tagged with a gene encoding a C-terminal sequential peptide "
    "affinity tag (SPA)."
)
_Q_BUTLAND_S = (
    "We then calculated an interaction score (S) to quantify the strength and "
    "confidence of genetic interaction determined for each mutant gene pair. Negative "
    "$S$ scores correspond to putative aggravating interactions and positive S scores, "
    "to putative alleviating interactions."
)
_Q_BUTLAND_MARKERS = (
    "An Hfr strain with a query gene deletion mutation marked with cat "
    "(chloramphenicol-resistance gene; pink box) is grown overnight in liquid LB with "
    "chloramphenicol (Cm) and pinned onto LB-Cm plates. Simultaneously, the recipient "
    "$\\mathsf { F } ^ { - }$ mutant array strains marked with kan "
    "(kanamycin-resistance gene; red box) are pinned onto LB-kanamycin (Kan) plates."
)
_Q_BUTLAND_AGAR = (
    "This conjugation plate is pinned onto a LB-Kan-Cm agar plate for a first "
    "selection of 48 hrs at $3 2 ~ ^ { \\circ } \\mathrm { C }$ ."
)

SOURCED_VALUES: dict[str, SourcedValue] = {
    "reference_strain": _paper(
        "MG1655",
        _Q_CONJUGATION,
        note="the assembly the records' identifiers are written against, NOT the "
        "chassis: the release keys every gene on an MG1655 b-number, and "
        "GCA_000005845.2 is the sha256-pinned bytes those tags mean. The physical "
        "strain is the Hfr C x Keio conjugant this quote describes, carried as the "
        "background's parents",
    ),
    "conjugation": _paper(
        "Hfr Cavalli donor x Keio recipient conjugant",
        _Q_CONJUGATION,
        note="how the measured double mutant was made: the donor's cat-marked deletion "
        "is transferred into the kan-marked F- recipient by conjugation",
    ),
    "recipient_array": _paper(
        3968,
        _Q_ARRAY,
        note="the recipient array is 3,968 Keio deletions plus 149 hypomorphs; the "
        "hypomorph half has no bacterial perturbation leaf, so its pairs are dropped",
    ),
    "n_hypomorphic_recipients": _paper(
        149,
        _Q_ARRAY,
        note="the COUNT only; the list is deferred to refs [13,16] (Butland 2008, Babu "
        "2011) and read from Butland 2008 Supplementary Table 1, whose 'SPA-tag "
        "essential' rows number exactly 149",
    ),
    "n_donors": _paper(
        163,
        _Q_DONORS,
        note="Table S1 holds exactly these 163 donors plus one footnote row",
    ),
    "n_samples": _protocol_s2(
        8,
        _Q_EIGHT_S2,
        note="EXACT, not a range: two replicate screens x four biological replicate "
        "recipient colonies = the eight colony measurements averaged into the stored "
        "score. The sample unit is a colony. Stored on every record as n_samples with "
        "sample_unit=colony since #793 added the replicate-design quartet to "
        "GeneInteractionPhenotype; preprocess/replicate_structure.json keeps the "
        "provenance copy so the stored value stays auditable",
    ),
    "n_samples_main_text": _paper(
        8,
        _Q_EIGHT_MAIN,
        note="the Results state the same count independently of Protocol S2",
    ),
    "score_definition": _butland(
        "interaction score (S), signed",
        _Q_BUTLAND_S,
        note="Babu 2014 Protocol S2 derives its scores 'essentially as described in our "
        "previous genomewide study [1]', and ref [1] of that protocol is this paper; "
        "the sign convention stored on the record comes from here. The S formula "
        "itself defers one step further, to Collins 2006 (Genome Biol 7:R63), which is "
        "NOT in the mirror, so the operational definition quoted here is as far as the "
        "chain reaches",
    ),
    "score_deferral": _protocol_s2(
        BUTLAND_KEY,
        _Q_DEFERRAL,
        note="the deferral itself: 124 new screens scored as in ref [1] = Butland 2008",
    ),
    "screen_sets": _protocol_s2(
        (124, 39),
        _Q_COMBINED39,
        note="Table S2 is a superset: 124 screens new here plus 39 previously "
        "published, which Table S1 attributes per donor and the record stores as "
        "screen_id",
    ),
    "cutoffs": _protocol_s2(
        (-3.3, 3.1),
        _Q_CUTOFFS,
        note="the released tail: measured on Table S2 the most negative kept score is "
        "-3.38787, matching the -3.3 cut exactly, and the smallest kept positive score "
        "is 3.08809, which matches the Results' '+3 or higher' rather than this "
        "protocol's 3.1 (213 kept rows lie between 3.08809 and 3.1)",
    ),
    "sign_split": _paper(
        (25239, 17466),
        _Q_THRESHOLD_COUNTS,
        note="asserted against Table S2 at build time: 25,239 negative and 17,466 "
        "positive scores, no zeros",
    ),
    "n_released_pairs": _protocol_s14(
        42705,
        _Q_42705,
        note="the row count of Table S2, stated independently of the Results",
    ),
    "medium_and_temperature": _paper(
        ("LB with kanamycin and chloramphenicol", 32.0, 36.0),
        _Q_MEDIUM_TEMP,
        note="the one condition: the double-drug selection plates the colonies were "
        "scored on. No component amount is stated anywhere in this paper for the "
        "screen plates, so every component of the stored medium carries "
        "concentration=None",
    ),
    "strains": _protocol_s16(
        ("Keio mutant library", "Hfr C"),
        _Q_STRAINS,
        note="the recipient collection and the donor background, and the statement "
        "that an essential-gene donor is a hypomorph rather than a deletion",
    ),
    "markers": _butland(
        ("cat", "kan"),
        _Q_BUTLAND_MARKERS,
        note="which cassette each side of the cross carries. Babu 2014 writes them as "
        "Cm-R and Kan-R; the gene names are from the method paper, and they are what "
        "makes a reciprocal pair two strains rather than two measurements",
    ),
    "spa_hypomorphs": _butland(
        149,
        _Q_BUTLAND_SPA,
        note="what a hypomorphic recipient IS: a kan-marked C-terminal SPA tag on an "
        "essential gene. The per-gene list is Butland Supplementary Table 1's 'SPA-tag "
        "essential' column, read at build time from the raw mirror",
    ),
    "agar": _butland_methods(
        "agar",
        _Q_BUTLAND_AGAR,
        note="the selection plates are agar plates; neither paper prints a percentage, "
        "so the gelling agent is named and unquantified",
    ),
}


# --------------------------------------------------------------------------- #
# The one medium and the one environment
# --------------------------------------------------------------------------- #
_NO_AMOUNT = (
    "the paper names the medium and its two marker drugs and prints no amount for any "
    "component of the screen plates, so none is asserted here"
)
_NO_DOSE = (
    "no dose is asserted: Babu 2014 states none, and Butland 2008's own SI prints "
    "kanamycin at 50 ug/mL for the array pre-culture and 25 ug/mL 'throughout', a "
    "disagreement within one mirrored source rather than a value"
)


def selection_medium() -> Media:
    """The LB-Kan-Cm selection plates the double-mutant colonies were scored on."""
    statement = SOURCED_VALUES["medium_and_temperature"]
    return Media(
        name="LB with kanamycin and chloramphenicol, amounts not stated (Babu 2014), "
        "solid",
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
                provenance=[statement, SOURCED_VALUES["markers"]],
                note=f"selects the recipient's kan cassette; {_NO_DOSE}",
            ),
            MediaComponent(
                compound=resolved_compound("chloramphenicol"),
                role=MediaComponentRole.selection_agent,
                concentration=None,
                provenance=[statement, SOURCED_VALUES["markers"]],
                note="selects the donor's cat cassette, so a colony on this medium is "
                "a double mutant; no dose is stated for the screen plates",
            ),
        ],
        provenance=[statement, SOURCED_VALUES["agar"]],
    )


SELECTION_MEDIUM = selection_medium()
"""Built once: one condition, so one medium object every record joins on."""

TEMPERATURE_C = 32.0
DURATION_HOURS = 36.0
AEROBICITY = "aerobic"


def screen_environment() -> Environment:
    """The one environment of the screen: LB-Kan-Cm plates, 32 C, 36 h, aerobic."""
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
    pubmed_id=PUBMED_ID,
    pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PUBMED_ID}/",
    doi=PAPER_DOI,
    doi_url=f"https://doi.org/{PAPER_DOI}",
)


# --------------------------------------------------------------------------- #
# Raw mirrors (the loader reads the mirrors, never a live URL)
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirrors and build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/babuQuantitativeGenomeWideGenetic2014``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def butland_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/butlandESGAColiSynthetic2008``."""
    return Path(data_root or _data_root()) / BUTLAND_RAW_DIR_REL


def library_dir(citation_key: str, data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-library/<citation_key>``."""
    return Path(data_root or _data_root()) / "torchcell-library" / citation_key


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def all_raw_files() -> tuple[RawFile, ...]:
    """Every file the loader consumes: Babu's two tables, then Butland's roster."""
    return (*RAW_FILES, BUTLAND_RAW_FILE)


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
    for raw in all_raw_files():
        if names is not None and raw.name not in names:
            continue
        path = dest / f"{raw.name}"
        write_verified(
            run_retriever(raw.retrieval),
            path,
            raw.sha256,
            raw.retrieval.source_url or raw.name,
        )
        out[raw.name] = path
    return out


def _deposit(
    root: Path,
    files: Sequence[RawFile],
    sources: Mapping[str, str | Path],
    *,
    citation_key: str,
    doi: str,
    title: str,
    not_mirrored: Sequence[str],
) -> Path:
    """Copy ``files`` into one raw mirror and write its ``manifest.json``.

    Idempotent by sha256: a mirror file with the pinned hash is left alone, and one
    with any other hash raises rather than being overwritten.
    """
    records: list[ArtifactRecord] = []
    for raw in files:
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
    manifest = Manifest(
        citation_key=citation_key,
        doi=doi,
        title=title,
        files=records,
        si_data_sources=[r.retrieval.source_url or r.name for r in files],
        si_expected=list(not_mirrored),
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return root


def deposit_raw_mirror(
    *, sources: Mapping[str, str | Path], data_root: str | None = None
) -> tuple[Path, Path]:
    """Write BOTH raw mirrors this loader reads, each under its own citation key.

    Babu's two tables go to ``torchcell-raw/<Babu key>`` and Butland's array roster to
    ``torchcell-raw/<Butland key>``: a deferral file belongs to the paper that released
    it, so it is never copied into the deferring paper's mirror.
    """
    missing = sorted(set(DATA_SHA256) - set(sources))
    if missing:
        raise KeyError(f"no source given for {missing}")
    babu = _deposit(
        raw_mirror_dir(data_root),
        RAW_FILES,
        sources,
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title="Quantitative Genome-Wide Genetic Interaction Screens Reveal Global "
        "Epistatic Relationships of Protein Complexes in Escherichia coli",
        not_mirrored=NOT_LOADED,
    )
    butland = _deposit(
        butland_mirror_dir(data_root),
        (BUTLAND_RAW_FILE,),
        sources,
        citation_key=BUTLAND_KEY,
        doi=BUTLAND_DOI,
        title="eSGA: E. coli synthetic genetic array analysis",
        not_mirrored=(
            "Butland 2008 Supplementary Tables 2-4 (si3.xls, si4.xls, si5.xls): the "
            "query primer list, that paper's own 1,288 high-confidence interactions "
            "and its raw and normalized colony sizes. Babu 2014's Table S2 already "
            "contains these 39 screens, marked by screen_id, so nothing here is a "
            "build input",
        ),
    )
    return babu, butland


def load_manifest_of(citation_key: str, data_root: str | None = None) -> Manifest:
    """Read one raw mirror's ``manifest.json``."""
    root = (
        butland_mirror_dir(data_root)
        if citation_key == BUTLAND_KEY
        else raw_mirror_dir(data_root)
    )
    return Manifest.model_validate_json((root / "manifest.json").read_text())


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
    """A released sheet whose title, sheet list or header row is not what we read."""


class ReleaseContentError(ValueError):
    """A released sheet whose contents contradict a sourced count."""


class SheetSpec(BaseModel):
    """One released sheet: its file, its sheet name, its title cell and its header."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    file: str
    sheet_name: str
    title: str
    header_row: int
    columns: tuple[str, ...]


TABLE_S2_SPEC = SheetSpec(
    file=TABLE_S2,
    sheet_name="WG_GI_Score_Mar_06_2013",
    title="Supplementary Table S2 | List of gene pairs with high-confidence epistatic "
    "interactions",
    header_row=2,
    columns=("Donor", "Recipient", "GI score"),
)
TABLE_S1_SPEC = SheetSpec(
    file=TABLE_S1,
    sheet_name="Sheet1",
    title="Supplementary Table S1| Catalog of donor query strains targeted in this "
    "study",
    header_row=2,
    columns=(
        "Bnum",
        "Gene",
        "Essentiality",
        "Donors screened in this and previous study",
    ),
)
BUTLAND_TABLE_S1_SPEC = SheetSpec(
    file=BUTLAND_TABLE_S1,
    sheet_name="st1",
    title='Supplementary Table 1. List of recipient F- "KEIO" mutant strainsa and '
    "SPA-tagged essential genesb used in this study. ",
    header_row=5,
    columns=("Essentiala/Non-essentialb", "Gene Name", "b-numberc"),
)

#: The released counts the build asserts against the sheets. Each is the ``value`` of its
#: ``SOURCED_VALUES`` entry, named here so a hermetic test can build a small synthetic
#: release and still exercise every check (the sourced quotes stay the authority).
N_DONORS: int = int(SOURCED_VALUES["n_donors"].value)
N_RELEASED_PAIRS: int = int(SOURCED_VALUES["n_released_pairs"].value)
SIGN_SPLIT: tuple[int, int] = (
    int(SOURCED_VALUES["sign_split"].value[0]),
    int(SOURCED_VALUES["sign_split"].value[1]),
)
N_HYPOMORPHIC_RECIPIENTS: int = int(SOURCED_VALUES["spa_hypomorphs"].value)

#: Table S1's ``Essentiality`` value that marks a donor as an essential-gene hypomorph.
ESSENTIAL_DONOR = "essential"
#: Butland Supplementary Table 1's label for one of the 149 hypomorphic recipients.
SPA_TAG_ESSENTIAL = "SPA-tag essential"
#: Its label for a Keio deletion recipient.
KEIO_NON_ESSENTIAL = "Non-essential"
#: Table S1's two ``Donors screened in this and previous study`` values, which become
#: the record's ``screen_id``.
SCREEN_THIS_STUDY = "This Study"
SCREEN_BUTLAND = "Butland et al."


def read_sheet(path: str | Path, spec: SheetSpec) -> pd.DataFrame:
    """One released sheet as a frame, refusing an unexpected sheet list, title or header.

    The title is read from the raw first cell, so a re-released file with a shifted
    header fails here instead of silently loading the wrong row as column names.
    """
    first = pd.read_excel(path, sheet_name=None, header=None, nrows=1)
    if spec.sheet_name not in first:
        raise SheetFormatError(f"{spec.file}: sheets {list(first)}")
    title = first[spec.sheet_name].iat[0, 0]
    if title != spec.title:
        raise SheetFormatError(f"{spec.file}: title {title!r}, not {spec.title!r}")
    frame = pd.read_excel(path, sheet_name=spec.sheet_name, header=spec.header_row)
    missing = [column for column in spec.columns if column not in frame.columns]
    if missing:
        raise SheetFormatError(f"{spec.file}: no {missing} column(s)")
    return frame


def read_table_s2(path: str | Path) -> pd.DataFrame:
    """Table S2 as ``donor_bnum, donor_gene, recipient_bnum, recipient_gene, score``.

    Both id columns are written ``<bnum>__<gene>``; the split is asserted, and the
    released sign split must be the one the Results state.
    """
    frame = read_sheet(path, TABLE_S2_SPEC)
    frame = frame[list(TABLE_S2_SPEC.columns)].copy()
    if frame.isna().to_numpy().any():
        raise SheetFormatError(f"{TABLE_S2}: a released cell is empty")
    for side in ("Donor", "Recipient"):
        parts = frame[side].astype(str).str.split("__", expand=True)
        if parts.shape[1] != 2 or parts.isna().to_numpy().any():
            raise SheetFormatError(f"{TABLE_S2}: {side} is not '<bnum>__<gene>'")
        frame[f"{side.lower()}_id"] = parts[0]
        frame[f"{side.lower()}_gene"] = parts[1]
    frame["score"] = frame["GI score"].astype(float)
    observed = (int((frame["score"] < 0).sum()), int((frame["score"] > 0).sum()))
    if observed != SIGN_SPLIT or len(frame) != N_RELEASED_PAIRS:
        raise ReleaseContentError(
            f"{TABLE_S2}: {len(frame)} rows with sign split {observed}; the paper "
            f"states {N_RELEASED_PAIRS} and {SIGN_SPLIT}"
        )
    return frame[
        ["donor_id", "donor_gene", "recipient_id", "recipient_gene", "score"]
    ].reset_index(drop=True)


class DonorSpec(BaseModel):
    """One Table S1 donor: whether it is a hypomorph, and which screen set it is from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    b_number: str
    gene: str
    is_hypomorph: bool
    screen_id: str


def read_donors(path: str | Path) -> dict[str, DonorSpec]:
    """Table S1's 163 donors, keyed on b-number, with the footnote row dropped.

    The footnote row carries a prose ``Bnum``; every real row's ``Bnum`` is a b-number
    and both read columns are filled, which is asserted rather than assumed.
    """
    frame = read_sheet(path, TABLE_S1_SPEC)
    screen_column = "Donors screened in this and previous study"
    rows = frame[frame["Bnum"].astype(str).str.match(B_NUMBER_PATTERN)]
    donors: dict[str, DonorSpec] = {}
    for _, row in rows.iterrows():
        essentiality = str(row["Essentiality"])
        screen = str(row[screen_column])
        if essentiality not in {ESSENTIAL_DONOR, "non-essential"}:
            raise SheetFormatError(f"{TABLE_S1}: Essentiality {essentiality!r}")
        if screen not in {SCREEN_THIS_STUDY, SCREEN_BUTLAND}:
            raise SheetFormatError(f"{TABLE_S1}: screen set {screen!r}")
        b_number = str(row["Bnum"])
        if b_number in donors:
            raise SheetFormatError(f"{TABLE_S1}: donor {b_number} listed twice")
        donors[b_number] = DonorSpec(
            b_number=b_number,
            gene=str(row["Gene"]),
            is_hypomorph=essentiality == ESSENTIAL_DONOR,
            screen_id=screen,
        )
    if len(donors) != N_DONORS:
        raise ReleaseContentError(
            f"{TABLE_S1}: {len(donors)} donors, the paper states {N_DONORS}"
        )
    return donors


def read_hypomorphic_recipients(path: str | Path) -> frozenset[str]:
    """The 149 ``SPA-tag essential`` b-numbers of Butland Supplementary Table 1.

    This is the deferral Babu's Results make for its 149 hypomorphic recipients. The
    count is asserted against the sourced 149, so a re-released roster that no longer
    carries exactly those strains fails the build instead of changing which pairs are
    dropped.
    """
    frame = read_sheet(path, BUTLAND_TABLE_S1_SPEC)
    label, _, b_number = BUTLAND_TABLE_S1_SPEC.columns
    rows = frame[frame[b_number].astype(str).str.match(B_NUMBER_PATTERN)]
    labels = set(rows[label].astype(str))
    if labels != {SPA_TAG_ESSENTIAL, KEIO_NON_ESSENTIAL}:
        raise SheetFormatError(f"{BUTLAND_TABLE_S1}: labels {sorted(labels)}")
    spa = frozenset(
        rows.loc[rows[label].astype(str) == SPA_TAG_ESSENTIAL, b_number].astype(str)
    )
    if len(spa) != N_HYPOMORPHIC_RECIPIENTS:
        raise ReleaseContentError(
            f"{BUTLAND_TABLE_S1}: {len(spa)} SPA-tag essential genes, Butland 2008 and "
            f"Babu 2014 both state {N_HYPOMORPHIC_RECIPIENTS}"
        )
    return spa


# --------------------------------------------------------------------------- #
# Retention ledger
# --------------------------------------------------------------------------- #
RULE_HYPOMORPH = "hypomorphic_allele_has_no_bacterial_perturbation_leaf"
RULE_NOT_A_TAG = "b_number_is_not_a_locus_tag_of_the_pinned_annotation"
RULE_REMAPPED = "b_number_remapped_by_the_annotation"
RULE_DUPLICATE = "contradictory_duplicate_pair"

DROP_RULE_DESCRIPTIONS: dict[str, str] = {
    RULE_HYPOMORPH: "one side of the pair is a hypomorph -- an essential-gene donor "
    "(Table S1 Essentiality == 'essential') or one of the 149 recipients Butland 2008 "
    "Supplementary Table 1 labels 'SPA-tag essential'. The allele is a kan cassette at "
    "the 3' end that alters transcript abundance, which none of the five bacterial "
    "gene-perturbation leaves can type: it is not a deletion, not a mapped transposon "
    "insertion, not a CRISPRi knockdown and not a promoter replacement. Typing it as a "
    "deletion would assert an absence the paper never claims, so the pair is dropped "
    "and the missing leaf is filed as issue #792",
    RULE_NOT_A_TAG: "a released id is not an MG1655 b-number at all (a Keio JW id, a "
    "gene symbol, or a 'bNNNN.1' sub-numbered Keio entry). A bacterial perturbation "
    "leaf's name validator refuses it, and storing it would put a record on a locus "
    "GCA_000005845.2 does not have",
    RULE_REMAPPED: "a released b-number that the pinned annotation carries as a "
    "/gene_synonym of a DIFFERENT locus. Storing the merged locus needs a "
    "DerivedIdentifierMapping and DerivedIdentifierRoute has no member for a retired "
    "tag of the pinned strain's own namespace (issue #753), so the pair is dropped "
    "rather than remapped silently",
    RULE_DUPLICATE: "an ordered (donor, recipient) pair Table S2 releases twice with "
    "different scores. The release gives no rule for choosing between them and one of "
    "the two pairs disagrees in sign, so both rows of both pairs are dropped",
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
    source_donors: int
    source_recipients: int
    kept_donors: int
    kept_recipients: int
    n_aggravating: int
    n_alleviating: int
    screen_id_counts: dict[str, int]
    rules: list[DropRule]


class ReplicateStructure(BaseModel):
    """The replicate design, with its provenance, written out beside the build.

    Since #793 the design is ALSO on every record (``n_samples=8``,
    ``sample_unit=colony``). This file is what makes that field auditable rather than
    asserted: it carries the verbatim Protocol S2 quote, the citation key, the source
    path and its sha256 beside the two numbers, which a graph property cannot.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_samples: int
    sample_unit: str
    replicate_screens: int
    colonies_per_screen: int
    quote: str
    citation_key: str
    source_uri: str
    sha256: str
    note: str


class ReciprocalPairs(BaseModel):
    """Unordered gene pairs measured in both directions, which are kept as two records."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_pairs: int
    n_records: int
    n_sign_disagreements: int
    pairs: list[str]


def replicate_structure() -> ReplicateStructure:
    """The eight-colony design of one stored score, with its verbatim source."""
    sourced = SOURCED_VALUES["n_samples"]
    return ReplicateStructure(
        n_samples=int(sourced.value),
        sample_unit="colony",
        replicate_screens=2,
        colonies_per_screen=4,
        quote=sourced.quote,
        citation_key=str(sourced.provenance.citation_key),
        source_uri=sourced.provenance.source_uri,
        sha256=str(sourced.provenance.sha256),
        note="EXACT, not a range: two replicate screens x four biological replicate "
        "recipient colonies. Stored on every record since #793; this file is the "
        "provenance copy of the same two numbers",
    )


# --------------------------------------------------------------------------- #
# The record
# --------------------------------------------------------------------------- #
DONOR_COLLECTION = "Hfr Cavalli (Hfr C) donor query strains"
RECIPIENT_COLLECTION = "Keio collection"
DONOR_CASSETTE = "cat"
RECIPIENT_CASSETTE = "kan"

UNITS_NOTE = (
    "eSGA interaction score (S) of a double mutant's normalized colony size against "
    "the normalized median colony size of all double mutants of the same donor screen; "
    "negative is aggravating (synthetic sick or lethal), positive alleviating"
)

P_VALUE_GAP = ProvenanceGap(
    field="gene_interaction_p_value",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=Provenance(
        source_uri=PROTOCOL_S2_MD,
        citation_key=CITATION_KEY,
        sha256=PROTOCOL_S2_MD_SHA256,
        method="full Protocol S2 and the Results' GI-network section read",
        page="Protocol S2; Results, 'Generating a genome-wide network of "
        "high-confidence GIs'",
    ),
    note="the paper states P<=0.05 for the released set as a whole (the |Z|>=2 cut), "
    "never a p-value per pair, and Table S2 releases no p-value column",
)


def donor_perturbation(locus_tag: str, symbol: str) -> BacterialDeletionPerturbation:
    """The donor side: the Hfr C query deletion, marked with ``cat``."""
    return BacterialDeletionPerturbation(
        systematic_gene_name=locus_tag,
        perturbed_gene_name=symbol,
        gene_namespace=MG1655_NAMESPACE,
        collection=DONOR_COLLECTION,
        cassette=DONOR_CASSETTE,
    )


def recipient_perturbation(
    locus_tag: str, symbol: str
) -> BacterialDeletionPerturbation:
    """The recipient side: the arrayed Keio deletion, marked with ``kan``."""
    return BacterialDeletionPerturbation(
        systematic_gene_name=locus_tag,
        perturbed_gene_name=symbol,
        gene_namespace=MG1655_NAMESPACE,
        collection=RECIPIENT_COLLECTION,
        cassette=RECIPIENT_CASSETTE,
    )


def pair_genotype(
    donor_tag: str, donor_gene: str, recipient_tag: str, recipient_gene: str
) -> Genotype:
    """The double mutant: the donor allele and the recipient allele.

    The two leaves differ in ``collection`` and ``cassette``, which is what keeps a
    reciprocally measured gene pair two strains rather than two measurements of one.
    """
    return Genotype(
        perturbations=[
            donor_perturbation(donor_tag, donor_gene),
            recipient_perturbation(recipient_tag, recipient_gene),
        ]
    )


def phenotype(score: float, screen_id: str) -> GeneInteractionPhenotype:
    """One released GI score, with its replicate design and the screen set it came from.

    ``n_samples`` is Protocol S2's eight colonies per pair, exactly
    (``SOURCED_VALUES["n_samples"]``), now a field of the phenotype rather than a file
    beside the build (#793). It is a constant of the release: every stored score is the
    average of the same eight colony measurements, two replicate screens x four
    biological replicate recipient colonies. No uncertainty accompanies it -- the paper
    releases no per-pair dispersion -- so the uncertainty pair stays None, which is a
    different statement from the p-value's typed gap (the source tested the SET).
    """
    return GeneInteractionPhenotype(
        gene_interaction=score,
        gene_interaction_p_value=None,
        screen_id=screen_id,
        n_samples=int(SOURCED_VALUES["n_samples"].value),
        sample_unit=SampleUnit.colony,
        provenance_gaps=[P_VALUE_GAP],
    )


def reference_phenotype(screen_id: str) -> GeneInteractionPhenotype:
    """The unperturbed chassis of the same screen: no epistasis, a score of 0."""
    return GeneInteractionPhenotype(
        gene_interaction=0.0,
        gene_interaction_p_value=None,
        screen_id=screen_id,
        provenance_gaps=[P_VALUE_GAP],
    )


def chassis_background() -> BacterialStrainBackground:
    """The conjugant chassis: an Hfr C donor crossed into a Keio (BW25113) recipient.

    ``reference_strain`` is MG1655 because that is the namespace every released
    identifier is written in and the assembly those tags mean; ``parents`` names the two
    strains the cross actually used. ``alleles`` is empty and ``genotype_statement`` is
    a typed gap: the two parental genotype strings are released only in Table S16's
    spreadsheet cells, which the provenance audit reads as bytes rather than text, so
    quoting them here would be a value with no auditable source.
    """
    return BacterialStrainBackground(
        name="Hfr Cavalli x Keio (K-12 BW25113) eSGA conjugant",
        reference_strain=REFERENCE_STRAIN_NAME,
        assembly_set=MG1655_ASSEMBLY_SET,
        parents=["Hfr Cavalli", "Keio collection (K-12 BW25113)"],
        construction="the donor's cat-marked deletion is transferred from the "
        "hyper-recombinant Hfr C donor into the F- Keio recipient by conjugation and "
        "homologous recombination, and the conjugant is selected on LB with kanamycin "
        "and chloramphenicol",
        genotype_statement=None,
        alleles=[],
        provenance=[
            SOURCED_VALUES["conjugation"],
            SOURCED_VALUES["strains"],
            SOURCED_VALUES["markers"],
        ],
        provenance_gaps=[
            ProvenanceGap(
                field="genotype_statement",
                reason=ProvenanceGapReason.not_reported_by_primary,
                looked_in=Provenance(
                    source_uri=PROTOCOL_S16_MD,
                    citation_key=CITATION_KEY,
                    sha256=PROTOCOL_S16_MD_SHA256,
                    method="full Protocol S16 and the Results' screen description read",
                    page="Protocol S16 (pgen.1004120.s021.pdf)",
                ),
                note="the running text names the Keio library and Hfr C but writes no "
                "genotype string; Table S16 carries both strings in spreadsheet cells, "
                "which is not an auditable text source",
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
    donor_tag: str,
    donor_gene: str,
    recipient_tag: str,
    recipient_gene: str,
    score: float,
    screen_id: str,
) -> BacterialGeneInteractionExperiment:
    """The record of one Table S2 row."""
    return BacterialGeneInteractionExperiment(
        dataset_name=dataset_name,
        genotype=pair_genotype(donor_tag, donor_gene, recipient_tag, recipient_gene),
        environment=SCREEN_ENVIRONMENT,
        phenotype=phenotype(score, screen_id),
    )


def build_reference(
    dataset_name: str, genome: AssemblyReferenceGenome, screen_id: str
) -> BacterialGeneInteractionExperimentReference:
    """The unperturbed chassis of one screen set, scoring 0."""
    return BacterialGeneInteractionExperimentReference(
        dataset_name=dataset_name,
        genome_reference=genome,
        environment_reference=SCREEN_ENVIRONMENT.model_copy(),
        phenotype_reference=reference_phenotype(screen_id),
    )


# --------------------------------------------------------------------------- #
# Retention
# --------------------------------------------------------------------------- #
class Retention(BaseModel):
    """Which Table S2 rows are stored, and the ledger that explains the rest."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    kept: pd.DataFrame
    drop_log: DropLog
    reconciliation: LocusTagReconciliation
    reciprocal: ReciprocalPairs


def _pair_label(row: Mapping[str, Any]) -> str:
    """``<donor bnum>__<donor gene> -> <recipient bnum>__<recipient gene>``."""
    return (
        f"{row['donor_id']}__{row['donor_gene']} -> "
        f"{row['recipient_id']}__{row['recipient_gene']}"
    )


def retain(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    genome: EcoliK12Genome,
    donors: Mapping[str, DonorSpec],
    hypomorphic_recipients: frozenset[str],
) -> Retention:
    """Apply the four retention rules, and refuse a ledger that misses a row."""
    unknown = sorted(set(frame["donor_id"]) - set(donors))
    if unknown:
        raise ReleaseContentError(
            f"{TABLE_S2}: {len(unknown)} donors are not in Table S1: {unknown[:5]}"
        )
    names = pd.Series(sorted(set(frame["donor_id"]) | set(frame["recipient_id"])))
    _, reconciliation = reconcile_locus_tags(
        genome, names, label=f"{dataset_name} Table S2 donor and recipient b-numbers"
    )
    resolutions = {name: genome.resolve_gene_name(name) for name in names}
    ambiguous = sorted(
        name
        for name, res in resolutions.items()
        if res.status is GeneNameStatus.AMBIGUOUS
    )
    if ambiguous:
        raise ReleaseContentError(
            "the pinned annotation resolves these released ids to more than one locus, "
            f"which this release's identifier shape does not produce: {ambiguous[:5]}"
        )
    retired = {
        name
        for name, res in resolutions.items()
        if res.status is GeneNameStatus.RETIRED
    }
    remapped = {
        name: str(res.systematic_name)
        for name, res in resolutions.items()
        if name not in retired
        and res.systematic_name is not None
        and res.systematic_name != name
    }

    hypomorph = frame["donor_id"].map(lambda b: donors[b].is_hypomorph) | frame[
        "recipient_id"
    ].isin(hypomorphic_recipients)
    not_a_tag = frame["donor_id"].isin(retired) | frame["recipient_id"].isin(retired)
    was_remapped = frame["donor_id"].isin(set(remapped)) | frame["recipient_id"].isin(
        set(remapped)
    )
    duplicate = frame.duplicated(subset=["donor_id", "recipient_id"], keep=False)

    rules = [
        DropRule(
            rule=RULE_HYPOMORPH,
            description=DROP_RULE_DESCRIPTIONS[RULE_HYPOMORPH],
            n_records=int(hypomorph.sum()),
            items=sorted(
                {b for b in frame.loc[hypomorph, "donor_id"] if donors[b].is_hypomorph}
                | {
                    b
                    for b in frame.loc[hypomorph, "recipient_id"]
                    if b in hypomorphic_recipients
                }
            ),
        ),
        DropRule(
            rule=RULE_NOT_A_TAG,
            description=DROP_RULE_DESCRIPTIONS[RULE_NOT_A_TAG],
            n_records=int((not_a_tag & ~hypomorph).sum()),
            items=sorted(retired),
        ),
        DropRule(
            rule=RULE_REMAPPED,
            description=DROP_RULE_DESCRIPTIONS[RULE_REMAPPED],
            n_records=int((was_remapped & ~hypomorph & ~not_a_tag).sum()),
            items=sorted(f"{name} -> {tag}" for name, tag in remapped.items()),
        ),
        DropRule(
            rule=RULE_DUPLICATE,
            description=DROP_RULE_DESCRIPTIONS[RULE_DUPLICATE],
            n_records=int((duplicate & ~hypomorph & ~not_a_tag & ~was_remapped).sum()),
            items=sorted(
                {_pair_label(dict(row)) for _, row in frame.loc[duplicate].iterrows()}
            ),
        ),
    ]
    kept = frame.loc[~(hypomorph | not_a_tag | was_remapped | duplicate)].reset_index(
        drop=True
    )
    not_itself = sorted(
        name
        for name in set(kept["donor_id"]) | set(kept["recipient_id"])
        if name in retired or name in remapped
    )
    if not_itself:
        raise RuntimeError(
            "a retained id is not a locus tag of the pinned annotation in its own "
            f"right, so it would need a DerivedIdentifierMapping: {not_itself[:5]}"
        )
    kept = kept.assign(screen_id=[donors[b].screen_id for b in kept["donor_id"]])
    unordered = [
        " <-> ".join(sorted((row["donor_id"], row["recipient_id"])))
        for _, row in kept.iterrows()
    ]
    counts = Counter(unordered)
    both_ways = {key for key, n in counts.items() if n > 1}
    signs: dict[str, set[bool]] = {}
    for key, score in zip(unordered, kept["score"], strict=True):
        if key in both_ways:
            signs.setdefault(key, set()).add(bool(score > 0))
    drop_log = DropLog(
        dataset=dataset_name,
        source_records=len(frame),
        kept_records=len(kept),
        dropped_records=len(frame) - len(kept),
        source_donors=int(frame["donor_id"].nunique()),
        source_recipients=int(frame["recipient_id"].nunique()),
        kept_donors=int(kept["donor_id"].nunique()),
        kept_recipients=int(kept["recipient_id"].nunique()),
        n_aggravating=int((kept["score"] < 0).sum()),
        n_alleviating=int((kept["score"] > 0).sum()),
        screen_id_counts={
            str(key): int(value)
            for key, value in kept["screen_id"].value_counts().items()
        },
        rules=rules,
    )
    if sum(rule.n_records for rule in rules) != drop_log.dropped_records:
        raise RuntimeError("drop rules do not account for every dropped row")
    return Retention(
        kept=kept,
        drop_log=drop_log,
        reconciliation=reconciliation,
        reciprocal=ReciprocalPairs(
            n_pairs=len(both_ways),
            n_records=sum(counts[key] for key in both_ways),
            n_sign_disagreements=sum(1 for value in signs.values() if len(value) > 1),
            pairs=sorted(both_ways),
        ),
    )


def stored_rows(kept: pd.DataFrame) -> Iterator[tuple[str, str, str, str, float, str]]:
    """Every stored row in LMDB order, as the record builder's arguments."""
    for row in kept.itertuples(index=False):
        yield (
            str(row.donor_id),
            str(row.donor_gene),
            str(row.recipient_id),
            str(row.recipient_gene),
            float(str(row.score)),
            str(row.screen_id),
        )


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
DATASET_ROOT_REL = "data/torchcell/gene_interaction_babu2014"
#: Records of the full build: 42,705 released pairs minus 3,420 hypomorph pairs, 183
#: pairs on an id the annotation does not carry, 519 pairs on an annotation-remapped
#: b-number and the 4 rows of the two contradictory duplicate pairs.
EXPECTED_RECORDS = 38579


@register_dataset
class GeneInteractionBabu2014Dataset(ExperimentDataset):
    """eSGA interaction scores of E. coli K-12 double deletion mutants (Table S2)."""

    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = REFERENCE_STRAIN_NAME

    def __init__(
        self,
        root: str = "data/torchcell/gene_interaction_babu2014",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        ecoli_genome: EcoliK12Genome | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; ``ecoli_genome`` is injected by the build entry points."""
        self.ecoli_genome = ecoli_genome
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
        """Every consumed file, linked from its own citation key's raw mirror."""
        return [raw.name for raw in all_raw_files()]

    def download(self) -> None:
        """Link each mirror file into ``raw/`` after checking its manifest and sha256.

        Babu's two tables come from Babu's raw mirror and Butland's array roster from
        Butland's own, each checked against that mirror's manifest. Every file is found
        and pinned BEFORE ``raw/`` is populated, so an incomplete mirror leaves no
        half-populated raw directory behind.
        """
        data_root = _data_root()
        manifests = {
            key: load_manifest_of(key, data_root) for key in (CITATION_KEY, BUTLAND_KEY)
        }
        sources = {
            **{
                raw.name: (raw_mirror_dir(data_root), CITATION_KEY) for raw in RAW_FILES
            },
            BUTLAND_RAW_FILE.name: (butland_mirror_dir(data_root), BUTLAND_KEY),
        }
        found: dict[str, Path] = {}
        for raw in all_raw_files():
            mirror, key = sources[raw.name]
            check_manifest_pin(
                raw.mirror_relpath,
                manifest_sha256(manifests[key], raw.mirror_relpath),
                raw.sha256,
            )
            src = mirror / raw.mirror_relpath
            if not src.exists():
                raise RuntimeError(f"required raw artifact missing from mirror: {src}")
            found[raw.name] = src
        os.makedirs(self.raw_dir, exist_ok=True)
        for raw in all_raw_files():
            link_verified(found[raw.name], osp.join(self.raw_dir, raw.name), raw.sha256)
        log.info("Babu 2014 raw files linked into %s (sha256 verified)", self.raw_dir)

    def _genome(self) -> EcoliK12Genome:
        """The injected genome, or the reference strain's default cache (a direct run);
        a genome of another assembly set is refused.
        """
        if self.ecoli_genome is None:
            self.ecoli_genome = bacterial_genome("ecoli", self.REFERENCE_STRAIN)
        expected = BACTERIAL_ASSEMBLY_SETS[self.REFERENCE_STRAIN]
        if self.ecoli_genome.ASSEMBLY_SET != expected:
            raise ValueError(
                f"{type(self).__name__} needs the {expected} genome, got "
                f"{self.ecoli_genome.ASSEMBLY_SET}"
            )
        return self.ecoli_genome

    @post_process
    def process(self) -> None:
        """Parse Table S2 into one record per retained pair, plus the ledgers."""
        verify_raw_files(self.raw_dir, DATA_SHA256)
        frame = read_table_s2(osp.join(self.raw_dir, TABLE_S2))
        donors = read_donors(osp.join(self.raw_dir, TABLE_S1))
        hypomorphic = read_hypomorphic_recipients(
            osp.join(self.raw_dir, BUTLAND_TABLE_S1)
        )
        retention = retain(
            frame,
            dataset_name=self.name,
            genome=self._genome(),
            donors=donors,
            hypomorphic_recipients=hypomorphic,
        )
        drop_log = retention.drop_log
        log.info(
            "Babu 2014: %d released pairs -> %d records (%d aggravating, %d "
            "alleviating); dropped %s; screen sets %s",
            drop_log.source_records,
            drop_log.kept_records,
            drop_log.n_aggravating,
            drop_log.n_alleviating,
            {rule.rule: rule.n_records for rule in drop_log.rules},
            drop_log.screen_id_counts,
        )

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        self._write_ledgers(retention)

        genome_reference = reference_genome()
        references = {
            screen_id: build_reference(self.name, genome_reference, screen_id)
            for screen_id in (SCREEN_THIS_STUDY, SCREEN_BUTLAND)
        }
        env_out, interned_env = self._open_write_lmdb(
            osp.join(self.processed_dir, "lmdb")
        )
        index = 0
        with env_out.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for (
                donor_tag,
                donor_gene,
                recipient_tag,
                recipient_gene,
                score,
                screen_id,
            ) in tqdm(
                stored_rows(retention.kept),
                total=drop_log.kept_records,
                desc="babu2014",
            ):
                experiment = build_experiment(
                    self.name,
                    donor_tag,
                    donor_gene,
                    recipient_tag,
                    recipient_gene,
                    score,
                    screen_id,
                )
                txn.put(
                    f"{index}".encode(),
                    self._intern_record(
                        experiment, references[screen_id], PUBLICATION, itxn
                    ),
                )
                index += 1
        env_out.close()
        interned_env.close()
        if index != drop_log.kept_records:
            raise RuntimeError(
                f"wrote {index} records, the ledger counted {drop_log.kept_records}"
            )
        log.info("Wrote %d Babu 2014 gene-interaction experiments to LMDB", index)

    def _write_ledgers(self, retention: Retention) -> None:
        """The drop log, the identifier reconciliation, the replicate design the schema
        cannot store, the reciprocal-pair census and the files read but not loaded.
        """
        out = Path(self.preprocess_dir)
        (out / "dropped_records.json").write_text(
            retention.drop_log.model_dump_json(indent=2)
        )
        (out / "identifier_reconciliation.json").write_text(
            retention.reconciliation.model_dump_json(indent=2)
        )
        (out / "replicate_structure.json").write_text(
            replicate_structure().model_dump_json(indent=2)
        )
        (out / "reciprocal_pairs.json").write_text(
            retention.reciprocal.model_dump_json(indent=2)
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
    source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/data/{TABLE_S2}",
    citation_key=CITATION_KEY,
    sha256=DATA_SHA256[TABLE_S2],
    method="Table S2, the GI score of every released high-confidence (donor, "
    "recipient) pair; one BacterialGeneInteractionExperiment per row, a two-leaf "
    "BacterialDeletionPerturbation genotype against MG1655 whose donor carries cat and "
    "whose recipient carries kan, reference = the unperturbed conjugant chassis at 0",
    page="Table S2 (pgen.1004120.s023.xls), sheet 'WG_GI_Score_Mar_06_2013'",
    retrieved=DATA_RETRIEVED_AT,
)


def _validate_record(record: Record) -> None:
    """L0: the experiment and its reference validate as the assembly-pinned pair."""
    BacterialGeneInteractionExperiment.model_validate(record["experiment"])
    BacterialGeneInteractionExperimentReference.model_validate(record["reference"])


def _l1_two_distinct_genes(records: Sequence[Record]) -> LevelResult:
    """L1: every record is a digenic pair of two DIFFERENT loci, one donor one recipient."""
    bad: list[str] = []
    for record in records:
        perturbations = record["experiment"]["genotype"]["perturbations"]
        tags = [p["systematic_gene_name"] for p in perturbations]
        cassettes = sorted(str(p["cassette"]) for p in perturbations)
        if (
            len(perturbations) != 2
            or len(set(tags)) != 2
            or cassettes != sorted((DONOR_CASSETTE, RECIPIENT_CASSETTE))
        ):
            bad.append("+".join(tags))
    return LevelResult(
        level=Level.L1,
        name="digenic_pair_of_one_donor_and_one_recipient",
        passed=bool(records) and not bad,
        message=f"{len(records) - len(bad)} of {len(records)} records are two distinct "
        "loci, one cat-marked donor and one kan-marked recipient",
        details={"n_bad": len(bad), "bad_examples": bad[:20]},
    )


def _l2_scores_match_table_s2(
    records: Sequence[Record], released: Mapping[tuple[str, str], float]
) -> LevelResult:
    """L2: every stored score is the released cell of its own (donor, recipient) pair."""
    mismatched: list[str] = []
    for record in records:
        experiment = record["experiment"]
        by_cassette = {
            str(p["cassette"]): str(p["systematic_gene_name"])
            for p in experiment["genotype"]["perturbations"]
        }
        key = (by_cassette[DONOR_CASSETTE], by_cassette[RECIPIENT_CASSETTE])
        stored = float(experiment["phenotype"]["gene_interaction"])
        if key not in released or released[key] != stored:
            mismatched.append(f"{key[0]}->{key[1]} stored {stored}")
    return LevelResult(
        level=Level.L2,
        name="gene_interaction_equals_table_s2_cell",
        passed=bool(records) and not mismatched,
        message=f"{len(records) - len(mismatched)} of {len(records)} stored scores "
        "equal their Table S2 cell",
        details={"n_mismatched": len(mismatched), "examples": mismatched[:20]},
    )


def _l3_sign_convention(records: Sequence[Record]) -> LevelResult:
    """L3: the stored axis is signed and both signs are present, never clamped."""
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
        name="signed_interaction_score_with_zero_reference",
        passed=negative > 0 and positive > 0 and zero == 0 and references == {0.0},
        message=f"{negative} aggravating and {positive} alleviating stored scores, "
        f"{zero} zeros, reference scores {sorted(references)}",
        details={
            "n_negative": negative,
            "n_positive": positive,
            "n_zero": zero,
            "reference_scores": sorted(references),
        },
    )


def _l3_screen_sets(records: Sequence[Record]) -> LevelResult:
    """L3: every record names one of Table S1's two screen sets as its ``screen_id``."""
    counts = Counter(
        str(record["experiment"]["phenotype"]["screen_id"]) for record in records
    )
    expected = {SCREEN_THIS_STUDY, SCREEN_BUTLAND}
    return LevelResult(
        level=Level.L3,
        name="screen_id_is_a_table_s1_screen_set",
        passed=set(counts) == expected,
        message=f"screen sets {dict(counts)}; Table S2 subsumes Butland 2008's 39 "
        "screens and every record says which set it came from",
        details={"counts": dict(counts)},
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
    released: Mapping[tuple[str, str], float],
    universe: set[str],
    expected_count: int,
    dataset_name: str = "gene_interaction_babu2014",
) -> VerificationReport:
    """The L0-L4 gate over built records, given the re-read Table S2 and the MG1655
    gene universe.
    """
    from torchcell.verification.common import shared_rule_results

    report = VerificationReport(
        dataset_name=dataset_name, provenance=VERIFIER_PROVENANCE
    )
    report.add(l0_structural(records, _validate_record))
    report.add(l1_count(len(records), expected_count))
    report.add(_l1_two_distinct_genes(records))
    report.add(_l2_scores_match_table_s2(records, released))
    report.add(_l3_sign_convention(records))
    report.add(_l3_screen_sets(records))
    for result in shared_rule_results(records):
        report.add(result)
    report.add(_l4_containment(records, universe))
    return report


def released_scores(path: str | Path) -> dict[tuple[str, str], float]:
    """Table S2 re-read as ``(donor b-number, recipient b-number) -> GI score``.

    The two contradictory duplicate pairs are excluded, since they have no single
    released value; their rows are dropped from the build for exactly that reason.
    """
    frame = read_table_s2(path)
    duplicate = frame.duplicated(subset=["donor_id", "recipient_id"], keep=False)
    rows = frame.loc[~duplicate]
    return {
        (str(row.donor_id), str(row.recipient_id)): float(str(row.score))
        for row in rows.itertuples(index=False)
    }


def _gene_universe(records: Sequence[Record], base: str) -> set[str]:
    """The L4 universe: every GenBank locus of the assembly the records themselves pin.

    Read from each record's own ``genome_reference`` rather than from a strain named
    here, so a record written against another assembly cannot be judged against
    MG1655's gene set.
    """
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
    released = released_scores(osp.join(abs_root, "raw", TABLE_S2))
    report = verify_records(
        records,
        released=released,
        universe=_gene_universe(records, base),
        expected_count=drops.kept_records,
    )
    library_root = Path(base) / "torchcell-library"
    for value in SOURCED_VALUES.values():
        report.add(audit_sourced_value(value, library_root))
    _write_report(report, osp.join(abs_root, "preprocess"))
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI: ``deposit`` the raw mirrors, ``build`` the dev LMDB, or ``verify`` it."""
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(prog="python -m torchcell.datasets.ecoli.babu2014")
    sub = parser.add_subparsers(dest="command", required=True)
    deposit = sub.add_parser("deposit", help="deposit both raw mirrors")
    deposit.add_argument(
        "--retrieve-into",
        default=None,
        help="re-run every recorded retrieval into this directory and deposit those "
        "bytes; without it the literature mirrors' captured SI files are used",
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
                raw.name: library_dir(
                    BUTLAND_KEY if raw is BUTLAND_RAW_FILE else CITATION_KEY, data_root
                )
                / "si"
                / raw.name
                for raw in all_raw_files()
            }
        for root in deposit_raw_mirror(sources=sources, data_root=data_root):
            print(root)
        return 0
    if args.command == "build":
        dataset = GeneInteractionBabu2014Dataset(
            root=osp.join(data_root, DATASET_ROOT_REL)
        )
        print(f"len = {len(dataset)}")
        return 0
    report = run_verification(data_root)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
