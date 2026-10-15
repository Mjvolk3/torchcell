# torchcell/datasets/scerevisiae/boocock2025
# [[torchcell.datasets.scerevisiae.boocock2025]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/boocock2025
# Test file: tests/torchcell/datasets/scerevisiae/test_boocock2025.py
"""Boocock 2025 one-pot single-cell eQTL panels (eLife, doi:10.7554/eLife.95566;
citation key ``boocockSinglecellEQTLMapping2025``).

Four segregant panels, each a set of MATa haploid progeny of one Bloom 2019 cross,
profiled by 10x 3' single-cell RNA-seq with every cell genotyped from its own
transcribed SNPs:

- ``Ap`` (Table S4 ``A*``): 7,124 cells of 393 PREVIOUSLY GENOTYPED BY x RM segregants
  (Albert 2018), pooled. Each cell is matched to the segregant whose whole-genome
  genotype correlates best with the cell's HMM genotype probabilities (the release's
  ``segMatch.RDS``), so a record is one SEGREGANT: the counts of its cells summed
  (pseudobulk) and its genome-sequenced genotype.
- ``A`` (BY x RM, 27,744 cells), ``B`` (YJM145 x YPS163, 44,784) and ``3004`` (Table S4
  ``C``, CBS2888 x YJM981, 6,595): DE NOVO segregants from a random-spore prep, one
  meiotic product per cell, so a record is one CELL: its counts and its own HMM
  posterior. Duplicate genotypes (cells of one spore clone) are kept as the authors kept
  them for A and B; for 3004 the authors already reduced 14,823 cells to 6,595 unique
  ones.

GENOTYPE (``SegregantGenotype``, a haplotype mosaic against two sha256-pinned parent
assemblies, never ``SequenceVariantPerturbation``). The parents are the Bloom 2019
crosses (A, B, 3004) and are pinned exactly as Bloom pinned them: BY to the SGD R64-4-1
reference sequence, the other five to their member of the Peter 2018 1011 assemblies
tarball in the genomes tier. No new assembly set is needed. The parent order is the
release's allele coding: the HMM posterior is P(parent 2), with parent 2 = RM, YPS163
and YJM981 (Table S4 ``effect_directions`` ``-BY,+RM``, ``-YJM145,+YPS163``,
``-CBS2888,+YJM981``), confirmed per panel at L4 by the MATa locus, which every sorted
cell inherits from its MATa parent.

- De novo cells: the release's per-cell posterior at the LD-pruned markers
  (``segData.RDS`` ``Gsub``, 2,190 to 3,268 markers) is stored as runs of constant
  called parent (posterior > 0.5) AND constant posterior level, the level being
  P(called parent) on a ``POSTERIOR_STEP`` grid. Re-expanding the blocks returns every
  marker's called parent exactly and its posterior within ``POSTERIOR_STEP / 2``; that
  round trip is the L2 check. The step is a measured cardinality trade-off (dendron
  note): a pure called-parent run would binarize the posterior, and an exact encoding
  is one block per marker.
- Pooled segregants: the whole-genome-sequencing hard calls the cells were matched to
  (``gdata_42k.RData``, 1,012 segregants x 42,052 markers, -1 = BY, +1 = RM), run-length
  encoded with posterior 1.0, as Bloom 2019 encodes its hard calls.

PHENOTYPE (``SingleCellExpressionPhenotype``, the one single-cell UMI-count leaf it
shares with Jackson 2020 since #1016): UMI counts over the panel's analysis transcripts
(``Yr``, the transcripts with nonzero counts in at least 128 cells: 4,995 / 5,435 / 5,556
/ 4,696, Table S4 ``n_transcripts``), nonzero counts only against a pinned feature
universe; ``total_umis`` is the release's ``nUMI`` over every feature
(``barcode.features``), the size factor the authors model; ``cell_cycle_stage_counts``
is the authors' per-cell stage assignment (``cell.cycle.df``), the cell-cycle occupancy
the growth/reproduction tradeoff is read from; ``library_cell_counts`` names the 10x
library, the authors' batch. ``cell_barcodes``, ``cell_libraries`` and
``cell_total_umis`` name the summed cells (barcode, library and per-cell ``nUMI``): a
barcode repeats across a panel's libraries, so the pair identifies the cell. No cassette
transcript is counted apart from the genes (the alignment reference is SGD R64-2-1), so
``marker_umi_count`` is a typed gap. The reference of each panel is the panel pseudobulk
(every cell summed), an absolute baseline as in Caudal 2024, not a centered zero.

ENVIRONMENT. Every culture grew at 30 C. The pooled panel grew in minimal YNB medium,
the paper's own restatement of ``SM`` (the same sentence also names YPD for the
overnight step; flagged). The de novo panels grew in 2x YNB + 2% glucose added to FACS
sheath fluid (``YNB_SHEATH_FLUID_BOOCOCK2025``); stated for cross A, a documented
representative for B and C, whose cells were "captured as early as practicable".

NOT LOADED (each a ledger row in the dendron note): the eQTL, hotspot and cell-cycle
occupancy QTL tables (derived analyses, not phenotypes); the F1 diploid allele-specific
expression cells (a diploid hybrid has no ``SegregantGenotype``); the GPA1
allele-replacement single cells, growth rates and mating efficiencies (engineered BY4741
derivatives, a separate record family). Raw reads (SRA PRJNA1049497) and the walled Dryad
deposit (doi:10.5061/dryad.xgxd254qb; HTTP 401 without a bearer token) are not
deposited; nothing loaded depends on either.

RAW MIRROR. ``$DATA_ROOT/torchcell-raw/boocockSinglecellEQTLMapping2025/`` with a
``manifest.json`` (``torchcell.literature.manifest.Manifest``): the four ``segData.RDS``
and ``segMatch.RDS`` members of the Zenodo deposit (one 111.9 GB ``.tar.gz``; each
member a ``tar_member`` retrieval pinned to the container's sha256), the Albert 2018
lookup genotypes from the authors' Dropbox link, Supplementary file 1 (Table S1 parents,
Table S4 counts) and the R files quoted below at their pinned commit.
"""

from __future__ import annotations

import hashlib
import logging
import os
import os.path as osp
import re
import shutil
from collections import Counter
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from torchcell.data import (
    ExperimentDataset,
    Organism,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.media import SM, YNB_SHEATH_FLUID_BOOCOCK2025, restated
from torchcell.datamodels.pydant import ModelStrict
from torchcell.datamodels.schema import (
    ArtifactRef,
    Environment,
    Experiment,
    ExperimentReference,
    HaplotypeBlock,
    Media,
    Publication,
    ReferenceGenome,
    SegregantExpressionExperiment,
    SegregantExpressionExperimentReference,
    SegregantGenotype,
    SegregantParent,
    SingleCellExpressionPhenotype,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datasets.scerevisiae.bloom2019 import (
    PARENT_PETER_ID,
    Marker,
    ParentAssemblyRefs,
    encode_blocks,
    expand_blocks,
    parse_marker,
)
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence import GeneSet
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY = "boocockSinglecellEQTLMapping2025"
PAPER_DOI = "10.7554/eLife.95566"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
LIBRARY_DIR_REL = f"torchcell-library/{CITATION_KEY}"
#: sha256 of the mirrored MinerU ``paper.md`` (the library key's ``manifest.json``).
PAPER_MD_SHA256 = "a2920750e12fb9977f36a687bb9f73aa0382527de84270b44609a3338bbeafe4"

ZENODO_DOI = "10.5281/zenodo.12695127"
ZENODO_URL = (
    "https://zenodo.org/api/records/12695128/files/"
    "additional_data_boocock_et_al_2023.tar.gz/content"
)
#: The Zenodo record's own checksum of the container (``md5:`` in the files API).
ZENODO_CONTAINER_MD5 = "d625073dbfe53914a7f38fa60bfec447"
#: sha256 of the same container, computed over the bytes this repo retrieved.
ZENODO_CONTAINER_SHA256 = (
    "ac3c27545462708ea6ac96e7a496a4c41d391041cd3806744920f82c64d57942"
)
ZENODO_CONTAINER_BYTES = 111_900_721_589

GDATA_URL = (
    "https://www.dropbox.com/scl/fi/e87xbf9ljr554wsmk19qx/gdata_42k.RData"
    "?rlkey=ydulm25lz8mwa88vcul2hop1o&dl=1"
)
SI1_URL = (
    "https://pmc-oa-opendata.s3.amazonaws.com/PMC11903034.1/elife-95566-supp1.xlsx"
)
SI1_KEY = "PMC11903034.1/elife-95566-supp1.xlsx"
DRYAD_DOI = "10.5061/dryad.xgxd254qb"
SRA_BIOPROJECT = "PRJNA1049497"
CODE_REPO = "joshsbloom/single_cell_eQTL"
CODE_COMMIT = "9c5f66a3185486fe38437668a1b6e37b34108d3e"
#: The R files whose lines this module and its verifier quote, by repo path.
CODE_FILES: dict[str, str] = {
    "processSegregants.R": "yeast/code/processSegregants.R",
    "processSegregantsSetup.R": "yeast/code/processSegregantsSetup.R",
    "processSegregantsPrevGeno.R": "yeast/code/processSegregantsPrevGeno.R",
    "processSegregantsGlobalVars.R": "yeast/code/processSegregantsGlobalVars.R",
    "additional_fxs.R": "yeast/code/additional_fxs.R",
    "runHMM_custom.R": "yeast/code/runHMM_custom.R",
}

SEGMATCH_NAME = "segMatch_Ap.RDS"
GDATA_NAME = "gdata_42k.RData"
SI1_NAME = "si1.xlsx"

#: Grid step of the stored posterior level, P(called parent) rounded to a multiple of
#: it; a re-expanded marker posterior is within half a step of the released value.
POSTERIOR_STEP = 0.05
#: Size of the stored posterior grid in steps: levels run 10..20 for a step of 0.05.
_LEVELS_PER_UNIT = round(1 / POSTERIOR_STEP)

DE_NOVO_CALL_METHOD = (
    "per-cell HMM posterior P(parent 2) at the cross's LD-pruned markers, released as "
    "segData.RDS Gsub (prior 0.5 per parental allele, binomial emission at sequencing "
    "error 0.005, transitions from the Bloom 2019 genetic maps; LD pruning at r > 0.999, "
    "processSegregantsSetup.R LDprune), encoded as runs of constant called parent "
    f"(posterior > 0.5) and constant P(called parent) on a {POSTERIOR_STEP} grid"
)
POOLED_CALL_METHOD = (
    "whole-genome-sequencing hard calls of the previously genotyped segregant "
    "(gdata_42k.RData, -1 = BY, +1 = RM, the lookup genotypes of Albert 2018), the "
    "segregant each cell was matched to by maximal correlation with its HMM genotype "
    "probabilities (segMatch.RDS best_match_seg; processSegregantsPrevGeno.R getSegMatch)"
)


def _paper_sv(value: object, quote: str, *, note: str | None = None) -> SourcedValue:
    """A SourcedValue anchored to the mirrored Boocock 2025 ``paper.md``."""
    return SourcedValue(
        value=value,
        provenance=Provenance(
            source_uri="paper.md", citation_key=CITATION_KEY, sha256=PAPER_MD_SHA256
        ),
        quote=quote,
        note=note,
    )


# Quotes are verbatim substrings of the MinerU ``paper.md`` (LaTeX markup kept; the
# non-breaking spaces of the OCR are written as \xa0).
SOURCED: dict[str, SourcedValue] = {
    "pilot_panel": _paper_sv(
        "393 previously genotyped segregants, 7,124 cells",
        "pooled 393 previously genotyped haploid segregants from a cross between a lab "
        "strain (BY) and a wine strain (RM)",
    ),
    "pilot_present": _paper_sv(
        "393 of the 480 MATa segregants were profiled",
        "393 of the 480 $( 8 1 . 9 \\% )$ MATa segregants were present in the "
        "scRNA-seq data.",
    ),
    "pilot_cells_per_segregant": _paper_sv(
        "median 17 cells per segregant; 277 segregants with more than ten cells",
        "We observed a median of 17\xa0cells per segregant, with 277 of the segregants "
        "sampled more than ten times",
    ),
    "pilot_agreement": _paper_sv(
        "median per-cell genotype agreement with whole-genome sequencing 92.5%",
        "median genotype agreement $9 2 . 5 \\%$ )",
    ),
    "segregant_match": _paper_sv(
        "a cell is the previously genotyped segregant whose genotype correlates best",
        "Segregant identity was determined by picking the previously genotyped "
        "segregant with genotypes that were most correlated with the genotype "
        "probability vector for a given single cell.",
    ),
    "pilot_medium": _paper_sv(
        "minimal YNB (6.7 g/L YNB w/o amino acids, 2% glucose), 30 C",
        "This pool of strains was then transferred to minimal YNB medium $\\left( 6 . 7 "
        "~ \\mathsf { g } / \\right|$ Difco Yeast Nitrogen Base w/o Amino Acids, $2 \\%$ "
        "glucose) and grown overnight at $3 0 ^ { \\circ } \\mathsf { C }$ in YPD $2 \\%$ "
        "bacto peptone, $1 \\%$ yeast extract, $2 \\%$ glucose).",
        note="the sentence names both minimal YNB and YPD; the YNB medium is recorded "
        "because the pool was transferred to it before the dilution and mid-log growth, "
        "and the reading is flagged for review in the dendron note",
    ),
    "de_novo_temperature": _paper_sv(
        30.0,
        "they were transferred to a shaking incubator and grown at $3 0 ^ { \\circ } "
        "\\mathsf { C }$ .",
    ),
    "de_novo_medium_a": _paper_sv(
        "cross A harvested in 2x YNB with sheath fluid",
        "For our experiments with the BY and RM cross (YLK3051), the cells were "
        "harvested in $2 \\times \\mathsf { Y N B }$ medium containing sheath fluid.",
    ),
    "de_novo_medium_bc": _paper_sv(
        "crosses B and C captured as early as practicable",
        "For our other de novo eQTL experiments, the cells were captured as early as "
        "practicable.",
        note="no harvest medium is named for B and C; the sheath-fluid medium is their "
        "documented representative",
    ),
    "hmm": _paper_sv(
        "per-cell HMM over the expressed SNPs",
        "We used a hidden Markov model (HMM) to reconstruct the patterns of inheritance "
        "of parental alleles in each cell based on the observed genotypes at expressed "
        "SNPs.",
    ),
    "hmm_prior": _paper_sv(
        "prior 0.5",
        "We defined prior genotype probabilities as 0.5 for each parental variant.",
    ),
    "ld_pruning": _paper_sv(
        "LD-pruned markers, r > 0.999",
        "Genotype probabilities were standardized, and markers in very high LD $r > 0 . "
        "9 9 9 )$ were pruned.",
    ),
    "expressed_transcripts": _paper_sv(
        "analysis transcripts: nonzero in at least 128 cells",
        "Transcripts with non-zero counts in at least 128\xa0cells were used for "
        "downstream analyses.",
    ),
    "duplicates": _paper_sv(
        "only cross C was reduced to unique genotypes",
        "This filtering step was only needed for the cross of YJM981 and CBS2888, and "
        "the procedure reduced the number of cells from 14,823 to 6595.",
    ),
    "batch": _paper_sv(
        "a 10x library is a batch",
        "Each sequencing library was treated as a different ‘batch’ for all downstream "
        "analyses.",
    ),
    "cross_names": _paper_sv(
        "cross A = BY x RM; B = YJM145 x YPS163; C = CBS2888 x YJM981",
        "Hereafter, we refer to the ${ \\mathsf { B Y } } \\times { \\mathsf { R M } }$ "
        "cross as cross A and the two new crosses as crosses B and C, respectively.",
    ),
    "cell_cycle": _paper_sv(
        "five haploid cell-cycle stages",
        "We classified individual haploid yeast cells into five different cell-cycle "
        "stages (M/G1, G1, G1/S, S, and G2/M)",
    ),
    "alignment_reference": _paper_sv(
        "reads counted against SGD R64-2-1 with 3' UTRs amended; no cassette sequence",
        "Sequencing reads were analyzed using Cellranger (version 5.01) using the S2888C "
        "reference genome (SGD, R64-2-1)",
    ),
}

#: No cassette transcript is counted apart from the genes: the alignment reference is the
#: SGD genome (no cassette sequence named, where Jackson 2020 states adding its two), and
#: ``segData.RDS`` releases per-feature counts only for the analysis transcripts.
MARKER_GAP = ProvenanceGap(
    field="marker_umi_count",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=SOURCED["alignment_reference"].provenance,
    note="reads were counted against the SGD R64-2-1 reference with 3' UTRs amended and "
    "no cassette sequence named, so no drug-cassette transcript is counted apart from "
    "the genes; segData.RDS releases per-feature counts only for the analysis "
    "transcripts (Yr), every one an S288C gene",
)

PILOT_MEDIUM: Media = restated(SM, SOURCED["pilot_medium"])
"""Minimal YNB of the pooled panel: library ``SM`` (6.7 g/L YNB w/o amino acids + 2%
glucose) plus the paper's own sentence. Same ``media_identity`` as ``SM``."""

GROWTH_TEMPERATURE_C = 30.0

# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
#: Boocock parent label (Table S4 ``parent1``/``parent2``) -> Bloom 2019 label, whose
#: ``PARENT_PETER_ID`` entry pins the assembly. Same strains: the de novo crosses are
#: the Bloom 2019 crosses A, B and 3004, whose genetic maps the HMM uses.
BLOOM_LABEL: dict[str, str] = {
    "BY": "BYa",
    "RM": "RMx",
    "YJM145": "YJM145x",
    "YPS163": "YPS163a",
    "CBS2888": "CBS2888a",
    "YJM981": "YJM981x",
}


class PanelSpec(ModelStrict):
    """One released segregant panel (a ``data/out/combined/<panel>`` directory)."""

    panel: str
    cross: str
    s4_label: str
    strain: str
    pooled: bool
    parent_1: str
    parent_2: str
    mata_parent: int
    libraries: tuple[str, ...]
    expected_cells: int
    expected_transcripts: int
    median_umi: float

    @property
    def segdata_name(self) -> str:
        """The ``raw/`` file name of this panel's ``segData.RDS``."""
        return f"segData_{self.panel}.RDS"

    @property
    def member(self) -> str:
        """The panel's ``segData.RDS`` member path inside the Zenodo tarball."""
        return f"data/out/combined/{self.panel}/segData.RDS"


#: Libraries per panel are the release's ``sets`` in processSegregantsGlobalVars.R;
#: counts are Table S4 (median UMI as tables/s4.csv prints it, 1513.5 for A*).
PANELS: dict[str, PanelSpec] = {
    "Ap": PanelSpec(
        panel="Ap",
        cross="A",
        s4_label="A*",
        strain="YLK1993",
        pooled=True,
        parent_1="BY",
        parent_2="RM",
        mata_parent=1,
        libraries=("00_BYxRM_480MatA_1", "00_BYxRM_480MatA_2"),
        expected_cells=7124,
        expected_transcripts=4995,
        median_umi=1513.5,
    ),
    "A": PanelSpec(
        panel="A",
        cross="A",
        s4_label="A",
        strain="YLK3051",
        pooled=False,
        parent_1="BY",
        parent_2="RM",
        mata_parent=1,
        libraries=("03_ByxRM_51-_1-2", "04_ByxRM_51-"),
        expected_cells=27744,
        expected_transcripts=5435,
        median_umi=4017.0,
    ),
    "B": PanelSpec(
        panel="B",
        cross="B",
        s4_label="B",
        strain="YLK3301",
        pooled=False,
        parent_1="YJM145",
        parent_2="YPS163",
        mata_parent=2,
        libraries=(
            "01_2444_44_1-2",
            "02_2444_44-",
            "07_2444-cross-1",
            "07_2444-cross-2",
        ),
        expected_cells=44784,
        expected_transcripts=5556,
        median_umi=2130.0,
    ),
    "3004": PanelSpec(
        panel="3004",
        cross="3004",
        s4_label="C",
        strain="YLK3004",
        pooled=False,
        parent_1="CBS2888",
        parent_2="YJM981",
        mata_parent=1,
        libraries=("10_3004_cross_10k_Feb_21", "11_3004_cross_5k_Feb_21"),
        expected_cells=6595,
        expected_transcripts=4696,
        median_umi=1982.0,
    ),
}
N_POOLED_SEGREGANTS = 393
EXPECTED_RECORDS = N_POOLED_SEGREGANTS + sum(
    p.expected_cells for p in PANELS.values() if not p.pooled
)


# --------------------------------------------------------------------------- #
# Reading the release
# --------------------------------------------------------------------------- #
class PanelRelease(ModelStrict):
    """The parts of one ``segData.RDS`` a record is built from, row-aligned by cell.

    ``counts`` is cells x transcripts (CSR rows), ``posterior`` cells x markers,
    P(parent 2) at ``markers`` (sorted by chromosome, position).
    """

    model_config = {"arbitrary_types_allowed": True, "frozen": True}

    barcodes: list[str]
    libraries: list[str]
    stages: list[str]
    n_umi: list[int]
    transcripts: list[str]
    counts: Any
    markers: list[Marker]
    posterior: Any


def _integral(values: np.ndarray[Any, Any], what: str) -> np.ndarray[Any, Any]:
    """``values`` as int64 after checking every entry is a whole number."""
    if not np.all(values == np.round(values)):
        raise ValueError(f"{what}: non-integral counts in the release")
    return values.astype(np.int64)


def read_panel_release(path: str | Path) -> PanelRelease:
    """Parse one ``segData.RDS`` (R list: Yr, Gsub, ..., barcode.features, cell.cycle.df).

    The five per-cell tables are aligned by row in the release (``processSegregants.R``
    subsets ``vg``, ``counts`` and ``cc.df`` by one index), which is asserted on the
    barcodes rather than assumed.
    """
    import rdata
    import scipy.sparse as sp

    obj = rdata.read_rds(str(path))
    yr = obj["Yr"]
    shape = (int(yr.Dim[0]), int(yr.Dim[1]))
    counts = sp.csc_matrix(
        (_integral(np.asarray(yr.x), "Yr"), np.asarray(yr.i), np.asarray(yr.p)),
        shape=shape,
    ).tocsr()
    cells = [str(b) for b in yr.Dimnames[0]]
    transcripts = [str(t) for t in yr.Dimnames[1]]
    gsub = obj["Gsub"]
    gsub_cells = [str(c) for c in gsub.coords["dim_0"].values]
    marker_names = [str(c) for c in gsub.coords["dim_1"].values]
    barcode_features = obj["barcode.features"]
    cell_cycle = obj["cell.cycle.df"]
    columns = [
        gsub_cells,
        [str(b) for b in barcode_features["barcode"]],
        [str(b) for b in cell_cycle["cell_name"]],
    ]
    if any(column != cells for column in columns):
        raise ValueError(
            f"{path}: Yr, Gsub, barcode.features and cell.cycle.df rows differ"
        )
    markers = [parse_marker(name, i) for i, name in enumerate(marker_names)]
    order = sorted(range(len(markers)), key=lambda i: _marker_key(markers[i]))
    if order != list(range(len(markers))):
        raise ValueError(
            f"{path}: Gsub markers are not in (chromosome, position) order"
        )
    n_umi = _integral(np.asarray(barcode_features["nUMI"], dtype=float), "nUMI")
    return PanelRelease(
        barcodes=cells,
        libraries=[str(x) for x in cell_cycle["named_dataset"]],
        stages=[str(x) for x in cell_cycle["cell_cycle"]],
        n_umi=[int(x) for x in n_umi],
        transcripts=transcripts,
        counts=counts,
        markers=markers,
        posterior=np.asarray(gsub.values, dtype=np.float64),
    )


class SegregantMatch(ModelStrict):
    """``segMatch.RDS``: per pooled-panel cell, the matched segregant and its correlation."""

    segregant: list[str]
    correlation: list[float]
    log_n_umi: list[float]


def read_segregant_match(path: str | Path) -> SegregantMatch:
    """Parse ``segMatch.RDS`` (``data.frame(mmp1, best_match_seg, best_match_seg.r)``)."""
    import rdata

    frame = rdata.read_rds(str(path))
    return SegregantMatch(
        segregant=[str(s) for s in frame["best_match_seg"]],
        correlation=[float(r) for r in frame["best_match_seg.r"]],
        log_n_umi=[float(v) for v in frame["log.barcode.features.nUMI."]],
    )


_GDATA_MARKER = re.compile(r"^(chr[IVX]+):(\d+)_([ACGT]+)/([ACGT,]+)$")


def read_lookup_genotypes(path: str | Path) -> tuple[list[str], list[Marker], Any]:
    """``gdata_42k.RData``: segregant ids, the 42,052 markers sorted, and the -1/+1 calls.

    Marker names are ``chr:pos_ref/alt``; the matrix columns are permuted into
    (chromosome, position) order and the permutation asserted to be complete.
    """
    import rdata

    gdata = rdata.read_rda(str(path))["gdata"]
    ids = [str(s) for s in gdata.coords["dim_0"].values]
    names = [str(c) for c in gdata.coords["dim_1"].values]
    markers: list[Marker] = []
    for column, name in enumerate(names):
        match = _GDATA_MARKER.match(name)
        if match is None:
            raise ValueError(f"lookup marker {name!r} is not chr:pos_ref/alt")
        chrom, pos, ref, alt = match.groups()
        markers.append(
            Marker(
                chromosome=chrom,
                position=int(pos),
                ref=ref,
                alt=alt,
                index=column,
                column=column,
            )
        )
    order = sorted(range(len(markers)), key=lambda i: _marker_key(markers[i]))
    calls = np.asarray(gdata.values, dtype=np.float64)[:, order]
    if not np.isin(calls, (-1.0, 1.0)).all():
        raise ValueError("lookup genotypes hold a call outside {-1, +1}")
    return ids, [markers[i] for i in order], calls


_CHROMOSOMES = [
    "chrI", "chrII", "chrIII", "chrIV", "chrV", "chrVI", "chrVII", "chrVIII", "chrIX",
    "chrX", "chrXI", "chrXII", "chrXIII", "chrXIV", "chrXV", "chrXVI",
]  # fmt: skip
_CHROM_ORDER = {name: i for i, name in enumerate(_CHROMOSOMES)}


def _marker_key(marker: Marker) -> tuple[int, int]:
    return (_CHROM_ORDER[marker.chromosome], marker.position)


def read_parent_backgrounds(path: str | Path) -> dict[str, dict[str, str]]:
    """Table S1 (Supplementary file 1): strain -> {parent label: genotype cell, verbatim}.

    The S1 ``Parent 1``/``Parent 2`` order is not the allele-coding order (cross B lists
    YPS163 first), so each cell is assigned to the parent label it starts with.
    """
    frame = pd.read_excel(path, sheet_name="TableS1. Yeast Strains", header=None)
    header_row = next(
        i for i in range(len(frame)) if str(frame.iloc[i, 0]).strip() == "Strain"
    )
    out: dict[str, dict[str, str]] = {}
    for i in range(header_row + 1, len(frame)):
        strain = str(frame.iloc[i, 0]).strip()
        if strain in {spec.strain for spec in PANELS.values()}:
            cells = [str(frame.iloc[i, 1]).strip(), str(frame.iloc[i, 2]).strip()]
            spec = next(s for s in PANELS.values() if s.strain == strain)
            assigned: dict[str, str] = {}
            for label in (spec.parent_1, spec.parent_2):
                hits = [c for c in cells if c.startswith(f"{label} ")]
                if len(hits) != 1:
                    raise ValueError(f"Table S1 {strain}: no unique cell for {label}")
                assigned[label] = hits[0]
            out[strain] = assigned
    missing = {spec.strain for spec in PANELS.values()} - set(out)
    if missing:
        raise ValueError(f"Table S1 lacks the panel strains {sorted(missing)}")
    return out


# --------------------------------------------------------------------------- #
# Posterior blocks
# --------------------------------------------------------------------------- #
def encode_posterior_blocks(
    posterior: np.ndarray[Any, Any], markers: Sequence[Marker]
) -> list[HaplotypeBlock]:
    """Runs of constant called parent and constant posterior level, per chromosome.

    ``posterior`` is P(parent 2) at ``markers`` (sorted). The called parent is 2 above
    0.5 and 1 below; the level is P(called parent) rounded to the ``POSTERIOR_STEP``
    grid. A marker at exactly 0.5 has no called parent and raises, since no grid level
    can state "no call" and dropping the marker would hide it.
    """
    if np.any(posterior == 0.5):
        raise ValueError("a marker posterior of exactly 0.5 has no called parent")
    parent = np.where(posterior > 0.5, 2, 1).astype(np.int8)
    q = np.where(parent == 2, posterior, 1.0 - posterior)
    level = np.rint(q * _LEVELS_PER_UNIT).astype(np.int64)
    chrom = np.fromiter((_CHROM_ORDER[m.chromosome] for m in markers), dtype=np.int64)
    change = np.ones(len(markers), dtype=bool)
    change[1:] = (
        (parent[1:] != parent[:-1])
        | (level[1:] != level[:-1])
        | (chrom[1:] != chrom[:-1])
    )
    # Python ints, so a Sequence index is an int for every numpy stub version.
    starts: list[int] = np.flatnonzero(change).tolist()
    ends = [*(s - 1 for s in starts[1:]), len(markers) - 1]
    return [
        HaplotypeBlock(
            chromosome=markers[s].chromosome,
            start=markers[s].position,
            end=markers[e].position,
            parent=int(parent[s]),  # type: ignore[arg-type]
            posterior=round(int(level[s]) / _LEVELS_PER_UNIT, 2),
            n_markers=e - s + 1,
        )
        for s, e in zip(starts, ends)
    ]


def expand_posterior_blocks(
    blocks: Sequence[HaplotypeBlock], markers: Sequence[Marker]
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Inverse of ``encode_posterior_blocks``: per marker, (called parent, level)."""
    parent = np.zeros(len(markers), dtype=np.int8)
    level = np.zeros(len(markers), dtype=np.float64)
    i = 0
    for block in blocks:
        end = i + block.n_markers
        run = markers[i:end]
        if (
            not run
            or run[0].chromosome != block.chromosome
            or run[0].position != block.start
            or run[-1].position != block.end
        ):
            raise ValueError(f"block {block} does not align with the markers at {i}")
        parent[i:end] = block.parent
        level[i:end] = block.posterior
        i = end
    if i != len(markers):
        raise ValueError(f"blocks cover {i} of {len(markers)} markers")
    return parent, level


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_root() -> str:
    from dotenv import load_dotenv

    load_dotenv()
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/boocockSinglecellEQTLMapping2025``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def raw_relpaths() -> dict[str, str]:
    """``raw/`` file name -> mirror-relative path, for every file the loader reads."""
    rel = {spec.segdata_name: f"data/zenodo/{spec.member}" for spec in PANELS.values()}
    rel[SEGMATCH_NAME] = "data/zenodo/data/out/combined/Ap/segMatch.RDS"
    rel[GDATA_NAME] = f"data/dropbox/{GDATA_NAME}"
    rel[SI1_NAME] = f"si/{SI1_NAME}"
    return rel


def code_relpath(name: str) -> str:
    """Mirror-relative path of one quoted R file."""
    return f"code/{name}"


def deposit_raw_mirror(
    *,
    zenodo_members_dir: str | Path,
    gdata_path: str | Path,
    si1_path: str | Path,
    code_dir: str | Path,
    zenodo_container_sha256: str,
    retrieved_at: str,
    data_root: str | None = None,
) -> Path:
    """Copy the retrieved files into the raw mirror and write its ``manifest.json``.

    Run once by hand after the retrievals in the module docstring; idempotent by sha256
    (a present file with the recorded hash is left alone, a differing one raises).
    ``zenodo_members_dir`` holds the members extracted from the Zenodo tarball at their
    tar paths; each is recorded as a ``tar_member`` retrieval pinned to the container's
    sha256 (``zenodo_container_sha256``), and the member's own sha256 is its anchor.
    """
    root = raw_mirror_dir(data_root)
    rel = raw_relpaths()
    sources: dict[str, tuple[Path, RetrievalRecord]] = {}

    def record(
        method: RetrievalMethod,
        source_url: str,
        retriever: str,
        sha: str,
        **params: Any,
    ) -> RetrievalRecord:
        return RetrievalRecord(
            method=method,
            source_url=source_url,
            retriever=f"torchcell.literature.retrieve.{retriever}",
            params=params,
            sha256=sha,
            retrieved_at=retrieved_at,
        )

    zenodo_members = {spec.segdata_name: spec.member for spec in PANELS.values()}
    zenodo_members[SEGMATCH_NAME] = "data/out/combined/Ap/segMatch.RDS"
    for name, member in zenodo_members.items():
        src = Path(zenodo_members_dir) / member
        sources[rel[name]] = (
            src,
            record(
                RetrievalMethod.zenodo,
                ZENODO_URL,
                "tar_member",
                _sha256(src),
                url=ZENODO_URL,
                member=member,
                container_sha256=zenodo_container_sha256,
            ),
        )
    sources[rel[GDATA_NAME]] = (
        Path(gdata_path),
        record(
            RetrievalMethod.direct_url,
            GDATA_URL,
            "direct_url",
            _sha256(gdata_path),
            url=GDATA_URL,
        ),
    )
    sources[rel[SI1_NAME]] = (
        Path(si1_path),
        record(
            RetrievalMethod.pmc_cloud,
            SI1_URL,
            "pmc_cloud_object",
            _sha256(si1_path),
            key=SI1_KEY,
        ),
    )
    for name, repo_path in CODE_FILES.items():
        src = Path(code_dir) / name
        url = f"https://raw.githubusercontent.com/{CODE_REPO}/{CODE_COMMIT}/{repo_path}"
        sources[code_relpath(name)] = (
            src,
            record(
                RetrievalMethod.direct_url,
                url,
                "direct_url",
                _sha256(src),
                url=url,
                repo=CODE_REPO,
                commit=CODE_COMMIT,
                path=repo_path,
            ),
        )
    files: list[ArtifactRecord] = []
    for relpath, (src, retrieval) in sources.items():
        dest = root / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if _sha256(dest) != retrieval.sha256:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            shutil.copy2(src, dest)
        files.append(
            ArtifactRecord(
                path=relpath,
                role=ROLE_RAW_DATA,
                bytes=dest.stat().st_size,
                sha256=retrieval.sha256,
                source=retrieval.source_url,
                retrieval=retrieval,
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title=(
            "Single-cell eQTL mapping in yeast reveals a tradeoff between growth and "
            "reproduction"
        ),
        files=files,
        si_data_sources=[
            f"https://doi.org/{ZENODO_DOI}",
            GDATA_URL,
            f"https://github.com/{CODE_REPO}/tree/{CODE_COMMIT}",
            f"https://doi.org/{DRYAD_DOI}",
            f"https://www.ncbi.nlm.nih.gov/bioproject/{SRA_BIOPROJECT}",
        ],
        si_expected=[
            "Zenodo additional_data_boocock_et_al_2023.tar.gz members "
            "data/out/combined/{Ap,A,B,3004}/segData.RDS and Ap/segMatch.RDS",
            f"Dryad {DRYAD_DOI} additional_data.tar (4,739,652,608 bytes, sha256 "
            "aa38021b2d26b5fd90f1a671c09aadad342f2d732af0266509ee4ec1ce79de9b): NOT "
            "retrieved; the download route answers HTTP 401 'Unauthorized, must have "
            "current bearer token' (2026-10-11). Manual recipe: open "
            f"https://doi.org/{DRYAD_DOI} in a browser, use 'Download dataset', deposit "
            "under data/dryad/ with RetrievalMethod.manual_browser. Nothing loaded "
            "depends on it.",
        ],
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return root


def load_manifest(data_root: str | None = None) -> Manifest:
    """Read the raw mirror's ``manifest.json``."""
    path = raw_mirror_dir(data_root) / "manifest.json"
    return Manifest.model_validate_json(path.read_text())


def manifest_sha256(manifest: Manifest, relpath: str) -> str:
    """The recorded sha256 of one mirror file."""
    for artifact in manifest.files:
        if artifact.path == relpath:
            return artifact.sha256
    raise KeyError(f"{relpath} is not in the raw-mirror manifest")


# --------------------------------------------------------------------------- #
# Parents
# --------------------------------------------------------------------------- #
def build_parent(
    label: str, background: str, assemblies: ParentAssemblyRefs
) -> SegregantParent:
    """A Boocock parent pinned to its assembly, by its Bloom 2019 Peter id.

    BY's ``assembly_ref`` is the S288C reference sequence; every other parent's is the
    1011 tarball narrowed to its member, and a Peter id absent from the member index
    raises.
    """
    peter = PARENT_PETER_ID[BLOOM_LABEL[label]]
    if peter is None:
        return SegregantParent(
            name=label,
            peter_strain_id=None,
            assembly_ref=assemblies.s288c,
            engineered_background=background,
        )
    if peter not in assemblies.index:
        raise RuntimeError(
            f"parent {label}: Peter id {peter} is not in the member index"
        )
    tarball = assemblies.tarball
    return SegregantParent(
        name=label,
        peter_strain_id=peter,
        assembly_ref=ArtifactRef(
            tier=tarball.tier,
            key=tarball.key,
            path=tarball.path,
            member=assemblies.index[peter],
            sha256=tarball.sha256,
            bytes=tarball.bytes,
            media_type=tarball.media_type,
        ),
        engineered_background=background,
    )


# --------------------------------------------------------------------------- #
# Phenotype
# --------------------------------------------------------------------------- #
def feature_universe_sha256(transcripts: Sequence[str]) -> str:
    """sha256 of the sorted feature ids joined by newlines."""
    return hashlib.sha256("\n".join(sorted(transcripts)).encode()).hexdigest()


def counts_phenotype(
    counts: Any, rows: Sequence[int], release: PanelRelease, universe_sha256: str
) -> SingleCellExpressionPhenotype:
    """Sum the CSR ``rows`` of a panel's counts into one phenotype (one row = one cell)."""
    summed = np.asarray(counts[list(rows)].sum(axis=0)).ravel()
    nonzero = np.flatnonzero(summed)
    return SingleCellExpressionPhenotype(
        umi_count={release.transcripts[j]: int(summed[j]) for j in nonzero},
        marker_umi_count=None,
        total_umis=sum(release.n_umi[r] for r in rows),
        n_features=len(release.transcripts),
        feature_universe_sha256=universe_sha256,
        n_cells=len(rows),
        cell_cycle_stage_counts=dict(Counter(release.stages[r] for r in rows)),
        library_cell_counts=dict(Counter(release.libraries[r] for r in rows)),
        cell_barcodes=[release.barcodes[r] for r in rows],
        cell_libraries=[release.libraries[r] for r in rows],
        cell_total_umis=[release.n_umi[r] for r in rows],
        provenance_gaps=[MARKER_GAP],
    )


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@register_dataset
class Boocock2025Dataset(ExperimentDataset):
    """Boocock 2025: 393 pooled segregants plus 79,123 de novo segregant cells."""

    organism = Organism.scerevisiae

    def __init__(
        self,
        root: str = "data/torchcell/boocock2025",
        io_workers: int = 0,
        genome: SCerevisiaeGenome | None = None,
        transform: Any | None = None,
        pre_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """A genome is REQUIRED (the gene set is the genes the markers span)."""
        self.genome = genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return SegregantExpressionExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return SegregantExpressionExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The four ``segData.RDS``, the pooled panel's match table and lookup
        genotypes, and Supplementary file 1.
        """
        return list(raw_relpaths())

    def download(self) -> None:
        """Symlink the manifest-listed mirror files into ``raw/`` and verify each sha256."""
        data_root = _data_root()
        manifest = load_manifest(data_root)
        rel = raw_relpaths()
        os.makedirs(self.raw_dir, exist_ok=True)
        for name in self.raw_file_names:
            src = raw_mirror_dir(data_root) / rel[name]
            if not src.exists():
                raise RuntimeError(f"required raw artifact missing from mirror: {src}")
            link_verified(
                src, osp.join(self.raw_dir, name), manifest_sha256(manifest, rel[name])
            )

    # ---- environment -------------------------------------------------------------
    @staticmethod
    def environment(spec: PanelSpec) -> Environment:
        """Minimal YNB for the pooled panel, the sheath-fluid medium for the de novo ones."""
        return Environment(
            media=PILOT_MEDIUM if spec.pooled else YNB_SHEATH_FLUID_BOOCOCK2025,
            temperature=Temperature(value=GROWTH_TEMPERATURE_C),
            aerobicity="aerobic",
        )

    def _reference(
        self, spec: PanelSpec, release: PanelRelease, universe: str
    ) -> SegregantExpressionExperimentReference:
        """The panel pseudobulk: every cell of the panel summed."""
        return SegregantExpressionExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae",
                strain="S288C",
                ploidy="haploid",
                assembly_set=SGD_S288C_R64,
                assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
            ),
            environment_reference=self.environment(spec),
            phenotype_reference=counts_phenotype(
                release.counts, range(len(release.barcodes)), release, universe
            ),
        )

    # ---- records -----------------------------------------------------------------
    def _panel_records(
        self,
        spec: PanelSpec,
        release: PanelRelease,
        parents: tuple[SegregantParent, SegregantParent],
        matrix_sha: str,
    ) -> Iterator[SegregantExpressionExperiment]:
        """One record per cell of a de novo panel."""
        universe = feature_universe_sha256(release.transcripts)
        environment = self.environment(spec)
        for row, (library, barcode) in enumerate(
            zip(release.libraries, release.barcodes)
        ):
            genotype = SegregantGenotype(
                cross=spec.cross,
                segregant_id=f"{library}:{barcode}",
                parent_1=parents[0],
                parent_2=parents[1],
                blocks=encode_posterior_blocks(release.posterior[row], release.markers),
                call_method=DE_NOVO_CALL_METHOD,
                marker_matrix_sha256=matrix_sha,
            )
            yield SegregantExpressionExperiment(
                dataset_name=self.name,
                genotype=genotype,
                environment=environment,
                phenotype=counts_phenotype(release.counts, [row], release, universe),
            )

    def _pooled_records(
        self,
        spec: PanelSpec,
        release: PanelRelease,
        match: SegregantMatch,
        lookup: tuple[list[str], list[Marker], Any],
        parents: tuple[SegregantParent, SegregantParent],
        lookup_sha: str,
    ) -> Iterator[SegregantExpressionExperiment]:
        """One record per matched segregant: its cells summed, its WGS genotype."""
        universe = feature_universe_sha256(release.transcripts)
        environment = self.environment(spec)
        ids, markers, calls = lookup
        row_of = {sid: i for i, sid in enumerate(ids)}
        cells_of: dict[str, list[int]] = {}
        for row, segregant in enumerate(match.segregant):
            cells_of.setdefault(segregant, []).append(row)
        for segregant in sorted(cells_of):
            coded = np.where(calls[row_of[segregant]] > 0, 2, 1).astype(np.int8)
            blocks = encode_blocks(coded, markers)
            if not np.array_equal(expand_blocks(blocks, markers), coded):
                raise ValueError(f"{segregant}: block round trip failed")
            genotype = SegregantGenotype(
                cross=spec.cross,
                segregant_id=segregant,
                parent_1=parents[0],
                parent_2=parents[1],
                blocks=blocks,
                call_method=POOLED_CALL_METHOD,
                marker_matrix_sha256=lookup_sha,
            )
            yield SegregantExpressionExperiment(
                dataset_name=self.name,
                genotype=genotype,
                environment=environment,
                phenotype=counts_phenotype(
                    release.counts, cells_of[segregant], release, universe
                ),
            )

    @post_process
    def process(self) -> None:
        """Build every panel's records, one panel at a time."""
        data_root = _data_root()
        manifest = load_manifest(data_root)
        rel = raw_relpaths()
        pins = {
            name: manifest_sha256(manifest, rel[name]) for name in self.raw_file_names
        }
        verify_raw_files(self.raw_dir, pins)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        backgrounds = read_parent_backgrounds(osp.join(self.raw_dir, SI1_NAME))
        assemblies = ParentAssemblyRefs.from_tier(data_root)
        publication = Publication(doi=PAPER_DOI, doi_url=f"https://doi.org/{PAPER_DOI}")

        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        index_rows: list[dict[str, Any]] = []
        block_counts: dict[str, list[int]] = {}
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for spec in PANELS.values():
                release = read_panel_release(osp.join(self.raw_dir, spec.segdata_name))
                self._check_release(spec, release)
                parents = (
                    build_parent(
                        spec.parent_1,
                        backgrounds[spec.strain][spec.parent_1],
                        assemblies,
                    ),
                    build_parent(
                        spec.parent_2,
                        backgrounds[spec.strain][spec.parent_2],
                        assemblies,
                    ),
                )
                reference = self._reference(
                    spec, release, feature_universe_sha256(release.transcripts)
                )
                if spec.pooled:
                    match = read_segregant_match(osp.join(self.raw_dir, SEGMATCH_NAME))
                    if len(match.segregant) != len(release.barcodes):
                        raise ValueError("segMatch.RDS and segData.RDS differ in cells")
                    records: Iterator[SegregantExpressionExperiment] = (
                        self._pooled_records(
                            spec,
                            release,
                            match,
                            read_lookup_genotypes(osp.join(self.raw_dir, GDATA_NAME)),
                            parents,
                            pins[GDATA_NAME],
                        )
                    )
                else:
                    records = self._panel_records(
                        spec, release, parents, pins[spec.segdata_name]
                    )
                counts: list[int] = []
                for experiment in records:
                    txn.put(
                        f"{idx}".encode(),
                        self._intern_record(experiment, reference, publication, itxn),
                    )
                    counts.append(experiment.genotype.n_blocks)
                    index_rows.append(
                        {
                            "panel": spec.panel,
                            "segregant_id": experiment.genotype.segregant_id,
                        }
                    )
                    idx += 1
                block_counts[spec.panel] = counts
                log.info("panel %s: %d records", spec.panel, len(counts))
        env.close()
        interned_env.close()
        if idx != EXPECTED_RECORDS:
            raise ValueError(f"wrote {idx} records, expected {EXPECTED_RECORDS}")
        pd.DataFrame(index_rows).to_csv(
            osp.join(self.preprocess_dir, "data.csv"), index=False
        )
        pd.DataFrame(
            [
                {"panel": panel, "n_blocks": n}
                for panel, values in block_counts.items()
                for n in values
            ]
        ).to_csv(osp.join(self.preprocess_dir, "block_counts.csv"), index=False)

    @staticmethod
    def _check_release(spec: PanelSpec, release: PanelRelease) -> None:
        """Cell, transcript and library counts are Table S4's and the release's sets."""
        if len(release.barcodes) != spec.expected_cells:
            raise ValueError(
                f"{spec.panel}: {len(release.barcodes)} cells, expected {spec.expected_cells}"
            )
        if len(release.transcripts) != spec.expected_transcripts:
            raise ValueError(
                f"{spec.panel}: {len(release.transcripts)} transcripts, expected "
                f"{spec.expected_transcripts}"
            )
        if set(release.libraries) != set(spec.libraries):
            raise ValueError(
                f"{spec.panel}: libraries {sorted(set(release.libraries))} != "
                f"{sorted(spec.libraries)}"
            )

    # ---- gene set ----------------------------------------------------------------
    @staticmethod
    def extract_systematic_gene_names(genotype: dict[str, Any]) -> list[str]:
        """Not applicable: a mosaic genotype has no gene-keyed perturbations."""
        raise NotImplementedError(
            "a SegregantGenotype carries no gene perturbations; "
            "Boocock2025Dataset.compute_gene_set returns the S288C genes its markers span"
        )

    def compute_gene_set(self) -> GeneSet:
        """S288C genes overlapping the marker span of any panel, per chromosome."""
        if self.genome is None:
            raise RuntimeError("Boocock2025Dataset requires a genome for its gene set")
        spans = self.marker_spans()
        gene_set = GeneSet()
        db = self.genome.db
        for gene_id in self.genome.gene_set:
            feature = db[gene_id]
            for start, end in spans.get(str(feature.chrom), []):
                if feature.start <= end and feature.end >= start:
                    gene_set.add(gene_id)
                    break
        if not gene_set:
            raise ValueError("no S288C gene overlaps any panel's marker span")
        return gene_set

    def marker_spans(self) -> dict[str, list[tuple[int, int]]]:
        """Per chromosome, the (first, last) marker position of every panel's markers."""
        import rdata

        spans: dict[str, list[tuple[int, int]]] = {}
        marker_sets: list[list[Marker]] = []
        for spec in PANELS.values():
            gsub = rdata.read_rds(osp.join(self.raw_dir, spec.segdata_name))["Gsub"]
            names = [str(c) for c in gsub.coords["dim_1"].values]
            marker_sets.append([parse_marker(n, i) for i, n in enumerate(names)])
        marker_sets.append(read_lookup_genotypes(osp.join(self.raw_dir, GDATA_NAME))[1])
        for markers in marker_sets:
            for chrom in _CHROMOSOMES:
                positions = [m.position for m in markers if m.chromosome == chrom]
                if positions:
                    spans.setdefault(chrom, []).append((min(positions), max(positions)))
        return spans

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing happens inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Records are assembled inside process()."""
        raise NotImplementedError("Boocock2025Dataset builds its records in process()")


def sourced_values() -> list[SourcedValue]:
    """Every quote this dataset's records rest on: the module's and its two media's."""
    media = [PILOT_MEDIUM, YNB_SHEATH_FLUID_BOOCOCK2025]
    return [
        *SOURCED.values(),
        *(sv for m in media for sv in m.provenance),
        *(sv for m in media for c in m.components for sv in c.provenance),
    ]


def verify_build(dataset_root: str, *, data_root: str | None = None) -> Any:
    """L0-L4 report for a built store (``torchcell.verification.segregant_expression``)."""
    import json

    from torchcell.sequence.genome.registry import PETER2018_1011, resolve
    from torchcell.verification import runners
    from torchcell.verification.segregant_expression import (
        verify_segregant_expression_streaming,
    )

    root = data_root or _data_root()
    spec = runners.SEGREGANT_EXPRESSION_DATASETS["boocock2025"]
    genome = SCerevisiaeGenome(
        genome_root=osp.join(root, "data/sgd/genome"),
        go_root=osp.join(root, "data/go"),
        overwrite=False,
    )
    index_path = resolve(PETER2018_1011, spec["assembly_index"], data_root=root)
    assembly_index = dict(
        line.split("\t") for line in Path(index_path).read_text().splitlines()
    )
    with open(osp.join(dataset_root, "preprocess", "gene_set.json")) as handle:
        gene_set = set(json.load(handle))
    return verify_segregant_expression_streaming(
        runners.stream_records(dataset_root),
        dataset_name="boocock2025",
        provenance=spec["provenance"],
        expected_count=spec["expected_count"],
        raw_dir=osp.join(dataset_root, "raw"),
        library_root=osp.join(root, spec["library"]),
        assembly_index=assembly_index,
        genome=genome,
        sgd_genes=runners._sgd_gene_set(root),
        gene_set=gene_set,
        sourced=sourced_values(),
    )


def main() -> None:
    """Build/load the dataset for interactive debugging."""
    data_root = _data_root()
    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    dataset = Boocock2025Dataset(
        root=osp.join(data_root, "data/torchcell/boocock2025"), genome=genome
    )
    print(f"len = {len(dataset)}")


if __name__ == "__main__":
    main()
