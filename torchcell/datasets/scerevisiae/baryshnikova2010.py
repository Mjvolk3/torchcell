# torchcell/datasets/scerevisiae/baryshnikova2010
# [[torchcell.datasets.scerevisiae.baryshnikova2010]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/baryshnikova2010
# Test file: tests/torchcell/datasets/scerevisiae/test_baryshnikova2010.py
"""Baryshnikova 2010 genome-scale single-mutant fitness (SMF) dataset.

Baryshnikova A et al. 2010 (Boone/Andrews lab), "Quantitative analysis of fitness and
genetic interactions in yeast on a genome scale", Nat Methods 7(12):1017-1024
(doi:10.1038/nmeth.1534). This is the SGA-scoring methods paper that Costanzo 2016 and
Kuzmin 2018 DEFER to for the fitness/interaction formulas, but it ALSO releases a NEW
genome-scale single-mutant fitness catalog (Supplementary Data 1) -- the data this loader
builds.

WHAT THE FILE CONTAINS (Supplementary Data 1, sheet ``S1_SMF_standard_100209``; 6023 rows,
3 columns, no header):
  - col0: SGA strain/allele id -- a plain systematic ORF for a nonessential deletion
    (e.g. YML062C), an essential-gene DAmP allele (``<ORF>_damp``, e.g. YAL001C_damp), or a
    temperature-sensitive allele (``<ORF>_tsq<NNN>``, e.g. YBR156C_tsq236).
  - col1: single-mutant fitness, WT-normalized so the fitness-distribution MODE == 1.0
    (``_SV_REFERENCE_FITNESS``). Range 0.063 .. 1.161.
  - col2: uncertainty = a BOOTSTRAP standard error of the median fitness estimate
    (``_SV_UNCERTAINTY_TYPE``), NOT a sample SD. Range 0.0006 .. 0.1141.

COMPOSITION (6023 = 4635 deletions + 1082 DAmP + 306 TS; ``_SV_COMPOSITION``):
  - 4635 nonessential gene deletions (kanMX-marked ARRAY) -> SgaKanMxDeletionPerturbation.
  - 1082 essential-gene DAmP hypomorphs (``_damp``) -> SgaDampPerturbation.
  - 306 temperature-sensitive essential alleles (``_tsq``) -> SgaTsAllelePerturbation.
  The full raw id is preserved on ``strain_id`` so an allelic series is distinct: 58 genes
  carry >1 TS allele (e.g. YAL041W x4), which the fitness L1 signature separates via
  strain_id (see torchcell/verification/fitness.py ``_genotype_signature``).

ARRAY SIDE vs QUERY SIDE, and why it decides ``n_samples``. SI Supplementary Table 2
  describes TWO measurement designs. The ARRAY side crosses a neutral natMX query against
  the nonessential deletion collection, and only THAT side carries a screen count
  ("Fitness measurements were based on 80 control screens.", ``_SV_N_SAMPLES_ARRAY``). The
  QUERY side arrays the natMX-marked query strains against a single kanMX control array,
  and the SI states no screen count for it. The released ids say which side a row is on:
  ``_tsq`` and ``_DAmP`` are the SGA QUERY strain labels (``_SV_TSQ_IS_QUERY``,
  ``_SV_DAMP_IS_QUERY``, both from the Costanzo 2016 SI, which uses the same lab
  nomenclature and lists both collections). So ``n_samples = 80`` is stored for the 4635
  deletion rows and the 1388 query-side rows carry a typed
  ``ProvenanceGap(field="n_samples")`` instead of the extrapolated 80. The main text's
  looser sentence (``_SV_COMPOSITION``) describes all 6023 as measured "using a neutral
  (control) query mutation", which is the array-side phrasing; the SI's two-sided table
  and the strain labels are the precise statement.

ENVIRONMENT: the SGA final-selection plate, SD/MSG -His/Arg/Lys +canavanine/thialysine/
  G418/clonNAT (``SGA_DM_SELECTION``, the shared library medium whose recipe is quoted from
  the same Tong and Boone 2006 protocol), solid, at the final-selection temperature. The
  temperature is SPLIT by allele kind and neither half is this paper's own number: the
  Online Methods defer ("as described previously1,7") and state no incubation temperature.
  - deletion + DAmP -> 30 C (``_SV_TEMPERATURE_30``, Tong and Boone 2006 step 18, the
    final kanR/natR selection incubation on exactly this medium; corroborated by
    ``_SV_DMA_30``).
  - TS (``_tsq``) -> 26 C (``_SV_TEMPERATURE_26``: every SGA selection step involving a TS
    allele runs at a permissive 22 C except the final haploid double-mutant selection at a
    semipermissive 26 C, and ``_SV_TS_26_EITHER_SIDE`` extends that to a TS allele used as
    a query OR as an array mutant).
  BACK-SOLVE (measured, not assumed): 291 of the 306 TS rows share their exact strain id
  with a Costanzo 2016 SMF row, and Costanzo releases that SMF at BOTH 26 C and 30 C. The
  Baryshnikova value tracks the 26 C column (Pearson r 0.8777, mean |difference| 0.0577)
  far better than the 30 C column (r 0.7105, mean |difference| 0.1401). In the same
  Costanzo release the 26 C and 30 C columns are EXACTLY equal for every deletion, _sn and
  _damp strain (3876/3876, 3845/3845, 733/733) and never equal for a TS strain (0/792
  _tsa, 0/961 _tsq) -- only TS alleles are temperature-resolved at all, which is what makes
  30 C the right value for the non-TS rows and 26 C the right value for the TS rows.

GENOTYPE / REFERENCE: each strain is one gene perturbation in the BY4741 SGA background
  (``_SV_BACKGROUND_STRAIN``; note the quote is the SI's border-control sentence, the only
  place the mirror names the array background, and the main text's BY4741 sentence is
  scoped to the serial-dilution/actin/RIM101 follow-up strains). The release carries only
  systematic allele ids (no common names), so ``perturbed_gene_name`` is the resolved
  systematic ORF and ``strain_id`` is the cross-dataset strain key; the allele names of the
  291 TS strains shared with Costanzo 2016 are NOT imported, because the Costanzo strain
  table is not in the raw mirror and a derivation cannot be pinned to it. The reference
  fitness 1.0 is the mode-normalization convention, not an averaged control measurement, so
  every reference phenotype gaps ``n_samples`` and ``sample_unit``.

DATA SOURCE: the publisher's Supplementary Data 1 (``SupplementaryData1_SMF.xls``, sha256
  086bfadf..., Springer ESM MediaObject MOESM168), deposited in
  ``$DATA_ROOT/torchcell-raw/baryshnikovaQuantitativeAnalysisFitness2010/`` with a
  ``manifest.json`` recording the URL, the retriever and the hash. It supersedes the
  ``data/S1_SMF_standard_100209.txt`` text export this loader used to read, which had no
  recorded retrieval. MEASURED: sorted on (id, fitness, se) the two files are the same data
  -- every fitness value is bit-identical and the SEs agree to 1.7e-18 -- and they differ
  only in ROW ORDER. The previous docstring's claim that they "differ by up to 0.031" was
  an artifact of joining on the duplicated YDL227C id, whose two rows (1.000155 and
  1.031582) cross-join.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import os.path as osp
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd
from tqdm import tqdm

from torchcell.data import ExperimentDataset, post_process
from torchcell.datamodels.media import SGA_DM_SELECTION
from torchcell.datamodels.schema import (
    Environment,
    Experiment,
    ExperimentReference,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    Genotype,
    Publication,
    ReferenceGenome,
    SampleUnit,
    SgaDampPerturbation,
    SgaKanMxDeletionPerturbation,
    SgaTsAllelePerturbation,
    Temperature,
    UncertaintyType,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DOI = "10.1038/nmeth.1534"

# --------------------------------------------------------------------------- #
# Provenance anchors: the three mirrored artifacts every constant below is pinned to.
# --------------------------------------------------------------------------- #
CITATION_KEY = "baryshnikovaQuantitativeAnalysisFitness2010"
_PAPER_MD = "paper.md"
_PAPER_MD_SHA = "7ebd3880c052d1808c055f2ba2445e016f79080f222591bb023b755806ccd3d0"
_SI_MD = "si/si1.md"
_SI_MD_SHA = "f85ea57ce9f5ec60b633b837ff2c5ee18cc65e113db2f034b892739690d8046e"

# Deferral targets. Baryshnikova 2010's Online Methods say the screens were run "as
# described previously1,7"; ref 1 (Methods Enzymol 470) and ref 7 (Costanzo 2010 Science)
# are NOT mirrored. These two ARE, they are the same lab describing the same SGA system,
# and between them they fix the final-selection temperature and the strain nomenclature.
_TONG2006 = "yantongSyntheticGeneticArray2006"
_TONG2006_TXT = "paper.txt"
_TONG2006_SHA = "dda5fc727c5e532e02884cd1d30ad0774bfb773ed55e8f6b7074ee2158ab9aca"
_COSTANZO2016 = "costanzoGlobalGeneticInteraction2016"
_COSTANZO2016_SI = "si/si1.md"
_COSTANZO2016_SI_SHA = (
    "1828703b0ff739fdf1c0d9232fe4fd81a3ce95a1b111780f55ef63bfa676880e"
)


def _sv(
    value: object,
    quote: str,
    *,
    ck: str = CITATION_KEY,
    uri: str = _SI_MD,
    sha: str = _SI_MD_SHA,
    page: str | None = None,
    note: str | None = None,
) -> SourcedValue:
    """A SourcedValue pinned to a mirrored, sha256-verified artifact (quote + hash)."""
    return SourcedValue(
        value=value,
        provenance=Provenance(source_uri=uri, citation_key=ck, sha256=sha, page=page),
        quote=quote,
        note=note,
    )


#: Replicate count for the ARRAY-side measurement (the nonessential deletion collection).
#: The sentence is inside the SI's "Single mutant fitness" row and attaches to the array
#: measurement described in the two sentences before it; the query-side description that
#: follows carries no count.
_SV_N_SAMPLES_ARRAY = _sv(
    80,
    "Thus, colony sizes derived from these control screens (after applying standard "
    "normalization procedures) reflect array strain single mutant fitness. Fitness "
    "measurements were based on 80 control screens.",
    page="Supplementary Table 2, 'Single mutant fitness'",
    note="scopes to the nonessential deletion ARRAY; the query-side paragraph that "
    "follows states no screen count, so the 1388 _damp and _tsq rows gap n_samples",
)

#: The query-side design, stated without a screen count. Anchors the gap's reason.
_SV_QUERY_SIDE_NO_COUNT = _sv(
    None,
    "Query strain single mutant fitness was obtained in a similar manner.",
    page="Supplementary Table 2, 'Single mutant fitness'",
    note="the query-side measurement is described but its number of control screens is "
    "never released",
)

#: What the released uncertainty column IS.
_SV_UNCERTAINTY_TYPE = _sv(
    UncertaintyType.bootstrap_se.value,
    "Variance in single mutant fitness estimates was estimated from bootstrap sampling "
    "of the median, and final fitness estimates were derived by pooling across each "
    "array spatial configuration.",
    page="Supplementary Note 1",
    note="a bootstrap SD of the estimator is already an SE, so it is used as fitness_se "
    "without dividing by sqrt(n)",
)

#: The WT-normalization convention that fixes the reference fitness at 1.0.
_SV_REFERENCE_FITNESS = _sv(
    1.0,
    "The parameter $\\alpha$ was estimated by setting the mode of the fitness "
    "distribution to 1, reflecting our assumption that most deletions should have near "
    "wild-type fitness.",
    page="Supplementary Note 1",
    note="the reference is a normalization convention, not an averaged WT measurement",
)

#: Genetic background of the arrayed collection.
_SV_BACKGROUND_STRAIN = _sv(
    "BY4741",
    "isogenic to BY4741",
    page="Supplementary Table 2, 'Plate border control'",
    note="the only place the mirror names the array background: every array plate's "
    "border-control strain is isogenic to BY4741 (his3 replaced with KanMX). The main "
    "text's BY4741 sentence (paper.md, Online Methods 'Strains') is scoped to the "
    "serial-dilution, actin-staining and RIM101 follow-up strains, so it is NOT the "
    "citation for the collection background",
)

#: Released composition, from the main text.
_SV_COMPOSITION = _sv(
    {"deletion": 4635, "damp": 1082, "ts": 306},
    "We measured colony sizes for 4,635 viable deletion mutants and 1,388 "
    "temperature-sensitive or hypomorphic alleles of essential genes after SGA analysis "
    "using a neutral (control) query mutation",
    uri=_PAPER_MD,
    sha=_PAPER_MD_SHA,
    page="Results, 'Quantitative analysis of single-mutant fitness'",
    note="1388 = 1082 DAmP + 306 TS, split here by the released id suffix",
)

#: The Online-Methods deferral that makes the temperature a cross-paper chain.
_SV_METHODS_DEFERRAL = _sv(
    None,
    "as described previously1,7",
    uri=_PAPER_MD,
    sha=_PAPER_MD_SHA,
    page="Online Methods, 'SGA screens'",
    note="ref 1 = Baryshnikova et al. Methods Enzymol. 470:146-180 (2010), ref 7 = "
    "Costanzo et al. Science 327:425-431 (2010); NEITHER is mirrored, so the "
    "temperature is sourced from the two mirrored papers of the same lab describing "
    "the same SGA system",
)

#: Final SGA selection incubation on exactly the SGA_DM_SELECTION plate.
_SV_TEMPERATURE_30 = _sv(
    30.0,
    "18. Incubate the kanR/natR-selection plates at 30°C for 2 d.",
    ck=_TONG2006,
    uri=_TONG2006_TXT,
    sha=_TONG2006_SHA,
    page="SGA screen protocol, MATa-kanR-natR meiotic progeny selection",
    note="step 17 pins onto the (SD/MSG) -His/Arg/Lys +canavanine/thialysine/G418/"
    "clonNAT plate, i.e. SGA_DM_SELECTION, and step 18 is its incubation",
)

#: Corroboration of 30 C for the nonessential deletion array.
_SV_DMA_30 = _sv(
    30.0,
    "Double mutant selection plates involving a nonessential deletion mutant query "
    "strain and the DMA were incubated at",
    ck=_COSTANZO2016,
    uri=_COSTANZO2016_SI,
    sha=_COSTANZO2016_SI_SHA,
    page="SGA query strain construction and screening",
    note="the sentence ends with the OCR'd LaTeX rendering of 30 C, so the quote stops "
    "before it; the value is the 30 C that follows",
)

#: The TS exception: a semipermissive 26 C for the final selection.
_SV_TEMPERATURE_26 = _sv(
    26.0,
    "All SGA selection steps involving a TS allele were conducted at permissive "
    "temperature $( 2 2 ^ { \\circ } \\mathsf { C } )$ except for the final selection of "
    "haploid double mutants, which were incubated at a semipermissive temperature "
    "$( 2 6 ^ { \\circ } \\mathsf { C } )$ prior to imaging.",
    ck=_COSTANZO2016,
    uri=_COSTANZO2016_SI,
    sha=_COSTANZO2016_SI_SHA,
    page="SGA query strain construction and screening",
)

#: ...and it holds whether the TS allele is the query or the array mutant.
_SV_TS_26_EITHER_SIDE = _sv(
    26.0,
    "higher quality interactions from screens involving a TS allele (either as a query "
    "or an array mutant) were obtained when plates were grown at",
    ck=_COSTANZO2016,
    uri=_COSTANZO2016_SI,
    sha=_COSTANZO2016_SI_SHA,
    page="SGA query strain construction and screening",
    note="Baryshnikova's TS rows are query strains (see _SV_TSQ_IS_QUERY), and this "
    "sentence covers that case explicitly",
)

#: What a ``_tsq`` id IS: an SGA QUERY strain, crossed to a neutral kanMX control array.
_SV_TSQ_IS_QUERY = _sv(
    "SGA TS-allele query strain",
    "ordered arrays of SGA query mutant strains carrying natMX-marked, nonessential "
    "deletion mutations (_sn#) or TS alleles of essential genes (_tsq#) were crossed to "
    "a different SGA control strain, which carried a kanMX marker inserted at a neutral "
    "genomic locus.",
    ck=_COSTANZO2016,
    uri=_COSTANZO2016_SI,
    sha=_COSTANZO2016_SI_SHA,
    page="Estimating single mutant fitness",
    note="the same lab nomenclature Baryshnikova's ids use, so a _tsq row is a "
    "query-side measurement and the 80-control-screen count does not apply to it",
)

#: ...and a DAmP allele is likewise a query mutant.
_SV_DAMP_IS_QUERY = _sv(
    "SGA DAmP-allele query strain",
    "a set of query mutants carrying hypomorphic, Decreased Abundance by mRNA "
    "Perturbation (labeled _DAmP#) alleles of essential genes, which are potential "
    "hypomorphic mutants, was screened against the DMA and TSA",
    ck=_COSTANZO2016,
    uri=_COSTANZO2016_SI,
    sha=_COSTANZO2016_SI_SHA,
    page="DAmP query mutant SGA screens",
)

#: Every SourcedValue above, for the test that re-audits them against the mirror.
SOURCED_VALUES: tuple[SourcedValue, ...] = (
    _SV_N_SAMPLES_ARRAY,
    _SV_QUERY_SIDE_NO_COUNT,
    _SV_UNCERTAINTY_TYPE,
    _SV_REFERENCE_FITNESS,
    _SV_BACKGROUND_STRAIN,
    _SV_COMPOSITION,
    _SV_METHODS_DEFERRAL,
    _SV_TEMPERATURE_30,
    _SV_DMA_30,
    _SV_TEMPERATURE_26,
    _SV_TS_26_EITHER_SIDE,
    _SV_TSQ_IS_QUERY,
    _SV_DAMP_IS_QUERY,
)

# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
XLS_NAME = "SupplementaryData1_SMF.xls"
XLS_REL = f"data/{XLS_NAME}"
XLS_SHEET = "S1_SMF_standard_100209"
XLS_SHA256 = "086bfadf2684f28940500dd87e3be74c53a957448d2016f7a02370540da8a04e"
XLS_URL = (
    "https://static-content.springer.com/esm/art%3A10.1038%2Fnmeth.1534/"
    "MediaObjects/41592_2010_BFnmeth1534_MOESM168_ESM.xls"
)

# --------------------------------------------------------------------------- #
# Release constants, checked against the file on every build
# --------------------------------------------------------------------------- #
_EXPECTED_ROWS = 6023

#: Raw allele ids whose ORF the CURRENT R64 annotation cannot resolve (retired, merged or
#: dubious ORFs removed since 2010), frozen so the drop is a documented release fact and
#: the record-count oracle is ``_EXPECTED_ROWS - len(_UNRESOLVABLE)`` rather than whatever
#: the genome on disk happens to yield. A build whose drop set differs from this raises.
_UNRESOLVABLE: MappingProxyType[str, str] = MappingProxyType(
    {
        "YAR037W": "deletion",
        "YAR040C": "deletion",
        "YAR043C": "deletion",
        "YAR061W": "deletion",
        "YAR062W": "deletion",
        "YCL006C": "deletion",
        "YCL013W": "deletion",
        "YCL026C": "deletion",
        "YCL074W": "deletion",
        "YCL075W": "deletion",
        "YDR134C": "deletion",
        "YER108C": "deletion",
        "YER109C": "deletion",
        "YFL056C": "deletion",
        "YFL057C": "deletion",
        "YIL080W": "deletion",
        "YIL167W": "deletion",
        "YIL168W": "deletion",
        "YIL170W": "deletion",
        "YIL171W_damp": "damp",
        "YIL174W": "deletion",
        "YIL175W": "deletion",
        "YIR043C": "deletion",
        "YLL016W": "deletion",
        "YLL017W": "deletion",
        "YOL153C": "deletion",
        "YOR031W": "deletion",
        "YPL060C-A": "deletion",
        "YPL275W": "deletion",
        "YPL276W": "deletion",
    }
)
DROP_RULE = (
    "a released allele whose ORF token does not resolve to a gene of the current R64 "
    "annotation (retired, merged or dubious ORF) is dropped; the set is frozen in "
    "_UNRESOLVABLE and asserted on every build"
)
EXPECTED_RECORDS = _EXPECTED_ROWS - len(_UNRESOLVABLE)

#: 30 C carries the deletions and the DAmP hypomorphs; 26 C carries the TS alleles.
_TEMPERATURE_C: MappingProxyType[str, float] = MappingProxyType(
    {
        "deletion": float(_SV_TEMPERATURE_30.value),
        "damp": float(_SV_TEMPERATURE_30.value),
        "ts": float(_SV_TEMPERATURE_26.value),
    }
)

#: The array side is the only side with a released screen count.
_ARRAY_SIDE_KIND = "deletion"

_N_SAMPLES_GAP = ProvenanceGap(
    field="n_samples",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=Provenance(
        source_uri=_SI_MD,
        citation_key=CITATION_KEY,
        sha256=_SI_MD_SHA,
        page="Supplementary Table 2, 'Single mutant fitness'",
    ),
    note="SI Supplementary Table 2 states 80 control screens for the ARRAY-side "
    "(nonessential deletion collection) measurement only; this row is a query strain "
    "(_tsq / _damp) and the query-side control-screen count is not released",
)
_REFERENCE_GAP_NOTE = (
    "the reference fitness 1.0 is the mode-normalization convention (the fitness "
    "distribution's mode is set to 1), not an averaged wild-type measurement, so the "
    "primary reports no replicate count or replicate unit for it"
)
_REFERENCE_GAPS = [
    ProvenanceGap(
        field=field,
        reason=ProvenanceGapReason.not_reported_by_primary,
        looked_in=Provenance(
            source_uri=_SI_MD,
            citation_key=CITATION_KEY,
            sha256=_SI_MD_SHA,
            page="Supplementary Note 1",
        ),
        note=_REFERENCE_GAP_NOTE,
    )
    for field in ("n_samples", "sample_unit")
]


def _parse_allele(allele: str) -> tuple[str, str]:
    """Split a raw SGA allele id into (systematic-ORF token, perturbation kind)."""
    if allele.endswith("_damp"):
        return allele[: -len("_damp")], "damp"
    if "_tsq" in allele:
        return allele.split("_tsq")[0], "ts"
    return allele, "deletion"


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
    """``$DATA_ROOT/torchcell-raw/baryshnikovaQuantitativeAnalysisFitness2010``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def deposit_raw_mirror(
    *, xls_path: str | Path, retrieved_at: str, data_root: str | None = None
) -> Path:
    """Write the raw mirror from an already-retrieved Supplementary Data 1 and its manifest.

    Run once by hand after retrieving ``XLS_URL``; idempotent by sha256 (an existing file
    with the recorded hash is left alone, a differing one raises). The Springer ESM CDN is
    directly scriptable, so the recorded retriever re-downloads these exact bytes.
    """
    root = raw_mirror_dir(data_root)
    digest = _sha256(xls_path)
    if digest != XLS_SHA256:
        raise RuntimeError(f"{xls_path} sha256 {digest} != expected {XLS_SHA256}")
    dest = root / XLS_REL
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if _sha256(dest) != XLS_SHA256:
            raise RuntimeError(f"{dest} exists with a different sha256; refusing")
    else:
        shutil.copy2(xls_path, dest)
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=(
            "Quantitative analysis of fitness and genetic interactions in yeast on a "
            "genome scale"
        ),
        files=[
            ArtifactRecord(
                path=XLS_REL,
                role=ROLE_RAW_DATA,
                bytes=dest.stat().st_size,
                sha256=XLS_SHA256,
                source=XLS_URL,
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.springer_esm,
                    source_url=XLS_URL,
                    retriever="torchcell.literature.retrieve.springer_esm",
                    params={"url": XLS_URL},
                    sha256=XLS_SHA256,
                    retrieved_at=retrieved_at,
                ),
            )
        ],
        si_data_sources=[XLS_URL],
        si_expected=["Supplementary Data 1 (single-mutant fitness)"],
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
    for record in manifest.files:
        if record.path == relpath:
            return record.sha256
    raise KeyError(f"{relpath} is not in the raw-mirror manifest")


@register_dataset
class SmfBaryshnikova2010Dataset(ExperimentDataset):
    """Baryshnikova 2010 genome-scale single-mutant fitness dataset (6023 alleles)."""

    def __init__(
        self,
        root: str = "data/torchcell/smf_baryshnikova2010",
        io_workers: int = 0,
        genome: SCerevisiaeGenome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset; a genome is REQUIRED to resolve ORFs to current R64."""
        self.genome = genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return FitnessExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return FitnessExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The publisher's Supplementary Data 1 workbook."""
        return [XLS_NAME]

    def download(self) -> None:
        """Symlink the manifest-listed mirror file into ``raw/`` and verify its sha256."""
        data_root = _data_root()
        manifest = load_manifest(data_root)
        src = raw_mirror_dir(data_root) / XLS_REL
        if not src.exists():
            raise RuntimeError(f"required raw artifact missing from mirror: {src}")
        expected = manifest_sha256(manifest, XLS_REL)
        got = _sha256(src)
        if got != expected:
            raise RuntimeError(
                f"{XLS_NAME} sha256 mismatch: got {got}, expected {expected}"
            )
        os.makedirs(self.raw_dir, exist_ok=True)
        dest = osp.join(self.raw_dir, XLS_NAME)
        if not osp.exists(dest):
            os.symlink(src, dest)
        log.info(
            "Baryshnikova 2010 raw file linked into %s (sha256 %s)", self.raw_dir, got
        )

    def _resolver(self) -> Callable[[str], str | None]:
        """Build an ORF -> current-R64-systematic-name resolver from the genome."""
        if self.genome is None:
            raise RuntimeError(
                "SmfBaryshnikova2010Dataset requires a genome for ORF resolution; "
                "inject SCerevisiaeGenome(...)"
            )
        genome = self.genome
        df = genome.gene_attribute_table
        ids = set(df["ID"])
        alias_map = genome.alias_to_systematic

        def resolve(token: str) -> str | None:
            gene = token.upper()
            if gene in ids:
                return gene
            candidates = alias_map.get(gene, [])
            if candidates and candidates[0] in ids:
                return candidates[0]
            return None

        return resolve

    def _read_smf(self) -> pd.DataFrame:
        """Read + validate Supplementary Data 1 (3 columns, no header, composition check)."""
        path = osp.join(self.raw_dir, XLS_NAME)
        df = pd.read_excel(
            path, sheet_name=XLS_SHEET, header=None, names=["allele", "fitness", "se"]
        )
        if len(df) != _EXPECTED_ROWS:
            raise RuntimeError(
                f"SMF row-count self-checksum failed: {len(df)} != {_EXPECTED_ROWS}"
            )
        kinds = df["allele"].map(lambda a: _parse_allele(str(a))[1])
        comp = kinds.value_counts().to_dict()
        if comp != dict(_SV_COMPOSITION.value):
            raise RuntimeError(
                f"SMF composition self-checksum failed: {comp} != {_SV_COMPOSITION.value}"
            )
        return df

    @staticmethod
    def _perturbation(
        kind: str, orf: str, strain_id: str
    ) -> SgaKanMxDeletionPerturbation | SgaDampPerturbation | SgaTsAllelePerturbation:
        """Build the SGA perturbation for one allele, preserving the raw id on strain_id."""
        if kind == "deletion":
            return SgaKanMxDeletionPerturbation(
                systematic_gene_name=orf, perturbed_gene_name=orf, strain_id=strain_id
            )
        if kind == "damp":
            return SgaDampPerturbation(
                systematic_gene_name=orf, perturbed_gene_name=orf, strain_id=strain_id
            )
        return SgaTsAllelePerturbation(
            systematic_gene_name=orf, perturbed_gene_name=orf, strain_id=strain_id
        )

    @staticmethod
    def environment(kind: str) -> Environment:
        """The SGA final-selection plate for one allele kind (30 C, or 26 C for TS)."""
        return Environment(
            media=SGA_DM_SELECTION, temperature=Temperature(value=_TEMPERATURE_C[kind])
        )

    @staticmethod
    def _n_samples(kind: str) -> int | None:
        """80 released control screens on the array side; nothing released on the query side."""
        if kind == _ARRAY_SIDE_KIND:
            return int(_SV_N_SAMPLES_ARRAY.value)
        return None

    def _phenotype(self, kind: str, fitness: float, se: float) -> FitnessPhenotype:
        """The measured fitness, with the query-side replicate count declared as a gap."""
        n_samples = self._n_samples(kind)
        return FitnessPhenotype(
            fitness=fitness,
            fitness_uncertainty=se,
            fitness_uncertainty_type=UncertaintyType(_SV_UNCERTAINTY_TYPE.value),
            n_samples=n_samples,
            sample_unit=SampleUnit.screen,
            provenance_gaps=[] if n_samples is not None else [_N_SAMPLES_GAP],
        )

    def _reference(self, kind: str) -> FitnessExperimentReference:
        """The wild-type reference: fitness 1.0 by the mode-normalization convention."""
        return FitnessExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae",
                strain=str(_SV_BACKGROUND_STRAIN.value),
            ),
            environment_reference=self.environment(kind),
            phenotype_reference=FitnessPhenotype(
                fitness=float(_SV_REFERENCE_FITNESS.value),
                provenance_gaps=list(_REFERENCE_GAPS),
            ),
        )

    @post_process
    def process(self) -> None:
        """Convert each SMF row into a fitness record; write LMDB."""
        resolve = self._resolver()
        df = self._read_smf()

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        # A few raw allele ids repeat in the source with DIFFERENT fitness -- YDL227C (HO,
        # the neutral SGA marker locus) appears twice (1.000155 vs 1.031582). Both source
        # rows are kept as distinct records; a positional suffix on strain_id
        # disambiguates them so the L1 (strain, environment) signature stays unique (never
        # guess-merge two measurements).
        dup_ids = set(df["allele"][df["allele"].duplicated(keep=False)].astype(str))
        occ: dict[str, int] = {}

        environments = {kind: self.environment(kind) for kind in _TEMPERATURE_C}
        references = {kind: self._reference(kind) for kind in _TEMPERATURE_C}
        publication = Publication(doi=DOI, doi_url=f"https://doi.org/{DOI}")

        dropped: dict[str, str] = {}
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="baryshnikova2010"):
                raw_allele = str(row["allele"])
                orf_token, kind = _parse_allele(raw_allele)
                orf = resolve(orf_token)
                if orf is None:
                    dropped[raw_allele] = kind
                    continue
                if raw_allele in dup_ids:
                    occ[raw_allele] = occ.get(raw_allele, 0) + 1
                    strain_id = f"{raw_allele}.{occ[raw_allele]}"
                else:
                    strain_id = raw_allele
                experiment = FitnessExperiment(
                    dataset_name=self.name,
                    genotype=Genotype(
                        perturbations=[self._perturbation(kind, orf, strain_id)]
                    ),
                    environment=environments[kind],
                    phenotype=self._phenotype(
                        kind, float(row["fitness"]), float(row["se"])
                    ),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(
                        experiment, references[kind], publication, itxn
                    ),
                )
                idx += 1
        env.close()
        interned_env.close()

        if dropped != dict(_UNRESOLVABLE):
            raise RuntimeError(
                "unresolvable allele set drifted from the frozen release constant: "
                f"unexpected {sorted(set(dropped) - set(_UNRESOLVABLE))}, "
                f"missing {sorted(set(_UNRESOLVABLE) - set(dropped))}"
            )
        if idx != EXPECTED_RECORDS:
            raise RuntimeError(f"wrote {idx} records, expected {EXPECTED_RECORDS}")
        with open(osp.join(self.preprocess_dir, "dropped_records.json"), "w") as fh:
            json.dump(
                {
                    "rule": DROP_RULE,
                    "n_raw_rows": _EXPECTED_ROWS,
                    "n_dropped": len(dropped),
                    "n_kept": idx,
                    "dropped_by_kind": {
                        kind: sum(1 for v in dropped.values() if v == kind)
                        for kind in sorted(set(dropped.values()))
                    },
                    "dropped": dict(sorted(dropped.items())),
                },
                fh,
                indent=2,
            )
        log.info(
            "Wrote %d Baryshnikova2010 fitness experiments to LMDB (%d dropped: %s)",
            idx,
            len(dropped),
            sorted(dropped),
        )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


def main() -> None:
    """Build/load the dataset for interactive debugging."""
    data_root = _data_root()
    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    root = osp.join(data_root, "data/torchcell/smf_baryshnikova2010")
    dataset = SmfBaryshnikova2010Dataset(root=root, genome=genome)
    print(f"len = {len(dataset)}")
    print(dataset[0])


if __name__ == "__main__":
    main()
