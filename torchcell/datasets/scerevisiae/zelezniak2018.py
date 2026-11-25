# torchcell/datasets/scerevisiae/zelezniak2018
# [[torchcell.datasets.scerevisiae.zelezniak2018]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/zelezniak2018
# Test file: tests/torchcell/datasets/scerevisiae/test_zelezniak2018.py
"""Zelezniak 2018 kinase-knockout multi-omics datasets (proteome + metabolome).

Zelezniak et al. 2018 (Cell Systems, doi:10.1016/j.cels.2018.08.001) profiled the
S. cerevisiae kinase-deletion collection. This module holds the two matching loaders,
both from the same Zenodo record 1320289 (concept DOI 10.5281/zenodo.1320288):

- ``ProteomeZelezniak2018Dataset`` -- the quantitative PROTEOME of 97 kinase-deletion
  strains (plus WT) by SWATH-MS (data-independent acquisition). We ingest the authors'
  processed matrix (``proteins_dataset.data_prep.tsv``): a long-format table of
  batch-corrected (SVA-adjusted) label-free protein signal per (protein, sample, KO
  strain, replicate), covering 726 proteins in each of 364 samples. ONE RECORD PER
  RELEASED SAMPLE (issue #1174, policy #1178): each sample is its own
  ``ProteinAbundancePhenotype`` (WS9) with ``protein_abundance = {protein_ORF -> that
  sample's log signal}`` verbatim, ``n_replicates = 1`` per protein, no SE map (a typed
  ``not_applicable_single_measurement`` gap instead), ``replicate_id`` = the release's
  own ``replicate`` label and ``measurement_type =
  "swath_ms_label_free_log_signal_sva"``. 352 knockout samples over 97 strains plus the
  12 parent **WT** samples (``KO_ORF == "WT"``), stored with an empty genotype. The
  reference profile is the WT summary (mean, SD / sqrt(n), n over the 12 WT samples,
  ``replicate_id=None``). Before 2026-11-24 the loader stored one mean + SE record per
  knockout strain (97 records) and kept the WT samples only inside the reference. A
  blank or non-finite value, a repeated (protein, strain, replicate) row, or a
  (strain, replicate) label that names two samples refuses the build (none occurs in
  the pinned release).

- ``MetaboliteZelezniak2018Dataset`` -- the targeted central-carbon/amino-acid
  METABOLOME of the same 95 kinase-KO strains (plus a measured WT) by LC-SRM. The file
  ``metabolites_dataset.data_prep.tsv`` is a long-format table of batch-corrected values
  per (protocol, metabolite, strain, replicate). The ``dataset`` column names one of
  three LC-SRM protocols (``ZELEZNIAK_METABOLITE_PROTOCOLS``) that do not share a unit:
  Dataset 1 is an UNCALIBRATED batch-corrected SRM signal measured on the proteomics
  cultures, Datasets 2 and 3 are externally calibrated values measured on separately
  re-grown cultures with different chromatography. ONE RECORD PER (protocol, strain,
  released biological replicate) (issue #1173, policy #1178): 187 knockout records and
  5 WT records (protocol 1 x1, protocol 2 x1, protocol 3 x3), each the verbatim values
  of one replicate, ``n_replicates = 1``, no SE map (typed gap), ``replicate_id`` = the
  release's ``replicate`` label, with the protocol's own ``measurement_type``. The
  reference is the summary of the WT measured under the SAME protocol, so no record or
  reference ever averages or compares values from two protocols (issue #595). Before
  2026-11-24 the loader stored one mean + SE record per (strain, protocol) (129
  records). This is the first dataset to populate ``target_metabolite_ids`` (metabolite
  -> Yeast9 ``s_NNNN``), sourced from ``YeastGEM`` (never invented), enabling
  constraint-based-model linkage.
    - Columns (differ from the Zenodo README): ``metabolite_id, kegg_id, official_name,
      dataset, genotype, replicate, value``. ``genotype`` is the strain (systematic
      kinase ORF, or literal ``WT``); NOT a KO_ORF column. ``metabolite_id`` is a
      BiGG-style id (50 total); ~5 are co-elution merges joined with ``;`` (e.g.
      ``3pg;2pg``, ``ala-L;ala-B``, ``g6p;f6p;g6p-B``) which we KEEP verbatim as dict
      keys (honest to source; resolved via the FIRST sub-id). ``dataset`` is the protocol
      used for generation (1/2/3) and is part of the record key.

WHAT A REPLICATE IS. The Zenodo ``README.txt`` (raw mirror
``torchcell-raw/zelezniakMachineLearningPredicts2018/data/zenodo/README.txt``, sha256
``9e301a8b...``) defines the metabolome column as ``replicate - biological replicate``
and the proteome columns as ``sample - sample name from the study`` and ``replicate -
replicate``. The paper (``REPLICATE_SOURCED_VALUES``) states triplicate cultures per
strain and "3 independent cultures" for metabolome Datasets 2 and 3. For the proteome,
24 strains (and WT) release more than three samples, named with an ``_1`` / ``_2``
suffix on a culture letter (``KL_Try_101_a`` and ``KL_Try_101_a_1``); the release and
the paper do not say whether a suffixed sample is a re-grown culture or a re-injection.
Measured on the pinned bytes, a suffixed sample is no closer to its same-letter sample
than to a different-letter sample of the other suffix group (median Pearson r of the
protein-centered profiles 0.228 over 59 same-letter pairs vs 0.259 over 137
different-letter pairs), so the data give no sign that it is a repeat injection of one
digest; every released sample is a record, and the open question is recorded in the
dataset note.

For both, the background is BY4741 made prototrophic by the pHLUM minichromosome. The
proteome ``?download=1`` URL works, but it 403s for the metabolome file, which is fetched
via the Zenodo API content endpoint instead.
"""

import logging
import math
import os
import os.path as osp
import pickle
import re
import urllib.request
from typing import Any

import lmdb
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    Organism,
    post_process,
    verify_raw_files,
    write_verified,
)
from torchcell.datamodels.media import SM_DEFERRED
from torchcell.datamodels.schema import (
    Environment,
    Experiment,
    ExperimentReference,
    Genotype,
    KanMxDeletionPerturbation,
    MetaboliteExperiment,
    MetaboliteExperimentReference,
    MetabolitePhenotype,
    ProteinAbundanceExperiment,
    ProteinAbundanceExperimentReference,
    ProteinAbundancePhenotype,
    Publication,
    ReferenceGenome,
    ReleasedIdentifierPerturbation,
    ReleasedKind,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.metabolism.yeast_GEM import YeastGEM
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

MEASUREMENT_TYPE = "swath_ms_label_free_log_signal_sva"
_SYSTEMATIC_RE = re.compile(r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?$")


def _released_deletion(
    orf: str, gene: str
) -> KanMxDeletionPerturbation | ReleasedIdentifierPerturbation:
    """The deletion under the released identifier (#1045): an ORF-shaped one on the
    placed KanMX leaf with the released KO_gene_name as the perturbed name, any other
    on the released-identifier leaf verbatim (kind deletion, marker KanMX).
    """
    if _SYSTEMATIC_RE.match(orf):
        return KanMxDeletionPerturbation(
            systematic_gene_name=orf, perturbed_gene_name=gene
        )
    return ReleasedIdentifierPerturbation(
        systematic_gene_name=orf,
        gene_namespace="scerevisiae_s288c_sgd_systematic",
        perturbed_gene_name=gene,
        identifier_kind="KO ORF column entry, as released",
        released_kind=ReleasedKind.deletion,
        cassette="KanMX",
    )


_WT = "WT"

# Zenodo record 1320289 -> proteins_dataset.data_prep.tsv (direct, pinned).
DATA_URL = (
    "https://zenodo.org/records/1320289/files/proteins_dataset.data_prep.tsv?download=1"
)
DATA_FILENAME = "proteins_dataset.data_prep.tsv"
DATA_SHA256 = "9ff81ecb1e2dd44d2f6e072ce5b628f0be1abdf57cdbd90d645db4d1fb64bfeb"

# --------------------------------------------------------------------------- #
# Metabolome protocols (issue #595). The ``dataset`` column of the metabolome file names
# the LC-SRM protocol; each protocol is its own unit system, sourced below.
# --------------------------------------------------------------------------- #
CITATION_KEY = "zelezniakMachineLearningPredicts2018"
PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "072bfb2d5b601d578dd5370ed25bfe2709bba0273843c87b5580e248573ce196"
ZENODO_CONCEPT_DOI_URL = "https://doi.org/10.5281/zenodo.1320288"


def _paper_sv(
    value: Any,
    quote: str,
    note: str | None = None,
    page: str = "STAR Methods, Metabolomics",
) -> SourcedValue:
    """A ``SourcedValue`` anchored to the mirrored ``paper.md`` by sha256."""
    return SourcedValue(
        value=value,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=CITATION_KEY,
            sha256=PAPER_MD_SHA256,
            method="MinerU OCR of the publisher PDF (mirrored artifact)",
            page=page,
        ),
        quote=quote,
        note=note,
    )


# The Zenodo record's README, in the RAW mirror
# ($DATA_ROOT/torchcell-raw/<citation_key>/data/zenodo/README.txt), fetched 2026-11-24
# from https://zenodo.org/api/records/1320289/files/README.txt/content (md5 matches the
# record's listing, 31cc2d70...).
ZENODO_README = "data/zenodo/README.txt"
ZENODO_README_SHA256 = (
    "9e301a8bf7a4a46d15e1e57a121909a97bfe2fb011c998483fb28ec0dd39efbd"
)


def _readme_sv(value: Any, quote: str, note: str | None = None) -> SourcedValue:
    """A ``SourcedValue`` anchored to the raw-mirrored Zenodo ``README.txt`` by sha256."""
    return SourcedValue(
        value=value,
        provenance=Provenance(
            source_uri=ZENODO_README,
            citation_key=CITATION_KEY,
            sha256=ZENODO_README_SHA256,
            method="Zenodo record 1320289 README.txt, raw mirror",
            page="README.txt column descriptions",
        ),
        quote=quote,
        note=note,
    )


# What one released replicate is, per modality (issues #1173, #1174; policy #1178).
# Paper quotes audit against the literature mirror, README quotes against the raw mirror.
REPLICATE_SOURCED_VALUES: dict[str, SourcedValue] = {
    "proteome_cultures": _paper_sv(
        "three cultures per kinase-deletion strain",
        "97 of the strains grew in triplicates",
        page="STAR Methods, Strains and Culture",
    ),
    "proteome_sample_count": _paper_sv(
        "397 whole-proteome samples stated; 364 released",
        "The measurement of the 97 proteomes mounted to 397 whole-proteome samples "
        "(triplicates plus controls)",
        note=(
            "the processed release holds 364 samples (352 knockout + 12 WT); the paper "
            "does not identify the other 33, so none is recorded as withdrawn"
        ),
        page="Results, Kinase Knockout Proteomes Are Dominated by Differential Enzyme "
        "Expression",
    ),
    "proteome_qc_samples": _paper_sv(
        "QC samples are a pooled digest, not a strain sample",
        "External standard quality control (QC) samples were prepared as a mixture of "
        "all proteomes and were measured every 8-12 injections.",
        page="STAR Methods, Proteomics",
    ),
    "dataset1_proteome_replicates_averaged_by_authors_for_modeling": _paper_sv(
        "the authors average proteome replicates only to pair them with Dataset 1",
        "For Dataset 1, replicates of proteomics experiments were averaged per genotype.",
        page="STAR Methods, Statistical Modelling Data Preparation",
    ),
    "dataset23_biological_replicates": _paper_sv(
        "Datasets 2 and 3 hold biological replicates",
        "For Datasets 2 and 3, where multiple biological replicates were available",
        page="STAR Methods, Statistical Modelling Data Preparation",
    ),
}

README_SOURCED_VALUES: dict[str, SourcedValue] = {
    "metabolome_replicate_column": _readme_sv(
        "metabolome replicate = biological replicate",
        "replicate - biological replicate",
    ),
    "proteome_sample_column": _readme_sv(
        "proteome sample = the study's sample name",
        "sample - sample name from the study",
    ),
    "proteome_replicate_column": _readme_sv(
        "proteome replicate = the release's replicate label (kind unstated)",
        "replicate - replicate",
    ),
}

_SINGLE_REPLICATE_SE_NOTE = (
    "a record is ONE released replicate (one proteome sample, or one metabolome "
    "culture of one protocol), and one measurement has no dispersion of its own, so "
    "there is no standard error to store; the per-strain mean and SE over the records "
    "of one strain are a modeling-time derivation"
)
_SINGLE_REFERENCE_SE_NOTE = (
    "the WT under this protocol is released as one replicate, so the reference summary "
    "is that one measurement and has no dispersion"
)


METABOLITE_SOURCED_VALUES: dict[str, SourcedValue] = {
    "dataset1_method": _paper_sv(
        "Dataset 1: Keller 2014 LC-SRM method for glycolytic and PPP intermediates",
        "The method used to obtain Dataset 1 (Figure S13) is described in Keller et al. "
        "(2014) for the quantification of glycolytic and pentose phosphate pathway "
        "metabolites and was expanded with additional transitions for ATP, ADP and AMP.",
    ),
    "dataset23_method": _paper_sv(
        "Datasets 2 and 3: Buescher 2010 chromatography with an optimized SRM set",
        "For Dataset 2 and 3 we adapted chromatographic parameters from Buescher et al. "
        "(2010) and added a SRM set",
    ),
    "dataset2_chromatography": _paper_sv(
        "Dataset 2: tributylamine ion-pair reversed-phase gradient",
        "In Dataset 2 analytes were separated by gradient elution using 10 mM TBA",
    ),
    "dataset3_chromatography": _paper_sv(
        "Dataset 3: HILIC separation of free amino acids",
        "In Dataset 3 (Figure S13), free amino acids were separated by hydrophilic "
        "interaction liquid chromatography (HILIC)",
    ),
    "calibration": _paper_sv(
        "Datasets 2 and 3 externally calibrated; Dataset 1 not calibrated",
        "were quantified by external calibration (except Dataset 1) with standards "
        "prepared at serial dilution from",
    ),
    "cultures": _paper_sv(
        "Dataset 1 from the proteomics cultures; Datasets 2 and 3 from re-grown cultures",
        "Dataset 1 was created from the same cells as grown for the proteomic "
        "experiments. Metabolomics datasets 2 and 3 were obtained by re-growing 3 "
        "independent cultures from strains with highly variable metabolite "
        "concentrations based on dataset 1",
    ),
    "batch_correction": _paper_sv(
        "all protocols ComBat batch-corrected after calibration where applicable",
        "All preprocessed metabolomics data (integrated SRM transition peaks after "
        "external calibration (where applicable)) were corrected for batch effects "
        "using ComBat approach as implemented in sva (Leek et al., 2012) R package.",
    ),
}

CALIBRATED_UNIT_GAP = ProvenanceGap(
    field="unit",
    reason=ProvenanceGapReason.deferred_pending_source_review,
    looked_in=Provenance(
        source_uri=PAPER_MD,
        citation_key=CITATION_KEY,
        sha256=PAPER_MD_SHA256,
        method="full Methods read; processed-file header carries no unit column",
        page="STAR Methods, Metabolomics and Enzyme Saturation",
    ),
    resolve_with=Provenance(
        source_uri=ZENODO_CONCEPT_DOI_URL,
        citation_key=CITATION_KEY,
        method="the authors' kinase_metabolism code "
        "(https://github.com/zelezniak-lab/kinase_metabolism); the Zenodo record "
        "1320289 README, raw-mirrored 2026-11-24 (sha256 9e301a8b...), was read and "
        "states no unit",
    ),
    note=(
        "the calibration standards are stated (500 uM to 100 nM serial dilution) and a "
        "dilution + cell-volume conversion is described for Figures 4F-4H, but neither "
        "the paper nor metabolites_dataset.data_prep.tsv states the unit of the "
        "deposited calibrated value, nor whether that conversion was applied to it; "
        "the Zenodo README describes the column only as 'metabolite signal obtained "
        "from SRM-MS/MS experiment, corrected for batch effects'"
    ),
)


class ZelezniakMetaboliteProtocol(BaseModel):
    """One LC-SRM protocol of the Zelezniak metabolome (one value of the ``dataset`` column).

    ``measurement_type`` is stored on every phenotype of the protocol, so a consumer can
    never treat values from two protocols as one scale. ``unit`` is the sourced unit, or
    ``None`` with a typed ``unit_gap`` when no mirrored source states it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: int
    measurement_type: str
    calibrated: bool
    unit: str | None
    unit_gap: ProvenanceGap | None
    sources: tuple[SourcedValue, ...]

    @model_validator(mode="after")
    def _unit_or_gap(self) -> "ZelezniakMetaboliteProtocol":
        """Exactly one of ``unit`` and ``unit_gap`` is set."""
        if (self.unit is None) == (self.unit_gap is None):
            raise ValueError(
                f"protocol {self.dataset}: exactly one of unit and unit_gap must be set"
            )
        return self


_SV = METABOLITE_SOURCED_VALUES
ZELEZNIAK_METABOLITE_PROTOCOLS: dict[int, ZelezniakMetaboliteProtocol] = {
    1: ZelezniakMetaboliteProtocol(
        dataset=1,
        # Uncalibrated: an ARBITRARY batch-corrected SRM signal, NOT a concentration.
        measurement_type="lc_srm_signal_batch_corrected_uncalibrated_dataset1",
        calibrated=False,
        unit="arbitrary units (batch-corrected SRM peak signal, NOT a concentration)",
        unit_gap=None,
        sources=(
            _SV["dataset1_method"],
            _SV["calibration"],
            _SV["cultures"],
            _SV["batch_correction"],
        ),
    ),
    2: ZelezniakMetaboliteProtocol(
        dataset=2,
        measurement_type=(
            "lc_srm_ion_pair_external_calibration_batch_corrected_unit_unstated_dataset2"
        ),
        calibrated=True,
        unit=None,
        unit_gap=CALIBRATED_UNIT_GAP,
        sources=(
            _SV["dataset23_method"],
            _SV["dataset2_chromatography"],
            _SV["calibration"],
            _SV["cultures"],
            _SV["batch_correction"],
        ),
    ),
    3: ZelezniakMetaboliteProtocol(
        dataset=3,
        measurement_type=(
            "lc_srm_hilic_external_calibration_batch_corrected_unit_unstated_dataset3"
        ),
        calibrated=True,
        unit=None,
        unit_gap=CALIBRATED_UNIT_GAP,
        sources=(
            _SV["dataset23_method"],
            _SV["dataset3_chromatography"],
            _SV["calibration"],
            _SV["cultures"],
            _SV["batch_correction"],
        ),
    ),
}

# Zenodo record 1320289 -> metabolites_dataset.data_prep.tsv. The ?download=1 URL 403s
# for this file, so we hit the Zenodo API content endpoint.
METABOLITE_DATA_URL = (
    "https://zenodo.org/api/records/1320289/files/"
    "metabolites_dataset.data_prep.tsv/content"
)
METABOLITE_DATA_FILENAME = "metabolites_dataset.data_prep.tsv"
METABOLITE_DATA_SHA256 = (
    "c4429fd8cef675d96ffacba1ed51e52ea483fd72d6978a22c04fa405f4e1b07d"
)


@register_dataset
class ProteomeZelezniak2018Dataset(ExperimentDataset):
    """SWATH-MS proteome of the yeast kinase-knockout collection, one record per released
    sample (352 knockout samples over 97 strains plus 12 WT samples).
    """

    organism = Organism.scerevisiae

    def __init__(
        self,
        root: str = "data/torchcell/proteome_zelezniak2018",
        io_workers: int = 0,
        transform: Any | None = None,
        pre_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset (KO_ORF is systematic; no genome injection needed)."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return ProteinAbundanceExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return ProteinAbundanceExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The Zenodo proteome matrix required before processing."""
        return [DATA_FILENAME]

    def download(self) -> None:
        """Download the proteome matrix from Zenodo and verify its sha256."""
        dest = osp.join(self.raw_dir, DATA_FILENAME)
        if osp.exists(dest):
            return
        os.makedirs(self.raw_dir, exist_ok=True)
        log.info("Downloading Zelezniak proteome from %s", DATA_URL)
        req = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = resp.read()
        write_verified(data, dest, DATA_SHA256, DATA_URL)
        log.info("Wrote %s (%d bytes, sha256 verified)", dest, len(data))

    @staticmethod
    def _check_rows(sub: pd.DataFrame, strain: str) -> None:
        """Refuse one strain's rows unless each is one distinct (protein, replicate) value.

        A blank value (which pandas would leave out of ``n`` unrecorded) and a repeated
        (protein, replicate) id (which would count as an extra replicate, or put two
        values under one replicate record) both refuse, naming the strain, and so does a
        non-finite value (``inf`` passes the blank check but has no mean or SE). A
        replicate label that names two released samples refuses too, since its record
        would mix two samples. Measured on the pinned release (sha256 ``9ff81ecb...``):
        0 blank, 0 non-finite and 0 repeated (ORF, KO_ORF, replicate) rows of 264,264,
        and the 364 samples map one to one onto 364 (KO_ORF, replicate) labels, so no
        refusal fires on the real file.
        """
        blank = sub[sub["value"].isna()]
        if len(blank):
            raise RuntimeError(
                f"Zelezniak proteome strain {strain}: {len(blank)} blank protein "
                f"value(s), first {blank['ORF'].iloc[0]} replicate "
                f"{blank['replicate'].iloc[0]}; a blank would drop out of n_replicates "
                "unrecorded"
            )
        infinite = sub[np.isinf(sub["value"].to_numpy(dtype="float64"))]
        if len(infinite):
            raise RuntimeError(
                f"Zelezniak proteome strain {strain}: {len(infinite)} non-finite "
                f"protein value(s), first {infinite['ORF'].iloc[0]} replicate "
                f"{infinite['replicate'].iloc[0]}; a non-finite value has no mean or SE"
            )
        repeated = sub[sub.duplicated(["ORF", "replicate"], keep=False)]
        if len(repeated):
            raise RuntimeError(
                f"Zelezniak proteome strain {strain}: {len(repeated)} rows share a "
                f"(protein, replicate) id, first {repeated['ORF'].iloc[0]} replicate "
                f"{repeated['replicate'].iloc[0]}; a repeated replicate would count "
                "as an extra replicate"
            )
        samples = sub.groupby("replicate")["sample"].nunique()
        shared = samples[samples > 1]
        if len(shared):
            raise RuntimeError(
                f"Zelezniak proteome strain {strain}: replicate {shared.index[0]} names "
                f"{int(shared.iloc[0])} samples; one replicate record would mix them"
            )

    @classmethod
    def _aggregate(cls, sub: pd.DataFrame, strain: str) -> dict[str, Any]:
        """The per-protein mean / SE / n over one strain's samples.

        This is the AGGREGATE the ``phenotype_reference`` states (the WT summary), not a
        record. ``n`` is the row count per protein, so the rows pass ``_check_rows``
        first.
        """
        cls._check_rows(sub, strain)
        grp = sub.groupby("ORF")["value"].agg(["mean", "std", "count"])
        abundance: dict[str, float] = {}
        se: dict[str, float] = {}
        n_reps: dict[str, int] = {}
        for orf, r in grp.iterrows():
            n = int(r["count"])
            abundance[str(orf)] = float(r["mean"])
            n_reps[str(orf)] = n
            se[str(orf)] = float(r["std"]) / math.sqrt(n) if n > 1 else float("nan")
        return {"abundance": abundance, "se": se, "n": n_reps}

    @classmethod
    def _replicate_profiles(
        cls, sub: pd.DataFrame, strain: str
    ) -> list[tuple[str, str, dict[str, float]]]:
        """``[(replicate_id, sample, {ORF: value})]`` for one strain, replicate order.

        Each released sample is one record: its values verbatim, under the release's own
        ``replicate`` label. The rows pass ``_check_rows`` first, so a label names one
        sample and holds one value per protein.
        """
        cls._check_rows(sub, strain)
        profiles: list[tuple[str, str, dict[str, float]]] = []
        for replicate, rep_rows in sub.groupby("replicate", sort=True):
            profiles.append(
                (
                    str(replicate),
                    str(rep_rows["sample"].iloc[0]),
                    {
                        str(orf): float(v)
                        for orf, v in zip(rep_rows["ORF"], rep_rows["value"])
                    },
                )
            )
        return profiles

    @post_process
    def process(self) -> None:
        """Write one ProteinAbundance experiment per released sample to LMDB."""
        verify_raw_files(self.raw_dir, {DATA_FILENAME: DATA_SHA256})
        df = pd.read_csv(osp.join(self.raw_dir, DATA_FILENAME), sep="\t")
        bad = df[~df["ORF"].astype(str).str.match(_SYSTEMATIC_RE)]
        if len(bad):
            raise RuntimeError(f"non-systematic protein ORF ids present: {len(bad)}")
        labels = df.groupby("sample")[["KO_ORF", "replicate"]].nunique()
        relabeled = labels[(labels > 1).any(axis=1)]
        if len(relabeled):
            raise RuntimeError(
                f"Zelezniak proteome sample {relabeled.index[0]} carries more than one "
                "(KO_ORF, replicate) label; it would be split across records"
            )

        wt_rows = df[df["KO_ORF"] == _WT]
        if wt_rows.empty:
            raise RuntimeError("Zelezniak matrix missing the WT reference strain")
        self._reference = self._aggregate(wt_rows, _WT)

        os.makedirs(self.preprocess_dir, exist_ok=True)
        released_names: list[str] = []
        rows: list[dict[str, Any]] = []
        for ko_orf, sub in df.groupby("KO_ORF"):
            strain = str(ko_orf)
            if strain != _WT and not _SYSTEMATIC_RE.match(strain):
                # a released strain id with no ORF form goes on the released-gene-name
                # leaf verbatim (#1045; 0 of 97 released strains)
                released_names.append(strain)
            gene = str(sub["KO_gene_name"].iloc[0])
            for replicate_id, sample, profile in self._replicate_profiles(sub, strain):
                rows.append(
                    {
                        "orf": strain,
                        "gene": gene,
                        "replicate_id": replicate_id,
                        "sample": sample,
                        "abundance": profile,
                    }
                )
        log.info(
            "Zelezniak: %d sample records (%d WT) over %d strains, WT reference with "
            "%d proteins, %d strain ids on the released-identifier leaf",
            len(rows),
            sum(r["orf"] == _WT for r in rows),
            df["KO_ORF"].nunique(),
            len(self._reference["abundance"]),
            len(released_names),
        )
        pd.DataFrame(
            [
                {
                    "orf": r["orf"],
                    "gene": r["gene"],
                    "replicate_id": r["replicate_id"],
                    "sample": r["sample"],
                }
                for r in rows
            ]
        ).to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e11))
        idx = 0
        with env.begin(write=True) as txn:
            for record_row in tqdm(rows):
                experiment, reference, publication = self.create_experiment(record_row)
                txn.put(
                    f"{idx}".encode(),
                    pickle.dumps(
                        {
                            "experiment": experiment.model_dump(),
                            "reference": reference.model_dump(),
                            "publication": publication.model_dump(),
                        }
                    ),
                )
                idx += 1
        env.close()
        log.info("Wrote %d Zelezniak proteome experiments to LMDB", idx)

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(  # type: ignore[override]
        self, row: dict[str, Any]
    ) -> tuple[
        ProteinAbundanceExperiment, ProteinAbundanceExperimentReference, Publication
    ]:
        """Build the ProteinAbundance experiment/reference/publication for one sample."""
        # Background = BY4741 kinase-deletion collection made prototrophic by the pHLUM
        # minichromosome (restores HIS3/LEU2/URA3/MET17); pHLUM not yet modeled here.
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain="BY4741",
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        )
        # A WT sample is the parent strain itself: the empty genotype on the background.
        genotype = Genotype(
            perturbations=(
                []
                if row["orf"] == _WT
                else [_released_deletion(row["orf"], row["gene"])]
            )
        )
        # SWATH-MS on cells in synthetic minimal (SM) liquid medium, 30 C. The
        # paper states NO recipe for that medium and sources both the strains and the
        # cultivation to Mulleder 2012, which is not mirrored, so ``SM_DEFERRED`` carries
        # the deferral rather than borrowing the Mulleder 2016 / Messner 2023 grams.
        environment = Environment(media=SM_DEFERRED, temperature=Temperature(value=30))
        abundance = dict(row["abundance"])
        phenotype = ProteinAbundancePhenotype(
            protein_abundance=abundance,
            protein_abundance_se=None,
            n_replicates=dict.fromkeys(abundance, 1),
            measurement_type=MEASUREMENT_TYPE,
            replicate_id=row["replicate_id"],
            provenance_gaps=[
                ProvenanceGap(
                    field="protein_abundance_se",
                    reason=ProvenanceGapReason.not_applicable_single_measurement,
                    note=_SINGLE_REPLICATE_SE_NOTE,
                )
            ],
        )
        ref = self._reference
        phenotype_reference = ProteinAbundancePhenotype(
            protein_abundance=dict(ref["abundance"]),
            protein_abundance_se=dict(ref["se"]),
            n_replicates=dict(ref["n"]),
            measurement_type=MEASUREMENT_TYPE,
        )
        experiment = ProteinAbundanceExperiment(
            dataset_name=self.name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )
        reference = ProteinAbundanceExperimentReference(
            dataset_name=self.name,
            genome_reference=genome_reference,
            environment_reference=environment.model_copy(),
            phenotype_reference=phenotype_reference,
        )
        publication = Publication(
            pubmed_id="30195436",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/30195436/",
            doi="10.1016/j.cels.2018.08.001",
            doi_url="https://doi.org/10.1016/j.cels.2018.08.001",
        )
        return experiment, reference, publication


def build_metabolite_s_id_map(kegg_by_metabolite: dict[str, str]) -> dict[str, str]:
    """Map each metabolite id -> a Yeast9 ``s_NNNN`` species id from ``YeastGEM``.

    Hybrid, model-sourced resolution (ids come only FROM the model, never invented):
    prefer the KEGG-compound annotation (matched on the first ``;``-separated ``kegg_id``
    token), fall back to the BiGG-metabolite annotation (matched on the first ``;``
    token of the metabolite id). For a ``;``-merged key, the FIRST sub-id is used.
    Cytosolic (compartment ``c``) hits are preferred; when a metabolite has no cytosolic
    form the available compartment is used (e.g. ``b124tc`` -> mitochondrial ``s_0454``).

    Reusable across targeted-metabolome loaders (e.g. Mulleder) by passing the loader's
    ``{metabolite_id: kegg_id}`` map; only this dataset wires it in for now.
    """
    model = YeastGEM().model
    kegg_index: dict[str, list[Any]] = {}
    bigg_index: dict[str, list[Any]] = {}
    for met in model.metabolites:
        for key, index in (
            ("kegg.compound", kegg_index),
            ("bigg.metabolite", bigg_index),
        ):
            ann = met.annotation.get(key)
            if ann is None:
                continue
            for token in ann if isinstance(ann, list) else [ann]:
                index.setdefault(str(token), []).append(met)

    def pick(cands: list[Any]) -> Any:
        cyto = [m for m in cands if m.compartment == "c"]
        return cyto[0] if cyto else cands[0]

    s_ids: dict[str, str] = {}
    for metabolite_id, kegg_id in kegg_by_metabolite.items():
        first_kegg = str(kegg_id).split(";")[0].strip() if kegg_id else ""
        first_bigg = metabolite_id.split(";")[0].strip()
        cands = kegg_index.get(first_kegg) or bigg_index.get(first_bigg)
        if not cands:
            raise RuntimeError(
                f"no Yeast9 s_NNNN found for metabolite '{metabolite_id}' "
                f"(kegg '{first_kegg}', bigg '{first_bigg}')"
            )
        s_ids[metabolite_id] = str(pick(cands).id)
    return s_ids


@register_dataset
class MetaboliteZelezniak2018Dataset(ExperimentDataset):
    """LC-SRM targeted metabolome of the kinase-knockout collection, one record per
    (protocol, strain, released biological replicate), WT included; each record and its
    WT reference share one protocol.
    """

    organism = Organism.scerevisiae

    def __init__(
        self,
        root: str = "data/torchcell/metabolite_zelezniak2018",
        io_workers: int = 0,
        transform: Any | None = None,
        pre_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset (genotype is systematic; no genome injection needed)."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return MetaboliteExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return MetaboliteExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The Zenodo metabolome matrix required before processing."""
        return [METABOLITE_DATA_FILENAME]

    def download(self) -> None:
        """Download the metabolome matrix from Zenodo and verify its sha256."""
        dest = osp.join(self.raw_dir, METABOLITE_DATA_FILENAME)
        if osp.exists(dest):
            return
        os.makedirs(self.raw_dir, exist_ok=True)
        log.info("Downloading Zelezniak metabolome from %s", METABOLITE_DATA_URL)
        req = urllib.request.Request(
            METABOLITE_DATA_URL, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = resp.read()
        write_verified(data, dest, METABOLITE_DATA_SHA256, METABOLITE_DATA_URL)
        log.info("Wrote %s (%d bytes, sha256 verified)", dest, len(data))

    @staticmethod
    def _check_rows(sub: pd.DataFrame, strain: str, dataset: int) -> None:
        """Refuse one (strain, protocol) block unless each row is one distinct value.

        A repeated (metabolite, replicate) id would put two values under one replicate
        record (or count as an extra replicate in a summary), and a blank or non-finite
        value is no measurement; each refuses, naming the strain and protocol. Measured
        on the pinned release (sha256 ``c4429fd8...``): 0 repeated (dataset, metabolite,
        genotype, replicate) rows and 0 blank or non-finite values of 3,522.
        """
        repeated = sub[sub.duplicated(["metabolite_id", "replicate"], keep=False)]
        if len(repeated):
            raise RuntimeError(
                f"Zelezniak metabolome strain {strain} protocol {dataset}: "
                f"{len(repeated)} rows share a (metabolite, replicate) id, first "
                f"{repeated['metabolite_id'].iloc[0]} replicate "
                f"{repeated['replicate'].iloc[0]}; a repeated replicate would count as "
                "an extra replicate"
            )
        values = pd.to_numeric(sub["value"], errors="coerce").to_numpy(dtype="float64")
        not_finite = sub[~np.isfinite(values)]
        if len(not_finite):
            raise RuntimeError(
                f"Zelezniak metabolome strain {strain} protocol {dataset}: "
                f"{len(not_finite)} blank or non-finite value(s), first "
                f"{not_finite['metabolite_id'].iloc[0]} replicate "
                f"{not_finite['replicate'].iloc[0]}"
            )

    @classmethod
    def _aggregate(cls, sub: pd.DataFrame, strain: str, dataset: int) -> dict[str, Any]:
        """The per-metabolite mean / SE / n over one (strain, protocol) block.

        This is the AGGREGATE a ``phenotype_reference`` states (the same-protocol WT
        summary), not a record. ``sub`` holds the rows of ONE protocol (one ``dataset``
        value), so mean = mean over that protocol's replicates, SE = sample_SD * n**-0.5
        when n>1 else NaN, n = row count. Protocols are never pooled (issue #595):
        Dataset 1 is an uncalibrated signal and Datasets 2/3 are calibrated values.
        """
        cls._check_rows(sub, strain, dataset)
        grp = sub.groupby("metabolite_id")["value"].agg(["mean", "std", "count"])
        level: dict[str, float] = {}
        se: dict[str, float] = {}
        n_reps: dict[str, int] = {}
        for metabolite_id, r in grp.iterrows():
            n = int(r["count"])
            level[str(metabolite_id)] = float(r["mean"])
            n_reps[str(metabolite_id)] = n
            se[str(metabolite_id)] = (
                float(r["std"]) / math.sqrt(n) if n > 1 else float("nan")
            )
        return {"level": level, "se": se, "n": n_reps}

    @post_process
    def process(self) -> None:
        """Write one Metabolite experiment per (protocol, strain, replicate) to LMDB."""
        verify_raw_files(
            self.raw_dir, {METABOLITE_DATA_FILENAME: METABOLITE_DATA_SHA256}
        )
        df = pd.read_csv(osp.join(self.raw_dir, METABOLITE_DATA_FILENAME), sep="\t")
        nonwt = df[df["genotype"] != _WT]["genotype"].astype(str)
        bad = nonwt[~nonwt.str.match(_SYSTEMATIC_RE)]
        if len(bad):
            raise RuntimeError(
                f"non-systematic strain genotype ids present: {len(bad)}"
            )

        # metabolite_id -> Yeast9 s_NNNN, sourced from YeastGEM (never invented).
        kegg_by_metabolite = {
            str(r["metabolite_id"]): str(r["kegg_id"])
            for _, r in df[["metabolite_id", "kegg_id"]].drop_duplicates().iterrows()
        }
        self._s_id_map = build_metabolite_s_id_map(kegg_by_metabolite)

        unknown = sorted(
            set(df["dataset"].tolist()) - set(ZELEZNIAK_METABOLITE_PROTOCOLS)
        )
        if unknown:
            raise RuntimeError(
                f"Zelezniak metabolome protocol(s) {unknown} have no sourced "
                "ZelezniakMetaboliteProtocol; their unit and calibration are unknown"
            )

        wt_rows = df[df["genotype"] == _WT]
        if wt_rows.empty:
            raise RuntimeError("Zelezniak metabolome missing the WT reference strain")
        # One WT reference PER PROTOCOL: a record is only ever compared with the WT
        # measured by the same protocol (issue #595).
        self._references: dict[int, dict[str, Any]] = {}
        for protocol in sorted(int(d) for d in wt_rows["dataset"].unique()):
            self._references[protocol] = self._aggregate(
                wt_rows[wt_rows["dataset"] == protocol], _WT, protocol
            )

        os.makedirs(self.preprocess_dir, exist_ok=True)
        rows: list[dict[str, Any]] = []
        strains = df[df["genotype"] != _WT]
        for protocol in sorted(int(d) for d in strains["dataset"].unique()):
            if protocol not in self._references:
                raise RuntimeError(
                    f"Zelezniak metabolome protocol {protocol} has strain rows but no "
                    "WT rows; its records would have no same-protocol reference"
                )
        # One record per released biological replicate of every (protocol, strain), the
        # WT replicates included (empty genotype). The WT summary above is the reference.
        for (_, genotype), sub in df.groupby(["dataset", "genotype"]):
            protocol = int(sub["dataset"].iloc[0])
            self._check_rows(sub, str(genotype), protocol)
            for replicate, rep_rows in sub.groupby("replicate", sort=True):
                level = {
                    str(m): float(v)
                    for m, v in zip(rep_rows["metabolite_id"], rep_rows["value"])
                }
                shared = set(level) & set(self._references[protocol]["level"])
                if not shared:
                    raise RuntimeError(
                        f"Zelezniak metabolome strain {genotype} protocol {protocol} "
                        f"replicate {replicate} shares no metabolite with that "
                        "protocol's WT; its reference would be empty"
                    )
                rows.append(
                    {
                        "orf": str(genotype),
                        "dataset": protocol,
                        "replicate_id": str(replicate),
                        "level": level,
                    }
                )
        log.info(
            "Zelezniak metabolome: %d (protocol, strain, replicate) records (%d WT) over "
            "protocols %s, WT reference metabolites per protocol %s, %d metabolite ids "
            "mapped to Yeast9 s_NNNN",
            len(rows),
            sum(r["orf"] == _WT for r in rows),
            sorted(self._references),
            {d: len(r["level"]) for d, r in sorted(self._references.items())},
            len(self._s_id_map),
        )
        pd.DataFrame(
            [
                {
                    "orf": r["orf"],
                    "dataset": r["dataset"],
                    "replicate_id": r["replicate_id"],
                    "n_metabolites": len(r["level"]),
                }
                for r in rows
            ]
        ).to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e11))
        idx = 0
        with env.begin(write=True) as txn:
            for record_row in tqdm(rows):
                experiment, reference, publication = self.create_experiment(record_row)
                txn.put(
                    f"{idx}".encode(),
                    pickle.dumps(
                        {
                            "experiment": experiment.model_dump(),
                            "reference": reference.model_dump(),
                            "publication": publication.model_dump(),
                        }
                    ),
                )
                idx += 1
        env.close()
        log.info("Wrote %d Zelezniak metabolome experiments to LMDB", idx)

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def _replicate_phenotype(
        self, level: dict[str, float], dataset: int, replicate_id: str
    ) -> MetabolitePhenotype:
        """One released replicate's values verbatim: n = 1 and no dispersion."""
        return MetabolitePhenotype(
            metabolite_level=dict(level),
            metabolite_level_se=None,
            n_replicates=dict.fromkeys(level, 1),
            measurement_type=ZELEZNIAK_METABOLITE_PROTOCOLS[dataset].measurement_type,
            target_metabolite_ids={m: self._s_id_map[m] for m in level},
            replicate_id=replicate_id,
            provenance_gaps=[
                ProvenanceGap(
                    field="metabolite_level_se",
                    reason=ProvenanceGapReason.not_applicable_single_measurement,
                    note=_SINGLE_REPLICATE_SE_NOTE,
                )
            ],
        )

    def _phenotype(self, agg: dict[str, Any], dataset: int) -> MetabolitePhenotype:
        """The reference summary over one protocol's WT replicates (``replicate_id=None``).

        When every key rests on one replicate the SE map is all NaN, so it is stored as
        ``None`` with a typed ``not_applicable_single_measurement`` gap (protocols 1 and
        2 release one WT replicate each).
        """
        level = dict(agg["level"])
        se = dict(agg["se"])
        # SE keys must be a subset of level keys; the all-NaN case collapses to None.
        level_se: dict[str, float] | None = se
        gaps: list[ProvenanceGap] = []
        if all(math.isnan(v) for v in se.values()):
            level_se = None
            gaps.append(
                ProvenanceGap(
                    field="metabolite_level_se",
                    reason=ProvenanceGapReason.not_applicable_single_measurement,
                    note=_SINGLE_REFERENCE_SE_NOTE,
                )
            )
        targets = {m: self._s_id_map[m] for m in level}
        return MetabolitePhenotype(
            metabolite_level=level,
            metabolite_level_se=level_se,
            n_replicates=dict(agg["n"]),
            measurement_type=ZELEZNIAK_METABOLITE_PROTOCOLS[dataset].measurement_type,
            target_metabolite_ids=targets,
            provenance_gaps=gaps,
        )

    def create_experiment(  # type: ignore[override]
        self, row: dict[str, Any]
    ) -> tuple[MetaboliteExperiment, MetaboliteExperimentReference, Publication]:
        """Build the Metabolite experiment/reference/publication for one replicate."""
        # Background = BY4741 kinase-deletion collection made prototrophic by the pHLUM
        # minichromosome (restores HIS3/LEU2/URA3/MET17); pHLUM not yet modeled here.
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain="BY4741",
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        )
        # No gene-name column in the metabolome file (only the systematic ORF), so the
        # systematic id is used for both names (as in the Mulleder loader). A WT
        # replicate is the parent strain itself: the empty genotype on the background.
        genotype = Genotype(
            perturbations=(
                []
                if row["orf"] == _WT
                else [
                    KanMxDeletionPerturbation(
                        systematic_gene_name=row["orf"], perturbed_gene_name=row["orf"]
                    )
                ]
            )
        )
        # SRM-MS/MS on cells in synthetic minimal (SM) liquid medium, 30 C (as proteome);
        # same unstated recipe, deferred to Mulleder 2012 on ``SM_DEFERRED``.
        environment = Environment(media=SM_DEFERRED, temperature=Temperature(value=30))
        dataset = int(row["dataset"])
        phenotype = self._replicate_phenotype(
            row["level"], dataset, row["replicate_id"]
        )
        # Reference = the WT measured by the SAME protocol, RESTRICTED to the metabolites
        # this replicate measured under it. Targeted-metabolomics coverage is sparse (the
        # protocol-1 WT measured 13 of that protocol's 17 ids, never adp/amp/atp/e4p), so
        # a strain can measure metabolites the WT lacks; those simply have no WT
        # baseline. Restricting keeps reference keys a subset of the experiment's and
        # every reference value a real same-protocol WT measurement (never invented);
        # process() refuses a replicate that shares no metabolite with its WT.
        exp_keys = set(row["level"])
        ref = self._references[dataset]
        ref_agg = {
            "level": {k: v for k, v in ref["level"].items() if k in exp_keys},
            "se": {k: v for k, v in ref["se"].items() if k in exp_keys},
            "n": {k: v for k, v in ref["n"].items() if k in exp_keys},
        }
        phenotype_reference = self._phenotype(ref_agg, dataset)
        experiment = MetaboliteExperiment(
            dataset_name=self.name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )
        reference = MetaboliteExperimentReference(
            dataset_name=self.name,
            genome_reference=genome_reference,
            environment_reference=environment.model_copy(),
            phenotype_reference=phenotype_reference,
        )
        publication = Publication(
            pubmed_id="30195436",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/30195436/",
            doi="10.1016/j.cels.2018.08.001",
            doi_url="https://doi.org/10.1016/j.cels.2018.08.001",
        )
        return experiment, reference, publication


def main() -> None:
    """Build/load the datasets for interactive debugging.

    Loads the existing LMDB if already built. To step through
    ``process()``/``create_experiment`` under a debugger, delete ``<root>/processed``
    first so the build re-runs.
    """
    from dotenv import load_dotenv

    load_dotenv()
    proteome_root = osp.join(
        os.environ["DATA_ROOT"], "data/torchcell/proteome_zelezniak2018"
    )
    proteome = ProteomeZelezniak2018Dataset(root=proteome_root)
    print(f"proteome len = {len(proteome)}")
    print(proteome[0])

    metabolite_root = osp.join(
        os.environ["DATA_ROOT"], "data/torchcell/metabolite_zelezniak2018"
    )
    metabolite = MetaboliteZelezniak2018Dataset(root=metabolite_root)
    print(f"metabolite len = {len(metabolite)}")
    print(metabolite[0])


if __name__ == "__main__":
    main()
