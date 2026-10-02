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
  processed per-strain matrix (``proteins_dataset.data_prep.tsv``): a long-format table
  of batch-corrected (SVA-adjusted) label-free protein signal per (protein, sample, KO
  strain, replicate), covering 726 proteins. We aggregate the replicate samples of each
  knockout strain to a per-protein mean + standard error, mapping to
  ``ProteinAbundancePhenotype`` (WS9): ``protein_abundance = {protein_ORF -> mean log
  signal}`` with ``measurement_type = "swath_ms_label_free_log_signal_sva"``. The parent
  **WT** strain (``KO_ORF == "WT"``) supplies the reference profile. Every protein has
  >=2 replicate samples per strain, so the standard error is always defined. A blank or
  non-finite value or a repeated (protein, strain, replicate) row refuses the build (none
  occurs in the pinned release), since each would make ``n``, the mean or the SE wrong.

- ``MetaboliteZelezniak2018Dataset`` -- the targeted central-carbon/amino-acid
  METABOLOME of the same 95 kinase-KO strains (plus a measured WT) by LC-SRM. The file
  ``metabolites_dataset.data_prep.tsv`` is a long-format table of batch-corrected values
  per (protocol, metabolite, strain, replicate). The ``dataset`` column names one of
  three LC-SRM protocols (``ZELEZNIAK_METABOLITE_PROTOCOLS``) that do not share a unit:
  Dataset 1 is an UNCALIBRATED batch-corrected SRM signal measured on the proteomics
  cultures, Datasets 2 and 3 are externally calibrated values measured on separately
  re-grown cultures with different chromatography. One record is written per (strain,
  protocol): a per-metabolite mean + standard error over that protocol's replicates,
  mapping to ``MetabolitePhenotype`` (WS9) with the protocol's own ``measurement_type``,
  and the reference is the WT measured under the SAME protocol, so no record or
  reference ever averages or compares values from two protocols (issue #595). This is
  the first dataset to populate ``target_metabolite_ids`` (metabolite -> Yeast9
  ``s_NNNN``), sourced from ``YeastGEM`` (never invented), enabling constraint-based-model
  linkage.
    - Columns (differ from the Zenodo README): ``metabolite_id, kegg_id, official_name,
      dataset, genotype, replicate, value``. ``genotype`` is the strain (systematic
      kinase ORF, or literal ``WT``); NOT a KO_ORF column. ``metabolite_id`` is a
      BiGG-style id (50 total); ~5 are co-elution merges joined with ``;`` (e.g.
      ``3pg;2pg``, ``ala-L;ala-B``, ``g6p;f6p;g6p-B``) which we KEEP verbatim as dict
      keys (honest to source; resolved via the FIRST sub-id). ``dataset`` is the protocol
      used for generation (1/2/3) and is part of the aggregation key, so
      ``n_replicates`` counts one protocol's rows only.

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
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.metabolism.yeast_GEM import YeastGEM
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


def _paper_sv(value: Any, quote: str, note: str | None = None) -> SourcedValue:
    """A ``SourcedValue`` anchored to the mirrored ``paper.md`` by sha256."""
    return SourcedValue(
        value=value,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=CITATION_KEY,
            sha256=PAPER_MD_SHA256,
            method="MinerU OCR of the publisher PDF (mirrored artifact)",
            page="STAR Methods, Metabolomics",
        ),
        quote=quote,
        note=note,
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
        method="Zenodo record 1320289 README (not mirrored) or the authors' "
        "kinase_metabolism code (https://github.com/zelezniak-lab/kinase_metabolism)",
    ),
    note=(
        "the calibration standards are stated (500 uM to 100 nM serial dilution) and a "
        "dilution + cell-volume conversion is described for Figures 4F-4H, but neither "
        "the paper nor metabolites_dataset.data_prep.tsv states the unit of the "
        "deposited calibrated value, nor whether that conversion was applied to it"
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
    """SWATH-MS proteome of the yeast kinase-knockout collection (97 strains)."""

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
    def _aggregate(sub: pd.DataFrame, strain: str) -> dict[str, Any]:
        """Aggregate one strain's replicate rows to per-protein mean/se/n dicts.

        ``n`` is the row count per protein, so every row must be one distinct replicate
        with a value. A blank value (which pandas would leave out of ``n`` unrecorded)
        and a repeated (protein, replicate) id (which would count as an extra replicate
        and shrink the SE) both refuse, naming the strain, and so does a non-finite
        value (``inf`` passes the blank check but has no mean or SE). Measured on the
        pinned release (sha256 ``9ff81ecb...``): 0 blank, 0 non-finite and 0 repeated
        (ORF, KO_ORF, replicate) rows of 264,264, so no refusal fires on the real file.
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

    @post_process
    def process(self) -> None:
        """Aggregate the proteome matrix into per-strain experiments and write LMDB."""
        verify_raw_files(self.raw_dir, {DATA_FILENAME: DATA_SHA256})
        df = pd.read_csv(osp.join(self.raw_dir, DATA_FILENAME), sep="\t")
        bad = df[~df["ORF"].astype(str).str.match(_SYSTEMATIC_RE)]
        if len(bad):
            raise RuntimeError(f"non-systematic protein ORF ids present: {len(bad)}")

        wt_rows = df[df["KO_ORF"] == _WT]
        if wt_rows.empty:
            raise RuntimeError("Zelezniak matrix missing the WT reference strain")
        self._reference = self._aggregate(wt_rows, _WT)

        os.makedirs(self.preprocess_dir, exist_ok=True)
        n_bad_orf = 0
        rows: list[dict[str, Any]] = []
        for ko_orf, sub in df[df["KO_ORF"] != _WT].groupby("KO_ORF"):
            if not _SYSTEMATIC_RE.match(str(ko_orf)):
                n_bad_orf += 1
                continue
            gene = str(sub["KO_gene_name"].iloc[0])
            rows.append(
                {
                    "orf": str(ko_orf),
                    "gene": gene,
                    "agg": self._aggregate(sub, str(ko_orf)),
                }
            )
        log.info(
            "Zelezniak: %d knockout strains, WT reference with %d proteins, "
            "%d non-systematic KO_ORF skipped",
            len(rows),
            len(self._reference["abundance"]),
            n_bad_orf,
        )
        pd.DataFrame([{"orf": r["orf"], "gene": r["gene"]} for r in rows]).to_csv(
            osp.join(self.preprocess_dir, "data.csv"), index=False
        )

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
        """Build the ProteinAbundance experiment/reference/publication for one strain."""
        # Background = BY4741 kinase-deletion collection made prototrophic by the pHLUM
        # minichromosome (restores HIS3/LEU2/URA3/MET17); pHLUM not yet modeled here.
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        )
        genotype = Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=row["orf"], perturbed_gene_name=row["gene"]
                )
            ]
        )
        # SWATH-MS on cells in synthetic minimal (SM) liquid medium, 30 C. The
        # paper states NO recipe for that medium and sources both the strains and the
        # cultivation to Mulleder 2012, which is not mirrored, so ``SM_DEFERRED`` carries
        # the deferral rather than borrowing the Mulleder 2016 / Messner 2023 grams.
        environment = Environment(media=SM_DEFERRED, temperature=Temperature(value=30))
        agg = row["agg"]
        phenotype = ProteinAbundancePhenotype(
            protein_abundance=agg["abundance"],
            protein_abundance_se=agg["se"],
            n_replicates=agg["n"],
            measurement_type=MEASUREMENT_TYPE,
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
    (strain, protocol); each record and its WT reference share one protocol.
    """

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
    def _aggregate(sub: pd.DataFrame, strain: str, dataset: int) -> dict[str, Any]:
        """Aggregate one (strain, protocol) block of rows to per-metabolite mean/se/n dicts.

        ``sub`` holds the rows of ONE protocol (one ``dataset`` value) for one strain, so
        mean = mean over that protocol's replicates, SE = sample_SD * n**-0.5 when n>1
        else NaN, n = row count. Protocols are never pooled (issue #595): Dataset 1 is an
        uncalibrated signal and Datasets 2/3 are calibrated values. ``n`` is the row
        count, so a repeated (metabolite, replicate) id inside the block refuses, naming
        the strain and protocol. Measured on the pinned release (sha256 ``c4429fd8...``):
        0 repeated (dataset, metabolite, genotype, replicate) rows of 3,522.
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
        """Aggregate the metabolome into per-(strain, protocol) experiments and write LMDB."""
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
        for (_, genotype), sub in strains.groupby(["dataset", "genotype"]):
            protocol = int(sub["dataset"].iloc[0])
            agg = self._aggregate(sub, str(genotype), protocol)
            shared = set(agg["level"]) & set(self._references[protocol]["level"])
            if not shared:
                raise RuntimeError(
                    f"Zelezniak metabolome strain {genotype} protocol {protocol} shares "
                    "no metabolite with that protocol's WT; its reference would be empty"
                )
            rows.append({"orf": str(genotype), "dataset": protocol, "agg": agg})
        log.info(
            "Zelezniak metabolome: %d (strain, protocol) records over protocols %s, WT "
            "reference metabolites per protocol %s, %d metabolite ids mapped to Yeast9 "
            "s_NNNN",
            len(rows),
            sorted(self._references),
            {d: len(r["level"]) for d, r in sorted(self._references.items())},
            len(self._s_id_map),
        )
        pd.DataFrame(
            [
                {
                    "orf": r["orf"],
                    "dataset": r["dataset"],
                    "n_metabolites": len(r["agg"]["level"]),
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

    def _phenotype(self, agg: dict[str, Any], dataset: int) -> MetabolitePhenotype:
        """Build a MetabolitePhenotype for one protocol's aggregated per-metabolite dict."""
        level = dict(agg["level"])
        se = dict(agg["se"])
        # SE keys must be a subset of level keys; the all-NaN case collapses to None.
        level_se: dict[str, float] | None = se
        if all(math.isnan(v) for v in se.values()):
            level_se = None
        targets = {m: self._s_id_map[m] for m in level}
        return MetabolitePhenotype(
            metabolite_level=level,
            metabolite_level_se=level_se,
            n_replicates=dict(agg["n"]),
            measurement_type=ZELEZNIAK_METABOLITE_PROTOCOLS[dataset].measurement_type,
            target_metabolite_ids=targets,
        )

    def create_experiment(  # type: ignore[override]
        self, row: dict[str, Any]
    ) -> tuple[MetaboliteExperiment, MetaboliteExperimentReference, Publication]:
        """Build the Metabolite experiment/reference/publication for one (strain, protocol)."""
        # Background = BY4741 kinase-deletion collection made prototrophic by the pHLUM
        # minichromosome (restores HIS3/LEU2/URA3/MET17); pHLUM not yet modeled here.
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        )
        # No gene-name column in the metabolome file (only the systematic ORF), so the
        # systematic id is used for both names (as in the Mulleder loader).
        genotype = Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=row["orf"], perturbed_gene_name=row["orf"]
                )
            ]
        )
        # SRM-MS/MS on cells in synthetic minimal (SM) liquid medium, 30 C (as proteome);
        # same unstated recipe, deferred to Mulleder 2012 on ``SM_DEFERRED``.
        environment = Environment(media=SM_DEFERRED, temperature=Temperature(value=30))
        dataset = int(row["dataset"])
        phenotype = self._phenotype(row["agg"], dataset)
        # Reference = the WT measured by the SAME protocol, RESTRICTED to the metabolites
        # this strain measured under it. Targeted-metabolomics coverage is sparse (the
        # protocol-1 WT measured 13 of that protocol's 17 ids, never adp/amp/atp/e4p), so
        # a strain can measure metabolites the WT lacks; those simply have no WT
        # baseline. Restricting keeps reference keys a subset of the experiment's and
        # every reference value a real same-protocol WT measurement (never invented);
        # process() refuses a (strain, protocol) that shares no metabolite with its WT.
        exp_keys = set(row["agg"]["level"])
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
