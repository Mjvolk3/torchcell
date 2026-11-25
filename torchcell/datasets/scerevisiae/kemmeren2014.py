# torchcell/datasets/scerevisiae/kemmeren2014
# [[torchcell.datasets.scerevisiae.kemmeren2014]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/kemmeren2014
# Test file: tests/torchcell/datasets/scerevisiae/test_kemmeren2014.py
"""Kemmeren 2014 deletion-mutant and wildtype microarray expression, one record per array."""

import logging
import os
import os.path as osp
import pickle
import re
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any, cast

import GEOparse
import httpx
import lmdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from torchcell.data import ExperimentDataset, Organism, post_process
from torchcell.datamodels.schema import (
    Environment,
    Experiment,
    ExperimentReference,
    GenePerturbationType,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    MicroarrayExpressionExperiment,
    MicroarrayExpressionExperimentReference,
    MicroarrayExpressionPhenotype,
    Publication,
    ReferenceGenome,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ============================================================================
# The record unit: one GEO array is one culture (2026.11.24, #1178, #1174)
# ============================================================================
# Source: torchcell-library kemmerenLargeScaleGeneticPerturbations2014/paper.md (sha256
# 299eac4b07fc977a89c24ce190d05cfd182342a9f0d56a05e7e477d72e4ad08d).
#
# Design, Extended Experimental Procedures, "Experiment Design" (paper.md line 307):
#     "Two independent cultures were hybridized on two separate microarrays. For the
#     first hybridization the Cy5 (red) labeled cRNA from the deletion mutant is
#     hybridized together with the Cy3 (green) labeled cRNA from the common
#     reference. For the replicate hybridization from the independent cultures, the
#     labels are swapped. Each gene is represented twice on the microarray, resulting
#     in four measurements per mutant."
# The cultures come from two isolates (line 247): "For each mutant strain two cultures
# from two isolates were grown in batches in parallel with two WT cultures (same day
# WTs)." The Figure S1B legend (line 496): "cRNA from replicate mutant cultures (a, b)
# was cohybridized in dye-swap with cRNA from a batch of common reference WT RNA."
# The main text's "Each culture was expression-profiled in technical replicate to yield
# four measurements for each profiling mutant" (line 130) is the duplicate spot: "Each
# gene probe is spotted twice on the array" (line 311), and one culture gives two
# measurements (line 394): "for some of the nonresponsive mutants there are four
# replicate measurements while for others there are only two."
#
# So an "-a" / "-b" array pair is two biological replicates (two cultures of two
# isolates) whose dye orientation differs, NOT one culture profiled twice with the dyes
# swapped, and each array is its own record. Measured on the GEO SOFT files
# (2026.11.24): GSE42527 1,387 arrays and GSE42526 1,246, every one a deletion; no
# mutant has more than 2 arrays (1,149 mutants with 2, 335 with 1, 2,633 arrays).
# GEO labels ch1 = Cy5 and ch2 = Cy3 on every array, and source_name_ch1 / ch2 name the
# channel of the common reference pool ("refpool"; "ref1" on 193 GSE42217 arrays);
# 3,061 of 3,061 arrays of the six series name it in exactly one channel. Under that
# assignment (2026.11.24, per array, mean over the gene's spots): the deleted gene reads
# lower in the deletion channel on 2,603 of the 2,625 deletion arrays that carry a probe
# for it, and on all 48 MATa arrays (28 WT, 20 deletion) the a-specific genes (STE2,
# MFA1, MFA2, BAR1, AGA2, STE6) minus the alpha-specific ones (STE3, MF(ALPHA)1,
# MF(ALPHA)2, SAG1) read +5.9 to +9.1 log2 against the refpool, a MATalpha RNA.
#
# The stored log2 ratio is log2(test / reference) of each spot's two "Signal Norm"
# values (as GEO releases them, after the LOESS and gene-specific dye bias correction
# its Sample_data_processing names), NOT the GEO VALUE column: VALUE's orientation is
# not uniform across the deposit. On 2,169 deletion arrays VALUE equals
# log2(test / reference) and on 464 it equals log2(reference / test); on the WT series
# 261 and 166, with one array (GSM1035586) matching neither within 1e-4. VALUE's column
# description does not say which (the same "normalized log2 ratio (Cy3/Cy5)" heads
# both kinds, and "normalized log2 ratio test/ref" heads 3 reversed arrays), and the
# deleted-gene and mating-type checks above hold on both kinds, so the channel signals
# are the consistent release value.
#
# The two spots of a gene on one array are technical (one hybridization of one
# culture's RNA) and are combined INSIDE the record: the record's log2 ratio of a gene
# is the mean of its spots' log2 ratios, and its linear signals are the means of its
# spots' "Signal Norm" values. GPL11232 carries 2 spots for 6,157 genes and 4 for 12
# (measured 2026.11.24). Before 2026.11.24 the loader kept the last spot of each gene
# and dropped the other.
#
# n_replicates is 1 for every gene of a record (one culture); the spot count is the
# technical level and is stated in the uncertainty gap note, because the phenotype class
# has no field for it. A record carries no dispersion: expression_log2_ratio_se and
# expression_log2_ratio_variance are None with a not_applicable_single_measurement gap.
# The mean and dispersion over a mutant's arrays are a modeling-time derivation.
#
# The reference of a record is the common reference pool's signal on that same array
# (log2 ratio 0, n 1): it is what the record's ratio is taken against. The refpool is one
# batch of RNA (GEO overall design: "RNA isolated from a large amount of wt yeast from a
# single culture was used as a common reference"), so there are no reference replicates
# to summarize; the wildtype cultures are records of their own (below).
#
# Withdrawn samples: none in the deposit. The paper removed failed hybridizations and
# profiles of remade strains before deposition (lines 237, 354, 360-364); nothing the
# six GEO series release is marked withdrawn, so no array is dropped.

# ============================================================================
# Wildtype cultures: the four WT series are records too (2026.11.24)
# ============================================================================
# The paper's WT pools (paper.md line 388):
#     "For the MATalpha mutants grown on the Tecan plate shaker, the WT pool consisted
#     of 200 MATalpha WTs (BY4742, ArrayExpress accession E-TABM-984, GEO accession
#     GSE42217). For the MATa mutants grown on the Tecan plate shaker, the WT pool
#     consisted of 20 WT-MATa replicates (BY4741, ArrayExpress accession E-MTAB-1351,
#     GEO accession GSE42241). For mutants grown in Erlenmeyers, the MATalpha WT pool
#     consisted of 200 WT-MATalpha replicates (ArrayExpress accession E-TABM-773, GEO
#     accession GSE42215). For MATa mutants grown in Erlenmeyers, the WT pool
#     consisted of 8 WT MATa replicates (ArrayExpress accession E-MTAB-1352, GEO
#     accession GSE42240)."
# Each WT array is one WT culture against the same refpool (GSE42215 summary: "by
# randomly taking 100 wt vs. refpool (pooled wts) and 100 refpool vs. wt
# hybridizations"; GSE42240 / GSE42241 sample description: "Growth curves were made for
# the mutant cultures (two cultures from two isolates) as well as for two wildtype (wt)
# inoculates, grown in parallel."). 428 arrays (200 + 200 + 8 + 20), stored with an
# empty genotype on the strain of their series. GSE42240's characteristics read
# "strain: BY4742" beside "genotype: wt-matA" on all 8 arrays; BY4742 is the MATalpha
# strain, and the paper, the series summary ("wildtype matA") and the mating-type genes
# (above) all say MATa, so the series is stored as BY4741. (The strain characteristic is
# per series, not per channel: GSE42241 gives "strain: BY4741" to its refpool channel
# too, while the marker genes show the refpool is MATalpha.) Before 2026.11.24 the four
# series were downloaded and never read.
WT_SERIES_STRAIN: dict[str, str] = {
    "GSE42215": "BY4742",  # MATalpha, Erlenmeyer, 200 arrays
    "GSE42217": "BY4742",  # MATalpha, Tecan plate, 200 arrays
    "GSE42240": "BY4741",  # MATa, Erlenmeyer, 8 arrays
    "GSE42241": "BY4741",  # MATa, Tecan plate, 20 arrays
}

_SINGLE_CULTURE_NOTE = (
    "a record is ONE GEO array, the hybridization of one culture's RNA against the "
    "common reference pool, and one culture has no dispersion of its own. The gene's "
    "two spots on the array (four for 12 genes on GPL11232; paper: 'Each gene probe is "
    "spotted twice on the array') are technical and are combined inside the record as "
    "the mean of their released log2 ratios, so n_replicates counts the one culture. "
    "The mean and dispersion over a mutant's arrays are a modeling-time derivation"
)
_REFERENCE_READING_NOTE = (
    "the reference is the common reference pool's signal on this record's own array "
    "(the mean of the gene's spots), one reading of one RNA batch, so there is no "
    "dispersion to store"
)


def _single_measurement_gaps(note: str) -> list[ProvenanceGap]:
    """The two uncertainty fields of a one-array phenotype, as typed absences."""
    return [
        ProvenanceGap(
            field=field,
            reason=ProvenanceGapReason.not_applicable_single_measurement,
            note=note,
        )
        for field in ("expression_log2_ratio_se", "expression_log2_ratio_variance")
    ]


# ============================================================================
# Gene names in GEO titles: the release's own crosswalk, plus one pinned residue
# ============================================================================
# A deletion array's GEO title names the deleted gene by a common name ("bck1-del-1-a",
# "[HS1991] cup9-del-a"). The release's own crosswalk is Table S1
# (kemmeren2014_table_s1.xlsx: 1,484 rows, columns ``gene``, ``orf name``, ``mating
# type``), and a title token is resolved ONLY through it: a token that is a Table S1
# ``gene`` takes that row's ``orf name``, and a token that IS a Table S1 ``orf name``
# is taken verbatim. No genome, alias table or reconciler is consulted (the genome is a
# separate layer and gates nothing).
#
# The residue below is the one remaining crosswalk. Measured 2026-10-12 over the 2,633
# deletion arrays of GSE42527 + GSE42526, 73 arrays carry a title token the release
# pairs with no ORF: 69 arrays of 36 names that Table S1 spells differently (GEO
# ``abc1-del``, Table S1 ``COQ8`` / YGL119W; GEO's sample characteristics carry only the
# same ``-del`` token, and only 12 of the 69 can be joined to Table S1 through a slide
# number in the supplementary-file name), and 4 arrays of TLC1 and CMS1, whose Table S1
# ``orf name`` cell is the common name itself. Each maps to the ORF the KG 3.0 build
# served it under, read from that build's preprocess/data.csv (``title``,
# ``systematic_gene_name``), never from the genome. Every ORF is a Table S1 ``orf name``
# (so the mating-type row joins) except YNCB0010W and YLR003C, which join through the
# TLC1 and CMS1 rows. This table is pending group 1's released-identifier leaf (a
# GenePerturbation child carrying the released symbol with ``systematic_gene_name`` as
# a typed gap); once it lands, these 38 names are stored verbatim and the table goes.
RELEASE_TITLE_NAMES_WITHOUT_AN_ORF: dict[str, str] = {
    "ABC1": "YGL119W",
    "ATG24": "YJL036W",
    "ATG31": "YDR022C",
    "ATG6": "YPL120W",
    "CAC1": "YPR018W",
    "CAF17": "YJR122W",
    "CAP": "YNL138W",
    "CDK8": "YPL042C",
    "CHF1": "YHR115C",
    "CHF2": "YNL116W",
    "CMS1": "YLR003C",
    "CSN12": "YJR084W",
    "CTH2": "YLR136C",
    "CYCC": "YNL025C",
    "FMP38": "YOR205C",
    "HPA1": "YPL086C",
    "HPR5": "YJL092W",
    "HSN1": "YHR127W",
    "KEM1": "YGL173C",
    "KRH1": "YAL056W",
    "KRH2": "YOR371C",
    "LUG1": "YCR087C-A",
    "MED12": "YCR081W",
    "MED13": "YDR443C",
    "MED15": "YOL051W",
    "MED16": "YNL236W",
    "MED18": "YGR104C",
    "MED20": "YHR041C",
    "MED3": "YGL025C",
    "MED31": "YGL127C",
    "MED5": "YGL151W",
    "MED9": "YNR010W",
    "NOT4": "YER068W",
    "RIS1": "YOR191W",
    "SET7": "YDR257C",
    "SHA3": "YPL026C",
    "SSN6": "YBR112C",
    "TLC1": "YNCB0010W",
}


@register_dataset
class MicroarrayKemmeren2014Dataset(ExperimentDataset):
    """Deletion-mutant and wildtype microarray expression from Kemmeren et al. 2014.

    One record per GEO array, that is per culture: 2,633 deletion arrays of the
    responsive (GSE42527) and non-responsive (GSE42526) series, then 428 wildtype
    arrays of the four WT series (GSE42215, GSE42217, GSE42240, GSE42241), 3,061 in
    all. A record's ``expression_log2_ratio`` is that array's log2(culture / common
    reference pool), its gene's duplicate spots combined, with ``n_replicates`` 1,
    ``replicate_id`` the array's GEO sample title, and no dispersion (typed
    ``not_applicable_single_measurement`` gaps). Its reference is the refpool signal on
    the same array. See the module comment for the design quotes.

    Each deletion is keyed by the ORF the release pairs with its GEO title token
    (Table S1, or the pinned ``RELEASE_TITLE_NAMES_WITHOUT_AN_ORF`` residue), and
    ``perturbed_gene_name`` is the release's own name for it; no genome is consulted. A
    wildtype record has an empty genotype on its series' strain (``WT_SERIES_STRAIN``).

    Data source: GEO accessions GSE42527, GSE42526 and the four WT series.
    Paper: Kemmeren et al. (2014) Cell.
    """

    organism = Organism.scerevisiae

    # GEO accessions for the dataset
    geo_accession_responsive = "GSE42527"  # Responsive mutants
    geo_accession_nonresponsive = "GSE42526"  # Non-responsive mutants

    # GEO accessions for wildtype reference datasets
    geo_accession_wt_mata_tecan = "GSE42241"  # MATa, Tecan plate, 20 replicates
    geo_accession_wt_mata_flask = "GSE42240"  # MATa, Erlenmeyer flask, 8 replicates
    geo_accession_wt_matalpha_tecan = (
        "GSE42217"  # MATalpha, Tecan plate, 200 replicates
    )
    geo_accession_wt_matalpha_flask = (
        "GSE42215"  # MATalpha, Erlenmeyer flask, 200 replicates
    )

    def __init__(
        self,
        root: str = "data/torchcell/microarray_kemmeren2014",
        io_workers: int = 0,
        process_workers: int = 0,  # For parallel processing of experiments
        batch_size: int = 10,  # Batch size for parallel processing
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset; gene names come from the release's Table S1."""
        self.process_workers = process_workers
        self.batch_size = batch_size

        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment model produced by this dataset."""
        return MicroarrayExpressionExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference model produced by this dataset."""
        return MicroarrayExpressionExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Raw GEO SOFT family files required before processing."""
        return [
            f"{self.geo_accession_responsive}_family.soft.gz",
            f"{self.geo_accession_nonresponsive}_family.soft.gz",
            f"{self.geo_accession_wt_mata_tecan}_family.soft.gz",
            f"{self.geo_accession_wt_mata_flask}_family.soft.gz",
            f"{self.geo_accession_wt_matalpha_tecan}_family.soft.gz",
            f"{self.geo_accession_wt_matalpha_flask}_family.soft.gz",
        ]

    def download(self) -> None:
        """Download supplementary Table S1 and expression data from GEO."""
        # First, download supplementary Table S1 with mating type information
        table_path = osp.join(self.raw_dir, "kemmeren2014_table_s1.xlsx")
        if not osp.exists(table_path):
            log.info(
                "Downloading supplementary Table S1 with mating type information..."
            )
            # Using Box shared link for the Excel file
            url = "https://uofi.box.com/shared/static/9n6ruj58ueup0cebhnek8ijdcy4om0bi"

            try:
                response = httpx.get(url, timeout=60, follow_redirects=True)
                response.raise_for_status()

                # Check if we got actual Excel content
                if len(response.content) < 1000:  # Excel files should be larger
                    raise ValueError(
                        f"Downloaded file too small ({len(response.content)} bytes), likely not the Excel file"
                    )

                with open(table_path, "wb") as f:
                    f.write(response.content)

                log.info(
                    f"Successfully downloaded Table S1 to {table_path} ({len(response.content)} bytes)"
                )

            except Exception as e:
                log.error(f"Failed to download Table S1: {e}")
                raise RuntimeError(
                    f"Failed to download Table S1 from {url}\n"
                    f"Error: {e}\n"
                    f"Please check the URL or save manually as: {table_path}"
                )

        # Then download both GEO datasets
        for geo_accession in [
            self.geo_accession_responsive,
            self.geo_accession_nonresponsive,
        ]:
            log.info(f"Downloading GEO dataset {geo_accession}...")

            try:
                gse = GEOparse.get_GEO(
                    geo=geo_accession, destdir=self.raw_dir, silent=False
                )
                log.info(f"Successfully downloaded {geo_accession}")

                # Save the parsed GEO object for use in process()
                geo_pkl_path = osp.join(self.raw_dir, f"{geo_accession}.pkl")
                with open(geo_pkl_path, "wb") as f:
                    pickle.dump(gse, f)

            except Exception as e:
                log.error(f"Failed to download GEO data: {e}")
                raise RuntimeError(f"Failed to download {geo_accession} from GEO")

        # Download wildtype reference datasets
        log.info("Downloading wildtype reference datasets...")
        for geo_accession in [
            self.geo_accession_wt_mata_tecan,
            self.geo_accession_wt_mata_flask,
            self.geo_accession_wt_matalpha_tecan,
            self.geo_accession_wt_matalpha_flask,
        ]:
            log.info(f"Downloading WT dataset {geo_accession}...")

            try:
                gse = GEOparse.get_GEO(
                    geo=geo_accession, destdir=self.raw_dir, silent=False
                )
                log.info(f"Successfully downloaded {geo_accession}")

                # Save the parsed GEO object
                geo_pkl_path = osp.join(self.raw_dir, f"{geo_accession}.pkl")
                with open(geo_pkl_path, "wb") as f:
                    pickle.dump(gse, f)

            except Exception as e:
                log.error(f"Failed to download WT data: {e}")
                raise RuntimeError(f"Failed to download {geo_accession} from GEO")

    def _load_series(self, geo_accession: str) -> Any:
        """The parsed GEO series: its ``raw/<accession>.pkl``, else ``get_GEO``."""
        geo_pkl_path = osp.join(self.raw_dir, f"{geo_accession}.pkl")
        if osp.exists(geo_pkl_path):
            with open(geo_pkl_path, "rb") as f:
                return pickle.load(f)
        return GEOparse.get_GEO(geo=geo_accession, destdir=self.raw_dir, silent=False)

    @staticmethod
    def _deletion_token(gsm: Any) -> str | None:
        """The deleted gene's name as the array's title or characteristics print it.

        The title's token before ``-del`` (an ``[HS1991]``-style prefix stripped);
        when the title names no deletion, the first ``genotype/variation:`` value that
        does, in either channel. ``None`` when neither names one.
        """
        title = str(gsm.metadata.get("title", [""])[0])
        if "-del" in title.lower():
            gene_part = title.split("-del")[0].strip()
            return gene_part.split("]")[-1].strip().upper()
        for channel in ("characteristics_ch1", "characteristics_ch2"):
            for char in gsm.metadata.get(channel, []):
                if "genotype/variation:" not in char:
                    continue
                genotype = str(char).split("genotype/variation:")[-1].strip()
                if "-del" in genotype:
                    gene_part = genotype.replace("-del", "").strip()
                    return gene_part.split("]")[-1].strip().upper()
        return None

    @post_process
    def process(self) -> None:
        """Write one experiment per GEO array: deletion arrays, then WT arrays."""
        # Resolution census: every deletion array resolves through exactly one route
        self.resolved_by_table_s1_gene = 0
        self.resolved_by_table_s1_orf = 0
        self.resolved_by_release_title_names = 0
        # ORF -> the release's own name for the deleted gene (perturbed_gene_name)
        self.released_gene_names: dict[str, str] = {}

        systematic_to_strain, common_to_systematic = self._load_mating_type_map()

        probe_to_gene_map: dict[str, str] = {}
        deletion_arrays: dict[str, list[tuple[str, Any]]] = {}
        for geo_accession in (
            self.geo_accession_responsive,
            self.geo_accession_nonresponsive,
        ):
            gse = self._load_series(geo_accession)
            if not probe_to_gene_map and hasattr(gse, "gpls"):
                probe_to_gene_map = self._extract_probe_to_gene_mapping(gse)
            for gsm in gse.gsms.values():
                token = self._deletion_token(gsm)
                if token is None:
                    raise ValueError(
                        f"{gsm.name} of {geo_accession} names no deletion; every array "
                        "of the two deletion series is a deletion culture"
                    )
                orf = self.resolve_gene_name_comprehensive(
                    token, common_to_systematic, systematic_to_strain
                )
                deletion_arrays.setdefault(orf, []).append((geo_accession, gsm))

        n_deletion_arrays = sum(len(arrays) for arrays in deletion_arrays.values())
        log.info(
            f"{n_deletion_arrays} deletion arrays of {len(deletion_arrays)} deletion "
            f"mutants; {len(probe_to_gene_map)} probes map to a gene"
        )
        log.info(
            "Resolved by Table S1 gene / Table S1 orf name / "
            "RELEASE_TITLE_NAMES_WITHOUT_AN_ORF: "
            f"{self.resolved_by_table_s1_gene} / {self.resolved_by_table_s1_orf} / "
            f"{self.resolved_by_release_title_names}"
        )

        # Check the channel assignment on the deleted genes' own probes
        self._validate_channel_assignment(
            {
                orf: [gsm for _, gsm in arrays]
                for orf, arrays in deletion_arrays.items()
            },
            probe_to_gene_map,
        )

        units: list[tuple[dict[str, Any], Any]] = []
        for orf, arrays in deletion_arrays.items():
            if orf not in systematic_to_strain:
                raise ValueError(
                    f"{orf} has no Table S1 mating type, so its strain (BY4741 or "
                    "BY4742) cannot be stated"
                )
            for geo_accession, gsm in arrays:
                units.append(
                    (
                        {
                            "series": geo_accession,
                            "geo_accession": gsm.name,
                            "replicate_id": gsm.metadata["title"][0],
                            "systematic_gene_name": orf,
                            "perturbed_gene_name": self.released_gene_names[orf],
                            "strain": systematic_to_strain[orf],
                        },
                        gsm,
                    )
                )

        for geo_accession, strain in WT_SERIES_STRAIN.items():
            gse = self._load_series(geo_accession)
            for gsm in gse.gsms.values():
                title = gsm.metadata["title"][0]
                test_name = self._test_source_name(gsm)
                if "-del" in title.lower() or "-del" in test_name.lower():
                    raise ValueError(
                        f"{gsm.name} of WT series {geo_accession} names a deletion"
                    )
                if ("mata" in test_name.lower()) != (strain == "BY4741"):
                    raise ValueError(
                        f"{gsm.name} of {geo_accession}: culture {test_name!r} does not "
                        f"match the series strain {strain}"
                    )
                units.append(
                    (
                        {
                            "series": geo_accession,
                            "geo_accession": gsm.name,
                            "replicate_id": title,
                            "systematic_gene_name": None,
                            "perturbed_gene_name": None,
                            "strain": strain,
                        },
                        gsm,
                    )
                )
        log.info(f"{len(units) - n_deletion_arrays} wildtype arrays")

        if self.process_workers > 0:
            blobs = self._process_parallel(units, probe_to_gene_map)
        else:
            blobs = self._build_records(units, probe_to_gene_map, self.name)
        self._write_lmdb(blobs)

        os.makedirs(self.preprocess_dir, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "record_index": index,
                    "series": info["series"],
                    "geo_accession": info["geo_accession"],
                    "replicate_id": info["replicate_id"],
                    "systematic_gene_name": info["systematic_gene_name"] or "",
                    "perturbed_gene_name": info["perturbed_gene_name"] or "",
                    "strain": info["strain"],
                    "reference_dye": self._reference_dye(gsm),
                }
                for index, (info, gsm) in enumerate(units)
            ]
        ).to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

    def _process_parallel(
        self, units: list[tuple[dict[str, Any], Any]], probe_to_gene_map: dict[str, str]
    ) -> list[bytes]:
        """``_build_records`` over batches of arrays in a ProcessPoolExecutor."""
        batches = [
            units[i : i + self.batch_size]
            for i in range(0, len(units), self.batch_size)
        ]
        log.info(f"Created {len(batches)} batches of size {self.batch_size}")
        blobs: list[bytes] = []
        with ProcessPoolExecutor(max_workers=self.process_workers) as executor:
            futures = [
                executor.submit(
                    self._build_records, batch, probe_to_gene_map, self.name
                )
                for batch in batches
            ]
            for future in tqdm(futures, desc="Processing batches"):
                blobs.extend(future.result())
        return blobs

    def _write_lmdb(self, blobs: list[bytes]) -> None:
        """Write the serialized records to ``processed/lmdb`` under keys 0..N-1."""
        env = lmdb.open(
            osp.join(self.processed_dir, "lmdb"),
            map_size=int(5e12),  # 5TB for expression data
        )
        with env.begin(write=True) as txn:
            for idx, blob in enumerate(blobs):
                txn.put(f"{idx}".encode(), blob)
        env.close()
        log.info(f"Wrote {len(blobs)} experiments to LMDB")

    @staticmethod
    def _build_records(
        units: list[tuple[dict[str, Any], Any]],
        probe_to_gene_map: dict[str, str],
        dataset_name: str,
    ) -> list[bytes]:
        """One serialized record per array, in order. Static for multiprocessing."""
        blobs = []
        for sample_info, gsm in units:
            signals = MicroarrayKemmeren2014Dataset._array_signals(
                gsm, probe_to_gene_map
            )
            experiment, reference, publication = (
                MicroarrayKemmeren2014Dataset.create_array_experiment(
                    dataset_name, sample_info, signals
                )
            )
            blobs.append(
                pickle.dumps(
                    {
                        "experiment": experiment.model_dump(),
                        "reference": reference.model_dump(),
                        "publication": publication.model_dump(),
                    }
                )
            )
        return blobs

    @staticmethod
    def _test_source_name(gsm: Any) -> str:
        """The ``source_name`` of the channel that is not the reference pool."""
        test_column, _ = MicroarrayKemmeren2014Dataset._channel_columns(gsm)
        dye = test_column.removeprefix("Signal Norm_")
        channel = 1 if gsm.metadata.get("label_ch1", [""])[0] == dye else 2
        return str(gsm.metadata.get(f"source_name_ch{channel}", [""])[0])

    @staticmethod
    def _reference_dye(gsm: Any) -> str:
        """The dye carrying the reference pool on this array (Cy5 or Cy3)."""
        _, reference_column = MicroarrayKemmeren2014Dataset._channel_columns(gsm)
        return reference_column.removeprefix("Signal Norm_")

    @staticmethod
    def _channel_columns(gsm: Any) -> tuple[str, str]:
        """Signal columns of the test culture and of the common reference pool, read
        from GEO's own channel metadata.

        Every array of the six series names the reference pool in the ``source_name``
        of exactly one channel ("refpool", or "ref1" on 193 GSE42217 arrays) and gives
        that channel's dye in ``label_ch1`` / ``label_ch2`` (Cy5 for ch1 and Cy3 for ch2
        throughout). Measured 2026-09-28 on the raw SOFT files: 3061 of 3061 arrays.
        The sample title is not consulted: on the "-a" arrays the reference pool is in
        Cy5, the reverse of what the title suffix was once taken to mean, and the
        deleted gene's own probes confirm GEO's labels on 702 of 705 arrays.

        Returns:
            (test_column, reference_column), for example
            ("Signal Norm_Cy3", "Signal Norm_Cy5") on a "-a" array.
        """
        metadata = gsm.metadata
        names = [
            metadata.get("source_name_ch1", [""])[0],
            metadata.get("source_name_ch2", [""])[0],
        ]
        labels = [
            metadata.get("label_ch1", [""])[0],
            metadata.get("label_ch2", [""])[0],
        ]
        is_reference = [
            name.strip().lower().startswith("ref") and "-del" not in name.lower()
            for name in names
        ]
        if sum(is_reference) != 1:
            raise ValueError(
                f"{gsm.name}: expected the reference pool in exactly one channel, "
                f"source names {names}"
            )
        reference = is_reference.index(True)
        reference_dye, test_dye = labels[reference], labels[1 - reference]
        if {reference_dye, test_dye} != {"Cy5", "Cy3"}:
            raise ValueError(
                f"{gsm.name}: channel labels {labels} are not one Cy5 and one Cy3"
            )
        return f"Signal Norm_{test_dye}", f"Signal Norm_{reference_dye}"

    @staticmethod
    def _array_signals(gsm: Any, probe_to_gene_map: dict[str, str]) -> pd.DataFrame:
        """Per-gene values of one array, its duplicate spots combined.

        Each mapped spot has two ``Signal Norm`` values, the test culture's and the
        reference pool's, assigned by ``_channel_columns``; its log2 ratio is
        log2(test / reference). The GEO ``VALUE`` column is not read: its orientation
        differs between arrays (module comment). A spot whose signal is not positive
        (or missing) raises, so no spot is dropped. Rows whose probe is not in the map
        are not genes and are left out.

        Returns:
            A frame indexed by gene (sorted) with ``log2_ratio``, ``test`` and
            ``reference`` (the means over the gene's spots) and ``n_spots``.
        """
        test_column, reference_column = MicroarrayKemmeren2014Dataset._channel_columns(
            gsm
        )
        table = gsm.table
        for column in ("ID_REF", test_column, reference_column):
            if column not in table.columns:
                raise ValueError(
                    f"{gsm.name}: column {column!r} missing from the sample table"
                )
        probe = table["ID_REF"].astype(int).astype(str)
        mapped = probe.isin(set(probe_to_gene_map)).to_numpy()
        test = table[test_column][mapped].astype(float).to_numpy()
        reference = table[reference_column][mapped].astype(float).to_numpy()
        not_positive = int((~((test > 0) & (reference > 0))).sum())
        if not_positive:
            raise ValueError(
                f"{gsm.name}: {not_positive} gene spots carry a signal that is not "
                "positive"
            )
        spots = pd.DataFrame(
            {
                "gene": probe[mapped].map(probe_to_gene_map).to_numpy(),
                "log2_ratio": np.log2(test / reference),
                "test": test,
                "reference": reference,
            }
        )
        return spots.groupby("gene", sort=True).agg(
            log2_ratio=("log2_ratio", "mean"),
            test=("test", "mean"),
            reference=("reference", "mean"),
            n_spots=("log2_ratio", "size"),
        )

    def _extract_probe_to_gene_mapping(self, gse: Any) -> dict[str, str]:
        """Extract probe ID to gene name mapping from GEO platform annotation."""
        probe_to_gene: dict[str, str] = {}

        # Check if platform data is available
        if not hasattr(gse, "gpls") or not gse.gpls:
            log.warning("No platform annotation found in GEO dataset")
            return probe_to_gene

        # Get the first platform (usually there's only one)
        for gpl_name, gpl in gse.gpls.items():
            log.info(f"Processing platform {gpl_name}")

            if hasattr(gpl, "table") and gpl.table is not None:
                table = gpl.table

                # Look for gene symbol columns
                gene_columns = [
                    "ORF",
                    "Gene",
                    "Gene Symbol",
                    "Gene_Symbol",
                    "GENE_SYMBOL",
                    "gene_symbol",
                    "SystematicName",
                    "Systematic_Name",
                    "SYSTEMATIC_NAME",
                ]

                id_column = None
                gene_column = None

                # Find ID and gene columns
                if "ID" in table.columns:
                    id_column = "ID"
                elif "SPOT" in table.columns:
                    id_column = "SPOT"

                for col in gene_columns:
                    if col in table.columns:
                        gene_column = col
                        break

                if id_column and gene_column:
                    for _, row in table.iterrows():
                        probe_id = str(
                            int(row[id_column])
                        )  # Convert to int first to avoid '1.0'
                        gene_name = str(row[gene_column])

                        # Clean and validate gene name
                        if gene_name and gene_name != "nan" and gene_name != "":
                            # Use gene name directly from platform annotation
                            # Platform should already have systematic names
                            gene_name_upper = gene_name.upper()
                            if re.match(r"Y[A-Z]{2}\d{3}[CW]", gene_name_upper):
                                # Already a systematic name
                                probe_to_gene[probe_id] = gene_name_upper
                            elif re.match(r"Q\d{4}", gene_name_upper):
                                # Mitochondrial genes are already systematic
                                probe_to_gene[probe_id] = gene_name_upper
                            else:
                                # Keep the gene name as is (might be common name)
                                # The expression averaging will handle mismatches
                                probe_to_gene[probe_id] = gene_name_upper

                    log.info(
                        f"Extracted {len(probe_to_gene)} probe-to-gene mappings from {gpl_name}"
                    )
                else:
                    log.warning(
                        f"Could not find appropriate columns in platform {gpl_name}"
                    )
                    log.info(f"Available columns: {list(table.columns)}")

        return probe_to_gene

    def _load_mating_type_map(self) -> tuple[dict[str, str], dict[str, str]]:
        """Load mating type information from supplementary Table S1.

        Returns:
            Tuple of:
            - Dictionary mapping each ``orf name`` cell as released (upper-cased; TLC1
              and CMS1 are common names and stay so) to its strain (BY4741 or BY4742)
            - Dictionary mapping each Table S1 ``gene`` to that row's ``orf name``
        """
        systematic_to_strain = {}
        common_to_systematic = {}

        # Path to the supplementary table
        table_path = osp.join(self.raw_dir, "kemmeren2014_table_s1.xlsx")

        # Check if the file exists - REQUIRED
        if not osp.exists(table_path):
            raise FileNotFoundError(
                f"Supplementary Table S1 not found at {table_path}\n"
                f"Please download it from: https://www.cell.com/cms/10.1016/j.cell.2014.02.054/attachment/7b6014f0-a526-4f16-ae4a-cd04fd03efce/mmc1.xlsx\n"
                f"And save it as: {table_path}"
            )

        try:
            # Read the Excel file
            df = pd.read_excel(table_path, sheet_name=0)  # First sheet
            log.info(f"Loaded Excel file with {len(df)} rows")
            log.info(
                f"Columns in Excel file: {list(df.columns)[:10]}"
            )  # Show first 10 columns

            # Look for columns containing gene names and mating type
            # MUST use "orf name" column which contains systematic names
            orf_col = None
            gene_col = None  # Common gene name column for validation
            mating_col = None

            # Find the orf name column (contains systematic names like YJL095W)
            for col in df.columns:
                col_lower = col.lower()
                if "orf" in col_lower and "name" in col_lower:
                    orf_col = col
                    break

            # Also find gene column for validation
            for col in df.columns:
                if col.lower() == "gene":
                    gene_col = col
                    break

            # Find the mating type column
            for col in df.columns:
                col_lower = col.lower()
                if "mating" in col_lower and "type" in col_lower:
                    mating_col = col
                    break

            if not orf_col:
                raise ValueError("Required 'orf name' column not found in Excel file!")

            if orf_col and mating_col:
                log.info(
                    f"Using columns: {orf_col} for systematic names, {mating_col} for mating type"
                )
                if gene_col:
                    log.info(f"Also found {gene_col} column for validation")

                # Debug: Track what's in the Excel
                excel_genes_original = []
                excel_genes_converted = []
                duplicate_systematics = (
                    set()
                )  # Track orf names with multiple gene names

                for _, row in df.iterrows():
                    systematic_name = row[orf_col]  # Use orf name directly
                    mating_type = row[mating_col]
                    common_name = row[gene_col] if gene_col else None

                    if pd.notna(systematic_name) and pd.notna(mating_type):
                        # The orf name cell as released (upper-cased; two cells,
                        # TLC1 and CMS1, are common names and stay as released)
                        systematic_name = str(systematic_name).upper()
                        excel_genes_original.append(systematic_name)
                        excel_genes_converted.append(systematic_name)

                        # No genome validation needed - Excel is authoritative

                        # Check for duplicates
                        if systematic_name in systematic_to_strain:
                            duplicate_systematics.add(systematic_name)
                            if common_name:
                                log.debug(
                                    f"Duplicate orf {systematic_name}: adding alias {common_name}"
                                )

                        # Map mating type to strain
                        mating_str = str(mating_type).upper()
                        if (
                            "MATALPHA" in mating_str
                            or "MATΑ" in mating_str
                            or "MAT ALPHA" in mating_str
                        ):
                            strain = "BY4742"
                        elif "MATA" in mating_str and "ALPHA" not in mating_str:
                            strain = "BY4741"
                        else:
                            log.warning(
                                f"Unknown mating type '{mating_type}' for gene {systematic_name}"
                            )
                            continue

                        systematic_to_strain[systematic_name] = strain

                        # Also create common name to systematic mapping
                        if common_name and pd.notna(common_name):
                            common_name_upper = str(common_name).upper()
                            common_to_systematic[common_name_upper] = systematic_name

                # Summary output only
                # Count strains
                by4741_count = sum(
                    1 for v in systematic_to_strain.values() if v == "BY4741"
                )
                by4742_count = sum(
                    1 for v in systematic_to_strain.values() if v == "BY4742"
                )
                log.info(f"Loaded mating type for {len(systematic_to_strain)} genes")
                log.info(f"Loaded {len(common_to_systematic)} common name mappings")
                log.info(
                    f"BY4741 (MATa): {by4741_count}, BY4742 (MATalpha): {by4742_count}"
                )

                if duplicate_systematics:
                    log.info(
                        f"Found {len(duplicate_systematics)} orf names with multiple gene names in Excel"
                    )
            else:
                if not orf_col:
                    log.error(
                        f"Missing required 'orf name' column. Available: {list(df.columns)}"
                    )
                if not mating_col:
                    log.error(
                        f"Missing mating type column. Available: {list(df.columns)}"
                    )

        except Exception as e:
            log.error(f"Failed to load mating type map: {e}")

        return systematic_to_strain, common_to_systematic

    def resolve_gene_name_comprehensive(
        self,
        gene_name: str,
        common_to_systematic: dict[str, str],
        systematic_to_strain: dict[str, str],
        already_assigned: set[str] | None = None,
    ) -> str:
        """The ORF a GEO title token names, through the release only.

        Routes, in order, each counted:

        1. ``RELEASE_TITLE_NAMES_WITHOUT_AN_ORF``, the pinned residue of 38 tokens the
           release pairs with no ORF (see the module comment). When the ORF has no
           Table S1 row of its own but the token does (TLC1, CMS1, whose ``orf name``
           cell is the common name), the token's mating-type row is joined to the ORF
           in ``systematic_to_strain``.
        2. A Table S1 ``gene`` -> that row's ``orf name``.
        3. A token that IS a Table S1 ``orf name``, taken verbatim.

        The release's own name for the deleted gene (the title token for routes 1 and
        2, the row's Table S1 ``gene`` for route 3, else the ORF) is recorded in
        ``released_gene_names`` and becomes ``perturbed_gene_name``. A token on none of
        the routes raises ``ValueError``: a deletion array the release ships is never
        skipped silently. ``already_assigned`` is accepted and not read.
        """
        gene_upper = gene_name.upper()
        released_name: str
        if gene_upper in RELEASE_TITLE_NAMES_WITHOUT_AN_ORF:
            systematic = RELEASE_TITLE_NAMES_WITHOUT_AN_ORF[gene_upper]
            if (
                systematic not in systematic_to_strain
                and gene_upper in systematic_to_strain
            ):
                systematic_to_strain[systematic] = systematic_to_strain[gene_upper]
            self.resolved_by_release_title_names += 1
            released_name = gene_upper
        elif gene_upper in common_to_systematic:
            systematic = common_to_systematic[gene_upper]
            self.resolved_by_table_s1_gene += 1
            released_name = gene_upper
        elif gene_upper in systematic_to_strain:
            systematic = gene_upper
            self.resolved_by_table_s1_orf += 1
            row_genes = [g for g, o in common_to_systematic.items() if o == systematic]
            released_name = row_genes[0] if row_genes else systematic
        else:
            raise ValueError(
                f"Kemmeren 2014 title token {gene_name!r} is neither a Table S1 gene, "
                "a Table S1 orf name, nor in RELEASE_TITLE_NAMES_WITHOUT_AN_ORF"
            )
        recorded = self.released_gene_names.setdefault(systematic, released_name)
        if recorded != released_name:
            raise ValueError(
                f"{systematic} is named both {recorded} and {released_name} by the release"
            )
        return systematic

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocess raw data - for Kemmeren this is handled in process()."""
        return df

    def create_experiment(self) -> None:
        """Required by base class but not used - see create_array_experiment."""
        pass

    @staticmethod
    def _validate_channel_assignment(
        deletion_samples_by_gene: dict[str, list[Any]],
        probe_to_gene_map: dict[str, str],
    ) -> float:
        """Fraction of deletion arrays on which the deleted gene reads lower in the
        deletion channel than in the reference channel.

        A deleted gene's transcript is absent from the deletion culture, so with the
        channels assigned correctly the median log2(deletion / refpool) over that
        gene's probes is negative; a swapped assignment makes it positive. Every array
        whose deleted gene has a probe on the platform is checked. On the real data
        (2026-09-28) 702 of 705 such arrays are negative under GEO's channel labels
        and 10 of 705 under the former title rule. Returns NaN when no array can be
        checked; warns below 0.9.
        """
        probes_by_gene: dict[str, list[str]] = {}
        for probe_id, gene in probe_to_gene_map.items():
            probes_by_gene.setdefault(gene, []).append(probe_id)

        checked = 0
        negative = 0
        for gene_name, gsm_list in deletion_samples_by_gene.items():
            if gene_name not in probes_by_gene:
                continue
            probe_ids = probes_by_gene[gene_name]
            for gsm in gsm_list:
                test_column, reference_column = (
                    MicroarrayKemmeren2014Dataset._channel_columns(gsm)
                )
                table = gsm.table
                rows = table[table["ID_REF"].astype(int).astype(str).isin(probe_ids)]
                deletion = rows[test_column].astype(float).to_numpy()
                refpool = rows[reference_column].astype(float).to_numpy()
                positive = (deletion > 0) & (refpool > 0)
                if not positive.any():
                    continue
                checked += 1
                ratio = np.median(np.log2(deletion[positive] / refpool[positive]))
                if ratio < 0:
                    negative += 1

        if checked == 0:
            log.warning(
                "Channel assignment check: no deletion array carries a probe for its "
                "own deleted gene"
            )
            return float("nan")
        fraction = negative / checked
        log.info(
            f"Channel assignment check: deleted gene lower in the deletion channel on "
            f"{negative} of {checked} arrays ({fraction:.3f})"
        )
        if fraction < 0.9:
            log.warning(
                "Channel assignment check: fewer than 90 percent of arrays show the "
                "deleted gene depleted; the channel metadata may be wrong"
            )
        return fraction

    @staticmethod
    def create_array_experiment(
        dataset_name: str, sample_info: dict[str, Any], signals: pd.DataFrame
    ) -> tuple[Any, Any, Any]:
        """Build the experiment, reference and publication of ONE array.

        ``signals`` is ``_array_signals`` of the array. The phenotype is the array's
        per-gene log2 ratio and linear culture signal, ``n_replicates`` 1 per gene, no
        dispersion (typed gaps) and ``replicate_id`` the array's GEO title; the
        reference is the refpool signal on the same array with log2 ratio 0. A
        ``systematic_gene_name`` of ``None`` is a wildtype culture: empty genotype.
        """
        if "strain" not in sample_info:
            raise ValueError(
                "Strain (BY4741 or BY4742) must be specified in sample_info"
            )
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain=sample_info["strain"],
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        )

        # ``perturbed_gene_name`` is the release's own name for the deleted gene
        # (``resolve_gene_name_comprehensive``); a wildtype culture deletes nothing
        perturbations: list[GenePerturbationType] = []
        if sample_info["systematic_gene_name"] is not None:
            perturbations.append(
                KanMxDeletionPerturbation(
                    systematic_gene_name=sample_info["systematic_gene_name"],
                    perturbed_gene_name=sample_info["perturbed_gene_name"],
                )
            )
        genotype = Genotype(perturbations=perturbations)

        environment = Environment(
            media=Media(
                name="SC", state="liquid", is_synthetic=True
            ),  # Kemmeren used SC medium
            temperature=Temperature(value=30),
        )

        genes = [str(gene) for gene in signals.index]
        ones = dict.fromkeys(genes, 1)
        phenotype = MicroarrayExpressionPhenotype(
            expression={g: float(v) for g, v in zip(genes, signals["test"])},
            expression_log2_ratio={
                g: float(v) for g, v in zip(genes, signals["log2_ratio"])
            },
            expression_log2_ratio_se=None,
            expression_log2_ratio_variance=None,
            n_replicates=ones,
            replicate_id=sample_info["replicate_id"],
            provenance_gaps=_single_measurement_gaps(_SINGLE_CULTURE_NOTE),
        )
        phenotype_reference = MicroarrayExpressionPhenotype(
            expression={g: float(v) for g, v in zip(genes, signals["reference"])},
            expression_log2_ratio=dict.fromkeys(genes, 0.0),
            expression_log2_ratio_se=None,
            expression_log2_ratio_variance=None,
            n_replicates=ones,
            provenance_gaps=_single_measurement_gaps(_REFERENCE_READING_NOTE),
        )

        reference = MicroarrayExpressionExperimentReference(
            dataset_name=dataset_name,
            genome_reference=genome_reference,
            environment_reference=environment.model_copy(),
            phenotype_reference=phenotype_reference,
        )
        experiment = MicroarrayExpressionExperiment(
            dataset_name=dataset_name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )
        publication = Publication(
            pubmed_id="24766815",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/24766815/",
            doi="10.1016/j.cell.2014.02.054",
            doi_url="https://doi.org/10.1016/j.cell.2014.02.054",
        )
        return experiment, reference, publication


if __name__ == "__main__":
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    dataset = MicroarrayKemmeren2014Dataset(
        root=osp.join(cast(str, DATA_ROOT), "data/torchcell/microarray_kemmeren2014"),
        io_workers=10,
        process_workers=10,
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset gene set size: {len(dataset.gene_set)}")
    record = dataset[0]
    experiment = record["experiment"]
    phenotype = experiment["phenotype"]
    print(f"Record 0: {phenotype['replicate_id']}, genotype {experiment['genotype']}")
    print(f"  {len(phenotype['expression_log2_ratio'])} genes measured")
