# torchcell/datasets/scerevisiae/kemmeren2014
# [[torchcell.datasets.scerevisiae.kemmeren2014]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/kemmeren2014
# Test file: tests/torchcell/datasets/scerevisiae/test_kemmeren2014.py
"""Kemmeren 2014 deletion-mutant microarray expression dataset."""

import logging
import os
import os.path as osp
import pickle
import re
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any, cast

import GEOparse
import lmdb
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sortedcontainers import SortedDict
from tqdm import tqdm

from torchcell.data import ExperimentDataset, Organism, post_process
from torchcell.datamodels.schema import (
    Environment,
    Experiment,
    ExperimentReference,
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

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ============================================================================
# Replicate Metadata - Extracted from Kemmeren et al. 2014
# ============================================================================

# CRITICAL: Replicate structure for deletion mutant expression measurements
# Quote: "Each mutant strain was grown twice, from two independently inoculated
#         cultures. Each culture was expression-profiled in technical replicate
#         to yield four measurements for each profiling mutant."
# Quote: "This included four replicates per responsive mutant, robotic procedures
#         optimized with external calibration controls, a common reference design
#         with wildtype (WT) reference RNA applied in dye-swap to each microarray"
# Source: Kemmeren et al. 2014, Cell, Expression Profiling section
# Paper: papers/kemmerenLargeScaleGeneticPerturbations2014/kemmerenLargeScaleGeneticPerturbations2014.mmd
# Date extracted: 2026-01-27
#
# Experimental design:
# - 2 biological replicates (two independently inoculated cultures)
# - Each culture profiled in technical replicate with the dyes swapped. GEO labels
#   ch1 = Cy5 and ch2 = Cy3 on every array of the six series, and source_name_ch1/ch2
#   say which channel holds the common reference pool ("refpool"; "ref1" on 193
#   GSE42217 arrays) and which the deletion or wildtype culture. On the "-a" arrays
#   the reference pool is in Cy5 and the deletion in Cy3; the "-b" arrays are the
#   swap. Measured 2026-09-28 on the raw SOFT files: 3061 of 3061 arrays name the
#   reference in exactly one channel, and the deleted gene's own probes read lower
#   in the deletion channel on 702 of 705 arrays that carry one.
# - Total: 4 measurements per gene per deletion mutant (2 biological x 2 dye orientations)
# - Common reference RNA (WT pool) used in dye-swap on each microarray
# - Applies to 1,484 deletion mutants (gene-specific regulators focus)
#
# NOTE: Actual n_replicates are COMPUTED from raw data (may be < 4 due to QC)
N_EXPECTED_BIOLOGICAL_REPLICATES = 2
N_EXPECTED_DYE_SWAP_TECHNICAL_REPLICATES = 2
N_EXPECTED_MAX_REPLICATES_DELETION = 4  # 2 biological × 2 dye-swap measurements

# Reference pool measurements
# Quote: "a common reference design was adopted with a batch of WT RNA applied
#         in dye-swap to one of the channels of each microarray"
# Quote: "Additional WT cultures were grown alongside batches of mutants on each
#         day and profiled in parallel to monitor batch effects"
# Source: Kemmeren et al. 2014, Cell, Expression Profiling section
# Date extracted: 2026-01-27
#
# Reference n_replicates (#484, decided 2026-10-02). The stored reference
# ``expression`` is the mean common-reference (refpool) signal over the mutant's own
# arrays (``create_expression_experiment``), so its ``n_replicates`` is the number of
# those arrays whose refpool value entered the mean: 1 or 2 per gene, the same arrays
# as the mutant's own count. It is NOT a count of wildtype arrays. The paper's WT
# pools are the comparison sets of its limma analysis, chosen per growth protocol,
# and none of them is the stored value. Paper (torchcell-library
# kemmerenLargeScaleGeneticPerturbations2014/paper.md, sha256
# 299eac4b07fc977a89c24ce190d05cfd182342a9f0d56a05e7e477d72e4ad08d, line 388):
#     "For the MATalpha mutants grown on the Tecan plate shaker, the WT pool consisted
#     of 200 MATalpha WTs (BY4742, ArrayExpress accession E-TABM-984, GEO accession
#     GSE42217). For the MATa mutants grown on the Tecan plate shaker, the WT pool
#     consisted of 20 WT-MATa replicates (BY4741, ArrayExpress accession E-MTAB-1351,
#     GEO accession GSE42241). For mutants grown in Erlenmeyers, the MATalpha WT pool
#     consisted of 200 WT-MATalpha replicates (ArrayExpress accession E-TABM-773, GEO
#     accession GSE42215). For MATa mutants grown in Erlenmeyers, the WT pool
#     consisted of 8 WT MATa replicates (ArrayExpress accession E-MTAB-1352, GEO
#     accession GSE42240)."
# The loader used to store the number of WT arrays of the two series of the mutant's
# mating type (400 for BY4742, 28 for BY4741), a count the paper never compares a
# mutant against. Nothing stored is computed from the four WT series any more, so
# process() no longer reads them; they stay in raw_file_names as the paper's named
# comparison pools.

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
# TLC1 and CMS1 rows. This table is pending group 1's released-gene-name leaf (a
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
    """Deletion-mutant microarray expression profiling from Kemmeren et al. 2014.

    Microarray gene expression data for single-gene deletion mutants profiled
    against a common wildtype reference pool in a dye-swap design. Combines the
    responsive and non-responsive mutant GEO series. The four wildtype series
    (BY4741 MATa, BY4742 MATalpha) are downloaded as the paper's named WT pools but
    no stored value is computed from them (#484).

    Each deletion is keyed by the ORF the release pairs with its GEO title token
    (Table S1, or the pinned ``RELEASE_TITLE_NAMES_WITHOUT_AN_ORF`` residue), and
    ``perturbed_gene_name`` is the release's own name for it; no genome is consulted.

    Data source: GEO accessions GSE42527, GSE42526, and wildtype references.
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
                response = requests.get(url, timeout=60)
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

    @post_process
    def process(self) -> None:
        """Parse GEO expression data into experiments and write the LMDB store."""
        # Resolution census: every deletion array resolves through exactly one route
        self.resolved_by_table_s1_gene = 0
        self.resolved_by_table_s1_orf = 0
        self.resolved_by_release_title_names = 0
        # ORF -> the release's own name for the deleted gene (perturbed_gene_name)
        self.released_gene_names: dict[str, str] = {}

        # Load mating type information from supplementary table
        systematic_to_strain, common_to_systematic = self._load_mating_type_map()

        # Load both GEO objects and extract platform annotation
        all_gsms = {}
        probe_to_gene_map: dict[str, str] = {}
        responsive_gsm_names = set()  # Track which samples are responsive mutants

        for geo_accession in [
            self.geo_accession_responsive,
            self.geo_accession_nonresponsive,
        ]:
            geo_pkl_path = osp.join(self.raw_dir, f"{geo_accession}.pkl")
            if osp.exists(geo_pkl_path):
                with open(geo_pkl_path, "rb") as f:
                    gse = pickle.load(f)
            else:
                # Re-download if pickle doesn't exist
                gse = GEOparse.get_GEO(
                    geo=geo_accession, destdir=self.raw_dir, silent=False
                )

            # Extract platform annotation for probe-to-gene mapping
            if not probe_to_gene_map and hasattr(gse, "gpls"):
                probe_to_gene_map = self._extract_probe_to_gene_mapping(gse)

            # Track responsive mutant GSM names
            if geo_accession == self.geo_accession_responsive:
                responsive_gsm_names.update(gse.gsms.keys())

            # Combine GSMs from both datasets
            all_gsms.update(gse.gsms)

        log.info(f"Processing {len(all_gsms)} GEO samples from both datasets...")
        log.info(f"Found {len(probe_to_gene_map)} probe-to-gene mappings")
        log.info(f"Loaded {len(systematic_to_strain)} systematic to strain mappings")
        log.info(f"Loaded {len(common_to_systematic)} common to systematic mappings")

        # Debug: Check for specific missing genes if needed
        # (Removed verbose YCR087C-A search to reduce output clutter)

        # Parse samples and extract metadata
        samples_data = []
        wt_samples = []
        deletion_samples_by_gene: dict[
            str, list[Any]
        ] = {}  # Group samples by gene for dye-swap averaging
        already_assigned: set[str] = set()  # Track assigned systematic names

        for sample_idx, (gsm_name, gsm) in enumerate(all_gsms.items()):
            # Extract metadata from characteristics
            sample_info = {
                "geo_accession": gsm_name,
                "title": gsm.metadata.get("title", [""])[0],
            }

            # Parse from title - look for gene-del pattern
            title = sample_info["title"]
            systematic_gene_name = None
            is_deletion = False
            is_wildtype = False
            gene_resolved_from_title = (
                False  # Track if we already tried resolution from title
            )

            # Check if it's a deletion mutant (contains -del)
            if "-del" in title.lower():
                is_deletion = True
                # Extract gene name before -del
                gene_part = title.split("-del")[0].strip()

                # Remove [HS1991] or similar prefixes if present
                if "]" in gene_part:
                    gene_part = gene_part.split("]")[-1].strip()

                common_name = gene_part.upper()

                # Convert to systematic name using comprehensive resolution
                systematic_gene_name = self.resolve_gene_name_comprehensive(
                    common_name,
                    common_to_systematic,
                    systematic_to_strain,
                    already_assigned,
                )
                gene_resolved_from_title = True  # Mark that we attempted resolution

            # Check the channel characteristics for confirmation. The deletion is in
            # ch2 on the "-a" arrays and in ch1 on the "-b" arrays (the dye swap), so
            # both channels are read; the other one names the reference pool.
            for channel in ("characteristics_ch1", "characteristics_ch2"):
                for char in gsm.metadata.get(channel, []):
                    if "genotype/variation:" not in char:
                        continue
                    genotype = char.split("genotype/variation:")[-1].strip()
                    if "-del" not in genotype:
                        continue
                    is_deletion = True
                    is_wildtype = False

                    # Only extract gene name if we haven't already tried from title
                    if not gene_resolved_from_title and not systematic_gene_name:
                        gene_part = genotype.replace("-del", "").strip()
                        # Remove [HS1991] or similar prefixes if present
                        if "]" in gene_part:
                            gene_part = gene_part.split("]")[-1].strip()
                        common_name = gene_part.upper()
                        systematic_gene_name = self.resolve_gene_name_comprehensive(
                            common_name,
                            common_to_systematic,
                            systematic_to_strain,
                            already_assigned,
                        )

            # Final check: Only mark as wildtype if no deletion was found
            # Most samples should be deletion mutants (with or without dye-swap)
            if not is_deletion and not systematic_gene_name:
                # This might be a true wildtype/control sample
                is_wildtype = True
            else:
                is_wildtype = False

            sample_info["systematic_gene_name"] = systematic_gene_name
            sample_info["is_deletion"] = is_deletion
            sample_info["is_wildtype"] = is_wildtype

            # Determine if this is a responsive mutant based on GEO accession
            sample_info["is_responsive_mutant"] = gsm_name in responsive_gsm_names

            sample_info["gsm_object"] = gsm

            if is_wildtype:
                wt_samples.append(gsm)
            elif is_deletion and systematic_gene_name:
                # Group deletion samples by gene for dye-swap averaging
                if systematic_gene_name not in deletion_samples_by_gene:
                    deletion_samples_by_gene[systematic_gene_name] = []
                    already_assigned.add(systematic_gene_name)  # Mark as assigned
                deletion_samples_by_gene[systematic_gene_name].append(gsm)

            samples_data.append(sample_info)

        log.info(f"Found {len(deletion_samples_by_gene)} unique gene deletions")
        log.info(f"Found {len(wt_samples)} wildtype reference samples")

        # Analyze sample distribution per gene
        replicate_counts = {}
        for gene, samples in deletion_samples_by_gene.items():
            count = len(samples)
            if count not in replicate_counts:
                replicate_counts[count] = 0
            replicate_counts[count] += 1

        log.info("\n=== Replicates per Gene Analysis ===")
        log.info(
            f"Average samples per gene: {2633 / len(deletion_samples_by_gene):.2f}"
        )
        for count in sorted(replicate_counts.keys()):
            log.info(f"Genes with {count} samples: {replicate_counts[count]}")
            # Print out genes with 4 samples to investigate YCR087C-A
            if count == 4:
                genes_with_4 = [
                    gene
                    for gene, samples in deletion_samples_by_gene.items()
                    if len(samples) == 4
                ]
                for gene in genes_with_4:
                    log.info(f"  Gene with 4 samples: {gene}")
                    for sample in deletion_samples_by_gene[gene]:
                        log.info(f"    - {sample.metadata.get('title', [''])[0]}")

        log.info("===")

        # Log gene resolution summary
        log.info("=== Gene Resolution Summary ===")
        log.info(f"Resolved by Table S1 gene: {self.resolved_by_table_s1_gene}")
        log.info(f"Resolved by Table S1 orf name: {self.resolved_by_table_s1_orf}")
        log.info(
            "Resolved by RELEASE_TITLE_NAMES_WITHOUT_AN_ORF: "
            f"{self.resolved_by_release_title_names}"
        )

        # Check for one-to-one mapping with Excel ORF names
        excel_orf_names = set(systematic_to_strain.keys())
        resolved_orf_names = set(deletion_samples_by_gene.keys())

        missing_in_geo = excel_orf_names - resolved_orf_names
        extra_in_geo = resolved_orf_names - excel_orf_names

        if missing_in_geo:
            log.warning("\n=== Missing ORFs Analysis ===")
            log.warning(
                f"Found {len(missing_in_geo)} ORFs in Excel but not in GEO: {missing_in_geo}"
            )
            for missing_orf in missing_in_geo:
                log.warning(
                    f"Missing: {missing_orf} (strain: {systematic_to_strain.get(missing_orf, 'Unknown')})"
                )
                # Try to find if it appears anywhere in common names
                for common, systematic in common_to_systematic.items():
                    if systematic == missing_orf:
                        log.warning(f"  - Has common name in Excel: {common}")

        if extra_in_geo:
            log.warning(
                f"Found {len(extra_in_geo)} ORFs in GEO but not in Excel: {extra_in_geo}"
            )

        # For now, just warn instead of asserting
        if excel_orf_names != resolved_orf_names:
            log.warning(
                f"\nWARNING: ORF name sets do not match exactly!\n"
                f"Excel has {len(excel_orf_names)} ORFs, GEO resolved to {len(resolved_orf_names)} ORFs\n"
                f"Missing in GEO: {missing_in_geo}\n"
                f"Extra in GEO: {extra_in_geo}\n"
                f"Continuing with {len(resolved_orf_names)} genes..."
            )
        else:
            log.info(
                f"✓ Perfect match: All {len(excel_orf_names)} Excel ORF names matched with GEO deletions"
            )

        # Debug: Check which genes from GEO are not in systematic_to_strain map
        missing_genes = []
        for gene in deletion_samples_by_gene.keys():
            if gene not in systematic_to_strain:
                missing_genes.append(gene)

        if missing_genes:
            log.warning(
                f"Found {len(missing_genes)} genes in GEO that are not in Excel systematic_to_strain map:"
            )
            log.warning(f"First 10 missing genes: {missing_genes[:10]}")

        # Check the channel assignment on the deleted genes' own probes
        self._validate_channel_assignment(deletion_samples_by_gene, probe_to_gene_map)

        # Choose processing method based on process_workers
        if self.process_workers > 0:
            log.info(
                f"Processing {len(deletion_samples_by_gene)} gene deletions in parallel with {self.process_workers} workers..."
            )
            self._process_parallel(
                deletion_samples_by_gene, probe_to_gene_map, systematic_to_strain
            )
        else:
            log.info(
                f"Processing {len(deletion_samples_by_gene)} gene deletions sequentially..."
            )
            self._process_sequential(
                deletion_samples_by_gene, probe_to_gene_map, systematic_to_strain
            )

        # Save preprocessed data
        os.makedirs(self.preprocess_dir, exist_ok=True)
        samples_df = pd.DataFrame(samples_data)
        samples_df = samples_df.drop(
            "gsm_object", axis=1
        )  # Remove GSM object before saving
        samples_df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        # Log final processing summary
        self._log_processing_summary(deletion_samples_by_gene, samples_data)

    def _process_sequential(
        self,
        deletion_samples_by_gene: dict[str, list[Any]],
        probe_to_gene_map: dict[str, str],
        systematic_to_strain: dict[str, str],
    ) -> None:
        """Process gene deletions sequentially (original implementation)."""
        # Initialize LMDB environment
        env = lmdb.open(
            osp.join(self.processed_dir, "lmdb"),
            map_size=int(5e12),  # 5TB for expression data
        )

        idx = 0
        with env.begin(write=True) as txn:
            # Process each unique gene deletion (collect replicate-level data)
            for gene_name, gsm_list in tqdm(deletion_samples_by_gene.items()):
                # Per-array (deletion, refpool) signal pairs, one pair per array
                replicate_pairs = self._collect_replicate_pairs_static(
                    gsm_list, probe_to_gene_map
                )

                # Skip if no expression data was extracted
                if not replicate_pairs:
                    log.warning(f"No expression data for gene {gene_name}, skipping...")
                    continue

                # Determine strain from systematic_to_strain map - REQUIRED
                if gene_name not in systematic_to_strain:
                    log.error(
                        f"No mating type found for {gene_name} in Table S1. This gene deletion is in GEO but not in the Excel file."
                    )
                    log.error(
                        f"Skipping {gene_name} - cannot determine correct strain (BY4741 vs BY4742)"
                    )
                    continue  # Skip this gene since we don't know the strain
                strain = systematic_to_strain[gene_name]

                # Create sample info for this gene deletion
                sample_info = {
                    "systematic_gene_name": gene_name,
                    "perturbed_gene_name": self.released_gene_names[gene_name],
                    "num_replicates": len(gsm_list),
                    "strain": strain,
                }

                experiment, reference, publication = self.create_expression_experiment(
                    self.name, sample_info, replicate_pairs
                )

                # Skip if experiment creation failed (returns None when log2 ratios can't be calculated)
                if experiment is None:
                    log.error(
                        f"Could not calculate log2 ratios for {gene_name}, skipping..."
                    )
                    continue

                # Serialize the Pydantic objects
                serialized_data = pickle.dumps(
                    {
                        "experiment": experiment.model_dump(),
                        "reference": reference.model_dump(),
                        "publication": publication.model_dump(),
                    }
                )
                txn.put(f"{idx}".encode(), serialized_data)
                idx += 1

        env.close()
        log.info(f"Wrote {idx} experiments to LMDB")
        log.info(f"Total gene deletions attempted: {len(deletion_samples_by_gene)}")
        if idx < len(deletion_samples_by_gene):
            log.warning(
                f"Skipped {len(deletion_samples_by_gene) - idx} genes (could not calculate log2 ratios)"
            )

    def _process_parallel(
        self,
        deletion_samples_by_gene: dict[str, list[Any]],
        probe_to_gene_map: dict[str, str],
        systematic_to_strain: dict[str, str],
    ) -> None:
        """Process gene deletions in parallel using ProcessPoolExecutor."""
        # Convert dictionary to list of tuples for batching
        gene_items = list(deletion_samples_by_gene.items())

        # Create batches
        batches = []
        for i in range(0, len(gene_items), self.batch_size):
            batch = gene_items[i : i + self.batch_size]
            batches.append(batch)

        log.info(f"Created {len(batches)} batches of size {self.batch_size}")

        # Process batches in parallel
        all_results = []
        with ProcessPoolExecutor(max_workers=self.process_workers) as executor:
            # Submit all batches
            futures = []
            for batch in batches:
                future = executor.submit(
                    self._process_batch,
                    batch,
                    probe_to_gene_map,
                    systematic_to_strain,
                    self.name,
                    self.released_gene_names,
                )
                futures.append(future)

            # Collect results with progress bar
            for future in tqdm(futures, desc="Processing batches"):
                batch_results = future.result()
                all_results.extend(batch_results)

        # Write all results to LMDB
        env = lmdb.open(
            osp.join(self.processed_dir, "lmdb"),
            map_size=int(5e12),  # 5TB for expression data
        )

        written_count = 0
        skipped_genes: list[int] = []
        with env.begin(write=True) as txn:
            for idx, serialized_data in enumerate(all_results):
                if serialized_data is not None:
                    txn.put(f"{written_count}".encode(), serialized_data)
                    written_count += 1
                else:
                    skipped_genes.append(idx)

        env.close()

        # Log statistics about experiments
        log.info(f"Wrote {written_count} experiments to LMDB")
        log.info(f"Total gene deletions attempted: {len(all_results)}")
        if skipped_genes:
            log.warning(
                f"Skipped {len(skipped_genes)} genes (could not calculate log2 ratios)"
            )
            log.warning(f"Indices of skipped genes: {skipped_genes}")

    @staticmethod
    def _process_batch(
        batch_items: list[Any],
        probe_to_gene_map: dict[str, str],
        systematic_to_strain: dict[str, str],
        dataset_name: str,
        released_gene_names: dict[str, str],
    ) -> list[bytes]:
        """Process a batch of gene deletions. Static method for multiprocessing."""
        results = []

        for gene_name, gsm_list in batch_items:
            # Per-array (deletion, refpool) signal pairs, one pair per array
            replicate_pairs = (
                MicroarrayKemmeren2014Dataset._collect_replicate_pairs_static(
                    gsm_list, probe_to_gene_map
                )
            )

            # Skip if no expression data was extracted
            if not replicate_pairs:
                continue

            # Determine strain from systematic_to_strain map - REQUIRED
            if gene_name not in systematic_to_strain:
                continue  # Skip this gene since we don't know the strain
            strain = systematic_to_strain[gene_name]

            # Create sample info for this gene deletion
            sample_info = {
                "systematic_gene_name": gene_name,
                "perturbed_gene_name": released_gene_names[gene_name],
                "num_replicates": len(gsm_list),
                "strain": strain,
            }

            experiment, reference, publication = (
                MicroarrayKemmeren2014Dataset.create_expression_experiment(
                    dataset_name, sample_info, replicate_pairs
                )
            )

            # Skip if experiment creation failed (returns None when log2 ratios can't be calculated)
            if experiment is None:
                continue

            # Serialize the Pydantic objects
            serialized_data = pickle.dumps(
                {
                    "experiment": experiment.model_dump(),
                    "reference": reference.model_dump(),
                    "publication": publication.model_dump(),
                }
            )
            results.append(serialized_data)

        return results

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
    def _extract_channels_from_gsm_static(
        gsm: Any, probe_to_gene_map: dict[str, str]
    ) -> tuple[SortedDict, SortedDict]:
        """Per-gene (test, reference) normalized signals of one array.

        Both values of a gene come from the same table row, so the pair shares its
        spot. Rows whose probe is not in the map and rows with a non-numeric cell are
        skipped; a gene with two probes keeps the last row, as before.
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

        test_data = SortedDict()
        reference_data = SortedDict()
        for _, row in table.iterrows():
            probe_id = str(int(row["ID_REF"]))
            if probe_id not in probe_to_gene_map:
                continue
            gene = probe_to_gene_map[probe_id]
            try:
                test_value = float(row[test_column])
                reference_value = float(row[reference_column])
            except (ValueError, TypeError):
                continue
            test_data[gene] = test_value
            reference_data[gene] = reference_value

        return test_data, reference_data

    @staticmethod
    def _collect_replicate_pairs_static(
        gsm_list: list[Any], probe_to_gene_map: dict[str, str]
    ) -> SortedDict:
        """Per-gene list of (deletion, refpool) signal pairs, one pair per array.

        The pairs feed ``create_expression_experiment``, which takes the log2 ratio
        within each array before averaging over arrays.
        """
        pairs = SortedDict()
        for gsm in gsm_list:
            deletion, refpool = (
                MicroarrayKemmeren2014Dataset._extract_channels_from_gsm_static(
                    gsm, probe_to_gene_map
                )
            )
            for gene, deletion_value in deletion.items():
                if gene not in pairs:
                    pairs[gene] = []
                pairs[gene].append((deletion_value, refpool[gene]))
        return pairs

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
        """Required by base class but not used - see create_expression_experiment."""
        pass

    def _log_processing_summary(
        self,
        deletion_samples_by_gene: dict[str, list[Any]],
        samples_data: list[dict[str, Any]],
    ) -> None:
        """Log consistent summary statistics for both sequential and parallel processing."""
        log.info(
            f"Processed {len(deletion_samples_by_gene)} unique gene deletion experiments"
        )

        # Find and log duplicate genes (genes appearing in multiple deletion experiments)
        from collections import Counter

        gene_counter = Counter(deletion_samples_by_gene.keys())
        duplicates = {gene: count for gene, count in gene_counter.items() if count > 1}

        if duplicates:
            log.info(f"Found {len(duplicates)} duplicate gene deletions:")
            for gene, count in sorted(duplicates.items()):
                log.info(f"  {gene}: {count} experiments")

        # Log statistics
        total_samples = len(samples_data)
        deletion_samples = sum(1 for s in samples_data if s["is_deletion"])
        wt_samples_count = sum(1 for s in samples_data if s["is_wildtype"])
        log.info(
            f"Total samples: {total_samples}, Deletion samples: {deletion_samples}, Wildtype: {wt_samples_count}"
        )
        log.info(f"Unique gene deletions: {len(deletion_samples_by_gene)}")

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
    def create_expression_experiment(
        dataset_name: str, sample_info: dict[str, Any], replicate_pairs: SortedDict
    ) -> tuple[Any, Any, Any]:
        """Build an experiment, reference, and publication from per-array signal pairs.

        ``replicate_pairs`` maps a gene to its (deletion, refpool) normalized signals,
        one pair per array (``_collect_replicate_pairs_static``). The log2 ratio is
        taken within each array, log2(deletion / refpool), so the two-color design
        cancels the spot; the mean, sample SD, SE and variance are then taken over the
        arrays. An array on which either signal is not positive is dropped for that
        gene. The stored linear ``expression`` is the mean deletion signal and the
        reference ``expression`` the mean refpool signal over the same arrays, so the
        reference ``n_replicates`` of a gene is the number of arrays whose refpool value
        entered that mean (1 or 2 on GEO), equal to the mutant's own count (#484; the
        paper's WT pools of 200, 200, 20 and 8 arrays are not the stored value, see the
        module comment at the reference pool constants).
        """
        # Genome reference - strain MUST be specified (BY4741 or BY4742)
        if "strain" not in sample_info:
            raise ValueError(
                "Strain (BY4741 or BY4742) must be specified in sample_info"
            )
        strain = sample_info["strain"]
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain=strain,
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        )

        # Create genotype for deletion mutant
        # ``perturbed_gene_name`` is the release's own name for the deleted gene
        # (``resolve_gene_name_comprehensive``)
        genotype = Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=sample_info["systematic_gene_name"],
                    perturbed_gene_name=sample_info["perturbed_gene_name"],
                )
            ]
        )

        # Environment - YPD medium at 30°C (or SC depending on dataset)
        environment = Environment(
            media=Media(
                name="SC", state="liquid", is_synthetic=True
            ),  # Kemmeren used SC medium
            temperature=Temperature(value=30),
        )
        environment_reference = environment.model_copy()

        # Within-array log2 ratios, then statistics over the arrays on the log2 scale
        mean_log2_ratios = SortedDict()
        log2_se = SortedDict()
        log2_variance = SortedDict()
        n_replicates_dict = SortedDict()
        mean_expression = SortedDict()  # Mean LINEAR deletion signal (for QC)
        refpool_expression = SortedDict()  # Mean LINEAR refpool signal (reference)
        refpool_n_replicates = SortedDict()  # Arrays in each refpool mean

        for gene, pairs in replicate_pairs.items():
            kept = [(d, r) for d, r in pairs if d > 0 and r > 0]
            if not kept:
                continue

            log2_ratios_per_array = [float(np.log2(d / r)) for d, r in kept]
            n = len(log2_ratios_per_array)
            mean_log2 = float(np.mean(log2_ratios_per_array))

            se_log2: float
            var_log2: float
            if n > 1:
                sd_log2 = float(np.std(log2_ratios_per_array, ddof=1))
                se_log2 = sd_log2 / n**0.5
                var_log2 = sd_log2**2
            else:
                # n=1: SE and variance are undefined
                se_log2 = float("nan")
                var_log2 = float("nan")

            mean_log2_ratios[gene] = mean_log2
            log2_se[gene] = se_log2
            log2_variance[gene] = var_log2
            n_replicates_dict[gene] = n
            mean_expression[gene] = float(np.mean([d for d, _ in kept]))
            refpool_expression[gene] = float(np.mean([r for _, r in kept]))
            refpool_n_replicates[gene] = n

        if not mean_log2_ratios:
            # No array with both signals positive - return None to signal skip
            return None, None, None

        # Create phenotype with new schema fields
        phenotype = MicroarrayExpressionPhenotype(
            expression=mean_expression,  # Mean LINEAR expression (for QC)
            expression_log2_ratio=mean_log2_ratios,  # Mean log2 ratios
            expression_log2_ratio_se=log2_se,  # SE on log2 scale
            expression_log2_ratio_variance=log2_variance,  # Variance on log2 scale
            n_replicates=n_replicates_dict,  # Number of replicates per gene
        )

        # Create reference phenotype (refpool expression)
        # For reference, create self-referential log2 ratios (all zeros)
        reference_log2_ratios = SortedDict()
        for gene in refpool_expression:
            reference_log2_ratios[gene] = 0.0  # log2(1) = 0

        phenotype_reference = MicroarrayExpressionPhenotype(
            expression=refpool_expression,
            expression_log2_ratio=reference_log2_ratios,
            n_replicates=refpool_n_replicates,  # Arrays in each refpool mean
        )

        # Create reference
        reference = MicroarrayExpressionExperimentReference(
            dataset_name=dataset_name,
            genome_reference=genome_reference,
            environment_reference=environment_reference,
            phenotype_reference=phenotype_reference,
        )

        # Create experiment
        experiment = MicroarrayExpressionExperiment(
            dataset_name=dataset_name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )

        # Publication
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
    # dataset = MicroarrayKemmeren2014Dataset(
    #     root=osp.join(DATA_ROOT, "data/torchcell/microarray_kemmeren2014"),
    # )
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset gene set size: {len(dataset.gene_set)}")
    print(f"First 10 genes in gene_set: {list(dataset.gene_set)[:10]}")

    if len(dataset) > 0:
        # Get raw data (dictionary format) - this is what dataset[0] returns
        data = dataset[0]

        # The data is returned as a dictionary with deserialized content
        print("\nFirst dataset item (index 0):")
        print(f"  Data type: {type(data)}")
        print(f"  Keys: {data.keys()}")

        # Access the dictionaries
        experiment = data["experiment"]
        reference = data["reference"]
        publication = data["publication"]

        print("\n=== Experiment Details ===")
        print(f"  Dataset: {experiment['dataset_name']}")
        perturbed_gene = experiment["genotype"]["perturbations"][0][
            "systematic_gene_name"
        ]
        print(f"  Perturbed gene: {perturbed_gene}")

        # Show expression data summary
        exp_expression = experiment["phenotype"]["expression"]
        print(f"  Expression measurements: {len(exp_expression)} genes")

        # Show first 5 expression values
        print("  First 5 expression values:")
        for i, (gene, value) in enumerate(list(exp_expression.items())[:5]):
            print(f"    {gene}: {value:.4f}")

        print("\n=== Reference Details ===")
        print(f"  Dataset: {reference['dataset_name']}")
        print(f"  Genome: {reference['genome_reference']}")
        print(f"  Environment: {reference['environment_reference']}")

        # Show reference expression (wildtype baseline)
        ref_expression = reference["phenotype_reference"]["expression"]
        print(f"  Reference expression: {len(ref_expression)} genes")

        # Show first 5 reference expression values
        print("  First 5 reference expression values (wildtype):")
        for i, (gene, value) in enumerate(list(ref_expression.items())[:5]):
            print(f"    {gene}: {value:.4f}")

        # Compare specific gene between experiment and reference
        print("\n=== Gene Comparison (exp vs ref) ===")
        # Pick first 3 genes for comparison
        sample_genes = list(exp_expression.keys())[:3]
        for gene in sample_genes:
            if gene in ref_expression:
                exp_val = exp_expression[gene]
                ref_val = ref_expression[gene]
                log2_ratio = np.log2(exp_val / ref_val) if ref_val != 0 else np.nan
                print(f"  {gene}:")
                print(f"    Deletion mutant: {exp_val:.4f}")
                print(f"    Wildtype (ref): {ref_val:.4f}")
                print(f"    Log2 ratio: {log2_ratio:.4f}")

        # Check perturbed gene presence
        print("\n=== Perturbed Gene Status ===")
        print(f"  Gene: {perturbed_gene}")
        print(f"  In deletion mutant expression: {perturbed_gene in exp_expression}")
        print(f"  In wildtype reference expression: {perturbed_gene in ref_expression}")

        print("\n=== Publication ===")
        print(f"  PubMed ID: {publication['pubmed_id']}")
        print(f"  DOI: {publication['doi']}")
