# torchcell/datasets/scerevisiae/sameith2015
# [[torchcell.datasets.scerevisiae.sameith2015]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/sameith2015
# Test file: tests/torchcell/datasets/scerevisiae/test_sameith2015.py

import GEOparse
import logging
import os
import os.path as osp
import pickle
import re
import urllib.request
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
import lmdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from sortedcontainers import SortedDict
from torchcell.datamodels.schema import (
    Environment,
    Genotype,
    MicroarrayExpressionExperiment,
    MicroarrayExpressionExperimentReference,
    MicroarrayExpressionPhenotype,
    Media,
    ReferenceGenome,
    SgaKanMxDeletionPerturbation,
    SgaNatMxDeletionPerturbation,
    Temperature,
    Experiment,
    ExperimentReference,
    Publication,
)
from torchcell.data import ExperimentDataset, post_process
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")


@register_dataset
class SmMicroarraySameith2015Dataset(ExperimentDataset):
    """Single mutant GSTF expression profiling from Sameith et al. 2015.

    Microarray gene expression data for 82 single mutant yeast strains with
    deletions in general stress transcription factors (GSTFs). All single mutants
    are from the yeast deletion library (BY4742, mata, KanMX marker).

    Data source: GEO accession GSE42536
    Paper: Sameith et al. (2015) BMC Biology
    DOI: 10.1186/s12915-015-0222-5
    """

    geo_accession = "GSE42536"

    def __init__(
        self,
        root: str = "data/torchcell/sm_microarray_sameith2015",
        io_workers: int = 0,
        process_workers: int = 0,
        batch_size: int = 10,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        **kwargs,
    ):
        self.process_workers = process_workers
        self.batch_size = batch_size

        # Initialize genome for gene name mapping BEFORE calling super().__init__
        self.genome = SCerevisiaeGenome(
            genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
            go_root=osp.join(DATA_ROOT, "data/go"),
            overwrite=True,
        )

        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        return MicroarrayExpressionExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        return MicroarrayExpressionExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        return [f"{self.geo_accession}_family.soft.gz"]

    def download(self):
        """Download single mutant microarray expression data from GEO."""
        log.info(f"Downloading GEO dataset {self.geo_accession}...")

        try:
            gse = GEOparse.get_GEO(
                geo=self.geo_accession, destdir=self.raw_dir, silent=False
            )
            log.info(f"Successfully downloaded {self.geo_accession}")

            # Save the parsed GEO object for use in process()
            geo_pkl_path = osp.join(self.raw_dir, f"{self.geo_accession}.pkl")
            with open(geo_pkl_path, "wb") as f:
                pickle.dump(gse, f)
            log.info(f"Saved GEO object to {geo_pkl_path}")

        except Exception as e:
            log.error(f"Failed to download GEO dataset: {e}")
            raise RuntimeError(f"GEO download failed")

        # Download supplementary Excel file
        suppl_url = "https://static-content.springer.com/esm/art%3A10.1186%2Fs12915-015-0222-5/MediaObjects/12915_2015_222_MOESM1_ESM.xlsx"
        suppl_path = osp.join(self.raw_dir, "12915_2015_222_MOESM1_ESM.xlsx")

        try:
            log.info(f"Downloading supplementary file...")
            urllib.request.urlretrieve(suppl_url, suppl_path)
            log.info(f"Downloaded supplementary file to {suppl_path}")
        except Exception as e:
            log.error(f"Failed to download supplementary file: {e}")
            raise RuntimeError(f"Failed to download supplementary data")

    def _load_authoritative_single_mutants(self):
        """Load authoritative single mutants from supplementary Excel file."""
        suppl_path = osp.join(self.raw_dir, "12915_2015_222_MOESM1_ESM.xlsx")
        df = pd.read_excel(suppl_path, sheet_name="Single mutants - info")
        valid_df = df[df["systematic name"].notna()].copy()

        log.info(f"Loaded {len(valid_df)} single mutants from supplementary file")

        single_mutants = {}
        for _, row in valid_df.iterrows():
            systematic_name = row["systematic name"].strip().upper()
            single_mutants[systematic_name] = {
                "systematic_name": systematic_name,
                "gene_symbol": row.get("gene symbol", ""),
                "strain": "BY4742",  # All from deletion library
            }

        return single_mutants

    @post_process
    def process(self):
        # Initialize resolution statistics
        self.resolved_by_excel = 0
        self.resolved_by_gene_table = 0
        self.resolved_by_alias = 0
        self.unresolved_genes = 0

        # Load the GEO object
        geo_pkl_path = osp.join(self.raw_dir, f"{self.geo_accession}.pkl")
        if osp.exists(geo_pkl_path):
            with open(geo_pkl_path, "rb") as f:
                gse = pickle.load(f)
        else:
            # Re-download if pickle doesn't exist
            gse = GEOparse.get_GEO(
                geo=self.geo_accession, destdir=self.raw_dir, silent=False
            )

        # Load authoritative single mutants from supplementary file
        self.single_mutants = self._load_authoritative_single_mutants()

        log.info("Processing GEO samples for single mutants...")

        # Extract platform annotation for probe-to-gene mapping
        probe_to_gene_map = self._extract_probe_to_gene_mapping(gse)
        log.info(f"Found {len(probe_to_gene_map)} probe-to-gene mappings")

        # Parse samples and extract metadata
        samples_data = []
        single_mutant_samples = {}  # Group by gene name
        wt_samples = []

        for gsm_name, gsm in gse.gsms.items():
            sample_info = {
                "geo_accession": gsm_name,
                "title": gsm.metadata.get("title", [""])[0],
            }

            # Extract gene names from title
            gene_names = self._extract_gene_names_from_title(sample_info["title"])

            # Determine sample type
            is_wildtype = (
                "wt" in sample_info["title"].lower()
                or "wildtype" in sample_info["title"].lower()
            )
            is_single = len(gene_names) == 1

            sample_info["is_wildtype"] = is_wildtype
            sample_info["is_single_mutant"] = is_single
            sample_info["gene_names"] = gene_names
            sample_info["gsm_object"] = gsm

            if is_wildtype:
                wt_samples.append(gsm)
            elif is_single:
                gene = gene_names[0]
                if gene not in single_mutant_samples:
                    single_mutant_samples[gene] = []
                single_mutant_samples[gene].append(sample_info)

            samples_data.append(sample_info)

        log.info(f"Found {len(single_mutant_samples)} unique single mutant genes")
        log.info(f"Found {len(wt_samples)} wildtype samples")

        # Calculate wildtype reference expression
        if wt_samples:
            self.wt_reference_expression, self.wt_std_expression = (
                self._calculate_wt_reference_with_std(wt_samples, probe_to_gene_map)
            )
        else:
            log.warning("No wildtype samples found, using synthetic reference")
            self.wt_reference_expression = SortedDict()
            self.wt_std_expression = SortedDict()

        # Group single mutant samples by gene using authoritative list
        single_mutant_groups = self._group_single_mutant_replicates(
            single_mutant_samples
        )
        log.info(
            f"Grouped into {len(single_mutant_groups)} replicate groups for processing"
        )

        # Process single mutants sequentially
        log.info(f"Processing {len(single_mutant_groups)} single mutant groups...")
        self._process_sequential(single_mutant_groups, probe_to_gene_map)

        # Save preprocessed data - filter for single mutants only
        os.makedirs(self.preprocess_dir, exist_ok=True)
        samples_df = pd.DataFrame(samples_data)
        samples_df = samples_df.drop("gsm_object", axis=1)
        # Filter to only include single mutant samples
        single_mutant_df = samples_df[samples_df["is_single_mutant"] == True]
        single_mutant_df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)
        log.info(f"Saved {len(single_mutant_df)} single mutant samples to preprocess data.csv")

        log.info(f"Processed {len(single_mutant_groups)} single mutant experiments")

    def _group_single_mutant_replicates(self, single_mutant_samples):
        """Group technical replicates by gene using authoritative single mutant list.

        Args:
            single_mutant_samples: Dict mapping gene -> list of sample_info dicts

        Returns:
            list: List of replicate groups
        """
        groups = []
        matched_count = 0
        unmatched_count = 0

        for gene, sample_list in single_mutant_samples.items():
            if gene in self.single_mutants:
                groups.append(sample_list)
                matched_count += len(sample_list)
            else:
                log.warning(f"Gene {gene} not in authoritative list")
                unmatched_count += len(sample_list)
                groups.append(sample_list)  # Still process

        log.info(f"Matched {matched_count} samples, {unmatched_count} unmatched")
        log.info(f"Found {len(groups)} unique genotypes")

        return groups

    def _process_sequential(self, single_mutant_groups, probe_to_gene_map):
        """Process single mutant replicate groups sequentially."""
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(5e12))

        idx = 0
        with env.begin(write=True) as txn:
            for replicate_group in tqdm(single_mutant_groups):
                # Use first sample for metadata
                sample_info = replicate_group[0]

                if (
                    not sample_info["is_single_mutant"]
                    or len(sample_info["gene_names"]) < 1
                ):
                    continue

                # Extract and average data from all replicates
                all_mutant_data = []
                all_refpool_data = []
                all_log2_ratio_data = []

                for rep_info in replicate_group:
                    gsm = rep_info["gsm_object"]
                    mutant_data, refpool_data, log2_ratio_data = (
                        self._extract_expression_from_gsm(gsm, probe_to_gene_map)
                    )

                    if mutant_data and refpool_data and log2_ratio_data:
                        all_mutant_data.append(mutant_data)
                        all_refpool_data.append(refpool_data)
                        all_log2_ratio_data.append(log2_ratio_data)

                # Skip if no data from any replicate
                if not all_mutant_data:
                    log.warning(f"No data for {sample_info['gene_names']}, skipping...")
                    continue

                # Calculate means and stds across replicates
                mutant_mean, mutant_std = self._calculate_mean_std(all_mutant_data)
                refpool_mean, refpool_std = self._calculate_mean_std(all_refpool_data)
                log2_mean, log2_std = self._calculate_mean_std(all_log2_ratio_data)

                experiment, reference, publication = (
                    self.create_single_mutant_expression_experiment(
                        self.name,
                        sample_info,
                        mutant_mean,
                        refpool_mean,
                        log2_mean,
                        mutant_std,
                        refpool_std,
                        log2_std,
                    )
                )

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
        log.info(f"Wrote {idx} single mutant experiments to LMDB")

    def _calculate_mean_std(self, data_list):
        """Calculate mean and std across technical replicates.

        Args:
            data_list: List of SortedDict objects containing gene -> value mappings

        Returns:
            tuple: (mean_dict, std_dict) - SortedDict objects with means and stds
        """
        if not data_list:
            return SortedDict(), SortedDict()

        # Collect all values per gene across replicates
        all_values = {}
        for data_dict in data_list:
            for gene, value in data_dict.items():
                if gene not in all_values:
                    all_values[gene] = []
                all_values[gene].append(value)

        # Calculate mean and std
        mean_dict = SortedDict()
        std_dict = SortedDict()

        for gene, values in all_values.items():
            mean_dict[gene] = np.mean(values)
            if len(values) > 1:
                std_dict[gene] = np.std(values, ddof=1)
            else:
                std_dict[gene] = np.nan

        return mean_dict, std_dict

    def _extract_gene_names_from_title(self, title: str) -> list[str]:
        """Extract systematic gene names from sample title."""
        gene_names = []

        # Look for systematic names (e.g., YAL001C or YBR089C-A)
        # Match the full systematic name, including optional suffix like -A
        systematic_pattern = r"\b(Y[A-P][LR]\d{3}[WC](?:-[A-Z])?)\b"
        matches = re.findall(systematic_pattern, title.upper())

        # Validate each match
        for match in matches:
            if self._is_valid_systematic_name(match):
                gene_names.append(match)

        # Also try to extract common names if needed
        if len(gene_names) < 1:
            # Extract potential gene names (all caps words)
            common_pattern = r"\b([A-Z][A-Z0-9]{2,})\b"
            potential_genes = re.findall(common_pattern, title.upper())

            for gene in potential_genes:
                # Skip common non-gene words
                if gene in ["DEL", "WT", "WILDTYPE", "MUTANT", "SINGLE"]:
                    continue

                # Try to convert to systematic
                systematic = self._convert_to_systematic(gene)
                if systematic and self._is_valid_systematic_name(systematic):
                    # Avoid duplicates
                    if systematic not in gene_names:
                        gene_names.append(systematic)
                        break  # We only need 1 gene for single mutants

        return gene_names[:1]  # Return at most 1 gene for single mutants

    def _is_valid_systematic_name(self, name: str) -> bool:
        """Validate that a systematic name matches the expected format."""
        if not name:
            return False
        # Must match exact pattern: Y + chromosome letter + L/R + 3 digits + C/W (with optional suffix like -A)
        return bool(re.match(r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?$", name.upper()))

    def _convert_to_systematic(self, gene_name: str) -> str | None:
        """Convert common gene name to systematic name."""
        if not gene_name:
            return None

        gene_upper = gene_name.upper()

        # Check if already systematic
        if re.match(r"Y[A-P][LR]\d{3}[WC](-[A-Z])?", gene_upper):
            return gene_upper

        # Try genome's gene_attribute_table
        if hasattr(self.genome, "gene_attribute_table"):
            df = self.genome.gene_attribute_table

            # Check gene column
            matches = df[df["gene"] == gene_upper]
            if not matches.empty:
                return matches.iloc[0]["ID"]

            # Check Alias column
            matches = df[df["Alias"] == gene_upper]
            if not matches.empty:
                return matches.iloc[0]["ID"]

        # Try alias_to_systematic
        if hasattr(self.genome, "alias_to_systematic"):
            candidates = self.genome.alias_to_systematic.get(gene_upper, [])
            if candidates:
                return candidates[0]  # Return first match

        return None

    def _extract_probe_to_gene_mapping(self, gse) -> dict:
        """Extract probe ID to gene name mapping from GEO platform annotation."""
        probe_to_gene = {}

        if not hasattr(gse, "gpls") or not gse.gpls:
            log.warning("No platform annotation found in GEO dataset")
            return probe_to_gene

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
                        probe_id = str(int(row[id_column]))
                        gene_name = str(row[gene_column])

                        # Clean and validate gene name
                        if gene_name and gene_name != "nan" and gene_name != "":
                            gene_name_upper = gene_name.upper()
                            if re.match(r"Y[A-Z]{2}\d{3}[CW]", gene_name_upper):
                                probe_to_gene[probe_id] = gene_name_upper
                            elif re.match(r"Q\d{4}", gene_name_upper):
                                probe_to_gene[probe_id] = gene_name_upper
                            else:
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

    def _extract_expression_from_gsm(
        self, gsm, probe_to_gene_map=None
    ) -> tuple[SortedDict, SortedDict, SortedDict]:
        """Extract expression values and log2 ratios from a GSM object.

        Handles dye swaps between technical replicates by checking metadata to determine
        which channel contains the mutant vs. reference pool.

        Returns:
            tuple: (mutant_data, refpool_data, log2_ratio_data)
                - mutant_data: Expression values for the deletion mutant
                - refpool_data: Expression values for the reference pool
                - log2_ratio_data: log2(mutant/refpool) ratios
        """
        mutant_data = SortedDict()
        refpool_data = SortedDict()
        log2_ratio_data = SortedDict()

        if hasattr(gsm, "table") and gsm.table is not None:
            table = gsm.table

            # Log available columns for the first sample
            if not hasattr(self, "_logged_columns"):
                log.info(f"Available columns in GSM table: {list(table.columns)}")
                self._logged_columns = True

            if "ID_REF" not in table.columns:
                return mutant_data, refpool_data, log2_ratio_data

            # Check for required columns
            has_cy5 = "Signal Norm_Cy5" in table.columns
            has_cy3 = "Signal Norm_Cy3" in table.columns
            has_value = "VALUE" in table.columns

            if not (has_cy5 and has_cy3 and has_value):
                log.warning(
                    f"Missing required columns. Available: {list(table.columns)}"
                )
                return mutant_data, refpool_data, log2_ratio_data

            # Determine which channel has mutant vs refpool by checking metadata
            # Due to dye swaps, this varies between technical replicates
            source_ch1 = (
                gsm.metadata.get("source_name_ch1", [""])[0]
                if hasattr(gsm, "metadata")
                else ""
            )

            # Determine which dye (Cy5/Cy3) corresponds to which sample (mutant/refpool)
            # Ch1 = Cy5, Ch2 = Cy3 (based on GEO data structure)
            if "refpool" in source_ch1.lower():
                # Cy5 = refpool, Cy3 = mutant
                mutant_channel = "Cy3"
                # VALUE is log2(Cy5/Cy3) = log2(refpool/mutant), so negate it
                ratio_sign = -1
            else:
                # Cy5 = mutant, Cy3 = refpool
                mutant_channel = "Cy5"
                # VALUE is log2(Cy5/Cy3) = log2(mutant/refpool), correct sign
                ratio_sign = 1

            for _, row in table.iterrows():
                probe_id = str(int(row["ID_REF"]))

                # Map probe ID to gene name
                if probe_to_gene_map and probe_id in probe_to_gene_map:
                    gene = probe_to_gene_map[probe_id]
                    try:
                        cy5_value = float(row["Signal Norm_Cy5"])
                        cy3_value = float(row["Signal Norm_Cy3"])
                        value = float(row["VALUE"])

                        # Assign to mutant/refpool based on channel assignment
                        if mutant_channel == "Cy5":
                            mutant_data[gene] = cy5_value
                            refpool_data[gene] = cy3_value
                        else:
                            mutant_data[gene] = cy3_value
                            refpool_data[gene] = cy5_value

                        # Adjust log2 ratio sign to always be log2(mutant/refpool)
                        log2_ratio_data[gene] = ratio_sign * value

                    except (ValueError, TypeError):
                        continue

        return mutant_data, refpool_data, log2_ratio_data

    def _calculate_wt_reference_with_std(
        self, wt_gsm_list, probe_to_gene_map
    ) -> tuple[SortedDict, SortedDict]:
        """Calculate average wildtype expression values and std from GSM objects."""
        if len(wt_gsm_list) == 0:
            log.warning("No wildtype samples found, returning empty reference")
            return SortedDict(), SortedDict()

        log.info(f"Calculating reference from {len(wt_gsm_list)} wildtype samples")

        # Collect all expression values per gene
        all_expressions = {}

        for gsm in wt_gsm_list:
            mutant_data, refpool_data, log2_ratio_data = (
                self._extract_expression_from_gsm(gsm, probe_to_gene_map)
            )
            # Use reference pool data for wildtype reference
            for gene, value in refpool_data.items():
                if gene not in all_expressions:
                    all_expressions[gene] = []
                all_expressions[gene].append(value)

        # Calculate mean and std
        wt_mean_expression = SortedDict()
        wt_std_expression = SortedDict()

        for gene, values in all_expressions.items():
            wt_mean_expression[gene] = np.mean(values)
            wt_std_expression[gene] = np.std(values, ddof=1) if len(values) > 1 else 0.0

        log.info(f"Calculated reference for {len(wt_mean_expression)} genes")

        return wt_mean_expression, wt_std_expression

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict | None = None
    ) -> pd.DataFrame:
        """Preprocess raw data - for Sameith this is handled in process()."""
        return df

    def create_experiment(self):
        """Required by base class but not used."""
        pass

    @staticmethod
    def create_single_mutant_expression_experiment(
        dataset_name,
        sample_info,
        mutant_data,
        refpool_data,
        log2_ratio_data,
        mutant_std=None,
        refpool_std=None,
        log2_ratio_std=None,
    ):
        """Create experiment for single deletion mutant.

        Args:
            mutant_data: Expression values for deletion mutant
            refpool_data: Expression values for reference pool
            log2_ratio_data: log2(mutant/refpool) ratios
            mutant_std: Technical std for mutant expression (optional)
            refpool_std: Technical std for refpool expression (optional)
            log2_ratio_std: Technical std for log2 ratios (optional)
        """
        # BY4742 for single mutants (from deletion library)
        # Paper: "Single mutants taken from Deletion library" = BY4742, mata
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain="BY4742"
        )

        # Create genotype for single deletion mutant
        gene_names = sample_info["gene_names"]
        perturbations = []

        # Single deletion (KanMX marker from deletion library)
        if len(gene_names) > 0:
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=gene_names[0],
                    perturbed_gene_name=gene_names[0],
                    strain_id=f"KanMX_{gene_names[0]}",
                )
            )

        genotype = Genotype(perturbations=perturbations)

        # Environment - SC medium at 30°C
        environment = Environment(
            media=Media(name="SC", state="liquid"), temperature=Temperature(value=30)
        )
        environment_reference = environment.model_copy()

        # Use pre-computed log2 ratios
        log2_ratios = log2_ratio_data

        # Create phenotype with expression data from mutant
        phenotype = MicroarrayExpressionPhenotype(
            expression=mutant_data,
            expression_log2_ratio=log2_ratios,
            expression_technical_std=mutant_std,
            expression_log2_ratio_std=log2_ratio_std,
        )

        # Create reference phenotype from reference pool
        reference_log2_ratios = SortedDict()
        for gene in refpool_data:
            reference_log2_ratios[gene] = 0.0

        phenotype_reference = MicroarrayExpressionPhenotype(
            expression=refpool_data,
            expression_log2_ratio=reference_log2_ratios,
            expression_technical_std=refpool_std,
            expression_log2_ratio_std=None,
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

        # Publication for Sameith et al. 2015
        publication = Publication(
            pubmed_id="26687005",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/26687005/",
            doi="10.1186/s12915-015-0222-5",
            doi_url="https://doi.org/10.1186/s12915-015-0222-5",
        )

        return experiment, reference, publication


@register_dataset
class DmMicroarraySameith2015Dataset(ExperimentDataset):
    # GEO accession for double mutant microarray expression dataset
    geo_accession = "GSE42536"

    def __init__(
        self,
        root: str = "data/torchcell/dm_microarray_sameith2015",
        io_workers: int = 0,
        process_workers: int = 0,
        batch_size: int = 10,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        **kwargs,
    ):
        self.process_workers = process_workers
        self.batch_size = batch_size

        # Initialize genome for gene name mapping BEFORE calling super().__init__
        self.genome = SCerevisiaeGenome(
            genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
            go_root=osp.join(DATA_ROOT, "data/go"),
            overwrite=True,
        )

        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        return MicroarrayExpressionExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        return MicroarrayExpressionExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        return [f"{self.geo_accession}_family.soft.gz"]

    def download(self):
        """Download double mutant microarray expression data from GEO and supplementary data."""
        log.info(f"Downloading GEO dataset {self.geo_accession}...")

        try:
            gse = GEOparse.get_GEO(
                geo=self.geo_accession, destdir=self.raw_dir, silent=False
            )
            log.info(f"Successfully downloaded {self.geo_accession}")

            # Save the parsed GEO object for use in process()
            geo_pkl_path = osp.join(self.raw_dir, f"{self.geo_accession}.pkl")
            with open(geo_pkl_path, "wb") as f:
                pickle.dump(gse, f)

        except Exception as e:
            log.error(f"Failed to download GEO data: {e}")
            raise RuntimeError(f"Failed to download {self.geo_accession} from GEO")

        # Download supplementary Excel file with authoritative GSTF pairs
        log.info("Downloading supplementary data file...")
        suppl_url = "https://static-content.springer.com/esm/art%3A10.1186%2Fs12915-015-0222-5/MediaObjects/12915_2015_222_MOESM1_ESM.xlsx"
        suppl_path = osp.join(self.raw_dir, "12915_2015_222_MOESM1_ESM.xlsx")

        try:
            if not osp.exists(suppl_path):
                urllib.request.urlretrieve(suppl_url, suppl_path)
                log.info(f"Successfully downloaded supplementary file to {suppl_path}")
            else:
                log.info("Supplementary file already exists")
        except Exception as e:
            log.error(f"Failed to download supplementary file: {e}")
            raise RuntimeError(f"Failed to download supplementary data")

    def _load_authoritative_gstf_pairs(self):
        """Load authoritative GSTF pairs WITH PER-SAMPLE STRAIN from supplementary Excel file.

        Returns:
            dict: Dictionary mapping (gene1, gene2) tuples to their metadata including strain
        """
        suppl_path = osp.join(self.raw_dir, "12915_2015_222_MOESM1_ESM.xlsx")

        # Read the "Double mutants - info" sheet
        df = pd.read_excel(suppl_path, sheet_name="Double mutants - info")

        # Filter for pairs that passed QC
        passed_df = df[df["curation"] == "passed"].copy()

        log.info(
            f"Loaded {len(passed_df)} GSTF pairs that passed QC from supplementary file"
        )

        # Create a dictionary of genotype pairs
        gstf_pairs = {}
        strain_counts = {"BY4742": 0, "BY4741": 0}

        for _, row in passed_df.iterrows():
            gstf1_sys = row["GSTF1,           systematic name"].strip()
            gstf2_sys = row["GSTF2,           systematic name"].strip()

            # Create canonical genotype key (sorted to handle order-independence)
            genotype_key = tuple(sorted([gstf1_sys.upper(), gstf2_sys.upper()]))

            # EXTRACT STRAIN FROM COMMENTS COLUMN
            comments = row.get("comments", "")
            if pd.notna(comments):
                comments_str = str(comments)
                if "MATa" in comments_str:
                    strain = "BY4742"  # mata mating type
                elif "MATα" in comments_str or "matA" in comments_str:
                    strain = "BY4741"  # matA/alpha mating type
                else:
                    strain = "BY4742"  # Default for comments without mating type
            else:
                strain = "BY4742"  # Default for blank comments

            strain_counts[strain] += 1

            gstf_pairs[genotype_key] = {
                "gstf1_systematic": gstf1_sys.upper(),
                "gstf2_systematic": gstf2_sys.upper(),
                "gstf1_symbol": row["GSTF1,           gene symbol"],
                "gstf2_symbol": row["GSTF2,           gene symbol"],
                "selection": row["selection"],
                "comments": comments,
                "strain": strain,  # NEW: Per-sample strain extracted from comments
            }

        log.info(
            f"Strain distribution: BY4742={strain_counts['BY4742']}, BY4741={strain_counts['BY4741']}"
        )
        return gstf_pairs

    @post_process
    def process(self):
        # Initialize resolution statistics
        self.resolved_by_excel = 0
        self.resolved_by_gene_table = 0
        self.resolved_by_alias = 0
        self.unresolved_genes = 0

        # Load the GEO object
        geo_pkl_path = osp.join(self.raw_dir, f"{self.geo_accession}.pkl")
        if osp.exists(geo_pkl_path):
            with open(geo_pkl_path, "rb") as f:
                gse = pickle.load(f)
        else:
            # Re-download if pickle doesn't exist
            gse = GEOparse.get_GEO(
                geo=self.geo_accession, destdir=self.raw_dir, silent=False
            )

        # Load authoritative GSTF pairs from supplementary file
        self.gstf_pairs = self._load_authoritative_gstf_pairs()

        log.info("Processing GEO samples for double mutants...")

        # Extract platform annotation for probe-to-gene mapping
        probe_to_gene_map = self._extract_probe_to_gene_mapping(gse)
        log.info(f"Found {len(probe_to_gene_map)} probe-to-gene mappings")

        # Parse samples and extract metadata
        samples_data = []
        single_mutant_samples = {}  # Group by gene name
        double_mutant_samples = []  # List of double mutant sample infos
        wt_samples = []

        for gsm_name, gsm in gse.gsms.items():
            sample_info = {
                "geo_accession": gsm_name,
                "title": gsm.metadata.get("title", [""])[0],
            }

            # Extract gene names from title
            gene_names = self._extract_gene_names_from_title(sample_info["title"])

            # Determine sample type
            is_wildtype = (
                "wt" in sample_info["title"].lower()
                or "wildtype" in sample_info["title"].lower()
            )
            is_double = len(gene_names) >= 2
            is_single = len(gene_names) == 1

            sample_info["is_wildtype"] = is_wildtype
            sample_info["is_double_mutant"] = is_double
            sample_info["is_single_mutant"] = is_single
            sample_info["gene_names"] = gene_names
            sample_info["gsm_object"] = gsm

            if is_wildtype:
                wt_samples.append(gsm)
            elif is_single:
                gene = gene_names[0]
                if gene not in single_mutant_samples:
                    single_mutant_samples[gene] = []
                single_mutant_samples[gene].append(gsm)
            elif is_double:
                double_mutant_samples.append(sample_info)

            samples_data.append(sample_info)

        log.info(f"Found {len(single_mutant_samples)} unique single mutant genes")
        log.info(f"Found {len(double_mutant_samples)} double mutant samples")
        log.info(f"Found {len(wt_samples)} wildtype samples")

        # Calculate wildtype reference expression
        if wt_samples:
            self.wt_reference_expression, self.wt_std_expression = (
                self._calculate_wt_reference_with_std(wt_samples, probe_to_gene_map)
            )
        else:
            log.warning("No wildtype samples found, using synthetic reference")
            self.wt_reference_expression = SortedDict()
            self.wt_std_expression = SortedDict()

        log.info(
            f"Found {len(single_mutant_samples)} single mutant genes (not processed separately)"
        )

        # Group double mutant samples by base name (to handle technical replicates)
        double_mutant_groups = self._group_technical_replicates(double_mutant_samples)
        log.info(
            f"Grouped {len(double_mutant_samples)} samples into {len(double_mutant_groups)} replicate groups"
        )

        # Choose processing method based on process_workers
        if self.process_workers > 0:
            log.info(
                f"Processing {len(double_mutant_groups)} double mutant groups in parallel with {self.process_workers} workers..."
            )
            self._process_parallel(double_mutant_groups, probe_to_gene_map)
        else:
            log.info(
                f"Processing {len(double_mutant_groups)} double mutant groups sequentially..."
            )
            self._process_sequential(double_mutant_groups, probe_to_gene_map)

        # Save preprocessed data - filter for double mutants only
        os.makedirs(self.preprocess_dir, exist_ok=True)
        samples_df = pd.DataFrame(samples_data)
        samples_df = samples_df.drop("gsm_object", axis=1)
        # Filter to only include double mutant samples
        double_mutant_df = samples_df[samples_df["is_double_mutant"] == True]
        double_mutant_df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)
        log.info(f"Saved {len(double_mutant_df)} double mutant samples to preprocess data.csv")

        log.info(f"Processed {len(double_mutant_samples)} double mutant experiments")

    def _group_technical_replicates(self, double_mutant_samples):
        """Group technical replicates by genotype using authoritative GSTF pairs.

        Matches GEO samples to authoritative GSTF pairs from supplementary file.
        Samples with the same pair of deleted genes are technical replicates.
        """
        groups = {}
        matched_count = 0
        unmatched_count = 0

        for sample_info in double_mutant_samples:
            # Extract gene names from sample title
            gene_names = sample_info["gene_names"]

            if len(gene_names) >= 2:
                # Create canonical genotype key (sorted)
                extracted_key = tuple(sorted(gene_names[:2]))

                # Try to match against authoritative pairs
                if extracted_key in self.gstf_pairs:
                    # Use the authoritative pair as the key
                    genotype_key = extracted_key
                    matched_count += 1
                else:
                    # Log unmatched sample for debugging
                    log.warning(
                        f"Sample {sample_info['geo_accession']} with genes {extracted_key} "
                        f"not found in authoritative GSTF pairs"
                    )
                    unmatched_count += 1
                    continue  # Skip unmatched samples

                if genotype_key not in groups:
                    groups[genotype_key] = []
                groups[genotype_key].append(sample_info)

        log.info(
            f"Matched {matched_count} samples to authoritative pairs, "
            f"{unmatched_count} unmatched"
        )
        log.info(f"Found {len(groups)} unique genotypes")

        return list(groups.values())

    def _process_sequential(self, double_mutant_groups, probe_to_gene_map):
        """Process double mutant replicate groups sequentially."""
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(5e12))

        idx = 0
        with env.begin(write=True) as txn:
            for replicate_group in tqdm(double_mutant_groups):
                # Use first sample for metadata
                sample_info = replicate_group[0]

                if (
                    not sample_info["is_double_mutant"]
                    or len(sample_info["gene_names"]) < 2
                ):
                    continue

                # Extract and average data from all replicates
                all_mutant_data = []
                all_refpool_data = []
                all_log2_ratio_data = []

                for rep_info in replicate_group:
                    gsm = rep_info["gsm_object"]
                    mutant_data, refpool_data, log2_ratio_data = (
                        self._extract_expression_from_gsm(gsm, probe_to_gene_map)
                    )

                    if mutant_data and refpool_data and log2_ratio_data:
                        all_mutant_data.append(mutant_data)
                        all_refpool_data.append(refpool_data)
                        all_log2_ratio_data.append(log2_ratio_data)

                # Skip if no data from any replicate
                if not all_mutant_data:
                    log.warning(f"No data for {sample_info['gene_names']}, skipping...")
                    continue

                # Calculate means and stds across replicates
                mutant_mean, mutant_std = self._calculate_mean_std(all_mutant_data)
                refpool_mean, refpool_std = self._calculate_mean_std(all_refpool_data)
                log2_mean, log2_std = self._calculate_mean_std(all_log2_ratio_data)

                # Extract strain for this genotype from gstf_pairs
                gene_names = sample_info["gene_names"]
                genotype_key = tuple(
                    sorted([gene_names[0].upper(), gene_names[1].upper()])
                )
                strain = self.gstf_pairs.get(genotype_key, {}).get("strain", "BY4742")

                experiment, reference, publication = (
                    self.create_double_mutant_expression_experiment(
                        self.name,
                        sample_info,
                        mutant_mean,
                        refpool_mean,
                        log2_mean,
                        mutant_std,
                        refpool_std,
                        log2_std,
                        strain=strain,  # NEW: Pass per-sample strain
                    )
                )

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
        log.info(f"Wrote {idx} double mutant experiments to LMDB")

    def _calculate_mean_std(self, data_list):
        """Calculate mean and std across technical replicates.

        Args:
            data_list: List of SortedDict objects containing gene -> value mappings

        Returns:
            tuple: (mean_dict, std_dict) - SortedDict objects with means and stds
        """
        if not data_list:
            return SortedDict(), SortedDict()

        # Collect all values per gene across replicates
        all_values = {}
        for data_dict in data_list:
            for gene, value in data_dict.items():
                if gene not in all_values:
                    all_values[gene] = []
                all_values[gene].append(value)

        # Calculate mean and std
        mean_dict = SortedDict()
        std_dict = SortedDict()

        for gene, values in all_values.items():
            mean_dict[gene] = np.mean(values)
            if len(values) > 1:
                std_dict[gene] = np.std(values, ddof=1)
            else:
                std_dict[gene] = np.nan

        return mean_dict, std_dict

    def _process_parallel(self, double_mutant_groups, probe_to_gene_map):
        """Process double mutant replicate groups in parallel."""
        # Create batches
        batches = []
        for i in range(0, len(double_mutant_groups), self.batch_size):
            batch = double_mutant_groups[i : i + self.batch_size]
            batches.append(batch)

        log.info(f"Created {len(batches)} batches of size {self.batch_size}")

        # Process batches in parallel
        all_results = []
        with ProcessPoolExecutor(max_workers=self.process_workers) as executor:
            futures = []
            for batch in batches:
                future = executor.submit(
                    self._process_batch,
                    batch,
                    probe_to_gene_map,
                    self.name,
                    self.gstf_pairs,  # NEW: Pass gstf_pairs for strain extraction
                )
                futures.append(future)

            # Collect results with progress bar
            for future in tqdm(futures, desc="Processing batches"):
                batch_results = future.result()
                all_results.extend(batch_results)

        # Write all results to LMDB
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(5e12))

        written_count = 0
        with env.begin(write=True) as txn:
            for idx, serialized_data in enumerate(all_results):
                if serialized_data is not None:
                    txn.put(f"{written_count}".encode(), serialized_data)
                    written_count += 1

        env.close()
        log.info(f"Wrote {written_count} double mutant experiments to LMDB")

    @staticmethod
    def _process_batch(
        batch_items,
        probe_to_gene_map,
        dataset_name,
        gstf_pairs,  # NEW: Pass gstf_pairs for strain extraction
    ):
        """Process a batch of replicate groups. Static method for multiprocessing."""
        results = []

        for replicate_group in batch_items:
            # Use first sample for metadata
            sample_info = replicate_group[0]

            if (
                not sample_info["is_double_mutant"]
                or len(sample_info["gene_names"]) < 2
            ):
                continue

            # Extract and average data from all replicates
            all_mutant_data = []
            all_refpool_data = []
            all_log2_ratio_data = []

            for rep_info in replicate_group:
                gsm = rep_info["gsm_object"]
                mutant_data, refpool_data, log2_ratio_data = (
                    DmMicroarraySameith2015Dataset._extract_expression_from_gsm_static(
                        gsm, probe_to_gene_map
                    )
                )

                if mutant_data and refpool_data and log2_ratio_data:
                    all_mutant_data.append(mutant_data)
                    all_refpool_data.append(refpool_data)
                    all_log2_ratio_data.append(log2_ratio_data)

            # Skip if no data from any replicate
            if not all_mutant_data:
                continue

            # Calculate means and stds across replicates
            mutant_mean, mutant_std = (
                DmMicroarraySameith2015Dataset._calculate_mean_std_static(
                    all_mutant_data
                )
            )
            refpool_mean, refpool_std = (
                DmMicroarraySameith2015Dataset._calculate_mean_std_static(
                    all_refpool_data
                )
            )
            log2_mean, log2_std = (
                DmMicroarraySameith2015Dataset._calculate_mean_std_static(
                    all_log2_ratio_data
                )
            )

            # Extract strain for this genotype from gstf_pairs
            gene_names = sample_info["gene_names"]
            genotype_key = tuple(sorted([gene_names[0].upper(), gene_names[1].upper()]))
            strain = gstf_pairs.get(genotype_key, {}).get("strain", "BY4742")

            experiment, reference, publication = (
                DmMicroarraySameith2015Dataset.create_double_mutant_expression_experiment(
                    dataset_name,
                    sample_info,
                    mutant_mean,
                    refpool_mean,
                    log2_mean,
                    mutant_std,
                    refpool_std,
                    log2_std,
                    strain=strain,  # NEW: Pass per-sample strain
                )
            )

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
    def _calculate_mean_std_static(data_list):
        """Static version of _calculate_mean_std for multiprocessing."""
        if not data_list:
            return SortedDict(), SortedDict()

        # Collect all values per gene across replicates
        all_values = {}
        for data_dict in data_list:
            for gene, value in data_dict.items():
                if gene not in all_values:
                    all_values[gene] = []
                all_values[gene].append(value)

        # Calculate mean and std
        mean_dict = SortedDict()
        std_dict = SortedDict()

        for gene, values in all_values.items():
            mean_dict[gene] = np.mean(values)
            if len(values) > 1:
                std_dict[gene] = np.std(values, ddof=1)
            else:
                std_dict[gene] = np.nan

        return mean_dict, std_dict

    def _extract_gene_names_from_title(self, title: str) -> list[str]:
        """Extract systematic gene names from sample title."""
        gene_names = []

        # Look for systematic names (e.g., YAL001C or YBR089C-A)
        # Match the full systematic name, including optional suffix like -A
        systematic_pattern = r"\b(Y[A-P][LR]\d{3}[WC](?:-[A-Z])?)\b"
        matches = re.findall(systematic_pattern, title.upper())

        # Validate each match
        for match in matches:
            if self._is_valid_systematic_name(match):
                gene_names.append(match)

        # Also try to extract common names (even if we found systematic names)
        # This handles cases like "rpn4-del+ydr026c-del" where one is common and one is systematic
        if len(gene_names) < 2:
            # Extract potential gene names (all caps words)
            common_pattern = r"\b([A-Z][A-Z0-9]{2,})\b"
            potential_genes = re.findall(common_pattern, title.upper())

            for gene in potential_genes:
                # Skip common non-gene words
                if gene in ["DEL", "WT", "WILDTYPE", "MUTANT", "DOUBLE"]:
                    continue

                # Try to convert to systematic
                systematic = self._convert_to_systematic(gene)
                if systematic and self._is_valid_systematic_name(systematic):
                    # Avoid duplicates
                    if systematic not in gene_names:
                        gene_names.append(systematic)
                        # Stop if we have 2 genes
                        if len(gene_names) >= 2:
                            break

        # Log if we couldn't extract valid gene names
        if len(gene_names) < 2 and not hasattr(self, "_logged_extraction_warning"):
            log.warning(f"Could not extract 2 valid gene names from title: {title}")
            log.warning(f"Extracted: {gene_names}")
            self._logged_extraction_warning = True

        return gene_names[:2]  # Return at most 2 genes for double mutants

    def _is_valid_systematic_name(self, name: str) -> bool:
        """Validate that a systematic name matches the expected format."""
        if not name:
            return False
        # Must match exact pattern: Y + chromosome letter + L/R + 3 digits + C/W (with optional suffix like -A)
        return bool(re.match(r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?$", name.upper()))

    def _convert_to_systematic(self, gene_name: str) -> str | None:
        """Convert common gene name to systematic name."""
        if not gene_name:
            return None

        gene_upper = gene_name.upper()

        # Check if already systematic
        if re.match(r"Y[A-P][LR]\d{3}[WC](-[A-Z])?", gene_upper):
            return gene_upper

        # Try genome's gene_attribute_table
        if hasattr(self.genome, "gene_attribute_table"):
            df = self.genome.gene_attribute_table

            # Check gene column
            matches = df[df["gene"] == gene_upper]
            if not matches.empty:
                return matches.iloc[0]["ID"]

            # Check Alias column
            matches = df[df["Alias"] == gene_upper]
            if not matches.empty:
                return matches.iloc[0]["ID"]

        # Try alias_to_systematic
        if hasattr(self.genome, "alias_to_systematic"):
            candidates = self.genome.alias_to_systematic.get(gene_upper, [])
            if candidates:
                return candidates[0]  # Return first match

        return None

    def _extract_probe_to_gene_mapping(self, gse) -> dict:
        """Extract probe ID to gene name mapping from GEO platform annotation."""
        probe_to_gene = {}

        if not hasattr(gse, "gpls") or not gse.gpls:
            log.warning("No platform annotation found in GEO dataset")
            return probe_to_gene

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
                        probe_id = str(int(row[id_column]))
                        gene_name = str(row[gene_column])

                        # Clean and validate gene name
                        if gene_name and gene_name != "nan" and gene_name != "":
                            gene_name_upper = gene_name.upper()
                            if re.match(r"Y[A-Z]{2}\d{3}[CW]", gene_name_upper):
                                probe_to_gene[probe_id] = gene_name_upper
                            elif re.match(r"Q\d{4}", gene_name_upper):
                                probe_to_gene[probe_id] = gene_name_upper
                            else:
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

    def _extract_expression_from_gsm(
        self, gsm, probe_to_gene_map=None
    ) -> tuple[SortedDict, SortedDict, SortedDict]:
        """Extract expression values and log2 ratios from a GSM object.

        Handles dye swaps between technical replicates by checking metadata to determine
        which channel contains the mutant vs. reference pool.

        Returns:
            tuple: (mutant_data, refpool_data, log2_ratio_data)
                - mutant_data: Expression values for the deletion mutant
                - refpool_data: Expression values for the reference pool
                - log2_ratio_data: log2(mutant/refpool) ratios
        """
        mutant_data = SortedDict()
        refpool_data = SortedDict()
        log2_ratio_data = SortedDict()

        if hasattr(gsm, "table") and gsm.table is not None:
            table = gsm.table

            # Log available columns for the first sample
            if not hasattr(self, "_logged_columns"):
                log.info(f"Available columns in GSM table: {list(table.columns)}")
                self._logged_columns = True

            if "ID_REF" not in table.columns:
                return mutant_data, refpool_data, log2_ratio_data

            # Check for required columns
            has_cy5 = "Signal Norm_Cy5" in table.columns
            has_cy3 = "Signal Norm_Cy3" in table.columns
            has_value = "VALUE" in table.columns

            if not (has_cy5 and has_cy3 and has_value):
                log.warning(
                    f"Missing required columns. Available: {list(table.columns)}"
                )
                return mutant_data, refpool_data, log2_ratio_data

            # Determine which channel has mutant vs refpool by checking metadata
            # Due to dye swaps, this varies between technical replicates
            source_ch1 = (
                gsm.metadata.get("source_name_ch1", [""])[0]
                if hasattr(gsm, "metadata")
                else ""
            )
            source_ch2 = (
                gsm.metadata.get("source_name_ch2", [""])[0]
                if hasattr(gsm, "metadata")
                else ""
            )
            label_ch1 = (
                gsm.metadata.get("label_ch1", [""])[0]
                if hasattr(gsm, "metadata")
                else ""
            )
            label_ch2 = (
                gsm.metadata.get("label_ch2", [""])[0]
                if hasattr(gsm, "metadata")
                else ""
            )

            # Determine which dye (Cy5/Cy3) corresponds to which sample (mutant/refpool)
            # Ch1 = Cy5, Ch2 = Cy3 (based on GEO data structure)
            if "refpool" in source_ch1.lower():
                # Cy5 = refpool, Cy3 = mutant
                mutant_channel = "Cy3"
                refpool_channel = "Cy5"
                # VALUE is log2(Cy5/Cy3) = log2(refpool/mutant), so negate it
                ratio_sign = -1
            else:
                # Cy5 = mutant, Cy3 = refpool
                mutant_channel = "Cy5"
                refpool_channel = "Cy3"
                # VALUE is log2(Cy5/Cy3) = log2(mutant/refpool), correct sign
                ratio_sign = 1

            for _, row in table.iterrows():
                probe_id = str(int(row["ID_REF"]))

                # Map probe ID to gene name
                if probe_to_gene_map and probe_id in probe_to_gene_map:
                    gene = probe_to_gene_map[probe_id]
                    try:
                        cy5_value = float(row["Signal Norm_Cy5"])
                        cy3_value = float(row["Signal Norm_Cy3"])
                        value = float(row["VALUE"])

                        # Assign to mutant/refpool based on channel assignment
                        if mutant_channel == "Cy5":
                            mutant_data[gene] = cy5_value
                            refpool_data[gene] = cy3_value
                        else:
                            mutant_data[gene] = cy3_value
                            refpool_data[gene] = cy5_value

                        # Adjust log2 ratio sign to always be log2(mutant/refpool)
                        log2_ratio_data[gene] = ratio_sign * value

                    except (ValueError, TypeError):
                        continue

        return mutant_data, refpool_data, log2_ratio_data

    @staticmethod
    def _extract_expression_from_gsm_static(
        gsm, probe_to_gene_map=None
    ) -> tuple[SortedDict, SortedDict, SortedDict]:
        """Static version of _extract_expression_from_gsm for multiprocessing.

        Handles dye swaps between technical replicates.

        Returns:
            tuple: (mutant_data, refpool_data, log2_ratio_data)
        """
        mutant_data = SortedDict()
        refpool_data = SortedDict()
        log2_ratio_data = SortedDict()

        if hasattr(gsm, "table") and gsm.table is not None:
            table = gsm.table

            if "ID_REF" not in table.columns:
                return mutant_data, refpool_data, log2_ratio_data

            has_cy5 = "Signal Norm_Cy5" in table.columns
            has_cy3 = "Signal Norm_Cy3" in table.columns
            has_value = "VALUE" in table.columns

            if not (has_cy5 and has_cy3 and has_value):
                return mutant_data, refpool_data, log2_ratio_data

            # Determine which channel has mutant vs refpool by checking metadata
            source_ch1 = (
                gsm.metadata.get("source_name_ch1", [""])[0]
                if hasattr(gsm, "metadata")
                else ""
            )

            # Determine channel assignment and ratio sign
            if "refpool" in source_ch1.lower():
                # Cy5 = refpool, Cy3 = mutant
                mutant_channel = "Cy3"
                ratio_sign = -1
            else:
                # Cy5 = mutant, Cy3 = refpool
                mutant_channel = "Cy5"
                ratio_sign = 1

            for _, row in table.iterrows():
                probe_id = str(int(row["ID_REF"]))

                if probe_to_gene_map and probe_id in probe_to_gene_map:
                    gene = probe_to_gene_map[probe_id]
                    try:
                        cy5_value = float(row["Signal Norm_Cy5"])
                        cy3_value = float(row["Signal Norm_Cy3"])
                        value = float(row["VALUE"])

                        # Assign to mutant/refpool based on channel assignment
                        if mutant_channel == "Cy5":
                            mutant_data[gene] = cy5_value
                            refpool_data[gene] = cy3_value
                        else:
                            mutant_data[gene] = cy3_value
                            refpool_data[gene] = cy5_value

                        # Adjust log2 ratio sign to always be log2(mutant/refpool)
                        log2_ratio_data[gene] = ratio_sign * value

                    except (ValueError, TypeError):
                        continue

        return mutant_data, refpool_data, log2_ratio_data

    def _calculate_wt_reference_with_std(
        self, wt_gsm_list, probe_to_gene_map
    ) -> tuple[SortedDict, SortedDict]:
        """Calculate average wildtype expression values and std from GSM objects."""
        if len(wt_gsm_list) == 0:
            log.warning("No wildtype samples found, returning empty reference")
            return SortedDict(), SortedDict()

        log.info(f"Calculating reference from {len(wt_gsm_list)} wildtype samples")

        # Collect all expression values per gene
        all_expressions = {}

        for gsm in wt_gsm_list:
            mutant_data, refpool_data, log2_ratio_data = (
                self._extract_expression_from_gsm(gsm, probe_to_gene_map)
            )
            # Use reference pool data for wildtype reference
            for gene, value in refpool_data.items():
                if gene not in all_expressions:
                    all_expressions[gene] = []
                all_expressions[gene].append(value)

        # Calculate mean and std
        wt_mean_expression = SortedDict()
        wt_std_expression = SortedDict()

        for gene, values in all_expressions.items():
            wt_mean_expression[gene] = np.mean(values)
            wt_std_expression[gene] = np.std(values, ddof=1) if len(values) > 1 else 0.0

        log.info(f"Calculated reference for {len(wt_mean_expression)} genes")

        return wt_mean_expression, wt_std_expression

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict | None = None
    ) -> pd.DataFrame:
        """Preprocess raw data - for Sameith this is handled in process()."""
        return df

    def create_experiment(self):
        """Required by base class but not used."""
        pass

    @staticmethod
    def create_double_mutant_expression_experiment(
        dataset_name,
        sample_info,
        mutant_data,
        refpool_data,
        log2_ratio_data,
        mutant_std=None,
        refpool_std=None,
        log2_ratio_std=None,
        strain="BY4742",  # NEW: Accept per-sample strain parameter
    ):
        """Create experiment for double deletion mutant with PER-SAMPLE strain.

        Args:
            mutant_data: Expression values for deletion mutant
            refpool_data: Expression values for reference pool
            log2_ratio_data: log2(mutant/refpool) ratios
            mutant_std: Technical std for mutant expression (optional)
            refpool_std: Technical std for refpool expression (optional)
            log2_ratio_std: Technical std for log2 ratios (optional)
            strain: Strain background (BY4742 or BY4741), extracted from Excel comments
        """
        # USE PROVIDED STRAIN (not hardcoded!)
        # Extracted from Excel "comments" column: "MATa" → BY4742, "MATα" → BY4741
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain=strain
        )

        # Create genotype for double deletion mutant
        gene_names = sample_info["gene_names"]
        perturbations = []

        # First deletion (KanMX marker)
        if len(gene_names) > 0:
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=gene_names[0],
                    perturbed_gene_name=gene_names[0],
                    strain_id=f"KanMX_{gene_names[0]}",
                )
            )

        # Second deletion (NatMX marker for SGA)
        if len(gene_names) > 1:
            perturbations.append(
                SgaNatMxDeletionPerturbation(
                    systematic_gene_name=gene_names[1],
                    perturbed_gene_name=gene_names[1],
                    strain_id=f"NatMX_{gene_names[1]}",
                )
            )

        genotype = Genotype(perturbations=perturbations)

        # Environment - SC medium at 30°C
        environment = Environment(
            media=Media(name="SC", state="liquid"), temperature=Temperature(value=30)
        )
        environment_reference = environment.model_copy()

        # Use pre-computed log2 ratios
        log2_ratios = log2_ratio_data

        # Create phenotype with expression data from mutant
        phenotype = MicroarrayExpressionPhenotype(
            expression=mutant_data,
            expression_log2_ratio=log2_ratios,
            expression_technical_std=mutant_std,
            expression_log2_ratio_std=log2_ratio_std,
        )

        # Create reference phenotype from reference pool
        # Reference log2 ratios are zeros (self-referential)
        reference_log2_ratios = SortedDict()
        for gene in refpool_data:
            reference_log2_ratios[gene] = 0.0

        phenotype_reference = MicroarrayExpressionPhenotype(
            expression=refpool_data,  # Reference pool
            expression_log2_ratio=reference_log2_ratios,
            expression_technical_std=refpool_std,
            expression_log2_ratio_std=None,  # Reference is self-referential
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

        # Publication for Sameith et al. 2015
        publication = Publication(
            pubmed_id="26687005",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/26687005/",
            doi="10.1186/s12915-015-0222-5",
            doi_url="https://doi.org/10.1186/s12915-015-0222-5",
        )

        return experiment, reference, publication


def main():
    """Demonstrate both Sameith2015 datasets (single and double mutants)."""

    print("=" * 80)
    print("SAMEITH2015 MICROARRAY EXPRESSION DATASETS DEMO")
    print("=" * 80)

    # Double Mutant Dataset
    print("\n" + "=" * 80)
    print("1. DOUBLE MUTANT EXPRESSION DATASET (DmMicroarraySameith2015Dataset)")
    print("=" * 80)

    dm_dataset = DmMicroarraySameith2015Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/dm_microarray_sameith2015"),
        io_workers=10,
        process_workers=0,  # Use sequential for demo
    )
    print(f"\nDataset loaded successfully")
    print(f"  Size: {len(dm_dataset)} double mutant genotypes")
    print(f"  Gene set size: {len(dm_dataset.gene_set)} unique genes")
    print(f"  First 10 genes: {list(dm_dataset.gene_set)[:10]}")

    if len(dm_dataset) > 0:
        data = dm_dataset[0]
        experiment = data["experiment"]
        reference = data["reference"]

        print(f"\n--- Example: First double mutant ---")
        perturbations = experiment["genotype"]["perturbations"]
        print(
            f"  Genotype: {perturbations[0]['systematic_gene_name']} × {perturbations[1]['systematic_gene_name']}"
        )
        print(f"  Strain: {reference['genome_reference']['strain']}")
        print(
            f"  Expression measurements: {len(experiment['phenotype']['expression'])} genes"
        )

        # Check if technical std is available
        exp_std = experiment["phenotype"].get("expression_technical_std")
        if exp_std:
            non_nan_count = sum(1 for v in exp_std.values() if not np.isnan(v))
            print(
                f"  Technical replicates: {non_nan_count}/{len(exp_std)} genes have std"
            )

    # Single Mutant Dataset
    print("\n" + "=" * 80)
    print("2. SINGLE MUTANT EXPRESSION DATASET (SmMicroarraySameith2015Dataset)")
    print("=" * 80)

    sm_dataset = SmMicroarraySameith2015Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/sm_microarray_sameith2015"),
        io_workers=10,
        process_workers=0,  # Use sequential for demo
    )
    print(f"\nDataset loaded successfully")
    print(f"  Size: {len(sm_dataset)} single mutant genotypes")
    print(f"  Gene set size: {len(sm_dataset.gene_set)} unique genes")
    print(f"  First 10 genes: {list(sm_dataset.gene_set)[:10]}")

    if len(sm_dataset) > 0:
        data = sm_dataset[0]
        experiment = data["experiment"]
        reference = data["reference"]

        print(f"\n--- Example: First single mutant ---")
        perturbations = experiment["genotype"]["perturbations"]
        print(f"  Genotype: {perturbations[0]['systematic_gene_name']} deletion")
        print(f"  Strain: {reference['genome_reference']['strain']}")
        print(
            f"  Expression measurements: {len(experiment['phenotype']['expression'])} genes"
        )

        # Check if technical std is available
        exp_std = experiment["phenotype"].get("expression_technical_std")
        if exp_std:
            non_nan_count = sum(1 for v in exp_std.values() if not np.isnan(v))
            print(
                f"  Technical replicates: {non_nan_count}/{len(exp_std)} genes have std"
            )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        f"Double mutants: {len(dm_dataset)} genotypes (per-sample strain extraction working)"
    )
    print(f"Single mutants: {len(sm_dataset)} genotypes (BY4742 deletion library)")
    print(f"Total genotypes: {len(dm_dataset) + len(sm_dataset)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
