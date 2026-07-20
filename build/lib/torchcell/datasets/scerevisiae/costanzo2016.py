"""Costanzo 2016 single/double mutant fitness and interaction datasets."""

# torchcell/datasets/scerevisiae/costanzo2016
# [[torchcell.datasets.scerevisiae.costanzo2016]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/costanzo2016
# Test file: tests/torchcell/datasets/scerevisiae/test_costanzo2016.py
import logging
import os
import os.path as osp
import shutil
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import lmdb
import pandas as pd
from torch_geometric.data import download_url
from tqdm import tqdm

from torchcell.data import ExperimentDataset, post_process
from torchcell.datamodels import (
    Environment,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    GeneInteractionExperiment,
    GeneInteractionExperimentReference,
    GeneInteractionPhenotype,
    Genotype,
    Publication,
    ReferenceGenome,
    SgaDampPerturbation,
    SgaKanMxDeletionPerturbation,
    SgaNatMxDeletionPerturbation,
    SgaSuppressorAllelePerturbation,
    SgaTsAllelePerturbation,
    Temperature,
)
from torchcell.datamodels.media import SGA_DM_SELECTION
from torchcell.datamodels.schema import (
    GenePerturbationType,
    SampleUnit,
    UncertaintyType,
)
from torchcell.datasets.dataset_registry import register_dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ============================================================================
# Sample Size Metadata - Extracted from Costanzo et al. 2016
# ============================================================================

# Default sample size for double mutant fitness measurements (technical replicates)
# Quote: "All screens were conducted a single time with 4 replicate colonies
#         per double mutant, unless otherwise indicated"
# Source: SI-costanzoGlobalGeneticInteraction2016.mmd, Line 74
# Verified: SI-costanzoGlobalGeneticInteraction2016.pdf, Page 5,
#           Section "SGA query strain construction and screening"
# Date extracted: 2026-01-20
N_SAMPLES_DOUBLE_MUTANT = 4

# Sample size for query strain single mutant fitness (control screens)
# Quote: "Colony size measurements of SGA deletion and TS query mutant strains
#         were based on an average of 17 replicate control screens performed
#         at 26°C or 30°C."
# Source: SI-costanzoGlobalGeneticInteraction2016.mmd, Line 88
# Verified: SI-costanzoGlobalGeneticInteraction2016.pdf, Page 5,
#           Section "Single mutant fitness standard"
# Date extracted: 2026-01-20
# Note: colonies within a screen are averaged BEFORE bootstrap resampling, so the
# SMF resampling unit (and n_samples) is the screen, not the 68 colonies.
N_SAMPLES_QUERY_SMF_SCREENS = 17

# Sample size for array strain single mutant fitness (control screens)
# Quote: "Colony size measurements of SGA deletion and TS array mutant strains
#         were based on an average of 350 replicate control screens performed
#         at 26°C or 30°C."
# Source: SI-costanzoGlobalGeneticInteraction2016.mmd, Line 88
# Verified: SI-costanzoGlobalGeneticInteraction2016.pdf, Page 5,
#           Section "Single mutant fitness standard"
# Date extracted: 2026-01-20
# Array-strain SMF uses 350 control screens (kept for provenance; the simplified
# loader records the query-strain screen count for SMF).
N_SAMPLES_ARRAY_SMF_SCREENS = 350

# Temperature-specific measurements (all use 4 technical replicates per screen)
# Quote: "Double mutant selection plates involving a nonessential deletion
#         mutant query strain and the DMA were incubated at 30°C."
# Quote: "All SGA selection steps involving a TS allele were conducted at
#         permissive temperature (22°C) except for the final selection of
#         haploid double mutants, which were incubated at a semipermissive
#         temperature (26°C) prior to imaging."
# Source: SI-costanzoGlobalGeneticInteraction2016.mmd, Line 74
# Verified: SI-costanzoGlobalGeneticInteraction2016.pdf, Page 5
# Date extracted: 2026-01-20
N_SAMPLES_TEMP_30C = 4  # For KanMX deletion mutants
N_SAMPLES_TEMP_26C = 4  # For TS alleles at semipermissive temperature


# ============================================================================
# Fitness
@register_dataset
class SmfCostanzo2016Dataset(ExperimentDataset):
    """Single-mutant fitness dataset from Costanzo et al. 2016."""

    url = (
        "https://thecellmap.org/costanzo2016/data_files/"
        "Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip"
    )

    def __init__(
        self,
        root: str = "data/torchcell/smf_costanzo2016",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the SMF dataset under ``root`` via the base dataset."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[FitnessExperiment]:
        """Return the FitnessExperiment type for SMF data."""
        return FitnessExperiment

    @property
    def reference_class(self) -> type[FitnessExperimentReference]:
        """Return the FitnessExperimentReference type for SMF data."""
        return FitnessExperimentReference

    @property
    def raw_file_names(self) -> str:  # type: ignore[override]  # single raw file as str, base is list[str]
        """Return the single-mutant fitness raw spreadsheet name."""
        return "strain_ids_and_single_mutant_fitness.xlsx"

    def download(self) -> None:
        """Download and unpack the Costanzo archive, keeping the SMF spreadsheet."""
        path = download_url(self.url, self.raw_dir)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(path)

        # Move the contents of the subdirectory to the parent raw directory
        sub_dir = os.path.join(
            self.raw_dir,
            "Data File S1. Raw genetic interaction datasets: Pair-wise interaction format",
        )
        for filename in os.listdir(sub_dir):
            shutil.move(os.path.join(sub_dir, filename), self.raw_dir)
        os.rmdir(sub_dir)
        # remove any excess files not needed
        for file_name in os.listdir(self.raw_dir):
            # if the file name ends in .txt remove it
            if file_name.endswith(".txt"):
                os.remove(osp.join(self.raw_dir, file_name))

    @post_process
    def process(self) -> None:
        """Read the SMF spreadsheet and write experiment records into LMDB."""
        xlsx_path = osp.join(self.raw_dir, "strain_ids_and_single_mutant_fitness.xlsx")
        df = pd.read_excel(xlsx_path)
        df = self.preprocess_raw(df)
        (phenotype_reference_std_26, phenotype_reference_std_30) = (
            self.compute_phenotype_reference_std(df)
        )

        # Save preprocssed df - mainly for quick stats
        os.makedirs(self.preprocess_dir, exist_ok=True)
        df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        log.info("Processing SMF Files...")

        # Initialize LMDB environment
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))

        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                experiment, reference, publication = self.create_experiment(
                    self.name,
                    row,
                    phenotype_reference_std_26=phenotype_reference_std_26,
                    phenotype_reference_std_30=phenotype_reference_std_30,
                )

                txn.put(
                    f"{index}".encode(),
                    self._intern_record(experiment, reference, publication, itxn),
                )

        env.close()
        interned_env.close()

    def preprocess_raw(self, df: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]  # dataset-specific signature
        """Derive perturbation type from strain IDs and clean the SMF table."""
        df["Strain_ID_suffix"] = df["Strain ID"].str.split("_", expand=True)[1]

        # Determine perturbation type based on Strain_ID_suffix
        df["perturbation_type"] = df["Strain_ID_suffix"].apply(
            lambda x: (
                "damp"
                if "damp" in x
                else (
                    "temperature_sensitive"
                    if "tsa" in x or "tsq" in x
                    else (
                        "KanMX_deletion"
                        if "dma" in x
                        else (
                            "NatMX_deletion"
                            if "sn" in x  # or "S" in x or "A_S" in x
                            else "suppression_allele"
                            if "S" in x
                            else "unknown"
                        )
                    )
                )
            )
        )

        # Create separate dataframes for the two temperatures
        df_26 = df[
            [
                "Strain ID",
                "Systematic gene name",
                "Allele/Gene name",
                "Single mutant fitness (26°)",
                "Single mutant fitness (26°) stddev",
                "perturbation_type",
            ]
        ].copy()
        df_26["Temperature"] = 26

        df_30 = df[
            [
                "Strain ID",
                "Systematic gene name",
                "Allele/Gene name",
                "Single mutant fitness (30°)",
                "Single mutant fitness (30°) stddev",
                "perturbation_type",
            ]
        ].copy()
        df_30["Temperature"] = 30

        # Rename the columns for fitness and stddev to be common for both dataframes
        df_26.rename(
            columns={
                "Single mutant fitness (26°)": "Single mutant fitness",
                "Single mutant fitness (26°) stddev": "Single mutant fitness stddev",
            },
            inplace=True,
        )

        df_30.rename(
            columns={
                "Single mutant fitness (30°)": "Single mutant fitness",
                "Single mutant fitness (30°) stddev": "Single mutant fitness stddev",
            },
            inplace=True,
        )

        # Concatenate the two dataframes
        combined_df = pd.concat([df_26, df_30], ignore_index=True)
        combined_df = combined_df.dropna()
        combined_df = combined_df.drop_duplicates()
        combined_df = combined_df.reset_index(drop=True)

        return combined_df

    @staticmethod
    def compute_phenotype_reference_std(df: pd.DataFrame) -> tuple[Any, Any]:
        # reason: pandas Series scalar indexing yields dynamically-typed values
        """Return mean SMF stddev at 26C and 30C used as reference noise."""
        mean_stds = df.groupby("Temperature")["Single mutant fitness stddev"].mean()
        phenotype_reference_std_26 = mean_stds[26]
        phenotype_reference_std_30 = mean_stds[30]
        return phenotype_reference_std_26, phenotype_reference_std_30

    @staticmethod
    def create_experiment(  # type: ignore[override]  # dataset-specific signature
        dataset_name: str,
        row: pd.Series,
        phenotype_reference_std_26: Any,
        phenotype_reference_std_30: Any,
    ) -> tuple[FitnessExperiment, FitnessExperimentReference, Publication]:
        """Build SMF experiment, reference, and publication objects for a row."""
        # Common attributes for both temperatures
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="S288C"
        )

        # Deal with different types of perturbations
        if "temperature_sensitive" in row["perturbation_type"]:
            genotype = Genotype(
                perturbations=[
                    SgaTsAllelePerturbation(
                        systematic_gene_name=row["Systematic gene name"],
                        perturbed_gene_name=row["Allele/Gene name"],
                        strain_id=row["Strain ID"],
                    )
                ]
            )
        elif "damp" in row["perturbation_type"]:
            genotype = Genotype(
                perturbations=[
                    SgaDampPerturbation(
                        systematic_gene_name=row["Systematic gene name"],
                        perturbed_gene_name=row["Allele/Gene name"],
                        strain_id=row["Strain ID"],
                    )
                ]
            )
        elif "KanMX_deletion" in row["perturbation_type"]:
            genotype = Genotype(
                perturbations=[
                    SgaKanMxDeletionPerturbation(
                        systematic_gene_name=row["Systematic gene name"],
                        perturbed_gene_name=row["Allele/Gene name"],
                        strain_id=row["Strain ID"],
                    )
                ]
            )
        elif "NatMX_deletion" in row["perturbation_type"]:
            genotype = Genotype(
                perturbations=[
                    SgaNatMxDeletionPerturbation(
                        systematic_gene_name=row["Systematic gene name"],
                        perturbed_gene_name=row["Allele/Gene name"],
                        strain_id=row["Strain ID"],
                    )
                ]
            )
        elif "suppression_allele" in row["perturbation_type"]:
            genotype = Genotype(
                perturbations=[
                    SgaSuppressorAllelePerturbation(
                        systematic_gene_name=row["Systematic gene name"],
                        perturbed_gene_name=row["Allele/Gene name"],
                        strain_id=row["Strain ID"],
                    )
                ]
            )

        environment = Environment(
            media=SGA_DM_SELECTION, temperature=Temperature(value=row["Temperature"])
        )
        environment_reference = environment.model_copy()
        # Phenotype based on temperature
        smf_key = "Single mutant fitness"
        smf_std_key = "Single mutant fitness stddev"

        # SMF stddev is a BOOTSTRAP standard error of the bootstrapped-mean fitness
        # estimate (SOM: "bootstrapped means ... used in variance estimation and
        # final fitness values"), so it is already an SE -> used AS-IS, never
        # divided by sqrt(n). The bootstrap resampling unit is the control screen
        # (17 for query strains); the 4 colonies per screen are averaged before
        # resampling, so n is screens, not the 68 colony measurements. fitness_se
        # is auto-derived from (uncertainty, type). See
        # [[torchcell.datasets.scerevisiae.costanzo2016.noise-computation]].
        fitness_std_val = row[smf_std_key]
        smf_unc_type = (
            UncertaintyType.bootstrap_se if fitness_std_val is not None else None
        )

        phenotype = FitnessPhenotype(
            fitness=row[smf_key],
            fitness_std=fitness_std_val,
            fitness_uncertainty=fitness_std_val,
            fitness_uncertainty_type=smf_unc_type,
            n_samples=N_SAMPLES_QUERY_SMF_SCREENS,
            sample_unit=SampleUnit.screen,
        )

        if row["Temperature"] == 26:
            phenotype_reference_std = phenotype_reference_std_26
        elif row["Temperature"] == 30:
            phenotype_reference_std = phenotype_reference_std_30

        # WT reference noise is the mean SMF stddev (an average of bootstrap SEs),
        # so it is likewise a bootstrap SE -> used as-is (no sqrt(n) division).
        ref_unc_type = (
            UncertaintyType.bootstrap_se
            if phenotype_reference_std is not None
            else None
        )

        phenotype_reference = FitnessPhenotype(
            fitness=1.0,
            fitness_std=phenotype_reference_std,
            fitness_uncertainty=phenotype_reference_std,
            fitness_uncertainty_type=ref_unc_type,
            n_samples=N_SAMPLES_QUERY_SMF_SCREENS,
            sample_unit=SampleUnit.screen,
        )

        reference = FitnessExperimentReference(
            dataset_name=dataset_name,
            genome_reference=genome_reference,
            environment_reference=environment_reference,
            phenotype_reference=phenotype_reference,
        )

        experiment = FitnessExperiment(
            dataset_name=dataset_name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )

        publication = Publication(
            pubmed_id="27708008",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/27708008/",
            doi="10.1126/science.aaf1420",
            doi_url="https://www.science.org/doi/10.1126/science.aaf1420",
        )

        return experiment, reference, publication


@register_dataset
class DmfCostanzo2016Dataset(ExperimentDataset):
    """Double-mutant fitness dataset from Costanzo et al. 2016."""

    url = (
        "https://thecellmap.org/costanzo2016/data_files/"
        "Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip"
    )

    def __init__(
        self,
        root: str = "data/torchcell/dmf_costanzo2016",
        subset_n: int | None = None,
        batch_size: int = int(1e4),
        io_workers: int = 1,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset with optional subsetting and batched IO."""
        self.io_workers = io_workers
        self.subset_n = subset_n
        self.batch_size = batch_size
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    def download(self) -> None:
        """Download and unpack the Costanzo archive, keeping the SGA text files."""
        path = download_url(self.url, self.raw_dir)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(path)

        # Move the contents of the subdirectory to the parent raw directory
        sub_dir = os.path.join(
            self.raw_dir,
            "Data File S1. Raw genetic interaction datasets: Pair-wise interaction format",
        )
        for filename in os.listdir(sub_dir):
            shutil.move(os.path.join(sub_dir, filename), self.raw_dir)
        os.rmdir(sub_dir)
        # remove any excess files not needed
        os.remove(osp.join(self.raw_dir, "strain_ids_and_single_mutant_fitness.xlsx"))

    @property
    def experiment_class(self) -> type[FitnessExperiment]:
        """Return the FitnessExperiment type for DMF data."""
        return FitnessExperiment

    @property
    def reference_class(self) -> type[FitnessExperimentReference]:
        """Return the FitnessExperimentReference type for DMF data."""
        return FitnessExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Return the SGA interaction text files used as raw input."""
        return ["SGA_DAmP.txt", "SGA_ExE.txt", "SGA_ExN_NxE.txt", "SGA_NxN.txt"]

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Concatenate and clean the SGA files into a double-mutant fitness table."""
        log.info("Preprocess on raw data...")

        # Function to extract gene name
        def extract_systematic_name(x: pd.Series) -> pd.Series:
            return cast("pd.Series", x.apply(lambda y: y.split("_")[0]))

        # Extract gene names
        df["Query Systematic Name"] = extract_systematic_name(df["Query Strain ID"])
        df["Array Systematic Name"] = extract_systematic_name(df["Array Strain ID"])
        Temperature = df["Arraytype/Temp"].str.extract(r"(\d+)").astype(int)
        df["Temperature"] = Temperature
        df["query_perturbation_type"] = df["Query Strain ID"].apply(
            lambda x: (
                "damp"
                if "damp" in x
                else (
                    "temperature_sensitive"
                    if "tsa" in x or "tsq" in x
                    else (
                        "KanMX_deletion"
                        if "dma" in x
                        else (
                            "NatMX_deletion"
                            if "sn" in x  # or "S" in x or "A_S" in x
                            else "suppression_allele"
                            if "S" in x
                            else "unknown"
                        )
                    )
                )
            )
        )
        df["array_perturbation_type"] = df["Array Strain ID"].apply(
            lambda x: (
                "damp"
                if "damp" in x
                else (
                    "temperature_sensitive"
                    if "tsa" in x or "tsq" in x
                    else (
                        "KanMX_deletion"
                        if "dma" in x
                        else (
                            "NatMX_deletion"
                            if "sn" in x  # or "S" in x or "A_S" in x
                            else "suppression_allele"
                            if "S" in x
                            else "unknown"
                        )
                    )
                )
            )
        )
        means = df.groupby("Temperature")[
            "Double mutant fitness standard deviation"
        ].mean()

        # Extracting means for specific temperatures
        self.phenotype_reference_std_26 = means.get(26, None)
        self.phenotype_reference_std_30 = means.get(30, None)

        return df

    @post_process
    def process(self) -> None:
        """Preprocess the raw SGA files and write records into LMDB in batches."""
        os.makedirs(self.preprocess_dir, exist_ok=True)

        # Initialize an empty DataFrame to hold all raw data
        df = pd.DataFrame()

        # Read and concatenate all raw files
        log.info("Reading and Concatenating Raw Files...")
        for file_name in tqdm(self.raw_file_names):
            file_path = os.path.join(self.raw_dir, file_name)
            # Reading data using Pandas; limit rows for demonstration
            df_temp = pd.read_csv(file_path, sep="\t")
            # Concatenating data frames
            df = pd.concat([df, df_temp], ignore_index=True)

        # Functions for data filtering... duplicates selection,
        df = self.preprocess_raw(df)

        # Subset
        if self.subset_n is not None:
            df = df.sample(n=self.subset_n, random_state=42).reset_index(drop=True)

        # Save preprocssed df - mainly for quick stats
        df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        log.info("Processing DMF Files...")

        # Initialize LMDB environment
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))

        # Create a ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.io_workers) as executor:
            futures = []
            for batch_start in range(0, df.shape[0], self.batch_size):
                batch_end = min(batch_start + self.batch_size, df.shape[0])
                batch_df = df.iloc[batch_start:batch_end]
                future = executor.submit(
                    self._process_batch, batch_df, env, interned_env
                )
                futures.append(future)

            # Wait for all futures to complete
            for future in tqdm(futures, total=len(futures)):
                future.result()

        env.close()
        interned_env.close()

    def _process_batch(
        self, batch_df: pd.DataFrame, env: lmdb.Environment, interned_env: Any
    ) -> None:
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for index, row in batch_df.iterrows():
                experiment, reference, publication = self.create_experiment(
                    self.name,
                    row,
                    phenotype_reference_std_26=self.phenotype_reference_std_26,
                    phenotype_reference_std_30=self.phenotype_reference_std_30,
                )

                txn.put(
                    f"{index}".encode(),
                    self._intern_record(experiment, reference, publication, itxn),
                )

    @staticmethod
    def create_experiment(  # type: ignore[override]  # dataset-specific signature
        dataset_name: str,
        row: pd.Series,
        phenotype_reference_std_26: Any,
        phenotype_reference_std_30: Any,
    ) -> tuple[FitnessExperiment, FitnessExperimentReference, Publication]:
        """Build DMF experiment, reference, and publication objects for a row."""
        # Common attributes for both temperatures
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="S288C"
        )
        # genotype
        perturbations: list[GenePerturbationType] = []
        # Query
        if "temperature_sensitive" in row["query_perturbation_type"]:
            perturbations.append(
                SgaTsAllelePerturbation(
                    systematic_gene_name=row["Query Systematic Name"],
                    perturbed_gene_name=row["Query allele name"],
                    strain_id=row["Query Strain ID"],
                )
            )
        elif "damp" in row["query_perturbation_type"]:
            perturbations.append(
                SgaDampPerturbation(
                    systematic_gene_name=row["Query Systematic Name"],
                    perturbed_gene_name=row["Query allele name"],
                    strain_id=row["Query Strain ID"],
                )
            )
        elif "KanMX_deletion" in row["query_perturbation_type"]:
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Query Systematic Name"],
                    perturbed_gene_name=row["Query allele name"],
                    strain_id=row["Query Strain ID"],
                )
            )

        elif "NatMX_deletion" in row["query_perturbation_type"]:
            perturbations.append(
                SgaNatMxDeletionPerturbation(
                    systematic_gene_name=row["Query Systematic Name"],
                    perturbed_gene_name=row["Query allele name"],
                    strain_id=row["Query Strain ID"],
                )
            )
        elif "suppression_allele" in row["query_perturbation_type"]:
            perturbations.append(
                SgaSuppressorAllelePerturbation(
                    systematic_gene_name=row["Query Systematic Name"],
                    perturbed_gene_name=row["Query allele name"],
                    strain_id=row["Query Strain ID"],
                )
            )

        # Array
        if "temperature_sensitive" in row["array_perturbation_type"]:
            perturbations.append(
                SgaTsAllelePerturbation(
                    systematic_gene_name=row["Array Systematic Name"],
                    perturbed_gene_name=row["Array allele name"],
                    strain_id=row["Array Strain ID"],
                )
            )
        elif "damp" in row["array_perturbation_type"]:
            perturbations.append(
                SgaDampPerturbation(
                    systematic_gene_name=row["Array Systematic Name"],
                    perturbed_gene_name=row["Array allele name"],
                    strain_id=row["Array Strain ID"],
                )
            )
        elif "KanMX_deletion" in row["array_perturbation_type"]:
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Array Systematic Name"],
                    perturbed_gene_name=row["Array allele name"],
                    strain_id=row["Array Strain ID"],
                )
            )

        elif "NatMX_deletion" in row["array_perturbation_type"]:
            perturbations.append(
                SgaNatMxDeletionPerturbation(
                    systematic_gene_name=row["Array Systematic Name"],
                    perturbed_gene_name=row["Array allele name"],
                    strain_id=row["Array Strain ID"],
                )
            )

        elif "suppression_allele" in row["array_perturbation_type"]:
            perturbations.append(
                SgaSuppressorAllelePerturbation(
                    systematic_gene_name=row["Array Systematic Name"],
                    perturbed_gene_name=row["Array allele name"],
                    strain_id=row["Array Strain ID"],
                )
            )
        genotype = Genotype(perturbations=perturbations)
        # genotype
        environment = Environment(
            media=SGA_DM_SELECTION, temperature=Temperature(value=row["Temperature"])
        )
        environment_reference = environment.model_copy()
        # Phenotype based on temperature
        dmf_key = "Double mutant fitness"
        dmf_std_key = "Double mutant fitness standard deviation"

        # DMF stddev is a SAMPLE SD over the 4 colony replicates of one screen
        # (SOM Data File S1 "Double mutant fitness standard deviation"), NOT a
        # bootstrap SE -> SE = SD/sqrt(4), auto-derived from (uncertainty, type).
        fitness_std_val = row[dmf_std_key]
        dmf_unc_type = (
            UncertaintyType.sample_sd if fitness_std_val is not None else None
        )

        phenotype = FitnessPhenotype(
            fitness=row[dmf_key],
            fitness_std=fitness_std_val,
            fitness_uncertainty=fitness_std_val,
            fitness_uncertainty_type=dmf_unc_type,
            n_samples=N_SAMPLES_DOUBLE_MUTANT,
            sample_unit=SampleUnit.colony,
        )

        if row["Temperature"] == 26:
            phenotype_reference_std = phenotype_reference_std_26
        elif row["Temperature"] == 30:
            phenotype_reference_std = phenotype_reference_std_30

        # WT/WT reference noise is the mean DMF stddev (an average of colony sample
        # SDs) -> sample_sd over the 4 colonies -> SE = SD/sqrt(4). (Was wrongly
        # divided by sqrt(68), conflating screens with colonies.)
        ref_unc_type = (
            UncertaintyType.sample_sd if phenotype_reference_std is not None else None
        )

        phenotype_reference = FitnessPhenotype(
            fitness=1.0,
            fitness_std=phenotype_reference_std,
            fitness_uncertainty=phenotype_reference_std,
            fitness_uncertainty_type=ref_unc_type,
            n_samples=N_SAMPLES_DOUBLE_MUTANT,
            sample_unit=SampleUnit.colony,
        )

        reference = FitnessExperimentReference(
            dataset_name=dataset_name,
            genome_reference=genome_reference,
            environment_reference=environment_reference,
            phenotype_reference=phenotype_reference,
        )

        experiment = FitnessExperiment(
            dataset_name=dataset_name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )

        publication = Publication(
            pubmed_id="27708008",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/27708008/",
            doi="10.1126/science.aaf1420",
            doi_url="https://www.science.org/doi/10.1126/science.aaf1420",
        )

        return experiment, reference, publication


# Interactions
@register_dataset
class DmiCostanzo2016Dataset(ExperimentDataset):
    """Double-mutant genetic interaction dataset from Costanzo et al. 2016."""

    url = (
        "https://thecellmap.org/costanzo2016/data_files/"
        "Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip"
    )

    def __init__(
        self,
        root: str = "data/torchcell/dmi_costanzo2016",
        subset_n: int | None = None,
        batch_size: int = int(1e4),
        io_workers: int = 1,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset with optional subsetting and batched IO."""
        self.io_workers = io_workers
        self.subset_n = subset_n
        self.batch_size = batch_size
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    def download(self) -> None:
        """Download and unpack the Costanzo archive, keeping the SGA text files."""
        path = download_url(self.url, self.raw_dir)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(path)

        # Move the contents of the subdirectory to the parent raw directory
        sub_dir = os.path.join(
            self.raw_dir,
            "Data File S1. Raw genetic interaction datasets: Pair-wise interaction format",
        )
        for filename in os.listdir(sub_dir):
            shutil.move(os.path.join(sub_dir, filename), self.raw_dir)
        os.rmdir(sub_dir)
        # remove any excess files not needed
        os.remove(osp.join(self.raw_dir, "strain_ids_and_single_mutant_fitness.xlsx"))

    @property
    def experiment_class(self) -> type[GeneInteractionExperiment]:
        """Return the GeneInteractionExperiment type for DMI data."""
        return GeneInteractionExperiment

    @property
    def reference_class(self) -> type[GeneInteractionExperimentReference]:
        """Return the GeneInteractionExperimentReference type for DMI data."""
        return GeneInteractionExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Return the SGA interaction text files used as raw input."""
        return ["SGA_DAmP.txt", "SGA_ExE.txt", "SGA_ExN_NxE.txt", "SGA_NxN.txt"]

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Concatenate and clean the SGA files into a genetic-interaction table."""
        log.info("Preprocess on raw data...")

        # Function to extract gene name
        def extract_systematic_name(x: pd.Series) -> pd.Series:
            return cast("pd.Series", x.apply(lambda y: y.split("_")[0]))

        # Extract gene names
        df["Query Systematic Name"] = extract_systematic_name(df["Query Strain ID"])
        df["Array Systematic Name"] = extract_systematic_name(df["Array Strain ID"])
        Temperature = df["Arraytype/Temp"].str.extract(r"(\d+)").astype(int)
        df["Temperature"] = Temperature
        df["query_perturbation_type"] = df["Query Strain ID"].apply(
            lambda x: (
                "damp"
                if "damp" in x
                else (
                    "temperature_sensitive"
                    if "tsa" in x or "tsq" in x
                    else (
                        "KanMX_deletion"
                        if "dma" in x
                        else (
                            "NatMX_deletion"
                            if "sn" in x
                            else "suppression_allele"
                            if "S" in x
                            else "unknown"
                        )
                    )
                )
            )
        )
        df["array_perturbation_type"] = df["Array Strain ID"].apply(
            lambda x: (
                "damp"
                if "damp" in x
                else (
                    "temperature_sensitive"
                    if "tsa" in x or "tsq" in x
                    else (
                        "KanMX_deletion"
                        if "dma" in x
                        else (
                            "NatMX_deletion"
                            if "sn" in x
                            else "suppression_allele"
                            if "S" in x
                            else "unknown"
                        )
                    )
                )
            )
        )

        return df

    @post_process
    def process(self) -> None:
        """Preprocess the raw SGA files and write records into LMDB in batches."""
        os.makedirs(self.preprocess_dir, exist_ok=True)

        # Initialize an empty DataFrame to hold all raw data
        df = pd.DataFrame()

        # Read and concatenate all raw files
        log.info("Reading and Concatenating Raw Files...")
        for file_name in tqdm(self.raw_file_names):
            file_path = os.path.join(self.raw_dir, file_name)
            # Reading data using Pandas; limit rows for demonstration
            df_temp = pd.read_csv(file_path, sep="\t")
            # Concatenating data frames
            df = pd.concat([df, df_temp], ignore_index=True)

        # Functions for data filtering... duplicates selection,
        df = self.preprocess_raw(df)

        # Subset
        if self.subset_n is not None:
            df = df.sample(n=self.subset_n, random_state=42).reset_index(drop=True)

        # Save preprocssed df - mainly for quick stats
        df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        log.info("Processing DMI Files...")

        # Initialize LMDB environment
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))

        # Create a ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.io_workers) as executor:
            futures = []
            for batch_start in range(0, df.shape[0], self.batch_size):
                batch_end = min(batch_start + self.batch_size, df.shape[0])
                batch_df = df.iloc[batch_start:batch_end]
                future = executor.submit(
                    self._process_batch, batch_df, env, interned_env
                )
                futures.append(future)

            # Wait for all futures to complete
            for future in tqdm(futures, total=len(futures)):
                future.result()

        env.close()
        interned_env.close()

    def _process_batch(
        self, batch_df: pd.DataFrame, env: lmdb.Environment, interned_env: Any
    ) -> None:
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for index, row in batch_df.iterrows():
                experiment, reference, publication = self.create_experiment(
                    self.name, row
                )

                txn.put(
                    f"{index}".encode(),
                    self._intern_record(experiment, reference, publication, itxn),
                )

    @staticmethod
    def create_experiment(  # type: ignore[override]  # dataset-specific signature
        dataset_name: str, row: pd.Series
    ) -> tuple[
        GeneInteractionExperiment, GeneInteractionExperimentReference, Publication
    ]:
        """Build DMI interaction experiment, reference, and publication objects."""
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="S288C"
        )

        perturbations: list[GenePerturbationType] = []
        # Query
        if "temperature_sensitive" in row["query_perturbation_type"]:
            perturbations.append(
                SgaTsAllelePerturbation(
                    systematic_gene_name=row["Query Systematic Name"],
                    perturbed_gene_name=row["Query allele name"],
                    strain_id=row["Query Strain ID"],
                )
            )
        elif "damp" in row["query_perturbation_type"]:
            perturbations.append(
                SgaDampPerturbation(
                    systematic_gene_name=row["Query Systematic Name"],
                    perturbed_gene_name=row["Query allele name"],
                    strain_id=row["Query Strain ID"],
                )
            )
        elif "KanMX_deletion" in row["query_perturbation_type"]:
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Query Systematic Name"],
                    perturbed_gene_name=row["Query allele name"],
                    strain_id=row["Query Strain ID"],
                )
            )
        elif "NatMX_deletion" in row["query_perturbation_type"]:
            perturbations.append(
                SgaNatMxDeletionPerturbation(
                    systematic_gene_name=row["Query Systematic Name"],
                    perturbed_gene_name=row["Query allele name"],
                    strain_id=row["Query Strain ID"],
                )
            )
        elif "suppression_allele" in row["query_perturbation_type"]:
            perturbations.append(
                SgaSuppressorAllelePerturbation(
                    systematic_gene_name=row["Query Systematic Name"],
                    perturbed_gene_name=row["Query allele name"],
                    strain_id=row["Query Strain ID"],
                )
            )

        # Array
        if "temperature_sensitive" in row["array_perturbation_type"]:
            perturbations.append(
                SgaTsAllelePerturbation(
                    systematic_gene_name=row["Array Systematic Name"],
                    perturbed_gene_name=row["Array allele name"],
                    strain_id=row["Array Strain ID"],
                )
            )
        elif "damp" in row["array_perturbation_type"]:
            perturbations.append(
                SgaDampPerturbation(
                    systematic_gene_name=row["Array Systematic Name"],
                    perturbed_gene_name=row["Array allele name"],
                    strain_id=row["Array Strain ID"],
                )
            )
        elif "KanMX_deletion" in row["array_perturbation_type"]:
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Array Systematic Name"],
                    perturbed_gene_name=row["Array allele name"],
                    strain_id=row["Array Strain ID"],
                )
            )
        elif "NatMX_deletion" in row["array_perturbation_type"]:
            perturbations.append(
                SgaNatMxDeletionPerturbation(
                    systematic_gene_name=row["Array Systematic Name"],
                    perturbed_gene_name=row["Array allele name"],
                    strain_id=row["Array Strain ID"],
                )
            )
        elif "suppression_allele" in row["array_perturbation_type"]:
            perturbations.append(
                SgaSuppressorAllelePerturbation(
                    systematic_gene_name=row["Array Systematic Name"],
                    perturbed_gene_name=row["Array allele name"],
                    strain_id=row["Array Strain ID"],
                )
            )
        genotype = Genotype(perturbations=perturbations)
        assert len(genotype) == 2, "Genotype must have 2 perturbations."

        environment = Environment(
            media=SGA_DM_SELECTION, temperature=Temperature(value=row["Temperature"])
        )
        environment_reference = environment.model_copy()

        # Digenic interaction: relates two gene nodes -> an edge (not a hyperedge,
        # which is >=3 genes). GeneInteractionPhenotype defaults to "hyperedge"
        # (the trigenic case), so digenic loaders override it explicitly.
        phenotype = GeneInteractionPhenotype(
            gene_interaction=row["Genetic interaction score (ε)"],
            gene_interaction_p_value=row["P-value"],
            graph_level="edge",
        )

        # By definition, the reference interaction would be 0.
        phenotype_reference = GeneInteractionPhenotype(
            gene_interaction=0.0, gene_interaction_p_value=None, graph_level="edge"
        )

        reference = GeneInteractionExperimentReference(
            dataset_name=dataset_name,
            genome_reference=genome_reference,
            environment_reference=environment_reference,
            phenotype_reference=phenotype_reference,
        )

        experiment = GeneInteractionExperiment(
            dataset_name=dataset_name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )

        publication = Publication(
            pubmed_id="27708008",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/27708008/",
            doi="10.1126/science.aaf1420",
            doi_url="https://www.science.org/doi/10.1126/science.aaf1420",
        )

        return experiment, reference, publication


def main() -> None:
    """Build and inspect the Costanzo 2016 datasets as a manual run."""
    from dotenv import load_dotenv

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    assert DATA_ROOT is not None, "DATA_ROOT must be set"

    # Single mutant fitness
    dataset = SmfCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/smf_costanzo2016"), io_workers=10
    )
    print(len(dataset))
    print(dataset[100])

    # Double mutant fitness
    dataset = DmfCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/dmf_costanzo2016_1e5"),
        io_workers=10,
        batch_size=int(1e4),
        subset_n=int(1e5),
    )
    print(len(dataset))
    print(dataset[0])

    dataset = DmfCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/dmf_costanzo2016_5e5"),
        io_workers=10,
        batch_size=int(1e4),
        subset_n=int(5e5),
    )
    print(len(dataset))
    print(dataset[0])

    dataset = DmfCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/dmf_costanzo2016"),
        io_workers=10,
        batch_size=int(1e4),
    )
    print(len(dataset))
    print(dataset[0])

    # Interactions
    dataset = DmiCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/dmi_costanzo2016_1e5"),
        io_workers=10,
        subset_n=int(1e5),
    )
    print(len(dataset))
    print(dataset[0])

    dataset = DmiCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/dmi_costanzo2016_5e5"),
        io_workers=10,
        subset_n=int(5e5),
    )
    print(len(dataset))
    print(dataset[0])

    dataset = DmiCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/dmi_costanzo2016"), io_workers=10
    )
    print(len(dataset))
    print(dataset[0])


if __name__ == "__main__":
    main()
