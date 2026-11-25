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

from torchcell.data import ExperimentDataset, Organism, post_process
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
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)

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
# 2026.11.24 (#1203 item 3): the loader used to stamp 17 on every SMF record. A
# record's n_samples now follows the collection its released estimate came from
# (see ``_smf_estimate_is_array_only``): 350 for an array strain (_dma, _tsa) whose
# estimate is the array collection's alone, 17 for every other record.
N_SAMPLES_ARRAY_SMF_SCREENS = 350

# How the released SMF value of a strain was combined, verbatim (si1.md line 96,
# sha256 1828703b0ff739fdf1c0d9232fe4fd81a3ce95a1b111780f55ef63bfa676880e):
#   "we treated the fitness of most single mutant array and query strains for the
#    same mutant as independent estimates and assigned the mean of the two estimates
#    to each strain; a small number (~55) of query and array strains for the same
#    mutant exhibited notably different colony sizes and thus separate fitness
#    estimates were retained for these mutants."
# The release shows the mean as an array strain and a query strain of one gene that
# carry byte-identical values in all four SMF cells: 3,113 _dma/_sn pairs and 616
# _tsa/_tsq pairs (strain_ids_and_single_mutant_fitness.xlsx, sha256
# 3b9d3351cdde8ee90832797193a1e8838be3ce6778b3761a9e57db2d3a6a9ccc), while 12 _dma/_sn
# and 45 _tsa/_tsq pairs of the same allele differ, the "~55" retained separately.
# n_samples for each record, and why:
# - array strain, no identical query twin: 350 (the array count above).
# - query strain (_sn, _tsq), no identical array twin: 17 (the query count above).
# - array/query twins (the mean of the two estimates): 17 on both strains. The source
#   states no single count for a mean of an array estimate and a query estimate, and
#   the conservative lower end (the query count) never overstates precision
#   (CLAUDE.md range rule). Flagged for review.
# - DAmP (_damp): 17. DAmP strains are query strains (si1.md line 76: "a set of query
#   mutants carrying hypomorphic, Decreased Abundance by mRNA Perturbation (labeled
#   _DAmP#) alleles"), but the SI's query count names "SGA deletion and TS query
#   mutant strains" only, so 17 is a representative, not a sourced value. Flagged.
# - TS strains are "computed separately at either 26 or 30" (line 96), while the
#   350 / 17 screens were "performed at 26°C or 30°C"; how many of them a single
#   temperature's TS estimate uses is not released, so 350 / 17 there is the SI's
#   collection count, not a per-temperature count. Flagged.
# n_samples is provenance only here: the stddev is a bootstrap SE used as-is.
SMF_ARRAY_SUFFIXES = ("dma", "tsa")
SMF_QUERY_SUFFIXES = ("sn", "tsq")
SMF_RELEASED_COLUMNS = (
    "Single mutant fitness (26°)",
    "Single mutant fitness (26°) stddev",
    "Single mutant fitness (30°)",
    "Single mutant fitness (30°) stddev",
)

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


class UnsupportedTemperatureError(ValueError):
    """A Costanzo 2016 fitness row at a temperature with no reference noise.

    The SMF spreadsheet and the DMF reference noise exist only at 26 and 30 C; the
    released Data File S1 has no other temperature (0 of 20,705,612 SGA rows, 0 of
    20,484 SMF records), so a row elsewhere is refused rather than given a reference.
    """


class BlankDoubleMutantFitnessError(ValueError):
    """A Costanzo 2016 DMF row with a blank fitness or a blank standard deviation.

    The released Data File S1 has neither (0 of 20,705,612 rows), so a blank cell is
    refused by name rather than stored as NaN or failing inside the phenotype model.
    """


def _smf_estimate_is_array_only(df: pd.DataFrame) -> pd.Series:
    """Mark the SMF strains whose released estimate is the array collection's alone.

    True for an array strain (suffix ``dma`` or ``tsa``) unless a query strain (suffix
    ``sn`` or ``tsq``) of the same systematic gene carries byte-identical values in all
    four released SMF cells, which is how the release shows the mean the authors
    assigned to both strains (see the comment above ``SMF_ARRAY_SUFFIXES``). A blank
    cell matches only a blank cell. Pairing is by gene and value, not allele name,
    because four twins carry a ``-supp`` allele name on one side only
    (``orm2-supp1`` / ``orm2``).
    """
    suffix_class = (
        df["Strain ID"].str.split("_", expand=True)[1].str.extract(r"^([A-Za-z]+)")[0]
    )
    released = (
        df[list(SMF_RELEASED_COLUMNS)]
        .astype(object)
        .where(df[list(SMF_RELEASED_COLUMNS)].notna(), "blank")
    )
    cells = released.astype(str).agg("|".join, axis=1)
    keys = df["Systematic gene name"].astype(str) + "#" + cells
    query_keys = set(keys[suffix_class.isin(SMF_QUERY_SUFFIXES)])
    is_array = suffix_class.isin(SMF_ARRAY_SUFFIXES)
    return is_array & ~keys.isin(query_keys)


def _reference_std_at(temperature: Any, std_26: Any, std_30: Any, strain: str) -> Any:
    """Return the reference noise for ``temperature``, refusing any but 26 and 30 C."""
    if temperature == 26:
        return std_26
    if temperature == 30:
        return std_30
    raise UnsupportedTemperatureError(
        f"Costanzo 2016 strain {strain} at {temperature} C: reference noise is "
        "defined only at 26 and 30 C"
    )


# ============================================================================
# Fitness
@register_dataset
class SmfCostanzo2016Dataset(ExperimentDataset):
    """Single-mutant fitness dataset from Costanzo et al. 2016.

    The release is aggregate-only: each SMF cell is the authors' bootstrapped mean over
    replicate control screens (si1.md line 94, "based on an average of 350 replicate
    control screens" for array strains and "17 replicate control screens" for query
    strains), and no per-screen value is released. One record per (strain,
    temperature) cell with a stated fitness and stddev: 20,484 records from 10,859
    strain rows (2026.11.24 per-replicate review, #1178; no record count changes).

    Open (#410, owner decision): for deletion and DAmP strains the 26 C and 30 C cells
    carry one temperature-combined value (si1.md line 96, "we combined measurements
    from different temperatures in the average for each deletion mutant"), so 8,454
    strains yield a 26 C and a 30 C record with the same value. The loader still
    emits both until the owner decides how a temperature-combined estimate is stated.
    """

    organism = Organism.scerevisiae

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

        # #1203 item 3: which collection the released estimate came from decides the
        # record's n_samples (350 array screens or 17 query screens).
        df["array_only_estimate"] = _smf_estimate_is_array_only(df)

        # Create separate dataframes for the two temperatures
        df_26 = df[
            [
                "Strain ID",
                "Systematic gene name",
                "Allele/Gene name",
                "Single mutant fitness (26°)",
                "Single mutant fitness (26°) stddev",
                "perturbation_type",
                "array_only_estimate",
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
                "array_only_estimate",
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
            species="Saccharomyces cerevisiae",
            strain="S288C",
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
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
        # (350 for an array-only estimate, 17 for every other record, see the
        # comment above SMF_ARRAY_SUFFIXES); the 4 colonies per screen are averaged
        # before resampling, so n is screens, not colony measurements. fitness_se
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
            n_samples=(
                N_SAMPLES_ARRAY_SMF_SCREENS
                if bool(row["array_only_estimate"])
                else N_SAMPLES_QUERY_SMF_SCREENS
            ),
            sample_unit=SampleUnit.screen,
        )

        phenotype_reference_std = _reference_std_at(
            row["Temperature"],
            phenotype_reference_std_26,
            phenotype_reference_std_30,
            row["Strain ID"],
        )

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

    organism = Organism.scerevisiae

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
        """Concatenate and clean the SGA files into a double-mutant fitness table.

        A blank ``Double mutant fitness`` or ``Double mutant fitness standard
        deviation`` raises ``BlankDoubleMutantFitnessError`` before any record or
        reference noise is computed.
        """
        log.info("Preprocess on raw data...")
        blank = (
            df["Double mutant fitness"].isna()
            | df["Double mutant fitness standard deviation"].isna()
        )
        if blank.any():
            first = df[blank].iloc[0]
            raise BlankDoubleMutantFitnessError(
                f"{int(blank.sum())} Costanzo 2016 DMF row(s) have a blank fitness "
                "or standard deviation (first: "
                f"{first['Query Strain ID']} x {first['Array Strain ID']} at "
                f"{first['Arraytype/Temp']})"
            )

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
            species="Saccharomyces cerevisiae",
            strain="S288C",
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
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

        phenotype_reference_std = _reference_std_at(
            row["Temperature"],
            phenotype_reference_std_26,
            phenotype_reference_std_30,
            f"{row['Query Strain ID']} x {row['Array Strain ID']}",
        )

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

    organism = Organism.scerevisiae

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
            species="Saccharomyces cerevisiae",
            strain="S288C",
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
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
