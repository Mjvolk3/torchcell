# torchcell/datasets/scerevisiae/Ohya2005
# [[torchcell.datasets.scerevisiae.Ohya2005]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/Ohya2005
# Test file: tests/torchcell/datasets/scerevisiae/test_Ohya2005.py


import hashlib
import json
import logging
import os
import os.path as osp
import pickle
import zipfile
from collections.abc import Callable
import lmdb
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from typing import Tuple, Optional
from tqdm import tqdm
from torchcell.datamodels.schema import (
    Environment,
    Genotype,
    CalMorphExperiment,
    CalMorphExperimentReference,
    CalMorphPhenotype,
    Media,
    ReferenceGenome,
    SgaKanMxDeletionPerturbation,
    Temperature,
    Experiment,
    ExperimentReference,
    Publication,
)
from torchcell.data import ExperimentDataset, post_process
from torchcell.datasets.dataset_registry import register_dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@register_dataset
class ScmdOhya2005Dataset(ExperimentDataset):
    # Primary URL from SCMD
    primary_url_mutant = (
        "http://www.yeast.ib.k.u-tokyo.ac.jp/SCMD/download.php?path=mt4718data.tsv"
    )
    primary_url_wt = (
        "http://www.yeast.ib.k.u-tokyo.ac.jp/SCMD/download.php?path=wt122data.tsv"
    )

    # Fallback URLs from Box
    fallback_url_mutant = (
        "https://uofi.box.com/shared/static/da9uevinx6euzp5lhkw88mvtzv9pq9sd.tsv"
    )
    fallback_url_wt = (
        "https://uofi.box.com/shared/static/ji5wzym3lc3vd0kbv0frfk9xmruqp7d2.tsv"
    )

    def __init__(
        self,
        root: str = "data/torchcell/scmd_ohya2005",
        io_workers: int = 0,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        **kwargs,
    ):
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        return CalMorphExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        return CalMorphExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        return ["mt4718data.tsv", "wt122data.tsv"]

    def download(self):
        """Download mutant and wildtype data with fallback support."""
        # Download mutant data
        mutant_path = osp.join(self.raw_dir, "mt4718data.tsv")
        if not osp.exists(mutant_path):
            success = self._download_with_safari_headers(
                self.primary_url_mutant, mutant_path
            )
            if not success:
                log.warning("Failed to download mutant data from primary source")
                log.info("Trying fallback URL for mutant data...")
                success = self._download_with_safari_headers(
                    self.fallback_url_mutant, mutant_path
                )
                if not success:
                    raise RuntimeError(
                        "Failed to download mutant data from all sources"
                    )

        # Download wildtype data
        wt_path = osp.join(self.raw_dir, "wt122data.tsv")
        if not osp.exists(wt_path):
            success = self._download_with_safari_headers(self.primary_url_wt, wt_path)
            if not success:
                log.warning("Failed to download wildtype data from primary source")
                log.info("Trying fallback URL for wildtype data...")
                success = self._download_with_safari_headers(
                    self.fallback_url_wt, wt_path
                )
                if not success:
                    raise RuntimeError(
                        "Failed to download wildtype data from all sources"
                    )

    def _download_with_safari_headers(self, url: str, output_path: str) -> bool:
        """Download with Safari user-agent headers to bypass restrictions."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
            "(KHTML, like Gecko) Version/16.0 Safari/605.1.15",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }

        try:
            log.info(f"Downloading from {url}...")
            session = requests.Session()

            # Try to access the main page first to get cookies
            main_url = "http://www.yeast.ib.k.u-tokyo.ac.jp/SCMD/download.php"
            session.get(main_url, headers=headers, timeout=60)

            # Now download the file
            response = session.get(url, headers=headers, stream=True, timeout=60)
            response.raise_for_status()

            # Save the file
            os.makedirs(osp.dirname(output_path), exist_ok=True)

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_size = osp.getsize(output_path)
            if file_size > 0:
                log.info(f"Downloaded {output_path} ({file_size/1024/1024:.2f} MB)")
                return True
            else:
                log.error("Downloaded file is empty")
                os.remove(output_path)
                return False

        except Exception as e:
            log.error(f"Error downloading: {e}")
            return False

    @post_process
    def process(self):
        # Load and preprocess data
        df_mutant = pd.read_csv(osp.join(self.raw_dir, "mt4718data.tsv"), sep="\t")
        df_wt = pd.read_csv(osp.join(self.raw_dir, "wt122data.tsv"), sep="\t")

        df = self.preprocess_calmorph_data(df_mutant, df_wt)

        # Save preprocessed data
        os.makedirs(self.preprocess_dir, exist_ok=True)
        df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        # Calculate average wildtype reference phenotype
        self.wt_reference_phenotype = self._calculate_wt_reference(df_wt)

        log.info("Processing CalMorph morphology data...")

        # Initialize LMDB environment
        env = lmdb.open(
            osp.join(self.processed_dir, "lmdb"),
            map_size=int(1e12),  # Adjust map_size as needed
        )

        with env.begin(write=True) as txn:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                experiment, reference, publication = self.create_calmorph_experiment(
                    self.name, row, wt_reference_phenotype=self.wt_reference_phenotype
                )

                # Serialize the Pydantic objects
                serialized_data = pickle.dumps(
                    {
                        "experiment": experiment.model_dump(),
                        "reference": reference.model_dump(),
                        "publication": publication.model_dump(),
                    }
                )
                txn.put(f"{index}".encode(), serialized_data)

        env.close()

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict | None = None
    ) -> pd.DataFrame:
        """Preprocess raw data - for CalMorph this is handled in process()."""
        # For this dataset, preprocessing happens in process() with both dfs
        # This method is required by base class but not used in our flow
        return df
    
    def preprocess_calmorph_data(
        self, df_mutant: pd.DataFrame, df_wt: pd.DataFrame
    ) -> pd.DataFrame:
        """Preprocess raw CalMorph data with mutant and wildtype dataframes."""
        # The mutant data has "ORF" column, wildtype has "NAME" column
        # Get morphology column names (all columns except strain info)
        info_columns = ["ORF"]  # Column name in mutant data
        morphology_columns = [
            col for col in df_mutant.columns if col not in info_columns
        ]

        # Process mutant data
        df_mutant["strain_type"] = "mutant"

        # Clean gene names - the ORF column contains the systematic gene name
        # Convert to uppercase for consistency with standard yeast nomenclature
        df_mutant["systematic_gene_name"] = df_mutant["ORF"].str.strip().str.upper()
        # For perturbed_gene_name, we'll use the same as systematic name 
        # since there's no separate gene column
        df_mutant["perturbed_gene_name"] = df_mutant["systematic_gene_name"]

        # Remove rows with invalid gene names
        df_mutant = df_mutant[df_mutant["systematic_gene_name"].notna()]
        df_mutant = df_mutant[df_mutant["systematic_gene_name"] != ""]

        # Reset index
        df_mutant = df_mutant.reset_index(drop=True)

        return df_mutant

    def _calculate_wt_reference(self, df_wt: pd.DataFrame) -> dict:
        """Calculate average wildtype morphology measurements."""
        # Wildtype data might have different column structure
        # Check for NAME or ORF column
        info_columns = []
        if "NAME" in df_wt.columns:
            info_columns.append("NAME")
        if "ORF" in df_wt.columns:
            info_columns.append("ORF")
        
        morphology_columns = [col for col in df_wt.columns if col not in info_columns]

        # Calculate mean across all wildtype replicates
        wt_means = {}
        for col in morphology_columns:
            # Convert to numeric, handling any non-numeric values
            numeric_values = pd.to_numeric(df_wt[col], errors="coerce")
            wt_means[col] = numeric_values.mean()

        return wt_means

    def create_experiment(self):
        """Required by base class but not used - see create_calmorph_experiment."""
        pass
    
    @staticmethod
    def create_calmorph_experiment(dataset_name, row, wt_reference_phenotype):
        # Genome reference - BY4741 (MATa his3Δ1 leu2Δ0 lys2Δ0 ura3Δ0)
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        )

        # Create genotype for deletion mutant
        genotype = Genotype(
            perturbations=[
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["systematic_gene_name"],
                    perturbed_gene_name=row["perturbed_gene_name"],
                    strain_id=f"{row['systematic_gene_name']}",
                )
            ]
        )

        # Environment
        environment = Environment(
            media=Media(name="YEPD", state="solid"), temperature=Temperature(value=30)
        )
        environment_reference = environment.model_copy()

        # Extract morphology measurements and separate base from CV parameters
        info_columns = [
            "ORF",
            "systematic_gene_name",
            "perturbed_gene_name",
            "strain_type",
        ]
        base_measurements = {}
        cv_measurements = {}

        for col in row.index:
            if col not in info_columns:
                # Convert to float, handling NaN values
                value = row[col]
                if pd.notna(value):
                    float_value = float(value)
                else:
                    float_value = 0.0  # Or handle missing values appropriately

                # Separate base parameters from CV parameters
                if col.startswith(("CCV", "ACV", "DCV", "TCV")):
                    cv_measurements[col] = float_value
                else:
                    base_measurements[col] = float_value

        # Create phenotype with separated base and CV measurements
        phenotype = CalMorphPhenotype(
            calmorph=base_measurements,
            calmorph_coefficient_of_variation=(
                cv_measurements if cv_measurements else None
            ),
        )

        # Create reference phenotype from wildtype average
        # Also need to separate wildtype measurements
        wt_base = {
            k: v
            for k, v in wt_reference_phenotype.items()
            if not k.startswith(("CCV", "ACV", "DCV", "TCV"))
        }
        wt_cv = {
            k: v
            for k, v in wt_reference_phenotype.items()
            if k.startswith(("CCV", "ACV", "DCV", "TCV"))
        }

        phenotype_reference = CalMorphPhenotype(
            calmorph=wt_base, calmorph_coefficient_of_variation=wt_cv if wt_cv else None
        )

        # Create reference
        reference = CalMorphExperimentReference(
            dataset_name=dataset_name,
            genome_reference=genome_reference,
            environment_reference=environment_reference,
            phenotype_reference=phenotype_reference,
        )

        # Create experiment
        experiment = CalMorphExperiment(
            dataset_name=dataset_name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )

        # Publication
        publication = Publication(
            pubmed_id="16365294",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/16365294/",
            doi="10.1073/pnas.0509436102",
            doi_url="https://www.pnas.org/doi/10.1073/pnas.0509436102",
        )

        return experiment, reference, publication


if __name__ == "__main__":
    dataset = ScmdOhya2005Dataset()
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        print(dataset[0])
