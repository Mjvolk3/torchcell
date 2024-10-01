# torchcell/datasets/scerevisiae/syn_leth_db_yeast
# [[torchcell.datasets.scerevisiae.syn_leth_db_yeast]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/syn_leth_db_yeast
# Test file: tests/torchcell/datasets/scerevisiae/test_syn_leth_db_yeast.py


import os
import pandas as pd
import pickle
import lmdb
from tqdm import tqdm
import os.path as osp
import requests
from torchcell.data import ExperimentDataset, post_process
from torchcell.datamodels.schema import (
    ReferenceGenome,
    Environment,
    Media,
    Temperature,
    Genotype,
    SgaKanMxDeletionPerturbation,
    SyntheticLethalityPhenotype,
    SyntheticLethalityExperiment,
    SyntheticLethalityExperimentReference,
    SyntheticRescuePhenotype,
    SyntheticRescueExperiment,
    SyntheticRescueExperimentReference,
    Publication,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from pydantic import field_validator
import re
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@register_dataset
class SynthLethalityYeastSynthLethDbDataset(ExperimentDataset):
    def __init__(
        self,
        root: str = "data/torchcell/syn_leth_db_yeast",
        genome: SCerevisiaeGenome = None,
        io_workers: int = 0,
        transform=None,
        pre_transform=None,
    ):
        self.genome = genome
        self.gene_name_to_systematic = {}
        self._build_gene_name_mapping()
        # delete to remove: cannot pickle 'sqlite3.Connection' object
        del genome
        del self.genome
        super().__init__(root, io_workers, transform, pre_transform)

    def _build_gene_name_mapping(self):
        print("Building gene name to systematic name mapping...")
        for feature in tqdm(self.genome.db.all_features()):
            if feature.featuretype == "gene":
                systematic_name = feature.id
                self.gene_name_to_systematic[systematic_name] = systematic_name
                if "gene" in feature.attributes:
                    gene_name = feature.attributes["gene"][0]
                    self.gene_name_to_systematic[gene_name] = systematic_name
                if "Alias" in feature.attributes:
                    for alias in feature.attributes["Alias"]:
                        self.gene_name_to_systematic[alias] = systematic_name

    @property
    def raw_file_names(self) -> list[str]:
        return ["Yeast_SL.csv"]

    @property
    def processed_file_names(self) -> list[str]:
        return ["lmdb"]

    @property
    def experiment_class(self) -> SyntheticLethalityExperiment:
        return SyntheticLethalityExperiment

    @property
    def reference_class(self) -> SyntheticLethalityExperimentReference:
        return SyntheticLethalityExperimentReference

    def download(self):
        url = (
            "https://synlethdb.sist.shanghaitech.edu.cn/static/download/SL/Yeast_SL.csv"
        )
        download_path = os.path.join(self.raw_dir, self.raw_file_names[0])

        if not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir)

        print(f"Downloading {url} to {download_path}")
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        with open(download_path, "wb") as f:
            f.write(response.content)

        print("Download completed.")

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict | None = None
    ) -> pd.DataFrame:
        print("Converting gene names to systematic names...")
        df["n1.systematic_name"] = df["n1.name"].apply(self.get_systematic_name)
        df["n2.systematic_name"] = df["n2.name"].apply(self.get_systematic_name)
        return df

    def get_systematic_name(self, gene_name):
        # Remove the prime character if present
        clean_gene_name = gene_name.rstrip("'")

        systematic_name = self.gene_name_to_systematic.get(clean_gene_name)

        if systematic_name is None:
            print(f"Warning: No systematic name found for gene {gene_name}")
            return gene_name  # Return original name if no match found

        return systematic_name

    @post_process
    def process(self):

        log.info("Processing Synthetic Lethality Yeast Data...")

        raw_data_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        df = pd.read_csv(raw_data_path)
        df = self.preprocess_raw(df)

        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        env = lmdb.open(os.path.join(self.processed_dir, "lmdb"), map_size=int(1e12))

        with env.begin(write=True) as txn:
            for index, row in tqdm(df.iterrows(), total=len(df)):
                experiment, reference, publication = self.create_experiment(
                    self.name, row
                )

                serialized_data = pickle.dumps(
                    {
                        "experiment": experiment.model_dump(),
                        "reference": reference.model_dump(),
                        "publication": publication.model_dump(),
                    }
                )
                txn.put(f"{index}".encode(), serialized_data)

        env.close()

    @staticmethod
    def create_experiment(dataset_name, row):
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="S288C"
        )

        genotype = Genotype(
            perturbations=[
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["n1.systematic_name"],
                    perturbed_gene_name=row["n1.name"],
                    strain_id="S288C",
                ),
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["n2.systematic_name"],
                    perturbed_gene_name=row["n2.name"],
                    strain_id="S288C",
                ),
            ]
        )

        environment = Environment(
            media=Media(name="YEPD", state="solid"), temperature=Temperature(value=30)
        )

        phenotype = SyntheticLethalityPhenotype(
            is_synthetic_lethal=True,
            synthetic_lethality_statistic_score=float(row["r.statistic_score"]),
        )

        phenotype_reference = SyntheticLethalityPhenotype(
            is_synthetic_lethal=False, synthetic_lethality_statistic_score=None
        )

        experiment = SyntheticLethalityExperiment(
            dataset_name=dataset_name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )

        reference = SyntheticLethalityExperimentReference(
            dataset_name=dataset_name,
            genome_reference=genome_reference,
            environment_reference=environment,
            phenotype_reference=phenotype_reference,
        )

        publication = Publication(
            pubmed_id=str(row["r.pubmed_id"]),
            pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{row['r.pubmed_id']}/",
            doi=None,
            doi_url=None,
        )
        return experiment, reference, publication


@register_dataset
class SynthRescueYeastSynthLethDbDataset(ExperimentDataset):
    def __init__(
        self,
        root: str = "data/torchcell/syn_rescue_db_yeast",
        genome: SCerevisiaeGenome = None,
        io_workers: int = 0,
        transform=None,
        pre_transform=None,
    ):
        self.genome = genome
        self.gene_name_to_systematic = {}
        self._build_gene_name_mapping()
        # delete to remove: cannot pickle 'sqlite3.Connection' object
        del genome
        del self.genome
        super().__init__(root, io_workers, transform, pre_transform)

    def _build_gene_name_mapping(self):
        print("Building gene name to systematic name mapping...")
        for feature in tqdm(self.genome.db.all_features()):
            if feature.featuretype == "gene":
                systematic_name = feature.id
                self.gene_name_to_systematic[systematic_name] = systematic_name
                if "gene" in feature.attributes:
                    gene_name = feature.attributes["gene"][0]
                    self.gene_name_to_systematic[gene_name] = systematic_name
                if "Alias" in feature.attributes:
                    for alias in feature.attributes["Alias"]:
                        self.gene_name_to_systematic[alias] = systematic_name

    @property
    def raw_file_names(self) -> list[str]:
        return ["Yeast_SR.csv"]

    @property
    def processed_file_names(self) -> list[str]:
        return ["lmdb"]

    @property
    def experiment_class(self) -> SyntheticRescueExperiment:
        return SyntheticRescueExperiment

    @property
    def reference_class(self) -> SyntheticRescueExperimentReference:
        return SyntheticRescueExperimentReference

    def download(self):
        url = "https://synlethdb.sist.shanghaitech.edu.cn/static/download/nonSL/Yeast_SR.csv"
        download_path = os.path.join(self.raw_dir, self.raw_file_names[0])

        if not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir)

        print(f"Downloading {url} to {download_path}")
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        with open(download_path, "wb") as f:
            f.write(response.content)

        print("Download completed.")

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict | None = None
    ) -> pd.DataFrame:
        print("Converting gene names to systematic names...")
        df["n1.systematic_name"] = df["n1.name"].apply(self.get_systematic_name)
        df["n2.systematic_name"] = df["n2.name"].apply(self.get_systematic_name)

        return df

    def get_systematic_name(self, gene_name):
        # Remove the prime character if present
        clean_gene_name = gene_name.rstrip("'")

        systematic_name = self.gene_name_to_systematic.get(clean_gene_name)

        if systematic_name is None:
            print(f"Warning: No systematic name found for gene {gene_name}")
            return gene_name  # Return original name if no match found

        return systematic_name

    @post_process
    def process(self):
        log.info("Processing Synthetic Rescue Yeast Data...")

        raw_data_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        df = pd.read_csv(raw_data_path)
        df = self.preprocess_raw(df)

        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        env = lmdb.open(os.path.join(self.processed_dir, "lmdb"), map_size=int(1e12))

        with env.begin(write=True) as txn:
            for index, row in tqdm(df.iterrows(), total=len(df)):
                experiment, reference, publication = self.create_experiment(
                    self.name, row
                )

                serialized_data = pickle.dumps(
                    {
                        "experiment": experiment.model_dump(),
                        "reference": reference.model_dump(),
                        "publication": publication.model_dump(),
                    }
                )
                txn.put(f"{index}".encode(), serialized_data)

        env.close()

    @staticmethod
    def create_experiment(dataset_name, row):
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="S288C"
        )

        genotype = Genotype(
            perturbations=[
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["n1.systematic_name"],
                    perturbed_gene_name=row["n1.name"],
                    strain_id="S288C",
                ),
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["n2.systematic_name"],
                    perturbed_gene_name=row["n2.name"],
                    strain_id="S288C",
                ),
            ]
        )

        environment = Environment(
            media=Media(name="YEPD", state="solid"), temperature=Temperature(value=30)
        )

        phenotype = SyntheticRescuePhenotype(
            is_synthetic_rescue=True,
            synthetic_rescue_statistic_score=(
                float(row["r.statistic_score"])
                if pd.notna(row["r.statistic_score"])
                else None
            ),
        )

        phenotype_reference = SyntheticRescuePhenotype(
            is_synthetic_rescue=False, synthetic_rescue_statistic_score=None
        )

        experiment = SyntheticRescueExperiment(
            dataset_name=dataset_name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )

        reference = SyntheticRescueExperimentReference(
            dataset_name=dataset_name,
            genome_reference=genome_reference,
            environment_reference=environment,
            phenotype_reference=phenotype_reference,
        )

        publication = Publication(
            pubmed_id=str(row["r.pubmed_id"]),
            pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{row['r.pubmed_id']}/",
            doi=None,
            doi_url=None,
        )
        return experiment, reference, publication


def main():
    import os
    from dotenv import load_dotenv

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    genome = SCerevisiaeGenome(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), overwrite=True
    )

    lethality_dataset = SynthLethalityYeastSynthLethDbDataset(genome=genome)
    print(lethality_dataset)

    rescue_dataset = SynthRescueYeastSynthLethDbDataset(genome=genome)
    print(rescue_dataset)


if __name__ == "__main__":
    main()
