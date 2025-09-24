# torchcell/datasets/scerevisiae/sgd
# [[torchcell.datasets.scerevisiae.sgd]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/sgd
# Test file: tests/torchcell/datasets/scerevisiae/test_sgd.py


import logging
import re
from urllib.parse import urlparse
from typing import Callable
import pandas as pd
import time
import lmdb
import pickle
import os
import random
from tqdm import tqdm
import os.path as osp
from torchcell.data import ExperimentDataset, post_process
from Bio import Entrez
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datamodels.schema import (
    ReferenceGenome,
    Environment,
    Media,
    Temperature,
    Genotype,
    SgaKanMxDeletionPerturbation,
    GeneEssentialityPhenotype,
    GeneEssentialityExperiment,
    GeneEssentialityExperimentReference,
    Publication,
)
from torchcell.graph import SCerevisiaeGraph
from torchcell.graph.sgd import main_get_all_genes

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_publication_info(pubmed_id):
    Entrez.email = "mvjolk3@illinois.edu"
    max_retries = 5
    base_delay = 1  # seconds

    for attempt in range(max_retries):
        try:
            handle = Entrez.efetch(
                db="pubmed", id=pubmed_id, rettype="xml", retmode="text"
            )
            records = Entrez.read(handle)
            handle.close()

            article = records["PubmedArticle"][0]["MedlineCitation"]["Article"]

            info = {
                "pubmed_id": pubmed_id,
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/",
                "doi": None,
                "doi_url": None,
            }

            # Try to get DOI from ELocationID
            doi = next(
                (
                    eid
                    for eid in article.get("ELocationID", [])
                    if eid.attributes["EIdType"] == "doi"
                ),
                None,
            )

            # If DOI not found in ELocationID, try ArticleId
            if not doi:
                article_id_list = records["PubmedArticle"][0]["PubmedData"][
                    "ArticleIdList"
                ]
                doi = next(
                    (
                        article_id
                        for article_id in article_id_list
                        if article_id.attributes["IdType"] == "doi"
                    ),
                    None,
                )

            if doi:
                info["doi"] = str(doi)
                info["doi_url"] = f"https://doi.org/{info['doi']}"

            return info

        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                print(
                    f"Attempt {attempt + 1} failed. Retrying in {delay:.2f} seconds..."
                )
                time.sleep(delay)
                continue
            else:
                print(f"Error fetching info for PubMed ID {pubmed_id}: {str(e)}")
                return None


@register_dataset
class GeneEssentialitySgdDataset(ExperimentDataset):
    def __init__(
        self,
        root: str = "data/torchcell/gene_essentiality_sgd",
        scerevisiae_graph: SCerevisiaeGraph = None,
        io_workers: int = 0,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        **kwargs,
    ):
        self.scerevisiae_graph = scerevisiae_graph
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> GeneEssentialityExperiment:
        return GeneEssentialityExperiment

    @property
    def reference_class(self) -> GeneEssentialityExperimentReference:
        return GeneEssentialityExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        return []  # Return an empty list if there are no raw files to download

    def download(self):
        # If there's nothing to download, you can just pass
        pass

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict | None = None
    ) -> pd.DataFrame:
        # If there's no preprocessing needed, you can return the DataFrame as is
        return df

    @post_process
    def process(self):
        log.info("Processing SGD Gene Essentiality Data...")

        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.preprocess_dir, exist_ok=True)

        # Check if SGD gene data exists
        sgd_genes_dir = osp.join(os.environ.get("DATA_ROOT", "data"), "sgd/genome/genes")
        if osp.exists(sgd_genes_dir):
            gene_files = [f for f in os.listdir(sgd_genes_dir) if f.endswith('.json')]
            gene_count = len(gene_files)
        else:
            gene_count = 0
        
        if gene_count < 100:  # Arbitrary threshold to check if genes are downloaded
            log.info(f"SGD gene data incomplete or missing (found {gene_count} files).")
            log.info("Downloading SGD gene data. This may take 15-30 minutes...")
            main_get_all_genes()
            log.info("SGD gene download complete!")
        else:
            log.info(f"SGD gene data already exists ({gene_count} gene files found)")

        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e12))

        with env.begin(write=True) as txn:
            index = 0
            for gene in tqdm(self.scerevisiae_graph.G_raw.nodes()):
                node_data = self.scerevisiae_graph.G_raw.nodes[gene]
                inviable_phenotypes = [
                    i
                    for i in node_data.get("phenotype_details", [])
                    if (
                        i["mutant_type"] == "null"
                        and i["strain"]["display_name"] == "S288C"
                        and i["phenotype"]["display_name"] == "inviable"
                    )
                ]

                for phenotype in inviable_phenotypes:
                    experiment, reference, publication = self.create_experiment(
                        self.name, gene, phenotype
                    )

                    serialized_data = pickle.dumps(
                        {
                            "experiment": experiment.model_dump(),
                            "reference": reference.model_dump(),
                            "publication": publication.model_dump(),
                        }
                    )
                    txn.put(f"{index}".encode(), serialized_data)
                    index += 1

        env.close()

    # HACK for this dataset all meta data is guessed,
    # since we have no way fo extracting it from the paper yet
    # It is a reasonable guess
    @staticmethod
    def create_experiment(dataset_name, gene, phenotype_data):
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="S288C"
        )

        genotype = Genotype(
            perturbations=[
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=gene,
                    perturbed_gene_name=phenotype_data["locus"]["display_name"],
                    strain_id="S288C",
                )
            ]
        )

        environment = Environment(
            media=Media(name="YEPD", state="solid"),
            temperature=Temperature(value=30),  # Assuming standard temperature
        )

        phenotype = GeneEssentialityPhenotype(is_essential=True)
        phenotype_reference = GeneEssentialityPhenotype(is_essential=False)

        pubmed_id = str(phenotype_data["reference"]["pubmed_id"])
        pub_info = get_publication_info(pubmed_id)

        if pub_info is None:
            raise ValueError(
                f"Unable to retrieve publication information for PubMed ID: {pubmed_id}"
            )

        experiment = GeneEssentialityExperiment(
            dataset_name=dataset_name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )

        reference = GeneEssentialityExperimentReference(
            dataset_name=dataset_name,
            genome_reference=genome_reference,
            environment_reference=environment,
            phenotype_reference=phenotype_reference,
        )

        publication = Publication(
            pubmed_id=pub_info["pubmed_id"],
            pubmed_url=pub_info["pubmed_url"],
            doi=pub_info["doi"],
            doi_url=pub_info["doi_url"],
        )

        return experiment, reference, publication


def main():
    import os
    from dotenv import load_dotenv
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=True,
    )
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )

    dataset = GeneEssentialitySgdDataset(scerevisiae_graph=graph)
    print(dataset)


if __name__ == "__main__":
    main()
