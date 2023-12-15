# torchcell/multidigraph/sgd.py
# [[torchcell.multidigraph.sgd]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/multidigraph/sgd.py
# Test file: torchcell/multidigraph/test_sgd.py
import asyncio
import json
import logging
import os
import os.path as osp
import tempfile
import time
from asyncio import Task
from collections.abc import Callable
from typing import Any, Optional

import aiohttp
from aiohttp import ClientError, ContentTypeError
from attrs import define, field
from tqdm import tqdm

from torchcell.multidigraph.validation.locus_related.locus import (
    Alias,
    InteractionOverview,
    LocusData,
    LocusDataUrl,
    PhysicalExperiments,
    Qualities,
    Reference,
    validate_data,
)

log = logging.getLogger(__name__)


@define
class Gene:
    locusID: str = "YAL001C"
    is_validated: bool = field(default=True, init=True, repr=False)
    sgd_url: str = "https://www.yeastgenome.org/backend/locus"
    headers: dict[str, str] = field(default={"accept": "application/json"})
    base_data_dir: str = "data/sgd/genome/genes"
    save_path: str = field(default=None, init=False, repr=True)
    _data: dict[str, dict[Any, Any] | list[Any]] = field(factory=dict, init=False)
    _data_task: Task[Any] | None = field(default=None, init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        if not osp.exists(self.base_data_dir):
            os.makedirs(self.base_data_dir)
        self.save_path: str = osp.join(self.base_data_dir, f"{self.locusID}.json")
        if osp.exists(self.save_path):
            self._data = self.read()
        else:
            # No fetch operation scheduled yet
            self._data_task = None

    @property
    def data(self) -> dict[str, dict[Any, Any] | list[Any]]:
        if self._data != {}:
            return self._data
        else:
            raise ValueError(
                "No data available. Call fetch_data(), e.g., `asyncio.run(gene.fetch_data())`"
            )

    async def fetch_data(self) -> None:
        if self._data_task is None:
            # Schedule the download
            self._data_task = asyncio.create_task(self.download_data())
            await self._data_task
        elif not self._data_task.done():
            # Download already scheduled but not yet finished
            await self._data_task
        # If task is done and _data is still empty, download_data failed silently
        if self._data == {}:
            raise ValueError("Data fetch failed.")

    async def download_data(self) -> None:
        self._data["locus"] = await self.locus()
        self._data["sequence_details"] = await self.sequence_details()
        self._data["neighbor_sequence_details"] = await self.neighbor_sequence_details()
        self._data["posttranslational_details"] = await self.posttranslational_details()
        self._data[
            "protein_experiment_details"
        ] = await self.protein_experiment_details()
        self._data["protein_domain_details"] = await self.protein_domain_details()
        self._data["go_details"] = await self.go_details()
        self._data["phenotype_details"] = await self.phenotype_details()
        self._data["interaction_details"] = await self.interaction_details()
        self._data["regulation_details"] = await self.regulation_details()
        self._data["literature_details"] = await self.literature_details()
        self.write()

    def write(self) -> None:
        with open(self.save_path, "w") as f:
            json.dump(self._data, f, indent=4)

    def read(self) -> dict[str, dict[Any, Any] | list[Any]]:
        if not osp.exists(self.save_path):
            raise ValueError(f"File {self.save_path} does not exist")
        with open(self.save_path) as f:
            data_in = json.load(f)
            if not isinstance(data_in, dict):
                raise ValueError(f"File {self.save_path} is not a dict")
        return data_in

    # With error handling
    async def _get_data(
        self, url: str, max_retries: int = 10
    ) -> dict[Any, Any] | list[Any] | None:
        for retry in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=self.headers) as response:
                        content_type = response.headers.get("Content-Type")

                        # Check if the content type indicates a JSON response
                        if "application/json" in content_type:
                            data = await response.json()
                            if not isinstance(data, (dict, list)):
                                raise ValueError(f"Data is not a dict or list: {data}")
                            return data
                        else:
                            # Save unexpected content (e.g., HTML) to a temporary file for debugging
                            content = await response.text()
                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=".html"
                            ) as temp:
                                temp.write(content.encode())
                                logging.error(
                                    f"Saved unexpected response to: {temp.name}"
                                )

                            error_message = (
                                f"Unexpected content type: {content_type}. URL: {url}"
                            )
                            logging.error(error_message)
                            raise ContentTypeError(error_message)

            except (ClientError, ContentTypeError) as e:
                if retry == max_retries - 1:  # If it's the last retry and still fails
                    logging.error(
                        f"Failed to fetch data after {max_retries} attempts. Error: {e}"
                    )
                    return None
                else:
                    logging.warning(f"Attempt {retry + 1} failed. Retrying...")
                    await asyncio.sleep(2**retry)  # Exponential backoff

        return None

    async def locus(self) -> dict[Any, Any] | list[Any]:
        url = osp.join(self.sgd_url, self.locusID)
        data = await self._get_data(url)
        if self.is_validated:
            data = validate_data(data)
            with open("locus_schema.json", "w") as f:
                json.dump(data.model_json_schema(), f, indent=4)
            data = data.model_dump()
        return data  # breakpoint, printed out data

    async def sequence_details(self) -> dict[Any, Any] | list[Any]:
        url = osp.join(self.sgd_url, self.locusID, "sequence_details")
        data = await self._get_data(url)
        return data

    async def neighbor_sequence_details(self) -> dict[Any, Any] | list[Any]:
        url = osp.join(self.sgd_url, self.locusID, "neighbor_sequence_details")
        data = await self._get_data(url)
        return data

    async def posttranslational_details(self) -> dict[Any, Any] | list[Any]:
        url = osp.join(self.sgd_url, self.locusID, "posttranslational_details")
        data = await self._get_data(url)
        return data

    async def protein_experiment_details(self) -> dict[Any, Any] | list[Any]:
        url = osp.join(self.sgd_url, self.locusID, "protein_experiment_details")
        data = await self._get_data(url)
        return data

    async def protein_domain_details(self) -> dict[Any, Any] | list[Any]:
        url = osp.join(self.sgd_url, self.locusID, "protein_domain_details")
        data = await self._get_data(url)
        return data

    async def go_details(self) -> dict[Any, Any] | list[Any]:
        url = osp.join(self.sgd_url, self.locusID, "go_details")
        data = await self._get_data(url)
        return data

    async def phenotype_details(self) -> dict[Any, Any] | list[Any]:
        url = osp.join(self.sgd_url, self.locusID, "phenotype_details")
        data = await self._get_data(url)
        return data

    async def interaction_details(self) -> dict[Any, Any] | list[Any]:
        url = osp.join(self.sgd_url, self.locusID, "interaction_details")
        data = await self._get_data(url)
        return data

    async def regulation_details(self) -> dict[Any, Any] | list[Any]:
        url = osp.join(self.sgd_url, self.locusID, "regulation_details")
        data = await self._get_data(url)
        return data

    async def literature_details(self) -> dict[Any, Any] | list[Any]:
        url = osp.join(self.sgd_url, self.locusID, "literature_details")
        data = await self._get_data(url)
        return data


async def process_gene(gene: Gene, progress_bar: Any) -> dict[Any, Any] | list[Any]:
    await gene.fetch_data()
    data = gene.data
    progress_bar.update()
    return data


async def download_genes(
    locus_ids: list[str], gene_factory: Callable[[str], Gene], is_validated: bool
) -> None:
    with tqdm(total=len(locus_ids)) as progress_bar:
        tasks = []
        for id_ in locus_ids:
            # Check if the file for the gene already exists
            file_path = f"data/sgd/genes/{id_}.json"

            if os.path.exists(file_path):
                logging.info(f"Data for gene {id_} already exists. Skipping...")
                progress_bar.update(1)
                continue

            gene = gene_factory(id_, is_validated)
            tasks.append(process_gene(gene, progress_bar))

        await asyncio.gather(*tasks)


def create_gene(locusID: str, is_validated: bool) -> Gene:
    return Gene(locusID=locusID, is_validated=is_validated)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


async def download_gene_chunk(chunk, create_gene_fn, validate_flag):
    """Download a chunk of genes with a delay before starting."""
    await asyncio.sleep(1)  # Give a small break between chunks
    await download_genes(chunk, create_gene_fn, validate_flag)


def main_get_all_genes():
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    genome = SCerevisiaeGenome()
    locus_ids = list(genome.gene_set)

    CHUNK_SIZE = 50  # Adjust this value based on what works best for you
    locus_id_chunks = list(chunks(locus_ids, CHUNK_SIZE))

    for chunk in locus_id_chunks:
        asyncio.run(download_gene_chunk(chunk, create_gene, False))


if __name__ == "__main__":
    main_get_all_genes()
    # main_median_protein()
    # main()


# def main_get_all_genes():
#     from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

#     genome = SCerevisiaeGenome()
#     # genome.drop_chrmt()
#     locus_ids = list(genome.gene_set)

#     # Chunk the locus_ids list by every 10 items
#     # locus_id_chunks = chunks(locus_ids, 10)

#     asyncio.run(download_genes(locus_ids[:100], create_gene, is_validated=False))

# def main_median_protein():
#     import os

#     from dotenv import load_dotenv

#     load_dotenv()
#     DATA_ROOT = os.getenv("DATA_ROOT")
#     from torchcell.datasets.scerevisiae import DmfCostanzo2016Dataset
#     from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

#     dmf_dataset = DmfCostanzo2016Dataset(
#         root=osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016_1e4"),
#         preprocess="low_dmf_std",
#     )
#     # genome = SCerevisiaeGenome(data_root=osp.join(DATA_ROOT, "data/sgd/genome"))
#     # genome.drop_chrmt()
#     # genome.drop_empty_go()
#     locus_ids = list(dmf_dataset.gene_set)
#     start_time = time.time()
#     asyncio.run(download_genes(locus_ids, create_gene, is_validated=False))
#     end_time = time.time()
#     print(f"Execution time: {end_time - start_time} seconds")

# def main() -> None:
#     start_time = time.time()
#     locus_ids = [
#         "YPR201W",
#         "YPR202W",
#         "YPR199C",
#         "YPR200C",
#         "YPR198W",
#         "YPR196W",
#         "YPR197C",
#         "YPR194C",
#         "YPR195C",
#         "YPR193C",
#         "YPR192W",
#         "YLR153C",
#     ]
#     asyncio.run(
#         download_genes(locus_ids, create_gene, is_validated=False)
#     )  # TODO Change for Dev
#     end_time = time.time()
#     print(f"Execution time: {end_time - start_time} seconds")
#     # gene = Gene("YDR210W")
#     # asyncio.run(gene.fetch_data())
#     # gene.data


# async def download_genes(
#     locus_ids: list[str], gene_factory: Callable[[str], Gene], is_validated: bool
# ) -> None:
#     with tqdm(total=len(locus_ids)) as progress_bar:
#         await asyncio.gather(
#             *(
#                 process_gene(gene_factory(id_, is_validated), progress_bar)
#                 for id_ in locus_ids
#             )
#         )

# async def _get_data(self, url: str) -> dict[Any, Any] | list[Any]:
#     async with aiohttp.ClientSession() as session:
#         async with session.get(url, headers=self.headers) as response:
#             data = await response.json()
#             if not isinstance(data, (dict, list)):
#                 raise ValueError(f"Data is not a dict or list: {data}")
#             return data
