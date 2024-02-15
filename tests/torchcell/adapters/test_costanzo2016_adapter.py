# tests/torchcell/adapters/test_costanzo2016_adapter.py
# [[tests.torchcell.adapters.test_costanzo2016_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_costanzo2016_adapter.py
# Test file: tests/tests/torchcell/adapters/test_test_costanzo2016_adapter.py

import hashlib
import logging
import os
import os.path as osp
import shutil
from random import randint

import pytest
from biocypher import BioCypher
from dotenv import load_dotenv

from torchcell.adapters import (
    DmfKuzmin2018Adapter,
    SmfKuzmin2018Adapter,
    TmfKuzmin2018Adapter,
)
from torchcell.adapters.costanzo2016_adapter import (
    DmfCostanzo2016Adapter,
    SmfCostanzo2016Adapter,
)
from torchcell.datasets.scerevisiae import (
    DmfCostanzo2016Dataset,
    DmfKuzmin2018Dataset,
    SmfCostanzo2016Dataset,
    SmfKuzmin2018Dataset,
    TmfKuzmin2018Dataset,
)

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
BIOCYPHER_CONFIG_PATH = os.getenv("BIOCYPHER_CONFIG_PATH")
SCHEMA_CONFIG_PATH = os.getenv("SCHEMA_CONFIG_PATH")


class LogCaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def log_capture():
    logger = logging.getLogger("biocypher")
    handler = LogCaptureHandler()
    logger.addHandler(handler)
    yield logger, handler
    logger.removeHandler(handler)


def test_no_duplicate_warnings_SmfCostanzo2016Dataset(log_capture):
    logger, handler = log_capture

    rand_int = hashlib.md5(str(randint(0, int(1e10))).encode()).hexdigest()
    output_directory = osp.join(DATA_ROOT, "database/biocypher-out", rand_int)
    bc = BioCypher(
        output_directory=output_directory,
        biocypher_config_path=BIOCYPHER_CONFIG_PATH,
        schema_config_path=SCHEMA_CONFIG_PATH,
    )
    dataset = SmfCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/smf_costanzo2016")
    )
    adapter = SmfCostanzo2016Adapter(dataset=dataset, num_workers=10)

    # Run the part of the script to be tested
    bc.write_nodes(adapter.get_nodes())
    bc.write_edges(adapter.get_edges())

    # Check that no log records contain the duplicate warning
    for record in handler.records:  # Using handler.records
        assert "Duplicate edge type" not in record.getMessage()
        assert "Duplicate node type" not in record.getMessage()

    # Deleting the directories and the specific log file
    try:
        shutil.rmtree(bc._output_directory)
        file_handler = next(
            (h for h in logger.handlers if isinstance(h, logging.FileHandler)), None
        )
        if file_handler and hasattr(file_handler, "baseFilename"):
            log_file_path = file_handler.baseFilename
            if osp.exists(log_file_path):
                os.remove(log_file_path)
    except OSError as e:
        print(f"Error: {e.filename} - {e.strerror}")


def test_no_duplicate_warnings_DmfCostanzo2016Dataset(log_capture):
    logger, handler = log_capture
    # we take a subset since the dataset is too large.
    rand_int = hashlib.md5(str(randint(0, int(1e10))).encode()).hexdigest()
    output_directory = osp.join(DATA_ROOT, "database/biocypher-out", rand_int)
    bc = BioCypher(
        output_directory=output_directory,
        biocypher_config_path=BIOCYPHER_CONFIG_PATH,
        schema_config_path=SCHEMA_CONFIG_PATH,
    )
    dataset = DmfCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/smf_costanzo2016_1e5"),
        subset_n=int(1e5),
    )
    adapter = DmfCostanzo2016Adapter(dataset=dataset, num_workers=10)

    # Run the part of the script to be tested
    bc.write_nodes(adapter.get_nodes())
    bc.write_edges(adapter.get_edges())

    # Check that no log records contain the duplicate warning
    for record in handler.records:  # Using handler.records
        assert "Duplicate edge type" not in record.getMessage()
        assert "Duplicate node type" not in record.getMessage()

    # Deleting the directories and the specific log file
    try:
        shutil.rmtree(bc._output_directory)
        file_handler = next(
            (h for h in logger.handlers if isinstance(h, logging.FileHandler)), None
        )
        if file_handler and hasattr(file_handler, "baseFilename"):
            log_file_path = file_handler.baseFilename
            if osp.exists(log_file_path):
                os.remove(log_file_path)
    except OSError as e:
        print(f"Error: {e.filename} - {e.strerror}")


if __name__ == "__main__":
    # Note: this will not utilize pytest's features
    logger = logging.getLogger("biocypher")
    handler = LogCaptureHandler()
    logger.addHandler(handler)
    test_no_duplicate_warnings_SmfCostanzo2016Dataset(handler)
    logger.removeHandler(handler)
