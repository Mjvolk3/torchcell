# tests/torchcell/adapters/test_kuzmin2018_adapter.py
# [[tests.torchcell.adapters.test_kuzmin2018_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_kuzmin2018_adapter.py
# Test file: tests/tests/torchcell/adapters/test_test_kuzmin2018_adapter.py


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
from torchcell.datasets.scerevisiae import (
    DmfKuzmin2018Dataset,
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


def test_no_duplicate_warnings_SmfKuzmin2018Dataset(log_capture):
    logger, handler = log_capture

    rand_int = hashlib.md5(str(randint(0, int(1e10))).encode()).hexdigest()
    output_directory = osp.join(DATA_ROOT, "database/biocypher-out", rand_int)
    bc = BioCypher(
        output_directory=output_directory,
        biocypher_config_path=BIOCYPHER_CONFIG_PATH,
        schema_config_path=SCHEMA_CONFIG_PATH,
    )
    dataset = SmfKuzmin2018Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/smf_kuzmin2018")
    )
    adapter = SmfKuzmin2018Adapter(dataset=dataset, num_workers=10)

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


def test_no_duplicate_warnings_DmfKuzmin2018Dataset(log_capture):
    logger, handler = log_capture

    rand_int = hashlib.md5(str(randint(0, int(1e10))).encode()).hexdigest()
    output_directory = osp.join(DATA_ROOT, "database/biocypher-out", rand_int)
    bc = BioCypher(
        output_directory=output_directory,
        biocypher_config_path=BIOCYPHER_CONFIG_PATH,
        schema_config_path=SCHEMA_CONFIG_PATH,
    )
    dataset = DmfKuzmin2018Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/dmf_kuzmin2018")
    )
    adapter = DmfKuzmin2018Adapter(dataset=dataset, num_workers=10)

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


def test_no_duplicate_warnings_TmfKuzmin2018Dataset(log_capture):
    logger, handler = log_capture

    rand_int = hashlib.md5(str(randint(0, int(1e10))).encode()).hexdigest()
    output_directory = osp.join(DATA_ROOT, "database/biocypher-out", rand_int)
    bc = BioCypher(
        output_directory=output_directory,
        biocypher_config_path=BIOCYPHER_CONFIG_PATH,
        schema_config_path=SCHEMA_CONFIG_PATH,
    )
    dataset = TmfKuzmin2018Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/tmf_kuzmin2018")
    )
    adapter = TmfKuzmin2018Adapter(dataset=dataset, num_workers=10)
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
    test_no_duplicate_warnings_SmfKuzmin2018Dataset(handler)
    logger.removeHandler(handler)
