# tests/torchcell/adapters/test_kuzmin2018_adapter.py
# [[tests.torchcell.adapters.test_kuzmin2018_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_kuzmin2018_adapter.py
# Test file: tests/tests/torchcell/adapters/test_test_kuzmin2018_adapter.py


import hashlib
import logging
import os
import shutil

import pytest
from biocypher import BioCypher

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

    bc = BioCypher()
    dataset = SmfKuzmin2018Dataset()
    adapter = SmfKuzmin2018Adapter(dataset=dataset)

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
            if os.path.exists(log_file_path):
                os.remove(log_file_path)
    except OSError as e:
        print(f"Error: {e.filename} - {e.strerror}")


def test_no_duplicate_warnings_DmfKuzmin2018Dataset(log_capture):
    logger, handler = log_capture

    bc = BioCypher()
    dataset = DmfKuzmin2018Dataset()
    adapter = DmfKuzmin2018Adapter(dataset=dataset)

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
            if os.path.exists(log_file_path):
                os.remove(log_file_path)
    except OSError as e:
        print(f"Error: {e.filename} - {e.strerror}")


def test_no_duplicate_warnings_TmfKuzmin2018Dataset(log_capture):
    logger, handler = log_capture

    bc = BioCypher()
    dataset = TmfKuzmin2018Dataset()
    adapter = TmfKuzmin2018Adapter(dataset=dataset)

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
            if os.path.exists(log_file_path):
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
