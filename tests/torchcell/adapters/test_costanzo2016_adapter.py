# tests/torchcell/adapters/test_costanzo2016_adapter.py
# [[tests.torchcell.adapters.test_costanzo2016_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_costanzo2016_adapter.py
# Test file: tests/tests/torchcell/adapters/test_test_costanzo2016_adapter.py

import logging
import os
import shutil

import pytest
from biocypher import BioCypher

from torchcell.adapters.costanzo2016_adapter import CostanzoSmfAdapter
from torchcell.datasets.scerevisiae import SmfCostanzo2016Dataset


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


def test_no_duplicate_warnings(log_capture):
    logger, handler = log_capture

    bc = BioCypher()
    dataset = SmfCostanzo2016Dataset()
    adapter = CostanzoSmfAdapter(dataset=dataset)

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
    test_no_duplicate_warnings(handler)
    logger.removeHandler(handler)
