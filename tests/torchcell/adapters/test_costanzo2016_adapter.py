# tests/torchcell/adapters/test_costanzo2016_adapter.py
# [[tests.torchcell.adapters.test_costanzo2016_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_costanzo2016_adapter.py
# Test file: tests/tests/torchcell/adapters/test_test_costanzo2016_adapter.py

import logging
import os
import shutil
import subprocess
from datetime import datetime

import pytest
from biocypher import BioCypher

from torchcell.adapters import SmfCostanzo2016Adapter
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


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call":
        item._test_outcome = rep


@pytest.fixture
def track_test_success(cache, request):
    key = f"last_success_{request.node.name}"
    last_success_str = cache.get(key, None)
    yield
    if hasattr(request.node, "_test_outcome") and request.node._test_outcome.passed:
        cache.set(key, datetime.now().isoformat())
    return last_success_str


def get_last_change_date(file_path):
    cmd = ["git", "log", "-1", "--format=%cd", file_path]
    try:
        last_change_date = subprocess.check_output(cmd, shell=True).decode().strip()
        return datetime.strptime(last_change_date, "%a %b %d %H:%M:%S %Y %z")
    except subprocess.CalledProcessError as e:
        print(f"Error running git log: {e}")
        return None


def test_no_duplicate_warnings_SmfCostanzo2016Dataset(log_capture, track_test_success):
    src_file_path = "/Users/michaelvolk/Documents/projects/torchcell/torchcell/adapters/costanzo2016_adapter.py"
    last_change_date = get_last_change_date(src_file_path)

    # Convert the last successful run from string to datetime
    last_success_str = track_test_success
    last_success_date = (
        datetime.fromisoformat(last_success_str) if last_success_str else None
    )

    # Skip test if src hasn't updated since last successful test run
    if last_success_date and last_change_date <= last_success_date:
        pytest.skip(
            "The source file has not been updated since the last successful test run."
        )

    # Rest is computationally expensive
    logger, handler = log_capture

    bc = BioCypher()
    dataset = SmfCostanzo2016Dataset()
    adapter = SmfCostanzo2016Adapter(dataset=dataset)

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
    test_no_duplicate_warnings_SmfCostanzo2016Dataset(handler)
    logger.removeHandler(handler)
