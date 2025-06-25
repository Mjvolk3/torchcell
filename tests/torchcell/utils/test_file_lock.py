# tests/torchcell/utils/test_file_lock.py
# [[tests.torchcell.utils.test_file_lock]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/utils/test_file_lock.py

import json
import multiprocessing as mp
import os
import tempfile
import time
from pathlib import Path

import pytest

from torchcell.utils.file_lock import FileLockHelper


def test_read_write_json_basic():
    """Test basic read/write functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test.json"
        test_data = {"key": "value", "number": 42}

        # Write data
        FileLockHelper.write_json_with_lock(file_path, test_data)

        # Read data
        read_data = FileLockHelper.read_json_with_lock(file_path)

        assert read_data == test_data


def test_create_if_missing():
    """Test creating file if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "new_file.json"
        default_data = {"default": "data"}

        # Read non-existent file with create_if_missing=True
        data = FileLockHelper.read_json_with_lock(
            file_path, create_if_missing=True, default_data=default_data
        )

        assert data == default_data
        assert file_path.exists()


def test_file_not_found():
    """Test FileNotFoundError when file doesn't exist and create_if_missing=False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "non_existent.json"

        with pytest.raises(FileNotFoundError):
            FileLockHelper.read_json_with_lock(file_path)


def test_update_json():
    """Test atomic JSON update functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "update_test.json"
        initial_data = {"counter": 0, "items": []}

        # Create initial file
        FileLockHelper.write_json_with_lock(file_path, initial_data)

        # Update function
        def increment_counter(data):
            data["counter"] += 1
            data["items"].append(f"item_{data['counter']}")
            return data

        # Update the file
        updated_data = FileLockHelper.update_json_with_lock(
            file_path, increment_counter
        )

        assert updated_data["counter"] == 1
        assert updated_data["items"] == ["item_1"]

        # Verify file contents
        read_data = FileLockHelper.read_json_with_lock(file_path)
        assert read_data == updated_data


def _concurrent_writer(file_path, process_id, num_writes):
    """Helper function for concurrent write test."""
    for i in range(num_writes):

        def update_func(data):
            if "writes" not in data:
                data["writes"] = []
            data["writes"].append(f"process_{process_id}_write_{i}")
            return data

        FileLockHelper.update_json_with_lock(
            file_path, update_func, create_if_missing=True, default_data={}
        )
        time.sleep(0.01)  # Small delay to increase chance of contention


def test_concurrent_writes():
    """Test that concurrent writes are properly serialized."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "concurrent_test.json"
        num_processes = 4
        num_writes_per_process = 5

        # Create processes
        processes = []
        for i in range(num_processes):
            p = mp.Process(
                target=_concurrent_writer,
                args=(str(file_path), i, num_writes_per_process),
            )
            processes.append(p)
            p.start()

        # Wait for all processes to complete
        for p in processes:
            p.join()

        # Read final result
        data = FileLockHelper.read_json_with_lock(file_path)

        # Verify all writes were recorded
        assert "writes" in data
        assert len(data["writes"]) == num_processes * num_writes_per_process

        # Verify no writes were lost
        for i in range(num_processes):
            for j in range(num_writes_per_process):
                expected_write = f"process_{i}_write_{j}"
                assert expected_write in data["writes"]


def test_cleanup_lock_files():
    """Test lock file cleanup functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create some files and write to them (creates lock files)
        for i in range(3):
            file_path = tmpdir / f"file_{i}.json"
            FileLockHelper.write_json_with_lock(file_path, {"id": i})

        # Check that lock files exist
        lock_files = list(tmpdir.glob("*.lock"))
        assert len(lock_files) == 3

        # Clean up lock files
        removed_count = FileLockHelper.cleanup_lock_files(tmpdir)
        assert removed_count == 3

        # Verify lock files are gone
        lock_files = list(tmpdir.glob("*.lock"))
        assert len(lock_files) == 0

        # Verify data files still exist
        data_files = list(tmpdir.glob("*.json"))
        assert len(data_files) == 3


def test_nested_directories():
    """Test that parent directories are created as needed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "nested" / "deep" / "test.json"
        test_data = {"nested": True}

        # Write to nested path
        FileLockHelper.write_json_with_lock(file_path, test_data)

        # Verify file and directories were created
        assert file_path.exists()
        assert file_path.parent.exists()

        # Read back data
        read_data = FileLockHelper.read_json_with_lock(file_path)
        assert read_data == test_data


if __name__ == "__main__":
    # Run basic tests
    test_read_write_json_basic()
    test_create_if_missing()
    test_update_json()
    test_concurrent_writes()
    test_cleanup_lock_files()
    test_nested_directories()
    print("All tests passed!")
