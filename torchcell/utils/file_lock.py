# torchcell/utils/file_lock.py
# [[torchcell.utils.file_lock]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/utils/file_lock.py
# Test file: tests/torchcell/utils/test_file_lock.py

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional, Union

from filelock import FileLock, Timeout

log = logging.getLogger(__name__)


class FileLockHelper:
    """Cross-platform file locking utility for JSON file operations.
    
    This class provides thread-safe and process-safe file operations using
    the filelock library, which works across all platforms including Windows,
    macOS, and Linux.
    
    Attributes:
        default_timeout: Default timeout in seconds for acquiring locks
        default_retry_delay: Default delay between retries in seconds
    """
    
    default_timeout: float = 60.0
    default_retry_delay: float = 0.1
    
    @classmethod
    def _get_lock_path(cls, file_path: Union[str, Path]) -> Path:
        """Get the lock file path for a given file.
        
        Args:
            file_path: Path to the file to lock
            
        Returns:
            Path to the lock file
        """
        file_path = Path(file_path)
        return file_path.with_suffix(file_path.suffix + '.lock')
    
    @classmethod
    def read_json_with_lock(
        cls,
        file_path: Union[str, Path],
        timeout: Optional[float] = None,
        retry_delay: Optional[float] = None,
        create_if_missing: bool = False,
        default_data: Optional[Any] = None
    ) -> Any:
        """Read JSON file with file locking for thread/process safety.
        
        Args:
            file_path: Path to the JSON file
            timeout: Maximum time to wait for lock acquisition
            retry_delay: Delay between lock acquisition retries
            create_if_missing: If True, create file with default_data if it doesn't exist
            default_data: Data to write if creating new file (defaults to empty dict)
            
        Returns:
            Parsed JSON data
            
        Raises:
            Timeout: If lock cannot be acquired within timeout
            json.JSONDecodeError: If file contains invalid JSON
            FileNotFoundError: If file doesn't exist and create_if_missing is False
        """
        file_path = Path(file_path)
        lock_path = cls._get_lock_path(file_path)
        timeout = timeout or cls.default_timeout
        retry_delay = retry_delay or cls.default_retry_delay
        
        if default_data is None:
            default_data = {}
        
        lock = FileLock(lock_path, timeout=timeout)
        
        try:
            with lock.acquire(timeout=timeout):
                if not file_path.exists():
                    if create_if_missing:
                        # Create parent directories if needed
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(file_path, 'w') as f:
                            json.dump(default_data, f, indent=2)
                        log.info(f"Created new file: {file_path}")
                        return default_data
                    else:
                        raise FileNotFoundError(f"File not found: {file_path}")
                
                with open(file_path, 'r') as f:
                    return json.load(f)
                    
        except Timeout:
            log.error(f"Failed to acquire lock for reading {file_path} within {timeout}s")
            raise
    
    @classmethod
    def write_json_with_lock(
        cls,
        file_path: Union[str, Path],
        data: Any,
        timeout: Optional[float] = None,
        retry_delay: Optional[float] = None,
        indent: int = 2,
        ensure_ascii: bool = False
    ) -> None:
        """Write JSON file with file locking for thread/process safety.
        
        Args:
            file_path: Path to the JSON file
            data: Data to serialize to JSON
            timeout: Maximum time to wait for lock acquisition
            retry_delay: Delay between lock acquisition retries
            indent: JSON indentation level
            ensure_ascii: If True, escape non-ASCII characters
            
        Raises:
            Timeout: If lock cannot be acquired within timeout
            TypeError: If data cannot be serialized to JSON
        """
        file_path = Path(file_path)
        lock_path = cls._get_lock_path(file_path)
        timeout = timeout or cls.default_timeout
        retry_delay = retry_delay or cls.default_retry_delay
        
        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        lock = FileLock(lock_path, timeout=timeout)
        
        try:
            with lock.acquire(timeout=timeout):
                # Write to temporary file first for atomicity
                temp_path = file_path.with_suffix('.tmp')
                with open(temp_path, 'w') as f:
                    json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
                
                # Atomic rename
                temp_path.replace(file_path)
                
        except Timeout:
            log.error(f"Failed to acquire lock for writing {file_path} within {timeout}s")
            raise
    
    @classmethod
    def update_json_with_lock(
        cls,
        file_path: Union[str, Path],
        update_func: callable,
        timeout: Optional[float] = None,
        retry_delay: Optional[float] = None,
        create_if_missing: bool = True,
        default_data: Optional[Any] = None
    ) -> Any:
        """Update JSON file atomically with a custom function.
        
        This method reads the file, applies the update function, and writes
        the result back, all within a single lock acquisition.
        
        Args:
            file_path: Path to the JSON file
            update_func: Function that takes current data and returns updated data
            timeout: Maximum time to wait for lock acquisition
            retry_delay: Delay between lock acquisition retries
            create_if_missing: If True, create file with default_data if it doesn't exist
            default_data: Initial data if creating new file
            
        Returns:
            The updated data
            
        Raises:
            Timeout: If lock cannot be acquired within timeout
        """
        file_path = Path(file_path)
        lock_path = cls._get_lock_path(file_path)
        timeout = timeout or cls.default_timeout
        retry_delay = retry_delay or cls.default_retry_delay
        
        if default_data is None:
            default_data = {}
        
        lock = FileLock(lock_path, timeout=timeout)
        
        try:
            with lock.acquire(timeout=timeout):
                # Read current data
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        current_data = json.load(f)
                elif create_if_missing:
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    current_data = default_data
                else:
                    raise FileNotFoundError(f"File not found: {file_path}")
                
                # Apply update
                updated_data = update_func(current_data)
                
                # Write back atomically
                temp_path = file_path.with_suffix('.tmp')
                with open(temp_path, 'w') as f:
                    json.dump(updated_data, f, indent=2, ensure_ascii=False)
                
                temp_path.replace(file_path)
                
                return updated_data
                
        except Timeout:
            log.error(f"Failed to acquire lock for updating {file_path} within {timeout}s")
            raise
    
    @classmethod
    def with_file_lock(
        cls,
        file_path: Union[str, Path],
        timeout: Optional[float] = None
    ) -> FileLock:
        """Get a FileLock instance for manual lock management.
        
        This is useful when you need more control over the locking process
        or need to perform non-JSON operations.
        
        Args:
            file_path: Path to the file to lock
            timeout: Maximum time to wait for lock acquisition
            
        Returns:
            FileLock instance
            
        Example:
            ```python
            with FileLockHelper.with_file_lock("data.txt") as lock:
                # Perform thread-safe operations
                with open("data.txt", 'a') as f:
                    f.write("new data\\n")
            ```
        """
        lock_path = cls._get_lock_path(file_path)
        timeout = timeout or cls.default_timeout
        return FileLock(lock_path, timeout=timeout)
    
    @classmethod
    def cleanup_lock_files(cls, directory: Union[str, Path]) -> int:
        """Remove all lock files in a directory.
        
        This should only be called when you're certain no processes are
        using the locks (e.g., during cleanup or testing).
        
        Args:
            directory: Directory to clean up
            
        Returns:
            Number of lock files removed
        """
        directory = Path(directory)
        count = 0
        
        for lock_file in directory.rglob('*.lock'):
            try:
                lock_file.unlink()
                count += 1
            except Exception as e:
                log.warning(f"Failed to remove lock file {lock_file}: {e}")
        
        return count