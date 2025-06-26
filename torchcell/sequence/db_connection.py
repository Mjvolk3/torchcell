# torchcell/sequence/db_connection
# [[torchcell.sequence.db_connection]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/sequence/db_connection
# Test file: tests/torchcell/sequence/test_db_connection.py


import os.path as osp
import threading
from typing import Protocol, TypeVar, Generic, Iterator, Any

import gffutils
from gffutils import FeatureDB


class DatabaseProtocol(Protocol):
    """Protocol for database-like objects that need connection management."""
    
    def __getitem__(self, key: str) -> Any:
        """Get item by key/ID."""
        ...
    
    def features_of_type(self, featuretype: str) -> Iterator[Any]:
        """Get features of a specific type."""
        ...
    
    def region(self, region: tuple, completely_within: bool = False) -> Iterator[Any]:
        """Get features in a specific region."""
        ...
    
    def all_features(self) -> Iterator[Any]:
        """Iterate through all features in the database."""
        ...
    
    def featuretypes(self) -> Iterator[str]:
        """Get all feature types in the database."""
        ...
    
    def update(self, features: list[Any], merge_strategy: str = "merge") -> None:
        """Update features in the database."""
        ...
    
    def delete(self, id: str, feature_type: str) -> None:
        """Delete a feature from the database."""
        ...
    
    @property
    def conn(self) -> Any:
        """Access to the underlying database connection."""
        ...


T = TypeVar("T", bound=DatabaseProtocol)


class DatabaseConnectionManager(Generic[T]):
    """
    Manages database connections for multiprocessing scenarios.

    This class ensures that each process/thread gets its own database connection,
    avoiding SQLite's concurrent access issues while supporting pickling for
    multiprocessing spawn mode.

    Args:
        db_path: Path to the database file
        db_class: Class to instantiate for database connection (default: gffutils.FeatureDB)
        db_args: Additional arguments to pass to db_class constructor
        db_kwargs: Additional keyword arguments to pass to db_class constructor
    """

    def __init__(
        self, db_path: str, db_class: type[T] = FeatureDB, *db_args, **db_kwargs
    ):
        self.db_path = db_path
        self.db_class = db_class
        self.db_args = db_args
        self.db_kwargs = db_kwargs
        self._local = threading.local()

    def get_connection(self) -> T:
        """
        Get thread/process-local database connection.

        Returns:
            Database connection instance

        Raises:
            FileNotFoundError: If database file doesn't exist
        """
        if not hasattr(self._local, "db") or self._local.db is None:
            if not osp.exists(self.db_path):
                raise FileNotFoundError(f"Database not found at {self.db_path}")

            # Create new connection for this thread/process
            self._local.db = self.db_class(
                self.db_path, *self.db_args, **self.db_kwargs
            )

        return self._local.db

    def close_connection(self):
        """Close the current thread/process connection if it exists."""
        if hasattr(self._local, "db") and self._local.db is not None:
            # FeatureDB doesn't have a close method, but we can clear the reference
            # The underlying SQLite connection will be garbage collected
            self._local.db = None

    def __reduce__(self):
        """
        Support pickling by only serializing the configuration.

        When unpickled, a new manager is created with the same configuration,
        but no active connections (they'll be created lazily in the new process).
        """
        return (
            self.__class__,
            (self.db_path, self.db_class) + self.db_args,
            self.db_kwargs,
        )

    def __getstate__(self):
        """Custom state for pickling - exclude the thread local storage."""
        state = self.__dict__.copy()
        # Remove thread-local storage which can't be pickled
        state.pop("_local", None)
        return state

    def __setstate__(self, state):
        """Restore state after unpicskling."""
        self.__dict__.update(state)
        # Recreate thread-local storage
        self._local = threading.local()


# Convenience type alias for gffutils databases
GffutilsConnectionManager = DatabaseConnectionManager[FeatureDB]
