"""Shared mocks for Databricks environment."""
import sys
import pytest
from unittest.mock import MagicMock
from datetime import datetime

class MockFileInfo:
    """Mock file info class for dbutils.fs operations."""
    def __init__(self, name, path, size=1024, modificationTime=None):
        self.name = name
        self.path = path
        self.size = size
        self.modificationTime = modificationTime or int(datetime.now().timestamp() * 1000)

@pytest.fixture(scope="function")
def mock_dataframe_extensions_databricks_env():
    """Fixture to mock Databricks environment."""
    # Store original modules
    original_databricks = sys.modules.get('databricks')
    original_databricks_sdk = sys.modules.get('databricks.sdk')
    original_databricks_sdk_runtime = sys.modules.get('databricks.sdk.runtime')
    original_dbutils = sys.modules.get('dbutils')  # Store original dbutils if it exists

    # Create mocks
    mock_dbutils = MagicMock()
    mock_fs = MagicMock()

    # Define a custom ls function that handles s3:// paths
    def mock_ls_function(path):
        # For s3:// paths, return mock file infos
        if path.startswith("s3://"):
            return [
                MockFileInfo(
                    name="file1.parquet",
                    path=f"{path}/file1.parquet",
                    modificationTime=int(datetime(2023, 1, 1).timestamp() * 1000)
                ),
                MockFileInfo(
                    name="file2.parquet",
                    path=f"{path}/file2.parquet",
                    modificationTime=int(datetime(2023, 1, 1).timestamp() * 1000)
                )
            ]
        # Return empty list for other paths
        return []

    # Assign the mock function
    mock_fs.ls = mock_ls_function
    mock_dbutils.fs = mock_fs

    # Create complete mock hierarchy
    mock_runtime = MagicMock()
    mock_runtime.dbutils = mock_dbutils
    mock_runtime.display = MagicMock()
    mock_runtime.spark = MagicMock()

    # Install mocks
    sys.modules['databricks'] = MagicMock()
    sys.modules['databricks.sdk'] = MagicMock()
    sys.modules['databricks.sdk.runtime'] = mock_runtime

    # Also install dbutils as a top-level module for direct imports
    sys.modules['dbutils'] = mock_dbutils

    # Yield the mock objects needed by the tests
    yield {
        'spark_session': mock_runtime.spark,
        'display_fun': mock_runtime.display
    }

    # Restore original modules
    if original_databricks is not None:
        sys.modules['databricks'] = original_databricks
    else:
        del sys.modules['databricks']

    if original_databricks_sdk is not None:
        sys.modules['databricks.sdk'] = original_databricks_sdk
    else:
        del sys.modules['databricks.sdk']

    if original_databricks_sdk_runtime is not None:
        sys.modules['databricks.sdk.runtime'] = original_databricks_sdk_runtime
    else:
        del sys.modules['databricks.sdk.runtime']

    # Restore original dbutils module or remove it
    if original_dbutils is not None:
        sys.modules['dbutils'] = original_dbutils
    elif 'dbutils' in sys.modules:
        del sys.modules['dbutils']
