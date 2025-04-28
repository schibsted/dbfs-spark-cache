"""Pytest configuration file for dbfs-spark-cache tests."""
import os
import pytest
import sys
from datetime import datetime
from unittest.mock import MagicMock

# Import the fixture from mocks_databricks.py
from .mocks_databricks import mock_dataframe_extensions_databricks_env

@pytest.fixture
def mock_spark_session():
    """Fixture for mock spark session."""
    spark = MagicMock()
    mock_catalog = MagicMock()
    spark.catalog = mock_catalog
    spark.read = MagicMock()
    return spark

class MockFileInfo:
    """Mock FileInfo class for dbutils.fs.ls results."""
    def __init__(self, name, path, size=1024, modificationTime=None):
        self.name = name
        self.path = path
        self.size = size
        self.modificationTime = modificationTime or int(datetime.now().timestamp() * 1000)

@pytest.fixture(scope="function", autouse=True)
def patch_databricks_runtime(monkeypatch, mock_spark_session): # Use mock_spark_session fixture
    """Patch dbutils.fs.ls and databricks.sdk.runtime.spark."""

    # --- Mock dbutils.fs.ls ---
    def mock_ls_function(path):
        """Mock function for dbutils.fs.ls that handles s3:// paths"""
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
                ),
                # Add a different type of file to test filtering
                MockFileInfo(
                    name="_delta_log/",
                    path=f"{path}/_delta_log/",
                    modificationTime=int(datetime(2023, 1, 1).timestamp() * 1000)
                )
            ]
        # Return empty list for other paths
        return []

    # Set up the monkeypatch to replace dbutils.fs.ls and spark
    try:
        import databricks.sdk.runtime
        # Patch dbutils.fs.ls
        monkeypatch.setattr(databricks.sdk.runtime.dbutils.fs, 'ls', mock_ls_function)
        # Patch spark to use the pytest fixture mock_spark_session
        monkeypatch.setattr(databricks.sdk.runtime, 'spark', mock_spark_session)
    except (ImportError, AttributeError, Exception) as e:
        # If patching fails for any reason, print a warning but continue
        print(f"\n[pytest warning] Failed to patch databricks.sdk.runtime: {e}")
        pass

    # --- Patch dbfs_spark_cache.caching.last_mod_datetime_from_s3_dir ---
    # (This part seems less critical for the ModuleNotFoundError, but keep it)
    try:
        # Create a patched version of the function
        def mock_last_mod_datetime_from_s3_dir(dir_path):
            """Mocked version that returns a fixed date for s3 paths and handles local paths normally."""
            if "s3://" in dir_path:
                return datetime(2023, 1, 1)

            # For non-s3 paths, use the normal implementation or return None
            return None

        # Apply the patch
        monkeypatch.setattr(
            'dbfs_spark_cache.caching.last_mod_datetime_from_s3_dir',
            mock_last_mod_datetime_from_s3_dir
        )
    except (ImportError, AttributeError):
        pass
