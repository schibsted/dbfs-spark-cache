"""Tests for DataFrame extensions in dbfs_spark_cache."""
import os
import sys
import tempfile
import pytest
from unittest.mock import MagicMock

# Mock the problematic PySpark modules before any other imports
mock_udtf = MagicMock()
mock_udtf.AnalyzeArgument = MagicMock()
mock_udtf.AnalyzeResult = MagicMock()
sys.modules['pyspark.sql.udtf'] = mock_udtf

# Mock pyspark.sql.utils with the missing enum_to_value function
mock_utils = MagicMock()
mock_utils.enum_to_value = MagicMock()
if 'pyspark.sql.utils' in sys.modules:
    original_utils = sys.modules['pyspark.sql.utils']
    for attr in dir(original_utils):
        if not attr.startswith('_'):
            setattr(mock_utils, attr, getattr(original_utils, attr))
sys.modules['pyspark.sql.utils'] = mock_utils

# Now import our modules
from dbfs_spark_cache.config import config


@pytest.fixture
def mock_spark_session(mock_dataframe_extensions_databricks_env):
    """Fixture for mock spark session."""
    spark = MagicMock()
    mock_catalog = MagicMock()
    spark.catalog = mock_catalog
    spark.read = MagicMock()
    return spark


@pytest.fixture
def mock_dataframe(mock_spark_session, mock_dataframe_extensions_databricks_env):
    """Fixture for mock DataFrame."""
    df = MagicMock()
    df.sparkSession = mock_spark_session
    df._jdf = MagicMock()
    df.write = MagicMock()
    df.unpersist = MagicMock()

    # Set up StorageLevel mock
    storage_level = MagicMock()
    storage_level.useMemory = False
    storage_level.useDisk = False
    df.storageLevel = storage_level

    # Mock DataFrames use these methods we need to mock
    df.inputFiles = MagicMock(return_value=["s3://bucket/data/file1.parquet"])
    df._jdf.queryExecution = MagicMock()

    return df


@pytest.fixture
def temp_test_dir(mock_dataframe_extensions_databricks_env):
    """Create temp directory for tests."""
    test_dir = tempfile.mkdtemp()
    orig_cache_dir = config.SPARK_CACHE_DIR
    orig_cache_database = config.CACHE_DATABASE

    # Override config paths for test
    config.SPARK_CACHE_DIR = f"{test_dir}/spark_cache/"
    # For tests, we'll use a test-specific database name
    config.CACHE_DATABASE = "test_cache_db"

    # Create directories
    os.makedirs(config.SPARK_CACHE_DIR, exist_ok=True)

    yield test_dir

    # Restore original config
    config.SPARK_CACHE_DIR = orig_cache_dir
    config.CACHE_DATABASE = orig_cache_database

    # Clean up
    import shutil
    shutil.rmtree(test_dir)


# Import the required functions for DataFrame extensions
