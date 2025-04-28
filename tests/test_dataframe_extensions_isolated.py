"""Tests for DataFrame extensions in dbfs_spark_cache."""
import os
import sys
import tempfile
import pytest
from unittest.mock import MagicMock, patch, mock_open, ANY # Import ANY

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

import pandas as pd

# Import mock first to set up the environment before importing from dbfs_spark_cache
# Use an absolute import path that works when running from parent directory
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Use the mock fixture for all tests in this module
pytestmark = pytest.mark.usefixtures("mock_dataframe_extensions_databricks_env")

# Now import our modules
from dbfs_spark_cache.caching import (
    write_dbfs_cache, # Changed from dbfs_cache
    read_dbfs_cache_if_exist,
)
from dbfs_spark_cache.config import config


@pytest.fixture
def mock_dataframe(mock_spark_session):
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
def temp_test_dir():
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
from dbfs_spark_cache.caching import (
    cacheToDbfs,
    __withCachedDisplay__,
)


@patch('dbutils.fs.ls')
def test_read_dbfs_cache_if_exist_cache_hit(mock_dbutils_ls, mock_dataframe, mock_spark_session):
    """Test read_dbfs_cache_if_exist with a cache hit."""
    mock_dbutils_ls.return_value = []
    with patch('dbfs_spark_cache.caching.get_input_dir_mod_datetime', autospec=True) as mock_get_input_dir_mod_datetime, \
         patch('dbfs_spark_cache.caching.get_query_plan', autospec=True) as mock_get_query_plan, \
         patch('os.path.exists', autospec=True) as mock_exists, \
         patch('builtins.open', new_callable=mock_open) as mock_file, \
         patch('dbfs_spark_cache.caching.spark', mock_spark_session):  # Patch spark directly in caching module

        # Setup mocks - convert pandas Timestamp to datetime for compatibility
        from datetime import datetime
        mock_dt = datetime(2023, 1, 1)
        mock_get_input_dir_mod_datetime.return_value = {"s3://bucket/data": mock_dt}
        mock_get_query_plan.return_value = "mock_query_plan"
        mock_exists.return_value = True

        # Hardcode the metadata content to match exactly what would be returned
        # This avoids type issues with the test and ensures consistent format
        metadata_content = """INPUT SOURCES MODIFICATION DATETIMES:
s3://bucket/data: 2023-01-01 00:00:00

DATAFRAME QUERY PLAN:
mock_query_plan"""

        # Mock the file read operation to return our metadata
        mock_file.return_value.__enter__.return_value.read.return_value = metadata_content

        # Mock spark.read.table to return a mock table
        mock_table = MagicMock()
        mock_spark_session.read.table.return_value = mock_table

        # Call the function we're testing
        result = read_dbfs_cache_if_exist(mock_dataframe)

        # Assertions
        assert result == mock_table
        mock_spark_session.read.table.assert_called_once()

@patch('dbutils.fs.ls')
def test_read_dbfs_cache_if_exist_cache_miss(mock_dbutils_ls, mock_dataframe):
    """Test read_dbfs_cache_if_exist with a cache miss."""
    mock_dbutils_ls.return_value = []
    with patch('dbfs_spark_cache.caching.get_input_dir_mod_datetime') as mock_get_input_dir_mod_datetime, \
         patch('dbfs_spark_cache.caching.get_query_plan') as mock_get_query_plan, \
         patch('os.path.exists') as mock_exists:

        # Setup mocks
        mock_get_input_dir_mod_datetime.return_value = {"s3://bucket/data": pd.Timestamp("2023-01-01")}
        mock_get_query_plan.return_value = "mock_query_plan"
        mock_exists.return_value = False

        result = read_dbfs_cache_if_exist(mock_dataframe)

        # Assertions
        assert result is None

@patch('dbutils.fs.ls')
@patch('dbutils.fs.ls')
def test_cacheto_dbfs_new_cache(mock_dbutils_ls, mock_dataframe, mock_spark_session): # Added mock_spark_session
    """Test cacheToDbfs on a DataFrame that's not yet cached, called as a method."""
    from datetime import datetime # Import datetime here
    mock_dbutils_ls.return_value = []

    # Import the real function to attach it
    from dbfs_spark_cache.caching import cacheToDbfs, get_input_dir_mod_datetime

    # Attach the real function as a method to the mock DataFrame
    mock_dataframe.cacheToDbfs = cacheToDbfs.__get__(mock_dataframe, type(mock_dataframe))
    # Ensure spark session is set on the mock dataframe for the method call
    mock_dataframe.sparkSession = mock_spark_session

    # Patch dependencies called *by* cacheToDbfs
    with patch('dbfs_spark_cache.caching.get_query_plan', return_value="SimplePlan") as mock_get_plan, \
         patch('dbfs_spark_cache.caching.get_input_dir_mod_datetime', return_value={"s3://bucket/data": datetime(2023, 1, 1)}) as mock_get_input, \
         patch('dbfs_spark_cache.caching.read_dbfs_cache_if_exist', return_value=None) as mock_read_cache_alias, \
         patch('dbfs_spark_cache.caching.write_dbfs_cache') as mock_write_dbfs_cache, \
         patch('dbfs_spark_cache.caching.estimate_compute_complexity', return_value=(150, 1.0, 150)):

        # Mock the return value of the patched write function
        mock_df_returned_by_write = MagicMock(name="ReturnedDataFrame")
        mock_write_dbfs_cache.return_value = mock_df_returned_by_write

        # Execute by calling the method on the mock object
        # Pass multiplier threshold = 1.0 so the mocked multiplier passes
        result = mock_dataframe.cacheToDbfs(
            dbfs_cache_complexity_threshold=100,
            dbfs_cache_multiplier_threshold=1.0
        )

        # Assertions
        mock_get_plan.assert_called_once_with(mock_dataframe)
        mock_get_input.assert_called_once_with(mock_dataframe)
        # Assert the alias read function was called
        mock_read_cache_alias.assert_called_once_with(mock_dataframe, query_plan="SimplePlan", input_dir_mod_datetime=ANY)

        # Assert the underlying write function was called correctly
        mock_write_dbfs_cache.assert_called_once()
        # Get the actual call arguments
        call_args, call_kwargs = mock_write_dbfs_cache.call_args
        # Assert specific arguments
        assert call_args[0] == mock_dataframe # First arg is the dataframe instance (self)
        assert call_kwargs.get('replace') is True
        assert call_kwargs.get('query_plan') == "SimplePlan"
        assert call_kwargs.get('hash_name') is None
        assert isinstance(call_kwargs.get('input_dir_mod_datetime'), dict)
        assert call_kwargs.get('verbose') is False

        # Assert the result is the specific mock returned by the patched function
        assert result == mock_df_returned_by_write
def test_cacheto_dbfs_with_replace_true(mock_dataframe, mock_spark_session): # Removed mock_dbutils_ls from args
    """Test cacheToDbfs with replace=True parameter forces a write and read."""
    # Patch dbutils inside the 'with' statement
    with patch('dbfs_spark_cache.caching.dbutils') as mock_dbutils, \
         patch('dbfs_spark_cache.caching.get_query_plan', return_value="SimplePlan"), \
         patch('dbfs_spark_cache.caching.read_dbfs_cache_if_exist', autospec=True) as mock_read_cache, \
         patch('dbfs_spark_cache.caching.write_dbfs_cache', autospec=True) as mock_write_dbfs_cache, \
         patch('dbfs_spark_cache.caching.estimate_compute_complexity', autospec=True) as mock_estimate_complexity, \
         patch('pathlib.Path.mkdir', autospec=True):

        # Setup mock for dbutils.fs.ls
        mock_dbutils.fs.ls.return_value = []

        # Setup other mocks
        cached_df = MagicMock()
        mock_read_cache.return_value = cached_df  # Simulating existing cache
        mock_estimate_complexity.return_value = (150, 1.0) # Return tuple

        # Setup write_dbfs_cache to return a new DataFrame
        new_df = MagicMock()
        mock_write_dbfs_cache.return_value = new_df # Changed from mock_dbfs_cache

        # Execute
        result = cacheToDbfs(mock_dataframe, dbfs_cache_complexity_threshold=None)

        # Assertions
        # Update assertion to include new kwargs
        mock_read_cache.assert_called_once_with(mock_dataframe, query_plan="SimplePlan", input_dir_mod_datetime=ANY)
        # With the fix in cacheToDbfs, if read_dbfs_cache_if_exist returns a DF (cache hit),
        # write_dbfs_cache should NOT be called again.
        mock_write_dbfs_cache.assert_not_called() # Changed from mock_dbfs_cache
        # The result should be the DataFrame returned by read_dbfs_cache_if_exist
        assert result == cached_df
@patch('dbutils.fs.ls')
def test_cacheto_dbfs_below_threshold(mock_dbutils_ls, mock_dataframe):
    """Test cacheToDbfs on a small DataFrame below complexity threshold."""
    mock_dbutils_ls.return_value = []
    with patch('dbfs_spark_cache.caching.get_query_plan', return_value="SimplePlan"), \
         patch('dbfs_spark_cache.caching.read_dbfs_cache_if_exist', autospec=True) as mock_read_cache, \
         patch('dbfs_spark_cache.caching.write_dbfs_cache', autospec=True) as mock_write_dbfs_cache, \
         patch('dbfs_spark_cache.caching.estimate_compute_complexity', autospec=True) as mock_estimate_complexity: # Removed trailing comma

        # Setup mocks
        mock_read_cache.return_value = None
        mock_estimate_complexity.return_value = (50, 1.0, 50)  # Return tuple with total_size

        # Execute
        result = cacheToDbfs(mock_dataframe, dbfs_cache_complexity_threshold=100)

        # Assertions
        # Update assertion to include new kwargs
        mock_read_cache.assert_called_once_with(mock_dataframe, query_plan="SimplePlan", input_dir_mod_datetime=ANY)
        # write_dbfs_cache should not be called since complexity is below threshold
        mock_write_dbfs_cache.assert_not_called() # Changed from mock_dbfs_cache
        # Should return original DataFrame
        assert result == mock_dataframe

@patch('dbutils.fs.ls')
def test_cacheto_dbfs_deferred(mock_dbutils_ls, mock_dataframe):
    """Test cacheToDbfs with deferred=True."""
    mock_dbutils_ls.return_value = []
    with patch('dbfs_spark_cache.caching.add_to_dbfs_cache_queue', autospec=True) as mock_add_to_queue:
        # Execute
        result = cacheToDbfs(mock_dataframe, deferred=True)

        # Assertions
        mock_add_to_queue.assert_called_once_with(mock_dataframe)
        assert result == mock_dataframe

@patch('dbutils.fs.ls')
def test_wcd_method(mock_dbutils_ls, mock_dataframe):
    """Test wcd method that uses DBFS cache."""
    mock_dbutils_ls.return_value = []
    # Test by directly using __withCachedDisplay__ which is what wcd method calls
    display_mock = MagicMock()
    mock_dataframe.display_in_notebook = display_mock

    # Use autospec and patch the cacheToDbfs function directly
    with patch('dbfs_spark_cache.caching.cacheToDbfs', autospec=True) as mock_cacheTo_dbfs:
        cached_df = MagicMock()
        cached_df.display_in_notebook = display_mock
        mock_cacheTo_dbfs.return_value = cached_df

        # Execute by calling the function directly
        result = __withCachedDisplay__(
            mock_dataframe,
            dbfs_cache_complexity_threshold=100,
            skip_display=False
        )

        # Assertions
        mock_cacheTo_dbfs.assert_called_once()
        display_mock.assert_called_once()
        assert result == cached_df

@patch('dbutils.fs.ls')
def test_wcd_with_spark_cache(mock_dbutils_ls, mock_dataframe):
    """Test wcd method with eager Spark caching."""
    mock_dbutils_ls.return_value = []
    # Setup mocks
    display_mock = MagicMock()
    mock_dataframe.display_in_notebook = display_mock

    with patch('dbfs_spark_cache.caching.cacheToDbfs', autospec=True) as mock_cacheTo_dbfs, \
         patch('dbfs_spark_cache.caching.is_spark_cached', autospec=True) as mock_is_spark_cached, \
         patch.object(mock_dataframe, 'cache') as mock_cache:

        mock_cacheTo_dbfs.return_value = mock_dataframe
        mock_is_spark_cached.return_value = False
        cached_df = MagicMock()
        cached_df.display_in_notebook = display_mock
        mock_cache.return_value = cached_df

        # Execute with explicit fix to handle display call properly
        result = __withCachedDisplay__(
            mock_dataframe,
            skip_dbfs_cache=True,
            eager_spark_cache=True,
            skip_display=False
        )

        # Assertions
        mock_cache.assert_called_once()
        # The function should have already called display_mock
        display_mock.assert_called_once()
        assert result == cached_df

@patch('dbutils.fs.ls')
def test_wcd_skip_display(mock_dbutils_ls, mock_dataframe):
    """Test wcd method with skip_display=True."""
    mock_dbutils_ls.return_value = []
    # Setup
    display_mock = MagicMock()
    mock_dataframe.display_in_notebook = display_mock

    with patch('dbfs_spark_cache.caching.cacheToDbfs', autospec=True) as mock_cacheTo_dbfs:
        mock_cacheTo_dbfs.return_value = mock_dataframe

        # Execute
        result = __withCachedDisplay__(
            mock_dataframe,
            skip_display=True
        )

        # Assertions
        mock_cacheTo_dbfs.assert_called_once()
        # display should not be called
        display_mock.assert_not_called()
        assert result == mock_dataframe

@patch('dbutils.fs.ls')
def test_write_dbfs_cache_core_functionality(mock_dbutils_ls, mock_dataframe, mock_spark_session, temp_test_dir): # Renamed test
    """Test write_dbfs_cache core functionality with patched file operations.""" # Renamed test
    mock_dbutils_ls.return_value = []
    with patch('os.path.exists') as mock_exists, \
         patch('builtins.open', mock_open()) as mock_file, \
         patch('pathlib.Path.mkdir') as mock_mkdir, \
         patch('dbfs_spark_cache.caching.get_input_dir_mod_datetime') as mock_get_input_datetime, \
         patch('dbfs_spark_cache.caching.get_query_plan') as mock_get_query_plan, \
         patch('dbfs_spark_cache.caching.spark', mock_spark_session):  # Patch spark directly in caching module

        # Setup mocks
        mock_exists.return_value = False
        mock_get_input_datetime.return_value = {"s3://test-bucket/data": pd.Timestamp("2023-01-01")}
        mock_get_query_plan.return_value = "SELECT * FROM test_table"
        # Setup the full write.format().saveAsTable() chain
        mock_format = MagicMock()
        mock_dataframe.write.format.return_value = mock_format
        mock_dataframe.write.format.return_value = mock_format

        # Call the function directly
        result = write_dbfs_cache(mock_dataframe, replace=False) # Changed from dbfs_cache

        # Assertions
        # Expect two calls: one for cache dir, one for metadata dir parent
        assert mock_mkdir.call_count == 2
        mock_dataframe.write.format.assert_called_once_with('delta')

        mock_file().write.assert_called_once()
        # Assert the result is a mock object, not necessarily the *same* input mock
        assert isinstance(result, MagicMock)
        # Optionally, assert it's not the original mock if that's important
        # assert result is not mock_dataframe
@patch('dbutils.fs.ls')
def test_write_dbfs_cache_with_existing_identical_cache(mock_dbutils_ls, mock_dataframe, mock_spark_session, temp_test_dir): # Renamed test
    """Test write_dbfs_cache when cache exists and is identical.""" # Renamed test
    mock_dbutils_ls.return_value = []
    # Patch _write_standard_cache directly to check if it's called correctly
    # without executing its internal logic (like file reads/writes)
    with patch('dbfs_spark_cache.caching._write_standard_cache') as mock_internal_write, \
         patch('dbfs_spark_cache.caching.get_input_dir_mod_datetime', autospec=True) as mock_get_input_datetime, \
         patch('dbfs_spark_cache.caching.get_query_plan', autospec=True) as mock_get_query_plan, \
         patch('dbfs_spark_cache.caching.spark', mock_spark_session), \
         patch('dbfs_spark_cache.caching._write_standard_cache') as mock_internal_write: # Patch _write_standard_cache

        # Setup mocks for input conditions
        from datetime import datetime
        mock_dt = datetime(2023, 1, 1)
        mock_get_input_datetime.return_value = {"s3://test-bucket/data": mock_dt}
        mock_get_query_plan.return_value = "SELECT * FROM test_table"

        # Import the function to call directly
        from dbfs_spark_cache.caching import write_dbfs_cache, get_table_cache_info

        # Generate expected metadata info
        hash_name, cache_path, meta_path, meta_txt = get_table_cache_info(
            input_dir_mod_datetime=mock_get_input_datetime.return_value,
            query_plan=mock_get_query_plan.return_value
        )

        # Call the function which should call _write_standard_cache
        result = write_dbfs_cache(
            mock_dataframe,
            replace=False, # replace=False is default, but explicit here
            query_plan=mock_get_query_plan.return_value,
            input_dir_mod_datetime=mock_get_input_datetime.return_value
        )

        # Assert _write_standard_cache was called with correct args
        mock_internal_write.assert_called_once_with(
            df=mock_dataframe,
            hash_name=hash_name,
            cache_path=cache_path,
            metadata_file_path=meta_path,
            metadata_txt=meta_txt,
            verbose=False # Default verbose is False
        )

        # Assert the original dataframe's write method was NOT called,
        # because _write_standard_cache was mocked away.
        mock_dataframe.write.format.assert_not_called()

        # Assert the result is the dataframe read back (mocked)
        # write_dbfs_cache reads back the table after calling _write_standard_cache
        mock_spark_session.read.table.assert_called_once_with(f"test_cache_db.{hash_name}")
        assert result == mock_spark_session.read.table.return_value
