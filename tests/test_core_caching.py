import os
import sys
import tempfile
from unittest.mock import ANY, MagicMock, patch
import pytest
import pandas as pd
from datetime import datetime

from pyspark.sql import DataFrame, SparkSession # For type hinting and session
from py4j.protocol import Py4JJavaError # For one of the tests

# Add project root to sys.path to allow importing dbfs_spark_cache
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dbfs_spark_cache import core_caching
from dbfs_spark_cache.config import config

# Fixtures
# Remove the real spark session fixture as we will use a mock
# @pytest.fixture(scope="module")
# def spark():
#     return SparkSession.builder.master("local[1]").appName("test_core_caching").getOrCreate() # type: ignore[attr-defined]

@pytest.fixture
def mock_spark_session():
    # Return a MagicMock instead of a real spark session
    mock_session = MagicMock(spec=SparkSession)
    # Mock the read attribute to return a mock DataFrameReader
    mock_session.read = MagicMock()
    return mock_session

@pytest.fixture
def mock_dataframe(mock_spark_session): # mock_spark_session is now a MagicMock
    df = MagicMock(spec=DataFrame)
    df.sparkSession = mock_spark_session
    df._jdf = MagicMock()
    df.write = MagicMock()
    df.inputFiles = MagicMock(return_value=["s3://bucket/data/file1.parquet"])
    return df

@pytest.fixture
def temp_test_dir():
    test_dir = tempfile.mkdtemp()
    original_cache_dir = config.SPARK_CACHE_DIR
    original_cache_database = config.CACHE_DATABASE
    # Ensure dbfs: prefix if dbutils is used by the functions under test for path construction
    config.SPARK_CACHE_DIR = f"dbfs:{test_dir}/spark_cache/"
    config.CACHE_DATABASE = "test_cache_db"

    # Create the directory if core_caching functions expect it (e.g., for local fallbacks or direct os ops)
    # For dbutils.fs.mkdirs, this might not be strictly necessary if dbutils itself is fully mocked.
    # However, if any part of the code uses os.makedirs or similar, this is useful.
    # Given the functions in core_caching.py use dbutils.fs.mkdirs, this os.makedirs might be redundant
    # unless there's a fallback path not using dbutils.
    # For safety with temp_test_dir, let's ensure the local equivalent exists if code ever falls back.
    local_cache_dir_equivalent = f"{test_dir}/spark_cache/"
    os.makedirs(local_cache_dir_equivalent, exist_ok=True)

    yield test_dir
    config.SPARK_CACHE_DIR = original_cache_dir
    config.CACHE_DATABASE = original_cache_database
    import shutil
    shutil.rmtree(test_dir)

# Tests from test_dataframe_extensions_isolated.py
@patch('dbfs_spark_cache.core_caching.dbutils') # dbutils is used by core_caching functions
@patch('dbfs_spark_cache.core_caching.spark') # Patch core_caching.spark
def test_read_dbfs_cache_if_exist_cache_hit(mock_spark_in_core_caching, mock_core_dbutils, mock_dataframe):
    # mock_spark_in_core_caching is the MagicMock that replaces core_caching.spark

    with patch('dbfs_spark_cache.core_caching.get_input_dir_mod_datetime', autospec=True) as mock_get_input_dir_mod_datetime, \
         patch('dbfs_spark_cache.core_caching.get_query_plan', autospec=True) as mock_get_query_plan:

        mock_dt = datetime(2023, 1, 1)
        input_mod_datetime_val = {"s3://bucket/data": mock_dt}
        query_plan_val = "mock_query_plan"

        mock_get_input_dir_mod_datetime.return_value = input_mod_datetime_val
        mock_get_query_plan.return_value = query_plan_val

        mock_core_dbutils.fs.head.return_value = "Some metadata content"

        mock_cached_df_from_table = MagicMock(spec=DataFrame)
        # Mock the table method on the patched spark session
        mock_spark_in_core_caching.read.table.return_value = mock_cached_df_from_table

        result = core_caching.read_dbfs_cache_if_exist(mock_dataframe)

        assert result is mock_cached_df_from_table
        mock_core_dbutils.fs.head.assert_called_once()
        mock_spark_in_core_caching.read.table.assert_called_once() # Assert on the patched spark

@patch('dbfs_spark_cache.core_caching.dbutils') # dbutils is used by core_caching functions
@patch('dbfs_spark_cache.core_caching.spark') # Patch core_caching.spark
def test_read_dbfs_cache_if_exist_cache_miss(mock_spark_in_core_caching, mock_core_dbutils, mock_dataframe):
    # mock_spark_in_core_caching is the MagicMock that replaces core_caching.spark

    # Simulate metadata file not found by raising a FileNotFoundError exception
    file_not_found_error = Exception("java.io.FileNotFoundException: No such file or directory")

    # Set the side_effect to raise the exception
    mock_core_dbutils.fs.head.side_effect = file_not_found_error

    with patch('dbfs_spark_cache.core_caching.get_input_dir_mod_datetime', autospec=True) as mock_get_input_dir_mod_datetime, \
         patch('dbfs_spark_cache.core_caching.get_query_plan', autospec=True) as mock_get_query_plan:
        mock_get_input_dir_mod_datetime.return_value = {"s3://bucket/data": pd.Timestamp("2023-01-01")}
        mock_get_query_plan.return_value = "mock_query_plan"

        result = core_caching.read_dbfs_cache_if_exist(mock_dataframe)
        assert result is None
        mock_core_dbutils.fs.head.assert_called_once() # This should now be called

@patch('dbfs_spark_cache.core_caching.dbutils')
@patch('dbfs_spark_cache.core_caching.spark') # Patch core_caching.spark
def test_write_dbfs_cache_with_existing_identical_cache(mock_spark_in_core_caching, mock_core_dbutils, mock_dataframe, temp_test_dir):
    # This test focuses on core_caching.write_dbfs_cache behavior when metadata is identical.
    # It calls _write_standard_cache internally.

    input_dt_val = {"s3://test-bucket/data": datetime(2023, 1, 1)}
    query_plan_val = "SELECT * FROM test_table"

    # get_table_cache_info will generate the expected metadata text and paths
    expected_hash, expected_cache_path, expected_meta_path, expected_metadata_txt = \
        core_caching.get_table_cache_info(
            input_dir_mod_datetime=input_dt_val,
            query_plan=query_plan_val,
            cache_path_base=config.SPARK_CACHE_DIR # Use configured SPARK_CACHE_DIR
        )

    # Simulate that dbutils.fs.head for the metadata file returns the exact same metadata content
    mock_core_dbutils.fs.head.return_value = expected_metadata_txt

    # Mock _write_standard_cache to inspect its call and prevent actual write
    # However, for this test, we want to test the logic *within* _write_standard_cache
    # that skips writing if metadata is identical.
    # So, we let _write_standard_cache be called, but mock df.write inside it.

    # Mock the DataFrame's write attribute that _write_standard_cache would call
    mock_dataframe_writer = MagicMock()
    mock_dataframe.write.format.return_value.mode.return_value.saveAsTable = mock_dataframe_writer

    # Mock dbutils.fs.put to check it's not called for metadata if identical
    mock_dbutils_fs_put = MagicMock()
    mock_core_dbutils.fs.put = mock_dbutils_fs_put

    # Mock spark.read.table for the return value
    mock_returned_df = MagicMock(spec=DataFrame)
    # Mock the table method on the patched spark session
    mock_spark_in_core_caching.read.table.return_value = mock_returned_df

    result = core_caching.write_dbfs_cache(
        mock_dataframe,
        replace=False, # Crucial: do not replace if identical
        query_plan=query_plan_val,
        input_dir_mod_datetime=input_dt_val,
        cache_path=config.SPARK_CACHE_DIR # Pass explicitly for clarity
    )

    mock_core_dbutils.fs.head.assert_called_once_with(expected_meta_path, 1024*1024) # Check metadata read
    mock_dataframe_writer.assert_not_called() # DataFrame should not be written
    mock_dbutils_fs_put.assert_not_called()   # Metadata should not be re-written

    # write_dbfs_cache should still return the DataFrame read from the existing table
    mock_spark_in_core_caching.read.table.assert_called_once_with(f"{config.CACHE_DATABASE}.{expected_hash}")
    assert result is mock_returned_df

# Test from test_dbfs_spark_cache.py
@patch('dbfs_spark_cache.core_caching.log')
@patch('dbfs_spark_cache.core_caching.datetime')
def test_get_input_dir_mod_datetime_handles_schema_change(mock_core_datetime, mock_core_log, mock_dataframe):
    # mock_dataframe is from fixture, ensure it has a sparkSession
    # Assign a mock spark session to the dataframe
    mock_dataframe.sparkSession = MagicMock(spec=SparkSession)

    # Create a standard exception with the Delta schema change error message
    error_message = (
        "py4j.protocol.Py4JJavaError: An error occurred while calling o123.inputFiles.\n"
        ": org.apache.spark.sql.delta.DeltaAnalysisException: DELTA_SCHEMA_CHANGE_SINCE_ANALYSIS: "
        "The schema of your Delta table has changed in an incompatible way since your DataFrame or DeltaTable object was created."
    )
    mock_dataframe.inputFiles.side_effect = Exception(error_message)

    fixed_time = datetime(2024, 1, 1, 12, 0, 0)
    expected_result = {"<schema_changed_placeholder>": fixed_time}

    mock_core_datetime.now.return_value = fixed_time
    mock_core_datetime.datetime = datetime # If core_caching.datetime.datetime is used

    result = core_caching.get_input_dir_mod_datetime(mock_dataframe)

    mock_dataframe.inputFiles.assert_called_once()
    assert result == expected_result, "Expected dict with placeholder and current time on schema change error"
    mock_core_log.warning.assert_called_once()
    args, _ = mock_core_log.warning.call_args
    assert "Could not get input files due to Delta schema change" in args[0]
    assert "Forcing cache invalidation" in args[0]
