"""Tests for dbfs_spark_cache package."""
import logging
import sys
from unittest.mock import ANY, MagicMock, mock_open, patch  # Import ANY

import pytest

# Import the fixture from mocks_databricks instead of redefining it

# Apply the fixture to all tests in this module
pytestmark = pytest.mark.usefixtures("mock_dataframe_extensions_databricks_env")

# Removed global mocking setup - mocks will be applied per test using @patch
def test_import_dbfs_spark_cache():
    """Test that the package can be imported."""
    try:
        import dbfs_spark_cache  # noqa: F401
        assert True
    except ImportError as e:
        assert False, f"Import failed for dbfs_spark_cache: {str(e)}"

def test_extend_dataframe_methods(mock_dataframe_extensions_databricks_env):
    """Test extend_dataframe_methods function modifies the global DataFrame class."""
    # Import the real DataFrame and extend_dataframe_methods
    from pyspark.sql import DataFrame

    from dbfs_spark_cache import extend_dataframe_methods

    # The fixture provides the mock spark session and display function
    mock_spark_session = mock_dataframe_extensions_databricks_env['spark_session']
    mock_display_fn = mock_dataframe_extensions_databricks_env['display_fun']

    # Call the initialization function
    extend_dataframe_methods(spark_session=mock_spark_session, display_fun=mock_display_fn)

    # Assert that the methods have been added to the global DataFrame class
    assert hasattr(DataFrame, 'withCachedDisplay')
    assert hasattr(DataFrame, 'wcd')
    assert hasattr(DataFrame, 'display_in_notebook')
    assert hasattr(DataFrame, 'cacheToDbfs')
    assert hasattr(DataFrame, 'clearDbfsCache')

    # Optionally, check if the added attributes are callable (basic check)
    assert callable(DataFrame.withCachedDisplay) # type: ignore[attr-defined]
    assert callable(DataFrame.wcd) # type: ignore[attr-defined]
    assert callable(DataFrame.display_in_notebook) # type: ignore[attr-defined]
    assert callable(DataFrame.cacheToDbfs) # type: ignore[attr-defined]
    assert callable(DataFrame.clearDbfsCache) # type: ignore[attr-defined]


def test_estimate_compute_complexity():
    """Test the real estimate_compute_complexity logic with mocked DataFrame and file sizes."""

    import dbfs_spark_cache.query_complexity_estimation as qce

    # Each test case: (query_plan, expected_multiplier)
    test_cases = [
        # --- Original Cases (Expected values rounded to 2 decimal places) ---
        # Simple count
        ("Aggregate [count(1)]", 1.00),
        # Join + Simple Aggregate
        ("Join Inner, Aggregate", 3.33),
        # Window
        ("Window [row_number]", 2.50),
        # Join + Window
        ("Join Inner, Window [row_number]", 8.33),
        # Complex aggregate
        ("Aggregate [sum(x)], groupingexpressions", 1.60),
        # Join + Complex aggregate
        ("Join Inner, Aggregate [sum(x)], groupingexpressions", 5.33),
        # Sort only
        ("Sort [x ASC]", 1.40),
        # Distinct only (Note: 'distinct' keyword counted)
        ("Aggregate [count(1)], distinct", 2.50),
        # UDF only
        ("UDF PythonEval", 1.20),

        # --- New Cases with Repeated Ops (Expected values rounded to 2 decimal places) ---
        # Repeated Joins (2) + Simple Aggregate (1)
        ("Join Inner, Join Inner, Aggregate", 5.00),
        # Repeated Windows (2)
        ("Window [row_number], Window [rank]", 4.50),
        # Repeated Complex Aggregates (2)
        ("Aggregate [sum(x)], groupingexpressions, Aggregate [max(y)], groupingexpressions", 2.24),
        # Repeated Sorts (2)
        ("Sort [x ASC], Sort [y DESC]", 1.82),
        # Repeated Joins (2) + Repeated Sorts (2)
        ("Join Inner, Sort [x ASC], Join Inner, Sort [y DESC]", 9.10),
        # Repeated Distinct (2)
        ("Aggregate [count(1)], distinct, distinct", 3.50),
        # Repeated UDFs (2)
        ("UDF PythonEval, UDF PythonEval", 1.44),
    ]

    # We'll patch get_input_file_sizes to always return [1.0] (1 GB)
    fail = False
    with patch.object(qce, "get_input_file_sizes", return_value=[1.0]):
        for query_plan, expected_multiplier in test_cases:
            mock_df = MagicMock()
            # Patch the query plan string
            mock_df._jdf.queryExecution().analyzed().toString.return_value = query_plan
            # Call the real function
            _, multiplier, _ = qce.estimate_compute_complexity(mock_df)
            # Complexity should be 1.0 * expected_multiplier
            # Use a more lenient tolerance for floating-point comparison
            # assert abs(multiplier - expected_multiplier) < 0.01, f"Multiplier failed for plan: {query_plan}"
            case_failed = not abs(multiplier - expected_multiplier) < 0.01
            if case_failed:
                print(f"Failed for plan: {query_plan}, multiplier: {multiplier}, expected: {expected_multiplier}")
            fail = fail or case_failed

    assert not fail, "Some tests failed"


# Test the internal calculation logic directly
def test_calculate_complexity_from_plan_count():
    """Test _calculate_complexity_from_plan returns 1.0 for a count() on 1GB."""
    # Import the internal function we want to test
    from dbfs_spark_cache.query_complexity_estimation import \
        _calculate_complexity_from_plan

    # Define a simple count query plan string (lowercase)
    count_query_plan = "aggregate [count(1) as count]" # Example simple count plan
    total_size_gb = 1.0

    # Call the internal function directly
    complexity, multiplier = _calculate_complexity_from_plan(count_query_plan, total_size_gb)

    # Assert the result
    assert multiplier == 1.0, f"Expected multiplier 1.0 for count plan, but got {multiplier}" # by complexity estimation design
    assert complexity == 1.0, f"Expected complexity 1.0 for count plan, but got {complexity}"


def test_cacheToDbfs_uses_existing_cache():
    """Test that cacheToDbfs returns existing cache without rewriting."""

    from dbfs_spark_cache import caching

    # Mock DataFrame to be cached
    mock_df_input = MagicMock(name="InputDataFrame")
    mock_df_input.sparkSession._jsc.sc.getExecutorMemoryStatus.return_value.size.return_value = 2 # Mock cluster size

    # Mock DataFrame returned by read_dbfs_cache_if_exist
    mock_df_from_cache = MagicMock(name="CachedDataFrame")

    # Patch the relevant functions within the caching module
    # Patch the relevant functions within the caching module
    # Mock get_query_plan to return a non-problematic plan
    with patch.object(caching, 'get_query_plan', return_value="== Physical Plan ==\nSimpleScan") as mock_get_plan, \
         patch.object(caching, 'read_dbfs_cache_if_exist', return_value=mock_df_from_cache) as mock_read, \
         patch.object(caching, 'write_dbfs_cache') as mock_write, \
         patch.object(caching, 'is_spark_cached', return_value=False) as mock_is_spark_cached, \
         patch.object(caching, 'estimate_compute_complexity') as mock_estimate, \
         patch('time.time', return_value=12345.0): # Mock time to avoid timing issues

        # Call the function under test
        result_df = caching.cacheToDbfs(mock_df_input)

        # Assertions
        mock_get_plan.assert_called_once_with(mock_df_input) # Verify query plan was checked first
        # Update assertion to include new kwargs
        mock_read.assert_called_once_with(mock_df_input, query_plan="== Physical Plan ==\nSimpleScan", input_dir_mod_datetime=ANY) # Verify it checked for cache
        mock_write.assert_not_called() # Verify it did NOT write the cache again
        mock_is_spark_cached.assert_called_once_with(mock_df_input) # Verify Spark cache status was checked on input DF
        mock_estimate.assert_not_called() # Complexity estimation should be skipped
        assert result_df is mock_df_from_cache # Verify it returned the df from cache


def test_get_hash_from_metadata_correct_group():
    """Test that get_hash_from_metadata extracts the correct hash (group 1)."""
    from dbfs_spark_cache import caching
    from dbfs_spark_cache.config import \
        config  # Import the actual config object

    # Store original value and modify directly (temporary workaround for patching issues)
    original_db_name = config.CACHE_DATABASE
    config.CACHE_DATABASE = 'test_cache_db'

    try:
        # Test case 1: Valid metadata with hash
        # The function should now use 'test_cache_db' internally
        metadata_with_hash = "Some text before spark_catalog.test_cache_db.abcdef1234567890abcdef1234567890 some text after"
        expected_hash = "abcdef1234567890abcdef1234567890"
        assert caching.get_hash_from_metadata(metadata_with_hash) == expected_hash, "Test Case 1 Failed"

        # Test case 2: Metadata without the specific pattern
        metadata_without_hash = "Some text without the spark catalog pattern"
        assert caching.get_hash_from_metadata(metadata_without_hash) is None, "Test Case 2 Failed"

        # Test case 3: Metadata with different database name (should not match)
        metadata_diff_db = "spark_catalog.other_db.abcdef1234567890abcdef1234567890"
        assert caching.get_hash_from_metadata(metadata_diff_db) is None, "Test Case 3 Failed"

        # Test case 4: Empty string
        assert caching.get_hash_from_metadata("") is None, "Test Case 4 Failed"
    finally:
        # Restore original value to avoid side effects
        config.CACHE_DATABASE = original_db_name

def test_cacheToDbfs_skips_on_existing_rdd():
    """Test that cacheToDbfs skips caching if 'Scan ExistingRDD' is in the query plan."""

    from dbfs_spark_cache import caching

    # Mock DataFrame to be cached
    mock_df_input = MagicMock(name="InputDataFrame")

    # Mock query plan containing the problematic node
    mock_query_plan = "== Physical Plan ==\nScan ExistingRDD[key#1, value#2]"

    # Patch the relevant functions
    # Patch the relevant functions
    with patch.object(caching, 'get_query_plan', return_value=mock_query_plan) as mock_get_plan, \
         patch.object(caching, 'read_dbfs_cache_if_exist', return_value=None) as mock_read, \
         patch.object(caching, 'write_dbfs_cache') as mock_write, \
         patch.object(caching, 'log') as mock_log: # Patch the logger

        # Call the function under test
        result_df = caching.cacheToDbfs(mock_df_input)

        # Assertions
        mock_get_plan.assert_called_once_with(mock_df_input) # Verify query plan was checked
        # Since the plan contains "Scan ExistingRDD", the function should return early
        mock_read.assert_not_called() # Should NOT check for cache existence
        mock_write.assert_not_called() # Should NOT write the cache
        assert result_df == mock_df_input # Should return original DF
        mock_log.info.assert_any_call("Skipping cache for DataFrame derived from RDD (Scan ExistingRDD found in plan).")


def test_get_input_dir_mod_datetime_handles_schema_change():
    """Test get_input_dir_mod_datetime returns {} on DELTA_SCHEMA_CHANGE error."""
    import datetime  # Import datetime for mocking

    from py4j.protocol import Py4JJavaError  # type: ignore[import-untyped]

    from dbfs_spark_cache import caching

    # Mock DataFrame
    mock_df = MagicMock(name="StaleDataFrame")

    # Mock the Java exception that Py4JJavaError would wrap
    mock_java_exception = MagicMock()
    # Configure the return value of the method attributes, ignoring Mypy error
    mock_java_exception.__str__.return_value = "DELTA_SCHEMA_CHANGE_SINCE_ANALYSIS some details" # type: ignore[assignment]
    mock_java_exception.getMessage.return_value = "DELTA_SCHEMA_CHANGE_SINCE_ANALYSIS some details" # type: ignore[assignment]


    # Configure inputFiles to raise the specific error
    mock_df.inputFiles.side_effect = Py4JJavaError(
        msg="An error occurred while calling o123.inputFiles.", # Example Py4J message
        java_exception=mock_java_exception
    )

    # Mock datetime.now() to return a fixed timestamp
    fixed_time = datetime.datetime(2024, 1, 1, 12, 0, 0)
    expected_result = {"<schema_changed_placeholder>": fixed_time}

    # Patch the logger and datetime.now
    with patch.object(caching, 'log') as mock_log, \
         patch('dbfs_spark_cache.caching.datetime') as mock_datetime:

        mock_datetime.now.return_value = fixed_time
        # Ensure the original datetime class is still available if needed elsewhere
        mock_datetime.datetime = datetime.datetime

        # Call the function under test
        result = caching.get_input_dir_mod_datetime(mock_df)

        # Assertions
        mock_df.inputFiles.assert_called_once() # Verify inputFiles was called
        assert result == expected_result, "Expected dict with placeholder and current time on schema change error"
        mock_log.warning.assert_called_once()
        # Check if the warning message contains the expected text
        args, kwargs = mock_log.warning.call_args
        assert "Could not get input files due to Delta schema change" in args[0]
        assert "Forcing cache invalidation" in args[0] # Check for updated warning text

        # Assertions
        mock_df.inputFiles.assert_called_once() # Verify inputFiles was called
        assert result == expected_result, "Expected dict with placeholder and current time on schema change error"
        mock_log.warning.assert_called_once()
        # Check if the warning message contains the expected text
        args, kwargs = mock_log.warning.call_args
        assert "Could not get input files due to Delta schema change" in args[0]
        assert "Forcing cache invalidation" in args[0] # Check for updated warning text
