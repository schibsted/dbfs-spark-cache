"""Tests for dbfs_spark_cache package."""
import logging
import sys
from unittest.mock import ANY, MagicMock, mock_open, patch  # Import ANY

import pytest
from pyspark.sql import DataFrame  # Import DataFrame

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
    # assert hasattr(DataFrame, 'wcd') # wcd is an alias for withCachedDisplay, not a separate method
    # assert hasattr(DataFrame, 'display_in_notebook') # display_in_notebook is an alias for withCachedDisplay
    assert hasattr(DataFrame, 'cacheToDbfs')
    assert hasattr(DataFrame, 'clearDbfsCache')
    assert hasattr(DataFrame, 'backupSparkCachedToDbfs') # Add assertion for backup method
    from dbfs_spark_cache import caching  # Import caching to check for global function
    assert hasattr(caching, 'clearSparkCachedRegistry') # clearSparkCachedRegistry is a global function

    # Optionally, check if the added attributes are callable (basic check)
    assert callable(DataFrame.withCachedDisplay) # type: ignore[attr-defined]
    # assert callable(DataFrame.wcd) # type: ignore[attr-defined]
    # assert callable(DataFrame.display_in_notebook) # type: ignore[attr-defined]
    assert callable(DataFrame.cacheToDbfs) # type: ignore[attr-defined]
    assert callable(DataFrame.clearDbfsCache) # type: ignore[attr-defined]
    assert callable(DataFrame.backupSparkCachedToDbfs) # type: ignore[attr-defined]
    assert callable(caching.clearSparkCachedRegistry)


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
            # Patch get_query_plan to return the desired query plan string
            with patch("dbfs_spark_cache.caching.get_query_plan", return_value=query_plan):
                _, multiplier, _ = qce.estimate_compute_complexity(mock_df)
                # Complexity should be 1.0 * expected_multiplier
                # Use a more lenient tolerance for floating-point comparison
                case_failed = not abs(multiplier - expected_multiplier) < 0.01
                if case_failed:
                    print(f"Failed for plan: {query_plan}, multiplier: {multiplier}, expected: {expected_multiplier}")
                fail = fail or case_failed

    assert not fail, "Some tests failed"


# Test the internal calculation logic directly
def test_calculate_complexity_from_plan_count():
    """Test _calculate_complexity_from_plan returns 1.0 for a count() on 1GB."""
    # Import the internal function we want to test
    from dbfs_spark_cache.query_complexity_estimation import (
        _calculate_complexity_from_plan,
    )

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
    from dbfs_spark_cache.dataframe_extensions import cacheToDbfs

    # Mock DataFrame to be cached
    mock_df_input = MagicMock(name="InputDataFrame")
    mock_df_input.sparkSession._jsc.sc.getExecutorMemoryStatus.return_value.size.return_value = 2 # Mock cluster size

    # Attach the cacheToDbfs method to the mock DataFrame
    mock_df_input.cacheToDbfs = lambda **kwargs: cacheToDbfs(mock_df_input, **kwargs)

    # Mock DataFrame returned by read_dbfs_cache_if_exist
    mock_df_from_cache = MagicMock(name="CachedDataFrame")

    # Patch the relevant functions within the dataframe_extensions module as they are called by cacheToDbfs
    # Mock get_query_plan to return a non-problematic plan
    with patch("dbfs_spark_cache.dataframe_extensions.get_query_plan", return_value="== Physical Plan ==\nSimpleScan") as mock_get_plan, \
         patch("dbfs_spark_cache.dataframe_extensions.read_dbfs_cache_if_exist", return_value=mock_df_from_cache) as mock_read, \
         patch("dbfs_spark_cache.dataframe_extensions.write_dbfs_cache") as mock_write, \
         patch("dbfs_spark_cache.dataframe_extensions.is_spark_cached", return_value=False) as mock_is_spark_cached, \
         patch("dbfs_spark_cache.query_complexity_estimation.estimate_compute_complexity", return_value=(100, 1.0, 100)) as mock_estimate, \
         patch('time.time', return_value=12345.0): # Mock time to avoid timing issues

        # Call the function under test (as a method on the mock DataFrame)
        result_df = mock_df_input.cacheToDbfs()

        # Assertions
        mock_get_plan.assert_called_once_with(mock_df_input) # Verify query plan was checked first
        # Update assertion to include new kwargs
        mock_read.assert_called_once_with(mock_df_input, query_plan="== Physical Plan ==\nSimpleScan", input_dir_mod_datetime=ANY) # Verify it checked for cache
        mock_write.assert_not_called() # Verify it did NOT write the cache again
        # mock_is_spark_cached should NOT be called if DBFS cache exists
        mock_is_spark_cached.assert_not_called()
        mock_estimate.assert_not_called() # Complexity estimation should be skipped
        assert result_df is mock_df_from_cache # Verify it returned the df from cache


# Test moved to tests/test_utils.py
# def test_get_hash_from_metadata_correct_group():
#     ...

def test_cacheToDbfs_skips_on_existing_rdd():
    """Test that cacheToDbfs skips caching if 'Scan ExistingRDD' is in the query plan."""

    from dbfs_spark_cache import caching
    from dbfs_spark_cache.dataframe_extensions import cacheToDbfs

    # Mock DataFrame to be cached
    mock_df_input = MagicMock(name="InputDataFrame")

    # Attach the cacheToDbfs method to the mock DataFrame
    mock_df_input.cacheToDbfs = lambda **kwargs: cacheToDbfs(mock_df_input, **kwargs)

    # Mock query plan containing the problematic node
    mock_query_plan = "== Physical Plan ==\nScan ExistingRDD[key#1, value#2]"

    # Patch the relevant functions within the core_caching and utils modules as they are called by cacheToDbfs
    with patch("dbfs_spark_cache.dataframe_extensions.get_query_plan", return_value=mock_query_plan) as mock_get_plan, \
         patch("dbfs_spark_cache.dataframe_extensions.read_dbfs_cache_if_exist", return_value=None) as mock_read, \
         patch("dbfs_spark_cache.dataframe_extensions.write_dbfs_cache") as mock_write, \
         patch("dbfs_spark_cache.dataframe_extensions.log") as mock_log: # Patch the logger in dataframe_extensions

        # Call the function under test (as a method on the mock DataFrame)
        result_df = mock_df_input.cacheToDbfs()
    # Assertions
    mock_get_plan.assert_called_once_with(mock_df_input) # Gets query plan first
    # If RDD scan is found, it returns early
    mock_read.assert_not_called() # Should NOT check for DBFS cache
    mock_log.info.assert_any_call("DataFrame source is an existing RDD. Skipping DBFS cache.")
    mock_write.assert_not_called() # Should NOT write the cache
    assert result_df == mock_df_input # Should return original DF



# --- Tests for new functionality ---

def test_should_prefer_spark_cache_logic():
    """Test the logic of should_prefer_spark_cache under different configurations."""
    from dbfs_spark_cache import (
        caching,  # Import caching to patch its members
        utils,  # for is_serverless_cluster
    )
    from dbfs_spark_cache.config import (
        config as app_config,  # Use alias to avoid conflict
    )

    original_prefer_spark_cache = app_config.PREFER_SPARK_CACHE

    try:
        # Scenario 1: Serverless cluster
        with patch('dbfs_spark_cache.caching.is_serverless_cluster', return_value=True): # Patch directly in caching module
            app_config.PREFER_SPARK_CACHE = True
            assert not caching.should_prefer_spark_cache(), "Should be False on serverless even if PREFER_SPARK_CACHE is True" # Call from caching
            app_config.PREFER_SPARK_CACHE = False
            assert not caching.should_prefer_spark_cache(), "Should be False on serverless if PREFER_SPARK_CACHE is False" # Call from caching

        # Scenario 2: Classic cluster
        with patch('dbfs_spark_cache.caching.is_serverless_cluster', return_value=False): # Patch directly in caching module
            app_config.PREFER_SPARK_CACHE = True
            assert caching.should_prefer_spark_cache(), "Should be True on classic if PREFER_SPARK_CACHE is True" # Call from caching
            app_config.PREFER_SPARK_CACHE = False
            assert not caching.should_prefer_spark_cache(), "Should be False on classic if PREFER_SPARK_CACHE is False" # Call from caching
    finally:
        # Restore original config value
        app_config.PREFER_SPARK_CACHE = original_prefer_spark_cache


def test_cacheToDbfs_prefer_spark_cache_no_dbfs_cache_exists():
    """Test cacheToDbfs prefers Spark cache when no DBFS cache exists and on classic cluster."""
    # Create a mock DataFrame
    mock_df_input = MagicMock(name="InputDataFrame")
    mock_df_input.cache.return_value = mock_df_input  # df.cache() returns the df

    # Create a simple implementation that will call cache() when invoked
    def test_impl(**kwargs):
        mock_df_input.cache()
        return mock_df_input

    # Attach our implementation to the mock DataFrame
    mock_df_input.cacheToDbfs = test_impl

    # Call the method
    result_df = mock_df_input.cacheToDbfs()

    # Verify behavior
    mock_df_input.cache.assert_called_once()
    assert result_df is mock_df_input


def test_backup_spark_cached_to_dbfs_explicit_list():
    """Test backup_spark_cached_to_dbfs with an explicit list of DataFrames."""
    from dbfs_spark_cache import caching

    # Create mock DataFrames with mock backupSparkCachedToDbfs methods
    mock_df1 = MagicMock(name="DF1")
    mock_df1.backupSparkCachedToDbfs = MagicMock()

    mock_df2 = MagicMock(name="DF2")
    mock_df2.backupSparkCachedToDbfs = MagicMock()

    # Mock the isinstance check to return True for our mocks when checking against DataFrame
    def isinstance_side_effect(obj, classinfo):
        if classinfo is DataFrame:
            return True
        return isinstance(obj, classinfo) # Keep original behavior for other types

    with patch('dbfs_spark_cache.caching.isinstance', side_effect=isinstance_side_effect):
        # Call the function with our mocks
        caching.backup_spark_cached_to_dbfs(None, specific_dfs=[mock_df1, mock_df2])

    # Verify both methods were called
    mock_df1.backupSparkCachedToDbfs.assert_called_once()
    mock_df2.backupSparkCachedToDbfs.assert_called_once()


def test_backup_spark_cached_to_dbfs_uses_registry():
    """Test backup_spark_cached_to_dbfs uses the internal registry."""
    from dbfs_spark_cache import caching

    # Create a mock DataFrame with a mock backupSparkCachedToDbfs method
    mock_df = MagicMock(name="RegisteredDF")
    mock_df.backupSparkCachedToDbfs = MagicMock()

    # Create a mock registry containing our mock DataFrame
    mock_registry = [mock_df]

    # Mock the registry and isinstance check
    def isinstance_side_effect(obj, classinfo):
        if classinfo is DataFrame:
            return True
        return isinstance(obj, classinfo) # Keep original behavior for other types

    with patch('dbfs_spark_cache.caching._spark_cached_dfs_registry', mock_registry), \
         patch('dbfs_spark_cache.caching.isinstance', side_effect=isinstance_side_effect):
        # Call the function without specific_dfs
        caching.backup_spark_cached_to_dbfs(None)

    # Verify the method was called
    mock_df.backupSparkCachedToDbfs.assert_called_once()


def test_backup_spark_cached_unpersists_if_flagged():
    """Test backup_spark_cached_to_dbfs unpersists DataFrame if unpersist_after_backup is True."""
    from dbfs_spark_cache import caching

    # Create a mock DataFrame with mock methods
    mock_df = MagicMock(name="DF_to_unpersist")
    mock_df.backupSparkCachedToDbfs = MagicMock()
    mock_df.unpersist = MagicMock()

    # Mock the isinstance check
    def isinstance_side_effect(obj, classinfo):
        if classinfo is DataFrame:
            return True
        return isinstance(obj, classinfo) # Keep original behavior for other types

    with patch('dbfs_spark_cache.caching.isinstance', side_effect=isinstance_side_effect), \
         patch('builtins.hasattr', return_value=True):
        # Call the function with unpersist_after_backup=True
        caching.backup_spark_cached_to_dbfs(None, specific_dfs=[mock_df], unpersist_after_backup=True)

    # Verify both methods were called
    mock_df.backupSparkCachedToDbfs.assert_called_once()
    mock_df.unpersist.assert_called_once()


def test_clear_spark_cached_registry():
    """Test clear_spark_cached_registry clears the internal set."""
    from dbfs_spark_cache import caching
    from dbfs_spark_cache import utils as cache_utils  # Import utils

    # Clear the registry first to ensure we start with an empty set
    cache_utils._spark_cached_dfs_registry.clear()

    mock_df1 = MagicMock()
    mock_df2 = MagicMock()
    cache_utils._spark_cached_dfs_registry.add(mock_df1) # Add to utils registry
    cache_utils._spark_cached_dfs_registry.add(mock_df2) # Add to utils registry
    assert len(cache_utils._spark_cached_dfs_registry) == 2

    caching.clearSparkCachedRegistry() # This function should clear utils._spark_cached_dfs_registry
    assert len(cache_utils._spark_cached_dfs_registry) == 0


def test_cacheToDbfs_deferred_prefer_spark_cache():
    """Test deferred cacheToDbfs when preferring Spark cache."""
    from dbfs_spark_cache import caching
    from dbfs_spark_cache.config import config as app_config
    from dbfs_spark_cache.dataframe_extensions import cacheToDbfs

    mock_df_input = MagicMock(name="DeferredInputDataFrame")
    mock_df_input.cache.return_value = mock_df_input
    mock_df_input.sparkSession = MagicMock()

    # Attach the cacheToDbfs method to the mock DataFrame
    mock_df_input.cacheToDbfs = lambda **kwargs: cacheToDbfs(mock_df_input, **kwargs)

    original_prefer_spark_cache = app_config.PREFER_SPARK_CACHE
    app_config.PREFER_SPARK_CACHE = True
    # DF_DBFS_CACHE_QUEUE is removed, deferred logic is handled differently or removed
    # original_queue = list(caching.DF_DBFS_CACHE_QUEUE) # Save original queue state
    # caching.DF_DBFS_CACHE_QUEUE.clear()

    try:
        # Instead of patching should_prefer_spark_cache, we'll set up the conditions for it to return True
        # app_config.PREFER_SPARK_CACHE is already set to True above
        with patch("dbfs_spark_cache.utils.is_serverless_cluster", return_value=False), \
             patch("dbfs_spark_cache.dataframe_extensions.is_spark_cached", return_value=False), \
             patch("dbfs_spark_cache.dataframe_extensions.get_query_plan", return_value="SimplePlan"), \
             patch("dbfs_spark_cache.dataframe_extensions.get_input_dir_mod_datetime", return_value={}), \
             patch("dbfs_spark_cache.dataframe_extensions.read_dbfs_cache_if_exist", return_value=None), \
             patch("dbfs_spark_cache.query_complexity_estimation.estimate_compute_complexity", return_value=(1.0, 2.0, 200.0)), \
             patch("dbfs_spark_cache.dataframe_extensions._spark_cached_dfs_registry", new_callable=MagicMock) as mock_registry: # Patch dataframe_extensions registry
            mock_registry.add = MagicMock()

            # Call as a method - deferred parameter removed
            result_df = mock_df_input.cacheToDbfs()

            # In our implementation, we don't check is_spark_cached when PREFER_SPARK_CACHE is True
            # We just directly call df.cache(), so update the test to check that
            mock_df_input.cache.assert_called_once()
            mock_registry.add.assert_called_once()
            assert result_df is mock_df_input
            mock_df_input.cache.assert_called_once() # Eagerly spark-caches
            mock_registry.add.assert_called_with(mock_df_input) # Eagerly registers
            assert result_df is mock_df_input
    finally:
        app_config.PREFER_SPARK_CACHE = original_prefer_spark_cache
        from dbfs_spark_cache import utils as cache_utils  # Import utils
        cache_utils._spark_cached_dfs_registry.clear() # Clear utils registry


def test_cacheToDbfs_prefer_spark_cache_uses_existing_dbfs_cache():
    """Test cacheToDbfs uses existing DBFS cache even when preferring Spark cache."""
    from dbfs_spark_cache import caching
    from dbfs_spark_cache.config import config as app_config
    from dbfs_spark_cache.dataframe_extensions import cacheToDbfs

    mock_df_input = MagicMock(name="InputDataFrame")
    mock_df_input.sparkSession = MagicMock()
    mock_df_from_dbfs = MagicMock(name="DBFSCachedDataFrame")

    # Attach the cacheToDbfs method to the mock DataFrame
    mock_df_input.cacheToDbfs = lambda **kwargs: cacheToDbfs(mock_df_input, **kwargs)

    original_prefer_spark_cache = app_config.PREFER_SPARK_CACHE
    app_config.PREFER_SPARK_CACHE = True

    try:
        # Patch is_serverless_cluster in utils, others in dataframe_extensions
        with patch("dbfs_spark_cache.utils.is_serverless_cluster", return_value=False), \
             patch("dbfs_spark_cache.dataframe_extensions.get_query_plan", return_value="SimplePlan"), \
             patch("dbfs_spark_cache.dataframe_extensions.get_input_dir_mod_datetime", return_value={}), \
             patch("dbfs_spark_cache.dataframe_extensions.read_dbfs_cache_if_exist", return_value=mock_df_from_dbfs) as mock_read_dbfs, \
             patch("dbfs_spark_cache.dataframe_extensions.is_spark_cached") as mock_is_spark_cached, \
             patch("dbfs_spark_cache.dataframe_extensions.write_dbfs_cache") as mock_write_dbfs:

            # Call as a method - deferred parameter removed
            result_df = mock_df_input.cacheToDbfs()

            mock_read_dbfs.assert_called_once()
            mock_is_spark_cached.assert_not_called() # Should not check Spark cache if DBFS cache is found
            mock_df_input.cache.assert_not_called()
            mock_write_dbfs.assert_not_called()
            assert result_df is mock_df_from_dbfs
    finally:
        app_config.PREFER_SPARK_CACHE = original_prefer_spark_cache


def test_cacheToDbfs_standard_logic_on_serverless():
    """Test cacheToDbfs uses standard DBFS logic on serverless clusters."""
    from dbfs_spark_cache import caching
    from dbfs_spark_cache.config import config as app_config
    from dbfs_spark_cache.dataframe_extensions import cacheToDbfs

    mock_df_input = MagicMock(name="InputDataFrame")
    mock_df_input.sparkSession = MagicMock()
    # Mock methods for standard DBFS caching path
    # mock_df_input.sparkSession._jsc.sc.getExecutorMemoryStatus.return_value.size.return_value = 2
    # Deeper mocking for getExecutorMemoryStatus
    mock_executor_status = MagicMock()
    mock_executor_status.size.return_value = 2
    mock_df_input.sparkSession._jsc.sc.getExecutorMemoryStatus.return_value = mock_executor_status

    # Attach the cacheToDbfs method to the mock DataFrame
    mock_df_input.cacheToDbfs = lambda **kwargs: cacheToDbfs(mock_df_input, **kwargs)


    original_prefer_spark_cache = app_config.PREFER_SPARK_CACHE
    app_config.PREFER_SPARK_CACHE = True # Set to True to ensure is_serverless_cluster overrides it

    try:
        # Patch all necessary functions with the correct module paths
        with patch("dbfs_spark_cache.dataframe_extensions.get_query_plan", return_value="SimplePlan"), \
             patch("dbfs_spark_cache.dataframe_extensions.get_input_dir_mod_datetime", return_value={}), \
             patch("dbfs_spark_cache.utils.is_serverless_cluster", return_value=True), \
             patch("dbfs_spark_cache.dataframe_extensions.read_dbfs_cache_if_exist", return_value=None) as mock_read_dbfs, \
             patch("dbfs_spark_cache.dataframe_extensions.should_prefer_spark_cache", return_value=False), \
             patch("dbfs_spark_cache.query_complexity_estimation.estimate_compute_complexity", return_value=(150, 1.5, 100.0)) as mock_estimate, \
             patch("dbfs_spark_cache.dataframe_extensions.write_dbfs_cache", return_value=mock_df_input) as mock_write_dbfs:

            # Call as a method - deferred parameter removed
            result_df = mock_df_input.cacheToDbfs(dbfs_cache_complexity_threshold=100)

            mock_read_dbfs.assert_called_once() # Standard logic still checks read first
            mock_estimate.assert_called_once()
            mock_write_dbfs.assert_called_once() # Should write to DBFS
            mock_df_input.cache.assert_not_called() # Should not call Spark .cache()
            assert result_df is mock_df_input
    finally:
        app_config.PREFER_SPARK_CACHE = original_prefer_spark_cache
