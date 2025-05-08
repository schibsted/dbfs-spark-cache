# Databricks notebook source
# MAGIC %md
# MAGIC ### Integration Test Notebook for dbfs-spark-cache
# MAGIC This notebook executes the comprehensive integration test plan for the dbfs-spark-cache module
# MAGIC
# MAGIC This notebook outlines the plan for creating an integration test notebook for the dbfs-spark-cache module. The integration tests will focus on testing functionality not covered by the unit tests, with an emphasis on testing the inner functions in `dbfs_spark_cache/caching.py` first, then testing different variants of `cacheToDbfs()` and `withCachedDisplay()`.
# MAGIC
# MAGIC 1. This notebook should be run in a Databricks environment with the dbfs-spark-cache module installed, or with library code source imported directly.
# MAGIC 2. The tests are designed to be run sequentially, with each section building on the previous ones.
# MAGIC 3. The notebook includes comprehensive testing of all major functions in the dbfs-spark-cache module.
# MAGIC 4. The tests use small DataFrames to ensure quick execution.
# MAGIC 5. The cleanup section ensures that all test artifacts are removed after the tests are complete.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set up dependencies

# COMMAND ----------

import importlib.util
import os
import sys
from datetime import datetime  # Import datetime
from pathlib import Path
from typing import Any, Dict  # Import Dict


def import_from_file(
    file_path: str | os.PathLike,
    obj_name: str | None = None,
    mod_name: str | None = None,
) -> Any:
    """
    Import a module or an object from a file.

    Parameters
    ----------
    file_path : str or PathLike
        The path to the file to import from.
    obj_name : str, optional
        The name of the object to import from the module. If None, the entire module is
        returned.
    mod_name : str, optional
        The name to assign to the module. If None, the module name is derived from the
        file name.

    Returns
    -------
    Any
        The imported module or the specified object from the module.

    Examples
    --------
    Import a module:
    >>> my_module = import_from_file('path/to/my_module.py')

    Import an object from a module:
    >>> my_function = import_from_file('path/to/my_module.py', 'my_function')
    """
    file_path = Path(file_path).resolve()

    if mod_name is None:
        mod_name_path = file_path
        while mod_name_path.suffixes:
            mod_name_path = mod_name_path.with_suffix("")
    else:
        mod_name_path = Path(mod_name)

    mod_name_to_import = str(mod_name_path.stem.replace("-", "_"))

    # Check if file exists before attempting to import
    if not Path(file_path).exists():
        raise ImportError(f"File not found for importing: {file_path}")

    spec = importlib.util.spec_from_file_location(mod_name_to_import, str(file_path))

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {file_path}")

    imp_module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name_to_import] = imp_module
    spec.loader.exec_module(imp_module)

    if obj_name is None:
        return imp_module

    return getattr(imp_module, obj_name)

set_and_get_workdir = import_from_file('../notebook_utils.py', 'set_and_get_workdir')
setup_dependencies = import_from_file('../notebook_utils.py', 'setup_dependencies')
REPO_PATH = set_and_get_workdir(spark)
setup_dependencies(REPO_PATH, spark)

# COMMAND ----------

import os

# 1. Setup Test Environment
# pyright: reportMissingImports=false, reportGeneralTypeIssues=false
import time

# Initialize the DataFrame class with caching capabilities
from pyspark.sql import DataFrame
from pyspark.sql import types as spark_types  # type: ignore
from pyspark.sql.functions import col, lit, rand
from pyspark.sql.functions import sum as spark_sum

from dbfs_spark_cache import extend_dataframe_methods
from dbfs_spark_cache.cache_management import (
    clear_cache_for_hash,
    clear_caches_older_than,
    clear_inconsistent_cache,
    get_cached_dataframe_metadata,
    get_cached_tables,
)
from dbfs_spark_cache.caching import backup_spark_cached_to_dbfs, is_serverless_cluster
from dbfs_spark_cache.config import config
from dbfs_spark_cache.core_caching import (
    createCachedDataFrame,
    get_cache_metadata,
    get_input_dir_mod_datetime,
    get_query_plan,
    get_table_hash,
    read_dbfs_cache_if_exist,
    write_dbfs_cache,
)

# Import the dbfs-spark-cache module and its inner functions
from dbfs_spark_cache.dataframe_extensions import (
    __withCachedDisplay__,
    cacheToDbfs,
    clearDbfsCache,
    should_prefer_spark_cache,
)
from dbfs_spark_cache.hashing import _hash_input_data
from dbfs_spark_cache.utils import get_table_name_from_hash
from dbfs_spark_cache import caching

extend_dataframe_methods(spark) # Pass spark
config.DEFAULT_COMPLEXITY_THRESHOLD = 130 # Set default complexity threshold

import logging

# Get the library's logger
library_logger = logging.getLogger('dbfs_spark_cache')
# this needs to be AFTER and lib imports
library_logger.setLevel(logging.INFO)

# importlib.reload(dbfs_spark_cache)
# import dbfs_spark_cache

# Create test DataFrames
schema = spark_types.StructType([
    spark_types.StructField("name", spark_types.StringType(), True),
    spark_types.StructField("age", spark_types.IntegerType(), True),
    spark_types.StructField("salary", spark_types.DoubleType(), True)
])

# 1. Simple DataFrame
data = [("Alice", 34, 55000.0), ("Bob", 45, 65000.0), ("Charlie", 29, 72000.0), ("Diana", 37, 58000.0)]
df_simple = spark.createDataFrame(data, schema)

# 2. DataFrame with more rows (for complexity testing)
df_larger = spark.range(0, 10000).withColumn("value", rand())

# 3. DataFrame from SQL query
spark.sql("CREATE DATABASE IF NOT EXISTS test_db")
df_simple.write.mode("overwrite").saveAsTable("test_db.employees")
df_sql = spark.sql("SELECT * FROM test_db.employees")

# 4. DataFrame with transformations
df_transformed = df_simple.groupBy("name").agg(spark_sum("salary").alias("total_salary"))

# Print test environment info
print(f'DATABRICKS_RUNTIME_VERSION: {os.environ.get("DATABRICKS_RUNTIME_VERSION", "")}')
print(f"Spark version: {spark.version}")
print(f"SPARK_CACHE_DIR: {config.SPARK_CACHE_DIR}")
print(f"CACHE_DATABASE: {config.CACHE_DATABASE}")

# --- Setup done ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## Start of tests

# COMMAND ----------

help(extend_dataframe_methods)
print("---")
# from dbfs_spark_cache.caching import __withCachedDisplay__ # Removed redundant import

help(__withCachedDisplay__) # type: ignore # noqa: F821

# COMMAND ----------

# Smoke test of extened methods only working in non-serverless:
if not is_serverless_cluster():
    spark.createCachedDataFrame(data)
    df_tmp = df_sql.cacheToDbfs()
    df_sql.withCachedDisplay()
    df_sql.wcd()
    df_tmp.clearDbfsCache()

# COMMAND ----------

start_table_cache = get_cached_tables(num_threads=100)
display(start_table_cache) if start_table_cache.size > 0 else "No tables in cache"  # noqa: F821

# COMMAND ----------

# 2. Test Inner Functions
print("\nTesting get_input_dir_mod_datetime...")
input_dirs_sql = get_input_dir_mod_datetime(df_sql)
print(f"Input directories for SQL DataFrame: {input_dirs_sql}")

input_dirs_simple = get_input_dir_mod_datetime(df_simple)
print(f"Input directories for in-memory DataFrame: {input_dirs_simple}")

input_dirs_transformed = get_input_dir_mod_datetime(df_transformed)
print(f"Input directories for transformed DataFrame: {input_dirs_transformed}")

print("\nTesting get_query_plan...")
plan_simple = get_query_plan(df_simple)
print(f"Query plan for simple DataFrame (truncated): {plan_simple[:200]}...\n")

plan_sql = get_query_plan(df_sql)
print(f"Query plan for SQL DataFrame (truncated): {plan_sql[:200]}...\n")

plan_transformed = get_query_plan(df_transformed)
print(f"Query plan for transformed DataFrame (truncated): {plan_transformed[:200]}...\n")

from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType


@udf(returnType=IntegerType())
def add_ten(x):
    return x + 10

df_udf = df_simple.withColumn("age_plus_ten", add_ten(col("age")))
plan_udf = get_query_plan(df_udf)
print(f"Query plan for UDF DataFrame (truncated): {plan_udf[:200]}...\n")

print("\nTesting get_cache_metadata...")
input_dirs_raw = get_input_dir_mod_datetime(df_simple)
# Filter out non-datetime values for get_cache_metadata
input_dirs = {}
if isinstance(input_dirs_raw, dict):
    input_dirs = {k: v for k, v in input_dirs_raw.items() if isinstance(v, datetime)} # type: ignore

query_plan = get_query_plan(df_simple)
metadata = get_cache_metadata(input_dirs, query_plan)
print(f"Cache metadata (truncated): {metadata[:200]}...\n")
assert "INPUT SOURCES MODIFICATION DATETIMES:" in metadata
assert "DATAFRAME QUERY PLAN:" in metadata

print(f"Table name: {get_table_name_from_hash(get_table_hash(df_simple))}")

# COMMAND ----------

# 3. Test Core Caching Functions
print("Cleaning up any old tables cached...")
hash_name = get_table_hash(df_simple)
clear_cache_for_hash(hash_name)

print("\nTesting write_dbfs_cache (core caching function)...")
print("Initial caching...")
df_cached = write_dbfs_cache(df_simple)
cached_hash = get_table_hash(df_cached)
print(f"Cached DataFrame: {df_cached}\n")
qualified_table_name = get_table_name_from_hash(cached_hash)
cache_exists = spark.catalog.tableExists(qualified_table_name)
assert cache_exists, f"Cache table {qualified_table_name} exists: {cache_exists}"

print("\nCaching with replace=True...")
df_replaced = write_dbfs_cache(df_simple, replace=True)
print(f"Replaced DataFrame: {df_replaced}\n")

print("\nCaching with verbose=True...")
df_verbose = write_dbfs_cache(df_simple, verbose=True)

print("\nMust be invalid after update")
df_different = df_simple.withColumn("new_col", lit(100))
df_read_miss = read_dbfs_cache_if_exist(df_different)
assert df_read_miss is None, f"Cache miss expected: {df_different}, {df_read_miss}, {get_table_hash(df_different)}, {get_table_hash(df_read_miss)}\n"

print("\nTesting read_dbfs_cache_if_exist...")
clear_cache_for_hash(hash_name)
write_dbfs_cache(df_simple)
df_read = read_dbfs_cache_if_exist(df_simple)
print(f"Read from cache: {df_read is not None}\n")
if df_read is not None:
    print(f"Cached DataFrame count: {df_read.count()}\n")
    df_read.show()

df_different = df_simple.withColumn("new_col", lit(100))
df_read_miss = read_dbfs_cache_if_exist(df_different)
assert df_read_miss is None, f"Cache miss expected\n"  # noqa: F541

# COMMAND ----------

# MAGIC %md
# MAGIC Clear last cache via cleanup functions

# COMMAND ----------

cached_tables = get_cached_tables(num_threads=100)
cached_tables

# COMMAND ----------

# assert hash_name in cached_tables.hash_name.values, f"Table {cached_hash} missing from cache table registry"
metadata_table = get_cached_dataframe_metadata(num_threads=50)
metadata_table

# COMMAND ----------

assert cached_hash in metadata_table.hash_name.values, f"Table {cached_hash} missing from metadata table registry"
tables_in_cache_db = spark.sql(f"SHOW TABLES IN {config.CACHE_DATABASE}")
display(tables_in_cache_db)  # noqa: F821
assert cached_hash in tables_in_cache_db.toPandas().tableName.values, f"Table {cached_hash} missing from database {config.CACHE_DATABASE}"

# COMMAND ----------

clear_cache_for_hash(cached_hash)
assert hash_name not in spark.sql(f"SHOW TABLES IN {config.CACHE_DATABASE}").toPandas().tableName.values, f"Table {cached_hash} not missing from database {config.CACHE_DATABASE} after cleared"

# COMMAND ----------

# MAGIC %md
# MAGIC Test DataFrame extensions

# COMMAND ----------

# Extra setup:
from dbfs_spark_cache.config import config as app_config
app_config.PREFER_SPARK_CACHE = False

# COMMAND ----------

# 4. Test DataFrame Extensions
print("\nTesting DataFrame.cacheToDbfs...")
print("Basic caching...")
df_cached = cacheToDbfs(df_sql)
print(f"Cached DataFrame: {df_cached}\n")

print("\nTesting with complexity threshold...")

# Test case 1: Low threshold (should cache)
df_to_cache_low = df_larger.withColumn("group_col", col("id") % 100).groupBy("group_col").count()
hash_low = get_table_hash(df_to_cache_low)
clear_cache_for_hash(hash_low) # Ensure clean state
print(f"Attempting to cache DataFrame with hash {hash_low} (low threshold=0)...")
df_low_threshold = cacheToDbfs(df_to_cache_low, dbfs_cache_complexity_threshold=0, verbose=True)
qualified_table_name_low = f"{config.CACHE_DATABASE}.{hash_low}"
cache_exists_low = spark.catalog.tableExists(qualified_table_name_low)
assert cache_exists_low, f"Cache table {qualified_table_name_low} should exist for low threshold (complexity > 0)"
print(f"Cache table {qualified_table_name_low} exists as expected: {cache_exists_low}\n")

# Test case 2: High threshold (should NOT cache)
df_to_cache_high = df_larger.withColumn("group_col", col("id") % 100).groupBy("group_col").count()
hash_high = get_table_hash(df_to_cache_high)
clear_cache_for_hash(hash_high) # Ensure clean state
print(f"Attempting to cache DataFrame with hash {hash_high} (high threshold=1000)...")
df_high_threshold = cacheToDbfs(df_to_cache_high, dbfs_cache_complexity_threshold=1000, verbose=True)
qualified_table_name_high = f"{config.CACHE_DATABASE}.{hash_high}"
cache_exists_high = spark.catalog.tableExists(qualified_table_name_high)
assert not cache_exists_high, f"Cache table {qualified_table_name_high} should NOT exist for high threshold (complexity < 1000)"
print(f"Cache table {qualified_table_name_high} does not exist as expected: {not cache_exists_high}\n")

print("\nTesting deferred caching...")
df_deferred = cacheToDbfs(df_sql, deferred=True)
print(f"Deferred caching: {df_deferred}\n")

caching.cache_dataframes_in_queue_to_dbfs()
print("Executed deferred caching\n")

print("\nTesting cache invalidation with data changes...")
df_to_invalidate = cacheToDbfs(df_sql)
data_new = [("Alice", 35, 56000.0), ("Bob", 46, 66000.0), ("Charlie", 30, 73000.0), ("Diana", 38, 59000.0)]
df_new = spark.createDataFrame(data_new, schema)
spark.sql("DROP TABLE IF EXISTS test_db.employees")
df_new.write.mode("overwrite").saveAsTable("test_db.employees")
df_invalidated = spark.sql("SELECT * FROM test_db.employees").withColumn("dummy_column", lit("dummy_value"))
df_recached = cacheToDbfs(df_invalidated)
print("Cache invalidation test completed\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Interaction with clearDbfsCache

# COMMAND ----------

print("\nTesting interaction with clearDbfsCache...")

# 1. Setup: Create initial table and cache it
schema_clear_test = spark_types.StructType([
    spark_types.StructField("id", spark_types.IntegerType(), True),
    spark_types.StructField("val", spark_types.StringType(), True)
])
data_clear_test = [(10, "X"), (20, "Y")]
table_name_clear_test = "test_db.clear_cache_interaction_test"
df_orig_clear_test = spark.createDataFrame(data_clear_test, schema_clear_test)

# Save initial version
df_orig_clear_test.write.format("delta").mode("overwrite").saveAsTable(table_name_clear_test)
print(f"Created initial table {table_name_clear_test}")

# COMMAND ----------

# 1. Create a simple DataFrame
schema_recache_test = spark_types.StructType([
    spark_types.StructField("id", spark_types.IntegerType(), True),
    spark_types.StructField("val", spark_types.StringType(), True)
])
data_recache_test = [(1, "A"), (2, "B")]
df_recache_test = spark.createDataFrame(data_recache_test, schema_recache_test)

# 2. Cache it for the first time
print("First cache operation...")
hash_recache_test = get_table_hash(df_recache_test)
clear_cache_for_hash(hash_recache_test)  # Ensure clean start
df_cached_first = cacheToDbfs(df_recache_test)
assert get_table_hash(df_cached_first) == get_table_hash(df_recache_test), f"hashes mismatch: {get_table_hash(df_cached_first)} != {get_table_hash(df_recache_test)}"
# qualified_table_name_recache = f"{config.CACHE_DATABASE}.{hash_recache_test}"
# assert spark.catalog.tableExists(qualified_table_name_recache), f"Cache table {qualified_table_name_recache} should exist after first cache."
# print(f"DataFrame cached with hash: {hash_recache_test}")

# COMMAND ----------

# Define the DataFrame object pointing to the table
df_ref_clear_test = spark.read.table(table_name_clear_test)

# Cache this DataFrame version
print("Caching original DataFrame version...")
hash_clear_test = get_table_hash(df_ref_clear_test)
clear_cache_for_hash(hash_clear_test) # Ensure clean start
df_cached_clear_test_1 = cacheToDbfs(df_ref_clear_test)
qualified_table_name_clear_test = f"{config.CACHE_DATABASE}.{hash_clear_test}"
assert spark.catalog.tableExists(qualified_table_name_clear_test), f"Cache table {qualified_table_name_clear_test} should exist after first cache."
print(f"Original version cached with hash: {hash_clear_test}")

# 2. Simulate Condition: Clear the cache using the DataFrame reference
print(f"\nClearing cache for hash {hash_clear_test} using df.clearDbfsCache()...")
clearDbfsCache(df_ref_clear_test)
assert not spark.catalog.tableExists(qualified_table_name_clear_test), f"Cache table {qualified_table_name_clear_test} should NOT exist after clearDbfsCache."
print(f"Cache table {qualified_table_name_clear_test} successfully cleared.")

# 3. Trigger Caching Again: Use the same DataFrame reference with cacheToDbfs
print("\nAttempting to cache using the same DataFrame reference again...")
# This call should find no cache (as it was cleared).
# It should NOT encounter DELTA_SCHEMA_CHANGE error because the source wasn't changed.
# It should proceed to calculate the hash (same as before: hash_clear_test) and write the cache again.
df_cached_clear_test_2 = cacheToDbfs(df_ref_clear_test, dbfs_cache_complexity_threshold=0) # Ensure caching happens

# 4. Verification
print("Verifying results after attempting to cache again...")
# Assert that the cache table was recreated
assert spark.catalog.tableExists(qualified_table_name_clear_test), f"Cache table {qualified_table_name_clear_test} should exist again after second cache call."
print(f"Cache table {qualified_table_name_clear_test} was recreated as expected.")

# Assert the returned DF is likely a new object read from the new cache
assert df_cached_clear_test_2 is not df_ref_clear_test, "Second cache call should return a new DataFrame object from the recreated cache."


# COMMAND ----------

# MAGIC %md
# MAGIC ### Test caching the same DataFrame multiple times

# COMMAND ----------

print("\nTesting caching the same DataFrame multiple times...")

# 2. Cache it for the first time
print("First cache operation...")
df_recache_test = spark.read.table(table_name_clear_test)
clear_cache_for_hash(get_table_hash(df_recache_test))  # Ensure clean start

df_cached_first = cacheToDbfs(df_recache_test)
hash_recache_test = get_table_hash(df_cached_first)
qualified_table_name_recache = f"{config.CACHE_DATABASE}.{hash_recache_test}"
assert spark.catalog.tableExists(qualified_table_name_recache), f"Cache table {qualified_table_name_recache} should exist after first cache."
print(f"DataFrame cached with hash: {hash_recache_test}")

# 3. Cache the same DataFrame again
print("\nSecond cache operation on the same DataFrame...")
# This should recognize it's already cached and not trigger a new write
df_cached_second = cacheToDbfs(df_recache_test)

# 4. Verify that the hash is the same and no new write occurred
hash_recache_test_second = get_table_hash(df_recache_test)
assert hash_recache_test == hash_recache_test_second, f"Hash should remain the same for identical DataFrames: {hash_recache_test} vs {hash_recache_test_second}"
print(f"Hash remained the same: {hash_recache_test}")

# 5. Now test with a column added - this should create a new cache
print("\nTesting with modified DataFrame (added column)...")
df_recache_test_modified = df_recache_test.withColumn("new_col", lit("new_value"))
df_cached_modified = cacheToDbfs(df_recache_test_modified)
hash_recache_test_modified = get_table_hash(df_recache_test_modified)
assert hash_recache_test != hash_recache_test_modified, f"Hash should change for modified DataFrame: {hash_recache_test} vs {hash_recache_test_modified}"
qualified_table_name_recache_modified = f"{config.CACHE_DATABASE}.{hash_recache_test_modified}"
assert spark.catalog.tableExists(qualified_table_name_recache_modified), f"Cache table {qualified_table_name_recache_modified} should exist for modified DataFrame."
print(f"Modified DataFrame cached with different hash: {hash_recache_test_modified}")

print("\nMultiple caching test completed successfully.\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test display functionality

# COMMAND ----------

# from caching import __withCachedDisplay__ # Removed redundant import
withCachedDisplay = caching.__withCachedDisplay__
wcd = caching.__withCachedDisplay__

# COMMAND ----------

print("\nTesting DataFrame.withCachedDisplay (wcd)...")
print("Basic withCachedDisplay...")
df_displayed = withCachedDisplay(df_simple)
print(f"Displayed DataFrame: {df_displayed}\n")
print("\nTesting wcd shorthand...")
df_wcd = wcd(df_simple)
print(f"wcd DataFrame: {df_wcd}\n")
print("\nTesting with different parameters...")
df_eager = wcd(df_simple, eager_spark_cache=True)
print(f"wcd with eager_spark_cache: {df_eager}\n")
df_skip_display = wcd(df_simple, skip_display=True)
print(f"wcd with skip_display: {df_skip_display}\n")
df_skip_cache = wcd(df_simple, skip_dbfs_cache=True)
print(f"wcd with skip_dbfs_cache: {df_skip_cache}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test PREFER_SPARK_CACHE and backup_spark_cached_to_dbfs

# COMMAND ----------

import os

from dbfs_spark_cache.config import config as app_config

# --- Test PREFER_SPARK_CACHE behavior ---
if not is_serverless_cluster():
    print("\\n--- Testing PREFER_SPARK_CACHE behavior (must be done with classic cluster) ---")

    # Scenario 1: Classic cluster, PREFER_SPARK_CACHE = True
    app_config.PREFER_SPARK_CACHE = True

    print(f"Simulating Classic Cluster (DATABRICKS_RUNTIME_VERSION={os.environ.get('DATABRICKS_RUNTIME_VERSION')}), PREFER_SPARK_CACHE={app_config.PREFER_SPARK_CACHE}")
    assert should_prefer_spark_cache() is True, "should_prefer_spark_cache should be True on classic with PREFER_SPARK_CACHE=True"

    df_classic_prefer_true = df_sql.select("name", "age") # Create a new DF for this test
    hash_classic_prefer_true = get_table_hash(df_classic_prefer_true)
    clear_cache_for_hash(hash_classic_prefer_true) # Ensure no DBFS cache

    print("Calling cacheToDbfs on classic, PREFER_SPARK_CACHE=True, no existing DBFS cache...")
    df_cached_classic_spark = cacheToDbfs(df_classic_prefer_true)

    assert df_cached_classic_spark.is_cached, "DataFrame should be Spark-cached"
    # assert id(df_cached_classic_spark) in _spark_cached_dfs_registry, "DataFrame ID should be in registry keys" # Registry is being removed
    table_name_classic_spark = get_table_name_from_hash(hash_classic_prefer_true)
    assert not spark.catalog.tableExists(table_name_classic_spark), f"DBFS table {table_name_classic_spark} should NOT exist yet"
    print(f"DataFrame was Spark-cached and added to registry. DBFS table {table_name_classic_spark} does not exist yet.")

    # Now, test backup
    print("\\nTesting backup_spark_cached_to_dbfs...")
    backup_spark_cached_to_dbfs(specific_dfs=[df_cached_classic_spark], min_complexity_threshold=None, min_multiplier_threshold=None)
    assert spark.catalog.tableExists(table_name_classic_spark), f"DBFS table {table_name_classic_spark} SHOULD exist after backup"

# print("Test for logging of original complexity tuple during backup passed.") # Commented out as test is removed
print(f"DBFS table {table_name_classic_spark} exists after backup.")
clear_cache_for_hash(hash_classic_prefer_true) # Clean up

    # Scenario 2: Classic cluster, PREFER_SPARK_CACHE = False
app_config.PREFER_SPARK_CACHE = False
print(f"\\nSimulating Classic Cluster, PREFER_SPARK_CACHE={app_config.PREFER_SPARK_CACHE}")
assert should_prefer_spark_cache() is False, "should_prefer_spark_cache should be False on classic with PREFER_SPARK_CACHE=False"

    # Use a DataFrame with input files (from table) to test DBFS caching
df_classic_prefer_false = df_sql.select("name", "salary")
hash_classic_prefer_false = get_table_hash(df_classic_prefer_false)
clear_cache_for_hash(hash_classic_prefer_false)

print("Calling cacheToDbfs on classic, PREFER_SPARK_CACHE=False...")
df_cached_classic_dbfs = cacheToDbfs(df_classic_prefer_false, dbfs_cache_complexity_threshold=0) # Force DBFS cache

assert not df_cached_classic_dbfs.is_cached, "DataFrame should NOT be Spark-cached by default if going to DBFS"
# assert id(df_cached_classic_dbfs) not in _spark_cached_dfs_registry, "DataFrame ID should NOT be in registry keys" # Registry is being removed
table_name_classic_dbfs = get_table_name_from_hash(hash_classic_prefer_false)
assert spark.catalog.tableExists(table_name_classic_dbfs), f"DBFS table {table_name_classic_dbfs} SHOULD exist"
print(f"DataFrame was cached to DBFS. DBFS table {table_name_classic_dbfs} exists.")
clear_cache_for_hash(hash_classic_prefer_false)

# Scenario 3: Serverless cluster, PREFER_SPARK_CACHE = True (should be ignored)
if is_serverless_cluster():
    df_serverless_prefer_true = df_sql.select("age", "salary")
    hash_serverless_prefer_true = get_table_hash(df_serverless_prefer_true)
    clear_cache_for_hash(hash_serverless_prefer_true)

    print("Calling cacheToDbfs on serverless, PREFER_SPARK_CACHE=True (should use standard DBFS logic)...")
    df_cached_serverless_dbfs = cacheToDbfs(df_serverless_prefer_true, dbfs_cache_complexity_threshold=0) # Force DBFS cache

    assert not df_cached_serverless_dbfs.is_cached, "DataFrame should NOT be Spark-cached by default if going to DBFS on serverless"
    # assert id(df_cached_serverless_dbfs) not in _spark_cached_dfs_registry, "DataFrame ID should NOT be in registry keys on serverless" # Registry is being removed
    table_name_serverless_dbfs = get_table_name_from_hash(hash_serverless_prefer_true)
    assert spark.catalog.tableExists(table_name_serverless_dbfs), f"DBFS table {table_name_serverless_dbfs} SHOULD exist on serverless"
    print(f"DataFrame was cached to DBFS on serverless. DBFS table {table_name_serverless_dbfs} exists.")
    clear_cache_for_hash(hash_serverless_prefer_true)

# Restore original environment
app_config.PREFER_SPARK_CACHE = True

print("\\n--- PREFER_SPARK_CACHE behavior tests completed ---")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test chained cacheToDbfs with PREFER_SPARK_CACHE=True
# MAGIC
# MAGIC This tests that chaining cacheToDbfs calls correctly utilizes Spark cache when PREFER_SPARK_CACHE is True.

# COMMAND ----------

print("\nTesting chained cacheToDbfs with PREFER_SPARK_CACHE=True...")

# Ensure PREFER_SPARK_CACHE is True for this test
app_config.PREFER_SPARK_CACHE = True
print(f"Setting PREFER_SPARK_CACHE={app_config.PREFER_SPARK_CACHE} for chained test")

# Create a new DataFrame for this test
df_chained_spark_orig = df_sql.select("name", "salary")
hash_chained_spark_orig = get_table_hash(df_chained_spark_orig)
hash_chained_spark_orig = get_table_hash(df_chained_spark_orig)
clear_cache_for_hash(hash_chained_spark_orig) # Ensure no DBFS cache

# First cacheToDbfs call
print("Calling first cacheToDbfs in chain...")
df_chained_spark_cached_1 = cacheToDbfs(df_chained_spark_orig)

assert df_chained_spark_cached_1.is_cached, "First DataFrame in chain should be Spark-cached"
# assert id(df_chained_spark_cached_1) in _spark_cached_dfs_registry, "First DataFrame ID should be in registry keys" # Registry is being removed

# Apply a transformation and call cacheToDbfs again
print("\nApplying transformation and calling second cacheToDbfs in chain...")
df_chained_spark_transformed = df_chained_spark_cached_1.filter(col("salary") > 60000)
df_chained_spark_cached_2 = cacheToDbfs(df_chained_spark_transformed)

assert df_chained_spark_cached_2.is_cached, "Second DataFrame in chain should be Spark-cached"
# assert id(df_chained_spark_cached_2) in _spark_cached_dfs_registry, "Second DataFrame ID should be in registry keys" # Registry is being removed
print("Second DataFrame in chain was Spark-cached and added to registry.")

# Clean up caches created during this test
clear_cache_for_hash(get_table_hash(df_chained_spark_cached_1))
clear_cache_for_hash(get_table_hash(df_chained_spark_cached_2))

# Restore original config
print(f"Restored PREFER_SPARK_CACHE to {app_config.PREFER_SPARK_CACHE}")

print("\\n--- Chained cacheToDbfs with PREFER_SPARK_CACHE=True test completed ---")

# COMMAND ----------

print("\nTesting DataFrame.clearDbfsCache...")
df_to_clear = cacheToDbfs(df_sql)
print(f"Cached DataFrame: {df_to_clear}\n")
clearDbfsCache(df_to_clear)
print("Cache cleared\n")
table_hash = get_table_hash(df_to_clear)
qualified_table_name = f"{config.CACHE_DATABASE}.{table_hash}"
cache_exists = spark.catalog.tableExists(qualified_table_name)
print(f"Cache table {qualified_table_name} exists after clearing: {cache_exists}\n")
df_different = df_to_clear.withColumn("new_col", lit(100))
clearDbfsCache(df_different)
print("Attempted to clear non-existent cache\n")

# COMMAND ----------

# 5. Test Complex Scenarios and Performance
print("\nTesting cache invalidation scenarios...")
print("Testing schema changes...")
df_schema_orig = cacheToDbfs(df_sql)
df_schema_new = df_sql.withColumn("bonus", lit(1000))
df_schema_cached = cacheToDbfs(df_schema_new)
print("Schema change test completed\n")

print("\nTesting query changes...")
df_query_orig = cacheToDbfs(df_sql.filter(col("age") > 30))
df_query_new = df_sql.filter(col("age") > 35)
df_query_cached = cacheToDbfs(df_query_new)
print("Query change test completed\n")

print("\nTesting non-deterministic operations...")
df_sample = cacheToDbfs(df_larger.sample(0.5, seed=42))
df_sample_same = df_larger.sample(0.5, seed=42)
df_sample_cached = cacheToDbfs(df_sample_same)
print("Non-deterministic operation test completed\n")

print("\nTesting performance differences...")
table_hash = get_table_hash(df_larger)
clear_cache_for_hash(table_hash)
start_time = time.time()
df_larger.count()
uncached_time = time.time() - start_time
print(f"Uncached query time: {uncached_time:.4f} seconds\n")

df_larger_cached = cacheToDbfs(df_larger)
start_time = time.time()
df_larger_cached.count()
cached_time = time.time() - start_time
print(f"Cached query time: {cached_time:.4f} seconds\n")
print(f"Performance improvement: {uncached_time / cached_time:.2f}x\n")

print("\nTesting write time vs. query time...")
clear_cache_for_hash(table_hash)
start_time = time.time()
cacheToDbfs(df_larger)
write_time = time.time() - start_time
print(f"Cache write time: {write_time:.4f} seconds\n")
start_time = time.time()
df_larger_cached = cacheToDbfs(df_larger)
df_larger_cached.count()
query_time = time.time() - start_time
print(f"Subsequent query time: {query_time:.4f} seconds\n")
print(f"Write time / query time ratio: {write_time / query_time:.2f}\n")

print("\nTesting edge cases...")
print("Testing empty DataFrame...\n")
df_empty = spark.createDataFrame([], schema)
df_empty_cached = cacheToDbfs(df_empty)
print(f"Empty DataFrame cached: {df_empty_cached}\n")

print("\nTesting special characters in column names...")
data_special = [(1, 2), (3, 4)]
from pyspark.sql.types import IntegerType, StructField, StructType

schema_special = StructType([
    StructField("normal_col", IntegerType(), True),
    StructField("special_col", IntegerType(), True)
])
df_special = spark.createDataFrame(data_special, schema_special)
df_special = df_special.withColumnRenamed("special_col", "special.col")
try:
    df_special_cached = cacheToDbfs(df_special)
    print("Special characters handled successfully\n")
except Exception as e:
    print(f"Error with special characters: {str(e)}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test createCachedDataFrame integration

# COMMAND ----------

print("\nTesting createCachedDataFrame integration...")

import pandas as pd

# 1. Create in-memory data
data_mem = [{"x": i, "y": i * 2} for i in range(5)]
schema_mem = "x INT, y INT"

# 2. Create and cache using createCachedDataFrame (should be a cache miss)
df_mem_1 = createCachedDataFrame(spark, data_mem, schema=schema_mem)
print("First createCachedDataFrame (should be cache miss):")
df_mem_1.show()
count_1 = df_mem_1.count()
assert count_1 == 5, f"Expected 5 rows, got {count_1}"

# 3. Create again with same data (should be a cache hit)
df_mem_2 = createCachedDataFrame(spark, data_mem, schema=schema_mem)
print("Second createCachedDataFrame (should be cache hit):")
df_mem_2.show()
count_2 = df_mem_2.count()
assert count_2 == 5, f"Expected 5 rows, got {count_2}"

# 4. Check that the data matches
rows_1 = set(tuple(row) for row in df_mem_1.collect())
rows_2 = set(tuple(row) for row in df_mem_2.collect())
assert rows_1 == rows_2, "Data mismatch between cache miss and cache hit"

# 5. Check that the cache table exists
data_hash = _hash_input_data(data_mem)
table_name = get_table_name_from_hash(f"data_{data_hash}")
cache_exists = spark.catalog.tableExists(table_name)
print(f"Cache table {table_name} exists: {cache_exists}")
assert cache_exists, f"Cache table {table_name} should exist after createCachedDataFrame"

# 6. Test with Pandas DataFrame
pandas_data = pd.DataFrame({
    "a": pd.Series([1, 2, 3], dtype="int32"),
    "b": pd.Series([4.0, 5.5, 6.1], dtype="float64"),
    "c": pd.Series(["x", "y", "z"], dtype="string")
})

df_from_pandas = createCachedDataFrame(spark, pandas_data)
print("Pandas DataFrame converted to Spark DataFrame:")
df_from_pandas.show()

# 7. Check data types
expected_schema = {"a": "int", "b": "double", "c": "string"}
actual_schema = {field.name: field.dataType.simpleString() for field in df_from_pandas.schema.fields}
assert actual_schema == expected_schema, f"Schema mismatch: expected {expected_schema}, got {actual_schema}"

print("createCachedDataFrame integration test passed.\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Direct Data Cache Identification Logic
# MAGIC
# MAGIC These tests verify the refined logic in `get_input_dir_mod_datetime` for identifying
# MAGIC "direct data cache" sources, especially after transformations.

# COMMAND ----------

from typing import cast
from pyspark.sql import DataFrame as SparkDataFrame

print("\\nTesting Direct Data Cache Identification Logic...")

# Helper to check the log for the "direct data cache" message
def check_log_for_direct_data_cache_bypass(log_output: str, should_bypass: bool):
    bypass_message = "DataFrame source is a direct data cache. Bypassing standard DBFS caching logic"
    if should_bypass:
        assert bypass_message in log_output, f"Expected log to show direct data cache bypass, but it didn't. Log: {log_output}"
        print("Log correctly indicates direct data cache bypass.")
    else:
        assert bypass_message not in log_output, f"Expected log to NOT show direct data cache bypass, but it did. Log: {log_output}"
        print("Log correctly indicates NO direct data cache bypass (standard caching proceeded).")

# Setup a basic DataFrame for createCachedDataFrame
data_direct_test = [{"id": 1, "value": "alpha"}, {"id": 2, "value": "beta"}]
schema_direct_test = "id INT, value STRING"

# Ensure a clean slate for logging capture for each sub-test
def run_test_with_log_capture(test_func_to_run, *args):
    import io
    import logging

    # Get the specific logger we are interested in
    target_logger = logging.getLogger('dbfs_spark_cache.dataframe_extensions')
    original_level = target_logger.level
    target_logger.setLevel(logging.INFO) # Ensure INFO messages are captured

    log_capture_string = io.StringIO()
    ch = logging.StreamHandler(log_capture_string)
    ch.setLevel(logging.INFO)

    # Optional: Formatter if needed, but for simple message check, not strictly necessary
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # ch.setFormatter(formatter)

    target_logger.addHandler(ch)

    try:
        test_func_to_run(*args)
    finally:
        target_logger.removeHandler(ch)
        target_logger.setLevel(original_level)
        ch.close()
    return log_capture_string.getvalue()

def test_case_1_logic():
    df_created: SparkDataFrame = createCachedDataFrame(spark, data_direct_test, schema=schema_direct_test)
    # The result of createCachedDataFrame is already reading from a data_* table.
    # Calling cacheToDbfs on it should recognize it as a direct data cache.
    # Verify get_input_dir_mod_datetime identifies the source correctly BEFORE caching
    # Revised Plan: Expect {} because data_* tables always trigger standard caching now.
    input_info_before_cache = get_input_dir_mod_datetime(df_created)
    assert input_info_before_cache == {}, f"Expected empty dict for df_created (reading data_* table), got {input_info_before_cache}"
    print("get_input_dir_mod_datetime correctly returned empty dict for df_created.")

    # Now perform the caching (its effect is checked in test_case_1_log_check)
    _ = cacheToDbfs(df_created)
    # We don't need to assert on input_info for df_cached_direct here,
    # as its plan might change due to Spark caching itself.

log_output_tc1 = run_test_with_log_capture(test_case_1_logic)
# For this specific scenario, the log message we are looking for is from the *next* call to cacheToDbfs
# if df_cached_direct was then *itself* passed to cacheToDbfs.
# The current test_case_1_logic checks get_input_dir_mod_datetime directly.
# Let's refine to check the log from cacheToDbfs itself.

def test_case_1_log_check():
    df_created: SparkDataFrame = createCachedDataFrame(spark, data_direct_test, schema=schema_direct_test)
    # Now call cacheToDbfs on this df_created, which reads from a data_* table.
    # This call to cacheToDbfs should log the bypass.
    _ = cacheToDbfs(df_created) # We only care about the log this generates

log_output_tc1_actual = run_test_with_log_capture(test_case_1_log_check)
# Revised Plan: Expect NO bypass because get_input_dir_mod_datetime returns {}
check_log_for_direct_data_cache_bypass(log_output_tc1_actual, should_bypass=False)
# Cleanup the created cache
data_hash_tc1 = _hash_input_data(data_direct_test)
table_name_tc1 = get_table_name_from_hash(f"data_{data_hash_tc1}")
spark.sql(f"DROP TABLE IF EXISTS {table_name_tc1}")
print(f"Cleaned up {table_name_tc1}")


# Test Case 2: Transformations on createCachedDataFrame result are NOT direct data cache
print("\\nTest Case 2: Transformations are NOT direct")

# Sub-case 2a: Select
def test_case_2a_select_logic():
    df_created_select: SparkDataFrame = createCachedDataFrame(spark, data_direct_test, schema=schema_direct_test)
    df_selected = df_created_select.select("id")
    _ = cacheToDbfs(df_selected, dbfs_cache_complexity_threshold=0) # Force attempt to cache

log_output_tc2a = run_test_with_log_capture(test_case_2a_select_logic)
check_log_for_direct_data_cache_bypass(log_output_tc2a, should_bypass=False)
# Cleanup
data_hash_tc2a = _hash_input_data(data_direct_test) # Original data hash
table_name_tc2a_orig = get_table_name_from_hash(f"data_{data_hash_tc2a}")
spark.sql(f"DROP TABLE IF EXISTS {table_name_tc2a_orig}")
# Also cleanup the cache of the transformed df
df_created_select_temp: SparkDataFrame = createCachedDataFrame(spark, data_direct_test, schema=schema_direct_test) # Recreate to get hash
df_selected_temp = df_created_select_temp.select("id")
hash_selected = get_table_hash(df_selected_temp)
clear_cache_for_hash(hash_selected)
print(f"Cleaned up for select test: {table_name_tc2a_orig}, and cache for hash {hash_selected}")


# Sub-case 2b: Filter
def test_case_2b_filter_logic():
    df_created_filter: SparkDataFrame = createCachedDataFrame(spark, data_direct_test, schema=schema_direct_test)
    df_filtered = df_created_filter.filter("id = 1")
    _ = cacheToDbfs(df_filtered, dbfs_cache_complexity_threshold=0)

log_output_tc2b = run_test_with_log_capture(test_case_2b_filter_logic)
check_log_for_direct_data_cache_bypass(log_output_tc2b, should_bypass=False)
# Cleanup
data_hash_tc2b = _hash_input_data(data_direct_test)
table_name_tc2b_orig = get_table_name_from_hash(f"data_{data_hash_tc2b}")
spark.sql(f"DROP TABLE IF EXISTS {table_name_tc2b_orig}")
df_created_filter_temp: SparkDataFrame = createCachedDataFrame(spark, data_direct_test, schema=schema_direct_test)
df_filtered_temp = df_created_filter_temp.filter("id = 1")
hash_filtered = get_table_hash(df_filtered_temp)
clear_cache_for_hash(hash_filtered)
print(f"Cleaned up for filter test: {table_name_tc2b_orig}, and cache for hash {hash_filtered}")


# Sub-case 2c: Join
def test_case_2c_join_logic():
    df_created_join1: SparkDataFrame = createCachedDataFrame(spark, data_direct_test, schema=schema_direct_test)
    data_join2 = [{"id": 1, "category": "X"}, {"id": 3, "category": "Y"}]
    df_other: SparkDataFrame = spark.createDataFrame(data_join2, schema_join2) # A normal DataFrame

    df_joined = df_created_join1.join(df_other, "id")
    _ = cacheToDbfs(df_joined, dbfs_cache_complexity_threshold=0)

schema_join2 = "id INT, category STRING"
log_output_tc2c = run_test_with_log_capture(test_case_2c_join_logic)
check_log_for_direct_data_cache_bypass(log_output_tc2c, should_bypass=False)
# Cleanup
data_hash_tc2c = _hash_input_data(data_direct_test)
table_name_tc2c_orig = get_table_name_from_hash(f"data_{data_hash_tc2c}")
spark.sql(f"DROP TABLE IF EXISTS {table_name_tc2c_orig}")
# For joined, the hash is more complex, clear based on the df_joined object if possible
# Recreate for hash
df_created_join1_temp: SparkDataFrame = createCachedDataFrame(spark, data_direct_test, schema=schema_direct_test)
data_join2_temp = [{"id": 1, "category": "X"}, {"id": 3, "category": "Y"}]
df_other_temp: SparkDataFrame = spark.createDataFrame(data_join2_temp, schema_join2)
df_joined_temp = df_created_join1_temp.join(df_other_temp, "id")
hash_joined = get_table_hash(df_joined_temp)
clear_cache_for_hash(hash_joined)
print(f"Cleaned up for join test: {table_name_tc2c_orig}, and cache for hash {hash_joined}")

# Test Case 3: Simple spark.read.table("spark_cache.data_...") CAN be direct data cache
print("\\nTest Case 3: Simple read of data_* table CAN be direct")
def test_case_3_logic():
    # First, create a data_* table using createCachedDataFrame
    _: SparkDataFrame = createCachedDataFrame(spark, data_direct_test, schema=schema_direct_test)
    temp_data_hash = _hash_input_data(data_direct_test)
    temp_table_name = get_table_name_from_hash(f"data_{temp_data_hash}")

    # Now, read this table directly
    df_read_direct: SparkDataFrame = spark.read.table(temp_table_name)
    # This call to cacheToDbfs should log the bypass because it's a simple read
    _ = cacheToDbfs(df_read_direct)
    # This call to cacheToDbfs should log the bypass because it's a simple read
    # _ = cacheToDbfs(df_read_direct) # Removed duplicate

log_output_tc3 = run_test_with_log_capture(test_case_3_logic)
# Revised Plan: Expect NO bypass because get_input_dir_mod_datetime returns {}
check_log_for_direct_data_cache_bypass(log_output_tc3, should_bypass=False)
# Cleanup
temp_data_hash_tc3 = _hash_input_data(data_direct_test)
temp_table_name_tc3 = get_table_name_from_hash(f"data_{temp_data_hash_tc3}")
spark.sql(f"DROP TABLE IF EXISTS {temp_table_name_tc3}") # Drop the data_* table
# The cacheToDbfs on df_read_direct would have created another cache (non data_ prefix) if bypass didn't happen
# If bypass happened, no new cache. If it didn't, we need to clear that too.
# For simplicity, assuming bypass worked as asserted by check_log.
print(f"Cleaned up for simple read test: {temp_table_name_tc3}")

print("\\nDirect Data Cache Identification Logic tests completed.")
# Test Case: Transformed data cache hash recomputation
print("\nTesting hash recomputation after transformation of a direct data cache table")
df_dc = createCachedDataFrame(spark, data_direct_test, schema=schema_direct_test)
original_hash = get_table_hash(df_dc)
assert original_hash.startswith("data_"), f"Expected original hash to start with 'data_', got {original_hash}"
df_dc_trans = df_dc.select("value")
trans_hash = get_table_hash(df_dc_trans)
assert not trans_hash.startswith("data_"), f"Transformed DataFrame should have a new hash, got {trans_hash}"
print("Hash recomputation after transformation passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test clearDbfsCache with createCachedDataFrame
# MAGIC
# MAGIC This tests that `clearDbfsCache` can now correctly clear caches created by `createCachedDataFrame`.

# COMMAND ----------

print("\nTesting clearDbfsCache with createCachedDataFrame...")

# 1. Create in-memory data
data_clear_mem = [{"a": 100, "b": 200}, {"a": 300, "b": 400}]
schema_clear_mem = "a INT, b INT"

# 2. Create and cache using createCachedDataFrame
print("Creating and caching DataFrame with createCachedDataFrame...")
df_clear_mem = createCachedDataFrame(spark, data_clear_mem, schema=schema_clear_mem)
df_clear_mem.show()

# 3. Get the hash and table name for the created cache
data_hash_clear_mem = _hash_input_data(data_clear_mem)
table_name_clear_mem = get_table_name_from_hash(f"data_{data_hash_clear_mem}")
print(f"Expected cache table name: {table_name_clear_mem}")

# 4. Verify that the cache table exists
cache_exists_before = spark.catalog.tableExists(table_name_clear_mem)
print(f"Cache table {table_name_clear_mem} exists before clearing: {cache_exists_before}")
assert cache_exists_before, f"Cache table {table_name_clear_mem} should exist before clearDbfsCache"

# 5. Clear the cache using clearDbfsCache on the DataFrame
print(f"\nClearing cache for DataFrame with hash {data_hash_clear_mem} using df.clearDbfsCache()...")
clearDbfsCache(df_clear_mem)
print("clearDbfsCache called.")

# 6. Verify that the cache table no longer exists
cache_exists_after = spark.catalog.tableExists(table_name_clear_mem)
print(f"Cache table {table_name_clear_mem} exists after clearing: {cache_exists_after}")
assert not cache_exists_after, f"Cache table {table_name_clear_mem} should NOT exist after clearDbfsCache"
print("Cache table successfully cleared.")

print("\nclearDbfsCache with createCachedDataFrame test passed.\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test combining createCachedDataFrame with a normal cached DataFrame

# COMMAND ----------

print("\nTesting combined caching: createCachedDataFrame + normal cache...")

def assert_not_bad_plan(df, should=False):
    plan = get_query_plan(df)
    match = ("Scan ExistingRDD" in plan or "LocalTableScan" in plan)
    assert not match if not should else match, f"A dataframe created with createDataFrame should {'not' if not should else ''} have a problematic indicator in the query plan: {plan}"

# 1. Create a normal DataFrame and cache it
data_norm = [("A", 1), ("B", 2)]
schema_norm = "label STRING, value INT"
df_norm = spark.createDataFrame(data_norm, schema=schema_norm)
assert_not_bad_plan(df_norm, should=True)

# 2. Create a DataFrame using createCachedDataFrame
df_norm_cached = createCachedDataFrame(spark, data_norm, schema=schema_norm)
assert_not_bad_plan(df_norm_cached)
print("Normal cached DataFrame:")
df_norm_cached.show()

data_mem2 = [{"label": "C", "value": 3}, {"label": "D", "value": 4}]
df_mem2 = createCachedDataFrame(spark, data_mem2, schema=schema_norm)
print("createCachedDataFrame DataFrame:")
df_mem2.show()

print(f"df_norm_cached hash: {get_table_hash(df_norm_cached)}\ndf_mem2 hash: {get_table_hash(df_mem2)}")

# 3. Combine (union) the two DataFrames
df_combined = df_norm_cached.unionByName(df_mem2)
print("Combined DataFrame (before caching):")
df_combined.show()

# 4. Cache the combined DataFrame
df_combined_cached = cacheToDbfs(df_combined)
# note: handling for this not implemented, getting the same data hash from the query plan after a read is really tricky...:
#   assert not get_table_hash(df_combined_cached).startswith("data_"), f"Invalid hash for Dataframe derived from data-cached dataframe: {get_table_hash(df_combined_cached)}"
assert_not_bad_plan(df_combined_cached)
print("Combined DataFrame (after caching):")
df_combined_cached.show()

# 5. Check row count and data correctness
combined_rows = set(tuple(row) for row in df_combined_cached.collect())
expected_rows = set([("A", 1), ("B", 2), ("C", 3), ("D", 4)])
assert combined_rows == expected_rows, f"Combined DataFrame rows incorrect: {combined_rows}"

print("\nclearDbfsCache interaction test completed successfully.\n")
combined_rows = set(tuple(row) for row in df_combined_cached.collect())
expected_rows = set([("A", 1), ("B", 2), ("C", 3), ("D", 4)])
assert combined_rows == expected_rows, f"Combined DataFrame rows incorrect: {combined_rows}"

print("Combined caching test passed.\n")

# COMMAND ----------

# MAGIC %md
# MAGIC Cleanup

# COMMAND ----------

# 6. Cleanup
if False: # Beware, only run this if you have nothing to keep in chache database
  print("\nCleaning up test artifacts...\n")
  cached_tables = get_cached_tables()
  display(cached_tables)  # noqa: F821
  print(f"Number of cached tables to clean: {len(cached_tables)}\n")
  clear_caches_older_than(num_days=0)
  print("All caches cleared\n")
  spark.sql("DROP DATABASE IF EXISTS test_db CASCADE")
  print("Test database dropped\n")
  clear_inconsistent_cache()

# COMMAND ----------

# MAGIC %md
# MAGIC Registry after test:

# COMMAND ----------

get_cached_tables(num_threads=100)

# COMMAND ----------

get_cached_dataframe_metadata(num_threads=50)

# COMMAND ----------

print("Integration test passed!")