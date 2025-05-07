import logging
from typing import Any  # Added Optional

# Import pyspark components first
from pyspark.sql import SparkSession

from .dataframe_extensions import (  # Import from dataframe_extensions
    __withCachedDisplay__,
    # backupSparkCachedToDbfs, # Removed as it's no longer an extension method
    cacheToDbfs,
    clearDbfsCache,
    clearSparkCachedRegistry,
    extend_dataframe_methods,
)
from .utils import (  # Import from utils
    _spark_cached_dfs_registry,
    get_hash_from_metadata,
    is_serverless_cluster,
)


def backup_spark_cached_to_dbfs(spark_session, specific_dfs=None, unpersist_after_backup=False):
    """
    Backup all Spark-cached DataFrames to DBFS, or a specific list if provided.
    """
    from pyspark.sql import DataFrame
    dfs_to_process = []
    if specific_dfs is None:
        # Iterate over a copy of the registry if processing all
        dfs_to_process = list(_spark_cached_dfs_registry)
    else:
        dfs_to_process = list(specific_dfs) # Ensure it's a list copy if it's an iterator

    processed_dfs = [] # Keep track of DataFrames successfully backed up

    for df in dfs_to_process:
        if not isinstance(df, DataFrame):
            log.warning(f"Item {df} in backup queue is not a DataFrame, skipping.")
            continue

        # Check is spark cached
        log.info(f"Is in spark cache, useMemory: {df.storageLevel.useMemory}, useDisk: {df.storageLevel.useDisk}, any: {df.storageLevel.useMemory or df.storageLevel.useDisk}, is_cached: {df.is_cached}")
        # Directly inline the logic that was in the DataFrame's backupAllSparkCachedToDbfs method
        try:
            # Ensure get_table_hash and write_dbfs_cache are available in this scope
            # They are imported from .core_caching at the module level.
            log.info(f"Backing up Spark-cached DataFrame (hash: {get_table_hash(df)}) to DBFS.")
            write_dbfs_cache(df)
            processed_dfs.append(df)  # Add to processed list only if write_dbfs_cache call succeeds

            if unpersist_after_backup and hasattr(df, "unpersist"):
                log.info("Unpersisting DataFrame after backup...")
                df.unpersist()
        except Exception as e:
            log.error(f"Error during DBFS backup for DataFrame (hash: {get_table_hash(df)}): {e}", exc_info=True)

    # Now, remove all successfully processed DataFrames from the global registry
    # This is only relevant if we were processing the global registry (specific_dfs was None)
    if specific_dfs is None:
        for df_processed in processed_dfs:
            try:
                _spark_cached_dfs_registry.remove(df_processed)
                log.info("Removed successfully backed-up DataFrame from Spark cache registry.")
            except KeyError:
                log.warning("Attempted to remove DataFrame from registry, but it was not found (already removed or never added).")

from .cache_management import (  # Import from cache_management
    clear_cache_for_hash,
    clear_caches_older_than,
    clear_inconsistent_cache,
    get_cached_dataframe_metadata,
    get_cached_tables,
    get_tables_from_database,
)
from .core_caching import (  # Import from core_caching
    DF_DBFS_CACHE_QUEUE,  # Re-export
    add_to_dbfs_cache_queue,
    cache_dataframes_in_queue_to_dbfs,
    createCachedDataFrame,
    get_cache_metadata,
    get_input_dir_mod_datetime,  # Re-export
    get_query_plan,
    get_table_cache_info,  # Used by tests
    get_table_hash,  # get_table_hash is used by DataFrame extensions
    read_dbfs_cache_if_exist,
    write_dbfs_cache,
)
from .query_complexity_estimation import estimate_compute_complexity

# Configure module-level logger using __name__ first
log = logging.getLogger(__name__)

# Then import databricks runtime stuff
try:
    from databricks.sdk.runtime import display, spark
except ImportError:
    log.warning("databricks.sdk.runtime not found. display, spark will not be available.")
    # Define display as a function placeholder, ignoring potential signature mismatch
    def display(*args: Any, **kwargs: Any) -> None: # type: ignore[misc]
        """Placeholder display function when databricks environment is not available."""
        if args:
            print(args[0]) # Print first arg like original lambda
    try:
        spark = SparkSession.builder.appName("dbfs_spark_cache_local").getOrCreate() # type: ignore[misc]
    except Exception as e:
        log.error(f"Failed to get or create SparkSession locally: {e}")
        spark = None # type: ignore[assignment]

from .config import config  # Import config to use PREFER_SPARK_CACHE


def should_prefer_spark_cache() -> bool:
    """Determines if Spark cache should be preferred over DBFS cache."""
    return config.PREFER_SPARK_CACHE and not is_serverless_cluster()

__all__ = [
    "createCachedDataFrame",
    "read_dbfs_cache_if_exist",
    "write_dbfs_cache",
    "add_to_dbfs_cache_queue",
    "cache_dataframes_in_queue_to_dbfs",
    "get_tables_from_database",
    "get_cached_tables",
    "clear_caches_older_than",
    "clear_inconsistent_cache",
    "clear_cache_for_hash",
    "get_cached_dataframe_metadata",
    "get_table_hash",
    "is_serverless_cluster",
    "get_hash_from_metadata",
    "get_query_plan",
    "get_cache_metadata",
    "get_table_cache_info",
    "get_input_dir_mod_datetime", # Re-export
    "DF_DBFS_CACHE_QUEUE", # Re-export
    "_spark_cached_dfs_registry", # Re-export
    "should_prefer_spark_cache", # Re-exported re-implementation
    "estimate_compute_complexity", # Re-export
    # Re-export DataFrame extension methods and related functions
    "cacheToDbfs",
    "clearSparkCachedRegistry",
    "clearDbfsCache",
    "__withCachedDisplay__",
    "extend_dataframe_methods",
]
