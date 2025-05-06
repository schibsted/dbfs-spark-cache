import logging
from typing import Any # Added Optional

# Import pyspark components first
from pyspark.sql import SparkSession
from .utils import is_serverless_cluster, get_hash_from_metadata, _spark_cached_dfs_registry # Import from utils
from .dataframe_extensions import ( # Import from dataframe_extensions
    extend_dataframe_methods,
    cacheToDbfs,
    backupSparkCachedToDbfs,
    clearSparkCachedRegistry,
    clearDbfsCache,
    __withCachedDisplay__,
)

def backup_spark_cached_to_dbfs(spark_session, specific_dfs=None, unpersist_after_backup=False):
    """
    Backup all Spark-cached DataFrames to DBFS, or a specific list if provided.
    """
    from pyspark.sql import DataFrame
    if specific_dfs is None:
        # Use the global registry
        dfs = list(_spark_cached_dfs_registry)
    else:
        dfs = specific_dfs
    for df in dfs:
        if not isinstance(df, DataFrame):
            continue
        from pyspark.sql import Column
        method = getattr(df, "backupSparkCachedToDbfs", None)
        if callable(method) and not isinstance(method, Column):
            method()
            if unpersist_after_backup and hasattr(df, "unpersist"):
                df.unpersist()

from .cache_management import ( # Import from cache_management
    get_tables_from_database,
    get_cached_tables,
    clear_caches_older_than,
    clear_inconsistent_cache,
    clear_cache_for_hash,
    get_cached_dataframe_metadata,
)
from .core_caching import ( # Import from core_caching
    createCachedDataFrame,
    read_dbfs_cache_if_exist,
    write_dbfs_cache,
    add_to_dbfs_cache_queue,
    cache_dataframes_in_queue_to_dbfs,
    get_table_hash, # get_table_hash is used by DataFrame extensions
    is_spark_cached, # is_spark_cached is used by DataFrame extensions
    get_query_plan,
    get_cache_metadata,
    get_table_cache_info, # Used by tests
    get_input_dir_mod_datetime, # Re-export
    DF_DBFS_CACHE_QUEUE, # Re-export
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
    "is_spark_cached",
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
    "backupSparkCachedToDbfs",
    "clearSparkCachedRegistry",
    "clearDbfsCache",
    "__withCachedDisplay__",
    "extend_dataframe_methods",
]
