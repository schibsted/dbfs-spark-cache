import logging  # Add logging
from datetime import datetime  # Add missing import
from typing import Dict, Optional  # Add missing imports

from pyspark.sql import DataFrame

from .config import config
from .core_caching import (  # createCachedDataFrame, # Not used by this version of cacheToDbfs
    get_input_dir_mod_datetime,
    get_query_plan,
    get_table_hash,  # for clearDbfsCache
    read_dbfs_cache_if_exist,
    write_dbfs_cache,
)
from .utils import _spark_cached_dfs_registry, is_serverless_cluster

# from .query_complexity_estimation import estimate_compute_complexity # Remove module-level import
log = logging.getLogger(__name__)

# Define locally to avoid circular import with caching.py
def should_prefer_spark_cache() -> bool:
    """Determines if Spark cache should be preferred over DBFS cache."""
    return config.PREFER_SPARK_CACHE and not is_serverless_cluster()



def cacheToDbfs(
    self: DataFrame,
    dbfs_cache_complexity_threshold: Optional[int] = None,
    dbfs_cache_multiplier_threshold: Optional[float] = None,
    replace: bool = False, # if true, will write even if complexity is low or cache exists
    verbose: bool = False,
    # deferred: bool = False, # Deferred logic removed for now, can be added back if needed
    **kwargs # Allow other kwargs to be passed to underlying functions if necessary
) -> DataFrame:
    """
    Extends DataFrame with a method to cache it to DBFS based on query plan and input data.
    Skips caching if DataFrame reads from an existing RDD (e.g. .parallelize()).
    """
    # Get query plan once and reuse it
    query_plan_str = get_query_plan(self)

    # Check if the plan indicates a scan of an existing RDD.
    # This is a common pattern for DataFrames created from spark.sparkContext.parallelize()
    # or other RDD-based sources that don't have a file-based input.
    if "Scan ExistingRDD" in query_plan_str:
        log.info("DataFrame source appears to be an existing RDD (e.g., from parallelize). Skipping DBFS and spark cache.")
        return self

    input_info = get_input_dir_mod_datetime(self)

    # Check for direct data cache bypass condition
    if input_info == {"<direct_data_cache>": True}:
        log.info("DataFrame source is a direct data cache. Bypassing standard DBFS caching logic and returning self.")
        # Ensure it's added to the Spark cache registry if it's a form of caching
        _spark_cached_dfs_registry.add(self)
        return self

    # Determine thresholds from config if not provided
    # Ensure input_info is Dict[str, datetime] for hashing, filter out bool if present
    # This is important because input_info could be {"<direct_data_cache>": True} if not caught above,
    # or {"<schema_changed_placeholder>": datetime.now()}
    datetime_input_info: Dict[str, datetime] = {
        k: v for k, v in input_info.items() if isinstance(v, datetime)
    }

    if not replace:
        # Pass the original input_info (which might be schema_changed_placeholder) to read_dbfs_cache_if_exist
        # as it handles these special markers.
        cached_df = read_dbfs_cache_if_exist(self, query_plan=query_plan_str, input_dir_mod_datetime=input_info)
        if cached_df is not None:
            log.info("Returning existing DBFS cache.")
            # Do not add to _spark_cached_dfs_registry here,
            # as this is a DBFS cache hit, not a Spark in-memory cache.
            return cached_df

    # If not replacing, check if complexity/multiplier thresholds (if provided and >0) warrant skipping.
    # If threshold parameters are None, these checks are skipped.
    should_skip_due_to_thresholds = False
    if not replace:
        # Determine if complexity/multiplier estimation is needed.
        # Estimation is needed if either threshold parameter is provided and is positive.
        needs_estimation = \
            (dbfs_cache_complexity_threshold is not None and dbfs_cache_complexity_threshold > 0) or \
            (dbfs_cache_multiplier_threshold is not None and dbfs_cache_multiplier_threshold > 0)

        if needs_estimation:
            try:
                from .query_complexity_estimation import estimate_compute_complexity
                total_size_gb, multiplier, compute_complexity = estimate_compute_complexity(self)
                log.info(f"Estimated compute complexity: {compute_complexity:.2f} (Size: {total_size_gb:.2f}GB, Multiplier: {multiplier:.2f}x)")

                # Check complexity if its parameter was provided and is positive
                if dbfs_cache_complexity_threshold is not None and dbfs_cache_complexity_threshold > 0:
                    if compute_complexity < dbfs_cache_complexity_threshold:
                        log.info(f"Complexity {compute_complexity:.2f} < explicit threshold {dbfs_cache_complexity_threshold}. Skipping DBFS cache.")
                        should_skip_due_to_thresholds = True

                # Check multiplier if its parameter was provided and is positive (and not already skipped)
                if not should_skip_due_to_thresholds and \
                   dbfs_cache_multiplier_threshold is not None and dbfs_cache_multiplier_threshold > 0:
                    if multiplier < dbfs_cache_multiplier_threshold:
                        log.info(f"Multiplier {multiplier:.2f}x < explicit threshold {dbfs_cache_multiplier_threshold:.2f}x. Skipping DBFS cache.")
                        should_skip_due_to_thresholds = True
            except Exception as e:
                log.warning(f"Could not estimate compute complexity: {e}. Proceeding with cache write attempt.")
                # should_skip_due_to_thresholds remains False, so it proceeds to cache

        if should_skip_due_to_thresholds:
            return self

    # Check if we should prefer Spark caching
    if should_prefer_spark_cache():
        log.info("Preferring Spark cache. Caching DataFrame in memory.")
        if replace:
            log.info("`replace=True` is ignored for DBFS operations as Spark in-memory cache is prioritized for this action.")
        # Cache the DataFrame in Spark memory
        cached_df = self.cache()
        # Add to registry for potential backup to DBFS later
        _spark_cached_dfs_registry.add(cached_df)
        return cached_df

    log.info("Writing to DBFS cache.")
    # write_dbfs_cache expects Dict[str, datetime]
    # Explicitly name all args for write_dbfs_cache to satisfy Pyright, even if some are defaults from the signature
    written_df = write_dbfs_cache(
        df=self,
        query_plan=query_plan_str,
        input_dir_mod_datetime=datetime_input_info,
        hash_name=kwargs.get("hash_name"), # Pass through if provided in kwargs
        cache_path=kwargs.get("cache_path", config.SPARK_CACHE_DIR), # Pass through or use default
        verbose=verbose
    )
    # Do not add to _spark_cached_dfs_registry here if we are writing to DBFS,
    # unless should_prefer_spark_cache() was true and it was added above.
    # The registry is for Spark in-memory cached DFs.
    return written_df


# backupSparkCachedToDbfs method is removed as its logic is now in caching.py

def clearSparkCachedRegistry() -> None:
    """
    Clear the global registry of Spark cached DataFrames.
    """
    _spark_cached_dfs_registry.clear()


def clearDbfsCache(self: DataFrame) -> None:
    """
    Clear the DBFS cache for this DataFrame.
    """
    from .cache_management import clear_cache_for_hash

    table_hash = get_table_hash(self)
    clear_cache_for_hash(table_hash)


def __withCachedDisplay__(self: DataFrame, *args, **kwargs) -> None:
    """
    Display the DataFrame with caching.
    """
    # This is a placeholder for a method that displays the DataFrame with caching
    # For example, it could use the databricks display function or similar
    try:
        from databricks.sdk.runtime import display
    except ImportError:
        display = print  # type: ignore[assignment] # fallback to print

    display(self)


def extend_dataframe_methods(*args, **kwargs) -> None: # Allow args/kwargs
    """
    Attach extension methods to the DataFrame class.
    """
    DataFrame.cacheToDbfs = cacheToDbfs  # type: ignore[attr-defined]
    # DataFrame.backupSparkCachedToDbfs is no longer attached as it's removed
    DataFrame.clearDbfsCache = clearDbfsCache  # type: ignore[attr-defined]
    DataFrame.withCachedDisplay = __withCachedDisplay__  # type: ignore[attr-defined]


# Optionally, attach clearSparkCachedRegistry to SparkSession or globally
# This can be done in caching.py or here if needed
