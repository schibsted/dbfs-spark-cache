import logging  # Add logging
from datetime import datetime  # Add missing import
from typing import Dict, Optional # Add missing imports
from functools import partial # Add import for partial

from pyspark.sql import DataFrame, SparkSession

from .config import config
from .core_caching import (
    createCachedDataFrame,
    get_input_dir_mod_datetime,
    get_query_plan,
    get_table_hash,  # for clearDbfsCache
    read_dbfs_cache_if_exist,
    write_dbfs_cache,
)
from .utils import is_serverless_cluster

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
    verbose: bool = False,
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
        return self

    # Determine thresholds from config if not provided
    # Ensure input_info is Dict[str, datetime] for hashing, filter out bool if present
    # This is important because input_info could be {"<direct_data_cache>": True} if not caught above,
    # or {"<schema_changed_placeholder>": datetime.now()}
    datetime_input_info: Dict[str, datetime] = {
        k: v for k, v in input_info.items() if isinstance(v, datetime)
    }

    # Always check for existing cache first (replace parameter removed)
    # Pass the original input_info (which might be schema_changed_placeholder) to read_dbfs_cache_if_exist
    # as it handles these special markers.
    cached_df = read_dbfs_cache_if_exist(self, query_plan=query_plan_str, input_dir_mod_datetime=input_info)
    if cached_df is not None:
        log.info("Returning existing DBFS cache.")
        return cached_df

    # --- Complexity Estimation (Moved earlier) ---
    # Estimate complexity regardless of thresholds, for logging purposes,
    # but only if we didn't hit an existing DBFS cache.
    compute_complexity: Optional[float] = None
    multiplier: Optional[float] = None
    total_size_gb: Optional[float] = None
    try:
        from .query_complexity_estimation import estimate_compute_complexity
        # Note: estimate_compute_complexity returns (complexity, multiplier, size)
        est_complexity, est_multiplier, est_size_gb = estimate_compute_complexity(self)
        compute_complexity = est_complexity
        multiplier = est_multiplier
        total_size_gb = est_size_gb
        log.info(f"Estimated compute complexity: {compute_complexity:.2f} (Size: {total_size_gb:.5f}GB, Multiplier: {multiplier:.2f}x)")
    except Exception as e:
        log.warning(f"Could not estimate compute complexity: {e}. Proceeding without complexity-based threshold checks.")
        # Keep complexity values as None

    # --- Threshold Checks --- (replace parameter removed)
    should_skip_due_to_thresholds = False
    # Always check thresholds now
    # Check complexity threshold if estimation was successful and threshold is set
    if compute_complexity is not None and dbfs_cache_complexity_threshold is not None and dbfs_cache_complexity_threshold > 0:
        if compute_complexity < dbfs_cache_complexity_threshold:
            log.info(f"Complexity {compute_complexity:.2f} < explicit threshold {dbfs_cache_complexity_threshold}. Skipping DBFS cache.")
            should_skip_due_to_thresholds = True

    # Check multiplier threshold if estimation was successful, threshold is set, and not already skipped
    if not should_skip_due_to_thresholds and multiplier is not None and \
       dbfs_cache_multiplier_threshold is not None and dbfs_cache_multiplier_threshold > 0:
        if multiplier < dbfs_cache_multiplier_threshold:
            log.info(f"Multiplier {multiplier:.2f}x < explicit threshold {dbfs_cache_multiplier_threshold:.2f}x. Skipping DBFS cache.")
            should_skip_due_to_thresholds = True

    # If skipping due to thresholds, return the original DataFrame without caching.
    if should_skip_due_to_thresholds:
        log.info("Skipping DBFS cache due to threshold. Returning original DataFrame without caching.")
        return self

    # --- Caching Decision ---
    # Check if we should prefer Spark caching (and haven't already returned due to thresholds)
    if should_prefer_spark_cache():
        log.info("Preferring Spark cache, using df.cache()")
        # if replace: # replace parameter removed
        #     log.info("`replace=True` is ignored for DBFS operations as Spark in-memory cache is prioritized for this action.")
        # Cache the DataFrame in Spark's memory or disk storage
        cached_df = self.cache()
        return cached_df

    log.info("Writing to DBFS cache.")
    # Explicitly name all args for write_dbfs_cache to satisfy Pyright, even if some are defaults from the signature
    written_df = write_dbfs_cache(
        df=self,
        query_plan=query_plan_str,
        input_dir_mod_datetime=datetime_input_info,
        hash_name=kwargs.get("hash_name"), # Pass through if provided in kwargs
        cache_path=kwargs.get("cache_path", config.SPARK_CACHE_DIR), # Pass through or use default
        verbose=verbose
    )
    return written_df

def clearDbfsCache(self: DataFrame) -> None:
    """
    Clear the DBFS cache for this DataFrame.
    """
    from .cache_management import clear_cache_for_hash

    table_hash = get_table_hash(self)
    clear_cache_for_hash(table_hash)


def __withCachedDisplay__(self: DataFrame, *args, **kwargs) -> DataFrame:
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
    return self


def extend_dataframe_methods(spark_session: SparkSession) -> None:
    """
    Attach extension methods to the DataFrame class and SparkSession.
    """
    DataFrame.cacheToDbfs = cacheToDbfs  # type: ignore[attr-defined]
    DataFrame.clearDbfsCache = clearDbfsCache  # type: ignore[attr-defined]
    DataFrame.withCachedDisplay = __withCachedDisplay__  # type: ignore[attr-defined]
    DataFrame.wcd = __withCachedDisplay__  # type: ignore[attr-defined]

    # Attach createCachedDataFrame to the SparkSession instance
    # The method expects spark_session as its first argument, so we bind it here.
    # However, the original createCachedDataFrame in core_caching.py already takes spark_session as its first argument.
    # So, we can directly assign it.
    # setattr(spark_session, "createCachedDataFrame", createCachedDataFrame)

    # Use functools.partial to ensure the spark_session instance is passed as the first argument
    # to the core_caching.createCachedDataFrame function.
    bound_create_cached_df = partial(createCachedDataFrame, spark_session)
    setattr(spark_session, "createCachedDataFrame", bound_create_cached_df)
