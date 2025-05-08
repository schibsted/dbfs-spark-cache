import logging
from typing import Any, Optional, Iterable # Added Optional
from tqdm import tqdm

# Import pyspark components first
from pyspark.sql import SparkSession, DataFrame # DataFrame moved here for global type hint access

from .dataframe_extensions import (  # Import from dataframe_extensions
    __withCachedDisplay__,
    cacheToDbfs,
    clearDbfsCache,
    extend_dataframe_methods,
)
from .utils import (  # Import from utils
    get_hash_from_metadata,
    is_serverless_cluster,
    get_table_name_from_hash,
)


def backup_spark_cached_to_dbfs(
    specific_dfs: Iterable["DataFrame"],
    unpersist_after_backup: bool = False,
    min_complexity_threshold: Optional[float] = None,
    min_multiplier_threshold: Optional[float] = None,
    process_in_reverse_order: bool = True,
):
    """
    Backup a specific list of Spark-cached DataFrames to DBFS.
    Uses tqdm for progress tracking and allows filtering by compute complexity.

    Parameters:
    specific_dfs: Iterable[DataFrame]
        An iterable of DataFrames to back up.
    unpersist_after_backup: bool = False
        If True, unpersists the DataFrame from Spark's cache after successful backup.
    min_complexity_threshold: Optional[float] = None
        If set, only DataFrames whose estimated compute complexity value
        exceeds this threshold will be backed up.
    min_multiplier_threshold: Optional[float] = None
        If set, only DataFrames whose original estimated complexity *multiplier*
        is greater than or equal to this threshold will be backed up.
        This can be used to skip backing up DataFrames with very simple query plans.
    process_in_reverse_order: bool = True
        If True, the list of DataFrames to process (either `specific_dfs` or
        the internal registry) will be processed in reverse order.
"""
# No longer need local import of DataFrame as it's global now

    # Initialize the list of DataFrames to process
    dfs_and_complexities_to_process: list[DataFrame] = []

    # If specific_dfs is provided, prepare the list of DataFrames.
    # Complexity will be estimated during pre-filtering if needed.
    dfs_and_complexities_to_process = list(specific_dfs)
    log.info(f"Processing {len(dfs_and_complexities_to_process)} specific DataFrames provided.")
    if process_in_reverse_order:
        dfs_and_complexities_to_process.reverse()
        log.info("Processing DataFrames in reverse order.")

    initial_df_count = len(dfs_and_complexities_to_process)
    log.info(f"Found {initial_df_count} DataFrames to potentially back up.")

    # Store eligible items as (DataFrame, Optional[complexity_tuple])
    # complexity_tuple is (value, multiplier, size_gb)
    eligible_items_for_backup: list[tuple[DataFrame, Optional[tuple[float, float, float]]]] = []

    if min_complexity_threshold is not None or min_multiplier_threshold is not None:
        log.info("Filtering DataFrames based on complexity thresholds before backup...")
        for df_candidate in dfs_and_complexities_to_process:
            if not isinstance(df_candidate, DataFrame): # pragma: no cover
                log.warning(f"Item '{str(df_candidate)[:100]}...' is not a DataFrame, skipping pre-filter.")
                continue

            current_df_hash_str = "N/A"
            try:
                current_df_hash_str = get_table_hash(df_candidate)
                if current_df_hash_str.startswith("data_"):
                    log.info(f"Skipping data cache table {current_df_hash_str} as it is already cached in DBFS.")
                    continue
            except Exception as e_hash: # pragma: no cover
                log.warning(f"Could not compute hash for DataFrame '{str(df_candidate)[:100]}...' during pre-filter: {e_hash}. Skipping...")
                continue

            # Always estimate complexity if thresholds are set
            try:
                estimated_complexity_tuple = estimate_compute_complexity(df_candidate)
                complexity_value, raw_multiplier, total_size_gb = estimated_complexity_tuple
                log.info(
                    f"Estimated complexity for DataFrame (hash: {current_df_hash_str}): "
                    f"Value = {complexity_value:.2f}, Raw Multiplier = {raw_multiplier:.2f}, Input Size GB = {total_size_gb:.2f}"
                )

                skip = False
                if min_multiplier_threshold is not None and raw_multiplier < min_multiplier_threshold:
                    log.info(
                        f"Pre-filter: Skipping DataFrame (hash: {current_df_hash_str}) because its estimated "
                        f"complexity multiplier ({raw_multiplier:.2f}) is less than the threshold ({min_multiplier_threshold})."
                    )
                    skip = True

                if not skip and min_complexity_threshold is not None and complexity_value <= min_complexity_threshold:
                    log.info(
                        f"Pre-filter: Skipping DataFrame (hash: {current_df_hash_str}) due to low estimated "
                        f"complexity value ({complexity_value:.2f}) not exceeding threshold ({min_complexity_threshold})."
                    )
                    skip = True

                if not skip:
                    eligible_items_for_backup.append((df_candidate, estimated_complexity_tuple)) # Add if not skipped
                # If skipped, do not add

            except Exception as e_complexity: # pragma: no cover
                log.warning(
                    f"Could not estimate complexity for DataFrame (hash: {current_df_hash_str}) during pre-filter. Error: {e_complexity}. "
                    f"Including in backup attempt as complexity check is inconclusive."
                )
                eligible_items_for_backup.append((df_candidate, None)) # Include if estimation fails, with None for complexity

        log.info(f"After complexity filtering, {len(eligible_items_for_backup)} out of {initial_df_count} DataFrames are eligible for backup.")
    else:
        # If no thresholds are set, all are eligible (after type check)
        for df_candidate in dfs_and_complexities_to_process:
            if not isinstance(df_candidate, DataFrame): # pragma: no cover
                log.warning(f"Item '{str(df_candidate)[:100]}...' is not a DataFrame, skipping.")
                continue
            eligible_items_for_backup.append((df_candidate, None)) # No complexity calculated here
        log.info(f"No complexity thresholds set. {len(eligible_items_for_backup)} DataFrames (after type check) will be processed.")


    processed_dfs = [] # Keep track of DataFrames successfully backed up

    # Iterate over eligible items
    for df, prefilter_complexity_tuple in tqdm(eligible_items_for_backup, desc="Backing up Spark-cached DataFrames to DBFS", disable=not eligible_items_for_backup, unit="df"):
        current_df_hash_str_loop = "N/A"
        try:
            current_df_hash_str_loop = get_table_hash(df)
        except Exception: # pragma: no cover
            log.warning(f"Could not re-compute hash for DataFrame '{str(df)[:100]}...' inside backup loop. Using 'N/A'.")

        # Log complexity before backup attempt
        if prefilter_complexity_tuple:
            complexity_value_loop, raw_multiplier_loop, total_size_gb_loop = prefilter_complexity_tuple
            log.info(
                f"Processing DataFrame (hash: {current_df_hash_str_loop}) for backup. Pre-filtered complexity: "
                f"Value = {complexity_value_loop:.2f}, Raw Multiplier = {raw_multiplier_loop:.2f}, Input Size GB = {total_size_gb_loop:.2f}"
            )
        else: # Complexity not available from pre-filter (e.g. no thresholds, or pre-filter estimation failed)
            try:
                complexity_value_loop, raw_multiplier_loop, total_size_gb_loop = estimate_compute_complexity(df)
                log.info(
                    f"Processing DataFrame (hash: {current_df_hash_str_loop}) for backup. Estimated complexity (in-loop): "
                    f"Value = {complexity_value_loop:.2f}, Raw Multiplier = {raw_multiplier_loop:.2f}, Input Size GB = {total_size_gb_loop:.2f}"
                )
            except Exception as e_loop_complexity: # pragma: no cover
                log.warning(
                    f"Could not estimate complexity for DataFrame (hash: {current_df_hash_str_loop}) in backup loop. Error: {e_loop_complexity}. "
                    f"Proceeding with backup attempt."
                )
        # Backup logic
        try:
            log.info(
                f"Attempting to back up Spark-cached DataFrame (hash: {current_df_hash_str_loop}) to DBFS. "
                f"Spark cache status: useMemory={df.storageLevel.useMemory}, useDisk={df.storageLevel.useDisk}, "
                f"is_cached={df.is_cached}"
            )
            write_dbfs_cache(df, replace=False)
            processed_dfs.append(df)
            log.info(f"Successfully backed up DataFrame (hash: {current_df_hash_str_loop}) to DBFS.")

            if unpersist_after_backup and hasattr(df, "unpersist"): # Keep hasattr for safety
                log.info(f"Unpersisting DataFrame (hash: {current_df_hash_str_loop}) after backup...")
                df.unpersist()
        except Exception as e_backup:
            log.error(f"Error during DBFS backup for DataFrame (hash: {current_df_hash_str_loop}): {e_backup}", exc_info=True)

from .cache_management import (  # Import from cache_management
    clear_cache_for_hash,
    clear_caches_older_than,
    clear_inconsistent_cache,
    get_cached_dataframe_metadata,
    get_cached_tables,
    get_tables_from_database,
)
from .core_caching import (  # Import from core_caching
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
    "get_table_name_from_hash",
    "get_input_dir_mod_datetime", # Re-export
    "should_prefer_spark_cache", # Re-exported re-implementation
    "estimate_compute_complexity", # Re-export
    # Re-export DataFrame extension methods and related functions
    "cacheToDbfs",
    "clearDbfsCache",
    "__withCachedDisplay__",
    "extend_dataframe_methods",
]
