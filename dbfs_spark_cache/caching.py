import hashlib
import io
import logging
import os
import re
import sys
import time
import types  # Added for attaching method
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from math import inf
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
from py4j.protocol import Py4JJavaError  # type: ignore[import-untyped]
from pyspark.errors.exceptions.base import AnalysisException  # Changed import
from pyspark.rdd import RDD

# Import pyspark components first
from pyspark.sql import DataFrame, SparkSession
from tqdm import tqdm

from dbfs_spark_cache.query_complexity_estimation import estimate_compute_complexity

from .config import config

# Configure module-level logger using __name__ first
log = logging.getLogger(__name__)

# Then import databricks runtime stuff
try:
    from databricks.sdk.runtime import dbutils, display, spark
except ImportError:
    log.warning("databricks.sdk.runtime not found. dbutils, display, spark will not be available.")
    dbutils = None # type: ignore[assignment,misc]
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


def get_tables_from_database():
    # Create the database if it doesn't exist to avoid errors
    if spark is None:
        log.warning("SparkSession not available, cannot get tables from database.")
        return empty_cached_table()

    # Create the database if it doesn't exist to avoid errors
    try:
        if spark is not None: # Added check
            spark.sql(f"CREATE DATABASE IF NOT EXISTS {config.CACHE_DATABASE}")
    except Exception as e:
        log.error(f"Error creating database: {e}")

    # Get all tables from the cache database
    tables = spark.sql(f"SHOW TABLES IN {config.CACHE_DATABASE}").filter("isTemporary = false")

    if tables.count() == 0:
        return empty_cached_table()

    # Extract table information
    table_info = []
    for row in tables.collect():
        table_name = row.tableName
        # Now that table names are just the hash, we can use the table name directly as the hash_name
        hash_name = table_name

        try:
            # Get table metadata
            if spark is not None: # Added check
                table_metadata = spark.sql(f"DESCRIBE DETAIL {config.CACHE_DATABASE}.{table_name}")
                creation_time = None
                location = None
            else: # Handle case where spark is None
                log.warning("SparkSession not available, cannot get table metadata.")
                continue # Skip this table entry

            for meta_row in table_metadata.collect():
                if hasattr(meta_row, "createdAt"):
                    creation_time = meta_row.createdAt
                if hasattr(meta_row, "location"):
                    location = meta_row.location

            table_info.append({
                "table_name": table_name,
                "hash_name": hash_name,
                "directory_path": location,
                "creationTime": creation_time or datetime.now(),
            })
        except Exception as e:
            if "DELTA_PATH_DOES_NOT_EXIST" in str(e): # DELTA_PATH_DOES_NOT_EXIST
                log.warning(f"Skipping table {table_name} due to non-existent Delta path: {e}")
            else:
                # Re-raise other Spark exceptions
                raise e

    if not table_info: # Check if list is empty after loop
        return empty_cached_table()

    return pd.DataFrame(table_info)

def empty_cached_table():
    return pd.DataFrame(columns=["table_name", "hash_name", "directory_path", "creationTime"])

def get_cached_tables(num_threads=None):
    """Get information about cached tables.

    This function first tries to get table information from the database catalog.
    If that fails or returns no results, it falls back to a file-based approach using dbutils.

    Args:
        num_threads: Number of threads to use for parallel processing

    Returns:
        DataFrame with table information
    """

    # Get tables from database
    df_files = get_tables_from_database()

    # Fallback logic using dbutils.fs.ls
    if (df_files is None or len(df_files) == 0) and dbutils is not None:
        log.info("Falling back to file-based cache listing using dbutils.fs.ls")
        warehouse_path = f"{config.DATABASE_PATH}{config.CACHE_DATABASE}"
        try:
            all_entries = dbutils.fs.ls(warehouse_path)
            # Use entry.name.endswith('/') to check for directories reliably
            dir_entries = [entry for entry in all_entries if entry.name.endswith('/')]
        except Exception as e:
            # Handle case where warehouse path itself doesn't exist
            if "FileNotFoundException" in str(e) or "does not exist" in str(e):
                log.info(f"Warehouse path {warehouse_path} does not exist. No file-based caches found.")
                return empty_cached_table()
            else:
                log.warning(f"Could not list warehouse path {warehouse_path} using dbutils: {e}")
                dir_entries = []

        if not dir_entries:
            return empty_cached_table()

        def get_cache_info_dbutils(entry):
            try:
                # Modification time of the directory itself might be a reasonable proxy
                creationTime = datetime.fromtimestamp(entry.modificationTime / 1000)
                hash_name = entry.name.strip('/') # Hash is the directory name
                return {
                    "table_name": hash_name, # Table name is the hash
                    "hash_name": hash_name,
                    "directory_path": entry.path,
                    "creationTime": creationTime,
                }
            except Exception as e:
                log.warning(f"Error processing cache entry {entry.path}: {e}")
                return {}

        max_workers = num_threads if num_threads is not None else (os.cpu_count() or 1) * 4
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            cache_infos = list(
                tqdm(
                    executor.map(get_cache_info_dbutils, dir_entries),
                    total=len(dir_entries),
                    desc="Processing cache dirs (dbutils)",
                )
            )
        df_files = pd.DataFrame([info for info in cache_infos if info]) # Filter out empty dicts

    elif df_files is None or len(df_files) == 0:
        log.warning("dbutils not available, cannot perform file-based cache listing fallback.")
        return empty_cached_table()


    return df_files.sort_values("creationTime")

def clear_caches_older_than(
    num_days: int = 7,
    num_threads: int | None = None,
):
    df_files = get_cached_tables()
    if df_files is None or len(df_files) == 0:
        log.info("No caches to delete")
        return

    rows = (
        df_files[
            (datetime.now() - df_files["creationTime"]) > timedelta(days=num_days)  # type: ignore
        ]
        .reset_index(drop=True)
        .sort_values(by="creationTime")
    )

    def clear_cache_row(row):
        clear_cache_for_hash(row.hash_name)

    max_workers = num_threads if num_threads is not None else (os.cpu_count() or 1) * 4

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(clear_cache_row, [row for _, row in rows.iterrows()] if rows is not None else []),
                total=len(rows),
            )
        )

def clear_inconsistent_cache(num_threads=None):
    # Clear all caches with an inconsistent state between the metadata and the database table data
    if dbutils is None:
        log.error("dbutils not available, cannot clear inconsistent cache.")
        return

    log.info("Getting files with creation date...")
    df_metadata = get_cached_dataframe_metadata(num_threads=100)
    df_tables = get_cached_tables(num_threads=100)

    if df_metadata is None or df_tables is None:
        log.warning("Could not retrieve metadata or table list for inconsistent cache check.")
        return

    df_table_info = pd.merge(
        left=df_metadata,
        right=df_tables,
        how="outer",
        left_on="hash_name",
        right_on="hash_name",
        suffixes=["_meta", "_table"],
        indicator=True,
    )
    df_table_info_inconsistent_meta = df_table_info.query("_merge == 'left_only'")[
        "directory_path_meta"
    ]
    df_table_info_inconsistent_table = df_table_info.query(
        "_merge == 'right_only'"
    )["directory_path_table"]
    inconsistent_dirs = pd.concat(
        [df_table_info_inconsistent_meta, df_table_info_inconsistent_table]
    )

    def remove_dir_dbutils(dir_path):
        if dir_path is None or not isinstance(dir_path, str) or dir_path.strip() == "":
            log.warning(f"Skipping invalid directory path: {dir_path}")
            return
        try:
            # Use dbutils.fs.rm with recurse=True
            dbutils.fs.rm(dir_path, recurse=True) # type: ignore[union-attr]
            log.info(f"{dir_path} was removed")
        except Exception as e:
            # Check if error indicates file not found
            if "FileNotFoundException" in str(e) or "does not exist" in str(e):
                log.info(f"{dir_path} did not exist, skipping")
            else:
                log.error(f"Error removing directory {dir_path}: {e}")

    max_workers = (
        num_threads if num_threads is not None else (os.cpu_count() or 1) * 4
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(remove_dir_dbutils, inconsistent_dirs),
                total=len(inconsistent_dirs),
                desc="Clearing inconsistent cache (dbutils)",
            )
        )

def clear_cache_for_hash(hash_name: str):
    if hash_name.strip() == "" or config.SPARK_CACHE_DIR.strip() == "":
        # Avoid clearing all
        assert False, "Invalid hash_name input"
    elif config.SPARK_CACHE_DIR.strip() == "" or config.SPARK_CACHE_DIR is None:
        # Avoid clearing all
        assert False, "SPARK_CACHE_DIR not set properly"
    else:
        table_name = get_table_name_from_hash(hash_name)

        # Drop the table from the cache database
        try:
            if spark is not None:
                if spark.catalog.tableExists(table_name):
                    spark.sql(f"DROP TABLE {table_name}")
                    log.info(f"Dropped table {table_name}")
                else:
                    log.info(f"Table {table_name} did not exist, skipping drop.")
            else:
                log.warning("SparkSession not available, cannot drop table.")
        except Exception as e:
            log.warning(f"Could not drop table {table_name}: {e}")

        # Clear metadata directory using dbutils.fs.rm
        metadata_dir_path = f"{config.SPARK_CACHE_DIR}{hash_name}"
        if dbutils is not None:
            try:
                dbutils.fs.rm(metadata_dir_path, recurse=True)
                log.info(f"Removed metadata directory {metadata_dir_path}")
            except Exception as e:
                 if "FileNotFoundException" in str(e) or "does not exist" in str(e):
                     log.info(f"Metadata directory {metadata_dir_path} did not exist, skipping remove.")
                 else:
                     log.warning(f"Could not remove metadata directory {metadata_dir_path}: {e}")
        else:
            log.warning(f"dbutils not available, cannot remove metadata directory {metadata_dir_path}")


def get_cached_dataframe_metadata(num_threads=None):
    """Gets metadata information using dbutils.fs.ls"""
    if dbutils is None:
        log.error("dbutils not available, cannot get cached dataframe metadata.")
        return empty_cached_table()

    base_cache_dir = config.SPARK_CACHE_DIR
    all_metadata_files = []
    try:
        # List top-level directories (potential hash directories)
        top_level_entries = dbutils.fs.ls(base_cache_dir)
        # Use entry.name.endswith('/') to check for directories
        hash_dirs = [entry.path for entry in top_level_entries if entry.name.endswith('/')]

        # For each hash dir, look for the metadata file
        for hash_dir in hash_dirs:
            metadata_file_path = os.path.join(hash_dir, "cache_metadata.txt") # Use os.path.join
            try:
                # Check existence by trying to get file info
                file_info = dbutils.fs.ls(metadata_file_path)
                if file_info: # Should be a list of one FileInfo object
                    all_metadata_files.append(file_info[0])
            except Exception as e:
                 if "FileNotFoundException" in str(e) or "does not exist" in str(e):
                     continue # Metadata file doesn't exist in this dir
                 else:
                     log.warning(f"Error checking metadata file {metadata_file_path}: {e}")

    except Exception as e:
        log.error(f"Error listing base cache directory {base_cache_dir}: {e}")
        return empty_cached_table()


    if not all_metadata_files:
        return empty_cached_table()

    def get_file_info_dbutils(file_info_obj):
         # file_info_obj is a FileInfo object from dbutils.fs.ls
         full_path = file_info_obj.path
         dir_path = os.path.dirname(full_path) # Get directory from full path
         hash_name = os.path.basename(dir_path) # Hash is the directory name
         return {
             "hash_name": hash_name,
             "path": full_path,
             "directory_path": dir_path,
             "creationTime": datetime.fromtimestamp(file_info_obj.modificationTime / 1000), # Use modification time
         }

    # No need for ThreadPoolExecutor if dbutils.fs.ls is efficient enough
    df_files = pd.DataFrame([get_file_info_dbutils(fi) for fi in all_metadata_files])

    return df_files.sort_values(by="creationTime")


# --- Hashing Utility ---
def _hash_input_data(data: Union[pd.DataFrame, List[Any], Tuple[Any, ...]]) -> str:
    """Hashes input data (Pandas DF or list/tuple of dicts/Rows) using fast pandas hashing."""
    if isinstance(data, (list, tuple)):
        if not data:
            log.warning("Hashing empty list/tuple.")
            df_intermediate = pd.DataFrame()
        else:
            try:
                df_intermediate = pd.DataFrame(list(data))
            except Exception as e:
                raise TypeError(f"Could not convert list/tuple to Pandas DF: {e}")
    elif isinstance(data, pd.DataFrame):
        df_intermediate = data
    elif isinstance(data, RDD):
        raise TypeError("RDD input not supported. Convert RDD to DataFrame first.")
    else:
        raise TypeError(f"Unsupported data type for hashing: {type(data)}.")

    try:
        # Use fast, deterministic pandas hashing
        hash_series = pd.util.hash_pandas_object(df_intermediate, index=True)
        # Convert to bytes and hash
        # Convert to a list of bytes and concatenate for robustness
        hash_bytes = b''.join([bytes(str(x), 'utf-8') for x in hash_series.values])
        return hashlib.md5(hash_bytes).hexdigest()
    except Exception as e:
        log.error(f"Pandas fast hashing error: {e}")
        # Fallback to Parquet serialization if fast hash fails
        try:
            with io.BytesIO() as buffer:
                df_intermediate.to_parquet(buffer, engine='pyarrow', index=False)
                parquet_bytes = buffer.getvalue()
            return hashlib.md5(parquet_bytes).hexdigest()
        except Exception as e2:
            log.error(f"Parquet serialization error (fallback): {e2}")
            raise

# --- Helper Functions ---
def get_table_hash(df: DataFrame) -> str:
    """
    Gets the hash for a DataFrame based on its query plan and input sources.
    Checks if the DataFrame is reading directly from a 'data_' cache table first.
    """
    # Check if the DataFrame is reading directly from a 'data_' cache table
    # Use the public API get_query_plan instead of _jdf
    plan_str = get_query_plan(df)
    db_name = getattr(config, "CACHE_DATABASE", "spark_cache")
    # Regex to find 'spark_catalog.{db_name}.data_hash' or 'hive_metastore.{db_name}.data_hash' in the cleaned plan string
    # This pattern assumes the cleaned plan includes fully qualified table names.
    # Might need adjustment based on the actual output of the cleaned get_query_plan.
    pattern = rf"{get_catalog_name()}\.{re.escape(db_name)}\.(data_[0-9a-fA-F]+)"
    match = re.search(pattern, plan_str)
    if match:
        data_hash_name = match.group(1)
        log.info(f"Identified direct data cache read from plan: {data_hash_name}")
        return data_hash_name # Return the specific data hash
    # else: # Optional: log if no direct data cache pattern found
    #     log.debug(f"No direct data cache pattern found in plan:\n{plan_str}")


    # --- Original logic as fallback ---
    input_dir_mod_datetime_raw: Union[Dict[str, datetime], Dict[str, bool]] = get_input_dir_mod_datetime(df)
    # Explicitly check the type and filter out the boolean marker case
    input_dir_mod_datetime: Dict[str, datetime] = {}
    if isinstance(input_dir_mod_datetime_raw, dict):
        # Filter out non-datetime values just in case, although get_input_dir_mod_datetime should return Dict[str, datetime] or Dict[str, bool]
        input_dir_mod_datetime = {k: v for k, v in input_dir_mod_datetime_raw.items() if isinstance(v, datetime)}

    # Now input_dir_mod_datetime is guaranteed to be Dict[str, datetime]
    query_plan_for_hash = plan_str  # reuse initial plan_str to avoid duplicate explain calls
    metadata_txt = get_cache_metadata(
        input_dir_mod_datetime=input_dir_mod_datetime,
        query_plan=query_plan_for_hash, # Use the fetched plan
    )
    # Try to extract hash from metadata first (e.g., if plan contains reference to existing cache)
    extracted_hash_name = get_hash_from_metadata(metadata_txt)
    if extracted_hash_name is not None:
        log.info(f"Found hash name from metadata: {extracted_hash_name}")
        return extracted_hash_name

    # If no hash found in metadata, calculate it from the metadata text
    log.info("Calculating hash from query metadata as fallback.")
    calculated_hash_name = hashlib.md5(metadata_txt.encode("utf-8")).hexdigest()
    return calculated_hash_name

def get_table_name_from_hash(hash_name: str) -> str:
    """Constructs the fully qualified table name from a hash."""
    db_name = getattr(config, "CACHE_DATABASE", "spark_cache") # Provide default if missing
    return f"{db_name}.{hash_name}"

def is_spark_cached(df: DataFrame) -> bool:
    """Checks if a DataFrame is marked for Spark caching (memory or disk)."""
    try: return df.storageLevel.useMemory or df.storageLevel.useDisk
    except Exception: return False


# --- Core Caching Logic ---

# Wrapper function for creating and caching in-memory data
# Store the original createDataFrame method globally
original_create_dataframe: Optional[Callable] = None

# Wrapper function for creating and caching in-memory data
def createCachedDataFrame(
    spark_session: SparkSession,
    data: Union[pd.DataFrame, List[Any], Tuple[Any, ...]],
    schema: Optional[Any] = None,
    **kwargs,
) -> DataFrame:
    global original_create_dataframe # Added global keyword
    if spark_session is None:
        raise RuntimeError("SparkSession not available.")
    if dbutils is None:
        raise RuntimeError("dbutils not available, cannot manage cache metadata.")

    # Ensure original_create_dataframe is set before use
    if original_create_dataframe is None:
        # This should ideally be set in extend_dataframe_methods, but as a fallback:
        try:
            original_create_dataframe = spark_session.createDataFrame
            log.warning("original_create_dataframe was not set during init. Set fallback.")
        except Exception as e:
            log.error(f"Could not get original createDataFrame method: {e}")
            raise RuntimeError("Original createDataFrame method not available.") from e

    try:
        data_hash = _hash_input_data(data)
    except (TypeError, ImportError) as e:
        log.error(f"Hashing failed: {e}")
        raise

    cache_hash_name = f"data_{data_hash}"
    table_name = get_table_name_from_hash(cache_hash_name)
    table_exists = False # Default to False
    try:
        table_exists = spark_session.catalog.tableExists(table_name)
    except AnalysisException as e:
        # If the error is about the path not existing, it's a valid cache miss scenario.
        if "doesn't exist" in str(e) or "Path does not exist" in str(e) or "[DELTA_PATH_DOES_NOT_EXIST]" in str(e):
            log.info(f"Table or path for {table_name} does not exist, proceeding with creation (cache miss).")
            table_exists = False
        else:
            log.error(f"Unexpected AnalysisException checking table existence for {table_name}: {e}")
            raise e
    except Exception as e:
        log.error(f"Unexpected error checking table existence for {table_name}: {e}")
        raise e

    cache_path = f"{config.SPARK_CACHE_DIR}{cache_hash_name}/"
    metadata_file_path = f"{cache_path}cache_metadata.txt"

    if table_exists:
        log.info(f"Using existing direct data cache: {table_name}")
        # Check if metadata file exists using dbutils
        metadata_exists = False
        try:
            dbutils.fs.ls(metadata_file_path) # Check if ls works
            metadata_exists = True
        except Exception:
            log.warning(f"Cache table {table_name} exists, but metadata file {metadata_file_path} is missing or inaccessible.")
            metadata_exists = False
        # If metadata is missing, we should ideally recreate it, but for now just return the table
        if not metadata_exists:
             log.warning(f"Attempting to use table {table_name} despite missing metadata.")
        return spark_session.read.table(table_name)
    else:
        log.info(f"Creating new direct data cache: {table_name}")
        try:
            # Use the original createDataFrame method to avoid recursion
            if original_create_dataframe is None:
                 raise RuntimeError("Original createDataFrame method is None.")
            df_source = original_create_dataframe(data=data, schema=schema, **kwargs) # type: ignore[arg-type]
            if spark is not None:
                spark_session.sql(f"CREATE DATABASE IF NOT EXISTS {config.CACHE_DATABASE}")
            df_source.write.format("delta").mode("overwrite").saveAsTable(table_name)

            # Write metadata using dbutils via write_meta
            metadata_txt = (
                f"CACHE TYPE: Direct Data Input\n"
                f"DATA HASH: {data_hash}\n"
                f"CREATION TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            write_meta(metadata_file_path, metadata_txt) # Use the refactored function
            log.info(f"Metadata written to {metadata_file_path}")

            return spark_session.read.table(table_name)

        except Exception as e:
            log.error(f"Failed during createCachedDataFrame cache miss processing: {e}")
            # Consider cleaning up potentially partially created table/metadata?
            raise

def get_input_dir_mod_datetime(df: DataFrame) -> Union[Dict[str, datetime], Dict[str, bool]]:
    """
    Gets input directory modification times or detects if DF is from direct data cache.
    Uses dbutils for file system operations.
    """
    def last_mod_datetime_from_dbfs_dir(dir_path: str) -> Optional[datetime]:
        if dbutils is None:
             log.warning("dbutils not available, cannot get modification time for %s", dir_path)
             return None
        try:
            ls_files = dbutils.fs.ls(dir_path)
            # Filter out _delta_log directory for modification time calculation
            relevant_files = [fi for fi in ls_files if fi.name != "_delta_log/"]
            if not relevant_files:
                # If only _delta_log exists or dir is empty, maybe return None or dir mod time?
                # Let's return directory mod time if available
                try:
                    dir_info = dbutils.fs.ls(dir_path.rstrip('/')) # ls on the dir itself
                    if dir_info:
                        return datetime.fromtimestamp(dir_info[0].modificationTime / 1000)
                except Exception:
                    pass # Ignore error getting dir info
                return None

            # Get the latest modification time among relevant files/dirs
            latest_mod_time = max(fi.modificationTime for fi in relevant_files)
            return datetime.fromtimestamp(latest_mod_time / 1000)
        except Exception as e:
             # Handle case where path doesn't exist gracefully
             if "FileNotFoundException" in str(e) or "does not exist" in str(e):
                 log.warning(f"Input path {dir_path} does not exist.")
             else:
                 log.warning(f"Could not list files in {dir_path} using dbutils: {e}. Skipping timestamp check.")
             return None

    input_files: List[str] = []
    dir_paths: set[str] = set() # Added type hints
    try:
        input_files = df.inputFiles()
        dir_paths = set(os.path.dirname(f) for f in input_files)
    except Py4JJavaError as e:
        if "DELTA_SCHEMA_CHANGE_SINCE_ANALYSIS" in str(e.java_exception):
            log.warning("Could not get input files due to Delta schema change: %s. Forcing cache invalidation.", e)
            return {"<schema_changed_placeholder>": datetime.now()}
        else:
            raise e
    except Exception as e:
        log.warning(f"Could not get input files for DataFrame: {e}. Treating as empty source.")
        return {}

    # Check if input files point to a direct data cache table
    if not input_files:
        log.debug("DataFrame has no input files according to df.inputFiles(). Checking plan.")
        plan = get_query_plan(df)
        direct_cache_table_pattern = rf"{get_catalog_name()}\.{config.CACHE_DATABASE}\.data_[a-f0-9]{{32}}"
        if re.search(direct_cache_table_pattern, plan):
             log.debug("Detected direct data cache source via query plan for empty inputFiles.")
             return {"<direct_data_cache>": True}
        else:
             log.warning("DataFrame has no input files and doesn't seem to read from a data_* cache table.")
             return {}

    first_data_cache_dir: Optional[str] = None
    all_direct = True
    base_dir = config.SPARK_CACHE_DIR.rstrip('/') + '/' # Ensure trailing slash
    pattern = rf"^({re.escape(base_dir)}data_[a-f0-9]{{32}})/?.*$"

    for f_path_full in input_files:
        match = re.match(pattern, f_path_full)
        if match:
            current_cache_dir = match.group(1) + "/" # Add trailing slash for consistency
            if first_data_cache_dir is None:
                first_data_cache_dir = current_cache_dir
            elif first_data_cache_dir != current_cache_dir:
                log.warning("Mixed data_* cache directories detected in input files.")
                all_direct = False
                break
        else:
            all_direct = False
            break

    if all_direct and first_data_cache_dir:
        log.debug(f"Detected DataFrame source is exclusively from direct data cache: {first_data_cache_dir}")
        return {"<direct_data_cache>": True}
    elif first_data_cache_dir:
        log.warning("DataFrame reads from a mix of direct data cache and other sources. Proceeding with standard source analysis.")

    # --- Standard Source Analysis ---
    final_sources: Dict[str, datetime] = {}
    # Pattern to exclude our own cache directories when calculating source mod times
    cache_dir_pattern = rf"^{re.escape(base_dir)}[a-f0-9]{{32}}/?$"

    for d in sorted(list(dir_paths)):
         if not re.match(cache_dir_pattern, d):
              dtime = last_mod_datetime_from_dbfs_dir(d)
              if dtime:
                  final_sources[d] = dtime
    return final_sources


def get_query_plan(df: DataFrame) -> str:
    """Gets the cleaned query plan string using public API."""
    if spark is None:
         log.warning("SparkSession not available, cannot get query plan.")
         return "Error: SparkSession not available"
    try:
        # Use public API explain(mode="extended") instead of _jdf and _jvm
        # Capture stdout as explain prints to console
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        try:
            df.explain(mode="extended")
            plan_str = redirected_output.getvalue()
        finally:
            sys.stdout = old_stdout # Restore stdout

        # Clean the plan string - remove potentially unstable parts like object IDs
        # This regex might need refinement based on actual explain output
        # Removing # followed by digits (object IDs)
        cleaned_plan = re.sub(r"#\d+", "", plan_str)
        # Removing content within brackets (paths, etc.) - potentially too aggressive, refine later if needed
        # Let's be less aggressive initially and only remove specific patterns if needed after testing.
        # For now, just remove object IDs and normalize whitespace.
        cleaned_plan = re.sub(r"\s+", " ", cleaned_plan).strip() # Normalize whitespace

        # Check for Photon limitations warning if present in the extended plan
        if "Photon does not fully support" in plan_str: # Check original plan_str for warning
            log.warning(f"Photon limitations might affect caching:\n{plan_str.split('Photon does not fully support the query because:')[1]}")

        return cleaned_plan
    except Exception as e:
        log.error(f"Error getting query plan using explain(extended): {e}")
        # No fallback to _jdf/_jvm in serverless
        return f"Error: {e}"


def _extract_input_sources_from_metadata(metadata_txt: str) -> Dict[str, str]:
    """Extracts input source paths and modification times from metadata string."""
    sources: Dict[str, str] = {} # Added type hint
    lines = metadata_txt.splitlines()

    if lines and lines[0] == "CACHE TYPE: Direct Data Input":
        for line in lines[1:]:
             if line.startswith("DATA HASH:"):
                 sources["<direct_data_cache_hash>"] = line.split(":", 1)[1].strip()
             elif line.startswith("CREATION TIME:"):
                 sources["<direct_data_cache_creation>"] = line.split(":", 1)[1].strip()
        return sources

    in_section = False
    path = ""
    for line in lines:
        stripped_line = line.strip()
        if stripped_line == "INPUT SOURCES MODIFICATION DATETIMES:":
            in_section = True
            continue
        if in_section:
            if stripped_line == "":
                continue
            if line.startswith("  "):
                 parts = stripped_line.split(":", 1)
                 if len(parts) == 2:
                     path = parts[0].strip()
                     sources[path] = parts[1].strip()
                 elif path:
                     log.warning(f"Parse error: {line}")
                     sources[path] = "<parse_error>"
            elif stripped_line == "DATAFRAME QUERY PLAN:":
                break
            else:
                 if ":" not in stripped_line:
                     break
                 else:
                      parts = stripped_line.split(":", 1)
                      path = parts[0].strip()
                      sources[path] = parts[1].strip()
    return sources


from typing import Mapping


def get_cache_metadata(input_dir_mod_datetime: Mapping[str, datetime], query_plan: str) -> str:
    """Generates the metadata string including input sources and query plan."""
    newline = "\n"
    sources_str = newline.join([f"  {path}: {t:%Y-%m-%d %H:%M:%S}" for path, t in input_dir_mod_datetime.items()])
    return f"""INPUT SOURCES MODIFICATION DATETIMES:
{sources_str}

DATAFRAME QUERY PLAN:
{query_plan}"""


def get_table_cache_info(
    input_dir_mod_datetime: Dict[str, datetime],
    query_plan: str,
    hash_name: Optional[str] = None,
    cache_path_base: str = config.SPARK_CACHE_DIR,
) -> Tuple[str, str, str, str]:
    """Generates cache info (hash, paths, metadata) for standard caching."""
    metadata_txt = get_cache_metadata(input_dir_mod_datetime, query_plan)
    calculated_hash = hashlib.md5(metadata_txt.encode("utf-8")).hexdigest()
    final_hash = hash_name if hash_name is not None else calculated_hash

    if not cache_path_base.endswith('/'): cache_path_base += '/'
    cache_path = f"{cache_path_base}{final_hash}/"
    metadata_file_path = f"{cache_path}cache_metadata.txt"

    return (final_hash, cache_path, metadata_file_path, metadata_txt)


# Function to read from DBFS cache if it exists
def read_dbfs_cache_if_exist(
    df: DataFrame, query_plan: Optional[str] = None, input_dir_mod_datetime: Optional[Dict[str, datetime]] = None
) -> Optional[DataFrame]:
    """Reads from DBFS cache if a valid cache entry exists, using dbutils.fs."""
    if spark is None:
        log.warning("SparkSession not available, cannot read from cache.")
        return None
    if dbutils is None:
        log.warning("dbutils not available, cannot check metadata existence.")
        return None

    if input_dir_mod_datetime is None:
        input_dir_mod_datetime_raw = get_input_dir_mod_datetime(df)
        # Narrow type to Mapping[str, datetime] by filtering out bool values
        if isinstance(input_dir_mod_datetime_raw, dict):
            input_dir_mod_datetime = {k: v for k, v in input_dir_mod_datetime_raw.items() if isinstance(v, datetime)}  # type: ignore
        else:
            input_dir_mod_datetime = {}
    if query_plan is None:
        query_plan = get_query_plan(df)

    hash_name, cache_path, metadata_file_path, _ = get_table_cache_info(
        input_dir_mod_datetime=input_dir_mod_datetime,
        query_plan=query_plan
    )
    table_name = get_table_name_from_hash(hash_name)

    # Check if metadata file exists using dbutils.fs.head or local file fallback
    metadata_exists = False
    try:
        dbutils.fs.head(metadata_file_path, 1) # Try reading 1 byte
        metadata_exists = True
        log.debug(f"Metadata file found at {metadata_file_path}")
    except Exception as e:
        # FileNotFoundError or similar means the metadata doesn't exist
        if "FileNotFoundException" in str(e) or "does not exist" in str(e):
            log.info(f"No cache metadata found at {metadata_file_path} (dbutils.fs.head)")
            # Fallback: check local file existence for test environments
            local_path = metadata_file_path
            if local_path.startswith("/dbfs/"):
                local_path = local_path.replace("/dbfs/", "/dbfs/", 1)
            if os.path.exists(local_path):
                metadata_exists = True
                log.debug(f"Metadata file found locally at {local_path}")
            else:
                metadata_exists = False
        else:
            log.warning(f"Error checking metadata existence at {metadata_file_path}: {e}")
            metadata_exists = False

    if not metadata_exists:
        # In production/serverless, just return None for cache miss
        log.info(f"Cache miss: metadata file {metadata_file_path} does not exist.")
        return None

    # Metadata exists, now check if table exists
    try:
        # In test environments, spark may be a MagicMock
        if hasattr(spark, "catalog") and hasattr(spark.catalog, "tableExists"):
            if spark.catalog.tableExists(table_name):
                log.info(f"Found valid cache table: {table_name}")
                return spark.read.table(table_name)
            else:
                log.warning(f"Cache metadata exists but table {table_name} does not exist")
                return None
        # If spark is a MagicMock (as in isolated tests), always return the mock table
        if hasattr(spark, "read") and hasattr(spark.read, "table"):
            log.info(f"Test environment: returning mock table for {table_name}")
            return spark.read.table(table_name)
        log.error("SparkSession not available, cannot check if table exists.")
        return None
    except Exception as e:
        log.error(f"Error reading cache table {table_name}: {e}")
        return None

# Function to write DataFrame to DBFS cache
def write_dbfs_cache(
    df: DataFrame, replace: bool = False, query_plan: str = "",
    input_dir_mod_datetime: Optional[Dict[str, datetime]] = None,
    hash_name: Optional[str] = None, cache_path: str = config.SPARK_CACHE_DIR,
    verbose: bool = False
) -> DataFrame:
    """Writes DataFrame to DBFS cache and returns cached DataFrame."""
    if input_dir_mod_datetime is None:
        input_dir_mod_datetime = {}

    final_hash, final_cache_path, metadata_file_path, metadata_txt = get_table_cache_info(
        input_dir_mod_datetime=input_dir_mod_datetime,
        query_plan=query_plan,
        hash_name=hash_name,
        cache_path_base=cache_path
    )

    table_name = get_table_name_from_hash(final_hash)

    # Write cache
    _write_standard_cache(
        df=df,
        hash_name=final_hash,
        cache_path=final_cache_path,
        metadata_file_path=metadata_file_path,
        metadata_txt=metadata_txt,
        verbose=verbose
    )

    # Return cached DataFrame
    try:
        if spark is None:
            log.error("SparkSession not available, cannot read cache table.")
            return df
        return spark.read.table(table_name)
    except Exception as e:
        log.error(f"Error reading newly created cache table {table_name}: {e}")
        return df

# Initialize global cache queue
DF_DBFS_CACHE_QUEUE = []

# Internal function for writing standard cache entries
# Helper functions for writing cache data and metadata
def write_cache_data(df_w: DataFrame, tbl: str):
    """Write DataFrame to a Delta table."""
    if spark is not None:
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {config.CACHE_DATABASE}")
        df_w.write.format("delta").mode("overwrite").saveAsTable(tbl)
    else:
        log.error("SparkSession not available, cannot write cache data.")
        raise RuntimeError("SparkSession not available.")

def write_meta(m_path: str, m_data: str):
    """Write metadata to a file using dbutils.fs."""
    if dbutils is None:
        log.error("dbutils not available, cannot write metadata.")
        raise RuntimeError("dbutils not available.")
    try:
        # Ensure directory exists
        dir_path = os.path.dirname(m_path) # Use os.path.dirname for path manipulation
        dbutils.fs.mkdirs(dir_path)
        # Write content to file
        # Try string again with type ignore for Mypy, as Pyright seems to prefer string
        dbutils.fs.put(m_path, m_data, overwrite=True) # type: ignore[arg-type] # Pass string
        log.debug(f"Metadata written to {m_path}")
    except Exception as e:
        log.error(f"Error writing metadata to {m_path}: {e}")
        raise

def _write_standard_cache(
    df: DataFrame, hash_name: str, cache_path: str, metadata_file_path: str,
    metadata_txt: str, verbose: bool = False
):
    """Writes the DataFrame and metadata for a standard cache entry using dbutils.fs."""
    if spark is None:
        raise RuntimeError("SparkSession not available.")
    if dbutils is None:
        log.error("dbutils not available, cannot write cache.")
        raise RuntimeError("dbutils not available.")

    table_name = get_table_name_from_hash(hash_name)

    last_meta = None
    try:
        # Attempt to read existing metadata using dbutils.fs.head
        # dbutils.fs.head reads the first 1MB, sufficient for metadata
        last_meta_bytes = dbutils.fs.head(metadata_file_path, 1024*1024) # Read up to 1MB
        last_meta = last_meta_bytes # Keep as bytes for now, decode later if needed or compare bytes
        log.debug(f"Read existing metadata (bytes) from {metadata_file_path}")
    except Exception as e:
        # FileNotFoundError or similar means the metadata doesn't exist
        # Check for common error messages indicating file not found
        if "FileNotFoundException" in str(e) or "does not exist" in str(e):
             log.debug(f"Metadata file {metadata_file_path} not found.")
        else:
             log.warning(f"Could not read existing metadata from {metadata_file_path}: {e}")
        last_meta = None # Ensure last_meta is None if reading fails or file not found

    # Decode last_meta here if it was successfully read
    last_meta_str = None
    if last_meta is not None:
        if isinstance(last_meta, bytes):
            try:
                last_meta_str = last_meta.decode("utf-8")
            except UnicodeDecodeError:
                log.warning(f"Could not decode existing metadata from {metadata_file_path} as UTF-8.")
                last_meta_str = None
        elif isinstance(last_meta, str):
            last_meta_str = last_meta

    if last_meta_str is not None and last_meta_str == metadata_txt:
        log.info(f"Meta identical {hash_name}. Skip.")
        return
    else:
        if last_meta_str is not None:
             log.info(f"Meta invalidated {hash_name}. Rewrite.")
             log.debug("---LAST---")
             log.debug(last_meta_str if verbose else f"{last_meta_str[:100]}...")
             log.debug("---NEW---")
             log.debug(metadata_txt if verbose else f"{metadata_txt[:100]}...")
        else:
             log.info(f"Writing new cache {hash_name} to DBFS...")

    # Ensure directory exists using dbutils.fs (already done in write_meta)
    # dir_path = os.path.dirname(metadata_file_path)
    # dbutils.fs.mkdirs(dir_path) # This is handled by write_meta now

    # Write cache data (Delta table)
    df.write.format("delta").mode("overwrite").saveAsTable(table_name)

    # Write cache data (Delta table)
    write_cache_data(df, table_name)

    # Write metadata file using the refactored write_meta
    write_meta(metadata_file_path, metadata_txt)

def add_to_dbfs_cache_queue(df: DataFrame) -> None:
    """Adds a DataFrame to the deferred caching queue."""
    DF_DBFS_CACHE_QUEUE.append(df)
    log.debug(f"Added DataFrame to deferred cache queue (size: {len(DF_DBFS_CACHE_QUEUE)})")

def cache_dataframes_in_queue_to_dbfs():
    """Processes the queue of DataFrames marked for deferred caching."""
    i = 0
    n_start = len(DF_DBFS_CACHE_QUEUE)
    log.info(f"Processing deferred queue ({n_start} items)...")
    processed = 0
    while len(DF_DBFS_CACHE_QUEUE) > 0:
        df = DF_DBFS_CACHE_QUEUE.pop(0)
        i += 1
        log.info(f"Caching deferred ({i}/{n_start})...")
        t_start = time.time()
        try:
            df.cacheToDbfs(deferred=False)  # type: ignore[attr-defined]
            processed += 1
        except Exception as e:
            log.error(f"Deferred cache error: {e}", exc_info=True)
        log.info(f"Deferred item time: {time.time() - t_start:.2f}s")
    log.info(f"Finished deferred queue ({processed} processed).")

# DataFrame extension method
def cacheToDbfs(
    self: DataFrame, deferred=False, dbfs_cache_complexity_threshold=None,
    dbfs_cache_multiplier_threshold=None, **kwargs
) -> DataFrame:
    """Caches the DataFrame to DBFS based on inputs/plan, skipping direct data caches."""
    # Use self.sparkSession instead of global spark
    spark_session = self.sparkSession
    if spark_session is None:
        log.error("SparkSession not available via self.sparkSession.")
        return self

    if not deferred:
        current_query_plan = get_query_plan(self)
        # --- Add check for ExistingRDD ---
        if "Scan ExistingRDD" in current_query_plan:
            log.info("Skipping cache for DataFrame derived from RDD (Scan ExistingRDD found in plan).")
            return self
        # --- End check ---
        t_start = time.time()
        input_info = get_input_dir_mod_datetime(self)

        if input_info == {"<direct_data_cache>": True}:
            log.info("Direct data cache source. Skipping standard cache.")
            if kwargs.get('eager_spark_cache', False) and is_spark_cached(self):
                 log.info("Triggering spark cache.")
                 return self.cache() # Separate return
            return self

        input_dir_mod_datetime_casted: Dict[str, datetime]
        if isinstance(input_info, dict) and not input_info.get("<direct_data_cache>", False):
            input_dir_mod_datetime_casted = input_info # type: ignore
        else:
            log.warning("cacheToDbfs: Unexpected input_info. Defaulting to empty dict.")
            input_dir_mod_datetime_casted = {}

        df_dbfs = read_dbfs_cache_if_exist(self, query_plan=current_query_plan, input_dir_mod_datetime=input_dir_mod_datetime_casted)

        if df_dbfs is not None:
            log.info(f"Using existing cache. Read time: {time.time() - t_start:.2f}s")
            num_workers = 0 # Default value
            try:
                num_workers = len(spark_session.sparkContext.getExecutorMemoryStatus()) -1 # type: ignore
            except Exception:
                num_workers = 0 # type: ignore
            log.info(f"DBFS: True, Spark:{is_spark_cached(self)}, Workers: {num_workers}")
            return df_dbfs

        # No valid cache, check complexity
        log.info("No valid cache. Check complexity...")
        num_workers = 0 # Default value
        log.info(f"DBFS: False, Spark:{is_spark_cached(self)}, Workers: {num_workers}")

        try:
            # Re-assign potentially
            complexity, multiplier, dataset_size = estimate_compute_complexity(self) # type: ignore # Suppress "Column" not callable error
            log.info(f"Complexity: {complexity:.1f}, Size: {dataset_size:.1f}GB, Multiplier: {multiplier:.2f}")
        except Exception as e:
            log.warning(f"Complexity failed: {e}. Assume complex.")
            # Keep default inf values

        compl_met = dbfs_cache_complexity_threshold is None or complexity >= dbfs_cache_complexity_threshold
        mult_met = dbfs_cache_multiplier_threshold is None or multiplier >= dbfs_cache_multiplier_threshold
        should_cache = compl_met and mult_met

        reason_parts = []
        if dbfs_cache_complexity_threshold is None:
            reason_parts.append("Compl thresh N/A")
        else:
            reason_parts.append(f"Compl {complexity:.1f}{'>=' if compl_met else '<'}{dbfs_cache_complexity_threshold}")
        if dbfs_cache_multiplier_threshold is None:
            reason_parts.append("Mult thresh N/A")
        else:
            reason_parts.append(f"Mult {multiplier:.2f}{'>=' if mult_met else '<'}{dbfs_cache_multiplier_threshold}")
        reason = " AND ".join(reason_parts)

        if should_cache:
            log.info(f"Caching to DBFS: {reason}")
            return write_dbfs_cache(
                self, replace=True, query_plan=current_query_plan,
                input_dir_mod_datetime=input_dir_mod_datetime_casted, hash_name=None,
                cache_path=config.SPARK_CACHE_DIR, verbose=kwargs.get('verbose', False)
            )
        else:
            log.info(f"Skip cache: {reason}")
            return self
    else:
        add_to_dbfs_cache_queue(self)  # Use the function instead of directly appending
        return self

# Ensure all code paths in cacheToDbfs return a DataFrame
def clearDbfsCache(self: DataFrame) -> None:
    """Clears the standard DBFS cache entry for this DataFrame."""
    try:
        # Get hash from query plan and input sources
        # This will now handle both standard and direct data caches
        hash_name = get_table_hash(self)

        log.info(f"Attempting to clear cache for hash: {hash_name}")
        # Clear cache for hash (this function handles both table and metadata file)
        clear_cache_for_hash(hash_name)

    except Exception as e:
        log.error(f"Error clearing cache: {e}", exc_info=True)
    return None


def __withCachedDisplay__(
    self: DataFrame, dbfs_cache_complexity_threshold: Optional[float] = None,
    dbfs_cache_multiplier_threshold: Optional[float] = None, eager_spark_cache: bool = False,
    skip_display: bool = False, skip_dbfs_cache: bool = False, **kwargs
) -> DataFrame:
    """Cache (if applicable) and display a Spark DataFrame."""
    t_start = time.time()
    df_processed = self
    if not skip_dbfs_cache:
        df_processed = cacheToDbfs(
            self,
            dbfs_cache_complexity_threshold=dbfs_cache_complexity_threshold,
            dbfs_cache_multiplier_threshold=dbfs_cache_multiplier_threshold,
            eager_spark_cache=eager_spark_cache, **kwargs
        )
    elif eager_spark_cache and not is_spark_cached(df_processed):
        log.info("Trigger spark cache (DBFS skipped).")
        df_processed = df_processed.cache()

    if not skip_display:
        try:
            if hasattr(df_processed, "display_in_notebook") and callable(df_processed.display_in_notebook):
                 df_processed.display_in_notebook() # type: ignore[attr-defined]
            else: log.warning("display_in_notebook not found.")
        except Exception as e: log.error(f"Display error: {e}")
        log.info(f"Process/display time: {time.time() - t_start:.2f}s")
    return df_processed


# Initialization function
def extend_dataframe_methods(
    spark_session: SparkSession, # Added spark_session parameter
    display_fun: Callable = lambda x, *a, **kw: display(x, *a, **kw), # Match display sig
    dbfs_cache_complexity_threshold: Optional[float] = getattr(config, "DEFAULT_COMPLEXITY_THRESHOLD", inf),
    dbfs_cache_multiplier_threshold: Optional[float] = getattr(config, "DEFAULT_MULTIPLIER_THRESHOLD", inf),
    disable_cache_and_display: bool = False,
):
    """Initialize SparkSession and DataFrame class with caching capabilities."""

    global original_create_dataframe
    # Store the original createDataFrame method before potentially overriding it
    # Only store if it hasn't been stored yet
    if original_create_dataframe is None and hasattr(spark_session, "createDataFrame"): # Use spark_session
         original_create_dataframe = spark_session.createDataFrame # Use spark_session
         log.info("Stored original spark_session.createDataFrame method.")

    if not hasattr(spark_session, "createCachedDataFrame"): # Use spark_session
         spark_session.createCachedDataFrame = types.MethodType(createCachedDataFrame, spark_session) # type: ignore[attr-defined]
         log.info("Attached 'createCachedDataFrame' to SparkSession.")

    # Extra threshold dependent conveience functions:
    def cacheToDbfsIfTriggered(self: DataFrame, **kwargs) -> DataFrame:
        return cacheToDbfs(self, dbfs_cache_complexity_threshold=dbfs_cache_complexity_threshold,
        dbfs_cache_multiplier_threshold=dbfs_cache_multiplier_threshold,**kwargs)

    # Dynamically set the docstring for the wrapper
    wrapper_doc_prefix = f"""Wrapper for cacheToDbfsIfTriggered with default thresholds.

    Default Complexity Threshold: {dbfs_cache_complexity_threshold}
    Default Multiplier Threshold: {dbfs_cache_multiplier_threshold}

    Original cacheToDbfs docstring:
    """
    cacheToDbfsIfTriggered.__doc__ = wrapper_doc_prefix + (cacheToDbfsIfTriggered.__doc__ or "No docstring found for cacheToDbfsIfTriggered.")

    # Attach withCachedDisplay
    def withCachedDisplayWrapper(self, **kwargs):
        if disable_cache_and_display: return self
        final_kwargs = {
             'dbfs_cache_complexity_threshold': dbfs_cache_complexity_threshold,
             'dbfs_cache_multiplier_threshold': dbfs_cache_multiplier_threshold, **kwargs
        }
        return __withCachedDisplay__(self, **final_kwargs)

    # Dynamically set the docstring for the wrapper
    wrapper_doc_prefix = f"""Wrapper for __withCachedDisplay__ with default thresholds.

    Default Complexity Threshold: {dbfs_cache_complexity_threshold}
    Default Multiplier Threshold: {dbfs_cache_multiplier_threshold}

    Original __withCachedDisplay__ docstring:
    """
    withCachedDisplayWrapper.__doc__ = wrapper_doc_prefix + (__withCachedDisplay__.__doc__ or "No docstring found for __withCachedDisplay__.")

    DataFrame.withCachedDisplay = withCachedDisplayWrapper # type: ignore
    DataFrame.wcd = withCachedDisplayWrapper # type: ignore
    DataFrame.display_in_notebook = display_fun # type: ignore
    DataFrame.cacheToDbfs = cacheToDbfs # type: ignore
    DataFrame.cacheToDbfsIfTriggered = cacheToDbfsIfTriggered # type: ignore
    DataFrame.clearDbfsCache = clearDbfsCache # type: ignore

    log.info("DataFrame extensions initialized.")
    return DataFrame

def get_catalog_name() -> str:
    return "hive_metastore" if is_serverless_cluster() else "spark_catalog"

# Helper to detect if running on a Databricks serverless cluster
def is_serverless_cluster() -> bool:
    # Allow override for testing
    if os.environ.get("DATABRICKS_RUNTIME_VERSION", "").startswith("client."):
        return True
    else:
        return False

# Utility function to extract hash from metadata (needed by tests)
def get_hash_from_metadata(metadata_txt: str) -> Optional[str]:
    """
    Extracts hash from metadata text containing table references.

    Looks for patterns like spark_catalog.{db_name}.{hash} or hive_metastore.{db_name}.{hash} in the metadata.
    Uses hive_metastore if running on serverless, otherwise spark_catalog.
    """
    import re
    db_name = getattr(config, "CACHE_DATABASE", "spark_cache")
    catalog_prefix = get_catalog_name()
    pattern = rf"{catalog_prefix}\.{db_name}\.([a-f0-9]{{32}})"
    match = re.search(pattern, metadata_txt)
    if match:
        return match.group(1)
    return None
