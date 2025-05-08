import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pandas as pd
from tqdm import tqdm

from .config import config
from .utils import empty_cached_table, get_table_name_from_hash # Import from utils
# Note: write_meta is used in clear_cache_for_hash, but it's part of the core caching write logic.
# It will be moved to core_caching.py later, and imported here then.
# For now, assume it's available or handle the dependency later.

# Configure module-level logger
log = logging.getLogger(__name__)

# Assume spark and dbutils are available globally or imported elsewhere if needed by these functions
# For now, keep the try-except block for spark/dbutils availability as it was in the original file
try:
    from databricks.sdk.runtime import dbutils, spark
except ImportError:
    log.warning("databricks.sdk.runtime not found. dbutils, spark will not be available for cache management.")
    dbutils = None # type: ignore[assignment,misc]
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
    specific_database: str | None = None, # Added specific_database
    confirm_delete: bool = False # Added confirm_delete
):
    # Use the specific_database if provided, otherwise use the configured default
    cache_db_to_clear = specific_database or config.CACHE_DATABASE
    original_config_db = config.CACHE_DATABASE

    if specific_database:
        config.CACHE_DATABASE = specific_database

    df_files = get_cached_tables()

    if specific_database: # Restore original config if it was changed
        config.CACHE_DATABASE = original_config_db
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

    if len(rows) == 0:
        log.info(f"No caches older than {num_days} days found in database '{cache_db_to_clear}'.")
        return

    if confirm_delete:
        # This part is tricky in a non-interactive notebook environment.
        # For now, we'll log and proceed if confirm_delete is True (default).
        # In a real CLI tool, you'd use input().
        log.info(f"Found {len(rows)} cache(s) older than {num_days} days in database '{cache_db_to_clear}' to delete.")
        # If this were interactive:
        # confirmation = input(f"Proceed with deleting {len(rows)} cache(s) from '{cache_db_to_clear}'? (yes/no): ")
        # if confirmation.lower() != 'yes':
        #     log.info("Deletion cancelled by user.")
        #     return
    elif not confirm_delete:
        log.info(f"confirm_delete is False. Proceeding with deletion of {len(rows)} cache(s) from '{cache_db_to_clear}'.")


    def clear_cache_row(row_info):
        # When clearing for a specific_database, ensure the hash is cleared from *that* database.
        # clear_cache_for_hash uses the global config.CACHE_DATABASE, so we need to set it temporarily.
        current_config_db = config.CACHE_DATABASE
        if specific_database:
            config.CACHE_DATABASE = specific_database

        try:
            clear_cache_for_hash(row_info.hash_name)
        finally:
            if specific_database: # Restore original config
                config.CACHE_DATABASE = current_config_db


    max_workers = num_threads if num_threads is not None else (os.cpu_count() or 1) * 4

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(clear_cache_row, [row for _, row in rows.iterrows()] if rows is not None else []),
                total=len(rows),
                desc=f"Clearing old caches from {cache_db_to_clear}"
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
