import hashlib
import io
import logging
import os
import re
import sys
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Mapping

import pandas as pd
from py4j.protocol import Py4JJavaError  # type: ignore[import-untyped]
from pyspark.errors.exceptions.base import AnalysisException  # Changed import
from pyspark.sql import DataFrame, SparkSession


from .config import config
from .utils import get_table_name_from_hash, get_hash_from_metadata # Import from utils
from .hashing import _find_catalog_table_pattern_in_text, _hash_input_data # Import from hashing

# Configure module-level logger
log = logging.getLogger(__name__)

# Assume spark and dbutils are available globally or imported elsewhere if needed by these functions
try:
    from databricks.sdk.runtime import dbutils, spark
except ImportError:
    log.warning("databricks.sdk.runtime not found. dbutils, spark will not be available for core caching.")
    dbutils = None # type: ignore[assignment,misc]
    spark = None # type: ignore[assignment]


# Store the original createDataFrame method globally
original_create_dataframe: Optional[Callable] = None


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
        last_meta_bytes = dbutils.fs.head(metadata_file_path, 1024*1024)
        last_meta = last_meta_bytes
        log.debug(f"Read existing metadata (bytes) from {metadata_file_path}")
    except Exception as e:
        if "FileNotFoundException" in str(e) or "does not exist" in str(e):
             log.debug(f"Metadata file {metadata_file_path} not found.")
        else:
             log.warning(f"Could not read existing metadata from {metadata_file_path}: {e}")
        last_meta = None

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

    df.write.format("delta").mode("overwrite").saveAsTable(table_name)
    write_cache_data(df, table_name)
    write_meta(metadata_file_path, metadata_txt)


# Core Caching Logic Functions
def createCachedDataFrame(
    spark_session: SparkSession,
    data: Union[pd.DataFrame, List[Any], Tuple[Any, ...]],
    schema: Optional[Any] = None,
    **kwargs,
) -> DataFrame:
    global original_create_dataframe
    if spark_session is None:
        raise RuntimeError("SparkSession not available.")
    if dbutils is None:
        raise RuntimeError("dbutils not available, cannot manage cache metadata.")

    if original_create_dataframe is None:
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
    table_exists = False
    try:
        table_exists = spark_session.catalog.tableExists(table_name)
    except AnalysisException as e:
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
        metadata_exists = False
        try:
            dbutils.fs.ls(metadata_file_path)
            metadata_exists = True
        except Exception:
            log.warning(f"Cache table {table_name} exists, but metadata file {metadata_file_path} is missing or inaccessible.")
            metadata_exists = False
        if not metadata_exists:
             log.warning(f"Attempting to use table {table_name} despite missing metadata.")
        df_read = spark_session.read.table(table_name)
        # Tag DataFrame as direct data cache for later hash detection
        setattr(df_read, "_is_direct_data_cache", True)
        setattr(df_read, "_direct_data_cache_hash", cache_hash_name)
        return df_read
    else:
        log.info(f"Creating new direct data cache: {table_name}")
        try:
            if original_create_dataframe is None:
                 raise RuntimeError("Original createDataFrame method is None.")
            df_source = original_create_dataframe(data=data, schema=schema, **kwargs) # type: ignore[arg-type]
            if spark is not None:
                spark_session.sql(f"CREATE DATABASE IF NOT EXISTS {config.CACHE_DATABASE}")
            df_source.write.format("delta").mode("overwrite").saveAsTable(table_name)

            metadata_txt = (
                f"CACHE TYPE: Direct Data Input\n"
                f"DATA HASH: {data_hash}\n"
                f"CREATION TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            write_meta(metadata_file_path, metadata_txt)
            log.info(f"Metadata written to {metadata_file_path}")

            df_read = spark_session.read.table(table_name)
            # Tag DataFrame as direct data cache for later hash detection
            setattr(df_read, "_is_direct_data_cache", True)
            setattr(df_read, "_direct_data_cache_hash", cache_hash_name)
            return df_read

        except Exception as e:
            log.error(f"Failed during createCachedDataFrame cache miss processing: {e}")
            raise


# _is_plan_complex function removed as per revised plan


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
            relevant_files = [fi for fi in ls_files if fi.name != "_delta_log/"]
            if not relevant_files:
                try: # Check modification time of the directory itself if no relevant files found
                    dir_info = dbutils.fs.ls(dir_path.rstrip('/'))
                    if dir_info: # Ensure dir_info is not empty
                        return datetime.fromtimestamp(dir_info[0].modificationTime / 1000)
                except Exception: # pylint: disable=broad-except
                    pass # Fall through if directory info cannot be fetched
                return None # No relevant files and no directory info

            latest_mod_time = max(fi.modificationTime for fi in relevant_files)
            return datetime.fromtimestamp(latest_mod_time / 1000)
        except Exception as e: # pylint: disable=broad-except
             if "FileNotFoundException" in str(e) or "does not exist" in str(e):
                 log.warning(f"Input path {dir_path} does not exist.")
             else:
                 log.warning(f"Could not list files in {dir_path} using dbutils: {e}. Skipping timestamp check.")
             return None

    plan = get_query_plan(df) # Get plan upfront

    # Check 1: Is it reading from a catalog 'data_*' table (created by createCachedDataFrame)?
    db_name_for_data_table_check = config.CACHE_DATABASE # Direct access is preferred
    if _find_catalog_table_pattern_in_text(plan, db_name_for_data_table_check, "data_"):
        # Revised Plan: If source is a catalog data_* table, always treat as non-bypass candidate.
        # Return {} to force standard caching logic based on the current plan.
        log.debug("DataFrame reads from a catalog 'data_*' table. Proceeding with standard cache logic (not bypassing).")
        return {}

    # If not a catalog 'data_*' table, proceed to check inputFiles for other cache types or real sources.
    input_files: List[str] = []
    dir_paths: set[str] = set()
    try:
        input_files = df.inputFiles()
        dir_paths = set(os.path.dirname(f) for f in input_files) # Use os.path.dirname
    except Py4JJavaError as e:
        error_str = str(e)
        if "DELTA_SCHEMA_CHANGE_SINCE_ANALYSIS" in error_str or "DeltaAnalysisException: DELTA_SCHEMA_CHANGE_SINCE_ANALYSIS" in error_str:
            log.warning("Could not get input files due to Delta schema change: %s. Forcing cache invalidation.", e)
            return {"<schema_changed_placeholder>": datetime.now()}
        raise e # Re-raise if not the specific Delta error
    except Exception as e: # pylint: disable=broad-except
        # Broader catch for other unexpected errors from inputFiles(), including potential AnalysisException for missing tables
        error_str = str(e)
        if "DELTA_SCHEMA_CHANGE_SINCE_ANALYSIS" in error_str or "DeltaAnalysisException: DELTA_SCHEMA_CHANGE_SINCE_ANALYSIS" in error_str: # Duplicate check, but fine
            log.warning("Could not get input files due to Delta schema change (general exception): %s. Forcing cache invalidation.", e)
            return {"<schema_changed_placeholder>": datetime.now()}
        log.warning(f"Could not get input files for DataFrame: {e}. Treating as empty source.")
        return {} # Treat as empty/unknown source if inputFiles fails for other reasons

    if not input_files:
        # No input files and not a catalog data_* table (checked above).
        # This could be df.parallelize(), df created from empty list, etc.
        # These are handled by "Scan ExistingRDD" check in dataframe_extensions.py or result in empty source here.
        log.debug("DataFrame has no input files and is not a catalog 'data_*' table.")
        return {}

    # Collect modification times for actual source directories,
    # skipping standard DBFS cache directories (hex-named under SPARK_CACHE_DIR).
    final_sources: Dict[str, datetime] = {}
    # Pattern for standard cache directories in SPARK_CACHE_DIR (hex hash name, not data_ prefixed)
    # These are caches written by write_dbfs_cache with an arbitrary hash.
    standard_cache_dir_base = config.SPARK_CACHE_DIR.rstrip('/')
    # Regex: ^<escaped_base_dir>/[a-f0-9]{32}/?$
    # Ensure standard_cache_dir_base itself is properly escaped if it contains regex special chars.
    # However, typical paths like /dbfs/FileStore/tables/cache/ are unlikely to.
    standard_cache_dir_pattern_str = f"^{re.escape(standard_cache_dir_base)}/[a-f0-9]{{32}}/?$"

    for d_path_str in sorted(list(dir_paths)):
        # If the directory 'd_path_str' is NOT a standard cache path
        if not re.match(standard_cache_dir_pattern_str, d_path_str):
            mod_time = last_mod_datetime_from_dbfs_dir(d_path_str)
            if mod_time:
                final_sources[d_path_str] = mod_time
    return final_sources


def get_query_plan(df: DataFrame) -> str:
    """Gets the cleaned query plan string using public API."""
    if spark is None:
         log.warning("SparkSession not available, cannot get query plan.")
         return "Error: SparkSession not available"
    try:
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        try:
            df.explain(mode="extended")
            plan_str_val = redirected_output.getvalue()
        finally:
            sys.stdout = old_stdout

        cleaned_plan = re.sub(r"#\d+", "", plan_str_val)
        cleaned_plan = re.sub(r"\s+", " ", cleaned_plan).strip()

        if "UDF" in plan_str_val:
            log.warning("UDF detected in query plan, cache invalidation for UDF code not implemented!")
        if "Photon does not fully support" in plan_str_val:
            log.info(f"Photon limitations might affect caching:\n{plan_str_val.split('Photon does not fully support the query because:')[1]}")
        return cleaned_plan
    except Exception as e:
        log.error(f"Error getting query plan using explain(extended): {e}")
        return f"Error: {e}"


def _extract_input_sources_from_metadata(metadata_txt: str) -> Dict[str, str]:
    """Extracts input source paths and modification times from metadata string."""
    sources: Dict[str, str] = {}
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
                elif path: # This elif should align with the `if len(parts) == 2:`
                    log.warning(f"Parse error: {line}")
                    sources[path] = "<parse_error>"
            elif stripped_line == "DATAFRAME QUERY PLAN:":
                break
            else: # This else should align with `if line.startswith("  "):`
                if ":" not in stripped_line: # This if should be indented under the else
                    break
                else: # This else should align with `if ":" not in stripped_line:`
                    parts = stripped_line.split(":", 1)
                    path = parts[0].strip()
                    sources[path] = parts[1].strip()
    return sources


def get_cache_metadata(input_dir_mod_datetime: Mapping[str, Union[datetime, bool]], query_plan: str) -> str:
    """Generates the metadata string including input sources and query plan."""
    newline = "\n"

    source_lines = []
    for path, t in input_dir_mod_datetime.items():
        if isinstance(t, datetime):
            source_lines.append(f"  {path}: {t:%Y-%m-%d %H:%M:%S}")
        elif isinstance(t, bool) and path == "<direct_data_cache>": # Handle the boolean marker
            source_lines.append(f"  {path}: {str(t)}")
            # Or skip, or handle as a special entry, depending on desired metadata format
    sources_str = newline.join(source_lines)

    return f"""INPUT SOURCES MODIFICATION DATETIMES:
{sources_str}

DATAFRAME QUERY PLAN:
{query_plan}"""


def get_table_cache_info(
    input_dir_mod_datetime: Mapping[str, Union[datetime, bool]],
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


def read_dbfs_cache_if_exist(
    df: DataFrame, query_plan: Optional[str] = None, input_dir_mod_datetime: Optional[Mapping[str, Union[datetime, bool]]] = None
) -> Optional[DataFrame]:
    """Reads from DBFS cache if a valid cache entry exists, using dbutils.fs."""
    if spark is None:
        log.warning("SparkSession not available, cannot read from cache.")
        return None
    if dbutils is None:
        log.warning("dbutils not available, cannot check metadata existence.")
        return None

    processed_input_dir_mod_datetime: Mapping[str, Union[datetime, bool]]
    if input_dir_mod_datetime is not None:
        processed_input_dir_mod_datetime = input_dir_mod_datetime
    else:
        processed_input_dir_mod_datetime = get_input_dir_mod_datetime(df)

    current_query_plan = query_plan if query_plan is not None else get_query_plan(df)

    hash_name, _, metadata_file_path, _ = get_table_cache_info(
        input_dir_mod_datetime=processed_input_dir_mod_datetime,
        query_plan=current_query_plan
    )
    table_name = get_table_name_from_hash(hash_name)

    metadata_exists = False
    try:
        dbutils.fs.head(metadata_file_path, 1)
        metadata_exists = True
        log.debug(f"Metadata file found at {metadata_file_path}")
    except Exception as e:
        if "FileNotFoundException" in str(e) or "does not exist" in str(e):
            log.info(f"No cache metadata found at {metadata_file_path} (dbutils.fs.head)")
            local_path = metadata_file_path
            if local_path.startswith("/dbfs/"): # pragma: no cover
                local_path = local_path.replace("/dbfs/", "/dbfs/", 1) # pragma: no cover
            if os.path.exists(local_path): # pragma: no cover
                metadata_exists = True # pragma: no cover
                log.debug(f"Metadata file found locally at {local_path}") # pragma: no cover
        else:
            log.warning(f"Error checking metadata existence at {metadata_file_path}: {e}")

    if not metadata_exists:
        log.info(f"Cache miss: metadata file {metadata_file_path} does not exist.")
        return None

    try:
        if hasattr(spark, "catalog") and hasattr(spark.catalog, "tableExists"):
            if spark.catalog.tableExists(table_name):
                log.info(f"Found valid cache table: {table_name}")
                return spark.read.table(table_name)
            else:
                log.warning(f"Cache metadata exists but table {table_name} does not exist")
                return None
        elif hasattr(spark, "read") and hasattr(spark.read, "table"): # For mock spark
            log.info(f"Test environment: returning mock table for {table_name}")
            return spark.read.table(table_name) # type: ignore
        else:
            log.error("SparkSession not available or not fully mocked, cannot check if table exists.")
            return None
    except Exception as e:
        log.error(f"Error reading cache table {table_name}: {e}")
        return None


def write_dbfs_cache(
    df: DataFrame, replace: bool = True, query_plan: str = "",
    input_dir_mod_datetime: Optional[Mapping[str, Union[datetime, bool]]] = None,
    hash_name: Optional[str] = None, cache_path: str = config.SPARK_CACHE_DIR,
    verbose: bool = False
) -> DataFrame:
    """Writes DataFrame to DBFS cache and returns cached DataFrame."""

    processed_input_dir_mod_datetime: Mapping[str, Union[datetime, bool]]
    if input_dir_mod_datetime is not None:
        processed_input_dir_mod_datetime = input_dir_mod_datetime
    else:
        processed_input_dir_mod_datetime = get_input_dir_mod_datetime(df)

    final_hash, final_cache_path, metadata_file_path, metadata_txt = get_table_cache_info(
        input_dir_mod_datetime=processed_input_dir_mod_datetime,
        query_plan=query_plan if query_plan else get_query_plan(df), # Ensure query_plan is not empty
        hash_name=hash_name,
        cache_path_base=cache_path
    )

    table_name = get_table_name_from_hash(final_hash)

    _write_standard_cache(
        df=df,
        hash_name=final_hash,
        cache_path=final_cache_path,
        metadata_file_path=metadata_file_path,
        metadata_txt=metadata_txt,
        verbose=verbose
    )

    try:
        if spark is None:
            log.error("SparkSession not available, cannot read cache table.")
            return df
        if replace:
            return spark.read.table(table_name)
        else:
            return df
    except Exception as e:
        log.error(f"Error reading newly created cache table {table_name}: {e}")
        return df


def is_spark_cached(df: DataFrame) -> bool:
    """Checks if a DataFrame is marked for Spark caching (memory or disk)."""
    try:
        return df.storageLevel.useMemory or df.storageLevel.useDisk
    except Exception:
        return False


def get_table_hash(df: DataFrame) -> str:
    """
    Gets the hash for a DataFrame based on its query plan and input sources.
    Checks if the DataFrame is reading directly from a 'data_' cache table first.
    """
    plan_str = get_query_plan(df)
    db_name = getattr(config, "CACHE_DATABASE", "spark_cache")

    # If DataFrame was tagged as direct data cache, return its original hash immediately
    if getattr(df, "_is_direct_data_cache", False):
        original_hash = getattr(df, "_direct_data_cache_hash", None)
        if original_hash:
            log.info(f"Using tagged direct data cache hash for DataFrame: {original_hash}")
            return original_hash

    # Next, if the plan references a direct data cache table, return that hash only for pure scans
    data_hash_match = _find_catalog_table_pattern_in_text(plan_str, db_name, "data_")
    if data_hash_match:
        # Ensure no additional operations beyond scan (e.g., no Project, Filter, Join)
        relation_pattern = rf"Relation\[.*\] {re.escape(db_name)}\.{re.escape(data_hash_match)}"
        plan_without_relation = re.sub(relation_pattern, "", plan_str).strip()
        # Only return direct data cache hash if nothing else remains in plan
        if not plan_without_relation:
            log.info(f"Identified pure direct data cache scan in plan: {data_hash_match}")
            return data_hash_match
        log.debug(f"Direct data cache referenced but plan has extra operations, computing new hash. Remainder: {plan_without_relation[:100]}")

    input_dir_mod_datetime_raw = get_input_dir_mod_datetime(df)
    input_dir_mod_datetime: Dict[str, datetime] = {}
    if isinstance(input_dir_mod_datetime_raw, dict):
        input_dir_mod_datetime = {k: v for k, v in input_dir_mod_datetime_raw.items() if isinstance(v, datetime)}

    query_plan_for_hash = plan_str
    metadata_txt = get_cache_metadata(
        input_dir_mod_datetime=input_dir_mod_datetime,
        query_plan=query_plan_for_hash,
    )
    extracted_hash_name = get_hash_from_metadata(metadata_txt)
    if extracted_hash_name is not None:
        log.info(f"Found hash name from metadata: {extracted_hash_name}")
        return extracted_hash_name

    log.info("Calculating hash from query metadata as fallback.")
    calculated_hash_name = hashlib.md5(metadata_txt.encode("utf-8")).hexdigest()
    return calculated_hash_name
