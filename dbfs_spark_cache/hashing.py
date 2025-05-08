import hashlib
import io
import logging
import re
from typing import Any, List, Optional, Tuple, Union

import pandas as pd
from pyspark.rdd import RDD


# Configure module-level logger
log = logging.getLogger(__name__)

def _find_catalog_table_pattern_in_text(text: str, db_name: str, table_prefix: str = "") -> Optional[str]:
    """
    Searches for table patterns like 'hive_metastore.db.prefix_hash' or 'spark_catalog.db.prefix_hash'
    in the given text. Returns the full table identifier (prefix + hash) if found.
    """
    # Check for 'hive_metastore' pattern first.
    # The capturing group (group 1) will capture (table_prefix + hash_value)
    pattern_hive = rf"hive_metastore\.{re.escape(db_name)}\.({re.escape(table_prefix)}[a-f0-9]{{32}})"
    match_hive = re.search(pattern_hive, text)
    if match_hive:
        log.debug(f"Found pattern with 'hive_metastore': {match_hive.group(1)}")
        return match_hive.group(1)

    # If not found, check for 'spark_catalog' pattern.
    pattern_spark = rf"spark_catalog\.{re.escape(db_name)}\.({re.escape(table_prefix)}[a-f0-9]{{32}})"
    match_spark = re.search(pattern_spark, text)
    if match_spark:
        log.debug(f"Found pattern with 'spark_catalog': {match_spark.group(1)}")
        return match_spark.group(1)

    log.debug(f"Could not find pattern for prefix '{table_prefix}' with 'hive_metastore' or 'spark_catalog' in text.")
    return None

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
