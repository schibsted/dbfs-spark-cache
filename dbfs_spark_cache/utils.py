import logging
import pandas as pd
from typing import Optional
# Configure module-level logger
log = logging.getLogger(__name__)


def empty_cached_table():
    """Returns an empty Pandas DataFrame with standard cache table columns."""
    return pd.DataFrame(columns=["table_name", "hash_name", "directory_path", "creationTime"])

# Helper to detect if running on a Databricks serverless cluster
def is_serverless_cluster() -> bool:
    # Allow override for testing
    import os # Import os here
    if os.environ.get("DATABRICKS_RUNTIME_VERSION", "").startswith("client."):
        return True
    else:
        return False

# Utility function to extract hash from metadata (needed by tests)
def get_hash_from_metadata(metadata_txt: str) -> Optional[str]:
    """
    Extracts hash from metadata text containing table references.

    Looks for patterns like {catalog}.{db_name}.{hash} in the metadata.
    It checks for 'hive_metastore' first, and if not found,
    it checks for 'spark_catalog'. This makes it robust to variations
    in how the catalog is represented in query plans.
    """
    import re # Import re here
    from .config import config # Import config

    db_name = getattr(config, "CACHE_DATABASE", "spark_cache")

    # Check for 'hive_metastore' pattern first.
    pattern_hive = rf"hive_metastore\.{re.escape(db_name)}\.([a-f0-9]{{32}})"
    match_hive = re.search(pattern_hive, metadata_txt)
    if match_hive:
        log.debug(f"Extracted hash '{match_hive.group(1)}' using 'hive_metastore' pattern from metadata.")
        return match_hive.group(1)

    # If not found with hive_metastore, check for 'spark_catalog' pattern.
    pattern_spark = rf"spark_catalog\.{re.escape(db_name)}\.([a-f0-9]{{32}})"
    match_spark = re.search(pattern_spark, metadata_txt)
    if match_spark:
        log.debug(f"Extracted hash '{match_spark.group(1)}' using 'spark_catalog' pattern from metadata.")
        return match_spark.group(1)

    log.debug("Could not extract hash from metadata using 'hive_metastore' or 'spark_catalog' patterns. Full metadata searched.")
    return None

def get_table_name_from_hash(hash_name: str) -> str:
    """Constructs the fully qualified table name from a hash."""
    from .config import config # Import config
    db_name = getattr(config, "CACHE_DATABASE", "spark_cache") # Provide default if missing
    return f"{db_name}.{hash_name}"
