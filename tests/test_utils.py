import os
from unittest.mock import patch
import pytest

from dbfs_spark_cache import utils
from dbfs_spark_cache.config import config

# This test was moved from tests/test_dbfs_spark_cache.py
def test_get_hash_from_metadata_correct_group():
    """Test that get_hash_from_metadata extracts the correct hash (group 1)."""
    # Store original value and modify directly
    original_db_name = config.CACHE_DATABASE
    config.CACHE_DATABASE = 'test_cache_db'

    try:
        # Test case 1: Valid metadata with hash
        metadata_with_hash = "Some text before spark_catalog.test_cache_db.abcdef1234567890abcdef1234567890 some text after"
        expected_hash = "abcdef1234567890abcdef1234567890"
        assert utils.get_hash_from_metadata(metadata_with_hash) == expected_hash, "Test Case 1 Failed"

        # Test case 2: Metadata without the specific pattern
        metadata_without_hash = "Some text without the spark catalog pattern"
        assert utils.get_hash_from_metadata(metadata_without_hash) is None, "Test Case 2 Failed"

        # Test case 3: Metadata with different database name (should not match)
        metadata_diff_db = "spark_catalog.other_db.abcdef1234567890abcdef1234567890"
        assert utils.get_hash_from_metadata(metadata_diff_db) is None, "Test Case 3 Failed"

        # Test case 4: Empty string
        assert utils.get_hash_from_metadata("") is None, "Test Case 4 Failed"
    finally:
        # Restore original value to avoid side effects
        config.CACHE_DATABASE = original_db_name

@patch.dict(os.environ, {"DATABRICKS_RUNTIME_VERSION": "client.12.0"}, clear=True)
def test_is_serverless_cluster_client_version():
    assert utils.is_serverless_cluster() is True

@patch.dict(os.environ, {"DATABRICKS_RUNTIME_VERSION": "12.0.x-scala..."}, clear=True)
def test_is_serverless_cluster_non_client_version():
    assert utils.is_serverless_cluster() is False

@patch.dict(os.environ, {}, clear=True)
def test_is_serverless_cluster_no_env_var():
    assert utils.is_serverless_cluster() is False
