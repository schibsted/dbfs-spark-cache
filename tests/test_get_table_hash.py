import unittest
from unittest.mock import MagicMock, patch
import hashlib

from dbfs_spark_cache.caching import get_table_hash, get_cache_metadata

class TestGetTableHash(unittest.TestCase):

    @patch("dbfs_spark_cache.core_caching.get_input_dir_mod_datetime")
    @patch("dbfs_spark_cache.core_caching.get_query_plan")
    @patch("dbfs_spark_cache.core_caching.get_hash_from_metadata")
    def test_get_table_hash_with_existing_hash(self, mock_get_hash_from_metadata, mock_get_query_plan, mock_get_input_dir_mod_datetime):
        # Arrange
        mock_df = MagicMock()
        mock_df._is_direct_data_cache = False # Ensure it doesn't take the direct data cache path
        from datetime import datetime
        mock_get_input_dir_mod_datetime.return_value = {"some_path": datetime(2025, 4, 22, 10, 0, 0)}
        mock_get_query_plan.return_value = "SELECT * FROM table"
        expected_hash = "abcdef1234567890abcdef1234567890"
        # get_hash_from_metadata returns a hash string
        mock_get_hash_from_metadata.return_value = expected_hash

        # Act
        result = get_table_hash(mock_df)

        # Assert
        self.assertEqual(result, expected_hash)
        mock_get_input_dir_mod_datetime.assert_called_once_with(mock_df)
        mock_get_query_plan.assert_called_once_with(mock_df)
        metadata_txt = get_cache_metadata(mock_get_input_dir_mod_datetime.return_value, mock_get_query_plan.return_value)
        mock_get_hash_from_metadata.assert_called_once_with(metadata_txt)

    @patch("dbfs_spark_cache.core_caching.get_input_dir_mod_datetime")
    @patch("dbfs_spark_cache.core_caching.get_query_plan")
    @patch("dbfs_spark_cache.core_caching.get_hash_from_metadata")
    def test_get_table_hash_without_existing_hash(self, mock_get_hash_from_metadata, mock_get_query_plan, mock_get_input_dir_mod_datetime):
        # Arrange
        mock_df = MagicMock()
        mock_df._is_direct_data_cache = False # Ensure it doesn't take the direct data cache path
        from datetime import datetime
        mock_get_input_dir_mod_datetime.return_value = {"some_path": datetime(2025, 4, 22, 10, 0, 0)}
        mock_get_query_plan.return_value = "SELECT * FROM table"
        # get_hash_from_metadata returns None
        mock_get_hash_from_metadata.return_value = None

        # Act
        result = get_table_hash(mock_df)

        # Assert
        metadata_txt = get_cache_metadata(mock_get_input_dir_mod_datetime.return_value, mock_get_query_plan.return_value)
        expected_hash = hashlib.md5(metadata_txt.encode("utf-8")).hexdigest()
        self.assertEqual(result, expected_hash)
        mock_get_input_dir_mod_datetime.assert_called_once_with(mock_df)
        mock_get_query_plan.assert_called_once_with(mock_df)
        mock_get_hash_from_metadata.assert_called_once_with(metadata_txt)

if __name__ == "__main__":
    unittest.main()
