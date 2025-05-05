from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pyspark.sql import Row, SparkSession

from dbfs_spark_cache import caching


@pytest.fixture(scope="module")
def spark():
    # Pyright may not recognize .master, but this is correct for PySpark
    return SparkSession.builder.master("local[1]").appName("test").getOrCreate()

def test_hash_input_data_pandas():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    h1 = caching._hash_input_data(df)
    h2 = caching._hash_input_data(df.copy())
    assert isinstance(h1, str) and len(h1) == 32
    assert h1 == h2  # Hash should be deterministic

def test_hash_input_data_list():
    data = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    h1 = caching._hash_input_data(data)
    h2 = caching._hash_input_data(tuple(data))
    assert isinstance(h1, str) and len(h1) == 32
    assert h1 == h2

def test_hash_input_data_empty():
    h = caching._hash_input_data([])
    assert isinstance(h, str) and len(h) == 32

def test_hash_input_data_rdd_error(spark):
    rdd = spark.sparkContext.parallelize([Row(a=1)])
    with pytest.raises(TypeError):
        caching._hash_input_data(rdd)

def test_hash_input_data_dataframe_edge_cases():
    # Different column order
    df1 = pd.DataFrame({
        "a": pd.Series([1, 2, 3], dtype="int32"),
        "b": pd.Series([4.0, 5.5, 6.1], dtype="float64"),
        "c": pd.Series(["x", "y", "z"], dtype="string")
    })
    df2 = pd.DataFrame({
        "b": pd.Series([4.0, 5.5, 6.1], dtype="float64"),
        "a": pd.Series([1, 2, 3], dtype="int32"),
        "c": pd.Series(["x", "y", "z"], dtype="string")
    })
    assert caching._hash_input_data(df1) != caching._hash_input_data(df2)

    # Different float value
    df3 = pd.DataFrame({
        "a": pd.Series([1, 2, 3], dtype="int32"),
        "b": pd.Series([4.0, 5.5, 6.1], dtype="float64"),
        "c": pd.Series(["x", "y", "z"], dtype="string")
    })
    df4 = pd.DataFrame({
        "a": pd.Series([1, 2, 3], dtype="int32"),
        "b": pd.Series([4.0, 5.5, 6.1000001], dtype="float64"),
        "c": pd.Series(["x", "y", "z"], dtype="string")
    })
    assert caching._hash_input_data(df3) != caching._hash_input_data(df4)

    # Different dtype
    df5 = pd.DataFrame({
        "a": pd.Series([1, 2, 3], dtype="int32"),
        "b": pd.Series([4.0, 5.5, 6.1], dtype="float64"),
        "c": pd.Series(["x", "y", "z"], dtype="string")
    })
    df6 = pd.DataFrame({
        "a": pd.Series([1, 2, 3], dtype="int32"),
        "b": pd.Series([4.0, 5.5, 6.1], dtype="float32"),
        "c": pd.Series(["x", "y", "z"], dtype="string")
    })
    assert caching._hash_input_data(df5) != caching._hash_input_data(df6)

@patch("dbfs_spark_cache.caching.get_table_name_from_hash", lambda x: "test_cache_db.data_" + "a"*32)
def test_create_cached_dataframe_miss_and_hit(spark, tmp_path):
    with patch.object(caching.config, "SPARK_CACHE_DIR", f"dbfs:/{tmp_path}/"), \
         patch.object(caching.config, "CACHE_DATABASE", "test_cache_db"), \
         patch.object(spark, "createDataFrame") as mock_create_df, \
         patch.object(spark.catalog, "tableExists") as mock_table_exists, \
             patch("pyspark.sql.readwriter.DataFrameReader.table") as mock_read_table, \
             patch("builtins.open"):
        # Setup mock DataFrame
        mock_df = MagicMock()
        mock_df.count.return_value = 2
        mock_df.collect.return_value = [("a", 1), ("b", 2)]
        mock_df.write.format.return_value.mode.return_value.saveAsTable.return_value = None
        mock_create_df.return_value = mock_df
        mock_read_table.return_value = mock_df
        mock_table_exists.return_value = False

        data = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        # Miss: should create cache
        df1 = caching.createCachedDataFrame(spark, data)
        assert df1.count() == 2
        # Simulate cache hit
        mock_table_exists.return_value = True
        df2 = caching.createCachedDataFrame(spark, data)
        assert df2.count() == 2
        # Should be the same data
        assert df1 is df2

@patch("dbfs_spark_cache.caching.get_table_name_from_hash", lambda x: "test_cache_db.data_" + "b"*32)
def test_create_cached_dataframe_schema(spark, tmp_path):
    with patch.object(caching.config, "SPARK_CACHE_DIR", f"dbfs:/{tmp_path}/"), \
         patch.object(caching.config, "CACHE_DATABASE", "test_cache_db"), \
         patch.object(spark, "createDataFrame") as mock_create_df, \
         patch.object(spark.catalog, "tableExists") as mock_table_exists, \
         patch("pyspark.sql.readwriter.DataFrameReader.table") as mock_read_table, \
         patch("builtins.open"):
        # Setup mock DataFrame with schema
        mock_schema = MagicMock()
        mock_schema.__getitem__.side_effect = lambda k: MagicMock(dataType=MagicMock(typeName=lambda: "integer" if k == "a" else "string"))
        mock_df = MagicMock()
        mock_df.schema = mock_schema
        mock_df.write.format.return_value.mode.return_value.saveAsTable.return_value = None
        mock_create_df.return_value = mock_df
        mock_read_table.return_value = mock_df
        mock_table_exists.return_value = False

        data = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        schema = "a INT, b STRING"
        df = caching.createCachedDataFrame(spark, data, schema=schema)
        assert df.schema["a"].dataType.typeName() == "integer"
        assert df.schema["b"].dataType.typeName() == "string"

def test_create_cached_dataframe_rdd_error(spark):
    rdd = spark.sparkContext.parallelize([Row(a=1)])
    with pytest.raises(TypeError):
        caching.createCachedDataFrame(spark, rdd)

@patch("dbfs_spark_cache.caching.get_table_name_from_hash", lambda x: "test_cache_db.data_" + "c"*32)
def test_cacheToDbfs_bypass_for_data_cache(spark, tmp_path):
    with \
        patch.object(caching.config, "SPARK_CACHE_DIR", f"dbfs:/{tmp_path}/"), \
        patch.object(caching.config, "CACHE_DATABASE", "test_cache_db"), \
        patch.object(spark, "createDataFrame") as mock_create_df, \
        patch.object(spark.catalog, "tableExists") as mock_table_exists, \
         patch("pyspark.sql.readwriter.DataFrameReader.table") as mock_read_table, \
         patch("builtins.open"), \
         patch("dbfs_spark_cache.caching.get_query_plan") as mock_get_query_plan:
        # Setup mock DataFrame
        mock_df = MagicMock()
        mock_df.count.return_value = 1
        mock_df.collect.return_value = [("a", 1)]
        mock_df.write.format.return_value.mode.return_value.saveAsTable.return_value = None
        mock_df.inputFiles.return_value = []
        mock_create_df.return_value = mock_df
        mock_read_table.return_value = mock_df
        mock_table_exists.return_value = False
        # Simulate direct data cache detection
        mock_get_query_plan.return_value = "spark_catalog.test_cache_db.data_cccccccccccccccccccccccccccccccc"

        data = [{"a": 1, "b": "x"}]
        df = caching.createCachedDataFrame(spark, data)
        # Should bypass cacheToDbfs logic and just return self
        with patch.object(caching.log, "info") as mock_log:
            result = caching.cacheToDbfs(df)
            assert result is df
            assert any("Direct data cache source. Skipping standard cache." in str(call) for call in mock_log.call_args_list)
