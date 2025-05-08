import pandas as pd
import pytest
from pyspark.sql import Row, SparkSession

from dbfs_spark_cache.hashing import _hash_input_data

@pytest.fixture(scope="module")
def spark():
    # Pyright may not recognize .master, but this is correct for PySpark
    return SparkSession.builder.master("local[1]").appName("test_hashing").getOrCreate() # type: ignore[attr-defined]

def test_hash_input_data_pandas():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    h1 = _hash_input_data(df)
    h2 = _hash_input_data(df.copy())
    assert isinstance(h1, str) and len(h1) == 32
    assert h1 == h2  # Hash should be deterministic

def test_hash_input_data_list():
    data = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    h1 = _hash_input_data(data)
    h2 = _hash_input_data(tuple(data))
    assert isinstance(h1, str) and len(h1) == 32
    assert h1 == h2

def test_hash_input_data_empty():
    h = _hash_input_data([])
    assert isinstance(h, str) and len(h) == 32

def test_hash_input_data_rdd_error(spark):
    rdd = spark.sparkContext.parallelize([Row(a=1)])
    with pytest.raises(TypeError):
        _hash_input_data(rdd)

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
    assert _hash_input_data(df1) != _hash_input_data(df2)

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
    assert _hash_input_data(df3) != _hash_input_data(df4)

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
    assert _hash_input_data(df5) != _hash_input_data(df6)
