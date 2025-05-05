import pytest
pytest.skip("Skipping query plan hash stability tests in non-JVM environment", allow_module_level=True)

import unittest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

from dbfs_spark_cache.caching import get_query_plan, get_table_hash

class TestQueryPlanHashStability(unittest.TestCase):
    spark: SparkSession

    @classmethod
    def setUpClass(cls):
        # Initialize a local SparkSession for testing
        cls.spark = (
            SparkSession.builder
            .master("local[1]")  # type: ignore[attr-defined]
            .appName("test_query_plan_hash_stability")
            .getOrCreate()
        )

    @classmethod
    def tearDownClass(cls):
        # Stop SparkSession after tests
        cls.spark.stop()

    def test_identical_queries_same_plan_and_hash(self):
        # Create two identical DataFrames with explicit schema
        data = [(1, "a"), (2, "b"), (3, "c")]
        schema = StructType([
            StructField("id", IntegerType(), True),
            StructField("val", StringType(), True)
        ])
        df1 = self.spark.createDataFrame(data, schema=schema)
        df2 = self.spark.createDataFrame(data, schema=schema)

        # Plans should be identical
        plan1 = get_query_plan(df1)
        plan2 = get_query_plan(df2)
        self.assertEqual(plan1, plan2, "Query plans for identical DataFrames should match")

        # Hashes should also be identical
        hash1 = get_table_hash(df1)
        hash2 = get_table_hash(df2)
        self.assertEqual(hash1, hash2, "Hashes for identical DataFrames should match")

    def test_different_queries_produce_different_hash(self):
        # Create two different DataFrames with explicit schema
        schema = StructType([
            StructField("id", IntegerType(), True),
            StructField("val", StringType(), True)
        ])
        df_small = self.spark.createDataFrame([(1, "a")], schema=schema)
        df_large = self.spark.createDataFrame([(1, "a"), (2, "b")], schema=schema)

        # Hashes should differ
        hash_small = get_table_hash(df_small)
        hash_large = get_table_hash(df_large)
        self.assertNotEqual(hash_small, hash_large, "Hashes for different DataFrames should differ")

if __name__ == "__main__":
    unittest.main()
