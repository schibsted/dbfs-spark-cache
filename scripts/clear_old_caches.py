# Databricks notebook source
from dbfs_spark_cache import caching

# COMMAND ----------

caching.get_cached_tables(num_threads=100)

# COMMAND ----------
# Only this is needed for cleanup, rest is for verification:
caching.clear_caches_older_than(num_days=7, num_threads=50)

# COMMAND ----------

caching.get_cached_dataframe_metadata(num_threads=50)

# COMMAND ----------
# Delete any tables or metadata with missing metadata or table.
caching.clear_inconsistent_cache(num_threads=50)

# COMMAND ----------

caching.get_cached_tables(num_threads=100)
