# dbfs-spark-cache

A Python library for conveniently caching PySpark DataFrames to DBFS (Databricks File System). As opposed to [Spark caching or Databricks disk cache](https://docs.databricks.com/aws/en/optimizations/disk-cache), this will persist your dataframe to a permanent location (as a table) and is kept until deleted, even if your data breaks cluster is terminated. Testing has shown that this can dramatically speed up your query performance (see [performance analysis](CACHE_PERFORMANCE_PROFILE.md) for test results), particularly for exploratory data analysis (EDA) where you often need to rerun the same queries many times. The more permanent nature of this cache also makes it useful for quickly resuming work in a notebook, and it can also avoid the need for different users having to re-run the same queries needlessly.

To some extent this library will trade some smaller amount of extra query latency for an expected much reduce future latency for subsequent queries involving the same dataframe. This tradeoff can be tuned be a parameter which controlles when caching is trigged automatically, but it can also be triggered manually. Cached DataFrames also generally allowes much better query performance on downstream queries through increased spark executor load, possibly at the cost of some extra compute utilization.

## Features

- **DataFrame caching**: Intelligent caching system using DBFS (Databricks File System).
- **Query complexity estimation**: Tools to analyze and estimate Spark query complexity and trigger caching if above some set threshold.
- **Hybrid Spark/DBFS caching**: On classic clusters, you can now prefer Spark's in-memory cache (`.cache()`) for fast iterative work, and only persist to DBFS when needed (see below).

## Installation

Requires Python 3.10 or higher. Install using pip:
```sh
    pip install dbfs-spark-cache
```

## Usage

### Initialize Environment

```python
    import dbfs_spark_cache as dbfs_cache

    # Configure caching behavior by extending the global DataFrame class with some methods
    dbfs_cache.extend_dataframe_methods()

    # .... Read your DataFrame as the variable df for example
```

### Caching
```python
    # User either
    df_cached = df.cacheToDbfs() # triggers a write to cache, or
    df_cached = df.withCachedDisplay() # caches automatically if needed, displays the DataFrame and assigns to a new DataFrame, or
    df_cached = df.wcd() # same a above, or skip assignment if you just wante the display:
    df.wcd()
    # subsequent queries on df_cached will be much faster, and rexecutions of either caching line above will automatically read from cache if it exists,
    # unless the cache has been invalidated by changed data or a change in the query
    # to only trigger caching if needed but not display do:
    df_maybe_cached = df.cacheToDbfsIfTriggered()

    # You can also chain display calls which is a more compact way than using the normal display function, eg this displays two dataframes:
    df.wcd().groupBy("foo").count().wcd()

    # It even enables quite quick iterations with python display utilities, even on larger dataframes, eg:
    df.cacheToDbfs().toPandas().plot(...)

    # Note: replace calls to
    spark.createDataFrame(...)
    # with:
    spark.createCachedDataFrame(...)
    # or override it to avoid porting existing code:
    spark.createDataFrame = spark.createCachedDataFrame
```

Note: serverless clusters do not support monkey patching, ie extending with new methods, so `df.cacheToDbfs()` needs to be replaced with `cacheToDbfs(df)` and similar for `createCachedDataFrame(spark, ...)`. However `cacheToDbfs` (imported from `dbfs_spark_cache.caching`) can be used with `df.transform(cacheToDbfs)`. Unfortunatley write (and read) performance is really poor with serverless clusters, so it can't be recommended for general use with library.

Default cache trigger thresholds can be set per notebook by calling
```python
    dbfs_cache.extend_dataframe_methods(
        dbfs_cache_complexity_threshold=130,  # Default complexity threshold (Input GB * Multiplier)
        dbfs_cache_multiplier_threshold=1.01   # Default multiplier threshold (based on query plan)
    )
```

Or in .env file to cover the whole environment:
```env
DBFS_CACHE_COMPLEXITY_THRESHOLD=130
DBFS_CACHE_MULTIPLIER_THRESHOLD=1.01
```
Set either threshold to None to disable that specific check.
Caching occurs only if BOTH conditions are met (or the threshold is None).

## Hybrid Spark/DBFS Caching and Backup

Because spark cache is faster than dbfs cache when used with clusers with enough memory or disk space (and fast SSD disks are use as well), we can use it for fast iterative work, and only persist to dbfs when needed, ie when shutting down the cluster.

- **Backup of Spark-cached DataFrames**: Use `backup_spark_cached_to_dbfs()` to persist all Spark-cached DataFrames to DBFS before cluster termination, although the performance win of having it in sparch chache is not that big compared to rerunning all with dbfs caching directly.
- **Configurable caching mode**: The config `PREFER_SPARK_CACHE` (default: True) controls whether Spark in-memory cache is preferred on classic clusters. On serverless clusters, DBFS caching is always used.
- **Automatic registry of Spark-cached DataFrames**: DataFrames cached via `.cacheToDbfs()` in Spark-cache mode are tracked and can be listed or backed up.
- **Full test coverage**: All new logic is covered by unit and integration tests.

By default (on classic clusters), calling `.cacheToDbfs()` will:
- Use Spark's in-memory cache (`.cache()`) if no DBFS cache exists, and register the DataFrame for backup.
- If a DBFS cache exists, it will be read as before.
- You can persist all Spark-cached DataFrames to DBFS at any time (e.g. before cluster shutdown) with:

```python
from dbfs_spark_cache.caching import backup_spark_cached_to_dbfs
backup_spark_cached_to_dbfs() # backs up all DataFrames cached with .cacheToDbfs() over some threshold criterion (default: min_multiplier_threshold > 3), likley less practical, or:
backup_spark_cached_to_dbfs(specific_dfs=[my_last_end_of_work_df]) # backs up one or more specific DataFrames, eg the final result of your work, where you can pick up later
```

You can also clear the registry of tracked Spark-cached DataFrames with:

```python
from dbfs_spark_cache.caching import clear_spark_cached_registry
clear_spark_cached_registry()
```

To force always caching to DBFS set:

```python
from dbfs_spark_cache.config import config
config.PREFER_SPARK_CACHE = False
```

On serverless clusters, DBFS caching is always used regardless of this setting (spark cache is not available). If you want to disable all calls to the extensions you can do:
```python
    dbfs_cache.extend_dataframe_methods(disable_cache_and_display=True)
```
and it will keep the DataFrame unchanged.


### Dataframe cache invalidation techniques that triggers cache invalidation?

Dataframe storage type|Query plan changes|Data changes
---|---|---
DBFS/Could storage|Yes|Yes, any change casuing a new modification date (but also overwrites with identical data)
In-Memory|No not directly, but via conversion to BDFS table through createCachedDataFrame|Yes via direct hash of data

### Tested Environment

This library has been primarily tested under the following Databricks environment configuration, but anything supported by Databricks and PySpark DataFrame API should or may work too:

- **Databricks database**: Hive Metastore
- **Databricks Runtime Version**: 15.4 LTS, client.1.13 (serverless cluster)
- **Storage Layer**: DBFS and S3
- **File Formats**: Parquet, JSON

Note that serverless performance when writing to DBFS is currently abysmal and can only be used for limited testing on small datasets. You can use file `serverless_env.yml` to automatically install the library on a serverless cluster.

#### What is "Total compute complexity" anyway?

It is a metric that roughly tries to estimate the time cost of the query ahead of time. It's calculated as: `Total Input Size (GB) * Query Plan Multiplier`. The multiplier is derived from analyzing the query plan for expensive operations like joins, window functions, complex aggregations, etc. A simple read or count operation has a multiplier of 1.0. The complexity threshold (`dbfs_cache_complexity_threshold`) checks this final value, while the `dbfs_cache_multiplier_threshold` checks the multiplier component directly. Both conditions must be met (or the respective threshold set to `None`) for automatic caching via `withCachedDisplay`/`wcd` to trigger. The unit is not given in seconds but just something roughly proportional to the time spent on the query for a given cluster.


### Configuration

Configuration is handled through environment variables (can be read from .env file):

- `SPARK_CACHE_DIR`: Directory for cached data (default: "/dbfs/FileStore/tables/cache/").
- `CACHE_DATABASE`: Database name for cached tables (default: "cache_db").
- `DATABASE_PATH`: Where cached tables are stored, only used for table deletion in corner cases (default: "/dbfs/user/hive/warehouse/").

### Logging

This library uses the standard Python `logging` module. To get some useful messages on when and how caching is performed you can set the logging level to `INFO` or `DEBUG`. E.g.:

```python
import dbfs_spark_cache
import logging

# Get the library's logger
library_logger = logging.getLogger('dbfs_spark_cache')
library_logger.setLevel(logging.INFO) # must be after imports
```


### Automatic cache cleanup
Create a Databircks job that runs `scripts/clear_old_caches.py` on some schedule and set the desired retentions period in number of days to `caching.clear_caches_older_than(num_days=...)`


### Limitations and quirks

* Since pyspark does not support writing DataFrames with columns that has a space in their name, you need to set names explicitly in some situations, eg:

```python
    df.groupBy("foo").agg(sum("bar")).cacheToDbfs()
```

will fail because of disallowed parathesis in column name `sum("bar")`, but this works

```python
    df.groupBy("foo").agg(sum("bar").alias("sum_bar")).cacheToDbfs()
```

* Cache invalidation can detect new data written to disk (if partitions get a newer modification date), but a DataFrame created with spark.createDataFrame() -- i.e. from memory will not work, and caching for these are disabled (cacheToDbfs returns the uncached DataFrame).

* In some tricky cases it is best to manually invalidate the cache with:

```python
    df_in_mem = spark.createDataFrame(df)

    df_in_mem_cached = df_in_mem.cacheToDbfs()
    df_in_mem_cached.clearCachedData()

    df_in_mem_cached = df_in_mem.cacheToDbfs()
```

The same if the case for (non-schema) chages in UDFs.


* Non-deterministic queries like `df.sample()` will not trigger cache invalidation, so clear cache as above if needed.

* Using `df.wcd()` with Databricks visualizations other than a table view may throw errors. In this case you can use `display(df.cacheToDbfs())` instead.

* For some dataframes with a query plan containing `Unsupported node: Scan ExistingRDD` (also printed with a warning log), repeated calls to `df.cacheToDbfs()` will not trigger a new write, even though an existing cache exists.

### Development

Run tests locally with:
```sh
    make validate
```

Run integration test remotley on Databricks with this and make sure it is successful:
```sh
    make integration-test
```
if you have the databricks CLI installed and a cluster variables configured, eg:
```sh
    DATABRICKS_PROFILE=DEFAULT
    DATABRICKS_CLUSTER_ID=xxxx-yyyyyy-zzzzzzzz
    DATABRICKS_NOTEBOOK_PATH=/Workspace/Repos/myname/dbfs-spark-cache/tests/notebooks/integration_test_notebook
```

Before merging a PR, the version in pyproject.toml needs to be updated and Changelog.md too with a matching entry.

### Release process

Make PR to main branch, get it approved and merged. It must contain a bump in the version in pyproject.toml and a matching entry in Changelog.md if a new release is to be made. Then run this on the main branch:
```sh
    make release
```

This will create a git tag, push it to origin and create a GitHub release (if you have set up the GitHub CLI and authenticated it) and publish the package to PyPI (using the GitHub release action workflow).
