# Databricks notebook source
import contextlib
import importlib.util
import os
import sys

from pathlib import Path
from typing import Any, Iterator


def import_from_file(
    file_path: str | os.PathLike,
    obj_name: str | None = None,
    mod_name: str | None = None,
) -> Any:
    """
    Import a module or an object from a file.

    Parameters
    ----------
    file_path : str or PathLike
        The path to the file to import from.
    obj_name : str, optional
        The name of the object to import from the module. If None, the entire module is
        returned.
    mod_name : str, optional
        The name to assign to the module. If None, the module name is derived from the
        file name.

    Returns
    -------
    Any
        The imported module or the specified object from the module.

    Examples
    --------
    Import a module:
    >>> my_module = import_from_file('path/to/my_module.py')

    Import an object from a module:
    >>> my_function = import_from_file('path/to/my_module.py', 'my_function')
    """
    file_path = Path(file_path).resolve()

    if mod_name is None:
        mod_name_path = file_path
        while mod_name_path.suffixes:
            mod_name_path = mod_name_path.with_suffix("")
    else:
        mod_name_path = Path(mod_name)

    mod_name_to_import = str(mod_name_path.stem.replace("-", "_"))

    # Check if file exists before attempting to import
    if not Path(file_path).exists():
        raise ImportError(f"File not found for importing: {file_path}")

    spec = importlib.util.spec_from_file_location(mod_name_to_import, str(file_path))

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {file_path}")

    imp_module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name_to_import] = imp_module
    spec.loader.exec_module(imp_module)

    if obj_name is None:
        return imp_module

    return getattr(imp_module, obj_name)

set_and_get_workdir = import_from_file('../notebook_utils.py', 'set_and_get_workdir')
setup_dependencies = import_from_file('../notebook_utils.py', 'setup_dependencies')

# COMMAND ----------

REPO_PATH = set_and_get_workdir(spark)  # noqa: F821

# COMMAND ----------

setup_dependencies(REPO_PATH, spark)  # noqa: F821

# COMMAND ----------

# Measure the extra latency incurred by createCachedDataFrame

import pandas as pd
import time
from dbfs_spark_cache import extend_dataframe_methods
extend_dataframe_methods(spark) # Pass spark

# Test createCachedDataFrame performace
# Create a large DataFrame (100,000 rows, 10 columns)
n_rows = 100_000
n_cols = 10
data = {
    f"col{i}": pd.Series(range(n_rows), dtype="int64")
    for i in range(n_cols)
}
df = pd.DataFrame(data)

from dbfs_spark_cache.hashing import _hash_input_data
from dbfs_spark_cache.caching import clear_cache_for_hash

data_hash = _hash_input_data(df)
clear_cache_for_hash(f"data_{data_hash}")

# Time hashing
t0 = time.time()
_ = spark.createCachedDataFrame(df)  # noqa: F821
t1 = time.time()
hash_time = t1 - t0

# Second
t0 = time.time()
_ = spark.createCachedDataFrame(df)  # noqa: F821
t1 = time.time()
second_hash_time = t1 - t0

# Time Spark createDataFrame
t2 = time.time()
_ = spark.createDataFrame(df)  # noqa: F821
t3 = time.time()
spark_time = t3 - t2

print(f"\n_hash_input_data time: {hash_time:.4f} s, ({second_hash_time:.4f}s second)")
print(f"Spark createDataFrame time: {spark_time:.4f} s")