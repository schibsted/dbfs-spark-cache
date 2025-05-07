"""Query complexity estimation utilities."""
import logging
from typing import List
from dbfs_spark_cache.utils import is_serverless_cluster

from pyspark.sql import DataFrame

# Import dbutils and get_quedy_plan grom_cach fr moduleom caching module
try:
    from databricks.sdk.runtime import dbutils
except ImportError:
    dbutils = None # type: ignore[assignment,misc]

log = logging.getLogger(__name__)


def get_input_file_sizes(df: DataFrame) -> List[float]:
    """Get sizes of input files in GB, using dbutils.fs.ls on serverless and Hadoop API otherwise."""
    input_files = df.inputFiles()
    if not input_files: # Optimization: if no input files, return early
        log.debug("No input files found for DataFrame.")
        return []
    file_sizes: List[float] = []

    if is_serverless_cluster():
        log.debug("Running on a serverless cluster, using dbutils.fs.ls for file sizes.")
        if dbutils is None:
            log.warning("dbutils not available on a serverless cluster, cannot get input file sizes.")
            return [] # Return empty list, as per original behavior if dbutils is missing

        for file_path in input_files:
            try:
                # Use dbutils.fs.ls to get file status which includes size
                # dbutils.fs.ls returns a list of FileInfo objects
                file_info_list = dbutils.fs.ls(file_path)
                if file_info_list:
                    # Assuming inputFiles returns actual file paths, not directories
                    # If it returns directories, we might need to list recursively
                    file_size = float(file_info_list[0].size)
                    # Convert bytes to GB
                    file_sizes.append(file_size / (1024 * 1024 * 1024))
                else:
                    log.warning(f"dbutils.fs.ls returned empty list for {file_path}. Skipping size.")
            except Exception as e:
                # Skip files that can't be accessed or listed
                log.warning(f"Could not get file size for {file_path} using dbutils.fs.ls: {e}. Skipping.")
                continue
    else:
        log.debug("Running on a non-serverless cluster, using Hadoop API for file sizes.")
        jsc = None
        try:
            jsc = df.sparkSession._jsc
            if jsc is None:
                log.error("Java SparkContext (jsc) is None on a non-serverless cluster. Cannot get input file sizes via Hadoop API.")
                return [] # Cannot proceed without jsc
        except AttributeError: # pragma: no cover
            # This case might be hard to test if SparkSession always has _jsc or raises different error
            log.error("Could not access Java SparkContext (df.sparkSession._jsc). Cannot get input file sizes via Hadoop API.")
            return [] # Cannot proceed without jsc

        # Ensure _jvm is accessible
        if not hasattr(df.sparkSession, '_jvm') or df.sparkSession._jvm is None: # pragma: no cover
            log.error("SparkSession._jvm is not accessible. Cannot use Hadoop API for file sizes.")
            return []

        for file_path in input_files:
            try:
                # Access JVM Path class and create Path object
                hadoop_path_class = df.sparkSession._jvm.org.apache.hadoop.fs.Path
                path_obj = hadoop_path_class(file_path)

                # Get FileSystem and FileStatus
                fs = path_obj.getFileSystem(jsc.hadoopConfiguration())
                file_status = fs.getFileStatus(path_obj)
                file_size_bytes = float(file_status.getLen())

                # Convert bytes to GB
                file_sizes.append(file_size_bytes / (1024 * 1024 * 1024))
            except Exception as e: # pragma: no cover
                # Broad exception catch as various issues can occur with FS interactions
                log.warning(f"Could not get file size for {file_path} using Hadoop API: {e}. Skipping.")
                continue
    return file_sizes


def _calculate_complexity_from_plan(query_plan: str, total_size: float) -> tuple[float, float]:  # Modified return type
    """Calculate complexity based on a query plan string and total input size.

    Internal helper function. Assumes query_plan is already lowercased.

    Parameters
    ----------
    query_plan : str
        The lowercased analyzed query plan string.
    total_size : float
        Total size of input files in GB.

    Returns
    -------
    tuple[float, float]
        A tuple containing:
        - The estimated complexity value (size * multiplier).
        - The calculated multiplier.
    """
    # --- Complexity Multiplier Calculation ---
    # Constants found by asking Gemini 2.5 Pro.
    # Baseline multiplier represents simplest operations like projections.
    # Operations increase the multiplier based on count using:
    # total_op_factor = base_factor * (incremental_factor ** (count - 1))
    multiplier_base = 0.6 # minimum AI estimated complexity of a query (a scan or similar) relative to a count.
    multiplier = multiplier_base # Start with the base

    # --- Define Factors ---
    # Base factor: Multiplier for the first occurrence (relative to multiplier_base)
    # Incremental factor: Multiplier for each subsequent occurrence
    factors = {
        "join":     {"base": 2.0 / multiplier_base, "inc": 1.5},
        "window":   {"base": 2.5 / multiplier_base, "inc": 1.8},
        "agg_simple": {"base": 1.0 / multiplier_base, "inc": 1.1},
        "agg_complex": {"base": 1.6 / multiplier_base, "inc": 1.4},
        "sort":     {"base": 1.4 / multiplier_base, "inc": 1.3},
        "distinct": {"base": 1.5 / multiplier_base, "inc": 1.4},
        "udf":      {"base": 1.2 / multiplier_base, "inc": 1.2},
    }

    # --- Count Operations ---
    op_counts = {
        "join": query_plan.count("join"),
        "window": query_plan.count("window"),
        "aggregate": query_plan.count("aggregate"), # Used for simple/complex logic below
        "sort": query_plan.count("sort") + query_plan.count("order by"),
        # Distinct check needs care - keyword or within aggregate
        "distinct": query_plan.count("distinct") + query_plan.count("distinct "), # Space helps avoid partial matches
        "udf": query_plan.count("udf") # Python UDFs might appear differently, e.g., 'pythoneval'
    }
    # Refine distinct count if it's part of an aggregate (avoid double counting multiplier later)
    if "aggregate" in query_plan and "distinct" in query_plan:
         # Heuristic: assume distinct inside aggregate doesn't add as much cost as a separate distinct op
         # Reduce count slightly, or handle differently if needed. For now, just count keyword.
         pass # Keep simple count for now

    # --- Determine Aggregate Type ---
    # Reuse original logic to differentiate simple vs complex based on structure
    is_complex_aggregate_structure = "groupingexpressions" in query_plan or query_plan.count("agg") > 1
    has_simple_aggregate = op_counts["aggregate"] > 0 and not is_complex_aggregate_structure
    has_complex_aggregate = op_counts["aggregate"] > 0 and is_complex_aggregate_structure

    # --- Apply Multipliers Based on Counts ---
    for op, count in op_counts.items():
        if count <= 0:
            continue

        current_count = count

        if op == "aggregate":
            # Skip generic 'aggregate' count, handle specific types below
            continue
        elif op == "distinct":
             # Apply distinct factor
             factor_details = factors["distinct"]
        elif op == "sort":
             factor_details = factors["sort"]
        elif op == "join":
             factor_details = factors["join"]
        elif op == "window":
             factor_details = factors["window"]
        elif op == "udf":
             factor_details = factors["udf"]
        else:
             continue # Should not happen with current op_counts keys

        base_factor = factor_details["base"]
        inc_factor = factor_details["inc"]
        total_op_factor = base_factor * (inc_factor ** (current_count - 1))
        multiplier *= total_op_factor

    # Handle aggregates separately using their specific logic and the aggregate count
    aggregate_count = op_counts["aggregate"]
    if aggregate_count > 0:
        if has_simple_aggregate:
            factor_details = factors["agg_simple"]
            base_factor = factor_details["base"]
            inc_factor = factor_details["inc"]
            total_op_factor = base_factor * (inc_factor ** (aggregate_count - 1))
            multiplier *= total_op_factor
        elif has_complex_aggregate: # Use elif to ensure only one aggregate type applies
            factor_details = factors["agg_complex"]
            base_factor = factor_details["base"]
            inc_factor = factor_details["inc"]
            total_op_factor = base_factor * (inc_factor ** (aggregate_count - 1))
            multiplier *= total_op_factor


    # Final complexity calculation
    complexity = total_size * multiplier
    # Optional: Ensure complexity doesn't fall below a reasonable minimum if needed
    # complexity = max(total_size * multiplier_base, complexity) if total_size > 0 else 0.0

    return complexity, multiplier


def estimate_compute_complexity(df: DataFrame) -> tuple[float, float, float]:
    """Calculate total compute complexity and multiplier based on input data size and query plan.

    The complexity is estimated as the sum of input file sizes in GB
    multiplied by a factor derived from the operations in the logical query plan.
    A simple count() operation on 1 GB of data should result in a complexity of 1.0
    and a multiplier of 1.0. Simpler operations (like projections) start at a baseline
    multiplier of 0.6.

    This function retrieves the input size and query plan, then delegates
    the calculation to _calculate_complexity_from_plan.

    Parameters
    ----------
    df : DataFrame
        The Spark DataFrame to analyze

    Returns
    -------
    tuple[float, float, float]
        A tuple containing:
        - The estimated complexity value (size * multiplier).
        - The calculated multiplier.
        - The total input size in GB.
    """
    file_sizes = get_input_file_sizes(df)
    total_size = sum(file_sizes)

    # If no input files or zero size, log a warning and return base multiplier
    if not total_size:
        log.warning("Could not determine input file sizes (df.inputFiles() might be empty due to transformations or RDD lineage). Complexity estimation will be based on plan only.")
        # Return 0 complexity, base multiplier, and 0 size.
        # The multiplier calculation below will still run based on the plan.
        # We return 0 size explicitly.
        # Let's calculate the multiplier based on the plan even without size.
        from dbfs_spark_cache.caching import get_query_plan
        query_plan_str = get_query_plan(df).lower()
        if query_plan_str.startswith("error:"):
             log.warning(f"Could not get query plan for complexity estimation: {query_plan_str}. Returning base complexity.")
             return 0.0, 1.0, 0.0 # Base multiplier, 0 size, 0 complexity

        # Calculate multiplier based on plan, but complexity is 0 due to unknown size
        _, multiplier = _calculate_complexity_from_plan(query_plan_str, 0.0) # Pass 0 size
        return 0.0, multiplier, 0.0 # Return 0 complexity, calculated multiplier, 0 size

    from dbfs_spark_cache.caching import get_query_plan
    query_plan_str = get_query_plan(df).lower()

    # If get_query_plan returned an error string, we can't estimate complexity
    if query_plan_str.startswith("error:"):
        log.warning(f"Could not get query plan for complexity estimation: {query_plan_str}. Returning base complexity.")
        return total_size, 1.0, total_size

    # Delegate the core calculation and return the tuple with total_size
    complexity, multiplier = _calculate_complexity_from_plan(query_plan_str, total_size)
    return complexity, multiplier, total_size
