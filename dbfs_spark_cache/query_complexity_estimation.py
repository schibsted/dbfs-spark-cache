"""Query complexity estimation utilities."""
from typing import List

from pyspark.sql import DataFrame


def get_input_file_sizes(df: DataFrame) -> List[float]:
    """Get sizes of input files in GB."""
    input_files = df.inputFiles()
    file_sizes = []
    for file_path in input_files:
        try:
            # Ensure SparkContext is available
            jsc = df.sparkSession._jsc
            if jsc is None:
                raise Exception("Java SparkContext not available")

            path = df.sparkSession._jvm.org.apache.hadoop.fs.Path(file_path)  # type: ignore[union-attr]
            fs = path.getFileSystem(jsc.hadoopConfiguration())
            file_size = float(fs.getFileStatus(path).getLen())
            # Convert bytes to GB
            file_sizes.append(file_size / (1024 * 1024 * 1024))
        except Exception:
            # Skip files that can't be accessed
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


def estimate_compute_complexity(df: DataFrame) -> tuple[float, float, float]:  # Modified return type
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

    # If no input files or zero size, complexity is 0
    if not total_size:
        return 0.0, 1.0, 0.0  # Return default multiplier and zero size

    # Get the analyzed logical query plan
    try:
        # Use analyzed plan for estimation before optimization
        query_plan_str = df._jdf.queryExecution().analyzed().toString().lower()
    except Exception:
        # If plan cannot be accessed (e.g., during DataFrame creation before action)
        # return base complexity (size * 1.0). Consider logging this.
        # logger.warning("Could not access query plan for complexity estimation.")
        return total_size, 1.0, total_size  # Equivalent to total_size * 1.0

    # Delegate the core calculation and return the tuple with total_size
    complexity, multiplier = _calculate_complexity_from_plan(query_plan_str, total_size)
    return complexity, multiplier, total_size
