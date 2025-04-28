# tests/utils.py

import os

def set_and_get_workdir(spark):
    """
    Sets and returns the repository path based on the Databricks notebook path.
    """
    # This function is intended to be run within a Databricks notebook environment.
    # It attempts to determine the repository path based on the notebook's path.
    # If the notebook path starts with "/Users/", it assumes a repository structure
    # where the notebook is located two directories deep within the repository.
    # Otherwise, it might not correctly determine the repository path.
    try:
        # Access dbutils, which is available in Databricks notebooks
        from pyspark.dbutils import DBUtils # type: ignore
        dbutils = DBUtils(spark) # type: ignore

        NB_PATH = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
        if NB_PATH.startswith("/Users/"):
            # Assuming notebook is in a structure like /Users/<user>/<repo>/tests/notebooks/...
            REPO_PATH = "/Workspace"+"/".join(NB_PATH.split("/")[:-3])
        else:
             # Fallback or alternative logic for other environments if needed
             # For now, raise an error or return a default if the path structure is unexpected
             raise ValueError(f"Unexpected notebook path structure: {NB_PATH}")

        print(f"REPO_PATH: {REPO_PATH}")
        return REPO_PATH
    except ImportError:
        # Handle cases where dbutils is not available (e.g., local testing)
        print("Warning: dbutils not found. Assuming local execution.")
        # You might want to add logic here to determine REPO_PATH in a local environment
        # For now, returning a placeholder or raising an error
        raise EnvironmentError("dbutils not available. This function is intended for Databricks notebooks.")
    except Exception as e:
        print(f"An error occurred while setting workdir: {e}")
        raise

def setup_dependencies(repo_path, spark):
    """
    Installs necessary dependencies using uv and pip within the repository context.
    This function is intended to be run within a Databricks notebook environment.
    """
    print(f"Setting up dependencies in REPO_PATH = {repo_path}")
    # This block assumes a specific setup for building and installing the wheel
    # It uses `uv` for building and `pip` for installing.
    # The `dbutils.library.restartPython()` call is specific to Databricks notebooks
    # and is used to ensure the newly installed library is available.
    try:
        # Access dbutils, which is available in Databricks notebooks
        from pyspark.dbutils import DBUtils # type: ignore
        dbutils = DBUtils(spark) # type: ignore

        # Use Databricks magic commands via dbutils.notebook.run or similar if needed,
        # but direct shell commands via `!` might work in some Databricks environments.
        # If `!` commands don't work, this section might need adjustment to use
        # Databricks-specific methods for running shell commands or installing libraries.

        # Example using `!` which might work in some Databricks notebook configurations:
        print("Installing uv...")
        os.system("pip install uv") # Using os.system for clarity, consider subprocess for more control

        print(f"Building wheel in {repo_path}...")
        # Using f-strings for command construction, ensure `repo_path` is safe if it comes from user input
        os.system(f"cd {repo_path} && pwd && ls -lat --full-time dist/*.whl | head -1 && uv build && ls -lat --full_time dist/*.whl | head -1")

        print(f"Installing latest wheel from {repo_path}/dist...")
        # Using os.system for clarity, consider subprocess for more control
        os.system(f"cd {repo_path} && pip install --force-reinstall --no-deps $(ls -t dist/*.whl | head -1)")

        print("Restarting Python kernel...")
        dbutils.library.restartPython()

    except ImportError:
        print("Warning: dbutils not found. Skipping dependency setup.")
        # Handle cases where dbutils is not available (e.g., local testing)
        # You might want to add logic here for local dependency setup if needed
    except Exception as e:
        print(f"An error occurred during dependency setup: {e}")
        raise
