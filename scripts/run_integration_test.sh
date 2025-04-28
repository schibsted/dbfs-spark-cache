#!/bin/bash
set -e # Ensure script exits on error

# Get the directory where the script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Go one level up to the dbfs-spark-cache directory
cd "$SCRIPT_DIR/.."

# Source environment variables from .env file if it exists
if [ -f ".env" ]; then
  echo "--- Sourcing configuration from .env file ---"
  # Use set -a to automatically export variables defined in the sourced file
  set -a
  source ".env"
  set +a
else
  echo "::error:: .env file not found in $(pwd)"
  exit 1
fi

# Check required variables
if [ -z "$DATABRICKS_PROFILE" ]; then
  echo "::error:: DATABRICKS_PROFILE is not set in .env file"
  exit 1
fi
if [ -z "$DATABRICKS_CLUSTER_ID" ]; then
  echo "::error:: DATABRICKS_CLUSTER_ID is not set in .env file"
  exit 1
fi
if [ -z "$DATABRICKS_NOTEBOOK_PATH" ]; then
  echo "::error:: DATABRICKS_NOTEBOOK_PATH is not set in .env file"
  exit 1
fi

# Check for required tools
if ! command -v databricks &> /dev/null; then
    echo "::error:: databricks CLI could not be found. Please install it."
    exit 1
fi
if ! command -v jq &> /dev/null; then
    echo "::error:: jq could not be found. Please install it."
    exit 1
fi

echo "--- Submitting Databricks integration test job ---"
echo "Using Profile: $DATABRICKS_PROFILE"
echo "Using Cluster ID: $DATABRICKS_CLUSTER_ID"
echo "Using Notebook Path: $DATABRICKS_NOTEBOOK_PATH"

echo "Defining JSON payload using printf..." # Debug echo
# Define the JSON payload for the Databricks job using printf for safety
DATABRICKS_JOB_JSON=$(printf '{
  "run_name": "Integration Test Run - dbfs-spark-cache (Script)",
  "tasks": [ {
    "task_key": "integration_test",
    "existing_cluster_id": "%s",
    "notebook_task": {
      "notebook_path": "%s"
    }
  } ]
}' "$DATABRICKS_CLUSTER_ID" "$DATABRICKS_NOTEBOOK_PATH")
echo "JSON payload defined." # Debug echo

# Create a temporary file for the JSON payload
TEMP_JSON_FILE=$(mktemp)
# Ensure the temp file is cleaned up on exit
trap 'rm -f "$TEMP_JSON_FILE"' EXIT

echo "$DATABRICKS_JOB_JSON" > "$TEMP_JSON_FILE"
echo "JSON payload written to temporary file: $TEMP_JSON_FILE"

# Submit the job using the temporary file and capture output.
# The 'databricks jobs submit' command waits for completion by default.
echo "Executing command: databricks jobs submit --profile \"$DATABRICKS_PROFILE\" --json @\"$TEMP_JSON_FILE\""
# Run the command and capture its output and exit code
JOB_OUTPUT=$(databricks jobs submit --profile "$DATABRICKS_PROFILE" --json @"$TEMP_JSON_FILE")
SUBMIT_EXIT_CODE=$?

echo "Job submission command finished with exit code: $SUBMIT_EXIT_CODE"
echo "Job submission final response:"
echo "$JOB_OUTPUT"

# Temp file is automatically removed by trap

# Check if the submit command itself failed (other than job failure)
if [ $SUBMIT_EXIT_CODE -ne 0 ]; then
  echo "::error:: 'databricks jobs submit' command failed with exit code $SUBMIT_EXIT_CODE."
  exit $SUBMIT_EXIT_CODE
fi

# Extract the final result state and run details from the command output
echo "Extracting final job status from response..."
RUN_ID=$(echo "$JOB_OUTPUT" | jq -r '.run_id')
RESULT_STATE=$(echo "$JOB_OUTPUT" | jq -r '.state.result_state')
STATE_MESSAGE=$(echo "$JOB_OUTPUT" | jq -r '.state.state_message')
RUN_PAGE_URL=$(echo "$JOB_OUTPUT" | jq -r '.run_page_url')

echo "Run ID: $RUN_ID"
echo "Final job state: $RESULT_STATE"
echo "Run page URL: $RUN_PAGE_URL"

# Check if the job succeeded based on the final result state
if [ "$RESULT_STATE" != "SUCCESS" ]; then
  echo "::error:: Databricks job $RUN_ID failed with state: $RESULT_STATE. Message: $STATE_MESSAGE"
  exit 1
fi

echo "Databricks job $RUN_ID completed successfully."
