# Makefile for dbfs-spark-cache project validation

# Default shell
SHELL := /bin/bash

# Get project version from pyproject.toml
PROJECT_VERSION := $(shell uv tool run tomlq -jr .project.version < pyproject.toml)

# Include .env file if it exists, suppressing errors if it doesn't
-include .env
export $(shell sed 's/=.*//' .env)

.PHONY: help setup lint typecheck validate-version validate-changelog validate integration-test

help:
	@echo "Available targets:"
	@echo "  setup              - Install dependencies using uv sync"
	@echo "  lint               - Run ruff linter"
	@echo "  typecheck          - Run mypy static type checker"
	@echo "  test               - Run pre-commit hooks and pytest"
	@echo "  validate-version   - Validate the project version in pyproject.toml"
	@echo "  validate-changelog - Validate the Changelog.md for the current version"
	@echo "  validate           - Run all validation steps (version, changelog, lint, typecheck, test)"

setup:
	@echo "--- Installing dependencies (including dev) ---"
	uv sync # Sync only main dependencies first
	uv pip install -e .[dev] # Explicitly install project editable with dev extras

lint: setup
	@echo "--- Running linter (ruff) ---"
	# Use path specific to this submodule
	# Exclude notebook_utils.py which contains Databricks notebook-specific syntax
	uvx ruff check --fix dbfs_spark_cache tests

typecheck: setup
	@echo "--- Running static type checker (mypy) ---"
	# Use path specific to this submodule
	# Exclude notebook_utils.py which contains Databricks notebook-specific syntax
	uv run mypy --config-file pyproject.toml dbfs_spark_cache tests

test: setup
	@echo "--- Running pre-commit hooks ---"
	uv run pre-commit run -a || true
	@echo "--- Running unit tests (pytest) ---"
	uv run pytest tests

check-licenses: setup ## Generate the NOTICE file from dependencies
	@echo "--- Generating license data and processing... ---"
	# Run pip-licenses and pipe its CSV output directly to the python script
	uv run pip-licenses --with-license-file --format=csv | uv run python scripts/check_licenses.py

validate-version:
	@echo "--- Validating project version v$(PROJECT_VERSION) ---"
	@if ! [[ "$(PROJECT_VERSION)" =~ ^[0-9]+\.[0-9]+\.[0-9]+$$ ]]; then \
		echo "::error::Version $(PROJECT_VERSION) must match semantic format X.Y.Z"; \
		exit 1; \
	fi

validate-release-version:
	@if git rev-parse "v$(PROJECT_VERSION)" >/dev/null 2>&amp;1; then \
		echo "::error::Tag v$(PROJECT_VERSION) already exists. Have you updated the version in pyproject.toml?"; \
		exit 1; \
	fi
	@LATEST_TAG=$$(git tag -l 'v*' | sort -V | tail -n 1); \
	echo "Latest tag found: $${LATEST_TAG:-none}"; \
	if [ ! -z "$$LATEST_TAG" ]; then \
		CURRENT_VERSION="$${LATEST_TAG#v}"; \
		if ! [ "$$(printf '%s\n' "$$CURRENT_VERSION" "$(PROJECT_VERSION)" | sort -V | tail -n 1)" = "$(PROJECT_VERSION)" ]; then \
			echo "::error::Version $(PROJECT_VERSION) must be greater than $$CURRENT_VERSION."; \
			exit 1; \
		fi \
	fi
	@echo "Version v$(PROJECT_VERSION) is a valid new tag (latest is $${LATEST_TAG:-none})."


validate-changelog:
	@echo "--- Validating Changelog.md for v$(PROJECT_VERSION) ---"
	@CHANGELOG_FILE="Changelog.md"; \
	if [ ! -f "$$CHANGELOG_FILE" ]; then \
		echo "::error::$$CHANGELOG_FILE is missing. Did you forget to create it?"; \
		exit 1; \
	fi
	# Hardcode filename to avoid shell variable issues (fix from parent Makefile)
	@grep -q "^## \\[v$(PROJECT_VERSION)\\]" "Changelog.md"; \
	if [ $$? -ne 0 ]; then \
		echo "::error::Changelog.md missing a section for version v$(PROJECT_VERSION). Did you forget to update it?"; \
		exit 1; \
	fi
	@echo "Changelog validation successful for v$(PROJECT_VERSION)."

# Note: Test step is commented out because it requires a Databricks environment
validate: validate-version validate-changelog lint typecheck test
	@echo "--- All validation steps passed ---"

# Target to run the integration test notebook as a Databricks job
# Requires databricks CLI and jq to be installed
# Reads configuration from .env file
integration-test:
	@echo "--- Running Databricks integration test script ---"
	@./scripts/run_integration_test.sh
