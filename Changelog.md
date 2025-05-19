# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [v0.5.1] 2025-05-19

- Add `override_prefer_spark_cache` argument to cacheToDbfs() to allow disabling spark caching (use DBFS caching instead) on classic clusters

## [v0.5.0] 2025-05-07

- Add spark caching by default for classic clusters (default to spark cache instead of eager dbfs cache for `.cacheToDbfs()`)
- Adds support for `.cacheToDbfsIfTriggered()` and `caching.backup_spark_cached_to_dbfs()`, see README
- Support serverless clusters (although performance is bad)
- Better handing of DataFrames from `createCacheDataFrame`

## [v0.4.9] 2025-04-29

- Setup PyPI publishing

## [v0.4.6] 2025-04-29

- Initial release
