"""Configuration for dbfs_spark_cache."""
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """Settings for dbfs_spark_cache."""

    # We no longer need WAREHOUSE_PATH as Databricks will manage the storage location
    # based on the database configuration

    SPARK_CACHE_DIR: str = Field(
        default="/dbfs/FileStore/tables/cache/",
        description="Path to store cache metadata",
    )

    # Database name for cache tables
    CACHE_DATABASE: str = Field(
        default="cache_db",
        description="Database name for cached tables",
    )

    CACHE_DATABASE_TEST: str = Field(
        default="", # Will be set dynamically
        description="Database name for cached tables during testing. Defaults to CACHE_DATABASE + '_test'.",
    )

    DATABASE_PATH: str = Field(
        default="/dbfs/user/hive/warehouse/",
        description="Base path for the Hive warehouse where cache databases reside.",
    )

    DEFAULT_COMPLEXITY_THRESHOLD: Optional[float] = Field(
        default=130,
        description="Default complexity threshold for caching",
    )

    DEFAULT_MULTIPLIER_THRESHOLD: Optional[float] = Field(
        default=1.01,
        description="Default multiplier threshold for caching",
    )

    PREFER_SPARK_CACHE: bool = Field(
        default=True,
        description="If True and running on a classic (non-serverless) cluster, prioritize Spark's in-memory cache (.cache()) over immediate DBFS writes. DBFS cache will still be read if it exists, and a backup function can persist Spark-cached DFs to DBFS later. This setting is ignored (effectively False) on serverless clusters."
    )

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def __init__(self, **values): # type: ignore
        super().__init__(**values)
        if not self.CACHE_DATABASE_TEST:
            self.CACHE_DATABASE_TEST = f"{self.CACHE_DATABASE}_test"

config: Settings = Settings()
