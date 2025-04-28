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

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


config: Settings = Settings()
