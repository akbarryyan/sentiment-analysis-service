from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    model_version: str = Field(default="nb-v1", alias="MODEL_VERSION")
    model_path: str = Field(default="app/ml/pipeline.joblib", alias="MODEL_PATH")
    training_dataset_export_url: str | None = Field(
        default=None,
        alias="TRAINING_DATASET_EXPORT_URL",
    )
    training_dataset_export_token: str | None = Field(
        default=None,
        alias="TRAINING_DATASET_EXPORT_TOKEN",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
