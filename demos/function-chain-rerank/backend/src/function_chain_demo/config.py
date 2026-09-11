"""Environment-backed settings for the isolated Function Chain demo."""

from __future__ import annotations

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

NAMESPACE_PREFIX = "milvus3_demos_function_chain_rerank"


class DemoSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    milvus_uri: str = "http://127.0.0.1:49530"
    milvus_token: SecretStr = SecretStr("")
    milvus_expected_version: str = "3.0.0"
    milvus_timeout_seconds: float = 5.0

    minio_endpoint: str = "minio:9000"
    minio_access_key: SecretStr = SecretStr("")
    minio_secret_key: SecretStr = SecretStr("")
    minio_secure: bool = False
    minio_bucket: str = "a-bucket"
    collection_name: str = f"{NAMESPACE_PREFIX}_products"
    model_resource_name: str = f"{NAMESPACE_PREFIX}_model"
    model_object_name: str = "files/milvus3-demos/function-chain-rerank/xgb-reranker.ubj"
    cleanup_on_shutdown: bool = True

    @field_validator("collection_name", "model_resource_name")
    @classmethod
    def validate_milvus_namespace(cls, value: str) -> str:
        if not value.startswith(f"{NAMESPACE_PREFIX}_"):
            raise ValueError(f"Milvus names must start with {NAMESPACE_PREFIX}_")
        return value

    @field_validator("model_object_name")
    @classmethod
    def validate_object_namespace(cls, value: str) -> str:
        prefix = "files/milvus3-demos/function-chain-rerank/"
        if not value.startswith(prefix) or value.endswith("/"):
            raise ValueError(f"Model object must be an object below {prefix}")
        return value

    def minio_credentials(self) -> tuple[str, str]:
        access_key = self.minio_access_key.get_secret_value()
        secret_key = self.minio_secret_key.get_secret_value()
        if not access_key or not secret_key:
            raise RuntimeError("MINIO_ACCESS_KEY and MINIO_SECRET_KEY are required")
        return access_key, secret_key

    def token_value(self) -> str:
        """Return the Milvus token only for the client constructor."""

        return self.milvus_token.get_secret_value()
