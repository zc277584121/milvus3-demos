"""Environment-backed settings shared by lightweight API services."""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class MilvusSettings(BaseSettings):
    """Connection settings that never expose credentials in representations."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    milvus_uri: str = "http://127.0.0.1:49530"
    milvus_token: SecretStr = SecretStr("")
    milvus_expected_version: str = "3.0.0"
    milvus_timeout_seconds: float = 5.0

    def token_value(self) -> str:
        """Return the token only for the client constructor."""

        return self.milvus_token.get_secret_value()
