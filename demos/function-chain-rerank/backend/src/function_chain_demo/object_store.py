"""Minimal object-store boundary used by the model FileResource."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from minio import Minio
from minio.error import S3Error

from function_chain_demo.config import DemoSettings


class ModelObjectStore(Protocol):
    def upload(self, source: Path) -> None: ...

    def remove(self) -> None: ...

    def exists(self) -> bool: ...


class MinioModelObjectStore:
    def __init__(self, settings: DemoSettings) -> None:
        self._settings = settings

    def _client(self) -> Minio:
        access_key, secret_key = self._settings.minio_credentials()
        return Minio(
            self._settings.minio_endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=self._settings.minio_secure,
        )

    def upload(self, source: Path) -> None:
        client = self._client()
        if not client.bucket_exists(self._settings.minio_bucket):
            raise RuntimeError("The Milvus object-storage bucket does not exist")
        client.fput_object(
            self._settings.minio_bucket,
            self._settings.model_object_name,
            str(source),
            content_type="application/octet-stream",
        )

    def remove(self) -> None:
        self._client().remove_object(
            self._settings.minio_bucket,
            self._settings.model_object_name,
        )

    def exists(self) -> bool:
        try:
            self._client().stat_object(
                self._settings.minio_bucket,
                self._settings.model_object_name,
            )
        except S3Error as exc:
            if exc.code in {"NoSuchKey", "NoSuchObject"}:
                return False
            raise
        return True
