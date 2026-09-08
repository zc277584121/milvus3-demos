import pytest
from pydantic import ValidationError

from function_chain_demo.config import DemoSettings


def test_resource_names_cannot_escape_the_demo_namespace() -> None:
    with pytest.raises(ValidationError):
        DemoSettings(collection_name="unrelated_collection")
    with pytest.raises(ValidationError):
        DemoSettings(model_resource_name="unrelated_resource")
    with pytest.raises(ValidationError):
        DemoSettings(model_object_name="files/unrelated/model.ubj")


def test_credentials_are_redacted_in_settings_representation() -> None:
    settings = DemoSettings(
        minio_access_key="local-access",
        minio_secret_key="local-secret",
        milvus_token="local-token",
    )

    settings_repr = repr(settings)
    assert "local-access" not in settings_repr
    assert "local-secret" not in settings_repr
    assert "local-token" not in settings_repr
