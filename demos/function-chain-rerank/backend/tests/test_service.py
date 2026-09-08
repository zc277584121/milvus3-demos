from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from function_chain_demo.catalog import PRODUCTS, QUERIES
from function_chain_demo.config import DemoSettings
from function_chain_demo.service import DemoProvisioner, SearchService


def make_hit(product_id: int, score: float) -> dict[str, Any]:
    product = PRODUCTS[product_id - 1]
    entity = product.as_milvus_row()
    entity.pop("embedding")
    return {
        "id": product_id,
        "distance": score,
        "entity": entity,
    }


class SearchClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.calls.append(kwargs)
        if "function_chains" in kwargs:
            return [[make_hit(3, 0.8), make_hit(2, 0.7), make_hit(4, 0.6), make_hit(1, 0.5)]]
        return [[make_hit(1, 1.0), make_hit(2, 0.9), make_hit(3, 0.8), make_hit(4, 0.7)]]

    def close(self) -> None:
        self.closed = True


def test_compare_uses_l0_xgboost_and_preserves_both_milvus_orders() -> None:
    client = SearchClient()
    service = SearchService(DemoSettings(), client_factory=lambda: client)

    query = QUERIES[0]
    comparison = service.compare(query.query_text, query_id=query.id)

    assert comparison.query_text == query.query_text
    assert [product.id for product in comparison.vector_order] == [1, 2, 3, 4]
    assert [product.id for product in comparison.business_order] == [3, 2, 4, 1]
    assert [product.rank for product in comparison.business_order] == [1, 2, 3, 4]
    assert "function_chains" not in client.calls[0]
    assert client.calls[0]["limit"] == 20
    assert client.calls[1]["limit"] == 20
    chain = client.calls[1]["function_chains"]
    assert chain.stage == 3
    assert chain.name == "xgb_business_rerank"
    assert len(chain.ops) == 5
    assert chain.ops[0].outputs == ("popularity",)
    assert chain.ops[0].expr.name == "num_combine"
    assert chain.ops[1].outputs == ("price_affinity",)
    assert chain.ops[1].expr.name == "decay"
    assert chain.ops[2].outputs == ("freshness",)
    assert chain.ops[2].expr.name == "decay"
    assert chain.ops[3].outputs == ("normalized_semantic_score",)
    assert chain.ops[3].expr.name == "round_decimal"
    assert chain.ops[3].expr.params == {"decimal": 5}
    assert chain.ops[4].expr.name == "xgboost"
    assert [argument.name for argument in chain.ops[4].expr.args] == [
        "normalized_semantic_score",
        "rating",
        "popularity",
        "price_affinity",
        "freshness",
    ]
    assert (
        chain.ops[4]
        .expr.params["model_resource"]
        .startswith("milvus3_demos_function_chain_rerank_")
    )
    assert client.closed is True


@pytest.mark.parametrize("query", QUERIES, ids=lambda query: query.id)
def test_every_fixed_natural_query_uses_the_shared_1024_dimension_contract(query) -> None:
    client = SearchClient()
    comparison = SearchService(DemoSettings(), client_factory=lambda: client).compare(
        query.query_text,
        query_id=query.id,
    )

    assert comparison.query_id == query.id
    assert comparison.query_text == query.query_text
    assert client.calls[0]["data"] == [list(query.embedding)]
    assert len(client.calls[0]["data"][0]) == 1024
    assert "function_chains" not in client.calls[0]
    assert client.calls[1]["function_chains"].stage == 3


class SchemaRecorder:
    def __init__(self) -> None:
        self.fields: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def add_field(self, *args: Any, **kwargs: Any) -> None:
        self.fields.append((args, kwargs))


class IndexRecorder:
    def __init__(self) -> None:
        self.indexes: list[dict[str, Any]] = []

    def add_index(self, **kwargs: Any) -> None:
        self.indexes.append(kwargs)


@dataclass(frozen=True)
class FileResourceInfoShape:
    name: str
    path: str


class LifecycleClient:
    def __init__(self, settings: DemoSettings, *, return_resource_info: bool = False) -> None:
        self.collection_exists = True
        self.resources = [settings.model_resource_name]
        self.model_object_name = settings.model_object_name
        self.return_resource_info = return_resource_info
        self.removed_resources: list[str] = []
        self.schema = SchemaRecorder()
        self.index = IndexRecorder()
        self.inserted_rows: list[dict[str, object]] = []
        self.closed = False

    def get_server_version(self) -> str:
        return "3.0.0"

    def list_file_resources(self) -> list[object]:
        if self.return_resource_info:
            return [
                FileResourceInfoShape(
                    name=name,
                    path=self.model_object_name,
                )
                for name in self.resources
            ]
        return list(self.resources)

    def remove_file_resource(self, name: str) -> None:
        self.removed_resources.append(name)
        self.resources.remove(name)

    def add_file_resource(self, name: str, path: str) -> None:
        assert path == self.model_object_name
        self.resources.append(name)

    def has_collection(self, collection_name: str) -> bool:
        return self.collection_exists

    def drop_collection(self, collection_name: str) -> None:
        self.collection_exists = False

    def create_schema(self, **kwargs: Any) -> SchemaRecorder:
        return self.schema

    def prepare_index_params(self) -> IndexRecorder:
        return self.index

    def create_collection(self, **kwargs: Any) -> None:
        self.collection_exists = True

    def insert(self, **kwargs: Any) -> None:
        self.inserted_rows = kwargs["data"]

    def flush(self, **kwargs: Any) -> None:
        return None

    def load_collection(self, **kwargs: Any) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class ObjectStoreRecorder:
    def __init__(self, *, object_name: str) -> None:
        self.object_name = object_name
        self.uploaded_bytes = 0
        self.object_exists = False

    def upload(self, source: Path) -> None:
        self.uploaded_bytes = len(source.read_bytes())
        self.object_exists = True

    def remove(self) -> None:
        self.object_exists = False

    def exists(self) -> bool:
        return self.object_exists


@pytest.mark.parametrize(
    "return_resource_info",
    [False, True],
    ids=["string-resource", "FileResourceInfo-shape"],
)
def test_prepare_and_cleanup_are_limited_to_the_demo_namespace(
    return_resource_info: bool,
) -> None:
    settings = DemoSettings()
    client = LifecycleClient(settings, return_resource_info=return_resource_info)
    store = ObjectStoreRecorder(object_name=settings.model_object_name)
    provisioner = DemoProvisioner(
        settings,
        client_factory=lambda: client,
        object_store=store,
    )

    assert provisioner.inspect_state().file_resource_exists is True

    manifest = provisioner.prepare()

    assert manifest.milvus_version == "3.0.0"
    assert manifest.product_count == 240
    assert manifest.dataset["revision"] == "synthetic-commerce-catalog-r1"
    assert len(manifest.model["sha256"]) == 64
    assert store.uploaded_bytes > 0
    assert len(client.inserted_rows) == 240
    assert client.inserted_rows[0]["image_mime"] == "image/jpeg"
    assert client.inserted_rows[0]["item_id"] == "SYN-DESK-001"
    assert client.inserted_rows[0]["embedding"] == list(PRODUCTS[0].embedding)
    assert store.object_name == settings.model_object_name
    assert client.resources == [settings.model_resource_name]
    assert client.removed_resources == [settings.model_resource_name]

    provisioner.cleanup()

    assert provisioner.inspect_state().collection_exists is False
    assert provisioner.inspect_state().file_resource_exists is False
    assert provisioner.inspect_state().model_object_exists is False
    assert client.removed_resources == [
        settings.model_resource_name,
        settings.model_resource_name,
    ]
